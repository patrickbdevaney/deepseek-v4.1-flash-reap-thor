// gate_tc_fp8_smem.cu — the smem-staged FP8 tensor-core GEMM (Finding 41).
//
// `tests/gate_units.cu` checks tc_fp8_gemm at exactly ONE (M,N,K) drawn from a stored testcase. The
// smem-staged kernel changes the B path, the block shape, the grid shape and the N-tail handling all
// at once, and every one of those fails at a different (M,N,K). So sweep the shapes decode actually
// issues, at every M the verify path actually uses, plus the tails:
//
//   N % 64 != 0   -> the staging loop must zero-fill rows past N and the epilogue must not store them
//   M % 16 != 0   -> the A fragment must zero-fill rows past M (m16 tile is always 16 rows)
//   M > 16        -> more than one m-tile, i.e. grid.y > 1
//
// Two comparisons, because they fail differently:
//   A. smem-staged vs the m16 register version — isolates the staging; the mma math is identical, so
//      this should agree to summation order only.
//   B. both vs fp8_block_gemm's warp-per-output oracle — catches a wrong scale index or tile mapping.
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <stdint.h>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#define CU(x) do{auto e=(x); if(e){printf("ERR %s:%d %d\n",__FILE__,__LINE__,(int)e);return 1;}}while(0)

void fp8_block_gemm(float*, const uint8_t*, const float*, const uint8_t*, const float*, int,int,int, cudaStream_t);
void tc_fp8_gemm(float*, const uint8_t*, const float*, const uint8_t*, const float*, int,int,int, cudaStream_t);
extern bool g_tc_fp8;
void tc_fp8_set_smem(int on);

struct Cmp { double cos, rms_rel, rel_max; };
static Cmp compare(const std::vector<float>& a, const std::vector<float>& b){
    double d=0,na=0,nb=0,se=0,md=0,amax=0;
    for(size_t i=0;i<a.size();++i){ d+=(double)a[i]*b[i]; na+=(double)a[i]*a[i]; nb+=(double)b[i]*b[i];
        double e=(double)a[i]-b[i]; se+=e*e; md=fmax(md,fabs(e)); amax=fmax(amax,fabs((double)a[i])); }
    Cmp c; c.cos=d/(sqrt(na)*sqrt(nb)+1e-30);
    c.rms_rel=sqrt(se/a.size())/(sqrt(na/a.size())+1e-30); c.rel_max=md/(amax+1e-30);
    return c;
}
// The deep-composition metric this project standardised on. Note the oracle accumulates in f32 in a
// different order than the mma does, over K up to 4096 terms, so elementwise abs tolerance is wrong.
static bool ok(Cmp c){ return c.cos>0.9999 && c.rms_rel<1e-2 && c.rel_max<5e-3; }

int main(){
    struct Shape { int N, K; const char* name; };
    const Shape shapes[] = {
        {1024, 4096, "wq_a   [1024,4096]"},
        {4096, 1024, "wq_b   [4096,1024]"},
        { 512, 4096, "wkv    [512,4096] "},
        {4096, 4096, "wo_b   [4096,4096]"},
        {2048, 4096, "shared [2048,4096]"},
        { 520, 1024, "N-tail [520,1024] "},     // 520 = 8*65: last block is 8 of 64 rows
    };
    const int Ms[] = {1,2,3,5,8,13,16,17,20};
    // MISALIGNED B. Every weight in this engine is a pointer INTO a mapped safetensors file and is
    // only 4-byte aligned; cudaMalloc always returns 256-byte-aligned memory, so a gate that
    // allocates its own B can never reproduce what decode actually passes. An earlier version of
    // this file did exactly that, passed at every shape and M, and the kernel then crashed prefill
    // on the first real weight. Run every case twice: once at offset 0, once at offset 4.
    const int OFFS[] = {0, 4};
    srand(7);
    bool pass = true;
    printf("sweeping M={1,2,3,5,8,13,16,17,20} x B offset {0,4}; GATE_VERBOSE=1 for every row\n");
    for (const Shape& sh : shapes){
        const int N=sh.N, K=sh.K, KB=K/128;
        std::vector<uint8_t> B((size_t)N*K);
        std::vector<float>   bsv((size_t)((N+127)/128)*KB);
        // 0x7f/0xff are e4m3 NaN — one of those poisons every metric and reports nothing useful.
        for(auto&b:B){ uint8_t v; do { v=(uint8_t)(rand()&0xff); } while(v==0x7f||v==0xff); b=v; }
        for(auto&x:bsv) x = (rand()%2000+100)/1000.f;
        uint8_t* dBbase; float* dbs;
        CU(cudaMalloc(&dBbase,B.size()+16)); CU(cudaMalloc(&dbs,bsv.size()*4));
        CU(cudaMemcpy(dbs,bsv.data(),bsv.size()*4,cudaMemcpyHostToDevice));
        for (int boff : OFFS){
        uint8_t* dB = dBbase + boff;
        CU(cudaMemcpy(dB,B.data(),B.size(),cudaMemcpyHostToDevice));
        for (int M : Ms){
            std::vector<uint8_t> A((size_t)M*K); std::vector<float> asv((size_t)M*KB);
            for(auto&b:A){ uint8_t v; do { v=(uint8_t)(rand()&0xff); } while(v==0x7f||v==0xff); b=v; }
            for(auto&x:asv) x = (rand()%2000+100)/1000.f;
            uint8_t* dA; float *das,*Cref,*Csm,*Cm16;
            CU(cudaMalloc(&dA,A.size())); CU(cudaMalloc(&das,asv.size()*4));
            CU(cudaMalloc(&Cref,(size_t)M*N*4)); CU(cudaMalloc(&Csm,(size_t)M*N*4)); CU(cudaMalloc(&Cm16,(size_t)M*N*4));
            CU(cudaMemcpy(dA,A.data(),A.size(),cudaMemcpyHostToDevice));
            CU(cudaMemcpy(das,asv.data(),asv.size()*4,cudaMemcpyHostToDevice));

            g_tc_fp8=false;                                    // -> fp8_block_gemm_kernel, the oracle
            fp8_block_gemm(Cref,dA,das,dB,dbs,M,N,K,0);
            tc_fp8_set_smem(1); tc_fp8_gemm(Csm ,dA,das,dB,dbs,M,N,K,0);
            tc_fp8_set_smem(0); tc_fp8_gemm(Cm16,dA,das,dB,dbs,M,N,K,0);
            tc_fp8_set_smem(1);
            CU(cudaDeviceSynchronize());

            std::vector<float> cr((size_t)M*N), cs((size_t)M*N), cm((size_t)M*N);
            CU(cudaMemcpy(cr.data(),Cref,(size_t)M*N*4,cudaMemcpyDeviceToHost));
            CU(cudaMemcpy(cs.data(),Csm ,(size_t)M*N*4,cudaMemcpyDeviceToHost));
            CU(cudaMemcpy(cm.data(),Cm16,(size_t)M*N*4,cudaMemcpyDeviceToHost));
            Cmp a=compare(cs,cm), b=compare(cs,cr);
            bool okab = ok(a) && ok(b); pass = pass && okab;
            if(!okab || getenv("GATE_VERBOSE"))
                printf("%-20s %4d B+%d | cos=%.9f rms=%.1e %-4s | cos=%.9f rms=%.1e %-4s\n",
                       sh.name, M, boff, a.cos,a.rms_rel, ok(a)?"PASS":"FAIL", b.cos,b.rms_rel, ok(b)?"PASS":"FAIL");
            cudaFree(dA); cudaFree(das); cudaFree(Cref); cudaFree(Csm); cudaFree(Cm16);
        }
        }
        printf("%-20s  all M, B+0 and B+4 -> %s\n", sh.name, pass?"PASS":"FAIL");
        cudaFree(dBbase); cudaFree(dbs);
    }
    printf("\nGate TC_FP8_SMEM: %s\n", pass?"PASS":"FAIL");
    return pass?0:1;
}
