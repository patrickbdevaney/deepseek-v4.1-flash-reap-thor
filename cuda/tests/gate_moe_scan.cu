// gate_moe_scan.cu — the parallel MoE grouping scans must be BIT-IDENTICAL to the <<<1,1>>> ones.
//
// k_moe_prefix and k_build_tiles were one thread walking nr=160 experts, once per layer (Finding 71's
// class). The replacements are one-block, 256-thread Hillis-Steele scans. Both produce integers by
// addition in the same order, so the correct gate is EQUALITY, not cosine — and equality is only
// meaningful if the harness spans the regimes the kernels see. Per LEVERS.md trap 9 ("check the
// harness can express the regime"), this sweeps:
//   - nr from 1 to 400, i.e. below, at, and ABOVE the 256-thread block width (the chunk carry);
//   - empty experts (me=0), single-tile, exactly-16, and multi-tile experts (me>16);
//   - the all-empty case and the all-on-one-expert case.
//
// build: nvcc -O2 -std=c++17 -gencode arch=compute_110a,code=sm_110a -I include \
//        tests/gate_moe_scan.cu kernels/moe.cu kernels/tc_moe_gemm.cu kernels/dscratch.cu \
//        kernels/dprof.cu kernels/fp8_block_gemm.cu kernels/tc_fp8_gemm.cu kernels/mla_attn.cu \
//        -o build/gate_moe_scan
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CU(x) do{cudaError_t e=(x); if(e){fprintf(stderr,"cuda %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)

__global__ void k_moe_prefix(int*, const int*, int);
__global__ void k_moe_prefix_par(int*, const int*, int);
__global__ void k_build_tiles(int*, int*, int*, const int*, int);
__global__ void k_build_tiles_par(int*, int*, int*, const int*, int);

static int g_fail = 0;

// One case: counts[] -> off[] (both scans) -> tiles[] (both scans). Every array compared exactly.
static void one(int nr, const std::vector<int>& counts, const char* what){
    int total = 0; for(int c : counts) total += c;
    const int maxt = total/16 + nr + 8;                     // upper bound on tiles

    int *d_counts,*d_offA,*d_offB,*d_teA,*d_teB,*d_r0A,*d_r0B,*d_ntA,*d_ntB;
    CU(cudaMalloc(&d_counts, nr*4));
    CU(cudaMalloc(&d_offA, (nr+1)*4));   CU(cudaMalloc(&d_offB, (nr+1)*4));
    CU(cudaMalloc(&d_teA, maxt*4));      CU(cudaMalloc(&d_teB, maxt*4));
    CU(cudaMalloc(&d_r0A, maxt*4));      CU(cudaMalloc(&d_r0B, maxt*4));
    CU(cudaMalloc(&d_ntA, 4));           CU(cudaMalloc(&d_ntB, 4));
    CU(cudaMemcpy(d_counts, counts.data(), nr*4, cudaMemcpyHostToDevice));
    // Poison the outputs: a kernel that writes nothing must not pass by inheriting the other's bytes.
    CU(cudaMemset(d_offA,0xEE,(nr+1)*4)); CU(cudaMemset(d_offB,0xEE,(nr+1)*4));
    CU(cudaMemset(d_teA,0xEE,maxt*4));    CU(cudaMemset(d_teB,0xEE,maxt*4));
    CU(cudaMemset(d_r0A,0xEE,maxt*4));    CU(cudaMemset(d_r0B,0xEE,maxt*4));

    k_moe_prefix    <<<1,1,  0,0>>>(d_offA, d_counts, nr);
    k_moe_prefix_par<<<1,256,0,0>>>(d_offB, d_counts, nr);
    k_build_tiles    <<<1,1,  0,0>>>(d_teA, d_r0A, d_ntA, d_offA, nr);
    k_build_tiles_par<<<1,256,0,0>>>(d_teB, d_r0B, d_ntB, d_offB, nr);
    CU(cudaDeviceSynchronize());
    CU(cudaGetLastError());

    std::vector<int> oA(nr+1), oB(nr+1), teA(maxt), teB(maxt), r0A(maxt), r0B(maxt);
    int ntA=0, ntB=0;
    CU(cudaMemcpy(oA.data(), d_offA,(nr+1)*4, cudaMemcpyDeviceToHost));
    CU(cudaMemcpy(oB.data(), d_offB,(nr+1)*4, cudaMemcpyDeviceToHost));
    CU(cudaMemcpy(&ntA, d_ntA,4, cudaMemcpyDeviceToHost));
    CU(cudaMemcpy(&ntB, d_ntB,4, cudaMemcpyDeviceToHost));
    CU(cudaMemcpy(teA.data(), d_teA,maxt*4, cudaMemcpyDeviceToHost));
    CU(cudaMemcpy(teB.data(), d_teB,maxt*4, cudaMemcpyDeviceToHost));
    CU(cudaMemcpy(r0A.data(), d_r0A,maxt*4, cudaMemcpyDeviceToHost));
    CU(cudaMemcpy(r0B.data(), d_r0B,maxt*4, cudaMemcpyDeviceToHost));

    int bad = 0;
    for(int e=0;e<=nr;++e) if(oA[e]!=oB[e]) ++bad;
    if(oA[nr]!=total) ++bad;                                 // the serial scan is itself checked
    if(ntA!=ntB) ++bad;
    for(int t=0;t<ntA && t<maxt;++t) if(teA[t]!=teB[t] || r0A[t]!=r0B[t]) ++bad;
    printf("[moe_scan] %-28s nr=%3d rows=%4d tiles=%4d/%4d : %s\n",
           what, nr, total, ntA, ntB, bad? "FAIL" : "PASS");
    if(bad) g_fail = 1;

    cudaFree(d_counts); cudaFree(d_offA); cudaFree(d_offB); cudaFree(d_teA); cudaFree(d_teB);
    cudaFree(d_r0A); cudaFree(d_r0B); cudaFree(d_ntA); cudaFree(d_ntB);
}

int main(){
    srand(1234);
    // The shipped shape: nr=160, bs*na=30 assignments, the Finding 70 rows-per-expert histogram.
    { std::vector<int> c(160,0); for(int i=0;i<30;++i) c[rand()%160]++; one(160,c,"shipped nr=160 30 rows"); }
    // Regime coverage. 256 and 257 straddle the block width; 400 needs two chunk carries.
    for(int nr : {1, 2, 15, 16, 17, 160, 255, 256, 257, 400}){
        std::vector<int> c(nr,0);
        for(int i=0;i<nr*3;++i) c[rand()%nr]++;              // dense: many multi-tile experts
        char t[64]; snprintf(t,sizeof t,"dense"); one(nr,c,t);
        std::vector<int> z(nr,0);
        one(nr,z,"all-empty");                              // every expert me=0 -> 0 tiles
        std::vector<int> s(nr,0); s[nr-1]=97;               // one expert, 7 tiles, LAST slot
        one(nr,s,"one-expert-last");
        std::vector<int> b(nr,0); for(int e=0;e<nr;++e) b[e]=16;   // exactly 16 -> exactly 1 tile each
        one(nr,b,"exactly-16-each");
    }
    printf("\nGate MOE_SCAN: %s\n", g_fail? "FAIL" : "PASS");
    return g_fail;
}
