// gate_tc_fp8_kc.cu — the K-chunked staging of tc_fp8_smemB_kernel (LOOP_LOG Finding 74) must be
// BIT-IDENTICAL to KC=1, the kernel it replaces. Equality, not cosine.
//
// gate_tc_fp8_smem already sweeps these shapes, but it is a COSINE gate against the warp-per-output
// oracle — it passes anything whose summation order merely differs. The Finding-74 claim is stronger
// and much cheaper to check: KC changes only how many 128-K-blocks are staged per barrier pair, and
// `acc` still takes each block's `cb` in kblk order 0,1,2,..., so every output float must match to
// the BIT. Finding 68 is why that distinction is gated rather than argued: a reduction-order change
// that looked fine on cosine sent the model into a repeating loop and produced a fake +28% tok/s.
//
// Per LEVERS.md trap 9, the sweep has to be able to express the regimes:
//   - M crosses the m16 tile boundary (1,2,3,5,8,13,16,17,20) so grid.y>1 is exercised
//   - every W the dispatcher picks: N=32768/4096 -> W=8, 1024 -> W=2, 512 -> W=1, plus an N%64 tail
//   - K=1024/2048/4096/8192 so KB=8..64, i.e. KC both divides KB exactly and is capped by it
//   - B at offset 0 AND 4: real weights are 4-byte aligned (Finding 66), so AL16=false is the
//     path decode actually runs and the uint4 path is the one only a gate ever sees
//   - outputs poisoned to 0xEE first, so a kernel that writes nothing cannot inherit a pass
//
//   build: nvcc -O2 -std=c++17 -arch=sm_110a -I include tests/gate_tc_fp8_kc.cu \
//            kernels/fp8_block_gemm.cu kernels/tc_fp8_gemm.cu kernels/dscratch.cu -o build/gate_tc_fp8_kc
#include <cuda_runtime.h>
#include <stdint.h>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#define CU(x) do{auto e=(x); if(e){printf("ERR %s:%d %d\n",__FILE__,__LINE__,(int)e);return 1;}}while(0)

void tc_fp8_gemm(float*, const uint8_t*, const float*, const uint8_t*, const float*, int,int,int, cudaStream_t);
extern bool g_tc_fp8;
void tc_fp8_set_smem(int on);
void tc_fp8_set_kc(int kc);
void tc_fp8_set_db(int on);
// cp.async ring depth (LEVERS.md B8-cpasync). Like KC and DB it changes only WHEN and HOW the staged
// bytes reach smem — one K-block per ring stage, `acc` still takes `cb` in kblk order — so it must be
// bit-identical to KC=1 too. NS is swept over the depths the dispatcher can pick, INCLUDING a depth
// larger than KB so the empty-commit_group padding in the prologue/epilogue is exercised: with
// K=1024, KB=8, and the wait depth held at a compile-time NS-2, a missing pad group would let the mma
// read a stage that has not landed. That is a race, so it is checked at every M and both offsets.
void tc_fp8_set_cpa(int ns);

int main(){
    struct Shape { int N, K; const char* name; };
    const Shape shapes[] = {
        { 1024, 4096, "wq_a   [1024,4096] W=2"},
        {32768, 1024, "wq_b   [32768,1024] W=8"},   // the REAL wq_b, not the bench's [4096,1024]
        {  512, 4096, "wkv    [512,4096]  W=1"},    // the only W=1 shape -> auto KC=8
        { 4096, 8192, "wo_b   [4096,8192] W=8"},    // K=8192 -> KB=64
        { 2048, 4096, "shared [2048,4096] W=8"},
        { 4096, 2048, "sw2    [4096,2048] W=8"},
        {  520, 1024, "N-tail [520,1024]  W=2"},    // 520 = 8*65: partial last n-block
    };
    const int Ms[]  = {1,2,3,5,8,13,16,17,20};
    const int KCs[]  = {2,4,8,-1};                  // -1 = the shipped auto rule
    const int KC1s[] = {1,2,4,8,-1};                // db=1 must also match at KC=1
    const int OFFS[] = {0,4};
    const int NSs[]  = {2,4,8};                     // cp.async ring depths (clamped per W by smem)
    srand(11);
    bool pass = true; int cases = 0, bad = 0;
    g_tc_fp8 = true; tc_fp8_set_smem(1);
    printf("KC in {2,4,8,auto} x DB in {0,1} + CPA in {2,4,8} vs KC=1/DB=0/CPA=0, bit-equality;\n");
    printf("M={1,2,3,5,8,13,16,17,20} x B offset {0,4}\n");
    for (const Shape& sh : shapes){
        const int N=sh.N, K=sh.K, KB=K/128;
        std::vector<uint8_t> B((size_t)N*K);
        std::vector<float>   bsv((size_t)((N+127)/128)*KB);
        // 0x7f/0xff are e4m3 NaN: a NaN makes every comparison below vacuously unequal.
        for(auto&b:B){ uint8_t v; do { v=(uint8_t)(rand()&0xff); } while(v==0x7f||v==0xff); b=v; }
        for(auto&x:bsv) x = (rand()%2000+100)/1000.f;
        uint8_t* dBbase; float* dbs;
        CU(cudaMalloc(&dBbase,B.size()+16)); CU(cudaMalloc(&dbs,bsv.size()*4));
        CU(cudaMemcpy(dbs,bsv.data(),bsv.size()*4,cudaMemcpyHostToDevice));
        bool shape_ok = true;
        for (int boff : OFFS){
            uint8_t* dB = dBbase + boff;
            CU(cudaMemcpy(dB,B.data(),B.size(),cudaMemcpyHostToDevice));
            for (int M : Ms){
                std::vector<uint8_t> A((size_t)M*K); std::vector<float> asv((size_t)M*KB);
                for(auto&b:A){ uint8_t v; do { v=(uint8_t)(rand()&0xff); } while(v==0x7f||v==0xff); b=v; }
                for(auto&x:asv) x = (rand()%2000+100)/1000.f;
                uint8_t* dA; float *das,*C1,*Ck;
                CU(cudaMalloc(&dA,A.size())); CU(cudaMalloc(&das,asv.size()*4));
                CU(cudaMalloc(&C1,(size_t)M*N*4)); CU(cudaMalloc(&Ck,(size_t)M*N*4));
                CU(cudaMemcpy(dA,A.data(),A.size(),cudaMemcpyHostToDevice));
                CU(cudaMemcpy(das,asv.data(),asv.size()*4,cudaMemcpyHostToDevice));

                CU(cudaMemset(C1,0xEE,(size_t)M*N*4));
                tc_fp8_set_kc(1); tc_fp8_set_db(0); tc_fp8_set_cpa(0);
                tc_fp8_gemm(C1,dA,das,dB,dbs,M,N,K,0);
                CU(cudaDeviceSynchronize());
                std::vector<float> c1((size_t)M*N), ck((size_t)M*N);
                CU(cudaMemcpy(c1.data(),C1,(size_t)M*N*4,cudaMemcpyDeviceToHost));

                // db=1 (Finding 78) only changes WHEN the staged chunk's global loads are issued —
                // same bytes, same smem, same kblk order — so it must be bit-identical too, at every
                // KC including 1. Reference is KC=1/db=0, the pre-F74 AND pre-F78 kernel.
                for (int db : {0,1})
                for (int ki=0, nk = db ? (int)(sizeof(KC1s)/sizeof(int)) : (int)(sizeof(KCs)/sizeof(int));
                     ki<nk; ++ki){
                    const int kc = (db ? KC1s : KCs)[ki];
                    CU(cudaMemset(Ck,0xEE,(size_t)M*N*4));
                    tc_fp8_set_kc(kc); tc_fp8_set_db(db); tc_fp8_set_cpa(0);
                    tc_fp8_gemm(Ck,dA,das,dB,dbs,M,N,K,0);
                    CU(cudaDeviceSynchronize());
                    CU(cudaMemcpy(ck.data(),Ck,(size_t)M*N*4,cudaMemcpyDeviceToHost));
                    ++cases;
                    // memcmp, not ==: this is a bit-pattern claim about floats.
                    if (memcmp(c1.data(), ck.data(), (size_t)M*N*4) != 0){
                        size_t i=0; while(i<c1.size() && c1[i]==ck[i]) ++i;
                        printf("  MISMATCH %-22s M=%-3d B+%d KC=%-2d DB=%d  first at [%zu]: %.9g vs %.9g\n",
                               sh.name, M, boff, kc, db, i, (double)c1[i], (double)ck[i]);
                        shape_ok=false; pass=false; ++bad;
                    }
                }
                // --- cp.async ring (B8-cpasync). Same reference: KC=1 / DB=0 / CPA=0. ---
                for (int ns : NSs){
                    CU(cudaMemset(Ck,0xEE,(size_t)M*N*4));
                    tc_fp8_set_kc(-1); tc_fp8_set_db(0); tc_fp8_set_cpa(ns);
                    tc_fp8_gemm(Ck,dA,das,dB,dbs,M,N,K,0);
                    CU(cudaDeviceSynchronize());
                    CU(cudaMemcpy(ck.data(),Ck,(size_t)M*N*4,cudaMemcpyDeviceToHost));
                    ++cases;
                    if (memcmp(c1.data(), ck.data(), (size_t)M*N*4) != 0){
                        size_t i=0; while(i<c1.size() && c1[i]==ck[i]) ++i;
                        printf("  MISMATCH %-22s M=%-3d B+%d CPA=%-2d      first at [%zu]: %.9g vs %.9g\n",
                               sh.name, M, boff, ns, i, (double)c1[i], (double)ck[i]);
                        shape_ok=false; pass=false; ++bad;
                    }
                }
                tc_fp8_set_cpa(0);
                CU(cudaFree(dA)); CU(cudaFree(das)); CU(cudaFree(C1)); CU(cudaFree(Ck));
            }
        }
        printf("%-24s all M x offset x KC -> %s\n", sh.name, shape_ok?"EXACT":"MISMATCH");
        CU(cudaFree(dBbase)); CU(cudaFree(dbs));
    }
    tc_fp8_set_kc(-1); tc_fp8_set_db(-1); tc_fp8_set_cpa(-1);
    printf("\nGate TC_FP8_KC: %d/%d exact -> %s\n", cases-bad, cases, pass?"PASS":"FAIL");
    return pass?0:1;
}
