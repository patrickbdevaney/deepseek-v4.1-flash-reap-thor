// gate_grouped_moe.cu — validate the zero-sync GROUPED W4A8 GEMM (Step 1b) cosine-1.0 vs the per-expert
// tc_fp4_gemm_pp loop on IDENTICAL repacked weights + scales. The grouped kernel dispatches up-to-16-row
// tiles across all experts in ONE launch, reading the expert map from device off[] (no host sync). Coverage:
// empty experts (me=0) and multi-tile experts (me>16), varied row counts. Same bytes -> expect ~bit-exact.
//   build: nvcc -O2 -std=c++17 -arch=sm_110a -I include tests/gate_grouped_moe.cu kernels/tc_moe_gemm.cu -o build/gate_grouped_moe
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#define CU(x) do{cudaError_t e=(x); if(e){fprintf(stderr,"cuda %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)
// oracle (moe.cu): warp-per-output fp8×fp4 GEMM, arbitrary M, reads ORIGINAL (non-repacked) weight + f32 scale.
void fp4_gemm(float*, const uint8_t*, const float*, const uint8_t*, const float*, int,int,int, cudaStream_t);
// tc_moe_gemm.cu exports:
void tc_ensure_repacked(uint8_t*, int, int, cudaStream_t);
void tc_a_to_fp16(__half*, const uint8_t*, const float*, int, int, cudaStream_t);
void tc_build_tiles(int*, int*, int*, const int*, int, cudaStream_t);
void tc_fp4_grouped_gemm(float*, const __half*, const uint8_t* const*, const float* const*,
                         const int*, const int*, const int*, const int*, int, int, int, cudaStream_t);

int main(int argc, char** argv){
    const int N=64, K=256;                              // N/8=8 n-blocks, K/128=2 k-groups, K/32=8 scale-blocks
    std::vector<int> me = {5, 0, 17, 3, 20, 8};         // per-expert row counts: empty (e=1), multi-tile (e2,e4)
    const int nr = me.size();
    std::vector<int> off(nr+1,0); for(int e=0;e<nr;++e) off[e+1]=off[e]+me[e];
    const int total = off[nr];
    srand(4242);

    // per-expert weights (fp4 packed [N,K/2]) + scales ([N,K/32] f32 pow2-ish), each its own device buffer.
    std::vector<uint8_t*> Wp(nr), Wp0(nr); std::vector<float*> Sp(nr);   // Wp: repacked (grouped) | Wp0: original (oracle)
    for(int ex=0;ex<nr;++ex){                            // NB: not 'e' — the CU macro declares cudaError_t e (would shadow)
        std::vector<uint8_t> b((size_t)N*(K/2)); std::vector<float> bs((size_t)N*(K/32));
        for(auto& v:b) v=rand()&0xff; for(auto& v:bs) v=0.5f+0.01f*(rand()%50);
        CU(cudaMalloc(&Wp[ex], b.size())); CU(cudaMalloc(&Wp0[ex], b.size())); CU(cudaMalloc(&Sp[ex], bs.size()*4));
        CU(cudaMemcpy(Wp[ex], b.data(), b.size(), cudaMemcpyHostToDevice));
        CU(cudaMemcpy(Wp0[ex], b.data(), b.size(), cudaMemcpyHostToDevice));
        CU(cudaMemcpy(Sp[ex], bs.data(), bs.size()*4, cudaMemcpyHostToDevice));
    }
    // gathered acts (already expert-grouped): fp8[total,K] + a_s[total,K/128]
    std::vector<uint8_t> A((size_t)total*K); std::vector<float> as((size_t)total*(K/128));
    for(auto& v:A) v=rand()%0x40; for(auto& v:as) v=0.5f+0.01f*(rand()%50);
    uint8_t* dA; float* das; CU(cudaMalloc(&dA,A.size())); CU(cudaMalloc(&das,as.size()*4));
    CU(cudaMemcpy(dA,A.data(),A.size(),cudaMemcpyHostToDevice)); CU(cudaMemcpy(das,as.data(),as.size()*4,cudaMemcpyHostToDevice));

    // ---- reference: fp4_gemm oracle (arbitrary M, original weights) per-expert contiguous row block ----
    float* Cref; CU(cudaMalloc(&Cref,(size_t)total*N*4)); CU(cudaMemset(Cref,0,(size_t)total*N*4));
    for(int e=0;e<nr;++e){ if(!me[e]) continue;
        fp4_gemm(Cref + (size_t)off[e]*N, dA + (size_t)off[e]*K, das + (size_t)off[e]*(K/128),
                 Wp0[e], Sp[e], me[e], N, K, 0); }
    CU(cudaDeviceSynchronize());

    // repack every expert weight IN PLACE once (identical to the load-time / pp-path repack) for the grouped path
    for(int e=0;e<nr;++e) tc_ensure_repacked(Wp[e], N, K, 0);
    CU(cudaDeviceSynchronize());

    // ---- grouped: single launch over all experts, expert map from device off[] ----
    const uint8_t** wptr_d; const float** sptr_d;
    CU(cudaMalloc(&wptr_d, nr*sizeof(void*))); CU(cudaMalloc(&sptr_d, nr*sizeof(void*)));
    CU(cudaMemcpy(wptr_d, Wp.data(), nr*sizeof(void*), cudaMemcpyHostToDevice));
    CU(cudaMemcpy(sptr_d, Sp.data(), nr*sizeof(void*), cudaMemcpyHostToDevice));
    int* off_d; CU(cudaMalloc(&off_d,(nr+1)*4)); CU(cudaMemcpy(off_d,off.data(),(nr+1)*4,cudaMemcpyHostToDevice));
    __half* x16; CU(cudaMalloc(&x16,(size_t)total*K*2)); tc_a_to_fp16(x16, dA, das, total, K, 0);
    int *tile_e,*tile_row0,*ntiles_d; CU(cudaMalloc(&tile_e,total*4)); CU(cudaMalloc(&tile_row0,total*4)); CU(cudaMalloc(&ntiles_d,4));
    tc_build_tiles(tile_e, tile_row0, ntiles_d, off_d, nr, 0);
    float* Cgrp; CU(cudaMalloc(&Cgrp,(size_t)total*N*4)); CU(cudaMemset(Cgrp,0,(size_t)total*N*4));
    tc_fp4_grouped_gemm(Cgrp, x16, (const uint8_t* const*)wptr_d, (const float* const*)sptr_d,
                        off_d, tile_e, tile_row0, ntiles_d, /*maxtiles*/total, N, K, 0);
    CU(cudaDeviceSynchronize());
    int nt=0; CU(cudaMemcpy(&nt, ntiles_d, 4, cudaMemcpyDeviceToHost));

    std::vector<float> cr((size_t)total*N), cg((size_t)total*N);
    CU(cudaMemcpy(cr.data(),Cref,cr.size()*4,cudaMemcpyDeviceToHost));
    CU(cudaMemcpy(cg.data(),Cgrp,cg.size()*4,cudaMemcpyDeviceToHost));
    double dot=0,nrr=0,ng=0,sd=0,sr=0,mx=0,ma=0;
    for(size_t i=0;i<cr.size();++i){ double r=cr[i],g=cg[i]; dot+=r*g; nrr+=r*r; ng+=g*g; sd+=(r-g)*(r-g); sr+=r*r; mx=fmax(mx,fabs(r)); ma=fmax(ma,fabs(r-g)); }
    double cosine=dot/(sqrt(nrr)*sqrt(ng)+1e-30), rms=sqrt(sd/(sr+1e-30)), absr=ma/(mx+1e-30);
    bool ok = cosine>0.999 && rms<3e-2;                  // fp16 mma (grouped) vs fp32 oracle (fp4_gemm) — same tol as gate_tc_moe
    printf("[grouped W4A8] nr=%d total=%d tiles=%d N=%d K=%d (vs fp4_gemm oracle)\n", nr, total, nt, N, K);
    printf("[grouped W4A8] cosine=%.7f rms_rel=%.2e max_abs/|c|max=%.2e -> %s\n", cosine, rms, absr, ok?"PASS":"FAIL");
    return ok?0:1;
}
