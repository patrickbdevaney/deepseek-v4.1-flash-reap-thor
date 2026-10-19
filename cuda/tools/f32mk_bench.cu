// f32mk_bench.cu — DECODE_LADDER 1.12. Times `gemm_fp32` at the shapes the ENGINE issues, for every
// (MM,NN) warp tile the dispatcher can select, and asserts every tile is BIT-IDENTICAL to the M=1
// warp-per-output path it replaces.
//
// Why a standalone bench: the shapes here are fixed by the model config, not by the weights, so
// this needs no checkpoint. The engine reaches the M=128 shape once every 128 positions, which is
// 8 samples in a 40-minute run; here it is 200 reps in a second.
//
//   build: see scripts/build_bench.sh (or the line in DECODE_LADDER 1.12)
//   run:   ./build/f32mk_bench [reps]
#include "compressor.h"
#include "deepseek_v4.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cmath>
#define CU(x) do{cudaError_t e=(x); if(e){fprintf(stderr,"cuda %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)

struct Tile { int mm, nn; };
static const Tile TILES[] = {{8,0},{8,1},{8,2},{8,4},{4,4},{4,8},{2,8},{16,1},{16,2},{32,1},{6,4},{4,6},{6,6},{5,5},{4,2},{2,4},{4,1},{2,2}};
static const int NT = sizeof(TILES)/sizeof(TILES[0]);

struct Shape { const char* name; int M, N, K; int per_fwd; };
// per_fwd = how many calls of this shape one EMIT step issues (20 odd layers are ratio-128; the 21
// even layers are ratio-4 and emit main+indexer). See kernels/compressed_decode.cu.
static const Shape SHAPES[] = {
    {"strided emit  wkv/wgate", 128, 512,  4096, 40},   // 20 layers x {wkv, wgate}, overlap=false -> od=d=512
    {"indexer emit  main",        8, 1024, 4096, 42},   // 21 layers x {wkv, wgate}, overlap=true  -> od=2*512
    {"indexer emit  idx",         8, 256,  4096, 42},   // 21 layers x {wkv, wgate}, overlap=true  -> od=2*128
    {"indexer iw (K=5 verify)",   5, 64,   4096, 21},
};
static const int NS = sizeof(SHAPES)/sizeof(SHAPES[0]);

int main(int argc, char** argv){
    const int reps = argc>1 ? atoi(argv[1]) : 50;
    srand(1234);
    int fail = 0;
    printf("f32mk_bench: reps=%d  (ms = mean over reps after 5 warmup; bytes = (M+N)*K*4 + M*N*4)\n\n", reps);
    for (int si=0; si<NS; ++si){
        const Shape& s = SHAPES[si];
        std::vector<float> ha((size_t)s.M*s.K), hb((size_t)s.N*s.K);
        for(auto&x:ha) x=(rand()%2000-1000)/1000.f;
        for(auto&x:hb) x=(rand()%2000-1000)/1000.f;
        float *dA,*dB,*dC,*dR;
        CU(cudaMalloc(&dA,ha.size()*4)); CU(cudaMalloc(&dB,hb.size()*4));
        CU(cudaMalloc(&dC,(size_t)s.M*s.N*4)); CU(cudaMalloc(&dR,(size_t)s.M*s.N*4));
        CU(cudaMemcpy(dA,ha.data(),ha.size()*4,cudaMemcpyHostToDevice));
        CU(cudaMemcpy(dB,hb.data(),hb.size()*4,cudaMemcpyHostToDevice));
        // reference: the warp-per-output-element path (NO_FP32MK), which is what the M=K path was
        // proven equal to in the first place (gate_bf16w).
        setenv("NO_FP32MK","1",1);
        gemm_fp32(dR,dA,dB,s.M,s.N,s.K,0); CU(cudaDeviceSynchronize());
        unsetenv("NO_FP32MK");
        std::vector<float> href((size_t)s.M*s.N), hout((size_t)s.M*s.N);
        CU(cudaMemcpy(href.data(),dR,href.size()*4,cudaMemcpyDeviceToHost));

        const double bytes = ((double)s.M + s.N)*s.K*4.0 + (double)s.M*s.N*4.0;
        printf("== %-26s M=%-4d N=%-5d K=%d   (%d calls per emit step)\n", s.name, s.M, s.N, s.K, s.per_fwd);
        double base_ms = 0;
        for (int t=0;t<NT;++t){
            gemm_fp32_set_tile(TILES[t].mm, TILES[t].nn);
            CU(cudaMemset(dC,0,(size_t)s.M*s.N*4));
            for(int w=0;w<5;++w) gemm_fp32(dC,dA,dB,s.M,s.N,s.K,0);
            CU(cudaDeviceSynchronize());
            cudaEvent_t e0,e1; CU(cudaEventCreate(&e0)); CU(cudaEventCreate(&e1));
            CU(cudaEventRecord(e0));
            for(int r=0;r<reps;++r) gemm_fp32(dC,dA,dB,s.M,s.N,s.K,0);
            CU(cudaEventRecord(e1)); CU(cudaEventSynchronize(e1));
            float ms=0; CU(cudaEventElapsedTime(&ms,e0,e1)); ms/=reps;
            CU(cudaEventDestroy(e0)); CU(cudaEventDestroy(e1));
            CU(cudaMemcpy(hout.data(),dC,hout.size()*4,cudaMemcpyDeviceToHost));
            double md=0; for(size_t i=0;i<hout.size();++i) md=fmax(md,fabs((double)hout[i]-href[i]));
            const bool exact = (md==0.0);
            if(!exact) ++fail;
            if(t==0) base_ms = ms;
            printf("   %2dx%-2d%s%8.4f ms  %7.1f GB/s  %5.2fx vs pre   per-emit %7.3f ms   max|diff|=%.1e %s\n",
                   TILES[t].mm, TILES[t].nn, TILES[t].nn?"  ":"* ", ms, bytes/ms/1e6, base_ms/ms, ms*s.per_fwd, md, exact?"EXACT":"*** DIFFERS ***");
        }
        printf("\n");
        cudaFree(dA);cudaFree(dB);cudaFree(dC);cudaFree(dR);
    }
    printf("f32mk_bench: %s\n", fail?"BIT-EXACTNESS FAIL":"all tiles bit-identical to the M=1 path");
    return fail?1:0;
}
