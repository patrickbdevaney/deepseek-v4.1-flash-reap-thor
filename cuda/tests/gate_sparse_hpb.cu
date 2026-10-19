// gate_sparse_hpb.cu — DECODE_LADDER 1.7. TWO claims about `sparse_attn`, in one binary, in seconds,
// with no checkpoint:
//
//   1. BIT-EXACTNESS, by memcmp and not by cosine. Every (hpb, smem) launch must produce the
//      BYTE-IDENTICAL output buffer to the hpb=1, smem=0 launch that shipped before 1.7. The claim
//      is value equality -- `sparse_attn` sums the gathered rows IN ORDER and its online softmax is
//      not associative -- so a tolerance gate would pass exactly the change this must catch. This is
//      the same reasoning as gate_tc_fp8_kc and gate_og_ws1, and the same reasoning that made
//      gate_topk_radix catch a reordering that was numerically perfect.
//
//   2. THE BAND, at the shapes the ENGINE issues and not at a shape that flatters the kernel.
//      `finish_attn` calls sparse_attn(b=1, m=K, h=64, d=512, n=ntot, topk=wmax+topkc) where
//      wmax = min(pos+K, WINDOW) = 128 and topkc = min(INDEX_TOPK, ctx/ratio) = min(512, ctx/4).
//      So the saturated shape is topk=640 and the pre-knee shape is topk=320 (ctx 768), which is
//      the concavity 1.3's re-attribution found and rule 7 warns about. K is the verify width; 0.4
//      measured a mean of 2, and 1 (base AR) and 6 (max) both occur.
//
// WHY A MICROBENCH IS THE RIGHT INSTRUMENT HERE AND NOT AN INSTRUMENT ITEM. It is phase 0 of the
// A/B, exactly as gate_index_score was for 1.5: it costs seconds and it decides WHICH kernel gets
// the five checkpoint loads. The engine-level before/after with tau is still the ratchet.
//
// A NEGATIVE CONTROL IS RUN, not asserted: `--control` perturbs one gathered row by one ulp and the
// memcmp must FAIL. A gate that has never failed has not been shown to be able to.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include "mla_attn.h"
#include "deepseek_v4.h"
using namespace dsv4;

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ printf("CUDA %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e)); exit(2);} }while(0)

struct Shape { const char* name; int m, topk, n; };

static uint32_t rng_s = 0x9e3779b9u;
static float frand(){ rng_s ^= rng_s<<13; rng_s ^= rng_s>>17; rng_s ^= rng_s<<5;
                      return ((float)(rng_s>>8) / 8388608.0f) - 1.0f; }

int main(int argc, char** argv){
    bool control = false;
    for(int i=1;i<argc;++i) if(!strcmp(argv[i],"--control")) control = true;
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,0));
    printf("device %s  SMs %d  L2 %.1f MB\n\n", p.name, p.multiProcessorCount, p.l2CacheSize/1e6);

    const int b=1, h=N_HEADS, d=HEAD_DIM;      // 1, 64, 512 — the engine's only shape
    Shape shapes[] = {
        {"m=1  topk=640 (base AR, ctx>=2048)", 1, 640, 20480},
        {"m=2  topk=640 (mean verify width)",  2, 640, 20480},
        {"m=6  topk=640 (max verify width)",   6, 640, 20480},
        {"m=2  topk=320 (ctx 768, pre-knee)",  2, 320,  5120},
        // PREFILL. `compressed_attn_forward` passes m=s and topk=s+T, and B9 sized the HPB
        // heuristic on exactly this shape -- so a new default chosen on the decode shapes has to be
        // shown not to give that back. total = 1*1022*64 = 65,408 warps, i.e. HPB=8 today.
        {"m=1022 topk=1277 (1022-tok prefill)", 1022, 1277, 1277},
        {"m=256  topk=320  (short prefill)",     256,  320,  320},
    };
    struct Cfg { int hpb, smem; };
    Cfg cfgs[] = {{1,0},{2,0},{4,0},{8,0},{1,1},{2,1},{4,1},{8,1},{2,2},{4,2},{8,2}};
    const int NC = (int)(sizeof(cfgs)/sizeof(cfgs[0]));

    int fails = 0, ctlfails = 0;
    for (Shape S : shapes) {
        const size_t qn = (size_t)b*S.m*h*d, kvn = (size_t)b*S.n*d, on = qn, in_ = (size_t)b*S.m*S.topk;
        std::vector<float> hq(qn), hkv(kvn), hs(h);
        std::vector<int>   hidx(in_);
        for (auto& v : hq)  v = frand()*0.5f;
        for (auto& v : hkv) v = frand();
        for (auto& v : hs)  v = frand();
        // Indices as the engine builds them: [window rows] then [selected compressed rows], and a
        // handful of -1 masked slots, which is the branch the block-uniform prefetch depends on.
        for (int mi=0; mi<S.m; ++mi) for (int k=0;k<S.topk;++k) {
            int v = (k<128) ? k : (128 + ((k*37 + mi*11) % (S.n-128)));
            if ((k % 97) == 96) v = -1;
            hidx[(size_t)mi*S.topk + k] = v;
        }
        float *dq,*dkv,*dsink,*dout,*dref; int* didx;
        CK(cudaMalloc(&dq,qn*4)); CK(cudaMalloc(&dkv,kvn*4)); CK(cudaMalloc(&dsink,h*4));
        CK(cudaMalloc(&dout,on*4)); CK(cudaMalloc(&dref,on*4)); CK(cudaMalloc(&didx,in_*4));
        CK(cudaMemcpy(dq,hq.data(),qn*4,cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dkv,hkv.data(),kvn*4,cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dsink,hs.data(),h*4,cudaMemcpyHostToDevice));
        CK(cudaMemcpy(didx,hidx.data(),in_*4,cudaMemcpyHostToDevice));
        const float scale = 1.f/sqrtf((float)d);

        // reference = the launch that shipped before 1.7
        sparse_attn_launch(dref,dq,dkv,dsink,didx,b,S.m,h,d,S.n,S.topk,scale,0,1,0);
        CK(cudaDeviceSynchronize());
        std::vector<float> href(on); CK(cudaMemcpy(href.data(),dref,on*4,cudaMemcpyDeviceToHost));

        printf("=== %s   (blocks at hpb: %d/%d/%d/%d on %d SMs) ===\n", S.name,
               b*S.m*h, b*S.m*h/2, b*S.m*h/4, b*S.m*h/8, p.multiProcessorCount);
        std::vector<float> hout(on);
        cudaEvent_t e0,e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
        float base_ms = 0.f;
        for (int c=0;c<NC;++c) {
            int hpb=cfgs[c].hpb, smem=cfgs[c].smem;
            CK(cudaMemset(dout,0xCD,on*4));
            sparse_attn_launch(dout,dq,dkv,dsink,didx,b,S.m,h,d,S.n,S.topk,scale,0,hpb,smem);
            CK(cudaDeviceSynchronize());
            CK(cudaMemcpy(hout.data(),dout,on*4,cudaMemcpyDeviceToHost));
            size_t diff=0; for(size_t z=0;z<on;++z) if(memcmp(&hout[z],&href[z],4)) ++diff;
            // timing: warm + timed, whole-launch wall time. Iteration count falls with the work
            // per launch so a prefill shape does not turn a seconds-long gate into a minutes-long one.
            const int ITER = (S.m >= 256) ? 10 : 100, WARM = (S.m >= 256) ? 3 : 20;
            for(int w=0;w<WARM;++w) sparse_attn_launch(dout,dq,dkv,dsink,didx,b,S.m,h,d,S.n,S.topk,scale,0,hpb,smem);
            CK(cudaDeviceSynchronize());
            CK(cudaEventRecord(e0));
            for(int w=0;w<ITER;++w) sparse_attn_launch(dout,dq,dkv,dsink,didx,b,S.m,h,d,S.n,S.topk,scale,0,hpb,smem);
            CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
            float ms=0; CK(cudaEventElapsedTime(&ms,e0,e1)); ms/=(float)ITER;
            if (hpb==1 && smem==0) base_ms = ms;
            // bytes the kernel would move with NO reuse at all, i.e. what it asks the hierarchy for
            double gb = (double)b*S.m*h*S.topk*d*4.0/1e9;
            printf("  hpb=%d smem=%d  %8.4f ms  %6.2fx  %7.1f GB/s(no-reuse)  bytediff=%zu %s\n",
                   hpb, smem, ms, base_ms/ms, gb/(ms/1e3), diff,
                   diff? "  <<<< NOT BIT-EXACT" : "");
            if (diff) ++fails;
        }
        if (control) {
            // NEGATIVE CONTROL: one ulp on one gathered row must be visible to this memcmp.
            const int crow = 5;   // a WINDOW row: hidx[k]=k for k<128, so it is gathered at every shape
            float one; CK(cudaMemcpy(&one, dkv + (size_t)crow*d + 7, 4, cudaMemcpyDeviceToHost));
            float bumped = nextafterf(one, 1e30f);
            CK(cudaMemcpy(dkv + (size_t)crow*d + 7, &bumped, 4, cudaMemcpyHostToDevice));
            sparse_attn_launch(dout,dq,dkv,dsink,didx,b,S.m,h,d,S.n,S.topk,scale,0,1,0);
            CK(cudaDeviceSynchronize());
            CK(cudaMemcpy(hout.data(),dout,on*4,cudaMemcpyDeviceToHost));
            size_t diff=0; for(size_t z=0;z<on;++z) if(memcmp(&hout[z],&href[z],4)) ++diff;
            printf("  [control] one ulp on kv row %d -> %zu of %zu floats differ %s\n", crow,
                   diff, on, diff? "(gate can fail: good)" : "  <<<< GATE IS BLIND");
            if (!diff) ++ctlfails;
            CK(cudaMemcpy(dkv + (size_t)crow*d + 7, &one, 4, cudaMemcpyHostToDevice));
        }
        printf("\n");
        CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
        CK(cudaFree(dq));CK(cudaFree(dkv));CK(cudaFree(dsink));CK(cudaFree(dout));CK(cudaFree(dref));CK(cudaFree(didx));
    }
    printf("%s  (bit-exactness failures %d, blind-control failures %d)\n",
           (fails||ctlfails)?"GATE SPARSE_HPB: FAIL":"GATE SPARSE_HPB: PASS", fails, ctlfails);
    return (fails||ctlfails) ? 1 : 0;
}
