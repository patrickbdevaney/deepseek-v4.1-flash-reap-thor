// hc_glue_bench.cu — where does the 8.53 ms of base-AR "glue" go, and is any of it recoverable?
//
// F137's corrected base-AR budget leaves one region nobody has attacked: 8.53 ms of the 70.03 ms
// K=1 step (12 %) spent in hc_pre / hc_post / rmsnorm / moe:router, moving 211 MB — i.e. ~25 GB/s
// against a 208 GB/s roofline. It is latency, not bytes, and LEVERS §4 has dismissed it as such
// ("latency-bound, not bytes") without ever taking it apart.
//
// The two marks that dominate it are `hc_pre (ffn)` at 2.77 ms and `hc_pre (attn)` at 1.59 ms
// (evidence/kchunk.log, K=1, 43 layers). **Those are the SAME FUNCTION on the SAME SHAPES with the
// same 1.57 MB weight, and one costs 1.74x the other** — 64.4 vs 37.0 us per layer. That asymmetry
// is not explainable from the source, so it is the thing to measure first: either the marks are
// telling us something real about the surrounding stream, or one of them is charging for work that
// belongs to its neighbour.
//
// This bench runs `hc_pre` standalone at the decode shape (bs=1, hc=4, d=4096 -> hcd=16384,
// mix_hc=24, iters=20) and splits it into its three launches, so the 37/64 us can be attributed to
// k_mixes vs sinkhorn vs k_combine rather than guessed at. No checkpoint load.
//
//   nvcc -O3 -arch=sm_110a -std=c++17 -I include -o build/hc_glue_bench \
//        tools/hc_glue_bench.cu kernels/hc.cu kernels/hc_sinkhorn.cu kernels/dscratch.cu
//   ./build/hc_glue_bench
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include "hc.h"
#include "dscratch.h"

#define CU(x) do{ cudaError_t e=(x); if(e){ printf("CUDA %s @%d\n", cudaGetErrorString(e), __LINE__); exit(1);} }while(0)

void hc_sinkhorn(float*, float*, float*, const float*, const float*, const float*, int, int, int, float, cudaStream_t);
__global__ void k_mixes(float*, const float*, const float*, int, int, int, float);
__global__ void k_combine(float*, const float*, const float*, int, int, int);

static float time_it(int reps, void (*fn)(void*), void* ctx){
    for (int i = 0; i < 20; ++i) fn(ctx);
    CU(cudaDeviceSynchronize());
    cudaEvent_t a, b; CU(cudaEventCreate(&a)); CU(cudaEventCreate(&b));
    CU(cudaEventRecord(a));
    for (int i = 0; i < reps; ++i) fn(ctx);
    CU(cudaEventRecord(b)); CU(cudaEventSynchronize(b));
    float ms; CU(cudaEventElapsedTime(&ms, a, b));
    return ms / reps * 1000.f;   // us
}

struct Ctx {
    float *y, *post, *comb, *x, *fn_, *scale, *base, *mixes, *pre;
    int bs, hc, d, hcd, mix_hc, iters; float eps;
};
static Ctx C;

static void run_full(void*)   { hc_pre(C.y, C.post, C.comb, C.x, C.fn_, C.scale, C.base,
                                       C.bs, C.hc, C.d, C.iters, C.eps, 0); }
static void run_mixes(void*)  { k_mixes<<<C.bs*C.mix_hc, 256>>>(C.mixes, C.x, C.fn_, C.bs, C.mix_hc, C.hcd, C.eps); }
static void run_sink(void*)   { hc_sinkhorn(C.pre, C.post, C.comb, C.mixes, C.scale, C.base, C.bs, C.hc, C.iters, C.eps, 0); }
static void run_comb(void*)   { k_combine<<<(C.bs*C.d+255)/256, 256>>>(C.y, C.pre, C.x, C.bs, C.hc, C.d); }

// Streaming reference in this binary, so "latency vs bytes" is decided against this box's rate
// today rather than a quoted one.
__global__ void k_stream(const uint4* __restrict__ p, size_t n4, float* sink){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x, st=(size_t)gridDim.x*blockDim.x; unsigned a=0;
    for(; i<n4; i+=st){ uint4 v=p[i]; a^=v.x^v.y^v.z^v.w; }
    if(a==0xdeadbeefu) *sink=1.f;
}

int main(int argc, char** argv){
    const int reps = argc > 1 ? atoi(argv[1]) : 2000;
    arena_init(64ull<<20);

    C.bs = 1; C.hc = 4; C.d = 4096; C.iters = 20; C.eps = 1e-6f;
    C.hcd = C.hc * C.d;                    // 16384
    C.mix_hc = (2 + C.hc) * C.hc;          // 24
    const size_t fn_elems = (size_t)C.mix_hc * C.hcd;      // 24 x 16384 f32 = 1.57 MB

    CU(cudaMalloc(&C.y,     (size_t)C.bs*C.d*4));
    CU(cudaMalloc(&C.post,  (size_t)C.bs*C.hc*4));
    CU(cudaMalloc(&C.comb,  (size_t)C.bs*C.hc*C.hc*4));
    CU(cudaMalloc(&C.x,     (size_t)C.bs*C.hcd*4));
    CU(cudaMalloc(&C.fn_,   fn_elems*4));
    CU(cudaMalloc(&C.scale, 4*4));
    CU(cudaMalloc(&C.base,  (size_t)(2*C.hc + C.hc*C.hc)*4));
    CU(cudaMalloc(&C.mixes, (size_t)C.bs*C.mix_hc*4));
    CU(cudaMalloc(&C.pre,   (size_t)C.bs*C.hc*4));
    { std::vector<float> h(fn_elems, 0.01f);
      CU(cudaMemcpy(C.fn_, h.data(), fn_elems*4, cudaMemcpyHostToDevice));
      std::vector<float> hx((size_t)C.bs*C.hcd, 0.02f);
      CU(cudaMemcpy(C.x, hx.data(), hx.size()*4, cudaMemcpyHostToDevice));
      std::vector<float> hs(4, 1.f);  CU(cudaMemcpy(C.scale, hs.data(), 16, cudaMemcpyHostToDevice));
      std::vector<float> hb(2*C.hc + C.hc*C.hc, 0.f);
      CU(cudaMemcpy(C.base, hb.data(), hb.size()*4, cudaMemcpyHostToDevice)); }

    {   const size_t n4 = (512ull<<20)/16;
        uint4* p; CU(cudaMalloc(&p, n4*16)); CU(cudaMemset(p, 1, n4*16));
        float* sink; CU(cudaMalloc(&sink, 4));
        k_stream<<<320,256>>>(p, n4, sink); CU(cudaDeviceSynchronize());
        cudaEvent_t a,b; CU(cudaEventCreate(&a)); CU(cudaEventCreate(&b)); CU(cudaEventRecord(a));
        for(int i=0;i<20;++i) k_stream<<<320,256>>>(p, n4, sink);
        CU(cudaEventRecord(b)); CU(cudaEventSynchronize(b));
        float ms; CU(cudaEventElapsedTime(&ms,a,b));
        printf("[roofline] %.1f GB/s\n\n", n4*16.0*20/(ms/1e3)/1e9); cudaFree(p); cudaFree(sink); }

    printf("hc_pre at the DECODE shape: bs=1 hc=4 d=4096 hcd=16384 mix_hc=24 iters=20\n");
    printf("  hc_fn = %zu x %d f32 = %.2f MB  ->  %.1f us at roofline\n\n",
           (size_t)C.mix_hc, C.hcd, fn_elems*4/1e6, fn_elems*4/208.7e9*1e6);

    const float t_full  = time_it(reps, run_full,  nullptr);
    const float t_mixes = time_it(reps, run_mixes, nullptr);
    const float t_sink  = time_it(reps, run_sink,  nullptr);
    const float t_comb  = time_it(reps, run_comb,  nullptr);

    printf("  %-28s %8.2f us   %5.1f%% of hc_pre\n", "k_mixes  (24 blk x 256)", t_mixes, 100*t_mixes/t_full);
    printf("  %-28s %8.2f us   %5.1f%%\n",           "hc_sinkhorn", t_sink,  100*t_sink/t_full);
    printf("  %-28s %8.2f us   %5.1f%%\n",           "k_combine (16 blk x 256)", t_comb, 100*t_comb/t_full);
    printf("  %-28s %8.2f us   (sum of parts %.2f, so %.2f us is launch/dmalloc glue)\n",
           "hc_pre TOTAL", t_full, t_mixes+t_sink+t_comb, t_full-(t_mixes+t_sink+t_comb));
    printf("\n  effective rate of k_mixes: %.1f GB/s (%.0f%% of roofline)\n",
           fn_elems*4/(t_mixes/1e6)/1e9, 100*(fn_elems*4/(t_mixes/1e6)/1e9)/208.7);
    printf("  in-situ marks to compare against: hc_pre(attn) 37.0 us, hc_pre(ffn) 64.4 us per layer\n");
    return 0;
}
