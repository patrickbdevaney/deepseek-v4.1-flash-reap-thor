// hc.cu — Hyper-Connections compose, correctness-first (Gate K oracle: ref/gen_units.py gen_hc).
#include "hc.h"
#include <stdint.h>
#include "dscratch.h"
#include "hc_sinkhorn.h"
#include <cstdio>
#define CU(x) do{cudaError_t e=(x); if(e){fprintf(stderr,"cuda %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)

// rsqrt[t] = 1/sqrt(mean(flatten(x[t])^2)+eps). One block per token over hcd = hc*d.
__global__ void k_rsqrt(float* __restrict__ rsq, const float* __restrict__ x, int bs, int hcd, float eps) {
    int t = blockIdx.x; if (t >= bs) return;
    const float* xr = x + (size_t)t * hcd;
    __shared__ float red[256];
    float s = 0.f; for (int j = threadIdx.x; j < hcd; j += blockDim.x) s += xr[j] * xr[j];
    red[threadIdx.x] = s; __syncthreads();
    for (int k = blockDim.x / 2; k > 0; k >>= 1) { if (threadIdx.x < k) red[threadIdx.x] += red[threadIdx.x + k]; __syncthreads(); }
    if (threadIdx.x == 0) rsq[t] = rsqrtf(red[0] / hcd + eps);
}
// mixes[t,m] = (Σ_j x[t,j]*hc_fn[m,j]) * rsq[t]. One BLOCK per (t,m).
//
// LOOP_LOG Finding 23. This was one WARP per (t,m), launched <<<bs*mix_hc, 32>>>. At decode
// (bs=1, mix_hc=24) that is 24 warps — 768 threads — to stream the 1.57 MB `hc_attn_fn`
// [24, 16384] f32, i.e. ~1.2 blocks per SM on a 20-SM device and nowhere near enough
// memory-level parallelism to cover the latency. HC ended up 9.4% of the decode step for 1.2%
// of the bytes (~7.8x off roofline).
//
// Now a full block per (t,m) with a two-stage reduction, and float4 loads (hcd = hc*d = 16384 is
// a multiple of 4 for every configuration this model uses). Same math, same order of accumulation
// within a lane, so it gates cosine-1.0.
//
// LOOP_LOG Finding 36 (Opt #12). `rsq` is no longer a separate kernel. k_rsqrt was launched
// <<<bs, 256>>> — at decode bs=1, i.e. ONE block, one SM, to reduce 64 KB, and every k_mixes block
// then had to wait on it. But k_mixes already streams the whole of `xr`, so it can accumulate
// Σx² from the very same registers for one extra FMA per element and zero extra loads. The
// redundancy across the 24 blocks is free (they all re-read `xr` from L2 regardless); what it buys
// is one less launch and one less cross-block dependency in a phase that is 6.3% of the step and
// almost entirely latency, not bandwidth.
__global__ void k_mixes(float* __restrict__ mixes, const float* __restrict__ x, const float* __restrict__ hc_fn,
                        int bs, int mix_hc, int hcd, float eps) {
    int gid = blockIdx.x; int m = gid % mix_hc, t = gid / mix_hc; if (t >= bs) return;
    const float* xr = x + (size_t)t * hcd; const float* fr = hc_fn + (size_t)m * hcd;
    float sq = 0.f;

    // float4 needs 16-BYTE alignment on BOTH pointers. `hc_fn` is a weight tensor mapped straight
    // out of the safetensors shard, and safetensors only guarantees 8-byte tensor offsets — so a
    // float4 load on it can fault with "misaligned address". This is a real hazard in this
    // codebase (the MoE GEMM carries `off_b = (uintptr_t)wprE & 15` + a funnel shift for exactly
    // the same reason), and the unit gate cannot see it because the gate allocates its own
    // 256-byte-aligned buffers. Checked at runtime; scalar fallback is always correct.
    float acc = 0.f;
    const bool vec4 = ((hcd & 3) == 0)
                   && ((((uintptr_t)xr) & 15) == 0)
                   && ((((uintptr_t)fr) & 15) == 0);
    if (vec4) {
        const float4* x4 = (const float4*)xr; const float4* f4 = (const float4*)fr;
        const int n4 = hcd >> 2;
        for (int j = threadIdx.x; j < n4; j += blockDim.x) {
            float4 a = x4[j], b = f4[j];
            acc += a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w;
            sq  += a.x*a.x + a.y*a.y + a.z*a.z + a.w*a.w;
        }
    } else {
        for (int j = threadIdx.x; j < hcd; j += blockDim.x) { acc += xr[j] * fr[j]; sq += xr[j] * xr[j]; }
    }

    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
    __shared__ float red[32], redq[32];
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) { acc += __shfl_down_sync(0xffffffff, acc, o);
                                       sq  += __shfl_down_sync(0xffffffff, sq,  o); }
    if (lane == 0) { red[w] = acc; redq[w] = sq; }
    __syncthreads();
    if (w == 0) {
        float v = (lane < nw) ? red[lane]  : 0.f;
        float q = (lane < nw) ? redq[lane] : 0.f;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) { v += __shfl_down_sync(0xffffffff, v, o);
                                           q += __shfl_down_sync(0xffffffff, q, o); }
        if (lane == 0) mixes[(size_t)t * mix_hc + m] = v * rsqrtf(q / hcd + eps);
    }
}
// y[t,e] = Σ_j pre[t,j] * x[t, j*d+e]. thread per (t,e).
__global__ void k_combine(float* __restrict__ y, const float* __restrict__ pre, const float* __restrict__ x,
                          int bs, int hc, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= bs * d) return;
    int t = i / d, e = i % d; float acc = 0.f;
    for (int j = 0; j < hc; ++j) acc += pre[(size_t)t * hc + j] * x[((size_t)t * hc + j) * d + e];
    y[i] = acc;
}
// pre[t,j] = sigmoid(mixes[t,j]*scale0 + base[j]) + eps  (hc_head)
__global__ void k_sigmoid_pre(float* __restrict__ pre, const float* __restrict__ mixes,
                              const float* __restrict__ scale, const float* __restrict__ base,
                              int bs, int hc, float eps) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= bs * hc) return;
    int j = i % hc; pre[i] = 1.f / (1.f + expf(-(mixes[i] * scale[0] + base[j]))) + eps;
}
// y[t,j,e] = post[t,j]*x_new[t,e] + Σ_k comb[t,j,k]*residual[t,k,e]. thread per (t,j,e).
__global__ void k_post(float* __restrict__ y, const float* __restrict__ x_new, const float* __restrict__ res,
                       const float* __restrict__ post, const float* __restrict__ comb, int bs, int hc, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= bs * hc * d) return;
    int e = i % d, jt = i / d, j = jt % hc, t = jt / hc;
    float acc = post[(size_t)t * hc + j] * x_new[(size_t)t * d + e];
    // model.py:692 sums comb's FIRST index: y[j,e] = post_j*x_new_e + Σ_i comb[i,j]*residual[i,e]
    for (int i2 = 0; i2 < hc; ++i2) acc += comb[((size_t)t * hc + i2) * hc + j] * res[((size_t)t * hc + i2) * d + e];
    y[i] = acc;
}

void hc_pre(float* y, float* post, float* comb, const float* x, const float* hc_fn,
            const float* hc_scale, const float* hc_base, int bs, int hc, int d,
            int sinkhorn_iters, float eps, cudaStream_t stream) {
    int mix_hc = (2 + hc) * hc, hcd = hc * d;
    float *mixes, *pre;
    mixes=(float*)dmalloc((size_t)bs*mix_hc*4); pre=(float*)dmalloc((size_t)bs*hc*4);
    k_mixes<<<bs * mix_hc, 256, 0, stream>>>(mixes, x, hc_fn, bs, mix_hc, hcd, eps);   // rsqrt fused in (Finding 36)
    hc_sinkhorn(pre, post, comb, mixes, hc_scale, hc_base, bs, hc, sinkhorn_iters, eps, stream);
    k_combine<<<(bs * d + 255) / 256, 256, 0, stream>>>(y, pre, x, bs, hc, d);
    dsync(stream); dfree(mixes); dfree(pre);
}

void hc_post(float* y, const float* x_new, const float* residual, const float* post,
             const float* comb, int bs, int hc, int d, cudaStream_t stream) {
    k_post<<<(bs * hc * d + 255) / 256, 256, 0, stream>>>(y, x_new, residual, post, comb, bs, hc, d);
}

void hc_head(float* y, const float* x, const float* hc_fn, const float* hc_scale,
             const float* hc_base, int bs, int hc, int d, float eps, cudaStream_t stream) {
    int hcd = hc * d;
    float *mixes, *pre;
    mixes=(float*)dmalloc((size_t)bs*hc*4); pre=(float*)dmalloc((size_t)bs*hc*4);
    k_mixes<<<bs * hc, 256, 0, stream>>>(mixes, x, hc_fn, bs, hc, hcd, eps);          // mix_hc = hc for head
    k_sigmoid_pre<<<(bs * hc + 255) / 256, 256, 0, stream>>>(pre, mixes, hc_scale, hc_base, bs, hc, eps);
    k_combine<<<(bs * d + 255) / 256, 256, 0, stream>>>(y, pre, x, bs, hc, d);
    dsync(stream); dfree(mixes); dfree(pre);
}
