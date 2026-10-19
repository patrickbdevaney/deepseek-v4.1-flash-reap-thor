// hc_sinkhorn.cu — HC split + Sinkhorn, correctness-first (Gate K oracle: ref/gen_units.py).
// One block per token; a single thread does the tiny hc x hc (=4x4) Sinkhorn. hc is small (4), iters=20;
// this is latency-trivial per token. Faithful to kernel.py:371-438 op order (row-softmax+eps, col-norm,
// then iters-1 of {row-norm, col-norm}). Vectorize/warp-per-token later, AFTER the gate passes.
#include <cstdlib>
#include "hc_sinkhorn.h"

#define HCMAX 8   // max hc (config uses 4)

__global__ void hc_sinkhorn_kernel(float* __restrict__ pre, float* __restrict__ post,
                                   float* __restrict__ comb,
                                   const float* __restrict__ mixes, const float* __restrict__ hc_scale,
                                   const float* __restrict__ hc_base, int n, int hc, int iters, float eps) {
    int i = blockIdx.x;
    if (i >= n || threadIdx.x != 0) return;
    int mh = (2 + hc) * hc;
    const float* mx = mixes + (size_t)i * mh;
    float s0 = hc_scale[0], s1 = hc_scale[1], s2 = hc_scale[2];

    // pre / post
    for (int j = 0; j < hc; ++j) {
        pre[(size_t)i * hc + j]  = 1.f / (1.f + expf(-(mx[j]      * s0 + hc_base[j])))       + eps;
        post[(size_t)i * hc + j] = 2.f / (1.f + expf(-(mx[hc + j] * s1 + hc_base[hc + j])));
    }
    // comb[hc,hc] = mixes[2hc:] * s2 + base
    float c[HCMAX * HCMAX];
    for (int j = 0; j < hc; ++j)
        for (int k = 0; k < hc; ++k)
            c[j * hc + k] = mx[2 * hc + j * hc + k] * s2 + hc_base[2 * hc + j * hc + k];

    // comb = softmax(-1) + eps   (row-wise)
    for (int j = 0; j < hc; ++j) {
        float mmax = -1e30f; for (int k = 0; k < hc; ++k) mmax = fmaxf(mmax, c[j*hc+k]);
        float sum = 0.f; for (int k = 0; k < hc; ++k) { c[j*hc+k] = expf(c[j*hc+k]-mmax); sum += c[j*hc+k]; }
        for (int k = 0; k < hc; ++k) c[j*hc+k] = c[j*hc+k] / sum + eps;
    }
    // comb = comb / (comb.sum(-2)+eps)   (col-normalize)
    for (int k = 0; k < hc; ++k) {
        float cs = 0.f; for (int j = 0; j < hc; ++j) cs += c[j*hc+k];
        cs += eps; for (int j = 0; j < hc; ++j) c[j*hc+k] /= cs;
    }
    // iters-1 Sinkhorn passes: row-normalize then col-normalize
    for (int it = 0; it < iters - 1; ++it) {
        for (int j = 0; j < hc; ++j) { float rs=0.f; for (int k=0;k<hc;++k) rs+=c[j*hc+k]; rs+=eps; for (int k=0;k<hc;++k) c[j*hc+k]/=rs; }
        for (int k = 0; k < hc; ++k) { float cs=0.f; for (int j=0;j<hc;++j) cs+=c[j*hc+k]; cs+=eps; for (int j=0;j<hc;++j) c[j*hc+k]/=cs; }
    }
    for (int j = 0; j < hc; ++j)
        for (int k = 0; k < hc; ++k)
            comb[((size_t)i * hc + j) * hc + k] = c[j * hc + k];
}

// ---------------------------------------------------------------------------------------------
// WARP-PARALLEL Sinkhorn (LOOP_LOG Finding 24).
//
// The scalar kernel above returns on every lane but threadIdx.x==0, so ONE thread performed all
// 20 iterations, and its `float c[HCMAX*HCMAX]` is indexed with runtime (j,k) — which nvcc cannot
// keep in registers, so it lands in LOCAL memory (DRAM-backed). ~640 dependent local round-trips
// per token made a 4x4 normalisation cost 86 us: 76% of hc_pre and 7.4 ms of the 115.8 ms decode
// step, for essentially no arithmetic.
//
// Here one warp owns one token and lane L holds comb[j,k] with j = L/hc, k = L%hc — entirely in
// registers. Row sums reduce over the low log2(hc) lane bits, column sums over the next log2(hc),
// via __shfl_xor_sync. Because those XOR masks never cross a group boundary, the idle lanes
// (L >= hc*hc) cannot contaminate a reduction.
//
// Requires hc a power of two with hc*hc <= 32 (config: hc=4 -> 16 lanes). Anything else falls back
// to the scalar kernel, so behaviour is unchanged for configurations this path does not cover.
__global__ void hc_sinkhorn_warp_kernel(float* __restrict__ pre, float* __restrict__ post,
                                        float* __restrict__ comb,
                                        const float* __restrict__ mixes, const float* __restrict__ hc_scale,
                                        const float* __restrict__ hc_base, int n, int hc, int iters, float eps) {
    const int i = blockIdx.x; if (i >= n) return;
    const int lane = threadIdx.x, hh = hc * hc, mh = (2 + hc) * hc;
    const float* mx = mixes + (size_t)i * mh;
    const float s0 = hc_scale[0], s1 = hc_scale[1], s2 = hc_scale[2];

    if (lane < hc) {
        pre [(size_t)i * hc + lane] = 1.f / (1.f + expf(-(mx[lane]      * s0 + hc_base[lane])))      + eps;
        post[(size_t)i * hc + lane] = 2.f / (1.f + expf(-(mx[hc + lane] * s1 + hc_base[hc + lane])));
    }

    const bool act = lane < hh;
    const float v = act ? (mx[2 * hc + lane] * s2 + hc_base[2 * hc + lane]) : 0.f;

    // row-wise softmax over k (the low log2(hc) lane bits), then +eps
    float m = act ? v : -1e30f;
    for (int b = 1; b < hc; b <<= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, b));
    float e = act ? expf(v - m) : 0.f;
    float rs = e;
    for (int b = 1; b < hc; b <<= 1) rs += __shfl_xor_sync(0xffffffff, rs, b);
    float c = act ? (e / rs + eps) : 0.f;

    // one column-normalise, then (iters-1) x (row-normalise, column-normalise) — matches the scalar path
    for (int it = 0; it < iters; ++it) {
        if (it > 0) {
            float r = c;
            for (int b = 1; b < hc; b <<= 1) r += __shfl_xor_sync(0xffffffff, r, b);
            r += eps; c = act ? c / r : 0.f;
        }
        float cs = c;
        for (int b = hc; b < hh; b <<= 1) cs += __shfl_xor_sync(0xffffffff, cs, b);
        cs += eps; c = act ? c / cs : 0.f;
    }
    if (act) comb[(size_t)i * hh + lane] = c;
}

void hc_sinkhorn(float* pre, float* post, float* comb,
                 const float* mixes, const float* hc_scale, const float* hc_base,
                 int n, int hc, int iters, float eps, cudaStream_t stream) {
    const bool pow2 = hc > 0 && (hc & (hc - 1)) == 0;
    if (pow2 && hc * hc <= 32 && getenv("HC_SCALAR") == nullptr)
        hc_sinkhorn_warp_kernel<<<n, 32, 0, stream>>>(pre, post, comb, mixes, hc_scale, hc_base, n, hc, iters, eps);
    else
        hc_sinkhorn_kernel<<<n, 32, 0, stream>>>(pre, post, comb, mixes, hc_scale, hc_base, n, hc, iters, eps);
}
