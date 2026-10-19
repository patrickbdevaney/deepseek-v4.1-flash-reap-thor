// attention.cu — single-sequence causal SDPA (fp32) with GQA + optional sliding window.
// Correctness-first: one block per (query position i, query head h). Threads cooperate to
// compute the dot-product scores, a two-pass softmax in shared memory, then the weighted V sum.
#include "attention.h"
#include <cfloat>
#include <cstdio>
#include <cstdlib>

#define ATT_THREADS 256

// grid = (n_heads, seq), block = ATT_THREADS
// dynamic shared layout: [ Qs(head_dim) | scores(seq) ]
__global__ void sdpa_kernel(float* out, const float* Q, const float* K, const float* V,
                            int seq, int n_heads, int n_kv, int head_dim,
                            int sliding_window, float scaling) {
    const int h = blockIdx.x;          // query head
    const int i = blockIdx.y;          // query position
    const int tid = threadIdx.x;
    const int group = n_heads / n_kv;  // query heads per kv head
    const int kv = h / group;          // shared kv head for this query head

    extern __shared__ float smem[];
    float* Qs     = smem;              // head_dim
    float* scores = smem + head_dim;   // seq (only [lo..i] used)
    __shared__ float red[ATT_THREADS];
    __shared__ float m_sh, l_sh;

    // lowest valid key position (causal, plus sliding window if enabled)
    int lo = 0;
    if (sliding_window > 0) {
        lo = i - sliding_window + 1;
        if (lo < 0) lo = 0;
    }

    // load this query vector into shared memory
    const float* q = Q + ((size_t)i * n_heads + h) * head_dim;
    for (int d = tid; d < head_dim; d += blockDim.x) Qs[d] = q[d];
    __syncthreads();

    // pass 0: scores[j] = scaling * dot(Q[i,h,:], K[j,kv,:]) for valid j
    for (int j = lo + tid; j <= i; j += blockDim.x) {
        const float* k = K + ((size_t)j * n_kv + kv) * head_dim;
        float dot = 0.f;
        for (int d = 0; d < head_dim; ++d) dot += Qs[d] * k[d];
        scores[j] = dot * scaling;
    }
    __syncthreads();

    // pass 1a: row max over valid j
    float lmax = -FLT_MAX;
    for (int j = lo + tid; j <= i; j += blockDim.x) lmax = fmaxf(lmax, scores[j]);
    red[tid] = lmax; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]);
        __syncthreads();
    }
    if (tid == 0) m_sh = red[0];
    __syncthreads();
    const float m = m_sh;

    // pass 1b: sum of exp(score - m)
    float lsum = 0.f;
    for (int j = lo + tid; j <= i; j += blockDim.x) lsum += expf(scores[j] - m);
    red[tid] = lsum; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    if (tid == 0) l_sh = red[0];
    __syncthreads();
    const float inv_l = 1.f / l_sh;

    // normalize scores into softmax weights
    for (int j = lo + tid; j <= i; j += blockDim.x) scores[j] = expf(scores[j] - m) * inv_l;
    __syncthreads();

    // pass 2: out[i,h,d] = sum_j weight[j] * V[j,kv,d]  (parallel over d, no atomics)
    float* o = out + ((size_t)i * n_heads + h) * head_dim;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.f;
        for (int j = lo; j <= i; ++j) acc += scores[j] * V[((size_t)j * n_kv + kv) * head_dim + d];
        o[d] = acc;
    }
}

// THE THIRD DYNAMIC-SHARED-MEMORY LAUNCH SIZED BY CONTEXT, closed here for the same reason as the
// four top-k scans (DECODE_LADDER item 1.4, include/indexer.h). `shmem` grows with `seq`, so past
// seq = 12,224 this asked for more than this device's 49,152 B per-block default; the launch would
// fail, the following synchronize would return success, and `out` would keep whatever it held. A
// full enumeration of every `<<<g,b,smem,s>>>` in kernels/ and src/ found exactly three families
// whose dynamic request is a function of a RUNTIME quantity -- the six top-k sites, this one, and
// three (moe router, act_quant) whose sizes are model or block constants and are bounded at 4 KiB.
// This one is not linked into build/decode or build/dsv4-server today (only tests/test_attention.cu
// calls it), which is exactly why it would have sat here: an unreachable path is not a fixed one,
// it is an unexercised one, and `sdpa` is the reference implementation a future dense path would
// reach for. Twelve lines now beats rediscovering §17 of wiki/measurement-and-traps.md later.
void sdpa(float* out, const float* Q, const float* K, const float* V,
          int seq, int n_heads, int n_kv, int head_dim, int sliding_window, float scaling, cudaStream_t s) {
    dim3 grid(n_heads, seq);
    size_t shmem = (size_t)(head_dim + seq) * sizeof(float);
    // THE DEVICE OPT-IN MAXIMUM IS NOT THIS KERNEL'S MAXIMUM, and that cost a leg to find out.
    // `cudaDevAttrMaxSharedMemoryPerBlockOptin` is 232,448 B here, and asking for exactly that on
    // `sdpa_kernel` returns `invalid argument` -- because this kernel also carries 1,040 B of STATIC
    // shared (`red[ATT_THREADS]`, `m_sh`, `l_sh`), and the settable dynamic size is the opt-in
    // maximum less that. Bisecting `cudaFuncSetAttribute` over a pair of kernels differing only in
    // static shared confirms the rule and its exact form: static 0 -> 232,448 settable, static 1,024
    // -> 231,424 settable, both launching successfully at exactly that size.
    // `cudaDevAttrReservedSharedMemoryPerBlock` is NOT a second deduction. The four top-k scan
    // kernels in include/indexer.h declare no static shared at all, which is the only reason setting
    // the raw device maximum works there.
    static int lim_default = 0, lim_max = 0;
    static size_t opted = 0;
    if (!lim_default) {
        int dev = 0, optin = 0; cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&lim_default, cudaDevAttrMaxSharedMemoryPerBlock,      dev);
        cudaDeviceGetAttribute(&optin,       cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
        if (!lim_default) lim_default = 48 * 1024;
        if (optin < lim_default) optin = lim_default;
        cudaFuncAttributes fa{}; cudaFuncGetAttributes(&fa, (const void*)sdpa_kernel);
        lim_max = optin - (int)fa.sharedSizeBytes;
        if (lim_max < lim_default) lim_max = lim_default;
    }
    if (shmem > (size_t)lim_default) {
        if (shmem > (size_t)lim_max) {
            fprintf(stderr, "[sdpa] FATAL seq=%d head_dim=%d needs %zu B of dynamic shared memory; "
                    "the most this kernel can be granted on this device is %d B (seq <= %d at "
                    "head_dim %d). Refusing to issue a launch that would fail and leave `out` "
                    "untouched.\n", seq, head_dim, shmem, lim_max, lim_max / 4 - head_dim, head_dim);
            fflush(stderr); abort();
        }
        if (shmem > opted) {
            cudaError_t e = cudaFuncSetAttribute((const void*)sdpa_kernel,
                                                 cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem);
            if (e != cudaSuccess) {
                fprintf(stderr, "[sdpa] FATAL cudaFuncSetAttribute(MaxDynamicSharedMemorySize, %zu) "
                        "failed: %s (kernel ceiling computed as %d B)\n",
                        shmem, cudaGetErrorString(e), lim_max);
                fflush(stderr); abort();
            }
            opted = shmem;
            fprintf(stderr, "[sdpa] opted in to %zu B dynamic shared memory at seq=%d "
                    "(default limit %d B, this kernel's ceiling %d B)\n",
                    shmem, seq, lim_default, lim_max);
            fflush(stderr);
        }
    }
    sdpa_kernel<<<grid, ATT_THREADS, shmem, s>>>(out, Q, K, V, seq, n_heads, n_kv,
                                                 head_dim, sliding_window, scaling);
    // A shared-memory over-request fails at LAUNCH CONFIGURATION time, on the host, before anything
    // is enqueued -- so a stream synchronize has nothing to report and returns success. Only
    // cudaGetLastError sees it. See wiki/measurement-and-traps.md §17.
    cudaError_t le = cudaGetLastError();
    if (le != cudaSuccess) {
        fprintf(stderr, "[sdpa] FATAL launch failed (seq=%d, %zu B dynamic shared): %s. `out` was "
                "NOT written.\n", seq, shmem, cudaGetErrorString(le));
        fflush(stderr); abort();
    }
}
