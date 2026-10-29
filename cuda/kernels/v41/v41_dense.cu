#include <type_traits>
// v41_dense.cu — dense linears, norms, RoPE, QAT-sim quantisers, hyper-connections, embedding.
// Reference semantics are cited per kernel; see include/v41_kernels.h for the conventions.
#include "v41_kernels.h"
#include "v41_config.h"
#include "v41_gemv.cuh"
#include <cuda_fp4.h>
#include <cstring>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <unordered_map>
#include <cmath>

using namespace v41d;
using namespace v41g;

namespace v41k {

// --------------------------------------------------------------------------- module scratch
// Split-K partials and HC partials. Sized once by v41k_init (outside any graph capture: cudaMalloc
// is illegal while capturing). A call that would need more falls back to no split rather than
// allocating.
static float* g_part = nullptr; static size_t g_part_cap = 0;
// split-K tile arrival counters (k_dense's last CTA per tile reduces); each is back to 0 after its launch
static constexpr int KS_TILES = 1 << 16;
static int* g_kcnt = nullptr;
void v41k_init(size_t part_floats) {
    static const bool kcnt_on = [] { const char* e = getenv("V41_KCNT"); return !(e && *e == '0'); }();
    if (kcnt_on && !g_kcnt && cudaMalloc(&g_kcnt, KS_TILES * sizeof(int)) == cudaSuccess) cudaMemset(g_kcnt, 0, KS_TILES * sizeof(int));
    if (g_part_cap >= part_floats) return;
    if (g_part) cudaFree(g_part);
    if (cudaMalloc(&g_part, part_floats * sizeof(float)) != cudaSuccess) { g_part = nullptr; g_part_cap = 0; return; }
    g_part_cap = part_floats;
}
static int g_nsm = 0;
static int nsm() { if (!g_nsm) { int d; cudaGetDevice(&d); cudaDeviceGetAttribute(&g_nsm, cudaDevAttrMultiProcessorCount, d); } return g_nsm; }

// --------------------------------------------------------------------------- dense GEMV
// grid (ceil(N/64), ceil(M/MT), ksplit). Group g = row / Ng reads x columns [g*K, (g+1)*K).
// RB rows' weights are loaded before any of their dots run, so a warp keeps RB rows of loads in
// flight (ncu: RB=1 sat at 20% of DRAM, 149 cycles/instruction on L1TEX). Each row's sum is unchanged.
//
// LPR = 8 (lanes per row): the warp's 8 rows run as 4 lane groups x 2 rows, a group's 8 lanes striding
// the chunks. For K where K/32 is not a multiple of 32 (wq_b K=1280: 40 chunks, shared/expert down
// K=2304: 72) the 32-lane stride leaves 37.5% / 25% of lanes idle on the last trip; 8 lanes fill it.
// It sums a row in a different order than LPR = 32, so the choice is a function of K alone (never M):
// AR, verify and prefill rows of one weight agree.
template <class Dec, int MT, int RB, int LPR>
__device__ __forceinline__ void dense_lpr8(float* __restrict__ y, const float4* xs4, const Dec& dec, int M, int N,
                                           int row0, int m0, int c_lo, int c_hi, int nch, int ksplit, float out_scale,
                                           int round, float* __restrict__ part) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, gi = lane >> 3, li = lane & 7;
    constexpr int R = ROWS_PER_WARP / 4;                     // rows per lane group
    float acc[R][MT];
    #pragma unroll
    for (int r = 0; r < R; ++r)
        #pragma unroll
        for (int t = 0; t < MT; ++t) acc[r][t] = 0.f;
    #pragma unroll
    for (int r0 = 0; r0 < R; r0 += RB) {
        int rows[RB];
        #pragma unroll
        for (int b = 0; b < RB; ++b) rows[b] = min(row0 + warp * ROWS_PER_WARP + (r0 + b) * 4 + gi, N - 1);
        for (int cb = c_lo; cb < c_hi; cb += 8) {
            const int c = cb + li, cc = min(c, c_hi - 1);
            float w[RB][32];
            #pragma unroll
            for (int b = 0; b < RB; ++b) dec.load(rows[b], cc, w[b]);
            float a[RB][MT];
            #pragma unroll
            for (int b = 0; b < RB; ++b)
                #pragma unroll
                for (int t = 0; t < MT; ++t) a[b][t] = 0.f;
            dot_chunk_rb<MT, RB>(a, w, xs4, cc, nch);
            if (c < c_hi) {
                #pragma unroll
                for (int b = 0; b < RB; ++b)
                    #pragma unroll
                    for (int t = 0; t < MT; ++t) acc[r0 + b][t] += a[b][t];
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < R; ++r)
        #pragma unroll
        for (int t = 0; t < MT; ++t) {
            float v = acc[r][t];
            #pragma unroll
            for (int o = 4; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
            const int row = row0 + warp * ROWS_PER_WARP + r * 4 + gi, m = m0 + t;
            if (li == 0 && row < N && m < M) {
                if (ksplit > 1) part[((size_t)blockIdx.z * M + m) * N + row] = v;
                else { float o = v * out_scale; y[(size_t)m * N + row] = round ? bf16r(o) : o; }
            }
        }
}
// Split-K: the last CTA of a (row tile, m tile) to arrive sums the ksplit partials in z order -- the
// order k_ksum used -- so the reduction needs no second launch and the sums are unchanged.
template <int MT>
__device__ __forceinline__ void ksplit_finish(float* __restrict__ y, const float* part, int* kcnt, int M, int N,
                                              int row0, int m0, int ksplit, float out_scale, int round) {
    __shared__ int last;
    __threadfence();
    __syncthreads();
    int* c = kcnt + blockIdx.y * gridDim.x + blockIdx.x;
    if (threadIdx.x == 0) last = atomicAdd(c, 1) == ksplit - 1;
    __syncthreads();
    if (!last) return;
    __threadfence();
    for (int i = threadIdx.x; i < ROWS_PER_CTA * MT; i += blockDim.x) {
        const int row = row0 + (i % ROWS_PER_CTA), m = m0 + i / ROWS_PER_CTA;
        if (row >= N || m >= M) continue;
        float s = 0.f;
        for (int z = 0; z < ksplit; ++z) s += __ldcg(part + ((size_t)z * M + m) * N + row);
        s *= out_scale; y[(size_t)m * N + row] = round ? bf16r(s) : s;
    }
    if (threadIdx.x == 0) *c = 0;
}
template <class Dec, int MT, int RB = 1, int LPR = 32>
__global__ void __launch_bounds__(256) k_dense(float* __restrict__ y, XSrc X, Dec dec, int M, int N, int K,
                                               int Ng, int ksplit, float out_scale, int round, float* __restrict__ part,
                                               int* __restrict__ kcnt) {
    extern __shared__ float4 xs4[];
    const int row0 = blockIdx.x * ROWS_PER_CTA;
    const int m0 = blockIdx.y * MT;
    const int g = row0 / Ng;
    const int nch = K / 32;
    const int cps = (nch + ksplit - 1) / ksplit;            // chunks per split
    const int c_lo = blockIdx.z * cps, c_hi = min(nch, c_lo + cps);
    for (int t = 0; t < MT; ++t) {
        const int m = min(m0 + t, M - 1);
        stage_row(xs4 + (size_t)t * nch * 8, X, m, g * K, K);
    }
    __syncthreads();
    if constexpr (LPR == 8) {
        dense_lpr8<Dec, MT, RB, LPR>(y, xs4, dec, M, N, row0, m0, c_lo, c_hi, nch, ksplit, out_scale, round, part);
        if (ksplit > 1 && kcnt) ksplit_finish<MT>(y, part, kcnt, M, N, row0, m0, ksplit, out_scale, round);
        return;
    }
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    float acc[ROWS_PER_WARP][MT];
    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r)
        #pragma unroll
        for (int t = 0; t < MT; ++t) acc[r][t] = 0.f;
    #pragma unroll
    for (int r0 = 0; r0 < ROWS_PER_WARP; r0 += RB) {
        int rows[RB];
        #pragma unroll
        for (int b = 0; b < RB; ++b) rows[b] = min(row0 + warp * ROWS_PER_WARP + r0 + b, N - 1);
        for (int cb = c_lo; cb < c_hi; cb += 32) {           // every lane takes every trip (shuffle-safe)
            const int c = cb + lane, cc = min(c, c_hi - 1);
            float w[RB][32];
            #pragma unroll
            for (int b = 0; b < RB; ++b) dec.load(rows[b], cc, w[b]);
            float a[RB][MT];
            #pragma unroll
            for (int b = 0; b < RB; ++b)
                #pragma unroll
                for (int t = 0; t < MT; ++t) a[b][t] = 0.f;
            dot_chunk_rb<MT, RB>(a, w, xs4, cc, nch);
            if (c < c_hi) {
                #pragma unroll
                for (int b = 0; b < RB; ++b)
                    #pragma unroll
                    for (int t = 0; t < MT; ++t) acc[r0 + b][t] += a[b][t];
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r)
        #pragma unroll
        for (int t = 0; t < MT; ++t) {
            const float v = warp_sum(acc[r][t]);
            const int row = row0 + warp * ROWS_PER_WARP + r, m = m0 + t;
            if (lane == 0 && row < N && m < M) {
                if (ksplit > 1) part[((size_t)blockIdx.z * M + m) * N + row] = v;
                else { float o = v * out_scale; y[(size_t)m * N + row] = round ? bf16r(o) : o; }
            }
        }
    if (ksplit > 1 && kcnt) ksplit_finish<MT>(y, part, kcnt, M, N, row0, m0, ksplit, out_scale, round);
}
__global__ void k_ksum(float* y, const float* part, int ks, size_t MN, float out_scale, int round) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i >= MN) return;
    float s = 0.f; for (int z = 0; z < ks; ++z) s += part[(size_t)z * MN + i];   // fixed order
    s *= out_scale; y[i] = round ? bf16r(s) : s;
}

template <class Dec>
static void launch_dense(float* y, const XSrc& X, const Dec& dec, int M, int N, int K, int G, bool round,
                         float out_scale, cudaStream_t st) {
    const int Ng = N / G;
    const int nch = K / 32;
    // MT = 8 lets a speculative verify (M = K+1 <= 8) read every weight once instead of twice
    // MT = M up to 6 (a padded tile row is a whole wasted dot product per weight)
    int mt = (M <= 6) ? M : 8;
    while (mt > 1 && (size_t)mt * K * sizeof(float) > 160 * 1024) mt >>= 1;   // smem cap (K=8192 stays at 4)
    const int ntile = (N + ROWS_PER_CTA - 1) / ROWS_PER_CTA, mtile = (M + mt - 1) / mt;
    // Split-K (and the lane mapping below, which follows the split) changes a row's summation order, so
    // both are chosen from the weight's shape alone, never from M: decode, a speculative verify and a
    // prefill chunk then sum every row identically, and decode == prefill holds bit for bit. When the
    // split depended on M, prefill rows summed in a different order, and the extra rounding noise
    // flipped MoE routing often enough to show GROSS consistency gaps (up to 1.47 logits).
    // V41_KSPLIT_LEGACY=1 restores the old M-dependent prefill split.
    static const bool legacy = [] { const char* e = getenv("V41_KSPLIT_LEGACY"); return e && *e == '1'; }();
    constexpr int SPEC_M = 8;
    int ks = 1;
    static const int want_x = [] { const char* e = getenv("V41_KSPLIT_WANT"); return e ? atoi(e) : 2; }();
    const int ctas = ntile * (legacy && M > SPEC_M ? (M + 3) / 4 : 1), want = want_x * nsm();
    if (ctas < want) {
        ks = std::min((want + ctas - 1) / ctas, std::max(1, nch / 8));
        if ((size_t)ks * std::max(M, SPEC_M) * N > g_part_cap) {
            static bool warned = false;          // rows would then sum differently from decode: say so once
            if (!warned) { fprintf(stderr, "[k_dense] split-K partials M=%d N=%d ks=%d exceed v41k_init capacity: unsplit\n", M, N, ks); warned = true; }
            ks = 1;
        }
    }
    dim3 grid(ntile, mtile, ks);
    const size_t smem = (size_t)mt * K * sizeof(float);
    // RB per decoder from bench_dense: NVFP4 gains ~10% at 2; bf16/FP8 lose at 2 (V41_DENSE_RB overrides)
    static const int rb_env = [] { const char* e = getenv("V41_DENSE_RB"); return e ? atoi(e) : 0; }();
    // MT > 1 shares each x read across RB rows (V41_DENSE_RBM overrides)
    static const int rbm_env = [] { const char* e = getenv("V41_DENSE_RBM"); return e ? atoi(e) : 0; }();
    const int rb = mt > 1 ? (rbm_env ? rbm_env : (mt <= 3 ? 4 : 2)) : rb_env ? rb_env : (std::is_same<Dec, DecNVFP4>::value ? 2 : 1);
    static const bool check_launch = [] { const char* e = getenv("V41_CHECK_LAUNCH"); return e && *e == '1'; }();
    static const bool ks_fuse_env = [] { const char* e = getenv("V41_KSUM_FUSE"); return !(e && *e == '0'); }();
    const bool fuse = ks > 1 && ks_fuse_env && g_kcnt && (size_t)ntile * mtile <= (size_t)KS_TILES;
    // 8 lanes per row where the 32-lane stride wastes > 10% of the last trip (a function of K only)
    static const int lpr_env = [] { const char* e = getenv("V41_DENSE_LPR"); return e ? atoi(e) : -1; }();
    const int cpr = ks > 1 ? (nch + ks - 1) / ks : nch;     // chunks a row's split covers
    const bool l8 = lpr_env >= 0 ? lpr_env == 8 && cpr % 8 == 0
                                 : (cpr % 8 == 0 && (double)cpr / (32.0 * ((cpr + 31) / 32)) < 0.9);
    #define V41_LAUNCH(MTV) do { auto kf = l8 ? ((rb >= 2) ? k_dense<Dec, MTV, 2, 8> : k_dense<Dec, MTV, 1, 8>) : \
        (rb >= 4 && MTV <= 4) ? k_dense<Dec, MTV, 4> : rb >= 2 ? k_dense<Dec, MTV, 2> : k_dense<Dec, MTV, 1>; \
        /* opt in above 48 KB minus the kernel's own static smem (ksplit_finish's flag): at exactly 48 KB */ \
        /* dynamic (wo_a K=4096 at MT=3) the launch otherwise fails silently and y keeps stale values */ \
        if (smem + 1024 > 48 * 1024) cudaFuncSetAttribute(kf, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem); \
        kf<<<grid, 256, smem, st>>>(y, X, dec, M, N, K, Ng, ks, out_scale, round ? 1 : 0, g_part, fuse ? g_kcnt : nullptr); \
        if (check_launch) { const cudaError_t e_ = cudaPeekAtLastError(); if (e_ != cudaSuccess) { \
            fprintf(stderr, "[k_dense] launch M=%d N=%d K=%d mt=%d smem=%zu: %s\n", M, N, K, MTV, smem, cudaGetErrorString(e_)); abort(); } } } while (0)
    switch (mt) { case 1: V41_LAUNCH(1); break; case 2: V41_LAUNCH(2); break; case 3: V41_LAUNCH(3); break;
                  case 4: V41_LAUNCH(4); break; case 5: V41_LAUNCH(5); break; case 6: V41_LAUNCH(6); break; default: V41_LAUNCH(8); }
    #undef V41_LAUNCH
    if (ks > 1) {
        size_t MN = (size_t)M * N;
        if (!fuse) k_ksum<<<(unsigned)((MN + 255) / 256), 256, 0, st>>>(y, g_part, ks, MN, out_scale, round ? 1 : 0);
    }
}

// NVFP4 dense overlay registry: a weight pointer registered here is NVFP4 codes (Ws = its e4m3/16
// scales), not FP8, and fp8_linear serves it with DecNVFP4. Filled at load time (bind_layer).
struct OvEntry { float g; };
static std::unordered_map<const void*, OvEntry>& ov_map() { static std::unordered_map<const void*, OvEntry> m; return m; }
void overlay_register(const uint8_t* codes, float g) { ov_map()[codes] = OvEntry{g}; }
size_t overlay_count() { return ov_map().size(); }
void fp8_linear(float* y, const uint8_t* xq, const float* xs, const uint8_t* W, const uint8_t* Ws,
                int M, int N, int K, bool round_bf16, cudaStream_t st, int G) {
    XSrc X; X.xq = xq; X.xs = xs; X.ldx = K * G;
    if (!ov_map().empty()) {
        auto it = ov_map().find(W);
        if (it != ov_map().end()) {
            DecNVFP4 d{W, Ws, it->second.g, K};
            launch_dense(y, X, d, M, N, K, G, round_bf16, 1.0f, st);
            return;
        }
    }
    DecFP8 d{W, Ws, K};
    launch_dense(y, X, d, M, N, K, G, round_bf16, 1.0f, st);
}
// bf16_permute: in-place reorder of each row's 2 KB chunk groups for DecBF16P, then registered so
// bf16_linear serves the pointer with it. K must be a multiple of 1024 (one group = 32 chunks of 32).
__global__ void k_bf16_permute(uint4* W, int K) {
    __shared__ uint4 t[128];
    uint4* g = W + (size_t)blockIdx.x * (K / 8) + (size_t)blockIdx.y * 128;
    t[threadIdx.x] = g[threadIdx.x];
    __syncthreads();
    const int n = threadIdx.x;                    // new slot h*32+lane <- old piece 4*lane+h
    g[n] = t[4 * (n & 31) + (n >> 5)];
}
static std::unordered_map<const void*, int>& perm_map() { static std::unordered_map<const void*, int> m; return m; }
bool bf16_permuted(const void* W) { return perm_map().count(W) > 0; }
bool bf16_permute(uint16_t* W, int N, int K, cudaStream_t st) {
    if (K % 1024 || perm_map().count(W)) return false;
    k_bf16_permute<<<dim3(N, K / 1024), 128, 0, st>>>(reinterpret_cast<uint4*>(W), K);
    perm_map()[W] = K;
    return true;
}
void bf16_linear(float* y, const float* x, const uint16_t* W, int M, int N, int K, bool round_bf16,
                 cudaStream_t st, float out_scale) {
    XSrc X; X.xf = x; X.ldx = K;
    if (!perm_map().empty() && perm_map().count(W)) {
        DecBF16P d{W, K};
        launch_dense(y, X, d, M, N, K, 1, round_bf16, out_scale, st);
        return;
    }
    DecBF16 d{W, K};
    launch_dense(y, X, d, M, N, K, 1, round_bf16, out_scale, st);
}

// --------------------------------------------------------------------------- quantisers
// act_quant(x, 32, "ue8m0"): one warp per 32-group.
__global__ void k_act_quant(uint8_t* q, float* s, const float* x, size_t ngroups) {
    size_t gi = (blockIdx.x * (size_t)blockDim.x + threadIdx.x) >> 5; if (gi >= ngroups) return;
    const int lane = threadIdx.x & 31;
    const float v = x[gi * 32 + lane];
    const float amax = fmaxf(warp_max(fabsf(v)), 1e-4f);
    const float sc = pow2_ceil(amax / 448.f);
    q[gi * 32 + lane] = f2e4m3(fminf(fmaxf(v / sc, -448.f), 448.f));
    if (lane == 0) s[gi] = sc;
}
void act_quant_fp8(uint8_t* q, float* s, const float* x, int M, int K, cudaStream_t st) {
    size_t ng = (size_t)M * K / 32;
    k_act_quant<<<(unsigned)((ng * 32 + 255) / 256), 256, 0, st>>>(q, s, x, ng);
}
__global__ void k_fp8_sim(float* x, int dim, size_t ngroups) {
    size_t gi = (blockIdx.x * (size_t)blockDim.x + threadIdx.x) >> 5; if (gi >= ngroups) return;
    const int lane = threadIdx.x & 31;
    float* p = x + gi * 32 + lane;
    const float v = *p;
    const float amax = fmaxf(warp_max(fabsf(v)), 1e-4f);
    const float sc = pow2_ceil(amax / 448.f);
    *p = bf16r(e4m3f(f2e4m3(fminf(fmaxf(v / sc, -448.f), 448.f))) * sc);
}
void fp8_sim(float* x, int rows, int dim, cudaStream_t st) {
    size_t ng = (size_t)rows * dim / 32;
    k_fp8_sim<<<(unsigned)((ng * 32 + 255) / 256), 256, 0, st>>>(x, dim, ng);
}
// fp4_act_quant(x, grp, inplace=True, scale_dtype): one warp covers 32 elements = 32/grp groups.
__global__ void k_fp4_sim(float* x, size_t n, int grp, int e4m3s, uint8_t* packed, uint8_t* scales) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i >= n) return;   // n % 32 == 0
    const int lane = threadIdx.x & 31;
    const float v = x[i];
    float amax = fabsf(v);
    for (int o = grp / 2; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    float sc; uint8_t sbyte;
    if (e4m3s) { amax = fmaxf(amax, 6.f * 0.001953125f); sbyte = f2e4m3(amax / 6.f); sc = e4m3f(sbyte); }
    else       { amax = fmaxf(amax, 6.f * 1.17549435e-38f); sc = pow2_ceil(amax / 6.f); sbyte = pow2_ceil_e8m0(amax / 6.f); }
    const uint8_t code = fp4_code(fminf(fmaxf(v / sc, -6.f), 6.f));
    x[i] = bf16r(fp4f(code) * sc);
    if (packed) {
        const uint32_t other = __shfl_xor_sync(0xffffffffu, (uint32_t)code, 1);
        if (!(lane & 1)) packed[i >> 1] = (uint8_t)(code | (other << 4));
        if ((i % grp) == 0) scales[i / grp] = sbyte;
    }
}
void fp4_sim(float* x, int rows, int dim, int grp, bool e4m3_scale, uint8_t* packed, uint8_t* scales, cudaStream_t st) {
    size_t n = (size_t)rows * dim;
    k_fp4_sim<<<(unsigned)((n + 255) / 256), 256, 0, st>>>(x, n, grp, e4m3_scale ? 1 : 0, packed, scales);
}
__global__ void k_round(float* x, size_t n) { size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i < n) x[i] = bf16r(x[i]); }
void round_bf16(float* x, size_t n, cudaStream_t st) { k_round<<<(unsigned)((n + 255) / 256), 256, 0, st>>>(x, n); }
__global__ void k_scale_bf16(float* x, size_t n, float c) { size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i < n) x[i] = bf16r(x[i] * c); }
void scale_bf16(float* x, size_t n, float c, cudaStream_t st) { k_scale_bf16<<<(unsigned)((n + 255) / 256), 256, 0, st>>>(x, n, c); }
__global__ void k_argmax(int* out, const float* x, int n) {
    float bv = -INFINITY; int bi = 0x7fffffff;
    for (int i = threadIdx.x; i < n; i += blockDim.x) { const float v = x[i]; if (v > bv || (v == bv && i < bi)) { bv = v; bi = i; } }
    __shared__ float sv[1024]; __shared__ int si[1024];
    sv[threadIdx.x] = bv; si[threadIdx.x] = bi; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) { const float v = sv[threadIdx.x + s]; const int j = si[threadIdx.x + s];
            if (v > sv[threadIdx.x] || (v == sv[threadIdx.x] && j < si[threadIdx.x])) { sv[threadIdx.x] = v; si[threadIdx.x] = j; } }
        __syncthreads();
    }
    if (threadIdx.x == 0) *out = si[0];
}
void argmax(int* out, const float* x, int n, cudaStream_t st) { k_argmax<<<1, 1024, 0, st>>>(out, x, n); }
__global__ void k_add_bf16(float* y, const float* a, size_t n) { size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i < n) y[i] = bf16r(y[i] + a[i]); }
void add_bf16(float* y, const float* a, size_t n, cudaStream_t st) { k_add_bf16<<<(unsigned)((n + 255) / 256), 256, 0, st>>>(y, a, n); }

// --------------------------------------------------------------------------- RMSNorm
// RMSNorm.forward: x.float(); x * rsqrt(mean(x^2) + eps); (weight * x).to(bf16)
__global__ void k_rmsnorm(float* y, const float* x, const uint16_t* w, int dim, float eps) {
    const float* xr = x + (size_t)blockIdx.x * dim; float* yr = y + (size_t)blockIdx.x * dim;
    __shared__ float red[32];
    // all loads issued before the first fma (the plain loop was latency-bound: 8.7 us for one 5120 row);
    // same per-thread order, so bit-identical. Rows longer than RN strides take the loop.
    constexpr int RN = 16;
    float v[RN];
    const bool regs = dim <= RN * (int)blockDim.x;
    float ss = 0.f;
    if (regs) {
        #pragma unroll
        for (int j = 0; j < RN; ++j) { const int i = threadIdx.x + j * blockDim.x; v[j] = i < dim ? xr[i] : 0.f; }
        #pragma unroll
        for (int j = 0; j < RN; ++j) if (threadIdx.x + j * blockDim.x < dim) ss = fmaf(v[j], v[j], ss);
    } else {
        for (int i = threadIdx.x; i < dim; i += blockDim.x) ss = fmaf(xr[i], xr[i], ss);
    }
    ss = warp_sum(ss);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = threadIdx.x < (blockDim.x >> 5) ? red[threadIdx.x] : 0.f;
        v = warp_sum(v); if (threadIdx.x == 0) red[0] = v;
    }
    __syncthreads();
    const float r = rsqrtf(red[0] / dim + eps);
    if (regs) {
        #pragma unroll
        for (int j = 0; j < RN; ++j) {
            const int i = threadIdx.x + j * blockDim.x;
            if (i < dim) { const float n = v[j] * r; yr[i] = bf16r(w ? bf16f(w[i]) * n : n); }
        }
        return;
    }
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        const float n = xr[i] * r;
        yr[i] = bf16r(w ? bf16f(w[i]) * n : n);
    }
}
void rmsnorm(float* y, const float* x, const uint16_t* w, int rows, int dim, float eps, cudaStream_t st) {
    k_rmsnorm<<<rows, dim >= 1024 ? 512 : 128, 0, st>>>(y, x, w, dim, eps);
}

// --------------------------------------------------------------------------- RoPE
// precompute_freqs_cis: freqs in fp32 exactly as the reference builds them (fp32 pow, fp32 YaRN
// blend), angle = fp32(pos * f) -- a single fp32 multiply, the same rounding torch.outer does --
// then accurate sincosf. Tables at 1M positions would cost 512 MB, so angles are formed on the fly.
__constant__ float c_freq[2][32];     // [0] base 10000 no YaRN ; [1] compress theta 160000 + YaRN
static bool g_freq_init = false;
static void init_freqs() {
    if (g_freq_init) return;
    float f[2][32];
    for (int k = 0; k < 2; ++k) {
        const float base = k ? v41::COMPRESS_ROPE_THETA : v41::ROPE_THETA;
        const int dim = v41::ROPE_DIM;
        for (int j = 0; j < 32; ++j) f[k][j] = 1.0f / powf(base, (float)(2 * j) / (float)dim);
        if (k) {
            const double orig = v41::YARN_ORIG_MAXPOS;
            auto corr = [&](double rot) { return dim * std::log(orig / (rot * 2 * M_PI)) / (2 * std::log((double)base)); };
            const int low = std::max((int)std::floor(corr(v41::YARN_BETA_FAST)), 0);
            const int high = std::min((int)std::ceil(corr(v41::YARN_BETA_SLOW)), dim - 1);
            for (int j = 0; j < 32; ++j) {
                float ramp = ((float)j - (float)low) / std::max((float)(high - low), 1e-3f);
                ramp = ramp < 0.f ? 0.f : (ramp > 1.f ? 1.f : ramp);
                const float smooth = 1.f - ramp;
                f[k][j] = f[k][j] / v41::YARN_FACTOR * (1.f - smooth) + f[k][j] * smooth;
            }
        }
    }
    cudaMemcpyToSymbol(c_freq, f, sizeof(f));
    g_freq_init = true;
}
void rope_set_tables() { init_freqs(); }   // call once outside capture (cudaMemcpyToSymbol)
__global__ void k_rope(float* x, int rows, int heads, int row_stride, int off, const int* d_pos, int pos_delta,
                       int inverse, int which) {
    const int r = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5); if (r >= rows) return;
    const int j = threadIdx.x & 31;                       // one complex pair per lane (64 dims)
    const int pos = d_pos[r / heads] + pos_delta;
    float s, c; sincosf((float)pos * c_freq[which][j], &s, &c);
    if (inverse) s = -s;
    float* p = x + (size_t)r * row_stride + off + 2 * j;
    const float re = p[0], im = p[1];
    p[0] = bf16r(re * c - im * s);
    p[1] = bf16r(re * s + im * c);
}
void rope(float* x, int rows, int heads, int row_stride, int off, const int* d_pos, int pos_delta,
          bool inverse, bool yarn, cudaStream_t st) {
    init_freqs();
    const int which = yarn ? 1 : 0;
    k_rope<<<(rows + 7) / 8, 256, 0, st>>>(x, rows, heads, row_stride, off, d_pos, pos_delta, inverse ? 1 : 0, which);
}

// --------------------------------------------------------------------------- hyper-connections
// hc_mixes: grid (M, 24 mixes + 1 sumsq, KS) partial dots over 20480 -> finalize (fixed order) + Sinkhorn.
static constexpr int HC_KS = 8;
// grid (M, HC_JG, KS): a CTA computes HC_JPG of the 25 dots over its z span, reading its x slice once
// (it was read once per mix). Each dot keeps the original per-thread order and block reduction, so
// the partials are bit-identical to one CTA per mix.
static constexpr int HC_JPG = 5, HC_JG = 25 / HC_JPG;
__global__ void __launch_bounds__(256) k_hc_partial(float* part, const float* x, const float* fn) {
    const int m = blockIdx.x, jg = blockIdx.y, z = blockIdx.z;
    constexpr int span = v41::HC_DIM / HC_KS, PT = span / 256;
    static_assert(span % 256 == 0, "x slice per thread");
    const float* xr = x + (size_t)m * v41::HC_DIM + z * span;
    float xv[PT];
    #pragma unroll
    for (int k = 0; k < PT; ++k) xv[k] = xr[threadIdx.x + k * 256];
    __shared__ float red[HC_JPG][8];
    #pragma unroll
    for (int jj = 0; jj < HC_JPG; ++jj) {
        const int j = jg * HC_JPG + jj;                 // j == 24 -> sum of squares
        const float* fr = fn + (size_t)(j < 24 ? j : 0) * v41::HC_DIM + z * span;
        float acc = 0.f;
        #pragma unroll
        for (int k = 0; k < PT; ++k) acc = fmaf(xv[k], j < 24 ? fr[threadIdx.x + k * 256] : xv[k], acc);
        acc = warp_sum(acc);
        if ((threadIdx.x & 31) == 0) red[jj][threadIdx.x >> 5] = acc;
    }
    __syncthreads();
    if (threadIdx.x < HC_JPG) {
        float s = 0.f; for (int w = 0; w < 8; ++w) s += red[threadIdx.x][w];
        part[((size_t)m * 25 + jg * HC_JPG + threadIdx.x) * HC_KS + z] = s;
    }
}
// One warp per token. Bit-identical to the single-thread version it replaced (which ran 24 reductions
// and 20 Sinkhorn iterations serially, 80x per token): lane j < 24 reduces mix j in the same z order;
// lane t < 16 owns c[t/4][t%4]; every row/column sum gathers its 4 terms by shuffle and adds them in
// the original index order, so every rounding matches.
__global__ void k_hc_finish(float* pre, float* post, float* comb, const float* part, const float* scale,
                            const float* base) {
    const int m = blockIdx.x, lane = threadIdx.x;
    float ss = 0.f; for (int z = 0; z < HC_KS; ++z) ss += part[((size_t)m * 25 + 24) * HC_KS + z];
    const float rs = rsqrtf(ss / v41::HC_DIM + v41::NORM_EPS);
    float mixv = 0.f;
    if (lane < 24) { float d = 0.f; for (int z = 0; z < HC_KS; ++z) d += part[((size_t)m * 25 + lane) * HC_KS + z]; mixv = d * rs; }
    const float eps = v41::HC_EPS;
    if (lane < 4) pre[m * 4 + lane] = 1.f / (1.f + expf(-(mixv * scale[0] + base[lane]))) + eps;
    else if (lane < 8) post[m * 4 + lane - 4] = 2.f / (1.f + expf(-(mixv * scale[1] + base[lane])));
    const int t = lane & 15, i = t >> 2, k = t & 3;
    const float mt = __shfl_sync(0xffffffffu, mixv, 8 + t);
    float c = mt * scale[2] + base[8 + t];
    auto row4 = [&](float v, int r, float (&o)[4]) { for (int q = 0; q < 4; ++q) o[q] = __shfl_sync(0xffffffffu, v, 4 * r + q); };
    auto col4 = [&](float v, int cl, float (&o)[4]) { for (int q = 0; q < 4; ++q) o[q] = __shfl_sync(0xffffffffu, v, 4 * q + cl); };
    float g[4];
    row4(c, i, g);
    float mx = -INFINITY; for (int q = 0; q < 4; ++q) mx = fmaxf(mx, g[q]);
    c = expf(c - mx);
    row4(c, i, g);
    { float sm = 0.f; for (int q = 0; q < 4; ++q) sm += g[q]; c = c / sm + eps; }
    auto colnorm = [&]() { col4(c, k, g); float sm = 0.f; for (int q = 0; q < 4; ++q) sm += g[q]; c /= (sm + eps); };
    auto rownorm = [&]() { row4(c, i, g); float sm = 0.f; for (int q = 0; q < 4; ++q) sm += g[q]; c /= (sm + eps); };
    colnorm();
    for (int it = 0; it < v41::HC_SINKHORN_ITERS - 1; ++it) { rownorm(); colnorm(); }
    if (lane < 16) comb[m * 16 + t] = c;
}
static float* g_hcpart = nullptr; static int g_hcpart_m = 0;
void hc_mixes(float* pre, float* post, float* comb, const float* x, const float* fn, const float* scale,
              const float* base, int M, cudaStream_t st) {
    if (M > g_hcpart_m) { fprintf(stderr, "hc_mixes: M=%d exceeds reserved %d (call hc_reserve)\n", M, g_hcpart_m); abort(); }
    k_hc_partial<<<dim3(M, HC_JG, HC_KS), 256, 0, st>>>(g_hcpart, x, fn);
    k_hc_finish<<<M, 32, 0, st>>>(pre, post, comb, g_hcpart, scale, base);
}
void hc_reserve(int max_m) {
    if (max_m <= g_hcpart_m) return;
    if (g_hcpart) cudaFree(g_hcpart);
    cudaMalloc(&g_hcpart, (size_t)max_m * 25 * HC_KS * sizeof(float)); g_hcpart_m = max_m;
}
__global__ void k_hc_collapse(float* y, const float* x, const float* pre, int M) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i >= (size_t)M * v41::DIM) return;
    const int m = (int)(i / v41::DIM), d = (int)(i % v41::DIM);
    const float* xr = x + (size_t)m * v41::HC_DIM + d;
    float s = 0.f;
    for (int j = 0; j < 4; ++j) s += pre[m * 4 + j] * xr[(size_t)j * v41::DIM];
    y[i] = bf16r(s);
}
void hc_collapse(float* y, const float* x, const float* pre, int M, cudaStream_t st) {
    size_t n = (size_t)M * v41::DIM;
    k_hc_collapse<<<(unsigned)((n + 255) / 256), 256, 0, st>>>(y, x, pre, M);
}
// y[j] = post[j]*f + sum_i comb[i][j]*res[i]  (the reference sums over the FIRST comb index)
__global__ void k_hc_post(float* y, const float* f, const float* res, const float* post, const float* comb, int M) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i >= (size_t)M * v41::DIM) return;
    const int m = (int)(i / v41::DIM), d = (int)(i % v41::DIM);
    const float fv = f[i];
    const float* rr = res + (size_t)m * v41::HC_DIM + d;
    float r[4]; for (int k = 0; k < 4; ++k) r[k] = rr[(size_t)k * v41::DIM];
    for (int j = 0; j < 4; ++j) {
        float s = 0.f;
        for (int k = 0; k < 4; ++k) s += comb[m * 16 + k * 4 + j] * r[k];
        y[(size_t)m * v41::HC_DIM + (size_t)j * v41::DIM + d] = bf16r(post[m * 4 + j] * fv + s);
    }
}
void hc_post(float* y, const float* f, const float* res, const float* post, const float* comb, int M, cudaStream_t st) {
    size_t n = (size_t)M * v41::DIM;
    k_hc_post<<<(unsigned)((n + 255) / 256), 256, 0, st>>>(y, f, res, post, comb, M);
}
__global__ void k_hc_ident(float* pre, int M) { int i = threadIdx.x + blockIdx.x * blockDim.x; if (i < M * 4) pre[i] = (i % 4 == 0) ? 1.f : 0.f; }
void hc_identity_pre(float* pre, int M, cudaStream_t st) { k_hc_ident<<<(M * 4 + 127) / 128, 128, 0, st>>>(pre, M); }
__global__ void k_embed(float* h, const uint16_t* emb, const int* ids, int M) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i >= (size_t)M * v41::DIM) return;
    const int m = (int)(i / v41::DIM), d = (int)(i % v41::DIM);
    const float v = bf16f(emb[(size_t)ids[m] * v41::DIM + d]);
    for (int j = 0; j < 4; ++j) h[(size_t)m * v41::HC_DIM + (size_t)j * v41::DIM + d] = v;
}
void embed_expand(float* h, const uint16_t* emb, const int* d_ids, int M, cudaStream_t st) {
    size_t n = (size_t)M * v41::DIM;
    k_embed<<<(unsigned)((n + 255) / 256), 256, 0, st>>>(h, emb, d_ids, M);
}

// --------------------------------------------------------------------------- SwiGLU
// Expert.forward: gate = w1(x).float(); up = w3(x).float(); up clamp [-L,L]; gate clamp max L;
// silu(gate)*up (*weights) -> .to(bf16)
__global__ void k_swiglu(float* act, const float* gate, const float* up, const float* wr, int M, int inter, float L,
                         int ld_in) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i >= (size_t)M * inter) return;
    const int m = (int)(i / inter), k = (int)(i % inter);
    float g = gate[(size_t)m * ld_in + k], u = up[(size_t)m * ld_in + k];
    if (L > 0.f) { u = fminf(fmaxf(u, -L), L); g = fminf(g, L); }
    float a = g / (1.f + expf(-g)) * u;
    if (wr) a *= wr[m];
    act[i] = bf16r(a);
}
void swiglu(float* act, const float* gate, const float* up, const float* w_row, int M, int inter, float limit, int ld,
            cudaStream_t st) {
    size_t n = (size_t)M * inter;
    k_swiglu<<<(unsigned)((n + 255) / 256), 256, 0, st>>>(act, gate, up, w_row, M, inter, limit, ld);
}

} // namespace v41k

namespace v41k {
// --------------------------------------------------------------------------- NVFP4 copy of a bf16 matrix
// For the drafter's head only (drafts change acceptance, never the emitted ids). DecNVFP4 layout: codes
// row-major K/2 bytes (low nibble first), E4M3 scale per 16, one global g = amax / (448 * 6).
// `permuted`: src was reordered in place by bf16_permute (DecBF16P); logical piece p of a 1024-wide
// group (uint4 of 8 values) sits at slot (p % 4) * 32 + p / 4.
__device__ __forceinline__ float bf16_at(const uint16_t* src, int K, int row, int k, bool permuted) {
    if (!permuted) return bf16f(src[(size_t)row * K + k]);
    const int grp = k >> 10, p = (k & 1023) >> 3, slot = (p & 3) * 32 + (p >> 2);
    return bf16f(src[(size_t)row * K + grp * 1024 + slot * 8 + (k & 7)]);
}
__global__ void k_absmax_bf16(unsigned* out, const uint16_t* src, size_t n) {
    float m = 0.f;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        m = fmaxf(m, fabsf(bf16f(src[i])));
    m = warp_max(m);
    if ((threadIdx.x & 31) == 0) atomicMax(out, __float_as_uint(m));    // non-negative floats order as uints
}
__global__ void k_nvfp4_quant(uint8_t* W, uint8_t* S, const uint16_t* src, int N, int K, const unsigned* amax, bool permuted) {
    const size_t gi = blockIdx.x * (size_t)blockDim.x + threadIdx.x;     // one thread per 16-group
    if (gi >= (size_t)N * (K / 16)) return;
    const int row = (int)(gi / (K / 16)), k0 = (int)(gi % (K / 16)) * 16;
    const float g = fmaxf(__uint_as_float(*amax), 1e-30f) / (448.f * 6.f);
    float v[16], m = 0.f;
    #pragma unroll
    for (int i = 0; i < 16; ++i) { v[i] = bf16_at(src, K, row, k0 + i, permuted); m = fmaxf(m, fabsf(v[i])); }
    const uint8_t sc = f2e4m3(m / 6.f / g);
    const float s = e4m3f(sc) * g, inv = s > 0.f ? 1.f / s : 0.f;
    #pragma unroll
    for (int i = 0; i < 16; i += 2)
        W[(size_t)row * (K / 2) + (k0 + i) / 2] = (uint8_t)__nv_cvt_float2_to_fp4x2(make_float2(v[i] * inv, v[i + 1] * inv), __NV_E2M1, cudaRoundNearest);
    S[gi] = sc;
}
float nvfp4_from_bf16(uint8_t* W, uint8_t* S, const uint16_t* src, int N, int K, bool permuted, cudaStream_t st) {
    unsigned* d; cudaMalloc(&d, 4); cudaMemsetAsync(d, 0, 4, st);
    k_absmax_bf16<<<512, 256, 0, st>>>(d, src, (size_t)N * K);
    const size_t ng = (size_t)N * (K / 16);
    k_nvfp4_quant<<<(unsigned)((ng + 255) / 256), 256, 0, st>>>(W, S, src, N, K, d, permuted);
    unsigned h; cudaMemcpyAsync(&h, d, 4, cudaMemcpyDeviceToHost, st); cudaStreamSynchronize(st); cudaFree(d);
    float a; std::memcpy(&a, &h, 4);
    return std::max(a, 1e-30f) / (448.f * 6.f);
}
void nvfp4_linear_f32(float* y, const float* x, const uint8_t* W, const uint8_t* S, float g, int M, int N, int K, cudaStream_t st) {
    XSrc X; X.xf = x; X.ldx = K;
    DecNVFP4 d{W, S, g, K};
    launch_dense(y, X, d, M, N, K, 1, false, 1.0f, st);
}
} // namespace v41k
