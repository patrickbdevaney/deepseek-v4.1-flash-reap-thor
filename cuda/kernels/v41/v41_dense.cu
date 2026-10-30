#include <type_traits>
// v41_dense.cu — dense linears, norms, RoPE, QAT-sim quantisers, hyper-connections, embedding.
// Reference semantics are cited per kernel; see include/v41_kernels.h for the conventions.
#include "v41_kernels.h"
#include "v41_config.h"
#include "v41_gemv.cuh"
#include "v41_nv_mma.cuh"
#include "v41_f8_mma.cuh"
#include <cuda_fp4.h>
#include <cstring>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <climits>
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
// Streams that run k_dense concurrently with the main stream get their own partials and counters
// (sharing g_part/g_kcnt across concurrent launches would mix their split-K sums).
struct DenseScr { cudaStream_t st; float* part; size_t cap; int* kcnt; };
static DenseScr g_scr[4]; static int g_nscr = 0;
void v41k_stream_scratch(cudaStream_t st) {
    for (int i = 0; i < g_nscr; ++i) if (g_scr[i].st == st) return;
    if (g_nscr == 4) { fprintf(stderr, "v41k_stream_scratch: too many streams\n"); abort(); }
    DenseScr d{st, nullptr, 0, nullptr};
    if (g_part_cap && cudaMalloc(&d.part, g_part_cap * sizeof(float)) != cudaSuccess) { fprintf(stderr, "v41k_stream_scratch: alloc\n"); abort(); }
    d.cap = g_part_cap;
    if (g_kcnt && cudaMalloc(&d.kcnt, KS_TILES * sizeof(int)) == cudaSuccess) cudaMemset(d.kcnt, 0, KS_TILES * sizeof(int));
    g_scr[g_nscr++] = d;
}
static void dense_scratch(cudaStream_t st, float*& part, size_t& cap, int*& kcnt) {
    part = g_part; cap = g_part_cap; kcnt = g_kcnt;
    for (int i = 0; i < g_nscr; ++i) if (g_scr[i].st == st) { part = g_scr[i].part; cap = g_scr[i].cap; kcnt = g_scr[i].kcnt; }
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
    float* part; size_t part_cap; int* kcnt;
    dense_scratch(st, part, part_cap, kcnt);
    int ks = 1;
    static const int want_x = [] { const char* e = getenv("V41_KSPLIT_WANT"); return e ? atoi(e) : 2; }();
    const int ctas = ntile * (legacy && M > SPEC_M ? (M + 3) / 4 : 1), want = want_x * nsm();
    if (ctas < want) {
        ks = std::min((want + ctas - 1) / ctas, std::max(1, nch / 8));
        if ((size_t)ks * std::max(M, SPEC_M) * N > part_cap) {
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
    const bool fuse = ks > 1 && ks_fuse_env && kcnt && (size_t)ntile * mtile <= (size_t)KS_TILES;
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
        kf<<<grid, 256, smem, st>>>(y, X, dec, M, N, K, Ng, ks, out_scale, round ? 1 : 0, part, fuse ? kcnt : nullptr); \
        if (check_launch) { const cudaError_t e_ = cudaPeekAtLastError(); if (e_ != cudaSuccess) { \
            fprintf(stderr, "[k_dense] launch M=%d N=%d K=%d mt=%d smem=%zu: %s\n", M, N, K, MTV, smem, cudaGetErrorString(e_)); abort(); } } } while (0)
    switch (mt) { case 1: V41_LAUNCH(1); break; case 2: V41_LAUNCH(2); break; case 3: V41_LAUNCH(3); break;
                  case 4: V41_LAUNCH(4); break; case 5: V41_LAUNCH(5); break; case 6: V41_LAUNCH(6); break; default: V41_LAUNCH(8); }
    #undef V41_LAUNCH
    if (ks > 1) {
        size_t MN = (size_t)M * N;
        if (!fuse) k_ksum<<<(unsigned)((MN + 255) / 256), 256, 0, st>>>(y, part, ks, MN, out_scale, round ? 1 : 0);
    }
}

// NVFP4 dense overlay registry: a weight pointer registered here is NVFP4 codes (Ws = its e4m3/16
// scales), not FP8, and fp8_linear serves it with DecNVFP4. Filled at load time (bind_layer).
struct OvEntry { float g; bool mma; };
static std::unordered_map<const void*, OvEntry>& ov_map() { static std::unordered_map<const void*, OvEntry> m; return m; }
// V41_NV_MMA=0 keeps the row-major layout and k_dense<DecNVFP4>; otherwise shapes that tile (N % 16, K % 256)
// are repacked in place for k_nvm (v41_nv_mma.cuh). Load time only (cudaMalloc of a temporary).
static bool nv_mma_on() { static const bool on = [] { const char* e = getenv("V41_NV_MMA"); return !(e && *e == '0'); }(); return on; }
void overlay_register(const uint8_t* codes, const uint8_t* scales, float g, int N, int K, int force_mma) {
    const bool mma = (force_mma >= 0 ? force_mma == 1 : nv_mma_on()) && N % 16 == 0 && K % 256 == 0;
    if (mma) {
        const size_t cb = (size_t)N * K / 2, sbytes = (size_t)N * K / 16;
        uint8_t* tmp = nullptr;
        if (cudaMalloc(&tmp, cb) != cudaSuccess) { fprintf(stderr, "overlay_register: tmp alloc\n"); abort(); }
        const size_t th = (size_t)(N / 16) * (K / 256) * 32;
        v41g::k_nv_mma_repack_codes<<<(unsigned)((th + 255) / 256), 256>>>(tmp, codes, N, K);
        cudaMemcpy((void*)codes, tmp, cb, cudaMemcpyDeviceToDevice);
        const size_t ts = (size_t)(N / 16) * (K / 256) * 16;
        v41g::k_nv_mma_repack_scales<<<(unsigned)((ts + 255) / 256), 256>>>(tmp, scales, N, K);
        cudaMemcpy((void*)scales, tmp, sbytes, cudaMemcpyDeviceToDevice);
        if (cudaDeviceSynchronize() != cudaSuccess) { fprintf(stderr, "overlay_register: repack failed\n"); abort(); }
        cudaFree(tmp);
    }
    ov_map()[codes] = OvEntry{g, mma};
}
static void launch_nvm(float* y, const XSrc& X, const uint8_t* W, const uint8_t* S, float g, int M, int N, int K, int G,
                       bool round, cudaStream_t st) {
    const int Ng = N / G;
    if (G > 1 && Ng % v41g::NVM_ROWS_PER_CTA) { fprintf(stderr, "k_nvm: group of %d rows straddles CTAs\n", Ng); abort(); }
    // M > 8 (prefill): 16 or 32 tokens per CTA (k_nvm_big), bit-identical rows; V41_NVM_BIG=0 keeps 8
    static const int big_env = [] { const char* e = getenv("V41_NVM_BIG"); return e ? atoi(e) : 32; }();
    const int mt = M == 1 ? 1 : M == 2 ? 2 : M <= 4 ? 4 : (M <= 8 || big_env < 16) ? 8 : (M <= 16 || big_env < 32) ? 16 : 32;
    const int ntile = (N + v41g::NVM_ROWS_PER_CTA - 1) / v41g::NVM_ROWS_PER_CTA, nsb = K / 256;
    // K split from the shape alone (decode == verify == prefill): >= 3 CTAs per SM, >= 4 superblocks each
    static const int want_x = [] { const char* e = getenv("V41_NVM_WANT"); return e ? atoi(e) : 3; }();
    int ks = 1;
    if (ntile < want_x * nsm()) ks = std::min((want_x * nsm() + ntile - 1) / ntile, std::max(1, nsb / 4));
    float* part; size_t cap; int* kcnt; dense_scratch(st, part, cap, kcnt);
    if (ks > 1 && (size_t)ks * std::max(M, 8) * N > cap) {
        fprintf(stderr, "[k_nvm] split-K partials M=%d N=%d ks=%d exceed capacity\n", M, N, ks); abort();
    }
    const int nk_max = ((nsb + ks - 1) / ks) * 256;
    const size_t smem = (size_t)mt * ((v41g::nvm_stage_bytes(nk_max) + 15) & ~(size_t)15);
    dim3 grid(ntile, (M + mt - 1) / mt, ks);
    static const bool ks_fuse_env = [] { const char* e = getenv("V41_KSUM_FUSE"); return !(e && *e == '0'); }();
    const bool fuse = ks > 1 && ks_fuse_env && kcnt && (size_t)grid.x * grid.y <= (size_t)KS_TILES;
    #define V41_NVM(MTV) do { auto kf = v41g::k_nvm<MTV>; \
        if (smem + 1024 > 48 * 1024) cudaFuncSetAttribute(kf, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem); \
        kf<<<grid, 256, smem, st>>>(y, X, W, S, g, M, N, K, Ng, ks, round ? 1 : 0, part, fuse ? kcnt : nullptr); } while (0)
    #define V41_NVMB(NTV) do { auto kf = v41g::k_nvm_big<NTV>; \
        if (smem + 1024 > 48 * 1024) cudaFuncSetAttribute(kf, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem); \
        kf<<<grid, 256, smem, st>>>(y, X, W, S, g, M, N, K, Ng, ks, round ? 1 : 0, part, fuse ? kcnt : nullptr); } while (0)
    // A10: decode / verify (mt <= 8) on k_nvt, the TMA-fed form (bit-identical). Copy shape from the
    // nvm_bench sweep (2026-10-30): split shapes (ks >= 4, little work per CTA) S2 C1; >= 12 MiB of codes
    // S2 C2; the rest stay on k_nvm. V41_NVT=0 = k_nvm everywhere; V41_PDL=0 = plain launches.
    static const int nvt_env = [] { const char* e = getenv("V41_NVT"); return e ? atoi(e) : 1; }();
    static const bool pdl_env = [] { const char* e = getenv("V41_PDL"); return !(e && *e == '0'); }();
    const size_t codes = (size_t)N * K / 2;
    const int nvt_cfg = nvt_env == 0 || mt > 8 ? 0 : ks >= 4 ? 1 : codes >= (12u << 20) ? 2 : nvt_env == 2 ? 1 : 0;
    if (nvt_cfg) {
        const int ns = 2, ch = nvt_cfg;
        const size_t smem_t = v41g::nvt_ring_bytes(ns, ch) + smem;
        cudaLaunchConfig_t cfg = {};
        cfg.gridDim = grid; cfg.blockDim = dim3(256); cfg.dynamicSmemBytes = smem_t; cfg.stream = st;
        cudaLaunchAttribute at[1];
        at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; at[0].val.programmaticStreamSerializationAllowed = 1;
        cfg.attrs = at; cfg.numAttrs = pdl_env ? 1 : 0;
        int* kc = fuse ? kcnt : nullptr; const int rnd = round ? 1 : 0;
        #define V41_NVT(MTV, CHV) do { auto kf = v41g::k_nvt<MTV, 2, CHV>;             static bool attr = false; if (!attr) { cudaFuncSetAttribute(kf, cudaFuncAttributeMaxDynamicSharedMemorySize, 200 * 1024); attr = true; }             cudaLaunchKernelEx(&cfg, kf, y, X, W, S, g, M, N, K, Ng, ks, rnd, part, kc); } while (0)
        if (ch == 1) { switch (mt) { case 1: V41_NVT(1, 1); break; case 2: V41_NVT(2, 1); break; case 4: V41_NVT(4, 1); break; default: V41_NVT(8, 1); } }
        else { switch (mt) { case 1: V41_NVT(1, 2); break; case 2: V41_NVT(2, 2); break; case 4: V41_NVT(4, 2); break; default: V41_NVT(8, 2); } }
        #undef V41_NVT
    } else
    switch (mt) { case 1: V41_NVM(1); break; case 2: V41_NVM(2); break; case 4: V41_NVM(4); break; case 8: V41_NVM(8); break;
                  case 16: V41_NVMB(2); break; default: V41_NVMB(4); }
    #undef V41_NVM
    #undef V41_NVMB
    if (ks > 1 && !fuse) {
        const size_t MN = (size_t)M * N;
        k_ksum<<<(unsigned)((MN + 255) / 256), 256, 0, st>>>(y, part, ks, MN, 1.0f, round ? 1 : 0);
    }
}
size_t overlay_count() { return ov_map().size(); }

// P6 FP8 tensor-core weights: a pointer registered here was repacked in place for k_f8m (v41_f8_mma.cuh)
static std::unordered_map<const void*, int>& f8m_map() { static std::unordered_map<const void*, int> m; return m; }
bool f8m_register(uint8_t* W, int N, int K) {
    if (N % 32 || K % 256 || f8m_map().count(W)) return false;
    const size_t bytes = (size_t)N * K;
    uint8_t* tmp = nullptr;
    if (cudaMalloc(&tmp, bytes) != cudaSuccess) { fprintf(stderr, "f8m_register: tmp alloc\n"); abort(); }
    const size_t th = (size_t)(N / 16) * (K / 256) * 1024;
    v41g::k_f8m_repack<<<(unsigned)((th + 255) / 256), 256>>>(tmp, W, N, K);
    cudaMemcpy(W, tmp, bytes, cudaMemcpyDeviceToDevice);
    if (cudaDeviceSynchronize() != cudaSuccess) { fprintf(stderr, "f8m_register: repack failed\n"); abort(); }
    cudaFree(tmp);
    f8m_map()[W] = K;
    return true;
}
static void launch_f8m(float* y, const XSrc& X, const uint8_t* W, const uint8_t* Ws, int M, int N, int K, bool round,
                       cudaStream_t st) {
    static const int nt_max = [] { const char* e = getenv("V41_F8M_NT"); return e ? atoi(e) : 8; }();
    const int nt = M <= 8 ? 1 : M <= 16 ? 2 : (M <= 32 || nt_max < 8) ? 4 : 8;
    const size_t SB = (v41g::nvm_stage_bytes(v41g::F8M_KC) + 15) & ~(size_t)15, smem = (size_t)8 * nt * SB;
    dim3 grid((N + v41g::F8M_ROWS_PER_CTA - 1) / v41g::F8M_ROWS_PER_CTA, (M + 8 * nt - 1) / (8 * nt));
    #define V41_F8M(NTV) do { auto kf = v41g::k_f8m<NTV>; \
        if (smem + 1024 > 48 * 1024) cudaFuncSetAttribute(kf, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem); \
        kf<<<grid, 256, smem, st>>>(y, X, W, Ws, M, N, K, round ? 1 : 0); } while (0)
    switch (nt) { case 1: V41_F8M(1); break; case 2: V41_F8M(2); break; case 4: V41_F8M(4); break; default: V41_F8M(8); }
    #undef V41_F8M
}
void fp8_linear(float* y, const uint8_t* xq, const float* xs, const uint8_t* W, const uint8_t* Ws,
                int M, int N, int K, bool round_bf16, cudaStream_t st, int G) {
    XSrc X; X.xq = xq; X.xs = xs; X.ldx = K * G;
    if (G == 1 && !f8m_map().empty() && f8m_map().count(W)) { launch_f8m(y, X, W, Ws, M, N, K, round_bf16, st); return; }
    if (!ov_map().empty()) {
        auto it = ov_map().find(W);
        if (it != ov_map().end()) {
            // P8: prefill rows on tcgen05 (V41_TC5=1, T1); decode / verify stay on k_nvm
            if (it->second.mma && M >= 64 && tc5_on() && tc5_nvfp4(y, xq, xs, K * G, W, Ws, it->second.g, M, N, K, G, round_bf16, st))
                return;
            if (it->second.mma) { launch_nvm(y, X, W, Ws, it->second.g, M, N, K, G, round_bf16, st); return; }
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
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");   // A10: a PDL-launched GEMV may start its weight stream
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
    // the weights too: loaded after the reduction they were a second dependent DRAM round trip
    float v[RN]; uint16_t wv[RN];
    const bool regs = dim <= RN * (int)blockDim.x;
    float ss = 0.f;
    if (regs) {
        #pragma unroll
        for (int j = 0; j < RN; ++j) {
            const int i = threadIdx.x + j * blockDim.x;
            v[j] = i < dim ? xr[i] : 0.f; wv[j] = (w && i < dim) ? w[i] : (uint16_t)0;
        }
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
            if (i < dim) { const float n = v[j] * r; yr[i] = bf16r(w ? bf16f(wv[j]) * n : n); }
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
// hc_collapse -> rmsnorm(w) -> act_quant_fp8 in one launch, bit for bit the three kernels: one CTA of 512 per
// row (k_rmsnorm's block for DIM, so the sum of squares reduces in the same order); thread t owns elements
// t + 512 j, so at each j warp w holds exactly the 32-group 16 j + w that k_act_quant gives one warp.
// Three dependent launches with a DRAM round trip each were ~13 us per sub-layer on the critical path.
__global__ void __launch_bounds__(512) k_cnq(float* y, uint8_t* q, float* qs, const float* x, const float* pre,
                                             const uint16_t* w, float eps) {
    constexpr int D = v41::DIM, NJ = D / 512;
    static_assert(D % 512 == 0, "k_cnq: DIM % 512");
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");   // A9: the router / a GEMV may start its weight stream
    const int m = blockIdx.x, t = threadIdx.x;
    const float* xr = x + (size_t)m * v41::HC_DIM;
    __shared__ float red[32];
    float p[4];
    #pragma unroll
    for (int j = 0; j < 4; ++j) p[j] = pre[m * 4 + j];
    float v[NJ]; uint16_t wv[NJ];
    #pragma unroll
    for (int j = 0; j < NJ; ++j) {
        const int i = t + 512 * j;
        float s = 0.f;
        #pragma unroll
        for (int k = 0; k < 4; ++k) s += p[k] * xr[(size_t)k * D + i];
        v[j] = bf16r(s); wv[j] = w[i];
    }
    float ss = 0.f;
    #pragma unroll
    for (int j = 0; j < NJ; ++j) ss = fmaf(v[j], v[j], ss);
    ss = warp_sum(ss);
    if ((t & 31) == 0) red[t >> 5] = ss;
    __syncthreads();
    if (t < 32) {
        float u = t < 16 ? red[t] : 0.f;
        u = warp_sum(u); if (t == 0) red[0] = u;
    }
    __syncthreads();
    const float r = rsqrtf(red[0] / D + eps);
    float* yr = y + (size_t)m * D;
    #pragma unroll
    for (int j = 0; j < NJ; ++j) {
        const int i = t + 512 * j;
        const float o = bf16r(bf16f(wv[j]) * (v[j] * r));
        yr[i] = o;
        const float amax = fmaxf(warp_max(fabsf(o)), 1e-4f);
        const float sc = pow2_ceil(amax / 448.f);
        q[(size_t)m * D + i] = f2e4m3(fminf(fmaxf(o / sc, -448.f), 448.f));
        if ((t & 31) == 0) qs[(size_t)m * (D / 32) + 16 * j + (t >> 5)] = sc;
    }
}
void collapse_norm_quant(float* y, uint8_t* q, float* qs, const float* x, const float* pre, const uint16_t* w, int M,
                         float eps, cudaStream_t st) {
    k_cnq<<<M, 512, 0, st>>>(y, q, qs, x, pre, w, eps);
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
// NVFP4 copy of an FP8 [N, K] matrix (e8m0 per 32x32), same RTN scheme: the drafter's linears (draft only)
__device__ __forceinline__ float fp8_at(const uint8_t* W, const uint8_t* Ws, int K, size_t i) {
    const size_t row = i / K, k = i % K;
    return e4m3f(W[i]) * e8m0f(Ws[(row >> 5) * (size_t)(K >> 5) + (k >> 5)]);
}
__global__ void k_absmax_fp8(unsigned* out, const uint8_t* W, const uint8_t* Ws, int K, size_t n) {
    float m = 0.f;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        m = fmaxf(m, fabsf(fp8_at(W, Ws, K, i)));
    m = warp_max(m);
    if ((threadIdx.x & 31) == 0) atomicMax(out, __float_as_uint(m));
}
__global__ void k_nvfp4_quant_fp8(uint8_t* W, uint8_t* S, const uint8_t* src, const uint8_t* srcs, int N, int K, const unsigned* amax) {
    const size_t gi = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (gi >= (size_t)N * (K / 16)) return;
    const int row = (int)(gi / (K / 16)), k0 = (int)(gi % (K / 16)) * 16;
    const float g = fmaxf(__uint_as_float(*amax), 1e-30f) / (448.f * 6.f);
    float v[16], m = 0.f;
    #pragma unroll
    for (int i = 0; i < 16; ++i) { v[i] = fp8_at(src, srcs, K, (size_t)row * K + k0 + i); m = fmaxf(m, fabsf(v[i])); }
    const uint8_t sc = f2e4m3(m / 6.f / g);
    const float s = e4m3f(sc) * g, inv = s > 0.f ? 1.f / s : 0.f;
    #pragma unroll
    for (int i = 0; i < 16; i += 2)
        W[(size_t)row * (K / 2) + (k0 + i) / 2] = (uint8_t)__nv_cvt_float2_to_fp4x2(make_float2(v[i] * inv, v[i + 1] * inv), __NV_E2M1, cudaRoundNearest);
    S[gi] = sc;
}
float nvfp4_from_fp8(uint8_t* W, uint8_t* S, const uint8_t* src, const uint8_t* srcs, int N, int K, cudaStream_t st) {
    unsigned* d; cudaMalloc(&d, 4); cudaMemsetAsync(d, 0, 4, st);
    k_absmax_fp8<<<512, 256, 0, st>>>(d, src, srcs, K, (size_t)N * K);
    const size_t ng = (size_t)N * (K / 16);
    k_nvfp4_quant_fp8<<<(unsigned)((ng + 255) / 256), 256, 0, st>>>(W, S, src, srcs, N, K, d);
    unsigned h; cudaMemcpyAsync(&h, d, 4, cudaMemcpyDeviceToHost, st); cudaStreamSynchronize(st); cudaFree(d);
    float a; std::memcpy(&a, &h, 4);
    return std::max(a, 1e-30f) / (448.f * 6.f);
}
void nvfp4_linear_f32(float* y, const float* x, const uint8_t* W, const uint8_t* S, float g, int M, int N, int K, cudaStream_t st) {
    XSrc X; X.xf = x; X.ldx = K;
    DecNVFP4 d{W, S, g, K};
    launch_dense(y, X, d, M, N, K, 1, false, 1.0f, st);
}

// --------------------------------------------------------------------------- two-stage greedy head
// Greedy decode needs only the argmax of x . head^T, not the 129280 logits. Stage 1 scores every row
// with the NVFP4 copy (a quarter of the bytes); stage 2 recomputes, with the bf16 weights and exactly
// k_dense's M = 1 arithmetic (same chunk order, same fmaf chain, same warp_sum: bit for bit the full
// head's logit), every row whose approximate logit is within `delta` of the approximate maximum, and
// takes the argmax of those (ties to the lower index, as k_argmax). The exact winner is in the
// shortlist whenever |err| at it plus |err| at the approximate winner is <= delta; delta is set from
// the measured error near the top (V41_HEAD_PROBE). More than `cap` candidates sets fb[r] and the host
// recomputes that row with the full head.
__device__ __forceinline__ int h2_key(float f) { const int i = __float_as_int(f); return i >= 0 ? i : i ^ 0x7fffffff; }
__device__ __forceinline__ float h2_val(int k) { return __int_as_float(k >= 0 ? k : k ^ 0x7fffffff); }
// grid (H2_BLK, n); hk[r] is INT_MIN on entry (k_h2_pick leaves it so)
static constexpr int H2_BLK = 64;
__global__ void k_h2_max(int* hk, const float* al, int N) {
    const float* a = al + (size_t)blockIdx.y * N;
    float m = -INFINITY;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += gridDim.x * blockDim.x) m = fmaxf(m, a[i]);
    m = warp_max(m);
    if ((threadIdx.x & 31) == 0) atomicMax(hk + blockIdx.y, h2_key(m));
}
__global__ void k_h2_sel(int* list, int* cnt, const int* hk, const float* al, int N, float delta, int cap) {
    const float* a = al + (size_t)blockIdx.y * N;
    const float thr = h2_val(hk[blockIdx.y]) - delta;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += gridDim.x * blockDim.x)
        if (a[i] >= thr) { const int p = atomicAdd(cnt + blockIdx.y, 1); if (p < cap) list[(size_t)blockIdx.y * cap + p] = i; }
}
// grid (cap/8, n), 256 threads: warp w rescoring candidate blockIdx.x*8 + w of row blockIdx.y
template <class Dec>
__global__ void __launch_bounds__(256) k_h2_rescore(float* resc, const float* x, Dec dec, const int* list, const int* cnt,
                                                    int cap, int K) {
    extern __shared__ float4 xs4[];
    const int r = blockIdx.y, n = min(cnt[r], cap);
    if (blockIdx.x * 8 >= n) return;
    XSrc X; X.xf = x + (size_t)r * K; X.ldx = K;
    stage_row(xs4, X, 0, 0, K);
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, j = blockIdx.x * 8 + warp, nch = K / 32;
    if (j >= n) return;
    const int row = list[(size_t)r * cap + j];
    float acc = 0.f;
    for (int cb = 0; cb < nch; cb += 32) {
        const int c = cb + lane, cc = min(c, nch - 1);
        float w[1][32];
        dec.load(row, cc, w[0]);
        float a[1][1] = {{0.f}};
        dot_chunk_rb<1, 1>(a, w, xs4, cc, nch);
        if (c < nch) acc += a[0][0];
    }
    const float v = warp_sum(acc);
    if (lane == 0) resc[(size_t)r * cap + j] = v * 1.0f;
}
// grid n, 256 threads: argmax of the rescored candidates -> out[r * ostride]; resets hk/cnt for the next call
__global__ void k_h2_pick(int* out, int ostride, int* fb, int* hk, int* cnt, const int* list, const float* resc, int cap) {
    const int r = blockIdx.x, c = cnt[r], n = min(c, cap);
    float bv = -INFINITY; int bi = 0x7fffffff;
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        const float v = resc[(size_t)r * cap + j]; const int i = list[(size_t)r * cap + j];
        if (v > bv || (v == bv && i < bi)) { bv = v; bi = i; }
    }
    __shared__ float sv[256]; __shared__ int si[256];
    sv[threadIdx.x] = bv; si[threadIdx.x] = bi; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) { const float v = sv[threadIdx.x + s]; const int j = si[threadIdx.x + s];
            if (v > sv[threadIdx.x] || (v == sv[threadIdx.x] && j < si[threadIdx.x])) { sv[threadIdx.x] = v; si[threadIdx.x] = j; } }
        __syncthreads();
    }
    if (threadIdx.x == 0) { out[(size_t)r * ostride] = si[0]; fb[r] = c > cap || c == 0; hk[r] = INT_MIN; cnt[r] = 0; }
}
void head2_argmax(int* out, int ostride, int* fb, const float* x, int n, const uint16_t* head, const uint8_t* h4,
                  const uint8_t* h4s, float h4g, H2Scr& sc, int N, int K, cudaStream_t st) {
    nvfp4_linear_f32(sc.alog, x, h4, h4s, h4g, n, N, K, st);
    k_h2_max<<<dim3(H2_BLK, n), 256, 0, st>>>(sc.hk, sc.alog, N);
    k_h2_sel<<<dim3(H2_BLK, n), 256, 0, st>>>(sc.list, sc.cnt, sc.hk, sc.alog, N, sc.delta, sc.cap);
    const size_t smem = (size_t)K * sizeof(float);
    if (perm_map().count(head)) {
        auto kf = k_h2_rescore<DecBF16P>;
        if (smem + 1024 > 48 * 1024) cudaFuncSetAttribute(kf, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        kf<<<dim3(sc.cap / 8, n), 256, smem, st>>>(sc.resc, x, DecBF16P{head, K}, sc.list, sc.cnt, sc.cap, K);
    } else {
        auto kf = k_h2_rescore<DecBF16>;
        if (smem + 1024 > 48 * 1024) cudaFuncSetAttribute(kf, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        kf<<<dim3(sc.cap / 8, n), 256, smem, st>>>(sc.resc, x, DecBF16{head, K}, sc.list, sc.cnt, sc.cap, K);
    }
    k_h2_pick<<<n, 256, 0, st>>>(out, ostride, fb, sc.hk, sc.cnt, sc.list, sc.resc, sc.cap);
}
// S11: two-stage head for SAMPLED rows. Stage 1 writes the NVFP4 logits of every token straight into
// the logits row; every token within *d_delta of the approximate max (the host sets it to
// T ln(1e9) + the greedy delta, so a token left approximate has probability <= 1e-9 x the top one's)
// is rescored exactly (k_h2_rescore: bit for bit bf16_linear) and written back over its approximation.
// fb[r] = 1 when more than cap tokens qualified: the host recomputes that row with the full head.
__global__ void k_h2_sel_d(int* list, int* cnt, const int* hk, const float* al, int N, const float* d_delta, int cap) {
    const float* a = al + (size_t)blockIdx.y * N;
    const float thr = h2_val(hk[blockIdx.y]) - d_delta[0];
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += gridDim.x * blockDim.x)
        if (a[i] >= thr) { const int p = atomicAdd(cnt + blockIdx.y, 1); if (p < cap) list[(size_t)blockIdx.y * cap + p] = i; }
}
__global__ void k_h2_put(float* logits, int N, int* fb, int* hk, int* cnt, const int* list, const float* resc, int cap) {
    const int r = blockIdx.x, c = cnt[r], n = min(c, cap);
    for (int j = threadIdx.x; j < n; j += blockDim.x) logits[(size_t)r * N + list[(size_t)r * cap + j]] = resc[(size_t)r * cap + j];
    __syncthreads();
    if (threadIdx.x == 0) { fb[r] = c > cap || c == 0; hk[r] = INT_MIN; cnt[r] = 0; }
}
void head2_logits(float* logits, int* fb, const float* x, int n, const uint16_t* head, const uint8_t* h4,
                  const uint8_t* h4s, float h4g, H2Scr& sc, const float* d_delta, int N, int K, cudaStream_t st) {
    nvfp4_linear_f32(logits, x, h4, h4s, h4g, n, N, K, st);
    k_h2_max<<<dim3(H2_BLK, n), 256, 0, st>>>(sc.hk, logits, N);
    k_h2_sel_d<<<dim3(H2_BLK, n), 256, 0, st>>>(sc.list, sc.cnt, sc.hk, logits, N, d_delta, sc.cap);
    const size_t smem = (size_t)K * sizeof(float);
    if (perm_map().count(head)) {
        auto kf = k_h2_rescore<DecBF16P>;
        if (smem + 1024 > 48 * 1024) cudaFuncSetAttribute(kf, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        kf<<<dim3(sc.cap / 8, n), 256, smem, st>>>(sc.resc, x, DecBF16P{head, K}, sc.list, sc.cnt, sc.cap, K);
    } else {
        auto kf = k_h2_rescore<DecBF16>;
        if (smem + 1024 > 48 * 1024) cudaFuncSetAttribute(kf, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        kf<<<dim3(sc.cap / 8, n), 256, smem, st>>>(sc.resc, x, DecBF16{head, K}, sc.list, sc.cnt, sc.cap, K);
    }
    k_h2_put<<<n, 256, 0, st>>>(logits, N, fb, sc.hk, sc.cnt, sc.list, sc.resc, sc.cap);
}
// S11: Gumbel-max sampling on the GPU. argmax_i (L_i / T + G_i), G_i iid Gumbel(0,1), is an exact draw from
// softmax(L / T); G_i from a counter hash of (seed, i), so a host-seeded run is reproducible. One CTA of
// 1024 per row; ties to the lower index. (top_p < 1 needs the sorted nucleus: the host path.)
__device__ __forceinline__ uint64_t smix64(uint64_t z) {
    z += 0x9E3779B97F4A7C15ull; z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull; z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
__global__ void __launch_bounds__(1024) k_gumbel_max(int* out, const float* logits, int N, float invT, uint64_t seed) {
    const float* L = logits + (size_t)blockIdx.x * N;
    float bk = -INFINITY; int bi = 0x7fffffff;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        const uint64_t h = smix64(seed ^ ((uint64_t)blockIdx.x << 40) ^ (uint64_t)i);
        // G = -log E, E = -log(1 - w) ~ Exp(1): log1pf keeps E's relative precision as w -> 0, which is
        // the Gumbel upper tail a tail token needs to win (fast __logf near 1 failed gate_gumbel at chi^2 1e6).
        // w = (k + 0.5) 2^-32: the tail is cut at G ~ 22.9 (p 1e-10 a token). fp32: fp64 is 1/64 rate here.
        const float w = ((float)(uint32_t)(h >> 32) + 0.5f) * 2.3283064365386963e-10f;
        const float k = L[i] * invT - logf(-log1pf(-w));
        if (k > bk || (k == bk && i < bi)) { bk = k; bi = i; }
    }
    __shared__ float sk[1024]; __shared__ int si[1024];
    sk[threadIdx.x] = bk; si[threadIdx.x] = bi; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) { const float k = sk[threadIdx.x + s]; const int j = si[threadIdx.x + s];
            if (k > sk[threadIdx.x] || (k == sk[threadIdx.x] && j < si[threadIdx.x])) { sk[threadIdx.x] = k; si[threadIdx.x] = j; } }
        __syncthreads();
    }
    if (threadIdx.x == 0) out[blockIdx.x] = si[0];
}
void sample_gumbel(int* out, const float* logits, int N, float temperature, uint64_t seed, cudaStream_t st) {
    k_gumbel_max<<<1, 1024, 0, st>>>(out, logits, N, 1.0f / temperature, seed);
}
bool H2Scr::alloc(int rows, int N, int cap_, float delta_) {
    cap = cap_; delta = delta_;
    if (cudaMalloc(&alog, (size_t)rows * N * 4) || cudaMalloc(&hk, rows * 4) || cudaMalloc(&cnt, rows * 4) ||
        cudaMalloc(&list, (size_t)rows * cap * 4) || cudaMalloc(&resc, (size_t)rows * cap * 4)) return false;
    std::vector<int> mn(rows, INT_MIN);
    cudaMemcpy(hk, mn.data(), rows * 4, cudaMemcpyHostToDevice);
    cudaMemset(cnt, 0, rows * 4); cudaMemset(list, 0, (size_t)rows * cap * 4);
    return true;
}
} // namespace v41k
