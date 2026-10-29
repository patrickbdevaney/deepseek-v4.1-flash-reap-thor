// v41_moe.cu — V4.1 router and routed/shared experts, for every expert weight format.
//
// Routed experts run in four steps, all with grids sized for the worst case so a captured graph
// never changes shape:
//   1. k_units      group the M*topk (token, slot) pairs by expert, ascending, into units of <= MT
//                   pairs (single thread: <= 768 pairs, deterministic order)
//   2. k_experts    w1|w3 as ONE 4608-row GEMV per unit; activations staged once per CTA
//   3. swiglu + act_quant on the pair rows
//   4. k_experts    w2 (5120 rows, K=2304), then k_combine sums each token's slots in ascending
//                   expert order -- the order of the reference's `for i in experts: y[idx] += ...`
#include "v41_kernels.h"
#include "v41_config.h"
#include "v41_gemv.cuh"
#include "v41_kq_fast.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>

using namespace v41d;
using namespace v41g;

namespace v41k {

// --------------------------------------------------------------------------- router
// Gate.forward. One CTA per token; E <= 512.
__global__ void k_router(float* wout, int* iout, const float* x, const uint16_t* Wg, const float* bias,
                         const float* bias_vl, const uint8_t* img, int E, int topk, float route_scale) {
    const int m = blockIdx.x;
    __shared__ float sc[512], sel[512];
    const float* xr = x + (size_t)m * v41::DIM;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5;
    for (int e = warp; e < E; e += nw) {
        const uint16_t* wr = Wg + (size_t)e * v41::DIM;
        float d = 0.f;
        for (int k = lane; k < v41::DIM; k += 32) d = fmaf(xr[k], bf16f(wr[k]), d);
        d = warp_sum(d);
        if (lane == 0) {
            const float sp = d > 20.f ? d : log1pf(expf(d));       // F.softplus (beta 1, threshold 20)
            const float s = sqrtf(sp);
            sc[e] = s;
            const float* b = (img && img[m] && bias_vl) ? bias_vl : bias;
            sel[e] = s + b[e];
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int ids[16]; float ws[16]; float sum = 0.f;
        for (int k = 0; k < topk; ++k) {                           // ties -> lowest index
            int best = -1; float bv = -INFINITY;
            for (int e = 0; e < E; ++e) {
                bool taken = false; for (int j = 0; j < k; ++j) taken |= (ids[j] == e);
                if (!taken && (best < 0 || sel[e] > bv)) { bv = sel[e]; best = e; }
            }
            ids[k] = best; ws[k] = sc[best]; sum += ws[k];
        }
        for (int k = 0; k < topk; ++k) {
            float w = ws[k];
            if (topk > 1) w = w / (sum + 1e-20f);
            wout[m * topk + k] = w * route_scale;
            iout[m * topk + k] = ids[k];
        }
    }
}
// Parallel form of k_router, bit-identical to it: the same one-warp-per-expert dot (lane-strided
// fmaf, warp_sum) spread over E/8 CTAs per token instead of one CTA, then one warp per token picks
// top-k by repeated parallel argmax with the same rule (highest sel, ties -> lowest index).
// k_router serialised a 1.97 MB read on one SM and ran the selection on one thread.
__global__ void __launch_bounds__(256) k_router_scores(float* sc_out, float* sel_out, const float* x,
                                                       const uint16_t* Wg, const float* bias, const float* bias_vl,
                                                       const uint8_t* img, int E) {
    const int m = blockIdx.y, e = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (e >= E) return;
    const float* xr = x + (size_t)m * v41::DIM;
    const uint16_t* wr = Wg + (size_t)e * v41::DIM;
    float d = 0.f;
    for (int k = lane; k < v41::DIM; k += 32) d = fmaf(xr[k], bf16f(wr[k]), d);
    d = warp_sum(d);
    if (lane == 0) {
        const float sp = d > 20.f ? d : log1pf(expf(d));
        const float s = sqrtf(sp);
        const float* b = (img && img[m] && bias_vl) ? bias_vl : bias;
        sc_out[(size_t)m * E + e] = s;
        sel_out[(size_t)m * E + e] = s + b[e];
    }
}
__global__ void k_router_select(float* wout, int* iout, const float* sc, const float* sel, int E, int topk,
                                float route_scale) {
    const int m = blockIdx.x, lane = threadIdx.x;
    const float* sm = sel + (size_t)m * E;
    uint32_t taken[16] = {0};                 // bit per expert, E <= 512
    int ids[16]; float ws[16]; float sum = 0.f;
    for (int k = 0; k < topk; ++k) {
        int best = -1; float bv = -INFINITY;
        for (int e = lane; e < E; e += 32) {
            if ((taken[e >> 5] >> (e & 31)) & 1u) continue;
            const float v = sm[e];
            if (best < 0 || v > bv) { bv = v; best = e; }     // ascending e per lane: ties keep the lower
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const int ob = __shfl_xor_sync(0xffffffffu, best, o);
            if (ob >= 0 && (best < 0 || ov > bv || (ov == bv && ob < best))) { bv = ov; best = ob; }
        }
        taken[best >> 5] |= 1u << (best & 31);
        ids[k] = best; ws[k] = sc[(size_t)m * E + best]; sum += ws[k];
    }
    if (lane == 0)
        for (int k = 0; k < topk; ++k) {
            float w = ws[k];
            if (topk > 1) w = w / (sum + 1e-20f);
            wout[m * topk + k] = w * route_scale;
            iout[m * topk + k] = ids[k];
        }
}
size_t router_scratch_bytes(int M, int E) { return (size_t)2 * M * E * sizeof(float); }
void router(float* w, int* idx, const float* x, const uint16_t* Wg, const float* bias, const float* bias_vl,
            const uint8_t* d_img, int M, int E, int topk, float route_scale, cudaStream_t st, float* scratch) {
    if (!scratch) { k_router<<<M, 512, 0, st>>>(w, idx, x, Wg, bias, bias_vl, d_img, E, topk, route_scale); return; }
    float* sc = scratch; float* sel = scratch + (size_t)M * E;
    k_router_scores<<<dim3((E + 7) / 8, M), 256, 0, st>>>(sc, sel, x, Wg, bias, bias_vl, d_img, E);
    k_router_select<<<M, 32, 0, st>>>(w, idx, sc, sel, E, topk, route_scale);
}

// --------------------------------------------------------------------------- grouping
static constexpr int UNIT_MT = 4;
struct Unit { int e, start, cnt; };
// pairs sorted by (expert, token): pair_tok[p], pair_slot[p]; units over them; n_units in units_n.
__global__ void k_units(Unit* units, int* units_n, int* pair_tok, int* pair_slot, int* inv, const int* idx, int M,
                        int topk, int mt) {
    if (threadIdx.x || blockIdx.x) return;
    const int P = M * topk;
    // insertion order (expert, token, slot); P <= 768 so a serial sort is microseconds
    int n = 0;
    for (int p = 0; p < P; ++p) {
        const int e = idx[p], t = p / topk;
        int j = n++;
        while (j > 0) {
            const int pe = idx[pair_tok[j - 1] * topk + pair_slot[j - 1]], pt = pair_tok[j - 1];
            if (pe < e || (pe == e && pt <= t)) break;
            pair_tok[j] = pair_tok[j - 1]; pair_slot[j] = pair_slot[j - 1]; --j;
        }
        pair_tok[j] = t; pair_slot[j] = p % topk;
    }
    for (int p = 0; p < P; ++p) inv[pair_tok[p] * topk + pair_slot[p]] = p;
    int u = 0;
    for (int p = 0; p < P;) {
        const int e = idx[pair_tok[p] * topk + pair_slot[p]];
        int q = p; while (q < P && q - p < mt && idx[pair_tok[q] * topk + pair_slot[q]] == e) ++q;
        units[u++] = Unit{e, p, q - p}; p = q;
    }
    *units_n = u;
}

// --------------------------------------------------------------------------- expert GEMV
// grid (row tiles, max units). Unit u: expert e, pairs [start, start+cnt). Input rows: for the
// gate/up pass, token rows of x (pair_tok); for the down pass, pair rows of act (identity).
template <class Dec, int MT>
__global__ void __launch_bounds__(256) k_experts(float* __restrict__ out, int ldo, XSrc X, bool x_by_pair,
                                                 const Unit* units, const int* units_n, const int* pair_tok,
                                                 typename Dec::Tab tab, int N, int K) {
    const int u = blockIdx.y;
    if (u >= *units_n) return;
    const Unit U = units[u];
    extern __shared__ float4 xs4[];
    const int nch = K / 32;
    for (int t = 0; t < MT; ++t) {
        const int p = U.start + min(t, U.cnt - 1);
        stage_row(xs4 + (size_t)t * nch * 8, X, x_by_pair ? p : pair_tok[p], 0, K);
    }
    __syncthreads();
    const Dec dec = Dec::make(tab, U.e, K, threadIdx.x & 31);
    const int row0 = blockIdx.x * ROWS_PER_CTA;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    float acc[ROWS_PER_WARP][MT];
    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r)
        #pragma unroll
        for (int t = 0; t < MT; ++t) acc[r][t] = 0.f;
    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) {
        const int row = min(row0 + warp * ROWS_PER_WARP + r, N - 1);
        for (int cb = 0; cb < nch; cb += 32) {
            const int c = cb + lane, cc = min(c, nch - 1);
            float w[32]; dec.load(row, cc, w);
            float a[MT];
            #pragma unroll
            for (int t = 0; t < MT; ++t) a[t] = 0.f;
            dot_chunk<MT>(a, w, xs4, cc, nch);
            if (c < nch) {
                #pragma unroll
                for (int t = 0; t < MT; ++t) acc[r][t] += a[t];
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r)
        #pragma unroll
        for (int t = 0; t < MT; ++t) {
            const float v = warp_sum(acc[r][t]);
            const int row = row0 + warp * ROWS_PER_WARP + r;
            if (lane == 0 && row < N && t < U.cnt) out[(size_t)(U.start + t) * ldo + row] = bf16r(v);
        }
}

// Per-format "make": resolve expert e's pointers from the device tables.
struct TabMX { const uint8_t* const* w1; const uint8_t* const* s1; const uint8_t* const* w3; const uint8_t* const* s3;
               const uint8_t* const* w2; const uint8_t* const* s2; int pass; };
template <bool HW> struct MXDec : DecMXFP4<HW> {
    using Tab = TabMX;
    __device__ static MXDec make(const Tab& t, int e, int K, int) {
        MXDec d;
        if (t.pass == 0) { d.W0 = t.w1[e]; d.S0 = t.s1[e]; d.W1 = t.w3[e]; d.S1 = t.s3[e]; d.split = v41::MOE_INTER; }
        else { d.W0 = d.W1 = t.w2[e]; d.S0 = d.S1 = t.s2[e]; d.split = 1 << 30; }
        d.K = K; return d;
    }
};
struct TabLUT { const uint8_t* const* w1; const uint8_t* const* s1; const uint8_t* const* w3; const uint8_t* const* s3;
                const uint8_t* const* w2; const uint8_t* const* s2; const float* const* lut; int pass; };
template <int B> struct LUTDec : DecLUT<B> {
    using Tab = TabLUT;
    __device__ static LUTDec make(const Tab& t, int e, int K, int lane) {
        LUTDec d; const float* L = t.lut[e];         // 3 x 16 floats: w1, w3, w2
        const int le = lane & 15;
        if (t.pass == 0) { d.W0 = t.w1[e]; d.S0 = t.s1[e]; d.W1 = t.w3[e]; d.S1 = t.s3[e]; d.split = v41::MOE_INTER;
                           d.lut0 = L[le]; d.lut1 = L[16 + le]; }
        else { d.W0 = d.W1 = t.w2[e]; d.S0 = d.S1 = t.s2[e]; d.split = 1 << 30; d.lut0 = d.lut1 = L[32 + le]; }
        d.K = K; return d;
    }
};
template <int QT> struct KQDec : DecKQ<QT> {
    using Tab = TabMX;
    __device__ static KQDec make(const Tab& t, int e, int K, int) {
        KQDec d;
        if (t.pass == 0) { d.W0 = t.w1[e]; d.W1 = t.w3[e]; d.split = v41::MOE_INTER; }
        else { d.W0 = d.W1 = t.w2[e]; d.split = 1 << 30; }
        d.K = K; return d;
    }
};
template <int B> struct KTDec : DecKT<B> {
    using Tab = TabLUT;
    __device__ static KTDec make(const Tab& t, int e, int K, int) {
        KTDec d;
        if (t.pass == 0) { d.W0 = t.w1[e]; d.S0 = t.s1[e]; d.W1 = t.w3[e]; d.S1 = t.s3[e]; d.split = v41::MOE_INTER; }
        else { d.W0 = d.W1 = t.w2[e]; d.S0 = d.S1 = t.s2[e]; d.split = 1 << 30; }
        d.K = K; return d;
    }
};

// KT passes read rotated activations: out[r] = FWHT_256(deq(x[r]) * sign) / 16 in fp32, with the
// butterfly order of ops/lib/kt.py::rht_rows (h = 1, 2, .., 128; pair (j, j+h) -> (a+b, a-b)), so
// tests/gate_v41_units.cu can hold it bit-exact. One CTA per (256-block, row).
__global__ void __launch_bounds__(256) k_rht_rows(float* __restrict__ out, XSrc X, const uint8_t* __restrict__ sign, int K) {
    __shared__ float v[256];
    const int r = blockIdx.y, t = threadIdx.x, k = blockIdx.x * 256 + t;
    float x;
    if (X.xf) x = X.xf[(size_t)r * X.ldx + k];
    else x = __fmul_rn(e4m3f(X.xq[(size_t)r * X.ldx + k]), X.xs[(size_t)r * (X.ldx / 32) + k / 32]);
    if ((sign[k >> 3] >> (k & 7)) & 1u) x = -x;
    v[t] = x;
    __syncthreads();
    #pragma unroll
    for (int h = 1; h < 256; h <<= 1) {
        const float nv = (t & h) ? __fsub_rn(v[t - h], v[t]) : __fadd_rn(v[t], v[t + h]);
        __syncthreads();
        v[t] = nv;
        __syncthreads();
    }
    out[(size_t)r * K + k] = __fmul_rn(v[t], 0.0625f);
}
void rht_rows(float* out, const float* x, const uint8_t* sign, int M, int K, cudaStream_t st) {
    XSrc X; X.xf = x; X.ldx = K;
    k_rht_rows<<<dim3(K / 256, M), 256, 0, st>>>(out, X, sign, K);
}

// combine: y[m] += sum over this token's slots in ascending expert order of down[pair]
__global__ void k_combine(float* y, const float* down, const int* idx, const int* inv, int M, int topk, const float* gain) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i >= (size_t)M * v41::DIM) return;
    const int m = (int)(i / v41::DIM), d = (int)(i % v41::DIM);
    int e[16], p[16];
    for (int j = 0; j < topk; ++j) {
        int ej = idx[m * topk + j], pj = inv[m * topk + j], k = j;
        while (k > 0 && e[k - 1] > ej) { e[k] = e[k - 1]; p[k] = p[k - 1]; --k; }
        e[k] = ej; p[k] = pj;
    }
    float s = y[i];
    for (int j = 0; j < topk; ++j) s += down[(size_t)p[j] * v41::DIM + d];
    y[i] = gain ? s * *gain : s;          // y is zeroed by the caller: s is the routed sum
}

// scratch layout (bytes, 256-aligned): units, n, pair_tok, pair_slot, gu [P,4608], act [P,2304],
// actq [P,2304], acts [P,72], down [P,5120], wpair [P]
static size_t al(size_t b) { return (b + 255) & ~(size_t)255; }
size_t moe_scratch_bytes(int M, int topk) {
    const size_t P = (size_t)M * topk;
    return al(P * sizeof(Unit)) + al(16) + 3 * al(P * 4) + al(P * 2 * v41::MOE_INTER * 4) + al(P * v41::MOE_INTER * 4)
         + al(P * v41::MOE_INTER) + al(P * (v41::MOE_INTER / 32) * 4) + al(P * v41::DIM * 4) + al(P * 4)
         + al((size_t)M * v41::DIM * 4) + al(P * v41::MOE_INTER * 4);          // KT rotated x (pass 0) / act (pass 1)
}
__global__ void k_pair_w(float* wp, const float* w, const int* pair_tok, const int* pair_slot, int P, int topk) {
    int p = blockIdx.x * blockDim.x + threadIdx.x; if (p < P) wp[p] = w[pair_tok[p] * topk + pair_slot[p]];
}

template <class Dec>
static void run_pass(float* out, int ldo, const XSrc& X, bool by_pair, const Unit* units, const int* nu, const int* ptok,
                     const typename Dec::Tab& tab, int N, int K, int max_units, int mt, cudaStream_t st) {
    dim3 grid((N + ROWS_PER_CTA - 1) / ROWS_PER_CTA, max_units);
    const size_t smem = (size_t)mt * K * sizeof(float);
    if (mt == 1) { auto kf = k_experts<Dec, 1>; kf<<<grid, 256, smem, st>>>(out, ldo, X, by_pair, units, nu, ptok, tab, N, K); }
    else { auto kf = k_experts<Dec, UNIT_MT>;
           if (smem > 48 * 1024) cudaFuncSetAttribute(kf, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
           kf<<<grid, 256, smem, st>>>(out, ldo, X, by_pair, units, nu, ptok, tab, N, K); }
}

// Q2_K / Q3_K half2 path (v41_kq_fast.cuh). V41_KQ_FAST=0 selects the bit-exact fp32 decode instead.
static int g_kq_fast = -1;
static bool kq_fast() {
    if (g_kq_fast < 0) { const char* e = getenv("V41_KQ_FAST"); g_kq_fast = (e && !strcmp(e, "0")) ? 0 : 1; }
    return g_kq_fast;
}
void kq_fast_set(bool on) { g_kq_fast = on ? 1 : 0; }
template <int QT, int MT>
static void launch_kq(float* out, int ldo, const XSrc& X, bool by_pair, const Unit* units, const int* nu, const int* ptok,
                      const KQTab& tab, int N, int K, int max_units, cudaStream_t st) {
    dim3 grid((N + ROWS_PER_CTA - 1) / ROWS_PER_CTA, max_units);
    const size_t smem = (size_t)MT * kq_stage_bytes(K);
    auto kf = k_kq<QT, MT, Unit>;
    if (smem > 48 * 1024) cudaFuncSetAttribute(kf, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    kf<<<grid, 256, smem, st>>>(out, ldo, X, by_pair, units, nu, ptok, tab, N, K);
}
static void run_kq(int bits, float* out, int ldo, const XSrc& X, bool by_pair, const Unit* units, const int* nu,
                   const int* ptok, const KQTab& tab, int N, int K, int max_units, int mt, cudaStream_t st) {
    if (bits == 2) { if (mt == 1) launch_kq<2, 1>(out, ldo, X, by_pair, units, nu, ptok, tab, N, K, max_units, st);
                     else launch_kq<2, UNIT_MT>(out, ldo, X, by_pair, units, nu, ptok, tab, N, K, max_units, st); }
    else           { if (mt == 1) launch_kq<3, 1>(out, ldo, X, by_pair, units, nu, ptok, tab, N, K, max_units, st);
                     else launch_kq<3, UNIT_MT>(out, ldo, X, by_pair, units, nu, ptok, tab, N, K, max_units, st); }
}

// one pass (0 = gate|up, 1 = down) in the format the table names for it
static void dispatch_pass(int pass, const ExpertTab& T, float* out, int ldo, const XSrc& X, bool by_pair,
                          const Unit* units, const int* nu, const int* ptok, int N, int K, int max_units, int mt,
                          cudaStream_t st) {
    const ExpertFmt f = pass == 0 ? T.fmt13 : T.fmt2;
    const int bits = pass == 0 ? T.bits13 : T.bits2;
    TabMX tm{T.w1, T.w1s, T.w3, T.w3s, T.w2, T.w2s, pass};
    TabLUT tl{T.w1, T.w1s, T.w3, T.w3s, T.w2, T.w2s, T.lut, pass};
    switch (f) {
    case ExpertFmt::MXFP4_HW: run_pass<MXDec<true>>(out, ldo, X, by_pair, units, nu, ptok, tm, N, K, max_units, mt, st); break;
    case ExpertFmt::MXFP4:    run_pass<MXDec<false>>(out, ldo, X, by_pair, units, nu, ptok, tm, N, K, max_units, mt, st); break;
    case ExpertFmt::LUT:
        if (bits == 2) run_pass<LUTDec<2>>(out, ldo, X, by_pair, units, nu, ptok, tl, N, K, max_units, mt, st);
        else if (bits == 3) run_pass<LUTDec<3>>(out, ldo, X, by_pair, units, nu, ptok, tl, N, K, max_units, mt, st);
        else run_pass<LUTDec<4>>(out, ldo, X, by_pair, units, nu, ptok, tl, N, K, max_units, mt, st);
        break;
    case ExpertFmt::KQ:
        if ((bits == 2 || bits == 3) && X.xq && kq_fast()) {
            run_kq(bits, out, ldo, X, by_pair, units, nu, ptok, KQTab{T.w1, T.w3, T.w2, pass}, N, K, max_units, mt, st);
            break;
        }
        if (bits == 2) run_pass<KQDec<2>>(out, ldo, X, by_pair, units, nu, ptok, tm, N, K, max_units, mt, st);
        else if (bits == 3) run_pass<KQDec<3>>(out, ldo, X, by_pair, units, nu, ptok, tm, N, K, max_units, mt, st);
        else run_pass<KQDec<4>>(out, ldo, X, by_pair, units, nu, ptok, tm, N, K, max_units, mt, st);
        break;
    case ExpertFmt::KT:
        if (bits == 2) run_pass<KTDec<2>>(out, ldo, X, by_pair, units, nu, ptok, tl, N, K, max_units, mt, st);
        else run_pass<KTDec<3>>(out, ldo, X, by_pair, units, nu, ptok, tl, N, K, max_units, mt, st);
        break;
    }
}

void moe_routed(float* y, const uint8_t* xq, const float* xs, const float* w, const int* idx, const ExpertTab& T,
                int M, int topk, void* scratch, cudaStream_t st) {
    const int P = M * topk, mt = (M == 1) ? 1 : UNIT_MT;
    char* b = (char*)scratch;
    Unit* units = (Unit*)b; b += al(P * sizeof(Unit));
    int* nu = (int*)b; b += al(16);
    int* ptok = (int*)b; b += al(P * 4);
    int* pslot = (int*)b; b += al(P * 4);
    int* inv = (int*)b; b += al(P * 4);
    float* gu = (float*)b; b += al((size_t)P * 2 * v41::MOE_INTER * 4);
    float* act = (float*)b; b += al((size_t)P * v41::MOE_INTER * 4);
    uint8_t* aq = (uint8_t*)b; b += al((size_t)P * v41::MOE_INTER);
    float* as = (float*)b; b += al((size_t)P * (v41::MOE_INTER / 32) * 4);
    float* down = (float*)b; b += al((size_t)P * v41::DIM * 4);
    float* wp = (float*)b; b += al((size_t)P * 4);
    float* xr13 = (float*)b; b += al((size_t)M * v41::DIM * 4);
    float* xr2 = (float*)b;
    k_units<<<1, 1, 0, st>>>(units, nu, ptok, pslot, inv, idx, M, topk, mt);
    k_pair_w<<<(P + 127) / 128, 128, 0, st>>>(wp, w, ptok, pslot, P, topk);
    XSrc X; X.xq = xq; X.xs = xs; X.ldx = v41::DIM;
    if (T.fmt13 == ExpertFmt::KT) {               // rotate the token rows once; every unit stages them
        k_rht_rows<<<dim3(v41::DIM / 256, M), 256, 0, st>>>(xr13, X, T.rht13, v41::DIM);
        X = XSrc(); X.xf = xr13; X.ldx = v41::DIM;
    }
    dispatch_pass(0, T, gu, 2 * v41::MOE_INTER, X, false, units, nu, ptok, 2 * v41::MOE_INTER, v41::DIM, P, mt, st);
    swiglu(act, gu, gu + v41::MOE_INTER, wp, P, v41::MOE_INTER, v41::SWIGLU_LIMIT, 2 * v41::MOE_INTER, st);
    act_quant_fp8(aq, as, act, P, v41::MOE_INTER, st);
    XSrc A; A.xq = aq; A.xs = as; A.ldx = v41::MOE_INTER;
    if (T.fmt2 == ExpertFmt::KT) {
        k_rht_rows<<<dim3(v41::MOE_INTER / 256, P), 256, 0, st>>>(xr2, A, T.rht2, v41::MOE_INTER);
        A = XSrc(); A.xf = xr2; A.ldx = v41::MOE_INTER;
    }
    dispatch_pass(1, T, down, v41::DIM, A, true, units, nu, ptok, v41::DIM, v41::MOE_INTER, P, mt, st);
    size_t n = (size_t)M * v41::DIM;
    k_combine<<<(unsigned)((n + 255) / 256), 256, 0, st>>>(y, down, idx, inv, M, topk, T.heal_gain);
}

// Shared expert: fp8 linears (32x32 blocks). moe_shared: o = shared(x) (o may be null: then it lives
// in scratch and is returned); moe_shared_add: y = bf16(y + shared(x)).
float* moe_shared(float* o, const uint8_t* xq, const float* xs, const uint8_t* w1, const uint8_t* w1s,
                  const uint8_t* w3, const uint8_t* w3s, const uint8_t* w2, const uint8_t* w2s, int M, void* scratch,
                  cudaStream_t st) {
    const int I = v41::MOE_INTER;
    char* b = (char*)scratch;
    float* g = (float*)b; b += al((size_t)M * I * 4);
    float* u = (float*)b; b += al((size_t)M * I * 4);
    float* a = (float*)b; b += al((size_t)M * I * 4);
    uint8_t* aq = (uint8_t*)b; b += al((size_t)M * I);
    float* as = (float*)b; b += al((size_t)M * (I / 32) * 4);
    if (!o) o = (float*)b;
    fp8_linear(g, xq, xs, w1, w1s, M, I, v41::DIM, true, st);
    fp8_linear(u, xq, xs, w3, w3s, M, I, v41::DIM, true, st);
    swiglu(a, g, u, nullptr, M, I, v41::SWIGLU_LIMIT, I, st);
    act_quant_fp8(aq, as, a, M, I, st);
    fp8_linear(o, aq, as, w2, w2s, M, v41::DIM, I, true, st);
    return o;
}
void moe_shared_add(float* y, const uint8_t* xq, const float* xs, const uint8_t* w1, const uint8_t* w1s,
                    const uint8_t* w3, const uint8_t* w3s, const uint8_t* w2, const uint8_t* w2s, int M, void* scratch,
                    cudaStream_t st) {
    float* o = moe_shared(nullptr, xq, xs, w1, w1s, w3, w3s, w2, w2s, M, scratch, st);
    add_bf16(y, o, (size_t)M * v41::DIM, st);
}

} // namespace v41k
