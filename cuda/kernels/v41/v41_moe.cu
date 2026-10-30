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
#include "v41_kq_mma.cuh"
#include <cub/block/block_scan.cuh>
#include <cub/block/block_radix_sort.cuh>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <unordered_map>
#include <vector>

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

// Router on a permuted gate (router_permute), bit-identical to k_router_scores + k_router_select.
// Lane l of an expert's warp sums x[l + 32 j] * W[l + 32 j] for j = 0, 1, ... in order (k_router_scores'
// chain), but W is stored so that its j = 8i..8i+7 sit in one 16-byte piece at i*512 + 16 l: one warp
// load is 512 contiguous bytes, where the row-major gate gave 2-byte lane-strided loads (28 us a token
// for 1.97 MB). x is staged in smem. The last CTA of a token to finish ranks the E selection scores in
// one pass (rank = how many beat it: higher sel, ties to the lower index -- the repeated-argmax order)
// instead of k_router_select's topk serial warp passes, then sums the weights in rank order.
static std::unordered_map<const void*, int>& rperm_map() { static std::unordered_map<const void*, int> m; return m; }
static int* g_rcnt = nullptr;                          // per-token CTA arrival counters, back to 0 after each call
static constexpr int RCNT = 4096;                     // per-row completion counters: rows of one call (State::MAX_CHUNK 2048)
static constexpr int RP_WARPS = 4;
__global__ void k_router_permute(uint16_t* W, int K) {   // one row per CTA, in place via smem
    extern __shared__ uint16_t rp_row[];
    uint16_t* r = W + (size_t)blockIdx.x * K;
    for (int k = threadIdx.x; k < K; k += blockDim.x) rp_row[k] = r[k];
    __syncthreads();
    for (int p = threadIdx.x; p < K; p += blockDim.x) {      // p = i*256 + l*8 + t  <-  k = l + 32 (8 i + t)
        const int i = p >> 8, l = (p >> 3) & 31, t = p & 7;
        r[p] = rp_row[l + 32 * (8 * i + t)];
    }
}
bool router_permute(uint16_t* Wg, int E, int K, cudaStream_t st) {
    if (K % 256 || rperm_map().count(Wg)) return false;
    if (!g_rcnt) { if (cudaMalloc(&g_rcnt, RCNT * sizeof(int)) != cudaSuccess) return false; cudaMemset(g_rcnt, 0, RCNT * sizeof(int)); }
    k_router_permute<<<E, 256, K * sizeof(uint16_t), st>>>(Wg, K);
    rperm_map()[Wg] = K;
    return true;
}
// Prefill: a CTA serves RP_TM tokens with the same register-resident row slice (the gate was re-read from
// L2 once per token: 2 MB x 256 rows = 2.1 ms a chunk); each (expert, token) chain is the decode one.
static constexpr int RP_TM = 8;
__global__ void __launch_bounds__(RP_WARPS * 32) k_router_p(float* wout, int* iout, float* sc_out, float* sel_out,
                                                            const float* x, const uint16_t* Wg, const float* bias,
                                                            const float* bias_vl, const uint8_t* img, int M, int E,
                                                            int topk, float route_scale, int* rcnt, int tm) {
    constexpr int K = v41::DIM;
    __shared__ __align__(16) float xs[K];
    __shared__ int lastf[RP_TM];
    const int m0 = blockIdx.y * tm, nm = min(tm, M - m0);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, e = blockIdx.x * RP_WARPS + warp;
    // all K/256 = 20 row loads issued before the x staging and the chain: the warp's 20 dependent
    // 512-B loads (unroll 4) left it latency-bound at ~40 GB/s; the fmaf order is unchanged
    uint4 wu[K / 256];
    if (e < E) {
        const uint4* wr = reinterpret_cast<const uint4*>(Wg + (size_t)e * K) + lane;
        #pragma unroll
        for (int i = 0; i < K / 256; ++i) wu[i] = __ldcs(wr + i * 32);
    }
    // PDL (A9): the gate does not depend on the previous kernel, x does; no-ops on a plain launch
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    asm volatile("griddepcontrol.wait;" ::: "memory");
    for (int mi = 0; mi < nm; ++mi) {
        const int m = m0 + mi;
        if (mi) __syncthreads();                          // the previous token's chains are done with xs
        {   // all of this thread's x loads in flight before the smem stores
            constexpr int XP = (K / 4 + RP_WARPS * 32 - 1) / (RP_WARPS * 32);
            float4 xv[XP];
            const float4* xr = reinterpret_cast<const float4*>(x + (size_t)m * K);
            #pragma unroll
            for (int q = 0; q < XP; ++q) { const int k = threadIdx.x + q * RP_WARPS * 32; if (k < K / 4) xv[q] = xr[k]; }
            #pragma unroll
            for (int q = 0; q < XP; ++q) { const int k = threadIdx.x + q * RP_WARPS * 32; if (k < K / 4) reinterpret_cast<float4*>(xs)[k] = xv[q]; }
        }
        __syncthreads();
        if (e < E) {
            float d = 0.f;
            #pragma unroll
            for (int i = 0; i < K / 256; ++i) {
                const uint4 u = wu[i];
                const uint32_t v[4] = {u.x, u.y, u.z, u.w};
                #pragma unroll
                for (int t = 0; t < 8; ++t) {
                    const float wv = __uint_as_float(t & 1 ? (v[t >> 1] & 0xFFFF0000u) : (v[t >> 1] << 16));
                    d = fmaf(xs[lane + 32 * (8 * i + t)], wv, d);
                }
            }
            d = warp_sum(d);
            if (lane == 0) {
                const float sp = d > 20.f ? d : log1pf(expf(d));
                const float s = sqrtf(sp);
                const float* b = (img && img[m] && bias_vl) ? bias_vl : bias;
                __stcg(sc_out + (size_t)m * E + e, s);
                __stcg(sel_out + (size_t)m * E + e, s + b[e]);
            }
        }
    }
    __threadfence();
    __syncthreads();
    if (threadIdx.x < nm) lastf[threadIdx.x] = atomicAdd(rcnt + m0 + threadIdx.x, 1) == (int)gridDim.x - 1;
    __syncthreads();
    __shared__ float sv[512]; __shared__ int ids[16];
    for (int mi = 0; mi < nm; ++mi) {
        if (!lastf[mi]) continue;                         // uniform across the CTA (smem flag)
        const int m = m0 + mi;
        __threadfence();
        for (int j = threadIdx.x; j < E; j += blockDim.x) sv[j] = __ldcg(sel_out + (size_t)m * E + j);
        __syncthreads();
        for (int j = threadIdx.x; j < E; j += blockDim.x) {
            const float v = sv[j]; int rank = 0;
            for (int q = 0; q < E; ++q) { const float o = sv[q]; rank += (o > v) || (o == v && q < j); }
            if (rank < topk) ids[rank] = j;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            float ws[16], sum = 0.f;
            for (int k = 0; k < topk; ++k) { ws[k] = __ldcg(sc_out + (size_t)m * E + ids[k]); sum += ws[k]; }
            for (int k = 0; k < topk; ++k) {
                float w = ws[k];
                if (topk > 1) w = w / (sum + 1e-20f);
                wout[m * topk + k] = w * route_scale;
                iout[m * topk + k] = ids[k];
            }
            rcnt[m] = 0;
        }
        __syncthreads();                                  // sv / ids reused by the next token
    }
}
size_t router_scratch_bytes(int M, int E) { return (size_t)2 * M * E * sizeof(float); }
void router(float* w, int* idx, const float* x, const uint16_t* Wg, const float* bias, const float* bias_vl,
            const uint8_t* d_img, int M, int E, int topk, float route_scale, cudaStream_t st, float* scratch) {
    if (!scratch) { k_router<<<M, 512, 0, st>>>(w, idx, x, Wg, bias, bias_vl, d_img, E, topk, route_scale); return; }
    if (rperm_map().count(Wg)) {
        if (M > RCNT || E > 512 || topk > 16) { fprintf(stderr, "router: M %d E %d topk %d out of range\n", M, E, topk); abort(); }
        static const int tm_env = [] { const char* e = getenv("V41_ROUTER_TM"); return e ? atoi(e) : RP_TM; }();
        const int tm = std::max(1, std::min(M == 1 ? 1 : tm_env, RP_TM));
        static const bool pdl = [] { const char* e = getenv("V41_PDL"); return !(e && *e == '0'); }();
        cudaLaunchConfig_t cfg = {};
        cfg.gridDim = dim3((E + RP_WARPS - 1) / RP_WARPS, (M + tm - 1) / tm); cfg.blockDim = dim3(RP_WARPS * 32); cfg.stream = st;
        cudaLaunchAttribute at[1];
        at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; at[0].val.programmaticStreamSerializationAllowed = 1;
        cfg.attrs = at; cfg.numAttrs = pdl ? 1 : 0;
        cudaLaunchKernelEx(&cfg, k_router_p, w, idx, scratch, scratch + (size_t)M * E, x, Wg, bias, bias_vl, d_img, M, E, topk,
                           route_scale, g_rcnt, tm);
        return;
    }
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

// Same output as k_units for P <= UNITS_PAR_MAX, one thread per pair: a pair's sorted position is the
// count of pairs ordered before it by (expert, token, pair index) -- the serial sort is stable, so this
// is its order -- and units start at every mt-th pair of an expert's run (prefix-counted by BlockScan).
// The serial kernel sat at 14 us per layer of pure single-thread latency at M = 1.
static constexpr int UNITS_PAR_MAX = 1024;
using UScan = cub::BlockScan<int, UNITS_PAR_MAX>;
__global__ void __launch_bounds__(UNITS_PAR_MAX) k_units_par(Unit* units, int* units_n, int* pair_tok, int* pair_slot,
                                                              int* inv, const int* idx, int M, int topk, int mt) {
    __shared__ typename UScan::TempStorage ts;
    __shared__ int se[UNITS_PAR_MAX], so_e[UNITS_PAR_MAX], so_left[UNITS_PAR_MAX], so_off[UNITS_PAR_MAX];
    const int P = M * topk, p = threadIdx.x;
    if (p < P) se[p] = idx[p];
    __syncthreads();
    if (p < P) {                                         // pass 1, by pair: sorted position r
        const int e = se[p], t = p / topk;
        int r = 0, s = 0, run = 0;
        for (int q = 0; q < P; ++q) {
            const int eq = se[q], tq = q / topk;
            s += eq < e; run += eq == e;
            r += eq < e || (eq == e && (tq < t || (tq == t && q < p)));
        }
        pair_tok[r] = t; pair_slot[r] = p % topk; inv[p] = r;
        so_e[r] = e; so_off[r] = r - s; so_left[r] = run - (r - s);
    }
    __syncthreads();
    // pass 2, by sorted position: units start at every mt-th pair of a run, numbered in sorted order
    const int start = p < P && so_off[p] % mt == 0;
    int u, total;
    UScan(ts).ExclusiveSum(start, u, total);
    if (start) units[u] = Unit{so_e[p], p, min(mt, so_left[p])};
    if (p == 0) *units_n = total;
}

// k_units_par for P up to UNITS_BIG_MAX (prefill chunks above 170 tokens). The order is (expert, token):
// a token's top-k experts are distinct, so key e << 12 | t is unique and a BlockRadixSort of the pairs by
// it gives k_units_par's sorted position r without the O(P^2) rank loop (2.2 ms per layer at P = 3072).
// Run offsets come from a per-expert histogram (counts are order-free, so the result is deterministic).
// IPT 4 up to P = 4096 (chunk 682), IPT 12 up to 12288 (chunk 2048); the sort storage is dynamic smem (> 48 KB
// at IPT 12) after the per-expert counts.
static constexpr int UB_EMAX = 511, UB_TBITS = 12, UNITS_BIG_MAX = UNITS_PAR_MAX * 12;
template <int IPT> union UBStore { typename cub::BlockRadixSort<int, UNITS_PAR_MAX, IPT, int>::TempStorage rs; typename UScan::TempStorage sc; };
template <int IPT> constexpr int ub_smem() { return 2 * (UB_EMAX + 1) * (int)sizeof(int) + (int)sizeof(UBStore<IPT>); }
template <int IPT>
__global__ void __launch_bounds__(UNITS_PAR_MAX) k_units_big(Unit* units, int* units_n, int* pair_tok, int* pair_slot,
                                                              int* inv, const int* idx, int M, int topk, int mt) {
    using URSort = cub::BlockRadixSort<int, UNITS_PAR_MAX, IPT, int>;
    extern __shared__ __align__(16) int ub[];
    int* cnt = ub; int* first = ub + (UB_EMAX + 1);
    UBStore<IPT>& ts = *reinterpret_cast<UBStore<IPT>*>(ub + 2 * (UB_EMAX + 1));
    const int P = M * topk;
    for (int e = threadIdx.x; e <= UB_EMAX; e += blockDim.x) cnt[e] = 0;
    __syncthreads();
    int key[IPT], val[IPT];
    #pragma unroll
    for (int i = 0; i < IPT; ++i) {
        const int p = threadIdx.x * IPT + i;
        if (p < P) { const int e = idx[p]; key[i] = e << UB_TBITS | (p / topk); val[i] = p; atomicAdd(&cnt[e], 1); }
        else { key[i] = (UB_EMAX << UB_TBITS) | ((1 << UB_TBITS) - 1); val[i] = -1; }   // pads sort last
    }
    URSort(ts.rs).Sort(key, val, 0, UB_TBITS + 9);
    __syncthreads();                                     // cnt complete; ts free for the scan
    if (threadIdx.x == 0) { int s = 0; for (int e = 0; e <= UB_EMAX; ++e) { first[e] = s; s += cnt[e]; } }
    __syncthreads();
    int st[IPT], u[IPT], total, ue[IPT], ul[IPT];
    #pragma unroll
    for (int i = 0; i < IPT; ++i) {
        const int r = threadIdx.x * IPT + i, p = val[i];
        st[i] = 0;
        if (r >= P) continue;
        const int e = key[i] >> UB_TBITS, off = r - first[e];
        pair_tok[r] = p / topk; pair_slot[r] = p % topk; inv[p] = r;
        st[i] = off % mt == 0; ue[i] = e; ul[i] = cnt[e] - off;
    }
    UScan(ts.sc).ExclusiveSum(st, u, total);
    #pragma unroll
    for (int i = 0; i < IPT; ++i)
        if (st[i]) units[u[i]] = Unit{ue[i], (int)threadIdx.x * IPT + i, min(mt, ul[i])};
    if (threadIdx.x == 0) *units_n = total;
}
template <int IPT> static bool units_big_launch(Unit* units, int* nu, int* ptok, int* pslot, int* inv, const int* idx, int M,
                                                int topk, int mt, cudaStream_t st) {
    static const bool attr = cudaFuncSetAttribute(k_units_big<IPT>, cudaFuncAttributeMaxDynamicSharedMemorySize, ub_smem<IPT>()) == cudaSuccess;
    if (attr) k_units_big<IPT><<<1, UNITS_PAR_MAX, ub_smem<IPT>(), st>>>(units, nu, ptok, pslot, inv, idx, M, topk, mt);
    return attr;
}
static bool units_big(Unit* units, int* nu, int* ptok, int* pslot, int* inv, const int* idx, int M, int topk, int mt, cudaStream_t st) {
    const int P = M * topk;
    if (P <= UNITS_PAR_MAX * 4) return units_big_launch<4>(units, nu, ptok, pslot, inv, idx, M, topk, mt, st);
    return units_big_launch<12>(units, nu, ptok, pslot, inv, idx, M, topk, mt, st);
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
         + al((size_t)M * v41::DIM * 4) + al(P * v41::MOE_INTER * 4)            // KT rotated x (pass 0) / act (pass 1)
         + al((size_t)M * kqt_stage_bytes(v41::DIM)) + al(P * kqt_stage_bytes(v41::MOE_INTER));   // k_kqt staged rows
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
bool kq_soa() { static const bool on = [] { const char* e = getenv("V41_KQ_SOA"); return kq_fast() && !(e && *e == '0'); }(); return on; }
__global__ void k_kq_soa(uint8_t* dst, const uint8_t* src, size_t cnt, int bits) {
    const size_t bi = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (bi >= cnt) return;
    if (bits == 2) {                                  // scales[16] qs[64] d,dmin[4] -> qs | scales | d
        const uint8_t* b = src + bi * 84;
        for (int i = 0; i < 64; ++i) dst[bi * 64 + i] = b[16 + i];
        for (int i = 0; i < 16; ++i) dst[64 * cnt + bi * 16 + i] = b[i];
        for (int i = 0; i < 4; ++i) dst[80 * cnt + bi * 4 + i] = b[80 + i];
    } else {                                          // hmask[32] qs[64] scales[12] d[2] -> same order, planar
        const uint8_t* b = src + bi * 110;
        for (int i = 0; i < 32; ++i) dst[bi * 32 + i] = b[i];
        for (int i = 0; i < 64; ++i) dst[32 * cnt + bi * 64 + i] = b[32 + i];
        for (int i = 0; i < 12; ++i) dst[96 * cnt + bi * 12 + i] = b[96 + i];
        for (int i = 0; i < 2; ++i) dst[108 * cnt + bi * 2 + i] = b[108 + i];
    }
}
void kq_soa_repack(uint8_t* W, int rows, int K, int bits, void* tmp, cudaStream_t st) {
    const size_t cnt = (size_t)rows * (K / 256), bytes = cnt * (bits == 2 ? 84 : 110);
    k_kq_soa<<<(unsigned)((cnt + 255) / 256), 256, 0, st>>>((uint8_t*)tmp, W, cnt, bits);
    cudaMemcpyAsync(W, tmp, bytes, cudaMemcpyDeviceToDevice, st);   // same stream: ordered after the kernel
}
// fp8 tensor-core path (v41_kq_mma.cuh): ~200 GB/s on every Q2_K/Q3_K expert shape in gate_kq_fast vs
// 110-148 for the SOA half2 kernel. V41_KQ_MMA=0 keeps the SOA layout and kernel.
bool kq_mma() { static const bool on = [] { const char* e = getenv("V41_KQ_MMA"); return kq_fast() && !(e && *e == '0'); }(); return on; }
void kq_mma_repack(uint8_t* W, int rows, int K, int bits, void* tmp, cudaStream_t st) {
    const size_t th = (size_t)(rows / 16) * (K / 256) * 32, bytes = (size_t)rows * (K / 256) * (bits == 2 ? 84 : 110);
    k_kq_mma_repack<<<(unsigned)((th + 255) / 256), 256, 0, st>>>((uint8_t*)tmp, W, rows, K / 256, bits);
    cudaMemcpyAsync(W, tmp, bytes, cudaMemcpyDeviceToDevice, st);
}
// gate_kqm_rt (2026-10-30), RT 4 vs 1, bitwise: Q2_K 1.03x gate|up, 1.11x down; Q3_K 0.88x / 0.82x -> Q2_K only
static constexpr int KQM_RT_DEFAULT = 4;
template <int QT, int MT, int RT = 1>
static void launch_kqm_(float* out, int ldo, const XSrc& X, bool by_pair, const Unit* units, const int* nu, const int* ptok,
                        const KQTab& tab, int N, int K, int max_units, cudaStream_t st, int cmin, int cmax) {
    dim3 grid((N + KQM_ROWS_PER_CTA * RT - 1) / (KQM_ROWS_PER_CTA * RT), max_units);
    const size_t smem = (size_t)MT * kqm_stage_bytes(K);
    auto kf = k_kqm<QT, MT, Unit, RT>;
    if (smem + 1024 > 48 * 1024) cudaFuncSetAttribute(kf, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    kf<<<grid, 256, smem, st>>>(out, ldo, X, by_pair, units, nu, ptok, tab, N, K, cmin, cmax);
}
// Q2_K prefill tiles (MT >= 8) run V41_KQM_RT row tiles per warp; bitwise for every RT
template <int QT, int MT>
static void launch_kqm(float* out, int ldo, const XSrc& X, bool by_pair, const Unit* units, const int* nu, const int* ptok,
                       const KQTab& tab, int N, int K, int max_units, cudaStream_t st, int cmin = 1, int cmax = 1 << 30) {
    static const int rt = [] { const char* e = getenv("V41_KQM_RT"); return e ? atoi(e) : KQM_RT_DEFAULT; }();
    if (QT == 2 && MT >= 8 && rt == 2) launch_kqm_<QT, MT, 2>(out, ldo, X, by_pair, units, nu, ptok, tab, N, K, max_units, st, cmin, cmax);
    else if (QT == 2 && MT >= 8 && rt == 4) launch_kqm_<QT, MT, 4>(out, ldo, X, by_pair, units, nu, ptok, tab, N, K, max_units, st, cmin, cmax);
    else launch_kqm_<QT, MT, 1>(out, ldo, X, by_pair, units, nu, ptok, tab, N, K, max_units, st, cmin, cmax);
}
template <int QT, int MT, bool SOA>
static void launch_kq(float* out, int ldo, const XSrc& X, bool by_pair, const Unit* units, const int* nu, const int* ptok,
                      const KQTab& tab, int N, int K, int max_units, cudaStream_t st, int cmin = 1, int cmax = 1 << 30) {
    dim3 grid((N + ROWS_PER_CTA - 1) / ROWS_PER_CTA, max_units);
    const size_t smem = (size_t)MT * kq_stage_bytes(K);
    // 16 lanes x 4 rows per lane where K/32 leaves a 32-lane stride underfilled (the down pass, K = 2304:
    // Q2 126 -> 136, Q3 112 -> 115 GB/s in gate_kq_fast). Same 64-row CTA. Chosen by K only (AR == verify).
    static const int lpr_env = [] { const char* e = getenv("V41_KQ_LPR"); return e ? atoi(e) : 0; }();
    const bool l16 = lpr_env ? lpr_env == 16 : (K / 32) % 32 != 0;
    static_assert(kq_rows_per_cta(16, 4) == ROWS_PER_CTA, "grid below assumes 64-row CTAs");
    auto kf = l16 ? k_kq<QT, MT, Unit, SOA, 16, 4> : k_kq<QT, MT, Unit, SOA>;
    if (smem + 1024 > 48 * 1024) cudaFuncSetAttribute(kf, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    kf<<<grid, 256, smem, st>>>(out, ldo, X, by_pair, units, nu, ptok, tab, N, K, cmin, cmax);
}
// Verify widths (M <= KQ_DUAL_M): most experts are hit by one row, and a 1-row unit in the MT = 4
// kernel stages and dots 4 rows at 1 CTA/SM. Those units go to the MT = 1 kernel (same per-row
// arithmetic, so the sums are unchanged) and only the shared ones to MT = 4. Prefill keeps one launch.
static constexpr int KQ_DUAL_M = 8;
static int g_kq_m = 1;                                // rows of the moe_routed call being dispatched
template <bool SOA>
static void run_kq_(int bits, float* out, int ldo, const XSrc& X, bool by_pair, const Unit* units, const int* nu,
                    const int* ptok, const KQTab& tab, int N, int K, int max_units, int mt, cudaStream_t st) {
    static const bool dual = [] { const char* e = getenv("V41_KQ_DUAL"); return !(e && *e == '0'); }();
    if (mt > 1 && dual && g_kq_m <= KQ_DUAL_M) {
        if (bits == 2) { launch_kq<2, 1, SOA>(out, ldo, X, by_pair, units, nu, ptok, tab, N, K, max_units, st, 1, 1);
                         launch_kq<2, UNIT_MT, SOA>(out, ldo, X, by_pair, units, nu, ptok, tab, N, K, max_units, st, 2); }
        else           { launch_kq<3, 1, SOA>(out, ldo, X, by_pair, units, nu, ptok, tab, N, K, max_units, st, 1, 1);
                         launch_kq<3, UNIT_MT, SOA>(out, ldo, X, by_pair, units, nu, ptok, tab, N, K, max_units, st, 2); }
        return;
    }
    if (bits == 2) { if (mt == 1) launch_kq<2, 1, SOA>(out, ldo, X, by_pair, units, nu, ptok, tab, N, K, max_units, st);
                     else launch_kq<2, UNIT_MT, SOA>(out, ldo, X, by_pair, units, nu, ptok, tab, N, K, max_units, st); }
    else           { if (mt == 1) launch_kq<3, 1, SOA>(out, ldo, X, by_pair, units, nu, ptok, tab, N, K, max_units, st);
                     else launch_kq<3, UNIT_MT, SOA>(out, ldo, X, by_pair, units, nu, ptok, tab, N, K, max_units, st); }
}
static void run_kqm(int bits, float* out, int ldo, const XSrc& X, bool by_pair, const Unit* units, const int* nu,
                    const int* ptok, const KQTab& tab, int N, int K, int max_units, int mt, cudaStream_t st) {
    static const bool dual = [] { const char* e = getenv("V41_KQ_DUAL"); return !(e && *e == '0'); }();
    if (mt > 1 && dual && g_kq_m <= KQ_DUAL_M) {
        if (bits == 2) { launch_kqm<2, 1>(out, ldo, X, by_pair, units, nu, ptok, tab, N, K, max_units, st, 1, 1);
                         launch_kqm<2, UNIT_MT>(out, ldo, X, by_pair, units, nu, ptok, tab, N, K, max_units, st, 2); }
        else           { launch_kqm<3, 1>(out, ldo, X, by_pair, units, nu, ptok, tab, N, K, max_units, st, 1, 1);
                         launch_kqm<3, UNIT_MT>(out, ldo, X, by_pair, units, nu, ptok, tab, N, K, max_units, st, 2); }
        return;
    }
    if (mt == 8) {
        if (bits == 2) launch_kqm<2, 8>(out, ldo, X, by_pair, units, nu, ptok, tab, N, K, max_units, st);
        else launch_kqm<3, 8>(out, ldo, X, by_pair, units, nu, ptok, tab, N, K, max_units, st);
        return;
    }
    if (mt == 16) {                                   // 9..16-token units share one expert read; the rest stage 8
        if (bits == 2) { launch_kqm<2, 8>(out, ldo, X, by_pair, units, nu, ptok, tab, N, K, max_units, st, 1, 8);
                         launch_kqm<2, 16>(out, ldo, X, by_pair, units, nu, ptok, tab, N, K, max_units, st, 9); }
        else           { launch_kqm<3, 8>(out, ldo, X, by_pair, units, nu, ptok, tab, N, K, max_units, st, 1, 8);
                         launch_kqm<3, 16>(out, ldo, X, by_pair, units, nu, ptok, tab, N, K, max_units, st, 9); }
        return;
    }
    if (bits == 2) { if (mt == 1) launch_kqm<2, 1>(out, ldo, X, by_pair, units, nu, ptok, tab, N, K, max_units, st);
                     else launch_kqm<2, UNIT_MT>(out, ldo, X, by_pair, units, nu, ptok, tab, N, K, max_units, st); }
    else           { if (mt == 1) launch_kqm<3, 1>(out, ldo, X, by_pair, units, nu, ptok, tab, N, K, max_units, st);
                     else launch_kqm<3, UNIT_MT>(out, ldo, X, by_pair, units, nu, ptok, tab, N, K, max_units, st); }
}
static void run_kq(int bits, bool soa, float* out, int ldo, const XSrc& X, bool by_pair, const Unit* units, const int* nu,
                   const int* ptok, const KQTab& tab, int N, int K, int max_units, int mt, cudaStream_t st) {
    if (soa) run_kq_<true>(bits, out, ldo, X, by_pair, units, nu, ptok, tab, N, K, max_units, mt, st);
    else run_kq_<false>(bits, out, ldo, X, by_pair, units, nu, ptok, tab, N, K, max_units, mt, st);
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
        if ((bits == 2 || bits == 3) && T.kq_mma) {     // tensor-core layout: only k_kqm can read it
            if (!X.xq) { fprintf(stderr, "kq_mma needs fp8 activations\n"); abort(); }
            run_kqm(bits, out, ldo, X, by_pair, units, nu, ptok, KQTab{T.w1, T.w3, T.w2, pass}, N, K, max_units, mt, st);
            break;
        }
        if ((bits == 2 || bits == 3) && T.kq_soa) {     // reordered bytes: only the SOA kernel can read them
            if (!X.xq) { fprintf(stderr, "kq_soa needs fp8 activations\n"); abort(); }
            run_kq(bits, true, out, ldo, X, by_pair, units, nu, ptok, KQTab{T.w1, T.w3, T.w2, pass}, N, K, max_units, mt, st);
            break;
        }
        if ((bits == 2 || bits == 3) && X.xq && kq_fast()) {
            run_kq(bits, false, out, ldo, X, by_pair, units, nu, ptok, KQTab{T.w1, T.w3, T.w2, pass}, N, K, max_units, mt, st);
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
    // Prefill on the tensor-core path: an mma tile holds 8 tokens, so 8-token units halve the expert weight
    // reads. Per-element arithmetic does not depend on the unit size, so prefill still equals decode.
    static const bool mt8 = [] { const char* e = getenv("V41_KQ_MT8"); return !(e && *e == '0'); }();
    // 16-token units (k_kqm NT = 2): at chunk 512 an expert averages 16 tokens, read once instead of twice
    static const int mtp = [] { const char* e = getenv("V41_KQ_MTP"); return e ? atoi(e) : 16; }();
    const bool all_mma = T.kq_mma && T.fmt13 == ExpertFmt::KQ && T.fmt2 == ExpertFmt::KQ &&
                         (T.bits13 == 2 || T.bits13 == 3) && (T.bits2 == 2 || T.bits2 == 3);
    // P8: Q2_K layers on tcgen05 (k_kqt) for units of V41_KQT_MIN..64 tokens, k_kqm below (units of <= 64).
    // Default on (V41_KQT=0 off): 8k prefill 291.1 -> 328.1 tok/s at chunk 512, T1 (KL vs default at the
    // default-vs-exact level on the 4-offset protocol with every unit forced onto k_kqt, NLL unchanged)
    static const bool kqt_env = [] { const char* e = getenv("V41_KQT"); return !(e && *e == '0'); }();
    static const int kqt_min = [] { const char* e = getenv("V41_KQT_MIN"); const int v = e ? atoi(e) : 9; return v < 1 ? 1 : v > 17 ? 17 : v; }();
    const bool kqt = kqt_env && all_mma && M > KQ_DUAL_M && T.fmt13 == ExpertFmt::KQ;   // all_mma: bits 2 / 3, both passes
    const int P = M * topk, mt = (M == 1) ? 1 : kqt ? 64 : (mt8 && all_mma && M > KQ_DUAL_M) ? (mtp == 16 ? 16 : 8) : UNIT_MT;
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
    float* xr2 = (float*)b; b += al((size_t)P * v41::MOE_INTER * 4);
    uint8_t* xt13 = (uint8_t*)b; b += al((size_t)M * kqt_stage_bytes(v41::DIM));
    uint8_t* xt2 = (uint8_t*)b;
    // k_kqm for the small units (cnt < kqt_min), k_kqt for the rest
    auto kqt_pass = [&](int pass, float* out, int ldo, const XSrc& Xp, const uint8_t* stg, bool by_pair, int N, int K) {
        const KQTab tab{T.w1, T.w3, T.w2, pass};
        const int bits = pass == 0 ? T.bits13 : T.bits2;
        if (kqt_min > 1) {
            if (bits == 2) launch_kqm<2, 8>(out, ldo, Xp, by_pair, units, nu, ptok, tab, N, K, P, st, 1, std::min(8, kqt_min - 1));
            else launch_kqm<3, 8>(out, ldo, Xp, by_pair, units, nu, ptok, tab, N, K, P, st, 1, std::min(8, kqt_min - 1));
        }
        if (kqt_min > 9) {
            if (bits == 2) launch_kqm<2, 16>(out, ldo, Xp, by_pair, units, nu, ptok, tab, N, K, P, st, 9, kqt_min - 1);
            else launch_kqm<3, 16>(out, ldo, Xp, by_pair, units, nu, ptok, tab, N, K, P, st, 9, kqt_min - 1);
        }
        if (!kqt_run(bits, out, ldo, stg, by_pair, units, nu, ptok, T.w1, T.w3, T.w2, pass, N, K, kqt_min, st)) {
            fprintf(stderr, "kqt_run: not applicable (N %d K %d)\n", N, K); abort();
        }
    };
    static const bool units_serial = [] { const char* e = getenv("V41_UNITS_SERIAL"); return e && *e == '1'; }();
    if (P <= UNITS_PAR_MAX && !units_serial) k_units_par<<<1, UNITS_PAR_MAX, 0, st>>>(units, nu, ptok, pslot, inv, idx, M, topk, mt);
    else if (P <= UNITS_BIG_MAX && !units_serial && units_big(units, nu, ptok, pslot, inv, idx, M, topk, mt, st)) {}
    else k_units<<<1, 1, 0, st>>>(units, nu, ptok, pslot, inv, idx, M, topk, mt);
    g_kq_m = M;
    k_pair_w<<<(P + 127) / 128, 128, 0, st>>>(wp, w, ptok, pslot, P, topk);
    XSrc X; X.xq = xq; X.xs = xs; X.ldx = v41::DIM;
    if (T.fmt13 == ExpertFmt::KT) {               // rotate the token rows once; every unit stages them
        k_rht_rows<<<dim3(v41::DIM / 256, M), 256, 0, st>>>(xr13, X, T.rht13, v41::DIM);
        X = XSrc(); X.xf = xr13; X.ldx = v41::DIM;
    }
    // prefill on the tensor-core path: stage x once per row (xr13 / xr2 are free: they hold KT rotations only)
    static const bool prep_env = [] { const char* e = getenv("V41_KQM_PREP"); return !(e && *e == '0'); }();
    const bool prep = prep_env && all_mma && M > KQ_DUAL_M && T.fmt13 == ExpertFmt::KQ;
    static_assert(v41g::kqm_stage_bytes(v41::DIM) <= (size_t)v41::DIM * 4 &&
                  v41g::kqm_stage_bytes(v41::MOE_INTER) <= (size_t)v41::MOE_INTER * 4, "staged rows fit the KT scratch");
    if (prep) { v41g::k_kqm_prep<<<M, 256, 0, st>>>((uint8_t*)xr13, X, v41::DIM); X.stg = (const uint8_t*)xr13; }
    if (kqt) {
        kqt_prep(T.bits13, xt13, X.xq, X.xs, X.ldx, M, v41::DIM, st);
        kqt_pass(0, gu, 2 * v41::MOE_INTER, X, xt13, false, 2 * v41::MOE_INTER, v41::DIM);
    } else dispatch_pass(0, T, gu, 2 * v41::MOE_INTER, X, false, units, nu, ptok, 2 * v41::MOE_INTER, v41::DIM, P, mt, st);
    swiglu(act, gu, gu + v41::MOE_INTER, wp, P, v41::MOE_INTER, v41::SWIGLU_LIMIT, 2 * v41::MOE_INTER, st);
    act_quant_fp8(aq, as, act, P, v41::MOE_INTER, st);
    XSrc A; A.xq = aq; A.xs = as; A.ldx = v41::MOE_INTER;
    if (T.fmt2 == ExpertFmt::KT) {
        k_rht_rows<<<dim3(v41::MOE_INTER / 256, P), 256, 0, st>>>(xr2, A, T.rht2, v41::MOE_INTER);
        A = XSrc(); A.xf = xr2; A.ldx = v41::MOE_INTER;
    }
    if (prep) { v41g::k_kqm_prep<<<P, 256, 0, st>>>((uint8_t*)xr2, A, v41::MOE_INTER); A.stg = (const uint8_t*)xr2; }
    if (kqt) {
        kqt_prep(T.bits2, xt2, A.xq, A.xs, A.ldx, P, v41::MOE_INTER, st);
        kqt_pass(1, down, v41::DIM, A, xt2, true, v41::DIM, v41::MOE_INTER);
    } else dispatch_pass(1, T, down, v41::DIM, A, true, units, nu, ptok, v41::DIM, v41::MOE_INTER, P, mt, st);
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

namespace v41k {
// k_units_par vs k_units on random routings (tests/gate_units.cu): mismatching cases, -1 on CUDA error
int units_selftest() {
    int bad = 0;
    unsigned seed = 12345;
    auto rnd = [&] { seed = seed * 1664525u + 1013904223u; return seed >> 8; };
    for (int topk : {6, 3})
        for (int mt : {1, 4, 8, 16, 64})
            for (int M : {1, 2, 3, 4, 5, 6, 8, 16, 64, 128, 200, 341, 512, 682, 683, 1024, 2048}) {
                const int P = M * topk; if (P > UNITS_BIG_MAX) continue;
                const bool big = P > UNITS_PAR_MAX;          // k_units_big against the serial reference
                for (int rep = 0; rep < (M >= 1024 ? 2 : 20); ++rep) {   // the serial reference is O(P^2)
                    const int E = rep % 2 ? 16 : 192;               // few experts: long runs
                    std::vector<int> h(P);
                    for (int m = 0; m < M; ++m)
                        for (int j = 0; j < topk; ++j) {             // distinct experts per token
                            int e; bool dup;
                            do { e = rnd() % E; dup = false; for (int k = 0; k < j; ++k) dup |= h[m * topk + k] == e; } while (dup);
                            h[m * topk + j] = e;
                        }
                    int* d; cudaMalloc(&d, (size_t)(P * 8 + 4) * 4 + 2 * (size_t)P * sizeof(Unit));
                    int* idx = d; int* buf[2][4]; Unit* un[2];
                    char* b = (char*)(d + P);
                    for (int k = 0; k < 2; ++k) { for (int i = 0; i < 4; ++i) { buf[k][i] = (int*)b; b += (i == 3 ? 4 : P * 4); }
                                                  un[k] = (Unit*)b; b += P * sizeof(Unit); }
                    cudaMemcpy(idx, h.data(), P * 4, cudaMemcpyHostToDevice);
                    k_units<<<1, 1>>>(un[0], buf[0][3], buf[0][0], buf[0][1], buf[0][2], idx, M, topk, mt);
                    if (big) {
                        if (!units_big(un[1], buf[1][3], buf[1][0], buf[1][1], buf[1][2], idx, M, topk, mt, 0)) { cudaFree(d); return -1; }
                    } else k_units_par<<<1, UNITS_PAR_MAX>>>(un[1], buf[1][3], buf[1][0], buf[1][1], buf[1][2], idx, M, topk, mt);
                    if (cudaDeviceSynchronize() != cudaSuccess) { cudaFree(d); return -1; }
                    std::vector<int> a(3 * P + 1), c(3 * P + 1);
                    for (int k = 0; k < 2; ++k) {
                        auto& o = k ? c : a;
                        for (int i = 0; i < 3; ++i) cudaMemcpy(o.data() + i * P, buf[k][i], P * 4, cudaMemcpyDeviceToHost);
                        cudaMemcpy(o.data() + 3 * P, buf[k][3], 4, cudaMemcpyDeviceToHost);
                    }
                    bool diff = a != c;
                    if (!diff) {
                        const int nu = a[3 * P];
                        std::vector<Unit> ua(nu), uc(nu);
                        cudaMemcpy(ua.data(), un[0], nu * sizeof(Unit), cudaMemcpyDeviceToHost);
                        cudaMemcpy(uc.data(), un[1], nu * sizeof(Unit), cudaMemcpyDeviceToHost);
                        for (int i = 0; i < nu; ++i) diff |= ua[i].e != uc[i].e || ua[i].start != uc[i].start || ua[i].cnt != uc[i].cnt;
                    }
                    if (diff && bad < 5) printf("units mismatch: topk %d mt %d M %d rep %d%s\n", topk, mt, M, rep, big ? " (big)" : "");
                    bad += diff;
                    cudaFree(d);
                }
            }
    return bad;
}
} // namespace v41k
