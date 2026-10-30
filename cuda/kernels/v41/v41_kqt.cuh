// v41_kqt.cuh — prefill Q2_K / Q3_K routed experts on tcgen05 (UMMA, TMEM accumulators). P8 stage 2b.
//
// WHY. k_kqm (mma.sync) is issue-bound at ~11 TFLOPS: the per-16 scale epilogue and operand building run
// per mma and per token tile. On tcgen05 the CTA decodes a 128-row x 256 (one superblock) weight tile ONCE
// into shared memory and one thread issues 16 UMMAs over all NT tokens of the unit, so the decode cost per
// token falls with NT and the tensor work leaves the issue slots.
//
// OPERANDS, exact integers in the A type, fp32 accumulation per superblock in TMEM:
//   Q2_K (kind bf16): A = bf16(128 + q*sc), q*sc <= 45 -> bits 0x4300 | v (IMUL + 2 PRMT per 4 codes)
//   Q3_K (kind f16):  A = f16(1280 + sc*c), sc*c in [-224, 217] -> bits 0x6500 + v (2 PRMT + 2 IMAD per 4 codes,
//                     two 16-bit lanes per IMAD; c = q | hbit << 2, weight sc * (c - 4))
// B = x rounded to the A type (the prep stage). The min / code-free term is a second small UMMA, Cmin =
// sum_sb A_m[sb] * xm[n][sb] with A_m = m (Q2) or sc (Q3) and xm split hi + lo. Per superblock each drain
// thread (one row = one TMEM lane) folds its columns:
//     Q2: acc[n] += d * (C[n] - 128 * xsum[n]) - dmin * Cmin[n]      Q3: acc[n] += d * (C[n] - 1280 * xsum[n] - 4 * Cmin[n])
// T1 vs k_kqm up to the rounding of x to bf16 / f16 (k_kqm multiplies e4m3 x by the fp32 scale after the mma).
// Weights in the kq_mma_repack layout, read as k_kqm reads them.
//
// SMEM (UMMA K-major SW128, bf16): element (row, k) of a 256-wide tile at swz(row * 64 + (k & 63) + (k >> 6) *
// 64 * ROWS), swz(o) = o ^ ((o >> 3) & 0x38): SW128 XORs byte bits 4-6 with 7-9 (cute applies Sw<3,4,3> to the BYTE address via
// smem_ptr_flag, so evaluating the layout shows no swizzle; in bf16 elements the mask is 0x38, not 0x70).
#pragma once
#include <cute/tensor.hpp>
#include <cute/atom/mma_traits_sm100.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cutlass/arch/barrier.h>
#include "v41_kq_mma.cuh"

namespace v41g {

__device__ __forceinline__ int kqt_swz(int o) { return o ^ ((o >> 3) & 0x38); }

__device__ __forceinline__ int kqt_swz64(int o) { return o ^ ((o >> 3) & 0x18); }   // SW64, bf16 elements, 32-wide rows
constexpr int KQT_PROD = 256;                      // producer warps 0..7 (decode, B, UMMA issue by warp 0)
constexpr int KQT_THREADS = KQT_PROD + 128;         // + drain warps 8..11 (TMEM lane quadrant = warp % 4)
#ifndef KQT_DBG
#define KQT_DBG 0                                  // timing breakdown only: 2 B tile loads, 8 drain math, 16 UMMA skipped
#endif

// operand type per K-quant: Q2_K bf16, Q3_K f16 (1280 + sc*c needs 11 significant bits)
template <int QT> struct KqtT;
template <> struct KqtT<2> {
    using T = cutlass::bfloat16_t;
    static __device__ __forceinline__ uint16_t enc(float v) { return __bfloat16_as_ushort(__float2bfloat16_rn(v)); }
    static __device__ __forceinline__ float dec(uint16_t b) { return __bfloat162float(__ushort_as_bfloat16(b)); }
};
template <> struct KqtT<3> {
    using T = cutlass::half_t;
    static __device__ __forceinline__ uint16_t enc(float v) { return __half_as_ushort(__float2half_rn(v)); }
    static __device__ __forceinline__ float dec(uint16_t b) { return __half2float(__ushort_as_half(b)); }
};

// prestaged x rows: K x (natural order) | per superblock 16 xm_hi + 16 xm_lo | K/256 fp32 xsum, in the QT
// operand type (xsum and xm sum the rounded x the MMA sees, so the offset * xsum cancels; hi + lo carries xm
// to ~2^-17 (bf16) / ~2^-22 (f16))
__host__ __device__ constexpr size_t kqt_stage_bytes(int K) { return ((size_t)K * 2 + (size_t)K / 4 + (size_t)K / 64 + 15) & ~(size_t)15; }
template <int QT>
static __global__ void k_kqt_prep(uint8_t* dst, XSrc X, int K) {
    using E = KqtT<QT>;
    uint8_t* d = dst + (size_t)blockIdx.x * kqt_stage_bytes(K);
    const uint8_t* q = X.xq + (size_t)blockIdx.x * X.ldx;
    const float* xs = X.xs + (size_t)blockIdx.x * (X.ldx / 32);
    uint16_t* xb = reinterpret_cast<uint16_t*>(d);
    uint16_t* xm = xb + K;
    float* xsum = reinterpret_cast<float*>(d + (size_t)K * 2 + (size_t)K / 4);
    auto xv = [&](int k) { return E::dec(E::enc(e4m3f(q[k]) * xs[k >> 5])); };
    for (int k = threadIdx.x; k < K; k += blockDim.x) xb[k] = E::enc(xv(k));
    for (int g = threadIdx.x; g < K / 16; g += blockDim.x) {
        float m = 0.f;
        for (int i = 0; i < 16; ++i) m += xv(g * 16 + i);
        const uint16_t hi = E::enc(m);
        xm[(g >> 4) * 32 + (g & 15)] = hi;
        xm[(g >> 4) * 32 + 16 + (g & 15)] = E::enc(m - E::dec(hi));
    }
    for (int s = threadIdx.x; s < K / 256; s += blockDim.x) {
        float a = 0.f;
        for (int i = 0; i < 256; ++i) a += xv(s * 256 + i);
        xsum[s] = a;
    }
}

template <int NT> struct KqtStage {                // 2-byte operands (bf16 or f16)
    alignas(1024) uint16_t A[128 * 256];           // offset + code * scale, K-major SW128
    alignas(1024) uint16_t B[NT * 256];            // x
    alignas(1024) uint16_t Am[128 * 32];           // m / sc per sub-block twice (hi, lo), K-major SW64
    alignas(1024) uint16_t Bm[NT * 32];            // xm_hi[16] | xm_lo[16]
};
template <int NT> struct KqtSmem {
    KqtStage<NT> st[2];
    float xsum[4][NT];                                 // superblock s at s % 4: B(s + 4) follows UMMA(s + 2), which waits on drain(s)
    alignas(16) uint64_t mbar[2];                      // UMMA(s) complete (umma_arrive): stage and TMEM pair s & 1 readable
    alignas(16) uint64_t tfree[2];                     // drain(s) done (128 arrivals): TMEM pair s & 1 writable
    alignas(16) uint32_t tmem;
};

// Q3_K 6-bit signed scale of sub-block sb from a row's 12 ggml bytes (x, y, z words), as k_kqm decodes it
__device__ __forceinline__ int kqt_q3_sc(const uint4& w, int sb) {
    const int t = sb >> 2, i = sb & 3;
    const uint32_t lo = (t & 1 ? w.y : w.x) >> (4 * (t >> 1)), hi = w.z >> (2 * t);
    return (int)(((lo >> (8 * i)) & 0xF) | (((hi >> (8 * i)) & 3) << 4)) - 32;
}
__device__ __forceinline__ void kqt_cp16(void* s, const void* g) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"((uint32_t)__cvta_generic_to_shared(s)), "l"(g) : "memory");
}

// grid (N / 128, GY), KQT_THREADS, persistent over units u = blockIdx.y, + GY, ... (< *units_n) with
// cmin <= cnt <= cmax and cnt <= NT; X.stg = k_kqt_prep rows. Per unit only ceil16(cnt) token columns are
// staged and drained (the UMMA runs all NT: a column of C depends on its own B row only, so stale rows stay
// in unused columns). Warp-specialized pipeline over the superblocks g of all units of the CTA, two smem
// stages and two TMEM accumulator pairs (C, Cmin):
//   producers (warps 0..7): decode A(g) -> named barrier -> warp 0: [drain(g - 2) done] issue UMMA(g) ->
//                           [UMMA(g - 1) done] cp.async B(g + 1) into the stage it freed
//   drain (warps 8..11):    [mbar[g & 1]] tcgen05.ld C, Cmin -> fold d, dmin -> arrive tfree[g & 1]
// so the decode of g + 1, the UMMA of g and the drain of g - 1 run at once.
template <int QT, int NT, class UnitT>
__global__ void __launch_bounds__(KQT_THREADS, 1) k_kqt(float* __restrict__ out, int ldo, XSrc X, bool x_by_pair,
                                                        const UnitT* units, const int* units_n, const int* pair_tok,
                                                        KQTab tab, int N, int K, int cmin, int cmax) {
    using namespace cute;
    using T = typename KqtT<QT>::T;
    static_assert(QT == 2 || QT == 3, "Q2_K / Q3_K");
    static_assert(NT % 16 == 0 && NT * 4 <= KQT_PROD, "NT");
    const int nunits = *units_n;
    {                                                      // any work at all for this CTA?
        bool any = false;
        for (int u = blockIdx.y; u < nunits && !any; u += gridDim.y) any = units[u].cnt >= cmin && units[u].cnt <= cmax;
        if (!any) return;
    }
    extern __shared__ __align__(1024) uint8_t kqt_raw[];
    KqtSmem<NT>& sm = *reinterpret_cast<KqtSmem<NT>*>(kqt_raw);
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int row0 = blockIdx.x * 128;
    const int nsb = K / 256;
    const size_t SB = kqt_stage_bytes(K);
    const bool up = tab.pass == 0 && row0 >= v41::MOE_INTER;
    const int lt0 = (up ? row0 - v41::MOE_INTER : row0) / 16, mrows = tab.pass == 0 ? v41::MOE_INTER : N;
    const size_t cnt = (size_t)(mrows / 16) * nsb;

    TiledMMA mma = make_tiled_mma(SM100_MMA_F16BF16_SS<T, T, float, 128, NT, UMMA::Major::K, UMMA::Major::K>{});
    auto sA_l = UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<T>{}, partition_shape_A(mma, make_shape(Int<128>{}, Int<256>{})));
    auto sB_l = UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<T>{}, partition_shape_B(mma, make_shape(Int<NT>{}, Int<256>{})));
    auto sAm_l = UMMA::tile_to_mma_shape(UMMA::Layout_K_SW64_Atom<T>{}, partition_shape_A(mma, make_shape(Int<128>{}, Int<32>{})));
    auto sBm_l = UMMA::tile_to_mma_shape(UMMA::Layout_K_SW64_Atom<T>{}, partition_shape_B(mma, make_shape(Int<NT>{}, Int<32>{})));
    ThrMMA cta_mma = mma.get_slice(0);
    Tensor gC = make_tensor(make_gmem_ptr((float*)nullptr), make_layout(make_shape(Int<128>{}, Int<NT>{}), make_stride(Int<NT>{}, Int<1>{})));
    Tensor tCtAcc = cta_mma.make_fragment_C(cta_mma.partition_C(gC));

    constexpr int TCOLS = 4 * NT < 32 ? 32 : 4 * NT;          // [buf][C | Cmin] x NT columns
    cute::TMEM::Allocator1Sm talloc{};
    if (warp == 0) talloc.allocate(TCOLS, &sm.tmem);
    if (tid == 0) {
        cute::initialize_barrier(sm.mbar[0], 1); cute::initialize_barrier(sm.mbar[1], 1);
        cute::initialize_barrier(sm.tfree[0], 128); cute::initialize_barrier(sm.tfree[1], 128);
    }
    cutlass::arch::fence_barrier_init();
    __syncthreads();
    const uint32_t tmem0 = sm.tmem;
    const int er = (warp & 3) * 32 + lane;                 // drain row (warps 8..11) = TMEM lane
    const int et = lt0 + er / 16, eg = er % 16;
    constexpr int NBR = (NT + KQT_PROD / 32 - 1) / (KQT_PROD / 32);
    struct APre { uint4 ca, cb, s0, s1, m, h; };      // Q3: s0 / s1 / m hold 12-byte scale rows (x, y, z)

    int g0 = 0;                                            // global superblock index of this unit's s = 0
    for (int u = blockIdx.y; u < nunits; u += gridDim.y) {
        const UnitT U = units[u];
        if (U.cnt < cmin || U.cnt > cmax) continue;
        const int ncol = (U.cnt + 15) & ~15;
        const uint8_t* W = tab.pass == 0 ? (up ? tab.w3[U.e] : tab.w1[U.e]) : tab.w2[U.e];
        auto trow = [&](int n) { const int p = U.start + min(n, U.cnt - 1); return X.stg + (size_t)(x_by_pair ? p : pair_tok[p]) * SB; };

        if (warp < KQT_PROD / 32) {
            // ---------------- producers
            const uint8_t* brow[NBR];                      // token rows: B rows n = warp + 8 i, Bm row tid / 4, xsum row tid
            #pragma unroll
            for (int i = 0; i < NBR; ++i) brow[i] = trow(min(warp + i * (KQT_PROD / 32), NT - 1));
            const uint8_t* bmrow = trow(min(tid >> 2, NT - 1)) + (size_t)K * 2;
            const uint8_t* xsrow = trow(min(tid, NT - 1)) + (size_t)K * 2 + K / 4;
            auto produce_b = [&](int s) {                  // B, Bm, xsum by cp.async into stage g & 1 (one commit group)
                const int g = g0 + s;
                KqtStage<NT>& S = sm.st[g & 1];
                uint16_t* Bs = reinterpret_cast<uint16_t*>(S.B);
                uint16_t* Bms = reinterpret_cast<uint16_t*>(S.Bm);
                const int k = lane * 8;                    // 16 B = 8 elements, each to its swizzled chunk
                #pragma unroll
                for (int i = 0; i < NBR; ++i) {
                    const int n = warp + i * (KQT_PROD / 32);
                    if (n < ncol && !(KQT_DBG & 2)) kqt_cp16(Bs + kqt_swz(n * 64 + (k & 63) + (k >> 6) * 64 * NT), brow[i] + ((size_t)s * 256 + k) * 2);
                }
                if (tid < ncol * 4) kqt_cp16(Bms + kqt_swz64(tid * 8), bmrow + ((size_t)s * 32 + (tid & 3) * 8) * 2);   // (tid / 4, 8 (tid % 4))
                if (tid < ncol)
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"r"((uint32_t)__cvta_generic_to_shared(&sm.xsum[g % 4][tid])),
                                 "l"(reinterpret_cast<const float*>(xsrow) + s) : "memory");
                asm volatile("cp.async.commit_group;" ::: "memory");
            };
            // weight bytes of superblock s for this thread's decode (codes, sub-block scales) and Am row, loaded
            // two superblocks ahead so their latency hides under a whole iteration
            auto load_a = [&](int s, APre& P) {
                const int t16 = tid >> 5, l = tid & 31, gq = l >> 2;
                const size_t idx = (size_t)(lt0 + t16) * nsb + s;
                const uint4* c4 = reinterpret_cast<const uint4*>(W + idx * 1024 + l * 32);
                P.ca = __ldg(c4); P.cb = __ldg(c4 + 1);
                if (QT == 2) {
                    P.s0 = __ldg(reinterpret_cast<const uint4*>(W + 1024 * cnt + idx * 256 + gq * 16));
                    P.s1 = __ldg(reinterpret_cast<const uint4*>(W + 1024 * cnt + idx * 256 + (gq + 8) * 16));
                    if (tid >= 128) P.m = __ldg(reinterpret_cast<const uint4*>(W + 1024 * cnt + ((size_t)(lt0 + (tid - 128) / 16) * nsb + s) * 256 + (tid % 16) * 16));
                } else {
                    P.h = __ldg(reinterpret_cast<const uint4*>(W + 1024 * cnt + idx * 512 + l * 16));
                    auto s12 = [&](size_t i, int r, uint4& o) {
                        const uint32_t* w = reinterpret_cast<const uint32_t*>(W + 1536 * cnt + i * 192 + r * 12);
                        o.x = __ldg(w); o.y = __ldg(w + 1); o.z = __ldg(w + 2);
                    };
                    s12(idx, gq, P.s0); s12(idx, gq + 8, P.s1);
                    if (tid >= 128) s12((size_t)(lt0 + (tid - 128) / 16) * nsb + s, tid % 16, P.m);
                }
            };
            auto produce_a = [&](int s, const APre& P) {   // A, Am by the threads; then B(s) landed and fenced
                KqtStage<NT>& S = sm.st[(g0 + s) & 1];
                uint16_t* As = reinterpret_cast<uint16_t*>(S.A);
                {                                          // 8 tiles of 16 rows x 32 lanes, one (tile, lane) pair per thread
                    const int t16 = tid >> 5, l = tid & 31, gq = l >> 2, t = l & 3;
                    const uint32_t cw[8] = {P.ca.x, P.ca.y, P.ca.z, P.ca.w, P.cb.x, P.cb.y, P.cb.z, P.cb.w};
                    const uint4 sv[2] = {P.s0, P.s1};
                    const uint32_t hw[4] = {P.h.x, P.h.y, P.h.z, P.h.w};
                    #pragma unroll
                    for (int sb = 0; sb < 16; ++sb)
                        #pragma unroll
                        for (int h = 0; h < 2; ++h) {
                            uint32_t codes = (cw[sb >> 1] >> (2 * (2 * (sb & 1) + h))) & 0x03030303u;
                            const int row = t16 * 16 + gq + 8 * h, k = 16 * sb + 4 * t;
                            uint2 w2;
                            if (QT == 2) {
                                const uint32_t sw = sb < 4 ? sv[h].x : sb < 8 ? sv[h].y : sb < 12 ? sv[h].z : sv[h].w;
                                const uint32_t v = codes * ((sw >> (8 * (sb & 3))) & 0xF);
                                w2.x = __byte_perm(v, 0x43434343u, 0x5140); w2.y = __byte_perm(v, 0x43434343u, 0x7362);
                            } else {
                                codes |= ((hw[sb >> 2] >> (2 * (sb & 3) + h)) & 0x01010101u) << 2;
                                w2.x = __byte_perm(codes, 0u, 0x4140) * (uint32_t)kqt_q3_sc(sv[h], sb) + 0x65006500u;
                                w2.y = __byte_perm(codes, 0u, 0x4342) * (uint32_t)kqt_q3_sc(sv[h], sb) + 0x65006500u;
                            }
                            *reinterpret_cast<uint2*>(As + kqt_swz(row * 64 + (k & 63) + (k >> 6) * 64 * 128)) = w2;
                        }
                }
                if (tid >= 128) {                          // Am: row r's 16 min scales (m <= 15, exact in bf16), twice
                    const int r = tid - 128;               // (against xm_hi and xm_lo); warps 4..7, warp 0 issues UMMAs
                    const uint32_t mw[4] = {P.m.x, P.m.y, P.m.z, P.m.w};
                    uint16_t* Ams = reinterpret_cast<uint16_t*>(S.Am);
                    auto mval = [&](int sb) {              // Q2: min scale m (0..15); Q3: scale sc (-32..31)
                        return QT == 2 ? (float)((mw[sb >> 2] >> (8 * (sb & 3) + 4)) & 0xF) : (float)kqt_q3_sc(P.m, sb);
                    };
                    #pragma unroll
                    for (int c = 0; c < 2; ++c) {          // 8 sub-blocks per 16 B chunk
                        uint32_t pk[4];
                        #pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            const int sb = c * 8 + 2 * j;
                            pk[j] = (uint32_t)KqtT<QT>::enc(mval(sb)) | (uint32_t)KqtT<QT>::enc(mval(sb + 1)) << 16;
                        }
                        const uint4 v4 = make_uint4(pk[0], pk[1], pk[2], pk[3]);
                        *reinterpret_cast<uint4*>(Ams + kqt_swz64(r * 32 + c * 8)) = v4;
                        *reinterpret_cast<uint4*>(Ams + kqt_swz64(r * 32 + 16 + c * 8)) = v4;
                    }
                }
                asm volatile("cp.async.wait_all;" ::: "memory");  // the one group in flight is B(s)
                cutlass::arch::fence_view_async_shared();  // generic-proxy and cp.async smem writes -> UMMA
            };
            auto issue = [&](int g) {                      // the whole warp 0: gemm / umma_arrive elect a lane inside
                KqtStage<NT>& S = sm.st[g & 1];
                Tensor tCrA = cta_mma.make_fragment_A(make_tensor(make_smem_ptr(reinterpret_cast<T*>(S.A)), sA_l));
                Tensor tCrB = cta_mma.make_fragment_B(make_tensor(make_smem_ptr(reinterpret_cast<T*>(S.B)), sB_l));
                Tensor tCrAm = cta_mma.make_fragment_A(make_tensor(make_smem_ptr(reinterpret_cast<T*>(S.Am)), sAm_l));
                Tensor tCrBm = cta_mma.make_fragment_B(make_tensor(make_smem_ptr(reinterpret_cast<T*>(S.Bm)), sBm_l));
                Tensor c = tCtAcc; c.data() = tmem0 + (g & 1) * 2 * NT;
                Tensor cm = tCtAcc; cm.data() = tmem0 + (g & 1) * 2 * NT + NT;
                mma.accumulate_ = UMMA::ScaleOut::Zero;
                #pragma unroll
                for (int kb = 0; kb < 16; ++kb) { gemm(mma, tCrA(_, _, kb), tCrB(_, _, kb), c); mma.accumulate_ = UMMA::ScaleOut::One; }
                mma.accumulate_ = UMMA::ScaleOut::Zero;
                #pragma unroll
                for (int kb = 0; kb < 2; ++kb) { gemm(mma, tCrAm(_, _, kb), tCrBm(_, _, kb), cm); mma.accumulate_ = UMMA::ScaleOut::One; }
                cutlass::arch::umma_arrive(&sm.mbar[g & 1]);
            };

            // stage g0 & 1 last fed UMMA(g0 - 2): the previous unit's last iteration did not wait for it
            if (g0 >= 2) cute::wait_barrier(sm.mbar[g0 & 1], ((g0 - 2) >> 1) & 1);
            APre pre, pre2;                                // superblocks s + 1, s + 2 in flight
            load_a(0, pre);
            if (nsb > 1) load_a(1, pre2);
            produce_b(0);
            for (int s = 0; s < nsb; ++s) {
                const int g = g0 + s;
                produce_a(s, pre);                         // stage g & 1: UMMA(g - 2) waited one iteration ago
                pre = pre2;
                if (s + 2 < nsb) load_a(s + 2, pre2);
                asm volatile("bar.sync 1, %0;" ::"n"(KQT_PROD) : "memory");
                if (warp == 0) {
                    if (g >= 2) cute::wait_barrier(sm.tfree[g & 1], ((g - 2) >> 1) & 1);
                    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
                    if (!(KQT_DBG & 16)) issue(g);
                    else if (lane == 0) cute::arrive_barrier(sm.mbar[g & 1]);
                }
                if (s + 1 < nsb) {                         // stage (g + 1) & 1 last fed UMMA(g - 1); UMMA(g) runs meanwhile
                    if (g >= 1) cute::wait_barrier(sm.mbar[(g + 1) & 1], ((g - 1) >> 1) & 1);
                    produce_b(s + 1);
                }
            }
        } else {
            // ---------------- drain
            float acc[NT];
            #pragma unroll
            for (int n = 0; n < NT; ++n) acc[n] = 0.f;
            auto load_dd = [&](int s) {                // Q2: d | dmin (f16 x 2); Q3: d (f16)
                if (QT == 2) return __ldg(reinterpret_cast<const uint32_t*>(W + 1280 * cnt + ((size_t)et * nsb + s) * 64 + eg * 4));
                return (uint32_t)__ldg(reinterpret_cast<const unsigned short*>(W + 1728 * cnt + ((size_t)et * nsb + s) * 32 + eg * 2));
            };
            uint32_t dd = load_dd(0);
            for (int s = 0; s < nsb; ++s) {
                const int g = g0 + s, b = g & 1;
                const uint32_t dn = s + 1 < nsb ? load_dd(s + 1) : 0u;
                cute::wait_barrier(sm.mbar[b], (g >> 1) & 1);
                asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
                const float d = f16_lo(dd), dmin = f16_lo(dd >> 16);
                const uint32_t ta = tmem0 + ((uint32_t)((warp & 3) * 32) << 16) + b * 2 * NT;
                const float* xsum = sm.xsum[g % 4];
                #pragma unroll
                for (int c0 = 0; c0 < NT; c0 += 16) {
                    if (c0 >= ncol) break;
                    uint32_t r[16], q[16];
                    asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
                                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]),
                                   "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15])
                                 : "r"(ta + c0));
                    asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
                                 : "=r"(q[0]), "=r"(q[1]), "=r"(q[2]), "=r"(q[3]), "=r"(q[4]), "=r"(q[5]), "=r"(q[6]), "=r"(q[7]),
                                   "=r"(q[8]), "=r"(q[9]), "=r"(q[10]), "=r"(q[11]), "=r"(q[12]), "=r"(q[13]), "=r"(q[14]), "=r"(q[15])
                                 : "r"(ta + NT + c0));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    if (KQT_DBG & 8) { for (int j = 0; j < 16; ++j) acc[c0 + j] += __uint_as_float(r[j]); continue; }
                    #pragma unroll
                    for (int j = 0; j < 16; ++j) {
                        const int n = c0 + j;
                        if (QT == 2) {
                            acc[n] = fmaf(d, __uint_as_float(r[j]) - 128.f * xsum[n], acc[n]);
                            acc[n] = fmaf(-dmin, __uint_as_float(q[j]), acc[n]);
                        } else {
                            acc[n] = fmaf(d, fmaf(-4.f, __uint_as_float(q[j]), __uint_as_float(r[j]) - 1280.f * xsum[n]), acc[n]);
                        }
                    }
                }
                asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
                cute::arrive_barrier(sm.tfree[b]);
                dd = dn;
            }
            #pragma unroll
            for (int n = 0; n < NT; ++n)
                if (n < U.cnt) out[(size_t)(U.start + n) * ldo + row0 + er] = bf16r(acc[n]);
        }
        g0 += nsb;
    }
    __syncthreads();
    if (warp == 0) { talloc.release_allocation_lock(); talloc.free(tmem0, TCOLS); }
}

} // namespace v41g
