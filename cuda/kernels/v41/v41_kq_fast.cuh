// v41_kq_fast.cuh — routed-expert GEMV for ggml Q2_K / Q3_K superblocks, half2 dot products.
//
// WHY. The generic engine (v41_gemv.cuh) decodes every K-quant weight to fp32 with separate
// roundings and then FMAs it against fp32 x: ~8-10 instructions per weight, compute-bound at
// 27-37% of streaming (s08_gemv_bench.log). At 2.6-3.4 bpw the GEMV must consume ~600-700
// Gweights/s to stay bandwidth-bound; this path costs ~2-2.5 instructions per weight.
//
// HOW. Expert inputs are always e4m3 codes x an fp32 scale per 32 (act_quant_fp8). An e4m3 value is
// exact in fp16, a K-quant code (Q2_K 0..3, Q3_K 0..7 before its -4 offset) is exact in fp16, and
// their product (<= 7 x 448) is exact in fp16. So per 16-weight sub-block
//     sum_i w_i x_i = xs * ( (d*sc) * sum_i q_i x8_i  -  (dmin*m) * sum_i x8_i )      Q2_K
//     sum_i w_i x_i = xs * ( (d*sc) * (sum_i c_i x8_i  -  4 * sum_i x8_i) )            Q3_K, c = q + 4
// where x8 is the e4m3 value without its scale. The integer-code dot runs as HFMA2 (codes become
// half2 by OR-ing them into the mantissa of 1024.0 and subtracting 1024); each half lane sums 8
// exact products, so the only rounding beyond the reference's is that 8-term fp16 sum (<= 25088 in
// magnitude, no overflow). sum_i x8_i is computed once per staged row in fp32. NOT bit-exact with
// the numpy decode (DecKQ); gated on tolerance by tests/gate_kq_fast.cu, the same contract the
// 0731's half2 MoE dequant (Opt #9) shipped under. M=1 and M=4 units run the same per-row
// arithmetic, so speculative verify stays identical to AR decode.
//
// LAYOUT. Same grid and rows as k_experts (64 rows per CTA, 8 per warp, lanes over 32-element
// chunks), so units, outputs and bf16 rounding are unchanged. Loop order is chunk-outer: a lane
// holds its chunk of every staged x row in registers and reuses it across its warp's 8 rows.
// x is staged as fp16 in "pair order": for elements 4g..4g+3, half2 (x0, x2) then (x1, x3), which is
// the order a (v >> s) & 0x00030003 extraction yields codes in.
#pragma once
#include "v41_gemv.cuh"

namespace v41g {

// one staged x row: K halves (pair order; chunk c's four 16-byte granules XOR-swizzled by c&3),
// K/16 fp32 sub-block sums of the unscaled e4m3 values, K/32 fp32 scales
__host__ __device__ constexpr size_t kq_stage_bytes(int K) { return (size_t)K * 2 + (size_t)K / 16 * 4 + (size_t)K / 32 * 4; }

__device__ __forceinline__ void kq_stage_row(uint8_t* dst, const XSrc& X, int row, int K) {
    const int nch = K / 32;
    uint4* xh = reinterpret_cast<uint4*>(dst);
    float* s16 = reinterpret_cast<float*>(dst + (size_t)K * 2);
    float* sc = s16 + K / 16;
    for (int g = threadIdx.x; g < nch * 2; g += blockDim.x) {     // one thread per 16-element sub-block
        const int c = g >> 1, h = g & 1;
        const uint4 u = *reinterpret_cast<const uint4*>(X.xq + (size_t)row * X.ldx + c * 32 + h * 16);
        const uint32_t w[4] = {u.x, u.y, u.z, u.w};
        float s = 0.f;
        #pragma unroll
        for (int k = 0; k < 2; ++k) {                              // granule i = 2h + k: elements 8i .. 8i+7
            uint32_t o[4];
            #pragma unroll
            for (int gg = 0; gg < 2; ++gg) {                       // group of 4 elements x0..x3
                const uint32_t v = w[2 * k + gg];
                const __half2 a = __half2(__nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(v & 0xFFFF), __NV_E4M3));
                const __half2 b = __half2(__nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(v >> 16), __NV_E4M3));
                const __half2 p02 = __lows2half2(a, b), p13 = __highs2half2(a, b);
                o[2 * gg] = *reinterpret_cast<const uint32_t*>(&p02);
                o[2 * gg + 1] = *reinterpret_cast<const uint32_t*>(&p13);
                const float2 fa = __half22float2(a), fb = __half22float2(b);
                s += (fa.x + fa.y) + (fb.x + fb.y);
            }
            const int i = 2 * h + k;
            xh[c * 4 + (i ^ (c & 3))] = make_uint4(o[0], o[1], o[2], o[3]);
        }
        s16[2 * c + h] = s;
        if (h == 0) sc[c] = X.xs[(size_t)row * (X.ldx / 32) + c];
    }
}

// 8 aligned 32-bit words starting at a 2-aligned byte address: 9 aligned loads + funnel shifts,
// branch-free (Q3_K superblocks are 110 B, so lanes of one warp see both alignments)
__device__ __forceinline__ void ld8w(const uint8_t* p, uint32_t (&w)[8]) {
    const uintptr_t a = reinterpret_cast<uintptr_t>(p);
    const uint32_t* q = reinterpret_cast<const uint32_t*>(a & ~(uintptr_t)3);
    const int sh = (int)(a & 3) * 8;
    uint32_t r[9];
    #pragma unroll
    for (int i = 0; i < 9; ++i) r[i] = q[i];
    #pragma unroll
    for (int i = 0; i < 8; ++i) w[i] = __funnelshift_r(r[i], r[i + 1], sh);
}

__device__ __forceinline__ __half2 code_h2(uint32_t v) {          // two codes at bits 0 and 16 -> half2
    const uint32_t m = v | 0x64006400u;
    return __hsub2(*reinterpret_cast<const __half2*>(&m), __half2(__half(1024.f), __half(1024.f)));
}
__device__ __forceinline__ float h2sum(__half2 a) { const float2 f = __half22float2(a); return f.x + f.y; }

// sum over one 16-element sub-block of code * x8, codes given as 4 words of 4 byte-lanes each
// (byte k of word i4 = element 4*i4 + k), x as 8 half2 in pair order
template <uint32_t MASK>
__device__ __forceinline__ float sub_dot(const uint32_t (&cw)[4], const uint32_t* xp) {
    __half2 acc = __float2half2_rn(0.f);
    #pragma unroll
    for (int i4 = 0; i4 < 4; ++i4) {
        const uint32_t v = cw[i4];
        acc = __hfma2(code_h2(v & MASK), *reinterpret_cast<const __half2*>(&xp[2 * i4]), acc);
        acc = __hfma2(code_h2((v >> 8) & MASK), *reinterpret_cast<const __half2*>(&xp[2 * i4 + 1]), acc);
    }
    return h2sum(acc);
}

__device__ __forceinline__ float f16_lo(uint32_t v) { return __half2float(__ushort_as_half((unsigned short)(v & 0xFFFF))); }

// expert pointer tables (device arrays indexed by expert); pass 0 = w1|w3 split at MOE_INTER, 1 = w2
struct KQTab { const uint8_t* const* w1; const uint8_t* const* w3; const uint8_t* const* w2; int pass; };

// grid (row tiles, max units); same contract as k_experts<KQDec<QT>, MT> in v41_moe.cu
// SOA: the matrix was reordered by kq_soa_repack (codes | scales | d planes, each 16-aligned), so a
// lane's 32 code bytes are two 16-byte loads; the values reaching each lane are the same.
// Blocks/SM at M=1 from gate_kq_fast (ncu: issue-bound, 0.66 eligible warps at 2): Q3_K gains 12% at 4
// despite ~140 B of spills; Q2_K loses at 3 or 4, so it stays at 2.
template <int QT, int MT, class UnitT, bool SOA = false>
__global__ void __launch_bounds__(256, MT == 1 ? (QT == 3 ? 4 : 2) : 1) k_kq(float* __restrict__ out, int ldo, XSrc X, bool x_by_pair,
                                            const UnitT* units, const int* units_n, const int* pair_tok,
                                            KQTab tab, int N, int K) {
    static_assert(QT == 2 || QT == 3, "Q2_K / Q3_K");
    constexpr int BB = QT == 2 ? 84 : 110;
    const int u = blockIdx.y;
    if (u >= *units_n) return;
    const UnitT U = units[u];
    extern __shared__ __align__(16) uint8_t kq_smem[];
    const size_t SB = kq_stage_bytes(K);
    for (int t = 0; t < MT; ++t) {
        const int p = U.start + min(t, U.cnt - 1);
        kq_stage_row(kq_smem + t * SB, X, x_by_pair ? p : pair_tok[p], K);
    }
    __syncthreads();
    const uint8_t *W0, *W1; int split;
    if (tab.pass == 0) { W0 = tab.w1[U.e]; W1 = tab.w3[U.e]; split = v41::MOE_INTER; }
    else { W0 = W1 = tab.w2[U.e]; split = 1 << 30; }
    const int nch = K / 32, nsb = K / 256;
    const int mrows = tab.pass == 0 ? split : N;          // rows of the one matrix a row index points into
    const int row0 = blockIdx.x * ROWS_PER_CTA;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    float acc[ROWS_PER_WARP][MT];
    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r)
        #pragma unroll
        for (int t = 0; t < MT; ++t) acc[r][t] = 0.f;
    for (int cb = 0; cb < nch; cb += 32) {
        const int c = cb + lane, cc = min(c, nch - 1);
        const bool live = c < nch;
        uint32_t xr[MT][16]; float xs16[MT][2], xsc[MT];
        #pragma unroll
        for (int t = 0; t < MT; ++t) {
            const uint8_t* st = kq_smem + t * SB;
            const uint4* xh = reinterpret_cast<const uint4*>(st);
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const uint4 g = xh[cc * 4 + (i ^ (cc & 3))];
                xr[t][4 * i] = g.x; xr[t][4 * i + 1] = g.y; xr[t][4 * i + 2] = g.z; xr[t][4 * i + 3] = g.w;
            }
            const float* s16 = reinterpret_cast<const float*>(st + (size_t)K * 2);
            xs16[t][0] = s16[2 * cc]; xs16[t][1] = s16[2 * cc + 1];
            xsc[t] = s16[K / 16 + cc];
        }
        const int j = cc & 7, sh = 2 * (j & 3);
        #pragma unroll
        for (int r = 0; r < ROWS_PER_WARP; ++r) {
            int row = min(row0 + warp * ROWS_PER_WARP + r, N - 1);
            const uint8_t* W = W0;
            if (row >= split) { W = W1; row -= split; }
            const size_t bi = (size_t)row * nsb + (cc >> 3), cnt = (size_t)mrows * nsb;
            const uint8_t* b = W + bi * BB;
            float a[2], m[2];
            uint32_t cw[8];
            if (QT == 2) {
                uint32_t q[8];
                if (SOA) {
                    const uint4* q4 = reinterpret_cast<const uint4*>(W + bi * 64 + (j >> 2) * 32);
                    const uint4 u0 = q4[0], u1 = q4[1];
                    q[0] = u0.x; q[1] = u0.y; q[2] = u0.z; q[3] = u0.w; q[4] = u1.x; q[5] = u1.y; q[6] = u1.z; q[7] = u1.w;
                } else {
                    const uint32_t* qa = reinterpret_cast<const uint32_t*>(b + 16 + (j >> 2) * 32);
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) q[i] = qa[i];
                }
                #pragma unroll
                for (int i = 0; i < 8; ++i) cw[i] = (q[i] >> sh) & 0x03030303u;
                const uint32_t dd = SOA ? *reinterpret_cast<const uint32_t*>(W + 80 * cnt + bi * 4)
                                        : *reinterpret_cast<const uint32_t*>(b + 80);
                const float d = f16_lo(dd), dm = f16_lo(dd >> 16);
                const uint32_t s2 = SOA ? *reinterpret_cast<const uint16_t*>(W + 64 * cnt + bi * 16 + 2 * j)
                                        : *reinterpret_cast<const uint16_t*>(b + 2 * j);
                #pragma unroll
                for (int hh = 0; hh < 2; ++hh) {
                    const uint32_t s = (s2 >> (8 * hh)) & 0xFF;
                    a[hh] = d * (float)(s & 0xF); m[hh] = dm * (float)(s >> 4);
                }
            } else {
                uint32_t qs[8], hm[8];
                if (SOA) {
                    const uint4* h4 = reinterpret_cast<const uint4*>(W + bi * 32);
                    const uint4* q4 = reinterpret_cast<const uint4*>(W + 32 * cnt + bi * 64 + (j >> 2) * 32);
                    const uint4 h0 = h4[0], h1 = h4[1], u0 = q4[0], u1 = q4[1];
                    hm[0] = h0.x; hm[1] = h0.y; hm[2] = h0.z; hm[3] = h0.w; hm[4] = h1.x; hm[5] = h1.y; hm[6] = h1.z; hm[7] = h1.w;
                    qs[0] = u0.x; qs[1] = u0.y; qs[2] = u0.z; qs[3] = u0.w; qs[4] = u1.x; qs[5] = u1.y; qs[6] = u1.z; qs[7] = u1.w;
                } else {
                    ld8w(b + 32 + (j >> 2) * 32, qs);
                    ld8w(b, hm);
                }
                #pragma unroll
                for (int i = 0; i < 8; ++i) cw[i] = ((qs[i] >> sh) & 0x03030303u) | (((hm[i] >> j) & 0x01010101u) << 2);
                const float d = __half2float(__ushort_as_half(SOA ? *reinterpret_cast<const unsigned short*>(W + 108 * cnt + bi * 2)
                                                                  : *reinterpret_cast<const unsigned short*>(b + 108)));
                const uint8_t* s12 = SOA ? W + 96 * cnt + bi * 12 : b + 96;
                #pragma unroll
                for (int hh = 0; hh < 2; ++hh) {
                    const int k = 2 * j + hh;
                    const int lo = k < 8 ? (s12[k] & 0xF) : (s12[k - 8] >> 4);
                    const int hi = (s12[8 + (k & 3)] >> (2 * (k >> 2))) & 3;
                    a[hh] = d * (float)((lo | (hi << 4)) - 32); m[hh] = 0.f;
                }
            }
            const uint32_t c0[4] = {cw[0], cw[1], cw[2], cw[3]}, c1[4] = {cw[4], cw[5], cw[6], cw[7]};
            constexpr uint32_t MASK = QT == 2 ? 0x00030003u : 0x00070007u;
            #pragma unroll
            for (int t = 0; t < MT; ++t) {
                const float d0 = sub_dot<MASK>(c0, &xr[t][0]), d1 = sub_dot<MASK>(c1, &xr[t][8]);
                float v;
                if (QT == 2) v = (a[0] * d0 - m[0] * xs16[t][0]) + (a[1] * d1 - m[1] * xs16[t][1]);
                else v = a[0] * (d0 - 4.f * xs16[t][0]) + a[1] * (d1 - 4.f * xs16[t][1]);
                if (live) acc[r][t] = fmaf(v, xsc[t], acc[r][t]);
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

// Rejected variants (gate_kq_fast, 2026-10-29, jetson_clocks on, Q2_K/Q3_K gate|up and down):
//  * per-warp coalesced row staging in shared memory: bit-identical, no faster (88-111 GB/s); ncu shows
//    the shared-memory pipe at 83% -- it moves the bottleneck from L1 to MIO.
//  * int16 fixed-point x + dp2a: exact integer dot, but 81-91 GB/s (same shared-memory limit).
//  * more rows per CTA (2x, 4x): slower; staging x is not the cost.
// What mattered: this kernel at 100 registers ran at 33% occupancy with nothing saturated (latency
// bound); capping it to 2 CTAs/SM (64 regs, no spills) took it from ~110 to 122-150 GB/s.

} // namespace v41g
