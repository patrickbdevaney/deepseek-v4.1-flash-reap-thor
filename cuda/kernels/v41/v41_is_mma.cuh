// v41_is_mma.cuh — lightning-indexer scores on fp8 tensor cores (mma.m16n8k16 e4m3), P3.
//
// WHY. k_index_scores decodes every key tile once per query row and runs 32 heads x 128 scalar FMAs per
// key: 5.1 s of an 8k prefill (34 ms/launch), the keys re-read M times per chunk.
//
// HOW. Queries and keys are both fp4_sim output: e2m1 codes x an e8m0 scale per 32 (q as packed codes
// from fp4_sim, keys as the packed cache). An e2m1 nibble re-encoded as the e4m3 byte s<<7 | (e1 e0 m)<<2
// is exactly the value x 2^-6 (v41_nv_mma.cuh), so one 32-wide chunk's dot is two mma k16 steps with
// exact products and an exact fp32 sum (multiples of 2^-14 below 2^11), then x 2^12 sq sk. The chunks
// add in order, so d equals k_index_scores' sequential fmaf chain whenever its partial sums are exact
// (scale spreads under ~2^10); the epilogue is the reference's: bf16(relu(bf16(d)) * w) per head, summed
// sequentially in groups of 4 heads, the groups combined in k_index_scores' xor-shuffle order.
//
// TILING. CTA = IM_ROWS query rows x IM_KEYS keys; warp w = (row w / 2, heads 16 (w & 1) ..), all
// IM_KEYS / 8 key tiles; per-head values go through smem for the ordered head sum.
#pragma once
#include "v41_gemv.cuh"

namespace v41g {

constexpr int IM_ROWS = 4, IM_KEYS = 64;

// 4 packed e2m1 nibbles (low first) -> 4 e4m3 bytes (x 2^-6)
__device__ __forceinline__ uint32_t im_e2m1x4_to_e4m3(uint32_t v16) {
    uint32_t r = 0;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint32_t n = (v16 >> (4 * i)) & 0xF;
        r |= (((n & 8) << 4) | ((n & 7) << 2)) << (8 * i);
    }
    return r;
}
__device__ __forceinline__ void im_mma(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t b0) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3},{%4,%5},{%6},{%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a0), "r"(a1), "r"(b0));
}
__device__ __forceinline__ float im_e8m0(uint8_t b) { return b == 0 ? 0x1p-127f : __uint_as_float((uint32_t)b << 23); }

// grid (ceil(M / IM_ROWS), ceil(ld / IM_KEYS)); qc [M, 32, 64] packed, qs [M, 32, 4] e8m0; kc [T, 64], ks [T, 4]
template <int NH, int HD>
__global__ void __launch_bounds__(256) k_index_mma(float* score, const uint8_t* qc, const uint8_t* qsc, const float* w,
                                                  const uint8_t* kc, const uint8_t* ks, const uint8_t* cand,
                                                  const int* d_pos, int ratio, int M, int ld, int ldc, int cand_block) {
    static_assert(NH == 32 && HD == 128, "2 head tiles x 4 chunks");
    constexpr int NT = IM_KEYS / 8, NC = HD / 32;
    __shared__ uint8_t ksm[IM_KEYS][HD / 2 + 4];           // packed codes (+4: bank skew)
    __shared__ float kss[IM_KEYS][NC];
    __shared__ float ev[IM_ROWS][IM_KEYS][NH + 1];         // bf16(relu(bf16 d) * w) per head
    __shared__ int nrow[IM_ROWS], live[IM_ROWS];
    const int m0 = blockIdx.x * IM_ROWS, t0 = blockIdx.y * IM_KEYS;
    if (threadIdx.x < IM_ROWS) {
        const int m = m0 + threadIdx.x;
        int n = 0, any = 0;
        if (m < M) {
            n = (d_pos[m] + 1) / ratio;
            if (t0 < n) {
                any = 1;
                if (cand) {
                    any = 0;
                    for (int b = t0 / cand_block; b <= (min(t0 + IM_KEYS, ld) - 1) / cand_block; ++b)
                        if (b < ldc) any |= cand[(size_t)m * ldc + b] != 0;
                }
            }
        }
        nrow[threadIdx.x] = n; live[threadIdx.x] = any;
    }
    __syncthreads();
    if (!(live[0] | live[1] | live[2] | live[3])) {
        for (int i = threadIdx.x; i < IM_ROWS * IM_KEYS; i += blockDim.x) {
            const int r = i / IM_KEYS, t = t0 + i % IM_KEYS;
            if (m0 + r < M && t < ld) score[(size_t)(m0 + r) * ld + t] = -INFINITY;
        }
        return;
    }
    int nmax = 0;
    #pragma unroll
    for (int r = 0; r < IM_ROWS; ++r) nmax = max(nmax, nrow[r]);
    for (int i = threadIdx.x; i < IM_KEYS * (HD / 8); i += blockDim.x) {      // 16 x 4-byte words per key
        const int r = i / (HD / 8), wd = i % (HD / 8), t = min(t0 + r, max(nmax, 1) - 1);
        *reinterpret_cast<uint32_t*>(&ksm[r][4 * wd]) = *reinterpret_cast<const uint32_t*>(kc + (size_t)t * (HD / 2) + 4 * wd);
        if (wd < NC) kss[r][wd] = im_e8m0(ks[(size_t)t * NC + wd]);
    }
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int rr = warp >> 1, ht = warp & 1, m = m0 + rr;
    if (m < M && live[rr]) {
        // A: heads 16 ht + g (+8), dims 16 kk + 4t .. +3 -> 2 bytes of packed nibbles each
        const uint8_t* qrow = qc + ((size_t)m * NH + 16 * ht) * (HD / 2);
        uint32_t a[8][2];
        #pragma unroll
        for (int kk = 0; kk < HD / 16; ++kk)
            #pragma unroll
            for (int h = 0; h < 2; ++h)
                a[kk][h] = im_e2m1x4_to_e4m3(*reinterpret_cast<const uint16_t*>(qrow + (size_t)(g + 8 * h) * (HD / 2) + 8 * kk + 2 * t));
        float sq[2][NC];
        #pragma unroll
        for (int h = 0; h < 2; ++h)
            #pragma unroll
            for (int c = 0; c < NC; ++c) sq[h][c] = im_e8m0(qsc[((size_t)m * NH + 16 * ht + g + 8 * h) * NC + c]) * 4096.f;
        const float wv[2] = {w[(size_t)m * NH + 16 * ht + g], w[(size_t)m * NH + 16 * ht + g + 8]};
        #pragma unroll 2
        for (int j = 0; j < NT; ++j) {
            const int key = 8 * j + g;                     // B column; C columns are 8 j + 2t (+1)
            float acc[4] = {0.f, 0.f, 0.f, 0.f};
            #pragma unroll
            for (int c = 0; c < NC; ++c) {
                float D[4] = {0.f, 0.f, 0.f, 0.f};
                #pragma unroll
                for (int k2 = 0; k2 < 2; ++k2) {
                    const int kk = 2 * c + k2;
                    const uint32_t b = im_e2m1x4_to_e4m3(*reinterpret_cast<const uint16_t*>(&ksm[key][8 * kk + 2 * t]));
                    im_mma(D, a[kk][0], a[kk][1], b);
                }
                #pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const float s = sq[e >> 1][c] * kss[8 * j + 2 * t + (e & 1)][c];
                    acc[e] += D[e] * s;
                }
            }
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                const float d = fmaxf(bf16r(acc[e]), 0.f);
                ev[rr][8 * j + 2 * t + (e & 1)][16 * ht + g + 8 * (e >> 1)] = bf16r(d * wv[e >> 1]);
            }
        }
    }
    __syncthreads();
    // ordered head sum: groups of 4 sequential, then ((g0+g1)+(g2+g3)) + ((g4+g5)+(g6+g7))
    for (int i = threadIdx.x; i < IM_ROWS * IM_KEYS; i += blockDim.x) {
        const int r = i / IM_KEYS, tk = i % IM_KEYS, tt = t0 + tk, mr = m0 + r;
        if (mr >= M || tt >= ld) continue;
        float v = -INFINITY;
        if (live[r] && tt < nrow[r] && !(cand && !cand[(size_t)mr * ldc + tt / cand_block])) {
            float p[8];
            #pragma unroll
            for (int q = 0; q < 8; ++q) {
                float s = 0.f;
                #pragma unroll
                for (int h = 4 * q; h < 4 * q + 4; ++h) s += ev[r][tk][h];
                p[q] = s;
            }
            v = bf16r(((p[0] + p[1]) + (p[2] + p[3])) + ((p[4] + p[5]) + (p[6] + p[7])));
        }
        score[(size_t)mr * ld + tt] = v;
    }
}

} // namespace v41g
