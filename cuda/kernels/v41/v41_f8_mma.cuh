// v41_f8_mma.cuh — FP8 (e4m3, e8m0 per 32x32) dense on fp8 tensor cores (mma.m16n8k16 e4m3), every M. P6.
//
// WHY. k_dense<DecFP8> decodes each weight to fp32 and runs M FMAs per weight: the Engram KV projection
// (25600 x 6144, layers 1 and 14) costs 46 ms per 256-row prefill chunk, 2.95 s of an 8k prefill, and it
// is compute-bound (MT 4 -> 8 through the smem cap bought 0.6%).
//
// HOW. The weight bytes ARE e4m3 A fragments and act_quant_fp8's x bytes are e4m3 B fragments (k_nvm's
// staging), so one m16n8k32 mma covers 16 rows x one 32-wide scale chunk x 8 tokens with exact products and
// an fp32 sum; per chunk   out[r][n] += ws[r/32][k/32] * (D[r][n] * xs[n][k/32]).
// NOT bit-identical to k_dense (different rounding: T1). The per-element arithmetic does not depend on M
// (each mma column is independent, K is never split), so decode == verify == prefill bit for bit.
//
// LAYOUT (f8m_repack, in place). Rows in tiles of 16; per (tile, 256-wide superblock) 4096 B: lane
// l = 4g + t holds 32 words, word j = 2 k16 + h = row g + 8h, weights 256 s + 16 k16 + 4t .. +3, at byte
// (j/4)*512 + 16 l + 4 (j%4) (each 16-byte load instruction reads 512 contiguous bytes across the warp).
// Scales stay row-major [N/32, K/32].
#pragma once
#include "v41_nv_mma.cuh"

namespace v41g {

constexpr int F8M_ROWS_PER_CTA = 128, F8M_KC = 1024;

// one 32-wide chunk: a0/a1 = rows g / g+8 at k 4t.., a2/a3 the same at k 16+4t.., b0/b1 = k 4t.. / 16+4t..
__device__ __forceinline__ void mma_e4m3_k32(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                             uint32_t b0, uint32_t b1) {
    d[0] = d[1] = d[2] = d[3] = 0.f;
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__global__ void k_f8m_repack(uint8_t* __restrict__ dst, const uint8_t* __restrict__ src, int N, int K) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x, nsb = K / 256;
    if (i >= (size_t)(N / 16) * nsb * 1024) return;
    const int j = (int)(i & 31), l = (int)((i >> 5) & 31);
    const size_t ts = i >> 10;
    const int s = (int)(ts % nsb), tile = (int)(ts / nsb), g = l >> 2, t = l & 3, k16 = j >> 1, h = j & 1;
    const size_t row = (size_t)tile * 16 + g + 8 * h, col = (size_t)s * 256 + 16 * k16 + 4 * t;
    *reinterpret_cast<uint32_t*>(dst + ts * 4096 + (j / 4) * 512 + 16 * l + 4 * (j % 4)) =
        *reinterpret_cast<const uint32_t*>(src + row * K + col);
}

// grid (ceil(N / 128), ceil(M / (8 NT))); NT n-tiles of 8 tokens per warp share each A fragment
template <int NT>
__global__ void __launch_bounds__(256) k_f8m(float* __restrict__ y, XSrc X, const uint8_t* __restrict__ W,
                                             const uint8_t* __restrict__ Ws, int M, int N, int K, int round) {
    constexpr int MT = 8 * NT;
    const int nsb = K / 256, m0 = blockIdx.y * MT, mrows = min(MT, M - m0);
    extern __shared__ __align__(16) uint8_t f8m_smem[];
    constexpr size_t SB = (nvm_stage_bytes(F8M_KC) + 15) & ~(size_t)15;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, gg = lane >> 2, t = lane & 3;
    const int tile = blockIdx.x * 8 + warp;
    const bool live = tile * 16 < N;
    const uint8_t* wsr = Ws + (size_t)(tile >> 1) * (K / 32);
    float acc[NT][2][2];
    #pragma unroll
    for (int j = 0; j < NT; ++j) acc[j][0][0] = acc[j][0][1] = acc[j][1][0] = acc[j][1][1] = 0.f;
    for (int kc = 0; kc < K; kc += F8M_KC) {
        const int nk = min(F8M_KC, K - kc), sc0 = kc / 256;
        __syncthreads();                                  // the previous chunk's reads of the stage are done
        for (int n = 0; n < MT; ++n) nvm_stage_row(f8m_smem + n * SB, X, m0 + min(n, mrows - 1), kc, nk);
        __syncthreads();
        if (!live) continue;
        for (int s = sc0; s < sc0 + nk / 256; ++s) {
            uint32_t cw[32];
            const uint8_t* wp = W + ((size_t)tile * nsb + s) * 4096 + lane * 16;
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                const uint4 v = *reinterpret_cast<const uint4*>(wp + i * 512);
                cw[4 * i] = v.x; cw[4 * i + 1] = v.y; cw[4 * i + 2] = v.z; cw[4 * i + 3] = v.w;
            }
            const uint2 sw = *reinterpret_cast<const uint2*>(wsr + s * 8);   // 8 chunk scales (one 32-row group)
            const uint8_t sb8[8] = {(uint8_t)sw.x, (uint8_t)(sw.x >> 8), (uint8_t)(sw.x >> 16), (uint8_t)(sw.x >> 24),
                                    (uint8_t)sw.y, (uint8_t)(sw.y >> 8), (uint8_t)(sw.y >> 16), (uint8_t)(sw.y >> 24)};
            #pragma unroll
            for (int j = 0; j < NT; ++j) {
                const uint8_t* xg = f8m_smem + (8 * j + gg) * SB;
                uint32_t xb[16];
                const uint4* x4 = reinterpret_cast<const uint4*>(xg + (size_t)((s - sc0) * 4 + t) * 64);
                #pragma unroll
                for (int i = 0; i < 4; ++i) { const uint4 v = x4[i]; xb[4 * i] = v.x; xb[4 * i + 1] = v.y; xb[4 * i + 2] = v.z; xb[4 * i + 3] = v.w; }
                const float* xs0 = reinterpret_cast<const float*>(f8m_smem + (8 * j + 2 * t) * SB + nk + 16) + (s - sc0) * 8;
                const float* xs1 = reinterpret_cast<const float*>(f8m_smem + (8 * j + 2 * t + 1) * SB + nk + 16) + (s - sc0) * 8;
                #pragma unroll
                for (int c = 0; c < 8; ++c) {                 // 32-wide chunk c: one weight scale, one x scale
                    float D[4];
                    mma_e4m3_k32(D, cw[4 * c], cw[4 * c + 1], cw[4 * c + 2], cw[4 * c + 3], xb[2 * c], xb[2 * c + 1]);
                    const float w = e8m0f(sb8[c]);
                    const float x0 = xs0[c], x1 = xs1[c];
                    acc[j][0][0] = fmaf(w, D[0] * x0, acc[j][0][0]);
                    acc[j][0][1] = fmaf(w, D[1] * x1, acc[j][0][1]);
                    acc[j][1][0] = fmaf(w, D[2] * x0, acc[j][1][0]);
                    acc[j][1][1] = fmaf(w, D[3] * x1, acc[j][1][1]);
                }
            }
        }
    }
    if (!live) return;
    #pragma unroll
    for (int j = 0; j < NT; ++j)
        #pragma unroll
        for (int h = 0; h < 2; ++h)
            #pragma unroll
            for (int e = 0; e < 2; ++e) {
                const int n = 8 * j + 2 * t + e, row = tile * 16 + gg + 8 * h;
                if (n >= mrows || row >= N) continue;
                const float v = acc[j][h][e];
                y[(size_t)(m0 + n) * N + row] = round ? bf16r(v) : v;
            }
}

} // namespace v41g
