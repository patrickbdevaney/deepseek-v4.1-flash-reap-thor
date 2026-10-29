// v41_nv_mma.cuh — NVFP4 dense GEMV/GEMM on fp8 tensor cores (mma.m16n8k16 e4m3), every M.
//
// WHY. k_dense<DecNVFP4> streams at 200-290 GB/s for M = 1, but for M > 1 it decodes each weight to fp32
// and runs M FMAs per weight: a 3-row speculative verify costs 1.2-1.85x the M = 1 time on the same
// bytes (spec2c profile, 2026-10-29), and prefill is compute-bound.
//
// HOW. fp8_linear's input is act_quant_fp8: e4m3 codes x an fp32 scale per 32, so B is the staged x
// bytes (k_kqm's staging). An e2m1 nibble s|e1 e0 m re-encoded as the e4m3 byte s<<7 | (e1 e0 m)<<2 is
// EXACTLY the e2m1 value x 2^-6, subnormals included (biases 1 and 7 differ by 6, and the mantissa
// bit lands on the e4m3 mantissa's top bit). One mma covers 16 rows x one 16-weight scale block x up
// to 8 tokens with exact products and an fp32 sum; per block
//     out[r][n] += sc[r][blk] * (D[r][n] * xs[n][chunk])        then  out *= 64 * g
// NOT bit-identical to k_dense (different, tighter rounding). The per-element arithmetic does not
// depend on M, and the K split is chosen from the shape, so decode == verify == prefill bit for bit.
//
// LAYOUT (nv_mma_repack, in place: codes stay in the codes buffer, scales in the scales buffer). Rows
// in tiles of 16; per (tile, 256-wide superblock) idx:
//   codes  2048 B : lane l = 4g + t holds 16 words; word sb at byte (sb/4)*512 + 16 l + 4 (sb%4), so each
//                   16-byte load instruction reads 512 contiguous bytes across the warp (a 64-byte lane
//                   stride half-filled every 32-byte sector: ~118 GB/s from DRAM); byte b of word sb: low
//                   nibble = code of row g, high nibble = code of row g + 8, weight 16 sb + 4t + b
//   scales  256 B : 16 rows x 16 e4m3 block scales (row-major, block order)
#pragma once
#include "v41_kq_mma.cuh"

namespace v41g {

constexpr int NVM_ROWS_PER_CTA = 128;

// x bytes of [k0, k0 + nk) in B-fragment order (word (s*4 + t)*16 + sb, s relative to k0), then nk/32 scales
__host__ __device__ constexpr size_t nvm_stage_bytes(int nk) { return (size_t)nk + 16 + (size_t)nk / 32 * 4; }
__device__ __forceinline__ void nvm_stage_row(uint8_t* dst, const XSrc& X, int row, int col0, int nk) {
    uint32_t* xb = reinterpret_cast<uint32_t*>(dst);
    float* xs = reinterpret_cast<float*>(dst + (size_t)nk + 16);
    const uint32_t* src = reinterpret_cast<const uint32_t*>(X.xq + (size_t)row * X.ldx + col0);
    for (int w = threadIdx.x; w < nk / 4; w += blockDim.x) {
        const int k0 = 4 * w, s = k0 >> 8, r = k0 & 255, sb = r >> 4, t = (r >> 2) & 3;
        xb[(s * 4 + t) * 16 + sb] = src[w];
    }
    for (int c = threadIdx.x; c < nk / 32; c += blockDim.x) xs[c] = X.xs[(size_t)row * (X.ldx / 32) + col0 / 32 + c];
}

// e2m1 nibbles at bits 0-3 / 4-7 of each byte -> e4m3 bytes (value x 2^-6)
__device__ __forceinline__ uint32_t e2m1_lo_e4m3(uint32_t w) { return ((w << 2) & 0x1C1C1C1Cu) | ((w << 4) & 0x80808080u); }
__device__ __forceinline__ uint32_t e2m1_hi_e4m3(uint32_t w) { return ((w >> 2) & 0x1C1C1C1Cu) | (w & 0x80808080u); }

// grid (ceil(N/128), ceil(M/MT), ks). Group gi = row / Ng reads x columns [gi*K, (gi+1)*K). ks > 1 writes
// part[z][m][n] (already x 64 g); with kcnt the last CTA of a (row tile, m tile) to arrive sums them in z
// order (k_ksum's order, so the sums are unchanged), else k_ksum does.
template <int MT>
__global__ void __launch_bounds__(256) k_nvm(float* __restrict__ y, XSrc X, const uint8_t* __restrict__ W,
                                             const uint8_t* __restrict__ S, float g, int M, int N, int K, int Ng,
                                             int ks, int round, float* __restrict__ part, int* __restrict__ kcnt) {
    const int nsb = K / 256, z = blockIdx.z;
    const int s0 = (int)((long)nsb * z / ks), s1 = (int)((long)nsb * (z + 1) / ks), nk = (s1 - s0) * 256;
    const int m0 = blockIdx.y * MT, mrows = min(MT, M - m0);
    const int row0 = blockIdx.x * NVM_ROWS_PER_CTA, gi = row0 / Ng;
    extern __shared__ __align__(16) uint8_t nvm_smem[];
    const size_t SB = (nvm_stage_bytes(nk) + 15) & ~(size_t)15;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, gg = lane >> 2, t = lane & 3;
    const int tile = blockIdx.x * 8 + warp;
    const bool live = tile * 16 < N;
    // next superblock's codes and scale words are loaded before this one is computed, and the first
    // superblock's before x is staged (weights do not depend on x; a CTA's life is a few superblocks)
    auto load = [&](int s, uint32_t (&cw)[16], uint32_t (&sw)[2]) {
        const size_t idx = (size_t)tile * nsb + s;
        #pragma unroll
        for (int i = 0; i < 4; ++i) { const uint4 v = *reinterpret_cast<const uint4*>(W + idx * 2048 + i * 512 + lane * 16); cw[4 * i] = v.x; cw[4 * i + 1] = v.y; cw[4 * i + 2] = v.z; cw[4 * i + 3] = v.w; }
        #pragma unroll
        for (int h = 0; h < 2; ++h) sw[h] = *reinterpret_cast<const uint32_t*>(S + idx * 256 + (gg + 8 * h) * 16 + 4 * t);
    };
    uint32_t cw[16], sw[2];
    if (live) load(s0, cw, sw);
    for (int n = 0; n < MT; ++n) nvm_stage_row(nvm_smem + n * SB, X, m0 + min(n, mrows - 1), gi * K + s0 * 256, nk);
    __syncthreads();
    if (live) {
    const uint8_t* xg = nvm_smem + min(gg, MT - 1) * SB;
    const float* xsn[2];
    #pragma unroll
    for (int e = 0; e < 2; ++e) xsn[e] = reinterpret_cast<const float*>(nvm_smem + min(2 * t + e, MT - 1) * SB + nk + 16);
    constexpr int NE = MT == 1 ? 1 : 2;
    float acc[2][NE];
    #pragma unroll
    for (int h = 0; h < 2; ++h)
        #pragma unroll
        for (int e = 0; e < NE; ++e) acc[h][e] = 0.f;
    const int src_q = lane & ~3;
    for (int s = s0; s < s1; ++s) {
        uint32_t ncw[16], nsw[2];
        if (s + 1 < s1) load(s + 1, ncw, nsw);
        // this lane's block scales: blocks 4t + i of rows gg, gg + 8
        float scf[2][4];
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            const __half2 lo = __half2(__nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(sw[h] & 0xFFFF), __NV_E4M3));
            const __half2 hi = __half2(__nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(sw[h] >> 16), __NV_E4M3));
            const float2 a = __half22float2(lo), b = __half22float2(hi);
            scf[h][0] = a.x; scf[h][1] = a.y; scf[h][2] = b.x; scf[h][3] = b.y;
        }
        uint32_t xb[16];
        {
            const uint4* x4 = reinterpret_cast<const uint4*>(xg + (size_t)((s - s0) * 4 + t) * 64);
            #pragma unroll
            for (int i = 0; i < 4; ++i) { const uint4 v = x4[i]; xb[4 * i] = v.x; xb[4 * i + 1] = v.y; xb[4 * i + 2] = v.z; xb[4 * i + 3] = v.w; }
        }
        float xs[2][8];
        #pragma unroll
        for (int e = 0; e < NE; ++e) {
            const float4* p = reinterpret_cast<const float4*>(xsn[e] + (s - s0) * 8);
            const float4 a = p[0], b = p[1];
            xs[e][0] = a.x; xs[e][1] = a.y; xs[e][2] = a.z; xs[e][3] = a.w; xs[e][4] = b.x; xs[e][5] = b.y; xs[e][6] = b.z; xs[e][7] = b.w;
        }
        #pragma unroll
        for (int sb = 0; sb < 16; ++sb) {
            float D[4];
            mma_e4m3(D, e2m1_lo_e4m3(cw[sb]), e2m1_hi_e4m3(cw[sb]), xb[sb]);
            const float c0 = __shfl_sync(0xffffffffu, scf[0][sb & 3], src_q | (sb >> 2));
            const float c1 = __shfl_sync(0xffffffffu, scf[1][sb & 3], src_q | (sb >> 2));
            #pragma unroll
            for (int e = 0; e < NE; ++e) {
                acc[0][e] = fmaf(c0, D[e] * xs[e][sb >> 1], acc[0][e]);
                acc[1][e] = fmaf(c1, D[2 + e] * xs[e][sb >> 1], acc[1][e]);
            }
        }
        if (s + 1 < s1) {
            #pragma unroll
            for (int i = 0; i < 16; ++i) cw[i] = ncw[i];
            sw[0] = nsw[0]; sw[1] = nsw[1];
        }
    }
    const float og = 64.f * g;
    #pragma unroll
    for (int h = 0; h < 2; ++h)
        #pragma unroll
        for (int e = 0; e < NE; ++e) {
            const int n = 2 * t + e, row = tile * 16 + gg + 8 * h;
            if (n >= mrows || row >= N) continue;
            const float v = acc[h][e] * og;
            if (ks > 1) part[((size_t)z * M + m0 + n) * N + row] = v;
            else y[(size_t)(m0 + n) * N + row] = round ? bf16r(v) : v;
        }
    }
    if (ks == 1 || !kcnt) return;
    __shared__ int last;
    __threadfence();
    __syncthreads();
    int* c = kcnt + blockIdx.y * gridDim.x + blockIdx.x;
    if (threadIdx.x == 0) last = atomicAdd(c, 1) == ks - 1;
    __syncthreads();
    if (!last) return;
    __threadfence();
    for (int i = threadIdx.x; i < NVM_ROWS_PER_CTA * MT; i += blockDim.x) {
        const int row = row0 + (i % NVM_ROWS_PER_CTA), m = m0 + i / NVM_ROWS_PER_CTA;
        if (row >= N || m >= M) continue;
        float v = 0.f;
        for (int zz = 0; zz < ks; ++zz) v += __ldcg(part + ((size_t)zz * M + m) * N + row);
        y[(size_t)m * N + row] = round ? bf16r(v) : v;
    }
    if (threadIdx.x == 0) *c = 0;
}

// row-major codes [N][K/2] (low nibble first) + scales [N][K/16] -> the layout above, via tmp
static __global__ void k_nv_mma_repack_codes(uint8_t* dst, const uint8_t* src, int N, int K) {
    const int nsb = K / 256;
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x, cnt = (size_t)(N / 16) * nsb;
    if (i >= cnt * 32) return;
    const int l = (int)(i & 31), gq = l >> 2, t = l & 3;
    const size_t idx = i >> 5;
    const int T = (int)(idx / nsb), s = (int)(idx % nsb);
    auto code = [&](int row, int k) { const uint8_t b = src[(size_t)row * (K / 2) + k / 2]; return (k & 1) ? (b >> 4) : (b & 15); };
    for (int sb = 0; sb < 16; ++sb) {
        uint32_t* o = reinterpret_cast<uint32_t*>(dst + idx * 2048 + (sb >> 2) * 512 + l * 16 + (sb & 3) * 4);
        uint32_t w = 0;
        for (int b = 0; b < 4; ++b) {
            const int k = s * 256 + sb * 16 + 4 * t + b;
            w |= (uint32_t)(code(T * 16 + gq, k) | (code(T * 16 + gq + 8, k) << 4)) << (8 * b);
        }
        *o = w;
    }
}
static __global__ void k_nv_mma_repack_scales(uint8_t* dst, const uint8_t* src, int N, int K) {
    const int nsb = K / 256;
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;   // one per (idx, row in tile)
    if (i >= (size_t)(N / 16) * nsb * 16) return;
    const int r = (int)(i & 15); const size_t idx = i >> 4;
    const int T = (int)(idx / nsb), s = (int)(idx % nsb);
    for (int j = 0; j < 16; ++j) dst[idx * 256 + r * 16 + j] = src[(size_t)(T * 16 + r) * (K / 16) + s * 16 + j];
}

} // namespace v41g
