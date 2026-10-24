// v41_engram.cu — Engram lookup (row dequant) and the gated residual write (model.py Engram).
//
// Rows arrive in the host-registered arena filled by src/engram_store.cpp (io_uring O_DIRECT reads
// from the packed row file). Each arena slot is 1 KiB; the row sits at the byte offset the reader
// recorded in `slot_off` (the 512-B read granule rarely starts exactly at the row).
#include "v41_kernels.h"
#include "v41_config.h"
#include "v41_dev.cuh"

using namespace v41d;

namespace v41k {

// OCP MXFP6 E3M2 element -> float (bias 3, no inf/nan): subnormal e==0 -> m/4 * 2^-2
__device__ __forceinline__ float e3m2f(uint32_t c) {
    const uint32_t e = (c >> 2) & 7u, mm = c & 3u;
    const float mag = e == 0 ? (float)mm * 0.0625f : ldexpf(1.f + 0.25f * (float)mm, (int)e - 3);
    return (c & 32u) ? -mag : mag;
}

// fmt 0: FP8 E4M3 x256 + E8M0 x8 (264 B, lossless repack of the checkpoint)
// fmt 1: MXFP6 E3M2 x256 packed 4-per-3-bytes (192 B) + E8M0 x8 (200 B)
__global__ void k_engram_dequant(float* e, const uint8_t* arena, const uint16_t* slot_off, int fmt) {
    const int slot = blockIdx.x;                              // m * 24 + col
    const uint8_t* row = arena + (size_t)slot * 1024 + slot_off[slot];
    const int d = threadIdx.x;                                // 0..255
    float v;
    if (fmt == 0) {
        v = e4m3f(row[d]) * e8m0f(row[256 + (d >> 5)]);
    } else {
        const int g = d >> 2, k = d & 3;                      // 4 codes in 3 bytes, little-endian bits
        const uint32_t w = row[3 * g] | (row[3 * g + 1] << 8) | (row[3 * g + 2] << 16);
        v = e3m2f((w >> (6 * k)) & 63u) * e8m0f(row[192 + (d >> 5)]);
    }
    e[(size_t)slot * v41::ENGRAM_HEAD_DIM + d] = bf16r(v);
}
void engram_dequant(float* e, const uint8_t* rows, int cap, int fmt, int M, cudaStream_t st) {
    // the slot-offset table lives right after ALL cap*24 1 KiB slots (see EngramStore::arena layout); M < cap
    // (decode, a prompt's last chunk) must not move it
    const uint16_t* off = reinterpret_cast<const uint16_t*>(rows + (size_t)cap * v41::ENGRAM_COLS * 1024);
    k_engram_dequant<<<M * v41::ENGRAM_COLS, v41::ENGRAM_HEAD_DIM, 0, st>>>(e, rows, off, fmt);
}

// One CTA per (token, hc copy). kv[m] = [key copy 0..3 | value], each `dim` wide, bf16 values.
__global__ void k_engram_gate(float* h, const float* kv, const uint16_t* qw, const uint16_t* kw, const uint8_t* mask,
                              float eps) {
    const int m = blockIdx.x, j = blockIdx.y, D = v41::DIM;
    float* hr = h + (size_t)m * v41::HC_DIM + (size_t)j * D;
    const float* key = kv + (size_t)m * v41::ENGRAM_KV_OUT + (size_t)j * D;
    const float* val = kv + (size_t)m * v41::ENGRAM_KV_OUT + (size_t)v41::HC_MULT * D;
    float hh = 0.f, kk = 0.f, hk = 0.f;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        const float hv = hr[d], kv_ = key[d];
        const float w = bf16f(qw[j * D + d]) * bf16f(kw[j * D + d]);
        hh = fmaf(hv, hv, hh); kk = fmaf(kv_, kv_, kk); hk += (hv * w) * kv_;
    }
    __shared__ float red[3][32];
    hh = warp_sum(hh); kk = warp_sum(kk); hk = warp_sum(hk);
    const int w_ = threadIdx.x >> 5, nw = blockDim.x >> 5;
    if ((threadIdx.x & 31) == 0) { red[0][w_] = hh; red[1][w_] = kk; red[2][w_] = hk; }
    __syncthreads();
    if (threadIdx.x == 0) {
        float a = 0.f, b = 0.f, c = 0.f;
        for (int i = 0; i < nw; ++i) { a += red[0][i]; b += red[1][i]; c += red[2][i]; }
        const float rstd = rsqrtf(a / D + eps) * rsqrtf(b / D + eps);
        const float dot = c * rstd * rsqrtf((float)D);
        const float sq = sqrtf(fmaxf(fabsf(dot), 1e-6f));
        float g = 1.f / (1.f + expf(-copysignf(sq, dot)));
        if (mask && !mask[m]) g = 0.f;
        red[0][0] = g;
    }
    __syncthreads();
    const float g = red[0][0];
    for (int d = threadIdx.x; d < D; d += blockDim.x) hr[d] = bf16r(hr[d] + g * val[d]);
}
void engram_gate(float* h, const float* kv, const uint16_t* qw, const uint16_t* kw, const uint8_t* mask, int M,
                 float eps, cudaStream_t st) {
    k_engram_gate<<<dim3(M, v41::HC_MULT), 512, 0, st>>>(h, kv, qw, kw, mask, eps);
}

} // namespace v41k
