// nvfp4_quant.cu — activation NVFP4 quantizer (E2M1 codes + E4M3 block-16 scales, swizzled).
#include "nvfp4_quant.h"
#include <cuda_fp8.h>

__device__ __forceinline__ uint8_t quant_fp4_code(float x) {
    float a = fabsf(x); uint8_t idx;
    // nearest E2M1 magnitude {0,.5,1,1.5,2,3,4,6}; midpoints {.25,.75,1.25,1.75,2.5,3.5,5}
    if (a < 0.25f) idx = 0; else if (a < 0.75f) idx = 1; else if (a < 1.25f) idx = 2;
    else if (a < 1.75f) idx = 3; else if (a < 2.5f) idx = 4; else if (a < 3.5f) idx = 5;
    else if (a < 5.0f) idx = 6; else idx = 7;
    return idx | ((x < 0.0f) ? 0x8 : 0x0);
}

// one thread per (row m, block b) of 16 elements
__global__ void quant_kernel(const float* __restrict__ X, uint8_t* __restrict__ packed,
                             uint8_t* __restrict__ scales_swz, int M, int K, float gscale,
                             int KB, int tile_cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * KB;
    if (idx >= total) return;
    int m = idx / KB, b = idx % KB;
    const float* xb = X + (size_t)m * K + b * 16;

    float amax = 0.0f;
    #pragma unroll
    for (int i = 0; i < 16; ++i) amax = fmaxf(amax, fabsf(xb[i]));

    // stored e4m3 block scale = round_e4m3(gscale * amax/6); local real scale = e4m3_val/gscale
    float e4m3_real = gscale * (amax * (1.0f / 6.0f));
    __nv_fp8_e4m3 e4 = __nv_fp8_e4m3(e4m3_real);
    float e4v = float(e4);
    float local = (e4v > 0.0f) ? (e4v / gscale) : 1.0f;
    float inv_local = 1.0f / local;

    // write swizzled scale byte
    size_t tile = (size_t)((m / 128) * tile_cols + (b / 4));
    int within = (m % 32) * 16 + ((m % 128) / 32) * 4 + (b % 4);
    scales_swz[tile * 512 + within] = e4.__x;

    // quantize 16 codes, pack 2/byte (even idx low nibble)
    uint8_t* po = packed + (size_t)m * (K / 2) + b * 8;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        uint8_t lo = quant_fp4_code(xb[2 * i]     * inv_local);
        uint8_t hi = quant_fp4_code(xb[2 * i + 1] * inv_local);
        po[i] = lo | (hi << 4);
    }
}

void nvfp4_quantize_activations(const float* dX, uint8_t* dPacked, uint8_t* dScalesSwz,
                                int M, int K, float gscale, cudaStream_t stream) {
    int KB = K / 16;
    int tile_cols = ((KB + 3) / 4);
    int total = M * KB;
    int threads = 256, blocks = (total + threads - 1) / threads;
    quant_kernel<<<blocks, threads, 0, stream>>>(dX, dPacked, dScalesSwz, M, K, gscale, KB, tile_cols);
}
