// nvfp4_quant.h — quantize fp32/bf16 activations to NVFP4 (E2M1 codes + E4M3 block-16 scales),
// writing scales directly into cuBLAS's swizzled tile layout. For the W4A4 decode path.
#pragma once
#include <cstdint>
#include <cuda_runtime.h>

// Quantize dX[M,K] (row-major fp32) -> dPacked[M,K/2] FP4 + dScalesSwz (swizzled E4M3, must be
// pre-allocated to nvfp4_scale_buffer_bytes(M,K) and is fully overwritten incl. padding).
// gscale = per-tensor global scale (compressed-tensors convention: 2688/amax). K must be mult of 16.
void nvfp4_quantize_activations(const float* dX, uint8_t* dPacked, uint8_t* dScalesSwz,
                                int M, int K, float gscale, cudaStream_t stream);
