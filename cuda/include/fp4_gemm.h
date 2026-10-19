// fp4_gemm.h — NVFP4 (E2M1, block-16, E4M3 scales) GEMM via cublasLt, for Gemma-4 NVFP4 Linear layers.
// Computes a TN GEMM equivalent to a Linear:  D[M,N] = (A[M,K] @ B[N,K]^T) * a_gscale * b_gscale
//   A = activations [M,K] NVFP4,  B = weights [N,K] NVFP4,  D = fp32 output, COLUMN-MAJOR (ld=M).
// Per-block FP8-E4M3 scale factors must be pre-swizzled into cuBLAS's 128x4-tile layout
// (use nvfp4_swizzle_scales for the logical [outer, K/16] -> physical mapping).
#pragma once
#include <cstdint>
#include <cstddef>
#include <cublasLt.h>
#include <cuda_runtime.h>

// Size (bytes) of the swizzled scale buffer for an [outer, K] NVFP4 operand.
size_t nvfp4_scale_buffer_bytes(int outer, int K);

// Swizzle logical E4M3 scale bytes (host, shape [outer, K/16], row-major) into the padded
// cuBLAS tile layout (host out buffer of size nvfp4_scale_buffer_bytes). OOB padding zeroed.
void nvfp4_swizzle_scales(const uint8_t* logical, uint8_t* swizzled, int outer, int K);

// Batched W4A16 GEMM: out[M,N] row-major = dequant(W[N,K]) @ x[M,K]^T. FP4 weight x fp32 activation
// (no activation quant). Raw unswizzled weight+scales, any alignment. Weight reused across M rows.
void w4a16_gemm(float* out, const uint8_t* wp, const uint8_t* ws, float w_gscale, const void* x16,
                int M, int N, int K, cudaStream_t s);

// FP4-weight GEMV for decode (M=1, W4A16). y[N] = dequant(W[N,K]) @ x[K]. x16 = FP16 activation [K]
// (8 halves loaded as one uint4/iter). wp/ws = raw unswizzled packed weight + E4M3 scales.
void fp4_gemv(float* y, const uint8_t* wp, const uint8_t* ws, float w_gscale, const void* x16,
              int N, int K, cudaStream_t s);

// One NVFP4 GEMM. All device pointers. D is fp32 col-major [M,N] ld=M. beta=0.
// Returns cublasStatus_t (CUBLAS_STATUS_SUCCESS on success).
int nvfp4_gemm(float* dD,
               const uint8_t* dA_packed, const uint8_t* dA_scales_swz, float a_gscale,
               const uint8_t* dB_packed, const uint8_t* dB_scales_swz, float b_gscale,
               int M, int N, int K,
               cublasLtHandle_t lt, void* workspace, size_t ws_bytes, cudaStream_t stream);
