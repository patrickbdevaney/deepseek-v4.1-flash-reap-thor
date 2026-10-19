// fp8_block_gemm.h — FP8 e4m3 block-scaled GEMM (deepseek_v4 dense/attn linears).
// C[M,N] = A[M,K] @ B[N,K]^T, dequant per-block:
//   A_real[m,k] = e4m3(A_fp8[m,k]) * a_s[m, k/128]     (per-row, per-128-K activation scale)
//   B_real[n,k] = e4m3(B_fp8[n,k]) * b_s[n/128, k/128] (per-128x128 weight block scale)
// Scales passed as fp32 (loader decodes e8m0 -> fp32). Correctness-first; accum fp32.
#pragma once
#include <cstdint>
#include <cuda_runtime.h>

// A_fp8:[M,K] e4m3 bytes; a_s:[M, K/128] f32; B_fp8:[N,K] e4m3 bytes; b_s:[N/128, K/128] f32; C:[M,N] f32.
void fp8_block_gemm(float* C, const uint8_t* A_fp8, const float* a_s,
                    const uint8_t* B_fp8, const float* b_s,
                    int M, int N, int K, cudaStream_t stream = 0);
