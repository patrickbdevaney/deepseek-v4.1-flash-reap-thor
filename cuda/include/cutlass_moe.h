// cutlass_moe.h — callable CUTLASS NVFP4 tensor-core GEMM (validated on Thor sm_110a).
#pragma once
#include <cuda_runtime.h>
#include <cstdint>
extern "C" {
// D[M,N] bf16 = alpha * dequant(A[M,K] E2M1 x SFA) @ dequant(B[N,K] E2M1 x SFB)^T
// A row-major, B col-major (weight [N,K] K-contiguous). SFA/SFB = CUTLASS-swizzled e4m3. workspace pre-alloc'd.
int  cutlass_nvfp4_gemm(void* D, const void* A_fp4, const void* B_fp4,
                        const void* SFA, const void* SFB, float alpha,
                        int M,int N,int K, void* workspace, cudaStream_t stream);
long cutlass_sfa_bytes(int M,int N,int K);
long cutlass_sfb_bytes(int M,int N,int K);
long cutlass_workspace_bytes(int M,int N,int K);
// convert our LINEAR e4m3 group-16 scales [rows][K/16] -> CUTLASS blocked SFA/SFB layout
void cutlass_swizzle_sfa(uint8_t* out,const uint8_t* in_linear,int M,int N,int K,cudaStream_t s);
void cutlass_swizzle_sfb(uint8_t* out,const uint8_t* in_linear,int M,int N,int K,cudaStream_t s);
}
