// Minimal viability probe: does cuBLASLt find an FP4 (MXFP4, e2m1 + ue8m0 block32) matmul algo on Thor sm_110?
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cstdio>
#define CK(x) do{auto e=(x); if(e){printf("ERR %s = %d\n",#x,(int)e);} }while(0)
int main(){
  int M=16,N=2048,K=4096;                        // decode-ish FFN shape
  cublasLtHandle_t lt; CK(cublasLtCreate(&lt));
  cublasLtMatmulDesc_t op; CK(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  cublasOperation_t T=CUBLAS_OP_T, Nn=CUBLAS_OP_N;
  CK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &T, sizeof(T)));
  CK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &Nn, sizeof(Nn)));
  int32_t sm = CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;   // MXFP4 block32 e8m0 (== our expert scale)
  CK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &sm, sizeof(sm)));
  CK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &sm, sizeof(sm)));
  void *as,*bs; cudaMalloc(&as,(size_t)M*K/32); cudaMalloc(&bs,(size_t)N*K/32);
  CK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &as, sizeof(as)));
  CK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &bs, sizeof(bs)));
  cublasLtMatrixLayout_t la,lb,lc;
  CK(cublasLtMatrixLayoutCreate(&la, CUDA_R_4F_E2M1, K, M, K));   // A: KxM (op-T), ld=K
  CK(cublasLtMatrixLayoutCreate(&lb, CUDA_R_4F_E2M1, K, N, K));   // B: KxN, ld=K
  CK(cublasLtMatrixLayoutCreate(&lc, CUDA_R_16BF, M, N, M));      // C/D: MxN, ld=M
  cublasLtMatmulPreference_t pref; CK(cublasLtMatmulPreferenceCreate(&pref));
  size_t ws=32ull<<20; CK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)));
  cublasLtMatmulHeuristicResult_t res[8]; int found=0;
  cublasStatus_t st = cublasLtMatmulAlgoGetHeuristic(lt, op, la, lb, lc, lc, pref, 8, res, &found);
  printf("\n[cuBLASLt FP4 (MXFP4 e2m1/ue8m0) on sm_110]  heuristic status=%d  algos_found=%d\n", (int)st, found);
  printf("VERDICT: %s\n", (st==0 && found>0) ? "✅ cuBLASLt FP4 GEMM VIABLE on Thor" : "❌ no FP4 algo (library FP4 not available on sm_110 in CUDA 13)");
  return 0;
}
