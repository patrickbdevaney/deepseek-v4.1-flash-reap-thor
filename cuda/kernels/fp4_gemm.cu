// fp4_gemm.cu — NVFP4 block-scaled GEMM via cublasLt (CUDA 13, Blackwell sm_110a).
// Recipe per reference/CUBLASLT_FP4_RECIPE.md (cuBLAS 13.3 "1D Block Scaling Factors Layout").
#include "fp4_gemm.h"
#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cstdio>
#include <cstring>

// ---- scale-factor swizzle (logical [outer, K/16] E4M3 -> 128x4-tile layout) ----
// Tile = 128 outer x 4 block-cols = 512 E4M3 bytes. Tiles row-major over (outer/128, block/4).
// within-tile = (m%32)*16 + ((m%128)/32)*4 + (b%4).
static inline int round_up(int x, int m) { return (x + m - 1) / m * m; }

size_t nvfp4_scale_buffer_bytes(int outer, int K) {
    int KB = K / 16;                       // blocks along K
    int tile_rows = round_up(outer, 128) / 128;
    int tile_cols = round_up(KB, 4) / 4;
    return (size_t)tile_rows * tile_cols * 512;
}

void nvfp4_swizzle_scales(const uint8_t* logical, uint8_t* swizzled, int outer, int K) {
    int KB = K / 16;
    int tile_cols = round_up(KB, 4) / 4;
    size_t total = nvfp4_scale_buffer_bytes(outer, K);
    memset(swizzled, 0, total);
    for (int m = 0; m < outer; ++m) {
        for (int b = 0; b < KB; ++b) {
            size_t tile = (size_t)((m / 128) * tile_cols + (b / 4));
            int within = (m % 32) * 16 + ((m % 128) / 32) * 4 + (b % 4);
            swizzled[tile * 512 + within] = logical[(size_t)m * KB + b];
        }
    }
}

#define LT_CHECK(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "cublasLt error %d at %s:%d\n", (int)s_, __FILE__, __LINE__); return (int)s_; } } while(0)

// ---- FP4-weight GEMV for single-token decode (W4A16: FP4 weight x fp32 activation) ----
// y[N] = sum_k (fp4(wp[n,k]) * e4m3(ws[n,k/16]) / w_gscale) * x[k].  Raw (unswizzled) weight+scales.
__device__ __forceinline__ float e2m1_dec(uint8_t c){ const float t[8]={0.f,.5f,1.f,1.5f,2.f,3.f,4.f,6.f}; float v=t[c&7]; return (c&8)?-v:v; }
__device__ __forceinline__ float e4m3_dec(uint8_t b){ int s=(b>>7)&1,e=(b>>3)&0xF,man=b&7; float v=(e==0)?(man*0.125f*0.015625f):(ldexpf(1.f+man*0.125f,e-7)); return s?-v:v; }
__device__ __forceinline__ __half2 dec_fp4x2(unsigned char b){ __half2_raw r=__nv_cvt_fp4x2_to_halfraw2((__nv_fp4x2_storage_t)b,__NV_E2M1); return *reinterpret_cast<__half2*>(&r); }
// PRODUCTION GEMV: HW FP4 decode (cvt.rn.f16x2.e2m1x2, 2 codes->half2) + fp16 acts (uint4 load) + half2 FMA.
// Replaces scalar table lookups (the memory-pipe instructions). One warp/output, shfl reduce.
__global__ void fp4_gemv_kernel(float* y, const uint8_t* wp, const uint8_t* ws, float wg_inv,
                                const __half* x, int N, int K){
    __shared__ float lut[256];
    for(int i=threadIdx.x;i<256;i+=blockDim.x) lut[i]=e4m3_dec((uint8_t)i)*wg_inv;
    __syncthreads();
    int lane=threadIdx.x&31, n=blockIdx.x*(blockDim.x>>5)+(threadIdx.x>>5);
    if(n>=N) return;
    const unsigned* wpn=(const unsigned*)(wp+(size_t)n*(K/2)); const uint8_t* wsn=ws+(size_t)n*(K/16);
    float acc=0.f; int nu=K/8;
    for(int vi=lane; vi<nu; vi+=32){
        unsigned w=__ldcs(&wpn[vi]); int k=vi*8; float sc=lut[__ldcs(&wsn[k>>4])];   // streaming weight loads (evict-first, keep acts cached)
        uint4 xpk=*(const uint4*)(x+k); const __half2* xh2=(const __half2*)&xpk;  // 4 half2 (8 fp16 acts)
        const unsigned char* wb=(const unsigned char*)&w;
        __half2 a2=__float2half2_rn(0.f);
        #pragma unroll
        for(int b=0;b<4;++b){
            __half2_raw wr=__nv_cvt_fp4x2_to_halfraw2((__nv_fp4x2_storage_t)wb[b], __NV_E2M1); // 2 FP4 -> half2
            a2=__hfma2(*reinterpret_cast<__half2*>(&wr), xh2[b], a2);
        }
        acc += sc * (__half2float(__low2half(a2)) + __half2float(__high2half(a2)));
    }
    #pragma unroll
    for(int o=16;o>0;o>>=1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if(lane==0) y[n]=acc;
}
void fp4_gemv(float* y, const uint8_t* wp, const uint8_t* ws, float w_gscale, const void* x16,
              int N, int K, cudaStream_t s){
    int wpb=8; fp4_gemv_kernel<<<(N+wpb-1)/wpb,wpb*32,0,s>>>(y, wp, ws, 1.0f/w_gscale, (const __half*)x16, N, K);
}

// ---- batched W4A16 GEMM (FP4 weight x fp32 activation, NO activation quant) ----
// out[M,N] row-major = dequant(W[N,K]) @ x[M,K]^T. Weight row n read ONCE (shared), reused over all M.
// Accurate (no act-quant), any M, any alignment (byte loads), unswizzled raw scales. Self-consistent w/ fp4_gemv.
// warp-per-output-column: each warp reads weight row n ONCE (vectorized uint32 + LUT), accumulates
// M dots with the M activation rows. No big shared (high occupancy). M<=16.
__global__ void w4a16_gemm_kernel(float* out, const uint8_t* wp, const uint8_t* ws, float wg_inv,
                                  const __half* x16, int M, int N, int K){
    __shared__ float lut[256]; for(int i=threadIdx.x;i<256;i+=blockDim.x) lut[i]=e4m3_dec((uint8_t)i)*wg_inv;
    __syncthreads();
    int lane=threadIdx.x&31, n=blockIdx.x*(blockDim.x>>5)+(threadIdx.x>>5); if(n>=N) return;
    const unsigned* wpn=(const unsigned*)(wp+(size_t)n*(K/2)); const uint8_t* wsn=ws+(size_t)n*(K/16);
    float acc[16]; for(int m=0;m<M;++m) acc[m]=0.f;
    int nu=K/8; const int U=4;   // K-unroll: prefetch U independent weight loads before decode/FMA -> raises MLP (M=1 was serial-chain starved)
    for(int vi=lane; vi<nu; vi+=32*U){
        unsigned w[U]; int vv[U];
        #pragma unroll
        for(int u=0;u<U;++u){ int v=vi+u*32; vv[u]=v; if(v<nu) w[u]=__ldcs(&wpn[v]); }   // U loads issued together
        #pragma unroll
        for(int u=0;u<U;++u){ int v=vv[u]; if(v>=nu) continue; int k=v*8; float sc=lut[__ldcs(&wsn[k>>4])];
            const unsigned char* wb=(const unsigned char*)&w[u]; __half2 wv2[4];
            #pragma unroll
            for(int b=0;b<4;++b) wv2[b]=dec_fp4x2(wb[b]);
            for(int m=0;m<M;++m){ uint4 xpk=*(const uint4*)(x16+(size_t)m*K+k); const __half2* xm=(const __half2*)&xpk;
                __half2 a2=__float2half2_rn(0.f);
                #pragma unroll
                for(int b=0;b<4;++b) a2=__hfma2(wv2[b], xm[b], a2);
                acc[m]+=sc*(__half2float(__low2half(a2))+__half2float(__high2half(a2))); } } }
    for(int m=0;m<M;++m){
        #pragma unroll
        for(int o=16;o>0;o>>=1) acc[m]+=__shfl_down_sync(0xffffffffu,acc[m],o);
        if(lane==0) out[(size_t)m*N+n]=acc[m];
    }
}
void w4a16_gemm(float* out, const uint8_t* wp, const uint8_t* ws, float w_gscale, const void* x16,
                int M, int N, int K, cudaStream_t s){
    int wpb=8; w4a16_gemm_kernel<<<(N+wpb-1)/wpb,wpb*32,0,s>>>(out, wp, ws, 1.0f/w_gscale, (const __half*)x16, M, N, K);
}

int nvfp4_gemm(float* dD,
               const uint8_t* dA_packed, const uint8_t* dA_scales_swz, float a_gscale,
               const uint8_t* dB_packed, const uint8_t* dB_scales_swz, float b_gscale,
               int M, int N, int K,
               cublasLtHandle_t lt, void* workspace, size_t ws_bytes, cudaStream_t stream) {
    // TN GEMM, all operands col-major:
    //   A described [K,M] col-major (= our [M,K] row-major packed), op_T -> [M,K]
    //   B described [K,N] col-major (= our [N,K] row-major packed), op_N -> [K,N]
    //   D = [M,N] col-major, ld=M
    cublasLtMatmulDesc_t op;
    LT_CHECK(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasOperation_t opT = CUBLAS_OP_T, opN = CUBLAS_OP_N;
    LT_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT)));
    LT_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)));

    void* aptr = (void*)dA_scales_swz; void* bptr = (void*)dB_scales_swz;
    int32_t vmode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    LT_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &aptr, sizeof(aptr)));
    LT_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &bptr, sizeof(bptr)));
    LT_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &vmode, sizeof(vmode)));
    LT_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &vmode, sizeof(vmode)));

    cublasLtMatrixLayout_t lA, lB, lD;
    LT_CHECK(cublasLtMatrixLayoutCreate(&lA, CUDA_R_4F_E2M1, K, M, K)); // [K,M] col-major
    LT_CHECK(cublasLtMatrixLayoutCreate(&lB, CUDA_R_4F_E2M1, K, N, K)); // [K,N] col-major
    LT_CHECK(cublasLtMatrixLayoutCreate(&lD, CUDA_R_32F, M, N, M));     // [M,N] col-major

    // global per-tensor scales fold into alpha (block mode has no separate A/B global input)
    float alpha = a_gscale * b_gscale, beta = 0.0f;

    cublasLtMatmulPreference_t pref;
    LT_CHECK(cublasLtMatmulPreferenceCreate(&pref));
    LT_CHECK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                  &ws_bytes, sizeof(ws_bytes)));
    cublasLtMatmulHeuristicResult_t heur[8];
    int nres = 0;
    LT_CHECK(cublasLtMatmulAlgoGetHeuristic(lt, op, lA, lB, lD, lD, pref, 8, heur, &nres));
    if (nres == 0) { fprintf(stderr, "nvfp4_gemm: no cublasLt algo found (M=%d N=%d K=%d)\n", M, N, K); return -100; }

    cublasStatus_t st = cublasLtMatmul(lt, op, &alpha,
                                       dA_packed, lA, dB_packed, lB,
                                       &beta, dD, lD, dD, lD,
                                       &heur[0].algo, workspace, ws_bytes, stream);

    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(lA); cublasLtMatrixLayoutDestroy(lB); cublasLtMatrixLayoutDestroy(lD);
    cublasLtMatmulDescDestroy(op);
    LT_CHECK(st);
    return 0;
}
