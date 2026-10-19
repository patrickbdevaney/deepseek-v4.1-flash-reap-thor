// v41_dev.cuh — device helpers shared by the V4.1 kernels: number formats and their rounding
// rules, written to reproduce the reference (inference/kernel.py + ops/lib/torch_kernels.py).
//
// The reference keeps its activations in bf16, so every place it materialises a bf16 tensor we
// round to bf16 too (bf16r). Our buffers are fp32 holding bf16-representable values; that keeps
// the arithmetic in fp32 where the reference's is fp32 and costs one rounding per store.
#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>

namespace v41d {

__device__ __forceinline__ float bf16r(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }
__device__ __forceinline__ float bf16f(uint16_t b) { return __uint_as_float(((uint32_t)b) << 16); }

// E8M0 scale byte -> 2^(b-127). 0xFF is NaN in the OCP spec; the checkpoint never stores it.
__device__ __forceinline__ float e8m0f(uint8_t b) { return __uint_as_float(b == 0 ? 0x00400000u : ((uint32_t)b << 23)); }

__device__ __forceinline__ float e4m3f(uint8_t b) {
    __half_raw h = __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)b, __NV_E4M3);
    return __half2float(__half(h));
}
// two e4m3 codes (low byte first) -> float2
__device__ __forceinline__ float2 e4m3x2f(uint16_t b2) {
    __half2_raw h = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)b2, __NV_E4M3);
    return __half22float2(__half2(h));
}
__device__ __forceinline__ uint8_t f2e4m3(float v) {           // RN, saturating (|v| <= 448)
    return (uint8_t)__nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
}

// 2 ** ceil(log2(v)) by the kernel's bit trick: exponent + 1 if any mantissa bit is set.
__device__ __forceinline__ float pow2_ceil(float v) {
    uint32_t b = __float_as_uint(v);
    int e = (int)((b >> 23) & 0xFF) - 127 + ((b & 0x7FFFFF) ? 1 : 0);
    return ldexpf(1.0f, e);
}
__device__ __forceinline__ uint8_t pow2_ceil_e8m0(float v) {     // the same scale as an E8M0 byte
    uint32_t b = __float_as_uint(v);
    int e = (int)((b >> 23) & 0xFF) + ((b & 0x7FFFFF) ? 1 : 0);
    return (uint8_t)(e < 0 ? 0 : (e > 254 ? 254 : e));
}

// E2M1: nearest representable, ties to the even code (PTX cvt.rn). Returns the CODE (sign bit 3).
__device__ __forceinline__ uint8_t fp4_code(float v) {
    float a = fabsf(v); uint8_t c;
    if (a <= 0.25f) c = 0; else if (a < 0.75f) c = 1; else if (a <= 1.25f) c = 2; else if (a < 1.75f) c = 3;
    else if (a <= 2.5f) c = 4; else if (a < 3.5f) c = 5; else if (a <= 5.0f) c = 6; else c = 7;
    return c | ((__float_as_uint(v) >> 28) & 0x8);
}
__constant__ static const float kFP4[16] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f,
                                            -0.f, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};
__device__ __forceinline__ float fp4f(uint8_t code) { return kFP4[code & 15]; }

// hardware E2M1x2 -> f16x2 (cvt.rn.f16x2.e2m1x2 on sm_100a/sm_110a; cuda_fp4.h emulates elsewhere).
// Low nibble -> .x, the order convert.py packs.
__device__ __forceinline__ __half2 e2m1x2_h2(uint8_t b) {
    __half2_raw r = __nv_cvt_fp4x2_to_halfraw2((__nv_fp4x2_storage_t)b, __NV_E2M1);
    return *reinterpret_cast<__half2*>(&r);
}

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

} // namespace v41d
