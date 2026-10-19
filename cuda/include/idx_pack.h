#pragma once
// idx_pack.h — MXFP4 storage for the DSA index cache. BIT-EXACT, and that is the whole point.
//
// WHAT IS ALREADY TRUE. Every row written into the index cache goes through
// `hadamard(...)` then `act_quant_fp4sim(out, groups, 128, 32, 128)`
// (kernels/compressor.cu:500/:541, kernels/compressed_decode.cu:510). That kernel computes
//
//     scale = exp2f(ceilf(log2f(amax / 6)));            // an EXACT power of two
//     x[i]  = round_e2m1(clamp(x[i]/scale, -6, 6)) * scale;   // dequantised, written back
//
// so the value stored in the FP32 cache is already exactly `g * 2^k` where g is one of the fifteen
// E2M1 grid points {0, +-0.5, +-1, +-1.5, +-2, +-3, +-4, +-6} and k is an integer. The cache holds
// 4.25 bits of information in 32 bits per element.
//
// WHY THE ROUND TRIP IS EXACT, not merely close. Packing keeps the 4-bit code and the exponent
// byte; unpacking computes `g * 2^k` again. Multiplying a float by a power of two is exact in IEEE
// 754 (it only adds to the exponent field) as long as the result neither overflows nor goes
// subnormal, and E2M1 magnitudes are bounded by 6 while `scale` came from the data itself. So
// unpack(pack(x)) == x bit-for-bit, and the acceptance test for this change is `memcmp`, not a
// tolerance and not a benchmark.
//
// LAYOUT. One row = INDEX_HEAD_DIM (128) elements = 4 blocks of 32:
//     bytes [0, 64)   : 128 E2M1 codes, two per byte, element 2i in the low nibble
//     bytes [64, 68)  : 4 UE8M0 scale bytes, one per 32-element block
// 68 B against 512 B, a 7.53x reduction -- and 68 B/token is exactly what SGLang ships for the
// DeepSeek-V4 FP4 indexer pool, which is a useful independent check on the arithmetic.
//
// The E2M1 code is the standard OCP MX ordering: bit 3 sign, bits 2..0 index into
// {0, 0.5, 1, 1.5, 2, 3, 4, 6}. UE8M0 stores the exponent of a power of two as `k + 127`, which is
// exactly the fp32 exponent field, so the scale survives losslessly in one byte.
#include <cstdint>
#include <cmath>
#include "deepseek_v4.h"
using namespace dsv4;

namespace dsv4idx {

static const int PACK_BLOCK = 32;
static const int IHD         = dsv4::INDEX_HEAD_DIM;
static const int PACK_NB    = IHD / PACK_BLOCK;               // 4
static const int PACK_BYTES = IHD / 2 + PACK_NB;              // 64 + 4 = 68
static_assert(IHD % PACK_BLOCK == 0, "index head dim must tile the MX block");

__host__ __device__ __forceinline__ float e2m1_value(int code) {
    const float mag[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    float m = mag[code & 7];
    return (code & 8) ? -m : m;
}

// Exact inverse of round_e2m1: the stored magnitude is one of eight grid points, so this is a
// lookup, not a rounding. Anything not on the grid means the write path changed and the caller
// should fail loudly rather than silently store the nearest code.
__host__ __device__ __forceinline__ int e2m1_code(float v) {
    // SIGN COMES FROM THE SIGN BIT, NOT FROM `v < 0`. round_e2m1 returns `(v<0) ? -m : m`, so a
    // negative input that rounds to zero is stored as NEGATIVE ZERO -- and `-0.0f < 0.0f` is false,
    // so testing the value drops the sign and the round trip stops being bit-exact. It is still
    // numerically perfect, which is exactly why a tolerance would have passed it: the first version
    // of this function had the bug and the gate caught 16,011 mismatches at worst |delta| = 0.
    float a = fabsf(v);
    int c = (a == 0.f) ? 0 : (a == 0.5f) ? 1 : (a == 1.f) ? 2 : (a == 1.5f) ? 3
          : (a == 2.f) ? 4 : (a == 3.f) ? 5 : (a == 4.f) ? 6 : 7;
    return signbit(v) ? (c | 8) : c;
}

// The fp32 exponent field IS the UE8M0 byte for an exact power of two.
__host__ __device__ __forceinline__ uint8_t ue8m0_from_pow2(float s) {
    uint32_t u;
#ifdef __CUDA_ARCH__
    u = __float_as_uint(s);
#else
    __builtin_memcpy(&u, &s, 4);
#endif
    return (uint8_t)((u >> 23) & 0xFF);
}
__host__ __device__ __forceinline__ float pow2_from_ue8m0(uint8_t e) {
    uint32_t u = ((uint32_t)e) << 23;
#ifdef __CUDA_ARCH__
    return __uint_as_float(u);
#else
    float f; __builtin_memcpy(&f, &u, 4); return f;
#endif
}

// Read element `i` of a packed row. One nibble extract, one lookup, one exponent add.
__host__ __device__ __forceinline__ float idx_unpack(const uint8_t* row, int i) {
    uint8_t b = row[i >> 1];
    int code = (i & 1) ? (b >> 4) : (b & 0xF);
    return e2m1_value(code) * pow2_from_ue8m0(row[IHD / 2 + (i >> 5)]);
}

} // namespace dsv4idx
