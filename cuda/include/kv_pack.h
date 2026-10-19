#pragma once
// kv_pack.h — FP8 E4M3 + UE8M0 storage for the MAIN KV cache. BIT-EXACT, and that is the point.
// DECODE_LADDER item 1b.2; the arithmetic and the citations are in KV_PRECISION_FINDINGS.md.
//
// WHAT IS ALREADY TRUE. Every row written into a KV cache in this engine ends in
// `act_quant_fp8sim(row, rows, NOPE_DIM=448, 64, HEAD_DIM=512)` (kernels/mla_attn.cu:378), which is
//
//     amax  = max(|x| over the 64-wide group, 1e-4)
//     scale = exp2f(ceilf(log2f(amax / 448)));                  // an EXACT power of two
//     x[i]  = (float)(half)__nv_fp8_e4m3(clamp(x[i]/scale, -448, 448)) * scale;   // written back
//
// so dims 0..447 already sit exactly on the E4M3 grid times an exact power of two, and are then
// stored in FP32. The cache holds 8 bits of information in 32 bits per element. Dims 448..511 are
// the RoPE half and are never passed to act_quant -- they stay FP32 here, because that is the one
// component the literature says is fragile (KV_PRECISION_FINDINGS.md §5).
//
// WHY THE ROUND TRIP IS EXACT, not merely close. We keep the E4M3 byte and the exponent byte.
// Unpacking computes `(float)(half)e4m3_byte * 2^k` -- the same two conversions and the same
// multiply the write path performed, in the same order. Multiplying a float by a power of two is
// exact in IEEE 754 (it only adds to the exponent field) and the product is the one that was
// already stored, so it neither overflows nor goes subnormal. unpack(pack(x)) == x BIT-FOR-BIT, and
// the acceptance test is memcmp, not a tolerance and not a benchmark.
//
// LAYOUT. One row = 720 B (the payload is 711 B; 720 is the smallest 16 B-aligned stride, which is
// what lets `sparse_attn`'s block gather the row with uint4 loads):
//     [  0, 448)  448 E4M3 bytes            dims 0..447
//     [448, 704)   64 FP32 RoPE values      dims 448..511      (16 B aligned: 448 % 16 == 0)
//     [704, 711)    7 UE8M0 scale bytes     one per 64-wide group
//     [711, 720)    padding
// 2048 -> 711 B of payload in a 720 B stride = 2.844x on the allocation. UE8M0 stores the exponent
// of a power of two as `k + 127`, which IS the fp32 exponent field, so the scale survives in one
// byte. Never store fp32 scales: that is what made llama.cpp's q4_0 KV use more RSS than f16.
//
// 720 is divisible by 4, so a packed cache is still addressed as `base + (size_t)row * g_kv_rowf`
// with `g_kv_rowf` floats per row (512 unpacked, 180 packed). That is what keeps this change to a
// stride substitution at ~30 call sites instead of a retyping of every cache pointer in the engine.
#include <cstdint>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include "deepseek_v4.h"

namespace dsv4kv {

static const int KVP_D     = dsv4::HEAD_DIM;                  // 512
static const int KVP_NOPE  = dsv4::NOPE_DIM;                  // 448
static const int KVP_ROPE  = dsv4::ROPE_DIM;                  // 64
static const int KVP_BLOCK = 64;                              // act_quant_fp8sim's group width
static const int KVP_NB    = KVP_NOPE / KVP_BLOCK;            // 7
static const int KVP_OFF_ROPE  = KVP_NOPE;                    // 448
static const int KVP_OFF_SCALE = KVP_NOPE + KVP_ROPE * 4;     // 704
static const int KVP_PAYLOAD   = KVP_OFF_SCALE + KVP_NB;      // 711
static const int KVP_ROWB      = 720;                         // 16 B-aligned row stride
static const int KVP_ROWF      = KVP_ROWB / 4;                // 180 floats -- see header note
static_assert(KVP_NOPE % KVP_BLOCK == 0, "nope dims must tile the quant block");
static_assert(KVP_ROWB % 16 == 0 && KVP_ROWB >= KVP_PAYLOAD, "row stride");
static_assert(KVP_OFF_ROPE % 16 == 0, "rope plane must be 16 B aligned");

// The fp32 exponent field IS the UE8M0 byte for an exact power of two.
__host__ __device__ __forceinline__ uint8_t kv_ue8m0(float s) {
    uint32_t u;
#ifdef __CUDA_ARCH__
    u = __float_as_uint(s);
#else
    __builtin_memcpy(&u, &s, 4);
#endif
    return (uint8_t)((u >> 23) & 0xFF);
}
__host__ __device__ __forceinline__ float kv_pow2(uint8_t e) {
    uint32_t u = ((uint32_t)e) << 23;
#ifdef __CUDA_ARCH__
    return __uint_as_float(u);
#else
    float f; __builtin_memcpy(&f, &u, 4); return f;
#endif
}

// e4m3 byte -> float, VIA HALF, because that is the path act_quant_fp8sim took when it wrote the
// value back. __nv_cvt_fp8_to_halfraw is exact (e4m3's whole range is inside half's) and
// __half2float is exact, so this is the identity on the stored value -- but going through a
// different conversion would be reasoning about two paths instead of replaying one.
__host__ __device__ __forceinline__ float kv_e4m3_to_float(uint8_t b) {
    __half_raw hr = __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)b, __NV_E4M3);
#ifdef __CUDA_ARCH__
    return __half2float(*reinterpret_cast<__half*>(&hr));
#else
    return (float)(*reinterpret_cast<__half*>(&hr));
#endif
}

// Read element `j` of a packed row.
__host__ __device__ __forceinline__ float kv_unpack(const uint8_t* row, int j) {
    if (j < KVP_NOPE)
        return kv_e4m3_to_float(row[j]) * kv_pow2(row[KVP_OFF_SCALE + (j >> 6)]);
    const float* rope = (const float*)(row + KVP_OFF_ROPE);
    return rope[j - KVP_NOPE];
}

} // namespace dsv4kv

// ---- the engine-wide switch ------------------------------------------------------------------
// g_kv_pack 0 = FP32 rows (2048 B), the pre-1b.2 path, byte-for-byte unchanged.
//           1 = packed rows (720 B).
// g_kv_rowf is the floats-per-row stride the two modes share, so cache pointer arithmetic is
// `base + (size_t)row * g_kv_rowf` in both. Set once, before any cache is allocated.
extern int g_kv_pack;
extern int g_kv_rowf;
void kv_pack_init();                        // reads DSV4_KV_PACK; idempotent
void kv_pack_init_seqmax(int seqmax);       // DSV4_KV_PACK wins; unset => auto above 32768

static inline float* kv_row(float* base, long long i) { return base + (size_t)i * (size_t)g_kv_rowf; }
static inline const float* kv_row(const float* base, long long i) { return base + (size_t)i * (size_t)g_kv_rowf; }
static inline size_t kv_rows_bytes(long long rows) { return (size_t)rows * (size_t)g_kv_rowf * 4; }

// PRODUCER SEAM. Every KV write path in this engine does
//     gemm -> rmsnorm -> rope -> act_quant_fp8sim,  writing straight into the cache row.
// `kv_stage` returns where the fp32 rows should be written and `kv_commit` finishes the store:
//   * FP32 mode: kv_stage returns the cache row itself and kv_commit IS the act_quant_fp8sim call
//     that was already there -- the same kernel on the same memory, so the before-arm is unchanged.
//   * packed mode: kv_stage returns dmalloc scratch and kv_commit quantises AND packs in one pass,
//     computing the identical amax/scale/e4m3 code and storing the byte instead of the dequantised
//     float. No FP32 copy of the cache is ever materialised (KV_PRECISION_FINDINGS.md §6).
float* kv_stage(float* cache_row, int rows);
void   kv_commit(float* cache_row, float* staged, int rows, cudaStream_t stream);
void   kv_stage_free(float* staged, float* cache_row);

// Device-index stores, for the CUDA-graph decode paths (d_idx / conditional commit).
void kv_store_at(float* cache_base, const float* src_row, const int* d_idx, cudaStream_t stream);
void kv_store_comp(float* comp_base, const float* cand, const int* d_T, const int* d_pos,
                   int ratio, cudaStream_t stream);
