// nvfp4_dense.h — optional NVFP4 replacement for the dense/MLA weights, keyed by FP8 device pointer.
//
// WHY A REGISTRY AND NOT A NEW CALL PATH. `fp8_block_gemm` is the single entry point for every dense
// linear in the engine -- compressed_attn, compressed_decode, indexer, dspark_real all route through
// it. Registering an NVFP4 replacement against the FP8 weight POINTER lets one `if` at the top of
// that function serve all of them, with zero edits at the call sites and zero risk to the shapes
// this does not cover.
//
// DEFAULT IS OFF AND BYTE-IDENTICAL. With no overlay registered the lookup misses and the original
// FP8 path runs unchanged. `original-fp8-path/*.orig` holds untouched copies of every file this
// feature touches.
//
//   DSV4_NVFP4_OVERLAY=<dir>   load the overlay and register its tensors
//   DSV4_NVFP4_OFF=1           registered but bypassed, for an A/B against the same process
#pragma once
#include <cstdint>
#include <cstddef>
#include <cuda_runtime.h>

struct NvFp4Weight {
    const uint8_t* packed;   // [N, K/2]  E2M1, two codes per byte, low nibble first
    const uint8_t* scale;    // [N, K/16] e4m3 group scale
    float          global;   // fp32: dequant step = e4m3(scale) / global
    int            N, K;
};

// Register/lookup by the FP8 weight pointer the engine already passes to fp8_block_gemm.
void nvfp4_register(const uint8_t* fp8_weight, const NvFp4Weight& w);
const NvFp4Weight* nvfp4_lookup(const uint8_t* fp8_weight);
bool  nvfp4_enabled();          // false unless an overlay was loaded and DSV4_NVFP4_OFF is unset
int   nvfp4_count();
void  nvfp4_force_enable(bool on);   // tests only: lookup is gated on the overlay having loaded

// C[1,N] = A[1,K] (fp8 e4m3 x a_s per 128) @ dequant(B)^T. Returns false if the shape is unsupported.
bool nvfp4_gemv_m1(float* C, const uint8_t* A_fp8, const float* a_s,
                   const NvFp4Weight& w, cudaStream_t stream);

// The M=K verify path. Must be routed whenever M=1 is, or AR and verify compute different functions.
bool nvfp4_gemv_mk(float* C, const uint8_t* A_fp8, const float* a_s, const NvFp4Weight& w,
                   int M, cudaStream_t stream);

// wo_a takes FLOAT activations and grouped rows, so it needs its own entry point.
bool nvfp4_ogroup_gemv(float* out, const float* o, const uint8_t* wo_fp8,
                       int G, int R, int Kd, cudaStream_t stream);

bool nvfp4_ogroup_mk(float* out, const float* o, const uint8_t* wo_fp8,
                     int M, int G, int R, int Kd, cudaStream_t stream);

// Load an overlay directory written by tools/requant_dense_nvfp4.py. `resolve` maps a checkpoint
// tensor name to the FP8 device pointer the engine holds for it (null if absent).
int nvfp4_load_overlay(const char* dir, const uint8_t* (*resolve)(const char*, void*), void* ctx);
