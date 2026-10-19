// megakernel.h — persistent counter-synced fused kernels (flag-gated, MEGAKERNEL=1).
#pragma once
#include <cuda_runtime.h>
#include <cstdint>
extern "C" {
// M1: fused input_rmsnorm + Q/K/V projection for mtok=1.
// Produces q[qd], k[kd], v[kd] from h[H] (residual) + g (input_layernorm bf16) + FP4 proj weights.
// vp==nullptr => full layer (v = k copy done inside). eps = RMSNorm epsilon.
void mega_qkv(float* q, float* k, float* v, const float* h, const uint16_t* g,
              const uint8_t* qp, const uint8_t* qs, float qwg,
              const uint8_t* kp, const uint8_t* ks, float kwg,
              const uint8_t* vp, const uint8_t* vs, float vwg,
              int H, int qd, int kd, float eps, cudaStream_t s);
// M2: fused o_proj + post_attention_norm + residual (h += post_norm(o_proj(ao))).
void mega_oproj(float* hres, const float* ao, const uint16_t* g,
                const uint8_t* op_p, const uint8_t* op_s, float owg,
                int H, int qd, float eps, cudaStream_t s);
// Full-step Stage-1: GRID-COOPERATIVE input_rmsnorm + QKV (grid-barrier reduce; foundation for the layer loop).
void mega_attn_front(float* q, float* k, float* v, const float* h, const uint16_t* g,
                     const uint8_t* qp, const uint8_t* qs, float qwg,
                     const uint8_t* kp, const uint8_t* ks, float kwg,
                     const uint8_t* vp, const uint8_t* vs, float vwg,
                     int H, int qd, int kd, float eps, cudaStream_t s);
}
