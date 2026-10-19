// mla_decode.h — M=1 KV-cache decode for a pure-sliding MLA layer (STRUCTURAL_PLAN Step 4, milestone 1).
// Splits mla_forward into (a) prefill that WRITES the window-KV cache, (b) a single-token decode step that
// reads the cache, appends the new token's KV, and attends over the sliding window. Reuses the Gate-K
// primitives (fp8_block_gemm, rmsnorm, rope, act_quant, sparse_attn, ogroup_gemm) — identical per-row math,
// so decode(pos) reproduces prefill's out[pos] bit-for-bit (equivalence gate: tests/gate_mla_decode.cu).
#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include "mla_forward.h"   // MLAWeights

// Compute window-KV for prefill tokens x[0..s-1] into kvcache[0..s-1][HEAD_DIM] (fp32, NoPE dims fp8sim'd,
// RoPE dims rotated). kvcache must hold >= s rows. This is exactly the kv mla_forward computes internally.
void mla_cache_kv(float* kvcache, const float* x, const MLAWeights& w, int s, cudaStream_t stream = 0);

// One decode step for the token x[1,DIM] at absolute position `pos`. Appends its KV to kvcache[pos], then
// attends q(pos) over the sliding window [max(0,pos-WINDOW+1) .. pos] of kvcache -> out[1,DIM].
// kvcache must already hold rows [0..pos-1] (from mla_cache_kv + prior decode steps).
void mla_decode_step(float* out, const float* x, const MLAWeights& w, float* kvcache, int pos,
                     cudaStream_t stream = 0);

// M=K VERIFY step (spec-decode): process K tokens x[K,DIM] at positions [pos..pos+K-1] in ONE forward — GEMMs
// at M=K read the weights ONCE for all K tokens (the spec-decode bandwidth win). Appends their KV to
// kvcache[pos..pos+K-1]; query i attends the sliding window [max(0,pos+i-W+1)..pos+i] (causal among the K +
// the existing cache). Equivalent to K sequential mla_decode_step calls (gate: tests/gate_mla_verify.cu).
void mla_decode_step_dp(float* out, const float* x, const MLAWeights& w, float* kvcache, const int* d_pos, int nkv, cudaStream_t stream=0);
void dpos_incr(int* d_pos, cudaStream_t stream=0);
void mla_verify_step(float* out, const float* x, const MLAWeights& w, float* kvcache, int pos, int K,
                     cudaStream_t stream = 0);
