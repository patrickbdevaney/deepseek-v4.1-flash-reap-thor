// compressed_decode.h — M=1 KV-cache decode for a COMPRESSED MLA layer (STRUCTURAL_PLAN Step 4, milestone 2).
// Window KV cache (sliding) + append-only compressed KV cache (compressor_emit_group every `ratio` tokens) +
// sparse_attn over [window ⊕ compressed]. Strided variant (ratio!=4): deterministic compressed idxs
// (t < (pos+1)/ratio). Reuses the bit-exact primitives from mla_decode + compressor_emit_group.
#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include "compressed_attn.h"   // CompressedAttnWeights

// Prefill: populate the window-KV cache (win_kv[0..s0-1]) and the compressed-KV cache (one row per COMPLETE
// group). Sets *T = number of compressed rows emitted. Non-overlap (strided) layers only for now.
void compressed_attn_cache(float* win_kv, float* comp_kv, int* T, const float* x,
                           const CompressedAttnWeights& w, int s0, int ratio, float eps, cudaStream_t stream = 0);

// One decode step (strided / ratio!=4) for the token at position `pos`. x_full supplies the completing group's
// tokens for the compressor emit (from xhist in a real loop). Appends window KV at win_kv[pos], emits a
// compressed row when (pos+1)%ratio==0, attends q(pos) over [win_kv[0..pos] ⊕ comp_kv[0..*T-1]] -> out[1,DIM].
void compressed_decode_step_strided(float* out, const float* x_cur, const float* x_full, int pos, const CompressedAttnWeights& w,
                                    float* win_kv, float* comp_kv, int* T, int ratio, float eps,
                                    cudaStream_t stream = 0);

// ratio-4 (DSA indexer) variant: adds the indexer-compressor cache (idx_ckv) for scoring; the main compressed
// KV (comp_kv) is OVERLAP-pooled. Decode: score query vs idx_ckv -> top-k main-compressed rows to attend.
void compressed_attn_cache_r4(float* win_kv, float* comp_kv, float* idx_ckv, int* T, const float* x,
                              const CompressedAttnWeights& w, int s0, int ratio, float eps, cudaStream_t stream = 0);
void compressed_decode_step_indexer(float* out, const float* x_cur, const float* x_full, int pos, const CompressedAttnWeights& w,
                                    float* win_kv, float* comp_kv, float* idx_ckv, int* T, int ratio,
                                    float eps, cudaStream_t stream = 0);

// M=K VERIFY (spec-decode): K tokens at [pos..pos+K-1], GEMMs at M=K (weights once). x_full supplies the
// attention-input history (xin) for the COMPRESSOR ONLY; `x_cur` is this block's attention input,
// which is what the Q/KV build and the indexer projection actually read. Separating them is what
// lets `xin` be a small ring: nothing but compressor_emit_group indexes it by absolute position,
// and that never looks further back than 2*ratio. ≡ K sequential decode steps.
void compressed_verify_step_strided(float* out, const float* x_cur, const float* x_full, int pos, int K, const CompressedAttnWeights& w,
                                    float* win_kv, float* comp_kv, int* T, int ratio, float eps, cudaStream_t stream = 0);
void compressed_verify_step_indexer(float* out, const float* x_cur, const float* x_full, int pos, int K, const CompressedAttnWeights& w,
                                    float* win_kv, float* comp_kv, float* idx_ckv, int* T, int ratio, float eps, cudaStream_t stream = 0);
