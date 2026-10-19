// attention.h — single-sequence scaled-dot-product attention (fp32) for the Gemma-4 forward.
// All activations flow in fp32, row-major. Causal masking, GQA, and optional sliding window.
#pragma once
#include <cuda_runtime.h>

// Single-sequence causal SDPA with GQA and optional sliding window. All fp32, row-major.
// Q: [seq, n_heads, head_dim]   K,V: [seq, n_kv, head_dim]   out: [seq, n_heads, head_dim]
// GQA: each kv head is shared by (n_heads/n_kv) query heads: q head h uses kv head h/(n_heads/n_kv).
// Causal: query position i attends to key positions j where j<=i. If sliding_window>0, additionally
//   require (i - j) < sliding_window  (i.e. j in [i-sliding_window+1, i]). scaling multiplies the
//   q.k scores (pass 1.0 for Gemma4 — NO 1/sqrt(d)). Softmax computed in fp32. No attention softcap.
void sdpa(float* out, const float* Q, const float* K, const float* V,
          int seq, int n_heads, int n_kv, int head_dim, int sliding_window, float scaling, cudaStream_t s);
