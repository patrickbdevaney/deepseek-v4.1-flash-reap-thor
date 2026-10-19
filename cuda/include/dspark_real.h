// dspark_real.h — real DSpark block-diffusion head pieces (DSpark-head repo model.py:744-874).
// Composable core: main_x (tap-proj) + Markov head. The block attention (main-KV window ⊕ block) and the
// host AR/verify loop are built on top (see DSPARK_HEAD_BUILD.md). Reuses gated primitives throughout.
#pragma once
#include <cstdint>
#include <cuda_runtime.h>

// main_x = main_norm( main_proj(main_hidden) ). main_hidden:[s, 3*dim] (cat of mean-pooled taps L40/41/42).
// main_proj: fp8 [dim, 3*dim] + e8m0->f32 scale.  main_norm: rmsnorm weight [dim] (f32). -> main_x:[s, dim].
void dspark_main_x(float* main_x, const float* main_hidden, const uint8_t* main_proj, const float* main_proj_s,
                   const float* main_norm, int s, int dim, float eps, cudaStream_t stream = 0);

// DSparkMarkovHead: token_ids:[n] -> logits_bias:[n, vocab] (bigram correction) + markov_embed:[n, rank].
// markov_w1/w2: [vocab, rank] BF16, read natively (LOOP_LOG Finding 26 — no f32 dequant).
void dspark_markov(float* logits_bias, float* markov_embed, const int* token_ids,
                   const void* markov_w1, const void* markov_w2, int n, int vocab, int rank,
                   cudaStream_t stream = 0);

// Piece 2: mean-pool one tapped layer state h:[s,hc,d] over hc -> writes into main_hidden[:, slot*d : slot*d+d]
// (main_hidden is [s, n_taps*d]; call once per tap layer 40/41/42 with slot 0/1/2).
void dspark_tap_pool(float* main_hidden, const float* h, int s, int hc, int d, int slot, int n_taps,
                     cudaStream_t stream = 0);

// Piece 3: forward_head. x_block:[s, block, hc, d] (draft block hidden). first_ids:[s] (real token per anchor).
// hc_head params + norm + shared lm_head (BF16, native) + markov tables (BF16, native). -> output_ids:[s, block+1] greedy proposed block.
void dspark_forward_head(int* output_ids, const float* x_block, const int* first_ids,
                         const float* hc_head_fn, const float* hc_head_scale, const float* hc_head_base,
                         const float* norm, const void* lm_head, const void* markov_w1, const void* markov_w2,
                         int s, int block, int hc, int d, int vocab, int rank, float eps,
                         float* margins = nullptr, cudaStream_t stream = 0);
