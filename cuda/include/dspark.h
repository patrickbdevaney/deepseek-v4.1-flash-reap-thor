// dspark.h — DeepSeek-V4 DSpark MTP draft head (model.py MTPBlock). A pure-sliding Block wrapped with a
// token-embedding fusion: e=enorm(embed(ids)); xh=hnorm(x); x'=e_proj(e)+h_proj(xh); x'=block(x');
// logits = hc_head(x') -> norm -> lm_head. Reuses block_forward + hc_head + fp8_block_gemm + gemm_fp32.
#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "block.h"

struct DSparkWeights {
    BlockWeights block;                                   // mtp.0.* pure-sliding Block (attn+MoE+HC+norms)
    const uint8_t *e_proj, *h_proj; const float *e_proj_s, *h_proj_s;   // fp8 [dim,dim] + e8m0->f32 scale
    const float *enorm, *hnorm, *norm;                    // rmsnorm weights (bf16->f32)
    const float *hc_head_fn, *hc_head_scale, *hc_head_base;             // f32 (MTP's own head-collapse params)
    const float *lm_head;                                 // shared head.weight (bf16->f32) [vocab, dim]
    const __nv_bfloat16 *embed;                           // shared embed.weight (bf16) [vocab, dim]
    int dim, hc, vocab;
};

// x:[s, hc, dim] = main model's final HC state (BEFORE the main hc_head). input_ids:[s] (shifted next tokens).
// -> logits:[s, vocab] = the draft head's next-token prediction.
void dspark_head_forward(float* logits, const float* x, const int* input_ids, const DSparkWeights& w,
                         int s, float eps, cudaStream_t stream = 0);
