// block.h — full DeepSeek-V4 transformer Block forward (pure-sliding layer), model.py:695-707.
// hc_pre -> attn_norm -> mla_forward -> hc_post -> hc_pre -> ffn_norm -> moe_forward -> hc_post.
#pragma once
#include "mla_forward.h"
#include "moe.h"

struct BlockWeights {
    MLAWeights attn;
    MoEWeights ffn;
    const float *attn_norm, *ffn_norm;                 // [dim]
    const float *hc_attn_fn, *hc_attn_scale, *hc_attn_base;   // [(2+hc)*hc, hc*dim], [3], [(2+hc)*hc]
    const float *hc_ffn_fn,  *hc_ffn_scale,  *hc_ffn_base;
    int dim, hc;
};

// x:[s, hc, dim] (HC state, b=1), input_ids:[s] -> out:[s, hc, dim].
void block_forward(float* out, const float* x, const int* input_ids, const BlockWeights& w,
                   int s, int sinkhorn_iters, float eps, cudaStream_t stream = 0);
