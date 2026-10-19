// compressed_block.h — full Block forward for a COMPRESSED layer (2..42). Identical structure to
// block_forward but the attention is compressed_attn_forward (window ⊕ compressed KV). model.py Block.
#pragma once
#include "compressed_attn.h"
#include "moe.h"

struct CompressedBlockWeights {
    CompressedAttnWeights attn;
    MoEWeights ffn;
    const float *attn_norm, *ffn_norm;                 // [dim]
    const float *hc_attn_fn, *hc_attn_scale, *hc_attn_base;
    const float *hc_ffn_fn,  *hc_ffn_scale,  *hc_ffn_base;
    int dim, hc, win, ratio;
};

// x:[s, hc, dim] (HC state, b=1), input_ids:[s] -> out:[s, hc, dim].
void compressed_block_forward(float* out, const float* x, const int* input_ids,
                              const CompressedBlockWeights& w, int s, int sinkhorn_iters, float eps,
                              cudaStream_t stream = 0);
