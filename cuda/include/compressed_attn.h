// compressed_attn.h — full compressed-layer MLA attention forward (ratio-4 indexer layer, prefill).
// = mla_forward + main-compressor(kv_compress) + indexer(compress idxs) -> sparse_attn over
// [window KV ⊕ compressed KV] with [window idxs ⊕ indexer idxs]. model.py Attention.forward (compress_ratio!=0).
#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include "mla_forward.h"   // MLAWeights

struct CompressedAttnWeights {
    MLAWeights attn;                         // wq_a/wq_b/wkv/wo_b(+scales), q_norm, kv_norm, wo_a, attn_sink,
                                             // cosT/sinT = query YaRN freqs [s, rope_dim/2]
    // main compressor (rotate=False, d = head_dim = 512, overlap ratio-4)
    const float *mc_wkv, *mc_wgate, *mc_ape, *mc_norm;   // fp32
    const float *cc_cos, *cc_sin;            // compressed-position freqs [s/ratio, rope_dim/2] (shared w/ indexer)
    // indexer (rotate compressor + scoring)
    const unsigned char *idx_wq_b; const float *idx_wq_b_s, *idx_weights_proj;
    const float *idx_c_wkv, *idx_c_wgate, *idx_c_ape, *idx_c_norm;
    int index_n_heads, index_head_dim, index_topk;
};

// x:[s, dim] fp32 -> out:[s, dim] fp32 (b=1 prefill). win = sliding window, ratio = compress ratio.
void compressed_attn_forward(float* out, const float* x, const CompressedAttnWeights& w,
                             int s, int win, int ratio, float eps, cudaStream_t stream = 0);
