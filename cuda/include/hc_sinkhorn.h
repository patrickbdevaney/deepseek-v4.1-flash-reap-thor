// hc_sinkhorn.h — Hyper-Connections split + Sinkhorn (deepseek_v4 hc_split_sinkhorn).
// From 24 mixes/token -> pre[hc], post[hc], comb[hc,hc] (comb made doubly-stochastic via Sinkhorn).
// See reference/DEEPSEEK_V4_MODELING_NOTES.md §2 ; kernel.py:371-438.
#pragma once
#include <cuda_runtime.h>

// mixes:[n, (2+hc)*hc] f32 ; hc_scale:[3] ; hc_base:[(2+hc)*hc] ;
// pre:[n,hc] ; post:[n,hc] ; comb:[n,hc,hc] (row-major).
void hc_sinkhorn(float* pre, float* post, float* comb,
                 const float* mixes, const float* hc_scale, const float* hc_base,
                 int n, int hc, int iters, float eps, cudaStream_t stream = 0);
