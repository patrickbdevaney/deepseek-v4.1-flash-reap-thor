// hc.h — Hyper-Connections compose (deepseek_v4, model.py:680-716) on top of the gated hc_sinkhorn.
// x is the HC state [bs, hc, d] (contiguous). Correctness-first, fp32.
#pragma once
#include <cuda_runtime.h>

// hc_pre: 4-copies -> 1. mixes = (flatten(x)@hc_fn^T)*rsqrt ; sinkhorn -> pre/post/comb ; y=Σ_j pre_j x_j.
// hc_fn:[(2+hc)*hc, hc*d]. -> y:[bs,d], post:[bs,hc], comb:[bs,hc,hc].
void hc_pre(float* y, float* post, float* comb, const float* x, const float* hc_fn,
            const float* hc_scale, const float* hc_base, int bs, int hc, int d,
            int sinkhorn_iters, float eps, cudaStream_t stream = 0);

// hc_post: 1 -> hc copies. y[j] = post_j * x_new + Σ_k comb[j,k] * residual_k. -> y:[bs,hc,d].
void hc_post(float* y, const float* x_new, const float* residual, const float* post,
             const float* comb, int bs, int hc, int d, cudaStream_t stream = 0);

// hc_head: 4 -> 1, no sinkhorn. pre = sigmoid(mixes*scale+base)+eps ; y = Σ_j pre_j x_j. hc_fn:[hc, hc*d].
void hc_head(float* y, const float* x, const float* hc_fn, const float* hc_scale,
             const float* hc_base, int bs, int hc, int d, float eps, cudaStream_t stream = 0);
