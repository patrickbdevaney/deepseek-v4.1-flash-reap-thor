// compressed_block.cu — full Block forward for a compressed layer. Mirrors block.cu, attn = compressed_attn_forward.
#include "compressed_block.h"
#include "hc.h"
#include "mla_attn.h"      // rmsnorm
#include <cstdio>
#define CU(x) do{cudaError_t e=(x); if(e){fprintf(stderr,"cuda %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)

void compressed_block_forward(float* out, const float* x, const int* input_ids,
                              const CompressedBlockWeights& w, int s, int iters, float eps, cudaStream_t stream) {
    const int bs = s, d = w.dim, hc = w.hc;
    float *x1, *post, *comb, *sub, *res2;
    CU(cudaMalloc(&x1,   (size_t)bs * d * 4));
    CU(cudaMalloc(&post, (size_t)bs * hc * 4));
    CU(cudaMalloc(&comb, (size_t)bs * hc * hc * 4));
    CU(cudaMalloc(&sub,  (size_t)bs * d * 4));
    CU(cudaMalloc(&res2, (size_t)bs * hc * d * 4));

    // --- attention block (compressed) ---
    hc_pre(x1, post, comb, x, w.hc_attn_fn, w.hc_attn_scale, w.hc_attn_base, bs, hc, d, iters, eps, stream);
    rmsnorm(x1, x1, w.attn_norm, bs, d, eps, true, stream);
    compressed_attn_forward(sub, x1, w.attn, s, w.win, w.ratio, eps, stream);
    hc_post(res2, sub, x, post, comb, bs, hc, d, stream);

    // --- feed-forward (MoE) block ---
    hc_pre(x1, post, comb, res2, w.hc_ffn_fn, w.hc_ffn_scale, w.hc_ffn_base, bs, hc, d, iters, eps, stream);
    rmsnorm(x1, x1, w.ffn_norm, bs, d, eps, true, stream);
    moe_forward(sub, x1, input_ids, w.ffn, bs, stream);
    hc_post(out, sub, res2, post, comb, bs, hc, d, stream);

    CU(cudaStreamSynchronize(stream));
    cudaFree(x1); cudaFree(post); cudaFree(comb); cudaFree(sub); cudaFree(res2);
}
