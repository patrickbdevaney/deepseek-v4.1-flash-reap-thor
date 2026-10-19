// block_decode.cu — full-Block M=1 KV-cache decode wrappers. See block_decode.h.
// Prefill-cache runs the normal block for the output AND populates the per-layer KV cache from the same
// attention-input x1 (redundant recompute, one-time, small s). Decode-step is the block at bs=1 with the
// attention swapped to the gated decode step. HC scratch is alloc'd per call for now (Step 2 pre-allocates).
#include "block_decode.h"
#include "compressor.h"
#include <cstdio>
#include <cstdlib>
#include "dprof.h"
#include "hc.h"
#include "mla_attn.h"        // rmsnorm
#include "mla_decode.h"
#include "compressed_decode.h"
#include "moe.h"
#include "deepseek_v4.h"
#include "dscratch.h"
#include <cstdio>
#define CU(x) do{cudaError_t e=(x); if(e){fprintf(stderr,"cuda %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)
using namespace dsv4;

// ================= sliding (ratio==0) =================
void block_prefill_cache(float* out, const float* x, const int* input_ids, const BlockWeights& w,
                         int s, int iters, float eps, LayerKV& kv, cudaStream_t stream){
    const int bs=s, d=w.dim, hc=w.hc;
    float *x1,*post,*comb,*sub,*res2;
    x1=(decltype(x1))dmalloc((size_t)bs*d*4); post=(decltype(post))dmalloc((size_t)bs*hc*4); comb=(decltype(comb))dmalloc((size_t)bs*hc*hc*4);
    sub=(decltype(sub))dmalloc((size_t)bs*d*4); res2=(decltype(res2))dmalloc((size_t)bs*hc*d*4);
    // B9 marks, same ids as the compressed path — without these the 2 pure-sliding layers drop out
    // of the prefill TOTAL and the report understates itself by exactly the control group.
    dprof_begin(DP_HC_PRE_ATTN,stream);  hc_pre(x1,post,comb,x,w.hc_attn_fn,w.hc_attn_scale,w.hc_attn_base,bs,hc,d,iters,eps,stream); dprof_end(DP_HC_PRE_ATTN,stream);
    dprof_begin(DP_RMSNORM_ATTN,stream); rmsnorm(x1,x1,w.attn_norm,bs,d,eps,true,stream);                                             dprof_end(DP_RMSNORM_ATTN,stream);
    dprof_begin(DP_A_KV,stream);         mla_cache_kv(kv.win_kv, x1, w.attn, s, stream);                                              dprof_end(DP_A_KV,stream);
    dprof_begin(DP_ATTN,stream);         mla_forward(sub, x1, w.attn, 1, s, stream);                                                  dprof_end(DP_ATTN,stream);
    dprof_begin(DP_HC_POST_ATTN,stream); hc_post(res2,sub,x,post,comb,bs,hc,d,stream);                                                dprof_end(DP_HC_POST_ATTN,stream);
    dprof_begin(DP_HC_PRE_FFN,stream);   hc_pre(x1,post,comb,res2,w.hc_ffn_fn,w.hc_ffn_scale,w.hc_ffn_base,bs,hc,d,iters,eps,stream); dprof_end(DP_HC_PRE_FFN,stream);
    dprof_begin(DP_RMSNORM_FFN,stream);  rmsnorm(x1,x1,w.ffn_norm,bs,d,eps,true,stream);                                              dprof_end(DP_RMSNORM_FFN,stream);
    dprof_begin(DP_MOE,stream);          moe_forward(sub,x1,input_ids,w.ffn,bs,stream);                                               dprof_end(DP_MOE,stream);
    dprof_begin(DP_HC_POST_FFN,stream);  hc_post(out,sub,res2,post,comb,bs,hc,d,stream);                                              dprof_end(DP_HC_POST_FFN,stream);
    dsync(stream);
    dfree(x1);dfree(post);dfree(comb);dfree(sub);dfree(res2);
}
void block_decode_step(float* out, const float* x, const int* input_ids, const BlockWeights& w,
                       int pos, int iters, float eps, LayerKV& kv, cudaStream_t stream){
    const int d=w.dim, hc=w.hc;
    float *x1,*post,*comb,*sub,*res2;
    x1=(decltype(x1))dmalloc((size_t)d*4); post=(decltype(post))dmalloc((size_t)hc*4); comb=(decltype(comb))dmalloc((size_t)hc*hc*4);
    sub=(decltype(sub))dmalloc((size_t)d*4); res2=(decltype(res2))dmalloc((size_t)hc*d*4);
    hc_pre(x1,post,comb,x,w.hc_attn_fn,w.hc_attn_scale,w.hc_attn_base,1,hc,d,iters,eps,stream);
    rmsnorm(x1,x1,w.attn_norm,1,d,eps,true,stream);
    mla_decode_step(sub, x1, w.attn, kv.win_kv, pos, stream);
    hc_post(res2,sub,x,post,comb,1,hc,d,stream);
    hc_pre(x1,post,comb,res2,w.hc_ffn_fn,w.hc_ffn_scale,w.hc_ffn_base,1,hc,d,iters,eps,stream);
    rmsnorm(x1,x1,w.ffn_norm,1,d,eps,true,stream);
    moe_forward(sub,x1,input_ids,w.ffn,1,stream);
    hc_post(out,sub,res2,post,comb,1,hc,d,stream);
    dsync(stream);
    dfree(x1);dfree(post);dfree(comb);dfree(sub);dfree(res2);
}

// ================= compressed (ratio 4 / 128) =================
// Copy the s attention-input rows x1 into the layer's xin history (for future compressor group emits).
__global__ void k_copy(float* dst, const float* src, size_t n){ size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x; if(i<n) dst[i]=src[i]; }

// XIN RING WRITE.
//
// `xin` is the attention-input history, and after the x_cur/x_full split the ONLY reader of it is
// compressor_emit_group, which never looks further back than 2*ratio positions (compressor.cu:500).
// So it does not have to be [seqmax, DIM] -- at fp32 x 4096 x 41 compressed layers that was
// 656 KiB per token of context, 6.6x the entire MLA+DSA KV cache and the largest remaining thing
// that scaled with context.
//
// The ring is R = 2*ratio rows with the first `ratio` rows MIRRORED at [R, R+ratio). A group window
// starts at a multiple of ratio and is at most 2*ratio long, so modulo R it lands at either 0 or
// ratio; the mirror makes the second case contiguous instead of wrapping, which keeps the reader a
// plain pointer and leaves the GEMM untouched. The duplicate write is one extra row for half the
// positions -- against a buffer that shrank from seqmax rows to 3*ratio.
static inline void xin_ring_check(int n, int R, int ratio){
    if (R && n > R - 2*ratio) {
        fprintf(stderr, "[xin_ring] batch %d exceeds ring %d - 2*%d; history would be clobbered\n",
                n, R, ratio);
        abort();
    }
}
__global__ void k_copy_xin_ring(float* xin, const float* src, int pos, int n, int d, int R, int mir){
    size_t i = blockIdx.x*(size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)n*d) return;
    const int j = (int)(i / d), c = (int)(i % d);
    const int r = (pos + j) % R;
    const float v = src[i];
    xin[(size_t)r*d + c] = v;
    if (r < mir) xin[(size_t)(r + R)*d + c] = v;   // mirrored margin
}

// Ring geometry lives in include/compressor.h -- shared with the allocator in engine.cu.
static inline void set_xin_ring(int ratio){ g_xin_ring = xin_ring_rows(ratio); }

void cblock_prefill_cache(float* out, const float* x, const int* input_ids, const CompressedBlockWeights& w,
                          int s, int iters, float eps, LayerKV& kv, cudaStream_t stream){
    const int bs=s, d=w.dim, hc=w.hc;
    float *x1,*post,*comb,*sub,*res2;
    x1=(decltype(x1))dmalloc((size_t)bs*d*4); post=(decltype(post))dmalloc((size_t)bs*hc*4); comb=(decltype(comb))dmalloc((size_t)bs*hc*hc*4);
    sub=(decltype(sub))dmalloc((size_t)bs*d*4); res2=(decltype(res2))dmalloc((size_t)bs*hc*d*4);
    // B9. This path carried NO dprof marks, so the first prefill profile attributed only the MoE
    // (whose marks live inside moe_forward) and left 57.4% of a 21.4 s prefill unaccounted — the
    // report caught itself with "moe children > MoE 0.00 ms" only because of its parent/child check.
    // Same ids as block_verify_step so the prefill and verify tables are directly comparable.
    dprof_begin(DP_HC_PRE_ATTN,stream);  hc_pre(x1,post,comb,x,w.hc_attn_fn,w.hc_attn_scale,w.hc_attn_base,bs,hc,d,iters,eps,stream); dprof_end(DP_HC_PRE_ATTN,stream);
    dprof_begin(DP_RMSNORM_ATTN,stream); rmsnorm(x1,x1,w.attn_norm,bs,d,eps,true,stream);                                             dprof_end(DP_RMSNORM_ATTN,stream);
    // retain attention-input history + populate KV caches from x1
    dprof_begin(DP_KV_XIN,stream);
    set_xin_ring(w.ratio);
    xin_ring_check(s, g_xin_ring, w.ratio);
    if (g_xin_ring) k_copy_xin_ring<<<((size_t)s*d+255)/256,256,0,stream>>>(kv.xin, x1, 0, s, d, g_xin_ring, w.ratio);
    else            k_copy<<<((size_t)s*d+255)/256,256,0,stream>>>(kv.xin, x1, (size_t)s*d);
    dprof_end(DP_KV_XIN,stream);
    // Cache population has no decode analogue (decode appends one position); DP_C_COMPRESS is the
    // honest id for it — this IS the compressor, run over the whole prompt.
    dprof_begin(DP_C_COMPRESS,stream);
    if(w.ratio==4) compressed_attn_cache_r4(kv.win_kv, kv.comp_kv, kv.idx_ckv, &kv.T, x1, w.attn, s, w.ratio, eps, stream);
    else           compressed_attn_cache   (kv.win_kv, kv.comp_kv,             &kv.T, x1, w.attn, s, w.ratio, eps, stream);
    dprof_end(DP_C_COMPRESS,stream);
    dprof_begin(DP_ATTN,stream);         compressed_attn_forward(sub, x1, w.attn, s, w.win, w.ratio, eps, stream);                    dprof_end(DP_ATTN,stream);
    dprof_begin(DP_HC_POST_ATTN,stream); hc_post(res2,sub,x,post,comb,bs,hc,d,stream);                                                dprof_end(DP_HC_POST_ATTN,stream);
    dprof_begin(DP_HC_PRE_FFN,stream);   hc_pre(x1,post,comb,res2,w.hc_ffn_fn,w.hc_ffn_scale,w.hc_ffn_base,bs,hc,d,iters,eps,stream); dprof_end(DP_HC_PRE_FFN,stream);
    dprof_begin(DP_RMSNORM_FFN,stream);  rmsnorm(x1,x1,w.ffn_norm,bs,d,eps,true,stream);                                              dprof_end(DP_RMSNORM_FFN,stream);
    dprof_begin(DP_MOE,stream);          moe_forward(sub,x1,input_ids,w.ffn,bs,stream);                                               dprof_end(DP_MOE,stream);
    dprof_begin(DP_HC_POST_FFN,stream);  hc_post(out,sub,res2,post,comb,bs,hc,d,stream);                                              dprof_end(DP_HC_POST_FFN,stream);
    dsync(stream);
    dfree(x1);dfree(post);dfree(comb);dfree(sub);dfree(res2);
}
void cblock_decode_step(float* out, const float* x, const int* input_ids, const CompressedBlockWeights& w,
                        int pos, int iters, float eps, LayerKV& kv, cudaStream_t stream){
    const int d=w.dim, hc=w.hc;
    float *x1,*post,*comb,*sub,*res2;
    x1=(decltype(x1))dmalloc((size_t)d*4); post=(decltype(post))dmalloc((size_t)hc*4); comb=(decltype(comb))dmalloc((size_t)hc*hc*4);
    sub=(decltype(sub))dmalloc((size_t)d*4); res2=(decltype(res2))dmalloc((size_t)hc*d*4);
    hc_pre(x1,post,comb,x,w.hc_attn_fn,w.hc_attn_scale,w.hc_attn_base,1,hc,d,iters,eps,stream);
    rmsnorm(x1,x1,w.attn_norm,1,d,eps,true,stream);
    set_xin_ring(w.ratio);   // store this position's attn input (ring-mapped when enabled)
    if (g_xin_ring) k_copy_xin_ring<<<((size_t)d+255)/256,256,0,stream>>>(kv.xin, x1, pos, 1, d, g_xin_ring, w.ratio);
    else            k_copy<<<((size_t)d+255)/256,256,0,stream>>>(kv.xin + (size_t)pos*d, x1, (size_t)d);
    if(w.ratio==4) compressed_decode_step_indexer(sub, x1, kv.xin, pos, w.attn, kv.win_kv, kv.comp_kv, kv.idx_ckv, &kv.T, w.ratio, eps, stream);
    else           compressed_decode_step_strided(sub, x1, kv.xin, pos, w.attn, kv.win_kv, kv.comp_kv,             &kv.T, w.ratio, eps, stream);
    hc_post(res2,sub,x,post,comb,1,hc,d,stream);
    hc_pre(x1,post,comb,res2,w.hc_ffn_fn,w.hc_ffn_scale,w.hc_ffn_base,1,hc,d,iters,eps,stream);
    rmsnorm(x1,x1,w.ffn_norm,1,d,eps,true,stream);
    moe_forward(sub,x1,input_ids,w.ffn,1,stream);
    hc_post(out,sub,res2,post,comb,1,hc,d,stream);
    dsync(stream);
    dfree(x1);dfree(post);dfree(comb);dfree(sub);dfree(res2);
}

// ================= M=K VERIFY block steps (spec-decode) =================
void block_verify_step(float* out, const float* x, const int* input_ids, const BlockWeights& w,
                       int pos, int K, int iters, float eps, LayerKV& kv, cudaStream_t stream){
    const int d=w.dim, hc=w.hc;
    float *x1,*post,*comb,*sub,*res2;
    x1=(float*)dmalloc((size_t)K*d*4); post=(float*)dmalloc((size_t)K*hc*4); comb=(float*)dmalloc((size_t)K*hc*hc*4);
    sub=(float*)dmalloc((size_t)K*d*4); res2=(float*)dmalloc((size_t)K*hc*d*4);
    dprof_begin(DP_HC_PRE_ATTN,stream);  hc_pre(x1,post,comb,x,w.hc_attn_fn,w.hc_attn_scale,w.hc_attn_base,K,hc,d,iters,eps,stream);  dprof_end(DP_HC_PRE_ATTN,stream);
    dprof_begin(DP_RMSNORM_ATTN,stream); rmsnorm(x1,x1,w.attn_norm,K,d,eps,true,stream);                                              dprof_end(DP_RMSNORM_ATTN,stream);
    dprof_begin(DP_ATTN,stream);         mla_verify_step(sub, x1, w.attn, kv.win_kv, pos, K, stream);                                  dprof_end(DP_ATTN,stream);
    dprof_begin(DP_HC_POST_ATTN,stream); hc_post(res2,sub,x,post,comb,K,hc,d,stream);                                                  dprof_end(DP_HC_POST_ATTN,stream);
    dprof_begin(DP_HC_PRE_FFN,stream);   hc_pre(x1,post,comb,res2,w.hc_ffn_fn,w.hc_ffn_scale,w.hc_ffn_base,K,hc,d,iters,eps,stream);   dprof_end(DP_HC_PRE_FFN,stream);
    dprof_begin(DP_RMSNORM_FFN,stream);  rmsnorm(x1,x1,w.ffn_norm,K,d,eps,true,stream);                                                dprof_end(DP_RMSNORM_FFN,stream);
    dprof_begin(DP_MOE,stream);          moe_forward(sub,x1,input_ids,w.ffn,K,stream);                                                 dprof_end(DP_MOE,stream);
    dprof_begin(DP_HC_POST_FFN,stream);  hc_post(out,sub,res2,post,comb,K,hc,d,stream);                                                dprof_end(DP_HC_POST_FFN,stream);
    dsync(stream); dfree(x1);dfree(post);dfree(comb);dfree(sub);dfree(res2);
}
void cblock_verify_step(float* out, const float* x, const int* input_ids, const CompressedBlockWeights& w,
                        int pos, int K, int iters, float eps, LayerKV& kv, cudaStream_t stream){
    const int d=w.dim, hc=w.hc;
    float *x1,*post,*comb,*sub,*res2;
    x1=(float*)dmalloc((size_t)K*d*4); post=(float*)dmalloc((size_t)K*hc*4); comb=(float*)dmalloc((size_t)K*hc*hc*4);
    sub=(float*)dmalloc((size_t)K*d*4); res2=(float*)dmalloc((size_t)K*hc*d*4);
    dprof_begin(DP_HC_PRE_ATTN,stream);  hc_pre(x1,post,comb,x,w.hc_attn_fn,w.hc_attn_scale,w.hc_attn_base,K,hc,d,iters,eps,stream);  dprof_end(DP_HC_PRE_ATTN,stream);
    dprof_begin(DP_RMSNORM_ATTN,stream); rmsnorm(x1,x1,w.attn_norm,K,d,eps,true,stream);                                              dprof_end(DP_RMSNORM_ATTN,stream);
    dprof_begin(DP_KV_XIN,stream);
    set_xin_ring(w.ratio);   // store attn-input history (ring-mapped when enabled)
    xin_ring_check(K, g_xin_ring, w.ratio);
    if (g_xin_ring) k_copy_xin_ring<<<((size_t)K*d+255)/256,256,0,stream>>>(kv.xin, x1, pos, K, d, g_xin_ring, w.ratio);
    else            k_copy<<<((size_t)K*d+255)/256,256,0,stream>>>(kv.xin+(size_t)pos*d, x1, (size_t)K*d);
    dprof_end(DP_KV_XIN,stream);
    dprof_begin(DP_ATTN,stream);
    if(w.ratio==4) compressed_verify_step_indexer(sub, x1, kv.xin, pos, K, w.attn, kv.win_kv, kv.comp_kv, kv.idx_ckv, &kv.T, w.ratio, eps, stream);
    else           compressed_verify_step_strided(sub, x1, kv.xin, pos, K, w.attn, kv.win_kv, kv.comp_kv,             &kv.T, w.ratio, eps, stream);
    dprof_end(DP_ATTN,stream);
    dprof_begin(DP_HC_POST_ATTN,stream); hc_post(res2,sub,x,post,comb,K,hc,d,stream);                                                  dprof_end(DP_HC_POST_ATTN,stream);
    dprof_begin(DP_HC_PRE_FFN,stream);   hc_pre(x1,post,comb,res2,w.hc_ffn_fn,w.hc_ffn_scale,w.hc_ffn_base,K,hc,d,iters,eps,stream);   dprof_end(DP_HC_PRE_FFN,stream);
    dprof_begin(DP_RMSNORM_FFN,stream);  rmsnorm(x1,x1,w.ffn_norm,K,d,eps,true,stream);                                                dprof_end(DP_RMSNORM_FFN,stream);
    dprof_begin(DP_MOE,stream);          moe_forward(sub,x1,input_ids,w.ffn,K,stream);                                                 dprof_end(DP_MOE,stream);
    dprof_begin(DP_HC_POST_FFN,stream);  hc_post(out,sub,res2,post,comb,K,hc,d,stream);                                                dprof_end(DP_HC_POST_FFN,stream);
    dsync(stream); dfree(x1);dfree(post);dfree(comb);dfree(sub);dfree(res2);
}

// ================= device-pos block steps (CUDA-graph capturable) =================
extern __global__ void k_append_at2(float* dst, const float* scr, const int* d_idx, int hd);   // from compressed_decode.cu
void mla_decode_step_dp(float* out, const float* x, const MLAWeights& w, float* kvcache, const int* d_pos, int nkv, cudaStream_t stream);
void compressed_decode_step_strided_dp(float* out, const float* x, const float* xin, const CompressedAttnWeights& w, float* kvc, const int* d_pos, int* d_T, int* d_g, int winmax, int Tmax, int ratio, float eps, cudaStream_t stream);
void compressed_decode_step_indexer_dp(float* out, const float* x, const float* xin, const CompressedAttnWeights& w, float* kvc, float* idx_kvc, const int* d_pos, int* d_T, int* d_g, int winmax, int Tmax, int ratio, float eps, cudaStream_t stream);
void block_decode_step_dp(float* out, const float* x, const int* d_curid, const BlockWeights& w,
                          const int* d_pos, int nkv, int iters, float eps, LayerKV& kv, cudaStream_t stream){
    const int d=w.dim, hc=w.hc; float *x1,*post,*comb,*sub,*res2;
    x1=(float*)dmalloc((size_t)d*4); post=(float*)dmalloc((size_t)hc*4); comb=(float*)dmalloc((size_t)hc*hc*4); sub=(float*)dmalloc((size_t)d*4); res2=(float*)dmalloc((size_t)hc*d*4);
    hc_pre(x1,post,comb,x,w.hc_attn_fn,w.hc_attn_scale,w.hc_attn_base,1,hc,d,iters,eps,stream); rmsnorm(x1,x1,w.attn_norm,1,d,eps,true,stream);
    mla_decode_step_dp(sub,x1,w.attn,kv.win_kv,d_pos,nkv,stream);
    hc_post(res2,sub,x,post,comb,1,hc,d,stream);
    hc_pre(x1,post,comb,res2,w.hc_ffn_fn,w.hc_ffn_scale,w.hc_ffn_base,1,hc,d,iters,eps,stream); rmsnorm(x1,x1,w.ffn_norm,1,d,eps,true,stream);
    moe_forward(sub,x1,d_curid,w.ffn,1,stream);
    hc_post(out,sub,res2,post,comb,1,hc,d,stream);
    dsync(stream); dfree(x1);dfree(post);dfree(comb);dfree(sub);dfree(res2);
}
void cblock_decode_step_dp(float* out, const float* x, const int* d_curid, const CompressedBlockWeights& w,
                           const int* d_pos, int* d_g, int winmax, int Tmax, int iters, float eps, LayerKV& kv, cudaStream_t stream){
    const int d=w.dim, hc=w.hc; float *x1,*post,*comb,*sub,*res2;
    x1=(float*)dmalloc((size_t)d*4); post=(float*)dmalloc((size_t)hc*4); comb=(float*)dmalloc((size_t)hc*hc*4); sub=(float*)dmalloc((size_t)d*4); res2=(float*)dmalloc((size_t)hc*d*4);
    hc_pre(x1,post,comb,x,w.hc_attn_fn,w.hc_attn_scale,w.hc_attn_base,1,hc,d,iters,eps,stream); rmsnorm(x1,x1,w.attn_norm,1,d,eps,true,stream);
    k_append_at2<<<((size_t)d+255)/256,256,0,stream>>>(kv.xin,x1,d_pos,d);          // xin[*d_pos]=x1
    if(w.ratio==4) compressed_decode_step_indexer_dp(sub,x1,kv.xin,w.attn,kv.kvc,kv.idx_kvc,d_pos,kv.d_T,d_g,winmax,Tmax,w.ratio,eps,stream);
    else           compressed_decode_step_strided_dp(sub,x1,kv.xin,w.attn,kv.kvc,           d_pos,kv.d_T,d_g,winmax,Tmax,w.ratio,eps,stream);
    hc_post(res2,sub,x,post,comb,1,hc,d,stream);
    hc_pre(x1,post,comb,res2,w.hc_ffn_fn,w.hc_ffn_scale,w.hc_ffn_base,1,hc,d,iters,eps,stream); rmsnorm(x1,x1,w.ffn_norm,1,d,eps,true,stream);
    moe_forward(sub,x1,d_curid,w.ffn,1,stream); hc_post(out,sub,res2,post,comb,1,hc,d,stream);
    dsync(stream); dfree(x1);dfree(post);dfree(comb);dfree(sub);dfree(res2);
}
