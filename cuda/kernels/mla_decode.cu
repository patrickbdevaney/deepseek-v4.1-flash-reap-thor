// mla_decode.cu — M=1 KV-cache decode for a pure-sliding MLA layer. See mla_decode.h.
#include "mla_decode.h"
#include "dprof.h"
#include "fp8_block_gemm.h"
#include "mla_attn.h"
#include "kv_pack.h"
#include "deepseek_v4.h"
#include "dscratch.h"
#include <vector>
#include <cmath>
#include <cstdio>
#define CU(x) do{cudaError_t e=(x); if(e){fprintf(stderr,"cuda %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)
using namespace dsv4;

// Build the sliding-window key indices for a single query at position `pos`: [base .. pos], base=max(0,pos-W+1).
__global__ void k_win_idx(int* idx, int base, int width){
    int k = blockIdx.x*blockDim.x + threadIdx.x; if(k>=width) return; idx[k] = base + k;
}
// M=K window idxs: query i (global pos+i) attends [max(0,pos+i-W+1) .. pos+i]; pad -1 up to `topk`.
__global__ void k_verify_win_idx(int* idx, int pos, int K, int topk){
    int gid=blockIdx.x*blockDim.x+threadIdx.x; if(gid>=K*topk) return; int i=gid/topk, k=gid%topk;
    int ig=pos+i, base=ig-WINDOW+1; if(base<0) base=0; int v=base+k;
    idx[gid] = (v<=ig)? v : -1;
}

// ---- prefill: fill window-KV cache for x[0..s-1] (identical to mla_forward's internal kv) ----
void mla_cache_kv(float* kvcache, const float* x, const MLAWeights& w, int s, cudaStream_t stream){
    uint8_t* xq; float* xs;
    xq=(decltype(xq))dmalloc((size_t)s*DIM); xs=(decltype(xs))dmalloc((size_t)s*(DIM/128)*4);
    act_quant_fp8(xq, xs, x, s, DIM, 128, stream);
    float* kvst = kv_stage(kvcache, s);          // == kvcache when the cache is FP32 (1b.2)
    fp8_block_gemm(kvst, xq, xs, w.wkv, w.wkv_s, s, HEAD_DIM, DIM, stream);
    rmsnorm(kvst, kvst, w.kv_norm, s, HEAD_DIM, EPS, true, stream);
    rope_interleaved(kvst + NOPE_DIM, w.cosT, w.sinT, s, ROPE_DIM, false, HEAD_DIM, 1, stream);   // 1 cos row per token
    kv_commit(kvcache, kvst, s, stream);
    dsync(stream); kv_stage_free(kvst, kvcache); dfree(xq); dfree(xs);
}

// ---- decode step: one token at position `pos` ----
void mla_decode_step(float* out, const float* x, const MLAWeights& w, float* kvcache, int pos, cudaStream_t stream){
    const int half = ROPE_DIM/2, Kd = N_HEADS*HEAD_DIM, GKd = Kd/O_GROUPS, OB = O_GROUPS*O_LORA;
    const float scale = 1.f/sqrtf((float)HEAD_DIM);
    const float *cosP = w.cosT + (size_t)pos*half, *sinP = w.sinT + (size_t)pos*half;   // this position's RoPE row

    uint8_t *xq,*qrq,*ogq; float *xs,*qrs,*ogs,*qr,*q,*o,*og;
    xq=(decltype(xq))dmalloc(DIM); xs=(decltype(xs))dmalloc((DIM/128)*4);
    qr=(decltype(qr))dmalloc(Q_LORA*4); qrq=(decltype(qrq))dmalloc(Q_LORA); qrs=(decltype(qrs))dmalloc((Q_LORA/128)*4);
    q=(decltype(q))dmalloc(Kd*4); o=(decltype(o))dmalloc(Kd*4); og=(decltype(og))dmalloc(OB*4);
    ogq=(decltype(ogq))dmalloc(OB); ogs=(decltype(ogs))dmalloc((OB/128)*4);

    // 1. quantize x once (shared by wq_a and wkv)
    act_quant_fp8(xq, xs, x, 1, DIM, 128, stream);

    // 2. q = rope( per-head-rms( wq_b( q_norm( wq_a(x) ) ) ) )
    fp8_block_gemm(qr, xq, xs, w.wq_a, w.wq_a_s, 1, Q_LORA, DIM, stream);
    rmsnorm(qr, qr, w.q_norm, 1, Q_LORA, EPS, true, stream);
    act_quant_fp8(qrq, qrs, qr, 1, Q_LORA, 128, stream);
    fp8_block_gemm(q, qrq, qrs, w.wq_b, w.wq_b_s, 1, Kd, Q_LORA, stream);
    rmsnorm(q, q, nullptr, N_HEADS, HEAD_DIM, EPS, false, stream);                          // per-head, 1 token
    rope_interleaved(q + NOPE_DIM, cosP, sinP, N_HEADS, ROPE_DIM, false, HEAD_DIM, N_HEADS, stream);  // all heads share pos row

    // 3. new token's KV -> append to cache[pos]
    float* kvrow = kv_row(kvcache, pos);
    float* kvn = kv_stage(kvrow, 1);
    fp8_block_gemm(kvn, xq, xs, w.wkv, w.wkv_s, 1, HEAD_DIM, DIM, stream);
    rmsnorm(kvn, kvn, w.kv_norm, 1, HEAD_DIM, EPS, true, stream);
    rope_interleaved(kvn + NOPE_DIM, cosP, sinP, 1, ROPE_DIM, false, HEAD_DIM, 1, stream);
    kv_commit(kvrow, kvn, 1, stream); kv_stage_free(kvn, kvrow);

    // 4. sparse attention over the sliding window [base..pos]
    int base = pos - WINDOW + 1; if(base < 0) base = 0; int width = pos + 1 - base;
    int* didx; didx=(decltype(didx))dmalloc( width*4);
    k_win_idx<<<(width+63)/64,64,0,stream>>>(didx, base, width);
    sparse_attn_kv(o, q, kvcache, w.attn_sink, didx, 1, 1, N_HEADS, HEAD_DIM, pos+1, width, scale, stream);

    // 5. de-rotate o, grouped o-LoRA, wo_b
    rope_interleaved(o + NOPE_DIM, cosP, sinP, N_HEADS, ROPE_DIM, true, HEAD_DIM, N_HEADS, stream);
    if(w.wo_a_native) ogroup_gemm_fp8(og, o, w.wo_a_fp8, w.wo_a_sc, 1, O_GROUPS, O_LORA, GKd, stream);
    else              ogroup_gemm    (og, o, w.wo_a,                1, O_GROUPS, O_LORA, GKd, stream);
    act_quant_fp8(ogq, ogs, og, 1, OB, 128, stream);
    fp8_block_gemm(out, ogq, ogs, w.wo_b, w.wo_b_s, 1, DIM, OB, stream);

    dsync(stream);
    dfree(xq);dfree(xs);dfree(qr);dfree(qrq);dfree(qrs);dfree(q);
    dfree(o);dfree(og);dfree(ogq);dfree(ogs);dfree(didx);
}

// ---- M=K verify step: K tokens at [pos..pos+K-1], GEMMs at M=K (weights read ONCE) ----
void mla_verify_step(float* out, const float* x, const MLAWeights& w, float* kvcache, int pos, int K, cudaStream_t stream){
    const int half = ROPE_DIM/2, Kd = N_HEADS*HEAD_DIM, GKd = Kd/O_GROUPS, OB = O_GROUPS*O_LORA;
    const float scale = 1.f/sqrtf((float)HEAD_DIM);
    const float *cosP = w.cosT + (size_t)pos*half, *sinP = w.sinT + (size_t)pos*half;   // per-token rows from pos

    uint8_t *xq,*qrq,*ogq; float *xs,*qrs,*ogs,*qr,*q,*o,*og;
    xq=(uint8_t*)dmalloc((size_t)K*DIM); xs=(float*)dmalloc((size_t)K*(DIM/128)*4);
    qr=(float*)dmalloc((size_t)K*Q_LORA*4); qrq=(uint8_t*)dmalloc((size_t)K*Q_LORA); qrs=(float*)dmalloc((size_t)K*(Q_LORA/128)*4);
    q=(float*)dmalloc((size_t)K*Kd*4); o=(float*)dmalloc((size_t)K*Kd*4); og=(float*)dmalloc((size_t)K*OB*4);
    ogq=(uint8_t*)dmalloc((size_t)K*OB); ogs=(float*)dmalloc((size_t)K*(OB/128)*4);

    dprof_begin(DP_A_QPROJ,stream);
    act_quant_fp8(xq, xs, x, K, DIM, 128, stream);
    fp8_block_gemm(qr, xq, xs, w.wq_a, w.wq_a_s, K, Q_LORA, DIM, stream);
    rmsnorm(qr, qr, w.q_norm, K, Q_LORA, EPS, true, stream);
    act_quant_fp8(qrq, qrs, qr, K, Q_LORA, 128, stream);
    fp8_block_gemm(q, qrq, qrs, w.wq_b, w.wq_b_s, K, Kd, Q_LORA, stream);
    rmsnorm(q, q, nullptr, K*N_HEADS, HEAD_DIM, EPS, false, stream);
    rope_interleaved(q + NOPE_DIM, cosP, sinP, K*N_HEADS, ROPE_DIM, false, HEAD_DIM, N_HEADS, stream);  // row t*N_HEADS+h -> cos row t
    dprof_end(DP_A_QPROJ,stream);
    // new tokens' KV -> cache[pos..pos+K-1]
    dprof_begin(DP_A_KV,stream);
    float* kvrow = kv_row(kvcache, pos);
    float* kvn = kv_stage(kvrow, K);
    fp8_block_gemm(kvn, xq, xs, w.wkv, w.wkv_s, K, HEAD_DIM, DIM, stream);
    rmsnorm(kvn, kvn, w.kv_norm, K, HEAD_DIM, EPS, true, stream);
    rope_interleaved(kvn + NOPE_DIM, cosP, sinP, K, ROPE_DIM, false, HEAD_DIM, 1, stream);
    kv_commit(kvrow, kvn, K, stream); kv_stage_free(kvn, kvrow);
    dprof_end(DP_A_KV,stream);
    // per-query window idxs
    dprof_begin(DP_A_SPARSE,stream);
    int ncache = pos + K; int topk = (ncache < WINDOW) ? ncache : WINDOW;
    int* didx; didx=(int*)dmalloc((size_t)K*topk*4);
    k_verify_win_idx<<<((size_t)K*topk+63)/64,64,0,stream>>>(didx, pos, K, topk);
    sparse_attn_kv(o, q, kvcache, w.attn_sink, didx, 1, K, N_HEADS, HEAD_DIM, ncache, topk, scale, stream);
    dprof_end(DP_A_SPARSE,stream);
    dprof_begin(DP_A_OGROUP,stream);
    rope_interleaved(o + NOPE_DIM, cosP, sinP, K*N_HEADS, ROPE_DIM, true, HEAD_DIM, N_HEADS, stream);
    if(w.wo_a_native) ogroup_gemm_fp8(og, o, w.wo_a_fp8, w.wo_a_sc, K, O_GROUPS, O_LORA, GKd, stream);
    else              ogroup_gemm    (og, o, w.wo_a,                K, O_GROUPS, O_LORA, GKd, stream);
    act_quant_fp8(ogq, ogs, og, K, OB, 128, stream);
    fp8_block_gemm(out, ogq, ogs, w.wo_b, w.wo_b_s, K, DIM, OB, stream);
    dsync(stream);
    dfree(xq);dfree(xs);dfree(qr);dfree(qrq);dfree(qrs);dfree(q);dfree(o);dfree(og);dfree(ogq);dfree(ogs);dfree(didx);
    dprof_end(DP_A_OGROUP,stream);
}

// ================= device-pos sliding decode (CUDA-graph capturable) =================
__global__ void k_win_idx_dp(int* idx, const int* d_pos){
    int k=blockIdx.x*blockDim.x+threadIdx.x; if(k>=WINDOW) return; int pos=*d_pos, base=pos-WINDOW+1; if(base<0)base=0;
    int v=base+k; idx[k]=(v<=pos)? v : -1; }                       // fixed WINDOW slots (pad -1) -> static grid
__global__ void k_append_at(float* kvcache, const float* scr, const int* d_pos, int hd){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=hd) return; kvcache[(size_t)(*d_pos)*hd + i]=scr[i]; }
__global__ void k_incr(int* p){ if(threadIdx.x==0&&blockIdx.x==0) (*p)++; }
void dpos_incr(int* d_pos, cudaStream_t s){ k_incr<<<1,1,0,s>>>(d_pos); }

// M=1 decode fully driven by device *d_pos (no pos baked into args/pointers) -> capturable + replayable.
void mla_decode_step_dp(float* out, const float* x, const MLAWeights& w, float* kvcache, const int* d_pos, int nkv, cudaStream_t stream){
    const int Kd=N_HEADS*HEAD_DIM, GKd=Kd/O_GROUPS, OB=O_GROUPS*O_LORA; const float scale=1.f/sqrtf((float)HEAD_DIM);
    uint8_t *xq,*qrq,*ogq; float *xs,*qrs,*ogs,*qr,*q,*o,*og,*kvs; int* didx;
    xq=(uint8_t*)dmalloc(DIM); xs=(float*)dmalloc((DIM/128)*4);
    qr=(float*)dmalloc(Q_LORA*4); qrq=(uint8_t*)dmalloc(Q_LORA); qrs=(float*)dmalloc((Q_LORA/128)*4);
    q=(float*)dmalloc(Kd*4); o=(float*)dmalloc(Kd*4); og=(float*)dmalloc(OB*4);
    ogq=(uint8_t*)dmalloc(OB); ogs=(float*)dmalloc((OB/128)*4); kvs=(float*)dmalloc(HEAD_DIM*4); didx=(int*)dmalloc(WINDOW*4);
    act_quant_fp8(xq,xs,x,1,DIM,128,stream);
    fp8_block_gemm(qr,xq,xs,w.wq_a,w.wq_a_s,1,Q_LORA,DIM,stream); rmsnorm(qr,qr,w.q_norm,1,Q_LORA,EPS,true,stream);
    act_quant_fp8(qrq,qrs,qr,1,Q_LORA,128,stream); fp8_block_gemm(q,qrq,qrs,w.wq_b,w.wq_b_s,1,Kd,Q_LORA,stream);
    rmsnorm(q,q,nullptr,N_HEADS,HEAD_DIM,EPS,false,stream);
    rope_interleaved_dp(q+NOPE_DIM,w.cosT,w.sinT,N_HEADS,ROPE_DIM,false,HEAD_DIM,N_HEADS,d_pos,stream);
    // new KV -> scratch -> append at *d_pos
    fp8_block_gemm(kvs,xq,xs,w.wkv,w.wkv_s,1,HEAD_DIM,DIM,stream); rmsnorm(kvs,kvs,w.kv_norm,1,HEAD_DIM,EPS,true,stream);
    rope_interleaved_dp(kvs+NOPE_DIM,w.cosT,w.sinT,1,ROPE_DIM,false,HEAD_DIM,1,d_pos,stream);
    // 1b.2: on the packed path the quantise and the append are ONE kernel -- kv_store_at computes
    // the identical amax/scale/e4m3 code act_quant_fp8sim would have and stores the code, so the
    // separate act_quant would be a no-op re-quantisation of its own output.
    if (g_kv_pack) kv_store_at(kvcache, kvs, d_pos, stream);
    else { act_quant_fp8sim(kvs,1,NOPE_DIM,64,HEAD_DIM,stream);
           k_append_at<<<(HEAD_DIM+255)/256,256,0,stream>>>(kvcache,kvs,d_pos,HEAD_DIM); }
    k_win_idx_dp<<<(WINDOW+63)/64,64,0,stream>>>(didx,d_pos);
    sparse_attn_kv(o,q,kvcache,w.attn_sink,didx,1,1,N_HEADS,HEAD_DIM,nkv,WINDOW,scale,stream);
    rope_interleaved_dp(o+NOPE_DIM,w.cosT,w.sinT,N_HEADS,ROPE_DIM,true,HEAD_DIM,N_HEADS,d_pos,stream);
    if(w.wo_a_native) ogroup_gemm_fp8(og,o,w.wo_a_fp8,w.wo_a_sc,1,O_GROUPS,O_LORA,GKd,stream);
    else              ogroup_gemm    (og,o,w.wo_a,               1,O_GROUPS,O_LORA,GKd,stream);
    act_quant_fp8(ogq,ogs,og,1,OB,128,stream); fp8_block_gemm(out,ogq,ogs,w.wo_b,w.wo_b_s,1,DIM,OB,stream);
    dsync(stream);
    dfree(xq);dfree(xs);dfree(qr);dfree(qrq);dfree(qrs);dfree(q);dfree(o);dfree(og);dfree(ogq);dfree(ogs);dfree(kvs);dfree(didx);
}
