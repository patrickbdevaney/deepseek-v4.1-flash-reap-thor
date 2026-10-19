// compressed_decode.cu — M=1 KV-cache decode for a compressed (strided, ratio!=4) MLA layer. See header.
// COMBINED KV CACHE.
//
// Attention here is sparse_attn, whose kernel loops `for (t = 0; t < topk; ++t)` -- it is O(topk),
// about 640 rows, and NOT O(context). `n` is only the row stride. Yet every decode step used to
// dmalloc a (pos + T) x HEAD_DIM buffer and memcpy win_kv and comp_kv into it, per layer, purely to
// make those 640 rows contiguous: 1.76 GB copied per token at 16K context, for an attention that
// reads 1.3 MB of it. That is why decode degraded with context on an architecture (MLA + DSA) whose
// whole point is that it should not.
//
// When the caller allocates win_kv and comp_kv as ONE buffer -- comp_kv == win_kv + winmax*HEAD_DIM
// -- it sets g_kv_winmax to that stride and the copy disappears: kv_all is win_kv, and the
// compressed rows are already where the index arrays say they are. LayerKV documented this layout
// from the start ("so attention needs no per-step copy"); the graph path used it and the serving
// path never did.
//
// 0 means the buffers are separate and the old copy path runs, which is what src/decode.cu and the
// unit gates rely on. Set by whoever owns the allocation.
int g_kv_winmax = 0;

#include "compressed_decode.h"
#include "fp8_block_gemm.h"
#include "mla_attn.h"
#include "compressor.h"
#include "indexer.h"
#include "topk_radix.h"
#include "deepseek_v4.h"
#include "kv_pack.h"
#include "dscratch.h"
#include "dprof.h"
#include <vector>
#include <cmath>
#include <cstdio>

// Pinned staging for the host-built `comb` index arrays. `cudaMemcpyAsync` from PAGEABLE host
// memory is not stream-capturable, and it was the only thing in the verify path that was not — the
// indexer variant already builds its comb on device (`k_comb_verify`). One grow-only pinned buffer,
// so a graph capture of the verify sees a capturable copy. (LOOP_LOG Finding 48.)
static int* comb_pinned(size_t n){
    static int* buf = nullptr; static size_t cap = 0;
    if(n > cap){ if(buf) cudaFreeHost(buf); cudaHostAlloc((void**)&buf, n*4, cudaHostAllocDefault); cap = n; }
    return buf;
}
#define CU(x) do{cudaError_t e=(x); if(e){fprintf(stderr,"cuda %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)
using namespace dsv4;

// single-query descending top-k over score[0..T-1]; out[k] = (k-th best t) + offset (or -1 if none left).
// Decode-time: all T cached rows are already causal-valid (t < (pos+1)/ratio == T), so no mask needed here.
__global__ void k_topk_decode(int* out, const float* score, int T, int topk, int offset){
    if(blockIdx.x) return;
    extern __shared__ float sh[]; const int L=threadIdx.x;
    for(int t=L;t<T;t+=32) sh[t]=score[t];
    __syncwarp();
    for(int k=0;k<topk;++k){ float best=-1e30f; int bi=T;
        for(int t=L;t<T;t+=32) if(sh[t]>best){best=sh[t];bi=t;}
        warp_argmax(best,bi);
        if(L==0){ if(bi<T) sh[bi]=-1e30f; out[k] = (bi>=T)? -1 : bi+offset; }
        __syncwarp(); }
}
// Radix-select twin (item 1.2). Same admission rule, same order, no dynamic shared memory.
// `sel` is written in place: topk_radix_select leaves raw source indices (or -1) in `out`, then the
// caller's own offset rule is applied to them -- which is exactly what the `if(L==0)` line did.
__global__ void k_topk_decode_rx(int* out, const float* score, int T, int topk, int offset, bool early){
    if(blockIdx.x) return;
    __shared__ TopkRadixSmem S;
    topk_radix_select<TOPK_RADIX_NT>(out, score, T, topk, -1e30f, S, early);
    for(int k=threadIdx.x;k<topk;k+=TOPK_RADIX_NT){ int b=out[k]; out[k] = (b<0)? -1 : b+offset; }
}
__global__ void k_iw_scale(float* y, float sc, int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) y[i]*=sc; }
// device verify top-k (per query, global causal threshold) + combined-idx build -> removes the host D2H sync.
__global__ void k_topk_verify(int* dtop, const float* score, int K, int Tf, int topkc, int pos, int ratio, int nwin){
    int i=blockIdx.x; if(i>=K) return; extern __shared__ float sh[]; const int L=threadIdx.x;
    const float* s=score+(size_t)i*Tf;
    for(int t=L;t<Tf;t+=32) sh[t]=s[t];
    __syncwarp();
    const int thr=(pos+i+1)/ratio; const int lim = thr<Tf? thr : Tf;
    for(int k=0;k<topkc;++k){ float best=-1e30f; int bi=Tf;
        for(int t=L;t<lim;t+=32) if(sh[t]>best){best=sh[t];bi=t;}
        warp_argmax(best,bi);
        if(L==0){ if(bi<Tf){ sh[bi]=-1e30f; dtop[(size_t)i*topkc+k]=nwin+bi; } else dtop[(size_t)i*topkc+k]=-1; }
        __syncwarp(); } }
__global__ void k_topk_verify_rx(int* dtop, const float* score, int K, int Tf, int topkc, int pos, int ratio, int nwin, bool early){
    int i=blockIdx.x; if(i>=K) return;
    __shared__ TopkRadixSmem S;
    const int thr=(pos+i+1)/ratio; const int lim = thr<Tf? thr : Tf;
    int* out = dtop + (size_t)i*topkc;
    topk_radix_select<TOPK_RADIX_NT>(out, score+(size_t)i*Tf, lim, topkc, -1e30f, S, early);
    for(int k=threadIdx.x;k<topkc;k+=TOPK_RADIX_NT){ int b=out[k]; out[k] = (b<0)? -1 : nwin+b; }
}
__global__ void k_comb_verify(int* dcomb, const int* dtop, int K, int topk, int wmax, int topkc, int pos){
    int gid=blockIdx.x*blockDim.x+threadIdx.x; if(gid>=K*topk) return; int i=gid/topk, k=gid%topk;
    int ig=pos+i, base=ig-WINDOW+1; if(base<0)base=0; int wid=ig+1-base;
    if(k<wmax) dcomb[gid]=(k<wid)? base+k : -1; else dcomb[gid]=dtop[(size_t)i*topkc+(k-wmax)]; }


// DSV4_TOPK_GATE=1 — the bit-exactness gate for item 1.2, run IN SITU on the shipped path at real
// context. DECODE_LADDER hard invariant 1 was amended by 1.0: token ids are not a valid test at
// context, because the engine stops reproducing itself part-way through a long run. The substitute
// is a memcmp of the changed kernel's ENTIRE OUTPUT BUFFER against the untouched reference, on
// every call, which is strictly stronger -- it proves the whole index array identical rather than
// one downstream consumer's tokens.
//
// The reference is the shipped warp selection sort, launched with the identical arguments into a
// private buffer. It aborts on the first differing slot rather than counting: a bit-exactness gate
// that reports and continues is a gate that gets ignored.
//
// NOT armed on the `_dp` graph path (`k_topk_masked`), which is CUDA-graph captured and cannot
// contain a host sync; that kernel is covered by tests/gate_topk_radix.cu only.
static int topk_gate_on(){
    static const int g = getenv("DSV4_TOPK_GATE") ? atoi(getenv("DSV4_TOPK_GATE")) : 0;
    return g;
}
static long long g_topk_checks = 0, g_topk_slots = 0;
static void topk_gate_report(const char* what, int n, int a0, int b0, int i, int ctx){
    if(i >= 0){
        fprintf(stderr, "[topk-gate] FAIL %s ctx=%d n=%d first diff at slot %d: radix %d vs warp %d\n",
                what, ctx, n, i, a0, b0);
        fflush(stderr); abort();
    }
    if(g_topk_checks <= 3 || g_topk_checks % 64 == 0){
        fprintf(stderr, "[topk-gate] PASS %s ctx=%d n=%d (%lld checks, %lld slots)\n",
                what, ctx, n, g_topk_checks, g_topk_slots); fflush(stderr);
    }
}
static void topk_gate_verify(const int* got, const float* score, int K, int Tf, int topkc,
                             int pos, int ratio, int nwin, cudaStream_t stream){
    size_t n = (size_t)K*topkc;
    int* ref = nullptr; CU(cudaMalloc(&ref, n*4));
    k_topk_verify<<<K,32,TOPK_SMEM(k_topk_verify,Tf),stream>>>(ref, score, K, Tf, topkc, pos, ratio, nwin);
    TOPK_LAUNCHED(k_topk_verify, Tf);
    CU(cudaStreamSynchronize(stream));
    std::vector<int> a(n), b(n);
    CU(cudaMemcpy(a.data(), got, n*4, cudaMemcpyDeviceToHost));
    CU(cudaMemcpy(b.data(), ref, n*4, cudaMemcpyDeviceToHost));
    cudaFree(ref);
    ++g_topk_checks; g_topk_slots += (long long)n;
    int bad = -1; for(size_t i=0;i<n;++i) if(a[i]!=b[i]){ bad=(int)i; break; }
    topk_gate_report("verify", (int)n, bad<0?0:a[bad], bad<0?0:b[bad], bad, pos);
}
static void topk_gate_decode(const int* got, const float* score, int T, int topk, int offset,
                             int pos, cudaStream_t stream){
    size_t n = (size_t)topk;
    int* ref = nullptr; CU(cudaMalloc(&ref, n*4));
    k_topk_decode<<<1,32,TOPK_SMEM(k_topk_decode,T),stream>>>(ref, score, T, topk, offset);
    TOPK_LAUNCHED(k_topk_decode, T);
    CU(cudaStreamSynchronize(stream));
    std::vector<int> a(n), b(n);
    CU(cudaMemcpy(a.data(), got, n*4, cudaMemcpyDeviceToHost));
    CU(cudaMemcpy(b.data(), ref, n*4, cudaMemcpyDeviceToHost));
    cudaFree(ref);
    ++g_topk_checks; g_topk_slots += (long long)n;
    int bad = -1; for(size_t i=0;i<n;++i) if(a[i]!=b[i]){ bad=(int)i; break; }
    topk_gate_report("decode", (int)n, bad<0?0:a[bad], bad<0?0:b[bad], bad, pos);
}

// ---- prefill: fill window-KV + compressed-KV caches ----
void compressed_attn_cache(float* win_kv, float* comp_kv, int* T, const float* x,
                           const CompressedAttnWeights& w, int s0, int ratio, float eps, cudaStream_t stream){
    const auto& a = w.attn;
    uint8_t* xq; float* xs;
    xq=(decltype(xq))dmalloc((size_t)s0*DIM); xs=(decltype(xs))dmalloc((size_t)s0*(DIM/128)*4);
    act_quant_fp8(xq, xs, x, s0, DIM, 128, stream);
    float* wst = kv_stage(win_kv, s0);           // == win_kv when the cache is FP32 (1b.2)
    fp8_block_gemm(wst, xq, xs, a.wkv, a.wkv_s, s0, HEAD_DIM, DIM, stream);
    rmsnorm(wst, wst, a.kv_norm, s0, HEAD_DIM, eps, true, stream);
    rope_interleaved(wst + NOPE_DIM, a.cosT, a.sinT, s0, ROPE_DIM, false, HEAD_DIM, 1, stream);
    kv_commit(win_kv, wst, s0, stream);
    dsync(stream); kv_stage_free(wst, win_kv); dfree(xq); dfree(xs);
    // compressed rows for every COMPLETE group (group g's last token g*ratio+ratio-1 <= s0-1)
    int Tc = 0;
    for(int g=0; g*ratio + ratio - 1 <= s0 - 1; ++g){
        compressor_emit_group(kv_row(comp_kv, g), x, g, ratio, w.mc_wkv, w.mc_wgate, w.mc_ape,
                              w.mc_norm, w.cc_cos, w.cc_sin, DIM, HEAD_DIM, false, ROPE_DIM, eps, false, stream);
        ++Tc;
    }
    *T = Tc;
}

// ---- decode step (strided) ----
void compressed_decode_step_strided(float* out, const float* x_cur, const float* x_full, int pos, const CompressedAttnWeights& w,
                                    float* win_kv, float* comp_kv, int* T, int ratio, float eps, cudaStream_t stream){
    const auto& a = w.attn;
    const int half=ROPE_DIM/2, Kd=N_HEADS*HEAD_DIM, GKd=Kd/O_GROUPS, OB=O_GROUPS*O_LORA;
    const float scale = 1.f/sqrtf((float)HEAD_DIM);
    const float *cosP = a.cosT + (size_t)pos*half, *sinP = a.sinT + (size_t)pos*half;
    const float* xt = x_cur;

    uint8_t *xq,*qrq,*ogq; float *xs,*qrs,*ogs,*qr,*q,*o,*og;
    xq=(decltype(xq))dmalloc(DIM); xs=(decltype(xs))dmalloc((DIM/128)*4);
    qr=(decltype(qr))dmalloc(Q_LORA*4); qrq=(decltype(qrq))dmalloc(Q_LORA); qrs=(decltype(qrs))dmalloc((Q_LORA/128)*4);
    q=(decltype(q))dmalloc(Kd*4); o=(decltype(o))dmalloc(Kd*4); og=(decltype(og))dmalloc(OB*4);
    ogq=(decltype(ogq))dmalloc(OB); ogs=(decltype(ogs))dmalloc((OB/128)*4);

    act_quant_fp8(xq, xs, xt, 1, DIM, 128, stream);
    // q = rope( per-head-rms( wq_b( q_norm( wq_a(x) ) ) ) )
    fp8_block_gemm(qr, xq, xs, a.wq_a, a.wq_a_s, 1, Q_LORA, DIM, stream);
    rmsnorm(qr, qr, a.q_norm, 1, Q_LORA, eps, true, stream);
    act_quant_fp8(qrq, qrs, qr, 1, Q_LORA, 128, stream);
    fp8_block_gemm(q, qrq, qrs, a.wq_b, a.wq_b_s, 1, Kd, Q_LORA, stream);
    rmsnorm(q, q, nullptr, N_HEADS, HEAD_DIM, eps, false, stream);
    rope_interleaved(q + NOPE_DIM, cosP, sinP, N_HEADS, ROPE_DIM, false, HEAD_DIM, N_HEADS, stream);
    // window kv new -> win_kv[pos]
    float* kvrow = kv_row(win_kv, pos);
    float* kvn = kv_stage(kvrow, 1);
    fp8_block_gemm(kvn, xq, xs, a.wkv, a.wkv_s, 1, HEAD_DIM, DIM, stream);
    rmsnorm(kvn, kvn, a.kv_norm, 1, HEAD_DIM, eps, true, stream);
    rope_interleaved(kvn + NOPE_DIM, cosP, sinP, 1, ROPE_DIM, false, HEAD_DIM, 1, stream);
    kv_commit(kvrow, kvn, 1, stream); kv_stage_free(kvn, kvrow);
    // emit compressed row if this token completes a group
    if((pos+1) % ratio == 0){
        int g = pos / ratio;
        compressor_emit_group(kv_row(comp_kv, *T), x_full, g, ratio, w.mc_wkv, w.mc_wgate, w.mc_ape,
                              w.mc_norm, w.cc_cos, w.cc_sin, DIM, HEAD_DIM, false, ROPE_DIM, eps, false, stream);
        ++(*T);
    }
    int Tn = *T, nwin = pos+1;
    // kv_all = [win_kv ; comp_kv] -- already contiguous when g_kv_winmax is set, else copied.
    const int coff = g_kv_winmax ? g_kv_winmax : nwin;      // row where compressed rows begin
    int ntot = coff + Tn;
    float* kv_all;
    if (g_kv_winmax) { kv_all = win_kv; }
    else {
        kv_all=(decltype(kv_all))dmalloc(kv_rows_bytes(ntot));
        CU(cudaMemcpyAsync(kv_all, win_kv, kv_rows_bytes(nwin), cudaMemcpyDeviceToDevice, stream));
        CU(cudaMemcpyAsync(kv_row(kv_all, nwin), comp_kv, kv_rows_bytes(Tn), cudaMemcpyDeviceToDevice, stream));
    }
    // combined idxs: window [base..pos] ⊕ compressed [nwin + t] for t<Tn (strided: all t<(pos+1)/ratio == Tn)
    int base = pos - WINDOW + 1; if(base<0) base=0; int wwidth = pos+1-base;
    int tot = wwidth + Tn;
    int* comb = comb_pinned((size_t)tot);
    for(int k=0;k<wwidth;++k) comb[k]=base+k;
    for(int t=0;t<Tn;++t) comb[wwidth+t]=coff+t;
    int* dcomb; dcomb=(decltype(dcomb))dmalloc((size_t)tot*4);
    CU(cudaMemcpyAsync(dcomb, comb, (size_t)tot*4, cudaMemcpyHostToDevice, stream));
    sparse_attn_kv(o, q, kv_all, a.attn_sink, dcomb, 1, 1, N_HEADS, HEAD_DIM, ntot, tot, scale, stream);
    // de-rotate, grouped o-LoRA, wo_b
    rope_interleaved(o + NOPE_DIM, cosP, sinP, N_HEADS, ROPE_DIM, true, HEAD_DIM, N_HEADS, stream);
    if(a.wo_a_native) ogroup_gemm_fp8(og, o, a.wo_a_fp8, a.wo_a_sc, 1, O_GROUPS, O_LORA, GKd, stream);
    else              ogroup_gemm    (og, o, a.wo_a,                1, O_GROUPS, O_LORA, GKd, stream);
    act_quant_fp8(ogq, ogs, og, 1, OB, 128, stream);
    fp8_block_gemm(out, ogq, ogs, a.wo_b, a.wo_b_s, 1, DIM, OB, stream);

    dsync(stream);
    dfree(xq);dfree(xs);dfree(qr);dfree(qrq);dfree(qrs);dfree(q);
    dfree(o);dfree(og);dfree(ogq);dfree(ogs);dfree(kv_all);dfree(dcomb);
}

// ================= ratio-4 (DSA indexer) decode =================
// Prefill cache for a ratio-4 layer: window KV + MAIN compressed KV (overlap) + INDEXER compressor KV
// (overlap+rotate, for scoring). All append-only. Sets *T = complete-group count.
void compressed_attn_cache_r4(float* win_kv, float* comp_kv, float* idx_ckv, int* T, const float* x,
                              const CompressedAttnWeights& w, int s0, int ratio, float eps, cudaStream_t stream){
    const auto& a = w.attn; const int idx_hd = w.index_head_dim;
    uint8_t* xq; float* xs;
    xq=(decltype(xq))dmalloc((size_t)s0*DIM); xs=(decltype(xs))dmalloc((size_t)s0*(DIM/128)*4);
    act_quant_fp8(xq, xs, x, s0, DIM, 128, stream);
    float* wst = kv_stage(win_kv, s0);           // == win_kv when the cache is FP32 (1b.2)
    fp8_block_gemm(wst, xq, xs, a.wkv, a.wkv_s, s0, HEAD_DIM, DIM, stream);
    rmsnorm(wst, wst, a.kv_norm, s0, HEAD_DIM, eps, true, stream);
    rope_interleaved(wst + NOPE_DIM, a.cosT, a.sinT, s0, ROPE_DIM, false, HEAD_DIM, 1, stream);
    kv_commit(win_kv, wst, s0, stream);
    dsync(stream); kv_stage_free(wst, win_kv); dfree(xq); dfree(xs);
    int Tc = 0;
    // ---- BATCHED CACHE POPULATION (DSV4_BATCH_COMPRESS=1) ----------------------------------------
    // The loop below emits ONE group per call, twice per group. At a 845-token prefill with ratio 4
    // that is 211 groups x 2 x 41 layers = 17,302 launches, each doing the work of four tokens and
    // each RE-READING the compressor's whole weight matrix -- the same defect compressor.cu already
    // documents ("one group emit read 8 x 33.6 MB where it needed 33.6"). `cattn:compress` is
    // 1429 ms of a 9.1 s prefill and this is what is in it.
    //
    // compressor_forward is the batched equivalent, already used by compressed_attn_forward on the
    // same weights with the same overlap/rotate flags: it runs one gemm over the whole sequence
    // (weights read ONCE), pools every group, and rmsnorms the lot. Per-token gemm rows and per-group
    // pooling are independent, so the arithmetic per group is the work the loop was doing.
    //
    // GUARDED ON LAYOUT, not assumed: compressor_forward writes [groups, d] contiguously, and
    // kv_row() strides by g_kv_rowf, so the two agree only when g_kv_rowf == HEAD_DIM -- true in the
    // default FP32 cache (2048 B/row) and false under DSV4_KV_PACK. idx_ckv is a plain contiguous
    // buffer in both modes.
    // DEFAULT ON (2026-08-26): measured bit-exact over 400 generated tokens and worth 78% of this
    // region -- cattn:compress 1431 -> 318 ms, prefill 92.8 -> 105.4 tok/s. Set DSV4_BATCH_COMPRESS=0
    // to restore the per-group loop (or run with a packed KV cache, which takes the guard below).
    const char* bce = getenv("DSV4_BATCH_COMPRESS");
    const bool batch_ok = (g_kv_rowf == HEAD_DIM) && (!bce || bce[0] != '0');
    if (batch_ok) {
        compressor_forward(comp_kv, x, w.mc_wkv, w.mc_wgate, w.mc_ape, w.mc_norm, w.cc_cos, w.cc_sin,
                           s0, DIM, HEAD_DIM, ratio, true, ROPE_DIM, eps, false, stream);
        compressor_forward(idx_ckv, x, w.idx_c_wkv, w.idx_c_wgate, w.idx_c_ape, w.idx_c_norm,
                           w.cc_cos, w.cc_sin, s0, DIM, idx_hd, ratio, true, ROPE_DIM, eps, true, stream);
        Tc = s0 / ratio;
    } else
    for(int g=0; g*ratio + ratio - 1 <= s0 - 1; ++g){
        compressor_emit_group(kv_row(comp_kv, g), x, g, ratio, w.mc_wkv, w.mc_wgate, w.mc_ape,
                              w.mc_norm, w.cc_cos, w.cc_sin, DIM, HEAD_DIM, true, ROPE_DIM, eps, false, stream);
        compressor_emit_group(idx_ckv + (size_t)g*idx_hd, x, g, ratio, w.idx_c_wkv, w.idx_c_wgate, w.idx_c_ape,
                              w.idx_c_norm, w.cc_cos, w.cc_sin, DIM, idx_hd, true, ROPE_DIM, eps, true, stream);
        ++Tc;
    }
    *T = Tc;
}

void compressed_decode_step_indexer(float* out, const float* x_cur, const float* x_full, int pos, const CompressedAttnWeights& w,
                                    float* win_kv, float* comp_kv, float* idx_ckv, int* T, int ratio,
                                    float eps, cudaStream_t stream){
    const auto& a = w.attn;
    const int half=ROPE_DIM/2, Kd=N_HEADS*HEAD_DIM, GKd=Kd/O_GROUPS, OB=O_GROUPS*O_LORA;
    const int nH=w.index_n_heads, idx_hd=w.index_head_dim, QD=nH*idx_hd, rd=ROPE_DIM;
    const float scale = 1.f/sqrtf((float)HEAD_DIM);
    const float wscale = rsqrtf((float)idx_hd) * rsqrtf((float)nH);
    const float *cosP = a.cosT + (size_t)pos*half, *sinP = a.sinT + (size_t)pos*half;
    const float* xt = x_cur;

    uint8_t *xq,*qrq,*ogq; float *xs,*qrs,*ogs,*qr,*q,*o,*og,*qidx,*qtmp,*iw,*iscore;
    xq=(decltype(xq))dmalloc(DIM); xs=(decltype(xs))dmalloc((DIM/128)*4);
    qr=(decltype(qr))dmalloc(Q_LORA*4); qrq=(decltype(qrq))dmalloc(Q_LORA); qrs=(decltype(qrs))dmalloc((Q_LORA/128)*4);
    q=(decltype(q))dmalloc(Kd*4); o=(decltype(o))dmalloc(Kd*4); og=(decltype(og))dmalloc(OB*4);
    ogq=(decltype(ogq))dmalloc(OB); ogs=(decltype(ogs))dmalloc((OB/128)*4);
    qidx=(decltype(qidx))dmalloc(QD*4); qtmp=(decltype(qtmp))dmalloc(QD*4); iw=(decltype(iw))dmalloc(nH*4);

    act_quant_fp8(xq, xs, xt, 1, DIM, 128, stream);
    // qr = q_norm(wq_a(x))  (shared by main-q and indexer-q)
    fp8_block_gemm(qr, xq, xs, a.wq_a, a.wq_a_s, 1, Q_LORA, DIM, stream);
    rmsnorm(qr, qr, a.q_norm, 1, Q_LORA, eps, true, stream);
    // main q
    act_quant_fp8(qrq, qrs, qr, 1, Q_LORA, 128, stream);
    fp8_block_gemm(q, qrq, qrs, a.wq_b, a.wq_b_s, 1, Kd, Q_LORA, stream);
    rmsnorm(q, q, nullptr, N_HEADS, HEAD_DIM, eps, false, stream);
    rope_interleaved(q + NOPE_DIM, cosP, sinP, N_HEADS, ROPE_DIM, false, HEAD_DIM, N_HEADS, stream);
    // window kv new -> win_kv[pos]
    float* kvrow = kv_row(win_kv, pos);
    float* kvn = kv_stage(kvrow, 1);
    fp8_block_gemm(kvn, xq, xs, a.wkv, a.wkv_s, 1, HEAD_DIM, DIM, stream);
    rmsnorm(kvn, kvn, a.kv_norm, 1, HEAD_DIM, eps, true, stream);
    rope_interleaved(kvn + NOPE_DIM, cosP, sinP, 1, ROPE_DIM, false, HEAD_DIM, 1, stream);
    kv_commit(kvrow, kvn, 1, stream); kv_stage_free(kvn, kvrow);
    // emit compressed rows (main overlap + indexer overlap+rotate) if group completes
    if((pos+1) % ratio == 0){
        int g = pos / ratio; int Tn = *T;
        compressor_emit_group(kv_row(comp_kv, Tn), x_full, g, ratio, w.mc_wkv, w.mc_wgate, w.mc_ape,
                              w.mc_norm, w.cc_cos, w.cc_sin, DIM, HEAD_DIM, true, ROPE_DIM, eps, false, stream);
        compressor_emit_group(idx_ckv + (size_t)Tn*idx_hd, x_full, g, ratio, w.idx_c_wkv, w.idx_c_wgate,
                              w.idx_c_ape, w.idx_c_norm, w.cc_cos, w.cc_sin, DIM, idx_hd, true, ROPE_DIM, eps, true, stream);
        ++(*T);
    }
    int Tn = *T, nwin = pos+1;
    // --- DSA indexer scoring for the single query -> top-k compressed idxs (mirrors indexer_forward, m=1) ---
    // `qrq/qrs` above are act_quant(qr, Q_LORA, 128) — the identical call, on an unmodified `qr`.
    fp8_block_gemm(qidx, qrq, qrs, w.idx_wq_b, w.idx_wq_b_s, 1, QD, Q_LORA, stream);
    rope_interleaved(qidx + (idx_hd - rd), cosP, sinP, nH, rd, false, idx_hd, nH, stream);
    hadamard(qtmp, qidx, nH, idx_hd, stream);
    act_quant_fp4sim(qtmp, nH, idx_hd, 32, idx_hd, stream);
    gemm_fp32(iw, xt, w.idx_weights_proj, 1, nH, DIM, stream);
    k_iw_scale<<<(nH+63)/64,64,0,stream>>>(iw, wscale, nH);
    iscore=(decltype(iscore))dmalloc((size_t)Tn*4);
    index_score(iscore, qtmp, idx_ckv, iw, 1, Tn, nH, idx_hd, stream);
    int topk = w.index_topk < Tn ? w.index_topk : Tn;
    int* dtop; dtop=(decltype(dtop))dmalloc((size_t)topk*4);
    const int coff = g_kv_winmax ? g_kv_winmax : nwin;      // row where compressed rows begin
    if(topk_radix_on() && topk<=TOPK_RADIX_CAP) k_topk_decode_rx<<<1,TOPK_RADIX_NT,0,stream>>>(dtop, iscore, Tn, topk, coff, topk_early_on());
    else{ k_topk_decode<<<1,32,TOPK_SMEM(k_topk_decode,Tn),stream>>>(dtop, iscore, Tn, topk, coff);
          TOPK_LAUNCHED(k_topk_decode, Tn); }
    if(topk_gate_on() && topk_radix_on() && topk<=TOPK_RADIX_CAP) topk_gate_decode(dtop, iscore, Tn, topk, coff, pos, stream);
    // --- kv_all = [win_kv ; comp_kv] -- contiguous already when g_kv_winmax is set ---
    int ntot = coff + Tn;
    float* kv_all;
    if (g_kv_winmax) { kv_all = win_kv; }
    else {
        kv_all=(decltype(kv_all))dmalloc(kv_rows_bytes(ntot));
        CU(cudaMemcpyAsync(kv_all, win_kv, kv_rows_bytes(nwin), cudaMemcpyDeviceToDevice, stream));
        CU(cudaMemcpyAsync(kv_row(kv_all, nwin), comp_kv, kv_rows_bytes(Tn), cudaMemcpyDeviceToDevice, stream));
    }
    // --- combined idxs: window [base..pos] ⊕ indexer topk ---
    int base = pos - WINDOW + 1; if(base<0) base=0; int wwidth = pos+1-base;
    int tot = wwidth + topk;
    int* hwin = comb_pinned((size_t)wwidth); for(int k=0;k<wwidth;++k) hwin[k]=base+k;
    int* comb; comb=(decltype(comb))dmalloc((size_t)tot*4);
    CU(cudaMemcpyAsync(comb, hwin, (size_t)wwidth*4, cudaMemcpyHostToDevice, stream));
    CU(cudaMemcpyAsync(comb + wwidth, dtop, (size_t)topk*4, cudaMemcpyDeviceToDevice, stream));
    sparse_attn_kv(o, q, kv_all, a.attn_sink, comb, 1, 1, N_HEADS, HEAD_DIM, ntot, tot, scale, stream);
    rope_interleaved(o + NOPE_DIM, cosP, sinP, N_HEADS, ROPE_DIM, true, HEAD_DIM, N_HEADS, stream);
    if(a.wo_a_native) ogroup_gemm_fp8(og, o, a.wo_a_fp8, a.wo_a_sc, 1, O_GROUPS, O_LORA, GKd, stream);
    else              ogroup_gemm    (og, o, a.wo_a,                1, O_GROUPS, O_LORA, GKd, stream);
    act_quant_fp8(ogq, ogs, og, 1, OB, 128, stream);
    fp8_block_gemm(out, ogq, ogs, a.wo_b, a.wo_b_s, 1, DIM, OB, stream);

    dsync(stream);
    dfree(xq);dfree(xs);dfree(qr);dfree(qrq);dfree(qrs);dfree(q);dfree(o);dfree(og);
    dfree(ogq);dfree(ogs);dfree(qidx);dfree(qtmp);dfree(iw);
    dfree(iscore);dfree(dtop);dfree(kv_all);dfree(comb);
}

// ================= M=K VERIFY steps (spec-decode) =================
// Process K tokens at [pos..pos+K-1] in ONE forward: GEMMs at M=K (weights read once), compressor emits any
// groups completing in the block, per-query combined idxs (window ⊕ compressed). ≡ K sequential decode steps.
static void build_qKV(const CompressedAttnWeights& w, const float* xK, int K, int pos, float* qOut, float* win_kv,
                      uint8_t** qrq_out, float** qrs_out,
                      float eps, cudaStream_t stream){
    const auto& a=w.attn; const int half=ROPE_DIM/2, Kd=N_HEADS*HEAD_DIM;
    const float *cosP=a.cosT+(size_t)pos*half, *sinP=a.sinT+(size_t)pos*half;
    uint8_t *xq,*qrq; float *xs,*qrs,*qr;
    xq=(uint8_t*)dmalloc((size_t)K*DIM); xs=(float*)dmalloc((size_t)K*(DIM/128)*4);
    qr=(float*)dmalloc((size_t)K*Q_LORA*4); qrq=(uint8_t*)dmalloc((size_t)K*Q_LORA); qrs=(float*)dmalloc((size_t)K*(Q_LORA/128)*4);
    dprof_begin(DP_C_QPROJ,stream);
    dprof_begin(DP_Q_AQX,stream);
    act_quant_fp8(xq,xs,xK,K,DIM,128,stream);
    dprof_end(DP_Q_AQX,stream);
    // C1 (Finding 55/56). The kv chain consumes `xq/xs` and nothing else — it is independent of the
    // whole wq_a -> rmsnorm -> act_quant -> wq_b chain that follows, and `wkv [512,4096]` is the
    // WORST shape on overlap_probe: 47.8 GB/s standalone at M=5 against 150.4 on four streams
    // (3.15x), because 512 rows is 32 m16 tiles and 32 blocks cannot cover DRAM latency on 20 SMs.
    //
    // g_side2, NOT g_side: this fork nests inside the compressor fork in
    // compressed_verify_step_indexer, which owns g_side. Re-recording g_side_fork here while that
    // pair is outstanding would rewire the dependency graph and still return plausible numbers.
    // NO_KV_SPLIT=1 restores the serial order for A/B.
    const bool kvsplit = g_side2 && !getenv("NO_KV_SPLIT");
    cudaStream_t ks = kvsplit ? g_side2 : stream;
    float* kvrow=kv_row(win_kv,pos);
    float* kvn=kv_stage(kvrow,K);
    if(kvsplit){
        cudaEventRecord(g_side2_fork,stream); cudaStreamWaitEvent(ks,g_side2_fork,0);
        fp8_block_gemm(kvn,xq,xs,a.wkv,a.wkv_s,K,HEAD_DIM,DIM,ks);
        rmsnorm(kvn,kvn,a.kv_norm,K,HEAD_DIM,eps,true,ks);
        rope_interleaved(kvn+NOPE_DIM,cosP,sinP,K,ROPE_DIM,false,HEAD_DIM,1,ks);
        kv_commit(kvrow,kvn,K,ks);
        cudaEventRecord(g_side2_join,ks);
    }
    dprof_begin(DP_Q_WQA,stream);
    fp8_block_gemm(qr,xq,xs,a.wq_a,a.wq_a_s,K,Q_LORA,DIM,stream);
    dprof_end(DP_Q_WQA,stream);
    // Two elementwise kernels on <=20 KB. If this mark is large it is dependent-kernel exposure, not
    // work, and fusing rmsnorm into act_quant is worth building; if it is small, q:wq_a is the GEMM
    // and fusion is not the lever. 0.4% vs 4% turns on this number.
    dprof_begin(DP_Q_RMSAQ,stream);
    rmsnorm(qr,qr,a.q_norm,K,Q_LORA,eps,true,stream);
    act_quant_fp8(qrq,qrs,qr,K,Q_LORA,128,stream);
    dprof_end(DP_Q_RMSAQ,stream);
    dprof_begin(DP_Q_WQB,stream);
    fp8_block_gemm(qOut,qrq,qrs,a.wq_b,a.wq_b_s,K,Kd,Q_LORA,stream);
    dprof_end(DP_Q_WQB,stream);
    dprof_begin(DP_Q_TAIL,stream);
    rmsnorm(qOut,qOut,nullptr,K*N_HEADS,HEAD_DIM,eps,false,stream);
    rope_interleaved(qOut+NOPE_DIM,cosP,sinP,K*N_HEADS,ROPE_DIM,false,HEAD_DIM,N_HEADS,stream);
    dprof_end(DP_Q_TAIL,stream);
    // How much of this region is the main stream WAITING on the C1 kv fork rather than doing work.
    dprof_begin(DP_Q_KVJOIN,stream);
    if(kvsplit){ cudaStreamWaitEvent(stream,g_side2_join,0); }        // win_kv is read by the caller
    else {
        fp8_block_gemm(kvn,xq,xs,a.wkv,a.wkv_s,K,HEAD_DIM,DIM,stream);
        rmsnorm(kvn,kvn,a.kv_norm,K,HEAD_DIM,eps,true,stream);
        rope_interleaved(kvn+NOPE_DIM,cosP,sinP,K,ROPE_DIM,false,HEAD_DIM,1,stream);
        kv_commit(kvrow,kvn,K,stream);
    }
    kv_stage_free(kvn,kvrow);
    dprof_end(DP_Q_KVJOIN,stream);
    dprof_end(DP_C_QPROJ,stream);
    // The indexer needs act_quant(rmsnorm(wq_a @ x)) — byte-for-byte what `qrq/qrs` already hold.
    // It used to rebuild them from x: act_quant(x) + the whole wq_a GEMM (4.19 MB/layer) + rmsnorm
    // + act_quant, four kernels and 88 MB of redundant weight traffic across the 21 indexer layers,
    // under a comment calling the recompute "cheap". Hand them out instead; the caller frees.
    if(qrq_out){ *qrq_out=qrq; *qrs_out=qrs; } else { dfree(qrq); dfree(qrs); }
    dfree(xq);dfree(xs);dfree(qr);
}
static void finish_attn(const CompressedAttnWeights& w, const float* q, const float* kv_all, const int* comb,
                        int K, int pos, int ntot, int topk, float* out, float eps, cudaStream_t stream){
    const auto& a=w.attn; const int half=ROPE_DIM/2, Kd=N_HEADS*HEAD_DIM, GKd=Kd/O_GROUPS, OB=O_GROUPS*O_LORA;
    const float *cosP=a.cosT+(size_t)pos*half, *sinP=a.sinT+(size_t)pos*half; const float scale=1.f/sqrtf((float)HEAD_DIM);
    float *o,*og; uint8_t *ogq; float *ogs;
    o=(float*)dmalloc((size_t)K*Kd*4); og=(float*)dmalloc((size_t)K*OB*4); ogq=(uint8_t*)dmalloc((size_t)K*OB); ogs=(float*)dmalloc((size_t)K*(OB/128)*4);
    dprof_begin(DP_C_SPARSE,stream);
    sparse_attn_kv(o,q,kv_all,a.attn_sink,comb,1,K,N_HEADS,HEAD_DIM,ntot,topk,scale,stream);
    dprof_end(DP_C_SPARSE,stream);
    dprof_begin(DP_C_OGROUP,stream);
    dprof_begin(DP_O_ROPE,stream);
    rope_interleaved(o+NOPE_DIM,cosP,sinP,K*N_HEADS,ROPE_DIM,true,HEAD_DIM,N_HEADS,stream);
    dprof_end(DP_O_ROPE,stream);
    dprof_begin(DP_O_WOA,stream);
    if(a.wo_a_native) ogroup_gemm_fp8(og,o,a.wo_a_fp8,a.wo_a_sc,K,O_GROUPS,O_LORA,GKd,stream);
    else              ogroup_gemm    (og,o,a.wo_a,               K,O_GROUPS,O_LORA,GKd,stream);
    act_quant_fp8(ogq,ogs,og,K,OB,128,stream);
    dprof_end(DP_O_WOA,stream);
    dprof_begin(DP_O_WOB,stream);
    fp8_block_gemm(out,ogq,ogs,a.wo_b,a.wo_b_s,K,DIM,OB,stream);
    dprof_end(DP_O_WOB,stream);
    dprof_end(DP_C_OGROUP,stream);
    dfree(o);dfree(og);dfree(ogq);dfree(ogs);
}

void compressed_verify_step_strided(float* out, const float* x_cur, const float* x_full, int pos, int K, const CompressedAttnWeights& w,
                                    float* win_kv, float* comp_kv, int* T, int ratio, float eps, cudaStream_t stream){
    const int Kd=N_HEADS*HEAD_DIM;
    float* q; q=(float*)dmalloc((size_t)K*Kd*4);
    build_qKV(w, x_cur, K, pos, q, win_kv, nullptr, nullptr, eps, stream);
    dprof_begin(DP_C_COMPRESS,stream);
    for(int j=pos;j<pos+K;++j) if((j+1)%ratio==0){                         // emit groups completing in the block
        compressor_emit_group(kv_row(comp_kv, *T), x_full, j/ratio, ratio, w.mc_wkv,w.mc_wgate,w.mc_ape,
                              w.mc_norm,w.cc_cos,w.cc_sin, DIM,HEAD_DIM,false,ROPE_DIM,eps,false,stream); ++(*T); }
    dprof_end(DP_C_COMPRESS,stream);
    int Tf=*T, nwin=pos+K;
    const int coff = g_kv_winmax ? g_kv_winmax : nwin;
    int ntot = coff + Tf;
    float* kv_all;
    if (g_kv_winmax) { kv_all = win_kv; }
    else {
        kv_all=(float*)dmalloc(kv_rows_bytes(ntot));
        CU(cudaMemcpyAsync(kv_all,win_kv,kv_rows_bytes(nwin),cudaMemcpyDeviceToDevice,stream));
        CU(cudaMemcpyAsync(kv_row(kv_all,nwin),comp_kv,kv_rows_bytes(Tf),cudaMemcpyDeviceToDevice,stream));
    }
    int wmax=0,tmax=0; for(int i=0;i<K;++i){int ig=pos+i,b=ig-WINDOW+1;if(b<0)b=0;int wid=ig+1-b;if(wid>wmax)wmax=wid;int Ti=(ig+1)/ratio;if(Ti>tmax)tmax=Ti;}
    int topk=wmax+tmax; int* comb = comb_pinned((size_t)K*topk);
    for(size_t z=0; z<(size_t)K*topk; ++z) comb[z]=-1;
    for(int i=0;i<K;++i){int ig=pos+i,b=ig-WINDOW+1;if(b<0)b=0;int wid=ig+1-b;int Ti=(ig+1)/ratio;
        for(int k=0;k<wid;++k) comb[(size_t)i*topk+k]=b+k;
        for(int t=0;t<Ti;++t) comb[(size_t)i*topk+wmax+t]=coff+t; }
    int* dcomb; dcomb=(int*)dmalloc((size_t)K*topk*4); CU(cudaMemcpyAsync(dcomb,comb,(size_t)K*topk*4,cudaMemcpyHostToDevice,stream));
    finish_attn(w, q, kv_all, dcomb, K, pos, ntot, topk, out, eps, stream);
    dsync(stream); dfree(q);dfree(kv_all);dfree(dcomb);
}

void compressed_verify_step_indexer(float* out, const float* x_cur, const float* x_full, int pos, int K, const CompressedAttnWeights& w,
                                    float* win_kv, float* comp_kv, float* idx_ckv, int* T, int ratio, float eps, cudaStream_t stream){
    const auto& a=w.attn; const int half=ROPE_DIM/2, Kd=N_HEADS*HEAD_DIM, nH=w.index_n_heads, ihd=w.index_head_dim, QD=nH*ihd, rd=ROPE_DIM;
    const float wscale=rsqrtf((float)ihd)*rsqrtf((float)nH); const float *cosP=a.cosT+(size_t)pos*half, *sinP=a.sinT+(size_t)pos*half;
    float* q; q=(float*)dmalloc((size_t)K*Kd*4);
    uint8_t* iqrq=nullptr; float* iqrs=nullptr;
    // CONCURRENT COMPRESSOR (LOOP_LOG Finding 55, ATTN_SPLIT=1). The two compressor emits read
    // `x_full` and nothing else — they never depended on build_qKV, they were just issued after it
    // on the same stream. In situ at K=5 that costs the full 8.40 ms of `cattn:compress` next to
    // 17.33 ms of `cattn:q_proj`, and neither saturates memory alone (overlap_probe: wq_a 86.4 ->
    // 194.2 GB/s at 4 streams). The join is before the indexer, which is the first consumer of
    // `idx_ckv`, and before kv_all, which is the first consumer of `comp_kv`.
    //
    // The arena is safe across the two streams for the same reason it is safe within one: every
    // dmalloc happens on the HOST in program order, so build_qKV's scratch and the emits' scratch
    // are disjoint ranges. What would NOT be safe is reusing a buffer, which is why the shared
    // expert needed its own (see moe.cu).
    // MEASURED AND ADOPTED (Finding 56): cattn:compress 8.56 -> 0.51 ms, cattn:q_proj 17.03 -> 23.85
    // (it absorbs the traffic, exactly as the shared expert did), K=5 verify 155.22 -> 153.48 ms,
    // spec 17.32 -> 17.48 tok/s, base AR 10.50 -> 10.68, token sequence byte-identical.
    // NO_ATTN_SPLIT=1 restores the serial order for A/B.
    // LADDER 1.11 — DEFER THE JOIN PAST i:qidx AND i:iw. 1.8 measured 2.50 ms of side-stream work
    // still un-hidden in cattn:compress on every emit step, because the join sat here, immediately
    // after build_qKV. But nothing between here and `index_score` reads what the emits write: the
    // first consumer of `idx_ckv` is index_score and of `comp_kv` is the kv_all copy, while i:qidx
    // (fp8_block_gemm/rope/hadamard on iqrq/iqrs) and i:iw (gemm_fp32 on x_cur) read neither. That
    // is 1.99 ms of independent main-stream work that was sitting behind the barrier.
    //
    // BIT-EXACT BY CONSTRUCTION: kernels move between streams, dependency order is unchanged, no
    // arithmetic is touched. The arena is safe for the same reason the fork already is — every
    // dmalloc happens on the HOST in program order, so the emits' scratch and qidx/qtmp/iw are
    // disjoint ranges; `compressor_emit_group` ends in dsync+dfree, both no-ops under the arena,
    // and arena_reset() is per layer, outside this function.
    //
    // READ THE dprof MARKS WITH CARE AFTER THIS. Deferring the join MOVES time between marks: with
    // the wait gone, cattn:compress collapses toward zero on the main stream and the emit traffic
    // reappears inside i:qidx / i:iw, exactly as 1.8 showed it reappearing in cattn:compress under
    // NO_ATTN_SPLIT=1. A shrinking cattn:compress is NOT the win; only the paired ms/forward is.
    // NO_JOIN_DEFER=1 restores the 1.8 join position for A/B on ONE binary.
    const bool asplit = g_side && !getenv("NO_ATTN_SPLIT");
    const bool jdefer = asplit && !getenv("NO_JOIN_DEFER");
    cudaStream_t cs = asplit ? g_side : stream;
    if(asplit){ cudaEventRecord(g_side_fork,stream); cudaStreamWaitEvent(cs,g_side_fork,0); }
    build_qKV(w, x_cur, K, pos, q, win_kv, &iqrq, &iqrs, eps, stream);
    dprof_begin(DP_C_COMPRESS,stream);
    // emit main + indexer compressed rows for groups completing in the block
    for(int j=pos;j<pos+K;++j) if((j+1)%ratio==0){ int g=j/ratio; int t=*T;
        compressor_emit_group(kv_row(comp_kv, t), x_full, g, ratio, w.mc_wkv,w.mc_wgate,w.mc_ape,w.mc_norm,w.cc_cos,w.cc_sin,DIM,HEAD_DIM,true,ROPE_DIM,eps,false,cs);
        compressor_emit_group(idx_ckv+(size_t)t*ihd, x_full, g, ratio, w.idx_c_wkv,w.idx_c_wgate,w.idx_c_ape,w.idx_c_norm,w.cc_cos,w.cc_sin,DIM,ihd,true,ROPE_DIM,eps,true,cs); ++(*T); }
    if(asplit) cudaEventRecord(g_side_join,cs);                 // "the emits are done" — recorded here
    if(asplit && !jdefer) cudaStreamWaitEvent(stream,g_side_join,0);   // ...but waited on where it is needed
    dprof_end(DP_C_COMPRESS,stream);
    int Tf=*T, nwin=pos+K;
    dprof_begin(DP_C_INDEXER,stream);
    // qidx = fp4sim(hadamard(rope(idx_wq_b(qr)))). `iqrq/iqrs` came out of build_qKV above.
    float *qidx,*qtmp,*iw,*iscore;
    qidx=(float*)dmalloc((size_t)K*QD*4); qtmp=(float*)dmalloc((size_t)K*QD*4); iw=(float*)dmalloc((size_t)K*nH*4);
    dprof_begin(DP_I_QIDX,stream);
    fp8_block_gemm(qidx,iqrq,iqrs,w.idx_wq_b,w.idx_wq_b_s,K,QD,Q_LORA,stream);
    rope_interleaved(qidx+(ihd-rd),cosP,sinP,K*nH,rd,false,ihd,nH,stream);
    hadamard(qtmp,qidx,K*nH,ihd,stream); act_quant_fp4sim(qtmp,K*nH,ihd,32,ihd,stream);
    dprof_end(DP_I_QIDX,stream);
    dprof_begin(DP_I_IW,stream);
    gemm_fp32(iw,x_cur,w.idx_weights_proj,K,nH,DIM,stream);
    k_iw_scale<<<((size_t)K*nH+63)/64,64,0,stream>>>(iw,wscale,K*nH);
    dprof_end(DP_I_IW,stream);
    iscore=(float*)dmalloc((size_t)K*Tf*4);
    // 1.11: THE TRUE DEPENDENCY. index_score is the first reader of `idx_ckv`, so this is the
    // earliest point the compressor emits must have landed. Everything above ran against them.
    if(jdefer) cudaStreamWaitEvent(stream,g_side_join,0);
    // index_score is one THREAD per (query, compressed-row) and there are only K*Tf ~ 95 of them, so
    // it launches a single block. If this mark is large that is 3 warps on one SM doing 1024 MACs
    // each, and the fix is warp-per-output — which changes the reduction order, so it needs the
    // lossless gate (Finding 68) rather than a cosine.
    dprof_begin(DP_I_SCORE,stream);
    index_score(iscore,qtmp,idx_ckv,iw,K,Tf,nH,ihd,stream);
    dprof_end(DP_I_SCORE,stream);
    // per-query top-k with GLOBAL causal threshold (t < (pos+i+1)/ratio), offset nwin
    int topkc = (w.index_topk<Tf)?w.index_topk:Tf;
    int* dtop; dtop=(int*)dmalloc((size_t)K*topkc*4);
    dprof_begin(DP_I_TOPK,stream);
    const int coff = g_kv_winmax ? g_kv_winmax : nwin;
    if(topk_radix_on() && topkc<=TOPK_RADIX_CAP) k_topk_verify_rx<<<K,TOPK_RADIX_NT,0,stream>>>(dtop, iscore, K, Tf, topkc, pos, ratio, coff, topk_early_on());
    else{ k_topk_verify<<<K,32,TOPK_SMEM(k_topk_verify,Tf),stream>>>(dtop, iscore, K, Tf, topkc, pos, ratio, coff);   // device top-k (no D2H sync)
          TOPK_LAUNCHED(k_topk_verify, Tf); }
    if(topk_gate_on() && topk_radix_on() && topkc<=TOPK_RADIX_CAP) topk_gate_verify(dtop, iscore, K, Tf, topkc, pos, ratio, coff, stream);
    dprof_end(DP_I_TOPK,stream);
    dprof_end(DP_C_INDEXER,stream);
    int ntot = coff + Tf;
    float* kv_all;
    if (g_kv_winmax) { kv_all = win_kv; }
    else {
        kv_all=(float*)dmalloc(kv_rows_bytes(ntot));
        CU(cudaMemcpyAsync(kv_all,win_kv,kv_rows_bytes(nwin),cudaMemcpyDeviceToDevice,stream));
        CU(cudaMemcpyAsync(kv_row(kv_all,nwin),comp_kv,kv_rows_bytes(Tf),cudaMemcpyDeviceToDevice,stream));
    }
    int wmax=0; for(int i=0;i<K;++i){int ig=pos+i,b=ig-WINDOW+1;if(b<0)b=0;int wid=ig+1-b;if(wid>wmax)wmax=wid;}
    int topk=wmax+topkc;
    int* dcomb; dcomb=(int*)dmalloc((size_t)K*topk*4);
    k_comb_verify<<<((size_t)K*topk+63)/64,64,0,stream>>>(dcomb, dtop, K, topk, wmax, topkc, pos);
    finish_attn(w, q, kv_all, dcomb, K, pos, ntot, topk, out, eps, stream);
    dsync(stream);
    dfree(q);dfree(iqrq);dfree(iqrs);dfree(qidx);dfree(qtmp);dfree(iw);dfree(iscore);dfree(dtop);dfree(kv_all);dfree(dcomb);
}

// ================= device-pos compressed decode (CUDA-graph capturable) =================
// combined cache kvc = [winmax(window) .. winmax+Tmax(compressed)][HEAD_DIM]; d_T = device compressed count.
#include "mla_forward.h"
__global__ void k_gather_win_dp(float* scr, const float* xin, const int* d_pos, int ntok, int tok_off, int dim, int ratio){
    if(((*d_pos)+1)%ratio!=0) return;   // only gather on commit steps (graph launch stays static)
    long i=(long)blockIdx.x*blockDim.x+threadIdx.x; if(i>=(long)ntok*dim) return; int r=i/dim, c=i%dim;
    int t=(*d_pos)+tok_off+r; if(t<0) t=0; scr[i]=xin[(size_t)t*dim+c]; }   // clamp: candidate computed every step (commit self-masked); early neg windows must not OOB
__global__ void k_dg(int* d_g, const int* d_pos, int ratio){ if(!threadIdx.x&&!blockIdx.x) *d_g=(*d_pos)/ratio; }
__global__ void k_append_at2(float* dst, const float* scr, const int* d_idx, int hd){          // dst[(*d_idx)*hd + i]=scr[i]
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=hd) return; dst[(size_t)(*d_idx)*hd + i]=scr[i]; }
__global__ void k_commit_comp(float* comp_region, const float* cand, const int* d_T, const int* d_pos, int ratio, int hd){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=hd) return;
    if(((*d_pos)+1)%ratio==0) comp_region[(size_t)(*d_T)*hd + i]=cand[i]; }          // append only when a group completes
__global__ void k_advance_T(int* d_T, const int* d_pos, int ratio){ if(!threadIdx.x&&!blockIdx.x && ((*d_pos)+1)%ratio==0) (*d_T)++; }
// strided combined idxs: window [base..pos] (region [0,winmax)) then compressed [winmax + t for t<*d_T]; fixed topk.
__global__ void k_comb_strided_dp(int* comb, const int* d_pos, const int* d_T, int winmax, int wtop, int Tmax){
    int k=blockIdx.x*blockDim.x+threadIdx.x; int topk=wtop+Tmax; if(k>=topk) return; int pos=*d_pos;
    if(k<wtop){ int base=pos-WINDOW+1; if(base<0)base=0; int v=base+k; comb[k]=(v<=pos)? v : -1; }
    else { int t=k-wtop; comb[k]=(t<*d_T)? winmax+t : -1; } }

// device-pos compressor emit: computes the candidate row from the last ntok tokens of xin (ending at *d_pos) and
// commits it to the compressed region + advances *d_T iff (*d_pos+1)%ratio==0. rope at compressed pos d_g.
static void emit_group_dp(float* comp_region, const float* xin, const int* d_pos, int* d_T, int* d_g, int ratio,
        const float* wkv, const float* wgate, const float* ape, const float* norm_w, const float* cc_cos, const float* cc_sin,
        int dim, int d, bool overlap, int rotate, float eps, cudaStream_t stream){
    int coff=overlap?2:1, od=coff*d; int ntok=overlap?2*ratio:ratio, tok_off=-(ntok-1), localg=overlap?1:0;
    float *scr,*kv,*score,*pooled,*cand;
    scr=(float*)dmalloc((size_t)ntok*dim*4); kv=(float*)dmalloc((size_t)ntok*od*4); score=(float*)dmalloc((size_t)ntok*od*4);
    pooled=(float*)dmalloc((size_t)(localg+1)*d*4); cand=(float*)dmalloc((size_t)d*4);
    k_gather_win_dp<<<((size_t)ntok*dim+255)/256,256,0,stream>>>(scr,xin,d_pos,ntok,tok_off,dim,ratio);
    gemm_fp32_cond(kv,scr,wkv,ntok,od,dim,d_pos,ratio,stream); gemm_fp32_cond(score,scr,wgate,ntok,od,dim,d_pos,ratio,stream);
    if(overlap) compressor_pool_overlap(pooled,kv,score,ape,localg+1,ratio,d,stream);
    else        compressor_pool(pooled,kv,score,ape,1,ratio,d,stream);
    rmsnorm(cand,pooled+(size_t)localg*d,norm_w,1,d,eps,true,stream);
    k_dg<<<1,1,0,stream>>>(d_g,d_pos,ratio);
    rope_interleaved_dp(cand+(d-ROPE_DIM),cc_cos,cc_sin,1,ROPE_DIM,false,d,1,d_g,stream);   // rope at compressed pos g
    if(rotate){ hadamard(cand,cand,1,d,stream); act_quant_fp4sim(cand,1,d,32,d,stream); }
    else if(!g_kv_pack) act_quant_fp8sim(cand,1,d-ROPE_DIM,64,d,stream);
    // 1b.2: on the packed MAIN-KV path the quantise and the conditional commit are ONE kernel, and
    // it carries k_commit_comp's own predicate -- see kv_store_comp. The `rotate` branch is the
    // INDEXER cache, which is 1b.1's format and is untouched here.
    if(!rotate && g_kv_pack) kv_store_comp(comp_region,cand,d_T,d_pos,ratio,stream);
    else k_commit_comp<<<(d+255)/256,256,0,stream>>>(comp_region,cand,d_T,d_pos,ratio,d);   // commit at *d_T; caller advances
    dfree(scr);dfree(kv);dfree(score);dfree(pooled);dfree(cand);
}

// strided (ratio!=4) device-pos compressed decode step. kvc = combined cache; d_pos/d_T device.
void compressed_decode_step_strided_dp(float* out, const float* x, const float* xin, const CompressedAttnWeights& w,
        float* kvc, const int* d_pos, int* d_T, int* d_g, int winmax, int Tmax, int ratio, float eps, cudaStream_t stream){
    const auto& a=w.attn; const int Kd=N_HEADS*HEAD_DIM, GKd=Kd/O_GROUPS, OB=O_GROUPS*O_LORA; const float scale=1.f/sqrtf((float)HEAD_DIM);
    uint8_t *xq,*qrq,*ogq; float *xs,*qrs,*ogs,*qr,*q,*o,*og,*kvs; int* comb;
    xq=(uint8_t*)dmalloc(DIM); xs=(float*)dmalloc((DIM/128)*4);
    qr=(float*)dmalloc(Q_LORA*4); qrq=(uint8_t*)dmalloc(Q_LORA); qrs=(float*)dmalloc((Q_LORA/128)*4);
    q=(float*)dmalloc(Kd*4); o=(float*)dmalloc(Kd*4); og=(float*)dmalloc(OB*4); ogq=(uint8_t*)dmalloc(OB); ogs=(float*)dmalloc((OB/128)*4);
    kvs=(float*)dmalloc(HEAD_DIM*4); int wtop=WINDOW; comb=(int*)dmalloc((size_t)(wtop+Tmax)*4);
    act_quant_fp8(xq,xs,x,1,DIM,128,stream);
    fp8_block_gemm(qr,xq,xs,a.wq_a,a.wq_a_s,1,Q_LORA,DIM,stream); rmsnorm(qr,qr,a.q_norm,1,Q_LORA,eps,true,stream);
    act_quant_fp8(qrq,qrs,qr,1,Q_LORA,128,stream); fp8_block_gemm(q,qrq,qrs,a.wq_b,a.wq_b_s,1,Kd,Q_LORA,stream);
    rmsnorm(q,q,nullptr,N_HEADS,HEAD_DIM,eps,false,stream);
    rope_interleaved_dp(q+NOPE_DIM,a.cosT,a.sinT,N_HEADS,ROPE_DIM,false,HEAD_DIM,N_HEADS,d_pos,stream);
    // window kv -> kvc[*d_pos]
    fp8_block_gemm(kvs,xq,xs,a.wkv,a.wkv_s,1,HEAD_DIM,DIM,stream); rmsnorm(kvs,kvs,a.kv_norm,1,HEAD_DIM,eps,true,stream);
    rope_interleaved_dp(kvs+NOPE_DIM,a.cosT,a.sinT,1,ROPE_DIM,false,HEAD_DIM,1,d_pos,stream);
    if(g_kv_pack) kv_store_at(kvc,kvs,d_pos,stream);
    else { act_quant_fp8sim(kvs,1,NOPE_DIM,64,HEAD_DIM,stream);
           k_append_at2<<<(HEAD_DIM+255)/256,256,0,stream>>>(kvc,kvs,d_pos,HEAD_DIM); }
    dprobe(stream);
    // compressor emit (device-conditional) into the compressed region [winmax..]
    emit_group_dp(kv_row(kvc, winmax), xin, d_pos, d_T, d_g, ratio, w.mc_wkv,w.mc_wgate,w.mc_ape,w.mc_norm,w.cc_cos,w.cc_sin,DIM,HEAD_DIM,false,0,eps,stream);
    k_advance_T<<<1,1,0,stream>>>(d_T,d_pos,ratio); dprobe(stream);
    // attention over combined cache
    k_comb_strided_dp<<<(wtop+Tmax+63)/64,64,0,stream>>>(comb,d_pos,d_T,winmax,wtop,Tmax);
    sparse_attn_kv(o,q,kvc,a.attn_sink,comb,1,1,N_HEADS,HEAD_DIM,winmax+Tmax,wtop+Tmax,scale,stream);
    rope_interleaved_dp(o+NOPE_DIM,a.cosT,a.sinT,N_HEADS,ROPE_DIM,true,HEAD_DIM,N_HEADS,d_pos,stream); dprobe(stream);
    if(a.wo_a_native) ogroup_gemm_fp8(og,o,a.wo_a_fp8,a.wo_a_sc,1,O_GROUPS,O_LORA,GKd,stream);
    else              ogroup_gemm    (og,o,a.wo_a,               1,O_GROUPS,O_LORA,GKd,stream);
    act_quant_fp8(ogq,ogs,og,1,OB,128,stream); fp8_block_gemm(out,ogq,ogs,a.wo_b,a.wo_b_s,1,DIM,OB,stream);
    dsync(stream);
    dfree(xq);dfree(xs);dfree(qr);dfree(qrq);dfree(qrs);dfree(q);dfree(o);dfree(og);dfree(ogq);dfree(ogs);dfree(kvs);dfree(comb);
}

// device-pos indexer (ratio-4) compressed decode.
__global__ void k_mask_scores(float* score, const int* d_T, int Tmax){ int t=blockIdx.x*blockDim.x+threadIdx.x; if(t<Tmax && t>=*d_T) score[t]=-1e30f; }
__global__ void k_topk_masked(int* out, const float* score, int Tmax, int topk, int winmax){   // top-k valid rows -> winmax+t, else -1
    if(blockIdx.x) return; extern __shared__ float sh[]; const int L=threadIdx.x;
    for(int t=L;t<Tmax;t+=32) sh[t]=score[t];
    __syncwarp();
    for(int k=0;k<topk;++k){ float best=-1e29f; int bi=Tmax;      // Tmax == "nothing found"; see warp_argmax
        for(int t=L;t<Tmax;t+=32) if(sh[t]>best){best=sh[t];bi=t;}
        warp_argmax(best,bi);
        if(L==0){ if(bi<Tmax){ sh[bi]=-1e30f; out[k]=winmax+bi; } else out[k]=-1; }
        __syncwarp(); } }
__global__ void k_topk_masked_rx(int* out, const float* score, int Tmax, int topk, int winmax, bool early){
    if(blockIdx.x) return;
    __shared__ TopkRadixSmem S;
    topk_radix_select<TOPK_RADIX_NT>(out, score, Tmax, topk, -1e29f, S, early);   // NOTE the -1e29f floor, verbatim
    for(int k=threadIdx.x;k<topk;k+=TOPK_RADIX_NT){ int b=out[k]; out[k] = (b<0)? -1 : winmax+b; }
}
__global__ void k_comb_join(int* comb, const int* win, const int* sel, int wtop, int topk_c){   // [window ⊕ selected]
    int k=blockIdx.x*blockDim.x+threadIdx.x; int tot=wtop+topk_c; if(k>=tot) return; comb[k]=(k<wtop)?win[k]:sel[k-wtop]; }

void compressed_decode_step_indexer_dp(float* out, const float* x, const float* xin, const CompressedAttnWeights& w,
        float* kvc, float* idx_kvc, const int* d_pos, int* d_T, int* d_g, int winmax, int Tmax, int ratio, float eps, cudaStream_t stream){
    const auto& a=w.attn; const int Kd=N_HEADS*HEAD_DIM, GKd=Kd/O_GROUPS, OB=O_GROUPS*O_LORA;
    const int nH=w.index_n_heads, ihd=w.index_head_dim, QD=nH*ihd, rd=ROPE_DIM; const float scale=1.f/sqrtf((float)HEAD_DIM);
    const float wscale=rsqrtf((float)ihd)*rsqrtf((float)nH);
    uint8_t *xq,*qrq,*ogq; float *xs,*qrs,*ogs,*qr,*q,*o,*og,*kvs,*qidx,*qtmp,*iw,*isc; int *win,*sel,*comb;
    xq=(uint8_t*)dmalloc(DIM); xs=(float*)dmalloc((DIM/128)*4); qr=(float*)dmalloc(Q_LORA*4); qrq=(uint8_t*)dmalloc(Q_LORA); qrs=(float*)dmalloc((Q_LORA/128)*4);
    q=(float*)dmalloc(Kd*4); o=(float*)dmalloc(Kd*4); og=(float*)dmalloc(OB*4); ogq=(uint8_t*)dmalloc(OB); ogs=(float*)dmalloc((OB/128)*4); kvs=(float*)dmalloc(HEAD_DIM*4);
    qidx=(float*)dmalloc((size_t)QD*4); qtmp=(float*)dmalloc((size_t)QD*4); iw=(float*)dmalloc((size_t)nH*4); isc=(float*)dmalloc((size_t)Tmax*4);
    int wtop=WINDOW, topk_c=(w.index_topk<Tmax)?w.index_topk:Tmax;
    win=(int*)dmalloc((size_t)wtop*4); sel=(int*)dmalloc((size_t)topk_c*4); comb=(int*)dmalloc((size_t)(wtop+topk_c)*4);
    act_quant_fp8(xq,xs,x,1,DIM,128,stream);
    fp8_block_gemm(qr,xq,xs,a.wq_a,a.wq_a_s,1,Q_LORA,DIM,stream); rmsnorm(qr,qr,a.q_norm,1,Q_LORA,eps,true,stream);
    act_quant_fp8(qrq,qrs,qr,1,Q_LORA,128,stream); fp8_block_gemm(q,qrq,qrs,a.wq_b,a.wq_b_s,1,Kd,Q_LORA,stream);
    rmsnorm(q,q,nullptr,N_HEADS,HEAD_DIM,eps,false,stream); rope_interleaved_dp(q+NOPE_DIM,a.cosT,a.sinT,N_HEADS,ROPE_DIM,false,HEAD_DIM,N_HEADS,d_pos,stream);
    fp8_block_gemm(kvs,xq,xs,a.wkv,a.wkv_s,1,HEAD_DIM,DIM,stream); rmsnorm(kvs,kvs,a.kv_norm,1,HEAD_DIM,eps,true,stream);
    rope_interleaved_dp(kvs+NOPE_DIM,a.cosT,a.sinT,1,ROPE_DIM,false,HEAD_DIM,1,d_pos,stream);
    if(g_kv_pack) kv_store_at(kvc,kvs,d_pos,stream);
    else { act_quant_fp8sim(kvs,1,NOPE_DIM,64,HEAD_DIM,stream);
           k_append_at2<<<(HEAD_DIM+255)/256,256,0,stream>>>(kvc,kvs,d_pos,HEAD_DIM); }
    // main (overlap) + indexer (overlap+rotate) emits at the SAME *d_T, then advance once
    emit_group_dp(kv_row(kvc, winmax), xin, d_pos, d_T, d_g, ratio, w.mc_wkv,w.mc_wgate,w.mc_ape,w.mc_norm,w.cc_cos,w.cc_sin,DIM,HEAD_DIM,true,0,eps,stream);
    emit_group_dp(idx_kvc, xin, d_pos, d_T, d_g, ratio, w.idx_c_wkv,w.idx_c_wgate,w.idx_c_ape,w.idx_c_norm,w.cc_cos,w.cc_sin,DIM,ihd,true,1,eps,stream);
    k_advance_T<<<1,1,0,stream>>>(d_T,d_pos,ratio);
    // indexer scoring for the single query -> select main-compressed rows
    fp8_block_gemm(qidx,qrq,qrs,w.idx_wq_b,w.idx_wq_b_s,1,QD,Q_LORA,stream);   // qrq/qrs == act_quant(qr): same call, unmodified qr
    rope_interleaved_dp(qidx+(ihd-rd),a.cosT,a.sinT,nH,rd,false,ihd,nH,d_pos,stream); hadamard(qtmp,qidx,nH,ihd,stream); act_quant_fp4sim(qtmp,nH,ihd,32,ihd,stream);
    gemm_fp32(iw,x,w.idx_weights_proj,1,nH,DIM,stream); dprobe(stream); k_iw_scale<<<(nH+63)/64,64,0,stream>>>(iw,wscale,nH); dprobe(stream);
    index_score(isc,qtmp,idx_kvc,iw,1,Tmax,nH,ihd,stream); dprobe(stream); k_mask_scores<<<(Tmax+63)/64,64,0,stream>>>(isc,d_T,Tmax); dprobe(stream);
    if(topk_radix_on() && topk_c<=TOPK_RADIX_CAP) k_topk_masked_rx<<<1,TOPK_RADIX_NT,0,stream>>>(sel,isc,Tmax,topk_c,winmax,topk_early_on());
    else{ k_topk_masked<<<1,32,TOPK_SMEM(k_topk_masked,Tmax),stream>>>(sel,isc,Tmax,topk_c,winmax);
          TOPK_LAUNCHED(k_topk_masked, Tmax); }   // NOTE: sized by Tmax (=seqmax/ratio), not the live T
    dprobe(stream);
    k_comb_strided_dp<<<(wtop+63)/64,64,0,stream>>>(win,d_pos,d_T,winmax,wtop,0); dprobe(stream);   // window part only (Tmax=0)
    k_comb_join<<<(wtop+topk_c+63)/64,64,0,stream>>>(comb,win,sel,wtop,topk_c); dprobe(stream);
    sparse_attn_kv(o,q,kvc,a.attn_sink,comb,1,1,N_HEADS,HEAD_DIM,winmax+Tmax,wtop+topk_c,scale,stream); dprobe(stream);
    rope_interleaved_dp(o+NOPE_DIM,a.cosT,a.sinT,N_HEADS,ROPE_DIM,true,HEAD_DIM,N_HEADS,d_pos,stream);
    if(a.wo_a_native) ogroup_gemm_fp8(og,o,a.wo_a_fp8,a.wo_a_sc,1,O_GROUPS,O_LORA,GKd,stream);
    else              ogroup_gemm    (og,o,a.wo_a,               1,O_GROUPS,O_LORA,GKd,stream);
    act_quant_fp8(ogq,ogs,og,1,OB,128,stream); fp8_block_gemm(out,ogq,ogs,a.wo_b,a.wo_b_s,1,DIM,OB,stream);
    dsync(stream);
    dfree(xq);dfree(xs);dfree(qr);dfree(qrq);dfree(qrs);dfree(q);dfree(o);dfree(og);dfree(ogq);dfree(ogs);dfree(kvs);
    dfree(qidx);dfree(qtmp);dfree(iw);dfree(isc);dfree(win);dfree(sel);dfree(comb);
}
