// dspark_attn.cu — DSparkAttention. See dspark_attn.h / DSPARK_HEAD_BUILD.md piece 4.
#include "dspark_attn.h"
#include "fp8_block_gemm.h"
#include "mla_attn.h"
#include "kv_pack.h"
#include "deepseek_v4.h"
#include <vector>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "dscratch.h"
#define CU(x) do{cudaError_t e=(x); if(e){fprintf(stderr,"cuda %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)
using namespace dsv4;

// main-KV from main_x: wkv -> kv_norm -> rope(per-position) -> act_quant fp8sim(nope). [s,dim]->[s,HEAD_DIM].
void dspark_main_kv(float* main_kv, const float* main_x, const MLAWeights& w, int s, float eps, cudaStream_t stream){
    uint8_t* xq; float* xs;
    CU(dkmalloc(&xq,(size_t)s*DIM)); CU(dkmalloc(&xs,(size_t)s*(DIM/128)*4));
    act_quant_fp8(xq, xs, main_x, s, DIM, 128, stream);
    float* mst = kv_stage(main_kv, s);          // == main_kv when the cache is FP32 (1b.2)
    fp8_block_gemm(mst, xq, xs, w.wkv, w.wkv_s, s, HEAD_DIM, DIM, stream);
    rmsnorm(mst, mst, w.kv_norm, s, HEAD_DIM, eps, true, stream);
    rope_interleaved(mst + NOPE_DIM, w.cosT, w.sinT, s, ROPE_DIM, false, HEAD_DIM, 1, stream);
    kv_commit(main_kv, mst, s, stream); kv_stage_free(mst, main_kv);
    dksync(stream); dkfree(xq); dkfree(xs);
}

// ================= LADDER 1.0 — the main-KV prefix is CACHED, not recomputed =================
//
// `dspark_main_kv` above rebuilds all `s` rows from scratch, and src/engine.cu's decode loop calls
// it NSTAGE=3 times per token over the full context. 0.4 measured that at 3.867 +/- 0.001 ms per
// 1000 context, R^2 1.000 -- 47.87 ms at ctx 12,288, the largest single row in the step and 55 % of
// the whole context term. It is O(context) work per token to recompute a quantity that does not
// change.
//
// WHY THE PREFIX IS IMMUTABLE. `main_x` is written exactly once per position: the prefill writes
// [0,n) and the decode loop writes [cpos, cpos+acc+1) for the COMMITTED tokens only, then advances
// cpos past them. Rejected drafts never reach it. So for any row r < cpos, `main_x[r]` is frozen.
//
// WHY THE RESULT IS BYTE-IDENTICAL AND NOT MERELY CLOSE. Every stage is row-independent:
//   act_quant_fp8   one block per (row, 128-wide group); reduction is within the group.
//   the wkv GEMM    each output element reduces over K in a fixed order (see the pin below).
//   rmsnorm         one block per row.
//   rope            per position; `crow = row / cos_stride_rows` with cos_stride_rows = 1, so the
//                   sub-range must be handed cosT/sinT offset by r0 -- that is the one place a
//                   naive split would silently rotate every row by the wrong angle.
//   act_quant_fp8sim one block per (row, 64-wide group).
// So computing rows [r0,s) into main_kv + r0*HEAD_DIM and keeping [0,r0) yields the same bytes.
//
// THE GEMM PIN IS LOAD-BEARING. `fp8_block_gemm`'s dispatch is M-DEPENDENT: M==1 takes a GEMV,
// M in [2,8] can take an NVFP4 overlay or the templated small-M GEMV, and only larger M reaches
// tc_fp8_gemm. The delta here is usually 1-3 rows while the from-scratch call was thousands, so
// going through that dispatch would compare a GEMV against a tensor-core tile and lose
// bit-exactness for a reason that has nothing to do with the caching. tc_fp8_gemm's own dispatch
// depends only on N and K (W from N, NS from the carveout, KC from tcb_pick_kc(W, K/128)); M only
// sets grid.y, and each output element accumulates over K in the same order regardless of which
// 16-row tile it lands in. So pinning to it makes the result independent of the split point. When
// g_tc_fp8 is off (unit gates, the `forward` binary) the fallback oracle is warp-per-output and
// equally M-independent.
extern bool g_tc_fp8;
void tc_fp8_gemm(float*, const uint8_t*, const float*, const uint8_t*, const float*, int, int, int, cudaStream_t);
static void main_kv_gemm(float* C, const uint8_t* A, const float* as, const uint8_t* B, const float* bs,
                         int M, int N, int K, cudaStream_t s){
    if(g_tc_fp8 && (N % 8 == 0) && (K % 128 == 0)){ tc_fp8_gemm(C, A, as, B, bs, M, N, K, s); return; }
    fp8_block_gemm(C, A, as, B, bs, M, N, K, s);
}

// main-KV for rows [r0, s) only. Scratch is sized by the DELTA, which also retires the per-step
// dkmalloc of s*DIM + s*(DIM/128)*4 -- 51.8 MB per stage at ctx 12,288, i.e. 155 MB bumped off a
// 640 MB arena before the step's first arena_reset(). At a 1-3 row delta it is ~12 KB.
static void main_kv_rows(float* main_kv, const float* main_x, const MLAWeights& w,
                         int r0, int s, float eps, cudaStream_t stream){
    const int m = s - r0;
    if(m <= 0) return;
    const int half = ROPE_DIM / 2;
    uint8_t* xq; float* xs;
    CU(dkmalloc(&xq,(size_t)m*DIM)); CU(dkmalloc(&xs,(size_t)m*(DIM/128)*4));
    float* krow = kv_row(main_kv, r0);
    float* kv = kv_stage(krow, m);
    act_quant_fp8(xq, xs, main_x + (size_t)r0*DIM, m, DIM, 128, stream);
    main_kv_gemm(kv, xq, xs, w.wkv, w.wkv_s, m, HEAD_DIM, DIM, stream);
    rmsnorm(kv, kv, w.kv_norm, m, HEAD_DIM, eps, true, stream);
    rope_interleaved(kv + NOPE_DIM, w.cosT + (size_t)r0*half, w.sinT + (size_t)r0*half,
                     m, ROPE_DIM, false, HEAD_DIM, 1, stream);
    kv_commit(krow, kv, m, stream); kv_stage_free(kv, krow);
    dksync(stream); dkfree(xq); dkfree(xs);
}

// DSV4_MAINKV_GATE=1 — the bit-exactness gate, run in situ on the shipped path at real context.
// Recomputes all `s` rows with the UNTOUCHED from-scratch function into a private buffer and
// memcmps the whole [s, HEAD_DIM] range. That covers both halves of the risk in one comparison:
// the row split AND the GEMM pin, since the reference goes through the original fp8_block_gemm
// dispatch. It aborts on the first differing float rather than counting them -- a bit-exactness
// gate that reports and continues is a gate that gets ignored.
//
// The reference recompute is forced onto the RAW allocator: it needs s*DIM bytes of scratch
// (50.3 MB at ctx 12,288) and the arena it would otherwise bump has live allocations and no reset
// in between -- three of those would be 155 MB on top of whatever the last draft pass left.
static void mainkv_gate_check(const float* main_kv, const float* main_x, const MLAWeights& w,
                              int s, int r0, float eps, cudaStream_t stream){
    extern bool g_draft_raw;
    static long long checks = 0, rows_kept = 0;
    // Buffers GROW and are never freed. The gate runs with ~2.8 GiB of headroom on a box that does
    // not OOM gracefully, and a 25 MB cudaMalloc/cudaFree pair three times per token is exactly the
    // churn worth not introducing there.
    static float* ref = nullptr; static size_t refcap = 0;
    static std::vector<float> a, b;
    // 1b.2: `n` is the cache in FLOAT-STRIDE units, which is 512/row unpacked and 180/row packed.
    // The memcmp is over whatever the cache actually holds, so the gate is layout-agnostic and
    // proves the PACKED bytes identical when the packed path is on.
    const size_t n = (size_t)s*(size_t)g_kv_rowf;
    if(n > refcap){ if(ref) cudaFree(ref); CU(cudaMalloc(&ref, n*4)); refcap = n; }
    if(a.size() < n){ a.resize(n); b.resize(n); }
    const bool saved = g_draft_raw; g_draft_raw = true;
    dspark_main_kv(ref, main_x, w, s, eps, stream);
    g_draft_raw = saved;
    CU(cudaStreamSynchronize(stream));
    CU(cudaMemcpy(a.data(), main_kv, n*4, cudaMemcpyDeviceToHost));
    CU(cudaMemcpy(b.data(), ref,     n*4, cudaMemcpyDeviceToHost));
    ++checks; rows_kept += r0;
    if(memcmp(a.data(), b.data(), n*4) != 0){
        size_t i = 0; while(i < n && memcmp(&a[i], &b[i], 4) == 0) ++i;
        fprintf(stderr, "[mainkv-gate] FAIL s=%d kept=%d first diff at row %zu col %zu: %.9g vs %.9g\n",
                s, r0, i/(size_t)g_kv_rowf, i%(size_t)g_kv_rowf, a[i], b[i]);
        fflush(stderr); abort();
    }
    if(checks <= 3 || checks % 64 == 0){
        fprintf(stderr, "[mainkv-gate] PASS s=%d kept=%d recomputed=%d (%lld checks, %lld rows kept)\n",
                s, r0, s-r0, checks, rows_kept); fflush(stderr);
    }
}

void dspark_main_kv_upto(float* main_kv, const float* main_x, const MLAWeights& w, int s, int* valid,
                         float eps, cudaStream_t stream){
    static const int gate = getenv("DSV4_MAINKV_GATE") ? atoi(getenv("DSV4_MAINKV_GATE")) : 0;
    int r0 = *valid;
    if(r0 < 0) r0 = 0;
    if(r0 > s) r0 = s;              // caller shrank the context without telling us; recompute nothing
    main_kv_rows(main_kv, main_x, w, r0, s, eps, stream);
    *valid = s;
    if(gate) mainkv_gate_check(main_kv, main_x, w, s, r0, eps, stream);
}

// F103 localised the port's 18557x divergence to dspark_block_forward. These stage dumps take it
// one level deeper: whichever sub-stage first disagrees is the bug. Written only when decode.cu
// points g_dspark_dump at a file, and only for the block it wants -- otherwise a no-op branch.
FILE* g_dspark_dump = nullptr;
static void dsp_dump(const char* tag, const float* dev, size_t n){
    if(!g_dspark_dump) return;
    std::vector<float> hv(n);
    if(cudaMemcpy(hv.data(),dev,n*4,cudaMemcpyDeviceToHost)!=cudaSuccess) return;
    uint32_t L=(uint32_t)strlen(tag), N=(uint32_t)n;
    fwrite(&L,4,1,g_dspark_dump); fwrite(tag,1,L,g_dspark_dump);
    fwrite(&N,4,1,g_dspark_dump); fwrite(hv.data(),4,n,g_dspark_dump);
}
void dspark_attn_forward(float* out, const float* xin, const float* main_kv, int t,
                         const MLAWeights& w, const float* cosB, const float* sinB,
                         int block, int win, float eps, cudaStream_t stream){
    const int Kd = N_HEADS*HEAD_DIM, GKd = Kd/O_GROUPS, OB = O_GROUPS*O_LORA;
    const float scale = 1.f/sqrtf((float)HEAD_DIM);
    int nwin = (t+1 < win) ? t+1 : win; int wstart = t+1-nwin; int n = nwin + block;

    uint8_t *xq,*qrq,*ogq; float *xs,*qrs,*ogs,*qr,*q,*bkv,*kv_all,*o,*og;
    CU(dkmalloc(&xq,(size_t)block*DIM)); CU(dkmalloc(&xs,(size_t)block*(DIM/128)*4));
    CU(dkmalloc(&qr,(size_t)block*Q_LORA*4)); CU(dkmalloc(&qrq,(size_t)block*Q_LORA)); CU(dkmalloc(&qrs,(size_t)block*(Q_LORA/128)*4));
    CU(dkmalloc(&q,(size_t)block*Kd*4)); CU(dkmalloc(&bkv,kv_rows_bytes(block)));
    CU(dkmalloc(&kv_all,kv_rows_bytes(n))); CU(dkmalloc(&o,(size_t)block*Kd*4)); CU(dkmalloc(&og,(size_t)block*OB*4));
    CU(dkmalloc(&ogq,(size_t)block*OB)); CU(dkmalloc(&ogs,(size_t)block*(OB/128)*4));

    // q
    act_quant_fp8(xq, xs, xin, block, DIM, 128, stream);
    fp8_block_gemm(qr, xq, xs, w.wq_a, w.wq_a_s, block, Q_LORA, DIM, stream);
    rmsnorm(qr, qr, w.q_norm, block, Q_LORA, eps, true, stream);
    act_quant_fp8(qrq, qrs, qr, block, Q_LORA, 128, stream);
    fp8_block_gemm(q, qrq, qrs, w.wq_b, w.wq_b_s, block, Kd, Q_LORA, stream);
    rmsnorm(q, q, nullptr, block*N_HEADS, HEAD_DIM, eps, false, stream);
    rope_interleaved(q + NOPE_DIM, cosB, sinB, block*N_HEADS, ROPE_DIM, false, HEAD_DIM, N_HEADS, stream);
    // block kv
    // 1b.2. `bkv` is a temporary, but it is concatenated into `kv_all` which `sparse_attn_kv`
    // reads, so it has to be in the SAME layout as the main-KV window it sits next to. Staged in
    // fp32 and committed into a packed block region.
    float* bkvp = bkv;
    float* bst = kv_stage(bkvp, block);
    fp8_block_gemm(bst, xq, xs, w.wkv, w.wkv_s, block, HEAD_DIM, DIM, stream);
    rmsnorm(bst, bst, w.kv_norm, block, HEAD_DIM, eps, true, stream);
    rope_interleaved(bst + NOPE_DIM, cosB, sinB, block, ROPE_DIM, false, HEAD_DIM, 1, stream);
    kv_commit(bkvp, bst, block, stream); kv_stage_free(bst, bkvp);
    if(g_dspark_dump){ cudaStreamSynchronize(stream);
        // The SHAPE is the hypothesis: nwin = min(t+1, win) context keys plus `block` block-keys.
        // If the port attends to a different COUNT the output magnitude changes by orders of
        // magnitude, which is exactly the 30161x rel_err F104 measured -- a wrong-count bug
        // rescales, a wrong-keys bug only rotates.
        float shp[4] = {(float)t,(float)nwin,(float)wstart,(float)n};
        uint32_t L=8,N=4; fwrite(&L,4,1,g_dspark_dump); fwrite("a_shape",1,7,g_dspark_dump);
        fwrite("",1,1,g_dspark_dump); fwrite(&N,4,1,g_dspark_dump); fwrite(shp,4,4,g_dspark_dump);
        dsp_dump("a_q", q, (size_t)block*Kd);
        if(!g_kv_pack) dsp_dump("a_bkv", bkv, (size_t)block*HEAD_DIM); }
    // kv = [main-KV window ⊕ block-KV]
    CU(cudaMemcpyAsync(kv_all, kv_row(main_kv, wstart), kv_rows_bytes(nwin), cudaMemcpyDeviceToDevice, stream));
    CU(cudaMemcpyAsync(kv_row(kv_all, nwin), bkv, kv_rows_bytes(block), cudaMemcpyDeviceToDevice, stream));
    if(g_dspark_dump && !g_kv_pack){ cudaStreamSynchronize(stream); dsp_dump("a_kv_all", kv_all, (size_t)n*HEAD_DIM); }
    // dense idxs [block, n]: every block query attends to all n (window ⊕ block), per get_dspark_topk_idxs
    std::vector<int> hidx((size_t)block*n); for(int m=0;m<block;++m) for(int k=0;k<n;++k) hidx[(size_t)m*n+k]=k;
    int* idx; CU(dkmalloc(&idx,(size_t)block*n*4)); CU(cudaMemcpyAsync(idx,hidx.data(),(size_t)block*n*4,cudaMemcpyHostToDevice,stream));
    sparse_attn_kv(o, q, kv_all, w.attn_sink, idx, 1, block, N_HEADS, HEAD_DIM, n, n, scale, stream);
    rope_interleaved(o + NOPE_DIM, cosB, sinB, block*N_HEADS, ROPE_DIM, true, HEAD_DIM, N_HEADS, stream);
    ogroup_gemm(og, o, w.wo_a, block, O_GROUPS, O_LORA, GKd, stream);
    act_quant_fp8(ogq, ogs, og, block, OB, 128, stream);
    fp8_block_gemm(out, ogq, ogs, w.wo_b, w.wo_b_s, block, DIM, OB, stream);
    dksync(stream);
    dkfree(xq);dkfree(xs);dkfree(qr);dkfree(qrq);dkfree(qrs);dkfree(q);dkfree(bkv);
    dkfree(kv_all);dkfree(o);dkfree(og);dkfree(ogq);dkfree(ogs);dkfree(idx);
}

// ---- DSparkBlock forward (block_forward with dspark_attn) ----
#include "hc.h"
#include "moe.h"
void dspark_block_forward(float* out, const float* x, const int* input_ids, const float* main_kv, int t,
                          const BlockWeights& w, const float* cosB, const float* sinB, int block, int win,
                          int iters, float eps, cudaStream_t stream){
    const int d=w.dim, hc=w.hc;
    float *x1,*post,*comb,*sub,*res2;
    CU(dkmalloc(&x1,(size_t)block*d*4)); CU(dkmalloc(&post,(size_t)block*hc*4)); CU(dkmalloc(&comb,(size_t)block*hc*hc*4));
    CU(dkmalloc(&sub,(size_t)block*d*4)); CU(dkmalloc(&res2,(size_t)block*hc*d*4));
    hc_pre(x1, post, comb, x, w.hc_attn_fn, w.hc_attn_scale, w.hc_attn_base, block, hc, d, iters, eps, stream);
    if(g_dspark_dump){ cudaStreamSynchronize(stream);
        dsp_dump("b_hcpre_attn_x1", x1, (size_t)block*d);
        dsp_dump("b_hcpre_attn_post", post, (size_t)block*hc);
        dsp_dump("b_hcpre_attn_comb", comb, (size_t)block*hc*hc); }
    rmsnorm(x1, x1, w.attn_norm, block, d, eps, true, stream);
    if(g_dspark_dump){ cudaStreamSynchronize(stream); dsp_dump("b_rmsnorm_attn", x1, (size_t)block*d); }
    dspark_attn_forward(sub, x1, main_kv, t, w.attn, cosB, sinB, block, win, eps, stream);
    if(g_dspark_dump){ cudaStreamSynchronize(stream); dsp_dump("b_attn_out", sub, (size_t)block*d); }
    hc_post(res2, sub, x, post, comb, block, hc, d, stream);
    if(g_dspark_dump){ cudaStreamSynchronize(stream); dsp_dump("b_hcpost_attn", res2, (size_t)block*hc*d); }
    hc_pre(x1, post, comb, res2, w.hc_ffn_fn, w.hc_ffn_scale, w.hc_ffn_base, block, hc, d, iters, eps, stream);
    rmsnorm(x1, x1, w.ffn_norm, block, d, eps, true, stream);
    if(g_dspark_dump){ cudaStreamSynchronize(stream); dsp_dump("b_rmsnorm_ffn", x1, (size_t)block*d); }
    moe_forward(sub, x1, input_ids, w.ffn, block, stream);
    if(g_dspark_dump){ cudaStreamSynchronize(stream); dsp_dump("b_moe_out", sub, (size_t)block*d); }
    hc_post(out, sub, res2, post, comb, block, hc, d, stream);
    dksync(stream);
    dkfree(x1);dkfree(post);dkfree(comb);dkfree(sub);dkfree(res2);
}
