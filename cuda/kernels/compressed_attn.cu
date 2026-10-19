// compressed_attn.cu — full compressed-layer MLA forward (prefill). See compressed_attn.h.
#include "compressed_attn.h"
#include "dprof.h"
#include "dscratch.h"
#include "fp8_block_gemm.h"
#include "mla_attn.h"
#include "compressor.h"
#include "indexer.h"
#include "deepseek_v4.h"
#include <vector>
#include <cmath>
#include <cstdio>
#define CU(x) do{cudaError_t e=(x); if(e){fprintf(stderr,"cuda %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)
// N1, ADOPTED (default ON; NO_ZERO_SCRATCH=1 disables). Finding 60: the engine is nondeterministic on identical input
// and DSV4_ARENA_ZERO exonerated the arena. This is the analogue the arena test could not reach —
// the PREFILL path allocates its scratch with raw cudaMalloc, once per layer per call, and never
// zeroes it, so every buffer starts life holding whatever the allocator last had at that address.
// Any kernel here that accumulates into its output rather than assigning it, or that writes fewer
// rows than it later reads, would inherit that and produce different numbers run to run.
//
// CLAIM RETRACTED (Finding 61). This was adopted on "zeroing moves distinct first-verify margin
// vectors from 8/8 to 5/8", read as evidence of an uninitialised read. tests/gate_scratch_init then
// tested the thing directly — same weights, same input, scratch filled with 0x00 vs 0xFF vs 0x3C,
// arena included — and compressed_attn_forward is bitwise IDENTICAL at every length 1..29. The
// prefill chain does not read uninitialised scratch AT THOSE LENGTHS, and that qualifier is
// load-bearing: ladder 1.9 measured this function's run-to-run reproducibility as a function of
// prefill length and found it byte-identical to 160 positions and NONDETERMINISTIC from 192 up, to
// 3071. The gate's range stops six times short of where the defect lives, so it is evidence about
// 1..29 and nothing else. 1.9 also re-ran the 128..256 ladder with DSV4_ARENA_ZERO=1 on both arms
// (unchanged verdict) and four times inside ONE process (four different answers), so the residual
// is a RACE in this file's call graph, not uninitialised memory. See DECODE_LADDER.md item 1.10 and
// wiki/measurement-and-traps.md §25. The 8/8 -> 5/8 was a 5-cycle in the engine
// (Finding 61) sampled 8 times, not a change caused by zeroing: with the sharper instrument the
// engine produces the SAME hash sequence zeroed and unzeroed.
//
// Kept ON anyway, and only because it is free: decode runs entirely out of the arena and never
// touches these allocations, so this costs the measured number nothing and removes a whole class of
// future doubt. It is NOT a fix for anything, and nothing should be attributed to it.
static inline cudaError_t zalloc(void** p, size_t n){
    static const bool z = getenv("NO_ZERO_SCRATCH") == nullptr;   // DEFAULT ON, see below
    cudaError_t e = cudaMalloc(p, n);
    if(e != cudaSuccess || !n) return e;
    const int idx = g_scratch_alloc_seq++;
    if(idx == 0) g_scratch_first_addr = (unsigned long long)*p;
    if(g_scratch_poison_idx == -1 || g_scratch_poison_idx == idx) cudaMemset(*p, g_scratch_poison_val, n);
    else if(z)                                                    cudaMemset(*p, 0, n);
    return e;
}

using namespace dsv4;

// combined_idxs[q, 0:wwidth] = window[q], combined[q, wwidth:] = compress[q]. one thread per (q,j).
__global__ void k_combine_idxs(int* comb, const int* window, const int* compress, int s, int wwidth, int itopk) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; int tot = wwidth + itopk; if (i >= s * tot) return;
    int q = i / tot, j = i % tot;
    comb[i] = (j < wwidth) ? window[(size_t)q * wwidth + j] : compress[(size_t)q * itopk + (j - wwidth)];
}

void compressed_attn_forward(float* out, const float* x, const CompressedAttnWeights& w,
                             int s, int win, int ratio, float eps, cudaStream_t stream) {
    const int bs = s, Kd = N_HEADS * HEAD_DIM, GKd = Kd / O_GROUPS, OB = O_GROUPS * O_LORA, T = s / ratio;
    const float scale = 1.f / sqrtf((float)HEAD_DIM);
    const auto& a = w.attn;

    uint8_t *xq, *qrq, *ogq; float *xs, *qrs, *ogs, *qr, *q, *kv_win, *kv_comp, *kv_all, *o, *og;
    CU(zalloc((void**)&xq,(size_t)bs*DIM)); CU(zalloc((void**)&xs,(size_t)bs*(DIM/128)*4));
    CU(zalloc((void**)&qr,(size_t)bs*Q_LORA*4)); CU(zalloc((void**)&qrq,(size_t)bs*Q_LORA)); CU(zalloc((void**)&qrs,(size_t)bs*(Q_LORA/128)*4));
    CU(zalloc((void**)&q,(size_t)bs*Kd*4)); CU(zalloc((void**)&kv_win,(size_t)bs*HEAD_DIM*4));
    CU(zalloc((void**)&kv_comp,(size_t)T*HEAD_DIM*4)); CU(zalloc((void**)&kv_all,(size_t)(bs+T)*HEAD_DIM*4));
    CU(zalloc((void**)&o,(size_t)bs*Kd*4)); CU(zalloc((void**)&og,(size_t)bs*OB*4));
    CU(zalloc((void**)&ogq,(size_t)bs*OB)); CU(zalloc((void**)&ogs,(size_t)bs*(OB/128)*4));

    // B9 marks. compressed_attn_forward was 6.9 s = 32% of a 1022-token prefill and the single
    // largest unattributed block in the engine, because it had no sub-marks at any level.
    // --- q ---
    dprof_begin(DP_C_QPROJ,stream);
    act_quant_fp8(xq, xs, x, bs, DIM, 128, stream); dprobe(stream);
    fp8_block_gemm(qr, xq, xs, a.wq_a, a.wq_a_s, bs, Q_LORA, DIM, stream); dprobe(stream);
    rmsnorm(qr, qr, a.q_norm, bs, Q_LORA, eps, true, stream);
    act_quant_fp8(qrq, qrs, qr, bs, Q_LORA, 128, stream);
    fp8_block_gemm(q, qrq, qrs, a.wq_b, a.wq_b_s, bs, Kd, Q_LORA, stream); dprobe(stream);
    rmsnorm(q, q, nullptr, bs * N_HEADS, HEAD_DIM, eps, false, stream);
    rope_interleaved(q + NOPE_DIM, a.cosT, a.sinT, bs * N_HEADS, ROPE_DIM, false, HEAD_DIM, N_HEADS, stream); dprobe(stream);
    dprof_end(DP_C_QPROJ,stream);

    // --- window kv ---
    dprof_begin(DP_A_KV,stream);
    fp8_block_gemm(kv_win, xq, xs, a.wkv, a.wkv_s, bs, HEAD_DIM, DIM, stream); dprobe(stream);
    rmsnorm(kv_win, kv_win, a.kv_norm, bs, HEAD_DIM, eps, true, stream);
    rope_interleaved(kv_win + NOPE_DIM, a.cosT, a.sinT, bs, ROPE_DIM, false, HEAD_DIM, 1, stream);
    act_quant_fp8sim(kv_win, bs, NOPE_DIM, 64, HEAD_DIM, stream); dprobe(stream);
    dprof_end(DP_A_KV,stream);

    // --- main compressor -> compressed kv, then combined kv = [window ⊕ compressed] ---
    // ratio==4: overlapping compressor + DSA indexer. ratio==128: non-overlap compressor + strided idxs.
    const bool overlap = (ratio == 4), has_indexer = (ratio == 4);
    dprof_begin(DP_C_INDEXER,stream);
    compressor_forward(kv_comp, x, w.mc_wkv, w.mc_wgate, w.mc_ape, w.mc_norm, w.cc_cos, w.cc_sin,
                       s, DIM, HEAD_DIM, ratio, overlap, ROPE_DIM, eps, false, stream); dprobe(stream);
    CU(cudaMemcpyAsync(kv_all, kv_win, (size_t)bs*HEAD_DIM*4, cudaMemcpyDeviceToDevice, stream));
    CU(cudaMemcpyAsync(kv_all + (size_t)bs*HEAD_DIM, kv_comp, (size_t)T*HEAD_DIM*4, cudaMemcpyDeviceToDevice, stream));

    // --- compressed idxs (offset = s, into the compressed region) ---
    int itopk; int* compress_topk;
    // ---- INDEXER BYPASS (DSV4_IDX_BYPASS=1). NOT BIT-EXACT, and mathematically exact. ----
    // `itopk = min(index_topk, T)`. index_topk is 512 and T = ceil(s/ratio), so for ratio 4 the
    // selection saturates only above ctx 2048 (ratio 128: above 65536). Below that the top-k selects
    // EVERY compressed row: k_topk_offset removes each picked row from contention, so over T slots
    // every one of the T rows is picked exactly once, and the causal mask is applied at the OUTPUT.
    // The emitted SET is therefore identical to the strided construction below -- provably, not
    // approximately. What differs is ORDER: the indexer emits descending-score, strided emits
    // ascending-position, and sparse attention's online softmax accumulates in fp32, so the sum is
    // reassociated and the last bits move. Same rows, same mathematics, different rounding.
    //
    // So at these lengths the indexer computes an s x T score matrix and a full sort to produce an
    // ordering that cannot change which rows are attended to. Bypassing it is worth the whole
    // `cattn:indexer` mark. It is OFF by default because this project's gate is bit-exactness, not
    // mathematical equivalence, and reassociating a reduction is exactly what that gate forbids.
    const bool idx_identity = has_indexer && (w.index_topk >= T) && getenv("DSV4_IDX_BYPASS")!=nullptr;
    if (has_indexer && !idx_identity) {
        itopk = w.index_topk < T ? w.index_topk : T;
        float* idx_score; CU(zalloc((void**)&idx_score,(size_t)s*T*4)); CU(zalloc((void**)&compress_topk,(size_t)s*itopk*4));
        indexer_forward(idx_score, compress_topk, x, qr, w.idx_wq_b, w.idx_wq_b_s, w.idx_weights_proj,
                        w.idx_c_wkv, w.idx_c_wgate, w.idx_c_ape, w.idx_c_norm, a.cosT, a.sinT, w.cc_cos, w.cc_sin,
                        s, DIM, Q_LORA, w.index_n_heads, w.index_head_dim, ROPE_DIM, ratio, w.index_topk, s, eps, stream);
        cudaFree(idx_score);
    } else {
        // Reached by ratio-128 (no indexer by construction) and, under DSV4_IDX_BYPASS, by ratio-4
        // whenever the top-k is the identity. Same code, same output shape, in both cases.
        // strided (get_compress_topk_idxs, prefill): compress[i,t] = (t >= (i+1)/ratio) ? -1 : t + s
        itopk = T; std::vector<int> hc((size_t)s * T);
        for (int i = 0; i < s; ++i) { int thr = (i + 1) / ratio;
            for (int t = 0; t < T; ++t) hc[(size_t)i * T + t] = (t >= thr) ? -1 : t + s; }
        CU(zalloc((void**)&compress_topk,(size_t)s*T*4));
        CU(cudaMemcpyAsync(compress_topk, hc.data(), (size_t)s*T*4, cudaMemcpyHostToDevice, stream));
    }

    dprof_end(DP_C_INDEXER,stream);

    // --- window idxs (host) ⊕ compressed idxs -> combined ---
    dprof_begin(DP_C_SPARSE,stream);
    int wwidth = s < win ? s : win;
    std::vector<int> hw((size_t)s * wwidth);
    for (int i = 0; i < s; ++i) { int base = i - win + 1; if (base < 0) base = 0;
        for (int k = 0; k < wwidth; ++k) { int v = base + k; hw[(size_t)i * wwidth + k] = (v > i) ? -1 : v; } }
    int* window_dev; CU(zalloc((void**)&window_dev,(size_t)s*wwidth*4));
    CU(cudaMemcpyAsync(window_dev, hw.data(), (size_t)s*wwidth*4, cudaMemcpyHostToDevice, stream));
    int tot = wwidth + itopk; int* combined; CU(zalloc((void**)&combined,(size_t)s*tot*4));
    k_combine_idxs<<<(s*tot+255)/256,256,0,stream>>>(combined, window_dev, compress_topk, s, wwidth, itopk); dprobe(stream);

    // --- sparse attention over combined KV, then de-rotate + grouped o-LoRA + wo_b ---
    sparse_attn(o, q, kv_all, a.attn_sink, combined, 1, s, N_HEADS, HEAD_DIM, s + T, tot, scale, stream); dprobe(stream);
    dprof_end(DP_C_SPARSE,stream);
    dprof_begin(DP_C_OGROUP,stream);
    rope_interleaved(o + NOPE_DIM, a.cosT, a.sinT, bs * N_HEADS, ROPE_DIM, true, HEAD_DIM, N_HEADS, stream);
    if(a.wo_a_native) ogroup_gemm_fp8(og, o, a.wo_a_fp8, a.wo_a_sc, bs, O_GROUPS, O_LORA, GKd, stream);
    else              ogroup_gemm    (og, o, a.wo_a,                bs, O_GROUPS, O_LORA, GKd, stream);
    act_quant_fp8(ogq, ogs, og, bs, OB, 128, stream);
    fp8_block_gemm(out, ogq, ogs, a.wo_b, a.wo_b_s, bs, DIM, OB, stream); dprobe(stream);
    dprof_end(DP_C_OGROUP,stream);

    CU(cudaStreamSynchronize(stream));
    cudaFree(xq);cudaFree(xs);cudaFree(qr);cudaFree(qrq);cudaFree(qrs);cudaFree(q);cudaFree(kv_win);
    cudaFree(kv_comp);cudaFree(kv_all);cudaFree(o);cudaFree(og);cudaFree(ogq);cudaFree(ogs);
    cudaFree(compress_topk);cudaFree(window_dev);cudaFree(combined);
}
