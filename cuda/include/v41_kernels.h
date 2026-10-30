// v41_kernels.h — the V4.1 kernel set (kernels/v41/*.cu).
//
// Conventions, uniform across the file:
//   * activations are fp32 buffers holding bf16-representable values wherever the reference holds
//     a bf16 tensor; `round_bf16` arguments say whether a kernel rounds its own output.
//   * M = rows (tokens) in this call. Decode is M = 1, a spec verify is M = K+1, a prefill chunk is
//     M <= 128. Per-row positions come from DEVICE memory (`d_pos[M]`), so every kernel that depends
//     on the position or on the context length can sit inside a captured CUDA graph.
//   * deterministic: no float atomics anywhere, so graph replay and eager runs are bit-identical.
//   * fp8 linears: W [N,K] e4m3, Ws [ceil(N/32), ceil(K/32)] e8m0; activations quantised per 32
//     along K with pow2 (ue8m0) scales, exactly the reference `act_quant(x, 32, "ue8m0")`.
#pragma once
#include <cstdint>
#include <cuda_runtime.h>

namespace v41k {

// One-time setup, OUTSIDE any graph capture: split-K partial buffer (floats) and HC partials for
// up to max_rows tokens per call.
void v41k_init(size_t part_floats);
// a stream that runs k_dense concurrently with another gets its own split-K partials (after v41k_init)
void v41k_stream_scratch(cudaStream_t st);
void rope_set_tables();
void hc_reserve(int max_rows);

// ------------------------------------------------------------------ dense
// x[M,K] -> q[M,K] e4m3 bytes + s[M,K/32] f32 pow2 scales. amax clamped at 1e-4 (kernel.py).
void act_quant_fp8(uint8_t* q, float* s, const float* x, int M, int K, cudaStream_t st);
// y[M, G*Ng] = for each group g: xq[M, g*Kg : (g+1)*Kg] @ W[g*Ng:(g+1)*Ng, Kg]^T. G = 1 for a plain
// linear; G = 8 is wo_a's block-diagonal einsum. Each warp owns R=8 output rows and reuses its
// activation registers across them (activation-traffic finding: x is 4 B per weight otherwise).
void fp8_linear(float* y, const uint8_t* xq, const float* xs, const uint8_t* W, const uint8_t* Ws,
                int M, int N, int K, bool round_bf16, cudaStream_t st, int G = 1);
// y[M,N] = x[M,K] @ W[N,K]^T with W bf16, fp32 accumulate. Covers the bf16 linears (compressor,
// indexer wk/weights_proj, router, head) — the reference's fp32-promoted ones are numerically the
// same product because their weights are bf16 values.
// Reorder W in place for coalesced loads (DecBF16P) and route bf16_linear(W) through it. Exact.
bool bf16_permute(uint16_t* W, int N, int K, cudaStream_t st);
bool bf16_permuted(const void* W);   // W was registered by bf16_permute
// NVFP4 copy (DecNVFP4 layout: W N*K/2 codes, S N*K/16 e4m3 scales) of a bf16 [N,K] matrix, permuted = it went
// through bf16_permute; returns the global scale g. nvfp4_linear_f32: y[M,N] = x[M,K] (fp32) . W^T.
float nvfp4_from_bf16(uint8_t* W, uint8_t* S, const uint16_t* src, int N, int K, bool permuted, cudaStream_t st);
// the same from an FP8 [N, K] weight with e8m0 per 32x32 (Ws); returns g
float nvfp4_from_fp8(uint8_t* W, uint8_t* S, const uint8_t* src, const uint8_t* srcs, int N, int K, cudaStream_t st);
void nvfp4_linear_f32(float* y, const float* x, const uint8_t* W, const uint8_t* S, float g, int M, int N, int K, cudaStream_t st);
// Two-stage greedy head: NVFP4 scores, then the bf16 head's exact logit (bit-identical to bf16_linear,
// M = 1 or any M) for every row within delta of the approximate max; argmax of row r -> out[r * ostride].
// fb[r] = 1 when more than cap rows qualified (out[r] is then not the argmax: recompute with the full head).
struct H2Scr {
    float* alog = nullptr; int* hk = nullptr; int* cnt = nullptr; int* list = nullptr; float* resc = nullptr;
    int cap = 0; float delta = 0.f;
    bool alloc(int rows, int N, int cap, float delta);
};
void head2_argmax(int* out, int ostride, int* fb, const float* x, int n, const uint16_t* head, const uint8_t* h4,
                  const uint8_t* h4s, float h4g, H2Scr& sc, int N, int K, cudaStream_t st);
void bf16_linear(float* y, const float* x, const uint16_t* W, int M, int N, int K,
                 bool round_bf16, cudaStream_t st, float out_scale = 1.0f);
// y = bf16(w * x * rsqrt(mean(x^2) + eps)); w bf16 or null. rows independent.
void rmsnorm(float* y, const float* x, const uint16_t* w, int rows, int dim, float eps, cudaStream_t st);
// interleaved-pair RoPE on x[rows][off .. off+64) (stride `row_stride`), `heads` rows per position.
// position of row r = d_pos[r / heads] + pos_delta. inverse => conj. Result rounded to bf16.
// yarn=false: rope_theta 10000, no YaRN (ratio-0 layers); true: compress_rope_theta + YaRN x16.
void rope(float* x, int rows, int heads, int row_stride, int off, const int* d_pos, int pos_delta,
          bool inverse, bool yarn, cudaStream_t st);
// in-place fp8 QAT-sim per 32 over `dim` of each row: q = e4m3(x/s)*s, s = pow2ceil(max(amax,1e-4)/448).
void fp8_sim(float* x, int rows, int dim, cudaStream_t st);
// in-place fp4 QAT-sim. e4m3_scale=false: E8M0 pow2 scale per `grp` (indexer, grp 32);
// true: E4M3 scale per `grp` (compressed KV, grp 16). Optionally emits the packed form
// (codes [rows, dim/2], low nibble first; scale bytes [rows, dim/grp]) for the cache.
void fp4_sim(float* x, int rows, int dim, int grp, bool e4m3_scale, uint8_t* packed, uint8_t* scales,
             cudaStream_t st);
void round_bf16(float* x, size_t n, cudaStream_t st);
// x = bf16(x * c)   (a bf16 tensor times a python float, as in Indexer: weights_proj(x) * scale)
void scale_bf16(float* x, size_t n, float c, cudaStream_t st);
// *out = argmax (first index on ties, as torch.argmax)
void argmax(int* out, const float* x, int n, cudaStream_t st);

// ------------------------------------------------------------------ hyper-connections
// mixes = (flatten(x) @ fn^T) * rsqrt(mean(flatten(x)^2) + norm_eps) ; split + Sinkhorn -> pre/post/comb.
void hc_mixes(float* pre, float* post, float* comb, const float* x, const float* fn,
              const float* scale, const float* base, int M, cudaStream_t st);
// y[M,dim] = bf16( sum_j pre[j] * x[j] )
void hc_collapse(float* y, const float* x, const float* pre, int M, cudaStream_t st);
// y[j] = bf16( post[j]*f + sum_i comb[i][j] * res[i] )   (y may alias res: no, use a scratch)
void hc_post(float* y, const float* f, const float* res, const float* post, const float* comb, int M,
             cudaStream_t st);
// pre_mix for the first layer: one-hot on copy 0
void hc_identity_pre(float* pre, int M, cudaStream_t st);
// h[M,4,dim] = bf16(embed[ids]) broadcast to the 4 copies
void embed_expand(float* h, const uint16_t* emb, const int* d_ids, int M, cudaStream_t st);

// ------------------------------------------------------------------ router / MoE
// scores = sqrtsoftplus(x @ Wg^T) (fp32, bf16 weight); select top-k of scores + (img ? bias_vl : bias);
// weights = gathered unbiased scores / (sum + 1e-20) * route_scale.
void router(float* w, int* idx, const float* x, const uint16_t* Wg, const float* bias, const float* bias_vl,
            const uint8_t* d_img, int M, int E, int topk, float route_scale, cudaStream_t st,
            float* scratch = nullptr);   // scratch >= router_scratch_bytes(M, E): parallel path; null: one CTA per token
size_t router_scratch_bytes(int M, int E);
// Reorder a [E, K] bf16 gate in place for router()'s coalesced path (same sums bit for bit). Load time only.
// hc_collapse(y, x, pre) + rmsnorm(y, y, w, DIM) + act_quant_fp8(q, qs, y) in one launch, bit-identical.
void collapse_norm_quant(float* y, uint8_t* q, float* qs, const float* x, const float* pre, const uint16_t* w, int M,
                         float eps, cudaStream_t st);
bool router_permute(uint16_t* Wg, int E, int K, cudaStream_t st);
// NVFP4 dense overlay: after overlay_register(codes, g), fp8_linear(.., W = codes, Ws = e4m3/16 scales, ..)
// decodes NVFP4 instead of FP8 (DecNVFP4). overlay_count() = registered tensors.
// N rows, K columns; force_mma -1 = env (V41_NV_MMA), 0 = row-major k_dense, 1 = repack for k_nvm
void overlay_register(const uint8_t* codes, const uint8_t* scales, float g, int N, int K, int force_mma = -1);
// P6: repack an FP8 [N, K] weight in place for the fp8 tensor-core path (k_f8m); fp8_linear then serves it
// with k_f8m at every M (T1 vs k_dense, decode == prefill). N % 32, K % 256.
bool f8m_register(uint8_t* W, int N, int K);
// Q2_K/Q3_K routed experts: half2 fast path (default; tolerance-gated by tests/gate_kq_fast.cu) or the
// bit-exact fp32 decode (V41_KQ_FAST=0). Tests switch it in-process.
void kq_fast_set(bool on);
// Reorder one Q2_K/Q3_K [rows, K] matrix in place into code | scale | d planes (same bytes), for the
// vector-load fast kernel. tmp >= rows*(K/256)*block bytes. kq_soa(): the loader should repack.
void kq_soa_repack(uint8_t* W, int rows, int K, int bits, void* tmp, cudaStream_t st);
bool kq_soa();
// fp8 tensor-core layout (v41_kq_mma.cuh), same bytes; rows % 16 == 0. kq_mma(): the loader should use it.
void kq_mma_repack(uint8_t* W, int rows, int K, int bits, void* tmp, cudaStream_t st);
bool kq_mma();
int units_selftest();   // k_units_par vs serial k_units on random routings: mismatches (-1 = CUDA error)
size_t overlay_count();

// MXFP4(_HW): thor-v1 MXFP4_SRC / the HF checkpoint. KQ: thor-v1 Q2_K/Q3_K/Q4_K (bits = 2/3/4).
// LUT/KT: the sub-3-bit proposal (FORMAT_PROPOSAL.md) -- NOT in thor-v1; a loader refuses them
// until the spec reserves a qtype for them.
enum class ExpertFmt : int { MXFP4 = 0, MXFP4_HW = 1, LUT = 2, KT = 3, KQ = 4 };
// One routed-expert projection set, per the format. Pointer tables are DEVICE arrays [E].
struct ExpertTab {
    // gate/up (w1|w3) and down (w2) may use different formats: the usual sub-3-bit recipe spends
    // more bits on down_proj (FORMAT_PROPOSAL.md §4)
    ExpertFmt fmt13 = ExpertFmt::MXFP4, fmt2 = ExpertFmt::MXFP4;
    int bits13 = 4, bits2 = 4;                      // KQ: 2/3/4 (Q2_K..Q4_K); LUT/KT code bits
    bool kq_mma = false;                            // KQ 2/3 matrices in kq_mma_repack layout (k_kqm only)
    bool kq_soa = false;                            // KQ 2/3 matrices reordered by kq_soa_repack (fast kernel only)
    const float* heal_gain = nullptr;               // thor-v1 arm B `ffn.heal_gain` [1]: routed sum x gain
    const uint8_t* const* w2 = nullptr;
    const uint8_t* const* w2s = nullptr;
    const float* const* lut = nullptr;              // LUT fmt: [E] -> 16 floats (w13 then w2: 2x16)
    // KT fmt (thor-v1 KT2/KT3): the pass's packed RHT signs [in/8] (bit k of byte j: element 8j+k
    // negative), one per layer and pass, shared by every expert; activations are rotated with them
    const uint8_t* rht13 = nullptr;
    const uint8_t* rht2 = nullptr;
    // MXFP4 split tables (checkpoint layout: w1, w3 separate)
    const uint8_t* const* w1 = nullptr; const uint8_t* const* w1s = nullptr;
    const uint8_t* const* w3 = nullptr; const uint8_t* const* w3s = nullptr;
};
// Routed experts for M tokens with top-k selections. y[M,dim] (fp32 accumulators, NOT rounded)
// receives sum over slots IN ASCENDING EXPERT ORDER of bf16(expert(x) * w) — the reference loop.
// xq/xs: act_quant_fp8 of x. scratch >= moe_scratch_bytes(M, topk).
size_t moe_scratch_bytes(int M, int topk);
// KT activation rotation (exposed for tests/gate_v41_units.cu): out[r] = FWHT_256(x[r] * sign) / 16,
// the butterfly order of ops/lib/kt.py::rht_rows, bit-exact. K a multiple of 256.
void rht_rows(float* out, const float* x, const uint8_t* sign, int M, int K, cudaStream_t st);
void moe_routed(float* y, const uint8_t* xq, const float* xs, const float* w, const int* idx,
                const ExpertTab& T, int M, int topk, void* scratch, cudaStream_t st);
// Shared expert (fp8) added on top: y = bf16(y + bf16(shared(x))). Uses the fp8 linears.
void moe_shared_add(float* y, const uint8_t* xq, const float* xs,
                    const uint8_t* w1, const uint8_t* w1s, const uint8_t* w3, const uint8_t* w3s,
                    const uint8_t* w2, const uint8_t* w2s, int M, void* scratch, cudaStream_t st);
// o = shared(x) only (no add); o null -> it lives in scratch. Returns o. scratch >= moe_scratch_bytes(M, 1).
float* moe_shared(float* o, const uint8_t* xq, const float* xs, const uint8_t* w1, const uint8_t* w1s,
                  const uint8_t* w3, const uint8_t* w3s, const uint8_t* w2, const uint8_t* w2s, int M, void* scratch,
                  cudaStream_t st);

// ------------------------------------------------------------------ CSA2 attention
// Compressor, ratio 2 (fp32 pooling): kv/score [M, 512] already computed (bf16 weight, fp32 out).
// Updates the carried 2-slot state; emits latent[M,512] (bf16, pre-RoPE, pre-norm=NO: normalised
// with `norm_w`) and valid[M] (1 where a group completed at that row).
// hist (nullable, [M, 1024]): each row's (kv, score) pair as written into the state, for rollback.
void compressor_pool2(float* latent, int* valid, float* kv_state, float* score_state, const float* kv,
                      const float* score, const uint16_t* norm_w, const int* d_pos, int M, cudaStream_t st,
                      float* hist = nullptr);
// Rollback after a speculative verify that committed through an EVEN position whose row is hist_row
// (from compressor_pool2's hist): slot 0 <- that row. An odd last position needs nothing: the next row
// (even) rewrites slot 0 and the one after rewrites slot 1 before any pooling reads them.
void pool2_restore_even(float* kv_state, float* score_state, const float* hist_row, cudaStream_t st);
// Index keys: k[M,128] (= k_norm(wk(latent)), RoPE'd at group position) -> fp4 pack into the
// cache at slot pos/ratio for rows with valid=1 (valid==null: all rows). Same for compressed KV.
void store_packed_rows(uint8_t* cache_codes, uint8_t* cache_scales, const uint8_t* codes, const uint8_t* scales,
                       const int* valid, const int* d_pos, int ratio, int M, int code_bytes, int scale_bytes,
                       cudaStream_t st);
// Index scores for M queries: score[m][t] for t < n_m = (pos_m+1)/ratio, -inf beyond (and outside
// candidate blocks when cand != null). q[M,32,128] fp4-sim values, w[M,32] (pre-scaled, bf16),
// keys packed [T,64]+[T,4]. Score arithmetic mirrors the reference's bf16 einsum.
void index_scores(float* score, const float* q, const float* w, const uint8_t* kcodes, const uint8_t* kscales,
                  const uint8_t* cand, const int* d_pos, int ratio, int M, int ld_score, cudaStream_t st,
                  const uint8_t* qcodes = nullptr, const uint8_t* qscales = nullptr);
// qcodes/qscales: q as fp4_sim's packed codes [M,32,64] + e8m0 [M,32,4]; with V41_IS_MMA=1 the scores run on
// fp8 tensor cores (v41_is_mma.cuh, P3), else (or without them) k_index_scores on q.
bool is_mma_on();
// Level-one candidate blocks (layer 20): cand[m][block] = 1 for the kept 2048 blocks of 8.
void candidate_blocks(uint8_t* cand, const float* score, const int* d_pos, int ratio, int M, int ld_score,
                      int ld_cand, void* scratch, cudaStream_t st);
size_t candidate_scratch_bytes(int M, int max_blocks);
// top-k (k = min(512, n_m)) per row, emitted ascending, + offset; unused slots -1.
void topk_rows(int* idxs, const float* score, const int* d_pos, int ratio, int M, int ld_score, int k,
               int offset, void* scratch, cudaStream_t st);
size_t topk_scratch_bytes(int M);
// Sparse MLA attention over a 256-slot window ring (fp32 rows, fp8-sim values) plus packed fp4
// compressed rows. Row m attends window positions max(0,pos_m-127)..pos_m and the compressed
// indices cidx[m][0..ncomp) (already shifted by `coff`; -1 = masked). o is bf16-rounded.
// Scratch for the split decode/verify form (M <= 8); without it sparse_attn uses the one-kernel form.
void sparse_attn_reserve(int max_m);
void topk_serial_set(bool on);           // gate use: the thread-0 scan form of topk_rows
void sparse_attn_split_set(bool on);
void sparse_attn(float* o, const float* q, const float* win_ring, int ring, const uint8_t* ccodes,
                 const uint8_t* cscales, const int* cidx, int ncomp, int coff, const float* sink,
                 const int* d_pos, int M, float scale, cudaStream_t st);
// write window KV rows into the ring at pos % ring
void ring_store(float* ring_buf, int ring, const float* kv, const int* d_pos, int M, cudaStream_t st);

// ------------------------------------------------------------------ Engram
// rows: [M, 24, row_bytes] in the arena (fp8 x256 + e8m0 x8 = 264 B; or MXFP6 = 200 B) -> e[M,6144]
// bf16 values (dequant exactly as ParallelEngramEmbedding.forward). cap: the arena's capacity in tokens
// (EngramStore max_tokens): the slot-offset table sits after cap*24 slots, not after M*24.
void engram_dequant(float* e, const uint8_t* rows, int cap, int fmt, int M, cudaStream_t st);
// kv[M, 25600] = wkv(e) (bf16). h[M,4,dim] += gate * value, gate per (token, copy). mask[M] (0 = shut).
void engram_gate(float* h, const float* kv, const uint16_t* qw, const uint16_t* kw, const uint8_t* mask,
                 int M, float eps, cudaStream_t st);

// SwiGLU with the training clamps: up in [-L, L], gate <= L; act = silu(gate)*up*(w) -> bf16
// gate/up rows have leading dimension `ld` (2*inter when they come out of one fused w1|w3 GEMV).
void swiglu(float* act, const float* gate, const float* up, const float* w_row, int M, int inter, float limit,
            int ld, cudaStream_t st);
void add_bf16(float* y, const float* a, size_t n, cudaStream_t st);

} // namespace v41k
