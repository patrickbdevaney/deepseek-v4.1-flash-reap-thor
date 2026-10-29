// v41_draft.h — the checkpoint's own DSpark speculative head (the three `mtp.*` blocks) on the v41k
// kernels, plus what speculation needs from the main forward (the tap layers' hidden states).
//
// What the reference fixes (inference/model.py DSparkBlock / DSparkAttention / forward_spec):
//   * main_hidden at position t = concat over the tap layers 37, 38, 39 of the hc-MEAN of the stream
//     ENTERING that layer (bf16), [3 * dim]. main_x = main_norm(main_proj(main_hidden)) (mtp.0 only),
//     and every block b keeps a window cache of main_kv_b(t) = fp8sim(rope_t(kv_norm_b(wkv_b(main_x)))).
//   * a draft at anchor p (the last position the main model has processed) takes the committed token
//     at p+1 and four noise tokens (block slots 0..4, positions p+1..p+5). Each block's attention
//     sees the window main_kv at positions p-127..p plus all five block-slot kvs (NOT causal within
//     the block), with the sink; ratio 0: base rope theta, no YaRN, no compressor, no indexer.
//   * each block's FFN is its own MoE: 128 routed experts (or the s11 in-model keep-set), top-3,
//     sqrtsoftplus + bias (no VL bias: drafts are text), route_scale 1.5, plus a shared expert.
//   * head: hc_pre with the last block's mix -> mtp.2.norm -> the MAIN head (bf16) for all 5 slots;
//     then a Markov chain: out[0] = tok, logits[i] += markov_head(out[i]) (embed 256 -> head),
//     out[i+1] = argmax(logits[i]). out[1..5] are the drafts for positions p+2 .. p+6.
//
// The window is written from the MAIN forward (tap_hook), for every row it runs: rows a speculative
// verify later rejects leave slots that the next verify rewrites before any draft reads them (the
// drafter only reads positions <= its anchor, and the ring has 256 slots for a 128 window).
//
// Two residency arms (ops/s11_spec.py): `inmodel` (an s11 head dir whose routed experts are the
// REAP-pruned keep-set, stacked K-quant, qtype in head.json) and `separate` (the source blocks: the
// thor build's mtp-*.safetensors or the HF shards 44-46). load_drafter() takes every mtp.* tensor from
// the head dir when one is given and falls back to the checkpoint for the rest.
#pragma once
#include "v41_model.h"

namespace v41m {

struct DBlock {
    const float* attn_sink = nullptr;
    const uint16_t *attn_norm = nullptr, *ffn_norm = nullptr, *q_norm = nullptr, *kv_norm = nullptr;
    const uint8_t *wq_a = nullptr, *wq_a_s = nullptr, *wq_b = nullptr, *wq_b_s = nullptr, *wkv = nullptr, *wkv_s = nullptr;
    const uint8_t *wo_a = nullptr, *wo_a_s = nullptr, *wo_b = nullptr, *wo_b_s = nullptr;
    const float *hc_attn_fn = nullptr, *hc_attn_base = nullptr, *hc_attn_scale = nullptr;
    const float *hc_ffn_fn = nullptr, *hc_ffn_base = nullptr, *hc_ffn_scale = nullptr;
    const uint16_t* gate_w = nullptr; const float* gate_bias = nullptr;
    Experts ex;
    const uint8_t *sh_w1 = nullptr, *sh_w1s = nullptr, *sh_w2 = nullptr, *sh_w2s = nullptr, *sh_w3 = nullptr, *sh_w3s = nullptr;
};

struct Drafter {
    std::unique_ptr<Store> head, base;   // head: an s11 head dir (may be null); base: the checkpoint's mtp.*
    DBlock b[v41::N_MTP];
    const uint8_t *main_proj = nullptr, *main_proj_s = nullptr;
    const uint16_t *main_norm = nullptr, *norm = nullptr, *mk_embed = nullptr, *mk_head = nullptr;
    int n_routed = 0;
    std::string desc;                    // "<dir> (<arm>, <experts>/block, <qtype>)" for logs and JSON
    double gib = 0;
    // state
    static constexpr int RING = 256;
    float* ring[v41::N_MTP] = {};        // [RING, 512] main_kv per position (slot pos % RING)
    int max_rows = 0;
    float* taps = nullptr;               // [max_rows, 3, DIM] bf16 values (hc-mean entering 37/38/39)
    float* mainx = nullptr;              // [max_rows, DIM]
    float* kvt = nullptr;                // [max_rows, 512]
    uint8_t* xq = nullptr; float* xs = nullptr;    // act_quant scratch, largest K x max(max_rows, 5)
    // draft scratch (5 rows)
    int *d_ids = nullptr, *d_pos = nullptr, *d_anchor = nullptr, *d_tok = nullptr;
    int* d_out = nullptr;                // [6]: out[0] = the committed token, out[1..5] = drafts
    float *h = nullptr, *h2 = nullptr, *pre = nullptr, *post = nullptr, *comb = nullptr, *pre_next = nullptr;
    float *x = nullptr, *xa = nullptr, *qa = nullptr, *qr = nullptr, *q = nullptr, *kv = nullptr, *o = nullptr;
    float *og = nullptr, *y = nullptr, *rw = nullptr; int* ridx = nullptr; float* rscr = nullptr;
    void* moe_scratch = nullptr;
    float *logits = nullptr, *mk_e = nullptr, *mk_bias = nullptr;
    // NVFP4 copy of the main head for drafting (V41_DRAFT_HEAD_FP4=1); null = draft with the bf16 head
    uint8_t *head4 = nullptr, *head4s = nullptr; float head4g = 0.f;
    std::vector<void*> allocs;
    size_t bytes = 0;
    ~Drafter();
};

// ckpt: the --ckpt dir (thor build or HF checkpoint). head_dir: an s11 head dir, or "" for the
// checkpoint's own (native, unpruned) blocks. Throws with the missing tensor's name.
void load_drafter(Drafter& D, const std::string& ckpt, const std::string& head_dir, bool mxfp4_hw);
void alloc_drafter(Drafter& D, int max_rows);
// V41_DRAFT_HEAD_FP4=1: quantize the model's head to NVFP4 for the draft (acceptance only; verify is bf16)
void prepare_draft_head(const Model& m, Drafter& D, cudaStream_t st);
void reset_drafter(Drafter& D, cudaStream_t st);
// A LayerHook for forward_layers over `M_total` rows: records the taps (after layers 36, 37, 38) and,
// after layer 38, writes main_kv of every row the call ran at layer 38 into the three window rings.
LayerHook tap_hook(Drafter& D, State& s, int M_total);
// Draft at anchor position p (device scalar D.d_anchor) with the committed token at p+1 (device
// scalar D.d_tok). Graph-capturable: every input is in device memory. -> D.d_out[0..5].
void draft_body(const Model& m, Drafter& D, cudaStream_t st, int K = 0);   // K: slots read (0 = all)
// Host helper: set the anchor and the token from host values, then draft_body.
void draft(const Model& m, Drafter& D, int anchor_pos, int tok, cudaStream_t st);

} // namespace v41m
