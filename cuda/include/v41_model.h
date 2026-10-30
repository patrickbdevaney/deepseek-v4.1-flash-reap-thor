// v41_model.h — DeepSeek-V4.1-Flash-1017 / REAP forward pass over the v41k kernels.
//
// One layer forward handles M rows (decode M=1, verify M=K+1, prefill chunk M<=128) at arbitrary
// positions; positions, ids and the image mask are read from device memory so the M=1 forward can
// be captured once and replayed as a CUDA graph.
#pragma once
#include "v41_config.h"
#include "v41_kernels.h"
#include "engram_store.h"
#include "safetensors.h"
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <cuda_runtime.h>

namespace v41m {

// --------------------------------------------------------------------------- weights
// Selective loader: pread each wanted tensor into ONE cudaMallocManaged arena (preferred location =
// device, prefetched), never mmap-faulting the data. Skips the Engram tables (row store), vision,
// and whatever `want` rejects. Partial checkpoints (missing shards) are tolerated: has() says.
struct DevT { const void* p = nullptr; std::string dtype; std::vector<int64_t> shape; size_t nbytes = 0; };
class Store {
public:
    using Filter = std::function<bool(const std::string&)>;
    // rename(file, tensor) -> stored name (default: the tensor name); used to prefix per-layer overlay files
    using Rename = std::function<std::string(const std::string&, const std::string&)>;
    Store(const std::string& dir, const Filter& want, const Rename& rename = nullptr);
    ~Store();
    bool has(const std::string& n) const { return t_.count(n) > 0; }
    const DevT& get(const std::string& n) const;
    template <class T> const T* p(const std::string& n) const { return (const T*)get(n).p; }
    template <class T> const T* opt(const std::string& n) const { return has(n) ? (const T*)get(n).p : nullptr; }
    double gib() const { return bytes_ / 1073741824.0; }
    const std::string& thor_json() const { return thor_json_; }   // empty for an HF checkpoint
    size_t count() const { return t_.size(); }
private:
    std::unordered_map<std::string, DevT> t_;
    void* arena_ = nullptr; size_t bytes_ = 0;
    std::string thor_json_;
    std::vector<std::unique_ptr<st::SafeTensors>> files_;
};

// Experts' on-disk formats: ops/lib/thor_format.md (thor-v1 stacked Q2_K/Q3_K/Q4_K or MXFP4_SRC),
// or the HF checkpoint's per-expert MXFP4. LUT/KT (FORMAT_PROPOSAL.md) have kernels but no loader
// path until the spec reserves qtypes for them.
struct Experts {
    v41k::ExpertTab tab;                 // device pointer tables
    int n_routed = 0;
    void* dev_tables = nullptr;          // owns the device arrays the tab points into
};

struct LayerW {
    int L = 0, ratio = 0;
    bool kv_src = false, idx_src = false, cand_src = false, uses_cand = false;
    int kv_owner = -1, idx_owner = -1;
    const float* attn_sink = nullptr;
    const uint16_t *attn_norm = nullptr, *ffn_norm = nullptr, *q_norm = nullptr, *kv_norm = nullptr;
    const uint8_t *wq_a = nullptr, *wq_a_s = nullptr, *wq_b = nullptr, *wq_b_s = nullptr, *wkv = nullptr, *wkv_s = nullptr;
    const uint8_t *wo_a = nullptr, *wo_a_s = nullptr, *wo_b = nullptr, *wo_b_s = nullptr;
    const uint16_t *cmp_wkv = nullptr, *cmp_wgate = nullptr, *cmp_norm = nullptr;
    const uint16_t *idx_wk = nullptr, *idx_k_norm = nullptr, *idx_wproj = nullptr;
    const uint8_t *idx_wq_b = nullptr, *idx_wq_b_s = nullptr;
    const float *hc_attn_fn = nullptr, *hc_attn_base = nullptr, *hc_attn_scale = nullptr;
    const float *hc_ffn_fn = nullptr, *hc_ffn_base = nullptr, *hc_ffn_scale = nullptr;
    const uint16_t* gate_w = nullptr; const float *gate_bias = nullptr, *gate_bias_vl = nullptr;
    Experts ex;
    const uint8_t *sh_w1 = nullptr, *sh_w1s = nullptr, *sh_w2 = nullptr, *sh_w2s = nullptr, *sh_w3 = nullptr, *sh_w3s = nullptr;
    const uint8_t *eg_wkv = nullptr, *eg_wkv_s = nullptr; const uint16_t *eg_q = nullptr, *eg_k = nullptr;
};

struct Model {
    std::unique_ptr<Store> store;
    std::unique_ptr<Store> overlay;      // NVFP4 dense overlay, names "layers.L.<tensor>"; its FP8 twins are never loaded
    std::vector<LayerW> layers;          // index = layer id; only `loaded` ones are valid
    std::vector<bool> loaded;
    const uint16_t *embed = nullptr, *head = nullptr, *norm = nullptr;
    uint8_t *head4 = nullptr, *head4s = nullptr; float head4g = 0.f;   // NVFP4 copy of head (enable_head2; the drafter shares it)
    std::string quant_format = "mxfp4";
    int n_routed = 0;
};
struct LoadOpts {
    int layer_lo = 0, layer_hi = v41::N_LAYERS - 1;  // inclusive range of blocks to load
    bool embed_head = true;
    bool mxfp4_hw = true;               // MXFP4 via cvt.rn.f16x2.e2m1x2 (false: bit-construct path)
    std::string keepset_sha256;         // thor-v1 builds: must equal thor.json keepset_sha256 (fatal otherwise)
    std::string overlay_dir;            // thor-overlay-nvfp4 dir (L00..L39.safetensors): replaces those FP8 dense weights
};
// Throws std::runtime_error with the missing tensor's name on a layout mismatch.
void load_model(Model& m, const std::string& dir, const LoadOpts& o);
void bind_layer(Model& m, int L, bool mxfp4_hw);
// Routed experts under a tensor prefix (decoder layers and the DSpark mtp blocks share it). qtype_of
// names a stacked tensor's qtype (thor.json / s11 head.json); empty function = per-expert HF layout
// only. inter = the experts' intermediate size. Returns the expert count; throws on a layout mismatch.
int bind_experts(const Store& S, const std::string& ep, const std::function<std::string(const std::string&)>& qtype_of,
                 bool mxfp4_hw, int inter, Experts& ex);

// --------------------------------------------------------------------------- state
struct State {
    int seqmax = 0, max_rows = 0;
    // per-layer window ring (256 slots: a prefill chunk of <=128 rows never overwrites a slot a
    // row of the same chunk still needs)
    // a chunk of M rows reads the 127 positions before its first row and writes M slots, so chunks may be
    // up to RING - WINDOW rows (prefill re-reads every touched expert once per chunk: bigger chunks)
    static constexpr int MAX_CHUNK = 512;
    static constexpr int RING = v41::WINDOW + MAX_CHUNK;
    std::vector<float*> ring;            // [N_LAYERS] -> [RING, 512]
    // per kv_source: compressed KV (fp4 codes 256 B + E4M3 scales 32 B) and index keys (64 + 4 B)
    struct Src { uint8_t *ckv = nullptr, *ckv_s = nullptr, *ik = nullptr, *ik_s = nullptr;
                 float *kv_state = nullptr, *score_state = nullptr; int rows = 0;
                 float* pool_hist = nullptr; };    // ratio-2 sources: [max_rows, kv 512 | score 512] (rollback)
    Src src[v41::N_LAYERS];
    // shared-attention slots handed down the stack (the reference's SharedAttentionRuntime)
    int* topk = nullptr;                 // [shared_rows, 512]
    uint8_t* cand = nullptr;             // [shared_rows, seqmax/8]
    int ld_cand = 0;
    // Layer-major runs (the golden gate) keep the shared slots for EVERY position, because layer L
    // finishes all chunks before layer L+1 reads them: slots are indexed shared_base + row.
    int shared_rows = 0, shared_base = 0;
    // inputs
    int *d_pos = nullptr, *d_ids = nullptr; uint8_t *d_img = nullptr, *d_noimg = nullptr;
    // activations + scratch
    float *h = nullptr, *h2 = nullptr, *pre = nullptr, *post = nullptr, *comb = nullptr, *pre_next = nullptr;
    float *x = nullptr, *xa = nullptr, *qa = nullptr, *qr = nullptr, *q = nullptr, *kv = nullptr, *o = nullptr;
    float *og = nullptr, *ck = nullptr, *cs = nullptr, *lat = nullptr, *ik = nullptr, *iq = nullptr, *iw = nullptr;
    uint8_t *iqp = nullptr, *iqs = nullptr;            // q as fp4 codes + e8m0 (V41_IS_MMA, P3)
    float *score = nullptr; int ld_score = 0, score_rows_ = 0;   // indexer scores [score_rows_, seqmax]
    float *rw = nullptr; int* ridx = nullptr; float* y = nullptr; float* rscr = nullptr;
    float *eg_e = nullptr, *eg_kv = nullptr;
    float *logits = nullptr;                          // [max_rows, VOCAB]: head_row writes row 0, head_rows rows 0..n-1
    v41k::H2Scr hd2; int* d_hfb = nullptr;             // two-stage greedy head scratch, per-row fallback flags
    int* d_argmax = nullptr;                          // [max_rows] argmax of each head_rows row
    uint8_t *xq = nullptr; float* xs = nullptr;       // act_quant scratch, largest K x max_rows
    uint8_t *pk = nullptr, *ps = nullptr;             // packed fp4 scratch
    int* valid = nullptr;
    void* moe_scratch = nullptr; void* cand_scratch = nullptr;
    // shared expert on a forked stream, joined before its bf16 add (bit-identical to the serial order)
    cudaStream_t side = nullptr; cudaEvent_t ev_fork = nullptr, ev_join = nullptr;
    cudaStream_t hcs = nullptr; cudaEvent_t ev_hf = nullptr, ev_hj = nullptr;   // hc_mixes overlap (V41_HC_OVERLAP)
    cudaStream_t kvs = nullptr; cudaEvent_t ev_kf = nullptr, ev_kj = nullptr;   // attention KV branch (V41_KV_OVERLAP)
    uint8_t* xq2 = nullptr; float* xs2 = nullptr;     // act_quant(qr) for the q branch while kvs reads xq
    float* sh_o = nullptr; void* sh_scratch = nullptr;
    std::vector<void*> allocs;
    size_t bytes = 0;
    State() = default;
    // owns its device buffers (freed in ~State): move-only, or a copy's destructor frees the original's
    State(const State&) = delete; State& operator=(const State&) = delete;
    State(State&&) = default; State& operator=(State&&) = default;
    ~State();
};
// seqmax: context ceiling (<= 1M). max_rows: largest M per call. score_budget caps the
// [M, seqmax] indexer score buffer; the prefill chunk shrinks to fit it.
void alloc_state(State& s, int seqmax, int max_rows, size_t score_budget = (size_t)1 << 30, int shared_rows = 0);
void reset_state(State& s, cudaStream_t st);     // zero caches and compressor state (new sequence)
int  max_chunk(const State& s);                  // rows per prefill chunk the score buffer allows

// --------------------------------------------------------------------------- forward
struct Engram {
    EngramStore* store = nullptr;        // null + rows_dev null => Engram OFF (wrong output; benches only)
    const uint8_t* rows_dev[2] = {nullptr, nullptr};   // alternatively: rows already in device memory
    int rows_cap = 0;                    // rows_dev arenas: capacity in tokens (offset table after rows_cap*24 slots)
    int fmt = 0;
    bool graph_wait = false;             // emit cuStreamWaitValue32/WriteValue32 around the layer
};
// Hook for the golden gate: called after each layer with that layer's output stream and pre_mix.
using LayerHook = std::function<void(int L, const float* h, const float* pre_mix, int M, cudaStream_t)>;

// Run layers [lo, hi] on rows [0, M) of s.h / s.pre (rows = tokens at positions s.d_pos[0..M)).
// Decoder layers (> ENC_LAST) only run rows whose position >= prompt_end - dec_span(L), when
// prompt_end > 0 (prefill of a prompt whose logits are needed only at its end). Returns the
// first row the LAST layer ran (the rows above it are valid outputs).
// pos0 = position of row 0 (rows are consecutive positions whenever prompt_end > 0).
// dec_margin widens every decoder span by that many rows, so a LATER prefix reuse that diverges up
// to dec_margin rows before prompt_end still finds valid window KV at every decoder layer.
int forward_layers(const Model& m, State& s, int lo, int hi, int M, int prompt_end, int pos0, const Engram& eg,
                   cudaStream_t st, const LayerHook& hook = nullptr, int dec_margin = 0);
// embed rows -> s.h (4 copies), identity pre-mix
void embed_rows(const Model& m, State& s, int M, cudaStream_t st);
// final collapse + norm + head for row r -> s.logits [129280] fp32
void head_row(const Model& m, State& s, int r, cudaStream_t st);
// Two-stage greedy head (v41k::head2_argmax): argmax only, s.logits untouched. head_row_greedy -> *d_tok,
// head_rows_greedy -> s.d_argmax[0..n); s.d_hfb[r] = 1 means row r must be redone by head_full_argmax.
// enable_head2 builds the NVFP4 head copy and the scratch (once, before any graph capture).
void enable_head2(Model& m, State& s, int cap, float delta, cudaStream_t st);
void head_row_greedy(const Model& m, State& s, int r, int* d_tok, cudaStream_t st);
void head_rows_greedy(const Model& m, State& s, int r0, int n, cudaStream_t st);
void head_full_argmax(const Model& m, State& s, int row, int* d_out, cudaStream_t st);   // s.x row already normed
// greedy argmax over s.logits into *d_tok
void argmax_logits(const State& s, int* d_tok, cudaStream_t st);
// Speculative verify: final collapse + norm + head for rows [r0, r0+n) -> s.logits rows 0..n-1, and their
// argmaxes -> s.d_argmax[0..n). One pass over the head weights for all n rows.
void head_rows(const Model& m, State& s, int r0, int n, cudaStream_t st);
// After a verify forward whose row `row` (position `pos`) is the last committed one: put back the
// carried state the rejected rows overwrote (the ratio-2 compressors' 2-slot pooling state). The window
// ring, the packed caches and the shared slots need nothing: a rejected position is rewritten by the
// next forward before any query can read it (ring 256 >= window 128 + verify width).
void rollback_state(State& s, int row, int pos, cudaStream_t st);

} // namespace v41m
