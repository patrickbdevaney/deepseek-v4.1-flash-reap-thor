// v41_model.cu — weights, state and the layer forward. The forward follows inference/model.py
// line by line; each block below names the reference code it reproduces.
#include "v41_model.h"
#include "safetensors.h"
#include <cuda.h>
#include <fcntl.h>
#include <unistd.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <set>
#include <unordered_map>
#include "third_party/json.hpp"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) throw std::runtime_error(std::string(#x) + ": " + cudaGetErrorString(e_)); } while (0)

namespace v41m {

using namespace v41;

// =========================================================================== Store
Store::Store(const std::string& dir, const Filter& want) {
    // Two layouts: a thor-v1 build (thor.json + layer-NN/globals/... files, ops/lib/thor_format.md)
    // or an HF checkpoint (model.safetensors.index.json). Missing files are skipped (partial
    // checkpoints for per-layer golden runs); has() reports what loaded.
    std::vector<std::string> files;
    try { thor_json_ = st::read_file(dir + "/thor.json"); } catch (...) {}
    if (!thor_json_.empty()) {
        const auto j = nlohmann::json::parse(thor_json_);
        if (j.value("format", "") != "thor-v1") throw std::runtime_error(dir + ": thor.json format is not thor-v1");
        if (!j.value("complete", false)) throw std::runtime_error(dir + ": thor build is not complete (thor.json complete != true)");
        for (auto it = j["files"].begin(); it != j["files"].end(); ++it) files.push_back(it.key());
    } else {
        const auto j = nlohmann::json::parse(st::read_file(dir + "/model.safetensors.index.json"));
        std::set<std::string> u;
        for (auto it = j["weight_map"].begin(); it != j["weight_map"].end(); ++it) u.insert(it.value().get<std::string>());
        files.assign(u.begin(), u.end());
    }
    struct Item { std::string name; int fd; size_t off, n; std::string dtype; std::vector<int64_t> shape; };
    std::vector<Item> items; size_t total = 0;
    for (const auto& f : files) {
        if (f.size() < 12 || f.compare(f.size() - 12, 12, ".safetensors") != 0) continue;
        std::unique_ptr<st::SafeTensors> sh;
        try { sh.reset(new st::SafeTensors(dir + "/" + f)); } catch (...) { continue; }
        bool any = false;
        for (auto& t : sh->all()) {
            if (t.first.compare(0, 6, "__pad.") == 0 || !want(t.first)) continue;   // thor-v1 padding tensors
            Item it{t.first, sh->fd(), sh->dataFileOffset() + (size_t)(t.second.data - sh->dataStart()), t.second.nbytes,
                    t.second.dtype, t.second.shape};
            total += (it.n + 255) & ~(size_t)255;
            items.push_back(std::move(it));
            any = true;
        }
        if (any) files_.push_back(std::move(sh));
    }
    if (!total) return;
    int dev = 0; CK(cudaGetDevice(&dev));
    CK(cudaMallocManaged(&arena_, total));
    size_t off = 0;
    for (auto& it : items) {
        char* dst = (char*)arena_ + off; size_t got = 0;
        while (got < it.n) {
            const size_t want_n = std::min(it.n - got, (size_t)1 << 30);
            const ssize_t r = pread(it.fd, dst + got, want_n, (off_t)(it.off + got));
            if (r <= 0) throw std::runtime_error("pread failed for " + it.name);
            got += (size_t)r;
        }
        posix_fadvise(it.fd, (off_t)it.off, (off_t)it.n, POSIX_FADV_DONTNEED);   // no page-cache double copy
        DevT d; d.p = dst; d.dtype = it.dtype; d.shape = it.shape; d.nbytes = it.n;
        t_.emplace(it.name, std::move(d));
        off += (it.n + 255) & ~(size_t)255;
    }
    bytes_ = total;
    files_.clear();                                   // unmap the headers; the data is in the arena
    cudaMemLocation loc{}; loc.type = cudaMemLocationTypeDevice; loc.id = dev;
    cudaMemAdvise(arena_, total, cudaMemAdviseSetPreferredLocation, loc);
    cudaMemPrefetchAsync(arena_, total, loc, 0, 0);
    CK(cudaStreamSynchronize(0));
}
Store::~Store() { if (arena_) cudaFree(arena_); }   // files_ close after the reads are done
const DevT& Store::get(const std::string& n) const {
    auto it = t_.find(n);
    if (it == t_.end()) throw std::runtime_error("weight not found: " + n);
    return it->second;
}

// =========================================================================== load


void bind_layer(Model& m, int L, bool hw) {
    const Store& S = *m.store;
    LayerW& w = m.layers[L];
    const std::string p = "layers." + std::to_string(L) + ".";
    w.L = L; w.ratio = compress_ratio(L);
    w.kv_src = is_kv_source(L); w.idx_src = is_index_source(L);
    w.cand_src = L == CAND_SOURCE; w.uses_cand = uses_candidates(L) && w.ratio > 0;
    w.kv_owner = w.ratio ? kv_owner(L) : -1; w.idx_owner = w.ratio ? index_owner(L) : -1;
    auto u8 = [&](const std::string& n) { return S.p<uint8_t>(p + n); };
    auto b16 = [&](const std::string& n) { return S.p<uint16_t>(p + n); };
    auto f32 = [&](const std::string& n) { return S.p<float>(p + n); };
    w.attn_sink = f32("attn.attn_sink");
    w.attn_norm = b16("attn_norm.weight"); w.ffn_norm = b16("ffn_norm.weight");
    w.q_norm = b16("attn.q_norm.weight"); w.kv_norm = b16("attn.kv_norm.weight");
    w.wq_a = u8("attn.wq_a.weight"); w.wq_a_s = u8("attn.wq_a.scale");
    w.wq_b = u8("attn.wq_b.weight"); w.wq_b_s = u8("attn.wq_b.scale");
    w.wkv = u8("attn.wkv.weight"); w.wkv_s = u8("attn.wkv.scale");
    w.wo_a = u8("attn.wo_a.weight"); w.wo_a_s = u8("attn.wo_a.scale");
    w.wo_b = u8("attn.wo_b.weight"); w.wo_b_s = u8("attn.wo_b.scale");
    if (S.get(p + "attn.wo_a.weight").dtype != "F8_E4M3")
        throw std::runtime_error(p + "attn.wo_a is not FP8 (a convert.py output? load the HF checkpoint layout)");
    if (w.kv_src) {
        w.cmp_wkv = b16("attn.compressor.wkv.weight"); w.cmp_norm = b16("attn.compressor.norm.weight");
        if (w.ratio > 1) w.cmp_wgate = b16("attn.compressor.wgate.weight");
    }
    if (w.idx_src) {
        w.idx_wq_b = u8("attn.indexer.wq_b.weight"); w.idx_wq_b_s = u8("attn.indexer.wq_b.scale");
        w.idx_wproj = b16("attn.indexer.weights_proj.weight");
        if (w.kv_src) { w.idx_wk = b16("attn.indexer.wk.weight"); w.idx_k_norm = b16("attn.indexer.k_norm.weight"); }
    }
    w.hc_attn_fn = f32("hc_attn_fn"); w.hc_attn_base = f32("hc_attn_base"); w.hc_attn_scale = f32("hc_attn_scale");
    w.hc_ffn_fn = f32("hc_ffn_fn"); w.hc_ffn_base = f32("hc_ffn_base"); w.hc_ffn_scale = f32("hc_ffn_scale");
    w.gate_w = b16("ffn.gate.weight"); w.gate_bias = f32("ffn.gate.bias"); w.gate_bias_vl = S.opt<float>(p + "ffn.gate.bias_vl");
    w.sh_w1 = u8("ffn.shared_experts.w1.weight"); w.sh_w1s = u8("ffn.shared_experts.w1.scale");
    w.sh_w2 = u8("ffn.shared_experts.w2.weight"); w.sh_w2s = u8("ffn.shared_experts.w2.scale");
    w.sh_w3 = u8("ffn.shared_experts.w3.weight"); w.sh_w3s = u8("ffn.shared_experts.w3.scale");
    if (engram_slot(L) >= 0) {
        w.eg_wkv = u8("engram.wkv.weight"); w.eg_wkv_s = u8("engram.wkv.scale");
        w.eg_q = b16("engram.q_weight"); w.eg_k = b16("engram.k_weight");
    }
    // routed experts. Three layouts:
    //   thor-v1 stacked ggml K-quant  ffn.experts.wX.qweight  U8 [n, out, row_bytes]  qtype Q2_K..Q4_K
    //   thor-v1 stacked MXFP4_SRC     ffn.experts.wX.weight/.scale  [n, out, in/2] / [n, out, in/32]
    //   HF checkpoint per expert      ffn.experts.E.wX.weight/.scale (MXFP4 I8 + E8M0)
    const std::string ep = p + "ffn.experts.";
    int E = 0;
    std::vector<const void*> tab;
    auto fmt_of = [&](const std::string& qt, v41k::ExpertFmt& f, int& bits) {
        if (qt == "MXFP4_SRC") { f = hw ? v41k::ExpertFmt::MXFP4_HW : v41k::ExpertFmt::MXFP4; bits = 4; }
        else if (qt == "Q2_K") { f = v41k::ExpertFmt::KQ; bits = 2; }
        else if (qt == "Q3_K") { f = v41k::ExpertFmt::KQ; bits = 3; }
        else if (qt == "Q4_K") { f = v41k::ExpertFmt::KQ; bits = 4; }
        else throw std::runtime_error(p + ": unknown expert qtype '" + qt + "' -- refusing (thor_format.md obligation 1)");
    };
    if (S.has(ep + "w1.qweight") || S.has(ep + "w1.weight")) {
        const bool kq = S.has(ep + "w1.qweight");
        const char* part = kq ? ".qweight" : ".weight";
        std::string qt13 = "MXFP4_SRC", qt2 = "MXFP4_SRC";
        if (!S.thor_json().empty()) {
            const auto j = nlohmann::json::parse(S.thor_json());
            const auto& T = j["tensors"];
            qt13 = T[ep + "w1" + part].value("qtype", "");
            qt2 = T[ep + "w2" + part].value("qtype", "");
            if (T[ep + "w3" + part].value("qtype", "") != qt13) throw std::runtime_error(p + ": w1/w3 qtypes differ");
        } else if (kq) throw std::runtime_error(p + ": K-quant experts without a thor.json manifest");
        fmt_of(qt13, w.ex.tab.fmt13, w.ex.tab.bits13);
        fmt_of(qt2, w.ex.tab.fmt2, w.ex.tab.bits2);
        const DevT& t1 = S.get(ep + "w1" + part);
        E = (int)t1.shape[0];
        tab.assign(6 * (size_t)E, nullptr);
        const char* proj[3] = {"w1", "w3", "w2"};
        const int slot[3] = {0, 2, 4};
        for (int j = 0; j < 3; ++j) {
            const DevT& t = S.get(ep + proj[j] + part);
            const int out = j < 2 ? MOE_INTER : DIM, in = j < 2 ? DIM : MOE_INTER;
            if ((int)t.shape[0] != E || (int)t.shape[1] != out) throw std::runtime_error(ep + proj[j] + ": bad stacked shape");
            const size_t stride = t.nbytes / E;
            const v41k::ExpertFmt f = j < 2 ? w.ex.tab.fmt13 : w.ex.tab.fmt2;
            const int bits = j < 2 ? w.ex.tab.bits13 : w.ex.tab.bits2;
            if (f == v41k::ExpertFmt::KQ) {
                const int bb = bits == 2 ? 84 : bits == 3 ? 110 : 144;
                if (stride != (size_t)out * (in / 256) * bb) throw std::runtime_error(ep + proj[j] + ": row bytes do not match the qtype");
            } else if (stride != (size_t)out * in / 2) throw std::runtime_error(ep + proj[j] + ": MXFP4_SRC shape mismatch");
            const DevT* sc = kq ? nullptr : &S.get(ep + proj[j] + ".scale");
            for (int e = 0; e < E; ++e) {
                tab[(size_t)slot[j] * E + e] = (const char*)t.p + (size_t)e * stride;
                tab[(size_t)(slot[j] + 1) * E + e] = sc ? (const char*)sc->p + (size_t)e * (sc->nbytes / E) : nullptr;
            }
        }
    } else {
        while (S.has(ep + std::to_string(E) + ".w1.weight")) ++E;
        if (!E) throw std::runtime_error("no routed experts under " + ep);
        fmt_of("MXFP4_SRC", w.ex.tab.fmt13, w.ex.tab.bits13);
        fmt_of("MXFP4_SRC", w.ex.tab.fmt2, w.ex.tab.bits2);
        tab.assign(6 * (size_t)E, nullptr);
        for (int e = 0; e < E; ++e) {
            const std::string q = ep + std::to_string(e);
            tab[0 * E + e] = S.p<void>(q + ".w1.weight"); tab[1 * E + e] = S.p<void>(q + ".w1.scale");
            tab[2 * E + e] = S.p<void>(q + ".w3.weight"); tab[3 * E + e] = S.p<void>(q + ".w3.scale");
            tab[4 * E + e] = S.p<void>(q + ".w2.weight"); tab[5 * E + e] = S.p<void>(q + ".w2.scale");
        }
    }
    if (S.get(p + "ffn.gate.weight").shape[0] != E)
        throw std::runtime_error(p + "router rows != expert count (REAP surgery must prune the router too)");
    if (E > 512) throw std::runtime_error(p + ": > 512 routed experts (k_router's shared arrays)");
    w.ex.n_routed = E; m.n_routed = E;
    w.ex.tab.heal_gain = S.opt<float>(p + "ffn.heal_gain");
    CK(cudaMalloc(&w.ex.dev_tables, tab.size() * sizeof(void*)));
    CK(cudaMemcpy(w.ex.dev_tables, tab.data(), tab.size() * sizeof(void*), cudaMemcpyHostToDevice));
    const uint8_t* const* T = (const uint8_t* const*)w.ex.dev_tables;
    w.ex.tab.w1 = T; w.ex.tab.w1s = T + E; w.ex.tab.w3 = T + 2 * E; w.ex.tab.w3s = T + 3 * E;
    w.ex.tab.w2 = T + 4 * E; w.ex.tab.w2s = T + 5 * E;
}

void load_model(Model& m, const std::string& dir, const LoadOpts& o) {
    auto want = [&](const std::string& n) {
        if (n.find(".engram.embed.") != std::string::npos) return false;        // row store
        if (n.compare(0, 7, "vision.") == 0 || n.compare(0, 8, "aligner.") == 0) return false;
        if (n.compare(0, 4, "mtp.") == 0) return false;                         // DSpark: not wired yet
        if (n.compare(0, 7, "layers.") == 0) {
            const int L = atoi(n.c_str() + 7);
            return L >= o.layer_lo && L <= o.layer_hi;
        }
        return o.embed_head;                                                    // embed, head, norm, image_*
    };
    m.store.reset(new Store(dir, want));
    if (!m.store->thor_json().empty()) {
        // thor_format.md obligation 2: the build must be for the keep-set this server is configured for
        const auto j = nlohmann::json::parse(m.store->thor_json());
        const std::string have = j.value("keepset_sha256", "");
        if (o.keepset_sha256.empty()) {
            if (!getenv("V41_NO_KEEPSET_CHECK"))
                throw std::runtime_error("thor build " + dir + " needs --keepset <s04/keepset.json> (keepset " + have +
                                         "); V41_NO_KEEPSET_CHECK=1 skips the check");
        } else if (o.keepset_sha256 != have)
            throw std::runtime_error("KEEPSET MISMATCH: build " + have + " != configured " + o.keepset_sha256);
        m.quant_format = "thor-v1 arm " + j.value("arm", std::string("?"));
    }
    m.layers.assign(N_LAYERS, LayerW{}); m.loaded.assign(N_LAYERS, false);
    for (int L = o.layer_lo; L <= o.layer_hi; ++L) { bind_layer(m, L, o.mxfp4_hw); m.loaded[L] = true; }
    if (o.embed_head) {
        m.embed = m.store->p<uint16_t>("embed.weight");
        m.head = m.store->p<uint16_t>("head.weight");
        m.norm = m.store->p<uint16_t>("norm.weight");
    }
}

// =========================================================================== state
State::~State() { for (void* p : allocs) cudaFree(p); }
template <class T> static T* dalloc(State& s, size_t n) {
    void* p = nullptr; CK(cudaMalloc(&p, n * sizeof(T) + 256)); CK(cudaMemset(p, 0, n * sizeof(T) + 256));
    s.allocs.push_back(p); s.bytes += n * sizeof(T); return (T*)p;
}
void alloc_state(State& s, int seqmax, int max_rows, size_t score_budget, int shared_rows) {
    if (seqmax > MAX_CTX) throw std::runtime_error("seqmax above 1M (YaRN x16 of 64K)");
    if (max_rows > WINDOW) throw std::runtime_error("max_rows > 128: the 256-slot window ring needs chunks <= 128");
    s.seqmax = seqmax; s.max_rows = max_rows;
    v41k::rope_set_tables();
    v41k::hc_reserve(max_rows);
    v41k::v41k_init((size_t)1 << 22);
    s.ring.assign(N_LAYERS, nullptr);
    for (int L = 0; L < N_LAYERS; ++L) s.ring[L] = dalloc<float>(s, (size_t)State::RING * HEAD_DIM);
    for (int L : KV_SOURCES) {
        State::Src& c = s.src[L];
        c.rows = seqmax / compress_ratio(L) + 1;
        c.ckv = dalloc<uint8_t>(s, (size_t)c.rows * 256); c.ckv_s = dalloc<uint8_t>(s, (size_t)c.rows * 32);
        c.ik = dalloc<uint8_t>(s, (size_t)c.rows * 64); c.ik_s = dalloc<uint8_t>(s, (size_t)c.rows * 4);
        c.kv_state = dalloc<float>(s, 2 * 512); c.score_state = dalloc<float>(s, 2 * 512);
    }
    const int R = max_rows;
    s.ld_cand = (seqmax + CAND_BLOCK - 1) / CAND_BLOCK;
    s.shared_rows = std::max(shared_rows, R);
    s.topk = dalloc<int>(s, (size_t)s.shared_rows * INDEX_TOPK);
    s.cand = dalloc<uint8_t>(s, (size_t)s.shared_rows * s.ld_cand);
    s.d_pos = dalloc<int>(s, R); s.d_ids = dalloc<int>(s, R); s.d_img = dalloc<uint8_t>(s, R); s.d_noimg = dalloc<uint8_t>(s, R);
    s.h = dalloc<float>(s, (size_t)R * HC_DIM); s.h2 = dalloc<float>(s, (size_t)R * HC_DIM);
    s.pre = dalloc<float>(s, R * 4); s.post = dalloc<float>(s, R * 4); s.comb = dalloc<float>(s, R * 16);
    s.pre_next = dalloc<float>(s, R * 4);
    s.x = dalloc<float>(s, (size_t)R * DIM); s.xa = dalloc<float>(s, (size_t)R * DIM); s.y = dalloc<float>(s, (size_t)R * DIM);
    s.qa = dalloc<float>(s, (size_t)R * Q_LORA); s.qr = dalloc<float>(s, (size_t)R * Q_LORA);
    s.q = dalloc<float>(s, (size_t)R * N_HEADS * HEAD_DIM); s.o = dalloc<float>(s, (size_t)R * N_HEADS * HEAD_DIM);
    s.kv = dalloc<float>(s, (size_t)R * HEAD_DIM); s.og = dalloc<float>(s, (size_t)R * O_GROUPS * O_LORA);
    s.ck = dalloc<float>(s, (size_t)R * HEAD_DIM); s.cs = dalloc<float>(s, (size_t)R * HEAD_DIM);
    s.lat = dalloc<float>(s, (size_t)R * HEAD_DIM); s.ik = dalloc<float>(s, (size_t)R * INDEX_HEAD_DIM);
    s.iq = dalloc<float>(s, (size_t)R * INDEX_N_HEADS * INDEX_HEAD_DIM); s.iw = dalloc<float>(s, (size_t)R * INDEX_N_HEADS);
    s.rw = dalloc<float>(s, (size_t)R * N_ACT); s.ridx = dalloc<int>(s, (size_t)R * N_ACT);
    s.eg_e = dalloc<float>(s, (size_t)R * ENGRAM_KV_IN); s.eg_kv = dalloc<float>(s, (size_t)R * ENGRAM_KV_OUT);
    s.logits = dalloc<float>(s, VOCAB);
    s.xq = dalloc<uint8_t>(s, (size_t)R * N_HEADS * HEAD_DIM); s.xs = dalloc<float>(s, (size_t)R * N_HEADS * HEAD_DIM / 32);
    s.pk = dalloc<uint8_t>(s, (size_t)R * 256); s.ps = dalloc<uint8_t>(s, (size_t)R * 32);
    s.valid = dalloc<int>(s, R);
    s.moe_scratch = dalloc<uint8_t>(s, v41k::moe_scratch_bytes(R, N_ACT));
    s.cand_scratch = dalloc<uint8_t>(s, v41k::candidate_scratch_bytes(R, s.ld_cand));
    s.ld_score = seqmax;
    const int rs = (int)std::max<size_t>(1, std::min<size_t>(R, score_budget / ((size_t)seqmax * 4)));
    s.score = dalloc<float>(s, (size_t)rs * seqmax);
    s.score_rows_ = rs;
    reset_state(s, 0);
    CK(cudaDeviceSynchronize());
}
int max_chunk(const State& s) { return s.max_rows; }
__global__ static void k_fill(float* p, int n, float v) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) p[i] = v; }
void reset_state(State& s, cudaStream_t st) {
    for (int L : KV_SOURCES) {
        State::Src& c = s.src[L];
        cudaMemsetAsync(c.kv_state, 0, 2 * 512 * 4, st);
        k_fill<<<4, 256, 0, st>>>(c.score_state, 1024, -INFINITY);
    }
    for (int L = 0; L < N_LAYERS; ++L) cudaMemsetAsync(s.ring[L], 0, (size_t)State::RING * HEAD_DIM * 4, st);
}

// =========================================================================== forward
struct Rows {
    int M, r0;
    const int* pos; const int* ids; const uint8_t* img; const uint8_t* noimg;
    float* h; float* pre;
};

// Attention.forward (+ Compressor, Indexer, select_candidate_blocks) for `R` rows. x: attn_norm'd.
static void attention(const LayerW& w, State& s, const Rows& R, const float* x, float* out, cudaStream_t st) {
    const int M = R.M;
    const bool yarn = w.ratio > 0;
    // qr = q_norm(wq_a(x)); q = wq_b(qr); rope(q[..., -64:])
    v41k::act_quant_fp8(s.xq, s.xs, x, M, DIM, st);
    v41k::fp8_linear(s.qa, s.xq, s.xs, w.wq_a, w.wq_a_s, M, Q_LORA, DIM, true, st);
    v41k::rmsnorm(s.qr, s.qa, w.q_norm, M, Q_LORA, NORM_EPS, st);
    // window kv from the SAME x quantisation (s.xq still holds act_quant(x))
    v41k::fp8_linear(s.kv, s.xq, s.xs, w.wkv, w.wkv_s, M, HEAD_DIM, DIM, true, st);
    v41k::rmsnorm(s.kv, s.kv, w.kv_norm, M, HEAD_DIM, NORM_EPS, st);
    v41k::rope(s.kv, M, 1, HEAD_DIM, NOPE_DIM, R.pos, 0, false, yarn, st);
    v41k::fp8_sim(s.kv, M, HEAD_DIM, st);
    v41k::ring_store(s.ring[w.L], State::RING, s.kv, R.pos, M, st);
    // compressor (kv_source layers only): the pooled latent before RoPE
    const int* valid = nullptr;
    bool have_latent = false;
    if (w.kv_src) {
        if (w.ratio > 1) {
            v41k::bf16_linear(s.ck, x, w.cmp_wkv, M, HEAD_DIM, DIM, false, st);
            v41k::bf16_linear(s.cs, x, w.cmp_wgate, M, HEAD_DIM, DIM, false, st);
            v41k::compressor_pool2(s.lat, s.valid, s.src[w.L].kv_state, s.src[w.L].score_state, s.ck, s.cs, w.cmp_norm,
                                   R.pos, M, st);
            valid = s.valid;
        } else {
            v41k::bf16_linear(s.ck, x, w.cmp_wkv, M, HEAD_DIM, DIM, true, st);
            v41k::rmsnorm(s.lat, s.ck, w.cmp_norm, M, HEAD_DIM, NORM_EPS, st);
        }
        have_latent = true;
    }
    // indexer (index_source layers): owners append keys first, then score + select
    int* topk = s.topk + (size_t)(s.shared_base + R.r0) * INDEX_TOPK;
    if (w.idx_src) {
        State::Src& own = s.src[w.kv_owner];
        if (w.kv_src && have_latent) {
            v41k::bf16_linear(s.ik, s.lat, w.idx_wk, M, INDEX_HEAD_DIM, HEAD_DIM, true, st);
            v41k::rmsnorm(s.ik, s.ik, w.idx_k_norm, M, INDEX_HEAD_DIM, NORM_EPS, st);
            // a latent stands for the first token of its group: position pos + 1 - ratio
            v41k::rope(s.ik, M, 1, INDEX_HEAD_DIM, INDEX_HEAD_DIM - ROPE_DIM, R.pos, 1 - w.ratio, false, true, st);
            v41k::fp4_sim(s.ik, M, INDEX_HEAD_DIM, IDX_FP4_GRP, false, s.pk, s.ps, st);
            v41k::store_packed_rows(own.ik, own.ik_s, s.pk, s.ps, valid, R.pos, w.ratio, M, 64, 4, st);
        }
        v41k::act_quant_fp8(s.xq, s.xs, s.qr, M, Q_LORA, st);
        v41k::fp8_linear(s.iq, s.xq, s.xs, w.idx_wq_b, w.idx_wq_b_s, M, INDEX_N_HEADS * INDEX_HEAD_DIM, Q_LORA, true, st);
        v41k::rope(s.iq, M * INDEX_N_HEADS, INDEX_N_HEADS, INDEX_HEAD_DIM, INDEX_HEAD_DIM - ROPE_DIM, R.pos, 0, false, true, st);
        v41k::fp4_sim(s.iq, M * INDEX_N_HEADS, INDEX_HEAD_DIM, IDX_FP4_GRP, false, nullptr, nullptr, st);
        v41k::bf16_linear(s.iw, x, w.idx_wproj, M, INDEX_N_HEADS, DIM, true, st);
        v41k::scale_bf16(s.iw, (size_t)M * INDEX_N_HEADS,
                         1.0f / std::sqrt((float)INDEX_HEAD_DIM) / std::sqrt((float)INDEX_N_HEADS), st);
        uint8_t* cand = s.cand + (size_t)(s.shared_base + R.r0) * s.ld_cand;
        const int rs = s.score_rows_;
        for (int a = 0; a < M; a += rs) {
            const int mm = std::min(rs, M - a);
            v41k::index_scores(s.score, s.iq + (size_t)a * INDEX_N_HEADS * INDEX_HEAD_DIM, s.iw + (size_t)a * INDEX_N_HEADS,
                               own.ik, own.ik_s, w.uses_cand ? cand + (size_t)a * s.ld_cand : nullptr, R.pos + a,
                               w.ratio, mm, s.ld_score, st);
            if (w.cand_src)
                v41k::candidate_blocks(cand + (size_t)a * s.ld_cand, s.score, R.pos + a, w.ratio, mm, s.ld_score,
                                       s.ld_cand, s.cand_scratch, st);
            v41k::topk_rows(topk + (size_t)a * INDEX_TOPK, s.score, R.pos + a, w.ratio, mm, s.ld_score, INDEX_TOPK, 0,
                            nullptr, st);
        }
    }
    // compressed KV: RoPE the latent, fp4 (E4M3 per 16) and append to this source's cache
    if (w.kv_src && have_latent) {
        State::Src& own = s.src[w.L];
        v41k::rope(s.lat, M, 1, HEAD_DIM, NOPE_DIM, R.pos, 1 - w.ratio, false, true, st);
        v41k::fp4_sim(s.lat, M, HEAD_DIM, CKV_FP4_GRP, true, s.pk, s.ps, st);
        v41k::store_packed_rows(own.ckv, own.ckv_s, s.pk, s.ps, valid, R.pos, w.ratio, M, 256, 32, st);
    }
    // q (deferred to here only to share s.xq): q = wq_b(qr), rope
    v41k::act_quant_fp8(s.xq, s.xs, s.qr, M, Q_LORA, st);
    v41k::fp8_linear(s.q, s.xq, s.xs, w.wq_b, w.wq_b_s, M, N_HEADS * HEAD_DIM, Q_LORA, true, st);
    v41k::rope(s.q, M * N_HEADS, N_HEADS, HEAD_DIM, NOPE_DIM, R.pos, 0, false, yarn, st);
    // o = sparse_attn(q, [window | compressed], sink, idxs); inverse rope on o
    const State::Src* own = w.ratio ? &s.src[w.kv_owner] : nullptr;
    v41k::sparse_attn(s.o, s.q, s.ring[w.L], State::RING, own ? own->ckv : nullptr, own ? own->ckv_s : nullptr,
                      w.ratio ? topk : nullptr, w.ratio ? INDEX_TOPK : 0, 0, w.attn_sink, R.pos, M,
                      1.0f / std::sqrt((float)HEAD_DIM), st);
    v41k::rope(s.o, M * N_HEADS, N_HEADS, HEAD_DIM, NOPE_DIM, R.pos, 0, true, yarn, st);
    // wo_a: block-diagonal over 8 groups on act_quant(o); wo_b
    v41k::act_quant_fp8(s.xq, s.xs, s.o, M, N_HEADS * HEAD_DIM, st);
    v41k::fp8_linear(s.og, s.xq, s.xs, w.wo_a, w.wo_a_s, M, O_GROUPS * O_LORA, O_GROUP_IN, true, st, O_GROUPS);
    v41k::act_quant_fp8(s.xq, s.xs, s.og, M, O_GROUPS * O_LORA, st);
    v41k::fp8_linear(out, s.xq, s.xs, w.wo_b, w.wo_b_s, M, DIM, O_GROUPS * O_LORA, true, st);
}

// MoE.forward: router (bias_vl on image rows), routed experts in ascending order, + shared
static void moe(const LayerW& w, State& s, const Rows& R, const float* x, float* out, cudaStream_t st) {
    const int M = R.M;
    v41k::router(s.rw, s.ridx, x, w.gate_w, w.gate_bias, w.gate_bias_vl, R.img, M, w.ex.n_routed, N_ACT, ROUTE_SCALE, st);
    v41k::act_quant_fp8(s.xq, s.xs, x, M, DIM, st);
    cudaMemsetAsync(out, 0, (size_t)M * DIM * 4, st);
    v41k::moe_routed(out, s.xq, s.xs, s.rw, s.ridx, w.ex.tab, M, N_ACT, s.moe_scratch, st);
    v41k::moe_shared_add(out, s.xq, s.xs, w.sh_w1, w.sh_w1s, w.sh_w3, w.sh_w3s, w.sh_w2, w.sh_w2s, M, s.moe_scratch, st);
}

static void engram(const LayerW& w, State& s, const Rows& R, const Engram& eg, cudaStream_t st) {
    const int slot = engram_slot(w.L);
    const uint8_t* rows = eg.store ? eg.store->arena_dev(slot) : eg.rows_dev[slot];
    if (!rows) return;                                               // Engram OFF (bench only)
    if (eg.store && eg.graph_wait)
        cuStreamWaitValue32((CUstream)st, (CUdeviceptr)eg.store->flag_dev(slot), 1, CU_STREAM_WAIT_VALUE_EQ);
    v41k::engram_dequant(s.eg_e, rows, eg.store ? eg.store->row_bytes() : 264, eg.store ? eg.store->format() : eg.fmt,
                         R.M, st);
    v41k::act_quant_fp8(s.xq, s.xs, s.eg_e, R.M, ENGRAM_KV_IN, st);
    v41k::fp8_linear(s.eg_kv, s.xq, s.xs, w.eg_wkv, w.eg_wkv_s, R.M, ENGRAM_KV_OUT, ENGRAM_KV_IN, true, st);
    v41k::engram_gate(R.h, s.eg_kv, w.eg_q, w.eg_k, R.noimg, R.M, NORM_EPS, st);
    if (eg.store && eg.graph_wait)
        cuStreamWriteValue32((CUstream)st, (CUdeviceptr)eg.store->flag_dev(slot), 0, CU_STREAM_WRITE_VALUE_DEFAULT);
}

// Block.forward with the deferred pre-mix: attention collapses with the mix the previous FFN made.
static void block(const LayerW& w, State& s, const Rows& R, const Engram& eg, cudaStream_t st) {
    const int M = R.M;
    if (engram_slot(w.L) >= 0) engram(w, s, R, eg, st);
    float* h2 = s.h2;                                                       // rows [0, M) of this call
    v41k::hc_mixes(s.pre_next, s.post, s.comb, R.h, w.hc_attn_fn, w.hc_attn_scale, w.hc_attn_base, M, st);
    v41k::hc_collapse(s.x, R.h, R.pre, M, st);
    v41k::rmsnorm(s.x, s.x, w.attn_norm, M, DIM, NORM_EPS, st);
    attention(w, s, R, s.x, s.xa, st);
    v41k::hc_post(h2, s.xa, R.h, s.post, s.comb, M, st);
    // FFN: its own mixes from h2; collapses with the ATTENTION's pre
    float* attn_pre = s.pre_next;
    v41k::hc_mixes(R.pre, s.post, s.comb, h2, w.hc_ffn_fn, w.hc_ffn_scale, w.hc_ffn_base, M, st);   // R.pre <- ffn_pre
    v41k::hc_collapse(s.x, h2, attn_pre, M, st);
    v41k::rmsnorm(s.x, s.x, w.ffn_norm, M, DIM, NORM_EPS, st);
    moe(w, s, R, s.x, s.y, st);
    v41k::hc_post(R.h, s.y, h2, s.post, s.comb, M, st);
}

int forward_layers(const Model& m, State& s, int lo, int hi, int M, int prompt_end, int pos0, const Engram& eg,
                   cudaStream_t st, const LayerHook& hook, int dec_margin) {
    int r0 = 0;
    for (int L = lo; L <= hi; ++L) {
        if (!m.loaded[L]) throw std::runtime_error("layer " + std::to_string(L) + " not loaded");
        if (prompt_end > 0 && L > ENC_LAST) {
            const int first = prompt_end - dec_span(L) - dec_margin;        // oldest position L must see
            r0 = std::max(r0, std::min(M, first - pos0));
        }
        if (r0 >= M) continue;
        Rows R{M - r0, r0, s.d_pos + r0, s.d_ids + r0, s.d_img + r0, s.d_noimg + r0,
               s.h + (size_t)r0 * HC_DIM, s.pre + (size_t)r0 * 4};
        block(m.layers[L], s, R, eg, st);
        if (hook) hook(L, R.h, R.pre, R.M, st);
    }
    return r0;
}

__global__ static void k_noimg(uint8_t* o, const uint8_t* i, int M) { int t = threadIdx.x + blockIdx.x * blockDim.x; if (t < M) o[t] = !i[t]; }
void embed_rows(const Model& m, State& s, int M, cudaStream_t st) {
    v41k::embed_expand(s.h, m.embed, s.d_ids, M, st);
    v41k::hc_identity_pre(s.pre, M, st);
    k_noimg<<<(M + 127) / 128, 128, 0, st>>>(s.d_noimg, s.d_img, M);
}
void head_row(const Model& m, State& s, int r, cudaStream_t st) {
    v41k::hc_collapse(s.x, s.h + (size_t)r * HC_DIM, s.pre + (size_t)r * 4, 1, st);
    v41k::rmsnorm(s.x, s.x, m.norm, 1, DIM, NORM_EPS, st);
    v41k::bf16_linear(s.logits, s.x, m.head, 1, VOCAB, DIM, false, st);
}
void argmax_logits(const State& s, int* d_tok, cudaStream_t st) { v41k::argmax(d_tok, s.logits, VOCAB, st); }

} // namespace v41m
