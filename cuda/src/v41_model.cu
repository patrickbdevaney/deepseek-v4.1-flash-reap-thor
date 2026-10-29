// v41_model.cu — weights, state and the layer forward. The forward follows inference/model.py
// line by line; each block below names the reference code it reproduces.
#include "v41_model.h"
#include "v41_spec.h"
#include "safetensors.h"
#include <cuda.h>
#include <dirent.h>
#include <fcntl.h>
#include <unistd.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <functional>
#include <memory>
#include <set>
#include <unordered_map>
#include "third_party/json.hpp"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) throw std::runtime_error(std::string(#x) + ": " + cudaGetErrorString(e_)); } while (0)

namespace v41m {

using namespace v41;

// =========================================================================== Store
Store::Store(const std::string& dir, const Filter& want, const Rename& rename) {
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
        std::string idx;
        try { idx = st::read_file(dir + "/model.safetensors.index.json"); } catch (...) {}
        if (!idx.empty()) {
            const auto j = nlohmann::json::parse(idx);
            std::set<std::string> u;
            for (auto it = j["weight_map"].begin(); it != j["weight_map"].end(); ++it) u.insert(it.value().get<std::string>());
            files.assign(u.begin(), u.end());
        } else {                                     // a plain directory of *.safetensors (an s11 head dir)
            DIR* d = opendir(dir.c_str());
            if (!d) throw std::runtime_error(dir + ": neither thor.json, model.safetensors.index.json nor a directory");
            while (dirent* e = readdir(d)) files.push_back(e->d_name);
            closedir(d);
            std::sort(files.begin(), files.end());
        }
    }
    struct Item { std::string name; int fd; size_t off, n; std::string dtype; std::vector<int64_t> shape; };
    std::vector<Item> items; size_t total = 0;
    for (const auto& f : files) {
        if (f.size() < 12 || f.compare(f.size() - 12, 12, ".safetensors") != 0) continue;
        std::unique_ptr<st::SafeTensors> sh;
        try { sh.reset(new st::SafeTensors(dir + "/" + f)); } catch (...) { continue; }
        bool any = false;
        for (auto& t : sh->all()) {
            const std::string nm = rename ? rename(f, t.first) : t.first;
            if (t.first.compare(0, 6, "__pad.") == 0 || !want(nm)) continue;   // thor-v1 padding tensors
            Item it{nm, sh->fd(), sh->dataFileOffset() + (size_t)(t.second.data - sh->dataStart()), t.second.nbytes,
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


// Routed experts under prefix `ep` ("layers.L.ffn.experts." or "mtp.i.ffn.experts."). Layouts:
//   thor-v1 stacked ggml K-quant  wX.qweight  U8 [n, out, row_bytes]  qtype Q2_K..Q4_K
//   thor-v1 stacked MXFP4_SRC     wX.weight/.scale  [n, out, in/2] / [n, out, in/32]
//   thor-v1 stacked trellis KT2/3 wX.qweight + .kt_scale F16 [n, out, in/128],
//                                 {w13,w2}.rht_sign U8 [in/8] (thor_format.md "Trellis experts")
//   HF checkpoint per expert      E.wX.weight/.scale (MXFP4 I8 + E8M0)
// qtype_of(name) gives a stacked tensor's qtype (thor.json, or an s11 head.json); empty = none known.
// Fills ex.tab's formats and device pointer tables; returns the expert count.
int bind_experts(const Store& S, const std::string& ep, const std::function<std::string(const std::string&)>& qtype_of,
                 bool hw, int inter, Experts& ex) {
    int E = 0;
    std::vector<const void*> tab;
    auto fmt_of = [&](const std::string& qt, v41k::ExpertFmt& f, int& bits) {
        if (qt == "MXFP4_SRC") { f = hw ? v41k::ExpertFmt::MXFP4_HW : v41k::ExpertFmt::MXFP4; bits = 4; }
        else if (qt == "Q2_K") { f = v41k::ExpertFmt::KQ; bits = 2; }
        else if (qt == "Q3_K") { f = v41k::ExpertFmt::KQ; bits = 3; }
        else if (qt == "Q4_K") { f = v41k::ExpertFmt::KQ; bits = 4; }
        else if (qt == "KT2") { f = v41k::ExpertFmt::KT; bits = 2; }
        else if (qt == "KT3") { f = v41k::ExpertFmt::KT; bits = 3; }
        else throw std::runtime_error(ep + ": unknown expert qtype '" + qt + "' -- refusing (thor_format.md obligation 1)");
    };
    if (S.has(ep + "w1.qweight") || S.has(ep + "w1.weight")) {
        const bool kq = S.has(ep + "w1.qweight");
        const char* part = kq ? ".qweight" : ".weight";
        std::string qt13 = "MXFP4_SRC", qt2 = "MXFP4_SRC";
        if (qtype_of) {
            qt13 = qtype_of(ep + "w1" + part);
            qt2 = qtype_of(ep + "w2" + part);
            if (qtype_of(ep + "w3" + part) != qt13) throw std::runtime_error(ep + ": w1/w3 qtypes differ");
        } else if (kq) throw std::runtime_error(ep + ": K-quant experts without a qtype manifest (thor.json / head.json)");
        fmt_of(qt13, ex.tab.fmt13, ex.tab.bits13);
        fmt_of(qt2, ex.tab.fmt2, ex.tab.bits2);
        const DevT& t1 = S.get(ep + "w1" + part);
        E = (int)t1.shape[0];
        tab.assign(6 * (size_t)E, nullptr);
        const char* proj[3] = {"w1", "w3", "w2"};
        const int slot[3] = {0, 2, 4};
        for (int j = 0; j < 3; ++j) {
            const DevT& t = S.get(ep + proj[j] + part);
            const int out = j < 2 ? inter : DIM, in = j < 2 ? DIM : inter;
            if ((int)t.shape[0] != E || (int)t.shape[1] != out) throw std::runtime_error(ep + proj[j] + ": bad stacked shape");
            const size_t stride = t.nbytes / E;
            const v41k::ExpertFmt f = j < 2 ? ex.tab.fmt13 : ex.tab.fmt2;
            const int bits = j < 2 ? ex.tab.bits13 : ex.tab.bits2;
            const DevT* sc = nullptr;
            if (f == v41k::ExpertFmt::KQ) {
                const int bb = bits == 2 ? 84 : bits == 3 ? 110 : 144;
                if (stride != (size_t)out * (in / 256) * bb) throw std::runtime_error(ep + proj[j] + ": row bytes do not match the qtype");
                // Q2_K/Q3_K: reorder to code | scale | d planes for the vector-load kernel (V41_KQ_SOA=0 keeps ggml order)
                if ((bits == 2 || bits == 3) && v41k::kq_mma() && out % 16 == 0 && (uintptr_t)t.p % 16 == 0 && stride % 16 == 0) {
                    static void* tmp = nullptr; static size_t tcap = 0;
                    if (stride > tcap) { if (tmp) cudaFree(tmp); CK(cudaMalloc(&tmp, stride)); tcap = stride; }
                    for (int e = 0; e < E; ++e) v41k::kq_mma_repack((uint8_t*)t.p + (size_t)e * stride, out, in, bits, tmp, 0);
                    CK(cudaDeviceSynchronize());
                    ex.tab.kq_mma = true;
                } else if ((bits == 2 || bits == 3) && v41k::kq_soa()) {
                    if ((uintptr_t)t.p % 16 || stride % 16)
                        throw std::runtime_error(ep + proj[j] + ": not 16-byte aligned for kq_soa (run with V41_KQ_SOA=0)");
                    static void* tmp = nullptr; static size_t tcap = 0;
                    if (stride > tcap) { if (tmp) cudaFree(tmp); CK(cudaMalloc(&tmp, stride)); tcap = stride; }
                    for (int e = 0; e < E; ++e) v41k::kq_soa_repack((uint8_t*)t.p + (size_t)e * stride, out, in, bits, tmp, 0);
                    CK(cudaDeviceSynchronize());
                    ex.tab.kq_soa = true;
                }
            } else if (f == v41k::ExpertFmt::KT) {
                if (stride != (size_t)out * (in / 128) * 16 * bits) throw std::runtime_error(ep + proj[j] + ": KT row bytes do not match the qtype");
                sc = &S.get(ep + proj[j] + ".kt_scale");
                if (sc->nbytes != (size_t)E * out * (in / 128) * 2) throw std::runtime_error(ep + proj[j] + ".kt_scale: bad size");
            } else {
                if (stride != (size_t)out * in / 2) throw std::runtime_error(ep + proj[j] + ": MXFP4_SRC shape mismatch");
                sc = &S.get(ep + proj[j] + ".scale");
            }
            for (int e = 0; e < E; ++e) {
                tab[(size_t)slot[j] * E + e] = (const char*)t.p + (size_t)e * stride;
                tab[(size_t)(slot[j] + 1) * E + e] = sc ? (const char*)sc->p + (size_t)e * (sc->nbytes / E) : nullptr;
            }
        }
        auto sign_of = [&](const char* pass, int in) -> const uint8_t* {
            const DevT& t = S.get(ep + pass + ".rht_sign");
            if (t.nbytes != (size_t)in / 8) throw std::runtime_error(ep + pass + ".rht_sign: expected " + std::to_string(in / 8) + " bytes");
            return (const uint8_t*)t.p;
        };
        if (ex.tab.fmt13 == v41k::ExpertFmt::KT) ex.tab.rht13 = sign_of("w13", DIM);
        if (ex.tab.fmt2 == v41k::ExpertFmt::KT) ex.tab.rht2 = sign_of("w2", inter);
    } else {
        while (S.has(ep + std::to_string(E) + ".w1.weight")) ++E;
        if (!E) throw std::runtime_error("no routed experts under " + ep);
        fmt_of("MXFP4_SRC", ex.tab.fmt13, ex.tab.bits13);
        fmt_of("MXFP4_SRC", ex.tab.fmt2, ex.tab.bits2);
        tab.assign(6 * (size_t)E, nullptr);
        for (int e = 0; e < E; ++e) {
            const std::string q = ep + std::to_string(e);
            tab[0 * E + e] = S.p<void>(q + ".w1.weight"); tab[1 * E + e] = S.p<void>(q + ".w1.scale");
            tab[2 * E + e] = S.p<void>(q + ".w3.weight"); tab[3 * E + e] = S.p<void>(q + ".w3.scale");
            tab[4 * E + e] = S.p<void>(q + ".w2.weight"); tab[5 * E + e] = S.p<void>(q + ".w2.scale");
        }
    }
    if (E > 512) throw std::runtime_error(ep + ": > 512 routed experts (k_router's shared arrays)");
    ex.n_routed = E;
    CK(cudaMalloc(&ex.dev_tables, tab.size() * sizeof(void*)));
    CK(cudaMemcpy(ex.dev_tables, tab.data(), tab.size() * sizeof(void*), cudaMemcpyHostToDevice));
    const uint8_t* const* T = (const uint8_t* const*)ex.dev_tables;
    ex.tab.w1 = T; ex.tab.w1s = T + E; ex.tab.w3 = T + 2 * E; ex.tab.w3s = T + 3 * E;
    ex.tab.w2 = T + 4 * E; ex.tab.w2s = T + 5 * E;
    return E;
}

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
    // an FP8 linear, or its NVFP4 overlay replacement (codes + e4m3/16 scales, registered with g)
    const Store* O = m.overlay.get();
    auto lin = [&](const std::string& n, const uint8_t*& W, const uint8_t*& Ws) {
        if (O && O->has(p + n + ".weight")) {
            W = O->p<uint8_t>(p + n + ".weight"); Ws = O->p<uint8_t>(p + n + ".weight_scale");
            float g = 0.f;
            CK(cudaMemcpy(&g, O->p<float>(p + n + ".weight_scale_2"), sizeof(float), cudaMemcpyDefault));
            const auto& sh = O->get(p + n + ".weight").shape;
            v41k::overlay_register(W, Ws, g, (int)sh[0], (int)sh[1] * 2);
        } else { W = u8(n + ".weight"); Ws = u8(n + ".scale"); }
    };
    auto f32 = [&](const std::string& n) { return S.p<float>(p + n); };
    w.attn_sink = f32("attn.attn_sink");
    w.attn_norm = b16("attn_norm.weight"); w.ffn_norm = b16("ffn_norm.weight");
    w.q_norm = b16("attn.q_norm.weight"); w.kv_norm = b16("attn.kv_norm.weight");
    lin("attn.wq_a", w.wq_a, w.wq_a_s);
    lin("attn.wq_b", w.wq_b, w.wq_b_s);
    lin("attn.wkv", w.wkv, w.wkv_s);
    lin("attn.wo_a", w.wo_a, w.wo_a_s);
    lin("attn.wo_b", w.wo_b, w.wo_b_s);
    if (!(O && O->has(p + "attn.wo_a.weight")) && S.get(p + "attn.wo_a.weight").dtype != "F8_E4M3")
        throw std::runtime_error(p + "attn.wo_a is not FP8 (a convert.py output? load the HF checkpoint layout)");
    if (w.kv_src) {
        w.cmp_wkv = b16("attn.compressor.wkv.weight"); w.cmp_norm = b16("attn.compressor.norm.weight");
        if (w.ratio > 1) w.cmp_wgate = b16("attn.compressor.wgate.weight");
    }
    if (w.idx_src) {
        lin("attn.indexer.wq_b", w.idx_wq_b, w.idx_wq_b_s);
        w.idx_wproj = b16("attn.indexer.weights_proj.weight");
        if (w.kv_src) { w.idx_wk = b16("attn.indexer.wk.weight"); w.idx_k_norm = b16("attn.indexer.k_norm.weight"); }
    }
    w.hc_attn_fn = f32("hc_attn_fn"); w.hc_attn_base = f32("hc_attn_base"); w.hc_attn_scale = f32("hc_attn_scale");
    w.hc_ffn_fn = f32("hc_ffn_fn"); w.hc_ffn_base = f32("hc_ffn_base"); w.hc_ffn_scale = f32("hc_ffn_scale");
    w.gate_w = b16("ffn.gate.weight"); w.gate_bias = f32("ffn.gate.bias"); w.gate_bias_vl = S.opt<float>(p + "ffn.gate.bias_vl");
    lin("ffn.shared_experts.w1", w.sh_w1, w.sh_w1s);
    lin("ffn.shared_experts.w2", w.sh_w2, w.sh_w2s);
    lin("ffn.shared_experts.w3", w.sh_w3, w.sh_w3s);
    if (engram_slot(L) >= 0) {
        w.eg_wkv = u8("engram.wkv.weight"); w.eg_wkv_s = u8("engram.wkv.scale");
        w.eg_q = b16("engram.q_weight"); w.eg_k = b16("engram.k_weight");
    }
    const std::string ep = p + "ffn.experts.";
    std::function<std::string(const std::string&)> qtype_of;
    if (!S.thor_json().empty()) {
        auto j = std::make_shared<nlohmann::json>(nlohmann::json::parse(S.thor_json()));
        qtype_of = [j](const std::string& n) {
            const auto& T = (*j)["tensors"];
            if (!T.contains(n) || !T[n].contains("qtype") || !T[n]["qtype"].is_string()) return std::string();
            return T[n]["qtype"].get<std::string>();
        };
    }
    const int E = bind_experts(S, ep, qtype_of, hw, MOE_INTER, w.ex);
    if (S.get(p + "ffn.gate.weight").shape[0] != E)
        throw std::runtime_error(p + "router rows != expert count (REAP surgery must prune the router too)");
    m.n_routed = E;
    w.ex.tab.heal_gain = S.opt<float>(p + "ffn.heal_gain");
}

void load_model(Model& m, const std::string& dir, const LoadOpts& o) {
    // NVFP4 dense overlay first: its tensor names decide which FP8 weights are never loaded
    std::set<std::string> ov_bases;                       // "layers.L.attn.wq_a" etc.
    if (!o.overlay_dir.empty()) {
        auto rename = [](const std::string& f, const std::string& t) {
            return "layers." + std::to_string(atoi(f.c_str() + 1)) + "." + t;      // L07.safetensors -> layers.7.
        };
        auto want_ov = [&](const std::string& n) {
            if (n.compare(0, 7, "layers.") != 0) return false;
            const int L = atoi(n.c_str() + 7);
            return L >= o.layer_lo && L <= o.layer_hi;
        };
        m.overlay.reset(new Store(o.overlay_dir, want_ov, rename));
        if (!m.overlay->count()) throw std::runtime_error("overlay " + o.overlay_dir + ": no tensors loaded");
        for (int L = o.layer_lo; L <= o.layer_hi; ++L)
            for (const char* b : {"attn.wq_a", "attn.wq_b", "attn.wkv", "attn.wo_a", "attn.wo_b", "attn.indexer.wq_b",
                                  "ffn.shared_experts.w1", "ffn.shared_experts.w2", "ffn.shared_experts.w3"}) {
                const std::string base = "layers." + std::to_string(L) + "." + b;
                if (m.overlay->has(base + ".weight")) ov_bases.insert(base);
            }
    }
    auto want = [&](const std::string& n) {
        if (!ov_bases.empty()) {                          // FP8 twin of an overlaid weight: skip (.weight and .scale)
            const size_t d = n.rfind('.');
            if (d != std::string::npos && ov_bases.count(n.substr(0, d))) return false;
        }
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
    if (m.overlay) m.quant_format += " + NVFP4 dense overlay (" + std::to_string(ov_bases.size()) + " tensors, " +
                                     std::to_string(m.overlay->gib()).substr(0, 5) + " GiB)";
    m.layers.assign(N_LAYERS, LayerW{}); m.loaded.assign(N_LAYERS, false);
    for (int L = o.layer_lo; L <= o.layer_hi; ++L) { bind_layer(m, L, o.mxfp4_hw); m.loaded[L] = true; }
    if (o.embed_head) {
        m.embed = m.store->p<uint16_t>("embed.weight");
        m.head = m.store->p<uint16_t>("head.weight");
        // coalesced head GEMV (DecBF16P): 117 -> 221 GB/s in bench_dense, bitwise equal logits
        if (!getenv("V41_HEAD_NOPERM")) { v41k::bf16_permute(const_cast<uint16_t*>(m.head), VOCAB, DIM, 0); CK(cudaDeviceSynchronize()); }
        m.norm = m.store->p<uint16_t>("norm.weight");
    }
}

// =========================================================================== state
State::~State() {
    for (void* p : allocs) cudaFree(p);
    if (ev_fork) cudaEventDestroy(ev_fork);
    if (ev_join) cudaEventDestroy(ev_join);
    if (side) cudaStreamDestroy(side);
    if (ev_hf) cudaEventDestroy(ev_hf);
    if (ev_hj) cudaEventDestroy(ev_hj);
    if (hcs) cudaStreamDestroy(hcs);
    if (ev_kf) cudaEventDestroy(ev_kf);
    if (ev_kj) cudaEventDestroy(ev_kj);
    if (kvs) cudaStreamDestroy(kvs);
}
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
    v41k::sparse_attn_reserve(max_rows);
    v41k::v41k_init((size_t)1 << 22);
    s.ring.assign(N_LAYERS, nullptr);
    for (int L = 0; L < N_LAYERS; ++L) s.ring[L] = dalloc<float>(s, (size_t)State::RING * HEAD_DIM);
    for (int L : KV_SOURCES) {
        State::Src& c = s.src[L];
        c.rows = seqmax / compress_ratio(L) + 1;
        c.ckv = dalloc<uint8_t>(s, (size_t)c.rows * 256); c.ckv_s = dalloc<uint8_t>(s, (size_t)c.rows * 32);
        c.ik = dalloc<uint8_t>(s, (size_t)c.rows * 64); c.ik_s = dalloc<uint8_t>(s, (size_t)c.rows * 4);
        c.kv_state = dalloc<float>(s, 2 * 512); c.score_state = dalloc<float>(s, 2 * 512);
        if (compress_ratio(L) == 2) c.pool_hist = dalloc<float>(s, (size_t)max_rows * 1024);
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
    s.rscr = dalloc<float>(s, v41k::router_scratch_bytes(R, 512) / sizeof(float));
    s.eg_e = dalloc<float>(s, (size_t)R * ENGRAM_KV_IN); s.eg_kv = dalloc<float>(s, (size_t)R * ENGRAM_KV_OUT);
    s.logits = dalloc<float>(s, (size_t)VOCAB * R);              // row r of a verify at s.logits + r * VOCAB
    s.d_argmax = dalloc<int>(s, R);
    s.xq = dalloc<uint8_t>(s, (size_t)R * N_HEADS * HEAD_DIM); s.xs = dalloc<float>(s, (size_t)R * N_HEADS * HEAD_DIM / 32);
    s.pk = dalloc<uint8_t>(s, (size_t)R * 256); s.ps = dalloc<uint8_t>(s, (size_t)R * 32);
    s.valid = dalloc<int>(s, R);
    s.moe_scratch = dalloc<uint8_t>(s, v41k::moe_scratch_bytes(R, N_ACT));
    if (!getenv("V41_NO_SIDE_STREAM")) {
        s.sh_scratch = dalloc<uint8_t>(s, v41k::moe_scratch_bytes(R, 1));
        s.sh_o = dalloc<float>(s, (size_t)R * DIM);
        CK(cudaStreamCreateWithFlags(&s.side, cudaStreamNonBlocking));
        CK(cudaEventCreateWithFlags(&s.ev_fork, cudaEventDisableTiming));
        CK(cudaEventCreateWithFlags(&s.ev_join, cudaEventDisableTiming));
    }
    if (!(getenv("V41_KV_OVERLAP") && *getenv("V41_KV_OVERLAP") == '0')) {
        CK(cudaStreamCreateWithFlags(&s.kvs, cudaStreamNonBlocking));
        CK(cudaEventCreateWithFlags(&s.ev_kf, cudaEventDisableTiming));
        CK(cudaEventCreateWithFlags(&s.ev_kj, cudaEventDisableTiming));
        v41k::v41k_stream_scratch(s.kvs);
    }
    s.xq2 = dalloc<uint8_t>(s, (size_t)R * Q_LORA); s.xs2 = dalloc<float>(s, (size_t)R * Q_LORA / 32);
    if (s.side) v41k::v41k_stream_scratch(s.side);
    if (!(getenv("V41_HC_OVERLAP") && *getenv("V41_HC_OVERLAP") == '0')) {
        CK(cudaStreamCreateWithFlags(&s.hcs, cudaStreamNonBlocking));
        CK(cudaEventCreateWithFlags(&s.ev_hf, cudaEventDisableTiming));
        CK(cudaEventCreateWithFlags(&s.ev_hj, cudaEventDisableTiming));
    }
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
    // Two branches after act_quant(x). KV (s.kvs): window kv, compressor, indexer keys, compressed-KV
    // append. Q (st): wq_a, q_norm, indexer queries, wq_b. They join before index_scores / sparse_attn.
    // Same kernels on the same data in the same per-buffer order, so bit-identical; the q branch
    // quantises qr into xq2 because the KV branch still reads act_quant(x) from xq.
    v41k::act_quant_fp8(s.xq, s.xs, x, M, DIM, st);
    cudaStream_t ks = s.kvs ? s.kvs : st;
    if (s.kvs) { CK(cudaEventRecord(s.ev_kf, st)); CK(cudaStreamWaitEvent(s.kvs, s.ev_kf, 0)); }
    // ---- KV branch
    v41k::fp8_linear(s.kv, s.xq, s.xs, w.wkv, w.wkv_s, M, HEAD_DIM, DIM, true, ks);
    v41k::rmsnorm(s.kv, s.kv, w.kv_norm, M, HEAD_DIM, NORM_EPS, ks);
    v41k::rope(s.kv, M, 1, HEAD_DIM, NOPE_DIM, R.pos, 0, false, yarn, ks);
    v41k::fp8_sim(s.kv, M, HEAD_DIM, ks);
    v41k::ring_store(s.ring[w.L], State::RING, s.kv, R.pos, M, ks);
    // compressor (kv_source layers only): the pooled latent before RoPE
    const int* valid = nullptr;
    bool have_latent = false;
    if (w.kv_src) {
        if (w.ratio > 1) {
            v41k::bf16_linear(s.ck, x, w.cmp_wkv, M, HEAD_DIM, DIM, false, ks);
            v41k::bf16_linear(s.cs, x, w.cmp_wgate, M, HEAD_DIM, DIM, false, ks);
            v41k::compressor_pool2(s.lat, s.valid, s.src[w.L].kv_state, s.src[w.L].score_state, s.ck, s.cs, w.cmp_norm,
                                   R.pos, M, ks, s.src[w.L].pool_hist + (size_t)R.r0 * 1024);
            valid = s.valid;
        } else {
            v41k::bf16_linear(s.ck, x, w.cmp_wkv, M, HEAD_DIM, DIM, true, ks);
            v41k::rmsnorm(s.lat, s.ck, w.cmp_norm, M, HEAD_DIM, NORM_EPS, ks);
        }
        have_latent = true;
    }
    // indexer keys (owners append before anyone scores)
    if (w.idx_src && w.kv_src && have_latent) {
        State::Src& own = s.src[w.kv_owner];
        v41k::bf16_linear(s.ik, s.lat, w.idx_wk, M, INDEX_HEAD_DIM, HEAD_DIM, true, ks);
        v41k::rmsnorm(s.ik, s.ik, w.idx_k_norm, M, INDEX_HEAD_DIM, NORM_EPS, ks);
        // a latent stands for the first token of its group: position pos + 1 - ratio
        v41k::rope(s.ik, M, 1, INDEX_HEAD_DIM, INDEX_HEAD_DIM - ROPE_DIM, R.pos, 1 - w.ratio, false, true, ks);
        v41k::fp4_sim(s.ik, M, INDEX_HEAD_DIM, IDX_FP4_GRP, false, s.pk, s.ps, ks);
        v41k::store_packed_rows(own.ik, own.ik_s, s.pk, s.ps, valid, R.pos, w.ratio, M, 64, 4, ks);
    }
    // compressed KV: RoPE the latent, fp4 (E4M3 per 16) and append to this source's cache
    if (w.kv_src && have_latent) {
        State::Src& own = s.src[w.L];
        v41k::rope(s.lat, M, 1, HEAD_DIM, NOPE_DIM, R.pos, 1 - w.ratio, false, true, ks);
        v41k::fp4_sim(s.lat, M, HEAD_DIM, CKV_FP4_GRP, true, s.pk, s.ps, ks);
        v41k::store_packed_rows(own.ckv, own.ckv_s, s.pk, s.ps, valid, R.pos, w.ratio, M, 256, 32, ks);
    }
    if (s.kvs) CK(cudaEventRecord(s.ev_kj, s.kvs));
    bool joined = !s.kvs;
    auto join = [&] { if (!joined) { CK(cudaStreamWaitEvent(st, s.ev_kj, 0)); joined = true; } };
    // ---- Q branch: qr = q_norm(wq_a(x))
    v41k::fp8_linear(s.qa, s.xq, s.xs, w.wq_a, w.wq_a_s, M, Q_LORA, DIM, true, st);
    v41k::rmsnorm(s.qr, s.qa, w.q_norm, M, Q_LORA, NORM_EPS, st);
    uint8_t* qxq = s.kvs ? s.xq2 : s.xq; float* qxs = s.kvs ? s.xs2 : s.xs;
    int* topk = s.topk + (size_t)(s.shared_base + R.r0) * INDEX_TOPK;
    if (w.idx_src) {
        State::Src& own = s.src[w.kv_owner];
        v41k::act_quant_fp8(qxq, qxs, s.qr, M, Q_LORA, st);
        v41k::fp8_linear(s.iq, qxq, qxs, w.idx_wq_b, w.idx_wq_b_s, M, INDEX_N_HEADS * INDEX_HEAD_DIM, Q_LORA, true, st);
        v41k::rope(s.iq, M * INDEX_N_HEADS, INDEX_N_HEADS, INDEX_HEAD_DIM, INDEX_HEAD_DIM - ROPE_DIM, R.pos, 0, false, true, st);
        v41k::fp4_sim(s.iq, M * INDEX_N_HEADS, INDEX_HEAD_DIM, IDX_FP4_GRP, false, nullptr, nullptr, st);
        v41k::bf16_linear(s.iw, x, w.idx_wproj, M, INDEX_N_HEADS, DIM, true, st);
        v41k::scale_bf16(s.iw, (size_t)M * INDEX_N_HEADS,
                         1.0f / std::sqrt((float)INDEX_HEAD_DIM) / std::sqrt((float)INDEX_N_HEADS), st);
        join();                                      // own.ik
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
    // q = wq_b(qr), rope
    v41k::act_quant_fp8(qxq, qxs, s.qr, M, Q_LORA, st);
    v41k::fp8_linear(s.q, qxq, qxs, w.wq_b, w.wq_b_s, M, N_HEADS * HEAD_DIM, Q_LORA, true, st);
    v41k::rope(s.q, M * N_HEADS, N_HEADS, HEAD_DIM, NOPE_DIM, R.pos, 0, false, yarn, st);
    join();                                          // ring, compressed KV
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
    if (s.side) {                                    // shared expert concurrently with the router and routed experts
        v41k::act_quant_fp8(s.xq, s.xs, x, M, DIM, st);
        CK(cudaEventRecord(s.ev_fork, st)); CK(cudaStreamWaitEvent(s.side, s.ev_fork, 0));
        v41k::moe_shared(s.sh_o, s.xq, s.xs, w.sh_w1, w.sh_w1s, w.sh_w3, w.sh_w3s, w.sh_w2, w.sh_w2s, M, s.sh_scratch, s.side);
        CK(cudaEventRecord(s.ev_join, s.side));
        v41k::router(s.rw, s.ridx, x, w.gate_w, w.gate_bias, w.gate_bias_vl, R.img, M, w.ex.n_routed, N_ACT, ROUTE_SCALE, st, s.rscr);
        cudaMemsetAsync(out, 0, (size_t)M * DIM * 4, st);
        v41k::moe_routed(out, s.xq, s.xs, s.rw, s.ridx, w.ex.tab, M, N_ACT, s.moe_scratch, st);
        CK(cudaStreamWaitEvent(st, s.ev_join, 0));
        v41k::add_bf16(out, s.sh_o, (size_t)M * DIM, st);
        return;
    }
    v41k::router(s.rw, s.ridx, x, w.gate_w, w.gate_bias, w.gate_bias_vl, R.img, M, w.ex.n_routed, N_ACT, ROUTE_SCALE, st, s.rscr);
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
    const int cap = eg.store ? eg.store->capacity() : eg.rows_cap;
    if (R.M > cap) throw std::runtime_error("engram: " + std::to_string(R.M) + " rows > arena capacity " + std::to_string(cap));
    v41k::engram_dequant(s.eg_e, rows, cap, eg.store ? eg.store->format() : eg.fmt, R.M, st);
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
    // The collapse uses the PREVIOUS mix's pre, so this sub-layer's mixes (pre_next/post/comb) are first
    // read by hc_post: with s.hcs they run beside collapse/norm/attention (same kernels, same data).
    auto fork = [&] { if (s.hcs) { CK(cudaEventRecord(s.ev_hf, st)); CK(cudaStreamWaitEvent(s.hcs, s.ev_hf, 0)); } };
    auto join = [&] { if (s.hcs) { CK(cudaEventRecord(s.ev_hj, s.hcs)); CK(cudaStreamWaitEvent(st, s.ev_hj, 0)); } };
    cudaStream_t hst = s.hcs ? s.hcs : st;
    fork();
    v41k::hc_mixes(s.pre_next, s.post, s.comb, R.h, w.hc_attn_fn, w.hc_attn_scale, w.hc_attn_base, M, hst);
    v41k::hc_collapse(s.x, R.h, R.pre, M, st);
    v41k::rmsnorm(s.x, s.x, w.attn_norm, M, DIM, NORM_EPS, st);
    attention(w, s, R, s.x, s.xa, st);
    join();
    v41k::hc_post(h2, s.xa, R.h, s.post, s.comb, M, st);
    // FFN: its own mixes from h2; collapses with the ATTENTION's pre
    float* attn_pre = s.pre_next;
    fork();
    v41k::hc_mixes(R.pre, s.post, s.comb, h2, w.hc_ffn_fn, w.hc_ffn_scale, w.hc_ffn_base, M, hst);   // R.pre <- ffn_pre
    v41k::hc_collapse(s.x, h2, attn_pre, M, st);
    v41k::rmsnorm(s.x, s.x, w.ffn_norm, M, DIM, NORM_EPS, st);
    moe(w, s, R, s.x, s.y, st);
    join();
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

void enable_head2(Model& m, State& s, int cap, float delta, cudaStream_t st) {
    if (!m.head4) {
        CK(cudaMalloc(&m.head4, (size_t)VOCAB * DIM / 2)); CK(cudaMalloc(&m.head4s, (size_t)VOCAB * DIM / 16));
        m.head4g = v41k::nvfp4_from_bf16(m.head4, m.head4s, m.head, VOCAB, DIM, v41k::bf16_permuted(m.head), st);
    }
    if (!s.hd2.alog && !s.hd2.alloc(s.max_rows, VOCAB, cap, delta)) throw std::runtime_error("head2 scratch alloc failed");
    if (!s.d_hfb) CK(cudaMalloc(&s.d_hfb, s.max_rows * sizeof(int)));
    CK(cudaDeviceSynchronize());
}
void head_row_greedy(const Model& m, State& s, int r, int* d_tok, cudaStream_t st) {
    v41k::hc_collapse(s.x, s.h + (size_t)r * HC_DIM, s.pre + (size_t)r * 4, 1, st);
    v41k::rmsnorm(s.x, s.x, m.norm, 1, DIM, NORM_EPS, st);
    v41k::head2_argmax(d_tok, 1, s.d_hfb, s.x, 1, m.head, m.head4, m.head4s, m.head4g, s.hd2, VOCAB, DIM, st);
}
void head_rows_greedy(const Model& m, State& s, int r0, int n, cudaStream_t st) {
    v41k::hc_collapse(s.x, s.h + (size_t)r0 * HC_DIM, s.pre + (size_t)r0 * 4, n, st);
    v41k::rmsnorm(s.x, s.x, m.norm, n, DIM, NORM_EPS, st);
    v41k::head2_argmax(s.d_argmax, 1, s.d_hfb, s.x, n, m.head, m.head4, m.head4s, m.head4g, s.hd2, VOCAB, DIM, st);
}
void head_full_argmax(const Model& m, State& s, int row, int* d_out, cudaStream_t st) {
    v41k::bf16_linear(s.logits + (size_t)row * VOCAB, s.x + (size_t)row * DIM, m.head, 1, VOCAB, DIM, false, st);
    v41k::argmax(d_out, s.logits + (size_t)row * VOCAB, VOCAB, st);
}

void head_rows(const Model& m, State& s, int r0, int n, cudaStream_t st) {
    v41k::hc_collapse(s.x, s.h + (size_t)r0 * HC_DIM, s.pre + (size_t)r0 * 4, n, st);
    v41k::rmsnorm(s.x, s.x, m.norm, n, DIM, NORM_EPS, st);
    v41k::bf16_linear(s.logits, s.x, m.head, n, VOCAB, DIM, false, st);
    for (int r = 0; r < n; ++r) v41k::argmax(s.d_argmax + r, s.logits + (size_t)r * VOCAB, VOCAB, st);
}

void rollback_state(State& s, int row, int pos, cudaStream_t st) {
    if (!v41spec::pool2_needs_restore(pos)) return;  // odd: nothing a later row reads before rewriting it
    for (int L : KV_SOURCES) {
        State::Src& c = s.src[L];
        if (c.pool_hist) v41k::pool2_restore_even(c.kv_state, c.score_state, c.pool_hist + (size_t)row * 1024, st);
    }
}

} // namespace v41m
