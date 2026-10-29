// v41_engine.cu — dsv4srv::Engine (include/dsv4_engine.h) for DeepSeek-V4.1-Flash-1017 / REAP, over
// v41_model. Same contract as the 0731 engine the server was written against: load once, keep the
// KV across requests, reuse the longest common prefix, stream tokens through a callback.
//
// What differs from 0731:
//   * DSpark speculation (cfg.spec_k > 0) uses the checkpoint's own mtp.* blocks (v41_draft.h): every
//     main forward records the tap layers and feeds the drafter's window; a verify runs K+1 rows
//     [committed tok, K drafts] in one forward and keeps the longest matching prefix plus the
//     target's own next token. Rejected rows need only the ratio-2 compressor state put back
//     (v41m::rollback_state); everything else is rewritten before it is read. `verifies` counts
//     forwards either way.
//   * Engram: for every row, the 48 row ids are hashed on the host and the reads are issued to the
//     row store BEFORE the forward is launched; the forward waits on the store's flags on-stream
//     (cuStreamWaitValue32), so the M=1 step is one CUDA graph and layer 0 runs while rows land.
//   * Prefix reuse rounds the reused length DOWN TO EVEN: the ratio-2 compressors carry a 2-slot
//     state that is only self-contained at a pair boundary. Everything else (window ring, packed
//     caches, Engram history) is overwritten position by position, so truncation is exact.
//   * Prefill is "encoder-decoder" shaped (v41_model forward_layers): layers > 20 only run the last
//     dec_span(L) rows of the prompt. Exact, because nothing a later step reads at those layers
//     comes from older rows (docs/V41_DELTA.md §2).
#include "dsv4_engine.h"
#include "v41_model.h"
#include "v41_draft.h"
#include "v41_spec.h"
#include <cuda.h>
#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <map>
#include <random>
#include <sstream>
#include "third_party/json.hpp"
#include <stdexcept>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) throw std::runtime_error(std::string(#x) + ": " + cudaGetErrorString(e_)); } while (0)

namespace dsv4srv {

using clk = std::chrono::steady_clock;
static double ms_since(clk::time_point t) { return std::chrono::duration<double, std::milli>(clk::now() - t).count(); }

struct Engine::Impl {
    EngineConfig cfg;
    v41m::Model m;
    v41m::State s;
    EngramStore store;
    bool engram_on = false;
    std::unique_ptr<EngramHasher> hasher;
    v41m::Engram eg;
    cudaStream_t st = nullptr;
    std::vector<int> ctx;
    int kv_len = 0;                 // positions whose KV is written at every layer (ctx may hold one more)
    // Decoder layers skip prompt rows older than prompt_end - dec_span(L) - dec_margin, so their
    // window KV below that is stale: a reuse must restart at or above reuse_floor, else from 0.
    int dec_margin = 256, reuse_floor = 0;
    // Highest position any forward has written since the last reset (rejected verify rows included).
    // A window slot holds position q until q + RING is written: reuse at k needs k-127.. intact.
    int hi_written = -1;
    bool loaded = false, prefix_cache = true;
    int chunk = 64;
    // pinned staging
    int *h_pos = nullptr, *h_ids = nullptr, *h_tok = nullptr, *d_tok = nullptr;
    uint8_t* h_img = nullptr;
    std::vector<int64_t> hid;
    std::vector<float> h_logits;
    // decode graph
    cudaGraphExec_t gexec = nullptr, gexec2 = nullptr;   // gexec2: greedy step with the two-stage head
    bool h2 = false;                                    // V41_HEAD2: two-stage head on greedy steps and verifies
    int* h_fb = nullptr;
    // speculation
    std::unique_ptr<v41m::Drafter> D;
    std::map<int, cudaGraphExec_t> vexec, vexec2;  // verify graphs by row count; key -K = the K-slot draft
    int *h_draft = nullptr, *h_tgt = nullptr;
    int active_k = 0;                              // set_spec_k(); cfg.spec_k sizes the drafter
    v41m::LayerHook hook_for(int M) { return D ? v41m::tap_hook(*D, s, M) : v41m::LayerHook(); }

    void stage_rows(const int* ids, int n, int pos0) {
        CK(cudaStreamSynchronize(st));                 // the staging buffers and the Engram arena are reused
        for (int i = 0; i < n; ++i) { h_pos[i] = pos0 + i; h_ids[i] = ids[i]; h_img[i] = 0; }
        CK(cudaMemcpyAsync(s.d_pos, h_pos, n * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(s.d_ids, h_ids, n * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(s.d_img, h_img, n, cudaMemcpyHostToDevice, st));
        if (engram_on) {
            hasher->truncate(pos0);
            hasher->hash(ids, h_img, n, pos0, hid.data());
            store.submit(hid.data(), n);
        }
    }
    void reap() {
        if (!engram_on) return;
        std::string err;
        if (!store.complete(&err)) throw std::runtime_error("Engram row store: " + err);
        static const bool stats = [] { const char* e = getenv("V41_ENGRAM_STATS"); return e && *e == '1'; }();
        if (stats) {
            static int n = 0; static double a = 0, b = 0, c = 0;
            a += store.layer_us(0); b += store.layer_us(1); c += store.last_us();
            if (++n % 64 == 0) fprintf(stderr, "[engram] %d batches: submit->L1 %.0f us, ->L14 %.0f us, ->all %.0f us (mean)\n",
                                       n, a / n, b / n, c / n);
        }
    }

    // rows [k, n) of ids; logits of row n-1 land in s.logits
    void prefill(const std::vector<int>& ids, int k) {
        const int n = (int)ids.size();
        for (int c0 = k; c0 < n; c0 += chunk) {
            const int M = std::min(chunk, n - c0);
            stage_rows(ids.data() + c0, M, c0);
            v41m::embed_rows(m, s, M, st);
            reap();                                    // flags go to 1 as the reads land; stream waits on them
            v41m::forward_layers(m, s, 0, v41::N_LAYERS - 1, M, n, c0, eg, st, hook_for(M), dec_margin);
            if (c0 + M == n) {                        // greedy sample() reads d_tok: without the argmax here the
                v41m::head_row(m, s, M - 1, st);       // first token after a prefill was the previous step's
                v41m::argmax_logits(s, d_tok, st);
            }
        }
        // window of a row restarted at k reaches k-127; layer 39 kept rows >= n - 128 - dec_margin
        if (k < n - v41::WINDOW - dec_margin) reuse_floor = std::max(reuse_floor, n - 1 - dec_margin);
        kv_len = n;
        hi_written = std::max(hi_written, n - 1);
    }

    void step_body(bool fast) {                       // M = 1, inputs already in d_pos/d_ids/d_img
        v41m::embed_rows(m, s, 1, st);
        v41m::forward_layers(m, s, 0, v41::N_LAYERS - 1, 1, 0, 0, eg, st, hook_for(1));
        if (fast) { v41m::head_row_greedy(m, s, 0, d_tok, st); return; }
        v41m::head_row(m, s, 0, st);
        v41m::argmax_logits(s, d_tok, st);
    }
    // Run `body` as a CUDA graph cached in *ex (captured on first use). Capture includes the Engram
    // flag waits (cuStreamWaitValue32). If this driver refuses any of it, fall back to eager launches
    // for the rest of the process rather than die: same kernels, same order, so the ids are
    // unchanged, only launch overhead differs.
    template <class F> void launch(cudaGraphExec_t* ex, F body) {
        if (cfg.graph) {
            if (!*ex) {
                cudaGraph_t g = nullptr;
                cudaError_t e = cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal);
                if (e == cudaSuccess) { body(); e = cudaStreamEndCapture(st, &g); }
                if (e == cudaSuccess) e = cudaGraphInstantiate(ex, g, 0);
                if (g) cudaGraphDestroy(g);
                if (e != cudaSuccess) {
                    fprintf(stderr, "[v41] graph capture failed (%s): decoding without graphs\n", cudaGetErrorString(e));
                    (void)cudaGetLastError();
                    *ex = nullptr; cfg.graph = false;
                    body(); return;
                }
            }
            CK(cudaGraphLaunch(*ex, st));
        } else body();
    }
    // greedy: the caller reads only d_tok (sample() with fast), so the two-stage head may serve it
    void step(int tok, int pos, bool greedy = false) {
        const bool fast = greedy && h2;
        stage_rows(&tok, 1, pos);
        // Eager (no graph, or before the first capture): ~40 layers of single launches overrun the launch
        // queue while the stream sits on the layer-1 flag wait, so cudaLaunchKernel blocks before reap()
        // can set the flag -- a deadlock (--no-graph hung forever). Land the rows first; reap() is then a no-op.
        if (!cfg.graph || !(fast ? gexec2 : gexec)) reap();
        launch(fast ? &gexec2 : &gexec, [&] { step_body(fast); });
        reap();
        hi_written = std::max(hi_written, pos);
    }

    // One speculative round. `tok` sits at position pos (committed, KV not yet written); the anchor
    // is pos-1. Drafts K tokens, verifies [tok, d1..dK] at pos..pos+K, appends the accepted drafts
    // and the target's own token after them to `out`, and leaves kv_len = pos + 1 + accepted.
    // Greedy: accept d_i iff it equals the argmax at row i-1. Sampling: the row's sample instead,
    // drawn in order, so every emitted token is still a draw from the target's conditional.
    int verify(int tok, int pos, int K, const GenParams& p, std::mt19937_64& rng, std::deque<int>& out) {
        const int anchor = pos - 1;
        CK(cudaStreamSynchronize(st));
        h_ids[0] = anchor; h_ids[1] = tok;
        CK(cudaMemcpyAsync(D->d_anchor, &h_ids[0], 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(D->d_tok, &h_ids[1], 4, cudaMemcpyHostToDevice, st));
        launch(&vexec[-K], [&] { v41m::draft_body(m, *D, st, K); });
        CK(cudaMemcpyAsync(h_draft, D->d_out, (v41::DSPARK_BLOCK + 1) * 4, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
        const int M = K + 1;
        std::vector<int> rows(M);
        rows[0] = tok;
        for (int i = 1; i < M; ++i) rows[i] = h_draft[i];
        stage_rows(rows.data(), M, pos);
        const bool fast = p.temperature <= 0.f && h2;
        auto& vx = fast ? vexec2 : vexec;
        if (!cfg.graph || !vx[M]) reap();             // eager: see step()
        launch(&vx[M], [&] {
            v41m::embed_rows(m, s, M, st);
            v41m::forward_layers(m, s, 0, v41::N_LAYERS - 1, M, 0, 0, eg, st, hook_for(M));
            if (fast) v41m::head_rows_greedy(m, s, 0, M, st);
            else v41m::head_rows(m, s, 0, M, st);
        });
        reap();
        hi_written = std::max(hi_written, pos + K);
        std::vector<int> tgt(M);
        if (p.temperature <= 0.f) {
            CK(cudaMemcpyAsync(h_tgt, s.d_argmax, M * 4, cudaMemcpyDeviceToHost, st));
            if (fast) CK(cudaMemcpyAsync(h_fb, s.d_hfb, M * 4, cudaMemcpyDeviceToHost, st));
            CK(cudaStreamSynchronize(st));
            for (int i = 0; i < M; ++i) tgt[i] = h_tgt[i];
            if (fast) for (int i = 0; i < M; ++i) if (h_fb[i]) tgt[i] = head_fallback(i);
        }
        int a = 0;
        if (p.temperature <= 0.f) {
            a = v41spec::accept(rows.data(), tgt.data(), K);
            for (int i = 0; i <= a; ++i) out.push_back(tgt[i]);     // tgt[a]: the target's own token, no KV yet
        } else {
            for (;;) {                                 // sample row by row: stop at the first mismatch
                const int t = sample_row(p, rng, a);
                out.push_back(t);
                if (a < K && t == rows[a + 1]) { ++a; continue; }
                break;
            }
        }
        v41m::rollback_state(s, a, pos + a, st);
        kv_len = pos + 1 + a;
        return a;
    }

    // a two-stage-head row with too many candidates: the full head on the same normed x
    int head_fallback(int row) {
        ++h2_fallbacks;
        v41m::head_full_argmax(m, s, row, d_tok, st);
        CK(cudaMemcpyAsync(h_tok, d_tok, 4, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
        return *h_tok;
    }
    long h2_fallbacks = 0;
    int sample(const GenParams& p, std::mt19937_64& rng, bool fast = false) {
        if (p.temperature <= 0.f) {
            CK(cudaMemcpyAsync(h_tok, d_tok, 4, cudaMemcpyDeviceToHost, st));
            if (fast) CK(cudaMemcpyAsync(h_fb, s.d_hfb, 4, cudaMemcpyDeviceToHost, st));
            CK(cudaStreamSynchronize(st));
            if (fast && *h_fb) return head_fallback(0);
            return *h_tok;
        }
        return sample_row(p, rng, 0);
    }
    // temperature / top-p sample from s.logits row r
    int sample_row(const GenParams& p, std::mt19937_64& rng, int row) {
        CK(cudaMemcpyAsync(h_logits.data(), s.logits + (size_t)row * v41::VOCAB, v41::VOCAB * 4, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
        const int V = v41::VOCAB;
        float mx = -INFINITY;
        for (int i = 0; i < V; ++i) mx = std::max(mx, h_logits[i]);
        std::vector<std::pair<float, int>> pr(V);
        double z = 0;
        for (int i = 0; i < V; ++i) { const float e = std::exp((h_logits[i] - mx) / p.temperature); pr[i] = {e, i}; z += e; }
        std::sort(pr.begin(), pr.end(), [](const auto& a, const auto& b) { return a.first > b.first || (a.first == b.first && a.second < b.second); });
        double cum = 0; int cut = V;
        for (int i = 0; i < V; ++i) { cum += pr[i].first / z; if (cum >= p.top_p) { cut = i + 1; break; } }
        double zz = 0; for (int i = 0; i < cut; ++i) zz += pr[i].first;
        double r = std::uniform_real_distribution<double>(0.0, zz)(rng), acc = 0;
        for (int i = 0; i < cut; ++i) { acc += pr[i].first; if (r < acc) return pr[i].second; }
        return pr[cut - 1].second;
    }

    void reset_all() {
        v41m::reset_state(s, st);
        if (D) v41m::reset_drafter(*D, st);
        reuse_floor = 0; hi_written = -1;
    }

    // first position that must be recomputed for `ids` given the resident context
    int reuse_len(const std::vector<int>& ids) const {
        if (!prefix_cache) return 0;
        int k = 0;
        while (k < kv_len && k < (int)ctx.size() && k < (int)ids.size() && ctx[k] == ids[k]) ++k;
        if (k >= (int)ids.size()) k = (int)ids.size() - 1;   // the last prompt row must run for its logits
        k = std::max(0, k & ~1);                             // ratio-2 compressor state: pair boundary
        // window rings (main and drafter, RING = 256 slots): row k reads positions k-127..k-1, and
        // position q's slot is gone once q + RING has been written. Rewinding further than that
        // (a regenerate / branch far behind the old end) must recompute from 0.
        if (!v41spec::window_reuse_ok(k, hi_written, v41::WINDOW, v41m::State::RING)) return 0;
        return k < reuse_floor ? 0 : k;
    }
};

Engine::Engine(const EngineConfig& cfg) : p_(new Impl) { p_->cfg = cfg; }
Engine::~Engine() {
    if (p_->gexec) cudaGraphExecDestroy(p_->gexec);
    if (p_->gexec2) cudaGraphExecDestroy(p_->gexec2);
    for (auto& kv : p_->vexec) if (kv.second) cudaGraphExecDestroy(kv.second);
    for (auto& kv : p_->vexec2) if (kv.second) cudaGraphExecDestroy(kv.second);
    delete p_;
}

void Engine::load() {
    Impl& I = *p_;
    const auto t0 = clk::now();
    CK(cudaStreamCreateWithFlags(&I.st, cudaStreamNonBlocking));
    v41m::LoadOpts o; o.mxfp4_hw = I.cfg.mxfp4_hw;
    // NVFP4 dense overlay (ops/s07_quant.py overlay_nvfp4): served when set; the engine log line below
    // names it, so a run labelled "overlay" can be checked against what was actually loaded
    if (const char* e = getenv("DSV4_NVFP4_OVERLAY")) if (*e) o.overlay_dir = e;
    if (!I.cfg.keepset.empty()) {
        std::ifstream f(I.cfg.keepset);
        if (!f) throw std::runtime_error("cannot read keepset " + I.cfg.keepset);
        std::stringstream ss; ss << f.rdbuf();
        o.keepset_sha256 = nlohmann::json::parse(ss.str()).value("sha256", "");
    }
    v41m::load_model(I.m, I.cfg.ckpt_dir, o);
    if (!I.cfg.quiet)
        printf("[engine] weights %.2f GiB (%zu tensors), %d routed experts/layer, experts %s, %.0f s\n",
               I.m.store->gib(), I.m.store->count(), I.m.n_routed, I.m.quant_format.c_str(), ms_since(t0) / 1e3);
    I.chunk = std::max(1, std::min({I.cfg.ext_chunk, v41::WINDOW}));
    if (const char* e = getenv("V41_DEC_MARGIN")) I.dec_margin = std::max(0, atoi(e));
    v41m::alloc_state(I.s, I.cfg.seqmax, I.chunk);
    I.chunk = std::min(I.chunk, v41m::max_chunk(I.s));
    // two-stage greedy head (B2), default on (V41_HEAD2=0 off): V41_HEAD2_DELTA (logits), V41_HEAD2_CAP (candidates,
    // multiple of 8). Delta 4: V41_HEAD_PROBE measured NVFP4-vs-bf16 error rms 0.30 near the top (max 1.17 over
    // 442), so 4 is ~9 sigma of the two errors a miss needs; held-out 27/27 prompts identical, 0 fallbacks.
    if (const char* e = getenv("V41_HEAD2"); !(e && *e == '0')) {
        const char* d = getenv("V41_HEAD2_DELTA"); const char* c = getenv("V41_HEAD2_CAP");
        const float delta = d ? (float)atof(d) : 4.0f; const int cap = c ? std::max(8, atoi(c) / 8 * 8) : 64;
        v41m::enable_head2(I.m, I.s, cap, delta, I.st);
        I.h2 = true;
        if (!I.cfg.quiet) printf("[engine] two-stage greedy head: NVFP4 shortlist within %.3g logits (cap %d) + exact bf16 rescore\n", delta, cap);
    }
    if (I.cfg.spec_k > 0) {
        if (I.cfg.spec_k > v41::DSPARK_BLOCK) throw std::runtime_error("spec_k > DSpark block " + std::to_string(v41::DSPARK_BLOCK));
        if (I.chunk < I.cfg.spec_k + 1) throw std::runtime_error("chunk < spec_k + 1: a verify is one forward");
        const auto td = clk::now();
        I.D.reset(new v41m::Drafter);
        v41m::load_drafter(*I.D, I.cfg.ckpt_dir, I.cfg.draft_dir, I.cfg.mxfp4_hw);
        v41m::alloc_drafter(*I.D, I.s.max_rows);
        v41m::prepare_draft_head(I.m, *I.D, I.st);
        I.active_k = I.cfg.spec_k;
        if (!I.cfg.quiet)
            printf("[engine] DSpark drafter %s, K=%d, state %.2f GiB, %.0f s\n", I.D->desc.c_str(), I.cfg.spec_k,
                   I.D->bytes / 1073741824.0, ms_since(td) / 1e3);
    }
    if (!I.cfg.quiet) printf("[engine] state %.2f GiB (seqmax %d, chunk %d)\n", I.s.bytes / 1073741824.0, I.cfg.seqmax, I.chunk);
    // Engram row store. OFF only on explicit request: without it the model is a different function.
    const char* off = getenv("V41_ENGRAM_OFF");
    if (off && atoi(off)) {
        fprintf(stderr, "[engine] WARNING: V41_ENGRAM_OFF=1 -- Engram disabled, output is NOT the model's (bench only)\n");
    } else {
        EngramStoreConfig ec;
        ec.dir = I.cfg.engram_dir.empty() ? I.cfg.ckpt_dir + "/engram" : I.cfg.engram_dir;
        ec.max_tokens = I.chunk;
        ec.tokmap = ec.dir + "/engram_tokmap.bin";
        if (FILE* f = fopen(ec.tokmap.c_str(), "rb")) fclose(f);
        else {                                          // the repo's copy (tools/engram_meta.py): <repo>/build/<exe>
            char exe[4096]; const ssize_t n = readlink("/proc/self/exe", exe, sizeof exe - 1);
            std::string r = n > 0 ? std::string(exe, n) : std::string();
            for (int up = 0; up < 2 && !r.empty(); ++up) r = r.substr(0, r.rfind('/'));
            ec.tokmap = r + "/tests/data/engram_tokmap.bin";
        }
        std::string err;
        if (!I.store.open(ec, &err)) throw std::runtime_error("Engram row store at " + ec.dir + ": " + err);
        I.hasher.reset(new EngramHasher(I.store.tables()));
        I.engram_on = true;
        I.eg.store = &I.store; I.eg.graph_wait = true;
        if (!I.cfg.quiet) printf("[engine] Engram row store %s (format %d, %d B rows)\n", ec.dir.c_str(), I.store.format(), I.store.row_bytes());
    }
    CK(cudaHostAlloc(&I.h_pos, I.chunk * 4, 0)); CK(cudaHostAlloc(&I.h_ids, I.chunk * 4, 0));
    CK(cudaHostAlloc(&I.h_img, I.chunk, 0)); CK(cudaHostAlloc(&I.h_tok, 4, 0)); CK(cudaHostAlloc(&I.h_fb, 4 * std::max(I.chunk, 8), 0));
    CK(cudaMalloc(&I.d_tok, 4));
    CK(cudaHostAlloc(&I.h_draft, (v41::DSPARK_BLOCK + 1) * 4, 0)); CK(cudaHostAlloc(&I.h_tgt, I.chunk * 4, 0));
    I.hid.resize((size_t)I.chunk * 2 * v41::ENGRAM_COLS);
    I.h_logits.resize(v41::VOCAB);
    I.loaded = true;
}
bool Engine::loaded() const { return p_->loaded; }

GenStats Engine::generate(const std::vector<int>& ids, const GenParams& p, const std::function<bool(int)>& on_token) {
    Impl& I = *p_;
    GenStats gs;
    if (ids.empty()) return gs;
    if ((int)ids.size() + p.max_tokens > I.cfg.seqmax)
        throw std::runtime_error("prompt + max_tokens exceeds seqmax " + std::to_string(I.cfg.seqmax));
    std::mt19937_64 rng(p.has_seed ? p.seed : std::random_device{}());
    gs.prompt_tokens = (int)ids.size();
    const int k = I.reuse_len(ids);
    gs.cached_tokens = k;
    if (k == 0) I.reset_all();
    auto t0 = clk::now();
    I.prefill(ids, k);
    I.ctx.assign(ids.begin(), ids.end());
    std::deque<int> q;                              // committed, not yet emitted; only q.front() may lack KV
    q.push_back(I.sample(p, rng));
    gs.prefill_ms = ms_since(t0);
    t0 = clk::now();
    const bool spec = I.D != nullptr && I.active_k > 0;
    int emitted_by_spec = 0;
    const long fb0 = I.h2_fallbacks;
    for (int n = 0; n < p.max_tokens; ++n) {
        const int tok = q.front(); q.pop_front();
        ++gs.completion_tokens;
        const bool stop = tok == eos_id() || std::find(p.stop_ids.begin(), p.stop_ids.end(), tok) != p.stop_ids.end();
        if (!on_token(tok) || stop || n + 1 == p.max_tokens) {
            if (!stop) I.ctx.push_back(tok);        // in the context; its KV may not be written:
            break;                                  // kv_len (clamped below) says what is
        }
        const int pos = (int)I.ctx.size();
        I.ctx.push_back(tok);
        if (!q.empty()) continue;                   // an accepted draft: the verify wrote its KV
        const int K = std::min(I.active_k, p.max_tokens - (n + 1) - 1);
        if (spec && K > 0) {
            const int a = I.verify(tok, pos, K, p, rng, q);
            emitted_by_spec += a + 1;
            if (K < SpecProfile::MAXW) ++gs.spec.accept_joint[K][a];
        } else {
            const bool greedy = p.temperature <= 0.f;
            I.step(tok, pos, greedy);
            I.kv_len = pos + 1;
            q.push_back(I.sample(p, rng, greedy && I.h2));
            if (spec) ++gs.spec.accept_joint[0][0];
        }
        ++gs.verifies;
    }
    I.kv_len = std::min(I.kv_len, (int)I.ctx.size());
    gs.decode_ms = ms_since(t0);
    gs.head2_fallbacks = (int)(I.h2_fallbacks - fb0);
    gs.tok_per_s = gs.decode_ms > 0 ? (gs.completion_tokens - 1) / (gs.decode_ms / 1e3) : 0;
    gs.tok_per_verify = gs.verifies ? (double)(gs.completion_tokens - 1) / gs.verifies : 0;
    if (!spec && gs.verifies < SpecProfile::MAXW) gs.spec.accept_joint[1][0] = gs.verifies;
    (void)emitted_by_spec;
    return gs;
}

void Engine::reset() { p_->ctx.clear(); p_->kv_len = 0; if (p_->loaded) p_->reset_all(); else p_->reuse_floor = 0; }
int Engine::context_len() const { return (int)p_->ctx.size(); }
const std::vector<int>& Engine::context_ids() const { return p_->ctx; }
int Engine::eos_id() const { return 1; }
int Engine::seqmax() const { return p_->cfg.seqmax; }
void Engine::set_prefix_cache(bool on) { p_->prefix_cache = on; }
void Engine::set_ext_chunk(int n) { p_->chunk = std::max(1, std::min(n, p_->s.max_rows)); }
int Engine::set_spec_k(int k) {
    Impl& I = *p_;
    I.active_k = I.D ? std::max(0, std::min(k, I.cfg.spec_k)) : 0;
    return I.active_k;
}
std::string Engine::drafter_desc() const { return p_->D ? p_->D->desc : std::string(); }

std::vector<float> Engine::debug_prefix_logits(const std::vector<int>& ids) {
    Impl& I = *p_;
    const int k = I.reuse_len(ids);
    if (k == 0) I.reset_all();
    I.prefill(ids, k);
    I.ctx.assign(ids.begin(), ids.end());
    std::vector<float> out(v41::VOCAB);
    CK(cudaMemcpyAsync(out.data(), I.s.logits, v41::VOCAB * 4, cudaMemcpyDeviceToHost, I.st));
    CK(cudaStreamSynchronize(I.st));
    return out;
}

std::vector<float> Engine::debug_forced_logits(const std::vector<int>& ids, const std::vector<int>& forced) {
    Impl& I = *p_;
    I.reset_all();
    I.prefill(ids, 0);
    I.ctx.assign(ids.begin(), ids.end());
    std::vector<float> out((size_t)forced.size() * v41::VOCAB);
    // V41_HEAD_PROBE=1: also an NVFP4 copy of the head on the same normed x (s.x row 0) -- where does the
    // exact argmax rank in the approximate order, and how large is the approximation error near the top?
    // (the B2 two-stage head: NVFP4 shortlist + exact bf16 rescore)
    static const bool probe = [] { const char* e = getenv("V41_HEAD_PROBE"); return e && *e == '1'; }();
    uint8_t *h4 = nullptr, *h4s = nullptr; float h4g = 0.f, *alog = nullptr;
    std::vector<float> ap(v41::VOCAB);
    std::vector<int> ranks; double se_all = 0, se_top = 0; size_t n_all = 0, n_top = 0; float maxe_top = 0;
    if (probe) {
        CK(cudaMalloc(&h4, (size_t)v41::VOCAB * v41::DIM / 2)); CK(cudaMalloc(&h4s, (size_t)v41::VOCAB * v41::DIM / 16));
        CK(cudaMalloc(&alog, (size_t)v41::VOCAB * 4));
        h4g = v41k::nvfp4_from_bf16(h4, h4s, I.m.head, v41::VOCAB, v41::DIM, v41k::bf16_permuted(I.m.head), I.st);
    }
    for (size_t i = 0; i < forced.size(); ++i) {
        CK(cudaMemcpyAsync(out.data() + i * v41::VOCAB, I.s.logits, v41::VOCAB * 4, cudaMemcpyDeviceToHost, I.st));
        CK(cudaStreamSynchronize(I.st));
        if (probe) {
            v41k::nvfp4_linear_f32(alog, I.s.x, h4, h4s, h4g, 1, v41::VOCAB, v41::DIM, I.st);
            CK(cudaMemcpyAsync(ap.data(), alog, v41::VOCAB * 4, cudaMemcpyDeviceToHost, I.st));
            CK(cudaStreamSynchronize(I.st));
            const float* ex = out.data() + i * v41::VOCAB;
            int am = 0; for (int r = 1; r < v41::VOCAB; ++r) if (ex[r] > ex[am]) am = r;
            int rank = 0; for (int r = 0; r < v41::VOCAB; ++r) if (ap[r] > ap[am] || (ap[r] == ap[am] && r < am)) ++rank;
            ranks.push_back(rank);
            for (int r = 0; r < v41::VOCAB; ++r) {
                const double e = ap[r] - ex[r]; se_all += e * e; ++n_all;
                if (ex[r] > ex[am] - 10.f) { se_top += e * e; ++n_top; maxe_top = std::max(maxe_top, (float)std::fabs(e)); }
            }
        }
        if (i + 1 == forced.size()) break;
        if (i + 1 == forced.size()) break;
        const int pos = (int)I.ctx.size();
        I.ctx.push_back(forced[i]);
        I.step(forced[i], pos);
        I.kv_len = pos + 1;
    }
    if (probe) {
        std::vector<int> r(ranks); std::sort(r.begin(), r.end());
        int c[6] = {0, 0, 0, 0, 0, 0}; const int lim[6] = {0, 3, 15, 63, 255, 1 << 30};
        for (int v : r) for (int j = 0; j < 6; ++j) if (v <= lim[j]) { ++c[j]; break; }
        printf("[probe] head NVFP4 vs bf16 over %zu positions: exact argmax rank in approx order max %d, median %d;"
               " rank 0: %d, <=3: %d, <=15: %d, <=63: %d, <=255: %d, more: %d\n", r.size(), r.back(), r[r.size() / 2],
               c[0], c[1], c[2], c[3], c[4], c[5]);
        printf("[probe] approx-exact logit error: rms all %.4f, rms within 10 of top %.4f (n %zu), max within 10 of top %.4f\n",
               std::sqrt(se_all / n_all), std::sqrt(se_top / std::max<size_t>(n_top, 1)), n_top, maxe_top);
        std::string rs; for (int v : ranks) rs += std::to_string(v) + " ";
        printf("[probe] ranks: %s\n", rs.c_str());
        cudaFree(h4); cudaFree(h4s); cudaFree(alog);
    }
    I.reset_all(); I.ctx.clear(); I.kv_len = 0;
    return out;
}

} // namespace dsv4srv
