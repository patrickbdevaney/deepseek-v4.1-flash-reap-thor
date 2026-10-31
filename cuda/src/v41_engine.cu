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
#include "v41_vision.h"
#include <cuda.h>
#include <cuda_bf16.h>
#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <future>
#include <condition_variable>
#include <dirent.h>
#include <fcntl.h>
#include <mutex>
#include <thread>
#include <fstream>
#include <map>
#include <unordered_map>
#include <random>
#include <sstream>
#include "third_party/json.hpp"
#include <stdexcept>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) throw std::runtime_error(std::string(#x) + ": " + cudaGetErrorString(e_)); } while (0)

namespace dsv4srv {

using clk = std::chrono::steady_clock;
static double ms_since(clk::time_point t) { return std::chrono::duration<double, std::milli>(clk::now() - t).count(); }

// V2: overwrite the hc copies of image-span rows with their embeddings (rounded to bf16 like the model's h)
__global__ static void k_img_rows(float* h, const float* emb, const int* rowmap, int M) {
    const int m = blockIdx.x, r = rowmap[m];
    if (r < 0) return;
    for (int d = threadIdx.x; d < v41::DIM; d += blockDim.x) {
        const float v = __bfloat162float(__float2bfloat16_rn(emb[(size_t)r * v41::DIM + d]));
        for (int j = 0; j < 4; ++j) h[(size_t)m * v41::HC_DIM + (size_t)j * v41::DIM + d] = v;
    }
}
// V2: span rows from the aligner rows and the learned delimiters; src[r] = aligner row, or -1 start / -2 newline / -3 end
__global__ static void k_img_span(float* dst, const float* al, const int* src, const uint16_t* st_, const uint16_t* nl,
                                  const uint16_t* en) {
    const int r = blockIdx.x, s = src[r];
    for (int d = threadIdx.x; d < v41::DIM; d += blockDim.x) {
        float v;
        if (s >= 0) v = al[(size_t)s * v41::DIM + d];
        else { const uint16_t b = (s == -1 ? st_ : s == -2 ? nl : en)[d]; v = __uint_as_float((uint32_t)b << 16); }
        dst[(size_t)r * v41::DIM + d] = v;
    }
}

__global__ static void k_logit_mask(float* L, const uint32_t* bits, int V) {   // C3: disallowed -> -inf
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < V && !((bits[i >> 5] >> (i & 31)) & 1u)) L[i] = -INFINITY;
}

struct Engine::Impl {
    EngineConfig cfg;
    v41m::Model m;
    v41m::State s;
    EngramStore store;
    EngramStore store2;                                // prefill prefetch: the next chunk's rows (V41_ENGRAM_PREFETCH=0 off)
    bool engram_prefetch = false;
    std::vector<uint8_t> img0;                         // is_image per hashed prompt row (V2: 1 inside image spans)
    // V2 vision: the tower, the request's span embeddings [rows, DIM], position -> span row (-1 text), and the
    // image identity per position (0 text) for prefix reuse: two images share token ids, never pixels
    std::unique_ptr<v41v::Vision> V;
    float* d_imgemb = nullptr; size_t imgemb_rows = 0;
    float* d_al = nullptr; uint16_t* d_patches = nullptr; int* d_spansrc = nullptr;
    int* d_imgrow = nullptr; int* h_imgrow = nullptr;
    std::vector<int> pos_img;
    std::vector<uint64_t> ctx_img, req_img;
    bool stage_has_img = false;
    static uint64_t img_at(const std::vector<uint64_t>& v, int i) { return i < (int)v.size() ? v[i] : 0; }
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
    uint32_t *h_mask = nullptr, *d_mask = nullptr;     // C3: VOCAB bits
    static constexpr int MASK_WORDS = (v41::VOCAB + 31) / 32;
    static_assert(MASK_WORDS == kMaskWords, "GenParams::mask buffer");
    uint8_t* h_img = nullptr;
    std::vector<int64_t> hid;
    std::vector<float> h_logits;
    // decode graph
    cudaGraphExec_t gexec = nullptr, gexec2 = nullptr;   // gexec2: greedy step with the two-stage head
    cudaGraphExec_t gexec3 = nullptr;                   // S11: sampled step with the sampled two-stage head
    bool h2s = false;                                   // V41_HEAD2_SAMPLE: sampled rows on head2_logits
    float h_delta = 0.f;                                // its window, host copy (set per call from T)
    int h2s_check = 0; double h2s_tv_sum = 0, h2s_tv_max = 0; long h2s_n = 0;   // V41_H2S_CHECK=1: TV vs the full head
    bool h2 = false;                                    // V41_HEAD2: two-stage head on greedy steps and verifies
    int* h_fb = nullptr;
    // speculation
    std::unique_ptr<v41m::Drafter> D;
    std::map<int, cudaGraphExec_t> vexec, vexec2, vexec3;  // verify graphs by row count; key -K = the K-slot draft
    int *h_draft = nullptr, *h_tgt = nullptr;
    int active_k = 0;                              // set_spec_k(); cfg.spec_k sizes the drafter
    // S5 adaptive K (V41_SPEC_ADAPT=1): per round pick K in [0, active_k] maximising expected tokens per
    // ms, E(K) = 1 + sum_{i<=K} prod_{j<=i} pa[j] over T(K). T = EMA of the measured round wall time at
    // that K (persists across requests); pa[i] = EMA (0.2) of P(draft i accepted | drafts < i accepted),
    // reset to 0.65 per request. Every 8th round below kmax runs kmax, so every position keeps learning
    // (probing K+1, with pa kept across requests, left the deep positions stale after a chat prompt and
    // under-drafted code by 0.1x). Greedy ids are the same at every K (spec identity), so this only
    // changes speed.
    struct AdaptK {
        static constexpr int MK = 8;
        double T[MK + 1] = {}, pa[MK + 1] = {};
        int cnt[MK + 1] = {};                       // rounds timed per K; the first is dropped (graph capture)
        int round = 0;
        void reset_accept() { for (int i = 0; i <= MK; ++i) pa[i] = 0.65; round = 0; }
        int pick(int kmax) {
            kmax = std::min(kmax, MK);
            for (int k = 0; k <= kmax; ++k) if (cnt[k] < 2) return k;           // time every K (after its capture)
            int best = 0; double bv = -1, e = 1, prod = 1;
            for (int k = 0; k <= kmax; ++k) {
                if (k > 0) { prod *= pa[k]; e += prod; }
                const double v = e / T[k];
                if (v > bv) { bv = v; best = k; }
            }
            if (++round % 8 == 0) best = kmax;
            return best;
        }
        void update(int k, int a, double ms) {
            if (cnt[k]++ > 0) T[k] = T[k] > 0 ? 0.9 * T[k] + 0.1 * ms : ms;
            for (int i = 1; i <= std::min(a + 1, k); ++i) pa[i] = 0.8 * pa[i] + 0.2 * (i <= a ? 1.0 : 0.0);
        }
    } adapt;
    bool early_launch = [] { const char* e = getenv("V41_EARLY_LAUNCH"); return !(e && *e == '0'); }();
    bool adapt_on = [] { const char* e = getenv("V41_SPEC_ADAPT"); return e && *e == '1'; }();
    // S8 suffix drafts (SuffixDecoding / prompt lookup, include/suffix_draft.h's rule): the longest suffix
    // of the context (>= 3 tokens) that occurred earlier, most recent occurrence first, proposes that
    // occurrence's continuation for free -- no drafter run. A round takes the suffix when its expected
    // tokens per ms beat the alternative (a DSpark round, or a plain step without a drafter):
    // E(K) = sum_{i<=K} q^i with q = EMA acceptance per match-length bucket, T(M) = EMA verify wall by
    // rows. Exact like every verify (the target's own tokens are emitted). V41_SFX=0 off, V41_SFX_K max drafts.
    struct Sfx {
        static constexpr int G = 3;
        std::unordered_map<uint64_t, std::vector<int>> m;     // 3-gram (21 bits a token) -> its end positions
        int n = 0;
        static uint64_t key(const int* t) { return (uint64_t)t[0] << 42 | (uint64_t)t[1] << 21 | (uint64_t)t[2]; }
        void clear() { m.clear(); n = 0; }
        void sync(const std::vector<int>& c) {
            for (; n < (int)c.size(); ++n) if (n >= G - 1) m[key(&c[n - G + 1])].push_back(n);
        }
        // longest earlier match of c's suffix; *from = the continuation's first index (< c.size())
        int find(const std::vector<int>& c, int* from) const {
            const int N = (int)c.size();
            if (N < G + 1) return 0;
            const auto it = m.find(key(&c[N - G]));
            if (it == m.end()) return 0;
            int best = 0, seen = 0;
            for (auto e = it->second.rbegin(); e != it->second.rend() && seen < 64; ++e) {
                if (*e >= N - 1) continue;
                ++seen;
                int L = G;
                while (L < 256 && *e - L >= 0 && c[*e - L] == c[N - 1 - L]) ++L;
                if (L > best) { best = L; *from = *e + 1; }
            }
            return best;
        }
        // arbitration state
        double q[4] = {0.5, 0.65, 0.8, 0.9};                  // P(next draft accepted | previous were), by match length 3 / 4-7 / 8-15 / 16+
        double Tv[33] = {};                                    // verify-only round wall by rows (ms), EMA
        double Tb = 0, Eb = 0;                                 // the alternative round: wall (ms), tokens
        static int bucket(int L) { return L >= 16 ? 3 : L >= 8 ? 2 : L >= 4 ? 1 : 0; }
        double tv(int M) const { return Tv[M] > 0 ? Tv[M] : (Tv[1] > 0 ? Tv[1] : 40.0) * (1.0 + 0.3 * (M - 1)); }
        // drafts to take from a match of length L with `avail` continuation tokens; 0 = do the alternative
        int pick(int L, int avail, int kmax) const {
            // V41_SFX_FORCE=1 (debug): every match proposes all it can, no timing -- reproducible verify widths
            static const bool force = [] { const char* e = getenv("V41_SFX_FORCE"); return e && *e == '1'; }();
            if (force) return std::min(avail, kmax);
            const double qq = q[bucket(L)];
            const double base = Tb > 0 ? Eb / Tb : 0.0;
            int best = 0; double bv = base, e = 1, pw = 1;
            for (int k = 1; k <= std::min(avail, kmax); ++k) {
                pw *= qq; e += pw;
                const double v = e / tv(k + 1);
                if (v > bv * 1.02) { bv = v; best = k; }
            }
            return best;
        }
        void upd_sfx(int L, int K, int a, double ms) {
            double& t = Tv[K + 1]; t = t > 0 ? 0.8 * t + 0.2 * ms : ms;
            double& qq = q[bucket(L)];
            for (int i = 1; i <= std::min(a + 1, K); ++i) qq = 0.85 * qq + 0.15 * (i <= a ? 1.0 : 0.0);
        }
        void upd_base(int M, int toks, double ms) {
            Tb = Tb > 0 ? 0.9 * Tb + 0.1 * ms : ms; Eb = Eb > 0 ? 0.9 * Eb + 0.1 * toks : toks;
            if (M == 1) { double& t = Tv[1]; t = t > 0 ? 0.9 * t + 0.1 * ms : ms; }
        }
    } sfx;
    bool sfx_on = [] { const char* e = getenv("V41_SFX"); return !(e && *e == '0'); }();
    // At most 7 drafts = 8 verify rows: every bit-identical-to-M=1 path (k_nvt, cluster CNQ, A9, split sparse_attn) is
    // M <= 8 only; a 9-row verify drops onto the prefill kernels and row 0 drifts from AR (ops/logs/q6d, q6e: forced
    // widths 1..7 identical to AR, 8 diverges on 8/10 prompts). Clamped, not just defaulted, so V41_SFX_K can't break it.
    int sfx_kmax = [] { const char* e = getenv("V41_SFX_K"); return e ? std::max(1, std::min(7, atoi(e))) : 7; }();
    v41m::LayerHook hook_for(int M) { return D ? v41m::tap_hook(*D, s, M) : v41m::LayerHook(); }

    void stage_dev(const int* ids, int n, int pos0) {
        CK(cudaStreamSynchronize(st));                 // the staging buffers and the Engram arena are reused
        stage_has_img = false;
        for (int i = 0; i < n; ++i) {
            const int pi = pos0 + i < (int)pos_img.size() ? pos_img[pos0 + i] : -1;
            h_pos[i] = pos0 + i; h_ids[i] = ids[i]; h_img[i] = pi >= 0; h_imgrow[i] = pi;
            stage_has_img = stage_has_img || pi >= 0;
        }
        CK(cudaMemcpyAsync(s.d_pos, h_pos, n * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(s.d_ids, h_ids, n * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(s.d_img, h_img, n, cudaMemcpyHostToDevice, st));
        if (stage_has_img) CK(cudaMemcpyAsync(d_imgrow, h_imgrow, n * 4, cudaMemcpyHostToDevice, st));
    }
    void submit_engram(EngramStore& t, const int* ids, int n, int pos0) {
        if (img0.size() < (size_t)n) img0.resize(n);
        for (int i = 0; i < n; ++i) img0[i] = pos0 + i < (int)pos_img.size() && pos_img[pos0 + i] >= 0;
        hasher->truncate(pos0);
        hasher->hash(ids, img0.data(), n, pos0, hid.data());
        t.submit(hid.data(), n);
    }
    void stage_rows(const int* ids, int n, int pos0) {
        stage_dev(ids, n, pos0);
        if (engram_on) submit_engram(store, ids, n, pos0);
    }
    static void reap_store(EngramStore& t) {
        std::string err;
        if (!t.complete(&err)) throw std::runtime_error("Engram row store: " + err);
    }
    void reap() {
        if (!engram_on) return;
        reap_store(store);
        static const bool stats = [] { const char* e = getenv("V41_ENGRAM_STATS"); return e && *e == '1'; }();
        if (stats) {
            static int n = 0; static double a = 0, b = 0, c = 0;
            a += store.layer_us(0); b += store.layer_us(1); c += store.last_us();
            if (++n % 64 == 0) fprintf(stderr, "[engram] %d batches: submit->L1 %.0f us, ->L14 %.0f us, ->all %.0f us (mean)\n",
                                       n, a / n, b / n, c / n);
        }
    }

    // Prefill chunk: launch the forward while a host thread reaps the Engram reads. The forward waits
    // on-stream for each Engram layer's flag (cuStreamWaitValue32), so layer 0 runs while layer 1's rows
    // land and layers 1..13 while layer 14's land; the reaper being a separate thread keeps a forward that
    // blocks the host (a grow-only scratch's cudaFree) from deadlocking on a flag only the host can set.
    // V41_ENGRAM_OVERLAP=0: reap first (the old order).
    // S7 (lossy, opt-in): V41_SPEC_BUDGET=B caps the routed-expert union of the verify rows at B (row 0 exact).
    // V41_S7_PREFILL=1 applies it to every prefill chunk too: with --chunk K+1, --score then measures the
    // verify rows' distribution on forced text (the T2 gate's KL), which a verify never exposes.
    int s7_budget = [] { const char* e = getenv("V41_SPEC_BUDGET"); return e ? std::max(0, atoi(e)) : 0; }();
    bool s7_prefill = [] { const char* e = getenv("V41_S7_PREFILL"); return e && *e == '1'; }();
    void forward_chunk(int M, int n, int c0) {
        struct S7 { v41m::State& s; S7(v41m::State& s_, int b) : s(s_) { s.route_budget = b; } ~S7() { s.route_budget = 0; } } s7g(s, s7_prefill ? s7_budget : 0);
        static const bool overlap = [] { const char* e = getenv("V41_ENGRAM_OVERLAP"); return !(e && *e == '0'); }();
        if (!engram_on || !overlap) {
            reap();
            v41m::forward_layers(m, s, 0, v41::N_LAYERS - 1, M, n, c0, eg, st, hook_for(M), dec_margin);
            return;
        }
        auto reaper = std::async(std::launch::async, [this] { reap(); });
        try {
            v41m::forward_layers(m, s, 0, v41::N_LAYERS - 1, M, n, c0, eg, st, hook_for(M), dec_margin);
        } catch (...) { reaper.wait(); throw; }
        reaper.get();                                  // rethrows an Engram I/O error
    }

    // Multi-chunk prefill with Engram prefetch: chunk i+1's rows are read into the other store while
    // chunk i runs. Overlapping a chunk's own reads hid only ~0.6 s of its ~1.2 s (2048 rows x 48 reads):
    // the forward reaches layer 1 first (nsys pf7: 4 x 0.63 s stalled before k_engram_dequant).
    // stage_dev syncs the stream, so chunk i-1 -- the other store's last user -- is done before its
    // arena is refilled; each forward re-arms the flags of the store it waited on.
    void prefill_prefetch(const int* ids, int k, int n) {
        EngramStore* S[2] = {&store, &store2};
        auto start = [&](EngramStore& t, int c0) {
            submit_engram(t, ids + c0, std::min(chunk, n - c0), c0);
            return std::async(std::launch::async, [&t] { reap_store(t); });
        };
        std::future<void> cur = start(*S[0], k), nxt;
        for (int c0 = k, i = 0; c0 < n; c0 += chunk, ++i) {
            const int M = std::min(chunk, n - c0);
            stage_dev(ids + c0, M, c0);
            if (c0 + M < n) nxt = start(*S[(i + 1) & 1], c0 + M);
            v41m::embed_rows(m, s, M, st);
            embed_images(M);
            eg.store = S[i & 1];
            s.route_budget = s7_prefill ? s7_budget : 0;   // S7 gate only (see forward_chunk)
            try {
                v41m::forward_layers(m, s, 0, v41::N_LAYERS - 1, M, n, c0, eg, st, hook_for(M), dec_margin);
            } catch (...) { eg.store = &store; s.route_budget = 0; throw; }   // the futures' destructors wait for the reapers
            s.route_budget = 0;
            eg.store = &store;
            cur.get();                                     // rethrows an Engram I/O error
            cur = std::move(nxt);
            if (c0 + M == n) { v41m::head_row(m, s, M - 1, st); v41m::argmax_logits(s, d_tok, st); }
        }
    }

    void embed_images(int M) {
        if (stage_has_img) k_img_rows<<<M, 256, 0, st>>>(s.h, d_imgemb, d_imgrow, M);
    }
    // V2: lay out the request's image spans: pos_img / req_img per position, span-row storage. Cheap (no tower):
    // the reuse decision needs req_img first, and only the images the reuse does not cover are encoded (V3).
    void plan_images(const std::vector<int>& ids, const std::vector<ImageSpan>& imgs) {
        pos_img.assign(ids.size(), -1);
        req_img.assign(ids.size(), 0);
        if (imgs.empty()) return;
        if (!V) throw std::runtime_error("request has images but the vision tower is not loaded (no vision.safetensors, or V41_VISION=0)");
        size_t rows = 0;
        for (const auto& im : imgs) rows += im.types.size();
        if (rows > imgemb_rows) {
            if (d_imgemb) cudaFree(d_imgemb);
            CK(cudaMalloc(&d_imgemb, rows * v41::DIM * 4)); imgemb_rows = rows;
        }
        size_t base = 0;
        for (const auto& im : imgs) {
            const int n = (int)im.types.size();
            if (im.start < 0 || im.start + n > (int)ids.size()) throw std::runtime_error("image span outside the prompt");
            for (int r = 0; r < n; ++r) {
                if (ids[im.start + r] != v41::IMAGE_TOKEN_ID) throw std::runtime_error("image span ids are not the image token");
                pos_img[im.start + r] = (int)(base + r);
                req_img[im.start + r] = im.hash * 0x9E3779B97F4A7C15ull + (uint64_t)r + 1;
            }
            base += n;
        }
    }
    // V3 encoder-output cache: aligner rows by image hash, LRU, V41_VCACHE entries (default 8, ~20 MB each at 1024 tokens)
    struct VEnt { uint64_t hash; float* rows; int n; uint64_t used; };
    std::vector<VEnt> vcache;
    uint64_t vclock = 0;
    int vcache_max = [] { const char* e = getenv("V41_VCACHE"); return e ? std::max(0, atoi(e)) : 8; }();
    long vit_runs = 0, vit_cached = 0, vit_skipped = 0;
    // write the span rows of every image whose span reaches past k (rows below k are reused KV, never embedded)
    void encode_images(const std::vector<ImageSpan>& imgs, int k) {
        size_t base = 0;
        for (const auto& im : imgs) {
            const int n = (int)im.types.size();
            if (im.start + n <= k) { base += n; ++vit_skipped; continue; }
            const int P = im.n_vit_h * im.n_vit_w, Lr = v41v::llm_rows(im.n_vit_h, im.n_vit_w);
            if ((int)im.patches.size() != P * v41v::PATCH_IN) throw std::runtime_error("image patches size mismatch");
            if (P > V->max_patches) throw std::runtime_error("image has too many patches");
            const float* al = nullptr;
            for (auto& e : vcache) if (e.hash == im.hash && e.n == Lr) { e.used = ++vclock; al = e.rows; ++vit_cached; break; }
            if (!al) {
                CK(cudaMemcpyAsync(d_patches, im.patches.data(), im.patches.size() * 2, cudaMemcpyHostToDevice, st));
                v41v::encode_image(*V, d_patches, im.n_vit_h, im.n_vit_w, d_al, st);
                ++vit_runs; al = d_al;
                if (vcache_max > 0) {
                    if ((int)vcache.size() >= vcache_max) {
                        size_t lru = 0; for (size_t i = 1; i < vcache.size(); ++i) if (vcache[i].used < vcache[lru].used) lru = i;
                        cudaFree(vcache[lru].rows); vcache.erase(vcache.begin() + lru);
                    }
                    VEnt e{im.hash, nullptr, Lr, ++vclock};
                    if (cudaMalloc(&e.rows, (size_t)Lr * v41::DIM * 4) == cudaSuccess) {
                        CK(cudaMemcpyAsync(e.rows, d_al, (size_t)Lr * v41::DIM * 4, cudaMemcpyDeviceToDevice, st));
                        vcache.push_back(e);
                    } else (void)cudaGetLastError();
                }
            }
            std::vector<int> src(n);
            int j = 0;
            for (int r = 0; r < n; ++r) { const int t = im.types[r]; src[r] = t == 1 ? j++ : t == 0 ? -1 : t == 2 ? -2 : -3; }
            if (j != Lr) throw std::runtime_error("image span has " + std::to_string(j) + " image rows, the aligner made " + std::to_string(Lr));
            CK(cudaMemcpyAsync(d_spansrc, src.data(), n * 4, cudaMemcpyHostToDevice, st));
            k_img_span<<<n, 256, 0, st>>>(d_imgemb + base * v41::DIM, al, d_spansrc, V->img_start, V->img_newline, V->img_end);
            CK(cudaStreamSynchronize(st));            // src / patches staging is reused by the next image
            base += n;
        }
    }

    // rows [k, n) of ids; logits of row n-1 land in s.logits
    void prefill(const std::vector<int>& ids, int k) {
        const int n = (int)ids.size();
        if (engram_prefetch && n - k > chunk) prefill_prefetch(ids.data(), k, n);
        else for (int c0 = k; c0 < n; c0 += chunk) {
            const int M = std::min(chunk, n - c0);
            stage_rows(ids.data() + c0, M, c0);
            v41m::embed_rows(m, s, M, st);
            embed_images(M);
            forward_chunk(M, n, c0);                   // flags go to 1 as the reads land; stream waits on them
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

    void step_body(int mode) {                        // M = 1, inputs already in d_pos/d_ids/d_img; mode 1 greedy two-stage, 2 sampled two-stage
        v41m::embed_rows(m, s, 1, st);
        v41m::forward_layers(m, s, 0, v41::N_LAYERS - 1, 1, 0, 0, eg, st, hook_for(1));
        if (mode == 1) { v41m::head_row_greedy(m, s, 0, d_tok, st); return; }
        if (mode == 2) { v41m::head_rows_sample(m, s, 0, 1, st); return; }
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
    // S11: the window for sampled rows at temperature T: a token left approximate has p <= 1e-9 x the top's
    bool sample_fast(const GenParams& p) {
        if (!h2s || p.temperature <= 0.f) return false;
        h_delta = p.temperature * 20.7233f + s.hd2.delta;           // ln(1e9) = 20.72
        CK(cudaMemcpyAsync(s.d_h2delta, &h_delta, sizeof(float), cudaMemcpyHostToDevice, st));
        return true;
    }
    void step(int tok, int pos, bool greedy = false, bool sfast = false) {
        const bool fast = greedy && h2;
        const int mode = fast ? 1 : sfast ? 2 : 0;
        cudaGraphExec_t* ex = mode == 1 ? &gexec2 : mode == 2 ? &gexec3 : &gexec;
        // Eager (no graph, or before the first capture): ~40 layers of single launches overrun the launch
        // queue while the stream sits on the layer-1 flag wait, so cudaLaunchKernel blocks before reap()
        // can set the flag -- a deadlock (--no-graph hung forever). Land the rows first; reap() is then a no-op.
        // A6: with a captured graph, launch first and hash + submit the Engram reads while layer 0 runs (the
        // graph waits on the layer-1 flag; the previous step's reads were consumed before its sync).
        if (cfg.graph && *ex && early_launch) {
            stage_dev(&tok, 1, pos);
            CK(cudaGraphLaunch(*ex, st));
            if (engram_on) submit_engram(store, &tok, 1, pos);
        } else {
            stage_rows(&tok, 1, pos);
            if (!cfg.graph || !*ex) reap();
            launch(ex, [&] { step_body(mode); });
        }
        reap();
        hi_written = std::max(hi_written, pos);
    }

    // One speculative round. `tok` sits at position pos (committed, KV not yet written); the anchor
    // is pos-1. Drafts K tokens, verifies [tok, d1..dK] at pos..pos+K, appends the accepted drafts
    // and the target's own token after them to `out`, and leaves kv_len = pos + 1 + accepted.
    // Greedy: accept d_i iff it equals the argmax at row i-1. Sampling: the row's sample instead,
    // drawn in order, so every emitted token is still a draw from the target's conditional.
    // sfx: K drafts from the suffix index (S8) instead of the drafter
    // S9 (default on, V41_SPEC_CONF=0 off; held-out K=3 geomean 1.299x -> 1.325x, worst prompt 0.96x -> 1.055x): after drafting K, verify only the first K' drafts, K' maximising expected tokens per ms:
    // E(K') = 1 + sum_{i<=K'} prod_{j<=i} q(c_j), q = acceptance EMA of the drafter-confidence bucket (prior: c),
    // T(K') = draft + verify wall at K' + 1 rows (EMAs). Exact like every verify; only the widths change.
    struct ConfStop {
        double q[10], Tv[9] = {}, Td = 0;
        ConfStop() { for (int b = 0; b < 10; ++b) q[b] = 0.05 + 0.1 * b; }
        static int bucket(float c) { return std::min(9, std::max(0, (int)(c * 10.f))); }
        double tv(int M) const { return Tv[M] > 0 ? Tv[M] : (Tv[1] > 0 ? Tv[1] : 40.0) * (1.0 + 0.3 * (M - 1)); }
        int pick(const float* c, int K) const {
            int best = K; double bv = -1, e = 1, pr = 1;
            for (int k = 0; k <= K; ++k) {
                if (k > 0) { pr *= q[bucket(c[k - 1])]; e += pr; }
                const double v = e / ((Td > 0 ? Td : 12.0) + tv(k + 1));
                if (v > bv) { bv = v; best = k; }
            }
            return best;
        }
        void update(const float* c, int K, int a, double td, double tvm) {
            Td = Td > 0 ? 0.9 * Td + 0.1 * td : td;
            double& t = Tv[K + 1]; t = t > 0 ? 0.8 * t + 0.2 * tvm : tvm;
            for (int i = 1; i <= std::min(a + 1, K); ++i) { double& v = q[bucket(c[i - 1])]; v = 0.9 * v + 0.1 * (i <= a ? 1.0 : 0.0); }
        }
    } cstop;
    bool cstop_on = [] { const char* e = getenv("V41_SPEC_CONF"); return !(e && *e == '0'); }();
    float* h_conf = nullptr;
    int last_k = 0;                                // drafts the last verify actually ran
    int verify(int tok, int pos, int K, const GenParams& p, std::mt19937_64& rng, std::deque<int>& out, const int* sfx = nullptr) {
        int M = K + 1;
        std::vector<int> rows(M);
        rows[0] = tok;
        double td = 0;
        const auto t_round = clk::now();
        if (sfx) for (int i = 1; i < M; ++i) rows[i] = sfx[i - 1];
        else {
            const int anchor = pos - 1;
            CK(cudaStreamSynchronize(st));
            h_ids[0] = anchor; h_ids[1] = tok;
            CK(cudaMemcpyAsync(D->d_anchor, &h_ids[0], 4, cudaMemcpyHostToDevice, st));
            CK(cudaMemcpyAsync(D->d_tok, &h_ids[1], 4, cudaMemcpyHostToDevice, st));
            launch(&vexec[-K], [&] { v41m::draft_body(m, *D, st, K); });
            CK(cudaMemcpyAsync(h_draft, D->d_out, (v41::DSPARK_BLOCK + 1) * 4, cudaMemcpyDeviceToHost, st));
            if (cstop_on) CK(cudaMemcpyAsync(h_conf, D->d_conf, v41::DSPARK_BLOCK * 4, cudaMemcpyDeviceToHost, st));
            CK(cudaStreamSynchronize(st));
            td = ms_since(t_round);
            if (cstop_on) { K = cstop.pick(h_conf, K); M = K + 1; rows.resize(M); }
            for (int i = 1; i < M; ++i) rows[i] = h_draft[i];
        }
        last_k = K;
        const auto t_fwd = clk::now();
        const bool fast = p.temperature <= 0.f && h2;
        const bool sfast = !fast && sample_fast(p);
        auto& vx = fast ? vexec2 : sfast ? vexec3 : vexec;
        const bool early = cfg.graph && vx[M] && early_launch;   // A6: see step()
        if (early) stage_dev(rows.data(), M, pos); else stage_rows(rows.data(), M, pos);
        if (!cfg.graph || !vx[M]) reap();             // eager: see step()
        if (early) { CK(cudaGraphLaunch(vx[M], st)); if (engram_on) submit_engram(store, rows.data(), M, pos); }
        else launch(&vx[M], [&] {
            v41m::embed_rows(m, s, M, st);
            s.route_budget = s7_budget;               // captured into this verify graph
            v41m::forward_layers(m, s, 0, v41::N_LAYERS - 1, M, 0, 0, eg, st, hook_for(M));
            s.route_budget = 0;
            if (fast) v41m::head_rows_greedy(m, s, 0, M, st);
            else if (sfast) v41m::head_rows_sample(m, s, 0, M, st);
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
            if (sfast) {
                CK(cudaMemcpyAsync(h_fb, s.d_hfb, M * 4, cudaMemcpyDeviceToHost, st));
                CK(cudaStreamSynchronize(st));
                std::vector<int> fbv(h_fb, h_fb + M);
                for (int i = 0; i < M; ++i) if (fbv[i] || h2s_check) sampled_fixup(p, i, fbv[i]);
            }
            for (;;) {                                 // sample row by row: stop at the first mismatch
                const int t = sample_row(p, rng, a);
                out.push_back(t);
                if (a < K && t == rows[a + 1]) { ++a; continue; }
                break;
            }
        }
        v41m::rollback_state(s, a, pos + a, st);
        kv_len = pos + 1 + a;
        if (cstop_on && !sfx) cstop.update(h_conf, K, a, td, ms_since(t_fwd));
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
    // C3: the full-head logits of row 0 (prefill tail or a mode-0 step) masked by p.mask, then picked
    int sample_masked(const GenParams& p, std::mt19937_64& rng) {
        if (!h_mask) {
            CK(cudaHostAlloc(&h_mask, MASK_WORDS * 4, 0)); CK(cudaMalloc(&d_mask, MASK_WORDS * 4));
        }
        memset(h_mask, 0, MASK_WORDS * 4);
        if (!p.mask(h_mask)) return sample(p, rng);
        CK(cudaMemcpyAsync(d_mask, h_mask, MASK_WORDS * 4, cudaMemcpyHostToDevice, st));
        k_logit_mask<<<(v41::VOCAB + 255) / 256, 256, 0, st>>>(s.logits, d_mask, v41::VOCAB);
        if (p.temperature <= 0.f) { v41m::argmax_logits(s, d_tok, st); return sample(p, rng); }
        return sample_row(p, rng, 0);
    }
    int sample(const GenParams& p, std::mt19937_64& rng, bool fast = false) {
        if (p.temperature <= 0.f) {
            CK(cudaMemcpyAsync(h_tok, d_tok, 4, cudaMemcpyDeviceToHost, st));
            if (fast) CK(cudaMemcpyAsync(h_fb, s.d_hfb, 4, cudaMemcpyDeviceToHost, st));
            CK(cudaStreamSynchronize(st));
            if (fast && *h_fb) return head_fallback(0);
            return *h_tok;
        }
        if (fast) {                                   // S11: the step ran head_rows_sample
            CK(cudaMemcpyAsync(h_fb, s.d_hfb, 4, cudaMemcpyDeviceToHost, st));
            CK(cudaStreamSynchronize(st));
            if (*h_fb || h2s_check) sampled_fixup(p, 0, *h_fb);
        }
        return sample_row(p, rng, 0);
    }
    // a sampled two-stage row: fallback (too many tokens in the window) -> the full head; V41_H2S_CHECK
    // also computes the full head and records the total variation between the two sampling distributions
    void sampled_fixup(const GenParams& p, int row, bool fb) {
        std::vector<float> a;
        if (!fb && h2s_check) {
            a.resize(v41::VOCAB);
            CK(cudaMemcpyAsync(a.data(), s.logits + (size_t)row * v41::VOCAB, v41::VOCAB * 4, cudaMemcpyDeviceToHost, st));
        }
        if (fb) ++h2_fallbacks;
        v41m::head_full_logits(m, s, row, st);
        if (a.empty()) { CK(cudaStreamSynchronize(st)); return; }
        std::vector<float> b(v41::VOCAB);
        CK(cudaMemcpyAsync(b.data(), s.logits + (size_t)row * v41::VOCAB, v41::VOCAB * 4, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
        auto probs = [&](const std::vector<float>& L) {
            float mx = -INFINITY; for (float v : L) mx = std::max(mx, v);
            std::vector<double> e(L.size()); double z = 0;
            for (size_t i = 0; i < L.size(); ++i) { e[i] = std::exp((double)(L[i] - mx) / p.temperature); z += e[i]; }
            for (auto& v : e) v /= z;
            return e;
        };
        const auto pa = probs(a), pb = probs(b);
        double tv = 0; for (size_t i = 0; i < pa.size(); ++i) tv += std::fabs(pa[i] - pb[i]);
        tv *= 0.5; h2s_tv_sum += tv; h2s_tv_max = std::max(h2s_tv_max, tv); ++h2s_n;
        if (h2s_check == 1) {                         // sample from the two-stage row, as without the check
            CK(cudaMemcpyAsync(s.logits + (size_t)row * v41::VOCAB, a.data(), v41::VOCAB * 4, cudaMemcpyHostToDevice, st));
            CK(cudaStreamSynchronize(st));
        }
    }
    // temperature / top-p sample from s.logits row r
    int sample_row(const GenParams& p, std::mt19937_64& rng, int row) {
        static const bool gpu = [] { const char* e = getenv("V41_GPU_SAMPLE"); return !(e && *e == '0'); }();
        if (gpu && p.top_p >= 1.f) {                  // S11: exact Gumbel-max draw on the GPU, 4 bytes back
            v41k::sample_gumbel(d_tok, s.logits + (size_t)row * v41::VOCAB, v41::VOCAB, p.temperature, rng(), st);
            CK(cudaMemcpyAsync(h_tok, d_tok, 4, cudaMemcpyDeviceToHost, st));
            CK(cudaStreamSynchronize(st));
            return *h_tok;
        }
        CK(cudaMemcpyAsync(h_logits.data(), s.logits + (size_t)row * v41::VOCAB, v41::VOCAB * 4, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
        const int V = v41::VOCAB;
        float mx = -INFINITY;
        for (int i = 0; i < V; ++i) mx = std::max(mx, h_logits[i]);
        std::vector<std::pair<float, int>> pr(V);
        double z = 0;
        for (int i = 0; i < V; ++i) { const float e = std::exp((h_logits[i] - mx) / p.temperature); pr[i] = {e, i}; z += e; }
        if (p.top_p >= 1.f) {          // no nucleus: draw in index order (same distribution, no sort)
            const double r = std::uniform_real_distribution<double>(0.0, z)(rng); double acc = 0;
            for (int i = 0; i < V; ++i) { acc += pr[i].first; if (r < acc) return i; }
            return V - 1;
        }
        // S11: the full sort of 129K pairs was ~9 ms a row on Thor's CPU. Sort a growing top-k prefix
        // instead: it is the full sort's prefix (same comparator), so cut and the draw are unchanged.
        auto cmp = [](const auto& a, const auto& b) { return a.first > b.first || (a.first == b.first && a.second < b.second); };
        int cut = V;
        for (int k = 256;; k *= 4) {
            k = std::min(k, V);
            std::nth_element(pr.begin(), pr.begin() + (k - 1), pr.end(), cmp);
            std::sort(pr.begin(), pr.begin() + k, cmp);
            double cum = 0; cut = -1;
            for (int i = 0; i < k; ++i) { cum += pr[i].first / z; if (cum >= p.top_p) { cut = i + 1; break; } }
            if (cut > 0) break;
            if (k == V) { cut = V; break; }
        }
        double zz = 0; for (int i = 0; i < cut; ++i) zz += pr[i].first;
        double r = std::uniform_real_distribution<double>(0.0, zz)(rng), acc = 0;
        for (int i = 0; i < cut; ++i) { acc += pr[i].first; if (r < acc) return pr[i].second; }
        return pr[cut - 1].second;
    }

    // C1 multi-session prefix cache. At the end of every generate the session's state is snapshotted
    // (device copies): the packed caches + compressor state of the KV sources up to kv_len, the last
    // C1_WIN window positions of every layer's ring (and the drafter's rings), and the host bookkeeping.
    // A later request whose prompt shares a longer prefix with a snapshot than with the resident session
    // restores it first; reuse_len then applies unchanged, so a hit is exactly the resident-session path
    // (the snapshot's hi_written is set so no restart reads a window slot older than the copied span).
    // V41_C1=0 off; V41_C1_MB: budget (LRU), default 2048.
    struct Snap {
        std::vector<int> toks;                     // ctx[0, kv_len)
        int hi = -1, floor = 0, w0 = 0;            // restored hi_written, reuse_floor; first window position copied
        uint8_t* buf = nullptr; size_t bytes = 0;
        uint64_t used = 0;
    };
    std::vector<Snap> snaps;
    uint64_t snap_clock = 0;
    bool c1_on = [] { const char* e = getenv("V41_C1"); return !(e && *e == '0'); }();
    size_t c1_budget = [] { const char* e = getenv("V41_C1_MB"); return (size_t)(e ? atol(e) : 2048) << 20; }();
    static constexpr int C1_WIN = 512;             // window positions kept: restarts within ~384 of the snapshot end
    size_t c1_bytes = 0;
    long c1_hits = 0, c1_saved_tokens = 0;
    int src_rows(int L, int n) const { return std::min(s.src[L].rows, n / std::max(1, v41::compress_ratio(L)) + 1); }
    // walk every copied region in a fixed order: f(device ptr, bytes, offset in the snapshot buffer)
    template <class F> size_t snap_walk(int n, int w0, F f) {
        size_t off = 0;
        auto reg = [&](void* p, size_t b) { f((uint8_t*)p, b, off); off += (b + 255) & ~(size_t)255; };
        for (int L : v41::KV_SOURCES) {
            v41m::State::Src& c = s.src[L];
            const size_t r = src_rows(L, n);
            reg(c.ckv, r * 256); reg(c.ckv_s, r * 32); reg(c.ik, r * 64); reg(c.ik_s, r * 4);
            reg(c.kv_state, 2 * 512 * 4); reg(c.score_state, 2 * 512 * 4);
        }
        auto ring = [&](float* base) {             // positions [w0, n): slot p % RING, at most two runs
            const int R = v41m::State::RING;
            for (int p = w0; p < n;) {
                const int sl = p % R, len = std::min(n - p, R - sl);
                reg(base + (size_t)sl * v41::HEAD_DIM, (size_t)len * v41::HEAD_DIM * 4);
                p += len;
            }
        };
        for (int L = 0; L < v41::N_LAYERS; ++L) ring(s.ring[L]);
        if (D) for (int i = 0; i < v41::N_MTP; ++i) ring(D->ring[i]);
        return off;
    }
    void c1_drop(size_t i) { cudaFree(snaps[i].buf); c1_bytes -= snaps[i].bytes; snaps.erase(snaps.begin() + i); }

    // C2: the NVMe tier under C1. V41_C1_DIR=dir turns it on (V41_C1_DISK_GB budget, default 64): a snapshot
    // the RAM tier evicts is copied to the host and written by one writer thread; lookups search both tiers;
    // a restarted engine re-indexes the files written for the same weights (fingerprint of ckpt, keepset,
    // overlay, drafter) and never touches any other file. Files: c1_<fp>_<n>.snap = header, ids, payload.
    struct DiskSnap { std::vector<int> toks; int hi = -1, floor = 0, w0 = 0; size_t bytes = 0; std::string path; uint64_t used = 0; };
    struct C1Hdr { char magic[8]; uint64_t fp; int32_t n, hi, floor, w0; uint64_t bytes; };
    static constexpr size_t C1_ALIGN = 4096;
    std::string c1_dir = [] { const char* e = getenv("V41_C1_DIR"); return std::string(e && *e ? e : ""); }();
    size_t c1_disk_budget = [] { const char* e = getenv("V41_C1_DISK_GB"); return (size_t)(e ? atol(e) : 64) << 30; }();
    uint64_t c1_fp = 0;
    std::vector<DiskSnap> dsnaps;                  // guarded by dmu
    size_t c1_disk_bytes = 0;
    std::mutex dmu; std::condition_variable dcv;
    std::deque<std::pair<DiskSnap, std::vector<uint8_t>>> dq; size_t dq_bytes = 0;
    std::vector<DiskSnap> dflight;                 // queued + being written (metadata only); guarded by dmu
    std::thread dwriter; bool dstop = false;
    uint8_t* c1_stage = nullptr; size_t c1_stage_bytes = 0;
    long c1_disk_hits = 0, c1_disk_writes = 0;
    static size_t c1_payload_off(int n) { return (sizeof(C1Hdr) + (size_t)n * 4 + C1_ALIGN - 1) / C1_ALIGN * C1_ALIGN; }

    void c1_disk_init() {
        if (c1_dir.empty() || !c1_on) return;
        std::string k = cfg.ckpt_dir + "|" + cfg.keepset + "|" + (getenv("DSV4_NVFP4_OVERLAY") ? getenv("DSV4_NVFP4_OVERLAY") : "") +
                        "|" + std::to_string(D != nullptr) + "|v1";
        uint64_t h = 1469598103934665603ull; for (unsigned char c : k) { h ^= c; h *= 1099511628211ull; }
        c1_fp = h;
        char pre[64]; snprintf(pre, sizeof pre, "c1_%016llx_", (unsigned long long)c1_fp);
        if (DIR* d = opendir(c1_dir.c_str())) {
            while (dirent* e = readdir(d)) {
                const std::string nm = e->d_name;
                if (nm.rfind(pre, 0) != 0 || nm.size() < 5 || nm.substr(nm.size() - 5) != ".snap") continue;
                DiskSnap ds; ds.path = c1_dir + "/" + nm;
                FILE* f = fopen(ds.path.c_str(), "rb"); if (!f) continue;
                C1Hdr hd; bool ok = fread(&hd, sizeof hd, 1, f) == 1 && !memcmp(hd.magic, "V41C1v1", 8) && hd.fp == c1_fp && hd.n > 0;
                if (ok) { ds.toks.resize(hd.n); ok = fread(ds.toks.data(), 4, hd.n, f) == (size_t)hd.n; }
                fclose(f);
                if (!ok) continue;
                ds.hi = hd.hi; ds.floor = hd.floor; ds.w0 = hd.w0; ds.bytes = hd.bytes;
                c1_disk_bytes += ds.bytes; dsnaps.push_back(std::move(ds));
            }
            closedir(d);
        } else { fprintf(stderr, "[v41] C1 disk tier: cannot open %s, off\n", c1_dir.c_str()); c1_dir.clear(); return; }
        dwriter = std::thread([this] { c1_writer(); });
        if (!cfg.quiet) printf("[engine] C1 disk tier %s: %zu snapshots indexed (%.2f GiB), budget %.0f GiB\n", c1_dir.c_str(),
                               dsnaps.size(), c1_disk_bytes / 1073741824.0, c1_disk_budget / 1073741824.0);
    }
    void c1_writer() {
        for (;;) {
            std::unique_lock<std::mutex> lk(dmu);
            dcv.wait(lk, [&] { return dstop || !dq.empty(); });
            if (dq.empty()) return;
            auto job = std::move(dq.front()); dq.pop_front();
            lk.unlock();
            DiskSnap& ds = job.first;
            const int n = (int)ds.toks.size();
            C1Hdr hd{}; memcpy(hd.magic, "V41C1v1", 8); hd.fp = c1_fp; hd.n = n; hd.hi = ds.hi; hd.floor = ds.floor; hd.w0 = ds.w0; hd.bytes = ds.bytes;
            const std::string tmp = ds.path + ".tmp";
            bool ok = false;
            if (FILE* f = fopen(tmp.c_str(), "wb")) {
                std::vector<uint8_t> head(c1_payload_off(n), 0);
                memcpy(head.data(), &hd, sizeof hd); memcpy(head.data() + sizeof hd, ds.toks.data(), (size_t)n * 4);
                ok = fwrite(head.data(), 1, head.size(), f) == head.size() && fwrite(job.second.data(), 1, ds.bytes, f) == ds.bytes;
                ok = (fclose(f) == 0) && ok;
            }
            if (ok) ok = rename(tmp.c_str(), ds.path.c_str()) == 0;
            if (!ok) { unlink(tmp.c_str()); fprintf(stderr, "[v41] C1 disk write %s failed\n", ds.path.c_str()); }
            lk.lock();
            dq_bytes -= ds.bytes;
            for (size_t i = 0; i < dflight.size(); ++i) if (dflight[i].path == ds.path) { dflight.erase(dflight.begin() + i); break; }
            dcv.notify_all();                          // an eviction may be waiting on the in-flight cap; a lookup on this file
            if (!ok) continue;
            ++c1_disk_writes;
            c1_disk_bytes += ds.bytes;
            dsnaps.push_back(std::move(ds));
            while (c1_disk_bytes > c1_disk_budget && dsnaps.size() > 1) {   // LRU among our own files
                size_t lru = 0; for (size_t i = 1; i < dsnaps.size(); ++i) if (dsnaps[i].used < dsnaps[lru].used) lru = i;
                unlink(dsnaps[lru].path.c_str()); c1_disk_bytes -= dsnaps[lru].bytes; dsnaps.erase(dsnaps.begin() + lru);
            }
        }
    }
    // RAM-tier eviction: to disk when the tier is on, else dropped
    void c1_evict(size_t i) {
        if (!c1_dir.empty()) {
            Snap& o = snaps[i];
            std::vector<uint8_t> h(o.bytes);
            CK(cudaMemcpy(h.data(), o.buf, o.bytes, cudaMemcpyDeviceToHost));
            DiskSnap ds; ds.toks = o.toks; ds.hi = o.hi; ds.floor = o.floor; ds.w0 = o.w0; ds.bytes = o.bytes; ds.used = o.used;
            char nm[96]; snprintf(nm, sizeof nm, "/c1_%016llx_%d_%llu.snap", (unsigned long long)c1_fp, (int)o.toks.size(),
                                  (unsigned long long)std::chrono::steady_clock::now().time_since_epoch().count());
            ds.path = c1_dir + nm;
            std::unique_lock<std::mutex> lk(dmu);
            dcv.wait(lk, [&] { return dq_bytes < ((size_t)4 << 30); });   // at most ~4 GiB of host copies in flight
            dq_bytes += ds.bytes;
            { DiskSnap m = ds; m.used = 0; dflight.push_back(std::move(m)); }
            dq.emplace_back(std::move(ds), std::move(h));
            dcv.notify_all();
        }
        c1_drop(i);
    }
    void c1_disk_stop() {
        if (!dwriter.joinable()) return;
        { std::lock_guard<std::mutex> lk(dmu); dstop = true; }
        dcv.notify_all(); dwriter.join();
    }
    void c1_save() {
        const int n = std::min(kv_len, (int)ctx.size());
        if (!c1_on || !prefix_cache || n < 2 * v41::WINDOW) return;
        for (uint64_t v : ctx_img) if (v) return;      // image sessions: the snapshot keys are token ids only
        const int w0 = std::max(0, n - C1_WIN);
        // superseded: an older snapshot of a prefix of this session whose usable restarts this one covers
        auto lowest = [](int w, int fl) { return std::max(w > 0 ? w + v41::WINDOW - 1 : 0, fl); };
        const int lo = lowest(w0, reuse_floor);
        for (size_t i = 0; i < snaps.size();) {
            const Snap& o = snaps[i];
            const bool prefix = (int)o.toks.size() <= n && std::equal(o.toks.begin(), o.toks.end(), ctx.begin());
            if (prefix && lowest(o.w0, o.floor) >= lo) c1_drop(i);
            else ++i;
        }
        if (!c1_dir.empty()) {                         // superseded files go too
            std::lock_guard<std::mutex> lk(dmu);
            for (size_t i = 0; i < dsnaps.size();) {
                const DiskSnap& o = dsnaps[i];
                const bool prefix = (int)o.toks.size() <= n && std::equal(o.toks.begin(), o.toks.end(), ctx.begin());
                if (prefix && lowest(o.w0, o.floor) >= lo) { unlink(o.path.c_str()); c1_disk_bytes -= o.bytes; dsnaps.erase(dsnaps.begin() + i); }
                else ++i;
            }
        }
        Snap sn;
        sn.bytes = snap_walk(n, w0, [](uint8_t*, size_t, size_t) {});
        if (sn.bytes > c1_budget) return;
        while (!snaps.empty() && c1_bytes + sn.bytes > c1_budget) {
            size_t lru = 0; for (size_t i = 1; i < snaps.size(); ++i) if (snaps[i].used < snaps[lru].used) lru = i;
            c1_evict(lru);
        }
        if (cudaMalloc(&sn.buf, sn.bytes) != cudaSuccess) { (void)cudaGetLastError(); return; }
        snap_walk(n, w0, [&](uint8_t* p, size_t b, size_t off) { CK(cudaMemcpyAsync(sn.buf + off, p, b, cudaMemcpyDeviceToDevice, st)); });
        CK(cudaStreamSynchronize(st));
        sn.toks.assign(ctx.begin(), ctx.begin() + n);
        sn.w0 = w0; sn.floor = reuse_floor;
        sn.hi = w0 > 0 ? w0 - 1 + v41m::State::RING : n - 1;   // window_reuse_ok then refuses restarts below w0 + 127
        sn.used = ++snap_clock;
        c1_bytes += sn.bytes;
        snaps.push_back(std::move(sn));
    }
    // restore the snapshot that serves `ids` best, if it beats the resident session by > 64 rows
    void c1_lookup(const std::vector<int>& ids) {
        if (!c1_on || !prefix_cache) return;
        std::unique_lock<std::mutex> dlk(dmu, std::defer_lock);
        if (!c1_dir.empty()) dlk.lock();
        if (snaps.empty() && dsnaps.empty() && dflight.empty()) return;
        const int kres = reuse_len(ids);
        int best = -1, bk = kres + 64;
        bool disk = false;
        auto usable = [&](const std::vector<int>& toks, int hi, int floor) {
            int c = 0; const int lim = std::min(toks.size(), ids.size());
            while (c < lim && toks[c] == ids[c] && !img_at(req_img, c)) ++c;
            if (c >= (int)ids.size()) c = (int)ids.size() - 1;
            c &= ~1;
            return v41spec::window_reuse_ok(c, hi, v41::WINDOW, v41m::State::RING) && c >= floor ? c : -1;
        };
        for (size_t i = 0; i < snaps.size(); ++i) {
            const int c = usable(snaps[i].toks, snaps[i].hi, snaps[i].floor);
            if (c > bk) { bk = c; best = (int)i; }
        }
        // an eviction still queued or being written would serve this request better: wait for it to land
        // (one ~45 MB file, well under a second) rather than re-prefill it (q9 run 1: follow-up missed, 7.5 s)
        for (;;) {
            std::string want;
            for (const DiskSnap& f : dflight) if (usable(f.toks, f.hi, f.floor) > bk) { want = f.path; break; }
            if (want.empty()) break;
            dcv.wait(dlk, [&] { for (const DiskSnap& f : dflight) if (f.path == want) return false; return true; });
        }
        for (size_t i = 0; i < dsnaps.size(); ++i) {          // RAM wins ties: only strictly longer from disk
            const int c = usable(dsnaps[i].toks, dsnaps[i].hi, dsnaps[i].floor);
            if (c > bk) { bk = c; best = (int)i; disk = true; }
        }
        if (best < 0) return;
        if (disk) {
            DiskSnap& o = dsnaps[best];
            const int n = (int)o.toks.size();
            if (snap_walk(n, o.w0, [](uint8_t*, size_t, size_t) {}) != o.bytes) return;   // other build: layout differs
            if (c1_stage_bytes < o.bytes) {
                if (c1_stage) cudaFreeHost(c1_stage);
                c1_stage_bytes = 0;
                if (cudaHostAlloc(&c1_stage, o.bytes, 0) != cudaSuccess) { (void)cudaGetLastError(); c1_stage = nullptr; return; }
                c1_stage_bytes = o.bytes;
            }
            const int fd = open(o.path.c_str(), O_RDONLY);
            if (fd < 0) return;
            size_t got = 0; const off_t base = (off_t)c1_payload_off(n);
            while (got < o.bytes) {
                const ssize_t r = pread(fd, c1_stage + got, std::min(o.bytes - got, (size_t)1 << 28), base + (off_t)got);
                if (r <= 0) break;
                got += (size_t)r;
            }
            close(fd);
            if (got != o.bytes) { fprintf(stderr, "[v41] C1 disk read %s short\n", o.path.c_str()); return; }
            CK(cudaStreamSynchronize(st));
            snap_walk(n, o.w0, [&](uint8_t* p, size_t b, size_t off) { CK(cudaMemcpyAsync(p, c1_stage + off, b, cudaMemcpyHostToDevice, st)); });
            CK(cudaStreamSynchronize(st));
            ctx = o.toks; ctx_img.clear(); kv_len = n; hi_written = o.hi; reuse_floor = o.floor;
            if (hasher) hasher->set_history(ctx.data(), n);
            o.used = ++snap_clock;
            ++c1_hits; ++c1_disk_hits; c1_saved_tokens += bk - kres;
            return;
        }
        Snap& o = snaps[best];
        const int n = (int)o.toks.size();
        CK(cudaStreamSynchronize(st));
        snap_walk(n, o.w0, [&](uint8_t* p, size_t b, size_t off) { CK(cudaMemcpyAsync(p, o.buf + off, b, cudaMemcpyDeviceToDevice, st)); });
        ctx = o.toks; ctx_img.clear(); kv_len = n; hi_written = o.hi; reuse_floor = o.floor;
        if (hasher) hasher->set_history(ctx.data(), n);
        o.used = ++snap_clock;
        ++c1_hits; c1_saved_tokens += bk - kres;
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
        while (k < kv_len && k < (int)ctx.size() && k < (int)ids.size() && ctx[k] == ids[k] && img_at(ctx_img, k) == img_at(req_img, k)) ++k;
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
    if (p_->gexec3) cudaGraphExecDestroy(p_->gexec3);
    for (auto& kv : p_->vexec3) if (kv.second) cudaGraphExecDestroy(kv.second);
    if (p_->h2s_n) fprintf(stderr, "[engine] S11 check: %ld sampled rows, TV vs full head mean %.3g max %.3g\n",
                           p_->h2s_n, p_->h2s_tv_sum / p_->h2s_n, p_->h2s_tv_max);
    p_->c1_disk_stop();
    if (p_->c1_hits) fprintf(stderr, "[engine] C1 prefix cache: %ld hits (%ld from disk), %ld prompt rows not recomputed; %ld disk writes\n",
                             p_->c1_hits, p_->c1_disk_hits, p_->c1_saved_tokens, p_->c1_disk_writes);
    for (auto& sn : p_->snaps) cudaFree(sn.buf);
    for (auto& e : p_->vcache) cudaFree(e.rows);
    if (p_->vit_runs + p_->vit_cached + p_->vit_skipped)
        fprintf(stderr, "[engine] vision: %ld tower runs, %ld encoder-cache hits, %ld images inside reused prefixes\n",
                p_->vit_runs, p_->vit_cached, p_->vit_skipped);
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
    I.chunk = std::max(1, std::min({I.cfg.ext_chunk, v41m::State::MAX_CHUNK}));
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
        // S11, default on (V41_HEAD2_SAMPLE=0 off): sampled rows exact within T ln(1e9) + delta of the max, cap 8192
        if (const char* e2 = getenv("V41_HEAD2_SAMPLE"); !(e2 && *e2 == '0')) {
            const char* c2 = getenv("V41_HEAD2_SAMPLE_CAP");
            v41m::enable_head2_sample(I.m, I.s, c2 ? std::max(8, atoi(c2) / 8 * 8) : 8192, I.st);
            I.h2s = true;
            if (const char* ck = getenv("V41_H2S_CHECK")) I.h2s_check = atoi(ck);
            if (!I.cfg.quiet) printf("[engine] two-stage sampled head: exact within T*ln(1e9)+%.3g logits (cap %d)%s\n", delta,
                                     I.s.hd2s.cap, I.h2s_check ? ", checking TV against the full head" : "");
        }
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
        I.D->conf_on = I.cstop_on;
        CK(cudaHostAlloc(&I.h_conf, v41::DSPARK_BLOCK * 4, 0));
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
        const char* pf = getenv("V41_ENGRAM_PREFETCH");
        if (!(pf && *pf == '0')) {
            if (!I.store2.open(ec, &err)) throw std::runtime_error("Engram row store (prefetch) at " + ec.dir + ": " + err);
            I.engram_prefetch = true;
        }
        I.hasher.reset(new EngramHasher(I.store.tables()));
        I.engram_on = true;
        I.eg.store = &I.store; I.eg.graph_wait = true;
        if (!I.cfg.quiet) printf("[engine] Engram row store %s (format %d, %d B rows)\n", ec.dir.c_str(), I.store.format(), I.store.row_bytes());
    }
    // V2 vision tower: loaded when the build has vision.safetensors (V41_VISION=0: never)
    if (const char* ve = getenv("V41_VISION"); !(ve && *ve == '0')) {
        if (FILE* f = fopen((I.cfg.ckpt_dir + "/vision.safetensors").c_str(), "rb")) {
            fclose(f);
            const auto tv = clk::now();
            I.V.reset(new v41v::Vision);
            v41v::load_vision(*I.V, I.cfg.ckpt_dir, 9216);
            CK(cudaMalloc(&I.d_al, (size_t)1024 * v41::DIM * 4));
            CK(cudaMalloc(&I.d_patches, (size_t)I.V->max_patches * v41v::PATCH_IN * 2));
            CK(cudaMalloc(&I.d_spansrc, 4096 * 4));
            if (!I.cfg.quiet) printf("[engine] vision tower %.2f GiB (weights + scratch for %d patches), %.0f s\n", I.V->gib,
                                     I.V->max_patches, ms_since(tv) / 1e3);
        }
    }
    CK(cudaMalloc(&I.d_imgrow, I.chunk * 4)); CK(cudaHostAlloc(&I.h_imgrow, I.chunk * 4, 0));
    CK(cudaHostAlloc(&I.h_pos, I.chunk * 4, 0)); CK(cudaHostAlloc(&I.h_ids, I.chunk * 4, 0));
    CK(cudaHostAlloc(&I.h_img, I.chunk, 0)); CK(cudaHostAlloc(&I.h_tok, 4, 0)); CK(cudaHostAlloc(&I.h_fb, 4 * std::max(I.chunk, 8), 0));
    CK(cudaMalloc(&I.d_tok, 4));
    CK(cudaHostAlloc(&I.h_draft, (v41::DSPARK_BLOCK + 1) * 4, 0)); CK(cudaHostAlloc(&I.h_tgt, I.chunk * 4, 0));
    I.hid.resize((size_t)I.chunk * 2 * v41::ENGRAM_COLS);
    I.h_logits.resize(v41::VOCAB);
    I.c1_disk_init();
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
    I.plan_images(ids, p.images);
    I.c1_lookup(ids);
    const int k = I.reuse_len(ids);
    if (!p.images.empty()) I.encode_images(p.images, k);
    gs.cached_tokens = k;
    if (k == 0) I.reset_all();
    auto t0 = clk::now();
    I.prefill(ids, k);
    I.ctx.assign(ids.begin(), ids.end());
    I.ctx_img = I.req_img;
    I.pos_img.clear();                              // the span rows are written; nothing after the prompt is an image
    std::deque<int> q;                              // committed, not yet emitted; only q.front() may lack KV
    const bool masked = (bool)p.mask;
    q.push_back(masked ? I.sample_masked(p, rng) : I.sample(p, rng));   // the prefill tail ran the full head
    gs.prefill_ms = ms_since(t0);
    t0 = clk::now();
    const bool spec = I.D != nullptr && I.active_k > 0 && !masked;
    int emitted_by_spec = 0;
    const long fb0 = I.h2_fallbacks;
    if (I.adapt_on) I.adapt.reset_accept();
    const int sfx_cap = std::min({I.sfx_kmax, I.chunk - 1, I.D ? I.D->max_rows - 1 : I.chunk - 1});
    if (I.sfx_on) {
        I.sfx.clear(); I.sfx.sync(I.ctx);
        if (I.sfx.Tb == 0) { I.sfx.Tb = spec ? 90.0 : 40.0; I.sfx.Eb = spec ? 2.6 : 1.0; }   // Thor priors until measured
    }
    std::vector<int> sprop;
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
        const int kcap = std::min(I.active_k, p.max_tokens - (n + 1) - 1);
        const bool adapt = spec && I.adapt_on && kcap == I.active_k;      // the tail's capped rounds are not timed
        const int K = adapt ? I.adapt.pick(kcap) : kcap;
        const auto tr = clk::now();
        int sk = 0, sL = 0;
        if (I.sfx_on && !masked) {
            I.sfx.sync(I.ctx);
            int from = 0;
            sL = I.sfx.find(I.ctx, &from);
            const int kc = std::min(sfx_cap, p.max_tokens - (n + 1) - 1);
            if (sL > 0 && kc > 0) {
                sk = I.sfx.pick(sL, (int)I.ctx.size() - from, kc);
                if (sk > 0) sprop.assign(I.ctx.begin() + from, I.ctx.begin() + from + sk);
            }
        }
        if (sk > 0) {
            const int a = I.verify(tok, pos, sk, p, rng, q, sprop.data());
            I.sfx.upd_sfx(sL, sk, a, ms_since(tr));
            ++gs.sfx_rounds; gs.sfx_accepted += a;
        } else if (spec && K > 0) {
            const int a = I.verify(tok, pos, K, p, rng, q);
            emitted_by_spec += a + 1;
            if (I.last_k < SpecProfile::MAXW) ++gs.spec.accept_joint[I.last_k][a];
            if (adapt) I.adapt.update(K, a, ms_since(tr));
            if (I.sfx_on) I.sfx.upd_base(K + 1, a + 1, ms_since(tr));
        } else if (masked) {
            I.step(tok, pos);                         // mode 0: full-head logits in row 0
            I.kv_len = pos + 1;
            q.push_back(I.sample_masked(p, rng));
        } else {
            const bool greedy = p.temperature <= 0.f;
            const bool sfast = I.sample_fast(p);
            I.step(tok, pos, greedy, sfast);
            I.kv_len = pos + 1;
            q.push_back(I.sample(p, rng, (greedy && I.h2) || sfast));
            if (spec) ++gs.spec.accept_joint[0][0];
            if (adapt) I.adapt.update(0, 0, ms_since(tr));
            if (I.sfx_on) I.sfx.upd_base(1, 1, ms_since(tr));
        }
        ++gs.verifies;
    }
    I.kv_len = std::min(I.kv_len, (int)I.ctx.size());
    gs.decode_ms = ms_since(t0);
    I.c1_save();
    gs.head2_fallbacks = (int)(I.h2_fallbacks - fb0);
    gs.tok_per_s = gs.decode_ms > 0 ? (gs.completion_tokens - 1) / (gs.decode_ms / 1e3) : 0;
    gs.tok_per_verify = gs.verifies ? (double)(gs.completion_tokens - 1) / gs.verifies : 0;
    if (!spec && gs.verifies < SpecProfile::MAXW) gs.spec.accept_joint[1][0] = gs.verifies;
    (void)emitted_by_spec;
    return gs;
}

bool Engine::has_vision() const { return p_->V != nullptr; }
void Engine::reset() { p_->ctx.clear(); p_->ctx_img.clear(); p_->kv_len = 0; if (p_->loaded) p_->reset_all(); else p_->reuse_floor = 0; }
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

std::vector<uint16_t> Engine::debug_taps(const std::vector<int>& ids, int from) {
    Impl& I = *p_;
    if (!I.D) throw std::runtime_error("debug_taps: no drafter loaded");
    I.reset_all(); I.ctx.clear(); I.kv_len = 0;
    const int n = (int)ids.size(), row = 3 * v41::DIM;
    from = std::max(0, std::min(from, n));
    std::vector<uint16_t> out((size_t)(n - from) * row);
    std::vector<float> buf((size_t)I.chunk * row);
    size_t odd = 0;
    for (int c0 = 0; c0 < n; c0 += I.chunk) {         // Impl::prefill's chunk loop, taps copied per chunk
        const int M = std::min(I.chunk, n - c0);
        I.stage_rows(ids.data() + c0, M, c0);
        v41m::embed_rows(I.m, I.s, M, I.st);
        I.forward_chunk(M, n, c0);
        const int lo = std::max(c0, from);
        if (lo >= c0 + M) continue;
        const size_t cnt = (size_t)(c0 + M - lo) * row;
        CK(cudaMemcpyAsync(buf.data(), I.D->taps + (size_t)(lo - c0) * row, cnt * 4, cudaMemcpyDeviceToHost, I.st));
        CK(cudaStreamSynchronize(I.st));
        uint16_t* o = out.data() + (size_t)(lo - from) * row;
        for (size_t i = 0; i < cnt; ++i) {            // bf16r values: the low half is zero
            uint32_t u; std::memcpy(&u, &buf[i], 4);
            odd += (u & 0xFFFF) != 0; o[i] = (uint16_t)(u >> 16);
        }
    }
    if (odd) throw std::runtime_error("debug_taps: " + std::to_string(odd) + " tap values are not bf16");
    I.reset_all(); I.ctx.clear(); I.kv_len = 0;
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

// -log p(forced[i]) per forced token (the debug_forced_logits stepping, one logits row on the host): long
// reference runs whose full logits would not fit beside the weights (0007 candidate-mask check at 32K)
std::vector<float> Engine::debug_forced_nll(const std::vector<int>& ids, const std::vector<int>& forced,
                                            std::vector<float>* top_lp, std::vector<int>* top_id) {
    Impl& I = *p_;
    I.reset_all();
    I.prefill(ids, 0);
    I.ctx.assign(ids.begin(), ids.end());
    std::vector<float> row(v41::VOCAB), nll(forced.size());
    std::vector<int> ord(v41::VOCAB);
    if (top_lp) { top_lp->assign(forced.size() * 32, 0.f); top_id->assign(forced.size() * 32, 0); }
    for (size_t i = 0; i < forced.size(); ++i) {
        CK(cudaMemcpyAsync(row.data(), I.s.logits, v41::VOCAB * 4, cudaMemcpyDeviceToHost, I.st));
        CK(cudaStreamSynchronize(I.st));
        float mx = row[0]; for (float v : row) mx = std::max(mx, v);
        double se = 0; for (float v : row) se += std::exp((double)v - mx);
        const double lse = mx + std::log(se);
        nll[i] = (float)(lse - row[forced[i]]);
        if (top_lp) {                                  // top-32 log-probs (s03 teacher format for v41-teacher-kld)
            for (int r = 0; r < v41::VOCAB; ++r) ord[r] = r;
            std::partial_sort(ord.begin(), ord.begin() + 32, ord.end(), [&](int a, int b) { return row[a] > row[b] || (row[a] == row[b] && a < b); });
            for (int j = 0; j < 32; ++j) { (*top_lp)[i * 32 + j] = (float)(row[ord[j]] - lse); (*top_id)[i * 32 + j] = ord[j]; }
        }
        if (i + 1 == forced.size()) break;
        const int pos = (int)I.ctx.size();
        I.ctx.push_back(forced[i]);
        I.step(forced[i], pos);
        I.kv_len = pos + 1;
    }
    I.reset_all(); I.ctx.clear(); I.kv_len = 0;
    return nll;
}

} // namespace dsv4srv
