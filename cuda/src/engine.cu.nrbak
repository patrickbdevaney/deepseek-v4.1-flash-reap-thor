// engine.cu — the persistent serving engine (see include/dsv4_engine.h for why it is not decode.cu).
//
// Carries exactly the shipping configuration and nothing else:
//   block 6, adaptK 1.5, VKPLUS on, the embedded mtp.0/1/2 DSpark head, greedy-or-sampled verify.
// No sweeps, no probes, no env knobs that change numerics. Everything here that touches the model
// is the same call, in the same order, with the same arguments as src/decode.cu's shipping path --
// tests/gate_engine.cu is what holds that claim to account.
#include "dsv4_engine.h"
#include "kv_pack.h"
#include "dsv4_load.h"
#include "nvfp4_dense.h"
#include "block.h"
#include "compressed_block.h"
#include "block_decode.h"
#include "hc.h"
#include "mla_attn.h"
#include "compressor.h"
#include "dscratch.h"
#include "dprof.h"
#define CU_V(x) do{ cudaError_t e_=(x); if(e_){ fprintf(stderr,"cuda %s @%d\n",cudaGetErrorString(e_),__LINE__); abort(); } }while(0)
#include "dspark_real.h"
#include "dspark_attn.h"
#include "yarn.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>

using namespace dsv4;
using namespace dsv4load;
#define CU(x) do{cudaError_t e=(x); if(e){fprintf(stderr,"cuda %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)

// Kernel-selection globals, defined in the kernel TUs at GLOBAL scope. Declaring them inside
// namespace dsv4srv would declare five different variables that happen to share a spelling, link
// nowhere, and -- if they had linked -- leave the real ones at their defaults.
extern bool g_tc_fp8;
extern bool g_tc_ogroup;
extern bool g_moe_grouped;
extern bool g_moe_gemv;
extern bool g_compressor_bf16;
extern int  g_kv_winmax;

namespace dsv4srv {

// ---- sampling -----------------------------------------------------------------------------------
// temperature 0 -> argmax, which is bit-identical to the measured decode path. Anything else draws
// from the target's own conditional, which keeps speculation exact (see dsv4_engine.h).
static int sample_row(const float* row, int vocab, const GenParams& p, std::mt19937_64& rng) {
    if (p.temperature <= 0.0f) {
        int am = 0;
        for (int v = 1; v < vocab; ++v) if (row[v] > row[am]) am = v;
        return am;
    }
    // top-p over a softmax at temperature. Sorting 129280 floats per token would cost more than the
    // forward that produced them, so: take the max for stability, exponentiate, then partial-sort
    // only the head of the distribution that top-p can possibly reach.
    float mx = row[0];
    for (int v = 1; v < vocab; ++v) if (row[v] > mx) mx = row[v];
    std::vector<std::pair<float,int>> pr(vocab);
    const float inv_t = 1.0f / p.temperature;
    double sum = 0;
    for (int v = 0; v < vocab; ++v) {
        const float e = expf((row[v] - mx) * inv_t);
        pr[v] = { e, v };
        sum += e;
    }
    const float top_p = (p.top_p <= 0.f || p.top_p > 1.f) ? 1.f : p.top_p;
    size_t keep = vocab;
    if (top_p < 1.0f) {
        // Nucleus: enough of the head to cover top_p mass. Grow the partial sort geometrically so
        // the common case (a peaked distribution) touches a few hundred entries, not 129k.
        keep = 0;
        double acc = 0;
        for (size_t cap = 64; ; cap = cap * 4) {
            if (cap > pr.size()) cap = pr.size();
            std::partial_sort(pr.begin(), pr.begin() + cap, pr.end(),
                              [](const auto& a, const auto& b) { return a.first > b.first; });
            acc = 0; keep = 0;
            for (size_t i = 0; i < cap; ++i) { acc += pr[i].first; ++keep; if (acc >= top_p * sum) break; }
            if (acc >= top_p * sum || cap == pr.size()) break;
        }
        sum = acc;
    }
    std::uniform_real_distribution<double> U(0.0, 1.0);
    double r = U(rng) * sum, c = 0;
    for (size_t i = 0; i < keep; ++i) { c += pr[i].first; if (c >= r) return pr[i].second; }
    return pr[keep ? keep - 1 : 0].second;
}

// ---- the engine ---------------------------------------------------------------------------------
struct Engine::Impl {
    EngineConfig cfg;
    bool is_loaded = false;
    bool use_prefix_cache = true;

    std::unique_ptr<st::WeightStore> Wp;
    std::unique_ptr<Loader> Lp;
    std::vector<void*> keep;

    int BLK = 6, VBMAX = 7, EXT_CHUNK = 64;
    int MAXB = 64;                 // allocated batch width of the verify/extend buffers
    int d = DIM, hc = HC_MULT, half = ROPE_DIM / 2;

    std::vector<LayerKV> KV;
    std::vector<BlockWeights> BW;
    std::vector<CompressedBlockWeights> CW;

    int *d_ids = nullptr;
    float *h0 = nullptr, *collapsed = nullptr, *logits = nullptr;
    const void* head_bf = nullptr;
    const float *norm_w = nullptr, *hc_fn = nullptr, *hc_sc = nullptr, *hc_bs = nullptr;
    const __nv_bfloat16* emb = nullptr;

    // per-layer MoE pointer tables (must outlive the weight structs that point into them)
    std::vector<std::vector<const uint8_t*>> P1, P2, P3, S18, S28, S38;

    // DSpark MTP head
    int NSTAGE = 0, NE = 0;
    std::vector<BlockWeights> mb;
    std::vector<float*> mkv;
    std::vector<std::vector<const uint8_t*>> HP1, HP2, HP3, HS1, HS2, HS3;
    const float *blk_cos = nullptr, *blk_sin = nullptr;
    const uint8_t* main_proj = nullptr;
    const float *main_proj_s = nullptr, *main_norm = nullptr;
    const float *hh_fn = nullptr, *hh_sc = nullptr, *hh_ba = nullptr, *hnorm = nullptr;
    const void *mw1 = nullptr, *mw2 = nullptr;

    float *main_x = nullptr;
    int *dbid = nullptr, *dfid = nullptr, *dout = nullptr;
    float *dmarg = nullptr, *xemb = nullptr, *xa = nullptr, *xb = nullptr;
    float *hv = nullptr, *hv2 = nullptr, *collK = nullptr, *logK = nullptr, *mh_v = nullptr;

    // resident context
    std::vector<int> ctx;          // committed token ids, ctx.size() == cpos
    int cpos = 0;
    // LADDER 1.0. Rows [0, mkv_valid) of every mkv[st] are already built from the CURRENT main_x.
    // `main_x` is write-once below the committed position, so those rows never go stale on their
    // own -- only a path that rewrites or discards main_x rows can invalidate them, and there are
    // exactly three: prefill_full (memsets), extend (writes [from,n)), rewind_to (drops to n).
    // Each clamps this down. Everything else may only grow it.
    int mkv_valid = 0;

    ~Impl() {
        for (void* p : keep) cudaFree(p);
        // comp_kv is a VIEW into win_kv when the combined cache is on -- freeing it then is a
        // double free. It is a real allocation only on the DSV4_KV_COMBINED=0 A/B path.
        for (auto& k : KV) { cudaFree(k.win_kv); cudaFree(k.idx_ckv); cudaFree(k.xin);
                             if (!g_kv_winmax) cudaFree(k.comp_kv); }
        for (float* p : mkv) cudaFree(p);
        for (void* p : { (void*)d_ids, (void*)h0, (void*)collapsed, (void*)logits,
                         (void*)main_x, (void*)dbid, (void*)dfid, (void*)dout, (void*)dmarg,
                         (void*)xemb, (void*)xa, (void*)xb, (void*)hv, (void*)hv2, (void*)collK,
                         (void*)logK, (void*)mh_v })
            cudaFree(p);
    }

    void say(const char* fmt, ...) {
        if (cfg.quiet) return;
        va_list a; va_start(a, fmt); vprintf(fmt, a); va_end(a); fflush(stdout);
    }

    void build_layer(int Lyr, const float* slide_cos, const float* slide_sin,
                     const float* cqc, const float* cqs, const float* cc4c, const float* cc4s,
                     const float* cc128c, const float* cc128s);
    void load();
    void prefill_full(const std::vector<int>& ids);
    void extend(const std::vector<int>& ids, int from);
    void rewind_to(int n);
    std::vector<float> debug_prefix_logits(const std::vector<int>& ids);
    GenStats generate(const std::vector<int>& ids, const GenParams& p,
                      const std::function<bool(int)>& on_token);
};

void Engine::Impl::build_layer(int Lyr, const float* slide_cos, const float* slide_sin,
                               const float* cqc, const float* cqs, const float* cc4c, const float* cc4s,
                               const float* cc128c, const float* cc128s) {
    Loader& L = *Lp;
    st::WeightStore& W = *Wp;
    const int ratio = compress_ratio(Lyr);
    const std::string lp = "layers." + std::to_string(Lyr) + ".";

    auto fill_moe = [&](const std::string& pfx, bool is_hash, MoEWeights& m, int idx) {
        const std::string p = pfx + "ffn.";
        auto &p1 = P1[idx], &p2 = P2[idx], &p3 = P3[idx];
        auto &s1 = S18[idx], &s2 = S28[idx], &s3 = S38[idx];
        p1.clear(); p2.clear(); p3.clear(); s1.clear(); s2.clear(); s3.clear();
        m.gate_w = L.bf16(p + "gate.weight"); m.is_hash = is_hash;
        m.gate_bias = is_hash ? nullptr : (W.has(p + "gate.bias") ? L.f32(p + "gate.bias") : nullptr);
        m.tid2eid = is_hash ? (const long*)W.get(p + "gate.tid2eid").dev : nullptr;
        for (int e = 0; e < N_ROUTED; ++e) {
            const std::string ep = p + "experts." + std::to_string(e) + ".";
            p1.push_back(L.raw(ep + "w1.weight")); p2.push_back(L.raw(ep + "w2.weight")); p3.push_back(L.raw(ep + "w3.weight"));
            s1.push_back(L.raw(ep + "w1.scale")); s2.push_back(L.raw(ep + "w2.scale")); s3.push_back(L.raw(ep + "w3.scale"));
        }
        m.w1p = p1.data(); m.w2p = p2.data(); m.w3p = p3.data();
        m.e8m0_scales = true; m.w1sp8 = s1.data(); m.w2sp8 = s2.data(); m.w3sp8 = s3.data();
        const std::string sp = p + "shared_experts.";
        m.sw1 = L.raw(sp + "w1.weight"); m.sw2 = L.raw(sp + "w2.weight"); m.sw3 = L.raw(sp + "w3.weight");
        m.sw1s = L.scale(sp + "w1.scale"); m.sw2s = L.scale(sp + "w2.scale"); m.sw3s = L.scale(sp + "w3.scale");
        m.n_routed = N_ROUTED; m.n_act = N_ACT; m.dim = DIM; m.inter = MOE_INTER; m.vocab = VOCAB;
        m.route_scale = ROUTE_SCALE; m.swiglu_limit = SWIGLU_LIMIT;
        m.use_tc_pp = true; m.batched = true; m.device_route = true;
    };
    auto fill_attn = [&](const std::string& pfx, MLAWeights& a, bool compressed) {
        const std::string p = pfx + "attn.";
        a.wq_a = L.raw(p + "wq_a.weight"); a.wq_a_s = L.scale(p + "wq_a.scale");
        a.wq_b = L.raw(p + "wq_b.weight"); a.wq_b_s = L.scale(p + "wq_b.scale");
        a.wkv = L.raw(p + "wkv.weight"); a.wkv_s = L.scale(p + "wkv.scale");
        a.wo_b = L.raw(p + "wo_b.weight"); a.wo_b_s = L.scale(p + "wo_b.scale");
        a.q_norm = L.bf16(p + "q_norm.weight"); a.kv_norm = L.bf16(p + "kv_norm.weight");
        a.wo_a_native = true; a.wo_a_fp8 = L.raw(p + "wo_a.weight"); a.wo_a_sc = L.raw(p + "wo_a.scale");
        a.attn_sink = L.f32(p + "attn_sink");
        a.cosT = compressed ? cqc : slide_cos; a.sinT = compressed ? cqs : slide_sin;
    };

    if (ratio == 0) {
        BlockWeights& b = BW[Lyr];
        fill_attn(lp, b.attn, false);
        fill_moe(lp, is_hash_layer(Lyr), b.ffn, Lyr);
        b.attn_norm = L.bf16(lp + "attn_norm.weight"); b.ffn_norm = L.bf16(lp + "ffn_norm.weight");
        b.hc_attn_fn = L.f32(lp + "hc_attn_fn"); b.hc_attn_scale = L.f32(lp + "hc_attn_scale"); b.hc_attn_base = L.f32(lp + "hc_attn_base");
        b.hc_ffn_fn = L.f32(lp + "hc_ffn_fn"); b.hc_ffn_scale = L.f32(lp + "hc_ffn_scale"); b.hc_ffn_base = L.f32(lp + "hc_ffn_base");
        b.dim = DIM; b.hc = HC_MULT;
    } else {
        CompressedBlockWeights& b = CW[Lyr];
        fill_attn(lp, b.attn.attn, true);
        const std::string p = lp + "attn.";
        b.attn.mc_wkv = L.bf16(p + "compressor.wkv.weight");
        b.attn.mc_wgate = L.bf16(p + "compressor.wgate.weight");
        b.attn.mc_ape = L.f32(p + "compressor.ape"); b.attn.mc_norm = L.bf16(p + "compressor.norm.weight");
        b.attn.cc_cos = (ratio == 4) ? cc4c : cc128c; b.attn.cc_sin = (ratio == 4) ? cc4s : cc128s;
        if (ratio == 4) {
            b.attn.idx_wq_b = L.raw(p + "indexer.wq_b.weight"); b.attn.idx_wq_b_s = L.scale(p + "indexer.wq_b.scale");
            b.attn.idx_weights_proj = L.bf16(p + "indexer.weights_proj.weight");
            b.attn.idx_c_wkv = L.bf16(p + "indexer.compressor.wkv.weight");
            b.attn.idx_c_wgate = L.bf16(p + "indexer.compressor.wgate.weight");
            b.attn.idx_c_ape = L.f32(p + "indexer.compressor.ape");
            b.attn.idx_c_norm = L.bf16(p + "indexer.compressor.norm.weight");
        }
        b.attn.index_n_heads = INDEX_N_HEADS; b.attn.index_head_dim = INDEX_HEAD_DIM; b.attn.index_topk = INDEX_TOPK;
        fill_moe(lp, is_hash_layer(Lyr), b.ffn, Lyr);
        b.attn_norm = L.bf16(lp + "attn_norm.weight"); b.ffn_norm = L.bf16(lp + "ffn_norm.weight");
        b.hc_attn_fn = L.f32(lp + "hc_attn_fn"); b.hc_attn_scale = L.f32(lp + "hc_attn_scale"); b.hc_attn_base = L.f32(lp + "hc_attn_base");
        b.hc_ffn_fn = L.f32(lp + "hc_ffn_fn"); b.hc_ffn_scale = L.f32(lp + "hc_ffn_scale"); b.hc_ffn_base = L.f32(lp + "hc_ffn_base");
        b.dim = DIM; b.hc = HC_MULT; b.win = WINDOW; b.ratio = ratio;
    }
}

void Engine::Impl::load() {
    const int seqmax = cfg.seqmax;
    BLK = cfg.blk;
    VBMAX = BLK + 1;

    say("[engine] loading %s (seqmax=%d, blk=%d, adaptK=%.2f)...\n", cfg.ckpt_dir.c_str(), seqmax, BLK, cfg.adaptK);
    Wp.reset(new st::WeightStore(cfg.ckpt_dir, key_map));
    Lp.reset(new Loader(*Wp));
    st::WeightStore& W = *Wp;
    Loader& L = *Lp;
    say("[engine] loaded %.2f GiB, %zu tensors (%s)\n", W.loadedGiB(), W.count(),
        W.managed() ? "MANAGED device-preferred" : "mapped-host");

    if (const char* ov = getenv("DSV4_NVFP4_OVERLAY")) {
        struct Ctx { st::WeightStore* w; } cx{ &W };
        nvfp4_load_overlay(ov, [](const char* name, void* c) -> const uint8_t* {
            auto* k = static_cast<Ctx*>(c);
            try { return (const uint8_t*)k->w->get(name).dev; } catch (...) { return nullptr; }
        }, &cx);
    }
    // Same global kernel-selection flags the shipping decode path sets. Not knobs: these ARE the
    // measured-best configuration (F31 for the MoE GEMV, F32 for the compressor).
    ::g_tc_fp8 = true;
    ::g_tc_ogroup = true;
    ::g_moe_grouped = true;
    ::g_moe_gemv = true;
    ::g_compressor_bf16 = false;

    std::vector<float> ssc, sss;
    yarn::freqs(ssc, sss, seqmax, ROPE_DIM, 0, ROPE_THETA, YARN_FACTOR, YARN_BETA_FAST, YARN_BETA_SLOW);
    const float *slide_cos = up_f(ssc, keep), *slide_sin = up_f(sss, keep);
    std::vector<float> cqc_h, cqs_h;
    yarn::freqs(cqc_h, cqs_h, seqmax, ROPE_DIM, YARN_ORIG_MAXPOS, COMPRESS_ROPE_THETA, YARN_FACTOR, YARN_BETA_FAST, YARN_BETA_SLOW);
    const float *cqc = up_f(cqc_h, keep), *cqs = up_f(cqs_h, keep);
    const float *cc4c = up_f(stride_rows(cqc_h, seqmax, half, 4), keep);
    const float *cc4s = up_f(stride_rows(cqs_h, seqmax, half, 4), keep);
    const float *cc128c = up_f(stride_rows(cqc_h, seqmax, half, 128), keep);
    const float *cc128s = up_f(stride_rows(cqs_h, seqmax, half, 128), keep);

    KV.assign(N_LAYERS, LayerKV());
    // The engine owns the xin allocation, so the engine is what enables the ring (compressor.cu
    // keeps it off by default because decode.cu and the unit gates share these kernels and allocate
    // the full history). Set before the loop below, which sizes xin from it.
    g_xin_ring_on = getenv("DSV4_XIN_RING") ? (atoi(getenv("DSV4_XIN_RING")) != 0) : true;
    // The ring has to survive the widest batch that will ever be written into it in one go, which is
    // MAXB (prefill and extend both move in EXT_CHUNK-sized blocks). MAXB is computed further down
    // from cfg.ext_chunk, so recompute it here rather than reordering the load.
    { const int ec = cfg.ext_chunk > 0 ? cfg.ext_chunk : 64;
      g_xin_ring_batch = ec > VBMAX ? ec : VBMAX; }
    // COMBINED KV CACHE. sparse_attn is O(topk) -- it gathers ~640 rows by index and uses `n` only
    // as a stride -- but the decode path used to dmalloc a (pos+T) x HEAD_DIM buffer and memcpy
    // win_kv and comp_kv into it EVERY STEP, PER LAYER, purely to make those rows contiguous. At 16K
    // context that is 1.76 GB copied per token to serve 1.3 MB of reads, and it is why decode
    // degraded with context on an architecture whose whole point is that it should not.
    //
    // Allocating the two as ONE buffer with comp_kv at a fixed stride removes the copy for free:
    // same bytes, same layout the copy was producing, just produced once instead of every step.
    // g_kv_winmax tells the kernels the stride; 0 leaves the old copy path for decode.cu and the
    // unit gates, which still allocate separately.
    // DEFAULT OFF -- the combined cache LOST its A/B and the hypothesis behind it was wrong.
    //
    // The reasoning was: sparse_attn is O(topk), so the per-step memcpy that makes (pos+T) rows
    // contiguous for it is pure waste, and removing it should flatten decode against context. It
    // does remove the copy. It does not make anything faster. Same server, same session, same
    // thermal state, temperature 0 (evidence/kv_combined_ab.log):
    //
    //     ms/token      500 ctx    4000 ctx    4963 ctx     slope (ms per token-of-context)
    //     copy path      54.68       68.12       74.54      0.00445
    //     combined       53.17       69.44       77.10      0.00536   <- WORSE
    //
    // Memory at ready is the same to 0.2 GiB, so there is no second reason to keep it either. The
    // likely mechanism for the regression is locality: comp_kv used to be its own compact allocation
    // that stayed hot, and folding it behind seqmax window rows puts it a long way from the rows
    // touched next.
    //
    // What this rules OUT is worth as much as a win: the context-dependent cost is NOT the copy. It
    // is the DSA index path, which scores every compressed row to select top-k -- O(context/ratio)
    // by construction -- with a `<<<1,32>>>` top-k on one warp on top of it. That is where the next
    // measurement goes. Kept behind DSV4_KV_COMBINED=1 so the experiment is repeatable.
    // DECODE_LADDER 1b.2. Must run BEFORE any KV cache is allocated: it is what sets the row
    // stride every allocation, memcpy and kernel below uses. See include/kv_pack.h.
    kv_pack_init_seqmax(seqmax);   // packing is bit-exact; auto-on above 32768 so long context fits
    g_kv_winmax = (getenv("DSV4_KV_COMBINED") && atoi(getenv("DSV4_KV_COMBINED")) == 1) ? seqmax : 0;
    const size_t comp_rows = (size_t)(seqmax / 4 + 2);   // widest compressed cache (ratio 4)
    for (int Lyr = 0; Lyr < N_LAYERS; ++Lyr) {
        const int ratio = compress_ratio(Lyr);
        // One allocation: [seqmax window rows][comp_rows compressed rows]. comp_kv is a VIEW.
        CU(cudaMalloc(&KV[Lyr].win_kv,
                      kv_rows_bytes((size_t)seqmax + ((ratio && g_kv_winmax) ? comp_rows : 0))));
        if (ratio) {
            if (g_kv_winmax) KV[Lyr].comp_kv = kv_row(KV[Lyr].win_kv, seqmax);
            else CU(cudaMalloc(&KV[Lyr].comp_kv, kv_rows_bytes(comp_rows)));
            // xin is the compressor's attention-input history. After the x_cur/x_full split its
            // only reader is compressor_emit_group, which never looks back further than 2*ratio
            // positions, so a ring of 2*ratio rows plus a `ratio`-row mirrored margin holds
            // everything that is ever read. At [seqmax, DIM] fp32 across 41 compressed layers this
            // was 656 KiB per token of CONTEXT -- 6.6x the entire MLA+DSA KV cache, and the largest
            // remaining term that scaled with seqmax. As a ring it is a fixed ~124 MiB in total.
            // R + ratio rows: R holds the batch plus the compressor's lookback, the extra `ratio`
            // is the mirrored margin that keeps a group window contiguous. See block_decode.cu.
            const int ring_alloc = xin_ring_alloc_rows(ratio);
            const size_t xin_rows = ring_alloc ? (size_t)ring_alloc : (size_t)seqmax;
            CU(cudaMalloc(&KV[Lyr].xin, xin_rows * DIM * 4));
            if (ratio == 4) CU(cudaMalloc(&KV[Lyr].idx_ckv, (size_t)(seqmax / ratio + 2) * INDEX_HEAD_DIM * 4));
        }
    }
    CU(cudaMalloc(&d_ids, (size_t)seqmax * 4));
    // h0 is per-BATCH, not per-context: both prefill and extend embed at most one chunk at a time.
    // It is allocated below, with the other batch-width buffers, once MAXB is known. `hbuf`/`hbuf2`
    // are gone entirely -- they were the seqmax-wide activation pair that only the one-shot prefill
    // used, and prefill now runs through the same chunked path everything else does. At 4 x d x fp32
    // per token they were 128 KiB/token of context, for a buffer whose live extent is one chunk.
    CU(cudaMalloc(&collapsed, (size_t)d * 4));
    CU(cudaMalloc(&logits, (size_t)VOCAB * 4));

    head_bf = (const void*)W.get("head.weight").dev;          // BF16, read natively (F26)
    norm_w = L.bf16("norm.weight");
    hc_fn = L.f32("hc_head_fn"); hc_sc = L.f32("hc_head_scale"); hc_bs = L.f32("hc_head_base");
    emb = (const __nv_bfloat16*)W.get("embed.weight").dev;

    P1.assign(N_LAYERS, {}); P2.assign(N_LAYERS, {}); P3.assign(N_LAYERS, {});
    S18.assign(N_LAYERS, {}); S28.assign(N_LAYERS, {}); S38.assign(N_LAYERS, {});
    BW.assign(N_LAYERS, BlockWeights()); CW.assign(N_LAYERS, CompressedBlockWeights());
    say("[engine] building %d layer structs (persistent)...\n", N_LAYERS);
    for (int Lyr = 0; Lyr < N_LAYERS; ++Lyr)
        build_layer(Lyr, slide_cos, slide_sin, cqc, cqs, cc4c, cc4s, cc128c, cc128s);

    // ---- DSpark MTP head. On 0731-REAP mtp.0/1/2 are EMBEDDED in the main checkpoint, so reuse
    // the same store: a second one would duplicate ~6.5 GiB against ~16 GiB of headroom.
    if (!W.has("mtp.0.attn_norm.weight")) { fprintf(stderr, "[engine] FATAL: no embedded mtp.* head\n"); exit(1); }
    std::vector<float> bc, bs2;
    yarn::freqs(bc, bs2, seqmax, ROPE_DIM, 0, ROPE_THETA, YARN_FACTOR, YARN_BETA_FAST, YARN_BETA_SLOW);
    blk_cos = up_f(bc, keep); blk_sin = up_f(bs2, keep);
    main_proj = L.raw("mtp.0.main_proj.weight");
    main_proj_s = L.scale("mtp.0.main_proj.scale");
    main_norm = L.bf16("mtp.0.main_norm.weight");
    NSTAGE = 0; while (W.has("mtp." + std::to_string(NSTAGE) + ".attn_norm.weight")) ++NSTAGE;
    NE = 0;     while (W.has("mtp.0.ffn.experts." + std::to_string(NE) + ".w1.weight")) ++NE;
    say("[engine] DSpark head: NSTAGE=%d experts=%d\n", NSTAGE, NE);

    mb.assign(NSTAGE, BlockWeights()); mkv.assign(NSTAGE, nullptr);
    HP1.assign(NSTAGE, {}); HP2.assign(NSTAGE, {}); HP3.assign(NSTAGE, {});
    HS1.assign(NSTAGE, {}); HS2.assign(NSTAGE, {}); HS3.assign(NSTAGE, {});
    for (int st = 0; st < NSTAGE; ++st) {
        const std::string b = "mtp." + std::to_string(st) + ".", p = b + "attn.";
        MLAWeights& a = mb[st].attn;
        a.wq_a = L.raw(p + "wq_a.weight"); a.wq_a_s = L.scale(p + "wq_a.scale");
        a.wq_b = L.raw(p + "wq_b.weight"); a.wq_b_s = L.scale(p + "wq_b.scale");
        a.wkv = L.raw(p + "wkv.weight"); a.wkv_s = L.scale(p + "wkv.scale");
        a.wo_b = L.raw(p + "wo_b.weight"); a.wo_b_s = L.scale(p + "wo_b.scale");
        a.q_norm = L.bf16(p + "q_norm.weight"); a.kv_norm = L.bf16(p + "kv_norm.weight");
        a.wo_a = L.wo_a(p + "wo_a.weight", p + "wo_a.scale");
        a.attn_sink = L.f32(p + "attn_sink"); a.cosT = blk_cos; a.sinT = blk_sin;
        mb[st].dim = DIM; mb[st].hc = HC_MULT;
        mb[st].attn_norm = L.bf16(b + "attn_norm.weight"); mb[st].ffn_norm = L.bf16(b + "ffn_norm.weight");
        mb[st].hc_attn_fn = L.f32(b + "hc_attn_fn"); mb[st].hc_attn_scale = L.f32(b + "hc_attn_scale"); mb[st].hc_attn_base = L.f32(b + "hc_attn_base");
        mb[st].hc_ffn_fn = L.f32(b + "hc_ffn_fn"); mb[st].hc_ffn_scale = L.f32(b + "hc_ffn_scale"); mb[st].hc_ffn_base = L.f32(b + "hc_ffn_base");
        MoEWeights& m = mb[st].ffn;
        const std::string fp = b + "ffn.";
        m.gate_w = L.bf16(fp + "gate.weight"); m.is_hash = false;
        m.gate_bias = W.has(fp + "gate.bias") ? L.f32(fp + "gate.bias") : nullptr;
        m.tid2eid = nullptr;
        for (int e = 0; e < NE; ++e) {
            const std::string ep = fp + "experts." + std::to_string(e) + ".";
            HP1[st].push_back(L.raw(ep + "w1.weight")); HP2[st].push_back(L.raw(ep + "w2.weight")); HP3[st].push_back(L.raw(ep + "w3.weight"));
            HS1[st].push_back(L.raw(ep + "w1.scale")); HS2[st].push_back(L.raw(ep + "w2.scale")); HS3[st].push_back(L.raw(ep + "w3.scale"));
        }
        m.w1p = HP1[st].data(); m.w2p = HP2[st].data(); m.w3p = HP3[st].data();
        m.e8m0_scales = true; m.w1sp8 = HS1[st].data(); m.w2sp8 = HS2[st].data(); m.w3sp8 = HS3[st].data();
        const std::string sp2 = fp + "shared_experts.";
        m.sw1 = L.raw(sp2 + "w1.weight"); m.sw2 = L.raw(sp2 + "w2.weight"); m.sw3 = L.raw(sp2 + "w3.weight");
        m.sw1s = L.scale(sp2 + "w1.scale"); m.sw2s = L.scale(sp2 + "w2.scale"); m.sw3s = L.scale(sp2 + "w3.scale");
        m.n_routed = NE; m.n_act = N_ACT; m.dim = DIM; m.inter = MOE_INTER; m.vocab = VOCAB;
        m.route_scale = ROUTE_SCALE; m.swiglu_limit = SWIGLU_LIMIT;
        m.use_tc_pp = true; m.batched = true; m.device_route = true;
        CU(cudaMalloc(&mkv[st], kv_rows_bytes(seqmax)));
    }
    const std::string LS = "mtp." + std::to_string(NSTAGE - 1) + ".";
    hh_fn = L.f32(LS + "hc_head_fn"); hh_sc = L.f32(LS + "hc_head_scale"); hh_ba = L.f32(LS + "hc_head_base");
    hnorm = L.bf16(LS + "norm.weight");
    mw1 = (const void*)W.get(LS + "markov_head.markov_w1.weight").dev;   // BF16 native (F26)
    mw2 = (const void*)W.get(LS + "markov_head.markov_w2.weight").dev;

    CU(cudaMalloc(&main_x, (size_t)seqmax * d * 4));
    CU(cudaMalloc(&dbid, (size_t)BLK * 4));
    CU(cudaMalloc(&dfid, 4));
    CU(cudaMalloc(&dout, (size_t)(BLK + 1) * 4));
    CU(cudaMalloc(&dmarg, (size_t)BLK * 4));
    CU(cudaMalloc(&xemb, (size_t)BLK * d * 4));
    CU(cudaMalloc(&xa, (size_t)BLK * hc * d * 4));
    CU(cudaMalloc(&xb, (size_t)BLK * hc * d * 4));
    // Verify AND extend share these. EXT_CHUNK > VBMAX so a prefix-cache extension moves in useful
    // strides; sizing at the max of the two is what keeps a VB=BLK+1 verify from overrunning by one
    // position, which would silently corrupt the last verified token rather than crash (F62).
    EXT_CHUNK = cfg.ext_chunk > 0 ? cfg.ext_chunk : 64;
    const int MB = EXT_CHUNK > VBMAX ? EXT_CHUNK : VBMAX;
    MAXB = MB;
    CU(cudaMalloc(&hv, (size_t)MB * hc * d * 4));
    CU(cudaMalloc(&hv2, (size_t)MB * hc * d * 4));
    CU(cudaMalloc(&collK, (size_t)MB * d * 4));
    CU(cudaMalloc(&logK, (size_t)MB * VOCAB * 4));
    CU(cudaMalloc(&mh_v, (size_t)MB * 3 * d * 4));
    CU(cudaMalloc(&h0, (size_t)MB * d * 4));            // batch-width, see the note above

    // THE ARENA IS SIZED BY THE BATCH, NOT BY THE CONTEXT.
    //
    // It used to be `(512 + 2 * seqmax) MiB`, which contains no model constant and reserved 2 MiB of
    // scratch per token of CONTEXT. What the arena actually has to cover is the widest M ever pushed
    // through a layer -- the GEMM and activation scratch for one batch -- and since prefill now goes
    // through the same chunked path as everything else, that M is MAXB, never seqmax. Scaling it by
    // seqmax was the single largest consumer of memory in this engine: 68 % of everything that grew
    // with context, and it is what held the server at 4096 tokens on a 122.8 GiB box while the MLA +
    // DSA KV cache it was nominally there to support costs 99.4 KiB/token.
    //
    // 2 MiB per batch row is kept as the slope because it is the slope the old line used and it has
    // never overflowed; it is now multiplied by 64 instead of by 8192. Undersizing is SAFE to probe:
    // `dmalloc` aborts with "[dscratch] arena overflow N>M" rather than corrupting, so a bad constant
    // is a loud crash on the first prefill, not a wrong answer later. `arena_hwm()` reports what was
    // actually touched, so this number can be checked against a measurement instead of trusted.
    size_t arena_bytes = (size_t)(512 + (size_t)MB * 2) << 20;
    arena_init(arena_bytes);
    dprof_init();

    { size_t fb, tb; cudaMemGetInfo(&fb, &tb);
      say("[engine] ready. mem %.1f/%.1f GiB  (seqmax %d, batch %d, arena %zu MiB)\n",
          (tb - fb) / 1073741824.0, tb / 1073741824.0, seqmax, MB, arena_bytes >> 20); }
    is_loaded = true;
}

// Prefill over [0..n-1] from an empty cache.
//
// This used to be a one-shot pass at M = n over `hbuf`/`hbuf2`, which is why those buffers were
// [seqmax, hc, d] and why the arena was scaled by seqmax: a 4096-token prompt really did push M=4096
// through every layer at once. `extend` has always been the same forward at an offset -- it is the
// batched-forward-at-a-position the accept path runs every round, and F135's prefix-cache gate
// already asserts that a turn served through it emits the SAME tokens as a full re-prefill. So the
// one-shot path was not buying correctness; it was buying GEMM efficiency at M=n, and charging
// 2.2 MiB/token of context for it.
//
// Prefill is therefore just `extend` from zero. The reset that used to be implicit in starting from
// scratch has to be explicit here: compressed layers carry a row count, and `main_x` is read at
// positions the DSpark head will visit, so both are cleared before the first chunk.
void Engine::Impl::prefill_full(const std::vector<int>& ids) {
    for (int L = 0; L < N_LAYERS; ++L) KV[L].T = 0;
    CU(cudaMemset(main_x, 0, (size_t)cfg.seqmax * d * 4));
    mkv_valid = 0;                             // main_x is gone; every cached main-KV row with it
    extend(ids, 0);
}

// Extend a cache that already covers [0..from-1] with ids[from..]. Uses the M=K verify kernel at an
// offset, which is the same batched-forward-at-position the accept path runs every round -- so a
// prefix-cached turn and a fresh one go through the identical arithmetic, only starting later.
void Engine::Impl::extend(const std::vector<int>& ids, int from) {
    const int n = (int)ids.size();
    if (mkv_valid > from) mkv_valid = from;    // rows [from,n) of main_x are about to be rewritten
    for (int base = from; base < n; base += EXT_CHUNK) {
        const int m = (n - base) < EXT_CHUNK ? (n - base) : EXT_CHUNK;
        int* dvt = d_ids + base;
        CU(cudaMemcpy(dvt, ids.data() + base, (size_t)m * 4, cudaMemcpyHostToDevice));
        k_embed<<<((size_t)m * d + 255) / 256, 256>>>(h0, emb, dvt, m, d);
        k_hc_expand<<<((size_t)m * hc * d + 255) / 256, 256>>>(hv, h0, m, hc, d);
        CU(cudaDeviceSynchronize());
        float *x = hv, *y = hv2;
        for (int L = 0; L < N_LAYERS; ++L) {
            arena_reset();
            if (compress_ratio(L) == 0) block_verify_step(y, x, dvt, BW[L], base, m, HC_SINKHORN_ITERS, EPS, KV[L]);
            else                        cblock_verify_step(y, x, dvt, CW[L], base, m, HC_SINKHORN_ITERS, EPS, KV[L]);
            std::swap(x, y);
            if (L == 40) dspark_tap_pool(mh_v, x, m, hc, d, 0, 3);
            else if (L == 41) dspark_tap_pool(mh_v, x, m, hc, d, 1, 3);
            else if (L == 42) dspark_tap_pool(mh_v, x, m, hc, d, 2, 3);
        }
        dspark_main_x(main_x + (size_t)base * d, mh_v, main_proj, main_proj_s, main_norm, m, d, EPS);
        CU(cudaDeviceSynchronize());
    }
    cpos = n;
    ctx = ids;
    // The arena is now sized from MAXB rather than from seqmax, so what it actually touches is worth
    // reporting rather than assuming. `g_arena_hwm` is the deepest the bump allocator ever got; if
    // this ever approaches the cap, the sizing constant is wrong and the next prefill aborts loudly.
    say("[engine] prefill %d tok, arena high-water %.0f MiB of %.0f MiB\n",
        n, g_arena_hwm / 1048576.0, g_arena_cap / 1048576.0);
}

// Drop the cache back to n committed positions. Sliding-window and xin caches are position-indexed,
// so they need nothing; only the compressed layers carry a row COUNT, and the number of compressed
// rows for positions [0..n-1] is exactly floor(n/ratio) -- the same accounting the accept path does
// when it discards rows written by rejected drafts.
void Engine::Impl::rewind_to(int n) {
    for (int L = 0; L < N_LAYERS; ++L) {
        const int ratio = compress_ratio(L);
        if (ratio) KV[L].T = n / ratio;
    }
    cpos = n;
    ctx.resize(n);
    if (mkv_valid > n) mkv_valid = n;          // positions above n are no longer committed
}

// Resolve the prompt's prefix the way generate() does, then read the target's logits for the next
// position without committing anything. Restores the compressed-layer row counts on the way out, so
// the caller's KV is exactly as it was found.
std::vector<float> Engine::Impl::debug_prefix_logits(const std::vector<int>& ids) {
    const int n = (int)ids.size();
    const int target = n - 1;
    int shared = 0;
    if (use_prefix_cache) {
        const int lim = (int)ctx.size() < target ? (int)ctx.size() : target;
        while (shared < lim && ctx[shared] == ids[shared]) ++shared;
    }
    if (shared < target && shared > 0) {
        rewind_to(shared);
        extend(std::vector<int>(ids.begin(), ids.begin() + target), shared);
    } else if (shared >= target) {
        rewind_to(target);
    } else {
        prefill_full(std::vector<int>(ids.begin(), ids.begin() + target));
    }

    std::vector<int> Tbefore(N_LAYERS);
    for (int L = 0; L < N_LAYERS; ++L) Tbefore[L] = KV[L].T;

    const int VB = 1, cur = ids[n - 1];
    int* dvt = d_ids + cpos;
    CU(cudaMemcpy(dvt, &cur, 4, cudaMemcpyHostToDevice));
    k_embed<<<((size_t)VB * d + 255) / 256, 256>>>(h0, emb, dvt, VB, d);
    k_hc_expand<<<((size_t)VB * hc * d + 255) / 256, 256>>>(hv, h0, VB, hc, d);
    CU(cudaDeviceSynchronize());
    float *vin = hv, *vout = hv2;
    for (int L = 0; L < N_LAYERS; ++L) {
        arena_reset();
        if (compress_ratio(L) == 0) block_verify_step(vout, vin, dvt, BW[L], cpos, VB, HC_SINKHORN_ITERS, EPS, KV[L]);
        else                        cblock_verify_step(vout, vin, dvt, CW[L], cpos, VB, HC_SINKHORN_ITERS, EPS, KV[L]);
        std::swap(vin, vout);
    }
    hc_head(collK, vin, hc_fn, hc_sc, hc_bs, VB, hc, d, HC_EPS);
    rmsnorm(collK, collK, norm_w, VB, d, EPS, true, 0);
    gemm_bf16w(logK, collK, head_bf, VB, VOCAB, d, 0);
    CU(cudaDeviceSynchronize());
    std::vector<float> out(VOCAB);
    CU(cudaMemcpy(out.data(), logK, (size_t)VOCAB * 4, cudaMemcpyDeviceToHost));

    for (int L = 0; L < N_LAYERS; ++L) KV[L].T = Tbefore[L];   // the probe committed nothing
    return out;
}

GenStats Engine::Impl::generate(const std::vector<int>& ids, const GenParams& gp,
                                const std::function<bool(int)>& on_token) {
    GenStats stats;
    stats.prompt_tokens = (int)ids.size();
    const int n = (int)ids.size();
    if (n < 1) return stats;
    if (n + gp.max_tokens + BLK + 2 > cfg.seqmax) {
        fprintf(stderr, "[engine] prompt %d + max_tokens %d exceeds seqmax %d\n", n, gp.max_tokens, cfg.seqmax);
        return stats;
    }

    // ---- prefix cache. The KV covers `ctx`; anything the new prompt shares with it is already
    // computed. One position is deliberately held back: the last prompt token is never prefilled,
    // it is the `cur` the first draft block conditions on.
    const auto t_pre0 = std::chrono::steady_clock::now();
    const int target = n - 1;                     // positions to have resident before decoding
    int shared = 0;
    if (use_prefix_cache) {
        const int lim = (int)ctx.size() < target ? (int)ctx.size() : target;
        while (shared < lim && ctx[shared] == ids[shared]) ++shared;
    }
    if (shared < target && shared > 0) {
        rewind_to(shared);
        extend(std::vector<int>(ids.begin(), ids.begin() + target), shared);
    } else if (shared >= target) {
        rewind_to(target);
    } else {
        prefill_full(std::vector<int>(ids.begin(), ids.begin() + target));
    }
    stats.cached_tokens = shared;
    CU(cudaDeviceSynchronize());
    stats.prefill_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_pre0).count();

    std::mt19937_64 rng(gp.has_seed ? gp.seed : 0x9E3779B97F4A7C15ULL);
    const int eos = 1;                            // <｜end▁of▁sentence｜>
    auto is_stop = [&](int t) {
        if (t == eos) return true;
        for (int s : gp.stop_ids) if (t == s) return true;
        return false;
    };

    int cur = ids[n - 1];                         // token at position cpos, not yet in the cache
    int produced = 0;
    bool stop = false;
    const auto t_dec0 = std::chrono::steady_clock::now();

    std::vector<int> bid(BLK), oo(BLK + 1), draft(BLK), vtok(BLK + 1), tam(BLK + 1), Tbefore(N_LAYERS);
    std::vector<float> hmarg(BLK, 0.f), lg((size_t)(BLK + 1) * VOCAB);

    while (produced < gp.max_tokens && !stop && cpos + BLK + 1 < cfg.seqmax) {
        // ---- DSV4_DPROF: attribute ONE WHOLE decode step -- draft and verify -- in the server
        // path, at real context. dprof_init() has been called at load since the marks were written,
        // but nothing in this file ever called dprof_report(), so every mark the kernels recorded
        // here was dropped on the floor. The only profiles this repo had were from src/decode.cu's
        // K-sweep, whose context is the length of argv[2] -- which is how 0.2 came to label two
        // ctx-9 runs "ctx 480" and "ctx 3000" and retire two ladder items on the difference.
        // Reporting here profiles the exact path, corpus and context tools/decode_fit_probe.py fits.
        //
        // Reset EVERY step, not just reported ones: the pool is fixed at 8192 marks and silently
        // STOPS recording on overflow, so an unreset pool would report the first few steps and then
        // quietly keep printing tables that were missing rows.
        const bool dp_step = g_dprof_on;
        if (dp_step) dprof_reset();

        const int anchor = cpos - 1, ctxlen = cpos;
        // LADDER 1.0. This used to be an O(ctxlen) from-scratch rebuild, NSTAGE times per token --
        // 3.867 +/- 0.001 ms per 1000 context, 47.87 ms at ctx 12,288, the largest single row in the
        // step and 55 % of the whole context term (0.4). It is now incremental: only the rows
        // committed since the last step are computed, which is `acc+1` of them, so the per-step cost
        // is O(tokens committed) instead of O(context) and the total over a generation is O(n)
        // instead of O(n^2). Byte-identical, not approximate -- see kernels/dspark_attn.cu, the
        // in-situ DSV4_MAINKV_GATE=1 memcmp, and tests/gate_mainkv_incr.cu.
        // DSV4_MAINKV_CACHE=0 restores the from-scratch path EXACTLY, which is the A/B arm.
        static const bool mainkv_cache = !(getenv("DSV4_MAINKV_CACHE") && atoi(getenv("DSV4_MAINKV_CACHE")) == 0);
        dprof_begin(DP_D_MAINKV, 0);
        if (mainkv_cache) {
            // One `valid` per stage because each call advances its own copy; the stages are always
            // in lockstep, so the shared high-water mark is written back once.
            for (int st = 0; st < NSTAGE; ++st) {
                int v = mkv_valid;
                dspark_main_kv_upto(mkv[st], main_x, mb[st].attn, ctxlen, &v, EPS);
            }
            mkv_valid = ctxlen;
        } else {
            for (int st = 0; st < NSTAGE; ++st) dspark_main_kv(mkv[st], main_x, mb[st].attn, ctxlen, EPS);
        }
        dprof_end(DP_D_MAINKV, 0);

        // ---- DRAFT: block [cur, noise x (BLK-1)]
        for (int i = 0; i < BLK; ++i) bid[i] = DSPARK_NOISE_TID;
        bid[0] = cur;
        float *cb = nullptr, *nb = nullptr;
        for (int pass = 0; pass < cfg.npass; ++pass) {
            arena_reset();
            CU(cudaMemcpy(dbid, bid.data(), (size_t)BLK * 4, cudaMemcpyHostToDevice));
            k_embed<<<((size_t)BLK * d + 255) / 256, 256>>>(xemb, emb, dbid, BLK, d);
            k_hc_expand<<<((size_t)BLK * hc * d + 255) / 256, 256>>>(xa, xemb, BLK, hc, d);
            CU(cudaDeviceSynchronize());
            cb = xa; nb = xb;
            dprof_begin(DP_D_BLOCK, 0);
            for (int st = 0; st < NSTAGE; ++st) {
                dspark_block_forward(nb, cb, dbid, mkv[st], anchor, mb[st],
                                     blk_cos + (size_t)ctxlen * half, blk_sin + (size_t)ctxlen * half,
                                     BLK, WINDOW, HC_SINKHORN_ITERS, EPS);
                std::swap(cb, nb);
            }
            dprof_end(DP_D_BLOCK, 0);
            CU(cudaMemcpy(dfid, &cur, 4, cudaMemcpyHostToDevice));
            dprof_begin(DP_D_HEAD, 0);
            dspark_forward_head(dout, cb, dfid, hh_fn, hh_sc, hh_ba, hnorm, head_bf, mw1, mw2,
                                1, BLK, hc, d, VOCAB, DSPARK_MARKOV_RANK, EPS, dmarg);
            dprof_end(DP_D_HEAD, 0);
            CU(cudaDeviceSynchronize());
            CU(cudaMemcpy(oo.data(), dout, (size_t)(BLK + 1) * 4, cudaMemcpyDeviceToHost));
            CU(cudaMemcpy(hmarg.data(), dmarg, (size_t)BLK * 4, cudaMemcpyDeviceToHost));
            for (int i = 0; i < BLK; ++i) draft[i] = oo[1 + i];
            for (int i = 1; i < BLK; ++i) bid[i] = draft[i - 1];
        }

        // ---- ADAPTIVE VERIFY WIDTH (F49). Truncating is lossless: verifying fewer proposals cannot
        // change what the target emits, only how many tokens one verify can commit.
        const int VKCAP = BLK + 1;
        int VK = VKCAP;
        if (cfg.adaptK > 0.f) {
            VK = 2;
            while (VK < VKCAP && hmarg[VK - 1] >= cfg.adaptK) ++VK;
        }
        const int VB = VK;
        vtok[0] = cur;
        for (int i = 1; i < VB; ++i) vtok[i] = draft[i - 1];
        for (int L = 0; L < N_LAYERS; ++L) Tbefore[L] = KV[L].T;

        int* dvt = d_ids + cpos;
        CU(cudaMemcpy(dvt, vtok.data(), (size_t)VB * 4, cudaMemcpyHostToDevice));
        k_embed<<<((size_t)VB * d + 255) / 256, 256>>>(h0, emb, dvt, VB, d);
        k_hc_expand<<<((size_t)VB * hc * d + 255) / 256, 256>>>(hv, h0, VB, hc, d);
        CU(cudaDeviceSynchronize());
        float *vin = hv, *vout = hv2;
        for (int L = 0; L < N_LAYERS; ++L) {
            arena_reset();
            if (compress_ratio(L) == 0) block_verify_step(vout, vin, dvt, BW[L], cpos, VB, HC_SINKHORN_ITERS, EPS, KV[L]);
            else                        cblock_verify_step(vout, vin, dvt, CW[L], cpos, VB, HC_SINKHORN_ITERS, EPS, KV[L]);
            std::swap(vin, vout);
            if (L == 40) dspark_tap_pool(mh_v, vin, VB, hc, d, 0, 3);
            else if (L == 41) dspark_tap_pool(mh_v, vin, VB, hc, d, 1, 3);
            else if (L == 42) dspark_tap_pool(mh_v, vin, VB, hc, d, 2, 3);
        }
        // Same two marks src/decode.cu's K-sweep uses, so the row sets are directly comparable.
        dprof_begin(DP_HEAD_HC, 0);
        hc_head(collK, vin, hc_fn, hc_sc, hc_bs, VB, hc, d, HC_EPS);
        rmsnorm(collK, collK, norm_w, VB, d, EPS, true, 0);
        dprof_end(DP_HEAD_HC, 0);
        dprof_begin(DP_LM_HEAD, 0);
        gemm_bf16w(logK, collK, head_bf, VB, VOCAB, d, 0);
        dprof_end(DP_LM_HEAD, 0);
        CU(cudaDeviceSynchronize());
        // The report syncs and prints; it lands AFTER the sync above, so it adds no stall inside the
        // step it describes -- but it is inside the decode timing window, so a DPROF run's tok/s is
        // an instrumented number and is not comparable to PERF.md. The per-mark ms are.
        if (dp_step) {
            static const int every = getenv("DSV4_DPROF_EVERY") ? atoi(getenv("DSV4_DPROF_EVERY")) : 1;
            if (every > 0 && stats.verifies % every == 0) {
                char tg[80];
                snprintf(tg, sizeof tg, "ctx=%d VB=%d verify=%d", cpos, VB, stats.verifies);
                dprof_report(tg);
            }
        }
        CU(cudaMemcpy(lg.data(), logK, (size_t)VB * VOCAB * 4, cudaMemcpyDeviceToHost));
        for (int i = 0; i < VB; ++i) tam[i] = sample_row(&lg[(size_t)i * VOCAB], VOCAB, gp, rng);

        // ---- accept the longest prefix the target agrees with, then its own correction
        int acc = 0;
        while (acc < VB - 1 && draft[acc] == tam[acc]) ++acc;
        const int correction = tam[acc];

        // Commit. `acc` accepted proposals then the correction; the callback may stop us mid-block,
        // but the cache advance below has to happen for the whole committed range regardless.
        std::vector<int> committed;
        committed.reserve(acc + 1);
        for (int i = 0; i < acc; ++i) committed.push_back(draft[i]);
        committed.push_back(correction);

        dspark_main_x(main_x + (size_t)cpos * d, mh_v, main_proj, main_proj_s, main_norm, acc + 1, d, EPS);
        CU(cudaDeviceSynchronize());
        for (int L = 0; L < N_LAYERS; ++L) {
            const int ratio = compress_ratio(L);
            if (!ratio) continue;
            int valid = 0;
            for (int j = cpos; j <= cpos + acc; ++j) if ((j + 1) % ratio == 0) ++valid;
            KV[L].T = Tbefore[L] + valid;          // drop rows written by rejected drafts
        }
        ctx.push_back(cur);                        // `cur` is now resident at position cpos
        for (int i = 0; i < acc; ++i) ctx.push_back(committed[i]);
        cpos += acc + 1;
        cur = correction;
        ++stats.verifies;
        // Joint (realised width, accepted prefix). Two integer increments, no sync, no device
        // memory -- the cost is unmeasurable against a verify that just ran a full forward, which
        // is what makes it safe to leave on in production. See SpecProfile in dsv4_engine.h.
        if (VB >= 0 && VB < SpecProfile::MAXW && acc >= 0 && acc < SpecProfile::MAXW)
            ++stats.spec.accept_joint[VB][acc];

        for (int t : committed) {
            if (produced >= gp.max_tokens) { stop = true; break; }
            ++produced;
            if (is_stop(t)) { stop = true; break; }
            if (!on_token(t)) { stop = true; break; }
        }
    }
    CU(cudaDeviceSynchronize());
    stats.decode_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_dec0).count();
    stats.completion_tokens = produced;
    stats.tok_per_s = produced > 0 ? produced / (stats.decode_ms / 1000.0) : 0.0;
    stats.tok_per_verify = stats.verifies > 0 ? (double)produced / stats.verifies : 0.0;
    return stats;
}

// ---- public shim --------------------------------------------------------------------------------
Engine::Engine(const EngineConfig& cfg) : p_(new Impl()) { p_->cfg = cfg; }
Engine::~Engine() { delete p_; }
void Engine::load() { p_->load(); }
bool Engine::loaded() const { return p_->is_loaded; }
void Engine::reset() { p_->rewind_to(0); }
int  Engine::context_len() const { return p_->cpos; }
const std::vector<int>& Engine::context_ids() const { return p_->ctx; }
int  Engine::eos_id() const { return 1; }
int  Engine::seqmax() const { return p_->cfg.seqmax; }
void Engine::set_prefix_cache(bool on) { p_->use_prefix_cache = on; }
std::vector<float> Engine::debug_prefix_logits(const std::vector<int>& ids) {
    return p_->debug_prefix_logits(ids);
}
void Engine::set_ext_chunk(int n) {
    // Never above what the buffers were allocated for: a wider chunk would overrun hv/collK/logK
    // by exactly the amount it exceeds them, corrupting positions rather than faulting (F62).
    if (n > 0) p_->EXT_CHUNK = n < p_->MAXB ? n : p_->MAXB;
}
GenStats Engine::generate(const std::vector<int>& ids, const GenParams& gp,
                          const std::function<bool(int)>& on_token) {
    return p_->generate(ids, gp, on_token);
}

} // namespace dsv4srv
