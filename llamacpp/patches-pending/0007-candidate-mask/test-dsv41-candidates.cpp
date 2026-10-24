// CPU-only test of the DeepSeek-V4.1 candidate-block mask (patch 0007, src/models/deepseek41-candidates.h).
//
// Builds, with tiny synthetic tensors, the same ggml ops deepseek41.cpp builds for
//   candidate source (layer 20):  idx score -> + causal mask -> dsv41_build_candidate_blocks
//   later index source (24..36):  idx score -> + causal mask -> dsv41_apply_candidate_blocks -> top_k
//   its reader attention:         top-k mask (as build_top_k_mask) -> softmax with sinks -> out
// on the CPU backend only, and dumps inputs and outputs for check_candidates.py (the NumPy
// transcription of model.py). For every case it also builds the graph WITHOUT the mask (the tree
// before 0007) and, below the threshold, a forced-mask graph, and reports their differences.
//
// No GPU, no model, no GGUF. Build: see build_test.sh next to this file.

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"

#include "deepseek41-candidates.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>
#include <algorithm>

struct cfg {
    std::string name;
    int n_stream;      // streams (sequences), each with its own positions
    int n_q;           // queries per stream in this ubatch
    std::vector<int> pos0;   // first query position per stream (ratio 1: n_visible = pos + 1)
    int pad;           // n_comp padded to a multiple of this (the cache pads to 256)
    int min_comp;      // n_comp at least this (a long-lived cache)
};

// model dims (tiny)
static const int IDX_H   = 8;    // indexer heads
static const int IDX_D   = 32;   // indexer head dim
static const int ATT_H   = 4;    // attention heads
static const int ATT_D   = 64;   // attention head dim
static const int BS      = 8;    // candidate_block_size
static const int TOPK_B  = 4;    // candidate_topk_blocks  -> threshold 32 rows
static const int IDX_TOPK = 6;   // index_topk of the reader

static void write_bin(const std::string & path, const void * data, size_t n) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", path.c_str()); exit(1); }
    fwrite(data, 1, n, f);
    fclose(f);
}

static std::vector<float> get_f32(ggml_tensor * t) {
    GGML_ASSERT(t->type == GGML_TYPE_F32);
    std::vector<float> v(ggml_nelements(t));
    // tensors may be views: copy element-wise through strides
    std::vector<char> raw(ggml_nbytes(t));
    for (int64_t i3 = 0; i3 < t->ne[3]; ++i3)
    for (int64_t i2 = 0; i2 < t->ne[2]; ++i2)
    for (int64_t i1 = 0; i1 < t->ne[1]; ++i1)
    for (int64_t i0 = 0; i0 < t->ne[0]; ++i0) {
        const char * p = (const char *) t->data + i0*t->nb[0] + i1*t->nb[1] + i2*t->nb[2] + i3*t->nb[3];
        v[((i3*t->ne[2] + i2)*t->ne[1] + i1)*t->ne[0] + i0] = *(const float *) p;
    }
    return v;
}

static std::vector<int32_t> get_i32(ggml_tensor * t) {
    GGML_ASSERT(t->type == GGML_TYPE_I32);
    std::vector<int32_t> v(ggml_nelements(t));
    for (int64_t i3 = 0; i3 < t->ne[3]; ++i3)
    for (int64_t i2 = 0; i2 < t->ne[2]; ++i2)
    for (int64_t i1 = 0; i1 < t->ne[1]; ++i1)
    for (int64_t i0 = 0; i0 < t->ne[0]; ++i0) {
        const char * p = (const char *) t->data + i0*t->nb[0] + i1*t->nb[1] + i2*t->nb[2] + i3*t->nb[3];
        v[((i3*t->ne[2] + i2)*t->ne[1] + i1)*t->ne[0] + i0] = *(const int32_t *) p;
    }
    return v;
}

struct inputs {
    int n_comp, n_q, n_stream;
    std::vector<float> idx_k;   // [S][n_comp][IDX_D]
    std::vector<float> q1, q2;  // [S][n_q][IDX_H][IDX_D]   source / reader indexer queries
    std::vector<float> w1, w2;  // [S][n_q][IDX_H]          indexer head weights (already scaled)
    std::vector<float> kv;      // [S][n_comp][ATT_D]       compressed KV (K = V, MLA style)
    std::vector<float> qa;      // [S][n_q][ATT_H][ATT_D]   reader attention query
    std::vector<float> sinks;   // [ATT_H]
    std::vector<int32_t> n_vis; // [S][n_q]
    std::vector<float> mask;    // [S][n_q][n_comp]  0 / -inf
};

enum mode { MODE_OLD = 0, MODE_NEW = 1, MODE_FORCE = 2 };

struct outputs {
    bool has_cand = false;
    std::vector<float>   score1;   // source score + mask           [S][n_q][n_comp]
    std::vector<float>   cand;     // block keep mask (0/-inf)      [S][n_q][n_blocks]
    std::vector<float>   score2;   // reader score after candidates [S][n_q][n_comp]
    std::vector<int32_t> top;      // reader top-k                  [S][n_q][k]
    std::vector<float>   out;      // attention out                 [S][n_q][H][ATT_D]
    int k = 0;
};

static outputs run_graph(const inputs & in, mode md, int n_threads) {
    const int S = in.n_stream, Q = in.n_q, N = in.n_comp;

    ggml_init_params ip = { (size_t) 256*1024*1024, nullptr, false };
    ggml_context * ctx = ggml_init(ip);

    auto new_f32 = [&](int64_t a, int64_t b, int64_t c, int64_t d, const std::vector<float> & src) {
        ggml_tensor * t = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, a, b, c, d);
        GGML_ASSERT((size_t) ggml_nelements(t) == src.size());
        memcpy(t->data, src.data(), src.size()*sizeof(float));
        return t;
    };

    ggml_tensor * idx_k = new_f32(IDX_D, N, 1, S, in.idx_k);
    ggml_tensor * mask  = new_f32(N, Q, 1, S, in.mask);

    // indexer score as build_indexer_top_k: sum_h relu(q_h . k_t) * w_h, then + causal mask
    auto idx_score = [&](const std::vector<float> & qv, const std::vector<float> & wv) {
        ggml_tensor * q = new_f32(IDX_D, IDX_H*Q, 1, S, qv);                 // [D, H*Q, 1, S]
        ggml_tensor * w = new_f32(IDX_H, Q, 1, S, wv);                       // [H, Q, 1, S]
        ggml_tensor * s = ggml_mul_mat(ctx, idx_k, q);                       // [N, H*Q, 1, S]
        s = ggml_reshape_4d(ctx, s, N, IDX_H, Q, S);
        s = ggml_cont(ctx, ggml_permute(ctx, s, 1, 0, 2, 3));                // [H, N, Q, S]
        s = ggml_relu(ctx, s);
        s = ggml_mul(ctx, s, ggml_reshape_4d(ctx, w, IDX_H, 1, Q, S));
        s = ggml_sum_rows(ctx, s);                                           // [1, N, Q, S]
        s = ggml_reshape_4d(ctx, s, N, Q, 1, S);
        return ggml_add(ctx, s, mask);
    };

    ggml_tensor * score1 = idx_score(in.q1, in.w1);
    ggml_tensor * cand = nullptr;
    if (md != MODE_OLD) {
        cand = dsv41_build_candidate_blocks(ctx, score1, mask, TOPK_B, BS, md == MODE_FORCE);
    }

    ggml_tensor * score2 = idx_score(in.q2, in.w2);
    if (cand) {
        score2 = dsv41_apply_candidate_blocks(ctx, score2, cand, BS);
    }
    const int k = std::min(IDX_TOPK, N);
    ggml_tensor * top = ggml_cont(ctx, ggml_top_k(ctx, score2, k));          // [k, Q, 1, S]

    // build_top_k_mask (deepseek4.cpp), verbatim apart from the context
    ggml_tensor * all = ggml_fill(ctx, mask, -INFINITY);
    all = ggml_view_4d(ctx, all, 1, all->ne[0], all->ne[1], all->ne[3], all->nb[0], all->nb[1], all->nb[2], 0);
    ggml_tensor * top3 = ggml_view_4d(ctx, top, top->ne[0], top->ne[1], top->ne[3], 1, top->nb[1], top->nb[2], top->ne[3]*top->nb[3], 0);
    ggml_tensor * zeros = ggml_fill(ctx, ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, top3->ne[0], top3->ne[1], top3->ne[2]), 0.0f);
    ggml_tensor * tmask = ggml_set_rows(ctx, all, zeros, top3);
    tmask = ggml_view_4d(ctx, tmask, tmask->ne[1], tmask->ne[2], 1, tmask->ne[3], tmask->nb[2], tmask->nb[3], tmask->nb[3], 0);
    tmask = ggml_add(ctx, tmask, mask);                                      // [N, Q, 1, S]

    // reader attention over the compressed rows only (the window part is untouched by 0007)
    ggml_tensor * kv = new_f32(ATT_D, N, 1, S, in.kv);
    ggml_tensor * qa = new_f32(ATT_D, ATT_H, Q, S, in.qa);                  // [D, H, Q, S]
    qa = ggml_permute(ctx, qa, 0, 2, 1, 3);                                  // [D, Q, H, S]
    ggml_tensor * kq = ggml_mul_mat(ctx, kv, qa);                            // [N, Q, H, S]
    ggml_tensor * sinks = new_f32(ATT_H, 1, 1, 1, in.sinks);
    kq = ggml_soft_max_ext(ctx, kq, tmask, 1.0f/sqrtf((float) ATT_D), 0.0f);
    ggml_soft_max_add_sinks(kq, sinks);
    ggml_tensor * vT = ggml_cont(ctx, ggml_transpose(ctx, ggml_reshape_3d(ctx, kv, ATT_D, N, S)));  // [N, D, S]
    vT = ggml_reshape_4d(ctx, vT, N, ATT_D, 1, S);
    ggml_tensor * out = ggml_mul_mat(ctx, vT, kq);                          // [D, Q, H, S]
    out = ggml_cont(ctx, ggml_permute(ctx, out, 0, 2, 1, 3));               // [D, H, Q, S]

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 4096, false);
    ggml_build_forward_expand(gf, score1);
    if (cand) ggml_build_forward_expand(gf, cand);
    ggml_build_forward_expand(gf, score2);
    ggml_build_forward_expand(gf, top);
    ggml_build_forward_expand(gf, out);

    ggml_backend_t be = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(be, n_threads);
    if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "graph compute failed\n");
        exit(1);
    }
    ggml_backend_free(be);

    outputs o;
    o.score1 = get_f32(score1);
    o.has_cand = cand != nullptr;
    if (cand) o.cand = get_f32(cand);
    o.score2 = get_f32(score2);
    o.top = get_i32(top);
    o.out = get_f32(out);
    o.k = k;
    ggml_free(ctx);
    return o;
}

static inputs make_inputs(const cfg & c, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    inputs in;
    in.n_stream = c.n_stream;
    in.n_q = c.n_q;
    int max_vis = 0;
    for (int s = 0; s < c.n_stream; ++s) max_vis = std::max(max_vis, c.pos0[s] + c.n_q);
    int n = std::max(max_vis, c.min_comp);
    in.n_comp = ((n + c.pad - 1)/c.pad)*c.pad;
    const int S = c.n_stream, Q = c.n_q, N = in.n_comp;

    auto fill = [&](std::vector<float> & v, size_t n, float sc) { v.resize(n); for (auto & x : v) x = nd(rng)*sc; };
    fill(in.idx_k, (size_t) S*N*IDX_D, 1.0f);
    fill(in.q1, (size_t) S*Q*IDX_H*IDX_D, 0.3f);
    fill(in.q2, (size_t) S*Q*IDX_H*IDX_D, 0.3f);
    fill(in.w1, (size_t) S*Q*IDX_H, 1.0f/sqrtf((float) IDX_D*IDX_H));
    fill(in.w2, (size_t) S*Q*IDX_H, 1.0f/sqrtf((float) IDX_D*IDX_H));
    fill(in.kv, (size_t) S*N*ATT_D, 1.0f);
    fill(in.qa, (size_t) S*Q*ATT_H*ATT_D, 1.0f);
    fill(in.sinks, ATT_H, 1.0f);
    // rows beyond what any query sees hold stale data in a real cache; keep them non-zero
    in.n_vis.resize((size_t) S*Q);
    in.mask.resize((size_t) S*Q*N);
    for (int s = 0; s < S; ++s) {
        for (int q = 0; q < Q; ++q) {
            const int nv = c.pos0[s] + q + 1;   // ratio 1: (pos + 1) / 1
            in.n_vis[s*Q + q] = nv;
            for (int j = 0; j < N; ++j) in.mask[((size_t) s*Q + q)*N + j] = j < nv ? 0.0f : -INFINITY;
        }
    }
    return in;
}

static double max_abs_diff(const std::vector<float> & a, const std::vector<float> & b) {
    GGML_ASSERT(a.size() == b.size());
    double m = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::isinf(a[i]) || std::isinf(b[i])) {
            if (a[i] != b[i]) return INFINITY;
            continue;
        }
        m = std::max(m, (double) fabsf(a[i] - b[i]));
    }
    return m;
}

static bool bit_equal(const std::vector<float> & a, const std::vector<float> & b) {
    return a.size() == b.size() && memcmp(a.data(), b.data(), a.size()*sizeof(float)) == 0;
}

int main(int argc, char ** argv) {
    const std::string dir = argc > 1 ? argv[1] : "dump";
    const int n_threads = 4;
    ggml_cpu_init();

    // ratio 1 so n_visible = pos + 1; threshold = TOPK_B * BS = 32 rows
    std::vector<cfg> cases = {
        // prefill from position 0 (model.py start_pos == 0)
        { "prefill_n20",        1,  20, {0},       32,   0 },  // below threshold, n_blocks <= topk: no mask built
        { "prefill_n32",        1,  32, {0},       32,   0 },  // exactly the threshold
        { "prefill_n33",        1,  33, {0},       32,   0 },  // one row over
        { "prefill_n100",       1, 100, {0},       32,   0 },
        { "prefill_n300",       1, 300, {0},       32,   0 },
        { "prefill_n20_pad256", 1,  20, {0},      256,   0 },  // below threshold but cache padded: mask built, must be a no-op
        // decode, one query (model.py start_pos > 0)
        { "decode_p19",         1,   1, {19},      32,   0 },  // 20 visible
        { "decode_p31",         1,   1, {31},      32,   0 },  // 32 visible (threshold)
        { "decode_p32",         1,   1, {32},      32,   0 },  // 33 visible, last block holds 1 row
        { "decode_p99",         1,   1, {99},      32,   0 },
        { "decode_p299",        1,   1, {299},    256,   0 },
        { "decode_p300_big",    1,   1, {300},    256, 1024 },  // cache much larger than the sequence
        // a mid-sequence ubatch (llama.cpp chunked prefill; per-query equal to model.py decode)
        { "chunk_p280_q16",     1,  16, {280},    256,   0 },
        // two streams at different lengths in one ubatch
        { "stream2_decode",     2,   1, {17, 200},256,   0 },
        { "stream2_chunk",      2,   8, {10, 150},256,   0 },
    };

    int n_fail = 0;
    FILE * idx = fopen((dir + "/cases.txt").c_str(), "w");
    if (!idx) { fprintf(stderr, "mkdir %s first\n", dir.c_str()); return 1; }

    for (size_t ci = 0; ci < cases.size(); ++ci) {
        const cfg & c = cases[ci];
        inputs in = make_inputs(c, 1234u + (uint32_t) ci);

        outputs o_old = run_graph(in, MODE_OLD,   n_threads);
        outputs o_new = run_graph(in, MODE_NEW,   n_threads);
        outputs o_frc = run_graph(in, MODE_FORCE, n_threads);

        const int n_blocks = in.n_comp / BS;
        int max_vis = 0;
        for (int v : in.n_vis) max_vis = std::max(max_vis, v);
        const bool below = max_vis <= TOPK_B*BS;

        const bool old_new_bits = bit_equal(o_old.out, o_new.out);
        const double old_new_diff = max_abs_diff(o_old.out, o_new.out);
        const bool old_frc_bits = bit_equal(o_old.out, o_frc.out);
        const double old_frc_diff = max_abs_diff(o_old.out, o_frc.out);

        printf("%-20s S=%d Q=%3d n_comp=%4d n_blocks=%3d max_vis=%3d  mask_built=%d  "
               "old-vs-new: bits_equal=%d max|d|=%.3g   old-vs-forced: bits_equal=%d max|d|=%.3g\n",
               c.name.c_str(), c.n_stream, c.n_q, in.n_comp, n_blocks, max_vis, (int) o_new.has_cand,
               (int) old_new_bits, old_new_diff, (int) old_frc_bits, old_frc_diff);

        if (below && !old_new_bits) { printf("  FAIL: below threshold the new graph must equal the old one\n"); ++n_fail; }
        if (below && !old_frc_bits) { printf("  FAIL: below threshold the forced mask must be a no-op\n"); ++n_fail; }
        if (!below && !o_new.has_cand) { printf("  FAIL: above threshold the mask must be built\n"); ++n_fail; }

        // dump the new graph's inputs and outputs (the forced one too, for the block ids below threshold)
        const outputs & o = o_new.has_cand ? o_new : o_frc;
        const std::string p = dir + "/" + c.name;
        write_bin(p + ".idx_k.f32", in.idx_k.data(), in.idx_k.size()*4);
        write_bin(p + ".q1.f32", in.q1.data(), in.q1.size()*4);
        write_bin(p + ".q2.f32", in.q2.data(), in.q2.size()*4);
        write_bin(p + ".w1.f32", in.w1.data(), in.w1.size()*4);
        write_bin(p + ".w2.f32", in.w2.data(), in.w2.size()*4);
        write_bin(p + ".kv.f32", in.kv.data(), in.kv.size()*4);
        write_bin(p + ".qa.f32", in.qa.data(), in.qa.size()*4);
        write_bin(p + ".sinks.f32", in.sinks.data(), in.sinks.size()*4);
        write_bin(p + ".n_vis.i32", in.n_vis.data(), in.n_vis.size()*4);
        write_bin(p + ".score1.f32", o.score1.data(), o.score1.size()*4);
        write_bin(p + ".cand.f32", o.cand.data(), o.cand.size()*4);
        write_bin(p + ".top.i32", o.top.data(), o.top.size()*4);
        write_bin(p + ".out.f32", o_new.out.data(), o_new.out.size()*4);
        write_bin(p + ".out_old.f32", o_old.out.data(), o_old.out.size()*4);
        fprintf(idx, "%s %d %d %d %d %d %d %d %d %d %d %d\n", c.name.c_str(), c.n_stream, c.n_q, in.n_comp,
                IDX_H, IDX_D, ATT_H, ATT_D, BS, TOPK_B, IDX_TOPK, o.k);
    }
    fclose(idx);

    printf("%s: %d structural failure(s)\n", n_fail ? "FAIL" : "OK", n_fail);
    return n_fail ? 1 : 0;
}
