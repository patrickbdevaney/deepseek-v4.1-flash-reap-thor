// v41-topk-bench: time ggml ops of the V4.1 indexer / candidate-mask path on one backend, at given shapes.
//
// usage: v41-topk-bench NCOLS NROWS K [reps=5]
//   times top_k(F32 [NCOLS, NROWS], K), argsort of the same tensor, and pool_1d(max, 8) over it,
//   each as its own graph on the first GPU backend; prints ms per graph.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

static double run(ggml_backend_t be, int ncols, int nrows, int k, int which, int reps) {
    ggml_init_params ip = { 16 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ncols, nrows);
    ggml_tensor * y = which == 0 ? ggml_top_k(ctx, x, k)
                    : which == 1 ? ggml_argsort(ctx, x, GGML_SORT_ORDER_DESC)
                                 : ggml_pool_1d(ctx, x, GGML_OP_POOL_MAX, 8, 8, 0);
    ggml_cgraph * g = ggml_new_graph(ctx);
    ggml_build_forward_expand(g, y);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    std::vector<float> h((size_t) ncols * nrows);
    std::mt19937 rng(1);
    std::normal_distribution<float> nd(0, 1);
    for (auto & v : h) v = nd(rng);
    ggml_backend_tensor_set(x, h.data(), 0, h.size() * sizeof(float));
    if (!ggml_backend_supports_op(be, y)) { ggml_backend_buffer_free(buf); ggml_free(ctx); return -1; }
    ggml_backend_graph_compute(be, g);   // warm
    ggml_backend_synchronize(be);
    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; r++) ggml_backend_graph_compute(be, g);
    ggml_backend_synchronize(be);
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps;
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return ms;
}

int main(int argc, char ** argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s NCOLS NROWS K [reps]\n", argv[0]); return 2; }
    const int ncols = atoi(argv[1]), nrows = atoi(argv[2]), k = atoi(argv[3]);
    const int reps = argc > 4 ? atoi(argv[4]) : 5;
    ggml_backend_load_all();
    ggml_backend_t be = ggml_backend_init_best();   // Thor registers as an IGPU
    printf("backend %s\n", ggml_backend_name(be));
    if (!be) { fprintf(stderr, "no GPU backend\n"); return 1; }
    const char * names[] = { "top_k", "argsort", "pool_1d" };
    for (int w = 0; w < 3; w++) {
        if (w == 2 && ncols % 8) continue;
        printf("%-8s ncols=%d nrows=%d k=%d : %.3f ms\n", names[w], ncols, nrows, k, run(be, ncols, nrows, k, w, reps));
    }
    ggml_backend_free(be);
    return 0;
}
