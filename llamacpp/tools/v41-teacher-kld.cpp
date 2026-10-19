// v41-teacher-kld: paired student-vs-teacher statistics on the s03 held-out rows, from banked teacher top-32.
//
// llama-perplexity --kl-divergence needs the teacher's full-vocab logits in its own file format. s03
// banked only top-32 log-probs, the logsumexp and the next-token NLL, which is enough for the measure
// the plan pre-registers: top-32 KL with one tail bucket.
//   KL = sum_{i in top32} p_i (log p_i - log q_i) + p_tail log(p_tail / q_tail)
// Here p_i = exp(teacher lp_i) is already normalised over the full vocabulary, and q is the student's
// full-vocab softmax.
//
// Input dir (written by ops/s09_teacher_eval.py prep):
//   rows.i64     [n_rows, 2]  (start, len) into the token stream
//   tokens.i32   [N]
//   t_lp.f32     [N, 32]      t_id.i32 [N, 32]
// Output dir: kl.f32 [N], top1.u8 [N] (student argmax == teacher argmax), s_nll.f32 [N]
// (NaN at each row's last position). Each row is a fresh sequence from position 0, as s03 ran it.
//
// usage: v41-teacher-kld -m MODEL.gguf -i IN_DIR -o OUT_DIR [-c 8192] [-b 2048] [-ub 512] [-ngl 999] [--rows N]

#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

template <typename T>
static std::vector<T> read_all(const std::string & path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        fprintf(stderr, "cannot open %s\n", path.c_str());
        exit(1);
    }
    const size_t n = (size_t) f.tellg();
    f.seekg(0);
    std::vector<T> v(n / sizeof(T));
    f.read((char *) v.data(), (std::streamsize) (v.size() * sizeof(T)));
    return v;
}

template <typename T>
static void write_all(const std::string & path, const std::vector<T> & v) {
    std::ofstream f(path, std::ios::binary);
    f.write((const char *) v.data(), (std::streamsize) (v.size() * sizeof(T)));
}

int main(int argc, char ** argv) {
    std::string model_path, in_dir, out_dir;
    int n_ctx = 8192, n_batch = 2048, n_ubatch = 512, ngl = 999;
    long max_rows = -1;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", a.c_str()); exit(1); } return std::string(argv[++i]); };
        if      (a == "-m")     model_path = next();
        else if (a == "-i")     in_dir     = next();
        else if (a == "-o")     out_dir    = next();
        else if (a == "-c")     n_ctx      = std::stoi(next());
        else if (a == "-b")     n_batch    = std::stoi(next());
        else if (a == "-ub")    n_ubatch   = std::stoi(next());
        else if (a == "-ngl")   ngl        = std::stoi(next());
        else if (a == "--rows") max_rows   = std::stol(next());
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 1; }
    }
    if (model_path.empty() || in_dir.empty() || out_dir.empty()) {
        fprintf(stderr, "usage: %s -m MODEL.gguf -i IN_DIR -o OUT_DIR [-c N] [-b N] [-ub N] [-ngl N] [--rows N]\n", argv[0]);
        return 1;
    }

    const auto rows   = read_all<int64_t>(in_dir + "/rows.i64");
    const auto tokens = read_all<int32_t>(in_dir + "/tokens.i32");
    const auto t_lp   = read_all<float>  (in_dir + "/t_lp.f32");
    const auto t_id   = read_all<int32_t>(in_dir + "/t_id.i32");
    const size_t N = tokens.size();
    if (t_lp.size() != N * 32 || t_id.size() != N * 32 || rows.size() % 2 != 0) {
        fprintf(stderr, "input sizes disagree: N=%zu lp=%zu id=%zu rows=%zu\n", N, t_lp.size(), t_id.size(), rows.size());
        return 1;
    }
    size_t n_rows = rows.size() / 2;
    if (max_rows >= 0) {
        n_rows = std::min(n_rows, (size_t) max_rows);
    }

    llama_backend_init();
    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = ngl;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) {
        fprintf(stderr, "failed to load %s\n", model_path.c_str());
        return 1;
    }
    auto cparams = llama_context_default_params();
    cparams.n_ctx    = n_ctx;
    cparams.n_batch  = n_batch;
    cparams.n_ubatch = n_ubatch;
    cparams.n_seq_max = 1;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "failed to create context\n");
        return 1;
    }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    std::vector<float>   kl(N, NAN), s_nll(N, NAN);
    std::vector<uint8_t> top1(N, 0);
    std::vector<double>  lq(n_vocab);
    llama_batch batch = llama_batch_init(n_batch, 0, 1);
    double sum_kl = 0;
    size_t n_kl = 0, n_top1 = 0;

    for (size_t r = 0; r < n_rows; ++r) {
        const int64_t start = rows[2*r], len = rows[2*r + 1];
        if (len > n_ctx) {
            fprintf(stderr, "row %zu has %lld tokens > n_ctx %d\n", r, (long long) len, n_ctx);
            return 1;
        }
        llama_memory_clear(llama_get_memory(ctx), true);
        for (int64_t c0 = 0; c0 < len; c0 += n_batch) {
            const int64_t c1 = std::min<int64_t>(len, c0 + n_batch);
            batch.n_tokens = 0;
            for (int64_t p = c0; p < c1; ++p) {
                const int j = batch.n_tokens++;
                batch.token[j] = tokens[start + p];
                batch.pos[j] = (llama_pos) p;
                batch.n_seq_id[j] = 1;
                batch.seq_id[j][0] = 0;
                batch.logits[j] = 1;
            }
            if (llama_decode(ctx, batch) != 0) {
                fprintf(stderr, "decode failed at row %zu pos %lld\n", r, (long long) c0);
                return 1;
            }
            for (int64_t p = c0; p < c1; ++p) {
                const float * lg = llama_get_logits_ith(ctx, (int32_t) (p - c0));
                double mx = -INFINITY;
                int arg = 0;
                for (int v = 0; v < n_vocab; ++v) {
                    if (lg[v] > mx) { mx = lg[v]; arg = v; }
                }
                double z = 0;
                for (int v = 0; v < n_vocab; ++v) {
                    z += std::exp((double) lg[v] - mx);
                }
                const double lse = mx + std::log(z);
                const size_t g = (size_t) (start + p);
                const float * tl = &t_lp[g*32];
                const int32_t * ti = &t_id[g*32];
                double k = 0, p_top = 0, q_top = 0;
                for (int i = 0; i < 32; ++i) {
                    const double pi = std::exp((double) tl[i]);
                    const double lqi = (double) lg[ti[i]] - lse;
                    k += pi * ((double) tl[i] - lqi);
                    p_top += pi;
                    q_top += std::exp(lqi);
                }
                const double p_tail = std::max(0.0, 1.0 - p_top), q_tail = std::max(1e-30, 1.0 - q_top);
                if (p_tail > 0) {
                    k += p_tail * std::log(p_tail / q_tail);
                }
                kl[g] = (float) k;
                top1[g] = arg == ti[0];
                sum_kl += k;
                n_kl++;
                n_top1 += top1[g];
                if (p + 1 < len) {
                    s_nll[g] = (float) (lse - (double) lg[tokens[start + p + 1]]);
                }
            }
        }
        if (r % 16 == 0 || r + 1 == n_rows) {
            fprintf(stderr, "row %zu/%zu  mean KL %.5f  top1 %.4f\n", r + 1, n_rows, sum_kl / std::max<size_t>(1, n_kl),
                    (double) n_top1 / std::max<size_t>(1, n_kl));
        }
    }

    write_all(out_dir + "/kl.f32", kl);
    write_all(out_dir + "/top1.u8", top1);
    write_all(out_dir + "/s_nll.f32", s_nll);
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
