// gate_v41_golden.cu — the CUDA forward against the reference model.py dump (tests/golden/ref_dump.py),
// layer by layer, one layer's weights resident at a time (same memory shape as the reference run).
//
//   build/gate_v41_golden --ckpt <weights: source checkpoint or thor build> --ref <ref_dump --out>
//                         [--keepset s04/keepset.json] [--chunk 128] [--sw-mxfp4] [--engram <v41rows dir>]
//
// Two streams per prompt, both through the same loaded layer:
//   ISOLATED  layer L is fed the REFERENCE's layer L-1 output: the error of this layer alone
//   CHAINED   the CUDA stream end to end: the error the server actually accumulates
// Engram rows come from the reference dump itself (bit-identical to the .v41rows files); with
// --engram the real row store is used instead, which also gates the hashing + I/O path in situ.
//
// PASS needs, on every prompt:  isolated per-layer min row cosine >= 0.9995 at every layer,
// chained final-layer mean cosine >= 0.995, and teacher-forced next-token argmax agreement >= 0.97
// over the saved rows. These thresholds are judgment calls, stated here so a reader can disagree:
// summation order (fp32 GEMV vs cuBLAS/TileLang tiles) moves bf16 roundings, and 40 layers amplify
// that; a real bug (a wrong scale, a swapped half, a misplaced RoPE) moves cosine to << 0.99.
#include "v41_model.h"
#include "npy.h"
#include "third_party/json.hpp"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e_)); exit(2); } } while (0)
using namespace v41;

static float bf(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }

struct Prompt {
    std::string name; int T = 0, from = 0;
    std::vector<int> ids;
    std::vector<uint8_t> eg[2];                    // [T, 24, 264]
    float *Hc = nullptr, *Pc = nullptr;            // chained stream [T,4,5120], [T,4] (device)
    float *Hi = nullptr, *Pi = nullptr;            // isolated
    v41m::State sc, si;
    std::vector<double> min_iso, mean_chain;
};

int main(int argc, char** argv) {
    std::string ckpt, ref, keepset, egdir;
    int chunk = 128; bool hw = true;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "--ckpt") ckpt = next(); else if (a == "--ref") ref = next(); else if (a == "--keepset") keepset = next();
        else if (a == "--chunk") chunk = atoi(next().c_str()); else if (a == "--sw-mxfp4") hw = false;
        else if (a == "--engram") egdir = next();
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
    }
    std::ifstream mf(ref + "/meta.json"); std::stringstream ms; ms << mf.rdbuf();
    const auto meta = nlohmann::json::parse(ms.str());
    const int lo = meta["layers"][0], hi = meta["layers"][1];
    std::string ks_sha;
    if (!keepset.empty()) { std::ifstream kf(keepset); std::stringstream k; k << kf.rdbuf(); ks_sha = nlohmann::json::parse(k.str()).value("sha256", ""); }
    if (!meta["keepset_sha256"].is_null() && meta["keepset_sha256"].get<std::string>() != ks_sha) {
        fprintf(stderr, "reference was dumped for keepset %s, gate configured for '%s'\n",
                meta["keepset_sha256"].get<std::string>().c_str(), ks_sha.c_str());
        return 2;
    }
    cudaStream_t st; CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));

    // globals: embed / head / norm only
    v41m::Model G;
    { v41m::LoadOpts o; o.layer_lo = 0; o.layer_hi = -1; o.embed_head = true; o.keepset_sha256 = ks_sha; v41m::load_model(G, ckpt, o); }

    std::vector<Prompt> P;
    for (auto it = meta["prompts"].begin(); it != meta["prompts"].end(); ++it) {
        Prompt p; p.name = it.key(); p.T = it.value()["T"]; p.from = it.value()["save_from"];
        const Npy t = npy_load(ref + "/" + p.name + "/tokens.npy");
        p.ids.assign(t.as<int32_t>(), t.as<int32_t>() + p.T);
        for (int l = 0; l < 2; ++l) {
            const Npy e = npy_load(ref + "/" + p.name + "/engram_L" + std::to_string(ENGRAM_LAYERS[l]) + ".npy");
            p.eg[l] = e.data;
        }
        CK(cudaMalloc(&p.Hc, (size_t)p.T * HC_DIM * 4)); CK(cudaMalloc(&p.Pc, (size_t)p.T * 4 * 4));
        CK(cudaMalloc(&p.Hi, (size_t)p.T * HC_DIM * 4)); CK(cudaMalloc(&p.Pi, (size_t)p.T * 4 * 4));
        const int seqmax = (p.T + 255) / 256 * 256;
        v41m::alloc_state(p.sc, seqmax, chunk, (size_t)1 << 30, p.T);
        v41m::alloc_state(p.si, seqmax, chunk, (size_t)1 << 30, p.T);
        // embed every row into both streams
        for (int c0 = 0; c0 < p.T; c0 += chunk) {
            const int M = std::min(chunk, p.T - c0);
            CK(cudaMemcpyAsync(p.sc.d_ids, p.ids.data() + c0, M * 4, cudaMemcpyHostToDevice, st));
            CK(cudaMemsetAsync(p.sc.d_img, 0, M, st));
            v41m::embed_rows(G, p.sc, M, st);
            CK(cudaMemcpyAsync(p.Hc + (size_t)c0 * HC_DIM, p.sc.h, (size_t)M * HC_DIM * 4, cudaMemcpyDeviceToDevice, st));
            CK(cudaMemcpyAsync(p.Pc + (size_t)c0 * 4, p.sc.pre, (size_t)M * 16, cudaMemcpyDeviceToDevice, st));
        }
        CK(cudaMemcpyAsync(p.Hi, p.Hc, (size_t)p.T * HC_DIM * 4, cudaMemcpyDeviceToDevice, st));
        CK(cudaMemcpyAsync(p.Pi, p.Pc, (size_t)p.T * 16, cudaMemcpyDeviceToDevice, st));
        P.push_back(std::move(p));
    }
    std::unique_ptr<EngramStore> store;
    std::unique_ptr<EngramHasher> hasher;
    if (!egdir.empty()) {
        store.reset(new EngramStore);
        EngramStoreConfig c; c.dir = egdir; c.max_tokens = chunk;
        c.tokmap = egdir + "/engram_tokmap.bin";
        if (FILE* f = fopen(c.tokmap.c_str(), "rb")) fclose(f); else c.tokmap = "tests/data/engram_tokmap.bin";
        std::string err;
        if (!store->open(c, &err)) { fprintf(stderr, "engram store: %s\n", err.c_str()); return 2; }
        hasher.reset(new EngramHasher(store->tables()));
    }
    // host-built arena when the dumped rows are used: slot s = row s, offset 0
    uint8_t* arena[2]; for (auto& a : arena) CK(cudaMalloc(&a, (size_t)chunk * ENGRAM_COLS * 1024 + (size_t)chunk * ENGRAM_COLS * 2));
    std::vector<uint8_t> h_arena((size_t)chunk * ENGRAM_COLS * 1024 + (size_t)chunk * ENGRAM_COLS * 2, 0);

    int fails = 0;
    printf("Gate V41 GOLDEN: %s vs %s, layers %d..%d, chunk %d\n", ckpt.c_str(), ref.c_str(), lo, hi, chunk);
    for (int L = lo; L <= hi; ++L) {
        v41m::Model m;
        { v41m::LoadOpts o; o.layer_lo = L; o.layer_hi = L; o.embed_head = false; o.mxfp4_hw = hw; o.keepset_sha256 = ks_sha; v41m::load_model(m, ckpt, o); }
        for (auto& p : P) {
            if (L > lo) {   // isolated stream := the reference's layer L-1 output (saved rows only are exact;
                            // rows below save_from keep the CUDA stream, so isolation is exact only when
                            // the whole prompt is saved -- meta says which)
                const Npy rh = npy_load(ref + "/" + p.name + "/L" + (L - 1 < 10 ? "0" : "") + std::to_string(L - 1) + "_h.npy");
                const Npy rp = npy_load(ref + "/" + p.name + "/L" + (L - 1 < 10 ? "0" : "") + std::to_string(L - 1) + "_pre.npy");
                std::vector<float> hf(rh.count());
                for (size_t i = 0; i < hf.size(); ++i) hf[i] = bf(rh.as<uint16_t>()[i]);
                CK(cudaMemcpy(p.Hi + (size_t)p.from * HC_DIM, hf.data(), hf.size() * 4, cudaMemcpyHostToDevice));
                CK(cudaMemcpy(p.Pi + (size_t)p.from * 4, rp.as<float>(), rp.data.size(), cudaMemcpyHostToDevice));
            }
            for (int pass = 0; pass < 2; ++pass) {
                v41m::State& s = pass ? p.si : p.sc;
                float* H = pass ? p.Hi : p.Hc; float* PM = pass ? p.Pi : p.Pc;
                if (L == lo) v41m::reset_state(s, st);
                for (int c0 = 0; c0 < p.T; c0 += chunk) {
                    const int M = std::min(chunk, p.T - c0);
                    std::vector<int> pos(M); for (int i = 0; i < M; ++i) pos[i] = c0 + i;
                    CK(cudaMemcpyAsync(s.d_pos, pos.data(), M * 4, cudaMemcpyHostToDevice, st));
                    CK(cudaMemcpyAsync(s.d_ids, p.ids.data() + c0, M * 4, cudaMemcpyHostToDevice, st));
                    CK(cudaMemsetAsync(s.d_img, 0, M, st)); CK(cudaMemsetAsync(s.d_noimg, 1, M, st));
                    CK(cudaMemcpyAsync(s.h, H + (size_t)c0 * HC_DIM, (size_t)M * HC_DIM * 4, cudaMemcpyDeviceToDevice, st));
                    CK(cudaMemcpyAsync(s.pre, PM + (size_t)c0 * 4, (size_t)M * 16, cudaMemcpyDeviceToDevice, st));
                    v41m::Engram eg;
                    const int slot = engram_slot(L);
                    if (slot >= 0) {
                        if (store) {
                            std::vector<int64_t> ids((size_t)M * 2 * ENGRAM_COLS);
                            hasher->truncate(c0);
                            hasher->hash(p.ids.data() + c0, nullptr, M, c0, ids.data());
                            CK(cudaStreamSynchronize(st));
                            std::string err;
                            if (!store->fetch(ids.data(), M, &err)) { fprintf(stderr, "engram fetch: %s\n", err.c_str()); return 2; }
                            eg.store = store.get(); eg.graph_wait = true;
                        } else {
                            CK(cudaStreamSynchronize(st));
                            for (int r = 0; r < M * ENGRAM_COLS; ++r)
                                memcpy(h_arena.data() + (size_t)r * 1024, p.eg[slot].data() + ((size_t)c0 * ENGRAM_COLS + r) * 264, 264);
                            memset(h_arena.data() + (size_t)chunk * ENGRAM_COLS * 1024, 0, (size_t)chunk * ENGRAM_COLS * 2);
                            CK(cudaMemcpyAsync(arena[slot], h_arena.data(), h_arena.size(), cudaMemcpyHostToDevice, st));
                            eg.rows_dev[slot] = arena[slot]; eg.fmt = 0;
                        }
                    }
                    s.shared_base = c0;
                    v41m::forward_layers(m, s, L, L, M, 0, c0, eg, st);
                    CK(cudaMemcpyAsync(H + (size_t)c0 * HC_DIM, s.h, (size_t)M * HC_DIM * 4, cudaMemcpyDeviceToDevice, st));
                    CK(cudaMemcpyAsync(PM + (size_t)c0 * 4, s.pre, (size_t)M * 16, cudaMemcpyDeviceToDevice, st));
                }
            }
            CK(cudaStreamSynchronize(st));
            // compare saved rows of both streams with the reference output of layer L
            const std::string ln = std::string("L") + (L < 10 ? "0" : "") + std::to_string(L);
            const Npy rh = npy_load(ref + "/" + p.name + "/" + ln + "_h.npy");
            const size_t S = (size_t)(p.T - p.from);
            double res[2][2];                                   // [stream][min row cos, mean cos]
            for (int pass = 0; pass < 2; ++pass) {
                std::vector<float> g(S * HC_DIM);
                CK(cudaMemcpy(g.data(), (pass ? p.Hi : p.Hc) + (size_t)p.from * HC_DIM, g.size() * 4, cudaMemcpyDeviceToHost));
                double mn = 1, sum = 0;
                for (size_t r = 0; r < S; ++r) {
                    double d = 0, a2 = 0, b2 = 0;
                    for (int k = 0; k < HC_DIM; ++k) {
                        const double a = g[r * HC_DIM + k], b = bf(rh.as<uint16_t>()[r * HC_DIM + k]);
                        d += a * b; a2 += a * a; b2 += b * b;
                    }
                    const double c = d / std::sqrt(a2 * b2 + 1e-300);
                    mn = std::min(mn, c); sum += c;
                }
                res[pass][0] = mn; res[pass][1] = sum / S;
            }
            p.min_iso.push_back(res[1][0]); p.mean_chain.push_back(res[0][1]);
            const bool bad = res[1][0] < 0.9995;
            fails += bad;
            printf("  L%02d %-8s isolated min %.6f mean %.6f | chained min %.6f mean %.6f%s\n", L, p.name.c_str(),
                   res[1][0], res[1][1], res[0][0], res[0][1], bad ? "   <-- FAIL" : "");
        }
    }
    if (hi == N_LAYERS - 1) {
        for (auto& p : P) {
            const Npy am = npy_load(ref + "/" + p.name + "/head_argmax.npy");
            long agree = 0;
            const int S = p.T - p.from;
            for (int r = 0; r < S; ++r) {
                CK(cudaMemcpy(p.sc.h, p.Hc + (size_t)(p.from + r) * HC_DIM, HC_DIM * 4, cudaMemcpyDeviceToDevice));
                CK(cudaMemcpy(p.sc.pre, p.Pc + (size_t)(p.from + r) * 4, 16, cudaMemcpyDeviceToDevice));
                v41m::head_row(G, p.sc, 0, st);
                int* dt; CK(cudaMalloc(&dt, 4));
                v41m::argmax_logits(p.sc, dt, st);
                int t; CK(cudaMemcpy(&t, dt, 4, cudaMemcpyDeviceToHost)); cudaFree(dt);
                agree += t == am.as<int32_t>()[r];
            }
            const double ag = (double)agree / S;
            const bool bad = ag < 0.97 || p.mean_chain.back() < 0.995;
            fails += bad;
            printf("  %-8s chained final mean cos %.6f, teacher-forced argmax agreement %.4f (%ld/%d)%s\n", p.name.c_str(),
                   p.mean_chain.back(), ag, agree, S, bad ? "   <-- FAIL" : "");
        }
    }
    printf("Gate V41 GOLDEN: %s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
