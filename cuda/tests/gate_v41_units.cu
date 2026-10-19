// gate_v41_units.cu — V4.1 kernels against the ops oracles, no checkpoint (GPU; run by the chain).
//
//   python -I tests/v41_unit_ref.py --out build/units
//   scripts/build_v41.sh gate_v41_units && build/gate_v41_units build/units
//
// Exactness classes (what each check may and may not tolerate):
//   EXACT   decoders (K-quants, MXFP4 hw + sw, KT2/KT3 trellis), act_quant bytes and scales, router
//           expert ids, the KT activation rotation, the KT Viterbi encoder's states
//   ULP     fp8/bf16 linears: fp32 summation order differs from torch's, so a bf16 output may sit
//           one ulp away; >1 ulp anywhere, or >1% of outputs off by one, is a failure
//   COS     moe_routed vs FQExpert-style math (bf16 roundings at the reference's points): cos >= 0.99999
#include "../kernels/v41/v41_gemv.cuh"
#include "v41_kernels.h"
#include "v41_config.h"
#include "npy.h"
#include <cmath>
#include <cstdio>
#include <vector>

extern "C" int kt_viterbi(const float* X, const float* W, int32_t* out, int n, int b, void* stream);   // kt_viterbi.cu

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e_)); exit(2); } } while (0)

static int g_fail = 0;
static void verdict(bool ok, const char* what, const char* detail) {
    printf("  [%s] %-34s %s\n", ok ? "PASS" : "FAIL", what, detail);
    if (!ok) ++g_fail;
}
template <class T> static T* up(const Npy& n) {
    T* d; CK(cudaMalloc(&d, n.data.size() + 256)); CK(cudaMemcpy(d, n.data.data(), n.data.size(), cudaMemcpyHostToDevice)); return d;
}
template <class T> static T* dzero(size_t n) { T* d; CK(cudaMalloc(&d, n * sizeof(T) + 256)); CK(cudaMemset(d, 0, n * sizeof(T) + 256)); return d; }
template <class T> static std::vector<T> down(const T* d, size_t n) {
    std::vector<T> h(n); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost)); return h;
}
static uint32_t bfbits(float f) { uint32_t u; memcpy(&u, &f, 4); return u >> 16; }

template <class Dec>
__global__ void k_decode(float* out, Dec d, int R, int K) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, nc = K / 32;
    if (i >= R * nc) return;
    float w[32]; d.load(i / nc, i % nc, w);
    for (int j = 0; j < 32; ++j) out[(size_t)(i / nc) * K + (i % nc) * 32 + j] = w[j];
}
template <class Dec>
static void dec_case(const char* name, Dec d, const Npy& ref) {
    const int R = (int)ref.shape[0], K = (int)ref.shape[1];
    float* o = dzero<float>((size_t)R * K);
    k_decode<<<(R * K / 32 + 127) / 128, 128>>>(o, d, R, K);
    auto h = down(o, (size_t)R * K);
    long bad = 0;
    for (size_t i = 0; i < h.size(); ++i) bad += h[i] != ref.as<float>()[i];
    char b[96]; snprintf(b, sizeof b, "%ld / %zu weights differ", bad, h.size());
    verdict(bad == 0, name, b);
    cudaFree(o);
}

int main(int argc, char** argv) {
    const std::string D = argc > 1 ? argv[1] : "build/units";
    auto L = [&](const char* n) { return npy_load(D + "/" + n + ".npy"); };
    v41k::v41k_init((size_t)1 << 22);
    cudaStream_t st = 0;
    printf("Gate V41 UNITS (%s)\n", D.c_str());

    // ---- decoders
    {
        const char* q[3] = {"Q2_K", "Q3_K", "Q4_K"};
        for (int t = 0; t < 3; ++t) {
            const Npy w = L((std::string("dec_") + q[t] + "_w").c_str()), ref = L((std::string("dec_") + q[t] + "_ref").c_str());
            const uint8_t* dw = up<uint8_t>(w);
            const int K = (int)ref.shape[1];
            char nm[48]; snprintf(nm, sizeof nm, "decode %s", q[t]);
            if (t == 0) { v41g::DecKQ<2> d; d.W0 = d.W1 = dw; d.K = K; d.split = 1 << 30; dec_case(nm, d, ref); }
            if (t == 1) { v41g::DecKQ<3> d; d.W0 = d.W1 = dw; d.K = K; d.split = 1 << 30; dec_case(nm, d, ref); }
            if (t == 2) { v41g::DecKQ<4> d; d.W0 = d.W1 = dw; d.K = K; d.split = 1 << 30; dec_case(nm, d, ref); }
        }
        const Npy w = L("dec_MXFP4_w"), s = L("dec_MXFP4_s"), ref = L("dec_MXFP4_ref");
        const uint8_t *dw = up<uint8_t>(w), *ds = up<uint8_t>(s);
        const int K = (int)ref.shape[1];
        { v41g::DecMXFP4<true> d; d.W0 = d.W1 = dw; d.S0 = d.S1 = ds; d.K = K; d.split = 1 << 30; dec_case("decode MXFP4 (hw cvt e2m1x2)", d, ref); }
        { v41g::DecMXFP4<false> d; d.W0 = d.W1 = dw; d.S0 = d.S1 = ds; d.K = K; d.split = 1 << 30; dec_case("decode MXFP4 (bit construct)", d, ref); }
        for (int b = 2; b <= 3; ++b) {
            const std::string q = b == 2 ? "KT2" : "KT3";
            const Npy kw = L(("dec_" + q + "_w").c_str()), ks = L(("dec_" + q + "_s").c_str()), kr = L(("dec_" + q + "_ref").c_str());
            const uint8_t *dkw = up<uint8_t>(kw), *dks = up<uint8_t>(ks);
            const int KK = (int)kr.shape[1];
            const std::string nm = "decode " + q + " (trellis, L=12)";
            if (b == 2) { v41g::DecKT<2> d; d.W0 = d.W1 = dkw; d.S0 = d.S1 = dks; d.K = KK; d.split = 1 << 30; dec_case(nm.c_str(), d, kr); }
            else { v41g::DecKT<3> d; d.W0 = d.W1 = dkw; d.S0 = d.S1 = dks; d.K = KK; d.split = 1 << 30; dec_case(nm.c_str(), d, kr); }
        }
    }

    // ---- KT activation rotation (exact butterfly) and the KT Viterbi encoder (exact states)
    for (int K : {5120, 2304}) {
        const std::string t = "rht" + std::to_string(K);
        const Npy x = L((t + "_x").c_str()), sg = L((t + "_sign").c_str()), ref = L((t + "_ref").c_str());
        const int M = (int)x.shape[0];
        float* o = dzero<float>((size_t)M * K);
        v41k::rht_rows(o, up<float>(x), up<uint8_t>(sg), M, K, st);
        auto h = down(o, (size_t)M * K);
        long bad = 0;
        for (size_t i = 0; i < h.size(); ++i) bad += h[i] != ref.as<float>()[i];
        char b[96]; snprintf(b, sizeof b, "%ld / %zu values differ", bad, h.size());
        verdict(bad == 0, (std::string("KT rotation K=") + std::to_string(K)).c_str(), b);
    }
    {
        const Npy vx = L("vit_x"), vw = L("vit_w");
        const int N = (int)vx.shape[0];
        float *dx = up<float>(vx), *dw = up<float>(vw);
        for (int b = 2; b <= 3; ++b) {
            const Npy ref = L(("vit_b" + std::to_string(b) + "_states").c_str());
            int32_t* o = dzero<int32_t>((size_t)N * 128);
            const int rc = kt_viterbi(dx, dw, o, N, b, nullptr);
            auto h = down(o, (size_t)N * 128);
            long bad = 0, badseq = 0;
            for (int n = 0; n < N; ++n) {
                long d = 0;
                for (int i = 0; i < 128; ++i) d += h[(size_t)n * 128 + i] != ref.as<int32_t>()[(size_t)n * 128 + i];
                bad += d; badseq += d > 0;
            }
            char bb[96]; snprintf(bb, sizeof bb, "rc %d, %ld windows in %ld / %d sequences differ", rc, bad, badseq, N);
            verdict(rc == 0 && bad == 0, (std::string("KT Viterbi encoder b=") + std::to_string(b)).c_str(), bb);
        }
    }

    // ---- act_quant
    {
        const Npy x = L("aq_x"), q = L("aq_q"), s = L("aq_s");
        const int M = (int)x.shape[0], K = (int)x.shape[1];
        float* dx = up<float>(x); uint8_t* dq = dzero<uint8_t>((size_t)M * K); float* ds = dzero<float>((size_t)M * K / 32);
        v41k::act_quant_fp8(dq, ds, dx, M, K, st);
        auto hq = down(dq, (size_t)M * K); auto hs = down(ds, (size_t)M * K / 32);
        long bq = 0, bs = 0;
        for (size_t i = 0; i < hq.size(); ++i) {
            const uint8_t a = hq[i], b = q.as<uint8_t>()[i];
            bq += a != b && !((a & 0x7f) == 0 && (b & 0x7f) == 0);       // +0 / -0
        }
        for (size_t i = 0; i < hs.size(); ++i) bs += hs[i] != s.as<float>()[i];
        char b[96]; snprintf(b, sizeof b, "%ld byte / %ld scale mismatches", bq, bs);
        verdict(bq == 0 && bs == 0, "act_quant (32, ue8m0)", b);
    }

    // ---- fp8 linear, bf16 linear
    auto ulp_check = [&](const char* nm, const std::vector<float>& got, const Npy& ref) {
        long one = 0, more = 0;
        for (size_t i = 0; i < got.size(); ++i) {
            const long d = std::labs((long)bfbits(got[i]) - (long)bfbits(ref.as<float>()[i]));
            one += d == 1; more += d > 1;
        }
        char b[96]; snprintf(b, sizeof b, "%ld off by 1 ulp, %ld by more (of %zu)", one, more, got.size());
        verdict(more == 0 && one * 100 <= (long)got.size(), nm, b);
    };
    {
        const Npy x = L("lin_x"), w = L("lin_w"), ws = L("lin_ws"), ref = L("lin_ref");
        const int M = (int)x.shape[0], K = (int)x.shape[1], N = (int)w.shape[0];
        float* dx = up<float>(x); uint8_t* dw = up<uint8_t>(w); uint8_t* dws = up<uint8_t>(ws);
        uint8_t* xq = dzero<uint8_t>((size_t)M * K); float* xs = dzero<float>((size_t)M * K / 32); float* y = dzero<float>((size_t)M * N);
        v41k::act_quant_fp8(xq, xs, dx, M, K, st);
        v41k::fp8_linear(y, xq, xs, dw, dws, M, N, K, true, st);
        ulp_check("fp8_linear", down(y, (size_t)M * N), ref);
    }
    {
        const Npy x = L("blin_x"), w = L("blin_w"), ref = L("blin_ref");
        const int M = (int)x.shape[0], K = (int)x.shape[1], N = (int)w.shape[0];
        float* dx = up<float>(x); uint16_t* dw = up<uint16_t>(w); float* y = dzero<float>((size_t)M * N);
        v41k::bf16_linear(y, dx, dw, M, N, K, true, st);
        ulp_check("bf16_linear", down(y, (size_t)M * N), ref);
    }

    // ---- router
    {
        const Npy x = L("rt_x"), w = L("rt_w"), b = L("rt_bias"), bv = L("rt_bias_vl"), img = L("rt_img"), ri = L("rt_idx"), rw = L("rt_wt");
        const int M = (int)x.shape[0], E = (int)w.shape[0], TK = (int)ri.shape[1];
        float* wo = dzero<float>(M * TK); int* io = dzero<int>(M * TK);
        v41k::router(wo, io, up<float>(x), up<uint16_t>(w), up<float>(b), up<float>(bv), up<uint8_t>(img), M, E, TK, 1.5f, st);
        auto hw = down(wo, M * TK); auto hi = down(io, M * TK);
        long bad_i = 0; double maxrel = 0;
        for (int i = 0; i < M * TK; ++i) {
            bad_i += hi[i] != ri.as<int32_t>()[i];
            maxrel = std::max(maxrel, (double)(std::fabs(hw[i] - rw.as<float>()[i]) / std::fabs(rw.as<float>()[i])));
        }
        char bb[96]; snprintf(bb, sizeof bb, "%ld id mismatches, max weight rel err %.2e", bad_i, maxrel);
        verdict(bad_i == 0 && maxrel < 1e-5, "router (sqrtsoftplus, bias_vl)", bb);
    }

    // ---- moe_routed, Q3_K gate/up + Q4_K down (per-pass formats)
    {
        const Npy x = L("moe_x"), idx = L("moe_idx"), wts = L("moe_wts"), ref = L("moe_ref");
        const Npy w1 = L("moe_w1"), w2 = L("moe_w2"), w3 = L("moe_w3");
        const int M = (int)x.shape[0], TK = (int)idx.shape[1], E = (int)w1.shape[0];
        uint8_t *d1 = up<uint8_t>(w1), *d2 = up<uint8_t>(w2), *d3 = up<uint8_t>(w3);
        std::vector<const uint8_t*> tab(6 * E, nullptr);
        for (int e = 0; e < E; ++e) {
            tab[0 * E + e] = d1 + (size_t)e * (w1.data.size() / E);
            tab[2 * E + e] = d3 + (size_t)e * (w3.data.size() / E);
            tab[4 * E + e] = d2 + (size_t)e * (w2.data.size() / E);
        }
        const uint8_t** dt; CK(cudaMalloc(&dt, tab.size() * sizeof(void*)));
        CK(cudaMemcpy(dt, tab.data(), tab.size() * sizeof(void*), cudaMemcpyHostToDevice));
        v41k::ExpertTab T;
        T.fmt13 = v41k::ExpertFmt::KQ; T.bits13 = 3; T.fmt2 = v41k::ExpertFmt::KQ; T.bits2 = 4;
        T.w1 = dt; T.w1s = dt + E; T.w3 = dt + 2 * E; T.w3s = dt + 3 * E; T.w2 = dt + 4 * E; T.w2s = dt + 5 * E;
        float* dx = up<float>(x);
        uint8_t* xq = dzero<uint8_t>((size_t)M * v41::DIM); float* xs = dzero<float>((size_t)M * v41::DIM / 32);
        float* y = dzero<float>((size_t)M * v41::DIM);
        void* scr; CK(cudaMalloc(&scr, v41k::moe_scratch_bytes(M, TK)));
        v41k::act_quant_fp8(xq, xs, dx, M, v41::DIM, st);
        v41k::moe_routed(y, xq, xs, up<float>(wts), up<int>(idx), T, M, TK, scr, st);
        auto h = down(y, (size_t)M * v41::DIM);
        double dot = 0, n1 = 0, n2 = 0, maxd = 0;
        for (size_t i = 0; i < h.size(); ++i) {
            const double a = h[i], b = ref.as<float>()[i];
            dot += a * b; n1 += a * a; n2 += b * b; maxd = std::max(maxd, std::fabs(a - b));
        }
        const double cs = dot / std::sqrt(n1 * n2);
        char bb[96]; snprintf(bb, sizeof bb, "cos %.8f, max |d| %.3g", cs, maxd);
        verdict(cs >= 0.99999, "moe_routed Q3_K/Q4_K, 8 experts", bb);
    }

    // ---- moe_routed, KT3 gate/up + KT2 down: rotated passes through the real dispatch
    {
        const Npy x = L("moekt_x"), idx = L("moekt_idx"), wts = L("moekt_wts"), ref = L("moekt_ref");
        const Npy w1 = L("moekt_w1"), w2 = L("moekt_w2"), w3 = L("moekt_w3");
        const Npy s1 = L("moekt_w1_s"), s2 = L("moekt_w2_s"), s3 = L("moekt_w3_s");
        const Npy g13 = L("moekt_s13"), g2 = L("moekt_s2");
        const int M = (int)x.shape[0], TK = (int)idx.shape[1], E = (int)w1.shape[0];
        uint8_t *d1 = up<uint8_t>(w1), *d2 = up<uint8_t>(w2), *d3 = up<uint8_t>(w3);
        uint8_t *e1 = up<uint8_t>(s1), *e2 = up<uint8_t>(s2), *e3 = up<uint8_t>(s3);
        std::vector<const uint8_t*> tab(6 * E, nullptr);
        for (int e = 0; e < E; ++e) {
            tab[0 * E + e] = d1 + (size_t)e * (w1.data.size() / E); tab[1 * E + e] = e1 + (size_t)e * (s1.data.size() / E);
            tab[2 * E + e] = d3 + (size_t)e * (w3.data.size() / E); tab[3 * E + e] = e3 + (size_t)e * (s3.data.size() / E);
            tab[4 * E + e] = d2 + (size_t)e * (w2.data.size() / E); tab[5 * E + e] = e2 + (size_t)e * (s2.data.size() / E);
        }
        const uint8_t** dt; CK(cudaMalloc(&dt, tab.size() * sizeof(void*)));
        CK(cudaMemcpy(dt, tab.data(), tab.size() * sizeof(void*), cudaMemcpyHostToDevice));
        v41k::ExpertTab T;
        T.fmt13 = v41k::ExpertFmt::KT; T.bits13 = 3; T.fmt2 = v41k::ExpertFmt::KT; T.bits2 = 2;
        T.w1 = dt; T.w1s = dt + E; T.w3 = dt + 2 * E; T.w3s = dt + 3 * E; T.w2 = dt + 4 * E; T.w2s = dt + 5 * E;
        T.rht13 = up<uint8_t>(g13); T.rht2 = up<uint8_t>(g2);
        float* dx = up<float>(x);
        uint8_t* xq = dzero<uint8_t>((size_t)M * v41::DIM); float* xs = dzero<float>((size_t)M * v41::DIM / 32);
        float* y = dzero<float>((size_t)M * v41::DIM);
        void* scr; CK(cudaMalloc(&scr, v41k::moe_scratch_bytes(M, TK)));
        v41k::act_quant_fp8(xq, xs, dx, M, v41::DIM, st);
        v41k::moe_routed(y, xq, xs, up<float>(wts), up<int>(idx), T, M, TK, scr, st);
        auto h = down(y, (size_t)M * v41::DIM);
        double dot = 0, n1 = 0, n2 = 0, maxd = 0;
        for (size_t i = 0; i < h.size(); ++i) {
            const double a = h[i], b = ref.as<float>()[i];
            dot += a * b; n1 += a * a; n2 += b * b; maxd = std::max(maxd, std::fabs(a - b));
        }
        const double cs = dot / std::sqrt(n1 * n2);
        char bb[96]; snprintf(bb, sizeof bb, "cos %.8f, max |d| %.3g", cs, maxd);
        verdict(cs >= 0.99999, "moe_routed KT3/KT2 (rotated), 6 experts", bb);
    }

    CK(cudaDeviceSynchronize());
    printf("Gate V41 UNITS: %s (%d failing)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
