// gate_kqt.cu — P8 stage 2b: Q2_K / Q3_K prefill experts on tcgen05 (v41_kqt.cuh) against fp64 and k_kqm.
//   build/gate_kqt    both passes; 24 experts x 64 tokens: k_kqt units of NT vs k_kqm units of 16; max err vs fp64; timing
#include "v41_kernels.h"
#include "v41_config.h"
#include "v41_kqt.cuh"
#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>
#include <random>
#include <algorithm>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
using namespace v41g;
struct Unit { int e, start, cnt; };
#define CU(x) do { cudaError_t e_ = (x); if (e_) { printf("CUDA %s @%d\n", cudaGetErrorString(e_), __LINE__); exit(1); } } while (0)
static std::vector<uint8_t> make_q2(int rows, int K, std::mt19937& g) {
    std::vector<uint8_t> h((size_t)rows * (K / 256) * 84);
    std::normal_distribution<float> nd(0.f, 1e-3f);
    for (size_t b = 0; b < h.size() / 84; ++b) {
        uint8_t* p = &h[b * 84];
        for (int i = 0; i < 80; ++i) p[i] = g() & 0xff;
        const __half d = __float2half(nd(g)), dm = __float2half(fabsf(nd(g)));
        memcpy(p + 80, &d, 2); memcpy(p + 82, &dm, 2);
    }
    return h;
}
static double q2w(const uint8_t* row, int k) {
    const uint8_t* b = row + (k / 256) * 84; const int kk = k & 255, sb = kk / 16;
    const int q = (b[16 + (kk >> 7) * 32 + (kk & 31)] >> (2 * ((kk & 127) >> 5))) & 3;
    __half d, dm; memcpy(&d, b + 80, 2); memcpy(&dm, b + 82, 2);
    return (double)__half2float(d) * (b[sb] & 0xF) * q - (double)__half2float(dm) * (b[sb] >> 4);
}
// ggml Q3_K: hmask[32] | qs[64] | scales[12] (6-bit, packed) | d f16 = 110 B; w = d (sc - 32) (q - (hbit ? 0 : 4))
static std::vector<uint8_t> make_q3(int rows, int K, std::mt19937& g) {
    std::vector<uint8_t> h((size_t)rows * (K / 256) * 110);
    std::normal_distribution<float> nd(0.f, 1e-4f);
    for (size_t b = 0; b < h.size() / 110; ++b) {
        uint8_t* p = &h[b * 110];
        for (int i = 0; i < 108; ++i) p[i] = g() & 0xff;
        const __half d = __float2half(nd(g));
        memcpy(p + 108, &d, 2);
    }
    return h;
}
static double q3w(const uint8_t* row, int k) {
    const uint8_t* b = row + (k / 256) * 110; const int kk = k & 255, sb = kk / 16;
    uint32_t aux[4]; memcpy(aux, b + 96, 12);
    const uint32_t tmp = aux[2], m1 = 0x03030303u, m2 = 0x0f0f0f0fu;
    aux[2] = ((aux[0] >> 4) & m2) | (((tmp >> 4) & m1) << 4);
    aux[3] = ((aux[1] >> 4) & m2) | (((tmp >> 6) & m1) << 4);
    aux[0] = (aux[0] & m2) | (((tmp >> 0) & m1) << 4);
    aux[1] = (aux[1] & m2) | (((tmp >> 2) & m1) << 4);
    const int sc = (int)reinterpret_cast<const uint8_t*>(aux)[sb] - 32;
    const int n = kk / 128, j = (kk & 127) / 32, l = kk & 31;
    const int q = (b[32 + n * 32 + l] >> (2 * j)) & 3, hb = (b[l] >> (n * 4 + j)) & 1;
    __half d; memcpy(&d, b + 108, 2);
    return (double)__half2float(d) * sc * (q - (hb ? 0 : 4));
}
static float e4(uint8_t v) { __nv_fp8_e4m3 x; x.__x = v; return (float)x; }
template <class F> static float time_it(F f) {
    for (int w = 0; w < 2; ++w) f();
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a); for (int i = 0; i < 5; ++i) f(); cudaEventRecord(b); cudaEventSynchronize(b);
    float ms; cudaEventElapsedTime(&ms, a, b); return ms / 5;
}
// kqt units of US tokens (<= NT) on a persistent grid of GY unit rows
template <int QT, int NT> static bool run(int pass, std::mt19937& g, int US, int GY) {
    const int N = pass == 0 ? 2 * v41::MOE_INTER : v41::DIM, K = pass == 0 ? v41::DIM : v41::MOE_INTER;
    const int E = 24, TPE = 64;                                  // tokens per expert
    const int rows13 = v41::MOE_INTER, rows2 = v41::DIM;
    std::vector<std::vector<uint8_t>> h1(E), h3(E), h2(E);
    std::vector<const uint8_t*> d1(E), d3(E), d2(E);
    const int BB = QT == 2 ? 84 : 110;                           // ggml block bytes
    auto mk = [&](int rows, int k) { return QT == 2 ? make_q2(rows, k, g) : make_q3(rows, k, g); };
    auto qw = [&](const uint8_t* row, int k) { return QT == 2 ? q2w(row, k) : q3w(row, k); };
    void* tmp; CU(cudaMalloc(&tmp, (size_t)v41::DIM * (v41::DIM / 256) * BB));
    auto put = [&](const std::vector<uint8_t>& h, int rows, int k) {
        uint8_t* d; CU(cudaMalloc(&d, h.size())); CU(cudaMemcpy(d, h.data(), h.size(), cudaMemcpyHostToDevice));
        v41k::kq_mma_repack(d, rows, k, QT, tmp, 0); return (const uint8_t*)d; };
    for (int e = 0; e < E; ++e) {
        if (pass == 0) { h1[e] = mk(rows13, K); h3[e] = mk(rows13, K); d1[e] = put(h1[e], rows13, K); d3[e] = put(h3[e], rows13, K); d2[e] = nullptr; }
        else { h2[e] = mk(rows2, K); d2[e] = put(h2[e], rows2, K); d1[e] = d3[e] = nullptr; }
    }
    CU(cudaDeviceSynchronize());
    const uint8_t **t1, **t3, **t2;
    CU(cudaMalloc(&t1, E * 8)); CU(cudaMalloc(&t3, E * 8)); CU(cudaMalloc(&t2, E * 8));
    CU(cudaMemcpy(t1, d1.data(), E * 8, cudaMemcpyHostToDevice)); CU(cudaMemcpy(t3, d3.data(), E * 8, cudaMemcpyHostToDevice));
    CU(cudaMemcpy(t2, d2.data(), E * 8, cudaMemcpyHostToDevice));
    KQTab tab{t1, t3, t2, pass};
    const int P = E * TPE;
    std::vector<uint8_t> hx((size_t)P * K); std::vector<float> hs((size_t)P * K / 32);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto& b : hx) b = (uint8_t)__nv_cvt_float_to_fp8(nd(g) * 40.f, __NV_SATFINITE, __NV_E4M3);
    for (auto& s : hs) s = ldexpf(1.f, -(int)(g() % 8) - 4);
    uint8_t* dx; float* dxs; CU(cudaMalloc(&dx, hx.size())); CU(cudaMalloc(&dxs, hs.size() * 4));
    CU(cudaMemcpy(dx, hx.data(), hx.size(), cudaMemcpyHostToDevice)); CU(cudaMemcpy(dxs, hs.data(), hs.size() * 4, cudaMemcpyHostToDevice));
    XSrc X; X.xq = dx; X.xs = dxs; X.ldx = K;
    // k_kqt units: NT tokens (last unit of each expert partial: TPE - e % 5 tokens in all); k_kqm units of 16
    std::vector<Unit> ut, um;
    for (int e = 0; e < E; ++e) {
        const int n = TPE - e % 5, p0 = e * TPE;
        for (int a = 0; a < n; a += US) ut.push_back(Unit{e, p0 + a, std::min(US, n - a)});
        for (int a = 0; a < n; a += 16) um.push_back(Unit{e, p0 + a, std::min(16, n - a)});
    }
    const int nut = (int)ut.size(), num = (int)um.size();
    Unit *dut, *dum; int *dnut, *dnum, *ptok;
    CU(cudaMalloc(&dut, nut * sizeof(Unit))); CU(cudaMalloc(&dum, num * sizeof(Unit))); CU(cudaMalloc(&dnut, 4)); CU(cudaMalloc(&dnum, 4));
    CU(cudaMalloc(&ptok, P * 4));
    std::vector<int> id(P); for (int i = 0; i < P; ++i) id[i] = i;
    CU(cudaMemcpy(ptok, id.data(), P * 4, cudaMemcpyHostToDevice));
    CU(cudaMemcpy(dut, ut.data(), nut * sizeof(Unit), cudaMemcpyHostToDevice)); CU(cudaMemcpy(dnut, &nut, 4, cudaMemcpyHostToDevice));
    CU(cudaMemcpy(dum, um.data(), num * sizeof(Unit), cudaMemcpyHostToDevice)); CU(cudaMemcpy(dnum, &num, 4, cudaMemcpyHostToDevice));
    uint8_t *stt, *stm; CU(cudaMalloc(&stt, (size_t)P * kqt_stage_bytes(K))); CU(cudaMalloc(&stm, (size_t)P * kqm_stage_bytes(K)));
    k_kqt_prep<QT><<<P, 256>>>(stt, X, K); k_kqm_prep<<<P, 256>>>(stm, X, K);
    XSrc Xt = X; Xt.stg = stt; XSrc Xm = X; Xm.stg = stm;
    float *ot, *om; CU(cudaMalloc(&ot, (size_t)P * N * 4)); CU(cudaMalloc(&om, (size_t)P * N * 4));
    CU(cudaMemset(ot, 0, (size_t)P * N * 4)); CU(cudaMemset(om, 0, (size_t)P * N * 4));
    const size_t smt = sizeof(KqtSmem<NT>) + 1024, smm = 16 * kqm_stage_bytes(K);
    constexpr int RT = QT == 2 ? 4 : 1;                          // k_kqm's shipped row tiles (V41_KQM_RT, Q2 only)
    CU(cudaFuncSetAttribute(k_kqt<QT, NT, Unit>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smt));
    CU(cudaFuncSetAttribute(k_kqm<QT, 16, Unit, RT>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smm));
    auto lt = [&] { k_kqt<QT, NT, Unit><<<dim3(N / 128, GY), KQT_THREADS, smt>>>(ot, N, Xt, true, dut, dnut, ptok, tab, N, K, 1, 1 << 30); };
    auto lm = [&] { k_kqm<QT, 16, Unit, RT><<<dim3((N + 128 * RT - 1) / (128 * RT), num), 256, smm>>>(om, N, Xm, true, dum, dnum, ptok, tab, N, K, 1, 1 << 30); };
    lt(); lm(); CU(cudaGetLastError()); CU(cudaDeviceSynchronize());
    std::vector<float> ht((size_t)P * N), hm(ht.size());
    CU(cudaMemcpy(ht.data(), ot, ht.size() * 4, cudaMemcpyDeviceToHost)); CU(cudaMemcpy(hm.data(), om, hm.size() * 4, cudaMemcpyDeviceToHost));
    double rms = 0, et = 0, em = 0; long nb = 0, bad = 0;
    std::vector<double> va, vr;
    for (int e : {0, 3, 23})
        for (int n : {0, 1, 17, 40, TPE - 1 - e % 5}) {
            const int p = e * TPE + n;
            for (int r = 0; r < N; r += 7) {
                const uint8_t* wr = pass == 0 ? (r < rows13 ? &h1[e][(size_t)r * (K / 256) * BB] : &h3[e][(size_t)(r - rows13) * (K / 256) * BB])
                                              : &h2[e][(size_t)r * (K / 256) * BB];
                double ref = 0;
                for (int k = 0; k < K; ++k) ref += qw(wr, k) * e4(hx[(size_t)p * K + k]) * hs[((size_t)p * K + k) / 32];
                rms += ref * ref; ++nb; vr.push_back(ref); va.push_back(ht[(size_t)p * N + r]);
                em = std::max(em, fabs(hm[(size_t)p * N + r] - ref));
            }
        }
    rms = sqrt(rms / nb);
    for (size_t i = 0; i < vr.size(); ++i) {
        et = std::max(et, fabs(va[i] - vr[i]) / rms);
        bad += fabs(va[i] - vr[i]) > std::max(1.01 * fabs(vr[i]) / 256.0, 1e-4 * rms);
    }
    const float tt = time_it(lt), tm = time_it(lm);
    const bool ok = bad == 0;
    printf("Q%d pass %d N=%d K=%d NT=%d US=%2d GY=%2d: max err/rms k_kqt %.3g k_kqm %.3g, out of tol %ld / %ld | %d tok: k_kqm %.3f ms  k_kqt %.3f ms  %.2fx  %s\n",
           QT, pass, N, K, NT, US, GY, et, em / rms, bad, nb, P, tm, tt, tm / tt, ok ? "ok" : "FAIL");
    return ok;
}
int main() {
    v41k::v41k_init((size_t)1 << 24);
    std::mt19937 g(3);
    bool ok = true;
    ok &= run<2, 64>(0, g, 64, 24); ok &= run<2, 64>(1, g, 64, 24);   // one unit per CTA row
    ok &= run<2, 64>(0, g, 64, 5);  ok &= run<2, 64>(1, g, 64, 7);    // persistent: several units per CTA
    ok &= run<2, 64>(0, g, 16, 10); ok &= run<2, 64>(0, g, 7, 10);    // small units on NT = 64 (16 / 7 columns staged)
    ok &= run<2, 32>(0, g, 32, 10);
    ok &= run<3, 64>(0, g, 64, 5);  ok &= run<3, 64>(1, g, 64, 7);    // Q3_K (f16 operands)
    ok &= run<3, 64>(0, g, 7, 10);
    printf("%s\n", ok ? "GATE PASS" : "GATE FAIL");
    return ok ? 0 : 1;
}
