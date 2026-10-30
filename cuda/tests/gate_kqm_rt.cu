// gate_kqm_rt.cu — k_kqm with RT row tiles per warp (prefill) reproduces RT = 1 bit for bit; timing.
//   build/gate_kqm_rt    Q2_K / Q3_K x gate|up / down, 24 experts x 4 units of 16 (the last one partial)
// Random repacked bytes with finite fp16 scales: the check is bitwise, not against a reference.
#include "v41_kernels.h"
#include "v41_config.h"
#include "v41_kq_mma.cuh"
#include <cstdio>
#include <cstring>
#include <vector>
#include <random>
using namespace v41g;
struct Unit { int e, start, cnt; };
#define CU(x) do { cudaError_t e_ = (x); if (e_) { printf("CUDA %s @%d\n", cudaGetErrorString(e_), __LINE__); exit(1); } } while (0)
template <class F> static float time_it(F f) {
    for (int w = 0; w < 2; ++w) f();
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a); for (int i = 0; i < 5; ++i) f(); cudaEventRecord(b); cudaEventSynchronize(b);
    float ms; cudaEventElapsedTime(&ms, a, b); return ms / 5;
}
template <int QT, int RT> static bool run(int pass, std::mt19937& g) {
    const int N = pass == 0 ? 2 * v41::MOE_INTER : v41::DIM, K = pass == 0 ? v41::DIM : v41::MOE_INTER;
    const int E = 24, R = 4, rows = pass == 0 ? v41::MOE_INTER : v41::DIM;
    const size_t bpe = (size_t)rows * (K / 256) * (QT == 2 ? 84 : 110);
    std::vector<uint8_t> h(bpe);
    for (size_t b = 0; b < bpe; ++b) h[b] = g() & 0xff;
    const size_t cnt = (size_t)(rows / 16) * (K / 256), doff = QT == 2 ? 1280 * cnt : 1728 * cnt, dlen = QT == 2 ? 64 * cnt : 32 * cnt;
    for (size_t b = doff + 1; b < doff + dlen; b += 2) h[b] &= 0x1f;   // fp16 d/dmin high bytes: small, finite
    std::vector<const uint8_t*> w(E);
    for (int e = 0; e < E; ++e) { uint8_t* d; CU(cudaMalloc(&d, bpe)); CU(cudaMemcpy(d, h.data(), bpe, cudaMemcpyHostToDevice)); w[e] = d; }
    const uint8_t** tw; CU(cudaMalloc(&tw, E * 8)); CU(cudaMemcpy(tw, w.data(), E * 8, cudaMemcpyHostToDevice));
    KQTab tab{tw, tw, tw, pass};
    const int P = E * R * 16;
    std::vector<uint8_t> hx((size_t)P * K); for (auto& b : hx) { do b = g() & 0xff; while ((b & 0x7f) == 0x7f); }
    std::vector<float> hs((size_t)P * K / 32); for (auto& s : hs) s = ldexpf(1.f, -(int)(g() % 8) - 4);
    uint8_t* dx; float* dxs; CU(cudaMalloc(&dx, hx.size())); CU(cudaMalloc(&dxs, hs.size() * 4));
    CU(cudaMemcpy(dx, hx.data(), hx.size(), cudaMemcpyHostToDevice)); CU(cudaMemcpy(dxs, hs.data(), hs.size() * 4, cudaMemcpyHostToDevice));
    XSrc X; X.xq = dx; X.xs = dxs; X.ldx = K;
    uint8_t* stm; CU(cudaMalloc(&stm, (size_t)P * kqm_stage_bytes(K))); k_kqm_prep<<<P, 256>>>(stm, X, K); X.stg = stm;
    std::vector<Unit> hu;
    for (int e = 0, p = 0; e < E; ++e) for (int r = 0; r < R; ++r) { hu.push_back(Unit{e, p, r == R - 1 ? 16 - e % 7 : 16}); p += 16; }
    const int nu = (int)hu.size();
    Unit* du; int* dnu; int* ptok; CU(cudaMalloc(&du, nu * sizeof(Unit))); CU(cudaMalloc(&dnu, 4)); CU(cudaMalloc(&ptok, P * 4));
    std::vector<int> id(P); for (int i = 0; i < P; ++i) id[i] = i;
    CU(cudaMemcpy(ptok, id.data(), P * 4, cudaMemcpyHostToDevice));
    CU(cudaMemcpy(du, hu.data(), nu * sizeof(Unit), cudaMemcpyHostToDevice)); CU(cudaMemcpy(dnu, &nu, 4, cudaMemcpyHostToDevice));
    float *o1, *o2; CU(cudaMalloc(&o1, (size_t)P * N * 4)); CU(cudaMalloc(&o2, (size_t)P * N * 4));
    CU(cudaMemset(o1, 0, (size_t)P * N * 4)); CU(cudaMemset(o2, 0, (size_t)P * N * 4));
    const size_t sm = 16 * kqm_stage_bytes(K);
    CU(cudaFuncSetAttribute(k_kqm<QT, 16, Unit, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm));
    CU(cudaFuncSetAttribute(k_kqm<QT, 16, Unit, RT>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm));
    auto l1 = [&] { k_kqm<QT, 16, Unit, 1><<<dim3((N + 127) / 128, nu), 256, sm>>>(o1, N, X, true, du, dnu, ptok, tab, N, K, 1, 1 << 30); };
    auto lr = [&] { k_kqm<QT, 16, Unit, RT><<<dim3((N + 128 * RT - 1) / (128 * RT), nu), 256, sm>>>(o2, N, X, true, du, dnu, ptok, tab, N, K, 1, 1 << 30); };
    l1(); lr(); CU(cudaGetLastError()); CU(cudaDeviceSynchronize());
    std::vector<float> a((size_t)P * N), b(a.size());
    CU(cudaMemcpy(a.data(), o1, a.size() * 4, cudaMemcpyDeviceToHost)); CU(cudaMemcpy(b.data(), o2, b.size() * 4, cudaMemcpyDeviceToHost));
    long diff = 0, nz = 0; for (size_t i = 0; i < a.size(); ++i) { diff += memcmp(&a[i], &b[i], 4) != 0; nz += a[i] != 0.f; }
    const float t1 = time_it(l1), tr = time_it(lr);
    printf("Q%d_K pass %d RT=%d: %ld / %zu differ (%ld nonzero)  RT=1 %.3f ms  RT=%d %.3f ms  %.2fx  %s\n", QT, pass, RT, diff, a.size(), nz,
           t1, RT, tr, t1 / tr, diff == 0 && nz > 0 ? "ok" : "FAIL");
    for (auto p : w) cudaFree((void*)p);
    cudaFree(tw); cudaFree(dx); cudaFree(dxs); cudaFree(stm); cudaFree(du); cudaFree(dnu); cudaFree(ptok); cudaFree(o1); cudaFree(o2);
    return diff == 0 && nz > 0;
}
int main() {
    std::mt19937 g(5);
    bool ok = true;
    for (int pass : {0, 1}) { ok &= run<2, 4>(pass, g); ok &= run<3, 4>(pass, g); ok &= run<2, 2>(pass, g); }
    printf("%s\n", ok ? "GATE PASS" : "GATE FAIL");
    return ok ? 0 : 1;
}
