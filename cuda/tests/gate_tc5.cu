// gate_tc5.cu — P8 stage 1: tcgen05 NVFP4-overlay GEMM (v41_tc5.cu) against k_nvm (fp8_linear) and fp64.
//   build/gate_tc5     every overlay shape at M=256: max error vs fp64 (both paths), agreement, timing
// Sane data as gate_nvm: random e2m1 codes, e4m3 block scales 0x30..0x3f, e4m3 x without NaN, pow2 x scales.
#include <cstdio>
#include <vector>
#include <random>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <cuda_fp8.h>
#include "v41_kernels.h"
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA %s line %d\n", cudaGetErrorString(e_), __LINE__); return 1; } } while (0)
template <class F> static float time_it(F f, int it = 20) {
    for (int w = 0; w < 3; ++w) f();
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a); for (int i = 0; i < it; ++i) f(); cudaEventRecord(b); cudaEventSynchronize(b);
    float ms; cudaEventElapsedTime(&ms, a, b); return ms / it;
}
static float e2m1(int c) { const float m[8] = {0, .5f, 1, 1.5f, 2, 3, 4, 6}; return (c & 8 ? -1.f : 1.f) * m[c & 7]; }
static float e4m3(uint8_t b) { __nv_fp8_e4m3 v; v.__x = b; return (float)v; }
int main() {
    v41k::v41k_init((size_t)1 << 24);
    std::mt19937 rng(7);
    bool ok = true;
    struct S { const char* n; int N, K, G; };
    for (S s : {S{"wq_a", 1280, 5120, 1}, S{"wkv", 512, 5120, 1}, S{"wq_b", 32768, 1280, 1}, S{"wo_a", 8192, 4096, 8},
                S{"wo_b", 5120, 8192, 1}, S{"sh_w1", 2304, 5120, 1}, S{"sh_w2", 5120, 2304, 1}, S{"idx_wq_b", 8192, 1280, 1}}) {
        const size_t cb = (size_t)s.N * s.K / 2, sb = (size_t)s.N * s.K / 16, KX = (size_t)s.K * s.G;
        std::vector<uint8_t> hw(cb), hsc(sb); for (auto& v : hw) v = rng() & 0xff; for (auto& v : hsc) v = 0x30 + rng() % 16;
        const float g = 0.0137f;
        const int MM = 256;
        std::vector<uint8_t> hx((size_t)MM * KX); for (auto& v : hx) { uint8_t b; do b = rng() & 0xff; while ((b & 0x7f) == 0x7f); v = b; }
        std::vector<float> hxs((size_t)MM * KX / 32); for (auto& v : hxs) v = ldexpf(1.f, -(int)(rng() % 6));
        uint8_t *W, *Sc, *xq; float *xs, *ya, *yb;
        CK(cudaMalloc(&W, cb)); CK(cudaMalloc(&Sc, sb));
        CK(cudaMemcpy(W, hw.data(), cb, cudaMemcpyHostToDevice)); CK(cudaMemcpy(Sc, hsc.data(), sb, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&xq, hx.size())); CK(cudaMalloc(&xs, hxs.size() * 4));
        CK(cudaMemcpy(xq, hx.data(), hx.size(), cudaMemcpyHostToDevice)); CK(cudaMemcpy(xs, hxs.data(), hxs.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&ya, (size_t)MM * s.N * 4)); CK(cudaMalloc(&yb, (size_t)MM * s.N * 4));
        v41k::overlay_register(W, Sc, g, s.N, s.K, 1);                  // k_nvm layout, as in the engine
        v41k::fp8_linear(ya, xq, xs, W, Sc, MM, s.N, s.K, false, 0, s.G);
        const bool ran = v41k::tc5_nvfp4(yb, xq, xs, (int)KX, W, Sc, g, MM, s.N, s.K, s.G, false, 0);
        CK(cudaDeviceSynchronize());
        if (!ran) { printf("%-9s tc5 not applicable\n", s.n); ok = false; continue; }
        std::vector<float> oa((size_t)MM * s.N), ob(oa.size());
        CK(cudaMemcpy(oa.data(), ya, oa.size() * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(ob.data(), yb, ob.size() * 4, cudaMemcpyDeviceToHost));
        double rms = 0, ea = 0, eb = 0; const int Ng = s.N / s.G; int n = 0;
        for (int m : {0, 1, 129, MM - 1})
            for (int r = 0; r < s.N; r += 7) {
                const int gi = r / Ng; double ref = 0;
                for (int k = 0; k < s.K; ++k) {
                    const uint8_t b = hw[(size_t)r * (s.K / 2) + k / 2];
                    const double wv = (double)e2m1(k & 1 ? b >> 4 : b & 15) * e4m3(hsc[(size_t)r * (s.K / 16) + k / 16]) * g;
                    const size_t xi = (size_t)m * KX + (size_t)gi * s.K + k;
                    ref += wv * (double)e4m3(hx[xi]) * hxs[xi / 32];
                }
                const size_t o = (size_t)m * s.N + r;
                rms += ref * ref; ++n; ea = std::max(ea, fabs(oa[o] - ref)); eb = std::max(eb, fabs(ob[o] - ref));
            }
        rms = sqrt(rms / n);
        const bool good = eb <= rms * 1e-4;          // tcgen05 fp32 accumulation: ~2e-5 relative (outputs round to bf16, 4e-3)
        const float ta = time_it([&] { v41k::fp8_linear(ya, xq, xs, W, Sc, MM, s.N, s.K, true, 0, s.G); });
        const float tb = time_it([&] { v41k::tc5_nvfp4(yb, xq, xs, (int)KX, W, Sc, g, MM, s.N, s.K, s.G, true, 0); });
        printf("%-9s N=%5d K=%5d G=%d  rms %.3e  maxerr k_nvm %.3e tc5 %.3e  M=256: k_nvm %.3f ms  tc5 %.3f ms  %.2fx  %s\n", s.n,
               s.N, s.K, s.G, rms, ea, eb, ta, tb, ta / tb, good ? "ok" : "FAIL");
        ok &= good;
        cudaFree(W); cudaFree(Sc); cudaFree(xq); cudaFree(xs); cudaFree(ya); cudaFree(yb);
    }
    printf("%s\n", ok ? "GATE PASS" : "GATE FAIL");
    return ok ? 0 : 1;
}
