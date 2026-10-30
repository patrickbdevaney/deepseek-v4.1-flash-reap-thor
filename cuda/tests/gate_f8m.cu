// gate_f8m.cu — FP8 dense on fp8 tensor cores (v41_f8_mma.cuh, P6) against k_dense<DecFP8> and fp64.
//   build/gate_f8m      Engram wkv shape (+ a small one): accuracy (M=1), rows bitwise across M, timing old vs new
// Sane data: e4m3 weights and x without NaN, e8m0 block scales 120..127, pow2 x scales.
#include <cstdio>
#include <vector>
#include <random>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <cuda_fp8.h>
#include "v41_kernels.h"
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA %s line %d\n", cudaGetErrorString(e_), __LINE__); return 1; } } while (0)
static float time_it(auto f, int it = 20) {
    for (int w = 0; w < 3; ++w) f();
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a); for (int i = 0; i < it; ++i) f(); cudaEventRecord(b); cudaEventSynchronize(b);
    float ms; cudaEventElapsedTime(&ms, a, b); return ms / it;
}
static float e4m3(uint8_t b) { __nv_fp8_e4m3 v; v.__x = b; return (float)v; }
static uint8_t rnd_e4m3(std::mt19937& rng) { uint8_t b; do b = rng() & 0xff; while ((b & 0x7f) == 0x7f); return b; }
int main() {
    v41k::v41k_init((size_t)1 << 22);
    std::mt19937 rng(7);
    bool ok = true;
    struct S { const char* n; int N, K; };
    for (S s : {S{"small", 512, 2048}, S{"engram_wkv", 25600, 6144}}) {
        const size_t wb = (size_t)s.N * s.K, sbn = (size_t)(s.N / 32) * (s.K / 32);
        std::vector<uint8_t> hw(wb), hsc(sbn);
        for (auto& v : hw) v = rnd_e4m3(rng);
        for (auto& v : hsc) v = 120 + rng() % 8;
        const int MM = 256;
        std::vector<uint8_t> hx((size_t)MM * s.K); for (auto& v : hx) v = rnd_e4m3(rng);
        std::vector<float> hxs((size_t)MM * s.K / 32); for (auto& v : hxs) v = ldexpf(1.f, -(int)(rng() % 6));
        uint8_t *Wa, *Wb, *Sc, *xq; float *xs, *ya, *yb, *y1;
        CK(cudaMalloc(&Wa, wb)); CK(cudaMalloc(&Wb, wb)); CK(cudaMalloc(&Sc, sbn));
        CK(cudaMemcpy(Wa, hw.data(), wb, cudaMemcpyHostToDevice)); CK(cudaMemcpy(Wb, hw.data(), wb, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(Sc, hsc.data(), sbn, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&xq, hx.size())); CK(cudaMalloc(&xs, hxs.size() * 4));
        CK(cudaMemcpy(xq, hx.data(), hx.size(), cudaMemcpyHostToDevice)); CK(cudaMemcpy(xs, hxs.data(), hxs.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&ya, (size_t)MM * s.N * 4)); CK(cudaMalloc(&yb, (size_t)MM * s.N * 4)); CK(cudaMalloc(&y1, (size_t)s.N * 4));
        if (!v41k::f8m_register(Wb, s.N, s.K)) { printf("%s: f8m_register refused\n", s.n); return 1; }
        v41k::fp8_linear(ya, xq, xs, Wa, Sc, 1, s.N, s.K, false, 0);
        v41k::fp8_linear(yb, xq, xs, Wb, Sc, 1, s.N, s.K, false, 0);
        CK(cudaDeviceSynchronize());
        std::vector<float> oa(s.N), ob(s.N);
        CK(cudaMemcpy(oa.data(), ya, s.N * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(ob.data(), yb, s.N * 4, cudaMemcpyDeviceToHost));
        double rms = 0, ea = 0, eb = 0;
        const int nchk = std::min(s.N, 2048);
        for (int r = 0; r < nchk; ++r) {
            double ref = 0;
            for (int k = 0; k < s.K; ++k) {
                const double wv = (double)e4m3(hw[(size_t)r * s.K + k]) * ldexp(1.0, (int)hsc[(size_t)(r / 32) * (s.K / 32) + k / 32] - 127);
                ref += wv * (double)e4m3(hx[k]) * hxs[k / 32];
            }
            rms += ref * ref; ea = std::max(ea, fabs(oa[r] - ref)); eb = std::max(eb, fabs(ob[r] - ref));
        }
        rms = sqrt(rms / nchk);
        // rows bitwise across M (rounded, like the engine): M=1 vs rows of M=3, 17, 256
        bool bw = true;
        v41k::fp8_linear(y1, xq, xs, Wb, Sc, 1, s.N, s.K, true, 0);
        std::vector<float> o1(s.N); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(o1.data(), y1, s.N * 4, cudaMemcpyDeviceToHost));
        for (int M : {3, 17, 256}) {
            v41k::fp8_linear(yb, xq, xs, Wb, Sc, M, s.N, s.K, true, 0);
            std::vector<float> om((size_t)M * s.N); CK(cudaDeviceSynchronize());
            CK(cudaMemcpy(om.data(), yb, om.size() * 4, cudaMemcpyDeviceToHost));
            if (memcmp(om.data(), o1.data(), s.N * 4)) { bw = false; printf("  M=%d row 0 differs from M=1\n", M); }
            // row M-1 against its own M=1 run
            v41k::fp8_linear(y1, xq + (size_t)(M - 1) * s.K, xs + (size_t)(M - 1) * s.K / 32, Wb, Sc, 1, s.N, s.K, true, 0);
            std::vector<float> ol(s.N); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(ol.data(), y1, s.N * 4, cudaMemcpyDeviceToHost));
            if (memcmp(om.data() + (size_t)(M - 1) * s.N, ol.data(), s.N * 4)) { bw = false; printf("  M=%d last row differs from its M=1 run\n", M); }
        }
        const bool acc_ok = eb <= std::max(ea * 2.0, rms * 1e-5);
        printf("%-10s N=%5d K=%5d  rms %.3e  maxerr k_dense %.3e  k_f8m %.3e  rows bitwise across M: %s  %s\n", s.n, s.N, s.K,
               rms, ea, eb, bw ? "yes" : "NO", acc_ok && bw ? "ok" : "FAIL");
        ok &= acc_ok && bw;
        for (int M : {1, 3, 256}) {
            const float ta = time_it([&] { v41k::fp8_linear(ya, xq, xs, Wa, Sc, M, s.N, s.K, true, 0); });
            const float tb = time_it([&] { v41k::fp8_linear(yb, xq, xs, Wb, Sc, M, s.N, s.K, true, 0); });
            printf("    M=%3d  k_dense %8.3f ms (%5.0f GB/s)  k_f8m %8.3f ms (%5.0f GB/s)  %.2fx\n", M, ta, wb / ta / 1e6, tb,
                   wb / tb / 1e6, ta / tb);
        }
        cudaFree(Wa); cudaFree(Wb); cudaFree(Sc); cudaFree(xq); cudaFree(xs); cudaFree(ya); cudaFree(yb); cudaFree(y1);
    }
    printf("%s\n", ok ? "GATE PASS" : "GATE FAIL");
    return ok ? 0 : 1;
}
