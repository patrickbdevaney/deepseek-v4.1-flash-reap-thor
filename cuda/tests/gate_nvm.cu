// gate_nvm.cu — NVFP4 dense on fp8 tensor cores (v41_nv_mma.cuh) against k_dense<DecNVFP4> and fp64.
//   build/gate_nvm      every overlay shape: accuracy (M=1), rows bitwise across M, timing M=1/3/8 old vs new
// Sane data: random e2m1 codes, e4m3 block scales 0x30..0x3f, e4m3 x without NaN, pow2 x scales.
#include <cstdio>
#include <vector>
#include <random>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <cuda_fp8.h>
#include "v41_kernels.h"
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA %s line %d\n", cudaGetErrorString(e_), __LINE__); return 1; } } while (0)
static float time_it(auto f, int it = 30) {
    for (int w = 0; w < 3; ++w) f();
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a); for (int i = 0; i < it; ++i) f(); cudaEventRecord(b); cudaEventSynchronize(b);
    float ms; cudaEventElapsedTime(&ms, a, b); return ms / it;
}
static float e2m1(int c) { const float m[8] = {0, .5f, 1, 1.5f, 2, 3, 4, 6}; return (c & 8 ? -1.f : 1.f) * m[c & 7]; }
static float e4m3(uint8_t b) { __nv_fp8_e4m3 v; v.__x = b; return (float)v; }
int main() {
    v41k::v41k_init((size_t)1 << 22);
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
        uint8_t *Wa, *Sa, *Wb, *Sb, *xq; float *xs, *ya, *yb, *y1;
        CK(cudaMalloc(&Wa, cb)); CK(cudaMalloc(&Sa, sb)); CK(cudaMalloc(&Wb, cb)); CK(cudaMalloc(&Sb, sb));
        CK(cudaMemcpy(Wa, hw.data(), cb, cudaMemcpyHostToDevice)); CK(cudaMemcpy(Wb, hw.data(), cb, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(Sa, hsc.data(), sb, cudaMemcpyHostToDevice)); CK(cudaMemcpy(Sb, hsc.data(), sb, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&xq, hx.size())); CK(cudaMalloc(&xs, hxs.size() * 4));
        CK(cudaMemcpy(xq, hx.data(), hx.size(), cudaMemcpyHostToDevice)); CK(cudaMemcpy(xs, hxs.data(), hxs.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&ya, (size_t)MM * s.N * 4)); CK(cudaMalloc(&yb, (size_t)MM * s.N * 4)); CK(cudaMalloc(&y1, (size_t)s.N * 4));
        v41k::overlay_register(Wa, Sa, g, s.N, s.K, 0);
        v41k::overlay_register(Wb, Sb, g, s.N, s.K, 1);
        // M=1 accuracy vs fp64 (row 0 of x), unrounded outputs
        v41k::fp8_linear(ya, xq, xs, Wa, Sa, 1, s.N, s.K, false, 0, s.G);
        v41k::fp8_linear(yb, xq, xs, Wb, Sb, 1, s.N, s.K, false, 0, s.G);
        CK(cudaDeviceSynchronize());
        std::vector<float> oa(s.N), ob(s.N); CK(cudaMemcpy(oa.data(), ya, s.N * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(ob.data(), yb, s.N * 4, cudaMemcpyDeviceToHost));
        double rms = 0, ea = 0, eb = 0; const int Ng = s.N / s.G;
        for (int r = 0; r < s.N; ++r) {
            const int gi = r / Ng; double ref = 0;
            for (int k = 0; k < s.K; ++k) {
                const uint8_t b = hw[(size_t)r * (s.K / 2) + k / 2];
                const double wv = (double)e2m1(k & 1 ? b >> 4 : b & 15) * e4m3(hsc[(size_t)r * (s.K / 16) + k / 16]) * g;
                const size_t xi = (size_t)gi * s.K + k;
                ref += wv * (double)e4m3(hx[xi]) * hxs[xi / 32];
            }
            rms += ref * ref; ea = std::max(ea, fabs(oa[r] - ref)); eb = std::max(eb, fabs(ob[r] - ref));
        }
        rms = sqrt(rms / s.N);
        // rows bitwise across M (rounded, like the engine)
        std::vector<float> r1((size_t)MM * s.N);
        for (int m = 0; m < MM; ++m) {
            v41k::fp8_linear(y1, xq + (size_t)m * KX, xs + (size_t)m * KX / 32, Wb, Sb, 1, s.N, s.K, true, 0, s.G);
            CK(cudaMemcpy(r1.data() + (size_t)m * s.N, y1, s.N * 4, cudaMemcpyDeviceToHost));
        }
        long diff = 0;
        for (int M : {2, 3, 4, 5, 8, 9, 16, 17, 33, 64, 256}) {
            v41k::fp8_linear(yb, xq, xs, Wb, Sb, M, s.N, s.K, true, 0, s.G);
            std::vector<float> rm((size_t)M * s.N); CK(cudaMemcpy(rm.data(), yb, rm.size() * 4, cudaMemcpyDeviceToHost));
            diff += memcmp(rm.data(), r1.data(), rm.size() * 4) != 0;
        }
        const bool pass = diff == 0 && eb <= std::max(ea * 1.5, 1e-5 * rms);
        ok &= pass;
        printf("%-8s %5dx%5d  max|err|/rms old %.2e new %.2e  rows across M %s  %s\n", s.n, s.N, s.K, ea / rms, eb / rms,
               diff ? "DIFF" : "bitwise", pass ? "ok" : "FAIL");
        // L2-cold timing: rotate copies of both layouts until the set exceeds 3x the 33.6 MB L2
        const int NC = std::max(2, (int)((100u << 20) / (cb + sb)) + 1);
        std::vector<uint8_t*> ca(NC), sa(NC), cbv(NC), sbv(NC);
        for (int c = 0; c < NC; ++c) {
            CK(cudaMalloc(&ca[c], cb)); CK(cudaMalloc(&sa[c], sb)); CK(cudaMalloc(&cbv[c], cb)); CK(cudaMalloc(&sbv[c], sb));
            CK(cudaMemcpy(ca[c], hw.data(), cb, cudaMemcpyHostToDevice)); CK(cudaMemcpy(sa[c], hsc.data(), sb, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(cbv[c], hw.data(), cb, cudaMemcpyHostToDevice)); CK(cudaMemcpy(sbv[c], hsc.data(), sb, cudaMemcpyHostToDevice));
            v41k::overlay_register(ca[c], sa[c], g, s.N, s.K, 0); v41k::overlay_register(cbv[c], sbv[c], g, s.N, s.K, 1);
        }
        int rot = 0;
        for (int M : {1, 3, 8, 64, 256}) {
            const float ta = time_it([&] { const int c = rot++ % NC; v41k::fp8_linear(ya, xq, xs, ca[c], sa[c], M, s.N, s.K, true, 0, s.G); }, 60);
            const float tb = time_it([&] { const int c = rot++ % NC; v41k::fp8_linear(yb, xq, xs, cbv[c], sbv[c], M, s.N, s.K, true, 0, s.G); }, 60);
            printf("    M=%d  old %7.1f us (%5.0f GB/s)  new %7.1f us (%5.0f GB/s)  %.2fx\n", M, 1e3 * ta, (cb + sb) / ta / 1e6,
                   1e3 * tb, (cb + sb) / tb / 1e6, ta / tb);
        }
        for (int c = 0; c < NC; ++c) { cudaFree(ca[c]); cudaFree(sa[c]); cudaFree(cbv[c]); cudaFree(sbv[c]); }
        cudaFree(Wa); cudaFree(Sa); cudaFree(Wb); cudaFree(Sb); cudaFree(xq); cudaFree(xs); cudaFree(ya); cudaFree(yb); cudaFree(y1);
    }
    printf("%s\n", ok ? "GATE PASS" : "GATE FAIL");
    return ok ? 0 : 1;
}
