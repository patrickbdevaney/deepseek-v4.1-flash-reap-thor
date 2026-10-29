// bench_dense: M=1 rates of the decode-step dense shapes through the production entry points.
//   build/bench_dense            all shapes, GB/s of weight bytes
// Shapes: head bf16 129280x5120; NVFP4 wq_a 1280x5120, wq_b 32768x1280, wo_a 8192x4096 (G=8 over
// 8x512... served as K=4096 per group), wo_b 5120x8192, shared w1/w3 2304x5120, w2 5120x2304.
#include <cstdio>
#include <vector>
#include <random>
#include <cstring>
#include "v41_kernels.h"
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA %s line %d\n", cudaGetErrorString(e_), __LINE__); return 1; } } while (0)
static float time_it(auto f, int it = 30) {
    for (int w = 0; w < 3; ++w) f();
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a); for (int i = 0; i < it; ++i) f(); cudaEventRecord(b); cudaEventSynchronize(b);
    float ms; cudaEventElapsedTime(&ms, a, b); return ms / it;
}
int main() {
    v41k::v41k_init((size_t)1 << 22);
    std::mt19937 rng(1);
    auto fill = [&](void* d, size_t n) { std::vector<uint8_t> h(n); for (auto& b : h) b = rng() & 0x7f; return cudaMemcpy(d, h.data(), n, cudaMemcpyHostToDevice); };
    // head
    { const int N = 129280, K = 5120;
      uint16_t* W; float *x, *y; CK(cudaMalloc(&W, (size_t)N * K * 2)); CK(cudaMalloc(&x, K * 4)); CK(cudaMalloc(&y, N * 4));
      CK(fill(W, (size_t)N * K * 2)); CK(fill(x, K * 4));
      float ms = time_it([&] { v41k::bf16_linear(y, x, W, 1, N, K, false, 0, 1.f); }, 10);
      printf("head  bf16 %6dx%5d  %8.1f us  %6.1f GB/s\n", N, K, 1000 * ms, (double)N * K * 2 / ms / 1e6);
      std::vector<float> y0(N), y1(N); cudaMemcpy(y0.data(), y, N * 4, cudaMemcpyDeviceToHost);
      v41k::bf16_permute(W, N, K, 0);
      ms = time_it([&] { v41k::bf16_linear(y, x, W, 1, N, K, false, 0, 1.f); }, 10);
      cudaMemcpy(y1.data(), y, N * 4, cudaMemcpyDeviceToHost);
      printf("head  bf16P %5dx%5d  %8.1f us  %6.1f GB/s  bitwise %s\n", N, K, 1000 * ms, (double)N * K * 2 / ms / 1e6,
             memcmp(y0.data(), y1.data(), N * 4) ? "DIFF" : "same");
      cudaFree(W); }
    struct S { const char* n; int N, K, G; };
    for (S s : {S{"wq_a", 1280, 5120, 1}, S{"wq_b", 32768, 1280, 1}, S{"wo_a", 8192, 4096, 8}, S{"wo_b", 5120, 8192, 1},
                S{"sh_w1", 2304, 5120, 1}, S{"sh_w2", 5120, 2304, 1}}) {
        const size_t cb = (size_t)s.N * s.K / 2, sb = (size_t)s.N * s.K / 16;
        uint8_t *W, *Sc, *xq; float *xs, *y;
        CK(cudaMalloc(&W, cb)); CK(cudaMalloc(&Sc, sb)); CK(cudaMalloc(&xq, (size_t)s.K * s.G)); CK(cudaMalloc(&xs, s.K * s.G / 32 * 4));
        CK(cudaMalloc(&y, s.N * 4)); CK(fill(W, cb)); CK(fill(Sc, sb)); CK(fill(xq, (size_t)s.K * s.G));
        std::vector<float> hs(s.K * s.G / 32, 1.f); CK(cudaMemcpy(xs, hs.data(), hs.size() * 4, cudaMemcpyHostToDevice));
        v41k::overlay_register(W, 1.f);
        float ms = time_it([&] { v41k::fp8_linear(y, xq, xs, W, Sc, 1, s.N, s.K, true, 0, s.G); });
        printf("%-5s nvfp4 %6dx%5d  %8.1f us  %6.1f GB/s\n", s.n, s.N, s.K, 1000 * ms, (double)(cb + sb) / ms / 1e6);
        cudaFree(W); cudaFree(Sc); cudaFree(xq); cudaFree(xs); cudaFree(y);
    }
    return 0;
}
