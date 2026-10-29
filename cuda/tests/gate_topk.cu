// gate_topk: topk_rows with block-scan search/emit (default) against the thread-0 scan form, exact.
// Rows with heavy ties (scores rounded to a few levels), n from 1 to 64K, k = 512; then timing.
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
#include "v41_kernels.h"
int main() {
    const int LD = 1 << 16, K = 512, M = 8;
    std::mt19937 rng(3);
    float* sc; int *pos, *o1, *o2;
    cudaMalloc(&sc, (size_t)M * LD * 4); cudaMalloc(&pos, M * 4); cudaMalloc(&o1, M * K * 4); cudaMalloc(&o2, M * K * 4);
    int fails = 0, cases = 0;
    for (int levels : {3, 50, 1 << 20})
    for (int n : {1, 7, 511, 512, 513, 1000, 4096, 30000, 65535}) {
        std::vector<float> h((size_t)M * LD);
        for (auto& v : h) v = (float)(int)(rng() % levels) * 0.37f - (levels > 1000 ? 1e5f : 3.f);
        std::vector<int> hp(M);
        for (int m = 0; m < M; ++m) hp[m] = (n - m % 2) * 4 + 3;      // (pos+1)/4 = n or n-1
        cudaMemcpy(sc, h.data(), h.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(pos, hp.data(), M * 4, cudaMemcpyHostToDevice);
        v41k::topk_serial_set(true);  v41k::topk_rows(o1, sc, pos, 4, M, LD, K, 0, nullptr, 0);
        v41k::topk_serial_set(false); v41k::topk_rows(o2, sc, pos, 4, M, LD, K, 0, nullptr, 0);
        std::vector<int> a(M * K), b(M * K);
        cudaMemcpy(a.data(), o1, a.size() * 4, cudaMemcpyDeviceToHost); cudaMemcpy(b.data(), o2, b.size() * 4, cudaMemcpyDeviceToHost);
        ++cases;
        if (memcmp(a.data(), b.data(), a.size() * 4)) { if (fails < 5) printf("MISMATCH levels %d n %d\n", levels, n); ++fails; }
    }
    printf("[gate_topk] %d/%d bit-identical -> %s\n", cases - fails, cases, fails ? "FAIL" : "PASS");
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    for (int n : {100, 30000}) {
        std::vector<int> hp(1, n * 4 + 3); cudaMemcpy(pos, hp.data(), 4, cudaMemcpyHostToDevice);
        for (int ser : {1, 0}) {
            v41k::topk_serial_set(ser);
            cudaEventRecord(e0);
            for (int i = 0; i < 50; ++i) v41k::topk_rows(o1, sc, pos, 4, 1, LD, K, 0, nullptr, 0);
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            printf("[gate_topk] n=%d %s %.1f us\n", n, ser ? "serial  " : "parallel", 1000 * ms / 50);
        }
    }
    return fails ? 1 : 0;
}
