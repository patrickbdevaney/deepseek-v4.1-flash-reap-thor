// gate_sa_split: the split sparse attention (k_sa_gather/scores/out) against k_sparse_attn_c, bitwise.
// Random q / ring / compressed rows; M = 1..8; positions before the window fills and long after; a
// third of the compressed indices -1; ncomp 0 / 64 / 512. Then timing for both at M=1 and M=6.
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
#include "v41_kernels.h"
#include "v41_config.h"
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA %s line %d\n", cudaGetErrorString(e_), __LINE__); return 1; } } while (0)
using namespace v41;
int main() {
    const int RING = 256, NROWS = 4096, MMAX = 8;
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    auto randv = [&](size_t n, float s) { std::vector<float> v(n); for (auto& x : v) x = nd(rng) * s; return v; };
    std::vector<float> hq = randv((size_t)MMAX * N_HEADS * HEAD_DIM, 1.f), hr = randv((size_t)RING * HEAD_DIM, 1.f);
    std::vector<float> hsink = randv(N_HEADS, 1.f);
    std::vector<uint8_t> hcc((size_t)NROWS * 256), hcs((size_t)NROWS * 32);
    for (auto& b : hcc) b = rng() & 0xff;
    for (auto& b : hcs) b = 0x30 + (rng() % 0x20);          // modest positive e4m3 scales
    float *q, *ring, *sink, *o1, *o2; uint8_t *cc, *cs; int *cidx, *pos;
    CK(cudaMalloc(&q, hq.size() * 4)); CK(cudaMalloc(&ring, hr.size() * 4)); CK(cudaMalloc(&sink, N_HEADS * 4));
    CK(cudaMalloc(&cc, hcc.size())); CK(cudaMalloc(&cs, hcs.size()));
    CK(cudaMalloc(&cidx, (size_t)MMAX * INDEX_TOPK * 4)); CK(cudaMalloc(&pos, MMAX * 4));
    const size_t on = (size_t)MMAX * N_HEADS * HEAD_DIM;
    CK(cudaMalloc(&o1, on * 4)); CK(cudaMalloc(&o2, on * 4));
    CK(cudaMemcpy(q, hq.data(), hq.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(ring, hr.data(), hr.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(sink, hsink.data(), N_HEADS * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(cc, hcc.data(), hcc.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(cs, hcs.data(), hcs.size(), cudaMemcpyHostToDevice));
    v41k::sparse_attn_reserve(MMAX);
    const float scale = 1.f / sqrtf((float)HEAD_DIM);
    int fails = 0, cases = 0;
    for (int ncomp : {0, 64, 512})
    for (int p0 : {0, 5, 126, 127, 300, 70000})
    for (int M = 1; M <= MMAX; ++M) {
        std::vector<int> hp(M), hi((size_t)M * INDEX_TOPK);
        for (int m = 0; m < M; ++m) hp[m] = p0 + m;
        for (auto& x : hi) x = (rng() % 3 == 0) ? -1 : (int)(rng() % NROWS);
        CK(cudaMemcpy(pos, hp.data(), M * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(cidx, hi.data(), hi.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemset(o1, 0xff, on * 4)); CK(cudaMemset(o2, 0xff, on * 4));
        v41k::sparse_attn_split_set(false);
        v41k::sparse_attn(o1, q, ring, RING, cc, cs, ncomp ? cidx : nullptr, ncomp, 0, sink, pos, M, scale, 0);
        v41k::sparse_attn_split_set(true);
        v41k::sparse_attn(o2, q, ring, RING, cc, cs, ncomp ? cidx : nullptr, ncomp, 0, sink, pos, M, scale, 0);
        CK(cudaDeviceSynchronize());
        std::vector<float> a((size_t)M * N_HEADS * HEAD_DIM), b(a.size());
        CK(cudaMemcpy(a.data(), o1, a.size() * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(b.data(), o2, b.size() * 4, cudaMemcpyDeviceToHost));
        ++cases;
        if (memcmp(a.data(), b.data(), a.size() * 4)) {
            size_t i = 0; while (i < a.size() && !memcmp(&a[i], &b[i], 4)) ++i;
            if (fails < 5) printf("MISMATCH ncomp %d pos %d M %d at %zu: %.9g vs %.9g\n", ncomp, p0, M, i, a[i], b[i]);
            ++fails;
        }
    }
    printf("[gate_sa_split] %d/%d cases bit-identical -> %s\n", cases - fails, cases, fails ? "FAIL" : "PASS");
    // timing at long context (all 640 entries live)
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    for (int M : {1, 6}) {
        std::vector<int> hp(M), hi((size_t)M * INDEX_TOPK);
        for (int m = 0; m < M; ++m) hp[m] = 70000 + m;
        for (auto& x : hi) x = (int)(rng() % NROWS);
        cudaMemcpy(pos, hp.data(), M * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(cidx, hi.data(), hi.size() * 4, cudaMemcpyHostToDevice);
        for (int sp : {0, 1}) {
            v41k::sparse_attn_split_set(sp);
            for (int w = 0; w < 5; ++w) v41k::sparse_attn(o1, q, ring, RING, cc, cs, cidx, 512, 0, sink, pos, M, scale, 0);
            cudaEventRecord(e0);
            for (int it = 0; it < 50; ++it) v41k::sparse_attn(o1, q, ring, RING, cc, cs, cidx, 512, 0, sink, pos, M, scale, 0);
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            printf("[gate_sa_split] M=%d ne=640 %s: %.1f us/call\n", M, sp ? "split  " : "compact", 1000.f * ms / 50);
        }
    }
    return fails ? 1 : 0;
}
