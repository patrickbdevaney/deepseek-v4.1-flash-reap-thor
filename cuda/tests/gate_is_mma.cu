// gate_is_mma: k_index_mma (v41_is_mma.cuh) vs k_index_scores on fp4_sim'd queries and keys.
// PASS: every score bit-identical (or, where not, |diff| within one bf16 ulp and counted), -inf masks equal,
// and timing at prefill shapes (M = 256 rows x ld keys).
#include "v41_kernels.h"
#include "v41_config.h"
#include "../kernels/v41/v41_is_mma.cuh"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
#define CU(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA %s at %d\n", cudaGetErrorString(e_), __LINE__); exit(1); } } while (0)
using namespace v41;

static bool run(int M, int T, int ratio, bool use_cand, std::mt19937& rng, bool bench) {
    const int NH = INDEX_N_HEADS, HD = INDEX_HEAD_DIM, ld = T, ldc = (ld + CAND_BLOCK - 1) / CAND_BLOCK;
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> hq((size_t)M * NH * HD), hk((size_t)T * HD), hw((size_t)M * NH);
    for (auto& v : hq) v = nd(rng);
    for (size_t i = 0; i < hk.size(); ++i) hk[i] = nd(rng) * (1.f + 3.f * ((i / 32) % 3));   // scale spread across chunks
    for (auto& v : hw) v = nd(rng) * 0.05f;
    for (auto& v : hw) { uint32_t u; memcpy(&u, &v, 4); u &= 0xFFFF0000u; memcpy(&v, &u, 4); }   // bf16 values
    std::vector<int> pos(M);
    for (int m = 0; m < M; ++m) pos[m] = (T * ratio) - 1 - (M - 1 - m) * 3;                     // causal spread
    std::vector<uint8_t> cand((size_t)M * ldc);
    for (auto& c : cand) c = (rng() % 3) != 0;
    float *dq, *dk, *dw, *s0, *s1; uint8_t *qc, *qs, *kc, *ks, *dc; int* dp;
    CU(cudaMalloc(&dq, hq.size() * 4)); CU(cudaMalloc(&dk, hk.size() * 4)); CU(cudaMalloc(&dw, hw.size() * 4));
    CU(cudaMalloc(&qc, hq.size() / 2)); CU(cudaMalloc(&qs, hq.size() / 32)); CU(cudaMalloc(&kc, hk.size() / 2)); CU(cudaMalloc(&ks, hk.size() / 32));
    CU(cudaMalloc(&s0, (size_t)M * ld * 4)); CU(cudaMalloc(&s1, (size_t)M * ld * 4)); CU(cudaMalloc(&dp, M * 4)); CU(cudaMalloc(&dc, cand.size()));
    CU(cudaMemcpy(dq, hq.data(), hq.size() * 4, cudaMemcpyHostToDevice)); CU(cudaMemcpy(dk, hk.data(), hk.size() * 4, cudaMemcpyHostToDevice));
    CU(cudaMemcpy(dw, hw.data(), hw.size() * 4, cudaMemcpyHostToDevice)); CU(cudaMemcpy(dp, pos.data(), M * 4, cudaMemcpyHostToDevice));
    CU(cudaMemcpy(dc, cand.data(), cand.size(), cudaMemcpyHostToDevice));
    v41k::fp4_sim(dq, M * NH, HD, IDX_FP4_GRP, false, qc, qs, 0);
    v41k::fp4_sim(dk, T, HD, IDX_FP4_GRP, false, kc, ks, 0);
    const uint8_t* cp = use_cand ? dc : nullptr;
    auto ref = [&] { v41k::index_scores(s0, dq, dw, kc, ks, cp, dp, ratio, M, ld, 0); };
    dim3 g((M + v41g::IM_ROWS - 1) / v41g::IM_ROWS, (ld + v41g::IM_KEYS - 1) / v41g::IM_KEYS);
    auto mma = [&] { v41g::k_index_mma<INDEX_N_HEADS, INDEX_HEAD_DIM><<<g, 256>>>(s1, qc, qs, dw, kc, ks, cp, dp, ratio, M, ld, ldc, CAND_BLOCK); };
    ref(); mma(); CU(cudaGetLastError()); CU(cudaDeviceSynchronize());
    std::vector<float> a((size_t)M * ld), b(a.size());
    CU(cudaMemcpy(a.data(), s0, a.size() * 4, cudaMemcpyDeviceToHost)); CU(cudaMemcpy(b.data(), s1, b.size() * 4, cudaMemcpyDeviceToHost));
    long diff = 0, inf_bad = 0, big = 0, fin = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::isinf(a[i]) != std::isinf(b[i])) { ++inf_bad; continue; }
        if (std::isinf(a[i])) continue;
        ++fin;
        if (memcmp(&a[i], &b[i], 4)) { ++diff; if (std::fabs(a[i] - b[i]) > std::fabs(a[i]) / 128.f + 1e-30f) ++big; }
    }
    printf("M=%3d T=%5d cand=%d: %ld finite, %ld differ (%ld beyond 1 bf16 ulp), %ld -inf mismatches %s\n", M, T, use_cand, fin,
           diff, big, inf_bad, inf_bad || big ? "FAIL" : "ok");
    if (bench) {
        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1); float t0 = 0, t1 = 0;
        for (int w = 0; w < 2; ++w) {
            cudaEventRecord(e0); for (int i = 0; i < 5; ++i) ref(); cudaEventRecord(e1); CU(cudaEventSynchronize(e1)); cudaEventElapsedTime(&t0, e0, e1);
            cudaEventRecord(e0); for (int i = 0; i < 5; ++i) mma(); cudaEventRecord(e1); CU(cudaEventSynchronize(e1)); cudaEventElapsedTime(&t1, e0, e1);
        }
        printf("    timing: k_index_scores %.2f ms  k_index_mma %.2f ms  %.2fx\n", t0 / 5, t1 / 5, t0 / t1);
    }
    cudaFree(dq); cudaFree(dk); cudaFree(dw); cudaFree(qc); cudaFree(qs); cudaFree(kc); cudaFree(ks);
    cudaFree(s0); cudaFree(s1); cudaFree(dp); cudaFree(dc);
    return !inf_bad && !big;
}

int main() {
    std::mt19937 rng(7);
    bool ok = true;
    ok &= run(1, 300, 4, false, rng, false);
    ok &= run(3, 777, 4, true, rng, false);
    ok &= run(17, 1024, 4, true, rng, false);
    ok &= run(256, 2048, 4, false, rng, true);
    ok &= run(256, 2048, 4, true, rng, true);
    printf("%s\n", ok ? "GATE PASS" : "GATE FAIL");
    return ok ? 0 : 1;
}
