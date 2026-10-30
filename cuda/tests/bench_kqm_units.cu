// bench_kqm_units.cu — prefill expert pass cost vs how many units share an expert (P8 stage 2 sizing).
// Random repacked Q2_K/Q3_K bytes (timing only). Units of 16 tokens; R units per expert, adjacent in the grid.
//   build/bench_kqm_units
#include "v41_kernels.h"
#include "v41_config.h"
#include "v41_kq_fast.cuh"
#include "v41_kq_mma.cuh"
#include <cstdio>
#include <vector>
#include <random>
using namespace v41g;
struct Unit { int e, start, cnt; };
#define CU(x) do { cudaError_t e_ = (x); if (e_) { printf("CUDA %s @%d\n", cudaGetErrorString(e_), __LINE__); exit(1); } } while (0)
template <int QT, int MT> static float bench(int pass, int R, int TOT) {
    const int N = pass == 0 ? 2 * v41::MOE_INTER : v41::DIM, K = pass == 0 ? v41::DIM : v41::MOE_INTER;
    const int E = TOT / R;                                    // distinct experts
    const size_t bpe = (size_t)(pass == 0 ? v41::MOE_INTER : v41::DIM) * (K / 256) * (QT == 2 ? 84 : 110);
    std::mt19937 g(1);
    std::vector<uint8_t> h(bpe); for (auto& b : h) b = g() & 0x3f;   // small fp16 d bits: finite
    std::vector<const uint8_t*> w1(E), w3(E), w2(E);
    for (int e = 0; e < E; ++e) {
        uint8_t* d; CU(cudaMalloc(&d, bpe)); CU(cudaMemcpy(d, h.data(), bpe, cudaMemcpyHostToDevice)); w1[e] = d; w2[e] = d;
        CU(cudaMalloc(&d, bpe)); CU(cudaMemcpy(d, h.data(), bpe, cudaMemcpyHostToDevice)); w3[e] = d;
    }
    const uint8_t **t1, **t3, **t2;
    CU(cudaMalloc(&t1, E * 8)); CU(cudaMalloc(&t3, E * 8)); CU(cudaMalloc(&t2, E * 8));
    CU(cudaMemcpy(t1, w1.data(), E * 8, cudaMemcpyHostToDevice)); CU(cudaMemcpy(t3, w3.data(), E * 8, cudaMemcpyHostToDevice));
    CU(cudaMemcpy(t2, w2.data(), E * 8, cudaMemcpyHostToDevice));
    KQTab tab{t1, t3, t2, pass};
    const int P = TOT * 16;
    uint8_t* dx; float* dxs; CU(cudaMalloc(&dx, (size_t)P * K)); CU(cudaMemset(dx, 0x30, (size_t)P * K));
    CU(cudaMalloc(&dxs, (size_t)P * K / 32 * 4)); CU(cudaMemset(dxs, 0, (size_t)P * K / 32 * 4));
    XSrc X{}; X.xq = dx; X.xs = dxs; X.ldx = K;
    std::vector<Unit> hu; for (int e = 0; e < E; ++e) for (int r = 0; r < R; ++r) hu.push_back(Unit{e, (int)hu.size() * 16, 16});
    std::vector<Unit> hm; for (const Unit& u : hu) for (int s = 0; s < 16; s += MT) hm.push_back(Unit{u.e, u.start + s, MT});
    const int nu = (int)hm.size();
    Unit* du; int* dnu; int* ptok; CU(cudaMalloc(&du, nu * sizeof(Unit))); CU(cudaMalloc(&dnu, 4)); CU(cudaMalloc(&ptok, P * 4));
    std::vector<int> id(P); for (int i = 0; i < P; ++i) id[i] = i;
    CU(cudaMemcpy(ptok, id.data(), P * 4, cudaMemcpyHostToDevice));
    CU(cudaMemcpy(du, hm.data(), nu * sizeof(Unit), cudaMemcpyHostToDevice)); CU(cudaMemcpy(dnu, &nu, 4, cudaMemcpyHostToDevice));
    float* out; CU(cudaMalloc(&out, (size_t)P * N * 4));
    dim3 grid((N + KQM_ROWS_PER_CTA - 1) / KQM_ROWS_PER_CTA, nu);
    const size_t sm = (size_t)MT * kqm_stage_bytes(K);
    auto kf = k_kqm<QT, MT, Unit>;
    CU(cudaFuncSetAttribute(kf, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm));
    for (int i = 0; i < 2; ++i) kf<<<grid, 256, sm>>>(out, N, X, true, du, dnu, ptok, tab, N, K, 1, 1 << 30);
    CU(cudaDeviceSynchronize());
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a);
    for (int i = 0; i < 5; ++i) kf<<<grid, 256, sm>>>(out, N, X, true, du, dnu, ptok, tab, N, K, 1, 1 << 30);
    cudaEventRecord(b); CU(cudaEventSynchronize(b));
    float ms; cudaEventElapsedTime(&ms, a, b); ms /= 5;
    const double ub = (double)E * bpe * (pass == 0 ? 2 : 1);
    printf("Q%d pass %d MT %2d  %3d experts x %d units of 16: %.3f ms  %.1f us/unit16  unique %.0f MB -> %.0f GB/s unique, %.0f GB/s if re-read\n",
           QT, pass, MT, E, R, ms, 1e3 * ms / TOT, ub / 1e6, ub / ms / 1e6, ub * R / ms / 1e6);
    for (int e = 0; e < E; ++e) { cudaFree((void*)w1[e]); cudaFree((void*)w3[e]); }
    cudaFree(t1); cudaFree(t3); cudaFree(t2); cudaFree(dx); cudaFree(dxs); cudaFree(du); cudaFree(dnu); cudaFree(ptok); cudaFree(out);
    return ms;
}
int main() {
    for (int pass : {0, 1})
        for (int R : {1, 2, 4, 8}) { bench<2, 16>(pass, R, 96); }
    for (int R : {1, 4}) bench<2, 8>(0, R, 96);
    for (int R : {1, 4}) bench<3, 16>(0, R, 96);
    return 0;
}
