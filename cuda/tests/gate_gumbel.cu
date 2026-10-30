// gate_gumbel.cu — S11: v41k::sample_gumbel draws from softmax(L / T). Chi-square of 200k draws over a
// synthetic 129280-token row whose mass sits on 40 tokens, at T = 0.7 and 1.0; plus seed determinism.
//   scripts/build_v41.sh gate_gumbel && build/gate_gumbel
#include "v41_kernels.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>
#include <cstdlib>

int main(int argc, char** argv) {
    const int N = 129280, HOT = 40, DRAWS = argc > 1 ? atoi(argv[1]) : 200000;
    const unsigned long long SEED = argc > 2 ? strtoull(argv[2], nullptr, 10) : 7;
    std::vector<float> L(N);
    std::mt19937 g(3); std::normal_distribution<float> nd(0.f, 1.f);
    for (int i = 0; i < N; ++i) L[i] = nd(g) - 14.f;               // tail: ~e^-14 each, ~10% of the mass at T = 1
    for (int j = 0; j < HOT; ++j) L[(j * 3203 + 17) % N] = 2.f - 0.15f * j;
    float* dL; int* dout; cudaMalloc(&dL, N * 4); cudaMalloc(&dout, 4);
    cudaMemcpy(dL, L.data(), N * 4, cudaMemcpyHostToDevice);
    bool ok = true;
    for (float T : {0.7f, 1.0f}) {
        double mx = -1e30; for (float v : L) mx = std::max(mx, (double)v);
        std::vector<double> p(N); double z = 0;
        for (int i = 0; i < N; ++i) { p[i] = std::exp((L[i] - mx) / T); z += p[i]; }
        for (auto& v : p) v /= z;
        std::vector<int> cnt(N, 0);
        std::mt19937_64 rng(SEED);
        for (int d = 0; d < DRAWS; ++d) {
            v41k::sample_gumbel(dout, dL, N, T, rng(), 0);
            int h; cudaMemcpy(&h, dout, 4, cudaMemcpyDeviceToHost); ++cnt[h];
        }
        // chi-square over the hot tokens + one pooled tail bin
        double chi = 0, tail_p = 1, tail_c = DRAWS; int df = 0;
        for (int j = 0; j < HOT; ++j) {
            const int i = (j * 3203 + 17) % N; const double e = p[i] * DRAWS;
            tail_p -= p[i]; tail_c -= cnt[i];
            if (e >= 5) { chi += (cnt[i] - e) * (cnt[i] - e) / e; ++df; }
        }
        const double et = tail_p * DRAWS; chi += (tail_c - et) * (tail_c - et) / et;
        // df ~ 40: chi^2 > 80 is p < 1e-4; also print the tail z-score
        printf("T=%.1f: chi^2 %.1f over %d bins (tail expected %.0f, got %.0f)  %s\n", T, chi, df + 1, et, tail_c, chi < 80 ? "ok" : "FAIL");
        ok = ok && chi < 80;
    }
    int a, b;
    v41k::sample_gumbel(dout, dL, N, 0.7f, 12345, 0); cudaMemcpy(&a, dout, 4, cudaMemcpyDeviceToHost);
    v41k::sample_gumbel(dout, dL, N, 0.7f, 12345, 0); cudaMemcpy(&b, dout, 4, cudaMemcpyDeviceToHost);
    printf("same seed same draw: %s\n", a == b ? "ok" : "FAIL"); ok = ok && a == b;
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
