// gate_lpr: dump every dense decoder/shape the model runs (bf16, permuted bf16, FP8, NVFP4) at several M to a
// file; run under V41_DENSE_LPR=32 and =8 and compare with ops/zenith/cmp_lpr.py (reassociation-level only).
#include <cstdio>
#include <vector>
#include <random>
#include "v41_kernels.h"
#include "v41_config.h"
int main(int argc, char** argv) {
    v41k::v41k_init((size_t)1 << 24);
    FILE* f = fopen(argv[1], "wb");
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    auto bf = [](float v) { uint32_t u; memcpy(&u, &v, 4); return (uint16_t)((u + 0x7fff + ((u >> 16) & 1)) >> 16); };
    struct S { const char* n; int kind, N, K, G; };       // kind 0 bf16, 1 bf16 permuted, 2 fp8, 3 nvfp4
    std::vector<S> sh = {{"idx_wk", 0, 128, 512, 1}, {"idx_wproj", 0, 32, 5120, 1}, {"cmp_wkv", 0, 512, 5120, 1},
                         {"mk_head", 0, 4096, 256, 1}, {"head", 1, 4096, 5120, 1}, {"eg_wkv", 2, 512, 1280, 1},
                         {"wkv8", 2, 512, 5120, 1}, {"wq_b8", 2, 4096, 1280, 1}, {"idx_wq_b", 2, 4096, 1280, 1},
                         {"sh_w2_8", 2, 5120, 2304, 1}, {"wo_a8", 2, 8192, 4096, 8}, {"wq_a4", 3, 1280, 5120, 1},
                         {"wq_b4", 3, 4096, 1280, 1}, {"sh_w2_4", 3, 5120, 2304, 1}, {"idx4", 3, 4096, 1280, 1}};
    for (const S& s : sh)
        for (int M : {1, 2, 3, 5, 9, 16, 37}) {
            const int KK = s.K * s.G;
            std::vector<float> x((size_t)M * KK); for (auto& v : x) v = nd(rng);
            float *dx, *dy; cudaMalloc(&dx, x.size() * 4); cudaMalloc(&dy, (size_t)M * s.N * 4);
            cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
            uint8_t *xq, *xs8; float* xs; cudaMalloc(&xq, x.size()); cudaMalloc(&xs, x.size() / 32 * 4);
            v41k::act_quant_fp8(xq, xs, dx, M, KK, 0);
            if (s.kind <= 1) {
                std::vector<uint16_t> w((size_t)s.N * s.K); for (auto& v : w) v = bf(nd(rng) * 0.05f);
                uint16_t* dw; cudaMalloc(&dw, w.size() * 2); cudaMemcpy(dw, w.data(), w.size() * 2, cudaMemcpyHostToDevice);
                if (s.kind == 1) v41k::bf16_permute(dw, s.N, s.K, 0);
                v41k::bf16_linear(dy, dx, dw, M, s.N, s.K, false, 0, 1.f);
                cudaDeviceSynchronize(); cudaFree(dw);
            } else if (s.kind == 2) {
                std::vector<uint8_t> w((size_t)s.N * s.K); for (auto& v : w) { v = rng() & 0xff; if ((v & 0x7f) == 0x7f) v &= 0xfe; }
                std::vector<uint8_t> sc((size_t)((s.N + 31) / 32) * (s.K / 32)); for (auto& v : sc) v = 120 + rng() % 8;
                uint8_t *dw, *ds; cudaMalloc(&dw, w.size()); cudaMalloc(&ds, sc.size());
                cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice); cudaMemcpy(ds, sc.data(), sc.size(), cudaMemcpyHostToDevice);
                v41k::fp8_linear(dy, xq, xs, dw, ds, M, s.N, s.K, true, 0, s.G);
                cudaDeviceSynchronize(); cudaFree(dw); cudaFree(ds);
            } else {
                std::vector<uint8_t> w((size_t)s.N * s.K / 2); for (auto& v : w) v = rng() & 0xff;
                std::vector<uint8_t> sc((size_t)s.N * s.K / 16); for (auto& v : sc) v = 0x30 + rng() % 16;
                uint8_t *dw, *ds; cudaMalloc(&dw, w.size()); cudaMalloc(&ds, sc.size());
                cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice); cudaMemcpy(ds, sc.data(), sc.size(), cudaMemcpyHostToDevice);
                v41k::overlay_register(dw, ds, 0.01f, s.N, s.K, 0);
                v41k::fp8_linear(dy, xq, xs, dw, ds, M, s.N, s.K, true, 0, s.G);
                cudaDeviceSynchronize(); cudaFree(dw); cudaFree(ds);
            }
            std::vector<float> y((size_t)M * s.N); cudaMemcpy(y.data(), dy, y.size() * 4, cudaMemcpyDeviceToHost);
            const cudaError_t e = cudaGetLastError();
            int hdr[4] = {s.kind, s.N, s.K, M}; fwrite(hdr, 4, 4, f); fwrite(s.n, 1, 16, f); fwrite(y.data(), 4, y.size(), f);
            if (e != cudaSuccess) printf("%s M=%d: %s\n", s.n, M, cudaGetErrorString(e));
            cudaFree(dx); cudaFree(dy); cudaFree(xq); cudaFree(xs); (void)xs8;
        }
    fclose(f);
    puts("dumped");
}
