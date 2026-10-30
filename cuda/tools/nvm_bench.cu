// nvm_bench.cu — A10: k_nvm (register double-buffered) vs k_nvt (TMA smem ring) on the V4.1 NVFP4 overlay shapes.
//   scripts/build_v41.sh nvm_bench && build/nvm_bench [--iters 50] [--want 3] [--mib 768]
// Per shape: the launcher's K split (want x SMs CTAs, >= 4 superblocks each), M = 1 and 4, weights cycled over
// --mib of copies so nothing is served from L2. Prints us/call, GB/s of weight bytes, and whether k_nvt's
// output is bit-identical to k_nvm's (it must be: same split, same per-lane arithmetic).
#include "v41_config.h"
#include "../kernels/v41/v41_gemv.cuh"
#include "../kernels/v41/v41_nv_mma.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e_)); exit(2); } } while (0)
using namespace v41g;

struct Shape { const char* name; int N, K, calls; };   // calls per layer per step

__global__ void k_fill(uint8_t* p, size_t n, uint32_t seed, uint8_t mask, uint8_t orv) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        uint32_t h = (uint32_t)i * 2654435761u ^ seed; h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
        p[i] = (uint8_t)((h & mask) | orv);
    }
}
static void fill(uint8_t* p, size_t n, uint32_t seed, uint8_t mask, uint8_t orv) { k_fill<<<160, 256>>>(p, n, seed, mask, orv); CK(cudaGetLastError()); }

int main(int argc, char** argv) {
    int iters = 50, want = 3; size_t mib = 768; int sms = 0;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--iters")) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--want")) want = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--mib")) mib = atol(argv[++i]);
    }
    CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    const Shape shapes[] = {
        {"wo_b",  5120, 8192, 1}, {"wq_b", 32768, 1280, 1}, {"wo_a", 8192, 4096, 1}, {"sh_w1", 2304, 5120, 2},
        {"sh_w2", 5120, 2304, 1}, {"wq_a",  1280, 5120, 1}, {"wkv",   512, 5120, 1},
    };
    const int Mmax = 4;
    uint8_t *xq; float *xs, *y0, *y1, *part; int* kcnt;
    CK(cudaMalloc(&xq, (size_t)Mmax * 8192)); CK(cudaMalloc(&xs, (size_t)Mmax * 8192 / 32 * 4));
    fill(xq, (size_t)Mmax * 8192, 7, 0xBF, 0);                           // e4m3, |x| < 2^? and never NaN (bit 6 clear)
    { std::vector<float> h((size_t)Mmax * 8192 / 32, 0.03125f); CK(cudaMemcpy(xs, h.data(), h.size() * 4, cudaMemcpyHostToDevice)); }
    CK(cudaMalloc(&y0, (size_t)Mmax * 32768 * 4)); CK(cudaMalloc(&y1, (size_t)Mmax * 32768 * 4));
    CK(cudaMalloc(&part, (size_t)16 * 8 * 32768 * 4)); CK(cudaMalloc(&kcnt, 4096 * 4)); CK(cudaMemset(kcnt, 0, 4096 * 4));
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    double tot_nvm = 0, tot_best = 0;
    printf("SMs %d  want %d  iters %d  copies over %zu MiB\n", sms, want, iters, mib);
    for (const Shape& sh : shapes) {
        const int N = sh.N, K = sh.K, nsb = K / 256, ntile = (N + 127) / 128;
        const size_t cb = (size_t)N * K / 2, sbb = (size_t)N * K / 16, wb = cb + sbb;
        const int ncopy = std::max<size_t>(2, (mib << 20) / wb);
        uint8_t *Wc, *Sc;
        CK(cudaMalloc(&Wc, cb * ncopy)); CK(cudaMalloc(&Sc, sbb * ncopy));
        fill(Wc, cb * ncopy, 11, 0xFF, 0); fill(Sc, sbb * ncopy, 13, 0x07, 0x30);   // scales e4m3 in [0.5, 0.94]
        int ks = 1;
        if (ntile < want * sms) ks = std::min((want * sms + ntile - 1) / ntile, std::max(1, nsb / 4));
        const int nk_max = ((nsb + ks - 1) / ks) * 256;
        const size_t SB = (nvm_stage_bytes(nk_max) + 15) & ~(size_t)15;
        for (int M : {1, 4}) {
            const int mt = M == 1 ? 1 : 4;
            XSrc X; X.xq = xq; X.xs = xs; X.ldx = K;
            dim3 grid(ntile, (M + mt - 1) / mt, ks);
            int* kc = ks > 1 ? kcnt : nullptr;
            auto run_nvm = [&](int c, float* y) {
                const size_t smem = (size_t)mt * SB;
                if (mt == 1) { auto kf = k_nvm<1>; CK(cudaFuncSetAttribute(kf, cudaFuncAttributeMaxDynamicSharedMemorySize, 160 * 1024));
                               kf<<<grid, 256, smem>>>(y, X, Wc + c * cb, Sc + c * sbb, 0.5f, M, N, K, N, ks, 0, part, kc); }
                else { auto kf = k_nvm<4>; CK(cudaFuncSetAttribute(kf, cudaFuncAttributeMaxDynamicSharedMemorySize, 160 * 1024));
                       kf<<<grid, 256, smem>>>(y, X, Wc + c * cb, Sc + c * sbb, 0.5f, M, N, K, N, ks, 0, part, kc); }
            };
            struct V { int ns, ch; bool nc; };
            static const V vars[] = {{2,1,false},{4,1,false},{2,2,false},{3,2,false},{2,4,false},{4,2,false},{2,2,true},{4,2,true},{2,4,true}};
            auto run_nvt = [&](int v, int c, float* y) {
                const V& q = vars[v];
                const size_t smem = nvt_ring_bytes(q.ns, q.ch) + (size_t)mt * SB;
                #define NVT(MTV, NSV, CHV, NCV) if (q.ns == NSV && q.ch == CHV && q.nc == NCV) { auto kf = k_nvt<MTV, NSV, CHV, NCV>; \
                    CK(cudaFuncSetAttribute(kf, cudaFuncAttributeMaxDynamicSharedMemorySize, 200 * 1024)); \
                    kf<<<grid, 256, smem>>>(y, X, Wc + c * cb, Sc + c * sbb, 0.5f, M, N, K, N, ks, 0, part, kc); return; }
                #define NVTM(MTV) NVT(MTV,2,1,false) NVT(MTV,4,1,false) NVT(MTV,2,2,false) NVT(MTV,3,2,false) NVT(MTV,2,4,false) NVT(MTV,4,2,false) \
                    NVT(MTV,2,2,true) NVT(MTV,4,2,true) NVT(MTV,2,4,true)
                if (mt == 1) { NVTM(1) } else { NVTM(4) }
                #undef NVT
                #undef NVTM
            };
            auto time = [&](auto&& f) {
                for (int c = 0; c < ncopy; ++c) f(c);
                CK(cudaDeviceSynchronize());
                cudaEventRecord(a);
                for (int it = 0; it < iters; ++it) f(it % ncopy);
                cudaEventRecord(b); CK(cudaEventSynchronize(b)); CK(cudaGetLastError());
                float ms; cudaEventElapsedTime(&ms, a, b); return ms * 1000.0 / iters;
            };
            const double t0 = time([&](int c) { run_nvm(c, y0); });
            printf("%-6s N=%5d K=%5d ks=%d M=%d  k_nvm  %7.1f us %6.1f GB/s\n", sh.name, N, K, ks, M, t0, wb / t0 / 1e3);
            double best = t0;
            for (int v = 0; v < (int)(sizeof vars / sizeof vars[0]); ++v) {
                const int ns = vars[v].ns;
                if (nvt_ring_bytes(ns, vars[v].ch) + (size_t)mt * SB > 200 * 1024) continue;
                const double t1 = time([&](int c) { run_nvt(v, c, y1); });
                // bitwise check on copy 0
                run_nvm(0, y0); run_nvt(v, 0, y1); CK(cudaDeviceSynchronize());
                std::vector<float> h0((size_t)M * N), h1((size_t)M * N);
                CK(cudaMemcpy(h0.data(), y0, h0.size() * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(h1.data(), y1, h1.size() * 4, cudaMemcpyDeviceToHost));
                const bool same = !memcmp(h0.data(), h1.data(), h0.size() * 4);
                printf("%-6s %35s k_nvt S%d C%d%s %7.1f us %6.1f GB/s  %.2fx  %s\n", "", "", ns, vars[v].ch, vars[v].nc ? " NOCOMP" : "", t1, wb / t1 / 1e3, t0 / t1,
                       vars[v].nc ? "-" : same ? "bitwise" : "DIFFERS");
                if (!vars[v].nc) best = std::min(best, t1);
            }
            if (M == 1) { tot_nvm += t0 * sh.calls; tot_best += best * sh.calls; }
        }
        CK(cudaFree(Wc)); CK(cudaFree(Sc));
    }
    printf("per layer M=1: k_nvm %.1f us, best k_nvt %.1f us; x40 layers: %.2f -> %.2f ms/step\n", tot_nvm, tot_best, tot_nvm * 40e-3, tot_best * 40e-3);
    return 0;
}
