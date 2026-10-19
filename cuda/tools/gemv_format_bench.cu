// gemv_format_bench.cu — weight-format GEMV microbench: achieved bytes/s of every expert format
// against a pure streaming read of the same bytes (the roofline a GEMV can at best reach).
//
//   scripts/build_v41.sh gemv_format_bench && build/gemv_format_bench [--mib 1536] [--iters 20] [--json out.json]
//
// Shapes are the expert passes: gate|up as one 4608 x 5120 GEMV and down 5120 x 2304, M = 1..4
// activation rows (decode, and a 4-token verify). Each format gets `--mib` of distinct weight copies
// cycled per launch so nothing is served from L2 (Thor L2 is far smaller than the default 1.5 GiB).
// Weights are random bytes with sane scales: GEMV speed does not depend on the values, only on the
// decode path, and the numbers are never checked here (tests/gate_v41_units.cu does that).
//
// Output per (format, shape, M): GB/s of weight bytes, % of the streaming read, and effective
// Gweights/s -- the number that decides decode tok/s once experts are the bound.
#include "../kernels/v41/v41_gemv.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e_)); exit(2); } } while (0)
using namespace v41g;

template <class Dec, int MT>
__global__ void __launch_bounds__(256) k_bench(float* out, const float* x, Dec dec, int N, int K) {
    extern __shared__ float4 xs4[];
    const int nch = K / 32;
    XSrc X; X.xf = x; X.ldx = K;
    for (int t = 0; t < MT; ++t) stage_row(xs4 + (size_t)t * nch * 8, X, t, 0, K);
    __syncthreads();
    const int row0 = blockIdx.x * ROWS_PER_CTA, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    float acc[ROWS_PER_WARP][MT] = {};
    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) {
        const int row = min(row0 + warp * ROWS_PER_WARP + r, N - 1);
        for (int cb = 0; cb < nch; cb += 32) {
            const int c = cb + lane, cc = min(c, nch - 1);
            float w[32]; dec.load(row, cc, w);
            float a[MT] = {};
            dot_chunk<MT>(a, w, xs4, cc, nch);
            if (c < nch) for (int t = 0; t < MT; ++t) acc[r][t] += a[t];
        }
    }
    for (int r = 0; r < ROWS_PER_WARP; ++r)
        for (int t = 0; t < MT; ++t) {
            const float v = warp_sum(acc[r][t]);
            const int row = row0 + warp * ROWS_PER_WARP + r;
            if (lane == 0 && row < N) out[(size_t)t * N + row] = v;
        }
}

__global__ void k_stream(const uint4* p, size_t n, unsigned* sink) {
    uint32_t a = 0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        const uint4 v = __ldg(p + i); a ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (a == 0x9e3779b9u) *sink = a;
}

struct Fmt { const char* name; double bytes_per_w; double extra_per_row_per_32; };

static float g_ms;
template <class F>
static double timeit(F f, int iters) {
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    f(); CK(cudaDeviceSynchronize());
    cudaEventRecord(a);
    for (int i = 0; i < iters; ++i) f();
    cudaEventRecord(b); CK(cudaEventSynchronize(b));
    cudaEventElapsedTime(&g_ms, a, b);
    return g_ms / iters;
}

int main(int argc, char** argv) {
    size_t mib = 1536; int iters = 20; std::string jout;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--mib") mib = strtoull(argv[++i], nullptr, 10);
        else if (a == "--iters") iters = atoi(argv[++i]);
        else if (a == "--json") jout = argv[++i];
    }
    const size_t pool = mib << 20;
    uint8_t* buf; CK(cudaMalloc(&buf, pool + 4096));
    {   // random bytes; then patch every 16-bit word into a finite fp16/e8m0 range by masking:
        // values do not matter for speed, NaN/denormal handling does not exist on these paths
        std::vector<uint8_t> h(1 << 24);
        uint32_t s = 12345;
        for (auto& b : h) { s = s * 1664525u + 1013904223u; b = (uint8_t)((s >> 24) & 0x3B); }
        for (size_t o = 0; o < pool; o += h.size()) CK(cudaMemcpy(buf + o, h.data(), std::min(h.size(), pool - o), cudaMemcpyHostToDevice));
    }
    float *x, *out; CK(cudaMalloc(&x, 4 * 5120 * 4)); CK(cudaMalloc(&out, 4 * 5120 * 4));
    CK(cudaMemset(x, 0, 4 * 5120 * 4));
    unsigned* sink; CK(cudaMalloc(&sink, 4));
    cudaFuncSetCacheConfig(k_stream, cudaFuncCachePreferL1);

    const double stream_ms = timeit([&] { k_stream<<<4096, 256>>>((const uint4*)buf, pool / 16, sink); }, iters);
    const double stream_gbs = pool / (stream_ms * 1e6);
    printf("streaming read: %.1f GB/s over %zu MiB\n\n", stream_gbs, mib);
    printf("%-14s %-12s %2s %10s %8s %7s %10s\n", "format", "shape", "M", "B/weight", "GB/s", "%strm", "Gweight/s");
    FILE* jf = jout.empty() ? nullptr : fopen(jout.c_str(), "w");
    if (jf) fprintf(jf, "{\"stream_gbs\": %.2f, \"rows\": [\n", stream_gbs);
    bool first = true;

    struct Shape { const char* nm; int N, K; } shapes[2] = {{"gate|up", 4608, 5120}, {"down", 5120, 2304}};
    auto run = [&](const char* nm, double bpw, auto make_dec, int MT_sel) {
        for (const auto& sh : shapes) {
            const size_t mat = ((size_t)(bpw * sh.N * sh.K) + 8191) & ~(size_t)4095;   // 4 KiB-aligned copies
            const int copies = (int)std::max<size_t>(1, pool / mat);
            int it = 0;
            const size_t smem = (size_t)MT_sel * sh.K * 4;
            auto launch = [&](auto mtc) {
                constexpr int MT = decltype(mtc)::value;
                auto dec = make_dec(buf + (size_t)(it++ % copies) * mat, sh.N, sh.K);
                k_bench<decltype(dec), MT><<<(sh.N + 63) / 64, 256, smem>>>(out, x, dec, sh.N, sh.K);
            };
            auto prep = [&](auto mtc) {
                constexpr int MT = decltype(mtc)::value;
                using D = decltype(make_dec(buf, sh.N, sh.K));
                cudaFuncSetAttribute(k_bench<D, MT>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
            };
            double ms;
            switch (MT_sel) {
            case 1: prep(std::integral_constant<int, 1>{}); ms = timeit([&] { launch(std::integral_constant<int, 1>{}); }, iters * 8); break;
            case 2: prep(std::integral_constant<int, 2>{}); ms = timeit([&] { launch(std::integral_constant<int, 2>{}); }, iters * 8); break;
            default: prep(std::integral_constant<int, 4>{}); ms = timeit([&] { launch(std::integral_constant<int, 4>{}); }, iters * 8); break;
            }
            CK(cudaGetLastError());
            const double bytes = bpw * sh.N * sh.K, gbs = bytes / (ms * 1e6);
            printf("%-14s %-12s %2d %10.4f %8.1f %6.1f%% %10.2f\n", nm, sh.nm, MT_sel, bpw, gbs, 100 * gbs / stream_gbs,
                   (double)sh.N * sh.K / (ms * 1e6));
            if (jf) fprintf(jf, "%s{\"format\": \"%s\", \"shape\": \"%s\", \"M\": %d, \"bpw_bytes\": %.5f, \"gbs\": %.2f, \"frac_stream\": %.4f, \"us\": %.2f}",
                            first ? "" : ",\n", nm, sh.nm, MT_sel, bpw, gbs, gbs / stream_gbs, ms * 1e3), first = false;
        }
    };
    for (int M : {1, 4}) {
        run("FP8 (dense)", 1.0 + 1.0 / 1024, [](uint8_t* p, int N, int K) { DecFP8 d; d.W = p; d.Ws = p + (size_t)N * K; d.K = K; return d; }, M);
        run("MXFP4 hw", 0.5 + 1.0 / 32, [](uint8_t* p, int N, int K) { DecMXFP4<true> d; d.W0 = d.W1 = p; d.S0 = d.S1 = p + (size_t)N * K / 2; d.K = K; d.split = 1 << 30; return d; }, M);
        run("MXFP4 sw", 0.5 + 1.0 / 32, [](uint8_t* p, int N, int K) { DecMXFP4<false> d; d.W0 = d.W1 = p; d.S0 = d.S1 = p + (size_t)N * K / 2; d.K = K; d.split = 1 << 30; return d; }, M);
        run("Q4_K", 144.0 / 256, [](uint8_t* p, int N, int K) { DecKQ<4> d; d.W0 = d.W1 = p; d.K = K; d.split = 1 << 30; return d; }, M);
        run("Q3_K", 110.0 / 256, [](uint8_t* p, int N, int K) { DecKQ<3> d; d.W0 = d.W1 = p; d.K = K; d.split = 1 << 30; return d; }, M);
        run("Q2_K", 84.0 / 256, [](uint8_t* p, int N, int K) { DecKQ<2> d; d.W0 = d.W1 = p; d.K = K; d.split = 1 << 30; return d; }, M);
        run("LUT-3 (prop.)", 3.0 / 8 + 1.0 / 32, [](uint8_t* p, int N, int K) { DecLUT<3> d; d.W0 = d.W1 = p; d.S0 = d.S1 = p + (size_t)N * K * 3 / 8; d.K = K; d.split = 1 << 30; d.lut0 = d.lut1 = 0.01f; return d; }, M);
        run("LUT-2 (prop.)", 2.0 / 8 + 1.0 / 32, [](uint8_t* p, int N, int K) { DecLUT<2> d; d.W0 = d.W1 = p; d.S0 = d.S1 = p + (size_t)N * K * 2 / 8; d.K = K; d.split = 1 << 30; d.lut0 = d.lut1 = 0.01f; return d; }, M);
        run("KT-3 (prop.)", 3.0 / 8 + 2.0 / 128, [](uint8_t* p, int N, int K) { DecKT<3> d; d.W0 = d.W1 = p; d.S0 = d.S1 = p + (size_t)N * K * 3 / 8; d.K = K; d.split = 1 << 30; return d; }, M);
        run("KT-2 (prop.)", 2.0 / 8 + 2.0 / 128, [](uint8_t* p, int N, int K) { DecKT<2> d; d.W0 = d.W1 = p; d.S0 = d.S1 = p + (size_t)N * K * 2 / 8; d.K = K; d.split = 1 << 30; return d; }, M);
        printf("\n");
    }
    if (jf) { fprintf(jf, "\n]}\n"); fclose(jf); }
    return 0;
}
