// gate_cnq_cl.cu — A9: collapse_norm_quant on the 8-CTA cluster (V41_CNQ_CL=1, default) vs k_cnq (=0).
// Writes y, q, qs for M = 1..8 rows of random stream data to argv[1]; run once per setting and cmp the files.
//   V41_CNQ_CL=0 build/gate_cnq_cl a.bin && V41_CNQ_CL=1 build/gate_cnq_cl b.bin && cmp a.bin b.bin
// argv[2] = "time": also the mean latency of M=1 over 2000 graph-free launches.
#include "v41_kernels.h"
#include "v41_config.h"
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: gate_cnq_cl OUT [time]\n"); return 2; }
    const int D = v41::DIM, H = v41::HC_DIM, MM = 8;
    std::mt19937 g(5); std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x((size_t)MM * H), pre(MM * 4); std::vector<uint16_t> w(D);
    for (auto& v : x) { float f = nd(g) * 3.f; uint32_t u; memcpy(&u, &f, 4); u &= 0xffff0000u; memcpy(&v, &u, 4); }
    for (auto& v : pre) v = nd(g) * 0.5f;
    for (auto& v : w) { float f = 1.f + 0.1f * nd(g); uint32_t u; memcpy(&u, &f, 4); v = (uint16_t)(u >> 16); }
    float *dx, *dp, *dy, *dqs; uint16_t* dw; uint8_t* dq;
    cudaMalloc(&dx, x.size() * 4); cudaMalloc(&dp, pre.size() * 4); cudaMalloc(&dw, D * 2);
    cudaMalloc(&dy, (size_t)MM * D * 4); cudaMalloc(&dq, (size_t)MM * D); cudaMalloc(&dqs, (size_t)MM * D / 32 * 4);
    cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dp, pre.data(), pre.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dw, w.data(), D * 2, cudaMemcpyHostToDevice);
    FILE* f = fopen(argv[1], "wb");
    std::vector<uint8_t> buf((size_t)MM * D * 6);
    for (int M = 1; M <= MM; ++M) {
        cudaMemset(dy, 0, (size_t)MM * D * 4); cudaMemset(dq, 0, (size_t)MM * D); cudaMemset(dqs, 0, (size_t)MM * D / 8);
        v41k::collapse_norm_quant(dy, dq, dqs, dx, dp, dw, M, 1e-6f, 0);
        if (cudaDeviceSynchronize() != cudaSuccess) { fprintf(stderr, "kernel failed: %s\n", cudaGetErrorString(cudaGetLastError())); return 1; }
        cudaMemcpy(buf.data(), dy, (size_t)M * D * 4, cudaMemcpyDeviceToHost); fwrite(buf.data(), 1, (size_t)M * D * 4, f);
        cudaMemcpy(buf.data(), dq, (size_t)M * D, cudaMemcpyDeviceToHost); fwrite(buf.data(), 1, (size_t)M * D, f);
        cudaMemcpy(buf.data(), dqs, (size_t)M * D / 8, cudaMemcpyDeviceToHost); fwrite(buf.data(), 1, (size_t)M * D / 8, f);
    }
    fclose(f);
    {   // hc_post + collapse_norm_quant vs the fused hc_post_cnq, in-process, every M
        std::vector<float> fr((size_t)MM * D), po(MM * 4), cb(MM * 16);
        for (auto& v : fr) v = nd(g); for (auto& v : po) v = nd(g); for (auto& v : cb) v = nd(g) * 0.5f;
        float *df, *dpo, *dcb, *dh1, *dh2, *dy2, *dqs2; uint8_t* dq2;
        cudaMalloc(&df, fr.size() * 4); cudaMalloc(&dpo, po.size() * 4); cudaMalloc(&dcb, cb.size() * 4);
        cudaMalloc(&dh1, (size_t)MM * H * 4); cudaMalloc(&dh2, (size_t)MM * H * 4);
        cudaMalloc(&dy2, (size_t)MM * D * 4); cudaMalloc(&dq2, (size_t)MM * D); cudaMalloc(&dqs2, (size_t)MM * D / 8);
        cudaMemcpy(df, fr.data(), fr.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dpo, po.data(), po.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dcb, cb.data(), cb.size() * 4, cudaMemcpyHostToDevice);
        bool ok = true;
        for (int M = 1; M <= MM && v41k::post_cnq_ok(M); ++M) {
            v41k::hc_post(dh1, df, dx, dpo, dcb, M, 0);
            v41k::collapse_norm_quant(dy, dq, dqs, dh1, dp, dw, M, 1e-6f, 0);
            v41k::hc_post_cnq(dh2, df, dx, dpo, dcb, dy2, dq2, dqs2, dp, dw, M, 1e-6f, 0);
            cudaDeviceSynchronize();
            auto same = [&](const void* a, const void* b, size_t n) {
                std::vector<uint8_t> x1(n), x2(n); cudaMemcpy(x1.data(), a, n, cudaMemcpyDeviceToHost); cudaMemcpy(x2.data(), b, n, cudaMemcpyDeviceToHost);
                return x1 == x2; };
            const bool e = same(dh1, dh2, (size_t)M * H * 4) && same(dy, dy2, (size_t)M * D * 4) && same(dq, dq2, (size_t)M * D) && same(dqs, dqs2, (size_t)M * D / 8);
            if (!e) printf("post_cnq M=%d: MISMATCH\n", M);
            ok = ok && e;
        }
        printf("post_cnq fused vs hc_post + collapse_norm_quant: %s\n", v41k::post_cnq_ok(1) ? (ok ? "BITWISE-OK" : "FAIL") : "skipped (off)");
        if (!ok) return 1;
    }
    if (argc > 2) {
        cudaStream_t st; cudaStreamCreate(&st);
        cudaGraph_t gr; cudaGraphExec_t ge;
        cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal);
        for (int i = 0; i < 100; ++i) v41k::collapse_norm_quant(dy, dq, dqs, dx, dp, dw, 1, 1e-6f, st);
        cudaStreamEndCapture(st, &gr); cudaGraphInstantiate(&ge, gr, 0);
        cudaGraphLaunch(ge, st); cudaStreamSynchronize(st);
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < 20; ++i) cudaGraphLaunch(ge, st);
        cudaStreamSynchronize(st);
        const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / 2000;
        printf("M=1 %.2f us per launch (graph, back to back)\n", us);
    }
    printf("wrote %s\n", argv[1]);
    return 0;
}
