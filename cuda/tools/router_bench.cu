// router_bench.cu — A9: isolated time of v41k::router (k_router_p, permuted gate) for the decode shape.
//   scripts/build_v41.sh router_bench && build/router_bench [--iters 400] [--M 1]
// E = 192 routed experts, K = DIM, gate cycled over copies so it is read from DRAM as in the step.
#include "v41_kernels.h"
#include "v41_config.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e_)); exit(2); } } while (0)

int main(int argc, char** argv) {
    int iters = 400, M = 1;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--iters")) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--M")) M = atoi(argv[++i]);
    }
    const int E = 192, K = v41::DIM, NC = 64;
    std::vector<uint16_t> hw((size_t)E * K);
    for (size_t i = 0; i < hw.size(); ++i) hw[i] = (uint16_t)(0x3C00 + (i * 2654435761u >> 20) % 512) ^ (i & 1 ? 0x8000 : 0);
    std::vector<float> hx((size_t)M * K), hb(E, 0.f);
    for (size_t i = 0; i < hx.size(); ++i) hx[i] = ((i * 40503u) % 1000) / 1000.f - 0.5f;
    uint16_t* W; float *x, *b, *w, *scr; int* idx;
    CK(cudaMalloc(&W, hw.size() * 2 * NC)); CK(cudaMalloc(&x, hx.size() * 4)); CK(cudaMalloc(&b, E * 4));
    CK(cudaMalloc(&w, M * 6 * 4)); CK(cudaMalloc(&idx, M * 6 * 4)); CK(cudaMalloc(&scr, v41k::router_scratch_bytes(M, E)));
    for (int c = 0; c < NC; ++c) {
        CK(cudaMemcpy(W + (size_t)c * E * K, hw.data(), hw.size() * 2, cudaMemcpyHostToDevice));
        v41k::router_permute(W + (size_t)c * E * K, E, K, 0);
    }
    CK(cudaMemcpy(x, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(b, hb.data(), E * 4, cudaMemcpyHostToDevice));
    CK(cudaDeviceSynchronize());
    cudaStream_t st; CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    auto run = [&](int c) { v41k::router(w, idx, x, W + (size_t)c * E * K, b, nullptr, nullptr, M, E, 6, 2.5f, st, scr); };
    for (int c = 0; c < NC; ++c) run(c);
    CK(cudaStreamSynchronize(st));
    // graph of `iters` back-to-back launches, as the engine replays them
    cudaGraph_t g; cudaGraphExec_t ex;
    CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
    for (int i = 0; i < iters; ++i) run(i % NC);
    CK(cudaStreamEndCapture(st, &g)); CK(cudaGraphInstantiate(&ex, g, 0));
    CK(cudaGraphLaunch(ex, st)); CK(cudaStreamSynchronize(st));
    cudaEvent_t a, bb; cudaEventCreate(&a); cudaEventCreate(&bb);
    cudaEventRecord(a, st); CK(cudaGraphLaunch(ex, st)); cudaEventRecord(bb, st); CK(cudaEventSynchronize(bb));
    float ms; cudaEventElapsedTime(&ms, a, bb);
    std::vector<int> hi(M * 6); CK(cudaMemcpy(hi.data(), idx, hi.size() * 4, cudaMemcpyDeviceToHost));
    printf("router M=%d E=%d K=%d: %.2f us/call (graph of %d)  ids", M, E, K, ms * 1000 / iters, iters);
    for (int i = 0; i < 6; ++i) printf(" %d", hi[i]);
    printf("\n");
    return 0;
}
