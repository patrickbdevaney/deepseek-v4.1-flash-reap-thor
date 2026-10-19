// gate_mainkv_incr.cu — LADDER 1.0. `dspark_main_kv_upto` (incremental) must be BIT-IDENTICAL to
// `dspark_main_kv` (from scratch) for every split point, at both GEMM settings.
//
// WHY THIS GATE EXISTS AND WHY IT IS A SEPARATE BINARY FROM THE ENGINE. The in-situ check
// (DSV4_MAINKV_GATE=1) proves the same property on the shipped path, but it costs a ~10-minute
// 100 GiB checkpoint load to find out. Everything that can actually be wrong here is shape logic,
// not weights:
//
//   1. THE ROPE OFFSET. `rope_kernel` takes `crow = row / cos_stride_rows`, i.e. the cos/sin row is
//      the LOCAL row index. A sub-range that forgets to advance cosT/sinT by r0 rotates every row
//      by the wrong angle -- and produces a perfectly plausible, perfectly wrong main-KV.
//   2. THE GEMM PIN. `fp8_block_gemm`'s dispatch is M-DEPENDENT (M==1 -> GEMV, M in [2,8] -> the
//      NVFP4 overlay or the templated small-M GEMV, larger -> tc_fp8_gemm). The from-scratch call
//      runs at M = thousands and the incremental delta at M = 1..6, so going through that dispatch
//      compares a GEMV against a tensor-core tile. The delta path pins tc_fp8_gemm, whose own
//      dispatch depends only on N and K; this gate is what proves the pin actually holds, at
//      exactly the M values the decode loop produces.
//   3. THE CLAMP. A caller that rewinds hands back a `valid` ABOVE the requested `s`.
//
// None of the three needs real weights, so this runs in seconds on random ones and is the first
// thing to break if someone edits the split. Equality, not cosine: the claim is byte-identity, so
// the instrument is memcmp (LOOP_LOG Finding 68 is why that distinction gets its own binary).
//
//   nvcc -O2 -std=c++17 -gencode arch=compute_110a,code=sm_110a -I include \
//     tests/gate_mainkv_incr.cu kernels/dspark_attn.cu kernels/dspark.cu kernels/mla_attn.cu \
//     kernels/mla_forward.cu kernels/mla_decode.cu kernels/fp8_block_gemm.cu kernels/tc_fp8_gemm.cu \
//     kernels/hc.cu kernels/hc_sinkhorn.cu kernels/moe.cu kernels/tc_moe_gemm.cu kernels/block.cu \
//     kernels/compressor.cu kernels/indexer.cu kernels/compressed_attn.cu kernels/compressed_block.cu \
//     kernels/compressed_decode.cu kernels/block_decode.cu kernels/dscratch.cu kernels/dprof.cu \
//     kernels/nvfp4_dense.cu -o build/gate_mainkv_incr
#include "dspark_attn.h"
#include "deepseek_v4.h"
#include <cuda_fp8.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <random>
#include <vector>
using namespace dsv4;

#define CU(x) do{cudaError_t e=(x); if(e){printf("cuda %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)

extern bool g_tc_fp8;

static const int S = 2048;                       // context depth the gate builds up to

template <class T> static T* up(const std::vector<T>& h) {
    T* d; CU(cudaMalloc(&d, h.size() * sizeof(T)));
    CU(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}

// One pass: build `ref` from scratch at S, build `inc` through `splits`, memcmp.
// The splits deliberately include M=1 deltas (the GEMV/tensor-core divergence), a delta that
// straddles the tensor core's 16-row M tile, a REWIND (a split BELOW the current high-water mark,
// which must recompute nothing and must not corrupt anything), and the exact-S final step.
static int run(const MLAWeights& w, const float* main_x, const char* label) {
    float *ref, *inc;
    CU(cudaMalloc(&ref, (size_t)S * HEAD_DIM * 4));
    CU(cudaMalloc(&inc, (size_t)S * HEAD_DIM * 4));
    CU(cudaMemset(inc, 0, (size_t)S * HEAD_DIM * 4));
    dspark_main_kv(ref, main_x, w, S, EPS);
    CU(cudaDeviceSynchronize());

    const int splits[] = { 7, 8, 9, 15, 16, 17, 23, 128, 129, 130, 400, 401, 402, 403,
                           300 /* rewind: below the high-water mark */, 700, 701, 1024, 1025,
                           2000, 2047, 2048 };
    int valid = 0;
    for (int s : splits) {
        if (s < valid) valid = s;                // what rewind_to()/extend() do in the engine
        dspark_main_kv_upto(inc, main_x, w, s, &valid, EPS);
        CU(cudaDeviceSynchronize());
        if (valid != s) { printf("  [%s] FAIL: valid=%d after a call for s=%d\n", label, valid, s); return 1; }
    }
    // The clamp: a caller that hands back a `valid` above `s` must not read or write past `s`.
    valid = S + 64;
    dspark_main_kv_upto(inc, main_x, w, S, &valid, EPS);
    CU(cudaDeviceSynchronize());

    std::vector<float> a((size_t)S * HEAD_DIM), b(a.size());
    CU(cudaMemcpy(a.data(), ref, a.size() * 4, cudaMemcpyDeviceToHost));
    CU(cudaMemcpy(b.data(), inc, b.size() * 4, cudaMemcpyDeviceToHost));
    cudaFree(ref); cudaFree(inc);

    if (memcmp(a.data(), b.data(), a.size() * 4) == 0) {
        printf("  [%s] PASS: %d x %d floats byte-identical across %zu split points\n",
               label, S, HEAD_DIM, sizeof(splits) / sizeof(splits[0]));
        return 0;
    }
    size_t i = 0, ndiff = 0;
    while (i < a.size() && memcmp(&a[i], &b[i], 4) == 0) ++i;
    for (size_t j = 0; j < a.size(); ++j) if (memcmp(&a[j], &b[j], 4) != 0) ++ndiff;
    printf("  [%s] FAIL: %zu of %zu floats differ; first at row %zu col %zu (%.9g vs %.9g)\n",
           label, ndiff, a.size(), i / HEAD_DIM, i % HEAD_DIM, a[i], b[i]);
    // Which COLUMN it starts in localises the stage: >= NOPE_DIM is rope, < NOPE_DIM is the
    // GEMM/rmsnorm/quant chain.
    printf("  [%s] first differing column %zu is in the %s half\n",
           label, i % HEAD_DIM, (i % HEAD_DIM) >= (size_t)NOPE_DIM ? "ROPE" : "NOPE");
    return 1;
}

int main() {
    std::mt19937 rng(20260820);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_int_distribution<int> ub(0, 255);

    // main_x: the projected taps. Scale does not matter (everything downstream is scale-aware), but
    // it must not be degenerate -- act_quant's amax floor of 1e-4 would hide a row mix-up.
    std::vector<float> hx((size_t)S * DIM);
    for (auto& v : hx) v = nd(rng) * 0.5f;

    // wkv: [HEAD_DIM, DIM] e4m3 bytes with [HEAD_DIM/128, DIM/128] pow2 block scales.
    std::vector<uint8_t> hwkv((size_t)HEAD_DIM * DIM);
    for (auto& v : hwkv) { float q = nd(rng); v = __nv_fp8_e4m3(q).__x; }
    std::vector<float> hwkvs((size_t)(HEAD_DIM / 128) * (DIM / 128));
    for (auto& v : hwkvs) v = exp2f((float)(ub(rng) % 5 - 2));
    std::vector<float> hnorm(HEAD_DIM);
    for (auto& v : hnorm) v = 0.5f + fabsf(nd(rng)) * 0.2f;

    // cos/sin over S positions. Real freqs, so a row read at the wrong position is a real rotation
    // error rather than a no-op -- a constant table would let the rope-offset bug through.
    const int half = ROPE_DIM / 2;
    std::vector<float> hc((size_t)S * half), hs((size_t)S * half);
    for (int p = 0; p < S; ++p)
        for (int j = 0; j < half; ++j) {
            const double freq = p / pow(10000.0, (2.0 * j) / ROPE_DIM);
            hc[(size_t)p * half + j] = (float)cos(freq);
            hs[(size_t)p * half + j] = (float)sin(freq);
        }

    MLAWeights w{};
    w.wkv = up(hwkv); w.wkv_s = up(hwkvs); w.kv_norm = up(hnorm);
    w.cosT = up(hc);  w.sinT = up(hs);
    float* main_x = up(hx);

    printf("gate_mainkv_incr: S=%d DIM=%d HEAD_DIM=%d NOPE_DIM=%d ROPE_DIM=%d\n",
           S, DIM, HEAD_DIM, NOPE_DIM, ROPE_DIM);
    int bad = 0;
    // BOTH SETTINGS, because they are different kernels and the decode path is the one that matters.
    // g_tc_fp8 is false in the gates and the `forward` binary, and true in decode (forward.cu sets
    // it); a gate that only ran the default would never touch the tensor-core path this change is
    // actually pinned to.
    g_tc_fp8 = false; bad += run(w, main_x, "oracle  g_tc_fp8=0");
    g_tc_fp8 = true;  bad += run(w, main_x, "tensorcore g_tc_fp8=1");

    printf("gate_mainkv_incr: %s\n", bad ? "FAIL" : "PASS");
    return bad ? 1 : 0;
}
