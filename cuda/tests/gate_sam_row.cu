// gate_sam_row: the P7a row form (k_sam_scores_row + k_sam_out_p) against K2 + K3 (k_sam_scores + k_sam_out), bitwise.
// Random q / ring / compressed rows as gate_sa_split; M up to 100 (two 64-row passes); positions before the
// window fills and long after; a third of the compressed indices -1; ncomp 0 / 64 / 512. Then timing at M = 64.
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
#include "v41_kernels.h"
#include "v41_config.h"
#include "v41_dev.cuh"
// the kernels again under gtest:: (distinct symbols from the engine's copies) + the old gather to compare Eb
namespace gtest {
namespace v41g { using namespace v41d; }
#include "v41_sa_mma.cuh"
namespace v41g {
// the pre-P7a gather, verbatim (16 entries per CTA, 4 B per item)
__global__ void __launch_bounds__(256) k_sam_gather_old(uint16_t* Eb, int* nent_out, const float* ring_buf, int ring,
                                                    const uint8_t* cc, const uint8_t* cs, const int* cidx, int ncomp,
                                                    const int* d_pos) {
    const int m = blockIdx.y, k0 = blockIdx.x * 16, pos = d_pos[m];
    __shared__ short ent[SAM_E];
    __shared__ int nent;
    const int* ci = cidx ? cidx + (size_t)m * ncomp : nullptr;
    if (threadIdx.x < 32) {
        const int lane = threadIdx.x;
        int n = 0;
        const int w0 = max(0, (v41::WINDOW - 1) - pos);
        for (int e = w0 + lane; e < v41::WINDOW; e += 32) ent[n + (e - w0)] = (short)e;
        n = v41::WINDOW - w0;
        for (int b = 0; b < ncomp; b += 32) {
            const int e = b + lane;
            const bool ok = e < ncomp && ci[e] >= 0;
            const unsigned bal = __ballot_sync(0xffffffffu, ok);
            if (ok) ent[n + __popc(bal & ((1u << lane) - 1u))] = (short)(v41::WINDOW + e);
            n += __popc(bal);
        }
        if (lane == 0) { nent = n; if (blockIdx.x == 0) nent_out[m] = n; }
    }
    __syncthreads();
    const int ne = nent;
    uint16_t* Em = Eb + (size_t)m * SAM_E * SAM_D;
    for (int i = threadIdx.x; i < 16 * SAM_D / 2; i += blockDim.x) {
        const int r = k0 + i / (SAM_D / 2), j = i % (SAM_D / 2);
        float a = 0.f, b = 0.f;                                  // rows past ne are zero (K3 multiplies them by p = 0)
        if (r < ne) {
            const int e = ent[r];
            if (e < v41::WINDOW) {
                const float* src = ring_buf + (size_t)((pos - (v41::WINDOW - 1) + e) % ring) * SAM_D;
                a = src[2 * j]; b = src[2 * j + 1];
            } else {
                const int t = ci[e - v41::WINDOW];
                const uint8_t byte = cc[(size_t)t * 256 + j];
                const float s = e4m3f(cs[(size_t)t * 32 + (2 * j) / v41::CKV_FP4_GRP]);
                a = fp4f(byte & 15) * s; b = fp4f(byte >> 4) * s;
            }
        }
        reinterpret_cast<uint32_t*>(Em + (size_t)r * SAM_D)[j] = (uint32_t)bf16_bits(a) | ((uint32_t)bf16_bits(b) << 16);
    }
}

} // namespace v41g
} // namespace gtest
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA %s line %d\n", cudaGetErrorString(e_), __LINE__); return 1; } } while (0)
using namespace v41;
int main() {
    const int RING = 256, NROWS = 4096, MMAX = 100;
    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.f, 1.f);
    auto randv = [&](size_t n, float s) { std::vector<float> v(n); for (auto& x : v) x = nd(rng) * s; return v; };
    std::vector<float> hq = randv((size_t)MMAX * N_HEADS * HEAD_DIM, 1.f), hr = randv((size_t)RING * HEAD_DIM, 1.f);
    std::vector<float> hsink = randv(N_HEADS, 1.f);
    std::vector<uint8_t> hcc((size_t)NROWS * 256), hcs((size_t)NROWS * 32);
    for (auto& b : hcc) b = rng() & 0xff;
    for (auto& b : hcs) b = 0x30 + (rng() % 0x20);
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
    for (int p0 : {0, 5, 127, 300, 70000})
    for (int M : {1, 3, 16, 64, 65, 100}) {
        std::vector<int> hp(M), hi((size_t)M * INDEX_TOPK);
        for (int m = 0; m < M; ++m) hp[m] = p0 + m;
        for (auto& x : hi) x = (rng() % 3 == 0) ? -1 : (int)(rng() % NROWS);
        CK(cudaMemcpy(pos, hp.data(), M * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(cidx, hi.data(), hi.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemset(o1, 0xff, on * 4)); CK(cudaMemset(o2, 0xff, on * 4));
        v41k::sam_row_set(1 << 30);
        v41k::sparse_attn(o1, q, ring, RING, cc, cs, ncomp ? cidx : nullptr, ncomp, 0, sink, pos, M, scale, 0);
        v41k::sam_row_set(1);
        v41k::sparse_attn(o2, q, ring, RING, cc, cs, ncomp ? cidx : nullptr, ncomp, 0, sink, pos, M, scale, 0);
        CK(cudaDeviceSynchronize());
        std::vector<float> a((size_t)M * N_HEADS * HEAD_DIM), b(a.size());
        CK(cudaMemcpy(a.data(), o1, a.size() * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(b.data(), o2, b.size() * 4, cudaMemcpyDeviceToHost));
        ++cases;
        size_t nan = 0; for (float v : a) nan += v != v;
        if (memcmp(a.data(), b.data(), a.size() * 4) || nan) {
            size_t i = 0; while (i < a.size() && !memcmp(&a[i], &b[i], 4)) ++i;
            if (fails < 5) printf("MISMATCH ncomp %d pos %d M %d at %zu: %.9g vs %.9g (nan %zu)\n", ncomp, p0, M, i, i < a.size() ? a[i] : 0.f, i < a.size() ? b[i] : 0.f, nan);
            ++fails;
        }
    }
    printf("[gate_sam_row] %d/%d cases bit-identical -> %s\n", cases - fails, cases, fails ? "FAIL" : "PASS");
    {   // new k_sam_gather vs the old one: every row < ne bitwise, ne equal
        using namespace gtest::v41g;
        uint16_t *ea, *eb; int *na, *nb;
        const size_t eb_n = (size_t)SAM_ROWS * SAM_E * SAM_D;
        CK(cudaMalloc(&ea, eb_n * 2)); CK(cudaMalloc(&eb, eb_n * 2)); CK(cudaMalloc(&na, SAM_ROWS * 4)); CK(cudaMalloc(&nb, SAM_ROWS * 4));
        int gf = 0, gc = 0;
        for (int ncomp : {0, 64, 512})
        for (int p0 : {0, 5, 127, 300, 70000}) {
            const int M = SAM_ROWS;
            std::vector<int> hp(M), hi((size_t)M * INDEX_TOPK);
            for (int m = 0; m < M; ++m) hp[m] = p0 + m;
            for (auto& x : hi) x = (rng() % 3 == 0) ? -1 : (int)(rng() % NROWS);
            CK(cudaMemcpy(pos, hp.data(), M * 4, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(cidx, hi.data(), hi.size() * 4, cudaMemcpyHostToDevice));
            k_sam_gather_old<<<dim3(SAM_E / 16, M), 256>>>(ea, na, ring, RING, cc, cs, ncomp ? cidx : nullptr, ncomp, pos);
            k_sam_gather<<<dim3(SAM_E / SAM_GE, M), 256>>>(eb, nb, ring, RING, cc, cs, ncomp ? cidx : nullptr, ncomp, pos);
            CK(cudaDeviceSynchronize());
            std::vector<uint16_t> a(eb_n), b(eb_n); std::vector<int> an(M), bn(M);
            CK(cudaMemcpy(a.data(), ea, eb_n * 2, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(b.data(), eb, eb_n * 2, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(an.data(), na, M * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(bn.data(), nb, M * 4, cudaMemcpyDeviceToHost));
            bool bad = an != bn;
            for (int m = 0; m < M && !bad; ++m)
                bad = memcmp(&a[(size_t)m * SAM_E * SAM_D], &b[(size_t)m * SAM_E * SAM_D], (size_t)an[m] * SAM_D * 2) != 0;
            ++gc; gf += bad;
            if (bad) printf("GATHER MISMATCH ncomp %d pos %d\n", ncomp, p0);
        }
        printf("[gate_sam_row] gather vs pre-P7a gather: %d/%d bit-identical -> %s\n", gc - gf, gc, gf ? "FAIL" : "PASS");
        fails += gf;
    }
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    const int M = 64;
    std::vector<int> hp(M), hi((size_t)M * INDEX_TOPK);
    for (int m = 0; m < M; ++m) hp[m] = 70000 + m;
    for (auto& x : hi) x = (int)(rng() % NROWS);
    cudaMemcpy(pos, hp.data(), M * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(cidx, hi.data(), hi.size() * 4, cudaMemcpyHostToDevice);
    for (int rowf : {0, 1}) {
        v41k::sam_row_set(rowf ? 1 : 1 << 30);
        for (int w = 0; w < 3; ++w) v41k::sparse_attn(o1, q, ring, RING, cc, cs, cidx, 512, 0, sink, pos, M, scale, 0);
        cudaEventRecord(e0);
        for (int it = 0; it < 20; ++it) v41k::sparse_attn(o1, q, ring, RING, cc, cs, cidx, 512, 0, sink, pos, M, scale, 0);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        printf("[gate_sam_row] M=64 ne=640 %s: %.3f ms/call\n", rowf ? "row form " : "K2 + K3  ", ms / 20);
    }
    return fails ? 1 : 0;
}
