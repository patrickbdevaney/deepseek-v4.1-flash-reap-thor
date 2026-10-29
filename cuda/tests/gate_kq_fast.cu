// gate_kq_fast.cu — the half2 Q2_K/Q3_K expert GEMV (v41_kq_fast.cuh) against the exact decode.
//
// CORRECTNESS. Random superblocks (sane fp16 d/dmin), random e4m3 activations with pow2 scales per 32.
// Reference: DecKQ (bit-exact with numpy) decoded to fp32, times e4m3*scale, accumulated in fp64.
// Pass if every output is within max(2e-3 * rms(ref), 2 bf16 ulps of |ref|) -- the kernel rounds its
// output to bf16 like the engine (one bf16 ulp is 2^-8 relative), plus an 8-term fp16 sum per lane.
// Also checks MT=4 against MT=1 on the same rows (verify must match AR decode).
//
// SPEED. Real shapes (gate|up: N 4608 K 5120; down: N 5120 K 2304), 6 experts x 1 row (decode),
// cycling expert sets out of a pool far larger than the 33.6 MB L2; reports GB/s of weight bytes
// for the old fp32-decode kernel (copied from v41_moe.cu k_experts) and the new one, and a
// streaming read of the same pool.
//
//   nvcc -O3 -std=c++17 -gencode arch=compute_110a,code=sm_110a -I include -I kernels/v41 \
//        tests/gate_kq_fast.cu -o build/gate_kq_fast && ./build/gate_kq_fast
#include "v41_kernels.h"
#include "v41_config.h"
#include "v41_kq_fast.cuh"
#include "v41_kq_mma.cuh"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <cstring>
// GATE_KQ=lpr,rpl: every k_kq launch uses that variant (default 32,8)
static int g_lpr = 32, g_rpl = 8;
static const bool g_init = [] { if (const char* e = getenv("GATE_KQ")) sscanf(e, "%d,%d", &g_lpr, &g_rpl); return true; }();
#define KQ_RPC kq_rows_per_cta(g_lpr, g_rpl)

using namespace v41g;
#define CU(x) do { cudaError_t e_ = (x); if (e_) { printf("CUDA %s @%d\n", cudaGetErrorString(e_), __LINE__); exit(1); } } while (0)

struct Unit { int e, start, cnt; };
template <int QT, int MT, bool SOA> static auto kqk() {
    if (g_lpr == 16 && g_rpl == 4) return k_kq<QT, MT, Unit, SOA, 16, 4>;
    if (g_lpr == 16 && g_rpl == 8) return k_kq<QT, MT, Unit, SOA, 16, 8>;
    return k_kq<QT, MT, Unit, SOA>;
}
#define KQK(QT, MT, SOA) kqk<QT, MT, SOA>()

// ---- the shipping fp32-decode kernel, verbatim from v41_moe.cu (for the speed baseline)
template <int QT> struct KQDecT : DecKQ<QT> {
    __device__ static KQDecT make(const KQTab& t, int e, int K) {
        KQDecT d;
        if (t.pass == 0) { d.W0 = t.w1[e]; d.W1 = t.w3[e]; d.split = v41::MOE_INTER; }
        else { d.W0 = d.W1 = t.w2[e]; d.split = 1 << 30; }
        d.K = K; return d;
    }
};
template <class Dec, int MT>
__global__ void __launch_bounds__(256) k_old(float* __restrict__ out, int ldo, XSrc X, const Unit* units,
                                             const int* units_n, KQTab tab, int N, int K) {
    const int u = blockIdx.y;
    if (u >= *units_n) return;
    const Unit U = units[u];
    extern __shared__ float4 xs4[];
    const int nch = K / 32;
    for (int t = 0; t < MT; ++t) stage_row(xs4 + (size_t)t * nch * 8, X, U.start + min(t, U.cnt - 1), 0, K);
    __syncthreads();
    const Dec dec = Dec::make(tab, U.e, K);
    const int row0 = blockIdx.x * ROWS_PER_CTA;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    float acc[ROWS_PER_WARP][MT];
    for (int r = 0; r < ROWS_PER_WARP; ++r) for (int t = 0; t < MT; ++t) acc[r][t] = 0.f;
    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) {
        const int row = min(row0 + warp * ROWS_PER_WARP + r, N - 1);
        for (int cb = 0; cb < nch; cb += 32) {
            const int c = cb + lane, cc = min(c, nch - 1);
            float w[32]; dec.load(row, cc, w);
            float a[MT];
            for (int t = 0; t < MT; ++t) a[t] = 0.f;
            dot_chunk<MT>(a, w, xs4, cc, nch);
            if (c < nch) for (int t = 0; t < MT; ++t) acc[r][t] += a[t];
        }
    }
    for (int r = 0; r < ROWS_PER_WARP; ++r)
        for (int t = 0; t < MT; ++t) {
            const float v = warp_sum(acc[r][t]);
            const int row = row0 + warp * ROWS_PER_WARP + r;
            if (lane == 0 && row < N && t < U.cnt) out[(size_t)(U.start + t) * ldo + row] = bf16r(v);
        }
}

// ---- reference: one thread per (pair, row), exact decode, fp64 sum
template <int QT>
__global__ void k_ref(double* out, int ldo, XSrc X, const Unit* units, int nunits, KQTab tab, int N, int K) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x, u = blockIdx.y;
    if (row >= N || u >= nunits) return;
    const Unit U = units[u];
    const KQDecT<QT> dec = KQDecT<QT>::make(tab, U.e, K);
    for (int t = 0; t < U.cnt; ++t) {
        const int p = U.start + t;
        double s = 0.0;
        for (int c = 0; c < K / 32; ++c) {
            float w[32]; dec.load(row, c, w);
            const float xs = X.xs[(size_t)p * (X.ldx / 32) + c];
            for (int i = 0; i < 32; ++i) s += (double)w[i] * (double)(e4m3f(X.xq[(size_t)p * X.ldx + c * 32 + i]) * xs);
        }
        out[(size_t)p * ldo + row] = s;
    }
}

static uint16_t f2h(float f) { __half h = __float2half(f); return *reinterpret_cast<uint16_t*>(&h); }

// one expert tensor [rows, K] of QT superblocks with random codes/scales and sane fp16 fields
static std::vector<uint8_t> make_tensor(int QT, int rows, int K, std::mt19937& g) {
    const int BB = QT == 2 ? 84 : 110, nsb = K / 256;
    std::vector<uint8_t> v((size_t)rows * nsb * BB + 16);
    for (auto& b : v) b = (uint8_t)g();
    std::uniform_real_distribution<float> ud(0.002f, 0.02f);
    for (size_t s = 0; s < (size_t)rows * nsb; ++s) {
        uint8_t* b = v.data() + s * BB;
        if (QT == 2) { uint16_t d = f2h(ud(g)), m = f2h(ud(g)); memcpy(b + 80, &d, 2); memcpy(b + 82, &m, 2); }
        else { uint16_t d = f2h(ud(g)); memcpy(b + 108, &d, 2); }
    }
    return v;
}

template <int QT>
static bool run(int pass, int P, int MT_units, std::mt19937& g, bool bench) {
    const int N = pass == 0 ? 2 * v41::MOE_INTER : v41::DIM, K = pass == 0 ? v41::DIM : v41::MOE_INTER;
    const int POOL = 48;   // experts in the pool; x 6 per set, so sets rotate through 8 distinct groups
    std::vector<uint8_t*> d1(POOL), d3(POOL), d2(POOL);
    size_t wbytes_per_expert = 0;
    for (int e = 0; e < POOL; ++e) {
        auto put = [&](int rows, int k) { auto h = make_tensor(QT, rows, k, g); uint8_t* d; CU(cudaMalloc(&d, h.size()));
                                          CU(cudaMemcpy(d, h.data(), h.size(), cudaMemcpyHostToDevice)); return d; };
        d1[e] = put(v41::MOE_INTER, v41::DIM); d3[e] = put(v41::MOE_INTER, v41::DIM); d2[e] = put(v41::DIM, v41::MOE_INTER);
    }
    wbytes_per_expert = (size_t)N * (K / 256) * (QT == 2 ? 84 : 110);
    const uint8_t** t1; const uint8_t** t3; const uint8_t** t2;
    CU(cudaMalloc(&t1, POOL * 8)); CU(cudaMalloc(&t3, POOL * 8)); CU(cudaMalloc(&t2, POOL * 8));
    CU(cudaMemcpy(t1, d1.data(), POOL * 8, cudaMemcpyHostToDevice)); CU(cudaMemcpy(t3, d3.data(), POOL * 8, cudaMemcpyHostToDevice));
    CU(cudaMemcpy(t2, d2.data(), POOL * 8, cudaMemcpyHostToDevice));
    KQTab tab{t1, t3, t2, pass};
    // the same pool reordered by kq_soa_repack, for the SOA kernel
    std::vector<uint8_t*> s1(POOL), s3(POOL), s2(POOL);
    {
        void* tmp; CU(cudaMalloc(&tmp, (size_t)v41::DIM * (v41::MOE_INTER / 256 + 1) * 110));
        auto soa = [&](const uint8_t* src, int rows, int k) {
            const size_t n = (size_t)rows * (k / 256) * (QT == 2 ? 84 : 110); uint8_t* d; CU(cudaMalloc(&d, n));
            CU(cudaMemcpy(d, src, n, cudaMemcpyDeviceToDevice)); v41k::kq_soa_repack(d, rows, k, QT, tmp, 0); return d; };
        for (int e = 0; e < POOL; ++e) {
            s1[e] = soa(d1[e], v41::MOE_INTER, v41::DIM); s3[e] = soa(d3[e], v41::MOE_INTER, v41::DIM);
            s2[e] = soa(d2[e], v41::DIM, v41::MOE_INTER);
        }
        CU(cudaDeviceSynchronize()); cudaFree(tmp);
    }
    const uint8_t** u1; const uint8_t** u3; const uint8_t** u2;
    CU(cudaMalloc(&u1, POOL * 8)); CU(cudaMalloc(&u3, POOL * 8)); CU(cudaMalloc(&u2, POOL * 8));
    CU(cudaMemcpy(u1, s1.data(), POOL * 8, cudaMemcpyHostToDevice)); CU(cudaMemcpy(u3, s3.data(), POOL * 8, cudaMemcpyHostToDevice));
    CU(cudaMemcpy(u2, s2.data(), POOL * 8, cudaMemcpyHostToDevice));
    KQTab tabs{u1, u3, u2, pass};
    // activations: P rows, e4m3 of N(0,1)*k, pow2 scales
    std::vector<uint8_t> hx((size_t)P * K); std::vector<float> hs((size_t)P * K / 32);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto& b : hx) b = (uint8_t)__nv_cvt_float_to_fp8(nd(g) * 40.f, __NV_SATFINITE, __NV_E4M3);
    for (auto& s : hs) s = ldexpf(1.f, -(int)(g() % 8) - 4);
    uint8_t* dx; float* dxs; CU(cudaMalloc(&dx, hx.size())); CU(cudaMalloc(&dxs, hs.size() * 4));
    CU(cudaMemcpy(dx, hx.data(), hx.size(), cudaMemcpyHostToDevice)); CU(cudaMemcpy(dxs, hs.data(), hs.size() * 4, cudaMemcpyHostToDevice));
    XSrc X; X.xq = dx; X.xs = dxs; X.ldx = K;
    // units: P pairs over distinct experts, MT_units rows each
    std::vector<Unit> hu; for (int p = 0, e = 0; p < P; p += MT_units, e += 5) hu.push_back(Unit{e % POOL, p, std::min(MT_units, P - p)});
    const int nu = (int)hu.size();
    Unit* du; int* dnu; int* ptok; CU(cudaMalloc(&du, 64 * sizeof(Unit))); CU(cudaMalloc(&dnu, 4)); CU(cudaMalloc(&ptok, P * 4));
    std::vector<int> id(P); for (int i = 0; i < P; ++i) id[i] = i;
    CU(cudaMemcpy(ptok, id.data(), P * 4, cudaMemcpyHostToDevice));
    CU(cudaMemcpy(du, hu.data(), nu * sizeof(Unit), cudaMemcpyHostToDevice)); CU(cudaMemcpy(dnu, &nu, 4, cudaMemcpyHostToDevice));
    float* dout; double* dref; CU(cudaMalloc(&dout, (size_t)P * N * 4)); CU(cudaMalloc(&dref, (size_t)P * N * 8));
    dim3 grid((N + KQ_RPC - 1) / KQ_RPC, nu);
    const size_t smem = (size_t)MT_units * kq_stage_bytes(K);
    auto launch_new = [&](const Unit* units) {
        if (MT_units == 1) { auto kf = KQK(QT, 1, false); kf<<<grid, 256, smem>>>(dout, N, X, true, units, dnu, ptok, tab, N, K, 1, 1 << 30); }
        else { auto kf = KQK(QT, 4, false); CU(cudaFuncSetAttribute(kf, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
               kf<<<grid, 256, smem>>>(dout, N, X, true, units, dnu, ptok, tab, N, K, 1, 1 << 30); }
    };
    launch_new(du); CU(cudaGetLastError());
    k_ref<QT><<<dim3((N + 127) / 128, nu), 128>>>(dref, N, X, du, nu, tab, N, K);
    CU(cudaDeviceSynchronize());
    std::vector<float> o((size_t)P * N); std::vector<double> r((size_t)P * N);
    CU(cudaMemcpy(o.data(), dout, o.size() * 4, cudaMemcpyDeviceToHost)); CU(cudaMemcpy(r.data(), dref, r.size() * 8, cudaMemcpyDeviceToHost));
    double rms = 0; for (double v : r) rms += v * v; rms = sqrt(rms / r.size());
    double worst = 0; long bad = 0;
    for (size_t i = 0; i < r.size(); ++i) {
        const double err = fabs(o[i] - r[i]), tol = std::max(2e-3 * rms, 2.0 * fabs(r[i]) / 256.0);
        worst = std::max(worst, err / rms);
        if (err > tol) ++bad;
    }
    printf("Q%d_K pass %d P=%d MT=%d: rms(ref) %.4g, max err/rms %.3g, out of tol %ld / %zu %s\n", QT, pass, P, MT_units, rms,
           worst, bad, r.size(), bad ? "FAIL" : "ok");
    bool ok = bad == 0;
    {   // SOA layout must give the AoS fast kernel's output bit for bit
        float* so; CU(cudaMalloc(&so, (size_t)P * N * 4));
        if (MT_units == 1) { auto kf = KQK(QT, 1, true); kf<<<grid, 256, smem>>>(so, N, X, true, du, dnu, ptok, tabs, N, K, 1, 1 << 30); }
        else { auto kf = KQK(QT, 4, true); CU(cudaFuncSetAttribute(kf, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
               kf<<<grid, 256, smem>>>(so, N, X, true, du, dnu, ptok, tabs, N, K, 1, 1 << 30); }
        CU(cudaDeviceSynchronize());
        std::vector<float> os((size_t)P * N); CU(cudaMemcpy(os.data(), so, os.size() * 4, cudaMemcpyDeviceToHost));
        long diff = 0; for (size_t i = 0; i < os.size(); ++i) diff += memcmp(&os[i], &o[i], 4) != 0;
        printf("  SOA vs AoS fast: %ld / %zu differ %s\n", diff, os.size(), diff ? "FAIL" : "ok");
        ok = ok && diff == 0; cudaFree(so);
    }
    if (bench) {
        // rotate sets of 6 experts through the pool so nothing is L2-resident
        const int SETS = POOL / 6, REPS = 200;
        std::vector<Unit*> sets(SETS);
        for (int s = 0; s < SETS; ++s) {
            std::vector<Unit> u; for (int k = 0; k < 6; ++k) u.push_back(Unit{s * 6 + k, k, 1});
            CU(cudaMalloc(&sets[s], 6 * sizeof(Unit))); CU(cudaMemcpy(sets[s], u.data(), 6 * sizeof(Unit), cudaMemcpyHostToDevice));
        }
        int six = 6; CU(cudaMemcpy(dnu, &six, 4, cudaMemcpyHostToDevice));
        dim3 g6((N + KQ_RPC - 1) / KQ_RPC, 6);
        cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b); float ms;
        const double bytes = 6.0 * wbytes_per_expert * REPS;
        const size_t smem_old = (size_t)K * 4;
        for (int w = 0; w < 2; ++w) {
            cudaEventRecord(a);
            for (int i = 0; i < REPS; ++i) k_old<KQDecT<QT>, 1><<<g6, 256, smem_old>>>(dout, N, X, sets[i % SETS], dnu, tab, N, K);
            cudaEventRecord(b); CU(cudaEventSynchronize(b)); cudaEventElapsedTime(&ms, a, b);
        }
        const double old_gbs = bytes / ms / 1e6;
        for (int w = 0; w < 2; ++w) {
            cudaEventRecord(a);
            for (int i = 0; i < REPS; ++i) { auto kf = KQK(QT, 1, false); kf<<<g6, 256, kq_stage_bytes(K)>>>(dout, N, X, true, sets[i % SETS], dnu, ptok, tab, N, K, 1, 1 << 30); }
            cudaEventRecord(b); CU(cudaEventSynchronize(b)); cudaEventElapsedTime(&ms, a, b);
        }
        const double new_gbs = bytes / ms / 1e6;
        for (int w = 0; w < 2; ++w) {
            cudaEventRecord(a);
            for (int i = 0; i < REPS; ++i) { auto kf = KQK(QT, 1, true); kf<<<g6, 256, kq_stage_bytes(K)>>>(dout, N, X, true, sets[i % SETS], dnu, ptok, tabs, N, K, 1, 1 << 30); }
            cudaEventRecord(b); CU(cudaEventSynchronize(b)); cudaEventElapsedTime(&ms, a, b);
        }
        printf("  bench SOA: %.1f GB/s\n", bytes / ms / 1e6);
        printf("  bench M=1 x6 experts: old %.1f GB/s  new %.1f GB/s  %.2fx  (%.0f us per launch)\n", old_gbs, new_gbs,
               new_gbs / old_gbs, 1e6 * bytes / REPS / (new_gbs * 1e9));
        CU(cudaMemcpy(dnu, &nu, 4, cudaMemcpyHostToDevice));
    }
    // MT=4 must reproduce MT=1 bit for bit on the same pairs (verify == AR)
    if (MT_units == 4) {
        std::vector<Unit> h1; for (const Unit& u : hu) for (int t = 0; t < u.cnt; ++t) h1.push_back(Unit{u.e, u.start + t, 1});
        const int n1 = (int)h1.size(); Unit* d1u; CU(cudaMalloc(&d1u, n1 * sizeof(Unit)));
        CU(cudaMemcpy(d1u, h1.data(), n1 * sizeof(Unit), cudaMemcpyHostToDevice)); CU(cudaMemcpy(dnu, &n1, 4, cudaMemcpyHostToDevice));
        float* d1o; CU(cudaMalloc(&d1o, (size_t)P * N * 4));
        auto kf = KQK(QT, 1, false);
        kf<<<dim3((N + KQ_RPC - 1) / KQ_RPC, n1), 256, kq_stage_bytes(K)>>>(d1o, N, X, true, d1u, dnu, ptok, tab, N, K, 1, 1 << 30);
        CU(cudaDeviceSynchronize());
        std::vector<float> o1((size_t)P * N); CU(cudaMemcpy(o1.data(), d1o, o1.size() * 4, cudaMemcpyDeviceToHost));
        long diff = 0; for (size_t i = 0; i < o1.size(); ++i) diff += o1[i] != o[i];
        printf("  MT=4 vs MT=1 on the same pairs: %ld / %zu differ %s\n", diff, o1.size(), diff ? "FAIL" : "ok");
        ok = ok && diff == 0;
    }
    {   // fp8 tensor-core kernel (v41_kq_mma.cuh) on the same pool, repacked
        std::vector<uint8_t*> m1(POOL), m3(POOL), m2(POOL);
        auto mm = [&](const uint8_t* src, int rows, int k) {
            const size_t n = (size_t)rows * (k / 256) * (QT == 2 ? 84 : 110); uint8_t* d; CU(cudaMalloc(&d, n));
            const size_t th = (size_t)(rows / 16) * (k / 256) * 32;
            k_kq_mma_repack<<<(unsigned)((th + 255) / 256), 256>>>(d, src, rows, k / 256, QT); return d; };
        for (int e = 0; e < POOL; ++e) {
            m1[e] = mm(d1[e], v41::MOE_INTER, v41::DIM); m3[e] = mm(d3[e], v41::MOE_INTER, v41::DIM);
            m2[e] = mm(d2[e], v41::DIM, v41::MOE_INTER);
        }
        CU(cudaDeviceSynchronize());
        const uint8_t** v1; const uint8_t** v3; const uint8_t** v2;
        CU(cudaMalloc(&v1, POOL * 8)); CU(cudaMalloc(&v3, POOL * 8)); CU(cudaMalloc(&v2, POOL * 8));
        CU(cudaMemcpy(v1, m1.data(), POOL * 8, cudaMemcpyHostToDevice)); CU(cudaMemcpy(v3, m3.data(), POOL * 8, cudaMemcpyHostToDevice));
        CU(cudaMemcpy(v2, m2.data(), POOL * 8, cudaMemcpyHostToDevice));
        KQTab tabm{v1, v3, v2, pass};
        dim3 gm((N + KQM_ROWS_PER_CTA - 1) / KQM_ROWS_PER_CTA, nu);
        float* mo; CU(cudaMalloc(&mo, (size_t)P * N * 4)); CU(cudaMemset(mo, 0, (size_t)P * N * 4));
        auto launch_m = [&](float* o_, const Unit* units, dim3 gg, int mt) {
            if (mt == 1) { auto kf = k_kqm<QT, 1, Unit>; kf<<<gg, 256, kqm_stage_bytes(K)>>>(o_, N, X, true, units, dnu, ptok, tabm, N, K, 1, 1 << 30); }
            else { auto kf = k_kqm<QT, 4, Unit>; const size_t sm = 4 * kqm_stage_bytes(K);
                   if (sm + 1024 > 48 * 1024) CU(cudaFuncSetAttribute(kf, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm));
                   kf<<<gg, 256, sm>>>(o_, N, X, true, units, dnu, ptok, tabm, N, K, 1, 1 << 30); }
        };
        launch_m(mo, du, gm, MT_units); CU(cudaGetLastError()); CU(cudaDeviceSynchronize());
        std::vector<float> om((size_t)P * N); CU(cudaMemcpy(om.data(), mo, om.size() * 4, cudaMemcpyDeviceToHost));
        double wm = 0, wf = 0; long badm = 0;
        for (size_t i = 0; i < r.size(); ++i) {
            const double err = fabs(om[i] - r[i]), tol = std::max(2e-3 * rms, 2.0 * fabs(r[i]) / 256.0);
            wm = std::max(wm, err / rms); wf = std::max(wf, fabs(o[i] - r[i]) / rms);
            if (err > tol) ++badm;
        }
        printf("  MMA: max err/rms %.3g (half2 %.3g), out of tol %ld %s\n", wm, wf, badm, badm ? "FAIL" : "ok");
        ok = ok && badm == 0;
        if (MT_units == 4) {   // MT=4 must equal MT=1 bit for bit
            std::vector<Unit> h1; for (const Unit& uu : hu) for (int tt = 0; tt < uu.cnt; ++tt) h1.push_back(Unit{uu.e, uu.start + tt, 1});
            const int n1 = (int)h1.size(); Unit* d1u; CU(cudaMalloc(&d1u, n1 * sizeof(Unit)));
            CU(cudaMemcpy(d1u, h1.data(), n1 * sizeof(Unit), cudaMemcpyHostToDevice)); CU(cudaMemcpy(dnu, &n1, 4, cudaMemcpyHostToDevice));
            float* o1d; CU(cudaMalloc(&o1d, (size_t)P * N * 4));
            launch_m(o1d, d1u, dim3(gm.x, n1), 1); CU(cudaDeviceSynchronize());
            std::vector<float> o1((size_t)P * N); CU(cudaMemcpy(o1.data(), o1d, o1.size() * 4, cudaMemcpyDeviceToHost));
            long diff = 0; for (size_t i = 0; i < o1.size(); ++i) diff += memcmp(&o1[i], &om[i], 4) != 0;
            printf("  MMA MT=4 vs MT=1: %ld / %zu differ %s\n", diff, o1.size(), diff ? "FAIL" : "ok");
            ok = ok && diff == 0; CU(cudaMemcpy(dnu, &nu, 4, cudaMemcpyHostToDevice)); cudaFree(o1d); cudaFree(d1u);
        }
        if (MT_units == 4 && P >= 16) {   // prefill NT = 2 tiles (MT = 16) and MT = 8 vs MT = 1, full and partial units
            XSrc XP = X; uint8_t* stg; CU(cudaMalloc(&stg, 16 * kqm_stage_bytes(K)));
            k_kqm_prep<<<16, 256>>>(stg, X, K); XP.stg = stg;
            auto kq16 = [&](float* o_, const Unit* units, int n, auto kf, int mt, bool pre = false) {
                const size_t sm = (size_t)mt * kqm_stage_bytes(K);
                if (sm + 1024 > 48 * 1024) CU(cudaFuncSetAttribute(kf, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm));
                CU(cudaMemcpy(dnu, &n, 4, cudaMemcpyHostToDevice));
                kf<<<dim3(gm.x, n), 256, sm>>>(o_, N, pre ? XP : X, true, units, dnu, ptok, tabm, N, K, 1, 1 << 30); CU(cudaGetLastError());
            };
            for (int cnt : {16, 11, 9}) {
                Unit hb{7, 0, cnt}; std::vector<Unit> h1; for (int tt = 0; tt < cnt; ++tt) h1.push_back(Unit{7, tt, 1});
                std::vector<Unit> h8; for (int tt = 0; tt < cnt; tt += 8) h8.push_back(Unit{7, tt, std::min(8, cnt - tt)});
                Unit *db, *d1u, *d8u; CU(cudaMalloc(&db, sizeof(Unit))); CU(cudaMalloc(&d1u, cnt * sizeof(Unit))); CU(cudaMalloc(&d8u, 2 * sizeof(Unit)));
                CU(cudaMemcpy(db, &hb, sizeof(Unit), cudaMemcpyHostToDevice));
                CU(cudaMemcpy(d1u, h1.data(), cnt * sizeof(Unit), cudaMemcpyHostToDevice));
                CU(cudaMemcpy(d8u, h8.data(), h8.size() * sizeof(Unit), cudaMemcpyHostToDevice));
                float *oa, *ob, *oc; CU(cudaMalloc(&oa, (size_t)16 * N * 4)); CU(cudaMalloc(&ob, (size_t)16 * N * 4)); CU(cudaMalloc(&oc, (size_t)16 * N * 4));
                kq16(oa, db, 1, k_kqm<QT, 16, Unit>, 16); kq16(ob, d1u, cnt, k_kqm<QT, 1, Unit>, 1);
                kq16(oc, d8u, (int)h8.size(), k_kqm<QT, 8, Unit>, 8, true); kq16(oa, db, 1, k_kqm<QT, 16, Unit>, 16, true);
                CU(cudaDeviceSynchronize());
                std::vector<float> va((size_t)cnt * N), vb(va.size()), vc(va.size());
                CU(cudaMemcpy(va.data(), oa, va.size() * 4, cudaMemcpyDeviceToHost)); CU(cudaMemcpy(vb.data(), ob, vb.size() * 4, cudaMemcpyDeviceToHost));
                CU(cudaMemcpy(vc.data(), oc, vc.size() * 4, cudaMemcpyDeviceToHost));
                long d16 = 0, d8 = 0; for (size_t i = 0; i < va.size(); ++i) { d16 += memcmp(&va[i], &vb[i], 4) != 0; d8 += memcmp(&vc[i], &vb[i], 4) != 0; }
                printf("  MMA cnt %2d: MT=16 vs MT=1 %ld, MT=8 vs MT=1 %ld / %zu differ %s\n", cnt, d16, d8, va.size(), d16 || d8 ? "FAIL" : "ok");
                ok = ok && !d16 && !d8; cudaFree(oa); cudaFree(ob); cudaFree(oc); cudaFree(db); cudaFree(d1u); cudaFree(d8u);
            }
            {   // weight-read timing: 48 experts x 16 tokens as 2 x MT8 units vs 1 x MT16 unit
                std::vector<Unit> u8, u16;
                for (int e = 0; e < POOL; ++e) { u8.push_back(Unit{e, 0, 8}); u8.push_back(Unit{e, 8, 8}); u16.push_back(Unit{e, 0, 16}); }
                Unit *d8u, *d16u; CU(cudaMalloc(&d8u, u8.size() * sizeof(Unit))); CU(cudaMalloc(&d16u, u16.size() * sizeof(Unit)));
                CU(cudaMemcpy(d8u, u8.data(), u8.size() * sizeof(Unit), cudaMemcpyHostToDevice));
                CU(cudaMemcpy(d16u, u16.data(), u16.size() * sizeof(Unit), cudaMemcpyHostToDevice));
                float* ot; CU(cudaMalloc(&ot, (size_t)16 * N * 4));
                cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b); float m8 = 0, m16 = 0;
                for (int w = 0; w < 2; ++w) {
                    cudaEventRecord(a); for (int i = 0; i < 10; ++i) kq16(ot, d8u, (int)u8.size(), k_kqm<QT, 8, Unit>, 8);
                    cudaEventRecord(b); CU(cudaEventSynchronize(b)); cudaEventElapsedTime(&m8, a, b);
                    cudaEventRecord(a); for (int i = 0; i < 10; ++i) kq16(ot, d16u, (int)u16.size(), k_kqm<QT, 16, Unit>, 16);
                    cudaEventRecord(b); CU(cudaEventSynchronize(b)); cudaEventElapsedTime(&m16, a, b);
                }
                printf("  bench 48 experts x 16 tok: MT8 %.0f us  MT16 %.0f us  %.2fx\n", 100 * m8, 100 * m16, m8 / m16);
                for (int w = 0; w < 2; ++w) {
                    cudaEventRecord(a); for (int i = 0; i < 10; ++i) kq16(ot, d8u, (int)u8.size(), k_kqm<QT, 8, Unit>, 8, true);
                    cudaEventRecord(b); CU(cudaEventSynchronize(b)); cudaEventElapsedTime(&m8, a, b);
                    cudaEventRecord(a); for (int i = 0; i < 10; ++i) kq16(ot, d16u, (int)u16.size(), k_kqm<QT, 16, Unit>, 16, true);
                    cudaEventRecord(b); CU(cudaEventSynchronize(b)); cudaEventElapsedTime(&m16, a, b);
                }
                printf("  bench prestaged:          MT8 %.0f us  MT16 %.0f us  %.2fx\n", 100 * m8, 100 * m16, m8 / m16);
                cudaFree(ot); cudaFree(d8u); cudaFree(d16u); cudaFree(stg);
            }
            CU(cudaMemcpy(dnu, &nu, 4, cudaMemcpyHostToDevice));
        }
        if (bench) {
            const int SETS = POOL / 6, REPS = 200;
            std::vector<Unit*> sets(SETS);
            for (int s_ = 0; s_ < SETS; ++s_) {
                std::vector<Unit> uu; for (int k = 0; k < 6; ++k) uu.push_back(Unit{s_ * 6 + k, k, 1});
                CU(cudaMalloc(&sets[s_], 6 * sizeof(Unit))); CU(cudaMemcpy(sets[s_], uu.data(), 6 * sizeof(Unit), cudaMemcpyHostToDevice));
            }
            int six = 6; CU(cudaMemcpy(dnu, &six, 4, cudaMemcpyHostToDevice));
            cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b); float ms = 0;
            const double bytes = 6.0 * wbytes_per_expert * REPS;
            for (int w = 0; w < 2; ++w) {
                cudaEventRecord(a);
                for (int i = 0; i < REPS; ++i) launch_m(mo, sets[i % SETS], dim3(gm.x, 6), 1);
                cudaEventRecord(b); CU(cudaEventSynchronize(b)); cudaEventElapsedTime(&ms, a, b);
            }
            printf("  bench MMA M=1 x6: %.1f GB/s (%.0f us per launch)\n", bytes / ms / 1e6, 1e3 * ms / REPS);
            CU(cudaMemcpy(dnu, &nu, 4, cudaMemcpyHostToDevice));
        }
        for (int e = 0; e < POOL; ++e) { cudaFree(m1[e]); cudaFree(m3[e]); cudaFree(m2[e]); }
        cudaFree(mo);
    }
    for (int e = 0; e < POOL; ++e) { cudaFree(d1[e]); cudaFree(d3[e]); cudaFree(d2[e]); cudaFree(s1[e]); cudaFree(s3[e]); cudaFree(s2[e]); }
    cudaFree(dx); cudaFree(dxs); cudaFree(dout); cudaFree(dref);
    return ok;
}

int main() {
    printf("[gate] k_kq LPR %d RPL %d\n", g_lpr, g_rpl); (void)g_init;
    std::mt19937 g(1234);
    bool ok = true;
    for (int pass = 0; pass < 2; ++pass) {
        ok &= run<2>(pass, 6, 1, g, true);
        ok &= run<3>(pass, 6, 1, g, true);
        ok &= run<2>(pass, 16, 4, g, false);
        ok &= run<3>(pass, 16, 4, g, false);
    }
    printf("%s\n", ok ? "GATE PASS" : "GATE FAIL");
    return ok ? 0 : 1;
}
