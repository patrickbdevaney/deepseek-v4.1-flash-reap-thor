// v41_dense.cu — dense linears, norms, RoPE, QAT-sim quantisers, hyper-connections, embedding.
// Reference semantics are cited per kernel; see include/v41_kernels.h for the conventions.
#include "v41_kernels.h"
#include "v41_config.h"
#include "v41_gemv.cuh"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>

using namespace v41d;
using namespace v41g;

namespace v41k {

// --------------------------------------------------------------------------- module scratch
// Split-K partials and HC partials. Sized once by v41k_init (outside any graph capture: cudaMalloc
// is illegal while capturing). A call that would need more falls back to no split rather than
// allocating.
static float* g_part = nullptr; static size_t g_part_cap = 0;
void v41k_init(size_t part_floats) {
    if (g_part_cap >= part_floats) return;
    if (g_part) cudaFree(g_part);
    if (cudaMalloc(&g_part, part_floats * sizeof(float)) != cudaSuccess) { g_part = nullptr; g_part_cap = 0; return; }
    g_part_cap = part_floats;
}
static int g_nsm = 0;
static int nsm() { if (!g_nsm) { int d; cudaGetDevice(&d); cudaDeviceGetAttribute(&g_nsm, cudaDevAttrMultiProcessorCount, d); } return g_nsm; }

// --------------------------------------------------------------------------- dense GEMV
// grid (ceil(N/64), ceil(M/MT), ksplit). Group g = row / Ng reads x columns [g*K, (g+1)*K).
template <class Dec, int MT>
__global__ void __launch_bounds__(256) k_dense(float* __restrict__ y, XSrc X, Dec dec, int M, int N, int K,
                                               int Ng, int ksplit, float out_scale, int round, float* __restrict__ part) {
    extern __shared__ float4 xs4[];
    const int row0 = blockIdx.x * ROWS_PER_CTA;
    const int m0 = blockIdx.y * MT;
    const int g = row0 / Ng;
    const int nch = K / 32;
    const int cps = (nch + ksplit - 1) / ksplit;            // chunks per split
    const int c_lo = blockIdx.z * cps, c_hi = min(nch, c_lo + cps);
    for (int t = 0; t < MT; ++t) {
        const int m = min(m0 + t, M - 1);
        stage_row(xs4 + (size_t)t * nch * 8, X, m, g * K, K);
    }
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    float acc[ROWS_PER_WARP][MT];
    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r)
        #pragma unroll
        for (int t = 0; t < MT; ++t) acc[r][t] = 0.f;
    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) {
        const int row = min(row0 + warp * ROWS_PER_WARP + r, N - 1);
        for (int cb = c_lo; cb < c_hi; cb += 32) {           // every lane takes every trip (shuffle-safe)
            const int c = cb + lane, cc = min(c, c_hi - 1);
            float w[32]; dec.load(row, cc, w);
            float a[MT];
            #pragma unroll
            for (int t = 0; t < MT; ++t) a[t] = 0.f;
            dot_chunk<MT>(a, w, xs4, cc, nch);
            if (c < c_hi) {
                #pragma unroll
                for (int t = 0; t < MT; ++t) acc[r][t] += a[t];
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r)
        #pragma unroll
        for (int t = 0; t < MT; ++t) {
            const float v = warp_sum(acc[r][t]);
            const int row = row0 + warp * ROWS_PER_WARP + r, m = m0 + t;
            if (lane == 0 && row < N && m < M) {
                if (ksplit > 1) part[((size_t)blockIdx.z * M + m) * N + row] = v;
                else { float o = v * out_scale; y[(size_t)m * N + row] = round ? bf16r(o) : o; }
            }
        }
}
__global__ void k_ksum(float* y, const float* part, int ks, size_t MN, float out_scale, int round) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i >= MN) return;
    float s = 0.f; for (int z = 0; z < ks; ++z) s += part[(size_t)z * MN + i];   // fixed order
    s *= out_scale; y[i] = round ? bf16r(s) : s;
}

template <class Dec>
static void launch_dense(float* y, const XSrc& X, const Dec& dec, int M, int N, int K, int G, bool round,
                         float out_scale, cudaStream_t st) {
    const int Ng = N / G;
    const int nch = K / 32;
    int mt = (M == 1) ? 1 : (M == 2 ? 2 : 4);
    while (mt > 1 && (size_t)mt * K * sizeof(float) > 160 * 1024) mt >>= 1;   // smem cap
    const int ntile = (N + ROWS_PER_CTA - 1) / ROWS_PER_CTA, mtile = (M + mt - 1) / mt;
    // Split-K changes the summation order, so it is chosen independently of M for M <= SPEC_M: a
    // speculative verify (M = K+1) then sums every row exactly as the M = 1 decode step does, and
    // greedy speculation emits the same ids as plain decode (cuda/CHAIN_STEPS.sh spec_identity).
    constexpr int SPEC_M = 8;
    int ks = 1;
    const int ctas = ntile * (M <= SPEC_M ? 1 : mtile), want = 2 * nsm();
    if (ctas < want) {
        ks = std::min((want + ctas - 1) / ctas, std::max(1, nch / 8));
        if ((size_t)ks * std::max(M, SPEC_M) * N > g_part_cap) ks = 1;
    }
    dim3 grid(ntile, mtile, ks);
    const size_t smem = (size_t)mt * K * sizeof(float);
    #define V41_LAUNCH(MTV) do { auto kf = k_dense<Dec, MTV>; \
        if (smem > 48 * 1024) cudaFuncSetAttribute(kf, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem); \
        kf<<<grid, 256, smem, st>>>(y, X, dec, M, N, K, Ng, ks, out_scale, round ? 1 : 0, g_part); } while (0)
    if (mt == 1) V41_LAUNCH(1); else if (mt == 2) V41_LAUNCH(2); else V41_LAUNCH(4);
    #undef V41_LAUNCH
    if (ks > 1) {
        size_t MN = (size_t)M * N;
        k_ksum<<<(unsigned)((MN + 255) / 256), 256, 0, st>>>(y, g_part, ks, MN, out_scale, round ? 1 : 0);
    }
}

void fp8_linear(float* y, const uint8_t* xq, const float* xs, const uint8_t* W, const uint8_t* Ws,
                int M, int N, int K, bool round_bf16, cudaStream_t st, int G) {
    XSrc X; X.xq = xq; X.xs = xs; X.ldx = K * G;
    DecFP8 d{W, Ws, K};
    launch_dense(y, X, d, M, N, K, G, round_bf16, 1.0f, st);
}
void bf16_linear(float* y, const float* x, const uint16_t* W, int M, int N, int K, bool round_bf16,
                 cudaStream_t st, float out_scale) {
    XSrc X; X.xf = x; X.ldx = K;
    DecBF16 d{W, K};
    launch_dense(y, X, d, M, N, K, 1, round_bf16, out_scale, st);
}

// --------------------------------------------------------------------------- quantisers
// act_quant(x, 32, "ue8m0"): one warp per 32-group.
__global__ void k_act_quant(uint8_t* q, float* s, const float* x, size_t ngroups) {
    size_t gi = (blockIdx.x * (size_t)blockDim.x + threadIdx.x) >> 5; if (gi >= ngroups) return;
    const int lane = threadIdx.x & 31;
    const float v = x[gi * 32 + lane];
    const float amax = fmaxf(warp_max(fabsf(v)), 1e-4f);
    const float sc = pow2_ceil(amax / 448.f);
    q[gi * 32 + lane] = f2e4m3(fminf(fmaxf(v / sc, -448.f), 448.f));
    if (lane == 0) s[gi] = sc;
}
void act_quant_fp8(uint8_t* q, float* s, const float* x, int M, int K, cudaStream_t st) {
    size_t ng = (size_t)M * K / 32;
    k_act_quant<<<(unsigned)((ng * 32 + 255) / 256), 256, 0, st>>>(q, s, x, ng);
}
__global__ void k_fp8_sim(float* x, int dim, size_t ngroups) {
    size_t gi = (blockIdx.x * (size_t)blockDim.x + threadIdx.x) >> 5; if (gi >= ngroups) return;
    const int lane = threadIdx.x & 31;
    float* p = x + gi * 32 + lane;
    const float v = *p;
    const float amax = fmaxf(warp_max(fabsf(v)), 1e-4f);
    const float sc = pow2_ceil(amax / 448.f);
    *p = bf16r(e4m3f(f2e4m3(fminf(fmaxf(v / sc, -448.f), 448.f))) * sc);
}
void fp8_sim(float* x, int rows, int dim, cudaStream_t st) {
    size_t ng = (size_t)rows * dim / 32;
    k_fp8_sim<<<(unsigned)((ng * 32 + 255) / 256), 256, 0, st>>>(x, dim, ng);
}
// fp4_act_quant(x, grp, inplace=True, scale_dtype): one warp covers 32 elements = 32/grp groups.
__global__ void k_fp4_sim(float* x, size_t n, int grp, int e4m3s, uint8_t* packed, uint8_t* scales) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i >= n) return;   // n % 32 == 0
    const int lane = threadIdx.x & 31;
    const float v = x[i];
    float amax = fabsf(v);
    for (int o = grp / 2; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    float sc; uint8_t sbyte;
    if (e4m3s) { amax = fmaxf(amax, 6.f * 0.001953125f); sbyte = f2e4m3(amax / 6.f); sc = e4m3f(sbyte); }
    else       { amax = fmaxf(amax, 6.f * 1.17549435e-38f); sc = pow2_ceil(amax / 6.f); sbyte = pow2_ceil_e8m0(amax / 6.f); }
    const uint8_t code = fp4_code(fminf(fmaxf(v / sc, -6.f), 6.f));
    x[i] = bf16r(fp4f(code) * sc);
    if (packed) {
        const uint32_t other = __shfl_xor_sync(0xffffffffu, (uint32_t)code, 1);
        if (!(lane & 1)) packed[i >> 1] = (uint8_t)(code | (other << 4));
        if ((i % grp) == 0) scales[i / grp] = sbyte;
    }
}
void fp4_sim(float* x, int rows, int dim, int grp, bool e4m3_scale, uint8_t* packed, uint8_t* scales, cudaStream_t st) {
    size_t n = (size_t)rows * dim;
    k_fp4_sim<<<(unsigned)((n + 255) / 256), 256, 0, st>>>(x, n, grp, e4m3_scale ? 1 : 0, packed, scales);
}
__global__ void k_round(float* x, size_t n) { size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i < n) x[i] = bf16r(x[i]); }
void round_bf16(float* x, size_t n, cudaStream_t st) { k_round<<<(unsigned)((n + 255) / 256), 256, 0, st>>>(x, n); }
__global__ void k_scale_bf16(float* x, size_t n, float c) { size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i < n) x[i] = bf16r(x[i] * c); }
void scale_bf16(float* x, size_t n, float c, cudaStream_t st) { k_scale_bf16<<<(unsigned)((n + 255) / 256), 256, 0, st>>>(x, n, c); }
__global__ void k_argmax(int* out, const float* x, int n) {
    float bv = -INFINITY; int bi = 0x7fffffff;
    for (int i = threadIdx.x; i < n; i += blockDim.x) { const float v = x[i]; if (v > bv || (v == bv && i < bi)) { bv = v; bi = i; } }
    __shared__ float sv[1024]; __shared__ int si[1024];
    sv[threadIdx.x] = bv; si[threadIdx.x] = bi; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) { const float v = sv[threadIdx.x + s]; const int j = si[threadIdx.x + s];
            if (v > sv[threadIdx.x] || (v == sv[threadIdx.x] && j < si[threadIdx.x])) { sv[threadIdx.x] = v; si[threadIdx.x] = j; } }
        __syncthreads();
    }
    if (threadIdx.x == 0) *out = si[0];
}
void argmax(int* out, const float* x, int n, cudaStream_t st) { k_argmax<<<1, 1024, 0, st>>>(out, x, n); }
__global__ void k_add_bf16(float* y, const float* a, size_t n) { size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i < n) y[i] = bf16r(y[i] + a[i]); }
void add_bf16(float* y, const float* a, size_t n, cudaStream_t st) { k_add_bf16<<<(unsigned)((n + 255) / 256), 256, 0, st>>>(y, a, n); }

// --------------------------------------------------------------------------- RMSNorm
// RMSNorm.forward: x.float(); x * rsqrt(mean(x^2) + eps); (weight * x).to(bf16)
__global__ void k_rmsnorm(float* y, const float* x, const uint16_t* w, int dim, float eps) {
    const float* xr = x + (size_t)blockIdx.x * dim; float* yr = y + (size_t)blockIdx.x * dim;
    __shared__ float red[32];
    float ss = 0.f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) ss = fmaf(xr[i], xr[i], ss);
    ss = warp_sum(ss);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = threadIdx.x < (blockDim.x >> 5) ? red[threadIdx.x] : 0.f;
        v = warp_sum(v); if (threadIdx.x == 0) red[0] = v;
    }
    __syncthreads();
    const float r = rsqrtf(red[0] / dim + eps);
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        const float n = xr[i] * r;
        yr[i] = bf16r(w ? bf16f(w[i]) * n : n);
    }
}
void rmsnorm(float* y, const float* x, const uint16_t* w, int rows, int dim, float eps, cudaStream_t st) {
    k_rmsnorm<<<rows, dim >= 1024 ? 512 : 128, 0, st>>>(y, x, w, dim, eps);
}

// --------------------------------------------------------------------------- RoPE
// precompute_freqs_cis: freqs in fp32 exactly as the reference builds them (fp32 pow, fp32 YaRN
// blend), angle = fp32(pos * f) -- a single fp32 multiply, the same rounding torch.outer does --
// then accurate sincosf. Tables at 1M positions would cost 512 MB, so angles are formed on the fly.
__constant__ float c_freq[2][32];     // [0] base 10000 no YaRN ; [1] compress theta 160000 + YaRN
static bool g_freq_init = false;
static void init_freqs() {
    if (g_freq_init) return;
    float f[2][32];
    for (int k = 0; k < 2; ++k) {
        const float base = k ? v41::COMPRESS_ROPE_THETA : v41::ROPE_THETA;
        const int dim = v41::ROPE_DIM;
        for (int j = 0; j < 32; ++j) f[k][j] = 1.0f / powf(base, (float)(2 * j) / (float)dim);
        if (k) {
            const double orig = v41::YARN_ORIG_MAXPOS;
            auto corr = [&](double rot) { return dim * std::log(orig / (rot * 2 * M_PI)) / (2 * std::log((double)base)); };
            const int low = std::max((int)std::floor(corr(v41::YARN_BETA_FAST)), 0);
            const int high = std::min((int)std::ceil(corr(v41::YARN_BETA_SLOW)), dim - 1);
            for (int j = 0; j < 32; ++j) {
                float ramp = ((float)j - (float)low) / std::max((float)(high - low), 1e-3f);
                ramp = ramp < 0.f ? 0.f : (ramp > 1.f ? 1.f : ramp);
                const float smooth = 1.f - ramp;
                f[k][j] = f[k][j] / v41::YARN_FACTOR * (1.f - smooth) + f[k][j] * smooth;
            }
        }
    }
    cudaMemcpyToSymbol(c_freq, f, sizeof(f));
    g_freq_init = true;
}
void rope_set_tables() { init_freqs(); }   // call once outside capture (cudaMemcpyToSymbol)
__global__ void k_rope(float* x, int rows, int heads, int row_stride, int off, const int* d_pos, int pos_delta,
                       int inverse, int which) {
    const int r = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5); if (r >= rows) return;
    const int j = threadIdx.x & 31;                       // one complex pair per lane (64 dims)
    const int pos = d_pos[r / heads] + pos_delta;
    float s, c; sincosf((float)pos * c_freq[which][j], &s, &c);
    if (inverse) s = -s;
    float* p = x + (size_t)r * row_stride + off + 2 * j;
    const float re = p[0], im = p[1];
    p[0] = bf16r(re * c - im * s);
    p[1] = bf16r(re * s + im * c);
}
void rope(float* x, int rows, int heads, int row_stride, int off, const int* d_pos, int pos_delta,
          bool inverse, bool yarn, cudaStream_t st) {
    init_freqs();
    const int which = yarn ? 1 : 0;
    k_rope<<<(rows + 7) / 8, 256, 0, st>>>(x, rows, heads, row_stride, off, d_pos, pos_delta, inverse ? 1 : 0, which);
}

// --------------------------------------------------------------------------- hyper-connections
// hc_mixes: grid (M, 24 mixes + 1 sumsq, KS) partial dots over 20480 -> finalize (fixed order) + Sinkhorn.
static constexpr int HC_KS = 8;
__global__ void k_hc_partial(float* part, const float* x, const float* fn) {
    const int m = blockIdx.x, j = blockIdx.y, z = blockIdx.z;   // j == 24 -> sum of squares
    const int span = v41::HC_DIM / HC_KS;
    const float* xr = x + (size_t)m * v41::HC_DIM + z * span;
    const float* fr = fn + (size_t)(j < 24 ? j : 0) * v41::HC_DIM + z * span;
    float acc = 0.f;
    for (int i = threadIdx.x; i < span; i += blockDim.x) acc = fmaf(xr[i], j < 24 ? fr[i] : xr[i], acc);
    __shared__ float red[8];
    acc = warp_sum(acc);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = acc;
    __syncthreads();
    if (threadIdx.x == 0) { float s = 0.f; for (int w = 0; w < (int)(blockDim.x >> 5); ++w) s += red[w];
        part[((size_t)m * 25 + j) * HC_KS + z] = s; }
}
__global__ void k_hc_finish(float* pre, float* post, float* comb, const float* part, const float* scale,
                            const float* base) {
    const int m = blockIdx.x;
    if (threadIdx.x != 0) return;
    float mix[24];
    float ss = 0.f; for (int z = 0; z < HC_KS; ++z) ss += part[((size_t)m * 25 + 24) * HC_KS + z];
    const float rs = rsqrtf(ss / v41::HC_DIM + v41::NORM_EPS);
    for (int j = 0; j < 24; ++j) { float d = 0.f; for (int z = 0; z < HC_KS; ++z) d += part[((size_t)m * 25 + j) * HC_KS + z]; mix[j] = d * rs; }
    const float eps = v41::HC_EPS;
    for (int i = 0; i < 4; ++i) {
        pre[m * 4 + i] = 1.f / (1.f + expf(-(mix[i] * scale[0] + base[i]))) + eps;
        post[m * 4 + i] = 2.f / (1.f + expf(-(mix[4 + i] * scale[1] + base[4 + i])));
    }
    float c[4][4];
    for (int i = 0; i < 4; ++i) {
        float mx = -INFINITY;
        for (int k = 0; k < 4; ++k) { c[i][k] = mix[8 + 4 * i + k] * scale[2] + base[8 + 4 * i + k]; mx = fmaxf(mx, c[i][k]); }
        float s = 0.f; for (int k = 0; k < 4; ++k) { c[i][k] = expf(c[i][k] - mx); s += c[i][k]; }
        for (int k = 0; k < 4; ++k) c[i][k] = c[i][k] / s + eps;
    }
    auto colnorm = [&]() { for (int k = 0; k < 4; ++k) { float s = 0.f; for (int i = 0; i < 4; ++i) s += c[i][k];
                                                          for (int i = 0; i < 4; ++i) c[i][k] /= (s + eps); } };
    auto rownorm = [&]() { for (int i = 0; i < 4; ++i) { float s = 0.f; for (int k = 0; k < 4; ++k) s += c[i][k];
                                                          for (int k = 0; k < 4; ++k) c[i][k] /= (s + eps); } };
    colnorm();
    for (int it = 0; it < v41::HC_SINKHORN_ITERS - 1; ++it) { rownorm(); colnorm(); }
    for (int i = 0; i < 4; ++i) for (int k = 0; k < 4; ++k) comb[m * 16 + i * 4 + k] = c[i][k];
}
static float* g_hcpart = nullptr; static int g_hcpart_m = 0;
void hc_mixes(float* pre, float* post, float* comb, const float* x, const float* fn, const float* scale,
              const float* base, int M, cudaStream_t st) {
    if (M > g_hcpart_m) { fprintf(stderr, "hc_mixes: M=%d exceeds reserved %d (call hc_reserve)\n", M, g_hcpart_m); abort(); }
    k_hc_partial<<<dim3(M, 25, HC_KS), 256, 0, st>>>(g_hcpart, x, fn);
    k_hc_finish<<<M, 32, 0, st>>>(pre, post, comb, g_hcpart, scale, base);
}
void hc_reserve(int max_m) {
    if (max_m <= g_hcpart_m) return;
    if (g_hcpart) cudaFree(g_hcpart);
    cudaMalloc(&g_hcpart, (size_t)max_m * 25 * HC_KS * sizeof(float)); g_hcpart_m = max_m;
}
__global__ void k_hc_collapse(float* y, const float* x, const float* pre, int M) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i >= (size_t)M * v41::DIM) return;
    const int m = (int)(i / v41::DIM), d = (int)(i % v41::DIM);
    const float* xr = x + (size_t)m * v41::HC_DIM + d;
    float s = 0.f;
    for (int j = 0; j < 4; ++j) s += pre[m * 4 + j] * xr[(size_t)j * v41::DIM];
    y[i] = bf16r(s);
}
void hc_collapse(float* y, const float* x, const float* pre, int M, cudaStream_t st) {
    size_t n = (size_t)M * v41::DIM;
    k_hc_collapse<<<(unsigned)((n + 255) / 256), 256, 0, st>>>(y, x, pre, M);
}
// y[j] = post[j]*f + sum_i comb[i][j]*res[i]  (the reference sums over the FIRST comb index)
__global__ void k_hc_post(float* y, const float* f, const float* res, const float* post, const float* comb, int M) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i >= (size_t)M * v41::DIM) return;
    const int m = (int)(i / v41::DIM), d = (int)(i % v41::DIM);
    const float fv = f[i];
    const float* rr = res + (size_t)m * v41::HC_DIM + d;
    float r[4]; for (int k = 0; k < 4; ++k) r[k] = rr[(size_t)k * v41::DIM];
    for (int j = 0; j < 4; ++j) {
        float s = 0.f;
        for (int k = 0; k < 4; ++k) s += comb[m * 16 + k * 4 + j] * r[k];
        y[(size_t)m * v41::HC_DIM + (size_t)j * v41::DIM + d] = bf16r(post[m * 4 + j] * fv + s);
    }
}
void hc_post(float* y, const float* f, const float* res, const float* post, const float* comb, int M, cudaStream_t st) {
    size_t n = (size_t)M * v41::DIM;
    k_hc_post<<<(unsigned)((n + 255) / 256), 256, 0, st>>>(y, f, res, post, comb, M);
}
__global__ void k_hc_ident(float* pre, int M) { int i = threadIdx.x + blockIdx.x * blockDim.x; if (i < M * 4) pre[i] = (i % 4 == 0) ? 1.f : 0.f; }
void hc_identity_pre(float* pre, int M, cudaStream_t st) { k_hc_ident<<<(M * 4 + 127) / 128, 128, 0, st>>>(pre, M); }
__global__ void k_embed(float* h, const uint16_t* emb, const int* ids, int M) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i >= (size_t)M * v41::DIM) return;
    const int m = (int)(i / v41::DIM), d = (int)(i % v41::DIM);
    const float v = bf16f(emb[(size_t)ids[m] * v41::DIM + d]);
    for (int j = 0; j < 4; ++j) h[(size_t)m * v41::HC_DIM + (size_t)j * v41::DIM + d] = v;
}
void embed_expand(float* h, const uint16_t* emb, const int* d_ids, int M, cudaStream_t st) {
    size_t n = (size_t)M * v41::DIM;
    k_embed<<<(unsigned)((n + 255) / 256), 256, 0, st>>>(h, emb, d_ids, M);
}

// --------------------------------------------------------------------------- SwiGLU
// Expert.forward: gate = w1(x).float(); up = w3(x).float(); up clamp [-L,L]; gate clamp max L;
// silu(gate)*up (*weights) -> .to(bf16)
__global__ void k_swiglu(float* act, const float* gate, const float* up, const float* wr, int M, int inter, float L,
                         int ld_in) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i >= (size_t)M * inter) return;
    const int m = (int)(i / inter), k = (int)(i % inter);
    float g = gate[(size_t)m * ld_in + k], u = up[(size_t)m * ld_in + k];
    if (L > 0.f) { u = fminf(fmaxf(u, -L), L); g = fminf(g, L); }
    float a = g / (1.f + expf(-g)) * u;
    if (wr) a *= wr[m];
    act[i] = bf16r(a);
}
void swiglu(float* act, const float* gate, const float* up, const float* w_row, int M, int inter, float limit, int ld,
            cudaStream_t st) {
    size_t n = (size_t)M * inter;
    k_swiglu<<<(unsigned)((n + 255) / 256), 256, 0, st>>>(act, gate, up, w_row, M, inter, limit, ld);
}

} // namespace v41k
