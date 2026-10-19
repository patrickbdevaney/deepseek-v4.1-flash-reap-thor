// kt_viterbi.cu — the KT2/KT3 encoder's tail-biting trellis search on the GPU (s07 thor-trellis).
// Same contract as the numpy oracle ops/lib/kt.py::viterbi, and the SAME STATES: float32
// ((x - c)^2 * w, then m + that) with separate roundings, first-minimum tie rule everywhere.
// tests/gate_v41_units.cu holds it to the oracle's states on random sequences.
//
// One CTA (256 threads) per 128-weight sequence; the whole trellis state lives in shared memory:
//   cost[4096]            the running path cost of every 12-bit window
//   m[2^(L-b)]            min over the 2^b predecessors of each successor's low bits
//   bp[step][b][..]       argmin predecessor, as b bit-planes of warp ballots (32 KB at b=2)
// Thread t owns windows t + 256k (k < 16) for the whole run, so their codebook values sit in
// registers. Pass 1 runs the 64 columns around the wrap point unconstrained and reads the overlap
// bits at column 0; pass 2 runs all 128 columns with the start and end windows constrained to that
// overlap, so every output is a valid circular stream.
//
// Built as build/libkt_viterbi.so (scripts/build_v41.sh libkt_viterbi), called from Python through
// ctypes with torch device pointers: kt_viterbi(X, W, out, n, b, stream) -> cudaError_t.
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>

namespace {

constexpr int L = 12, NS = 1 << L, T = 128, HW = 32, NT = 256, SPT = NS / NT;

__device__ __forceinline__ float cb_value(uint32_t s) {            // 3INST, as kt_value in v41_gemv.cuh
    const uint32_t x = s * 89226354u + 64248484u;
    const uint32_t m = (x & 0x8fff8fffu) ^ 0x3b603b60u;
    return __fadd_rn(__half2float(__ushort_as_half((unsigned short)(m & 0xFFFFu))),
                     __half2float(__ushort_as_half((unsigned short)(m >> 16))));
}

template <int B>
struct Smem {
    static constexpr int NL = L - B, NH = 1 << NL, PW = NH / 32;
    float cost[NS];
    float m[NH];
    uint32_t bp[T][B][PW];
    float xs[T], ws[T];
    int path[T];
    float rv[NT];
    int ri[NT];
    int o;
};

template <int B>
__global__ void __launch_bounds__(NT) k_viterbi(const float* __restrict__ X, const float* __restrict__ Wt,
                                                int32_t* __restrict__ out) {
    using S = Smem<B>;
    constexpr int NL = S::NL, NH = S::NH, NP = 1 << B, HPT = NH / NT;
    constexpr uint32_t LOW = (1u << NL) - 1u;
    extern __shared__ __align__(16) unsigned char raw[];
    S& sm = *reinterpret_cast<S*>(raw);
    const size_t seq = blockIdx.x;
    const int t = threadIdx.x, lane = t & 31;
    if (t < T) { sm.xs[t] = X[seq * T + t]; sm.ws[t] = Wt[seq * T + t]; }
    float cb[SPT];
    #pragma unroll
    for (int k = 0; k < SPT; ++k) cb[k] = cb_value((uint32_t)(t + NT * k));
    __syncthreads();

    auto dist = [&](int col, int k) {
        const float d = __fsub_rn(sm.xs[col], cb[k]);
        return __fmul_rn(__fmul_rn(d, d), sm.ws[col]);
    };
    // one pass over `n` columns: col(k) = (start + k) mod 128. constrained -> start windows must have
    // low bits == sm.o. Leaves cost[] = final costs and bp[1..n-1].
    auto run = [&](int start, int n, bool constrained) {
        const int c0 = start % T;
        #pragma unroll
        for (int k = 0; k < SPT; ++k) {
            const uint32_t s = (uint32_t)(t + NT * k);
            const float v = dist(c0, k);
            sm.cost[s] = (constrained && (s & LOW) != (uint32_t)sm.o) ? INFINITY : v;
        }
        __syncthreads();
        for (int step = 1; step < n; ++step) {
            const int col = (start + step) % T;
            #pragma unroll
            for (int j = 0; j < HPT; ++j) {
                const int hi = t + NT * j;
                float best = sm.cost[hi * NP];
                int bi = 0;
                #pragma unroll
                for (int p = 1; p < NP; ++p) {
                    const float v = sm.cost[hi * NP + p];
                    if (v < best) { best = v; bi = p; }
                }
                sm.m[hi] = best;
                #pragma unroll
                for (int bt = 0; bt < B; ++bt) {
                    const uint32_t plane = __ballot_sync(0xffffffffu, (bi >> bt) & 1);
                    if (lane == 0) sm.bp[step][bt][(hi >> 5)] = plane;
                }
            }
            __syncthreads();
            #pragma unroll
            for (int k = 0; k < SPT; ++k) {
                const uint32_t s = (uint32_t)(t + NT * k);
                sm.cost[s] = __fadd_rn(sm.m[s & LOW], dist(col, k));
            }
            __syncthreads();
        }
    };
    auto pred = [&](int step, uint32_t s) -> uint32_t {     // predecessor window of window s at `step`
        const uint32_t low = s & LOW;
        uint32_t p = 0;
        #pragma unroll
        for (int bt = 0; bt < B; ++bt) p |= ((sm.bp[step][bt][low >> 5] >> (low & 31)) & 1u) << bt;
        return (low << B) | p;
    };

    // ---- pass 1: columns 96..127, 0..31, unconstrained; argmin over every window (lowest index on ties)
    run(T - HW, 2 * HW, false);
    {
        float bv = INFINITY; int bi = NS;
        #pragma unroll
        for (int k = 0; k < SPT; ++k) {
            const int s = t + NT * k;
            const float v = sm.cost[s];
            if (v < bv || (v == bv && s < bi)) { bv = v; bi = s; }
        }
        sm.rv[t] = bv; sm.ri[t] = bi;
        __syncthreads();
        for (int w = NT / 2; w > 0; w >>= 1) {
            if (t < w) {
                const float v2 = sm.rv[t + w]; const int i2 = sm.ri[t + w];
                if (v2 < sm.rv[t] || (v2 == sm.rv[t] && i2 < sm.ri[t])) { sm.rv[t] = v2; sm.ri[t] = i2; }
            }
            __syncthreads();
        }
        if (t == 0) {
            uint32_t s = (uint32_t)sm.ri[0];
            for (int step = 2 * HW - 1; step > HW; --step) s = pred(step, s);    // window at step HW = column 0
            sm.o = (int)(s & LOW);
        }
        __syncthreads();
    }
    // ---- pass 2: all 128 columns, start low bits == o, end high bits == o
    run(0, T, true);
    if (t == 0) {
        uint32_t s = ((uint32_t)sm.o << B);
        float bv = sm.cost[s];
        for (int p = 1; p < NP; ++p) {
            const float v = sm.cost[((uint32_t)sm.o << B) | p];
            if (v < bv) { bv = v; s = ((uint32_t)sm.o << B) | p; }
        }
        sm.path[T - 1] = (int)s;
        for (int step = T - 1; step > 0; --step) { s = pred(step, s); sm.path[step - 1] = (int)s; }
    }
    __syncthreads();
    if (t < T) out[seq * T + t] = sm.path[t];
}

template <int B>
cudaError_t launch(const float* X, const float* W, int32_t* out, int n, cudaStream_t st) {
    const int smem = (int)sizeof(Smem<B>);
    cudaError_t e = cudaFuncSetAttribute(k_viterbi<B>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    if (e != cudaSuccess) return e;
    for (int s0 = 0; s0 < n; s0 += 1 << 30) {              // grid.x limit with margin
        const int cnt = n - s0 < (1 << 30) ? n - s0 : (1 << 30);
        k_viterbi<B><<<cnt, NT, smem, st>>>(X + (size_t)s0 * T, W + (size_t)s0 * T, out + (size_t)s0 * T);
    }
    return cudaGetLastError();
}

}  // namespace

extern "C" int kt_viterbi(const float* X, const float* W, int32_t* out, int n, int b, void* stream) {
    if (n <= 0) return 0;
    const cudaStream_t st = (cudaStream_t)stream;
    const cudaError_t e = b == 2 ? launch<2>(X, W, out, n, st) : b == 3 ? launch<3>(X, W, out, n, st) : cudaErrorInvalidValue;
    return (int)e;
}
