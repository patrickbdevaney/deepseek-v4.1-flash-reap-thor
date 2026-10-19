// indexer.cu — DSA Indexer primitives, correctness-first (Gate K: ref/gen_units gen_hadamard/gen_index_score).
#include "indexer.h"
#include "topk_radix.h"
#include "dscratch.h"
#include <cstdlib>   // getenv/atoi for the 1.10 A/B arm

// Hadamard: y[r,j] = D^-0.5 * Σ_i x[r,i] * (-1)^popcount(i&j).
//
// DECODE_LADDER 1.10 -- THIS WAS THE RACE, and the flat kernel below is kept only as its A/B arm.
// Every thread of a row reads ALL D elements of that row and writes ONE of them, which is correct
// exactly while `y != x`. Three call sites pass the same pointer twice -- compressor.cu's `rotate`
// branch (both the prefill `hadamard(out,out,groups,d)` and the single-group emit) and
// compressed_decode.cu's candidate emit -- so a reading thread could see a neighbour's ALREADY
// TRANSFORMED value instead of the input, and which it saw was a scheduling outcome. Only the
// indexer's compressor sets `rotate`, and `indexer_forward` runs only on compress_ratio == 4
// layers, which is why all 56 of 1.9's first-differing prefill layers were ratio 4 and not one was
// ratio 128 (tools/lhash_pairs.py, DECODE_LADDER 1.10).
//
// THE STAGED KERNEL IS BIT-IDENTICAL WHERE IT WAS ALREADY CORRECT. One block owns one row, stages
// it in shared memory, and reduces out of shared -- so the k-order of the sum is still i = 0..D-1
// and every non-aliased caller gets the same float it got before. What changes is that the input a
// thread reduces can no longer be another thread's output.
//
// The old mapping raced at every length in principle and fired only past 20 blocks, because
// blockDim 256 / D 128 puts two rows in a block and while blocks <= SM count each block has an SM
// to itself and its warps stay in lockstep. That is the whole reason a defect this crude survived:
// tests/gate_scratch_init exercised prefills 1..29 (<= 4 blocks) and 1.9's ladder stopped at 160
// (exactly 20 blocks on this 20-SM box). tests/gate_hadamard_alias sweeps across the boundary.
// DSV4_HADAMARD_STAGE=0 restores the pre-1.10 flat kernel on the same binary. `hadamard_set_stage`
// exists next to it for the same reason `gemm_fp32_set_tile` and `index_score_impl` do: the env is
// read through a function-local static, so a process that only had the variable could not flip arms,
// and a gate that has to fork per arm is a gate that ends up comparing two different builds.
static int g_hadamard_stage = -1;
void hadamard_set_stage(int on){ g_hadamard_stage = on ? 1 : 0; }
static bool hadamard_stage_on(){
    if (g_hadamard_stage >= 0) return g_hadamard_stage != 0;
    static const int env = [](){ const char* e = getenv("DSV4_HADAMARD_STAGE"); return (!e || atoi(e) != 0) ? 1 : 0; }();
    return env != 0;
}
// NO `__restrict__` on y and x: they legitimately alias here, and telling the compiler otherwise is
// how a correct kernel becomes an incorrect one at -O3.
__global__ void hadamard_stage_kernel(float* y, const float* x, int rows, int D, float scale) {
    extern __shared__ float sx[];
    const int r = blockIdx.x; if (r >= rows) return;
    const float* xr = x + (size_t)r * D;
    for (int i = threadIdx.x; i < D; i += blockDim.x) sx[i] = xr[i];
    __syncthreads();                                   // every read of x is now behind this barrier
    float* yr = y + (size_t)r * D;
    for (int j = threadIdx.x; j < D; j += blockDim.x) {
        float acc = 0.f;
        for (int i = 0; i < D; ++i) acc += (__popc(i & j) & 1) ? -sx[i] : sx[i];
        yr[j] = acc * scale;
    }
}
__global__ void hadamard_kernel(float* __restrict__ y, const float* __restrict__ x, int rows, int D, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= rows * D) return;
    int r = idx / D, j = idx % D;
    const float* xr = x + (size_t)r * D;
    float acc = 0.f;
    for (int i = 0; i < D; ++i) acc += (__popc(i & j) & 1) ? -xr[i] : xr[i];
    y[idx] = acc * scale;
}
void hadamard(float* y, const float* x, int rows, int D, cudaStream_t stream) {
    // LOOP_LOG Finding 53's failure mode, avoided rather than reproduced: rows == 0 would launch
    // with gridDim 0, which fails and leaves cudaErrorInvalidValue in the thread's last-error slot
    // for the next unrelated check to find. There is nothing to transform.
    if (rows <= 0 || D <= 0) return;
    float scale = rsqrtf((float)D);
    const size_t smem = (size_t)D * sizeof(float);
    // 48 KiB is the launch default; D would have to exceed 12,288 to miss it and the largest D any
    // caller passes is INDEX_HEAD_DIM = 128. Falling back rather than requesting an opt-in carveout
    // keeps this off the CUDA-graph capture paths (item 1.4's reason), and the fallback is only
    // reachable at a D no call site has.
    if (hadamard_stage_on() && smem <= 48u * 1024u) {
        const int threads = D < 32 ? 32 : (D > 256 ? 256 : D);
        hadamard_stage_kernel<<<rows, threads, smem, stream>>>(y, x, rows, D, scale);
        return;
    }
    hadamard_kernel<<<(rows * D + 255) / 256, 256, 0, stream>>>(y, x, rows, D, scale);
}

// index_score[s,t] = Σ_h relu(Σ_d q[s,h,d]*kv[t,d]) * weights[s,h]. One thread per (s,t).
__global__ void index_score_kernel(float* __restrict__ score, const float* __restrict__ q,
                                   const float* __restrict__ kv, const float* __restrict__ weights,
                                   int S, int T, int H, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= S * T) return;
    int s = idx / T, t = idx % T;
    const float* kvt = kv + (size_t)t * d;
    float acc = 0.f;
    for (int h = 0; h < H; ++h) {
        const float* qh = q + (((size_t)s * H + h) * d);
        float dot = 0.f; for (int e = 0; e < d; ++e) dot += qh[e] * kvt[e];
        acc += fmaxf(dot, 0.f) * weights[(size_t)s * H + h];      // relu * head weight
    }
    score[(size_t)s * T + t] = acc;
}
// WARP PER OUTPUT (LOOP_LOG Finding 71). The kernel above puts one THREAD on each (query, row) pair,
// and at decode there are only S*T ~ 95 of them — a single block, three warps, one SM, each thread
// walking H*d = 1024 MACs serially with a stride-1 read that no other lane shares. Measured in situ
// it is 6.05 ms of the indexer's 9.14, i.e. 4.2% of the whole K=5 verify, for 97k MACs.
//
// One warp per pair instead: 32x the threads, the d-loop is lane-strided so consecutive lanes read
// consecutive floats of both operands, and the per-head dot finishes in a shuffle tree.
//
// NOT bit-exact: the dot over `d` changes from serial to tree order. That is why it ships behind the
// LOSSLESS gate (Finding 68) rather than a cosine — a tolerance that is fine for one dot product says
// nothing about what a different top-k selection does 43 layers later. NO_IXWARP=1 restores the
// scalar kernel for A/B.
__global__ void index_score_warp_kernel(float* __restrict__ score, const float* __restrict__ q,
                                        const float* __restrict__ kv, const float* __restrict__ weights,
                                        int S, int T, int H, int d) {
    const int gid = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (gid >= S * T) return;
    const int s = gid / T, t = gid % T, lane = threadIdx.x & 31;
    const float* kvt = kv + (size_t)t * d;
    float acc = 0.f;
    for (int h = 0; h < H; ++h) {
        const float* qh = q + (((size_t)s * H + h) * d);
        float dot = 0.f;
        for (int e = lane; e < d; e += 32) dot += qh[e] * kvt[e];
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) dot += __shfl_down_sync(0xffffffff, dot, o);
        dot = __shfl_sync(0xffffffff, dot, 0);
        acc += fmaxf(dot, 0.f) * weights[(size_t)s * H + h];   // same relu, same head order
    }
    if (lane == 0) score[(size_t)s * T + t] = acc;
}
// ================= DECODE_LADDER 1.5 — the same arithmetic, off a different memory hierarchy =====
//
// WHAT WAS WRONG WITH THE KERNEL ABOVE, precisely. `index_score_warp_kernel` is warp-per-(query,row)
// and its inner body reads BOTH operands from global on every head:
//
//     for h in 0..H:  for e = lane; e < d; e += 32:  dot += qh[e] * kvt[e];
//
// `q` for one query is H*d = 64*128 = 8192 floats = 32 KiB — the whole thing — and it is re-read
// once per row `t`. `kv[t]` is d = 128 floats and it is re-read once per HEAD. So the kernel moves
// 2*S*T*H*d*4 bytes = 1.2 GB per call at the verify shape (S=6, T=3072) to do 151 M MACs, an
// arithmetic intensity of 0.5 FLOP/byte on a machine whose FFMA peak is 5.45 TFLOPS. It also cannot
// unroll: `d` is a runtime argument, so the e-loop is a serial chain of two dependent global loads.
// That is the whole of the 6.58 ms `i:score` costs at ctx 12,288 (ladder 0.4).
//
// THE FIX IS MEMORY PLACEMENT, NOT MATHEMATICS, AND THAT IS DELIBERATE. Both re-reads are removed:
//   * `q` for this query is staged ONCE per block into shared memory (33 KiB incl. the head
//     weights) and read from there by every row the block owns;
//   * `kv[t]` is held in REGISTERS by the warp that owns row t (d/32 = 4 floats per lane) and read
//     once, not H times;
//   * `d` becomes a template parameter so the e-loop unrolls into independent FFMAs.
//
// THE ARITHMETIC IS BYTE-IDENTICAL TO THE WARP KERNEL, ON PURPOSE. Same lane->element mapping
// (lane, lane+32, lane+64, ...), same serial accumulation into `dot` in that order, same 5-step
// `__shfl_down_sync` tree, same `fmaxf` then same serial accumulation over h in h order. Nothing is
// reassociated, so this is a memcmp claim and not a cosine claim — which matters here because
// LOOP_LOG Finding 68 is exactly the case of a reduction-order change to THIS kernel that a
// tolerance gate would have passed and that moves which rows the top-k selects 43 layers later.
// tests/gate_index_score.cu is the memcmp. NO_IXTILE=1 restores the warp kernel for A/B.
//
// The broadcast `__shfl_sync(...,0)` that the warp kernel does after its tree is dropped: only
// lane 0 stores, lane 0's tree result is already the full sum, and every lane still executes the
// tree so the warp stays converged.
template<int D>
__global__ __launch_bounds__(256) void index_score_tiled_kernel(
        float* __restrict__ score, const float* __restrict__ q, const float* __restrict__ kv,
        const float* __restrict__ weights, int T, int H) {
    extern __shared__ float smem[];
    float* __restrict__ sq = smem;                  // [H*D] this query's heads
    float* __restrict__ sw = smem + (size_t)H * D;  // [H]   this query's head weights
    const int s = blockIdx.y;
    {   const float* qs = q + (size_t)s * H * D;
        for (int i = threadIdx.x; i < H * D; i += blockDim.x) sq[i] = qs[i];
        for (int i = threadIdx.x; i < H;     i += blockDim.x) sw[i] = weights[(size_t)s * H + i];
    }
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    for (int t = blockIdx.x * nw + warp; t < T; t += gridDim.x * nw) {
        const float* kvt = kv + (size_t)t * D;
        float kr[D / 32];
        #pragma unroll
        for (int j = 0; j < D / 32; ++j) kr[j] = kvt[lane + 32 * j];
        float acc = 0.f;
        for (int h = 0; h < H; ++h) {
            const float* qh = sq + h * D;
            float dot = 0.f;
            #pragma unroll
            for (int j = 0; j < D / 32; ++j) dot += qh[lane + 32 * j] * kr[j];   // same order as e=lane,lane+32,...
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) dot += __shfl_down_sync(0xffffffff, dot, o);
            acc += fmaxf(dot, 0.f) * sw[h];       // same relu, same head order
        }
        if (lane == 0) score[(size_t)s * T + t] = acc;
    }
}

// ================= DECODE_LADDER 1.5 — the GEMM, and the reference order it restores ==========
//
// THE TILED KERNEL ABOVE IS BIT-EXACT WITH THE WARP KERNEL AND THAT IS WHAT CAPS IT AT 2x. Its
// per-(row,head) cost is 4 FFMA of useful work against ~16 instructions of overhead, and half that
// overhead is the 5-step `__shfl_down_sync` tree. SHFL retires at one warp-instruction per SM per
// clock on this part, so 1.18 M (row,head) pairs x 5 SHFL over 20 SMs is a ~200 us floor at the
// verify shape no matter how the operands are staged — and the measured tiled kernel is 451 us
// against exactly that arithmetic. The tree cannot be removed while the claim is "bit-identical to
// the warp kernel", because the tree IS the warp kernel's summation order.
//
// SO CHANGE WHICH KERNEL THE CLAIM IS AGAINST — TOWARDS THE REFERENCE, NOT AWAY FROM IT.
// `index_score_kernel` at the top of this file is the correctness-first scalar version that
// `tests/gate_units.cu` checks against `ref/goldens/unit_index_score.safetensors`. Its order is
// SERIAL over d:  dot += qh[0]*kvt[0]; dot += qh[1]*kvt[1]; ...  That is precisely the order a
// register-tiled GEMM accumulates in, so a GEMM can be BIT-IDENTICAL to the reference while the
// shipped warp kernel is not. LOOP_LOG Finding 68 records that the warp kernel was itself adopted
// as a deviation from this order, behind the LOSSLESS gate; 1.5 spends that deviation back.
//
// SHAPE. score[s,t] = sum_h relu(q[s,h,:] . kv[t,:]) * w[s,h] is a GEMM P[(s,h),t] = q . kv^T with
// M = H = 64, N = T, K = d = 128, plus a reduction over M in the epilogue. One block owns ALL H
// heads of one query s and IXG_BT = 128 rows, so:
//   * the k-loop is 8x8 register-tiled — 16 shared loads feed 64 FFMAs, ~4:1, against the tiled
//     kernel's 1:1 — and accumulates serially in k, chunk by chunk, which preserves the reference
//     order exactly;
//   * P for the whole tile goes to shared (H x IXG_BT floats), and the epilogue then walks
//     h = 0..H-1 IN ORDER for each row, which preserves the reference's serial `acc +=` over heads.
//     Reducing over h in registers instead would be 8x faster and WRONG: thread-local partials
//     recombined pairwise are not the same float as a serial sum, and this file's whole history is
//     that such a difference relocates the top-k boundary.
// The register tiling is STRIDED, not blocked (t = tx + j*16, h = ty + i*8): consecutive threads
// then read consecutive shared floats with plain scalar loads and no bank conflict, which is what
// a blocked mapping would need float4 and padding games to get.
//
// NOT bit-exact with the shipped warp kernel — bit-exact with the SCALAR reference. Both facts are
// measured, both ways, by tests/gate_index_score.cu. NO_IXGEMM=1 falls back to the tiled kernel.
#define IXG_BT 128        // rows of kv per block
#define IXG_KC  32        // k staged per pass; d % IXG_KC == 0 required
#define IXG_TH   8        // heads per thread
#define IXG_TT   8        // rows per thread
__global__ __launch_bounds__(128) void index_score_gemm_kernel(
        float* __restrict__ score, const float* __restrict__ q, const float* __restrict__ kv,
        const float* __restrict__ weights, int T, int H, int d) {
    extern __shared__ float sm[];
    const int SQS = H + 1, SKS = IXG_BT + 1, SPS = IXG_BT + 1;   // +1 = bank-conflict padding
    float* __restrict__ sq  = sm;                                 // [IXG_KC][H+1]   staged q^T
    float* __restrict__ skv = sm + (size_t)IXG_KC * SQS;          // [IXG_KC][BT+1]  staged kv^T
    const int s = blockIdx.y, t0 = blockIdx.x * IXG_BT, nth = blockDim.x;
    const int NTX = IXG_BT / IXG_TT;                              // 16
    const int tx = threadIdx.x % NTX, ty = threadIdx.x / NTX;
    const int NTY = H / IXG_TH;
    float c[IXG_TH][IXG_TT];
    #pragma unroll
    for (int i = 0; i < IXG_TH; ++i)
        #pragma unroll
        for (int j = 0; j < IXG_TT; ++j) c[i][j] = 0.f;
    const float* qs = q + (size_t)s * H * d;
    for (int k0 = 0; k0 < d; k0 += IXG_KC) {
        __syncthreads();
        // COALESCED in global (consecutive threads walk k), CONFLICT-FREE in shared (the +1 pad).
        for (int i = threadIdx.x; i < H * IXG_KC; i += nth) {
            const int h = i / IXG_KC, kk = i % IXG_KC;
            sq[kk * SQS + h] = qs[(size_t)h * d + k0 + kk];
        }
        for (int i = threadIdx.x; i < IXG_BT * IXG_KC; i += nth) {
            const int tt = i / IXG_KC, kk = i % IXG_KC, t = t0 + tt;
            skv[kk * SKS + tt] = (t < T) ? kv[(size_t)t * d + k0 + kk] : 0.f;
        }
        __syncthreads();
        #pragma unroll 8
        for (int kk = 0; kk < IXG_KC; ++kk) {
            float a[IXG_TH], b[IXG_TT];
            #pragma unroll
            for (int i = 0; i < IXG_TH; ++i) a[i] = sq [kk * SQS + ty + i * NTY];
            #pragma unroll
            for (int j = 0; j < IXG_TT; ++j) b[j] = skv[kk * SKS + tx + j * NTX];
            #pragma unroll
            for (int i = 0; i < IXG_TH; ++i)
                #pragma unroll
                for (int j = 0; j < IXG_TT; ++j) c[i][j] += a[i] * b[j];   // serial in k == reference
            }
    }
    __syncthreads();
    float* __restrict__ sP = sm;                                  // [H][BT+1], overlays the staging
    #pragma unroll
    for (int i = 0; i < IXG_TH; ++i)
        #pragma unroll
        for (int j = 0; j < IXG_TT; ++j) sP[(size_t)(ty + i * NTY) * SPS + tx + j * NTX] = c[i][j];
    __syncthreads();
    const float* ws = weights + (size_t)s * H;
    for (int tt = threadIdx.x; tt < IXG_BT; tt += nth) {
        const int t = t0 + tt; if (t >= T) continue;
        float acc = 0.f;
        for (int h = 0; h < H; ++h) acc += fmaxf(sP[(size_t)h * SPS + tt], 0.f) * ws[h];  // serial in h
        score[(size_t)s * T + t] = acc;
    }
}

static bool ixgemm_launch(float* score, const float* q, const float* kv, const float* weights,
                          int S, int T, int H, int d, cudaStream_t stream) {
    if ((d % IXG_KC) != 0 || (H % IXG_TH) != 0 || H <= 0) return false;
    const int nth = (H / IXG_TH) * (IXG_BT / IXG_TT);
    if (nth > 1024) return false;
    const size_t stage = (size_t)IXG_KC * (H + 1) + (size_t)IXG_KC * (IXG_BT + 1);
    const size_t pbuf  = (size_t)H * (IXG_BT + 1);
    const size_t bytes = (stage > pbuf ? stage : pbuf) * sizeof(float);
    if (bytes > 48 * 1024) return false;
    dim3 grid((unsigned)((T + IXG_BT - 1) / IXG_BT), (unsigned)S, 1);
    index_score_gemm_kernel<<<grid, nth, bytes, stream>>>(score, q, kv, weights, T, H, d);
    return true;
}

// Grid shape. Every block pays for one 32 KiB stage of `q`, so the block count is a cost, not just a
// parallelism knob: it is capped near one full occupancy wave (7 blocks/SM is what 33 KiB of shared
// out of 228 KiB allows) and the rows are handed out grid-stride inside that. DSV4_IXTILE_BLK
// overrides the total for a sweep.
//
// This is the ONLY CUDA API call on any index_score launch path, and it is on the TILED path, which
// the engine never takes (it needs H % 8 != 0 or d % 32 != 0, and the model is H=64 d=128). That
// matters because `compressed_decode_step_indexer` is CUDA-graph captured: `ixgemm_launch` issues a
// kernel launch and nothing else, so the shipped path is capture-safe by construction rather than
// by testing. gate_indexer_graph / gate_compressed_graph are the tests anyway.
static inline int ixtile_total_blocks() {
    static const int tb = [](){
        const char* e = getenv("DSV4_IXTILE_BLK");
        if (e && atoi(e) > 0) return atoi(e);
        int sm = 20; cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, 0);
        return 7 * sm;
    }();
    return tb;
}

template<int D>
static bool ixtile_launch(float* score, const float* q, const float* kv, const float* weights,
                          int S, int T, int H, cudaStream_t stream) {
    const size_t bytes = ((size_t)H * D + H) * sizeof(float);
    if (bytes > 48 * 1024) return false;            // stay under the default per-block ceiling
    const int threads = 256, nw = threads >> 5;
    const int need = (T + nw - 1) / nw;
    int gx = ixtile_total_blocks() / (S > 0 ? S : 1); if (gx < 1) gx = 1; if (gx > need) gx = need;
    dim3 grid((unsigned)gx, (unsigned)S, 1);
    index_score_tiled_kernel<D><<<grid, threads, bytes, stream>>>(score, q, kv, weights, T, H);
    return true;
}

// ONE dispatch, THREE implementations, and the gate drives this rather than the env vars — because
// `NO_IXWARP`/`NO_IXTILE` are read through function-local statics, so a single process cannot flip
// arms and a gate that has to fork per shape is a gate nobody runs. Returns false if `impl` cannot
// serve this shape, which is how index_score below falls back rather than silently doing nothing.
bool index_score_impl(int impl, float* score, const float* q, const float* kv, const float* weights,
                      int S, int T, int H, int d, cudaStream_t stream) {
    if (S <= 0 || T <= 0 || H <= 0 || d <= 0) return false;
    if (impl == IXS_GEMM) return ixgemm_launch(score, q, kv, weights, S, T, H, d, stream);
    if (impl == IXS_TILED) {
        // Only the head dims this engine actually issues are templated; anything else reports
        // false and the caller falls back, rather than silently taking a different code path.
        if (d == 128) return ixtile_launch<128>(score, q, kv, weights, S, T, H, stream);
        if (d ==  64) return ixtile_launch< 64>(score, q, kv, weights, S, T, H, stream);
        if (d ==  32) return ixtile_launch< 32>(score, q, kv, weights, S, T, H, stream);
        if (d == 256) return ixtile_launch<256>(score, q, kv, weights, S, T, H, stream);
        return false;
    }
    if (impl == IXS_WARP) {
        if ((d % 32) != 0) return false;
        const int threads = 256, wpb = threads >> 5;
        index_score_warp_kernel<<<((size_t)S * T + wpb - 1) / wpb, threads, 0, stream>>>(score, q, kv, weights, S, T, H, d);
        return true;
    }
    index_score_kernel<<<((size_t)S * T + 255) / 256, 256, 0, stream>>>(score, q, kv, weights, S, T, H, d);
    return true;
}

void index_score(float* score, const float* q, const float* kv, const float* weights,
                 int S, int T, int H, int d, cudaStream_t stream) {
    static const bool warpk = getenv("NO_IXWARP") == nullptr;
    static const bool tilek = getenv("NO_IXTILE") == nullptr;
    static const bool gemmk = getenv("NO_IXGEMM") == nullptr;
    if (warpk && tilek && gemmk && index_score_impl(IXS_GEMM, score, q, kv, weights, S, T, H, d, stream)) return;
    if (warpk && tilek && index_score_impl(IXS_TILED, score, q, kv, weights, S, T, H, d, stream)) return;
    if (warpk && index_score_impl(IXS_WARP, score, q, kv, weights, S, T, H, d, stream)) return;
    index_score_impl(IXS_SCALAR, score, q, kv, weights, S, T, H, d, stream);
}

// ================= DSA Indexer forward =================
#include "fp8_block_gemm.h"
#include "mla_attn.h"      // act_quant_fp8, rope_interleaved, act_quant_fp4sim
#include "compressor.h"    // gemm_fp32, compressor_forward
#include <cstdio>
#define CUI(x) do{cudaError_t e=(x); if(e){fprintf(stderr,"cuda %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)

// N1, ADOPTED (default ON; NO_ZERO_SCRATCH=1 disables). Finding 60: the engine is nondeterministic on identical input
// and DSV4_ARENA_ZERO exonerated the arena. This is the analogue the arena test could not reach —
// the PREFILL path allocates its scratch with raw cudaMalloc, once per layer per call, and never
// zeroes it, so every buffer starts life holding whatever the allocator last had at that address.
// Any kernel here that accumulates into its output rather than assigning it, or that writes fewer
// rows than it later reads, would inherit that and produce different numbers run to run.
//
// CLAIM RETRACTED (Finding 61). This was adopted on "zeroing moves distinct first-verify margin
// vectors from 8/8 to 5/8", read as evidence of an uninitialised read. tests/gate_scratch_init then
// tested the thing directly — same weights, same input, scratch filled with 0x00 vs 0xFF vs 0x3C,
// arena included — and compressed_attn_forward is bitwise IDENTICAL at every length 1..29. The
// prefill chain does not read uninitialised scratch AT THOSE LENGTHS, and that qualifier is
// load-bearing: ladder 1.9 measured this function's run-to-run reproducibility as a function of
// prefill length and found it byte-identical to 160 positions and NONDETERMINISTIC from 192 up, to
// 3071. The gate's range stops six times short of where the defect lives, so it is evidence about
// 1..29 and nothing else. 1.9 also re-ran the 128..256 ladder with DSV4_ARENA_ZERO=1 on both arms
// (unchanged verdict) and four times inside ONE process (four different answers), so the residual
// is a RACE in this file's call graph, not uninitialised memory. See DECODE_LADDER.md item 1.10 and
// wiki/measurement-and-traps.md §25. The 8/8 -> 5/8 was a 5-cycle in the engine
// (Finding 61) sampled 8 times, not a change caused by zeroing: with the sharper instrument the
// engine produces the SAME hash sequence zeroed and unzeroed.
//
// Kept ON anyway, and only because it is free: decode runs entirely out of the arena and never
// touches these allocations, so this costs the measured number nothing and removes a whole class of
// future doubt. It is NOT a fix for anything, and nothing should be attributed to it.
static inline cudaError_t zalloc(void** p, size_t n){
    static const bool z = getenv("NO_ZERO_SCRATCH") == nullptr;   // DEFAULT ON, see below
    cudaError_t e = cudaMalloc(p, n);
    if(e != cudaSuccess || !n) return e;
    const int idx = g_scratch_alloc_seq++;
    if(idx == 0) g_scratch_first_addr = (unsigned long long)*p;
    if(g_scratch_poison_idx == -1 || g_scratch_poison_idx == idx) cudaMemset(*p, g_scratch_poison_val, n);
    else if(z)                                                    cudaMemset(*p, 0, n);
    return e;
}

__global__ void k_scale(float* y, float sc, int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) y[i]*=sc; }

// causal mask: score[si,t] = -inf where t >= (si+1)/ratio.
__global__ void k_causal_mask(float* score, int s, int T, int ratio){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=s*T) return; int si=i/T, t=i%T;
    if (t >= (si+1)/ratio) score[i] = -1e30f;
}
// per query: descending top-k of score[si,:T]; then idx = (t >= (si+1)/ratio) ? -1 : t+offset.
__global__ void k_topk_offset(int* out, const float* score, int s, int T, int topk, int ratio, int offset){
    int si=blockIdx.x; if(si>=s) return;
    extern __shared__ float sh[]; const int L=threadIdx.x;
    for(int t=L;t<T;t+=32) sh[t]=score[(size_t)si*T+t];
    __syncwarp();
    int thr=(si+1)/ratio;
    for(int k=0;k<topk;++k){
        float best=-1e30f; int bi=T;
        // NOTE the asymmetry, preserved verbatim: the scan covers the FULL row, and out-of-range
        // picks are rejected only at the OUTPUT. They still consume a slot. Bounding the scan by
        // `thr` here would change which rows land in later slots.
        for(int t=L;t<T;t+=32) if(sh[t]>best){best=sh[t];bi=t;}
        warp_argmax(best,bi);
        if(L==0){
            if(bi<T) sh[bi]=-1e30f;
            out[(size_t)si*topk+k] = (bi>=T || bi>=thr) ? -1 : bi+offset;
        }
        __syncwarp();
    }
}

// Radix-select twin (item 1.2). The asymmetry above is preserved: the SELECTION covers the full row
// and out-of-range picks are rejected only at the OUTPUT, still consuming a slot.
__global__ void k_topk_offset_rx(int* out, const float* score, int s, int T, int topk, int ratio, int offset, bool early){
    int si=blockIdx.x; if(si>=s) return;
    __shared__ TopkRadixSmem S;
    int* o = out + (size_t)si*topk;
    topk_radix_select<TOPK_RADIX_NT>(o, score+(size_t)si*T, T, topk, -1e30f, S, early);
    const int thr=(si+1)/ratio;
    for(int k=threadIdx.x;k<topk;k+=TOPK_RADIX_NT){ int b=o[k]; o[k] = (b<0 || b>=thr)? -1 : b+offset; }
}

void indexer_forward(float* index_score_out, int* topk_idxs, const float* x, const float* qr,
                     const unsigned char* wq_b, const float* wq_b_s, const float* weights_proj,
                     const float* c_wkv, const float* c_wgate, const float* c_ape, const float* c_norm,
                     const float* q_cos, const float* q_sin, const float* c_cos, const float* c_sin,
                     int s, int dim, int q_lora, int n_heads, int idx_hd, int rd, int ratio,
                     int index_topk, int offset, float eps, cudaStream_t stream) {
    int T = s / ratio, QD = n_heads * idx_hd;
    // Same defect as compressor_forward's (LOOP_LOG Finding 53), one layer up: with T = s/ratio == 0
    // there is no compressed row to score, both outputs are empty ([s,0] and [s,min(topk,0)]), and
    // index_score / k_causal_mask were being launched with gridDim (s*0+255)/256 = 0 — a launch that
    // fails and leaves cudaErrorInvalidValue behind. Reached at prompt lengths <= ratio.
    if (T <= 0) return;
    float softmax_scale = rsqrtf((float)idx_hd), wscale = softmax_scale * rsqrtf((float)n_heads);
    unsigned char* qrq; float *qrs, *q, *qtmp, *ckv, *weights;
    CUI(zalloc((void**)&qrq,(size_t)s*q_lora)); CUI(zalloc((void**)&qrs,(size_t)s*(q_lora/128)*4));
    CUI(zalloc((void**)&q,(size_t)s*QD*4)); CUI(zalloc((void**)&qtmp,(size_t)s*QD*4));
    CUI(zalloc((void**)&ckv,(size_t)T*idx_hd*4)); CUI(zalloc((void**)&weights,(size_t)s*n_heads*4));

    act_quant_fp8(qrq, qrs, qr, s, q_lora, 128, stream); dprobe(stream);
    fp8_block_gemm(q, qrq, qrs, wq_b, wq_b_s, s, QD, q_lora, stream); dprobe(stream);              // [s, n_heads*idx_hd]
    rope_interleaved(q + (idx_hd - rd), q_cos, q_sin, s*n_heads, rd, false, idx_hd, n_heads, stream); dprobe(stream);
    hadamard(qtmp, q, s*n_heads, idx_hd, stream); dprobe(stream);                                 // out!=in
    act_quant_fp4sim(qtmp, s*n_heads, idx_hd, 32, idx_hd, stream); dprobe(stream);                // fp4-sim
    compressor_forward(ckv, x, c_wkv, c_wgate, c_ape, c_norm, c_cos, c_sin, s, dim, idx_hd, ratio, true, rd, eps, true, stream); dprobe(stream);
    gemm_fp32(weights, x, weights_proj, s, n_heads, dim, stream); dprobe(stream);
    k_scale<<<(s*n_heads+255)/256,256,0,stream>>>(weights, wscale, s*n_heads); dprobe(stream);
    index_score(index_score_out, qtmp, ckv, weights, s, T, n_heads, idx_hd, stream); dprobe(stream);
    k_causal_mask<<<(s*T+255)/256,256,0,stream>>>(index_score_out, s, T, ratio); dprobe(stream);
    int topk = index_topk < T ? index_topk : T;
    if(topk_radix_on() && topk<=TOPK_RADIX_CAP) k_topk_offset_rx<<<s, TOPK_RADIX_NT, 0, stream>>>(topk_idxs, index_score_out, s, T, topk, ratio, offset, topk_early_on());
    else{ k_topk_offset<<<s, 32, TOPK_SMEM(k_topk_offset, T), stream>>>(topk_idxs, index_score_out, s, T, topk, ratio, offset);
          TOPK_LAUNCHED(k_topk_offset, T); }
    dprobe(stream);
    CUI(cudaStreamSynchronize(stream));
    cudaFree(qrq);cudaFree(qrs);cudaFree(q);cudaFree(qtmp);cudaFree(ckv);cudaFree(weights);
}
