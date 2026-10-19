// fp8_block_gemm.cu — FP8 e4m3 block-scaled GEMM, correctness-first (Gate K oracle: ref/gen_units.py).
// One warp per output element C[m,n]; the warp reduces over K, decoding e4m3 via the HW intrinsic and
// applying per-128-block activation and weight scales. Matches kernel.py fp8_gemm math (fp32 accumulate).
// Optimization (mma.sync tiling) comes AFTER this passes its gate — never before (CONSTITUTION Art. I).
#include "fp8_block_gemm.h"
#include "nvfp4_dense.h"
#include <cuda_fp8.h>

#define BLK 128   // scale block along K (weight_block_size)

__device__ __forceinline__ float dec_e4m3(uint8_t b) {
    __half_raw r = __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)b, __NV_E4M3);
    return __half2float(*reinterpret_cast<__half*>(&r));
}

// grid.x = N, grid.y = M ; blockDim.x = 32 (one warp -> one output C[by, bx]).
__global__ void fp8_block_gemm_kernel(float* __restrict__ C,
                                      const uint8_t* __restrict__ A, const float* __restrict__ a_s,
                                      const uint8_t* __restrict__ B, const float* __restrict__ b_s,
                                      int M, int N, int K) {
    int n = blockIdx.x, m = blockIdx.y;
    if (m >= M || n >= N) return;
    int lane = threadIdx.x & 31;
    int KB = K / BLK;                                  // #K-blocks
    const uint8_t* Arow = A + (size_t)m * K;
    const uint8_t* Brow = B + (size_t)n * K;
    const float*  asr = a_s + (size_t)m * KB;
    const float*  bsr = b_s + (size_t)(n / BLK) * KB;

    float acc = 0.f;
    // walk K-blocks; within a block the two scales are constant, so accumulate the raw dot then scale once.
    for (int kb = 0; kb < KB; ++kb) {
        float sub = 0.f;
        int base = kb * BLK;
        for (int j = lane; j < BLK; j += 32)
            sub += dec_e4m3(Arow[base + j]) * dec_e4m3(Brow[base + j]);
        acc += sub * asr[kb] * bsr[kb];
    }
    // warp reduce
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffff, acc, o);
    if (lane == 0) C[(size_t)m * N + n] = acc;
}

// M=1 GEMV: one warp per output n. Lanes read the B[n] row uint-vectorized (4 fp8/load), coalesced (32 lanes
// = 128 contiguous bytes); per-128-block scales applied per element; single warp-reduce. Bandwidth-bound —
// beats the m16-tile TC at M=1 (which is mma-latency bound). Gated cosine vs fp8_block_gemm (tests/gate_fp8_gemv).
// GRID-STRIDE (LOOP_LOG Finding 20). This used to launch exactly ceil(N*32/256) blocks — one warp
// per output row, grid sized by the problem. At the small-N shapes that is a fractional number of
// waves, and the tail wave costs almost a full kernel duration for a sliver of the work:
//   wkv  N=512   ->   64 blocks = 0.53 waves -> 54% of achievable
//   wq_a N=1024  ->  128 blocks = 1.07 waves -> 19%   (ncu: "up to 50.0% of the total runtime")
//   wo_b N=4096  ->  512 blocks = 4.3  waves -> 94%
//   wq_b N=32768 -> 4096 blocks = 34   waves -> 89%
// Efficiency tracks wave count, not shape. Now the grid is capped at exactly the number of blocks
// the device can hold resident, and each warp strides over its share of rows — so the launch is a
// whole number of waves by construction and the tail disappears. Math per row is unchanged, so this
// gates cosine-1.0 (tests/gate_fp8_gemv).
__global__ void fp8_gemv_m1_kernel(float* __restrict__ C, const uint8_t* __restrict__ A, const float* __restrict__ as,
                                   const uint8_t* __restrict__ B, const float* __restrict__ bs, int N, int K){
    const int lane = threadIdx.x & 31; const int KB = K/128;
    const int warp0  = (blockIdx.x*blockDim.x + threadIdx.x) >> 5;
    const int stride = (gridDim.x*blockDim.x) >> 5;
    for (int n = warp0; n < N; n += stride) {
        const uint8_t* Brow = B + (size_t)n*K; const float* bsr = bs + (size_t)(n/128)*KB;
        float acc = 0.f;
        // MEMORY-LEVEL PARALLELISM (research/GB10_COMPARISON.md). The original loop issued ONE
        // 4-byte load per iteration and consumed it immediately — ILP = 1. Measured on this box:
        // a streaming loop at ILP=1 sustains 110-132 GB/s; at ILP>=2 it sustains 224-237. Our whole
        // engine measured 116.6 GB/s, which is squarely the ILP=1 number. Little's Law again: one
        // outstanding request per warp cannot cover DRAM latency no matter how well it coalesces.
        // Unroll by 4 and issue all eight loads before consuming any, so four independent misses
        // are in flight per warp.
        // Accumulation order across kb is UNCHANGED (0,1,2,3,4,...), so this stays bit-exact.
        auto dot4 = [](unsigned av, unsigned bv) {
            float s = 0.f;
            #pragma unroll
            for (int i = 0; i < 4; ++i) s += dec_e4m3((av>>(i*8))&0xff) * dec_e4m3((bv>>(i*8))&0xff);
            return s;
        };
        const int KB4 = KB & ~3;
        int kb = 0;
        for (; kb < KB4; kb += 4){
            const int b0 = (kb+0)*128 + lane*4, b1 = (kb+1)*128 + lane*4,
                      b2 = (kb+2)*128 + lane*4, b3 = (kb+3)*128 + lane*4;
            // all loads issued before any use -> 8 outstanding requests per warp
            unsigned bv0 = *(const unsigned*)(Brow + b0), bv1 = *(const unsigned*)(Brow + b1),
                     bv2 = *(const unsigned*)(Brow + b2), bv3 = *(const unsigned*)(Brow + b3);
            unsigned av0 = *(const unsigned*)(A + b0), av1 = *(const unsigned*)(A + b1),
                     av2 = *(const unsigned*)(A + b2), av3 = *(const unsigned*)(A + b3);
            acc += dot4(av0,bv0) * as[kb+0] * bsr[kb+0];
            acc += dot4(av1,bv1) * as[kb+1] * bsr[kb+1];
            acc += dot4(av2,bv2) * as[kb+2] * bsr[kb+2];
            acc += dot4(av3,bv3) * as[kb+3] * bsr[kb+3];
        }
        for (; kb < KB; ++kb){
            int base = kb*128 + lane*4;
            unsigned av = *(const unsigned*)(A + base);
            unsigned bv = *(const unsigned*)(Brow + base);
            acc += dot4(av,bv) * as[kb] * bsr[kb];
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffff, acc, o);
        if (lane == 0) C[n] = acc;
    }
}
// Blocks that exactly fill the device for this kernel: SMs x max resident blocks/SM. Queried once.
__attribute__((unused)) static int gemv_m1_full_grid(int threads){
    static int g = 0;
    if (!g) {
        int dev = 0, sms = 20, per_sm = 1;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, (const void*)fp8_gemv_m1_kernel, threads, 0);
        if (per_sm < 1) per_sm = 1;
        g = sms * per_sm;
    }
    return g;
}
// M=K GEMV (small M, e.g. spec-decode block-verify M=5): one warp per output n reads the B[n] weight row ONCE
// (uint-vectorized) and dots it against ALL M activation rows -> the weight bandwidth is amortized M× with no
// wasted rows (vs the m16-tile TC which wastes 16-M rows). Bandwidth-bound. Gated cosine vs fp8_block_gemm.
#define GEMV_MK_MAXM 16
// M=K GEMV, TEMPLATED on M (LOOP_LOG Finding 28).
//
// The untemplated version declared `float acc[GEMV_MK_MAXM]` = 16 registers regardless of the real
// M, which capped occupancy and is why the prior project's A/B rejected it (334 -> 362 ms). With M
// a compile-time constant the accumulator is exactly M registers, and the reason to want this path
// at all is the access pattern:
//
//   this GEMV     : lane L reads B[n*K + kb*128 + L*4]  -> 32 lanes x 4B = 128 CONTIGUOUS bytes
//   tc_fp8 m16    : lane L reads B[(n0+L/4)*K + ...]    -> 8 rows strided by K, 32B each
//
// Cold (which is always, in situ — every layer's weights come from a 100 GiB working set) the m16
// tile costs ~3x the GEMV; hot, L2 hides it entirely, which is why the isolated bench missed this
// for so long. See gemm_bench's HOT vs COLD rows.
template<int MM>
__global__ void fp8_gemv_mkT_kernel(float* __restrict__ C, const uint8_t* __restrict__ A, const float* __restrict__ as,
                                    const uint8_t* __restrict__ B, const float* __restrict__ bs, int N, int K){
    int warp=(blockIdx.x*blockDim.x+threadIdx.x)>>5; if(warp>=N) return; int n=warp; int lane=threadIdx.x&31; int KB=K/128;
    const uint8_t* Brow=B+(size_t)n*K; const float* bsr=bs+(size_t)(n/128)*KB;
    float acc[MM];
    #pragma unroll
    for(int m=0;m<MM;++m) acc[m]=0.f;
    for(int kb=0; kb<KB; ++kb){
        int base=kb*128+lane*4; unsigned bv=*(const unsigned*)(Brow+base); float wsc=bsr[kb];
        float b0=dec_e4m3(bv&0xff)*wsc, b1=dec_e4m3((bv>>8)&0xff)*wsc, b2=dec_e4m3((bv>>16)&0xff)*wsc, b3=dec_e4m3((bv>>24)&0xff)*wsc;
        #pragma unroll
        for(int m=0; m<MM; ++m){
            unsigned av=*(const unsigned*)(A+(size_t)m*K+base); float asc=as[(size_t)m*KB+kb];
            acc[m]+=(dec_e4m3(av&0xff)*b0 + dec_e4m3((av>>8)&0xff)*b1 + dec_e4m3((av>>16)&0xff)*b2 + dec_e4m3((av>>24)&0xff)*b3)*asc;
        }
    }
    #pragma unroll
    for(int m=0; m<MM; ++m){ float a=acc[m];
        #pragma unroll
        for(int o=16;o>0;o>>=1) a+=__shfl_down_sync(0xffffffff,a,o);
        if(lane==0) C[(size_t)m*N+n]=a; }
}
// NEGATIVE RESULT, retired with a measurement (Finding 41). A shared-A variant of the above —
// decode A once per block into smem with the activation scale folded in, so the inner loop is one
// fp8 decode of B plus M fmas — was SLOWER at every M: 0.347/0.438/0.621/0.940 ms at M=2/3/5/8 on
// wq_b [32768,1024] vs 0.347/0.438/0.517/0.770 for the register version. The A decode is not the
// binding constraint; a float4 smem read per m per k is 16 bytes where the fp8 global read was 4,
// i.e. it trades L1 traffic for 4x the smem traffic. Do not re-queue it.
// Largest M for which the small-M GEMV is preferred over the m16 tile. The templated kernels exist
// up to M=8; whether they WIN up to M=8 is a measurement, not a guess, so the cutoff is a tunable
// with a measured default (see OPTIMIZATION_LOG, "fp8 small-M crossover"). GEMV_MK_MAXM=<n> in the
// environment moves it for A/B; TC_MK=1 disables the path entirely.
// MEASURED DEFAULT = 1, i.e. the GEMV is used at M=1 only. Cold, on the six shapes decode issues,
// the smem-staged m16 tile beats this GEMV at every M>=2 — at M=5 by 1.7-2.9x, and at M=2..4 (where
// the GEMV used to be the default) by 1.5-2.3x. The GEMV is exactly linear in M; the tile is flat.
// Kept reachable via GEMV_MK_MAXM=<n> because it is the only non-mma path at M>=2 and therefore the
// fallback if an mma result is ever in doubt.
static int g_gemv_mk_maxm = -1;
void fp8_set_gemv_mk_maxm(int m){ g_gemv_mk_maxm = m > 8 ? 8 : m; }   // gemm_bench sweeps this
static int gemv_mk_maxm(){
    if (g_gemv_mk_maxm < 0) { const char* e = getenv("GEMV_MK_MAXM"); g_gemv_mk_maxm = e ? atoi(e) : 1;
                              if (g_gemv_mk_maxm > 8) g_gemv_mk_maxm = 8; }
    return g_gemv_mk_maxm;
}
#define GEMV_MK_MAXM_DISPATCH gemv_mk_maxm()
// dispatch to the right instantiation; returns false if M is outside the templated set.
static bool fp8_gemv_mkT(float* C, const uint8_t* A, const float* as, const uint8_t* B, const float* bs,
                         int M, int N, int K, cudaStream_t s){
    const int threads=256, blocks=(N*32+threads-1)/threads;
    switch(M){
        case 2: fp8_gemv_mkT_kernel<2><<<blocks,threads,0,s>>>(C,A,as,B,bs,N,K); return true;
        case 3: fp8_gemv_mkT_kernel<3><<<blocks,threads,0,s>>>(C,A,as,B,bs,N,K); return true;
        case 4: fp8_gemv_mkT_kernel<4><<<blocks,threads,0,s>>>(C,A,as,B,bs,N,K); return true;
        case 5: fp8_gemv_mkT_kernel<5><<<blocks,threads,0,s>>>(C,A,as,B,bs,N,K); return true;
        case 6: fp8_gemv_mkT_kernel<6><<<blocks,threads,0,s>>>(C,A,as,B,bs,N,K); return true;
        case 7: fp8_gemv_mkT_kernel<7><<<blocks,threads,0,s>>>(C,A,as,B,bs,N,K); return true;
        case 8: fp8_gemv_mkT_kernel<8><<<blocks,threads,0,s>>>(C,A,as,B,bs,N,K); return true;
        default: return false;
    }
}
// Global toggle: route dense fp8 GEMMs through the native FP8 tensor core (tc_fp8_gemm, ~18x). Default OFF so
// gates use this warp-per-output oracle (bit-exact). forward.cu sets it true for decode. All our fp8 GEMM
// shapes satisfy N%8==0 && K%128==0 (checked); fall back to the oracle otherwise.
bool g_tc_fp8 = false;
void tc_fp8_gemm(float*, const uint8_t*, const float*, const uint8_t*, const float*, int, int, int, cudaStream_t);
void fp8_block_gemm(float* C, const uint8_t* A_fp8, const float* a_s,
                    const uint8_t* B_fp8, const float* b_s,
                    int M, int N, int K, cudaStream_t stream) {
    // NVFP4 DENSE OVERLAY, opt-in. If an NVFP4 replacement was registered for this weight pointer,
    // serve M=1 from it: half the weight bytes for the same output shape. One dispatch here covers
    // every dense linear in the engine (compressed_attn, compressed_decode, indexer, dspark_real)
    // with no edits at the call sites. `nvfp4_lookup` returns null unless an overlay was loaded, so
    // with no env set this is one predictable branch and the original path below is untouched.
    // M>=2 (spec-decode verify) must use the SAME weights as M=1, or the engine computes one layer
    // two ways. Routed before the fp8 M=K path for exactly that reason.
    // NVFP4_MK_OFF=1 -- diagnostic. At M>=2 the FP8 path is tc_fp8_gemm, a TENSOR-CORE tile, and
    // this file already records that "against the smem-staged tile the GEMV loses at every M>=2 on
    // every shape". nvfp4_gemv_mk is a warp-per-row GEMV, i.e. structurally the losing shape. This
    // switch isolates how much of the suite regression is that choice.
    if (M >= 2 && M <= 8 && nvfp4_enabled() && getenv("NVFP4_MK_OFF")==nullptr) {
        if (const NvFp4Weight* w = nvfp4_lookup(B_fp8))
            if (w->N == N && w->K == K && nvfp4_gemv_mk(C, A_fp8, a_s, *w, M, stream)) return;
    }
    if (M == 1 && nvfp4_enabled()) {
        void nvfp4_note_hit(long long); void nvfp4_note_miss();
        if (const NvFp4Weight* w = nvfp4_lookup(B_fp8)) {
            if (w->N == N && w->K == K && nvfp4_gemv_m1(C, A_fp8, a_s, *w, stream)) {
                nvfp4_note_hit((long long)N*K/2); return; }
        }
        nvfp4_note_miss();
    }
    // M=1 decode: vectorized GEMV (bandwidth-bound, beats the mma-latency-bound TC at M=1). K%128==0 always here.
    if (g_tc_fp8 && M == 1 && (K % 128 == 0) && getenv("NO_GEMV")==nullptr) {
        int threads=256;
        // NEGATIVE RESULT — see LOOP_LOG.md Opt #2. Two wave-quantisation schemes were tried on
        // top of the grid-stride kernel and NEITHER beat this plain one-warp-per-row launch:
        //   clamp to a single full wave      -> wq_b 214.6 -> 186.3 GB/s, wo_b 225.5 -> 194.7 (worse)
        //   round down to whole waves        -> within run-to-run noise; not demonstrable
        // The grid-stride loop is retained (it is a no-op when blocks == want, since the stride then
        // covers N in one pass) so the mechanism stays available, but the launch is the original.
        int blocks = (N*32+threads-1)/threads;                     // one warp per output row
        (void)gemv_m1_full_grid;                                   // kept for future experiments
        fp8_gemv_m1_kernel<<<blocks, threads, 0, stream>>>(C, A_fp8, a_s, B_fp8, b_s, N, K); return; }
    // small-M M=K GEMV (env GEMV_MK=1): A/B'd SLOWER than TC at M>=2 (334->362 ms verify) — TC reads the weight
    // ONCE *and* does the M×N compute via mma, while this GEMV does M scalar dots/weight-read. Kept as a gated
    // reference/negative result; the M=1 GEMV above still wins (trivial compute at M=1). Default OFF for M>=2.
    // small-M: the templated GEMV reads B fully coalesced, which beats the m16 tile on COLD
    // weights by ~3x (Finding 28). TC_MK=1 forces the old m16 path for A/B.
    // Threshold from the COLD A/B (gemm_bench). This USED to read "GEMV beats the m16 tile 2.0x at
    // M=2 ... a wash at M=5" and set the cutoff at M=4. That comparison was against the m16 tile
    // BEFORE its B path was fixed (Finding 41); against the smem-staged tile the GEMV loses at every
    // M>=2 on every shape, so the cutoff is now M=1. Left in place because the dispatch is what the
    // env knob moves, and because M=1 still belongs to the GEMV above.
    if (g_tc_fp8 && M >= 2 && M <= GEMV_MK_MAXM_DISPATCH && (K % 128 == 0) && getenv("TC_MK")==nullptr) {
        if (fp8_gemv_mkT(C, A_fp8, a_s, B_fp8, b_s, M, N, K, stream)) return; }
    if (g_tc_fp8 && (N % 8 == 0) && (K % 128 == 0)) { tc_fp8_gemm(C, A_fp8, a_s, B_fp8, b_s, M, N, K, stream); return; }
    dim3 grid(N, M);
    fp8_block_gemm_kernel<<<grid, 32, 0, stream>>>(C, A_fp8, a_s, B_fp8, b_s, M, N, K);
}
