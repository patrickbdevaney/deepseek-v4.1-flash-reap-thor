// compressor.cu — KV Compressor gated-pooling core, correctness-first (Gate K: ref/gen_units gen_compressor).
#include <cstdlib>
#include "compressor.h"
// When true, `wkv`/`wgate` pointers are BF16 storage, not f32. Set once at engine init.
int g_xin_ring = 0;   // rows in the xin ring for the CURRENT layer; 0 = full history
// Master switch. DEFAULT OFF, and deliberately so: these kernels are shared with src/decode.cu and
// the unit gates, which allocate xin as the full [seqmax, DIM] history. A global default-on would
// make the emit index modulo a ring those callers never built, and read the wrong rows silently.
// The switch is therefore owned by whoever OWNS THE ALLOCATION -- src/engine.cu turns it on right
// before sizing xin as a ring, and nobody else touches it. DSV4_XIN_RING=0 there restores the full
// layout for an A/B, which is how this was validated.
bool g_xin_ring_on = false;
// Widest batch the ring must survive (MAXB). Set by whoever owns the allocation.
int  g_xin_ring_batch = 0;
bool g_compressor_bf16 = false;
#include <cuda_bf16.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include "dscratch.h"

// C[M,N] = A[M,K] @ B[N,K]^T. One warp per (m,n).
//
// LOOP_LOG Finding 34 (Opt #10). The original launched <<<dim3(N,M), 32>>> with a scalar ILP=1
// K-loop, and so lost twice over:
//   * one warp per block caps occupancy at 50% — the identical defect Finding 21 fixed in the MoE
//     and Finding 26 fixed in `gemm_bf16w`; this call site was simply never revisited.
//   * `acc += a[k]*b[k]` is a single dependent load->fma chain, i.e. one outstanding request per
//     lane. Little's Law says achieved bandwidth = MSHRs x bytes / latency, so ILP=1 leaves most
//     of the memory level parallelism on the floor. Opt #7 measured +3.4% end-to-end from exactly
//     this fix on the dense FP8 GEMV.
// float4 gives 4 independent accumulator chains and turns each lane's 4 B loads into 1. This is
// the "fix the inner loop BEFORE narrowing the format" step that Finding 32 concluded with: the
// BF16-native compressor lost because it added conversion ALU to a kernel that was ALU/issue
// bound, not bandwidth bound. Make it bandwidth bound first, then the byte saving pays.
//
// float4 needs 16-byte alignment. B is a mapped safetensors tensor, so it is checked at runtime
// and the kernel falls back to the scalar loop — Finding 25 is why that is not optional.
__global__ void gemm_fp32_kernel(float* __restrict__ C, const float* __restrict__ A,
                                 const float* __restrict__ B, int M, int N, int K, int vec4) {
    const int n = blockIdx.x*(blockDim.x>>5) + (threadIdx.x>>5);
    const int m = blockIdx.y;
    if (m >= M || n >= N) return;
    const int lane = threadIdx.x & 31;
    const float* a = A + (size_t)m * K; const float* b = B + (size_t)n * K;
    float acc = 0.f;
    if (vec4) {
        const float4* a4 = (const float4*)a; const float4* b4 = (const float4*)b;
        const int n4 = K >> 2;
        float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
        for (int k = lane; k < n4; k += 32) {
            const float4 av = a4[k], bv = b4[k];
            a0 = fmaf(av.x, bv.x, a0); a1 = fmaf(av.y, bv.y, a1);
            a2 = fmaf(av.z, bv.z, a2); a3 = fmaf(av.w, bv.w, a3);
        }
        acc = (a0 + a1) + (a2 + a3);
    } else {
        for (int k = lane; k < K; k += 32) acc += a[k] * b[k];
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffff, acc, o);
    if (lane == 0) C[(size_t)m * N + n] = acc;
}
// 16-byte alignment of a *row* needs the base pointer aligned AND the row stride K*4 a multiple
// of 16, i.e. K a multiple of 4 — which the K%4==0 test already implies.
static inline int gemm_vec4_ok(const void* A, const void* B, int K) {
    return ((K & 3) == 0) && ((((uintptr_t)A) & 15) == 0) && ((((uintptr_t)B) & 15) == 0);
}
// M=K variant, same defect and same fix as Finding 42's lm_head. `compressor_emit_group` calls
// gemm_fp32 with M = ntok = 2*ratio = 8 on `wkv`/`wgate`, which Finding 33 made f32 = 16.8 MB each
// — so one group emit read 8 x 33.6 MB where it needed 33.6. By the dprof marks `cattn:compress` is
// 10.11 ms of a K=5 verify and fires every `ratio` steps in base decode too.
//
// Chunked in 8s rather than templated on the full M, because ntok is 2*ratio and ratio is 128 for
// twenty of the layers: a 128-row template would spill, and B re-read ceil(M/8) times is already
// 16x better than M times.
template<int MM>
__global__ void gemm_fp32_mk_kernel(float* __restrict__ C, const float* __restrict__ A,
                                    const float* __restrict__ B, int m0, int N, int K){
    const int n = blockIdx.x*(blockDim.x>>5) + (threadIdx.x>>5);
    if (n >= N) return;
    const int lane = threadIdx.x & 31;
    const float4* b4 = (const float4*)(B + (size_t)n*K);
    const int n4 = K >> 2;
    // Four independent chains per row, exactly as the M=1 kernel: same operation order, so every
    // row is BIT-identical to what the old path produced and the gate can assert rms == 0.
    float acc[MM][4];
    #pragma unroll
    for (int m=0;m<MM;++m){ acc[m][0]=acc[m][1]=acc[m][2]=acc[m][3]=0.f; }
    for (int k = lane; k < n4; k += 32) {
        const float4 bv = b4[k];
        #pragma unroll
        for (int m=0;m<MM;++m){
            const float4 av = ((const float4*)(A + (size_t)(m0+m)*K))[k];
            acc[m][0]=fmaf(av.x,bv.x,acc[m][0]); acc[m][1]=fmaf(av.y,bv.y,acc[m][1]);
            acc[m][2]=fmaf(av.z,bv.z,acc[m][2]); acc[m][3]=fmaf(av.w,bv.w,acc[m][3]);
        }
    }
    #pragma unroll
    for (int m=0;m<MM;++m){ float a=(acc[m][0]+acc[m][1])+(acc[m][2]+acc[m][3]);
        #pragma unroll
        for (int o=16;o>0;o>>=1) a += __shfl_down_sync(0xffffffff,a,o);
        if (lane==0) C[(size_t)(m0+m)*N + n] = a; }
}
// ---------------------------------------------------------------------------------------------
// LADDER 1.12 — N-TILING. The kernel above is one warp per (m-chunk, n), so it reads B once per
// m-chunk: `ceil(M/8)` passes. `compressor_emit_group` on the twenty ratio-128 layers calls this
// with M = ratio = 128, N = d = 512, K = DIM = 4096, i.e. SIXTEEN passes over B = [512,4096] f32 =
// 8.39 MB, twice per layer (`wkv` and `wgate`). Fitting B in the 32 MB L2 is why that costs L2
// bandwidth rather than DRAM bandwidth, which is why 1.8 saw 43-50 ms and not the 335 ms that
// sixteen DRAM passes over 20 layers would have been -- but it is still the traffic, and the
// arithmetic intensity is the thing to fix.
//
// A warp that owns MM rows of A and NN rows of B issues (MM + NN) loads for MM*NN*4 FMAs, so the
// total operand traffic is M*N*K*(1/NN + 1/MM) floats. The shipped (MM=8, NN=1) is 1.125; (8,4) is
// 0.375 -- 3x less -- and (8,8) would be 0.25 but needs 256 accumulator registers and spills.
//
// BIT-EXACT BY CONSTRUCTION, and the constraint that forces this shape rather than a classic
// shared-memory tile: every output must still be produced by ONE warp with lane `l` accumulating
// k4 indices l, l+32, l+64, ... into the same four chains, summed (a0+a1)+(a2+a3) and then reduced
// down the same shuffle tree. Widening the tile changes WHICH warp owns an output and how many
// outputs it owns; it does not touch the order of a single dot product. gate_bf16w asserts
// max|diff| == 0 against the M=1 path for every (MM,NN) this file can dispatch.
//
// The m-chunk loop also moves from the HOST to gridDim.y: sixteen launches per GEMM x 2 x 20 layers
// was 640 launches on the emit step, and on this box a null launch is not free.
template<int MM, int NN>
__global__ void gemm_fp32_mkn_kernel(float* __restrict__ C, const float* __restrict__ A,
                                     const float* __restrict__ B, int M, int N, int K){
    const int n0 = (blockIdx.x*(blockDim.x>>5) + (threadIdx.x>>5))*NN;
    if (n0 >= N) return;
    const int m0 = blockIdx.y*MM;
    const int lane = threadIdx.x & 31;
    const int n4 = K >> 2;
    // Clamp the OOB tail rows onto a valid row and drop them in the epilogue: reading row M-1 twice
    // is harmless, reading past the end of a mapped tensor is not.
    const float4* ap[MM]; const float4* bp[NN];
    #pragma unroll
    for (int m=0;m<MM;++m){ const int r = (m0+m < M) ? (m0+m) : (M-1); ap[m] = (const float4*)(A + (size_t)r*K); }
    #pragma unroll
    for (int j=0;j<NN;++j){ const int c = (n0+j < N) ? (n0+j) : (N-1); bp[j] = (const float4*)(B + (size_t)c*K); }
    float acc[MM][NN][4];
    #pragma unroll
    for (int m=0;m<MM;++m)
        #pragma unroll
        for (int j=0;j<NN;++j){ acc[m][j][0]=acc[m][j][1]=acc[m][j][2]=acc[m][j][3]=0.f; }
    for (int k = lane; k < n4; k += 32) {
        float4 bv[NN];
        #pragma unroll
        for (int j=0;j<NN;++j) bv[j] = bp[j][k];
        #pragma unroll
        for (int m=0;m<MM;++m){
            const float4 av = ap[m][k];
            #pragma unroll
            for (int j=0;j<NN;++j){
                acc[m][j][0]=fmaf(av.x,bv[j].x,acc[m][j][0]); acc[m][j][1]=fmaf(av.y,bv[j].y,acc[m][j][1]);
                acc[m][j][2]=fmaf(av.z,bv[j].z,acc[m][j][2]); acc[m][j][3]=fmaf(av.w,bv[j].w,acc[m][j][3]);
            }
        }
    }
    #pragma unroll
    for (int m=0;m<MM;++m){
        #pragma unroll
        for (int j=0;j<NN;++j){
            float a=(acc[m][j][0]+acc[m][j][1])+(acc[m][j][2]+acc[m][j][3]);
            #pragma unroll
            for (int o=16;o>0;o>>=1) a += __shfl_down_sync(0xffffffff,a,o);
            if (lane==0 && m0+m < M && n0+j < N) C[(size_t)(m0+m)*N + (n0+j)] = a;
        }
    }
}
// The (MM,NN) actually dispatched. Default is 1.12's measured winner; `DSV4_F32MK_TILE=MMxNN` and
// `gemm_fp32_set_tile()` exist so ONE binary can run both arms of the A/B, and (8,1) reproduces the
// pre-1.12 arithmetic launch-for-launch except for the host-side m loop.
static int g_f32mk_mm = 4, g_f32mk_nn = 4;
static int g_f32mk_env = 0;
void gemm_fp32_set_tile(int mm, int nn){ g_f32mk_mm = mm; g_f32mk_nn = nn; g_f32mk_env = 1; }
void gemm_fp32_get_tile(int* mm, int* nn){ *mm = g_f32mk_mm; *nn = g_f32mk_nn; }
static inline void f32mk_tile_from_env(){
    if (g_f32mk_env) return;
    g_f32mk_env = 1;
    const char* e = getenv("DSV4_F32MK_TILE");
    // `nn >= 0`, NOT `nn > 0`. The first engine A/B of 1.12 ran BOTH ARMS AT 4x4 because this test
    // was `nn > 0` and the control arm is spelled `8x0` -- the guard silently rejected the only
    // value that selects the before-arm and left the default in place. It measured a PERFECT null
    // (cattn:compress 14.50 -> 14.47, TOTAL +1.08 at g==0 and -0.73 at g>=1) and every one of those
    // numbers was correct: they are the run-to-run spread of two loads of the SAME code.
    // measurement-and-traps §34. Hence the unconditional line below: an arm that cannot be read
    // back out of the log is an arm you are guessing at.
    // WHITELIST, not just a parse. The dispatcher below is an if-chain over instantiated (MM,NN)
    // templates with a `<8,1>` fallback whose grid was sized for a DIFFERENT mm -- so an (mm,nn)
    // that is parsed but not instantiated would launch a kernel that strides m by 8 over a grid
    // built for mm, and quietly compute the wrong rows. An unrecognised value must fall back to a
    // shape that is correct, not to one that is merely present.
    static const int OK[][2] = {{8,0},{8,1},{8,2},{8,4},{4,4},{4,8},{2,8},{16,1},{16,2},{32,1},
                                {6,4},{4,6},{6,6},{5,5},{4,2},{2,4},{4,1},{2,2}};
    if (e) { int mm=0,nn=0;
        if (sscanf(e,"%dx%d",&mm,&nn)==2 && mm>0 && nn>=0){
            bool ok=false; for (auto& t : OK) if (t[0]==mm && t[1]==nn) ok=true;
            if (ok){ g_f32mk_mm=mm; g_f32mk_nn=nn; }
            else fprintf(stderr,"[f32mk] DSV4_F32MK_TILE=%s is not an instantiated tile -- IGNORED\n", e);
        } else fprintf(stderr,"[f32mk] DSV4_F32MK_TILE=%s did not parse as MMxNN -- IGNORED\n", e);
    }
    fprintf(stderr,"[f32mk] tile %dx%d%s (DSV4_F32MK_TILE=%s)\n", g_f32mk_mm, g_f32mk_nn,
            g_f32mk_nn==0 ? " = PRE-1.12 host chunk loop" : "", e ? e : "unset");
    fflush(stderr);
}
// ---------------------------------------------------------------------------------------------
// CALL-SITE TRACE (DSV4_GEMM_TRACE=1). 1.12's first engine A/B came back a PERFECT null -- the
// `cattn:compress` mark on ratio-128 emit steps was 14.50 ms before and 14.47 after -- while the
// microbenchmark said the shape it targets is 2.32x faster. Exactly one of those can be true of the
// engine, and the ladder entry's premise ("the strided path calls it with ntok = 2*ratio = 256")
// was already known to be wrong in its arithmetic, so the premise about WHICH KERNEL RUNS was worth
// checking rather than reasoning about. This prints, once per distinct (route, M, N, K, vec4) with
// a call count and the bytes it moved, at process exit. It costs one host-side lookup per call and
// is OFF unless the variable is set.
#include <map>
#include <mutex>
namespace {
struct GKey { int route, M, N, K, vec4; bool operator<(const GKey& o) const {
    return route!=o.route?route<o.route : M!=o.M?M<o.M : N!=o.N?N<o.N : K!=o.K?K<o.K : vec4<o.vec4; } };
const char* GROUTE[] = {"fp32:mkn", "fp32:chunk8", "fp32:m1warp", "fp32:cond", "bf16w:mk", "bf16w:m1"};
struct GTrace {
    std::map<GKey,long long> hits; std::mutex mu; bool on;
    GTrace(): on(getenv("DSV4_GEMM_TRACE")!=nullptr) {}
    ~GTrace(){ if(!on || hits.empty()) return;
        fprintf(stderr,"[gemmtrace] %zu distinct (route,M,N,K,vec4) shapes\n", hits.size());
        for(auto& kv : hits){ const GKey& k=kv.first;
            double b = ((double)k.M + k.N)*k.K*4.0 + (double)k.M*k.N*4.0;
            fprintf(stderr,"[gemmtrace] %-12s M=%-6d N=%-7d K=%-6d vec4=%d  calls=%-9lld  %8.2f MB/call  %9.2f GB total\n",
                    GROUTE[k.route], k.M, k.N, k.K, k.vec4, kv.second, b/1e6, b*kv.second/1e9); }
        fflush(stderr); }
    void hit(int route,int M,int N,int K,int vec4){ if(!on) return;
        std::lock_guard<std::mutex> g(mu); ++hits[GKey{route,M,N,K,vec4}]; }
};
GTrace g_gtrace;
}
void gemm_trace_hit(int route,int M,int N,int K,int vec4){ g_gtrace.hit(route,M,N,K,vec4); }
void gemm_fp32(float* C, const float* A, const float* B, int M, int N, int K, cudaStream_t stream) {
    const int wpb = 4, threads = 32*wpb;
    if (M >= 2 && gemm_vec4_ok(A, B, K) && getenv("NO_FP32MK")==nullptr) {
        f32mk_tile_from_env();
        const int mm = g_f32mk_mm, nn = g_f32mk_nn;
        // MEASURED FLOOR (tools/f32mk_bench.cu). Below M=8 the widened tile LOSES -- the M=5
        // `idx_weights_proj` GEMV the indexer issues on EVERY verify step goes 0.0148 -> 0.0205 ms
        // at 4x4, because a 4-row chunk plus a 1-row tail is two launches where the legacy path
        // templates the whole thing into one. That GEMV fires 21x per forward and the tiled shapes
        // fire once every 4 or 128 positions, so a regression there would have eaten the win. The
        // guard also keeps both arms of the A/B IDENTICAL below M=8, which is what makes the paired
        // measurement attributable to the emit and nothing else.
        // gridDim.y is 16-bit: fall back rather than silently truncate the m range.
        // nn == 0 is the EXACT pre-1.12 arm: the host-side 8-row chunk loop below, sixteen
        // launches of gemm_fp32_mk_kernel<8> at the M=128 shape. `DSV4_F32MK_TILE=8x0` selects it,
        // so the before-arm of the A/B is the ORIGINAL CODE and not a re-expression of it.
        const int mfull = (nn >= 1 && M >= 8 && (M/mm) <= 65535) ? (M/mm)*mm : 0;
        if (mfull > 0) {
            const int nwarps = (N + nn - 1)/nn;
            dim3 grid((nwarps + wpb - 1)/wpb, mfull/mm);
            #define F32MKN(a,b) if (mm==a && nn==b) gemm_fp32_mkn_kernel<a,b><<<grid,threads,0,stream>>>(C,A,B,mfull,N,K); else
            gemm_trace_hit(0,mfull,N,K,nn);
            F32MKN(8,4) F32MKN(8,2) F32MKN(8,1) F32MKN(4,4) F32MKN(4,8) F32MKN(16,2) F32MKN(16,1) F32MKN(32,1) F32MKN(2,8) F32MKN(6,4) F32MKN(4,6) F32MKN(6,6) F32MKN(5,5) F32MKN(4,2) F32MKN(2,4) F32MKN(4,1) F32MKN(2,2)
            { gemm_fp32_mkn_kernel<8,1><<<grid,threads,0,stream>>>(C,A,B,mfull,N,K); }
            #undef F32MKN
        }
        // rows [mfull, M) -- the old chunked path, which templates on the exact remainder.
        const int blocks = (N + wpb - 1)/wpb;
        if (mfull < M) gemm_trace_hit(1,M-mfull,N,K,1);
        for (int m0 = mfull; m0 < M; m0 += 8) {
            const int r = (M - m0 < 8) ? (M - m0) : 8;
            #define F32MK(MM) case MM: gemm_fp32_mk_kernel<MM><<<blocks,threads,0,stream>>>(C,A,B,m0,N,K); break;
            switch(r){ F32MK(1) F32MK(2) F32MK(3) F32MK(4) F32MK(5) F32MK(6) F32MK(7) F32MK(8) }
            #undef F32MK
        }
        return;
    }
    gemm_trace_hit(2,M,N,K,gemm_vec4_ok(A,B,K));
    dim3 grid((N + wpb - 1)/wpb, M);
    gemm_fp32_kernel<<<grid, threads, 0, stream>>>(C, A, B, M, N, K, gemm_vec4_ok(A, B, K));
}

// ---------------------------------------------------------------------------------------------
// BF16-WEIGHT GEMV: C[M,N] = A[M,K] (f32) @ B[N,K]^T with B read NATIVELY as bf16.
//
// LOOP_LOG Finding 26. `lm_head` [129280, 4096] and the DSpark markov heads [129280, 256] ship as
// BF16 but were being materialised to f32 by `Loader::bf16` and then fed to gemm_fp32. That
// doubled both the resident footprint and, more importantly, the bytes read every single step:
// lm_head alone went 1059 -> 2118 MB, ~19 ms of a 108 ms decode. The markov head is worse in the
// draft, where it is re-read once per block position (5x).
//
// Two defects fixed together, since they are one code path:
//   1. read bf16 directly (halves the bytes, frees 2.1 GiB of headroom)
//   2. several warps per block — gemm_fp32 launched <<<dim3(N,M), 32>>>, i.e. 646,400 one-warp
//      blocks for lm_head, capping occupancy at 50% exactly as Finding 21 did for the MoE.
//
// bf16x2 loads need 4-byte alignment; B is a mapped safetensors tensor (>=8-byte offsets) and the
// row stride n*K*2 is a multiple of 4 for any even K, but it is checked at runtime anyway —
// Finding 25 is the reason that is not optional here.
__global__ void gemm_bf16w_kernel(float* __restrict__ C, const float* __restrict__ A,
                                  const __nv_bfloat16* __restrict__ B, int M, int N, int K, int vec2) {
    const int n = blockIdx.x*(blockDim.x>>5) + (threadIdx.x>>5);
    const int m = blockIdx.y;
    if (m >= M || n >= N) return;
    const int lane = threadIdx.x & 31;
    const float* a = A + (size_t)m * K;
    const __nv_bfloat16* b = B + (size_t)n * K;
    float acc = 0.f;
    if (vec2 == 2) {
        // Finding 34: 8 bf16 per B load (one float4) + 4 independent accumulators. The vec2 path
        // below was still ILP=1 on a dependent load->fma chain; this is what makes the kernel
        // bandwidth bound rather than issue bound, which is the precondition for the BF16 byte
        // saving to pay at all (Finding 32).
        const float4* b4 = (const float4*)b; const float4* a4 = (const float4*)a;
        const int n8 = K >> 3;
        float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
        for (int k = lane; k < n8; k += 32) {
            const float4 bv = b4[k];
            const float4 av0 = a4[2*k], av1 = a4[2*k+1];
            const __nv_bfloat162* bp = (const __nv_bfloat162*)&bv;
            const float2 p0 = __bfloat1622float2(bp[0]), p1 = __bfloat1622float2(bp[1]);
            const float2 p2 = __bfloat1622float2(bp[2]), p3 = __bfloat1622float2(bp[3]);
            a0 = fmaf(av0.x, p0.x, a0); a1 = fmaf(av0.y, p0.y, a1);
            a2 = fmaf(av0.z, p1.x, a2); a3 = fmaf(av0.w, p1.y, a3);
            a0 = fmaf(av1.x, p2.x, a0); a1 = fmaf(av1.y, p2.y, a1);
            a2 = fmaf(av1.z, p3.x, a2); a3 = fmaf(av1.w, p3.y, a3);
        }
        acc = (a0 + a1) + (a2 + a3);
    } else if (vec2) {
        const __nv_bfloat162* b2 = (const __nv_bfloat162*)b;
        const int n2 = K >> 1;
        for (int k = lane; k < n2; k += 32) {
            const float2 bv = __bfloat1622float2(b2[k]);
            acc += a[2*k] * bv.x + a[2*k+1] * bv.y;
        }
    } else {
        for (int k = lane; k < K; k += 32) acc += a[k] * __bfloat162float(b[k]);
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffff, acc, o);
    if (lane == 0) C[(size_t)m * N + n] = acc;
}
// M=K variant (LOOP_LOG Finding 42). The kernel above is one warp per OUTPUT ELEMENT with
// grid(N/wpb, M), so at M>1 it reads the whole B matrix M times. For every other caller B is a few
// MB and nobody noticed; for the lm_head B is [129280, 4096] bf16 = 1.06 GB, and the spec-decode
// verify calls it at M=BLK. Five passes over 1.06 GB is 5.3 GB of DRAM per verify — more traffic
// than any single MoE phase — to compute 5 rows.
//
// Same shape of fix as ogroup_gemv_mk (Finding 40): one warp per n, load the B row ONCE, dot it
// against all MM activation rows. B traffic becomes 1x. A is re-read per warp, but A is MM*K*4 =
// 80 KB at the verify shape — an L2 resident, not DRAM — so it trades 4.2 GB of DRAM for L2 hits.
// V8=true reads 8 bf16 per B load (float4, needs a 16-byte-aligned base); V8=false reads 2 (one
// unsigned, needs 4). Both variants exist because B here is a MAPPED tensor whose alignment nothing
// guarantees — the same fact that crashed the fp8 staging kernel (Finding 41). Falling back to the
// M=1 kernel when the base is only 4-byte aligned would silently give the M-times-B-traffic
// behaviour this whole change exists to remove, which is the worst of the three outcomes.
template<int MM, bool V8>
__global__ void gemm_bf16w_mk_kernel(float* __restrict__ C, const float* __restrict__ A,
                                     const __nv_bfloat16* __restrict__ B, int N, int K){
    const int n = blockIdx.x*(blockDim.x>>5) + (threadIdx.x>>5);
    if (n >= N) return;
    const int lane = threadIdx.x & 31;
    const __nv_bfloat16* b = B + (size_t)n*K;
    float acc[MM];
    #pragma unroll
    for (int m=0;m<MM;++m) acc[m]=0.f;
    if (V8) {
        const float4* b4 = (const float4*)b;
        const int n8 = K >> 3;
        for (int k = lane; k < n8; k += 32) {
            const float4 bv = b4[k];
            const __nv_bfloat162* bp = (const __nv_bfloat162*)&bv;
            const float2 p0=__bfloat1622float2(bp[0]), p1=__bfloat1622float2(bp[1]);
            const float2 p2=__bfloat1622float2(bp[2]), p3=__bfloat1622float2(bp[3]);
            #pragma unroll
            for (int m=0;m<MM;++m){
                const float4* a4 = (const float4*)(A + (size_t)m*K);
                const float4 av0 = a4[2*k], av1 = a4[2*k+1];
                // Same accumulation order as the M=1 vec2==2 path, so a single row is bit-identical.
                float s0=fmaf(av0.x,p0.x,0.f), s1=fmaf(av0.y,p0.y,0.f), s2=fmaf(av0.z,p1.x,0.f), s3=fmaf(av0.w,p1.y,0.f);
                s0=fmaf(av1.x,p2.x,s0); s1=fmaf(av1.y,p2.y,s1); s2=fmaf(av1.z,p3.x,s2); s3=fmaf(av1.w,p3.y,s3);
                acc[m] += (s0+s1)+(s2+s3);
            }
        }
    } else {
        const __nv_bfloat162* b2 = (const __nv_bfloat162*)b;
        const int n2 = K >> 1;
        for (int k = lane; k < n2; k += 32) {
            const float2 bv = __bfloat1622float2(b2[k]);
            #pragma unroll
            for (int m=0;m<MM;++m){
                const float* a = A + (size_t)m*K;
                acc[m] += a[2*k]*bv.x + a[2*k+1]*bv.y;
            }
        }
    }
    #pragma unroll
    for (int m=0;m<MM;++m){ float a=acc[m];
        #pragma unroll
        for (int o=16;o>0;o>>=1) a += __shfl_down_sync(0xffffffff,a,o);
        if (lane==0) C[(size_t)m*N + n] = a; }
}
// ===================== B9: bf16 TENSOR-CORE gemm_bf16w (prefill) =====================
// gemm_bf16w_kernel below is warp-per-output-ELEMENT. At decode M=1 that is right. At prefill it is
// handed [1022 x 4096] x [4096 x 1024] twice per layer by compressor_forward and launches
// (N/4, M) = 261,632 blocks of 128 threads to run a dense GEMM at **0.32 TFLOPS** — against a
// MEASURED fp32 peak of 5.45 and a MEASURED bf16 tensor-core peak of 53.08 (tools/flops_probe.cu).
// This is `cattn:compress` (2231 ms) and part of `cattn:indexer` (2081 ms).
//
// B already ships bf16, so only A needs converting. THAT IS A NUMERICS CHANGE and the reason this
// ships behind a flag: the fp32 activation is rounded to bf16 (8 mantissa bits) before the mma.
// It is defensible here specifically because compressor_forward's very next step quantises this
// output to fp8 (64-wide) or fp4 (32-wide) — 4 to 8 bits — so bf16 is far above the precision the
// result is about to be squeezed into. It is NOT defensible by default anywhere else, which is why
// the dispatch below only takes this path for the compressor's shapes.
//
// Fragment layout follows ogm_mma / tc_ogroup exactly (row = lane/4 and +8, k = 2*(lane%4) and +8),
// which is already gated by gate_ogroup_gemv, rather than being re-derived from the PTX docs.
__device__ __forceinline__ void bfw_mma(float* c, const unsigned* a, const unsigned* b){
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
      :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1])); }

#define BFW_BM 64
#define BFW_BN 64
#define BFW_BK 32
// NS = number of bf16 terms the fp32 activation is split into (3xTF32-style).
//
// F89 measured what a SINGLE bf16 term costs: rms_rel 5.5e-4 in the GEMM, and acceptance 4.00 ->
// 2.18 on the full model. The mechanism is not accumulated rounding — it is that the indexer's
// output drives a **top-512 selection**, and a perturbation far below the fp4 quantisation step
// still flips near-ties in the ranking, changing WHICH keys the sparse attention reads. A discrete
// selection has no error budget.
//
// So split instead of round. bf16 carries 8 bits of precision; a = hi + mid + lo recovers ~24, i.e.
// fp32, at 3x the mma instructions — which against a MEASURED 53.08 TFLOPS bf16 peak is still
// ~17.7 TFLOPS effective, versus the 0.32 TFLOPS the scalar kernel achieves. Two terms (~16 bits)
// are available via TC_BF16W_NS=2 for the accuracy/speed A/B.
//
// The split is exact by construction: each residual is what the previous term could not represent,
// so hi+mid+lo reproduces the fp32 value to within the last term's rounding.

template<int NS>
__global__ __launch_bounds__(128,2) void gemm_bf16w_tc_kernel(
        float* __restrict__ C, const float* __restrict__ A, const __nv_bfloat16* __restrict__ B,
        int M, int N, int K){
    __shared__ __nv_bfloat16 As[NS][BFW_BM*BFW_BK];
    __shared__ __nv_bfloat16 Bs[BFW_BN*BFW_BK];
    const int tid=threadIdx.x, lane=tid&31, warp=tid>>5;
    const int wm=warp>>1, wn=warp&1;                 // 2x2 warp grid; each warp owns [32 x 32]
    const int gid=lane>>2, t4=lane&3;
    const int m0=blockIdx.y*BFW_BM, n0=blockIdx.x*BFW_BN;
    float c[2][4][4];
    #pragma unroll
    for(int i=0;i<2;++i)
        #pragma unroll
        for(int j=0;j<4;++j){ c[i][j][0]=c[i][j][1]=c[i][j][2]=c[i][j][3]=0.f; }

    for(int k0=0;k0<K;k0+=BFW_BK){
        for(int t=tid;t<BFW_BM*BFW_BK;t+=128){
            const int i=t/BFW_BK, j=t%BFW_BK, gm=m0+i, gk=k0+j;
            float v = (gm<M && gk<K) ? A[(size_t)gm*K+gk] : 0.f;
            #pragma unroll
            for(int sp=0; sp<NS; ++sp){
                __nv_bfloat16 h = __float2bfloat16(v);
                As[sp][t] = h;
                v -= __bfloat162float(h);            // residual for the next term
            }
        }
        for(int t=tid;t<BFW_BN*BFW_BK;t+=128){
            const int i=t/BFW_BK, j=t%BFW_BK, gn=n0+i, gk=k0+j;
            Bs[t] = (gn<N && gk<K) ? B[(size_t)gn*K+gk] : (__nv_bfloat16)0.f;
        }
        __syncthreads();
        #pragma unroll
        for(int ks=0; ks<BFW_BK; ks+=16){
            unsigned b[4][2];
            #pragma unroll
            for(int nt=0; nt<4; ++nt){
                const int cn=wn*32+nt*8+gid, kk=ks+2*t4;
                b[nt][0]=*(const unsigned*)&Bs[cn*BFW_BK+kk];
                b[nt][1]=*(const unsigned*)&Bs[cn*BFW_BK+kk+8];
            }
            // Most-significant term first: the accumulator grows monotonically in magnitude, so the
            // small corrections are added last and are not lost to a large running sum.
            #pragma unroll
            for(int sp=0; sp<NS; ++sp){
                unsigned a[2][4];
                #pragma unroll
                for(int mt=0; mt<2; ++mt){
                    const int r0=wm*32+mt*16+gid, r8=r0+8, kk=ks+2*t4;
                    a[mt][0]=*(const unsigned*)&As[sp][r0*BFW_BK+kk];
                    a[mt][1]=*(const unsigned*)&As[sp][r8*BFW_BK+kk];
                    a[mt][2]=*(const unsigned*)&As[sp][r0*BFW_BK+kk+8];
                    a[mt][3]=*(const unsigned*)&As[sp][r8*BFW_BK+kk+8];
                }
                #pragma unroll
                for(int mt=0; mt<2; ++mt)
                    #pragma unroll
                    for(int nt=0; nt<4; ++nt) bfw_mma(c[mt][nt], a[mt], b[nt]);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for(int mt=0; mt<2; ++mt){
        const int r0=m0+wm*32+mt*16+gid, r8=r0+8;
        #pragma unroll
        for(int nt=0; nt<4; ++nt){
            const int cn=n0+wn*32+nt*8+2*t4;
            if(r0<M && cn  <N) C[(size_t)r0*N+cn  ]=c[mt][nt][0];
            if(r0<M && cn+1<N) C[(size_t)r0*N+cn+1]=c[mt][nt][1];
            if(r8<M && cn  <N) C[(size_t)r8*N+cn  ]=c[mt][nt][2];
            if(r8<M && cn+1<N) C[(size_t)r8*N+cn+1]=c[mt][nt][3];
        }
    }
}
void gemm_bf16w(float* C, const float* A, const void* Bbf16, int M, int N, int K, cudaStream_t stream) {
    const int wpb = 4, threads = 32*wpb;
    gemm_trace_hit(4,M,N,K,0);
    // 2 = float4 (8 bf16) path, 1 = bf16x2 path, 0 = scalar. B rows are 16-byte aligned iff the
    // base is and K*2 is a multiple of 16, i.e. K%8==0; A rows likewise need K*4 %16 == 0.
    const int vec2 = (((K & 7) == 0) && ((((uintptr_t)Bbf16) & 15) == 0) && ((((uintptr_t)A) & 15) == 0)) ? 2
                   : ((((K & 1) == 0) && ((((uintptr_t)Bbf16) & 3) == 0)) ? 1 : 0);
    // M=K path: only when the vectorised alignment holds (it is the only variant implemented) and
    // only up to M=8, past which MM accumulators plus 2 float4 A loads per m stop paying.
    if (vec2 >= 1 && M >= 2 && M <= 16 && getenv("NO_BF16MK")==nullptr) {
        const int blocks = (N + wpb - 1)/wpb; const bool v8 = (vec2 == 2);
        #define BFMK(MM) case MM: if(v8) gemm_bf16w_mk_kernel<MM,true ><<<blocks,threads,0,stream>>>(C,A,(const __nv_bfloat16*)Bbf16,N,K); \
                                  else   gemm_bf16w_mk_kernel<MM,false><<<blocks,threads,0,stream>>>(C,A,(const __nv_bfloat16*)Bbf16,N,K); return;
        switch(M){ BFMK(2)  BFMK(3)  BFMK(4)  BFMK(5)  BFMK(6)  BFMK(7)  BFMK(8) BFMK(9)
                   BFMK(10) BFMK(11) BFMK(12) BFMK(13) BFMK(14) BFMK(15) BFMK(16) default: break; }
        #undef BFMK
    }
    // Tensor-core path: prefill shapes only, and OPT-IN (TC_BF16W=1) because it rounds the fp32
    // activation to bf16. M>=64 keeps decode (M=1) and the verify (M<=16) on the exact kernel.
    static int tcdbg = -1; if(tcdbg<0) tcdbg = getenv("TC_BF16W_DEBUG")!=nullptr;
    if(tcdbg) fprintf(stderr,"[bf16w] M=%d N=%d K=%d tc=%d\n", M,N,K,
                      (getenv("TC_BF16W") && M>=64 && (K%16==0) && (N%8==0)) ? 1:0);
    if(getenv("TC_BF16W") && M>=64 && (K%16==0) && (N%8==0)){
        dim3 g((N+BFW_BN-1)/BFW_BN, (M+BFW_BM-1)/BFW_BM);
        const char* nse = getenv("TC_BF16W_NS");
        int NS = nse ? atoi(nse) : 2;
        // NS=2 by default, MEASURED not assumed: gate_bf16w_tc gives rms_rel 5.46e-4 at NS=1,
        // 1.02e-5 at NS=2, and 1.02e-5 at NS=3 -- the third term buys NOTHING, because at that
        // point the residual is no longer activation precision but the accumulation-order
        // difference between this tiled kernel and the scalar reference, which an exact fp32
        // kernel would also have. So 2 terms sit at the floor of what the comparison can resolve
        // and the third is 50% more mma for nothing.
        if(NS<1||NS>3) NS=3;
        switch(NS){
          case 1: gemm_bf16w_tc_kernel<1><<<g,128,0,stream>>>(C,A,(const __nv_bfloat16*)Bbf16,M,N,K); break;
          case 2: gemm_bf16w_tc_kernel<2><<<g,128,0,stream>>>(C,A,(const __nv_bfloat16*)Bbf16,M,N,K); break;
          default:gemm_bf16w_tc_kernel<3><<<g,128,0,stream>>>(C,A,(const __nv_bfloat16*)Bbf16,M,N,K); break; }
        return;
    }
    dim3 grid((N + wpb - 1)/wpb, M);
    gemm_bf16w_kernel<<<grid, threads, 0, stream>>>(C, A, (const __nv_bfloat16*)Bbf16, M, N, K, vec2);
}
// device-conditional gemm (CUDA-graph emit): launch stays static but the block early-exits when this step is
// NOT a group-completion (so the expensive K-loop only runs on commit steps ~ every `ratio` tokens).
// grid-STRIDE (small fixed grid) so non-commit steps schedule few blocks (all early-return), not M*N of them.
__global__ void gemm_fp32_cond_kernel(float* __restrict__ C, const float* __restrict__ A, const float* __restrict__ B,
                                      int M, int N, int K, const int* __restrict__ d_pos, int ratio, int vec4){
    if(((*d_pos)+1)%ratio != 0) return;
    int lane=threadIdx.x&31, wpb=blockDim.x>>5, warp=blockIdx.x*wpb + (threadIdx.x>>5), nw=gridDim.x*wpb, tot=M*N;
    for(int idx=warp; idx<tot; idx+=nw){ int m=idx/N, n=idx%N;
        const float* a=A+(size_t)m*K; const float* b=B+(size_t)n*K; float acc=0.f;
        if(vec4){                                                  // Finding 34: same ILP fix as gemm_fp32
            const float4* a4=(const float4*)a; const float4* b4=(const float4*)b; const int n4=K>>2;
            float a0=0.f,a1=0.f,a2=0.f,a3=0.f;
            for(int k=lane;k<n4;k+=32){ const float4 av=a4[k], bv=b4[k];
                a0=fmaf(av.x,bv.x,a0); a1=fmaf(av.y,bv.y,a1); a2=fmaf(av.z,bv.z,a2); a3=fmaf(av.w,bv.w,a3); }
            acc=(a0+a1)+(a2+a3);
        } else for(int k=lane;k<K;k+=32) acc+=a[k]*b[k];
        #pragma unroll
        for(int o=16;o>0;o>>=1) acc+=__shfl_down_sync(0xffffffff,acc,o);
        if(lane==0) C[idx]=acc; }
}
void gemm_trace_hit(int,int,int,int,int);
void gemm_fp32_cond(float* C, const float* A, const float* B, int M, int N, int K, const int* d_pos, int ratio, cudaStream_t stream){
    gemm_trace_hit(3,M,N,K,gemm_vec4_ok(A,B,K));
    gemm_fp32_cond_kernel<<<256,256,0,stream>>>(C,A,B,M,N,K,d_pos,ratio,gemm_vec4_ok(A,B,K));   // 256*8=2048 warps grid-stride M*N
}

// pooled[g,e] = Σ_p softmax_p(score[g*ratio+p,e]+ape[p,e]) * kv[g*ratio+p,e]. One thread per (g,e).
__global__ void compressor_pool_kernel(float* __restrict__ pooled, const float* __restrict__ kv,
                                       const float* __restrict__ score, const float* __restrict__ ape,
                                       int groups, int ratio, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= groups * d) return;
    int g = i / d, e = i % d;
    float mx = -1e30f;
    for (int p = 0; p < ratio; ++p) { float s = score[((size_t)(g * ratio + p)) * d + e] + ape[(size_t)p * d + e]; mx = fmaxf(mx, s); }
    float sum = 0.f, acc = 0.f;
    for (int p = 0; p < ratio; ++p) {
        float s = score[((size_t)(g * ratio + p)) * d + e] + ape[(size_t)p * d + e];
        float w = expf(s - mx); sum += w;
        acc += w * kv[((size_t)(g * ratio + p)) * d + e];
    }
    pooled[i] = acc / sum;
}
void compressor_pool(float* pooled, const float* kv, const float* score, const float* ape,
                     int groups, int ratio, int d, cudaStream_t stream) {
    compressor_pool_kernel<<<(groups * d + 255) / 256, 256, 0, stream>>>(pooled, kv, score, ape, groups, ratio, d);
}

// ---------------- overlap pooling (ratio==4) ----------------
// kv,score:[s,2d] (s=groups*ratio); ape:[ratio,2d]. Slot q in [0,2*ratio): q>=ratio -> current group token
// (q-ratio), dims [d:2d]; q<ratio -> previous group (g-1) token q, dims [0:d] (masked for g=0).
__global__ void compressor_pool_overlap_kernel(float* __restrict__ pooled, const float* __restrict__ kv,
                                               const float* __restrict__ score, const float* __restrict__ ape,
                                               int groups, int ratio, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= groups * d) return;
    int g = i / d, e = i % d; int twod = 2 * d, nslot = 2 * ratio;
    float mx = -1e30f;
    for (int q = 0; q < nslot; ++q) {
        float sc;
        if (q >= ratio) { int tok = g * ratio + (q - ratio); sc = score[(size_t)tok * twod + d + e] + ape[(size_t)(q - ratio) * twod + d + e]; }
        else if (g >= 1) { int tok = (g - 1) * ratio + q; sc = score[(size_t)tok * twod + e] + ape[(size_t)q * twod + e]; }
        else sc = -1e30f;
        mx = fmaxf(mx, sc);
    }
    float sum = 0.f, acc = 0.f;
    for (int q = 0; q < nslot; ++q) {
        float sc, kvv;
        if (q >= ratio) { int tok = g * ratio + (q - ratio); sc = score[(size_t)tok * twod + d + e] + ape[(size_t)(q - ratio) * twod + d + e]; kvv = kv[(size_t)tok * twod + d + e]; }
        else if (g >= 1) { int tok = (g - 1) * ratio + q; sc = score[(size_t)tok * twod + e] + ape[(size_t)q * twod + e]; kvv = kv[(size_t)tok * twod + e]; }
        else continue;
        float w = expf(sc - mx); sum += w; acc += w * kvv;
    }
    pooled[i] = acc / sum;
}
void compressor_pool_overlap(float* pooled, const float* kv, const float* score, const float* ape,
                             int groups, int ratio, int d, cudaStream_t stream) {
    compressor_pool_overlap_kernel<<<(groups * d + 255) / 256, 256, 0, stream>>>(pooled, kv, score, ape, groups, ratio, d);
}

// ---------------- full Compressor forward ----------------
#include "mla_attn.h"     // rmsnorm, rope_interleaved, act_quant_fp8sim/fp4sim
#include "kv_pack.h"     // 1b.2 packed main-KV store
#include "indexer.h"      // hadamard
#include <cstdio>
#define CU2(x) do{cudaError_t e=(x); if(e){fprintf(stderr,"cuda %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)
void compressor_forward(float* out, const float* x, const float* wkv, const float* wgate,
                        const float* ape, const float* norm_w, const float* cosT, const float* sinT,
                        int s, int dim, int d, int ratio, bool overlap, int rope_dim, float eps,
                        bool rotate, cudaStream_t stream) {
    int coff = overlap ? 2 : 1, groups = s / ratio, od = coff * d;
    // LOOP_LOG Finding 53. groups = s/ratio is ZERO for every ratio-128 layer at every prompt this
    // project has ever run (the longest is 18 tokens against a ratio of 128), and it was falling
    // through to five launches with gridDim = (0*d+255)/256 = 0. Each one fails to launch and leaves
    // cudaErrorInvalidValue in the thread's last-error slot; none of them could have done any work,
    // because there is no complete group to emit yet. Returning early is what the code always meant.
    if (groups <= 0) return;
    float *kv, *score;
    kv=(decltype(kv))dmalloc( (size_t)s * od * 4); score=(decltype(score))dmalloc( (size_t)s * od * 4);
    // BF16-NATIVE (LOOP_LOG Finding 32): wkv/wgate ship as BF16 and were being expanded to f32 by
    // Loader::bf16, doubling 526 MB/step to 1052. gemm_bf16w reads them natively. Same defect class
    // as the lm_head fix (Opt #4); same proven template.
    if (g_compressor_bf16) { gemm_bf16w(kv, x, (const void*)wkv, s, od, dim, stream);
                             gemm_bf16w(score, x, (const void*)wgate, s, od, dim, stream); }
    else                   { gemm_fp32 (kv, x, wkv, s, od, dim, stream);
                             gemm_fp32 (score, x, wgate, s, od, dim, stream); }
    if (overlap) compressor_pool_overlap(out, kv, score, ape, groups, ratio, d, stream);
    else         compressor_pool(out, kv, score, ape, groups, ratio, d, stream);
    rmsnorm(out, out, norm_w, groups, d, eps, true, stream);
    rope_interleaved(out + (d - rope_dim), cosT, sinT, groups, rope_dim, false, d, 1, stream);
    if (rotate) { hadamard(out, out, groups, d, stream); act_quant_fp4sim(out, groups, d, 32, d, stream); }  // indexer compressor
    else        { act_quant_fp8sim(out, groups, d - rope_dim, 64, d, stream); }                             // main compressor NoPE
    dsync(stream);
    dfree(kv); dfree(score);
}

// ---- incremental single-group emit (STRUCTURAL_PLAN Step 4 decode) ----
// Emit ONE compressed row = compressor_forward's out[g], from just this group's tokens (append-only KV: a
// compressed row finalizes once its `ratio` tokens exist and never changes). Non-overlap (ratio!=4): pools
// x[g*ratio .. g*ratio+ratio-1]. Overlap (ratio==4): pools the 2 local groups [(g-1)*ratio .. g*ratio+ratio-1]
// (prev half masked for g==0) and takes the current one. Bit-exact vs compressor_forward (same per-group math).
void compressor_emit_group(float* out_row, const float* x, int g, int ratio, const float* wkv,
                           const float* wgate, const float* ape, const float* norm_w,
                           const float* cc_cos, const float* cc_sin, int dim, int d, bool overlap,
                           int rope_dim, float eps, bool rotate, cudaStream_t stream){
    int coff = overlap ? 2 : 1, od = coff * d;
    int ntok, tok0, localg;
    if(overlap){ tok0 = (g>=1) ? (g-1)*ratio : 0; ntok = (g>=1) ? 2*ratio : ratio; localg = (g>=1) ? 1 : 0; }
    else       { tok0 = g*ratio; ntok = ratio; localg = 0; }
    // XIN RING. `x` is the attention-input history, and the ONLY thing this function ever reads of
    // it is [tok0, tok0+ntok) with ntok <= 2*ratio -- which is why that history does not have to be
    // kept for the whole context. When g_xin_ring is set, the history is a ring of that many rows
    // and the group's window is mapped into it. tok0 is always a multiple of ratio and the ring is
    // a multiple of ratio, so tok0 % ring is too, and the caller mirrors the first `ratio` rows past
    // the end of the ring -- so the window is CONTIGUOUS at either of the two offsets it can land
    // on, and this stays a single pointer with no wrap handling in the GEMM below.
    const int tok0r = g_xin_ring ? (tok0 % g_xin_ring) : tok0;
    const float* xg = x + (size_t)tok0r*dim;
    float *kv,*score,*pooled;
    kv=(decltype(kv))dmalloc((size_t)ntok*od*4); score=(decltype(score))dmalloc((size_t)ntok*od*4);
    pooled=(decltype(pooled))dmalloc((size_t)(localg+1)*d*4);
    if (g_compressor_bf16) { gemm_bf16w(kv, xg, (const void*)wkv, ntok, od, dim, stream);
                             gemm_bf16w(score, xg, (const void*)wgate, ntok, od, dim, stream); }
    else                   { gemm_fp32 (kv, xg, wkv, ntok, od, dim, stream);
                             gemm_fp32 (score, xg, wgate, ntok, od, dim, stream); }
    if(overlap) compressor_pool_overlap(pooled, kv, score, ape, localg+1, ratio, d, stream);
    else        compressor_pool(pooled, kv, score, ape, 1, ratio, d, stream);
    float* prow = pooled + (size_t)localg*d;                 // the target group's pooled row
    // 1b.2. `rotate` is the INDEXER compressor and its cache is 1b.1's format, untouched here. The
    // non-rotate, d == HEAD_DIM case is the MAIN compressed-KV cache and nothing else: every caller
    // of it in this engine passes a `comp_kv` row (kernels/compressed_decode.cu), and the one-shot
    // prefill uses `compressor_forward` next door, not this function. So the pack switch is the
    // global one and there is no second source of truth about which buffer this is.
    const bool packed_out = (g_kv_pack && !rotate && d == (int)dsv4::HEAD_DIM);
    float* dst = packed_out ? kv_stage(out_row, 1) : out_row;
    rmsnorm(dst, prow, norm_w, 1, d, eps, true, stream);
    rope_interleaved(dst + (d - rope_dim), cc_cos + (size_t)g*(rope_dim/2), cc_sin + (size_t)g*(rope_dim/2),
                     1, rope_dim, false, d, 1, stream);
    if(rotate){ hadamard(dst, dst, 1, d, stream); act_quant_fp4sim(dst, 1, d, 32, d, stream); }
    else if(packed_out) { kv_commit(out_row, dst, 1, stream); }
    else      { act_quant_fp8sim(dst, 1, d - rope_dim, 64, d, stream); }
    dsync(stream);
    if(packed_out) kv_stage_free(dst, out_row);
    dfree(kv); dfree(score); dfree(pooled);
}
