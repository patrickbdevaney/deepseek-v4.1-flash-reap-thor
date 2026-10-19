// indexer.h — DSA Indexer primitives (deepseek_v4, model.py:386-439).
// hadamard: randomized Hadamard rotation (rotate_activation, model.py:253-257) = (x @ H_D) * D^-0.5.
// index_score: DSA scoring index_score[s,t] = Σ_h relu(q[s,h]·kv[t]) * weights[s,h] (model.py:426-427).
#pragma once
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdio>
#include <cstdlib>

// Dynamic shared-memory request for the four scalar top-k scan kernels (k_topk_offset in indexer.cu;
// k_topk_decode / k_topk_verify / k_topk_masked in compressed_decode.cu). LOOP_LOG Finding 53.
//
// Every one of them declares `extern __shared__ float sh[]`, fills sh[0..n-1] and scans it with
// `for(t=0;t<n;++t)`, and every one of them was launched with exactly `n*sizeof(float)`. nvcc widens
// that scan into a 12-BYTE vectorised shared load, so any n < 3 reads past the allocation:
// compute-sanitizer reports "Invalid __shared__ read of size 12 bytes" and kills the launch, which
// the caller then reports as a fault on its own cudaStreamSynchronize line. Measured on this box
// (tests/gate_indexer_decode under memcheck): T=1 and T=2 fault, T>=3 are clean.
//
// n is a compressed-row count — 1..a few hundred — so rounding up costs nothing. `n+3` covers a
// 4-float load issued at the last element; the &~3 keeps the request 16-byte aligned.
// ---- warp argmax, the primitive behind the parallel top-k --------------------------------------
// The four top-k kernels used to open with `if(threadIdx.x||blockIdx.x) return;` and run an
// O(topk x T) selection sort on ONE thread of the 32 launched, on 1 of 20 SMs. At 24k context that
// is 512 x 6000 serial comparisons per ratio-4 layer, 21 layers, per target forward -- measured as
// the whole of the context-linear term in tools/decode_model.py's fit, and 14-28x slower than the
// same algorithm spread across the warp that was already there.
//
// TIE-BREAKING IS THE CORRECTNESS ARGUMENT. The serial scans ran ascending with a STRICT `>`, so
// equal scores resolved to the LOWEST index. This must reproduce that exactly: the selected set
// being right is not enough, because sparse_attn sums the selected rows IN ORDER and a reordering
// changes fp32 association. Callers therefore pass `T` (one past any valid index) as the "nothing
// found" sentinel rather than -1, so a lane that found nothing loses every comparison and lower
// indices win ties for free. Verified bit-identical against the originals on nine shapes and three
// distributions -- including one built entirely of exact ties -- by tests/gate_topk_warp.cu.
__device__ __forceinline__ void warp_argmax(float& best, int& bi){
    #pragma unroll
    for(int o=16;o>0;o>>=1){
        float ob = __shfl_down_sync(0xffffffffu, best, o);
        int   oi = __shfl_down_sync(0xffffffffu, bi,   o);
        if(ob > best || (ob == best && oi < bi)){ best = ob; bi = oi; }
    }
    bi   = __shfl_sync(0xffffffffu, bi,   0);
    best = __shfl_sync(0xffffffffu, best, 0);
}

static inline size_t topk_scan_smem(int n){
    int m = n + 3; m = (m + 3) & ~3; return (size_t)m * sizeof(float);
}

// ---- DECODE_LADDER item 1.4: dynamic shared memory above the 48 KiB default --------------------
//
// THE DEFECT, MEASURED ON THIS BOX rather than inferred. The four scan kernels above stage the
// whole score row in DYNAMIC shared memory, so they ask for ~4T bytes, and the per-block default is
// `cudaDevAttrMaxSharedMemoryPerBlock` = 49,152 B here -- the largest T that fits is 12,285,
// i.e. context 49,140 at ratio 4 (the sizer rounds up by up to 3 floats, so 12,288 is already over).
// One block above that:
//
//     T= 12288 smem= 49152  launch=cudaSuccess           sync=cudaSuccess  out=CORRECT
//     T= 16384 smem= 65536  launch=cudaErrorInvalidValue sync=cudaSuccess  out=UNTOUCHED
//
// Read the middle column. The LAUNCH fails, the following synchronize returns **success**, and the
// output buffer is left exactly as it was -- so the engine's own defences do not see it. `dprobe()`
// is compiled in but inert unless `DSV4_SYNCPROBE` is set; `dsync()` does check the error slot but
// only *warns* unless `DSV4_STRICT_LAUNCH` is set, and the arena path skips its sync entirely.
//
// WHAT THE CONSUMER THEN READS WAS MEASURED, not assumed, by driving the engine's own decode entry
// point at context 49,207 with the fix switched off (tests/gate_topk_smem_ctx.cu, ladder 1.4's
// engine-path leg). `dtop` in `compressed_decode_step_indexer` comes from `dmalloc`, which is a bump
// allocator and does NOT zero -- so this is not "attends to row 0 forever", it is worse: the step
// completes, `cudaDeviceSynchronize` returns success, the process exits 0, and `out` is 4096/4096
// nonzero with no NaN and a different hash from the correct answer. A wrong answer wearing the
// costume of a right one.
//
// THE FIX IS THE OPT-IN, AND WHERE THE OPT-IN RUNS OUT IT IS AN ABORT. `cudaFuncSetAttribute` with
// `cudaFuncAttributeMaxDynamicSharedMemorySize` raises the ceiling to
// `cudaDevAttrMaxSharedMemoryPerBlockOptin` = 232,448 B here: T = 58,112, context 232,448, a 4.7x
// headroom increase. Above THAT there is no opt-in on any device, so this aborts with the numbers
// instead of returning a size that will fail silently. An abort is the correct outcome: the
// alternative is not "slower", it is "wrong answers that look like answers".
//
// COST IS ZERO IN THE REGIME THE ENGINE ACTUALLY RUNS IN. The request is compared against the
// default limit first and returns immediately when it fits, so at `seqmax` 16,384 (T = 4,096,
// 16 KiB) not one CUDA call is added. Above the default it is one `cudaFuncSetAttribute` per
// kernel for the lifetime of the process -- the attribute is a MAXIMUM, so it is set once to this
// kernel's ceiling and never touched again. That matters because these kernels ARE the
// `DSV4_TOPK_RADIX=0` A/B arm: adding a per-launch host call to them would inflate every future
// re-measurement of 1.2.
//
// WHY OPT IN RATHER THAN DELETE. The shipped path (item 1.2's radix select) requests zero dynamic
// shared memory and cannot hit this at any T, so deleting these four kernels would also close the
// item. It would also delete the `DSV4_TOPK_RADIX=0` comparison arm and the `DSV4_TOPK_GATE=1`
// in-situ bit-exactness reference -- the only thing in this repo that can prove a future top-k
// change still matches the original selection sort. The reference is worth more than the twelve
// lines it costs to keep it correct.
//
// THE KILL-SWITCH IS WHAT MAKES THIS ITEM MEASURABLE, and it is the reason it is here at all.
// `DSV4_TOPK_SMEM_OPTIN=0` restores the EXACT pre-1.4 behaviour in the same binary: no
// `cudaFuncSetAttribute`, no post-launch error check, the size handed back regardless. Without it,
// "the fix works" is a claim about a binary that no longer exists, and the before-arm would have to
// be a git checkout, a rebuild, and a different set of everything-else. With it, before and after
// are two runs of one executable over one input, which is the only form of before/after this ladder
// accepts. It is deliberately loud: restoring a silent-garbage path prints that it did so, every
// time, so a switch left set in a script cannot masquerade as a passing run. Default is ON.
static inline int topk_smem_optin_on(){
    static const int v = getenv("DSV4_TOPK_SMEM_OPTIN") ? atoi(getenv("DSV4_TOPK_SMEM_OPTIN")) : 1;
    return v;
}
static inline size_t topk_scan_smem_optin(const void* fn, int n, const char* what){
    const size_t need = topk_scan_smem(n);
    // Drain the error slot BEFORE the launch so `topk_smem_launch_check` below can attribute what
    // it finds to THIS launch and nothing else. Draining silently would hide someone else's fault,
    // so a pre-existing error is reported here and named as pre-existing -- which is more than the
    // previous behaviour (dsync warns only, and is a no-op entirely while the arena is on).
    // Two host-side TLS reads per fallback launch, against a kernel that takes 30-700 us.
    { cudaError_t pre = cudaGetLastError();
      if(pre != cudaSuccess) fprintf(stderr, "[topk-smem] pre-existing CUDA error before %s: %s\n",
                                     what, cudaGetErrorString(pre)); }
    // `lim_optin` is a DEVICE attribute and is NOT what a given kernel can be granted. MEASURED on
    // this box by bisecting `cudaFuncSetAttribute` over two kernels that differ only in their static
    // shared memory (/tmp/probe_smem2.cu, reproduced in the ladder entry):
    //
    //     optin = 232,448   reserved = 1,024
    //     static 0 B     -> max settable 232,448, and a launch at exactly that size succeeds
    //     static 1,024 B -> max settable 231,424, and a launch at exactly that size succeeds
    //
    // So the rule is `optin - the kernel's own STATIC shared`, and `ReservedSharedMemoryPerBlock` is
    // NOT a second deduction -- it is already accounted for in the opt-in figure. Subtracting it as
    // well looks conservative and is not free: it lowered the ceiling by 1,024 B, which is 256 rows
    // of context, and immediately aborted tests/gate_topk_radix.cu's T = 58,045 leg -- a leg that
    // had passed. A guessed safety margin that fails a passing gate is a bug with good manners.
    // These four scan kernels declare no static shared at all, so for them `lim_fn == lim_optin`;
    // the query is here so the helper stays correct for a kernel that has not been written yet.
    static int lim_default = 0, lim_optin = 0;
    if(!lim_default){
        int dev = 0; cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&lim_default, cudaDevAttrMaxSharedMemoryPerBlock,      dev);
        cudaDeviceGetAttribute(&lim_optin,   cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
        if(!lim_default) lim_default = 48*1024;                       // query failed; assume the floor
        if(lim_optin < lim_default) lim_optin = lim_default;
    }
    int lim_fn = lim_optin;
    { cudaFuncAttributes fa{};
      if(cudaFuncGetAttributes(&fa, fn) == cudaSuccess) lim_fn = lim_optin - (int)fa.sharedSizeBytes;
      if(lim_fn < lim_default) lim_fn = lim_default; }
    if(need <= (size_t)lim_default) return need;                      // the whole reachable regime
    if(!topk_smem_optin_on()){
        fprintf(stderr, "[topk-smem] DSV4_TOPK_SMEM_OPTIN=0: handing %s the pre-1.4 %zu B request at "
                "T=%d against a %d B default limit. This launch WILL fail and the following "
                "synchronize WILL report success. Deliberate; A/B before-arm only.\n",
                what, need, n, lim_default);
        fflush(stderr); return need;
    }
    if(need > (size_t)lim_fn){
        fprintf(stderr, "[topk-smem] FATAL %s wants %zu B of dynamic shared memory for T=%d, but the "
                "most this kernel can be granted on this device is %d B (T <= %d, context <= %d at "
                "ratio 4). Refusing to issue a launch that would fail and leave the output buffer "
                "untouched. Use the radix select (DSV4_TOPK_RADIX=1, the default), which needs "
                "none.\n",
                what, need, n, lim_fn, (int)(lim_fn/4) - 3, ((int)(lim_fn/4) - 3) * 4);
        fflush(stderr); abort();
    }
    static const void* done[8]; static int ndone = 0;                 // one opt-in per kernel, ever
    for(int i = 0; i < ndone; ++i) if(done[i] == fn) return need;
    cudaError_t e = cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, lim_fn);
    if(e != cudaSuccess){
        fprintf(stderr, "[topk-smem] FATAL cudaFuncSetAttribute(%s, MaxDynamicSharedMemorySize, %d) "
                "failed: %s\n", what, lim_fn, cudaGetErrorString(e));
        fflush(stderr); abort();
    }
    if(ndone < 8) done[ndone++] = fn;
    fprintf(stderr, "[topk-smem] %s opted in to %d B dynamic shared memory (needed %zu B at T=%d; "
            "default limit is %d B)\n", what, lim_fn, need, n, lim_default);
    fflush(stderr);
    return need;
}
#define TOPK_SMEM(fn, n) topk_scan_smem_optin((const void*)(fn), (n), #fn)

// A launch that asks for too much dynamic shared memory fails at LAUNCH, and the next
// cudaStreamSynchronize returns success (measured, see above) -- so a sync-based check cannot catch
// it. This is the host-side check that can: it is a pure host call, adds no sync, and turns a
// silently-wrong answer into a stopped process. Placed after every launch that requests a
// context-sized shared allocation.
static inline void topk_smem_launch_check(const char* what, int n, size_t bytes){
    cudaError_t e = cudaGetLastError();
    if(e == cudaSuccess) return;
    if(!topk_smem_optin_on()){
        fprintf(stderr, "[topk-smem] DSV4_TOPK_SMEM_OPTIN=0: launch of %s (T=%d, %zu B dynamic "
                "shared) failed with %s and the check is disabled, so this process now continues on "
                "an UNWRITTEN output buffer. This is what pre-1.4 did on every call.\n",
                what, n, bytes, cudaGetErrorString(e));
        fflush(stderr); return;
    }
    fprintf(stderr, "[topk-smem] FATAL launch of %s (T=%d, %zu B dynamic shared) failed: %s. The "
            "output buffer was NOT written.\n", what, n, bytes, cudaGetErrorString(e));
    fflush(stderr); abort();
}
#define TOPK_LAUNCHED(fn, n) topk_smem_launch_check(#fn, (n), topk_scan_smem(n))

// y[rows,D] = (x[rows,D] @ H_D) * D^-0.5, H_D[i,j] = (-1)^popcount(i&j). D must be a power of two.
void hadamard(float* y, const float* x, int rows, int D, cudaStream_t stream = 0);
// DECODE_LADDER 1.10. `hadamard(y,y)` -- three call sites do it -- was a read/write race: every
// thread of a row reads the whole row and overwrites one element of it. The shipped kernel stages
// the row in shared memory first and is bit-identical wherever the buffers were already distinct.
// This flips the arm inside ONE process (0 = the pre-1.10 flat kernel); DSV4_HADAMARD_STAGE=0 is
// the same switch from the environment. tests/gate_hadamard_alias.cu.
void hadamard_set_stage(int on);

// index_score[s,t] = Σ_h relu(Σ_d q[s,h,d]*kv[t,d]) * weights[s,h].
// q:[S,H,d]  kv:[T,d]  weights:[S,H]  -> score:[S,T].
void index_score(float* score, const float* q, const float* kv, const float* weights,
                 int S, int T, int H, int d, cudaStream_t stream = 0);

// DECODE_LADDER 1.5. Explicit-implementation entry point, for gates and A/B only: the shipped
// `index_score` picks IXS_TILED and falls back leftward. Returns false when `impl` cannot serve the
// shape (tiled is templated on `d`; warp needs d%32==0), so a caller can chain.
enum { IXS_SCALAR = 0, IXS_WARP = 1, IXS_TILED = 2, IXS_GEMM = 3 };
bool index_score_impl(int impl, float* score, const float* q, const float* kv, const float* weights,
                      int S, int T, int H, int d, cudaStream_t stream = 0);

// Full DSA Indexer forward (prefill, b=1). wq_b:[n_heads*idx_hd, q_lora] fp8 + scale; weights_proj:[n_heads,dim];
// rotate-compressor weights (c_*); q_cos/q_sin:[s,rd/2] (query freqs), c_cos/c_sin:[s/ratio,rd/2] (compressed).
// Outputs: index_score:[s, s/ratio] (post causal-mask, for gating) and topk_idxs:[s, min(index_topk,s/ratio)]
// (offset-applied, -1 where masked). offset = position base for the compressed idxs in the KV.
void indexer_forward(float* index_score_out, int* topk_idxs, const float* x, const float* qr,
                     const unsigned char* wq_b, const float* wq_b_s, const float* weights_proj,
                     const float* c_wkv, const float* c_wgate, const float* c_ape, const float* c_norm,
                     const float* q_cos, const float* q_sin, const float* c_cos, const float* c_sin,
                     int s, int dim, int q_lora, int n_heads, int idx_hd, int rd, int ratio,
                     int index_topk, int offset, float eps, cudaStream_t stream = 0);
