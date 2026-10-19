// mla_attn.cu — MLA attention primitives, correctness-first (Gate K oracle: ref/gen_units.py).
// Optimization (mma, smem KV staging, bf16) comes AFTER these pass their gate (CONSTITUTION Art. I).
#include <cstdint>
#include <cstdlib>
#include <cuda_fp16.h>
#include "dscratch.h"
#include "mla_attn.h"
#include "kv_pack.h"
#include "nvfp4_dense.h"

// ---------------- sparse_attn ----------------
// One warp per (b, m, head): stream over the top-k gathered KV with online (running max/sum) softmax,
// then fold in the learnable sink (denominator-only). MLA => single latent KV shared across heads.
// Lanes hold d/32 elements of the accumulator and of q; scores are warp-reduced dot products.
__global__ void sparse_attn_kernel(float* __restrict__ o, const float* __restrict__ q,
                                   const float* __restrict__ kv, const float* __restrict__ attn_sink,
                                   const int* __restrict__ topk_idxs,
                                   int b, int m, int h, int d, int n, int topk, float scale) {
    int gid = blockIdx.x;                       // (b*m*h) index
    int head = gid % h; int bm = gid / h; int mi = bm % m; int bi = bm / m;
    int lane = threadIdx.x & 31;
    int per = (d + 31) / 32;                    // elems per lane along d
    const float* qp = q + (((size_t)(bi * m + mi) * h + head) * d);
    const int*   ip = topk_idxs + ((size_t)(bi * m + mi) * topk);

    float qreg[32];                             // per<=32 for d<=1024
    #pragma unroll
    for (int r = 0; r < 32; ++r) { int j = lane + r * 32; qreg[r] = (r < per && j < d) ? qp[j] : 0.f; }

    float acc[32]; for (int r = 0; r < 32; ++r) acc[r] = 0.f;
    float run_max = -1e30f, run_sum = 0.f;

    for (int t = 0; t < topk; ++t) {
        int idx = ip[t];
        if (idx < 0) continue;                  // masked slot
        const float* kp = kv + (((size_t)bi * n + idx) * d);
        // score = scale * dot(q, kv[idx])
        float part = 0.f;
        #pragma unroll
        for (int r = 0; r < 32; ++r) { int j = lane + r * 32; if (r < per && j < d) part += qreg[r] * kp[j]; }
        #pragma unroll
        for (int o2 = 16; o2 > 0; o2 >>= 1) part += __shfl_down_sync(0xffffffff, part, o2);
        float score = __shfl_sync(0xffffffff, part, 0) * scale;   // broadcast to all lanes
        // online softmax update
        float new_max = fmaxf(run_max, score);
        float corr = expf(run_max - new_max);
        float p = expf(score - new_max);
        run_sum = run_sum * corr + p;
        #pragma unroll
        for (int r = 0; r < 32; ++r) { int j = lane + r * 32; if (r < per && j < d) acc[r] = acc[r] * corr + p * kp[j]; }
        run_max = new_max;
    }
    // sink: contributes exp(sink-max) to denominator only
    run_sum += expf(attn_sink[head] - run_max);
    float inv = (run_sum > 0.f) ? 1.f / run_sum : 0.f;
    float* op = o + (((size_t)(bi * m + mi) * h + head) * d);
    #pragma unroll
    for (int r = 0; r < 32; ++r) { int j = lane + r * 32; if (r < per && j < d) op[j] = acc[r] * inv; }
}

// ===================== B9: prefill-shaped sparse attention =====================
// The kernel above is M=1-shaped and prefill inherited it unchanged: 2.927 s = 13.7% of a 1022-token
// prefill at 0.72 TFLOPS (F84). Two bit-exact defects, both invisible at m=1:
//
//  (1) REGISTERS. qreg[32]/acc[32] are sized for d<=1024 but every caller in this engine passes
//      d = HEAD_DIM = 512, so per = 16 and HALF of both arrays is dead -- 32 float registers per
//      thread burned on a kernel whose problem is occupancy. PER as a template parameter sizes them
//      to the real d.
//  (2) KEY REUSE. num_key_value_heads == 1, so `kv` has NO head dimension and all h=64 heads of a
//      query read the IDENTICAL key vectors. Those were 64 separate 32-thread blocks, free to land
//      on 64 different SMs, each re-reading the same 2 KB. Putting HPB heads of one query in ONE
//      block puts them on ONE SM, where the second through HPB-th reads hit L1.
//
// Both preserve the per-(query,head) accumulation order over t exactly -- the online-softmax state
// and the lane reduction are untouched -- so this is BIT-EXACT, not merely close.
//
// HPB MUST FOLLOW THE BATCH, and this is the trap: at m=1 decode there are only b*m*h = 64 warps
// total, so HPB=8 would launch 8 blocks onto 20 SMs and starve the machine. Size HPB so the grid
// still covers the device (>= 4 blocks/SM), which gives HPB=1 at decode -- the original kernel,
// byte for byte -- and HPB=8 at a 1022-token prefill.
template<int PER, int HPB, bool PACKED = false>
__global__ void sparse_attn_kernel_t(float* __restrict__ o, const float* __restrict__ q,
                                     const float* __restrict__ kv, const float* __restrict__ attn_sink,
                                     const int* __restrict__ topk_idxs,
                                     int b, int m, int h, int d, int n, int topk, float scale) {
    int gid = blockIdx.x * HPB + (int)(threadIdx.x >> 5);
    if (gid >= b * m * h) return;
    int head = gid % h; int bm = gid / h; int mi = bm % m; int bi = bm / m;
    int lane = threadIdx.x & 31;
    const float* qp = q + (((size_t)(bi * m + mi) * h + head) * d);
    const int*   ip = topk_idxs + ((size_t)(bi * m + mi) * topk);

    float qreg[PER];
    #pragma unroll
    for (int r = 0; r < PER; ++r) { int j = lane + r * 32; qreg[r] = (j < d) ? qp[j] : 0.f; }

    float acc[PER];
    #pragma unroll
    for (int r = 0; r < PER; ++r) acc[r] = 0.f;
    float run_max = -1e30f, run_sum = 0.f;

    for (int t = 0; t < topk; ++t) {
        int idx = ip[t];
        if (idx < 0) continue;
        const float* kp = kv + (((size_t)bi * n + idx) * d);
        // 1b.2. With PACKED=false the ternary folds away at compile time, so the FP32 path below is
        // the identical instruction stream `gate_sparse_hpb` uses as its memcmp oracle.
        const uint8_t* kp8 = (const uint8_t*)kv + ((size_t)bi * (size_t)n + idx) * dsv4kv::KVP_ROWB;
        float part = 0.f;
        #pragma unroll
        for (int r = 0; r < PER; ++r) { int j = lane + r * 32; if (j < d) part += qreg[r] * (PACKED ? dsv4kv::kv_unpack(kp8, j) : kp[j]); }
        #pragma unroll
        for (int o2 = 16; o2 > 0; o2 >>= 1) part += __shfl_down_sync(0xffffffff, part, o2);
        float score = __shfl_sync(0xffffffff, part, 0) * scale;
        float new_max = fmaxf(run_max, score);
        float corr = expf(run_max - new_max);
        float p = expf(score - new_max);
        run_sum = run_sum * corr + p;
        #pragma unroll
        for (int r = 0; r < PER; ++r) { int j = lane + r * 32; if (j < d) acc[r] = acc[r] * corr + p * (PACKED ? dsv4kv::kv_unpack(kp8, j) : kp[j]); }
        run_max = new_max;
    }
    run_sum += expf(attn_sink[head] - run_max);
    float inv = (run_sum > 0.f) ? 1.f / run_sum : 0.f;
    float* op = o + (((size_t)(bi * m + mi) * h + head) * d);
    #pragma unroll
    for (int r = 0; r < PER; ++r) { int j = lane + r * 32; if (j < d) op[j] = acc[r] * inv; }
}

// ---- 1b.2: stage one PACKED (720 B) KV row into the shared FP32 row the kernel already uses ----
// 112 uchar4 loads for the 448 E4M3 codes + 16 float4 loads for the FP32 RoPE half = 128 vector
// loads per row, the SAME COUNT the FP32 layout issues, over 720 B instead of 2048 B.
//
// THE SHAPE OF THIS LOOP IS THE WHOLE COST, AND THE FIRST VERSION GOT IT WRONG. It read a uint4 --
// 16 codes per thread -- which is only 28 vector loads for the 448 codes and looks like the better
// number. It is not: the block is 32*HPB threads (128 at the default HPB=4), so 28 of 128 threads
// did SIXTEEN unpacks each while 100 waited, on a staging step that sits on the double-buffered
// prefetch's critical path. `gate_kv_pack --bench` measured that at **0.70-0.79x of the FP32
// launch** -- 0.509 -> 0.728 ms at the base-AR shape -- and the engine agreed: +5.00 +/- 0.37 ms
// per forward, which is 43 sparse_attn calls x ~0.19 ms. Four elements per thread instead of
// sixteen puts 112 of 128 threads to work for one round each.
//
// Per thread: one 4 B load, TWO `cvt.rn.f16x2.e4m3x2` (the packed-pair intrinsic converts two
// codes per instruction), four multiplies by the group's power-of-two scale, and one 16 B shared
// store -- `sh` is __align__(16) and j is a multiple of 4, so the float4 store is aligned. The four
// elements a thread owns never straddle a 64-wide quant group (4 divides 64), so one scale byte is
// read per thread and the whole row costs 112 scale-byte reads, all L1 hits.
__device__ __forceinline__ void kvp_stage_row(float* sh, const uint8_t* p, int tid, int nthr) {
    using namespace dsv4kv;
    for (int v = tid; v < KVP_NOPE / 4; v += nthr) {
        const uchar4 c = ((const uchar4*)p)[v];
        const int j = v * 4;
        const float sc = kv_pow2(p[KVP_OFF_SCALE + (j >> 6)]);
        __half2_raw h0 = __nv_cvt_fp8x2_to_halfraw2(
            (__nv_fp8x2_storage_t)((uint16_t)c.x | ((uint16_t)c.y << 8)), __NV_E4M3);
        __half2_raw h1 = __nv_cvt_fp8x2_to_halfraw2(
            (__nv_fp8x2_storage_t)((uint16_t)c.z | ((uint16_t)c.w << 8)), __NV_E4M3);
        const float2 f0 = __half22float2(*reinterpret_cast<__half2*>(&h0));
        const float2 f1 = __half22float2(*reinterpret_cast<__half2*>(&h1));
        *(float4*)(sh + j) = make_float4(f0.x * sc, f0.y * sc, f1.x * sc, f1.y * sc);
    }
    for (int v = tid; v < KVP_ROPE / 4; v += nthr)
        ((float4*)(sh + KVP_NOPE))[v] = ((const float4*)(p + KVP_OFF_ROPE))[v];
}

// ===================== 1.7: the same kernel, but the block reads the row ONCE =================
// DECODE_LADDER 1.7. `cattn:sparse` is ~20 ms of every forward above ctx 2048 and the shape says
// why: topk = WINDOW(128) + min(INDEX_TOPK(512), ctx/ratio), so it doubles over the first 2k and is
// FLAT after -- the ~11 ms floor at ctx 768 and the ~21 ms ceiling are the same kernel at topk 320
// and 640. What it moves at the ceiling is topk x 2 KB x (m*h) = 168 MB per compressed layer at the
// K=2 verify width, and 63/64ths of that is the SAME 640 rows re-read once per head, because
// num_key_value_heads == 1: `ip` is indexed by (b,m) and NOT by head, so every head of a query
// gathers the identical latent KV rows.
//
// TWO SEPARATE THINGS ARE WRONG AND HPB ALONE ONLY FIXES ONE.
//  (1) REUSE. HPB>1 puts HPB heads of one query in one block, so warps 2..HPB hit L1 instead of L2.
//      That is `sparse_attn_kernel_t` above and it already exists -- but the launch heuristic gives
//      HPB=1 below 160 warps, which is exactly the decode/verify regime (b*m*h = 64 at m=1, 128 at
//      the measured mean verify width of 2).
//  (2) ISSUE PRESSURE. Every warp issues 32 scalar loads per t (16 for the dot, 16 more for the
//      accumulate) at stride 128 B. They are perfectly coalesced, so this is not a byte problem --
//      it is 32 LSU instructions per 2 KB, on a kernel with 3-6 warps per SM and a 5-deep
//      __shfl_down_sync chain on the critical path, i.e. nothing to hide them behind.
//
// This kernel fixes (2) without touching the arithmetic. The block stages the gathered row into
// shared memory ONCE per t with float4 loads -- 4 vector loads per row for the whole block instead
// of 32 scalar loads per warp -- double-buffered so t+1's fetch is issued before t's dot product,
// and then each warp reads its lanes back out of smem IN THE ORIGINAL MAPPING (lane + r*32).
//
// WHY THIS IS BIT-EXACT AND NOT MERELY CLOSE. Vectorising the GLOBAL load would normally destroy
// bit-exactness, because float4 changes which lane owns which element and therefore the shape of
// the partial-sum tree. Staging through smem decouples the two: the load pattern becomes a pure
// memory-placement decision, while the reduction still sees lane l holding {l, l+32, ..., l+480},
// the same 5-step shuffle tree, and the same online-softmax update order over t. Every float
// operation, in every order, is the one `sparse_attn_kernel_t` performs. `gate_sparse_hpb` proves
// it by memcmp against the HPB=1 launch rather than by this paragraph.
//
// THE PREFETCH BRANCH IS BLOCK-UNIFORM, which is what makes the __syncthreads legal: `ip` does not
// depend on head, so `idx < 0` (the masked slot) is the same answer for every warp in the block.
// The dispatcher enforces the other precondition -- h % HPB == 0, so every warp of a block shares
// one (b, m) and there is no partial block to diverge on.
//
// PACKED (DECODE_LADDER 1b.2) changes ONLY the staging loop: the block reads a 720 B packed row and
// writes the SAME 512 floats into the SAME shared buffer, so every line below the staging -- the
// lane mapping, the shuffle tree, the online-softmax order over t -- is one copy of the source, not
// two that have to be argued equal. The unpack is exact (kv_pack.h), so the floats in `sh` are
// bit-identical to the ones the FP32 path stages, and therefore so is the output.
template<int PER, int HPB, bool KREG, bool PACKED = false>
__global__ __launch_bounds__(32*HPB)
void sparse_attn_kernel_s(float* __restrict__ o, const float* __restrict__ q,
                          const float* __restrict__ kv, const float* __restrict__ attn_sink,
                          const int* __restrict__ topk_idxs,
                          int b, int m, int h, int d, int n, int topk, float scale) {
    constexpr int D  = PER * 32;
    constexpr int NV = D / 4;
    __shared__ __align__(16) float sh[2][D];
    const int lane = threadIdx.x & 31;
    const int gid0 = blockIdx.x * HPB;              // block-uniform; h % HPB == 0 => one (bi,mi)
    const int gid  = gid0 + (int)(threadIdx.x >> 5);
    const int bm = gid0 / h; const int mi = bm % m, bi = bm / m;
    const int head = gid % h;
    const float* qp  = q + (((size_t)(bi * m + mi) * h + head) * d);
    const int*   ip  = topk_idxs + ((size_t)(bi * m + mi) * topk);
    const float* kvb = kv + (size_t)bi * n * d;
    const uint8_t* kvb8 = (const uint8_t*)kv + (size_t)bi * (size_t)n * dsv4kv::KVP_ROWB;
    const int nthr = 32 * HPB;

    float qreg[PER];
    #pragma unroll
    for (int r = 0; r < PER; ++r) { int j = lane + r * 32; qreg[r] = (j < d) ? qp[j] : 0.f; }
    float acc[PER];
    #pragma unroll
    for (int r = 0; r < PER; ++r) acc[r] = 0.f;
    float run_max = -1e30f, run_sum = 0.f;

    if (topk > 0) { int i0 = ip[0];
        if (i0 >= 0) {
            if (PACKED) kvp_stage_row(sh[0], kvb8 + (size_t)i0 * dsv4kv::KVP_ROWB, threadIdx.x, nthr);
            else { const float4* src = (const float4*)(kvb + (size_t)i0 * d);
                   for (int v = threadIdx.x; v < NV; v += nthr) ((float4*)sh[0])[v] = src[v]; } } }
    __syncthreads();

    for (int t = 0; t < topk; ++t) {
        const int idx = ip[t];
        const int cur = t & 1;
        if (t + 1 < topk) { int i1 = ip[t + 1];
            if (i1 >= 0) {
                if (PACKED) kvp_stage_row(sh[cur ^ 1], kvb8 + (size_t)i1 * dsv4kv::KVP_ROWB, threadIdx.x, nthr);
                else { const float4* src = (const float4*)(kvb + (size_t)i1 * d);
                       for (int v = threadIdx.x; v < NV; v += nthr) ((float4*)sh[cur ^ 1])[v] = src[v]; } } }
        if (idx >= 0) {
            const float* kp = sh[cur];
            // KREG: hold this lane's PER elements of the row in registers across BOTH consumers.
            // The dot product and the accumulate read the identical PER values, but `__shfl_sync`
            // sits between them and blocks the compiler from reusing the first set of loads, so the
            // row is fetched from shared memory TWICE -- 32 LSU instructions per t on a kernel whose
            // measured problem is instruction issue. Same values, same order: bit-exact.
            float kreg[KREG ? PER : 1];
            if (KREG) {
                #pragma unroll
                for (int r = 0; r < PER; ++r) { int j = lane + r * 32; kreg[KREG ? r : 0] = (j < d) ? kp[j] : 0.f; }
            }
            float part = 0.f;
            #pragma unroll
            for (int r = 0; r < PER; ++r) { int j = lane + r * 32; if (j < d) part += qreg[r] * (KREG ? kreg[KREG ? r : 0] : kp[j]); }
            #pragma unroll
            for (int o2 = 16; o2 > 0; o2 >>= 1) part += __shfl_down_sync(0xffffffff, part, o2);
            float score = __shfl_sync(0xffffffff, part, 0) * scale;
            float new_max = fmaxf(run_max, score);
            // expf(0.f) is exactly 1.f, so skipping the call when the running max did not move is a
            // value-preserving branch and not an approximation. It is warp-uniform: `score` is the
            // lane-0 broadcast, so run_max and new_max are the same in every lane.
            float corr = (new_max == run_max) ? 1.f : expf(run_max - new_max);
            float p = expf(score - new_max);
            run_sum = run_sum * corr + p;
            #pragma unroll
            for (int r = 0; r < PER; ++r) { int j = lane + r * 32; if (j < d) acc[r] = acc[r] * corr + p * (KREG ? kreg[KREG ? r : 0] : kp[j]); }
            run_max = new_max;
        }
        __syncthreads();
    }
    run_sum += expf(attn_sink[head] - run_max);
    float inv = (run_sum > 0.f) ? 1.f / run_sum : 0.f;
    float* op = o + (((size_t)(bi * m + mi) * h + head) * d);
    #pragma unroll
    for (int r = 0; r < PER; ++r) { int j = lane + r * 32; if (j < d) op[j] = acc[r] * inv; }
}

void sparse_attn_launch(float* o, const float* q, const float* kv, const float* attn_sink,
                        const int* topk_idxs, int b, int m, int h, int d, int n, int topk,
                        float scale, cudaStream_t stream, int hpb, int smem, bool packed) {
    long total = (long)b * m * h;
    if (hpb != 1 && hpb != 2 && hpb != 4 && hpb != 8) hpb = 1;
    // PER is only specialised for the d this engine actually uses; anything else takes the
    // original kernel rather than a guessed specialisation.
    if (d != 512) {
        if (packed) { fprintf(stderr, "sparse_attn: packed KV requires d==512, got %d\n", d); abort(); }
        sparse_attn_kernel<<<(int)total, 32, 0, stream>>>(o, q, kv, attn_sink, topk_idxs, b, m, h, d, n, topk, scale);
        return;
    }
    if (h % hpb) hpb = 1;                       // every warp of a block must share one (b, m)
    // The smem path stages the row with float4, so a caller that hands us an unaligned base gets the
    // scalar path rather than a misaligned access. Every allocator in this engine is 256 B-aligned
    // and the row stride is d*4 = 2048 B, so this never fires today -- it fires the day it does.
    // A packed row is 720 B, also a multiple of 16, and the same test covers it (1b.2).
    if ((((uintptr_t)kv) & 15) || ((size_t)(packed ? dsv4kv::KVP_ROWB : d * 4) & 15)) smem = 0;
    if (hpb == 1) smem = 0;                     // hpb=1 + smem measured 0.71x: no reuse to pay for the sync
    int blocks = (int)((total + hpb - 1) / hpb);
    if (packed) {
        if (smem >= 2) {
            switch (hpb) {
                case 1: sparse_attn_kernel_s<16,1,true,true><<<blocks, 32,0,stream>>>(o,q,kv,attn_sink,topk_idxs,b,m,h,d,n,topk,scale); return;
                case 2: sparse_attn_kernel_s<16,2,true,true><<<blocks, 64,0,stream>>>(o,q,kv,attn_sink,topk_idxs,b,m,h,d,n,topk,scale); return;
                case 4: sparse_attn_kernel_s<16,4,true,true><<<blocks,128,0,stream>>>(o,q,kv,attn_sink,topk_idxs,b,m,h,d,n,topk,scale); return;
                default:sparse_attn_kernel_s<16,8,true,true><<<blocks,256,0,stream>>>(o,q,kv,attn_sink,topk_idxs,b,m,h,d,n,topk,scale); return;
            }
        }
        if (smem) {
            switch (hpb) {
                case 1: sparse_attn_kernel_s<16,1,false,true><<<blocks, 32,0,stream>>>(o,q,kv,attn_sink,topk_idxs,b,m,h,d,n,topk,scale); return;
                case 2: sparse_attn_kernel_s<16,2,false,true><<<blocks, 64,0,stream>>>(o,q,kv,attn_sink,topk_idxs,b,m,h,d,n,topk,scale); return;
                case 4: sparse_attn_kernel_s<16,4,false,true><<<blocks,128,0,stream>>>(o,q,kv,attn_sink,topk_idxs,b,m,h,d,n,topk,scale); return;
                default:sparse_attn_kernel_s<16,8,false,true><<<blocks,256,0,stream>>>(o,q,kv,attn_sink,topk_idxs,b,m,h,d,n,topk,scale); return;
            }
        }
        switch (hpb) {
            case 1: sparse_attn_kernel_t<16,1,true><<<blocks, 32,0,stream>>>(o,q,kv,attn_sink,topk_idxs,b,m,h,d,n,topk,scale); return;
            case 2: sparse_attn_kernel_t<16,2,true><<<blocks, 64,0,stream>>>(o,q,kv,attn_sink,topk_idxs,b,m,h,d,n,topk,scale); return;
            case 4: sparse_attn_kernel_t<16,4,true><<<blocks,128,0,stream>>>(o,q,kv,attn_sink,topk_idxs,b,m,h,d,n,topk,scale); return;
            default:sparse_attn_kernel_t<16,8,true><<<blocks,256,0,stream>>>(o,q,kv,attn_sink,topk_idxs,b,m,h,d,n,topk,scale); return;
        }
    }
    if (smem >= 2) {
        switch (hpb) {
            case 1: sparse_attn_kernel_s<16,1,true><<<blocks, 32,0,stream>>>(o,q,kv,attn_sink,topk_idxs,b,m,h,d,n,topk,scale); return;
            case 2: sparse_attn_kernel_s<16,2,true><<<blocks, 64,0,stream>>>(o,q,kv,attn_sink,topk_idxs,b,m,h,d,n,topk,scale); return;
            case 4: sparse_attn_kernel_s<16,4,true><<<blocks,128,0,stream>>>(o,q,kv,attn_sink,topk_idxs,b,m,h,d,n,topk,scale); return;
            default:sparse_attn_kernel_s<16,8,true><<<blocks,256,0,stream>>>(o,q,kv,attn_sink,topk_idxs,b,m,h,d,n,topk,scale); return;
        }
    }
    if (smem) {
        switch (hpb) {
            case 1: sparse_attn_kernel_s<16,1,false><<<blocks, 32,0,stream>>>(o,q,kv,attn_sink,topk_idxs,b,m,h,d,n,topk,scale); return;
            case 2: sparse_attn_kernel_s<16,2,false><<<blocks, 64,0,stream>>>(o,q,kv,attn_sink,topk_idxs,b,m,h,d,n,topk,scale); return;
            case 4: sparse_attn_kernel_s<16,4,false><<<blocks,128,0,stream>>>(o,q,kv,attn_sink,topk_idxs,b,m,h,d,n,topk,scale); return;
            default:sparse_attn_kernel_s<16,8,false><<<blocks,256,0,stream>>>(o,q,kv,attn_sink,topk_idxs,b,m,h,d,n,topk,scale); return;
        }
    }
    switch (hpb) {
        case 1: sparse_attn_kernel_t<16,1><<<blocks, 32,0,stream>>>(o,q,kv,attn_sink,topk_idxs,b,m,h,d,n,topk,scale); return;
        case 2: sparse_attn_kernel_t<16,2><<<blocks, 64,0,stream>>>(o,q,kv,attn_sink,topk_idxs,b,m,h,d,n,topk,scale); return;
        case 4: sparse_attn_kernel_t<16,4><<<blocks,128,0,stream>>>(o,q,kv,attn_sink,topk_idxs,b,m,h,d,n,topk,scale); return;
        default:sparse_attn_kernel_t<16,8><<<blocks,256,0,stream>>>(o,q,kv,attn_sink,topk_idxs,b,m,h,d,n,topk,scale); return;
    }
}

// THE DEFAULT, RETUNED BY `gate_sparse_hpb` 2026-08-20 (DECODE_LADDER 1.7). Every number below is
// that gate's, at the shapes the engine issues, memcmp-clean against the pre-1.7 launch:
//
//   shape                          hpb=1/smem=0   hpb=8/smem=0   hpb=4/smem=1   hpb=4/smem=2
//   m=1   topk=640  (base AR)         0.737 ms      0.771 (0.96x)  0.593 (1.24x)  0.506 (1.46x)
//   m=2   topk=640  (mean verify)     0.746         0.773 (0.97x)  0.613 (1.22x)  0.548 (1.36x)
//   m=6   topk=640  (max verify)      0.818         1.581 (0.52x)  0.734 (1.11x)  0.719 (1.14x)
//   m=2   topk=320  (ctx 768)         0.375         0.389 (0.96x)  0.309 (1.21x)  0.281 (1.33x)
//   m=256 topk=320  (short prefill)  15.78         20.47  (0.77x) 13.03  (1.21x) 13.86  (1.14x)
//   m=1022 topk=1277 (prefill)      263.2         328.5   (0.80x) 205.1  (1.28x) 220.4  (1.19x)
//
// TWO THINGS IN THAT TABLE. First, `hpb` ON ITS OWN IS A NULL -- 1.00x at hpb=2 and hpb=4 at every
// shape -- so B9's REUSE story was not the binding constraint: the redundant per-head re-reads were
// already being caught by L1. The win is entirely the smem staging, i.e. ISSUE PRESSURE. Second,
// THE OLD DEFAULT WAS A REGRESSION AT ITS OWN DESIGN POINT: hpb=8/smem=0 is 0.80x of hpb=1 at the
// 1022-token prefill B9 sized it on, and 0.52x at the K=6 verify. That number was never re-measured
// after the shape moved. See wiki/measurement-and-traps.md.
//
// KREG (smem=2) holds the row in registers across the dot and the accumulate, costing PER=16 extra
// registers. It wins where the kernel is LATENCY-bound and there are few blocks (decode: 1.36x vs
// 1.22x) and loses where it is OCCUPANCY-bound and there are thousands (prefill: 1.19x vs 1.28x),
// which is why the switch is on `total` and not a constant.
void sparse_attn(float* o, const float* q, const float* kv, const float* attn_sink,
                 const int* topk_idxs, int b, int m, int h, int d, int n, int topk,
                 float scale, cudaStream_t stream) {
    long total = (long)b * m * h;
    // DSV4_SPARSE_HPB=1 DSV4_SPARSE_SMEM=0 restores the pre-1.7 launch exactly, for the A/B.
    static const char* hpb_env  = getenv("DSV4_SPARSE_HPB");
    static const char* smem_env = getenv("DSV4_SPARSE_SMEM");
    int hpb  = hpb_env  ? atoi(hpb_env)  : 4;
    int smem = smem_env ? atoi(smem_env) : (total <= 1024L ? 2 : 1);
    sparse_attn_launch(o, q, kv, attn_sink, topk_idxs, b, m, h, d, n, topk, scale, stream, hpb, smem, false);
}

// DECODE_LADDER 1b.2. The KV-CACHE entry point: identical to `sparse_attn` except that it honours
// `g_kv_pack`, so the cache call sites read whichever layout the caches were allocated in while
// every other caller (the one-shot prefill temporaries, the unit gates) keeps calling `sparse_attn`
// and keeps its FP32 contract.
void sparse_attn_kv(float* o, const float* q, const float* kv, const float* attn_sink,
                    const int* topk_idxs, int b, int m, int h, int d, int n, int topk,
                    float scale, cudaStream_t stream) {
    long total = (long)b * m * h;
    static const char* hpb_env  = getenv("DSV4_SPARSE_HPB");
    static const char* smem_env = getenv("DSV4_SPARSE_SMEM");
    int hpb  = hpb_env  ? atoi(hpb_env)  : 4;
    int smem = smem_env ? atoi(smem_env) : (total <= 1024L ? 2 : 1);
    sparse_attn_launch(o, q, kv, attn_sink, topk_idxs, b, m, h, d, n, topk, scale, stream, hpb, smem,
                       g_kv_pack != 0);
}

// ---------------- rope_interleaved ----------------
// x[rows, rope_dim]; pairs (2j,2j+1) rotated by (cos_j, sin_j). inverse => sin -> -sin.
__global__ void rope_kernel(float* __restrict__ x, const float* __restrict__ cosT,
                            const float* __restrict__ sinT, int rows, int rope_dim, int inv,
                            int row_stride, int cos_stride_rows) {
    int row = blockIdx.x; if (row >= rows) return;
    int half = rope_dim / 2;
    float* xr = x + (size_t)row * row_stride;
    int crow = row / cos_stride_rows;
    const float* c = cosT + (size_t)crow * half;
    const float* s = sinT + (size_t)crow * half;
    for (int j = threadIdx.x; j < half; j += blockDim.x) {
        float a = xr[2 * j], bb = xr[2 * j + 1];
        float sj = inv ? -s[j] : s[j], cj = c[j];
        xr[2 * j]     = a * cj - bb * sj;
        xr[2 * j + 1] = a * sj + bb * cj;
    }
}

void rope_interleaved(float* x, const float* cosT, const float* sinT,
                      int rows, int rope_dim, bool inverse, int row_stride, int cos_stride_rows,
                      cudaStream_t stream) {
    if (row_stride < 0) row_stride = rope_dim;
    rope_kernel<<<rows, 64, 0, stream>>>(x, cosT, sinT, rows, rope_dim, inverse ? 1 : 0,
                                         row_stride, cos_stride_rows);
}
// DEVICE-POS rope (for CUDA-graph capture): cos row = (*d_pos) + row/cos_stride_rows, read from the FULL table.
// M=1 decode: q rows=N_HEADS,stride=N_HEADS -> crow=pos; kv rows=1,stride=1 -> crow=pos. No pos baked into args.
__global__ void rope_kernel_dp(float* __restrict__ x, const float* __restrict__ cosT, const float* __restrict__ sinT,
                               int rows, int rope_dim, int inv, int row_stride, int cos_stride_rows, const int* __restrict__ d_pos){
    int row=blockIdx.x; if(row>=rows) return; int half=rope_dim/2;
    float* xr = x + (size_t)row*row_stride; int crow=(*d_pos) + row/cos_stride_rows;
    const float* c=cosT + (size_t)crow*half; const float* s=sinT + (size_t)crow*half;
    for(int j=threadIdx.x;j<half;j+=blockDim.x){ float a=xr[2*j], bb=xr[2*j+1]; float sj=inv?-s[j]:s[j], cj=c[j];
        xr[2*j]=a*cj-bb*sj; xr[2*j+1]=a*sj+bb*cj; }
}
void rope_interleaved_dp(float* x, const float* cosT, const float* sinT, int rows, int rope_dim, bool inverse,
                         int row_stride, int cos_stride_rows, const int* d_pos, cudaStream_t stream){
    if(row_stride<0) row_stride=rope_dim;
    rope_kernel_dp<<<rows,64,0,stream>>>(x,cosT,sinT,rows,rope_dim,inverse?1:0,row_stride,cos_stride_rows,d_pos);
}

// ---------------- rmsnorm ----------------
__global__ void rmsnorm_kernel(float* __restrict__ y, const float* __restrict__ x,
                               const float* __restrict__ w, int rows, int dim, float eps, int has_w) {
    int row = blockIdx.x; if (row >= rows) return;
    const float* xr = x + (size_t)row * dim; float* yr = y + (size_t)row * dim;
    __shared__ float red[256];
    float ss = 0.f; for (int j = threadIdx.x; j < dim; j += blockDim.x) ss += xr[j] * xr[j];
    red[threadIdx.x] = ss; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) { if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s]; __syncthreads(); }
    float inv = rsqrtf(red[0] / dim + eps);
    for (int j = threadIdx.x; j < dim; j += blockDim.x) yr[j] = xr[j] * inv * (has_w ? w[j] : 1.f);
}

void rmsnorm(float* y, const float* x, const float* weight, int rows, int dim,
             float eps, bool has_weight, cudaStream_t stream) {
    rmsnorm_kernel<<<rows, 256, 0, stream>>>(y, x, weight, rows, dim, eps, has_weight ? 1 : 0);
}

// ---------------- act_quant_fp8sim ----------------
// Per (row, block-of-`block`): amax -> pow2 scale (ue8m0) -> clamp(x/scale) to e4m3 -> dequant*scale.
// One block per (row, group); threads cover the group. Matches kernel.py act_quant(inplace, round_scale).
#include <cuda_fp8.h>
__global__ void act_quant_fp8sim_kernel(float* __restrict__ x, int rows, int active_dim, int block, int row_stride) {
    int ng = active_dim / block; int gid = blockIdx.x; if (gid >= rows * ng) return;
    int row = gid / ng, g = gid % ng;
    float* xr = x + (size_t)row * row_stride + (size_t)g * block;
    extern __shared__ float red[];
    float v = (threadIdx.x < block) ? fabsf(xr[threadIdx.x]) : 0.f;
    red[threadIdx.x] = v; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) { if (threadIdx.x < s) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]); __syncthreads(); }
    float amax = fmaxf(red[0], 1e-4f);
    float scale = exp2f(ceilf(log2f(amax * (1.f / 448.f))));      // pow2 (ue8m0)
    if (threadIdx.x < block) {
        float q = fminf(fmaxf(xr[threadIdx.x] / scale, -448.f), 448.f);
        __nv_fp8_e4m3 e = __nv_fp8_e4m3(q);
        __half_raw hr = __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)e.__x, __NV_E4M3);
        xr[threadIdx.x] = __half2float(*reinterpret_cast<__half*>(&hr)) * scale;
    }
}
void act_quant_fp8sim(float* x, int rows, int active_dim, int block, int row_stride, cudaStream_t stream) {
    if (row_stride < 0) row_stride = active_dim;
    int threads = block < 32 ? 32 : block;
    act_quant_fp8sim_kernel<<<rows * (active_dim / block), threads, threads * sizeof(float), stream>>>(x, rows, active_dim, block, row_stride);
}

// Real activation quant -> fp8 bytes + f32 pow2 scale (the activation half of an fp8 linear).
__global__ void act_quant_fp8_kernel(uint8_t* __restrict__ a, float* __restrict__ as,
                                     const float* __restrict__ x, int rows, int K, int block) {
    int nb = K / block; int gid = blockIdx.x; if (gid >= rows * nb) return;
    int row = gid / nb, b = gid % nb;
    const float* xr = x + (size_t)row * K + (size_t)b * block;
    uint8_t* ar = a + (size_t)row * K + (size_t)b * block;
    extern __shared__ float red[];
    float v = (threadIdx.x < block) ? fabsf(xr[threadIdx.x]) : 0.f;
    red[threadIdx.x] = v; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) { if (threadIdx.x < s) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]); __syncthreads(); }
    float amax = fmaxf(red[0], 1e-4f);
    float scale = exp2f(ceilf(log2f(amax * (1.f / 448.f))));
    if (threadIdx.x == 0) as[(size_t)row * nb + b] = scale;
    if (threadIdx.x < block) {
        float q = fminf(fmaxf(xr[threadIdx.x] / scale, -448.f), 448.f);
        ar[threadIdx.x] = __nv_fp8_e4m3(q).__x;
    }
}
void act_quant_fp8(uint8_t* a_fp8, float* a_s, const float* x, int rows, int K, int block, cudaStream_t stream) {
    int threads = block < 32 ? 32 : block;
    act_quant_fp8_kernel<<<rows * (K / block), threads, threads * sizeof(float), stream>>>(a_fp8, a_s, x, rows, K, block);
}

// ---------------- ogroup_gemm ----------------
// out[bs,G,R] = sum_d o[bs,G,d] * wo_a[G,R,d]. One warp per (bs,G,R) reducing over d.
__global__ void ogroup_gemm_kernel(float* __restrict__ out, const float* __restrict__ o,
                                   const float* __restrict__ wo_a, int bs, int G, int R, int Kd) {
    int gid = blockIdx.x; int r = gid % R; int bg = gid / R; int gg = bg % G; int bb = bg / G;
    int lane = threadIdx.x & 31;
    const float* op = o + (((size_t)bb * G + gg) * Kd);
    const float* wp = wo_a + (((size_t)gg * R + r) * Kd);
    float acc = 0.f;
    for (int d = lane; d < Kd; d += 32) acc += op[d] * wp[d];
    #pragma unroll
    for (int s = 16; s > 0; s >>= 1) acc += __shfl_down_sync(0xffffffff, acc, s);
    if (lane == 0) out[((size_t)bb * G + gg) * R + r] = acc;
}
// --- TC version: fp16 mma per group (matches the bf16-einsum reference; ~13% warp-per-output -> tensor core) ---
// o,wo_a fp32 -> fp16 (transient, memory-neutral), m16n8k16 fp16 GEMM per group. Dispatched by g_tc_ogroup.
__device__ __forceinline__ void ogm_mma(float* c, const unsigned* a, const unsigned* b){
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
      :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1])); }
__global__ void k_f2h(__half* o, const float* in, size_t n){ size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x; if(i<n) o[i]=__float2half(in[i]); }
__global__ void tc_ogroup_kernel(float* out, const __half* o16, const __half* wo16, int bs, int G, int R, int Kd){
    int lane=threadIdx.x&31, gid=lane>>2, t4=lane&3;
    int gg=blockIdx.y, n0=(blockIdx.x*(blockDim.x>>5) + (threadIdx.x>>5))*8; if(n0>=R) return;
    const int mb=blockIdx.z*16, r0=mb+gid, r8=mb+gid+8;       // row tiles — same defect as the fp8 twin
    const __half* xg0 = o16 + ((size_t)r0*G+gg)*Kd;           // A row bb=r0 (stride G*Kd), group gg
    const __half* xg8 = o16 + ((size_t)r8*G+gg)*Kd;
    const __half* Bg  = wo16 + (size_t)gg*R*Kd;               // B = wo16[gg] [R,Kd]
    bool m0=r0<bs, m8=r8<bs; float c[4]={0,0,0,0};
    for(int k0=0;k0<Kd;k0+=16){
        unsigned a[4],b[2];
        a[0]=m0?*(const unsigned*)(xg0+k0+2*t4):0u; a[1]=m8?*(const unsigned*)(xg8+k0+2*t4):0u;
        a[2]=m0?*(const unsigned*)(xg0+k0+2*t4+8):0u; a[3]=m8?*(const unsigned*)(xg8+k0+2*t4+8):0u;
        const __half* wr=Bg+(size_t)(n0+gid)*Kd;
        b[0]=(n0+gid<R)?*(const unsigned*)(wr+k0+2*t4):0u; b[1]=(n0+gid<R)?*(const unsigned*)(wr+k0+2*t4+8):0u;
        ogm_mma(c,a,b);
    }
    int cn=2*t4;
    if(r0<bs && n0+cn  <R) out[((size_t)r0*G+gg)*R + n0+cn  ]=c[0];
    if(r0<bs && n0+cn+1<R) out[((size_t)r0*G+gg)*R + n0+cn+1]=c[1];
    if(r8<bs && n0+cn  <R) out[((size_t)r8*G+gg)*R + n0+cn  ]=c[2];
    if(r8<bs && n0+cn+1<R) out[((size_t)r8*G+gg)*R + n0+cn+1]=c[3];
}
#include <cuda_fp8.h>
__device__ __forceinline__ float ogm_e4m3(uint8_t b){
    __half_raw r=__nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)b,__NV_E4M3); return __half2float(*reinterpret_cast<__half*>(&r)); }
// FUSED fp8 wo_a TC ogroup: decode fp8 wo_a * e8m0 block-scale -> f16 IN the mma inner loop (no wo16 buffer,
// no per-token full-tensor conversion). Reads fp8 (half the bytes of f16). Bit-identical to convert-then-mma.
// ROW TILES (LOOP_LOG Finding 62). This is a SINGLE m16 mma tile: gid = lane>>2 covers rows 0..7 and
// gid+8 covers 8..15, and there was no loop over row tiles — so for bs > 16 every row from 16 up was
// never computed and never stored, leaving the caller's `og` buffer UNINITIALISED there. The masks
// (m0/m8) hid it by making the reads safe, so nothing faulted and nothing gated: the only symptom was
// that prefill output for positions >=16 in the two pure-sliding layers was whatever the allocator
// last left at that address. That is why the engine's prefill differed between byte-identical sweep
// points on an 18-token prompt (PSp=17) and not on an 11-token one, and why the difference tracked
// `seqmax` — it is a read of another allocation's contents, not of anything the engine wrote.
// blockIdx.z now walks the rows in steps of 16. compute-sanitizer --tool initcheck named the read;
// tests/gate_ogroup_gemv covers M=17,24,33 and fails without this (cosine 0.94/0.67/0.70).
__global__ void tc_ogroup_fp8_kernel(float* out, const __half* o16, const uint8_t* wo, const uint8_t* wsc,
                                     int bs, int G, int R, int Kd){
    int lane=threadIdx.x&31, gid=lane>>2, t4=lane&3;
    int gg=blockIdx.y, n0=blockIdx.x*8; if(n0>=R) return;
    const int mb=blockIdx.z*16, r0=mb+gid, r8=mb+gid+8;
    const __half* xg0 = o16 + ((size_t)r0*G+gg)*Kd;
    const __half* xg8 = o16 + ((size_t)r8*G+gg)*Kd;
    const uint8_t* Bg = wo + (size_t)gg*R*Kd; int scw=Kd/128;
    bool m0=r0<bs, m8=r8<bs; float c[4]={0,0,0,0};
    int n=n0+gid; const uint8_t* wr = Bg + (size_t)n*Kd; size_t grow=(size_t)gg*R+n;
    for(int k0=0;k0<Kd;k0+=16){
        unsigned a[4],b[2];
        a[0]=m0?*(const unsigned*)(xg0+k0+2*t4):0u; a[1]=m8?*(const unsigned*)(xg8+k0+2*t4):0u;
        a[2]=m0?*(const unsigned*)(xg0+k0+2*t4+8):0u; a[3]=m8?*(const unsigned*)(xg8+k0+2*t4+8):0u;
        if(n<R){ int kk=k0+2*t4;
            float s0=exp2f((float)wsc[(grow/128)*scw + kk/128]-127.f);
            float s1=exp2f((float)wsc[(grow/128)*scw + (kk+8)/128]-127.f);
            __half2 p0=__halves2half2(__float2half(ogm_e4m3(wr[kk])*s0),   __float2half(ogm_e4m3(wr[kk+1])*s0));
            __half2 p1=__halves2half2(__float2half(ogm_e4m3(wr[kk+8])*s1), __float2half(ogm_e4m3(wr[kk+9])*s1));
            b[0]=*(unsigned*)&p0; b[1]=*(unsigned*)&p1;
        } else { b[0]=0u; b[1]=0u; }
        ogm_mma(c,a,b);
    }
    int cn=2*t4;
    if(r0<bs && n0+cn  <R) out[((size_t)r0*G+gg)*R + n0+cn  ]=c[0];
    if(r0<bs && n0+cn+1<R) out[((size_t)r0*G+gg)*R + n0+cn+1]=c[1];
    if(r8<bs && n0+cn  <R) out[((size_t)r8*G+gg)*R + n0+cn  ]=c[2];
    if(r8<bs && n0+cn+1<R) out[((size_t)r8*G+gg)*R + n0+cn+1]=c[3];
}
// ===================== B9: m-tile-amortised tc_ogroup (prefill) =====================
// tc_ogroup_fp8_kernel above is the bs>16 path, i.e. the one PREFILL takes, and it was written for
// the verify's small bs. At bs=1022 it launches (R/8, G, bs/16) = 128 x 8 x 64 = 65,536 blocks of
// ONE WARP, and — the expensive part — every one of those 64 m-tiles re-loads and re-DEQUANTISES
// the same weight row: two exp2f, four scalar byte reads and four fp8->half converts per k-step,
// repeated 64 times for bytes that never change.
//
// This is the F64 row-amortisation transformation applied to the ogroup TC path: hoist the weight
// load and dequant out of an m-tile loop so MT tiles share one dequant. Weight work drops MT-fold;
// the activation loads stay per-tile because they genuinely differ.
//
// BIT-EXACT: each output (r,n) still accumulates over k0 in the same order through the same mma
// sequence. Only the ORDER IN WHICH DIFFERENT OUTPUTS are computed changes, and they are
// independent. `gate_ogroup_gemv` memcmps this family.
template<int MT>
__global__ void tc_ogroup_fp8_mt_kernel(float* out, const __half* o16, const uint8_t* wo,
                                        const uint8_t* wsc, int bs, int G, int R, int Kd){
    int lane=threadIdx.x&31, gid=lane>>2, t4=lane&3;
    // WARPS PER BLOCK (Finding 21's class, the same fix moe_wpb() already applies to the MoE GEMMs).
    // This launched <<<grid,32>>>: one warp per block, so a 32-thread block burns a whole block slot
    // and half the SM's warp slots are unusable. The warps here are fully independent -- no shared
    // memory, no __syncthreads, and n is the only thing that differs -- so several can share a block,
    // each taking its own n-block. Pure launch geometry: the per-warp math is byte-identical.
    int gg=blockIdx.y, n0=(blockIdx.x*(blockDim.x>>5) + (threadIdx.x>>5))*8; if(n0>=R) return;
    const uint8_t* Bg = wo + (size_t)gg*R*Kd; int scw=Kd/128;
    int n=n0+gid; const uint8_t* wr = Bg + (size_t)n*Kd; size_t grow=(size_t)gg*R+n;
    const int mb0 = blockIdx.z*(16*MT);
    float c[MT][4];
    #pragma unroll
    for(int t=0;t<MT;++t){ c[t][0]=c[t][1]=c[t][2]=c[t][3]=0.f; }
    for(int k0=0;k0<Kd;k0+=16){
        // ---- weight: loaded and dequantised ONCE for all MT m-tiles ----
        unsigned b[2];
        if(n<R){ int kk=k0+2*t4;
            float s0=exp2f((float)wsc[(grow/128)*scw + kk/128]-127.f);
            float s1=exp2f((float)wsc[(grow/128)*scw + (kk+8)/128]-127.f);
            __half2 p0=__halves2half2(__float2half(ogm_e4m3(wr[kk])*s0),   __float2half(ogm_e4m3(wr[kk+1])*s0));
            __half2 p1=__halves2half2(__float2half(ogm_e4m3(wr[kk+8])*s1), __float2half(ogm_e4m3(wr[kk+9])*s1));
            b[0]=*(unsigned*)&p0; b[1]=*(unsigned*)&p1;
        } else { b[0]=0u; b[1]=0u; }
        #pragma unroll
        for(int t=0;t<MT;++t){
            const int mb=mb0+t*16, r0=mb+gid, r8=mb+gid+8;
            if(mb>=bs) continue;
            const __half* xg0 = o16 + ((size_t)r0*G+gg)*Kd;
            const __half* xg8 = o16 + ((size_t)r8*G+gg)*Kd;
            bool m0=r0<bs, m8=r8<bs;
            unsigned a[4];
            a[0]=m0?*(const unsigned*)(xg0+k0+2*t4):0u; a[1]=m8?*(const unsigned*)(xg8+k0+2*t4):0u;
            a[2]=m0?*(const unsigned*)(xg0+k0+2*t4+8):0u; a[3]=m8?*(const unsigned*)(xg8+k0+2*t4+8):0u;
            ogm_mma(c[t],a,b);
        }
    }
    int cn=2*t4;
    #pragma unroll
    for(int t=0;t<MT;++t){
        const int mb=mb0+t*16, r0=mb+gid, r8=mb+gid+8;
        if(mb>=bs) continue;
        if(r0<bs && n0+cn  <R) out[((size_t)r0*G+gg)*R + n0+cn  ]=c[t][0];
        if(r0<bs && n0+cn+1<R) out[((size_t)r0*G+gg)*R + n0+cn+1]=c[t][1];
        if(r8<bs && n0+cn  <R) out[((size_t)r8*G+gg)*R + n0+cn  ]=c[t][2];
        if(r8<bs && n0+cn+1<R) out[((size_t)r8*G+gg)*R + n0+cn+1]=c[t][3];
    }
}
bool g_tc_ogroup = false;   // forward.cu sets true; gates use the warp-per-output oracle
// M=1 ogroup GEMV: one warp per output (g,r), out[g*R+r] = sum_d o[g][d]*fp8(wo[gr][d])*e8m0scale. Coalesced
// strided reads (lanes read consecutive bytes), NO mma waste (bs=1). Kills the tc_ogroup's scalar-byte m16 mma
// (~32 ms/tok -> bandwidth). Gated cosine vs ogroup_gemm oracle (tests/gate_ogroup_gemv.cu).
//
// LOOP_LOG Finding 35 (Opt #11). The inner loop above read ONE BYTE per lane:
//     for(dd=kb*128+lane; dd<(kb+1)*128; dd+=32) acc += og[dd]*ogm_e4m3(wr[dd])*ws;
// 32 lanes x 1 byte = a 32-byte request, so streaming `wo_a` costs 4x the requests a 128-byte
// access would. This is the SAME defect Finding 15 traced under the M>=2 verify penalty — there
// the m16 tile issued 8x32B where the GEMV issued 1x128B — and it was sitting in the M=1 path all
// along. `wo_a` is [G*R, Kd] fp8 = ~33 MB/layer, ~1.44 GB/token: by the dprof sub-marks this
// kernel is ~22 ms of a ~98 ms step, i.e. ~65 GB/s against 240 achievable.
//
// One uint32_t per lane makes the warp cover 128 contiguous bytes in a single request, and gives
// 4 independent accumulator chains for free (Little's Law: ILP=1 leaves MLP on the floor).
// `ws` is a power of two, so folding it into the weight instead of the product is exact.
__global__ void ogroup_gemv_fp8_kernel(float* __restrict__ out, const float* __restrict__ o,
                                       const uint8_t* __restrict__ wo, const uint8_t* __restrict__ wsc, int G, int R, int Kd, int vec4){
    int warp=(blockIdx.x*blockDim.x+threadIdx.x)>>5; int total=G*R; if(warp>=total) return;
    int gr=warp, g=gr/R; int lane=threadIdx.x&31, scw=Kd/128;
    const uint8_t* wr=wo+(size_t)gr*Kd; const uint8_t* sr=wsc+(size_t)(gr/128)*scw; const float* og=o+(size_t)g*Kd;
    float acc=0.f;
    if(vec4){
        float a0=0.f,a1=0.f,a2=0.f,a3=0.f;
        for(int kb=0; kb<Kd/128; ++kb){
            const float ws=exp2f((float)sr[kb]-127.f);
            const int base=kb*128+lane*4;
            const uint32_t w4=*(const uint32_t*)(wr+base);
            const float4  o4=*(const float4*)(og+base);
            a0=fmaf(o4.x, ogm_e4m3((uint8_t)(w4    ))*ws, a0);
            a1=fmaf(o4.y, ogm_e4m3((uint8_t)(w4>> 8))*ws, a1);
            a2=fmaf(o4.z, ogm_e4m3((uint8_t)(w4>>16))*ws, a2);
            a3=fmaf(o4.w, ogm_e4m3((uint8_t)(w4>>24))*ws, a3);
        }
        acc=(a0+a1)+(a2+a3);
    } else {
        for(int kb=0; kb<Kd/128; ++kb){ float ws=exp2f((float)sr[kb]-127.f);
            for(int dd=kb*128+lane; dd<(kb+1)*128; dd+=32) acc += og[dd]*ogm_e4m3(wr[dd])*ws; }
    }
    #pragma unroll
    for(int o2=16;o2>0;o2>>=1) acc+=__shfl_down_sync(0xffffffff,acc,o2);
    if(lane==0) out[gr]=acc;
}
// M=K ogroup GEMV (LOOP_LOG Finding 40). At the M=5 spec-decode verify, `cattn:ogroup` measured
// 64.34 ms against 16.63 at M=1 — a 3.87x scaling on weights that are IDENTICAL for all M tokens
// and should therefore cost ~1.0x. The m16 mma tile re-reads `wo_a` per 16-row tile and issues
// 8x32B requests per fragment (Finding 15's mechanism), so at M=5 it both wastes 11 of 16 rows and
// quadruples the request count. Reading each weight row ONCE and dotting it against all M
// activation rows removes both at a stroke — the same shape as `fp8_gemv_mkT_kernel` (Finding 28),
// templated on M so `acc[M]` is a real register array rather than 16 spilled ones.
// MULTI-ROW (Finding 43). Finding 40's kernel reads the weight once per M rows, which fixed the
// weight traffic — and left `cattn:ogroup` at 1.73x from K=1 to K=5 on weights that do not change.
// The residual is the OTHER operand. Per warp per 128-byte K-block this reads 32x4 = 128 weight
// bytes and M x 32 x 16 = M*512 bytes of `o`, so the f32 activation traffic is 4M times the fp8
// weight traffic — 20x at M=5. `o` is 655 KB, i.e. L2-resident, so none of it is DRAM; it is L2
// bandwidth, and at 27 GB per verify across 41 layers that is the binding constraint.
//
// `o` depends only on the group g, and gr = g*R + r, so NR consecutive output rows share it
// exactly. Give each warp NR rows: the `o` float4 is loaded once and used against NR weights, and
// the ratio falls from 4M to 4M/NR. Accumulators are NR*M registers, which is why NR is templated
// alongside M rather than fixed — 4x5 = 20 is comfortable, 8x16 would spill.
#ifndef OGMK_BLOCKS_PER_SM
#define OGMK_BLOCKS_PER_SM 4      // A/B: -DOGMK_BLOCKS_PER_SM=n caps registers at 65536/(256*n)
#endif
// WS1 (this cycle). ncu on the shipped `<5,4>` instantiation said **806,912 local-memory spilling
// requests, 100% spill overhead, 64 registers** — the `__launch_bounds__(256,4)` cap is
// 65536/(256*256/32... ) = 64 regs/thread and the live set does not fit: acc[4][5]=20, o4[5]=20,
// NR weight pointers, M activation pointers, w[4] and ws[4]. The kernel is not bandwidth-bound
// (Memory Throughput 41%, Compute 49%); 62.3% of its 13.0 cycles between issues are L1TEX
// scoreboard stalls, and a spilling kernel puts its own local traffic in that queue.
//
// `ws[r]` is the cheapest 4 of those registers to give back, and giving them back is FREE:
// ws[r] is indexed (rr/128, kb) with rr = gr0+r, gr0 a multiple of NR, and NR | 128 — so every
// row a warp owns lands in the SAME 128-row scale block and **ws[0..NR-1] are the same value,
// always**. The compiler cannot prove it because `rr` goes through the tail ternary, so it emitted
// NR scale loads and NR `exp2f` per k-block instead of one. WS1=true computes it once. This is
// value equality, not a reassociation — every fma sees the identical float it saw before, which is
// why it can be gated with memcmp rather than a cosine (Finding 68).
// The transform is CORRECT and it is FREE and it is worth nothing: see the dispatch note below for
// the in-situ numbers that retired it. OG_WS1=1 selects it; the default is WS1=false.
template<int M, int NR, bool WS1>
__global__ __launch_bounds__(256, OGMK_BLOCKS_PER_SM) void ogroup_gemv_mk_kernel(float* __restrict__ out, const float* __restrict__ o,
                                      const uint8_t* __restrict__ wo, const uint8_t* __restrict__ wsc,
                                      int G, int R, int Kd){
    int warp=(blockIdx.x*blockDim.x+threadIdx.x)>>5; int total=G*R;
    int gr0=warp*NR; if(gr0>=total) return;
    int g=gr0/R; int lane=threadIdx.x&31, scw=Kd/128;
    // NR rows share a group only if the run does not straddle a g boundary; R%NR==0 guarantees it.
    const float* og=o+(size_t)g*Kd;
    // gr0 < total is established above, so this row of the scale plane is always in range.
    // A 32-bit OFFSET, not a hoisted `const uint8_t*`. The pointer form measured WORSE at the
    // shipped <5,4>: ptxas -v went 44 -> 60 bytes of spill and the bench lost 4.4%, because a
    // 64-bit pointer live across the whole kb loop costs two permanently-allocated registers and
    // this kernel is already 8 registers over its 64-register cap. One int costs one.
    const int scoff = (gr0/128)*scw;
    float acc[NR][M];
    #pragma unroll
    for(int r=0;r<NR;++r){
        #pragma unroll
        for(int m=0;m<M;++m) acc[r][m]=0.f; }
    // REGISTER PRESSURE (ncu, Finding 47). The first version materialised `d[NR][4]` — the
    // dequantised weights for all NR rows — before touching the activations. At NR=8, M=5 that is
    // 40 accumulators + 32 weight floats and the kernel compiled to **128 registers**, so Block
    // Limit Registers was 2 and theoretical occupancy 33.3% (16 of 48 warps). ncu: 66% of stall
    // cycles on `long_scoreboard`, i.e. waiting on global loads, with Memory Throughput at 14% —
    // latency-bound with too few warps to hide it, NOT bandwidth-bound.
    //
    // Keep the loads in flight but not the dequantised values: issue the NR raw uint32 weight loads
    // and the M activation float4s up front (that is the memory-level parallelism the phase needs),
    // then dequantise one row at a time into 4 registers reused across r. Same arithmetic, same
    // order, so this stays bit-identical; it just stops holding 32 floats live across the m loop.
    for(int kb=0; kb<Kd/128; ++kb){
        const int base=kb*128+lane*4;
        uint32_t w[NR]; float ws[NR];
        #pragma unroll
        for(int r=0;r<NR;++r){
            const int rr = (gr0+r<total) ? (gr0+r) : gr0;   // tail warps: duplicate, store is masked
            // NEGATIVE RESULT, retired with a measurement (this cycle). ncu said this kernel is
            // latency-bound (long_scoreboard 7.33/issue) with lts__t_sector_hit_rate 65.6%, where
            // activation-only reuse predicts ~84% — i.e. the 33.6 MB weight stream looked like it
            // was evicting the 0.7 MB `og` from L2. Marking the weight load evict-first with
            // __ldcs, the policy the roofline-achieving MXFP4 expert GEMV uses, DID cut the stall
            // ratio to 3.46 — and made the kernel SLOWER: 242.8 -> 280.4 us on one ncu launch and
            // 0.2001 -> 0.2565 ms in gemm_bench at M=5/NR=4. __ldg on `og` on top of it was worse
            // still. The sector hit rate fell (65.6 -> 61.8) rather than rose, so the weight line
            // was not the evictor; dropping it evict-first just costs the 4 sectors of the 128-byte
            // line their reuse across the warp. Do not re-queue a cache-hint change here without a
            // hit-rate measurement that says L2 is actually the binding resource.
            w[r]  = *(const uint32_t*)(wo+(size_t)rr*Kd+base);
            if(!WS1) ws[r] = exp2f((float)wsc[(size_t)(rr/128)*scw + kb]-127.f);  // power of two: exact to fold
        }
        if(WS1){ const float s = exp2f((float)wsc[scoff + kb]-127.f);
            #pragma unroll
            for(int r=0;r<NR;++r) ws[r]=s; }                     // same value NR ways, one load + one ex2
        float4 o4[M];
        #pragma unroll
        for(int m=0;m<M;++m) o4[m]=*(const float4*)(og+(size_t)m*G*Kd+base);
        #pragma unroll
        for(int r=0;r<NR;++r){
            const float d0=ogm_e4m3((uint8_t)(w[r]    ))*ws[r], d1=ogm_e4m3((uint8_t)(w[r]>> 8))*ws[r],
                        d2=ogm_e4m3((uint8_t)(w[r]>>16))*ws[r], d3=ogm_e4m3((uint8_t)(w[r]>>24))*ws[r];
            #pragma unroll
            for(int m=0;m<M;++m){
                acc[r][m]=fmaf(o4[m].x,d0,acc[r][m]); acc[r][m]=fmaf(o4[m].y,d1,acc[r][m]);
                acc[r][m]=fmaf(o4[m].z,d2,acc[r][m]); acc[r][m]=fmaf(o4[m].w,d3,acc[r][m]);
            }
        }
    }
    #pragma unroll
    for(int r=0;r<NR;++r){
        #pragma unroll
        for(int m=0;m<M;++m){ float a=acc[r][m];
            #pragma unroll
            for(int o2=16;o2>0;o2>>=1) a+=__shfl_down_sync(0xffffffff,a,o2);
            if(lane==0 && gr0+r<total) out[(size_t)m*total+gr0+r]=a; }
    }
}
// SMEM-STAGED ACTIVATION VARIANT of the kernel above. NO_OGSMEM=1 selects the plain one for A/B.
//
// The observation the NR trick did not go far enough on. In the kernel above, `base` is
// kb*128 + lane*4 and `og` is o + g*Kd — neither depends on which WARP is running. So all 8 warps
// of a block issue the SAME M float4 loads of `o`, every k-block: the activation traffic is 8x
// redundant WITHIN the block, on top of the 4M/NR ratio the NR trick already addressed. Measured
// per launch at M=5/NR=4: 33.6 MB of weight and 168 MB of `o`, against an `o` that is only 0.7 MB.
// ncu agrees it is not bandwidth — long_scoreboard 7.33 stalls per issued instruction at 59% warp
// occupancy — but 168 MB of redundant requests is what those warps are queued behind.
//
// Staging removes the redundancy instead of re-prioritising it (which is what __ldcs tried, and it
// lost: see the note above). The block cooperatively loads KC k-blocks of every activation row into
// shared memory once, then all 8 warps read their float4 from smem. Activation global traffic falls
// 8x, to ~21 MB. Arithmetic, operand order and accumulation order are untouched, so this is
// bit-identical to the kernel above and gates against it directly.
//
// Two structural requirements, both checked by the dispatch rather than assumed (Finding 41):
//   (WPB*NR) | R    — so every warp in the block is in the same o-group `g` and shares the staged
//                     rows. R = O_LORA = 1024 and WPB*NR is 8..64, so this holds for every NR.
//   KC | (Kd/128)   — so no chunk runs off the end of the row. Kd/128 = 32.
// A 128-bit smem load is serviced in 4 phases of 8 lanes = 32 banks exactly, so the float4 reads
// below are conflict-free.
//
// LAZY (Finding 79, this cycle). LEVERS.md B8' — the one remaining idea on `o:wo_a` and the only one
// that moves ~16 registers rather than 3. `float4 o4[M]` below is M float4s = 4M registers held live
// across the whole r loop (20 at the shipped M=5), in a kernel whose `<5,4>` instantiation is
// ALREADY 8 registers over its 64-register `__launch_bounds__` cap. In the NON-staged kernel that
// array is load-bearing: those are global loads and hoisting them out of the r loop is the
// memory-level parallelism the phase needs. Here they are SMEM reads — ~30 cycles, no MLP argument —
// so they can be re-read per (r,m) instead of held. Same address, same value, same fma order, so
// this is bit-identical and gates with memcmp (`gate_ogroup_gemv`, which sweeps it).
// LAZY=false is the F55 kernel unchanged; the ptxas numbers for BOTH are in the dispatch note.
template<int M, int NR, bool LAZY>
__global__ __launch_bounds__(256, OGMK_BLOCKS_PER_SM) void ogroup_gemv_mk_smem_kernel(
        float* __restrict__ out, const float* __restrict__ o,
        const uint8_t* __restrict__ wo, const uint8_t* __restrict__ wsc,
        int G, int R, int Kd){
    constexpr int WPB = 256/32;                                   // warps per block
    constexpr int KC  = (M<=4) ? 16 : ((M<=8) ? 8 : 4);           // k-blocks staged per chunk
    __shared__ __align__(16) float sh[M*KC*128];                  // <=16 KB, so 4 blocks/SM still fit
    const int warpInBlk = threadIdx.x>>5, lane = threadIdx.x&31;
    const int total = G*R, scw = Kd/128;
    const int gr0 = (blockIdx.x*WPB + warpInBlk)*NR;
    // One `g` for the whole block. No early return anywhere in this kernel: every thread must reach
    // every __syncthreads, so out-of-range warps participate in the staging and are masked only at
    // the store.
    int gblk = (blockIdx.x*WPB*NR)/R; if(gblk > G-1) gblk = G-1;
    const float* og = o + (size_t)gblk*Kd;
    float acc[NR][M];
    #pragma unroll
    for(int r=0;r<NR;++r){
        #pragma unroll
        for(int m=0;m<M;++m) acc[r][m]=0.f; }
    for(int kb0=0; kb0<scw; kb0+=KC){
        for(int t=threadIdx.x; t<M*KC*32; t+=256){
            const int m=t/(KC*32), q=t%(KC*32);
            *(float4*)&sh[m*KC*128 + q*4] =
                *(const float4*)(og + (size_t)m*G*Kd + (size_t)kb0*128 + q*4);
        }
        __syncthreads();
        #pragma unroll
        for(int kbl=0; kbl<KC; ++kbl){
            const int kb=kb0+kbl, base=kb*128+lane*4;
            uint32_t w[NR]; float ws[NR];
            #pragma unroll
            for(int r=0;r<NR;++r){
                const int rr = (gr0+r<total) ? (gr0+r) : (total-1);   // tail warps: clamped, store masked
                w[r]  = *(const uint32_t*)(wo+(size_t)rr*Kd+base);
                ws[r] = exp2f((float)wsc[(size_t)(rr/128)*scw + kb]-127.f);
            }
            const float* shk = &sh[kbl*128 + lane*4];
            float4 o4[LAZY ? 1 : M];
            if constexpr(!LAZY){
                #pragma unroll
                for(int m=0;m<M;++m) o4[m]=*(const float4*)&shk[m*KC*128];
            }
            #pragma unroll
            for(int r=0;r<NR;++r){
                const float d0=ogm_e4m3((uint8_t)(w[r]    ))*ws[r], d1=ogm_e4m3((uint8_t)(w[r]>> 8))*ws[r],
                            d2=ogm_e4m3((uint8_t)(w[r]>>16))*ws[r], d3=ogm_e4m3((uint8_t)(w[r]>>24))*ws[r];
                #pragma unroll
                for(int m=0;m<M;++m){
                    // Identical bytes either way: LAZY re-reads the same 16 smem bytes per r instead
                    // of keeping them in 4M registers. Same operand values, same fma order.
                    float4 om;
                    if constexpr(LAZY) om = *(const float4*)&shk[m*KC*128];
                    else               om = o4[m];
                    acc[r][m]=fmaf(om.x,d0,acc[r][m]); acc[r][m]=fmaf(om.y,d1,acc[r][m]);
                    acc[r][m]=fmaf(om.z,d2,acc[r][m]); acc[r][m]=fmaf(om.w,d3,acc[r][m]);
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for(int r=0;r<NR;++r){
        #pragma unroll
        for(int m=0;m<M;++m){ float a=acc[r][m];
            #pragma unroll
            for(int o2=16;o2>0;o2>>=1) a+=__shfl_down_sync(0xffffffff,a,o2);
            if(lane==0 && gr0+r<total) out[(size_t)m*total+gr0+r]=a; }
    }
}
// NR falls as M rises to keep NR*M accumulators in registers. NO_OGNR=1 pins NR=1, which is
// exactly Finding 40's kernel, for A/B.
#define OG_MK_LAUNCH(MM,NRV) do{ \
    if(ogsmem&&oglazy) ogroup_gemv_mk_smem_kernel<MM,NRV,true ><<<(nb+NRV-1)/NRV,threads,0,stream>>>(out,o,wo_fp8,wo_sc,G,R,Kd); \
    else if(ogsmem) ogroup_gemv_mk_smem_kernel<MM,NRV,false><<<(nb+NRV-1)/NRV,threads,0,stream>>>(out,o,wo_fp8,wo_sc,G,R,Kd); \
    else if(ogws1)  ogroup_gemv_mk_kernel     <MM,NRV,true ><<<(nb+NRV-1)/NRV,threads,0,stream>>>(out,o,wo_fp8,wo_sc,G,R,Kd); \
    else            ogroup_gemv_mk_kernel     <MM,NRV,false><<<(nb+NRV-1)/NRV,threads,0,stream>>>(out,o,wo_fp8,wo_sc,G,R,Kd); \
  }while(0)
#define OG_MK_CASE(M) case M: \
    if(ognr==8)      OG_MK_LAUNCH(M,8); \
    else if(ognr==4) OG_MK_LAUNCH(M,4); \
    else if(ognr==2) OG_MK_LAUNCH(M,2); \
    else             OG_MK_LAUNCH(M,1); break;
// fp8-native TC ogroup: o(f32)->f16 + fused fp8 wo_a decode in the mma. No per-token wo16 conversion.
// ============ ogroup MT x NB: the activation loaded once, used NB times =========================
// The MT kernel above amortises the WEIGHT dequant across m-tiles, which was the right lever for the
// weights -- and the weights were never the cost. At bs=845, G=8, R=1024, Kd=4096, per layer:
//
//     weights, ideal                     33.6 MB      activations, ideal             55.4 MB
//     weights, MT=8 (what ships)        234.9 MB      activations, NB=1            7088.4 MB
//
// A warp owns 8 of R's 1024 columns, so the activation block is re-read 128 times and is 97 % of the
// traffic. The A fragment depends on (row, k) and NOT on n -- the same fact the MoE's MOE_NBLK
// exploits -- so NB n-blocks per warp issue NB mma against one A load. Predicted from traffic alone:
// MT=8/NB=1 1231 ms against 1543 measured (the model explains it), MT=8/NB=2 635, MT=4/NB=4 377.
//
// MT and NB trade against the same register budget (MT*NB*4 accumulators), and since activations
// dominate, spending it on NB beats spending it on MT.
//
// BIT-EXACT: each output (r,n) still accumulates over k0 in the same order through the same mma.
template<int MT, int NB>
__global__ void tc_ogroup_fp8_mtnb_kernel(float* out, const __half* o16, const uint8_t* wo,
                                         const uint8_t* wsc, int bs, int G, int R, int Kd){
    int lane=threadIdx.x&31, gid=lane>>2, t4=lane&3;
    int gg=blockIdx.y, nb0=blockIdx.x*NB;
    const uint8_t* Bg = wo + (size_t)gg*R*Kd; int scw=Kd/128;
    const int mb0 = blockIdx.z*(16*MT);
    int nq[NB]; const uint8_t* wrq[NB]; size_t growq[NB]; bool nok[NB];
    #pragma unroll
    for(int q=0;q<NB;++q){ int n0=(nb0+q)*8; nq[q]=n0+gid; nok[q]=(n0<R) && (nq[q]<R);
        int nn = nok[q]?nq[q]:0;
        wrq[q]=Bg+(size_t)nn*Kd; growq[q]=(size_t)gg*R+nn; }
    float c[MT][NB][4];
    #pragma unroll
    for(int t=0;t<MT;++t) {
        #pragma unroll
        for(int q=0;q<NB;++q){ c[t][q][0]=c[t][q][1]=c[t][q][2]=c[t][q][3]=0.f; } }
    for(int k0=0;k0<Kd;k0+=16){
        unsigned b[NB][2];
        #pragma unroll
        for(int q=0;q<NB;++q){
            if(nok[q]){ int kk=k0+2*t4;
                float s0=exp2f((float)wsc[(growq[q]/128)*scw + kk/128]-127.f);
                float s1=exp2f((float)wsc[(growq[q]/128)*scw + (kk+8)/128]-127.f);
                __half2 p0=__halves2half2(__float2half(ogm_e4m3(wrq[q][kk])*s0),   __float2half(ogm_e4m3(wrq[q][kk+1])*s0));
                __half2 p1=__halves2half2(__float2half(ogm_e4m3(wrq[q][kk+8])*s1), __float2half(ogm_e4m3(wrq[q][kk+9])*s1));
                b[q][0]=*(unsigned*)&p0; b[q][1]=*(unsigned*)&p1;
            } else { b[q][0]=0u; b[q][1]=0u; } }
        #pragma unroll
        for(int t=0;t<MT;++t){
            const int mb=mb0+t*16, r0=mb+gid, r8=mb+gid+8;
            if(mb>=bs) continue;
            const __half* xg0 = o16 + ((size_t)r0*G+gg)*Kd;
            const __half* xg8 = o16 + ((size_t)r8*G+gg)*Kd;
            bool m0=r0<bs, m8=r8<bs;
            unsigned a[4];
            a[0]=m0?*(const unsigned*)(xg0+k0+2*t4):0u; a[1]=m8?*(const unsigned*)(xg8+k0+2*t4):0u;
            a[2]=m0?*(const unsigned*)(xg0+k0+2*t4+8):0u; a[3]=m8?*(const unsigned*)(xg8+k0+2*t4+8):0u;
            #pragma unroll
            for(int q=0;q<NB;++q) ogm_mma(c[t][q],a,b[q]);
        }
    }
    int cn=2*t4;
    #pragma unroll
    for(int t=0;t<MT;++t){
        const int mb=mb0+t*16, r0=mb+gid, r8=mb+gid+8;
        if(mb>=bs) continue;
        #pragma unroll
        for(int q=0;q<NB;++q){
            int n0=(nb0+q)*8; if(n0>=R) continue;
            if(r0<bs && n0+cn  <R) out[((size_t)r0*G+gg)*R + n0+cn  ]=c[t][q][0];
            if(r0<bs && n0+cn+1<R) out[((size_t)r0*G+gg)*R + n0+cn+1]=c[t][q][1];
            if(r8<bs && n0+cn  <R) out[((size_t)r8*G+gg)*R + n0+cn  ]=c[t][q][2];
            if(r8<bs && n0+cn+1<R) out[((size_t)r8*G+gg)*R + n0+cn+1]=c[t][q][3];
        }
    }
}
void ogroup_gemm_fp8(float* out, const float* o, const uint8_t* wo_fp8, const uint8_t* wo_sc,
                     int bs, int G, int R, int Kd, cudaStream_t stream){
    // NVFP4 overlay for wo_a -- MEASURED AND DEFAULT OFF (DSV4_NVFP4_WOA=1 to re-enable).
    //
    // wo_a is 1.44 GB/token, the largest group in the overlay, and invisible to the fp8_block_gemm
    // dispatch because this path has its own kernel. Routing it through NVFP4 failed twice:
    //
    //   1. SLOWER: 14.31 -> 13.26 tok/s. The fp8 ogroup kernel reads uint32 weights + float4
    //      activations; the NVFP4 version reads 8 scalar floats per 4 weight bytes, so it trades
    //      weight bandwidth for activation bandwidth and loses.
    //   2. LOSSLESS GATE FAIL, which is the disqualifying one. Only bs==1 was routed, so AR (bs==1)
    //      computed with NVFP4 while the spec VERIFY (bs>1) computed the same layer with FP8 --
    //      two different functions inside one engine. The gate caught it at token 5.
    //
    // Fixing (2) means routing the M=K verify path too, so both sides agree. Until then this stays
    // off: an engine whose AR and verify paths disagree is broken regardless of its speed.
    // DEFAULT OFF on THROUGHPUT, not correctness: gate-verified and LOSSLESS-clean, best AR measured
    // (14.83 vs 14.25), but -24% on spec decode (20.88 -> 15.86) because this NR=1 M=K kernel lacks
    // the activation reuse / smem staging / per-M tuning the FP8 ogroup M=K path has. Spec is what
    // ships, so it loses. Tuning the M=K kernel is the unlock.
    if(bs==1 && nvfp4_enabled() && getenv("DSV4_NVFP4_WOA")!=nullptr
       && nvfp4_ogroup_gemv(out, o, wo_fp8, G, R, Kd, stream)){ dsync(stream); return; }
    if(bs==1 && getenv("NO_OGGEMV")==nullptr){          // M=1 decode: bandwidth-bound GEMV (no mma waste). bs>1
        // uint32 weight / float4 activation loads need Kd%128==0 (already required by the e8m0
        // block-scale stride) plus real pointer alignment — `wo_fp8` is a mapped safetensors
        // tensor, and Finding 25 is why that is checked rather than assumed.
        const int vec4 = ((Kd % 128)==0) && ((((uintptr_t)wo_fp8)&3)==0) && ((((uintptr_t)o)&15)==0)
                      && (getenv("NO_OGVEC4")==nullptr);
        int threads=256; ogroup_gemv_fp8_kernel<<<((size_t)G*R*32+threads-1)/threads,threads,0,stream>>>(out,o,wo_fp8,wo_sc,G,R,Kd,vec4);
        dsync(stream); return; }
    // M=K GEMV for the spec-decode verify (Finding 40). The old note here said "M=K GEMV A/B'd
    // SLOWER: acc[bs] array kills occupancy" — that was the UNTEMPLATED version, whose acc[] was
    // sized to the maximum M regardless of the real M; Finding 28 established the same fix for the
    // dense fp8 GEMV. Templated on M, acc[] is M registers, not 16.
    // The M=K kernel is vec4-only (uint32 weight + float4 activation). Both alignments are
    // structural here — wo_a rows are Kd-strided with Kd%128==0, and `o` is arena-allocated — but
    // check anyway: a mapped weight that is merely 1-byte aligned would fault, and Finding 41 is
    // the reminder that "structurally aligned" is a claim about code that has since been rewritten.
    // Unaligned falls through to the f16 tensor-core path below, not to a wrong answer.
    // wo_a verify side. Gated by tests/gate_nvfp4_ogroup_mk.cu, which compares it against M=1 run
    // M times on the real [8x1024, 4096] shape: 0/40960 elements differ.
    //
    // Getting there took two corrections, and neither was a layout error -- the layouts were right
    // from the start. The kernel was NUMERICALLY CORRECT (cosine 1.00000000) and still broke
    // LOSSLESS, because this model turns 1 ulp into a different token and the engine compares the
    // M=1 path's tokens against the M=K path's. Bit-agreement, not correctness, was the bar:
    //   (a) reduce with four accumulators as (a0+a1)+(a2+a3), matching M=1   5 -> 4 elements differing
    //   (b) dot THEN scale, rather than folding the scale into the weights   4 -> 0
    // Both are algebraically identical and round differently. An end-to-end run cannot tell you
    // which of them is wrong; the unit gate answered it in two iterations.
    if(bs>1 && bs<=8 && nvfp4_enabled() && getenv("DSV4_NVFP4_WOA")!=nullptr
       && nvfp4_ogroup_mk(out, o, wo_fp8, bs, G, R, Kd, stream)){ dsync(stream); return; }
    const int ogvec4 = ((Kd % 128)==0) && ((((uintptr_t)wo_fp8)&3)==0) && ((((uintptr_t)o)&15)==0)
                    && (getenv("NO_OGVEC4")==nullptr);
    if(bs>1 && bs<=16 && ogvec4 && getenv("NO_OGMK")==nullptr){
        // NR consecutive output rows share `o` only if the run stays inside one group: R%NR==0.
        // NR is the activation-reuse factor: activation traffic is 4*M/NR times the weight traffic.
        // OG_NR=<n> overrides for A/B; NO_OGNR=1 pins NR=1 (Finding 40's kernel).
        // Measured per M on the real shape [8 x 1024, 4096], 12 rotating copies (gemm_bench
        // "ogroup wo_a COLD"), at OGMK_BLOCKS_PER_SM=4. NR*M accumulators trade activation reuse
        // against occupancy and the optimum is not monotone, so this is a lookup, not a rule:
        //   M=2 -> NR=2 (0.177 ms)  M=3 -> NR=2 (0.213)  M=5 -> NR=4 (0.200-0.228)  M=8 -> NR=2 (0.389)
        // NR=8 spills against the 64-register cap (40 accumulators alone) and loses badly there.
        int ognr = (getenv("NO_OGNR")!=nullptr) ? 1 : (bs<=4 ? 2 : (bs<=6 ? 4 : 2));
        if(const char* e=getenv("OG_NR")) ognr = atoi(e);
        if(ognr!=1 && ognr!=2 && ognr!=4 && ognr!=8) ognr = 4;
        while(ognr>1 && (R % ognr)!=0) ognr >>= 1;
        const int threads=256; const size_t nb=((size_t)G*R*32+threads-1)/threads;
        // SMEM staging of `o` (see ogroup_gemv_mk_smem_kernel). DEFAULT OFF — opt in with OG_SMEM=1.
        //
        // MEASURED, NOT ADOPTED. It does exactly what it claims to the traffic (8x less activation
        // reading) and it is bit-exact at every M, but on the real shape [8 x 1024, 4096] at M=5,
        // 12 rotating copies, it does not beat the kernel it replaces:
        //     NR=1   0.320 -> 0.293      NR=2   0.264 -> 0.200
        //     NR=4   0.199 -> 0.280      NR=8   0.341 -> 0.554
        // The engine runs NR=4 at M=5, where this is a 40% REGRESSION; its best point (NR=2, after
        // doubling KC to halve the barrier count) lands at 0.1996-0.2012 against the incumbent's
        // 0.1968-0.2041. That is a wash, not a win. The mechanism is visible in the NR column: the
        // __syncthreads pair per chunk forces the 8 warps into lockstep and destroys the skew that
        // was hiding load latency, and the more work each warp holds (higher NR) the more that
        // costs. Traffic was not the binding resource; overlap was.
        // Kept reachable because it is the only variant that removes the intra-block redundancy,
        // and a future double-buffered version would start here.
        //
        // BUT THE 40% KILL WAS M=5-SPECIFIC (Finding 79). F55 swept NR at M=5 ONLY. Sweeping M as
        // well (evidence/oglazy_bench.log, same COLD harness, 3 alternating replicates) shows the
        // sign flips below M=4 — against the NR the dispatch actually picks at each M:
        //     M=2  dispatch NR=2:  base 0.1717 -> smem 0.1533  (-10.7%)
        //     M=3  dispatch NR=2:  base 0.1962 -> smem 0.1729  (-11.9%)
        //     M=5  dispatch NR=4:  base 0.1958 -> smem 0.2729  (+39.4%)   <- F55's number
        //     M=8  dispatch NR=2:  base 0.3610 -> smem 0.2810  (-22.2%)
        // So an M-gated `ogsmem` (on at M<=3, off at M>=4) is a real candidate — but priced against
        // the dprof K columns it is worth only ~0.4% end-to-end (o:wo_a is 10.74 ms of an 85.9 ms
        // K=2 verify and 10.26 of 104.7 at K=3, and only 4 of 9 verifies run at K<=3), i.e.
        // SUB-1%, and trap 3 plus F76/F78 both say the bench overstates. LEVERS.md B8''.
        // The two structural preconditions are CHECKED, not assumed: every warp in a block must
        // land in the same o-group, and the staged chunk must divide the row. Both hold for this
        // model's shape, but a shape that broke either would read `o` for the wrong group and
        // still return a plausible number.
        const bool ogsmem = (getenv("OG_SMEM")!=nullptr)
                         && ((R % ((threads/32)*ognr)) == 0) && (((Kd/128) % 8) == 0);
        // OG_SMEM_LAZY: read the staged activation float4 from smem per (r,m) instead of hoisting
        // M of them into 4M registers (LEVERS.md B8', Finding 79). Only meaningful with OG_SMEM=1.
        //
        // MEASURED, NOT ADOPTED, and it CLOSES B8'. The premise was that `float4 o4[M]` is 4M = 20
        // registers at M=5 in a kernel whose <5,4> instantiation is already over its 64-register
        // cap, and that giving them back would cross a real occupancy step (trap 21). The
        // falsification was specified as `ptxas -v` first, and ptxas answered NO:
        //     ogroup_gemv_mk_smem_kernel<5,4>   LAZY=0: 64 regs, 396 B spill
        //                                       LAZY=1: 64 regs, 368 B spill   (-7%, not -16 regs)
        // ptxas simply re-hoists the smem loads back to where `o4[M]` was — it is the better
        // schedule and the source cannot forbid it — so the live set barely moves. The bench agrees
        // (evidence/oglazy_bench.log, COLD, 12 rotating copies, 3 alternating replicates): at the
        // shipped M=5/NR=4 it is -1.1% against the OG_SMEM arm it modifies and still **+37.8%
        // against the kernel that actually ships** (0.2697 vs 0.1958 ms). Across every (M,NR) the
        // lazy-vs-smem delta is inside +-3.4%. Bit-identical at all 9 M (gate_ogroup_gemv).
        const bool oglazy = (getenv("OG_SMEM_LAZY")!=nullptr);
        // WS1: one scale load + one exp2f per k-block instead of NR identical ones (see the kernel
        // note). DEFAULT OFF — opt in with OG_WS1=1.
        //
        // MEASURED, NOT ADOPTED (Finding 76). It is bit-identical (gate_og_ws1, 72/72), it is never
        // slower than the incumbent at any (M,NR) the dispatch picks, and in `gemm_bench` it is
        // worth −7.6% at M=2/NR=2 and −14.8% at M=3/NR=4. In situ it is worth NOTHING: the nine
        // spec verifies pair 1:1 at identical K and accept counts for a total of **+0.1%**
        // (evidence/ogws1.log vs evidence/kchunk.log), spec 21.67 vs 21.68 tok/s, and every ksweep
        // K moved less than 0.5%. The mechanism is in the ncu report that motivated it: this kernel
        // spends 62.3% of its 13.0 cycles between issues stalled on an L1TEX scoreboard, i.e. it is
        // LATENCY-bound, not issue-bound. Deleting instructions from a kernel that is waiting on
        // memory returns the memory latency, which is zero. That closes the whole family — the
        // fp8x2 cvt pairing and the exp2f→`__int_as_float(e<<23)` rewrite have the same shape and
        // need not be built.
        // Kept reachable, not deleted, because the value-equality argument and its gate are the
        // expensive part and a future variant that changes the MEMORY behaviour may want them.
        const bool ogws1 = (getenv("OG_WS1")!=nullptr) && ((128 % ognr) == 0);
        switch(bs){ OG_MK_CASE(2)  OG_MK_CASE(3)  OG_MK_CASE(4)  OG_MK_CASE(5)
                    OG_MK_CASE(6)  OG_MK_CASE(7)  OG_MK_CASE(8)  OG_MK_CASE(9)
                    OG_MK_CASE(10) OG_MK_CASE(11) OG_MK_CASE(12) OG_MK_CASE(13)
                    OG_MK_CASE(14) OG_MK_CASE(15) OG_MK_CASE(16) default: break; }
        dsync(stream); return;
    }
    __half* o16; o16=(__half*)dmalloc((size_t)bs*G*Kd*2);
    k_f2h<<<((size_t)bs*G*Kd+255)/256,256,0,stream>>>(o16,o,(size_t)bs*G*Kd);
    // MT amortises the weight dequant across m-tiles. It only pays when there ARE many m-tiles, so
    // it follows bs: the verify (bs<=16) never reaches here, and a shape with one m-tile gets MT=1,
    // which is the original kernel. OG_TC_MT=1 forces the original for the A/B.
    {   const char* mte = getenv("OG_TC_MT");
        int ntile = (bs+15)/16;
        int MT = mte ? atoi(mte) : (ntile>=8 ? 8 : ntile>=4 ? 4 : ntile>=2 ? 2 : 1);   // may be lowered to 4 by OG_NB below
        if(MT!=1&&MT!=2&&MT!=4&&MT!=8) MT=1;
        int ogwpb = 1; if(const char* w=getenv("OG_WPB")) ogwpb=atoi(w);   // NULL RESULT, see negative-results
        if(ogwpb<1) ogwpb=1; if(ogwpb>8) ogwpb=8;
        const int nb8 = (R+7)/8, gx = (nb8 + ogwpb - 1)/ogwpb, thr = 32*ogwpb;
        // OG_NB: n-blocks per warp. Attacks the activation re-read, which is 97% of this region's
        // traffic. Combos are ENUMERATED after the MoE lesson (trap 46) -- a product rule admitted a
        // wrong kernel there -- and each is verified by an end-to-end token-stream diff.
        // DEFAULT MT=4, NB=4 (2026-08-26), chosen by measurement and not by the traffic model that
        // motivated the knob -- the model predicted 377 ms here and 301 at MT=2/NB=8; measured 1333
        // and 1869, the latter being the WORST config tried. See negative-results: the re-reads this
        // was built to remove are largely L2-served, so removing them does not buy their DRAM cost,
        // and what actually binds is the MT*NB*4 accumulator budget. 4x4 balances it best.
        int ognb = 4; if(const char* e=getenv("OG_NB")) ognb=atoi(e);
        if(ognb!=1&&ognb!=2&&ognb!=4&&ognb!=8) ognb=1;
        if(ognb==4 && MT>4) MT=4;
        if(ognb>1 && ogwpb==1){
            dim3 gridn((nb8+ognb-1)/ognb, G, (ntile+MT-1)/MT);
            #define OGNB_LAUNCH(M,B) tc_ogroup_fp8_mtnb_kernel<M,B><<<gridn,32,0,stream>>>(out,o16,wo_fp8,wo_sc,bs,G,R,Kd)
            bool done=true;
            if      (MT==8&&ognb==2) OGNB_LAUNCH(8,2);
            else if (MT==4&&ognb==2) OGNB_LAUNCH(4,2);
            else if (MT==4&&ognb==4) OGNB_LAUNCH(4,4);
            else if (MT==2&&ognb==4) OGNB_LAUNCH(2,4);
            else if (MT==2&&ognb==8) OGNB_LAUNCH(2,8);
            else if (MT==1&&ognb==8) OGNB_LAUNCH(1,8);
            else done=false;
            #undef OGNB_LAUNCH
            if(done){ dsync(stream); dfree(o16); return; }
        }
        if(MT==1){ dim3 grid(gx, G, ntile);
                   tc_ogroup_fp8_kernel<<<grid,thr,0,stream>>>(out,o16,wo_fp8,wo_sc,bs,G,R,Kd); }
        else {     dim3 grid(gx, G, (ntile+MT-1)/MT);
                   switch(MT){
                     case 2: tc_ogroup_fp8_mt_kernel<2><<<grid,thr,0,stream>>>(out,o16,wo_fp8,wo_sc,bs,G,R,Kd); break;
                     case 4: tc_ogroup_fp8_mt_kernel<4><<<grid,thr,0,stream>>>(out,o16,wo_fp8,wo_sc,bs,G,R,Kd); break;
                     default:tc_ogroup_fp8_mt_kernel<8><<<grid,thr,0,stream>>>(out,o16,wo_fp8,wo_sc,bs,G,R,Kd); break; } }
    }
    dsync(stream); dfree(o16);
}
void ogroup_gemm(float* out, const float* o, const float* wo_a,
                 int bs, int G, int R, int Kd, cudaStream_t stream) {
    if (g_tc_ogroup && Kd%16==0) {
        __half *o16,*wo16; o16=(__half*)dmalloc((size_t)bs*G*Kd*2); wo16=(__half*)dmalloc((size_t)G*R*Kd*2);
        k_f2h<<<((size_t)bs*G*Kd+255)/256,256,0,stream>>>(o16,o,(size_t)bs*G*Kd);
        k_f2h<<<((size_t)G*R*Kd+255)/256,256,0,stream>>>(wo16,wo_a,(size_t)G*R*Kd);
        dim3 grid((R+7)/8, G, (bs+15)/16); tc_ogroup_kernel<<<grid,32,0,stream>>>(out,o16,wo16,bs,G,R,Kd);
        dsync(stream); dfree(o16); dfree(wo16); return;
    }
    ogroup_gemm_kernel<<<bs * G * R, 32, 0, stream>>>(out, o, wo_a, bs, G, R, Kd);
}

// ---------------- act_quant_fp4sim ----------------
// FP4 e2m1 QAT-sim (quant->dequant), pow2 scale, fp4_max=6, block=32. Matches kernel.py fp4_act_quant inplace.
__device__ __forceinline__ float round_e2m1(float v) {   // nearest signed E2M1 grid value {0,.5,1,1.5,2,3,4,6}
    float a = fabsf(v), m;
    if (a < 0.25f) m = 0.f; else if (a < 0.75f) m = 0.5f; else if (a < 1.25f) m = 1.f;
    else if (a < 1.75f) m = 1.5f; else if (a < 2.5f) m = 2.f; else if (a < 3.5f) m = 3.f;
    else if (a < 5.f) m = 4.f; else m = 6.f;
    return (v < 0.f) ? -m : m;
}
__global__ void act_quant_fp4sim_kernel(float* __restrict__ x, int rows, int active_dim, int block, int row_stride) {
    int ng = active_dim / block; int gid = blockIdx.x; if (gid >= rows * ng) return;
    int row = gid / ng, g = gid % ng;
    float* xr = x + (size_t)row * row_stride + (size_t)g * block;
    extern __shared__ float red[];
    float v = (threadIdx.x < block) ? fabsf(xr[threadIdx.x]) : 0.f;
    red[threadIdx.x] = v; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) { if (threadIdx.x < s) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]); __syncthreads(); }
    float amax = fmaxf(red[0], 6.f * 7.5231631e-37f);              // 6*2^-126
    float scale = exp2f(ceilf(log2f(amax * (1.f / 6.f))));
    if (threadIdx.x < block) {
        float q = fminf(fmaxf(xr[threadIdx.x] / scale, -6.f), 6.f);
        xr[threadIdx.x] = round_e2m1(q) * scale;
    }
}
void act_quant_fp4sim(float* x, int rows, int active_dim, int block, int row_stride, cudaStream_t stream) {
    if (row_stride < 0) row_stride = active_dim;
    int threads = block < 32 ? 32 : block;
    act_quant_fp4sim_kernel<<<rows * (active_dim / block), threads, threads * sizeof(float), stream>>>(x, rows, active_dim, block, row_stride);
}

// ================== DECODE_LADDER 1b.2 — packed KV storage ====================================
// The format, the exactness argument and the byte arithmetic are in include/kv_pack.h and
// KV_PRECISION_FINDINGS.md. What lives here is the one kernel that writes it and the four host
// entry points the cache call sites use.
int g_kv_pack = 0;
int g_kv_rowf = dsv4::HEAD_DIM;
// AUTO-ENABLE ABOVE A CONTEXT THRESHOLD (2026-08-26). Packing is BIT-EXACT -- verified by identical
// token streams over 400 generated tokens -- because the FP32 path already runs act_quant_fp8sim on
// these rows and then stores the dequantised float in 4 bytes. Packing stores the e4m3 code instead.
// Same values, 720 B/row instead of 2048.
//
// It is not free in TIME: the consumer redoes the conversion on every read, measured at -12 % prefill
// (107.9 -> 94.8 tok/s). So it should not be unconditional -- an 8k session would pay for headroom it
// never uses. It should also not be off by default, because without it long context does not fit:
//
//     seqmax    fp32                        packed
//      8,192    ready, 119.6/122.8 GiB      --
//     32,768    ready, 119.9/122.8 GiB      --
//    131,072    NEVER REACHES READY         ready, 117.0/122.8 GiB
//
// fp32 is proven to 32,768 and proven to fail at 131,072, so the threshold sits at the top of what
// has actually been measured to work. DSV4_KV_PACK=0 forces it off, =1 forces it on; unset means
// "decide from seqmax", which is the only setting that is right at both ends of that table.
void kv_pack_init_seqmax(int seqmax) {
    const char* e = getenv("DSV4_KV_PACK");
    if (e) g_kv_pack = atoi(e) ? 1 : 0;
    else   g_kv_pack = (seqmax > 32768) ? 1 : 0;
    g_kv_rowf = g_kv_pack ? dsv4kv::KVP_ROWF : dsv4::HEAD_DIM;
}
void kv_pack_init() { kv_pack_init_seqmax(0); }

// k_kv_pack IS act_quant_fp8sim, with the store changed and nothing else.
//
// Same 64-thread block per (row, 64-wide group), same shared max-reduction over the same 64 |x|
// values, same `fmaxf(amax, 1e-4f)`, the same `exp2f(ceilf(log2f(amax/448)))` scale, the same clamp
// and the same `__nv_fp8_e4m3` conversion. act_quant_fp8sim then converts the e4m3 code BACK to a
// float and stores 4 bytes; this stores the code itself plus one exponent byte per group, and the
// consumer redoes that conversion at read time. A max-reduction is exact and order-independent, so
// `amax` is bit-identical whatever order the tree runs in, and every step after it is a pure
// function of amax and x. The value a reader sees is therefore the value act_quant_fp8sim would
// have written -- which is what makes memcmp, and not a tolerance, the acceptance test for 1b.2.
//
// d_idx  : destination row index read from device memory (the CUDA-graph decode paths).
// d_pos  : when non-null with ratio>0, the whole store is skipped unless (*d_pos+1)%ratio==0 --
//          k_commit_comp's "append only when a group completes" predicate, unchanged.
__global__ void k_kv_pack(uint8_t* __restrict__ dst, const float* __restrict__ src,
                          int rows, int src_stride, long long dst_row0,
                          const int* __restrict__ d_idx, const int* __restrict__ d_pos, int ratio) {
    using namespace dsv4kv;
    if (d_pos && ratio > 0 && (((*d_pos) + 1) % ratio) != 0) return;
    const int r = blockIdx.x; if (r >= rows) return;
    const long long drow = (d_idx ? (long long)(*d_idx) : dst_row0) + r;
    const float* xr = src + (size_t)r * (size_t)src_stride;
    uint8_t* p = dst + (size_t)drow * (size_t)KVP_ROWB;
    const int tid = threadIdx.x;                      // blockDim.x == KVP_BLOCK == 64
    __shared__ float red[KVP_BLOCK];
    for (int g = 0; g < KVP_NB; ++g) {
        const float x = xr[g * KVP_BLOCK + tid];
        red[tid] = fabsf(x); __syncthreads();
        for (int s = KVP_BLOCK / 2; s > 0; s >>= 1) {
            if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]);
            __syncthreads();
        }
        const float amax  = fmaxf(red[0], 1e-4f);
        const float scale = exp2f(ceilf(log2f(amax * (1.f / 448.f))));
        const float q     = fminf(fmaxf(x / scale, -448.f), 448.f);
        p[g * KVP_BLOCK + tid] = (uint8_t)(__nv_fp8_e4m3(q).__x);
        if (tid == 0) p[KVP_OFF_SCALE + g] = kv_ue8m0(scale);
        __syncthreads();
    }
    for (int j = tid; j < KVP_ROPE; j += KVP_BLOCK)   // RoPE half: exact fp32 copy, never quantised
        ((float*)(p + KVP_OFF_ROPE))[j] = xr[KVP_NOPE + j];
}

float* kv_stage(float* cache_row, int rows) {
    if (!g_kv_pack) return cache_row;                 // write in place, exactly as before 1b.2
    return (float*)dmalloc((size_t)rows * (size_t)dsv4::HEAD_DIM * 4);
}
void kv_commit(float* cache_row, float* staged, int rows, cudaStream_t stream) {
    if (!g_kv_pack) { act_quant_fp8sim(cache_row, rows, dsv4::NOPE_DIM, 64, dsv4::HEAD_DIM, stream); return; }
    if (rows <= 0) return;
    k_kv_pack<<<rows, dsv4kv::KVP_BLOCK, 0, stream>>>((uint8_t*)cache_row, staged, rows,
                                                      dsv4::HEAD_DIM, 0, nullptr, nullptr, 0);
}
void kv_stage_free(float* staged, float* cache_row) {
    if (g_kv_pack && staged != cache_row) dfree(staged);
}
// Device-index store (replaces k_append_at2 on the packed path). `src_row` is the un-quantised
// fp32 row: the pack kernel does the quantisation the caller would otherwise have done.
void kv_store_at(float* cache_base, const float* src_row, const int* d_idx, cudaStream_t stream) {
    k_kv_pack<<<1, dsv4kv::KVP_BLOCK, 0, stream>>>((uint8_t*)cache_base, src_row, 1,
                                                   dsv4::HEAD_DIM, 0, d_idx, nullptr, 0);
}
// Conditional compressed-row store (replaces k_commit_comp on the packed path).
void kv_store_comp(float* comp_base, const float* cand, const int* d_T, const int* d_pos,
                   int ratio, cudaStream_t stream) {
    k_kv_pack<<<1, dsv4kv::KVP_BLOCK, 0, stream>>>((uint8_t*)comp_base, cand, 1,
                                                   dsv4::HEAD_DIM, 0, d_T, d_pos, ratio);
}
