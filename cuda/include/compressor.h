// compressor.h — KV Compressor primitives (deepseek_v4, model.py:285-383).
// Core = learned gated-softmax pooling over `ratio` consecutive tokens. wkv/wgate are fp32 linears.
// This header covers the non-overlap pooling core (ratio!=4). Overlap (ratio==4) + DSA indexer come next.
#pragma once
#include <cuda_runtime.h>

// C[M,N] = A[M,K] @ B[N,K]^T, all fp32.
void gemm_fp32(float* C, const float* A, const float* B, int M, int N, int K, cudaStream_t stream = 0);
// LADDER 1.12. (MM,NN) warp tile for the M>=2 path. Set from DSV4_F32MK_TILE=MMxNN unless a caller
// pins it first; a setter exists so one binary can run both arms of an A/B without re-exec.
void gemm_fp32_set_tile(int mm, int nn);
void gemm_fp32_get_tile(int* mm, int* nn);
// C[M,N] = A[M,K](f32) @ B[N,K]^T with B read natively as BF16 (no f32 dequant). See LOOP_LOG
// Finding 26: lm_head and the markov heads ship BF16 and were being doubled to f32 before every use.
extern bool g_compressor_bf16;   // wkv/wgate are BF16 storage when true
void gemm_bf16w(float* C, const float* A, const void* Bbf16, int M, int N, int K, cudaStream_t stream = 0);
void gemm_fp32_cond(float* C, const float* A, const float* B, int M, int N, int K, const int* d_pos, int ratio, cudaStream_t stream = 0);

// Gated pooling: pooled[g,e] = Σ_p softmax_p(score[g*ratio+p,e] + ape[p,e]) * kv[g*ratio+p,e].
// kv,score:[groups*ratio, d]; ape:[ratio,d]; pooled:[groups,d].
void compressor_pool(float* pooled, const float* kv, const float* score, const float* ape,
                     int groups, int ratio, int d, cudaStream_t stream = 0);

// Overlap pooling (ratio==4, model.py overlap_transform + softmax). kv,score:[groups*ratio, 2d];
// ape:[ratio,2d]; pooled:[groups,d]. Each group softmaxes over 2*ratio slots: current group (dims [d:2d])
// + previous group (dims [0:d], masked for g=0).
void compressor_pool_overlap(float* pooled, const float* kv, const float* score, const float* ape,
                             int groups, int ratio, int d, cudaStream_t stream = 0);

// Full Compressor forward (prefill, remainder-free): gemm(wkv/wgate) -> pool -> norm -> RoPE(last 64) ->
// [rotate ? hadamard(full d) + fp4-sim(full d) : fp8-sim(NoPE)]. x:[s,dim] -> out:[s/ratio, d].
// cos/sin:[s/ratio, 64/2] (compressed-position freqs). rotate=True is the DSA indexer's compressor.
void compressor_forward(float* out, const float* x, const float* wkv, const float* wgate,
                        const float* ape, const float* norm_w, const float* cosT, const float* sinT,
                        int s, int dim, int d, int ratio, bool overlap, int rope_dim, float eps,
                        bool rotate, cudaStream_t stream = 0);

// Incremental: emit ONE compressed row (= compressor_forward's out[g]) from just group g's tokens (decode
// append-only KV). Non-overlap pools x[g*ratio..]; overlap (ratio==4) pools [(g-1)*ratio .. g*ratio+ratio-1].
// Rows in the xin ring (0 = xin holds the full history). See compressor.cu for the invariant.
extern int g_xin_ring;      // rows for the current layer (0 = full history)
extern bool g_xin_ring_on;
extern int  g_xin_ring_batch;  // widest batch the ring must survive

// XIN RING GEOMETRY -- defined once, here, because two places depend on it and they must agree:
// src/engine.cu ALLOCATES the buffer and kernels/block_decode.cu INDEXES into it. If those ever
// disagreed the result would be silently wrong rows rather than a crash.
//
//   R (rows)  = batch + 2*ratio, rounded up to a multiple of ratio
//               -- a step writes its whole batch before emitting any group, so the live window is
//                  the batch PLUS the compressor's 2*ratio lookback, not the lookback alone.
//   alloc     = R + ratio
//               -- tok0 % R is at most R - ratio and a group window is at most 2*ratio long, so it
//                  reaches R + ratio - 1; the first `ratio` rows are mirrored there to keep the
//                  window contiguous for the GEMM.
static inline int xin_ring_rows(int ratio) {
    if (!g_xin_ring_on || ratio <= 0) return 0;
    const int need = g_xin_ring_batch + 2 * ratio;
    return ((need + ratio - 1) / ratio) * ratio;
}
static inline int xin_ring_alloc_rows(int ratio) {
    const int R = xin_ring_rows(ratio);
    return R ? R + ratio : 0;
}

void compressor_emit_group(float* out_row, const float* x, int g, int ratio, const float* wkv,
                           const float* wgate, const float* ape, const float* norm_w,
                           const float* cc_cos, const float* cc_sin, int dim, int d, bool overlap,
                           int rope_dim, float eps, bool rotate, cudaStream_t stream = 0);
