// block_decode.h — full-Block M=1 KV-cache decode wrappers (STRUCTURAL_PLAN Step 4, milestone 3).
// A Block = hc_pre -> rmsnorm -> attention -> hc_post (attn) ; hc_pre -> rmsnorm -> moe -> hc_post (ffn).
// All ops except attention are per-row, so M=1 reuses them directly; attention swaps to the gated decode steps
// (mla_decode / compressed_decode_*), reading per-layer KV caches populated during a prefill pass.
#pragma once
#include <cuda_runtime.h>
#include "block.h"
#include "compressed_block.h"

// Per-layer KV cache (append-only).
//
// ROW STRIDE IS `g_kv_rowf` FLOATS, NOT HEAD_DIM (DECODE_LADDER 1b.2, include/kv_pack.h). The main
// KV rows are 512 fp32 (2048 B) in the default layout and 180 fp32 (720 B, FP8 E4M3 + 7 x UE8M0 +
// fp32 RoPE) under DSV4_KV_PACK=1, and the two hold the IDENTICAL float values -- the FP32 layout
// was already storing values that sit exactly on the E4M3 grid. Address a row with `kv_row(base, i)`
// and size an allocation or a memcpy with `kv_rows_bytes(rows)`; `+ i * HEAD_DIM` is wrong in one of
// the two layouts. `idx_ckv` is NOT covered by this -- it is the DSA index cache, 1b.1's format,
// still [Tmax, index_head_dim] fp32.
//
// win_kv:[seqmax,row]; comp_kv:[Tmax,row] (compressed);
// idx_ckv:[Tmax,index_head_dim] (ratio-4 only); xin:[seqmax,DIM] = per-position ATTENTION-INPUT history that
// the compressor pools (overlap groups span the prefill/decode boundary, so prefill x1 must be retained).
// T = compressed rows emitted so far.
struct LayerKV { float* win_kv=nullptr; float* comp_kv=nullptr; float* idx_ckv=nullptr; float* xin=nullptr; int T=0;
    // device-pos / CUDA-graph fields: combined cache [seqmax(window) + Tmax(compressed)][row] so attention
    // needs no per-step copy; d_T = device compressed-row count (graph reads it, the emit advances it on device).
    float* kvc=nullptr; float* idx_kvc=nullptr; int* d_T=nullptr; int winmax=0; };

// ---- sliding (ratio==0) ----
void block_prefill_cache(float* out, const float* x, const int* input_ids, const BlockWeights& w,
                         int s, int iters, float eps, LayerKV& kv, cudaStream_t stream = 0);
void block_decode_step(float* out, const float* x, const int* input_ids, const BlockWeights& w,
                       int pos, int iters, float eps, LayerKV& kv, cudaStream_t stream = 0);

// ---- compressed (ratio==4 indexer / ratio==128 strided; branch on w.ratio) ----
void cblock_prefill_cache(float* out, const float* x, const int* input_ids, const CompressedBlockWeights& w,
                          int s, int iters, float eps, LayerKV& kv, cudaStream_t stream = 0);
void cblock_decode_step(float* out, const float* x, const int* input_ids, const CompressedBlockWeights& w,
                        int pos, int iters, float eps, LayerKV& kv, cudaStream_t stream = 0);

// M=K VERIFY block steps (spec-decode): K tokens [pos..pos+K-1] through the block, GEMMs+MoE at M=K.
void block_verify_step(float* out, const float* x, const int* input_ids, const BlockWeights& w,
                       int pos, int K, int iters, float eps, LayerKV& kv, cudaStream_t stream = 0);
void cblock_verify_step(float* out, const float* x, const int* input_ids, const CompressedBlockWeights& w,
                        int pos, int K, int iters, float eps, LayerKV& kv, cudaStream_t stream = 0);

// device-pos block steps (CUDA-graph capturable): driven by device *d_pos, *d_g; d_curid = 1-elem device token id
// (hash-layer routing). Compressed uses the combined cache kv.kvc / kv.idx_kvc + device kv.d_T.
void block_decode_step_dp(float* out, const float* x, const int* d_curid, const BlockWeights& w,
                          const int* d_pos, int nkv, int iters, float eps, LayerKV& kv, cudaStream_t stream);
void cblock_decode_step_dp(float* out, const float* x, const int* d_curid, const CompressedBlockWeights& w,
                           const int* d_pos, int* d_g, int winmax, int Tmax, int iters, float eps, LayerKV& kv, cudaStream_t stream);
