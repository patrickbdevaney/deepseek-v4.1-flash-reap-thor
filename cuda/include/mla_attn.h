// mla_attn.h — MLA attention primitives (deepseek_v4). Correctness-first, fp32 I/O for Gate K.
// sparse_attn: gathered top-k attention w/ learnable sink (kernel.py:276-368).
// rope_interleaved: interleaved-pair RoPE fwd/inverse (model.py:238-250).
// rmsnorm: fp32 RMSNorm, optional weight (model.py:189-202; per-head q-norm has no weight).
#pragma once
#include <cstdint>
#include <cuda_runtime.h>

// q:[b,m,h,d] kv:[b,n,d] attn_sink:[h] topk_idxs:[b,m,topk] (-1=masked) -> o:[b,m,h,d]. All fp32; sink fp32.
void sparse_attn(float* o, const float* q, const float* kv, const float* attn_sink,
                 const int* topk_idxs, int b, int m, int h, int d, int n, int topk,
                 float scale, cudaStream_t stream = 0);

// Explicit-launch form (DECODE_LADDER 1.7). `sparse_attn` resolves (hpb, smem) from the env and the
// heuristic and then calls this; a gate/bench calls it directly so it can sweep BOTH knobs inside one
// process, which `static getenv` caching in `sparse_attn` makes impossible.
//   hpb  = heads-per-block, 1|2|4|8. hpb==1 is the kernel that shipped before 1.7, byte for byte.
//   smem = 0 -> every warp re-reads the gathered KV row from global; 1 -> the block stages the row in
//          shared memory ONCE per t, with float4 loads and double buffering, and each warp reads its
//          lanes back out in the ORIGINAL lane mapping. Bit-exact with smem=0 by construction: the
//          per-lane partial sums, the 5-step shuffle tree and the online-softmax order are untouched.
//   packed = the KV cache is in the 720 B FP8+UE8M0 layout of include/kv_pack.h (DECODE_LADDER
//          1b.2). Changes only how the block STAGES a gathered row; the lane mapping, the shuffle
//          tree and the online-softmax order are the same source lines, so it is bit-exact by
//          construction and `gate_kv_pack` proves it by memcmp against the FP32 launch.
void sparse_attn_launch(float* o, const float* q, const float* kv, const float* attn_sink,
                        const int* topk_idxs, int b, int m, int h, int d, int n, int topk,
                        float scale, cudaStream_t stream, int hpb, int smem, bool packed = false);

// KV-CACHE entry point: `sparse_attn` with the layout taken from `g_kv_pack` (see include/kv_pack.h).
void sparse_attn_kv(float* o, const float* q, const float* kv, const float* attn_sink,
                    const int* topk_idxs, int b, int m, int h, int d, int n, int topk,
                    float scale, cudaStream_t stream = 0);

// In-place interleaved-pair RoPE on the rope_dim slice of each of `rows` rows. inverse => conj.
// row_stride: element stride between consecutive rows in x (default rope_dim = contiguous). Pass x already
//   offset to the slice start (e.g. head_base + nope_dim) with row_stride = head_dim for per-head slices.
// cos_stride_rows: how many consecutive x-rows share one cos/sin row (=num heads per position; 1 if 1:1).
void rope_interleaved(float* x, const float* cosT, const float* sinT,
                      int rows, int rope_dim, bool inverse,
                      int row_stride = -1, int cos_stride_rows = 1, cudaStream_t stream = 0);
// Device-pos variant (CUDA-graph capturable): cos/sin row = (*d_pos)+row/cos_stride_rows from the full table.
void rope_interleaved_dp(float* x, const float* cosT, const float* sinT, int rows, int rope_dim, bool inverse,
                         int row_stride, int cos_stride_rows, const int* d_pos, cudaStream_t stream = 0);

// y[rows,dim] = x * rsqrt(mean(x^2)+eps) * (weight if has_weight). fp32.
void rmsnorm(float* y, const float* x, const float* weight, int rows, int dim,
             float eps, bool has_weight, cudaStream_t stream = 0);

// Fused FP8 QAT-sim: per `block`-group over the first `active_dim` of each `row_stride`-strided row,
// quant->e4m3 (pow2/ue8m0 scale)->dequant, in-place. fp32. row_stride<0 => active_dim (contiguous).
void act_quant_fp8sim(float* x, int rows, int active_dim, int block, int row_stride = -1,
                      cudaStream_t stream = 0);

// Real activation quant: x_fp32[rows,K] -> a_fp8[rows,K] (e4m3 bytes) + a_s[rows,K/128] (f32 pow2 scale).
// The activation half of an fp8 linear (ref linear(): act_quant then fp8_gemm). block=128 along K.
void act_quant_fp8(uint8_t* a_fp8, float* a_s, const float* x, int rows, int K, int block,
                   cudaStream_t stream = 0);

// Fused FP4 QAT-sim: per `block`(=32)-group over first `active_dim` of each row_stride-strided row,
// quant->e2m1 (pow2 scale, fp4_max=6)->dequant, in-place fp32 (kernel.py fp4_act_quant inplace=True).
void act_quant_fp4sim(float* x, int rows, int active_dim, int block, int row_stride = -1,
                      cudaStream_t stream = 0);

// Grouped o-LoRA GEMM: out[bs,G,R] = sum_d o[bs,G,d] * wo_a[G,R,d]. fp32.
void ogroup_gemm(float* out, const float* o, const float* wo_a,
                 int bs, int G, int R, int Kd, cudaStream_t stream = 0);
// NATIVE wo_a variant: wo_a as fp8 bytes + e8m0 block scale (no per-token f32 dequant). TC path.
void ogroup_gemm_fp8(float* out, const float* o, const uint8_t* wo_fp8, const uint8_t* wo_sc,
                     int bs, int G, int R, int Kd, cudaStream_t stream = 0);
