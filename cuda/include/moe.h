// moe.h — DeepSeek-V4 MoE primitives (deepseek_v4). Correctness-first, fp32 I/O for Gate K.
// fp4_gemm: FP8-act x FP4-weight GEMM (kernel.py fp4_gemm) — the routed-expert GEMM.
// moe_router_score: sqrtsoftplus + noaux_tc (bias-for-selection) topk + renorm*route_scale.
#pragma once
#include <cstdint>
#include <cuda_runtime.h>

// C[M,N] = A_fp8[M,K] @ B_fp4[N,K]^T. a_s[M,K/128] (per-128 act, f32). B_fp4 packed [N,K/2] (2 nibbles/byte).
// b_s[N,K/32] (per-32 weight, f32 pow2). E2M1 nibble = sign(bit3)|magnitude-index.
void fp4_gemm(float* C, const uint8_t* A_fp8, const float* a_s,
              const uint8_t* B_fp4, const float* b_s, int M, int N, int K, cudaStream_t stream = 0);

// Score router. x[n,dim], gate_w[n_routed,dim] (f32). -> weights[n,topk], indices[n,topk].
// scores = sqrtsoftplus(x@gate_w^T); select top-k of (scores+bias); weights = gather(scores)/sum*route_scale.
void moe_router_score(float* weights, int* indices, const float* x, const float* gate_w,
                      const float* bias, int n, int dim, int n_routed, int topk,
                      float route_scale, cudaStream_t stream = 0);

// Full MoE forward (model.py:551-649). Routed experts fp4 SwiGLU (top-k) + always-on fp8 shared expert.
// Weights: routed w1/w3 packed [E,inter,dim/2] + scale [E,inter,dim/32]; w2 [E,dim,inter/2] + [E,dim,inter/32].
// Shared sw1/sw3 fp8 [inter,dim] + scale [ceil(inter/128),ceil(dim/128)]; sw2 [dim,inter] + [.,.].
struct MoEWeights {
    const float* gate_w;                       // [n_routed, dim] f32
    const float* gate_bias;                    // [n_routed] f32 (null for hash layers)
    const long*  tid2eid;                      // [vocab, n_act] i64 (hash layers) else null
    bool is_hash;
    const uint8_t *w1, *w2, *w3;  const float *w1s, *w2s, *w3s;     // routed experts (fp4), STACKED [E,...]
    const uint8_t *sw1, *sw2, *sw3; const float *sw1s, *sw2s, *sw3s; // shared expert (fp8)
    int n_routed, n_act, dim, inter, vocab;
    float route_scale, swiglu_limit;
    bool use_tc = false;   // route routed-expert GEMMs through tc_fp4_gemm (Marlin TC, ~3x). false = fp4_gemm oracle (bit-exact gate).
    bool use_tc_pp = false;// repack-at-load: tc_fp4_gemm_pp_auto (in-place repack once, then pp GEMM; no OOM, no per-layer repack).
    bool device_route = false;// device-side MoE grouping (counting sort on GPU) — removes the per-layer host round-trip (Step 1 toward CUDA graphs).
    bool batched = false;  // group tokens by expert -> one GEMM per expert at M=count (amortize weight loads). false = per-token oracle.
    // Real-checkpoint path: per-expert device-pointer tables (HOST arrays of device ptrs, len n_routed).
    // If w1p != null, expert e uses w1p[e]/w1sp[e]/... instead of the stacked w1+e*stride. Gates leave null.
    const uint8_t *const *w1p = nullptr, *const *w2p = nullptr, *const *w3p = nullptr;
    const float   *const *w1sp = nullptr, *const *w2sp = nullptr, *const *w3sp = nullptr;
    // NATIVE-e8m0 decode path: per-expert e8m0 scale-BYTE tables (persistent WeightStore ptrs, no dequant).
    // When e8m0_scales, the grouped MoE reads these instead of w1sp/... and dequants exp2(byte-127) in-kernel.
    bool e8m0_scales = false;
    const uint8_t *const *w1sp8 = nullptr, *const *w2sp8 = nullptr, *const *w3sp8 = nullptr;
    // LOOP_LOG Finding 37. The six pointer tables above are HOST arrays and the grouped MoE needs
    // them on the device, so it was issuing six cudaMemcpyAsync H2D copies -- per layer, per token.
    // 43 layers x 6 = 258 copies every step, of data that is fixed for the life of the process and
    // is not even pinned. Cached in one device allocation on first use; the layer structs are
    // persistent (decode.cu builds them once), so `mutable` here is a genuine cache, not state.
    mutable void* dev_ptr_tables = nullptr;    // 6 * n_routed device pointers, laid out w1,w3,w2,s1,s3,s2
};
// x:[bs,dim] fp32, input_ids:[bs] i32 (for hash routing) -> out:[bs,dim] fp32.
extern long long g_moe_union_sum;   // DSV4_MOEUNION=1: summed distinct experts per call
extern int       g_moe_union_calls;
extern long long g_moe_rows_sum;
// B9. A grouped tile is 16 rows (tc_build_tiles: tile_row0 = r0 + 16*j) and EVERY tile re-reads its
// expert's whole weight matrix, so weight traffic is ntiles x (w1+w3+w2), not n_experts x (...).
// At bs=1 that distinction is invisible; at a 1022-token prefill it is the whole question of
// whether the MoE is latency-bound or is simply moving several times the bytes it needs to.
extern long long g_moe_tiles_sum;   // DSV4_MOEUNION=1: summed 16-row tiles per call
// TILES ARE NOT THE TRAFFIC. The weight load in k_grouped_fp4_gemv_e8m0 sits INSIDE the `rb` loop,
// so one tile costs ceil(me/RB) full weight reads, not one. Counting tiles undercounted the prefill
// by 3.4x and made a bandwidth-bound kernel look latency-bound. Count the chunks.
extern long long g_moe_chunks_sum;  // DSV4_MOEUNION=1: summed RB-chunks = actual weight reads
extern int       g_moe_rows_max;
extern long long g_moe_rows_hist[10];

void moe_forward(float* out, const float* x, const int* input_ids, const MoEWeights& w,
                 int bs, cudaStream_t stream = 0);
