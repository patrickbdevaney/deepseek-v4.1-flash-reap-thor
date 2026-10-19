// mla_forward.h — full MLA attention forward for a pure-sliding layer (compress_ratio=0), prefill start_pos=0.
// Composition of the Gate-K-validated primitives (fp8_block_gemm, rmsnorm, rope, act_quant, sparse_attn,
// ogroup_gemm) in model.py:490-548 order. Correctness-first, fp32 activations. Gate: tests/gate_mla.cu.
#pragma once
#include <cstdint>
#include <cuda_runtime.h>

struct MLAWeights {
    // fp8 e4m3 weight bytes + f32 (pow2) block scales
    const uint8_t *wq_a, *wq_b, *wkv, *wo_b;
    const float   *wq_a_s, *wq_b_s, *wkv_s, *wo_b_s;
    const float   *q_norm, *kv_norm;     // f32 [q_lora], [head_dim]
    const float   *wo_a;                 // f32 [n_groups, o_lora, n_heads*head_dim/n_groups]  (dequanted path)
    const float   *attn_sink;            // f32 [n_heads]
    const float   *cosT, *sinT;          // f32 [s, rope_dim/2]  (per position)
    // NATIVE wo_a decode path (memory-neutral, no per-token fp8->f32 dequant): fp8 bytes + e8m0 block scale,
    // converted straight to f16 for the TC ogroup GEMM. When wo_a_native, ogroup uses these instead of wo_a.
    bool           wo_a_native = false;
    const uint8_t *wo_a_fp8 = nullptr, *wo_a_sc = nullptr;   // [G*R, Kd] fp8 ; e8m0 scale [G*R/128, Kd/128]
};

// x:[b*s, dim] fp32 -> out:[b*s, dim] fp32. b must be 1 (single-sequence prefill) for the cos indexing.
void mla_forward(float* out, const float* x, const MLAWeights& w, int b, int s, cudaStream_t stream = 0);
