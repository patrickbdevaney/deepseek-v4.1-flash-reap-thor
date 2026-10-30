# P8 — prefill GEMMs on tcgen05 (Thor sm_110a)

Measured 2026-10-30 (CUTLASS e8ecfad, CUDA 13.0, builds in the session scratchpad, `-arch=sm_110a -lcuda`,
examples patched only for the `props.major` / CUDA-version checks).

## Why

8k prefill (193.5 tok/s, pf3 profile) is compute-bound on `mma.sync`: k_nvm_big 10.2 s, k_kqm 14.2 s,
Engram k_dense / k_f8m 2.9 / 0.85 s. Our fp8 `mma.sync` kernels reach ~6-11 TFLOPS. That is NOT the
mma.sync peak (scratch `mmarate.cu`, 160 CTAs x 8 warps x 4 independent chains): bf16 m16n8k16 64,
e4m3 m16n8k16 119, e4m3 m16n8k32 226 TFLOPS (8.0 / 4.3 / 4.6 cycles per mma per SMSP). Our kernels lose
to the scalar work around each mma (scale epilogues, operand building), not to the tensor pipe. tcgen05 + TMEM (the
only Blackwell-class tensor path on Thor; `mma.sync ... block_scale` is sm_120a-only) reaches:

| kernel (CUTLASS example) | shape M x N x K | runtime | TFLOPS |
|---|---|---:|---:|
| 72b NVFP4 x NVFP4 block-scaled | 256 x 5120 x 5120 | 0.056 ms | 238 |
| 72b | 256 x 32768 x 1280 | 0.071 ms | 303 |
| 72b | 2048 x 8192 x 8192 | 0.84 ms | 326 |
| 70 FP8 (unscaled) | 256 x 5120 x 5120 | 0.055 ms | 244 |
| 70 | 256 x 25600 x 6144 | 1.14 ms | 70 (BW: 157 MB) |
| 86 mixed input e2m1 + bf16 scale/16, bf16 act | 256 x 32768 x 1280 | 0.32 ms | 66 |
| 86 | 256 x 5120 x 8192 | 0.30 ms | 72 |
| 86 | 256 x 2304 x 5120 | 0.10 ms | 58 |
| 86 | 256 x 25600 x 6144 | 1.09 ms | 74 |

## Exactness of the operands (T1: only accumulation order changes)

- Activations: act_quant_fp8 scales are `pow2_ceil` -> x = e4m3 x 2^k, exact in bf16 (and exact MXFP8).
- NVFP4 overlay weights: e2m1 (2 sig bits) x e4m3 scale (4 sig bits) = <= 6 sig bits, exact in bf16;
  the global g is the epilogue alpha. -> mixed-input (86) path computes on exactly the decode operands.
- FP8 weights (Engram wkv, drafter): e4m3 x e8m0 per 32x32 = exact MXFP8 -> block-scaled MXFP8 (kind::mxf8f6f4).
- K-quant experts: codes 0..3 / -4..3 are exact e2m1, but the per-16 scale d(fp16) x sc(4-bit) and the
  min term are not representable as a ue8m0/ue4m3 block scale -> bf16 dequant (rounds d*sc*q: T1/T2, KL gate).

Decode / verify keep k_nvm / k_kqm, so decode == verify stays bitwise; prefill != decode by accumulation
order (consistency check becomes a near-tie check, judged by teacher-forced KL as with P3/P6).

## Plan

1. Dense NVFP4 overlay prefill (k_nvm_big 10.2 s): per layer, convert the repacked codes/scales into the
   CUTLASS layout in a per-layer scratch (85 MB read + write ~0.9 ms vs ~6.7 ms saved) -> no extra resident
   memory; x -> bf16 once per act_quant. wo_a's 8 groups as L = 8 batched problems. Expected 8k prefill
   193 -> ~245 tok/s.
2. Engram FP8 via block-scaled MXFP8 (exact operands): 13.2 -> ~1.1 ms/launch.
3. Experts (k_kqm 14.2 s): grouped GEMM (CUTLASS 92 MoE / 75 grouped) on bf16-dequantized tiles; largest
   remaining share; T1/T2 with KL gate.

## Stage 2 (experts) findings, 2026-10-30

- `tests/bench_kqm_units.cu`: k_kqm at MT = 16 costs the same per token whether 8 units of one expert
  re-read it (L2) or 96 distinct experts stream from DRAM -> issue-bound (~89 GB/s, ~11 TFLOPS), so
  bigger units / chunks cannot help by themselves.
- k_kqb (bf16 m16n8k16, A = bf16(128 + q*sc) built with IMUL + 2 PRMT, code-free term as a second mma on
  bf16 m and hi/lo xm; T1): correct (gate vs fp64 at k_kqm's error), 2.7x fewer instructions, but the
  bf16 x stage (11.6 KB/token) allows 1 CTA/SM and the A build is still ~7 int ops per HMMA because one
  converted weight serves only 16 tokens. Best variant (RT 8, 2-ahead LDG prefetch, 4 accumulator chains)
  1.11x gate|up, 1.28x down vs k_kqm; 7-11% over the bitwise k_kqm RT fix. Not shipped.
- Shipped: k_kqm RT = 4 row tiles per warp for Q2_K (stage copied once per 512 rows; bitwise), radix-sort
  k_units_big (bitwise), k_f8m default on.
- What would move experts: tcgen05 with the K-quant -> bf16 conversion done once per 128-row smem tile and
  consumed by UMMA across all tokens of the unit (conversion cost / token falls with MT), or MT >= 64.
