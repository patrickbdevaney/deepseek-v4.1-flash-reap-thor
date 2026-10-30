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

## Stage 2b: k_kqt, K-quant experts on tcgen05 (2026-10-30)

`cuda/kernels/v41/v41_kqt.cuh` (+ host `v41_kqt.cu`, gate `tests/gate_kqt.cu`). One CTA = 128 weight rows
of one expert unit (<= 64 tokens); per superblock the CTA decodes the 128 x 256 tile ONCE into smem and
UMMA consumes it for every token of the unit.

- Operands, exact integers: Q2_K A = bf16(128 + q*sc) (`0x4300 | v`, IMUL + 2 PRMT per 4 codes);
  Q3_K A = f16(1280 + sc*c) (`0x6500 + v`, two 16-bit lanes per IMAD, kind f16). B = x rounded to the
  operand type in a prep stage (with xsum / xm summed from the ROUNDED x so the offset cancels).
  Min / code-free term = a second K=32 UMMA (A_m = m or sc, B_m = xm hi | lo) into its own TMEM columns.
- Pipeline: 8 producer warps (decode A, cp.async B, warp 0 issues 18 UMMAs) + 4 drain warps (tcgen05.ld,
  fold d / dmin), two smem stages + two TMEM accumulator pairs, mbarriers mma_done[2] / tmem_free[2];
  A bytes register-prefetched two superblocks ahead; persistent over units (grid N/128 x V41_KQT_GY = 5).
- Gate (24 experts x 64 tokens, vs fp64 and k_kqm): error = k_kqm's (bf16 output rounding dominates).
  Speed vs k_kqm(RT4) on 64-token units: Q2 gate|up 3.16x, down 2.92x; Q3 3.01x / 2.76x. A unit costs
  ~the same at any token count (the A decode dominates), so units < 9 tokens stay on k_kqm (V41_KQT_MIN).
- Progression of the gate|up time (ms, k_kqm 6.5): sequential 7.5 (wrong swizzle fixed) -> software
  pipeline 3.76 -> register prefetch 3.10 -> balanced roles 2.51 -> warp-specialized, wait moved after the
  issue 2.18 -> persistent 2.00.
- Bugs worth remembering: SW128 bf16 swizzle mask is 0x38 (cute's layout evaluation does not show the
  swizzle: it is applied to the byte address); a 16-element run is two swizzled 8-element chunks; gemm /
  umma_arrive need the whole warp.
- Chunk: State::MAX_CHUNK 512 -> 2048 (ring 640 -> 2176 slots, +137 MB), router RCNT 1024 -> 4096,
  k_units_big IPT 12 up to P = 12288 (units gate PASS incl. M 2048, mt 64).

8k prefill (ids8k, decode ids IDENTICAL to P7a in every arm):

| config | tok/s |
|---|---|
| P7a (k_kqm, chunk 512) | 291.1 |
| k_kqt Q2 only, chunk 512 | 328.1 |
| k_kqt Q2 + Q3, chunk 512 / 1024 / 2048 | 349.1 / 376.7 / 389.9 |

Quality (4 offsets x 1024 + 128 forced, every unit forced onto k_kqt with V41_KQT_MIN=1, Q2 + Q3): KL vs
exact 0.0082 / 0.0096 / 0.0300 / 0.0350 (default f8m 0.0097 / 0.0092 / 0.0303 / 0.0475), KL vs default at
the default-vs-exact level, NLL 1.6945 vs 1.6955 default / 1.7028 exact -> indistinguishable; default ON.

Engram I/O after the chunk 2048 step (nsys pf6): 6.6 s of the 21 s were idle, 4 x ~1.55 s blocked on the
Engram reap before each chunk (the forward waits on the per-layer flags anyway).
- `forward_chunk`: reap runs on a std::async thread while forward_layers is enqueued
  (V41_ENGRAM_OVERLAP=0 restores reap-then-forward). 8k prefill 389.9 -> **476.4 tok/s**.
- Continuous io_uring SQ refill (engram_store: backlog refilled as completions land instead of after the
  ring drains): gate_engram_store PASS byte for byte, 8k 474.1 tok/s = neutral once overlapped (kept: lower
  per-chunk I/O latency for shorter chunks).
- Decode ids IDENTICAL to P7a in both arms; decode after 8k 24.33 tok/s.

Stage 2b total: 8k prefill 291.1 -> 476.4 tok/s (1.64x), chat decode 25.1-25.3 tok/s.

Engram prefetch (nsys pf7 after the overlap: 4 x 0.63 s still stalled before k_engram_dequant -- the
forward reaches layer 1 ~0.6 s into a chunk whose own ~1.2 s of reads it overlaps): a second EngramStore
(own ring, arena, flags; +48 MB pinned at chunk 1024) and `prefill_prefetch` reads chunk i+1's rows while
chunk i runs (V41_ENGRAM_PREFETCH=0 off). 8k prefill: chunk 2048 476.4 -> **503.8**, chunk 1024 458.0 ->
**502.5** tok/s, ids IDENTICAL both. Chunk 1024 (the default, 1.97 GiB state) now equals 2048.

## P8c: fused sparse attention (no Eb)

k_sam_gather materialized Eb[m][640][512] bf16 (655 KB a row) for K2R / K3P to read back. Two findings:
- the gather was issue-bound on `fp4f` (a `__constant__` table read with a per-lane index: 8 serialized
  lookups per item). fp4 x e4m3 has <= 6 significant bits, so the hardware e2m1x2 -> f16x2 cvt + HMUL2 is
  exact and gives the same bf16 bits (`sam_fp4x8`).
- k_sam_scores_fused / k_sam_out_fused build each bf16 tile in smem straight from the sources. K2F resolves
  every entry once into a source code (ring slot | SAM_WIN, or the compressed row) and writes the codes for
  K3F; a first version that re-resolved ent -> cidx -> pointer per tile made K3F 1.7x slower than K3P.
  Passes of SAM_FROWS = 256 rows (no Eb bound).
- gate_sam_row: 240/240 bit-identical (fused vs K2 + K3, M up to 300), gather vs pre-P7a gather 15/15.
  M = 256, ne = 640: K2 + K3 8.13, row form 4.80, fused **2.74 ms**.
- 8k prefill (chunk 1024): 502.5 -> **549.8 tok/s**, decode ids IDENTICAL to P7a; chat decode 25.15 / 25.24
  (fused off), chat ids identical, consistency PASS. V41_SAM_FUSED=0 = the row form.
