# Expert weight formats: what the CUDA server reads, and a proposal for below Q2_K

**Status.** When this fork started, `ops/lib/thor_format.md` did not exist, so this file was meant to be the format. It exists now: `thor-v1`, written by `ops/s07_quant.py`. **thor-v1 is the format.** The CUDA server reads it as specified and refuses everything else, per obligations 1–4 in that file. Section 1 records how the server reads it. Sections 2–4 propose two sub-3-bit qtypes for the slots thor-v1 reserves. Their kernels exist and are benchmarked, but **no loader path accepts them** until the spec names them.

## 1. thor-v1, as consumed by `src/v41_model.cu`

| qtype | decoder (`kernels/v41/v41_gemv.cuh`) | bpw | gate |
|---|---|---|---|
| `Q2_K` (84 B / 256) | `DecKQ<2>` | 2.625 | bit-exact vs `ggml_quants.dequant`, host + device |
| `Q3_K` (110 B / 256) | `DecKQ<3>` | 3.4375 | bit-exact, host + device |
| `Q4_K` (144 B / 256) | `DecKQ<4>` | 4.5 | bit-exact, host + device |
| `MXFP4_SRC` (arm B) | `DecMXFP4<true>`, which uses `cvt.rn.f16x2.e2m1x2` (`__nv_cvt_fp4x2_to_halfraw2`); `<false>` is a bit-construction fallback | 4.25 | bit-exact vs `src_dequant`, both paths |

**Loading.**

- **Stacked tensors.** `layers.L.ffn.experts.wX.{qweight | weight, scale}` are stacked `[n, out, row_bytes]`. The loader builds per-expert pointer tables from them and checks that each stride equals `out × in/256 × block_bytes`, or `out × in/2` for MXFP4.
- **Formats per pass.** `w1`/`w3` share one format and `w2` may use another (`thor.json → tensors[…].qtype`). The MoE runs gate|up and down as two passes, and each pass dispatches its own decoder.
- **Refusals.** Any unknown qtype is refused. So is a build with `complete != true`, a `keepset_sha256` different from `--keepset`, or a router whose row count differs from the expert count.
- **Heal gain.** `layers.L.ffn.heal_gain` (arm B, reserved) multiplies the routed sum when present.

**Arithmetic.** A decoded weight is the ggml value computed exactly as numpy computes it: `(d·sc)·q − dmin·m`, rounded at each step with no FMA.

`tests/gate_kq_host.cu` runs the **same source** on the CPU against the numpy reference: 0 mismatches over 64×2304 for each type. That is the only check in this repo that has run on real numbers so far. The device run is `tests/gate_v41_units.cu`, in the chain.

**Known inefficiency (measured by `tools/gemv_format_bench.cu` in the chain, not yet run).**

- A K-quant lane decodes one 32-weight chunk, so the 8 lanes sharing a superblock each re-read its scales. L1 absorbs that, but the bytes still count against the load units.
- For Q2_K/Q3_K, four lanes read the same 32 `qs` bytes and keep different 2-bit planes.
- A layout-aware variant (one lane per 128 weights, or a warp-cooperative superblock) is the obvious next step once the bench shows how far the simple form is from the streaming roofline.

## 2. Proposed qtype `LUT2` / `LUT3` / `LUT4`: per-tensor codebook, in-register

**Problem.** Q2_K spends 0.625 bpw on metadata: 16-element sub-blocks, each with a 4-bit scale and a 4-bit min. At 192 experts that metadata is about 6 GiB of the expert budget.

**Layout.** One row of `in` weights is `in/32` chunk records. Each chunk record is `4·b` bytes:

- **b = 2:** two u32 words of 2-bit codes; weight i is `(word[i/16] >> 2(i%16)) & 3`.
- **b = 3:** the same two words for the low 2 bits, then one u32 of high bits; weight i's high bit is bit i of the third word.
- **b = 4:** four u32 words of nibbles.

The record is followed by one E4M3 scale per chunk (`[out, in/32]`), and one fp16 codebook of 2^b entries per (expert, projection).

**Size.** b + 0.25 bpw: LUT2 = 2.25, LUT3 = 3.25.

**Value.** `codebook[code] × e4m3(scale)`. The codebook is fitted per tensor (k-means over scale-normalised weights, weighted by the imatrix).

**Kernel.** `DecLUT<B>`:

- Each lane holds one codebook entry in a register (16 lanes cover b ≤ 4).
- A weight is decoded with one `__shfl_sync(lut, code)`.
- There are no shared-memory or constant-cache reads per weight, so lanes never contend on a table.
- The two tensors of the gate|up pass carry two codebooks (`lut0`/`lut1`).

**Fit for the 192-expert arm.** LUT3 on gate/up plus LUT4 on down is about 3.4 bpw, with a learned codebook instead of a uniform grid. The 4-bit codebook replaces Q4_K's 0.5 bpw of sub-block metadata.

## 3. Proposed qtype `KT2` / `KT3`: QTIP-style trellis ("3INST")

**Layout.**

- A tail-biting bit-shift trellis over blocks of 128 weights.
- Each block is `4·b` u32 words, with one fp16 scale per block (`[out, in/128]`).
- Weight i of a block reads the 16-bit window starting at bit `i·b`, modulo `128·b`.
- The window maps to a value by the computed codebook `x = s·89226354 + 64248484; h = (x & 0x8fff8fff) ^ 0x3b603b60`, read as two fp16 values and summed.

**Size.** b + 0.125 bpw: KT2 = 2.125, KT3 = 3.125.

**Why the chunking still works.** A 32-weight chunk q of a block starts exactly at word `q·b`. A lane therefore loads four consecutive words (wrapping at the block end), and every funnel-shift amount is a compile-time constant (`DecKT<B>`).

**Why.** QTIP reports trellis codes at 2 bpw matching scalar 3-bit quality. For an arm forced down to about 2.2 bpw, that is the difference between usable and not. The encoder (Viterbi over the trellis, about 1 GPU-hour per 100B parameters in QTIP) **is not written**. `tools/lowbit_ref.py`, the planned Python encoder/decoder pair, was not written in this session. The kernel is only benchmarked for speed.

## 4. What adopting either would take

1. **Spec.** Reserve the names in `thor_format.md` and add byte layouts identical to sections 2–3.
2. **Encoders.** s07 encoders, plus a numpy decoder in `ggml_quants.py`. The unit gate then holds the CUDA decoders to it bit-exactly, the same way the K-quants are held today.
3. **GGUF.** Neither format is a ggml type. The s09 GGUF writer could not copy the bytes, which breaks thor-v1's "one handoff, two runtimes" property. Either llama.cpp gets matching types (IQ2_XXS / IQ3_XXS are the nearest relatives, and their E8 lattice codebooks are a different design), or these become CUDA-server-only builds with their own keep-set fingerprint. The second option is cheaper and honest about the split.
4. **Evidence.** `gemv_format_bench` must first show the trellis decode is not slower per byte than Q2_K's. At about 6 integer ops per weight it may not be, on Thor's 117 GB/s budget.
