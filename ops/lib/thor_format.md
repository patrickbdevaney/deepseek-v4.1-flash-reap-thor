# Thor build format (`thor-v1`)

The s07 Thor build is the single on-disk handoff from the quantization stage to both runtimes:

- the s08 CUDA server, which mmaps the files directly;
- the s09 GGUF writer, which copies the expert bytes into GGUF tensors without re-quantizing.

The format is written by `ops/s07_quant.py` (`thor` for arm A, `thor-b` for arm B) through `ops/lib/thor_format.py`, and read by `ops/lib/replay.py` for the s10 evaluation.

## Directory

```
$V41_WORK/thor/<arm><n>/          e.g. thor/A192 (requantized), thor/B128 (bit-exact MXFP4)
  thor.json                       manifest (below)
  layer-00.safetensors … layer-39.safetensors   one decoder layer each
  globals.safetensors             embed.weight, head.weight, norm.weight, image_start/_end/_newline
  vision.safetensors              vision.* (ViT) and aligner.*, unchanged
  mtp-0.safetensors … mtp-2.safetensors         DSpark/MTP layers, unchanged (128 experts, not pruned)
$V41_WORK/thor/overlay_nvfp4/     optional NVFP4 dense overlay (separate gate), see below
```

The Engram tables are **not** in the build:
- `layers.{1,14}.engram.embed.weight` holds 384M rows × 256 B (FP8).
- `.scale` holds 384M rows × 8 B (E8M0, one per 32 values).

`thor.json["engram_tables"]` gives the base repo, revision, file, tensor names and sha256 of the two source shards the row store is built from. The small Engram tensors (`q_weight`, `k_weight`, `wkv.weight/scale`) **are** included, in `layer-01` and `layer-14`.

## Files

Every file is a valid safetensors file with two extra guarantees:

- **Alignment.** Every tensor starts at a multiple of **256 bytes from the start of the file**. The header is space-padded, and gaps are filled by explicit padding tensors named `__pad.<n>` (dtype U8). The format forbids holes, so padding has to be real tensors. **Consumers must skip every tensor whose name starts with `__pad.`.** This prevents the 4-byte-alignment trap that broke vectorised loads in the GLM CUDA server.
- **Metadata.** `__metadata__` holds `format = "thor-v1"`, plus `layer`, `keepset_sha256` and (arm A) `heal_gain_folded` and `calib_tokens_per_expert` (JSON) on layer files.

## Tensor names and dtypes

Names are the reference `inference/model.py` names (no prefix).

**Routed experts are renumbered** `0..n-1` in ascending order of their original id. `thor.json` → `keepset_sha256`, together with `s04/keepset.json`, gives the original ids.

### Dense tensors

These pass through **bit-exact** from the source shard.

| tensor | dtype | notes |
|---|---|---|
| `layers.L.attn.{wq_a,wq_b,wkv,wo_a,wo_b}.weight`, `…indexer.wq_b.weight`, `…ffn.shared_experts.w{1,2,3}.weight`, `…engram.wkv.weight` | F8_E4M3 | + `.scale` F8_E8M0, one scale per **32×32** block (`[ceil(out/32), ceil(in/32)]`) |
| norms, `compressor.*`, `indexer.{wk,weights_proj,k_norm}`, `engram.{q,k}_weight` | BF16 | |
| `hc_{attn,ffn}_{fn,base,scale}`, `attn.attn_sink` | F32 | |
| `layers.L.ffn.gate.weight` | BF16 `[n, 5120]` | rows of the kept experts, in keep order |
| `layers.L.ffn.gate.bias`, `…gate.bias_vl` | F32 `[n]` | same slice; `bias_vl` selects for image-span tokens |

`wo_a` stays FP8 here. The reference `convert.py` dequantises it to bf16 at load time, and consumers do the same.

### Routed experts

Each layer and projection is **stacked** into one tensor.

**Requantized (arm A):**

| tensor | dtype | stored shape | logical shape |
|---|---|---|---|
| `layers.L.ffn.experts.w1.qweight` | U8 | `[n, 2304, row_bytes(qtype, 5120)]` | `[n, 2304, 5120]` |
| `layers.L.ffn.experts.w3.qweight` | U8 | `[n, 2304, row_bytes(qtype, 5120)]` | `[n, 2304, 5120]` |
| `layers.L.ffn.experts.w2.qweight` | U8 | `[n, 5120, row_bytes(qtype, 2304)]` | `[n, 5120, 2304]` |

`qtype` comes from `thor.json → tensors[name].qtype`. `w1` and `w3` always share a qtype; `w2` may differ. Each output row is the concatenation of whole ggml blocks along the input dimension: `row_bytes = in / block_elems * block_bytes`. The blocks are **byte-identical to ggml's** `block_q2_K` / `block_q3_K` / `block_q4_K` (`ggml-common.h`).

| qtype | ggml type id | block | bytes | bpw | layout |
|---|---|---|---|---|---|
| Q2_K | 10 | 256 | 84 | 2.625 | `scales[16]` (low 4 bits scale, high 4 bits min) · `qs[64]` · `d` f16 · `dmin` f16 |
| Q3_K | 11 | 256 | 110 | 3.4375 | `hmask[32]` · `qs[64]` · `scales[12]` (6-bit, −32 offset) · `d` f16 |
| Q4_K | 12 | 256 | 144 | 4.5 | `d` f16 · `dmin` f16 · `scales[12]` (6-bit scale + min) · `qs[128]` |

The layouts are verified against llama.cpp's `gguf-py` `dequantize_blocks` in `ops/tests/test_post.py`. The CUDA server can port llama.cpp's MIT `dequantize_q*_K` / `vec_dot_q*_K` for exactly these bytes. **The s09 GGUF writer copies them verbatim:** a GGUF `ffn_{gate,up,down}_exps` tensor with ne = `[in, out, n]` has the same byte order as `[n, out, row_bytes]`.

Q5_K, Q6_K and the IQ types are reserved. s07 v1 does not produce them, and a reader must refuse an unknown qtype. The trellis types KT2 / KT3 are defined below; only the `thor-trellis` sub-arm produces them.

**Bit-exact MXFP4 (arm B):** `qtype = "MXFP4_SRC"`.

| tensor | dtype | shape |
|---|---|---|
| `layers.L.ffn.experts.wX.weight` | U8 | `[n, out, in/2]`. Byte k holds element 2k in the **low** nibble and 2k+1 in the high nibble. Nibble = E2M1 code: bit 3 = sign, values `0, .5, 1, 1.5, 2, 3, 4, 6`. |
| `layers.L.ffn.experts.wX.scale` | F8_E8M0 | `[n, out, in/32]`, value 2^(e−127) |

This is the source checkpoint's own layout, so the CUDA server's existing hardware FP4 GEMV consumes it unchanged. ggml's `block_mxfp4` stores the same codes and exponent byte in a different nibble order: element i is the low nibble and element i+16 the high nibble of byte i, within each 32-block, with the `e` byte first. The s09 writer reorders losslessly with `ggml_quants.src_to_ggml_mxfp4`. The low-nibble-first assumption about the source is checked by s01 against the TileLang kernel.

### Trellis experts (KT2 / KT3, arm A sub-arm `A<n>kt`)

Written by `s07_quant.py thor-trellis` into `$V41_WORK/thor/A<n>kt`. It is adopted for release only through the s10 `trellis` panel (`s10_gates.json` → `trellis`). **No ggml type carries these bytes.** A KT build is CUDA-server-only, and the GGUF ladder is unaffected because s09 quantizes from the MXFP4 master.

| tensor | dtype | stored shape |
|---|---|---|
| `layers.L.ffn.experts.wX.qweight` | U8 | `[n, out, in/128 × 16·b]` |
| `layers.L.ffn.experts.wX.kt_scale` | F16 | `[n, out, in/128]` |
| `layers.L.ffn.experts.w13.rht_sign` | U8 | `[DIM/8]`, shared by w1 and w3 |
| `layers.L.ffn.experts.w2.rht_sign` | U8 | `[INTER/8]` |

`b` is 2 for KT2 (2.125 bpw with scales) and 3 for KT3 (3.125 bpw). `w1`/`w3` share a qtype, and `w2` may differ, as for the K-quants.

**Rotation.** Logical weight rows are rotated along the input dimension: `w' = R w`, with `R = blockdiag(H₂₅₆ · diag(sign)) / 16`, where `H₂₅₆` is the Sylvester Hadamard matrix. Bit k of byte j of `rht_sign` is 1 iff element 8j+k is negative. Consumers either rotate the activations once per pass (`x' = R x`, so `W' x' = W x`; the CUDA server's `k_rht_rows`) or un-rotate the decoded weights (`w = Rᵀ w'`; `kt.dequant`). The reference butterfly order is `ops/lib/kt.py::rht_rows`: stages h = 1, 2, …, 128, with pair (j, j+h) → (a+b, a−b), then ×1/16.

**Blocks.** A rotated row is cut into 128-weight blocks. Each block is a circular stream of 128·b bits stored as 4·b little-endian u32 words, in which stream bit k is bit k%32 of word k/32. Weight i's code is the **12-bit window** starting at stream bit i·b (mod 128·b).

**Value.** For window `s`, compute `x = s·89226354 + 64248484 (mod 2³²)` and `m = (x & 0x8fff8fff) ^ 0x3b603b60`. The value is `fp32(fp16(m & 0xffff)) + fp32(fp16(m >> 16))` (QTIP's 3INST codebook) times `fp32(kt_scale)`, with separate roundings. The decoders `DecKT<2|3>` in `cuda/kernels/v41/v41_gemv.cuh` are held bit-exact to `kt.dequant` on the host (`gate_kt_host`) and on the device (`gate_v41_units`).

**Encoder.** Block LDLQ over 128-column groups of the rotated weights, with the rotated Hessian `R H Rᵀ` from the same calibration tokens GPTQ uses. Each block is encoded by a tail-biting Viterbi over the 4096 window states (`cuda/kernels/v41/kt_viterbi.cu`, states identical to `kt.viterbi`), then the per-block scale is refit by weighted least squares and stored as fp16. `thor.json["kt"]` records the parameters.

### Routing semantics consumers must reproduce

These are unchanged from the reference:

- `scores = sqrtsoftplus(gate.weight · x)`.
- The selection key is `scores + bias`, with `bias_vl` on image-span tokens. Take the top-6 of that key.
- The routing weights are `scores` at the selected experts, re-normalised to sum 1, then ×`route_scale` (1.5).
- The routing weight multiplies the SwiGLU output **before** `w2` (`Expert.forward`). Clamps: `up` to [−10, 10] and `gate` ≤ 10.
- The shared expert is added unweighted.

### Heal gain

The heal gain is one scalar per layer on the routed-expert sum (s06, adopted only if the s10 heal gate passes):

- **Arm A:** already **folded into `w2`** before requantization. `thor.json["heal"]` records the gains and `"folded_into": "w2 before GPTQ"`. Consumers do nothing.
- **Arm B:** the build is bit-exact and unhealed. The name `layers.L.ffn.heal_gain` (F32 `[1]`) is **reserved**. When present, multiply the routed sum by it; absent means 1.0. GGUF cannot express a per-layer routed scale, so an s09 writer that meets it must stop rather than drop it.

## `thor.json`

```json
{
  "format": "thor-v1",
  "arm": "A",
  "keepset_sha256": "<sha256 of the canonical keep-set, ops/lib/keepset.py>",
  "n_routed_experts": 192,
  "source_revision": "1bae0e1088993b9b82de08f59de5be5a34e42170",
  "heal": {"gains": {"0": 0.93}, "folded_into": "w2 before GPTQ"},
  "alloc": {"0": {"w13": "Q3_K", "w2": "Q3_K"}, "3": {"w13": "Q2_K", "w2": "Q3_K"}},
  "protected": [0, 1, 2, 8, 14, 20, 37, 38, 39],
  "files": {"layer-00.safetensors": {"bytes": 0, "sha256": "…"}},
  "tensors": {"layers.0.ffn.experts.w1.qweight": {"file": "layer-00.safetensors", "dtype": "U8",
              "qtype": "Q3_K", "stored_shape": [192, 2304, 2200], "shape": [192, 2304, 5120]}},
  "engram_tables": {"base_repo": "deepseek-ai/DeepSeek-V4.1-Flash-1017", "revision": "…",
                    "tables": [{"layer": 1, "file": "model-00047-of-00048.safetensors",
                                "tensor": "layers.1.engram.embed.weight", "scale": "layers.1.engram.embed.scale",
                                "sha256": "…", "size": 0}]},
  "expert_bytes": 0,
  "expert_bpw": 2.83,
  "complete": true
}
```

The numbers in this example illustrate the schema only. They are not results.

### Consumer obligations

1. Refuse a build with `complete != true`, an unknown `format`, or an unknown `qtype`.
2. Check `keepset_sha256` against the keep-set the consumer was configured for. **A mismatch is fatal**, never a warning (PLAN.md, GLM v3 defect 1).
3. Verify file sha256 against `files` at least once per build (`Build.verify()`).
4. Skip `__pad.*` tensors.

## NVFP4 dense overlay (`thor/overlay_nvfp4/`, optional)

`L<LL>.safetensors` holds one file per decoder layer. Each F8_E4M3 dense Linear of the layer (every `*.weight` except routed experts) is quantized round-to-nearest to NVFP4:

| tensor | dtype | shape |
|---|---|---|
| `<module>.weight` | U8 | `[out, in/2]`, low nibble = even element |
| `<module>.weight_scale` | F8_E4M3 | `[out, in/16]` |
| `<module>.weight_scale_2` | F32 | `[1]` (global) |

`<module>` is the path inside the block, for example `attn.wq_a` or `ffn.shared_experts.w1`. The dequantized value is `E2M1 × weight_scale × weight_scale_2`.

Consumers use the overlay **only** if `s10/gates.json["overlay_adopted"]` is true.
