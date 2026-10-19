# V4.1-1017 vs 0731: what this fork changes and why

This directory is a copy of `~/deepseek-v4-flash-0731-cuda` at commit `8b476b9`. That commit is recorded in `docs/UPSTREAM_0731_COMMIT`, and its README is kept as `docs/UPSTREAM_0731_README.md`. The 0731 engine is still here, unchanged: `src/engine.cu`, `src/decode.cu` and `kernels/*.cu`. It is no longer built.

The V4.1 path is a separate set of kernels and source files:

- `kernels/v41/` (GEMV engine, dense, MoE, CSA2 attention, Engram)
- `src/v41_model.cu`, `src/v41_engine.cu`, `src/v41_decode.cu`
- `src/engram_store.cpp` and `src/engram_hash.cpp`

The 0731 kernels could not be reused. They are built around 128-token blocks, compression ratios 4 and 128, hash routing and a Hadamard rotation, and none of those exist in V4.1. Only the infrastructure carried over: the server, tokenizer, safetensors reader and SSE stream splitter.

All geometry below was read from `inference/config.json` and from all 48 shard headers at revision `1bae0e10` (`docs/hdrs/`). It was not taken from the model card.

## 1. Geometry

| | 0731 | V4.1-1017 |
|---|---|---|
| layers, dim | 43, 4096 | 40, 5120 |
| heads × head_dim | 64 × 512 | 64 × 512 (rope 64, nope 448) |
| q_lora / o_lora / o_groups | — | 1280 / 1024 / 8 |
| routed experts, top-k | 256, 6 | 384 (REAP: 192 or 128, discovered at load), 6 |
| router | softmax, hash routing on early layers | sqrtsoftplus. The bias selects experts and never weights them. `bias_vl` applies on image rows. Weights are renormalised, then ×1.5 |
| expert inter, clamp | 2048 | 2304, swiglu_limit 10 (up clamped to ±10, gate clamped to ≤ 10) |
| norm_eps | 1e-6 | 1e-20 |
| attention | per-layer compressor at ratio 4/128 | CSA2: KV sources {2,8,14,20}, index sources {2,8,…,36}, and reuse layers that read the source's caches |
| compress ratio | 4 / 128 | 0 (layers 0–1), 2 (layers 2–19), 1 (layers 20–39) |
| candidates | — | Layer 20 scores 2048 blocks of 8. Index layers > 20 only score inside the kept blocks (identity at ≤ 16,384 rows) |
| Engram | — | Layers 1 and 14: 24 n-gram rows/token, wkv 6144→25600, sigmoid(signed-sqrt) gate, image rows gated to 0 |

## 2. Encoder/decoder prefill: the exact replay span is 2414 rows, not 128

PLAN.md and `notes/thor_serving_speed.md` quote the paper: the decoder "runs only an SWA replay over the last n_win = 128 prompt tokens".

That is exact only if a decoder layer's window KV for the replayed rows could come from somewhere other than the previous decoder layer. It cannot. The window KV of layer L at row r is `wkv(norm(input_L[r]))`, and `input_L[r]` is layer L−1's output at row r. That output is itself an attention result over rows r−127..r of layer L−1.

So the rows a decoder layer must compute grow by 127 per layer going down the stack:

```
dec_span(39) = 128                       (the last row, and its window)
dec_span(L)  = dec_span(L+1) + 127       =>  dec_span(L) = 128 + (39 − L) · 127
dec_span(21) = 2414                      (layer 20 is the global-KV source: it runs over every row)
```

The compressed (global) path does not grow the span. Layers 21–39 attend layer 20's compressed rows and index keys, and those are computed for every prompt row because layer 20 runs in full.

`forward_layers(…, prompt_end, …)` skips decoder rows below `prompt_end − dec_span(L)`. The result is bit-identical to running every row, by the argument above. `tests/gate_v41_golden.cu` checks this against `model.py`, which runs every row. Prefill cost for decoder layers is therefore at most 2414 rows per layer, about 1.9% of a 128K prompt and 0.2% of a 1M prompt.

A 128-row replay would be an approximation of `model.py`. Nothing in this repo does it.

Two consequences for prefix reuse (`src/v41_engine.cu`):

- **Reuse floor.** Rows a decoder layer skipped have stale window KV at that layer. A later request may reuse the prefix only from position `n − 1 − dec_margin` upward, where `n` is the end of the last skipping prefill. Below that it re-prefills from 0. `dec_margin` (default 256, env `V41_DEC_MARGIN`) widens every decoder span, so a conversation that edits its last turn still reuses.
- **Even reuse lengths.** The ratio-2 compressors carry a two-slot state that is self-contained only at a pair boundary, so the reused length is rounded down to even.

## 3. Caches at 1M context

Every value the reference writes into its compressed-KV and index-key buffers has just gone through `fp4_act_quant` in place. It is an e2m1 code times a scale. The code and the scale are therefore stored, not the bf16 product. This is exact, not an approximation:

| | per row | rows at 1M | packed (this server) | reference bf16 |
|---|---|---|---|---|
| compressed KV, ratio 2 (sources 2, 8, 14) | 256 B codes + 32 B E4M3 scales (per 16) | 3 × 524,289 | 453 MB | 1.61 GB |
| index keys, ratio 2 | 64 B + 4 B E8M0 (per 32) | 3 × 524,289 | 107 MB | 403 MB |
| compressed KV, ratio 1 (source 20) | 288 B | 1,048,577 | 302 MB | 1.07 GB |
| index keys, ratio 1 | 68 B | 1,048,577 | 71 MB | 268 MB |
| **total** | | | **0.93 GB (0.87 GiB)** | 3.36 GB |

The packed total matches the paper's 890 B/token, 0.83 GiB at 1M.

Fixed costs on top of that:

- window rings: 40 × 256 slots × 512 × fp32 = 21 MB
- indexer score scratch: `score_budget`, default 1 GiB, which is 256 rows at 1M
- candidate mask: `max_rows × 131,072` B (17 MB at 128 rows)
- `alloc_state` prints the total

**1M context costs about 2 GiB of state.** The weights decide whether the model fits, not the context.

## 4. Numerics that are easy to get wrong (each one is implemented as the reference does it)

- `act_quant` blocks are **32** wide with a pow2 (ue8m0) scale and amax clamped at 1e-4. `ops/lib/torch_kernels.act_quant` defaults to 128; the model passes `fp8_block_size = 32`.
- Index scores are `bf16(Σ_h bf16(relu(bf16(q·k)) · w_h))`, which matches the reference's bf16 einsum, `relu_`, multiply and sum. `weights_proj` is rounded twice: `bf16(bf16(proj) · 128^-½ · 32^-½)`.
- RoPE angles are computed per use as `fp32(pos × f)`, with `f` built in fp32 the way `precompute_freqs_cis` builds it, then `sincosf`. There are two frequency sets: base 10000 without YaRN for ratio-0 layers, and base 160000 with YaRN ×16 for compressed layers. A compressed row stands for the first token of its group, so it uses `pos + 1 − ratio`.
- The HC pre-mix is deferred. Attention collapses with the previous FFN's mix, the FFN collapses with this layer's attention mix, and the head collapses with the last FFN's mix. There is no `hc_head_fn`.
- Routed experts:
  - **Order.** They are summed per token in ascending expert order, which is the reference's loop order.
  - **Output rounding.** Each expert output is rounded to bf16, and so is the shared expert's.
  - **Final add.** The shared expert is added last.
- Top-k uses a radix select (11/11/10 bits). Ties go to the lowest index and the result is emitted ascending. `torch.topk` may break exact ties differently, which can only matter on bit-equal scores.
- Thor-build experts are decoded to the exact fp32 value of the ggml block. `ops/lib/replay.py` (the s10 evaluator) first rounds them to bf16. The difference is below bf16 resolution per weight, and the golden gate on a thor arm sees it as a slightly lower cosine. It is not a bug on either side.
