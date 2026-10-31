# DeepSeek-V4.1-Flash CUDA server for Jetson AGX Thor

This is a pure-CUDA inference engine and OpenAI-compatible server for [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017), pruned with REAP to 192 of 384 routed experts per layer. It runs on one NVIDIA Jetson AGX Thor (sm_110a, 128 GB unified memory): one binary, no Python on the request path. The two 101 GB Engram tables are read from NVMe for every token.

| | |
|---|---|
| Weights | [`patrickbdevaney/DeepSeek-V4.1-Flash-1017-REAP192-Thor`](https://huggingface.co/patrickbdevaney/DeepSeek-V4.1-Flash-1017-REAP192-Thor): the `thor-v1` build, 2.83 bpw K-quant experts + NVFP4 dense overlay |
| Pipeline (pruning, quantization, gates, evaluation) | [`patrickbdevaney/deepseek-v4.1-flash-reap-thor`](https://github.com/patrickbdevaney/deepseek-v4.1-flash-reap-thor) |
| llama.cpp alternative (GGUF, vision, DSpark) | branch [`deepseek-v4.1-reap-thor`](https://github.com/patrickbdevaney/llama.cpp/tree/deepseek-v4.1-reap-thor) of [`patrickbdevaney/llama.cpp`](https://github.com/patrickbdevaney/llama.cpp) + [`…-REAP192-GGUF`](https://huggingface.co/patrickbdevaney/DeepSeek-V4.1-Flash-1017-REAP192-GGUF) |

## Headline numbers

Measured 2026-10-30 on a Jetson AGX Thor (CUDA 13.0, Linux 6.8.12-tegra). The server was launched with `./start.sh` at its defaults and queried over `/v1/chat/completions`, with greedy sampling and one stream.

| | this server (`v41_server`) | llama.cpp fork, GGUF L3 (same box, same prompts) |
|---|---:|---:|
| **decode** (256-token chat answers: code / explanation / math) | **33.1 / 25.1 / 40.2 tok/s** (DSpark K=3); 25.3 with `SPEC=0` | 12.3–12.5 tok/s; 17.3–20.7 with `DSPARK=1` |
| **prefill**, 1.8K-token prompt | **398 tok/s** | 159 tok/s |
| **prefill**, 7.2K-token prompt (5.9K for llama.cpp) | **739 tok/s** | 147 tok/s |
| memory left while serving | 9 GiB of 122 (32K context; 17 GiB with `SPEC=0`) | 25 GiB (16K context) |
| image input | yes (the checkpoint's own vision tower, in CUDA) | yes (mmproj) |

- **Speculative decoding is on by default** (`SPEC=3`: the checkpoint's own DSpark drafter proposes 3 tokens, the model verifies them in one pass). Greedy output is token-identical to plain decoding (0 divergences on 27 held-out prompts), and sampled output keeps the model's exact distribution. On the held-out set it averages 1.345× over plain decoding: chat 1.09×, agentic 1.23×, code 1.40×, long context 1.40×, math 1.63×. No prompt type is slower.
- **Draft length per round adapts to the drafter's confidence** (S9): after drafting K tokens the engine verifies only the prefix with the best expected tokens per millisecond. On the held-out set the mean speedup rises from 1.30× to 1.32×, and no prompt is slower than plain decoding (the worst went from 0.96× to 1.06×).
- **Sampling runs at greedy speed.** At temperature 0.7 it decodes at 27.1 / 22.7 / 34.9 tok/s on the same three prompts with speculation, and at 25.0–25.2 without (25.3 greedy). Sampled rows use a two-stage head: NVFP4 logits for the whole vocabulary, recomputed exactly in bf16 for every token within T·ln(10⁹) + 4 logits of the maximum, with a full-head fallback. The sampling distribution differs from the full bf16 head by a total variation of at most 2×10⁻⁹ (331 rows measured). With `top_p` = 1 the draw is a Gumbel-max on the GPU; with `top_p` < 1 the nucleus is sorted on the host.
- **Decode** without speculation is at about 74% of the reachable bandwidth roofline: about 7.6 GB per token in 40 ms, against a 31 ms floor at 247 GB/s. That ceiling is measured with TMA bulk copies (`tools/bw_sweep.cu`); plain loads top out at 227–243 GB/s. The per-kernel breakdown is in the pipeline repo's [`docs/zenith/`](https://github.com/patrickbdevaney/deepseek-v4.1-flash-reap-thor/tree/main/docs/zenith).
- **Prefill.** `v41_decode` reaches 580.7 tok/s on the fixed 8,192-token benchmark prompt. The server's 739 tok/s is for a different, more regular prompt. Every prefill change in the history below kept the greedy token ids identical to the previous build.
- **Suffix drafting** (S8, on by default, at most 7 tokens a round). Each round looks up the longest earlier match of the last three tokens in the context and proposes what followed it, when that is expected to beat the DSpark round or a plain step in tokens per millisecond. It costs nothing when there is no match. On edit-style prompts it lifts plain decoding from 25.8 to 35.2 tok/s and DSpark K=3 from 43.9 to 48.8; on long-context prompts, 36.9 to 44.8 with K=3; other domains are within 1%. Every round width up to 7 is exact against plain decoding in forced-width tests. An 8-draft round needs 9 verify rows, which leaves the kernels that match single-token decoding bit for bit, so `V41_SFX_K` is capped at 7.
- **Session cache** (C1, on by default). At the end of every text request the engine keeps a 2 GiB-budget snapshot of the attention state (KV-source caches, compressor state, the last 512 window positions, drafter rings). A later request that extends an earlier session restores it even after another session ran in between: a 3.5K-token follow-up prefills in 165 ms instead of 7.65 s, with identical ids. `V41_C1_DIR=dir` adds an NVMe tier for snapshots evicted from RAM (`V41_C1_DISK_GB`, default 64), which also survives a restart: the same follow-up restored from disk prefills in 160 ms, in-process or after restarting the engine.
- **Greedy output** is deterministic run to run. It is also identical between the 64- and 1024-row prefill chunks (all six bench prompts were checked on 2026-10-30).

## Quick start

```bash
# 1. weights (~110 GiB) and the tokenizer
hf download patrickbdevaney/DeepSeek-V4.1-Flash-1017-REAP192-Thor --local-dir ~/models/DeepSeek-V4.1-Flash-1017-REAP192-Thor

# 2. Engram row files (two ~101 GB files, local NVMe only; shared with the llama.cpp build)
#    built once from base-model shards 47/48 with ops/s09_engram_rowfile.py in the pipeline repo:
#    https://github.com/patrickbdevaney/deepseek-v4.1-flash-reap-thor/blob/main/llamacpp/ENGRAM_ROWFILE.md
#    put them (or a symlink) at ~/models/DeepSeek-V4.1-Flash-1017-REAP192-Thor/engram_rows, or set ENGRAM_DIR

# 3. build (CUDA 13, sm_110a, liburing) and run
scripts/build_v41.sh v41_server
./start.sh                                   # http://127.0.0.1:8000, 32K context, NVFP4 overlay on
```

`start.sh` checks the shards, tokenizer, keep-set and row files. It refuses row files on removable media. It also refuses to start beside another model-loading process, because on Thor's unified memory two such processes can hard-reboot the box. If less than about 108 GiB is free, it drops the page cache before loading. Loading takes about 70 s.

| variable | default | |
|---|---|---|
| `CKPT` | `~/models/DeepSeek-V4.1-Flash-1017-REAP192-Thor` | the HF download |
| `ENGRAM_DIR` | `$CKPT/engram_rows` | `engram_L1.v41rows`, `engram_L14.v41rows` |
| `SEQMAX` | `32768` | context ceiling. It sizes the KV caches, which are small: a 1M-token state is 1.66 GiB |
| `OVERLAY` | `1` | serve the dense weights from `$CKPT/overlay_nvfp4` (`DSV4_NVFP4_OVERLAY`). `0` uses FP8/bf16 and is slower |
| `HOST`, `PORT` | `127.0.0.1`, `8000` | |
| `EXTRA` | | further `v41_server` flags, e.g. `--no-graph` |

### API

```bash
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":256}'
```

- `GET /` serves a self-contained web UI.
- `/health`, `/metrics` (Prometheus format) and `/v1/models` are available.
- `POST /v1/chat/completions` supports streaming and non-streaming, tool calls in DeepSeek's DSML format, and thinking blocks (`reasoning_content`).
- Images: `{"type":"image_url","image_url":{"url":"data:image/...;base64,..."}}` parts in user messages (JPEG/PNG/BMP/GIF). `http(s)` URLs are refused; local paths only with `V41_IMAGE_FILES=1`. The vision tower (32-layer ViT + aligner, 1.4 GiB) runs on tensor cores; a 640x488 page costs about 0.8 s including its prefill, and encoder outputs are cached by image hash (`V41_VCACHE`, 8 images).
- Constrained output: `response_format` `{"type":"json_object"}` or `{"type":"json_schema","json_schema":{"schema":{...}}}`, or the `grammar` extension `{"type":"lark"|"regex"|"json","data":...}` ([llguidance](https://github.com/guidance-ai/llguidance) syntax). In thinking mode the constraint starts after `</think>`, so give the request enough `max_tokens` (or `reasoning_effort: "low"`) to finish reasoning. Masks cost about 8 µs a token; a constrained request decodes without speculation. A malformed grammar is a 400.
- A non-streaming request whose client disconnects stops generating at the next token.
- `POST /v1/completions` takes a raw prompt.
- Requests are served one at a time; the queue depth is in `/metrics`.

`build/v41_decode` is the single-prompt benchmark and debugging driver: `--chat`, `--ids-file`, `--repeat`, `--consistency`, `--score` (teacher-forced), `--json`.

## Correctness

- **Golden gate** (`tests/gate_v41_golden.cu`): per-layer cosine, isolated and chained, plus teacher-forced next-token agreement against the reference `model.py`. Every hidden-state check passed. The registered near-tie rule (at least 90% of disagreements must be near-ties) missed: 89.1% on this build and 80.5% on the source checkpoint. The owner amendments and per-row diagnostics are in the pipeline repo's [`ops/decisions/`](https://github.com/patrickbdevaney/deepseek-v4.1-flash-reap-thor/tree/main/ops/decisions).
- **CPU gates.**
  - The chat encoder is byte-exact with the reference `encoding.py` on 5 golden vectors and 400 fuzz cases.
  - The tokenizer matches HF `tokenizers` on 2,116 cases.
  - Engram n-gram hashing matches the reference on 18,096 ids.
  - The K-quant decoders give 0 mismatches against `ggml`.
  - The OpenAI and SSE mapping passes 31 + 11 cases.
- **Determinism.** Every reduction has a fixed order and there are no atomics in any sum. Greedy decode is checked with `--repeat`. Graph and no-graph runs give the same token ids. Decode is checked against prefill with `--consistency`.
- **Every speed change** is accepted only if the greedy ids are identical to the previous build (exact tier). A change that only reorders a summation must also match on teacher-forced NLL and KL against the true text. The A/B harness is in the pipeline repo's `ops/bench/`.
- **Long context.** A 1,048,576-token state allocation, and decode after a 64K-token prefill, passed the s08 `ctx1m` gate on 2026-10-25, on the older, slower engine. The default here is 32K.

## What is in it

| addition | where |
|---|---|
| Engram row store | `src/engram_store.cpp`. Reads are io_uring O_DIRECT, aligned 1 KiB per row, into a `cudaHostRegister`ed arena. Completion is published through flag words that the captured decode graph waits on (`cuStreamWaitValue32`), so decode stays one graph launch per token. Hashing is in `src/engram_hash.cpp`; the gate+wkv kernel is in `kernels/v41/v41_engram.cu`. |
| CSA2 attention | KV sources {2,8,14,20}, index sources {2,8,…,36}, and reuse layers that read the source's caches. Layer 20 picks 2048 candidate blocks × 8 past 16,384 rows. The KV is packed FP4 (e2m1 + E4M3/E8M0 scales), which is exact to the reference. |
| encoder/decoder prefill | Decoder layers skip rows below `prompt_end − dec_span(L)`, which is exact. Prefix reuse respects the resulting floor. |
| router | sqrtsoftplus; the bias selects but does not weight; `bias_vl`; ×1.5; expert count read from the checkpoint |
| routed experts | thor-v1 Q2_K / Q3_K (bit-exact ggml decode). Decode is a GEMV with one warp per row, reusing x across an expert pair. Prefill uses grouped tensor-core GEMMs (`kernels/v41/v41_kqt.cuh`, `kernels/v41/v41_tc5.cu`). |
| dense path | The NVFP4 overlay is served on FP8 tensor cores for M > 1 and a streaming GEMV for M = 1. The FP8 and bf16 remainder is read once per token. A two-stage greedy head runs an NVFP4 shortlist, then an exact bf16 rescore. |
| tcgen05 prefill | sm_110a UMMA/tcgen05 GEMMs for the large prefill shapes ([`docs/zenith/P8_TCGEN05.md`](https://github.com/patrickbdevaney/deepseek-v4.1-flash-reap-thor/blob/main/docs/zenith/P8_TCGEN05.md) in the pipeline repo) |
| DSML | `include/encoding_dsv41.h`: a line-for-line port of V4.1 `encoding/encoding.py` |

### Prefill history (8,192-token benchmark prompt, `v41_decode`)

| build | tok/s |
|---|---:|
| first working forward, 2026-10-25 (64K prompt) | 35.9 |
| P7a: grouped K-quant tensor-core GEMM, chunk 512 | 291.1 |
| without TC5 (`V41_TC5=0`) | 335.7 |
| P8: tcgen05, chunk 1024 | 549.8 |
| P8 + vectorized activation/output rounding (current) | **580.7** |

## Limits

- **Images and the session cache.** Requests with images are not snapshotted by C1 (snapshots are keyed by token ids); they still reuse the resident prefix, and re-sent images hit the encoder cache.
- **One stream.** One request at a time, by design: a bandwidth-bound decode at batch 1 has nothing to gain from batching on this box.
- **Thor only.** The kernels target sm_110a. Some paths, tcgen05 for example, do not exist on sm_120/121.

## Build

```bash
scripts/build_v41.sh            # everything: v41_decode, v41_server, gates, benches
scripts/build_v41.sh v41_server # one target
```

The build needs CUDA 13 (sm_110a), liburing and `-lcuda` (for stream memops). The engine started as a copy of the DeepSeek-V4-Flash-0731 engine ([`patrickbdevaney/deepseek-v4-flash-0731-cuda`](https://github.com/patrickbdevaney/deepseek-v4-flash-0731-cuda)); what changed for V4.1 is in `docs/V41_DELTA.md`. Expert formats are in `FORMAT_PROPOSAL.md`.

## License

The code is MIT. `ref/` contains DeepSeek's MIT-licensed reference implementation, used by the gates. The model weights carry the base model's MIT license.
