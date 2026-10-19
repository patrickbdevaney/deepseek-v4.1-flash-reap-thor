# ops/: the V4.1-Flash-1017 REAP pipeline, stage by stage

This directory holds the code that turns [`pipeline-plan/PLAN.md`](../pipeline-plan/PLAN.md) into runs.

**Status (2026-10-19):**
- Stages s00-s03, the drift guard and the shared library are **written but not yet run**.
- The checks done so far are:
  - `py_compile` on every file;
  - the 16 numpy-only tests in `tests/test_pure.py`.
- Nothing has loaded the model or touched the GPU, because a leafllm vLLM teacher was serving at the time. Every number below about speed or memory is still an estimate.
- **s01 is the first thing to run once the GPU is free.**

## Operating rules (carried from GLM-5.3 / MiMo; not optional)

**Never co-run.** Never start a GPU stage beside a vLLM or llama.cpp server. Thor's unified memory hard-rebooted the board once and cost 12 hours. `env.assert_no_gpu_server()` refuses to start in that case.

**Detach every long stage.** Run it under `setsid nohup … &` or systemd, never as a child of an interactive shell. An SSH drop once cost 16 hours.

**One decoder layer per process.** A process exit does not return the Tegra nvmap pool; only `drop_caches 3` does. The s03 orchestrator therefore runs each unit as its own process and then calls `env.reclaim(expect_gib=…)`. If the reclaim does not restore the floor, that is a **failure**, not a warning.

**Memory admission inside the process.** `lib/memfence.py` (vendored) is called with each row's predicted peak before the row is allocated.

**Ledgers everywhere.**
- Every resumable unit is in a JSON ledger written with tmp + fsync + rename.
- Outputs are written before the ledger says "done".
- s03 adds a journal protocol so that no row is counted twice or lost (see below).

**Keep-set fingerprints are fatal.** Any keep-set or fingerprint mismatch in s04 and later stages raises; it is never downgraded to a WARN. That is the GLM v3 defect, recorded in PLAN.md.

**Read-only neighbours.** `~/deepseek-v4-flash-0731-cuda` and `~/glm-5.3-reap` are read-only. Code from them is **copied** here and labelled "vendored" with its origin.

## Files

| file | what it is |
|---|---|
| `s00_fetch.sh` | Pinned-revision download plus sha256 verification of all 48 shards. |
| `s01_layer_probe.py` | Runs layer 3 in reference numerics on sm_110a and compares it with a CPU fp32 oracle. Reports tok/s and peak memory. |
| `corpus_spec.py` | The calibration mixture, sources, provenance and licence policy, and interleaving. |
| `s02_calib_corpus.py` | Builds the corpus. Has `--dry-run` (list only, offline) and `--probe` (live licence and yield). |
| `s03_saliency.py` | Layer-major saliency pass. Units: `hash`, `engram`, `embed`, `layer L`, `head`, and the `run` orchestrator. |
| `drift_guard.py` | Position-binned routing-drift test and long-context mask guard. `protect` runs before selection, `check` runs after. |
| `lib/env.py` | Paths, logging, `Ledger`, `reclaim`, the no-server guard, and the reference-module import. |
| `lib/shards.py` | Reads shards and builds one reference `Block` at a time, applying convert.py's mp=1 transforms. Also holds the Engram `RowSource`. |
| `lib/torch_kernels.py` | Pure-torch versions of the six TileLang kernels. It is both the CPU oracle and the fallback. |
| `lib/hooks.py` | Instrumented MoE (all statistics), query-chunked Indexer with candidate persistence, and token-chunked Block. |
| `lib/engram_rows.py` | safetensors memmap and a sequential windowed gather of Engram rows (no torch). |
| `lib/drift.py` | KQ4 drift statistics, row-cluster bootstrap, sink and rising-expert guard (numpy only). |
| `lib/imatrix.py` | Writer and reader for the llama.cpp legacy `.dat` imatrix, built from the s03 bank with keep-set remapping. |
| `lib/hope.py`, `lib/memfence.py`, `lib/imatrix_verify.py` | Vendored from `~/glm-5.3-reap/scripts`. The only change: `hope.py` imports torch lazily. |
| `tests/test_pure.py` | numpy-only tests: the HOPE solver against brute force, imatrix round-trip and verify, Engram gathers, ledger, corpus rules, s03 state topology, drift guard (true and false positives), and the tokenizer shim. |

## How to run (in order, once the GPU is free)

```bash
cd ~/deepseek-v4.1-flash-reap-thor
python3 ops/tests/test_pure.py                                   # anytime; no torch

# s00: download + verify (already wired to wait for the leafllm runner to stop)
setsid nohup bash ops/s00_fetch.sh > /dev/null 2>&1 &

# s01: kernel gate (~15 GB: layers 2 and 3 + embedding; GPU arm then CPU oracle)
setsid nohup python3 ops/s01_layer_probe.py --tokens 2048 > ops/logs/s01.log 2>&1 &
#   if TileLang will not import/build for sm_110a, s01 exits 3 and records the error, then:
setsid nohup python3 ops/s01_layer_probe.py --kernels torch > ops/logs/s01_torch.log 2>&1 &
#   validate the token-chunked Block path s03 uses for long rows:
setsid nohup python3 ops/s01_layer_probe.py --tokens 4096 --block-chunk 1024 > ops/logs/s01_chunk.log 2>&1 &
python3 -c 'import sys; sys.path.insert(0, "."); from ops.lib.env import reclaim; reclaim("s01")'

# s02: corpus (CPU + network)
python3 ops/s02_calib_corpus.py --dry-run
setsid nohup python3 ops/s02_calib_corpus.py --probe > ops/logs/s02_probe.log 2>&1 &
setsid nohup python3 ops/s02_calib_corpus.py > ops/logs/s02.log 2>&1 &

# s03: saliency (hours; resumable; Engram shards may live on another drive)
setsid nohup python3 ops/s03_saliency.py run --engram-dir /path/to/shards47-48 > ops/logs/s03.log 2>&1 &
python3 ops/s03_saliency.py status

# guard (CPU): protect lists before s04 selection, verdict on each candidate mask after
python3 ops/drift_guard.py protect
python3 ops/drift_guard.py check --keep ~/models/v41-work/s04/mask_hope.json
```

**What s01 must answer before s03 starts:**
1. Does TileLang 0.1.12 import and build on sm_110a? s01 records the error if not.
2. Does the stream output agree with the oracle at cosine ≥ 0.999 (verdict PASS)? Does top-6 routing agree?
3. What are tok/s and peak memory per 2k tokens? `BYTES_PER_TOKEN_PEAK` in s03 (240 kB, estimated) must be replaced by the measured value.
4. Does the s03 query-chunked Indexer match the reference? The GPU arm uses the chunked one and the oracle uses the original, so the cosine checks the patch too.

## Design decisions

### s01: probe layer 3 with its real source layer

**Why layer 3.** Layer 3 is a ratio-2 layer that only consumes cross-layer state. It reads compress_kv, index_k and top-k indices from layer 2.

**What runs.** The probe builds layers 2 and 3 and runs them in order. The cross-layer state is therefore produced by the real source layer, not stubbed. The input is the real embedding of a fixed prompt, expanded to the 4 hc streams. Layers 0-1, and with them Engram, are skipped. The report says so: this is a kernel-agreement probe, not a quality measurement. `--hidden FILE` runs layer 3 from a recorded stream instead.

**The oracle.** The oracle is the same reference `model.py` with `lib/torch_kernels.py` on CPU. bf16 parameters and buffers are promoted to fp32; fp8/fp4 weights and their e8m0 scales stay as stored.

**What the torch kernels reproduce.** Each torch kernel reproduces its TileLang counterpart's clamps, power-of-two scale rounding, e2m1 round-to-nearest-even, and the bf16 cast of the unnormalised softmax weights in `sparse_attn`.

**Default device.** model.py creates some tensors without a device (`get_window_topk_idxs`, which is also `lru_cache`d). Each arm therefore sets the default device the way generate.py does, and clears that cache between arms.

### s02: corpus (final mix)

The PLAN mix summed to 98% and had seven pools. The user then added four requirements (2026-10-19):
- a **long-context** pool;
- a **multilingual** pool;
- a **long-tail world-knowledge** slice inside general;
- **every pool ≥ 10%**, with vision at about 14%.

Nine pools at 10% leave 10 points to place. They went to vision (+4) and to the PLAN's largest pools (+2 each to general, agentic and code):

| pool | share | 2M-token plan | contents |
|---|---|---|---|
| general | 12% | 29 × 8k | fineweb-edu, plus a world-knowledge slice: encyclopedic-URL-filtered fineweb-edu, TriviaQA rc, MMLU auxiliary_train, OpenBookQA |
| multilingual | 10% | 25 × 8k | fineweb-2: Chinese is half the pool, then ru, ja, ko, de, fr, es, ar, pt, vi, id |
| agentic | 12% | 29 × 8k | V4.1 self-generated traces (once they exist), SWE-rebench / SWE-Gym / SWE-bench issue→patch (human), xlam function calling rendered as DSML (`unverified`, off by default) |
| code | 12% | 29 × 8k | self-gen, OpenCodeReasoning-2 (R1/QwQ), OpenCodeInstruct (Llama-3.3), KernelBook |
| vision | 14% | 34 × 8k | real images through the reference `image_processor`: the_cauldron (chartqa, ai2d, docvqa, textvqa, infographic_vqa, screen2words), Docmatix, CLEVR-Math |
| math | 10% | 25 × 8k | self-gen, OpenMathReasoning, DeepMath, OpenR1-Math, finemath, Lean-Workbook |
| science | 10% | 25 × 8k | self-gen, OpenThoughts3, MMLU-Pro*, ChemBench, science-URL fineweb-edu |
| finance | 10% | 24 × 8k | t2-ragbench (FinQA / ConvFinQA / TAT-DQA), bizbench, tat-llm, finance-URL fineweb-edu |
| longctx | 10% | ~200k tokens | whole 32k-128k documents, one per row: PG-19 books, upstream llama.cpp and CUTLASS subtrees (read from the upstream ref with `git show`), self-gen long agentic traces |

*MMLU-Pro test is in the science pool, as it was in GLM's corpus. If MMLU-Pro is in the s10 panel, it must leave the corpus first (contamination).

**Rows and order.**
- Rows are bucket-pure, so every token's pool is known.
- They are written in a smooth weighted round-robin order, so every prefix of the pass is balanced. GLM's grouped order put all media at 85% of the run.
- The last 10% of each pool's rows form the **frozen held-out split**. s03 gives those rows teacher logits and routing, but they never touch an accumulator. s04 arbitration and s10 use them.

**Provenance and licences.**
- No Claude-written text. Known Claude sets (SWE-smith trajectories, OpenHands-Sampled-Trajectories, pixmo-docs) are listed in `EXCLUDED`.
- Sources whose generator is unknown or mixed are excluded too.
- Sources named from memory and not yet checked against their card are marked `unverified` and stay off unless `--allow-unverified` is passed.
- `--generator-policy human-or-self` drops all other-model text (R1, QwQ, Llama, Phi-3). It keeps only human text and V4.1-1017's own output.
- Licences: the card licence is re-read live at build time and recorded beside the declared one. A source whose live licence is NC or share-alike is refused.
- Wikipedia and Natural Questions (CC-BY-SA) are excluded under the standing policy. See open questions.

**Format.** Chat-shaped sources are rendered by the checkpoint's own `encoding/encoding.py`. That includes thinking mode with `reasoning_content` and tools as DSML. The router therefore sees V4.1's real serving format. Raw documents get the BOS token.

**Dead-source handling.** Media sources die silently (on GLM, 9 of 15 yielded zero images). s02 therefore counts rows read and kept per source, gives up on a source after 2,000 rows with zero kept, and passes its quota to the next source in the pool. A pool below 90% of its budget is a hard error; another pool never silently absorbs it.

**Image token spans.** These are computed with the reference `plan_image_grid`. s03 recomputes them with `load_image` and asserts equality.

### s03: layer-major saliency

**The hidden stream.**
- The stream is held on NVMe as one in-place array: `h.npy` [T, 4, 5120] bf16 plus `pre_mix.npy` [T, 4] fp32, about 82 GB per 2M tokens.
- Each layer unit streams every row through one resident layer.
- Rows are ragged: 8k for most pools, 32k-128k for longctx.

**Journal protocol.**
1. Compute the row.
2. Write the journal file (fsync, rename).
3. Every `--ckpt-rows` rows, write an atomic accumulator checkpoint that already covers those rows.
4. Apply the journals to the stream, routing and state files.
5. Mark the rows applied and delete the journals.

On restart, covered rows are re-applied (this is idempotent). Uncovered rows are recomputed from the untouched stream, and their stale journals are dropped.

**Cross-layer attention state.**

| source | publishes | read by |
|---|---|---|
| KV-source layers 2, 8, 14, 20 | compress_kv, index_k | later layers |
| Index-source layers 2, 8, …, 36 | top-k | later layers |
| Layer 20 | candidate blocks | layers 24-36 |

- Each publisher saves its state per row under `state/<kind>_L<src>/`, and each consumer re-installs it into `model.shared_attn`.
- `s03.sources_for()` encodes the topology. The tests check it against the config.
- A state directory is deleted once its last consumer has finished.

**Indexer.**
- Chunked over queries, with a 2 GiB score budget per chunk. The reference materialises [s, 32, t] fp32, which is 8 GiB per 8k row.
- Up to 16,384 compressed positions, the candidate mask is the identity on top of the causal mask, so nothing is stored.
- Beyond that (ratio-1 rows longer than 16k), layer 20 runs the reference `select_candidate_blocks` per chunk and bit-packs the block mask (2 KiB per token at 128k). Layers 24-36 unpack it.

**Block.**
- Everything per-token (hc mixes and Sinkhorn, hc_pre, the norms, the MoE, hc_post) runs in 8k token chunks. Attention runs over the whole row.
- The reference hc_post alone would materialise [s, 4, 4, 5120] fp32, 43 GB at 128k.
- Rows ≤ 8k take the untouched reference path.
- Estimated peak at 128k is about 30 GB of activations plus the layer. **s01 must measure this.**

**Engram.**
- Hash ids come from the reference `NgramHashState`, using the `tokenizers`-only shim. That shim asserts the 99,092 compressed vocab.
- The rows are gathered in one sequential sweep of shard 47 or 48 per Engram layer, which can sit on another drive (`--engram-dir`).
- The cost is 6,336 B per token per layer (24 × (256 + 8)), about 12.7 GB per 2M tokens per layer.
- During the forward pass, a `RowSource` serves the rows in place of the 98 GB table.
- Engram is never switched off.

**Statistics per layer** (`bank/L<L>.npz`):
- **REAP family**, per pool × expert: sum and sq of g‖f‖, nrm/nsq, gat/gsq, cnt.
- **Log-saliency histogram.**
- **Gated-output vector sum**, for heal-gain measurement.
- **HOPE F**: 384×384 sum and count.
- **Super-Expert guard**: `fmax`, the max |down_proj output| ungated (and `gfmax`, gated).
- **NVFP4 input scales**: `amax_in` and `amax_mid`.
- **imatrix**: `x2_in` and `x2_mid` (per-column Σx²) and `n_tok`.
- **Shared expert**: the same amax and x² terms.

**Routing** (`routing/L<L>_{idx,w}.npy`): the teacher's top-6 ids and weights for every token, held-out rows included. This feeds heal replay, the drift test and the s04 KL arbitration.

**Head** (`teacher/`): top-32 log-probs and ids, logsumexp, and next-token NLL for every token.

**DSpark taps** (`taps/`, held-out rows by default): the mean over hc copies at the inputs of layers 37-39, which is what DSpark reads.

**Why the gated output is the REAP term.** The reference multiplies the routing weight in before w2, so the expert's real contribution is w2(g·act). Its norm is recorded as the REAP term g‖f‖, and ‖f‖ = that norm / g. The two differ from an ungated w2(act) only through the fp8 activation grid. Recording the gated value keeps the term exactly what the model adds to the stream.

**Disk at 2M tokens** (estimate):

| item | size |
|---|---|
| stream | 82 GB |
| Engram rows | 25 GB |
| hashes | 0.2 GB |
| routing | 1.9 GB |
| teacher | 0.5 GB |
| live cross-layer state | ≤ 6 GB |
| journals | ≤ 3 GB |
| held-out taps | ~6 GB |
| **total** | **~125 GB**, plus the shards (all 475 GiB, or one 7.4 GB layer at a time with `--fetch-shards --drop-shards`; Engram shards 47/48 on another drive) |

### Drift test and long-context guard (`lib/drift.py`, `drift_guard.py`)

This implements KQ4 of `notes/long_context_routing_drift.md`.

**Rates.** Per (layer, log-position bin, row), three rates of the prune set: slot, token, and **gate mass** (the primary one), plus a top-1-pruned rate. The bins are [0,8), [8,64), [64,256), [256,1k), [1k,2k), … up to [64k,128k).

**Statistics.**
- Intervals use a cluster bootstrap over rows, because the row is the independence unit.
- Drift is tested paired within row against the [256,1k) bin, with a sign test and a row-fixed-effects slope on log-position.
- An across-row, same-bin control sits beside it.

**Blocking rules.**
- **(a) Sink.** BLOCK if the mask prunes a **sink-coupled** expert, or a Super-Expert candidate (from `fmax`) that is routed in [0,8). Sink-coupled means the expert's share in [0,8) is at least 10× its share elsewhere.
- **(a') Rising.** BLOCK if the mask prunes a **position-rising** expert. Rising requires three things together:
  1. the row-bootstrap CI of the slope is above 0;
  2. the pooled late-bin share is at least 1.5× the pooled [8,1k) share;
  3. the late-bin share is non-trivial.

  The effect-size gate exists because, at 384 experts, a CI alone falsely flags about 6 flat experts. That was measured in the test suite, and the gate brings it to 0.
- **(b) Position-sensitive.** POSITION_SENSITIVE if the mask's mass slope is positive with its CI above 0 in ≥ 25% of layers.

**How the guard is used.** `drift_guard.py protect` evaluates every expert and writes per-layer protect lists. s04 passes these to HOPE as `protect`, so no candidate mask can prune them. `check` then re-verifies each final mask.

### s02b (later, optional): teacher self-generated traces

**Why.** The user wants calibration in V4.1's own chat, thinking and agentic format, using text V4.1-1017 wrote itself. Claude-written or other-model text is never used for this.

**How generation runs.** It needs a running 1017 teacher. Thor cannot hold the unpruned 475 GiB model, and a layer-streamed forward cannot sample. Generation therefore runs on:
- PLAN source B: a rented 2×B200 or 4×H200 vLLM node at the pinned revision; or
- source C: the official API, only after its outputs pass an equality check against source A, the s03 teacher top-32.

**Prompts.** Prompts come from human sources: SWE-rebench issues, MMLU-Pro-style questions with answers removed, finance tables, and long-context tasks over PG-19 and repo subtrees, which produce the long agentic traces.

**Output.** It writes `$V41_WORK/selfgen/{agentic,code,math,science,longctx}.jsonl`. The format is in `corpus_spec.py`, and every record carries `generator` and `teacher_check`.

**Effect on s02.** s02 already lists these files first in their pools, and they yield nothing until they exist. Once they do, rebuild with `--generator-policy human-or-self`.

## Remaining stages (inputs → outputs)

**s04 selection + arbitration (CPU, then GPU for KL)**
- **Inputs:**
  - `s03/bank/L*.npz`
  - `s03/routing/`, `s03/teacher/` (held-out rows)
  - `s04/position_protect.json` (drift guard)
- **Steps:**
  1. Rank with HOPE pᵀFp, using the REAP-sum diagonal and renormalised top-k weights.
  2. Sweep `protect_frac` and take the measured knee. It is not monotone: on MiMo, pf=0.01 was worse than 0.
  3. Rank vision per modality against `bias_vl`.
  4. Apply the Super-Expert guard.
  5. Never prune the shared expert, or the vision tower or aligner (they are not MoE and are kept whole).
  6. Run `drift_guard.py check` on every candidate.
- **Pre-registered arms:**
  - **A:** 192 kept experts, requantised to about 2.85 bpw (s07).
  - **B:** 128 kept experts left bit-exact MXFP4. That is about 107 GiB with FP8 dense, less with NVFP4 dense, and it uses the proven hardware FP4 GEMV.
  - Both arms get HOPE-knee, REAP-sum, routing-mass, random and frequency masks.
- **Arbitration:** paired held-out top-32 KL and top-1 agreement against the teacher, with proper re-routing over the survivors, evaluated in memory, **at matched resident bytes**.
- **Note on prior evidence:** the "pruning is 14× more damaging than quant" figure is regime-dependent. The user has measured NVFP4 50% REAPs agreeing with the base better than unsloth ~1-bit quants at equal memory. It must not be used to rule out arm B.
- **Outputs:**
  - `s04/mask_<arm>.json` ({layer: kept ids}) with a sha256 fingerprint;
  - `s04/arbitration.json` (per-mask KL, top-1 and CIs);
  - drift reports.

**s05 surgery + MXFP4 master upload**
- **Inputs:** the chosen mask, the 1017 shards (sha256-pinned).
- **Steps:**
  - Stream shard by shard: copy the kept experts' packed E2M1 + E8M0 tensors, renumber `ffn.experts.{0..N-1}`, and slice `gate.weight`, `gate.bias` and `gate.bias_vl`.
  - Set `n_routed_experts`, regenerate the index, and verify the kept indices against the source. A mismatch is fatal.
  - DSpark (`mtp.*`) stays unpruned. Engram is shipped as a sha256 pointer to base shards 47/48.
- **Outputs:**
  - `…-1017-REAP192-MXFP4` (about 151.6 GiB) and, if arm B wins or ships too, `…-REAP128-MXFP4`;
  - one detached upload unit per repo (`HF_XET_FIXED_UPLOAD_CONCURRENCY=8`).

**s06 heal**
- **Inputs:** the master, `s03/routing` (router replay), `bank osum`.
- **Steps:**
  - Measure the per-layer router-replay gain **on the final keep-set** immediately before applying it. A keep-set mismatch is fatal. There is no first-moment fallback.
  - E8M0 cannot carry a non-power-of-two gain, so the gain is folded into the NVFP4 per-tensor global scale and into the dequantised weights before the one requant.
  - Router-bias refit, MoRA and frozen-expert LoRA enter only through a paired s10 gate.
- **Outputs:** `s06/heal_gain.json` (keep-set fingerprinted).

**s07 quantization, last and once**
- **Inputs:** master, heal gain, `bank amax_*`, `lib/imatrix.layer_entries()` remapped to the keep-set (then `imatrix_verify`).
- **NVFP4:** a lossless repack of MXFP4, with input scales from the banked amax and the global scale carrying the heal gain.
- **GGUF:**
  - Experts are requantised from the MXFP4 source with GPTQ-style error feedback, about 35 layers near 3 bpw and the 4-6 least salient near 2. The protected layers are 0-1, Engram 1/14, KV sources 2/8/14/20 and DSpark taps 37-39.
  - Attention, compressor and indexer: Q8_0. Shared experts, embedding and output: Q6_K. Router: BF16. Hyper-connection, norm and sink tensors: F32.
  - Arm B instead keeps the experts bit-exact with **llama.cpp's MXFP4 type** (lossless).
- **Outputs:** `…-NVFP4` (about 159.6 GiB) and `…-GGUF` (about 101 GiB weights, split below 50 GB, Engram excluded and built locally as the O_DIRECT row file).

**s08 CUDA server**
- **Source:** a **copy** of `~/deepseek-v4-flash-0731-cuda`, adapted in a new location (`~/deepseek-v4.1-flash-reap-thor/cuda/` or a sibling repo). The 0731 repo is read-only.
- **Additions:**
  - the Engram O_DIRECT row store, with prefetch at sampling time, and the gate kernel;
  - the CSA2 roles (KV-source / index-source / reuse layers and the 2,048×8 candidate pass);
  - `bias_vl` routing and the ViT/aligner;
  - a sub-3-bit expert GEMV (arm A) or the existing hardware FP4 GEMV (arm B);
  - DSML encoding from `encoding.py`.
- **Gates:** bit-exactness against s03's teacher top-32 at matched weights, then the ≥15 tok/s gate.

**s09 llama.cpp fork**
- A lean fork used as the independent reference and as the GGUF consumer.
- Contents: V4.1 graph, Engram row-file reader, MXFP4 experts. The upstream PR is unmerged and capped near 16K context.
- Cross-runtime noise is bounded by running one model in both runtimes.

**s10 eval gates**
- **Inputs:** the frozen held-out split plus the s03 teacher cache.
- **Primary measure:** paired top-32 KL, top-1 and ΔNLL, with a Model Confidence Set across competitors.
- **Also checked:** long-context drift (`drift_guard.py check` and a 32k-128k panel), vision captioning and VQA, and the capability panel scored against teacher source B.
- **Go/no-go:** beat GLM-5.3 REAP50 v2 and reach ≥15 tok/s.

**s11 DSpark refit**
- **Inputs:** `s03/taps` and the final pruned target.
- DSpark's 3 blocks (128 experts, top-3) are refit to the pruned target's hidden states. The output is the drafter for s08.

## Open questions

1. **TileLang on sm_110a.** This is unknown until s01 runs. The torch fallback is correct by construction but possibly 2-3× slower. The PLAN estimates 3-4 h per 1M tokens with TileLang.
2. **Disk.**
   - s03 needs about 125 GB of work space on top of the shards.
   - Holding all 475 GiB of shards locally needs the 0731 move to the Seagate to finish first. Otherwise use `--fetch-shards --drop-shards`.
   - Engram shards 47/48 (203 GB) are best gathered from the external drive with `--engram-dir`.
3. **Other-model reasoning traces.** The default policy (`no-claude`) keeps R1/QwQ/Llama-generated sets. Should they be dropped (`human-or-self`) as soon as s02b exists, or now?
4. **Wikipedia / NQ (CC-BY-SA) for the world-knowledge slice.** Calibration produces statistics, not a derivative text, but the standing policy excludes share-alike. Deciding this is the user's call.
5. **`unverified` sources.** These are xlam, the Nemotron tool-calling split and AI-CUDA-Engineer. Their cards need a human read before `--allow-unverified` is used.
6. **Long-row memory.** The ~30 GB activation estimate at 128k is unmeasured. Run s01 with `--tokens 32768`, or a first long s03 row, before committing to 128k rows.
7. **DSpark taps.** Held-out-only taps are about 6 GB, while all rows would be about 60 GB. How much does s11 need?

## Decisions taken 2026-10-19 (standing autonomy grant, see memory dsv41-1017-reap-pipeline)
1. **Generator policy stays `no-claude`.** Other open-model reasoning traces (R1 is V4's own lineage, QwQ, Llama) are
   calibration input only: they shape routing statistics, nothing is redistributed. Claude-written sources stay excluded.
   Switch to `human-or-self` once s02b has teacher-self-generated traces.
2. **Share-alike allowed for calibration** (Wikipedia, Natural Questions): the output is saliency statistics, not a
   derivative text. `wikimedia/wikipedia 20231101.en` added to `general` (w 0.12). NC and GPL stay excluded.
3. **Corpus 2M -> 5M tokens** so every pool gets >= ~500k routed tokens (dose-response note; GLM pass 1's main defect was
   under-calibration). Hidden states page to NVMe (~205 GB).
4. **Engram shards 47/48 move to the Seagate before s03** (sha256-verified copy, then the NVMe copies are deleted), and
   s03 runs with `--engram-dir "/media/patrickd/Seagate Portable Drive/v41_1017_engram"`. Calibration reads them once,
   sequentially, which USB handles. **Serving never reads Engram from the Seagate:** the packed O_DIRECT row file is
   rebuilt on the NVMe once the source backbone shards are gone.
5. The three default-off sources (xlam, Nemotron tool-calling split, AI-CUDA-Engineer) are enabled only after their
   dataset cards are read and recorded (generator + licence).

## chain2: s08 -> s09 -> s12 (GGUF) -> s11 -> s12 (`chain2.sh`)

`ops/chain2.sh` is the post-chain. Launch it detached at any time (`setsid nohup bash ops/chain2.sh > /dev/null 2>&1 < /dev/null &`): it waits for `chain.sh` to log "chain reached the end of implemented stages", exits if `logs/chain.STOP` exists, and takes `/tmp/dsv41-chain2.lock`. It follows chain.sh's rules: `quiet()` before and `reclaim()` after every model-loading step, `done.c2_*` markers (the s08/s09 step files keep `done.s08_*`/`done.s09_*`), and a failure writes `chain.STOP` prefixed `chain2:`.

Step order:

1. **Phase 0 (space).** Delete the dead `*.incomplete` partials of s00's killed first attempt (`$SRC/.cache/huggingface/download`, 17 GiB). Move the losing Thor arm's build (if both exist) to the Seagate, verified against its `thor.json` sha256s.
2. **Phase A (source still on NVMe).** s09 `build`; s08 `build`, `cpu_gates`; GPU: s08 `units`, `gemv_bench`; s09 `backend_ops`; s08 `golden_ref`/`golden` for `src`, then for the served arm (A192 or B128, from `s10/gates.json`).
3. **Move the backbone.** Shards 1-43 go to the Seagate, each sha256-checked against the pinned tree, then symlinked. This happens only after s10final. Shards 44-46 (the DSpark head) and the small files stay.
4. **Phase B.** s09 `rowfile` (Engram rows on NVMe); s08 `golden_store` (the golden comparison through the real row store), `decode`, `ctx1m`, `server`.
5. **Phase C (GGUF).**
   - Re-download the REAP master from its private repo (`HF_XET_HIGH_PERFORMANCE=1`) to the Seagate and verify every shard against the s05 manifest.
   - s09 `convert` (source GGUF on NVMe), then `imatrix`.
   - `quant` L0..L3, each written to a Seagate directory symlinked as `$V41_WORK/s09/<LV>`.
   - Delete the source GGUF (regenerable).
   - Per level: stage the splits to NVMe, run `kld` and `smoke` (partial `-ngl` when a level is over the memory budget), copy the results back, and drop the staged copy.
   - `ops/s09_gate.py`, then `manifest --levels <passing>`, then `s12_publish.py --only gguf`. If no level passes, it logs that and publishes nothing; this is not a STOP.
6. **s11** (`ops/s11_spec.py`): `traces` eval/train, `tapdump` eval/train, `bank`, `prune`, `quant`, `accept` native ×2 arms, `refit` ×2, `accept` tuned ×2, `curve`, `e2e`, `decide`, `publish`.
7. **s12 again**, so every card carries the s11 result. If s11's end-to-end gate is blocked (no draft path in `cuda/`), chain2 ends with a STOP that says so. leafllm stays paused.

### chain2 disk plan (GiB; measured 2026-10-19 unless marked est.)

NVMe is 936 GiB. Before s00b finished there were 715 GiB used and 175 GiB free:
- the source directory: 397, which includes shard 48 (94.6, mid-move) and 17 of dead partials;
- everything else on the drive: about 318 (est.).

The source backbone, shards 1-46, is 286.1, of which shards 44-46 are 7.4. The Seagate is 3.7 TiB with 2.1 TiB free.

| point in time | NVMe used | NVMe free | what changed |
|---|---|---|---|
| chain2 start, arm A won (est.) | ~751 | ~185 | other 318 + source 303 + thor/A192 ~100 + s03/s04 caches ~30 |
| chain2 start, arm B won (est.) | ~851 | ~85 | plus thor/B128 ~100 |
| after phase 0 | ~734 | ~202 | -17 dead partials, -100 losing build (arm B case) |
| after phase A | ~744 | ~192 | golden dumps ~10 |
| after the backbone move | ~465 | ~471 | -279 (shards 1-43) |
| phase B: Engram rows | ~654 | ~282 | +189 (guard: 200 free) |
| phase C: source GGUF | ~806 | ~130 | +152 (guard: 170 free); the levels go to the Seagate |
| phase C: after dropping the source GGUF, L0 staged | ~799 | ~137 | -152, +145 (guard: level + 25) |
| s11 | ~680 | ~256 | taps ~9, heads ~20 |

Seagate use:

| item | GiB | notes |
|---|---|---|
| Engram shards (s00b, already there) | 189 | |
| losing Thor build | ~100 | if any |
| backbone | 279 | |
| master | ~152 | guard: 170 free |
| GGUF levels | ~451 | L0 145, L1 118, L2 100, L3 88 |
| transient unsplit level | up to 145 | guard: 320 free per level |

That totals about 1.17 TiB, leaving about 0.9 TiB.

s03's NVMe paging (~82 GB for a 2M-token corpus) belongs to chain 1, and s04 drops it (`--drop-s03-stream`) before chain2 starts.

**Never served from the Seagate.** The Seagate is NTFS over USB at ~130 MB/s. It is used only for staging:
- the s09 row files and every model that a GPU step loads are read from NVMe;
- `kld` refuses a partial offload from `/media`.

The exception is `s11_tapdump`, which reads the source backbone's dense weights through the symlinks at about 36 min per pass. That is calibration-style sequential reading, not serving.
