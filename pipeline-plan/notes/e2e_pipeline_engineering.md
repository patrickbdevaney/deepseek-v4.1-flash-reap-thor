# End-to-end engineering plan: DeepSeek-V4.1-Flash-1017 -> REAP/HOPE prune -> heal -> {MXFP4 safetensors, NVFP4, GGUF} on one Jetson AGX Thor

Written 2026-10-19. Conventions: GiB = 2^30 B, GB = 10^9 B. **[MEASURED]** = measured on this Thor or by a cited source; **[DERIVED]** = arithmetic from config/headers in this note or a cited local note; **[EXTRAPOLATION]** = scaled from a different model or setup, not measured for V4.1. LOCAL citations are files on this machine. No downloads or GPU jobs were run for this note; the only command touching state was a read-only `df`/`du`.

**One-paragraph plan (for the report writer).** The repo is 510.3 GB across 48 shards. 203.1 GB of that is the two Engram tables (shards 47 and 48, one Engram layer each); the remaining 307.2 GB is the backbone, which carries 268.95 GiB of MXFP4 experts. Today the NVMe has 102 GB free, so the binding constraint is **disk, not RAM or compute**. The cheapest workable path has six parts.
1. Run saliency **layer-major**: download one ~7.4 GB layer shard, run that layer over the whole calibration set (hidden states resident in RAM), accumulate REAP/HOPE statistics, delete the shard, and move to the next. In the same pass, capture the teacher's top-k logits, per-expert activation amax (for NVFP4) and per-column x² (for the GGUF imatrix).
2. Precompute the Engram rows for the calibration tokens off-box, or on a second drive.
3. Do surgery as a byte-exact stream (download shard, slice, write, delete). Legal expert counts are 320, 256, 192 and 128 only.
4. Heal gradient-free, or with block-wise router/non-expert updates. Never train the MXFP4 experts themselves.
5. Make NVFP4 by a lossless, deterministic MXFP4->NVFP4 repack. NVIDIA reports that every weight block converts losslessly.
6. Make the GGUF with the open llama.cpp PR #28696 converter. The imatrix comes from our own streamed statistics, because the pruned model at native precision does not fit in RAM. The trellis types (IQ*_KT) need ik_llama.cpp, which is unverified on sm_110.

A USB-attached second NVMe is the single purchase that removes most of the pain. A rented big-RAM CPU box is the cheapest fallback for the Engram-row precompute and for GGUF quantisation.

---

## Q1. Download and storage: total repo size, which files saliency needs vs Engram, per-shard streaming, second-drive requirement

### Takeaway
`deepseek-ai/DeepSeek-V4.1-Flash-1017` is 510.31 GB (475.3 GiB) in 48 shards.
- Shards 47 and 48, at 101.5 GB each, are the Engram tables: 203.1 GB, one table per Engram layer.
- Shards 1-46, 307.2 GB, are the backbone. Shards 3-42 are ~7.39 GB each, which looks like one decoder layer per shard.

The NVMe has 102 GB free, so neither the full backbone nor a single Engram shard plus working space fits. There are three ways to proceed:
- **(a)** Layer-major streaming straight from HF. A ~49 min download per full pass is cheap at the measured 105 MB/s.
- **(b)** A second drive. The only attachment options are USB 3.2 Gen2 or the Key-E x1 slot.
- **(c)** Freeing project-owned space. This is an operator decision; 98 GB sits in `~/glm-5.3-reap/output`.

### Cited Findings
- Repo file listing: 48 safetensors shards totalling 510,296,708,312 B, plus 13.8 MB of other files (`config.json`, `tokenizer.json`, `model.safetensors.index.json` at 7.47 MB, `LICENSE` at 1,084 B, `README.md`, and `inference/` and `encoding/` dirs). Grand total 510,310,557,884 B. Sizes by shard:

  | Shard(s) | Size each |
  |---|---|
  | 1 | 0.97 GB |
  | 2 | 1.32 GB |
  | 3-42 | ~7.39-7.41 GB |
  | 43 | 1.32 GB |
  | 44 | 2.65 GB |
  | 45 | 2.57 GB |
  | 46 | 2.71 GB |
  | 47 | 101,535,150,936 B |
  | 48 | 101,537,926,640 B |

  — [HF API tree, DeepSeek-V4.1-Flash-1017](https://huggingface.co/api/models/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/main)
- LOCAL byte accounting for 1017: 475.24 GiB total. Of that, experts at MXFP4 are 268.95 GiB, Engram is 189.13 GiB (183.40 GiB FP8 tables + 5.72 GiB E8M0 scales + 0.29 GiB wkv, in shards 47 and 48), and the fixed floor without Engram is 9.77 GiB, plus 7.39 GiB of DSpark. "The tables changed in the 1017 release. Do not reuse the Engram shards of the older DeepSeek-V4.1-Flash" — [LOCAL v41_architecture_pruning_delta.md](file:///home/patrickd/glm-5.3-reap/research_notes/Compressing%201T%20MoE%20for%20Thor/v41_architecture_pruning_delta.md), citing LOCAL target_models.md
- Config: `engram_layer_ids [1, 14]`, `engram_num_embeddings [384006168, 384016682]`, 8 hash heads, `engram_head_dim 256`, n-gram orders 2-4. 40 MoE layers, each with 384 routed experts plus 1 shared, top-6, `moe_intermediate_size 2304`, hidden 5120 — [config.json](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/raw/main/config.json) via LOCAL v41 note
- Disk now: `/dev/nvme0n1p1` 936G size, 787G used, 102G available (89%) — [MEASURED `df -h /`, this session]. `~/glm-5.3-reap/output` holds 98G — [MEASURED `du`, this session]
- Download rate on this link: 328.3 GB in "~52 min at the measured 105 MB/s link rate"; local NVMe read 3.4 GB/s, "~32x the link" — [LOCAL s01_source.py](file:///home/patrickd/glm-5.3-reap/scripts/stages/s01_source.py). Local NVMe write 487 MB/s vs read 3.4 GB/s — [LOCAL s03_saliency.py](file:///home/patrickd/glm-5.3-reap/scripts/stages/s03_saliency.py)
- GLM precedent: surgery deleted "each source shard once its survivors have been written, which makes free space grow monotonically… safe because the source is public and re-downloadable" — [LOCAL s04b_surgery.py](file:///home/patrickd/glm-5.3-reap/scripts/stages/s04b_surgery.py)
- Thor attachment points: one M.2 Key M PCIe Gen5 x4 (occupied by the NVMe), one M.2 Key E Gen5 x1 (Wi-Fi), QSFP28 4x25GbE, 5GbE, USB 3.2 Gen2. "A second NVMe would have to use the Key E x1 lane or USB" — [LOCAL engram_off_device_thor.md](file:///home/patrickd/glm-5.3-reap/research_notes/Compressing%201T%20MoE%20for%20Thor/engram_off_device_thor.md)
- LibertAI built the Engram tables "on `meta`", mmap'd them as "a gather, never resident", and excluded them from the pruned checkpoint (served from base shards 47 and 48) — [Libertai/deepseek-v41-flash-reap](https://github.com/Libertai/deepseek-v41-flash-reap)

### Inferences
- [DERIVED] Backbone = 510.3 − 203.07 = **307.2 GB (286.1 GiB)**. That is 40 layer shards × ~7.39 GB = 295.6 GB, plus ~11.6 GB in shards 1-2 and 43-46 (embeddings, head, DSpark/MTP, vision). Per-layer MXFP4 experts are 6.72 GiB = 7.22 GB, which matches the 7.39 GB shard size, so **one shard ≈ one decoder layer**. Verify against `model.safetensors.index.json` (7.47 MB, fetch it first) before relying on this.
- [DERIVED] Saliency needs all of shards 1-46. It also needs the Engram rows for the calibration tokens only, not the tables: 24 rows × 264 B = **6.3 KB per token per Engram layer**. 1M calibration tokens therefore need ~12.7 GB of gathered rows for both layers combined (before deduplication). The rows are a pure function of token ids, so they can be precomputed anywhere that has the tables.
- Disk options, cheapest first:
  1. **Layer-major streaming from HF, no new hardware.** Peak disk is about one 7.4 GB shard, plus the Engram-row cache (~6-25 GB), plus outputs. A full download pass costs ~49 min (307 GB at 105 MB/s), so re-downloading the backbone for each of saliency, surgery, NVFP4 and GGUF costs ~3.3 h of link time in total. That is acceptable.
  2. **Engram shards.** Each 101.5 GB shard is effectively the whole free disk; staging one with 0.5 GB of margin is not acceptable. Options:
     - (i) Rent a cheap x86 VM with ≥250 GB disk, download shards 47 and 48 there, compute gathered rows for the calibration token ids, and ship back ~13-25 GB.
     - (ii) Attach a 2-4 TB NVMe over USB 3.2 Gen2 (~1 GB/s class link; no Thor-specific measurement found).
     - (iii) Operator frees `~/glm-5.3-reap/output` (98 GB) if those artifacts are already on HF. Never delete outside the active project.
  3. **For serving**, Engram must live on local NVMe anyway: 189 GiB FP8, or 97.3 GiB MXFP4, per [LOCAL engram_off_device_thor.md](file:///home/patrickd/glm-5.3-reap/research_notes/Compressing%201T%20MoE%20for%20Thor/engram_off_device_thor.md). So a second drive, or a cleaned disk, is required eventually regardless of how calibration is done.

### Gaps
- The exact tensor-to-shard mapping was not fetched. Fetching the 7.47 MB `model.safetensors.index.json` would confirm one-layer-per-shard, and show where shards 43-46 put DSpark, vision and the head.
- No measured USB 3.2 Gen2 NVMe throughput on Thor.

---

## Q2. Saliency on a 763B model on Thor: layer-streamed forward, token budget and sequence length, time, Engram's role, REAP/HOPE accumulation cost

### Takeaway
Run the reference `inference/model.py` layer by layer, with no HF transformers implementation. The calibration that fits is 8k tokens × 16 rows per chunk:
- Peak is ~28 GiB with MXFP4 kept resident and dequantised per expert, or ~47 GiB with the whole layer dequantised to bf16 [DERIVED].
- The indexer must be evaluated row-chunked; done naively it reaches 148-167 GiB and OOMs.
- Engram must be applied before blocks 1 and 14 and must never be disabled: perplexity doubles without it.
- REAP and HOPE statistics are negligible in cost (a HOPE F-matrix is 576 KiB per layer).

Wall-clock time is set by the MoE forward: about **3-4 h per 1M calibration tokens** [EXTRAPOLATION from the GLM rate of 3,461 tok/s per MoE layer]. A 1-2M token budget is the practical sweet spot. LibertAI's whole V4.1 prune used 262k text tokens.

### Cited Findings
- No HF `transformers` implementation of `deepseek_v41` exists, so calibration must run on the checkpoint's own `inference/model.py`. The REAP hook goes inside `MoE.forward`/the expert loop after `hc_pre`, where the input is already single-stream `[tokens, 5120]`. Cross-layer state (`shared_attn`: `compress_kv` from layers 2/8/14/20, `index_k`, `topk_idxs`) lives in a module-level object, which "break[s] the 'one tensor + topk' invariant" of the GLM streamer — [LOCAL v41_architecture_pruning_delta.md](file:///home/patrickd/glm-5.3-reap/research_notes/Compressing%201T%20MoE%20for%20Thor/v41_architecture_pruning_delta.md)
- Memory per layer at S=8192, B=16 [DERIVED in the local note]:
  - Layer weights: 6.72 GiB MXFP4, or 25.31 GiB if dequantised to bf16.
  - 4-stream residual: 5.00 GiB.
  - Dense indexer scores: 4.0 GiB per row bf16, so 64 GiB for 16 rows.
  - Peak with a row-chunked indexer: **46.6 GiB** with the whole layer in bf16, **28.0 GiB** with MXFP4 resident. Naive: 166.6 / 148.0 GiB (OOM).
  - S=4096, B=16: 34.7 / 16.1 GiB.
  — [LOCAL v41 note, Q6](file:///home/patrickd/glm-5.3-reap/research_notes/Compressing%201T%20MoE%20for%20Thor/v41_architecture_pruning_delta.md)
- GLM-5.3 on this Thor: 8,192 tokens ≈ 57 GiB, and 16,384 OOMs from 93 GiB headroom ("the quadratic DSA/attention term… not the MoE") — [LOCAL long_context_routing_drift.md via v41 note](file:///home/patrickd/glm-5.3-reap/research_notes/Compressing%201T%20MoE%20for%20Thor/long_context_routing_drift.md)
- GLM streamed saliency rates on this Thor:
  - "3,461 tok/s, 281/288 experts fired" on one real MoE layer.
  - Per-layer times of 0.9-2.4 min after fixes, and an "ETA… ~60–85 min for the full 45-layer pass" of the 512-sample × 2048 = 1.05M-token corpus.
  — [LOCAL wiki/00-log.md](file:///home/patrickd/glm-5.3-reap/wiki/00-log.md)
- LibertAI V4.1 calibration and memory:
  - `--seq-len 2048 --n-seq 128 --batch 32`; "past roughly 256 tokens essentially every expert fires anyway".
  - Caveats state 262,144 text tokens + 256 images.
  - "A full forward pass streams 269 GiB of experts at about 0.65 GiB/s, with peak residency of 19.4 GiB" on "one 120 GB unified-memory box".
  - Wall-clock times are not reported.
  - Patched `patch_kernels.py` / `patch_engram.py` were needed.
  — [Libertai/deepseek-v41-flash-reap](https://github.com/Libertai/deepseek-v41-flash-reap)
- LibertAI's page also cites "64.4M text tokens and 13.0M image tokens", which it does not reconcile with the 262k caveat — [Libertai/deepseek-v41-flash-reap](https://github.com/Libertai/deepseek-v41-flash-reap)
- Engram mechanics and evidence:
  - Engram is applied by `Transformer.forward` **before** block 1 and block 14, as `h + gate·value` into all four streams.
  - Lookup cost is 48 rows per token across both layers (24 per layer).
  - "with the Engram lookups switched off… perplexity rises to 4.1648 (code) / 9.0627 (prose)" vs 1.9094 / 6.4520 with them on.
  - Rule: "never calibrate with Engram disabled".
  — [LOCAL v41 note Q4](file:///home/patrickd/glm-5.3-reap/research_notes/Compressing%201T%20MoE%20for%20Thor/v41_architecture_pruning_delta.md); [llama.cpp PR #28696](https://github.com/ggml-org/llama.cpp/pull/28696)
- HOPE needs no change for mHC. F per layer is 384² fp32 = 576 KiB, so 22.5 MiB for 40 layers; only K (8→6) and the expert count change from the GLM code — [LOCAL v41 note Q1](file:///home/patrickd/glm-5.3-reap/research_notes/Compressing%201T%20MoE%20for%20Thor/v41_architecture_pruning_delta.md)
- Chunk ordering on GLM: states kept per chunk, all layers swept per chunk, because "read is 7x faster than write" — [LOCAL s03_saliency.py](file:///home/patrickd/glm-5.3-reap/scripts/stages/s03_saliency.py)
- Calibration-order trap: chunks arriving grouped by domain bias partial passes — [LOCAL memory calibration-corpus-visit-order](file:///home/patrickd/.claude/projects/-home-patrickd/memory/calibration-corpus-visit-order.md)

### Inferences
- **Recommended order is layer-major for V4.1, the opposite of GLM's chunk-major.** GLM could keep the source on local disk and re-read it at 3.4 GB/s, but V4.1's backbone is 3× the free disk. With hidden states resident in the unified pool, layer-major costs one download pass. [DERIVED] RAM budget:
  - 4-stream bf16 is 40,960 B per token, so 1M tokens = 38.1 GiB and 1.5M = 57.2 GiB.
  - Add ~28 GiB of MXFP4-resident layer working set, ~6 GiB of `shared_attn` state (≈52 MiB per 8k row × 122 rows), and the 12 GiB runtime reserve. That makes **~1.5M tokens the in-RAM ceiling** inside 117 GiB.
  - Above that, page hidden states to NVMe in place: ~82 GB for 2M tokens, at 487 MB/s write ≈ 2.8 min per layer, ≈ 1.9 h over 40 layers.
- Layer-major also makes cross-layer `shared_attn` trivial: layer 2's compressed KV for every row is complete before layer 3 starts.
- [EXTRAPOLATION] Time: assuming V4.1 reaches the GLM per-MoE-layer rate (~3.5k tok/s per layer), a full 40-layer sweep runs at ~85 tok/s, which gives:

  | Calibration tokens | Sweep time |
  |---|---|
  | 262k (LibertAI scale) | ~0.9 h |
  | 1M | ~3.3 h |
  | 2M | ~6.5 h |
  | 4M | ~13 h |

  Add ~49 min of download per pass. Per-layer active compute is 6 routed + 1 shared expert of 35.4M params each, plus attention. That is comparable to GLM, but V4.1's pure-torch MXFP4 dequant and the dense indexer fallback (no `fp4_gemm` kernel on sm_110a) could make it 2-3× slower. Measure on layer 3 before committing.
- Recommended budget: 1-2M tokens at 8k sequence length, domain-interleaved, plus the existing GLM corpus machinery (`s02_corpus`). That is 4-8× LibertAI's budget, which the local GLM work showed matters for 50% prunes (mass criterion plus a balanced corpus). Vision calibration is optional, since images need their own processor path. LibertAI found vision collapses below 272 experts under text-only ranking, so it matters only if the vision tower is shipped.
- **Capture in the same pass, for free:**
  - (a) Teacher top-k logits at the final layer. 12 importance-sampled logits ≈ 36 B per token, per [LOCAL layer_streamed_healing_infra.md](file:///home/patrickd/glm-5.3-reap/research_notes/Compressing%201T%20MoE%20for%20Thor/layer_streamed_healing_infra.md).
  - (b) Per-expert input amax, which supplies the activation scales for NVFP4 W4A4.
  - (c) Per-expert and per-tensor Σx² per input column, which is the GGUF imatrix.

  This saves a second 763B forward for each artifact.
- Engram inside the streamed loop: before running layer 1 and layer 14, add `gate·value` computed from the cached gathered rows (6.3 KB per token per layer) and the layer's small `wkv` / norm weights, which are in the backbone shards. The gate needs the current hidden state, so only the row gather is precomputed, not the output.

### Gaps
- No measured V4.1 forward throughput on sm_110a with any stack. LibertAI's 0.65 GiB/s is an I/O rate on a different box.
- Whether the reference `fp4_gemm` / `fp4_act_quant` and sparse-indexer kernels build on sm_110a is unknown; LibertAI needed kernel patches on GB10.
- The calibration budget for 50% REAP on a 384-expert, top-6 model has no published dose-response. The local GLM dose-response notes cover a different model.

---

## Q3. Healing: block-wise / per-expert healing, router recalibration with MXFP4 experts, time; KD from cached teacher logits

### Takeaway
Gradient training of MXFP4 experts is not possible: no framework trains FP4, and a bf16 upcast loses the bit-exact MXFP4 property and needs a requant. What works on Thor is layer-local:
- Closed-form, gradient-free routing fixes (ExFold-style scalar folds; refitting the noaux_tc bias on the pruned support). These need ~10^5 tokens and stream layer by layer.
- Block-wise training of router and non-expert weights (attention, shared expert, LoRA), with the teacher layer streamed beside the student.

The block-wise route is estimated at **~1-2 h per 2M tokens and 4-10 h per 8M tokens** [EXTRAPOLATION], at ~75 GB resident. No software for this exists; it must be built. KD on cached teacher logits is cheap to store (36 B per token) but needs an end-to-end student forward and backward, which does not fit for a ~150 GiB student.

### Cited Findings
- "Nothing published does LoRA on a >100B MoE streamed from NVMe on one device." Block-wise variant: "~0.5-1.1 h per 1M tokens at 2 epochs/block, i.e. 2M tokens in 1-2 h and 8M in 4-9 h"; "~75 GB resident"; "no framework streams FP4 MoE experts from NVMe on aarch64" [extrapolation in source] — [LOCAL layer_streamed_healing_infra.md](file:///home/patrickd/glm-5.3-reap/research_notes/Compressing%201T%20MoE%20for%20Thor/layer_streamed_healing_infra.md)
- Ranked healing options per Thor-hour:
  - (1) ExFold-style routing-level scalar folding: "gradient-free, ~10^5 tokens, layer-streamable, +7–11 avg points".
  - (2) Progressive re-ranking at 12.5% per stage.
  - (3) Frozen-expert LoRA plus full-router self-distillation on cached teacher logits, which "requires a resident student".
  — [LOCAL healing_recovery.md](file:///home/patrickd/glm-5.3-reap/research_notes/Compressing%201T%20MoE%20for%20Thor/healing_recovery.md)
- The published recipe converges on "LoRA rank 16–64 on attention (and shared-expert) projections, routed experts frozen, router either frozen or trained in full, 1–8M tokens, one epoch" — [LOCAL healing_recovery.md](file:///home/patrickd/glm-5.3-reap/research_notes/Compressing%201T%20MoE%20for%20Thor/healing_recovery.md)
- "currently no framework supports fp4 or MXFP4 training" (Unsloth). The documented paths are upcast→SFT→QAT, or QLoRA — [LOCAL healing_recovery.md](file:///home/patrickd/glm-5.3-reap/research_notes/Compressing%201T%20MoE%20for%20Thor/healing_recovery.md)
- Cached targets: "36 B/token for 12 importance-sampled logits" vs hidden-state targets at "10 KB/token/layer" — [LOCAL layer_streamed_healing_infra.md](file:///home/patrickd/glm-5.3-reap/research_notes/Compressing%201T%20MoE%20for%20Thor/layer_streamed_healing_infra.md)
- The GLM pipeline shipped only a first-moment `down_proj` gain correction. Full layer-local distillation was judged "not tractable" for a ~165B student — [LOCAL s05_heal.py](file:///home/patrickd/glm-5.3-reap/scripts/stages/s05_heal.py)
- V4.1 router facts:
  - `scores = linear(x, W)/gate_temp`, `sqrtsoftplus`.
  - The bias selects top-k and the unbiased scores weight it.
  - Gates are renormalised and multiplied by `route_scale` 1.5.
  - The `bias_vl` variant applies to image tokens.
  - "No published result on re-fitting noaux_tc biases after REAP; LibertAI and 0xSero sliced without refitting."
  — [LOCAL v41 note](file:///home/patrickd/glm-5.3-reap/research_notes/Compressing%201T%20MoE%20for%20Thor/v41_architecture_pruning_delta.md); [LOCAL memory noaux-tc-router-formula](file:///home/patrickd/.claude/projects/-home-patrickd/memory/noaux-tc-router-formula.md)
- The shared expert is built with `dtype=None` (not fp4), and routed experts are `float4_e2m1fn_x2` with an E8M0 `.scale` of shape `[out, in//32]` — [inference/model.py](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/raw/main/inference/model.py) via LOCAL v41 note

### Inferences
- **Trainable on Thor without breaking MXFP4:**
  - Router `gate.weight` rows, `gate.bias` and `bias_vl`, which are non-FP4.
  - The shared expert and attention projections (FP8/bf16 source precision).
  - LoRA adapters merged only into non-FP4 tensors.

  Routed experts stay byte-identical, which keeps the "bit-exact expert deletion" claim true for the MXFP4 artifact.
- [Inference] The GLM s05 per-layer `down_proj` gain does **not** port to MXFP4 bit-exactly. A gain can be applied exactly only as a power of two, by shifting the E8M0 exponent. Any other gain means requantising the experts, or folding the gain into a non-FP4 tensor. One candidate is scaling the router rows, but the score passes through sqrtsoftplus and renormalisation, so that is not a linear gain. Recommendation: express first-moment correction through the shared-expert or `hc_post` path, or accept a power-of-two gain, and test it.
- Bias refit ([inference]): after slicing, re-fit `gate.bias` on the pruned support so the per-expert load the bias was trained for is restored. This is a closed-form or short iterative fit on the routing statistics already captured in the saliency pass, at seconds-to-minutes cost.
- Block-wise healing streams the teacher layer beside the student layer, so the teacher forward is recomputed rather than cached (hidden-state caching would be ~10 KB per token per layer). Disk is not a constraint for this, since the student is local. The teacher layer must be re-downloaded (7.4 GB per layer) unless a second drive holds the backbone, which adds the ~49 min download per pass.
- DGX Spark or cloud adds little here. The student does not fit a single GB10 either. An 8×H100/B200 node could run a resident bf16 LoRA KD, but that means uploading ~300 GB, and no time or price evidence was found.

### Gaps
- No measured quality for layer-local healing on an expert-pruned MoE, and no V4.1 healing precedent of any kind.
- No measured effect of re-fitting noaux_tc biases after REAP.
- No cited cloud GPU price list was retrieved, so cost comparison is a gap.

---

## Q4. Export: (a) pruned MXFP4 safetensors, (b) NVFP4 requant, (c) GGUF with trellis/i-quants plus imatrix

### Takeaway
- **(a) MXFP4 safetensors.** This is a byte copy of the kept experts' packed E2M1 and E8M0 tensors, plus slicing the rows of `gate.weight`, `gate.bias` and `gate.bias_vl`, plus setting `n_routed_experts` to a vLLM-legal count: 320, 256, 192 or 128. A 50% prune therefore lands on 192, not a free choice. The result is ~152 GiB without Engram [DERIVED].
- **(b) NVFP4.** NVIDIA's ModelOpt 0.47.0rc0 (with V4.1 patches) converted every MXFP4 weight block to NVFP4 **losslessly**. Only the W4A4 activation scales need calibration. The repack can be done per tensor with the local `s07` streamer, giving ~160 GiB [DERIVED].
- **(c) GGUF.** It depends on llama.cpp PR #28696, which is **still open, unmerged**, and has open correctness issues. The converter streams with <8 GB peak RAM, carries the experts as MXFP4, and includes the Engram tables. Trellis types (IQ*_KT) exist only in ik_llama.cpp, whose V4.1 path has an open NaN bug on sm_12x and no Thor report. The imatrix must come from our own streamed statistics, or from a heavily quantised vehicle, because the pruned model does not fit in RAM at native precision.

### Cited Findings
- **(a) Surgery and config:**
  - LibertAI sliced "128 experts per layer, along with their rows in `gate.weight`, `gate.bias`, and `gate.bias_vl`".
  - 475.2 → 385.5 GiB at 256 experts. Experts stay MXFP4, dense stays FP8, and Engram is excluded from the repo and byte-identical to base. MTP/DSpark was kept unpruned at 3 × 128 experts.
  — [LibertAIDAI/DeepSeek-V4.1-Flash-REAP-256E](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-REAP-256E) via LOCAL v41 note
- **(a) The prune is written and verified by:** `prune_convert_mp.py --keep N --mp 2`, `verify_shards.py`, and `verify_prune.py`, which checks the written checkpoint matches the source at kept indices; experts are "not re-quantized" — [Libertai/deepseek-v41-flash-reap](https://github.com/Libertai/deepseek-v41-flash-reap)
- **(a) Legal expert counts are a vLLM kernel constraint:**
  - The fused `topk_softplus_sqrt` kernel accepts only {1,2,4,8,16,32,64,128,192,256,320,384,448,512,576}.
  - LibertAI got "Unsupported expert number: 272".
  - The DSv4 fast kernel runs only for 256 or 384.
  - 0xSero's V4-Flash K160 needed a torch-fallback patch.
  — [LOCAL v41 note](file:///home/patrickd/glm-5.3-reap/research_notes/Compressing%201T%20MoE%20for%20Thor/v41_architecture_pruning_delta.md); [vLLM topk_softplus_sqrt_kernels.cu](https://raw.githubusercontent.com/vllm-project/vllm/main/csrc/moe/topk_softplus_sqrt_kernels.cu)
- **(a) Measured GLM surgery precedent:** "Nothing is dequantised… removing an expert is deleting six tensors… bit-identical". It used a completed-shard ledger, deleted source shards progressively, and kept the MTP block with experts ranked by weight norm because `num_local_experts` is a single scalar — [LOCAL s04b_surgery.py](file:///home/patrickd/glm-5.3-reap/scripts/stages/s04b_surgery.py)
- **(b) NVIDIA DeepSeek-V4.1-Flash-NVFP4:**
  - Built with nvidia-modelopt v0.47.0rc0 "with V4.1-specific compatibility changes".
  - Routed experts went MXFP4 → NVFP4 W4A4, group 16; everything else kept source precision.
  - Calibration was 512 cnn_dailymail + 512 Nemotron-Post-Training-v2 samples at seq 512, batch 4.
  - Size ~476 → ~492 GiB, or ~400 GiB with the Engram tables stored as MXFP4 (lossy: "about 12% relative L2 error on the table").
  - "all 16,986,931,200 weight blocks passed lossless conversion"; "18 of 46,080 projection entries used fallback activation scales".
  - Tested on 4× GB300; current revision needs vLLM PR #61530; licence MIT.
  — [nvidia/DeepSeek-V4.1-Flash-NVFP4](https://huggingface.co/nvidia/DeepSeek-V4.1-Flash-NVFP4). A news writeup ([datanorth.ai](https://datanorth.ai/news/nvidia-releases-deepseek-v4-1-flash-nvfp4)) agrees on MIT and the size; NVIDIA's NGC catalog page reportedly says the NVIDIA Open Model Agreement instead ([NGC](https://catalog.ngc.nvidia.com/orgs/nim/deepseek-ai/models/deepseek-v4.1-flash/-)). This is a licence conflict across NVIDIA surfaces.
- **(b) GLM NVFP4 precedent on this Thor.** Every model-level route failed: `device_map="cpu"` exceeds RAM; `device_map="auto"` hits `cudaErrorIllegalAddress` on unified memory; llm-compressor `oneshot` with disk offload asserts on meta. So it "streams safetensors shard by shard" with `scripts/nvfp4_tensor.py`, which is "verified bit-identical to compressed-tensors' own compressor" — [LOCAL s07_quantize.py](file:///home/patrickd/glm-5.3-reap/scripts/stages/s07_quantize.py)
- **(b) vLLM on Thor:** runs NVFP4 via cutlass MoE, and MXFP4 via Marlin only — [LOCAL thor_serving_performance.md / extreme_quantization.md](file:///home/patrickd/glm-5.3-reap/research_notes/Compressing%201T%20MoE%20for%20Thor/extreme_quantization.md)
- **(c) llama.cpp PR #28696 "model : add DeepSeek V4.1 (DeepseekV41ForCausalLM)":**
  - Open; opened 2026-09-10, ready for review 09-22, last activity seen 10-10, not merged.
  - Arch is `deepseek41`. It adds `ENGRAM_EMBD/Q/K/WKV`, with tables "row-streamed into a disk-backed memmap", and nine engram metadata keys.
  - Routed experts come out as MXFP4 via the inherited V4 path.
  - "121 GiB of RAM… Peak host memory stayed under 8 GB above baseline."
  - Full-model Q8_0 staging is 507.9 GB; Q2_K is 246.3 GiB; Q3_K_M is 323.4 GiB.
  - "Q2_K_S and IQ2_M both refuse to build without an imatrix."
  - Known issues: the two-level candidate mask is not implemented (~16K context cap); MTP and vision are unmapped; there is a CUDA MMQ out-of-bounds read; reused-index sparse attention may attend all rows beyond 512; hyper-connection tensors need F32.
  — [llama.cpp PR #28696](https://github.com/ggml-org/llama.cpp/pull/28696)
- **(c) Local llama.cpp master** (465e49b9c, 2026-09-07) has `deepseek4` but no `deepseek41` — [LOCAL v41 note](file:///home/patrickd/glm-5.3-reap/research_notes/Compressing%201T%20MoE%20for%20Thor/v41_architecture_pruning_delta.md)
- **(c) ik_llama.cpp:**
  - Trellis types IQ1_KT through IQ4_KT, with CUDA in PR 113.
  - Open issue #2519, "DeepSeek-V4.1-Flash: NaN logits / illegal memory access when MXFP4 routed experts are offloaded to CUDA (sm_12x)".
  - Zero Thor issues.
  — [LOCAL thor_serving_performance.md](file:///home/patrickd/glm-5.3-reap/research_notes/Compressing%201T%20MoE%20for%20Thor/thor_serving_performance.md); [ik_llama.cpp mirror README](https://github.com/RodriMora/ik_llama.cpp)
- **(c) The web search found no source confirming ik_llama V4/V4.1 trellis quants.** antirez's V4-Flash fork is CPU/Metal only — [sourcepulse antirez fork](https://www.sourcepulse.org/projects/29658251)
- **(c) Double-quant penalty** for re-quantising already 4-bit (INT4-QAT) experts, Kimi-K2 proxies (extra percentage points vs a BF16 source):

  | Quant | Extra penalty |
  |---|---|
  | smol-IQ2_KL | ~2.4 pp |
  | IQ2_KL | ~6 pp |
  | smol-IQ1_KT | ~30 pp |

  — [LOCAL extreme_quantization.md](file:///home/patrickd/glm-5.3-reap/research_notes/Compressing%201T%20MoE%20for%20Thor/extreme_quantization.md)
- **(c) GLM imatrix precedent.** Streaming Q8_0 at ~84 MB/s would be "~35 min/chunk against the ~100 chunks", so a Q4_K_M vehicle that fits in RAM (92.5 GiB) was used, with held-out text, `-c 512 -ngl 0`. The `--save-frequency` checkpoints filled the disk, and rc=0 hid a truncated `.dat` (`imatrix_verify.py` now checks for this) — [LOCAL gguf_imatrix.sh](file:///home/patrickd/glm-5.3-reap/scripts/gguf_imatrix.sh); [LOCAL memory llama-imatrix-save-frequency-trap](file:///home/patrickd/.claude/projects/-home-patrickd/memory/llama-imatrix-save-frequency-trap.md)
- **(c) GLM GGUF disk precedent.** "Q8_0 intermediate is 175 GB… each 4-bit level is ~87 GB… Nothing can be built until the previous level has been pushed and removed" — [LOCAL gguf_ship.sh](file:///home/patrickd/glm-5.3-reap/scripts/gguf_ship.sh)
- **(c) Only one EOS id fits a GGUF header.** Dropped turn terminators need `--override-kv` — [LOCAL memory gguf-multi-eos-token-trap](file:///home/patrickd/.claude/projects/-home-patrickd/memory/gguf-multi-eos-token-trap.md)

### Inferences
- [DERIVED] **Artifact sizes at 192 experts:**

  | Artifact | Expert bytes | Total |
  |---|---|---|
  | MXFP4 | 268.95/2 = 134.5 GiB | + 9.77 floor + 7.39 DSpark = **~151.6 GiB** (162.8 GB) without Engram |
  | NVFP4 | 0.5625 B/param × 13.59B params/layer × 20 = 142.4 GiB | **~159.6 GiB** |
  | Either, Engram bundled | | **+189.1 GiB** FP8, or +97.3 GiB MXFP4 |

  The NVFP4 expert figure scales consistently with NVIDIA's +16 GiB at 384 experts. Engram is better left as a pointer to base shards 47 and 48, as LibertAI did.
- **The surgery runs as a streaming stage (GLM s04b pattern):**
  1. Download shard L (~7.4 GB).
  2. Copy the kept experts' tensors, renumbered.
  3. Slice the router rows.
  4. Write the shard.
  5. Delete the source shard.

  Peak disk ≈ 7.4 GB source + the growing output (~163 GB final). So the MXFP4 output alone exceeds today's 102 GB free; **it must be uploaded shard-by-shard and deleted, or a second drive is needed.** [DERIVED] Time ≈ 49 min of download + ~6 min of writes at 487 MB/s.
- **Config edits for surgery:**
  - `n_routed_experts` → 192.
  - Keep `num_experts_per_tok` 6, `routed_scaling_factor` 1.5 and `norm_topk_prob` unchanged.
  - Leave the DSpark/MTP pools unpruned, as LibertAI did. Otherwise rank them by weight norm as in GLM. Note that a single scalar expert count could force the MTP layers to 192 too: check whether DSpark has its own config field.
  - Regenerate `model.safetensors.index.json`.
  - Rebuild `verify_prune.py`-style hash checks.
- **NVFP4 on Thor needs no ModelOpt at all.**
  - The conversion is the lossless repack NVIDIA demonstrated: each MXFP4 32-block with an E8M0 scale becomes two 16-blocks with an E4M3 scale and a per-tensor FP32 global scale. It is exact whenever the tensor's E8M0 exponent spread fits E4M3's range. [inference about the mechanism; NVIDIA reports 100% lossless]
  - It runs tensor-by-tensor in the s07 streamer: ~160 GiB written; minutes of compute; download-bound at ~49 min unless the MXFP4 output is still local.
  - Activation `input_scale` comes from the per-expert amax captured in the saliency pass. Pruning concentrates tokens, which should reduce NVIDIA's 18 fallback-scale cases.
  - The output format should mirror NVIDIA's `hf_quant_config.json` / modelopt layout, so that vLLM's `deepseek_v41` loader accepts it. Diff against the NVIDIA repo's config.
- **GGUF route** (the hardest of the three; time and disk both bind):
  1. **Convert.** Apply PR #28696 locally (`gh pr checkout`) and run `convert_hf_to_gguf` on the pruned MXFP4 dir. Experts pass through as MXFP4. Engram tables must be inside the GGUF unless split loading is supported, which gives a single GGUF of ~152 GiB + ~189 GiB FP8 Engram ≈ 340 GiB [DERIVED]. **That does not fit the disk without a second drive.** Option: a pre-shrunk Engram (the MXFP4 table variant from NVIDIA's current revision), at a quality risk.
  2. **imatrix.** The pruned model at MXFP4 is ~152 GiB plus Engram, which does not fit in RAM; `-ngl 0` streaming from NVMe costs ~30 min per chunk, as on GLM. Two alternatives:
     - **(i)** Write the imatrix ourselves from the streamed reference forward. The imatrix is per-tensor Σx² per input column, and for `_exps` tensors per expert. Use the legacy `.dat` layout already understood locally (`imatrix_verify.py`). Run it on held-out text in a dedicated ~100-chunk pass (~0.3-0.5 h at the extrapolated rate), keeping the GLM rule that imatrix data ≠ REAP data.
     - **(ii)** Build a Q2_K vehicle (~2.6-2.9 bpw → ~90-100 GiB experts + floor), which barely fits, and run `llama-imatrix` with `-ot engram=CPU`. On Thor "CPU" is the same pool, so Engram must be mmap'd from disk; the local finding is that mmap drags 64 KiB per touch.

     Option (i) is cheaper and more faithful.
  3. **Quantise.** Run `llama-quantize` (i-quants) or ik_llama `quantize` (IQ*_KT) streaming from the MXFP4 GGUF. This is CPU-bound, and the duration for ~150B params was not found. The quality floor is double-quantisation.
  4. **Ship and delete each level serially**, as in GLM's `gguf_ship.sh`.

  Since trellis types are ik_llama-only and that tool has an open V4.1 NaN bug and no Thor runs, ship mainline i-quants (IQ2_M/IQ3_XXS) as the primary GGUF and trellis as experimental.
- **Cheapest offload for GGUF:** a rented x86 box with ≥512 GB RAM and ≥1.5 TB disk can convert, run imatrix at RAM speed, and quantise all levels without the disk juggling. This is the standard ubergarm-style workflow for 1T models.

### Gaps
- Whether llama.cpp/ik_llama can load Engram tables from a separate GGUF file (split) is unknown.
- `llama-quantize` wall time for IQ*_KT on ~150B params was not found.
- ik_llama.cpp V4.1 arch support status is not confirmed beyond the open issue title.
- The NVIDIA NGC vs HF licence statement conflict (MIT vs NVIDIA Open Model Agreement) is unresolved. It matters only if the NVFP4 artifact were derived from NVIDIA's repo; deriving from the DeepSeek MIT source avoids it.

---

## Q5. Hugging Face publishing: layouts, cards, MIT attribution, size limits, detached uploads

### Takeaway
HF has no per-repo size cap. Recommended file size is <200 GB (500 GB hard limit), <10k entries per folder, and <100 files per commit, with `upload_folder` and `hf upload` auto-splitting. Public storage is "best-effort" free, and PRO includes up to 10 TB.

Locally measured: folder uploads reach ~118 MB/s. A single-file Xet upload crawls at ~400 KB/s unless `HF_XET_FIXED_UPLOAD_CONCURRENCY=8`, which gave ~110 MB/s. `HF_XET_HIGH_PERFORMANCE=1` OOMed the box.

Publish three repos, or one repo with sub-dirs per pass. Exclude the Engram tables (point to base 1017 shards 47 and 48). Copy DeepSeek's MIT `LICENSE`.

### Cited Findings
- HF repository limits:
  - Files per repo <100k.
  - Entries per folder <10k.
  - File size <200 GB recommended; "no single file will exceed 500GB".
  - Commit size <100 files; "`upload_folder`/`hf upload` auto-split large folders into multiple commits".
  - "There is no per-repo size limit… uploads count against your account's total storage quota".
  - Free public storage is "Best-effort"; PRO gets "Up to 10TB included"; private free storage is 100GB.
  — [HF Hub storage limits](https://huggingface.co/docs/hub/storage-limits)
- Local upload measurements: "a lone 93 GB GGUF crawled at ~400 KB/s"; FP8/NVFP4 folder uploads hit "~118 MB/s"; "HF_XET_HIGH_PERFORMANCE=1 fixed the speed (391 KB/s -> 110 MB/s) and then OOMED THE BOX"; fixed with `HF_XET_FIXED_UPLOAD_CONCURRENCY=8` and a 2 GiB chunk cache — [LOCAL gguf_ship.sh](file:///home/patrickd/glm-5.3-reap/scripts/gguf_ship.sh)
- Unattended jobs must be `setsid`/systemd-detached, never children of a Claude Code Bash call ("SSH drop cost 16h") — [LOCAL memory detach-unattended-local-workloads](file:///home/patrickd/.claude/projects/-home-patrickd/memory/detach-unattended-local-workloads.md)
- The source repo ships a 1,084-byte `LICENSE` — [HF API tree](https://huggingface.co/api/models/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/main). The NVIDIA derivative card states MIT and notes "NVIDIA did not own or develop the model" — [nvidia/DeepSeek-V4.1-Flash-NVFP4](https://huggingface.co/nvidia/DeepSeek-V4.1-Flash-NVFP4). LibertAI: "Tooling: MIT. The model and its bundled reference implementation keep their own license" — [Libertai/deepseek-v41-flash-reap](https://github.com/Libertai/deepseek-v41-flash-reap)
- LibertAI's card limitations: "Calibration is small, and perplexity is the only metric"; "No downstream benchmarks were run" — [LibertAI REAP-256E card](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-REAP-256E) via LOCAL v41 note
- GLM card lesson: card quality numbers must be filled "from the artifacts, never from literals" (pass-1 numbers printed on pass-2 would be "a confident, specific, wrong claim") — [LOCAL s06_emit.py](file:///home/patrickd/glm-5.3-reap/scripts/stages/s06_emit.py)

### Inferences
- **Layout.** Use three repos, following GLM's practice: `…-REAP{N}-MXFP4`, `…-REAP{N}-NVFP4`, `…-REAP{N}-GGUF`.
  - Within each, use sub-dirs per pass (e.g. `pass1/`, `pass2-hope/`), or tag revisions, so that published A/B baselines are never overwritten. The GLM s06 comment explicitly forbids emitting into a populated tree.
  - GGUFs go in per-level sub-dirs, split with `gguf-split` to under 50 GB per part. That is well below 200 GB and keeps single-file Xet stalls contained.
- **Engram.** Do not re-host it: it is 189 GiB of byte-identical upstream data. The card gives the `hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 --include "model-00047*" "model-00048*"` instruction and warns that the 0909 tables are incompatible.
- **Licence and card.** Copy `LICENSE` verbatim. State "derived from deepseek-ai/DeepSeek-V4.1-Flash-1017 (MIT)" plus the prune method. The card should list:
  - The legal expert-count constraint.
  - Serving flags: vLLM `--enforce-eager`, TRITON/FlashMLA notes.
  - That the GGUF depends on an unmerged PR.
  - What was measured (PPL plus a downstream set), unlike LibertAI.
- **Upload time [DERIVED]** at ~110 MB/s:

  | Artifact | Time |
  |---|---|
  | MXFP4 (163 GB) | ~25 min |
  | NVFP4 (171 GB) | ~26 min |
  | Each GGUF level (~60-100 GB) | 10-15 min |

  Run the upload stage as its own detached systemd unit, with upload-then-delete per shard while disk is tight.

### Gaps
- No HF policy text on re-hosting very large derivative models on free accounts beyond "best-effort". At ~0.5 TB total across three repos, PRO may be prudent.

---

## Q6. Existing open-source pipelines for DeepSeek-V3/V4-class REAP and their compute

### Takeaway
Cerebras REAP does **not** support V4 or V4.1 ("DeepSeek-V3.2 (planned)"); it would need a `MODEL_ATTRS` entry and cannot load V4.1 through transformers anyway.

The only working V4.1 pipeline is **LibertAI's `deepseek-v41-flash-reap`**. It provides:
- Reference-implementation layer streaming at 19.4 GiB peak on one 120 GB unified-memory box.
- Engram mmap gather.
- Bit-exact MXFP4 slicing and verification.

It is the natural donor. 0xSero's V4-Flash-0731 REAP is a V4 precedent. Neither publishes wall-clock times.

### Cited Findings
- REAP repo families: DeepSeek-V3.2 (planned), GLM-4.5-Air/4.6, Qwen3, Mixtral, Llama-4-Scout, ERNIE, Kimi-K2, Kimi-Linear, MiniMax-M2. There is no V4/V4.1 and no mention of hyper-connections or Engram. Registration is via `MODEL_ATTRS` in `src/reap/model_util.py` — [CerebrasResearch/reap](https://github.com/CerebrasResearch/reap) via LOCAL v41 note
- LibertAI's pipeline:
  - Steps: `patch_kernels.py`, `patch_engram.py`, `build_calib.py` (128 × 2048), `build_calib_vl.py` (256 images), `reap_calibrate.py`, `eval_ppl.py --prune-keep` (in-memory prune eval), `prune_convert_mp.py --keep 272 --mp 2`, `verify_shards.py`, `verify_prune.py`.
  - Hardware: "one 120 GB unified-memory box" for calibration, 2× GB10 for serving.
  - Wall-clock times "Not reported".
  — [Libertai/deepseek-v41-flash-reap](https://github.com/Libertai/deepseek-v41-flash-reap)
- LibertAI quality on 0909: text PPL +5.49% at 256 experts and +14.66% at 192. Vision collapses below 272 under text-only ranking — [LOCAL v41 note](file:///home/patrickd/glm-5.3-reap/research_notes/Compressing%201T%20MoE%20for%20Thor/v41_architecture_pruning_delta.md), citing the LibertAI card
- LibertAI serving: TP=2 on 2× GB10, `--enforce-eager` mandatory, 32K context, DSpark k=5, 19-49 tok/s — [LOCAL v41 note](file:///home/patrickd/glm-5.3-reap/research_notes/Compressing%201T%20MoE%20for%20Thor/v41_architecture_pruning_delta.md)
- 0xSero V4-Flash-0731 REAP K160 "preserves top-6 routing, shared experts, attention, embeddings, output head, mHC, compressor, and indexer tensors"; rankings were transferred from an earlier run; vLLM needed a router torch-fallback patch for 160 experts — [0xSero/DeepSeek-V4-Flash-0731-REAP](https://huggingface.co/0xSero/DeepSeek-V4-Flash-0731-REAP)
- The local vLLM nightly aarch64 image (0.32.0rc1) ships `models/deepseek_v41/` with the router kernel compiled for SM 110 — [LOCAL v41 note](file:///home/patrickd/glm-5.3-reap/research_notes/Compressing%201T%20MoE%20for%20Thor/v41_architecture_pruning_delta.md)
- NVIDIA ModelOpt 0.47.0rc0 with V4.1 patches produced the NVFP4 build on 4× GB300 — [nvidia/DeepSeek-V4.1-Flash-NVFP4](https://huggingface.co/nvidia/DeepSeek-V4.1-Flash-NVFP4)

### Inferences
- **Build plan:** reuse the local orchestrator stages (s00-s09, ledgered, resumable) and port the model-specific bodies from LibertAI's scripts:
  - **s01 source:** becomes a per-shard downloader, not a full stage.
  - **s03 saliency:** gets the reference `Block`, layer-major ordering, `shared_attn` carry, the Engram pre-hook and the extra captures.
  - **s04b surgery:** adds `gate.bias_vl` and the legal-count assertion.
  - **s05 heal:** limited to router/bias/non-expert changes.
  - **s07 quantize:** switches to the MXFP4→NVFP4 repack.
  - **GGUF scripts:** move to PR #28696 plus a self-written imatrix.
- **Where a DGX Spark helps:**
  - **(i)** Data-parallel saliency: split the token set, sum the accumulators. This halves the ~3-7 h sweep, but needs its own shard download.
  - **(ii)** Its own disk can stage the Engram shards and precompute rows.
  - **(iii)** It is the only tested V4.1 serving platform (GB10 TP=2).

  It does not solve the RAM limit for healing or imatrix, being the same 128 GB class.
- **Where cloud helps most:**
  - A CPU-only, big-RAM, big-disk box for the Engram-row precompute and the GGUF convert/imatrix/quantise.
  - Optionally one 8-GPU node for resident KD healing.

### Gaps
- No public wall-clock for any V4.1 REAP pass, LibertAI's included.
- No public pipeline does HOPE (vs REAP) on V4/V4.1.

---

## Stage table (summary for the report writer; every time and peak is [DERIVED] or [EXTRAPOLATION] unless marked)

| # | Stage | Tooling | Compute time | RAM peak | Disk peak (free 102 GB today) | Fits one Thor? / cheapest workaround |
|---|---|---|---|---|---|---|
| 0 | Index fetch + smoke on 1 layer (layer 3) | reference `inference/model.py` + LibertAI `patch_kernels` | <1 h | ~28-47 GiB | 7.4 GB | Yes. Measure tok/s and peak here before anything else. |
| 1 | Corpus build (1-2M tok, 8k seq, interleaved) | local `s02_corpus` | hours (CPU) | small | few GB | Yes |
| 2 | Engram row precompute (2 tables × 101.5 GB) | own gather script | ~32 min download + minutes | small | 101.5 GB per shard: **does not fit** | Rent a VM or Spark with ≥250 GB disk, or a USB NVMe; ship back 13-25 GB of rows |
| 3 | Layer-major saliency (REAP + HOPE F + teacher top-k logits + amax + imatrix Σx²) | ported s03 | ~3.3 h/1M tok (~6.5 h/2M) + 49 min download [EXTRAP] | ~70-100 GiB (hs resident ≤1.5M tok) | ~7.4 GB shard + rows; or +82 GB if hs paged (2M tok) | Yes, with layer-major order. A Spark data-parallel halves it. |
| 4 | Sweep / select (192 vs 256), masked eval | `eval_ppl --prune-keep`-style, no checkpoint write | 1 held-out forward per ratio (~0.5-1 h each) | as stage 3 | as stage 3 | Yes |
| 5 | Surgery to MXFP4 (bit-exact) | ported s04b + `verify_prune` | ~55 min (download-bound) | <10 GiB | output grows to ~163 GB: **does not fit** | Upload-and-delete per shard, or a second drive |
| 6 | Heal: bias refit + ExFold fold (gradient-free); optional block-wise router/attn/shared LoRA | new code | minutes; block-wise 1-2 h/2M tok, 4-10 h/8M [EXTRAP] | ~75 GB block-wise | needs student shards local (163 GB) | Bias/fold yes. Block-wise needs the student local (second drive). Resident KD only on cloud. |
| 7 | NVFP4 repack (lossless) + input_scale from stage-3 amax | adapted `nvfp4_tensor.py` / s07 | minutes compute; download/IO-bound ~1 h | <10 GiB | ~171 GB output: **does not fit** | Stream: convert, upload, delete per shard |
| 8 | GGUF convert (PR #28696) | `convert_hf_to_gguf` | ~1-2 h IO-bound [EXTRAP] | <8 GB [MEASURED by PR author] | ~340 GiB with FP8 Engram: **does not fit** | Second drive, or a cloud box |
| 9 | imatrix | self-written from streamed forward (held-out ~100 chunks), legacy `.dat` | ~0.5 h + download [EXTRAP] | as stage 3 | small | Yes (option i). `llama-imatrix` on a Q2_K vehicle is the fallback. |
| 10 | Quantise (IQ2/IQ3 mainline; IQ*_KT via ik_llama) | `llama-quantize` / ik `quantize` | unknown (CPU) | streaming | input GGUF + one output level | Serial quantise, gate, push, delete (GLM `gguf_ship.sh`). Big-RAM cloud is easier. |
| 11 | Cards + uploads | `huggingface_hub upload_folder`, `HF_XET_FIXED_UPLOAD_CONCURRENCY=8`, systemd-detached | ~25 min per 160 GB repo | ~2 GiB cache | per artifact | Yes |

Critical-path total on one Thor with a USB second drive [EXTRAPOLATION]: about 2-3 days of wall clock. The parts:
- ~4-8 h of saliency.
- ~4 × 49 min of downloads.
- Healing at 0.1-10 h depending on depth.
- GGUF convert/quantise as the largest unknown.

Without a second drive, the same work is feasible only as upload-and-delete streams, plus an off-box Engram precompute and an off-box GGUF build.
