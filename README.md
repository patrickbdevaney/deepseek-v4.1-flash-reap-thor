# DeepSeek-V4.1-Flash-1017 REAP for Jetson AGX Thor (design, not yet built)

> **Status: speculative.** This repo is a research-backed *plan*. No weights have been pruned,
> healed, quantized or served. Every quality and speed number below is either cited from the
> literature or extrapolated, and is marked that way in [`REPORT.md`](REPORT.md).

**Goal:** run `deepseek-ai/DeepSeek-V4.1-Flash-1017` (~763B MoE) memory-resident on a single
Jetson AGX Thor (122 GiB unified memory, ~117 GiB usable, sm_110a).

## Proposed recipe

| component | plan | bytes |
|---|---|---|
| Engram (hashed n-gram table, ~97 GiB at MXFP4) | **off-pool on NVMe** via an O_DIRECT row store (not mmap, not pinned-host) | 0 in pool |
| Routed experts | delete **192 of 384** (bit-exact on MXFP4; 192 is a count vLLM's fused router accepts), block-wise heal, trellis requant survivors to **~2.9 bpw** | bulk |
| Backbone + KV | fixed floor ~9.8 GiB; KV ~890 B/token (1M ctx < 1 GiB) | |
| **Total** | including a 12 GiB runtime reserve | **≈113.7 GiB** |

**Measured locally:** Engram row-store reads on this Thor's NVMe come in at **1.13–1.30 ms/token p50**
(48 concurrent 1 KiB reads), which prefetch can hide.

**Everything else is extrapolated.** In particular:
- 50% is the most expert deletion any *generalist* MoE has survived within 10% of base.
- Requantizing MXFP4-source experts to 2.9 bpw is a double quantization (about +8% PPL, extrapolated).

## Showstoppers (open)
1. **Disk.** The checkpoint is about 475 GiB and Engram needs at least 97 GiB, but the NVMe has about 26 GB free. A second drive is needed.
2. **Runtime.** Nobody has booted V4.1 on sm_110a yet.
3. **Source format.** The experts ship pre-quantized (MXFP4), so only bit-exact operations are free.

## First experiments (cheapest decisive)
1. A 192E mask built with a balanced calibration corpus and per-modality ranking, compared against LibertAI's +14.7% PPL baseline.
2. A 2 h one-layer dequant + grouped-GEMM fwd/dgrad microbench on Thor, to pin down what block-wise healing costs.
3. A top-k 8→6 scan using the k2 renormalization rule.

## Contents
- [`REPORT.md`](REPORT.md): the full synthesis. It also gives verdicts for GLM-5.3-753B (marginal) and MiMo-V2.6-Pro-MOPD (no-go).
- [`notes/`](notes/): 33 research notes. [`notes/INDEX.md`](notes/INDEX.md) gives a one-line result for each note and lists the first-wave claims that were later overturned.

Prior proven work this builds on: [glm-5.3-reap](https://github.com/patrickbdevaney/glm-5.3-reap)
(GLM-5.3-Flash 321B → REAP50 + heal + NVFP4, 93.6 GiB, three passes shipped).
