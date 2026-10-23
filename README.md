# DeepSeek-V4.1-Flash-1017 REAP-192 for Jetson AGX Thor

Run `deepseek-ai/DeepSeek-V4.1-Flash-1017` (a ~763B MoE) memory-resident on one Jetson AGX Thor: 122 GiB unified
memory with about 117 GiB usable, on sm_110a. The method keeps 192 of 384 routed experts per layer, selected with
HOPE/REAP on a balanced nine-bucket calibration corpus. Expert deletion is bit-exact on the MXFP4 master. The quantized
builds are made from that master, and the 189 GiB Engram tables are served from NVMe as O_DIRECT row files.

This repo is the complete record of the run, not only its code:

| | |
|---|---|
| [`RESULTS.md`](RESULTS.md) | every measured result: each candidate keep-set against the teacher per domain, the heal decision, release gates, GGUF ladder, DSpark sidecar. Rendered from the artifacts, so no number is typed by hand |
| [`DECISIONS.md`](DECISIONS.md) | the choices made and who made them, including why the shipped keep-set favours general and finance over multilingual |
| [`results/`](results/) | the intermediate data: every JSON result, ledger, keep-set, mask, drift check and gate decision, the run logs, and the calibration corpus provenance (sources, revisions, licences, per-row token counts). [`results/MANIFEST.json`](results/MANIFEST.json) has the sha256 of each file |
| HF dataset (see [`results/DATASET.json`](results/DATASET.json) once published) | the bulk arrays: per-layer expert saliency and HOPE interaction banks, the teacher routing trace and top-32 cache, heal measurements, and the per-token metrics of every arm |
| [`pipeline-plan/PLAN.md`](pipeline-plan/PLAN.md) | the end-to-end plan the run executed (2026-10-19) and its research notes |
| [`ops/`](ops/) | the pipeline: stages `s00`–`s13`, the pre-registered gates ([`ops/s10_gates.json`](ops/s10_gates.json), sha256-locked before the first evaluation), the systemd units that ran it unattended ([`ops/sd/`](ops/sd/)) |
| [`cuda/`](cuda/) | the purpose-built CUDA runner for the MXFP4/NVFP4 weights on Thor |
| [`llamacpp/`](llamacpp/) | the pinned llama.cpp fork and patches for the GGUF release (V4.1 graph, Engram row store, DSpark sidecar export, context clamp) |
| [`REPORT.md`](REPORT.md), [`notes/`](notes/) | the 2026-10-17 research that chose this model and recipe. Superseded by `PLAN.md` wherever they disagree |

## How it ran

The stages are `s02` corpus, `s03` teacher calibration pass (saliency banks), `s04` candidate keep-sets and full-model
arbitration against the teacher, `s05` bit-exact expert surgery on MXFP4, `s06` heal, `s10` evaluation panels, `s07`
native Thor build, `s09` GGUF ladder, `s11` speculative head and `s12` publish. Each stage is resumable. Model-loading
work runs one process at a time, with a page-cache reclaim and a memory check between units. A repo is made public
only if its pre-registered gate passed. [`ops/chain.sh`](ops/chain.sh) and [`ops/chain2.sh`](ops/chain2.sh) are the
exact sequence.

Every decision is measured as agreement with the unpruned teacher (top-1 and top-32 KL, with row-clustered bootstrap
CIs and paired McNemar tests). Perplexity is not used. Calibration and evaluation rows are disjoint, and the gates were
fixed before results existed.

## Known limitation

At 192 of 384 experts, multilingual text loses more agreement with the teacher than any other domain does. This was
an explicit trade-off: see [`DECISIONS.md`](DECISIONS.md) §1 and the s04 table in [`RESULTS.md`](RESULTS.md).

Prior work this builds on: [glm-5.3-reap](https://github.com/patrickbdevaney/glm-5.3-reap) (GLM-5.3-Flash 321B →
REAP50 + NVFP4, 93.6 GiB).
