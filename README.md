# DeepSeek-V4.1-Flash-1017 REAP-192 for Jetson AGX Thor

Run `deepseek-ai/DeepSeek-V4.1-Flash-1017` (a ~763B MoE) memory-resident on one Jetson AGX Thor: 122 GiB unified
memory with about 117 GiB usable, on sm_110a. The method keeps 192 of 384 routed experts per layer, selected with
HOPE/REAP on a balanced nine-bucket calibration corpus. Expert deletion is bit-exact on the MXFP4 master. The quantized
builds are made from that master, and the 189 GiB Engram tables are served from NVMe as O_DIRECT row files.

## Status (2026-10-26)

| Release | State |
|---|---|
| [MXFP4 master](https://huggingface.co/patrickbdevaney/DeepSeek-V4.1-Flash-1017-REAP192-MXFP4) | **Public.** The bit-exact REAP-192 master that every other build is made from. Its s10 final gate failed as registered and was amended by the owner to exclude multilingual (see [`DECISIONS.md`](DECISIONS.md)) |
| [Thor CUDA build](https://huggingface.co/patrickbdevaney/DeepSeek-V4.1-Flash-1017-REAP192-Thor) | **Public** (106.9 GiB, 49 files). Routed experts are K-quants (Q2_K on 31 layers, Q3_K on 9, GPTQ, ~2.85 bpw); attention and dense weights FP8; Engram rows served from NVMe. Passed the serving gates: decode determinism + consistency, 1M-context state with a 64k prefill, OpenAI-style server. The golden layer-by-layer gate failed its pooled near-tie rule (46 disagreements, 41 near-ties) and was waived by the owner, see [`ops/decisions/2026-10-25-golden-a192-drift-amendment.md`](ops/decisions/2026-10-25-golden-a192-drift-amendment.md) |
| GGUF (llama.cpp) | **In progress.** L2 is being scored against the teacher; L3 is the fallback level. A level ships if it passes the s09 gate; if every level misses only the KL/top-1 floor while its llama-server smoke passes, the closest one ships under an owner near-miss rule and its card says so ([`ops/s09_owner_release.py`](ops/s09_owner_release.py)) |
| NVFP4 dense overlay | **Held.** The V4.1 engine does not load the overlay yet (`DSV4_NVFP4_OVERLAY` is read only by the older V4 engine), so it cannot be gated on Thor |
| DSpark speculative decoding | Not measured yet. The checkpoint's own `mtp.*` drafter is wired into the CUDA engine; the chain benchmarks it after the GGUF release |

**Thor CUDA performance (measured, greedy, no speculation):** 8.10 tok/s decode at short context, 6.59 tok/s at 64k
context; a 64k-token prefill takes about 30 min. This is the first-pass engine at about 40% of the memory-bandwidth
ceiling (~20 tok/s for 11.45 GB read per token). [`docs/DECODE_PARITY_0731.md`](docs/DECODE_PARITY_0731.md) compares it
with the DeepSeek-V4-Flash-0731 server (13.8 tok/s plain, 28.4 tok/s with a tuned DSpark head) and ranks the levers:
the K-quant expert GEMV (27–37% of streaming bandwidth), router/sparse-attention kernels, and the speculative verify
pass. These are engine changes; the published weights do not change.

Engine bugs fixed on the way to the Thor release: the first token after a prefill was read from a stale buffer (wrong
first token, runs disagreed; the consistency gate now checks position 0), and `--no-graph` decoding deadlocked against
the Engram row-store flag wait.

This repo is the complete record of the run, not only its code:

| | |
|---|---|
| [`RESULTS.md`](RESULTS.md) | every measured result: each candidate keep-set against the teacher per domain, the heal decision, release gates, GGUF ladder, DSpark sidecar. Rendered from the artifacts, so no number is typed by hand |
| [`DECISIONS.md`](DECISIONS.md) | the choices made and who made them, including why the shipped keep-set favours general and finance over multilingual |
| [`results/`](results/) | the intermediate data: every JSON result, ledger, keep-set, mask, drift check and gate decision, the run logs, and the calibration corpus provenance (sources, revisions, licences, per-row token counts). [`results/MANIFEST.json`](results/MANIFEST.json) has the sha256 of each file |
| HF dataset (see [`results/DATASET.json`](results/DATASET.json) once published) | the bulk arrays: per-layer expert saliency and HOPE interaction banks, the teacher routing trace and top-32 cache, heal measurements, and the per-token metrics of every arm |
| [`pipeline-plan/PLAN.md`](pipeline-plan/PLAN.md) | the end-to-end plan the run executed (2026-10-19) and its research notes |
| [`ops/`](ops/) | the pipeline: stages `s00`–`s13`, the pre-registered gates ([`ops/s10_gates.json`](ops/s10_gates.json), sha256-locked before the first evaluation), the systemd units that ran it unattended ([`ops/sd/`](ops/sd/)) |
| [`cuda/`](cuda/) | the purpose-built CUDA engine and server for the Thor build (K-quant experts, FP8 dense, Engram NVMe row store, DSpark drafter) |
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
