# DeepSeek-V4.1-Flash-1017 REAP-192 for a single Jetson AGX Thor

Expert-pruned DeepSeek-V4.1-Flash-1017 that runs memory-resident on one 128 GB NVIDIA Jetson AGX Thor.
It keeps 192 of 384 routed experts per layer and quantizes the survivors to about 2.85 bits per weight. The
189 GiB Engram tables are read from NVMe per token, so they need no memory. This repository is the full pipeline and
the record of every measurement and decision behind the released weights.

## Status (2026-10-30)

Research release. The weights are usable for evaluation and experimentation. They are not a drop-in replacement for
the unpruned model: agreement with it is about 80% top-1 on held-out text, and multilingual quality drops the most
(see [Limitations](#limitations)).

| Artifact | Format | State |
|---|---|---|
| [`REAP192-MXFP4`](https://huggingface.co/patrickbdevaney/DeepSeek-V4.1-Flash-1017-REAP192-MXFP4) | Safetensors, MXFP4 experts (bit-exact subset of the source) | Public. The master every other build is made from |
| [`REAP192-Thor`](https://huggingface.co/patrickbdevaney/DeepSeek-V4.1-Flash-1017-REAP192-Thor) | Thor-native layout, Q2_K/Q3_K experts, FP8 dense, 106.9 GiB | Public. Served by the CUDA server [`deepseek-v4.1-flash-cuda-server`](https://github.com/patrickbdevaney/deepseek-v4.1-flash-cuda-server) (source also in [`cuda/`](cuda/)) |
| [`REAP192-GGUF`](https://huggingface.co/patrickbdevaney/DeepSeek-V4.1-Flash-1017-REAP192-GGUF) | GGUF level L3 for llama.cpp, 85.6 GiB in 3 splits, Engram from NVMe | Public, as a near-miss release: KL 0.508 passes (≤ 0.8), top-1 0.7725 is just under the 0.78 floor (95% CI 0.712–0.829); smoke passed. L2 failed both bars and is not published |
| llama.cpp support | [`patrickbdevaney/llama.cpp@deepseek-v4.1-reap-thor`](https://github.com/patrickbdevaney/llama.cpp/tree/deepseek-v4.1-reap-thor) | Text and image input (mmproj), Engram row store, DSpark drafter (confidence-gated), `thor/start.sh`. 16K default, 32K validated |
| NVFP4 dense overlay | `overlay_nvfp4/` in `REAP192-Thor` (3.5 GiB, a separate copy; the MXFP4 master is unchanged) | Public, served by default by the CUDA server |

See [`llamacpp/NEXT.md`](llamacpp/NEXT.md) for what is left on long context.

## Results

The evaluation is a teacher-forced comparison against the unpruned model. It runs on a pre-registered confirm split
of held-out documents that no pruning or quantization decision saw: 26 documents, 185,565 scored tokens. Top-1 is
agreement with the teacher's argmax. kl32 is KL(teacher ‖ model) over the teacher's top 32 tokens. Brackets are 95%
document-cluster bootstrap intervals.

| Model (vs the unpruned teacher) | Experts | top-1 | kl32 | Release gate |
|---|---|---|---|---|
| Teacher (control) | 384 | 1.000 | 0.000 | pass |
| **REAP-192 MXFP4 master** | 192 | **0.798** | **0.441** | pass (amended, see below) |
| **REAP-192 Thor build** (Q2_K/Q3_K) | 192 | **0.776** | **0.519** | pass |
| Random 192-expert keep-set (baseline) | 192 | 0.751 | 0.542 | — |
| REAP-128 (rejected) | 128 | 0.709 | 0.803 | fail |

- **The selection criterion beats random.** Against a uniform random keep-set of the same size, the master has
  +4.7 points of top-1 (McNemar z = 48.4, 185,565 tokens). The paired kl32 difference is −0.10 [−0.22, +0.04],
  which does not exclude zero.
- **Quantization costs less than pruning.** Going from MXFP4 to the Thor build's ~2.85 bpw K-quants costs
  2.2 points of top-1. Pruning 384 → 192 costs about 20.
- **Healing did not ship.** A per-layer heal lowered NLL (−0.056 [−0.123, −0.022]) but moved the argmax away from
  the teacher (McNemar z −3.73). The released weights are unhealed. See [`RESULTS.md`](RESULTS.md).

**Selection.** Fourteen candidate keep-sets were replayed through the full model on a separate screen split
(177,810 tokens). They included HOPE at several protection levels, REAP-sum, routing mass, frequency, three random
draws, multilingual-protected variants and 128-expert variants. The shipped `hope_sum_pf0` had the best overall
kl32 (0.298). All 14 arms are in [`RESULTS.md`](RESULTS.md).

**Serving on Thor** (measured 2026-10-30 through each runtime's `start.sh`, greedy, one stream):

| | CUDA server (`REAP192-Thor`) | llama.cpp fork (`REAP192-GGUF` L3) |
|---|---:|---:|
| Decode, 256-token chat answers (code / explanation / math) | **33.0 / 25.0 / 40.2 tok/s** (DSpark K=3, default); 25.0 plain | 12.3–12.5 tok/s; 16.6–19.9 with `DSPARK=1` |
| Prefill, ~1.8K-token prompt | 398 tok/s | 159 tok/s |
| Prefill, ~6–7K-token prompt | 739 tok/s | 147 tok/s |
| Image input | no | yes |

The CUDA server's decode runs at 78% of the bandwidth roofline for the 8.57 GB it reads per token
([`docs/zenith/LEDGER.md`](docs/zenith/LEDGER.md)). Every speedup since the first working build (8.10 tok/s) kept
greedy ids identical. The history is in [`docs/ZENITH_LOOP.md`](docs/ZENITH_LOOP.md) and [`docs/zenith/`](docs/zenith/).

## Method

```
DeepSeek-V4.1-Flash-1017 (384 experts/layer, MXFP4 experts, 763B)
  │  s02  balanced calibration corpus, 9 domain buckets, licence-filtered
  │  s03  teacher pass on Thor: per-expert REAP saliency + HOPE interaction banks + teacher top-32 cache
  │  s04  14 candidate keep-sets → full-model replay vs teacher → hope_sum_pf0
  │  s05  bit-exact expert deletion on MXFP4                     → REAP192-MXFP4
  │  s06  heal candidate (rejected at s10)
  │  s07  GPTQ K-quant requant, per-layer Q2_K/Q3_K mix           → REAP192-Thor
  │  s09  llama.cpp GGUF ladder (imatrix, L2/L3), Engram row files → REAP192-GGUF
  │  s10  pre-registered gates on the confirm split
  └  s12  publish only what passed
```

- **Selection: HOPE.** Each expert gets a REAP-sum saliency score, and HOPE adds pairwise expert interactions
  (pᵀFp). The criterion is measured against random and frequency anchors, not assumed.
- **Pre-registration.** The gates in [`ops/s10_gates.json`](ops/s10_gates.json) were sha256-locked before the first
  evaluation. Any later change is a dated, attributed amendment in [`DECISIONS.md`](DECISIONS.md).
- **Engram on NVMe.** The Engram n-gram tables (189 GiB) are repacked into O_DIRECT row files of 48 rows per token.
  They are read with io_uring while layer 0 runs, so they never occupy unified memory
  ([`llamacpp/ENGRAM_ROWFILE.md`](llamacpp/ENGRAM_ROWFILE.md)).
- **Execution.** One Thor ran every stage unattended under systemd, one model-loading process at a time
  ([`ops/chain.sh`](ops/chain.sh), [`ops/chain2.sh`](ops/chain2.sh)).

## Getting started

**llama.cpp (GGUF).** Requires a CUDA GPU or unified-memory system with enough memory for the 85.6 GiB L3 GGUF and
its runtime. You also need about 200 GB of NVMe for the Engram row files.

```bash
git clone -b deepseek-v4.1-reap-thor https://github.com/patrickbdevaney/llama.cpp && cd llama.cpp
thor/build.sh
hf download patrickbdevaney/DeepSeek-V4.1-Flash-1017-REAP192-GGUF --local-dir ~/models/DeepSeek-V4.1-Flash-1017-REAP192-GGUF
# Engram row files: build once from two source shards (ops/s09_engram_rowfile.py), then
thor/start.sh            # text + images;  DSPARK=1 thor/start.sh for speculative decoding
```

See [`thor/README.md`](https://github.com/patrickbdevaney/llama.cpp/blob/deepseek-v4.1-reap-thor/thor/README.md) on the
branch and the GGUF model card for the row-file commands and settings.

**Thor CUDA server (faster, text only).** It lives in
[`patrickbdevaney/deepseek-v4.1-flash-cuda-server`](https://github.com/patrickbdevaney/deepseek-v4.1-flash-cuda-server),
split from [`cuda/`](cuda/) here:

```bash
git clone https://github.com/patrickbdevaney/deepseek-v4.1-flash-cuda-server && cd deepseek-v4.1-flash-cuda-server
hf download patrickbdevaney/DeepSeek-V4.1-Flash-1017-REAP192-Thor --local-dir ~/models/DeepSeek-V4.1-Flash-1017-REAP192-Thor
scripts/build_v41.sh v41_server && ./start.sh      # same Engram row files as llama.cpp
```

## Compatibility

| Platform | State |
|---|---|
| Jetson AGX Thor (128 GB, `sm_110a`), CUDA 13 | Tested: CUDA engine and llama.cpp |
| Other CUDA GPUs with ≥ 110 GB combined GPU memory, or partial offload | Expected to work with llama.cpp (`CUDA_ARCH=<sm>`); not tested |
| vLLM, SGLang, upstream llama.cpp | Not supported. V4.1 support in upstream llama.cpp is an unmerged PR ([#28696](https://github.com/ggml-org/llama.cpp/pull/28696)); this fork adds nine commits on top of it |

## Limitations

- **Multilingual.** At 192 experts, multilingual agreement drops the most of any domain: top-1 0.52 on the screen
  split, against 0.75–0.99 elsewhere. The keep-set trades it for general and finance text. The s10 gate passed only
  after a dated owner amendment that evaluates beat-random without the multilingual bucket
  ([`DECISIONS.md`](DECISIONS.md) §1).
- **Long context is untested for quality.** No confirm-split bin past 8k tokens was adequately sampled. The CUDA
  engine runs a 64k prefill correctly, but retention at long context is unmeasured. llama.cpp runs 32K with the
  candidate-block mask (`CTX=32768 thor/start.sh`; validated 2026-10-30, non-inferior to no mask), at about 15 minutes
  per 32K-token prompt.
- **Vision.** The llama.cpp fork serves images through the mmproj in the GGUF repository. The encoder was
  golden-tested against the reference, and end-to-end image QA was spot-checked. Image-domain quality after
  pruning has not been benchmarked. The CUDA server is text only.
- **Golden layer gate.** The Thor build's layer-by-layer golden gate failed its pooled near-tie rule
  (46 disagreements, 41 of them near-ties). The owner waived it, and the build shipped on its serving gates
  ([decision record](ops/decisions/2026-10-25-golden-a192-drift-amendment.md)).
- **Evaluation scope.** All quality numbers are agreement with the teacher, not downstream task benchmarks. The
  confirm split rests on 26 documents, so per-domain figures are directional.

## Repository map

| Path | Contents |
|---|---|
| [`RESULTS.md`](RESULTS.md) | Every measured result, rendered from artifacts by `ops/s13_results.py`; no hand-typed numbers |
| [`DECISIONS.md`](DECISIONS.md), [`ops/decisions/`](ops/decisions/) | Each choice, who made it, when, and the evidence |
| [`results/`](results/) | Intermediate data: JSON results, ledgers, keep-sets, masks, gate decisions, logs, corpus provenance; sha256 in [`results/MANIFEST.json`](results/MANIFEST.json) |
| [`ops/`](ops/) | Pipeline stages `s00`–`s13`, pre-registered gates, systemd units |
| [`cuda/`](cuda/) | CUDA engine and OpenAI-compatible server for the Thor build |
| [`llamacpp/`](llamacpp/) | llama.cpp patches, REAP-aware converter, Engram row-file tools, GGUF ladder, follow-on plan |
| [`docs/`](docs/) | Engineering analyses (decode parity with the V4-Flash-0731 engine) |
| [`pipeline-plan/PLAN.md`](pipeline-plan/PLAN.md), [`REPORT.md`](REPORT.md), [`notes/`](notes/) | The 2026-10-19 plan and the research that chose model and recipe; `PLAN.md` wins where they disagree |

## Reproducing

Every stage is resumable and writes its artifacts under `$V41_WORK`. `bash ops/chain.sh` followed by
`bash ops/chain2.sh` runs the exact sequence used for this release. The teacher pass needs the 475 GiB source
checkpoint on local storage. The per-stage logs in [`results/`](results/) record each stage's wall-clock time.

## Contributing and contact

This is a single-maintainer research repository. Issues are welcome for reproducibility problems and errors in
reported numbers. Pull requests are not being accepted yet.

## License

- **Model weights:** MIT, inherited from
  [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017).
- **llama.cpp fork:** MIT.
- **Code in this repository:** MIT ([`LICENSE`](LICENSE)). Vendored third-party code keeps its own license
  (`cuda/include/third_party/httplib.h` and `json.hpp`, both MIT).
- **Calibration data:** licence-filtered to exclude non-commercial and share-alike sources. Per-source licences are in
  the corpus provenance under [`results/`](results/).

## Citation

```bibtex
@misc{devaney2026dsv41reap,
  title  = {DeepSeek-V4.1-Flash-1017 REAP-192 for a single Jetson AGX Thor},
  author = {Devaney, Patrick},
  year   = {2026},
  url    = {https://github.com/patrickbdevaney/deepseek-v4.1-flash-reap-thor}
}
```

This work builds on REAP (router-weighted expert activation pruning, Lasby et al., 2025, arXiv:2510.13999) and HOPE
(interaction-aware expert pruning, arXiv:2609.18916). It also builds on DeepSeek's V4.1-Flash release and on the
llama.cpp V4.1 port in PR #28696. Prior work by the same author:
[glm-5.3-reap](https://github.com/patrickbdevaney/glm-5.3-reap) (GLM-5.3-Flash 321B → 144 experts + NVFP4, 93.6 GiB).
