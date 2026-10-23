# Decisions

This file records the choices made during the run, and who made each one. A decision made by a pre-registered gate is
recorded in [`ops/s10_gates.json`](ops/s10_gates.json), which was sha256-locked before any s10 result existed (the lock
is in [`results/work/s10/prereg_lock.json`](results/work/s10/prereg_lock.json)). A decision that is the owner's is
marked as one. The numbers referred to here are in [`RESULTS.md`](RESULTS.md), which is rendered from the artifacts.

## 1. Ship `hope_sum_pf0`, with multilingual as a documented weakness (owner, 2026-10-22)

**What happened.** At 192 of 384 experts per layer, every saliency-based keep-set lost multilingual agreement with
the teacher. That includes HOPE, REAP, gate mass and frequency. On the multilingual bucket, no criterion did better than
uniform random keep-sets by any real margin, and most did worse. On every other bucket the criteria beat random by a
wide margin (see the s04 table in `RESULTS.md`). The round-1 winner, `hope_sum_pf0`, was below the pre-registered per-bucket floor (`min_bucket_top1`)
on multilingual and above it everywhere else.

**What was tried.** Round 2 ([commit 1561669](https://github.com/patrickbdevaney/deepseek-v4.1-flash-reap-thor/commit/1561669)) built three more arms. Each one takes the
winner's selection and first protects the top 48, 96 or 144 multilingual experts per layer by gate mass:
`hope_sum_ml48`, `hope_sum_ml96` and `hope_sum_ml144`. Only `ml144` reached the multilingual floor. To do it, it took
those slots from the experts that carry general, finance and science, and every one of those buckets dropped. Its
overall KL to the teacher was worse than the winner's, and its advantage over the random arms was no longer
significant. At 192 slots, multilingual and the other domains compete for the same experts. No arm keeps all of them.

**The choice.** There were two options. One was to ship `ml144` and pass every floor. The other was to ship
`hope_sum_pf0` and accept a lower multilingual score.

**The owner's rationale.** It was a choice. At a given REAP ratio and a given final bits per weight, the owner judged
that general and finance capability, kept at the highest level the expert budget allows, mattered more than a relative,
scalar decline in multilingual capability. The model this release serves is meant to be strongest at general
reasoning and finance first. Multilingual is not abandoned. It is guarded against collapse and stated plainly.

**How it is enforced** ([commit 65e10b4](https://github.com/patrickbdevaney/deepseek-v4.1-flash-reap-thor/commit/65e10b4)):
- `pruned_floor.bucket_overrides.multilingual.min_top1` in `ops/s10_gates.json` replaces the multilingual floor with a
  collapse guard. It was amended before any s10 result and its `why` field carries the measured numbers. Every other
  floor is unchanged.
- `ops/lib/gates.py` reports any arm that passes only under the override as "passes only under its override floor
  (known limitation)". That note is printed in `RESULTS.md`.
- `ops/s12_publish.py` (`limitations_text()`) renders a known-limitation paragraph into every model card. It is built
  from the gate file and the measured multilingual score, so the card can never state a better number than was measured.

**What it means for users.** Expect the model to be weaker at non-English text, Chinese included, than the unpruned
teacher, and weaker than at the other domains in the s04 table. For multilingual work, use the unpruned model or a
keep-set built for it. The s04 artifacts in [`results/work/s04/`](results/work/s04/) contain every arm's full keep-set
and per-bucket metrics, and the saliency banks in the HF dataset (see `results/DATASET.json` once published) let you
build a multilingual-weighted keep-set without re-running the 763B calibration pass.

## 2. The per-layer heal is not shipped (pre-registered gate, 2026-10-22)

s06 fit a per-layer output gain to recover the pruned expert mass. The s10 heal panel compared healed and unhealed
weights on the screen split. The heal lowered NLL but moved the model's top-1 token away from the teacher's more often
than it moved it back, and the gate requires both KL and top-1 not to get worse. The release therefore carries the
unhealed REAP weights, and the NVFP4 repack that would only have restated the MXFP4 master with gains folded in was
skipped. This is the same pattern we saw on GLM-5.3 v2: an improvement in reconstruction does not, by itself,
predict agreement with the teacher.

## 3. Every GGUF level is quantized from the MXFP4 master only (owner, 2026-10-22)

The GGUF ladder is quantized once, from the MXFP4 master, with its own imatrix. It is not derived from the native
Thor build (s07), even where that would be byte-cheaper. The master stays the single source of truth. As a result,
the native build and the GGUF are separate quantized bodies. Each one is evaluated against the same unpruned teacher
on the same rows, and any speculative-decoding drafter is refit per body.

## 4. Distribution (owner, 2026-10-22)

- **Engram row files are not rehosted.** They are a deterministic function of the checkpoint, and one build command
  (`ops/s09_engram_rowfile.py`) produces them, in the same way that a lockfile is published but `node_modules` is not.
- **GGUF ladder.** L2 is the release level. L3 is published beside it as an add-on, only if it passes the same gate.
- **DSpark drafter.** The checkpoint's own drafter is shipped as an opt-in `dspark-*.gguf` sidecar at the GGUF repo
  root, gated on a tensor-exact source check, output identity and acceptance (`dspark_gguf` in the gate file).
- **Runtimes.** A purpose-built CUDA runner serves the MXFP4/NVFP4 weights on Thor. A pinned llama.cpp fork
  ([`llamacpp/`](llamacpp/)) serves the GGUFs. vLLM and SGLang are out of scope.
- **Context clamp in the GGUF fork.** llama.cpp does not yet implement V4.1's candidate-block mask, so patch 0006 caps
  context at the span where the output is exact. The card states the cap.
