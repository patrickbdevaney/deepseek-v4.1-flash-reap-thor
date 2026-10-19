# deepseek-v4-flash-0731-cuda

A from-scratch pure-CUDA inference server for **`0xSero/DeepSeek-V4-Flash-0731-REAP`** (K160, native
MXFP4) with embedded **DSpark** self-speculative decoding, MLA + DSA attention, hand-tuned for
**Jetson AGX Thor, `sm_110a`**.

No Python on the hot path. Every kernel gated against a PyTorch oracle before it is trusted.

> ## Why
>
> **Local frontier-adjacent intelligence, on hardware you own, fast enough to do long-horizon
> agentic work unattended.** Neither half is rare alone — frontier capability is available through
> an API, fast local decode is available on small models. **Both at once, on one box, with no
> network in the loop, is not.**
>
> The four categories this engine runs *fastest* — long context 30.77, tool/JSON format 29.98,
> multi-turn 28.97, code edit 26.81 tok/s — are precisely the shapes agentic coding produces.
> Speculation pays where continuation is constrained, and agentic work is constrained by definition.
>
> **Every gain here is lossless**: emitted tokens identical to base AR, checked on every run. That
> invariant is what keeps "fast" and "frontier" from becoming a trade.
>
> **[`NORTH_STAR.md`](NORTH_STAR.md)** — the full argument, including the one number in it that is
> still inherited rather than measured.

---

## Where the programme is (2026-08-22)

The work runs as **four phases**, in this order, on one box that can hold **one model at a time**.
Full text and per-phase gates: [`PRODUCTION_PLAN.md`](PRODUCTION_PLAN.md).

| phase | what it is | state |
|---|---|---|
| **1. draft head + spec decode** | exhaust every acceptance lever before measuring anything | **in flight** — P2.5 landed (+4.13 %), P2.6 running, 2.3 not started |
| **2. prefill to the roofline** | 62.4 tok/s and ~3.3 min TTFT at 12 k was the largest remaining gap | **90.8 tok/s, bit-exact** (+45 %) — attention is now the larger half |
| **3. prefix caching** | prove the OpenAI-compatible server is usable by a real agentic harness | not started |
| **4. the eval battery** | run **once**, at the final configuration | armed, deliberately not started |

**The eval battery is suspended mid-flight and stays that way until phase 3 closes.** That is a
decision, not a stall: the battery is idempotent and resumptive, it costs ~3 days of GPU, and every
day of phase 1–3 makes those 3 days cheaper and the resulting numbers more representative of what
actually ships. Resume with `bash scripts/eval_resume.sh`; full state in
[`evidence/evals/SUSPENDED.md`](evidence/evals/SUSPENDED.md). `dsv4-evalstage.service` exists and is
**deliberately disarmed** so nothing starts it by accident.

**One rule the battery inherits from this suspension:** accuracy rows pool across engine revisions,
throughput rows do not. `tools/stamp_eval_provenance.py` records head, block width and engine rev at
every battery start, and `tools/eval_publish.py` prints that block next to the results, so a reader
can see which rows are poolable.

### Where each row actually stands

| task | row | n | acc | trunc | state |
|---|---|---:|---:|---:|---|
| math500 | low | 100 | 95.0 | 1.0 % | **quotable** |
| humaneval | low | 164 | 95.1 | 3.7 % | **quotable** |
| bfcl | low | 240 | 86.2 | 1.2 % | **quotable** |
| bfcl_live | low | 508 | 78.7 | 1.0 % | **quotable** |
| aime24 | low24k | 60 | 91.7 | 10.0 % | extended and landed; **over the 5 % gate → needs forcing** |
| mmlu_pro | low24k | 141/150 | *76.6* | *1.4 %* | ⚠️ **extension partial** |
| gpqa_diamond | low24k | 147/198 | *93.9* | *0.0 %* | ⚠️ **extension partial** |
| aime25 | low24k | 45/60 | *96.2* | *2.2 %* | ⚠️ **extension partial** — suspended cleanly after item 6/21 |
| scicode | low | 291 | 30.2 | 18.6 % | extension not started |
| lcb | low | 175 | 46.9 | **59.4 %** | extension not started |

> **The three italicised rows are NOT partial results — they are biased ones, and must not be
> quoted.** An extension writes the traces that *terminated* first and continues the truncated ones
> one at a time, so a partial file contains a disproportionate share of the easy half. GPQA reading
> 93.9 % at 0.0 % truncation is that artefact, not a number: its base row is 72.7 % at 25.8 %
> truncated. `tools/eval_ext_complete.py` is the oracle both the retry and forcing stages ask, and
> `tools/stage_top.py` shows the same thing live.

**On resume**, `eval_extend_retry.sh` finishes the three partial files first, then the sweep reaches
scicode and lcb, then forcing, then BFCL multi-turn. Nothing needs to be re-done by hand.

---

## The target: 31–35 tok/s

**22.66 → 28.38 tok/s is banked (+25.3 %).** The remaining ladder is written down rung by rung,
with what each is worth and what it costs: [`DECODE_ENDGAME.md`](DECODE_ENDGAME.md), full
mechanisms in [`ROADMAP.md`](ROADMAP.md).

> **This headline was 35–42 until 2026-08-23.** The C(k) sweep measured its largest rung — adaptive
> block width, estimated at +20–25 % — at an oracle upper bound of **+1.8 %**. The estimate was
> wrong by an order of magnitude, so the target came down with it. See
> [`wiki/negative-results.md` §4m](wiki/negative-results.md).

| # | rung | worth | cost | state |
|---|---|---|---|---|
| — | **banked**: width 5 + fine-tuned head | 22.66 → **28.38** | done | ✅ |
| 1 | corpus — agentic-weighted, 2× size and depth | +4–9 % est. | wall clock | **running** |
| 2 | remaining arms — anchor shape | +0–2 % | wall clock | **spent** — all below incumbent |
| 3 | C(k) sweep, widths 4–12 | *priced rung 4* | wall clock | ✅ **done** |
| 4 | adaptive block width | **+1.8 % measured ceiling** | CUDA | ⚠️ not worth the rewrite |
| 5 | **AR kernel headroom** | +5–10 % est. | CUDA, hard | **now the largest lever** |
| — | **prefill to the roofline** | **6.6× TTFT** | CUDA | practicality, not throughput |

**The levers already used do not repeat.** Twelve draft-head arms at block 5 put their top five
within **1.3 %** of each other against a 3.5 % promotion bar — ce/tv swept three ways, β bracketed on
both sides, anchor shape bracketed at pow 1 and 2, HASS and the confidence loss term both retired.
At **~13.8 tok/s per unit `tau`**, an excellent further arm is worth +1.5 tok/s. That is why rung 1
is *data* rather than another hyperparameter.

**What C(k) actually found.** k\* does vary — {4, 5, 7, 8} across the suite — so the lever is real,
just small. An engine that read each prompt's best width off the table *with hindsight* would gain
**+1.8 %** on the suite mean; a live engine must **predict** k\* per position from the confidence
head (AUC 0.88) and would realise a fraction of that. Four of nine prompts already sit at their
optimum, and the two directions cancel: the agentic categories want **wider** (multi_turn 7,
agentic_format 8) while control, code_gen and explanation want **narrower** (4). The sweep also
re-confirmed, on a fresh measurement, that **5 is the best fixed width** (27.98 tok/s suite mean,
against 27.75 at width 4 and monotone decay above 5).

**Why the ceiling is low, and it is not the head's fault.** `tau`'s ceiling **is** the draft width,
so at a fixed 5 even a perfect head is worth 1.30× — and perfect is impossible, because acceptance
is bounded by the target's *entropy*, not by our ignorance. Varying the width was the one lever that
removes that bound. It has now been measured, and it does not remove much: the width the target's
entropy supports is close to 5 almost everywhere. **The remaining headroom is in the kernels, not in
the speculator.**

**Prefill is not on the ladder and may matter more than all of it.** 90.8 tok/s and **~2.2 min TTFT
at 12 k** contribute nothing to tok/s, and a three-minute time-to-first-token makes throughput
academic for an agentic harness.

## Where the numbers are today

| | measured | ceiling | |
|---|---|---|---|
| **speculative decode**, 8-prompt suite mean | **28.38 tok/s** (`s3recap-p25-b0.1`, live) | — | +25.3 % over the stock head this project started from |
| acceptance τ, suite mean | **3.84 / 5** | 5 at block 5 | **77 %** of the width ceiling |
| **base AR decode** | **14.61 tok/s** | 14.33–15.98 | at the realistic floor |
| **prefill (PS=845)** | **90.8 tok/s** | ≥ 410 target | +45 % on 2026-08-26, all bit-exact; `ATTENTION` is now 50.5 % of it |

The shipped speculator is **`s3recap-p25-b0.1`**, promoted at `tau` 3.8413 against a same-width
incumbent of 3.6888. Every candidate, rejects included, is in
[`HEAD_REGISTRY.md`](HEAD_REGISTRY.md) with its weights archived under `~/model-backups/heads/`; the
programme that produced it is [`wiki/draft-head-finetuning.md` §9](wiki/draft-head-finetuning.md).
**Nothing is ever deleted from the archive** — a refused head is still a measured point on the
acceptance curve, and two of this project's rulers turned out to be wrong after the fact.

**τ is not comparable across block widths.** τ counts tokens committed per target forward and its
ceiling *is* the draft width, so 3.84/5 and 3.84/6 are not the same measurement. Ladder 2.1 moved
the served width from 6 to 5 — which is what `config.json`'s own `dspark_block_size` always said —
and that alone re-prices every τ recorded before 2026-08-21. `s3` reads 3.8438 at width 6 and
**3.6888 at width 5, same weights.**

**One caveat that belongs next to the headline.** Trained heads win the frozen suite and can *lose*
on held-out continuation drafting against a true paired control (F116/F117: training helps where the
head is weak and hurts where it is strong). P2.5's β anchor is the first lever that addressed this
mechanically rather than by choosing a corpus — it pulls the head back toward its pre-training self
in proportion to how well it is *already* accepting — and it is why that arm promoted when the four
loss-reweighting arms before it did not.

**The measurement protocol is part of the number.** τ is quoted as an **8-prompt suite mean at
NGEN0 ≥ 200** — past the drafter's 128-token sliding window. F92 measured τ at **1.39 over the first
32 generated tokens**, rising to ~3.2 only after ~128, so a short-generation acceptance figure is a
transient and is not comparable to anything, including this project's own earlier numbers.

**The single most important correction this project has made to its own model of itself:** the
long-quoted "19.0 tok/s AR roofline" is a *normalisation constant, not a target*. It assumes every
kernel moves bytes at full DRAM bandwidth and that the non-byte part of the step is zero. Neither
holds — 22.3 ms of a 71.4 ms step is not bytes at all, and the byte-moving marks average 191 GB/s,
not 233. See `wiki/measurement-and-traps.md`.

## The wiki

| page | what it holds |
|---|---|
| **[`NORTH_STAR.md`](NORTH_STAR.md)** | why this project exists, what it is for, and the open capability question |
| **[`wiki/kernel-optimisations.md`](wiki/kernel-optimisations.md)** | every adopted AR/spec-decode optimisation: mechanism, measured gain, and the gate that proved it |
| **[`wiki/negative-results.md`](wiki/negative-results.md)** | the levers that were built and **retired**, with the number that killed each. Larger than the win list, and more useful. |
| **[`wiki/prefill-optimisation.md`](wiki/prefill-optimisation.md)** | B9 — why prefill ran decode-shaped kernels, and the four fixes (+30.3 %) |
| **[`wiki/draft-head-finetuning.md`](wiki/draft-head-finetuning.md)** | S5 — the ML: architecture, loss, data, hyperparameters, feasibility arithmetic, and what the literature actually says |
| **[`wiki/measurement-and-traps.md`](wiki/measurement-and-traps.md)** | how a number becomes trustworthy here, and the 30+ ways one has failed to |
| **[`wiki/hardware-sm110a.md`](wiki/hardware-sm110a.md)** | Thor: measured bandwidth **and compute** peaks, and the `sm_110a` ISA facts already settled |
| **[`wiki/cross-model-decode-comparison.md`](wiki/cross-model-decode-comparison.md)** | why this checkpoint decodes at half Qwen's rate on the same box — and why that is a quantisation ranking, not an engine ranking |
| **[`wiki/nvfp4-migration.md`](wiki/nvfp4-migration.md)** | if an NVFP4 REAP existed: what transfers, why the kernel work is a translation not a rewrite, and why requant must come BEFORE the dense GEMV work |
| **[`wiki/dense-mla-gemv.md`](wiki/dense-mla-gemv.md)** | the real lever — dense MLA GEMVs at 115–195 GB/s against a peer's 228–236 — and the bit-exactness invariant it collides with |
| **[`wiki/context-scaling.md`](wiki/context-scaling.md)** | how the forward grows with context, the fit that predicts it, and which items pay only at long context |
| **[`wiki/context-ceiling-is-not-the-kv-cache.md`](wiki/context-ceiling-is-not-the-kv-cache.md)** | what actually bounds usable context here, and why the obvious answer is wrong |
| **[`wiki/moe-gemv-ceiling.md`](wiki/moe-gemv-ceiling.md)** | the MoE GEMV bandwidth ceiling and the repack that needs rows-per-expert the decode shape cannot supply |
| **[`wiki/roofline-why-the-needle-wont-move.md`](wiki/roofline-why-the-needle-wont-move.md)** | why base AR decode is at its realistic floor, and what the roofline number is and is not |
| **[`wiki/oom-and-memory-safety.md`](wiki/oom-and-memory-safety.md)** | 100.4 GiB of weights in a 122 GiB pool: single-tenancy, the memguard, and how runs are launched |
| **[`wiki/README.md`](wiki/README.md)** | **the wiki's own index and the state-in-one-table** — start here if you are new |

## Reference documents

| file | what it holds |
|---|---|
| `ROOFLINE.md` | the arithmetic that governs the project |
| `MODEL_INVENTORY.md` | checkpoint identity + every architectural constant, each traceable to a file |
| `HARDWARE.md` | the box, the memory constraint, `sm_110a` empirical facts |
| `LEVERS.md` | the implementation dedup ledger — what is open, what is closed, and why |
| `LOOP_LOG.md` | 111 findings, chronological. The primary source for everything in the wiki. |
| `RESEARCH_LOG.md` | the search dedup ledger |
| `S5_RECIPE.md` | the draft-head fine-tuning recipe |
| `S5_PROGRESSION.md` | the training session cadence, with stopping rules fixed before the data |
| `HEAD_REGISTRY.md` | every draft-head candidate and whether it was promoted — **rejects included** |
| `RUNS.md` | every fine-tune run and measurement with a link to its evidence log — generated, never hand-edited |
| `ARTIFACTS.md` | **where the draft-head weights live**, and which directory to upload |
| `protocol/suite_prompts.txt` | the frozen 8-prompt eval suite, as token ids |
| `DECODE_FLYWHEEL.md` | the autonomous optimisation loop's operating manual |
| `MODEL_SURVEY_APPENDIX.md` | **which other models fit this box**, what they would decode at, why the frontier open-weights do not fit at any 4-bit quantisation, and why weight streaming cannot rescue them |
| `COMPRESSION_PLAYBOOK.md` | how to get a frontier MoE *resident*: the prune/quant/distill method space, which corners are arithmetically reachable, and the cheap KL-sweep protocol |
| `HARDWARE_ENSEMBLE.md` | **which of the three boxes runs what** — the 3090/Thor/Orin regime boundaries, Orin as an async trace generator, and pipeline-parallel Thor+desktop for 269 GB |
| `EVAL_BUDGET_PROTOCOL.md` | how to choose `max_tokens` — the needed budget is not identifiable from a run at a budget that is too small |
| `PRODUCTION_PLAN.md` | **the four-phase programme and its per-phase gates** — the document that sequences everything above |
| `PHASE2_PLAN.md` | prefill to the roofline: the measured gap, the candidate items, and what each would have to be worth |
| `EVALS.md` / `WHY_THESE_EVALS.md` | the battery, and the argument for why these tasks and not others |
| `CLAUDE.md` | operating rules: **detachment** for unattended work, and why a stage that "completes" against a dead engine is worse than one that dies |

## The model

43 MoE backbone layers + 3 chained DSpark MTP blocks (layers 40/41/42, `mtp.0/1/2`). Hidden 4096,
64 heads × head_dim 512, Q-LoRA/O-LoRA rank 1024, 8 o-groups, 160 routed experts top-6 + 1 shared,
`moe_intermediate` 2048, hyper-connections ×4 with 20 Sinkhorn iterations, sliding window 128,
vocab 129280.

Quantisation **as shipped, never re-quantised**: MLA/dense FP8 e4m3 with F8_E8M0 128×128 block
scales; routed experts OCP MXFP4 (E2M1 + E8M0, block 32); norms/embed/lm_head/compressor/indexer
BF16. `B_tok` = **12.26 GB/token**.

## Hard constraints this repo operates under

- **No additional quantisation.** The checkpoint is used as shipped.
- **No invented model constants.** Every number traces to `config.json`, `REAP_MANIFEST.json` or
  `reap_plan.json`. This has been violated once — a head size copied from a *paper* about a
  different model — and the correction is recorded in `RESEARCH_LOG.md` §6(a).
- **Token ids come from the checkpoint's own tokenizer** (`tools/encode_prompt.py`, which gates
  itself on reproducing the canonical prompt). Inventing ids is the exact mistake the rule exists
  to stop.
- **Correctness gates before speed gates.** One change per measurement. Report bands, not points.
- **DSpark is not DFlash.** `~/gemma-cuda-hybrid` and `~/laguna-s1-cuda-server` are read-only
  references.

## Build and run

```bash
bash scripts/build_decode.sh                      # the engine
bash scripts/build_gate.sh                        # the unit gates

# the ONLY sanctioned launcher: enforces single-tenancy and detaches
scripts/run_model.sh <log> ./build/decode <ckpt> "0,671,6102,294,8760,344" 8
```

The six-id prompt is `BOS + "The capital of France is"` and the expected first decoded token is
`11111`. **Do not abbreviate it** — a truncated list still runs, still prints a tok/s, and silently
reports `GATE FAIL` against a different sequence.

Useful environment flags are catalogued in `LEVERS.md` §5. The two that matter most:
`MOE_MMA=1` (tensor-core MoE — right for prefill, **wrong for decode**) and `DSV4_DPROF=1`
(multi-level named GPU-phase timing, ~0.4 % overhead).

### The server (Phase 6)

```bash
bash scripts/build_server.sh    # gates, server, terminal client
bash scripts/serve.sh           # CPU gates as preflight, then listen on :8080
build/dsv4-chat                 # terminal client
                                # web UI at http://localhost:8080/
```

OpenAI-compatible: `/v1/chat/completions` (streaming and not), `/v1/completions`, `/v1/models`,
`/health`, `/metrics`. Tool calls, thinking blocks, and a KV prefix cache for agentic turns. One
binary, no Python on the request path. See **`SERVER.md`** for the surface, the gates and the
design decisions — in particular why the tokenizer is not gemma's and why the engine is a separate
translation unit from `src/decode.cu`.
