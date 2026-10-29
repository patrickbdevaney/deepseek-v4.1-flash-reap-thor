# ZENITH LOOP — the research/implementation architecture for the V4.1 Thor CUDA server

*2026-10-29. Synthesised from the four loop architectures in `~/agent-loop-docs/` (ARCHIPELAGO, Duck,
MANUK Gen-A/Gen-B, O'Forge) and from what this session's first day of decode work actually taught.
Status: design + the first harness pieces (see §9). Numbers marked [M] are measured, [E] estimated.*

---

## 0. Why the four game/product loops do not transfer as-is

All four loops solve the same hard problem: **the oracle is weak**. A game's "is it beautiful", a browser's
"does it match Chrome", an econometrics memo's "is the claim honest" — each needs judges, held-out
oracles, falsifiers, drift scoring, cross-family panels, because the metric can be gamed and the agent
grades itself.

A CUDA inference server is the opposite case. **The oracle is cheap, objective and nearly ungameable**:

| question | oracle | cost |
|---|---|---|
| is it correct? | text md5 against a pinned reference; bitwise unit gates; KL/ppl for lossy levers | seconds–minutes |
| is it faster? | tok/s on a pinned prompt, repeated, page cache dropped | ~3 min per arm |
| where is the time? | nsys per-kernel ledger, ncu per-kernel limiter | ~5 min |
| what is the ceiling? | bytes/token ÷ measured streaming BW (roofline) | arithmetic |

So the machinery that dominates those loops (look judges, drift judges, MAP-Elites over taste) is mostly
unnecessary here. What transfers is the **control discipline**. The thing that does NOT exist in any of
them, and is the reason "we got to 18 so fast", is **an explicit model of the ceiling** that ranks work by
the milliseconds it can still recover. A naive loop optimises whatever it last looked at. A zenith loop
always knows the size of every remaining gap and works the largest one that has a credible lever.

---

## 1. What the first day proved (the evidence the design rests on)

Decode went 8.09 → 18.71 tok/s [M] in one session, all bit-exact. Every win came from one of four
discovery modes, and the loop must run all four on purpose:

| mode | wins it found | how it was found |
|---|---|---|
| **profile → biggest anomaly** | split sparse attention (11.4 ms → ~2 ms), top-k serial scans (1.2 ms → ~0) | nsys kernel table: a kernel with large time and tiny bytes |
| **rate vs. roofline per shape** | bf16 head 117 → 221 GB/s (coalescing permute), NVFP4 RB=2 | per-shape GB/s against the 220 GB/s the FP8 path proves the engine can reach |
| **ncu limiter, not guess** | Q3_K is issue-bound, not memory-bound → 4 blocks/SM (+12%); SoA load layout only +3–10% (load count was NOT the limiter) | ncu SOL + warp-state |
| **path audit** | the "exact" MT=8 change silently moved prefill's split-K → md5 drift; spec decode is 0.70× (verify+draft = 130 ms vs 54 ms AR) | md5 gate against a pinned reference; the spec sweep |

And four failure modes that cost time and must become mechanism (MANUK: "a lesson without a mechanism is
a decoration"):

1. A change believed exact was not (MT=8 split-K). → **every lever declares its exactness tier and the
   gate enforces it**.
2. `CHAIN_STEPS.sh` rebuilt `v41_decode` from an ungated tree and almost benchmarked unverified code. →
   **GPU jobs run only a binary whose tree hash has a gate receipt** (ARCHIPELAGO/MANUK receipt-by-tree).
3. systemd PATH without `/usr/sbin` made `gpu_ok` report a wedged driver. → **preflight canary**
   (O'Forge's `preflight.py` after 1,266 empty iterations).
4. A GPU gate was started while a model process was still exiting. → **one GPU queue, flock'd, with
   a quiet check by executable** (unified-memory OOM = hard reboot; this is the one failure that is
   not cheap to revert).

---

## 2. The ceiling model (the selector's objective function)

This is the core new component. It replaces ARCHIPELAGO's ALP bandit and O'Forge's MAP-Elites with
physics.

**Per decode step**, from the latest nsys trace, for each kernel family k:

```
t_k        measured ms/token
B_k        bytes/token it must move (weights + activations + KV)          — from shapes, exact
floor_k    B_k / BW_peak  (BW_peak = 228 GB/s measured streaming)         — its speed of light
gap_k      t_k − floor_k                                                  — recoverable ms
```

Plus two non-kernel terms: **launch/sync idle** (wall − Σt_k) and **structural bytes** (bytes that a
different design would not move at all: bf16 head vs FP8, the drafter's 807 MB/round of routed bytes).

Today's ledger [M, ov2 trace, overlay build, ~53.4 ms/token at 18.71 tok/s]:

| family | t (ms) | B (MB) | floor (ms) | gap (ms) | known limiter |
|---|---:|---:|---:|---:|---|
| NVFP4 dense (MLA+shared+indexer) | ~20.5 | 3,672 | 16.1 | ~4.4 | wq_b K=1280 lane waste; shared w1/w3 1.2-wave tail |
| K-quant experts (Q2/Q3) | ~20 | 3,003 | 13.2 | ~6.8 | issue-bound (0.45 IPC, 33–67% occ.) |
| bf16 head | 5.6 | 1,324 | 5.8 | ~0 | at roofline — only bytes (FP8/NVFP4 head) move it |
| FP8 Engram wkv | 1.4 | 315 | 1.4 | ~0 | at roofline |
| sparse attn (split) | ~2.2 | ~25 | 0.1 | ~2.1 | 3 launches/layer, latency |
| glue (norm, hc, act_quant, rope, router, units) | ~6 | ~200 | 0.9 | ~5 | ~1,500 launches/step, latency |
| **AR total** | **~53.4** | **8,590** | **37.7** | **~15.7** | AR zenith ≈ 26.5 tok/s at today's bytes |

The AR zenith moves only by moving bytes (FP8/NVFP4 head −0.66 GB → 29 tok/s ceiling; needs a quality
gate). Beyond AR, **spec decode multiplies**: the 0731 server got 2× from correlated routing [M on 0731].
Today spec is 0.70× [M]: the verify+draft round is ~130 ms against a 54 ms AR step. That round is
currently the single largest gap in the whole system (~76 ms per round of overhead), larger than every
AR gap combined.

**Selector rule** (deterministic, no LLM): rank open levers by
`expected_ms_recovered × P(success) / est_cost_hours`, where expected_ms comes from the ledger row the
lever targets, capped by that row's gap. Spec-path levers are scored in ms/accepted-token. Ties go to the
lever that also produces information (a profile, an ncu limiter) — ARCHIPELAGO's "frontier" tick in
numeric form.

---

## 3. The graph (MANUK Gen-B shape, with physics in the nodes)

`docs/zenith/GRAPH.tsv` — one row per lever:

```
id  ring  family  tier  model_floor  deps  status  target_row  expected_ms  gate  notes
```

- **ring**: 0 = instrumentation and harness; 1 = AR kernels; 2 = spec path; 3 = structural (megakernel,
  persistent decode, byte reductions); 4 = production server (batching, memory, API).
- **tier** (pre-declared, enforced by the gate):
  - **T0 bitwise**: text md5 equals the pinned reference in both the fast and `V41_KQ_FAST=0` arms, and
    the lever's unit gate compares old vs new with memcmp.
  - **T1 reassociation**: consistency gate 0 GROSS, greedy text may differ only at measured near-ties;
    a teacher-forced agreement number is recorded.
  - **T2 lossy**: pre-registered quality budget (owner-adopted: d_kl32 ≤ +0.018 for the overlay class),
    measured on the held-out corpus, never on the prompt the speed is measured on.
- **gate**: the unit gate that must exist before the code lands, and must be shown able to go RED under a
  planted perturbation (MANUK `falsify.sh`, Duck selftests). `gate_sa_split`, `gate_topk`,
  `gate_kq_fast` are the templates.
- The agent may append child nodes (MANUK); depth > 2 demotes two rings unless something depends on it.

Picker: `IN_PROGRESS` first (crash resume), else the selector score in §2 over `TODO` nodes whose deps are
`DONE`.

Live graph (2026-10-29): `docs/zenith/GRAPH.tsv` is grouped by `goal`, and `ops/zenith/next.py
[--goal G] [--all]` picks from it. Each goal has its own decode target:

- `ar_ceiling` — about 41-42 ms/step, roughly 24 tok/s. Exact kernels only (A*).
- `head_fp4` — bytes. B1 adds a separate NVFP4 head copy built from the bf16 master, which stays
  untouched (T2). B2 adds an NVFP4 shortlist plus an exact bf16 rescore, which keeps greedy output
  exact.
- `spec` — round cost (S*).
- `draft_ft` — drafter fine-tune (D1-D4). Verify keeps the output exact, so its gate is acceptance,
  speed and spec identity.

Together these give an estimated (not measured) 32-35 tok/s on predictable text with exact weights,
and 35-38 tok/s with the NVFP4 head.

---

## 4. One tick (the commit protocol)

ARCHIPELAGO/MANUK's tick, adapted to a GPU that cannot be shared:

```
1 ORIENT      read ledger (latest nsys) + RATCHET + GRAPH; pick node (selector)
2 HYPOTHESIS  journal entry BEFORE code: target row, mechanism, expected ms, tier, how it could fail
3 MICRO       write the kernel + its unit gate (old vs new, bitwise or tolerance) + a bench in the same file
4 GPU QUEUE   gate + bench run through gpuq (flock, quiet-by-executable, MemAvailable ≥ need + 10 GiB)
5 E2E A/B     ab_decode: N≥3 repeats per arm, drop_caches, pinned prompt, control arm untouched;
              accept only if Δ > 2σ of the arm noise AND md5 matches the tier's reference
6 RECEIPT     gate receipt = git write-tree hash of the verified tree; gpuq refuses to run a model
              binary built from a tree without a receipt (closes the CHAIN_STEPS rebuild trap)
7 RATCHET     tok/s (AR, spec best-K), per-family GB/s, resident GiB, peak MemAvailable floor;
              max-only banking; a regression is a revert, never a re-baseline
8 COMMIT      only through tick script; message carries tier, Δms, gate names
9 RE-PROFILE  every 3rd landed tick or any tick > 1 ms: fresh nsys → ledger regenerated
```

A red at 4 or 5 is a revert of the tick's files (Duck: "there is no third outcome"). Exploratory variants
that lose (SoA's +3%, RB=4 on bf16) are recorded in the kernel file's rejected-variants comment and in
`docs/zenith/REJECTED.md` with their numbers, so no later tick re-runs them.

---

## 5. Parallelism: one GPU, many minds

Thor's GPU is a single serialised resource (and co-running = hard reboot). The CPU, the compiler and the
reasoning are not. The architecture separates them (O'Forge's builder/researcher duty cycle):

| lane | runs on | does | never does |
|---|---|---|---|
| **GPU lane** (`gpuq`) | Thor GPU, one job at a time | gates, benches, A/B, nsys/ncu, spec sweeps | anything concurrent |
| **build lanes** (1–2) | Thor CPU (and the desktop for nvcc-only syntax checks; sm_110a compiles anywhere with CUDA 13) | write kernels + unit gates in git worktrees, compile | touch the GPU; land without a receipt |
| **research lane** | Claude subagents / web | recon ticks: "how does X reach roofline" — Marlin, FlashInfer decode, TRT-LLM MoE GEMV, llama.cpp MMVQ, Hazy/Mirage megakernels, EAGLE-3/MTP verify trees | write code |
| **observer** | this interactive session | owns the harness, the ledger generator, the gates' references, the ratchet; arbitrates | — |

The queue is the throughput limit, so the loop's job is to **keep the GPU lane full of high-value jobs**:
while a 15-minute A/B runs, the build lane prepares the next two candidates and their gates, and the
research lane answers the next open "why is this kernel at 60%" question. That is the difference from
today's serial loop, where the GPU idled while I wrote code and I idled while it ran.

Model routing (ARCHIPELAGO/MANUK Gen-B ladder): Sonnet for mechanical ticks (wiring, gates, harness
scripts), Opus for kernel design and anything that failed twice. Escalations are recorded so the model
floor per family becomes measured.

---

## 6. Research as a routed tick, not a prompt sentence

ARCHIPELAGO's hardest-won rule: "priority beats prompting" — "do a look pass" sat in a prompt for 13
launches and was ignored. So research is a **separate node kind** with a mechanical "did the act
happen" check: a `docs/zenith/RECON-<topic>.md` with ≥ 2 primary sources and a concrete, testable lever
that becomes a GRAPH row. It is routed when:

- a ledger row's gap is > 2 ms and has no open lever (nothing credible queued against the biggest hole), or
- two consecutive ticks on the same row landed < 0.3 ms each (plateau → the technique, not the tuning,
  is wrong — MANUK's `lever-pivot` staleness).

The first recon topics, from today's ledger: (1) issue-bound K-quant GEMV — how llama.cpp MMVQ and
Marlin-style kernels hit BW at 2–3 bit; (2) persistent/megakernel decode for ~1,500 launches/step;
(3) MoE spec-verify cost — why one extra row costs 76 ms here and how EAGLE/MTP servers keep the verify
near one AR step (correlated routing, expert-union batching, draft reuse of target expert bytes).

---

## 7. Gates that cannot be gamed (and the ones that can)

The oracle is strong, but three things can still lie, and each gets a mechanism:

1. **The benchmark prompt.** Speed on one short chat prompt over-weights short context (k == n top-k
   fast path, ring-only attention). The ratchet tracks three pinned workloads: 128-token chat, 8K
   context, 32K context; a lever must not regress any of them.
2. **Noise.** Arm-to-arm spread is up to ±4% (16.82–18.23 within one arm today). Accept on mean Δ > 2σ,
   carry an untouched control arm in every A/B (the KDA-launch-bound lesson), and record σ in the ratchet.
3. **Lossy creep.** T2 levers are judged on the held-out quality corpus with a pre-registered budget
   (O'Forge SUCCESS.md: thresholds frozen before the first measurement), never on the speed prompt.

Gate self-tests: each unit gate has a `--plant` mode that perturbs one element/one ULP and must report
FAIL (Duck/MANUK falsifiers). A gate that cannot go red is not a gate.

---

## 8. Operational guards (each one is an incident from this session or a sibling loop)

- **gpuq**: flock + quiet-by-executable (not `pgrep -f`, which self-matches — MANUK's recurring hang
  class) + MemAvailable floor + drop_caches + clocks check (`jetson_clocks` on, gpc 1575 MHz).
- **preflight canary**: PATH has `/usr/sbin` (nvidia-smi), `nvcc` resolves, kernel launch smoke, before any
  long job.
- **receipt gate**: model binaries carry the tree hash they were built from; gpuq refuses an unreceipted
  binary.
- **detached**: every GPU job is `systemd-run --user`, never a child of an interactive shell (SSH-drop
  lesson, 16 h lost).
- **liveness = landing**: the loop is healthy iff the ratchet moved in the last N hours, not iff processes
  are alive (ARCHIPELAGO's HEAD-moved rule).
- **quota**: research and build lanes back off on usage-limit text; the GPU lane keeps running queued
  jobs regardless (it needs no model).

---

## 9. What exists now and what gets built first

Exists [M]: `ops/bench/ab_decode.sh` (sequential arms, md5 + speed), `ops/prof/nsys_decode.sh`,
unit gates (`gate_sa_split`, `gate_topk`, `gate_kq_fast`, `bench_dense`, `gate_v41_units`), memory
notes with the md5 references.

Build order (ring 0, each a tick of its own):

1. `ops/zenith/ledger.py` — nsys sqlite → the §2 table (t, B, floor, gap per family) + spec-round
   decomposition, written to `docs/zenith/LEDGER.md`. Shapes give B exactly.
2. `ops/zenith/gpuq.sh` — the single GPU lane with the §8 guards and receipts; `ab_decode` and
   `nsys_decode` run only through it.
3. `docs/zenith/GRAPH.tsv` + `ops/zenith/next.py` — seeded from §2's gaps (below).
4. `docs/zenith/RATCHET.tsv` + `ops/zenith/ratchet.py` (max-only banking, three workloads).
5. Then the loop runs: this session as observer + build lane, subagents as research lane; a headless
   `claude -p` supervisor only if the owner wants it unattended overnight (it spends the shared weekly
   quota that ARCHIPELAGO also draws on).

**Seed graph** (ranked by §2):

| id | ring | lever | target | expected | tier |
|---|---|---|---|---|---|
| S1 | 2 | profile one spec round (draft vs verify vs sync), nsys | spec round 130 ms | information | — |
| S2 | 2 | verify-path experts: union the K+1 rows' experts, read each expert once (0731's k-invariant fix) | spec round | −20–40 ms/round [E] | T0 |
| S3 | 2 | drafter cost: its own profile; MXFP4 draft experts → reuse target-resident experts where routing overlaps | spec round | −10–30 ms [E] | T1 |
| A1 | 1 | K-quant: recon (MMVQ/Marlin) then instruction-diet kernel (pre-split Q3 hi plane, 12-bit scale pairs) | experts 6.8 ms gap | −2–4 ms | T0 |
| A2 | 1 | NVFP4: fuse w1‖w3 and wq_a‖wkv‖indexer launches; 2-chunk lanes for K=1280 | dense 4.4 ms gap | −1.5–3 ms | T0 |
| A3 | 3 | glue fusion: norm+act_quant, hc partial+finish, rope into q epilogue | glue 5 ms gap | −2–3 ms | T0 |
| A4 | 3 | persistent decode megakernel (recon first) | launches + glue | −3–6 ms [E] | T0 |
| A5 | 1 | sparse attn: fuse gather+scores, one launch | 2.1 ms gap | −1 ms | T0 |
| B1 | 3 | FP8/NVFP4 head | head bytes −0.66 GB | −2.9 ms | T2 |
| P1 | 4 | production: v41_server on the fast path, KV/state memory audit, 32K/128K workloads in the ratchet | — | — | T0 |

The ordering is the point: S1–S3 sit on a ~76 ms/round hole; A1–A5 together sit on ~16 ms. A naive
loop would keep polishing GEMVs.
