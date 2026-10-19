# Stages s04-s12: select, cut, heal, quantize, gate, publish

These stages consume exactly the s03 outputs (`$V41_WORK/s03/bank`, `routing`, `teacher`, `engram`, the `corpus/` tokens). They are chained in `ops/chain.sh` after s03, with the same quiet / setsid / stop / mark / reclaim pattern:

```
s04 → s05 → s06 → s10 heal → s07 nvfp4 → s07 plan + thor → s07 overlay → s10 final → [s07 thor-b] → s12
```

Only one model-loading process runs at a time. Every GPU step is a subprocess unit (`orchestrate.run_unit`) with a `MemAvailable` floor and a reclaim afterwards, and every stage resumes from its own ledger in `ops/logs/`.

None of this code has been executed against weights yet. The only tests are pure: `ops/tests/test_post.py` (numpy) and `ops/tests/test_gptq_cpu.py` (torch on CPU, tiny tensors).

## Invariants

- **Keep-set fingerprint.** `ops/lib/keepset.py` computes sha256 over the canonical `[[layer, sorted ids]]`. Every downstream artifact records it: s05 manifest and `reap_keepset.json`, s06 gains and bank, s07 alloc, Thor `thor.json`, NVFP4 manifest, s10 decisions, and the s12 cards. Every reader calls `keepset.require`. A mismatch raises `KeepsetMismatch`, a `SystemExit` subclass, so `except Exception` cannot swallow it (GLM v3 defect 1).
- **Proper re-routing.** Every evaluation of a pruned model sets the router bias and `bias_vl` to −inf at pruned experts, so top-6 is re-selected over the survivors and the weights re-normalise. There is never a post-hoc drop of the cut experts. The shared expert, the dense path, the vision tower, the aligner and the MTP layers are never pruned.
- **Split discipline.** Held-out rows are split by `sha256(spec_hash:row)` into **screen** and **confirm** halves.
  - Screen is used for every tuning decision: s04 arbitration and the s10 heal panel.
  - Confirm is used only for the release panel.
  - Calibration (non-held-out) rows feed s06 and the s07 Hessians.
- **Pre-registered gates.** `ops/s10_gates.json` holds the gates. Its sha256 is locked into `s10/prereg_lock.json` at the first s10 result, and any later edit is fatal. The cards carry the sha.
- **No hand-typed numbers on cards.** `ops/lib/cards.py` lints the templates and rejects any literal decimal or percentage.

## s04 `s04_select_arbitrate.py` — candidates and arbitration

1. **select** (CPU, `--procs` pool) builds per-layer prune masks.
   - Guards:
     - drift_guard protect lists (from the s03 `drift/` pre-pass);
     - a Super-Expert guard: the top experts by per-expert max `down_proj` output, banked as `nrm`;
     - per-pool protection at a swept protect-fraction. The pools are the s03 buckets, plus image tokens ranked on their own `bias_vl` routing.
   - Arm A (192) candidates:
     - `hope_sum_knee`, `hope_cond_knee` (HOPE pᵀFp, F from s03 `F_sum` / conditional `F_cnt`) at the knee of the marginal-retention sweep;
     - `hope_sum_pf0`;
     - `reap_sum`, `mass`, `freq`;
     - `random0-2`, which are anchors and ineligible.
   - Pre-registered arm B (128): `b128_hope_sum_knee` and `b128_reap_sum`.
   - Every candidate runs `drift_guard check`. A `BLOCK` verdict makes it ineligible.
2. **arbitrate** (GPU replay, `lib/replay.py`).
   - All 384 experts are resident, so the candidates are evaluated in memory, layer-streamed, with no checkpoint written. The screen subset is `--tokens 196608`.
   - The control (all experts) and every candidate get top-32 KL against the cached teacher.
   - Arms are batched by free disk. `--drop-s03-stream` deletes the s03 `h.npy` / `pre_mix.npy` once nothing needs them.
3. **decide** writes:
   - `s04/keepset.json` (arm A winner);
   - `s04/keepset_128.json` (arm B winner);
   - `s04/arbitration.json` (all scores, drift verdicts, knees);
   - `s04/masks/*.json`.

## s05 `s05_surgery.py` — the MXFP4 master

- Streams shard by shard. Each shard is checked against the pinned HF tree's sha256.
- Kept experts are copied **byte for byte** and renumbered in ascending order. `gate.weight`, `gate.bias` and `gate.bias_vl` are sliced.
- Each layer is verified byte-exact against the source.
- Each file is written, uploaded to a **private** repo, and deleted (`HF_XET_FIXED_UPLOAD_CONCURRENCY=8`, `HF_XET_HIGH_PERFORMANCE` removed). The uploader refuses a repo that is already public.
- Engram shards 47/48 are not rehosted. Instead the repo gets `engram_pointer.json` (base repo, revision, sha256, tensor names), and the regenerated index maps those tensors to the base file names.
- `config.json` and `inference/config.json` get `n_routed_experts = n`. `LICENSE` and `encoding.py` (the chat template) are copied verbatim.
- Output: `s05/manifest_REAP<n>.json`.
- Disk: shards stage on `$V41_WORK`, or the Seagate when that is short (`orchestrate.staging_dir`).

## s06 `s06_heal.py` — measured heal gain (GLM v2 recipe)

- A two-phase replay on the calibration subset (`--tokens 131072`), sequential over layers, on the final keep-set A.
- At each layer, the same FFN input is routed twice: once over all 384 experts (teacher routing) and once re-routed over the survivors.
- The per-layer scalar is GLM's shipped replay estimate: per token `e = Σ_k w_k² m_k²` (m = the s03 bank's mean expert output norm), `g = 1 / mean_tok √(e_post / e_pre)`. It is applied before the stream advances, so later layers see healed inputs.
- The direct estimator (actual routed-sum norm ratio, all 384 experts resident) and the LS estimator `Σ⟨y_post, y_pre⟩ / Σ‖y_post‖²` are logged next to it.
- Outputs:
  - `s06/heal_gain.json`;
  - the post-prune activation bank `s06/bank/L*.npz` (`amax_in`, `amax_mid_ungated`, `x2_in`, `x2_mid`, `n_tok` per kept expert), used by s07.
- The gains are **adopted only if** the paired **s10 heal panel** (screen split, healed vs unhealed) passes. The MXFP4 master always ships unhealed: an E8M0 scale moves only by powers of two.

## s07 `s07_quant.py` — quantization, once, from the MXFP4 source

- **`nvfp4`** (CPU, streamed like s05) produces a lossless repack.
  - The E2M1 bytes are unchanged. Each E8M0 block scale becomes two e4m3 16-block scales, and an FP32 global scale carries the remaining exponent and the adopted gain (w2 only).
  - A tensor with any underflowing block, or with a dequant mismatch on sampled rows, is fatal.
  - `input_scale = amax / (6·448)` comes from the s06 post-prune bank. For w2 it is the *ungated* input amax, assuming the loader applies the routing weight after w2. Unseen experts fall back to the s03 amax.
  - Dense stays FP8. `hf_quant_config.json` uses `MIXED_PRECISION`.
  - Goes to a private repo `…-REAP<n>-NVFP4`. **vLLM loader compatibility is unverified**, and the cards say so.
- **`plan`** (CPU) sets the per-layer qtype for w1|w3 and for w2 from `Q2_K / Q3_K / Q4_K`.
  - Error: imatrix-weighted RTN error on sampled experts, weighted by the layer's kept REAP mass.
  - Protected layers {0, 1, 2, 8, 14, 20, 37, 38, 39} start at Q3_K.
  - Greedy Δerror/Δbyte upgrades run under a budget of **arm B's resident expert bytes** (128 × MXFP4 = 2.833 bpw at 192). The A/B comparison is at matched bytes, as pre-registered.
  - Output: `s07/alloc.json`.
- **`thor`** (GPU) runs a two-phase replay per layer, sequentially.
  - Calibration rows pass through the already-quantized layers. The cached FFN inputs give each kept expert's input Hessian.
  - w1|w3 are GPTQ-rounded jointly (`lib/gptq.py`). w2's Hessian is then built from the **quantized** w1/w3 activations, and w2 (gain folded) is GPTQ-rounded.
  - Experts with fewer than `--min-tokens` routed tokens get a diagonal s03 imatrix Hessian.
  - The layer then runs with its quantized experts and the stream advances.
  - Dense tensors pass through bit-exact, with HC/norms in F32 / BF16 as in the source.
  - Output: `$V41_WORK/thor/A<n>/`, format **`ops/lib/thor_format.md`**, read by s08 (CUDA) and s09 (GGUF).
- **`overlay`** (CPU) is an NVFP4 RTN dense overlay in `thor/overlay_nvfp4/`. It is used only if the s10 overlay gate adopts it.
- **`thor-b`** (CPU) builds arm B: the 128 kept experts copied bit-exact as `MXFP4_SRC`. Run only when s10 picks arm B.

## s10 `s10_eval.py` — paired gates

**heal panel** (screen split) produces `s10/heal/decision.json`.

**final panel** (confirm split) runs these arms:

| arm | what it is |
|---|---|
| `control` | all 384 experts |
| `mxfp4_A` | the master |
| `healed_A` | the NVFP4 function |
| `thorA` | the Thor A build |
| `b128` | arm B |
| `thorA_overlay` | Thor A with the dense overlay |
| `random0` | random anchor |

For each arm it reports:
- overall, per-bucket and per-log-position-bin kl32, top-1 agreement and ΔNLL, with row-cluster bootstrap CIs;
- McNemar on top-1;
- p99 kl32 and the confident-flip rate;
- caption perplexity over the text after the first image in vision rows;
- a 90% Model Confidence Set over the Thor finalists.

The gates and the **pre-registered arm decision** follow `s10_gates.json`. The winner is A only if the paired Δkl32 CI excludes 0 in A's favour, then by McNemar sign, otherwise B.

Outputs: `s10/final/results.json` and `s10/gates.json` (`mxfp4_master`, `nvfp4`, `thor`, `thor_arm`, `overlay_adopted`).

## s12 `s12_publish.py` — cards and visibility

- Renders `ops/cards/{mxfp4,nvfp4,gguf}.md.tmpl` from the s10, s04, s06 and corpus artifacts.
- Each card carries the provenance: teacher revision, corpus spec hash, keep-set sha, gates sha, the MIT licence, and Engram download instructions taken from the pinned tree.
- Uploads the card. A repo is made **public only if its gate passed**; otherwise it stays private with its card.
- The GGUF repo is published from `$V41_WORK/s09/manifest.json` when present. If s09 has not written it, re-run `python3 ops/s12_publish.py --only gguf` later. `--dry-run` renders the cards only.

## Assumptions about s03 formats (all checked only by reading s03's code, not its outputs)

- **`s03/bank/L<L>.npz`**:
  - keys `sum`, `cnt`, `gat`, `nrm` with shape [n_buckets, 384];
  - `F_sum`, `F_cnt` with shape [384, 384];
  - `buckets` (names);
  - `amax_in`, `x2_in`, `x2_mid`, `n_tok` with shape [384, …].
- **`s03/routing`**: idx `uint16` and w `float16`, both [T, 6] over **all** corpus tokens, including held-out rows.
  - It stores the teacher's top-6 only. That is why s04, s06 and s07 re-route live instead of replaying offline.
- **`s03/teacher`**:
  - `top32_ids` int32 and `top32_logprob` float32, both [T, 32], sorted descending;
  - `nll_next` float32 [T], NaN at each row's last token.
- **Engram values** are indexed by global token position.
- **`corpus/rows.jsonl`** rows have `row`, `bucket`, `offset`, `length`, `heldout` and `images`.
- **`corpus/manifest.json`** has `spec_hash`, `max_row_len` and `tokens_total`.
- **Importable from s03**: `Corpus`, `run_engram`, `grab_state`, `build_vision`, `merge_images`, `sources_for`, `published_by`, `last_consumer`.
- **`drift_guard check`** accepts `--out` and `--rows`.
- **The s03 stream** (`h.npy`, `pre_mix.npy`) is overwritten per layer, so replay re-embeds its own rows rather than reading it.
- **The s09 manifest** (`s09/manifest.json`, `{"repo"?, "files": [{"path", "sha256"}]}`) is assumed. It is owned by the GGUF agent.
