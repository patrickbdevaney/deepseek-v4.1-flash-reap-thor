# Internal learnings playbook: what our GLM-5.3-Flash and MiMo-V2.6-Flash REAP-50 runs proved, for a DeepSeek-V4.1-Flash-1017 REAP on one Thor

Scope: only our own measured work on this machine. Sources are local files, cited as paths. "(inference)" marks my reasoning; anything without a path is not a finding. Written 2026-10-19 from read-only inspection: no GPU jobs were run and nothing outside this notes file was changed.

**Three things a reader must know first, because they contradict the published README:**
1. **v3 (pass3-hope) was healed with the first-moment derivation that we had already measured as ~30% over-correcting. It did not use the measured router-replay gain.** The keep-set guard fired: `heal_refit.json` had been measured on the pass-2 keep-set `9d2ed9e33db85f90`, while v3 applies `cfab1fb7c846ad41`. The guard raised `RuntimeError` inside a `try/except` that caught it and fell back with only a WARN: "NO measured gains available - applying the first-moment derivation, which is known to over-correct by ~30%" (median 0.7664, range 0.6673–0.8533, 6048 tensors scaled with the layer scalar and 0 per-expert) — [/home/patrickd/glm-5.3-reap/logs/s05_heal.log](/home/patrickd/glm-5.3-reap/logs/s05_heal.log), [/home/patrickd/glm-5.3-reap/artifacts/s05_heal.json](/home/patrickd/glm-5.3-reap/artifacts/s05_heal.json). On pass 2 the measured replay gain was 0.9111 against a first-moment 0.7149 — [/home/patrickd/glm-5.3-reap/wiki/70-healing.md](/home/patrickd/glm-5.3-reap/wiki/70-healing.md). The README describes v3 as healing "to a median gain of 0.766", and its version table says v3 healing is "per-expert". The log contradicts both — [/home/patrickd/glm-5.3-reap/README.md](/home/patrickd/glm-5.3-reap/README.md).
2. **v3 has never been evaluated end-to-end.** The pipeline log shows `s09_eval:pending` through 2026-10-01 23:59 — [/home/patrickd/glm-5.3-reap/logs/pipeline.log](/home/patrickd/glm-5.3-reap/logs/pipeline.log). The only `s09_eval.json` is dated 2026-08-29 and names student `glm-5.3-flash-reap50-nvfp4-pass2-attnkda`, which is a pass-2 arm — [/home/patrickd/glm-5.3-reap/artifacts/s09_eval.json](/home/patrickd/glm-5.3-reap/artifacts/s09_eval.json). So nothing measures whether v3 (HOPE + protect_frac) beats v2.
3. **The v3 NVFP4 was built with `nvfp4_attn_mode="kda"`, which also quantises the KDA q/k/v attention projections.** — [/home/patrickd/glm-5.3-reap/state/state.db](/home/patrickd/glm-5.3-reap/state/state.db) kv table; `s07_quantize.json` 2026-10-01 01:19, 93.6 GiB — [/home/patrickd/glm-5.3-reap/artifacts/s07_quantize.json](/home/patrickd/glm-5.3-reap/artifacts/s07_quantize.json). On pass 2 we measured that arm at −0.00230 top-1 (5.8σ) and declined it for the shipped artifact — [/home/patrickd/glm-5.3-reap/wiki/00-log.md](/home/patrickd/glm-5.3-reap/wiki/00-log.md). (inference) The published v3 NVFP4 therefore carries an attention-quant cost that v2 deliberately avoided. Memory confirms that the local `…-nvfp4-pass2-attnkda` directory is v3, with sha256 matching HF `pass3-hope/` — [/home/patrickd/.claude/projects/-home-patrickd/memory/glm53-flash-reap.md](/home/patrickd/.claude/projects/-home-patrickd/memory/glm53-flash-reap.md).

---

## Q1. What exactly is our current best 50% recipe end to end (s01..s09), with parameters and measured outcomes per pass?

### Takeaway
The best end-to-end measured GLM 50% recipe is **v2 = pass-2 mask + per-layer scalar healing from router replay (FP8 top-1 0.84249)**. v3 adds HOPE pᵀFp selection, `protect_frac=0.04`, and a rebalanced token-space corpus. These are the methodologically best choices, but v3 is unmeasured end-to-end and shipped with two regressions: the first-moment heal and NVFP4 on the KDA attention projections. The recipe to carry forward is therefore v3's *selection* combined with v2's *healing discipline*, plus one end-to-end eval arm before anything ships.

### Cited Findings
**Pipeline and stage map (GLM)**
- Stages: s00 smoke → s01 stage 328.3 GB FP8 (byte-verified) → s01b placement by arithmetic (verdict "stream") → s02 corpus → s03 saliency (layer streaming, raw accumulators) → s04 sweep (re-rank every ratio from cached scores; go/no-go) → s04b surgery (tensor-level prune on safetensors; deletes each source shard after its survivors are written) → s05 heal → s06 emit FP8 → s07 NVFP4A16 via compressed-tensors → s08 document → s09 paired eval. — [/home/patrickd/glm-5.3-reap/README.md](/home/patrickd/glm-5.3-reap/README.md)
- The model is never loaded whole. The pipeline streams one decoder layer at a time from mmap'd shards, and pruning in FP8 is lossless on every retained weight because each expert has its own `weight_scale_inv`. — [/home/patrickd/glm-5.3-reap/README.md](/home/patrickd/glm-5.3-reap/README.md)
- State is kept in SQLite (`state/state.db`). Every stage is idempotent and resumable, and the pipeline runs under `systemd --user` with lingering. — [/home/patrickd/glm-5.3-reap/README.md](/home/patrickd/glm-5.3-reap/README.md)

**s03 parameters (pass 3)**
- `MAX_LEN=2048`, `CALIB_TOKENS` default 5.5M, `CHUNK_TOKENS` 500k, `calib_batch` 2, `LAYER_COST_GIB=53.0` (measured host footprint, not weight size), `RECLAIM_FLOOR_GIB` 30, `HARD_FLOOR_GIB` 6. — [/home/patrickd/glm-5.3-reap/scripts/stages/s03_saliency.py](/home/patrickd/glm-5.3-reap/scripts/stages/s03_saliency.py)
- Pass-3 audit: 42 layers, 10 chunks, 45 blocks; min 1,909 and mean 71,895 tokens per expert; 2 of 12,096 expert slots below the 2,000-token floor; path `stream-blocked`. — [/home/patrickd/glm-5.3-reap/artifacts/s03_saliency.json](/home/patrickd/glm-5.3-reap/artifacts/s03_saliency.json)
- Pass 3 was calibrated on 866,678,064 routed slots (42 layers × 10 chunks). — [/home/patrickd/glm-5.3-reap/wiki/31-pass3-hope-method.md](/home/patrickd/glm-5.3-reap/wiki/31-pass3-hope-method.md). (inference) 866,678,064 / 42 / 8 ≈ 2.58M routed tokens per layer, about half the 5.5M nominal. The cause is unexplained in the files (padding or the valid-mask are candidates).

**s04 sweep at 0.50 (pass 3)**
- Routing mass retained 0.4738 (0.948 of expected), saliency mass retained 0.6686, min-layer routing mass 0.3748, rest FP8 90.6 GiB, gate "amber"; 0.30 is over the envelope. Arm B (quantile 0.6·mean+0.4·p99) was not available because the tracker keeps only sum/count. — [/home/patrickd/glm-5.3-reap/artifacts/s04_sweep.json](/home/patrickd/glm-5.3-reap/artifacts/s04_sweep.json)

**s04b selection (pass 3)**
- `select_mode` defaults to `hope`. HOPE calls `select_run(..., "hope", "reap_1_1_1", protect_frac=pf)`, reading pf late from `conf/protect_frac_override.txt`. A separate `GLM5_REAP_CRITERION` (default `mass`) applies only to the scalar/curve path. The MTP block's experts are ranked by log-Frobenius weight norm. — [/home/patrickd/glm-5.3-reap/scripts/stages/s04b_surgery.py](/home/patrickd/glm-5.3-reap/scripts/stages/s04b_surgery.py)
- Result: 144 of 288 kept, 43 MoE layers, 86 routers sliced, 38,956 tensors kept and 37,152 dropped; FP8 160.6 GiB; NVFP4 93.6 GiB (62 shards), which fits the 117 GiB envelope with 23 GiB spare. — [/home/patrickd/glm-5.3-reap/wiki/31-pass3-hope-method.md](/home/patrickd/glm-5.3-reap/wiki/31-pass3-hope-method.md), [/home/patrickd/glm-5.3-reap/artifacts/s04b_surgery.json](/home/patrickd/glm-5.3-reap/artifacts/s04b_surgery.json)
- protect_frac sweep at ratio 0.5. pf=0: worst_ret 0.5399, icost 0.1183. pf=0.04: 0.5580 / 0.1270. Marginal gain/cost is 1.41 into 0.04 and 0.54 after it; 0.12 is infeasible. The worst domain is `general` (17.58% of corpus) at every pf. — [/home/patrickd/glm-5.3-reap/conf/protect_frac_override.txt](/home/patrickd/glm-5.3-reap/conf/protect_frac_override.txt)
- Per-domain retention at the shipped setting: finance 0.686, math 0.687, agentic 0.706, code 0.722, vision 0.724. Bio: 0 of 866.7M slots, because every source is excluded by licence or clinical scope. — [/home/patrickd/glm-5.3-reap/wiki/31-pass3-hope-method.md](/home/patrickd/glm-5.3-reap/wiki/31-pass3-hope-method.md)

**Per-pass outcomes (FP8, 241,516 held-out tokens, cached teacher, paired)**

| pass | mask | heal | top-1 | ΔNLL | top-k KL | source |
|---|---|---|---|---|---|---|
| v1 published (re-healed) | REAP scalar, 0.52M tok | scalar | 0.83703 | 0.1979 | 0.6948 | [HEALING_ABLATION](/home/patrickd/glm-5.3-reap/research/HEALING_ABLATION.md) |
| v2 as first built | pass-2 | per-expert LS | 0.83693 | 0.19396 | 0.69388 | same |
| **v2 shipped (reverted)** | pass-2 | **per-layer scalar (router replay)** | **0.84249** | **0.17563** | **0.65030** | same |
| v2 NVFP4 (attn untouched) | pass-2 | scalar | 0.83857 | — | — | [00-log](/home/patrickd/glm-5.3-reap/wiki/00-log.md) |
| v2 NVFP4 + KDA q/k/v NVFP4 | pass-2 | scalar | 0.83627 (s09 file: 0.83627) | 0.1909 | 0.6864 | [s09_eval.json](/home/patrickd/glm-5.3-reap/artifacts/s09_eval.json) |
| v3 | HOPE + pf 0.04 | **first-moment scalar 0.766** | **not measured** | — | — | [pipeline.log](/home/patrickd/glm-5.3-reap/logs/pipeline.log) |

- The v2 mask is worth +0.00545 top-1 (McNemar z=9.3, paired s.e. 0.00058). Per-expert healing cost −0.00556 (z=11.8, paired s.e. 0.00047), so the two cancelled into an apparent null. — [/home/patrickd/glm-5.3-reap/research/HEALING_ABLATION.md](/home/patrickd/glm-5.3-reap/research/HEALING_ABLATION.md)
- Per-domain top-1 for the v2 NVFP4 attnkda arm: code 0.916, math 0.916, agentic 0.867, science 0.819, finance 0.747, ballast 0.579. Vision had only 532 tokens and is marked insufficient. — [/home/patrickd/glm-5.3-reap/artifacts/s09_eval.json](/home/patrickd/glm-5.3-reap/artifacts/s09_eval.json)
- Pass 1 had real defects: calibrated on 0.52M of 48.3M corpus tokens, 39 image samples, padding counted as tokens, no evaluation, and the MTP layer 45 dropped. — [/home/patrickd/.claude/projects/-home-patrickd/memory/glm53-flash-reap.md](/home/patrickd/.claude/projects/-home-patrickd/memory/glm53-flash-reap.md)

**Calibration corpus design (pass 3)**
- Targets are set in **token** space: agentic 0.23, code 0.19, math 0.15, ballast 0.15, multimodal 0.12, science 0.10, finance 0.06. Difficulty is medium 0.60 / hard 0.30 / easy 0.10. Held-out fraction is 0.08, stratified per bucket. — [/home/patrickd/glm-5.3-reap/scripts/corpus_spec.py](/home/patrickd/glm-5.3-reap/scripts/corpus_spec.py)
- Why token space: pass 2 specified the mixture in samples, and the realised token shares were agentic 31.6%, math 29.9%, science 21.3%, multimodal 6.2%, ballast 4.9%, code 5.7% (intended 21%), finance 0.5% (intended 8%). Per-bucket accumulators allow offline re-weighting via `S_j = Σ_b w_b·sum_b,j/cnt_b,j`. — [/home/patrickd/glm-5.3-reap/wiki/80-calibration.md](/home/patrickd/glm-5.3-reap/wiki/80-calibration.md)
- The pass-3 corpus meets its difficulty spec (31.0% hard, mean 4,761 tok, max 16,384). `MAX_LEN=2048` discards 73.5% of collected tokens (35.5M of 48.3M), and the corpus is 0.3% CJK-bearing. — [/home/patrickd/glm-5.3-reap/wiki/00-log.md](/home/patrickd/glm-5.3-reap/wiki/00-log.md)

**Healing actually used**
- s05 multiplies F32 `down_proj.weight_scale_inv` (FP8 block scales), which makes it invertible. The measured gain comes from replaying post-prune routing out of the router cache (`heal_refit.py`). The first-moment derivation is a WARN fallback only. — [/home/patrickd/glm-5.3-reap/scripts/stages/s05_heal.py](/home/patrickd/glm-5.3-reap/scripts/stages/s05_heal.py)

**Layer allocation**
- Uniform. `num_local_experts` is one scalar in glm5_next, vLLM and the GGUF converters, and the CUDA server hard-codes `N_ROUTED_EXPERT=144`. The exact min-plus DP would have gained −4.98% gross error at 50% but is unloadable. — [/home/patrickd/.claude/projects/-home-patrickd/memory/reap-criterion-and-corpus-audit.md](/home/patrickd/.claude/projects/-home-patrickd/memory/reap-criterion-and-corpus-audit.md)
- MiMo's EvoESAP search gained +0.0115 worst-domain (0.7331→0.7446) but was "NOT APPLIED — uniform budget keeps the checkpoint loadable". — [/home/patrickd/xiaomi-2.6-flash-REAP/logs/reap_run.log](/home/patrickd/xiaomi-2.6-flash-REAP/logs/reap_run.log)

**Router handling**
- Routers are sliced by row. `norm_topk_prob` conserves gate mass under pruning (measured 2.5000 pre and post). Post-prune routing is exactly replayable offline because `n_group==1`. — [/home/patrickd/glm-5.3-reap/STATE.md](/home/patrickd/glm-5.3-reap/STATE.md)
- Router KD (layer-local) beat the teacher's sliced router on 0 of 47 MiMo layers. — [/home/patrickd/xiaomi-2.6-flash-REAP/logs/reap_run.log](/home/patrickd/xiaomi-2.6-flash-REAP/logs/reap_run.log)

**Quantization**
- v2 NVFP4 with attention untouched is 98.2 GiB and top-1 0.83857, i.e. −0.0039 vs FP8 0.84249 (inference: by subtraction across runs). — [/home/patrickd/glm-5.3-reap/wiki/00-log.md](/home/patrickd/glm-5.3-reap/wiki/00-log.md)
- Attention NVFP4 arms vs v2 NVFP4. `kda` (KDA q/k/v): −0.00230, 93.6 GiB. `post` (o_proj + MLA): −0.00551. `all`: −0.00673, 90.2 GiB. Damage is sub-additive (0.00781 predicted by summing, 0.00673 measured). — [/home/patrickd/glm-5.3-reap/wiki/00-log.md](/home/patrickd/glm-5.3-reap/wiki/00-log.md)
- NVFP4 dense overlay (CUDA server, AR-path bf16 → 3.70 GiB overlay): B_tok 19.761→9.762 G, decode 8.36→11.86 tok/s, PPL +1.56%, 90.10% top-1 vs bf16, 0.138% confident flips. Kept. — [/home/patrickd/.claude/projects/-home-patrickd/memory/glm53-nvfp4-dense-overlay.md](/home/patrickd/.claude/projects/-home-patrickd/memory/glm53-nvfp4-dense-overlay.md)
- GGUF: built FP8 → Q8_0 → 4-bit; the Q8_0 intermediate adds +0.78% (a BF16 intermediate would need 308 GiB); no imatrix was used for the shipped files. 8×512 PPL: IQ3_M 4.6253, IQ4_XS 4.6002, Q3_K_M 4.6502, Q4_K_M 4.5959, Q4_K_S 4.5833, explicitly "within noise", a corruption tripwire only. NIAH on Q4_K_M found 9/9 needles at 2k/8k/32k. — [/home/patrickd/glm-5.3-reap/gguf_README.md](/home/patrickd/glm-5.3-reap/gguf_README.md), [/home/patrickd/glm-5.3-reap/artifacts/niah_GLM-5.3-Flash-REAP50-Q4_K_M.json](/home/patrickd/glm-5.3-reap/artifacts/niah_GLM-5.3-Flash-REAP50-Q4_K_M.json)
- On Thor, llama.cpp must run with `-ngl 0 --no-repack`; `-ngl 99` on 93 GiB OOMs the machine. — [/home/patrickd/glm-5.3-reap/gguf_README.md](/home/patrickd/glm-5.3-reap/gguf_README.md)

**Long-context assurance**
- The teacher's routing was captured *before* surgery (surgery deletes the source): 42 layers × 16 rows × 8192 tokens, top-8 int16, 5,505,024 tokens. 16k died from 93.0 GiB free because attention is quadratic; 8k used ~57 GiB. — [/home/patrickd/glm-5.3-reap/wiki/31-pass3-hope-method.md](/home/patrickd/glm-5.3-reap/wiki/31-pass3-hope-method.md)

### Inferences
- (inference) Best measured end-to-end recipe: v2 selection, scalar router-replay healing, FP8 or NVFP4 with attention left in higher precision. Best *method*: v3 selection with v2 healing. The v3 artifact is likely MoE under-scaled by about 0.766/0.91 ≈ 0.84×, if the v3 measured gain lands near pass 2's 0.91. That is unverified and has to be measured on the v3 keep-set with `heal_refit.py`. The fix costs a multiply on F32 scales (on V4.1 these would be E8M0 scales, which changes this; see Q5).
- (inference) For V4.1 the order should be: s03 accumulates sum/sq/nrm/nsq/gat/cnt **plus F off-diagonal plus the router cache plus per-expert max** in one pass → HOPE at a measured pf → re-run `heal_refit` on the *final* keep-set before s05 (make the keep-set mismatch fatal, not a WARN) → one paired s09 arm → quantize.

### Gaps
- No v3 end-to-end numbers exist, so whether HOPE and protect_frac actually helped is unknown.
- I found no file explaining why pass-3 routed tokens (~2.58M/layer, inferred) are half the 5.5M nominal.

---

## Q2. Which proxies predicted end-to-end quality, and which did not?

### Takeaway
Per-domain saliency retention predicted per-domain damage well (r=0.942). The reconstruction residual had the right sign for **masks** but the wrong sign for **weight-space corrections**. Per-tensor cosine badly understated end-to-end change. Saliency-mass "wins" are circular. Teacher-forced metrics cannot see looping. Every weight-space change needs one end-to-end paired arm.

### Cited Findings
- **Predicted:** per-domain retention from calibration accumulators vs measured held-out top-1 gives Pearson r=0.942, Spearman ρ=0.714. Ballast was both least retained (0.487) and least agreeing (0.572). — [/home/patrickd/glm-5.3-reap/wiki/97-evaluation.md](/home/patrickd/glm-5.3-reap/wiki/97-evaluation.md)
- **Right sign for masks, understated size:** the v2 mask reduced residual by 2.5% (0.2730→0.2663, 30/42 layers) and saliency mass by +1.524%. End-to-end it was +0.00545 at 9.3σ. — [/home/patrickd/glm-5.3-reap/research/HEALING_ABLATION.md](/home/patrickd/glm-5.3-reap/research/HEALING_ABLATION.md), [/home/patrickd/glm-5.3-reap/wiki/30-reap.md](/home/patrickd/glm-5.3-reap/wiki/30-reap.md)
- **Wrong sign for corrections:** per-expert healing lowered held-out residual in 41/42 layers (0.2914→0.2663), yet top-1 got worse by 0.00545 (11.8σ). The coefficient reduces to `c_j = gate-before/gate-after`, which suppresses promoted substitutes: 62.0% of 5,760 coefficients sat below the layer scalar and the worst was 3.3×. — [/home/patrickd/glm-5.3-reap/research/HEALING_ABLATION.md](/home/patrickd/glm-5.3-reap/research/HEALING_ABLATION.md)
- **Why a richer correction can't help:** ‖μ‖²/E‖f‖² = 0.034, so 96.6% of an expert's output energy is token-dependent. Expert outputs are near-orthogonal (off-diagonal mass 1.9%, mean |cos| 0.091). The 0.27 residual is deleted information. — [/home/patrickd/glm-5.3-reap/wiki/70-healing.md](/home/patrickd/glm-5.3-reap/wiki/70-healing.md)
- **Derivation vs measurement:** the first-moment heal gain was 0.6964 shipped, 0.7149 recomputed, 0.9111 measured by router replay, a 30.8% disagreement (real output inflation 1.1015× vs 1.3925× implied). It converged at one chunk (3 chunks: 0.9099; per-layer drift median 0.10%). — [/home/patrickd/glm-5.3-reap/wiki/70-healing.md](/home/patrickd/glm-5.3-reap/wiki/70-healing.md)
- **Cosine misleads:** dense NVFP4 overlay cos 0.9972, but +1.56% PPL and a 9.9% top-1 flip rate. Memory says to quote PPL and confident flips, never the cosine. — [/home/patrickd/.claude/projects/-home-patrickd/memory/glm53-nvfp4-dense-overlay.md](/home/patrickd/.claude/projects/-home-patrickd/memory/glm53-nvfp4-dense-overlay.md)
- **Keep-set identity is the wrong gate:** split-half overlap was 0.9446 at 2.75M tok/half, with 19/42 layers below 0.95. Retained-mass disagreement was only 0.133% mean (0.562% max), and the disputed experts sat a median 2.42% from the cut. `split_half.py` now passes on mass. — [/home/patrickd/glm-5.3-reap/wiki/30-reap.md](/home/patrickd/glm-5.3-reap/wiki/30-reap.md), [/home/patrickd/glm-5.3-reap/artifacts/split_half.json](/home/patrickd/glm-5.3-reap/artifacts/split_half.json)
- **Criterion shootout (keep-set overlap vs REAP at full budget):** var_aware 0.9635, quantile 0.9385, mix_sample 0.9378, gate_only 0.9238, mix_codemath 0.9231, norm_only 0.8940, frequency 0.7553. Stock REAP was "not measurably improvable" at this ratio. — [/home/patrickd/glm-5.3-reap/wiki/30-reap.md](/home/patrickd/glm-5.3-reap/wiki/30-reap.md)
- **Circular rulers:** "both numbers we have used to judge a mask are rulers that one of the candidate selectors maximises by construction"; MiMo shipped on those rulers alone. — [/home/patrickd/glm-5.3-reap/research/REAP_METHOD_AND_FINETUNE_VIABILITY_2026-09-26.md](/home/patrickd/glm-5.3-reap/research/REAP_METHOD_AND_FINETUNE_VIABILITY_2026-09-26.md). MiMo's log says the same: "The two rulers disagree BY CONSTRUCTION". HOPE vs REAP worst-domain was +0.2571 on reap_1_1_1 but −0.0132 on total_0_1_1. — [/home/patrickd/xiaomi-2.6-flash-REAP/logs/reap_run.log](/home/patrickd/xiaomi-2.6-flash-REAP/logs/reap_run.log)
- **"mass" criterion claim (offline only):** dropping REAP's 1/N_j cut reconstruction error 0.26974→0.19594 and improved all domains. Memory flags that this is partly circular and corpus-dependent. — [/home/patrickd/.claude/projects/-home-patrickd/memory/reap-criterion-and-corpus-audit.md](/home/patrickd/.claude/projects/-home-patrickd/memory/reap-criterion-and-corpus-audit.md). It has no end-to-end test, and v3 used HOPE on `reap_1_1_1`, not `mass` — [/home/patrickd/glm-5.3-reap/scripts/stages/s04b_surgery.py](/home/patrickd/glm-5.3-reap/scripts/stages/s04b_surgery.py).
- **Sequence length vs token count:** an S=8192 vs S=2048 keep-set agrees at 0.9745 mean (KDA 0.9757, DSA 0.9722), while the same S on disjoint tokens agrees at only 0.8380 at 131k tokens. Tokens matter more than length; the prediction that KDA would be S-sensitive was falsified. — [/home/patrickd/glm-5.3-reap/wiki/93-seqlen-ladder-results.md](/home/patrickd/glm-5.3-reap/wiki/93-seqlen-ladder-results.md)
- **Position:** cross-quant hidden-state agreement rises with position: 0.63 at pos 0, 0.82 at 16–64, 0.913 past 1024. — [/home/patrickd/.claude/projects/-home-patrickd/memory/glm53-quant-hidden-state-drift.md](/home/patrickd/.claude/projects/-home-patrickd/memory/glm53-quant-hidden-state-drift.md)
- **Affected rate:** on MiMo's mask only 1.13% of routed slots hit a pruned expert (8.71% of tokens; layer 7 is the outlier at 18.18% of slots). This is why router KD had no work to do. — [/home/patrickd/xiaomi-2.6-flash-REAP/wiki/98-router-kd-global.md](/home/patrickd/xiaomi-2.6-flash-REAP/wiki/98-router-kd-global.md)
- **Teacher-forced metrics are blind to generation failures:** top-1/ΔNLL measure distance from the teacher, not capability. — [/home/patrickd/glm-5.3-reap/wiki/97-evaluation.md](/home/patrickd/glm-5.3-reap/wiki/97-evaluation.md). Prior synthesis says the loop gate needs n≥2000 free-running prompts and BFCL multi_turn_base is the collapse diagnostic (external literature, not our measurement) — [/home/patrickd/glm-5.3-reap/research_notes/Compressing 1T MoE for Thor/never_prune_guards.md](/home/patrickd/glm-5.3-reap/research_notes/Compressing%201T%20MoE%20for%20Thor/never_prune_guards.md)

### Inferences
- (inference) For V4.1, rank candidate masks with `heal_perexpert.py --keep-set` (~8 min per mask on GLM) and per-domain retention, then accept only on a paired McNemar top-1/KL arm. Size the held-out set from 241,516 tokens giving s.e. 0.00075 unpaired and 0.00047 paired.

### Gaps
- No measurement exists of whether per-domain retention predicts generative benchmarks (only teacher-forced agreement), nor of free-running loop rates on any of our REAPs.

---

## Q3. What failures, OOMs and traps cost the most time, and what guards now prevent them?

### Takeaway
On Thor the dominant cost was memory behaviour no in-process model could see. Fixes: one layer or block per process, `drop_caches 3` with a verified return to baseline, reclaim on every unit of work, and a memguard that only kills resumable named stages. The second-biggest cost was confident-but-wrong stages: a silent heal no-op, placeholder scoring, a capture that captured nothing, a resume double-count, and a guard that downgraded itself to a WARN. Fix: artifact-level verification by something that did not compute the result.

### Cited Findings
**Memory**
- Per-layer memory is never returned in-process. Measured: a layer forward costs −13.54 GiB and `empty_cache` returns only +0.03 GiB. `s03` would need 259 GiB in one process. Process exit plus `echo 3 > drop_caches` returned 88.71→121.23 GiB; `echo 1` is insufficient. — [/home/patrickd/glm-5.3-reap/wiki/96-the-real-oom-cause.md](/home/patrickd/glm-5.3-reap/wiki/96-the-real-oom-cause.md)
- `LAYER_COST_GIB` was 5.76 (the weight size), but the true host footprint is ~53 GiB. Blocks of 9, then 5, then 2 layers all died. The fix was one layer per process, with reclaim keyed to a memory floor. A floor below the cost of one unit is useless: the capture costs 57 GiB per unit, so reclaim is unconditional. — [/home/patrickd/glm-5.3-reap/wiki/31-pass3-hope-method.md](/home/patrickd/glm-5.3-reap/wiki/31-pass3-hope-method.md)
- A 9-layer block took ~22 min, and the 50-block sweep took ~18 h. 113.8 GiB was available before a block and 113.4 after. — [/home/patrickd/glm-5.3-reap/wiki/96-the-real-oom-cause.md](/home/patrickd/glm-5.3-reap/wiki/96-the-real-oom-cause.md)
- A 2048 MiB cudaMalloc charges only 42 MiB to the cgroup, so `MemoryMax` cannot bound it and the OOM killer scores by RSS. In one collapse all processes summed to 2.06 GiB while 115 GiB was held. memguard tier-1 `echo 1` reclaimed 0 bytes ~8,600 times a day and logged it as success; it now logs the delta and treats a no-op as a failure. — [/home/patrickd/xiaomi-2.6-flash-REAP/research/OOM_CONTAINMENT.md](/home/patrickd/xiaomi-2.6-flash-REAP/research/OOM_CONTAINMENT.md), [/home/patrickd/.claude/projects/-home-patrickd/memory/thor-unified-memory-oom.md](/home/patrickd/.claude/projects/-home-patrickd/memory/thor-unified-memory-oom.md)
- The MiMo vision tower allocated ~70 GB in 5 s: dense `[1,32,L,L]` sink-bias copies at L=21,760 → 30.3 GB each. `max_pixels` was ignored by the processor. Fixes: `MAX_PATCH_ROWS=4096` (4.73 GiB peak), towers in a disposable `oom_score_adj=1000` subprocess, stages at 500, a 90 GiB preflight floor, 5 s memory trace, and restarts capped at 6 per 6 h. — [/home/patrickd/xiaomi-2.6-flash-REAP/research/OOM_CONTAINMENT.md](/home/patrickd/xiaomi-2.6-flash-REAP/research/OOM_CONTAINMENT.md)
- An enabled `glm53-reap.service` auto-started a second 306 GB streaming pass alongside an experiment, which caused two wedges. memguard's whitelist then killed the healthy 47-minute stage. — [/home/patrickd/glm-5.3-reap/wiki/94-memguard-killed-the-wrong-process.md](/home/patrickd/glm-5.3-reap/wiki/94-memguard-killed-the-wrong-process.md)
- The `diag_router_kd` fixture allocated 275 GB (`[N,C,H,H]` gather). After the OOM, a suspend/resume could not allocate the GSP buffer and the GPU dropped off the bus, halting pass 3. `Restart=on-failure` then re-ran 3× against the dead device. — [/home/patrickd/glm-5.3-reap/wiki/00-log.md](/home/patrickd/glm-5.3-reap/wiki/00-log.md)
- Co-running a trainer beside a vLLM server hard-rebooted the box (~12 h lost). Subagents count as co-running. — [/home/patrickd/.claude/projects/-home-patrickd/memory/thor-unified-memory-oom.md](/home/patrickd/.claude/projects/-home-patrickd/memory/thor-unified-memory-oom.md)
- `expandable_segments:True` is set in the service, but upstream disables it on Tegra (`[OPEN]`). — [/home/patrickd/glm-5.3-reap/wiki/96-the-real-oom-cause.md](/home/patrickd/glm-5.3-reap/wiki/96-the-real-oom-cause.md)
- A session-child eval died when SSH dropped (16 h lost). Rule: `setsid`/systemd user unit, and verify parentage. — [/home/patrickd/.claude/projects/-home-patrickd/memory/detach-unattended-local-workloads.md](/home/patrickd/.claude/projects/-home-patrickd/memory/detach-unattended-local-workloads.md)

**Silent-wrong stages**
- Heal ledger keyed on shard filenames, which repeat across passes, so pass 2 "healed" 0 tensors while the card described per-expert healing. The ledger is now fingerprinted `{target, keep_set_sha}` and the stage raises on zero scaled; `verify_heal.py` reads the weights back. — [/home/patrickd/glm-5.3-reap/wiki/70-healing.md](/home/patrickd/glm-5.3-reap/wiki/70-healing.md)
- Eval scored image placeholders: teacher NLL 16.94, sign inverted, top-1 reported as 0.674 vs a true 0.837. The fix belongs in `compare()`, not capture, because the teacher capture is irreplaceable after surgery. — [/home/patrickd/glm-5.3-reap/wiki/97-evaluation.md](/home/patrickd/glm-5.3-reap/wiki/97-evaluation.md)
- `capture_teacher_routing` read `prev_topk_indices` (None) as expert ids and logged "captured" for every layer. Only the orchestrator's "exited 0 but wrote nothing" check caught it. `sorted(glob("*.pt"))[0]` also broke once `accumulators.pt` existed; `common.layer_dumps()` fixes it. — [/home/patrickd/glm-5.3-reap/wiki/31-pass3-hope-method.md](/home/patrickd/glm-5.3-reap/wiki/31-pass3-hope-method.md)
- Resume double-count: accumulators were dumped per layer but resumed per chunk, so the interrupted chunk got double weight. The ledger is now keyed `(chunk, block)` and `gate_s03_blocks.py` passes 16/16. Five s03 defects were found by gating; four produced plausible wrong answers. — [/home/patrickd/glm-5.3-reap/wiki/96-the-real-oom-cause.md](/home/patrickd/glm-5.3-reap/wiki/96-the-real-oom-cause.md), [/home/patrickd/glm-5.3-reap/wiki/97-end-to-end-audit.md](/home/patrickd/glm-5.3-reap/wiki/97-end-to-end-audit.md)
- Bio was dead by licence. `assert_domains_live` refused, so `conf/structurally_empty_domains.txt` now distinguishes "cannot populate" from "partial pass". s04b failed 5× in 8 min on DeadDomainError before that. — [/home/patrickd/glm-5.3-reap/conf/structurally_empty_domains.txt](/home/patrickd/glm-5.3-reap/conf/structurally_empty_domains.txt), [/home/patrickd/glm-5.3-reap/logs/s04b_surgery.log](/home/patrickd/glm-5.3-reap/logs/s04b_surgery.log)
- **New, found here:** the s05 keep-set guard raises but sits inside `except Exception`, so a mismatched `heal_refit.json` falls back to the known-biased first-moment gain with only a WARN. This is how v3 shipped. — [/home/patrickd/glm-5.3-reap/scripts/stages/s05_heal.py](/home/patrickd/glm-5.3-reap/scripts/stages/s05_heal.py), [/home/patrickd/glm-5.3-reap/logs/s05_heal.log](/home/patrickd/glm-5.3-reap/logs/s05_heal.log)
- Blocked visit order put MiMo media at 74–85% of a 33-h pass. Interleaving moved image 74%→11%, audio 81%→0%, video 85%→4%. Worst-domain objectives are identically 0 while any domain is empty. — [/home/patrickd/.claude/projects/-home-patrickd/memory/calibration-corpus-visit-order.md](/home/patrickd/.claude/projects/-home-patrickd/memory/calibration-corpus-visit-order.md)
- 9 of 15 image sources (52% of image weight) yielded zero images because they were path-only. Per-source weights were ignored (greedy fill in list order). Fix: probe yield at runtime, split by weight, enforce an underfill floor. — [/home/patrickd/.claude/projects/-home-patrickd/memory/media-corpus-sources-silently-dead.md](/home/patrickd/.claude/projects/-home-patrickd/memory/media-corpus-sources-silently-dead.md)
- Editing a running bash script shifts its offsets. Late-bound decisions go through a file read fresh (`conf/protect_frac_override.txt`). — [/home/patrickd/.claude/projects/-home-patrickd/memory/editing-a-running-bash-script.md](/home/patrickd/.claude/projects/-home-patrickd/memory/editing-a-running-bash-script.md)

**Quant and eval traps**
- `llama-imatrix --save-frequency` wrote 34 × 264 MB files, filled the disk, and exited rc=0 with a truncated `.dat`. `imatrix_verify.py` parses the file, including the required trailer. — [/home/patrickd/.claude/projects/-home-patrickd/memory/llama-imatrix-save-frequency-trap.md](/home/patrickd/.claude/projects/-home-patrickd/memory/llama-imatrix-save-frequency-trap.md)
- GGUF keeps a single eos id. GLM ends turns with `<|user|>` 154827, so the fix is `--override-kv tokenizer.ggml.eot_token_id=int:154827` and `eom_token_id=int:154829`; check the `printing all EOG tokens` line. — [/home/patrickd/.claude/projects/-home-patrickd/memory/gguf-multi-eos-token-trap.md](/home/patrickd/.claude/projects/-home-patrickd/memory/gguf-multi-eos-token-trap.md)
- A `$!` caught the subshell, so 2 of 3 head-to-head arms re-measured GLM (~13 GPU-h). Fix: `exec`, assert served identity, check liveness before health. — [/home/patrickd/.claude/projects/-home-patrickd/memory/eval-orphaned-server-trap.md](/home/patrickd/.claude/projects/-home-patrickd/memory/eval-orphaned-server-trap.md)
- Unpinned `reasoning_effort` (GLM server default Max never terminates) produced 41.5% vs 77.4% HumanEval. A fixed budget measures tokens-to-answer, and pass@1 inverted once truncation was conditioned on. Changing `--seqmax` perturbs greedy output. — [/home/patrickd/.claude/projects/-home-patrickd/memory/eval-pin-generation-config.md](/home/patrickd/.claude/projects/-home-patrickd/memory/eval-pin-generation-config.md), [/home/patrickd/.claude/projects/-home-patrickd/memory/eval-fixed-budget-measures-economy.md](/home/patrickd/.claude/projects/-home-patrickd/memory/eval-fixed-budget-measures-economy.md), [/home/patrickd/.claude/projects/-home-patrickd/memory/eval-seqmax-perturbs-greedy-output.md](/home/patrickd/.claude/projects/-home-patrickd/memory/eval-seqmax-perturbs-greedy-output.md)

### Inferences
- (inference) V4.1 checklist, all already proven on GLM or MiMo:
  - `mem_lookahead` + `memfence.require()` before each layer;
  - one process per block, with a verified `drop_caches 3` between blocks;
  - unconditional reclaim when a unit costs more than the floor;
  - the `(chunk, block)` ledger;
  - interleaved domain order;
  - a source-yield probe;
  - teacher routing and teacher eval captured **before** surgery;
  - fail-hard keep-set fingerprints on every derived artifact;
  - the s09 class-A leak fixed (s09 sweeps 45 layers in one process and is "AT RISK").

### Gaps
- Blocked vs unblocked accumulator equivalence on real weights is `[OPEN]` — [/home/patrickd/glm-5.3-reap/wiki/96-the-real-oom-cause.md](/home/patrickd/glm-5.3-reap/wiki/96-the-real-oom-cause.md). Whether the s09 leak was fixed after the audit is not recorded beyond commit `ace3221` ("reclaim the nvmap pool mid-sweep") — [/home/patrickd/glm-5.3-reap](/home/patrickd/glm-5.3-reap) git log.

---

## Q4. What did the MiMo-V2.6-Flash REAP (MXFP4-source experts) teach about pruning pre-quantized experts?

### Takeaway
On MXFP4-source experts, deletion is bit-exact and was the only lever. The 50% ratio was forced by bytes, not chosen. HOPE + protect_frac 0.08 + uniform allocation shipped, and router KD was a provable no-op. The checkpoint was published but **never served, benchmarked or evaluated end-to-end**, so its quality is unmeasured.

### Cited Findings
- Experts are 158 GB of packed MXFP4 `U8` (92.9% of the 172.9 GB index), so "the quant lever is spent". REAP-50 = 94.7 GB weights + 11.5 KV@1M + 2.94 DFlash + ~5 workspace = 114.1 GB. REAP-40 = 129.7 GB, which does not fit. — [/home/patrickd/xiaomi-2.6-flash-REAP/README.md](/home/patrickd/xiaomi-2.6-flash-REAP/README.md)
- Run: 27/27 chunks, criterion `reap_1_1_1` / mode `hope` at 0.50, uniform, protect_frac 0.08, worst domain audio at 0.99345; 128/256 kept on all 47 layers. Router KD kept the teacher router on 47/47 layers. — [/home/patrickd/xiaomi-2.6-flash-REAP/logs/reap_run.log](/home/patrickd/xiaomi-2.6-flash-REAP/logs/reap_run.log)
- protect_frac 0.08 was measured on 23/27 chunk accumulators: it buys 53% of the worst-domain residual loss for +1.15% pFp, and fixes 35 of 128 kept slots per layer. pf=0.01 was *worse* than 0, so the curve is not monotone near zero. — [/home/patrickd/xiaomi-2.6-flash-REAP/conf/protect_frac_override.txt](/home/patrickd/xiaomi-2.6-flash-REAP/conf/protect_frac_override.txt)
- GLM did not inherit 0.08: on GLM it sits at marginal efficiency 0.43, and GLM's knee is 0.04. — [/home/patrickd/glm-5.3-reap/wiki/31-pass3-hope-method.md](/home/patrickd/glm-5.3-reap/wiki/31-pass3-hope-method.md)
- Router KD (layer-local) results: lr 1e-3 cost −299.5% on the held-out population, 1e-5 cost −7.5%. The fixture bias was drawn from RMS instead of spread (0.059), which gave a fake 0.00% affected rate. — [/home/patrickd/xiaomi-2.6-flash-REAP/wiki/98-router-kd-global.md](/home/patrickd/xiaomi-2.6-flash-REAP/wiki/98-router-kd-global.md)
- noaux_tc formula: the bias selects and the unbiased sigmoid weights. The naive formula agreed with the module on 1 of 8 experts (0.125) with 1.77e-1 max weight error; the correct one matches to 2.98e-08. The bias has zero gradient. — [/home/patrickd/.claude/projects/-home-patrickd/memory/noaux-tc-router-formula.md](/home/patrickd/.claude/projects/-home-patrickd/memory/noaux-tc-router-formula.md)
- HOPE solver gates: projected gradient hit the optimum on 27/40 random problems, and swap refinement raised that to 40/40. `F[k,k]==sq/cnt` caught a valid-mask deflation. F costs 47 MiB for 47×256 and solves in 0.47 s/layer. The off-diagonal cannot be recovered after a pass. — [/home/patrickd/xiaomi-2.6-flash-REAP/README.md](/home/patrickd/xiaomi-2.6-flash-REAP/README.md), [/home/patrickd/.claude/projects/-home-patrickd/memory/hope-supersedes-reap.md](/home/patrickd/.claude/projects/-home-patrickd/memory/hope-supersedes-reap.md)
- Corpus: 27 chunks, 48.99M tokens at S=4096; media 6.0M tokens (12.3%); 61.9% of rows carry a span over 4,000 tokens. Video coverage after 5 chunks was 35.4% mean / 6.2% min per-domain expert coverage, and it is diversity-limited. — [/home/patrickd/xiaomi-2.6-flash-REAP/wiki/91-capability-coverage.md](/home/patrickd/xiaomi-2.6-flash-REAP/wiki/91-capability-coverage.md), [/home/patrickd/.claude/projects/-home-patrickd/memory/media-corpus-sources-silently-dead.md](/home/patrickd/.claude/projects/-home-patrickd/memory/media-corpus-sources-silently-dead.md)
- GGUF: MXFP4 repacked natively into `GGMLQuantizationType.MXFP4` (MXFP4_MOE 86.06 GiB, "exact bit-level fidelity"). Q2_K hybrid 61.64 GiB keeps down_proj in MXFP4. — [/home/patrickd/xiaomi-2.6-flash-REAP/gguf_README.md](/home/patrickd/xiaomi-2.6-flash-REAP/gguf_README.md). (The card's HOPE description, "cross-expert Hessian blocks", does not match the actual pᵀFp of g·‖f‖ co-activation statistics — [/home/patrickd/.claude/projects/-home-patrickd/memory/hope-supersedes-reap.md](/home/patrickd/.claude/projects/-home-patrickd/memory/hope-supersedes-reap.md).)
- Never served or benchmarked; "~5 tok/s" is arithmetic. Source deleted (166 GB), so global KD needs a re-download. — [/home/patrickd/.claude/projects/-home-patrickd/memory/mimo-v26-flash-thor.md](/home/patrickd/.claude/projects/-home-patrickd/memory/mimo-v26-flash-thor.md), [/home/patrickd/xiaomi-2.6-flash-REAP/wiki/98-router-kd-global.md](/home/patrickd/xiaomi-2.6-flash-REAP/wiki/98-router-kd-global.md)
- Prior synthesis: on FP4-source experts only bit-exact ops are free; REAM loses to REAP by 13 pt at 48% on FP4. — [/home/patrickd/.claude/projects/-home-patrickd/memory/compressing-1t-moe-thor-research.md](/home/patrickd/.claude/projects/-home-patrickd/memory/compressing-1t-moe-thor-research.md)

### Inferences
- (inference) V4.1 experts are MXFP4 with E8M0 scales [`float8_e8m0fnu`, `[out, in//32]`] — [/home/patrickd/glm-5.3-reap/research_notes/Compressing 1T MoE for Thor/v41_architecture_pruning_delta.md](/home/patrickd/glm-5.3-reap/research_notes/Compressing%201T%20MoE%20for%20Thor/v41_architecture_pruning_delta.md). Consequences:
  - GLM's healing trick of multiplying F32 block scales is **not** available losslessly. An E8M0 scale can only shift by powers of 2, so a ~0.9 scalar gain must instead go on the dense shared/output side, or be folded into a re-quant. That needs a design decision.
  - MiMo shipped without any heal, which is consistent with this.

### Gaps
- MiMo REAP-50 has no top-1/KL/PPL vs teacher, no NIAH, and no tok/s, so "HOPE+pf on MXFP4 at 50%" has no end-to-end quality number from us. GLM and MiMo retention numbers use different rulers (GLM worst_ret ~0.54–0.56, MiMo ~0.99), and the files do not define them comparably.

---

## Q5. What is directly reusable for a V4.1 pipeline, and what needs rework (Engram, 384 experts, MXFP4, MLA/CSA)?

### Takeaway
Reusable as-is:
- the orchestrator/state/memguard/memfence stack;
- the HOPE F accumulator and solver;
- `sweep_protect_frac.py`;
- split-half-on-mass;
- the per-bucket accumulators with offline re-weighting;
- the paired s09 compare logic;
- the corpus spec with its licence filters;
- the imatrix/EOS/eval guards.

Rework needed:
- the saliency and router patching against the reference `inference/model.py` (no HF class);
- the sqrtsoftplus/bias_vl router replay;
- Engram pre-hooks at layers 1/14;
- row-chunked indexer memory;
- healing on E8M0 scales;
- vLLM-legal expert counts.

### Cited Findings
- V4.1: 384 routed experts + 1 shared, top-6, `sqrtsoftplus` + noaux_tc, `routed_scaling_factor 1.5`, no n_group, 40 MoE layers. The vLLM fused router accepts {128,192,256,320,384}, so 192 = exactly 50%. There is no HF transformers class: patch `inference/model.py`. — [/home/patrickd/glm-5.3-reap/research_notes/Compressing 1T MoE for Thor/v41_architecture_pruning_delta.md](/home/patrickd/glm-5.3-reap/research_notes/Compressing%201T%20MoE%20for%20Thor/v41_architecture_pruning_delta.md), [/home/patrickd/glm-5.3-reap/research_notes/Compressing 1T MoE for Thor/_COORDINATOR_STATE.md](/home/patrickd/glm-5.3-reap/research_notes/Compressing%201T%20MoE%20for%20Thor/_COORDINATOR_STATE.md)
- mHC delta ≈ 0: GLM already has hc_mult=4 and the local hook sits after `hc_pre`. Engram is applied before blocks 1 and 14 and must be in the sweep; switching Engram off roughly doubles PPL. `bias_vl` drives image-span selection, and text-only ranking is worse than random for vision, so rank per modality. The shared expert must be in the heal norm target. Indexer score tensor: 4 GiB/row at 8k, so evaluate row-wise; 8k × 16 then peaks at ~47 GiB. These are prior-synthesis inferences from the reference code, not our measurements. — [/home/patrickd/glm-5.3-reap/research_notes/Compressing 1T MoE for Thor/v41_architecture_pruning_delta.md](/home/patrickd/glm-5.3-reap/research_notes/Compressing%201T%20MoE%20for%20Thor/v41_architecture_pruning_delta.md)
- Bytes: V4.1-1017 is 475 GiB total; experts MXFP4 268.95 GiB; Engram 189 GiB; floor without Engram 9.77 GiB; KV 890 B/tok. Engram can live on an NVMe O_DIRECT row store (measured 1.13–1.30 ms/tok here) but not on mmap. NVMe has 51 GB free, so a second drive is needed. — [/home/patrickd/.claude/projects/-home-patrickd/memory/thor-base-model-shortlist.md](/home/patrickd/.claude/projects/-home-patrickd/memory/thor-base-model-shortlist.md), [/home/patrickd/.claude/projects/-home-patrickd/memory/compressing-1t-moe-thor-research.md](/home/patrickd/.claude/projects/-home-patrickd/memory/compressing-1t-moe-thor-research.md)
- Super-Expert guard: neither `stream_saliency.py` nor `mimo_saliency.py` keeps a per-expert max statistic (grep for amax/max/super/shared found nothing). Prior synthesis says to bank per-expert MAX down_proj output in the same pass and never prune the shared expert. — [/home/patrickd/glm-5.3-reap/scripts/stream_saliency.py](/home/patrickd/glm-5.3-reap/scripts/stream_saliency.py), [/home/patrickd/glm-5.3-reap/research_notes/Compressing 1T MoE for Thor/never_prune_guards.md](/home/patrickd/glm-5.3-reap/research_notes/Compressing%201T%20MoE%20for%20Thor/never_prune_guards.md)
- REAP logit-renorm fix: our hook uses the renormalised `top_k_weights`, which already matches upstream 2026-03-11. Upstream hard-codes softmax, which is wrong for sigmoid routers. — [/home/patrickd/glm-5.3-reap/research_notes/Compressing 1T MoE for Thor/v41_architecture_pruning_delta.md](/home/patrickd/glm-5.3-reap/research_notes/Compressing%201T%20MoE%20for%20Thor/v41_architecture_pruning_delta.md), [/home/patrickd/glm-5.3-reap/research_notes/Compressing 1T MoE for Thor/router_post_prune_rules.md](/home/patrickd/glm-5.3-reap/research_notes/Compressing%201T%20MoE%20for%20Thor/router_post_prune_rules.md). Note that memory `reap-logit-renorm-fix` still frames this as "may predate"; the newer delta note (2026-10-17) resolves it.
- Calibration recommendation from prior synthesis: gen20/agent15/code15/vis14/math12/sci12/fin10, with a 10% floor and ≥250k tokens per domain. Ours had general at 4.76% (pass 2) and 15% target / 17.58% realised (pass 3). — [/home/patrickd/.claude/projects/-home-patrickd/memory/compressing-1t-moe-thor-research.md](/home/patrickd/.claude/projects/-home-patrickd/memory/compressing-1t-moe-thor-research.md), [/home/patrickd/.claude/projects/-home-patrickd/memory/reap-criterion-and-corpus-audit.md](/home/patrickd/.claude/projects/-home-patrickd/memory/reap-criterion-and-corpus-audit.md)
- Damage ratio: REAP50 dNLL 0.1909 ≈ +21.0% PPL vs NVFP4 +1.56%, so pruning is ~14× more damaging per unit of compression than quantisation. — [/home/patrickd/.claude/projects/-home-patrickd/memory/reap-criterion-and-corpus-audit.md](/home/patrickd/.claude/projects/-home-patrickd/memory/reap-criterion-and-corpus-audit.md). **Contradiction:** the same memory cites a "GGUF bit-width step = +1.46%" from the 8-chunk PPL runs, which the GGUF card says are within noise — [/home/patrickd/glm-5.3-reap/gguf_README.md](/home/patrickd/glm-5.3-reap/gguf_README.md).

### Inferences
- (inference) Directly portable modules:
  - `hope_fmatrix.py` (resize to 384 experts: 384² × 4 B = 0.56 MiB/layer);
  - `sweep_protect_frac.py` (re-measure the knee; do not inherit 0.04 or 0.08);
  - `split_half.py`;
  - `heal_perexpert.py --keep-set` as a mask filter only;
  - `router_replay.py`/`heal_refit.py` logic (re-derived for sqrtsoftplus and two biases);
  - `memguard.sh`, `memfence.py`, `mem_lookahead.py`, `probe_layer_mem.py`;
  - `pipeline.py`/`run_stage.py`/`state.db`;
  - `s09_eval.compare` (placeholder exclusion, `sufficient` flag);
  - `heal_paired_test.py` (McNemar);
  - `corpus_spec.py` licence exclusions;
  - `imatrix_verify.py`, `gguf_card.py`.
- (inference) Needs rework:
  - `s03` patch targets and the per-layer build (reference model, FP4 linear, shared-attn state for compressor/indexer layers, Engram pre-hook);
  - `s04b` surgery for MXFP4 + E8M0 tensors and `gate.bias`/`bias_vl` slicing;
  - `s05` healing, since E8M0 scales cannot take a 0.9× multiply;
  - `s07`, since experts stay MXFP4 or go to trellis/IQ ~2.9 bpw per prior synthesis;
  - a per-modality ranking for vision.

### Gaps
- Nothing in our files measures any V4.1 forward on Thor (throughput, per-layer memory, Engram in-loop cost during calibration). The ~47 GiB peak is arithmetic. `gate_temp` for 1017 is unconfirmed.
- **Contradictions to resolve:**
  - README v3 table "per-expert" vs `s05_heal.json` `per_expert_tensors: 0`;
  - README "v2 per-expert" vs the HEALING_ABLATION revert to scalar;
  - wiki/30 "more tokens don't help" (split-half at 2.75M, 2026-08-28) vs wiki/93b "token count is the lever" (131k-token control, 2026-09-27). The newer one concerns a 42× smaller budget, so the two are not truly in conflict.
