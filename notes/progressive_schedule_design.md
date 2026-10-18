# Progressive / iterative expert-pruning schedule design (REAP→HEAL→REAP) for 60–75% total removal on a 763B–1T MoE, layer-streamed on one 122 GiB Thor

Scope note: this file covers *schedule design* (stage count, per-stage ratio, between-stage operation, max-ratio extension, failure modes, Thor cost). Healing methods per se are in `healing_recovery.md`; >50% one-shot evidence is in `expert_pruning_beyond_50.md`. Where I rely on those sibling notes I say so; everything else is cited to the primary source. All "[extrapolation]" tags are mine.

## Q1. Which paper produced Mixtral 14.74→10.44, and what was the exact schedule?

### Takeaway
It is Jaiswal et al., "Finding Fantastic Experts in MoEs" (arXiv 2504.05586), Table 2: Mixtral-8×7B Base, C4 ppl (dense 7.44), Min-EAN criterion, one-shot 14.74 vs iterative 10.44 vs "MoE Lottery" (iterative + ~1M-token FT per round) 9.76 at 50% dropped. The schedule is the finest possible: one expert per layer per round (12.5% of experts per round), criterion re-estimated on the pruned model after every round, so 50% = 4 rounds, 75% = 6 rounds.

### Cited Findings
- Procedure: "32 experts (12.5%), one per layer per round"; the criterion "is re-estimated after each round"; a *global* criterion "creates bottleneck layers", which is why they drop exactly one expert per layer per round. — [Finding Fantastic Experts, arXiv 2504.05586, §3.1 / Fig. 3](https://arxiv.org/html/2504.05586v2)
- Calibration per round: "task-agnostic C4 validation set of 256 samples with max_seq_len 2048" (≈0.5M tokens per re-rank). — [2504.05586, App. C](https://arxiv.org/html/2504.05586v2)
- Full Table 2, Mixtral-8×7B Base, C4 ppl (dense 7.44), columns = one-shot / iterative / MoE-Lottery:
  - Min-EAN: 12.5% 7.95/7.90/7.89; 25% 8.56/8.53/8.38; 37.5% 12.87/9.35/9.00; 50% 14.74/10.44/9.76; 62.5% 21.36/12.55/11.00; 75% 30.59/17.39/13.05.
  - Min-EGE: 12.5% 7.89/7.89/7.88; 25% 8.47/8.41/8.26; 37.5% 13.33/9.48/9.09; 50% 15.37/10.72/9.88; 62.5% 22.21/12.81/11.00; 75% 35.83/17.70/13.09.
  - Random: 12.5% 11.25/7.94/7.89; 25% 12.74/10.98/11.01; 37.5% 13.89/13.19/12.22; 50% 17.08/15.85/14.13; 62.5% 30.41/18.79/20.60; 75% 36.92/32.73/27.33. — [2504.05586, Table 2](https://arxiv.org/html/2504.05586v2)
- Downstream Table 3 (Base, zero-shot; dense MMLU 60.01, WinoGrande 56.59), values at 12.5→75% in 12.5% steps:
  - MMLU one-shot 52.97, 43.97, 13.55, 18.91, 12.63, 5.82; iterative 48.51, 47.81, 45.63, 35.74, 29.71, 23.88; lottery 49.54, 49.65, 47.13, 40.79, 37.24, 28.12.
  - WinoGrande one-shot 55.13, 50.09, 37.45, 36.91, 20.44, 24.63; iterative 55.90, 52.17, 49.96, 48.53, 47.11, 50.35; lottery 55.92, 52.98, 50.96, 49.56, 49.30, 50.74. — [2504.05586, Table 3](https://arxiv.org/html/2504.05586v2)
- Disagreement: Fig. 5 is qualitative only — "significantly high disagreement" between one-shot and lottery expert sets, "notable high agreement" between iterative and lottery at 50%. No numeric Jaccard/overlap is reported. — [2504.05586, Fig. 5](https://arxiv.org/html/2504.05586v2)
- MoE-Lottery FT: tokens per round double starting from 0.2M; AdamW, cosine, max LR 1e-6, batch 8, optimizer reset every round, 8×A100; "about 1M tokens suffice" (Table 5, 75% Min-EAN, Base: 13.55 / 13.51 / 13.05 / 13.01 ppl at 0.25M / 0.51M / 1.13M / 2.27M tokens). — [2504.05586, §3.3, App. B, Table 5](https://arxiv.org/html/2504.05586v2)
- REAP evaluates "in the one-shot setting, with no additional fine-tuning" and cites the iterative benefits of Jaiswal et al. without testing them; Cerebras lists "iterative pruning schedules" as future work. — [REAP, arXiv 2510.13999](https://arxiv.org/html/2510.13999v2); [Cerebras blog](https://www.cerebras.ai/blog/reap)
- MoE-Pruner (2410.12013) is **not** the source: it is explicitly "one-shot, requiring no retraining or weight updates", prunes *weights inside* experts (Wanda-style S = |W|·‖X·Gate‖), and reports no iterative results. — [MoE-Pruner, arXiv 2410.12013](https://arxiv.org/html/2410.12013)

### Inferences
- Of the 50%-point gap (14.74 → 9.76), re-ranking alone closes 4.30 of 4.98 ppl points (86%); the per-round ~1M-token FT adds only 0.68. The mechanism is overwhelmingly the saliency recomputation on the already-pruned model, not the heal. [inference from Table 2]
- The 12.5% rows should be identical for one-shot and iterative (one round) but differ slightly (7.95 vs 7.90 ppl; 52.97 vs 48.51 MMLU), so there is some run-to-run / calibration-sample noise of order 0.05 ppl and several MMLU points in these tables; treat sub-point differences as noise. [inference]
- Jaiswal's schedule is "linear, one expert per layer per round". For a 256–384-expert layer the analogous unit step is 1 expert/layer/round = 0.26–0.39% per round, which is not a usable schedule on Thor (Q9); the paper gives no evidence that coarser rounds lose the benefit, but SlimMoE/SlimQwen/Minitron (Q2) do.

### Gaps
- No numeric one-shot-vs-iterative expert-set overlap is published; no experiment on a >8-expert-per-layer model; no result on a model where one-shot 50% is near-lossless (REAP-class models).

## Q2. Diminishing returns: at what stage count does iteration stop improving?

### Takeaway
Every controlled comparison that varies stage count finds 2–3 stages captures essentially all of the benefit: SlimQwen 3-stage = 2-stage; SlimMoE multi-stage (T=3) = fully iterative at 1/2.4 the initialization cost; Minitron T=1/2/4 converge to the same loss after retraining; Nemotron Puzzle only compared 1 vs 3. No MoE-expert-deletion study sweeps stage count without fine-tuning.

### Cited Findings
- SlimQwen (Qwen3-Next-80A3B → 23A2B, 400B tokens): one-stage MMLU 75.86; two-stage depth-first (40B + 360B) 77.39; three-stage (20B + 20B + 360B) depth-first 77.29, width-first 76.46 — three-stage "achieve[s] performance comparable to the two-stage setup … a two-stage progressive pruning strategy is already sufficient". — [SlimQwen, arXiv 2605.08738, Tables 5, 9](https://arxiv.org/pdf/2605.08738) (numbers as extracted in sibling note `expert_pruning_beyond_50.md`)
- SlimMoE (Phi-3.5-MoE 42B → Phi-tiny-MoE 3.8B, expert *width* slimming, 16 experts kept): one-stage MMLU 58.40 / GSM8K 62.89; iterative 60.05 / 69.60; multi-stage (T=3) 60.08 / 69.90; iterative "requires loading the full-size model … approximately 2.4× computation time during initialization phase"; multi-stage reaches the one-stage final score at 0.74× (mini) / 0.91× (tiny) of the compute; intermediate stages consume "30–35% of the total training steps". — [SlimMoE, arXiv 2506.18349, Table 3, §5](https://arxiv.org/html/2506.18349v1)
- Minitron (Nemotron-4 15B, embedding-dimension pruning to 4096, T = 1/2/4 iterations with importance re-estimated each iteration, then 1.8B-token retrain): zero-shot loss 5.43 / 5.55 / 5.24; after retraining 1.92 / 1.92 / 1.92 — "while the iterative approach appears to be better before retraining, all 3 candidates converge to the same loss value, indicating no benefit" → Best Practice #3 "Use single-shot importance estimation; iterative provides no benefit". — [Minitron, arXiv 2407.14679, Table 14, §4](https://arxiv.org/pdf/2407.14679)
- Minitron depth: iteratively removing 1 layer + 1.8B-token distill ×16 is *worse* than removing 16 layers once + 30.2B tokens, even when the iterative budget is raised to 86.4B tokens; "it is best to follow the one-shot method". — [Minitron, App. A.5](https://arxiv.org/pdf/2407.14679)
- Minitron coarse 2-step (15B → 8B at ~46% → 4B at ~50%, each with 100B-token distillation) vs one-shot 15B → 4B (73.3%): MMLU 42.45 vs 37.81, HellaSwag 52.04 vs 51.04 ("12% improvement in MMLU"). — [Minitron, Table 11, Best Practice #8](https://arxiv.org/pdf/2407.14679)
- Nemotron Puzzle (Nemotron-3-Super 120.7B/12.8B-active → 75.3B/9.3B-active): 3 stages (MoE weights 75% → 60% of teacher capacity → 50% active budget), healed 24B / 43.2B / 52.8B tokens, replacement scores "recomputed in the context of the current compressed model"; 3-step avg 69.05 vs 1-step 68.48 over 10 benchmarks (largest gains AA-LCR +2.2, GPQA +1.5; IFBench −0.2/−0.5). No 2-step or 4+-step arm. — [Nemotron Puzzle, arXiv 2607.04371, §2.3, Table 6](https://arxiv.org/html/2607.04371)
- Dense/CNN benchmark (2508.13836): step-count ablation (Fig. 7) is "non-monotonic"; one-shot beats iterative below ~80% sparsity, iterative wins above (~90% crossover for structured Hessian/Taylor); hybrid schedule = prune 60–80% of the target in one step then small geometric steps. — [One Shot vs. Iterative, arXiv 2508.13836](https://arxiv.org/html/2508.13836v1)

### Inferences
- The stage-count benefit saturates at 2–3 when there is retraining between stages (SlimQwen, SlimMoE, Minitron). Without retraining there is only Jaiswal's single data point (fine iteration, no stage sweep). Combining the two: for a re-rank-only pipeline the gain comes from re-estimating saliency after routing redistributes, and that redistribution is largest after the *first* big cut; a 3-stage schedule should capture most of the fully-iterative gain, with the 4th–5th stage buying little. [extrapolation]
- Minitron's "no benefit" is for *width* axes healed with 1.8B tokens — the retrain erases the ranking advantage. On Thor there is no between-stage gradient heal (Q5), so the Minitron null result does not transfer; the pre-retrain numbers (5.24 vs 5.43 loss at T=4 vs 1) and Jaiswal's no-FT rows are the relevant regime, and both favour iteration. [inference]

### Gaps
- No MoE expert-deletion study varies stage count at fixed total ratio with *no* fine-tuning. The 2-vs-3-vs-5 question is unmeasured in exactly our regime.

## Q3. Ratio per stage: geometric vs linear vs front-loaded vs back-loaded

### Takeaway
The only direct schedule-shape ablations (all with fine-tuning in the loop) favour front-loaded: MC-SMoE cubic > quadratic > linear on switch-base-32; Task-Specific Expert Pruning's threshold-based "eager" drop beats one-expert-per-window "staged" drop by +1.5 GLUE; the dense benchmark recommends 60–80% of the target in one step then small geometric steps. SlimMoE uses equal geometric stages; Jaiswal uses equal linear steps. Nobody has compared shapes for MoE expert deletion without fine-tuning.

### Cited Findings
- MC-SMoE (switch-base-32, Table A14, pruning-ratio schedule of the compression phase): Linear COPA 59.00 / MultiRC 73.83; Quadratic 61.00 / 73.92; Cubic 67.00 / 73.98 — "in the early stages of pruning, an aggressive pruning schedule is less likely to lose useful information in the weights; while it is the opposite in the later stages". Cubic schedule p_t = P_T + (1−P_T)(1 − (t−T_i)/(T−T_i−T_f))^3 per Zhu & Gupta. — [MC-SMoE, arXiv 2310.01334, App. A1.4](https://arxiv.org/pdf/2310.01334)
- Task-Specific Expert Pruning (Chen et al. 2022, Switch-style MoE fine-tuned to a single expert): dynamic threshold T = β·(1/Z) with Z = surviving experts, decided at the end of each of E/2 training windows, force-drop at half-way; "MoE-eager-pruning" (threshold, many per window) GLUE avg 86.16 vs "MoE-staged-pruning" (exactly one per window) 84.63 (MoE-ft 86.80, Dense-ft 85.32); SQuAD2 F1 80.40 vs 80.08. — [Task-Specific Expert Pruning, arXiv 2206.00277, §4.4, Tables 1–2](https://arxiv.org/pdf/2206.00277)
- SlimMoE: "geometric compression schedule", each stage reduces by ≈α^(1/T); Phi-mini d_expert 6400 → 2240 → 960 (×0.35, ×0.43); Phi-tiny 6400 → 2624 → 1024 → 448 (×0.41, ×0.39, ×0.44). — [SlimMoE, Table 7](https://arxiv.org/html/2506.18349v1)
- Dense benchmark: "Iterative constant: p×W/steps removed per step. Iterative geometric: W×(1−p/steps)^n remains after step n"; hybrid "prune 60–80% of the target in one step, then geometric steps" with final per-step ~10% for targets <80%. — [arXiv 2508.13836, §3, Table 3](https://arxiv.org/html/2508.13836v1)
- Jaiswal: fixed 1 expert/layer/round (linear). — [2504.05586](https://arxiv.org/html/2504.05586v2)
- Minitron Best Practice #8: the winning 2-step was ~46% then ~50% of the remainder (roughly geometric-equal). — [Minitron](https://arxiv.org/pdf/2407.14679)

### Inferences
- Front-loading has a mechanistic justification for our pipeline beyond the three citations: the first pass is the only one whose saliency is estimated on unperturbed routing, so it is the highest-confidence ranking; and REAP/HOPE one-shot is empirically near-lossless at 40–50% on 128–384-expert models (sibling notes), so the first stage can take the whole "known-safe" ratio. Later stages are where routing has shifted and where re-ranking earns its cost. [inference]
- Back-loaded schedules have no supporting evidence anywhere in the literature I found.

### Gaps
- No head-to-head of geometric vs linear vs front-loaded for MoE expert deletion without fine-tuning, on any model.

## Q4. Central question: does iteration extend the MAXIMUM tolerable ratio, or only improve quality at ratios both survive?

### Takeaway
Direct evidence (one model, Mixtral-8×7B) says it extends the ratio: on MMLU, one-shot collapses to chance at 37.5% dropped (13.55) while iterative holds 45.63 at 37.5%, 35.74 at 50% and stays above chance to 62.5% (29.71), lottery to 75% (28.12); on C4 ppl, iterative at 62.5% (12.55) beats one-shot at 50% (14.74) and lottery at 75% (13.05) beats one-shot at 50%. At a fixed quality floor the iterative regime therefore sits ~15–25 percentage points of ratio to the right of one-shot. But the quality at those ratios is far below "acceptable" for a general model on an 8-expert MoE; no study tests the question on a 128–384-expert model where one-shot 50% is already near-lossless.

### Cited Findings
- Iso-quality reading of Table 2 (Min-EAN, C4 ppl, dense 7.44): one-shot 14.74 @50%, 21.36 @62.5%, 30.59 @75%; iterative 10.44 @50%, 12.55 @62.5%, 17.39 @75%; lottery 9.76 @50%, 11.00 @62.5%, 13.05 @75%. — [2504.05586, Table 2](https://arxiv.org/html/2504.05586v2)
- Collapse points on MMLU (chance = 25): one-shot 13.55 @37.5%, 18.91 @50%; iterative 45.63 @37.5%, 35.74 @50%, 29.71 @62.5%, 23.88 @75%; lottery 47.13 @37.5%, 40.79 @50%, 37.24 @62.5%, 28.12 @75%. WinoGrande (chance = 50): one-shot 37.45 @37.5%, 20.44 @62.5%; iterative ≥47.11 through 62.5%, 50.35 @75%; lottery ≥49.30 through 75%. — [2504.05586, Table 3](https://arxiv.org/html/2504.05586v2)
- Dense/CNN benchmark: iterative "proves superior for higher pruning rates" (~>80–90%), one-shot superior below; on TinyStories-33M the iterative advantage at high ratios is larger and one-shot "degrades with only 10–20% removed". The paper does not claim a change in *maximum* tolerable sparsity explicitly. — [arXiv 2508.13836](https://arxiv.org/html/2508.13836v1)
- Minitron coarse 2-step vs one-shot at 73.3% total: +4.6 MMLU with the same 100B retrain tokens. — [Minitron, Table 11](https://arxiv.org/pdf/2407.14679)
- SlimMoE attributes the one-stage deficit to "model collapse at the initial pruning step, limiting effectiveness of subsequent distillation". — [SlimMoE, Fig. 1, §5](https://arxiv.org/html/2506.18349v1)
- Over-dispersed-router paper: at r=0.50 on gpt-oss-120B a uniform router-mass criterion "scores 0.00 on every task"; OLMoE-1B-7B MESA sweep retains 92.8% @25%, 83.4% @37.5%, 69.9% @50% — one-shot ratios ≥50% are at or past the cliff on small-E models. — [arXiv 2609.04453](https://arxiv.org/html/2609.04453)
- Sibling note: "no primary source reports a general-purpose MoE surviving >50% expert pruning within 10% of base" — [`expert_pruning_beyond_50.md`, local]

### Inferences
- Define the cliff as "MMLU at chance". On Mixtral it moves from 37.5% (one-shot) to ~62.5–75% (iterative, no FT) to >75% (iterative + 1M-token FT/round). The ratio extension is real and large *relative to the one-shot cliff*. [inference from Table 3]
- Translating to a 256–384-expert, top-8 model: the one-shot cliff is much further right (REAP/HOPE near-lossless at 50% on code/agentic; sibling notes). If the *relative* extension holds, iterative re-ranking should move a 50–55% one-shot cliff to ~65–75%. This is the whole bet of the 60–75% target and it is an **extrapolation from one 8-expert model**; the sibling note's negative (no >50% general-purpose survivor) is one-shot-only, so it neither confirms nor refutes it. [extrapolation]
- The quality metric for "acceptable" must not be ppl: on gpt-oss-20b, pruning that *lowers* ppl lowers GSM8K, Spearman ρ(ppl, GPQA) = −0.10 (Q8). Stage gates should be task evals / per-domain retained mass.

### Gaps
- No iterative-vs-one-shot sweep on any model with ≥64 experts per layer, and none at ≥100B parameters. The central claim for 60–75% on a 763B–1T model is unmeasured; the first stage of our own pipeline will be the first data point.

## Q5. Between-stage operation: re-rank only vs re-rank + routing fold vs re-rank + gradient heal; does early healing change later selections?

### Takeaway
Re-ranking on the pruned model is the dominant lever (86% of the 50% gap on Mixtral, Q1). Gradient healing between stages barely changes which experts are selected later (Jaiswal: iterative vs lottery sets "high agreement"; Minitron: rankings "continue to hold post-retraining"), so skipping between-stage heals costs quality but not mask correctness. A closed-form routing fold (ExFold-style) is a one-pass, gradient-free heal that recovers most of the one-shot REAP loss at 50% and is the only heal that is feasible between stages when the student does not fit in 122 GiB; whether folding *before* re-ranking changes the next stage's selection is unmeasured.

### Cited Findings
- Re-rank only closes 14.74 → 10.44; adding ~1M-token FT per round reaches 9.76 (Mixtral 50%); at 75%: 30.59 → 17.39 → 13.05. — [2504.05586, Table 2](https://arxiv.org/html/2504.05586v2)
- Expert sets chosen with vs without between-round FT show "notable high agreement" at 50%; one-shot vs either shows "significantly high disagreement"; FT "mainly addresses load imbalance and performance drop, not expert selection". — [2504.05586, §3.3, Fig. 5](https://arxiv.org/html/2504.05586v2)
- Minitron: importance rankings of pruned candidates "continue to hold post-retraining" (400 steps, 1.8B tokens). — [Minitron, §4, Fig. 5](https://arxiv.org/pdf/2407.14679)
- ExFold fold: per-pair scalar by weighted least squares s* = Σ w_i⟨v_i,u_i⟩ / (Σ w_i‖v_i‖² + λ), w_i = ‖u_i‖₂, λ = 1e-3, scalars clipped to [−4,4]; calibration 32 unlabeled sequences ≤4096 tokens (Qwen3-30B-A3B) / 16 sequences ≤1024 tokens (GLM-4.5-Air); Qwen3-30B-A3B decode-pool results (orig avg 69.04): REAP-D64 61.50 → ExFold-D64 68.75; REAP-D32 22.25 → ExFold-D32 68.97; GLM-4.5-Air (orig 67.78): REAP-D64 55.45 → ExFold-D64 66.71; DeepSeek-V4-Flash (256 experts, orig 72.68): ExFold-D64 72.54. Retained-expert selection by S_i×H_i (router score × output magnitude). Qwen3.5 P4 excludes pairs with loss >0.99 and rescales retained weights "to preserve the routed-weight sum". No iterative/multi-stage fold; no 75% prefill test. — [ExFold, arXiv 2608.24938](https://arxiv.org/html/2608.24938v1)
- Caveat on ExFold numbers: D32/D64 are *decode-pool* budgets — excluded experts are still present for prefill and only folded at decode, so these are not weight-deletion results. — [ExFold](https://arxiv.org/html/2608.24938v1)
- OMP-MoE re-scores experts sequentially on a residual (Δe = 2⟨R_t,V_e⟩ − ‖V_e‖²) over cached statistics from a single forward pass ("expert importance changes after each selected expert because the residual signal changes, making static one-shot ranking insufficient"); Qwen3-30B-A3B 50%: 0.597 vs NAEE 0.515, DiEP 0.469 (orig 0.640); 64 WikiText-2 samples saturate; search 641 s on 8×H20. Internal numbers are inconsistent between Table 1 and §4.3 (0.597 vs 0.643). — [OMP-MoE, arXiv 2609.31631](https://arxiv.org/html/2609.31631)
- Gradient heals need a resident student (backward pass); sibling note: frozen-expert LoRA on cached teacher logits is feasible only for ≤~100 GiB students, otherwise needs a custom layer-streamed backward (~1M tokens/h). — [`healing_recovery.md`, local]
- Between-stage FT in the dense literature: SlimMoE 30–35% of steps in intermediate stages; Nemotron Puzzle 24B/43.2B/52.8B tokens; Minitron 100B per step — all orders of magnitude beyond a Thor budget. — [SlimMoE](https://arxiv.org/html/2506.18349v1); [Puzzle](https://arxiv.org/html/2607.04371); [Minitron](https://arxiv.org/pdf/2407.14679)

### Inferences
- Ordering of marginal value per stage for our pipeline: (1) re-rank on the pruned model — large, costs one streamed pass; (2) closed-form fold / renormalisation — medium (recovers most of REAP's one-shot loss at 50% in ExFold's setting), costs a per-layer least-squares fit on captured activations; (3) gradient heal — small incremental and infeasible between stages for a non-resident student. Reserve gradients for the *final* model only if it fits. [inference]
- Whether to fold *before* the next re-rank is a real design choice: folding changes router weights, which changes the survivors' gated outputs g‖f‖ and hence their REAP/HOPE saliency. Jaiswal/Minitron say heals don't move the mask much, so either order is probably fine; the safe default is to rank on the *unfolded* pruned model (matches the only measured regime) and fold once at the end, unless a stage-1 A/B shows the folded ranking agrees ≥95%. [extrapolation]
- OMP-MoE's residual re-scoring and HOPE's pairwise F are "within-pass iteration" over cached statistics: they recover selection co-dependence but under *fixed* routing. They cannot see post-prune routing redistribution, which requires a new forward pass. That is the part of Jaiswal's gain that only a real multi-pass schedule captures. [inference]

### Gaps
- No experiment isolates "heal between stages" from "re-rank between stages" at equal total heal tokens for MoE expert deletion.
- No measurement of whether an ExFold-style fold applied before re-ranking changes the next stage's prune set.

## Q6. Interaction with HOPE: does interaction-aware selection reduce or compound the benefit of iteration?

### Takeaway
No HOPE iterative result exists and the HOPE paper does not discuss iteration. HOPE's advantage over REAP grows with ratio (Qwen3.5-122B Tulu mean at 50%: 43.75 vs 40.18) and its F-matrix captures selection co-dependence under fixed routing; iteration captures routing redistribution. They address different error terms, so the expected effect is compounding with partially reduced marginal gain — unmeasured.

### Cited Findings
- HOPE per-ratio Tulu means (Evol-CA calibration, HOPE vs REAP): Qwen3.5-35B-A3B (base 48.87): 10% 48.49/48.45; 20% 48.81/48.17; 25% 48.51/48.73; 30% 48.05/48.45; 40% 46.94/43.16; 50% 42.93/43.00. Qwen3.5-122B-A10B (base 39.34): 40% 40.15/38.89; 50% 43.75/40.18. GLM-4.5-Air (base 46.83): 40% 40.48/42.07; 50% 43.26/44.69. "Finished last in only 1/54 conditions", worst deficit −4.5%. — [HOPE, arXiv 2609.18916, Tables S1–S3](https://arxiv.org/html/2609.18916v3)
- HOPE "prunes experts in each layer independently"; iteration not discussed; cross-layer allocation is future work and "may introduce non-uniform budget-allocation instability (e.g. over-pruning layers with weak cooperative structure)". — [HOPE, App. C](https://arxiv.org/html/2609.18916v3)
- Calibration convergence to 99% prune-set agreement at 50% on Qwen3.5-122B: "REAP requires 8k prompts whereas HOPE requires 10k prompts"; QP solve "1–2 seconds per layer"; HOPE calibration "about 6–7% longer than REAP". — [HOPE](https://arxiv.org/html/2609.18916v3)
- 1B-token SFT on LiveCodeBench traces for REAP- and HOPE-pruned Qwen3.5-122B: "fine-tuning narrows the gap but does not close it" (Fig. 6, no numbers in text). — [HOPE](https://arxiv.org/html/2609.18916v3)
- HOPE's off-diagonal is the only statistic needing per-token co-activation structure, so it must be accumulated in every pass or the pass is lost. — [local memory `hope-supersedes-reap`]

### Inferences
- Each stage pass should accumulate the full HOPE F-matrix (47 MiB/47 layers, ~8×8 outer product per token) in addition to REAP accumulators, so each stage's mask can be solved by HOPE QP + greedy swap on the current routing. Cost is ~7% of pass time. [inference from HOPE cost figures + local notes]
- HOPE's own tables show non-monotone behaviour across ratios (e.g. Qwen3.5-122B 50% mean 43.75 > base 39.34; GLM-4.5-Air HOPE < REAP at 40–50%), i.e. 50% is within noise of base on that eval for the 122B model — this is the regime where iteration's benefit will be hardest to see and a 60–75% target is needed to measure it. [inference]

### Gaps
- No HOPE (or any interaction-aware criterion) iterative result; no data on whether the F-matrix computed at stage k predicts stage k+1 selections.

## Q7. Layer-adaptive ratios in a progressive scheme: per-layer caps?

### Takeaway
Yes, enforce uniform per-layer removal at every stage. Jaiswal found a global (cross-layer) criterion "creates bottleneck layers" and therefore removed exactly one expert per layer per round; HOPE warns cross-layer allocation can over-prune layers with weak cooperative structure; our own DP showed non-uniform allocation buys only −4.98%/−3.83%/−3.24% error at 50/65/75% (gain shrinks with ratio) and is unshippable because `num_local_experts` is a single scalar.

### Cited Findings
- "Estimating the criterion globally creates bottleneck layers"; drop one expert per layer per round. — [2504.05586, §3.1](https://arxiv.org/html/2504.05586v2)
- Cross-layer allocation "may introduce non-uniform budget-allocation instability (e.g. over-pruning layers with weak cooperative structure)". — [HOPE, App. C](https://arxiv.org/html/2609.18916v3)
- Nemotron Puzzle stage 3 allocates the 50% active budget "heterogeneously across layers" — but Puzzle is a NAS over a block library with KD healing, not a saliency criterion. — [Puzzle §2.3](https://arxiv.org/html/2607.04371)
- DiEP argues existing MoE pruning uses uniform sparsity and proposes layer-level rates; OMP-MoE beats DiEP at 50% on all four models (Qwen 0.597 vs 0.469). — [DiEP, NeurIPS 2025](https://neurips.cc/virtual/2025/poster/116036); [OMP-MoE Table 1](https://arxiv.org/html/2609.31631)
- Local: exact min-plus DP with 16-expert floor −4.98% gross error at 50%, −3.83% at 65%, −3.24% at 75% (85..238 experts/layer); greedy water-fill made it worse; non-uniform checkpoints unloadable by vLLM/GGUF/our CUDA server (`N_ROUTED_EXPERT` constexpr). — [local memory `reap-criterion-and-corpus-audit`]

### Inferences
- A progressive schedule with uniform per-layer quotas at every stage is both what the only no-FT iterative study did and what the serving stack requires; the per-layer cap *is* the schedule. Monitor per-layer retained saliency mass per stage anyway — if one layer's survivors carry <~50% of its stage-0 mass while others carry >80%, that is the "bottleneck layer" signal and the global ratio should be lowered rather than the layer quota relaxed. [inference]

### Gaps
- No study measures whether later rounds of a progressive scheme *naturally* concentrate removals in redundant layers when a global criterion is allowed (Jaiswal forbade it from the start).

## Q8. Failure modes of iterative pruning and mitigations

### Takeaway
Reported failures are (a) load concentration on few survivors after router-entry removal, (b) "de-dispersion" of over-load-balanced routers making ppl a misleading gate (ρ(ppl, GPQA) = −0.10; uniform router-mass criterion scores 0.00 everywhere at 50% on gpt-oss-120B), and (c) criterion transfer failures across architectures (MESA +244% ppl on Mixtral). Mitigations with evidence: renormalise top-k router logits in the saliency (REAP upstream fix, 2.6% → 1.9% mean drop), rescale retained weights to preserve routed-weight sum after fold (ExFold), gate each stage on task evals not ppl, and light FT restores load balance (Jaiswal Fig. 6). Router temperature / load-balance re-fit after pruning: no evidence found.

### Cited Findings
- "Ad-hoc removal of router entries induces heavier load on few among remaining n−r experts"; load balance improves after C4 fine-tuning (Fig. 6, layers 6 and 30, 50%); FT addresses load imbalance rather than selection. — [2504.05586, §3.3, Fig. 6](https://arxiv.org/html/2504.05586v2)
- Over-dispersed MoEs: pruning "forces the router to redistribute mass to the remaining experts, de-dispersing the routing distribution"; gpt-oss-20b r=0.25: random pruning gives the *lowest* WikiText ppl (40.95 vs 126.02 baseline) but GSM8K 78.54 vs 88.86, REAP the highest ppl (145.04) yet GSM8K 89.31; Spearman ρ(ppl, GSM8K) 0.70, ρ(ppl, GPQA) −0.10; per-token router entropy 90.8% of max on gpt-oss vs 74.3% on Mixtral; r=0.5 gpt-oss-120B RM-Uniform 0.00 on every task; MESA on Mixtral-8×7B-Instruct ppl +244% (16.43), GSM8K 3.49 vs 31.42; MESA's internal minimax loop converges by T=15 rounds (worst-domain loss 0.0524 at T=1 → 0.0491 at T≥15; agreement with T=80 reference 76.1% → 95.9%). No router renormalisation, temperature or re-fit discussed. — [arXiv 2609.04453](https://arxiv.org/html/2609.04453)
- REAP analyses pruning as removal + renormalisation over remaining experts and argues substitution error dominates renormalisation error; upstream REAP added top-k logit renormalisation on 2026-03-11, mean accuracy drop 1.9% vs 2.6% across five models. — [REAP, arXiv 2510.13999](https://arxiv.org/html/2510.13999v3); [local memory `reap-logit-renorm-fix`]
- ExFold: exclude pairs with fold loss >0.99 and "rescale retained weights to preserve the routed-weight sum". — [ExFold](https://arxiv.org/html/2608.24938v1)
- MoE-Pruner uses coefficient of variation of expert activation frequency as a load-balance score; models trained from scratch "often show imbalanced expert loads", upcycled models are more balanced. — [MoE-Pruner §3.1](https://arxiv.org/html/2410.12013)
- Yuan3.0 Ultra (1T-class): pruning removes under-used experts but leaves "significant workload imbalance across the remaining experts", handled by expert rearrangement across devices (serving-side). — [Yuan3.0 Ultra, arXiv 2601.14327](https://arxiv.org/pdf/2601.14327) (search summary, not fetched)
- Local: noaux_tc routers — bias selects, unbiased sigmoid weights; the wrong version "agrees on 1 expert of 8". — [local memory `noaux-tc-router-formula`]

### Inferences
- The progressive scheme has a failure mode that one-shot does not: a mis-ranked stage is irreversible and its routing shift *becomes the calibration signal* for the next stage, so errors compound. Mitigation is a per-stage gate on task evals + per-domain retained mass (not ppl), with a rollback to the previous stage's checkpoint and a smaller step. [inference]
- For a noaux_tc / sigmoid-gated 1T model the "renormalisation" is over bias-selected top-k; the router bias terms of pruned experts must be deleted with the experts and the per-stage saliency must use the unbiased weights (local memory). [inference]
- Load-balance re-fit between stages is only possible as a closed-form router-bias adjustment (noaux_tc bias is a per-expert scalar that can be re-fit to equalise load on the calibration set without gradients); no paper does this, so treat as an experiment, not a plan. [extrapolation]

### Gaps
- No published mitigation for routing collapse *specific to iterative* schedules; no router-temperature or router re-fit results after expert pruning.

## Q9. Derived schedules for 60 / 70 / 75% and Thor wall-clock arithmetic

### Takeaway
Use a front-loaded 2-stage (60%) or 3-stage (70%, 75%) schedule whose first stage is the known-safe one-shot ratio (45–50%), with a full HOPE+REAP re-rank pass on the pruned model between stages, uniform per-layer quotas, no gradient heal between stages, an ExFold-style fold at the end (optionally per stage), and a task-eval gate per stage. On Thor at 5.5 h per full pass this is 11–17 h of calibration for 60–70% and 17–22 h for 75%; a 5-stage schedule costs 22–28 h and fully-iterative (1 expert/layer/round) is 150–290 passes and infeasible.

### Cited Findings
- Pass cost: ~5.5 h per full calibration pass (45 layers × 10 chunks) on Thor. — [assignment brief, local]
- Saliency stabilisation: 8k (REAP) / 10k (HOPE) prompts for 99% prune-set agreement at 50% on Qwen3.5-122B. — [HOPE](https://arxiv.org/html/2609.18916v3); Jaiswal used ~0.5M tokens per re-rank (256 × 2048). — [2504.05586 App. C](https://arxiv.org/html/2504.05586v2); OMP-MoE saturates at 64 WikiText samples. — [OMP-MoE](https://arxiv.org/html/2609.31631)
- Per-stage mask solve: HOPE QP 1–2 s/layer. — [HOPE](https://arxiv.org/html/2609.18916v3)
- Fold fit: per-layer weighted least squares on 32 × 4096-token captures. — [ExFold](https://arxiv.org/html/2608.24938v1); sibling note estimates "minutes/layer". — [`healing_recovery.md`, local]
- Schedule-shape evidence: front-loaded > uniform (Q3); 2–3 stages suffice (Q2); first-stage safe ratio 40–50% one-shot on 128–384-expert models (sibling notes; HOPE/REAP tables above).
- Student residency: 763B–1T total; frontier open weights are 3–12× over the 117 GiB envelope; NVFP4 is really ~0.62 B/weight. — [local memories `thor-model-fit-frontier`, `thor-base-model-shortlist`]

### Inferences — the schedules
Notation: r_k = fraction of the *remaining* experts removed at stage k; survivors f_k; passes = number of streamed calibration passes (pass 0 on the full model, one more after each non-final stage). All per-layer uniform. [all extrapolation from Q1–Q8]

| Target | Stages | r_1, r_2, r_3(, r_4) | survivors after each | passes | why |
|---|---|---|---|---|---|
| 60% | 2 (front-loaded) | 45%, 27% | 0.55 → 0.40 | 2 | stage 1 = proven one-shot level; one re-rank for the last 15 pts |
| 60% | 3 (geometric) | 26.3% ×3 | 0.737 → 0.543 → 0.40 | 3 | SlimMoE-style equal multiplicative steps |
| 70% | 3 (front-loaded, **recommended**) | 45%, 30%, 22% | 0.55 → 0.385 → 0.30 | 3 | each later stage ≤30% of remaining, the largest step Jaiswal's regime tolerated without FT at good quality |
| 70% | 3 (geometric) | 33% ×3 | 0.67 → 0.45 → 0.30 | 3 | — |
| 75% | 3 (front-loaded) | 50%, 35%, 23% | 0.50 → 0.325 → 0.25 | 3 | aggressive; stage-2 35% is the riskiest step |
| 75% | 4 (front-loaded, **recommended**) | 45%, 30%, 25%, 13.5% | 0.55 → 0.385 → 0.289 → 0.25 | 4 | keeps every post-stage-1 step ≤30% |
| 75% | 5 (geometric) | 24.2% ×5 | 0.758 → 0.575 → 0.436 → 0.330 → 0.25 | 5 | diminishing-returns control arm |
| any | fully iterative | 1 expert/layer/round | — | 0.6E to 0.75E passes (154–288 for E=256–384) | infeasible (Jaiswal's regime) |

Cumulative removal check: 1 − Π(1 − r_k): 2-stage 60%: 1 − 0.55×0.73 = 0.598; 3-stage 70%: 1 − 0.55×0.70×0.78 = 0.700; 4-stage 75%: 1 − 0.55×0.70×0.75×0.865 = 0.750.

### Inferences — Thor wall-clock
Pass-time model: T(f) = 5.5 h × (a + (1 − a)·f), f = surviving expert fraction, a = share of the pass that does not scale with expert count (attention, I/O overhead, chunk bookkeeping). a is unmeasured; two bounds are shown — worst case a = 1 (every pass 5.5 h) and an optimistic a = 0.3. [extrapolation; measure a on the stage-1 pass]

Per-stage overheads (added once per non-final stage): mask solve <5 min (HOPE QP 1–2 s/layer × 45); checkpoint surgery (rewrite surviving experts, I/O-bound) ~0.5 h [extrapolation]; fold fit if done per stage ~1–2 h for 45 layers [extrapolation from "minutes/layer"]; task-eval gate — use the existing eval harness, cost not estimated here. Call the overhead ~1.5 h/stage with per-stage fold, ~0.5 h/stage without.

| Schedule | passes at f = | calibration, worst (a=1) | calibration, optimistic (a=0.3) | + overheads | total |
|---|---|---|---|---|---|
| 2-stage 60% | 1.0, 0.55 | 11.0 h | 5.5 + 3.8 = 9.3 h | 1 × 0.5–1.5 h | **10–12.5 h** |
| 3-stage 60% geometric | 1.0, 0.737, 0.543 | 16.5 h | 5.5 + 4.5 + 3.7 = 13.7 h | 2 × 0.5–1.5 h | 15–19.5 h |
| 3-stage 70% front-loaded | 1.0, 0.55, 0.385 | 16.5 h | 5.5 + 3.8 + 3.1 = 12.4 h | 2 × 0.5–1.5 h | **13.5–19.5 h** |
| 3-stage 75% front-loaded | 1.0, 0.50, 0.325 | 16.5 h | 5.5 + 3.6 + 2.9 = 12.0 h | 2 × 0.5–1.5 h | 13–19.5 h |
| 4-stage 75% front-loaded | 1.0, 0.55, 0.385, 0.289 | 22.0 h | 5.5 + 3.8 + 3.1 + 2.8 = 15.2 h | 3 × 0.5–1.5 h | **16.5–26.5 h** |
| 5-stage 75% geometric | 1.0, 0.758, 0.575, 0.436, 0.330 | 27.5 h | 5.5 + 4.6 + 3.9 + 3.3 + 2.9 = 20.2 h | 4 × 0.5–1.5 h | 22–33.5 h |
| fully iterative 75%, E=256 | 192 passes | ~1,056 h | ~630 h | — | infeasible |

Plus one final fold/verification pass on the finished model if fold statistics were not captured in the last ranking pass (+2.5–5.5 h). Everything above is one Thor; per the local rule, run each pass detached (setsid/systemd), never as a child of an interactive session.

Calibration-token budget per stage: keep the full 10-chunk budget at stage 0 (the irreversible, largest cut) and at the final stage; for intermediate stages the HOPE 99%-agreement figure (10k prompts) is the right target, and the cheapest empirical check is chunk-ablation on banked per-chunk saliency from stage 0 — if the 5-chunk mask agrees ≥97% with the 10-chunk mask at the stage-2 ratio, intermediate passes can use 5 chunks, halving their cost (optimistic 3-stage 70% drops to ~9 h). [extrapolation]

What is *not* in the schedule: between-stage gradient heals. At 60–75% removal of a 763B–1T model the student is still ~190–400B parameters (~120–250 GB at 0.62 B/w), i.e. not resident in 122 GiB, so LoRA/KD between stages would need the custom layer-streamed backward (~1M tokens/h, sibling note). Jaiswal's ~1M-token per-round FT bought only 0.68 of 4.98 ppl points at 50% (14% of the gap) and did not change selections — spend that budget on the extra re-rank pass instead, and heal once at the end only if the final model fits. [inference]

Recommended control arms for the first run (each is one extra pass): (i) one-shot 60% from the stage-0 saliency (free — same pass) to measure the iterative gain at the first ratio we care about; (ii) fold-before-rerank vs rank-unfolded at stage 1 (Q5 gap) — +1 pass; (iii) the geometric 3-stage 60% vs front-loaded 2-stage 60% — +2 passes.

### Gaps
- a (non-expert share of pass time) is unmeasured; the optimistic column could be off by 2× either way.
- Checkpoint-surgery and fold-fit times for a ~600 GB NVFP4 checkpoint on Thor NVMe are extrapolated, not measured.
- No published progressive expert-deletion result ≥100B parameters or ≥64 experts/layer exists; the schedules above are derived from an 8-expert model (Jaiswal), width-slimming (SlimMoE/SlimQwen/Minitron) and a 120B NAS (Puzzle), and the 60–75% targets are outside every measured regime for general-purpose quality.
