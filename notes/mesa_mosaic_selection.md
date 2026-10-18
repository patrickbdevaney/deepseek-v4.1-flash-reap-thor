# MESA and Mosaic Pruning as drop-in selection improvements for a HOPE/REAP pipeline

Scope: two formal "keep the thin domains alive" expert-selection methods (MESA = minimax per-domain routing-mass reweighting; Mosaic Pruning = cluster-then-select), their exact inputs, results, scaling evidence, sibling methods, and how they compose with a HOPE pᵀFp objective. Researched 2026-10-17. Paper identities confirmed: MESA = arXiv 2609.04453 (Kapusuzoglu et al., Capital One; NeurIPS 2026 workshop poster), Mosaic Pruning = arXiv 2511.19822 (Hu et al., Xi'an Jiaotong / TeleAI; AAAI 2026).

Local full-text extractions used (pdftotext of the arXiv PDFs) live in `/tmp/claude-2002/-home-patrickd/3952824c-51a0-4b40-aa8e-acf3b3b02ab9/scratchpad/pdfs/` (`mesa.txt`, `mop.txt`, `cprune.txt`, plus pre-existing `maestro.txt`, `hope.txt`, `conmoe.txt`). Numbers below are quoted from those extractions unless another URL is cited.

---

## KQ1. MESA: identity, minimax formulation, solver, results, AIME mechanism

### Takeaway
MESA (arXiv 2609.04453) minimises, over a globally-ranked pruning set P of size ⌊r·L·E⌋, the maximum over calibration domains d of the fraction of domain d's total router mass that lies in P; it is solved by a 20-step CPU-only greedy reweighting of an L×E router-mass matrix. At r=0.25 it turns REAP's −16.75 pt GPQA into +1.01 on gpt-oss-20b at the cost of AIME'25 −13.34 (vs REAP −3.34); the paper offers no mechanistic account of the AIME loss, and MESA is catastrophic on Mixtral (GSM8K 31.42→3.49), so it is a regime-specific tool, not a universal upgrade.

### Cited Findings

**Identity and venue**
- Title "When Load-Balancing Goes Too Far: Expert Pruning in Over-Dispersed Mixture-of-Experts Models", arXiv 2609.04453, authors Berkcan Kapusuzoglu, Connor Pryor, Sangwoo Cho, Supriyo Chakraborty, Shi-Xiong Zhang, Sambit Sahu, Milind Naphade; paper licence CC BY 4.0 — [arXiv abs](https://arxiv.org/abs/2609.04453)
- NeurIPS 2026 poster in the workshop "On-Device Intelligence: Foundation Models under Real-World Constraints"; OpenReview id WG8rzkNs5w; no code link on the venue page — [NeurIPS listing](https://neurips.cc/virtual/2026/170047)

**Problem framing: over-dispersed routing**
- gpt-oss-20b was trained with λ_aux = 0.9 (vs ~0.001 Mixtral, 0.01 Switch), pushing router probabilities toward uniform; mean per-token routing entropy is 4.54 bits = 90.8% of log₂32 (183K tokens) vs Mixtral 2.23 bits = 74.3% of log₂8; OLMoE 92.5%; Gemma-4-26B-A4B ≈ 69%. Diagnostic: H̄ > 0.85·log₂E, described as a "soft guideline" — [arXiv HTML §3.1](https://arxiv.org/html/2609.04453)
- "Consequence 2": each of 32 experts receives ~3.1% of global traffic, but e.g. Math routes 3.8% to Expert A while Code routes 2.4%; a global importance score averages these micro-signals away. Per-domain routing-mass heatmaps (Fig. 5) span only 0.028–0.038 around the uniform 1/32 ≈ 0.031 — [arXiv HTML §3.1, App. E](https://arxiv.org/html/2609.04453)
- Cross-domain overlap of each domain's top-8 experts per layer is 0.41–0.47 Jaccard (Math–Science 0.47, Code–Instruction 0.41), which the paper says is why "naive per-domain scoring (keeping each domain's top experts separately) is insufficient" — [arXiv HTML App. E.2](https://arxiv.org/html/2609.04453)
- PPL confound: gpt-oss uses top-k-then-softmax; after pruning 32→24 the smaller pool biases top-4 toward stronger experts and PPL *improves* (baseline 126.02, MESA 107.14, REAP 145.04, RM-Uniform 83.29, Random 40.95) while task accuracy falls, so PPL is not a quality signal in this regime — [arXiv HTML Table 1, App. A](https://arxiv.org/html/2609.04453)

**Objective (Eq. 2), verbatim structure**
- Let R ∈ ℝ^{L×E} be "per-layer normalized router mass scores" and D_d ∈ ℝ^{L×E} "per-domain router mass for domain d". Objective: min over P ⊆ [L]×[E], |P| = ⌊r·L·E⌋, of max over d of ( Σ_{(l,e)∈P} D_d[l,e] / Σ_{l,e} D_d[l,e] ) — i.e. minimise the worst domain's *fractional loss of routing mass*. Budget is global across layers (non-uniform per-layer counts fall out of global ranking) — [arXiv HTML §3.2](https://arxiv.org/html/2609.04453)
- Fig. 5 caption defines D_d[l,e] as "the fraction of routing probability mass that domain d assigns to expert e at layer l" — [arXiv HTML App. E.1](https://arxiv.org/html/2609.04453)

**Solver (Algorithm 1), verbatim structure**
- Inputs: R, {D_d}, r, boost η₀ = 0.15, steps T = 20. S ← R; S* ← R; ℓ* ← ∞. For t = 1..T: P ← ALLOCATE(S, r) {global ranking, lowest S removed}; ℓ_d ← Σ_{P} D_d / Σ_{all} D_d for every d; d* ← argmax_d ℓ_d; if ℓ_{d*} < ℓ* then (ℓ*, S*) ← (ℓ_{d*}, S); η_t ← η₀·(1 − 0.5·t/(T−1)); S ← RowNorm(S + η_t·D_{d*}). Return S*; final P is global ranking on S*. Described as "a greedy heuristic solver … inspired by distributionally robust optimization, without formal convergence guarantees" — [arXiv HTML Alg. 1](https://arxiv.org/html/2609.04453)
- Hyperparameter sensitivity at r = 0.25 on gpt-oss-20b: worst-domain loss 0.0524 at T=1 (i.e. plain router-mass ranking), 0.0509 at T=3, 0.0492 at T=10, 0.0491 at T≥15 (Jaccard vs T=80 reference 95.9% at T=15); any η₀ ∈ [0.05, 0.60] reaches 0.0491; linear/cosine/constant decay all converge in 8–17 steps — [arXiv HTML Table 12, App. D.2](https://arxiv.org/html/2609.04453)
- Compute profile (Table 10): "MESA's allocation step (Algorithm 1) operates entirely on a small, pre-computed L × E router-mass matrix (e.g., 24 × 32 for gpt-oss-20B) rather than requiring additional model forward passes"; scoring compute = "Static per-domain L × E router-mass matrix, already produced by the calibration harvest"; extra forward passes beyond harvest = 0 — [arXiv HTML §5.1 Table 10](https://arxiv.org/html/2609.04453)

**Calibration**
- 1000 examples per domain × 6 domains (math, coding, commonsense, science, instruction following, general knowledge — Fig. 5 labels them Math, Code, Science, News, Instruction, General) × 8192 tokens = 49.2M tokens, vs REAP's 24,576 × 16,384 = 402.7M; "topic-level domain text, not benchmark-formatted examples" — [arXiv HTML §4.1, §5.2](https://arxiv.org/html/2609.04453)
- Prune-map agreement with the s=2000/l=8192 reference: s=20,l=2048 64.1%; s=1000,l=4096 82.9%; s=1000,l=8192 97.9%; sequence length matters more than sample count (+15.0% from doubling length vs +5.8% from 10× samples). "Higher calibration shifts the trade-off toward domain balance (commonsense gains, PIQA +2%) at the cost of knowledge-intensive tasks (MMLU-Pro −4%). Lower calibration acts as implicit regularization" — [arXiv HTML App. D.1](https://arxiv.org/html/2609.04453)

**Results — gpt-oss-20b, r = 0.25 (32→24/layer), 5-seed means, temperature 0.6 (Table 2 subset)**

| Method | AIME'25 | GSM8K | GPQA-D | MMLU | MMLU-Pro |
|---|---|---|---|---|---|
| Unpruned | 76.67±3.27 | 88.86±0.38 | 64.47±1.46 | 84.01±0.10 | 72.51±0.22 |
| MESA | 63.33±6.06 | 92.87±0.45 | 65.48±4.2 | 79.51±0.6 | 67.25±0.9 |
| REAP | 73.33±9.7 | 89.31±1.8 | 47.72±5.6 | 77.36±0.8 | 64.27±1.2 |
| RM-Uniform | 46.67±8.5 | 83.24±1.5 | 56.85±5.1 | 80.31±0.5 | 66.32±0.8 |
| EvoESAP | 56.67±8.0 | 78.32±2.0 | 44.16±4.5 | 80.15±0.5 | 66.69±0.9 |
| Random | 33.33±7.2 | 78.54±2.3 | 49.75±4.8 | 74.76±0.9 | 56.06±1.3 |

— [arXiv HTML Table 2](https://arxiv.org/html/2609.04453)

**Results — gpt-oss-20b, r = 0.25, single-seed Δ vs unpruned across all 11 benchmarks (Table 3)**

| Benchmark | MESA | REAP | EvoESAP | RM-Uniform | Random |
|---|---|---|---|---|---|
| AIME'25 | −13.34 | −3.34 | −20.00 | −30.00 | −43.34 |
| GSM8K | +4.01 | +0.45 | −10.54 | −5.62 | −10.32 |
| ARC-C | −2.64 | −3.67 | −0.34 | −0.94 | −5.29 |
| ARC-E | −0.72 | −3.29 | −0.68 | −0.76 | −1.86 |
| GPQA-D | +1.01 | −16.75 | −20.31 | −7.62 | −14.72 |
| OpenBookQA | −2.61 | −0.81 | −3.81 | −0.81 | −6.02 |
| MMLU | −4.50 | −6.65 | −3.86 | −3.70 | −9.25 |
| MMLU-Pro | −5.26 | −8.24 | −5.82 | −6.19 | −16.45 |
| IFEval | −1.75 | −0.79 | −17.60 | −5.60 | −1.99 |
| PIQA | −2.61 | −8.44 | −2.94 | −8.49 | −4.73 |
| WinoGrande | −7.19 | +1.66 | −10.90 | −7.51 | −3.87 |

MESA beats REAP on 7/11 (GPQA +17.8, PIQA +5.8, GSM8K +3.6, MMLU-Pro +3.0, ARC-E +2.6, MMLU +2.2, ARC-C +1.0); REAP leads on 4 (AIME +10.0, WinoGrande +8.9, OBQA +1.8, IFEval +1.0). GPQA gap z = 3.57, p < 0.001 on 198 questions — [arXiv HTML Table 3, §4.3](https://arxiv.org/html/2609.04453)
- Reasoning/knowledge decomposition (Table 9): ΔR (mean AIME+GSM8K) / ΔK (mean ARC-C+ARC-E): MESA −4.67 / −1.68 "Balanced"; REAP −1.45 / −3.48 "Reasoning-preserving"; EvoESAP −15.27 / −0.51; RM-Uniform −17.81 / −0.85; Random −26.83 / −3.58 — [arXiv HTML Table 9](https://arxiv.org/html/2609.04453)

**Results — Mixtral-8x7B-Instruct, r = 0.25 (8→6): MESA FAILS**
- WikiText-2 PPL: Unpruned 4.78, REAP 6.00, Random 8.31, RM-Uniform 14.35, MESA 16.43. GSM8K: Unpruned 31.42, REAP 35.00, EvoESAP 40.56, Random 20.71, MESA 3.49, RM-Uniform 2.96. "REAP retains 98% of baseline average accuracy and MESA retains 63%." Paper's framing: under standard routing "all methods degrade proportionally and no domain imbalance emerges" — [arXiv HTML Table 4, §6](https://arxiv.org/html/2609.04453)

**Results — Gemma-4-26B-A4B-it, r = 0.25 (128→96/layer, top-8), temperature 0.6, single measurement (Table 8)**

| Method | AIME'25 | GSM8K | GPQA-D | ARC-C | ARC-E | OBQA | PIQA | MMLU | MMLU-Pro | WinoGr | IFEval |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Unpruned | 75.56 | 96.41 | 71.21 | 96.62 | 98.78 | 94.47 | 90.82 | 86.58 | 82.05 | 80.27 | 89.81 |
| MESA | 72.22 | 96.21 | 67.85 | 95.53 | 98.30 | 92.13 | 88.03 | 83.73 | 79.40 | 77.56 | 88.98 |
| REAP | 35.56 | 95.60 | 53.54 | 95.82 | 98.22 | 90.67 | 86.85 | 78.94 | 67.90 | 76.95 | 88.70 |
| RM-Uniform | 50.00 | 95.12 | 46.63 | 91.52 | 96.55 | 88.13 | 81.27 | 78.26 | 69.56 | 73.24 | 86.67 |

— [arXiv HTML Table 8](https://arxiv.org/html/2609.04453)
- Temperature-0, 3-seed replication (Table 13): GPQA-D Unpruned 74.24±1.21, MESA 65.66±1.82, REAP 55.56±1.95, RM 48.99±2.15; AIME'25 MESA 90.00±3.33 vs REAP 76.67±4.71 vs unpruned 86.67±3.33. A "domain-aware REAP control (REAP given the same per-domain conditioning as MESA) reaches GPQA-Diamond 58.4", which the authors use to argue the gain "comes from the minimax worst-case objective itself" — [arXiv HTML Table 13, App. H](https://arxiv.org/html/2609.04453)
- Paper states Gemma-4 "is not over-dispersed by our diagnostic (H̄ ≈ 0.69)" and presents it "solely as evidence that MESA's worst-domain protection generalizes to an architecturally distinct, concentrated-routing model" — [arXiv HTML §4.7](https://arxiv.org/html/2609.04453)

**Results — gpt-oss-120B, r = 0.5 (128→64/layer), single measurement (Table 14)**

| Method | AIME'25 | GSM8K | GPQA-D | ARC-C | ARC-E | OBQA | PIQA | MMLU | MMLU-Pro | WinoGr | IFEval |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Unpruned | 74.00 | 95.63 | 70.36 | 96.38 | 98.67 | 93.91 | 84.93 | 87.76 | 77.78 | 81.03 | 86.50 |
| MESA | 72.22 | 95.58 | 58.46 | 92.38 | 95.88 | 93.56 | 77.95 | 83.01 | 64.20 | 77.27 | 87.15 |
| REAP | 48.89 | 96.26 | 51.85 | 95.14 | 98.18 | 93.33 | 77.93 | 82.05 | 68.96 | 81.16 | 84.50 |
| RM-Uniform | 0.00 ×11 | | | | | | | | | | |

"RM-Uniform collapses completely at r = 0.5 on gpt-oss-120B (all-task 0.00); this is a real degeneracy of the method under aggressive pruning at this scale, not an unmeasured cell." MESA wins the worst domain (GPQA +6.6) plus AIME, IFEval, MMLU, OBQA, PIQA; REAP wins GSM8K, ARC-C, ARC-E, MMLU-Pro, WinoGrande — [arXiv HTML App. I, Table 14](https://arxiv.org/html/2609.04453)

**Results — OLMoE-1B-7B-0924 (base, 64 experts), cloze accuracy, single seed**
- r = 0.25: commonsense avg Unpruned 68.6; MESA (prefill-calibrated) 63.6 (92.8%); MESA (gen-calibrated) 57.3; REAP (gen) 57.9; RM (gen) 57.3. Ratio sweep for prefill-MESA: r=0.25 63.6 / r=0.375 57.2 (83.4%) / r=0.50 47.9 (69.9%), "degrades gracefully with no cliff" — [arXiv HTML Tables 6–7](https://arxiv.org/html/2609.04453)
- Critical mechanism: under a *decode-time* (gen-route) calibration harvest "the four calibration domains routed almost identically (per-domain routing-mass correlation 0.57–0.86), so the minimax objective had no per-domain signal to grip and degenerated toward router mass" (MESA vs RM-Uniform Jaccard ≈ 1.0). Under *prefill* calibration, cross-domain routing-mass correlation "collapsed to ≈ 0 (math–knowledge −0.068, science–knowledge 0.004)", and MESA/RM Jaccard fell to 0.705 — [arXiv HTML §4.6](https://arxiv.org/html/2609.04453)

**Internal inconsistency on r = 0.5**
- Conclusion: MESA "does not survive aggressive pruning (r=0.50), where REAP's activation-aware signal remains robust." Appendix I: "MESA's worst-domain protection holds even at r = 0.5, though it does not dominate every benchmark." Not reconciled in the text — [arXiv HTML §6 vs App. I](https://arxiv.org/html/2609.04453)

**Limitations stated by authors**
- Greedy heuristic without convergence guarantees; main results at a single ratio r = 0.25; OLMoE and Gemma-4 single-seed; REAP/RM baselines on OLMoE calibrated under decode while headline MESA used prefill — [arXiv HTML §4.6, §6](https://arxiv.org/html/2609.04453)

### Inferences
- The minimax objective is a *budget re-allocator*, not a new saliency: it starts from plain router mass (T=1 is exactly RM global ranking) and pushes budget toward whichever domain's mass is being cut most. It never looks at activation norms, so a domain whose experts have concentrated gate *and* large output norms (math on gpt-oss) loses the protection REAP's norm term gave it. That, plus the explicit statement that heavier calibration "shifts the trade-off toward domain balance … at the cost of knowledge-intensive tasks", is the most plausible mechanism for the −13 pt AIME: the objective only bounds the *worst* domain's fractional loss; every other domain is free to absorb cuts up to that bound, and math was the domain REAP had been over-protecting. AIME is also the highest-variance benchmark in the suite (MESA ±6.06, REAP ±9.7), so part of the −13.34 single-seed delta is noise (5-seed means give 63.33 vs 76.67, still −13.3).
- The Mixtral collapse (GSM8K 3.49) shows the down-side: in a concentrated-routing model, "fractional routing mass" protection can select a set that is globally wrong because mass ≠ contribution. The Gemma-4 success (H̄≈0.69) contradicts a clean "only use in over-dispersed regime" rule, so the regime diagnostic is not predictive on its own; it is likely that what matters is whether per-domain routing-mass vectors are decorrelated (≈0 on prefill harvests) rather than entropy per se.
- RM-Uniform's 0.00 across the board at r=0.5 on 120B is a router-mass-only method degenerating completely; MESA shares that scoring basis and only differs by reallocation, so MESA at ≥50% is one reweighting away from the same cliff. The paper's own conclusion flags r=0.50 as unsafe.
- The "domain-aware REAP control" (58.4 vs 65.66 GPQA) is the closest thing in the literature to "REAP + per-domain conditioning", i.e. to what a per-domain-protection rule inside a REAP/HOPE pipeline already does; MESA's extra +7 pts over it is the value of the minimax reallocation specifically.

### Gaps
- The exact definition of "router mass" (full-softmax probability summed over all experts, vs only the top-k gate weights of selected experts) is never written down; for gpt-oss (top-k-then-softmax) a full softmax does not exist in the forward pass, so it is most likely the realised top-k gate weights, but this is unconfirmed. "Per-layer normalized" is consistent with each layer's row summing to 1 (Fig. 5 values ≈ 1/32).
- No MESA code release found (NeurIPS page, arXiv page and a GitHub search for "Minimax Expert Score Allocation" return nothing).
- No code-generation benchmark (HumanEval/MBPP/LiveCodeBench) anywhere in the MESA evaluation, so its effect on a coding domain is unmeasured.
- No per-domain fractional-loss values are reported for REAP or RM (only MESA's 0.0491 at r=0.25), so the size of the protection cannot be quantified against baselines.
- No ratio sweep between 0.25 and 0.5 on a generative model (only OLMoE cloze at 0.375 and 120B at 0.5 single-seed).

---

## KQ2. Mosaic Pruning: identity, clustering, within-cluster selection, results, the HumanEval collapse

### Takeaway
Mosaic Pruning (MoP, arXiv 2511.19822, AAAI 2026) first keeps m "general" experts by reconstruction-loss enumeration, then clusters the remaining experts by the Spearman similarity of their *per-domain single-expert reconstruction-error profiles* (domains discovered by K-Means on token hidden states, K = r − m) with Ward linkage, and keeps from each cluster the expert with the highest Activation Variability Score (KL of its token-activation distribution from uniform). On Mixtral 8→4 it keeps HumanEval at 71.34 where enumeration pruning scores 1.22; but it is evaluated only up to 50% on an 8-expert model and 33% on a 60-expert model, an independent re-implementation (MAESTRO) finds it slightly *below* REAP on gpt-oss-20b and could not run it at 128 experts because it stores every expert's output for every calibration token.

### Cited Findings

**Identity and code**
- "Mosaic Pruning: A Hierarchical Framework for Generalizable Pruning of Mixture-of-Experts Models", Wentao Hu, Mingkuan Zhao, Shuangyong Song, Xiaoyan Zhu, Xin Lai, Jiayin Wang; arXiv 2511.19822; AAAI 2026 — [arXiv abs](https://arxiv.org/abs/2511.19822); [AAAI OJS](https://ojs.aaai.org/index.php/AAAI/article/view/39341)
- Code: github.com/Saul-James/MoP, MIT licence, 2 commits, 2 stars, README consists of the single line "# MoP", one folder `MOP/` whose GitHub API listing returned no files — [GitHub](https://github.com/Saul-James/MoP); [raw README](https://raw.githubusercontent.com/Saul-James/MoP/main/README.md)

**Algorithm**
- Stage 1 (shared with the authors' intermediate method GVP): choose E_general = argmin_{S⊂E,|S|=m} L_recon(D_cache, S) — m general experts minimising output reconstruction loss on a cache of input hidden states and original layer outputs (exact enumeration for Mixtral n=8; greedy single-expert removal for Qwen n=60). The value of m is not stated anywhere in the paper text — [arXiv HTML Methodology](https://arxiv.org/html/2511.19822v1)
- Activation Variability Score (Eq. 2): S_var(i) = Σ_t (p_{t,i}/Z_i) · log₂((p_{t,i}/Z_i) · N_total), where p_{t,i} is "the Softmax activation probability for expert i on token t" and Z_i = Σ_t p_{t,i}; i.e. KL(normalised activation distribution over tokens ‖ uniform over tokens). "A high S_var score indicates a concentrated activation pattern on a specific subset of inputs, suggesting it is a functional specialist" — [arXiv HTML Eq. 2](https://arxiv.org/html/2511.19822v1)
- GVP (ablation step): E_final = E_general ∪ argmax_{S⊂E_cand,|S|=r−m} Σ S_var(i). Flaw stated: "it may retain two functionally similar yet highly specialized experts" — [arXiv HTML GVP](https://arxiv.org/html/2511.19822v1)
- Stage 2 of MoP, domain discovery: K-Means on the input hidden states {x_t} of all calibration tokens, minimising WCSS, with K = r − m clusters, "one-to-one mapping between the discovered functional clusters and the experts to be selected". Domains are therefore unlabeled, data-driven, and the count is tied to the budget — [arXiv HTML Eq. 5](https://arxiv.org/html/2511.19822v1)
- Performance vector (Eq. 7): v_{i,k} = (1/|T_k|) Σ_{t∈T_k} ‖E_i(x_t) − z_real,t‖², where E_i(x_t) is "the output of the MoE layer only with expert i … effectively simulating a scenario where only this single expert is activated" and z_real,t the true (unpruned) MoE layer output — [arXiv HTML Eq. 7](https://arxiv.org/html/2511.19822v1)
- Similarity: Spearman ρ on rank-transformed performance vectors, S_perf = ½(1+ρ), S_comp = S_perf (Eq. 6); distance D = 1 − S_comp; agglomerative clustering with Ward's linkage (merge cost |C_a||C_b|/(|C_a|+|C_b|)·‖μ_a−μ_b‖² on centroids of performance vectors) until r − m clusters remain; keep argmax S_var in each cluster (Eq. 13) — [arXiv HTML Eqs. 6–13](https://arxiv.org/html/2511.19822v1)
- Calibration: enumeration baseline uses C4 (following Lu et al. 2024); GVP/MoP use a "mixed-diversity calibration dataset" whose construction is deferred to Appendix A, which is absent from both the arXiv and AAAI PDFs; token count not given. Expert-diversity validation uses six domain test sets: Math-GSM8K, Code-HumanEval, Commonsense-PIQA, Summarization-XSum, Science-ARC-C, Logic-MNLI — [arXiv HTML Experiments](https://arxiv.org/html/2511.19822v1)

**Results — general benchmarks (Table 1, zero-shot accuracy; "Experts" = retained per layer)**

| Model | Method | Experts | ARC-c | ARC-e | BoolQ | HellaSwag | MMLU | OBQA | WinoGr | Avg |
|---|---|---|---|---|---|---|---|---|---|---|
| Mixtral-8x7B-Instruct | None | 8 | 80.21 | 87.29 | 81.47 | 75.91 | 67.43 | 83.20 | 53.26 | 75.54 |
| | Random | 6 | 69.97 | 78.11 | 83.18 | 68.64 | 53.44 | 74.20 | 53.15 | 68.67 |
| | Random | 4 | 44.54 | 47.22 | 64.86 | 59.83 | 48.21 | 67.20 | 55.72 | 55.37 |
| | Frequency | 6 | 70.90 | 78.15 | 78.28 | 56.85 | 58.47 | 78.20 | 58.25 | 68.44 |
| | Frequency | 4 | 50.26 | 64.56 | 67.46 | 37.50 | 46.76 | 63.20 | 50.27 | 54.29 |
| | Enu | 6 | 73.04 | 83.16 | 84.89 | 67.49 | 61.14 | 81.00 | 58.41 | 72.73 |
| | Enu | 4 | 59.59 | 74.41 | 83.12 | 52.02 | 49.45 | 63.80 | 54.46 | 62.41 |
| | MoP | 6 | 75.43 | 82.37 | 86.51 | 74.28 | 63.50 | 80.20 | 55.80 | 74.01 |
| | MoP | 4 | 69.62 | 79.84 | 83.33 | 68.85 | 54.26 | 72.00 | 54.85 | 68.96 |
| Qwen1.5-MoE-A2.7B-Chat | None | 60 | 54.18 | 64.56 | 78.90 | 58.99 | 37.44 | 61.00 | 48.38 | 57.64 |
| | Enu | 50 | 54.61 | 66.16 | 33.03 | 62.89 | 53.69 | 65.20 | 50.19 | 55.11 |
| | Enu | 40 | 30.63 | 35.65 | 22.05 | 42.36 | 43.49 | 49.80 | 46.25 | 38.60 |
| | MoP | 50 | 60.70 | 75.04 | 38.10 | 67.28 | 55.60 | 66.80 | 50.19 | 59.10 |
| | MoP | 40 | 44.97 | 59.76 | 68.56 | 56.49 | 48.44 | 59.40 | 52.57 | 55.74 |

— [arXiv HTML Table 1](https://arxiv.org/html/2511.19822v1)
- Ablation (Table 2, avg accuracy): Mixtral 6 experts — Enu(C4) 72.73, Enu(Mixed) 72.79, GVP 73.50, MoP 74.01; Qwen 50 experts — Enu(C4) 55.11, Enu(Mixed) 49.47, GVP 55.56, MoP 59.10. I.e. switching the enumeration baseline to the mixed calibration set alone does *not* help (and hurts Qwen by 5.6 pts) — [arXiv HTML Table 2](https://arxiv.org/html/2511.19822v1)

**Results — specialised benchmarks (Tables 4–5; no unpruned row reported)**

| Model | Method | Experts | GSM8K | MATH | HumanEval | MBPP |
|---|---|---|---|---|---|---|
| Mixtral | Enu | 6 | 28.73 | 9.00 | 82.31 | 14.00 |
| Mixtral | Enu | 4 | 22.37 | 3.80 | 1.22 | 1.55 |
| Mixtral | MoP | 6 | 51.33 | 9.20 | 83.54 | 15.95 |
| Mixtral | MoP | 4 | 28.35 | 3.00 | 71.34 | 10.89 |
| Qwen | Enu | 50 | 34.34 | 7.39 | 1.21 | 8.17 |
| Qwen | Enu | 40 | 16.76 | 5.40 | 1.83 | 1.95 |
| Qwen | MoP | 50 | 35.63 | 11.20 | 14.02 | 9.34 |
| Qwen | MoP | 40 | 21.91 | 7.20 | 3.05 | 8.17 |

Claimed averages: +7.24% on general tasks, +8.92% on specialised tasks over enumeration — [arXiv HTML Tables 4–5](https://arxiv.org/html/2511.19822v1)
- Memory/speed: Mixtral 87.87 GB → 66.87 GB (6 exp, 1.22×) → 45.87 GB (4 exp, 1.52×); Qwen 27.79 → 23.91 (1.12×) → 20.04 GB (1.36×) — [arXiv HTML Table 3](https://arxiv.org/html/2511.19822v1)

**Explanation of the HumanEval collapse**
- The paper's account is "functional collapse"/"functional homogenization": enumeration pruning "inherently favor[s] retaining generalist experts that make moderate contributions across the board, while discarding specialist experts". Figure 3 heatmaps (Mixtral→4 experts, activations across GSM8K/HumanEval/PIQA/XSum/ARC-C/MNLI, 32 layers × 4 experts) show "a few generalist experts dominating all tasks" under Enu vs "different experts activating for different tasks" under MoP. No HumanEval-specific mechanism is given — [arXiv HTML Fig. 3, Intro](https://arxiv.org/html/2511.19822v1)

**Independent evaluation of MoP by a third party (MAESTRO paper, arXiv 2607.08601, after LoRA recovery fine-tuning on attention only)**
- GPT-OSS-20B (32 experts/layer), Avg retained performance (RP) / Std RP across 17 tasks in 5 domains: 25% — MoP 95.63 / 5.10, REAP 97.67 / 2.70, MAESTRO 98.90 / 1.90, Random 95.38 / 6.00, HC-SMoE 90.46 / 8.60. 50% — MoP 88.41 / 10.80, REAP 89.37 / 10.40, MAESTRO 92.59 / 6.30, Random 83.92 / 11.50. Domain-accuracy column at 50%: MoP 43.06, REAP 44.70, MAESTRO 50.61 (base 56.82). 25%→50% RP drop: MoP 7.22, REAP 8.30, MAESTRO 6.31 — [MAESTRO Table 1 and §4](https://arxiv.org/abs/2607.08601) (numbers from local `maestro.txt`)
- "Note that MoP holds each expert's output for every calibration token in memory. While this is manageable for GPT-OSS-20B (32 experts per layer), it requires a prohibitively high amount of memory for Qwen-3-30B (128 experts per layer). As a result, we are unable to run MoP on Qwen" — [MAESTRO §4 Baselines](https://arxiv.org/abs/2607.08601)

### Inferences
- The within-cluster selector S_var is a pure router-concentration statistic and the Stage-1 enumeration is a reconstruction objective; the only *domain-aware* piece is the Spearman clustering of per-domain error profiles. So MoP's protection of code on Mixtral 8→4 comes from forcing one survivor per functional cluster (a coverage constraint), not from any explicit minority-domain objective. This is why it would also keep an expert for a domain that is *tiny* in the calibration mix: cluster count is tied to budget, not to domain mass.
- The Mixtral HumanEval 71.34 vs 1.22 result is a 2-of-8 → 4-of-8 experiment on an 8-expert model where every expert carries ~12% of the network; one wrong removal is catastrophic there in a way it is not with 256–384 experts. The Qwen (60-expert) numbers are far less dramatic (HumanEval 14.02 vs 1.21 at 17% pruning, 3.05 vs 1.83 at 33%) and the absolute values are so low that the harness may be broken; MBPP 14.00 for a 6-expert Mixtral is also implausibly low. Treat the headline as a Mixtral-specific demonstration.
- Third-party numbers (MAESTRO) put MoP *at or slightly below REAP* on a 32-expert model at both 25% and 50% and with *higher* cross-task variance at 25% (5.10 vs 2.70) — the opposite of what a "keep thin domains alive" method should show. On present evidence MoP is not a demonstrated improvement over REAP for the fairness objective.
- The compute profile (single-expert dense forward of every expert on every token, plus storing those outputs) is what REAP/HOPE calibration explicitly avoids; at 256–384 experts this is a different order of cost and is the stated reason MAESTRO could not run it at 128.

### Gaps
- m (number of Stage-1 general experts) and the mixed-diversity calibration recipe (Appendix A) are absent from every available copy of the paper, so the method cannot be reproduced exactly from the text; the GitHub repo contains no usable code at time of checking.
- No unpruned GSM8K/MATH/HumanEval/MBPP numbers are reported, so retention ratios cannot be computed.
- No results beyond 50% pruning on any model, and no model with >60 experts/layer in the paper itself.

---

## KQ3. Inputs: what each method consumes vs the banked accumulators

### Takeaway
MESA needs only the per-(layer, expert, domain) router-mass sums — which the banked `gate_sum[l,e,d]` (and `cnt` for a frequency variant) provide directly — so it is a zero-GPU CPU step on existing accumulators, with one caveat about calibration regime. Exact MoP is *not* computable from the accumulators (it needs per-token hidden states for K-Means and per-token single-expert outputs for the error profiles), but a "MoP-lite" that replaces discovered domains with the banked domain labels and the error profile with the per-domain saliency profile is.

### Cited Findings
- MESA inputs, per Algorithm 1: R ∈ ℝ^{L×E} aggregate router mass and {D_d ∈ ℝ^{L×E}} per-domain router mass; nothing else. Scoring compute = "Static per-domain L × E router-mass matrix, already produced by the calibration harvest"; allocation = "CPU-only iterative minimax reweighting over the L × E matrix (∼15–20 steps)"; extra forward passes = 0 — [arXiv HTML Alg. 1, Table 10](https://arxiv.org/html/2609.04453)
- MESA calibration-regime dependency: with a decode-time harvest the per-domain mass vectors correlated 0.57–0.86 and MESA "degenerated toward router mass"; with a prefill harvest cross-domain correlation ≈ 0 — [arXiv HTML §4.6](https://arxiv.org/html/2609.04453)
- MoP inputs: cached "input hidden states and their corresponding original layer outputs" (D_cache) for Stage-1 enumeration; token hidden states {x_t} for K-Means; single-expert outputs E_i(x_t) for every expert i and every token t for Eq. 7; per-token softmax probabilities p_{t,i} for S_var — [arXiv HTML Eqs. 1, 2, 5, 7](https://arxiv.org/html/2511.19822v1); confirmed by MAESTRO: "MoP holds each expert's output for every calibration token in memory" — [MAESTRO §4](https://arxiv.org/abs/2607.08601)
- Generic TB-Coverage inputs: a REAP-style score per expert, per layer, per corpus ("router weight multiplied by the expert output L2 norm, averaged over tokens where the expert is in the top-k") computed separately on WikiText2 and C4 — [arXiv HTML 2607.01710](https://arxiv.org/html/2607.01710)
- SHAPE inputs: "Routing decisions and gating weights from 25 calibration examples per task, collected with standard inference on the unpruned model. No retraining or extra forward evaluations"; coalition utilities approximated from co-occurrence counts — [arXiv HTML 2606.09886](https://arxiv.org/html/2606.09886v1)
- EASY-EP inputs: per token, gate value × L2 norm of expert output (g·‖e‖, only when g > 0), weighted by s_t = 1 − cos(hidden before, hidden after routed experts); mixed-domain score = average of per-domain normalised scores; 25 instances per domain — [arXiv HTML 2504.06792v2](https://arxiv.org/html/2504.06792v2)
- HOPE: calibration-set sensitivity "is shared across all methods, and HOPE is not more sensitive than REAP (HOPE's average calibration Jaccard index is 0.60, versus REAP's 0.61)" — [HOPE §4.4](https://arxiv.org/abs/2510.13999) (local `hope.txt`; HOPE arXiv id not re-verified here)

### Inferences (mapping to the banked accumulators: per-layer, per-expert, per-domain `sum_saliency, count, sum_by_bucket, cnt_by_bucket, gate_sum, norm_sum, hist`; per-layer interaction matrix F; per-token top-40 router-score cache)
- **MESA — fully offline, zero GPU.** D_d[l,e] := gate_sum[l,e,d] / Σ_e gate_sum[l,e,d] (per-layer row-normalised, matching Fig. 5's ≈1/E scale); R[l,e] := Σ_d gate_sum[l,e,d] / Σ_{e,d} (or token-weighted). If `gate_sum` is the realised top-k gate weight this matches the most likely reading of MESA's "router mass" for gpt-oss; if MESA used full-softmax mass, the top-40 router-score cache can reconstruct a near-exact full-softmax mass per domain (40 ≫ k). Also trivially computable variants: frequency-mass MESA (D_d from `count`), and saliency-mass MESA (D_d from `sum_saliency`, i.e. REAP mass) — the latter is a natural "MESA-on-REAP" that keeps the activation-norm term MESA discards, and is what the paper's "domain-aware REAP control" presumably approximates (it scored 58.4 vs MESA 65.66 on Gemma GPQA, so expect it to land between REAP and MESA).
- **Calibration-regime caveat.** Our accumulators come from a prefill (teacher-forced) pass over domain-labelled text. That is the regime where MESA found decorrelated per-domain mass and worked on OLMoE; it is *not* the decode-time regime of gpt-oss/Mixtral headline runs. If the per-domain gate_sum vectors are highly correlated (check: Pearson across domains of the flattened D_d; MESA saw 0.57–0.86 → degenerate, ≈0 → useful), MESA will reduce to plain router-mass ranking and buy nothing.
- **MoP-exact — not offline.** Needs per-token x_t and per-token single-expert outputs E_i(x_t) for all i; none of these are banked, and F (pairwise interaction) is a different object from per-expert-per-domain error. A new calibration pass with a dense all-experts forward is required (the thing REAP/HOPE calibration avoids), and at 256–384 experts it is the cost MAESTRO called prohibitive at 128.
- **MoP-lite — offline.** Replace K-Means domains with the banked domain labels (K = number of domains, decoupled from budget), replace v_{i,k} with the per-domain saliency profile sum_saliency[l,e,d]/count[l,e,d] (a "how much does expert e matter to domain d" vector — same monotone-rank role as the reconstruction error, inverted), cluster per layer by Spearman/Ward, and pick the survivor per cluster by S_var computed from `hist`/`cnt_by_bucket` (KL of the gate-value histogram vs the all-expert pooled histogram is a coarse proxy for KL-over-tokens) or exactly from the top-40 cache. Everything is CPU on L×E×D arrays. This is an extrapolation; nobody has published it.
- **TB-Coverage — fully offline.** Per-corpus REAP score = sum_saliency[l,e,d]/count[l,e,d]; round-robin over domains per layer with a budget B; protect, then fill the rest by aggregate score. Exactly our per-domain protection rule with a specific tie-breaking order.
- **SHAPE — offline given the top-40 cache.** Co-occurrence counts of top-k sets per layer are derivable from the cache; F may already encode a related pairwise term. Its per-layer coverage rule (keep smallest prefix covering α of the layer's score mass, bisection on α for the global rate) is a pure post-processing step on any score matrix, including HOPE's.
- **MAESTRO — offline given the top-40 cache** (needs cross-layer routing transitions per token; the cache gives per-token per-layer top sets). Its own evidence of fairness is lower Std RP across tasks, not a per-domain objective.

### Gaps
- Whether `gate_sum` in the banked accumulators is top-k gate weight or full softmax, and whether the harvest was prefill or decode, determines MESA's behaviour; neither can be read from the papers, only from our pipeline.

---

## KQ4. Composition with HOPE's pᵀFp objective and with per-domain protection

### Takeaway
No paper combines a minimax/coverage objective with an interaction-aware (HOPE-style) objective; the closest published pairings are TB-Coverage (round-robin protection *then* fill by aggregate score) and SHAPE (coalition scores *then* a per-layer coverage rule). MESA's Algorithm 1 is structurally a dual-ascent on per-domain linear constraints, so the clean composition is: HOPE objective + linear per-domain mass constraints, solved by alternating HOPE's selection step with MESA's multiplier boost — this is an inference, not a published result.

### Cited Findings
- MESA Algorithm 1 updates the score matrix additively, S ← RowNorm(S + η_t·D_{d*}), where d* is the currently worst domain, and the paper describes it as "inspired by distributionally robust optimization" — [arXiv HTML §3.2](https://arxiv.org/html/2609.04453)
- MESA's authors explicitly reject naive per-domain protection ("keeping each domain's top experts separately … would produce nearly identical selections for all domains") because top-8 overlap across domains is 0.41–0.47 — [arXiv HTML App. E.2](https://arxiv.org/html/2609.04453)
- Generic TB-Coverage composition: start from an ExpertSparsity-style reconstruction-stable candidate mask, force-retain the round-robin protected set (B = 20 Qwen / 24 DeepSeek per layer), then "drop the lowest-scoring unprotected experts by average score until K remain"; it beats REAP at 25%/50% retention (Qwen1.5-MoE avg 0.448 vs 0.438 at 25% retain; 0.598 vs 0.552 at 50% retain; DeepSeek-MoE-16B 0.448 vs 0.395; 0.606 vs 0.542) but is below REAP on MMLU for Qwen 25% (0.234 vs 0.247) and DeepSeek 50% (0.242 vs 0.287); authors list "No comparison against simple multi-corpus aggregation (mean or max of per-corpus scores)" as a missing ablation — [arXiv HTML 2607.01710](https://arxiv.org/html/2607.01710)
- SHAPE composition: Shapley-approximate coalition scores, negatives clipped to zero, then per layer keep the smallest prefix covering an α fraction of the layer's total score with bisection on α to hit the global keep rate; motivation is "layer collapse, where retained experts concentrate in a few layers"; Qwen3-30B-A3B avg at 40% pruning: SHAPE 81.31 vs EASY-EP 79.05 vs REAP 74.22 (unpruned 82.92); SHAPE trails REAP and EASY-EP on HumanEval at 40% — [arXiv HTML 2606.09886](https://arxiv.org/html/2606.09886v1)
- EASY-EP mixed-domain selection is a *mean* of normalised per-domain scores and loses a minority domain badly: DeepSeek-R1 FinanceIQ 82.1 → 57.95 under mixed-domain pruning at 50% retention — [arXiv HTML 2504.06792v2](https://arxiv.org/html/2504.06792v2)
- ConMoE (arXiv 2605.29350) separates "constructing an expert pool" (Anchor Coverage) from "assigning the original experts' workloads to that pool"; assignment alone adds +1.82 / +0.35 pp macro accuracy at 50% / 75% expert retention — [arXiv abs 2605.29350](https://arxiv.org/abs/2605.29350) (local `conmoe.txt`)

### Inferences
- Write MESA's objective in HOPE's variables. With keep-indicator p ∈ {0,1}^{L×E}, domain d's fractional loss is ℓ_d(p) = Σ_{l,e} D_d[l,e](1 − p_{l,e}) / Σ D_d — *linear* in p. MESA is min_p max_d ℓ_d(p) s.t. Σp = budget. Its Algorithm 1 is a Lagrangian/dual-ascent heuristic: the boost η_t·D_{d*} is a step on the multiplier of the active worst-domain constraint, and "global ranking on S" is the primal step for a linear objective. HOPE's objective is pᵀFp + linear saliency term (same primal variable). So the natural combination is min_p [pᵀFp − sᵀp] s.t. ℓ_d(p) ≤ τ ∀d (or minimax), solved by alternating (i) HOPE's greedy/QP step on the *reweighted* linear term s + Σ_d λ_d D_d and (ii) MESA's multiplier update λ_{d*} += η_t on the worst domain. Cost: T ≈ 15–20 HOPE solves on CPU. This is a straightforward extension of both algorithms; nothing in the literature reports it.
- Two cheaper compositions bracket it: (a) MESA as a *prior*: run Algorithm 1 on D_d built from per-domain saliency (sum_saliency) rather than gate mass, take S* as the linear term in HOPE — one HOPE solve, keeps F; (b) MESA as a *constraint*: take MESA's protected complement (experts whose removal pushes some ℓ_d past the worst-domain value), force-keep them, and let HOPE choose among the rest — this is exactly TB-Coverage's "protect then fill" pattern with a minimax-derived protected set instead of round-robin.
- Per-domain protection as currently implemented ("keep each domain's top experts") is the thing MESA's authors call insufficient because of 41–47% cross-domain top-set overlap: the protected sets coincide and the minority domain's *marginal* experts (the 0.41–0.47 overlap zone) are what gets cut. MESA's iterative reweighting targets exactly those marginal experts. The gain over a per-domain-protection REAP is bounded by the Gemma-4 control: 65.66 vs 58.4 GPQA (REAP plain 55.56).
- Caution from the trade-off data: a hard worst-domain constraint moves budget off the majority domains; on gpt-oss-20b that bought GPQA +17.8 over REAP and cost AIME −10.0 and WinoGrande −8.9. A τ-bounded (not minimax) constraint lets this be dialled: τ at the REAP-plan worst-domain loss reproduces REAP; τ at MESA's 0.0491 reproduces MESA; between is unexplored.

### Gaps
- No published method combines an interaction/coalition objective with a per-domain minimax or coverage constraint; SHAPE's coverage is per-*layer*, not per-domain. The composition above is reasoning, with no numbers.
- No ablation anywhere of "mean vs max of per-domain scores" vs minimax (TB-Coverage explicitly lists this as missing).

---

## KQ5. Scaling: largest models, fine-grained (128–384-expert) evidence

### Takeaway
MESA has been run on gpt-oss-120B (128 experts/layer, r=0.5) and Gemma-4-26B-A4B (128 experts, r=0.25); MoP's largest is Mixtral-8x7B/Qwen1.5-MoE (60 experts) and a third party could not run it at 128 experts. No method in this family has been demonstrated on 256–384-expert DeepSeek/Kimi/GLM-class routers; the only 256-expert expert-pruning numbers are EASY-EP's on DeepSeek-R1/V3 (mean-of-domains, not minimax).

### Cited Findings
- MESA: gpt-oss-120B 128→64 experts/layer at r=0.5 (Table 14 above); Gemma-4-26B-A4B-it 128→96 (Table 8); OLMoE 64→48/40/32 — [arXiv HTML App. I, §4.7, §4.6](https://arxiv.org/html/2609.04453)
- MoP: Mixtral-8x7B-Instruct (8 experts, "47B total") and Qwen1.5-MoE-A2.7B-Chat (60 experts, 14.3B) — [arXiv HTML 2511.19822v1](https://arxiv.org/html/2511.19822v1); MAESTRO "unable to run MoP on Qwen[3-30B]" at 128 experts for memory reasons — [MAESTRO §4](https://arxiv.org/abs/2607.08601)
- EASY-EP on DeepSeek-R1 and DeepSeek-V3-0324 (256 experts/layer): at 128 retained, domain-specific avg 66.55 vs 67.20 full (R1); mixed-domain 61.47 (≈91%); V3-0324 domain-specific 54.12 vs 54.08 full, mixed 54.92; gating-score-only pruning 56.49, frequency 31.59, M-SMoE merging 22.10 (R1, 128 experts); also reports 64 retained (25%) — [arXiv HTML 2504.06792v2](https://arxiv.org/html/2504.06792v2)
- SHAPE on Qwen3-30B-A3B (128 experts), GPT-OSS-20B, DeepSeek-V2-Lite (64) at 20%/40% — [arXiv HTML 2606.09886](https://arxiv.org/html/2606.09886v1)
- TB-Coverage on Qwen1.5-MoE (60) and DeepSeek-MoE-16B (64) at 25/50/75% retention; authors list "larger expert counts" as untested — [arXiv HTML 2607.01710](https://arxiv.org/html/2607.01710)
- C-PRUNE (PMLR v317) on DeepSeek-V2-Lite and Qwen1.5-MoE-A2.7B at a fixed 20% pruning rate only — [PMLR v317 yao26a](https://proceedings.mlr.press/v317/yao26a.html)
- MESA's "over-dispersed" diagnostic: gpt-oss 90.8%, OLMoE 92.5%, Mixtral 74.3%, Gemma-4 ≈69% of max entropy — [arXiv HTML §3.1, §4.7](https://arxiv.org/html/2609.04453)

### Inferences
- MESA's cost is O(T·L·E·D) on a matrix; at L≈60, E≈256–384, D≈10 that is trivial. The open question for fine-grained MoEs is not compute but whether per-domain gate mass is informative: with sigmoid/noaux_tc routing and 8-of-256 selection, per-token entropy is far below MESA's over-dispersed band, which puts a GLM/DeepSeek-class model in Mixtral's regime (where MESA collapsed) rather than gpt-oss's — unless, like Gemma-4, the per-domain mass vectors are decorrelated. The Gemma-4 case (128 experts, top-8, concentrated routing, MESA +14.3 GPQA over REAP) is the most relevant precedent and is favourable, but single-measurement.
- The EASY-EP FinanceIQ example (82.1→57.95 under domain-averaged selection at 50% on a 256-expert model) is direct evidence that mean-of-domains selection lets a thin domain collapse at exactly the scale and ratio of interest; it is the failure MESA's minimax is designed to prevent, but nobody has tested MESA there.

### Gaps
- No MESA, MoP, TB-Coverage or SHAPE result on any 256-expert or larger router; no result from any of them above 50% pruning except TB-Coverage's 75%-pruned (25%-retained) rows on 60/64-expert models and OLMoE cloze at 50%.

---

## KQ6. Sibling methods (minimax / fairness / coverage / clustering) with numbers

### Takeaway
Four siblings carry usable numbers: Generic TB-Coverage (round-robin per-corpus protection; beats REAP at 50–75% pruning on 60/64-expert models), SHAPE (coalition scores + per-layer coverage; +7 pts over REAP at 40% on Qwen3-30B), MAESTRO (Markov-stationary ranking; lowest cross-task variance; +3 pts RP over REAP at 50% after LoRA RFT), and EASY-EP (domain-specific few-shot; the 256-expert reference). C-PRUNE and ConMoE are clustering/coverage siblings with weaker or less relevant evidence.

### Cited Findings
- **Generic TB-Coverage (arXiv 2607.01710)**: per-layer round-robin over per-corpus REAP-score rankings until a protection budget B (20 Qwen / 24 DeepSeek) is filled; candidate mask from ExpertSparsity on C4; full table (avg of 6 zero-shot tasks / WikiText2 PPL / C4 PPL):
  - Qwen1.5-MoE-A2.7B — retain 25%: Random 0.417±0.021, REAP 0.438, ExpertSparsity 0.407, TB-Cov 0.448 (PPL 146.22 vs REAP 193.65); retain 50%: Random 0.535, REAP 0.552, ES 0.519, TB-Cov 0.598 (PPL 16.76 vs 23.77); retain 75%: Random 0.645, REAP 0.629, ES 0.655, TB-Cov 0.662.
  - DeepSeek-MoE-16B — retain 25%: Random 0.403, REAP 0.395, ES 0.404, TB-Cov 0.448; retain 50%: Random 0.488, REAP 0.542, ES 0.571, TB-Cov 0.606; retain 75%: Random 0.568, REAP 0.649, ES 0.678, TB-Cov 0.683.
  - Limitations: GSM8K/Math500 "near zero for all methods"; REAP baseline used Evol-CodeAlpaca calibration; no code link; arXiv non-exclusive licence — [arXiv HTML 2607.01710](https://arxiv.org/html/2607.01710)
- **SHAPE (arXiv 2606.09886)**: Qwen3-30B-A3B avg — Unpruned 82.92; 20%: REAP 81.86, EASY-EP 82.35, SHAPE 82.43; 40%: REAP 74.22, EASY-EP 79.05, SHAPE 81.31; GPT-OSS 40% SHAPE avg reported as 79.02 (Table I) vs 77.53 (Table III) — an internal inconsistency; code github.com/Alizen-1009/Shapley-Moe — [arXiv HTML 2606.09886](https://arxiv.org/html/2606.09886v1)
- **MAESTRO (arXiv 2607.08601)**: expert activation across layers modelled as ergodic Markov chains; importance = stationary distribution; LoRA recovery FT on attention projections only; GPT-OSS-20B 50%: MAESTRO RP 92.59 / Std 6.30 vs REAP 89.37 / 10.40 vs MoP 88.41 / 10.80; Qwen3-30B 50%: MAESTRO 93.40 / 10.50 vs REAP 90.25 / 7.10; relative margin over REAP 3.60% (GPT-OSS) and 3.49% (Qwen3) at 50% — [arXiv abs 2607.08601](https://arxiv.org/abs/2607.08601) (local `maestro.txt`)
- **EASY-EP (arXiv 2504.06792, NeurIPS 2025)**: see KQ5; 5 demonstrations recover >90% of critical experts, 25 capture ≈99%; "reasoning models (R1) are less robust than V3, with long-generation ability possibly compromised" — [arXiv HTML 2504.06792v2](https://arxiv.org/html/2504.06792v2)
- **C-PRUNE (PMLR v317, Yao et al. 2026)**: expert embedding φ(f_i) = mean expert output over calibration tokens (functional, not weights, despite the abstract's "parameter similarity"); cosine affinity; agglomerative merge while |C| > N−r; prune experts outside a layer-specific radius of kept centroids; hyperparameters: 12 hierarchical clusters, 6 or 8 global clusters, 20% pruning rate. DeepSeek-V2-Lite 15.7B→13.0B, MMLU 47.88→45.37 (random 16.77, Seer 29.59, Group&Merge 31.58); Qwen1.5-MoE 14.3B→11.8B, MMLU 43.29→39.82; MMLU-Medical DSV2L 56.79→48.27, MedQA 41.96→40.70; code github.com/Fighoture/MoE_unsupervised_pruning — [PMLR v317 yao26a](https://proceedings.mlr.press/v317/yao26a.html) (local `cprune.txt`)
- **Attribution-Guided Coverage-Maximized (arXiv 2606.18304)**: coverage is of *channel* importance mass within a layer/expert, not of domains; 25% (+4-bit) and 50% only; Qwen3-30B-A3B avg 75.64 → 70.81 at P50%; code github.com/yifu-ding/MoE-Slimming — [arXiv HTML 2606.18304](https://arxiv.org/html/2606.18304v1)
- **Robust calibration study (arXiv 2610.15578)**: empirical only (Lu et al. 2024 enumeration on gpt-oss-20b); "scaling calibration data gives little benefit beyond small sizes"; optimal region ≈ three-quarters of experts retained; no minimax/clustering proposal — [arXiv HTML 2610.15578](https://arxiv.org/html/2610.15578)
- **ConMoE (arXiv 2605.29350)**: Anchor Coverage pool + router-conditioned functional assignment; 50%/75% retention; +1.82 pp macro from assignment at 50% — [arXiv abs 2605.29350](https://arxiv.org/abs/2605.29350)

### Inferences
- TB-Coverage is the most directly transferable sibling: it is literally "per-domain protection with round-robin order, then REAP fill", runs on the banked per-domain saliency, and has numbers at 50% and 75% *pruning* — the only sibling evidence in the 50–70% band — where it beats REAP by +4.6/+1.0 (Qwen 50%/75% pruned) and +6.4/+5.3 (DeepSeek) avg points. Its own authors note it never tested mean/max aggregation as a control, so part of the gain may be just multi-corpus scoring.
- MAESTRO's lower Std RP is the only "fairness" signal measured across 17 tasks; it is obtained *after* LoRA RFT, so the pre-RFT fairness of the selection is unknown.

### Gaps
- No method explicitly labelled "balanced expert pruning" or "domain-robust pruning" surfaced in searches; the closest are MESA (minimax) and TB-Coverage (coverage).

---

## KQ7. Failure modes: looping/degeneration vs benchmark accuracy

### Takeaway
None of the papers measures repetition, looping or open-ended generation quality; the documented catastrophic outcomes are score collapses (MESA on Mixtral GSM8K 3.49; RM-Uniform 0.00 on all tasks at r=0.5 on gpt-oss-120B; enumeration pruning HumanEval 1.22), which are consistent with but not evidence of degeneration.

### Cited Findings
- MESA on Mixtral-8x7B-Instruct r=0.25: GSM8K 3.49 vs 31.42 unpruned, WikiText-2 PPL 16.43 vs 4.78 — [arXiv HTML Table 4](https://arxiv.org/html/2609.04453)
- RM-Uniform at r=0.5 on gpt-oss-120B: 0.00 on all 11 tasks, "a real degeneracy of the method under aggressive pruning" — [arXiv HTML Table 14](https://arxiv.org/html/2609.04453)
- Enumeration pruning Mixtral 8→4: HumanEval 1.22, MBPP 1.55 — [arXiv HTML Table 5](https://arxiv.org/html/2511.19822v1)
- EASY-EP: on reasoning models "long-generation ability possibly compromised" — [arXiv HTML 2504.06792v2](https://arxiv.org/html/2504.06792v2)
- MESA evaluations use vLLM, temperature 0.6, "medium reasoning effort, 128K max tokens" — i.e. long generations were allowed, and AIME/GSM8K scores were still obtained — [arXiv HTML §4.1](https://arxiv.org/html/2609.04453)
- A targeted search for repetition/degeneration in pruned MoEs returned only the REAP paper and general pruning papers; no independent reports of looping in REAP-pruned models were found — [search summary citing REAP ICLR 2026](https://arxiv.org/abs/2510.13999)

### Inferences
- Because MESA shifts budget away from high-mass "generalist" experts toward marginal domain-specialists, and because the router after pruning is unchanged (top-k over the survivors), a MESA plan removes more of the experts that handle the common-token stream. If looping at high ratios is driven by losing the experts that stabilise frequent tokens, a minimax plan is *more* exposed to it than REAP at the same ratio, not less; MESA's GSM8K improvement (+4.01) argues against this at r=0.25 on gpt-oss, but the Mixtral result argues for it in a concentrated-routing model. This is reasoning, not measurement.
- MoP's S_var selector deliberately prefers experts with *concentrated* token activation (specialists) within each cluster, i.e. it also de-prioritises the broad-coverage experts; the same exposure applies.

### Gaps
- No paper in this set reports repetition rate, n-gram loop metrics, or generation length distributions for pruned models; the looping failure at high ratios remains undocumented in the literature found.

---

## KQ8. Code availability and licence

### Takeaway
MESA: no code found (paper CC BY 4.0). MoP: MIT-licensed GitHub repo that is effectively empty. Siblings with code: SHAPE, C-PRUNE, MoE-Slimming.

### Cited Findings
- MESA: paper licence CC BY 4.0; no repository linked on arXiv or the NeurIPS page; GitHub search for the method name finds nothing — [arXiv abs](https://arxiv.org/abs/2609.04453); [NeurIPS listing](https://neurips.cc/virtual/2026/170047)
- MoP: github.com/Saul-James/MoP, MIT licence, 2 commits, README "# MoP", `MOP/` folder with no listed contents — [GitHub](https://github.com/Saul-James/MoP)
- SHAPE: github.com/Alizen-1009/Shapley-Moe — [arXiv HTML 2606.09886](https://arxiv.org/html/2606.09886v1)
- C-PRUNE: github.com/Fighoture/MoE_unsupervised_pruning — [PMLR v317](https://proceedings.mlr.press/v317/yao26a.html)
- MoE-Slimming (attribution/coverage): github.com/yifu-ding/MoE-Slimming — [arXiv HTML 2606.18304](https://arxiv.org/html/2606.18304v1)
- TB-Coverage, MAESTRO, EASY-EP: no code link found on their arXiv pages.

### Inferences
- MESA's Algorithm 1 is fully specified in the paper (≈12 lines, two hyperparameters with a flat sensitivity surface), so absence of code is not a blocker; it is a ~50-line NumPy function over the banked `gate_sum` tensor.

### Gaps
- Licence of the (absent) MESA code and of the MoP `MOP/` contents cannot be determined.
