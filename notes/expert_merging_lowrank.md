# Expert merging and low-rank/SVD expert compression as alternatives or complements to expert pruning (500B–1T MoE)

Scope: methods that reduce MoE expert count or expert size by MERGING or FACTORIZING experts (MC-SMoE/M-SMoE, HC-SMoE, EEP, SEER-MoE, MoE-SVD, D2-MoE, MoE-I2, MoLAE, MoBE, RFID-MoE, Sub-MoE, REAM, SlimQwen, MiLo, TileQ, KBVQ-MoE, EAC-MoE), with REAP (pruning) as the reference point. Research date: 2026-10-17. All numbers are as self-reported by the papers unless stated; no independent replications were found for any of them.

## Key Question 1: Per-method byte/expert reduction, quality delta, scale tested, training requirement

### Takeaway
Training-free merging methods (HC-SMoE, M-SMoE, REAM) have only been demonstrated up to ~106B (GLM-4.5-Air) and lose far more than pruning at 50% expert removal on generative tasks; the only factorization method demonstrated at 671B–1T scale is MoBE (24–30% parameter reduction, 1–2 point accuracy drop, but requires gradient-based basis fitting on H100s and a custom inference kernel). No merge/factorize method has reported a 50% or larger byte reduction at >100B scale with near-lossless generative quality.

### Cited Findings

**MC-SMoE / M-SMoE (ICLR 2024, Li et al.)**
- Two phases: M-SMoE aligns neurons across experts, groups dominant experts with members, merges each group weighted by activation frequency; the "compress" phase decomposes merged experts into low-rank + structurally sparse forms — [arXiv 2310.01334](https://arxiv.org/html/2310.01334v2)
- Base models: Switch-base-32 (2.0B total / 220M active, 32 experts per layer) and fairseq-moe-15b (15B, 512 experts per layer). Largest model tested: 15B — [arXiv 2310.01334v2](https://arxiv.org/html/2310.01334v2)
- Switch-base-32: 32 → average 8 experts per layer; model size 2.0B → 381M (~80%), memory 3895MB → 715MB; TFLOPs 4.65 → 3.83 (~18%, paper rounds to 20%) — [arXiv 2310.01334v2](https://arxiv.org/html/2310.01334v2)
- Fine-tuned benchmark deltas (Switch): SST-2 95.75→93.35, MRPC 90.20→89.22, MultiRC 76.19→73.98, WinoGrande 61.80→59.52, SQuAD F1 85.81→85.30 — [arXiv 2310.01334v2](https://arxiv.org/html/2310.01334v2)
- Zero-shot deltas (fairseq-moe-15b, 512→16 experts, NO fine-tune): MRPC 60.54→47.55, OpenBookQA 36.80→34.60, WinoGrande 51.78→49.09 — [arXiv 2310.01334v2](https://arxiv.org/html/2310.01334v2)
- Training: the Switch results use post-merge fine-tuning with KD from the full SMoE (amount not quantified); low-rank rank r=32, sparse ratio 0.1 in the compress phase — [arXiv 2310.01334v2](https://arxiv.org/html/2310.01334v2)
- Code: [github.com/UNITES-Lab/MC-SMoE](https://github.com/UNITES-Lab/MC-SMoE/blob/main/README.md)

**HC-SMoE (ICML 2025, Chen et al.) — retraining-free hierarchical-clustering merge**
- Clusters experts by output similarity on a calibration set (32 × 2048-token C4 sequences, ~65.5K tokens), merges each cluster with frequency-weighted averaging; no training — [arXiv 2410.08589v4](https://arxiv.org/html/2410.08589v4)
- Qwen1.5-MoE-A2.7B: 60→45 experts (14.3B→11.2B) avg 0.6008→0.5716; 60→30 (→8.1B) 0.5223. Mixtral 8x7B: 8→6 (46.7B→35.6B) 0.6748→0.6425; 8→4 (→24.3B) 0.5729 — [arXiv 2410.08589v4](https://arxiv.org/html/2410.08589v4)
- DeepSeek-MoE-16B-Base 64→32 experts: average 0.5691 → 0.4780 — [arXiv 2410.08589v4](https://arxiv.org/html/2410.08589v4)
- Memory: Mixtral 87.49GB → 45.49GB at 4 experts; throughput unchanged (13.45→13.96 tok/ms). Merge runtime: Mixtral ~111s on 8×V100; DeepSeek-MoE-16B under 10 min on one A100 — [arXiv 2410.08589v4](https://arxiv.org/html/2410.08589v4)
- Largest model tested in the paper: Mixtral 8x7B (46.7B). Code: github.com/wazenmai/HC-SMoE — [arXiv 2410.08589](https://www.arxiv.org/abs/2410.08589v1)

**EEP — Efficient Expert Pruning (+ merging as fine-tune substitute), 2024**
- Gradient-free evolutionary search of a router-mapping matrix and an expert-merging matrix; pruning phase uses one-hot rows (select), merging phase uses continuous rows (blend). 40 search iterations for pruning, 160 for merging — [arXiv 2407.00945v1](https://arxiv.org/html/2407.00945v1)
- Mixtral 8x7B-Instruct (10-task avg): full 62.4; 4 experts 70.3 (prune) / 74.2 (prune+merge); 2 experts 59.7 / 65.6. Mixtral 8x22B-Instruct (5-task avg): full 66.5; 4 experts 78.1 / 80.4; 2 experts 63.7 / 67.1 — [arXiv 2407.00945v1](https://arxiv.org/html/2407.00945v1)
- Caveat: fitness is accuracy on a sample of each task's training set, so these are task-specialised (IID) numbers; MMLU OOD drops 72.6 → 64.6 at 4 experts with merging — [arXiv 2407.00945v1](https://arxiv.org/html/2407.00945v1)
- Peak GPU memory Mixtral 8x7B: 88.6GB → 46.6GB (4 experts) → 25.6GB (2 experts). Largest model: Mixtral 8x22B (141B). Search cost not reported; the paper itself says the search is "still potentially costly". Code: github.com/imagination-research/EEP — [arXiv 2407.00945v1](https://arxiv.org/html/2407.00945v1)

**SEER-MoE (2024, Muzio, Sun, He)**
- Two stages: heavy-hitters-count pruning of expert count, then regularised fine-tuning to reduce experts-per-token. Example: 25% fewer experts + top-1 activation → ~25% params and ~27% FLOPs reduction with "slight" drop; target model Mixtral 8x7B — [arXiv 2404.05089](https://www.arxiv.org/abs/2404.05089)
- Requires fine-tuning; no merging or factorisation component; no open code found in this search (gap).

**MoE-SVD (ICML 2025, Li et al.) — training-free expert SVD**
- Decomposes experts into low-rank matrices, selects which expert layers to decompose by sensitivity (weight singular values + activation statistics), shares a single V-matrix across experts, top-k selection of U-matrices; reports 60% compression ratio and 1.5× faster inference on Mixtral, Phi-3.5, DeepSeek and Qwen2 MoE "with minimal performance loss"; no training — [PMLR v267](https://proceedings.mlr.press/v267/li25az.html)
- Code: github.com/lliai/MoE-SVD (per ICML poster page) — [ICML 2025 poster](https://icml.cc/virtual/2025/poster/44786)
- Gap: I could not retrieve the full text (no arXiv ID found; the PDF was not fetched), so per-benchmark numbers are unavailable here. D2-MoE from the same group is the better-documented sibling.

**D2-MoE — Delta Decompression (ICML 2025)**
- Each expert = shared Fisher-merged base weight + expert delta; delta compressed with activation-aware truncated SVD; base pruned column-wise (semi-dynamic). Training-free, 512 WikiText-2 calibration samples — [arXiv 2502.17298v1](https://arxiv.org/html/2502.17298v1)
- SVD energy retained at 20/40/60/70/80% compression: 68.05 / 47.34 / 26.62 / 16.26 / 5.93% — [arXiv 2502.17298v1](https://arxiv.org/html/2502.17298v1)
- Mixtral-8x7B (orig zero-shot avg 0.63): 20% → 0.60 (WikiText PPL 4.65), 40% → 0.57 (5.28), 60% → 0.52 (6.46). DeepSeekMoE-16B (orig 0.56): 20% 0.54, 40% 0.49, 60% 0.41. Qwen2-57B-A14B (orig 0.59) at 40%: 0.58. Phi-3.5-MoE (orig 0.62) at 40%: 0.60 — [arXiv 2502.17298v1](https://arxiv.org/html/2502.17298v1)
- Compute (Mixtral): Fisher merging ~11 min at 131 GB memory; SVD stage 25–44 min at 109 GB. Code: github.com/lliai/D2MoE — [arXiv 2502.17298v1](https://arxiv.org/html/2502.17298v1)
- Scale limit: MoBE reports D2-MoE is "computationally infeasible on an 8×H100 machine" for Qwen3-235B / DeepSeek-V3 / Kimi-K2 due to its backward-pass cost — [arXiv 2508.05257v1](https://arxiv.org/html/2508.05257v1)

**MoE-I2 (2024) — inter-expert pruning + intra-expert low-rank**
- Layer-wise genetic search prunes experts non-uniformly per layer, then low-rank decomposes survivors; tested on Qwen1.5-MoE-A2.7B, DeepSeek-V2-Lite, Mixtral-8x7B; abstract gives no ratios or numbers; code github.com/xiaochengsky/MoEI-2 — [arXiv 2411.01016](https://arxiv.org/abs/2411.01016)
- In D2-MoE's tables MoE-I2 is substantially worse than D2-MoE (e.g. Mixtral 60%: zero-shot 0.36 vs 0.52; PTB PPL 182.99 vs 23.63) — [arXiv 2502.17298v1](https://arxiv.org/html/2502.17298v1)

**MoLAE — Mixture of Latent Experts (Huawei, 2025)**
- Factorises each expert into a shared projection plus small expert-specific matrices; argued to pay off when hidden dim >> expert intermediate dim (DeepSeek-V3 is the motivating case); conversion from a pre-trained MoE via a two-step algorithm — [arXiv 2503.23100v2](https://arxiv.org/html/2503.23100v2)
- As a baseline in MoBE's Table 3 at 24–30% compression: DeepSeek-V3-0324 79.3 → 73.1 (MoLAE) vs 78.0 (MoBE); Kimi-K2-Instruct 82.4 → 76.6 (MoLAE) vs 81.1 (MoBE); Qwen3-235B 81.5 → 73.2 vs 80.9 — [arXiv 2508.05257v1](https://arxiv.org/html/2508.05257v1)
- LatentMoE (2026) criticises MoLAE as compressing only the FC2 experts and so forgoing dispatch-communication and bandwidth savings — [arXiv 2601.18089](https://arxiv.org/pdf/2601.18089)

**MoBE — Mixture-of-Basis-Experts (ICLR 2026, Inclusion AI) — the only factorisation demonstrated at 671B–1T**
- Factorises up and gate matrices only: W^i ≈ A^i · f(Σ_j α^{i,j} B^j), A^i expert-specific, B^j shared per layer, α convex weights, f = SiLU/Tanh; down matrices are NOT factorised; rank r = p (intermediate dim); basis counts m=32 (Qwen3-30B/235B), 64 (DeepSeek-V3), 128 for Kimi-K2 split into two sequential groups of 64 — [arXiv 2508.05257v1](https://arxiv.org/html/2508.05257v1)
- Parameter reduction (Table 3): Qwen3-235B-A22B-2507 24%, DeepSeek-V3-0324 30%, Kimi-K2-Instruct 24%, Qwen3-30B-A3B 24% — [arXiv 2508.05257v1](https://arxiv.org/html/2508.05257v1)
- 15-benchmark average: Qwen3-235B 81.5 → 80.9 (−0.6); DeepSeek-V3-0324 79.3 → 78.0 (−1.3); Kimi-K2-Instruct 82.4 → 81.1 (−1.3); Qwen3-30B-A3B 78.6 → 75.8 (−2.8) — [arXiv 2508.05257v1](https://arxiv.org/html/2508.05257v1)
- Fitting: data-free, but gradient-based — Adam lr 0.07 on H100s, reconstruction-MSE objective, layer by layer (Algorithm 1); README gives 10,000 epochs for DeepSeek-V3 and 15,000 for Kimi-K2; GPU-hours and peak memory NOT reported — [arXiv 2508.05257v1](https://arxiv.org/html/2508.05257v1); [github.com/inclusionAI/MoBE](https://github.com/inclusionAI/MoBE)
- Inference: no speed measured; the limitations section says MoBE calls the fused-MoE kernel multiple times and needs a dedicated mega-kernel; MoBE adds activated params, so MoBE† reduces top-k (8→6) to compensate — [arXiv 2508.05257v1](https://arxiv.org/html/2508.05257v1)
- README: SGLang inference with fused-MoE and "MoBE mega-kernel" listed, "optimized inference kernels upcoming"; no HF checkpoints released; 40 stars, 7 commits — [github.com/inclusionAI/MoBE](https://github.com/inclusionAI/MoBE)

**RFID-MoE (Feb 2026) — rank allocation by effective rank × routing frequency**
- Factorises up/gate only (down dense); experts grouped by routing frequency in groups of 4 with a shared SVD basis; ranks allocated ∝ C_g = ξ·EffRank + (1−ξ)·Freq, ξ≈0.7–0.8; plus a ~3%-of-params trainable residual vector; frozen model, Adam on Frobenius reconstruction loss (no dataset, but gradient-based); 1,024 WikiText-2 samples for routing frequencies — [arXiv 2602.09316v2](https://arxiv.org/html/2602.09316v2)
- Qwen3-30B-A3B at 60% compression: WikiText-2 PPL 9.57 vs MoBE 13.96; avg accuracy 0.64 vs MoBE 0.61; at 20%: 0.72 vs 0.71. Largest model: Qwen3-235B-A22B-2507 (figure only; PTB 10.67 vs MoBE 11.13 vs original 10.11). Hardware: 8×A100. No code link — [arXiv 2602.09316v2](https://arxiv.org/html/2602.09316v2)

**Sub-MoE (2025, work in progress) — joint-SVD subspace merging**
- K-means on expert-output cosine similarity, then joint SVD on concatenated expert weights to share U within a group, merge V by frequency; Mixtral-8x7B retains ~96% of zero-shot at 25% expert reduction and ~86% at 50%; code github.com/lliai/MoERazor "will be released" — [arXiv 2506.23266v1](https://arxiv.org/abs/2506.23266v1)

**REAM — Router-weighted Expert Activation Merging (Apr 2026, Mila/Samsung SAIL)**
- "Pseudo-pruning": top-N′ experts by REAP saliency become centroids; the rest are absorbed into centroids (group size C=16 or 32), most centroids stay singletons; gate-logit + softmax-gated output similarity; neuron permutation alignment; sequential layer-by-layer merging with re-forward; training-free — [arXiv 2604.04356v1](https://arxiv.org/html/2604.04356v1)
- Models: Qwen3-30B-A3B-2507 (25%/50%), Qwen3-Next-80B-A3B, Qwen3-Coder-Next, GLM-4.5-Air (25% only). Cost: ~1–1.5 h at ~30 GB VRAM for Qwen3-30B. Code github.com/SamsungSAILMontreal/ream; HF collection SamsungSAILMontreal/ream — [arXiv 2604.04356v1](https://arxiv.org/html/2604.04356v1)
- Numbers in Key Question 2.

**SlimQwen (May 2026, Alibaba) — merging with continued pretraining**
- Qwen3-Next-80B-A3B (512 experts) → 23B-A2B (256 experts/layer, 8 routed + 1 shared active) with continued pretraining (120B–400B tokens); "partial-preservation merging": keep top half of target experts intact, merge each discarded expert into its most cosine-similar base with importance-weighted averaging — [arXiv 2605.08738](https://arxiv.org/html/2605.08738)
- Abstract: different one-shot expert compression methods reach similar final performance after large-scale continued pretraining; progressive pruning beats one-shot at equal tokens — [arXiv 2605.08738 abs](https://arxiv.org/abs/2605.08738)

**SlimMoE (Microsoft, 2025)** — keeps all experts, prunes neurons inside experts plus staged distillation; Phi-3.5-MoE → Phi-mini-MoE/Phi-tiny-MoE with <10% of original training data; one-shot pruning caused large early drops, so intermediate distillation is used — [arXiv 2506.18349v1](https://arxiv.org/html/2506.18349v1)

### Inferences
- Byte reduction from the merge family at 50% expert removal is identical to pruning (same expert count) — the question is only quality, and at 50% every training-free merge method tested loses more generative quality than REAP (see KQ2).
- Factorisation methods tested at 1T scale (MoBE) buy only ~24–30% of parameters; for a 1T BF16 model that is ~1.5–1.6 TB → ~1.1–1.2 TB, nowhere near a 122 GiB envelope on its own. Extrapolation: a 50% REAP + 24% MoBE + 4-bit would land near 1T × 0.5 × 0.76 × 0.25 ≈ 95 GB equivalent-parameter bytes, but no paper has tested this composition.

### Gaps
- MoE-SVD per-benchmark numbers (full text not retrieved).
- SEER-MoE code availability and exact benchmark deltas.
- MoBE GPU-hours and peak memory for DeepSeek-V3/Kimi-K2 fitting are not reported anywhere I found.

## Key Question 2: Merging vs dropping at the same final expert count — head-to-heads

### Takeaway
Three independent 2025–2026 head-to-heads agree: merging wins or ties on perplexity/multiple-choice and at mild (25%) compression, but at 50% REAP pruning beats every training-free merge method on generative tasks (coding, math, instruction following); the 2026 REAM hybrid (mostly pruning, a little merging) edges REAP at 25% on 80–106B models but still loses to REAP at 50%.

### Cited Findings
- REAP (ICLR 2026, Cerebras), Qwen3-30B-A3B at 50%: Code avg REAP 0.541 vs M-SMoE 0.397 vs HC-SMoE 0.364 (baseline 0.558); WildBench REAP 0.718 vs M-SMoE 0.725 vs HC-SMoE 0.008; Math REAP 0.857 vs 0.831 vs 0.728; MC REAP 0.503 vs 0.451 vs 0.542. GLM-4.5-Air at 50%: Code REAP 0.515 vs M-SMoE 0.284 vs HC-SMoE 0.419; WildBench 0.754 vs 0.391 vs 0.593; Math 0.857 vs 0.465 vs 0.700 — [arXiv 2510.13999v2](https://arxiv.org/html/2510.13999v2)
- At 25% the gap is small: Qwen3-30B Code REAP 0.551 vs M-SMoE 0.537; GLM-4.5-Air Code REAP 0.585 vs HC-SMoE 0.550 vs M-SMoE 0.528 — [arXiv 2510.13999v2](https://arxiv.org/html/2510.13999v2)
- REAP's summary: excluding large models, REAP's mean coding decrease is 1.9% (25%) and 6.9% (50%); both merging methods exceed 5% at 25% and 20% at 50%. HC-SMoE is generally stronger than REAP on MC — [arXiv 2510.13999v2](https://arxiv.org/html/2510.13999v2)
- Theory: merging two experts under a summed gate has an irreducible error proportional to Var[r(x)] (router-policy variability) × expert gap; "any fixed α cannot overcome" it; pruning keeps the router's input-dependent control over survivors. The paper flags this as a simplified (linear-expert) analysis — [arXiv 2510.13999v2](https://arxiv.org/html/2510.13999v2)
- Collapse evidence: merging shrinks PCA spread of expert outputs by up to 100× in late layers of Qwen3-30B; merged models show lower n-gram diversity and faster logit divergence; distances between clustered experts exceed those between pretrained and instruction-tuned checkpoints; capping clusters at 32 experts causes large coding-quality drops — [arXiv 2510.13999v2](https://arxiv.org/html/2510.13999v2); [Cerebras blog](https://www.cerebras.ai/blog/reap)
- Merging baselines were NOT run on the large models (Table 3 has no merging rows for Qwen3-Coder-480B / Kimi-K2) — [arXiv 2510.13999v2](https://arxiv.org/html/2510.13999v2)
- HC-SMoE's own comparison (zero-shot MC-style avg, same expert count): Qwen 60→30: HC-SMoE 0.5223 vs best prune 0.4528 vs M-SMoE 0.3221; Mixtral 8→4: HC-SMoE 0.5729 vs O-prune 0.5728 (tie); Mixtral 8→3: O-prune 0.5531 beats HC-SMoE 0.5458 — [arXiv 2410.08589v4](https://arxiv.org/html/2410.08589v4)
- REAM (Apr 2026) GEN suite (IFEval, AIME25, GSM8K, GPQA-D, HumanEval, LCB), Qwen3-30B at 25% (best calibration mix): REAM 69.8, REAP 68.6, Freq 67.6, HC-SMoE 67.4 (orig 70.9). At 50%: REAP 62.8 best vs REAM 60.7 vs HC-SMoE 52.1 vs Freq 48.7; REAM ranges 33.8–60.7 across calibration mixes (high sensitivity) — [arXiv 2604.04356v1](https://arxiv.org/html/2604.04356v1)
- REAM 25% on larger models (Original → REAP → REAM): Qwen3-Next-80B-A3B 72.9 → 69.6 → 71.5; Qwen3-Coder-Next 72.9 → 70.7 → 72.9; GLM-4.5-Air 77.1 → 71.9 → 73.9. Ablation: replacing REAP saliency with routing frequency costs ΔAVG −8.7 — [arXiv 2604.04356v1](https://arxiv.org/html/2604.04356v1)
- SlimQwen (with 400B tokens of continued pretraining after compression): REAP pruning MMLU 69.11 / GSM8K 73.69 / EvalPlus 53.59; partial-preservation merging (Soft Logits + Router Weights) MMLU 69.28 / GSM8K 74.18 / EvalPlus 48.00; authors conclude no single one-shot method is consistently best once retrained — [arXiv 2605.08738](https://arxiv.org/html/2605.08738)
- A 2026 survey frames expert reduction as the two paradigms (pruning, merging); a June 2026 paper reports "routing breakdown" where merged routers stop sending tokens to suitable experts — [search summary, chatpaper 290243 — not fetched; treat as unverified](https://chatpaper.com/paper/290243)

### Inferences
- For the Thor target (50% expert reduction on a 1T model, generative/agentic use, no retraining budget), the evidence favours REAP-style dropping over merging. The only regime where merging has a demonstrated edge is 25% reduction or MC-style evaluation, and REAM's edge at 25% comes mostly from using REAP saliency (its ablation).
- REAM's design (singleton centroids for most experts, only a few absorbed) is effectively "REAP plus a small merge correction"; it is the most promising merge-flavoured hybrid, but it has not been run at 50% on >30B or at any ratio on >106B.
- With large continued-pretraining budgets (SlimQwen, 400B tokens), prune-vs-merge differences wash out — irrelevant on a single-machine budget.

### Gaps
- No head-to-head of any merge method against REAP on 480B–1T models exists.
- No paper measures merging at 50% on the REAM GEN suite for models above 30B.

## Key Question 3: Low-rank factorisation — rank fraction, byte reduction, interaction with 4-bit

### Takeaway
Plain per-expert SVD is close to useless on frontier MoEs because expert matrices are near full effective rank (Kimi-K2's average effective rank exceeds the 1593 break-even rank in most layers); shared-basis schemes (MoBE/RFID) get 24–30% at ~1–2 point cost, and 60% only on ≤30B with multi-PPL-point losses. No 2025–26 paper tests MoBE/RFID stacked on 4-bit weights; the methods that do combine low-rank with quantisation (MiLo, TileQ, KBVQ) use low-rank as an error compensator on top of 3-bit, not as a byte reducer.

### Cited Findings
- MoBE Appendix B: effective rank r_e = smallest k whose top-k singular values capture >95% energy; Kimi-K2/DeepSeek-V3 expert matrices are 7168×2048 so rank decomposition only compresses if rank ≤ ~1593 (7168·2048/(7168+2048)); for Kimi-K2 "the average effective rank is larger than 1593 in most layers", so pure rank decomposition "can't produce model compression without performance loss" — [arXiv 2508.05257v1](https://arxiv.org/html/2508.05257v1)
- MoBE Section 2.2: effective rank "consistently exceeds the compression threshold of SVD", which the authors say explains SVD-based methods' drops; MoBE's reconstruction MSE is often >50% lower than MoLAE/D2-MoE — [arXiv 2508.05257v1](https://arxiv.org/html/2508.05257v1)
- D2-MoE energy retained at 20/40/60% compression: 68.05% / 47.34% / 26.62% (i.e. at 40% compression the delta-SVD keeps under half the singular energy); Mixtral avg 0.63 → 0.57 at 40%, 0.52 at 60% — [arXiv 2502.17298v1](https://arxiv.org/html/2502.17298v1)
- RFID-MoE 60% on Qwen3-30B: WikiText-2 PPL 9.57 (MoBE 13.96); avg accuracy 0.64 vs 0.61 — i.e. 60% factorisation still costs several PPL points on a 30B model — [arXiv 2602.09316v2](https://arxiv.org/html/2602.09316v2)
- MoBE at 24–30% on 235B–1T costs 0.6–1.3 points average; rank r = p (no rank reduction), compression comes only from sharing the basis across experts — [arXiv 2508.05257v1](https://arxiv.org/html/2508.05257v1)
- Quantisation vs pruning (REAP App. E): weight quantisation gives better accuracy than expert pruning down to 4 bits; Qwen3-30B Eval+: W16 128 experts 81.4 (100% size); W16 64 experts 78.0 (50%); W4A16-G128 128 experts 80.5 (25%); W4A16 + 64 experts 77.6 (12.5%); W2A16 + 64 experts 28.6 (12.5%). Kimi-K2: 4-bit + 50% pruning = 87.5% total size reduction, "matching that footprint with quantization alone would need 2-bit weights" — [arXiv 2510.13999v2](https://arxiv.org/html/2510.13999v2)
- REAP lists quantization, low-rank decomposition and weight sparsity as orthogonal to expert pruning/merging but reports no low-rank experiments — [arXiv 2510.13999v2](https://arxiv.org/html/2510.13999v2)
- MiLo (2025): INT3 weights + low-rank compensators (also INT3) computed from the quantisation residual E = W − W_dq via SVD on half-precision weights; Mixtral s1 memory 20.8GB vs 20.5GB HQQ (+1.5%); WikiText PPL 4.61 → 4.03 (FP16 3.42); zero-shot avg 75.59 → 79.56; DeepSeek-MoE MMLU 35.63 → 41.92; training-free, ~20 iterations; custom INT3×FP16 kernel 1.2× over MARLIN; Ampere required; code github.com/Supercomputing-System-AI-Lab/MiLo — [arXiv 2504.02658v1](https://arxiv.org/html/2504.02658v1)
- TileQ (May 2026): fine-tuning-free PTQ sharing low-rank factors across both input and output dims of experts, "up to 10× reduction in additional memory", ~5% latency overhead; models Qwen1.5-MoE, Qwen3-30B-A3B, Mixtral-8x7B, DeepSeek-MoE-16B; bit widths not in abstract — [arXiv 2605.09281](https://arxiv.org/abs/2605.09281)
- KBVQ-MoE (Feb 2026): KLT-guided SVD + bias-corrected vector quantisation; 3-bit Qwen1.5-MoE-A2.7B avg 67.99 vs FP16 68.07 — [search summary of arXiv 2602.11184](https://arxiv.org/pdf/2602.11184) (full text not readable; unverified)
- EAC-MoE (2025): quantisation changes which experts the router picks ("expert-shift"); fix = router calibration + pruning rarely selected experts; Mixtral 93.41GB → 18.98GB (4.92×) at ~3-bit with <1.25% avg loss — [alphaxiv overview 2508.01625](https://www.alphaxiv.org/overview/2508.01625)
- Collaborative Compression (Sep 2025): DeepSeek-V3 671B compressed with expert pruning (retain 75%, ~508B) + top-k 6 + llama.cpp IQ1_M mixed precision → 103–130GB (paper inconsistent), fits a 128GB Strix Halo at >5 tok/s; MMLU 84.55 vs 85.57 for uniform Q2_K_L at 230GB; no low-rank, no merging; compression itself ran on 8×H200 / 8×H100 — [arXiv 2509.25689](https://arxiv.org/html/2509.25689)

### Inferences
- Byte-wise, a shared-basis factorisation at 24% on top of 4-bit is a ~24% saving IF the basis and A-matrices quantise as well as the originals — untested. Since MoBE's A^i matrices are new learned tensors with different statistics, 4-bit quantisation of them is an open risk.
- Factorised experts change the inference kernel (extra GEMM through the shared basis, activation f between factors in MoBE). On Thor's custom CUDA server this means new kernels and more activation traffic per expert, which is the binding term already (per memory: activation traffic dominates decode). MoBE's own authors say they need a mega-kernel and did not measure speed.
- Low-rank-as-compensator (MiLo/TileQ) is the one low-rank idea that clearly composes with sub-4-bit quantisation: it buys back accuracy for +1.5–9% bytes rather than removing bytes. It could matter if the Thor plan ever needs 3-bit experts.

### Gaps
- No paper stacks MoBE/RFID/D2-MoE on 4-bit quantised experts or reports post-quantisation accuracy of factorised experts.
- No paper combines expert pruning at 50% with factorisation of survivors on >47B models (MoE-I2 is the only combination and only on ≤47B, with weak numbers).

## Key Question 4: Experts already quantised at source (FP4/MXFP4) — merge/factorise without BF16 dequant

### Takeaway
No method found operates directly on quantised expert weights; every merge/factorisation method averages or decomposes BF16/FP16 tensors, and REAP explicitly notes merging must reconcile block-shared quantisation scales (i.e. dequantise, merge, re-quantise) whereas pruning needs no reconciliation. For pre-quantised experts the only demonstrated path is the one Thor already uses: prune on the quantised checkpoint.

### Cited Findings
- REAP App. E: pruning was applied directly to Kimi-K2-Instruct-W4A16 (RedHatAI INT4 release) and Qwen3-Coder-480B FP8 "with no adjustment", whereas for merging "the fine-grained shared quantization scales across parameter groups must be reconciled during merging" — [arXiv 2510.13999v2](https://arxiv.org/html/2510.13999v2)
- gpt-oss MXFP4 experts: packed uint8 blocks + uint8 E8M0 scales; weights do not support backward; dequantising on load (Mxfp4Config(dequantize=True)) raises memory from ~14GB to ~42GB for gpt-oss-20b — [TowardsAI practitioner guide](https://pub.towardsai.net/fine-tuning-openais-gpt-oss-20b-a-practitioner-s-guide-to-lora-on-moe-models-920171bf5258)
- NVIDIA ModelOpt has a "gpt_oss_pruned_to_mxfp4" path that dequantises MXFP4 expert weights from the original gpt-oss-120b, matches experts by weight comparison, and re-packs MXFP4 — i.e. a dequant → operate → repack pipeline, for pruning not merging — [ModelOpt docs](https://nvidia.github.io/Model-Optimizer/pr-preview/pr-1190/reference/generated/modelopt.torch.puzzletron.anymodel.models.gpt_oss.gpt_oss_pruned_to_mxfp4.html)
- Megatron Bridge notes dequantised-MXFP4 checkpoints have a different expert weight orientation from BF16 checkpoints and the loader must detect/align it — [NVIDIA Megatron Bridge docs](https://docs.nvidia.com/nemo/megatron-bridge/0.4.1/apidocs/bridge/bridge.models.gpt_oss.gpt_oss_bridge.html)
- E-PMQ (May 2026) studies quantising AFTER merging (Task Arithmetic/TIES merges of task models, then 4-bit GPTQ) and shows merge error and quantisation error stack; it is about merging full task models, not MoE experts, and does not address merging already-quantised weights — [arXiv 2605.16882](https://arxiv.org/abs/2605.16882)
- MiLo computes its compensators from the BF16 → INT3 residual, so it requires the BF16 weights — [arXiv 2504.02658v1](https://arxiv.org/html/2504.02658v1)
- MoBE fits A^i and B^j to BF16 originals by reconstruction MSE; README lists BF16 models as supported — [github.com/inclusionAI/MoBE](https://github.com/inclusionAI/MoBE)

### Inferences
- Merging two MXFP4 experts in-format is mathematically ill-defined because each 32-element block has its own E8M0 scale; averaging requires dequantising at least the block being merged. Dequantisation can be done per expert (a few MB) rather than per model, so memory is not the blocker — the blocker is the re-quantisation error on top of merge error (E-PMQ's point) and that FP4 expert weights carry 2× less information than the BF16 experts all merge papers were tuned on.
- Factorising pre-FP4 experts (e.g. V4.1-Flash, MiMo-V2.6) would mean fitting a basis to already-lossy weights then re-quantising the factors: two quantisation passes. No evidence this works; treat as untested.

### Gaps
- No paper merges or factorises experts stored in FP4/MXFP4/NVFP4; no paper measures re-quantisation error of merged experts.

## Key Question 5: Compute/memory — can these run layer-streaming on a single 122 GiB machine?

### Takeaway
Calibration-based merging (HC-SMoE, REAM) is layer-local and cheap (minutes to ~1.5 h at ~30 GB VRAM for 30B), and could in principle be streamed, but REAM's sequential variant re-runs a forward through each merged layer; the factorisation methods that work at 1T (MoBE) are gradient-based fits over all 384 experts of a layer at once (Kimi-K2 had to be split into two groups even on H100s), and D2-MoE was infeasible on 8×H100 at 235B+. None of the papers report a CPU-offload/streaming recipe.

### Cited Findings
- HC-SMoE: 32 × 2048-token calibration; Mixtral merge ~111 s on 8×V100; Qwen 290–323 s on 4×V100; DeepSeek-MoE-16B under 10 min on one A100 — [arXiv 2410.08589v4](https://arxiv.org/html/2410.08589v4)
- REAM: Qwen3-30B-A3B ~1 h non-sequential, ~1.5 h sequential, ~30 GB VRAM; sequential variant recomputes activations after each merged layer; calibration 3,072 × 512 tokens; no compute details for the 80–106B models — [arXiv 2604.04356v1](https://arxiv.org/html/2604.04356v1)
- D2-MoE on Mixtral-8x7B: 131 GB memory for Fisher merging, 109 GB for delta SVD; MoBE says D2-MoE is infeasible on 8×H100 for Qwen3-235B/DeepSeek-V3/Kimi-K2 because of backward-pass cost — [arXiv 2502.17298v1](https://arxiv.org/html/2502.17298v1); [arXiv 2508.05257v1](https://arxiv.org/html/2508.05257v1)
- MoBE: Algorithm 1 processes one MoE layer at a time; Kimi-K2's 384 experts per layer split into two sequential groups of 64 basis matrices because joint optimisation of all 384 was difficult; README training takes a layer range and is run separately per matrix type (gate_proj / up_proj); 10,000 epochs (DeepSeek-V3) / 15,000 (Kimi-K2); GPU count, memory and wall-clock not reported; inference example uses 8 GPUs × 120 GiB plus CPU offload — [arXiv 2508.05257v1](https://arxiv.org/html/2508.05257v1); [github.com/inclusionAI/MoBE](https://github.com/inclusionAI/MoBE)
- RFID-MoE: 8×A100; memory/compute not measured — [arXiv 2602.09316v2](https://arxiv.org/html/2602.09316v2)
- EEP: evolutionary search over expert-merging matrices with inference-only fitness; profiling on 2×A100; search cost described as "potentially costly" and not reported — [arXiv 2407.00945v1](https://arxiv.org/html/2407.00945v1)
- Collaborative Compression (DeepSeek-V3 → 103–130 GB): expert pruning on 8×H200, imatrix calibration on 8×H100 NVL — [arXiv 2509.25689](https://arxiv.org/html/2509.25689)
- REAP: hardware for pruning 480B/1T models is not stated in the sections retrieved — [arXiv 2510.13999v2](https://arxiv.org/html/2510.13999v2)

### Inferences
- Memory-wise, every merge/factorise method is layer-local in its weight access (an MoE layer of Kimi-K2 is 384 × 3 × 7168×2048 BF16 ≈ 33.8 GB, ~17 GB at 4-bit), so a streaming implementation is feasible on 122 GiB in principle; the published code does not do it, so it would be custom work.
- Activation statistics for merging (HC-SMoE output similarity, REAM gated outputs) require a forward pass over the calibration set — the same pass Thor's REAP pipeline already does, so merging adds little compute beyond the clustering. MoBE-style fitting is the expensive one: thousands of Adam epochs per layer over all experts; at 61 layers for Kimi-K2 this is a multi-day job even on H100s (extrapolation — the paper gives no wall-clock).

### Gaps
- No paper reports a single-GPU or CPU-offload recipe for any method at >100B.
- MoBE and RFID GPU-hours are unreported.

## Key Question 6: Working open-source implementations

### Takeaway
Merging: MC-SMoE, HC-SMoE, EEP, REAM (plus HF checkpoints) have public code; factorisation: MoBE (SGLang path, no checkpoints, low activity), D2-MoE, MoE-SVD, MoE-I2 (promised), Sub-MoE (promised), MiLo (with INT3 kernel). RFID-MoE, SEER-MoE, TileQ, KBVQ-MoE and EAC-MoE have no code link found.

### Cited Findings
- MC-SMoE: [github.com/UNITES-Lab/MC-SMoE](https://github.com/UNITES-Lab/MC-SMoE/blob/main/README.md)
- HC-SMoE: github.com/wazenmai/HC-SMoE — [arXiv 2410.08589](https://www.arxiv.org/abs/2410.08589v1)
- EEP: github.com/imagination-research/EEP — [arXiv 2407.00945v1](https://arxiv.org/html/2407.00945v1)
- REAM: github.com/SamsungSAILMontreal/ream; HF collection SamsungSAILMontreal/ream — [arXiv 2604.04356v1](https://arxiv.org/html/2604.04356v1)
- MoBE: [github.com/inclusionAI/MoBE](https://github.com/inclusionAI/MoBE) — supports Ling, Qwen3-MoE, DeepSeek-V3, Kimi-K2 BF16; SGLang inference with fused-MoE and mega-kernel listed, optimised kernels "upcoming"; reconstructed standard-MoE outputs compatible with HF/vLLM/SGLang; no released checkpoints; 40 stars, 7 commits, 2 open issues
- D2-MoE: github.com/lliai/D2MoE — [arXiv 2502.17298v1](https://arxiv.org/html/2502.17298v1)
- MoE-SVD: github.com/lliai/MoE-SVD — [ICML 2025 poster](https://icml.cc/virtual/2025/poster/44786)
- MoE-I2: github.com/xiaochengsky/MoEI-2 ("will be available") — [arXiv 2411.01016](https://arxiv.org/abs/2411.01016)
- Sub-MoE: github.com/lliai/MoERazor ("will be released") — [arXiv 2506.23266v1](https://arxiv.org/abs/2506.23266v1)
- MiLo: github.com/Supercomputing-System-AI-Lab/MiLo (INT3×FP16 Tensor Core kernel, Ampere) — [arXiv 2504.02658v1](https://arxiv.org/html/2504.02658v1)
- RFID-MoE: no repository link in the paper — [arXiv 2602.09316v2](https://arxiv.org/html/2602.09316v2)
- EAC-MoE: no code link found — [alphaxiv 2508.01625](https://www.alphaxiv.org/overview/2508.01625)
- A unified MoE compression toolkit (pruning + merging + quantisation + decomposition) exists: github.com/daizedong/unified-moe-compression — [search result](https://github.com/daizedong/unified-moe-compression) (not inspected; unverified scope)

### Inferences
- REAM is the only merge codebase that already targets 2025-generation architectures (Qwen3-Next, GLM-4.5-Air) and ships checkpoints; it is the natural thing to try if a merge step is ever added after REAP (it reuses REAP saliency and the same calibration pass).
- MoBE is the only factorisation code claiming Kimi-K2/DeepSeek-V3 support, but it is low-activity, has no checkpoints, requires its own model classes for native inference, and its kernel story is unfinished — a high-integration-cost path for Thor's custom CUDA server.

### Gaps
- I did not verify that any of the GitHub repositories currently build or run; stars/activity only for MoBE.
- Whether github.com/daizedong/unified-moe-compression implements merging/decomposition for 2025 architectures is unknown.

## Summary table (self-reported numbers; largest scale demonstrated)

| Method | Type | Largest model | Reduction | Quality cost | Training | Code |
|---|---|---|---|---|---|---|
| MC-SMoE (2024) | merge + low-rank/sparse | 15B (fairseq-moe) | ~80% params (Switch 2.0B→381M) | −1 to −2.4 pts fine-tuned; large zero-shot drops | KD fine-tune (Switch) | yes |
| HC-SMoE (2025) | merge | Mixtral 8x7B (47B) | 50% experts | Mixtral avg 0.675→0.573; WildBench collapse 0.008 on Qwen3-30B at 50% (REAP) | none | yes |
| EEP (2024) | prune + search-merge | Mixtral 8x22B (141B) | 50–75% experts | gains on IID tasks, MMLU OOD −8 | inference-only search | yes |
| SEER-MoE (2024) | prune + reg. fine-tune | Mixtral 8x7B | ~25% params | "slight" | fine-tune | not found |
| REAM (2026) | prune-centred merge | GLM-4.5-Air (106B) | 25% (50% on 30B) | 25%: beats REAP by 0.8–2 GEN pts; 50%: 60.7 vs REAP 62.8 | none | yes + HF |
| D2-MoE (2025) | base+delta SVD | Qwen2-57B | 40–60% | Mixtral 0.63→0.57 (40%), 0.52 (60%) | none (calib) | yes |
| MoE-SVD (2025) | shared-V SVD | Qwen2 MoE | 60% | "minimal" (no numbers retrieved) | none | yes |
| MoLAE (2025) | shared projection | Kimi-K2 (as baseline) | 24–30% | Kimi-K2 82.4→76.6 | fit | n/a |
| MoBE (ICLR 2026) | shared basis | Kimi-K2 1T | 24–30% params | −0.6 to −1.3 pts (235B–1T) | Adam basis fit, 10–15k epochs | yes (no ckpts) |
| RFID-MoE (2026) | rank-allocated shared SVD | Qwen3-235B (figure) | 20–60% | Qwen3-30B 60%: PPL 9.57 vs orig; 0.64 vs 0.72 acc | Adam fit, no data | none |
| MoE-I2 (2024) | prune + low-rank | Mixtral 8x7B | n/a | worse than D2-MoE | n/a | promised |
| MiLo (2025) | INT3 + low-rank compensator | Mixtral 8x7B | +1.5–9% bytes over INT3 | recovers ~4 pts zero-shot | none | yes + kernel |
| REAP (ICLR 2026, reference) | prune | Kimi-K2 1T W4A16 | 50% experts + 4-bit = 87.5% | Kimi-K2 code 0.631→0.624 | none | yes |
