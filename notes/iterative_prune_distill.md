# Iterative / progressive prune-then-distill compression recipes (industrial practice, 2024–2026) and what one Jetson AGX Thor can replicate

Scope: Minitron/Nemotron-family structured pruning + KD, MoE-specific staged recipes (SlimMoE, SlimQwen, Puzzle-75B-A9B, Nemotron Elastic), expert-pruning one-shot-vs-iterative evidence, offline/top-K teacher logits, data budgets, quality ceilings, and a Thor compute scaling. Every number is cited; values the source does not give are marked UNREPORTED. Thor extrapolations are marked **[THOR EXTRAPOLATION]**.

## Cross-recipe summary table (all values from the cited primary papers; see per-question sections for detail)

| Recipe (year) | Start → end params | Stages | What is pruned | KD objective | Distillation tokens | Compute | Quality vs teacher / vs scratch |
|---|---|---|---|---|---|---|---|
| Minitron (Nemotron-4 15B→8B→4B, 2024) [1] | 15.6B → 8.3B → 4.2B | 1 per size (chained: 15→8→4) | width: hidden 6144→4096→3072, MLP 24576→16384→9216, heads 48→48→24; depth kept 32 | logit KL only (τ=1); +intermediate losses only if depth cut | ~94B tokens per model (+1.8B lightweight retrain for candidate ranking) | 128×A100-80GB; GPU-h UNREPORTED | 8B: MMLU 63.8 vs teacher 66.6; 4B pruned-from-8B 42.45 MMLU @100B vs scratch 26.24 @400B |
| Minitron-in-practice (Mistral-NeMo 12B→8B; Llama-3.1 8B→4B, 2024) [2] | 12.4B→8.4B; 8B→4.5B | 1 (single-shot) | width (hidden 5120→4096, MLP 14336→11520) or depth (32→16 layers) | forward-KL on logits only, no CE | 380B (MN-8B), 94B (Llama-4B), +~100B teacher-correction | 32×DGX H100; GPU-h UNREPORTED | MN-8B MMLU 69.5 vs teacher-FT 70.1; Llama-4B-width 60.5 vs 65.3 |
| Nemotron-H 56B→47B MiniPuzzle (2025) [3] | 56B → 47B | 1 (NAS + 7B-token screen of 3 candidates, then 63B final) | layers (attn 10→5, Mamba 54→44, FFN 54→49) + FFN 32768→30720 | forward-KL logits, FP8, seq 8192, batch 768 | 7B (screen) + 63B (final) | 6144 H100 for pretrain; compression GPU-h UNREPORTED | MMLU 83.6 vs 84.2, GSM8K 93.3 vs 93.7 ("near-lossless"); ~300× fewer tokens than scratch |
| Nemotron Elastic 12B→9B/6B (2025) [4] | 12B → 9B + 6B nested | 1 run, 2 phases | depth, embed, FFN, Mamba heads/channels, attn heads (router-chosen) | forward-KL on temperature-softened logits from frozen 12B | 65B (8K ctx) + 45B (49K ctx) = 110B | GPU-h UNREPORTED | 9B avg 75.95 vs NanoV2-9B 75.99; Minitron-SSM alternative est. 750B tokens |
| **Puzzle-75B-A9B from Nemotron-3-Super (MoE, 2026)** [5] | 120.7B-A12.8B → 75.3B-A9.3B | **3 iterative rounds** | expert intermediate 2688→1280–2688 (avg 59.9%), top-k 22→4–18 (avg 50%), SSM state 128→96; expert COUNT unchanged; no layers removed | logit-matching KD (no hidden/router loss stated) | 24B + 43.2B + 52.8B = 120B heal + up to 100B/seq-len long-context KD | GPU-h UNREPORTED | avg 69.05 (3-step) vs 68.48 (1-step); MMLU-Pro 82.4 vs parent 83.8 |
| **SlimMoE Phi-3.5-MoE 42B→7.6B/3.8B (MoE, 2025)** [6] | 41.9B-A6.6B → 7.6B-A2.4B (2 stages), 3.8B-A1.1B (3 stages) | 2–3, geometric α^(1/T) | expert intermediate 6400→960 (15%) / 448 (7%); 16 experts KEPT; GQA groups 32/8→16/4 (tiny) | KL on teacher **top-8** probabilities + load-balance loss | 400B total; intermediate stages 30–35% of steps | 64×A100; GPU-h UNREPORTED | multi-stage MMLU 69.87 vs one-stage 68.58 (mini); 60.08 vs 58.40 (tiny); teacher MMLU 78.36 |
| **SlimQwen Qwen3-Next-80B-A3B→23B-A2B (MoE, 2026)** [7] | 80B-A3.8B → 23B-A2.0B | 2 (depth-first: 40B + 360B) | 12 of 48 blocks, hidden 2048→1536, experts 512→256, top-k 10→8 | (1−λ)LM + λKD + β·MTP-KD, λ 1.0→0.75 | 400B (ablations at 120B) | Alibaba Cloud Megatron; GPU-h UNREPORTED | progressive MMLU 77.39 vs one-stage 75.86; pruned+KD 73.45 avg vs random-init+KD 61.66 @120B; teacher avg 82.68 |
| MoE→dense (Qwen3-30B-A3B etc., 2026) [8] | 30B-A3B MoE → dense (size UNREPORTED in abstract) | 1 | experts scored/grouped/concatenated into dense FFN | KD from MoE teacher | ~4B | 1.6× faster wall-clock than dense-to-dense | +6.3 pp avg vs dense-to-dense pruning at matched params |
| MoE-Pruner Mixtral-8x7B (2024) [9] | 46.7B → 50% unstructured | 1 | weights inside experts (unstructured / 2:4) | expert-wise KD from unpruned model | 1000 C4 samples, 3 epochs, ~1 h on 16×H100 | ~16 GPU-h (inferred) | avg 68.40 vs 69.16 (98.9%) at 50% |
| Fantastic Experts "MoE Lottery" Mixtral-8x7B (2025) [10] | 8 experts/layer → 12.5–75% dropped | k rounds of 12.5% (1 expert/layer/round) | whole experts | LM-loss fine-tune on C4 between rounds (no teacher) | ~1M tokens/round suffices (0.25–2.27M tested) | UNREPORTED | C4 ppl @50%: one-shot 14.74 / iterative 10.44 / lottery 9.76 (orig 7.44) |
| REAP Qwen3-Coder-480B, Kimi-K2 (2025) [11] | 50% of experts | 1, no retraining | whole experts | none | 0 | calibration only | ≤~2% drop on code at 50% (secondary summary) |
| Llama 3.2 1B/3B (Meta, 2024) [12] | 8B → 1B/3B | 1 ("single shot") | UNREPORTED axes | logits of 8B+70B as token-level targets during pretraining | UNREPORTED | UNREPORTED | UNREPORTED numerically |
| Apple AFM-on-device (2024) [13] | 6.4B → ~3B | 1 (Soft-Top-K mask learned over 188B tokens) | FFN hidden dim only | 0.9 teacher-label / 0.1 true-label convex combination | 188B mask-learning; main pretrain 6.3T | UNREPORTED | pruned init +0–2%; distillation +~5% MMLU, +~3% GSM8K |

[1] https://arxiv.org/abs/2407.14679 · [2] https://arxiv.org/abs/2408.11796 · [3] https://arxiv.org/abs/2504.03624 · [4] https://arxiv.org/abs/2511.16664 · [5] https://arxiv.org/abs/2607.04371 · [6] https://arxiv.org/abs/2506.18349 · [7] https://arxiv.org/abs/2605.08738 · [8] https://arxiv.org/abs/2605.28207 · [9] https://arxiv.org/abs/2410.12013 · [10] https://arxiv.org/abs/2504.05586 · [11] https://arxiv.org/abs/2510.13999 · [12] https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/ · [13] https://arxiv.org/abs/2407.21075

## KQ1. Minitron (Nemotron-4 → 8B/4B, Mistral-NeMo-Minitron, Llama-3.1-Minitron, Nemotron-H 47B): exact recipe, tokens, iterative vs one-shot

### Takeaway
Minitron's recovery budget is ~94B tokens per model (confirmed: 94B for Nemotron-4 15B→8B and →4B, 94B for Llama-3.1-8B→4B, 380B for Mistral-NeMo-12B→8B, 63B+7B for Nemotron-H 56B→47B), always logit-only forward-KL; Minitron's "iterative" finding is nuanced: iterative *importance estimation* and iterative *depth* pruning gave no benefit (one-shot preferred), but chaining size steps (15B→8B→4B, pruning from the model closest to target) beat one-shot 15B→4B by ~12% MMLU at equal tokens.

### Cited Findings
- Nemotron-4 15B (15.6B) was width-pruned to Minitron 8B (8.3B: hidden 6144→4096, MLP 24576→16384, heads 48 kept, 32 layers kept) and 4B (4.2B: hidden 3072, MLP 9216, heads 24); overall parameter reduction ~47% (8B) and ~73% (4B) — [Minitron, arXiv 2407.14679](https://arxiv.org/abs/2407.14679)
- Final retraining used ~94B tokens for each of the 8B and 4B models; lightweight retraining for candidate ranking used 1.8B tokens (400 steps); importance estimation used 1024 calibration samples; hardware 16 DGX A100 nodes (128× A100-80GB); GPU-hours UNREPORTED — [Minitron](https://arxiv.org/abs/2407.14679)
- Up to 40× fewer tokens per model than from-scratch (Nemotron-3 8B used 3.8T tokens vs 94B), 1.8× compute saving for the whole family; up to 16% MMLU improvement over training from scratch — [Minitron](https://arxiv.org/abs/2407.14679)
- Best loss: logit-only KL (τ=1.0) with no CE term when depth is not significantly reduced; intermediate-state/embedding losses (cosine best) help only when depth is reduced significantly; reverse-KL worse (LM loss 2.665 vs 2.155); **top-K teacher-logit filtering with K≤100 hurt accuracy** — [Minitron](https://arxiv.org/abs/2407.14679)
- Iterative vs one-shot, three distinct findings: (a) single-shot importance estimation is recommended because iterative re-estimation gave no benefit after retraining (all candidates converged to 1.92 val loss); (b) for depth pruning, one-shot prune-and-distill beat iterative (30.2B vs 86.4B tokens); (c) across sizes, 15B→8B→4B gave MMLU 42.45 vs 37.81 for one-shot 15B→4B at 100B tokens (~12% better) — "prune from the model closest to the target size" — [Minitron](https://arxiv.org/abs/2407.14679)
- Retraining-token ablation for 8B: 18.9B tokens → MMLU 58.3 / HellaSwag 73.8; 94B tokens → 62.8 / 79.7 — [Minitron](https://arxiv.org/abs/2407.14679)
- Best-practice list (10 items) includes: train largest then prune+distill iteratively; width over depth at ≤15B; retrain with KLD not CE; prune from the final-stage checkpoint if multi-phase — [Minitron](https://arxiv.org/abs/2407.14679)
- Minitron-in-practice: Mistral-NeMo 12B → MN-Minitron-8B (hidden 5120→4096, MLP 14336→11520, 40 layers kept) with 380B distillation tokens; Llama-3.1-8B → 4B width (hidden 3072, MLP 9216) and 4B depth (32→16 layers) with 94B tokens each; teacher correction ~100B tokens; single-shot pruning, no NAS; forward-KL on logits only, CE ignored; 32 DGX H100 nodes; GPU-hours UNREPORTED — [arXiv 2408.11796](https://arxiv.org/abs/2408.11796)
- Quality: MN-Minitron-8B MMLU 69.5 (teacher-FT 70.1), GSM8K 58.5 (55.7), HumanEval 36.2 (23.8); Llama-3.1-Minitron-4B-Width MMLU 60.5 / GSM8K 41.2 vs Llama-3.1-8B 65.3 / 48.6; depth variant GSM8K collapsed to 16.8 — [arXiv 2408.11796](https://arxiv.org/abs/2408.11796)
- Teacher correction (fine-tuning the teacher on the distillation corpus) reduced LM validation loss by >6% — [arXiv 2408.11796](https://arxiv.org/abs/2408.11796)
- Nemotron-H 56B → 47B (MiniPuzzle): pruned layers (attn 10→5, Mamba-2 54→44, FFN 54→49) and FFN 32768→30720; Mamba-head pruning "caused severe accuracy degradation" and was excluded; ~400 NAS candidates → 130 → 3, each screened with ~7B-token distillation, winner distilled for 63B tokens; forward-KL logits, FP8, seq 8192, batch 768; ~300× fewer tokens than from scratch; result MMLU 83.6 vs 84.2, GSM8K 93.3 vs 93.7, MMLU-Pro 61.8 vs 60.5 — [Nemotron-H, arXiv 2504.03624](https://arxiv.org/abs/2504.03624)
- Short-distillation effect at 47B: candidate averages rose from 63.3/62.1/53.7 → 69.3/69.6/69.5 after ~7B tokens (teacher 70.6) — [Nemotron-H](https://arxiv.org/abs/2504.03624)
- Nemotron Elastic (Nano V2 12B → nested 9B/6B) used 110B tokens total (65B at 8K ctx + 45B at 49K ctx) with forward-KL from a frozen 12B teacher, and estimates the Minitron-SSM alternative at 750B tokens (480B exploratory + 270B final) — [Nemotron Elastic, arXiv 2511.16664](https://arxiv.org/abs/2511.16664)
- Llama 3.2 1B/3B: "structured pruning in a single shot manner from the Llama 3.1 8B"; logits from 8B and 70B used as token-level targets in the pre-training stage; token counts UNREPORTED — [Meta blog](https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/)
- Apple AFM-on-device: initialized from a pruned 6.4B model, pruning only the FFN hidden dimension with Soft-Top-K masks learned over 188B tokens, then trained with 0.9-weight teacher labels; pruned init improves data efficiency and results by 0–2%, distillation adds ~5% MMLU / ~3% GSM8K — [Apple, arXiv 2407.21075](https://arxiv.org/abs/2407.21075)

### Inferences
- "94B tokens" is the recurring Minitron budget for ~50% dense reductions; the 380B MN-Minitron run shows NVIDIA scales tokens up when the target is an outright-competitive release, not a minimum.
- Minitron's "iterative" result is really "stage by size with full recovery between stages", not "many small prune steps with light recovery"; and the explicit test of small-step iteration within one axis (depth) lost to one-shot.

### Gaps
- GPU-hours are UNREPORTED in every NVIDIA Minitron/Nemotron-H/Elastic paper; only node counts are given.
- Meta's Llama 3.2 pruning axes and token budgets are not published anywhere I found.

## KQ2. Minitron-style results on MoE models (Nemotron-3 Super/Nano, DeepSeek-V3, Qwen3-MoE, Kimi-K2)

### Takeaway
The only NVIDIA MoE compression with a full recipe is Puzzle-75B-A9B (Nemotron-3-Super 120.7B-A12.8B → 75.3B-A9.3B, three iterative rounds, 120B heal tokens, expert count unchanged — it slims experts and cuts top-k); the other published MoE pipelines are SlimMoE (Phi-3.5-MoE 42B→7.6B/3.8B, 400B tokens) and SlimQwen (Qwen3-Next-80B→23B, 400B tokens). No DeepSeek-V3 or Kimi-K2 prune+distill recipe with numbers exists; Kimi-K2/Qwen3-Coder-480B appear only in zero-retraining REAP, and Nemotron-3 Nano 30B-A3B was trained from scratch, not pruned.

### Cited Findings
- Puzzle-75B-A9B: parent 120.7B/12.8B active → 75.3B/9.3B active (62.4% / 73.1%); routed-expert intermediate size 2688 → 1280–2688 per layer (avg 59.9%); top-k 22 → 4–18 per layer (avg 50%); active routed capacity avg 30.9% of parent; SSM state 128→96; number of experts unchanged; no layers removed (88 blocks: 40 Mamba, 40 MoE, 8 attention) — [arXiv 2607.04371](https://arxiv.org/abs/2607.04371)
- Iterative Puzzle = 3 compress-and-heal rounds with replacement scores recomputed each round: Stage 1 MoE capacity 75% + SSM 75%, healed 24B tokens; Stage 2 MoE capacity 60%, healed 43.2B; Stage 3 routed-expert budget 50% heterogeneously allocated, healed 52.8B; then long-context KD extended 128K→512K with up to 100B tokens per sequence length and 16M-token global batch — [arXiv 2607.04371](https://arxiv.org/abs/2607.04371)
- 3-step vs 1-step Puzzle at the same final budget: average 69.05 vs 68.48 (+0.57), MMLU-Pro +0.7, GPQA +1.5, AA-LCR +2.2, LiveCodeBench +0.7; one-shot baseline token count UNREPORTED — [arXiv 2607.04371](https://arxiv.org/abs/2607.04371)
- KD objective: logit matching to the Nemotron-3-Super teacher; data 30% pretraining / 70% Nemotron-3-Nano SFT data; hardware and GPU-hours UNREPORTED — [arXiv 2607.04371](https://arxiv.org/abs/2607.04371)
- Final quality (BF16, Puzzle vs Super): MMLU-Pro 82.4/83.8, GPQA 78.6/80.5, AIME25 89.7/92.2, LiveCodeBench 81.1/82.1, SWE-Bench 56.9/59.5, RULER-1M 92.2/93.9; ~2× server throughput at matched user throughput on 8×B200 — [arXiv 2607.04371](https://arxiv.org/abs/2607.04371)
- SlimMoE: Phi-3.5-MoE (41.9B/6.6B active, 16 experts top-2, expert dim 6400) → Phi-mini-MoE 7.6B/2.4B (expert dim 960, 2 stages) and Phi-tiny-MoE 3.8B/1.1B (expert dim 448, GQA 32/8→16/4, 3 stages); experts are slimmed (neurons pruned uniformly across experts), expert count unchanged — [SlimMoE, arXiv 2506.18349](https://arxiv.org/abs/2506.18349)
- SlimMoE budget: 400B tokens total (<10% of ~4T original), 64×A100, intermediate stages use 30–35% of steps with early stopping; KD loss = KL over the teacher's top-8 next-token probabilities + load-balancing loss; pruning criterion |grad×weight| of the top-8 KD loss on 16K calibration examples — [SlimMoE](https://arxiv.org/abs/2506.18349)
- SlimMoE criterion ablation at 50% (metric unstated in text): expert slimming+KL 63.76, slimming+CLM 59.30, **expert (whole) pruning+KL 53.41**, expert pruning by frequency 48.59, M-SMoE merging 38.54 — [SlimMoE](https://arxiv.org/abs/2506.18349)
- SlimQwen: Qwen3-Next-80B-A3.8B → 23B-A2.0B by removing 12 blocks (3 full-attn, 9 linear-attn), hidden 2048→1536, 512→256 experts per layer, top-k 10→8, 1 shared expert kept; 400B tokens on Megatron-LM / Alibaba Cloud; GPU-hours UNREPORTED — [SlimQwen, arXiv 2605.08738](https://arxiv.org/abs/2605.08738)
- SlimQwen finds different one-shot expert-compression methods reach similar final performance after large-scale continual pretraining, and proposes partial-preservation expert merging — [SlimQwen](https://arxiv.org/abs/2605.08738)
- MoE→dense: Qwen3-30B-A3B experts scored/grouped/concatenated into a dense FFN and refined with ~4B tokens of KD; +6.3 pp over dense-to-dense pruning at matched params, 1.6× faster wall-clock; 350 configurations tested, also DeepSeek-V2-Lite and GPT-OSS-20B — [arXiv 2605.28207](https://arxiv.org/abs/2605.28207)
- REAP: pruning 50% of experts from Qwen3-Coder-480B and Kimi-K2 with no retraining left code-generation accuracy nearly unchanged (≤~2% per secondary summaries); merging methods failed catastrophically at 50% — [REAP, ICLR 2026](https://arxiv.org/abs/2510.13999)
- Nemotron-3 Nano 4B was derived from Nemotron Nano 9B v2 via Nemotron Elastic (router-guided structured pruning jointly trained with KD), not a separate prune/NAS/distill pipeline — [HF blog](https://huggingface.co/blog/nvidia/nemotron-3-nano-4b)
- Nemotron 3 Nano 30B-A3B (31.6B total / 3.2B active) is described by NVIDIA as trained from scratch; no evidence it was pruned from Super — [NVIDIA model card](https://build.nvidia.com/nvidia/nemotron-3-nano-30b-a3b.md)
- A community model card (MiniMax-M2-THRIFT-55) claims ~55% of experts removed in ~11 stages of ~5% with teacher-matching on outputs, hidden states and router between stages; it is self-described "research in progress" with no benchmark table — [HF card, unverified](https://huggingface.co/osmapi/MiniMax-M2-THRIFT-55/blob/main/README.md)

### Inferences
- Industrial MoE recipes (Puzzle, SlimMoE) avoid deleting experts: they shrink expert width and top-k, keeping the router's full expert vocabulary. SlimMoE's ablation (slimming 63.76 vs whole-expert pruning 53.41 at 50%) is direct evidence that whole-expert deletion is the more damaging axis at equal parameter savings — the axis REAP/our pipeline uses.
- Total-parameter ratios achieved with full recovery: Puzzle 0.62×, SlimMoE 0.18×/0.09×, SlimQwen 0.29×. All used 120–400B tokens.

### Gaps
- No published prune+distill recipe for DeepSeek-V3/V4, Kimi-K2, or Qwen3-235B/480B-class MoEs; only zero-retraining REAP results exist at that scale.
- Puzzle-75B's one-shot baseline token budget and all GPU-hour figures are UNREPORTED.

## KQ3. Progressive EXPERT pruning with recovery between stages — does the achievable ratio exceed one-shot?

### Takeaway
Direct evidence says staged pruning with recovery between stages beats one-shot at the SAME final ratio and SAME token budget, but the margin is modest when recovery is long (Puzzle +0.57 avg; SlimMoE +1.3–7 points; SlimQwen +1.5 MMLU) and large only when recovery is tiny (Fantastic Experts: C4 ppl 9.76 vs 14.74 at 50% experts dropped with ~1M tokens/round); no paper shows that staging raises the *maximum* ratio at which quality is retained — at 62.5–75% expert drop every variant degrades badly, and three stages did not beat two (SlimQwen), while Minitron found one-shot better than iterative within depth.

### Cited Findings
- Fantastic Experts (Mixtral-8×7B Base, Min-EAN criterion, C4 ppl; original 7.44): at 50% experts dropped one-shot 14.74, iterative (re-estimate every 12.5%) 10.44, MoE-Lottery (re-estimate + fine-tune every 12.5%) 9.76; at 75%: 30.59 / 17.39 / 13.05; at 25%: 8.56 / 8.53 / 8.38 — [arXiv 2504.05586](https://arxiv.org/abs/2504.05586)
- Fantastic Experts zero-shot MMLU (Mixtral Base, Min-EAN): one-shot 43.97 @12.5% → 13.55 @25% → 12.63 @62.5%; iterative 48.51 → 47.81 → 29.71; lottery 49.65 → 47.13 → 37.24 (original 60.01); one-shot falls below random guess at merely 25% — [arXiv 2504.05586](https://arxiv.org/abs/2504.05586)
- Fantastic Experts Appendix B: at 75% drop, fine-tuning saturates — ppl 13.55 @0.25M tokens, 13.05 @1.13M, 13.01 @2.27M; "~1 million training tokens are sufficient to address the abrupt impact created by expert dropping" — [arXiv 2504.05586](https://arxiv.org/abs/2504.05586)
- One-shot and iterative choose significantly different experts to drop; the authors attribute one-shot's weakness to subnetwork identification quality, and fine-tuning primarily fixes load imbalance — [arXiv 2504.05586](https://arxiv.org/abs/2504.05586)
- SlimMoE: one-stage vs multi-stage at 400B tokens — Phi-mini MMLU 68.58 vs 69.87, WinoGrande 70.85 vs 75.85, GSM8K 74.52 vs 77.48; Phi-tiny MMLU 58.40 vs 60.08, GSM8K 62.89 vs 69.90; one-stage "collapses at the first pruning step"; multi-stage matches one-stage final quality at 0.74× (mini) / 0.91× (tiny) compute; a fully iterative variant got similar Phi-tiny scores but needed ~2.4× compute during initialization — [SlimMoE](https://arxiv.org/abs/2506.18349)
- SlimMoE: longer intermediate stages consistently help (25B vs 135B tokens for mini; 50B vs 130B for tiny); even at a 90B-token budget multi-stage beat one-stage (MMLU 65.59 vs 64.11) — [SlimMoE](https://arxiv.org/abs/2506.18349)
- SlimQwen (400B tokens): one-stage MMLU 75.86 / MMLU-Pro 52.97 / GSM8K 85.22 vs depth-first 2-stage 77.39 / 53.22 / 85.82; width-first 77.14 / 52.80 / 84.00; a three-stage variant (20B+20B+360B) "did not add benefit" — [SlimQwen](https://arxiv.org/abs/2605.08738)
- Puzzle-75B-A9B: 3-round iterative vs 1-round +0.57 average (69.05 vs 68.48) — [arXiv 2607.04371](https://arxiv.org/abs/2607.04371)
- Minitron: within the depth axis, one-shot prune+distill (30.2B tokens) beat iterative (86.4B); iterative importance re-estimation gave no benefit — [Minitron](https://arxiv.org/abs/2407.14679)
- MoE-Pruner: 50%-sparsity Mixtral-8×7B recovered to 68.40 avg (98.9% of 69.16) with expert-wise KD on 1000 C4 samples in ~1 h on 2×8 H100 — [arXiv 2410.12013](https://arxiv.org/abs/2410.12013)
- Half the Experts (Qwen3.6-35B-A3B, Gemma-4-26B-A4B): half the experts removable with no statistically detectable loss on the primary code benchmark; a lightweight fine-tune recovers about half of what aggressive pruning loses; single-shot benchmarks can overstate penalties — [arXiv 2607.16721](https://arxiv.org/abs/2607.16721)
- Local prior (memory): GLM-5.2 REAP keeps 168/256 experts (34% pruned) using router-gate KL distillation that trains ~0.016% of parameters; real published REAP rates are 34–37.5%, not 50–75% — [[thor-model-fit-frontier]] /home/patrickd/.claude/projects/-home-patrickd/memory/thor-model-fit-frontier.md

### Inferences
- The central hypothesis ("progressive beats one-shot") is supported as a quality-at-fixed-ratio claim in every MoE paper that tested it, and the benefit is largest exactly in the low-token regime Thor lives in (Fantastic Experts: ~1M tokens per round, ~3× ppl gain at 50%).
- It is NOT supported as a ratio-extension claim: no result shows staged pruning keeping quality at ≥62.5% expert removal (lottery MMLU 37.24 vs 60.01 original at 62.5%), and gains flatten past two stages (SlimQwen 3-stage no gain; SlimMoE fully-iterative costs 2.4× for no gain).
- For a 763B–1T MoE that must land at ~0.15–0.2× total params to fit Thor, the literature's largest full-recovery total-param ratios on MoE are 0.09–0.29× (SlimMoE/SlimQwen) and those used 400B tokens and slimmed experts rather than deleting them.

### Gaps
- No paper reports the achievable ratio frontier (quality vs ratio curve) for staged *whole-expert* deletion with teacher KD between stages at >50% removal on a >100B MoE.
- Fantastic Experts uses LM-loss fine-tuning, not teacher KD, and the smallest model class (Mixtral 8 experts/layer); transfer to 256-expert fine-grained MoEs is untested.

## KQ4. Distillation when the teacher cannot fit beside the student: cached top-K logits, how many tokens, storage, top-K vs full KD

### Takeaway
Offline cached top-K KD is standard (Gemma, SlimMoE top-8, Minitron-practice K=100) and matches online KD at near-identical loss (K=100, Llama-3.1-8B→3.2B, 29% faster, no resident teacher), but top-K is a biased estimator: at K=100 only ~55% of the CE→FullKD gap is recovered and K≤12 is worse than plain CE, whereas 12 importance-sampled tokens per position (~3 bytes each, 3.6 TB per 100B tokens) is unbiased and matches full KD.

### Cited Findings
- Storage: full-vocabulary float32 logits for 100B tokens ≈ 58.6 PB; top-p 0.95 then top-100 truncation cuts this ~4000× to ~15 TB — [Pre-training Distillation, arXiv 2410.16215](https://arxiv.org/abs/2410.16215)
- Full KD storage ≈ 10 PB per 100B tokens; Top-300 ≈ 90 TB; 12 sampled tokens at 24 bits (17-bit id + 7-bit prob) ≈ 3.6 TB per 100B tokens — [Sparse Logit Sampling, arXiv 2503.16870](https://arxiv.org/abs/2503.16870)
- Top-K bias (3B teacher → 300M student, 10B tokens): loss FullKD 2.75, CE 2.81, Top-1 3.04, Top-5 2.96, Top-12 2.87, Top-25 2.82, Top-100 2.77 (~55% of gap), Top-300 2.76 (~78%); ECE rises as K falls; at 8B→3B/100B tokens Top-12 loss 2.50 vs FullKD 2.34 — [Sparse Logit Sampling](https://arxiv.org/abs/2503.16870)
- Random-sampling KD (12 unique sampled tokens/position, forward KL on sparse targets) matched FullKD: 3B/100B tokens loss 2.350 vs 2.344, 0-shot 57.5 = FullKD, instruction score 59.4 vs 58.4; 1.7–2.6× faster than FullKD, ~10% slower than CE — [Sparse Logit Sampling](https://arxiv.org/abs/2503.16870)
- Offline top-K=100 KD (teacher Llama-3.1-8B-Instruct, ~3.2B student, Megatron-Bridge, single H200, 8K ctx): matches online at near-identical loss, peak memory 103→78 GB, 29% faster/iteration (25.9→18.5 s), 237→331 TFLOP/s; fused chunked KL lowers peak to 58 GB and enables 32K context; "the teacher need only be run once … a 70B teacher need not sit in the training loop" — [arXiv 2608.03796](https://arxiv.org/abs/2608.03796)
- Minitron: top-K teacher-logit filtering with K≤100 hurt accuracy (so NVIDIA's Minitron runs used full logits online) — [Minitron](https://arxiv.org/abs/2407.14679)
- SlimMoE distilled against the teacher's top-8 probabilities (others masked to zero) for all 400B tokens and reports multi-stage Phi-mini MMLU 69.87 (teacher 78.36) — [SlimMoE](https://arxiv.org/abs/2506.18349)
- Pre-training distillation: offline logits from a fully-trained teacher worked best; online logits from an unconverged teacher hurt; α=0.9 KD + 0.1 LM best static mix; +1.6% average over LM-only at 100B tokens (GSM8K +24.6%); larger students benefit more; a larger teacher does not necessarily help (capacity gap) — [arXiv 2410.16215](https://arxiv.org/abs/2410.16215)
- Puzzle-75B used 30% pretraining / 70% SFT data for its logit-matching KD; Gemma-family practice cited as "K of order a few hundred" cached logits — [arXiv 2607.04371](https://arxiv.org/abs/2607.04371); [arXiv 2608.03796](https://arxiv.org/abs/2608.03796)

### Inferences
- **[THOR EXTRAPOLATION]** Storage per token is the Thor-relevant quantity: K=100 top-K at (4-byte id + 2-byte prob) ≈ 600 B/token → 100M tokens ≈ 60 GB; importance-sampled 12 tokens ≈ 36 B/token → 100M tokens ≈ 3.6 GB; either fits the ~247 GB free disk noted in local memory. Caching is cheap; *generating* the cache with a 763B–1T teacher that does not fit in 122 GiB is the problem (see KQ7).
- For the teacher-can't-fit case, importance-sampled sparse KD is strictly preferable to top-K: unbiased, 100× smaller, and validated at 100B tokens against full KD.

### Gaps
- No study of top-K vs full KD specifically for prune-recovery of a MoE at >100B scale; the bias results are for from-scratch students ≤3B.
- Whether a teacher that must be streamed from disk can produce logits at a useful rate on Thor is unmeasured (local memory says streaming decode is 0.6–2.4 tok/s; prefill-batched capture rate is not measured).

## KQ5. Data for distillation: mixture, tokens, and whether a small license-clean corpus (~1–5B tokens) suffices

### Takeaway
Industrial recoveries use 63–400B tokens drawn from the pretraining mix (Puzzle: 30% pretrain / 70% SFT), and Minitron shows quality still climbing from 18.9B to 94B tokens; but "shock recovery" after expert dropping saturates at ~1M tokens (Fantastic Experts) and 1000 C4 samples restore 98.9% at 50% weight sparsity (MoE-Pruner), so a 1–5B-token corpus is enough to repair routing/load-imbalance damage, not to re-learn pruned capacity.

### Cited Findings
- Minitron 8B retraining: 18.9B tokens → MMLU 58.3; 94B → 62.8; retraining only on the final-phase (Phase 2) data gave MMLU 61.9 vs 54.7 for Phase 1+2 mixed (113B) — data quality/phase matters more than volume — [Minitron](https://arxiv.org/abs/2407.14679)
- Teacher correction: ~100B tokens fine-tuning the teacher on the new corpus reduced distillation validation loss >6% when original pretraining data is unavailable — [arXiv 2408.11796](https://arxiv.org/abs/2408.11796)
- Puzzle-75B-A9B heal data: 30% pretraining + 70% Nemotron-3-Nano SFT data; 120B heal tokens plus up to 100B long-context tokens — [arXiv 2607.04371](https://arxiv.org/abs/2607.04371)
- SlimMoE: 400B tokens, a subset of the Phi-3.5 pretraining corpus (<10% of ~4T) — [SlimMoE](https://arxiv.org/abs/2506.18349)
- SlimQwen: 120B and 400B budgets; pruned+KD at 120B reaches avg 73.45 (86.5% of teacher 82.68) — [SlimQwen](https://arxiv.org/abs/2605.08738)
- Nemotron-H: ~7B tokens lifted candidates from 53.7–63.3 → 69.3–69.6 avg (teacher 70.6), i.e., most of the gap closed in 7B tokens; 63B more tokens for the final model — [Nemotron-H](https://arxiv.org/abs/2504.03624)
- Fantastic Experts: ~1M tokens of C4 suffices after 75% expert drop; more brings "marginal or no gain" (13.55 → 13.01 ppl from 0.25M → 2.27M tokens) — [arXiv 2504.05586](https://arxiv.org/abs/2504.05586)
- MoE-Pruner: 1000 C4 samples, 3 epochs, ~1 h on 16 H100 → 98.9% of original average at 50% sparsity — [arXiv 2410.12013](https://arxiv.org/abs/2410.12013)
- MoE→dense: ~4B tokens of KD on Qwen3-30B-A3B conversions — [arXiv 2605.28207](https://arxiv.org/abs/2605.28207)
- Pre-training distillation corpus scaling tested to 500B tokens — [arXiv 2410.16215](https://arxiv.org/abs/2410.16215)
- Local prior (memory): REAP gap is corpus imbalance (general domain 4.76%), not criterion; pruning is ~14× more damaging than quantization per unit — [[reap-criterion-and-corpus-audit]] /home/patrickd/.claude/projects/-home-patrickd/memory/reap-criterion-and-corpus-audit.md

### Inferences
- Two regimes with a ~1000× gap: (i) repair of routing shock, 1M–1B tokens, restores 97–99% at ≤50% pruning (MoE-Pruner, Fantastic Experts, Half-the-Experts); (ii) re-learning capacity after aggressive width/depth/expert cuts, 60–400B tokens, where quality keeps rising with tokens (Minitron 18.9B→94B: +4.5 MMLU). A 1–5B license-clean corpus covers regime (i) only.
- Nemotron-H's 7B-token screen closing ~90% of the gap suggests the first few B tokens have very high marginal value; the remaining tail is what costs 10× more.

### Gaps
- No paper isolates "small clean corpus vs pretraining-mix" at fixed tokens for MoE recovery; mixture composition for Puzzle/SlimMoE beyond ratios is UNREPORTED.

## KQ6. The quality ceiling: does pruned+distilled quality approach a from-scratch model of the final size, and at what ratio does it fall below?

### Takeaway
At token-matched budgets pruned+distilled models beat from-scratch models by large margins (Minitron 4B: 42.45 vs 26.24 MMLU with 4× fewer tokens; SlimQwen: 73.45 vs 61.66 avg at 120B), and at ≤0.6× total params with 60–120B tokens they land within 1–2 points of the teacher (Nemotron-H 47B, Puzzle-75B); at 0.1–0.3× (SlimMoE, SlimQwen) they trail the teacher by 8–10 MMLU points even after 400B tokens, and no result shows a pruned model beating a *fully trained* same-size model from scratch at the frontier.

### Cited Findings
- Minitron 4B (Table 11): random init 400B tokens MMLU 26.24; pruned from 15B + 100B tokens 37.81; pruned from 8B + 100B tokens 42.45 — [Minitron](https://arxiv.org/abs/2407.14679)
- Minitron 8B MMLU 63.8 vs Nemotron-3 8B (from scratch, 3.8T tokens) 54.7, Llama-3 8B 65.3, teacher 66.6 — [Minitron](https://arxiv.org/abs/2407.14679)
- SlimQwen (120B tokens): random-init+KD avg 61.66 vs pruned+LM 69.96 vs pruned+KD 73.45; teacher 82.68; pruning "consistently outperforms training the target architecture from scratch" at the same budget — [SlimQwen](https://arxiv.org/abs/2605.08738)
- SlimMoE at 0.18× params: Phi-mini-MoE MMLU 70.68 vs teacher 78.36; Phi-tiny (0.09×) 60.83 vs Llama-3.2-3B 61.73; pruned Phi-3.5-MoE consistently beat pruned dense Phi-3-medium, with a larger gap at 16% retention — [SlimMoE](https://arxiv.org/abs/2506.18349)
- Puzzle-75B (0.62×): MMLU-Pro 82.4 vs 83.8, GPQA 78.6 vs 80.5, SWE-Bench 56.9 vs 59.5 after 120B heal + long-context KD — [arXiv 2607.04371](https://arxiv.org/abs/2607.04371)
- Nemotron-H 47B (0.84×): MMLU 83.6 vs 84.2, GSM8K 93.3 vs 93.7 after 63B tokens, "near-lossless" — [Nemotron-H](https://arxiv.org/abs/2504.03624)
- Nemotron Elastic 9B (0.75×) matches a from-scratch NanoV2-9B (75.95 vs 75.99 avg) at 110B tokens; 6B (0.5×) 70.61 with no from-scratch baseline — [arXiv 2511.16664](https://arxiv.org/abs/2511.16664)
- Llama-3.1-Minitron-4B-Width (0.56×, 94B tokens) MMLU 60.5 vs teacher 65.3; depth variant 58.7 and GSM8K 16.8 vs 48.6 — [arXiv 2408.11796](https://arxiv.org/abs/2408.11796)
- Expert deletion curve (Mixtral, best staged method, MMLU zero-shot): 49.65 @12.5%, 47.13 @25%, 40.79 @37.5%, 37.24 @50% (text lists 37.24 at 62.5% column; table alignment ambiguous), 28.12 @75% vs 60.01 original — [arXiv 2504.05586](https://arxiv.org/abs/2504.05586)
- Pruned-init alone is worth only 0–2% at Apple (6.4B→3B, FFN-only), distillation adds ~5% MMLU — [Apple, arXiv 2407.21075](https://arxiv.org/abs/2407.21075)

### Inferences
- Rule of thumb from the table: ratio ≥0.6× with ≥60B KD tokens → within ~1–3 points of teacher; ratio 0.2–0.3× with 400B tokens → ~8–10 MMLU points below teacher but above same-size from-scratch at matched tokens; whole-expert deletion beyond ~37.5% with light recovery loses ≥20 MMLU points.
- The Minitron "approaches from-scratch quality" claim is a *token-matched* claim; it never compares against a same-size model trained for its full budget (e.g., Llama-3 8B 65.3 > Minitron 8B 63.8).

### Gaps
- No published quality-vs-ratio curve for a ≥500B MoE pruned to ≤0.2×; the closest data points are SlimMoE (42B→3.8B) and SlimQwen (80B→23B).

## KQ7. Compute scaling to one Thor: feasible distillation tokens in 24 h / 168 h and whether it is enough

### Takeaway
On measured Thor numbers (cuBLASLt BF16 121–156 TFLOP/s, NVFP4 571–657, mma.sync BF16 53 / FP8 92 at 1.05 GHz), full-parameter distillation of a ~400B-total/~40B-active student at a realistic 40–100 TFLOP/s sustained yields roughly 15–35M tokens/day and 0.1–0.25B tokens/week — about 300–1000× short of the 63–400B-token budgets of every industrial recipe, and full-parameter training does not fit in 122 GiB anyway; what fits is the "shock-repair" regime (1M–100M tokens; routers/shared/LoRA or one layer at a time), which the literature shows restores 97–99% only at ≤50% expert removal.

### Cited Findings
- Datasheet (Jan 2026): T5000 "2070 TFLOPS (FP4—sparse)", 2560-core Blackwell GPU, 1.57 GHz max, 128 GB 256-bit LPDDR5X 273 GB/s, 40–130 W; no FP8/FP16/BF16 row is published — [NVIDIA Jetson Thor datasheet](https://dam-cdn.nvd.orangelogic.com/AssetLink/x20v2h2y15uh0ne54rpb73xh4x7s0m1s.pdf)
- Third-party aggregator: T5000 FP8 dense 517 / sparse 1035 TFLOPS; FP4 dense 1035 / sparse 2070; BF16 "no verified data" — [flopper.io](https://flopper.io/gpu/nvidia-jetson-agx-thor-t5000-128gb) (conflicts with local project notes that assumed 1035 dense FP8)
- Forum measurements: torch.mm FP16 best mean 155.5 TFLOPS @3584×3072×3584, max 163.3; FP32 5.3 TFLOPS; cuBLAS ~130 TFLOPS FP16 vs "500 claimed"; kernel choice swings 60 vs 160 TFLOPS — [NVIDIA forum thread 344399](https://forums.developer.nvidia.com/t/thor-torch-mm-benchmark-results-float32-float16-float8-e3m2fn/344399), [thread 349494](https://forums.developer.nvidia.com/t/performance-benchmarking-on-jetson-thor/349494) (fetch returned 403; numbers from search snippets)
- Local measurement (this box, `tools/flops_probe.cu`, 20 SMs, clock 1.049 GHz): FP32 FFMA 5.45, FP16/BF16 mma.sync 53.07/53.08, FP8 e4m3 mma.sync 92.18 TFLOPS; cuBLASLt NVFP4 (tcgen05) 571–657 TFLOP/s vs BF16 121–156 — [LOOP_LOG Finding 87](/home/patrickd/deepseek-v4-flash-0731-cuda/LOOP_LOG.md) line 5292; [IMPLEMENTATION_PLAN.md](/home/patrickd/deepseek-v4-flash-0731-cuda/IMPLEMENTATION_PLAN.md) line 18
- Local: unified memory 122.8 GiB usable; a full 180B NVFP4 forward uses ~90% of it; co-running anything beside a resident model forces hard reboots — [[dspark-v4flash-180b-thor]], [[thor-unified-memory-oom]] (memory dir)
- Jetson AI Lab fine-tune tutorial: full SFT of Qwen3.5-4B, 500 samples, 1 epoch ≈ 5 min at ~42 GB; QLoRA Qwen3.5-27B 512 samples ≈ 10 min; ~115 GB usable for training; tokens/s UNREPORTED — [jetson-ai-lab](https://www.jetson-ai-lab.com/tutorials/finetune-on-jetson)
- Reference GPU numbers for scale: a single H200 sustains 331 TFLOP/s distilling a 3.2B student offline — [arXiv 2608.03796](https://arxiv.org/abs/2608.03796); Puzzle heal ran 16M-token global batches on Megatron-LM — [arXiv 2607.04371](https://arxiv.org/abs/2607.04371)
- Local prior: streaming an over-size model from NVMe gives ~1–2.4 tok/s decode on Thor and is "lossless"; a 744B GLM at REAP50 is ~126 GB at 2-bit and still does not fit — [[thor-model-fit-frontier]] (memory dir)

### Inferences (all **[THOR EXTRAPOLATION]**, arithmetic shown)
- Training FLOPs per token for a MoE student ≈ 6 × active params (fwd+bwd, ignoring attention/lm_head). For a ~400B-total / ~40B-active student: 6 × 40e9 = 2.4e11 FLOP/token. Offline cached logits remove the teacher pass; online teacher would add 2 × 40e9 = 8e10 (but the teacher does not fit).
- Sustained throughput assumptions: BF16 cuBLASLt peak 121–156 TFLOP/s measured; single-device training with activation checkpointing, MoE dispatch and vocab-150k KD loss typically reaches 30–50% MFU → **40–75 TFLOP/s BF16**; FP8 tensor cores measured 1.74× BF16 on mma.sync, so an FP8-trained path might reach **~70–130 TFLOP/s** if cuBLASLt FP8 scales similarly (unmeasured on this box).
  - Tokens/day = T × 86400 / 2.4e11: at 40 TFLOP/s → 14.4M; at 75 → 27M; at 130 → 47M.
  - Tokens/week: 0.10B / 0.19B / 0.33B.
  - Versus recipes: Minitron 94B → 280–650 weeks; Puzzle 120B → 360–830 weeks; SlimMoE/SlimQwen 400B → 1200–2800 weeks; Nemotron-H 7B screen → 21–48 weeks; MoE→dense 4B → 12–28 weeks; Fantastic-Experts ~1M/round → ~1 hour/round; MoE-Pruner 1000 samples (~4M tokens at 4k) → ~4 h.
- Memory makes full-parameter KD of a 400B student impossible regardless of FLOPs: BF16 weights 800 GB, Adam states 12 B/param ≈ 4.8 TB. Even NVFP4 weights (~0.62 B/w measured, ≈250 GB) exceed 122 GiB. Feasible modes: (a) parameter-efficient recovery (router gates + shared experts + norms + LoRA; GLM-5.2 REAP trained 0.016% of params this way), (b) one layer at a time against cached teacher-layer activations (Puzzle-style block-wise local distillation); for (b) each cached (input, output) pair at hidden 4096 × bf16 ≈ 16 KB/token, so 10M tokens ≈ 160 GB per layer, re-generated per layer, with the teacher itself streamed from disk.
- Teacher logit capture: a 763B–1T teacher at ~0.62 B/w is 470–620 GB, not resident; prefill-only, teacher-forced capture amortizes one weight stream (~100–200 s at 3–6 GB/s NVMe) over a batch; at 2 × 40e9 FLOP/token and ~100 TFLOP/s NVFP4-effective, 262k tokens/pass ≈ 210 s compute → ~1k tokens/s → ~90M tokens/day of cached logits if the pipeline overlaps I/O. This is unmeasured; local measurements cover streaming *decode* only (0.6–2.4 tok/s).
- Net: Thor can afford 10^7–10^8 distillation tokens per week, which is the regime where the literature shows 97–99% retention at ≤50% expert pruning (MoE-Pruner, Fantastic Experts, REAP) and where staging gives its largest relative gain (~3× ppl at 1M tokens/round). It cannot afford the 60–400B-token regime where ratios of 0.1–0.3× were recovered, so a 763B–1T → ≤0.2× target cannot be expected to approach teacher quality on Thor alone.

### Gaps
- No measured end-to-end training tokens/s on Thor for any model >9B (only tutorial wall-clock for ≤27B QLoRA); the 30–50% MFU assumption is from H100-class practice, not Thor.
- cuBLASLt FP8 GEMM throughput on this box is unmeasured (only mma.sync 92 TFLOPS at a 1.049 GHz probe clock, below the 1.57 GHz max).
- Batched prefill teacher-logit capture rate with disk streaming is unmeasured.
