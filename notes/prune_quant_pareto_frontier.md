# Joint frontier of expert-pruning ratio x weight bit-width for large MoE models

Researcher notes, 2026-10-17. Scope: which (prune ratio, bits/weight) combinations have measured quality, where the cliff is, and what moves it. All bits/weight figures below are TRUE bytes-on-disk x 8 / total params unless marked "nominal". Interpolations and extrapolations are explicitly labelled. Internal project measurements (GLM-5.3-Flash REAP50) are confined to the Inferences subsections and labelled as such; they have no public URL.

**Correction to the assignment premise, up front.** The W2 collapse in REAP Appendix E (Table A5) is NOT a Kimi-K2 result and the W2 point is almost certainly NOT pruned. It is Qwen3-30B-A3B, AWQ W2A16-G128 via LLM Compressor, and the prose says only the 4-bit checkpoint was pruned ("We additionally compress the 4-bit checkpoint by using REAP ... to prune half the experts"); the W2 row's 12.5% relative size is only consistent with 128 experts (W2 on 64 experts would be 6.25%). The "64" in the W2 row is therefore most likely a typo. The Kimi-K2 result is W4A16 (RedHat AWQ INT4 checkpoint) + 50% REAP, and it is near-lossless on code. Details in KQ1.

---

## KQ1. Reproduce the REAP App. E table and every other prune x quant grid on MoE

### Takeaway
Only one paper (REAP A5, Qwen3-30B-A3B) and one preprint (arXiv 2607.16721, Qwen3.6-35B-A3B) report static-expert-pruning x bit-width grids, and both stop at 50% pruning and at GGUF/AWQ-class quantizers. Every other "prune+quant" MoE paper (EAC-MoE, MC-MoE) uses dynamic token-level expert skipping (8-15% effective), not 25-70% static removal. The literature has NO measured point at prune >= 60% with any quantization, and NO measured point at prune >= 25% with bits < 4.

### Cited Findings

**REAP Appendix E, Table A5 (Qwen3-30B-A3B, EvalPlus = mean of HumanEval+ and MBPP+, AWQ via LLM Compressor, calibrated on evol-codealpaca-v1):**

| Quantization | Num. experts | Approx. relative size | Eval+ |
|---|---|---|---|
| W16A16 | 128 | 100.0% | 81.4 |
| W16A16 | 64 (50% REAP) | 50.0% | 78.0 |
| W4A16-G128 | 128 | 25.0% | 80.5 |
| W4A16-G128 | 64 (50% REAP) | 12.5% | 77.6 |
| W2A16-G128 | "64" (see note) | 12.5% | 28.6 |

Verbatim prose: "we calibrate and quantize Qwen3-30B-A3B using the evol-codealpaca-v1 dataset to 4- and 2-bit weights with the LLM Compressor library ... We additionally compress the 4-bit checkpoint by using REAP calibrated on the same dataset to prune half the experts ... This preliminary analysis suggests that REAP combined with 4-bit weights outperforms more aggressive quantization. A more thorough exploration of various low-bit schemes and pruning combinations remains an important direction for future work." Caption: "quantization outperforms expert pruning up to 4-bit weights. Beyond 4-bit weights, combining quantization with expert pruning yields significantly better accuracy than more aggressive quantization." — [REAP, arXiv 2510.13999 v3, App. E](https://arxiv.org/pdf/2510.13999) (extracted from the PDF with pdftotext; the arXiv HTML truncates before App. E)
- The Kimi-K2 model used throughout REAP "is an INT4 weight-quantized version of Kimi-K2-Instruct released by RedHatAI" (Kimi-K2-Instruct-W4A16, 384 experts, 1026.4B params) and Qwen3-Coder-480B is used as FP8 — so the 87.5% figure is W4 x 50% prune measured directly on Kimi-K2 — [REAP App. E and Table of models](https://arxiv.org/pdf/2510.13999); [RedHatAI/Kimi-K2-Instruct-quantized.w4a16](https://huggingface.co/RedHatAI/Kimi-K2-Instruct-quantized.w4a16)
- Kimi-K2-Instruct-W4A16 (Table 3): Baseline Eval+ 0.828 / LiveCode 0.434 / SWE-Bench-Verified 0.554 / BFCLv3 0.666 / MC avg 0.780; 25% REAP 0.840 / 0.440 / 0.580 / 0.635 / 0.773; 50% REAP 0.819 / 0.429 / 0.576 / 0.564 / 0.643. Frequency-pruning at 50%: Eval+ 0.112 (collapse) — [REAP HTML v3, Table 3](https://arxiv.org/html/2510.13999v3)
- Qwen3-30B-A3B 16-bit (Table 2): Baseline Eval+ 0.814 / MC avg 0.721; 25% REAP 0.797 / 0.665; 50% REAP 0.780 / 0.503 — [REAP HTML v3, Table 2](https://arxiv.org/html/2510.13999v3)
- REAP states all results are "evaluated in the one-shot setting, with no additional fine-tuning after compression" — [REAP PDF](https://arxiv.org/pdf/2510.13999)

**"Half the Experts, All the Code" (arXiv 2607.16721), Qwen3.6-35B-A3B, 50% expert pruning (128 of 256 per layer), GGUF, Table 5 (greedy pass@1):**

| Budget | Model | Size | HE+ | MBPP+ |
|---|---|---|---|---|
| ~11-13.5 GB | pruned 50%, Q4 | 11.4 GB | 0.902 | 0.720 |
| ~11-13.5 GB | pruned 50%, Q5 | 13.3 GB | 0.915 | 0.749 |
| ~11-13.5 GB | base, IQ2_M | 13.0 GB | 0.896 | 0.730 |
| ~11-13.5 GB | base, Q2_K | 13.5 GB | 0.848 | 0.754 |
| ~17-22 GB | base, Q3_K_M | 17.1 GB | 0.890 | 0.775 |
| ~17-22 GB | pruned 50%, Q8 | 19.75 GB | 0.896 | 0.728 |
| ~17-22 GB | base, Q4_K_M | 22.3 GB | 0.884 | 0.759 |

Significant: pruned-Q5 beats 2-bit base on HE+ (+6.7, CI [+3.0,+11.0], p=0.003); 3-bit base beats pruned-Q8 on MBPP+ (-4.8, CI [-7.7,-2.1], p=0.0009); all other pairings within noise. Guidance: "quantize down to 3 bits before pruning anything, and prune below that boundary"; "2-bit quantization visibly damages the full model"; the 3-bit base "loses essentially nothing relative to Q8" — [arXiv 2607.16721](https://arxiv.org/html/2607.16721)

**EAC-MoE (ACL 2025): GPTQ-style g128 experts at 2/2.5/3 bit, attention at 4-bit, router FP, plus dynamic skipping of experts selected < alpha x average (alpha = 0.3 or 0.7; NOT static removal):**
- Mixtral-8x7B 0-shot avg: FP 72.64; QESC 3.03-bit 72.21; QESC 3.03-bit + PESF alpha=0.3 71.68; memory 93.41 GB -> 18.98 GB. Phi3.5-MoE 69.62 / 68.49 / 68.31; DeepSeek-MoE-16B 61.38 / 61.22 / 61.09; Qwen1.5-MoE 64.72 / 62.89 / 62.73.
- vs MC-MoE on Mixtral with alpha=0.3 pruning: 2.06-bit avg 65.90 (MC-MoE 62.56); 2.56-bit 68.60 (MC-MoE 68.65).
- PESF applies only in prefill; largest model ~50B — [EAC-MoE arXiv 2508.01625](https://arxiv.org/html/2508.01625v1)

**MC-MoE (Mixtral 8x7B / 8x22B): expert bits in {1,2,3} by integer program, attention+gate at 4-bit ("no more than 0.05 bits" overhead), ODP dynamic pruning of ~14.9-15.8% of activated expert params, GPTQ, no fine-tuning:**
- 8x7B (FP16 71.29, 96.80 GB): uniform GPTQ 2.00-bit 42.67 (13.61 GB); PMQ 2.54 67.50 (16.24 GB); PMQ+ODP 2.54 66.94; PMQ 2.05 63.25 (13.41 GB); PMQ+ODP 2.05 62.68; PMQ 1.57 54.49.
- 8x22B (FP16 76.33): uniform 2.00 50.44; PMQ 2.54 72.08; PMQ+ODP 2.54 71.21; PMQ 2.05 67.94; 1.57 59.29.
- WikiText2 PPL (no pruning): FP16 3.84; uniform 2-bit 16.38; 2.54 5.09; 2.05 5.91; 1.57 8.50 — [MC-MoE arXiv 2410.06270](https://arxiv.org/html/2410.06270)
- True bpw from reported GB / 46.7B: PMQ "2.54" = 2.78 bpw; PMQ "2.05" = 2.30 bpw; uniform "2-bit" = 2.33 bpw (computed here from the paper's GB column).

**Pruning-only curves (no quant) at ratios the grid needs:**
- HOPE (arXiv 2609.18916), mean score across Qwen3.5-122B / Qwen3.5-35B / GLM-4.5-Air: base 48.87; 10% HOPE 48.49; 40% HOPE 46.94 vs 40% REAP 43.16 (search-snippet figures; the full table was not fetched) — [HOPE](https://arxiv.org/pdf/2609.18916)
- MAESTRO (arXiv 2607.08601), after LoRA recovery fine-tune, Avg Retained Performance: GPT-OSS-20B 25% REAP 97.67% / MAESTRO 98.90%; 50% REAP 89.37% / MAESTRO 92.59%. Qwen3-30B 25% REAP 98.13% / MAESTRO 99.40%; 50% REAP 90.25% / MAESTRO 93.40%. Log-PPL Qwen3-30B base 1.97; 50% REAP 2.45; 50% MAESTRO 2.04 — [MAESTRO PDF](https://arxiv.org/pdf/2607.08601)
- Quasar-438B (Multiverse): GLM-5.2 pruned 265 -> 148 experts/layer (~44%), "quantization-aware compression" to FP8 and NVFP4, healing pass after pruning; NO size, bpw, token count or benchmark scores published — [Multiverse blog](https://multiversecomputing.com/papers/inside-quasar-438b-by-multiverse-computing)

**Quantization-only curves on frontier MoEs (0% prune) — the bit-width axis of the grid:**
- DeepSeek-R1 (arXiv 2505.02390, Table 1/2): Q4_K_M 377 GB "avg quants 4.82" accuracy drop 0.68%; Q3_K_M 298 GB (3.81) -1.80%; DQ3_K_M 281 GB (3.59) -0.34%; UD-Q2_K_XL 212 GB (2.70) -0.94%. DeepSeek-V3-0324 (Table 4): Q4_K_M -1.35%; Q3_K_M -1.85%; Q2_K_L 228 GB (2.91) -14.66% (AIME 57.9 -> 31.25, C-Eval 89.6 -> 73.6); DQ3_K_M -0.30% — [arXiv 2505.02390](https://arxiv.org/pdf/2505.02390). True bpw from GB/671B: Q4_K_M 4.49, Q3_K_M 3.55, DQ3_K_M 3.35, Q2_K_L 2.72, UD-Q2_K_XL 2.53 (computed here; the paper's "avg quants" column is its own accounting).
- DeepSeek-V3.1 Unsloth Aider Polyglot, non-reasoning (full 71.6): Q5_K_XL 484 GB 70.7; Q4_K_XL 387 GB 69.7; Q4_K_M 405 GB 69.7; IQ4_XS 357 GB 69.2; Q3_K_XL 300 GB 68.4; IQ3_XXS 279 GB 66.8; Q2_K_XL 255 GB 65.8; IQ2_M 235 GB 64.3; IQ2_XXS 225 GB 61.2; IQ1_M 206 GB 55.7; TQ1_0 170 GB 50.7. Reasoning mode (full 76.1): 3-bit 75.6; 2-bit 66.7; 1-bit 57.8. Non-Unsloth 1-2-bit quants "either failed to load or produced gibberish and looping outputs" — [Unsloth Aider page](https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs/unsloth-dynamic-ggufs-on-aider-polyglot). True bpw (GB x 8 / 671B, computed here): Q2_K_XL 3.04; IQ2_M 2.80; IQ2_XXS 2.68; IQ1_M 2.46; TQ1_0 2.03; Q3_K_XL 3.58; Q4_K_XL 4.61.
- Kimi-K2-Thinking (ubergarm, ik_llama, wiki.test PPL, all attention + first dense layer + shared expert at q8_0 in every recipe): Q4_X 543.6 GiB 4.549 bpw PPL 2.0818; smol-IQ4_KSS 4.059 2.1343; IQ3_K 3.845 2.1456; smol-IQ3_KS 3.249 2.2363; IQ2_KL 2.920 2.3735; smol-IQ2_KL 2.755 2.4550; smol-IQ2_KS 2.261 2.9361; smol-IQ1_KT 1.832 3.5931 — [ubergarm/Kimi-K2-Thinking-GGUF](https://huggingface.co/ubergarm/Kimi-K2-Thinking-GGUF)
- Unsloth Kimi-K2-Thinking sizes (no evals): UD-TQ1_0 247 GB; UD-IQ1_S 285; UD-IQ1_M 309; UD-IQ2_XXS 335; UD-IQ2_M 354; UD-Q2_K_XL 386; UD-IQ3_XXS 422; UD-Q3_K_XL 455; IQ4_XS 547; UD-Q4_K_XL 646; Q8_0 1.09 TB — [unsloth/Kimi-K2-Thinking-GGUF](https://huggingface.co/unsloth/Kimi-K2-Thinking-GGUF). True bpw at 1026.4B: 1.93 / 2.22 / 2.41 / 2.61 / 2.76 / 3.01 / 3.29 / 3.55 / 4.26 / 5.04 / 8.50 (computed here).
- EXL3 (QTIP-family trellis) on Step-3.5-Flash (MoE), wikitext2 10 rows: 2.00 bpw PPL 2.629 KL 0.653; 2.08 bpw 2.154 / 0.466; 3.00 bpw 1.521 / 0.142; 3.05 bpw 1.478 / 0.118; 4.00 bpw 1.379 / 0.053; original 1.336 — [turboderp/Step-3.5-Flash-exl3](https://huggingface.co/turboderp/Step-3.5-Flash-exl3/blob/main/README.md)
- Qwen3-30B-A3B 0-shot 7-task avg (BitsMoE Table 1, no pruning): FP16 71.61; GPTQ 3-bit 68.99; BitsMoE 3-bit 70.57; HQQ 2-bit 43.18; GPTQ 2-bit 42.65; BitsMoE 2.0 64.29; 1.9 62.08; 1.8 60.04. DeepSeek-V2-Lite: FP16 57.58; GPTQ 2 44.66; BitsMoE 2.0 47.44 — [BitsMoE arXiv 2606.00079](https://arxiv.org/pdf/2606.00079) (PDF text extracted locally)

### Measured grid (prune ratio x true bits/weight) — compiled from the citations above

Quality column = relative change vs the unpruned high-precision baseline on the stated benchmark. "n/m" = not measured anywhere found.

| Prune | ~2.0 bpw | ~2.5 bpw | ~3.0 bpw | ~3.5 bpw | ~4.0-4.6 bpw | 8-16 bit |
|---|---|---|---|---|---|---|
| 0% | Qwen3-30B GPTQ/AWQ naive INT2: -40 to -65% (collapse); BitsMoE 2.0 -10%; Mixtral uniform GPTQ -40%, MC-MoE 2.3 true -11%; Kimi-K2-T 2.26 bpw PPL +41%; V3.1 TQ1_0 2.03 Aider -29% | DS-R1 UD-Q2_K_XL 2.53: -0.94% (9-bench); V3.1 IQ1_M 2.46: -22% Aider; IQ2_XXS 2.68: -14.5%; V3-0324 naive Q2_K_L 2.72: -14.7%; Kimi-K2-T 2.76: PPL +18%; Mixtral MC-MoE 2.78 true: -5.3%; EAC-MoE 2.56: -5.6% | V3.1 Q2_K_XL 3.04: -8.1% (Aider), reasoning -12%; Kimi-K2-T 2.92: PPL +14%; Step-3.5-Flash EXL3 3.0: KL 0.14; Qwen3.6-35B IQ2_M 2.97: HE+ ~0 / Q2_K 3.09: -5% | DS-R1 Q3_K_M 3.55: -1.8%, DQ3_K_M 3.35: -0.3%; V3.1 Q3_K_XL 3.58: -4.5%; Kimi-K2-T 3.25: PPL +7%; Qwen3-30B BitsMoE 3-bit: -1.5%, GPTQ -3.7% | DS-R1 Q4_K_M 4.49: -0.7%; V3.1 Q4_K_XL 4.61: -2.7%; Qwen3-30B AWQ W4 g128: -1.1% EvalPlus; Kimi-K2-T 4.06: PPL +2.5%; EXL3 4.0: KL 0.05 | baseline |
| 25% | n/m | n/m | n/m | n/m | Kimi-K2 W4A16 + REAP25: Eval+ +1.4%, MC -0.9%, BFCL -4.7% | Qwen3-30B REAP25: Eval+ -2.1%, MC -7.8%; MAESTRO/REAP RP 97.7-99.4% (post-LoRA) |
| 40-44% | n/m | n/m | n/m | n/m | Quasar-438B (GLM-5.2, 44%, NVFP4/FP8 + healing): no numbers published | HOPE 40%: -4%; REAP 40%: -11.7% (mean over 3 models) |
| 50% | REAP A5 "W2+64" row if read literally (1.0 bit/original weight): 28.6 = -65% — but see premise correction | n/m | n/m | n/m | Qwen3-30B W4 g128 + REAP50: 77.6 = -4.7% (12.5% bytes); Kimi-K2 W4A16 + REAP50: Eval+ -1.1%, LiveCode -1.2%, SWE -? (+4%), BFCL -15%, MC -17.6%; Qwen3.6-35B pruned Q4 (4.62 on survivors): HE+ 0.902 vs base Q8-class ~0.89 | Qwen3-30B REAP50: Eval+ -4.2%, MC -30%; MAESTRO/REAP RP 89-93% post-LoRA |
| 60% | n/m | n/m | n/m | n/m | n/m | n/m (no published 60% static expert-pruning point with evals found) |
| 70% | n/m | n/m | n/m | n/m | n/m | n/m |

### Inferences
- The REAP A5 W2 point is best read as "naive AWQ INT2 on the UNPRUNED model collapses to 28.6" compared iso-byte against "W4 + 50% REAP = 77.6". Under that reading the table is an iso-byte comparison at 12.5% and the conclusion is "prune 50% at 4-bit beats quantize to naive 2-bit", not "W2 + pruning collapses". If the "64" is literal, the point is 1.0 bit/original weight and is not iso-byte with anything in the table. Either way it says nothing about SOTA 2-bit codebooks on pruned models.
- On the 0% row the data are consistent across four model families: 4.0-4.6 bpw costs 0.7-3%; 3.3-3.6 bpw costs 0.3-4.5%; ~3.0 bpw costs 5-12% on hard code tasks (Aider) but ~0 on HE+/MBPP+; 2.5-2.8 bpw is bimodal — near-lossless (-0.9%) when attention/dense/shared tensors are protected (UD-Q2_K_XL on R1) and -14 to -22% when they are not (naive Q2_K_L on V3, IQ2_XXS/IQ1_M on V3.1 Aider); <= 2.3 bpw is a cliff everywhere (PPL +41%, Aider -29%, naive INT2 -40 to -65%).
- Pruning cost steepens between 25% and 50% on every model: Qwen3-30B MC -7.8% -> -30%; Kimi-K2 BFCL -4.7% -> -15%, MC -0.9% -> -17.6%; MAESTRO RP 98-99% -> 89-93%. Code benchmarks are the exception (near-flat to 50%) because the calibration corpus is code. There is no public point beyond 50% with quantization and none at 60-70% with any evaluation, so that part of the grid is an extrapolation of a convex curve.
- Internal (project memory, no URL): on GLM-5.3-Flash REAP50 the pruning step cost +21% ppl while a full GGUF step (Q4_K_S -> Q3_K_M) cost +1.46% and NVFP4 +1.56%, i.e. pruning was ~14x more damaging per unit compression than quantization at the 4-bit end. That matches the public 0% row (quant cheap down to ~3.3 bpw) and the public 50% row (prune expensive on general tasks).
- Budget arithmetic for the stated target: 117 GiB = 125.6 GB. For 763B params that is 1.32 bits per ORIGINAL weight; for 1.03T it is 0.98. Candidate products (prune fraction kept x bpw on survivors): 50% x 2.6, 60% x 3.3, 70% x 4.4 for 763B; 50% x 2.0, 60% x 2.5, 70% x 3.3 for 1T — before KV cache and activations. Every one of these lies in the n/m region of the grid.

### Gaps
- No published measurement of any static expert-pruned MoE (>= 25%) at < 4 bits with a SOTA quantizer. The two grids that exist (REAP A5, 2607.16721) both bottom out at Q4/W4 on the pruned side.
- No published 60% or 70% static expert-pruning point with evals for any model (HOPE reports up to 40-50%; REAP up to 50%; the 44% Quasar has no numbers).
- The alphaxiv/search snippets for HOPE 40% could not be verified against the full table (PDF not fetched).
- REAM and MoE-Pruner contain no prune x bit-width grids (MoE-Pruner is unstructured weight pruning inside experts, not expert removal).

---

## KQ2. Is the W2 + 50% collapse specific to naive INT2, or does it hold for SOTA 2-bit (QTIP/EXL3, AQLM, IQ2_K/IQ1_KT, spectral mixed precision)?

### Takeaway
The collapse in REAP A5 is a naive-uniform-INT2 artefact: on the same Qwen3-30B-A3B, unpruned, GPTQ/HQQ INT2 lose 40% while BitsMoE at the same 2.0 average bits loses 10%, and on DeepSeek-R1 a 2.53-bpw dynamic GGUF loses 0.94%. But every SOTA 2-2.5-bit datapoint is on an UNPRUNED model; there is no measurement of a SOTA 2-bit codebook on a 50%-pruned MoE, so "it also holds for SOTA 2-bit" is neither confirmed nor refuted.

### Cited Findings
- Qwen3-30B-A3B, 0% prune, 7-task 0-shot avg: FP16 71.61; GPTQ 2-bit 42.65; HQQ 2-bit 43.18; BitsMoE 2.0-bit 64.29 (GSM8K 71.04 vs GPTQ 4.32); GEMQ 2.0 is 2.80 points below BitsMoE (= 61.49) — [BitsMoE](https://arxiv.org/pdf/2606.00079)
- Mixtral-8x7B 0-shot avg at 2.0 bits per expert (attention 4-bit, GPTQ underneath): Uniform 55.72; PMQ (MC-MoE) 60.38; GEMQ 59.98; at 2.5 bpe: Uniform 65.49, PMQ 64.34, GEMQ 65.13; at 1.5 bpe: Uniform 47.45, GEMQ 52.00. SpQR 1.5-bit: WT2 PPL Inf, 31.87 avg — [GEMQ arXiv 2605.23078](https://arxiv.org/html/2605.23078v1)
- DeepSeek-R1 UD-Q2_K_XL (212 GB, 2.53 true bpw; routed expert gate/up q2_k, down_exps 94.8% q2_k + 5.2% q3_k, attention q4-q6, dense/shared q4-q6): 9-benchmark drop 0.94%, "outperforms standard 3-bit quantization (Q3_K_M) across multiple benchmarks". DeepSeek-V3 naive Q2_K_L (228 GB, attn_kv_b/q_a/q_b at q2_k, dense ffn q2_k, shared q2_k): drop 14.66% — [arXiv 2505.02390 Tables 2, 4, 7](https://arxiv.org/pdf/2505.02390)
- Unsloth DeepSeek-R1 Flappy-Bird test, 3 seeds: dynamic IQ1_S 131 GB avg 6.92 vs basic IQ1_S 133 GB 0 ("Repeats 'with Dark Colurs' forever"); dynamic IQ1_M 158 GB 9.08 vs basic 149 GB 1.67 ("Black screen only"); dynamic IQ2_XXS 183 GB 9.17 vs basic 175 GB 6.17 — [Unsloth R1 dynamic page](https://unsloth.ai/docs/basics/deepseek-r1-dynamic-1.58-bit)
- Kimi-K2-Thinking ik_llama trellis/K-quant family, PPL vs Q4_X 2.0818: IQ2_KL 2.920 bpw +14.0%; smol-IQ2_KL 2.755 +17.9%; smol-IQ2_KS 2.261 +41.0%; smol-IQ1_KT (trellis) 1.832 +72.6% — [ubergarm/Kimi-K2-Thinking-GGUF](https://huggingface.co/ubergarm/Kimi-K2-Thinking-GGUF)
- EXL3 (QTIP variant, tail-biting trellis) on Step-3.5-Flash: KL 0.653 at 2.00 bpw vs 0.142 at 3.00 and 0.053 at 4.00 — [turboderp/Step-3.5-Flash-exl3](https://huggingface.co/turboderp/Step-3.5-Flash-exl3/blob/main/README.md); ExLlamaV3 docs describe EXL3 as a QTIP variant and "much faster to produce than AQLM" — [ExLlamaV3 docs](https://docsearch.algolia.com/mcp/docs/repo/turboderp-org/exllamav3)
- The only EXL3 release of a REAP-pruned model found (ArtusDev, cerebras Qwen3-Coder-REAP-25B-A3B) lists 2.5-8.0 bpw files (2.5_H6 8.57 GB ... 4.0_H6 13.13 GB) with NO quality metrics — [ArtusDev REAP-25B EXL3](https://huggingface.co/ArtusDev/cerebras_Qwen3-Coder-REAP-25B-A3B-EXL3/blob/main/README.md)
- TriS-MoE (NeurIPS 2026): "at near-1-bit weights, scaling factors, bitmaps, and grouping indices can push the effective precision to about 3 bits" — [NeurIPS 2026 poster](https://neurips.cc/virtual/2026/poster/153046)
- KBVQ-MoE (ICLR 2026, vector quantization, KLT-SVD shared across experts): Qwen1.5-MoE-A2.7B 3-bit avg 67.99 vs FP16 68.07; no 2-bit or large-model numbers in the abstract — [KBVQ-MoE](https://mlanthology.org/iclr/2026/xu2026iclr-kbvqmoe/)
- Dense-model reference for what SOTA 2-bit codebooks achieve WITHOUT pruning: Llama-2-70B at ~2.0 bits, 0-shot avg vs FP16 72.4: AQLM 2.07 68.75 (fine-tuned), QuIP# 2.01 68.60, PV-Tuning 2.07 70.72; GPTQ 2-bit g128 34.38 — [PV-Tuning](https://arxiv.org/html/2405.14852v1); [EfficientQAT Tables 1/17](https://arxiv.org/html/2407.11062v3)

### Inferences
- Three independent lines (BitsMoE vs GPTQ on Qwen3-30B; UD-Q2_K_XL vs Q2_K_L on DeepSeek; dynamic vs basic IQ1 on R1) show that at a fixed ~2-2.5 average bpw the loss is 10-40x smaller when (a) attention/dense/shared tensors are protected and (b) bits are allocated non-uniformly across experts. The REAP A5 W2 row did neither (uniform AWQ g128 on all linears). So the 28.6 is a naive-INT2 artefact, not a property of the 2-bit regime.
- However, the SOTA-2-bit numbers are all 0%-prune. The two mechanisms that make them work — cross-expert redundancy (BitsMoE shared basis, KBVQ shared KLT components, GEMQ global allocation across many experts) and low-saliency experts absorbing 1-bit — both shrink after 50% pruning removes exactly the redundant/low-saliency experts. Expect the pruned model's 2-bit loss to sit between the unpruned SOTA figure (-10%) and the naive figure (-40%); this is an extrapolation, not a measurement.
- At 2.0-2.3 true bpw (which is where a 50%-pruned 1T model must land for 117 GiB) even the best 0%-prune results are a cliff: Kimi-K2 +41% PPL at 2.26 bpw, V3.1 -29% Aider at 2.03 bpw, Mixtral/Qwen3 -10% at 2.0 nominal (2.3 true). The usable floor for an unpruned frontier MoE with dynamic allocation is ~2.5 true bpw (R1 -0.94%, V3.1 IQ1_M 2.46 -22% on Aider — task-dependent).
- Internal (project memory): INT2 is a decode LOSS on Thor (no FP4->FP8-style hardware path for 2-bit), so a 2-bit expert format would also need a custom dequant kernel; EXL3/QTIP trellis kernels are decode-friendly but that is a separate feasibility question not covered here.

### Gaps
- No published evaluation of any pruned MoE at 2-2.5 bpw with EXL3/QTIP, AQLM, IQ2_K/IQ1_KT or spectral mixed precision. The ArtusDev REAP-25B EXL3 repo is the only artefact and has no evals.
- No KL/PPL for EXL3 on Qwen3-235B, DeepSeek-V3 or Kimi-K2 found (Step-3.5-Flash is the only large-MoE EXL3 table found).
- No AQLM or QTIP paper results on any MoE > 50B found.

---

## KQ3. Does healing AFTER quantization (PV-Tuning, EfficientQAT, QLoRA-style, router-only) rescue the collapsed region?

### Takeaway
On dense models, quantization-aware fine-tuning turns a 2-bit collapse (GPTQ 34.4 on Llama-2-70B) into a -2 to -4% result (PV-Tuning 70.72, EfficientQAT 69.48 vs 72.41). On MoE the only measured recoveries are router-only fine-tuning (GEMQ, 1.5-bit Mixtral PPL 9.66 -> 7.69) and LoRA/distillation healing of PRUNED-but-unquantized models (+1.2 to +3 points); no paper measures healing of a model that is both expert-pruned and sub-4-bit.

### Cited Findings
- EfficientQAT 2-bit (Block-AP then E2E-QP, 4096 RedPajama samples per phase): Llama-2-70B W2g64 0-shot 69.48 vs FP16 72.41, WikiText2 4.52 vs 3.32; Llama-3-70B W2g64 67.89 vs 75.33 (PPL 6.07 vs 2.85); Llama-3-8B 60.76 vs 68.58; GPTQ 2-bit g128 Llama-2-70B 34.38 (appendix Table 17); cost 41 h on one A100-80GB for 70B; beats QA-LoRA by +5.1/+4.0 (7B/13B MMLU) and PEQA by +4.5/+8.7 — [EfficientQAT](https://arxiv.org/html/2407.11062v3)
- PV-Tuning (vector-quant fine-tuning): Llama-2-7B 2.02 bits 0-shot 61.35 vs AQLM 56.47 / QuIP# 57.51 / FP16 64.80; 70B 2.07 bits 70.72 vs 68.75/68.60/72.40; Llama-3-70B 2.07 bits 70.38 vs QuIP 48.71 / FP16 75.37; Llama-3-8B 2.01 bits 64.36 vs QuIP 36.8. Ablation Llama-2-7B AQLM 2.01: calibration-only 53.2 -> continuous-params-only 56.57 -> PV+STE 58.19. Cost up to 1.5x QuIP# fine-tuning wall-clock; 70B needs 8xA100 or RAM offload. 1-bit: Llama-2-70B 1.14 bits 64.58 — [PV-Tuning](https://arxiv.org/html/2405.14852v1)
- RILQ (search snippet): "LoRA-based quantization error compensation has underperformed in sub-4-bit scenarios" — [search result, RILQ](https://arxiv.org/pdf/2407.11062v3) (indirect; full RILQ paper not fetched)
- GEMQ router-only fine-tuning (attention and expert weights frozen, routers < 0.04% of params), Mixtral-8x7B at 1.5 bits per expert, WT2 / C4 PPL: no calibration 9.66 / 23.53; FP router logits 9.29 / 22.72; router fine-tuned 7.69 / 17.18 — [GEMQ](https://arxiv.org/html/2605.23078v1)
- MoE-Pruner (50% UNSTRUCTURED sparsity inside experts, Mixtral-8x7B 9-task avg): pretrained 69.16; pruned 67.23; after expert-wise KD on 1000 C4 samples, 3 epochs, "1 hour" on 2x8 H100: 68.40 (gap 1.93 -> 0.76 points) — [MoE-Pruner](https://arxiv.org/html/2410.12013v1)
- MAESTRO recovery fine-tune (LoRA r=16 on attention projections only, experts and router frozen, 2000 SlimOrca conversations x 1024 tokens, 1 epoch, lr 2e-4): ablating RFT costs a 20.59% relative drop in average retained performance and 1289% higher cross-task std; "MAESTRO -RFT outperforms REAP -RFT by almost 1%"; with RFT, REAP 50% retains 89.37% (GPT-OSS-20B) / 90.25% (Qwen3-30B) — [MAESTRO PDF](https://arxiv.org/pdf/2607.08601)
- Route-Consistent Adaptation (NeurIPS 2026): QAT for MoE at MXFP4 on Mixtral/DeepSeek/Qwen3 that replays BF16 routes, freezes routers, trains only expert-side adapters, plus load-aware low-rank compensation; abstract gives no numbers — [NeurIPS 2026 poster](https://neurips.cc/virtual/2026/poster/149368)
- MiLo (training-free, calibration-free low-rank compensators on INT3 HQQ, 20 alternating SVD iterations): Mixtral-8x7B W3 g64 0-shot avg HQQ 75.59 / GPTQ 77.20 -> MiLo-s1 79.56 (+0.3 GB, 20.8 vs 20.5 GB); WikiText2 4.61 -> 4.03; MMLU 60.93 -> 67.07. DeepSeek-MoE-16B avg 71.77 -> 75.00. INT3 compensators at rank 64 cost 424 MB on Mixtral — [MiLo](https://arxiv.org/html/2504.02658v1)
- Quasar-438B used "a healing pass" after 44% expert pruning with FP8/NVFP4 QAT, token count not published — [Multiverse](https://multiversecomputing.com/papers/inside-quasar-438b-by-multiverse-computing)

### Inferences
- The dense evidence is unambiguous that end-to-end QAT/PV-style fine-tuning of the quantized representation (not LoRA on top of frozen quantized weights) moves the 2-bit cliff from "collapse" to "-2 to -5%", and that the gain is largest exactly where PTQ is worst (Llama-3-70B: QuIP 48.7 -> PV 70.4). There is no reason in the mechanism for this to fail on MoE experts, but nobody has published it on a > 50B MoE, and the 70B dense cost (8xA100, 41 h single-A100 for EfficientQAT) scales with total params, i.e. ~10-15x for a 763B-1T model.
- Cheap partial recoveries exist for MoE and stack: router-only FT recovers ~20% of the 1.5-bit PPL excess (GEMQ); attention-only LoRA recovers the pruning perturbation (MAESTRO: dropping it costs 20% retained performance); MiLo's SVD compensators recover ~4 points at 3-bit for ~1.5% extra bytes with no training. None of these is measured on a pruned+sub-4-bit model.
- Internal (project memory): FP8 teacher logits for GLM-5.3-Flash are already banked, so KD-style healing on the pruned+quantized model is feasible without re-running the teacher; and a draft head is quant-specific (hidden-state cosine 0.89 across quants), which argues for healing the FINAL quantized model, not an intermediate one.

### Gaps
- No measurement anywhere of healing a model that is both expert-pruned (>= 25%) and < 4-bit. This is the single largest hole in the literature relative to the Thor plan.
- Route-Consistent Adaptation numbers (the only MoE QAT paper found) are not in the abstract; full paper not fetched.
- No PV-Tuning / EfficientQAT / AQLM result on any MoE found.

---

## KQ4. Mixed precision by saliency: 4-bit for high-saliency experts, 2-3-bit for low-saliency ones — savings and quality

### Takeaway
Expert-wise mixed precision is the best-evidenced lever below 3 bits: at the same average bits it recovers 20-25 points of 0-shot accuracy over uniform INT2 (Mixtral 42.67 -> 67.50 at 2.54; Qwen3-30B 42.65 -> 64.29 at 2.0), and every method keeps attention/gating/shared tensors at >= 4-bit, costing ~0.05 bits on average. The saliency signals used (activation frequency, router weights, quantization error, spectral energy) are the same signals REAP uses to prune, so after 50% pruning the low-saliency experts that would have absorbed 1-2 bits are mostly gone.

### Cited Findings
- MC-MoE PMQ: expert bits chosen from {1,2,3} by integer programming on activation/importance; attention + gate 4-bit "adds no more than 0.05 bits"; Mixtral-8x7B avg-bit vs 0-shot: 2.54 67.50 / 2.05 63.25 / 1.57 54.49 vs uniform GPTQ 2.00 42.67 (FP16 71.29); 8x22B 2.54 72.08 / 2.05 67.94 vs uniform 2.00 50.44 (FP16 76.33). WikiText2: uniform 2-bit 16.38 vs PMQ 2.54 5.09, 2.05 5.91 — [MC-MoE](https://arxiv.org/html/2410.06270)
- GEMQ: global LP over all experts in the model using quantization-error analysis, candidate {1,2,3} bits, attention 4-bit, router FT, progressive quantization; Mixtral 2.5 bpe 65.13 / 2.0 59.98 / 1.5 52.00 vs uniform 65.49 / 55.72 / 47.45 (FP 70.97); Qwen3-30B-A3B at 1.5 bpe: PMQ WT2 20.85 avg 42.08 vs GEMQ 12.16 / 52.45 — [GEMQ](https://arxiv.org/html/2605.23078v1)
- BitsMoE: shared-basis spectral decomposition across experts + per-component ILP; Qwen3-30B-A3B 2.0-bit avg 64.29 (GEMQ 61.49, GPTQ 42.65, HQQ 43.18); 3-bit 70.57 vs GPTQ 68.99 (FP16 71.61); 16.47x faster offline quantization than GEMQ, up to 6.46x GPTQ decode throughput — [BitsMoE](https://arxiv.org/pdf/2606.00079)
- EAC-MoE: experts 2/2.5/3 bit with attention 4-bit and router FP give true averages 2.06 / 2.54 / 3.03 bits; router re-calibration (QESC) fixes expert-selection drift; Mixtral 3.03-bit 72.21 vs FP 72.64 — [EAC-MoE](https://arxiv.org/html/2508.01625v1)
- Router-shift allocation (OpenReview, Switch/Mixtral): experts whose router weights changed least during training get more bits; high intra-neuron variance experts also get more bits — [OpenReview PDF](https://openreview.net/pdf/dd52b79450b90e19483ff245a69f11e72dfe55c0.pdf)
- AlphaQ (ICLR 2026): calibration-free bit allocation for MoE, motivated by calibration data being a biased stand-in for the training distribution — [ICLR 2026](https://iclr.cc/virtual/2026/10017148)
- Unsloth practice on DeepSeek-V3-0324: 2.42-bit IQ2_XXS or 2.71-bit Q2_K_XL "stressing per-tensor bit allocation in MoE down_proj layers" — [search snippet, promptlayer listing](https://www.promptlayer.com/models/deepseek-v3-0324-gguf); the arXiv 2505.02390 Table 7 shows UD-Q2_K_XL's ffn_down_exps split 94.8% q2_k / 5.2% q3_k, i.e. per-layer mixed precision within the routed experts — [arXiv 2505.02390 Table 7](https://arxiv.org/pdf/2505.02390)
- DQ3_K_M (arXiv 2505.02390): dynamic 3-bit with ffn_down_exps 75.9% q3_k / 20.7% q4_k / 3.4% q6_k, attention q4-q6, shared experts q4-q6; 281 GB (3.35 true bpw) with -0.34% (R1) / -0.30% (V3), better than uniform Q3_K_M (-1.80/-1.85%) and matching Q4_K_M (-0.68/-1.35%) at 25% fewer bytes — [arXiv 2505.02390](https://arxiv.org/pdf/2505.02390)
- MxMoE (ICML 2025) is mixed-precision quantization for MoE with accuracy/performance co-design; it is quantization-only — [MxMoE](https://proceedings.mlr.press/v267/duanmu25a.html)
- MODE (arXiv 2606.17118): modality-decomposed expert-level mixed precision for multimodal MoE — [MODE](https://arxiv.org/pdf/2606.17118) (not fetched; title/abstract only)

### Inferences
- Measured savings at ~2.5 average bits on Mixtral-class models: -5% (MC-MoE 2.54, GEMQ 2.5, EAC-MoE 2.56) vs -8% uniform at the same bits and -40% uniform at 2.0. At ~2.0-2.3 true bits the best mixed-precision methods lose 10-15% (0-shot avg) on both Mixtral and Qwen3-30B — that is the frontier at 0% prune.
- Mixed precision and REAP compete for the same signal. REAP ranks experts by mean gated activation norm; MC-MoE/GEMQ/BitsMoE give the lowest bits to low-activation/low-error experts. After 50% REAP, the surviving experts are the top half of that distribution, so the spread that mixed precision exploits is roughly halved. Expect a smaller mixed-precision dividend on the pruned model than the -5%-at-2.5-bits figure above (extrapolation).
- The "cheap" form of mixed precision — keep all non-expert tensors at 4-8 bit and only drive routed expert gate/up to 2-bit while keeping down_proj a step higher — is exactly UD-Q2_K_XL / DQ3_K_M, and it is the form with the best measured large-model numbers (R1 -0.94% at 2.53 true bpw). That form does not depend on cross-expert redundancy and should survive pruning better than global-LP schemes (inference).

### Gaps
- No mixed-precision expert quantization paper evaluates a statically pruned model, and none evaluates a model > 141B (Mixtral 8x22B) except via GGUF practice (DeepSeek/Kimi).
- GEMQ/BitsMoE true bytes including scales/indices are not reported; TriS-MoE warns metadata can be ~1 bit at the 1-2-bit end.

---

## KQ5. Order: prune->heal->quant vs prune->quant->heal vs quant->prune — measured comparisons

### Takeaway
No MoE-expert ordering experiment exists. The one systematic study (ICLR 2026, dense LLaMA/ViT) finds order irrelevant at >= 6 bits and strongly favours "weaker perturbation first" at 4-bit (prune-then-quantize wins by 49.9 negative-PPL points for SparseGPT at 5% sparsity + 4-bit); REAP itself quantized first (W4A16 checkpoint) and pruned second because pruning needs no re-quantization, without comparing.

### Cited Findings
- "Prune-then-Quantize or Quantize-then-Prune?" (ICLR 2026, arXiv 2603.18426): Progressive Intensity Hypothesis — "weaker perturbations should precede stronger ones"; advantage grows monotonically with the compression-equivalent-ratio gap. Table 1 (LLaMA, 5% pruning + QuaRot, metric = negative PPL, advantage = M(Q then P) - M(P then Q), negative favours prune-first): 9-bit +0.002 / 0; 8-bit +0.001 / 0; 7-bit -0.003 / 0; 6-bit -0.013 / 0; 5-bit -0.017 / -0.057; 4-bit -49.899 (SparseGPT) / -9.379 (SLEB). Pruning AFTER QuaRot rotation causes "a drastic drop" that widens with sparsity. LoRA after quantization "effectively compensat[es] quantization-induced performance loss". Models: LLaMA-2 7B/13B, LLaMA-3 8B, Mistral, ResNet-18, DeiT-B; no MoE — [arXiv 2603.18426](https://arxiv.org/html/2603.18426)
- REAP: "pruning can be easily applied to quantized models without any additional steps required to reconcile block scales or re-quantize following compression"; expert MERGING "necessitates re-quantization for block quantization formats"; Kimi-K2 pipeline = RedHat W4A16 checkpoint -> REAP; Qwen3-30B A5 = AWQ W4 -> REAP — [REAP HTML v3](https://arxiv.org/html/2510.13999v3)
- QP-SNN (spiking nets, not MoE): quantize-first chosen because pruning on full-precision weights "can select kernels that become misaligned after quantization" — [arXiv 2502.05905](https://arxiv.org/pdf/2502.05905)
- MAESTRO and MoE-Pruner both do prune -> heal (LoRA on attention / expert-wise KD) with no quantization stage — [MAESTRO](https://arxiv.org/pdf/2607.08601); [MoE-Pruner](https://arxiv.org/html/2410.12013v1)
- GEMQ does quant -> router-FT inside a progressive loop — [GEMQ](https://arxiv.org/html/2605.23078v1)
- Quasar-438B: prune -> heal, with "quantization-aware compression" for FP8/NVFP4 (order relative to healing not stated) — [Multiverse](https://multiversecomputing.com/papers/inside-quasar-438b-by-multiverse-computing)

### Inferences
- Under the Progressive Intensity Hypothesis the stronger perturbation should come last. At the Thor operating point the pruning step is the stronger perturbation (internal: +21% ppl at REAP50 vs +1.5% at NVFP4), which argues quant -> prune — the order REAP already used — UNLESS the quantizer goes to <= 2.5 bpw, where its perturbation (+14-41% PPL on Kimi-K2-Thinking) rivals or exceeds pruning, flipping the recommendation to prune -> quant. That flip is an inference from a dense-model hypothesis, not a measurement.
- Practical constraints decide more than the hypothesis: REAP needs full-precision expert outputs for saliency (so prune on a BF16/FP8 model, or at least calibrate saliency there); block-scaled formats (NVFP4, GGUF k-quants) are unaffected by expert removal so prune-after-quant costs nothing; QAT/PV-Tuning must be LAST because it trains the quantized representation; and the draft head must be trained on the final served quant (internal hidden-state drift finding). Net: calibrate saliency on FP8 -> prune -> (LoRA/KD heal, optional) -> quantize -> QAT/PV heal -> train draft head. No paper has run this chain.

### Gaps
- No MoE-specific ordering experiment with recovery numbers found; the ICLR 2026 paper's appendices (not read beyond 100k chars) may contain more LLMs but not MoE.
- No published prune -> quant -> heal pipeline with numbers on any MoE.

---

## KQ6. Iso-byte: prune 60% at 4-bit vs prune 40% at ~2.7-bit (and the general "prune more, quantize less" question)

### Takeaway
Every measured iso-byte comparison says: quantize first down to ~3 true bpw with protected attention/dense tensors (cost 0-5%), and only then prune; below ~3 bpw a 50%-pruned 4-5-bit model beats the unpruned 2-3-bit model on HE+ but loses on MBPP+/general. The specific 60%@4-bit vs 40%@2.7-bit question has no measured point on either side; the curves extrapolate to both being 15-25% below baseline on general tasks, with 40%@2.7-dynamic the less risky because its 0%-prune component (2.7 bpw dynamic: -1 to -15%) and its 40% component (HOPE -4%, REAP -12%) are each measured.

### Cited Findings
- REAP A5 iso-byte at 12.5% of BF16 bytes (Qwen3-30B-A3B, EvalPlus): W4 + 50% REAP 77.6 vs naive W2 (unpruned per prose) 28.6; at 25%: W4 + 0% 80.5; at 50%: W16 + 50% REAP 78.0 (quantization "outperforms a 50% expert-pruned REAP model at 16-bits, despite having half the checkpoint size") — [REAP App. E](https://arxiv.org/pdf/2510.13999)
- Qwen3.6-35B-A3B at ~13 GB (computed here as bits per ORIGINAL weight = GB x 8 / 35B): pruned-50% Q5 (3.04 b/orig-w) HE+ 0.915 / MBPP+ 0.749; base IQ2_M (2.97) 0.896 / 0.730; base Q2_K (3.09) 0.848 / 0.754. At ~17-20 GB: base Q3_K_M (3.91) 0.890 / 0.775 vs pruned-50% Q8 (4.51) 0.896 / 0.728 — MBPP+ favours quant-only by 4.8 (p=0.0009). Paper's rule: "quantize down to 3 bits before pruning anything, and prune below that boundary" — [arXiv 2607.16721](https://arxiv.org/html/2607.16721)
- DeepSeek-R1 at 2.53 true bpw (UD-Q2_K_XL, 0% prune): -0.94% over 9 benchmarks; DeepSeek-V3-0324 naive Q2_K_L 2.72 bpw: -14.66% — [arXiv 2505.02390](https://arxiv.org/pdf/2505.02390)
- DeepSeek-V3.1 Aider Polyglot at 0% prune: 2.68 bpw -14.5%; 2.80 -10.2%; 3.04 -8.1%; 3.58 -4.5%; 4.61 -2.7% — [Unsloth Aider](https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs/unsloth-dynamic-ggufs-on-aider-polyglot)
- Pruning-only cost at 40%: HOPE -4.0% / REAP -11.7% (3-model mean, snippet) — [HOPE](https://arxiv.org/pdf/2609.18916); at 50% REAP on Kimi-K2 (W4): MC -17.6%, BFCL -15.3%, Eval+ -1.1% — [REAP Table 3](https://arxiv.org/html/2510.13999v3)
- Kimi-K2-Thinking 0% prune: 2.755 bpw PPL +17.9%; 2.920 +14.0%; 3.249 +7.4%; 4.059 +2.5% — [ubergarm](https://huggingface.co/ubergarm/Kimi-K2-Thinking-GGUF)

### Inferences
- Byte accounting for the user's two options (bits per original weight, experts only, ignoring the fixed attention/embedding share): 60% prune x 4.25 bpw (W4 g128 true) = 1.70; 40% prune x 2.7 = 1.62. They are iso-byte to within 5%.
- Option A (60% @ 4-bit): the quant component is measured and cheap (-0.7 to -2.7%); the prune component is unmeasured and on a steepening curve (25% -> 50% roughly triples the general-task loss on every model). Linear extrapolation of REAP MC loss on Kimi-K2 (-0.9% at 25%, -17.6% at 50%) to 60% gives ~-24%; convex extrapolation is worse. HOPE's better 40% point (-4%) does not have a 60% datum.
- Option B (40% @ 2.7 dynamic): prune component -4% (HOPE) to -12% (REAP) measured at 40%; quant component at 2.7 true bpw measured on three frontier MoEs as -1% (R1, 9 broad benchmarks) to -14% (V3.1 Aider, V3 naive). If the two losses compound multiplicatively: -5% to -24%, with the low end requiring the UD-style protected-tensor recipe and the broad benchmark mix.
- Both options are extrapolations. The measured evidence (REAP A5, 2607.16721) consistently prefers spending bytes on quantization down to ~3 bpw before pruning, and 2607.16721 finds the crossover at exactly 3 bpw — so a third option, 50% @ 3.0-3.3 dynamic (= 1.5-1.65 b/orig-w, same budget), has BOTH components measured (REAP50: code ~-1%, general -17%; 3.0-3.3 bpw: -0.3 to -8%) and is the lowest-risk iso-byte point.
- Internal (project memory): on GLM-5.3-Flash the same two axes were measured (REAP50 +21% ppl vs one GGUF step +1.5%), which is why the project rule is "keep prune at 50%, spend on the quant axis" — consistent with the public evidence, but the public evidence also shows the quant axis stops being cheap at ~2.5 bpw.

### Gaps
- No measured point for any static prune ratio > 50% with any quantization, and none for 40% with < 4-bit. The iso-byte question in the brief cannot be answered from measurements, only from the two single-axis curves.
- Bits per original weight above exclude the attention/embedding/shared share, which is model-specific and must be added from the safetensors index (internal rule: NVFP4 lands at ~0.62 B/w, i.e. ~5 bpw all-in, not 4.5).

---

## KQ7. Attention / dense / embedding / shared-expert sensitivity: what bits are safe on MLA-class models

### Takeaway
Every successful sub-3-bit recipe on DeepSeek/Kimi (MLA) keeps attention (especially attn_k_b / kv_b), the first dense FFN layers, shared experts, embeddings and output at 4-8 bits and sends only routed experts to 2-3 bits; the single naive recipe that put attention and dense FFN at q2_k (DeepSeek-V3 Q2_K_L) lost 14.7% while the protected one at fewer total bytes lost 0.9%. Academic mixed-precision papers uniformly fix attention at 4-bit. No data on DSA or KDA layers.

### Cited Findings
- DeepSeek-R1/V3 per-tensor recipes (arXiv 2505.02390 Table 7): UD-Q2_K_XL (-0.94%): output q6_k, token_embd q4_k, attn_kv_a_mqa q6_k, attn_kv_b q6_k, attn_output q4_k, attn_q_a/q_b q4_k, dense ffn_down q6_k / gate,up q4_k, ffn_down_shexp q6_k, gate/up_shexp q4_k, routed gate/up_exps q2_k, routed down_exps 94.8% q2_k + 5.2% q3_k. Q2_K_L (-14.66%): attn_kv_b q2_k, attn_q_a/q_b q2_k, attn_output q3_k, dense gate/up q2_k, shexp gate/up q2_k, routed gate/up q2_k, routed down q3_k. DQ3_K_M (-0.3%): attention q4-q6, shared q4-q6, routed gate/up q3_k, routed down 75.9% q3_k / 20.7% q4_k / 3.4% q6_k — [arXiv 2505.02390](https://arxiv.org/pdf/2505.02390)
- Kimi-K2-Thinking (ubergarm): "Every recipe keeps these at q8_0: attention tensors (attn_k_b, attn_v_b, attn_kv_a_mqa, attn_q_a, attn_q_b, attn_output), the first dense layer's ffn_down/gate/up, and the shared expert's ffn_down/gate/up"; embeddings iq4_k, output iq6_k; only routed experts vary from iq4_kss down to iq1_kt — [ubergarm/Kimi-K2-Thinking-GGUF](https://huggingface.co/ubergarm/Kimi-K2-Thinking-GGUF)
- Unsloth DeepSeek-V3.1 ablation: keeping attn_k_b at 4-bit instead of 8-bit "cost accuracy" on Aider; raising those tensors adds "about 100MB, or under 0.1% of model size" — [Unsloth Aider page](https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs/unsloth-dynamic-ggufs-on-aider-polyglot)
- Unsloth R1: "basic" (uniform) IQ1_S/IQ1_M fully failed (looping text / black screen) while the dynamic recipes at the same size worked — [Unsloth R1 dynamic](https://unsloth.ai/docs/basics/deepseek-r1-dynamic-1.58-bit)
- MC-MoE: "attention and gating weights are quantized to 4-bit" while experts get 1-3 bits — [MC-MoE](https://arxiv.org/html/2410.06270); GEMQ: "quantize all attention modules to 4 bits" — [GEMQ](https://arxiv.org/html/2605.23078v1); EAC-MoE: attention 4-bit, router full precision — [EAC-MoE](https://arxiv.org/html/2508.01625v1)
- MoQE (Microsoft, 2023): 2-bit weight-only quantization applied ONLY to expert weights; "expert layers ... proved much more robust to quantization than conventional feedforward layers" — [MoQE](https://arxiv.org/html/2310.02410v1)
- TriS-MoE (NeurIPS 2026): architecture is "a shared high-precision input-side backbone with binarized remaining components" — [NeurIPS 2026 poster](https://neurips.cc/virtual/2026/poster/153046)
- EXL3 REAP-25B release uses head (lm_head) at 6 bits for every 2.5-6.0 bpw file and 8 bits for the 8.0 file — [ArtusDev](https://huggingface.co/ArtusDev/cerebras_Qwen3-Coder-REAP-25B-A3B-EXL3/blob/main/README.md)

### Inferences
- Safe floor for MLA attention on DeepSeek/Kimi from the recipes that work: q4_k minimum for q_a/q_b/output, q6_k-q8_0 for kv_a_mqa/kv_b/k_b (the low-rank compressed-KV projections are the most sensitive, consistent with Unsloth's attn_k_b ablation). Dense FFN layers and shared experts: q4-q8. Embeddings q4, output q6. Nothing in the literature supports taking any of these below 4-bit.
- These tensors are a small fraction of a 1T MoE (ubergarm's whole recipe reaches 1.83 bpw average with all of them at q8_0), so protecting them costs ~0.1-0.3 bpw of average — the cheapest insurance on the grid. The attention share grows after expert pruning (fixed numerator, smaller denominator), which raises the all-in bpw of a pruned model slightly; this is the reason NVFP4 checkpoints measure ~0.62 B/w rather than 0.5625 (internal measurement).
- For DSA (DeepSeek sparse attention indexer) and KDA (Gated-DeltaNet style linear attention) there is no public quantization-sensitivity data at all; by analogy with the MLA compressed-KV projections, the indexer and the delta-rule state projections should be treated as >= 6-bit until measured (internal: KDA is 47% of per-token bytes on GLM-5.3, so this matters for decode, not just quality).

### Gaps
- No published per-tensor sensitivity study for DSA indexers or KDA/linear-attention projections at any bit-width.
- The Unsloth R1 blog section explaining WHICH down_proj matrices were sensitive could not be fetched (403 on the blog, redirected docs page lacks that section); only the outcome table was recovered.
- No academic paper tests attention below 4-bit on an MoE and reports the breakdown.

---

## Source list (primary)
- REAP, arXiv 2510.13999 v3 (ICLR 2026) — https://arxiv.org/pdf/2510.13999 and https://arxiv.org/html/2510.13999v3
- Half the Experts, All the Code, arXiv 2607.16721 — https://arxiv.org/html/2607.16721
- Quantitative Analysis of Performance Drop in DeepSeek Model Quantization, arXiv 2505.02390 — https://arxiv.org/pdf/2505.02390
- MC-MoE, arXiv 2410.06270 — https://arxiv.org/html/2410.06270
- EAC-MoE, arXiv 2508.01625 (ACL 2025) — https://arxiv.org/html/2508.01625v1
- GEMQ, arXiv 2605.23078 (ICML 2026) — https://arxiv.org/html/2605.23078v1
- BitsMoE, arXiv 2606.00079 — https://arxiv.org/pdf/2606.00079
- MiLo, arXiv 2504.02658 (MLSys 2025) — https://arxiv.org/html/2504.02658v1
- MAESTRO, arXiv 2607.08601 — https://arxiv.org/pdf/2607.08601
- MoE-Pruner, arXiv 2410.12013 — https://arxiv.org/html/2410.12013v1
- HOPE, arXiv 2609.18916 — https://arxiv.org/pdf/2609.18916
- Prune-then-Quantize or Quantize-then-Prune?, arXiv 2603.18426 (ICLR 2026) — https://arxiv.org/html/2603.18426
- PV-Tuning, arXiv 2405.14852 — https://arxiv.org/html/2405.14852v1
- EfficientQAT, arXiv 2407.11062 — https://arxiv.org/html/2407.11062v3
- TriS-MoE (NeurIPS 2026) — https://neurips.cc/virtual/2026/poster/153046
- Route-Consistent Adaptation (NeurIPS 2026) — https://neurips.cc/virtual/2026/poster/149368
- KBVQ-MoE (ICLR 2026) — https://mlanthology.org/iclr/2026/xu2026iclr-kbvqmoe/
- MoQE, arXiv 2310.02410 — https://arxiv.org/html/2310.02410v1
- Unsloth Aider Polyglot tables — https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs/unsloth-dynamic-ggufs-on-aider-polyglot
- Unsloth R1 dynamic — https://unsloth.ai/docs/basics/deepseek-r1-dynamic-1.58-bit
- ubergarm Kimi-K2-Thinking-GGUF — https://huggingface.co/ubergarm/Kimi-K2-Thinking-GGUF
- unsloth Kimi-K2-Thinking-GGUF — https://huggingface.co/unsloth/Kimi-K2-Thinking-GGUF
- turboderp Step-3.5-Flash-exl3 — https://huggingface.co/turboderp/Step-3.5-Flash-exl3/blob/main/README.md
- ArtusDev REAP-25B EXL3 — https://huggingface.co/ArtusDev/cerebras_Qwen3-Coder-REAP-25B-A3B-EXL3/blob/main/README.md
- Multiverse Quasar-438B — https://multiversecomputing.com/papers/inside-quasar-438b-by-multiverse-computing
