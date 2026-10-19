# Ultra-low-bit trellis/codebook formats for a pruned, MXFP4-source DeepSeek-V4.1-Flash-1017 on one Thor

Researched 2026-10-19. Local prior notes used: `research_notes/Compressing 1T MoE for Thor/{extreme_quantization,per_expert_mixed_precision,prune_quant_pareto_frontier,structural_and_memory_budget,target_models,v41_architecture_pruning_delta,vision_retention_pruning}.md` (cited as LOCAL). Percent deltas computed by me from the published tables are marked "(my arithmetic)".

**Headline for the writer:**
1. The best evidence is now on the same model family. A 50-row PPL/KLD table for DeepSeek-V4-Flash-0731, whose experts are also FP4 at source, runs from 4.25 down to 1.81 expert-bpw. On it, 2.91 bpw costs +7.1% PPL, 2.48 costs +15.2% and 2.25 costs +21.3%.
2. An EXL3 (trellis) quant of V4.1-Flash-1017 itself, with experts at 3 bpw, costs only +0.4% prose PPL and +1.8% code PPL.
3. The 2.3-bpw / 256-expert option does not fit, because 256 experts need ≤2.24 average expert bpw. It is also worse than 192 @ ~2.9 on every number we have.
4. NVFP4 is the wrong output format for experts. At 4.5 bpw it is larger than the native MXFP4's 4.25.

## Q1. Trellis/codebook methods: measured quality at 2.0–2.5 bpw on large MoEs

### Takeaway
On large MoEs, the 2.0–2.5 bpw band costs about +15–45% wikitext PPL with GGUF i-quants or ik_llama IQ2_KS-class quants. Trellis types (IQ*_KT, EXL3/QTIP) are worth roughly 0.1–0.2 bpw over i-quants of equal error, which is not enough to move a 2.2-bpw model out of that band. At ~3 bpw the gap looks much larger. EXL3 at 3 bpw on V4.1-1017 costs +0.4–1.8%, while GGUF IQ3_XXS at 2.91 bpw on V4-Flash-0731 costs +7.1%, but the two use different corpora and baselines.

### Cited Findings
**Same model family, FP4-source experts (DeepSeek-V4-Flash-0731; reference = bit-exact BF16 dequant, PPL 4.5289; wikitext-2, ctx 5632, llama.cpp PR #24162, 8× RTX 5090).** The AtomicChat "AD" ladder is shown as expert-bpw / total GB / PPL / mean KLD / top-1:
- AD-MXFP4 4.25 / 154.5 / 4.5446 / 0.156 / 87.4%
- IQ3_M 3.71 / 135.8 / 4.5695 / 0.180 / 86.3%
- IQ3_XS 3.20 / 118.2 / 4.6657 / 0.206 / 85.4%
- IQ3_XXS 2.91 / 108.1 / 4.8491 / 0.250 / 83.8%
- IQ2_M 2.79 / 104.0 / 4.8822 / 0.257 / 83.6%
- IQ2_S_XL 2.58 / 96.8 / 5.1406 / 0.319 / 81.5%
- IQ2_S 2.48 / 93.4 / 5.2152 / 0.334 / 81.0%
- IQ2_XS 2.25 / 85.1 / 5.4917 / 0.395 / 79.2%
- IQ2_XXS 2.06 / 78.5 / 5.7878 / 0.454 / 77.5%
- IQ1_M 1.81 / 70.2 / 6.3813 / 0.564 / 74.5%

Sources: [AtomicChat/DeepSeek-V4-Flash-0731-GGUF](https://huggingface.co/AtomicChat/DeepSeek-V4-Flash-0731-GGUF); [AtomicChat/dsv4-eval-artifacts](https://huggingface.co/datasets/AtomicChat/dsv4-eval-artifacts)

- PPL delta vs BF16 for that ladder (my arithmetic): 4.25 +0.3%; 3.71 +0.9%; 3.56 +1.6%; 3.20 +3.0%; 2.91 +7.1%; 2.79 +7.8%; 2.58 +13.5%; 2.48 +15.2%; 2.25 +21.3%; 2.06 +27.8%; 1.89 +36.4%; 1.81 +40.9%. KLD above the MXFP4 row is 0.050 @3.20, 0.093 @2.91, 0.162 @2.58, 0.238 @2.25 and 0.298 @2.06. — computed from [AtomicChat](https://huggingface.co/AtomicChat/DeepSeek-V4-Flash-0731-GGUF)
- AtomicChat recipe, constant on every rung: Q8_0 on attn_output_a/b, attn_q_b, attn_kv, compressor, indexer q_b; Q6_K on attn_q_a, shared experts (`ffn_*_shexp`), token_embd and output; BF16 router; F32 for the hc_* streams. Gate/up types are split between 18 "higher-energy" layers (20, 26–38, 40–42) and the other 25. `ffn_down_exps` stays native MXFP4 on layers 41–42 in every rung. — [AtomicChat card](https://huggingface.co/AtomicChat/DeepSeek-V4-Flash-0731-GGUF)
- On MI300X, token generation is flat at 36.9–37.8 tok/s from IQ2_M to IQ3_M_XL. The card calls the difference smaller than run-to-run error. — [AtomicChat card](https://huggingface.co/AtomicChat/DeepSeek-V4-Flash-0731-GGUF)
- Other V4-Flash-0731 rows in the same dataset:
  - DevQuasar naive Q2_K: 103.1 GB, PPL 9.11, KLD 0.969. It collapses.
  - Q3_K_M: 135.3 GB, PPL 4.914 (+8.5%).
  - unsloth UD-Q2_K_XL: 96.8 GB, PPL 5.159.
  - unsloth UD-IQ2_XXS: 90.9 GB, PPL 5.416.
  - bullerwins "IQ3_XXS-Experts-Q8_0": 113.9 GB, PPL 4.730, KLD 0.218. Its "IQ2_XS-IQ3_XXS-GateUp-MXFP4-Down-Q8_0" variant is 120.3 GB with PPL 4.765 and KLD 0.222.

  — [AtomicChat/dsv4-eval-artifacts](https://huggingface.co/datasets/AtomicChat/dsv4-eval-artifacts)
- Measurement-floor caveat: native MXFP4 already shows KLD 0.156 and top-1 87.4% against the bit-exact BF16 reference, and Q4_K_M lands at 0.1556. A floor of ~0.155 is therefore runtime numerics (activation quantization in the MMQ path plus non-expert Q8/Q6 tensors), not expert error. — [AtomicChat/dsv4-eval-artifacts](https://huggingface.co/datasets/AtomicChat/dsv4-eval-artifacts)

**V4.1-Flash-1017 itself**
- LibertAI EXL3 2.9 bpw (exllamav3 1.6.2 dev, "mul1" codebook) has routed experts at 3 bpw, except layers 18–22 at 2 bpw. Shared experts are at 4–5 bpw, attention at 5 bpw (layer 0 wq_a/wkv at 6), indexer wk at 8 and head at 6. It is 196.3 GiB without Engram.
  - Held-out PPL (65,504 tokens): code 1.8749 → 1.9094 (+1.8%); prose 6.4277 → 6.4520 (+0.4%).
  - The EXL3 3.5-bpw build is 239.9 GiB: code 1.8895, prose 6.4341.
  - Calibration: 250 × 2048 tokens, "every routed expert sees every calibration token".
  - Upstream exllamav3 has no DeepseekV41 class, so it cannot load this checkpoint; the target runtime is a vLLM-based 2× DGX Spark kit.

  — [LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw)
- acrogenesis IQ2_XXS (all 384 experts, dense and Engram at Q8) reaches held-out PPL 7.57 vs 6.34 for an MXFP4/Q8 reference: +19.5%, SE 2.94 pp, mean KLD 0.365, top-1 76.9%, with only 2,040 scored positions. It passed 5/8 exact-answer checks against 8/8 for the reference. — [acrogenesis/DeepSeek-V4.1-Flash-1017-IQ2_XXS-GGUF](https://huggingface.co/acrogenesis/DeepSeek-V4.1-Flash-1017-IQ2_XXS-GGUF)

**Trellis types in ik_llama.cpp: design and efficiency**
- PR 113:
  - IQ2_KT: 16-bit seed per 8 weights, 32-weight blocks with 4-bit scales, about 2.125 bpw.
  - IQ3_KT: 12-bit seed per 4 weights, 3.125 bpw.
  - IQ4_KT: 15-bit seed per 4 weights, 8-bit block scales, 4.0 bpw.
  - Uses QTIP's "3INST" generator, with no Hadamard and no Viterbi search.
  - The author found about 0.2 bpw less than i-/iqk-quants at equal quantization error on LLaMA-3.1-8B. IQ2_KT on LLaMA-v2-7B gave PPL 6.36 vs f16 4.94, against QTIP's reported error of 33.2%.
  - Conclusion as quoted: trellis is "a small improvement, not the big leap the hype suggested".
  - The PR was superseded by #441/#453/#471/#475/#482.

  — [ik_llama.cpp PR 113](https://github.com/ikawrakow/ik_llama.cpp/pull/113)
- GLM-4.5 (355B, BF16 source, ubergarm; BF16 PPL 3.1788). Rows are total bpw / PPL / delta (my arithmetic):

  | Quant | Total bpw | PPL | vs BF16 |
  |---|---|---|---|
  | IQ4_KSS | 4.164 | 3.3261 | +4.6% |
  | IQ3_KT | 3.537 | 3.4369 | +8.1% |
  | IQ2_KL | 3.062 | 3.7569 | +18.2% |
  | IQ2_KS | 2.671 | 4.3196 | +35.9% |
  | IQ1_KT | 2.009 | 5.3270 | +67.6% |

  Recipes: IQ3_KT is routed gate/up iq3_kt with down iq4_kss. IQ1_KT is gate/up iq1_kt, down iq2_kt and attention iq4_kt. — [ubergarm/GLM-4.5-GGUF](https://huggingface.co/ubergarm/GLM-4.5-GGUF)
- Kimi-K2.5 (1T, INT4-QAT source, ubergarm; Q4_X baseline PPL 1.8235). Attention, shared experts and dense layer 0 are q8_0 throughout. Rows are total bpw / PPL / delta (my arithmetic):

  | Quant | Total bpw | PPL | vs Q4_X |
  |---|---|---|---|
  | IQ3_K | 3.845 | 1.8775 | +3.0% |
  | smol-IQ3_KS | 3.249 | 1.9562 | +7.3% |
  | smol-IQ2_KL | 2.755 | 2.1813 | +19.6% |
  | smol-IQ2_KS | 2.261 | 2.6209 | +43.7% |
  | smol-IQ1_KT | 1.832 | 3.2450 | +78.0% |

  The card says KT quants are "slower for TG on CPU, but fast on GPU, similar to EXL3". — [ubergarm/Kimi-K2.5-GGUF](https://huggingface.co/ubergarm/Kimi-K2.5-GGUF)
- DeepSeek-R1-0528 (ubergarm; Q8_0 PPL 3.2130): IQ3_KS 3.598 bpw +2.7%; IQ2_K_R4 2.799 bpw +9.1%; IQ1_S_R4 1.664 +51.9% (my arithmetic). IQ2_KT is described as "slightly smaller than IQ2_K_R4 with slightly worse perplexity and KLD" and was not released. — [ubergarm/DeepSeek-R1-0528-GGUF README](https://huggingface.co/ubergarm/DeepSeek-R1-0528-GGUF/blob/main/README.md)
- Prior LOCAL baselines:
  - Kimi-K2-Instruct (BF16 source): IQ2_KL 2.923 bpw +7.8%; IQ2_KS 2.429 +24.8%; IQ1_KT 1.959 +34.7%.
  - QTIP on Llama-3.1-405B: 2-bit 3.52 → 5.08 (+44%); 3-bit 4.12 (+17%).
  - No QTIP/AQLM/QuIP#/PV-Tuning/GPTAQ/HQQ paper reports a MoE larger than Mixtral-8x7B.

  — [LOCAL extreme_quantization.md], [ubergarm/Kimi-K2-Instruct-GGUF](https://huggingface.co/ubergarm/Kimi-K2-Instruct-GGUF), [QTIP](https://arxiv.org/html/2406.11235v4)
- EXL3 vs GGUF at matched size on a small model (RTX 5090 blog): EXL3 4.00 bpw had KLD 0.0248 at 5.8 GiB vs Q4_K_M 0.0398 at 6.6 GiB, but Q5_K_M at 7.84 GiB (0.0172) beat a same-size EXL3. llama.cpp decoded 15–21% faster in the size-matched pairs. This is a vendor blog. — [atomic.chat exl3 vs gguf](https://atomic.chat/blog/guides/exl3-vs-gguf)

### Inferences
- Re-verifying the "2.9 → 2.4 cliff": the V4-Flash FP4-source ladder confirms it and places the knee more precisely between 2.79 and 2.58 expert-bpw. Excess KLD jumps from 0.100 to 0.162 for 0.21 bpw, which is about 0.30 KLD/bpw. From 3.2 to 2.9 the slope is about 0.15 KLD/bpw. Below 2.5 the curve is roughly linear at about +6–7 pp PPL per 0.2 bpw.
- The EXL3 V4.1 +0.4% (prose) vs GGUF V4-Flash IQ3_XXS +7.1% at about the same expert bpw is the strongest hint that trellis plus Hadamard plus per-expert calibration (EXL3) beats i-quants well beyond the "0.2 bpw" figure ikawrakow measured on dense 8B models. It is confounded three ways: different model (0731 vs 1017), different corpus (wikitext vs LibertAI held-out), and a baseline that differs in kind (EXL3 is measured vs dequantized base; GGUF rows include the runtime floor). Treat it as a hypothesis to test locally, not a fact.
- At 2.0–2.3 expert-bpw, every large-MoE datapoint (V4-Flash +21–28%, V4.1 IQ2_XXS +19.5%, Kimi-K2.5 +43.7%, GLM-4.5 +36–68%) is in the "visibly damaged" zone. No trellis datapoint in this band on a ≥300B MoE shows better than about +20%.

### Gaps
- No published IQ2_KT/IQ3_KT ladder exists for any DeepSeek-V4/V4.1 model, and no EXL3 row below 2.9 avg exists for V4.1.
- No KLD tables for the ik_llama KT rows on large MoEs; only PPL is available.
- AQLM, QuIP#, GPTAQ and HQQ: still no MoE above Mixtral in any paper (unchanged from LOCAL).

## Q2. Double quantization: cost of requantizing MXFP4/FP4-QAT experts to sub-4-bit

### Takeaway
No paper measures FP4 → 2–3-bit requantization directly. The matched-model evidence says FP4-source experts tolerate about 3 bpw well, with the V4.1 EXL3 at +0.4–1.8% and the V4-Flash GGUF at 3.2 bpw costing +3%. Below about 2.5 bpw the cost is steep and similar to or worse than BF16-source models. The INT4-QAT Kimi-K2.5 is the worst case: +43.7% at 2.26 bpw. Expect double-quant to add a modest premium above about 2.7 bpw and a large one below about 2.3.

### Cited Findings
- V4-Flash-0731 FP4 source: 2.91 bpw +7.1%, 2.48 +15.2%, 2.25 +21.3% (my arithmetic). — [AtomicChat card](https://huggingface.co/AtomicChat/DeepSeek-V4-Flash-0731-GGUF)
- The BF16-source DeepSeek-R1-0528 is at +9.1% at 2.799 total bpw (IQ2_K_R4) on ik_llama types. — [ubergarm R1-0528](https://huggingface.co/ubergarm/DeepSeek-R1-0528-GGUF/blob/main/README.md)
- Kimi family:
  - INT4-QAT source: K2.5 IQ2_KS 2.261 bpw +43.7%; K2-Thinking smol-IQ2_KS 2.261 +41.0%.
  - BF16 source: K2-Instruct IQ2_KS 2.429 +24.8%.
  - At ~2.75 bpw (smol-IQ2_KL): K2.5 +19.6%, K2-Thinking +17.9%, K2-Instruct +15.5%.

  — [ubergarm/Kimi-K2.5-GGUF](https://huggingface.co/ubergarm/Kimi-K2.5-GGUF); [LOCAL extreme_quantization.md] citing [ubergarm/Kimi-K2-Thinking-GGUF](https://huggingface.co/ubergarm/Kimi-K2-Thinking-GGUF)
- V4.1-1017 EXL3: experts at 3 bpw from FP4 source cost +0.4% prose PPL. — [LibertAI EXL3](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw)
- Keeping down_proj bit-exact MXFP4 and requantizing only gate/up does not beat uniform requantization at matched bytes. On V4-Flash-0731, the "GateUp-IQ2_XS/IQ3_XXS + Down-MXFP4" build is 120.3 GB with KLD 0.222, vs uniform IQ3_XXS experts at 113.9 GB with KLD 0.218. — [dsv4-eval-artifacts](https://huggingface.co/datasets/AtomicChat/dsv4-eval-artifacts)
- MXFP4→NVFP4 can be a bit-exact transcode: "The 4-bit codes are untouched; only the block scales are re-expressed" (1×32 → 1×16), with 0 mismatches reported. — [LibertAIDAI/DeepSeek-V4.1-Flash-1017-NVFP4](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-NVFP4)
- MXFP4 error analysis papers (2603.08713, 2604.17789 DuQuant++, 2605.19561 TORQ, 2605.20402) cover FP4 quantization of high-precision weights, not requantization of an FP4 checkpoint to lower bits. — [arXiv 2605.20402](https://arxiv.org/abs/2605.20402); [arXiv 2604.17789](https://arxiv.org/pdf/2604.17789)
- The BF16 round trip of E2M1×E8M0 is exact, so the whole cost is the second quantizer (prior analysis). — [LOCAL extreme_quantization.md]

### Inferences
- A QAT'd FP4 source does not tolerate sub-2.5 bpw better than a BF16 source. If anything it tolerates it worse: Kimi INT4-QAT costs +41–44% at 2.26 vs BF16-source +24.8% at 2.43. The likely mechanism is that QAT has already spent the weights' redundancy fitting the 16-level grid, so a coarser second grid lands on top of the first one's error. Above about 2.7 bpw the premium is small (2–4 pp).
- For V4.1 specifically, the 3-bpw EXL3 point says the "expert floor" for a nearly lossless requant is about 3 bpw with trellis. Going to 2.2–2.3 trades about +20% PPL (i-quant evidence), about 3× the 3-bpw penalty.
- A Hadamard rotation before the trellis (EXL3/QTIP) spreads the FP4 grid structure into a Gaussian-like distribution. That plausibly explains why EXL3 handles the FP4 source well. ik_llama KT types have no Hadamard (PR 113), which is untested on FP4 sources (hypothesis).

### Gaps
- No same-model A/B of BF16-source vs FP4-source requant exists. The DeepSeek V4 family has only ever been released as FP4 experts.
- No IQ*_KT on any MXFP4-source model.

## Q3. Per-expert / per-layer mixed precision

### Takeaway
GGUF cannot vary type per expert, but it can vary type per layer and per projection, and every good V4-Flash recipe already does this. The pattern is: protect the high-energy layers and down_proj, and keep attention, shared experts, the compressor/indexer and hc tensors at Q8/Q6/F32. That is the practical lever. Per-expert precision needs our own CUDA server or EXL3-style per-expert tensors.

### Cited Findings
- AtomicChat splits gate/up types between 18 "higher-energy" layers (20, 26–38, 40–42) and the other 25. It keeps last-layer down_proj at MXFP4. — [AtomicChat card](https://huggingface.co/AtomicChat/DeepSeek-V4-Flash-0731-GGUF)
- antirez mixes: experts on layers 37–42 at Q4_K, other layers IQ2_XXS gate/up + Q2_K down, attention projections, shared experts and output at Q8. That build is 97.6 GB with KLD 0.369, vs 86.7 GB / KLD 0.408 for the all-IQ2_XXS-with-Q2_K-down variant. — [dsv4-eval-artifacts](https://huggingface.co/datasets/AtomicChat/dsv4-eval-artifacts)
- LibertAI EXL3 on V4.1: one K per layer, layers 18–22 at 2 bpw and the rest at 3. It also gives the shared expert, attention, indexer and Engram projection 4–8 bpw. — [LibertAI EXL3](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw)
- A naive Q2_K that drags non-expert tensors down collapses on V4-Flash (PPL 9.11, KLD 0.97). — [dsv4-eval-artifacts](https://huggingface.co/datasets/AtomicChat/dsv4-eval-artifacts)
- Prior LOCAL findings:
  - GGUF has one `ggml_type` per fused `ffn_*_exps` tensor (no per-expert types).
  - vLLM reads the scheme of expert 0 only.
  - Per-expert mixing saves resident memory but barely changes decode bytes (routing-weighted).
  - Matched-bit gains are +0.3 to +13 pt, tracking routing skew, and are unmeasured after REAP-50.

  — [LOCAL per_expert_mixed_precision.md]

### Inferences
- After REAP-192 the cold tail is gone, so per-layer allocation is the realistic lever: about 5 layers at 2.x bpw and the rest at about 3 (LibertAI's shape). The layers to protect are the "higher-energy" ones (V4-Flash 0731 indexing; V4.1 has 40 layers, so recompute from our saliency bank) and the last layers that feed DSpark (37–39).
- Shared experts are 1.42B params. At Q8 they are about 1.4 GiB and are read every token, so Q6_K (≈1.1 GiB) is the cheap compromise AtomicChat uses.

### Gaps
- No V4.1 per-layer sensitivity scan is published. The 0731 "18 high-energy layers" list does not map 1:1 onto V4.1's 40 layers.

## Q4. Byte arithmetic: experts that fit, and the prune-vs-quant break-even

### Takeaway
The expert budget is about 94.4–95.1 GiB. That is 117 GiB minus a 12 GiB reserve, minus the 9.77 GiB non-Engram, non-DSpark floor, minus 0.1–0.8 GiB of KV for 128k–1M context. The maximum average expert bpw is 4.48 at 128 experts, 2.98 at 192, 2.56 at 224 and 2.24 at 256 (all my arithmetic). So 256 @ 2.3 does not fit, and 256 needs uniform IQ2_KT/IQ2_KS-class (2.125–2.19 bpw) everywhere including down_proj. On the measured curves, 192 @ ~2.85 (about +15% prune, +5–8% quant) beats 256 @ 2.2 (about +5.5% prune, +22–28% quant). 128 @ 4.25 native fits with ~5 GiB spare but has no measured quality and an extrapolated prune cost above +30%.

### Cited Findings
- V4.1-1017 expert geometry:
  - w1/w3 are `[2304, 5120]` and w2 is `[5120, 2304]`, so each expert is 3 × 2304 × 5120 = 35.39M params.
  - There are 40 MoE layers × 384 experts, totalling 268.95 GiB at MXFP4.
  - The floor without Engram and the DSpark drafter is 9.77 GiB (8.87 without vision).
  - The DSpark drafter is 7.39 GiB.
  - KV is 890 B/token, about 0.83 GiB at 1M tokens.

  — [LOCAL target_models.md]
- Envelope: 117 GiB with a 12 GiB runtime reserve; the box exposes 122.83 GiB. — [LOCAL structural_and_memory_budget.md]
- Measured V4.1 (0909) REAP text PPL (65,504 tokens): 272 kept +4.13%; 256 +5.49% (mean ranking) or +14.1% (balanced ranking for vision); 240 +7.12%; 192 +14.66%. vLLM's fused MoE accepts only {128,192,256,320,384}. — [LibertAIDAI/DeepSeek-V4.1-Flash-REAP-256E](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-REAP-256E) via [LOCAL v41_architecture_pruning_delta.md]
- Local measurement on GLM-5.3-Flash: REAP-50 +21% PPL vs one GGUF bit-step +1.46%, so pruning is about 14× more damaging per byte at the 4-bit end. — [LOCAL memory reap-criterion-and-corpus-audit.md]

**Byte table (my arithmetic).** Expert GiB = N × 40 × 35.39M × bpw / 8. Total = experts + 9.77 floor + 0.83 KV + 12 reserve, against a 117 limit.

| Keep | Expert recipe (gate/up : down) | Avg bpw | Experts GiB | Total GiB | Fits? |
|---|---|---|---|---|---|
| 128 | MXFP4 native (bit-exact) | 4.25 | 89.7 | 112.3 | yes (4.7 spare) |
| 128 | NVFP4 transcode | 4.50 | 94.9 | 117.5 | **no** |
| 192 | IQ3_KT : IQ3_KT | 3.125 | 98.9 | 121.5 | no |
| 192 | IQ3_XXS : IQ3_XXS | 3.0625 | 96.9 | 119.5 | no |
| 192 | IQ2_KL : IQ3_KS | 2.854 | 90.3 | 112.9 | yes (4.1 spare) |
| 192 | per-layer 35×3.0 + 5×2.0 (LibertAI shape) | ≈2.875 | 91.0 | 113.6 | yes |
| 192 | IQ2_KT : IQ3_KT | 2.458 | 77.8 | 100.4 | yes (16.6 spare) |
| 256 | IQ2_XS : IQ3_XXS | 2.562 | 108.1 | 130.7 | no |
| 256 | IQ2_XS : IQ2_XS | 2.3125 | 97.6 | 120.2 | **no (the "2.3 bpw" option)** |
| 256 | IQ2_KT : IQ2_KT | 2.125 | 89.7 | 112.3 | yes (4.7 spare) |
| 384 | any ≥2.0 | — | ≥126.6 | — | no |

(Type sizes: IQ2_KT 2.125, IQ3_KT 3.125 and IQ4_KT 4.0 come from [PR 113](https://github.com/ikawrakow/ik_llama.cpp/pull/113). IQ2_XS 2.3125, IQ3_XXS 3.0625, IQ2_KL 2.6875 and IQ3_KS 3.1875 are ggml/ik type-table values I did not re-fetch here.)

### Inferences
- Iso-byte comparison, assuming prune and quant costs add (the LOCAL finding is that they are sub-additive, so these are upper bounds):
  - (A) 256 @ ~2.2: prune +5.5% (or +14% with a vision floor), plus quant +21–28% from the V4-Flash i-quant ladder at 2.06–2.25. Total ≈ +27–35% (+35–42% with a vision floor).
  - (B) 192 @ ~2.85: prune +14.7% (expected lower with our balanced corpus and mass criterion), plus quant +7–8% (i-quant IQ3_XXS/IQ2_M) or ~+0.4–1.8% (EXL3-class trellis). Total ≈ +15–23%.
  - (C) 128 @ 4.25 native: prune extrapolated > +30%, no quant cost.

  B wins. A loses even if trellis buys a full 0.2 bpw (2.125 KT ≈ 2.3 i-quant ≈ +20%). Its only advantage is vision retention, and that is erased by the vision-floor cost.
- Revised break-even: on V4.1's measured curves the prune-vs-quant break-even sits near 192 experts at about 2.6–2.8 bpw. Going from 192 to 256 experts (−9 pp of prune cost) is worth only about 0.3–0.4 bpw of expert precision before it loses, and fitting 256 needs 0.6–0.7 bpw. This agrees with the prior "~50% @ 2.9–3.0".
- A 192 @ IQ2_KT/IQ3_KT (2.46) build leaves about 16 GiB spare. It is a useful fallback if the reserve proves to be needed for DSpark (7.39 GiB native, or about 5 GiB requantized) or for long-context activations.
- Trimming the floor buys bpw. Attention, compressor and indexer at 5–6 bpw instead of FP8 (EXL3 used 5) saves about 1.5–2 GiB, and dropping vision saves 0.9 GiB. That ~2.5 GiB is worth only about +0.08 bpw on 192 experts (my arithmetic). It is not enough to reach uniform IQ3_XXS/IQ3_KT (3.06–3.125, which needs about +4.5 GiB), so it is a marginal lever.

### Gaps
- No measured V4.1 PPL at 128 kept experts, and no prune × quant joint point on any V4 model.
- The V4-Flash (0731, 304B) quant curve is transferred to V4.1 (40 layers, 384 experts), which is unverified.

## Q5. Runtimes on sm_110 (Thor), and the decode cost of trellis dequant

### Takeaway
ik_llama.cpp is the only runtime with both IQ*_KT/IQ*_K types and merged V4.1 support (PR #2455, 2026-09-21). Mainline llama.cpp V4.1 (PR #28696) is still unmerged, and exllamav3 cannot load V4.1. Thor-specific risks:
- Engram is read through mmap / pinned CPU in both llama.cpp ports, which is wrong on Thor and needs an O_DIRECT side-path patch.
- MXFP4-expert CUDA paths have open crash bugs.
- The IQ-on-Blackwell "wrong results" scare does not reproduce on Thor with CUDA 13.2.

Expert bpw barely moves decode speed, because attention dominates per-token bytes.

### Cited Findings
- ik_llama PR #2455 "model : add DeepSeek-V4.1 (deepseek41)" was merged 2026-09-21. It covers architecture, loader, graph and cache. Engram tables "stay in the model file's memory mapping … read only through get_rows", with `posix_madvise(WILLNEED)` row prefetch that cut major faults from 41–62 to 1–11 per token. Tested Q3_K_M (323 GiB) and Q4_K_M.
  - CUDA partial offload: 20.4–21.3 tok/s.
  - Not implemented: MTP/DSpark graph, vision, session save of compressed streams.
  - Linked open issue #2519: NaN logits and illegal memory access with MXFP4 routed experts on CUDA sm_120.

  — [ik_llama.cpp PR #2455](https://github.com/ikawrakow/ik_llama.cpp/pull/2455); [issue #2438](https://github.com/ikawrakow/ik_llama.cpp/issues/2438)
- Mainline llama.cpp V4.1 PR #28696 is still open (ready for review 2026-09-22). Engram tables are stored as GGUF tensors `blk.{1,14}.engram_embd.weight` and pinned to CPU with `-ot "engram_embd.weight=CPU"`. `TENSOR_READ_LAZY` is mentioned as an untested disk option. Other reports:
  - An MXFP4 MUL_MAT_ID MMQ illegal-memory-access was reported (padding; fix #29953).
  - Q2_K on a DGX Spark ran at 2.8 tok/s.
  - With a DSpark draft, decode went from 17.7 to 22.5–22.8 tok/s in one fork.

  — [llama.cpp PR #28696](https://github.com/ggml-org/llama.cpp/pull/28696)
- PR #27902 (IQ1_S/IQ2_S/IQ3_S wrong on Blackwell) was closed unmerged on 2026-10-06 as "a 13.2 bug". A tester on Jetson AGX Thor (sm_110, CUDA 13.2.86) could not reproduce the failures without the patch. The patch's byte-safe path was 1.02–1.18× slower on sm_110. — [llama.cpp PR #27902](https://github.com/ggml-org/llama.cpp/pull/27902)
- KT kernels: ikawrakow's CUDA IQ2_KT ran at 194 t/s vs IQ2_XXS 208–216 t/s (RTX 4080, LLaMA-3.1-8B). That is about 7–10% slower, with QTIP estimated at about 155. There are no quantized mat-vec kernels; it decodes to float. — [PR 113](https://github.com/ikawrakow/ik_llama.cpp/pull/113)
- Users report KT "fast on GPU, similar to EXL3" but CPU-bound in hybrid offload. — [ubergarm/Kimi-K2.5-GGUF](https://huggingface.co/ubergarm/Kimi-K2.5-GGUF); [Level1Techs thread](https://forum.level1techs.com/t/deepseek-deep-dive-r1-at-home/225826?page=15)
- llama.cpp builds for Thor only via `GGML_NATIVE` (sm_110 is not in the default arch list). IQ3_M/IQ4_XS GGUFs have run locally on Thor. — [LOCAL extreme_quantization.md]
- exllamav3 has no DeepseekV41 class ("cannot load this checkpoint directly"). There is no sm_110 evidence, and there are open sm_120 crashes. — [LibertAI EXL3](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw); [LOCAL extreme_quantization.md]
- On V4-Flash, decode was flat (36.9–37.8 tok/s) from 2.79 to 3.94 expert-bpw on MI300X. — [AtomicChat card](https://huggingface.co/AtomicChat/DeepSeek-V4-Flash-0731-GGUF)
- I-quants cost 5–10% inference speed vs K-quants (Unsloth, Qwen3.5). — [Unsloth GGUF benchmarks](https://unsloth.ai/docs/de/modelle/qwen3.5/gguf-benchmarks)
- LOCAL Thor facts:
  - mmap of a sparse table drags 64 KiB of page cache per touch into the GPU pool, and pinned-host offload frees 0 bytes.
  - O_DIRECT Engram rows were measured at 1.13–1.30 ms/token.
  - Effective bandwidth is about 116 of ~200 GB/s achievable.
  - INT2 is a net decode loss on Thor.
  - Block-scaled tcgen05 does not work on sm_110a.

  — [LOCAL memory compressing-1t-moe-thor-research.md, thor-base-model-shortlist.md]

### Inferences
- Decode-bytes arithmetic (my arithmetic, from LOCAL active params of 8.49B routed, 1.42B shared, 5.13B attention and 0.66B head):

  | Component | Precision | GB per token |
  |---|---|---|
  | Routed experts | 2.3 bpw | 2.44 |
  | Routed experts | 2.9 bpw | 3.08 |
  | Attention | Q8 | 5.45 |
  | Attention | 5 bpw | 3.2 |
  | Shared experts | Q8 | 1.5 |

  The routed-expert bpw choice moves B_tok by only about 6%. Attention precision moves it by about 20%. Expert bpw is a fit decision; attention bpw (and the 116→200 GB/s kernel gap) is the speed decision.
- Trellis decode cost on Thor: Thor has relatively less ALU per byte than an RTX 4080, so the 7–10% KT penalty seen there is a lower bound. KT may become instruction-bound in our tall-thin GEMV regime. This is unmeasured; the first thing to measure is a single-layer `test-backend-ops -o MUL_MAT_ID` perf for iq2_kt/iq3_kt vs iq2_xs/iq3_xxs/mxfp4 on Thor.
- Neither llama.cpp V4.1 port is Thor-safe for Engram out of the box. Both put the 189 GiB FP8 tables in host memory or mmap, which on Thor competes with the GPU pool. Serving needs a patch to replace the Engram `get_rows` with an O_DIRECT side-path.

### Gaps
- No IQ*_KT, IQ*_K or MXFP4-expert MUL_MAT_ID performance or correctness data on sm_110.
- ik_llama #2519 (MXFP4 experts on sm_120) is unresolved, and its applicability to sm_110 is unknown.
- No source confirms ik_llama V4.1 runs the FP4 KV cache (packed FP4 KV lives in an unmerged branch), so KV may be larger than 890 B/token in GGUF runtimes.

## Q6. NVFP4 requant for vLLM: Thor status and byte cost

### Takeaway
NVFP4 is a lossless transcode of MXFP4 experts but costs more bytes (4.5 vs 4.25 bpw). It therefore cannot hold 128 experts (117.5 GiB total) and gives no sub-4-bit option. It only makes sense if vLLM-on-Thor is the runtime and only cutlass-NVFP4 MoE works there. For a GGUF/ik_llama deployment it is irrelevant for experts. NVFP4 for the dense/attention overlay is a separate, LOCAL-proven bandwidth win.

### Cited Findings
- LibertAI NVFP4 of V4.1-1017:
  - Experts are NVFP4 W4A4 (FP32 global scale plus input_scale), transcoded from MXFP4 with 0 mismatches.
  - Engram tables are re-encoded FP8 → MXFP4: 188.83 → 97.28 GiB, 136 B/row (vs 264), lossy at cosine 0.9934.
  - Attention and shared experts stay FP8.
  - Total 475.24 → 399.51 GiB.
  - Needs vLLM#61530 for the MXFP4 Engram. SGLang does not support it. Tested on GB300.
  - vLLM keeps the tables in pinned host memory by default.

  — [LibertAIDAI/DeepSeek-V4.1-Flash-1017-NVFP4](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-NVFP4)
- Real NVFP4 checkpoints average about 0.62 B/w. On sm_110a, block-scaled tcgen05 does not work, Marlin MoE crashed on a 122B NVFP4, and cutlass MoE works. The local NVFP4 dense overlay gave −50.6% B_tok at +1.56% PPL. — [LOCAL memory thor-base-model-shortlist.md, dflash-122b-marlin-moe-crash.md, glm53-nvfp4-dense-overlay.md]

### Inferences
- The MXFP4 Engram re-encode (97 GiB, 136 B/row) is relevant to the NVMe side-file even outside vLLM. It halves Engram disk (the LOCAL gate is that the NVMe has only 51 GB free) and per-token read size (48 × 136 B). Its quality cost is unmeasured beyond cos 0.9934, so ship FP8 rows if disk allows.
- On vLLM/Thor at 192 experts there is no sub-4-bit MoE path: Marlin WNA16 is {4,8} bits only (LOCAL), so 192 @ NVFP4 = 142 GiB does not fit. vLLM is only viable for the 128-expert native/transcoded build, and only with Engram patched out of pinned-host memory.

### Gaps
- No PPL/KLD for the NVFP4 or MXFP4-Engram V4.1 build.
- vLLM MXFP4 MoE (non-NVFP4) on sm_110 is unverified.

## Q7. Recommendation: GGUF quant mix and Engram storage

### Takeaway
Ship 192/384 experts in ik_llama.cpp GGUF:
- Routed experts at about 2.85 avg bpw, per-layer mixed.
- Attention, compressor, indexer and shared experts at Q8_0/Q6_K.
- hc/norm/router at F32/BF16.
- Engram in an O_DIRECT side-file (or GGUF tensors that are excluded from mmap/load and read by O_DIRECT pread).

Build two candidate expert recipes at matched bytes and A/B them against the dequantized base with KLD: (a) i/K-quant IQ2_KL:IQ3_KS, and (b) trellis IQ3_KT on most layers with IQ2_KT on the lowest-saliency layers. Keep 128 @ native MXFP4 as the zero-requant control.

### Cited Findings
- Good V4-Flash recipes protect attention (Q8_0), shared experts and embed/output (Q6_K), the router (BF16) and hc (F32). Naive Q2_K collapses. — [AtomicChat card](https://huggingface.co/AtomicChat/DeepSeek-V4-Flash-0731-GGUF); [dsv4-eval-artifacts](https://huggingface.co/datasets/AtomicChat/dsv4-eval-artifacts)
- "Hyper-connection tensors should be kept in F32" (fixed Q2_K build); Q1_0 broke because of a 1-bit embedding. — [llama.cpp PR #28696](https://github.com/ggml-org/llama.cpp/pull/28696)
- The EXL3 V4.1 per-layer shape is 35 layers at 3 bpw and 5 middle layers (18–22) at 2 bpw. — [LibertAI EXL3](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw)
- The Engram table layout in GGUF is `blk.{1,14}.engram_embd.weight`, written via disk-backed memmap in row blocks. — [llama.cpp PR #28696](https://github.com/ggml-org/llama.cpp/pull/28696)
- Engram traffic is 48 rows/token (~12.4 KiB at FP8+scale), with addresses known from token ids. — [LOCAL target_models.md], [LibertAI NVFP4](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-NVFP4)

### Inferences
Concrete recipe (192 kept, about 113 GiB total; my arithmetic, to be tuned to the envelope):

**Non-expert tensors**
- `attn_*`, `compressor_*`, `indexer.*`: q8_0. Consider iq6_k/q6_k on wq_b/wo_* as a speed lever.
- Shared experts `ffn_*_shexp`: q6_k.
- token_embd: q6_k.
- output: q6_k.
- `ffn_gate_inp`: bf16.
- hc_*, norms, sinks, `exp_probs_b`: f32.
- DSpark layers: drop, or a separate sidecar at iq4_kt/q4_k.

**Routed experts**
- Default layers: gate/up iq3_kt (3.125) or iq2_kl (2.6875); down iq3_ks (3.1875) or iq4_kt.
- Choose per layer from our saliency bank so the layer-weighted average is ≤2.95, protecting layers 0–1, the Engram layers (1, 14), the kv-source layers (2, 8, 14, 20) and 37–39 (DSpark taps).
- Drop the 4–6 least-sensitive layers to iq2_kt gate/up.

Everything is imatrix'd on the balanced calibration corpus with Engram ON. Per-layer types are set with `--custom-q` regex. That is legal in GGUF because it is per tensor name, i.e. per layer.

**Engram storage.** Do not rely on the ports' mmap/CPU-pin. Options, best first:
1. A raw side-file per table, rows packed at 264 B (FP8+E8M0) or 136 B (MXFP4), read with O_DIRECT aligned 1 KiB windows (the shape measured locally at 1.13–1.30 ms/tok), prefetched from token ids one step ahead.
2. Keep the tensors in the GGUF but skip them at load, and pread them with O_DIRECT from their GGUF data offset. GGUF tensor data is alignment-padded, so offsets are computable. This keeps one artifact.

Both need a small patch replacing Engram `get_rows` with a host-gathered 48-row buffer uploaded per step. Disk: 189 GiB FP8, or 97 GiB MXFP4, plus the ~120 GiB GGUF. Neither fits the 51 GB free; a second NVMe is required.

**Why not the alternatives**
- 256 @ 2.3 does not fit, and 256 @ 2.125 is about +27–35%.
- 128 @ MXFP4 has no requant penalty but a prune cost extrapolated above +30%, with zero headroom gain over 192.
- EXL3 has no V4.1 model class and no Thor evidence.
- vLLM NVFP4 cannot go below 4.5 bpw.

**Validation gate before committing.**
- Single-layer MUL_MAT_ID correctness and perf for iq2_kt, iq3_kt, iq2_kl, iq3_ks and mxfp4 on sm_110 under CUDA 13.2. If KT is more than 15% slower than iq-quants at batch 1, prefer the IQ2_KL:IQ3_KS mix.
- Then full-model KLD vs dequantized base on ≥43k tokens (LOCAL eval protocol).

### Gaps
- Whether ik_llama's V4.1 graph runs fully on CUDA with all 192-expert layers resident on Thor, and its real tok/s there, is untested.
- Whether KT dequant on Thor is instruction-bound at batch 1 is unmeasured.
- The quality of EXL3-vs-KT on the same FP4 source has never been compared.
