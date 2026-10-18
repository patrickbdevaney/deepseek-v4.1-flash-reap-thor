# GO/NO-GO: MiMo-V2.6-Pro-MOPD (1T) on a single Jetson AGX Thor vs MiMo-V2.6-Flash

Method note: all byte figures below are GiB = 2^30 unless marked GB (10^9). Pro byte split (494.86 GiB routed experts, 32.29 GiB remainder) is taken as verified from the sibling note `target_models.md` (computed from all 130 shard headers of the live HF repo on 2026-10-17). Every budget line shows its terms. Assumptions are labelled [ASSUMPTION]; extrapolations beyond any measured point are labelled [EXTRAPOLATION]. The calculation script with every term is at `/tmp/claude-2002/-home-patrickd/3952824c-51a0-4b40-aa8e-acf3b3b02ab9/scratchpad/calc/budget.py`.

**Decision statement (numbers first):** On a 117 GiB envelope, after 18 GiB of fully-quantized non-expert weights, 8 GiB runtime and 6.25 GiB of BF16 KV at 128k, 84.75 GiB remains for Pro's 494.86 GiB of QAT-MXFP4 experts. That forces removal of **82.9% of experts at native 4.25 bpw, 79.2% at 3.5 bpw, 75.7% at 3.0 bpw, 70.9% at 2.5 bpw, and still 63.6% at 2.0 bpw** (+43.7% PPL double-quant). The literature's ceiling for a general-purpose MoE is 50% removal (no primary source reports any multi-domain MoE surviving beyond it within 10% of base); to stay at 50% Pro would need **1.46 bpw** on experts that were trained at 4 bits, which is below every measured point on this checkpoint (2.0 bpw = +43.7% PPL) and in the regime where 1.7-1.9 bpw GGUFs of other 1T-class MoEs lose 50-130% PPL. Even the single-domain specialist ceiling (75%) only closes at ≤3.0 bpw (+9.8% PPL) with 3.7 GiB left for KV (~77k tokens BF16). Pro is therefore **NO-GO as a generalist, NO-GO as a multi-domain specialist, and at best a 75%-pruned + 3-bpw single-domain curiosity with ~1.5-2.7 tok/s decode**. MiMo-V2.6-Flash REAP-50 (88 GiB on disk, 23,040 B/token BF16 KV, 2.22x less KV per token than Pro) already exists in the user's own HF account and fits 1M context at FP8 KV with ~9.5 GiB to spare; it is the Pareto-better choice on every axis measured (fit, prune ratio, decode bytes/token). **GO: Flash REAP-50. NO-GO: Pro.**

## KQ1. Memory budget table for Pro and the required expert-removal ratio at 4.25 / 3.5 / 3.0 / 2.5 bpw

### Takeaway
At every context from 32k to 256k and every expert bit-width from native 4.25 down to 2.0 bpw, the required removal ratio is 60-84%; at the native MXFP4 width it is 82-84%, i.e. keep only 61-69 of 384 experts per layer. There is no (context, bpw) cell in the table that lands at or below the 50% generalist ceiling.

### Cited Findings
- Pro main index = 527.15 GiB; routed experts = 465.75 GiB packed E2M1 `U8 [2048, 3072]` + 29.11 GiB E8M0 scales `U8 [2048, 192]` = **494.86 GiB = 93.9%** (69 MoE layers × 384 experts × 3 matrices = 79,488 matrices); non-expert remainder 32.29 GiB = attention 24.00 (`o_proj` BF16 13.13 + `qkv_proj` FP8 10.87) + embed/head 3.49 + MTP 2.29 + vision 1.38 + audio 0.55 + gates 0.30 + dense layer-0 0.28 — [sibling note target_models.md §Q3, computed from index.json + shard headers](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-MOPD/resolve/main/model.safetensors.index.json)
- Non-expert floor if everything is quantized: attention 18.72 B params at 4-bit/per-32-scale ≈ 9.8 GiB, embed+head 1.87 B at 8-bit ≈ 1.75 GiB, rest as-is → "≈ 17-19 GiB" — [sibling note target_models.md §Q4 inference]; this 17-19 figure keeps MTP 2.29 + vision 1.38 + audio 0.55 = 4.22 GiB at BF16, so dropping all three gives ≈ 13.8 GiB.
- 29.11 / 465.75 = 0.0625 = 1/16 → scales add exactly 0.25 bpw on top of 4.0 → native expert width is **4.25 bpw**, so `expert_bytes(bpw) = 494.86 × bpw / 4.25` is exact for this layout (my arithmetic on the header-derived split).
- KV per token = 10 GA layers × 8 KV heads × (192 K + 128 V) × 2 B = **51,200 B** BF16 (GA layers at `hybrid_layer_pattern` zeros 0,7,15,23,31,39,47,55,62,69; 60 SWA-128 layers are window-capped) — [config.json](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-MOPD/resolve/main/config.json), split counted in [sibling note target_models.md §Q3].
- Thor envelope: "117 GiB available, 240 GB/s measured" — [local memory thor-model-fit-frontier.md]; "Thor usable: 117 GiB = 125.6 GB (measured)" — [local survey MIMO_V26_FLASH_SURVEY_2026-09-23.md §3]. Prior Flash fit tables used "~5 GB workspace" (= 4.7 GiB); this note uses the caller's 8 GiB runtime figure [ASSUMPTION: 8 GiB covers CUDA context + allocator + nvmap pool on Tegra; no published Thor measurement of CUDA-context size was found — NVIDIA's jetson-memory-audit skill only documents that the Thor RM holds freed pages in its own pool after a context exits and that `cudaMemGetInfo` under-reports, [skills.sh/nvidia/jetson-memory-audit](https://www.skills.sh/nvidia/skills/jetson-memory-audit)].
- Cross-check of the active-param path: 8 × 3 × 2048 × 6144 × 69 = 20.84 B routed + 18.72 B attention + 0.30 B dense + 1.87 B embed/head + 0.16 B gates ≈ 41.9 B ≈ the 42B headline — [sibling note target_models.md §Q3].

**Budget table (E = 117 GiB; R = 8 GiB; NE = 18 GiB keeping MTP/vision/audio; BF16 KV). Terms: KV = ctx × 51,200 B; fixed = NE + KV + R; B = E − fixed; required removal = 1 − B / (494.86 × bpw/4.25).**

| ctx | KV GiB | fixed = 18 + KV + 8 | expert budget B | removal @4.25 | @3.5 | @3.0 | @2.5 | @2.0 | bpw needed to stay at 50% removal | bpw needed at 75% |
|---|---|---|---|---|---|---|---|---|---|---|
| 32k | 1.56 | 27.56 | 89.44 | **81.9%** | 78.1% | 74.4% | 69.3% | 61.6% | 1.54 | 3.07 |
| 128k | 6.25 | 32.25 | 84.75 | **82.9%** | 79.2% | 75.7% | 70.9% | 63.6% | 1.46 | 2.91 |
| 256k | 12.50 | 38.50 | 78.50 | **84.1%** | 80.7% | 77.5% | 73.0% | 66.3% | 1.35 | 2.70 |
| 1M | 50.00 | 76.00 | 41.00 | 91.7% | 89.9% | 88.3% | 85.9% | 82.4% | 0.70 | 1.41 |

**Same, FP8 KV (25,600 B/token):**

| ctx | KV GiB | fixed | B | @4.25 | @3.5 | @3.0 | @2.5 | @2.0 | bpw @50% | bpw @75% |
|---|---|---|---|---|---|---|---|---|---|---|
| 32k | 0.78 | 26.78 | 90.22 | 81.8% | 77.9% | 74.2% | 69.0% | 61.3% | 1.55 | 3.10 |
| 128k | 3.12 | 29.12 | 87.88 | 82.2% | 78.4% | 74.8% | 69.8% | 62.3% | 1.51 | 3.02 |
| 256k | 6.25 | 32.25 | 84.75 | 82.9% | 79.2% | 75.7% | 70.9% | 63.6% | 1.46 | 2.91 |
| 1M | 25.00 | 51.00 | 66.00 | 86.7% | 83.8% | 81.1% | 77.3% | 71.7% | 1.13 | 2.27 |

**Most favourable corner (drop MTP + vision + audio, NE = 13.78 GiB; FP8 KV):** 32k → B = 94.44 → removal 80.9% @4.25, 76.8% @3.5, 73.0% @3.0, 67.6% @2.5, 59.4% @2.0; bpw needed at 50% removal = 1.62. 128k → B = 92.09 → 81.4% / 77.4% / 73.6% / 68.4% / 60.5%; bpw at 50% = 1.58. Dropping the three towers (4.22 GiB) moves the required removal by only ~1 point.

**Totals at the literature ceilings (32k, BF16 KV, fixed 27.56):**

| removal | experts @4.25 | total | over | @3.5 total | @3.0 total | @2.5 total | @2.0 total |
|---|---|---|---|---|---|---|---|
| 50% (generalist ceiling) | 247.4 | 275.0 | +158.0 | 231.3 (+114) | 202.2 (+85) | 173.1 (+56) | 144.0 (+27) |
| 62.5% (math-specialist point) | 185.6 | 213.1 | +96.1 | 180.4 (+63) | 158.6 (+42) | 136.7 (+20) | 114.9 (−2.1, fits) |
| 75% (single-domain + heal) | 123.7 | 151.3 | +34.3 | 129.4 (+12) | 114.9 (−2.1, fits) | 100.3 (−17) | 85.8 (−31) |

### Inferences
- The first cell that fits is 75% removal at 3.0 bpw, or 62.5% at 2.0 bpw — both are "ceiling on one axis × cliff on the other". Nothing fits at the 50% generalist ratio above 1.6 bpw.
- The runtime assumption (8 vs 4.7 GiB) and the tower drop (4.22 GiB) each move the required removal by ≤1 point; the KV choice (BF16→FP8) moves it by 0.1-0.7 points at ≤128k. None of the fixed-cost knobs matters: the 494.86 GiB expert block is 4.2x the whole envelope.

### Gaps
- No Thor-specific measurement of CUDA-context + allocator overhead exists in any source found; 8 GiB is the caller's figure and prior local tables used 4.7 GiB. Either is immaterial to the verdict.

## KQ2. Double-quant penalty on QAT-MXFP4 experts: PPL-only or task evals? Any mixed-precision measurement?

### Takeaway
The only measured ladder on this exact checkpoint (AesSedai) reports perplexity and KL-divergence only, no task evals, and no saliency-aware mixed precision; the KLD at 3.5 bpw (0.133) already exceeds the KLD class of the user's own NVFP4 overlay (+1.56% PPL, 0.138% confident flips), and nobody has measured per-expert mixed precision on the MiMo family.

### Cited Findings
- AesSedai ladder (MiMo-V2.6-Pro-MOPD, FFN tensors quantized, rest Q8_0/Q6_K): MXFP4 537.99 GiB 4.52 BPW PPL 3.171 KLD 0.000 | BPW3.5 416.74 GiB PPL 3.254 (+2.66%) KLD 0.1334 | BPW3.0 357.11 GiB PPL 3.479 (+9.76%) KLD 0.189 | BPW2.5 297.63 GiB PPL 3.790 (+19.56%) KLD 0.262 | BPW2.0 238.22 GiB PPL 4.554 (+43.67%) KLD 0.414. "No top-1 agreement metric is reported... Task evals (MMLU, GSM8K, coding): None reported. Only PPL and KLD." PPL test set and context length not stated. Produced with the bpw-size PR llama.cpp #15550; card does not say whether the mix is per layer or per expert; "The MXFP4 quant is the 'full quality' version, as the model has MXFP4 experts." — [AesSedai/MiMo-V2.6-Pro-MOPD-GGUF](https://huggingface.co/AesSedai/MiMo-V2.6-Pro-MOPD-GGUF)
- The BPW labels are whole-file averages that include Q8_0/Q6_K non-expert tensors; the experts themselves sit below the label. E.g. BPW3.0 = 357.11 GiB total; with ~21 B non-expert params at Q8 (~22 GiB) the experts are ~335 GiB ≈ 2.88 bpw. The caller's mapping `494.86 × bpw/4.25` is therefore slightly optimistic about what "3.0 bpw" costs (my arithmetic on the card's sizes) [INFERENCE].
- QAT provenance: "We employ MXFP4 quantization-aware training (QAT) during mid-training"; "we use MXFP4 as the experts' data type during rollout... quantize–dequantize (QDQ) to the experts after each parameter update" — tech report §3/§6.4 — [MiMo_V2_6_technical_report.pdf via sibling note target_models.md §Q4](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL/blob/main/MiMo_V2_6_technical_report.pdf)
- ggml-org's own Pro GGUF: "The MXFP4 output keeps the routed experts at their native MXFP4 precision. The Q2_K output keeps the expert down projections at MXFP4, and quantizes the gate/up projections to Q2_K" (~516 GiB and ~390 GiB) — no PPL/KLD published — [ggml-org/MiMo-V2.6-Pro-MOPD-GGUF](https://huggingface.co/ggml-org/MiMo-V2.6-Pro-MOPD-GGUF)
- Only matched proxy for a 4-bit-QAT source re-quantized lower: ubergarm Kimi-K2-Thinking INT4-source vs Kimi-K2-Instruct BF16-source, same recipes: "≈ +2-6 pp extra ppl at 2.75-2.9 bpw and ≈ +30 pp extra at 1.83 bpw" — [sibling note extreme_quantization.md KQ3]
- Sub-2-bpw reference points on other 1T-class MoEs (not MiMo): 6block DeepSeek-V4-Pro IQ1_M 1.89 bpw PPL 3.6966 vs Q3_K_M 1.6217 (+128%), IQ1_S 1.72 bpw 4.1095 (+153%); "IQ2 expert path breaks down on this sparse-routing MoE"; ubergarm DeepSeek-R1-0528 IQ1_S_R4 1.664 bpw PPL +51.9% vs Q8_0 — [sibling note sota_shipped_artifacts.md, citing 6block and ubergarm cards](https://huggingface.co/6block/DeepSeek-V4-Pro-0813-GGUF)
- User's own scale on GLM-5.3-Flash: NVFP4 (4-bit from FP8) = +1.56% ppl, 0.138% confident flips; one GGUF step Q4_K_S→Q3_K_M = +1.46%; REAP-50 pruning = +21.0% ppl — [local memory reap-criterion-and-corpus-audit.md; glm53-nvfp4-dense-overlay.md]
- Other sub-4-bit Pro artifacts exist with no quality numbers: MikeRoz EXL3 2.53bpw-h6 and 4.04bpw-h8, doth4580 EXL3 3.0bpw, jarrelscy EXL3/ARVQ hybrids, kernelpool MXFP4 GGUF — [HF API search 2026-10-17](https://huggingface.co/api/models?search=MiMo-V2.6-Pro)

### Inferences
- Perplexity is the only ruler on this ladder, and this project has already measured that perplexity hides generation failures (looping, confident flips) on pruned/quantized MoEs; the 3.0-bpw +9.8% PPL / 0.189 KLD point is roughly half the damage of a full REAP-50 (+21% ppl on GLM) and would be stacked ON TOP of a ≥75% prune. The 2.0-bpw point (+43.7%) is 2x a REAP-50 by itself.
- The 1.46-1.62 bpw needed to hold Pro at the 50% ceiling is below the lowest measured point on this checkpoint (2.0) and in the band where every other 1T-class MoE GGUF shows +50-150% PPL [EXTRAPOLATION].
- Mixed precision by expert saliency (MXFP4 on high-saliency experts, 2.5-3 bpw on the rest) has zero measurements on the MiMo family; the only mixed recipes (AesSedai, ggml-org Q2_K, Baekpica "Mixed-Quant", tacodevs "mixed-GPTQ" for Flash) are per-tensor-type, not per-expert, and none ships task evals.

### Gaps
- No task-level eval (coding, MC, agentic) of any sub-4-bit MiMo-V2.6-Pro quant was found anywhere.
- No per-expert mixed-precision experiment on MiMo (or on any QAT-MXFP4 MoE) was found.

## KQ3. Expert deletion on the MiMo-V2.6 family: public REAP artifacts, what breaks first, no-shared-expert effect

### Takeaway
The only REAP artifacts of MiMo-V2.6 are the user's own Flash REAP-50 (HF + GGUF, no public evals) and tacodevs' MLX REAP-25/37/50 (text-only, no evals); no pruned Pro exists in any format (HF listing of 41 Pro-derived repos, 2026-10-17). The literature's >50% survivors are all single-domain specialists, and the one healing recipe that recovered general ability at 75% did so "entirely from re-routing and from the shared, always-on pathway" — a pathway MiMo does not have.

### Cited Findings
- HF repos matching "MiMo-V2.6-Pro" on 2026-10-17 (41 repos): quantizations (AesSedai, ggml-org, kernelpool, pcuenq, smalinin, DevQuasar, MikeRoz, doth4580, jarrelscy EXL3/ARVQ, mlx-community, inferencerlabs), abliterations, control vectors, "heretic" variants — **no REAP/pruned Pro of any kind** — [HF API search](https://huggingface.co/api/models?search=MiMo-V2.6-Pro)
- Flash REAP artifacts: `patrickbdevaney/MiMo-V2.6-Flash-REAP50` 87.9 GiB (2026-09-27), `patrickbdevaney/MiMo-V2.6-Flash-REAP50-GGUF` 158.1 GiB (2026-10-12: MXFP4_MOE 86.06 GiB, Q2_K hybrid 61.64 GiB, mmproj BF16/Q8_0, MTP BF16/Q8_0); `tacodevs/MiMo-V2.6-Flash-RL-MLX-REAP25-mixed-GPTQ-MTP`, `-REAP25-mixed3bit-GPTQ-MTP`, `-REAP37-mixed-GPTQ-MTP`, `-REAP50-mxfp4-MTP` (MLX) — [HF API](https://huggingface.co/api/models?search=MiMo-V2.6-Flash); local GGUF card `/home/patrickd/xiaomi-2.6-flash-REAP/gguf_README.md`
- Local REAP-50 config: `n_routed_experts 128` (of 256), 48 layers, `num_key_value_heads 4` (GA) / `swa_num_key_value_heads 8`, `sliding_window 128`, `num_nextn_predict_layers 3`, 1,048,576 ctx; on-disk 88 GiB (du) — [local /home/patrickd/models/MiMo-V2.6-Flash-REAP50/config.json]
- REAP run record: "REAP COMPLETE ... (criterion reap_1_1_1 / hope at 0.50)", 27/27 chunks, uniform allocation, published PUBLIC 2026-10-16 — [local /home/patrickd/xiaomi-2.6-flash-REAP/logs/WATCHDOG_STATUS.txt, PUBLISH_STATUS.txt]
- Cerebras REAP collection ships no MiMo-V2 checkpoint; "No Cerebras/0xSero REAP of ... MiMo-V2 was found on HF" — [sibling note sota_shipped_artifacts.md](https://huggingface.co/collections/cerebras/cerebras-reap)
- Ceiling, generalist: "no primary source reports a general-purpose (multi-domain, MC/knowledge-preserving) MoE surviving >50% expert pruning within 10% of base; every >50% result in the literature is a domain specialist ... evaluated only in-domain, and the best of those (75% pruned) needed a healing pass and still paid 4-6% rel on the target domain while general ability collapsed (ppl +131% to +500%)" — [sibling note expert_pruning_beyond_50.md, summary]
- At 50%: Qwen3-Coder-480B loses 1.7 pt Code Avg and 5.8 pt MC; fine-grained general models (Qwen3-30B-A3B, GLM-4.5-Air, Kimi-K2) lose 14-22 pt on multiple-choice — [sibling note expert_pruning_beyond_50.md Q1, citing REAP arXiv:2510.13999](https://arxiv.org/abs/2510.13999v1)
- Specialist points: DeepSeek-R1 62.5% (PreMoe): MATH-500 −0.6 pt but GPQA −8.6, LCB −8.1; 75%: GPQA −27.8, LCB −19.5; Qwen3.6-35B-A3B (256/8, MiMo-Flash's expert granularity) 62.5%: HumanEval+ 0.890→0.860, general ppl 6x; 75%: HumanEval+ −10.3 pt, routing-mass/random collapse; 75% + heal: 0.841 (−5.5% rel); Gemma-4-26B-A4B 75%: HumanEval+ 0.000 (total collapse) — [Half the Experts arXiv:2607.16721](https://arxiv.org/pdf/2607.16721); [PreMoe](https://arxiv.org/html/2505.17639)
- Shared-expert dependence of recovery: Half-the-Experts heals with "LoRA r64 on attention/shared expert, router full-trained, routed experts frozen ... Recovery thus comes entirely from re-routing and from the shared, always-on pathway" — [Half the Experts §5.7](https://arxiv.org/pdf/2607.16721)
- MiMo has no shared expert: Pro `n_shared_experts: null`; tech report: "Both the SWA block and the GA block utilize a sparse MoE FFN without shared experts" — [config.json](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-MOPD/resolve/main/config.json); tech report §2.1 via [sibling note target_models.md]. Flash likewise "256 experts, top-8, NO shared expert" — [local memory mimo-v26-flash-thor.md]
- Flash REAP-50 mask audit at ≤4,095 tokens: 1.13% of routed slots / 8.71% of tokens hit a pruned expert; layer 7 alone 18.18% of slots — [local wiki 92-long-context-program.md §3]
- Post-pruning failure modes that PPL hides: looping (GLM-5.2 34% REAP: loop rate 3.6%→7.2%, "cannot fully restore the termination behavior" with router KD); Gemma-4 at 50% "free-running generation collapses into repetition while perplexity looks fine"; 75% Qwen3.6 variants degenerate on 2-6 of 8 smoke prompts — [sibling note expert_pruning_beyond_50.md Q7]
- Pruning vs quant damage on the same model: REAP-50 = +21.0% ppl vs one GGUF bit step +1.46% → "Pruning is ~14x more damaging per unit of compression than quantization" — [local memory reap-criterion-and-corpus-audit.md]

### Inferences
- MiMo's no-shared-expert design removes the always-on pathway that the only measured >50% healing recipe relied on; with 384 experts top-8 (48:1), a pruned-expert hit has nowhere to fall except another routed expert. This makes >50% deletion on Pro worse-than-literature, not better — the literature's 62.5-75% specialist points were on models with a shared expert (Qwen3.6-35B) or on DeepSeek-R1 (shared expert present) [INFERENCE from the cited recipes].
- Pro would need 82-84% removal at native width (keep 61-69 of 384); the only >75% datapoints are translation-only (GPT-OSS-20B 81-87.5% with full-parameter distillation, −5 to −7% in-domain) and Gemma-4 75% total collapse. No datapoint at 82% exists for any model with 384 experts [EXTRAPOLATION beyond all data].
- The 8.71%-of-tokens pruned-hit rate on Flash at 50% is the only MiMo-family measurement of "what breaks": routing mass has to be reassigned for ~9% of tokens with no shared expert absorbing it; at 82% removal that fraction would scale far above the point where Half-the-Experts saw collapse.

### Gaps
- No public evaluation (benchmarks, NIAH, LongPPL) of any MiMo-V2.6 REAP artifact, including the user's own, was found; the local wiki lists NIAH/LongPPL/agentic evals as planned, not run.
- No paper isolates the shared-expert vs no-shared-expert effect on prune tolerance; the inference above is from recipe descriptions.

## KQ4. KV at 51,200 B/token on GQA-128/8: when does it exceed the budget, which reductions are proven

### Takeaway
KV alone never exceeds the 117 GiB envelope inside the native 1M window (BF16 KV reaches 117 GiB at 2.45M tokens; with the 26 GiB NE+runtime floor at 1.91M), but at 1M it is 50 GiB and halves the expert budget; the binding constraint is the expert block, and only at the 75%+3.0-bpw corner does KV become the limiter (3.7 GiB left → 77k tokens BF16 / 154k FP8). FP8 KV is proven on GQA-8 models to 1M (vLLM MRCR), 2-bit KIVI on Llama-3-8B-Instruct (GQA-8) costs 0.8 LongBench points, SnapKV-style eviction holds NIAH at 32x; post-hoc KV-head deletion and GA-layer reduction have no published results on GQA-8 models.

### Cited Findings
- Thresholds (my arithmetic): 117 GiB / 51,200 B = 2.45M tokens (BF16), 4.91M (FP8); (117 − 18 − 8) GiB / 51,200 B = 1.91M tokens BF16, 3.82M FP8. KV at 128k = 6.25 GiB (5.3% of envelope); 256k = 12.5; 512k = 25; 1M = 50 GiB (42.7%).
- At the only "fitting" specialist corner, 75% removal @3.0 bpw: experts 87.3 + NE 18 + R 8 = 113.3 → 3.7 GiB for KV → 77k tokens BF16 / 154k FP8. At 75% @2.5 bpw (+19.6% PPL): 18.2 GiB → 382k BF16 / 764k FP8 (my arithmetic).
- Pro vs peers: 51,200 B/token is 2.22x MiMo-Flash's 23,040 B (9 GA × 4 KV × 320 × 2) and ~57x DeepSeek-V4.1-Flash's 890 B/token — [local memory thor-base-model-shortlist.md]; caller's "~57× DeepSeek-V4.1" matches.
- FP8 KV, vLLM (2026-04): Hopper FA3 128k NIAH BF16 91% → FP8 13% before fix → 89% after two-level accumulation; Llama-3.3-70B-Instruct (GQA, 8 KV heads) MRCR recovers 97-98% of baseline AUC@128k; Qwen3-30B-A3B-Instruct MRCR to 256k ~94-98%; Qwen3.5-27B MRCR to 1M "matched baseline aggregate AUC@1M"; reasoning evals within 1-2 pt; B200/FlashInfer "accumulation issue does not occur"; no INT4 KV results — [vLLM blog: The State of FP8 KV-Cache](https://vllm.ai/blog/fp8-kvcache)
- KIVI 2-bit (per-channel K, per-token V, G=32, R=128): Llama-3-8B-Instruct (GQA, 8 KV heads) LongBench 45.21 → 44.37 (KIVI-4: 45.31); Mistral-7B-Instruct-v0.2 (GQA, 8 KV heads) 43.54 → 42.43 (KIVI-4: 43.53); Falcon-7B (MQA, 1 KV head) "needs 4-bit KIVI to maintain accuracy"; 2.6x peak memory reduction incl. weights — [KIVI arXiv:2402.02750 Tables 4, 8, 9](https://arxiv.org/html/2402.02750v2)
- KVQuant 3-bit: "< 0.1 perplexity degradation with 3-bit quantization on both Wikitext-2 and C4" across LLaMA, Llama-2, Llama-3, Mistral; the 1M-context demo is LLaMA-7B (MHA) on one A100-80GB — [KVQuant arXiv:2401.18079](https://arxiv.org/abs/2401.18079v6)
- llama.cpp: `--cache-type-k/-v` allowed values f32, f16, bf16, q8_0, q4_0 (local build `tools/server/README.md:71`); community figures Q4 K + Q8 V +0.4% ppl, Q4/Q4 +2.1% ppl (secondary, unverified) — [techplained](https://www.techplained.com/kv-cache-quantization); a DeepSeek-V4-Flash llama.cpp KV-quant experiment found "average perplexity almost unchanged while materially changing the model's actual token choices" — [dev.to breachprotocol](https://dev.to/breachprotocol/quantizing-v4-flashs-kv-cache-in-llamacpp-changes-which-tokens-it-picks-38np)
- Token eviction: SnapKV on Command-R at 128k, 32x KV compression, modified NIAH 9.866 → 9.819 (−0.5%); 380k tokens on one A100-80GB — [SnapKV arXiv:2404.14469](https://arxiv.org/pdf/2404.14469); at 10% KV retention on Llama-3.1-8B (GQA-8): SnapKV NIAH 73.6 vs StreamingLLM 18.6; H2O "significant accuracy degradation" — [FastKV arXiv:2502.01068](https://arxiv.org/pdf/2502.01068); KVzap 2-4x compression "negligible accuracy loss" on Qwen3-8B, Llama-3.1-8B-Instruct, Qwen3-32B — [KVzap arXiv:2601.07891](https://arxiv.org/html/2601.07891)
- Head-level: DuoAttention (retrieval vs streaming heads, streaming heads keep a fixed small cache) "memory cuts of up to 1.67x for GQA models, with minimal accuracy loss"; HeadKV/FastKV allocate per head — none deletes KV heads — [DuoAttention arXiv:2410.10819](https://arxiv.org/html/2410.10819v1)
- Prior local finding on 128k vs FP8 on Thor: "our FA3 build crashes and MLA requires TRITON_MLA" — [local memory dflash-thor-build.md; mistral-small4-thor-stack.md]

### Inferences
- The proven ladder for a GQA-8 model is FP8 (≈lossless to 1M per vLLM, 2x) → KIVI-style 2-bit (~0.8-1.1 LongBench pt on GQA-8, 4-8x) → SnapKV/KVzap eviction (2-32x, NIAH-safe); INT4 K + Q8 V in llama.cpp sits between the first two. All of these are token/precision reductions; none changes the model.
- KV-head pruning on Pro's 8 KV heads and dropping GA layers (10 of 70) are architecture changes with no post-hoc results on any GQA-8 model; the SWA layers already contribute only ~37.5 MiB, so there is no further "layer" lever short of retraining [INFERENCE].
- For Pro, 128k is feasible only as a KV matter: at the 75%+3.0 bpw corner it needs FP8 KV to reach 154k; at any corner that keeps ≥25% of experts at ≥3.5 bpw nothing fits regardless of KV.
- DuoAttention-style head splitting (1.67x on GQA) would take Pro's 1M KV from 50 → ~30 GiB BF16, still 25% of the envelope [EXTRAPOLATION from a 7-8B-model result].

### Gaps
- No paper reports deleting KV heads or GA layers post-hoc on a 128/8 GQA model at long context.
- KIVI/KVQuant results are on 7-8B models; no 2-3-bit KV result on a ≥300B MoE at ≥128k was found. The vLLM FP8 result on Llama-3.3-70B (GQA-8) is the largest matched-architecture datapoint.

## KQ5. MiMo-V2.6-Flash as the alternative: size, experts, bytes, KV/token, REAP-50/60 arithmetic, prior measurement

### Takeaway
Flash is 309B/15B, 48 layers (47 MoE, 256 experts top-8, no shared expert), 9 GA + 39 SWA-128, experts 149.8 GiB native MXFP4 (92.9% of 161 GiB index), KV 23,040 B/token BF16 (2.22x less than Pro); the user's REAP-50 is 88 GiB on disk and fits 1M at FP8 KV with 9.5 GiB spare, 512k at BF16 KV, and fits at the literature's 50% ceiling rather than 3.9x beyond it. It was sized and published, not served: no serve/bench log, no GGUF on local disk, and the "~5 tok/s" is derived (B_tok ÷ measured bandwidth), not measured.

### Cited Findings
- Flash: 309B/15B active, 48 layers (47 MoE + layer 0 dense), 9 full-attention + 39 SWA-128 at zeros [0,5,11,17,23,29,35,41,47], GA 4 KV heads / SWA 8 KV heads, head_dim 192 / v_head_dim 128, 256 experts top-8 no shared expert, MTP 3 layers + separate 5-layer DFlash drafter, 1,048,576 ctx, MIT; index total 172.9 GB (161.0 GiB), repo 177.8 GB; experts 158.4 GB `U8` = 92.9%; 13.37 MB per expert; attention 6.09 GB (o_proj BF16 in `ignored_layers`); non-expert ≈ 14.3 GB — [local survey MIMO_V26_FLASH_SURVEY_2026-09-23.md §2-3; local memory mimo-v26-flash-thor.md]
- Expert bytes (my arithmetic, matches the 13.37 MB/expert header figure): 47 × 256 × 3 × 4096 × 2048 × 0.53125 B = 160.9 GB = **149.8 GiB**; REAP-50 = 74.9 GiB; REAP-60 = 59.9 GiB; REAP-40 = 89.9 GiB. REAP-50 + 13.3 GiB non-expert = 88.2 GiB ↔ measured 88 GiB on disk / 87.9 GiB on HF / 94.7 GB tacodevs MLX REAP50 — [local du; HF API; local survey §3]
- Flash KV/token: 9 GA × 4 KV × (192+128) × 2 B = **23,040 B BF16 / 11,520 B FP8**; 39 SWA layers window-capped ≈ 12.8 MB total — [local survey §3]. Note two prior local figures disagree: memory note uses 11,520 B (FP8) → "11.5 GB at 1M"; wiki 92 uses 27.00 GiB at 1M (which corresponds to 192+192 at BF16, i.e. it over-counts V by 192/128). Correct BF16 at 1M = 22.5 GiB, FP8 = 11.25 GiB (my arithmetic).

**Flash fit table (E = 117, R = 8 GiB, weights = experts × (1−removal) + 13.3 GiB non-expert at BF16 attention):**

| REAP | weights | 128k BF16 / FP8 | 256k BF16 / FP8 | 512k BF16 / FP8 | 1M BF16 / FP8 |
|---|---|---|---|---|---|
| 40% | 103.2 | 114.0 ✓ / 112.6 ✓ | 116.8 ✓(+0.2) / 114.0 ✓ | 122.5 ✗ / 116.8 ✓(+0.2) | 133.7 ✗ / 122.5 ✗ |
| **50%** | **88.2** | 99.0 ✓ / 97.6 ✓ | 101.8 ✓ / 99.0 ✓ | 107.5 ✓ / 101.8 ✓ | **118.7 ✗ (−1.7) / 107.5 ✓ (+9.5)** |
| 60% | 73.2 | 84.1 ✓ / 82.6 ✓ | 86.9 ✓ / 84.1 ✓ | 92.5 ✓ / 86.9 ✓ | 103.7 ✓ / 92.5 ✓ |

- Local README table (125.6 GB usable, ~5 GB workspace): REAP-50 94.7 + 11.5 KV@1M + 2.94 DFlash + ~5 = 114.1 GB ✓; REAP-40 129.7 ✗ — [local /home/patrickd/xiaomi-2.6-flash-REAP/README.md]
- Serving evidence: no `llama-server`/`vllm serve`/`llama-bench` invocation or output for the REAP50 exists in shell history, logs (`gguf_ship.log`, `publish.log`, `reap_run.log`), systemd units (only `reap-publish`, `reap-watchdog`), and no `*MiMo*gguf` > 1 GB is on local disk (`gguf/` is empty; the GGUFs were pushed to HF) — [local grep 2026-10-17]. The wiki states the 1M fit as arithmetic: "Treat 1M-on-MiMo as 'demonstrable' rather than 'operationally comfortable'. [EST]" and lists NIAH/LongPPL as `[OPEN]` — [local wiki 92-long-context-program.md §1, §3-4]
- The "~5 tok/s base, ~15 with DFlash" is derived: "B_tok 13.5 GB short / 25 GB at 1M. Divide by our MEASURED 57-66 GB/s effective" — [local memory mimo-v26-flash-thor.md]; recomputed here: experts 8×47×13.37 MB = 5.03 GB + attention 6.09 + lm_head 1.25 + dense/gates 1.40 = 13.8 GB → 4.1-4.8 tok/s at 57-66 GB/s, 17.4 at 240 GB/s (my arithmetic).
- Capability (third-party, same harness): Artificial Analysis Intelligence Index MiMo-V2.6-Pro **46**, MiMo-V2.6-Flash **38** (#11 of 120; comparable-model median 19; 51.9 tok/s hosted; text+image) — [AA Pro](https://artificialanalysis.ai/models/mimo-v2-6-pro); [AA Flash](https://artificialanalysis.ai/models/mimo-v2-6-flash). Vendor card: Flash Terminal-Bench 4.0 28.8 vs GLM-5.3-Flash 33 — [local survey §6]
- Public Flash sub-4-bit/REAP comparators: ggml-org Flash Q2_K 126 GB (over envelope), MXFP4 167 GB; tacodevs MLX REAP-25/37/50; AMAImedia BPW2.5, fresherbz IQ2_XXS, EXL3 2.2-2.5 bpw builds — all without published task evals — [HF API](https://huggingface.co/api/models?search=MiMo-V2.6-Flash); [local survey §5]

### Inferences
- The Flash REAP-50's 1M claim depends on FP8 KV (or ≤6.3 GiB runtime at BF16): at BF16 KV + 8 GiB runtime it misses by 1.7 GiB. FP8 KV is the proven-lossless lever on GQA-8 (KQ4), so the claim stands, but it has not been exercised.
- REAP-40 Flash is a 256k (BF16) / 512k (FP8) model; REAP-60 would serve 1M at BF16 KV but is above the generalist ceiling. REAP-50 at FP8 KV is the only point that is simultaneously (a) at the ceiling, not beyond, (b) 1M-capable, (c) already built.
- Pareto comparison with the best Pro corner (75% removal, 3.0 bpw, ~77k ctx BF16): Flash REAP-50 is at 50% removal with native-precision experts (no second quant), 2.22x less KV per token, 2.8x fewer active bytes per token, 1M-capable; it loses 8 AA points of base capability (46 → 38) before either compression, whereas Pro's corner stacks a 75% prune (literature: general collapse) on a +9.8% PPL double-quant. Flash dominates.

### Gaps
- No measured decode, NIAH, or LongPPL for the Flash REAP-50 on Thor; "~5 tok/s" and "fits 1M" are both derived. A serve at `-c 1048576 --cache-type-k q8_0 --cache-type-v q8_0` with the published MXFP4_MOE GGUF would settle both in one run.
- Which KV dtype the local vLLM/llama.cpp path would actually use for the GA layers at 1M is unverified; the local llama.cpp build has q8_0/q4_0 KV types, FA3 is known to crash on this box.

## KQ6. Serve-time: runtime for MXFP4 experts on Thor, decode bytes/token for 384-expert top-8

### Takeaway
The local llama.cpp (2026-09-07, built for CUDA arch 110) has the `mimo2` architecture and native MXFP4 MMQ/MMVQ CUDA kernels ("MXFP4 and NVFP4 on Blackwell" SRAM layout), so Pro's experts would run natively; vLLM's MXFP4 MoE kernels are documented for SM100 (B200) only, and sm_110 is unverified. Decode is memory-bound at ~36.6 GiB (39.3 GB) read per token for Pro at short context — 1.45-1.68 tok/s at the 57-66 GB/s this box has ever achieved, 6-7 tok/s at rated bandwidth — and pruning does not reduce it.

### Cited Findings
- Local llama.cpp HEAD 465e49b9c (2026-09-07): `src/llama-arch.cpp:147 { LLM_ARCH_MIMO2, "mimo2" }`; `ggml/src/ggml-cuda/mmq.cuh:128 GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, // MXFP4 and NVFP4 on Blackwell`, `case GGML_TYPE_MXFP4` tile loaders; `vecdotq.cuh:310 vec_dot_mxfp4_q8_1` (MMVQ, the decode path); `CMakeCache.txt: CMAKE_CUDA_ARCHITECTURES=110` — [local /home/patrickd/llama.cpp]
- The user's own GGUF converter "natively repacks these blocks directly into GGMLQuantizationType.MXFP4, avoiding costly lossy dequantization cycles"; recommended hardware for the 86.06 GiB MXFP4_MOE file: "1x 128GB Thor" — [local gguf_README.md]; conversion log: `mimo2.context_length u32 = 1048576` — [local logs/gguf_ship.log]
- ggml-org ships Pro GGUFs with "MTP sidecars (Q4_0 and Q8_0) ... DFlash drafter sidecar (BF16 and Q8_0)" — [ggml-org/MiMo-V2.6-Pro-MOPD-GGUF](https://huggingface.co/ggml-org/MiMo-V2.6-Pro-MOPD-GGUF)
- vLLM MXFP4: "a FlashInfer MoE kernel on Blackwell that runs on native MXFP4 tensor cores" (B200, SM100); v0.20.0 "MXFP4 W4A4 CUTLASS MoE for SM100"; v0.21.0 adds "FP8 on NVIDIA Thor/SM110 (#39712)" — FP8, not MXFP4 — [vLLM gpt-oss blog](https://vllm.ai/blog/gpt-oss); [vLLM v0.20.0](https://newreleases.io/project/pypi/vllm/release/0.20.0); [vLLM v0.21.0](https://newreleases.io/project/pypi/vllm/release/0.21.0). Local experience on Thor: NVFP4 MoE needed the cutlass path + TRITON_ATTN; FLASHINFER was invalid for MLA; FA3 crashes — [local memory dflash-122b-marlin-moe-crash.md; mistral-small4-thor-stack.md]
- Thor sm_110a: "tcgen05 NO, cp.async/FP4x2-unpack YES" — [local memory dspark-decode-gap-research.md]
- Measured effective bandwidth on this box's own servers: 57-66 GB/s (GLM 11.86 tok/s at B_tok 4.8 GB; DSV4-0731 14.61 tok/s); "Rated bandwidth has never been achievable here" — [local survey §4; local memory glm53-decode-gap-profile.md]; rated 240 GB/s measured peak — [local memory thor-model-fit-frontier.md]
- Pro bytes/token at short context (my arithmetic from the header split): experts top-8 × 69 layers × 3 × 2048 × 6144 × 0.53125 B = 10.31 GiB; attention (all 70 layers, `o_proj` BF16 13.13 + `qkv` FP8 10.87) = 24.00 GiB; lm_head 1.75; gates 0.30; dense-0 0.28 → **36.64 GiB = 39.3 GB/token** → 1.45 tok/s @57, 1.68 @66, 6.1 @240, 6.9 @273 GB/s. With attention requantized to 4-bit (24 → 9.8 GiB): 22.4 GiB = 24.1 GB → 2.37-2.74 tok/s effective, 10.0 @240. KV read adds 6.25 GiB/token at 128k BF16, 50 GiB at 1M.
- Flash bytes/token: 13.8 GB → 4.1-4.8 tok/s effective, 17.4 @240 (my arithmetic, consistent with the prior 13.5 GB figure).
- Decision-procedure ceiling rule "243.6 / (active × 0.62), ×0.47 for real throughput, reject <20 tok/s ceiling": Pro 243.6/(42×0.62) = **9.4 tok/s ceiling** (4.4 real); Flash 26.2 (12.3 real) — [local memory thor-base-model-shortlist.md]
- Spec-decode on 128 GB-class boxes scales inversely with active size/expert count: MiMo-V2.5 310B NVFP4 on 2× Spark 22.3 → 27.6 prose / 45.4 code tok/s with DFlash; GLM-5.3-Flash GGUF MTP +34%; DeepSeek-V4-Flash + DSpark a net slowdown — [sibling note thor_serving_performance.md, citing the NVIDIA forum thread](https://forums.developer.nvidia.com/t/mimo-v2-5-dflash-speculative-decoding-on-a-2x-dgx-spark-pair-22-67-tok-s-depending-on-workload/375607); local: "Spec-decode MoE wall... 0.95x best, bounded 1.27x even at perfect acceptance" on GLM-5.3 NVFP4 — [local memory glm53-spec-decode-moe-wall.md]

### Inferences
- Pro's attention block alone (24 GiB BF16/FP8 read per token across all 70 layers) is 1.7x Flash's entire per-token read; `o_proj` at BF16 (13.13 GiB) is the largest single term and the NVFP4 dense overlay would cut it to ~3.3 GiB. Even so, Pro's floor is ~22 GiB/token vs Flash's ~12.8 GiB.
- Pro fails the project's own decode gate (9.4 tok/s ceiling < 20) independently of the memory gate; pruning cannot fix it because top-8 is unchanged.
- llama.cpp is the only runtime on this box with a confirmed native-MXFP4 path for `mimo2` on sm_110; vLLM MXFP4 MoE on Thor is unverified and the prior Thor vLLM experience (cutlass fallback, FA3 crash) suggests it would need the same workarounds.

### Gaps
- No vLLM issue/PR confirming or denying MXFP4 MoE kernels on sm_110 was found.
- No measured tok/s for any MiMo-V2.6 (Pro or Flash) on Thor; all decode numbers above are bandwidth models.

## KQ7. Decision statement

### Takeaway
NO-GO on MiMo-V2.6-Pro-MOPD in every regime; GO on MiMo-V2.6-Flash REAP-50 at FP8 KV, with the two open measurements (serve at 1M, decode tok/s) as the next step.

### Cited Findings
- (a) **Generalist: NO-GO.** Required removal at native 4.25 bpw is 81.9% (32k) / 82.9% (128k) / 84.1% (256k) vs the 50% documented ceiling; holding 50% requires 1.35-1.62 bpw on QAT-MXFP4 experts, below the lowest measured point (2.0 bpw = +43.7% PPL) [tables in KQ1; ceilings in KQ3].
- (b) **Multi-domain specialist: NO-GO.** The 62.5% point (math on DeepSeek-R1, already −8.6 GPQA / −8.1 LCB off-domain) needs 2.0 bpw (+43.7% PPL) to fit at 32k and does not fit at 128k (bpw needed 1.9x) [KQ1 totals table; KQ3].
- (c) **Single-domain specialist: technically fits at 75% removal + ≤3.0 bpw at 32k (114.9 GiB, 2.1 spare, ~77k tokens of BF16 KV), stacking the literature's worst measured in-domain loss (−5.5% rel after healing, general ppl +500%) with a +9.8% PPL / 0.189 KLD double quant, on a model with no shared expert to heal through, at ~1.5-2.7 tok/s; no healing dataset, no eval, and an 8-GPU/2-node reference stack** [KQ1-KQ3, KQ6]. This is below any bar this project has set for a servable artifact (reject <20 tok/s ceiling; reject >50% prune) — [local memory thor-base-model-shortlist.md decision procedure].
- **Flash REAP-50: GO.** 88 GiB weights + 11.25 GiB FP8 KV at 1M + 8 GiB runtime = 107.5 GiB (9.5 spare); 99.0 GiB at 128k BF16; at the 50% ceiling, native-precision experts, 23,040 B/token KV, AA index 38 (vs Pro 46, both unpruned), B_tok 13.8 GB → ~4-5 tok/s effective, llama.cpp `mimo2` + MXFP4 kernels confirmed on arch 110, artifact already published with mmproj + MTP sidecars [KQ5, KQ6].

### Inferences
- The 8-point AA gap (46 vs 38) is the whole case for Pro, and it is a property of the unpruned 1T model; after an 82% prune nothing in the literature suggests the pruned Pro would retain more capability than an unpruned-at-50% Flash, and the one measured MiMo-family number (8.71% of tokens hitting a pruned expert at 50% on Flash) says the routing stress is already material at the ceiling.
- Next action is a measurement, not a build: serve `patrickbdevaney/MiMo-V2.6-Flash-REAP50-GGUF` MXFP4_MOE (86.06 GiB) on the local llama.cpp with `-c 1048576 --cache-type-k q8_0 --cache-type-v q8_0`, record tok/s and run NIAH; this converts both derived claims ("fits 1M", "~5 tok/s") into measured ones (detached per the local rule, never beside another CUDA process).

### Gaps
- Pro's AA index is for the hosted unpruned model; no measurement of any pruned 1T-class MoE at >75% exists to bound what Pro-at-82% would score.
