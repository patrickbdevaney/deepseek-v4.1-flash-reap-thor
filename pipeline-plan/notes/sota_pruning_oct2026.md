# SOTA MoE expert pruning / saliency / merging (Jul-Oct 2026) and community GLM-5.3-Flash / DeepSeek-V4.x-Flash compressions

Scope: methods that could beat a well-tuned 50% REAP/HOPE mask for DeepSeek-V4.1-Flash-1017 (384 routed experts/layer, experts shipped MXFP4) on one Jetson AGX Thor (117 GiB). Researched 2026-10-19. Repo sizes below come from the HF `api/models/<repo>/tree/main?recursive=true` file listings, summed by me, in decimal GB. They are not taken from the model cards. Vendor and community quality numbers are **unverified self-reports** unless marked otherwise.

Architecture facts used below:
- **GLM-5.3-Flash has 288 routed experts per layer, not 384.** It has 42-45 MoE layers, about 321B total and 18B active parameters. [lapadatbogdan](https://huggingface.co/lapadatbogdan/GLM-5.3-Flash-REAP144-Coder-GGUF), [lte-ai](https://huggingface.co/lte-ai/GLM-5.3-Flash-Strata-Q2Q3-Experts) and [pfeifferj](https://huggingface.co/pfeifferj/GLM-5.3-Flash-GSQ-RCO-GGUF) all give these figures.
- **DeepSeek-V4.1-Flash** has 40 layers with 384 routed experts each and no hash layers. The routed experts total 543.6B parameters (72%). The two Engram tables total 196.6B parameters (26%), and the whole model is 754.6B parameters. Source: [pipenetwork card](https://huggingface.co/pipenetwork/DeepSeek-V4.1-Flash-REAP50-MLX-mixed-4_8bit). This 754.6B conflicts with the ~763B in the brief.
- **DeepSeek-V4-Flash-0731** is a different shape: 256 experts, top-6, 43 layers. Source: [puwaer](https://huggingface.co/puwaer/DeepSeek-V4-Flash-0731-reap-150b).

## Q1. Which Jul-Oct 2026 papers improve expert saliency, mask search, progressive prune-then-heal, router re-calibration or merging at >=50% for fine-grained MoEs?

### Takeaway
No Jul-Oct 2026 paper reports beating HOPE. No paper I found has been evaluated on a model with 384 experts. HOPE (2609.18916) is still the strongest published one-shot selection criterion for fine-grained MoEs at 40-50% pruning. Two other recent papers deserve a look: MoRA (2610.00367) learns router biases and adds a cheap per-channel affine "proxy" for each removed expert, and 2607.16721 shows the best criterion changes from model to model. The best evidence on our own model family is community work on GLM-5.3-Flash, not papers. It found that the calibration domain and the REAP-sum aggregation matter more than the choice of criterion. It also found that naive KL mask search, which scores masks without re-routing, was worse than a frequency mask.

### Cited Findings
**HOPE: higher-order (pairwise) expert pruning, arXiv 2609.18916, Sep 2026**
- REAP is a special case of HOPE with the pairwise interaction terms set to zero. The pruning objective is second-order, pᵀFp. — [arXiv 2609.18916](https://arxiv.org/abs/2609.18916)
- HOPE beat REAP in every agentic experiment at 40-50% pruning, by +2.8% on average and up to +6.1%. Its lead grows with the pruning rate. — [arXiv 2609.18916](https://arxiv.org/abs/2609.18916)
- HOPE needs "marginally more" calibration data than REAP because it estimates statistics between experts. — [arXiv 2609.18916 (v2 html)](https://arxiv.org/html/2609.18916v2)
- Our earlier project note adds the following. HOPE was evaluated on Qwen3.5-122B-A10B (256 experts, top-8) with coding and SWE-trajectory calibration. Its average rank was 1.58 against REAP's 2.42. The code is in awslabs/hybrid-model-factory. — internal note `hope-supersedes-reap.md`, summarising [arXiv 2609.18916](https://arxiv.org/abs/2609.18916)

**MoRA: router bias learning plus expert approximation, arXiv 2610.00367, 30 Sep 2026 (HIT Shenzhen)**
- **Method:**
  - MoRA adds a learnable scalar bias to each expert's frozen router logit. It trains the biases on LM loss plus a routing-diversity penalty, for 1000 iterations on C4.
  - It removes the experts with the lowest learned bias, then discards the biases.
  - For each removed expert it fits a per-channel affine map (scale and bias) from one retained expert's output to the removed expert's output, by weighted ridge regression. It uses the retained expert with the highest R² as the proxy, and only if R² > 0.4.
  - This is not weight merging. The fit uses 2x64 sequences of 1024 tokens from C4.
  - — [arXiv 2610.00367](https://arxiv.org/html/2610.00367)
- **Results at 50% pruning, mean of nine zero-shot tasks:**

  | Model | MoRA | Best baseline |
  |---|---:|---|
  | Qwen3-30B-A3B (128 experts) | 65.83 | 59.40 (MoNE) |
  | DeepSeek-V2-Lite | 54.40 | 53.84 (HC-SMoE) |
  | Moonlight-16B-A3B | 55.36 | 54.39 (MoNE) |

  — [arXiv 2610.00367](https://arxiv.org/html/2610.00367)
- **Limits of the evidence:**
  - MoRA was **not compared against REAP or HOPE**.
  - It has not been tested with 384 experts or with noaux_tc routing.
  - The authors say the link between bias and sensitivity is stronger in models with more experts, but they did not test it at our scale.
  - The affine parameters are under 0.08% of the retained parameters, and the extra FLOPs are under 0.0005%.
  - The code is "to be released".
  - — [arXiv 2610.00367](https://arxiv.org/html/2610.00367)

**"Half the Experts, All the Code", arXiv 2607.16721, 18 Jul 2026 (independent author)**
- **The best criterion depends on the model.** Results at 50% keep, HumanEval+, Q8_0:

  | Model | REAP | Routing mass | Other criteria |
  |---|---:|---:|---|
  | Qwen3.6-35B-A3B (256 experts; base HE+ 0.890) | 0.896 | 0.872 | MAN 0.884 |
  | Gemma-4-26B-A4B (128 experts; base HE+ 0.927) | 0.762 | 0.890 | |

  On Gemma, routing mass beats REAP by +12.8 points, CI [+6.7, +18.9]. At 25% keep, REAP scores 0.000 on Gemma. — [arXiv 2607.16721](https://arxiv.org/html/2607.16721)
- **Expert magnitude is the signal that matters.** A domain-contrastive score collapses when used alone (HE+ 0.201 at 50% on Qwen). — [arXiv 2607.16721](https://arxiv.org/html/2607.16721)
- **Calibration domain:**
  - Calibrating on coding made a 50% prune nearly lossless on HE+ for Qwen, but general-chat perplexity rose 2.3x.
  - Perplexity can mislead. A random prune of Qwen had better held-out code perplexity than the base model (4.82 vs 5.38) yet lost 57.9 points of pass@1.
  - — [arXiv 2607.16721](https://arxiv.org/html/2607.16721)
- **Pruning versus quantization at equal memory:**
  - Quantizing the full model is safer wherever 3 bits or more fit.
  - Below that, the pruned specialist wins. A pruned Q5 beats the base at Q2_K by +6.7 HE+ points (p=0.003).
  - — [arXiv 2607.16721](https://arxiv.org/html/2607.16721)
- **Healing:** self-distillation with the routed experts frozen raised HE+ at 25% keep from 0.787 to 0.841. — [arXiv 2607.16721](https://arxiv.org/html/2607.16721)

**Generic TB-Coverage, arXiv 2607.01710, 2 Jul 2026**
- It scores each expert separately on each generic corpus (WikiText2 and C4), then applies a coverage rule that keeps the high-utility experts from each corpus. — [arXiv 2607.01710](https://arxiv.org/abs/2607.01710)
- It reports gains over REAP, random and ExpertSparsity at 25% and 50% retention. The models are only small, coarse MoEs (Qwen1.5-MoE-A2.7B and DeepSeek-MoE-16B). The abstract gives no exact numbers and no code is listed. — [arXiv 2607.01710](https://arxiv.org/abs/2607.01710)

**Router sensitivity under lightweight fine-tuning, arXiv 2608.07890 (v3, 10 Oct 2026, Columbia)**
- It scores each expert by the change in its router-row norm after a 500-step router-only LoRA, then prunes in one shot. It was tested only on Mixtral (8 experts) and Qwen1.5-MoE (20 experts). — [arXiv 2608.07890](https://arxiv.org/html/2608.07890)
- At 50%, its retention relative to the unpruned model is below REAP's on BoolQ (74.2 vs 94.8) and below AIMER's on GSM8K (23.5 vs 31.6). It is slightly ahead on ARC-C (86.1 vs REAP 84.0). The authors themselves call it "competitive at moderate compression". — [arXiv 2608.07890](https://arxiv.org/html/2608.07890)

**Context papers from before July 2026 that the newer work builds on**
- **REAM (2604.04356, Apr 2026)** merges grouped experts using gate-logit similarity plus output similarity. It reports a trade-off between MC and GEN scores that depends on the calibration mix. — [arXiv 2604.04356](https://arxiv.org/abs/2604.04356)
- **EvoESAP (2603.06003)** runs an evolutionary search over the number of experts kept in each layer, using REAP to rank experts within each layer. — [arXiv 2603.06003](https://arxiv.org/pdf/2603.06003)
- **A unified scoring formulation (2606.15716).** — [arXiv 2606.15716](https://arxiv.org/pdf/2606.15716)
- **SlimQwen (2605.08738, ICML 2026)** finds that progressive pruning schedules beat one-shot compression at equal training tokens, and that KD plus LM loss beats KD alone. This was in pre-training scale on Qwen3-Next. — [ICML 2026](https://icml.cc/virtual/2026/75227)
- **Upstream REAP changes.**
  - On 2026-03-11 REAP started renormalizing the top-k router logits (mean drop 1.9% vs 2.6%).
  - On 2026-03-30 it added a layer-wise observer for running on one GPU.
  - It has no HOPE support, and Cerebras has published no GLM-5.3 or DeepSeek-V4 checkpoints.
  - — [CerebrasResearch/reap](https://github.com/CerebrasResearch/reap)

### Inferences
- **Selection criterion.** No published criterion has been shown to beat HOPE at 50% on a fine-grained MoE. HOPE remains the default. The REAP-sum versus routing-mass reversal in 2607.16721 says no single saliency mass can be trusted per model. So candidate masks should be chosen by **held-out teacher KL with proper re-routing** (see Q4), not by assuming one criterion.
- **MoRA is the one cheap add-on with a mechanism we lack.** It is a per-channel affine proxy for each removed expert, under 0.1% extra parameters, fit by ridge regression with no gradient steps. It fits Thor well:
  - It runs after the mask is chosen.
  - It is calibration-only.
  - It plausibly composes with HOPE.
  - Its router-bias half matches V4's noaux_tc bias, which already exists.

  It has not been validated against REAP, so it needs a gate.
- **Merging is the weakest lever on the DSV4 family.** The puwaer REAM-versus-REAP numbers in Q3 show merging behind pruning at 132/256. MoRA's "approximate, don't merge" formulation is the safer version of the same idea.

### Gaps
- I found no independent replication of HOPE, and no HOPE run at 288 or 384 experts or on V4-family (noaux_tc-style biased) routing.
- I did not systematically search OpenReview, the HF papers page or r/LocalLLaMA. This was a time-bounded sweep, so some Jul-Oct 2026 papers may be missing.
- MoRA's bias-learning token count and batch size are not stated. Its cost on a 384-expert, 40-layer model is unknown.

## Q2. What is peasantsmith/GLM-5.3-Flash-Maya-GGUF, and does its method have anything we lack?

### Takeaway
Maya is **only a quantization recipe** plus a companion runtime, Project Maya, which offloads experts across VRAM, RAM and SSD. It does no pruning, merging or fine-tuning. All 288 experts are kept. The method has three parts:
- a per-expert imatrix computed from the FP8 model layer by layer;
- GPTQ-style error-feedback rounding of each expert's gate/up projections against that expert's own input statistics (from ISTA-DASLab), which cuts expert output error by about a quarter compared with plain imatrix rounding;
- a domain-weighted calibration set.

Of these, per-expert GPTQ rounding is the part that transfers to our V4.1 plan.

### Cited Findings
**What the repo contains**
- **Variants:** Maya-S, S24, M and L of zai-org/GLM-5.3-Flash. All include the NextN/MTP layer, and the license is MIT. The card describes no pruning or merging. — [Maya card](https://huggingface.co/peasantsmith/GLM-5.3-Flash-Maya-GGUF)
- **Sizes**, verified from the file listing:

  | Variant | Size (GB) | bpw (my calculation, 321B params) |
  |---|---:|---:|
  | Maya-S-v2 | 96.47 | about 2.40 |
  | Maya-S24 | 94.70 | about 2.36 |
  | Maya-M | 116.00 | about 2.89 |
  | Maya-L | 156.32 | about 3.90 |
  | vision / mmproj | 1.14 | |

  The bpw values are my own calculation. The card gives no bpw figure. — [Maya card](https://huggingface.co/peasantsmith/GLM-5.3-Flash-Maya-GGUF)
- **Per-tensor recipe for Maya-S:**
  - Routed gate/up: IQ2_XXS with error feedback.
  - Routed down: IQ3_XXS in the first and last 4 MoE layers, IQ2_S elsewhere.
  - Attention (KDA, MLA/DSA) and shared experts: Q6_K. S24 uses Q4_K here.
  - Small KDA projections and the DSA indexer: Q8_0.
  - Router, norms and stream-mixing weights: F32.
  - Maya-M uses IQ2_S gate/up and IQ3_S/IQ3_XXS down. Maya-L uses IQ3_S gate/up and Q5_K/IQ4_XS down.
  - — [Maya card](https://huggingface.co/peasantsmith/GLM-5.3-Flash-Maya-GGUF)
- **How it was made:**
  - Calibration was 128 sequences of 2048 tokens in GLM's chat template, with reasoning blocks included. It covered chat, multilingual chat, reasoning traces, web code (HTML/CSS/JS, three.js, WebGL), other code and tool calls. M and L were weighted further toward tool calls and front-end code.
  - Statistics came from the FP8 model layer by layer, with "an importance matrix for every expert separately".
  - Gate/up used GPTQ-style error-feedback rounding at the quantizer's 256-weight blocks, which "lowers the experts' output error by about a quarter against plain importance-weighted rounding at the same size".
  - The output is ordinary ggml GGUF.
  - — [Maya card](https://huggingface.co/peasantsmith/GLM-5.3-Flash-Maya-GGUF)
- **Quality against FP8, as self-reported by the author:**

  | Metric | FP8 | Maya-S | Maya-S24 | Maya-M | Maya-L |
  |---|---:|---:|---:|---:|---:|
  | Zero-shot avg, 5 tasks x 400 questions | 82.5 | 80.8 (97.9%) | 80.6 | 80.7 (97.9%) | 81.8 (99.2%) |
  | Top-1 agreement with FP8 | | 83.3% | 83.1% | 86.2% | 90.0% |
  | KL vs FP8 | | 0.428 | 0.444 | 0.329 | 0.188 |

  The token-level rows use 7,672 held-out positions. The card says outputs are "furthest on tool-call formats and on memorized text". — [Maya card](https://huggingface.co/peasantsmith/GLM-5.3-Flash-Maya-GGUF)
- **Speed, self-reported:** Maya-S decodes at up to 40 tok/s on 2x V100 32 GB with MTP drafting, and up to 19 tok/s on 1x V100 with 64 GB RAM. — [Maya card](https://huggingface.co/peasantsmith/GLM-5.3-Flash-Maya-GGUF)
- **The "Maya-M-Derisked" derivative** is Maya-M IQ2_S with Blackfrost_AI's proprietary "Directional Weight Modification", a refusal-ablation edit. It is 116.0 GB and byte-identical in size to the source. It is "**not** an SFT, DPO, distillation, expert-pruning, or router-surgery release". It has no drift eval. — [Derisked card](https://huggingface.co/peasantsmith/GLM-5.3-Flash-Maya-M-Derisked-IQ2_S-GGUF)
- **The author's other repos** are all quantization releases under "PS" or "Escha" names (IQ2_XXS/IQ4_NL GGUFs and EXL3), for Qwen3.5/3.6, gpt-oss, GLM-4.5-Air and Laguna. I found no pruning work. — [HF author listing](https://huggingface.co/peasantsmith)
- **The same technique family elsewhere:** pfeifferj used IST-DASLab's GSQ (Gumbel-Softmax grid learning) plus RCO (tensor bit allocation under an exact budget) on GLM-5.3-Flash.

  | Build | Size (GB) | MMLU-Pro | PPL vs Q8_0 | Approx. KL |
  |---|---:|---:|---:|---:|
  | Q8_0 | | 61.95% | | |
  | 3.0 bpw | 117.48 | 60.00% (-1.95 pp, p=0.0097) | +9.07% | 0.142 |
  | 3.5 bpw | 137.07 | 60.55% (-1.40 pp) | +4.49% | 0.071 |

  RCO "did not improve on step 0" for either target. — [pfeifferj GSQ-RCO](https://huggingface.co/pfeifferj/GLM-5.3-Flash-GSQ-RCO-GGUF)

### Inferences
**What we lack**
1. **Per-expert GPTQ/error-feedback rounding.** A per-expert imatrix is likely already in our llama-imatrix flow. GPTQ-style column-wise error feedback on top of it is the new part, worth about 25% lower expert output error at equal size according to the author.
2. **Domain-weighted calibration aimed at the deployment workload,** here tool calls and front-end code.
3. **Measuring token-level agreement with FP8, not only MC accuracy.** Maya's own data shows MC accuracy cannot tell S from M (both 97.9%), while top-1 agreement and KL separate them.

**Transfer to V4.1-Flash-1017**
- V4.1 experts are already MXFP4, about 4.25 bpw, and our envelope needs about 2.9 bpw (project note `thor-base-model-shortlist.md`). So we must requantize experts below the source precision.
- That is where Maya's GPTQ error feedback applies directly: round each expert against its own input Hessian, computed from the MXFP4-dequantized teacher.
- The "statistics from the source model, quantize once from the source" principle is independently confirmed by lte-ai (see Q3): requantizing from a 2-bit intermediate nearly doubled KL.
- Nothing in Maya bears on expert *selection*.

### Gaps
- Maya's "about a quarter lower output error" claim is unverified, and no ablation is published in the repo. The code lives in the `tools/maya_quant` directory of [project-maya](https://github.com/mw00/project-maya), which I did not inspect.
- No perplexity numbers are given.
- The zero-shot evals use 400 questions per task, which the author admits is within noise.

## Q3. Which community GLM-5.3-Flash and DeepSeek-V4.x-Flash(-1017) compressions exist, and what did they measure?

### Takeaway
- **As of 2026-10-19 there is no public expert-pruned V4.1-Flash-1017** (released 2026-10-17). Every 1017 repo found is a quantization.
- **V4.1-Flash (pre-1017) REAP builds** exist only as MLX artifacts:
  - pipenetwork REAP50 (192/384) and REAP25;
  - rapid-mlx (336/384).
  The only quality number among them is a wikitext PPL ratio of x1.168 for the generic-calibration REAP50.
- **GLM-5.3-Flash** has the richest evidence:
  - lte-ai independently measured the user's own `patrickbdevaney/GLM-5.3-Flash-REAP50-GGUF` mask (144/288) at KL 0.393 and top-1 agreement 80.4% against FP8, with HumanEval 154/164.
  - lapadatbogdan showed that a code-calibrated REAP-sum mask far outperforms generic and frequency masks on agentic tasks.
- **V4-Flash-0731:** puwaer's REAP 132/256 matched or beat base on math/code, and REAM merging was much worse.

### Cited Findings
**GLM-5.3-Flash pruned builds**

*lapadatbogdan/GLM-5.3-Flash-REAP144-Coder-GGUF (2026-10-13; 144/288 experts kept)*
- **Calibration:** REAP-sum `sum(router weight x ||expert output||)` over agentic Python coding sessions written by the full model. The model goes from 321B-A18B to 161B-A17B. — [card](https://huggingface.co/lapadatbogdan/GLM-5.3-Flash-REAP144-Coder-GGUF)
- **Sizes:**

  | Quant | Size (GB) |
  |---|---:|
  | Q4_K_M | 97.55 |
  | Q3_K | 71.42 |
  | IQ2_S | 54.77 |
  | IQ2_XXS | 45.26 |
  | MTP block | 2.59 |

  — [card](https://huggingface.co/lapadatbogdan/GLM-5.3-Flash-REAP144-Coder-GGUF)
- **Held-out KL to full, at 144 experts:**

  | Mask | KL |
  |---|---:|
  | REAP-sum | 0.73 |
  | REAP-mean | 0.85 |
  | Frequency | 0.94 |

  At 96 experts the same three are 1.07, 1.26 and 1.47. At 120 experts, REAP-sum is 0.88. — [GitHub repo](https://github.com/lapadatbogdan/GLM-5.3-Flash-REAP144-Coder)
- **Agentic results** on 48 attempts (30 Python repo tasks, 200 turns):
  - REAP144: 48/48 at Q4_K_M and Q3_K, 45/48 at IQ2_S, 39/48 at IQ2_XXS.
  - Turns relative to the full model: x1.58, x1.80, x1.83 and x2.77 respectively.
  - Code-scored mask at Q8: x1.21 turns. Frequency mask at 144: x2.21. Public 7-domain REAP 50% mask: x2.29.
  - "Which experts you keep matters more than how many."
  - — [card](https://huggingface.co/lapadatbogdan/GLM-5.3-Flash-REAP144-Coder-GGUF)
- **Perplexity on code:**

  | Model | PPL | vs full |
  |---|---:|---:|
  | Full Q8 | 1.681 | |
  | REAP144 Q8 | 2.053 | +22.1% |
  | REAP144 IQ2_XXS | 2.303 | +37.0% |

  Quantization KLD relative to the pruned Q8: 0.051 (Q4_K_M), 0.084 (Q3_K), 0.147 (IQ2_S), 0.208 (IQ2_XXS). — [card](https://huggingface.co/lapadatbogdan/GLM-5.3-Flash-REAP144-Coder-GGUF)
- **Known issue (15 Oct):** calibrating only on Python broke general reasoning at every quant, including wrong 17x23 arithmetic and loops on other programming languages. — [card](https://huggingface.co/lapadatbogdan/GLM-5.3-Flash-REAP144-Coder-GGUF)
- **What failed:**
  - RCO search over experts reached KL 1.52 at 144 experts, worse than the frequency mask's 0.94. The author's explanation: it "scored masks without re-routing removed experts; 12,096 decisions too noisy".
  - LoRA healing by top-20-logit distillation from the full model diverged after about 16 steps, and diverged immediately when router and norms were trainable. It gave no gain at 2 bit.
  - Pruning and 2-bit quantization compound: the public REAP-50 mask at 2 bit solved 0/4 tasks.
  - GSQ instead of imatrix at about 3 bit made no difference (KLD 0.0711 vs 0.0702).
  - — [GitHub repo](https://github.com/lapadatbogdan/GLM-5.3-Flash-REAP144-Coder)

*lte-ai/GLM-5.3-Flash-Strata-Q2Q3-Experts (Strata fork; REAP-50 base is patrickbdevaney/GLM-5.3-Flash-REAP50-GGUF)*
- **Results** against exact FP8 logits on 16 held-out sequences (8,192 tokens). FP8 PPL is 4.449.

  | Build | Size (GB) | PPL | KL | Top-1 | HumanEval |
  |---|---:|---:|---:|---:|---:|
  | Full Q23, experts quantized from FP8 | 119.9 | 4.606 (+3.5%) | 0.117 | 88.3% | 155/164 |
  | Same, first release requantized from Unsloth 2-bit | | 4.850 | 0.212 | 84.7% | |
  | Unsloth UD-Q2_K_XL | 108.7 | 4.688 | 0.144 | 87.5% | |
  | **REAP-50 (144/288) Q23 from FP8** | 63.2 | 5.184 (+16.5%) | 0.393 | 80.4% | 154/164 |
  | REAP-50 Q23, first release (from Q4_K_M experts) | | 5.632 (+26.6%) | 0.479 | 77.6% | 149/164 |

  — [card](https://huggingface.co/lte-ai/GLM-5.3-Flash-Strata-Q2Q3-Experts)
- **REAP-224 (224/288, 95.5 GB):** HumanEval 152 then 150/164; PPL +4.9% and KL 0.064 relative to unpruned Q23. Their release floor is ppl within +6%, KL within +0.08 and top-1 within 3 points. — [card](https://huggingface.co/lte-ai/GLM-5.3-Flash-Strata-Q2Q3-Experts)
- They recovered the REAP-50 expert map from the routers (`reap50-expert-map.txt`). — [card](https://huggingface.co/lte-ai/GLM-5.3-Flash-Strata-Q2Q3-Experts)

*Other GLM-5.3-Flash repos*
- These are quantization or abliteration releases only, not pruning: unsloth, nvidia/RedHat/RadixArk/LibertAIDAI NVFP4, amd Quark-MXFP4, turboderp exl3, bartowski BF16, ggml-org, Mia-AiLab EXL3, and others. — [HF search](https://huggingface.co/models?search=GLM-5.3-Flash)

**DeepSeek-V4.1-Flash (pre-1017) pruned builds**
- **pipenetwork/DeepSeek-V4.1-Flash-REAP50-MLX-mixed-4_8bit (192/384, all 40 layers):**
  - Size: 274.66 GB, of which 2x55.3 GB is 8-bit Engram. REAP25 is 351.14 GB.
  - Saliency: mean applied routing weight x ||expert output|| over 65,536 tokens of wikitext-2.
  - Wikitext-2 PPL 3.3833. Paired against unpruned: **x1.1681 [1.1407, 1.1991], worse on 138/140 windows**.
  - Its finding on Engram precision: 6-bit is indistinguishable from FP8, while 4-bit costs +7.3% in free-running divergence.
  - — [pipenetwork REAP50](https://huggingface.co/pipenetwork/DeepSeek-V4.1-Flash-REAP50-MLX-mixed-4_8bit)
- **rapid-mlx/DeepSeek-V4.1-Flash-REAP-2bit-MLX (336/384, 212.94 GB):**
  - "Conservative routing-saliency pruning" from an affine 2-bit/g64 source, retaining 99.9949% of routing mass on average (worst layer 99.8564%).
  - Calibration: 2,048 tokens. The keep sets from two split halves overlap 92.1%.
  - The only quality evidence is a smoke test ("Paris"). It decodes at 7.31 tok/s on an M3 Ultra.
  - — [rapid-mlx](https://huggingface.co/rapid-mlx/DeepSeek-V4.1-Flash-REAP-2bit-MLX)

**DeepSeek-V4.1-Flash-1017 repos (all quantization, none pruned)**

| Repo | Size (GB) | Contents | Quality numbers |
|---|---:|---|---|
| [LibertAIDAI EXL3-2.9bpw](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw) | 210.82 (196.3 GiB) | Engram not included. Experts 3 bpw (layers 18-22 at 2 bpw); attention 5 bpw | PPL +1.8% code (1.9094 vs 1.8749), +0.4% prose. 3.5 bpw: 1.8895 / 6.4341. Engram off: 4.16 code (sanity check) |
| [elvezjp ELVEZ-Q2](https://huggingface.co/elvezjp/DeepSeek-V4.1-Flash-1017-ELVEZ-Q2-gguf) | 365.71 | IQ2_XXS gate/up, Q2_K down, 8,192-token imatrix; Q8_0 attention/shared; FP8 Engram packed in (188.8 GiB); about 151.8 GiB resident; IXV-engine only. DSpark head 7.94 GB | |
| [mrfakename GGUF](https://huggingface.co/mrfakename/DeepSeek-V4.1-Flash-1017-GGUF) | Q2_K 264.52; Q3_K_M 347.27; Q4_K_M 444.74; Q8_0 507.95 | | none on card |
| [Lucebox ROCMFP](https://huggingface.co/Lucebox/DeepSeek-V4.1-Flash-1017-ROCMFP-GGUF) | 382.7-405.4 | ROCMFP2S/23S | |
| [kyuz0](https://huggingface.co/kyuz0/deepseek-v4.1-flash-gguf) | 1017-Q2 365.71 | | |
| [mlx-community oQ2.5e-mtp](https://huggingface.co/mlx-community/DeepSeek-V4.1-Flash-1017-oQ2.5e-mtp) | 342.72 | | |
| [gnahzhang LSQ3-G128](https://huggingface.co/gnahzhang/DeepSeek-V4.1-Flash-1017-LSQ3-G128-MLX) | 451.55 | includes 203.07 Engram | |

- elvezjp notes: "earlier V4.1 Flash GGUFs cannot be reused for 1017 — language **and vision** weights changed". — [elvezjp](https://huggingface.co/elvezjp/DeepSeek-V4.1-Flash-1017-ELVEZ-Q2-gguf)
- LibertAIDAI notes the 1017 Engram tables are new. — [LibertAIDAI](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw)
- **apetersson MixedQ2** (pre-1017 V4.1; 168.93 GB, no Engram):
  - IQ2_XXS gate/up plus Q2_K down, 2.2500 bpw over 543.58B routed weights.
  - Not imatrix-calibrated.
  - Minimum cosine against the MXFP4 source: IQ2_XXS 0.9407, Q2_K 0.9633.
  - Not runnable end to end.
  - V4.1 uses 32x32 E8M0 block scales for the FP8 dense weights, against 128x128 in V4.
  - — [apetersson](https://huggingface.co/apetersson/DeepSeek-V4.1-Flash-MixedQ2-GGUF)
- **pokey2025 "ik-llama"** contains only a DSpark MXFP4 draft (7.91 GB) and an mmproj; there are no trunk weights. — [file listing](https://huggingface.co/pokey2025/DeepSeek-V4.1-Flash-ik-llama-GGUF)

**DeepSeek-V4-Flash-0731 (256 experts) pruned builds**
- **puwaer reap-150b (132/256 kept, about 48% pruned):**
  - 84.70 GB safetensors. GGUFs: MXFP4_MOE 85.05, Q3_K_M 70.94, IQ3_XXS 67.00, Q2_K 62.39 GB. Tool: [moe-compress](https://github.com/puwaer/moe-expert-compress).
  - Calibration: 3,072 samples x 512 tokens, 30% math and 70% code. No gradient steps.
  - MTP modules dropped.
  - Benchmarks, base / REAP-150b / REAM-150b:

    | Benchmark | Base | REAP-150b | REAM-150b |
    |---|---:|---:|---:|
    | GSM8K | 0.9484 | 0.9295 | 0.6922 |
    | MATH-500 | 0.7060 | 0.7140 | 0.5020 |
    | HE+ | 0.8720 | 0.8963 | 0.8537 |
    | MBPP+ | 0.7407 | 0.7593 | 0.7328 |
    | Mean | 0.8168 | 0.8248 | 0.6952 |

    REAP-200b (178/256) has a mean of 0.8102.
  - MoE-block reconstruction cosine averages 0.9445, with a minimum of 0.8417.
  - — [puwaer reap-150b](https://huggingface.co/puwaer/DeepSeek-V4-Flash-0731-reap-150b), [GGUF](https://huggingface.co/puwaer/DeepSeek-V4-Flash-0731-reap-150b-gguf)
- **Tinker-Stack REAP-150B TQ3_4S (65.51 GB, 3.49 bpw, needs the turbo-tan fork):**
  - Wikitext-2 PPL about 13.34, against an IQ3_XXS baseline of 13.3073 ± 0.23. IQ2_S measured 14.30.
  - Its thinking-mode scores are not comparable to the thinking-off baseline; the card discloses this itself.
  - — [Tinker-Stack](https://huggingface.co/Tinker-Stack/DeepSeek-V4-Flash-0731-REAP-150b-TQ3_4S-GGUF)
- I saw puwaer reap-200b-gguf (IQ1_M at 60.52 GB up to MXFP4_MOE at 111.51 GB) and KeinNiemand V4-Flash-0731-IK_GGUF in the listing but did not read them. — [HF search](https://huggingface.co/models?search=V4-Flash)

### Inferences
- **Our GLM-5.3 REAP-50 mask now has an independent teacher-agreement score:** KL 0.393 and top-1 80.4% against FP8, after Q2/Q3 quantization from FP8.
  - Unpruned Q23 scores KL 0.117. So at 50% the pruning, not the quantization, is the dominant error. That is consistent with our "pruning 14x more damaging than quant" note.
  - HumanEval barely moves (154 vs 155). Use KL and top-1 as the sensitive gate metrics.
- **The pipenetwork x1.168 PPL ratio** comes from wikitext-calibrated REAP-mean on V4.1 at 192/384. It is the only V4.1 50% baseline published, and it is weak:
  - REAP-mean is worse than REAP-sum on GLM (KL 0.85 vs 0.73).
  - Wikitext calibration is the worst domain choice.
  
  A HOPE or REAP-sum mask on a capability-aligned corpus should beat it. No public artifact makes that comparison.
- **For V4-family models specifically,** puwaer's data says pruning to about 50% with math/code calibration keeps math/code benchmarks, and merging (REAM) loses badly. Treat merging as not worth pursuing for V4.1.
- **Two failure modes to design against** (lapadatbogdan):
  - single-domain calibration destroys general reasoning;
  - 50% pruning combined with 2-bit compounds into loops.

  Our planned 50% plus about 2.9 bpw sits at the edge of the second.

### Gaps
- Searches for ubergarm, unsloth or bartowski V4.1-Flash(-1017) quants and for ik_llama trellis V4.1 quants turned up no trunk releases, so they may not exist yet. These were only HF name searches limited to 200 results, ranked by downloads.
- There is no community KL or top-1 measurement against the teacher for any V4.1 pruned model.
- I did not read the discussions tabs of any repo.

## Q4. Are there new teacher-agreement objectives (KL-to-teacher mask search, output-aware saliency, routing distillation) that rank masks better than saliency mass?

### Takeaway
I found no published Jul-Oct 2026 method that searches masks by KL to the teacher and beats HOPE or REAP on a large fine-grained MoE. The only real attempt on our model family, RCO expert search on GLM-5.3-Flash, did *worse* than a frequency mask (KL 1.52 vs 0.94). It failed because it scored masks without re-routing the removed experts' probability mass. The usable objective is to rank a few candidate masks by **held-out top-k-renormalized teacher KL**. The closest published "teacher-aware" options are HOPE, which is output-aware through pairwise co-activation, and MoRA, which learns router biases with LM loss.

### Cited Findings
- **RCO (an IST-DASLab method) used as an expert-mask search** on GLM-5.3-Flash at 144/288 reached held-out KL 1.52, against 0.94 for the frequency mask and 0.73 for REAP-sum. The diagnosis: it "scored masks without re-routing removed experts; 12,096 decisions too noisy". — [lapadatbogdan GitHub](https://github.com/lapadatbogdan/GLM-5.3-Flash-REAP144-Coder)
- **The same author ranked masks by held-out KL** (REAP-sum 0.73 < REAP-mean 0.85 < frequency 0.94), and that ranking matched the agentic-task ranking (x1.21 vs x2.21 turns). — [card](https://huggingface.co/lapadatbogdan/GLM-5.3-Flash-REAP144-Coder-GGUF), [GitHub](https://github.com/lapadatbogdan/GLM-5.3-Flash-REAP144-Coder)
- **lte-ai uses a corpus gate** of PPL within +6%, KL within +0.08 and top-1 within 3 points. It accepted REAP-224, yet REAP-224 still failed HumanEval in a faster configuration. A KL gate needs a task check alongside it. — [lte-ai card](https://huggingface.co/lte-ai/GLM-5.3-Flash-Strata-Q2Q3-Experts)
- **Deterministic Differentiable Structured Pruning (2603.08065)** optimizes masks only, with the dense model as a fixed KL teacher, and has an MoE extension that prunes experts. I did not see a REAP comparison. — [arXiv 2603.08065](https://arxiv.org/pdf/2603.08065)
- **MoRA** trains router biases on LM loss, which makes them output-aware, and picks the lowest-bias experts. It beats count/gate-mass/MoNE/HC-SMoE baselines at 50% but was not compared with REAP. — [arXiv 2610.00367](https://arxiv.org/html/2610.00367)
- **Healing and distillation evidence:**
  - LoRA top-20 logit distillation on GLM REAP144 diverged within about 16 steps. — [lapadatbogdan GitHub](https://github.com/lapadatbogdan/GLM-5.3-Flash-REAP144-Coder)
  - Self-distillation with frozen routed experts did help at 25% keep on a 35B model (HE+ 0.787 to 0.841). — [arXiv 2607.16721](https://arxiv.org/html/2607.16721)
  - SlimQwen finds that progressive pruning beats one-shot pruning at equal tokens, at pre-training scale. — [ICML 2026](https://icml.cc/virtual/2026/75227)

### Inferences and recommendations for the V4.1-Flash-1017 REAP on Thor
**Adopt**
1. **HOPE pᵀFp selection with the REAP-sum diagonal.** Use `sum`, not `mean`; the GLM evidence gives 0.73 vs 0.85 KL. Renormalize top-k router weights as upstream REAP does.
2. **A capability-aligned, multi-domain calibration mix,** never single-domain.
3. **A final arbiter.** Build three to five candidate masks: HOPE, REAP-sum, routing-mass, and HOPE with per-layer counts nudged. Rank them by **held-out teacher KL and top-1 with proper top-k re-routing over survivors**. On Thor this means streamed layer-by-layer logits from the MXFP4 teacher. Confirm the winner on a small task battery.
4. **Per-expert GPTQ error-feedback rounding when requantizing the MXFP4 experts to about 2.9 bpw,** as in Maya. Quantize once from the source, as both Maya and lte-ai show.

**Gate before adopting**
- MoRA's affine expert proxies, because they are cheap and calibration-only. Its learned router-bias re-calibration may also suit noaux_tc.

**Do not adopt**
- Merging (REAM/HC-SMoE). puwaer's V4-0731 result is 0.6952 vs 0.8248.
- Unconstrained KL mask search with combinatorial or Riemannian optimizers (RCO-style).
- LoRA logit-distillation healing at this scale on Thor.

**Gate thresholds:** use lte-ai's numbers (KL +0.08, top-1 -3 points, relative to the unpruned quant at the same bpw) plus a task check.

### Gaps
- I found no paper giving a learned or searched mask that beats HOPE on a 256-384 expert model.
- There is no published comparison of KL-ranked versus saliency-ranked masks beyond lapadatbogdan's GLM-5.3-Flash numbers, which come from one author on one domain (Python agentic).
- Whether HOPE's off-diagonal gain holds with V4.1 routing and hyper-connections is untested.
