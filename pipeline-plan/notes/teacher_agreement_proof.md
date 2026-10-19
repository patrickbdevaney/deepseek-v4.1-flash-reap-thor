# Proving "best teacher agreement among unified-RAM-viable compressions" for DeepSeek-V4.1-Flash-1017

Scope: a measurement design that turns "our pruned+quantized V4.1-Flash-1017 is the unified-RAM-viable model that agrees best with the unpruned teacher" into a claim with stated error bars, when the teacher (476 GiB checkpoint) does not fit Thor's ~117 GiB. Written 2026-10-19. This note builds on, and does not repeat, the sibling protocol note `Compressing 1T MoE for Thor/eval_protocol_pruned_moe.md` (loop metrics, BFCL/RULER/PopQA sizing, tiers, Thor wall-clock). Every number that is not from a cited source is marked **[arithmetic]** or **[extrapolation]**. Nothing was run on the GPU for this note. The only local reads were existing JSON artefacts.

## 1. Teacher reference outputs without fitting the teacher

### Takeaway
There are three viable teacher sources, and they are complementary rather than alternatives. (A) A layer-streamed teacher-forced forward on Thor itself is the only exact-precision source that also yields routing and hidden states. The GLM-5.3 pipeline already did this for 345k tokens, and it extrapolates to about 1 h per 2M tokens on V4.1, but it needs the full 476 GiB checkpoint on disk. (B) A rented 2×B200 or 4×H200 vLLM node gives prompt logprobs for millions of tokens for tens of dollars. (C) The official DeepSeek API serves exactly `DeepSeek-V4.1-Flash-1017` and returns top-20 logprobs, but only on its own generated tokens and with no prompt logprobs. Use C for cheap free-running teacher trajectories and A or B for teacher-forced scoring. Cross-validate C against A or B before trusting it.

### Cited Findings
- **Checkpoint and footprint.** The 1017 checkpoint is "about 511 GB on disk (476 GiB)": routed experts ≈259.5 GiB, Engram tables ≈183.1 GiB, plus smaller parts. The vLLM recipe's minimum VRAM is 614 GB including 1.2× headroom. It serves on H200 at TP4 with Engram CPU offload, on B200/B300 at TP2 with Engram offload, and on one DGX Station GB300 at TP1 with Engram and 60 GiB of experts in host memory — [vLLM recipe](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash)
- **Local storage gate.** The full checkpoint (backbone 286 GiB + Engram 189 GiB) must be on local disk for any faithful forward, and this Thor's NVMe had 51 GB free when the note was written — [LOCAL v41_architecture_pruning_delta.md](file:///home/patrickd/glm-5.3-reap/research_notes/Compressing%201T%20MoE%20for%20Thor/v41_architecture_pruning_delta.md)
- **NVMe bandwidth (measured locally).** Direct-I/O sequential read is 3.9 GB/s on the WD SN5000S — [LOCAL layer_streamed_healing_infra.md](file:///home/patrickd/glm-5.3-reap/research_notes/Compressing%201T%20MoE%20for%20Thor/layer_streamed_healing_infra.md)
- **Streamed-teacher cost estimate (local extrapolation).** 40 layers × 7.4 GB = 296 GB of NVMe per pass, which is 76 s per pass. At 64k tokens per pass, 2M tokens is 32 passes, about 40 min of NVMe plus 16–40 min of compute, so "~1 h per 2M tokens of cached teacher logits". The Engram random reads were not measured in this estimate — [LOCAL layer_streamed_healing_infra.md §KQ4](file:///home/patrickd/glm-5.3-reap/research_notes/Compressing%201T%20MoE%20for%20Thor/layer_streamed_healing_infra.md)
- **Engram row reads (measured locally).** 48 rows × 264 B = 12.4 KiB per token. 48 concurrent O_DIRECT row reads take 1.1–1.3 ms p50 — [LOCAL decode_bytes_per_step.md](file:///home/patrickd/glm-5.3-reap/research_notes/Compressing%201T%20MoE%20for%20Thor/decode_bytes_per_step.md)
- **Precedent: the GLM-5.3 pipeline already streamed its teacher.** `s09_eval` scores teacher and student "in SEPARATE passes, not lockstep", keeps about 140 B per token (gold NLL, argmax, top-32 index and logprob), so "a 200k-token eval is ~28 MB". Generation from the streamed teacher is "impossible here: 328 GB / 3.0 GB/s is ~110 s per decoded token". The teacher is banked (`teacher_from_cache`), and `mask_eval.json` records `teacher_tokens: 345320` — [LOCAL scripts/stages/s09_eval.py](file:///home/patrickd/glm-5.3-reap/scripts/stages/s09_eval.py), [LOCAL artifacts/mask_eval.json](file:///home/patrickd/glm-5.3-reap/artifacts/mask_eval.json)
- **Routing capture precedent.** Teacher top-k routing per token per layer is tiny: 42 layers × 1M tokens × top-8 int16 is 0.66 GiB. Once captured, "affected-rate-versus-position is computable OFFLINE against any candidate mask … with no GPU and no teacher". The capture must run before surgery deletes the source shards. The host footprint of sweeping one layer measured ~53 GiB against a 5.76 GiB weight size, so the capture runs one layer per process — [LOCAL scripts/capture_teacher_routing.py](file:///home/patrickd/glm-5.3-reap/scripts/capture_teacher_routing.py)
- **V4.1 long-context memory hazard.** The dense indexer score tensor is `[rows, 32, S, S]`, which is 4 GiB bf16 per row at 8k. The candidate stage is a no-op below 16k, so a streamed teacher at ≤8k does not exercise long-context attention selection — [LOCAL v41_architecture_pruning_delta.md](file:///home/patrickd/glm-5.3-reap/research_notes/Compressing%201T%20MoE%20for%20Thor/v41_architecture_pruning_delta.md)
- **Official API.** `deepseek-flash` maps to DeepSeek-V4.1-Flash-1017, and the retired `deepseek-v4-flash` names are now served by 1017. Pricing per 1M tokens, peak/off-peak: input cache-miss $0.30/$0.15, output $1.20/$0.60. Context is 1M, and thinking mode is the default — [DeepSeek pricing](https://api-docs.deepseek.com/quick_start/pricing)
- **API logprobs.** `logprobs` (bool) and `top_logprobs` (0–20) apply to output tokens. The docs have no echo or prompt-logprob parameter. The response schema also carries logprobs for `reasoning_content`. In thinking mode, `temperature` and `top_p` have no effect — [DeepSeek chat-completion API](https://api-docs.deepseek.com/api/create-chat-completion)
- **API logprobs are noisy and can drift.** "Logprobs are not deterministic in practice" because of batch effects and routing across GPUs. A single-token logprob test detected changes "as small as one step of fine-tuning". Monitoring 189 endpoints over 4 months flagged 37 suspected changes, 34 of them on open-weight models. Most providers return top-20 logprobs, Fireworks/Azure 5, xAI 8 — [Log Probability Tracking of LLM APIs, arXiv 2512.03816](https://arxiv.org/html/2512.03816v1)
- **Hosted open-weight endpoints often differ from the reference weights.** 11 of 31 Llama endpoints "serve different distributions than reference weights". An MMD test with a string kernel had a median power of 77.4% at about 10 samples per prompt — [Model Equality Testing, ICLR 2025, arXiv 2410.20247](https://arxiv.org/html/2410.20247v2)
- **Cloud GPU prices (Oct 2026).** Median on-demand prices are H200 $5.70 and B200 $8.19 per GPU-hour — [Uvik GPU price index](https://uvik.net/blog/gpu-price-index/). Nebius from 1 Oct 2026: H200 $5.40, B200 $8.50 — [usagepricing.com](https://usagepricing.com/blueprint/activity/nebius-ai-2026-09-24-gpu-price-increase). B200 at neo-clouds runs from about $3.70 — [Spheron](https://www.spheron.network/blog/nvidia-b200-cloud-pricing-2026/)
- **Throughput proxy (V4-Flash, not V4.1).** A B200 serving benchmark at 8,192 input, 1,024 output and 32 concurrent requests gave 1,222 output tok/s with 1.70 s mean TTFT. The page mixes in SGLang/EAGLE rows, so the stack for each row is unclear — [Lambda V4-Flash](https://lambda.ai/inference-models/deepseek-ai/deepseek-v4-flash). 2× DGX Spark vLLM: about 1,574 tok/s prefill at pp2048 c1 — [NVIDIA forum](https://forums.developer.nvidia.com/t/deepseek-v4-flash-aiden-recipe-from-reddit-1m-token-session-operational-cuda-12-1-tailored-for-dgx-spark-gb10/372268/19)
- **vLLM prompt logprobs.** They are returned only when `num_prompt_logprobs` is set, and the first prompt token's logprob is None — [vLLM logprobs source docs](https://docs.vllm.ai/en/stable/api/vllm/v1/engine/logprobs/)

### Inferences
- **Source A: Thor layer-streamed, teacher-forced. This is the gold reference.**
  - Cost: about 1 h per 2M tokens **[extrapolation from the local estimate]**. A 2–5M-token reference set therefore costs one evening.
  - Output: full-vocabulary logits (so top-k for any k, the exact gold NLL, and the exact argmax), top-6 routing per layer, and hidden-state taps. All are in the checkpoint's own MXFP4/FP8 numerics, through `inference/model.py`, the same code path that defines the model.
  - Storage per token **[arithmetic]**: top-32 logits ≈140 B; routing 40 layers × 6 × int16 = 480 B. 5M tokens is therefore ≈0.7 GB of logits plus 2.4 GB of routing.
  - Blockers: a second drive for the 476 GiB checkpoint, and an exclusive Thor window. Never run it beside a vLLM server (memory note thor-unified-memory-oom). Contexts are limited to about 8k by the indexer tensor unless the indexer is query-chunked.
  - Must run **before** any surgery deletes source shards, as `capture_teacher_routing.py` warns.
- **Source B: rented node, teacher-forced via vLLM `prompt_logprobs`. The scale-out and long-context source.**
  - A 2×B200 TP2 node is about $16–17/h, and 4×H200 TP4 about $22–23/h **[arithmetic from the median prices]**.
  - If the steady-state input throughput implied by Lambda's numbers (1,222 out/s ÷ 1,024 × 8,192 ≈ 9.8k input tok/s, mixed with decode) carries over, 10M teacher-forced tokens take under an hour of GPU time. Budget about 2–3 h including the 511 GB download and setup, which is **≈$50–100 per 10M tokens [extrapolation]**.
  - Use it for 32k–128k-context references, which Thor cannot stream, and for long free-running teacher generations under pinned sampling.
  - Pin the following: no speculative decoding, fp8 KV as in the recipe, `--language-model-only`, raw (not processed) logprobs, a fixed max-logprobs k (≥20; 32 to match `s09`), and the exact 1017 revision hash.
  - Validate B against A on a 50k-token overlap. Their KL should sit at the runtime-noise floor, not the compression-damage level.
- **Source C: DeepSeek API. A cheap teacher for free-running behaviour only.**
  - It cannot score held-out text, because there are no prompt logprobs. It can produce teacher continuations with top-20 logprobs per generated token.
  - The student is then teacher-forced on those continuations. That yields KL/agreement on the **teacher's own trajectory**, which is the distribution that matters for "does the student say what the teacher would say", and yields spec-decode-style acceptance length for free.
  - Cost: 2M generated tokens is about $1.2 off-peak (output $0.60/1M) plus prompt cost **[arithmetic]**. Cost is effectively zero.
  - Caveats:
    - Serving precision and kernels are unknown; KV in particular may be FP4/FP8.
    - Logprobs are non-deterministic under batching.
    - Thinking is on by default, so the reasoning content must be pinned or explicitly disabled in every call.
    - The endpoint can change silently.
  - Mitigation: run the Model-Equality-style check of C against A or B on a few hundred prompts **before** using C as the reference, and record the API response date with each cached item.
- **What the API cannot give: routing.** Routing agreement and affected-rate-versus-position need A, or B with router hooks.
- **Recommended combination [extrapolation]:**
  - A for 2–5M tokens of teacher-forced references plus routing at ≤8k context, run once and banked.
  - B for long context (32k/64k/128k), if long-context agreement is part of the claim.
  - C for 1–2k teacher free-running trajectories across the prompt families, cross-checked against B on a sample.

### Gaps
- No measured Thor wall-clock exists for a V4.1 streamed forward. The 1 h/2M-token figure is an estimate that ignores NVMe/LPDDR contention and Engram gather overhead.
- No published prefill-only tokens/s exists for V4.1-Flash on B200/H200 vLLM. The 9.8k tok/s is derived from an end-to-end V4-Flash serving run.
- The DeepSeek API's serving precision, KV dtype and whether it applies MTP/speculation are undocumented. Whether top_logprobs are reported for reasoning tokens in practice, as opposed to only in the schema, was not tested.
- The vLLM max-logprobs cap and logprobs-mode flags for the current vLLM version were not confirmed; the docs page fetched returned 404.

## 2. Which agreement metrics actually discriminate between candidate masks and quant mixes

### Takeaway
In the regime our candidates live in (top-1 agreement ≈0.84, top-32 KL ≈0.65 nats/token on GLM-5.3 REAP-50), per-token KL and top-1 agreement both rank compressions well against downstream quality (|ρ| ≈ 0.7–0.86). Both lose ranking power only in a near-lossless "silent zone" (KLD < 0.03–0.06) that a 50% prune is far outside. Our own paired McNemar tests already separated masks at z = 9–12. The "reconstruction improved 2.5% but agreement didn't move" result is best explained as a real null on output, diluted across tokens the change never touched. The metric that would have exposed a real effect is a paired, affected-token-stratified KL or flip test, not a different global metric.

### Cited Findings
- **KL and top-1 agreement versus benchmark score, full cohort.** KLD versus composite benchmark score: ρ = −0.72 (Qwen3.6-35B-A3B, 28 quants) and −0.86 (Devstral-24B, 41 quants). Top-1 agreement: +0.70 / +0.86.
- **Same metrics in the silent zone.** Below a cohort-specific τ (Qwen 0.064, Devstral 0.027) correlations vanish: KLD 0.00 / −0.24, top-1 agreement −0.04 / +0.25.
- **No variant rescues the silent zone.** None of 14 variants (mean/median/p90/p99 KLD, ΔlogPPL, PPL ratio, top-1 at 512 and 8k, KLD at 8k, response-only top-k 20/60 KLD) is significant there. KLD tracks the *volume* of disagreement (ρ +0.94 / +0.55), not its direction. The authors' recommendation: "fidelity metrics should filter damaged candidates, not rank near-baseline ones" — [Displacement Is Not Direction, arXiv 2606.19558](https://arxiv.org/html/2606.19558)
- **Flips track KL, and accuracy hides them.** Flips vs KL: Spearman 0.981 on MMLU. Compression with negligible accuracy change still produces many flips, and high-flip models score worse on MT-Bench — [Accuracy is Not All You Need, arXiv 2407.09141](https://arxiv.org/abs/2407.09141)
- **Behaviour shifts even when accuracy and perplexity hold.** Quantized variants "usually have a shift in behavior even when accuracy and perplexity are preserved". The paper proposes per-item "Correctness Agreement"; its full numbers were not retrieved — [The Illusion of Equivalency, arXiv 2607.08734](https://arxiv.org/abs/2607.08734)
- **llama.cpp `--kl-divergence` outputs.** It reports Mean KLD (Gaussian SE), "Same top p" (argmax agreement, binomial SE), Δp statistics and KLD percentiles. Llama-3-8B examples: q4_K_M KLD 0.028 / 91.9% same-top; q2_K 0.332 / 71.1%; bf16-vs-fp16 control 2.5e-5 / 99.74% with max 0.012 — [llama.cpp perplexity README](https://github.com/ggml-org/llama.cpp/blob/master/tools/perplexity/README.md)
- **turboderp's EXL3 cards report KL by bitrate.** Step-3.5-Flash: 2.00 bpw KL 0.653, 3.00 bpw 0.142, 4.00 bpw 0.053, on "10 rows of wikitext2" — [turboderp Step-3.5-Flash-exl3](https://huggingface.co/turboderp/Step-3.5-Flash-exl3/blob/main/README.md)
- **Top-k KL convention.** "the top-k always applies to the reference model, never the target model", because storing full reference logits is prohibitive — [Optimal Formats for Weight Quantisation, arXiv 2505.12988](https://arxiv.org/pdf/2505.12988)
- **ubergarm's ik_llama.cpp cards report wiki.test.raw PPL, not KLD.** Example: Kimi-K2-Thinking Q4_X 2.0818 ± 0.009 down to smol-IQ1_KT 3.5931 ± 0.019, plus one aider-polyglot score (smol-IQ3_KS 77.3%) — [ubergarm/Kimi-K2-Thinking-GGUF](https://huggingface.co/ubergarm/Kimi-K2-Thinking-GGUF). Community practice for multi-hundred-GB MoE GGUFs is therefore PPL-centric, and PPL is the weakest of these metrics.
- **Spec-decode acceptance as a fitness metric.** EvoESAP's teacher-forced acceptance score beat KL and NLL as a search fitness on OLMoE coding/math — [arXiv 2603.06003](https://arxiv.org/abs/2603.06003)
- **Local, measured: masks are separable with paired top-1.**
  - pass1 vs pass2 mask (healing fixed): Δtop1 = +0.545%, discordant 19,925 of 241,516, paired SE 0.000584, McNemar χ² 86.9, z = 9.3 — [LOCAL artifacts/eval/mask_paired_test.json](file:///home/patrickd/glm-5.3-reap/artifacts/eval/mask_paired_test.json)
  - Scalar vs per-expert healing: Δ = +0.556%, discordant 13,020, paired SE 0.000472 versus 0.000746 unpaired, z = 11.75 — [LOCAL artifacts/eval/healing_paired_test.json](file:///home/patrickd/glm-5.3-reap/artifacts/eval/healing_paired_test.json)
  - Global aggregates: REAP-50 FP8 pass1 top-32 KL 0.695 / top-1 0.8370; pass2 0.650 / 0.8425; NVFP4 pass2 0.672 / 0.8386 — [LOCAL artifacts/eval/eval_glm-5.3-flash-reap50-*.json](file:///home/patrickd/glm-5.3-reap/artifacts/eval/)
- **Local: domain imbalance is where damage concentrates.** In the `mean` mask arm, ballast top-1 is 0.509 with dNLL 1.74, versus code 0.941 / 0.070 — [LOCAL artifacts/mask_eval.json](file:///home/patrickd/glm-5.3-reap/artifacts/mask_eval.json)
- **Local: KL is heavy-tailed and a few tokens dominate.** dNLL p50 0.0024 versus mean 0.39 and p95 3.09 in the same file.
- **Pruning damage compounds in generation.** Pruning deviations "accumulate across time steps", while multiple-choice selection stays robust — [arXiv 2603.24652](https://arxiv.org/abs/2603.24652)
- **PPL is blind to degeneration.** PPL 21.69 vs 22.05 while CREP is 7.3% vs 2.23% — [arXiv 2608.26676](https://arxiv.org/abs/2608.26676) (via sibling note)

### Inferences
- **Our regime is the "lossy" one, where fidelity metrics do rank.** GLM REAP-50 sits at top-32 KL ≈0.65, roughly 10–25× the silent-zone τ of 0.027–0.064, and close to turboderp's 2.0-bpw (0.653) and llama.cpp's q2_K (0.33) bands.
  - So 2606.19558's ρ ≈ −0.72 to −0.86 is the applicable evidence for ranking V4.1 REAP candidates by KL.
  - The silent-zone caveat applies only when comparing near-identical candidates, such as two quant mixes on the same mask that differ by ≤0.03 KL. Those must be decided by capability tests or declared ties.
- **Primary metric: paired mean KL(teacher‖student) on the teacher's top-k support.**
  - Use k = 32, or k = 20 when the reference is the API.
  - Compute it per token, with document-cluster SEs, per domain.
  - Truncation bias is identical for every candidate measured against the same reference, so it does not bias the ranking. It does prevent direct comparison with llama.cpp full-vocab KLD numbers.
- **Secondary metrics:**
  - Top-1 agreement (McNemar paired), which is interpretable and was measured to separate masks at z ≈ 9–12.
  - Confident-flip rate: disagreements where the teacher's p_max ≥ 0.9. It isolates the tokens where a disagreement is almost certainly an error rather than a near-tie.
  - Tail statistics: p99 KL and the fraction of tokens with KL > 1 nat. Damage is concentrated (dNLL p50 0.002 vs mean 0.39), and 2606.19558's best full-cohort Qwen variant was p99 (ρ −0.75).
- **Why reconstruction +2.5% showed agreement −0.0001: the effect was diluted.**
  - A layer-local reconstruction gain only reaches the output on tokens routed through the changed experts or layers. If a fraction f of tokens is affected, the global effect is f × (effect on affected tokens), while the noise comes from all tokens.
  - **Detecting metric:** use the banked teacher routing to restrict the paired test to tokens whose teacher top-k touches a changed expert, or tokens downstream of a changed layer. For small f this raises power by roughly 1/f in effect size **[inference]**.
  - Then report a paired KL difference on the affected subset with a document bootstrap.
  - If that is still null, the reconstruction gain does not propagate. The proxy is the problem, not the metric, which matches CAMERA's finding that layer-output error "does not always align with downstream task performance" (sibling note).
- **Routing agreement is a diagnostic, not the claim metric.**
  - Use it for two things. First, the share of teacher top-6 routed slots that land on pruned experts (the "affected rate"), as a function of position and domain. Second, after pruning, the Jaccard of the student's re-selected top-6 against the teacher's among surviving experts.
  - It is computable offline from cached routing (sibling `capture_teacher_routing.py`).
  - It explains *where* KL comes from, but the claim must rest on output agreement.
- **Free-running agreement: the user-visible metric.** For each teacher trajectory (source C or B), teacher-force the student and record:
  - (i) per-token KL/top-1 on the teacher's own path;
  - (ii) expected greedy accept length, i.e. the run length until the first top-1 disagreement (ESAP-style);
  - (iii) under greedy decoding of both models, the first-divergence position and post-divergence task outcome.
  Metric (ii) has an operational meaning ("how many tokens the student would reproduce the teacher") and is the EvoESAP-favoured fitness.
- **Do not use PPL as a ranking metric.** It is the community default on ubergarm cards, so report it only for comparability.

### Gaps
- No paper measures KL/flip sensitivity specifically for expert-pruning masks. 2606.19558 is quantization-only, on dense 24B and MoE 35B models.
- The affected-token stratification is our own design; no published precedent was found.
- The per-token paired KL variance between two of our masks was not computed for this note. The capture arrays were not located by a quick search, and nothing heavy was run.

## 3. Sample sizes and standard errors

### Takeaway
Using our measured paired discordance (5.4–8.3% of tokens between two masks), Miller's formula gives an MDE of ≈0.13–0.16% top-1 at 241k tokens. Resolving 0.1% needs 0.42–0.65M tokens, and 0.25% needs 68–104k, all before document-cluster inflation (×1.2–9.3 in tokens). Ranking K candidates with family-wise control costs another ×1.5 (K=5) to ×2.1 (K=20) in tokens. A 2–5M-token teacher reference, about one evening of source A, covers all of this for per-domain claims at about 0.25% resolution.

### Cited Findings
- **Paired power formula.** n = (z_{α/2}+z_β)²·ω²/δ² for deterministic scoring (Miller Eq. 9). Clustered/naive SE ratios on real evals ranged 1.10–3.05 — [Miller, Adding Error Bars to Evals, arXiv 2411.00640](https://arxiv.org/pdf/2411.00640)
- **Local paired SEs.** McNemar SE equals sqrt(discordance/n) to within 1e-6: 0.000584 (mask test, discordance 8.25%) and 0.000472 (healing test, 5.39%), versus unpaired 0.000746 — [LOCAL mask_paired_test.json, healing_paired_test.json](file:///home/patrickd/glm-5.3-reap/artifacts/eval/)
- **Earlier sibling estimate.** Unpaired per-token variance 0.136 gives 42.7k tokens for 0.5%, 267k for 0.2% and 1.07M for 0.1% — [LOCAL eval_protocol_pruned_moe.md §KQ5](file:///home/patrickd/glm-5.3-reap/research_notes/Compressing%201T%20MoE%20for%20Thor/eval_protocol_pruned_moe.md)
- **Noise sources.** Paired prediction noise typically exceeds paired data noise, and averaging repeated samples raises power — [arXiv 2512.21326](https://arxiv.org/pdf/2512.21326)
- **llama.cpp SEs.** Its KLD SE assumes per-token Gaussian KL, while the bf16 control's max/mean KL is about 500× — [llama.cpp README](https://github.com/ggml-org/llama.cpp/blob/master/tools/perplexity/README.md)

### Inferences
- **Top-1 MDE [arithmetic, 80% power, α=0.05 two-sided, ω² = measured discordance, no clustering]:**

  | Target δ | ω² = 0.0825 (different masks) | ω² = 0.054 (same mask, different heal/quant) |
  |---|---|---|
  | 0.5% | 26k tokens | 17k |
  | 0.25% | 104k | 68k |
  | 0.1% | 647k | 423k |
  | MDE at 241k | 0.164% | 0.132% |

  With cluster inflation c (SE ratio), tokens scale by c². At c = 1.9 (Miller's MGSM value), 0.1% at ω² = 0.08 needs about 2.3M tokens. **First action:** measure c on the existing 241k GLM run by document-clustered SE (Miller Eq. 4) before fixing the V4.1 budget.
- **Better than the earlier sibling estimate.** The measured *paired* variance is 0.083, not the unpaired 0.136, so paired designs need about 40% fewer tokens than that note assumed **[arithmetic]**.
- **Per-domain claims multiply the budget by the number of domains.** For 7 domains at δ = 0.25% and c ≈ 1.5: 104k × 2.25 ≈ 234k per domain, about 1.6M total. At 1 h per 2M tokens on source A, the teacher side is about 1 h **[arithmetic + extrapolation]**.
- **KL SE:**
  - Compute the paired per-token KL difference d_t = KL_A,t − KL_B,t and report a document-bootstrap CI (B = 2,000).
  - Do not trust the Gaussian ±. The heavy tail (dNLL mean/p50 ≈ 160×) makes per-token Gaussian SEs optimistic.
  - KL is continuous, so it should have more power per token than binary agreement for the same underlying shift. The exact ratio must be measured, not assumed (gap).
- **Free-running and capability evals.** The sizes in the sibling note stand: loop rate n ≈ 260 one-sample against a cached teacher for a doubling; BFCL multi_turn_base ≈ 100 paired; MMLU ≈ 1,600 log-prob items for 2 points; PopQA ≈ 1,000 for 3 points.

### Gaps
- The document-cluster inflation c for token metrics in our pipeline is still unmeasured.
- The paired KL-difference variance is unmeasured, so the KL-vs-top-1 power ratio is unknown.

## 4. Capability evals that complement agreement

### Takeaway
Agreement metrics filter damage and rank lossy candidates. Capability evals are needed for two reasons. First, the agreement winner must not be inferior to the runner-up on what users do. Second, near-tied candidates (inside the silent zone, or inside each other's KL CIs) can only be separated, or declared tied, by task outcomes. The small high-signal panel is: BFCL v3 multi_turn_base (the early-warning column for REAP), a generative factoid set (PopQA), log-prob-scored MMLU-Pro or MMLU, EvalPlus/LiveCodeBench for code, MATH-500, and RULER multi-key/multi-value at 32k (64k on the cloud teacher). The capability claim should be framed as **non-inferiority**, not superiority.

### Cited Findings
- **BFCL multi-turn is the column that moves first.** REAP Kimi-K2 multi_turn: 0.355 → 0.263 (25%) → 0.164 (50%), while Non-Live/Live/code stayed within about 2 points at 25% — [arXiv 2510.13999 Table 3](https://arxiv.org/abs/2510.13999). multi_turn_base has 200 items — [Gorilla BFCL v3 blog](https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v3_multi_turn.html)
- **MC scoring hides generative loss.** Wanda-50% open generation dropped 9.1 points against about 3 points on MC, a headroom-normalized ratio of about 4× — [Benchmark Illusion, arXiv 2606.17609](https://arxiv.org/abs/2606.17609)
- **Long-context quant damage appears only at 64K/128K,** measured with 100 examples per task and length under a GLMM with Bonferroni contrasts — [arXiv 2505.20276](https://arxiv.org/abs/2505.20276). RULER multi-key spreads models best; plain NIAH is saturated — [HELMET, arXiv 2410.02694](https://arxiv.org/abs/2410.02694)
- **Recommended benchmark mix.** 2606.19558 used HumanEval, LiveCodeBench v6, IFEval, BFCL v3, GSM8K, MMLU and MATH-500, and still insists downstream evaluation "remain[s] essential" — [arXiv 2606.19558](https://arxiv.org/html/2606.19558)
- **Local eval hygiene.**
  - Pin generation config and reasoning_effort.
  - Pin seqmax, since it re-routes greedy decoding.
  - A fixed shared max_tokens measures economy, not capability.
  - Assert served-model identity in every result.
  - Detach long evals.
  - Sources: memory notes [eval-pin-generation-config](/home/patrickd/.claude/projects/-home-patrickd/memory/eval-pin-generation-config.md), [eval-seqmax-perturbs-greedy-output](/home/patrickd/.claude/projects/-home-patrickd/memory/eval-seqmax-perturbs-greedy-output.md), [eval-fixed-budget-measures-economy](/home/patrickd/.claude/projects/-home-patrickd/memory/eval-fixed-budget-measures-economy.md), [eval-orphaned-server-trap](/home/patrickd/.claude/projects/-home-patrickd/memory/eval-orphaned-server-trap.md), [detach-unattended-local-workloads](/home/patrickd/.claude/projects/-home-patrickd/memory/detach-unattended-local-workloads.md)

### Inferences
- **Panel and sizes.** n = 7.84·ω²/δ² paired, with ω² the item discordance; budget ω² ≈ 0.08–0.12 until measured on the first pair.

  | Suite | Items | Paired MDE | Purpose |
  |---|---|---|---|
  | BFCL v3 multi_turn_base | 200 (all) | ≈ 7–9 pts **[arithmetic, ω² 0.10–0.15]** | Collapse detector |
  | PopQA free-form, 0-shot | 1,000 | ≈ 3 pts | Ballast/knowledge, generative |
  | MMLU-Pro by log-prob | 2,000 stratified | ≈ 1.8 pts (ω² 0.08) | Cheap, prefill-only |
  | EvalPlus (HumanEval+ + MBPP+) | ~560 | ≈ 4 pts | Code |
  | MATH-500 | 500 | ≈ 4 pts | Math |
  | RULER MK+MV @32k | 300 | ≈ 6–8 pts | Long context; 64k needs Thor-viable prefill or cloud |

- **Run the teacher on the panel once, on the cloud node (source B), and cache it.** That gives an absolute "fraction of teacher score retained" per suite, which is what public REAP cards report.
- **Frame the capability claim as non-inferiority.** Pre-register a margin per suite (for example −2 points MMLU-Pro, −3 points PopQA, −5 points BFCL multi-turn, matching the sibling note's gates). Require the lower 95% CI bound of (winner − competitor) to exceed −margin.
  - Superiority on capability is not needed for the agreement claim.
  - Superiority is also not achievable at these n for differences of a few points.

### Gaps
- No published paired discordance rates exist for these suites under expert pruning, so the MDEs above are extrapolated.
- The BFCL multi-turn token cost per item on Thor is unknown (±2×).

## 5. Paired designs and multiple-comparison control when ranking many candidates

### Takeaway
Rank all candidates on one identical frozen token set against one cached teacher, pair everything, and cluster by document. Then split the work into two stages:
1. A screening split, where any number of masks or quant mixes can be explored freely.
2. A pre-registered, untouched confirmation split, where the final K finalists are compared with simultaneous inference: Hsu-style multiple comparisons with the best via Bonferroni/Holm on K−1 one-sided paired tests, or a Model Confidence Set.

The honest output is a confidence set of best models. "Best" can be claimed outright only if that set contains a single model.

### Cited Findings
- **Model Confidence Set.** It is "a set of models that is constructed such that it will contain the [best] model with a given level of confidence". Uninformative data give a large set, and it "does not assume that a particular model is the true model" — Hansen, Lunde & Nason, Econometrica 2011 ([Econometrica](https://www.econometricsociety.org/publications/econometrica/2011/03/01/model-confidence-set); [working paper PDF](https://www.atlantafed.org/-/media/documents/research/publications/wp/2005/wp0507a.pdf)). A Python implementation with stationary/block bootstrap exists: [model-confidence-set on PyPI](https://pypi.org/project/model-confidence-set). An LLM-adjacent application used block bootstrap over five configurations, with a set of 2 at 90% and 1 at 75% — [arXiv 2605.05739](https://arxiv.org/pdf/2605.05739)
- **GLMM with Bonferroni contrasts** was used for multi-arm quantization comparisons on long context — [arXiv 2505.20276](https://arxiv.org/abs/2505.20276)
- **Paired SE.** "Paired differences represent a 'free' reduction in estimator variance" — [Miller, arXiv 2411.00640](https://arxiv.org/pdf/2411.00640). Locally, pairing reduced the SE from 0.000746 to 0.000472 — [LOCAL healing_paired_test.json](file:///home/patrickd/glm-5.3-reap/artifacts/eval/healing_paired_test.json)
- **Leakage controls.** Calibration/eval leakage is controlled by 8-gram disjointness, plus a calibration-vs-held-out KL leak check — [LOCAL eval_protocol_pruned_moe.md §KQ6](file:///home/patrickd/glm-5.3-reap/research_notes/Compressing%201T%20MoE%20for%20Thor/eval_protocol_pruned_moe.md)

### Inferences
- **Token-cost multiplier for family-wise α = 0.05, one-sided "winner beats each of K−1", 80% power [arithmetic]:** K = 5 → ×1.54; K = 10 → ×1.85; K = 20 → ×2.13; K = 50 → ×2.49; K = 100 → ×2.76. With K = 10 finalists at δ = 0.25% and ω² = 0.08, the confirmation split needs ≈ 104k × 1.85 ≈ 190k tokens per pooled comparison before clustering. This is cheap enough that the only real constraint is keeping the split untouched.
- **Winner's curse.** Picking the best of many candidates on the same tokens inflates its apparent margin. The confirmation split must be disjoint from the screening split and from every calibration corpus, and must not be looked at until the finalist list is frozen and logged with a hash.
- **Model Confidence Set over Bonferroni** when K is large or candidates are correlated (masks sharing most experts). The bootstrap resamples **documents**, which preserves both within-document clustering and cross-candidate correlation. Use the paired per-document mean KL as the loss.
- **Per-domain claims need a second multiplicity layer.** Either claim "best on pooled tokens, non-inferior on every domain" (one superiority test plus D non-inferiority tests with Holm), or report the per-domain MCS. Do not claim per-domain superiority from a pooled test.
- **Pre-register:** the metric (paired mean top-k KL as primary), the margin δ, the confirmation-split hash, K, the test (MCS at 90%/95%, or Holm-MCB), and the capability non-inferiority margins. Everything else is reported as exploratory.

### Gaps
- No LLM-compression paper using a Model Confidence Set or Hsu MCB was found. The design transfers from econometrics and forecast evaluation.

## 6. Comparing fairly against competing compressions, and making the Pareto claim rigorous

### Takeaway
Fairness requires five things:
1. Every competitor is scored against the **same** cached teacher, on the **same** token IDs, at the **same** context/chunking, with **no** chat template or BOS drift.
2. Each competitor runs in its intended runtime, emitting top-k logprobs into one common scoring format.
3. A runtime-noise control bounds cross-runtime numeric differences.
4. "Unified-RAM-viable" is defined operationally: measured peak memory at a stated context on Thor, with the Engram-on-NVMe policy stated, rather than file size.
5. The claim is a Pareto statement: no viable competitor at ≤ our bytes is in the confidence set of best agreement, and baselines (random, frequency, REAP, HOPE, community quants) populate the frontier so the curve is not cherry-picked.

### Cited Findings
- **Competitors exist for V4.1 and V4.**
  - LibertAI's REAP-256E for V4.1-Flash slices "128 experts per layer, along with their rows in `gate.weight`, `gate.bias`, and `gate.bias_vl`" — [LibertAIDAI/DeepSeek-V4.1-Flash-REAP-256E](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-REAP-256E)
  - 0xSero's V4-Flash-0731 REAP K160 — [0xSero/DeepSeek-V4-Flash-0731-REAP](https://huggingface.co/0xSero/DeepSeek-V4-Flash-0731-REAP)
  - V4-Flash GGUFs from Unsloth (UD-Q4_K_XL keeps expert weights and quantizes non-expert tensors, about 4% of the model, to Q8_0) and from several community uploaders — [unsloth/DeepSeek-V4-Flash-GGUF](https://huggingface.co/unsloth/DeepSeek-V4-Flash-GGUF), [sokann](https://huggingface.co/sokann/DeepSeek-V4-Flash-GGUF), [teamblobfish](https://huggingface.co/teamblobfish/DeepSeek-V4-Flash-GGUF)
- **Not all GGUFs are faithful.** llama.cpp DeepSeek-V4 support landed in PR 24162, with a multi-turn q8_0-KV bug fixed in PR 25202. Unsloth's docs claim GGUFs from other providers "were converted without those code paths and so deviate from the official weights" (a vendor claim, unverified) — [Unsloth DeepSeek-V4 docs](https://unsloth.ai/docs/de/modelle/deepseek-v4.md)
- **The tables changed in 1017.** "Do not reuse the Engram shards of the older deepseek-ai/DeepSeek-V4.1-Flash". A V4.1 (0909) or V4 (0731) derivative is therefore a *different teacher's* compression — [LOCAL target_models.md](file:///home/patrickd/glm-5.3-reap/research_notes/Compressing%201T%20MoE%20for%20Thor/target_models.md)
- **Prior frontier knowledge.** REAP rates are 34–37.5%; pruning is about 14× more damaging than quantization per byte saved; HOPE adds +2.8–6.1% agentic at 40–50% — memory notes [thor-model-fit-frontier](/home/patrickd/.claude/projects/-home-patrickd/memory/thor-model-fit-frontier.md), [reap-criterion-and-corpus-audit](/home/patrickd/.claude/projects/-home-patrickd/memory/reap-criterion-and-corpus-audit.md), [hope-supersedes-reap](/home/patrickd/.claude/projects/-home-patrickd/memory/hope-supersedes-reap.md)
- **Gemma-4 -it BOS trap.** The -it tokenizer omits `<bos>`, giving raw NLL 9.6 vs 4.2. Force identical special-token handling in every harness — [memory gemma4-it-bos-trap](/home/patrickd/.claude/projects/-home-patrickd/memory/gemma4-it-bos-trap.md)
- **Frequency-mask baseline collapses.** In the REAP table, 50% Frequency pruning on Kimi gives BFCL overall 0.218 versus 0.564 for REAP, so frequency and random masks are meaningful lower anchors — [arXiv 2510.13999](https://arxiv.org/abs/2510.13999)
- **Runtime/format numerics are themselves detectable.** The bf16-vs-fp16 control gives KLD 2.5e-5 and 99.74% same-top — [llama.cpp README](https://github.com/ggml-org/llama.cpp/blob/master/tools/perplexity/README.md)

### Inferences
- **Common scoring harness.**
  - Every candidate (ours, LibertAI REAP-256E, any 1017 GGUF or EXL3, the random/frequency/REAP/HOPE masks we build ourselves) emits, for the frozen confirmation token stream, top-k (k ≥ 32) logprobs plus the logprob of the teacher's top-k indices.
  - vLLM provides this via `prompt_logprobs`, llama.cpp via `n_probs` or an exported logits file, and the custom CUDA server via a native dump.
  - Score centrally against the cached teacher with the same `compare()` code (`s09_eval`).
  - Do **not** rely on `llama-perplexity --kl-divergence-base`. It needs the teacher itself in GGUF and full-vocab logit files, about 37 GiB for Llama-3 on Wikitext-2 per the sibling note. A shared top-k format is cheaper and runtime-neutral.
- **Runtime-noise control.** Run one model in two runtimes, for example our winner in vLLM and in the CUDA server, or one GGUF in llama.cpp and ik_llama.cpp. The KL between those two arms is the floor below which no cross-runtime difference can be attributed to compression. Report it next to every cross-runtime comparison.
- **Teacher identity.** Only compressions of **1017** are comparable on agreement with the 1017 teacher. Score V4/0909 derivatives anyway, but label them as "different base". Their KL includes base drift, so beating them is not a like-for-like win.
- **Faithfulness gate for community artifacts.** Before a GGUF enters the frontier, check its KL against the teacher at a high-bit quant (Q8_0-class if one exists) to confirm the conversion is faithful. A conversion bug would otherwise masquerade as quant damage.
- **Viability axis, measured not computed.** Report peak resident memory (nvmap plus host) at the stated serving context with the stated runtime, the Engram placement (NVMe O_DIRECT row store vs resident), and KV dtype. Use ≤ 117 GiB, the user's measured usable envelope. Plot agreement against *measured* GiB. Models that do not fit at the stated context are plotted but marked non-viable.
- **Baselines that make the frontier honest, on the same byte axis [design]:**
  - (i) uniform-random mask at our keep count, three seeds;
  - (ii) frequency mask;
  - (iii) plain REAP;
  - (iv) HOPE;
  - (v) our pipeline;
  - (vi) every public 1017 compression that fits;
  - (vii) quant-only points: the unpruned teacher at the lowest-bit GGUF that fits 117 GiB, if one exists, which tests the "pruning is 14× worse than quant" prior directly.
  Random and frequency masks anchor the bottom of the scale. An agreement gap smaller than our-vs-REAP would be uninformative without them.
- **Claim template:** "On N_doc held-out documents (N_tok tokens, 7 domains, disjoint from all calibration data, hash H), against DeepSeek-V4.1-Flash-1017 reference logits (source A, native MXFP4/FP8), model X has mean top-32 KL κ_X [95% doc-bootstrap CI] and top-1 agreement a_X [CI]. Among the K candidates whose measured peak memory at C-token context is ≤ 117 GiB, the 90% Model Confidence Set is {X}. X is non-inferior to each of them on the capability panel at the pre-registered margins. Across the byte range 80–117 GiB, no viable candidate dominates X on (bytes, KL)."
- **Free-running confirmation.** Add API-teacher trajectories (source C, after its equality check) with expected accept length and first-divergence position, to show that the teacher-forced agreement ordering survives generation.

### Gaps
- No public V4.1-Flash-**1017** GGUF or EXL3 with KLD numbers was found. Competitor coverage may be limited to LibertAI's REAP and whatever appears later.
- Unsloth's claim that other converters' V4 GGUFs deviate from the official weights is unverified, and was observed on V4, not V4.1.
- Whether llama.cpp supports the V4.1 architecture deltas (Single-Pass mHC, Engram, `bias_vl`) was not confirmed. Competitor GGUFs may not exist or may not run.
