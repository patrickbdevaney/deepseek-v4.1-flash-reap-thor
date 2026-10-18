# Evaluation protocol for heavily pruned/quantized MoE models (Thor, cached teacher)

Scope: which tests, how many prompts/tokens per test for a stated MDE, paired design against the cached FP8 teacher, bootstrap over documents, the free-running loop detector, the per-domain breakdown, gates, and Thor wall-clock per tier. All sample-size arithmetic uses Miller's Eq. 9 (paired power formula) or the standard two-proportion / one-proportion formulas; every number that is not from a cited source is marked **[extrapolation]** or **[arithmetic]**.

Throughput assumptions used throughout (see the Thor section for sources): batch-1 decode D1 = 10 tok/s (user-measured 8–12 on REAP-50 GLM-5.3 NVFP4); batched decode D8 ≈ 30 tok/s aggregate (3.2x from Thor vLLM C1→C8 on gpt-oss-120b, **[extrapolation]** to a 100 GiB model; the user's own CUDA server batches at only ~1.4–1.5x); prefill P_low = 20 tok/s (user's CUDA server, 48–52 ms/tok at chunk 32), P_mid = 300 tok/s (llama.cpp-class ~100 GB MoE, Spark-measured 270–291, Thor ≈ same order), P_high = 900 tok/s (gpt-oss-120b MXFP4 llama.cpp pp2048 on Thor).

## KQ1. Looping/degeneration metrics and the n ≥ 2000 free-running protocol

### Takeaway
The community "loop rate" (0xSero) is a prompt-level attractor detector run with no max_tokens and reported as fraction of prompts that never terminate naturally or enter a repeat/`</think>`-restart loop; the academic metrics are CREP (fraction of sequences where one 4–16-gram covers ≥30% of tokens) and SER@4 (per-step suffix-extension rate). Measured unpruned baselines are 3.6% prompt-level (GLM-5.2 teacher, agentic coding prompts), 0.088 SER@4 (GPT-OSS-120B), and 5.9% CREP sampled / 26.6% greedy (Llama-3.1-8B open-ended). n=2000 per arm resolves a 1.5x rise from a 3.6% base; ~260 prompts resolve a doubling; ~115 agentic prompts resolve the measured 6%→13% agentic doubling when the teacher arm is cached and tested one-sample.

### Cited Findings
- 0xSero GLM-5.2-504B (REAP keep-168 = 34% pruned, Router-KD, NVFP4): n=2000 held-out prompts "harvested from genuine coding-agent traces (codex, opencode, cursor, claude-code)", "raw sampling, no `max_tokens`, no timeout", loops "detected, never truncated". Attractor/loop rate 0.072 pruned vs 0.036 teacher ("z≈5, p<0.0001"); natural-EOS rate 0.928 vs 0.965; distinct-4 0.880 vs 0.921; median output length 1267 vs 1207 tokens; codex/opencode sources loop "~13% here vs ~6% on the teacher"; a brevity system prompt halves median length (1267→507) but does not reduce looping; a full-data floor KD gave 8.6% (worse); "about 1% of prompts loop on the teacher too" (the card is internally inconsistent: 3.6% in the table vs "about 1%" in prose). Sampler mitigation: min_p=0.05 + rep-penalty 1.05 → 4.9%; 1.10 → 2.3% (distinct-4 0.95). The card does NOT define the loop detector (no n-gram window or threshold), does not state temperature for this eval, and does not state whether both arms used the same 2000 prompts; it points to a REPORT.md not retrieved — [0xSero/GLM-5.2-504B](https://huggingface.co/0xSero/GLM-5.2-504B)
- Check of the card's z: with p̄=0.054, SE = sqrt(2·0.054·0.946/2000) = 0.00715, z = 0.036/0.00715 = 5.03 — consistent with an unpaired two-proportion z-test on independent arms **[arithmetic]**.
- CREP definition (FOCUS/RePAIR, pruned-LLM degeneration paper): Coverage(r, s) = fraction of tokens explained by occurrences of N-gram r; sequence-level Coverage = max over r; a sequence is degenerate if Coverage ≥ τ; "CREP computes the fraction of sentences dominated by repeated n-grams (above 30%). For each sentence, we evaluate repetition across n-gram sizes r ∈ [4, 16] and mark it as degenerated if any r exceeds the threshold"; onset threshold for negative samples set to 0.3. Measured on the unpruned Llama-3.1-8B: CREP 5.9% under top-p 0.9 sampling and 26.6% under greedy; replacing only the first two onset tokens cuts these to 0.7% / 10.8% ("degeneration often depends on a small subset of loop-sensitive contexts near the first onset"). Setup: 1,000 WikiText-103 prefixes (50-token prefix, 100 generated tokens) and 1,000 Self-Instruct prompts, top-p 0.9; pruned models (LLM-Pruner 25/35/45% width) + KD baseline: CREP 7.3% (KD) vs 2.23% (RePAIR) on WikiText-103 at near-identical PPL (21.69 vs 22.05) — i.e. PPL is flat while CREP moves 3x — [arXiv 2608.26676](https://arxiv.org/abs/2608.26676)
- SER@L definition (DRY paper): "the fraction of decoding steps where the sampled token extends a previously seen suffix of length ≥ L"; "SER@4 ranks methods identically to the token-level repeated-n-gram rate [Welleck 2019] but counts only true suffix continuations"; also reports MMSL, rep-4, rep-8, loop-free length (LFL), distinct-4, MAUVE. Uncontrolled baselines (macro over Qwen2.5-1.5B/Llama-3.2-3B/14B, 3 seeds, 9 prompt families, 16,566 generations): SER@4 0.124 [95% CI .087–.160], SER@8 0.069, D-4 0.958, LFL 100.1 tokens; GPT-OSS-120B uncontrolled SER@4 0.088 [.082–.095]; "baseline loop rates decline as scale grows but the failure mode is far from eliminated". CIs are a cluster bootstrap across model×seed cells, 5000 resamples; seeds 7/42/123 — [arXiv 2608.22761](https://arxiv.org/abs/2608.22761)
- LZ-Penalty paper's degenerate-repetition check: an LLM judge plus "a naive exact-substring search (a substring of length at least 3 repeated at least 20 times)" — [arXiv 2504.20131](https://arxiv.org/abs/2504.20131)
- REAP paper's own loop handling: for SWE-Bench, rollouts are restarted up to 20 times "unless the model produces degenerate responses like repeating strings"; for τ²-bench, "Every 30 steps of the simulation, a model (gpt-4.1-mini) is given the past 30 episodes ... with a repetition checking prompt to determine whether the agent is stuck in the loop"; all large-model evals use repetition_penalty 1.05, greedy for EvalPlus/SWE-Bench/BFCLv3, max output 4,096 (BFCL), 16,384 (LiveCodeBench, Kimi SWE-Bench) — [arXiv 2510.13999 App. C](https://arxiv.org/abs/2510.13999)
- Adjacent documented breakages: "Gemma-4 at 50%: free-running generation collapses into repetition while perplexity looks fine; 75%-pruned Qwen3.6 variants degenerate on 2–6 of 8 smoke prompts" — [local note expert_pruning_beyond_50.md §223](/home/patrickd/glm-5.3-reap/research_notes/Compressing%201T%20MoE%20for%20Thor/expert_pruning_beyond_50.md)
- Error amplification is the mechanism: pruning deviations "accumulate across time steps and lead to substantial degradation during generation" while "retrieval and multiple-choice selection" stay robust — [arXiv 2603.24652](https://arxiv.org/abs/2603.24652)

### Inferences
- **Loop metric to implement (three-way, all computed on the same generations):**
  (1) `loop_prompt` ∈ {0,1}: sequence is degenerate if CREP-coverage ≥ 0.30 for any n ∈ [4,16] (FOCUS threshold) OR an exact substring of ≥3 tokens recurs ≥20 times (LZ-penalty rule) OR generation hits the token cap without EOS (0xSero natural-EOS complement). Report the prompt-level rate; this is the number comparable to 0xSero's 3.6%/7.2%.
  (2) `SER@4` per token (DRY), which is continuous and gives ~10x more statistical power per prompt than the binary rate.
  (3) `distinct-4` and `natural_EOS` as 0xSero does, so the result is directly comparable to the one published REAP looping measurement.
  Loops must be detected, not truncated by a repetition penalty: run the detector with the sampler the model will ship with (0xSero shows 1.10 rep-penalty hides 2/3 of loops).
- **Sample sizes [arithmetic, two-proportion, α=0.05 two-sided, 80% power, unpaired]:** detect 3.6%→5.4% (1.5x): n ≈ 2,080 per arm — this is why 0xSero's n=2000 is the right order; 3.6%→7.2% (2x): n ≈ 620 per arm; 3.6%→10.8% (3x): n ≈ 200; 3.6%→20%: n ≈ 60. On the agentic subset (6%→13%): n ≈ 275 per arm.
- **Cache the teacher arm.** The teacher's free-running outputs are the same for every candidate mask, so after one 2000-prompt teacher run every later candidate is a one-sample test against a known p0 **[arithmetic, one-proportion]**: 3.6%→7.2% needs ≈ 262 prompts; 6%→13% (agentic) ≈ 114 prompts; 3.6%→5.4% ≈ 905. Use the same prompt IDs so a McNemar paired test is also available.
- **Generation config:** greedy inflates CREP ~4.5x over top-p 0.9 (26.6% vs 5.9%), so the loop rate must be measured under the shipping sampler AND under greedy; greedy is the more sensitive screen (higher base rate → more power per prompt) but is not what users see. Pin temperature/top-p/min_p/rep-penalty, max_tokens, seqmax and reasoning_effort identically across arms (internal evidence: changing --seqmax re-routes greedy decoding; servers disagree on default reasoning_effort — memory notes eval-seqmax-perturbs-greedy-output, eval-pin-generation-config).
- **Prompt sources:** mirror 0xSero: held-out agentic traces (codex/opencode/cursor/claude-code style, multi-tool system prompts), because the loop rate is ~2x higher there (13% vs 6%) and that is where it bit; add a WikiText/Self-Instruct open-ended family for comparability with the CREP literature. Keep at least 9 prompt families (DRY) and report per-family rates; include "necessary-repetition" families (lists, refrains, code with repeated labels) so the detector's false-positive rate is known.
- **Max tokens:** 0xSero's median was 1267 tokens with no cap and loops are onset events that can start late; a 512-token cap is a cheap screen that will miss late loops (onset distribution unmeasured — gap). Use 1024 in tier 1/2, uncapped (or 8192) in tier 3.
- **Threshold:** pass if candidate prompt-level loop rate ≤ 1.5x teacher's on the same prompts AND natural-EOS ≥ 0.95x teacher's; fail if ≥ 2x teacher (0xSero shipped a 2x model only with sampler mitigation, and the card's own framing treats 2x as the problem) **[threshold is extrapolated from the single published REAP measurement]**.

### Gaps
- No source defines the exact detector behind 0xSero's 0.072/0.036; the REPORT.md was not retrieved. The prompt-level number may not be reproducible by another detector.
- No paper reports the onset-position distribution of loops in pruned models, so the power loss from a 512/1024-token cap is unknown.
- No unpruned-teacher loop baseline exists for GLM-5.3 itself; the 3.6% figure is GLM-5.2 on one prompt set.

## KQ2. Multi-turn / agentic collapse: which subset detects the 50%-REAP collapse cheaply

### Takeaway
The REAP paper's own table shows the collapse is specific to Kimi-K2-W4A16 (BFCLv3 multi-turn 0.355→0.164 at 50%) while Qwen3-Coder-480B-FP8 stayed flat (0.380→0.371), and single-turn code/Non-Live/SWE-Bench stayed flat on both; BFCL v3 multi-turn is 4×200 items, and ~80–100 multi_turn_base items suffice to detect a 19-point collapse, but each item is a multi-turn 4k-max-output rollout that costs hours at Thor decode speeds.

### Cited Findings
- REAP Table 3 (large-scale, greedy, rep-penalty 1.05, BFCL max 4,096 tokens, original Gorilla framework): Kimi-K2-Instruct-W4A16 baseline Non-Live 0.840 / Live 0.802 / Multi-Turn 0.355 / Overall 0.666; 25% REAP 0.842 / 0.801 / 0.263 / 0.635; 50% REAP 0.785 / 0.743 / 0.164 / 0.564; 50% EAN 0.778 / 0.767 / 0.173 / 0.573; 50% Frequency 0.255 / 0.397 / 0.003 / 0.218. Same model code: Eval+ 0.828→0.819, LiveCode 0.434→0.429, SWE-Bench-Verified 0.554→0.576 at 50%. Qwen3-Coder-480B-FP8: Multi-Turn 0.380 → 0.392 (25%) → 0.371 (50%); SWE-Bench 0.540→0.522. Paper text: "On tool use, EAN and REAP are comparable, with REAP slightly outperforming at 50% compression with a mean decrease in accuracy of 5.9% versus 6.2%" — [arXiv 2510.13999 Table 3](https://arxiv.org/abs/2510.13999)
- Note the Kimi multi-turn drop already appears at 25% (0.355→0.263, −9.2 pts) while every other Kimi column is within ±2 pts at 25% — the multi-turn subset is the early-warning column — [same table]
- REAP τ²-bench (Qwen3-Coder-480B-FP8, greedy, 4,096 max tokens, gpt-4.1 user simulator, 100 max steps, 3 trials/domain): Retail pass^1 0.643 → 0.661 (25%) → 0.632 (50%); Airline 0.460 → 0.487 → 0.447; Telecom 0.500 → 0.529 → 0.471; pass^3 Telecom 0.325 → 0.421 → 0.263 — [arXiv 2510.13999 Table A9](https://arxiv.org/abs/2510.13999)
- BFCLv3 at intermediate ratios: GLM-4.6-FP8 (thinking) 0.784 → 0.773 (25%) → 0.768 (30%) → 0.742 (40%); MiniMax-M2 0.626 → 0.615 → 0.599 → 0.579 — [arXiv 2510.13999 Table A11](https://arxiv.org/abs/2510.13999)
- BFCL v3 composition: Multi Turn Base 200, Missing Function 200, Missing Parameter 200, Long Context 200 (official blog; "this leaderboard composition is now outdated and the BFCL V4 blog has the most updated composition") — [Gorilla BFCL v3 blog](https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v3_multi_turn.html); evalscope lists the same four subsets at 200 each, and BFCL v4 overall weights: Agentic 40%, Multi-Turn 30%, Live 10%, Non-Live 10%, Hallucination 10%; v4 adds web_search_base/no_snippet (100 each) and memory_kv/vector/rec_sum — [evalscope BFCL v4](https://evalscope.readthedocs.io/en/latest/third_party/bfcl_v4.html)
- τ²-bench sizes: Airline 50, Retail 115, Telecom 114 tasks (subsampled from 2,285 generated); "Each task is run four times"; pass^k = fraction of k independent runs that succeed; no variance/CI reported; requires an LLM user simulator — [arXiv 2506.07982](https://arxiv.org/abs/2506.07982)
- Flip-rate evidence that accuracy-preserving compression still changes individual answers: six quantization schemes, seven tasks, negligible accuracy change but "large number of flips" for everything except GPTQ W8A16; lower bits → more flips; KL vs flips Spearman 0.981 on MMLU; high-flip models score worse on MT-Bench — [arXiv 2407.09141](https://arxiv.org/abs/2407.09141)
- Local prior: Cerebras GLM-4.6 40%: BFCL 78.4→74.2, GPQA-D 78.8→69.7, code within 1–3 pts — [local note sota_shipped_artifacts.md §134](/home/patrickd/glm-5.3-reap/research_notes/Compressing%201T%20MoE%20for%20Thor/sota_shipped_artifacts.md)

### Inferences
- **Which subset:** BFCLv3 `multi_turn_base` (200 items) is the single most diagnostic column: it moved −9 pts at 25% and −19 pts at 50% on Kimi while Non-Live/Live/code were flat. `multi_turn_long_context` is the second (it combines the multi-turn failure with the long-context one). Non-Live/Live single-turn are nearly useless as gates (they moved ≤6 pts even at 50%).
- **Minimum items [arithmetic, two-proportion unpaired]:** detect 0.355→0.164 (Δ=0.19): n ≈ 82 per arm; detect 0.355→0.263 (Δ=0.09, the 25% early warning): n ≈ 380 per arm — more than the 200 available, so the 25%-size signal is only resolvable paired (same items, McNemar) or by pooling base+long_context (400). At n=200 unpaired the MDE at p≈0.35 is ≈ 2.8·sqrt(2·0.35·0.65/200) ≈ 13 pts; paired MDE depends on the discordance rate (unmeasured) and would be lower.
- **Gate:** fail a candidate if paired multi_turn_base accuracy drops > 5 pts vs the previous stage (REAP's "acceptable" 50% tool-use result was −5.9% mean on the Overall column; the collapse was −19) **[threshold extrapolated from the two published data points]**. Because the drop was already −9 pts at 25% on Kimi, a per-stage gate at 5 pts would have stopped that pipeline at stage 1.
- τ²-bench is not Thor-local (needs a gpt-4.1-class user simulator and 100-step rollouts) and was less sensitive than BFCL multi-turn on the one model where both were run; keep it out of the per-stage gate.
- **Cost [extrapolation — BFCL per-item token counts are not published]:** assume ~15k prefill + ~3k generated tokens per multi-turn item across turns. 100 items: prefill 1.5M tokens (P_mid 83 min; P_low 21 h) + decode 300k tokens (D1 8.3 h; D8 2.8 h). So a 100-item multi_turn_base gate ≈ 4–5 h on Thor with P_mid+D8, ≈ 10 h at batch 1. The full 800-item v3 multi-turn ≈ 30–80 h.

### Gaps
- No source explains why Kimi-K2 (W4A16 int4 base) collapsed and Qwen3-Coder (FP8) did not; whether it is the int4 base × pruning interaction, Kimi's routing, or calibration is unknown. This matters because our base is NVFP4.
- BFCL per-item token/turn counts are not published; the Thor cost above is an estimate.
- No paper reports per-item variance or paired SEs for BFCL multi-turn.

## KQ3. Long-context: RULER/NIAH/LongBench-v2/HELMET subsets sized for Thor at 32k–128k

### Takeaway
Mekala et al. (EMNLP 2025) show 4-bit damage is invisible at 8K and significant only at 64K/128K (BNB-nf4 p<0.001; GPTQ-int4 p=0.002/<0.001), using 100 examples per {task, length} and a GLMM with Bonferroni contrasts; HELMET finds NIAH correlates ≤0.8 with real tasks and recommends RAG tasks for fast development; 100 items/length only resolves ~15-pt drops unpaired, and prefill cost (≥3M tokens per 100 items at 32k) makes 64k+ a tier-3-only test on Thor.

### Cited Findings
- Mekala et al.: Llama-3.1 8B/70B, Qwen-2.5 7B/32B/72B; FP8, GPTQ-int8, AWQ-int4, GPTQ-int4, BNB-nf4; RULER at 8K/64K/128K with "100 examples" per {task, context-length} (3 tasks: needle, multi-key, multi-value → 300 per length, 27,000 observations total); "At the 8K context length, all methods perform similarly to BF16"; "8-bit methods remain robust, with accuracy drops under 5%"; 4-bit drops using 64K+128K only: GPTQ-int4 Llama-3.1-8B 21.2%, Qwen-2.5-72B 10.3%; BNB-nf4 Llama-3.1-8B 30.7%, Qwen-2.5-7B 29.7%, Qwen-2.5-32B 15.7%; Llama-3.1-70B BNB-nf4 "32% performance drop" while Qwen-2.5-72B "remains robust under BNB-nf4"; non-English worsens. Statistics: binomial-logit GLMM (`glmer`, lme4), emmeans pairwise contrasts, Bonferroni; P(BF16 more accurate) BNB-nf4 0.682 (8K, p=0.58) / 0.668 (64K, p<0.001) / 0.759 (128K, p<0.001); GPTQ-int4 0.819 (8K, p<0.001) / 0.584 (64K, p=0.002) / 0.640 (128K, p<0.001); AWQ-int4 0.514 (64K, n.s.) / 0.579 (128K, p=0.001); per-task BNB-nf4 needle 0.718, multi-key 0.651, multi-value 0.611 (all p<0.001) — [arXiv 2505.20276](https://arxiv.org/abs/2505.20276)
- Conflicting: Neural Magic/Red Hat W4A16 (INT4 weight-only) Llama-3.1 8B/70B kept >99% of unquantized RULER through 64K, with limits only at 128K where the unquantized model also struggled — [Red Hat article](https://developers.redhat.com/articles/2024/02/03/how-well-do-quantized-models-handle-long-context-tasks). Both agree RULER exposes weaknesses NIAH misses.
- HELMET: "randomly sample 100 to 600 examples from each dataset"; Spearman at 128K across 35 models: "none of the synthetic tasks achieves an average correlation higher than 0.8", original NIAH ≤0.8, RULER average <0.85; RAG (HotpotQA) "consistently achieve higher correlation with other real-world tasks"; original NIAH "largely saturated"; RULER MK (distracting needles) "better distributes model performance between 0% and 100%"; "we recommend using our RAG tasks for fast model development" — [arXiv 2410.02694](https://arxiv.org/abs/2410.02694)
- RULER: 13 task configurations (niah variants, variable_tracking, common/freq_words_extraction, qa), results at 4K–128K; default `synthetic` benchmark; num_samples not stated on the README — [NVIDIA/RULER](https://github.com/NVIDIA/RULER)
- LongBench v2: 503 MC items; Short (<32k words) 180, Medium (32k–128k) 215, Long (>128k) 108 — [arXiv 2412.15204](https://arxiv.org/abs/2412.15204)
- REAP long-context: LongBench v2 and FRAMES evaluated with middle truncation above 131,072 tokens; "even though the maximum sequence length during calibration is 16,384 tokens, the REAP-compressed model retains performance at context lengths far exceeding that value" (Kimi-Linear-48B at 30%) — [arXiv 2510.13999 App.](https://arxiv.org/abs/2510.13999)

### Inferences
- **Which lengths detect pruning vs quantization:** quantization damage is a 64K+ phenomenon in Mekala's data and absent at 8K; pruning damage to long context is undocumented beyond the single REAP Kimi-Linear statement. So a 32k test is a pruning-specific probe (quant noise is still near zero there), and a 64k/128k test is where quant and pruning stack. Run 32k in tier 2 and 64k in tier 3.
- **Tasks:** RULER multi-key NIAH (most discriminative per HELMET) + multi-value + one RAG-style task (HotpotQA-in-context) at each length; skip plain NIAH (saturated).
- **Sample sizes [arithmetic]:** at p≈0.8, detecting a 10-pt drop unpaired needs ≈ 290 items per length per arm; 100 items/length (Mekala) has unpaired MDE ≈ 17 pts; paired (same needles/haystacks, same positions) MDE ≈ 10–12 pts **[extrapolation; discordance unmeasured]**. Mekala resolved 10–30% drops with 300 items/length pooled across tasks in a GLMM — pool tasks the same way.
- **Thor cost [arithmetic]:** 100 items × 32k = 3.2M prefill tokens → P_mid 3.0 h, P_high 1.0 h, P_low 44 h (infeasible). 300 items × 64k = 19.2M → P_mid 18 h, P_high 5.9 h. Decode is negligible (≤64 tokens per item). Long-context gating is therefore only affordable on a fast-prefill stack (llama.cpp/vLLM class), not on the 20 tok/s custom server.
- **Gate:** fail if paired 32k RULER-MK+MV drops > 5 pts vs the previous stage (Mekala's "8-bit stays within 5%" is the only published tolerance) **[extrapolated threshold]**.

### Gaps
- No paper measures long-context degradation from expert pruning as a function of pruning ratio; the one REAP statement is qualitative.
- Mekala's raw accuracies are only in heatmap figures; the 3–16% band in the brief corresponds to their 10–30% drops and the Red Hat <1% result — the discrepancy is method-dependent (BNB-nf4/GPTQ-int4 vs W4A16) and model-dependent.

## KQ4. Ballast/general knowledge: MMLU, PopQA, BBH, factoid QA, per-domain KL — minimum items for a 2-point MDE

### Takeaway
Knowledge-intensive generative tests (PopQA/FreebaseQA free-form, LLM-KICK style) are the published ballast detectors — pruning loses "over 97%" of FreebaseQA at N:M sparsity while PPL is flat, and the Benchmark Illusion paper shows 50% Wanda-pruned Qwen3-8B drops 9.1 pts in open generation but only ~3 pts in MC; paired on the same items, ~1,600 MMLU items resolve 2 pts and ~1,000 PopQA items resolve 3 pts, but MC must be scored by next-token log-prob (teacher-forced) not letter generation.

### Cited Findings
- LLM-KICK (ICLR 2024): re-evaluates compression beyond perplexity; "pruning methods lose significant performance, sometimes at trivial sparsity ratios (e.g., 25-30%)"; N:M sparsity "catastrophic failure on factual question-answering (e.g., FreebaseQA), losing over 97% of accuracy"; quantization holds up better; pruned models at ~50% still do in-context retrieval and summarization — [arXiv 2310.01382](https://arxiv.org/abs/2310.01382), [VITA-Group/llm-kick](https://github.com/VITA-Group/llm-kick)
- Benchmark Illusion (Wanda 50%, Qwen3-8B, TyDiQA 200 Q/language): open-generation mean 46.9%→37.8% (−9.1 pts) while MC 90.0%→86.8% (standard distractors) and 85.0%→82.1% (hard); headroom-normalized drops 0.21 (open generation) vs 0.05 (MC); of items both models got right in MC, 12% (English) to 50% (Bengali) become "MC-only" after pruning; gold first-token median rank 3.5 on MC-only items; beam-5 recovers 20.2% and 10-sample 30.2% of 560 greedy failures; 1-shot shrinks the gap (Arabic 12.5→1.0 pts); "Compressed models should be tested on what they can produce, not only on what they can recognize" — [arXiv 2606.17609](https://arxiv.org/abs/2606.17609)
- Layer-pruning analogue: multi-step generative reasoning (arithmetic, balanced-parenthesis generation) degrades first; fine-tuning recovers ~90% on classification but "recovery for generative reasoning remains fundamentally limited" — [arXiv 2602.01997](https://arxiv.org/abs/2602.01997)
- Letter-only MC artifact (internal): Gemma-4-26B-A4B 0.48 letter-only vs 0.967 reasoned on Belebele-en — re-test reasoned before blaming quant — [memory: letter-only-mc-format-artifact](/home/patrickd/.claude/projects/-home-patrickd/memory/letter-only-mc-format-artifact.md)
- Benchmark sizes: PopQA ~14k entity-centric questions, 16 Wikidata relations, long-tail entities (13k after dedup in some papers) — [alphaxiv PopQA](https://alphaxiv.org/benchmarks/university-of-washington/popqa); BBH 23 tasks, 6,511 examples, most tasks 250 — [alphaxiv BBH](https://alphaxiv.org/benchmarks/stanford-university/big-bench-hard)
- Miller: for next-token-probability scoring "the standard error of expected scores across questions" applies (conditional variance σ² = 0); SE_Bernoulli = sqrt(s̄(1−s̄)/n); paired SE = sqrt(Var(s_A−s_B)/n); fictional worked example ω²=1/9, δ=0.03 → n≈969, "new evals should contain at least 1,000 questions" — [arXiv 2411.00640](https://arxiv.org/abs/2411.00640)
- Local prior: general-knowledge benchmarks "moved 20–30 points in HOPE/REAP at 50% with code-only calibration while GSM8K stayed ~95"; our own ballast top-1 agreement 0.579 vs code 0.916 — [local note calibration_and_evaluation.md §164, §213](/home/patrickd/glm-5.3-reap/research_notes/Compressing%201T%20MoE%20for%20Thor/calibration_and_evaluation.md)

### Inferences
- **Score MC by teacher-forced log-prob of the answer token** (Miller's variance-reduction recommendation, also what lm-eval `acc_norm` does): σ²=0, one prefill per item, no decode, and no letter-format artifact. Then separately run a **generative** factoid set (PopQA free-form exact-match, 0-shot and 1-shot) because the Benchmark Illusion shows MC under-detects pruning loss by ~4x headroom-normalized.
- **Items for a 2-pt MDE [arithmetic, Miller Eq. 9 paired, 80% power, α=0.05]:** n = 7.84·ω²/δ². For binary scores ω² ≈ discordance rate. With a 5–10% flip rate (the Accuracy-is-not-all-you-need band): ω²≈0.08 → MMLU n ≈ 1,570 items for 2 pts, ≈ 700 for 3 pts; unpaired would need ≈ 7,350. Full MMLU (14,042) paired gives MDE ≈ 2.8·sqrt(0.08/14042) ≈ 0.67 pt. PopQA at p≈0.45 with discordance ≈0.12: n ≈ 1,045 for 3 pts, ≈ 2,350 for 2 pts. BBH: 250/task means per-task MDE ≈ 2.8·sqrt(0.08/250) ≈ 5 pts — use BBH only pooled (6,511 → ~1 pt).
- **Per-domain held-out KL is the cheapest ballast detector** (see KQ5): the 0.579-vs-0.916 agreement gap was 34 pts, far above any plausible MDE; the gate is "no domain's top-1 agreement below X" (KQ8).
- **Confident factual errors:** define `confident_flip` = teacher-forced token where student argmax ≠ teacher argmax and student p_max ≥ 0.9 (internal NVFP4-overlay run measured 0.138% confident flips — memory glm53-nvfp4-dense-overlay); and on PopQA, wrong answers whose first-token probability ≥ 0.9. Both are paired and cost nothing extra.
- **Thor cost [arithmetic]:** MMLU 1,600 items × ~900 tokens 5-shot = 1.44M prefill → P_mid 80 min, P_high 27 min (no decode). PopQA 1,000 items × 120 prompt tokens = 120k prefill (7 min) + ≤32 output tokens ×1000 = 32k decode → D8 18 min, D1 53 min.

### Gaps
- No published paired discordance rates for MMLU/PopQA under expert pruning; the ω² ≈ 0.08–0.12 used above is extrapolated from quantization flip rates (3–13.6%). Measure it on the first candidate and re-plan.
- LLM-KICK's own per-task item counts were not retrieved.

## KQ5. Teacher-paired KL/agreement at scale: tokens per domain for 0.5% / 0.2% MDE, bootstrap over documents, pinning config

### Takeaway
With our measured per-token paired variance (SE 0.00075 on 241,516 tokens → Var = 0.136), Miller Eq. 9 gives 42.7k tokens for a 0.5% top-1-agreement MDE, 267k for 0.2%, 1.07M for 0.1% — before within-document clustering, which Miller measured at 1.10–3.05x SE inflation (1.2–9.3x tokens); llama.cpp's KLD tooling reports Mean KLD to ±0.9% relative on Wikitext-2 but KL is heavy-tailed (max 0.012 vs mean 2.5e-5 for bf16-vs-fp16) so document-level bootstrap, not the Gaussian SE, is the right CI.

### Cited Findings
- Miller Eq. 9: n = (z_{α/2}+z_β)²(ω² + σ²_A/K_A + σ²_B/K_B)/δ², ω² = Var(x_A)+Var(x_B)−2Cov(x_A,x_B); for deterministic scoring σ²=0; Eq. 10 inverts to MDE = (z_{α/2}+z_β)·sqrt(ω²/n); clustered SE Eq. 4 adds the within-cluster cross-terms; real Anthropic data: DROP clustered/naive SE ratio 3.05, RACE-H 1.10, MGSM 1.88 ("up to 3X"); cluster-adjusted versions of Eq. 9/10 in App. C; bootstrapping is "unnecessary unless a complicated sampling scheme or estimator is being used" (CLT applies) — [arXiv 2411.00640](https://arxiv.org/abs/2411.00640)
- Internal: top-1 agreement SE 0.00075 on 241,516 held-out tokens paired against cached FP8 teacher; v2 pass Δ = −0.0001 (0.13 SE) despite +2.5% reconstruction proxy — [local note calibration_and_evaluation.md §175, §196, §201](/home/patrickd/glm-5.3-reap/research_notes/Compressing%201T%20MoE%20for%20Thor/calibration_and_evaluation.md)
- llama.cpp `--kl-divergence`: Mean KLD with uncertainty assuming per-token KL is Gaussian; Same top p = % tokens where both models' argmax agree, uncertainty from the binomial Gaussian approximation; also Mean Δp, RMS Δp, Δp percentiles, Pearson of correct-token probs. Llama-3-8B on Wikitext-2: q8_0 KLD 0.001355 ± 0.000006 / Same top p 97.674 ± 0.040%; q6_K 0.005452 ± 0.000035 / 96.031 ± 0.051; q4_K_M 0.028152 ± 0.000240 (imatrix) / 91.901 ± 0.072; q3_K_M 0.101913 ± 0.000677; q2_K 0.332223 ± 0.001572 / 71.138 ± 0.119; iq1_S 2.293527 ± 0.004885; bf16-vs-fp16 control: Mean KLD 0.00002515 ± 0.0000002, max 0.012206, 99.9% 0.000799, median 0.000013, Same top p 99.739 ± 0.013% — [llama.cpp perplexity README](https://github.com/ggml-org/llama.cpp/blob/master/tools/perplexity/README.md)
- Flips track KL: Spearman 0.981 between KL divergence and flip rate on MMLU — [arXiv 2407.09141](https://arxiv.org/abs/2407.09141)
- EvoESAP: its ESAP fitness is "a teacher-forced score, inspired by speculative decoding, that estimates how closely a pruned model tracks the full model"; on OLMoE-1B-7B, ESAP beat KL, NLL and a spec-acceptance proxy as a search fitness on coding/math; true spec-decode acceptance as fitness gave "only small downstream differences" but was impractical (ESAP ~18x faster search) — [arXiv 2603.06003](https://arxiv.org/abs/2603.06003)
- Internal: changing --seqmax re-routes greedy decoding (breaks splice exactness); servers disagree on default reasoning_effort — [memory: eval-seqmax-perturbs-greedy-output](/home/patrickd/.claude/projects/-home-patrickd/memory/eval-seqmax-perturbs-greedy-output.md), [memory: eval-pin-generation-config](/home/patrickd/.claude/projects/-home-patrickd/memory/eval-pin-generation-config.md)
- Internal: DRY's CIs used a cluster bootstrap (5000 resamples) over model×seed cells — [arXiv 2608.22761](https://arxiv.org/abs/2608.22761)

### Inferences
- **Token budgets [arithmetic from Miller Eq. 9 with Var=0.136]:** 0.5% → 42.7k; 0.2% → 267k; 0.1% → 1.07M; 0.05% → 4.3M tokens. Cluster inflation c ∈ [1.1, 3.05] multiplies tokens by c²: 0.5% → 52k–397k; 0.2% → 323k–2.5M. **First action: estimate c from the existing 241k run** (cluster SE by document via Miller Eq. 4 vs naive SE); if c ≈ 1.9 (MGSM-like), 0.2% needs ≈ 0.96M tokens.
- **Per-domain:** resolve 0.5% per domain at c=1.9 → ≈ 155k tokens per domain; 7 domains → ≈ 1.1M tokens total (tier 2); 0.2% per domain → ≈ 0.96M per domain, 6.7M total (tier 3 only).
- **KL CI:** use document-level bootstrap (resample documents with replacement, B=2000, percentile CI) rather than the Gaussian ±; KL's max/mean ratio of ~500 in the bf16 control means the per-token Gaussian assumption understates the tail. Report cluster (document) count with every number as Miller Table 3 does.
- **Metric hierarchy to report per domain, all paired on identical tokens:** mean KL(teacher‖student), top-1 agreement, confident-flip rate (p_max ≥ 0.9 disagreements), ΔNLL, and an ESAP-style acceptance score (teacher-forced spec-decode acceptance) since EvoESAP found it the best fitness among teacher-forced metrics.
- **Reference bands (Llama-3-8B/Wikitext-2 numbers, for scale only, not thresholds):** a q4_K_M-class perturbation is KLD ≈ 0.03 / agreement ≈ 92%; q2_K-class is KLD ≈ 0.33 / 71%. Our ballast domain at 0.579 agreement is far below even the q2_K band, i.e. not a quantization-size perturbation.
- **Pinning:** identical tokenization, identical seqmax/arena, identical chunking, identical sampler and reasoning_effort across every arm; the teacher logits are cached once at the pinned seqmax and never regenerated.

### Gaps
- The within-document cluster inflation factor for token-level KL/agreement in our pipeline is unmeasured (local note §208 says the same).
- No paper reports the token count needed for a given KL MDE on a MoE; llama.cpp does not state its Wikitext-2 token count.

## KQ6. Calibration-overlap control: disjointness and leak detection

### Takeaway
Standard contamination practice is exact n-gram overlap (13-gram GPT-3 index; 7–8-gram for recall, n>8 gives false negatives), which is feasible here because the calibration corpus is fully known; the pruning-specific leak signal is a calibration-vs-held-out KL gap larger than the document-bootstrap CI.

### Cited Findings
- Contamination detection: "a standard pre-training contamination index uses 13-grams, and some labs use 7- or 8-grams for higher recall"; ablation found "values of n larger than 8 lead to false negatives"; paraphrases/translations bypass n-gram filters; Min-k%-prob log-probability methods complement n-grams — [arXiv 2411.03923](https://arxiv.org/abs/2411.03923), [arXiv 2608.29463](https://arxiv.org/abs/2608.29463)
- No retrieved source addresses overlap between pruning calibration sets and evaluation sets — search result (gap).
- Calibration visit-order effects already observed internally (chunks grouped by domain; partial passes make worst-domain objectives degenerate) — [memory: calibration-corpus-visit-order](/home/patrickd/.claude/projects/-home-patrickd/memory/calibration-corpus-visit-order.md)

### Inferences
- **Disjointness construction:** hash every 8-gram (token-level, our tokenizer) of the calibration corpus into a set; drop any held-out document with ≥1 hit for exact-dup protection, and additionally drop documents whose 8-gram hit fraction exceeds 1% to catch near-duplicates **[protocol choice; 8 chosen because n>8 misses]**. Also split by source URL/repo so that different files of one repository are not split across calibration and held-out.
- **Leak detector (per candidate, free):** compute mean KL on (a) 50k tokens of the calibration set and (b) the held-out set, same domain; if KL_calib < KL_heldout by more than the 95% document-bootstrap CI, either the held-out set is contaminated or the mask overfits calibration — both are fail conditions for using that held-out set as a gate. For the benchmark sets (MMLU/PopQA/BFCL/RULER), run the same 8-gram scan of calibration against benchmark prompts.
- Min-k%-prob is not useful here (we need overlap between two known corpora, not membership inference).

### Gaps
- No literature on calibration/eval leakage specific to expert pruning; the thresholds above are protocol choices.

## KQ7. Correlation of cheap metrics with final task quality after pruning

### Takeaway
No paper reports ρ between a teacher-forced metric and downstream accuracy across a population of pruning masks; the available evidence is pairwise: KL↔flips ρ=0.981 (quantization), ESAP > KL > NLL as a search fitness (EvoESAP, comparative), layer-output error does not track accuracy (CAMERA), PPL and accuracy uncorrelated under over-dispersed routing (MESA Table 1), MC accuracy under-detects generative loss by ~4x (Benchmark Illusion), and our own reconstruction proxy predicted nothing.

### Cited Findings
- KL vs flip rate Spearman 0.981 (MMLU, quantized models); accuracy flat while flips large — [arXiv 2407.09141](https://arxiv.org/abs/2407.09141)
- CAMERA appendix: L2/cosine between pruned and original MoE layer outputs "does not always align with downstream task performance"; proposed explanation: the share of tokens with no loss may matter more than best average approximation — [arXiv 2508.02322](https://arxiv.org/abs/2508.02322)
- MESA Table 1 (gpt-oss-20b 25%): WikiText-2 PPL / GSM8K: base 126.02/88.86; MESA 107.14/92.87; REAP 145.04/89.31; RM-Uniform 83.29/83.24; Random 40.95/78.54 — "PPL and task accuracy are uncorrelated in over-dispersed models"; over-dispersion diagnostic H̄ > 0.85·log₂E (gpt-oss-20b 90.8%, Mixtral 74.3%); "Pruning in this regime removes experts and forces the router to redistribute mass to the remaining experts, de-dispersing the routing distribution" — [MESA arXiv 2609.04453](https://arxiv.org/pdf/2609.04453) via [local note expert_pruning_beyond_50.md §226–227](/home/patrickd/glm-5.3-reap/research_notes/Compressing%201T%20MoE%20for%20Thor/expert_pruning_beyond_50.md)
- EvoESAP: ESAP beat KL, NLL and a spec-acceptance proxy as fitness on coding/math (OLMoE); true acceptance as fitness ≈ ESAP downstream — [arXiv 2603.06003](https://arxiv.org/abs/2603.06003)
- Benchmark Illusion: headroom-normalized drop 0.21 (open generation) vs 0.10 (candidate-shown) vs 0.05 (MC) — [arXiv 2606.17609](https://arxiv.org/abs/2606.17609)
- HELMET: synthetic NIAH ≤0.8 Spearman with real long-context tasks; RULER avg <0.85 — [arXiv 2410.02694](https://arxiv.org/abs/2410.02694)
- Internal: +2.5% reconstruction proxy → −0.0001 top-1 agreement (0.13 SE) — [local note calibration_and_evaluation.md §175](/home/patrickd/glm-5.3-reap/research_notes/Compressing%201T%20MoE%20for%20Thor/calibration_and_evaluation.md)
- Internal: FOCUS/RePAIR Table 3: PPL 21.69 vs 22.05 while CREP 7.3% vs 2.23% — PPL is blind to a 3x degeneration difference — [arXiv 2608.26676](https://arxiv.org/abs/2608.26676)

### Inferences
- Sensitivity ordering supported by the evidence: per-token KL ≈ ESAP/acceptance > top-1 agreement/flips > ΔNLL ≫ PPL (uncorrelated under de-dispersed routing) ≫ MC accuracy (recognition-only) — and none of the teacher-forced metrics sees looping (a free-running property) or multi-turn collapse. The protocol therefore needs exactly three independent families: teacher-forced distributional, free-running degeneration, and multi-turn/long-context task tests; no single family substitutes for another.
- The brief's ρ(ppl, GPQA) = −0.10 claim was not found in any retrieved source or local note; the closest published support is MESA Table 1 (five masks, PPL rank order 40.95 < 83.29 < 107.14 < 126.02 < 145.04 vs GSM8K 78.54 < 83.24 < 92.87 > 88.86 < 89.31 — essentially uncorrelated). Treat −0.10 as unverified.
- Build the correlation in-house: every candidate mask the pipeline produces gets tier-1 KL/agreement/ESAP and (when it reaches tier 2/3) task scores; after ~10 masks, compute Spearman of each cheap metric against BFCL-multi-turn/PopQA/code and drop the cheap metrics that do not rank. EvoESAP's evidence says ESAP is the best candidate to keep.

### Gaps
- No population-level ρ(cheap metric, task score) across masks exists in the literature; the EvoESAP "within ~1.5 pts" and CAMERA claims in the brief are comparative, not correlations.

## KQ8. Per-stage gating in a progressive pipeline: tiers, thresholds, and what fails a candidate

### Takeaway
Tier 1 (teacher-forced KL/agreement on ~500k tokens + a 120-prompt greedy loop screen) catches ballast collapse and catastrophic looping; tier 2 adds per-domain KL at 0.5% MDE, a 300-prompt loop test that resolves a doubling, 100 BFCL multi_turn_base items, 1,000 PopQA and 32k RULER-MK; tier 3 is the n=2000 uncapped loop run, full BFCL multi-turn, 64k RULER, 1,600+ MMLU and 0.2% per-domain KL. Thresholds are anchored to the only published tolerances (REAP −5.9% tool-use "acceptable", 0xSero 2x loop rate "problem", Mekala 8-bit <5%, GLM-4.6 GPQA −9 at 40% "unacceptable").

### Cited Findings
- Published tolerances: REAP 50% tool-use mean decrease 5.9% (REAP) / 6.2% (EAN) described as "comparable"; Kimi multi-turn −19 pts is the failure; code "near-lossless" = 0.16%/1.2% mean decrease — [arXiv 2510.13999](https://arxiv.org/abs/2510.13999)
- 0xSero: 2x teacher loop rate shipped only with sampler mitigation (2.3% at rep-penalty 1.10) — [0xSero/GLM-5.2-504B](https://huggingface.co/0xSero/GLM-5.2-504B)
- 8-bit long-context "accuracy drops under 5%" is the robust band — [arXiv 2505.20276](https://arxiv.org/abs/2505.20276)
- GLM-4.6 40%: GPQA-D 78.8→69.7, BFCL 78.4→74.2 — [local note sota_shipped_artifacts.md §134](/home/patrickd/glm-5.3-reap/research_notes/Compressing%201T%20MoE%20for%20Thor/sota_shipped_artifacts.md)
- Miller: report SE, pairwise differences, pairwise SE and correlations; "new evals should contain at least 1,000 questions" — [arXiv 2411.00640](https://arxiv.org/abs/2411.00640)
- Internal: evals must be detached (setsid/systemd), served-model identity asserted in every result, pilot at the ceiling budget — [memory: detach-unattended-local-workloads](/home/patrickd/.claude/projects/-home-patrickd/memory/detach-unattended-local-workloads.md), [memory: eval-orphaned-server-trap](/home/patrickd/.claude/projects/-home-patrickd/memory/eval-orphaned-server-trap.md), [memory: eval-budget-choice-protocol](/home/patrickd/.claude/projects/-home-patrickd/memory/eval-budget-choice-protocol.md)

### Inferences
**Tier 1 — after every stage (target ≤ 1 h on P_mid/D8):**
- T1a. Teacher-forced, paired, pinned: 500k held-out tokens stratified ≥ 50k per domain (7 domains), disjoint from calibration (KQ6). Metrics: mean KL (doc-bootstrap CI), top-1 agreement, confident-flip rate, ESAP acceptance; per-domain. MDE on the pooled 500k ≈ 0.15% agreement (c=1), ≈ 0.28% at c=1.9 **[arithmetic]**.
  Gates: (i) pooled Δagreement vs previous stage not worse than −0.5% at 95% CI (lower CI bound); (ii) **no domain's agreement below 0.90x the best domain's** (our 0.579/0.916 = 0.63 ratio was the collapse; q4_K_M-class noise costs ~8 pts uniformly) **[extrapolated threshold]**; (iii) confident-flip rate ≤ 2x the previous stage's; (iv) calibration-vs-held-out KL gap within CI (leak/overfit check).
- T1b. Loop screen: 120 agentic prompts (fixed IDs), greedy, 1024-token cap, teacher outputs cached. One-sample test vs cached teacher rate (resolves 6%→13% at 80% power with 114 prompts **[arithmetic]**; at 1024 cap it misses late-onset loops). Gate: fail if prompt-level loop rate > 2x teacher's or natural-EOS < 0.90; also fail if SER@4 > 1.5x teacher's (continuous metric, more power).
- Also at tier 1, zero-cost: routing-entropy per layer (MESA H̄ vs 0.85·log₂E) to flag the de-dispersed regime where PPL is meaningless.

**Tier 2 — at every ~2 stages or any tier-1 marginal (≈ 6–8 h on P_mid/D8; ≈ 3 h only if aggregate decode reaches ~100 tok/s):**
- Per-domain KL at 0.5% MDE: ~155k tokens × 7 = 1.1M tokens (c=1.9).
- Loop: 300 prompts, shipping sampler + greedy, 1024 cap (resolves 2x at 3.6% base one-sample with 262).
- BFCL v3 multi_turn_base: 100 items paired (detects the Kimi −19-pt collapse; gate: paired drop > 5 pts fails).
- PopQA 1,000 free-form (0-shot and 1-shot; gate: paired drop > 3 pts fails; report confident-wrong rate).
- RULER MK+MV at 32k, 100 items paired (gate: drop > 5 pts fails; MDE only ~10–12 pts paired — this catches collapse, not drift).
- MMLU by log-prob, 1,600 items stratified by subject (gate: paired drop > 2 pts fails).

**Tier 3 — at milestones (≈ 3–5 days on Thor):**
- Loop n=2000 uncapped (resolves 1.5x), both samplers, per-family breakdown, distinct-4, natural-EOS — the publishable number comparable to 0xSero.
- Per-domain KL at 0.2% MDE (≈ 1M tokens per domain at c=1.9; 6.7M total).
- Full BFCL v3 multi-turn (800), RULER MK+MV+RAG at 32k and 64k (300/length), full MMLU log-prob (14k), PopQA 5k, BBH pooled, plus code (EvalPlus) and MATH-500 as domain checks.

**What fails a candidate (any one):** a tier-1 gate (i)–(iv); loop rate > 2x teacher; multi_turn_base −5 pts; PopQA −3 pts; 32k RULER −5 pts; MMLU −2 pts; or any single domain losing > 2x the pooled loss (ballast rule). Marginal results (inside CI) → promote to the next tier rather than pass.

### Gaps
- All pass/fail thresholds are anchored to 1–2 published data points each and are marked extrapolations; the brief's demand for "literature-justified" thresholds cannot be met more tightly because no paper proposes gates for pruned models.
- The tier-2 "~3 h" target in the brief is not reachable at batch-1 10 tok/s decode with the decode-bound tests included (see KQ9); only the teacher-forced half fits.

## KQ9. Thor throughput assumptions → wall-clock per tier; batching strategy

### Takeaway
Teacher-forced tests are prefill-only and cost N_tokens/P: 500k tokens is 28 min at 300 tok/s but 7 h on the 20 tok/s custom server; free-running tests are decode-bound at 8–12 tok/s batch-1 (0xSero's uncapped n=2000 run is ~2.5M generated tokens ≈ 70 h at batch 1, ≈ 23 h at a 3x batched aggregate), so the loop test must be tiered by n and cap, and the teacher arm must be cached.

### Cited Findings
- User-measured: REAP-50 GLM-5.3 NVFP4 decode 8–12 tok/s batch-1 — [local note thor_serving_performance.md §89](/home/patrickd/glm-5.3-reap/research_notes/Compressing%201T%20MoE%20for%20Thor/thor_serving_performance.md); custom CUDA server decode 8.36→11.86 tok/s, prefill 84.0→51.7 ms/tok at chunk 32 (≈ 19 tok/s), later 47.79 ms/tok — [memory: glm53-nvfp4-dense-overlay](/home/patrickd/.claude/projects/-home-patrickd/memory/glm53-nvfp4-dense-overlay.md); batch forward gives ~1.4–1.5x on that server — [memory: glm53-cuda-server-spec-batch](/home/patrickd/.claude/projects/-home-patrickd/memory/glm53-cuda-server-spec-batch.md)
- Thor llama.cpp b6767: gpt-oss-120b MXFP4 (59 GiB) pp2048 937.8 / tg32 41.82 tok/s; GLM-4.5-Air Q4_K_M 461.9 pp / 20.58 tg — [JetsonHacks Thor bench](https://jetsonhacks.com/wp-content/uploads/2025/10/thor-llamacpp-bench.html)
- Thor vLLM: gpt-oss-120B NVFP4 30.95 tok/s C1 → 97.84 tok/s C8 (3.16x aggregate); Nemotron-3-Super-120B NVFP4 10.4–10.7 tok/s decode — [Thor vLLM MoE thread](https://forums.developer.nvidia.com/t/jetson-agx-thor-vllm-26-02-moe-performance-significantly-below-reference-missing-fused-moe-config/364663), [Nemotron Super on Thor](https://forums.developer.nvidia.com/t/running-nvidia-nemotron-3-super-120b-a12b-nvfp4-on-the-nvidia-jetson-thor/363485)
- DGX Spark (comparable BW class) GLM-5.3-Flash UD-IQ3_XXS/UD-Q2_K_XL llama.cpp: prefill ~270–291 tok/s, decode 15–18 tok/s; Thor TG is 1.26–1.40x slower than Spark — [local note thor_serving_performance.md §75–76, §89](/home/patrickd/glm-5.3-reap/research_notes/Compressing%201T%20MoE%20for%20Thor/thor_serving_performance.md)
- Thor MoE ITL degrades 4.7x with concurrency on one model (Nemotron Nano), so batched aggregate gains are not guaranteed — [Nemotron Nano thread](https://forums.developer.nvidia.com/t/nemotron-3-nano-on-jetson-thor-vllm-itl-degrades-4-7x-with-concurrency-mtp-rejected/370044)
- Never co-run probes beside a vLLM server on Thor (unified-memory OOM, hard reboot) — [memory: thor-unified-memory-oom](/home/patrickd/.claude/projects/-home-patrickd/memory/thor-unified-memory-oom.md)
- 0xSero median output 1267 tokens uncapped — [0xSero/GLM-5.2-504B](https://huggingface.co/0xSero/GLM-5.2-504B)

### Inferences
**Wall-clock arithmetic [all arithmetic; P_low/P_mid/P_high = 20/300/900 tok/s prefill; D1/D8 = 10/30 tok/s decode]:**
- Tier 1a (500k teacher-forced tokens): P_low 6.9 h; P_mid 28 min; P_high 9 min.
- Tier 1b (120 prompts × ~800 mean generated at 1024 cap = 96k tokens; prompts ~2k each = 240k prefill): decode D8 53 min / D1 2.7 h; prefill P_mid 13 min. Tier 1 total ≈ 1.5 h (P_mid+D8), ≈ 3.5 h (P_mid+D1), ≈ 10 h on the custom server.
- Tier 2: KL 1.1M tokens → P_mid 61 min; loop 300 × 800 = 240k gen → D8 2.2 h / D1 6.7 h; BFCL 100 items → ≈ 4–5 h (D8) / 10 h (D1) [token counts extrapolated]; PopQA 1,000 → ≈ 25 min (D8); RULER 32k × 100 = 3.2M prefill → P_mid 3 h / P_high 1 h; MMLU 1,600 × 900 = 1.44M prefill → P_mid 80 min. Tier 2 total ≈ 12–13 h (P_mid+D8), ≈ 22 h (P_mid+D1); with P_high and ~100 tok/s aggregate decode (vLLM-C8-class) ≈ 5 h.
- Tier 3: loop n=2000 uncapped ≈ 2.5M gen → D8 23 h / D1 70 h; KL 6.7M → P_mid 6.2 h; BFCL 800 → ≈ 35 h (D8); RULER 300 × (32k+64k) × 3 tasks = 86M prefill → P_mid 80 h / P_high 27 h (trim to 100/length/task → 29M → P_mid 27 h); MMLU 14k × 900 = 12.6M → P_mid 12 h. Tier 3 ≈ 4–7 days on Thor at P_mid/D8.
- **Batching strategy:** teacher-forced KL is pure prefill — run at the largest chunk the server supports (the custom server's prefill cost "falls with width" via expert gathering, so chunk ≥ 144 where it saturates); run it on the fastest-prefill stack available (llama.cpp/vLLM class, 300–900 tok/s) rather than the 20 tok/s custom server, because the metric only needs logits. Free-running tests should run as 8-way concurrent requests on a vLLM-class server (3.2x aggregate measured on Thor for a 120B NVFP4 MoE), with a scheduler that retires finished prompts early (loops run to the cap; non-loops finish early, so mean generated length ≈ 0.6–0.8 of the cap).
- **Cache everything teacher-side:** FP8 teacher logits for the held-out set (one pass), teacher free-running outputs for the loop prompt set (one pass), teacher BFCL/PopQA/RULER/MMLU results (one pass); thereafter each candidate pays only its own arm.
- The brief's tier-1 "30 min" is achievable for the KL half on a 300 tok/s stack; the loop screen adds ~1 h at D8. The brief's tier-2 "3 h" is not achievable on this hardware with multi-turn included; a realistic tier 2 is 12 h, which fits overnight between pipeline stages.

### Gaps
- No Thor measurement of prefill tok/s for a ~100 GiB NVFP4 MoE on vLLM/llama.cpp exists (the 937.8 is a 59 GiB MXFP4 model; the 270–291 is Spark llama.cpp on a 2–3-bit GGUF; the user's 20 tok/s is the custom server). P_mid is the best-supported assumption and is marked extrapolation.
- No published per-item token counts for BFCL multi-turn or PopQA on these models; tier-2/3 decode costs are estimates to ±2x.
- Batched-decode aggregate on Thor for a 100 GiB model is unmeasured; 3.2x is from a 59 GiB model.
