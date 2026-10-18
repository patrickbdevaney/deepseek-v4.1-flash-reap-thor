# Coordinator state — "Compressing 1T MoE for Thor"

Written 2026-10-17 14:41 EDT so a fresh session can resume if this one is cut off.
Deep-research skill: `/home/patrickd/.claude/skills/synced/2947dca4-7576-4f23-8369-54962a7f83d6_1b5d3611-405e-48d8-b622-d5265328718f/deep-research`
Notes dir: this folder. Report target: `/home/patrickd/glm-5.3-reap/reports/Compressing 1T MoE for Thor.md`

## The question (pass verbatim to the report writer)

What is the highest-performing, minimally-lossy method to compress a 763B–1T open-weight MoE LLM
(DeepSeek-V4.1-Flash-1017, MiMo-V2.6-Pro-MOPD; GLM-5.3-753B added as a third candidate) to run with
maximal intelligence on one Jetson AGX Thor (122 GiB unified, 117 GiB envelope, 12 GiB runtime
reserve, sm_110a), calibrated for general-text ballast + code/math/agentic/finance/science/vision —
under the premise that expert pruning can be pushed PAST the proven 50%. Deliverable: ranked recipes
per target with memory arithmetic ≤117 GiB; per-lever ceilings with citations; proven vs
extrapolated; a concrete progressive REAP→HEAL pipeline for one Thor; honest showstoppers.
EXTENSION (user, 14:40): also cover reducing bytes moved per decode step (active experts, bpw,
expert skipping/caching, activation sparsity) + spec decode + CUDA kernel optimizations for decode
speed on Thor — the goal is frontier intelligence at the 128 GB level AND fast decode.

## Hard constraints the writer must respect
- Thor host == device: ONE 122 GiB pool. Pinned-host offload frees nothing. Engram must be NVMe
  O_DIRECT row store (measured 1.13–1.30 ms/tok on this box) or V4.1-Flash is a no-go.
- On FP4/MXFP4-source experts only BIT-EXACT ops are free: deletion, block-aligned width
  slimming, Engram offload. Merging/SVD/requant pay double-quant.
- `~/leafllm` and anything outside `glm-5.3-reap` must never be touched (user rule).
- A vLLM server was holding ~106 GiB on the box during research; no memory probes.

## Seeds (28) and status at 14:41

| # | file | status | one-line result |
|---|---|---|---|
| 1 | target_models.md | DONE | V4.1-1017: 475 GiB, experts MXFP4 269 GiB, Engram 189 GiB not removable, floor w/o Engram 9.77 GiB, KV 890 B/tok. MiMo-Pro: 527 GiB, experts 495 GiB QAT-MXFP4, KV 51,200 B/tok, no shared expert |
| 2 | expert_pruning_beyond_50.md | DONE | No generalist survives >50% within 10%; all >50% are single-domain specialists; fine-grained models lose 14–22 pt MC at 50% |
| 3 | expert_merging_lowrank.md | DONE | REAP dropping beats all merges at 50%; MoBE only factorization at 1T (24–30%, needs training, no kernel) |
| 4 | healing_recovery.md | DONE | ExFold claim later CORRECTED (see #16); progressive re-rank Mixtral 14.74→10.44 no FT; LoRA heal feasible for 96 GiB student only; CPT infeasible |
| 5 | extreme_quantization.md | DONE | Expert cliff 2.9→2.4 bpw; non-expert of 1T ~11.7B params; heal order P→KD→Q (5.05 vs 24–53); llama.cpp only proven 2–3-bit path on Thor |
| 6 | structural_and_memory_budget.md | DONE | Small levers <9 GB; V4.1 fits only Engram off-pool + ~64% @4.25bpw (or ~51% @3); MiMo-Pro needs 83–85%; reserve 12 GiB; layer-drop kills reasoning |
| 7 | calibration_and_evaluation.md | DONE | Coverage not tokens; equal-by-domain + per-domain protection; general needs ~25–30%; KL > top-1 > ΔNLL ≫ benchmarks; 0.5% top-1 ≈ 43k tok |
| 8 | sota_shipped_artifacts.md | DONE | Nothing ≥500B ships ≤128 GB with evals; best ≤128 GiB is Qwen3.5-397B 2.34 bpw (+12.3% PPL); highest shipped prune ≥400B = 50%; looping + BFCL collapse documented |
| 9 | thor_serving_performance.md | DONE | Sub-3-bit buys fit not speed (decode flat 16–20 tok/s GB10); Thor 1.3× slower; IQ1–3 wrong on Blackwell until unmerged PR #27902; no 1T resident on 128 GB |
| 10 | iterative_prune_distill.md | DONE | Progressive > one-shot at fixed ratio, NOT ratio-extension; SlimMoE width-slim 63.8 vs delete 53.4 (needs KD); Thor 300–3000× short of distill budgets |
| 11 | expert_redundancy_mask_search.md | DONE | Rarely-routed ≠ dead (Frequency collapses); HOPE at cheap-search frontier; proxies don't transfer; super experts exist |
| 12 | long_context_routing_drift.md | DONE | Nobody measured drift vs position; Super Experts (2507.23279) catastrophic to prune; attention-side precision is the long-ctx cliff |
| 13 | prune_quant_pareto_frontier.md | DONE | CORRECTION: W2+50% "collapse" was unpruned Qwen3-30B naive W2, not Kimi-K2; grid EMPTY where plan lives; lowest-risk 50% @ 3.0–3.3 dynamic; quantize to 3 bits before pruning |
| 14 | ream_post_reap.md | DONE | REAM loses to REAP by 13 pt at 48% on FP4 experts (double-quant); CLOSED as lever |
| 15 | domain_protection_schemes.md | DONE | GEC gain confounded by corpus, MMLU lower than REAP; protection composes with HOPE; protect_frac=0.04 ≈ 35–42% of mask at 50%, 60–70% at 70% |
| 16 | exfold_routing_recovery.md | DONE | CORRECTION: ExFold is serving-accel, not healing; "61.5→68.8" compares static vs DYNAMIC pool; routing-level heal ceiling ~52% of gap; renorm rule alone swings MMLU 4.3 |
| 17 | calibration_mixture_dose_response.md | DONE | Knee 17–25% share; balance value rises with ratio (+4.6@30%→+18.8@60%); 0% general = −44 MMLU at 50%; rec: gen20/agent15/code15/vis14/math12/sci12/fin10, floor 10% & ≥250k tok |
| 18 | progressive_schedule_design.md | DONE | STRONGEST SUPPORT FOR PREMISE: Mixtral iterative re-rank moves cliff ~25 pts right (62.5% iter beats 50% one-shot); re-rank alone = 86% of gain; 70% = 3 stages 45→30→22, 13.5–19.5 h Thor; 8-expert model only |
| 19 | mesa_mosaic_selection.md | DONE | MESA zero-GPU from gate_sum but degenerates when domains correlate, Mixtral catastrophic; Mosaic needs hidden states, repo empty |
| 20 | never_prune_guards.md | DONE | G1–G7 checklist; Super Experts need per-expert MAX down_proj norm (one hooked pass); never prune shared expert; loop gate n≥2000 |
| 21 | expert_width_slimming.md | DONE | SlimMoE gap needs KD; training-free slim beats delete on ≤80B (TENP 59.06 vs 48.02 @60%); down_proj block-aligned (256 for K-quant); renorm at trained k cuts top-k loss 4.65→0.35 |
| 22 | mimo_pro_go_nogo.md | DONE | NO-GO every regime (82–84% removal; 1.5 tok/s); Flash REAP-50 GO at 1M FP8 KV (+9.5 GiB); Flash REAP-50 was PUBLISHED NEVER SERVED |
| 23 | engram_off_device_thor.md | DONE | VIABLE off-pool: measured 1.13–1.30 ms/tok on this Thor via O_DIRECT; mmap WRONG; pinned-host frees 0; resident = 1.7 GiB left; NVMe has only 51 GB free |
| 24 | specialist_strategy.md | DONE | multi-domain cost near-free to ~60% (1.0@50, 2.8@62.5, 13.4@75); ballast = factual long-tail not fluency; 4-domain @60–65%: −3–6 gen, −10–20 MC; BFCL multi-turn is the gate |
| 25 | glm53_753b_go_nogo.md | DONE | (b)/(c): FP8 source (quant lever intact) but REAP-50 needs 1.96 bpw experts; 67.3% @3.0 / 60.8% @2.5; unpruned 2.30 bpw = +72% PPL; FAILS decode gate (~5 tok/s real, attention 12.9B dominates); disk needs ≥1.1 TB vs 51 GB free; prior "REAP-50 @ 2.2 bit" was 7 pp optimistic (no KV term) |
| 26 | router_post_prune_rules.md | DONE | UPSTREAM REAP hard-codes softmax + unbiased topk = WRONG for sigmoid/noaux_tc (GLM/DeepSeek/Kimi); k2 rule (renorm over original-pool mass, k2=k_train) −4.65→−0.35 MMLU, needs masked rows; Router-KD no-op on REAP masks |
| 27 | eval_protocol_pruned_moe.md | DONE | 3-way loop detector (CREP≥0.30 / ≥3-tok×20 / cap-no-EOS), n≈2,080/arm; BFCL multi_turn_base 200 items is THE collapse diagnostic; Tier1≈1.5h, Tier2≈12h (not 3h), Tier3≈4–7d; 0.5% top-1 = 43k tok ×1.2–9.3 cluster; ρ(ppl,GPQA)=−0.10 NOT found — use MESA T1 |
| 28 | per_expert_mixed_precision.md | DONE | CORRECTION: "+20–25 pt" was vs uniform 2.0-bit GPTQ; matched-bits gain 0.3–13 pt tracking routing skew; GGUF cannot express, vLLM MIS-LOADS, our CUDA server can (warp-uniform tag); saves memory not tok/s (bytes ∝ routing-weighted width); value on Thor = fit only; post-REAP-50 gain unmeasured |
| 29 | decode_bytes_per_step.md | DONE | BIGGEST LEVER = close 116→200 GB/s effective-BW gap on our CUDA server (row-reuse at low bpw) — worth more than all expert-side levers; top-k 8→6 w/ k2=16 denominator = +0.75 MMLU free; locality non-lever for resident weights; MTP bound on Thor 1.13× @95% (the "+34%" was a Spark number, not local); GLM-5-shape 744B REAP-50 needs 2.2 bpw to fit (3 bpw = 136 GB) |
| 30 | spec_decode_kernels_thor.md | DONE | 1.27× ceiling is NOT the ceiling: K=2 verify 40% above union bound (KDA non-amortised) → lever = fused m-token attention kernel, not drafter; union ceiling 1.56–1.88×; 753B (no KDA) → ~1.3× MTP / ~1.9× DFlash2 [extrap]; sm_110a: BF16 tcgen05 YES, block-scaled NVFP4 tcgen05 NO ("870 TFLOPS FP4" unreachable); BW residue = tall-thin gemv + instr-bound dequant |
| 31 | v41_architecture_pruning_delta.md | DONE | mHC delta ≈ 0 (GLM-5.3-Flash already hc_mult=4, LOCAL code handles it); no HF transformers class → use checkpoint inference/model.py; Engram applied BEFORE block, must be in sweep; 8k fits w/ ~70 GiB headroom (row-chunk indexer); vLLM fused router legal counts {128,192,256,320,384} → 51% target = 192 = 50% exactly; 1017 Engram tables changed → recompute rankings |
| 32 | vision_retention_pruning.md | DONE | V4.1 has `gate.bias_vl` (corr −0.05 w/ text bias) → text-only ranking WORSE than random for vision (+15.9% vs +7.2% caption PPL); fix = per-modality-normalised ranking; shipped GLM REAP-50 has NO vision eval; ChartQA GPL confirmed; the 15.85% vision share is a LOCAL accumulator measurement (this session), not wiki |
| 33 | layer_streamed_healing_infra.md | DONE | REVERSES "not feasible": BLOCK-WISE healing fits a weekend (2M tok ≈ 3–4 h, 8M ≈ 9–14 h, ~75 GB resident); NVMe paid once/layer; no FP4 backward needed (dgrad on dequant BF16, NeMo MXFP4GroupedMM pattern); hidden-state caches 640 GB don't fit → co-stream teacher layer; FIRST EXPERIMENT = 1-layer dequant+grouped-GEMM fwd/dgrad microbench on Thor |

Seeding STOPPED at 14:47 (33 total). ALL 33 LANDED by 14:58 — no flush needed. Report writer spawned on Sonnet at 14:58; timer fired 15:00. Next: deliver reports/Compressing 1T MoE for Thor.md via SendUserFile.

## Resume procedure (if cut off)
1. `ls` this folder; any seed above marked IN FLIGHT/SPAWNING whose .md is missing was lost —
   re-spawn with the same objective (prompts are summarised in the table; the full prompts are in
   the session transcript if available).
2. Then spawn the report writer per the skill (step 6) with the question above and
   `Earlier research: none`. It must read `.../deep-research/references/report-writer.md` first.
3. Deliver via SendUserFile and summarise in 3–5 sentences.

## Two first-wave findings that were overturned by follow-ups (writer must use the corrected ones)
- "2-bit + 50% collapses on Kimi-K2" → was unpruned Qwen3-30B naive W2 (file 13 supersedes file 3).
- "ExFold heals REAP-50 to near-lossless" → ExFold is dynamic serving, not healing (file 16 supersedes file 4).
