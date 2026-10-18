# Reducing bytes moved per decode step for a ~100 GiB REAP-50 1T-class MoE on Jetson AGX Thor

Scope note. Every number carries its model / device / quant / source. `LOCAL` = measured on this Thor and recorded in `~/.claude/projects/-home-patrickd/memory/*.md` or `~/glm-5.3-reap/wiki/`; `[comparable device]` = DGX Spark GB10 or other non-Thor hardware; `[EST]` = arithmetic I did here, not a measurement; `[VEN]` = vendor-reported. The budget model in the last section is parametric so it can be re-run against real safetensors headers (`tools/roofline.py` in `~/glm-5.3-flash-cuda-server`, which the local notes say "reads safetensors headers, nothing is estimated").

Three local facts frame everything below (all `LOCAL`):
- GLM-5.3-Flash REAP-50 NVFP4 (45 layers = 34 KDA + 11 MLA, 144 experts top-8): `B_tok` = 19.76 GB before, **9.762 GB after** NVFP4-ing the dense weights (−50.6%); split before the overlay was routed experts 24.1%, KDA 47.4%, MLA 13.1%, lm_head 6.4% — [LOCAL memory glm53-cuda-server-roofline.md, glm53-nvfp4-dense-overlay.md]. Overlay quality: PPL 4.430 → 4.500 (+1.56%), top-1 agreement 90.1%, confident flips (ref NLL < 0.5) 0.138% — [LOCAL glm53-nvfp4-dense-overlay.md].
- Decode after overlay: 11.86 → 11.91 tok/s, i.e. **~116 GB/s effective** (9.762 GB × 11.91/s); the kernels individually stream at 190–200 GB/s against a measured 235–247 GB/s streaming ceiling on this Thor; wall exceeds the dprof kernel sum by only 0.4%, so CUDA graphs are "not a lever here" — [LOCAL glm53-decode-gap-profile.md, glm53-kda-launch-bound.md].
- The binding term after NVFP4 is activation traffic, not weight bytes: "a gemv reads 4 bytes of activation per weight ... 7.1 per byte of NVFP4"; halving `B_tok` first bought only 1.4% until each block reused x across R=5 output rows — [LOCAL glm53-activation-traffic-finding.md]. Any "bytes removed → tok/s" projection below is therefore an upper bound unless the kernel reaches the streaming wall.

---

## Q1. Top-k reduction at inference (8→6→4→2), renormalization-denominator correction, sigmoid vs softmax routers

### Takeaway
The −0.35 vs −4.65 MMLU figure is real and comes from arXiv 2609.04575 (Qwen3.6-35B-A3B, 8→4 experts): the damage from lowering k is mostly the *gain* change caused by renormalizing over fewer experts, and anchoring the denominator to a larger reference set (k2 = 2k) recovers it to within noise; it replicates on Qwen3.5-397B (10→5: −0.55 pp, n.s.). No equivalent table exists for DeepSeek-V3/R1, GLM-5 or Mixtral, and nothing exists for a REAP-pruned model; the sigmoid/noaux_tc case is covered only by a vendor page (DeepSeek-V4-Flash 6→4) with no methodology.

### Cited Findings
- **Formula and gain.** For a router with trained k, select top-k1 experts but divide their weights by the probability mass of the top-k2 set: k2 = k1 is standard renormalization, k2 = E (all experts) removes renormalization, k2 = k anchors the branch gain to the trained value. Top-k router mass on WikiText is only 0.182 (Qwen3.6-35B) / 0.192 (Qwen3.5-397B), so renormalization is a 5.50× / 5.22× gain — [arXiv 2609.04575](https://arxiv.org/html/2609.04575)
- **Qwen3.6-35B-A3B (256 experts, trained k=8), MMLU 5-shot n=2000, native 81.65%:** (k1,k2)=(6,6) 79.05 (−2.60); (6,8) 80.55 (−1.10); (6,16) 82.40 (+0.75); (6,256) 54.20 (−27.45); (4,4) 77.00 (−4.65); (4,8) 78.55 (−3.10); **(4,16) 81.30 (−0.35, p=0.66, not significant)** — [arXiv 2609.04575](https://arxiv.org/html/2609.04575)
- **Same model, GSM8K 0-shot n=500, native 95.0%:** (6,6) 92.8 (−2.2); (6,8) 95.0 (0.0); (4,4) 88.4 (−6.6); (4,16) 94.4 (−0.6) — [arXiv 2609.04575](https://arxiv.org/html/2609.04575)
- **Qwen3.5-397B-A17B (512 experts, trained k=10), MMLU native 89.45%:** (8,8) 89.60 (+0.15); (8,10) 89.00 (−0.45); (8,20) 88.45 (−1.00); (5,5) 87.35 (−2.10); **(5,10) 88.90 (−0.55, p=0.24, n.s.)**; (5,20) 87.75 (−1.70); (5,512) 70.90 (−18.55). GSM8K native 96.6: (8,8) 94.6; (8,10) 96.2; (5,5) 92.4; (5,10) 95.8 — [arXiv 2609.04575](https://arxiv.org/html/2609.04575)
- **Perplexity disagrees with accuracy on which k2 is best:** Qwen3.6-35B WikiText native 6.643: (6,6) +4.08%, (6,8) −1.61%, (6,16) +1.81%, (6,32) +23.25%, (6,256) +356%; Qwen3.5-397B native 4.159: (8,8) +1.06%, (8,10) +0.46%, (8,20) +4.98%, (8,40) +17.18%, (8,512) +128%. The authors warn perplexity is minimized at k2=8 while MMLU peaks at k2=16 on the 35B model, so "unlabeled text alone is not a reliable guide"; they recommend scanning k2 ∈ {k1, k, 2k} with paired significance tests. No throughput was measured; only active-expert parameter counts (1.132B → 0.629B for the 35B model) — [arXiv 2609.04575](https://arxiv.org/html/2609.04575)
- **Contrary evidence in a training setting:** a self-distillation paper on Qwen3-30B-A3B finds that renormalizing after removing experts "leads to a consistent accuracy drop compared with the default formulation", attributed to renormalization inflating the expert-branch scale — same mechanism, opposite recommendation because it zeroes experts during SFT rather than reducing k at inference — [arXiv 2605.18643](https://arxiv.org/abs/2605.18643) (via search summary; full table not fetched)
- **Uniform reduction without correction, DeepSeek-V2-Lite (64 experts, k=6) and DeepSeek-V3 (256, k=8, sigmoid):** V3 average over ARC-C/ARC-E/BoolQ/OBQA/RTE/WinoGrande by (first, mid, last) active-expert schedule: (8,8,8) 73.0; (8,6,6) 73.1; (6,6,6) 72.9; (6,8,8) 73.3; (4,4,4) 70.6; (4,2,2) 66.0; (2,2,2) 60.8. V2-Lite baseline 66.0: uniform k=2 60.4; an average of ~3.3 active experts keeps the drop within 1%. Throughput on V2-Lite (sglang 0.4.4, A800): k=6→2 gives +31.7% at concurrency 2, +50.7% at 4, +9.6% at 128. The paper does not state how weights are renormalized — [arXiv 2505.03531](https://arxiv.org/html/2505.03531)
- **DeepSeek-V3 (sigmoid/noaux_tc) at 8→6 in a compression pipeline:** Collaborative Compression prunes 25% of experts and sets top-k 8→6 before IQ1_M-based mixed quantization; it reports only combined numbers (130 GB mixed model MMLU 84.55 / GSM8K 93.56 vs Q2_K_L 230 GB 85.57 / 93.4) and no top-k-only ablation — [arXiv 2509.25689](https://arxiv.org/html/2509.25689v1)
- **DeepSeek-V4-Flash-4E (noaux_tc, 284B, k 6→4, no retraining) [VEN, single B300, vLLM 0.23, FP4+FP8]:** HumanEval 157/164 both; MMLU-Pro 41.46% vs 37.77% (+3.69 pp, with per-category swings of −9.45 to +17.36 pp); HumanEval generation 11–13% faster, MMLU-Pro 8–9% faster. The page says nothing about renormalization — [Friendli DeepSeek-V4-Flash-4E](https://friendli.ai/models/autotrust/DeepSeek-V4-Flash-4E). I treat the +3.69 pp as evaluation variance, not a real gain; the ±10–17 pp category swings are far outside a 4-expert ablation's expected effect.
- **Qwen3-30B-A3B 4-bit on M4 Max (MLX, k=8 trained) [comparable device]:** decode 126.5 tok/s at k=8 → 136.1 (k=6, +7.6%) → 140.3 (k=5, +10.9%) → 147.3 (k=4, +16.5%); 3-bit k=8 138.6, 3-bit k=6 147.1 ("quality divergence measurable"), 3-bit k=4 157.3 ("output quality breaks"). Quality cells are "TBD"; first 16 greedy tokens matched baseline at 4-bit k=4. No renormalization change — [vllm-mlx moe-top-k guide](https://vllm-mlx.is-a.dev/guides/moe-top-k/)
- **Why k=4 on a 3-bit checkpoint broke but k=4 on 4-bit did not:** the guide attributes it to compounding error "past the point where argmax is stable" and recommends at most one aggressive knob — [vllm-mlx moe-top-k guide](https://vllm-mlx.is-a.dev/guides/moe-top-k/)
- **Layer-adaptive k (LExI):** data-free, assigns per-layer active-expert counts from weights alone; on Qwen1.5-MoE ~10% better accuracy than expert pruning at the same vLLM/H100 throughput; the abstract claims pruning "mainly reduces memory without significantly improving GPU inference performance" — [arXiv 2509.02753](https://arxiv.org/abs/2509.02753)
- **Budget-aware allocation (Alloc-MoE):** layer-level knapsack + token-level redistribution; at half the original expert budget on DeepSeek-V2-Lite, 1.15× prefill, 1.34× decode; preserves expert-load ranking (Spearman > 0.93); background claim that 6→3 experts on an unnamed model costs >17% — [Alloc-MoE arXiv 2604.08133](https://www.alphaxiv.org/abs/2604.08133)
- **Fixed-k models are brittle to k changes unless trained for it:** Matryoshka MoE reports "increasing the number of activated experts yields minimal performance gains, while reducing it leads to sharp performance degradation" and trains with variable k to fix it — [arXiv 2509.26520](https://arxiv.org/html/2509.26520v1)
- **Sigmoid routers normalize over the selected set too:** DeepSeek-V3's noaux_tc uses the bias only for selection while the gate value is the unbiased sigmoid affinity normalized over the selected top-k — [LOCAL memory noaux-tc-router-formula.md]; [DeepSeek-V3 report](https://arxiv.org/pdf/2412.19437). So the k2 denominator trick applies unchanged (divide by the top-k_trained sigmoid mass), but no paper has measured it on a sigmoid router.

### Inferences
- [EST] For a top-8 model, k=4 with k2=16 cuts routed-expert bytes by exactly 50%; k=6 by 25%. On the local GLM-5.3-Flash budget (routed = 24.1% of 19.76 GB = 4.76 GB; after the overlay routed ≈ 4.76 of 9.76 GB = 49%) that is −2.38 GB/token (k=4) or −1.19 GB (k=6) off 9.76 GB, i.e. up to −24% `B_tok`. On a 1T-class model the routed share is larger (see budget model), so the lever is worth more there.
- The Qwen3.6 result is on an *unpruned* 256-expert router whose top-8 mass is only 0.18. After REAP-50 the surviving experts carry more mass each (half the competitors are gone), so the trained-k anchor (k2 = k) changes meaning; the right k2 for a REAP'd model must be re-scanned, not copied. The renormalization gain (5.5×) is the quantity to re-measure first.
- The 2505.03531 schedules with no quality loss on V3 are all "keep 8 in the first layers, drop to 6 later" ((8,6,6) = 73.1 vs 73.0 baseline); the (6,8,8) 73.3 row says the opposite direction also works on V3 ("less smooth, best strategy is ascending"). That instability across schedules on 6 small benchmarks means the V3 numbers are within noise of each other; they establish only that uniform k=6 is near-lossless and k=4 costs ~2.4 points on that suite without the denominator fix.

### Gaps
- No k=8/6/4/2 table for DeepSeek-V3/R1, GLM-5, Kimi, or Mixtral with the k2 correction; the correction paper covers only Qwen3.6-35B and Qwen3.5-397B (plus a Gemma appendix scan at k1=6).
- No k=2 result anywhere with the correction; without it k=2 costs 5–7 points on V2-Lite/V3 suites.
- No measurement of k-reduction on a REAP/HOPE-pruned model; interaction with the pruned router mass is unmeasured.
- No sigmoid-router (noaux_tc) measurement with controlled renormalization; the Friendli 6→4 page is the only data point and reports no method.
- No decode-speed measurement on Thor for k reduction; the MLX +16.5% at k 8→4 is on an M4 Max with a 3B-active model where attention is a large fraction of bytes.

---

## Q2. Dynamic / confidence-based expert skipping (2505.03531 and successors): rule, bytes saved, quality, interaction with pruning

### Takeaway
2505.03531 is a *static layer-wise schedule*, not a per-token confidence rule; the per-token methods (EAC-MoE PESF, MC-MoE ODP, XShare dynamic skipping, elbow routing, ZEDA) save 5–15% of expert reads at ~0.3–0.5-point cost when training-free, and ~50% only with self-distillation (ZEDA, 1.2× end-to-end). The strongest published signal about training-free dynamic skipping is ZEDA beating "the strongest dynamic MoE baseline" by 4–6 points at the same 50% budget, i.e. training-free per-token skipping at 50% is *not* quality-neutral. Nothing measures skipping on a pruned model.

### Cited Findings
- **2505.03531's rule** is a four-tuple (b, h, e, p): active experts b at layer 1, h at layer p, e at the last layer, linearly interpolated; no threshold, no per-token decision; claims "at least 10%" throughput gain on DeepSeek-V3 with no degradation (sglang 0.4.4, 8× H200) — [arXiv 2505.03531](https://arxiv.org/html/2505.03531)
- **EAC-MoE PESF** (skip experts whose routing score < α × mean selected score, α = 0.3 or 0.7) applied only in prefill; Mixtral-8x7B 0-shot avg FP 72.64 → QESC 3.03-bit 72.21 → +PESF α=0.3 71.68 (−0.53 from quant-only) — [EAC-MoE arXiv 2508.01625](https://arxiv.org/html/2508.01625v1) (numbers as compiled in the sibling note `prune_quant_pareto_frontier.md`)
- **MC-MoE ODP** dynamically prunes 14.9–15.8% of activated expert parameters: Mixtral-8x7B PMQ 2.54-bit 67.50 → +ODP 66.94 (−0.56); 8x22B PMQ 2.54 72.08 → +ODP 71.21 (−0.87) — [MC-MoE arXiv 2410.06270](https://arxiv.org/html/2410.06270)
- **Elbow-based routing** picks k per token from the sorted router probabilities (the "elbow"); on OLMoE 5.3% average latency reduction with accuracy preserved on six benchmarks incl. MMLU — [arXiv 2608.04401](https://arxiv.org/pdf/2608.04401) (search summary)
- **XShare dynamic skipping** drops lower-ranked experts when the gate score falls sharply; framed for batched decoding where "memory bandwidth bottleneck dominates" — [XShare arXiv 2602.07265](https://arxiv.org/html/2602.07265v1) (search summary)
- **ZEDA (self-distillation, Qwen3-30B-A3B and GLM-4.7-Flash):** parameter-free zero-output experts let the router skip real experts; eliminates >50% of expert FLOPs, ~1.20× end-to-end, "marginal accuracy loss" on 11 math/code/IF benchmarks, and **beats the strongest dynamic-MoE baseline by 6.1 and 4.0 points** on the two models — [arXiv 2605.18643](https://arxiv.org/abs/2605.18643)
- **Mixtral token-expert dropping (ICML 2026):** threshold-based dropping plus post-training, 1.19× MoE-layer speedup with +0.5% accuracy on Mixtral-8x7B; inference-only on OLMoE-Instruct / DeepSeek-V2-Lite-Chat up to 1.41× MoE speedup at <0.5% loss — [PMLR v306 cai26i](https://proceedings.mlr.press/v306/cai26i.html)
- **Whole-expert bytes are what matter:** on one MI300X, Qwen3-30B-A3B latency rose 4.7× going from 16 to 128 distinct active experts at batch 64; "decode latency is governed by the number of distinct experts activated" — [ELDR arXiv 2607.00466](https://arxiv.org/pdf/2607.00466) (search summary)

### Inferences
- [EST] Training-free per-token skipping buys 5–15% of routed bytes (≈ −0.25 to −0.7 GB/token on the local 9.76 GB budget, since routed ≈ 4.76 GB) at a measured 0.3–0.9-point cost on small suites; the static (k, 2k) denominator trick from Q1 buys 25–50% at ≤0.6 points. The static method dominates on this metric, and it is simpler to put in a CUDA gemv path (fixed k, no data-dependent launch shapes).
- After REAP-50 the router's second-choice experts are gone, so the gap between selected and dropped scores widens; a threshold rule tuned on the unpruned model would skip *fewer* experts, not more. Direction only; unmeasured.
- Dynamic skipping changes the number of expert gemvs per token per layer, which on the local server is already the kernel that scales worst with width (`ffn:moe` 0.82 at width 4) — variable k is friendlier to the batch-1 gemv than to any batched verify.

### Gaps
- No paper measures dynamic skipping on a statically pruned (REAP/HOPE) model.
- No bytes-per-token or bandwidth accounting in any dynamic-skipping paper; all report FLOPs or wall-clock.
- 2505.03531's DeepSeek-V3 throughput table is mis-aligned in the HTML (4 values under 5 headers); the "≥10%" claim could not be tied to a specific k.

---

## Q3. The IQ2-vs-IQ3 speed puzzle on GB10/Thor: where the time goes when bytes shrink

### Takeaway
On DGX Spark llama.cpp, DeepSeek-V4-Flash-0731 decodes at 17.35 tok/s (UD-IQ2_M, 84.7 GiB) vs 16.56 (UD-IQ3_XXS, 97.1 GiB): a two-point fit gives a fixed per-step term of ~37 ms (≈63% of the step) and a bandwidth-proportional term running at ~200 GB/s [EST]; the residual is launch/attention/dequant overhead, not DRAM. The Thor CUDA server is in a different regime (fixed overhead 0.4%, kernels at 190–200 GB/s), where bytes do convert — but only once activation traffic is fixed.

### Cited Findings
- **[Spark]** DSV4-Flash-0731 (284B/13B active), llama.cpp b10216, CUDA 13.0: UD-IQ2_M 17.35 / 17.05 / 16.05 tok/s at ctx 2k/8k/32k vs UD-IQ3_XXS 16.56 / 16.21 / 15.49; "2-bit was about 3.6–5.2% faster than 3-bit for decode. The file size ratio (0.872) would predict about 1.15×"; prefill 448 vs 411 tok/s — [Classmethod DSV4 llama.cpp on Spark](https://dev.classmethod.jp/en/articles/dgx-spark-deepseek-v4-flash-0731-llama-cpp/)
- **[Spark]** Same model, build 10235, FA on: UD-IQ2_M tg128 19.09 (d0) → 18.90 (d4096) → 18.15 (d16384); 1→2→3-bit curve "flattens at ~16–19 tok/s", i.e. 55–65 ms/token — [NVIDIA forum 1× Spark DSV4](https://forums.developer.nvidia.com/t/1x-dgx-spark-deepseek-v4-flash-0731-on-llama-cpp-131k-ctx-slot-19-7-tok-s-single-stream-52-tok-s-4-concurrent/379129); synthesis in sibling note `thor_serving_performance.md` (LOCAL)
- **[GB10 kernel data]** llama.cpp PR "CUDA: faster MMVQ for low-bit types", merged 2026-10-17: batch-1 kernel speedups on GB10 of 1.08× q2_K, 1.15× iq2_s, 1.11× iq3_s, 1.00× iq1_s/iq1_m, 1.00× q4_K/q8_0, and no change on RTX PRO 6000/5090 — low-bit dequant was measurably compute-limited on GB10 before the PR — [llama.cpp PR #30517](https://github.com/ggml-org/llama.cpp/pull/30517)
- **[Thor, LOCAL]** GLM-5.3-Flash CUDA server: after the MoE kernel rewrite (warp-per-row, hardware `__nv_cvt_fp4x2_to_halfraw2` unpack replacing a `__constant__` nibble LUT that serialized up to 8 ways), `ffn:moe` went 4038 → 857 ms (4.71×) and from 13% to 82% of achievable bandwidth; every remaining phase runs at 190–200 GB/s against a 235–247 GB/s streaming measurement; an inline `fp8e4m3()` scale decode instead of a LUT was a 19% *regression* ("the kernel is instruction-bound") — [LOCAL glm53-decode-gap-profile.md]
- **[Thor, LOCAL]** The 0731 (DSV4-Flash) CUDA engine got 1.17× from a CUDA graph; the GLM-5.3 engine got nothing because wall exceeds the kernel sum by 0.4% — [LOCAL glm53-decode-gap-profile.md]. `kda:gates` ran at 16.3 GB/s (7% of achievable) because it issued 238 launches/token to move 2.7 MB; fusing to one launch gave +4.8% total, and the fused projections reached 118.7 GB/s, not the 160 projected: "launch overhead and kernel efficiency are SEPARATE terms" — [LOCAL glm53-kda-launch-bound.md]
- **[Thor, LOCAL]** NVFP4 gemv activation traffic: "7.1 bytes of x per byte of NVFP4"; stubbing the FP4 unpack changed nothing, stubbing the x reads gave 19×; fix was R=5 row reuse (gemv bandwidth 2→115, 4→150, 5→161, 6→135 GB/s); a wider uint4 load was 3.4× *slower* than uint32 — [LOCAL glm53-activation-traffic-finding.md]
- **[Thor, LOCAL, DSV4-Flash engine]** before kernel work the engine ran at ~25% of peak bandwidth (8.77 GB/token active, 126.7 ms/step); hardware facts: `cp.async` OK, `__nv_cvt_fp4x2_to_halfraw2` OK, `tcgen05` assembles for sm_110a but `mma.sync...kind::mxf4.block_scale` needs sm_120a, so RTX-50/Spark FP4 MMA kernels do not transfer — [LOCAL dspark-decode-gap-research.md]
- No llama-bench IQ2 vs IQ3 vs Q4 run exists on Thor hardware; no per-kernel nsys/ncu profile of a Spark/Thor llama.cpp decode step was found; ncu on this Thor was blocked by `ERR_NVGPUCTRPERM` — [sibling note thor_serving_performance.md, LOCAL dspark-decode-gap-research.md]

### Inferences
- [EST, two-point fit] DSV4-Flash active ≈ 13B params. At UD-IQ2_M (84.7 GiB / 284B = 2.56 bpw average) active bytes ≈ 13 × 0.32 = 4.2 GB; at UD-IQ3_XXS (2.93 bpw) ≈ 4.76 GB. Step times 57.6 and 60.4 ms. Solving t = F + B/BW: BW ≈ 0.56 GB / 2.8 ms ≈ **200 GB/s**, F ≈ **36.6 ms**. So on Spark llama.cpp ~63% of a decode step is byte-independent (launches, attention on tiny tensors, MLA/indexer/Engram glue, sampling, host sync), and the byte-proportional part already runs near the 200 GB/s this class of part achieves. This is why 13% fewer bytes gives 5% speed, and why "1-bit is only ~5% faster than 2-bit".
- [EST] The crossover where decode becomes latency-bound on that stack is B ≈ F × BW ≈ 36.6 ms × 200 GB/s ≈ 7 GB/token: below ~7 GB/token on Spark llama.cpp, less than half the step is bandwidth. Reaching 15 tok/s there needs F < 67 ms regardless of bytes — satisfied — but reaching 25 tok/s needs F < 40 ms *and* B < 0.7 GB, i.e. the fixed term, not the quant, is the wall.
- The Thor CUDA server is the opposite regime: F ≈ 0.4% of the step, so bytes convert at ~116 GB/s effective (not 200) because the gemvs are activation-traffic bound at NVFP4. For bpw below 4.5 (NVFP4) the x-bytes-per-weight-byte ratio gets *worse* (at 2 bpw it would be ~16 B of x per weight byte without row reuse), so a 2-bit format on Thor needs R ≥ 8–10 row reuse per block to stay at the streaming wall — otherwise the local INT2 result ("the byte saving does not translate into B_tok gain", sibling note `extreme_quantization.md`) repeats.
- Dequant cost is real but secondary on both parts: PR #30517's 1.11–1.15× on iq2_s/iq3_s is the size of the dequant penalty, and the local LUT-vs-inline experiment shows the scale decode is instruction-bound. The dominant non-byte cost on llama.cpp is launch count/attention glue, on the CUDA server activation traffic.

### Gaps
- No per-kernel profile of llama.cpp decode on GB10/Thor exists publicly; the fixed-term estimate above is a fit to two points, not a trace.
- No same-model IQ2/IQ3/Q4 comparison on Thor itself; the Spark→Thor penalty is 1.08–1.42× on decode (sibling note).
- Whether IQ2_K/IQ3_K trellis kernels (ik_llama) compile for sm_110 is untested.

---

## Q4. Expert temporal locality and caching when all experts are already resident

### Takeaway
Measured expert-cache hit rates with half the experts cached are 60–78% under LRU (Mixtral 60%, Qwen1.5-MoE 65%, DeepSeek-V2 72%, Phi-3.5-MoE 78%) and up to 93–97% with cache-aware routing — but every such system is an *offload* system where a hit avoids a PCIe/SSD transfer. With all weights resident in Thor's unified memory a "hit" would have to be served from L2, and one token's routed-expert bytes per layer (tens of MB at any bpw) exceed what any L2 can hold across a 75-layer pass, so locality does not reduce DRAM bytes at batch 1. The only resident-weights use of locality is in batched verify (speculative decoding), where the distinct-expert union sets the cost.

### Cited Findings
- **LRU hit rate, cache = half the experts, WikiText (Table 2):** Qwen1.5-MoE 30/60 cached → 35% miss (65% hit), expert lifetime 22 tokens; DeepSeek-V2 32/64 → 28% miss (72% hit), 20 tokens; Phi-3.5-MoE 8/16 → 22% miss (78% hit); Mixtral-8x7B 4/8 → 40% miss (60% hit), lifetime 5 tokens. With cache-aware routing (λ=0.5): 7% / 3% / 9% / 21% miss, lifetimes 111 / 188 / 55 / 10 tokens; "over 2×" speedup over LRU on Snapdragon llama.cpp, "no accuracy loss" on MMLU/GSM8K — [arXiv 2412.00099](https://arxiv.org/html/2412.00099v1)
- The same paper's premise: "the expert selection of state-of-the-art MoEs lacks temporal locality, which leads LRU caching to evict and reload often" — [arXiv 2412.00099](https://arxiv.org/html/2412.00099v1)
- **DALI (local-PC offload):** on Mixtral-8x7B a heatmap of high-workload experts (top 3 by workload) at token i vs i+1 shows "a pronounced diagonal pattern ... if an expert is high-workload for token i, it is highly likely to remain so for token i+1"; PCIe transfer accounts for up to 78.1% of inference time in their setup; HybriMoE's score-based replacement achieves only 25.3% hit rate in one configuration; LRU/HybriMoE hit rates on DeepSeek-V2-Lite top-k=1 at cache sizes 2–32 range 36.7–40.6% (HybriMoE) and 11.8–35.3% (EdgeMoE); Mixtral top-k=2 HybriMoE 56.4–65.2% — [DALI arXiv 2602.03495](https://arxiv.org/pdf/2602.03495) (text extracted from the PDF; figure-axis numbers partially garbled)
- **Across 20 MoE LLMs:** "a strong trade-off exists between local routing consistency and local load balance"; shared experts reduce local routing consistency; "most models balance cache effectiveness and efficiency with cache sizes of about twice the number of active experts" — [arXiv 2505.16056](https://arxiv.org/abs/2505.16056)
- **Predictor-based prefetch:** MoE-Beyond on DeepSeek-V2-Lite traces raises GPU cache hit rate 17% → 72% with 10% of experts cached (simulation) — [arXiv 2508.17137](https://arxiv.org/abs/2508.17137v1) (search summary); Least-Stale argues access is "deterministic sequential layer pattern, not recency-based reuse", >88% hit on OLMoE at 5% capacity — [arXiv 2602.03921](https://www.arxiv.org/pdf/2602.03921) (search summary)
- **Batched verify dilation (resident weights, LOCAL Thor):** with 144 experts top-8, M tokens touch `144(1-(1-8/144)^M)` distinct experts: 8.0 per token at M=1, 7.8 at M=2, 7.4 at M=4, 6.6 at M=8, 5.4 at M=16; measured `ffn:moe` time ratio at width 4 is 0.82 of width 1 (model predicts 0.86); K=4 tokens select 29.4 distinct of a possible 32 — [LOCAL glm53-spec-decode-moe-wall.md, glm53-cuda-server-spec-batch.md]
- **EcoSpec (resident weights, HBM):** on Qwen3-235B the average unique experts loaded per verify step falls 23.7 → 20.5 by choosing draft paths that share experts; speedup 1.22× → 1.36×, up to 1.62× on MT-Bench; ~11.2 GB/step less HBM read; acceptance length unchanged (lossless) — [EcoSpec arXiv 2607.12696](https://www.alphaxiv.org/abs/2607.12696)
- **KV cache is L2-resident on Thor (LOCAL):** "the latent KV cache is L2-RESIDENT, so re-reading it is nearly free and the lever is occupancy, not traffic ... Byte models predict DRAM, not L2; check which one a tensor lives in" — [LOCAL glm53-decode-gap-profile.md]

### Inferences
- [EST] Per-layer routed-expert bytes per token on the 1T-class budget (8 × 37.7M params × 0.275 B at 2.2 bpw) ≈ 83 MB, × 75 layers ≈ 6.2 GB per token. Even a 60–78% "same expert as last token" rate cannot be cashed as DRAM savings because by the time the next token reaches layer l, ~6 GB of other weights have streamed through; no on-chip cache survives that. Temporal locality is a non-lever for resident weights at batch 1.
- Where locality *does* convert bytes is a K-wide verify: the local measurement (29.4 of 32 distinct at K=4) says GLM-5.3's routing is close to independent-uniform within a verify window; EcoSpec's 23.7 → 20.5 on Qwen3-235B (top-8, K unspecified) is the same effect with a predictor. The local note already proposes the cheap test: "instrument the router in a K=4 verify and count ACTUAL distinct experts; if correlation puts it near 20 rather than 29.4, the bound moves".
- Cache-aware routing (2412.00099) is a quality-affecting change to routing (it adds a cache prior to the logits); its "no accuracy loss" claim is on MMLU/GSM8K for Qwen1.5-MoE/Phi-3.5 only. For a resident-weights server it has no byte payoff, so it should not be adopted.

### Gaps
- No published "same expert as previous token" rate per layer on DeepSeek-V3/GLM-5/Kimi-class routers (only V2-Lite / Mixtral / Qwen1.5 / OLMoE / Phi-3.5).
- Thor L2 size and the achievable L2 bandwidth for expert reuse are not in any source; the local note gives only "L2 tops out near 450 GB/s".
- No measurement of expert-set correlation across the K tokens of a verify on GLM-5.3 (the local 29.4 is the independent-uniform model; the measured `ffn:moe` 0.82 ratio hints at mild correlation).

---

## Q5. Activation sparsity inside experts (TEAL/CATS/DejaVu-style) on MoE

### Takeaway
Intra-expert activation sparsity on off-the-shelf MoEs is real and recent: arXiv 2605.08575 (eight MoEs, 1B–400B) reports up to 90% sparsity inside each expert "without significant accuracy loss", 2.5× MoE-layer and 1.2× end-to-end speedup vs dense vLLM, but at its 85% cutoff it keeps only ~95% of baseline benchmark score, which fails the user's "no compromise" bar; the safe operating point is unreported. Only up/down projections can skip rows (gate stays dense), so the byte ceiling is 2/3 of expert bytes × sparsity, and a row-skipping gemv on sm_110a is a gather kernel that will fight the activation-traffic wall already measured locally.

### Cited Findings
- **Intra-expert sparsity (2605.08575):** eight off-the-shelf MoE models from 1B to 400B; "sparsity of up to 90% exists within each expert"; threshold-based neuron masking with a lookup table from target sparsity to per-neuron threshold; at the model's sparsity cutoff of 85% sparse execution retains ~95% of baseline average benchmark score and reaches 1.55× (MoE layer) at that cutoff; up to 2.5× MoE-layer and 1.2× end-to-end vs dense vLLM; "only the up-down projection kernel benefits from more sparsity, and the gate projection stays dense" — [arXiv 2605.08575](https://arxiv.org/abs/2605.08575) (abstract + search summary; full tables not fetched)
- **TEAL (dense models):** magnitude thresholding of hidden states, training-free; 1.53× at 40% and 1.8× at 50% sparsity with custom kernels, on dense Llama/Mistral-class models — [TEAL arXiv 2408.14690](https://arxiv.org/html/2408.14690v1). No Mixtral/DeepSeek evaluation found.
- **CATS:** a contextual threshold on the gate activation; the only hit was a third-party summary with no MoE decode numbers — [themoonlight CATS review](https://themoonlight.io/review/cats-contextually-aware-thresholding-for-sparsity-in-large-language-models)
- **TurboSparse (dReLU, Mixtral/Mistral):** retrains the model to be ReLU-sparse rather than thresholding a pretrained one; 2–5× decode speedup claimed — [TurboSparse arXiv 2406.05955](https://arxiv.org/html/2406.05955v1) (search summary). Not training-free; not applicable post-hoc to a frontier checkpoint.
- **Tensor/neuron-level MoE sparsity (ICML 2026):** Mixtral-8x7B, OLMoE-Instruct, DeepSeek-V2-Lite-Chat; neuron-level sparsity exists but the page gives no percentage; inference-only 1.41× MoE speedup at <0.5% loss on the two smaller models; Mixtral 1.19× with +0.5% after post-training — [PMLR v306 cai26i](https://proceedings.mlr.press/v306/cai26i.html)
- **Local kernel constraint (Thor):** a uint4 load that ties a lane to 32 contiguous weights made the warp touch 32 different 128-byte lines and was 3.4× slower than uint32; row-granular gathers have the same access pattern problem — [LOCAL glm53-activation-traffic-finding.md]

### Inferences
- [EST] Byte ceiling: expert = gate + up + down, each 1/3 of expert bytes. Skipping rows of up/down at sparsity s removes (2/3)·s of routed-expert bytes: s=0.5 → −33% routed bytes (on the 1T budget, −2.1 GB/token at top-8 2.2 bpw); s=0.85 → −57%, but at the ~5% quality cost 2605.08575 reports. Combined with k=4 (Q1) the two levers multiply: routed bytes × 0.5 × 0.67 = 0.33.
- The CATS/TEAL style needs the gate activation (SiLU(x·W_gate)) *before* deciding which up/down rows to read — so the gate gemv runs dense, then a data-dependent row index is built per token per expert, then up/down run as gathers. On the local server a gemv is already activation-traffic-bound at NVFP4; a gather adds index traffic and breaks the R=5 row-reuse scheme that got the gemv to 161 GB/s. Expect the realized gain to be well below the byte saving until a dedicated sparse-row kernel is tuned on sm_110a — this is the INT2 lesson again.
- Quality at s≈0.5 on a frontier MoE is unknown; the only MoE number (95% retained at s=0.85) is not "no compromise". Treat intra-expert sparsity as a measured-later lever, after the static k reduction.

### Gaps
- No per-sparsity quality table for any 100B+ MoE; 2605.08575's full tables were not retrievable (abstract only).
- No TEAL/CATS result on Mixtral, DeepSeek, Qwen3-MoE or GLM.
- No sparse-row gemv measurement on sm_110a or GB10.
- Whether REAP-pruned experts (which absorb merged/retained routing mass) are more or less activation-sparse than unpruned ones is unmeasured.

---

## Q6. Attention/dense side: NVFP4/INT4 on attention projections, FP8/FP4 KV, linear-attention state precision

### Takeaway
The dense side is the lever with the best measured byte-to-quality ratio: locally NVFP4 on 13.16 GiB of bf16 dense weights halved `B_tok` (19.76 → 9.76 GB) for +1.56% PPL and 0.138% confident flips; NVFP4 KV cache is reported <1% loss by NVIDIA on Qwen3-480B (vendor) and FP8 KV is the "safest default" per vLLM's TurboQuant study; the KDA recurrent state is 145.56 MiB total on GLM-5.3-Flash so its precision is a quality question, not a byte question.

### Cited Findings
- **NVFP4 dense overlay, GLM-5.3-Flash (LOCAL Thor):** 13.16 GiB bf16 → 3.70 GiB NVFP4 overlay; `B_tok` 19.761 → 9.762 GB; decode 8.36 → 11.86 tok/s (speedup came from the activation-traffic fix, not the bytes); PPL 4.430457 → 4.499673 (+1.56%, 8.8σ), top-1 agreement 90.101%, confident disagreements 0.138%; "quote +1.56% ppl and 0.138% confident flips, never the 0.9972 cosine"; left bf16: `kv_b_proj`, hyper-connection `fn`, router; reverting lm_head alone would add +0.912 GB/token (~−8.5% decode) — [LOCAL glm53-nvfp4-dense-overlay.md]
- **KDA/linear-attention dense weights:** four bf16 [8192,4096] matrices per KDA layer, 268 MB/layer, read every token; NVFP4 per-weight cos 0.9955 but layer-output cos 0.9999992; the state after 64 tokens is cos 0.9977 vs bf16, "better than the weights driving it" — [LOCAL glm53-cuda-server-roofline.md]
- **KDA state size:** the recurrent checkpoint is 145.56 MiB, "fixed by the model, independent of context length" — [LOCAL glm53-mtp-serving-path.md]
- **`kv_b_proj` double read:** `k_absorb_q` and `k_expand_v` each stream the whole 33.55 MB per token (0.738 GB/token, 7.6% of B_tok); fixed by batching per chunk, so "converting kv_b to NVFP4 is no longer worth a second kernel" — [LOCAL glm53-nvfp4-dense-overlay.md]
- **NVFP4 KV cache [VEN, Blackwell, values read from charts]:** Qwen3-480B-A35B FP16 / FP8 / NVFP4: LiveCodeBench ~58 / ~58 / ~58; MMLU-Pro ~78.2 / ~78.1 / ~77.4; MBPP ~80.8 / ~79.7 / ~79.9; RULER-64K ~95.6 / ~95.5 / ~94.6 (−1.0 from FP16, at the "<1%" boundary); Llama-3.3-70B MMLU FP8 ~82.5 / NVFP4 ~81.9 / MXFP4 ~77.8; ~50% smaller than FP8 KV; implementation dequantizes NVFP4 → FP8 before attention; no MLA statement — [NVIDIA NVFP4 KV cache blog](https://developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache/)
- **vLLM TurboQuant study:** FP8 is "the safest default" for KV quantization; TurboQuant 4-bit is the most practical of that family with "moderate accuracy, latency, and throughput costs"; 3-bit variants show larger drops on long-context and reasoning tasks — [vLLM TurboQuant blog](https://vllm.ai/blog/turboquant)
- **vLLM KV support:** `kv_cache_dtype` auto / fp8 / fp8_e4m3 / fp8_e5m2 only; scales default to 1.0 unless calibrated; sliding-window layers are "more sensitive" and can be skipped — [vLLM quantized KV docs](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache.html) (as compiled in sibling note `extreme_quantization.md`)
- **Attention-weight quantization in MoE compression papers:** MC-MoE and EAC-MoE quantize attention + gate at 4-bit with "no more than 0.05 bits" model-wide overhead and attribute their losses to the 2–3-bit experts, not the 4-bit attention — [MC-MoE arXiv 2410.06270](https://arxiv.org/html/2410.06270), [EAC-MoE arXiv 2508.01625](https://arxiv.org/html/2508.01625v1)
- **DSA caps KV bytes:** on GLM-5.3 the DSA indexer selects a top-2048 subset and is a no-op below 2048 context — [LOCAL glm53-cuda-server-build.md (as summarized in MEMORY.md)]

### Inferences
- [EST] For the 1T-class budget (below), dense weights at bf16 are the single largest per-token term (~16 GB for ~8B attention params + 5.7 GB shared experts + 1.9 GB lm_head); NVFP4 (4.5 bpw incl. E4M3 block scales) removes ~72% of that. This is where the local overlay's −50.6% came from and it transfers directly.
- [EST] KV bytes are second order on MLA models: 576-dim latent × 78 layers × 2 B ≈ 90 KB/token; at 8K context that is 0.74 GB/step bf16, 0.37 GB FP8, 0.18 GB NVFP4 — 3–5% of the dense-NVFP4 budget. With DSA's top-2048 selection the read is capped at ~2048 keys per layer regardless of context, so FP8 KV is enough; NVFP4 KV buys ~0.2 GB/step for a measurable RULER hit.
- Linear-attention state (145 MiB read + written per token ≈ 0.29 GB/step) is 3% of the local 9.76 GB budget; quantizing it would save ~0.15 GB and the upstream caution that KDA states are "susceptible to rounding errors" argues for leaving it bf16/fp32. Not a lever.

### Gaps
- No independent (non-vendor) NVFP4-KV accuracy study, and none on MLA latent caches or on DSA-selected caches.
- The local overlay's lm_head share of the 0.138% confident flips is unmeasured ("likely owns a disproportionate share").
- No measurement of INT4 (GPTQ/AWQ-style) vs NVFP4 on attention projections at matched bytes for a KDA/MLA hybrid.

---

## Q7. MTP / multi-token heads measured as tokens per weight read

### Takeaway
Vendor MTP numbers (DeepSeek-V3 85–90% second-token acceptance → 1.8× TPS; GLM-5 accept length 2.76; MiMo-V2-Flash 3.6 accept length → 1.86–2.70× at batch 32–128; GLM-5.3 DFlash-2 4.85–5.86 accept length → 2.39–2.79× on 4× GB300) are all on HBM parts where a K-wide verify costs about one weight pass. On Thor the local verify-cost curve (K=2 costs 2.16 AR steps, K=4 2.64, K=8 4.98) is set by expert-union dilation, so measured net gain at 72.3% acceptance is 0.77–0.95× and the bound is 1.13× at 95% / 1.27× at perfect acceptance. Tokens per *routed-expert* read is the right metric and on a fine-grained MoE it barely exceeds 1 at batch 1.

### Cited Findings
- **DeepSeek-V3 (671B, 1 MTP module):** "the acceptance rate of the second token prediction ranges between 85% and 90% across various generation topics ... 1.8 times TPS" — [DeepSeek-V3 report](https://arxiv.org/pdf/2412.19437); hardware paper gives 80–90% from "real world practice data", same 1.8× — [arXiv 2505.09343](https://arxiv.org/pdf/2505.09343); AMD SGLang on Instinct: 1.25–2.11× random dataset, 1.36–1.80× ShareGPT, shrinking with concurrency — [AMD ROCm MTP blog](https://rocm.blogs.amd.com/software-tools-optimization/mtp/README.html)
- **GLM-5:** accept length 2.76 vs 2.55 for DeepSeek-V3.2; parameters shared across 3 MTP layers during training to keep draft memory near DeepSeek-V3's while raising later-token acceptance — [GLM-5 report arXiv 2602.15763](https://arxiv.org/pdf/2602.15763) (search summary); GLM-5.2 extends MTP from 3 to 5 draft tokens — [vLLM recipe GLM-5.2](https://recipes.vllm.ai/zai-org/GLM-5.2)
- **MiMo-V2-Flash (309B/15B, 256 experts top-8, 3 MTP heads of ~0.33B each, dense FFN + SWA):** up to 3.6 acceptance length; Table 10 (16K in / 1K out, SGLang FP8): at batch 32 speedup 1.86× (accept 2.8) → 2.52× (3.8); batch 96 1.99× → 2.70×; batch 128 1.82× → 2.46×; acceptance length vs next-token cross-entropy fits y = 4(1 − 0.58x^0.58), R² 0.995 — [MiMo-V2-Flash report](https://arxiv.org/html/2601.02780v1)
- **GLM-5.3-Flash native MTP vs DFlash-2 [VEN, 4× GB300 TP4 SGLang, block 8]:** accept length MTP 5.06 / 4.95 / 4.70 / 4.26 vs DFlash-2 5.78 / 5.86 / 5.32 / 4.85 on GSM8K / MATH-500 / HumanEval / MBPP; throughput MTP 1.93× / 2.05× / 1.94× / 1.78× vs DFlash-2 2.42× / 2.79× / 2.62× / 2.39× — [LOCAL wiki 55-speculative-decoding.md, vendor table]
- **Vanilla single-head MTP reused recursively:** acceptance ~70% at step 1 falling to ~10% at step 2; FastMTP's self-distilled head on MiMo-7B reaches τ = 3.16 at K=3 on math and 2.03× average speedup (vanilla reuse 1.21×), with speed peaking at K=3 and declining after — [FastMTP arXiv 2509.18362](https://arxiv.org/pdf/2509.18362) (text extracted from PDF)
- **Thor measurement, GLM-5.3-Flash NVFP4 CUDA server (LOCAL):** verify-cost curve vs K=1: K=2 **2.164**, K=3 2.198, K=4 2.636, K=8 4.977, K=16 9.477; native un-fine-tuned MTP block 72.3% acceptance; expected net: depth 1 0.77×, depth 2 0.95×, depth 3 0.91×, depth 7 0.60×; bound for any head: 85% → 1.02×, 95% → 1.13×, perfect → 1.27×; per-width dprof at width 4: `attn:kda` 0.54, `attn:mla` 0.46, `mla:absorb_q` 0.25, **`ffn:moe` 0.82** of width-1 time — [LOCAL glm53-spec-decode-moe-wall.md]
- **The byte model had predicted 1.38–1.49× at depth 3** (non-expert weights read once = 15.005 GB, +0.5945 GB per distinct expert, draft lm_head 1.269 GB/drafted token, NVFP4 on the draft head "system-level lossless") before the real verify curve was measured; "measure the serving-side batching curve BEFORE building or capturing for a draft head" — [LOCAL glm53-cuda-server-spec-batch.md, glm53-spec-decode-moe-wall.md]
- **GGUF path:** llama.cpp `op_offload_min_batch_size = 32` makes a K+1 verify cost nearly K+1 decodes; IQ3_M MTP depth 3 at 58.5% acceptance was 1.85 vs 2.99 tok/s baseline — [LOCAL glm53-mtp-serving-path.md]
- **Draft-side cost on a fine-grained MoE is dominated by lm_head:** 1.269 GB per drafted token vs 0.371 GB for the MTP block's MLA+MoE layer — [LOCAL glm53-cuda-server-spec-batch.md]

### Inferences
- [EST] Tokens per weight read = τ / cost(K) where cost(K) is in AR-step units. HBM parts: cost(K) ≈ 1.0–1.1, so τ=2.76 → ~2.5× and τ=3.6 → ~2.6× (MiMo's table confirms ~linear in τ). Thor NVFP4 server: cost(3) = 2.20, so even τ=2.76 gives 1.25×, and at the measured 72.3% per-token acceptance τ(3) ≈ 1+0.72+0.52+0.38 = 2.6 → 1.19× *before* the draft's own cost (0.08 AR-steps/draft token) → ~0.95×, matching the local result.
- [EST] The reason cost(K) is ~2.2 not ~1.2 is that routed experts are ~49% of the post-overlay budget and K tokens read K× nearly-disjoint expert sets; non-expert bytes (51%) are read once. cost(K) ≈ 0.51 + 0.49 × E[distinct]/8 — at K=3: 0.51 + 0.49 × 22.4/8 = 1.88 (measured 2.20, the remainder is sub-linear batching of the attention kernels at small M). Any lever that shrinks the routed share (k=4, NVFP4→2-bit experts) *improves* the spec-decode bound, and any lever that grows it (bf16 attention → NVFP4) worsens it — the two families interact.
- The user's "+34% at 95% acceptance" figure was not found in the local notes; the closest local numbers are the pre-measurement byte-model prediction (1.38× at depth 3, 1.49× with NVFP4 draft lm_head) and the post-measurement bound (1.13× at 95%). The later measurement supersedes the prediction.
- Block-diffusion drafters (DFlash-2) raise τ to ~5–6 on HBM, but on Thor τ=5.5 at cost(7) ≈ 4.5 gives ~1.2×; the dilation wall binds them too unless the verify is restricted to paths that share experts (EcoSpec's 23.7 → 20.5 distinct experts is the published version of the "count actual distinct experts" experiment the local note proposes).

### Gaps
- No per-step acceptance curve (p1, p2, p3) for GLM-5/5.3 or DeepSeek-V3.2 MTP; only mean accept length.
- No published verify-cost curve for any fine-grained MoE on a unified-memory part other than the local one.
- Whether draft-path expert-sharing (EcoSpec) is implementable with a KDA recurrence (state slots) on the local server is unexamined.

---

## Q8. Weight sharing across layers and lookup/retrieval components (Engram)

### Takeaway
Cross-layer expert sharing (CS-MoE, Megrez2, Sticky Routing) is a train-from-scratch architecture family: CS-MoE reports a 45% reduction in unique memory traffic per token at 8B scale on 8× H200 (batch 1) but it cannot be applied to an existing checkpoint. Engram-style lookup tables are the opposite trade: 12.4 KiB/token of random reads from NVMe that cost ~1% of a decode step, so they move ~100–200 GB of capacity *off* the bandwidth path for free — the only lever here that reduces bytes without touching quality, and only on models that already have it.

### Cited Findings
- **CS-MoE:** layer-independent experts plus a globally shared pool accessed by every layer; 0.6B–12B models; "outperforming dense baselines in perplexity while requiring only 55% active parameters"; 8× H200 batch-1 profiling: "with cache optimization, CS-MoE approaches the inference speed of Dense models and outperforms standard MoEs due to a 45% reduction in unique memory traffic per token" because "adjacent layers frequently access overlapping subsets of shared experts, active weights can be retained in GPU L2/SRAM" — [arXiv 2609.22199](https://arxiv.org/pdf/2609.22199) (text extracted from PDF)
- **Sticky Routing:** a soft consistency loss during training reduces expert switch rate by up to ~59% at both scales with −0.5% to −4.1% perplexity (improvement) at low λ; LRU cache hit rate up to 0.889; small models (PPL 245–263); hard consistency costs +5.6% PPL — [arXiv 2607.08780](https://arxiv.org/pdf/2607.08780) (text extracted from PDF)
- **Megrez2:** expert pools shared across groups of consecutive layers, each layer keeps its own gate and projections; motivated by memory-limited serving — [Megrez2 report arXiv 2507.17728](https://arxiv.org/pdf/2507.17728) (search summary)
- **Engram on Thor (LOCAL, measured 2026-10-17):** per token each of two Engram modules reads 24 rows (3 n-gram orders × 8 hash heads), 48 rows × 264 B = 12.4 KiB/token from ~48 random addresses that are a pure function of token ids (known before the forward); 48 concurrent O_DIRECT row reads = 1.1–1.3 ms p50 on the WD SN5000S, ~1% of a 100 ms step; Engram resident at MXFP4 is 97.3 GiB, so off-pool is mandatory — [LOCAL research note engram_off_device_thor.md]
- **DeepSeek-V4.1-Flash composition:** 510.3 GB = routed experts 288.8 + Engram 203.1 (FP8) + MTP 7.9 + everything else 10.4 — [LOCAL thor-base-model-shortlist.md as compiled in sibling note extreme_quantization.md]

### Inferences
- [EST] Engram is the one component whose per-token bytes are 12 KB not GB: it contributes ~0.0001% of `B_tok` while holding 40% of the checkpoint. A model that puts capacity into lookup tables instead of experts is structurally the right shape for a bandwidth-bound device, and it is the only lever in this note that is free of quality cost by construction (the table is exact, not approximated).
- Cross-layer sharing cannot be retrofitted: REAP/HOPE prune experts, they do not tie them. The CS-MoE 45% figure is a from-scratch architecture result and should not enter the budget for an existing 1T checkpoint. The nearest post-hoc analogue — merging similar experts across layers (REAM-style, sibling note `ream_post_reap.md`) — reduces resident bytes, not per-token bytes, because each layer still reads its own selected experts.

### Gaps
- CS-MoE's Table 11 latency rows were not extractable from the PDF text (model/config labels only).
- No from-scratch cross-layer-sharing model exists at 100B+ with public weights, so there is nothing to serve.

---

## Q9. Worked B_tok budget: ~100 GiB REAP-50 1T-class model on Thor

### Takeaway
On a GLM-5-shaped 744B model (75 MoE layers × 256 experts top-8 + 1 shared, expert 37.7M params, ~8B attention params [EST]) at REAP-50, routed experts must sit at ~2.2 bpw to fit 117 GiB (367B × 2.2/8 ≈ 101 GB + ~10 GB dense NVFP4), which is exactly the local fit-frontier note's conclusion; at that point `B_tok` ≈ 13.6 GB with NVFP4 dense + FP8 KV, and ≥15 tok/s needs either ~200 GB/s effective (the kernel-achievable rate, not the 116 GB/s the local server realizes) or `B_tok` ≤ 7.7 GB at the realized 116 GB/s — reachable with k=4 (k2=2k) plus 50% intra-expert sparsity, each of which carries a ≤0.6-point / unmeasured quality cost respectively.

### Cited Findings (inputs to the arithmetic)
- GLM-5: 744B total, ~40B active, 256 routed experts, 8 active + 1 shared per layer, expert intermediate 2048, hidden 6144, 78 layers of which 3 dense — [GLM-5 report arXiv 2602.15763](https://arxiv.org/pdf/2602.15763) (shape summary; counts cross-checked against LOCAL glm53-753b-thor-envelope.md "97.5% routed experts")
- 1T-class fit: GLM-5.3 753B "fits at proven REAP50 if quant ~2.2-bit"; practical GPU-visible envelope ~117 GiB — [LOCAL glm53-753b-thor-envelope.md, wiki 20-host-thor.md]
- NVFP4 effective bits: "NVFP4 is really 0.62 B/w" (≈4.96 bpw incl. scales and padding as shipped) — [LOCAL thor-base-model-shortlist.md]; nominal 4.5 bpw (4-bit + E4M3 scale per 16) used below
- Bandwidth denominators: 57–66 GB/s (user-stated effective at batch 1 on GGUF); **116 GB/s** realized on the local NVFP4 CUDA server (9.762 GB × 11.91 tok/s); 190–200 GB/s per kernel; 235–247 GB/s streaming ceiling — [LOCAL glm53-decode-gap-profile.md, glm53-kda-launch-bound.md]
- Per-expert bytes at NVFP4 on GLM-5.3-Flash: 0.5945 GB per distinct expert per verify; non-expert bytes 15.005 GB bf16 → after overlay routed ≈ 4.76 GB of 9.76 — [LOCAL glm53-cuda-server-spec-batch.md, glm53-cuda-server-roofline.md]

### Budget model [EST — all arithmetic mine; replace parameter counts with `tools/roofline.py` output]

Shape: GLM-5-class, REAP-50 → 128 routed experts/layer survive, top-8 still read. Per token:

| Component | Params read/token | bf16 | FP8 | NVFP4 (4.5 bpw) | 3.0 bpw | 2.2 bpw |
|---|---:|---:|---:|---:|---:|---:|
| Routed experts, k=8 (8 × 37.7M × 75) | 22.6B | 45.2 GB | 22.6 | 12.7 | 8.48 | **6.22** |
| Shared expert (1 × 37.7M × 75) | 2.83B | 5.66 | 2.83 | **1.59** | 1.06 | 0.78 |
| Attention (MLA+DSA, ~91M/layer × 78) | ~7.1–8B | 16.0 | 8.0 | **4.5** | — | — |
| Dense FFN (3 layers) | ~0.6B | 1.2 | 0.6 | **0.34** | — | — |
| lm_head (154,880 × 6144) | 0.95B | 1.9 | 0.95 | **0.54** | — | — |
| KV read, 8K ctx (576 × 78 × 8192) | — | 0.74 | **0.37** | 0.18 | — | — |
| Router / norms / activations | — | <0.1 | | | | |

Resident check at REAP-50: routed 128 × 75 × 37.7M = 362B → 2.2 bpw = 99.6 GB; 3.0 bpw = 136 GB (**does not fit**; needs REAP-65 or 2-bpw); dense ~11.5B at NVFP4 ≈ 6.5 GB, plus KV. So the "3 bpw experts" row of the user's requested table is a non-fitting configuration on this shape; it is shown for the bytes arithmetic only.

Scenario table (bold = fits 117 GiB):

| # | Configuration | B_tok (GB) | tok/s @60 | @116 | @200 | Quality evidence |
|---|---|---:|---:|---:|---:|---|
| A | experts 2.2 bpw, dense bf16, KV bf16 | 31.7 | 1.9 | 3.7 | 6.3 | baseline, "quantized where the disk was" |
| **B** | **A + NVFP4 dense/shared/lm_head + FP8 KV** | **13.6** | 4.4 | 8.5 | 14.7 | local overlay: +1.56% PPL, 0.138% confident flips |
| C | B but experts 3.0 bpw (does not fit) | 15.9 | 3.8 | 7.3 | 12.8 | — |
| **D** | **B + k=6 (k2=2k renorm)** | **12.0** | 5.0 | 9.7 | 16.7 | Qwen3.6-35B (6,16): MMLU +0.75 n.s., GSM8K 0.0 |
| **E** | **B + k=4 (k2=2k renorm)** | **10.5** | 5.7 | 11.0 | 19.0 | Qwen3.6-35B (4,16): MMLU −0.35 (p=0.66), GSM8K −0.6; no GLM data |
| **F** | **E + 50% intra-expert sparsity on up/down** | **9.4** | 6.4 | 12.3 | 18.8 | 2605.08575: 95% retained at 85% sparsity; 50% point unmeasured |
| **G** | **E + NVFP4 KV + lm_head bf16 revert** | 11.7 | 5.1 | 9.9 | 17.1 | lm_head revert likely removes most confident flips (local, unmeasured) |
| H | B + MTP depth 3 at 95% accept (Thor bound) | 13.6 ÷ 1.13 eff. | 5.0 | 9.6 | 16.6 | lossless by construction; bound from local verify curve |
| **I** | **E + MTP depth 3 at 95% (bound improves as routed share falls)** | 10.5 ÷ ~1.2 eff. | 6.9 | 13.3 | 23 | lossless; the 1.2 is [EST] from cost(K) ≈ 0.57 + 0.43·E[distinct]/4 |

Deltas per lever on top of B (13.6 GB): NVFP4 dense −18.1 GB (the big one, already in B); k 8→6 −1.56; k 8→4 −3.11; intra-expert 50% sparsity at k=4 −1.04 (−2.07 at k=8); experts 2.2→2.0 bpw −0.57 (k=8); FP8→NVFP4 KV −0.19 at 8K; KDA/linear-state precision n/a on this shape (MLA model); Engram-style lookup 0 bytes (not present in GLM-5).

Where 15 tok/s lands:
- At the user's 60 GB/s effective: no configuration reaches 15 tok/s; the best lossless-ish stack (I) is ~7 tok/s. That denominator is a GGUF-regime number (fixed ~37 ms term, Q3); the lever there is the serving stack, not bytes.
- At the local server's realized 116 GB/s: configuration F or I reaches 12–13 tok/s; 15 tok/s needs B_tok ≤ 7.7 GB, i.e. E plus either 2.0-bpw experts and intra-expert sparsity, or k=4 plus MTP once the routed share is small enough that cost(K) drops toward 1.3.
- At the kernel-achieved 200 GB/s: B alone is 14.7 tok/s and D/E clear 15 tok/s with no quality lever beyond the NVFP4 dense overlay and k=6. Closing the 116 → 200 GB/s gap is worth more than every expert-side lever combined — it is the activation-traffic problem (R-row reuse at low bpw) in [LOCAL glm53-activation-traffic-finding.md].

### Inferences
- Order of operations by measured quality cost per GB saved: (1) NVFP4 dense/shared/lm_head (−18 GB, +1.56% PPL measured locally); (2) FP8 KV (−0.4 GB, "safest default"); (3) k 8→6 with k2 = 2k (−1.6 GB, within noise on Qwen3.6); (4) k 8→4 with k2 = 2k (−3.1 GB, −0.35/−0.6 pp on Qwen3.6, unmeasured on GLM or on a pruned router); (5) MTP depth 2–3 (lossless, but worth only 1.1–1.2× on Thor until the routed share is small); (6) intra-expert sparsity (bytes attractive, quality and kernel both unproven); (7) 2.0-bpw experts (small byte win, kernel risk per the INT2 result).
- Levers (3)/(4) and (5) compound favourably: cutting k lowers the expert-union term that caps spec decode on Thor. Levers (1) and (5) compound unfavourably (NVFP4 dense raises the routed share from 24% to ~49%, which is why the local spec-decode bound fell from the predicted 1.49× to 1.13×).
- "Without compromise" is met by (1)+(2)+(5) and arguably (3); it is *not yet* demonstrated for (4), (6), (7) on any 100B+ sigmoid-router model, and never on a REAP'd one. The cheapest experiment that moves the whole table is a k2 scan ({4,8,16}) at k1 ∈ {6,4} on the existing GLM-5.3-Flash REAP-50 NVFP4 server with the local `perplexity.cu` + a 2000-item MMLU paired test, since the server already exposes per-token routing.

### Gaps
- The attention parameter count for GLM-5 (~7–8B) is estimated from the architecture, not read from headers; the local roofline tool should replace it.
- No source gives a measured decode rate for any model with k reduction *and* NVFP4 dense *and* MTP together; the scenario rows are additive byte arithmetic, not measurements.
- The 60 GB/s "effective" denominator the user quotes could not be reconciled with the local 116 GB/s server measurement; they are different stacks (GGUF vs CUDA server) and the note treats them as separate regimes.
