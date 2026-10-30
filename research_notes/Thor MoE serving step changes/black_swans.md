# Black-swan and step-change techniques for single-user MoE decode on Jetson AGX Thor (DSV4.1-Flash REAP-192)

Scope and conventions. Research date 2026-10-30. Baseline under review: 25 tok/s decode, 78% of the bandwidth roofline (8.57 GB/token, about 214 GB/s effective against about 228 GB/s nominal); spec decode is about 1.6x on math and about 1x on chat; prefill is 739 tok/s.

Evidence labels:
- **[MEASURED-SCALE]**: measured on models of 100B+ parameters or in production.
- **[MEASURED-SMALL]**: measured, but only on models of 30B or less, or on a single setup.
- **[CLAIM]**: an abstract, blog post or vendor statement with no numbers I could verify.
- **[EXTRAPOLATION]**: my own arithmetic, not a source finding.

Several arXiv PDFs did not extract (FlateDecode), so some figures come from abstracts only. This is flagged where it applies.

Meta-finding: most of the 2025-2026 "MoE speedup" literature reports gains **against an expert-offloading baseline** (PCIe or Flash). On Thor every expert is resident in unified memory, so those headline numbers (2-5x) mostly do not transfer. The techniques that do transfer reduce **bytes touched per accepted token**: fewer experts, fewer unique experts per verify block, fewer bits per weight, or more tokens per pass.

---

## Q1. Can expert prediction, prefetch, caching or temporal expert locality make decode or spec-decode verify cheaper?

### Takeaway
Prefetch and caching solve an offloading problem that Thor does not have, because every expert is already resident at the same bandwidth tier. Their reported speedups are all measured against offloading baselines and do not transfer. The transferable result is **temporal expert overlap inside a verify block**: unique experts for a K=3 block are well below (K+1)x, and a NeurIPS-2026 paper (2610.38981) finds that *fine-grained* MoEs get **more** spec-decode speedup at batch size 1, partly because L2 absorbs repeated reads of co-routed experts. Cost-aware drafting (EcoSpec) and verifier expert-set restriction (AcceptMoE) are the 2026 levers aimed at the same "chat ~1x" wall this project hit.

### Cited Findings
- **[CLAIM, abstract only; accepted at NeurIPS 2026, posted 2026-10-28] (2610.38981).** Covers 10 sparse MoEs from 30B to 1T parameters with 8-512 experts, on H100. Verify cost depends on activated-expert bytes, temporal overlap across draft tokens, and GPU execution dynamics. Recent fine-grained models show *higher* speedup at low batch, contradicting the earlier "concave in batch" prediction. Kernel profiling attributes the gain to MoE kernels reaching higher bandwidth utilization than attention, and to "L2 cache absorbing repeated reads of co-routed experts". A max-entropy IPF model predicts unique-expert counts within 4% error. The abstract gives no per-model numbers and does not name the models. — [arXiv 2610.38981](https://arxiv.org/abs/2610.38981)
- **[MEASURED-SCALE, production Cohere MoE (k=8 of 128 experts, with shared experts), K=3] Cohere blog.**
  - 4 verify tokens touch about 20.36 unique experts, versus 8 for one token: 2.55x top-k, not 4x.
  - Adjacent-token expert overlap is 0.381.
  - Verify time is Tt(1)/Tt(4)=0.80 at BS=1, so verifying 4 tokens costs only 1.25x one token.
  - Observed SD speedup is 1.95x at acceptance length 2.73.
  - The blog's own "1.25-1.42x" headline mixes a time ratio with an expert-count ratio. Use the 2.55x and 0.80 figures.
  — [Cohere](https://cohere.com/blog/mixture-of-experts-models-get-more-from-speculative-decoding)
- **[CLAIM, abstract] AcceptMoE (2608.02989).** A verifier-side expert selector that sizes the verifier's expert set per block from router scores plus offline "commitment" probabilities.
  - With all experts resident (SGLang, BS=1): 1.290x over EAGLE-3.
  - Under offloading: 2.06x.
  - Accuracy cost: −0.27 pp mean over 12 model-task pairs.
  - **Not lossless**: it changes the target distribution.
  — [arXiv 2608.02989](https://arxiv.org/abs/2608.02989)
- **[CLAIM, search snippet] EcoSpec (2607.12696).** Adds the expert working set as an explicit cost term when selecting the draft tree, because standard acceptance-maximizing selection ignores expert locality. — [arXiv 2607.12696](https://arxiv.org/html/2607.12696)
- **[CLAIM] Related verify-cost work.**
  - EVICT (2605.00342) picks tree prefixes by trading expected accepted length against profiled verify cost. — [arXiv 2605.00342](https://arxiv.org/pdf/2605.00342)
  - Utility-driven SD (2506.20675) finds that expert-token ratio and speculation cost both show short-horizon temporal locality, and uses it to adapt K per iteration. — [arXiv 2506.20675](https://arxiv.org/pdf/2506.20675)
  - A September 2026 paper (2609.22471) finds that routers with high expert coactivation run much faster under verification. — [arXiv 2609.22471](https://arxiv.org/html/2609.22471v1)
- **[MEASURED-SMALL, offloading baselines only] Prefetch and caching systems.** Fate reports 97.15% cross-layer prefetch accuracy (from the previous layer's gate input) and up to 4.1x over on-demand loading. ProMoE reports 2.07x average and 5.02x peak against offloading. MoE-SpeQ reports 2.34x on Phi-MoE against an offloading framework. All of these speedups come from hiding PCIe or Flash transfers. — [Fate 2502.12224](https://arxiv.org/pdf/2502.12224v1); [ProMoE](https://arxiv.org/html/2410.22134v3); [MoE-SpeQ](https://arxiv.org/html/2511.14102v1)

### Inferences
- **[EXTRAPOLATION]** On Thor, "expert offloading inverted" has exactly one useful form: L2 or on-chip reuse across the tokens of one verify block. Thor's L2 is small next to an expert of about 2-3 bit × 5120 × FFN width, so it may help only when the block is processed expert-major (load each expert once and apply it to every token in the block that routed to it). This is the same "reuse x/W across rows" lesson as the project's activation-traffic finding.
- **[EXTRAPOLATION]** The "chat ~1x" result fits the Cohere numbers. A 2.55x unique-expert blow-up for 4 tokens means verify bytes grow about 2.5x on the routed-expert share. With chat acceptance lengths of about 1.5-2, that cancels the gain. Two levers attack this directly:
  - (a) cost-aware tree selection (EcoSpec/EVICT style), which drafts only branches whose experts are already in the union;
  - (b) a lossy, restricted verifier set (AcceptMoE, about −0.3 pp).

  Neither has a published ceiling above about 1.3x when all experts are resident.
- Prefetching Engram rows from NVMe is the only place on this system where the offloading literature applies directly.

### Gaps
- None of the sources gives unique-expert counts for DeepSeek-V3/V4-family routing (top-6 or top-8 over 256-384 experts, noaux_tc). 2610.38981 may include one, but the model list sits in an unextracted PDF.
- I found no published measurement of verify-block L2 reuse on a small-L2 SoC (Thor or Orin).

---

## Q2. Can the AR MoE be converted to a parallel decoder (diffusion conversion, CLLM, Jacobi Forcing, MTP fine-tunes) for more than 2x lossless?

### Takeaway
2025-2026 produced credible AR-to-parallel conversions, but their compute bills do not fit this project:
- **Jacobi Forcing** (ICML 2026): about 4x on code and math at 7B, causal, KV-reusing, with near-AR quality.
- **ZAYA1-8B-Diffusion**: an AR MoE converted, with a 4.6x lossless-sampler "speedup".
- **SDAR**: a 30B MoE adapted to block diffusion.

All are at 7-30B scale. ZAYA used about 1.1T tokens of conversion training. None of them measures the MoE expert-union penalty, which is exactly what kills multi-token decode on this model. No conversion of a model above 100B has been reported.

### Cited Findings
- **[MEASURED-SMALL, 7B dense; project-reported] Jacobi Forcing (Hao AI Lab, ICML 2026).** The model trains on its own Jacobi trajectories with noisy future blocks, stays causal, and keeps the KV cache.
  - About 4x wall-clock on code and math, with up to 4.5x tokens per forward.
  - HumanEval: 4.0x at 83.5% pass@1. GSM8K: 3.7x at 91.4%.
  - Against D2F diffusion: 4.0x/82.3% versus 1.8x/54.3%.
  - About 1.7x the throughput of CLLM at the same scale.
  - No independent replication.
  — [GitHub](https://github.com/hao-ai-lab/JacobiForcing); [arXiv 2512.14681](https://arxiv.org/pdf/2512.14681); [Snowflake blog](https://www.snowflake.com/en/blog/engineering/jacobi-forcing-casual-parallel-decoding/); [ICML poster](https://icml.cc/virtual/2026/poster/64353)
- **[MEASURED-SMALL, 8B MoE; vendor preview] ZAYA1-8B-Diffusion-Preview (Zyphra).**
  - Conversion from the AR MoE base took about 600B tokens of diffusion mid-training at 32k, then about 500B tokens of context extension to 128k, then diffusion SFT.
  - It drafts 16 tokens per block.
  - The lossless sampler accepts with min(1, p/q), using the AR distribution p against the diffusion distribution q. It reports "4.6x", which the page calls both an acceptance rate and a speedup.
  - The lossy mixed-logit sampler reports 7.7x.
  - No wall-clock latency figures, and no discussion of expert-union cost.
  — [Zyphra](https://www.zyphra.com/our-work/zaya1-8b-diffusion-preview)
- **[CLAIM, abstract] SDAR (2510.06303).** Adapts AR models, dense and MoE including a 30B MoE, to blockwise diffusion with a "brief adaptation" stage, and claims parallel speedup without accuracy loss. — [arXiv 2510.06303](https://arxiv.org/pdf/2510.06303v3)
- **[MEASURED-SMALL] CLLMs (2024).** Consistency distillation on Jacobi trajectories; the earlier baseline for this line of work. — [arXiv 2403.00835](https://arxiv.org/pdf/2403.00835)
- **[CLAIM] Related diffusion-LM speed work.** D2F (ICLR 2026) claims "faster-than-AR" dLLM inference. — [GitHub D2F](https://github.com/SJTU-DENG-Lab/Discrete-Diffusion-Forcing). Free Draft-and-Verification (2510.00294) is a lossless parallel decoder for dLLMs. — [arXiv 2510.00294](https://www.arxiv.org/pdf/2510.00294)

### Inferences
- **[EXTRAPOLATION]** Every parallel-decoding method is a multi-token forward. On a fine-grained MoE at BS=1, the cost of a 16-token block is driven by the *union* of routed experts. With 6 of 192 experts per token and the Cohere-like overlap of about 0.38, a 16-token block could touch a large fraction of all experts and approach a full sweep of expert bytes. The "tokens per forward" figures (4-5x) therefore will not convert into a 4-5x wall-clock gain here. Plausibly it will be far less on chat.
- This is the same wall as Q1, and it explains why the 4x results come from **dense 7B** models and from code and math, which have long predictable spans.
- **[EXTRAPOLATION]** Conversion cost: ZAYA spent about 1.1T tokens at 8B. A Jacobi-Forcing-style fine-tune on a 1T-class REAP model, even at 1-5B tokens, is far beyond what one Thor can train. Treat this as not feasible locally.
- The native DSpark/MTP head is already the cheap version of this idea.

### Gaps
- I found no AR-to-diffusion or Jacobi conversion of any model above 30B, and no MoE expert-union measurement for any parallel-decoding method.
- I did not verify the wall-clock figures for SDAR or ZAYA.

---

## Q3. Can a very cheap drafter (expert subset, shared expert only, layer skip) be verified at near-zero cost?

### Takeaway
Self-speculation from an expert subset is now an active edge-MoE topic. **DraftExpert** builds its drafter from the shared expert, the top-1 routed expert and one trained draft expert per layer. It reaches 84-87% acceptance and 1.45x, but only on DeepSeek-V2-Lite and Moonlight-16B in *offloaded* setups. No paper shows such drafters beating a dedicated MTP/DSpark head when all experts are resident. The verify cost, not draft cost, is the binding term on this system.

### Cited Findings
- **[MEASURED-SMALL, offload only] DraftExpert (2607.24434).**
  - The drafter is fixed-footprint: shared expert, plus top-1 routed expert, plus one self-distilled "draft expert" per layer.
  - Acceptance is 84-87%. Decode improves 1.45x on average. Prefetch hit rate is 86-88%.
  - Tested on DeepSeek-V2-Lite and Moonlight-16B-A3B, with CPU-to-GPU and Flash-to-NPU offload.
  - It verifies exactly. There are no results with all experts resident.
  — [arXiv 2607.24434](https://arxiv.org/abs/2607.24434)
- **[CLAIM] Other self-speculative MoE variants.**
  - SpecMoE (2604.10152) drafts from 4 subset experts per MoE block. — [arXiv 2604.10152](https://arxiv.org/html/2604.10152)
  - SS-MoE drafts from a routed-expert subset and uses confidence-adaptive verify length. — [ACM DOI](https://doi.org/10.1145/3774904.3792218)
  - S²-MoE (2608.15018) compares against layer-skip self-speculation, with layers chosen by Bayesian optimization. — [arXiv 2608.15018](https://arxiv.org/html/2608.15018)
- A pure "shared-expert-only" drafter: I found **no** published result.

### Inferences
- **[EXTRAPOLATION]** The project already has a native drafter with high acceptance, so a cheaper drafter buys almost nothing. Its draft cost is already small next to verify; Cohere measures drafter overhead at about 14% of one target step. The upside lies on the verify side (Q1) or in fewer bytes per token (Q4/Q5).
- A shared-plus-top-1 drafter is still worth a cheap test as a **zero-training baseline** for acceptance on chat. If it reaches about 80% and uses only about 1/6 of routed bytes, it could replace the draft lm_head cost noted in prior GLM work.

### Gaps
- No result for DeepSeek-V3/V4-scale self-speculation, and no measurement with all experts resident.

---

## Q4. Lossy acceleration with small, bounded quality cost: router top-k reduction, expert skipping, early exit

### Takeaway
The strongest 2026 finding:
- **Halving active experts is nearly free if renormalization is fixed.** Activate k₁ experts but normalize by the probability mass of k₂ > k₁. On Qwen3.5-397B-A17B (10→5 experts) MMLU drops 0.55 points; on Qwen3.6-35B-A3B (8→4) it drops 0.35 points, against 4.65 points with naive renormalization. This needs zero training.
- A controlled comparison (2610.28369) finds that **no dynamic skipping method beats a static top-k reduction** at matched measured K, and that generative tasks collapse below about K_eff=2.
- Wall-clock gains are diluted by attention and non-expert bytes. Even 8→1 experts gave only 1.22x at BS=1 on Qwen3-30B-A3B.

### Cited Findings
- **[MEASURED-SCALE, 397B; abstract numbers] "Training-Free Halving of Activated Experts" (2609.04575).**
  - Standard renormalization ties expert output strength to the training top-k.
  - The fix is a single integer k₂: normalize by the top-k₂ probability mass while activating only k₁ experts.
  - Results:

    | Model | Experts | MMLU drop, standard renorm | MMLU drop, with k₂ |
    |---|---|---|---|
    | Qwen3.6-35B-A3B | 8→4 | −4.65 | −0.35 (k₂=16) |
    | Qwen3.5-397B-A17B | 10→5 | not given | −0.55 |

  - Removing renormalization entirely is catastrophic.
  - Perplexity and downstream accuracy can prefer different k₂ values.
  - No wall-clock numbers. Only MMLU is reported in the abstract, with no generative tasks.
  — [arXiv 2609.04575](https://arxiv.org/abs/2609.04575)
- **[MEASURED-SMALL] "Don't Skip the Baseline" (2610.28369).**
  - Compares NAEE, DiEP, MoDES and Ada-K against a static top-k baseline on Qwen3-30B-A3B and Gemma4-26B-A4B; no DeepSeek model is tested.
  - No dynamic method consistently beats static top-k at matched K_eff.
  - GSM8K and HumanEval collapse below K_eff≈2 on Qwen3.
  - At K_eff≈1, about 50-65% of capability is lost.
  - Wall-clock on vLLM, single A100, BS=1: K 8→1 gives 111→135 tok/s (1.22x), with ITL −18%. Attention, KV-cache and non-expert bytes are untouched.
  — [arXiv 2610.28369](https://arxiv.org/html/2610.28369v1)
- **[CLAIM] Other skipping and routing methods.**
  - EAT reports history-aware top-p with about 25% fewer activated experts. — [arXiv 2609.33614](https://arxiv.org/html/2609.33614v1)
  - An elbow-based k plugin reports −5.3% latency on OLMoE. — [arXiv 2608.04401](https://arxiv.org/pdf/2608.04401)
  - ACE is calibration-free expert skipping that argues gates reflect relative preference, not contribution. — [arXiv 2609.05228](https://arxiv.org/pdf/2609.05228)
  - SERE (ICLR 2026) re-routes to similar experts and was tested on DeepSeek-V2-Lite. — [SERE](https://proceedings.iclr.cc/paper_files/paper/2026/file/5d07eb2335d88b1dfcefcc3d2da2a230-Paper-Conference.pdf)
  - Inference-time top-k changes cause distribution shift. — [ACL Findings EMNLP 2026](https://aclanthology.org/2026.findings-emnlp.2320/)

### Inferences
- **[EXTRAPOLATION]** Effect on this model: let f be the routed-expert share of the 8.57 GB/token. Top-6→top-3 with k₂ renormalization gives a speedup of 1/(1−f/2). The DSV4-0731 note in project memory had attention (MLA) at 41% of bytes per token, so routed experts are perhaps 40-55%. That gives about **1.25-1.4x decode** with no training and no new kernel: it is just a router change.
- **[EXTRAPOLATION]** It *also* shrinks the spec-decode expert union by about 2x, which compounds with Q1. This is the highest expected-value, lowest-cost item found. It must be gated on generative evals, not MMLU, because of the 2610.28369 collapse finding.
- **Interaction risk.** The project's router is noaux_tc: the bias selects experts, and an unbiased sigmoid weights them. The k₂ normalization trick must be applied to the sigmoid weights, not to the biased selection scores.
- **Interaction risk.** REAP-192 has already removed half the experts. The remaining experts may be less redundant, so the top-k slack could be smaller than on unpruned Qwen. This needs a measurement.

### Gaps
- No DeepSeek-V3/V4-family top-k reduction study found.
- No early-exit or layer-skip result at 100B+ MoE scale with generative quality numbers.

---

## Q5. Can lossless entropy coding of low-bit weights, decoded in-kernel, push decode past the roofline?

### Takeaway
Lossless coding of **already-low-bit** weights has small headroom. Formats that are already compact (FP4, INT4, K-quants) carry little redundancy, so measured system gains come from bf16 or int4 at datacenter scale and are framed as throughput or batch-size gains, not as single-stream decode speedup. The more interesting black swan is **EntQuant (ICML 2026)**: keep an 8-bit base format, optimize the scales to *minimize entropy*, then ANS-code to about 2 bits per parameter. That is a quality-per-byte route, not a speed route, and it requires an ANS decoder in the GEMV kernel.

### Cited Findings
- **[CLAIM, abstract; models 1.5B-405B] "Approaching Shannon Bound" (2606.15789).**
  - Reports that effective entropy is 2-10x below the stored bitwidth across formats from bf16 to int4 and AWQ/SQ8. "Up to 10x" is an entropy bound, not a measured ratio.
  - Uses a tile-level, GEMM-aligned ANS decoder, landing within 0.01-0.1 bit of the Shannon limit.
  - In SGLang:
    - Qwen-14B: max batch size 47→75, up to 1.2x throughput.
    - Mixtral-176B: batch size 20→95, up to 1.6x throughput.
  - No batch-1 latency figures.
  — [arXiv 2606.15789](https://arxiv.org/abs/2606.15789)
- **[CLAIM, secondary summary] Fixed low-bit formats compress less.** The redundancy that byte-oriented coders exploit (bf16 exponents) disappears after the cast to a fixed low-bit format. — [Transforms for LLM Quantization 2608.25188](https://arxiv.org/pdf/2608.25188)
- **[MEASURED-SCALE, production; vendor] Cloudflare Unweight.** 15-22% lossless (bit-exact) compression of bf16 weights, with a reconstructive matmul kernel that rebuilds bf16 in shared memory. — [Cloudflare](https://blog.cloudflare.com/unweight-tensor-compression/)
- **[CLAIM, ICML 2026] EntQuant / "Float8@2bits".** Keeps an 8-bit base, optimizes per-channel scales for low entropy, then ANS-codes to about 2 bits per parameter, data-free. — [arXiv 2601.22787](https://arxiv.org/pdf/2601.22787); [paper note](https://en.papernotes.org/ICML2026/model_compression/float82bits_entropy_coding_enables_data-free_model_compression/)
- **[CLAIM] Decode-placement taxonomy.** NeuZip and DFloat11 decompress into a scratch tile. ZipServ fuses a fixed-length bitmap decode into the GEMM mainloop with no scratch tensor. — [ICLR 2026 note](https://en.papernotes.org/ICLR2026/model_compression/to_compress_or_not_pushing_the_frontier_of_lossless_genai_model_weights_compress/)

### Inferences
- **[EXTRAPOLATION]** On 2-3 bit K-quant experts, residual entropy savings are probably about 5-15%; I found no direct measurement on K-quants. Thor's compute is also tight: 20 SMs must decode about 214 GB/s worth of rANS symbols. Treat the ceiling as about 1.05-1.15x, and below 1.5x in any case. This is not a black swan for speed.
- **[EXTRAPOLATION]** EntQuant matters for **quality per byte**. If 8-bit-base weights at about 2 effective bits beat 2-3 bit K-quants on quality at equal bytes, that would be a step change. It is unverified at MoE scale, and the decode kernel cost on Thor is unknown.

### Gaps
- No measured entropy for Q2_K, IQ2, IQ3 or MXFP4 expert tensors.
- No batch-1 single-stream speedup result for any in-kernel entropy decoder.
- No EntQuant result on a MoE above 100B.

---

## Q6. Hardware tricks on unified-memory SoCs: CPU/DLA/PVA concurrency, LPDDR effects, more than 90% of peak, L2 residency control

### Takeaway
I found no published measurement of concurrent CPU and GPU decode on Thor, of LPDDR5X bank or page tuning, or of L2 persistence for LLM weights on a unified-memory SoC. Reference points:
- NVIDIA's own TensorRT-Edge-LLM batch-1 numbers imply roughly 70% of peak bandwidth (my arithmetic), so this project's 78% is already above vendor.
- L2 persistence helps only when the persisted set fits the set-aside and is not evicted by streamed weights. A llama.cpp PR saw a *slowdown*.

### Cited Findings
- **[MEASURED-SCALE, vendor] TensorRT-Edge-LLM batch-1 decode on Thor, 2048-token KV.**

  | Model | Format | tok/s |
  |---|---|---|
  | Qwen3-30B-A3B | NVFP4 | 84.1 |
  | Qwen3.5-27B | NVFP4 | 14.5 |
  | Gemma-4-31B | FP8 | 7.4 |

  — [TRT-Edge-LLM 0.10.1](https://nvidia.github.io/TensorRT-Edge-LLM/0.10.1/user_guide/performance/performance-benchmarks.html); [0.11.0](https://nvidia.github.io/TensorRT-Edge-LLM/0.11.0/user_guide/performance/performance-benchmarks.html); [0.9.1](https://nvidia.github.io/TensorRT-Edge-LLM/0.9.1/user_guide/performance/performance-benchmarks.html)
- **[CLAIM] Thor bandwidth spec.** A third-party blog gives 273 GB/s of 256-bit LPDDR5X for Thor and Spark; it is unverified for Thor alone. The project measures about 228 GB/s. — [openclawdc](https://openclawdc.com/blog/jetson-thor-vs-dgx-spark/)
- **[MEASURED-SMALL, community] JetsonHacks.** Thor got "3.5x faster after launch" through software updates, which shows the software stack matters. — [JetsonHacks](https://jetsonhacks.com/2025/12/01/the-jetson-agx-thor-is-3-5-times-faster-after-launch-but-how/)
- **[DOC] L2 persistence semantics.** Benefits fade once the persisting volume exceeds the set-aside, and streaming accesses use the set-aside only when it is idle. — [CUDA Programming Guide, L2 cache control](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/l2-cache-control.html)
- **[MEASURED-SMALL] Contrary evidence on L2 persistence.** A llama.cpp PR's opt-in L2 persistence of recurrent states showed a small slowdown on consumer cards. Weights streamed between layers evict the persisted state. — [torad-labs/llama.cpp PR #82](https://github.com/torad-labs/llama.cpp/pull/82)

### Inferences
- **[EXTRAPOLATION]** 25 tok/s × 8.57 GB = 214 GB/s. Even 100% of 228 GB/s gives only 26.6 tok/s, and 100% of the 273 GB/s spec gives 31.9 tok/s, so bandwidth tricks are capped at **1.06-1.28x**. Pinning the router, shared expert and lm_head in L2 would save only their bytes; they are a small fraction of 8.57 GB.
- The CPU cores share the same DRAM, so CPU-side expert compute cannot add bandwidth, only contend for it.

### Gaps
- No public Thor measurements of achievable STREAM-style GPU bandwidth, of L2 size and set-aside behavior, or of CPU+GPU contention.
- No DLA or PVA use for LLM layers found.

---

## Q7. Quality-per-byte black swans: QAT healing, ternary conversion of large MoEs, expert distillation, frontier quality under 100 GB

### Takeaway
There are no 2026 public results for QAT or PV-tuning-style healing of a DeepSeek-class MoE at 2 bits. Ternary PTQ has reached 235B (CAT-Q, claim only), but measured ternary results show real losses: Qwen3-4B fell from 64.5% to 54.7%. Generic-text calibration collapses math and code on larger MoEs, which echoes this project's corpus-imbalance finding. The practical 2-bit recipe in the community is still mixed precision: routed experts in IQ2/Q2_K, everything else in Q8. I found no credible claim of frontier quality under 100 GB.

### Cited Findings
- **[CLAIM] Ternary post-training quantization.**
  - CAT-Q (June 2026): post-training ternarization on 10 LLMs from 1.7B to 235B, dense and MoE, with 512 calibration samples; claims to beat BitNet b1.58 v1 and v2. — [CAT-Q](https://www.alphaxiv.org/abs/2606.26650.md)
  - ScaleQ-1.58 (Intel, August 2026): ternary PTQ for reasoning LLMs. Qwen3-4B gains 8.97 pp absolute over the ternary baseline. On a larger MoE, generic-text calibration "leads to model collapse on most mathematics and coding tasks". — [arXiv 2608.01078](https://arxiv.org/pdf/2608.01078)
- **[MEASURED-SMALL] Ternary Qwen3-4B (September 2026).** The real stored rate is 1.641 bpw. Accuracy falls from 64.5% to 54.7% across 10 capability comparisons. — [emergentmind 2609.01962](https://www.emergentmind.com/papers/2609.01962)
- **[MEASURED-SMALL] Ternary fine-tuning normalization.** RMSNorm before each linear layer matched or beat complex KD pipelines in ternary fine-tuning. — [arXiv 2505.08823](https://arxiv.org/abs/2505.08823v1)
- **[MEASURED-SMALL] BitsMoE (2606.00079).** Spectral-space bit allocation without fine-tuning. Reaches 64.29% average over 7 tasks at 2-bit on Qwen3-30B-A3B. — [arXiv 2606.00079](https://arxiv.org/html/2606.00079)
- **[MEASURED-SMALL] ECO (2601.22101).** On DeepSeek-MoE-16B at INT4, quantized training without master weights diverges under RTN or SR. ECO fixes this. — [arXiv 2601.22101](https://arxiv.org/pdf/2601.22101)
- **[MEASURED-SCALE, vendor, FP8 only] Fireworks.** Offers QAT for DeepSeek V3/R1. — [Fireworks](https://fireworks.ai/blog/deepseek-fine-tuning)
- **[CLAIM, community; weak evidence] antirez on DeepSeek V4 Flash at 2-bit.** Quantize only the routed experts (gate and up in IQ2_XXS, down in Q2_K), and keep shared experts, projections and routing at Q8. — [antirez on X](https://x.com/antirez/status/2048885632869523869)
- **[CLAIM] Colla-Q (2609.18131).** Minimax precision balancing across collaborating experts. — [arXiv 2609.18131](https://arxiv.org/pdf/2609.18131)

### Inferences
- **[EXTRAPOLATION]** On a 192-expert REAP model, the largest quality-per-byte lever with evidence behind it is still *equal-bytes reallocation*: drop more experts and give the rest more bits, or use mixed precision by expert importance. Ternary experts are high-risk, with a measured −10 pp at 4B and collapse reports at MoE scale.
- Calibration-domain sensitivity in ScaleQ independently corroborates the project's REAP corpus-imbalance note.

### Gaps
- No ParetoQ, PV-tuning or EfficientQAT results on a model above 100B MoE turned up in these searches.
- No verified "frontier quality under 100 GB" result.

---

## Q8. Recent 2026 items the mainstream has not absorbed (consolidated ranking for this system)

### Takeaway
Ranked by expected decode ceiling × evidence × cost on Thor. All ceilings are **[EXTRAPOLATION]**.

| Rank | Technique | Evidence | Ceiling on this system | Main cost or risk |
|---|---|---|---|---|
| 1 | k₁/k₂ renormalized top-6→top-3 | Measured on a 397B model, MMLU only | About 1.25-1.4x AR, plus a smaller verify union | Generative-task collapse; REAP interaction |
| 2 | Expert-locality-aware spec decode (EcoSpec/EVICT tree costing; expert-major verify kernel for L2 reuse) | Claim (NeurIPS-26 abstract) and production Cohere numbers | Could move chat from about 1x toward the math-like 1.3-1.6x | Kernel work |
| 3 | AcceptMoE restricted verifier | Claim (abstract); −0.27 pp | About 1.29x over EAGLE-3 with all experts resident | Lossy |
| 4 | EntQuant-style entropy-shaped 8-bit base at about 2 bpw | Claim (ICML-26) | A quality-per-byte gain, not speed | ANS-in-GEMV on 20 SMs is untested |
| 5 | Jacobi Forcing or diffusion conversion | Measured at 7-8B | Above 3x on paper; expert union and training cost make it infeasible here | Training cost |
| 6 | Prefetch and caching | Measured, but against offload baselines | About 1.0x (no offload on Thor), except Engram NVMe rows | n/a |
| 7 | L2 persistence and bandwidth tuning | Doc plus one negative PR | Capped at 1.06-1.28x by the roofline arithmetic | n/a |

### Cited Findings
- See the per-question citations above. The most recent items are 2610.38981 (2026-10-28, NeurIPS 2026), 2610.28369, 2609.04575 and 2608.02989.

### Inferences
- **[EXTRAPOLATION]** Combined best case: item 1 (about 1.3x), then items 2 and 3 applied to the smaller union (about 1.3x), gives roughly 1.7x on chat. Each factor must be gated separately. Nothing found offers a lossless chat speedup of 2x or more for a fine-grained MoE at BS=1 when all experts are resident.

### Gaps
- No source on DeepSeek-V4-family routing statistics, mHC or DSA interactions with any of these techniques, or Engram-specific acceleration.
- I could not read several 2026 PDFs in full (2609.04575, 2606.15789), so their per-task tables are unverified.
