# Speculative decoding and draft-head training for batch-1, bandwidth-bound MoE (DSV4.1-Flash REAP-192 on Jetson AGX Thor)

Scope note: researched 2026-10-30. Many sources are 2026 arXiv preprints, often from a single group and not reproduced. Hardware is given for every number. Unless a line says "batch 1", the number was not measured at batch 1. Lines marked **[EXTRAPOLATION]** are my own arithmetic, not anything measured. Internal measurements are cited from the project memory files (local paths).

Shared cost model, used in every section (from the Cohere paper, [Ranjan et al. 2610.38981](https://arxiv.org/html/2610.38981)): target efficiency is TE = T_AR / T_SD ≈ (C_dense + k·C_expert) / (C_dense + unique_sd·C_expert). Spec decode pays off when E[tokens committed per step] > T_verify(k+1)/T_AR + T_draft/T_AR.

---

## 1. SOTA drafters 2025–2026: acceptance length and batch-1 speedup, especially on MoE

### Takeaway
Block-parallel drafters lead in 2026. DFlash, DFlash 2, DSpark and P-EAGLE all propose a whole block in one drafter pass. Each beats autoregressive drafters (EAGLE-3/3.1, MTP chains) by about 2x on dense targets: τ≈6.5 against EAGLE-3's ≈3.0–3.5 on Qwen3-8B. On MoE targets the advantage shrinks noticeably: 2.6–3.5x at concurrency 1 on Qwen3-Coder-30B-A3B (B200), and only 1.3–1.7x on GPT-OSS-120B at concurrency 8. DeepSeek V4 already ships a DSpark block drafter, so the drafter architecture is not the gap for us. The gap is MoE verify cost (Section 3).

### Cited Findings
- **DFlash** (Chen, Liang, Liu; arXiv 2602.06036, ICML 2026). Block-diffusion drafter, one denoising step, block 16, 5 draft layers. Target features from 5 layers are injected into the draft's KV. Results on H200 with the Transformers backend, batch 1, 8-benchmark average:
  - Qwen3-8B at T=0: DFlash 4.86x / τ 6.49. EAGLE-3 (tree 16) gets 1.76x / τ 2.96; EAGLE-3 (tree 60) gets 2.02x / τ 3.40.
  - Qwen3-8B at T=1: DFlash 4.03x / τ 5.48.
  - Weakest task is MT-Bench (chat): 2.75x / τ 4.24, against 5.1–5.5x on math and code.
  - — [DFlash HTML](https://arxiv.org/html/2602.06036)
- **DFlash on an MoE target** (SGLang, single B200, FA4), speedup at concurrency 1 → 32 on Qwen3-Coder-30B-A3B:
  - HumanEval 3.5x → 3.1x (τ 8.09); LiveCodeBench 2.6x → 2.3x (τ 6.42); MBPP 3.2x → 3.1x (τ 7.23).
  - For comparison, dense Qwen3-8B on Math500 gets 5.1x at c=1 with τ 8.01.
  - At c=8 on Math500 / HumanEval / MT-Bench: Qwen3.5-35B-A3B 2.4x / 2.3x / 1.7x; GPT-OSS-120B 1.6x / 1.7x / 1.3x. Native MTP on Qwen3.5 gets 1.2–1.7x in the same table.
  - — [DFlash HTML, Tables 3 and 11](https://arxiv.org/html/2602.06036)
- **DFlash ablations:**
  - 3, 5 and 8 draft layers give τ 5.64, 5.99 and 6.33 on Math500, but speedup peaks at 5 layers (4.71x) because the extra layers cost draft time.
  - 5 target-feature layers beat 3 (τ 5.64 vs 5.38).
  - Train and test block size should match: train 16 / test 8 gives τ 5.09, against 6.33 for 16/16.
  - KV injection beats input fusion (EAGLE-style): τ 4.2 vs 3.5 on GSM8K.
  - — [DFlash HTML](https://arxiv.org/html/2602.06036)
- **DFlash 2** ("Keep Drafting Parallel", Aug 2026). Checkpoints exist for Muse-Glimmer-30B and Qwen3.8-27B. The public DFlash collection covers Qwen3.5/3.6 MoE (35B-A3B, 122B-A10B, 397B-A17B), Kimi K2.5–K2.7, MiniMax M2.5/M2.7, GLM 5.1 and GPT-OSS, but **no DeepSeek models**. llama.cpp support is in PR #27342. The README advises block_size ≤ 5 for quantized targets or drafts, because verify width hurts quantized matmul. — [z-lab/dflash GitHub](https://github.com/z-lab/dflash)
- **DeepSeek DSpark** (V4 family). A parallel draft backbone feeds a small sequential head, plus a Markov previous-token bias (rank 512) and a per-position **confidence head**.
  - V4-Pro config: block 5, taps target layers 58/59/60 (the last three of 61).
  - Training code is DeepSpec (MIT). It regenerates target outputs, caches them, then trains Eagle3, DSpark or DFlash drafters. Its Qwen3-4B example needs about a 38 TB cache.
  - No acceptance or speedup numbers in this secondary source. — [lilting.ch write-up of the V4-Pro-DSpark card](https://lilting.ch/en/articles/deepseek-v4-pro-dspark-speculative-decoding)
  - Secondary press claims "up to ~30% longer accepted length going from draft length 4 to 16" and "60–85% per-user gain over MTP-1". **Unverified against a DeepSeek primary source.** — [search summary citing gmagazine / MarkTechPost](https://gmagazine.it.com/?p=454)
  - A community V4-Flash DSpark GGUF drafter reports 52–91% acceptance depending on workload. — [bleysg GGUF card](https://huggingface.co/bleysg/DeepSeek-V4-Flash-DSpark-drafter-GGUF)
- **Jetson AGX Thor** (closest hardware to ours), from the NVIDIA Jetson AI Lab tutorial:
  - Qwen3.8-27B (dense) with MTP depth 3: about 13 → 35 tok/s (about 2.7x), llama.cpp.
  - Nemotron 3.5 Lightning 30B-A3B (MoE) NVFP4 with DSpark depth 4 on vLLM: 115 tok/s. No baseline was given, so no speedup can be computed.
  - It recommends DSpark depth about 4 and DFlash about 15, but says to test lower when "parallel verification activates extra MoE experts".
  - — [Jetson AI Lab](https://www.jetson-ai-lab.com/tutorials/speculative-decoding)
- **EAGLE 3.1** (May 2026, EAGLE + vLLM + TorchSpec). Adds FC normalization on the target hidden-state inputs and feeds post-norm hidden states to the next step. Backward-compatible with EAGLE-3 checkpoints. No speedup numbers in the snippet. — [vLLM blog](https://vllm.ai/blog/2026-05-26-eagle-3-1)
- **P-EAGLE** (Amazon): parallel multi-token drafting on top of EAGLE-3, in vLLM Speculators. — [Red Hat Developers, 2026-09-03](https://developers.redhat.com/articles/2026/09/03/speeding-llm-inference-p-eagle-vllm-speculators)
  - A secondary blog says vanilla EAGLE-3 throughput peaks at K=3 in vLLM sweeps while parallel drafters keep improving up to K=7. **Unverified.** — [DEV Community](https://dev.to/mr_manushukla/vllm-speculative-decoding-in-2026-p-eagle-vs-dflash-vs-dspark-for-faster-cheaper-llm-serving-243c)
- **GRIFFIN** (NeurIPS 2025) adds token-alignable training to HASS. It reports +8% acceptance length and +7% speedup over HASS, and +20% / +18% over EAGLE-2, on LLaMA2/3 and Vicuna (dense). — [GRIFFIN arXiv 2502.11018](https://arxiv.org/html/2502.11018v3)
- **On MoE, EAGLE-3 and native MTP are weak at batch 1** (EcoSpec, 8×H200, HF Transformers, batch 1, γ=4, greedy):
  - EAGLE-3 on Qwen3-235B-A22B: 1.22x, accept length 2.41.
  - EAGLE-3 on GPT-OSS-120B: 1.14x, accept length 1.88.
  - Native MTP on DeepSeek-V3.1: 1.10x, accept length 2.85.
  - — [EcoSpec 2607.12696](https://arxiv.org/html/2607.12696)
  - EVICT (A100, batch 1) measured EAGLE-3 τ 3.65 on Qwen3-30B-A3B and 3.57 on Qwen3-235B. — [EVICT 2605.00342](https://arxiv.org/html/2605.00342v1)
- **Our internal measurements:**
  - Native DSpark/MTP in our CUDA engine: about 1.6x on math, 0.98x on chat. Spec math says 1.6x, chat 0.98x. — [/home/patrickd/.claude/projects/-home-patrickd/memory/v41-cuda-zenith-mandate.md]
  - On the previous GLM-5.3 target, spec decode was bounded at 1.27x even at perfect acceptance (MoE routing wall). — [/home/patrickd/.claude/projects/-home-patrickd/memory/glm53-spec-decode-moe-wall.md]

### Inferences
- The best published MoE batch-1 number with a lossless drafter is DFlash at 3.5x on Qwen3-Coder-30B-A3B (B200). That target has only 3B active parameters with fp16 experts, and on a B200 its batch-1 step is far from pure expert-bandwidth bound. On a bandwidth-bound target the speedup should fall towards the EcoSpec/EVICT regime (1.1–1.4x over AR for AR-style drafters). **[EXTRAPOLATION]**
- Swapping DSpark for DFlash or EAGLE-3 is unlikely to step-change anything by itself. DSpark is already a block drafter, and our chat gap (0.98x) is a verify-cost problem. Raising τ only helps if verify cost per committed token falls.
- DFlash's large-block advantage relies on cheap verify width. Its own README advises block ≤5 for quantized targets, which is our case (K-quant experts).

### Gaps
- No primary DeepSeek DSpark benchmark table was retrieved. The V4 paper (arXiv 2606.19348) and the DeepSeek-V4-Pro-DSpark model card are the places to check.
- No verified numbers this session for PARD, Falcon, Jakiro, Hydra or HASS-on-MoE.
- No measured DFlash or EAGLE-3 result on any DeepSeek-V3/V4 target at batch 1 on bandwidth-bound hardware.

---

## 2. Draft-head fine-tuning for a modified (pruned and quantized) target

### Takeaway
The consensus recipe is to regenerate the training data with the *modified* target and train on that. This is self-distillation; fixed-dataset tokens are worse. Train the drafter against the frozen quantized target's hidden states. Expect a large recovery when the target has drifted:
- EDA: τ 4.37 → 4.79 on a fine-tuned target, at 61% of full-retrain cost.
- Our own GLM-5.3 measurement showed every quant drifts the hidden states enough to need its own head (cos 0.89–0.93).

Losses in current block drafters are plain CE with position decay. KL is not essential. Data scale is about 100K–800K target-generated samples.

### Cited Findings
- **EDA** (arXiv 2603.09527). A drafter trained for a base model loses much of its acceptance on a fine-tuned target. EDA regenerates the data with the new target and adapts a lightweight private component. On Qwen2.5-7B to Qwen2.5-Math-7B it reaches τ 4.79 vs 4.37 for baseline adaptation, at 60.8% of full-retraining cost. — [EDA](https://arxiv.org/html/2603.09527v1)
- **"Quantize the Target, Quantize the Drafter"** (arXiv 2607.04244; Qwen3.5-4B, A10G). Recovers the INT4 target with quantization-aware distillation, then trains the drafter in stages on the frozen *quantized* target. It also quantizes the drafter and gives it sliding-window attention to cut draft overhead. — [alphaXiv 2607.04244](https://www.alphaxiv.org/abs/2607.04244)
- **SD2** (arXiv 2504.08838, ICML 2025). Prunes the drafter (fine-grained sparsity beats layer pruning) and recovers it by fine-tuning on self-distilled target outputs. — [SD2](https://arxiv.org/pdf/2504.08838v2)
- **DFlash recipe** (a template for block drafters):
  - Data: about 800K samples (Nemotron Post-Training V2 + CodeAlpaca), with responses generated by the target. Ablations use 100K samples.
  - Schedule: 6 epochs, AdamW, lr 6e-4, cosine with 0.04 warmup, seq len 3072–4096, 512 random anchors per sequence.
  - Loss: CE with exponential position decay w_k = exp(−(k−1)/γ), with γ=7 for block 16. No KL.
  - Random anchor sampling beats standard block construction: τ 5.64 vs 4.94 on Math500.
  - — [DFlash HTML](https://arxiv.org/html/2602.06036)
- **DeepSpec** (DeepSeek) works the same way: regenerate and cache target outputs, then train the DSpark, Eagle3 or DFlash head on the cache. The default example cache is about 38 TB (Qwen3-4B). — [lilting.ch](https://lilting.ch/en/articles/deepseek-v4-pro-dspark-speculative-decoding)
- **DraftExpert** (arXiv 2607.24434, an MoE self-drafter). Distills residual, logit/token **and router-agreement** losses from the frozen target MoE into one resident draft expert per layer. Draft acceptance reaches 84–87%, and router agreement allows 86–88% prefetch hit rates for verifier experts (DeepSeek-V2-Lite, Moonlight-16B-A3B). — [DraftExpert PDF](https://arxiv.org/pdf/2607.24434)
- **Training-time test (TTT, EAGLE-3).** Training on the drafter's own unrolled predictions remains standard for AR drafters; EAGLE 3.1 adds input normalization on top. — [vLLM EAGLE 3.1 blog](https://vllm.ai/blog/2026-05-26-eagle-3-1). I did not retrieve TTT ablation numbers this session.
- **Internal evidence:**
  - GLM-5.3 IQ3_M / IQ4_XS / NVFP4 hidden states sit at cos 0.89–0.93 from each other, against 0.999 for a same-quant control. Every quant needs its own draft head. — [/home/patrickd/.claude/projects/-home-patrickd/memory/glm53-quant-hidden-state-drift.md]
  - Even so, the GLM-5.3 native head was 72.3% accepted without fine-tuning; the long pole was the server, not the head. — [glm53-nvfp4-mtp-gate2.md]
  - The user's published DSpark head fine-tuned to a V4-Flash REAP variant reports about 2x acceptance on reasoning, explanation and code, with regressions elsewhere. — [HF patrickbdevaney/dspark-mtp-draft-head-s3recap-p25-b0.1](https://huggingface.co/patrickbdevaney/dspark-mtp-draft-head-s3recap-p25-b0.1)

### Inferences
- For REAP-192 at 2–3 bit, the training targets should be the pruned + quantized model's *own* hidden states (taps 37/38/39, per [v41-dspark-gguf-tap-offset.md]) and its own sampled outputs.
- Router-agreement distillation (DraftExpert) is the one loss that directly attacks our bottleneck. A head that predicts the target's top-6 experts lets verify run expert-budgeted (Section 3) with little quality loss. **[EXTRAPOLATION]**
- Budget: 100K samples recovered most of DFlash's τ in the ablations. Generating 100K × ~1–2K tokens on Thor at 25 tok/s is roughly 46–90+ days of wall-clock. Generation, not training, binds. Mitigations: reuse cached hidden states at teacher-forced prefill speed (549.8 tok/s at 8k measured, about 20x decode), with sampled outputs only for a small slice. **[EXTRAPOLATION]**

### Gaps
- No paper found that measures acceptance loss for a drafter after *expert pruning* of the target specifically, nor for combined prune + quant.
- No published KL-vs-CE head-to-head for block drafters was found this session.

---

## 3. MoE-specific spec decoding: the expert-union cost, and methods that reduce it

### Takeaway
This is the binding term for us. Verifying k+1 tokens reads the union of their experts. Measured 4-token unions are 2.4x top-k (Qwen3-30B, 128 experts) up to 2.9x top-k (Kimi-K2.5, 384 experts), so verify costs about 2x an AR step for typical expert byte shares.

The 2026 fixes have two forms:
- **Lossless (draft-side)**, which pick fewer or cheaper nodes: EVICT, EcoSpec, Cascade. These give +5% to +25% over EAGLE-3.
- **Lossy (verifier-side)**, which restrict the expert set: MoE-Spec, AcceptMoE. These give +10% to +30%, for −0.27 pp accuracy.

The upside is that real temporal routing overlap makes verify cheaper than an independence model predicts.

### Cited Findings
- **Ranjan et al. (Cohere, arXiv 2610.38981, 28 Oct 2026)**, H100/B200, vLLM, K=3:
  - Unique experts over 4 tokens on Qwen3-30B (top-8 of 128): 19.5 empirical (2.4×k), against 26.9 under independence. Kimi-K2.5 (top-8 of 384) reaches 2.9×k.
  - Temporal overlap cuts unique experts 21–34% versus independence. Mean step-1 overlap is 35.1%, falling to 26.4% at step 3.
  - Batch-1 TE observed vs predicted: Qwen3-235B fp8 0.889 vs 0.517; Qwen3-30B fp8 0.818 vs 0.511; Mixtral 0.488–0.505 vs 0.428.
  - Part of that gap is because the MoE kernels ran at only 45–49% of peak HBM bandwidth at batch 1, and L2 hit rate rose from 5.7% to 27.8% under spec decode.
  - Routing-based rejection predictors are of little practical use: accepted and rejected token overlap distributions overlap about 85%.
  - — [arXiv 2610.38981](https://arxiv.org/html/2610.38981)
- **Cascade / utility-driven SD** (Saxena et al., NVIDIA/GT, arXiv 2506.20675):
  - Draft tokens raise MoE verify time 2–3x, and vanilla spec decode slows decoding by up to 1.5x.
  - Cascade turns spec decode off when utility (token gain / verify cost) is below 1 and otherwise picks K. It limits slowdown to 5% and gains +7–14% over static K (vLLM, 5 MoEs).
  - — [arXiv 2506.20675](https://arxiv.org/abs/2506.20675)
- **EVICT** (arXiv 2605.00342), A100, batch 1, lossless and training-free. Truncates the draft tree before verify:
  - 74.7% fewer tokens verified, but only 32.5% fewer experts activated, and 26.6% lower verify latency.
  - End-to-end over EAGLE-3: 1.21x (Qwen3-30B-A3B; the per-benchmark table averages to about 1.14x), 1.22x (Ling-flash-2.0), 1.25x (Qwen3-235B). Up to 2.35x over AR.
  - — [EVICT](https://arxiv.org/html/2605.00342v1)
- **EcoSpec** (arXiv 2607.12696), 8×H200, batch 1, lossless. Scores nodes by draft probability divided by marginal new-expert cost:
  - Qwen3-235B: 1.22x → 1.36x over AR (experts per layer 23.7 → 20.5).
  - GPT-OSS-120B: 1.14x → 1.31x.
  - **DeepSeek-V3.1 with MTP: 1.10x → 1.15x**, with expert union barely reduced (31.4 → 31.2 per layer for top-8 of 256 at γ=4, about 3.9×k).
  - — [EcoSpec](https://arxiv.org/html/2607.12696)
- **MoE-Spec** (McDanel et al., arXiv 2602.16052), training-free, lossy. Keeps a top-B expert budget per layer during verify, aggregating router mass across the block; routes outside the budget are truncated or substituted. +10–30% throughput over EAGLE-3 at comparable quality (OLMoE, Qwen3-30B-A3B, Mixtral). — [MoE-Spec](https://arxiv.org/html/2602.16052v1)
- **AcceptMoE** (Imperial, arXiv 2608.02989), SGLang, **batch 1**, lossy:
  - Weights router demand by offline commitment probability per draft position. Set size is self-chosen from the effective rank of the demand (23–34 experts for the targets tested).
  - 1.290x over EAGLE-3 with all experts resident (RTX PRO 6000 Blackwell, up to 257 tok/s), and 2.06x under offloading (RTX 5090).
  - Mean accuracy −0.27 pp versus natural routing (per pair −1.83 to +1.22). Beats MoE-Spec by +2.45 pp accuracy at matched budget.
  - — [AcceptMoE PDF](https://arxiv.org/pdf/2608.02989)
- **DraftExpert** (arXiv 2607.24434). Fixed-footprint shared + top-1 + trained draft-expert self-drafter, with confidence–expansion truncation (stop drafting when a low-confidence token would add too many new verifier experts). 1.45x decode on DeepSeek-V2-Lite and Moonlight under offload, lossless verify. — [DraftExpert PDF](https://arxiv.org/pdf/2607.24434)
- **Speculative-Limits** paper (arXiv 2609.22156). Oracle analysis on Qwen3-Coder + EAGLE-3: rejected candidates fall behind a linear boundary in ΔE[Cost]/ΔE[accepted]. No numbers on the abstract page. — [arXiv 2609.22156](https://arxiv.org/abs/2609.22156)
- **MoESD** (NeurIPS 2025). Spec decode on MoE is *least* effective at small batch; on Qwen2-57B-A14B gains peak at medium batch (up to 2.29x), and sparser models widen the useful batch range. — [MoESD](https://arxiv.org/html/2505.19645v4). Contrast: Ranjan 2026 finds very sparse modern MoEs peak at batch 1.

### Inferences: break-even for our system **[EXTRAPOLATION, all lines]**
- Let f be the share of per-token bytes that are routed experts; split our 8.57 GB/token accordingly. Then T_verify(n tokens)/T_AR ≈ (1−f) + f·m(n), where m(n) is the union multiplier (unique experts / 6).
- Our model has top-6 of 192 experts with REAP-concentrated routing. Using the measured 2.4–2.9x at n=4 and about 1.6x at n=2 (step-1 overlap about 0.4):

| f (routed share) | n=2 (k=1) verify / AR | n=4 (k=3) verify / AR |
|---|---|---|
| 0.5 | 1.30 | 1.70–1.95 |
| 0.7 | 1.42 | 1.98–2.33 |
| 0.85 | 1.51 | 2.19–2.62 |

- Expected tokens per step at k=3 with per-position acceptance a: a=0.55 gives 2.02; a=0.72 gives 2.61. At f≈0.7 that is about 1.0–1.3x, which matches our measured 0.98x chat / about 1.6x math and the GLM-5.3 1.27x ceiling.
- At k=1: E = 1+a, so a=0.75 gives about 1.75/1.42 ≈ 1.23x. **k=1 is roughly as good as k=3 for us.**
- The step-change lever is m(n), not τ:
  - With a verifier expert budget capped near m≈1.3 (lossy, MoE-Spec/AcceptMoE style), k=3 at a=0.72 gives about 2.61 / (0.3 + 0.7·1.3) ≈ 2.16x.
  - Lossless node selection (EVICT/EcoSpec) moves m only modestly. On DeepSeek-V3.1 with MTP, EcoSpec moved it 0.6%.
- Our engine already runs at 78% of roofline, so the "better than predicted" slack Ranjan found (MoE kernels at 45–49% of peak) is mostly absent. Expect TE nearer the bytes model than their numbers.
- Engram n-gram rows (NVMe) must be fetched for each of the k+1 verified tokens; they are independent and can be batched in one I/O submission. Check this does not serialize.

### Gaps
- No paper measures union growth with REAP-pruned experts (192 of 384). Pruning should *raise* overlap, since routing mass concentrates, but this is unmeasured.
- No batch-1 spec-decode MoE paper on unified-memory edge hardware with a fully resident model.
- Quality cost of a verifier expert budget on long reasoning or agentic chains: AcceptMoE's 4 benchmarks are short-form.

---

## 4. Tree vs chain at batch 1, dynamic draft length, and lossy acceptance

### Takeaway
On a bandwidth-bound MoE, prefer a short chain (or a very narrow tree) with dynamic, confidence-gated length. Trees raise τ only slightly at batch 1 while inflating the expert union. Use the DSpark confidence head or SpecDec++-style stopping, and Cascade-style "spec decode off" when utility falls below 1. Lossy typical acceptance is a cheap +11% (measured internally on DFlash).

### Cited Findings
- **Tree vs chain** (arXiv 2601.11580, SGLang, server GPUs). Trees give a slightly higher speedup than chains at batch 1, and the edge vanishes with batch. On Qwen3-8B GSM8K, accepted length goes from 2.25 (chain) to 2.51 (k=6 tree) and 2.92 (k=21 tree), while acceptance rate drops from 0.415 to 0.300 and 0.095. — [Speculative Decoding: Performance or Illusion?](https://arxiv.org/pdf/2601.11580)
- **EVICT** shows that MoE trees verify far more nodes than get accepted: verifying 74.7% fewer nodes cost only about 15% of τ. — [EVICT](https://arxiv.org/html/2605.00342v1)
- **SpecDec++** stops drafting when a trained acceptance-prediction head estimates P(any rejection) above a threshold. Llama-2-chat 7B/70B: 2.04x vs 1.90x (Alpaca), 2.26x (GSM8K), 2.23x (HumanEval). — [SpecDec++](https://arxiv.org/html/2405.19715v3)
- **DSpark** has a per-position confidence head and (per secondary press) a load-aware scheduler that verifies more when the GPU is idle. — [lilting.ch](https://lilting.ch/en/articles/deepseek-v4-pro-dspark-speculative-decoding); [gmagazine](https://gmagazine.it.com/?p=454)
- **Cascade**: utility-gated K, with spec decode disabled when utility < 1, limits worst-case slowdown to 5%. — [arXiv 2506.20675](https://arxiv.org/abs/2506.20675)
- **Consumer hardware**:
  - Best case 1.61x at K=6, with acceptance falling from 69.7% at K=1 to 37.8% at the optimum.
  - 3 of 5 configs slowed down when "parallel" verify ran serially in a quantized backend.
  - — [Lossless but Not Free 2607.17283](https://www.alphaxiv.org/abs/2607.17283v1)
- **Internal**: Medusa-style typical acceptance on linear DFlash gave +11% tok/s at T=0.3, and DDTree did not beat linear (depth-dominated). — [/home/patrickd/.claude/projects/-home-patrickd/memory/dflash-typical-acceptance-win.md]

### Inferences
- The DSpark confidence head can serve as the SpecDec++ stopping rule for free. Combine it with a per-token expected-new-experts estimate (DraftExpert's confidence–expansion rule) to get a cost-aware stop. **[EXTRAPOLATION]**
- Verify-width sanity check: our K-quant expert GEMVs at n=2–4 must actually reuse x across rows (see the activation-traffic finding) so that verify(n) does not cost n·gemv. The "Lossless but Not Free" serial-verify failure mode is the same trap.

### Gaps
- No published quality measurements for typical acceptance on reasoning-heavy MoE outputs. Our +11% figure has no matched-quality eval attached.

---

## 5. Lookahead, n-gram, suffix and prompt-lookup drafting, and hybrids

### Takeaway
For agentic and code workloads, SuffixDecoding is the strongest model-free drafter: up to 3.9x on SWE-Bench / Text-to-SQL, about 2.2x over EAGLE-2/3 (NeurIPS 2025). It is cheap to bolt on and dynamically switches with a model drafter. Its accepted runs are long, but verify width is still union-priced.

### Cited Findings
- **SuffixDecoding** (Oliaro et al., NeurIPS 2025):
  - Up to 3.9x on agentic benchmarks, about 2.2x faster than EAGLE-2/3, and 1.6–1.9x faster than Token Recycling.
  - Hybrid variants switch between suffix and EAGLE-3 drafting based on match confidence.
  - Up to 4.5x end-to-end speculative speedup on SWE-Bench task completion.
  - In ArcticInference / vLLM.
  - — [NeurIPS poster](https://neurips.cc/virtual/2025/poster/115453); [arXiv v3](https://arxiv.org/pdf/2411.04975v3); [vLLM docs](https://docs.vllm.ai/en/v0.16.0/features/spec_decode/)
- A third-party summary's 5.3x / 2.8x figures do not match the paper and should be ignored. — [takara tldr](https://tldr.takara.ai/p/2411.04975)

### Inferences
- Suffix drafts are free to produce (CPU tree lookup), so the T_draft term vanishes. Gate them by match length: in our cost model only drafts with expected accepted length above about 2 pay at m≈2. **[EXTRAPOLATION]**
- Agentic tool loops (tool JSON, file echoes) are where this matters for the leafllm and agent use cases. Chat will not benefit.

### Gaps
- No hybrid-vs-pure-suffix numbers retrieved. No MoE-target SuffixDecoding numbers. Lookahead/Jacobi 2025–26 results were not covered this session.

---

## 6. Self-speculation (layer skip, early exit, quantized self-draft)

### Takeaway
These are mostly dense-model results at 1.3–2.2x. For us, a low-bit self-draft is weak: the experts are already 2–3 bit, so a lower-bit copy saves little bandwidth. An MoE-native self-draft (shared + top-1 + a trained draft expert, DraftExpert) is the relevant variant. It also cuts the drafter's expert reads.

### Cited Findings
- **LayerSkip**: up to 2.16x on CNN/DM, 1.82x on code, 2.0x on TOPv2. Needs training with layer dropout and early-exit loss. — [LayerSkip](https://arxiv.org/abs/2404.16710)
- **SWIFT**: training-free on-the-fly layer skipping, 1.3–1.6x (1.56x on LLaMA-2-13B). — [SWIFT](https://arxiv.org/pdf/2410.06916)
- **Quantized self-speculation**:
  - QSpec drafts with low-precision activations over shared 4-bit weights.
  - A hierarchical framework reaches 2.78x on W4 Llama-3-70B (A100) with an intermediate drafter, because tree verify erodes the 4-bit gains.
  - Quasar quantizes the *verifier*: 1.28x throughput, up to 1.6x on GSM8K.
  - — [arXiv 2505.22179](https://arxiv.org/html/2505.22179v1); [Quasar 2603.01399](https://arxiv.org/html/2603.01399v1)
- **DraftExpert**: 1.45x, 84–87% acceptance, MoE self-drafter. — [arXiv 2607.24434](https://arxiv.org/pdf/2607.24434)

### Inferences
- A "draft with shared expert + top-1 expert only" self-draft (no extra weights) would read about 1/6 of the routed bytes per draft token. It needs a trained correction (a draft expert) to reach usable acceptance. Its router-agreement output doubles as the verifier expert-budget predictor. **[EXTRAPOLATION]**

### Gaps
- No MoE-at-2-bit self-speculation result exists in the sources found.

---

## 7. Black-swan results (>3x lossless at batch 1 on MoE)

### Takeaway
None found that transfer. The only MoE results above 3x at concurrency 1 are DFlash on Qwen3-Coder-30B-A3B (code, B200) and SuffixDecoding on agentic traces. Both are workload- and hardware-specific. Every bandwidth-bound MoE batch-1 study lands at 1.1–1.4x over EAGLE-3 or MTP, or up to about 2.35x over AR.

### Cited Findings
- DFlash, Qwen3-Coder-30B-A3B on a B200 at c=1: 3.5x (HumanEval). — [DFlash](https://arxiv.org/html/2602.06036)
- SuffixDecoding: up to 3.9x on agentic traces. — [NeurIPS 2025](https://neurips.cc/virtual/2025/poster/115453)
- EVICT: up to 2.35x over AR on a Qwen3 MoE (A100, batch 1). — [EVICT](https://arxiv.org/html/2605.00342v1)
- DeepSeek-V3.1 native MTP: 1.10x; with EcoSpec 1.15x (H200, batch 1). — [EcoSpec](https://arxiv.org/html/2607.12696)

### Inferences
- The realistic stack for a step-change is roughly 1.6–2.2x on mixed chat **[EXTRAPOLATION]**:
  1. A REAP-quant-fine-tuned DSpark head with router-agreement distillation.
  2. Confidence-gated short chains (k≈1–3).
  3. A commitment-weighted verifier expert budget (lossy, about −0.3 pp).
  4. Suffix drafting for agentic and code turns.
  5. Typical acceptance.
- Without a verifier expert budget, the cost model caps us near 1.3x on chat whatever the drafter.

### Gaps
- Nobody has published spec decode for DSA sparse attention plus hyper-connection (mHC) targets at batch 1. Verify-time behaviour of the DSA indexer (top-512) and of compressed KV for k+1 tokens is unmeasured in the literature.
