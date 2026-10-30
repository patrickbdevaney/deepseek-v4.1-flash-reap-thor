# Efficient vision-language serving in a single-stream CUDA server (DeepSeek-V4.1-Flash-1017 on Thor)

Scope: what to build for the ViT + aligner path in the pure-CUDA server, which token-reduction and caching techniques produce step changes in image-request latency, how MoE expert pruning interacts with vision, and how to measure image quality cheaply. Date of research: 2026-10-30. Search budget was ~17 tool calls, so several per-method tables were read from abstracts only; this is flagged inline.

Model facts used below (verified from local header/config notes, not the model card): vision_config is 32 layers, hidden 1024, 16 heads, intermediate 2816, patch 14, `downsample_ratio 3`, `max_image_tokens 1024`, `min_pixels 295936`; aligner `w1` [5120, 9216] and `w2` [5120, 5120]; vision + aligner + image_* is 0.90 GiB BF16; image tokens select experts with `bias_vl` and get no Engram contribution; KV is 890 B/token — [local architecture note](/home/patrickd/deepseek-v4.1-flash-reap-thor/pipeline-plan/notes/v41_1017_architecture_runtime.md), [config.json](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/resolve/main/config.json), [local REPORT.md](/home/patrickd/deepseek-v4.1-flash-reap-thor/REPORT.md).

## Q1. Visual token reduction for LLM prefill: what is removed and what it costs (esp. OCR)

### Takeaway
For general VQA, 50-75% of visual tokens can go at a 2-5% accuracy cost, and simple spatially uniform methods (random, pooling) match or beat attention-score methods such as FastV. OCR/document tasks are the exception: the useful-information horizon sits deeper, and methods designed for natural-image VQA do worse than random on an OCR model. V4.1's aligner already applies a 9x (3x3) merge, so further pruning should default to off for document/chart prompts and be a conservative opt-in elsewhere.

### Cited Findings
- **Simple baselines win.** On LLaVA-1.5-7B (vanilla TextVQA 58.2, MMBench 64.7), at 75% pruning TextVQA scores were Random 51.7, Pooling 53.4, Vanilla FastV 53.6 and SparseVLM 54.9; MMBench scores were Random 62.2, Pooling 62.5, FastV 59.3 and SparseVLM 59.5. Average retained performance at 75% was 95.0% (Random), 96.4% (Pooling), 93.5% (SparseVLM) and 89.8% (FastV). At 88.9% (64 tokens kept) it was 89.1%, 87.6%, 87.3% and 78.2%. Random and Pooling beat the carefully designed methods on nearly 2/3 of benchmarks — [Wen et al., ACL Findings 2025, arXiv 2502.11501](https://arxiv.org/html/2502.11501v2)
- **Positional bias.** FastV-style attention selection keeps mostly tokens at the end of the visual sequence. Enforcing spatial uniformity ("Window FastV") cuts the average drop by 3.4 points at 75% and 9 points at 88.9% — [arXiv 2502.11501](https://arxiv.org/html/2502.11501v2)
- **FLOPs are not latency.** On LLaVA-NeXT-7B, SparseVLM had only 2.8% more FLOPs than FastV (15.6% vs 12.8%) but 26.8% higher latency, because it is FlashAttention-incompatible and pays per-layer pruning overhead. The authors recommend pruning early with cheap ops and measuring wall-clock time — [arXiv 2502.11501](https://arxiv.org/html/2502.11501v2)
- **Deep-layer pruning is random-equivalent.** Beyond about layer 20, existing training-free methods do no better than random ("vanishing token information"). The information horizon is deeper for OCR than for VQA, and deeper for stronger models (Qwen2.5-VL vs LLaVA-1.5). DivPrune plus random deep-layer pruning keeps 96.9% of Qwen2.5-VL-7B's performance at 50% pruned — [arXiv 2512.07580](https://arxiv.org/abs/2512.07580) (abstract only)
- **OCR models break general-purpose pruners.** On DeepSeek-OCR, pruning methods built for conventional VLMs do worse than random, because OCR needs exhaustive reconstruction of all text and layout rather than sparse question-relevant regions. RTPrune keeps 99.47% of OmniDocBench accuracy at 84.25% token retention (about 16% pruned) for a 1.23x prefill speedup — [RTPrune, arXiv 2605.00392](https://arxiv.org/abs/2605.00392) (abstract only)
- **Text-rich pruning needs adaptive budgets.** ET-Prune (Qwen3-VL-8B, InternVL3.5-8B) protects text-like regions and sets a per-sample token floor. At about 54% average retention it leads the strongest pruned baselines on OCRBench-v2 by 1.80 and 0.68 points, and on MMBench v1.1 scores 0.8467 vs 0.8437 for the unpruned model (single runs) — [ET-Prune, arXiv 2608.01979](https://arxiv.org/abs/2608.01979) (abstract only)
- **Accuracy hides failures.** On Qwen3-VL-8B at 30% retention, a targeted selector matched full accuracy (0.786 vs 0.783, CI [-0.014, +0.020]), yet selectors of equal accuracy differed sharply in whether the kept tokens covered the supporting OCR region (0.620 vs 0.270 for random). Up to 4.32x batch-prefill speedup was reported — [arXiv 2608.00077](https://arxiv.org/abs/2608.00077) (abstract only)
- **What the tokens are matters.** On Qwen3-VL-30B-A3B ChartQA, pruning the "key" vision tokens (top 20% by text-to-vision attention) first drops accuracy from 85.20 to about 66 at 20% pruning, while pruning redundant tokens first holds 84-85% at 50% pruning — [MODE, arXiv 2606.17118](https://arxiv.org/html/2606.17118v1)
- **VScan** (encoder-side global/local scan with merging, plus mid-LLM pruning): 2.91x prefill speedup, 10x fewer FLOPs, 95.4% performance retained on LLaVA-NeXT-7B across 16 benchmarks — [arXiv 2505.22654](https://arxiv.org/abs/2505.22654) (abstract only)
- **VisionZip** (pre-LLM selection of dominant plus contextual tokens): 8x prefill-time improvement, at least 5% better than prior SOTA in nearly all settings — [arXiv 2412.04467](https://arxiv.org/abs/2412.04467) (abstract only)
- **HiRED** (budgeted, pre-LLM, attention-guided drop): at a 20% token budget on LLaVA-NeXT-7B it gives 4.7x generation throughput, 78% lower response latency and 14% less memory (P40 GPU) — [arXiv 2408.10945](https://arxiv.org/abs/2408.10945) (abstract only)

### Inferences
- **Do not prune for documents.** V4.1 already compresses 9 patches into 1 token (3x3 unfold). The baseline studies above prune tokens from uncompressed CLIP grids (576 tokens for LLaVA-1.5), so their redundancy rates almost certainly overstate what is left in V4.1's tokens (extrapolation). The OCR evidence (RTPrune, the deeper OCR horizon) points to at most about 15% pruning on document/chart images. Default to off for OCR-like prompts.
- **Prefer pre-LLM pruning.** Pruning before layer 0 (VisionZip/HiRED/DivPrune-style, or pooling) is the only kind that is cheap to build in our server. V4.1 attention is a 128-token sliding window plus a top-512 compressed sparse KV shared across layer groups, so mid-stack FastV/PyramidDrop-style pruning would have to drop rows consistently from the mHC residual streams and the shared compressed-KV state. That is a large build for an effect the literature says is random-equivalent past the mid-stack (inference).
- **Spatial uniformity is the safe default.** If pruning is added for natural-photo prompts, use 2x2 pooling or random/grid selection that preserves the `[NEWLINE]` row structure, rather than attention-score top-k. Re-emit `[NEWLINE]` per surviving row so the 2D layout the model was trained on stays intact (inference).
- **Cost model for Thor (derived).** At the minimum resolution, 295,936 px = 544², so about 38x38 = 1,444 patches become about 13x13 = 169 tokens plus newlines. That matches the ~200-token prompt in today's llama.cpp test. At 739 tok/s text prefill in our server (if image rows cost about the same as text rows), 200 tokens take about 0.27 s and 1,024 tokens about 1.4 s. Halving the tokens saves about 0.14 s on a small image and 0.7 s on a max-size image. The decode KV saving is negligible at 890 B/token. Token reduction therefore matters only for large images.

### Gaps
- No paper found that benchmarks token pruning on a model with a built-in 3x3 (or 2x2) pixel-shuffle aligner, such as DeepSeek-VL2 or InternVL, on DocVQA/ChartQA specifically. The full per-benchmark tables for VisionZip, PyramidDrop, DivPrune, TopV and FasterVLM on DocVQA/ChartQA were not retrieved in this session (abstracts only).
- ToMe-in-ViT numbers for 2D-RoPE ViTs were not retrieved. ToMe changes the patch grid, which interacts with 2D RoPE and the 3x3 unfold, so it is likely incompatible without retraining (inference).

## Q2. ViT encoder efficiency: precision, attention, caching, resolution

### Takeaway
At V4.1's typical sizes the ViT is a small fraction of image-request latency (LLM prefill dominates), except at the 1,024-token maximum, where global attention over about 9,216 patches dominates the ViT itself. Build a BF16 ViT with fused FlashAttention-style tiling first, cache encoder outputs by image hash, and treat FP8 ViT linears as a later, separately gated lever (W8A8 on the ViT has been reported near-lossless).

### Cited Findings
- **W8A8 ViT can be near-lossless.** NVILA quantizes its vision tower to W8A8 because the tower was over 90% of its prefill latency, and reports this as nearly lossless — [NVILA, arXiv 2412.04468](https://arxiv.org/pdf/2412.04468) (via search summary)
- **FP8 end to end.** In LiteVLM's pipeline, FP8 raised the end-to-end speedup from 2.4x (FP16) to 3.2x with slight accuracy loss; the per-stage ViT gain was about 1.5x — [LiteVLM, arXiv 2506.07416](https://arxiv.org/pdf/2506.07416) (via search summary)
- **Small ViT gain from W4A8.** MBQ applies weight-activation quantization to the ViT and weight-only to the decoder, and measures only about 1.15x ViT speedup from W4A8 — [MBQ, arXiv 2412.19509](https://arxiv.org/pdf/2412.19509) (via search summary)
- **Practitioners often keep the ViT high-precision.** Some released FP8 VLM checkpoints keep the vision encoder in fp16 for accuracy — [Qwen3.5-9B-FP8-Dynamic card](https://huggingface.co/ishaqinu/Qwen3.5-9B-FP8-Dynamic)
- **Lab precedent.** A 24-block ViT was previously implemented in CUDA here and gated at cos 1.0. That work hit a warp-local shfl bug, and a bf16 oracle turned out to be the wrong reference — [local memory: GLM-5.3 vision tower](/home/patrickd/.claude/projects/-home-patrickd/memory/glm53-vision-tower-cuda.md)

### Inferences
- **Size of the tower (derived from config).** Per layer the ViT has 4·1024² = 4.19M attention parameters plus a gated MLP of 3·1024·2816 = 8.65M, so about 12.8M per layer and about 411M over 32 layers. The aligner is 5120·9216 + 5120² = 73.4M. Together (about 484M x 2 B = 0.90 GiB) this matches the measured 0.90 GiB, which supports a gated (3-matrix) MLP. Verify against `inference/model.py`.
- **ViT FLOPs (derived).** About 2·411M = 0.82 GFLOP per patch for the linears, plus attention at 4·N·d per layer per patch.
  - Min resolution (N ≈ 1,444): about 1.2 TFLOP linear plus 0.27 TFLOP attention.
  - Max resolution (N ≈ 9,216 if global attention): about 7.6 TFLOP linear plus about 11 TFLOP attention, so attention dominates.
  - At an assumed 50-100 TFLOPS realized BF16 on Thor's 20 SMs (unmeasured), that is roughly 15-30 ms per min-size image and 0.2-0.4 s per max-size image. Both are below the LLM prefill cost of the resulting tokens (0.27 s and 1.4 s). Measure the realized TFLOPS first.
- **Build order.**
  1. BF16 ViT with a fused attention kernel (FA2-style online softmax, 64-dim heads, 2D RoPE applied in-kernel). Reuse the UMMA/tcgen05 path already proven on sm_110a. Observe the SW128 swizzle-mask traps from [local memory](/home/patrickd/.claude/projects/-home-patrickd/memory/umma-tcgen05-thor-traps.md).
  2. Fuse the 3x3 unfold into the aligner `w1` GEMM's A-operand gather. It is a strided gather, so no materialized copy is needed.
  3. Write aligner outputs straight into the LLM prefill embedding buffer at the `[IMAGE]` positions, with learned `image_start/end/newline` rows interleaved.
  4. FP8 E4M3 W8A8 for the ViT linears only as a separately gated arm (cos/KL against BF16 and DocVQA delta). The ViT is 0.76 GiB, so bandwidth is not the motivation; compute at max resolution is.
- **Windowed attention is not an option.** The released weights were trained with whatever attention the reference uses (32-layer global ViT per config). Do not introduce windowing; it changes the model.
- **Resolution policy is a pure serving knob within [min_pixels, max_image_tokens].** Tokens scale with area, so a "fast" policy that caps at, say, 384 tokens cuts both ViT attention (quadratic) and LLM prefill (linear). That matches the 384-tokens/image cap that third-party write-ups attribute to DeepSeek's hosted V4-Flash-Vision (see Q3 Gaps). Pick the cap per request class: documents get the full 1,024, photos a lower cap. This is likely a bigger and safer latency lever than in-LLM pruning (inference).
- **Tiles need no batching kernel.** In a single-stream server a single image is one variable-length sequence. Multiple images can share one varlen attention launch with cu_seqlens-style offsets, so separate tile batching is unnecessary (inference).

### Gaps
- No primary measurement of FP8 ViT accuracy on a DeepSeek ViT, and no FlashAttention-for-ViT benchmark on sm_110/Thor, was found. Thor's realized dense BF16/FP8 TFLOPS on 20 SMs was not sourced; it must be measured locally.
- No official DeepSeek V4 / V4.1 vision technical report was found; only third-party articles, which conflict with each other.

## Q3. Serving design in vLLM / SGLang / TRT-LLM: what a single-user server should copy

### Takeaway
Copy two things: (1) an encoder-output cache keyed by a content hash of the preprocessed image, and (2) prefix/KV caching whose block keys include that image hash, so a follow-up turn about the same image skips both the ViT and the image-span prefill. Disaggregated encode (EPD) is a multi-GPU throughput feature that can hurt latency for image-light single-user traffic. Skip it.

### Cited Findings
- **vLLM prefix-cache keys.** Each image becomes a run of identical placeholder tokens, so vLLM folds the frontend image processor's image hash into each block hash as an "extra hash", alongside LoRA IDs and the cache salt. Every block in the image span carries the same image hash — [vLLM prefix caching design](https://docs.vllm.ai/en/stable/design/prefix_caching.html)
- **vLLM disaggregated encoder.** The encoder runs in a separate instance, and an EC connector (`vllm/distributed/ec_transfer`) moves encoder-cache embeddings to the prefill/PD instance. The stated benefits are independent scaling, lower TTFT and cross-process reuse of encoder outputs. The scheduler checks whether an embedding already exists and schedules a load; the worker loads it — [vLLM disagg encoder docs](https://docs.vllm.ai/en/stable/features/disagg_encoder/)
- **Cache dependency.** Cross-process caching requires the multimodal processor cache to stay enabled (do not set `--mm-processor-cache-gb 0`) — [vLLM-Ascend EPD guide](https://docs.vllm.ai/projects/ascend/en/main/_sources/user_guide/feature_guide/epd_disaggregation.md)
- **NVIDIA Dynamo** adds a CPU-side LRU embedding cache in the prefill worker. A hit skips the encode worker entirely; aggregated vLLM uses the native `ec_both` connector (vLLM ≥ 0.17) — [Dynamo embedding cache](https://docs.dynamo.nvidia.com/dynamo/multimodal/embedding-cache)
- **SGLang EPD.** Encoder servers (`--encoder-only`) scale by data parallelism, since ViTs do not benefit from higher TP. `--enable-prefix-mm-cache` skips repeated ViT work (default 4 GB cache, `SGLANG_VLM_CACHE_SIZE_MB`).
  - Measured on 8x H20 with Qwen3-VL-235B-A22B-FP8 at 1080p, 1-8 images/request: TTFT about 6-8x lower and TPOT about 8-10x lower than colocation at 1 QPS, and about 2x throughput.
  - SGLang notes that image-light workloads can see higher TTFT, because embedding transfer can cost more than the encode it saves — [LMSYS EPD blog, 2026-01-12](https://www.lmsys.org/blog/2026-01-12-epd)
- **LMCache** hashes image-side tokens (`mm_hashes`) and caches their KV for reuse across requests — [LMCache blog](https://blog.lmcache.ai/en/tag/kv-cache-en/) (third-party integration)

### Inferences
- **Single-user design to build.**
  - (a) Hash the decoded pixel tensor after resize/normalize (e.g. xxh3-128 over bytes plus target H×W). Hashing after resize makes the key resolution-policy aware.
  - (b) Keep an LRU encoder cache of aligner outputs. Each entry is ≤ 1,024 x 5120 x 2 B = 10 MiB BF16, so even 50 images is about 0.5 GiB. That fits the 12 GiB runtime reserve, but count it against the Thor unified-memory budget.
  - (c) Extend the existing prefix/KV reuse with the image hash as an extra key on image-span positions.
- **Why the KV cache is the bigger win here (derived).** V4.1 KV is 890 B/token, so a full 1,024-token image span is about 0.9 MB of KV. Unlike vLLM-scale models, the image's prefill state is far cheaper to keep than to recompute. Multi-turn chat about the same image should never re-prefill it. For our model the state also includes the compressed-KV/index state of layers 2/8/14/20 and the sliding-window rows, so the snapshot must capture all of them; reuse the recurrent-state checkpoint idea from earlier servers.
- **Exactness caveat.** Image-span KV is reusable only when the preceding text is identical, because of causal attention. Cache the snapshot keyed on (prefix-token hash, image hash). In practice the system prompt plus first user message is stable across turns, so this hits on every follow-up turn (inference).
- **Do not build EPD.** On one Thor the ViT and LLM share the same SMs and memory. The only useful concurrency is overlapping the ViT of image k+1 with LLM prefill of the text before it, which helps only multi-image prompts.

### Gaps
- vLLM's exact image hash function (blake3/sha over pixel bytes plus processor kwargs) and its encoder-cache eviction policy were not confirmed from primary docs this session. TensorRT-LLM multimodal encoder caching was not covered.

## Q4. MoE-specific multimodal routing and how expert pruning or quantization affects vision

### Takeaway
Vision and text tokens use measurably different experts. Because vision tokens dominate token counts, globally ranked expert statistics hide text-critical and "key-vision" experts. V4.1 makes this explicit with a separate `bias_vl`. Prune and quantize with per-modality statistics. A flip side reported for other MoE VLMs is that vision tokens are more tolerant of reduced expert compute than text tokens.

### Cited Findings
- **MODE** (Qwen3-VL-30B-A3B, Kimi-VL-A3B, InternVL3.5-30B-A3B). Global routing frequency is dominated by vision tokens, so text-critical experts (e.g. L12 E56/E85) rank below average. Key and redundant vision tokens activate different experts, and key-token expert profiles are stable across calibration sets (pairwise cosine above 0.74). Ranking by text plus key-vision frequency protected ChartQA/MMBench better than global ranking.
  - 10-benchmark average: BF16 72.65, MODE W3 69.81, MODE W2 63.29 (Qwen3-VL), vs plain GPTQ W2 32.79 and prior best VEQ-ME W2 59.29 — [MODE, arXiv 2606.17118](https://arxiv.org/html/2606.17118v1)
- **MoDES** (Kimi-VL, Qwen3-VL-MoE-30B-A3B, InternVL-3.5). The FFN changes text tokens much more than vision tokens, and experts are more redundant for vision. Reducing top-k hurts text-driven performance faster, and shallow-layer experts matter more.
  - Retention with modality-specific thresholds: 97.33% at 88% experts skipped (Qwen3-VL-MoE) vs 86.66% for MC-MoE. On Kimi-VL, 99.91% at 50% skipped.
  - Speedups on H200: about 2.16x prefill and 1.26x decode — [MoDES, arXiv 2511.15690](https://arxiv.org/html/2511.15690v1)
- **MC#** combines pre-loading mixed-precision expert quantization with online top-any pruning for multimodal MoE — [MC#, arXiv 2510.10962](https://arxiv.org/pdf/2510.10962) (abstract only)
- **V4.1 specifics (local, measured by others).**
  - `gate.bias_vl` correlates -0.05 with the text bias.
  - Text-only ranking at 272 experts was worse than random on vision (+15.9% vs +7.2% caption perplexity).
  - Plain mean saliency at 256 experts gave +27.4% caption perplexity vs +5.5% text.
  - Per-modality normalization recovered caption to +8.9%, at the cost of text +14.1% (LibertAI 50/50 variant) — [local REPORT.md](/home/patrickd/deepseek-v4.1-flash-reap-thor/REPORT.md), [local vision note](/home/patrickd/glm-5.3-reap/research_notes/Compressing%201T%20MoE%20for%20Thor/vision_retention_pruning.md)

### Inferences
- **Serving consequence.** The CUDA router must select image-span rows with `bias_vl` (including `[START]/[NEWLINE]/[END]`? Check `image_mask` in `inference/model.py` to see exactly which rows it covers). Weights must stay the unbiased scores. Getting this wrong silently changes vision quality, not text, so the golden test must include an image prompt.
- **A cheap speed lever not yet in the plan.** MoDES-style modality-specific top-k reduction on image rows during prefill (e.g. top-6 → top-4 on deeper layers for image tokens only) could cut MoE prefill work on image spans. The evidence comes from A3B-class models, not V4.1, and it needs its own vision gate (extrapolation).
- **Pruned-model precondition.** After the REAP-192 keep-set, image prompts should be checked first, before any token pruning is layered on, because the two degradations compound on the same tokens (inference).

### Gaps
- No paper evaluates expert pruning (as opposed to quantization or skipping) on a 384-expert VLM with a dedicated vision routing bias. The V4.1 evidence is caption perplexity only, not task accuracy.

## Q5. Cheap single-device eval protocol for image-domain quality retention

### Takeaway
Use paired, fixed-subset evaluation against the unpruned or bf16 reference, built from three layers: teacher-forced caption/answer perplexity (cheapest), per-token KL over image-conditioned answers, and small fixed task subsets of DocVQA/ChartQA/TextVQA/OCRBench/MMBench. For OCR also report spatial provenance or ANLS, not just accuracy.

### Cited Findings
- **The literature's standard suite.** ChartQA, InfoVQA, TextVQA (OCR/doc); VizWiz, MME-RealWorld, GQA (real-world); MMMU, MMStar, MMBench (reasoning); POPE (hallucination) — [MODE](https://arxiv.org/html/2606.17118v1)
- **Report more than accuracy for OCR.** Equal-accuracy selectors can differ 2.3x in covering the supporting OCR region, so pruning work should report surviving spatial provenance and realized cost alongside quality — [arXiv 2608.00077](https://arxiv.org/abs/2608.00077)
- **Measure wall-clock time, not FLOPs** — [arXiv 2502.11501](https://arxiv.org/html/2502.11501v2)
- **Plan already in place.** The local plan's Tier 2 vision gate is 2,048-token caption perplexity plus paired ChartQA/DocVQA/TextVQA. ChartQA is GPL, so it is eval-only, never calibration — [local REPORT.md](/home/patrickd/deepseek-v4.1-flash-reap-thor/REPORT.md)

### Inferences
- **Recommended gate for the CUDA vision path**, in order:
  1. Bit-level parity vs llama.cpp mmproj and vs `inference/model.py`, checked per ViT block and on aligner outputs (cos ≥ 0.9999 BF16). Lab precedent is cos 1.0 on a previous ViT.
  2. Teacher-forced NLL of reference answers on about 300 image QA items (minutes).
  3. Paired subsets: 500 DocVQA (ANLS), 500 ChartQA (relaxed accuracy), 500 TextVQA, the 1,000-item OCRBench, and 500 MMBench-dev, with greedy decoding and a pinned generation config, scored as paired differences with bootstrap CIs.
  4. Latency: time-to-first-token for 200- and 1,024-token images, measured cold vs encoder-cache hit vs KV-snapshot hit.
- **Statistics.** With n = 500 paired items, a 2-point drop is roughly at the edge of detectability at 95% (inference based on binomial variance; pairing helps).
- **Reasoned prompts for multiple choice.** Evaluate MMBench with reasoning prompts, not letter-only. A previous lab finding showed letter-only MC heavily understates capability — [local memory](/home/patrickd/.claude/projects/-home-patrickd/memory/letter-only-mc-format-artifact.md)

### Gaps
- No single published "cheap VLM retention" protocol was found. The above is assembled from method papers plus local practice. DocVQA/InfoVQA licences for local use were not checked in this session.
- **Conflicting third-party claims about DeepSeek's hosted vision models:** a 384-token per-image cap, about 1344x1344 max resolution, and "32-layer encoder" for V4-Flash-Vision-Exp / V4.1 Flash — [intuitionlabs](https://intuitionlabs.ai/articles/deepseek-v4-flash-vision-document-understanding), [eesel](https://www.eesel.ai/blog/deepseek-v4-flash-vision-exp), [propakistani](https://propakistani.pk/2026/09/11/deepseek-v4-1-flash-launches-with-lower-prices-and-native-vision/). These conflict with `max_image_tokens 1024` in the 1017 config.json. Trust the config. A 384 cap may be an API-side resolution policy, which would be the hosted precedent for the "lower cap for photos" knob in Q2 (unverified).
