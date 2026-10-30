# Single-stream MoE decode efficiency on bandwidth-bound GPUs (2025-2026): weight formats, kernels, bytes/token

Research date 2026-10-30. Target context: DSV4.1-Flash REAP-192, Jetson AGX Thor (sm_110a, 20 SMs). Engine: 25.0 tok/s, 48.2 ms/token against a 37.6 ms roofline floor, so it is already at about 78% of the floor. Per-token budget: NVFP4 dense 21.4 ms (3.67 GB), K-quant experts 16.5 ms (3.0 GB), glue 6.2 ms at 32 GB/s, bf16 LM head 5.7 ms (1.32 GB), attention 2.2 ms, fp8 dense 1.4 ms.

Method note: about 18 tool calls. Rows marked (fetched) were read from the primary page. Rows marked (search summary) come from search-result summaries and are less reliable. Anything marked EXTRAPOLATION is my own arithmetic and is not a published measurement.

## 1. Low-bit weight formats with fast GEMV at 2-3 bpw (QTIP, AQLM, QuIP#, HIGGS, FLUTE, any4, IQ/KT, NVFP4)

### Takeaway
Trellis codes (QTIP, which also underlies EXL3 and ik_llama's IQ*_KT) are the best published quality per bit at 2-3 bpw. At batch 1 they decode as fast as QuIP#, and both are more than 2x faster than AQLM. QTIP 3-bit beats AQLM and QuIP# 3-bit, and QTIP 2-bit gets close to their 3-bit quality. Every speed number found is from high-bandwidth discrete GPUs. None is from a 20-SM, about 228 GB/s part, where the ALU cost of decoding (2-4 instructions per weight) may no longer be hidden behind memory. On low-compute GPUs, community data shows llama.cpp K/IQ quants decoding faster than EXL3 at matched size.

### Cited Findings
- QTIP, batch 1, RTX 6000 Ada (960 GB/s), Llama-2 (fetched):
  - 2-7B: QTIP-2bit 188 tok/s, QuIP#-2bit 186, AQLM-2bit 81.5, FP16 55.9.
  - 2-70B: QTIP-2bit 23.5, QuIP# 22.2, AQLM 8.78.
  - QTIP at 3 bits and 4 bits: 161 / 140 tok/s (7B) and 19.1 / 16.3 tok/s (70B).
  - Source: [QTIP arXiv 2406.11235v4, Table 4](https://arxiv.org/html/2406.11235v4)
- QTIP on lower-bandwidth GPUs (fetched):
  - RTX 3090, 2-7B: 127 / 119 / 109 tok/s at 2 / 3 / 4 bits (FP16 52.5).
  - RTX A6000: 116 / 106 / 95 tok/s (7B), and 70B at 15.0 / 13.1 / 11.7.
  - Each extra bit costs only about 7-9% here, so decode is not purely proportional to bytes. Some ALU or latency cost is visible.
  - Source: [QTIP Table 17](https://arxiv.org/html/2406.11235v4)
- QTIP quality, Wikitext2 ppl, Llama-2 7B / 13B / 70B (fetched):

  | Method | 7B | 13B | 70B |
  |---|---|---|---|
  | FP16 | 5.12 | 4.57 | 3.12 |
  | QTIP-3 | 5.28 | 4.69 | 3.26 |
  | AQLM-3 | 5.38 | 4.78 | 3.36 |
  | QuIP#-3 | 5.41 | 4.78 | 3.35 |
  | QTIP-2 | 5.86 | 5.11 | 3.70 |
  | AQLM-2 | 6.14 | 5.33 | 3.83 |
  | QuIP#-2 | 6.19 | 5.35 | 3.91 |

  Source: [QTIP Table 5](https://arxiv.org/html/2406.11235v4)
- QTIP decode cost per weight (fetched):
  - 1MAD is about 4 instructions.
  - 3INST is 3 ALU instructions plus one data-movement instruction.
  - HYB (hash into a small LUT) is about 2 instructions amortized.
  - The paper claims its matvec runs at more than 80% of peak memory bandwidth "on modern GPUs". That is a kernel-level figure, not end-to-end.
  - Source: [QTIP](https://arxiv.org/html/2406.11235v4)
- EXL3 (the QTIP-derived format in exllamav3) vs GGUF, from blog and community data (search summary):
  - Quality: EXL3 4.0 bpw has KLD 0.0248 at 5.79 GiB, against Q4_K_M at 0.0398 and 6.62 GiB. EXL3 5.0 bpw roughly ties Q6_K.
  - Decode speed: llama.cpp was faster in both size-matched pairs on an RTX 5090. On an RTX 3060, llama.cpp ran 39 tok/s against 30 for EXL3, and 36 against 23 under a clock limit.
  - So the trellis decode cost shows up when compute is scarce.
  - Sources: [atomic.chat EXL3 vs GGUF](https://atomic.chat/blog/guides/exl3-vs-gguf); [Kaitchup](https://kaitchup.substack.com/p/serving-exllamav3-with-tabbyapi-accuracy)
- ik_llama.cpp trellis quants (IQ1_KT to IQ4_KT) (search summary):
  - They use an "integer-base trellis" meant to be cheap on CPU.
  - They have CUDA kernels (PR 113; "Faster IQ3_KT and IQ4_KT" PR 453; MMQ kernels for all types in PR 557). No tok/s figures were found.
  - Community MoE quants use IQ2_KT for routed experts and IQ3_KT for down_proj.
  - Sources: [ik_llama.cpp fork README](https://github.com/RodriMora/ik_llama.cpp); [ubergarm GigaChat3 card](https://huggingface.co/ubergarm/GigaChat3-10B-A1.8B-GGUF)
- FLUTE (LUT GEMM, EMNLP 2024) (search summary):
  - Kernel speed: 2-4x over prior LUT kernels at batch below 32 with group size 128.
  - End-to-end: 1.5-2x.
  - LLaMA-3-70B on 2xA100: FP16 19.9 tok/s, 4-bit g32 28.3 tok/s, 3-bit g256 (3.06 b/p) 34.8 tok/s. The 3-bit result's Wikitext2 ppl was 15.4 against 2.9 for FP16, because the NF grid is poor at 3 bits on Llama-3.
  - Source: [FLUTE arXiv 2407.10960](https://arxiv.org/pdf/2407.10960)
- HIGGS (Hadamard plus MSE-optimal grid, data-free, runs on FLUTE) (search summary):
  - The only speed figure found is 4-bit on an RTX 4090 at batch 1, Llama-3.1-8B: 124 forward passes/s compiled, against 59 for bf16.
  - No 2-3 bit speeds were found.
  - Source: [Transformers HIGGS docs](https://huggingface.co/docs/transformers/v5.13.1/quantization/higgs.md)
- any4 / tinygemm (search summary):
  - tinygemm targets M=1-16.
  - The authors say that tensor cores "still outperform" scalar matmul even at M=1.
  - Source: [any4 arXiv 2507.04610](https://arxiv.org/pdf/2507.04610)
- PyTorch GemLite (search summary):
  - At batch 1 a GEMV kernel beats GEMM.
  - int4 weight-only gave 1.95x over compiled bf16 on Llama-3.1-8B (131 to 255 tok/s).
  - Source: [PyTorch blog](https://pytorch.org/blog/accelerating-llm-inference/)
- NVFP4/MXFP4 at batch 1 on Blackwell: no primary measurement of GB/s efficiency was found (search summary).
  - Nearest data: an SM120 model card measured single-stream decode with NVFP4A16 (weight-only) faster than W4A4: 18.8 vs 14.5 tok/s eager, and 142.5 vs 119.2 tok/s with piecewise graphs.
  - Modular's Apple FP4 GEMV reaches about 88% of peak bandwidth on the bf16 path.
  - Sources: [friendli KAT-Coder card](https://friendli.ai/models/Ttimms/KAT-Coder-V2.5-Dev-REAP-50-NVFP4-W4A4); [Modular fp4_gemv](https://docs.modular.com/api/mojo/linalg/matmul/gpu/apple/fp4_gemv/)
- TileQ claims Mixtral-8x7B ppl 4.78 at 2 bits where GPTQ collapses. This is unverified beyond a summary (search summary). Source: [emergentmind 2605.09281](https://www.emergentmind.com/papers/2605.09281)

### Inferences
- **Dense NVFP4 path (3.67 GB, 21.4 ms) — EXTRAPOLATION.**
  - NVFP4 stores about 4.5 bpw including scales. Moving the bf16-origin dense weights to a 3-bit trellis (about 3.1 bpw with overhead) would cut about 31% of bytes. That is about 1.1 GB, or about 4.9 ms at 228 GB/s.
  - The QTIP Table 5 ordering suggests 3-bit trellis quality is close to or better than 4-bit uniform. However, the dense overlay was accuracy-tested only for NVFP4 (+1.56% ppl). A trellis variant needs its own accuracy gate.
  - ALU check: 3.67 GB at 0.5625 B/weight is about 6.5 G weights per token. At about 2 instructions per weight (HYB) that is about 13 G instructions per token. On 20 SMs x 128 lanes, assuming a clock of about 1.5 GHz (Thor's GPU clock not verified here), the ceiling is about 3.8 T simple ops/s. That gives about 3.4 ms of ALU per token, which is near the 16 ms memory time for the new byte count.
  - So it is plausible but not free. The RTX 3060 data (EXL3 losing 23-36% to llama.cpp under a clock limit) is a direct warning for a low-SM part.
- **Routed experts (Q2_K/Q3_K, 3.0 GB) — EXTRAPOLATION.**
  - The quality-per-bit literature consistently ranks trellis (QTIP, EXL3, IQ*_KT) above K-quants.
  - The expected lever is "same quality at about 0.5 bpw less" rather than "better quality at the same bytes". At about 2.85 bpw average, 0.5 bpw is about 17% of expert bytes, roughly 0.5 GB or 2.3 ms.
  - ik_llama IQ2_KT/IQ3_KT is the most directly portable form, since it is already integer-trellis, CPU/CUDA-friendly, and GGUF.
- AQLM (additive codebooks) is about 2.3x slower than QTIP/QuIP# at batch 1 even on a 960 GB/s GPU. Reject it for Thor.
- At 3 bits FLUTE/NF loses badly on Llama-3-class models (ppl 15.4). Prefer incoherence-processed formats (QTIP, HIGGS) over plain LUT-NF at 3 bits or below.

### Gaps
- No primary measurement of trellis or KT GEMV on Jetson or any SM count at or below 20, and no GB/s for NVFP4 GEMV at M=1 on Blackwell. These would need a micro-benchmark on Thor.
- No HIGGS speed at 2-3 bit. No VPTQ speed numbers retrieved; VPTQ was not searched within budget.
- No matched-bpw K-quant vs IQ*_KT perplexity table from a primary source was retrieved.

## 2. MoE-specific bytes/token reduction at batch 1 (dynamic top-k, skipping, merging)

### Takeaway
The strongest 2026 evidence (a NeurIPS 2026 controlled study) says dynamic expert skipping rarely beats simply lowering a static k. Calibrated "effective k" also does not transfer to test time. For a 6-of-192 model, the practical lever is a static, layer-wise top-k reduction that protects the shallow layers. Fine-grained MoEs including DeepSeek-V3 show at least 10% gains "without measurable loss".

### Cited Findings
- "Don't Skip the Baseline" (NeurIPS 2026, Gemma4-26B-A4B, Qwen3-30B-A3B, Qwen3-VL) (fetched abstract):
  - Four dynamic-skipping methods were compared at matched expert compute.
  - Static k reduction is "hard to beat consistently". The dynamic methods "often underperform" it, and their gains, when present, are minor.
  - Calibration-time effective k differs from test-time effective k.
  - Recommendation: calibration-free static k reduction, split by modality.
  - Source: [NeurIPS 2026 poster 160490](https://neurips.cc/virtual/2026/160490)
- ASET / "Faster MoE LLM Inference for Extremely Large Models" (Findings ACL 2026) (search summary):
  - Static layer-wise top-k allocation gives 10-78% throughput gains across fine-grained MoEs.
  - It gives at least 10% on DeepSeek-V3 "without measurable loss".
  - Source: [ACL Anthology 2026.findings-acl.2140](https://preview.aclanthology.org/ingest-acl/2026.findings-acl.2140/)
- MoDES (CVPR 2026) (search summary):
  - Qwen3-VL-30B retains 97.3% of the average score at 88% skipping. Decode is 1.26x faster and prefill 2.16x.
  - Shallow MoE layers are much more sensitive to a lower top-k, so the authors protect them and skip more in deeper layers.
  - Source: [CVPR 2026 slides](https://cvpr.thecvf.com/media/cvpr-2026/Slides/37515.pdf)
- Opportunistic Expert Activation (arXiv 2511.02237) (search summary):
  - On Qwen3-30B-A3B, MoE latency falls to 0.61x at the most aggressive setting.
  - At that setting AIME24 drops from 43.0 to 22.2. The aggressive regime is costly on reasoning.
  - Source: [arXiv 2511.02237](https://arxiv.org/pdf/2511.02237)
- ACE (arXiv 2609.05228) does calibration-free skipping that scores each expert slot by its estimated output contribution rather than the gate value. No numbers retrieved (search summary). Source: [arXiv 2609.05228](https://arxiv.org/pdf/2609.05228)
- SERE (ICLR 2026) re-routes to similar experts, reaching up to 2.0x. It is batch-oriented, so at batch 1 it reduces to skipping or merging (search summary). Source: [arXiv 2602.07616](https://arxiv.org/html/2602.07616v1)
- DualSparse-MoE gives a 1.41x MoE-module speedup at 0.5% average accuracy loss, training-free (search summary). Source: [arXiv 2508.18376](https://arxiv.org/html/2508.18376v1)
- The early "Not All Experts are Equal" study (Mixtral) combines pruning with dynamic skipping (search summary). Source: [arXiv 2402.14800](https://arxiv.org/html/2402.14800v2)

### Inferences
- **EXTRAPOLATION:** going from top-6 to top-5 in the deeper half of the 40 layers only removes about 1/12 of routed-expert bytes. That is about 0.25 GB, or about 1.1 ms (about 2.3% tok/s).
- Going to top-5 on all layers removes about 0.5 GB, or about 2.2 ms (about 5% tok/s), at an unknown quality cost.
- These are small levers. The routed experts are only 3.0 of about 11 GB/token, and REAP-192 has already concentrated the routing mass.
- Any test must be run on top of the REAP model with the pinned-generation-config eval protocol. Reasoning benchmarks (AIME-type) degrade first.
- The shared expert is always read. A "shared-only fast path" for high-confidence tokens is a form of dynamic skipping, so the NeurIPS 2026 result argues it will not beat static-k.

### Gaps
- No published dynamic or static top-k result on DeepSeek-V4-family or mHC models.
- No batch-1 tok/s measurements on edge hardware for any skipping method.
- Inference-time expert merging (as opposed to merge-then-prune at compression time) produced no primary results within budget.

## 3. LM head reduction (1.32 GB bf16, 5.7 ms)

### Takeaway
The published vocab-reduction work (FR-Spec, VocabTrim, SpecVocab) targets drafter heads. It cuts head compute by about 75% with exact final distributions only because the target re-verifies. A two-stage NVFP4-shortlist plus bf16-rescore scheme is the target-side analogue. If the bf16 rescore currently reads the full 1.32 GB, it is wasted bytes. Lossless bf16 entropy coding (DFloat11, ZipServ) cuts about 30% of a bf16 head with bit-identical output.

### Cited Findings
- FR-Spec (search summary):
  - It restricts the draft head to a frequency-ranked subset, cutting LM-head compute by 75%.
  - The output distribution stays equivalent thanks to target verification. Average speedup is 1.12x over EAGLE-2.
  - On LLaMA-3-8B the full-vocabulary head is 64% of draft FLOPs.
  - Source: [FR-Spec arXiv 2502.14856](https://huggingface.co/papers/2502.14856)
- SGLang notes that FR-Spec-style reduction is incompatible with EAGLE-3 (search summary). Source: [arXiv 2603.05210](https://arxiv.org/html/2603.05210)
- SpecVocab picks a vocab subset at each step. It gets higher acceptance than EAGLE-3, FR-Spec and VocabTrim, and up to 8.1% higher throughput than EAGLE-3 (search summary). Source: [arXiv 2602.13836](https://www.arxiv.org/pdf/2602.13836)
- DFloat11 (NeurIPS 2025) (search summary):
  - It Huffman-codes the bf16 exponent bits, giving about 30% smaller weights with bit-identical outputs.
  - The decode LUTs sit in SRAM with a two-phase kernel.
  - Source: [arXiv 2504.11651](https://arxiv.org/abs/2504.11651)
- ZipServ (arXiv 2603.17435, Mar 2026) (fetched):
  - It uses a fixed-length TCA-TBE format: 3-bit codes for the top-7 exponents, which cover over 95% of weights, plus an outlier buffer.
  - It is lossless bf16, decoded in registers straight into tensor cores (ZipGEMM).
  - Kernel speedup over cuBLAS on an RTX 4090 averages 1.31x, up to 1.71x per layer and up to 2.21x in the abstract.
  - End-to-end it is 1.22x over vLLM, and latency is 82% lower than with DFloat11.
  - Evaluated on Ada (4090/L40S), with RTX 5090 forward compatibility claimed.
  - Source: [ZipServ](https://www.alphaxiv.org/abs/2603.17435v1)
- ECF8 applies the same exponent-Huffman idea to FP8 weights, which would cover the fp8 dense path (search summary). Source: [arXiv 2510.02676](https://arxiv.org/pdf/2510.02676)

### Inferences
- If 5.7 ms equals a full bf16 head read, an exact two-stage head should read the NVFP4 head plus only the top-K bf16 rows:
  - The full bf16 head is about 1.32 GB, which implies about 0.66 G parameters.
  - At NVFP4 that is about 0.37 GB, or about 1.6 ms.
  - Rescoring K=64-256 rows of the bf16 head costs about 1-3 MB, which is negligible.
  - Net saving is about 4 ms per token, about +9% tok/s. EXTRAPOLATION; this assumes the reported 5.7 ms is not already the two-stage path.
- Exactness of argmax/top-p needs a margin check: fall back to a full rescore when the NVFP4 margin between the K-th and (K+1)-th logit is below the quantization error bound.
- If the bf16 read must stay, ZipServ-style fixed-length exponent coding saves about 0.4 GB, or about 1.7 ms, losslessly.
- ZipServ's design is the better fit for a GEMV than DFloat11's variable-length Huffman, which is slower than ZipServ per the ZipServ comparison.

### Gaps
- No published target-model shortlist-head scheme with a measured exactness rate was found.
- Low-rank or SVD heads were not retrieved within budget.
- ZipServ batch-1 GEMV numbers exist only as a chart.

## 4. Megakernels, persistent kernels, whole-model fusion vs CUDA graphs

### Takeaway
At batch 1 the best published persistent megakernels reach about 78-80% of memory bandwidth end to end, against 50-69% for CUDA-graph systems (vLLM/SGLang). The engine is already at about 78% of the roofline, so a megakernel is unlikely to move the total much. The exception is the 6.2 ms glue segment running at 32 GB/s, which is the one clearly latency-bound region. Fusing it (norms, mHC mixing, router) into the adjacent GEMV prologue and epilogue is the highest-value overhead target.

### Cited Findings
- Hazy Research "No Bubbles", Llama-3.2-1B bf16, batch 1 (fetched):
  - The megakernel uses 78% of H100 memory bandwidth, against at most 50% for vLLM/SGLang, which use CUDA graphs plus torch.compile.
  - It is about 2.5x faster than vLLM and more than 1.5x faster than SGLang on H100, and runs under 680 µs per pass on B200 (more than 3.5x vLLM).
  - Kernel launch costs about 2.1 µs on a stream and about 1.3 µs inside a CUDA graph.
  - Their model assumes about 5 µs of per-kernel stall (launch, tail, prologue latency), which caps throughput at about 770 passes/s against a memory limit of about 1350.
  - Remaining megakernel overhead on B200: about 250 µs from activation store/load dependency waits (about 500 ns per load or store), about 40 µs of sync (about 60 ns per async barrier), and about 80 µs of setup.
  - No direct megakernel-vs-bare-CUDA-graph number is given.
  - Source: [Hazy blog](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles); [code](https://github.com/HazyResearch/Megakernels)
- Mirage Persistent Kernel (MPK, OSDI '26) (fetched):
  - Qwen3-8B on A100 at batch 1: vLLM/SGLang take 14.5 ms per token and MPK 12.5 ms, against a bound of about 10 ms. That is about 69% to 80% of the bound by my derivation.
  - Single-batch speedups across models are 1.0-1.7x, larger for smaller models and newer GPUs.
  - MoE (Qwen3-30B-A3B) is supported with a hybrid workload balancer. In SGLang at batch 1, the MoE gather preprocessing takes up to 11% of MoE time.
  - Cross-task pipelining speeds the final linear by 1.2-1.3x on B200.
  - The smallest GPU tested is the A100 (108 SMs).
  - Source: [MPK arXiv 2512.22219](https://arxiv.org/html/2512.22219v1); [OSDI'26 PDF](https://www.usenix.org/system/files/osdi26-cheng.pdf)
- The MPK README claims 1.2-6.7x, but the baseline for the 6.7x figure could not be verified (search summary). Source: [Mirage GitHub](https://github.com/tiledsl/mirage)

### Inferences
- **EXTRAPOLATION, glue segment:**
  - At 32 GB/s, the 6.2 ms of glue moves about 0.2 GB. At roofline that is under 1 ms, so about 5 ms is latency, sync and launch-tail overhead.
  - If the glue is around 400-1000 small graph nodes per token (40 layers x mHC/norm/router ops), Hazy's about 5 µs per-kernel stall model predicts 2-5 ms. That matches.
  - Fusing glue into GEMV epilogues and prologues, or using a persistent kernel with an on-device instruction queue, could recover about 3-5 ms. That is about +7-12% tok/s, the largest overhead lever.
- The dense and expert GEMVs are already at 172-182 GB/s, which is 75-80% of 228 and matches the best megakernel efficiency. A full megakernel would not lift them much. The real gain is overlapping each GEMV's prologue (weight-fetch latency) with the previous op's tail, which Hazy and MPK both attribute their wins to.
- With only 20 SMs, tail effects per kernel are smaller in absolute terms than on a 148-SM B200, but per-launch and dependency latency does not shrink. The relative value of fusion is therefore similar.

### Gaps
- No published megakernel measurement on Jetson or any SM count at or below 20.
- No FlashInfer persistent or TensorRT-LLM batch-1 Blackwell numbers retrieved within budget.

## 5. Blackwell (tcgen05/TMA) and Jetson Thor unified-memory specifics

### Takeaway
Thor's rated bandwidth is 273 GB/s (LPDDR5X-8533, 256-bit). If the engine's ~228 GB/s is a measured ceiling, it is about 84% of spec, which is typical for LPDDR. No independent STREAM-type measurement of Thor was found. There is also no primary evidence that tcgen05/UMMA beats a CUDA-core GEMV at M=1. The any4 authors argue that tensor cores win even at M=1 on their hardware.

### Cited Findings
- Thor 128 GB: LPDDR5X on a 256-bit bus at 273 GB/s (spec) — [RidgeRun wiki](https://developer.ridgerun.com/wiki/index.php/NVIDIA_Jetson_Thor); [flopper spec](https://flopper.io/gpu/nvidia-jetson-agx-thor-t5000-128gb)
- One review lists 276 GB/s "measured" with no method given. That is above spec and should be treated as unreliable — [ThinkRobotics review](https://thinkrobotics.com/blogs/product-reviews-buying-guides/nvidia-jetson-agx-thor-developer-kit-review-blackwell-powered-physical-ai-for-the-next-generation-of-robots)
- ServeTheHome reports 149.1 tok/s on Llama 3.1 8B on Thor, with the configuration not retrieved — [STH](https://www.servethehome.com/nvidia-jetson-agx-thor-developer-kit-blackwell-for-robotics/2/)
- tinygemm/any4: tensor cores "still outperform" scalar matmul at M=1 per the authors' analysis — [any4](https://arxiv.org/pdf/2507.04610)

### Inferences
- If 273 GB/s is reachable at about 90% with large sequential reads (about 245 GB/s), the true roofline floor is about 34-35 ms rather than 37.6 ms. EXTRAPOLATION; the engine should run its own streaming-read probe to fix the denominator.
- The engine's 172-182 GB/s GEMVs are then 63-67% of spec, which leaves more headroom than the 78%-of-floor framing suggests.
- tcgen05 at M=1 only helps if dequant goes through the tensor-core path (NVFP4 native). For trellis or K-quants, decode stays on the CUDA cores, and the issue is ALU-to-bytes balance on 20 SMs.

### Gaps
- No Thor page size, TLB, or LPDDR5X efficiency study was found.
- No published sm_110 GEMV benchmark was found.

## 6. Activation sparsity in MoE experts (TEAL, CATS, R-Sparse, Deja Vu-style) on 2-bit weights

### Takeaway
TEAL gives 1.53x / 1.8x decode at 40% / 50% sparsity on dense Llama/Mistral models and is claimed compatible with quantization. No MoE or 2-bit evaluation was found. Activation sparsity can only skip weight columns. With 2-bit group-quantized or trellis-coded blocks, the skippable unit is the whole quant block, so the realized byte savings at 2 bits are doubtful.

### Cited Findings
- TEAL (ICLR 2025): training-free magnitude sparsity. Single-batch decode is 1.53x at 40% and 1.8x at 50% model-wide sparsity. Evaluated on Llama-2, Llama-3 and Mistral at 7B-70B. "Compatible with weight quantization." — [arXiv 2408.14690](https://arxiv.org/abs/2408.14690v2); [GitHub](https://github.com/FasterDecoding/TEAL)
- Dynamic activation sparsity on 4-bit quantized Llama on Apple chips: 1.32-1.55x (A19 Pro 1.55x, M2 Max 1.45x) — [arXiv 2511.04477](https://arxiv.org/pdf/2511.04477)
- BlockFFN, which needs training, gets up to 3.67x on end-side devices by co-designing chunk-level sparsity with spec decode — [arXiv 2507.08771](https://arxiv.org/html/2507.08771v2)

### Inferences
- For down_proj (input dim = moe_intermediate 2304), TEAL-style input sparsity skips rows of W_down^T. With K-quant super-blocks of 256 along the input dim, a skip needs a whole 256-wide block of zeros, which is rare at 40-50% random sparsity.
- Re-blocking so that the skip granularity is per input channel would require a column-major packing of the 2-bit experts. EXTRAPOLATION; not demonstrated anywhere I found.
- On the NVFP4 dense path (group size 16), skip granularity is finer. Dense attention projections are a more plausible TEAL target than the 2-bit experts.

### Gaps
- No TEAL, CATS, or R-Sparse evaluation on MoE experts or at 2 bits.
- No quality data on DeepSeek-family models.

## 7. Black swans: compressed-form decode, entropy-coded weights, breaking the roofline

### Takeaway
Lossless entropy coding of floating-point exponents is now real on GPUs. ZipServ (fixed-length codes fused into GEMM) beats cuBLAS on Ada by 1.31x and beats DFloat11 by a wide margin end to end. It only applies to bf16 or fp8 tensors, which here are the LM head (1.32 GB) and the fp8 dense path. No published method entropy-codes 2-4-bit quantized weights with on-the-fly GPU decoding that beats a native low-bit GEMV.

### Cited Findings
- DFloat11: about 30% lossless bf16 compression via exponent Huffman with SRAM LUTs. Bit-identical. NeurIPS 2025 — [arXiv 2504.11651](https://arxiv.org/html/2504.11651v3)
- ZipServ: fixed-length TCA-TBE, 30% smaller, fused ZipGEMM 1.31x average over cuBLAS on an RTX 4090, 1.22x end to end over vLLM — [ZipServ](https://www.alphaxiv.org/abs/2603.17435v1)
- ECF8: Huffman-coded FP8 exponents with a parallel GPU decode — [arXiv 2510.02676](https://arxiv.org/pdf/2510.02676)
- QTIP's compute-generated codes are themselves a "compute for bandwidth" trade. Weights are generated from a small trellis state with 2-4 instructions per weight, so no codebook needs to be fetched — [QTIP](https://arxiv.org/html/2406.11235v4)

### Inferences
- Applying ZipServ-style coding to the bf16 head and the fp8 dense weights would save about 0.4 GB + about 0.06 GB, or about 2 ms per token losslessly. EXTRAPOLATION; it assumes the exponent distributions are similar for the head and fp8 tensors.
- Ranked lever list for this engine (all EXTRAPOLATION, ms saved per token from 48.2):
  1. Fix the LM head bytes with an exact NVFP4 shortlist plus a top-K bf16 row gather: about 4 ms.
  2. Glue fusion or persistent epilogue fusion: about 3-5 ms.
  3. 3-bit trellis on the dense weights: about 4-5 ms of bytes, minus the ALU risk on 20 SMs, and it needs an accuracy gate.
  4. IQ*_KT experts at -0.5 bpw for equal quality: about 2 ms.
  5. Static top-5 in the deep layers: about 1-2 ms.
  6. ZipServ on any remaining bf16/fp8: about 0.5-2 ms.
- Combined, these could take the engine to roughly 32-36 ms per token (about 28-31 tok/s). That is below the current 37.6 ms "floor" because levers 1, 3, 4 and 5 shrink the bytes the floor is computed from.

### Gaps
- Huff-LLM and ZipNN primary sources were not retrieved.
- No entropy-coded sub-4-bit GPU GEMV was found.
- No published result breaks the bandwidth roofline at batch 1 without reducing bytes. Speculative decoding is the standard way out and is outside this note's scope.
