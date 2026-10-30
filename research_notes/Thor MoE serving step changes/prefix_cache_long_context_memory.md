# Prefix caching, prefill acceleration, long context and memory efficiency for a single-user DSV4.1-Flash-REAP server on Thor

Scope: what a single-stream CUDA server for DeepSeek-V4.1-Flash-1017 REAP-192 on Jetson Thor should implement. Baseline: 739 tok/s prefill at ~7K tokens (580 tok/s on the 8K bench), 25 tok/s decode, a 1M-token KV state of 1.66 GiB (FP4), about 17 GiB of 122 free, basic "extend-the-last-prompt" prefix reuse, and an encoder/decoder prefill trick.
Research date: 2026-10-30. The search budget was capped at about 20 tool calls, so some primary PDFs were read only in part. Where that matters it is noted.

Most important finding: the DeepSeek-V4.1-Flash tech report (arXiv 2609.19969) describes this exact architecture and DeepSeek's own serving design for it, including SWA Bounded Replay and persistent-cache rules. It is the main reference, and much of the other literature (sparse prefill, KV eviction and quantization) does not apply to this model.

---

## 1. Prefix caching designs (radix tree, APC, LMCache, tiered/NVMe session caches, caching for compressed/sparse KV)

### Takeaway
For this model the right design comes from DeepSeek V4.1 and SGLang. Make the persistent cache unit the compressed "global" KV plus indexer K, cut on compression-block boundaries. Do not persist SWA KV. Recompute it by replaying only the last 128 tokens. Put that inside a small radix tree of sessions, so subagents and branches that share a system prompt all hit. The existing "the new prompt extends the old one" reuse fails exactly where agent traffic diverges. With KV at about 1.7 KB/token, a 32K-token session snapshot is about 57 MB. A cold 32K prefill at 739 tok/s takes about 43 s. A multi-session RAM/NVMe snapshot cache is therefore the largest TTFT lever on the list.

### Cited Findings
**DeepSeek's own design for this architecture (V4.1 report)**
- V4.1 uses a Causal Encoder-Decoder (CED) split: bottom L/2 = 20 layers are the encoder, upper 20 the decoder. Decoder global KV is projected from the encoder's final hidden state, so only encoder layers run full-prompt computation. Prefill complexity goes from O(NL) to O(NL/2 + n_win·L/2), which "nearly halves" prefill compute. 8B active params during prefill vs 16B during decode. No measured TTFT was in the parts read. — [DeepSeek-V4.1-Flash report, arXiv 2609.19969](https://arxiv.org/html/2609.19969)
- **SWA Bounded Replay:** instead of replaying L×n_win tokens for exact SWA reconstruction, replay only the most recent n_win = 128 tokens and truncate SWA to that segment. This is approximate. Encoder replay regenerates only SWA KV and reuses cached global KV. At every prefill the last 128 prompt tokens are replayed through the decoder layers, and the resulting decoder SWA KV is used for decoding and **not written to the prefix cache**. Exact reconstruction would need 2,560 decoder token-layers; bounded replay needs 128 tokens. — [arXiv 2609.19969](https://arxiv.org/html/2609.19969)
- Persistent cache policy: in V4, SWA KV was cached at prompt end and output end and made up nearly half of the persistent cache, with both kinds resident for over 72 h. In V4.1, SWA KV is **not persisted**. It lives in a pool built from 10% of host DRAM with a TTL of minutes. Global KV is persisted for at least 72 h. On a global-KV hit with an SWA miss, only n_win tokens are recomputed. The persistent footprint is about 1/8 of V4-Flash. The paper says replay causes "only negligible performance degradation" but no figure was found in the sections read. — [arXiv 2609.19969](https://arxiv.org/html/2609.19969)
- Global KV (HBM) is 890 bytes/token, about 1/4 of V4-Flash. Main KV is FP4 E2M1 with one E4M3 scale per 16 channels (no NVFP4 global scale), quantized after RoPE, with QAT in post-training. SWA KV stays FP8. — [arXiv 2609.19969](https://arxiv.org/html/2609.19969)
- CSA2 cross-layer layout: encoder layers 1-2 are pure SWA; the remaining 18 run CSA2 with m=2 in 3 groups of (1 Full + 5 Reuse). Decoder layers run m=1 in 5 groups of 4: group 1 is Full + 3 Reuse, groups 2-5 are Reindex + 3 Reuse. Reuse layers share main KV and Top-K indices. Every layer keeps its own SWA KV. The hierarchical sparse indexer (decoder only) has the first Full layer score all positions, take the max per block, and keep up to 2,048 blocks of 8 (16,384 candidates). Later Reindex layers score only those candidates. Top-K = 512, with 32 indexer heads × 128 dims. — [arXiv 2609.19969](https://arxiv.org/html/2609.19969)

**vLLM / SGLang implementations for V4 (the closest analogue; vLLM has V4.1 only via V4 code paths as far as found)**
- vLLM V4: every compressed layer uses a logical block of 256 native tokens (64 entries for c4a, 2 for c128a). Compressor state is registered under the sliding-window KV spec with `sliding_window = coff × compress_ratio` (8 for C4, 128 for C128), so no separate snapshot or residual path is needed. **Cache hits land on 256-position block boundaries**, where the compressor state is the correct handoff point. In disaggregated prefill only blocks inside the window are transferred. 1M context with bf16 KV is 9.62 GiB/sequence for V4. Fusion kernels give 1.4-3x (compressor+norm+RoPE+insert); multi-stream overlap gives 5-6% end-to-end latency at low batch (hardware not stated). — [vLLM blog, DeepSeek V4](https://vllm.ai/blog/2026-04-24-deepseek-v4)
- SGLang **ShadowRadix**: a radix tree over virtual full-token slots, with per-pool "shadows" into the SWA, C4 and C128 pools. SWA slots are tombstoned once the window passes while C4/C128 shadows stay shareable. A 10K-token request keeps 128 SWA tokens plus its full compressed KV. Each node has a two-counter lock (`full_lock_ref`, `swa_lock_ref`). Matching is "SWA-safe": it requires 128 consecutive live tokens from the match point. Compressor ring slot = `swa_page × ring_size + pos % ring_size`. Under speculative decoding, ring sizes are doubled (C4 8→16, C128 128→256) so draft writes cannot clobber live slots. — [LMSYS blog, DeepSeek-V4 on Day 0](https://www.lmsys.org/blog/2026-04-25-deepseek-v4/)
- SGLang HiSparse offloads inactive C4 KV to a pinned CPU mirror and swaps pages in with LRU. It claims up to 3x token capacity and throughput (V4-Flash, 2×B200, 200K in / 20K out). SGLang Lightning TopK: about 256K candidates at 1M ctx; radix-select cut BS=1 top-k from >100 µs to about 15 µs. Flash Compressor reaches up to 80% of peak bandwidth on H200. — [LMSYS blog](https://www.lmsys.org/blog/2026-04-25-deepseek-v4/)
- Known bugs (open reports against vllm-ascend, not settled behaviour): V4 prefix-cache hit rate is 0% for prompts under 4,096 tokens because of the C4 compressor-state block size, and MTP speculative decoding reduces the reported prefix-hit length. — [vllm-ascend #16270](https://github.com/vllm-project/vllm-ascend/issues/16270); [vllm-ascend #9247](https://github.com/vllm-project/vllm-ascend/issues/9247)
- vLLM v0.23.0 added selective prefix-cache retention for sliding-window KV (#43447). — [vLLM v0.23.0 release notes](https://github.com/vllm-project/vllm/releases/tag/v0.23.0)
- V3.2-era DSA: the indexer K cache is a separate buffer, quantized per block (values first, scales after), so a token's indexer entry is not contiguous. Block size 64 only. MLA cache is 656 B/token. The V3.2 blog says nothing about prefix caching. — [vLLM blog, V3.2-Exp](https://vllm.ai/blog/2025-09-29-deepseek-v3-2). SGLang's V3.2 day-0 launch example disables the radix cache on one hardware path. — [LMSYS V3.2 blog](https://www.lmsys.org/blog/2025-09-29-deepseek-V32)
- Together AI on V4: the engine stores compressed CSA/HCA state and rebuilds the SWA path when a prefix is reused. — [Together AI blog](https://www.together.ai/blog/serving-deepseek-v4-why-million-token-context-is-an-inference-systems-problem) (search snippet only, not fetched)

**Tiered and multi-session caches**
- CachedAttention/AttentionStore (USENIX ATC'24) saves a session's KV when idle and reloads it on return, using layer-wise preloading and async saves. It reports up to 87% lower TTFT, 7.8× prefill throughput and 70% lower cost for multi-turn chat (ATC version; the preprint said 88%/8.2×/56%). — [arXiv 2403.19708](https://arxiv.org/html/2403.19708v3)
- SGLang HiCache (GPU→host→storage) claims up to 6× throughput and up to 80% lower TTFT. Novita AI reported average session TTFT −56% and hit rate 40%→80% with the 3FS backend (single deployment report). — [LMSYS HiCache blog](https://lmsys.org/blog/2025-09-10-sglang-hicache)
- NVIDIA Dynamo agentic docs: KV pinning for agent sessions (TTL clamped to 5 min-1 h, write-through to host) and 85-97% cache hits on later calls to the same worker. — [Dynamo SGLang agentic docs](https://docs.nvidia.com/dynamo/backends/sg-lang/agentic-workloads); [Dynamo agentic inference](https://docs.dynamo.nvidia.com/dynamo/digest/agentic-inference)

**Agentic hit rates (what the cache will see)**
- SemiAnalysis AgentX (real coding-agent traces): median 43 turns per session, median input 142K tokens, median output 444 tokens, prefix-cache hit rate >96%. 44% of sessions spawn at least one subagent, with a median of 4 subagent rollouts. — [vLLM × AgentX blog](https://vllm.ai/blog/2026-09-08-vllm-agentx)
- TraceLab (UW; Claude Code / Codex traces): 95.7% token-weighted hit rate overall but only 84.4% on user-initiated steps (secondary write-up). — [TraceLab paper](https://tracelab.cs.washington.edu/paper.pdf); [write-up](https://codex.danielvaughan.com/2026/07/17/tracelab-coding-agent-workload-characterisation-codex-cli-prefix-cache-compaction-token-budget-serving/)
- Microsoft (GitHub Copilot traces, 13.5M sessions sampled): hit rates drop to about 55% at turn boundaries and about 8% when the model switches (search excerpt; PDF not read). — [MSR ghcp traces PDF](https://www.microsoft.com/en-us/research/wp-content/uploads/2026/08/ghcp_traces-6.pdf)

### Inferences
- **Design for our server (extrapolated, not measured):**
  - Keep a radix tree (or an LRU of chains) keyed by token IDs, cut at block boundaries that align with every layer's compression ratio. With m=2 encoder and m=1 decoder CSA2, plus the 8-row candidate blocks at layer 20, a 256-token logical block like vLLM's is a safe LCM-style choice.
  - Each node stores global KV plus indexer K for KV-source layers {2,8,14,20} only, since reader layers reuse them. Nodes also store compressor partial state at the boundary, or else replay up to the last boundary.
  - On a hit, replay the last 128 tokens (SWA Bounded Replay) instead of storing SWA.
  - The replay cost is 128 tokens, about 0.17-0.2 s at 739 tok/s plus fixed launch overhead. Snapshotting the 128-row SWA per layer instead would be about 40 layers × 128 rows. That is small, so for an in-RAM hot tier, storing SWA exactly may be cheaper than replay and keeps bit-exactness. Use replay only for NVMe-tier entries, as DeepSeek does.
- **Size and speed (extrapolated):**
  - KV is 1.66 GiB / 1M tokens, about 1.78 KB/token. A 32K agent context is about 57 MB, and a 142K context (the AgentX median) about 250 MB.
  - With ~17 GiB free, a RAM tier could keep about 60 sessions of 142K, or a few hundred of 32K. An NVMe tier is effectively unbounded.
  - Restoring 250 MB from NVMe at several GB/s takes well under 0.2 s. Recomputing 142K would take minutes at the current prefill rate, which is also length-dependent and degrades. That is orders of magnitude in TTFT on every miss the current single-chain reuse suffers.
- **Why single-chain reuse is insufficient:**
  - Subagents (44% of sessions) share the system prompt and tool schema but diverge. With one KV state, every return from a subagent to the parent re-prefills the whole parent context.
  - The MSR trace's 55% at turn boundaries suggests clients often edit history: compaction, stripping prior reasoning, rewriting tool results. A radix tree still salvages the longest common prefix.
- **Hyper-connection (mHC) streams are per-token**, so there is no recurrent state to snapshot beyond the compressor rings and SWA. This is the same as the "save/restore of recurrent state" problem solved by vLLM's "compressor-state-as-SWA" spec.
- **Watch the two vllm-ascend bugs as test cases:**
  - prompts shorter than one compression block, which got 0% hits;
  - MTP/DSpark speculative decoding writing compressor rings ahead of acceptance, solved by SGLang with doubled rings. A cache snapshot must be taken only at accepted positions.

### Gaps
- No TTFT or hit-rate numbers for SWA Bounded Replay, and no quality delta beyond "negligible", in the parts of the V4.1 report read (about 200K of 212K characters).
- No found source measures LMCache, Mooncake or IMPRESS on a single-user edge box. Mooncake and IMPRESS were not fetched within budget.
- No source states whether DeepSeek's V4.1 chat template strips earlier-turn reasoning in agentic tool flows. That determines whether multi-turn prompts are pure extensions. **Verify against the template.**
- Whether exact SWA snapshot or bounded replay is cheaper on Thor must be measured locally.

---

## 2. Non-prefix reuse (CacheBlend, EPIC, position-independent caching) and its quality cost

### Takeaway
These methods give 2-8× TTFT on RAG-style chunk reuse with about 1-7% quality loss. They were designed for dense full-attention models with standard RoPE KV, and none has been validated on compressed/indexed KV with SWA and a learned indexer. For an agentic single-user server, prefix hit rates are already 84-96%, so this is low priority and high risk.

### Cited Findings
- CacheBlend (EuroSys'25 best paper) reuses KV of non-prefix chunks and selectively recomputes a small subset of tokens. It reports 2.2-3.3× lower TTFT and 2.8-5× throughput vs full recompute across 3 LLMs and 4 datasets. The quality drop is described as "negligible" by the authors and as 1-3% by the UCM docs. — [LMCache blog](https://blog.lmcache.ai/2025-03-31-eurosys); [UCM docs](https://ucm.readthedocs.io/en/latest/user-guide/sparse-attention/cacheblend.html)
- EPIC (ICML'25, LegoLink) says CacheBlend's own overhead is about 16-64% of TTFT. EPIC reports up to 3× better TTFT than CacheBlend for a single request (8× TTFT and 7× throughput across multiple requests) with accuracy loss within 7%. — [EPIC, arXiv 2410.15332](https://arxiv.org/html/2410.15332v3)
- SemPIC (2026) argues that naively stitched position-independent caches lose quality because each chunk was encoded without context, so quality is still an open problem. — [SemPIC](https://www.alphaxiv.org/abs/2607.28069)
- CachedAttention decouples positional encoding from cached KV so that truncating history does not invalidate the cache. That is relevant to clients that drop the oldest turns. — [arXiv 2403.19708](https://arxiv.org/html/2403.19708v3)

### Inferences
- In DSV4.1, a non-prefix chunk's cached compressed KV would also carry wrong indexer K context and wrong compressor block alignment. The cross-layer Reuse/Reindex sharing and CED decoder projection from the encoder's final hidden state also mean that "recompute a few tokens" must touch the encoder stack. Expect worse quality than on Llama-class models. This is an extrapolation; there is no source.
- A cheaper partial win is to cache tokenized and templated tool-schema blocks, so the system prompt stays byte-identical across turns. That keeps prefix hits exact and needs no PIC.

### Gaps
- No source evaluates CacheBlend, EPIC or PIC on MLA/DSA/CSA models.

---

## 3. Prefill acceleration at batch 1

### Takeaway
Classic sparse-prefill methods (MInference, FlexPrefill, XAttention, SampleAttention) do not apply: DSV4.1 attention is already top-512 sparse over compressed KV, and at 7-8K the prefill is GEMM (MoE and projections) bound. The real levers are:
- the CED encoder/decoder split already in place, worth about 2× by DeepSeek's account;
- prefix caching (section 1);
- fused compressor/indexer kernels and multi-stream overlap (5-6% at low batch in vLLM);
- a fast top-k (SGLang: >100 µs to 15 µs at BS=1);
- optionally SpecPrefill-style token dropping, which carries quality risk for agent tool outputs.

### Cited Findings
- On Llama-3.1-8B at 128K RULER, MInference, FlexPrefill and XAttention score about the same (87.30, 87.42, 87.37). Attention-operator speedups over FA2 at 128K are 2.45×, 5.18× and 3.48×. At 4K, XAttention is 0.79× and the others about 0.12×, so all are **slower than dense at short context**. — [FlashPrefill V2, arXiv 2608.19758](https://arxiv.org/pdf/2608.19758)
- XAttention self-reports up to 13.5× attention speedup at 256K. MInference claims up to 10× prefill at 1M on one A100. — [XAttention, arXiv 2503.16428](https://arxiv.org/pdf/2503.16428); [MInference](https://arxiv.org/html/2407.02490v2)
- SpecPrefill (ICML'25): a small speculator selects important prompt tokens and only those reach the base model. Up to 7.66× TTFT and 7× max QPS on Llama-3.1-405B-FP8. The same small model can also draft for speculative decoding. Cross-family speculators were studied in 2026. — [arXiv 2502.02789](https://arxiv.org/abs/2502.02789); [Cross-Family SpecPrefill, arXiv 2603.02631](https://arxiv.org/html/2603.02631v3)
- GemFilter uses an early layer's attention (layer 19 on Mistral-Nemo-12B / Phi-3.5) to select tokens, then runs the full model on the selection. It reports 2.4× speedup and 30% less memory vs SnapKV (70% vs standard attention), with comparable LongBench scores. — [GemFilter, arXiv 2409.17422](https://arxiv.org/pdf/2409.17422)
- LLMLingua-2 claims 1.6-2.9× end-to-end latency at 2-5× compression. An independent 2026 study (>30,000 runs, prompts of 100-50K tokens) found **at most about 18%** end-to-end speedup, and only when length, ratio and hardware are well matched. — [Prompt Compression in the Wild, arXiv 2604.02985](https://arxiv.org/pdf/2604.02985)
- vLLM V4: fused compressor+RMSNorm+RoPE+cache-insert gives 1.4-3×; inverse-RoPE+fp8 quant gives 2-3×; fused Q-norm+KV-RoPE+K-insert gives 10-20× over unfused; multi-stream overlap cuts end-to-end latency 5-6% at low batch. — [vLLM V4 blog](https://vllm.ai/blog/2026-04-24-deepseek-v4)
- SGLang V4: overlaps the indexer Q-projection, weights projection and compressor GEMM on separate streams (small batch only). Prefill context parallelism is multi-GPU only. — [LMSYS V4 blog](https://www.lmsys.org/blog/2026-04-25-deepseek-v4/)
- vLLM AgentX: PCP8 vs TP8 gives 2.65× prefill at 32K (multi-GPU, not applicable). A long-prefill chunk threshold of 512 tokens gave up to 93% higher throughput and 2.3× better P90 interactivity for concurrent users, at the cost of long-request TTFT. — [vLLM AgentX blog](https://vllm.ai/blog/2026-09-08-vllm-agentx)

### Inferences
- For a single user, chunked-prefill sizing only matters if (a) decode of a second stream is interleaved, or (b) compute buffers scale with the chunk. Choose the largest chunk that keeps the MoE grouped-GEMM at full efficiency and fits memory. Interleaving a small decode step into each prefill chunk (prefill-decode overlap) only pays once 2+ streams are served.
- The 739 → 580 tok/s drop from 7K to 8K suggests a cost that grows with length. Per the V4.1 report, the Full-layer indexer scans the whole causal range, O(N²/m) scoring, and layer-20 candidate-block selection kicks in past 16,384 rows. Profiling the indexer share of prefill vs context length is the first step before any sparse-prefill work. This is an inference from architecture; the attribution is not measured.
- **SpecPrefill-style dropping is the only "big multiplier" left at batch 1 (7.66× claimed).** It conflicts with exact prefix caching, because dropped tokens never enter the KV, and with agent correctness on code and tool outputs. If tried, restrict it to one-shot long documents and use the existing MTP drafter or a layer-k indexer score, in the spirit of GemFilter, as the selector. The DSA indexer is in effect a built-in token-importance scorer. This is an extrapolation.
- Prompt compression (LLMLingua-2) is not worth implementing: the independent measurement gave ≤18%, and it destroys prefix-cache hits.

### Gaps
- No public BS=1 prefill profile for V4/V4.1 on any GPU was found (both blogs give kernel-level speedups only).
- No measured CED prefill speedup in the V4.1 text read.

---

## 4. Long-context quality and efficiency (KV quantization, eviction, RoPE extension)

### Takeaway
KV quantization and eviction (KIVI, KVQuant, SnapKV, H2O, PyramidKV) are essentially irrelevant. DSV4.1 KV is already 890 B/token FP4 with QAT, a 1M-token state is 1.66 GiB in our build, and top-512 selection is learned eviction-at-read. RoPE/YaRN extension is unnecessary because the model is trained for 1M. The long-context work worth doing is making the indexer and candidate-block path O(N)-efficient, and fixing llama.cpp's dense+mask fallback.

### Cited Findings
- V4.1 main KV is FP4 E2M1 with E4M3 scales per 16 channels, quantized after RoPE, with QAT. The authors "have not observed any systematic degradation". — [arXiv 2609.19969](https://arxiv.org/html/2609.19969)
- vLLM V4: fp4 indexer cache + fp8 attention cache roughly halves the bf16 9.62 GiB/1M footprint. A packed KV layout saves about 10% more with the FP4 indexer. — [vLLM V4 blog](https://vllm.ai/blog/2026-04-24-deepseek-v4); [vLLM AgentX blog](https://vllm.ai/blog/2026-09-08-vllm-agentx)
- SGLang: V4 decode throughput falls under 10% from 4K to 900K context (B200 199→180 tok/s; H200 266→240 tok/s; single-batch decode). — [LMSYS V4 blog](https://www.lmsys.org/blog/2026-04-25-deepseek-v4/)
- V4.1: going from 4K to 1M context raises single-token decode FLOPs by only about 1/4. — [arXiv 2609.19969](https://arxiv.org/html/2609.19969)
- llama.cpp status:
  - The sparse-KV-indices MMA flash-attention PR, which covered V3.2/GLM-5 and V4 CSA layers, was **closed unmerged** on 2026-09-02. — [llama.cpp #25917](https://github.com/ggml-org/llama.cpp/pull/25917)
  - `GGML_OP_GATHER` for DSA is still open. — [llama.cpp #21458](https://github.com/ggml-org/llama.cpp/pull/21458)
  - Merged 2026-10-27: dropping `kq_mask` for QSA/DSA arches saves about 2 GB of compute buffer at ubatch 4096 and 4 GB at 8192. On GLM-5.3-Flash (RTX 5090) it moved decode from 7.9 to 9.5 tok/s at ub 4096 / 512K. — [llama.cpp #31009](https://github.com/ggml-org/llama.cpp/pull/31009)
  - A CDNA2 lightning-indexer PR is open. — [llama.cpp #29050](https://github.com/ggml-org/llama.cpp/pull/29050)

### Inferences
- The "32K is very slow in llama.cpp" symptom matches upstream state: the sparse gather/FA path is not merged, so attention is dense over all compressed rows. The best upstream route is reviving #25917 (sparse top-k indices in MMA FA) and #21458 (GATHER) for the deepseek4 arch, then taking the #31009 mask removal.
- Our KV-only memory at 1M (1.66 GiB) is not the long-context bottleneck on Thor. Indexer scoring time and Engram I/O per token are. Spend no effort on KV eviction or quantization.

### Gaps
- No V4.1 long-context benchmark numbers (MRCR/RULER vs V4-Flash) were in the sections read.

---

## 5. Memory efficiency on unified-memory edge devices (weight residency, NVMe Engram tables, hot-row caching, io_uring/GDS)

### Takeaway
Engram lookups are deterministic from the input token IDs, so the whole prompt's rows are known before prefill and the next token's rows are known as soon as it is sampled. That is ideal for deduplicated batch reads during prefill and one-token-ahead prefetch during decode. Caching hot n-gram rows in the ~17 GiB of free RAM is the obvious lever, but no source quantifies Engram row Zipf skew. GPUDirect Storage does not help on Thor's unified memory as documented. Keep io_uring O_DIRECT into pinned buffers.

### Cited Findings
- V4.1 Engram: 196B params across two modules at layers 1 and 14 (zero-indexed). N-gram orders {2,3,4}, 8 hash heads, embedding dim 2048 per order, about 16M entries per head table, FP8 tables. Deterministic addressing allows prefetch from host memory via background RDMA. The first module's prefetch overlaps with the first transformer block. During RL, tables are kept GPU-resident. — [arXiv 2609.19969](https://arxiv.org/html/2609.19969)
- The original Engram paper (DeepSeek, "Conditional Memory via Scalable Lookup") says indices are known before the forward pass and can be fetched from host memory over PCIe during earlier layers. Secondary summaries quote "<3%" overhead for a 100B table offloaded to CPU; **not confirmed from the primary PDF**. — [Engram summary](https://arxiviq.substack.com/p/conditional-memory-via-scalable-lookup); [Quintarelli](https://blog.quintarelli.it/?p=38970)
- GDS peer-to-peer mode does not support migratable memory (cudaMallocManaged or system-allocated) on UVM platforms. Without nvidia_fs.ko, cuFile falls back to compatibility mode (POSIX pread/pwrite/aio). No Jetson Thor-specific GDS statement was found. — [NVIDIA GDS docs](https://docs.nvidia.com/gpudirect-storage/index.html)
- llama.cpp `--moe-cache-mib` style expert caches reach 16-32% hit rates on a 16 GB budget (GLM-5.3-Flash, 5090), which shows hot-subset caching is viable for MoE but skew is modest. — [llama.cpp #31009](https://github.com/ggml-org/llama.cpp/pull/31009)
- Internal record (not external): the V4.1 Engram NVMe O_DIRECT row store measured about 1.2 ms/token on Thor. — project memory `thor-base-model-shortlist.md`

### Inferences
- Rough numbers (extrapolated): 8 heads × 3 orders × 2 modules = 48 rows per token. FP8 rows of 2048/8 = 256 B per head-order give about 12 KB/token of useful bytes, but 4 KB O_DIRECT pages make it about 200 KB/token of I/O. The 1.2 ms/token measurement is consistent with that being IOPS-bound, not bandwidth-bound. Check the actual row size against the table layout.
- Prefill: dedup n-gram indices across the prompt (code and agent JSON repeat heavily) and issue one large sorted io_uring batch per Engram layer before the forward pass reaches layer 1 or 14.
- Decode: start the fetch for token t+1 as soon as it is sampled. With MTP/DSpark drafts, prefetch the drafted tokens' n-grams speculatively.
- Hot-row cache: 17 GiB holds about 70M 256 B rows (~7% of 2×101 GB). N-gram distributions over hashed buckets are typically Zipfian for low orders (2-grams), so a frequency-seeded LFU cache from a calibration corpus probably catches most 2-gram hits and fewer 4-gram hits. This is unsourced and needs a local trace replay.
- Huge pages for the pinned Engram cache and KV snapshot pool reduce TLB misses on unified memory. No Thor-specific measurement was found.
- Avoid page-cache-backed reads: drop_caches and OOM wedges are documented hazards in project memory.

### Gaps
- No primary-source Engram prefetch overhead or n-gram row-frequency distribution was found.
- No Jetson Thor GDS/cuFile support statement was found.
- Huge-page gains on Tegra unified memory are not quantified.

---

## 6. Server features for a production single-user server

### Takeaway
Must-haves:
- an OpenAI-compatible `/v1/chat/completions` with SSE streaming and usage stats including `cached_tokens`;
- tool-call parsing for the DeepSeek DSML format;
- cancellation that leaves the KV/radix state consistent, rolled back to the last accepted block boundary;
- pinned `reasoning_effort` and other generation defaults;
- grammar-constrained decoding.

At 25 tok/s (40 ms/token), any modern grammar engine's per-token mask cost (5-140 µs) is under 0.4% overhead, so pick on schema coverage and dynamic-schema compile time. Batching 2-4 streams only helps if the MoE batch curve is favourable. Internal measurements show 1.4-1.5× at most.

### Cited Findings
- XGrammar (MLSys'25) reports near-zero overhead on H100 (SGLang TPOT 6.8 ms with XGrammar vs 44.2 ms with Outlines, Llama-3.1-8B). An independent 2026 paper measured per-step masks of about 5-10 µs for XGrammar vs 73-141 µs for llguidance (Qwen3-8B, 151K vocab). llguidance compiles fastest thanks to lazy Earley parsing. — [XGrammar, arXiv 2411.15100](https://arxiv.org/pdf/2411.15100); [Trie Automata, arXiv 2608.12574](https://arxiv.org/pdf/2608.12574)
- llguidance README: about 1.5 ms for a full mask on a typical JSON schema (128K tokenizer), with near-zero startup. — [llguidance](https://github.com/guidance-ai/llguidance)
- XGrammar-2 (2026) claims over 6× speedup vs llguidance for dynamic tool-calling and conditional formats (author-reported). — [arXiv 2601.04426](https://arxiv.org/pdf/2601.04426)
- JSONSchemaBench: XGrammar has the fewest compile errors but the most under-constrained failures; Guidance/llguidance has the best coverage. — [arXiv 2501.10868](https://arxiv.org/pdf/2501.10868)
- llama.cpp merged DeepSeek V3.2/V4 chat-parser fixes for message delimiters and for keeping DSML markup out of string tool arguments. These are useful reference implementations of the tool-call format. — [llama.cpp #29008](https://github.com/ggml-org/llama.cpp/pull/29008); [llama.cpp #28612](https://github.com/ggml-org/llama.cpp/pull/28612)
- Dynamo agentic serving: KV pinning per session with TTL (5 min-1 h) and write-through to host. This is the multi-turn session-cache model to copy. — [Dynamo agentic docs](https://docs.nvidia.com/dynamo/backends/sg-lang/agentic-workloads)
- Internal records (not external): spec-decode and batch on the GLM-5.3 CUDA server hit an MoE-routing wall (batch curve about 1.4-1.5×); V4.1 spec math gave 1.6× but chat 0.98×. Servers disagree on default `reasoning_effort`, which voided an earlier head-to-head. — project memory `glm53-cuda-server-spec-batch.md`, `v41-cuda-zenith-mandate.md`, `eval-pin-generation-config.md`

### Inferences
- Use llguidance or XGrammar for JSON tool arguments with lazy grammars, triggered only after the tool-call start token, so free-text turns pay nothing.
- With a single shared KV state, 2-4 stream batching requires the multi-session KV pool from section 1 anyway. Build the pool first, then batching becomes a scheduler feature. The expected gain is bounded by the measured batch curve, about 1.4× aggregate, and is not worth it for interactive single-user latency.
- Cancellation must truncate the radix node to the last committed block, and must discard speculative compressor-ring writes per SGLang's doubled-ring scheme.
- Report `prompt_tokens_details.cached_tokens` so clients and evals can see hit rates. Assert the served model identity in responses, per the orphaned-server lesson in project memory.

### Gaps
- No primary measurement of grammar-engine overhead on DeepSeek's 129K vocab at batch 1 on ARM CPUs (Thor's Neoverse cores). Masks are computed on the CPU, so ARM numbers may be 2-3× x86. That is unmeasured and needs a local check.
