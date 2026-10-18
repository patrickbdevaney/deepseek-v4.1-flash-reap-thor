# Target models: DeepSeek-V4.1-Flash-1017 and MiMo-V2.6-Pro-MOPD — architecture, on-disk format, compressibility

Method note: every byte figure below was computed directly from the live HF repos on 2026-10-17 by downloading `config.json`, `model.safetensors.index.json` and the safetensors *header* of every shard (HTTP range requests; 48 shards for DeepSeek, 130 in-index shards + 2 sidecar shards for MiMo), then summing `data_offsets` per tensor. Parameter counts treat packed 4-bit tensors as 2 params/byte and exclude scale tensors. GiB = 2^30 bytes; GB = 10^9 bytes. Repo SHAs: DeepSeek `1bae0e1088993b9b82de08f59de5be5a34e42170` (lastModified 2026-10-17T05:46Z), MiMo `6a7f202e8dbd0e57b3fe1da03917d08d75abc4b8` (lastModified 2026-10-09T06:30Z) — [DeepSeek API](https://huggingface.co/api/models/deepseek-ai/DeepSeek-V4.1-Flash-1017), [MiMo API](https://huggingface.co/api/models/XiaomiMiMo/MiMo-V2.6-Pro-MOPD).

## Q1. DeepSeek-V4.1-Flash-1017 fact sheet (architecture, dtypes, bytes)

### Takeaway
763.2B parameters on disk = 552B backbone + 196.9B Engram + 14.2B DSpark drafter, shipped as 475.24 GiB (510.3 GB) of mixed-precision safetensors: routed experts are already MXFP4 (E2M1 packed, E8M0 scale per 32), everything dense is FP8 E4M3 with 32x32 E8M0 block scales, Engram tables are FP8, embeddings/head/vision are BF16. All 40 layers are MoE (384 routed + 1 shared, top-6); attention is a 512-channel MLA-style latent with 128-token SWA plus top-512 compressed sparse attention; 1M context; MIT licence.

### Cited Findings
**Config (verbatim from `config.json`)** — [config.json](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/resolve/main/config.json)
- `architectures: DeepseekV41ForCausalLM`, `model_type: deepseek_v41`, `transformers_version: 5.6.0`, pipeline `image-text-to-text` (native vision).
- `quantization_config: {quant_method: "fp8", activation_scheme: "dynamic", weight_block_size: [32, 32], scale_fmt: "ue8m0", expert_dtype: "fp4"}`.
- text: `hidden_size 5120`, `num_hidden_layers 40`, `vocab_size 129280`, `moe_intermediate_size 2304`, `n_routed_experts 384`, `n_shared_experts 1`, `num_experts_per_tok 6`, `scoring_func sqrtsoftplus`, `topk_method noaux_tc`, `norm_topk_prob true`, `routed_scaling_factor 1.5`, `swiglu_limit 10.0`.
- attention: `num_attention_heads 64`, `num_key_value_heads 1`, `head_dim 512`, `qk_rope_head_dim 64`, `q_lora_rank 1280`, `o_lora_rank 1024`, `o_groups 8`, `sliding_window 128`, `index_n_heads 32`, `index_head_dim 128`, `index_topk 512`, `kv_source_layer_ids [2, 8, 14, 20]`, `index_source_layer_ids [2, 8, 14, 20, 24, 28, 32, 36]`, `compress_ratios` = `[0,0, 2×18 (layers 2-19), 1×20 (layers 20-39), 0,0,0 (3 DSpark layers)]`, `compress_rope_theta 160000`, `candidate_source_layer_id 20`, `candidate_topk_blocks 2048`, `candidate_block_size 8`.
- context: `max_position_embeddings 1048576`, `rope_scaling: yarn, factor 16, original_max_position_embeddings 65536`, `rope_theta 10000`.
- hyper-connections: `hc_mult 4`, `hc_sinkhorn_iters 20`, `hc_eps 1e-6`.
- Engram: `engram_layer_ids [1, 14]`, `engram_num_embeddings [384006168, 384016682]`, `engram_max_ngram_size 4`, `engram_vocab_size 16000000`, `engram_n_heads 8`, `engram_head_dim 256`, `engram_compressed_vocab_size 99092`.
- DSpark drafter: `num_nextn_predict_layers 3`, `dspark_block_size 5`, `dspark_target_layer_ids [37, 38, 39]`, `dspark_markov_rank 256`, `dspark_n_routed_experts 128`, `dspark_num_experts_per_tok 3`.
- vision: `num_hidden_layers 32`, `hidden_size 1024`, `num_attention_heads 16`, `intermediate_size 2816`, `patch_size 14`, `downsample_ratio 3`, `max_image_tokens 1024`.
- `generation_config.json` does not exist in the repo (404) and the README says "This release does not include a Jinja-format chat template" — [README](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/resolve/main/README.md).

**Byte accounting (computed from all 48 shard headers; sum 510,286,023,000 B = 475.24 GiB = 510.3 GB, matches `metadata.total_size` in the index)** — [index.json](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/resolve/main/model.safetensors.index.json), [HF file listing](https://huggingface.co/api/models/deepseek-ai/DeepSeek-V4.1-Flash-1017?blobs=true)

| Tensor group | On-disk dtype (exact header dtypes) | Bytes | % | Params |
|---|---|---:|---:|---:|
| Routed experts `layers.N.ffn.experts.E.w1/w2/w3` (40×384×3 = 46,080 matrices) | `I8` packed E2M1 `[2304, 2560]` (= 2304×5120 values, 2 per byte) + `F8_E8M0` scale `[2304, 160]` (block 32) → MXFP4 | 253.12 + 15.82 = **268.95 GiB** | 56.6% | 543.58 B |
| Engram tables `layers.{1,14}.engram.embed.weight/scale` + `engram.wkv` | `F8_E4M3 [384006168, 256]` + `F8_E8M0 [384006168, 8]` (block 32); wkv FP8 `[25600, 6144]` | 183.40 + 5.72 + 0.29 = **189.13 GiB** | 39.8% | 196.93 B |
| DSpark drafter `mtp.0-2.*` (3 layers × 128 experts MXFP4 + FP8 attention/shared + BF16 markov/confidence heads) | I8/E8M0 experts 6.73 GiB; FP8 dense 0.53; BF16 0.13 | **7.39 GiB** | 1.6% | 14.23 B |
| Attention incl. indexer + compressor (`wq_a/wq_b/wkv/wo_a/wo_b`) | `F8_E4M3` + `F8_E8M0` 32×32 block scales; compressor/indexer `wk`/`weights_proj` BF16 | **4.80 GiB** | 1.0% | 5.13 B |
| Embedding + LM head | `BF16 [129280, 5120]` ×2 | **2.47 GiB** | 0.5% | 1.32 B |
| Shared experts (1/layer, 2304-wide) | `F8_E4M3` + E8M0 | **1.32 GiB** | 0.3% | 1.42 B |
| Vision tower (32 blocks) + aligner | `BF16` | **0.90 GiB** | 0.2% | 0.49 B |
| Router gates (`gate.weight` BF16 `[384,5120]`, `gate.bias`/`bias_vl` F32) | BF16/F32 | 0.15 GiB | 0.0% | 0.08 B |
| Hyper-connection params (`hc_attn_fn`/`hc_ffn_fn` F32 `[24, 20480]` per layer) | `F32` | 0.15 GiB | 0.0% | 0.04 B |
| Norms, sinks | BF16/F32 | 0.001 GiB | | |
| **Total** | | **475.24 GiB** | 100% | **763.21 B** |

- Cross-check: 763.21 − 196.93 (Engram) − 14.23 (DSpark) = 552.05 B, matching DeepSeek's "552B backbone parameters" and the recipe's "552B backbone parameters plus 196B of Engram memory" — [arXiv 2609.19969](https://arxiv.org/abs/2609.19969); [vLLM recipe](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash).
- Active params: "8B parameters per token during prefill and 16B during decode" — [arXiv HTML](https://arxiv.org/html/2609.19969v1). Decode check from headers: 6 routed × 3 × 2304 × 5120 × 40 = 8.49 B + shared 1.42 B + attention 5.13 B + embed/head 1.32 B ≈ 16.4 B.
- Dense vs MoE layers: the headers contain `layers.N.ffn.experts.*` for exactly 40 × 384 = 15,360 experts, i.e. **all 40 layers are MoE; there are zero dense-FFN layers**; each layer also has 1 shared expert (computed from index).
- Causal Encoder-Decoder: "The 40 layers form a 20-layer causal encoder and a 20-layer decoder. The first two encoder layers use SWA only. Decoder global KV is projected from the encoder's final hidden state" — [arXiv HTML](https://arxiv.org/html/2609.19969v1). This matches `compress_ratios` (layers 0-1 = 0, 2-19 = 2, 20-39 = 1) and `kv_source_layer_ids [2,8,14,20]`.
- Attention type (from the reference implementation): "Latent attention over two KV sources at once ... a sliding window of raw KV, plus -- when compress_ratio > 0 -- `index_topk` compressed positions reaching further back. Q and the output projection are both low-rank, the latter grouped." `wkv = Linear(dim, head_dim)` i.e. a single 512-channel KV latent shared by all 64 query heads (MLA-style, `num_key_value_heads 1`), `wq_b: q_lora_rank 1280 → 64×512`, output via `wo_a` (8 groups × o_lora_rank 1024) then `wo_b` — [inference/model.py](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/resolve/main/inference/model.py). "compress_ratio > 0 does not mean the layer compresses its own KV: only kv_source_layers do, the rest read that same cache" (same file).
- The compressed KV latent is quantized to FP4 in-cache: "Compressed KV uses groups of 16 with E4M3 scales; the indexer uses 32 with E8M0." `fp4_act_quant(latent, 16, True, scale_dtype=torch.float8_e4m3fn)` — [inference/model.py](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/resolve/main/inference/model.py). Paper: "Main KV cache: FP4 with QAT, E2M1 values with one E4M3 scale per 16 channels ... SWA KV stays FP8" — [arXiv HTML](https://arxiv.org/html/2609.19969v1).
- Hyper-connections: "the residual stream is carried as hc_mult parallel copies" (hc_mult = 4); paper: "Expansion factor n = 4 residual streams, with 20 Sinkhorn-Knopp iterations" — [inference/model.py](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/resolve/main/inference/model.py); [arXiv HTML](https://arxiv.org/html/2609.19969v1).
- DSpark (the only speculative path; "V4.1 dropped the MTP module used in V3 and V4"): "three stages with 128 routed experts each (3 activated). It drafts a 5-token block and reads the attention input of layers 37-39 ... The drafter adds roughly 14B parameters" — [vLLM recipe](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash). Header count: 14.23 B params / 7.39 GiB (computed).
- 1017 vs 0909: README: "DeepSeek-V4.1-Flash-1017 is a minor update to DeepSeek-V4.1-Flash" with the same architecture; recommended `temperature 1.0`, `top_p 1.0`, `context_window 1M`, `max_tokens ≥ 256K` — [README](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/resolve/main/README.md). The Engram tables changed between releases: "The tables changed in the 1017 release. Do not reuse the Engram shards of the older deepseek-ai/DeepSeek-V4.1-Flash" — [LibertAI 1017 EXL3 card](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw).
- Licence: `license: mit` in README front-matter, `LICENSE` file (1,084 B) present, HF tag `license:mit` — [README](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/resolve/main/README.md).
- Vision tower: 32-layer ViT, hidden 1024, BF16, 0.90 GiB incl. aligner (computed); inference needs it only for image inputs; vLLM offers `--language-model-only` — [nvidia NVFP4 card](https://huggingface.co/nvidia/DeepSeek-V4.1-Flash-NVFP4).

### Inferences
- The on-disk format is a strict "MXFP4 experts / MXFP8-block-32 dense / FP8 Engram / BF16 embeddings" split; there is no BF16 master copy of the experts anywhere in the repo, so any requantization of routed experts starts from 4-bit codes.
- The 3 DSpark layers are MoE blocks with their own 128-expert pool (6.73 GiB) — they are a separable, optional drafter (only needed for speculative decoding), so they count toward the "floor" only if spec-decode is wanted.
- Per-layer SWA KV (128 × 512 ch) is a fixed ring buffer and does not grow with context.

### Gaps
- `inference/README.md` says the self-test exercises "the real dense-fp8 / MoE-fp4 kernels" but no file in the repo states whether the routed experts were trained with FP4 QAT or post-training quantized; the arXiv text I could fetch only confirms FP4 QAT for the KV cache and indexer. UNVERIFIED whether a higher-precision expert master exists at DeepSeek.
- The reference `window_kv_cache` buffer is allocated with `torch.zeros` default dtype; the paper says SWA KV is FP8 but I did not confirm the serving-engine SWA dtype.

## Q2. DeepSeek-V4.1-Flash-1017: pre-quantized FP4 experts and the Engram component — confirmed

### Takeaway
Both prior findings hold for the -1017 build: routed experts ship as MXFP4 (E2M1 + E8M0/32, 268.95 GiB) and Engram is 189.13 GiB = 203.1 GB (the "~203 GB" figure is the decimal-GB size of the FP8 tables). Engram is a hashed n-gram lookup table read sparsely (24 rows × 256 ch per layer per token ≈ 12.4 KiB/token total), so it need not be GPU-resident (vLLM supports pinned-host offload with unchanged output) and it can be stored at MXFP4 (97.28 GiB) at a small measured cost; it cannot be removed (perplexity roughly doubles) and no row-pruning method is published.

### Cited Findings
**FP4 experts in -1017**
- Header evidence: every `layers.N.ffn.experts.E.w{1,2,3}.weight` is dtype `I8` with shape `[2304, 2560]` (w1/w3) or `[5120, 1152]` (w2) — half the logical width — paired with `F8_E8M0` scales of shape `[2304, 160]` / `[5120, 72]`, i.e. one scale per 32 values (computed from [shard headers](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/main)). `config.json`: `expert_dtype: "fp4"`, `scale_fmt: "ue8m0"` — [config.json](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/resolve/main/config.json).
- "DeepSeek ships the routed experts in MXFP4. NVFP4's 16-element blocks are a strict refinement of MXFP4's 32-element blocks, and every E8M0 power-of-two scale is exactly representable as E4M3 × a power-of-two global scale. So the same 4-bit codes map to NVFP4 with zero error." — [LibertAI 1017-NVFP4 card](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-NVFP4). Same card: routed experts 268.95 GiB upstream → 284.77 GiB as NVFP4 (finer scales cost +16 GiB).
- NVIDIA on the 0909 build: "The source experts already use four-bit weights. The finer NVFP4 scale layout increases checkpoint size from approximately 476 GiB to 492 GiB" and "All 16,986,931,200 weight blocks passed lossless conversion" — [nvidia/DeepSeek-V4.1-Flash-NVFP4](https://huggingface.co/nvidia/DeepSeek-V4.1-Flash-NVFP4).
- Going below 4 bits is lossy and has been done: LibertAI EXL3 2.9 bpw (experts 3 bpw, layers 18-22 at 2 bpw) 196.3 GiB excluding Engram: perplexity 1.8749 → 1.9094 (code, +1.8%) and 6.4277 → 6.4520 (prose, +0.4%) vs the dequantized base — [LibertAI 1017 EXL3 card](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw).

**What Engram is**
- Reference code: "Writes an n-gram lookup into the residual stream, gated by how well it matches that stream. The hash ids fetch `n_hash_cols` rows; `wkv` turns them into one key per hc copy plus a shared value." `n_hash_cols = (max_ngram_size − 1) × n_heads` = 3 × 8 = 24 rows of 256 ch per token per Engram layer; "The n-gram hash table, sharded over its rows. Stays fp8: rows are dequantized on lookup." — [inference/model.py](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/resolve/main/inference/model.py).
- Hashing: tokens are mapped to a 99,092-entry normalized ("compressed") vocab, then 2-/3-/4-gram hashes XOR-combined and reduced modulo per-(ngram,head) primes just above 16,000,000; "every hash multiplier derives from the compressed vocab size, so a mismatch there would silently rehash the whole table" — [inference/engram.py](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/resolve/main/inference/engram.py). Row counts 384,006,168 / 384,016,682 = 24 prime buckets ≈ 16M each (computed).
- Paper: "196B Engram parameters, split evenly across two modules at layers 1 and 14 ... N-gram orders {2,3,4}, 8 hash heads ... Embeddings and key/value projections use FP8 ... lookups are deterministic, so embeddings can be prefetched from host memory via background RDMA transfers" — [arXiv HTML](https://arxiv.org/html/2609.19969v1). Modifications vs the Jan-2026 Engram paper: "the short causal convolution is dropped, and the embeddings use momentum-based updates followed by Sinkhorn balancing" — [search summary of V4.1 report](https://arxiv.org/pdf/2609.19969).

**Size (the "203 GB" question)**
- Computed: `layers.1.engram.embed.weight` + `layers.14.engram.embed.weight` = 183.40 GiB FP8 + 5.72 GiB E8M0 scales + 0.29 GiB wkv = 189.13 GiB = **203.08 GB** (189.13 × 1.0737). So the prior "~203 GB" is the decimal-GB size and is unchanged in -1017 (same row counts as the 0909 config: vLLM recipe "Engram tables (FP8) 183.1 GiB") — [vLLM recipe](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash).
- Engram lives in shards 47 and 48 only (header sizes 656 B and 664 B, i.e. two tensors each) — computed from headers; the EXL3 kit confirms "shards model-00047-of-00048.safetensors and model-00048-of-00048.safetensors" hold the tables — [LibertAI 1017 EXL3 card](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw).

**Separable / resident?**
- vLLM: "Offload is controlled by `--engram-config '{"cpu_offload":true}'`, which moves the tables to pinned host memory ... Host offload is read through UVA, and the page says the output is unchanged." Recommended 8×H200 config: "TP4 with Engram CPU offload, leaving about 81.2 GiB of resident weights and 38.5 GiB of KV per GPU"; "DGX Station GB300: TP1, with both Engram tables and 60 GiB of routed experts in pinned host memory" — [vLLM recipe](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash).
- LibertAI REAP deployment on 2× GB10: "weights ~94 GiB/rank; Engram 47.2 GiB/rank left on disk" — [LibertAI REAP-256E card](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-REAP-256E).
- EXL3 kits ship without the tables and read them from the base repo's shards 47/48 at serve time — [LibertAI 1017 EXL3 card](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw).

**Removable? No.**
- "Engram check on this checkpoint: with the Engram lookups switched off in the same forward, perplexity rises to 4.1648 (code) / 9.0627 (prose)" vs 1.9094 / 6.4520 with them on — [LibertAI 1017 EXL3 card](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw).

**Quantizable? Yes, with measured cost.**
- NVIDIA (10/14/2026 revision): Engram "converted from the source FP8 to MXFP4: E2M1 values packed two per byte, with one UE8M0 scale per 32 elements. Unlike the expert conversion, this step is lossy (about 12% relative L2 error on the table)"; checkpoint 476 → 400 GiB; "With the tables in pinned host memory (the vLLM default), they take 12.2 GiB per rank at TP=4 instead of 23.6 GiB"; needs vLLM PR #61530; "not supported in SGLang yet" — [nvidia/DeepSeek-V4.1-Flash-NVFP4](https://huggingface.co/nvidia/DeepSeek-V4.1-Flash-NVFP4).
- LibertAI 1017-NVFP4: Engram 188.83 → 97.28 GiB as MXFP4, "lossy, cosine 0.9934"; total 475.24 → 399.51 GiB — [LibertAI 1017-NVFP4 card](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-NVFP4).
- MLX ladder (0909 build): "6-bit engram is indistinguishable from the shipped fp8 tables (free-running 0.1945 vs 0.1948) while 4-bit engram costs +7.3% free-running" — [pipenetwork engram6 card](https://huggingface.co/pipenetwork/DeepSeek-V4.1-Flash-MLX-mixed-4_8bit-engram6).
- INT4 g32 Engram + INT4 GPTQ experts (0909 build): 451.7 GB; "gsm8k (raw 5-shot, thinking off, n=1319): 93.93 / 94.01 vs baseline 92.87 — lossless within noise" — [INCModel3 W4A16-Engram-AutoRound](https://huggingface.co/INCModel3/DeepSeek-V4.1-Flash-W4A16-Engram-AutoRound).

### Inferences
- Per-token Engram read traffic ≈ 2 layers × 24 rows × (256 B FP8 + 8 B scales) = 12,672 B ≈ 12.4 KiB/token (from the module code); at 4-bit it halves. This is why host/NVMe residency works: the table is bandwidth-trivial but capacity-huge.
- Engram is addressed by hash modulo fixed primes, so shrinking the row count means re-hashing with new primes and would scramble every lookup; "pruning" Engram rows is therefore not a drop-in operation like expert pruning. Quantization (FP8 → MXFP4/INT4-g32/6-bit) is the only demonstrated Engram lever.
- For a 117 GiB-class device the tables must be off-device: even at MXFP4 (97.3 GiB) they would consume ~83% of the envelope before any backbone weights.

### Gaps
- No source reports an ablation of Engram *size* (fewer heads/rows) on the trained model; the paper excerpt only notes the dropped causal convolution. UNVERIFIED whether a partially-loaded table (e.g. most-frequent buckets only) degrades gracefully.
- The vLLM "output is unchanged" claim for host offload is the recipe author's statement; I found no independent numeric check.
- The 1017 Engram tables differ from 0909 (per LibertAI's sha256 note) but no source quantifies how.

## Q3. MiMo-V2.6-Pro-MOPD fact sheet (architecture, dtypes, bytes)

### Takeaway
1.024T parameters on disk (1.02T headline), 527.15 GiB in the main index (+5.16 GiB DFlash drafter +1.74 GiB audio tokenizer = 534.07 GiB safetensors). 70 layers: layer 0 is global-attention + dense FFN; the other 69 are MoE (384 routed, top-8, no shared experts) with 60 SWA (window 128) and 10 global-attention layers; GQA 128 Q / 8 KV heads, head_dim 192 (QK) / 128 (V). Routed experts are stored MXFP4 (`save_format: mxfp4`) and make up 93.9% of bytes; attention is split FP8 (`qkv_proj`) / BF16 (`o_proj`, explicitly un-quantized); 1M native context; MIT licence.

### Cited Findings
**Config (verbatim from `config.json`)** — [config.json](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-MOPD/resolve/main/config.json)
- `architectures: MiMoV2ForCausalLM`, `model_type: mimo_v2`, `custom_code` (`auto_map` → `modeling_mimo_v2.py`), `transformers_version 5.3.0`.
- `hidden_size 6144`, `num_hidden_layers 70`, `vocab_size 152576`, `intermediate_size 16384` (dense FFN), `moe_intermediate_size 2048`, `n_routed_experts 384`, `n_shared_experts null`, `num_experts_per_tok 8`, `scoring_func sigmoid`, `topk_method noaux_tc`, `n_group 1`, `topk_group 1`, `norm_topk_prob true`, `routed_scaling_factor null`, `moe_router_dtype bfloat16`.
- `moe_layer_freq`: `[0, 1×69]` → layer 0 dense, layers 1-69 MoE. `hybrid_layer_pattern` (0 = global attention, 1 = SWA): zeros at indices 0, 7, 15, 23, 31, 39, 47, 55, 62, 69 → 10 GA / 60 SWA (counted from the array).
- attention: `num_attention_heads 128`, `num_key_value_heads 8`, `head_dim 192`, `v_head_dim 128`, `swa_num_attention_heads 128`, `swa_num_key_value_heads 8`, `swa_head_dim 192`, `swa_v_head_dim 128`, `sliding_window 128`, `attention_chunk_size 128`, `attention_projection_layout fused_qkv`, `attention_value_scale 0.612`, `add_swa_attention_sink_bias true`, `add_full_attention_sink_bias false`, `partial_rotary_factor 0.334`, `rope_theta 10000000` (GA), `swa_rope_theta 10000`, `rope_type default` (no YaRN).
- context: `max_position_embeddings 1048576`.
- `quantization_config: {quant_method: "fp8", fmt: "e4m3", activation_scheme: "dynamic", weight_block_size: [128, 128], store_dtype: "mxfp4", mxfp4_block_size: 32, ignored_layers: [model.layers.0-69.self_attn.o_proj, model.decoder.self_attn.o_proj]}`.
- vision: `depth 28`, `hidden_size 1280`, `num_heads 32`, `num_key_value_heads 8`, `fullatt_block_indexes [0, 9, 18, 27]`, `patch_size 16`, `spatial_merge_size 2`, `use_sink true`; audio: 6-layer local transformer (`input_local_layers 6`, dim 1024), `audio_segment_size 6000`, 20 RVQ codebooks.
- `generation_config.json`: `temperature 1.0`, `top_p 0.95`, `eos_token_id [151643, 151645, 151672]` — [generation_config.json](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-MOPD/resolve/main/generation_config.json).
- Index metadata: `{"save_format": "mxfp4", "total_size": 566027990784, "tp_size": 8}` — [index.json](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-MOPD/resolve/main/model.safetensors.index.json).

**Byte accounting (computed from all 130 in-index shard headers; sum 566,027,990,784 B = 527.15 GiB = 566.0 GB, matches index `total_size`)**

| Tensor group | On-disk dtype (exact header dtypes) | Bytes | % | Params |
|---|---|---:|---:|---:|
| Routed experts `model.layers.N.mlp.experts.E.{gate,up,down}_proj` (69×384×3 = 79,488 matrices) | `.weight` `U8 [2048, 3072]` (= 2048×6144 values, E2M1 packed 2/byte); `.weight_scale` `U8 [2048, 192]` (E8M0, one per 32) → MXFP4 | 465.75 + 29.11 = **494.86 GiB** | 93.9% | 1000.19 B |
| Attention (70 layers) | `qkv_proj` `F8_E4M3 [27136, 6144]` + `F32` `weight_scale_inv [216, 48]` (128×128 blocks) = 10.87 GiB; `o_proj` **`BF16 [6144, 16384]`** = 13.13 GiB | **24.00 GiB** | 4.6% | 18.72 B |
| Embedding + LM head | `BF16 [152576, 6144]` ×2 | **3.49 GiB** | 0.7% | 1.87 B |
| MTP `model.mtp.layers.0-2` (3 layers: FP8 qkv + dense FFN, BF16 o_proj + `eh_proj [6144, 12288]`) | F8_E4M3 1.31 GiB + BF16 0.98 GiB | **2.29 GiB** | 0.4% | 1.93 B |
| Vision tower `visual.*` (28 blocks + merger) | `BF16` | **1.38 GiB** | 0.3% | 0.74 B |
| Audio encoder + 20 speech embeddings | `BF16` | **0.55 GiB** | 0.1% | 0.29 B |
| Router gates (`mlp.gate.weight` BF16 `[384, 6144]` ×69, `e_score_correction_bias` F32) | BF16/F32 | 0.30 GiB | 0.1% | 0.16 B |
| Dense FFN, layer 0 (`gate/up/down_proj` `[16384, 6144]`) | `F8_E4M3` + F32 128×128 scales | 0.28 GiB | 0.1% | 0.30 B |
| Norms, sink biases | BF16 | 0.002 GiB | | |
| **Main index total** | | **527.15 GiB** | 100% | **1024.22 B** |
| Sidecar `dflash/dflash_draft_model.safetensors` (5-layer SWA drafter, BF16, window 1024, block 8, target layers [0,15,31,47,69]) | BF16 | 5.16 GiB (5,536,639,384 B) | | 2.77 B |
| Sidecar `audio_tokenizer/model.safetensors` (24-layer enc + 24-layer dec codec, 20 RVQ) | | 1.74 GiB (1,872,618,384 B) | | ~0.94 B |
| **All safetensors in repo** | | **534.07 GiB** (573,457,361,664 B) | | |

Sources: [index.json](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-MOPD/resolve/main/model.safetensors.index.json), [dflash/config.json](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-MOPD/resolve/main/dflash/config.json), [dflash index](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-MOPD/resolve/main/dflash/model.safetensors.index.json) (`total_size 5536632576`), [audio_tokenizer/config.json](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-MOPD/resolve/main/audio_tokenizer/config.json), [HF file listing](https://huggingface.co/api/models/XiaomiMiMo/MiMo-V2.6-Pro-MOPD?blobs=true).

- Headline counts: "Sparse MoE, 1.02T total / 42B activated parameters", "Vision Encoder: 681M-param MiMo ViT (28 layers: 24 SWA + 4 Full)", "Audio Encoder: 308M AudioTokenizer + 127M audio patch encoder", "Multi-Token Prediction (MTP): 5-layer speculative decoder" — [README](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-MOPD/resolve/main/README.md). Active-param check from headers: 8 × 3 × 2048 × 6144 × 69 = 20.84 B routed + 18.72 B attention + 0.30 B dense + 1.87 B embed/head + 0.16 B gates ≈ 41.9 B.
- Tech report Table 1 (Pro column): "Layers (Total/SWA/GA) 70/60/10; Hidden Size 6144; SWA Heads (Q/KV) 128/8; Sliding Window Size 128; GA Heads (Q/KV) 128/8; Head Dimensions (QK/V) 192/128; Experts (Total/Activated) 384/8; # Total Parameters 1.02T; # Active Parameters 42B" — [MiMo_V2_6_technical_report.pdf](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL/blob/main/MiMo_V2_6_technical_report.pdf).
- "The only exception is the very first Transformer block, which uses global attention with a dense Feed-Forward Network (FFN) ... Both the SWA block and the GA block utilize a sparse MoE FFN without shared experts." — tech report §2.1.
- Two drafters exist: the report's "MTP-3 configuration inherited from SFT" (the in-index `model.mtp.layers.0-2`) and "block-6 DFlash ... The drafter comprises 5 Transformer layers with dense feed-forward networks" (the `dflash/` sidecar); "average accepted length is 31.3% higher than with the MTP configuration" — tech report §2.1/§6.4. ggml-org ships both as sidecars ("MTP sidecars (Q4_0 and Q8_0) ... DFlash drafter sidecar (BF16 and Q8_0)") — [ggml-org GGUF card](https://huggingface.co/ggml-org/MiMo-V2.6-Pro-MOPD-GGUF).
- Context: "We train RL at a 1M-token context length." — tech report §6.4; README "Context Length: 1M tokens".
- Licence: README front-matter `license: mit`, HF tag `license:mit`; no `LICENSE` file is present in the file listing — [README](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-MOPD/resolve/main/README.md).
- The shipped `modeling_mimo_v2.py` has no MXFP4/FP8 dequantization path at all: a grep for `weight_scale|mxfp|e4m3|fp8|dequant` matches only the audio RVQ "quantizer" classes; the only `mtp` reference is a `_keys_to_ignore` pattern `model\.mtp\..*` — [modeling_mimo_v2.py](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-MOPD/resolve/main/modeling_mimo_v2.py). Official serving is SGLang (`--tp 16 --dp 2 --ep 16`, 2 nodes) or vLLM (`--tensor-parallel-size 8`) — [README](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-MOPD/resolve/main/README.md).

### Inferences
- `U8` weights with half-width shapes plus `U8` scale tensors of width `in/32` are the MXFP4 layout; the index metadata `save_format: mxfp4` and config `store_dtype: mxfp4` confirm it. There is no higher-precision expert copy in the repo.
- The `o_proj` matrices (13.13 GiB, 13.3% of the non-expert bytes) are the single largest still-unquantized block; `qkv_proj` is FP8 with 128×128 blocks (coarser than DeepSeek's 32×32).
- Loading the U8 experts in plain transformers is not supported by the shipped code; any Thor-side harness must consume MXFP4 natively (as llama.cpp, SGLang and vLLM do).

### Gaps
- `qkv_proj.weight_scale_inv` has shape `[216, 48]` while 27136/128 = 212; I did not resolve whether the stored qkv is padded or the scale tensor over-allocated. UNVERIFIED, immaterial to byte totals.
- The README says the "MTP" is 5 SWA layers (the DFlash sidecar) while the main checkpoint also carries a 3-layer DeepSeek-style MTP; which one SGLang's `--enable-multi-layer-eagle --speculative-num-steps 3` uses was not confirmed.

## Q4. MiMo-V2.6-Pro-MOPD: experts already MXFP4 (quantization exhausted), what "MOPD" is, byte shares

### Takeaway
Confirmed for the 1T Pro-MOPD: experts are stored MXFP4 and were trained that way (MXFP4 QAT in mid-training, quantize-dequantize after every RL update against the MXFP4 GEMM numerics), so the shipped 4-bit codes are the model; every sub-4-bit community quant shows large KLD/perplexity cost. MOPD is "Multi-Prefix Multi-Teacher On-Policy Distillation (MOPD2)", a post-RL distillation stage that does not change the architecture or weight format. Routed experts are 93.9% of main-index bytes; the dense/attention remainder that could still be quantized is 32.3 GiB, of which 24.0 GiB is attention (13.1 GiB BF16 `o_proj` + 10.9 GiB FP8 `qkv_proj`).

### Cited Findings
- "We employ MXFP4 quantization-aware training (QAT) during mid-training, allowing the model to adapt to low-precision computation while preserving model quality." — tech report §3 (mid-training) — [MiMo_V2_6_technical_report.pdf](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL/blob/main/MiMo_V2_6_technical_report.pdf).
- "In MiMo-V2.6, we use MXFP4 as the experts' data type during rollout ... we apply quantize–dequantize (QDQ) to the experts after each parameter update. The quantization follows the numeric constraints of the MXFP4 Humming GEMM kernels (vLLM Project, 2026) used during rollout, so the two engines see identical expert weights." — tech report §6.4.
- "To stabilize MXFP4 training, we initialize RL by ..." (RL optimizer section) — tech report §5.
- MOPD definition: "After mixed RL, we use Multi-Prefix Multi-Teacher On-Policy Distillation (MOPD2) to combine capabilities from teachers trained for different tasks, including tasks that are hard to verify. Building on MOPD from MiMo-V2-Flash ... we retain autonomous student rollouts in domains with suitable mixRL teachers (Standard MOPD) and ... SFT-Prefix OPD therefore starts each rollout from a fixed demonstration prefix" — tech report §5.6. README: "This is the MOPD upgrade of the MiMo-V2.6-Pro-RL checkpoint" that also fixes tool-call repetition — [README](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-MOPD/resolve/main/README.md).
- Community evidence that MXFP4 is the quality ceiling and anything lower is lossy (AesSedai, Pro-MOPD specifically; FFN tensors quantized, rest Q8_0/Q6_K): "MXFP4 537.99 GiB (4.52 BPW) PPL 3.171130 ± 0.015 KLD -0.000000 | BPW3.5 416.74 GiB PPL 3.254415 (+2.66%) KLD 0.1334 | BPW3.0 357.11 GiB PPL 3.479 (+9.76%) KLD 0.189 | BPW2.5 297.63 GiB PPL 3.790 (+19.56%) KLD 0.262 | BPW2.0 238.22 GiB PPL 4.554 (+43.67%) KLD 0.414"; "The MXFP4 quant is the 'full quality' version, as the model has MXFP4 experts." — [AesSedai/MiMo-V2.6-Pro-MOPD-GGUF](https://huggingface.co/AesSedai/MiMo-V2.6-Pro-MOPD-GGUF).
- ggml-org: "The MXFP4 output keeps the routed experts at their native MXFP4 precision. The Q2_K output keeps the expert down projections at MXFP4, and quantizes the gate/up projections to Q2_K." MXFP4 split ≈ 516 GiB, Q2_K ≈ 390 GiB (file sizes from listing) — [ggml-org/MiMo-V2.6-Pro-MOPD-GGUF](https://huggingface.co/ggml-org/MiMo-V2.6-Pro-MOPD-GGUF).
- Byte shares (computed, see Q3 table): routed experts 494.86 / 527.15 GiB = 93.9% of the main index (92.7% of all 534.07 GiB safetensors). Non-expert remainder 32.29 GiB: attention 24.00 (74%), embed+head 3.49, MTP 2.29, vision 1.38, audio 0.55, gates 0.30, dense layer-0 0.28.
- `o_proj` is explicitly excluded from FP8 by `quantization_config.ignored_layers` and is BF16 on disk (13.13 GiB) — [config.json](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-MOPD/resolve/main/config.json) + headers.

### Inferences
- Quantization headroom on the non-expert side: attention 18.72 B params at BF16/FP8 = 24.0 GiB → at 4-bit with per-32 scales (0.5625 B/param) ≈ 9.8 GiB (saves ~14 GiB); embed+head 1.87 B → 8-bit ≈ 1.75 GiB (saves 1.75 GiB). Realistic floor for "everything except routed experts, all quantized to 4-8 bit" ≈ 17-19 GiB. That headroom is 3-4% of the model; it does not change the picture that only pruning moves the 495 GiB expert block.
- Because the experts were QAT'd to MXFP4 and the RL loop QDQ'd them every step, a MXFP4 → NVFP4 transcode would be exactly lossless (same argument LibertAI makes for DeepSeek), but any 2-3 bpw scheme is a second, lossy quantization on top of a QAT'd representation — AesSedai's ladder quantifies it (+2.7% ppl at 3.5 bpw, +9.8% at 3.0 bpw).

### Gaps
- No MiMo source gives a per-task benchmark for sub-MXFP4 quants (only PPL/KLD on AesSedai's set). UNVERIFIED how +9.8% PPL maps to agentic-task accuracy.
- I did not find any statement that the FP8 attention/dense tensors were QAT'd; `activation_scheme: dynamic` suggests post-training FP8, so re-quantizing them is a first quantization (lower risk) — inference only.

## Q5. Prunable fraction and the byte floor if ALL routed experts were removed

### Takeaway
DeepSeek: routed experts are only 56.6% of bytes; removing all of them leaves a 206.3 GiB floor, 189.1 GiB of which is Engram (97.3 GiB if stored MXFP4); the non-Engram, non-drafter floor is 9.8 GiB. MiMo: routed experts are 93.9%; the floor is 32.3 GiB (39.2 GiB with the DFlash and audio-tokenizer sidecars), dropping to roughly 17-19 GiB if the remainder is quantized. Pruning therefore has far more leverage on MiMo; on DeepSeek the Engram placement decision dominates.

### Cited Findings (arithmetic from the header sums in Q1/Q3)
**DeepSeek-V4.1-Flash-1017** (total 475.24 GiB)
- Prunable routed experts: 268.95 GiB (56.6%). Each expert (w1+w2+w3 with scales) = 268.95 GiB / 15,360 = 17.93 MiB; each layer's 384 experts = 6.72 GiB.
- Floor with all routed experts removed: 475.24 − 268.95 = **206.29 GiB**, composed of Engram 189.13 + DSpark 7.39 + attention 4.80 + embed/head 2.47 + shared experts 1.32 + vision 0.90 + gates 0.15 + HC 0.15 + norms 0.001.
- Floor excluding Engram (host/NVMe-resident) and excluding the optional DSpark drafter: 206.29 − 189.13 − 7.39 = **9.77 GiB** (attention + embed/head + shared experts + vision + gates + HC). Excluding the vision tower too: 8.87 GiB.
- Floor with Engram stored MXFP4 but GPU-resident: 97.28 + 17.16 = 114.4 GiB before a single routed expert (Engram MXFP4 size from [LibertAI 1017-NVFP4 card](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-NVFP4)).
- Worked REAP points (bytes scale linearly with experts kept, Engram untouched): keep 256/384 → 475.24 − 268.95 × 128/384 = 385.6 GiB (LibertAI measured "475.2 GiB → 385.5 GiB" — [REAP-256E card](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-REAP-256E)); keep 192/384 → 340.8 GiB (151.7 GiB without Engram); keep 96/384 → 273.5 GiB (84.4 GiB without Engram).
- Measured REAP quality on 0909 (text PPL, 65,504 held-out tokens): "384 (unpruned) 3.6552 | 272 3.8060 (+4.13%) | 256 mean-saliency 3.8557 (+5.49%) | 240 3.9156 (+7.12%) | 192 4.1910 (+14.66%)"; vision degrades much faster ("Below 272 the vision path falls off a cliff"; caption PPL 24.3 → 31.0 at 256 with text-only ranking); vLLM's fused MoE kernel only accepts certain expert counts ("Unsupported expert number: 272 ... 256 is the largest legal value below 384") — [REAP-256E card](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-REAP-256E).

**MiMo-V2.6-Pro-MOPD** (main index 527.15 GiB; +6.90 GiB sidecars)
- Prunable routed experts: 494.86 GiB (93.9%). Each expert = 494.86 GiB / 26,496 = 19.12 MiB; each layer's 384 experts = 7.17 GiB.
- Floor with all routed experts removed: 527.15 − 494.86 = **32.29 GiB** (attention 24.00 + embed/head 3.49 + MTP 2.29 + vision 1.38 + audio 0.55 + gates 0.30 + dense-0 0.28). With DFlash (5.16) and audio tokenizer (1.74): 39.19 GiB. Text-only, no drafters: 32.29 − 2.29 − 1.38 − 0.55 = 28.07 GiB.
- Worked REAP points: keep 192/384 (50%) → 32.29 + 247.43 = 279.7 GiB; keep 96/384 (25%) → 156.0 GiB; keep 64/384 (16.7%) → 114.8 GiB; keep 48/384 (12.5%) → 94.1 GiB.
- Existing REAP on the sibling 310B Flash model (user's own): `patrickbdevaney/MiMo-V2.6-Flash-REAP50` 87.84 GiB safetensors — [HF listing](https://huggingface.co/api/models/patrickbdevaney/MiMo-V2.6-Flash-REAP50?blobs=true). No REAP/pruned variant of Pro-MOPD exists on HF as of 2026-10-17 (HF search "MiMo-V2.6-Pro" returns none tagged `reap`/`pruned`) — [HF search](https://huggingface.co/api/models?search=MiMo-V2.6-Pro).

### Inferences
- A 117 GiB envelope on MiMo-Pro implies keeping ≤ ~64 of 384 experts per layer (83% pruning) at native MXFP4 — well beyond the 34-50% REAP rates that have been shown to hold quality; combining 50% REAP with a lossy ~2.5 bpw expert requant (AesSedai's BPW2.5 is 297.6 GiB at 384 experts → ~150 GiB at 192) still does not fit.
- On DeepSeek-Flash, with Engram off-device and DSpark dropped, the 117 GiB envelope allows roughly (117 − 9.8) / 6.72 GiB per layer-set ≈ 153 of 384 experts per layer at native MXFP4 (≈60% pruning), or ~190-200 experts if the experts are additionally taken to ~3 bpw (EXL3 kit: 196 GiB at 384 experts, i.e. ~0.49 GiB/expert-set-per-layer-equivalent).

### Gaps
- All pruning-quality numbers are for the 0909 DeepSeek build and for text perplexity; none exist for 1017 or for MiMo-Pro. UNVERIFIED that MiMo-Pro's 384-expert pool prunes as gracefully as the Flash model.

## Q6. KV-cache bytes per token (native, 32k, 128k, 1M)

### Takeaway
DeepSeek-V4.1-Flash: 890 B/token of growing global KV (FP4 main latent 720 B + FP4 indexer keys 170 B; derived from the config and matches the paper's "890 bytes per token" exactly) plus a fixed ≈2.7 MiB of 128-token SWA windows; 1M tokens ≈ 890 MiB. MiMo-V2.6-Pro: 51,200 B/token in BF16 (10 global-attention GQA layers × 8 KV heads × (192+128) × 2 B) plus a fixed ≈37.5 MiB of SWA windows; 1M tokens = 50.0 GiB (25 GiB with FP8 KV). MiMo's per-token KV is ~57× DeepSeek's.

### Cited Findings
**DeepSeek derivation** (config + reference code)
- Only `kv_source_layer_ids [2, 8, 14, 20]` write a compressed KV cache (`compress_kv_cache` sized `max_seq_len // compress_ratio × head_dim 512`); layers 2, 8, 14 have `compress_ratio 2`, layer 20 has `compress_ratio 1`; all other compress-mode layers "just read the cache" — [inference/model.py](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/resolve/main/inference/model.py), [config.json](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/resolve/main/config.json).
- Main latent per token = (½ + ½ + ½ + 1) latents × 512 ch = 1,280 ch; stored FP4 E2M1 with one E4M3 scale per 16 ch = 0.5625 B/ch → **720 B/token**.
- Indexer K cache: only the 4 kv-source layers own an `indexer.wk` (headers show exactly 4 `layers.N.attn.indexer.wk.weight` tensors vs 8 `indexer.wq_b`), 128 ch each, FP4 with E8M0 per 32 ("the indexer uses 32 with E8M0") = 0.53125 B/ch; 2.5 latents × 128 = 320 ch → **170 B/token**.
- Sum = **890 B/token**, equal to the paper's "global KV cache footprint (always in HBM) to 890 bytes per token, about 1/4 of DeepSeek-V4-Flash" — [arXiv HTML](https://arxiv.org/html/2609.19969v1); vLLM: "A 1M-token prompt holds under 1 GB of global KV, plus a fixed 128-token window per layer" — [vLLM recipe](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash).
- Fixed SWA ring buffers: `window_kv_cache [max_batch, 128, 512]` per layer; paper says "SWA KV stays FP8" → 128 × 512 × 1 B = 64 KiB per layer × 43 layers (40 + 3 DSpark) = 2.69 MiB per sequence (computed).
- Context totals (growing part only): 32,768 × 890 = 29.2 MB = 27.8 MiB; 131,072 × 890 = 116.7 MB = 111.3 MiB; 1,048,576 × 890 = 933.2 MB = 890.0 MiB (native context is 1,048,576).
- Deployment option: "SWA Bounded Replay ... reduces its persistent KV cache footprint (always on SSD or in host memory) to roughly 1/8 of that of DeepSeek-V4-Flash" — [arXiv HTML](https://arxiv.org/html/2609.19969v1).

**MiMo derivation** (config)
- Global-attention layers: 10 (zeros in `hybrid_layer_pattern`); `num_key_value_heads 8`, `head_dim 192` (K), `v_head_dim 128` (V) → 8 × (192 + 128) = 2,560 values × 2 B (BF16) = 5,120 B per layer → **51,200 B/token = 50.0 KiB/token** (computed from [config.json](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-MOPD/resolve/main/config.json)). The `qkv_proj [27136, 6144]` shape confirms 128×192 + 8×192 + 8×128 = 27,136.
- SWA layers: 60 × 128 tokens × 5,120 B = 39.3 MB = 37.5 MiB fixed per sequence (computed; SGLang serves with `--page-size 64 --swa-full-tokens-ratio 0.3`, so the practical reservation differs — [README](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-MOPD/resolve/main/README.md)).
- Context totals (growing part): 32,768 × 51,200 = 1.678 GB = **1.5625 GiB**; 131,072 × 51,200 = 6.71 GB = **6.25 GiB**; 1,048,576 × 51,200 = 53.69 GB = **50.0 GiB** (native 1M). FP8 KV halves these (diffbot ships an "FP8KV" Flash variant: [diffbot/MiMo-V2.6-Flash-MOPD-FP8KV-W4A8](https://huggingface.co/diffbot/MiMo-V2.6-Flash-MOPD-FP8KV-W4A8-2x-RTX-PRO-6000)).
- Tech report confirms the structure that makes this cheap under context parallelism: "MiMo-V2.6 interleaves 128-token sliding-window layers with full attention ... the sliding-window layers exchange only the KV their queries can reach—at most a window-sized segment" — tech report §6.4.

### Inferences
- The DeepSeek figure is per *sequence* regardless of batch; the Engram hash cache (`int64` per position) adds 8 B/token in the reference code — negligible.
- For a Thor-class 117 GiB envelope, MiMo at 128k context needs 6.25 GiB (BF16) or 3.1 GiB (FP8) of KV, i.e. 3-5% of memory; DeepSeek's KV is irrelevant to the budget.

### Gaps
- The exact per-token cost of DeepSeek's DSpark drafter cache (window only, `compress_ratio 0`) and of the `candidate_topk_blocks` path were not derived; both are fixed-size.
- MiMo's drafter KV (DFlash: 5 layers, window 1024; MTP: 3 layers) not included.

## Q7. Known community pruned/quantized variants on HF (as of 2026-10-17)

### Takeaway
DeepSeek-V4.1-Flash has a large derivative ecosystem (NVFP4, EXL3 2.0-3.5 bpw, REAP 256E/272E, REAP25/50 MLX, GGUF Q2-Q8, MXFP4/INT4 Engram re-packs), and 1017-specific builds appeared within a day of release (NVFP4 399.5 GiB; EXL3 2.9 bpw 196.3 GiB without Engram; GGUF Q2_K 246 GiB to Q8_0 473 GiB; MLX oQ2e-oQ4e). MiMo-V2.6-Pro-MOPD has native-MXFP4 GGUF splits (ggml-org, AesSedai, kernelpool), lossy 2.0-3.5 bpw GGUFs with published PPL/KLD, EXL3 2.53/3.0/4.04 bpw, and an ARVQ-hybrid — but no pruned variant.

### Cited Findings
**DeepSeek-V4.1-Flash-1017 derivatives** (sizes from HF file listings; all repos listed by [HF search](https://huggingface.co/api/models?search=DeepSeek-V4.1-Flash-1017))
- `LibertAIDAI/DeepSeek-V4.1-Flash-1017-NVFP4` — 399.51 GiB; routed experts bit-exact MXFP4→NVFP4 transcode, Engram MXFP4 (cosine 0.9934), rest byte-identical; `input_scale`s borrowed from NVIDIA's calibration — [card](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-NVFP4).
- `LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw` — 196.3 GiB (Engram not included; must be read from base shards 47/48); PPL 1.8749→1.9094 code, 6.4277→6.4520 prose — [card](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw). Also `PixelML/...-EXL3-3bpw`, `LakyFx/...-EXL3-3.5bpw-TensorFold` (no quality data fetched).
- `mrfakename/DeepSeek-V4.1-Flash-1017-GGUF` — Q2_K 246.35 GiB, Q3_K_M 323.42 GiB, Q4_K_M 414.19 GiB, Q8_0 473.07 GiB (whether Engram is inside these GGUFs: UNVERIFIED) — [listing](https://huggingface.co/api/models/mrfakename/DeepSeek-V4.1-Flash-1017-GGUF?blobs=true).
- `mlx-community/DeepSeek-V4.1-Flash-1017-oQ2e-mtp` 291.75 GiB; also oQ2.5e / oQ3e / oQ4e-mtp variants; `pipenetwork/...-1017-MLX-mixed-4_8bit`; `gnahzhang/...-1017-LSQ3-G128-MLX`; `Baekpica/...-1017-VQ-GGUF`; `pqhaz/...-1017-heretic(-NVFP4/-GGUF)` — [HF search](https://huggingface.co/api/models?search=DeepSeek-V4.1-Flash-1017).
- No REAP/pruned variant of the **1017** build exists yet (search returns none tagged `reap`).

**DeepSeek-V4.1-Flash (0909) derivatives relevant as recipes**
- `nvidia/DeepSeek-V4.1-Flash-NVFP4` — 399.52 GiB (after 10/14 MXFP4-Engram update; was ~492 GiB with FP8 Engram); W4A4 experts g16; accuracy table on GPQA-Diamond/AA-LCR/SciCode/IFBench/MMMU-Pro/Terminal-Bench 2.1 is in the card (rows "MXFP4 (source)" vs "NVFP4 + MXFP4 Engram") — [card](https://huggingface.co/nvidia/DeepSeek-V4.1-Flash-NVFP4).
- `LibertAIDAI/DeepSeek-V4.1-Flash-REAP-256E` — 196.42 GiB listed (card says 385.5 GiB total with Engram; the HF listing appears to exclude the Engram shards), text PPL +5.5% (mean ranking) / +14.1% (balanced ranking, better vision); `-REAP-272E` 207.64 GiB, +4.13% text PPL — [card](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-REAP-256E).
- `pipenetwork/DeepSeek-V4.1-Flash-REAP50-MLX-mixed-4_8bit` 255.79 GiB and `-REAP25-` (MLX, no quality data fetched); `rapid-mlx/DeepSeek-V4.1-Flash-REAP-2bit-MLX` 198.31 GiB — [HF search](https://huggingface.co/api/models?search=DeepSeek-V4.1-Flash).
- `antirez/deepseek-v4.1-flash-gguf` (most-downloaded GGUF, 1.63M downloads): Q2 340.60 GiB, Q4 483 GiB (2 parts), separate 0.90 GiB vision GGUF — [listing](https://huggingface.co/api/models/antirez/deepseek-v4.1-flash-gguf?blobs=true).
- Engram re-packs: `INCModel3/...-W4A16-Engram-AutoRound` (451.7 GB, gsm8k 93.93 vs 92.87), `INCModel3/...-MXFP4-Engram-AutoRound`, `pipenetwork/...-engram6`, `aidendle94/...-NVFP4-Engram`, `satgeze/...-EXL3-2.0bpw-Ablit-EngramQ4-SM120` — [INCModel3 card](https://huggingface.co/INCModel3/DeepSeek-V4.1-Flash-W4A16-Engram-AutoRound); [HF search](https://huggingface.co/api/models?search=DeepSeek-V4.1-Flash).

**MiMo-V2.6-Pro-MOPD derivatives** ([HF search](https://huggingface.co/api/models?search=MiMo-V2.6-Pro))
- `ggml-org/MiMo-V2.6-Pro-MOPD-GGUF` — MXFP4 split ≈ 516 GiB (native experts), Q2_K split ≈ 390 GiB (down_proj MXFP4, gate/up Q2_K), plus MTP sidecars (BF16 7.10 / Q8_0 3.78 / Q4_0 2.00 GiB), DFlash sidecars (BF16 8.65 / Q8_0 4.60 GiB), mmproj (BF16 2.64 / Q8_0 1.50 GiB) — [card](https://huggingface.co/ggml-org/MiMo-V2.6-Pro-MOPD-GGUF).
- `AesSedai/MiMo-V2.6-Pro-MOPD-GGUF` — MXFP4 538.00 GiB (KLD 0), BPW3.5 416.74 GiB (+2.66% PPL, KLD 0.133), BPW3.0 357.12 GiB (+9.76%, KLD 0.189), BPW2.5 297.64 GiB (+19.56%, KLD 0.262), BPW2.0 238.22 GiB (+43.67%, KLD 0.414); imatrix provided — [card](https://huggingface.co/AesSedai/MiMo-V2.6-Pro-MOPD-GGUF).
- `kernelpool/MiMo-V2.6-Pro-MOPD-MXFP4-GGUF` ≈ 519 GiB; `pcuenq/MiMo-V2.6-Pro-MOPD-GGUF` — [listing](https://huggingface.co/api/models/kernelpool/MiMo-V2.6-Pro-MOPD-MXFP4-GGUF?blobs=true).
- EXL3: `MikeRoz/MiMo-V2.6-Pro-MOPD-2.53bpw-h6-exl3` 305.38 GiB and `-4.04bpw-h8-exl3` 484.71 GiB ("I'm going to try and compute and add KLD results" — none yet); `doth4580/MiMo-V2.6-Pro-MOPD-EXL3-3.0bpw-hq` 365.82 GiB — [MikeRoz card](https://huggingface.co/MikeRoz/MiMo-V2.6-Pro-MOPD-2.53bpw-h6-exl3).
- `jarrelscy/MiMo-V2.6-Pro-RL-ARVQ-hybrid` (RL base) 300.55 GiB, tag `nvfp4_arvq_hybrid`; `drowzeys/keys-MiMo-V2.6-Pro-MOPD-Jarrelscy-ARVQ-hybrid21` — [listing](https://huggingface.co/api/models/jarrelscy/MiMo-V2.6-Pro-RL-ARVQ-hybrid?blobs=true).
- `mlx-community/MiMo-V2.6-Pro-RL-mxfp4-q8` 515.84 GiB (RL base, native experts + 8-bit rest).
- Pruned: none for Pro. For Flash (310B sibling): `patrickbdevaney/MiMo-V2.6-Flash-REAP50` (87.84 GiB) and `-REAP50-GGUF`, `tacodevs/MiMo-V2.6-Flash-RL-MLX-REAP25/37/50-*`, `webmp3/Sakura-MiMo-V2.6-Flash-MOPD-P160-GGUF` (expert-pruning tag) — [HF search](https://huggingface.co/api/models?search=MiMo-V2.6).

### Inferences
- The size ladder for Pro-MOPD is essentially linear in expert bits: ≈ 32 GiB fixed + 384-expert block at (bpw/4) × 495 GiB; the published KLD curve means ~3.5 bpw is the lowest "cheap" point and 2.0-2.5 bpw is a different model.
- For DeepSeek-1017 the community has already demonstrated every lever the engineer needs separately (lossless NVFP4 transcode, lossy MXFP4 Engram, REAP to 256, EXL3 ~3 bpw) but not combined into one checkpoint.

### Gaps
- Download counts for all 1017 derivatives read 0 (repo created within 24 h), so no popularity signal yet.
- No task-level benchmarks exist for any MiMo-Pro quant or any DeepSeek REAP build; only PPL/KLD.
- Whether the 1017 GGUFs (mrfakename, Baekpica) embed the Engram tables or require the sidecar was not checked.
