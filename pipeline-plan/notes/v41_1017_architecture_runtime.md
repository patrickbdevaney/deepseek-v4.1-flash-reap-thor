# DeepSeek-V4.1-Flash-1017: verified architecture and runtime-support status for a REAP + Engram-on-NVMe pipeline (as of 2026-10-19)

Method: every structural fact below was read on 2026-10-19 from the live HF repo at SHA `1bae0e1088993b9b82de08f59de5be5a34e42170` (lastModified 2026-10-17T05:46:55Z, the same SHA as the 2026-10-17 notes). Sources: `config.json`, `inference/config.json`, `inference/model.py`, `inference/engram.py`, `inference/convert.py`, `model.safetensors.index.json`, and the safetensors headers of **all 48 shards**, fetched with HTTP Range requests. Byte sums come from header `data_offsets` and match the index `metadata.total_size` = 510,286,023,000 B exactly. GiB = 2^30 B. "LOCAL" means I inspected it directly on this Thor, in the `vllm/vllm-openai:nightly-aarch64` image (vLLM `0.32.0rc1.dev190+g3f1bcfd82`). I read the image's files only: no GPU work, no server. A Gemma-E2B vLLM server was running on the box at the time. "(inference)" marks my own reasoning.

**Corrections and additions to the 2026-10-17 notes** (`v41_architecture_pruning_delta.md`, `engram_off_device_thor.md`, `target_models.md`):
1. **All 48 weight shards changed between 0909 and 1017, the two Engram shards included.** `config.json`, `inference/*`, `model.safetensors.index.json`, `tokenizer*` and `LICENSE` are byte-identical. So any Engram table, REAP keep-set or saliency file built from 0909 is invalid for 1017. That includes LibertAI's REAP-256E, which points users at the 0909 shards 47/48.
2. **Attention kernels are the Thor blocker, not the router.** In the local image, `_flashmla_C` and `_flashmla_extension_C` contain only `sm_90a` and `sm_100` cubins, and `_deepselect_C` only `sm_100`. None of the three embeds PTX. On compute capability 11 the DSv4.1 backend selector falls through to FlashMLA, so this image cannot run V4.1 attention on Thor as shipped. The 10-17 note left this "unverified".
3. The fused `sqrtsoftplus` router kernel set {1,2,4,8,16,32,64,128,192,256,320,384,448,512,576} is re-verified on vLLM `main`. The kernel guards on `__CUDA_ARCH__ >= 900` only, and `_moe_C_stable_libtorch` in the local image does carry `sm_110` cubins.
4. `compress_ratios` has 43 entries (40 backbone + 3 DSpark). The values are 0/1/2, not the V4 4/128.
5. The `mtp.*` tensors are the DSpark drafter: 3 blocks with **128** routed experts each, top-3. The vLLM recipe says "MTP was dropped in V4.1. DSpark is the only speculative method".

---

## Q1. Config and structure: layers, experts, router, attention, MTP/DSpark, vision, context, quantization

### Takeaway
V4.1-Flash-1017 has 40 layers, hidden size 5120, and is MoE in every layer: 384 routed experts (SwiGLU, intermediate 2304) plus 1 shared, top-6. The router uses `sqrtsoftplus` scores with a `noaux_tc` bias that selects experts only, plus a **second per-expert bias `bias_vl` for image-span tokens**. Selected weights are renormalised and multiplied by 1.5. Attention is a 512-channel single-latent MLA-style attention over a 128-token sliding window, plus top-512 compressed-sparse KV shared across layer groups. The residual uses 4-stream mHC. Engram sits at layers 1 and 14, a 3-block DSpark drafter (128 experts, top-3) is appended, a 32-layer ViT handles vision, and context is 1M (YaRN ×16 from 64K). Routed experts ship as MXFP4 (E2M1 + E8M0 per 32). Dense linears are FP8 E4M3 with 32×32 E8M0 block scales.

### Cited Findings
- **Top level** — [config.json](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/resolve/main/config.json):
  - `architectures ["DeepseekV41ForCausalLM"]`, `model_type deepseek_v41`, `transformers_version 5.6.0`, `image_token_id 129264`, bos/eos/pad = 0/1/2.
  - `quantization_config`: `quant_method fp8`, `activation_scheme dynamic`, `weight_block_size [32,32]`, `scale_fmt ue8m0`, `expert_dtype fp4`.
- **text_config: backbone and router** — [config.json](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/resolve/main/config.json):
  - Shape: `vocab_size 129280`, `hidden_size 5120`, `moe_intermediate_size 2304`, `num_hidden_layers 40`, `num_attention_heads 64`, `num_key_value_heads 1`, `head_dim 512`, `qk_rope_head_dim 64`, `q_lora_rank 1280`, `o_lora_rank 1024`, `o_groups 8`, `swiglu_limit 10.0`, `rms_norm_eps 1e-20`, `tie_word_embeddings false`.
  - Context: `max_position_embeddings 1048576`, `rope_theta 10000`, `rope_scaling {yarn, factor 16, beta_fast 32, beta_slow 1, original_max_position_embeddings 65536}`.
  - Routing: `n_routed_experts 384`, `n_shared_experts 1`, `num_experts_per_tok 6`, `scoring_func sqrtsoftplus`, `topk_method noaux_tc`, `norm_topk_prob true`, `routed_scaling_factor 1.5`.
  - Attention: `sliding_window 128`.
- **text_config: compression, mHC, Engram, DSpark** — [config.json](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/resolve/main/config.json):
  - `compress_ratios`: layers 0–1 → 0; layers 2–19 → 2; layers 20–39 → 1; three trailing DSpark entries → 0 (43 entries in all).
  - `compress_rope_theta 160000`, `kv_source_layer_ids [2,8,14,20]`, `index_source_layer_ids [2,8,14,20,24,28,32,36]`, `index_n_heads 32`, `index_head_dim 128`, `index_topk 512`, `candidate_source_layer_id 20`, `candidate_topk_blocks 2048`, `candidate_block_size 8`.
  - mHC: `hc_mult 4`, `hc_sinkhorn_iters 20`, `hc_eps 1e-6`.
  - Engram: `engram_layer_ids [1,14]`, `engram_num_embeddings [384006168, 384016682]`, `engram_max_ngram_size 4`, `engram_vocab_size 16000000`, `engram_n_heads 8`, `engram_head_dim 256`, `engram_pad_token_id 2`, `engram_compressed_vocab_size 99092`.
  - DSpark: `num_nextn_predict_layers 3`, `dspark_block_size 5`, `dspark_noise_token_id 128799`, `dspark_target_layer_ids [37,38,39]`, `dspark_markov_rank 256`, `dspark_n_routed_experts 128`, `dspark_num_experts_per_tok 3`.
- **vision_config**: 32 layers, hidden 1024, 16 heads, intermediate 2816, patch 14, `downsample_ratio 3`, `max_image_tokens 1024`, `min_pixels 295936` — [config.json](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/resolve/main/config.json)
- **Router (reference `Gate`)**:
  - "The correction bias steers expert selection only; the routing weights come from the unbiased scores. Image-span tokens use a separate bias (training `noaux_tc_for_vl`)."
  - Code path: `scores = F.softplus(linear(x,W)/gate_temp).sqrt()`; `bias = where(image_mask, bias_vl, bias)`; `indices = (scores+bias).topk(6)`; `weights = scores.gather(indices)`; `weights /= sum + 1e-20`; `weights *= route_scale`.
  - There is no group routing (`n_group`/`topk_group` are absent) and no hash routing.
  - [inference/model.py](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/resolve/main/inference/model.py)
- **Router tensors** (headers): `layers.N.ffn.gate.weight` BF16 [384,5120], `gate.bias` F32 [384], `gate.bias_vl` F32 [384] for all 40 layers. The `mtp.N.ffn.gate.*` tensors are [128,…] — [shard headers](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/main)
- **Experts**:
  - "Top-k routed experts plus one shared expert every token goes through". `assert args.n_shared_experts == 1`.
  - SwiGLU with clamps: "the up branch is clamped on both sides, the gate branch only from above" (limit 10).
  - Routed experts load as `torch.float4_e2m1fn_x2`. The shared expert is FP8.
  - [inference/model.py](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/resolve/main/inference/model.py)
- **Expert tensor shapes** (headers):
  - `w1`/`w3`: `I8 [2304, 2560]` + `F8_E8M0 [2304, 160]`. `w2`: `I8 [5120, 1152]` + `F8_E8M0 [5120, 72]`. That is packed E2M1 at two values per byte with one UE8M0 scale per 32 along K. This holds for all 40 layers × 384 experts and for `mtp.{0,1,2}` × 128 experts.
  - Shared experts are `F8_E4M3 [2304,5120]` with `F8_E8M0 [72,160]` (32×32 blocks).
  - [shard headers](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/main)
- **Attention (reference)**:
  - "Latent attention over two KV sources at once … a sliding window of raw KV, plus -- when compress_ratio > 0 -- `index_topk` compressed positions reaching further back. Q and the output projection are both low-rank, the latter grouped."
  - "compress_ratio > 0 does not mean the layer compresses its own KV: only kv_source_layers do, the rest read that same cache."
  - Compressor: "ratio 1 is a plain projection, so it stays in the checkpoint's bf16". A ratio >1 uses a softmax-gated pool (`wgate`) in fp32.
  - "Compressed KV uses groups of 16 with E4M3 scales; the indexer uses 32 with E8M0."
  - The indexer is "level two" of a two-level top-k. Layer 20 is the candidate source: `select_candidate_blocks` keeps 2048 blocks of 8.
  - Each layer has an `attn_sink` (F32 [64]).
  - [inference/model.py](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/resolve/main/inference/model.py)
- **Per-layer attention roles (computed from index names)**:
  - `compressor.wkv`/`norm` sit in layers [2,8,14,20]. `compressor.wgate` sits in [2,8,14] only, because layer 20 has ratio 1.
  - `indexer.wk`/`k_norm` sit in [2,8,14,20]. `indexer.wq_b`/`weights_proj` sit in [2,8,14,20,24,28,32,36].
  - Every one of the 40 layers has `wq_a` FP8 [1280,5120], `wq_b` FP8 [32768,1280], `wkv` FP8 [512,5120], `wo_a` FP8 [8192,4096] and `wo_b` FP8 [5120,8192].
  - [index.json](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/resolve/main/model.safetensors.index.json)
- **mHC (reference `Block`)**:
  - "The coefficients a sublayer computes are used by the *next* one". The attention input uses the previous block's `ffn_pre`.
  - `mix_hc = (2+hc_mult)*hc_mult` = 24.
  - Tensors `hc_{attn,ffn}_fn` F32 [24, 20480], `hc_*_base` [24], `hc_*_scale` [3].
  - [inference/model.py](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/resolve/main/inference/model.py)
- **DSpark drafter**:
  - `Transformer.__init__` appends `DSparkBlock(args.n_layers + layer_id)` × `n_mtp_layers` (3), sharing `embed` and `head`.
  - "the MTP head reads the attention input of its target layers, not their output": `main_hiddens.append(h.mean(dim=2))` at layers 37–39.
  - Extra tensors: `mtp.N.main_proj` FP8 [5120,15360], `markov_head.embed/head` BF16 [129280,256] and `confidence_head.proj` BF16 [1,5376].
  - The file notes that "nothing calls forward_spec … the speculative-decoding loop itself is out of scope for this repo".
  - [inference/model.py](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/resolve/main/inference/model.py)
- vLLM's recipe states "MTP was dropped in V4.1. DSpark is the only speculative method", with a 5-token block and a drafter of about 14B params — [vLLM recipe](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash)
- **Vision**: `vision.blocks.N.*` BF16 ×32 blocks, `aligner.w1` [5120, 9216] (9216 = 1024·3·3) and `aligner.w2` [5120,5120], plus learned `image_start/end/newline` [5120]. All of it is in shard 1. Image tokens route through `bias_vl` and get **no Engram contribution** ("image tokens take no part in an n-gram and get no engram contribution") — [index.json](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/resolve/main/model.safetensors.index.json); [inference/model.py](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/resolve/main/inference/model.py)
- **Reference runtime**: `requirements.txt` pins `torch>=2.10.0` and `tilelang==0.1.12`. `kernel.py` turns off `TL_DISABLE_WARP_SPECIALIZED` and `TL_DISABLE_TMA_LOWER` (both set True) and includes a TileLang `fp4_gemm_kernel` ("C[M,N] = A_fp8[M,K] @ B_fp4[N,K]^T") — [inference/kernel.py](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/resolve/main/inference/kernel.py)

### Inferences
- REAP surgery on each of the 40 backbone layers has to slice **three** router tensors: `gate.weight` rows, `gate.bias` and `gate.bias_vl`. It then renumbers `ffn.experts.{0..N-1}` contiguously. `convert.py`'s `infer_num_experts` takes `max(idx)+1`, and vLLM reads `n_routed_experts`, so `config.json` `text_config.n_routed_experts` must be edited too. The DSpark `mtp.*` experts are a separate 128-expert pool, untouched by a backbone keep-set.
- The routing weight a REAP hook should record is `scores.gather(indices)` after renormalisation × 1.5, which lies in [0, 1.5]. Selection uses `scores + bias` (or `bias_vl`), so text-only calibration never exercises `bias_vl`. If vision matters, the calibration set needs image spans so `bias_vl`-selected experts get saliency.
- TMA and warp specialisation are switched off in the reference TileLang kernels, so the reference `inference/` stack is the most plausibly portable path to sm_110 for a calibration forward. This is unverified on Thor: TileLang 0.1.12 sm_110 support was not checked.

### Gaps
- `gate_temp` is not in `config.json`, so the reference default of 1.0 applies. I did not verify whether vLLM reads any other temperature.
- I did not fetch `hc_split_sinkhorn` in `kernel.py` line by line.

---

## Q2. Byte breakdown by tensor class and total repo size

### Takeaway
The repo is 510.31 GB (`siblings` sum 510,311,045,295 B). The weights are 510,286,023,000 B = **475.24 GiB**. Routed experts are 268.95 GiB (56.6%) and Engram tables 188.83 GiB (39.7%). Everything else is about 17.5 GiB: DSpark 7.39, attention 4.80, shared experts 1.32, embed and head 1.23 each, vision 0.90, Engram projections 0.29, norms/mHC 0.15, router 0.15. One routed expert is exactly 18,800,640 B (17.93 MiB), so each pruned expert per layer saves 17.93 MiB and a full 384-wide layer is 6.724 GiB.

### Cited Findings
Computed from all 48 shard headers (sum matches `metadata.total_size` 510,286,023,000) — [index.json](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/resolve/main/model.safetensors.index.json), [HF API listing](https://huggingface.co/api/models/deepseek-ai/DeepSeek-V4.1-Flash-1017?blobs=true):

| Class | GiB | % | tensors | dtype |
|---|---|---|---|---|
| Routed experts (40 × 384) | 268.95 | 56.6 | 92,160 | I8 (packed E2M1) + E8M0/32 |
| Engram tables `layers.{1,14}.engram.embed.{weight,scale}` | 188.83 | 39.7 | 4 | F8_E4M3 [~384M,256] + F8_E8M0 [~384M,8] |
| DSpark `mtp.*` (3 blocks, incl. 3 × 128 experts = 6.72 GiB) | 7.39 | 1.6 | 2,401 | mixed |
| Attention (incl. compressor/indexer) | 4.80 | 1.0 | 563 | FP8 + BF16 |
| Shared experts | 1.32 | 0.3 | 240 | FP8 32×32 |
| `embed.weight` / `head.weight` | 1.23 / 1.23 | 0.3 / 0.3 | 1 / 1 | BF16 [129280,5120] |
| Vision + aligner + image_* | 0.90 | 0.2 | 266 | BF16 |
| Engram `wkv` / `q_weight` / `k_weight` | 0.29 | 0.1 | 8 | FP8 [25600,6144]; BF16 [4,5120] |
| Norms + mHC | 0.15 | 0.0 | 321 | F32/BF16 |
| Router (`gate.weight/bias/bias_vl`) | 0.15 | 0.0 | 120 | BF16/F32 |

- By dtype: I8 259.45 GiB, F8_E4M3 190.00, F8_E8M0 21.94, BF16 3.68, F32 0.16 (computed).
- **Shard layout**:
  - Engram is only in `model-00047` (layer 1: 101,535,150,936 B) and `model-00048` (layer 14: 101,537,926,640 B).
  - Within each Engram shard, `embed.weight` is one contiguous block at offset 0. `embed.scale` follows as a separate contiguous block (offset 98,305,579,008 in shard 47), and then `q_weight`, `k_weight`, `wkv.weight` and `wkv.scale`.
  - DSpark is in shards 44–46. Vision is in shard 1.
  - (computed from headers and index)

### Inferences
- Expert totals at the legal vLLM counts (× 40 layers × 17.93 MiB): 320 → 224.13 GiB; 256 → 179.30; 192 → 134.48; 128 → 89.65 GiB. The non-expert, non-Engram, non-DSpark backbone is about 10.1 GiB.
- A raw-safetensors Engram row read touches **two** disjoint regions: 256 B of weight and 8 B of scale. With O_DIRECT that is 2 × 4 KiB I/Os per row unless the table is repacked. MiaAI repacks weight and scale adjacently for exactly this reason (see Q3).

### Gaps
- None for byte accounting.

---

## Q3. Engram: structure, hashing, addressing, per-layer, offload flags, community NVMe row stores, prune/quantize

### Takeaway
There are two independent Engram modules, one before layer 1 and one before layer 14, each with its own FP8 table: 384,006,168 and 384,016,682 rows × 256 B, plus 8 B of E8M0 scales per row. The reference hashes each position's 2-, 3- and 4-gram, over a 99,092-entry case- and accent-folded "compressed" vocab, into 8 heads each. That gives 24 prime-modulus buckets of about 16M rows, so **24 rows per token per Engram layer** (48 per token). The hash ids depend **only on token ids**, so they can be computed before the forward pass, and for layer 14 they are ready 13 layers ahead. The output is added to all four hc streams through a per-stream sigmoid gate against a key that `wkv` projects from the 24 concatenated rows.

Official offload is host memory only:
- vLLM: `--engram-config '{"cpu_offload":true}'`, with UVA, async prefetch and THP.
- SGLang: `SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE`.

Neither has an NVMe mode. On Thor, host memory is the same unified pool, so NVMe needs a community-style row store. Such stores exist: MiaAI's `row_store.cpp` (O_DIRECT, 4 KiB per row, 96 threads, SGLang), tonyd2wild's vLLM Engram-on-disk patch, and springrain, 0xBakeer and helgard-orlm variants. The table cannot be row-pruned without rehashing. It can be quantized: MXFP4 is lossy at about 12% L2 / cos 0.9934, and 6-bit is reported as indistinguishable.

### Cited Findings
- **Module (reference)**:
  - "Writes an n-gram lookup into the residual stream, gated by how well it matches that stream. The hash ids fetch `n_hash_cols` rows; `wkv` turns them into one key per hc copy plus a shared value."
  - `n_hash_cols = (max_ngram_size - 1) * n_heads` = 24. `wkv: Linear(24*256=6144 → dim*(hc_mult+1) = 25600)`.
  - Gate: `sigmoid(copysign(sqrt(clamp_min(|dot|,1e-6)), dot))`, "normalized per (token, hc copy) over `dim`, NOT jointly over the copies". Output `h + gate * value`.
  - Table: "The n-gram hash table, sharded over its rows. Stays fp8: rows are dequantized on lookup" (block 32).
  - [inference/model.py](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/resolve/main/inference/model.py)
- **Placement**: in `Transformer.forward`, `if layer.engram is not None: h = layer.engram(h, engram_hashes[:, :, layer.engram.layer_hash_index, :], engram_mask)` runs **before** `h, pre_mix = layer(h, …)`, on the 4-copy stream. The hashes are computed once per forward, before the embedding: `engram_hashes = self.engram_hash(input_ids, start_pos, engram_mask)` — [inference/model.py](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/resolve/main/inference/model.py)
- **Hashing (`engram.py`)**:
  - Compressed vocab: NFKC → NFD → StripAccents → Lowercase → whitespace collapse, so that `" The", "the" and "THE" all hash the same way`.
  - The assert `vocab_size == args.engram_compressed_vocab_size` guards against a mismatch, because "every hash multiplier derives from the compressed vocab size, so a mismatch there would silently rehash the whole table".
  - Multipliers: per-layer `np.random.default_rng(10007 * layer_id)` odd int64.
  - Rolling XOR of `token × multiplier` over look-backs, then `% prime[layer][ngram][head] + offset`.
  - Primes are "the smallest prime above `start` that has not been handed out yet", starting from `engram_vocab_size - 1`, never reused across layers.
  - Look-back stops at sequence start and at image spans (`DEAD`), padding with `engram_pad_id` 2.
  - The hash state keeps a `[max_batch, max_seq_len]` int64 cache of compressed ids across prefill and decode.
  - [inference/engram.py](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/resolve/main/inference/engram.py)
- **Verified LOCAL**: re-running the prime-allocation loop (2 layers × 3 n-gram orders × 8 heads, next unused prime above 15,999,999) reproduces `engram_num_embeddings` = [384006168, 384016682] exactly. The tables are therefore **per-layer and disjoint**: the two layers' prime sets differ, so each layer has its own buckets (computed with sympy).
- **TP sharding**: `convert.py` shards `.engram.embed.` by rows, with `shard_size = ceil(rows/mp)`. It pads the tail with 0 for the weight and 1 for the scale. Each rank all-reduces its masked partial lookup — [inference/convert.py](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/resolve/main/inference/convert.py)
- **Paper**:
  - "196B Engram parameters, split evenly across two modules at layers 1 and 14 … N-gram orders {2,3,4}, 8 hash heads … lookups are deterministic, so embeddings can be prefetched from host memory via background RDMA transfers."
  - Versus the Jan-2026 Engram paper, the short causal conv is dropped.
  - [arXiv 2609.19969](https://arxiv.org/pdf/2609.19969) (as quoted in the 10-17 notes; not re-fetched today)
- **vLLM**:
  - Recipe: "`--engram-config '{"cpu_offload":true}'` moves the Engram tables to pinned host memory, read through UVA". Other keys: `use_thp`, `embedding_across_dp`. The env var `VLLM_PLE_CPU_OFFLOAD` "defaults offload on in some builds".
  - The recipe gives no disk or NVMe option for Engram; its filesystem tier is for KV cache only.
  - [vLLM recipe](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash)
- **vLLM blog**: "CPU-offloaded lookups are prefetched asynchronously" (PR #56512); co-located DP replicas share host tables (PR #57651); THP gives up to 10× faster prefill lookup (PR #56926) — [vLLM blog 2026-10-07](https://vllm.ai/blog/2026-10-07-deepseek-v41-flash)
- **LOCAL vLLM 0.32.0rc1 image**:
  - `vllm/config/engram.py` fields: `cpu_offload: bool = True` (default **on**), `embedding_across_dp`, `dp_shared_memory`, `use_thp`.
  - `models/deepseek_v41/common/engram.py` raises if `cpu_offload and not is_uva_available()`. It has an `EngramBatch` / `_start_prefetch` staging path and Triton lookup kernels.
  - The only O_DIRECT code in vLLM is `v1/simple_kv_offload/disk_backend.py`, which is KV, not Engram.
- **SGLang**: Engram host table via `SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1` and `SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=per_rank`. Only nightly-dev images are listed, e.g. `lmsysorg/sglang:nightly-dev-cu13-20261014-684fea8a`, and no disk flag appears in the portion read. The cookbook states V4.1 "has not shipped in an SGLang release yet" (search summary) — [SGLang cookbook](https://docs.sglang.io/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1)
- **MiaAI-Lab (3–4× DGX Spark, SGLang `lmsysorg/sglang:dev-dsv41`)**:
  - `adapter/row_store.cpp`: "O_DIRECT, one 4 KiB read per row", 96 miss threads (`DSV41_IO_THREADS=96`), sized for NVMe at ~112k IOPS at QD64.
  - `scripts/pack_engram.py` repacks each rank's rows with "weight and scale stored adjacently", about 10 min, ~48 GiB per node at TP4.
  - `DSV41_CACHE_GIB=0`, because "n-gram rows have roughly 0% reuse".
  - Host callbacks cost about 1–3 ms per decode step across both layers. Side-stream prefetch saves 2.3–2.9 ms per step.
  - 51.0 tok/s on 3 Sparks and 87.7 tok/s on 4. Pinned to the 1017 SHA `1bae0e1…`. Licence AGPL-3.0-or-later.
  - [MiaAI-Lab repo](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks)
- **tonyd2wild (4× DGX Spark, vLLM TP4)**: the tables "stay in the safetensors files, and rows are read on demand". The lookup moved into `prepare_inputs` with parallel reads ("Engram fast staging and 128 read threads"), at 2–3 ms per step with node-local rows; NFS cost 5.9–7.8 ms. The README does not say whether it uses O_DIRECT — [tonyd2wild repo](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark)
- **Other NVMe variants**: springrain offers "NVMe or locked RAM Engram offload" on 4× RTX PRO 6000; 0xBakeer runs a single GB10 with hot experts resident plus NVMe streaming (work in progress); helgard-orlm runs on an 8 GB RTX 5060 and reports "~1.6 tok/s disk-only, ~2.4 tok/s with a RAM cache" — [springrain](https://github.com/springrain/deepseek-v4.1-flash-4x-rtx-pro-6000), [0xBakeer](https://github.com/0xBakeer/deepseek-v41-flash-spark), [helgard-orlm](https://github.com/helgard-orlm/deepseek-v41-flash-8gb)
- **Quantization**:
  - NVIDIA converted Engram FP8 → MXFP4, "lossy (about 12% relative L2 error on the table)", which needs vLLM PR #61530 — [nvidia/DeepSeek-V4.1-Flash-NVFP4](https://huggingface.co/nvidia/DeepSeek-V4.1-Flash-NVFP4)
  - LibertAI's 1017 MXFP4 Engram is 97.28 GiB, "cosine 0.9934" — [LibertAI 1017-NVFP4](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-NVFP4)
  - "6-bit engram is indistinguishable from the shipped fp8 tables" on 0909 MLX — [pipenetwork engram6](https://huggingface.co/pipenetwork/DeepSeek-V4.1-Flash-MLX-mixed-4_8bit-engram6)
  - These three were cited in the 10-17 notes and not re-fetched today.
- **Engram is not removable**: with Engram off, perplexity rises to 4.1648 (code) and 9.0627 (prose), versus 1.9094 and 6.4520 with it on — [LibertAI 1017 EXL3 card](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw) (via the 10-17 notes)

### Inferences
- **Addressing**: a decode step's hashes need the token just sampled, so they cannot be issued earlier than sampling. Within the step, though, the hashes for both layers are known before the layer-0 forward begins. The layer-1 fetch therefore has about one layer of slack, and the layer-14 fetch about 13. Pipeline the I/O: issue both at sampling time and wait on layer 1 first.
- **Per-token I/O**: 48 row reads. With O_DIRECT that is 48 × 4 KiB = 192 KiB/token if weight and scale are packed adjacently, or 96 I/Os (384 KiB/token) on raw safetensors. The useful payload is 48 × 264 B = 12.4 KiB, so 4 KiB-sector amplification is about 15×. The LOCAL memory note's "1.2 ms/tok measured" is consistent with a packed store at about 40k IOPS.
- **Prefill**: a P-token prompt needs 48·P rows. At 100k IOPS a 32K-token prompt is about 15 s of Engram I/O alone, so prefill, not decode, is where NVMe Engram hurts. This needs measuring.
- **Pruning/quantizing**: dropping rows requires new primes, which would scramble every hash. A safe "row prune" would have to keep the prime layout and replace cold rows with zeros or a shared fallback, which saves disk only with a sparse format. Requantizing in place (FP8 → 6-bit / MXFP4) keeps addressing intact and halves I/O size only if the row record shrinks below 4 KiB alignment granularity. At 4 KiB O_DIRECT it does not reduce IOPS, so quantization saves **disk capacity** (189 → ~97 GiB) but not decode I/O count.
- Image spans contribute no Engram lookups, so vision-heavy prompts cost less Engram I/O.

### Gaps
- No HF or GitHub artifact under "josephdrose" was found. Searches surfaced the MiaAI, tonyd2wild, 0xBakeer, springrain and helgard-orlm repos instead.
- No source measures quality when rows are cold-zeroed or frequency-truncated.
- The SemiAnalysis claim that DRAM offload can beat HBM for most of the Pareto curve was seen only as a search snippet and is unverified — [SemiAnalysis](https://newsletter.semianalysis.com/p/engrams-embedding-entendre-codesign)

---

## Q4. What changed in -1017 vs earlier V4.1-Flash (0909)

### Takeaway
1017 is a weights-only update. Every one of the 48 safetensors shards has a new sha256, the two Engram shards included, and sizes are unchanged. `config.json`, `inference/` code, the index and the tokenizer are byte-identical. The 1017 repo drops `chat_template.jinja`, the tech-report PDF and `evaluation/` that the 0909 repo carries.

### Cited Findings
- **LFS sha256 comparison**, computed from the HF APIs for [1017](https://huggingface.co/api/models/deepseek-ai/DeepSeek-V4.1-Flash-1017?blobs=true) and [0909](https://huggingface.co/api/models/deepseek-ai/DeepSeek-V4.1-Flash?blobs=true):
  - Identical files: `config.json`, `inference/{config.json,model.py,engram.py,convert.py,kernel.py,...}`, `model.safetensors.index.json` (also `cmp`-identical), `tokenizer.json`, `tokenizer_config.json`, `LICENSE`, `encoding/*`.
  - All 48 `model-*.safetensors` differ. Shard 47 is 0909 `824db488…` vs 1017 `76a05106…`; shard 48 is `976330f4…` vs `9dc51433…`; sizes are identical.
  - Present only in 0909: `chat_template.jinja`, `DeepSeek_V41_Tech_Report.pdf`, `evaluation/*`, `assets/*`.
  - The 0909 repo SHA is `314113f8…` (lastModified 2026-10-16).
- **README**: 1017 "is a minor update to DeepSeek-V4.1-Flash, delivering substantial improvements across reasoning, coding, agentic tasks, and visual benchmarks … reduces task execution time and output length". Terminal-Bench 4.0 goes 31.2 → 42.3 and HLE 36.8 → 42.6. "This release does not include a Jinja-format chat template". Sampling: temperature 1.0, top_p 1.0, max_tokens ≥ 256K — [README](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/resolve/main/README.md)
- **vLLM recipe**: "Architecture, config.json, tokenizer, and prompt encoding match the 0909 launch checkpoint. Only the weights change". Its hardware badges and measured numbers "come from the 0909 checkpoint" — [vLLM recipe](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash)
- tonyd2wild: "The 1017 config, index and tokenizer match it byte for byte, and all 48 shards are new" — [tonyd2wild repo](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark)
- **LibertAI REAP-256E**: the card tells users to fetch shards 47/48 "from `deepseek-ai/DeepSeek-V4.1-Flash`", which is the 0909 repo. It does not state which build its experts came from — [LibertAI REAP-256E](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-REAP-256E)

### Inferences
- **The Engram tables were retrained, or at least re-exported, for 1017.** Mixing 1017 experts with 0909 Engram shards loads silently because shapes match, but it is wrong. Any shipped REAP derivative must pin the 1017 Engram shards by sha256.
- The chat template is gone from 1017. Serving needs `encoding/encoding.py` or the 0909 `chat_template.jinja` (the encoding code is identical), or a server-side DeepSeek-V4.1 parser.

### Gaps
- No source quantifies how far the 1017 weights moved: expert-routing statistics or Engram row deltas. Measuring routing overlap between 0909 and 1017 would show whether 0909 saliency is reusable as a prior.

---

## Q5. Runtime support matrix (vLLM, SGLang, llama.cpp, ik_llama.cpp, ExLlamaV3, TensorRT-LLM) and sm_110 / Jetson Thor

### Takeaway
Upstream V4.1 support exists in vLLM (≥0.30/0.31), TRT-LLM (merged), ik_llama.cpp (merged, including Engram conversion, vision and DSpark drafts) and SGLang (nightly/dev only). llama.cpp mainline PR #28696 is still open, with known correctness bugs, and ExLlamaV3 PR #449 is open.

**No source reports V4.1 running on sm_110 / Jetson Thor.** In the local vLLM image the fused router and MoE ops do have sm_110 cubins. The DSv4.1 attention stack does not: FlashMLA and DeepSelect are sm_90a/sm_100 only, with no PTX, and the SM120 FlashInfer path is selected only for capability major 12. On Thor, vLLM needs either a FlashMLA/DeepSelect rebuild for sm_110 or forcing `FLASHINFER_MLA_SPARSE_DSV41`, which then needs FlashInfer sparse-MLA kernels for sm_110; both are unverified. ik_llama.cpp, which has CUDA plus aarch64 fixes, is the most complete non-vLLM path.

The fused router accepts {…,128,192,256,320,384,…}, so the legal REAP points are 320/256/192/128.

### Cited Findings
- **vLLM**:
  - Minimum 0.31.0 ("architecture first shipped in 0.30.0" via #56228; #57145 KV-offload fix).
  - Backends: Blackwell TP `FLASHINFER_MLA_SPARSE_DSV41`, DEP `FLASHMLA_MEGA_ATTN_DSV41`. MoE `deep_gemm_mega_moe` requires EP.
  - Hardware listed: H100/H200/B200/B300/GB200/GB300/DGX Station plus AMD. "The page does not mention sm_110, sm_120, sm_121, GB10, or Thor".
  - [vLLM recipe](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash)
- **vLLM fused router** (`csrc/moe/topk_softplus_sqrt_kernels.cu`, main):
  - The `switch (num_experts)` has cases 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 192, 320, 384, 448, 576; anything else hits `STD_TORCH_CHECK(false, "Unsupported expert number: ", num_experts)`.
  - A DSv4 fast path runs when `use_hash && topk == 6 && renormalize && (num_experts == 256 || num_experts == 384)`.
  - Arch guards are `__CUDA_ARCH__ >= 900` only.
  - [topk_softplus_sqrt_kernels.cu](https://raw.githubusercontent.com/vllm-project/vllm/main/csrc/moe/topk_softplus_sqrt_kernels.cu) (fetched 2026-10-19)
- **LibertAI on 2× GB10 (TP=2)**:
  - A count of 272 fails with "Unsupported expert number".
  - Required flags: `--enforce-eager` ("CUDA graphs fail"), `--gpu-memory-utilization 0.90`, 32K practical context. DSpark `num_speculative_tokens` must be a multiple of 5.
  - "FlashInfer needs sparse-MLA kernel entries for this model's topk values (1152 and 640), keyed to TP degree. Patching the `.cu` alone is insufficient".
  - [LibertAI REAP-256E](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-REAP-256E)
- **LOCAL image (`vllm/vllm-openai:nightly-aarch64`, 0.32.0rc1.dev190)**:
  - `vllm/models/deepseek_v41/{common,nvidia,amd}/` is present.
  - `torch.cuda.get_arch_list()` returns `sm_80, sm_90, sm_100, sm_110, sm_120`.
  - `cuobjdump --list-elf`: `_moe_C_stable_libtorch` has sm_80/87/89/90/100/110/120, and `_C_stable_libtorch` includes sm_110. `_flashmla_C` and `_flashmla_extension_C` have **sm_90a and sm_100 only**, `_deepselect_C` has **sm_100 only**, and `--list-ptx` is empty for all three.
  - FlashInfer is 0.7.1.
  - The backend selector in `models/deepseek_v41/nvidia/model.py`: "Without an explicit backend: SM12 takes FlashInfer, SM100 takes mega attention where the topology allows it, and everything else keeps the FlashMLA path". An explicit `FLASHINFER_MLA_SPARSE_DSV41` on major ≠ 12 returns `DeepseekV4FlashInferMLAAttention`.
- **vLLM issues on non-datacenter Blackwell**:
  - #56837: "no FlashInfer SM120 sparse-MLA kernel for (num_heads=64, topk=1152); FlashMLA sparse is SM90a/SM100f only".
  - #56461: GB10 at e77daef89 fails on "SWA cache block 32 vs SM120 decode page 64; ratio-1 indexer block_kv=128 vs DeepGEMM sm120 (64 only)".
  - #59203: RTX PRO 6000 "page_block_size=32 has no FlashInfer sparse-MLA decode kernel instantiation".
  - #57144: an SM8x feature request.
  - [vLLM issues search](https://github.com/vllm-project/vllm/issues?q=DeepSeek-V4.1)
- **tonyd2wild on sm_121**:
  - FlashInfer 0.6.18 lacked an SM120 sparse-MLA decode kernel for `page_block_size=32`.
  - A runtime FlashInfer MXFP8 GEMM JIT with 22 jobs "exhausted host memory" and wedged all four nodes; the fix was prebuilt kernels plus `MAX_JOBS=2`.
  - DeepGEMM needed `block_kv` of 32 or 64, so SM12x indexer pages went to 64.
  - `persistent_topk` needs 128 KB of shared memory per block against 99 KB on GB10, so it was replaced by `top_k_per_row_decode`.
  - MiaAI: "`wo_a` has no FP8 kernel on SM121 yet".
  - [tonyd2wild](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark); [MiaAI-Lab](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks)
- **SGLang**: nightly-dev images only. Flags include `--speculative-algorithm DSPARK --speculative-dspark-block-size 5`, `--moe-runner-backend flashinfer_mxfp4` and `--moe-a2a-backend megamoe`. Hardware listed is GB300/GB200/B300/B200/H200/H100 — [SGLang cookbook](https://docs.sglang.io/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1)
  - SGLang PR #38963 fixes "VL routing dropping the fused shared expert": with fusion, top_k becomes 7 and num_experts 385 — [sglang #38963](https://github.com/sgl-project/sglang/pull/38963)
- **llama.cpp**:
  - PR #28696 "model : add DeepSeek V4.1 (DeepseekV41ForCausalLM)" is **OPEN**, last updated 2026-10-10, with 26 comments.
  - It covers conversion plus a runtime graph with Engram tensors `ENGRAM_EMBD/Q/K/WKV`. The tables are written through a disk-backed memmap because whole-tensor FP8 dequant would be about 393 GB of float32. The PR says the tables "want to be host resident".
  - It warns that reusing V4's compress-ratio and q-norm logic "drops the long range half of attention while still producing fluent text".
  - A comment reports that at head `2076412`, layers that reuse a source's selection "attend to every compressed row" (a correctness bug).
  - A community CUDA fork exists: smalinin `my_build_deepseek41`, with GGUFs at `smalinin/DeepSeek-V4.1-Flash-GGUF`.
  - [llama.cpp #28696](https://github.com/ggml-org/llama.cpp/pull/28696)
- **ik_llama.cpp** merged PRs ([ik_llama.cpp PRs](https://github.com/ikawrakow/ik_llama.cpp/pulls?q=deepseek41)):
  - #2455 "model : add DeepSeek-V4.1 (deepseek41)"
  - #2506 "convert: support DeepSeek-V4.1 and write the engram constant keys"
  - #2512 separate-arch graph
  - #2545 vision support
  - #2522 "dflash : load DeepSeek-V4.1 DSpark drafts"
  - #2601 compressed-stream state in sequence state
  - #2456 "iqk: fix DeepSeek-V4 top-k indexer on aarch64"
  - Still open: #2507 "compute the index keys from the pre-RoPE compressed latent", which suggests a correctness fix is pending.
- **ExLlamaV3**: PR #449 "Add DeepseekV41ForCausalLM" is open, #533 "DeepSeek-V4.1: tensor-parallel support" is closed, and the RowTable n-gram table PRs #466/#467/#469 are merged. Community EXL3 1017 kits exist (LibertAI 2.9 bpw; the MiaAI EXL3 2× Spark repo) — [exllamav3 PRs](https://github.com/turboderp-org/exllamav3/pulls?q=DeepSeek); [MiaAI EXL3](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks)
- **TensorRT-LLM**: #19727 "[TRTLLM-16621][feat] Support DeepSeek-V4.1-Flash" **merged** 2026-10-11. Open: #20512 "use BF16 CSA2 projections off SM100/SM103", #20192 NVFP4 target "(sm_103 …)" and #20078 DSpark confidence scheduling — [TRT-LLM PRs](https://github.com/NVIDIA/TensorRT-LLM/pulls?q=V4.1)
- **GGUF of 1017**: `Baekpica/DeepSeek-V4.1-Flash-1017-VQ-GGUF` is sized for one 128 GB Spark, but "serving on DGX Spark has not been measured for this artifact yet" (search summary) — [Baekpica VQ-GGUF](https://huggingface.co/Baekpica/DeepSeek-V4.1-Flash-1017-VQ-GGUF)

| Runtime | V4.1 status (2026-10-19) | Engram | Pruned expert counts | sm_110 / Thor |
|---|---|---|---|---|
| vLLM ≥0.31 (local 0.32.0rc1 nightly aarch64) | Upstream, production | Host pinned + UVA (default on), async prefetch, THP; **no disk**. Community disk patch (tonyd2wild) | Fused router: 128/192/256/320/384 (and 448/512/576) | Router/MoE cubins have sm_110. **FlashMLA/DeepSelect lack sm_110 (no PTX)**, so the default attention path cannot load. FlashInfer DSV41 path untested. Likely needs `--enforce-eager` (GB10 precedent) |
| SGLang | Nightly/dev images only, no release | Host table env flag; MiaAI O_DIRECT row-store adapter | No constraint found | No Thor image; not present locally |
| llama.cpp mainline | PR #28696 open, known attention bug | Tensors converted; host-resident | GGUF `n_expert` header (any count, inference) | CUDA generic; untested |
| ik_llama.cpp | Merged (arch, convert, vision, DSpark drafts) | Engram keys written at convert | Any count (inference) | aarch64 indexer fix merged; untested on Thor |
| ExLlamaV3 | PR #449 open; community kits exist | RowTable n-gram table code merged | Any (inference) | x86 wheels; untested |
| TensorRT-LLM | Merged #19727 | Not checked | Not checked | NVFP4 work targets sm_100/103 |
| Reference `inference/` (TileLang 0.1.12) | Official | Fully resident FP8 table (TP-sharded) | `infer_num_experts` from names; must divide `mp` | TMA and WS disabled, so plausibly portable; untested |

### Inferences
- **Prune-point choice**: if vLLM is the serving target, 192 (−50%) is the only legal count near the memory-note goal (~51%), with 128 the next step down. Non-vLLM runtimes (ik_llama.cpp GGUF, EXL3) accept arbitrary counts.
- **Thor serving risk**: the gating item is attention, not MoE. Validate first, against the local image and away from any running vLLM server per the unified-memory OOM rule, that `--attention-backend FLASHINFER_MLA_SPARSE_DSV41` JIT-compiles on sm_110 for topk 640 and 1152 (128-window + index top-k combinations). Otherwise budget a FlashMLA-sparse and DeepSelect rebuild with `TORCH_CUDA_ARCH_LIST=11.0`, which may not compile because FlashMLA's sm_100 path uses tcgen05; the memory note says tcgen05 is not available on sm_110a.
- **Calibration**: there is no HF `transformers` modeling file for `deepseek_v41` in the repo (`config.json` has `transformers_version 5.6.0`, but there is no `modeling_*.py` and no `auto_map`), so a REAP calibration forward should be built on `inference/model.py`, which is self-contained and TP-aware.

### Gaps
- Whether transformers 5.6.0 natively includes `DeepseekV41ForCausalLM` was not verified.
- FlashInfer 0.7.1 sparse-MLA support for sm_110 was not tested.
- TensorRT-LLM Engram offload support and any Jetson Thor build were not examined.

---

## Q6. Licence and redistribution of pruned derivatives

### Takeaway
MIT, for both code and weights, with no use-based restrictions. Pruned or quantized derivatives may be redistributed if the copyright and permission notice is kept. Existing pruned derivatives (LibertAI REAP-256E) are published as MIT. Note that some community tooling is copyleft (MiaAI: AGPL-3.0), which matters if its row-store code is vendored into a shipped server.

### Cited Findings
- README front-matter `license: mit`. "This repository and the model weights are licensed under the MIT License". The `LICENSE` file (1,084 B) begins "MIT License / Copyright (c) 2023 DeepSeek" and grants rights "to use, copy, modify, merge, publish, distribute, sublicense, and/or sell" — [README](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/resolve/main/README.md); [LICENSE](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/resolve/main/LICENSE)
- The HF API reports `cardData.license = mit` and `gated = False` — [HF API](https://huggingface.co/api/models/deepseek-ai/DeepSeek-V4.1-Flash-1017)
- The `LICENSE` file is byte-identical between 0909 and 1017 (same LFS/blob id, computed).
- LibertAI REAP-256E "inherits the base model's license"; its repo metadata says MIT — [LibertAI REAP-256E](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-REAP-256E)
- MiaAI-Lab's repo is AGPL-3.0-or-later, includes an Apache-2.0 SGLang file and builds on 0xSero's MIT recipe — [MiaAI-Lab](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks)

### Inferences
- A REAP-192 derivative of 1017 can ship under MIT. It can also exclude the Engram shards and point to the upstream 1017 shards by sha256, as LibertAI and the EXL3 kits do, which avoids re-hosting 203 GB.
- Write the row-store and serving code clean-room, or check AGPL compatibility, before shipping anything derived from MiaAI.

### Gaps
- I did not check whether DeepSeek's separate API or usage policies apply to self-hosted weights; under the MIT text they do not.
