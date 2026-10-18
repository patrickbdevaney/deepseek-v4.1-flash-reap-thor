# GLM-5.3 (753B, zai-org) on a single Jetson AGX Thor — GO/NO-GO

Researched 2026-10-17 against live HF repos (config.json, LICENSE, README, all 141 safetensors headers range-read), upstream llama.cpp/vLLM/transformers sources, and the local pipeline/memory. Every byte figure below was computed from the safetensors headers, not from model cards. GiB = 2^30 throughout; "GB" only where a source uses it.

**Bottom line (details in Q8):** GLM-5.3-753B is verdict **(b)/(c): it fits Thor only in the unmeasured region** — REAP-50 needs ≈2.0 bpw on routed experts with attention/dense/shared protected at FP8 (below the ≤2.3 bpw cliff), or 60–70 % expert deletion at the measured-safe 2.5–3.3 bpw (past the >50 % deletion ceiling). The prior internal "fits at REAP-50 if quant ≈ 2.2 bit" was computed on a 110 GB all-tensors-same-bpw budget; under the stated 117 − 12 = 105 GiB rule with the non-expert 19.4 GiB protected, REAP-50 @ 2.2 bpw lands at ~115 GiB and needs ~55 % pruning. It also fails the user's own step-1 decode gate (A40B → ≈10 tok/s roofline ceiling, ≈5 tok/s expected) and the machine has 51 GB of free disk against a 704 GiB FP8 source.

---

## Q1. Exact HF repo, config.json facts, shard dtypes, total bytes, module breakdown, licence

### Takeaway
The repo is **`zai-org/GLM-5.3`** (FP8 e4m3, block 128×128, 703.7 GiB, 753.33B params incl. a 9.95B MTP layer; "743B/744B" in prose = total minus MTP). A BF16 twin `zai-org/GLM-5.3-BF16` (1403 GiB) also exists. Architecture `GlmMoeDsaForCausalLM`: 78 layers (3 dense + 75 MoE) + 1 MTP block, 256 routed experts top-8 + 1 shared, MLA with a 64-dim RoPE part, DeepSeek-Sparse-Attention indexer on 21 "full" layers (57 "shared"), text-only, 1M context, MIT-like licence.

### Cited Findings
- Repo list under zai-org for GLM-5: `GLM-5.3`, `GLM-5.3-BF16`, `GLM-5.3-Flash`, `GLM-5.3-Flash-BF16`, `GLM-5.2`, `GLM-5.2-FP8`, `GLM-5.1`, `GLM-5` — [HF API author=zai-org](https://huggingface.co/api/models?author=zai-org&search=GLM-5). There is no "GLM-5.3-753B" repo; the full model is plain `zai-org/GLM-5.3`.
- `safetensors` metadata on the repo: `F8_E4M3: 751,226,191,872` params, `BF16: 2,103,729,152`, `F32: 19,456`, **total 753,329,940,480**; `usedStorage` 755,681,496,428 B = **703.78 GiB**; 141 shards — [HF API zai-org/GLM-5.3](https://huggingface.co/api/models/zai-org/GLM-5.3).
- `config.json`: `architectures: ["GlmMoeDsaForCausalLM"]`, `model_type: glm_moe_dsa`, `hidden_size 6144`, `num_hidden_layers 78`, `num_attention_heads 64`, `num_key_value_heads 64`, `n_routed_experts 256`, `num_experts_per_tok 8`, `n_shared_experts 1`, `moe_intermediate_size 2048`, `intermediate_size 12288` (dense), `first_k_dense_replace 3`, `mlp_layer_types` = 3 dense + 75 sparse, `q_lora_rank 2048`, `kv_lora_rank 512`, `qk_nope_head_dim 192`, `qk_rope_head_dim 64`, `qk_head_dim 256`, `v_head_dim 256`, `head_dim 192`, `rope_theta 8,000,000`, `rope_interleave true`, `max_position_embeddings 1,048,576`, `vocab_size 154,880`, `tie_word_embeddings false`, `num_nextn_predict_layers 1`, `scoring_func sigmoid`, `topk_method noaux_tc`, `n_group 1`, `topk_group 1`, `routed_scaling_factor 2.5`, `norm_topk_prob true`, `moe_router_dtype float32`, `eos_token_id [154820,154827,154829]` — [config.json](https://huggingface.co/zai-org/GLM-5.3/raw/main/config.json).
- DSA indexer config: `index_n_heads 32`, `index_head_dim 128`, `index_topk 2048`, `index_topk_freq 4`, `index_skip_topk_offset 3`, `index_share_for_mtp_iteration true`, `indexer_types` = **21 "full" + 57 "shared"** (full at layers 0,1,2,6,10,14,…,74) — [config.json](https://huggingface.co/zai-org/GLM-5.3/raw/main/config.json).
- `quantization_config`: `quant_method fp8`, `fmt e4m3`, `activation_scheme dynamic`, `weight_block_size [128,128]`; `modules_to_not_convert` = `lm_head`, `model.embed_tokens`, every norm, every `mlp.gate` (+ `e_score_correction_bias`), every `indexer.k_norm`, `indexers_proj`/`weights_proj`, MTP `enorm/hnorm/eh_proj/shared_head.norm` — [config.json](https://huggingface.co/zai-org/GLM-5.3/raw/main/config.json).
- transformers: `GlmMoeDsaConfig(DeepseekV32Config)`; `indexer_types`: `"full"` runs the indexer, `"shared"` reuses the previous full layer's top-k; `GlmMoeDsaAttention.skip_topk = indexer_types[layer_idx]=="shared"` and `self.indexer = None` on shared layers; `GlmMoeDsaForCausalLM(DeepseekV32ForCausalLM)` — [modular_glm_moe_dsa.py](https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/glm_moe_dsa/modular_glm_moe_dsa.py). The model card requires `transformers_version 5.15.0` — [config.json](https://huggingface.co/zai-org/GLM-5.3/raw/main/config.json).
- README: "GLM-5.3 uses the same base model as GLM-5.2 — every gain comes from post-training"; serving listed for SGLang, vLLM, TokenSpeed, Transformers, KTransformers, Unsloth; `reasoning_effort` ∈ {low, high, max}, default max; vendor benchmarks Terminal-Bench 2.1 88.2, DeepSWE 66.9, HLE w/ tools 62.5, Toolathlon 73.0, AutomationBench 48.2 — [README](https://huggingface.co/zai-org/GLM-5.3/resolve/main/README.md). Text-only: the config has no `vision_config` and the 141 headers contain no vision tensors (computed, see Q2).
- Licence: "GLM-5.3 License … Permission is hereby granted, free of charge … to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies … fine-tune … create derivative works"; sole extra condition: a "Model as a Service" business with >US$10B revenue in any 12 months must pass Z.AI's security review — [LICENSE](https://huggingface.co/zai-org/GLM-5.3/resolve/main/LICENSE). HF tags it `license: other`, `license_name: glm-5.3`.
- Per-tensor-group bytes/params from all 141 safetensors headers (my computation, scales excluded from param counts):

| group | bytes (GiB) | share | params (B) | on-disk dtype |
|---|---:|---:|---:|---|
| routed experts (75 layers × 256) | 675.16 | 95.94 % | 724.776 | F8_E4M3 (+F32 block scales 0.16 GiB) |
| attention (MLA, 78 layers) | 11.99 | 1.70 % | 12.872 | F8_E4M3 |
| MTP layer 78 (256 experts + attn + eh_proj) | 9.34 | 1.33 % | 9.953 | F8_E4M3 (+BF16 0.14) |
| shared expert (75 layers) | 2.64 | 0.37 % | 2.831 | F8_E4M3 |
| lm_head | 1.77 | 0.25 % | 0.952 | BF16 |
| embed_tokens | 1.77 | 0.25 % | 0.952 | BF16 |
| router (gate + bias) | 0.43 | 0.06 % | 0.344 | BF16 gate, F32 bias |
| dense FFN layers 0–2 | 0.42 | 0.06 % | 0.453 | F8_E4M3 |
| DSA indexer (21 full layers + MTP) | 0.19 | 0.03 % | 0.197 | F8_E4M3 / BF16 |
| norms | ~0 | — | 0.001 | BF16 |
| **total** | **703.72** | | **753.330** | |

  Sample shapes (layer 3): `experts.N.gate_proj [2048,6144]`, `up_proj [2048,6144]`, `down_proj [6144,2048]` → 37.75M params/expert; `q_a_proj [2048,6144]`, `q_b_proj [16384,2048]`, `kv_a_proj_with_mqa [576,6144]`, `kv_b_proj [28672,512]`, `o_proj [6144,16384]`; indexer `wq_b [4096,2048]`, `wk [128,6144]`, `weights_proj [32,6144]`, `k_norm` LayerNorm with bias — headers of [model-00001…00141](https://huggingface.co/zai-org/GLM-5.3/tree/main).
- Reconciliation of "743B" vs "753B": 753.33B − 9.95B (MTP) = 743.4B main model. pipenetwork and autotrust say "744B"; 0xSero, nota-ai and the HF safetensors counter say "753B" — [pipenetwork card](https://huggingface.co/pipenetwork/GLM-5.3-REAP50-MLX-4bit), [nota-ai card](https://huggingface.co/nota-ai/GLM-5.3-Nota-NVFP4-Global-Pruned-17.75).
- GLM-5.3-BF16: `usedStorage` 1,506,687,604,850 B = 1403 GiB, 282 shards — [HF API](https://huggingface.co/api/models/zai-org/GLM-5.3-BF16).

### Inferences
- Routed-expert share by params = 724.776 / 753.330 = **96.2 %** (97.5 % if MTP's 256 experts are counted as experts and the denominator is the 743B main model; the prior memory's 97.5 % used that convention). Either way it is more expert-dominated than Flash (94.8 %).
- Attention is 12.87B params = 165M/layer — larger per layer than Flash's MLA (hidden 6144, 64 heads, q_lora 2048). This matters for decode bytes (Q8).
- Unlike Flash (`qk_rope_head_dim = 0`, pure NoPE), the 753B MLA carries a 64-dim RoPE part with `rope_interleave` — the 0731 MLA-with-rope kernel half that the Flash server skipped IS needed here.

### Gaps
- GLM-5 technical report (arXiv 2602.15763) abstract only confirms "adopts DSA"; the full paper's parameter/layer table was not read.

---

## Q2. Routed-expert bytes, floor with all experts removed, non-expert bytes at FP8 / NVFP4

### Takeaway
Routed experts are 724.776B params = 675.2 GiB at FP8; the floor with all routed experts removed (MTP dropped) is ≈19.4 GiB at FP8 (≈11.0 GiB if attention/dense/shared go NVFP4), i.e. the non-expert term alone is 18 % of the 105 GiB envelope.

### Cited Findings
- Routed experts: 724.776B params, 9.664B/layer, 37.75M/expert; MTP layer holds a further 256-expert MoE (9.664B) plus 0.289B of attention/indexer/`eh_proj` — computed from headers, [zai-org/GLM-5.3](https://huggingface.co/zai-org/GLM-5.3/tree/main).
- Non-expert (main model, MTP excluded) = **18.601B**: attention 12.872 + indexer 0.197 + dense FFN 0.453 + shared expert 2.831 + embed 0.952 + lm_head 0.952 + router 0.344 + norms 0.001 — computed from headers.
- NVFP4 real density on vendor checkpoints is 0.60–0.64 B/weight (own measurement 0.663 on the Flash REAP50 NVFP4) — [memory: thor-base-model-shortlist](/home/patrickd/.claude/projects/-home-patrickd/memory/thor-base-model-shortlist.md).

### Inferences (shown arithmetic)
- Expert bytes at nominal routed-expert bpw b: `724.776e9 × b / 8 / 2^30` → 4.0 bpw 337.5 GiB; 3.5 → 295.3; 3.0 → 253.1; 2.5 → 210.9; 2.2 → 185.6; 2.0 → 168.8 GiB. (These are effective bits including scale overhead; GGUF IQ2_XXS = 2.06, EXL3 "3.0bpw" ≈ 3.0+, NVFP4 ≈ 4.5–5.0 effective.)
- Fixed floor A (protected): attention+indexer+dense+shared at FP8 (16.353 GB = 15.23 GiB) + embed+lm_head at BF16 (3.81 GB = 3.55 GiB) + router/norms BF16 (0.64 GiB) = **19.42 GiB**. Floor B: same but embed+head FP8 = 17.65 GiB. Floor C: attention/dense/shared at NVFP4 (4.5 bpw eff.) + embed/head FP8 = **10.98 GiB**. Keeping MTP adds 0.27 GiB non-expert + its 256 experts at the same prune ratio and bpw (≈1.2 GiB at REAP-50 @ 2.2 bpw).

### Gaps
- None material; all numbers are from headers.

---

## Q3. KV cache: layout, bytes/token at 32k/128k, FP8-KV feasibility, reconciliation with the "89,856 B + ~10 KB indexer" estimate

### Takeaway
MLA latent cache = 576 × 2 B × 78 layers = **89,856 B/token at BF16 — exactly the structural researcher's number**. The indexer key cache exists only on the 21 "full" layers (shared layers carry none), so the DSA term is **5,376 B/token BF16 (≈2,772 B FP8), not ~10 KB** — the 10 KB figure counted all 78 layers. KV is a minor term: 2.9 GiB at 32k, 11.6 GiB at 128k (BF16); 1.5 / 5.8 GiB at FP8.

### Cited Findings
- Per-layer cached latent = `kv_lora_rank 512 + qk_rope_head_dim 64 = 576` (the `kv_a_proj_with_mqa` output is [576, 6144]) across all 78 layers — [config.json](https://huggingface.co/zai-org/GLM-5.3/raw/main/config.json) and headers.
- llama.cpp: "only 'full' layers own an indexer, so the shared layers need no indexer cache" (`filter_lid = hparams.is_indexer_full(il)`), and "The NextN/MTP draft head runs dense MLA (no DSA indexer)" — [llama-model.cpp](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/src/llama-model.cpp) (lines ~2823–2943 at fetch time).
- transformers: shared layers "reuse the previous full layer's top-k selection" and have `self.indexer = None` — [modular_glm_moe_dsa.py](https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/glm_moe_dsa/modular_glm_moe_dsa.py).
- FP8 KV on this exact architecture: nota-ai serves `--kv-cache-dtype fp8_e4m3` on 2×B300 and reports 1,590,400 cached tokens at `gpu-memory-utilization 0.96` — [nota-ai card](https://huggingface.co/nota-ai/GLM-5.3-Nota-NVFP4-Global-Pruned-17.75); Mia-AiLab serves "FP8 KV cache" and an experimental "4-bit KV cache" at 655,360 tokens — [Mia-AiLab card](https://huggingface.co/Mia-AiLab/GLM-5.3-EXL3-2.75bpw-TensorFold).
- Indexer is dense/no-op below ~2048 tokens (`index_topk 2048`): "Prompts up to 2048 tokens are unaffected (the indexer is bypassed below index_topk)" — [pipenetwork card](https://huggingface.co/pipenetwork/GLM-5.3-REAP50-MLX-4bit); the user's Flash server measured the exact limit at 2051 — [memory: glm53-cuda-server-build](/home/patrickd/.claude/projects/-home-patrickd/memory/glm53-cuda-server-build.md).

### Inferences (shown arithmetic)
- MLA: 78 × 576 × 2 B = 89,856 B/tok (BF16); 44,928 B/tok (FP8).
- Indexer keys: 21 × 128 × 2 B = 5,376 B/tok (BF16); 21 × (128 + 4) ≈ 2,772 B/tok (FP8 + scale). The ~10 KB estimate = 78 × 128 = 9,984 B assumed an indexer on every layer; the config's `indexer_types` and both runtimes say 21.
- Totals: BF16 95,232 B/tok → **32k = 2.91 GiB, 128k = 11.62 GiB**; FP8 ≈ 47,700 B/tok → **32k = 1.46 GiB, 128k = 5.82 GiB**.
- DSA bounds attention compute (top-2048 keys/query) but NOT KV storage — the full latent cache for every token must stay resident for the indexer to score it.

### Gaps
- Exact indexer-cache dtype in llama.cpp (whether it is stored at FP8 or F16) was not read; both bounds are given above.

---

## Q4. Memory budget table and required expert-removal ratio at 4.0 / 3.5 / 3.0 / 2.5 bpw vs the documented ceilings

### Takeaway
With attention/dense/shared protected at FP8 and 32k BF16 KV, the expert budget is 82.7 GiB and the required removal is **75.5 % @ 4.0, 72.0 % @ 3.5, 67.3 % @ 3.0, 60.8 % @ 2.5, 55.5 % @ 2.2, 51.0 % @ 2.0 bpw**. Every measured-safe bit-width (≥2.5 with protection) demands >60 % deletion — past the >50 % ceiling; REAP-50 only fits at ≈2.0 bpw (FP8 non-expert) or ≈2.2 bpw with the non-expert tensors also pushed to NVFP4 and FP8 KV, i.e. with the protection precondition removed.

### Cited Findings
- Envelope: 117 GiB usable, 240 GB/s measured — [memory: thor-model-fit-frontier](/home/patrickd/.claude/projects/-home-patrickd/memory/thor-model-fit-frontier.md); 12 GiB runtime reserve per the task brief.
- Documented ceilings (task brief + local notes): no generalist survives >50 % deletion within 10 %; 2.5–2.8 bpw is fine only with attention/dense/shared protected; ≤2.3 bpw is a cliff everywhere; the lowest-risk measured point is 50 % @ 3.0–3.3 bpw dynamic — [prune_quant_pareto_frontier.md](/home/patrickd/glm-5.3-reap/research_notes/Compressing%201T%20MoE%20for%20Thor/prune_quant_pareto_frontier.md), [extreme_quantization.md](/home/patrickd/glm-5.3-reap/research_notes/Compressing%201T%20MoE%20for%20Thor/extreme_quantization.md) ("the measured quality cliff sits between ~2.9 and ~2.4 bpw on routed experts … safe zone ≈3.0–3.6 true bpw; 2.4–2.8 usable only with 'smol'/dynamic recipes and costs 15–25 % ppl").
- Prior internal estimate: ~110 GB weight budget, uniform B/w across all tensors: NVFP4 0.5625 → 75.5 %, 3-bit 0.42 → 66.4 %, 2.5-bit 0.34 → 57.9 %, 2.2-bit 0.28 → 48.3 % — [memory: glm53-753b-thor-envelope](/home/patrickd/.claude/projects/-home-patrickd/memory/glm53-753b-thor-envelope.md).

### Inferences — the budget table (every term shown; experts = 724.776B params; MTP dropped unless stated)

Budget = 117 − 12 (reserve) − fixed − KV. Required keep = budget / (724.776e9 × bpw / 8 / 2^30). Experts/layer = keep × 256.

**Scenario A — protected non-expert (attn/indexer/dense/shared FP8 = 15.23 GiB, embed+head BF16 = 3.55, router/norms 0.64) → fixed 19.42 GiB**

| ctx / KV | KV GiB | expert budget | 4.0 bpw | 3.5 | 3.0 | 2.5 | 2.2 | 2.0 |
|---|---:|---:|---|---|---|---|---|---|
| 32k BF16 | 2.91 | 82.68 | prune 75.5 % (63 exp) | 72.0 % (72) | 67.3 % (84) | 60.8 % (100) | 55.5 % (114) | 51.0 % (125) |
| 32k FP8 | 1.46 | 84.13 | 75.1 % (64) | 71.5 % (73) | 66.8 % (85) | 60.1 % (102) | 54.7 % (116) | 50.1 % (128) |
| 128k BF16 | 11.62 | 73.96 | 78.1 % (56) | 75.0 % (64) | 70.8 % (75) | 64.9 % (90) | 60.2 % (102) | 56.2 % (112) |
| 128k FP8 | 5.82 | 79.76 | 76.4 % (60) | 73.0 % (69) | 68.5 % (81) | 62.2 % (97) | 57.0 % (110) | 52.7 % (121) |

**Scenario B — non-expert also compressed (attn/dense/shared NVFP4 ≈4.5 bpw eff. = 7.2 GiB, embed+head FP8 = 1.77, router/norms 0.64) → fixed 10.98 GiB** [this violates the "protect attention/dense/shared" precondition of the 2.5–2.8 bpw regime]

| ctx / KV | KV GiB | expert budget | 4.0 | 3.5 | 3.0 | 2.5 | 2.2 | 2.0 |
|---|---:|---:|---|---|---|---|---|---|
| 32k BF16 | 2.91 | 91.11 | 73.0 % (69) | 69.1 % (79) | 64.0 % (92) | 56.8 % (111) | 50.9 % (126) | 46.0 % (138) |
| 32k FP8 | 1.46 | 92.56 | 72.6 % (70) | 68.7 % (80) | 63.4 % (94) | 56.1 % (112) | **50.1 % (128)** | 45.1 % (140) |
| 128k FP8 | 5.82 | 88.19 | 73.9 % (67) | 70.1 % (76) | 65.2 % (89) | 58.2 % (107) | 52.5 % (122) | 47.7 % (134) |

Keeping MTP (sliced to the same ratio and bpw) shifts Scenario A at 32k BF16 by +0.4–0.8 pp of required prune (e.g. 2.2 bpw: 56.2 %, 2.0 bpw: 51.8 %).

- Bit-width needed for exactly REAP-50 under Scenario A: 32k BF16 KV → 82.68 × 2^30 × 8 / (0.5 × 724.776e9) = **1.96 bpw**; 128k BF16 → 1.75 bpw; 128k FP8 → 1.89 bpw.
- Reconciling the prior "2.2 bit → 48.3 %": it used 110 GB (decimal) ≈ 102.4 GiB for weights alone, zero KV, and the same 0.28 B/w on attention/embeddings/shared. Correcting the three assumptions (105 GiB weights+KV; non-expert protected at FP8 = 19.4 GiB; 2.9 GiB KV) moves the required prune from 48 % to **55.5 %**, or equivalently the needed expert width from 2.2 to **2.0 bpw**. The only way to recover the prior number is Scenario B (everything to ≈NVFP4/FP8, FP8 KV): 50.1 % @ 2.2 bpw.
- Mapping to ceilings:
  - 4.0 / 3.5 / 3.0 bpw: 64–78 % deletion → **far past the >50 % ceiling** (2609.04453: router-mass criteria score 0.00 at r=0.5 on gpt-oss-120B; Half-the-Experts 25 %-keep collapses — [expert_pruning_beyond_50.md](/home/patrickd/glm-5.3-reap/research_notes/Compressing%201T%20MoE%20for%20Thor/expert_pruning_beyond_50.md)).
  - 2.5 bpw: 60–65 % deletion → past the ceiling AND at the bottom of the "only with protection" band.
  - 2.2 bpw (Scenario A): 55 % deletion, and ≤2.3 bpw is "a cliff everywhere".
  - 2.0 bpw: REAP-50 fits, but 2.0 bpw sits below every measured-safe point; the only ≤2.1 bpw artifacts of this model are IQ1_S/IQ1_M/IQ2_XXS class (Q5 shows their cost).
  - The "lowest-risk measured point" (50 % @ 3.0–3.3 bpw) needs 253–278 GiB of experts + 19.4 fixed ≈ **272–297 GiB**: 2.6–2.8× the envelope.

### Gaps
- Effective-bpw overhead of the specific quantizer (GGUF IQ-types vs EXL3 vs NVFP4) shifts each column by ±0.1–0.3 bpw; the table uses nominal effective bits.

---

## Q5. Quality expectation at the feasible points — closest published analogues

### Takeaway
Both axes have been measured on this exact model and both are bad at the required settings: unpruned GLM-5.3 at 2.30 bpw (Unsloth UD-IQ1_S) costs **+72 % PPL / mean KLD 0.69 / top-1 72.6 %**, and REAP-50 at 4-bit (pipenetwork) costs **+75.6 % wikitext PPL** vs unpruned 4-bit. 0xSero's REAP ladder on GLM-5.3 shows KL rising ~linearly with experts removed (0.089 → 0.511 at 34 %) and states "the pruning drives the fidelity; the quant format barely moves it". Stacking REAP-50 with ≈2.0 bpw therefore has no measured analogue better than "both cliffs at once"; the closest prior data points (GLM-5 REAP-50 + IQ2_M 130 GB, GLM-5.2 REAP50 + Q3_K_M +37.5 % PPL) are the same family at a *looser* operating point than Thor requires.

### Cited Findings
- Unsloth GLM-5.3 (full model) GGUF ladder, vs their Q6/Q8 baseline: UD-IQ1_S 216.7 GB top-1 72.56 %, mean KLD 0.688, PPL 4.6130; UD-IQ1_M 228.5 GB KLD 0.565, PPL 4.1410; UD-IQ2_M 238.6 GB KLD 0.454, PPL 3.7433; UD-Q2_K_XL 253.9 GB KLD 0.374, PPL 3.5048; UD-IQ3_XXS 281.7 GB KLD 0.273, PPL 3.2482; UD-Q3_K_XL 343.0 GB KLD 0.141, PPL 2.9107; UD-IQ4_XS 365.3 GB KLD 0.101, PPL 2.8460; UD-Q4_K_XL 467.3 GB KLD 0.037, PPL 2.7006; UD-Q6_K_XL 684.4 GB KLD 0.013, PPL 2.6771 — [unsloth docs GLM-5.3](https://unsloth.ai/docs/models/glm-5.3). Repo sizes: UD-IQ1_S 201.8 GiB, UD-IQ2_M 222.2 GiB, UD-Q2_K_XL 236.4 GiB — [HF API unsloth/GLM-5.3-GGUF](https://huggingface.co/api/models/unsloth/GLM-5.3-GGUF).
- pipenetwork GLM-5.3 MLX ladder, wikitext-2 test PPL (288,627 tokens, identical windows): 4bit 418.6 GB 2.8636; mixed-4_8bit 2.7420; mixed-3_6bit 332.6 GB 3.0338; **REAP25-4bit 316.6 GB 3.2872; REAP37-4bit 267.2 GB 3.8517; REAP50-4bit 214.7 GB 5.0295 [4.7571, 5.3137]**. REAP saliency = mean `router_weight × ‖expert_output‖` over only 65,536 calibration tokens; kept experts carry 67.3 % of saliency mass; MTP dropped; "Greedy generation … is coherent on every published build" — [pipenetwork/GLM-5.3-REAP50-MLX-4bit](https://huggingface.co/pipenetwork/GLM-5.3-REAP50-MLX-4bit).
- 0xSero GLM-5.3 REAP ladder (EXL3 3.0 bpw experts, attention/shared/router BF16; KL vs BF16 on a sealed 25-prompt panel over the full 154,880 vocab): 256 kept 293 GB KL 0.089; 224 → 258 GB 0.195; 208 → 240 GB 0.283; 192 → 223 GB 0.361 (top-1 ≈82 %); 180 → 210 GB 0.428; **168 (34 % pruned) → 197 GB 0.511 (top-1 ≈78 %)**. Max-over-domain criterion KL 0.361 vs frequency-only 0.635 at keep-192. "The pruning drives the fidelity; the quant format barely moves it." No row at or below 128 experts — [0xSero/GLM-5.3-500B-EXL3-3.0bpw](https://huggingface.co/0xSero/GLM-5.3-500B-EXL3-3.0bpw), [fidelity study](https://huggingface.co/datasets/0xSero/glm-5.3-reap-fidelity-study).
- Mia-AiLab EXL3 2.75 bpw (experts only, mixed 2/3/4-bit per expert, everything else BF16), unpruned: KL 0.086 all sets (math 0.012, code 0.035, chat 0.043, wiki 0.126, tool calls 0.201), top-1 92.8 %; "as served" with dense layers at 4-bit: KL 0.124, top-1 90.8 %; 273 GiB — [Mia-AiLab card](https://huggingface.co/Mia-AiLab/GLM-5.3-EXL3-2.75bpw-TensorFold).
- autotrust SLIM-E192 (25 % pruned by routing frequency) FP8 vs original, author's vLLM A/B: HumanEval 95.1 → 95.1, CyberMetric 88.0 → 87.7, BFCL live 69.6 → 69.4, BFCL multi-turn 73.0 → 72.5, AIME −2.5 pt, C-Eval 92.0 → 84.7, GPQA-Diamond 86.9 → 83.3; IQ2_XXS build "2.06 bits/weight on routed experts, Q8_0 elsewhere" = 149.7 GiB — [autotrust/GLM-5.3-GGUF-DGX-Spark](https://huggingface.co/autotrust/GLM-5.3-GGUF-DGX-Spark).
- nota-ai global-pruned 17.75 % (208–256 experts per layer, non-uniform) + NVFP4: Terminal-Bench 2.1 88.2 → 83.1, DeepSWE 66.9 → 63.7 ("inside the sampling noise of a single run"), 365 GiB — [nota-ai card](https://huggingface.co/nota-ai/GLM-5.3-Nota-NVFP4-Global-Pruned-17.75).
- Prior analogues (local notes): GLM-5 744B REAP-50 + UD-IQ2_M 2.72 bpw, 130 GB, 50-item pilot 80 % vs 88 % for the same prune at Q3_K_M, no vs-unpruned column; GLM-5.2 REAP50 + Q3_K_M 169 GB "~+37.5 % perplexity" vs full GLM-5.2 (REAP-25 sibling +2.3 %) — [sota_shipped_artifacts.md](/home/patrickd/glm-5.3-reap/research_notes/Compressing%201T%20MoE%20for%20Thor/sota_shipped_artifacts.md).
- User's own GLM-5.3-Flash REAP-50: dNLL 0.1909 = +21.0 % ppl; one GGUF bit-width step +1.46 %; NVFP4 +1.56 %; pruning ≈14× more damaging per unit compression — [memory: reap-criterion-and-corpus-audit](/home/patrickd/.claude/projects/-home-patrickd/memory/reap-criterion-and-corpus-audit.md).

### Inferences
- Effective bpw of the Unsloth rows (GB × 8e9 / 753.33e9): UD-IQ1_S 2.30, UD-IQ1_M 2.43, UD-IQ2_M 2.53, UD-Q2_K_XL 2.70, UD-IQ3_XXS 2.99, UD-Q3_K_XL 3.64, UD-IQ4_XS 3.88. PPL vs Q6: +72 %, +55 %, +40 %, +31 %, +21 %, +8.7 %, +6.3 %. This is the 753B's own quant-only cliff and it sits where the documented ceiling says (≤2.3 collapses; 2.5–2.7 costs 30–40 % even with Unsloth's protected tensors).
- EXL3 trellis at 2.75 bpw (KL 0.086) is dramatically better than GGUF at 2.70 (KLD 0.374) on this model — different test sets, so directional only — which says the ≈2.0–2.2 bpw corner is only worth attempting with an EXL3/KT-class quantizer, not IQ2_XXS. [directional]
- Linear extrapolation of the 0xSero ladder (ΔKL ≈ 0.0048 per expert removed) to 128 kept gives KL ≈ 0.70 at 3 bpw — the same KL class as unpruned IQ1_S. [extrapolation]
- The pipenetwork REAP-50 (+75.6 % PPL) used only 65k calibration tokens and the mean criterion; the user's mass criterion + 5.5M-token balanced corpus + HOPE should land well below that (Flash: +21 % at 0.52M tokens). But the required *additional* ≈2.0 bpw expert quant on this model adds a measured +40–72 % PPL band on its own (unpruned), so the stacked expectation is roughly ×1.6–2.2 PPL, i.e. nothing like the "within 10 %" bar. [extrapolation from two single-axis measurements]
- The only sub-130 GiB measured points for ≥500B sources remain ubergarm R1-0528 IQ1_S_R4 (+52 % PPL) and GLM-5 REAP-50 IQ2_M (pilot only) — [sota_shipped_artifacts.md](/home/patrickd/glm-5.3-reap/research_notes/Compressing%201T%20MoE%20for%20Thor/sota_shipped_artifacts.md); a GLM-5.3 REAP-50 @ ~2.0 bpw would be a new, worse operating point than both.

### Gaps
- No published GLM-5.3 artifact combines ≥50 % pruning with ≤3 bpw; the stacked cost is an extrapolation.
- 0xSero's KL panel is 25 prompts; Unsloth's KLD set is unspecified; pipenetwork is wikitext only — none measures agentic/tool-use multi-turn, the documented REAP failure mode.

---

## Q6. Existing artifacts on HF and what can serve this architecture on Thor

### Takeaway
No pruned sub-4-bit GLM-5.3-753B artifact ≤130 GiB exists; the smallest is autotrust's 25 %-pruned IQ2_XXS at 149.7 GiB (two DGX Sparks). REAP checkpoints exist only at 17.75–50 % in EXL3/MLX/NVFP4 forms (none in FP8/BF16 safetensors publicly). llama.cpp supports `glm-dsa` upstream including the shared-indexer schedule (the local checkout has it); vLLM maps `GlmMoeDsaForCausalLM` onto its DeepSeek-V3.2 implementation whose sparse-attention kernels are unverified on sm_110.

### Cited Findings
- Full-model quants: `nvidia/GLM-5.3-NVFP4` 432.3 GiB; `unsloth/GLM-5.3-GGUF` UD-IQ1_S 201.8 / UD-IQ1_M 212.8 / UD-IQ2_M 222.2 / UD-Q2_K_XL 236.4 / UD-IQ3_XXS 262.3 / UD-Q3_K_XL 319.4 / UD-IQ4_XS 340.2 GiB; `antirez/glm-5.3-gguf` UD-IQ2_XXS routed IQ2_XXS 196.6 GiB (no card, no evals); `Infatoshi/GLM-5.3-UNCENSORED-EXL3-3.0bpw` 272.6 GiB; `Mia-AiLab/GLM-5.3-EXL3-2.75bpw-TensorFold` 272.7 GiB; `Justvugg/GLM-5.3-colibri-int4-g64` 390.5 GiB; `kindlingai/glm-5.3-exlr8-k2-k3-k4` 780 GiB — [HF API](https://huggingface.co/api/models?search=GLM-5.3).
- Pruned: `0xSero/GLM-5.3-{661,615,569,533,500}B-EXL3-3.0bpw` (224→168 kept; 258→197 GB; 183.6 GiB for 500B), `0xSero/GLM-5.3-{569,621,488}B-EXL3-TR3-3.42bpw`, `0xSero/GLM-5.3-569B-W4A16`; `pipenetwork/GLM-5.3-REAP{25,37,50}-MLX-4bit` (REAP50 = 199.9 GiB, MLX only); `nota-ai/GLM-5.3-Nota-NVFP4-Global-Pruned-17.75` 365.5 GiB (non-uniform per-layer counts, needs a patched vLLM `deepseek_v2.py`); `autotrust/GLM-5.3-SLIM-E192` (gated — API returns "Invalid username or password") and its GGUF `autotrust/GLM-5.3-GGUF-DGX-Spark` 149.7 GiB — [HF API](https://huggingface.co/api/models?search=GLM-5.3%20REAP), [0xSero](https://huggingface.co/0xSero/GLM-5.3-500B-EXL3-3.0bpw), [autotrust](https://huggingface.co/autotrust/GLM-5.3-GGUF-DGX-Spark).
- HF searches for "GLM-5.3 HOPE", "GLM-5.3 ubergarm", "GLM-5.3 753B" return nothing; ubergarm has only GLM-5 and GLM-5.1 GGUFs — [HF API](https://huggingface.co/api/models?author=ubergarm&search=GLM-5).
- llama.cpp upstream: `LLM_ARCH_GLM_DSA "glm-dsa"`; `llama_model_glm_dsa`; hparams `is_indexer_full_impl` ("Shared indexers reuse top-k from previous full layer"), `GLM_5_2_DEFAULT_INDEXER_TYPES` back-compat table; indexer tensors loaded as `TENSOR_NOT_REQUIRED` — [llama-arch.cpp](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/src/llama-arch.cpp), [llama-model.cpp](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/src/llama-model.cpp); PR #24770 "model : glm-dsa load DSA indexer tensors as optional" merged 2026-06-20; PR #27773 "add GLM-5.3-Flash (GLM5-Next) support" merged 2026-09-30; PR #27917 "Add MTP for GLM-5.3-Flash" still open — [GitHub API](https://api.github.com/repos/ggml-org/llama.cpp/pulls/27773). autotrust: "`glm-dsa` is supported by upstream llama.cpp" — [autotrust card](https://huggingface.co/autotrust/GLM-5.3-GGUF-DGX-Spark).
- Local: `/home/patrickd/llama.cpp` at 465e49b9c (2026-09-07) has `src/models/glm-dsa.cpp` with 87 indexer references and `gguf-py` `MODEL_ARCH.GLM_DSA`; its `build/bin/llama-server` is dated Apr 5 2026 (stale vs the checkout); `/home/patrickd/prism-llama.cpp` (Jul 19 2026) has a Jul 27 binary — local `ls`/`git log`.
- vLLM upstream registry: `"GlmMoeDsaForCausalLM": ("vllm.models.deepseek_v32", "GlmMoeDsaForCausalLM")` — [registry.py](https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/model_executor/models/registry.py); nota-ai (vLLM 0.28.0): "vLLM serves GLM-5.3 through its DeepSeek modeling file — GlmMoeDsaForCausalLM is registered to deepseek_v2" — [nota-ai card](https://huggingface.co/nota-ai/GLM-5.3-Nota-NVFP4-Global-Pruned-17.75). vLLM's DSA launch targeted H100/H200/B200 — [vLLM blog](https://vllm.ai/blog/deepseek-v3-2). Thor: pip vLLM wheels lack sm_110 kernels, build from source — [HackMD Spark+Thor cluster](https://hackmd.io/@johnnynunez/HJ6uQtec-x). No source shows GLM-5.3 (full) on a Thor.
- SGLang cookbook lists H200/B200/B300/GB300/MI300-class hardware only — [SGLang GLM-5.3](https://docs.sglang.io/cookbook/autoregressive/GLM/GLM-5.3).
- Mia-AiLab/TensorFold runs EXL3 2.75 bpw on **three** DGX Sparks (~84 GiB/Spark) at 29–40 tok/s with the DSpark speculator — [Mia-AiLab card](https://huggingface.co/Mia-AiLab/GLM-5.3-EXL3-2.75bpw-TensorFold). autotrust: one Spark with the 149.7 GiB file = "partial residency … low single-digit t/s" — [autotrust card](https://huggingface.co/autotrust/GLM-5.3-GGUF-DGX-Spark).
- The user's Flash REAP50 GGUFs (IQ3_M/IQ4_XS, `glm5-next`) were served on Thor with llama.cpp, and `glm5-next` uses the same DSA indexer-cache code path ("the DSA layers use a K-only MLA cache plus an indexer cache") — [memory: glm53-quant-hidden-state-drift](/home/patrickd/.claude/projects/-home-patrickd/memory/glm53-quant-hidden-state-drift.md), [llama-model.cpp](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/src/llama-model.cpp).

### Inferences
- A REAP-50 + IQ2_XXS GGUF of the 753B can be sized from autotrust's E192 build: non-expert Q8_0 ≈ 18.6B × 1.06 ≈ 18.4 GiB; experts at 192 kept = 149.7 − 18.4 = 131.3 GiB → 128 kept = 87.5 GiB → **≈106 GiB total before KV** — right at the 105 GiB line, with a 2.06-bpw quantizer the 753B measurements (Q5) already show is in the +70 % PPL class.
- llama.cpp is the only runtime with a credible Thor path today (CUDA, `glm-dsa` upstream, IQ kernels already run on sm_110 for Flash); the open question is whether the sparse-indexer kernel runs on sm_110 above 2051 tokens — the Flash serving may never have exercised it. vLLM's DSA path depends on sparse-MLA kernels (FlashMLA/FlashInfer) that are undocumented for sm_110 and the user already found FLASHINFER invalid for MLA on Thor — [memory: mistral-small4-thor-stack](/home/patrickd/.claude/projects/-home-patrickd/memory/mistral-small4-thor-stack.md).
- exllamav3/TensorFold (the EXL3 quantizer that makes 2.75 bpw look good) has only been shown on sm_121 DGX Spark; sm_110 is unverified.
- None of the existing REAP artifacts short-cuts the pipeline: the only REAP-50 checkpoint is MLX 4-bit (not a usable FP8/BF16 master), 0xSero stops at 34 %, autotrust's FP8 E192 is gated.

### Gaps
- Whether llama.cpp's `glm-dsa` sparse indexer path (not the ≤2051-token dense fallback) runs on sm_110 — UNVERIFIED.
- Whether the local Apr-5 `llama-server` binary or the prism fork served the Flash GGUFs — UNVERIFIED.
- exllamav3 on sm_110 — UNVERIFIED.

---

## Q7. Pipeline delta: what in the GLM-5.3-Flash REAP pipeline changes for the 753B

### Takeaway
The model class, config field names, expert/layer counts, MTP handling and the absence of mHC/KDA/vision all change; every `glm5_next` monkeypatch must be re-targeted at the `GlmMoeDsa`/`DeepseekV32` modules. One MoE layer forward at 8k fits the 93 GiB constraint with large margin (one layer = 9.0 GiB FP8 / 18 GiB BF16). The binding practical blocker is disk: 51 GB free against ≥1.1 TB needed (704 GiB source + ~360 GiB REAP-50 FP8 master + quant outputs).

### Cited Findings
- Hard-coded Flash assumptions in the pipeline: `transformers.models.glm5_next.modeling_glm5_next.Glm5NextTextExperts` / `Glm5NextTextTopkRouter` / `Glm5NextForConditionalGeneration` monkeypatches in `channel_saliency.py`, `gate_glm_hope_port.py`, `capture_teacher_routing.py`, `drafter_capture.py`, `exp_seqlen_saliency.py`; `hc_mult` collapse in `drafter_capture.py` and `capture_teacher_routing.py`; `KEEP = 144 # 288 experts -> 50%` in `exp_seqlen_saliency.py`; KDA-vs-DSA split logic in `exp_seqlen_saliency.py`; vision bucket logic in `card_addendum.py`, `eval_recompare.py`, `gate_glm_hope_port.py`; `FIXED_GIB = 19.73 # reloaded states (17.5) + resident embeddings/vision (2.23)` — local grep of `/home/patrickd/glm-5.3-reap/scripts/`.
- `num_local_experts` is a single scalar in glm5_next; the 753B config uses `n_routed_experts` (scalar) — [config.json](https://huggingface.co/zai-org/GLM-5.3/raw/main/config.json); nota-ai's non-uniform build needs a patched vLLM that reads `num_experts_per_layer` — [nota-ai card](https://huggingface.co/nota-ai/GLM-5.3-Nota-NVFP4-Global-Pruned-17.75). Same conclusion as the Flash audit: uniform allocation is the only shippable mask — [memory: reap-criterion-and-corpus-audit](/home/patrickd/.claude/projects/-home-patrickd/memory/reap-criterion-and-corpus-audit.md).
- Router semantics identical to Flash: `sigmoid`, `noaux_tc`, `n_group 1`, `norm_topk_prob true`, `routed_scaling_factor 2.5`, `e_score_correction_bias` F32 — [config.json](https://huggingface.co/zai-org/GLM-5.3/raw/main/config.json); so offline router replay and the bias-selects/sigmoid-weights rule carry over — [memory: noaux-tc-router-formula](/home/patrickd/.claude/projects/-home-patrickd/memory/noaux-tc-router-formula.md).
- Calibration memory constraint: "16384-token forward needs >93 GiB — attention is quadratic in sequence length" → calibration capped near 8,192 — [glm-5.3-reap/README.md](/home/patrickd/glm-5.3-reap/README.md) line 74.
- MTP handling in the wild: 0xSero slices "the routers + MTP layer … to match"; pipenetwork drops layer 78; the user's pass-1 dropped MTP and pass-2 preserves it — [0xSero 488B card](https://huggingface.co/0xSero/GLM-5.3-488B-EXL3-TR3-3.42bpw), [pipenetwork](https://huggingface.co/pipenetwork/GLM-5.3-REAP50-MLX-4bit), [memory: glm53-flash-reap](/home/patrickd/.claude/projects/-home-patrickd/memory/glm53-flash-reap.md).
- Disk: `/dev/nvme0n1p1` 936 GB, 838 GB used, **51 GB free**; `~/glm-5.3-reap` 312 GB, `~/models` 212 GB — local `df -h`, `du`.
- FP8 source is lossy vs BF16: "dequantized FP8 weights differ from bf16 by up to 1.6e-2 on values of 0.46 — half an e4m3 step"; FP8 teacher-forced per-layer rel-L2 0.0174 ≈ the 6-bit row — [pipenetwork card](https://huggingface.co/pipenetwork/GLM-5.3-REAP50-MLX-4bit).
- HOPE F-matrix cost on Flash: 47 MiB for 47 layers × 256 experts — [memory: hope-supersedes-reap](/home/patrickd/.claude/projects/-home-patrickd/memory/hope-supersedes-reap.md).

### Inferences
- Concrete delta list: (1) load via native `AutoModelForCausalLM` (`GlmMoeDsaForCausalLM`, transformers ≥5.15) — no `glm5_next_support.py` shim; retarget the experts/router hooks to the DeepSeek-V3-style MoE module (exact class name unverified); (2) `KEEP = 128` of 256, 75 MoE layers (indices 3–77) plus the MTP block's own 256 experts (slice with the same mask or drop); (3) delete `hc_mult`/mHC handling (none), KDA handling (none), vision bucket/corpus/media sources (text-only), `Glm5NextTextIndexer` reference (`dsa_reference.py`) → HF's `GlmMoeDsaIndexer` with the 21-full/57-shared schedule — at 8k calibration the sparse path is live (limit 2051), and shared layers consume the previous full layer's top-k; (4) `FIXED_GIB` shrinks (no vision tower; embeddings 1.77 GiB); (5) HOPE F accumulation = 75 × 256 × 256 × 4 B = 18.75 MiB, trivial; (6) per-layer streaming block size: one MoE layer = 9.66B params = 9.0 GiB FP8 (18.0 GiB if upcast to BF16) + 0.17 GiB attention — so `LAYERS_PER_BLOCK` can stay ≥2 even in BF16 within 93 GiB, and 8k activations (8192 × 6144 × 2 B = 96 MiB per hidden state) are negligible; the indexer score matrix at 8k is ≤ 32 heads × 8192² × 4 B = 8 GiB if fully materialised — fits.
- Pass cost: a full calibration pass streams 704 GiB from NVMe (≈4–6 min of pure I/O at 2–3 GB/s, not the bottleneck); compute scales with expert params touched per pass ≈ 724.8B / ~300B (Flash) ≈ 2.4× → the Flash pass-2 estimate of 38–55 Thor-h becomes ≈90–130 Thor-h per 5.5M-token pass. [extrapolation]
- Disk is a hard gate: FP8 source 704 GiB (+ BF16 1403 GiB if the FP8 double-quant floor is to be avoided) + REAP-50 FP8 master ≈ 19.4 + 337.6 ≈ 357 GiB + 2-bit output ≈ 105 GiB + saliency/router caches. With 51 GB free, nothing starts without an external ≥2 TB NVMe or deleting the 312 GB Flash workspace + 212 GB models dir. Streaming shards from HF per pass would re-download 704 GiB × (number of chunked passes).
- NVFP4 is not a candidate for the experts here (needs 75 % pruning); the quant stage changes from NVFP4 to an EXL3/KT-class ~2.0–2.2 bpw quantizer, which has no verified sm_110 runtime — a new build project on top of the pipeline.

### Gaps
- Exact module names in `modeling_glm_moe_dsa.py` (experts/router) for the monkeypatch — not read in full.
- Whether HF eager `GlmMoeDsa` materialises the full indexer score matrix at 8k — not measured.

---

## Q8. Verdict: (a) / (b) / (c), and is it a better target than DeepSeek-V4.1-Flash-1017 or MiMo-V2.6-Pro?

### Takeaway
**(b) feasible only in the unmeasured region, functionally (c) under the documented ceilings.** REAP-50 fits only at ≈2.0 bpw on experts (protected non-expert) or ≈2.2 bpw with attention/shared/dense also at NVFP4 — both below the ≤2.3 bpw cliff this very model exhibits (+72 % PPL unpruned at 2.30 bpw); every measured-safe bit-width needs 60–70 % deletion. Independently, the A40B active footprint gives a decode ceiling of ≈10 tok/s roofline (≈5 tok/s at Thor's measured efficiency), which the user's own decision procedure rejects at step 1, and the machine has 51 GB of free disk against a 704 GiB source. It is **not** a better target than V4.1-Flash-1017 on prune ratio or decode; its only advantages are no Engram, no pre-FP4 experts, and an MIT-like licence.

### Cited Findings
- User's decision procedure: "(1) decode ceiling = 243.6 / (active × 0.62), ×0.47 for real throughput, reject <20 tok/s ceiling — pruning NEVER improves this; … (4) required prune >50 % → REJECT" — [memory: thor-base-model-shortlist](/home/patrickd/.claude/projects/-home-patrickd/memory/thor-base-model-shortlist.md).
- GLM-5.3 active ≈40B: "each token is routed to 8 of 256 … only ~40B parameters are active" — [0xSero card](https://huggingface.co/0xSero/GLM-5.3-500B-EXL3-3.0bpw); nota-ai says ~42B — [nota-ai card](https://huggingface.co/nota-ai/GLM-5.3-Nota-NVFP4-Global-Pruned-17.75).
- DeepSeek-V4.1-Flash: experts pre-quantized FP4 (`expert_dtype: "fp4"`), required prune 62.9 %, Engram 203.1 GB (39.8 %) with host-table offload that lands in the same unified pool on Thor; REJECTED — [memory: thor-base-model-shortlist](/home/patrickd/.claude/projects/-home-patrickd/memory/thor-base-model-shortlist.md). MiMo-Pro needs 83–85 % pruning — task brief.
- Flash measured decode on Thor: 11.86 tok/s (NVFP4 dense overlay) — [memory: glm53-nvfp4-dense-overlay](/home/patrickd/.claude/projects/-home-patrickd/memory/glm53-nvfp4-dense-overlay.md).
- autotrust on one 128 GB Spark with a 149.7 GiB glm-dsa GGUF: "Works, but slow (low single-digit t/s)" — [autotrust card](https://huggingface.co/autotrust/GLM-5.3-GGUF-DGX-Spark).

### Inferences (shown arithmetic)
- Active params per token (from headers): 8 × 37.75M × 75 = 22.65B routed + 2.83B shared + 12.87B attention + 0.20B indexer + 0.45B dense + 0.95B lm_head ≈ **39.96B**. Decode bytes/token in Scenario A at 2.2 bpw: experts 22.65B × 0.275 = 6.2 GB + non-expert FP8 17.3B × 1 = 17.3 GB + KV ≈ **23.5 GB/tok → 10.2 tok/s roofline at 240 GB/s, ≈4.8 tok/s at the measured 0.47 efficiency**. Scenario B (non-expert NVFP4 0.62): 10.7 + 6.2 ≈ 16.9 GB → 14.2 roofline / ≈6.7 real. The user's step-1 formula: 243.6 / (40 × 0.62) = 9.8 tok/s < 20 → REJECT. Note the attention block (12.87B) is now the dominant per-token read, which is exactly the tensor group the quality ceilings say must stay protected.
- Prune axis: required 67–70 % at 3.0–3.3 bpw (vs V4.1-Flash 62.9 % at its already-spent FP4) — GLM is *worse* than V4.1-Flash on the prune axis at measured-safe bit-widths, and reaches 50 % only by spending the quant lever past its own measured cliff. It is better than MiMo-Pro (83–85 %) on that axis alone.
- Comparison summary: GLM-5.3-753B wins on structure (96 % routed experts, no Engram, no mHC, no KDA, FP8/BF16 source, permissive licence, native transformers + llama.cpp support); loses on decode (A40B vs ~A13B class), on the prune/quant corner (both cliffs at once), on disk (704 GiB source), and on quality evidence (only wikitext/KL, +75 % PPL at REAP-50 before any 2-bit quant).
- What would flip it to GO: (i) an external ≥2 TB NVMe; (ii) a verified sm_110 EXL3/KT-class ≈2.0–2.2 bpw expert kernel in llama.cpp or exllamav3; (iii) accepting ≈5 tok/s and a quality target of "coherent agentic coder at ×1.5–2 PPL" rather than "within 10 %"; (iv) running the user's mass-criterion + HOPE + balanced-corpus REAP-50 on the FP8 source and measuring the stacked cost (≈90–130 Thor-h per calibration pass). None of these is in hand.

### Gaps
- Active-param and bytes/token of DeepSeek-V4.1-Flash-1017 and MiMo-V2.6-Pro-MOPD were not re-derived here (out of scope); the comparison uses the memory's rejection grounds.
- No measurement exists for REAP-50 stacked with ≤2.5 bpw on GLM-5.3; the ×1.6–2.2 PPL expectation is an extrapolation from two single-axis measurements on the same model.
