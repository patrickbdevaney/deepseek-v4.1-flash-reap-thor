# ARCH_DELTA.md — what ports, what is new, and from where

The directive diffs this model against `gemma-cuda-hybrid` (dense GQA) and
`laguna-s1-cuda-server` (the other MoE precedent), and concludes that MLA, DSA, MXFP4 and
embedded-MTP are "the largest new engineering surface in the project".

**That conclusion is wrong, and the reason matters.** The correct baseline is
**`~/dspark-cuda-reap-finetune`**, which is not — as §13.1 of the directive hedges — a repo that
"may de-risk" the work. It is a **7,870-line pure-CUDA engine for the same 43-layer backbone**
that already runs the full model correctly on this box, with every kernel gated bit-exact or
cosine-1.0 against a PyTorch oracle. The 0731 checkpoint's backbone is byte-identical in
geometry to the one it was built against.

**So the honest framing is: this project is a checkpoint migration plus a decode-efficiency
grind, not a from-scratch kernel build.** The genuinely new surface is the 3-stage embedded
DSpark head and the server layer.

---

## 1. Config diff, 0731-REAP vs the prior 180B-REAP

47 of 48 shared config keys are **identical**, including every architectural constant:
`hidden_size` 4096, `num_hidden_layers` 43, `num_attention_heads` 64, `head_dim` 512,
`q_lora_rank`/`o_lora_rank` 1024, `o_groups` 8, `qk_rope_head_dim` 64, `index_n_heads` 64,
`index_topk` 512, `index_head_dim` 128, `n_routed_experts` 160, `num_experts_per_tok` 6,
`n_shared_experts` 1, `num_hash_layers` 3, `moe_intermediate_size` 2048, `hc_mult` 4,
`hc_sinkhorn_iters` 20, `sliding_window` 128, `scoring_func` sqrtsoftplus, `topk_method`
noaux_tc, `routed_scaling_factor` 1.5, `vocab_size` 129280, `tie_word_embeddings` false,
`quantization_config` (fp8 e4m3, ue8m0 scales, block 128×128), `expert_dtype` fp4,
`max_position_embeddings` 1048576, YaRN `factor` 16 over 65536.

**The only deltas:**

| | 180B-REAP | 0731-REAP |
|---|---|---|
| `compress_ratios` | 43 entries | **46** entries (3 appended zeros — the MTP blocks are pure-sliding) |
| MTP | `mtp.0` only: a *plain* head (`enorm`/`hnorm`/`e_proj`/`h_proj`) | **`mtp.0/1/2`: the full 3-stage DSpark chain**, `main_proj`+`main_norm` on stage 0, `markov_head`(rank 256)+`confidence_head`+`hc_head`+`norm` on stage 2 |
| MTP experts | n/a | **160 per stage — REAP-pruned consistently with the backbone** |
| New config keys | — | `dspark_block_size` 5, `dspark_markov_rank` 256, `dspark_noise_token_id` 128799, `dspark_target_layer_ids` [40,41,42] |
| Provenance | `deepseek-ai/DeepSeek-V4-Flash` (preview) | `deepseek-ai/DeepSeek-V4-Flash-0731` @ `9e165c30` |
| Size | 96.022 GiB / 46 shards | 100.400 GiB / 48 shards |
| `B_tok` | **11.202 GB** | **11.202 GB** (identical — see `ROOFLINE.md` §3) |

**This is the checkpoint the prior project wanted.** It had to load a separate unpruned
256-expert DSpark head alongside the model (+10.12 GiB, mismatched pruning). Here the head is
embedded, pruned in-family, and costs 6.529 GiB inside the 100.4 GiB total.

---

## 2. Port verdict, per subsystem

Legend: **PORT** = existing gated CUDA applies directly · **RETARGET** = existing kernel, new
shapes/wiring · **NEW** = write from scratch.

| Subsystem | gemma-4 / Laguna | `dspark-cuda-reap-finetune` (the real baseline) | 0731-REAP | Verdict |
|---|---|---|---|---|
| MLA attention | none | `kernels/mla_forward.cu`, `mla_attn.cu`, `mla_decode.cu` — gated on real weights, max_rel 3.6e-3 | identical geometry | **PORT** |
| KV compressor (ratio 4 / 128) | none | `kernels/compressor.cu` — Gate K 14/14, both variants gated on real layers 2 & 3 | identical | **PORT** |
| DSA lightning indexer + sparse attn | none | `kernels/indexer.cu` (hadamard, index_score, full top-512 select) + `compressed_attn.cu` — Gate K 19/19 bit-exact | identical | **PORT** |
| Hyper-connections + Sinkhorn | none | `kernels/hc.cu`, `hc_sinkhorn.cu` — Gate K 11/11 | identical | **PORT** |
| MXFP4 expert dequant + GEMM | NVFP4 (E4M3 group-16) | `kernels/moe.cu` `fp4_gemm` walks `KBw = K/32` with **e8m0** scales; `tc_fp4_grouped_gemm_e8m0` / `_gemv_e8m0` are champions | **byte-identical layout**: `I8[2048,2048]` + `F8_E8M0[2048,128]`, block 32 | **PORT** — the directive's "new MXFP4 kernel" premise is void |
| FP8 block GEMM (128×128) | none | `kernels/fp8_block_gemm.cu`, `tc_fp8_gemm.cu` (17.9× champion) | identical | **PORT** |
| MoE router (sqrtsoftplus + noaux_tc + hash layers) | different | `kernels/moe.cu` `moe_router_score`, device-side counting-sort dispatch | identical; **no REAP remap needed** (`gate.weight` already `[160,4096]`, ids dense 0..159, `identity_nearest_all: true`) | **PORT** |
| Loader (48 lazy-mmap shards) | different | `include/safetensors.h::ShardedSafeTensors`, `weight_store.h` (pread → mapped host alloc + `posix_fadvise(DONTNEED)`) | 48 shards vs 46, +`mtp.1/2` keys | **RETARGET** (trivial) |
| KV cache | GQA | fp32, `src/decode.cu:105-108,243` | same | **RETARGET** → move to fp8 (`ROOFLINE.md` §4/§6) |
| CUDA graph decode capture | yes | full 43-layer step captured bit-exact, device-pos, all 3 attention flavours; measured at parity | same | **PORT** |
| **DSpark MTP head** | DFlash (separate draft ckpt) | `dspark_real.cu` + `dspark_attn.cu` implement `main_x` / markov / tap_pool / forward_head and **ran the real 3-stage chain**, but against the *external unpruned 256-expert* head | **embedded, 160-expert, 3-stage** | **RETARGET** — structure known and implemented; rewire to in-checkpoint `mtp.*` tensors |
| Verify / accept loop | DFlash propose-verify | M=K verify + accept-longest-prefix built and gated; determinised MoE lifted acceptance 1.9→2.5 | `dspark_block_size` 5 | **RETARGET** |
| Chat encoding | Jinja | not built | **no Jinja** — Python `encoding/encoding_dsv4.py` + 4 golden test vectors | **NEW** |
| Server (OpenAI API, streaming, tools, prefix cache) | `gemma-cuda-hybrid` full server | `server/`, `include/webui.h` scaffolding only | reasoning_effort low/high/max | **NEW** (port gemma's abstractions) |
| Correctness oracle | — | `ref/` = pure-torch, reimplements all 5 tilelang kernels + Walsh-Hadamard so the verbatim reference `model.py` runs **without `tilelang`/`fast_hadamard_transform`** (neither builds on `sm_110a`) | same reference code | **PORT** — this resolves risk §13.5 outright |

**Score: 11 PORT, 4 RETARGET, 2 NEW.** Gates G1–G7 of the directive are substantially pre-built
and pre-gated. G8/G9 exist and passed on the sibling checkpoint.

---

## 3. Risks from the directive, re-priced

| # | Directive's risk | Actual state |
|---|---|---|
| 1 | "MLA+DSA is the single biggest schedule risk" | **Retired.** Both are built and bit-exact-gated on real weights. Remaining work is efficiency, not correctness. |
| 2 | "DSpark's verify-loop shape is unconfirmed" | **Mostly resolved.** The prior project read the reference `model.py`: the head is a **3-stage chained block-diffusion MTP** (`block_size` 5, `noise_token_id` 128799), taps mean-pooled HC hidden of layers 40–42 — **not** Medusa-tree, **not** DFlash-style. Note: the reference `generate.py` ships **no accept/verify loop at all** — that is ours to implement, and the prior project already built one. |
| 3 | "Speed projection is weakly anchored" | **Inverted — it is strongly anchored.** Identical `B_tok`, same box, already measured. `ROOFLINE.md` §3. |
| 4 | "Memory tight, KV not the problem" | **Confirmed**, and KV is even smaller than assumed. The directive's 37.7 KB/token prior is ~11× too high. |
| 5 | "ARM64/runtime compat never resolved" | **Resolved, and the answer is 'don't need it'.** SGLang does not exist on this box; the reference `inference/` needs `tilelang` + `fast_hadamard_transform` (neither builds on `sm_110a`) + `transformers>=5.0`. The prior project's `ref/` reimplements those kernels in pure torch and runs the verbatim reference `model.py` CPU-only. **That is the oracle** — no ARM64 runtime port required. |
| 6 | "Checkpoint identity matters" | **Verified.** `source_model: deepseek-ai/DeepSeek-V4-Flash-0731`, `source_revision: 9e165c30…`, `tensor_bytes 107803320952`, `tensor_count 45821`, structural validation `status: pass`, 0 failures. Downloading `0xSero/DeepSeek-V4-Flash-0731-REAP` only. |

## 4. Genuinely open

- **Efficiency, not correctness, is the whole game.** 32% → 70–80% of peak bandwidth.
- **DSA verify-step cost** — unmeasured, no precedent anywhere including the prior repo.
- **`c_v(5) = 2.6×` vs the modelled 2.12×** — inherited unexplained.
- **0731 acceptance rate α** — the input the speedup band is most sensitive to.
- **Chat encoding + server layer** — the two real NEW items.
