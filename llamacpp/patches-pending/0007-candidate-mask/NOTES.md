# Patch 0007: DeepSeek-V4.1 candidate-block mask (CPU development notes)

`0007-deepseek41-candidate-block-mask.patch` goes on top of 0006 (`git am` checked on a clean worktree of 92031e6a4). It was built and tested CPU-only, with `GGML_CUDA=OFF`, 4 build jobs and tiny synthetic tensors. No GPU, model or GGUF was touched.

## 1. What model.py does

Sources: `/home/patrickd/models/DeepSeek-V4.1-Flash-1017/inference/model.py` and `kernel.py`.

- **Roles (l.502-503).** In `Indexer.__init__`:
  - `is_candidate_source = layer_id == candidate_source_layer`, which is layer 20.
  - `uses_candidates = 0 <= candidate_source_layer < layer_id`.
  - Only index sources own an `Indexer` (`Attention.__init__`). So the mask is *consumed* only by the index sources after 20, which are **24, 28, 32 and 36**.
  - Layers 21-23, 25-27 and so on run no indexer. They reuse `shared_attn.topk_idxs` (l.723-737; the reuse is l.727), so they inherit their source's restriction. This is what patch 0002 implements.
  - The phrase "layers 21-39 restrict their attention" (in 0006) holds only in that inherited sense.
- **Index score (l.555-562).** `index_score[s,t] = sum_h relu(q_h · k_t) * w_h` over `t < end_pos // ratio`.
- **Causal edge (l.564-568).**
  - Prefill (`start_pos == 0`) uses `compress_lens = (arange(1, seqlen+1) // ratio)`, and every `t >= compress_lens` is set to -inf.
  - Decode uses `compress_lens = end_pos // ratio`, and the key slice already ends there.
  - At ratio 1 a query at position p therefore sees rows `0..p` (its own row included).
- **Level one, at the source (l.570-573).** `shared_attn.candidates = select_candidate_blocks(index_score, compress_lens, 2048, 8)`. The source's own top-k is **not** masked; only the `elif` at l.574 masks.
- **`select_candidate_blocks` (l.584-611):**
  - **Blocks.** Rows are grouped into blocks `[8b, 8b+8)` (aligned at row 0). The tail is padded with -inf (l.599).
  - **Block score.** Each block scores its maximum row (l.600).
  - **Pin.** The block `last = (compress_lens - 1) // 8` gets +inf (l.605-606). It holds the newest row and is only partly filled. If `compress_lens == 0`, then `last = -1` and nothing is pinned.
  - **Top-k.** `topk(min(2048, num_blocks))` (l.608) picks the blocks; ties are left to torch, which does not specify their order.
  - **Keep.** Only picks whose value is `> -inf` are kept (l.610), so unreachable blocks are never kept.
  - **Expand.** The block mask is repeated back to rows (l.611).
  - **Below the threshold.** With at most 2048 blocks, every reachable block is kept, so the mask is the causal mask again and changes nothing.
- **Level two (l.574-576).** `index_score.masked_fill(~candidates, -inf)` is applied with the layer's **own** q and weights. Then `topk = min(512, end_pos // ratio)`, indices are sorted, and `idx >= compress_lens` becomes -1 (l.579-581).
- **The sliding window (`get_window_topk_idxs`, `_window_kv`) is untouched by the mask.** In `Attention.forward` it is simply concatenated with the compressed picks.
- **`sparse_attn` (kernel.py l.311-402).** A softmax over the picked rows plus `exp(attn_sink - max)` in the denominator (l.383).
- **Shared state.** `SharedAttentionRuntime.candidates` (l.1169-1179) holds one slot, written by layer 20 and read by every later index source in the same forward.

## 2. What was implemented

Files changed:

| file | change |
|---|---|
| `src/models/deepseek41-candidates.h` (new) | the mask as a ggml subgraph; header-only and free of llama.cpp dependencies, so the CPU test builds the exact same code |
| `src/models/deepseek41.cpp` | `build_indexer_top_k`: build the mask at the candidate source; apply it at later index sources. Loader reads and validates the KVs |
| `src/models/models.h` | `published_candidates` on the graph object, next to 0002's `published_top_k` |
| `src/llama-hparams.h`, `src/llama-arch.{h,cpp}` | `deepseek41.candidate.{source_layer,topk_blocks,block_size}` (uint32, optional) |
| `src/llama-context.cpp` | the 0006 clamp applies only when the file lacks those keys |
| `conversion/deepseek.py`, `gguf-py/gguf/constants.py` | the converter writes the keys from `config.json` |

**The graph (no new op).** Every op used already has a CPU and a CUDA kernel. `GGML_CUDA_USE_CUB` is defined for CUDA ≥ 11.7, so `top_k` over more than 1024 blocks stays on the GPU.

- **Block score.** `ggml_pool_1d(score + causal mask, MAX, k=8, s=8)`.
  - An all -inf block comes out as -FLT_MAX, because pool_1d starts from -FLT_MAX on CPU and CUDA alike.
  - That still ranks below every reachable block. Its rows are -inf in every consumer, so keeping it changes nothing.
  - This is the only place where the port differs from l.610 (`values > -inf`). The check counts these blocks as `unreachable_kept`; they are never reachable.
- **Pin.**
  - Visibility is monotone (`j < n_visible`), so a block is visible iff its first row is. Taking the first row of each block from the plan's causal mask and applying exp gives 1 or 0.
  - `vis[b] - vis[b+1]` (built with pad, view and sub) is 1 exactly at block `(n_visible - 1) / 8`.
  - Scaled by 1e30 and added to the block score, it ranks that block above every other block. This uses +1e30 rather than +inf to avoid inf arithmetic; only the kept set matters.
  - A query that sees nothing pins nothing.
  - No new graph input is needed: the per-query `n_visible` already lives in the plan's mask, for prefill, chunked prefill, decode and multiple streams alike.
- **Top-k.** `ggml_top_k(block_score, 2048)`, then `fill(-inf)` and `set_rows(0)` give the keep mask F32 `[n_blocks, n_q, 1, n_stream]`. This is the same scatter `build_top_k_mask` uses.
- **Consumer.** The consumer's score is reshaped to `[8, n_blocks, n_q, n_stream]` and the block mask is added as `[1, n_blocks, ...]` (a broadcast add). The tensor carried from layer 20 to 36 is block-level, 1/8 the size of the score.
- **Short-context identity.** When `n_comp / 8 <= 2048` (n_comp is the plan's padded row count), nothing is built and the graph is exactly the pre-0007 graph. This holds for every context of at most 16384 compressed rows.

**Keys and the clamp.**

- **Load-time checks.** The loader rejects a file where:
  - the source is not an index source;
  - a later index source's ratio differs from the source's (the reference would fail to broadcast there);
  - the block size does not divide 256 (the plans and caches pad rows to multiples of 256).
- **Clamp.** With the keys present, `n_ctx_seq` is **not** clamped.
- **`LLAMA_DSV41_UNSAFE_LONG_CTX` is kept.** It still lifts the clamp on a file *without* the keys, which is useful for measuring the unmasked divergence. It has no effect when the mask is present.
  - I kept it rather than removing it because GGUFs produced before 0007 lack the keys.
- **Files that predate the keys** get the mask without reconversion:
  `--override-kv deepseek41.candidate.source_layer=int:20 --override-kv deepseek41.candidate.topk_blocks=int:2048 --override-kv deepseek41.candidate.block_size=int:8`.
  Checked in code: `llama_model_loader::get_key` applies an override even when the key is absent, and the INT override type is accepted for uint32.

## 3. Test results (CPU)

**Files** (all in this folder):

- `test-dsv41-candidates.cpp`, built by `build_test.sh` against the clone's CPU libggml;
- `check_candidates.py`, the NumPy transcription;
- logs `test-cpu.log`, `check-numpy.log` and `check-numpy-mutation-nopin.log`;
- dumps in `dump/` (5.3 MB).

**Dimensions:**

- indexer: 8 heads × 32;
- attention: 4 heads × 64;
- block_size 8, topk_blocks 4 (threshold 32 rows), index_topk 6;
- ratio 1.

**15 cases:**

| group | cases |
|---|---|
| prefill from 0 | n = 20, 32, 33, 100, 300, plus n = 20 with the cache padded to 256 rows (mask built but must be a no-op) |
| decode | p = 19, 31, 32, 99, 299, plus 300 in a 1024-row cache |
| chunked ubatch | 16 queries at 280..295 |
| two streams | decode, and 8-query chunks at different lengths |

**Old graph vs new graph (`test-cpu.log`):**

- Every case with ≤ 32 visible rows produces **bit-identical** attention output in the new and old graphs. That is 6 cases, including the padded case where the mask is built and must be a no-op.
- The **forced** mask (built even below the threshold) is also bit-identical there.
- Above the threshold the outputs differ, as they must: max |Δ| is 0.73 to 3.5.
- `prefill_n33` happens to pick no dropped row and stays identical.

**NumPy reference (`check-numpy.log`), float64, transcribed from model.py:**

- Selected block ids are identical in all 15 cases and every query (all blocks holding a reachable row).
- The reader's top-k sets are identical in all 15 cases.
- The worst attention max abs error is **4.46e-7**.
- The worst level-one index score max abs error is **1.32e-7**.
- The smallest gap at the top-k boundary was 4.7e-5 (prefill_n300). There were no exact ties.

**Transcription vs model.py itself.** The NumPy `select_candidate_blocks` was compared with model.py's own function on every case. That function was extracted with `ast` and run in CPU torch with `CUDA_VISIBLE_DEVICES=""`. It agrees on all 15 cases (`model.py_select_equal=1`).

**Mutation check.** With the pin removed, the checker fails 9 of 15 cases (attention error up to 3.3), so it does tell a right mask from a wrong one.

**Full build.** The whole `libllama` builds CPU-only with no warnings (`build-llama-cpu.log`), and the converter edits pass `py_compile`. The `build_indexer_top_k` wiring itself (about 15 lines) could not be run without a model. The test builds the same helper calls on a score and mask of the same shape, `[n_comp, n_q, 1, n_stream]`, which I traced through the permutes there.

## 4. Divergence from the CUDA server

`cuda/kernels/indexer.cu` is the V4 DSA indexer and has no candidate logic. The V4.1 code is in `cuda/kernels/v41/v41_attn.cu`: `k_index_scores` l.86-138, `k_block_max` l.228-237 and `k_cand_select` l.239-266. It is wired in `cuda/src/v41_model.cu` l.203 and l.413-417. It matches model.py on roles, blocks, the pin (+inf at `(n-1)/8`), the `nb <= 2048` keep-all shortcut and the `> -inf` filter. It differs as follows (model.py is authoritative):

1. **Ties at the 2048 boundary.**
   - The CUDA server keeps the *lowest-index* tied blocks (l.262).
   - torch.topk leaves the order unspecified.
   - ggml's CPU `top_k` is `std::partial_sort`, also unspecified, and ggml CUDA uses CUB.
   - All three can pick different blocks on an exact tie.
2. **Score precision.**
   - The CUDA server rounds the index score to bf16 (`bf16r`, l.134) and fp4-simulates q and k, as the reference's bf16 einsum and `fp4_act_quant` do.
   - The fork computes the score in F32 with no fp4 simulation, and this was already true before 0007.
   - bf16 scores make exact block-max ties *likely* at long context, so the reference and the CUDA server can resolve ties differently from each other and from llama.cpp. Near-ties can also flip.
   - This is the main residual source of selection differences, and it is not a defect of the mask.
3. **Unreachable blocks.** The CUDA server writes 0 (not kept) for them; 0007 may mark some as kept. This makes no difference, because those rows are -inf in every consumer.

## 5. GPU validation still to run (on the real GGUF; not done here)

1. **KLD at long context.** Run `tools/v41-teacher-kld.cpp` against the cached teacher on one or more rows of 32K tokens or more. Compare three runs:
   - 0007 with the keys present;
   - the same file without them and `LLAMA_DSV41_UNSAFE_LONG_CTX=1` (the unmasked baseline);
   - a context of 16K or less (a regression check, which should be byte-identical to the 0006 build).
   - Pass condition: KL beyond 16K close to KL below 16K, and the unmasked run clearly worse.
2. **The CUDA backend path.** All ops are existing CUDA kernels and no new op was added, but on Thor confirm:
   - `test-backend-ops` coverage for POOL_1D, TOP_K (ne0 > 1024, CUB), SET_ROWS, FILL, PAD and EXP at these shapes;
   - with `GGML_SCHED` debug, that no op of the subgraph falls back to the CPU (a split would copy `[n_comp, n_ubatch]` scores to the host).
   - The CPU test can be repointed at the CUDA backend for a direct check.
3. **Memory and speed at long context.**
   - The mask adds one full-size `[n_comp, n_ubatch]` F32 add at each of 24/28/32/36, and about 1/8-size tensors at 20. At 128K context and ubatch 512 the full-size add is about 256 MB per tensor, on top of the existing indexer scores.
   - Check the compute buffer that graph reserve reports. The reserve plan uses the full `kv_size`, so the mask is in the worst-case graph.
   - `pool_1d` on the CPU is single-threaded. That only matters for CPU-only runs.
4. **Pipeline integration.** GGUFs already produced by the running pipeline lack the `deepseek41.candidate.*` keys.
   - Either reconvert, or serve with the three `--override-kv` flags above.
   - `llama-quantize` copies KVs, so levels made from a 0007-converted source inherit the keys.
   - `s09_gguf_ladder.py verify` could assert the keys are present.
   - The DSpark sidecar converter (`DeepseekV41DSparkModel`) calls `DeepseekV41Model.set_gguf_parameters` and so also writes the keys under its own arch prefix. They are harmless and unused there.

## 6. Assumptions I could not verify

- **Tie and near-tie behaviour on real data** (see §4). With bf16-valued scores in the reference, exact block-max ties at rank 2048 may be common. What the teacher's torch topk does on them is unknown, so selection can never be shown bit-exact against the teacher. Only KLD can show whether it matters.
- **The fork's index score.** I assumed it is otherwise right (F32, no fp4 simulation, Hadamard `k_rot` applied to q and k). 0007 only adds the mask on top of it.
- **The compressed rows.** I assumed every comp plan row `j` is compressed position `j` of its stream, in order, starting at 0 per stream, and that the plan's `kq_mask` is exactly `j < (pos+1)/ratio`. I read this from `dsv4_build_comp_plan` (`n_visible`, `cache_off + pos/ratio`) and `dsv4_set_kq_mask`, but did not test it end to end.
- **The reserve graph.** I assumed the reserve plan's mask, never filled, only sizes the buffers.
- **The key values.** I assumed the real KVs will be (20, 2048, 8) as in `config.json`. Other values are handled generally, but only block sizes dividing 256 are accepted.
- **Wiring vs the test graph.** The `build_indexer_top_k` and loader wiring compiles but has not been run. The helper it calls is what the tests run.
