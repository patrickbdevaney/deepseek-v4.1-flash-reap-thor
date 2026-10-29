# Decode parity: V4.1-1017 REAP-192 (A192) vs the DeepSeek-V4-Flash-0731 engine

Written 2026-10-26. This is a read-only audit. Nothing was run on the GPU and no model was loaded.
Shapes and bytes come from the safetensors headers in `~/models/v41-work/thor/A192` and
`~/models/v41-work/thor/overlay_nvfp4`, read with header-only parsing, and from
`~/models/DeepSeek-V4.1-Flash-1017/inference/config.json`.

Every number carries one of three tags:

- **[M]** measured, with the log or document cited.
- **[C]** derived from code or headers. The arithmetic is shown, or the file:line is cited.
- **[E]** estimate or extrapolation. Treat it as a hypothesis to measure.

"230 GB/s" is the streaming read measured by `cuda/tools/gemv_format_bench.cu`: 229.7 GB/s
(`ops/logs/s08_gemv_bench.log:1`).

---

## 0. Bottom line

1. **The two models read almost the same bytes per token.** V4.1 A192 reads 11.45 GB per token and
   the 0731 reads 11.20 GB [C]. At 230 GB/s that puts the AR wall at 20.1 and 20.5 tok/s. The
   architectures differ, but the byte budgets landed in the same place:
   - **V4.1 reads more:** wider MLA projections (dim 5120), the head, the shared expert, and Engram.
   - **V4.1 reads less:** cheaper experts (2.83 bpw K-quant against MXFP4 at 4.25 bpw), and a much
     smaller compressor and indexer (82 MB against 801 MB).
2. **The 8.10 tok/s today is 40% of the 230 GB/s roofline (92.7 GB/s)** [M]
   (`ops/logs/s08_decode_A192.log`). The 0731's best AR, 13.83 tok/s, ran at 155 GB/s, which is 67% [M].
   The gap is kernel efficiency, not algorithm. The one big structural difference is the format the
   experts are served in.
3. **AR parity with the 0731 is about 14.3 tok/s [E].** That is the 0731's measured per-region
   bandwidths applied to V4.1's bytes. Adding the already-built NVFP4 dense overlay lowers B_tok to
   8.59 GB (wall 26.8 tok/s at 230), and AR projects to about 18-19 tok/s [E].
4. **About 26 tok/s is reachable only through speculation [E].** V4.1's k-invariant share is 73.8%,
   against 69.2% on the 0731, so the 2x mechanism the 0731 used still applies. It needs three things:
   (a) AR at about 14, (b) a verify path that does not re-read weights at M = K+1, and (c) a draft
   head with τ of about 3.6-3.9 on A192's hidden states, which needs a fine-tune like the 0731's
   s3recap run.
5. **Top three levers, in order:**
   1. Rewrite the K-quant expert GEMV. It runs at 27-37% of streaming against 66-85% for FP8 and
      MXFP4. Worth about -22 ms/token [E].
   2. Fix the latency-bound glue the 0731 already fixed for itself: a single-CTA router, sparse
      attention on 8 CTAs over 640 mostly-masked entries, and no side streams. Worth about
      -10 to -20 ms [E].
   3. Fix the verify path at M=K+1 (M = 5..8 re-reads every dense weight) and fine-tune the head.
      This is the ×1.5-2.0 step.

## 1. Corrections to the brief (verified)

| brief says | the files say |
|---|---|
| "MXFP4 hardware e2m1x2 unpack" is the served path | **The Thor build A192 is thor-v1 arm A K-quants.** `thor.json` `alloc` puts **Q3_K** on layers 0,1,2,8,14,20,37,38,39, Q2_K on gate/up with **Q3_K** on down for layers 28, 29 and 36, and **Q2_K** on the other 28 layers; `expert_bpw` 2.828. The headers agree: `w1.qweight U8 [192,2304,1680]` is 1680 B per 5120 weights, which is 20 × 84 B Q2_K superblocks, and the Q3_K layers have 2200 B per row. The engine prints `experts thor-v1 arm A` (`s08_decode_A192.log`). The MXFP4 hardware path (`kernels/v41/v41_gemv.cuh:116-146`) is used only by the drafter's native `mtp.*` experts (I8 + E8M0). MXFP4 at 192 experts would be 192 × 40 × 18.80 MB = 144 GB, which does not fit. |
| V4.1 has NO native MTP head | **It has one.** `config.json` sets `n_mtp_layers: 3`, `dspark_block_size: 5`, `dspark_n_routed_experts: 128`, `dspark_n_activated_experts: 3` and `dspark_target_layer_ids: [37,38,39]`. A192 ships `mtp-0/1/2.safetensors` (7.93 GB = 7.39 GiB; 128 MXFP4 experts per block, `main_proj [5120,15360]`, `markov_head` and a `confidence_head`). `src/v41_draft.cu` wires it as the drafter (`load_drafter`, `:126`). The "sidecar" is these blocks, or an s11 head directory that replaces them. |
| `cuda/README.md` "nothing has run on the GPU" | Stale. s08 decode ran 2026-10-25: 8.09-8.10 tok/s over 3 runs, deterministic, consistency PASS, graph on, seqmax 8192, prompt 16 tokens, 104 decoded [M] (`ops/logs/s08_decode_A192.log`; `OWNER_FOLLOWUP` 23:51). |

## 2. Architecture diff (config + headers + code)

| | 0731-REAP | V4.1-1017 A192 |
|---|---|---|
| layers, dim | 43, 4096 | 40, 5120 (`v41_config.h:30-31`) |
| attention | MLA, 1 KV head, 64×512, q_lora 1024, o_lora 1024 ×8 groups | the same family: 64×512 (rope 64), q_lora **1280**, o_lora 1024 ×8 |
| MLA shapes | `wq_a [1024,4096]`, `wq_b [32768,1024]`, `wkv [512,4096]`, `wo_a [8192,4096]`, `wo_b [4096,8192]`; FP8 128×128 | `wq_a [1280,5120]`, `wq_b [32768,1280]`, `wkv [512,5120]`, `wo_a [8192,4096]`, `wo_b [5120,8192]`; FP8 E4M3 with E8M0 scale per **32×32** |
| KV compression | compressor on 41 layers (ratio 4 ×21, ratio 128 ×20); DSA indexer 64 heads, top-512, Hadamard | **CSA2, shared.** Ratios 0,0 \| 2 ×18 \| 1 ×20. Compressor only on kv-sources {2,8,14,20}; indexer (32 heads ×128, top-512, no Hadamard) only on {2,8,14,20,24,28,32,36}. Other layers reuse the owner's caches and top-k (`v41_config.h:79-97`). L20 picks 2048 candidate blocks of 8. |
| compressed caches | fp8-sim | fp4 codes + E4M3/16 (KV) and E8M0/32 (index keys), packed exactly: 288 B and 68 B per row |
| hyper-connections | hc_mult 4, Sinkhorn | hc_mult 4, 20 Sinkhorn iterations, **deferred pre-mix** (`docs/V41_DELTA.md` §4); fn `[24, 20480]` fp32 ×2 per layer |
| routed experts | 160 (from 256), top-6, MXFP4, inter 2048 | **192** (from 384), top-6, inter **2304**, Q2_K/Q3_K (2.83 bpw), sqrtsoftplus router; the bias selects, ×1.5 |
| shared expert | 1, FP8, inter 2048 | 1, FP8 32×32, inter 2304 |
| dense FFN layers | none (hash-routed early layers) | none (no hash routing) |
| Engram | — | L1 and L14: 24 n-gram rows (fp8 256 + 8 E8M0 = 264 B) per token per layer from NVMe; `engram.wkv [25600,6144]` FP8 on GPU |
| vocab / head | 129,280, bf16 `[129280,4096]`, untied | 129,280, bf16 `[129280,5120]` = 1.32 GB, untied |
| MTP / DSpark | 3 blocks, block 5, taps 40/41/42 | 3 blocks, block 5, taps 37/38/39, 128 experts top-3 (unpruned, MXFP4) |
| stored | 100.4 GiB (85.7 GiB experts) | 98.58 GiB resident without mtp/vision (89.48 GiB experts) [M: engine log] + 7.39 GiB drafter |

## 3. B_tok, component by component (M=1, bytes read per token)

The V4.1 column is header-summed over all 40 layer files plus globals. The routed column is
6/192 of the stored expert bytes. The total reproduces the 98.58 GiB resident size the engine
prints [C]. The 0731 column is `ROOFLINE.md` §2 (`:114` onward).

| component | stored/served dtype (V4.1) | V4.1 MB | % | 0731 MB | % |
|---|---|---:|---:|---:|---:|
| MLA projections (`wq_a`+`wq_b`+`wkv`+`wo_a`+`wo_b`) | FP8 E4M3 + E8M0/32² | **5069.6** | 44.3 | 4599.5 | 41.1 |
| routed experts (top-6) | Q2_K / Q3_K (ggml, bit-exact decode) | **3002.6** | 26.2 | 3449.3 (MXFP4) | 30.8 |
| shared expert | FP8 | 1417.0 | 12.4 | 1082.2 | 9.7 |
| head | bf16 | 1323.8 | 11.6 | 1059.1 | 9.5 |
| Engram `wkv` (L1, L14) | FP8 | 315.0 | 2.8 | — | — |
| HC `fn` | fp32 | 157.3 | 1.4 | 135.3 | 1.2 |
| router | bf16 | 78.7 | 0.7 | 75.0 | 0.7 |
| indexer (8 layers) | FP8 `wq_b` + bf16 | 45.1 | 0.4 | 275.4 | 2.5 |
| compressor (4 layers) | bf16 | 36.7 | 0.3 | 525.7 | 4.7 |
| norms, embedding row | bf16 | 1.0 | 0.0 | 1.0 | 0.0 |
| **B_tok** | | **11,446.8** | | **11,202.4** | |

**Arithmetic.**

- **MLA, per layer.** `wq_a` 1280·5120 = 6.554 MB. `wq_b` 32768·1280 = 41.94 MB. `wkv` 512·5120 =
  2.62 MB. `wo_a` 8192·4096 = 33.55 MB. `wo_b` 5120·8192 = 41.94 MB. The E8M0 scales add N·K/1024 =
  0.124 MB. Total 126.74 MB, ×40 = 5069.6 MB. The 0731 is about 107 MB per layer ×43.
- **Routed, per expert.** Q2_K: 3·2304·5120·84/256 = 11.61 MB. Q3_K: 3·2304·5120·110/256 = 15.21 MB.
  The mixed layers: 7.74 + 5.07 = 12.81 MB.
  - Per token: 6·(28·11.61 + 9·15.21 + 3·12.81) = 6·500.4 = **3002.6 MB**.
  - The 0731: 3·2048·4096·0.53125 = 13.37 MB, ×6 ×43 = 3449 MB.
- **Shared expert.** 3·2304·5120 + scales = 35.42 MB, ×40.
- **Head.** 129280·5120·2 B = 1323.8 MB.
- **Engram.** 2 × (25600·6144 + 0.15 MB scales) = 315.0 MB.
- **HC.** 2·24·20480·4 B = 3.93 MB, ×40 = 157.3 MB.
- **Compressor.** 3 × (wkv + wgate) bf16 2·512·5120·2 B = 31.5 MB. L20 has wkv only, 5.2 MB.
  Total 36.7 MB.

**Context traffic (not in B_tok) [C].**

- Window rings are fp32: 128 rows × 2 KB, read twice per layer by `k_sparse_attn`, about 20 MB per
  token.
- Compressed rows: 512 × 288 B per compressed layer = 5.6 MB per token.
- Index keys grow with n·68 B on the 4 sources. At 128K that is about 27 MB per token, about 0.2%.
- Bytes are not the context problem; the selection kernels are (§6).

**With the NVFP4 dense overlay** (`overlay_nvfp4`). The overlay replaces every FP8 dense weight
except the routed experts and Engram with E2M1 + E4M3/16 + an fp32 global scale, 0.5625 B per
weight against 1.001.

- MLA drops to 2848.8 MB, the shared expert to 796.3 MB, and the indexer to 26.7 MB.
- **B_tok = 8,587.0 MB (-25.0%)**. The routed share rises to 35.0%; the head is 15.4%.

**AR rooflines at 230 GB/s [C].**

| | B_tok | wall | measured best |
|---|---|---|---|
| 0731 | 11.20 GB | 48.7 ms = 20.5 tok/s | 13.83 [M] (67%) |
| V4.1 A192 | 11.45 GB | 49.8 ms = 20.1 tok/s | 8.10 [M] (40%) |
| V4.1 A192 + overlay | 8.59 GB | 37.3 ms = 26.8 tok/s | — |
| V4.1 A192 + overlay + FP8 head (needs a quality gate) | 7.93 GB | 34.5 ms = 29.0 tok/s | — |

## 4. Where the 123.5 ms goes today [E, no profile exists]

The V4.1 tree has no phase marks. `dprof` is not used by any `v41_*` file. This attribution uses
`gemv_format_bench` rates and kernel geometry. Its sum, about 100-120 ms, brackets the measured
123.5 ms.

| region | bytes | basis | est. ms |
|---|---:|---|---:|
| routed experts | 3003 MB | bench at M=1: Q2_K 83.9 / 67.0 GB/s, Q3_K 77.6 / 63.2 (gate\|up / down) | **~40** |
| FP8 dense (MLA, shared, Engram, indexer) | 6842 MB | FP8 bench 188-195 GB/s; in situ assumed 160-190 | 36-43 |
| head (bf16) | 1324 MB | at about 190 GB/s | ~7 |
| router | 79 MB | `k_router` is one CTA per token: 16 warps of scalar bf16 loads over 1.97 MB, then a serial top-6 on thread 0 (`kernels/v41/v41_moe.cu:23-64`) | 3-12 |
| `k_sparse_attn` ×38 | ~25 MB | grid `(M, 8)` = **8 CTAs** on 20 SMs (`v41_attn.cu:358`), 80 tiles × 2 passes × 2 barriers, always 640 entries (`v41_model.cu:436`) | 6-12 |
| HC (`partial` + single-thread `finish`) ×80 | 157 MB | `v41_dense.cu:278-326` | 2-3 |
| other glue | — | about 50 kernels per layer, ~1500 small launches in the graph | 4-6 |

**The point.** Routed experts are 26% of the bytes but an estimated 32% of the time. The router
and sparse attention are each a candidate for 5-10% of the step. Instrument before ranking these
two; that is the 0731's own rule (`OPTIMIZATION_LOG.md`, "Instrument policy").

## 5. Kernel parity: each 0731 measured win against V4.1

| 0731 lever (measured win) | V4.1 status | file:line | V4.1 expected gain |
|---|---|---|---|
| **half2 / hardware-cvt MoE dequant**, Opt #9: MoE GEMV 121.5 → 314.6 GB/s, 2.59× (`OPTIMIZATION_LOG.md:16`) | **MISSING for the served format.** The K-quant decode is scalar fp32 with deliberately unfused `__fmul_rn`/`__fsub_rn` for bit-exactness with numpy, then an fp32 FMA against x. That is compute-bound: 27.5-36.5% of streaming. The MXFP4 hardware path exists but serves only the drafter. | `v41_gemv.cuh:186-260`, `:116-146`; bench `s08_gemv_bench.log` | **#1. ~-20 to -25 ms [E]** (40 → ~17.7 ms at 170 GB/s) |
| MoE occupancy, Opt #1: 1.107× | HAS: 256-thread CTAs, grid (N/64, units) | `v41_moe.cu:100`, `:248-253` | — |
| HC warp-parallel Sinkhorn + block-per-output mixes, Opt #3: 1.072×, -6.1 ms | PARTIAL: the partial dots are 200-CTA parallel, but `k_hc_finish` does 24 reductions + 20 Sinkhorn iterations on **one thread** per token (`if (threadIdx.x != 0) return`) | `v41_dense.cu:292-319` (`:295`) | 0.5-1 ms [E] |
| BF16-native head, Opt #4: 1.034× | HAS (`DecBF16`, native bf16 loads) | `v41_gemv.cuh:97-112` | — |
| ILP ≥ 2 in the GEMV, Opt #7: 1.034× | HAS: two `uint4` per lane per chunk; FP8 reaches 85% of streaming | `v41_gemv.cuh:78-95` | — |
| "read B once" multi-row (F40/42/43/64) | HAS for M ≤ 4 (`MT` ∈ {1,2,4}; MoE units of ≤ 4 rows). **GAP at M = 5..8: `mtile` = 2 re-reads every dense weight** | `v41_dense.cu:92-105`, `v41_moe.cu:67` | verify / drafter only (§7) |
| smem-staged fp8 **tensor-core** tile + KC staging (F41, F74 +6.1%) | **MISSING.** No mma path. The 0731 found the GEMV loses to the tile at every M ≥ 2 (`LOOP_LOG.md` F129). | — | verify only |
| managed device-preferred weights (F44) | HAS | `v41_model.cu:74`, `:91-93` | — |
| full-step CUDA graph (F44: 92.5 → 79.3 ms, 1.17×) | HAS, including the Engram `cuStreamWaitValue32` waits; one verify graph per M | `v41_engine.cu:129-152`, `:165`, `:173`; `v41_model.cu:460-469` | — |
| arena / no per-call malloc (F44, F83) | HAS | `v41_model.cu:292-340` | — |
| device-side draft loop, Opt #5 | HAS, graph-captured | `v41_draft.cu:319-339`; `v41_engine.cu:165` | — |
| **shared expert on a side stream** (F56: -2.78 ms) | **MISSING.** One stream, and `moe()` runs routed then shared serially. V4.1's shared expert is 1417 MB against 1082 MB on the 0731. | `v41_engine.cu:261`; `v41_model.cu:452-453` | -2.5 to -3.5 ms [E] |
| kv chain forked off the q chain (F57: +0.7%) | **MISSING.** `wkv` (N=512, split-K) runs serially after `wq_a` | `v41_model.cu:365-372` | -0.5 to -1 ms [E]; on the 8 index layers the indexer chain is also forkable |
| compressor-emit fork (F56; 0.81 ms per forward) | MISSING, but there are only 4 kv-source layers | `v41_model.cu:376-388` | ~0.2 ms; skip |
| `jetson_clocks` (F60: +3.0% steady, +20.7% cold) | Not referenced in `cuda/CHAIN_STEPS.sh` | — | +3% [E, environmental] |
| router / grouping parallelised (F73 `moe:group` -20%; 0731 router+group = 3.99 ms) | **MISSING.** `k_router` is 1 CTA per token (`<<<M,512>>>`) with a serial top-k; `k_units` is `<<<1,1>>>` | `v41_moe.cu:23-64`, `:63`, `:70-94`, `:301` | -3 to -10 ms [E]: run the gate through `bf16_linear` (split-K to 40 CTAs) plus a one-warp select |
| `sparse_attn` smem staging + vectorised gather (ladder 1.7: -4.2 ms) | PARTIAL: rows are staged, but the gathers are byte-wise; only 8 CTAs at M=1; all 512 compressed slots are iterated even when masked | `v41_attn.cu:281-302`, `:358`; `v41_model.cu:436` | -4 to -9 ms [E] |
| `index_score` warp-per-output (F71: +4.4%) | HAS (a 32-key tile, 256 threads), on 8 layers only | `v41_attn.cu:88-138` | — |
| radix top-k replacing a selection sort (ladder 1.2: context slope -65%) | HAS, one CTA per row | `v41_attn.cu:155-223` | context term only |
| adaptive verify width (F49: +2.1%; +9-11% where it engages) | **MISSING.** K is fixed. The server parses `--adaptk` but `v41_engine.cu` never reads `cfg.adaptK`. The checkpoint's `mtp.2.confidence_head` is unused. | `v41_engine.cu:354`, `:383-387`; `server/server.cpp:203` | spec only, +2-10% [E] |
| block 6 → 5 (ladder 2.1) | HAS (`DSPARK_BLOCK = 5`) | `v41_config.h:126` | — |
| draft-head fine-tune (s3recap: τ 2.89 → 3.84; 22.15 → 28.4 tok/s; `FINAL_STATE.md`) | **MISSING.** There is no s11 head in `~/models/v41-work`. The native head was trained on the unpruned 384-expert model. | — | the ×1.3 on spec (§7) |
| NVFP4 dense overlay (F127: +5.5% AR; F129: +0% on spec; `LOOP_LOG.md:7586-7700`) | **The overlay is built, but there is no loader or decoder in V4.1.** `DSV4_NVFP4_OVERLAY` is read only by the dead 0731 sources (`src/decode.cu:235`, `src/engine.cu:268`). | `CHAIN_STEPS.sh:106-108` | see the trap in §8; -12 to -17 ms AR at parity [E] |
| KV pack, Hadamard staging, `gemm_fp32` emit tile, uint2 misalignment loads (1b.2, 1.10, 1.12, F72) | n/a. V4.1 caches are already fp4-packed; it has no Hadamard; the compressor is on 4 layers; thor-v1 tensors are padded and aligned | — | — |

## 6. V4.1-specific needs the 0731 never had

1. **Expert format.** The fit forces about 2.83 bpw, so experts cannot be MXFP4. At 2.6-3.4 bpw a
   GEMV must decode about 600-700 Gweights/s to stay bandwidth-bound at 200 GB/s. Today it decodes
   147-256 [M].
   - The fix is the llama.cpp `mmvq` shape: either quantise x to q8 per 32 with a precomputed sum,
     or keep it fp16, then take dp4a/HFMA2 dot products over the 2/3-bit planes and finish with
     `d·sc·Σqx − dmin·m·Σx` per 16-weight sub-block.
   - That costs about 0.5-2 instructions per weight against about 8-10 now.
   - It gives up bit-exactness with the numpy K-quant decode (`FORMAT_PROPOSAL.md` §1). Gate it on
     tolerance, as the 0731 did for Opt #9.
   - It must also be bit-identical between M=1 and M=K. The 0731 learned this in F127/F128.
   - llama.cpp's own CUDA path already serves these exact bytes with q8_1 activations, so the
     GGUF evaluations have already covered that numeric regime.
2. **Engram.** The `wkv` GEMV is 315 MB per token (2.8%), about 1.7-1.9 ms [E]. Each token issues
   48 O_DIRECT reads (1.13-1.30 ms p50, 1.35-1.89 ms p95 [M], `notes/engram_off_device_thor.md:34`,
   `:47`).
   - The reads are submitted before the graph launch and overlap with layer 0. The L1 block waits
     on a flag (`v41_engine.cu:87-91`, `v41_model.cu:460-461`).
   - At parity, layer 0 takes about 1.7 ms, so p50 is hidden. At the overlay's speed (about
     1.3 ms per layer) up to about 0.6 ms of p95 would be exposed [E].
   - `stage_rows` synchronises the stream before hashing and submitting (`v41_engine.cu:82`).
     That is inherent for AR, since the ids come from the previous argmax.
3. **CSA2 sparse attention.** 38 layers attend 128 window rows plus 512 compressed rows.
   - At the measured context (≤ 120 tokens), at most (pos+1)/2 ≤ 60 compressed slots are valid. The
     kernel still walks 640 entries, so about 80% of `k_sparse_attn` work is on masked rows [C].
   - Fixes:
     - Bound the loop by the valid count, which is cheap and bit-exact.
     - Split the entries across more CTAs. The final max is exact, so this matches the two-pass
       reference.
4. **The indexer score grid scales with `seqmax`, not context** (`v41_attn.cu:141-143`).
   - At `seqmax` 1M: 32,768 CTAs per index layer × 8 layers per token, mostly writing `-inf`, plus
     4 MB of score writes per index layer [C]. That costs several ms per token [E].
   - At the server default `seqmax` 8192 (`server/server.cpp:180`) it is negligible.
   - The single-CTA top-k (`:204-223`) and L20's single-CTA candidate select (`:239-266`) are
     ms-scale at 64K-1M entries [E].
   - This is the context term. It is unmeasured; repeat the 0731's context sweep (ladder 0.3).
5. **HC at dim 5120.** 157 MB of fp32 `fn` per token, plus 80 single-thread Sinkhorn finishes.
6. **Head.** 25% larger than the 0731's. The drafter reads it at 5 rows, and as coded that is 2
   m-tiles, so twice (§7).
7. **Memory for spec.** The resident build is 98.58 GiB and the native drafter adds 7.39 GiB.
   - The chain's fit guard computes `du(A192)` + 12. That is 106 + 12 = 118 GiB against
     MemAvailable 118 GiB (`ops/logs/chain.log:952`), with zero margin (`CHAIN_STEPS.sh:116-118`) [C].
   - An in-model REAP drafter with K-quant experts (the s11 `inmodel` arm, `v41_draft.h:22-25`)
     would cut both memory and draft bytes.

## 7. Speculative decoding

**Why the 0731's 2× worked.** 69.2% of B_tok is k-invariant, read once per verify
(`ROOFLINE.md:223`). Routing is correlated: at K=5 the verify touched 17.53 distinct experts of 30
slots, against 27.8 if routing were independent (`DECODE_LADDER.md` §stop). The verify kernels
ended up near the byte model (c_v 1.82 against 1.84, `OPTIMIZATION_LOG.md` session 08-06).

- Measured cycle: 28.4 tok/s at τ 3.84 gives 135 ms per round, against a 72.3 ms AR step. That is
  1.87 AR steps per round and **2.05×** [M] (`FINAL_STATE.md`, `wiki/kernel-optimisations.md:7`).

**V4.1, from the byte model.** Union = 192·(1 − (1 − 6/192)^m).

| verify rows m | union (independent) | c_v A192 indep. / corr. 0.63 | c_v A192+overlay indep. / corr. |
|---:|---:|---|---|
| 4 | 22.9 | 1.74 / 1.37 | 1.99 / 1.49 |
| 5 | 28.2 | 1.97 / 1.51 | 2.29 / 1.69 |
| 6 (K=5) | 33.3 | 2.19 / 1.66 | 2.59 / 1.87 |

Three things follow:

- The k-invariant share is **73.8% on A192** and **65.0% with the overlay**. Speculation is
  structurally a little better on A192 than on the 0731, and worse once the overlay is on, which is
  what the 0731 measured in F129.
- The 0.63 correlation factor is borrowed from the 0731 [E]. V4.1's REAP-192 routing correlation is
  unmeasured.
- K-quant expert time grows with rows per expert, because the decoder is compute-bound. At M=4 the
  bench drops to 52.5 / 40.7 GB/s on Q2_K, against 83.9 / 67.0 at M=1 [M]. The routed term
  therefore scales by rows, not by union, until §6.1 is fixed.

**As coded, the verify will be far worse than this model predicts.** For M = 5..8, `launch_dense`
uses `mt` = 4, so `mtile` = 2 and every FP8 and bf16 dense weight is read twice
(`v41_dense.cu:92-94`, `:105`). That is 2 × 8.44 GB of k-invariant bytes, so c_v(6) ≥ 2.9 [C/E], and
speculation would be about 1.0-1.3× even at τ 3.8.

**Fixes:**

- Add an `MT` = 8 instantiation. At K ≤ 5120, fp32 x staging is 160 KB. For `wo_b`'s K=8192, stage
  e4m3 codes instead of fp32.
- Or add an mma tile, which is what the 0731 settled on.
- Apply the same fix to the head in `head_rows` and in the drafter, which is 5 rows at
  `v41_draft.cu:327`.

**Draft cost per round [C].**

| part | MB |
|---|---:|
| 3 blocks × (MLA 126.7 + shared 35.4 + HC 3.9 + router 1.3) | 501 |
| routed: 128-expert MXFP4, top-3 over 5 rows, union 14.3 × 18.80 MB × 3 blocks | 807 (508 if correlated) |
| main head: once if `MT` ≥ 5, twice as coded | 1324 or 2648 |
| Markov head: 5 × 66.2 MB | 331 |
| **total** | **2.96-4.29 GB = 0.26-0.37 AR steps** |

The tap hook also adds `main_proj` + 3 × `wkv` = 86.5 MB to **every** main forward once a drafter
is attached (`v41_draft.cu:246-276`). For comparison, the 0731 draft was about 16% of a round.

**The head.** The native head was trained on the unpruned bf16/fp4 384-expert model. A192 is REAP-50
plus requantised to 2.83 bpw. Every local quant comparison found hidden-state drift (cos 0.89-0.93,
memory note `glm53-quant-hidden-state-drift`), so expect native τ below the 0731's untuned 2.89-3.12
[E].

- Measure it with `CHAIN_STEPS.sh` `spec_sweep`.
- Then fine-tune on A192 taps, as the 0731 did (+33% τ).
- Wire `confidence_head` for an adaptive K.

**Projection [E].** Spec tok/s = AR × τ / cycle, with a cycle of about 1.9-2.2 AR steps once the
verify is fixed.

| AR | τ 3.0 | τ 3.8 |
|---|---|---|
| 14.3 (parity) | 19.5-22.6 | 24.7-28.6 |
| 18.7 (overlay; cycle about 2.2-2.5) | 22-25 | 28-32 |

The 0731 measured the overlay at **0% on spec** (F129), so the 18.7 row is the less certain one.

## 8. Traps found during the audit

1. **The overlay label is a lie today.** `CHAIN_STEPS.sh:106-108` labels runs `_overlay` when
   `DSV4_NVFP4_OVERLAY` is set. The V4.1 engine never reads that variable; only the unbuilt 0731
   `src/decode.cu:235` and `src/engine.cu:268` do. An "overlay" decode or server run would measure
   the FP8 base under the overlay's name. That is the orphaned-arm trap again.
   - **Fix:** wire the overlay, and have the engine print the served dense format, then assert it in
     the chain.
2. **Bit-exactness and spec identity.** Any new expert or dense kernel must route M=1 and M=K
   identically in the same change (0731 F127/F128). `launch_dense` already fixes split-K
   independently of M ≤ 8 for exactly this reason (`v41_dense.cu:95-104`). Keep that property in the
   rewrites.
3. **`--adaptk` is silently ignored** by the V4.1 engine (§5).

## 9. Ranked plan

Baseline: 123.5 ms = **8.10 tok/s [M]**.

| # | step | basis | est. ms/token after | est. tok/s |
|---|---|---|---:|---:|
| 0 | Add per-phase marks to `forward_layers`/`block` (`include/dprof.h` is already in the tree); confirm §4 | 0731 instrument policy | 123.5 | 8.1 [M] |
| 1 | K-quant expert GEMV rewrite (§6.1) to ≥ 170 GB/s on Q2_K/Q3_K; tolerance gate + M=1/M=K identity | bench 63-84 → target 170; 0731 Opt #9 got 2.59× on the same kind of rewrite | ~101 | **~9.9 [E]** |
| 2 | Latency-bound glue: router via GEMV + parallel select; `sparse_attn` bounded by the valid count + more CTAs + vectorised gather; warp-parallel `hc_finish`; parallel `k_units` | §4/§5 geometry; 0731 F71/F73/ladder 1.7 | ~88-95 | **10.5-11.4 [E]** |
| 3 | Side streams (shared ∥ routed, kv ∥ q, indexer ∥ q) + `jetson_clocks` | 0731: +3.1% (3 forks) + 3.0% (clocks) [M on 0731] | ~83-89 | **11.2-12.0 [E]** |
| 4 | Bring dense FP8/bf16 to the 0731's in-situ rates (attention 172 GB/s, MoE 195, head 183; `wiki/roofline-why-the-needle-wont-move.md:41-45`) + fuse norm/act_quant/rope glue | the 0731 region table applied to V4.1 bytes: MoE 4420/195 = 22.7, attention 5151/172 = 29.9, head 7.2, glue 7.9, Engram 2.0 | **~69.7** | **~14.3 [E] — AR parity** |
| 5 | NVFP4 overlay: loader + `DecNVFP4` (hardware e2m1x2 + E4M3/16 + fp32 global) in the existing GEMV engine; route M=1 and M=K together; fix the env trap | -2.86 GB/token at parity rates: -12.9 ms MLA, -3.2 ms shared | ~53.5 | **~18.7 [E]**; 0731 measured only +5.5% with `wo_a` unrouted |
| 6 | Verify path: `MT` = 8 or an mma tile for dense + head at M = 5..8; multi-row K-quant units; adaptive K | §7: removes the 2× k-invariant re-read | — | enables step 7 |
| 7 | Spec with the native head; measure τ and c_v at 128 and 8K (`spec_sweep`) | — | — | **~17-23 [E]** (τ unknown) |
| 8 | Fine-tune the head on A192 taps (+ a REAP'd K-quant drafter to cut 7.4 GiB and the 807 MB per round of routed draft bytes) | 0731 s3recap: τ +33% [M on 0731] | — | **~25-29 at AR 14.3; ~28-32 with the overlay (if F129 does not repeat) [E]** |

**Order note.** Steps 1-4 are lossless engineering on the critical path for every later number.
Step 5 has an owner-adopted quality gate (`OWNER_FOLLOWUP` 2026-10-24: d_kl32 +0.018). Steps 6-8 are
where the "about 26" lives. Rank by `bytes × (1 − efficiency)`, not by deficit alone (the 0731
lesson from Opt #2). On that measure the experts (3.0 GB at about 35%) and MLA (5.1 GB at an
assumed 80%) dominate.

---

## 10. Results after the plan ran (2026-10-29) [M]

Every row below was measured with `ops/bench/ab_decode.sh` on the A192 build with the NVFP4 dense
overlay (printing-press prompt, 256 tokens, 3 repeats). Every row passed the consistency gate. Commit
hashes refer to this repository.

| step | tok/s | exactness | commit |
|---|---:|---|---|
| audit baseline | 8.09 | — | — |
| exact rewrites (router, sparse attn, hc_finish, side-stream shared, MT8) | 10.91 | bitwise | 4627951 |
| half2 Q2_K/Q3_K experts | 12.60 | gated tolerance | 4627951 |
| NVFP4 dense overlay | 14.02 | accuracy-gated | 4627951 |
| split sparse attention, M ≤ 8 | 15.94 | bitwise | 62d3f27 |
| coalesced bf16 head + NVFP4 2-row loads | 17.59 | bitwise | 088198b |
| block-scan top-k | 18.43 | bitwise | 06679db |
| planar K-quant experts, Q3 4 blocks/SM | 18.71 | bitwise | bffc364 |
| verify-width fixes (spec only); AR re-measured | 19.07 | bitwise | 96819b5 |
| dense 8 lanes/row where K/32 underfills the warp | 19.82 | T1 reassociation | cab548a |
| decode == prefill (M-independent split-K), kq 16-lane down pass | 20.16 | T1, then bitwise decode/prefill | 4e0478c |
| fp8 tensor-core Q2_K/Q3_K experts (`v41_kq_mma.cuh`); 8-token prefill units | 21.15 | T1 (KL to exact 0.018 vs half2 0.021) | 335135c |
| hc_mixes on its own stream beside collapse/norm/attention and MoE; hc_partial x reuse; rmsnorm loads batched | 21.71 (paired: 21.07 → 21.71, +3.0%) | bitwise (teacher-forced logits identical) | 63a9a45 |
| shared expert forked before the router; attention KV branch on its own stream (per-stream split-K scratch) | **22.08** (paired: 21.61 → 22.08, +2.2%) | bitwise | 8cfe5b7 |
| NVFP4 dense on fp8 mma, every M (`v41_nv_mma.cuh`) | 21.98 AR (paired −0.8%); **spec K=3 25.93** (1.19x); 8k prefill 80.0 | T1 (KL to exact 0.0173) | b667022 |
| k_nvm sums its split-K partials in the last CTA (no `k_ksum` launch) | **22.28** (paired vs 8cfe5b7: 22.27 → 22.28, parity) | bitwise vs b667022 | 5295ecd |
| two-stage greedy head: NVFP4 shortlist + exact bf16 rescore (`head2_argmax`) | **24.52** (paired: 22.24 → 24.52, +10.3%) | greedy ids exact (T0e: recall by measured error, Δ = 9σ) | c5c5ff2 |
| hc_collapse + rmsnorm + act_quant fused (`k_cnq`); coalesced router + fused top-k; rmsnorm weight prefetch | **24.55** (paired: 24.44 → 24.55, +0.5%) | bitwise | c5575a6 |
| prefill chunks up to 512 rows (window ring WINDOW + 512, `k_units_big`) | 8k prefill 79.4 → 88.0 | bitwise (ids identical at every chunk) | dd5cc13 |
| sparse attention on bf16 tensor cores, every M (`v41_sa_mma.cuh`) | 8k prefill **88 → 160 tok/s**; 8k-context decode 21.4 → 22.6 | T1 (KL to exact 0.0173 → 0.0124) | (this commit) |

The audit's AR wall for these bytes is 20.1 tok/s, computed with the head counted at bf16 and the
FP8 dense read in full. With the overlay the step reads 8.57 GB, so the wall is 37.6 ms, or 26.6 tok/s
(`docs/zenith/LEDGER.md`). Today's step is 47.3 ms, which is **80% of roofline**. The best of our
other servers, the 0731, reached 67%.

**Speculative decoding (DSpark, 128 context).** A K=2 round cost 171 ms before this work and costs
about 112 ms now:

- The draft fell from 24 ms to 12 ms. The head and the Markov chain now run over K slots only, and
  the draft uses an NVFP4 copy of the head.
- The verify fell from 146 ms to about 100 ms. The fused multi-row dense dot fixed the dense
  layers, and the dual kq launch keeps 1-row expert units off the MT=4 kernel.

K=2 now runs at 1.017x plain decode. At 8k context with highly predictable input it reaches 1.05x at
K=3. The verify's routed experts are the remaining wall. Three rows hit about 2.6x the unique experts
of one row, so the verify reads 2.6x the expert bytes, and no exact kernel can avoid that. Spec decode
therefore pays only when acceptance is high.

**Exactness contract.** Split-K is now chosen from the weight shape alone, so a decode step, a
verify row and a prefill row sum every row in the same order. `--consistency 32` therefore expects
zero mismatches, near-ties included, and any mismatch is a bug. Before this change the split depended
on M. Prefill rows then summed in a different order from decode, and that rounding noise flipped MoE
routing often enough to produce logit gaps of up to 1.47 (found in commit 4e0478c's investigation).

**Tensor-core experts (2026-10-29).** `k_kq` was issue-bound. ncu measured about 4.5 thread
instructions per weight, with the expert pass streaming at 164 GB/s. `k_kqm` exploits two facts:

- A K-quant code read as an e4m3 byte is the subnormal c·2^-9.
- Expert inputs are already e4m3.

So the masked code bytes feed `mma.m16n8k16.e4m3` directly. One mma covers 16 rows, one 16-weight
sub-block and up to 8 tokens. Every product is exact and the sum is fp32.

| | result |
|---|---|
| Speed, `gate_kq_fast`, all four Q2/Q3 shapes | about 200 GB/s (half2 SOA: 110-148) |
| AR decode | 20.16 → 21.15-21.31 tok/s |
| 8k prefill | 48.7 → 61.5 tok/s; 63.8 with 8-token units (an mma tile holds 8 tokens, so expert weights are read half as often) |
| Decode at 8k context | 17.81 → 18.71 tok/s |
| Spec at 128 context | K=2 1.068x, K=3 1.043x; identity PASS |

Greedy ids changed (T1). Teacher-forced over the exact path's 99-token output, KL to the exact fp32
decode (`V41_KQ_FAST=0`) is 0.018 for the mma kernel and 0.021 for half2. The tool is
`ops/bench/score_arms.sh` with `score_cmp.py`. Its arithmetic per output element does not depend on
how many tokens share a unit. The consistency gate (33 positions, 0 mismatches) and an ids check
(4-token vs 8-token prefill units, identical) both confirm it.

The new ids reference is `ops/logs/bench/mma2_arm0.json`. `V41_KQ_MMA=0` restores the SOA half2
kernel, and `V41_KQ_MT8=0` restores 4-token prefill units.

**Glue (2026-10-29).** The block defers the pre-mix: each collapse uses the previous mix's `pre`. So
a sub-layer's `hc_mixes` is first needed by `hc_post`, and the `hc_partial` + `hc_finish` pair (about
18 µs, 80 per step) now runs on `State::hcs` beside the collapse, the norm and the attention or MoE.
`V41_HC_OVERLAP=0` turns this off. The result is bitwise identical, at +3.0% in a paired binary A/B
(`ops/bench/ab_bin.sh`, A/B alternated over 3 rounds). One attempt was rejected: running Sinkhorn on
one lane in registers made `hc_finish` go from 6 to 30 µs, because 640 serial IEEE divides cost more
than the shuffles they replace.

Two other findings from the same profile:

- **Not real gaps:**
  - Engram layer 1 sat idle for 1.07 ms under nsys. Without the profiler its rows land
    1.08 ms after submit (`V41_ENGRAM_STATS=1`), about when layer 1 needs them.
  - Shared-expert NVFP4 launches run at 87 GB/s in the trace only because they overlap the routed
    experts on the side stream. The same shape in `bench_dense` runs at 203 GB/s.
- **A trap:** single runs swing ±0.5 tok/s between loads, so judge sub-5% levers by paired A/B only.

**NVFP4 dense on tensor cores (2026-10-29).** An e2m1 nibble rewritten as the e4m3 byte
`sign<<7 | magnitude<<2` is exactly the e2m1 value × 2^-6, subnormals included. So the overlay's
codes feed `mma.m16n8k16.e4m3` against the act-quantized x after about 3 bit ops per 4 weights.
`k_nvm` serves every M with the same per-element arithmetic, and its K split is chosen from the
shape, so decode, verify and prefill still match bit for bit.

Speed, by row count:

- **M=3 (the verify width):** about 1.1–1.4x in `gate_nvm` (L2-cold, rotating weight copies).
- **M=8:** 2.4–4.5x.
- **Engine:**

  | | before | after |
  |---|---|---|
  | plain decode | 22.16 tok/s | 21.98 (paired, −0.8%) |
  | spec K=3 | | 25.93 tok/s (1.19x) |
  | 8k prefill | 63.8 tok/s | 80.0 tok/s |
  | 8k decode | 18.71 tok/s | 19.09 tok/s |

**Layout trap.** A lane reading 64 contiguous bytes means each 16-byte load instruction strides
lanes 64 bytes apart and half-fills every 32-byte sector. That capped DRAM reads at about 118 GB/s
on every shape. The fix was to interleave the layout so each instruction reads 512 contiguous bytes.
The first gate timed L2-hot weights and hid the problem completely, so `gate_nvm` now rotates more
than 100 MB of copies.

New ids reference: `ops/logs/bench/nvm2_arm0.json`. `V41_NV_MMA=0` restores `k_dense<DecNVFP4>`.

**Two-stage greedy head (2026-10-29).** The bf16 head is 1.32 GB a token, which is 5.7 ms at roofline.
Greedy decode needs only the argmax, so the head now runs in two stages:

1. Score every row with an NVFP4 copy of the head (0.37 GiB, which the drafter now shares).
2. Recompute, with the bf16 weights, every row within Δ = 4 logits of the approximate maximum. The
   recompute (`k_h2_rescore`) reruns k_dense's M=1 chunk order, fmaf chain and warp_sum, so each
   value is bit for bit the logit the full head writes.

If more than 64 rows qualify, the host recomputes that row with the full head.

**Why Δ = 4 is safe.** The true winner can be missed only if the approximation errors at it and at
the approximate winner add up to more than Δ. The probe (`V41_HEAD_PROBE=1` on the forced
continuation) measured:

- **Rank of the exact argmax** in the NVFP4 order: 0 at 97 of 99 positions, 1 at the other 2.
- **Error near the top:** rms 0.30 logits, max 1.17 over 442 rows.

So Δ is about 9σ of that pair.

**Measured.**

- **Held-out prompts:** greedy ids identical to the full head on 27/27 (128 tokens each, plain and
  spec K=3), with 0 fallbacks.
- **Paired decode:** 22.24 → 24.52 tok/s.

Sampling (temperature > 0) and the debug and scoring paths keep the full head.

**Spec by domain (held-out, K=3).** On top of the 24.1 tok/s plain decode:

| domain | spec K=3 |
|---|---|
| math | 1.60x (38.6 tok/s) |
| code | 1.43x |
| agentic | 1.21x |
| long | 1.15x |
| chat | 0.98x |

The speed prompt is a chat prompt, so it is the drafter's worst case. S5 (adaptive K) and D1–D4
(drafter fine-tune) should report per-domain numbers.

**Plain decode is now at the byte floor (2026-10-29).** The nsys ledger at 24.5 tok/s (`ops/logs/prof/h2a.sqlite`):

| family | ms/token | MB/token |
|---|---|---|
| NVFP4 dense | 20.8 | 3672 |
| K-quant experts | 15.4 | 3003 |
| glue | 7.3 | 200 |
| sparse attention | 2.2 | 25 |
| NVFP4 head | 1.6 | 372 |

The kernel times add up to more than the 40.8 ms wall because the side streams overlap them. At
228 GB/s, the 8.6 GB a token reads takes 39.2 ms, so decode runs at ~96% of the byte floor.

Glue work that shortens launch chains but moves no bytes therefore buys little:

- **Coalesced router** (28 µs a token to a fraction of that): paired change within noise.
- **Fused collapse/norm/quant** (three launches into one): +0.5% paired.

The remaining AR levers move bytes, and those are quality-tier (T2) decisions. Speed beyond that
comes from speculation.

**Prefill (2026-10-29).** Two changes:

- **Chunk size:** prefill took 64-row chunks and re-read every expert it touched once per chunk.
  With chunks up to 512 rows, 8k prefill went 79 → 88 tok/s, then plateaued.
- **Sparse attention on tensor cores:** the nsys trace showed attention at 53% of the prefill.
  Each row is MQA (64 heads against one 512-wide KV row per entry), so it runs as two bf16 GEMMs
  on `mma.m16n8k16`. Every product is exact, because q, the KV rows and bf16(p) are all bf16
  values; only the summation order changes. The softmax keeps the reference's two passes. Per-row
  instructions are identical at every M, so decode == prefill is still bitwise.

| | before | after |
|---|---|---|
| 8k prefill | 88 tok/s | 160 tok/s |
| decode at 8k context | 21.4 tok/s | 22.6 tok/s |
| KL to exact | 0.0173 | 0.0124 |

`V41_SA_MMA=0` restores the fp32 kernels. New ids reference: `ops/logs/bench/sam1_arm0.json`.

