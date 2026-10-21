# DeepInfer survey (DeepSeek's own V4.1 runner) — 2026-10-21

The "deepinfer" runner exists. Its name is **DeepInfer**.
Repo: https://github.com/deepseek-ai/DeepInfer. Licence: Apache-2.0 (LICENSE file). Released 2026-10-20.
Sources: https://thenextweb.com/news/deepseek-deepinfer-open-source-inference-engine and https://pasqualepillitteri.it/en/news/25886/deepseek-deepinfer-inference-engine-open-source.
I read a shallow clone at HEAD 9bd9cf2 (2026-10-21) in the scratchpad. I ran no code.

## Verified facts (from README, doc/inference_deployment_qa.md, and the source)
- **Scope.** The only supported model is DeepSeek-V4.1-Flash(-1017). The release does offline inference only. The KV-cache storage system, P/D disaggregation and load balancers are not included. The engine is Python + C++ scheduler + TileLang ops, and it was "originally built primarily on vLLM".
- **Hardware.** NVIDIA **SM100** (CUDA 13) and **Huawei Ascend 950** (CANN 9.2). There is no sm_110 or Jetson support. It depends on FlashMLA, DeepGEMM, DeepEP, TileKernels and DeepSelect, all of which are datacenter-only. The benchmarks used 4 GPUs for prefill and 64 for decode. The recommended decode setup is EP64-128 Wide-EP. The smallest *efficient* unit is 64P+64D GPUs.
- **Engram storage.** This is NOT SSD and NOT disk.
  - The tables are sharded row-wise across an `engram_parallel` group, padded to lcm(256, group size) (`v4_modules/engram_parameters.py`).
  - By default they live in **CPU host memory registered for RDMA**. They are fetched through a "DeepEP-V2-Engram" `EngramBuffer`: per-layer GPU receive buffers sized by max_num_tokens × entries/token, using RDMA QPs (`parallel_utils/userbuffer.py: allocate_engram`, `use_cpu_rdma_storage`).
  - `DEEPINFER_ENGRAM_FORCE_GPU_STORAGE=1` puts them in HBM instead, and this is forced on Ascend (`environ.py:147`).
  - The engine runs no NVMe tier, mmap or io_uring of its own.
  - The prefetch-like piece is **ZOS (zero-overhead scheduling)**. N-gram token ids for the *next* step are assembled on the CPU in pinned buffers (`csrc/engine/zos.cpp`). With async scheduling they are patched on the GPU from the previous step's sampled and accepted draft tokens (`tilelang_ops/cuda/gather_engram_tokens_zos.py`). That lets the Engram lookup indices be known before the step's forward runs. An image sentinel pads the n-gram after an image token. A vocab `lookup_table` compresses token ids.
- **KV.**
  - Main compressed KV is FP4 with per-16 E4M3 scales (NVFP4 without the global scale).
  - Indexer KV is MXFP4 (per-32 ue8m0).
  - SWA and main KV are paged MXFP8/FP8 ("V4.1 KV cache is paged FP8", `attention_backend.py:134`).
  - Routed experts are MXFP4.
  - SWA lives in a fixed per-sequence state slot. Only the first 20 layers' SWA is persisted, at 128-token multiples.
- **SWA bounded replay ("YOCO").**
  - Decoder layers recompute only the last N prompt tokens. The SWA window is truncated to the replayed segment, which makes it an *approximate* KV.
  - The default is `YOCO_PREFILL_TRUNCATION_MODE=compact` with `KEEP_TOKENS=140` (window 128 + backtracking ≥ nextn+1). The QA says outright that the exact version would need about 2,560 tokens × 20 decoder layers, and that production uses the 128 approximation. The resulting KV is used for decode only and never enters the prefix cache.
- **Spec decode.**
  - "DSpark": the draft generates 5 tokens (`DEEPINFER_USE_NEXTN_DECODE=5`).
  - A **confidence head**, `sigmoid(cat(hidden, markov_emb) @ W / T)` (`tilelang_ops/dflash.py`), predicts per-token acceptance.
  - Variable-length verification picks the throughput-maximising prefix against a profiled throughput curve. It verifies about 30% fewer tokens (5 → 3.2-4.4) at the same accepted length (about 3.2-3.3), for +4-7% at about 60 sequences per rank.
  - Enable it with `DEEPINFER_ENABLE_MTP_VARLEN=1`.
- **MoE.** EP with EPLB (redundant experts across GPUs). There is no host or SSD expert offload.

## Inference (not verified)
- The RDMA Engram path depends on DeepEP-V2 with IB/RoCE NICs. It has no analogue on Thor, so nothing of its transport is portable. The *idea* carries over: keep Engram off the accelerator and fetch only the indexed rows, using indices known one step early.
- On a single GPU without a NIC, they would presumably use FORCE_GPU_STORAGE. That needs about 197 GB, more than Thor has. Our NVMe O_DIRECT row store has no counterpart in DeepInfer, so the design does not change.
- Our V41_DELTA §2 calls the 128-row replay "an approximation that nothing in this repo does". DeepSeek ships exactly that approximation in production, so its quality cost is presumably acceptable to them. Our exact 2414-row span is the conservative choice. An optional `--yoco-keep 140` mode is legitimate if prefill time matters.

## Comparison with our server (cuda/README.md, docs/V41_DELTA.md)
| topic | DeepInfer | ours |
|---|---|---|
| Engram | host RAM over RDMA, or HBM; ZOS index prep one step ahead | NVMe io_uring O_DIRECT, 1 KiB rows, flag words + cuStreamWaitValue32 in the graph |
| KV | FP4 main / MXFP4 index, paged FP8 SWA | same FP4/E4M3 and E8M0 packing (0.87 GiB @1M); fp32 window rings |
| decoder prefill | approximate replay, 140 tokens | exact replay, 2414 rows |
| experts | MXFP4, EP | Q2_K/Q3_K/Q4_K + MXFP4 (cvt.e2m1x2), 192E REAP |
| spec | DSpark 5 + confidence-varlen | none yet |

## Ranked ideas worth porting
1. **Engram index precompute one step ahead (ZOS idea).**
   - *What:* at decode, the n-gram ids for step t+1 depend only on known tokens plus the token sampled at t. Hash all suffix candidates (or the greedy and top-k tokens) as soon as the sampled id is on the device, and submit io_uring reads before the next forward reaches layer 1. The real dependency is only layer 1's position in the forward.
   - *Where:* `src/engram_store.cpp`, `src/engram_hash.cpp`, the decode loop in `src/v41_decode.cu`.
   - *Benefit:* hides the NVMe latency for L1 behind sampling and layer 0, and for L14 behind layers 0-13.
   - *Risk:* low. Our flag-word design already supports it. A mispredict just falls back to a synchronous read.
2. **Optional approximate decoder replay (`KEEP_TOKENS=140`, compact mode).**
   - *Where:* `forward_layers` dec_span in `src/v41_model.cu` / `src/v41_engine.cu`.
   - *Benefit:* decoder prefill drops from 2414 to 140 rows per layer. That matters at short and medium prompts, and makes prefix-reuse margins simpler.
   - *Risk:* medium. It is not bit-exact against model.py, so put it behind a flag and A/B it with the golden gate plus an eval. DeepSeek's production use suggests the quality cost is small.
3. **Confidence-scheduled variable-length verify (once a DSpark/MTP head is served).**
   - *Where:* a new spec path in `src/v41_decode.cu`. The confidence head is a single small GEMV.
   - *Benefit:* our memory notes say Thor spec decode is bounded by the MoE batch wall. Dropping low-confidence drafts cuts the verify width where the batch curve is steep. DeepSeek reports +4-7% at datacenter scale. On Thor's single stream, the gain is in *not losing* (avoiding 0.95x outcomes).
   - *Risk:* high. It needs the DSpark draft weights and a confidence head in the checkpoint, so check that the 1017 shards contain them.
4. **Image-sentinel n-gram padding and the vocab `lookup_table` compression.** Cross-check our `engram_hash.cpp` against `csrc/engine/zos.cpp: fill_engram_row`, including the pad token (default 2) and the image cut. *Benefit:* correctness on VL inputs. *Risk:* none (it is read-only verification). Our gate only covered `inference/engram.py`.
5. **Persist SWA only for the first 20 layers at 128-token boundaries (prefix cache).** *Where:* `src/v41_engine.cu` prefix reuse. *Benefit:* a cheaper multi-turn reuse design than our dec_margin floor. *Risk:* medium, because it couples with idea 2.

Not portable: DeepEP/RDMA Engram, Wide-EP/EPLB, FlashMLA/DeepGEMM (SM100-only), the Ascend paths.
