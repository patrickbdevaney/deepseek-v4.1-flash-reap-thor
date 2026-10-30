# llamacpp/: the community GGUF path for the V4.1-1017 REAP (s09)

This is a lean llama.cpp fork that serves two purposes:

- **Independent reference** for the CUDA server (`cuda/`).
- **Consumer of the GGUF ladder** that the community uses.

The fork is upstream llama.cpp at the **head of PR #28696** (DeepSeek-V4.1), which is still open and unmerged, plus four patches. `PINNED` records the shas.

```
setup.sh      clone upstream, fetch pull/28696/head, check out 2076412598e9…, git am patches/  (branch v41-thor)
build.sh      CUDA build, sm_110 (Thor) by default, CUDA_ARCH=<sm> for another GPU; -j4 by default; compile only
patches/      0001 HC tensors F32 · 0002 reader layers reuse the index source's top-k · 0003 Engram O_DIRECT row store · 0004 REAP heal gain
convert_v41_reap.py    REAP-aware converter (wraps the fork's convert_hf_to_gguf.py, no edits to it)
ENGRAM_ROWFILE.md      the packed row-file format, shared with the CUDA server
tools/v41-teacher-kld.cpp   paired top-32 KL / top-1 / NLL against the cached teacher
CHAIN_STEPS.sh         every s09 step the chain runs (build, rowfile, backend_ops, convert, imatrix, quant, drop_source,
                       stage, kld, smoke, unstage, manifest)
MODEL_CARD_TEMPLATE.md rendered by ops/s09_card.py from artifacts only
tests/test_s09.py      CPU tests: keep-set, converter checks + Engram view + V4.1 template, row file, planner, imatrix
                       keep-set, kld prep split, card (`.venv/bin/python llamacpp/tests/test_s09.py`, no pytest needed)
```

The stage code lives in `ops/`:

- `s09_keyset.py`: master keep-set checks on top of `ops/lib/keepset.py`, the pipeline's single fingerprint scheme.
- `s09_engram_rowfile.py`: builds and verifies the row files.
- `s09_gguf_ladder.py`: plan, imatrix, quant, split and verify.
- `s09_ef.py`: optional GPTQ-style error feedback.
- `s09_teacher_eval.py`: KL prep and summary.
- `s09_card.py`: renders a per-level detail card into `$V41_WORK/s09/<level>/README.md`. The published HF card is s12's `ops/cards/gguf.md.tmpl`.

## Patches on top of PR #28696

1. **HC tensors stay F32** (`src/llama-quant.cpp`). The hyper-connection mixers feed a Sinkhorn normalisation that runs 80 times per token, and the reference keeps them in fp32. `llama-quantize` now forces `*.hc_*` and `output_hc_*` to F32, whatever the source type or any `--tensor-type` rule.

2. **Reader layers reuse their index source's top-k** (`src/models/deepseek41.cpp`). The reference has 8 index sources (2, 8, 14, 20, 24, 28, 32, 36). The layers in between read the top-k their source published, through `shared_attn.topk_idxs`. The PR built no top-k on those layers, so 30 of the 38 compressed layers attended to *every* compressed row. In this config, layers 2-19 compress at ratio 2 and layers 20-39 at ratio 1, so every reader shares its source's ratio. That is wrong numerics, and it is also why their cost grew with context.
   - The fix publishes each source's top-k on the graph object and reuses it when the ratios match. A ratio mismatch can't happen, because load already rejects any reader whose ratio differs from its KV source.
   - The fix also makes `n_kv_max` sparse on those layers.

3. **Engram O_DIRECT row store** (`src/models/deepseek41-engram-rowstore.h`). Set `LLAMA_ENGRAM_ROWFILE_DIR=<dir>` and the model opens `engram_L1.v41rows` and `engram_L14.v41rows`. It checks their headers, row width and bucket coverage, then replaces the `get_rows` on the 100 GB table with an F32 graph input.
   - `set_input` fills that input with 16-thread aligned `pread`s (`O_DIRECT`).
   - Values are decoded as FP8 × E8M0 and rounded to bf16, exactly as `ParallelEngramEmbedding` does.
   - The GGUF then needs no `engram_embd` tensors; if they are present they are skipped.
   - Without the variable, the PR's lazy mmap path is unchanged.

4. **REAP heal gain on the routed sum** (`src/models/deepseek41.cpp`, KV plumbing in `llama-arch`, `llama-hparams.h`, `llama-model-loader.cpp`). An optional f32 array KV `deepseek41.reap.routed_gain`, one entry per decoder layer. When present, layer `il`'s routed-MoE output is multiplied by `gain[il]` (`ggml_scale`) before the shared expert is added; the shared expert is not scaled.
   - This is the same gain the NVFP4 and Thor builds fold into every routed expert's `w2` (`ops/s07_quant.py`): the reference computes `y = sum_k w_k * w2_k(...) + shared(x)`, so a gain on every routed `w2` is a gain on the routed weighted sum.
   - The MXFP4 master cannot carry it (an E8M0 scale moves only by powers of two), so the GGUF carries it as metadata instead and every level inherits it through `llama-quantize`.
   - Absent key: graph unchanged. Load rejects a wrong length and any non-positive or non-finite gain.

The REAP-aware converter is `convert_v41_reap.py`. It refuses to convert unless:

- `<master>/reap_keepset.json` exists (s05 writes it with `keepset.save`);
- its keep-set hashes to its own stored `sha256` and to `--expect-keep-sha` (default in CHAIN_STEPS: the sha of `$V41_WORK/s04/keepset.json`);
- every layer keeps exactly `n_routed_experts` (192) of 384;
- the index holds experts 0..191 for w1, w2 and w3 and nothing above;
- every `gate.weight`, `gate.bias` and `bias_vl` has 192 rows.

All of these checks are fatal; none is a warning. `KeepsetMismatch` subclasses `SystemExit`, so no `except Exception` can swallow it. The output is a *lossless* source GGUF:

- experts are bit-exact MXFP4;
- FP8 dense weights are exact BF16 (the PR writes Q8_0 here, which would make every Q6_K level a double quantization);
- Engram is dropped.

The fingerprint, source expert count, kept ids and teacher revision go into `deepseek41.reap.*`.

Two more things the wrapper does on top of the PR:

- **Engram shards 47/48.** s05 does not rehost them, but the master's index still maps every `layers.{1,14}.engram.*` tensor to them, and the PR opens every shard the index names (`FileNotFoundError` otherwise). `--engram-src <base checkpoint>` builds a symlink view (CHAIN_STEPS: `$V41_WORK/s09/src/<NAME>`) of the master plus the shards it lacks. A borrowed shard must hold exactly the tensors the index maps to it, all `.engram.`. Nothing is written into the master. `embed` is then dropped; `wkv`, `q_weight` and `k_weight` are converted (the fork requires them).
- **Chat template.** The PR picks the template from the directory *name*, so `master_REAP192` would get the V4 one, and V4.1 tool calls would come back as plain content. The wrapper always writes `models/templates/deepseek-ai-DeepSeek-V4.1-Flash.jinja`, and `s09_gguf_ladder.py` refuses a source, level or split whose template is not exactly that file.

`--heal-gains PATH` (JSON `{"0": g, ..., "39": g}`, or `{"keepset_sha256", "gains": {...}}`) writes `deepseek41.reap.routed_gain` only if some gain differs from 1.0 (patch 4 applies it). CHAIN_STEPS `convert` derives the file with `ops/s07_quant.py heal_decision()`, the same rule as the NVFP4/Thor builds, and fails if `s10/heal/decision.json` is missing; arm B ships unhealed, as `s07 thor-b` does. `quant` verifies that every level keeps the source's gain, and `manifest` records it as `routed_gain` for s12's card.

### Contracts with the other stages

- **Keep-set:** the fingerprint is `ops/lib/keepset.fingerprint`, the same one every stage uses. Kept experts are renumbered `0..n-1` in ascending source-id order (`keepset.remap`). The imatrix remap relies on that.
- **s12:** `manifest` writes `$V41_WORK/s09/manifest.json` = `{"files": [{"path", "sha256", "bytes", "level"}], "dense_precision", "keepset_sha256", "n_routed_experts", "levels", "llama_cpp", ...}`, plus `"repo"` if `S09_REPO` is set.
  - Only levels whose splits re-verify are listed, and all of them must carry the same keep-set.
  - s12 uploads the files flat by name; the split names carry the level, so the names are unique.
- **Arm:** `V41_NKEEP` (default 192) names the files and the source GGUF, so arm B (128) gets its own paths.

## The ladder (`ops/s09_gguf_ladder.py`)

Each level is one `llama-quantize` run from the same lossless source. No level is ever made from another level.

| level | routed experts | protected layers | est. size* |
|---|---|---|---|
| L0 | MXFP4, copied bit-exact | MXFP4 | ~145 GiB |
| L1 | IQ3_XXS / IQ3_S mix, 3.40 bpw | MXFP4 | ~118 GiB |
| L2 (flagship, 128 GB devices) | IQ2_XS..IQ3_S mix (uniform scores: mostly IQ2_S / IQ3_XXS), 2.85 bpw | IQ3_S | ~101 GiB |
| L3 (fallback) | IQ2_XXS / IQ2_XS / IQ2_S, 2.46 bpw | IQ3_XXS | ~88 GiB |

\* The expert part comes from config geometry: 40 × 192 × 3 × 5120 × 2304 = 271.8 B params. About 10 GiB of dense weights is added on top. `plan` prints the exact size from the source GGUF before anything is written.

Fixed types for the rest of the file:

- attention, compressor, indexer and `engram_wkv`: Q8_0;
- shared experts, `token_embd` and `output`: Q6_K;
- router: BF16 (as stored);
- HC: F32 (patch 1);
- norms, sinks and biases: F32;
- any unnamed 2-D weight: Q8_0, never an expert type.

The protected layers come from config: 0-1, Engram 1/14, KV sources 2/8/14/20, and DSpark taps 37-39. They get the level's top type. The other layers are promoted one type step at a time, most sensitive first, until the expert-average target is reached. The ranking uses `--layer-scores` if s04 provides one, and uniform scores otherwise. Within a layer, `down_proj` ranks above `gate`/`up`.

Never Q4_K, Q5_K or IQ4 for experts. The source is FP4. A 4-bit non-FP4 grid spends bytes and still loses information. At 4.25 bpw the only right answer is MXFP4 copied as-is.

Other details:

- **imatrix:** our own, built from the s03 streamed per-expert x² bank and remapped through the manifest's keep order. It is written in the legacy `.dat` format and checked by `imatrix_verify.py` before use, which guards against the save-frequency truncation trap. Its dataset string records `keepset=<fingerprint>`; `quant` refuses an imatrix built for another keep-set than the source GGUF's, and the CHAIN_STEPS `imatrix` step rebuilds a stale one even if its done marker exists.
- **Error feedback (optional, `--ef-hessians D`):** `s09_ef.py` quantizes each expert superblock by superblock. It pushes each block's rounding error onto the columns not yet quantized (block-OBS / GPTQ), using ggml's own codec through ctypes. That way the patched bytes have the identical type and size. It needs full input second moments per layer, and s03 banks only their diagonal, so it stays off until a pass produces `L<L>.npz`. Its `selftest` is CPU-only.
- **Splits:** below 50 GB each (`--split-max-size 45G`).
- **EOS:** `eos = eot = 1` (`<｜end▁of▁sentence｜>`), written by `--override-kv` and verified on split 1. This guards against the multi-EOS trap.
- **Checks:** every level is verified from the written file for:
  - its tensor types;
  - the keep-set fingerprint against the source;
  - eos and eot;
  - the absence of Engram tensors.

## Engram row files

`python ops/s09_engram_rowfile.py build && … verify --full` reads base shards 47/48 sequentially, using `posix_fadvise(DONTNEED)` on both sides. It writes two files of about 101 GB each to `$V41_WORK/engram_rows` on NVMe. It refuses `/media` paths, because a USB drive is never served from. The header carries the source shard sha256, the teacher revision and the data sha256. See `ENGRAM_ROWFILE.md`.

## GPU test plan (`CHAIN_STEPS.sh`, not yet run)

1. **`backend_ops`:** `test-backend-ops test -b CUDA0`, every op, then `MUL_MAT`, `MUL_MAT_ID` and `GET_ROWS` for each weight type the levels use: mxfp4, q8_0, q6_k, iq3_s, iq3_xxs, iq2_s, iq2_xs, iq2_xxs, bf16 and f32. Any FAIL stops the chain.
2. **`kld LEVEL`:**
   - Input: the pre-registered **confirm** split (`s10_gates.json` `gguf.split`, computed by `ops/lib/rowsets.split_rows` exactly as s04/s10 compute theirs), text-only rows of at most `KLD_CTX` = 8192 tokens (longer rows are skipped and listed in `meta.json`), each decoded as a fresh sequence at `-c 8192` with the row store. On the current corpus that is 22 rows, 180,224 tokens. `s09_gate.py` fails a result that does not record the confirm split.
   - Measures: paired top-32 KL with a tail bucket, top-1 and ΔNLL against the cached teacher, with row-bootstrap CIs.
   - Output: the served identity (path, size, keep-set) is written into `teacher_kld.json`.
   - Levels that cannot fit with a 12 GiB reserve are skipped and logged (L0, and probably L1).
3. **`smoke LEVEL`:**
   - Start `llama-server --jinja`, and assert that `/props` `model_path` is the expected file.
   - Run one greedy chat turn. It must end with `finish_reason=stop`, must not contain `<｜User｜>`/`<｜Assistant｜>`, and must answer correctly.
   - The pid comes from an `exec`, so the server cannot be orphaned.

GPU steps run only under `quiet` (no leafllm vLLM, no runner, no other model loader, matched by process name or python argv, never by a whole command line). `reclaim` (`drop_caches`, then a ≥100 GiB baseline check) runs after each one. kld and smoke refuse a level on `/media`: `stage LEVEL` copies it to NVMe first and `unstage LEVEL` brings `teacher_eval/` back.

## Disk (NVMe), the order chain2 runs

`quant` writes each level to `S09_QUANT_ROOT` (the Seagate staging), so NVMe holds at most: the source GGUF (~152 GiB, convert to `drop_source`), then the row files (~189 GiB, from `rowfile` on) plus one staged level (~101 GiB L2 / ~88 GiB L3, `stage` to `unstage`). The two big ones never coexist with the source GGUF in that order: convert -> imatrix -> quant L2 + L3 -> drop_source -> rowfile -> stage/kld/smoke/unstage.

## Known limits

- **Context:** practical context is about 16K. The PR has no two-level (2048×8) candidate pass. Patch 2 makes reader layers sparse, but the index sources still score every compressed row.
- **DSpark drafting:** the runtime is upstream (`--spec-type draft-dspark`, `dflash: accept DeepSeek-V4.1 DSpark drafts`). Patch 0005 adds the V4.1 drafter export (`convert_hf_to_gguf.py --dspark`): its own 128 experts, the Markov head names, and target layers written as configured (V4.1 reads layer inputs; V4 is layer + 1). `ops/s09_dspark.py` converts, checks every tensor against a pinned reference drafter, benches it beside the largest level it fits next to, and gates it (`dspark_gguf`).
- **Vision:** mmproj encoder ported and golden-tested against the reference (`tests/v41_vision_golden.py`); end-to-end image QA not yet run.
- **CUDA MMQ:** the PR's notes mention an out-of-bounds read in CUDA MMQ. That bug is upstream, and `backend_ops` and `kld` are where it would show.
- **ik_llama.cpp:** it has merged `deepseek41` and the IQ*_KT trellis types. They are not used here: they have an open NaN bug (#2519) and are unverified on sm_110.
