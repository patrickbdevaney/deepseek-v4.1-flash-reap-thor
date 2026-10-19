---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
tags: [gguf, deepseek, moe, reap, llama.cpp]
quantized_by: s09 (deepseek-v4.1-flash-reap-thor)
---

<!-- Rendered by ops/s09_card.py. Every double-brace field comes from an artifact (plan.json, teacher_kld.json, the s10 result);
     the renderer refuses to write a card with an unresolved field or a level that failed its gate. -->

# DeepSeek-V4.1-Flash-1017 REAP-192: GGUF {{level}}

These are GGUF weights for an expert-pruned DeepSeek-V4.1-Flash-1017.

- **Pruning:** each MoE layer keeps 192 of its 384 routed experts. Keep-set `{{keep_set_sha256}}`.
- **Size:** this level is **{{size_gib}} GiB** in {{n_splits}} file(s).
- **Routed experts:** average **{{expert_bpw}} bpw**.

## What is in this file

| part | type |
|---|---|
| routed experts | {{expert_type_summary}} |
| protected layers ({{protected_layers}}) | {{protected_type}} |
| attention, compressor, indexer | Q8_0 |
| shared expert, embedding, output | Q6_K |
| router | BF16 |
| hyper-connection, norms, sinks | F32 |
| Engram tables | **not in the GGUF**: build the row files locally (below) |

Every level was quantized once, directly from the lossless MXFP4 master. No level was made from another level. The importance matrix came from our own calibration pass over the teacher, with the held-out split excluded.

## Quality against the unpruned teacher

The held-out text rows have {{kld.tokens}} tokens. Every statistic is paired per token against the teacher (FP8 + MXFP4), and the 95% CIs come from resampling rows.

| | mean | 95% CI |
|---|---|---|
| top-32 KL (nats) | {{kld.stats.kl_top32.mean}} | {{kld.stats.kl_top32.ci95}} |
| top-1 agreement | {{kld.stats.top1.mean}} | {{kld.stats.top1.ci95}} |
| ΔNLL (nats/token) | {{kld.stats.dnll.mean}} | {{kld.stats.dnll.ci95}} |

{{s10_table}}

## Running it

The build must be our pinned fork: upstream llama.cpp with PR #28696 plus three patches. Upstream llama.cpp cannot run this model yet. See [the fork notes]({{repo_url}}/tree/main/llamacpp).

```bash
git clone {{repo_url}} && cd deepseek-v4.1-flash-reap-thor/llamacpp && ./setup.sh && ./build.sh
# Engram row files (~189 GiB, NVMe), from base shards 47/48 of the unpruned checkpoint:
python ops/s09_engram_rowfile.py build --dst /nvme/engram_rows && python ops/s09_engram_rowfile.py verify --dst /nvme/engram_rows
LLAMA_ENGRAM_ROWFILE_DIR=/nvme/engram_rows ./llama.cpp/build-thor/bin/llama-server \
  -m {{first_split}} -c 8192 -ngl 999 --jinja
```

## Limits

- **Context:** practical context is about 16K. The PR has no two-level candidate mask, so attention beyond that reads every compressed row. Speed and memory degrade above it; the output stays correct.
- **Vision and DSpark:** not wired in this GGUF. There is no mmproj, and MTP/DSpark drafting is not included.
- **Measured scope:** the numbers above are for text only. Vision quality is measured by the CUDA-server release, not here.
- **End of turn:** `eos` and `eot` are both `<｜end▁of▁sentence｜>` (id 1), which is the only token that ends an assistant turn.

## Provenance

- **Teacher:** `deepseek-ai/DeepSeek-V4.1-Flash-1017` @ `{{source_revision}}`.
- **Fork:** llama.cpp `{{fork_sha}}`, which is PR #28696 head `2076412` plus patches.
- **Measured:** {{kld.measured}}.
