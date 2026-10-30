# llama.cpp follow-on (owner priority 2026-10-26: GGUF users first; Thor CUDA engine work is parked)

Public branch: https://github.com/patrickbdevaney/llama.cpp/tree/deepseek-v4.1-reap-thor
= upstream PR #28696 head (2076412598e9) + patches/0001-0006. `llamacpp/setup.sh` reproduces it from upstream.

| Need | State | Work left |
|---|---|---|
| Text decode (V4.1 graph, HC F32, reader top-k, REAP heal gain) | done, gated by s09 KLD + llama-server smoke | rebase when PR #28696 changes or merges |
| Engram from NVMe (`LLAMA_ENGRAM_ROWFILE_DIR`, O_DIRECT row files) | done (0003, `ENGRAM_ROWFILE.md`) | the card must say how users get the 189 GiB row files (download vs build from the source checkpoint with `ops/s09_engram_rowfile.py`) and that they belong on NVMe |
| DSpark sidecar (`--dspark` export, `--spec-type draft-dspark`) | done (0005); benched by chain2 after the GGUF release | none unless the bench fails identity |
| Long context | clamped to 16384 (0006); `patches-pending/0007` candidate-block mask CPU-verified 15/15 | GPU: test-backend-ops on its ops, KLD at >=32K with/without mask, compute-buffer check; then move to patches/, re-publish card (existing GGUFs work via --override-kv, see patches-pending/README.md) |
| Vision (`vision_config`: deepseek_v41_vision ViT, 32 layers, d1024, patch 14, downsample 3, <=1024 image tokens) | **encoder validated 2026-10-30**: fork d7e51c015 (deepseek41v clip graph from PR #30817, engram off on image ubatches, `bias_vl` on mtmd ubatches) + 2030c5df5 (mmproj converter map); mmproj `s09/mmproj/mmproj-DeepSeek-V4.1-Flash-1017-bf16.gguf` (972 MB). `tests/v41_vision_golden.py` vs torch f32 ViT+Aligner: CPU and CUDA PASS on two images (206-tok native, 191-tok upscaled+padded), every row at least as close as the reference bf16 path, sentinel rows exact | end-to-end image QA through L3 (`llama-mtmd-cli`); publish the mmproj into the GGUF repo; README line 148 |
| HF GGUF card | rendered from artifacts by s12 | after publish: polish run instructions (branch, build, row files on NVMe, ctx limit, DSpark flag), state gate result honestly |
