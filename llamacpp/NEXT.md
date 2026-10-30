# llama.cpp follow-on (owner priority 2026-10-26: GGUF users first; Thor CUDA engine work is parked)

Public branch: https://github.com/patrickbdevaney/llama.cpp/tree/deepseek-v4.1-reap-thor (head d42d8daf7, 2026-10-30)
= upstream PR #28696 head (2076412598e9) + patches/0001-0006 (92031e6a4) + the 0005 target-layer key fix, 0007 candidate mask,
vision projector, mmproj export, and `thor/` (build.sh, start.sh, README with measured numbers). `llamacpp/setup.sh`
reproduces the 0001-0006 base from upstream; the local `v41-thor` branch has the same tree minus `thor/`.

| Need | State | Work left |
|---|---|---|
| Text decode (V4.1 graph, HC F32, reader top-k, REAP heal gain) | done, gated by s09 KLD + llama-server smoke | rebase when PR #28696 changes or merges |
| Engram from NVMe (`LLAMA_ENGRAM_ROWFILE_DIR`, O_DIRECT row files) | done (0003, `ENGRAM_ROWFILE.md`) | the card must say how users get the 189 GiB row files (download vs build from the source checkpoint with `ops/s09_engram_rowfile.py`) and that they belong on NVMe |
| DSpark sidecar (`--dspark` export, `--spec-type draft-dspark`) | done; published 2026-10-30 (keys added, 78/78 tensors byte-identical, reference check PASS). Bench via `DSPARK=1 thor/start.sh`: 16.6-19.9 vs 12.3-12.5 tok/s, accept 55-72% at n=3; greedy text diverges from plain decode at near-ties (batched verify numerics) | sweep `DRAFT_N`; test with mmproj loaded |
| Long context | **0007 validated at 32K (2026-10-30, m7b)**: control <16K identical on/off; >=16K mask on vs off dKL +0.00055 [-0.00001, +0.00122], dNLL +0.00024 [-0.00024, +0.00081] (non-inferior, no measurable effect at 32K: 2048 of 4096 blocks kept). The earlier >2 h stall was not the mask: the indexer's cont(permute 2,1,0,3) fell to ggml-cuda's scalar copy (79% of GPU time, nsys `ops/logs/prof/m7on32k`); fork f27313c40 does it as two tiled 0<->1 transposes (bit-identical, release row 1 KL 0.26541 reproduced) -> 32K row 14.6 min. `thor/start.sh` passes the keys when CTX > 16384 | 64K+ validation (where the mask bites); a fused indexer kernel for long prefill; regenerate patches/ from the branch |
| Vision (`vision_config`: deepseek_v41_vision ViT, 32 layers, d1024, patch 14, downsample 3, <=1024 image tokens) | **done 2026-10-30**: encoder golden-tested (CPU+CUDA, `tests/v41_vision_golden.py`); end-to-end through `thor/start.sh`: test-1.jpeg OCR exact (masthead, date, headline, subheads), 84 tok/s prompt / 12.4 tok/s answer; mmproj published to the GGUF repo. A 531x229 low-res crop made it loop on the masthead | image-domain quality benchmark |
| HF GGUF card | rendered from artifacts by s12 | after publish: polish run instructions (branch, build, row files on NVMe, ctx limit, DSpark flag), state gate result honestly |
