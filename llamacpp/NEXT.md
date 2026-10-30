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
| Long context | clamped to 16384 (0006); 0007 candidate-block mask CPU-verified 15/15, on the public branch but off unless the `deepseek41.candidate.*` keys are given | 2026-10-30 m7: test-backend-ops PASS, CUDA teacher at 32K done (mean NLL 0.0571), but the mask-on teacher-kld arm ran >2 h on one 32K row with no output (the 8K run does 97 tok/s) and was stopped. Next: profile one 20K row with mask on to find the slow op, then rerun the on/off KLD |
| Vision (`vision_config`: deepseek_v41_vision ViT, 32 layers, d1024, patch 14, downsample 3, <=1024 image tokens) | **done 2026-10-30**: encoder golden-tested (CPU+CUDA, `tests/v41_vision_golden.py`); end-to-end through `thor/start.sh`: test-1.jpeg OCR exact (masthead, date, headline, subheads), 84 tok/s prompt / 12.4 tok/s answer; mmproj published to the GGUF repo. A 531x229 low-res crop made it loop on the masthead | image-domain quality benchmark |
| HF GGUF card | rendered from artifacts by s12 | after publish: polish run instructions (branch, build, row files on NVMe, ctx limit, DSpark flag), state gate result honestly |
