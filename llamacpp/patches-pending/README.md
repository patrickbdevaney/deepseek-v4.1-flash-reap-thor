# Pending llama.cpp patches (not applied by `llamacpp/setup.sh`)

`0007-deepseek41-candidate-block-mask.patch` ports V4.1's candidate-block mask (model.py l.570-606) so the
16384-token clamp of 0006 applies only to GGUFs without the `deepseek41.candidate.*` keys. CPU-verified
(`0007-candidate-mask/`: 15/15 cases match the NumPy reference, max abs err 4.46e-7, bit-identical below the
threshold). NOT yet run on the GPU, so the release GGUF is built and gated without it.

Before it moves into `patches/` (GGUF v2, long context):
1. `test-backend-ops` on CUDA for the ops it uses; none may fall back to the CPU.
2. KLD vs the cached teacher on a >= 32K-token row: with the mask, without it (`LLAMA_DSV41_UNSAFE_LONG_CTX=1`),
   and <= 16K as a regression check. Existing GGUFs lack the keys: serve them with
   `--override-kv deepseek41.candidate.source_layer=int:20 --override-kv deepseek41.candidate.topk_blocks=int:2048
   --override-kv deepseek41.candidate.block_size=int:8`.
3. Compute buffer at long context (~256 MB extra per consumer layer at 128K, ubatch 512).
Known: ties at the 2048th block (bf16 scores in the reference) make selection non-bit-exact; KLD decides.
