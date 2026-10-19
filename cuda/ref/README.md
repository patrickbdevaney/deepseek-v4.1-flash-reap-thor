# ref/ — PyTorch golden-activation harness

Reference oracle for gating the CUDA port. Runs the reference deepseek_v4 architecture with the
tilelang/Hadamard kernels swapped for pure-PyTorch equivalents (no unbuildable deps), so every CUDA
kernel can be gated against a numerical golden. Runs on CPU torch (GPU-in-docker uses --runtime nvidia
which wedges on this box; CPU is sufficient for per-op/per-block goldens).

- deepseek_v4_ref.py     verbatim copy of the reference model.py (logic unchanged)
- kernel.py              pure-torch replacements: act_quant, fp4_act_quant, fp8_gemm, fp4_gemm,
                         sparse_attn, hc_split_sinkhorn  (shadows the reference's tilelang kernel.py)
- fast_hadamard_transform.py   pure-torch Walsh-Hadamard (shadows the CUDA extension)
- raw_loader.py          load raw HF shards into ref modules (fp8/fp4 native + .scale; dequant wo_a)
- gen_golden.py          smoke (tiny random model) | block (real weights, dump boundary goldens)

Run (CPU, in the torch container):
  docker run --rm --network none -e CUDA_VISIBLE_DEVICES="" -v $PWD/ref:/ref \
    vllm-dflash-thor:sglang python3 /ref/gen_golden.py smoke
