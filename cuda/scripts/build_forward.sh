#!/usr/bin/env bash
set -e; cd "$(dirname "$0")/.."
nvcc -O2 -lineinfo -std=c++17 -gencode arch=compute_110a,code=sm_110a -I include \
  src/forward.cu kernels/fp8_block_gemm.cu kernels/mla_attn.cu kernels/moe.cu kernels/hc.cu \
  kernels/hc_sinkhorn.cu kernels/mla_forward.cu kernels/block.cu kernels/compressor.cu \
  kernels/indexer.cu kernels/compressed_attn.cu kernels/compressed_block.cu kernels/tc_moe_gemm.cu kernels/tc_fp8_gemm.cu kernels/dspark_attn.cu kernels/dspark_real.cu kernels/dspark.cu kernels/dscratch.cu \
  -o build/forward
echo "built build/forward"
