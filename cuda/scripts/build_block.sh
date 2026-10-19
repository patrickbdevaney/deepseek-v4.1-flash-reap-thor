#!/usr/bin/env bash
set -e; cd "$(dirname "$0")/.."
nvcc -O2 -std=c++17 -gencode arch=compute_110a,code=sm_110a -I include \
  tests/gate_block.cu kernels/block.cu kernels/mla_forward.cu kernels/fp8_block_gemm.cu \
  kernels/mla_attn.cu kernels/moe.cu kernels/hc.cu kernels/hc_sinkhorn.cu \
  -o build/gate_block
echo "built build/gate_block"
