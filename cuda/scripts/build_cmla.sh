#!/usr/bin/env bash
set -e; cd "$(dirname "$0")/.."
nvcc -O2 -std=c++17 -gencode arch=compute_110a,code=sm_110a -I include \
  tests/gate_cmla.cu kernels/compressed_attn.cu kernels/mla_forward.cu kernels/fp8_block_gemm.cu \
  kernels/mla_attn.cu kernels/compressor.cu kernels/indexer.cu kernels/hc_sinkhorn.cu \
  -o build/gate_cmla
echo "built build/gate_cmla"
