#!/bin/bash
# Hybrid build. --default-stream per-thread REQUIRED for CUDA-graph decode capture.
# Links the CUTLASS NVFP4 tensor-core primitive (cutlass_moe.o) for the TC verify path.
cd ~/gemma-cuda-hybrid
set -e
CUTLASS="-I $HOME/cutlass/include -I $HOME/cutlass/tools/util/include"
mkdir -p build
# CUTLASS object is slow to compile; rebuild only if its source changed.
if [ kernels/cutlass_moe.cu -nt build/cutlass_moe.o ] || [ ! -f build/cutlass_moe.o ]; then
  echo "compiling cutlass_moe.o (slow)..."
  nvcc -std=c++17 -O2 --expt-relaxed-constexpr -c $CUTLASS -I include \
    -gencode arch=compute_110a,code=sm_110a kernels/cutlass_moe.cu -o build/cutlass_moe.o
fi
nvcc -O2 -gencode arch=compute_110a,code=sm_110a --default-stream per-thread -Xcompiler -pthread -I include \
  src/forward.cu src/draft.cu src/megakernel.cu kernels/fp4_gemm.cu kernels/tc_verify_gemm.cu kernels/nvfp4_quant.cu kernels/elementwise.cu kernels/attention.cu \
  build/cutlass_moe.o \
  -lcublasLt -o build/forward "$@"
echo "build exit=$?"
