#!/bin/bash
# CPU-side build of the pinned+patched fork. Compiles only; runs nothing on the GPU.
# Default target: Jetson AGX Thor (sm_110, the only GPU this release was measured on). Other NVIDIA GPUs:
#   CUDA_ARCH=<compute capability without the dot, e.g. 89 or "86;89">  llamacpp/build.sh
# (the build dir keeps the name build-thor either way, so every documented path stays valid).
set -euo pipefail
cd "$(dirname "$0")/llama.cpp"
J=${J:-4}
echo "build.sh: CMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH:-110}"
cmake -S . -B build-thor -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH:-110}" \
  -DGGML_NATIVE=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=ON -DBUILD_SHARED_LIBS=ON
nice -n 19 cmake --build build-thor -j "$J" --target llama-cli llama-server llama-quantize llama-imatrix \
  llama-perplexity llama-gguf-split test-backend-ops llama-gguf
