#!/bin/bash
# CPU-side build of the pinned+patched fork for Thor (sm_110). Compiles only; runs nothing on the GPU.
set -euo pipefail
cd "$(dirname "$0")/llama.cpp"
J=${J:-4}
cmake -S . -B build-thor -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=110 \
  -DGGML_NATIVE=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=ON -DBUILD_SHARED_LIBS=ON
nice -n 19 cmake --build build-thor -j "$J" --target llama-cli llama-server llama-quantize llama-imatrix \
  llama-perplexity llama-gguf-split test-backend-ops llama-gguf
