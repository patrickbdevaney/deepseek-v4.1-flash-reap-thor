#!/bin/bash
# CPU-only build of the candidate-mask test against the clone's CPU libggml (GGML_CUDA=OFF).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
src="${LLAMA_MASK_SRC:-$here/../llama-mask}"
g++ -O2 -std=c++17 -Wall -Wextra \
    -I"$src/ggml/include" -I"$src/src/models" \
    "$here/test-dsv41-candidates.cpp" -o "$here/test-dsv41-candidates" \
    -L"$src/build-cpu/bin" -lggml -lggml-base -lggml-cpu -Wl,-rpath,"$src/build-cpu/bin"
