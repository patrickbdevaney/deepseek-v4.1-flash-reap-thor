#!/usr/bin/env bash
# Build the production server surface: the OpenAI HTTP server, the terminal client, and the gates
# that hold them to account. One binary, no Python on the request path.
set -e; cd "$(dirname "$0")/.."
mkdir -p build

KERNELS="kernels/fp8_block_gemm.cu kernels/mla_attn.cu kernels/moe.cu kernels/hc.cu \
  kernels/hc_sinkhorn.cu kernels/mla_forward.cu kernels/block.cu kernels/compressor.cu \
  kernels/indexer.cu kernels/compressed_attn.cu kernels/compressed_block.cu kernels/tc_moe_gemm.cu \
  kernels/tc_fp8_gemm.cu kernels/mla_decode.cu kernels/compressed_decode.cu kernels/block_decode.cu \
  kernels/dscratch.cu kernels/dprof.cu kernels/dspark_real.cu kernels/dspark_attn.cu kernels/dspark.cu \
  kernels/nvfp4_dense.cu"

# CPU-only gates first: they are seconds, they need no GPU and no checkpoint, and every one of them
# guards something that would otherwise fail silently at request time.
g++ -O2 -std=c++17 -I include tests/gate_tokenizer.cpp -o build/gate_tokenizer && echo "built build/gate_tokenizer"
g++ -O2 -std=c++17 -I include tests/gate_stream.cpp    -o build/gate_stream    && echo "built build/gate_stream"
g++ -O2 -std=c++17 -I include tests/gate_encoding.cpp  -o build/gate_encoding  && echo "built build/gate_encoding"
g++ -O2 -std=c++17 -I include tests/gate_api.cpp       -o build/gate_api       && echo "built build/gate_api"

# The terminal client talks HTTP only — no CUDA, no model, runs anywhere.
g++ -O2 -std=c++17 -I include -pthread server/chat.cpp -o build/dsv4-chat && echo "built build/dsv4-chat"

# The server and the engine gate both link the engine against the same kernels decode.cu uses.
nvcc -O2 -std=c++17 -gencode arch=compute_110a,code=sm_110a -I include -Xcompiler -pthread \
  server/server.cpp src/engine.cu $KERNELS -o build/dsv4-server
echo "built build/dsv4-server"

nvcc -O2 -std=c++17 -gencode arch=compute_110a,code=sm_110a -I include -Xcompiler -pthread \
  tests/gate_engine.cu src/engine.cu $KERNELS -o build/gate_engine
echo "built build/gate_engine"
