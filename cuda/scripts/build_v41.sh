#!/usr/bin/env bash
# Build the V4.1 server stack. Compile only -- nothing here touches the GPU or the weights.
#   scripts/build_v41.sh            all targets, -j4
#   scripts/build_v41.sh v41_decode one target
set -e; cd "$(dirname "$0")/.."
mkdir -p build/obj
NVCC="nvcc -O3 -std=c++17 -gencode arch=compute_110a,code=sm_110a --default-stream per-thread -I include -Xcompiler -Wall,-Wno-unused-function,-Wno-unknown-pragmas"
KOBJ=""
obj() {  # obj <src> -> build/obj/<name>.o, only when stale
  local o=build/obj/$(basename "${1%.*}").o
  if [ ! -f "$o" ] || [ "$1" -nt "$o" ] || [ -n "$(find include kernels/v41 -newer "$o" -name '*.h*' -print -quit)" ]; then
    local extra=""; case "$1" in *v41_tc5.cu|*v41_kqt.cu|*v41_vision.cu) extra="-I $HOME/cutlass/include -I $HOME/cutlass/examples/41_fused_multi_head_attention -I $HOME/cutlass/tools/util/include --expt-relaxed-constexpr -Xcompiler -Wno-all";; esac
    echo "  nvcc $1"; $NVCC $extra -c "$1" -o "$o" || exit 255      # 255 stops xargs, and so the build
  fi
  KOBJ="$KOBJ $o"
}
# kernels + model + engram (in parallel, at most 4)
SRCS="kernels/v41/v41_dense.cu kernels/v41/v41_tc5.cu kernels/v41/v41_kqt.cu kernels/v41/v41_moe.cu kernels/v41/v41_attn.cu kernels/v41/v41_engram.cu kernels/v41/v41_vision.cu
      src/v41_model.cu src/v41_draft.cu src/v41_engine.cu src/engram_store.cpp src/engram_hash.cpp"
export -f obj; export NVCC
echo $SRCS | tr ' ' '\n' | grep . | xargs -P4 -I{} bash -c 'obj {}'
for s in $SRCS; do KOBJ="$KOBJ build/obj/$(basename "${s%.*}").o"; done
LIBS="-lcuda"
want() { [ -z "$1" ] || [ "$1" = "$2" ]; }
if want "$1" v41_decode; then $NVCC src/v41_decode.cu $KOBJ $LIBS -o build/v41_decode; echo "built build/v41_decode"; fi
# C3: llguidance (Rust, MIT) as a static library; crates pinned by third_party/llg/Cargo.lock
LLG=third_party/llg/target/release/libllg_ffi.a
if want "$1" v41_server; then
  [ -f $LLG ] || (cd third_party/llg && cargo build --release --locked)
  $NVCC -Xcompiler -pthread server/server.cpp $KOBJ $LLG $LIBS -ldl -lm -o build/v41_server; echo "built build/v41_server"
fi
if want "$1" gate_llg; then   # CPU only: build/gate_llg <tokenizer.json>
  [ -f $LLG ] || (cd third_party/llg && cargo build --release --locked)
  g++ -O2 -std=c++17 -I include -I /usr/local/cuda/include tests/gate_llg.cpp $LLG -ldl -lpthread -lm -o build/gate_llg; echo "built build/gate_llg"
fi
if want "$1" gate_v41_units; then $NVCC tests/gate_v41_units.cu kernels/v41/kt_viterbi.cu $KOBJ $LIBS -o build/gate_v41_units; echo "built build/gate_v41_units"; fi
# KT2/KT3 encoder for s07 thor-trellis (ops/lib/kt.py loads it through ctypes)
if want "$1" libkt_viterbi; then $NVCC -shared -Xcompiler -fPIC kernels/v41/kt_viterbi.cu -o build/libkt_viterbi.so; echo "built build/libkt_viterbi.so"; fi
if want "$1" gate_sa_split; then $NVCC tests/gate_sa_split.cu $KOBJ $LIBS -o build/gate_sa_split; echo "built build/gate_sa_split"; fi
if want "$1" gate_sam_row; then $NVCC -I kernels/v41 tests/gate_sam_row.cu $KOBJ $LIBS -o build/gate_sam_row; echo "built build/gate_sam_row"; fi
if want "$1" gate_units_par; then $NVCC tests/gate_units_par.cu $KOBJ $LIBS -o build/gate_units_par; echo "built build/gate_units_par"; fi
if want "$1" gate_lpr; then $NVCC tests/gate_lpr.cu $KOBJ $LIBS -o build/gate_lpr; echo "built build/gate_lpr"; fi
if want "$1" gate_nvm; then $NVCC tests/gate_nvm.cu $KOBJ $LIBS -o build/gate_nvm; echo "built build/gate_nvm"; fi
if want "$1" bench_dense; then $NVCC tests/bench_dense.cu $KOBJ $LIBS -o build/bench_dense; echo "built build/bench_dense"; fi
if want "$1" gate_is_mma; then $NVCC tests/gate_is_mma.cu $KOBJ $LIBS -o build/gate_is_mma; echo "built build/gate_is_mma"; fi
if want "$1" gate_tc5; then $NVCC tests/gate_tc5.cu $KOBJ $LIBS -o build/gate_tc5; echo "built build/gate_tc5"; fi
if want "$1" gate_kqm_rt; then $NVCC -I kernels/v41 tests/gate_kqm_rt.cu $KOBJ $LIBS -o build/gate_kqm_rt; echo "built build/gate_kqm_rt"; fi
if want "$1" gate_kqt; then $NVCC -I kernels/v41 -I $HOME/cutlass/include -I $HOME/cutlass/tools/util/include --expt-relaxed-constexpr -Xcompiler -Wno-all tests/gate_kqt.cu $KOBJ $LIBS -o build/gate_kqt; echo "built build/gate_kqt"; fi
if want "$1" gate_f8m; then $NVCC tests/gate_f8m.cu $KOBJ $LIBS -o build/gate_f8m; echo "built build/gate_f8m"; fi
if want "$1" gate_topk; then $NVCC tests/gate_topk.cu $KOBJ $LIBS -o build/gate_topk; echo "built build/gate_topk"; fi
if want "$1" gate_kq_fast; then $NVCC -I kernels/v41 tests/gate_kq_fast.cu $KOBJ $LIBS -o build/gate_kq_fast; echo "built build/gate_kq_fast"; fi
if want "$1" gate_v41_golden; then $NVCC tests/gate_v41_golden.cu $KOBJ $LIBS -o build/gate_v41_golden; echo "built build/gate_v41_golden"; fi
if want "$1" gemv_format_bench; then $NVCC tools/gemv_format_bench.cu $LIBS -o build/gemv_format_bench; echo "built build/gemv_format_bench"; fi
if want "$1" nvm_bench; then $NVCC tools/nvm_bench.cu $LIBS -o build/nvm_bench; echo "built build/nvm_bench"; fi
if want "$1" router_bench; then $NVCC tools/router_bench.cu $KOBJ $LIBS -o build/router_bench; echo "built build/router_bench"; fi
if want "$1" gate_gumbel; then $NVCC tests/gate_gumbel.cu $KOBJ $LIBS -o build/gate_gumbel; echo "built build/gate_gumbel"; fi
if want "$1" gate_vision; then $NVCC tests/gate_vision.cu $KOBJ $LIBS -o build/gate_vision; echo "built build/gate_vision"; fi
if want "$1" gate_cnq_cl; then $NVCC tests/gate_cnq_cl.cu $KOBJ $LIBS -o build/gate_cnq_cl; echo "built build/gate_cnq_cl"; fi

# CPU gates
if want "$1" cpu_gates; then
  for g in gate_encoding_v41 gate_api gate_stream gate_tokenizer; do g++ -O2 -std=c++17 -I include tests/$g.cpp -o build/$g; done
  g++ -O2 -std=c++17 -I include tests/gate_engram_hash.cpp src/engram_hash.cpp -o build/gate_engram_hash
  g++ -O2 -std=c++17 -I include tests/gate_engram_store.cpp src/engram_store.cpp src/engram_hash.cpp \
      -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart -o build/gate_engram_store
  nvcc -O2 -std=c++17 -I include -Xcompiler -ffp-contract=off tests/gate_kq_host.cu -o build/gate_kq_host
  nvcc -O2 -std=c++17 -I include -Xcompiler -ffp-contract=off tests/gate_kt_host.cu -o build/gate_kt_host
  g++ -O2 -std=c++17 -I include tests/gate_spec_host.cpp -o build/gate_spec_host
  echo "built CPU gates"
fi
