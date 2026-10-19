#!/bin/bash
# s08 CUDA server: steps for ops/chain.sh to run, one per call:  bash cuda/CHAIN_STEPS.sh <step> [ARM]
# NOTHING HERE HAS RUN ON THE GPU: this file was written compile-only (the chain owns the GPU). Every step
# is idempotent (ops/logs/done.s08_*), appends to ops/logs/chain.log, and exits non-zero with a reason so
# the chain can write chain.STOP. GPU steps never co-run with anything (quiet + reclaim, as s09).
#
# ARM = src (the unpruned source checkpoint, layer-at-a-time golden only) | A192 | B128 ... (a thor build
# under $V41_WORK/thor/<ARM>, ops/lib/thor_format.md).
#
# Order:
#   build                CPU  compile everything (scripts/build_v41.sh, -j4) + the CPU gates' binaries
#   cpu_gates            CPU  encoding (5 goldens + 400 reference-fuzz cases + strict parse), API, SSE stream,
#                             tokenizer (2116 cases / 168k tokens vs HF), Engram hash (18k ids vs engram.py),
#                             Engram store (sparse full-size .v41rows through io_uring O_DIRECT), K-quant decode
#   units                GPU  kernels vs the ops oracles (decoders bit-exact, act_quant exact, linears <= 1 ulp,
#                             router ids exact, moe_routed Q3_K/Q4_K cos >= 0.99999). Small: < 1 GiB.
#   gemv_bench           GPU  every expert format's GEMV bytes/s vs a streaming read -> $OUT/gemv_bench.json
#   golden_ref ARM       GPU  reference model.py dump, one layer at a time (tests/golden/ref_dump.py)
#   golden ARM           GPU  CUDA vs that dump, layer by layer: isolated + chained cosine, argmax agreement;
#                             then again through the real Engram row store ($ROWS, needs s09 rowfile)
#   decode ARM           GPU  whole-model greedy: determinism over 3 runs, graph == no-graph ids, decode-vs-
#                             prefill consistency, tok/s at 128 and 8k context -> $OUT/decode_<ARM>.json
#   ctx1m ARM            GPU  state for a 1,048,576-token context: allocation fits, tok/s after a 64k prefill
#   server ARM           GPU  v41_server smoke: health, chat turn ends on EOS, DSML tool call parsed, SSE
set -uo pipefail
cd ~/deepseek-v4.1-flash-reap-thor
PY=.venv/bin/python; L=ops/logs; mkdir -p $L; C=$L/chain.log
WORK=${V41_WORK:-$HOME/models/v41-work}; SRC=${V41_SRC:-$HOME/models/DeepSeek-V4.1-Flash-1017}
ROWS=$WORK/engram_rows; OUT=$WORK/s08; mkdir -p $OUT
KEEP=$WORK/s04/keepset.json
B=cuda/build
log() { echo "[$(date -Is)] s08 $*" | tee -a $C; }
die() { log "FAIL: $*"; exit 1; }
done_() { [ -f $L/done.s08_$1 ]; }; mark() { touch $L/done.s08_$1; log "done $1"; }
quiet() { docker ps --format '{{.Names}}' 2>/dev/null | grep -q leaf-teacher && die "a leafllm vLLM server is up"
  pgrep -f pipeline/runner.sh >/dev/null && ! [ -f ~/leafllm/run/runner.stop ] && die "leafllm runner active"
  pgrep -f 'llama-server|v41-teacher-kld|s03_saliency|s01_layer_probe|v41_server|v41_decode|gate_v41|ref_dump' >/dev/null \
    && die "another model-loading process is up"; true; }
reclaim() { sync; sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches'; sleep 5
  a=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo); log "reclaim after $1: MemAvailable ${a} GiB"
  [ "$a" -ge 100 ] || die "memory did not return to baseline after $1 (${a} GiB)"; }
ckpt_of() { [ "$1" = src ] && echo $SRC || echo $WORK/thor/$1; }
keep_of() { [ "$1" = src ] && echo "" || echo "--keepset $KEEP"; }
# GPU job: detached from the chain's shell (setsid), output to a log, waited on by pid
gpu() { local name=$1; shift; log "start $name: $*"; setsid "$@" > $L/s08_$name.log 2>&1 < /dev/null & local p=$!
  wait $p; local rc=$?; tail -n 40 $L/s08_$name.log | tee -a $C >/dev/null; return $rc; }

step=${1:?step}; ARM=${2:-}
case $step in
build)
  done_ build && exit 0
  cuda/scripts/build_v41.sh 2>&1 | tee -a $C | grep -q "built CPU gates" || die "build"
  mark build ;;
cpu_gates)
  done_ cpu_gates && exit 0
  cd cuda
  ./build/gate_encoding_v41 $SRC/encoding/tests tests/data/encoding_v41_fuzz.json || die "encoding"
  ./build/gate_api && ./build/gate_stream || die "api/stream"
  ./build/gate_tokenizer $SRC tests/data/tokenizer_vectors.json || die "tokenizer"
  ./build/gate_engram_hash tests/data || die "engram hash"
  mkdir -p build/engram_gate && ./build/gate_engram_store build/engram_gate tests/data/engram_tokmap.bin || die "engram store"
  [ -f build/units/dec_Q4_K_ref.npy ] || ../$PY -I tests/v41_unit_ref.py --out build/units || die "unit refs"
  ./build/gate_kq_host build/units || die "K-quant host decode"
  cd ..; mark cpu_gates ;;
units)
  done_ units && exit 0; quiet
  [ -f cuda/build/units/moe_ref.npy ] || $PY -I cuda/tests/v41_unit_ref.py --out cuda/build/units || die "unit refs"
  gpu units $B/gate_v41_units cuda/build/units || die "gate_v41_units"
  reclaim units; mark units ;;
gemv_bench)
  done_ gemv_bench && exit 0; quiet
  gpu gemv_bench $B/gemv_format_bench --mib 1536 --iters 20 --json $OUT/gemv_bench.json || die "gemv bench"
  reclaim gemv_bench; mark gemv_bench ;;
golden_ref)
  : ${ARM:?ARM}; done_ golden_ref_$ARM && exit 0; quiet
  extra=""; [ "$ARM" = src ] || extra="--keepset $KEEP --thor $WORK/thor/$ARM"
  gpu golden_ref_$ARM $PY -I cuda/tests/golden/ref_dump.py --out $WORK/golden/$ARM --long $extra || die "ref_dump $ARM"
  reclaim golden_ref_$ARM; mark golden_ref_$ARM ;;
golden)
  : ${ARM:?ARM}; done_ golden_$ARM && exit 0; quiet
  done_ golden_ref_$ARM || die "run golden_ref $ARM first"
  gpu golden_$ARM $B/gate_v41_golden --ckpt $(ckpt_of $ARM) --ref $WORK/golden/$ARM $(keep_of $ARM) || die "golden $ARM"
  reclaim golden_$ARM
  if [ -f $ROWS/engram_L14.v41rows ]; then
    gpu golden_store_$ARM $B/gate_v41_golden --ckpt $(ckpt_of $ARM) --ref $WORK/golden/$ARM $(keep_of $ARM) --engram $ROWS \
      || die "golden $ARM through the row store"
    reclaim golden_store_$ARM
  else log "no $ROWS yet: golden ran on the dumped Engram rows only (row-store pass deferred)"; fi
  mark golden_$ARM ;;
decode)
  : ${ARM:?ARM}; [ "$ARM" = src ] && die "decode needs a thor build (the source checkpoint does not fit Thor)"
  done_ decode_$ARM && exit 0; quiet
  [ -f $ROWS/engram_L14.v41rows ] || die "Engram row files missing ($ROWS): run llamacpp/CHAIN_STEPS.sh rowfile"
  K="--ckpt $(ckpt_of $ARM) --keepset $KEEP --engram $ROWS"
  gpu decode_$ARM $B/v41_decode $K --chat "Write a short paragraph about the history of the printing press." \
      --n 256 --repeat 3 --consistency 8 --json $OUT/decode_$ARM.json || die "decode $ARM (determinism/consistency)"
  reclaim decode_$ARM
  gpu decode_ng_$ARM $B/v41_decode $K --chat "Write a short paragraph about the history of the printing press." \
      --n 256 --no-graph --json $OUT/decode_nograph_$ARM.json || die "decode no-graph $ARM"
  reclaim decode_ng_$ARM
  $PY -c "import json,sys;a=json.load(open('$OUT/decode_$ARM.json'));b=json.load(open('$OUT/decode_nograph_$ARM.json'));sys.exit(a['ids']!=b['ids'])" \
    || die "graph and no-graph decode emitted different ids"
  $PY -c "import numpy as n;print(' '.join(map(str,n.load('$WORK/golden/$ARM/long20k/tokens.npy')[:8192])))" > $OUT/ids8k.txt \
    || die "ids8k (needs golden_ref $ARM --long)"
  gpu decode8k_$ARM $B/v41_decode $K --seqmax 16384 --ids-file $OUT/ids8k.txt --n 128 --json $OUT/decode8k_$ARM.json \
    || die "decode at 8k"
  reclaim decode8k_$ARM; mark decode_$ARM ;;
ctx1m)
  : ${ARM:?ARM}; done_ ctx1m_$ARM && exit 0; quiet
  $PY -c "import numpy as n;t=list(n.load('$WORK/golden/$ARM/long20k/tokens.npy'))*4;print(' '.join(map(str,t[:65536])))" \
    > $OUT/ids64k.txt || die "ids64k (needs golden_ref $ARM --long)"
  gpu ctx1m_$ARM $B/v41_decode --ckpt $(ckpt_of $ARM) --keepset $KEEP --engram $ROWS --seqmax 1048576 --chunk 128 \
      --ids-file $OUT/ids64k.txt --n 64 --json $OUT/ctx1m_$ARM.json || die "1M-context state / 64k prefill"
  reclaim ctx1m_$ARM; mark ctx1m_$ARM ;;
server)
  : ${ARM:?ARM}; done_ server_$ARM && exit 0; quiet
  PORT=18108
  setsid $B/v41_server --ckpt $(ckpt_of $ARM) --keepset $KEEP --engram $ROWS --port $PORT --host 127.0.0.1 \
      > $L/s08_server_$ARM.log 2>&1 < /dev/null & SP=$!
  for i in $(seq 1 180); do curl -sf localhost:$PORT/health >/dev/null && break; kill -0 $SP 2>/dev/null || break; sleep 10; done
  ok=1
  r=$(curl -sf localhost:$PORT/v1/chat/completions -H 'Content-Type: application/json' \
      -d '{"messages":[{"role":"user","content":"Say hello in five words."}],"max_tokens":64,"temperature":0,"thinking_mode":"chat"}') || ok=0
  echo "$r" | $PY -c "import json,sys;d=json.load(sys.stdin);c=d['choices'][0];sys.exit(not(c['finish_reason']=='stop' and c['message']['content'].strip()))" || ok=0
  t=$(curl -sf localhost:$PORT/v1/chat/completions -H 'Content-Type: application/json' -d '{"messages":[{"role":"user","content":"What is the weather in Paris? Use the tool."}],"tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],"max_tokens":256,"temperature":0,"thinking_mode":"chat"}') || ok=0
  echo "$t" >> $L/s08_server_$ARM.log
  echo "$t" | $PY -c "import json,sys;d=json.load(sys.stdin);tc=d['choices'][0]['message'].get('tool_calls') or [];sys.exit(not(tc and tc[0]['function']['name']=='get_weather'))" || ok=0
  kill $SP; wait $SP 2>/dev/null
  reclaim server_$ARM
  [ $ok = 1 ] || die "server smoke $ARM (see $L/s08_server_$ARM.log)"
  mark server_$ARM ;;
*) die "unknown step $step" ;;
esac
