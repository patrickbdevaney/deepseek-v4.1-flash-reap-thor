#!/bin/bash
# s09 GGUF path: steps for ops/chain.sh to run, one per call:  bash llamacpp/CHAIN_STEPS.sh <step> [LEVEL]
# Nothing here was run when it was written: the chain owns the GPU. Every step is idempotent (ops/logs/done.s09_*),
# appends to ops/logs/chain.log, and exits non-zero with a reason so the chain can write chain.STOP.
#
# Order (CPU steps can run while the GPU is busy with something else; GPU steps never co-run with anything):
#   build                 CPU  compile the pinned+patched fork (sm_110) and the teacher-KL tool
#   rowfile               CPU  Engram row files from shards 47/48 -> $V41_WORK/engram_rows (~189 GiB NVMe) + verify
#   backend_ops           GPU  test-backend-ops vs CPU for every op/type the levels use, on sm_110
#   convert               CPU  MXFP4 master (local dir, MASTER=) -> lossless source GGUF; keep-set must equal KEEP_SHA
#                              (default: the sha of $V41_WORK/s04/keepset.json)
#   imatrix               CPU  legacy .dat from the s03 bank, remapped to the master's reap_keepset.json, imatrix_verify
#   quant LEVEL           CPU  plan -> llama-quantize [-> EF] -> gguf-split (<50 GB) -> verify (L0 L1 L2 L3)
#   kld LEVEL             GPU  paired top-32 KL / top-1 / dNLL vs the cached teacher on the held-out text rows
#   smoke LEVEL           GPU  llama-server: health, identity (/props model_path), chat turn ends on EOS, no self-talk
#   manifest              CPU  $V41_WORK/s09/manifest.json for ops/s12_publish.py (verified levels only, sha256 per file)
# Env: V41_NKEEP (default 192, the arm s10 chose), MASTER, KEEP_SHA, EF_HESSIANS (optional), S09_REPO (optional)
set -uo pipefail
cd ~/deepseek-v4.1-flash-reap-thor
PY=.venv/bin/python; L=ops/logs; mkdir -p $L; C=$L/chain.log
WORK=${V41_WORK:-$HOME/models/v41-work}; G=$WORK/s09; ROWS=$WORK/engram_rows
export V41_NKEEP=${V41_NKEEP:-192}
BIN=llamacpp/llama.cpp/build-thor/bin
NAME=DeepSeek-V4.1-Flash-1017-REAP$V41_NKEEP
log() { echo "[$(date -Is)] s09 $*" | tee -a $C; }
die() { log "FAIL: $*"; exit 1; }
done_() { [ -f $L/done.s09_$1 ]; }; mark() { touch $L/done.s09_$1; log "done $1"; }
quiet() { docker ps --format '{{.Names}}' 2>/dev/null | grep -q leaf-teacher && die "a leafllm vLLM server is up"
  pgrep -f pipeline/runner.sh >/dev/null && ! [ -f ~/leafllm/run/runner.stop ] && die "leafllm runner active"
  pgrep -f 'llama-server|v41-teacher-kld|s03_saliency|s01_layer_probe' >/dev/null && die "another model-loading process is up"; true; }
reclaim() { sync; sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches'; sleep 5
  a=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo); log "reclaim after $1: MemAvailable ${a} GiB"
  [ "$a" -ge 100 ] || die "memory did not return to baseline after $1 (${a} GiB)"; }
split1() { ls $G/$1/$NAME-$1-00001-of-*.gguf 2>/dev/null | head -1; }
fits() {   # a level is GPU-testable on Thor only if its weights + 12 GiB reserve fit the available memory
  local f bytes avail; bytes=0; for f in $G/$1/$NAME-$1-0*-of-*.gguf; do bytes=$((bytes + $(stat -c%s "$f"))); done
  avail=$(awk '/MemAvailable/{print $2*1024}' /proc/meminfo)
  [ $((bytes + 12*1024*1024*1024)) -le "$avail" ] || { log "$1 is $((bytes>>30)) GiB: does not fit Thor, skipped (not a failure)"; return 1; }; }
export LLAMA_ENGRAM_ROWFILE_DIR=$ROWS

step=${1:?step}; LV=${2:-}
case $step in
build)
  done_ build && exit 0
  J=4 bash llamacpp/build.sh > $L/s09_build.log 2>&1 || die "build (see $L/s09_build.log)"
  g++ -O2 -std=c++17 llamacpp/tools/v41-teacher-kld.cpp -Illamacpp/llama.cpp/include -Illamacpp/llama.cpp/ggml/include \
      -L$BIN -lllama -Wl,-rpath,$(readlink -f $BIN) -o $BIN/v41-teacher-kld || die "v41-teacher-kld build"
  mark build ;;
rowfile)
  done_ rowfile && exit 0
  $PY ops/s09_engram_rowfile.py build > $L/s09_rowfile.log 2>&1 || die "rowfile build"
  $PY ops/s09_engram_rowfile.py verify --full >> $L/s09_rowfile.log 2>&1 || die "rowfile verify"
  mark rowfile ;;
backend_ops)
  done_ backend_ops && exit 0; quiet
  # every op the V4.1 graph uses, all types; then the matmul paths for exactly the weight types the levels carry
  $BIN/test-backend-ops test -b CUDA0 > $L/s09_backend_ops_all.log 2>&1; rc=$?
  for t in mxfp4 q8_0 q6_k iq3_s iq3_xxs iq2_s iq2_xs iq2_xxs bf16 f32; do
    $BIN/test-backend-ops test -b CUDA0 -o MUL_MAT,MUL_MAT_ID,GET_ROWS -p "type_a=$t" >> $L/s09_backend_ops_types.log 2>&1 || rc=1
  done
  grep -h 'FAIL' $L/s09_backend_ops_*.log | head -20 >> $C
  [ $rc = 0 ] && ! grep -q '\bFAIL\b' $L/s09_backend_ops_*.log || die "test-backend-ops failures on sm_110 (see $L/s09_backend_ops_*.log)"
  reclaim backend_ops; mark backend_ops ;;
convert)
  done_ convert && exit 0; quiet
  : "${MASTER:?MASTER=<local REAP-MXFP4 dir with reap_keepset.json>}"
  KEEP_SHA=${KEEP_SHA:-$($PY -c "import sys;sys.path.insert(0,'.');from ops.lib import keepset;print(keepset.load('$WORK/s04/keepset.json')[1])")} \
      || die "no arbitrated keep-set (s04/keepset.json)"
  mkdir -p $G/src
  $PY llamacpp/convert_v41_reap.py "$MASTER" --expect-keep-sha "$KEEP_SHA" --no-mtp \
      --outfile $G/src/V4.1-1017-REAP$V41_NKEEP-src.gguf > $L/s09_convert.log 2>&1 || die "convert (keep-set errors are fatal by design)"
  reclaim convert; mark convert ;;
imatrix)
  done_ imatrix && exit 0
  : "${MASTER:?MASTER}"
  $PY ops/s09_gguf_ladder.py imatrix --keepset "$MASTER/reap_keepset.json" > $L/s09_imatrix.log 2>&1 || die "imatrix"
  mark imatrix ;;
quant)
  : "${LV:?quant LEVEL}"; done_ quant_$LV && exit 0
  EF=(); [ -n "${EF_HESSIANS:-}" ] && EF=(--ef-hessians "$EF_HESSIANS")
  SC=(); [ -f $WORK/s04/layer_sensitivity.json ] && SC=(--layer-scores $WORK/s04/layer_sensitivity.json)
  $PY ops/s09_gguf_ladder.py quant $LV "${SC[@]}" "${EF[@]}" > $L/s09_quant_$LV.log 2>&1 || die "quant $LV"
  rm -f $G/$LV/$NAME-$LV.gguf    # the unsplit copy; the verified splits are the artifact
  mark quant_$LV ;;
kld)
  : "${LV:?kld LEVEL}"; done_ kld_$LV && exit 0; quiet; fits $LV || { mark kld_$LV; exit 0; }
  done_ rowfile || die "row files not built"
  [ -f $G/teacher_eval/meta.json ] || $PY ops/s09_teacher_eval.py prep > $L/s09_kld_prep.log 2>&1 || die "kld prep"
  out=$G/$LV/teacher_eval; mkdir -p $out
  setsid $BIN/v41-teacher-kld -m "$(split1 $LV)" -i $G/teacher_eval -o $out -c 8192 -b 2048 -ub 512 -ngl 999 \
      > $L/s09_kld_$LV.log 2>&1 || die "kld run $LV"
  reclaim kld_$LV
  $PY ops/s09_teacher_eval.py summarize $LV "$(split1 $LV)" >> $L/s09_kld_$LV.log 2>&1 || die "kld summarize $LV"
  mark kld_$LV ;;
smoke)
  : "${LV:?smoke LEVEL}"; done_ smoke_$LV && exit 0; quiet; fits $LV || { mark smoke_$LV; exit 0; }
  M=$(split1 $LV); PORT=18089
  # exec, so the recorded pid IS the server (a subshell pid would orphan it; eval-orphaned-server lesson)
  setsid bash -c "exec $BIN/llama-server -m '$M' -c 8192 -ngl 999 --jinja --port $PORT --host 127.0.0.1" \
      > $L/s09_smoke_$LV.log 2>&1 < /dev/null & SPID=$!
  trap 'kill $SPID 2>/dev/null; sleep 5; kill -9 $SPID 2>/dev/null' EXIT
  for i in $(seq 1 180); do curl -sf localhost:$PORT/health >/dev/null && break; kill -0 $SPID 2>/dev/null || die "server died"; sleep 10; done
  served=$(curl -sf localhost:$PORT/props | $PY -c 'import json,sys;print(json.load(sys.stdin)["model_path"])')
  [ "$(readlink -f "$served")" = "$(readlink -f "$M")" ] || die "served $served, expected $M"
  R=$(curl -sf localhost:$PORT/v1/chat/completions -H 'Content-Type: application/json' -d '{
      "messages":[{"role":"user","content":"In one sentence: what is the capital of France?"}],
      "max_tokens":512,"temperature":0}') || die "chat request failed"
  echo "$R" >> $L/s09_smoke_$LV.log
  echo "$R" | $PY -c '
import json,sys; r=json.load(sys.stdin); c=r["choices"][0]; t=(c["message"].get("content") or "")
assert c["finish_reason"]=="stop", "turn did not end on EOS: %s" % c["finish_reason"]
assert "<｜User｜>" not in t and "<｜Assistant｜>" not in t, "model kept talking past its turn"
assert "paris" in t.lower(), "wrong answer: %r" % t[:200]
print("smoke ok:", t[:120])' >> $L/s09_smoke_$LV.log 2>&1 || die "smoke $LV (see $L/s09_smoke_$LV.log)"
  kill $SPID; wait $SPID 2>/dev/null; trap - EXIT
  reclaim smoke_$LV; mark smoke_$LV ;;
manifest)
  R=(); [ -n "${S09_REPO:-}" ] && R=(--repo "$S09_REPO")
  $PY ops/s09_gguf_ladder.py manifest "${R[@]}" > $L/s09_manifest.log 2>&1 || die "manifest"
  log "manifest written: $G/manifest.json" ;;
*) die "unknown step $step" ;;
esac
