#!/bin/bash
# s09 GGUF path: steps for ops/chain2.sh to run, one per call:  bash llamacpp/CHAIN_STEPS.sh <step> [LEVEL]
# Nothing here was run when it was written: the chain owns the GPU. Every step is idempotent (ops/logs/done.s09_*),
# appends to ops/logs/chain.log, and exits non-zero with a reason so the chain can write chain.STOP.
#
# Order (CPU steps can run while the GPU is busy with something else; GPU steps never co-run with anything).
# The disk-lean order chain2 uses: convert -> imatrix -> quant L2 + quant L3 (to S09_QUANT_ROOT on the Seagate)
# -> drop_source -> rowfile -> stage L2 -> kld L2 + smoke L2 -> unstage L2 (L3 the same only if L2 fails its gate).
#   build                 CPU  compile the pinned+patched fork (sm_110) and the teacher-KL tool
#   rowfile               CPU  Engram row files from shards 47/48 -> $V41_WORK/engram_rows (~189 GiB NVMe) + verify
#   backend_ops           GPU  test-backend-ops vs CPU for every op/type the levels use, on sm_110
#   convert               CPU  MXFP4 master (MASTER=) + base Engram shards 47/48 (V41_SRC) -> lossless source GGUF
#                              (~152 GiB on NVMe, $V41_WORK/s09/src); keep-set must equal KEEP_SHA (default: the
#                              sha of $V41_WORK/s04/keepset.json). A done marker whose GGUF is gone or carries another
#                              keep-set is ignored and the GGUF is rebuilt.
#   imatrix               CPU  legacy .dat from the s03 bank, remapped to the master's reap_keepset.json, imatrix_verify;
#                              rebuilt when the recorded keep-set is not the master's
#   quant LEVEL           CPU  plan -> llama-quantize [-> EF] -> gguf-split (<50 GB) -> verify (L0 L1 L2 L3). With
#                              S09_QUANT_ROOT=<dir>, $V41_WORK/s09/LEVEL is a symlink to <dir>/LEVEL and every byte
#                              of the level (unsplit ~1x + splits ~1x) is written there, not to NVMe
#   drop_source           CPU  delete the source GGUF + its view and clear the convert markers, once every level in
#                              S09_LEVELS (default "L2 L3") is quantized; a later quant then needs convert again
#   stage LEVEL           CPU  copy a level that lives under /media to $V41_WORK/s09/stage/LEVEL on NVMe and point
#                              $V41_WORK/s09/LEVEL at it (no-op when the level is already on NVMe)
#   kld LEVEL             GPU  paired top-32 KL / top-1 / dNLL vs the cached teacher on the CONFIRM split (s10_gates
#                              gguf.split), rows <= KLD_CTX tokens (levels over the memory budget: partial -ngl from
#                              NVMe, recorded in teacher_eval/ngl.txt). Refuses a level on /media (stage it first)
#   smoke LEVEL           GPU  llama-server: health, identity (/props model_path), chat turn ends on EOS, no self-talk
#   unstage LEVEL         CPU  copy teacher_eval back next to the level's origin, re-point the symlink, delete the stage
#   manifest              CPU  $V41_WORK/s09/manifest.json for ops/s12_publish.py (verified levels only, sha256 per file)
# Env: V41_NKEEP (default 192, the arm s10 chose), MASTER, KEEP_SHA, V41_SRC (base checkpoint, for Engram shards
#      47/48 the master does not rehost), S09_QUANT_ROOT, S09_LEVELS, EF_HESSIANS (optional), S09_REPO (optional),
#      S09_ALLOW_USB=1 (let kld/smoke read a level from /media; off by default)
set -uo pipefail
cd ~/deepseek-v4.1-flash-reap-thor
PY=.venv/bin/python; L=ops/logs; mkdir -p $L; C=$L/chain.log
WORK=${V41_WORK:-$HOME/models/v41-work}; G=$WORK/s09; ROWS=$WORK/engram_rows
V41SRC=${V41_SRC:-$HOME/models/DeepSeek-V4.1-Flash-1017}
export V41_NKEEP=${V41_NKEEP:-192}
BIN=llamacpp/llama.cpp/build-thor/bin
NAME=DeepSeek-V4.1-Flash-1017-REAP$V41_NKEEP
SRCG=$G/src/V4.1-1017-REAP$V41_NKEEP-src.gguf
KLD_CTX=8192          # -c for v41-teacher-kld AND the row-length guard in s09_teacher_eval prep (one value)
GIB=1073741824
log() { echo "[$(date -Is)] s09 $*" | tee -a $C; }
die() { log "FAIL: $*"; exit 1; }
done_() { [ -f $L/done.s09_$1 ]; }; mark() { touch $L/done.s09_$1; log "done $1"; }
free_gib() { echo $(( $(df -B1 --output=avail "$1" | tail -1) / GIB )); }
gpu_busy() {  # a model-loading process is up. Matches the process NAME (comm, <= 15 chars, so prefixes) of binaries
  # and python's argv (interpreter first, then the script), never a whole command line: an editor, `tail -f`, or a
  # `bash -c` wrapper that merely mentions one of these names is not a model loader.
  pgrep '^(llama-server|llama-cli|llama-perplexi|test-backend-op|v41-teacher-kld|v41_server|v41_decode|gate_v41_)' >/dev/null \
    || pgrep -f '^[^ ]*python[0-9.]*( -[^ ]+)* [^ ]*(s03_saliency|s01_layer_probe|ref_dump)\.py( |$)' >/dev/null; }
quiet() { docker ps --format '{{.Names}}' 2>/dev/null | grep -q leaf-teacher && die "a leafllm vLLM server is up"
  pgrep -f pipeline/runner.sh >/dev/null && ! [ -f ~/leafllm/run/runner.stop ] && die "leafllm runner active"
  gpu_busy && die "another model-loading process is up"; true; }
reclaim() { sync; sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches'; sleep 5
  a=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo); log "reclaim after $1: MemAvailable ${a} GiB"
  [ "$a" -ge 100 ] || die "memory did not return to baseline after $1 (${a} GiB)"; }
split1() { ls $G/$1/$NAME-$1-00001-of-*.gguf 2>/dev/null | head -1; }
gguf_fp() { $PY -c "import sys;sys.path.insert(0,'.');from ops.s09_gguf_ladder import gguf_kv;print(gguf_kv(__import__('pathlib').Path(sys.argv[1])).get('deepseek41.reap.keep_set_sha256'))" "$1"; }
on_usb() { case "$(readlink -f "$1")" in /media/*) return 0;; esac; return 1; }
ngl_for() {  # -> GPU layer count for level $1: 999 if weights + 12 GiB reserve fit, else the layers that fit beside
  # a 20 GiB margin (the rest stays mmap'd from NVMe page cache, evictable). A partial run from the USB Seagate would
  # stream ~40-60 GiB per ubatch at ~130 MB/s, so it is refused: stage the level to NVMe first.
  local f bytes avail n per
  bytes=0; for f in $G/$1/$NAME-$1-0*-of-*.gguf; do bytes=$((bytes + $(stat -L -c%s "$f"))); done
  avail=$(awk '/MemAvailable/{print $2*1024}' /proc/meminfo)
  if [ $((bytes + 12*1024*1024*1024)) -le "$avail" ]; then echo 999; return 0; fi
  on_usb "$(split1 $1)" && { log "$1 is $((bytes>>30)) GiB on the USB disk: partial offload refused" >&2; return 1; }
  n=$($PY -c "import json;print(json.load(open('$V41SRC/inference/config.json'))['n_layers'])")
  per=$((bytes / (n + 2)))                       # + token_embd/output, roughly a layer each
  echo $(( (avail - 20*1024*1024*1024) / per - 2 )); }
need_nvme_level() {  # kld/smoke read the staged NVMe copy, never the USB original (unless S09_ALLOW_USB=1)
  local m; m=$(split1 $1); [ -n "$m" ] || die "$2 $1: no splits under $G/$1"
  if on_usb "$m" && [ "${S09_ALLOW_USB:-0}" != 1 ]; then die "$2 $1: the level is on the USB disk ($(readlink -f "$m")): run 'stage $1' first"; fi; }
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
  : > $L/s09_backend_ops_all.log; : > $L/s09_backend_ops_types.log     # a retry is judged on its own output only
  # every op the V4.1 graph uses, all types; then the matmul paths for exactly the weight types the levels carry
  $BIN/test-backend-ops test -b CUDA0 > $L/s09_backend_ops_all.log 2>&1; rc=$?
  # ggml type names are case-sensitive (q6_K, not q6_k): a misspelt filter matches nothing and prints "0/0 tests passed"
  for t in mxfp4 q8_0 q6_K iq3_s iq3_xxs iq2_s iq2_xs iq2_xxs bf16 f32; do
    out=$($BIN/test-backend-ops test -b CUDA0 -o MUL_MAT,MUL_MAT_ID,GET_ROWS -p "type_a=$t" 2>&1) || rc=1
    echo "$out" >> $L/s09_backend_ops_types.log
    echo "$out" | grep -qE '^ +[1-9][0-9]*/[0-9]+ tests passed' || { log "backend_ops: no tests ran for type_a=$t"; rc=1; }
  done
  grep -h 'FAIL' $L/s09_backend_ops_*.log | head -20 >> $C
  [ $rc = 0 ] && ! grep -q '\bFAIL\b' $L/s09_backend_ops_*.log || die "test-backend-ops failures on sm_110 (see $L/s09_backend_ops_*.log)"
  reclaim backend_ops; mark backend_ops ;;
convert)
  : "${MASTER:?MASTER=<local REAP-MXFP4 dir with reap_keepset.json>}"
  KEEP_SHA=${KEEP_SHA:-$($PY -c "import sys;sys.path.insert(0,'.');from ops.lib import keepset;print(keepset.load('$WORK/s04/keepset.json')[1])")} \
      || die "no arbitrated keep-set (s04/keepset.json)"
  if done_ convert; then
    if [ -s $SRCG ] && [ "$(gguf_fp $SRCG)" = "$KEEP_SHA" ]; then exit 0; fi
    log "convert: done marker present but $SRCG is missing or carries another keep-set: converting again"
    rm -f $L/done.s09_convert
  fi
  quiet
  mkdir -p $G/src
  # heal gains: the SAME rule as the NVFP4/Thor builds (ops/s07_quant.py heal_decision: s10/heal/decision.json
  # adopt + s06/heal_gain.json, both keep-set checked; no decision file -> fatal, the GGUF must not precede it).
  # Arm B ships unhealed, as s07 thor-b does (s06 measured arm A's keep-set).
  $PY - "$KEEP_SHA" "$G/src/heal_gains.json" > $L/s09_heal_gains.log 2>&1 <<'EOF' || die "heal gains (see $L/s09_heal_gains.log)"
import json, sys
sys.path.insert(0, ".")
from ops.lib import env, keepset
from ops.s07_quant import N_LAYERS, heal_decision
sha, out = sys.argv[1], sys.argv[2]
dec = env.WORK / "s10" / "heal" / "decision.json"
b = env.WORK / "s04" / "keepset_128.json"
if dec.exists() and b.exists() and keepset.load(b)[1] == sha and json.load(open(dec))["keepset_sha256"] != sha:
    gains, why = {str(L): 1.0 for L in range(N_LAYERS)}, "arm B ships unhealed (as s07 thor-b)"
else:
    gains, why = heal_decision(sha), "s07 heal_decision"
json.dump({"keepset_sha256": sha, "source": why, "gains": gains}, open(out, "w"), indent=1)
print(f"heal gains ({why}): {sum(v != 1.0 for v in gains.values())} of {len(gains)} layers != 1.0 -> {out}")
EOF
  cat $L/s09_heal_gains.log >> $C
  # s05 does not rehost Engram shards 47/48 but the master's index maps engram.* to them, and the converter opens
  # every shard the index names: it reads a symlink view (master + 47/48 from V41_SRC; nothing written to the master).
  # The view's name is the GGUF's general.name; the V4.1 chat template is forced whatever the name.
  # --use-temp-file: upstream repacks each routed-expert tensor into an eager numpy array and the writer keeps
  # every one until the final write (~145 GiB for 40 layers): without it memguard killed convert at layer 32 and
  # the GPU wedged on a GC6 exit under the memory pressure (2026-10-24). The spill (~150 GiB) lives next to the
  # output on NVMe and is deleted when the writer closes: peak NVMe use is spill + output.
  F=$(free_gib $G/); [ "$F" -ge 320 ] || die "convert: $F GiB free on NVMe, need 320 (output ~152 + temp spill ~150)"
  rm -rf $G/tmp; mkdir -p $G/tmp
  TMPDIR=$G/tmp $PY llamacpp/convert_v41_reap.py "$MASTER" --expect-keep-sha "$KEEP_SHA" --heal-gains $G/src/heal_gains.json --no-mtp \
      --engram-src "$V41SRC" --view-dir $G/src/$NAME --use-temp-file \
      --outfile $SRCG > $L/s09_convert.log 2>&1 || die "convert (keep-set errors are fatal by design; see $L/s09_convert.log)"
  rm -rf $G/tmp
  reclaim convert; mark convert ;;
imatrix)
  : "${MASTER:?MASTER}"
  if done_ imatrix; then
    $PY ops/s09_gguf_ladder.py imatrix --keepset "$MASTER/reap_keepset.json" --check >> $L/s09_imatrix.log 2>&1 && exit 0
    log "imatrix: the existing .dat was built for another keep-set (or records none): rebuilding"; rm -f $L/done.s09_imatrix
  fi
  $PY ops/s09_gguf_ladder.py imatrix --keepset "$MASTER/reap_keepset.json" > $L/s09_imatrix.log 2>&1 || die "imatrix"
  mark imatrix ;;
quant)
  : "${LV:?quant LEVEL}"; done_ quant_$LV && exit 0
  [ -s $SRCG ] || die "quant $LV: source GGUF $SRCG is missing (drop_source removed it?): run 'convert' first"
  mkdir -p $G
  if [ -n "${S09_QUANT_ROOT:-}" ]; then   # the level's bytes go to S09_QUANT_ROOT (the Seagate staging), not NVMe
    mkdir -p "$S09_QUANT_ROOT/$LV" || die "quant $LV: cannot create $S09_QUANT_ROOT/$LV"
    if [ -d $G/$LV ] && ! [ -L $G/$LV ]; then rmdir $G/$LV 2>/dev/null || die "$G/$LV is a real non-empty directory (expected a symlink to $S09_QUANT_ROOT/$LV)"; fi
    ln -sfn "$S09_QUANT_ROOT/$LV" $G/$LV
  fi
  mkdir -p $G/$LV
  f=$(free_gib $G/$LV/); s=$(( $(stat -c%s $SRCG) / GIB ))
  # the unsplit level and its splits coexist until the split verifies: ~2x the level, bounded by 2x the source
  [ "$f" -ge $(( s * 3 / 2 + 20 )) ] || die "quant $LV: $f GiB free under $(readlink -f $G/$LV), need $(( s * 3 / 2 + 20 ))"
  EF=(); [ -n "${EF_HESSIANS:-}" ] && EF=(--ef-hessians "$EF_HESSIANS")
  SC=(); [ -f $WORK/s04/layer_sensitivity.json ] && SC=(--layer-scores $WORK/s04/layer_sensitivity.json)
  $PY ops/s09_gguf_ladder.py quant $LV "${SC[@]}" "${EF[@]}" > $L/s09_quant_$LV.log 2>&1 || die "quant $LV"
  rm -f $G/$LV/$NAME-$LV.gguf    # the unsplit copy; the verified splits are the artifact
  mark quant_$LV ;;
drop_source)
  for lv in ${S09_LEVELS:-L2 L3}; do done_ quant_$lv || die "drop_source: $lv is not quantized yet (S09_LEVELS='${S09_LEVELS:-L2 L3}')"; done
  if [ -e $SRCG ]; then
    s=$(( $(stat -c%s $SRCG) / GIB )); rm -f $SRCG; log "drop_source: removed $SRCG (${s} GiB, regenerable by convert)"
  fi
  rm -rf $G/src/$NAME                        # the symlink view (symlinks only)
  # the convert marker, so a later convert call rebuilds the GGUF (and quant fails clearly until it has). chain2's own
  # done.c2_* markers are chain2's business: it must not re-call convert once the quants are done.
  rm -f $L/done.s09_convert
  log "drop_source: convert marker cleared; $(free_gib $G/) GiB free on $(df --output=target $G/ | tail -1)" ;;
stage)
  : "${LV:?stage LEVEL}"; done_ quant_$LV || die "stage $LV: not quantized"
  M=$(split1 $LV); [ -n "$M" ] || die "stage $LV: no splits under $G/$LV"
  ST=$G/stage/$LV
  [ "$(readlink -f $G/$LV)" = "$(readlink -f $ST)" ] && [ -f $ST/.origin ] && { log "stage $LV: already staged"; exit 0; }
  on_usb "$M" || { log "stage $LV: already on NVMe ($(readlink -f $G/$LV)), nothing to copy"; exit 0; }
  ORIGIN=$(readlink -f $G/$LV)
  sz=$(du -s -B1 -L "$ORIGIN" | cut -f1); f=$(free_gib $G/)
  [ "$f" -ge $(( sz / GIB + 25 )) ] || die "stage $LV: needs $(( sz / GIB + 25 )) GiB free on NVMe, only $f"
  mkdir -p $ST; rsync -a --partial "$ORIGIN/" $ST/ || die "stage $LV: copy"
  for p in "$ORIGIN"/$NAME-$LV-0*-of-*.gguf; do
    [ "$(stat -c%s "$p")" = "$(stat -c%s "$ST/$(basename "$p")")" ] || die "stage $LV: $(basename "$p") size differs after copy"
  done
  echo "$ORIGIN" > $ST/.origin; ln -sfn $ST $G/$LV
  log "stage $LV: $(( sz / GIB )) GiB copied to $ST (origin $ORIGIN)" ;;
unstage)
  : "${LV:?unstage LEVEL}"; ST=$G/stage/$LV
  [ -f $ST/.origin ] || { log "unstage $LV: nothing staged"; exit 0; }
  ORIGIN=$(cat $ST/.origin)
  if [ -d $ST/teacher_eval ]; then rsync -a $ST/teacher_eval/ "$ORIGIN/teacher_eval/" || die "unstage $LV: copy results back"; fi
  ln -sfn "$ORIGIN" $G/$LV && rm -rf $ST && log "unstage $LV: results back in $ORIGIN, stage removed" ;;
kld)
  : "${LV:?kld LEVEL}"; done_ kld_$LV && exit 0; quiet; need_nvme_level $LV kld
  NGL=$(ngl_for $LV) || die "kld $LV: stage it to NVMe"
  [ "$NGL" -ge 1 ] || die "kld $LV: no layer fits on the GPU (ngl $NGL)"
  done_ rowfile || die "row files not built"
  # idempotent: rewrites the arrays only if meta.json is not already this split + ctx + row list
  $PY ops/s09_teacher_eval.py prep --ctx $KLD_CTX > $L/s09_kld_prep.log 2>&1 || die "kld prep (see $L/s09_kld_prep.log)"
  out=$G/$LV/teacher_eval; mkdir -p $out
  [ "$NGL" = 999 ] && echo all > $out/ngl.txt || { echo $NGL > $out/ngl.txt; log "kld $LV: partial offload, $NGL GPU layers"; }
  setsid $BIN/v41-teacher-kld -m "$(split1 $LV)" -i $G/teacher_eval -o $out -c $KLD_CTX -b 2048 -ub 512 -ngl $NGL \
      > $L/s09_kld_$LV.log 2>&1 || die "kld run $LV"
  reclaim kld_$LV
  $PY ops/s09_teacher_eval.py summarize $LV "$(split1 $LV)" >> $L/s09_kld_$LV.log 2>&1 || die "kld summarize $LV"
  mark kld_$LV ;;
smoke)
  : "${LV:?smoke LEVEL}"; done_ smoke_$LV && exit 0; quiet; need_nvme_level $LV smoke
  NGL=$(ngl_for $LV) || die "smoke $LV: stage it to NVMe"
  M=$(split1 $LV); PORT=18089
  # exec, so the recorded pid IS the server (a subshell pid would orphan it; eval-orphaned-server lesson)
  setsid bash -c "exec $BIN/llama-server -m '$M' -c 8192 -np 1 -ngl $NGL --jinja --port $PORT --host 127.0.0.1" \
      > $L/s09_smoke_$LV.log 2>&1 < /dev/null & SPID=$!
  trap 'kill $SPID 2>/dev/null; sleep 5; kill -9 $SPID 2>/dev/null' EXIT
  for i in $(seq 1 180); do curl -sf --max-time 10 localhost:$PORT/health >/dev/null && break; kill -0 $SPID 2>/dev/null || die "server died"; sleep 10; done
  served=$(curl -sf --max-time 60 localhost:$PORT/props | $PY -c 'import json,sys;print(json.load(sys.stdin)["model_path"])')
  [ "$(readlink -f "$served")" = "$(readlink -f "$M")" ] || die "served $served, expected $M"
  R=$(curl -sf --max-time 600 localhost:$PORT/v1/chat/completions -H 'Content-Type: application/json' -d '{
      "messages":[{"role":"user","content":"In one sentence: what is the capital of France?"}],
      "max_tokens":512,"temperature":0}') || die "chat request failed (or exceeded 600 s)"
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
