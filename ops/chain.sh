#!/bin/bash
# Autonomous chain s00 -> s00b -> s01 (gates) -> s02 -> s03 -> s04 -> s05 -> s06 -> s10heal -> s07 -> s10final -> s12
# (user grant 2026-10-19). Detached: setsid nohup bash ops/chain.sh
# Each step is idempotent/resumable; a failed gate writes ops/logs/chain.STOP with the reason and exits (the 30-min
# check-in fixes it, removes the STOP file and relaunches the chain). One GPU/model-loading process at a time; reclaim
# (drop_caches + baseline check) between model-loading units. Never runs while leafllm's runner or a vLLM server is up.
cd ~/deepseek-v4.1-flash-reap-thor
PY=.venv/bin/python; L=ops/logs; mkdir -p $L; C=$L/chain.log
log() { echo "[$(date -Is)] $*" >> $C; }
stop() { log "STOP: $*"; echo "$(date -Is) $*" > $L/chain.STOP; exit 1; }
[ -f $L/chain.STOP ] && { log "chain.STOP present, not starting: $(cat $L/chain.STOP)"; exit 1; }
exec 9> /tmp/dsv41-chain.lock; flock -n 9 || exit 0
pgrep -f ops/memguard.sh >/dev/null || setsid nohup bash ops/memguard.sh >/dev/null 2>&1 </dev/null &
SEA="/media/patrickd/Seagate Portable Drive/v41_1017_engram"; SRC=~/models/DeepSeek-V4.1-Flash-1017
reclaim() { sync; sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches'; sleep 5
  a=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo); log "reclaim after $1: MemAvailable ${a} GiB"
  [ "$a" -ge 100 ] || stop "memory did not return to baseline after $1 (${a} GiB < 100)"; }
quiet() { docker ps --format '{{.Names}}' | grep -q leaf-teacher && stop "a leafllm vLLM server is up"
  pgrep -f pipeline/runner.sh >/dev/null && ! [ -f ~/leafllm/run/runner.stop ] && stop "leafllm runner active"; true; }
step_done() { [ -f $L/done.$1 ]; }; mark() { touch $L/done.$1; log "done $1"; }

log "chain start"
# s00: wait for the verified download
until grep -q 'SHA VERIFY' ops/logs/s00_fetch.log 2>/dev/null; do sleep 120; done
grep -q 'SHA VERIFY PASS' ops/logs/s00_fetch.log || stop "s00 sha verify failed: $(grep 'SHA VERIFY' ops/logs/s00_fetch.log)"
quiet

# s00b: Engram shards 47/48 -> Seagate (calibration reads them once, sequentially); verified, then NVMe copies removed
if ! step_done s00b; then
  mkdir -p "$SEA"
  for s in model-00047-of-00048.safetensors model-00048-of-00048.safetensors; do
    [ -L "$SRC/$s" ] && continue
    rsync -t --partial "$SRC/$s" "$SEA/$s" || stop "s00b copy $s"
    want=$(python3 -c "import json;print([x['lfs']['oid'] for x in json.load(open('$SRC/../v41_1017_tree.json')) if x['path']=='$s'][0])")
    got=$(sha256sum "$SEA/$s" | cut -d' ' -f1)
    [ "$want" = "$got" ] || stop "s00b sha mismatch on Seagate copy of $s"
    rm -f "$SRC/$s" && ln -s "$SEA/$s" "$SRC/$s"; log "s00b $s on Seagate, sha ok"
  done
  mark s00b
fi

# s01: kernel gate (TileLang first, torch fallback), then token-chunked path, then a 32k probe for the long-row path
KERN=tilelang; [ -f $L/kernels ] && KERN=$(cat $L/kernels)
verdict() { python3 -c "import json;print(json.load(open('$L/s01_layer_probe.json')).get('verdict'))" 2>/dev/null; }
run01() { quiet; setsid $PY ops/s01_layer_probe.py "$@" > $L/s01_t$2_$(echo "$@" | tr -dc "a-z0-9" | tail -c 12).log 2>&1; }
if ! step_done s01; then
  run01 --tokens 2048 --kernels $KERN; rc=$?
  if [ $rc = 3 ] && [ $KERN = tilelang ]; then log "s01: TileLang unavailable on sm_110a, torch fallback"; KERN=torch; echo torch > $L/kernels
    reclaim s01-tilelang; run01 --tokens 2048 --kernels torch; rc=$?; fi
  [ $rc = 0 ] && [ "$(verdict)" = PASS ] || stop "s01 2k gate: rc=$rc verdict=$(verdict) (see $L/s01_*.log)"
  cp $L/s01_layer_probe.json $L/s01_2k.json; reclaim s01-2k
  run01 --tokens 4096 --block-chunk 1024 --kernels $KERN; rc=$?
  [ $rc = 0 ] && [ "$(verdict)" = PASS ] || stop "s01 chunked gate: rc=$rc verdict=$(verdict)"
  cp $L/s01_layer_probe.json $L/s01_4k_chunk.json; reclaim s01-4k
  # just above 16k exercises the candidate-block path; 32k would put a ~64 GiB dense indexer score tensor on the CPU oracle
  run01 --tokens 17408 --block-chunk 4096 --kernels $KERN; rc=$?
  [ $rc = 0 ] && [ "$(verdict)" = PASS ] || stop "s01 17k gate (candidate-block path): rc=$rc verdict=$(verdict)"
  cp $L/s01_layer_probe.json $L/s01_17k.json; reclaim s01-17k; mark s01
fi

# s02: corpus (CPU + network): probe live licences/yield, then build
if ! step_done s02; then
  $PY ops/s02_calib_corpus.py --probe > $L/s02_probe.log 2>&1 || stop "s02 probe failed"
  $PY ops/s02_calib_corpus.py > $L/s02.log 2>&1 || stop "s02 build failed"
  mark s02; reclaim s02
fi

# s03: layer-major saliency (hours, resumable)
if ! step_done s03; then
  quiet; setsid $PY ops/s03_saliency.py run --kernels $(cat $L/kernels 2>/dev/null || echo tilelang) --engram-dir "$SEA" > $L/s03.log 2>&1 \
    || stop "s03 failed (resumable: fix, rm chain.STOP, relaunch)"
  mark s03; reclaim s03
  $PY ops/drift_guard.py protect > $L/drift_protect.log 2>&1 || stop "drift_guard protect failed"
fi

# ---------------------------------------------------------------------------------------------- s04 -> s12
# Same pattern: one model-loading process at a time, quiet() before every GPU stage, reclaim after every
# model-loading stage, a failed stage writes chain.STOP. Every stage resumes from its own ledger.
KERN=$(cat $L/kernels 2>/dev/null || echo tilelang)

# s04: candidate masks (CPU) + arbitration by held-out KL with re-routing (GPU replay) -> keepset.json / keepset_128.json
if ! step_done s04; then
  quiet; setsid $PY ops/s04_select_arbitrate.py run --kernels $KERN --drop-s03-stream > $L/s04.log 2>&1 \
    || stop "s04 failed (resumable; see $L/s04.log)"
  mark s04; reclaim s04
fi

# s05: bit-exact MXFP4 master, streamed shard -> verify -> PRIVATE HF upload -> delete (CPU + network)
if ! step_done s05; then
  $PY ops/s05_surgery.py run > $L/s05.log 2>&1 || stop "s05 surgery/upload failed (resumable)"
  mark s05; reclaim s05
fi

# s06: measured per-layer heal gain by router replay on the final keep-set + post-prune activation bank (GPU)
if ! step_done s06; then
  quiet; setsid $PY ops/s06_heal.py run --kernels $KERN > $L/s06.log 2>&1 || stop "s06 heal failed (resumable)"
  mark s06; reclaim s06
fi

# s10 heal panel: paired healed vs unhealed on the SCREEN split -> s10/heal/decision.json (GPU)
if ! step_done s10heal; then
  quiet; setsid $PY ops/s10_eval.py heal --kernels $KERN > $L/s10_heal.log 2>&1 || stop "s10 heal panel failed"
  mark s10heal; reclaim s10heal
fi

# s07a: lossless NVFP4 repack with folded gains, streamed to a PRIVATE repo (CPU + network)
if ! step_done s07nvfp4; then
  $PY ops/s07_quant.py nvfp4 > $L/s07_nvfp4.log 2>&1 || stop "s07 nvfp4 failed (resumable)"
  mark s07nvfp4; reclaim s07nvfp4
fi

# s07b: bit allocation (CPU), then the sequential GPTQ Thor build (GPU), then the optional dense overlay (CPU)
if ! step_done s07thor; then
  $PY ops/s07_quant.py plan > $L/s07_plan.log 2>&1 || stop "s07 plan failed"
  reclaim s07plan
  quiet; setsid $PY ops/s07_quant.py thor --kernels $KERN > $L/s07_thor.log 2>&1 || stop "s07 thor build failed (resumable)"
  mark s07thor; reclaim s07thor
fi
if ! step_done s07overlay; then
  $PY ops/s07_quant.py overlay > $L/s07_overlay.log 2>&1 || stop "s07 overlay failed"
  mark s07overlay; reclaim s07overlay
fi

# s10 final panel: CONFIRM split, every release gate, the pre-registered A/B arm decision (GPU)
if ! step_done s10final; then
  quiet; setsid $PY ops/s10_eval.py final --kernels $KERN > $L/s10_final.log 2>&1 || stop "s10 final panel failed"
  mark s10final; reclaim s10final
fi

# arm B won the matched-bytes decision: its Thor build is a bit-exact copy (CPU)
if ! step_done s07thorb && [ "$($PY -c "import sys,json;sys.path.insert(0,'.');from ops.lib import env;print(json.load(open(env.WORK/'s10'/'gates.json'))['thor_arm'])")" = B ]; then
  $PY ops/s07_quant.py thor-b > $L/s07_thorb.log 2>&1 || stop "s07 thor-b failed"
  mark s07thorb; reclaim s07thorb
fi

# s12: cards from s10 artifacts; a repo goes public only if its pre-registered gate passed (network).
# The GGUF repo is published by re-running s12 once s09 has written $V41_WORK/s09/manifest.json.
if ! step_done s12; then
  $PY ops/s12_publish.py > $L/s12.log 2>&1 || stop "s12 publish failed"
  mark s12
fi
log "chain reached the end of implemented stages (s08 CUDA server / s09 GGUF are owned elsewhere; re-run s12 after s09)"
