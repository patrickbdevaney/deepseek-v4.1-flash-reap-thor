#!/bin/bash
# ops/chain2.sh: the post-chain. Restructured 2026-10-21 so the REAP + llama.cpp release is MECHANICAL and never waits
# on the CUDA server (cuda/ compiles but has never run on the GPU; it may need an engineer):
#   HARD (a failure writes chain.STOP): llama.cpp build + backend_ops -> backbone to the Seagate -> Engram row files ->
#        master -> convert (+heal gain) -> imatrix -> GGUF L2 (the 128 GB level; L3 only if L2 fails its gate) ->
#        kld + smoke -> s09 gate -> manifest -> s12 GGUF publish -> s12 final card re-render.
#   SOFT (a failure is recorded in ops/logs/soft_fail.<step> + ops/logs/OWNER_FOLLOWUP, and the chain goes on):
#        s08 CUDA build/gates/golden, the trellis sub-arm (needs gemv_bench), s08 serving gates + the Thor repo, s11.
# Detached: systemd (ops/sd/dsv41-chain2.service) or setsid nohup bash ops/chain2.sh > /dev/null 2>&1 < /dev/null &
# Waits for ops/chain.sh to log "chain reached the end of implemented stages"; exits if ops/logs/chain.STOP exists.
# Same conventions as chain.sh: one model-loading process at a time, quiet() before and reclaim() after every
# model-loading step, done markers (ops/logs/done.c2_* here; the s08/s09 step files keep their own done.s08_*/done.s09_*).
# Every step is idempotent: fix, rm the STOP file (or the soft_fail.<step> file), relaunch. Disk: every large write has
# a df guard. Cleanup deletes only regenerable artifacts under $V41_WORK; the source checkpoint is MOVED (verified copy,
# then symlink) to the Seagate.
cd ~/deepseek-v4.1-flash-reap-thor
PY=.venv/bin/python; L=ops/logs; mkdir -p $L; C=$L/chain.log
WORK=${V41_WORK:-$HOME/models/v41-work}; SRC=${V41_SRC:-$HOME/models/DeepSeek-V4.1-Flash-1017}
G=$WORK/s09; ROWS=$WORK/engram_rows
SEAROOT="/media/patrickd/Seagate Portable Drive"; SEAW="$SEAROOT/v41_staging"       # staging only, never served from
GIB=1073741824
log() { echo "[$(date -Is)] chain2 $*" >> $C; }
stop() { log "STOP: $*"; echo "$(date -Is) chain2: $*" > $L/chain.STOP; exit 1; }
[ -f $L/chain.STOP ] && { log "chain.STOP present, not starting: $(cat $L/chain.STOP)"; exit 1; }
exec 8> /tmp/dsv41-chain2.lock; flock -n 8 || exit 0
step_done() { [ -f $L/done.c2_$1 ]; }; mark() { touch $L/done.c2_$1; log "done $1"; }
reclaim() { sync; sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches'; sleep 5
  a=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo); log "reclaim after $1: MemAvailable ${a} GiB"
  [ "$a" -ge 100 ] || stop "memory did not return to baseline after $1 (${a} GiB < 100)"; }
source ops/lib/quiet.sh       # gpu_busy / leaf_runner_up: match program names, never whole command lines (audit W5)
quiet() { local b
  b=$(docker ps --format '{{.Names}}' 2>/dev/null | grep -E '^(leaf-teacher|vllm)' | head -1)
  [ -n "$b" ] && stop "a vLLM server is up (docker $b)"
  leaf_runner_up && ! [ -f ~/leafllm/run/runner.stop ] && stop "leafllm runner active"
  b=$(gpu_busy); [ -n "$b" ] && stop "another model-loading process is up: $b"; true; }
free_gib() { echo $(( $(df -B1 --output=avail "$1" | tail -1) / GIB )); }
need() { local f; f=$(free_gib "$1"); [ "$f" -ge "$2" ] || stop "disk: $3 needs ${2} GiB free on $1, only ${f} GiB"; log "df ok for $3: ${f} GiB free on $1 (need $2)"; }
gpu() { quiet; "$@" || stop "$*"; reclaim "$*"; }        # a model-loading step
cpu() { "$@" || stop "$*"; }
s08() { bash cuda/CHAIN_STEPS.sh "$@"; }
FOLLOW=$L/OWNER_FOLLOWUP
soft_failed() { [ -f $L/soft_fail.$1 ]; }
soft() {          # soft NAME CMD...: a CUDA/s11 step whose failure must not block the release
  local name=$1; shift; step_done soft_$name && return 0; soft_failed $name && return 1
  quiet; "$@"; local rc=$?; reclaim "soft $name"
  [ $rc = 0 ] && { mark soft_$name; return 0; }
  echo "$(date -Is) rc=$rc: $*" > $L/soft_fail.$name
  echo "$(date -Is) $name failed (rc=$rc): $*  [rm $L/soft_fail.$name to retry]" >> $FOLLOW
  log "SOFT FAIL $name (rc=$rc): $* -- the release path continues"; return 1; }
s09() { bash llamacpp/CHAIN_STEPS.sh "$@"; }
S11_OK=1
s11() { local name=$1; shift; step_done s11_$name && return 0; [ $S11_OK = 1 ] || return 1
  soft_failed s11_$name && { S11_OK=0; return 1; }
  setsid $PY ops/s11_spec.py "$@" >> $L/s11_$name.log 2>&1 < /dev/null; local rc=$?
  if [ $rc != 0 ]; then S11_OK=0; echo "$(date -Is) rc=$rc" > $L/soft_fail.s11_$name
    echo "$(date -Is) s11 $name failed (rc=$rc, see $L/s11_$name.log)  [rm $L/soft_fail.s11_$name to retry]" >> $FOLLOW
    log "SOFT FAIL s11 $name (rc=$rc); the rest of s11 is skipped"; return 1; fi
  mark s11_$name; }
sha_ok() {  # sha_ok FILE OID
  [ "$(sha256sum "$1" | cut -d' ' -f1)" = "$2" ]; }

# ------------------------------------------------------------------------------------------------ wait for chain 1
log "chain2 start (waiting for chain.sh to finish)"
# done.s12 also counts: chain.sh marks it just before logging the end line, and a kill between the two must not wedge us
until grep -q "chain reached the end of implemented stages" $C 2>/dev/null || [ -f $L/done.s12 ]; do
  [ -f $L/chain.STOP ] && { log "chain.STOP appeared while waiting: $(cat $L/chain.STOP)"; exit 1; }
  sleep 300
done
while pgrep -f '^(/usr)?(/bin/)?bash ops/chain\.sh$' >/dev/null; do sleep 30; done
mountpoint -q "$SEAROOT" || stop "the Seagate is not mounted at $SEAROOT"
[ -f $L/done.s10final ] && [ -f $WORK/s10/final/results.json ] || stop "s10final has not passed; chain2 needs it"
NKEEP=$($PY -c "import sys;sys.path.insert(0,'.');from ops.lib import keepset;print(len(keepset.load('$WORK/s04/keepset.json')[0]['0']))") \
  || stop "no s04 keep-set"
KERN=$(cat $L/kernels 2>/dev/null || echo tilelang)
export V41_NKEEP=$NKEEP                                         # the GGUF ladder is always the published REAP<n> master
log "GGUF master REAP$NKEEP, kernels $KERN (the served arm is resolved after the s10 trellis panel)"
mkdir -p "$SEAW"

# ------------------------------------------------------------------------------------------------ phase 0: space
if ! step_done stale_cache; then           # s00's killed first attempt; s00 verified every shard, these are dead partials
  find "$SRC/.cache/huggingface/download" -name '*.incomplete' -delete 2>/dev/null; mark stale_cache
fi
# Dead replay embeddings (s04, s10 heal, s10 final): their runs are finished and nothing re-reads them (~30 GiB on NVMe).
if ! step_done drop_embeds; then
  for d in $WORK/s04/replay/_embed $WORK/s10/heal/replay/_embed $WORK/s10/final/replay/_embed; do
    [ -d $d ] && { rm -rf $d; log "dropped dead replay stream $d"; }
  done
  mark drop_embeds
fi
# A Thor build parked on the Seagate is staging only: it is copied back to NVMe before any GPU step loads it.
verify_build() { $PY - "$1" <<'E2'
import sys
sys.path.insert(0, ".")
from pathlib import Path
from ops.lib import thor_format as TF
bad = TF.Build(Path(sys.argv[1])).verify()
sys.exit(f"bad files: {bad}" if bad else 0)
E2
}
park_build() {   # park_build NAME: thor/NAME -> Seagate (verified copy), NVMe dir replaced by a symlink
  local b=$1; [ -L $WORK/thor/$b ] && return 0
  need "$SEAROOT" $(( $(du -s -B1 $WORK/thor/$b | cut -f1) / GIB + 20 )) "park thor/$b"
  mkdir -p "$SEAW/thor"; rsync -a --partial $WORK/thor/$b/ "$SEAW/thor/$b/" || stop "copy thor/$b"
  verify_build "$SEAW/thor/$b" || stop "thor/$b Seagate copy does not verify"
  rm -rf $WORK/thor/$b && ln -s "$SEAW/thor/$b" $WORK/thor/$b; log "thor/$b parked on the Seagate (verified)"; }
unpark_build() {  # unpark_build NAME: a parked build back on NVMe (verified) before it is served
  local b=$1; [ -L $WORK/thor/$b ] || return 0
  need / $(( $(du -s -B1 -L $WORK/thor/$b | cut -f1) / GIB + 25 )) "unpark thor/$b"
  rsync -a --partial "$SEAW/thor/$b/" $WORK/thor/.$b.unpark/ || stop "copy back thor/$b"
  verify_build $WORK/thor/.$b.unpark || stop "thor/$b NVMe copy does not verify"
  rm -f $WORK/thor/$b && mv $WORK/thor/.$b.unpark $WORK/thor/$b; log "thor/$b back on NVMe (verified)"; }

# ------------------------------------------------------------------------------------------------ phase A: source on NVMe
# build markers may be left by an earlier QA run against an older tree: rebuild once here (incremental, minutes)
step_done s09_build_ || { rm -f $L/done.s09_build $L/done.s08_build $L/done.s08_cpu_gates; cpu s09 build; mark s09_build_; }
gpu s09 backend_ops
CUDA_OK=1
{ soft s08_build s08 build && soft s08_cpu_gates s08 cpu_gates && soft s08_units s08 units && soft s08_gemv s08 gemv_bench; } || CUDA_OK=0

# ------------------------------------------------------------------------------------------------ s07/s10 trellis sub-arm (soft)
# Pre-registered (s10_gates.json "trellis"): the KT2/KT3 build of arm A at <= the K-quant build's expert bytes. Its
# panel needs s08's gemv_bench.json (a missing bench means not adopted), so without a working CUDA build it is skipped.
NA=$NKEEP; KQ=$WORK/thor/A$NA
trellis() {
  unpark_build A$NA                 # s07 thor-b parks A on the Seagate when B wins; the panel must not mmap it over USB
  [ -f $KQ/thor.json ] || { echo "no K-quant arm-A build at $KQ"; return 1; }
  if ! step_done s07trellis; then
    ktneed=$(( $(du -s -B1 -L $KQ | cut -f1) / GIB + 60 ))
    if [ "$(free_gib /)" -lt $ktneed ]; then
      for d in $WORK/thor/B*/; do [ -f $d/thor.json ] && park_build $(basename $d); done
    fi
    [ "$(free_gib /)" -ge $ktneed ] || { echo "disk: trellis needs $ktneed GiB"; return 1; }
    setsid $PY ops/s07_quant.py thor-trellis --kernels $KERN > $L/s07_trellis.log 2>&1 < /dev/null || return 1
    reclaim s07trellis; mark s07trellis
  fi
  quiet; setsid $PY ops/s10_eval.py trellis --kernels $KERN > $L/s10_trellis.log 2>&1 < /dev/null; }
if [ $CUDA_OK = 1 ]; then soft trellis trellis || true
else log "trellis sub-arm skipped: the CUDA build/gates did not pass (no gemv_bench)"; fi

# ------------------------------------------------------------------------------------------------ the served arm
# s10/gates.json thor_build_dir (written by the trellis panel: A<n>, A<n>kt or B<n>), else thor_arm (ops/s11_spec.served)
ARM=$($PY -c "import sys;sys.path.insert(0,'.');from ops.s11_spec import served;print(served()['name'])" 2>>$L/c2_arm.log) \
  || { ARM=A$NKEEP; log "served arm not resolvable; using $ARM for disk placement only"; }
unpark_build $ARM
log "served arm $ARM ($(readlink -f $WORK/thor/$ARM))"
for d in $WORK/thor/*/; do        # EVERY losing build (up to two of A<n>, A<n>kt, B<n>): kept, off NVMe
  b=$(basename $d); [ "$b" = "$ARM" ] && continue; [ -f $d/thor.json ] || continue
  step_done move_thor_$b && continue
  park_build $b; mark move_thor_$b
done

if [ $CUDA_OK = 1 ]; then         # the golden gates read the source checkpoint, so they run before the backbone moves
  { soft golden_ref_src s08 golden_ref src && soft golden_src s08 golden src \
    && soft golden_ref_arm s08 golden_ref $ARM && soft golden_arm s08 golden $ARM; } || CUDA_OK=0
fi

# ------------------------------------------------------------------------------------------------ move the backbone
# Shards 1-43 (the decoder backbone, ~279 GiB) -> Seagate, sha256-verified against the pinned tree, then symlinked.
# 44-46 (the DSpark head, 7.4 GiB; s11 reads them repeatedly) and the small files stay. From here on the source is
# read only by s11_tapdump (~36 min per pass over USB) and s11's head loading (embed/head).
if ! step_done move_backbone; then
  quiet
  tot=0; for i in $(seq 1 43); do f=$SRC/model-$(printf %05d $i)-of-00048.safetensors; [ -L $f ] || tot=$((tot + $(stat -c%s $f))); done
  need "$SEAROOT" $(( tot / GIB + 20 )) "move the source backbone"
  mkdir -p "$SEAW/backbone"
  for i in $(seq 1 43); do
    s=model-$(printf %05d $i)-of-00048.safetensors
    [ -L "$SRC/$s" ] && continue
    oid=$(python3 -c "import json;print([x['lfs']['oid'] for x in json.load(open('$SRC/../v41_1017_tree.json')) if x['path']=='$s'][0])")
    rsync -t --partial "$SRC/$s" "$SEAW/backbone/$s" || stop "backbone copy $s"
    sha_ok "$SEAW/backbone/$s" $oid || stop "backbone: Seagate copy of $s does not match the pinned sha256"
    rm -f "$SRC/$s" && ln -s "$SEAW/backbone/$s" "$SRC/$s"
  done
  log "backbone on the Seagate: $(( tot / GIB )) GiB freed on NVMe"; mark move_backbone
fi

# ------------------------------------------------------------------------------------------------ phase B: rows + serving gates
# (the Engram row files are built after the source GGUF is dropped: see phase C)

# ------------------------------------------------------------------------------------------------ phase C: GGUF ladder
MASTER="$SEAW/master_REAP$NKEEP"
LOCAL=$($PY -c "import json;print(json.load(open('$WORK/s05/manifest_REAP$NKEEP.json')).get('local_dir') or '')") || stop "no s05 manifest"
if [ -n "$LOCAL" ] && [ -f "$LOCAL/reap_keepset.json" ]; then MASTER=$LOCAL; step_done master || mark master; fi   # upload was off
if ! step_done master; then        # s05 deleted the local master after its private upload; fetch it back (download: xet HP on)
  need "$SEAROOT" 170 "master download"
  REPO=$($PY -c "import json;print(json.load(open('$WORK/s05/manifest_REAP$NKEEP.json'))['repo'])") || stop "no s05 manifest"
  setsid .venv/bin/hf download "$REPO" --local-dir "$MASTER" >> $L/c2_master.log 2>&1 \
    || stop "master download $REPO"
  $PY - "$MASTER" "$WORK/s05/manifest_REAP$NKEEP.json" <<'E' >> $L/c2_master.log 2>&1 || stop "master download does not verify against the s05 manifest"
import hashlib, json, sys
from pathlib import Path
d, m = Path(sys.argv[1]), json.load(open(sys.argv[2]))
files = m["files"]
items = files.items() if isinstance(files, dict) else ((f["path"], f) for f in files)
bad = []
for name, rec in items:
    p = d / Path(name).name
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        while b := fh.read(1 << 24):
            h.update(b)
    if h.hexdigest() != rec["sha256"]:
        bad.append(name)
print("master verify:", "OK" if not bad else bad)
sys.exit(1 if bad else 0)
E
  [ -f "$MASTER/reap_keepset.json" ] || stop "master has no reap_keepset.json"
  mark master
fi
# Disk order (ops/logs/audit/ops.md D1/D2): convert -> imatrix -> quant L2 and L3 onto the Seagate -> drop the source
# GGUF (regenerable) -> Engram row files on NVMe -> stage one level to NVMe for kld + smoke -> unstage -> gate.
# L2 (~2.85 bpw, the 128 GB level) is the release; L3 (~2.46 bpw) is its fallback if L2 fails its gate, and
# otherwise a soft add-on published next to L2 once the release is out (owner decision 2026-10-22).
export MASTER S09_QUANT_ROOT="$SEAW/s09" S09_LEVELS="L2 L3"
quants_done() { [ -f $L/done.s09_quant_L2 ] && [ -f $L/done.s09_quant_L3 ]; }
if ! quants_done; then
  [ -f $L/done.s09_convert ] || need / 170 "s09 source GGUF (~152 GiB)"; gpu s09 convert
  cpu s09 imatrix
  for LV in L2 L3; do [ -f $L/done.s09_quant_$LV ] || { need "$SEAROOT" 250 "quant $LV"; cpu s09 quant $LV; }; done
fi
cpu s09 drop_source
step_done s09_rowfile_ || { [ -f $L/done.s09_rowfile ] || need / 200 "s09 Engram row files (~189 GiB)"; cpu s09 rowfile; mark s09_rowfile_; }
evaluate() { local LV=$1
  if ! { [ -f $L/done.s09_kld_$LV ] && [ -f $L/done.s09_smoke_$LV ]; }; then
    cpu s09 stage $LV; gpu s09 kld $LV; gpu s09 smoke $LV; fi
  cpu s09 unstage $LV; }
evaluate L2
# the verdict is read from gates.json (stdout carries log lines): a crashed gate STOPs instead of reading as "L2 failed"
gate() { rm -f $G/gates.json; $PY ops/s09_gate.py --levels "$@" >> $L/c2_s09_gate.log 2>&1 || stop "s09 gate crashed (see $L/c2_s09_gate.log)"
  PASS=$($PY -c "import json;print(' '.join(json.load(open('$G/gates.json'))['passing']))") || stop "s09 gate wrote no $G/gates.json"; }
# once the L3 add-on has measured L3, every re-run gates both, so gates.json never drops a published L3
L3M=$( [ -f $L/done.s09_kld_L3 ] && [ -f $L/done.s09_smoke_L3 ] && echo L3)
gate L2 $L3M
if [ -z "$PASS" ]; then
  log "GGUF L2 failed its gate; evaluating the L3 fallback"; evaluate L3
  gate L2 L3
fi
log "s09 gate: passing levels [${PASS}]"
if [ -n "$PASS" ]; then
  step_done s09_manifest || { $PY ops/s09_gguf_ladder.py manifest --levels $PASS > $L/s09_manifest.log 2>&1 || stop "s09 manifest"; mark s09_manifest; }
  if ! step_done s12_gguf; then
    env -u HF_XET_HIGH_PERFORMANCE $PY ops/s12_publish.py --only gguf > $L/s12_gguf.log 2>&1 || stop "s12 gguf publish"
    mark s12_gguf
  fi
else
  log "no GGUF level passed its gate: nothing published (recorded in $G/gates.json)"
  echo "$(date -Is) no GGUF level passed its gate (see $G/gates.json): nothing published" >> $FOLLOW
fi
# The release (MXFP4 master from chain 1 + the GGUF) is out; refresh its cards now, before any soft work.
if ! step_done s12_release; then
  env -u HF_XET_HIGH_PERFORMANCE $PY ops/s12_publish.py --only mxfp4,nvfp4,gguf > $L/s12_release.log 2>&1 || stop "s12 release re-render"
  mark s12_release
fi
log "RELEASE DONE: REAP + GGUF published per their gates; CUDA server and s11 follow (soft)"

# ------------------------------------------------------------------------------------------------ GGUF L3 add-on (soft)
# Only when L2 passed (a failed L2 already evaluated L3 as its fallback above). L3 goes through the same s09 gate; a
# pass re-writes the manifest with both levels and s12 uploads only the new files (its ledger skips L2's) + the card.
gguf_l3() {
  if ! { [ -f $L/done.s09_kld_L3 ] && [ -f $L/done.s09_smoke_L3 ]; }; then
    { s09 stage L3 && s09 kld L3 && reclaim "kld L3" && quiet && s09 smoke L3; } || { s09 unstage L3; return 1; }
  fi
  s09 unstage L3 || return 1
  rm -f $G/gates.json; $PY ops/s09_gate.py --levels L2 L3 >> $L/c2_s09_gate.log 2>&1 || return 1
  PASS=$($PY -c "import json;print(' '.join(json.load(open('$G/gates.json'))['passing']))") || return 1
  [ "$PASS" = "L2 L3" ] || { log "GGUF L3 did not pass its gate (passing: [$PASS]); L2 stays the only level"; return 1; }
  $PY ops/s09_gguf_ladder.py manifest --levels L2 L3 > $L/s09_manifest_l3.log 2>&1 || return 1
  env -u HF_XET_HIGH_PERFORMANCE $PY ops/s12_publish.py --only gguf > $L/s12_gguf_l3.log 2>&1; }
if [ "$PASS" = L2 ]; then soft gguf_l3 gguf_l3; fi

# ------------------------------------------------------------------------------------------------ DSpark sidecar (soft)
# The checkpoint's own drafter as an opt-in dspark-*.gguf at the GGUF repo root (ops/s09_dspark.py, gate dspark_gguf):
# convert (patch 0005) -> verify every tensor against the pinned reference drafter -> bench beside the largest passing
# level it fits next to (L2 + 7.4 GiB is at the 128 GB edge) -> gate -> s12 uploads it (pass) and the card says so.
dspark() { local lv rc
  $PY ops/s09_dspark.py convert >> $L/s09_dspark.log 2>&1 || return 1
  $PY ops/s09_dspark.py verify  >> $L/s09_dspark.log 2>&1 || return 1
  if [ ! -f $G/dspark/bench.json ]; then
    lv=$($PY ops/s09_dspark.py pick --passing $PASS 2>> $L/s09_dspark.log | tail -1)
    if [ -n "$lv" ]; then
      s09 stage $lv || return 1
      quiet; $PY ops/s09_dspark.py bench $lv >> $L/s09_dspark.log 2>&1; rc=$?
      s09 unstage $lv; reclaim "dspark bench $lv"
      [ $rc = 0 ] || return 1
    else log "DSpark: no passing level [$PASS] fits beside the drafter; not benched"; fi
  fi
  $PY ops/s09_dspark.py gate >> $L/s09_dspark.log 2>&1 || return 1
  env -u HF_XET_HIGH_PERFORMANCE $PY ops/s12_publish.py --only gguf > $L/s12_gguf_dspark.log 2>&1; }
if [ -n "$PASS" ]; then soft dspark dspark; fi

# ------------------------------------------------------------------------------------------------ CUDA serving gates (soft)
if [ $CUDA_OK = 1 ]; then
  { soft golden_store s08 golden_store $ARM && soft decode s08 decode $ARM && soft ctx1m s08 ctx1m $ARM \
    && soft server s08 server $ARM; } || CUDA_OK=0
fi
if [ $CUDA_OK = 1 ] && ! step_done s12_thor; then   # the served Thor build + runtime, now that s08 decoded it
  env -u HF_XET_HIGH_PERFORMANCE $PY ops/s12_publish.py --only thor > $L/s12_thor.log 2>&1 || stop "s12 thor publish"
  mark s12_thor
fi

# ------------------------------------------------------------------------------------------------ s11: the speculative head
quiet; s11 traces_eval traces --set eval;   reclaim s11_traces_eval
quiet; s11 traces_train traces --set train; reclaim s11_traces_train
quiet; s11 tapdump_eval tapdump --set eval --kernels $KERN;   reclaim s11_tapdump_eval
quiet; s11 tapdump_train tapdump --set train --kernels $KERN; reclaim s11_tapdump_train
quiet; s11 bank bank --kernels $KERN; reclaim s11_bank
s11 prune prune
s11 quant quant; reclaim s11_quant
for A in separate inmodel; do quiet; s11 accept_${A}_native accept --arm $A --head native --kernels $KERN; reclaim s11_accept_${A}_native; done
for A in separate inmodel; do quiet; s11 refit_$A refit --arm $A; reclaim s11_refit_$A; done
for A in separate inmodel; do quiet; s11 accept_${A}_tuned accept --arm $A --head tuned --kernels $KERN; reclaim s11_accept_${A}_tuned; done
quiet; s11 curve curve; reclaim s11_curve
# s08 speculation on the build's own mtp.* head: greedy ids with --spec K == without (a GROSS mismatch stops here:
# a verify/rollback bug, not a head problem), then tok/s K=1..4 at 128 and 8k vs no-spec. s11 e2e runs the same
# spec_bench step per s11 head (cuda/CHAIN_STEPS.sh spec_bench ARM DIR TAG).
if [ $CUDA_OK = 1 ]; then
  { soft spec_identity s08 spec_identity $ARM && soft spec_bench s08 spec_bench $ARM; } || CUDA_OK=0
fi
if [ $CUDA_OK = 1 ]; then quiet; s11 e2e e2e; reclaim s11_e2e
else log "s11 e2e skipped: no verified CUDA spec path"; fi
s11 decide decide
if [ $S11_OK = 1 ] && [ -f $WORK/s11/decision.json ] && ! step_done s11_publish; then
  env -u HF_XET_HIGH_PERFORMANCE $PY ops/s11_spec.py publish >> $L/s11_publish.log 2>&1 \
    && mark s11_publish || { echo "$(date -Is) s11 publish failed (see $L/s11_publish.log)" >> $FOLLOW; log "SOFT FAIL s11 publish"; }
fi

# ------------------------------------------------------------------------------------------------ cards carry the s11 result
if ! step_done s12_final; then
  env -u HF_XET_HIGH_PERFORMANCE $PY ops/s12_publish.py > $L/s12_final.log 2>&1 || stop "s12 final re-render"
  mark s12_final
fi
ST11=$($PY -c "import json;print(json.load(open('$WORK/s11/decision.json'))['status'])" 2>/dev/null || echo none)
if [ "$ST11" = blocked ]; then      # a decision for the owner, not a crash: the CUDA server has no draft/verify path
  echo "$(date -Is) s11 end-to-end gate BLOCKED: no verified DSpark draft+verify path in cuda/ (see $WORK/s11/decision.json)" >> $FOLLOW
fi
# leafllm stays paused while OWNER_FOLLOWUP lists Thor work (ops/sd/watchdog.py checks it)
log "chain2 reached the end: s11 decision $ST11, CUDA_OK=$CUDA_OK, follow-ups: $( [ -f $FOLLOW ] && wc -l < $FOLLOW || echo 0)"
