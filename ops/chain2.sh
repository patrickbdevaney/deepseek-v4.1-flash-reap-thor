#!/bin/bash
# ops/chain2.sh: the post-chain. s07/s10 trellis -> s08 CUDA server gates -> s12 (Thor repo) -> s09 GGUF ladder -> s12 (GGUF repo)
# -> s11 speculative head (s08 spec_identity/spec_bench on the native head, then s11 e2e per head) -> s12.
# Detached: setsid nohup bash ops/chain2.sh > /dev/null 2>&1 < /dev/null &
# Waits for ops/chain.sh to log "chain reached the end of implemented stages"; exits if ops/logs/chain.STOP exists.
# Same conventions as chain.sh: one model-loading process at a time, quiet() before and reclaim() after every
# model-loading step, done markers (ops/logs/done.c2_* here; the s08/s09 step files keep their own done.s08_*/done.s09_*),
# and a failed step writes ops/logs/chain.STOP (prefixed "chain2:") and exits. Every step is idempotent: fix, rm the
# STOP file, relaunch. Disk: the space plan is in ops/README.md ("chain2 disk plan"); every large write has a df guard.
# Cleanup deletes only regenerable artifacts under $V41_WORK; the source checkpoint is MOVED (verified copy, then
# symlink) to the Seagate, and only after s10final.
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
quiet() { docker ps --format '{{.Names}}' | grep -q -E '^(leaf-teacher|vllm)' && stop "a vLLM server is up"
  pgrep -f pipeline/runner.sh >/dev/null && ! [ -f ~/leafllm/run/runner.stop ] && stop "leafllm runner active"
  pgrep -f 'llama-server|v41-teacher-kld|v41_server|v41_decode|gate_v41|ref_dump|s11_tapdump|s11_spec|s03_saliency|s10_eval' >/dev/null \
    && stop "another model-loading process is up"; true; }
free_gib() { echo $(( $(df -B1 --output=avail "$1" | tail -1) / GIB )); }
need() { local f; f=$(free_gib "$1"); [ "$f" -ge "$2" ] || stop "disk: $3 needs ${2} GiB free on $1, only ${f} GiB"; log "df ok for $3: ${f} GiB free on $1 (need $2)"; }
gpu() { quiet; "$@" || stop "$*"; reclaim "$*"; }        # a model-loading step
cpu() { "$@" || stop "$*"; }
s08() { bash cuda/CHAIN_STEPS.sh "$@"; }
s09() { bash llamacpp/CHAIN_STEPS.sh "$@"; }
s11() { local name=$1; shift; step_done s11_$name && return 0
  setsid $PY ops/s11_spec.py "$@" >> $L/s11_$name.log 2>&1 < /dev/null; local rc=$?
  [ $rc = 0 ] || stop "s11 $name failed rc=$rc (see $L/s11_$name.log)"; mark s11_$name; }
sha_ok() {  # sha_ok FILE OID
  [ "$(sha256sum "$1" | cut -d' ' -f1)" = "$2" ]; }

# ------------------------------------------------------------------------------------------------ wait for chain 1
log "chain2 start (waiting for chain.sh to finish)"
until grep -q "chain reached the end of implemented stages" $C 2>/dev/null; do
  [ -f $L/chain.STOP ] && { log "chain.STOP appeared while waiting: $(cat $L/chain.STOP)"; exit 1; }
  sleep 300
done
while pgrep -f 'bash ops/chain.sh' >/dev/null; do sleep 30; done
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
step_done s09_build_ || { cpu s09 build; mark s09_build_; }
step_done s08_build_ || { cpu s08 build; mark s08_build_; }
step_done s08_cpu_gates_ || { cpu s08 cpu_gates; mark s08_cpu_gates_; }
gpu s08 units
gpu s08 gemv_bench

# ------------------------------------------------------------------------------------------------ s07/s10 trellis sub-arm
# Pre-registered (s10_gates.json "trellis"): the KT2/KT3 build of arm A at <= the K-quant build's expert bytes. Both
# read the source and replay it, so they run here, before the backbone move; the s10 panel needs gemv_bench.json.
NA=$NKEEP; KQ=$WORK/thor/A$NA
[ -f $KQ/thor.json ] || stop "no K-quant arm-A build at $KQ (chain 1 s07thor)"
if ! step_done s07trellis; then
  ktneed=$(( $(du -s -B1 -L $KQ | cut -f1) / GIB + 60 ))           # a second ~100 GiB build + its replay streams
  if [ "$(free_gib /)" -lt $ktneed ]; then                         # arm B won s10final: its build is not read by the panel
    for d in $WORK/thor/B*/; do [ -f $d/thor.json ] && park_build $(basename $d); done
  fi
  need / $ktneed "s07 thor-trellis (second Thor build)"
  quiet; setsid $PY ops/s07_quant.py thor-trellis --kernels $KERN > $L/s07_trellis.log 2>&1 < /dev/null \
    || stop "s07 thor-trellis failed (resumable; see $L/s07_trellis.log)"
  reclaim s07trellis; mark s07trellis
fi
if ! step_done s10trellis; then
  [ -f $WORK/s08/gemv_bench.json ] || stop "s10 trellis needs $WORK/s08/gemv_bench.json (a missing bench means not adopted)"
  quiet; setsid $PY ops/s10_eval.py trellis --kernels $KERN > $L/s10_trellis.log 2>&1 < /dev/null \
    || stop "s10 trellis panel failed (see $L/s10_trellis.log)"
  reclaim s10trellis; mark s10trellis
fi

# ------------------------------------------------------------------------------------------------ the served arm
# s10/gates.json thor_build_dir (written by the trellis panel: A<n>, A<n>kt or B<n>), else thor_arm (ops/s11_spec.served)
ARM=$($PY -c "import sys;sys.path.insert(0,'.');from ops.s11_spec import served;print(served()['name'])") \
  || stop "the served Thor arm (s10/gates.json thor_build_dir/thor_arm + a complete thor build) is not readable"
unpark_build $ARM
log "served arm $ARM ($(readlink -f $WORK/thor/$ARM))"
for d in $WORK/thor/*/; do        # EVERY losing build (up to two of A<n>, A<n>kt, B<n>): kept, off NVMe
  b=$(basename $d); [ "$b" = "$ARM" ] && continue; [ -f $d/thor.json ] || continue
  step_done move_thor_$b && continue
  park_build $b; mark move_thor_$b
done

gpu s09 backend_ops
gpu s08 golden_ref src
gpu s08 golden src
gpu s08 golden_ref $ARM
gpu s08 golden $ARM

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
step_done s09_rowfile_ || { [ -f $L/done.s09_rowfile ] || need / 200 "s09 Engram row files (~189 GiB)"; cpu s09 rowfile; mark s09_rowfile_; }
gpu s08 golden_store $ARM
gpu s08 decode $ARM
gpu s08 ctx1m $ARM
gpu s08 server $ARM
if ! step_done s12_thor; then      # the served Thor build, now that s08 has decoded it (gate: thor + fits the budget)
  env -u HF_XET_HIGH_PERFORMANCE $PY ops/s12_publish.py --only thor > $L/s12_thor.log 2>&1 || stop "s12 thor publish"
  mark s12_thor
fi

# ------------------------------------------------------------------------------------------------ phase C: GGUF ladder
MASTER="$SEAW/master_REAP$NKEEP"
LOCAL=$($PY -c "import json;print(json.load(open('$WORK/s05/manifest_REAP$NKEEP.json')).get('local_dir') or '')") || stop "no s05 manifest"
if [ -n "$LOCAL" ] && [ -f "$LOCAL/reap_keepset.json" ]; then MASTER=$LOCAL; step_done master || mark master; fi   # upload was off
if ! step_done master; then        # s05 deleted the local master after its private upload; fetch it back (download: xet HP on)
  need "$SEAROOT" 170 "master download"
  REPO=$($PY -c "import json;print(json.load(open('$WORK/s05/manifest_REAP$NKEEP.json'))['repo'])") || stop "no s05 manifest"
  HF_XET_HIGH_PERFORMANCE=1 setsid .venv/bin/hf download "$REPO" --local-dir "$MASTER" >> $L/c2_master.log 2>&1 \
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
export MASTER
step_done s09_convert_ || { [ -f $L/done.s09_convert ] || need / 170 "s09 source GGUF (~152 GiB)"; gpu s09 convert; mark s09_convert_; }
step_done s09_imatrix_ || { cpu s09 imatrix; mark s09_imatrix_; }
for LV in L0 L1 L2 L3; do         # levels are written to the Seagate (2x a level at peak: unsplit + splits)
  [ -f $L/done.s09_quant_$LV ] && continue
  need "$SEAROOT" 320 "quant $LV"
  mkdir -p "$SEAW/s09/$LV" $G
  if [ -d $G/$LV ] && ! [ -L $G/$LV ]; then rmdir $G/$LV 2>/dev/null || stop "$G/$LV is a real non-empty directory (expected the Seagate symlink)"; fi
  ln -sfn "$SEAW/s09/$LV" $G/$LV
  cpu s09 quant $LV
done
if ! step_done drop_src_gguf; then    # regenerable (convert), and the largest thing on NVMe in this phase
  rm -f $G/src/V4.1-1017-REAP$NKEEP-src.gguf; mark drop_src_gguf
fi
for LV in L0 L1 L2 L3; do         # kld + smoke from an NVMe copy (fast load; partial offload from USB is refused)
  [ -f $L/done.s09_kld_$LV ] && [ -f $L/done.s09_smoke_$LV ] && continue
  ST=$G/stage/$LV
  if [ "$(readlink -f $G/$LV)" != "$(readlink -f $ST)" ]; then
    sz=$(du -s -B1 -L "$SEAW/s09/$LV" | cut -f1)
    need / $(( sz / GIB + 25 )) "stage $LV to NVMe"
    mkdir -p $ST; rsync -a --partial "$SEAW/s09/$LV/" $ST/ || stop "stage $LV"
    ln -sfn $ST $G/$LV
  fi
  gpu s09 kld $LV
  gpu s09 smoke $LV
  rsync -a $ST/teacher_eval/ "$SEAW/s09/$LV/teacher_eval/" || stop "unstage $LV results"
  ln -sfn "$SEAW/s09/$LV" $G/$LV && rm -rf $ST
done
PASS=$($PY ops/s09_gate.py 2>>$L/c2_s09_gate.log | tail -1) || stop "s09 gate"
log "s09 gate: passing levels [${PASS}]"
if [ -n "$PASS" ]; then
  step_done s09_manifest || { $PY ops/s09_gguf_ladder.py manifest --levels $PASS > $L/s09_manifest.log 2>&1 || stop "s09 manifest"; mark s09_manifest; }
  if ! step_done s12_gguf; then
    env -u HF_XET_HIGH_PERFORMANCE $PY ops/s12_publish.py --only gguf > $L/s12_gguf.log 2>&1 || stop "s12 gguf publish"
    mark s12_gguf
  fi
else
  log "no GGUF level passed its gate: nothing published (recorded in $G/gates.json)"
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
gpu s08 spec_identity $ARM
gpu s08 spec_bench $ARM
quiet; s11 e2e e2e; reclaim s11_e2e
s11 decide decide
env -u HF_XET_HIGH_PERFORMANCE $PY ops/s11_spec.py publish >> $L/s11_publish.log 2>&1 || stop "s11 publish"

# ------------------------------------------------------------------------------------------------ cards carry the s11 result
if ! step_done s12_final; then
  env -u HF_XET_HIGH_PERFORMANCE $PY ops/s12_publish.py > $L/s12_final.log 2>&1 || stop "s12 final re-render"
  mark s12_final
fi
ST11=$($PY -c "import json;print(json.load(open('$WORK/s11/decision.json'))['status'])")
log "chain2 reached the end: s11 decision $ST11"
if [ "$ST11" = blocked ]; then      # a decision for the owner, not a crash: the CUDA server has no draft/verify path
  stop "s11 end-to-end gate BLOCKED: cuda/ has no DSpark draft+verify path (offline acceptance and refit are done; see $WORK/s11/decision.json). leafllm stays paused: Thor-exclusive work remains"
fi
