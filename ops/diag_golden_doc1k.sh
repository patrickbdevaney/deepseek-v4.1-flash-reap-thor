#!/bin/bash
# One-shot diagnostic for the golden_src soft fail (2026-10-24, run 4: 41 disagreements, 33 near-ties < 90%, all 8
# confident ones on doc1k). Waits for chain2's s09 convert to finish, holds chain2 (chain.STOP, so the watchdog does
# not restart it), runs the doc1k-only golden gate with the per-row diagnostics (isolated head + confident rows +
# GOLDEN_ROW_COS trajectories), then releases chain2. imatrix, if it had started, re-runs from scratch.
set -u
R=~/deepseek-v4.1-flash-reap-thor; cd $R
L=ops/logs; D=$L/diag_golden_doc1k; mkdir -p $D
log() { echo "[$(date -Iseconds)] $*" | tee -a $D/run.log; }
source ops/lib/quiet.sh
REF=$HOME/models/v41-work/golden/diag_doc1k
mkdir -p $REF; ln -sfn $HOME/models/v41-work/golden/src/doc1k $REF/doc1k
python3 - $HOME/models/v41-work/golden/src/meta.json $REF/meta.json <<'E'
import json, sys
m = json.load(open(sys.argv[1])); m["prompts"] = {"doc1k": m["prompts"]["doc1k"]}
json.dump(m, open(sys.argv[2], "w"))
E
log "waiting for done.s09_convert"
until [ -f $L/done.s09_convert ]; do
  [ -f $L/chain.STOP ] && { log "chain2 STOPped before convert finished: $(cat $L/chain.STOP); running anyway"; break; }
  sleep 10
done
echo "manual hold: golden doc1k diagnostic (ops/diag_golden_doc1k.sh); released by that script" > $L/chain.STOP.hold
[ -f $L/chain.STOP ] || cp $L/chain.STOP.hold $L/chain.STOP
systemctl --user stop dsv41-chain2
log "chain2 held: $(cat $L/chain.STOP)"
for i in $(seq 60); do b=$(gpu_busy); [ -z "$b" ] && break; sleep 10; done
[ -n "$(gpu_busy)" ] && { log "GPU still busy ($(gpu_busy)); not running"; exit 1; }
sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1
log "gate start"
GOLDEN_ROW_COS=$D/row_cos.txt cuda/build/gate_v41_golden --ckpt $HOME/models/DeepSeek-V4.1-Flash-1017 --ref $REF \
  > $D/gate.log 2>&1
log "gate rc=$? ($(tail -1 $D/gate.log))"
if cmp -s $L/chain.STOP $L/chain.STOP.hold; then rm -f $L/chain.STOP; systemctl --user start dsv41-chain2; log "chain2 released"
else log "chain.STOP is not our hold ($(cat $L/chain.STOP)); chain2 left as is"; fi
rm -f $L/chain.STOP.hold
touch $D/DONE
