#!/bin/bash
# One-shot (2026-10-25): per-row golden diagnostic for A192 (doc1k, where all confident disagreements are), run in
# the gap after s09 quant L2 while chain2 is held. Then releases chain2 (which resumes the GGUF ladder).
set -u; R=~/deepseek-v4.1-flash-reap-thor; cd $R; L=ops/logs; D=$L/diag_golden_a192; mkdir -p $D
log() { echo "[$(date -Iseconds)] diag-a192: $*" | tee -a $D/run.log >> $L/chain.log; }
HOLD="manual hold: golden A192 diagnostic (ops/diag_golden_a192.sh); released by that script"
until [ -f $L/done.s09_quant_L2 ]; do
  systemctl --user is-active -q dsv41-chain2 || { log "chain2 stopped before quant L2 finished; diagnostic not run"; exit 1; }
  sleep 30; done
[ -f $L/chain.STOP ] && { log "chain.STOP already present ($(cat $L/chain.STOP)); not running"; exit 1; }
echo "$HOLD" > $L/chain.STOP
systemctl --user stop dsv41-chain2; sleep 10; pkill -f llama-quantize; sleep 5
sync; sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches'
nvidia-smi -L >/dev/null 2>&1 || { log "GPU unavailable; chain2 stays held"; exit 1; }
REF=$HOME/models/v41-work/golden/diag_a192_doc1k; mkdir -p $REF
ln -sfn $HOME/models/v41-work/golden/A192/doc1k $REF/doc1k
python3 - $HOME/models/v41-work/golden/A192/meta.json $REF/meta.json <<'P'
import json, sys
m = json.load(open(sys.argv[1])); m["prompts"] = {"doc1k": m["prompts"]["doc1k"]}
json.dump(m, open(sys.argv[2], "w"))
P
log "gate start"
GOLDEN_ROW_COS=$D/row_cos.txt cuda/build/gate_v41_golden --ckpt $HOME/models/v41-work/thor/A192 --ref $REF \
  --keepset $HOME/models/v41-work/s04/keepset.json > $D/gate.log 2>&1
log "gate rc=$? ($(tail -1 $D/gate.log))"
sync; sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches'
if [ "$(cat $L/chain.STOP 2>/dev/null)" = "$HOLD" ]; then rm -f $L/chain.STOP; systemctl --user start dsv41-chain2; log "chain2 released"
else log "chain.STOP is not our hold; chain2 left as is"; fi
touch $D/DONE
