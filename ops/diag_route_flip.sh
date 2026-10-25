#!/bin/bash
# One-shot (2026-10-25): hold chain2, run ops/diag_route_flip.py (GPU, doc1k, layers 0..39), release chain2.
set -u; R=~/deepseek-v4.1-flash-reap-thor; cd $R; L=ops/logs; D=$L/diag_route_flip; mkdir -p $D
W=$HOME/models/v41-work
log() { echo "[$(date -Iseconds)] diag-route: $*" | tee -a $D/run.log >> $L/chain.log; }
HOLD="manual hold: routing-flip diagnostic (ops/diag_route_flip.sh); released by that script"
[ -f $L/chain.STOP ] && { log "chain.STOP already present ($(cat $L/chain.STOP)); not running"; exit 1; }
echo "$HOLD" > $L/chain.STOP
systemctl --user stop dsv41-chain2; sleep 10; pkill -x llama-quantize; sleep 5
pgrep -x llama-quantize >/dev/null && { log "llama-quantize still running; chain2 stays held"; exit 1; }
sync; sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches'
nvidia-smi -L >/dev/null 2>&1 || { log "GPU unavailable; chain2 stays held"; exit 1; }
log "start"
.venv/bin/python -I ops/diag_route_flip.py --ref $W/golden/A192 --keepset $W/s04/keepset.json --thor $W/thor/A192 \
  --targets 6:850 9:877 33:988 39:180 39:530 39:721 --cuda-cos $L/diag_golden_a192/row_cos.txt \
  --out $D/result.json > $D/diag.log 2>&1
log "rc=$? ($(grep -c '^\[diag\] L[0-9]* pos' $D/diag.log) targets reported)"
sync; sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches'
if [ "$(cat $L/chain.STOP 2>/dev/null)" = "$HOLD" ]; then rm -f $L/chain.STOP; systemctl --user start dsv41-chain2; log "chain2 released"
else log "chain.STOP is not our hold; chain2 left as is"; fi
touch $D/DONE
