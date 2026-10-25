#!/bin/bash
# One-shot (2026-10-25): golden_store soft-failed at 09:37 on a harness path bug (the gate's tokmap fallback is
# "tests/data/engram_tokmap.bin", relative to cuda/, but CHAIN_STEPS runs from the repo root), so chain2 skipped the
# Thor serving gates and went on to quant L3. Fix: engram_tokmap.bin copied into the row store dir (looked up first).
# This script waits for quant L3 (killing it would waste ~6 h), holds chain2, reruns golden_store, and judges it under
# the owner's golden A192 waiver: PASS-under-waiver iff the only FAIL line is the pooled near-tie rule and the store
# run is no worse than the waived run (disagreements <= 46, near-ties/disagreements >= 41/46). Then chain2 restarts
# and runs decode -> ctx1m -> server -> s12 thor -> overlay before the GGUF staging. Otherwise the hold stays.
set -u; R=~/deepseek-v4.1-flash-reap-thor; cd $R; L=ops/logs; D=$L/thor_first; mkdir -p $D
log() { echo "[$(date -Iseconds)] thor-first: $*" | tee -a $D/run.log >> $L/chain.log; }
HOLD="manual hold: golden_store rerun under the A192 waiver (ops/thor_first_after_L3.sh); released by that script"
until [ -f $L/done.s09_quant_L3 ]; do
  [ -f $L/chain.STOP ] && { log "chain.STOP appeared while waiting for quant L3 ($(cat $L/chain.STOP)); not running"; exit 1; }
  systemctl --user is-active -q dsv41-chain2 || { log "chain2 not active while waiting for quant L3; not running"; exit 1; }
  sleep 60
done
echo "$HOLD" > $L/chain.STOP
systemctl --user stop dsv41-chain2; sleep 10
pgrep -x llama-quantize >/dev/null && { log "llama-quantize still running after the stop; chain2 stays held"; exit 1; }
sync; sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches'
nvidia-smi -L >/dev/null 2>&1 || { log "GPU unavailable; chain2 stays held"; exit 1; }
log "quant L3 done; chain2 held; rerunning golden_store A192"
rm -f $L/soft_fail.golden_store
bash cuda/CHAIN_STEPS.sh golden_store A192; rc=$?
G=$L/s08_golden_store_A192.log
cp $G $D/golden_store.log
fails=$(grep -c -- '<-- FAIL' $G); pooled=$(grep -E '^\s+pooled ' $G)
dis=$(echo "$pooled" | awk '{print $2}'); nt=$(echo "$pooled" | awk '{print $4}')
diff <(grep -E '^\s+L[0-9]+ ' $L/s08_golden_A192.log) <(grep -E '^\s+L[0-9]+ ' $G) > $D/layer_diff.txt
log "golden_store rc=$rc; FAIL lines $fails; pooled: ${pooled:-none}; per-layer lines differing from the waived run: $(grep -c '^>' $D/layer_diff.txt)"
ok=0
if [ $rc = 0 ]; then ok=1
elif [ "$fails" = 1 ] && echo "$pooled" | grep -q -- '<-- FAIL' && [ -n "$dis" ] && [ "$dis" -gt 0 ] \
     && [ "$dis" -le 46 ] && [ $(( nt * 46 )) -ge $(( 41 * dis )) ]; then ok=1
  echo "$(date -Is) golden_store A192 passed under the owner's golden A192 waiver: only the pooled near-tie rule failed ($pooled); waived run 46/41" > $L/soft_fail.golden_store.owner-waived
fi
if [ $ok = 1 ]; then
  touch $L/done.c2_soft_golden_store
  cat >> ops/decisions/2026-10-25-golden-a192-drift-amendment.md <<EOF

## golden_store rerun ($(date -Is))

The first golden_store run (09:37) never computed anything: \`engram store: cannot open tests/data/engram_tokmap.bin\`
(a relative fallback path; CHAIN_STEPS runs from the repo root, not cuda/). Fixed by placing engram_tokmap.bin in the
row store directory. Rerun result: rc=$rc, pooled line \`$(echo $pooled)\`, $(grep -c '^>' $D/layer_diff.txt) per-layer
lines differ from the waived golden A192 run (ops/logs/thor_first/layer_diff.txt). Judged under the owner's waiver
(only the pooled rule failed, no worse than 46/41): released to decode -> ctx1m -> server -> Thor publish.
EOF
  if [ "$(cat $L/chain.STOP 2>/dev/null)" = "$HOLD" ]; then rm -f $L/chain.STOP; systemctl --user start dsv41-chain2; log "chain2 released (golden_store OK under the waiver)"
  else log "chain.STOP is not our hold; chain2 left as is"; fi
else
  echo "$(date -Is) rc=$rc: golden_store rerun not covered by the waiver ($fails FAIL lines; $pooled)" > $L/soft_fail.golden_store
  echo "$(date -Is) golden_store rerun FAILED beyond the waiver (see $D/): chain2 held for the owner" >> $L/OWNER_FOLLOWUP
  log "golden_store not covered by the waiver; chain2 stays held"
fi
touch $D/DONE
