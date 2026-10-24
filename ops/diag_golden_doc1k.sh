#!/bin/bash
# One-shot diagnostic for the golden_src soft fail (2026-10-24, run 4: 41 disagreements, 33 near-ties < 90%, all 8
# confident ones on doc1k): the doc1k-only golden gate with the per-row diagnostics (isolated head + confident rows +
# GOLDEN_ROW_COS trajectories), run while chain2 is held by chain.STOP (so the watchdog does not restart it).
#   pre-reboot  (ops/diag_golden_doc1k.sh reboot): owner OK'd a reboot for the wedged GPU (2026-10-24). Waits for
#               chain2's s09 convert, holds chain2, enables the boot unit, reboots.
#   boot        (no argument, from dsv41-diag-golden-boot.service): waits for the GPU and the Seagate, runs the gate,
#               releases chain2, disables the boot unit.
set -u
R=~/deepseek-v4.1-flash-reap-thor; cd $R
L=ops/logs; D=$L/diag_golden_doc1k; mkdir -p $D
log() { echo "[$(date -Iseconds)] $*" | tee -a $D/run.log; }
HOLD="manual hold: golden doc1k diagnostic (ops/diag_golden_doc1k.sh); released by that script"
UNIT=~/.config/systemd/user/dsv41-diag-golden-boot.service
SEA="/media/patrickd/Seagate Portable Drive"

if [ "${1:-}" = reboot ]; then
  log "pre-reboot: waiting for done.s09_convert and chain2 to stop"
  until [ -f $L/done.s09_convert ] && ! systemctl --user is-active -q dsv41-chain2; do
    if [ -f $L/chain.STOP ] && ! grep -q "GPU unavailable" $L/chain.STOP && ! systemctl --user is-active -q dsv41-chain2; then
      log "chain2 STOPped for another reason: $(cat $L/chain.STOP); rebooting anyway (the GPU is wedged), STOP kept"
      break
    fi
    sleep 30
  done
  if [ ! -f $L/chain.STOP ] || grep -q "GPU unavailable" $L/chain.STOP; then
    [ -f $L/chain.STOP ] && mv $L/chain.STOP $L/stop_history/chain.STOP.gpu-wedge-$(date +%H%M)
    echo "$HOLD" > $L/chain.STOP
  fi
  echo "$HOLD" > $L/chain.STOP.hold
  cat > $UNIT <<E
[Unit]
Description=DSV4.1 golden doc1k diagnostic after the GPU-wedge reboot (one-shot, disables itself)
[Service]
Type=oneshot
WorkingDirectory=$R
ExecStart=/bin/bash $R/ops/diag_golden_doc1k.sh
TimeoutStartSec=4h
[Install]
WantedBy=default.target
E
  systemctl --user daemon-reload && systemctl --user enable dsv41-diag-golden-boot.service
  echo "[$(date -Iseconds)] manual: rebooting to recover the wedged GPU (owner OK 2026-10-24); golden doc1k diagnostic runs at boot, then releases chain2" >> $L/chain.log
  log "rebooting"; sync; sudo -n systemctl reboot
  exit 0
fi

finish() { systemctl --user disable dsv41-diag-golden-boot.service 2>/dev/null; rm -f $UNIT; systemctl --user daemon-reload; }
log "boot: waiting for the GPU and the Seagate"
for i in $(seq 60); do nvidia-smi -L >/dev/null 2>&1 && break; sleep 10; done
nvidia-smi -L >/dev/null 2>&1 || { log "GPU still unavailable after boot; chain2 stays held"; echo "$(date -Is) GPU unavailable after the reboot" >> $L/NEEDS_OWNER; finish; exit 1; }
for i in $(seq 30); do mountpoint -q "$SEA" && break
  udisksctl mount -b /dev/disk/by-uuid/C2B23E6000C4A8C2 --no-user-interaction >/dev/null 2>&1; sleep 10; done
mountpoint -q "$SEA" || { log "Seagate not mounted (the source backbone lives there); chain2 stays held"; finish; exit 1; }
log "GPU: $(nvidia-smi -L | head -1)"
REF=$HOME/models/v41-work/golden/diag_doc1k
mkdir -p $REF; ln -sfn $HOME/models/v41-work/golden/src/doc1k $REF/doc1k
python3 - $HOME/models/v41-work/golden/src/meta.json $REF/meta.json <<'E'
import json, sys
m = json.load(open(sys.argv[1])); m["prompts"] = {"doc1k": m["prompts"]["doc1k"]}
json.dump(m, open(sys.argv[2], "w"))
E
log "gate start"
GOLDEN_ROW_COS=$D/row_cos.txt cuda/build/gate_v41_golden --ckpt $HOME/models/DeepSeek-V4.1-Flash-1017 --ref $REF \
  > $D/gate.log 2>&1
log "gate rc=$? ($(tail -1 $D/gate.log))"
sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1
if cmp -s $L/chain.STOP $L/chain.STOP.hold; then rm -f $L/chain.STOP; systemctl --user start dsv41-chain2; log "chain2 released"
else log "chain.STOP is not our hold ($(cat $L/chain.STOP 2>/dev/null)); chain2 left as is"; fi
rm -f $L/chain.STOP.hold
finish
touch $D/DONE
