#!/bin/bash
# ExecCondition for dsv41-chain{,2}.service: exit 0 = start the chain, 1 = skip quietly (systemd treats 1-254 as
# "condition not met", not a failure). A STOP is never cleared here: ops/sd/watchdog.py owns that decision.
cd ~/deepseek-v4.1-flash-reap-thor || exit 1
L=ops/logs
[ -f $L/chain.STOP ] && { echo "chain.STOP present: $(cat $L/chain.STOP)"; exit 1; }
case "$1" in
  chain)  [ -f $L/done.s12 ] && { echo "chain 1 finished"; exit 1; } ;;
  chain2) [ -f $L/done.c2_s12_final ] && grep -q 'chain2 reached the end' $L/chain.log && { echo "chain2 finished"; exit 1; } ;;
  *) echo "usage: should_run.sh chain|chain2"; exit 1 ;;
esac
# After a reboot nothing mounts the Seagate until a desktop session does (udisks, no fstab entry): try once.
SEA="/media/patrickd/Seagate Portable Drive"
if ! mountpoint -q "$SEA"; then
  udisksctl mount -b /dev/disk/by-uuid/C2B23E6000C4A8C2 --no-user-interaction >/dev/null 2>&1
  mountpoint -q "$SEA" || { echo "the Seagate is not mounted at $SEA; skipping"; exit 1; }
fi
exit 0
