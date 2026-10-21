#!/bin/bash
# ExecCondition for dsv41-chain{,2}.service: exit 0 = start the chain, 1 = skip quietly (systemd treats 1-254 as
# "condition not met", not a failure). A STOP is never cleared here: ops/sd/watchdog.py owns that decision.
R=~/deepseek-v4.1-flash-reap-thor
cd $R || exit 1
L=ops/logs
[ -f $L/chain.STOP ] && { echo "chain.STOP present: $(cat $L/chain.STOP)"; exit 1; }
# a chain bash of THIS repo with exactly this argv (anchored: never a `bash -c "... bash ops/chain.sh ..."` wrapper,
# a grep, or an editor); a setsid-launched copy holds the flock anyway, this only skips the no-op start
running() { local p; for p in $(pgrep -f "^(/usr)?(/bin/)?bash ops/$1\\.sh\$"); do
  [ "$(readlink /proc/$p/cwd 2>/dev/null)" = "$(readlink -f $R)" ] && return 0; done; return 1; }
case "$1" in
  # done.s12 alone is not "finished": chain.sh marks it BEFORE logging the end line, and chain2 waits for that line
  chain)  [ -f $L/done.s12 ] && grep -q 'chain reached the end of implemented stages' $L/chain.log \
            && { echo "chain 1 finished"; exit 1; }
          running chain && { echo "chain 1 already running"; exit 1; } ;;
  chain2) [ -f $L/done.c2_s12_final ] && grep -q 'chain2 reached the end' $L/chain.log && { echo "chain2 finished"; exit 1; }
          running chain2 && { echo "chain2 already running"; exit 1; } ;;
  *) echo "usage: should_run.sh chain|chain2"; exit 1 ;;
esac
# The Seagate is required by BOTH chains: chain2 stages everything on it, and chain 1's remaining stages s05 (local
# master), s07 nvfp4 and s12 SystemExit without it (ops/s05_surgery.py master_dir). After a reboot nothing mounts it
# until a desktop session does (udisks, no fstab entry): try once. A skip here is not silent for long: the watchdog
# writes NEEDS_OWNER once the Seagate has been unmounted > 1 h while a chain is due.
SEA="/media/patrickd/Seagate Portable Drive"
if ! mountpoint -q "$SEA"; then
  udisksctl mount -b /dev/disk/by-uuid/C2B23E6000C4A8C2 --no-user-interaction >/dev/null 2>&1
  mountpoint -q "$SEA" || { echo "the Seagate is not mounted at $SEA; skipping"; exit 1; }
fi
exit 0
