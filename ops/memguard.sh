#!/usr/bin/env bash
# memguard.sh — keep the streaming pass alive on a box where the kernel cannot help.
#
# WHY THE KERNEL NEVER HELPS HERE. Established on this machine by the DSpark project
# (deepseek-v4-flash-0731-cuda/scripts/memguard.sh, measured 2026-08-20): a touched 2048 MiB
# cudaMalloc charges **42 MiB** to the calling cgroup. Tegra unified memory comes from the
# driver's allocator, not the page allocator, so it is invisible to every kernel accounting
# path. The OOM killer scores by RSS and therefore never selects the real consumer -- which is
# why s03 died six times with NO oom-kill line in dmesg. memory.max cannot bound it and this
# kernel has no PSI, so systemd-oomd cannot run at all.
#
# WHAT IS DIFFERENT ABOUT THIS WORKLOAD, AND WHY TIER 1 EXISTS. s03 mmaps 306 GB of source
# shards. That page cache IS reclaimable -- dropping it returned the box from "107 GiB used,
# 350 MB available" to "3 GiB used, 119 GiB available" instantly -- but Tegra under-reports it
# in both Cached and MemAvailable, and the kernel evidently will not reclaim it fast enough to
# satisfy a driver allocation. So the first response is to drop page cache, which costs nothing
# but re-reads and harms no process. Killing is the last resort, not the first.
#
# NARROW LICENSE. Tier 2 may only ever kill this project's own stage children (run_stage.py).
# It never picks "the biggest RSS", because RSS is precisely the number already proven not to
# reflect who holds the memory.
set -u
ROOT="${ROOT:-/home/patrickd/deepseek-v4.1-flash-reap-thor}"
cd "$ROOT" || exit 1
CACHE_MB="${CACHE_MB:-16000}"     # tier 1: drop page cache below this
# MEASURED 2026-08-27: this workload's healthy plateau during the layer sweep is 2-3 GiB
# available, because ~105 GiB is mmap page cache the kernel holds but will release under a
# host allocation. A 4000 MB floor therefore killed perfectly healthy runs. Same reasoning as
# the DSpark memguard: the floor cannot be set above the plateau, so the LEVEL test is set
# below it and the SLOPE test catches genuine runaways.
# MEASURED AGAIN 2026-08-27: a large model load (s07 places 157 GiB across RAM and disk)
# legitimately drives MemAvailable to a few hundred MB, and the box does not die there - the
# earlier real collapse was observed at 317 MB with a 1730 MB/s slope. A 900 MB floor killed
# a healthy s07 at 866 MB. The LEVEL test keeps moving below the observed plateau; the SLOPE
# test is what actually distinguishes a load from a runaway.
# SINGLETON. Stacked instances happened three times here: each restart left the previous one
# running, so several guards polled independently and could each fire. flock releases on exit,
# including on kill, so a dead holder never wedges the next start.
LOCK=/tmp/dsv41-memguard.lock
exec 8>"$LOCK"
if ! flock -n 8; then
    echo "[memguard $(date -Is)] another instance holds the lock - exiting" >> logs/memguard.log
    exit 0
fi

FLOOR_MB="${FLOOR_MB:-1000}"       # tier 2: kill our stage below this
# A NORMAL MoE layer build legitimately drops ~28 GiB in ~8 s (~3500 MB/s), so a 900 MB/s
# slope threshold fires on healthy work - it killed a run at 11 GB available doing exactly
# that. The slope test only earns its place if it is well above normal operation, and the
# level floor remains the real backstop.
RATE_MB_S="${RATE_MB_S:-6000}"    # tier 2 slope trigger
DANGER_MB="${DANGER_MB:-4000}"    # slope only counts below this
BREACHES="${BREACHES:-3}"
SLOPE_N="${SLOPE_N:-2}"
POLL_S="${POLL_S:-0.5}"
DROP_COOLDOWN_S="${DROP_COOLDOWN_S:-10}"
LOG="${LOG:-$ROOT/logs/memguard.log}"
mkdir -p "$(dirname "$LOG")"
say(){ echo "[memguard $(date -Is)] $*" >> "$LOG"; }

# Spawn-free wait: sleep is not a builtin, and one fork per poll is noise on the thing we watch.
# fd 7, NOT 8: fd 8 is the singleton lock above, and re-using it closed the lock (audit O4, 2026-10-21: the running
# guard's fd 8 pointed at the deleted FIFO and nothing held /tmp/dsv41-memguard.lock).
_FIFO=$(mktemp -u); mkfifo "$_FIFO" 2>/dev/null && exec 7<>"$_FIFO" && rm -f "$_FIFO"
napp(){ if [ -e /proc/self/fd/7 ]; then read -t "$POLL_S" -u 7 _x 2>/dev/null; else sleep "$POLL_S"; fi; return 0; }

# ---- tier-2 victim selection (rewritten 2026-09-26, licence widened 2026-10-21) -----------------------------------
# MEASURED 2026-09-26: at 21:20:51 this guard logged "killing stage s03_saliency pid 2197" while the actual runaway was
# exp_seqlen_saliency.py, which was not on the licence at all. The pipeline stage had been running stably for 47
# minutes; the experiment had started minutes earlier. The guard killed the healthy long-running job, left the
# runaway alive, and the box wedged anyway -- twice. Two rules follow:
#   1. The licence must cover every heavy job this project can start, not just the ones that existed when it was
#      written. A job outside the licence is not merely unkillable, it is a job whose presence makes the guard kill
#      something else instead. AUDIT O1 (2026-10-21): the CUDA binaries (v41_decode/v41_server load the whole
#      ~107 GiB build), the gates, the GGUF kld and test-backend-ops, and the non-ops/ scripts were all missing, so a
#      runaway v41_decode under s11 would have cost the s11_spec.py parent (which frees nothing) instead.
#   2. Among licensed candidates, prefer the one that STARTED MOST RECENTLY. The newcomer is what changed a working
#      situation into a failing one. RSS is deliberately NOT used to rank -- it is the number proven not to track
#      Tegra unified allocations, which is why the kernel OOM killer picks wrong on this box.
# Match on the script/binary BASENAME, never on " scripts/<name>" (an absolute-path launch was once missed). Because a
# pattern also matches any shell, pager or editor whose command line merely MENTIONS a licensed name (a Claude Code
# `bash -c "... ops/s10_eval.py ..."` wrapper is always the newest), candidates whose comm is a shell/pager/editor/
# agent are never selected (audit O1). V4.1: only this project's own stage processes are kill candidates.
LICENCE=("ops/s[0-9][0-9][a-z]?_[a-z_]*\.py"
         "v41_decode|v41_server|gate_v41_|gemv_format_bench|v41-teacher-kld|test-backend-ops"
         "ref_dump\.py|v41_unit_ref\.py|convert_v41_reap\.py"
         "llama-(quantize|imatrix|cli|server|perplexity|completion)"
         "hf upload|huggingface_hub")
NEVER_COMM=" bash sh dash zsh fish claude node less more vi vim nvim view nano emacs tail head cat grep egrep pgrep watch "
licensed_cands(){ local pat q c cands=""
  for pat in "${LICENCE[@]}"; do
    for q in $(pgrep -f "$pat" 2>/dev/null); do
      case " $cands " in *" $q "*) continue ;; esac
      c=""; { read -r c < "/proc/$q/comm"; } 2>/dev/null || continue
      case "$NEVER_COMM" in *" $c "*) continue ;; esac
      cands="$cands $q"
    done
  done
  echo "$cands"; }
# NEVER select ourselves (CLAUDE.md §7: a pattern that matches your own command line is a self-kill waiting to happen).
pick_victim(){ local q st pid="" newest=0
  for q in $(licensed_cands); do
    [ "$q" = "$$" ] && continue
    st=$(awk '{print $22}' "/proc/$q/stat" 2>/dev/null)
    [ -z "$st" ] && continue
    if [ "$st" -ge "$newest" ]; then newest=$st; pid=$q; fi
  done
  echo "$pid"; }

say "started: cache_drop<${CACHE_MB}MB kill<${FLOOR_MB}MB slope>${RATE_MB_S}MB/s below ${DANGER_MB}MB poll=${POLL_S}s"

bad=0; srate=0; prev=""; last_drop=0; warned=0
while true; do
  avail=""
  while read -r _k _v _; do
    if [ "$_k" = "MemAvailable:" ]; then avail=$((_v/1024)); break; fi
  done < /proc/meminfo 2>/dev/null
  [ -z "$avail" ] && { napp; continue; }

  now=$(date +%s)

  # ---- tier 1: reclaim page cache. Free, reversible, and the actual cause here. ----------
  if [ "$avail" -lt "$CACHE_MB" ] && [ $((now - last_drop)) -ge "$DROP_COOLDOWN_S" ]; then
    sync
    sudo -n sh -c 'echo 1 > /proc/sys/vm/drop_caches' 2>/dev/null
    last_drop=$now
    after=""
    while read -r _k _v _; do
      if [ "$_k" = "MemAvailable:" ]; then after=$((_v/1024)); break; fi
    done < /proc/meminfo 2>/dev/null
    # MEASURED 2026-09-23: `echo 1` reclaimed NOTHING on this box for 9 hours straight
    # (3050MB -> 3051MB, every 10 s, 8640 times a day) while 115 GiB sat unreferenced -- not in
    # any RSS (all processes summed to 2.06 GiB), not in Cached (579 MB), not in Slab (585 MB).
    # A single `echo 3` returned the box to 119 GiB available instantly. The pool is the nvmap
    # driver allocator, which /proc/meminfo attributes to no one, and only the full shrinker
    # pass releases it. Tier 1 therefore ESCALATES rather than assuming page cache, and a drop
    # that reclaims nothing is logged as a FAILURE -- the previous version logged the no-op in
    # exactly the same words as a success, which is why a dead guard looked healthy for months.
    esc=""
    if [ -n "$after" ] && [ $((after - avail)) -lt 500 ] && [ "$after" -lt "$CACHE_MB" ]; then
      sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null
      esc=" escalated=3"
      while read -r _k _v _; do
        if [ "$_k" = "MemAvailable:" ]; then after=$((_v/1024)); break; fi
      done < /proc/meminfo 2>/dev/null
    fi
    if [ -n "$after" ] && [ $((after - avail)) -lt 200 ]; then
      say "tier1 NO-OP: ${avail}MB -> ${after}MB${esc} -- nothing reclaimable; if avail stays low the memory is held by a live allocation, not cache"
    else
      say "tier1 drop_caches at ${avail}MB -> ${after:-?}MB${esc}"
    fi
    prev=""; bad=0; srate=0
    napp; continue
  fi

  # ---- tier 2: our stage is genuinely running away. Kill it so it can be retried. --------
  trip=""
  if [ -n "$prev" ]; then
    rate=$(( (prev - avail) * 2 ))    # POLL_S=0.5 -> MB/s
    if [ "$avail" -lt "$DANGER_MB" ] && [ "$rate" -gt "$RATE_MB_S" ]; then srate=$((srate+1)); else srate=0; fi
    [ "$srate" -ge "$SLOPE_N" ] && trip="MemAvailable ${avail}MB falling ${rate}MB/s"
  fi
  prev=$avail
  if [ "$avail" -lt "$FLOOR_MB" ]; then bad=$((bad+1)); else bad=0; fi
  [ "$bad" -ge "$BREACHES" ] && trip="MemAvailable ${avail}MB < floor ${FLOOR_MB}MB x${bad}"

  if [ -n "$trip" ]; then
    # VICTIM SELECTION: the newest licensed, non-shell process (pick_victim above; never RSS).
    pid=$(pick_victim)
    if [ -n "$pid" ]; then
      stg=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | cut -c1-160)
      say "!!! $trip — killing stage ${stg:-?} pid $pid (orchestrator will retry)"
      kill -9 "$pid" 2>/dev/null
      sync; sudo -n sh -c 'echo 1 > /proc/sys/vm/drop_caches' 2>/dev/null
      say "killed; MemAvailable now $(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)MB"
      bad=0; srate=0; prev=""; last_drop=$(date +%s)
    elif [ "$warned" = 0 ]; then
      say "WARN $trip — no licensed heavy job running; doing nothing (narrow license)"
      warned=1
    fi
  else
    warned=0
  fi
  napp
done
