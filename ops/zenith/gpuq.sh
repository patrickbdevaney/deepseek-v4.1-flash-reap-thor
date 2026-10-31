#!/usr/bin/env bash
# One GPU lane (H4): serialise GPU jobs and refuse to co-run beside another model.
#   ops/zenith/gpuq.sh <tag> <command...>        (run it setsid-detached; it blocks until the lane is free)
# Waits for: the lane lock (flock), no v41_decode / v41_server / llama-server / llama-cli process
# (pgrep -x only), and MemAvailable >= GPUQ_MIN_GIB (default 100 GiB: Thor is unified memory, so a
# vLLM or leafllm job holding the GPU shows up here even when it is a plain python process).
# Then drops the page cache, runs the command and appends a receipt to ops/logs/gpuq/receipts.tsv.
set -uo pipefail
cd ~/deepseek-v4.1-flash-reap-thor
tag=$1; shift
Q=ops/logs/gpuq; mkdir -p $Q
exec 9>$Q/lane.lock
flock 9
min=${GPUQ_MIN_GIB:-100}
busy() {
    for p in v41_decode v41_server llama-server llama-cli; do pgrep -x $p >/dev/null && { echo "proc $p"; return 0; }; done
    local a; a=$(awk '/^MemAvailable/{print int($2/1048576)}' /proc/meminfo)
    [ "$a" -lt "$min" ] && { echo "MemAvailable ${a} GiB < ${min}"; return 0; }
    return 1
}
n=0
while why=$(busy); do
    [ $((n % 30)) -eq 0 ] && echo "[gpuq] $tag waiting: $why"
    n=$((n + 1)); sleep 20
done
sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null
rev=$(git -C ~/deepseek-v4.1-flash-reap-thor rev-parse --short HEAD 2>/dev/null || echo -)
t0=$(date +%s); echo "[gpuq] $tag start $(date -Is) rev $rev"
"$@"; rc=$?
t1=$(date +%s)
printf '%s\t%s\t%s\t%d\t%d\t%s\n' "$(date -Is -d @$t0)" "$tag" "$rev" $rc $((t1 - t0)) "$*" >> $Q/receipts.tsv
echo "[gpuq] $tag rc=$rc $((t1 - t0)) s"
exit $rc
