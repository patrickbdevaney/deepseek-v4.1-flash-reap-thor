#!/usr/bin/env bash
# A/B decode on the Thor build: one model load per arm, sequential, page cache dropped between.
#   ops/bench/ab_decode.sh <tag> "<ENV=val ...>" ["<ENV=val ...>" ...]
# Each arm: v41_decode --chat (printing press), --n 256 --repeat 3 --consistency 32. Decode == prefill bitwise since
# split-K stopped depending on M, so ANY mismatch (near-tie included) is a bug; 9 positions once missed one.
set -uo pipefail
cd ~/deepseek-v4.1-flash-reap-thor
export PATH=/usr/local/cuda/bin:$PATH
W=$HOME/models/v41-work; OUT=ops/logs/bench; mkdir -p $OUT
tag=$1; shift
(cd cuda && bash scripts/build_v41.sh v41_decode) || exit 1
K="--ckpt $W/thor/A192 --keepset $W/s04/keepset.json --engram $W/engram_rows"
i=0
for arm in "$@"; do
  sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null
  log=$OUT/${tag}_arm$i.log
  echo "[ab] arm $i: $arm  (MemAvailable $(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo) GiB, gpc $(cat /sys/class/devfreq/gpu-gpc-0/cur_freq))" | tee $log
  env $arm cuda/build/v41_decode $K --chat "Write a short paragraph about the history of the printing press." \
      --n 256 --repeat 3 --consistency 32 --json $OUT/${tag}_arm$i.json >> $log 2>&1
  echo "[ab] arm $i rc=$?" | tee -a $log
  grep -E "tok/s mean|consistency|identical|FAIL" $log | tail -3
  i=$((i+1))
done
