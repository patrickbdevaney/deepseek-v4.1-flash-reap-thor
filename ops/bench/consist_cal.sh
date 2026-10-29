#!/usr/bin/env bash
# Consistency calibration: --consistency 32 over 2 prompts per arm; prints every mismatch with its gap.
#   ops/bench/consist_cal.sh <tag> "<ENV ...>" ["<ENV ...>" ...]
set -uo pipefail
cd ~/deepseek-v4.1-flash-reap-thor
W=$HOME/models/v41-work; OUT=ops/logs/bench; tag=$1; shift
K="--ckpt $W/thor/A192 --keepset $W/s04/keepset.json --engram $W/engram_rows"
P1="Write a short paragraph about the history of the printing press."
P2="Explain step by step how to compute the derivative of x^3 * sin(x), then give the final answer."
i=0
for arm in "$@"; do
  for j in ${CAL_PROMPTS:-1 2}; do
    p=P$j
    sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null
    env $arm ${BIN:-cuda/build/v41_decode} $K --chat "${!p}" --n 256 --consistency 32 > $OUT/${tag}_arm${i}_p$j.log 2>&1
    echo "[cal] arm $i ($arm) prompt $j rc=$?"; grep -E "consistency" $OUT/${tag}_arm${i}_p$j.log
  done
  i=$((i+1))
done
