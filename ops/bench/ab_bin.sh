#!/usr/bin/env bash
# Paired binary A/B: alternate two v41_decode builds (A B A B ...) so drift hits both arms equally.
#   ops/bench/ab_bin.sh <tag> <binA> <binB> [rounds=3] ["ENV=val ..."]
set -uo pipefail
cd ~/deepseek-v4.1-flash-reap-thor
W=$HOME/models/v41-work; OUT=ops/logs/bench; tag=$1; A=$2; B=$3; R=${4:-3}; envs=${5:-DSV4_NVFP4_OVERLAY=$W/thor/overlay_nvfp4}
K="--ckpt $W/thor/A192 --keepset $W/s04/keepset.json --engram $W/engram_rows"
for r in $(seq 1 $R); do
  for arm in A B; do
    bin=${!arm}
    sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null
    env $envs $bin $K --chat "Write a short paragraph about the history of the printing press." --n 256 --repeat 2 \
        > $OUT/${tag}_${arm}$r.log 2>&1
    echo "[abbin] round $r $arm: $(grep 'tok/s mean' $OUT/${tag}_${arm}$r.log)"
  done
done
