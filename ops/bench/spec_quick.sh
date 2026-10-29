#!/usr/bin/env bash
# Quick spec sweep (128 ctx, printing press, --spec-sweep 1..4) with the overlay; no rebuild.
#   ops/bench/spec_quick.sh <tag> ["<ENV=val ...>"]
set -uo pipefail
cd ~/deepseek-v4.1-flash-reap-thor
W=$HOME/models/v41-work; OUT=ops/logs/bench; mkdir -p $OUT
tag=$1; envs=${2:-}
sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null
env DSV4_NVFP4_OVERLAY=$W/thor/overlay_nvfp4 $envs cuda/build/v41_decode --ckpt $W/thor/A192 --keepset $W/s04/keepset.json \
    --engram $W/engram_rows --chat "Write a short paragraph about the history of the printing press." --n 256 --seqmax 4096 \
    --spec-sweep 1,2,3,4 --json $OUT/spec_$tag.json > $OUT/spec_$tag.log 2>&1
echo "[spec] rc=$?" >> $OUT/spec_$tag.log
grep -E "spec K=|no-spec|identity" $OUT/spec_$tag.log
