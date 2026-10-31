#!/usr/bin/env bash
# Teacher-forced logits per arm (v41_decode --score): same prompt + forced continuation, one load per arm.
#   ops/bench/score_arms.sh <tag> <forced_ids.txt> "<ENV=val ...>" ...   -> $SCORE_DIR/<tag>_armN.f32
# Compare with ops/bench/score_cmp.py (KL / top-1 / NLL against arm 0). SCORE_EXTRA adds decode arguments to every arm.
set -uo pipefail
cd ~/deepseek-v4.1-flash-reap-thor
export PATH=/usr/local/cuda/bin:$PATH
W=$HOME/models/v41-work; OUT=ops/logs/bench; SD=${SCORE_DIR:-$OUT}
tag=$1; forced=$2; shift 2
(cd cuda && bash scripts/build_v41.sh v41_decode) || exit 1
i=0
for arm in "$@"; do
  sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null
  echo "[score] arm $i: $arm" | tee -a $OUT/${tag}.log
  env $arm cuda/build/v41_decode --ckpt $W/thor/A192 --keepset $W/s04/keepset.json --engram $W/engram_rows \
      --chat "Write a short paragraph about the history of the printing press." \
      --score $forced --score-out $SD/${tag}_arm$i.f32 ${SCORE_EXTRA:-} >> $OUT/${tag}.log 2>&1
  echo "[score] arm $i rc=$?" | tee -a $OUT/${tag}.log
  i=$((i+1))
done
