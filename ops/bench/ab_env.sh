#!/usr/bin/env bash
# Paired env A/B on one binary: alternate arms (A B A B ...) so drift hits both equally.
#   ops/bench/ab_env.sh <tag> "<ENV A>" "<ENV B>" [rounds=3] [bin=cuda/build/v41_decode]
#   AB_EXTRA="--spec 3" adds decode arguments to both arms
set -uo pipefail
cd ~/deepseek-v4.1-flash-reap-thor
W=$HOME/models/v41-work; OUT=ops/logs/bench; tag=$1; EA=$2; EB=$3; R=${4:-3}; bin=${5:-cuda/build/v41_decode}
K="--ckpt $W/thor/A192 --keepset $W/s04/keepset.json --engram $W/engram_rows"
for r in $(seq 1 $R); do
  for arm in A B; do
    e=EA; [ $arm = B ] && e=EB
    sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null
    env ${!e} $bin $K --chat "Write a short paragraph about the history of the printing press." --n 256 --repeat 2 ${AB_EXTRA:-} \
        --json $OUT/${tag}_${arm}$r.json > $OUT/${tag}_${arm}$r.log 2>&1
    echo "[abenv] round $r $arm: $(grep 'tok/s mean' $OUT/${tag}_${arm}$r.log)"
  done
done
python3 - $OUT/${tag}_A1.json $OUT/${tag}_B1.json <<'PY'
import json,sys
a,b=(json.load(open(f))['ids'] for f in sys.argv[1:3])
print('[abenv] ids', 'IDENTICAL' if a==b else 'DIFFER at %d'%next(i for i,(x,y) in enumerate(zip(a,b)) if x!=y))
PY
