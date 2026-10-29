#!/usr/bin/env bash
# nsys trace of an 8k-token prefill (s08 ids8k), CUDA kernels only.
#   ops/prof/nsys_prefill.sh <tag> ["ENV=val ..."] ["extra v41_decode args, e.g. --chunk 256"]
set -uo pipefail
cd ~/deepseek-v4.1-flash-reap-thor
export PATH=/usr/local/cuda/bin:$PATH
export TMPDIR=$HOME/.cache/nsys-tmp; mkdir -p $TMPDIR
W=$HOME/models/v41-work; OUT=ops/logs/prof; mkdir -p $OUT
tag=$1; envs=${2:-}; extra=${3:-}
sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null
env $envs nsys profile -t cuda -o $OUT/$tag -f true \
  cuda/build/v41_decode --ckpt $W/thor/A192 --keepset $W/s04/keepset.json --engram $W/engram_rows \
  --seqmax 16384 --ids-file $W/s08/ids8k.txt --n 2 $extra > $OUT/$tag.log 2>&1
echo "rc=$?" >> $OUT/$tag.log
nsys export -t sqlite -o $OUT/$tag.sqlite -f true $OUT/$tag.nsys-rep >> $OUT/$tag.log 2>&1
