#!/usr/bin/env bash
# nsys trace of the v41 decode step: one model load, short chat, CUDA kernels only.
#   ops/prof/nsys_decode.sh <tag> ["ENV=val ..."] ["extra v41_decode args, e.g. --spec 2"]
# Writes ops/logs/prof/<tag>.nsys-rep and <tag>_kern.csv (cuda_gpu_kern_sum).
set -uo pipefail
cd ~/deepseek-v4.1-flash-reap-thor
export PATH=/usr/local/cuda/bin:$PATH
export TMPDIR=$HOME/.cache/nsys-tmp; mkdir -p $TMPDIR
W=$HOME/models/v41-work; OUT=ops/logs/prof; mkdir -p $OUT
tag=$1; envs=${2:-}; extra=${3:-}
sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null
env $envs nsys profile -t cuda --cuda-graph-trace=node -o $OUT/$tag -f true \
  cuda/build/v41_decode --ckpt $W/thor/A192 --keepset $W/s04/keepset.json --engram $W/engram_rows \
  --chat "Write a short paragraph about the history of the printing press." --n 128 --repeat 1 $extra \
  > $OUT/$tag.log 2>&1
echo "rc=$?" >> $OUT/$tag.log
nsys stats -r cuda_gpu_kern_sum -f csv -o $OUT/${tag} $OUT/$tag.nsys-rep >> $OUT/$tag.log 2>&1
ls $OUT/${tag}*
