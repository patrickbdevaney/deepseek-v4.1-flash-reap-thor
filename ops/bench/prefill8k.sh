#!/usr/bin/env bash
# 8k-token prefill timing (s08 ids8k), --n 16; arms as env strings
cd ~/deepseek-v4.1-flash-reap-thor; tag=$1; shift; i=0
for arm in "$@"; do sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null
  env $arm cuda/build/v41_decode --ckpt /home/patrickd/models/v41-work/thor/A192 --keepset /home/patrickd/models/v41-work/s04/keepset.json --engram /home/patrickd/models/v41-work/engram_rows --seqmax 16384 --ids-file /home/patrickd/models/v41-work/s08/ids8k.txt --n 16 > ops/logs/bench/${tag}_arm$i.log 2>&1
  echo "[p8k] arm $i ($arm): $(grep 'run 0' ops/logs/bench/${tag}_arm$i.log)"; i=$((i+1)); done
