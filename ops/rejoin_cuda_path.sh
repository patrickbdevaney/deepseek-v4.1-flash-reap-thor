#!/bin/bash
# One-shot (2026-10-24): golden_src passed under the owner's drift amendment, but the running chain2 computed
# CUDA_OK=0 at startup. Wait for the current step (s09 quant L2) to finish, hold chain2, restart it so it re-evaluates
# CUDA_OK and runs golden A192 -> serving gates -> Thor publish -> overlay, then the GGUF ladder resumes (idempotent).
set -u; cd ~/deepseek-v4.1-flash-reap-thor; L=ops/logs
log() { echo "[$(date -Iseconds)] rejoin: $*" | tee -a $L/chain.log; }
until [ -f $L/done.s09_quant_L2 ]; do
  systemctl --user is-active -q dsv41-chain2 || { log "chain2 stopped before quant L2 finished; not touching it"; exit 1; }
  sleep 30; done
HOLD="manual hold: restarting chain2 onto the CUDA path (ops/rejoin_cuda_path.sh)"
[ -f $L/chain.STOP ] && { log "chain.STOP already present ($(cat $L/chain.STOP)); not restarting"; exit 1; }
echo "$HOLD" > $L/chain.STOP
systemctl --user stop dsv41-chain2; sleep 10
pkill -f llama-quantize; sleep 5
sync; sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches'
[ "$(cat $L/chain.STOP)" = "$HOLD" ] && rm -f $L/chain.STOP
systemctl --user start dsv41-chain2; log "chain2 restarted after quant L2 (CUDA path re-enabled)"
