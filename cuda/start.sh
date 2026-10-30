#!/bin/bash
# Launch v41_server (OpenAI-compatible) for DeepSeek-V4.1-Flash-1017-REAP192 on one Jetson AGX Thor.
#
#   ./start.sh                    32K context, port 8000, NVFP4 dense overlay on, DSpark speculation K=3
#   SPEC=0 ./start.sh             plain decoding (frees ~6 GiB)
#   SEQMAX=131072 ./start.sh
#
# Settings (environment):
#   CKPT        the thor-v1 build (hf download patrickbdevaney/DeepSeek-V4.1-Flash-1017-REAP192-Thor)
#                                                                      default ~/models/DeepSeek-V4.1-Flash-1017-REAP192-Thor
#   ENGRAM_DIR  engram_L1.v41rows / engram_L14.v41rows on NVMe         default $CKPT/engram_rows
#   SEQMAX      context ceiling (sizes the KV caches)                  default 32768
#   OVERLAY     1 = serve the dense weights from $CKPT/overlay_nvfp4   default 1
#   SPEC        DSpark drafts per verify with the checkpoint's own drafter (0 = off)   default 3
#   HOST PORT   listen address                                         default 127.0.0.1 8000
#   EXTRA       further v41_server arguments, appended verbatim
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
CKPT=${CKPT:-$HOME/models/DeepSeek-V4.1-Flash-1017-REAP192-Thor}
ENGRAM_DIR=${ENGRAM_DIR:-$CKPT/engram_rows}
SEQMAX=${SEQMAX:-32768}
OVERLAY=${OVERLAY:-1}
SPEC=${SPEC:-3}         # held-out: 1.345x mean over AR, never below 1.0x; greedy ids identical
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8000}

die() { echo "start.sh: $*" >&2; exit 1; }

[ -x "$here/build/v41_server" ] || die "no build/v41_server; run scripts/build_v41.sh v41_server first"
for f in thor.json globals.safetensors layer-39.safetensors tokenizer.json reap_keepset.json; do
    [ -f "$CKPT/$f" ] || die "missing $CKPT/$f"
done
for l in 1 14; do
    [ -f "$ENGRAM_DIR/engram_L$l.v41rows" ] || die "missing $ENGRAM_DIR/engram_L$l.v41rows (see README, 'Engram row files')"
done
case $(realpath "$ENGRAM_DIR") in /media/*|/mnt/usb*) die "ENGRAM_DIR is on removable media; copy the row files to NVMe" ;; esac

# Thor has unified memory: a second model-loading process beside this one can hard-reboot the box.
for p in v41_server v41_decode llama-server llama-cli llama-mtmd-cli vllm; do
    pgrep -x "$p" >/dev/null && die "$p is already running; stop it first (unified memory)"
done
avail=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
if [ "$avail" -lt 112 ]; then
    echo "start.sh: only $avail GiB available; dropping page cache" >&2
    sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1 || true
    avail=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
    [ "$avail" -ge 108 ] || die "only $avail GiB available, need ~108"
fi

if [ "$OVERLAY" = 1 ]; then
    [ -d "$CKPT/overlay_nvfp4" ] || die "OVERLAY=1 but no $CKPT/overlay_nvfp4"
    export DSV4_NVFP4_OVERLAY=$CKPT/overlay_nvfp4
fi

args=(--ckpt "$CKPT" --keepset "$CKPT/reap_keepset.json" --engram "$ENGRAM_DIR" --seqmax "$SEQMAX"
      --host "$HOST" --port "$PORT" --model DeepSeek-V4.1-Flash-1017-REAP192)
[ "$SPEC" -gt 0 ] && args+=(--spec "$SPEC")
echo "start.sh: v41_server ${args[*]} ${EXTRA:-}" >&2
# shellcheck disable=SC2086
exec "$here/build/v41_server" "${args[@]}" ${EXTRA:-}
