#!/bin/bash
# Stage 0 fetch (user 2026-10-19): after the leafllm runner pauses and the model moves finish, drop swap and download the
# unpruned 1017 teacher (510.3 GB, pinned revision) to NVMe, then verify every file against the hub's sha256.
REV=1bae0e1088993b9b82de08f59de5be5a34e42170; D=~/models/DeepSeek-V4.1-Flash-1017
L=~/deepseek-v4.1-flash-reap-thor/ops/logs/s00_fetch.log; mkdir -p $(dirname $L)
log() { echo "[$(date -Is)] $*" >> $L; }
log "waiting: leafllm runner stop + model moves"
until grep -q 'runner stopped' ~/leafllm/logs/runner.log && ! pgrep -f pipeline/runner.sh >/dev/null \
      && ! docker ps --format '{{.Names}}' | grep -q leaf-teacher && grep -q '\] done' ~/models/move_to_external.log 2>/dev/null; do sleep 60; done
log "runner stopped, moves done: $(tail -1 ~/models/move_to_external.log)"
if swapon --show | grep -q /swapfile; then
  sudo -n swapoff /swapfile && sudo -n rm -f /swapfile && sudo -n sed -i.bak_swap '/\/swapfile/d' /etc/fstab && log "swap removed"
fi
log "free before download: $(df -h ~ | tail -1)"
for pass in 1 2 3; do
  HF_XET_HIGH_PERFORMANCE=1 ~/leafllm/.venv/bin/hf download --max-workers 16 deepseek-ai/DeepSeek-V4.1-Flash-1017 --revision $REV --local-dir $D \
    --exclude "model-00047-*" "model-00048-*" >> $L 2>&1 && \
  HF_XET_HIGH_PERFORMANCE=1 ~/leafllm/.venv/bin/hf download --max-workers 16 deepseek-ai/DeepSeek-V4.1-Flash-1017 --revision $REV --local-dir $D >> $L 2>&1 && break
  log "download pass $pass failed, retrying"; sleep 120
done
log "download finished: $(du -sh $D | cut -f1); free $(df -h ~ | tail -1)"
curl -s "https://huggingface.co/api/models/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/$REV?recursive=1" > $D/../v41_1017_tree.json
python3 - "$D" >> $L 2>&1 <<'P'
import json, hashlib, os, sys
D = sys.argv[1]; t = json.load(open(D + "/../v41_1017_tree.json")); bad = []
for x in t:
    if x.get("type") != "file" or "lfs" not in x: continue
    h = hashlib.sha256(); f = open(os.path.join(D, x["path"]), "rb")
    for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
    ok = h.hexdigest() == x["lfs"]["oid"]; print(x["path"], "ok" if ok else "SHA MISMATCH", flush=True); bad += [] if ok else [x["path"]]
json.dump({"revision": "1bae0e1088993b9b82de08f59de5be5a34e42170", "bad": bad}, open(D + "/../v41_1017_sha_verify.json", "w"))
print("SHA VERIFY", "PASS" if not bad else f"FAIL {bad}")
P
log "s00 done"
