"""Park decoder-layer source shards on the Seagate while s03's 204 GB stream occupies the NVMe, and
restore them before anything replays the source again (2026-10-19).

Why: s03's in-place stream h.npy is T x 4 x 5120 bf16 = ~204 GB for the 5M-token corpus, its Engram
outputs another ~63 GB (s04/s06/s10 replay reads them, so they stay), and the NVMe had ~206 GB free
when the stream was due. A layer shard is dead to s03 once `layer/L<L>` is done, so each one is
copied to the Seagate, sha256-checked against the pinned tree, and only then removed from the NVMe.

Parked shards are NOT symlinked back: the Seagate is NTFS over FUSE and safetensors mmap faults run
at ~8 MB/s there, so a reader must fail loudly on a missing shard rather than crawl. `restore()` copies
them back with plain sequential reads, sha-checked, once the dead s03 stream has been dropped.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

from . import env

PARK = Path("/media/patrickd/Seagate Portable Drive/v41_1017_parked")
TREE = env.SRC.parent / "v41_1017_tree.json"
LEDGER = env.LOGS / "park_ledger.json"


def shard_of(L: int) -> str:
    return f"model-{L + 3:05d}-of-00048.safetensors"


def _want(name: str) -> str:
    return next(x["lfs"]["oid"] for x in json.load(open(TREE)) if x["path"] == name)


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(64 << 20):
            h.update(b)
    return h.hexdigest()


def _copy(src: Path, dst: Path) -> None:
    tmp = dst.with_name("." + dst.name + ".part")
    with open(src, "rb") as fi, open(tmp, "wb") as fo:
        shutil.copyfileobj(fi, fo, 64 << 20)
        fo.flush()
        os.fsync(fo.fileno())
    os.replace(tmp, dst)


def park(L: int) -> bool:
    """Move layer L's shard NVMe -> Seagate (verified). True if parked now or already parked."""
    name = shard_of(L)
    src, dst = env.SRC / name, PARK / name
    led = env.Ledger(LEDGER)
    if led.done(f"park/{name}"):
        return True
    if not src.is_file() or src.is_symlink():
        return False
    PARK.mkdir(parents=True, exist_ok=True)
    if not (dst.exists() and dst.stat().st_size == src.stat().st_size and _sha(dst) == _want(name)):
        _copy(src, dst)
        if _sha(dst) != _want(name):
            dst.unlink()
            raise IOError(f"sha mismatch on the Seagate copy of {name}")
    src.unlink()
    led.finish(f"park/{name}", layer=L)
    env.log(f"parked {name} on the Seagate (sha ok), NVMe copy removed", "park")
    return True


def parked() -> list[str]:
    led = env.Ledger(LEDGER)
    return [shard_of(L) for L in range(40) if led.done(f"park/{shard_of(L)}") and not (env.SRC / shard_of(L)).is_file()]


def restore(drop_dead_s03_stream: bool = True) -> int:
    """Copy every parked shard back to the NVMe (sha-checked). Drops s03's dead stream first when the
    s03 head unit is done and the disk is short. Returns the number restored."""
    todo = parked()
    if not todo:
        return 0
    need = sum((PARK / n).stat().st_size for n in todo) + (40 << 30)
    if shutil.disk_usage(env.SRC).free < need and drop_dead_s03_stream:
        if not env.Ledger(env.LOGS / "s03_ledger.json").done("head"):
            raise SystemExit("restore: disk short and the s03 stream is still live (s03 head not done)")
        for p in ("h.npy", "pre_mix.npy"):
            q = env.WORK / "s03" / p
            if q.exists():
                env.log(f"dropping dead s03 stream {p} to restore parked shards", "park")
                q.unlink()
    if shutil.disk_usage(env.SRC).free < need:
        raise SystemExit(f"restore: need {need / 2**30:.0f} GiB free on the NVMe")
    led = env.Ledger(LEDGER)
    for name in todo:
        dst = env.SRC / name
        _copy(PARK / name, dst)
        if _sha(dst) != _want(name):
            dst.unlink()
            raise IOError(f"sha mismatch restoring {name}")
        led.finish(f"restore/{name}")
        env.log(f"restored {name} to the NVMe (sha ok)", "park")
    return len(todo)
