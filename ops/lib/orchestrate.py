"""Subprocess-per-unit orchestration shared by s04-s10 (the s03 `run` pattern, factored out).

Every GPU unit is its own process and is followed by a verified reclaim (drop_caches; a process
exit does not return the Tegra nvmap pool), and a unit that fails stops the stage with its exit
code: the stage is resumable, the chain writes chain.STOP, the check-in fixes and relaunches.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

from . import env


def run_unit(script: Path, argv: list[str], *, stage: str, gpu: bool, min_avail_gib: float) -> None:
    env.log(f"start {' '.join(argv)}", stage)
    t0 = time.time()
    rc = subprocess.run([sys.executable, str(script), *argv]).returncode
    if gpu:
        env.reclaim(stage, expect_gib=min_avail_gib)
    if rc != 0:
        raise SystemExit(f"{stage}: unit {' '.join(argv)} failed rc={rc}; fix and re-run (it resumes)")
    env.log(f"done {' '.join(argv)} in {time.time() - t0:.0f}s", stage)


def disk_free(path: Path) -> int:
    p = Path(path)
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(p).free


def batches(names: list[str], per_item: int, free: int, margin: int, cap: int | None = None) -> list[list[str]]:
    """Split arms into consecutive batches whose streams fit the free disk minus a margin."""
    k = max(1, (free - margin) // max(per_item, 1))
    if cap:
        k = min(k, cap)
    return [names[i:i + k] for i in range(0, len(names), k)]


SEAGATE = Path("/media/patrickd/Seagate Portable Drive")


def staging_dir(need_bytes: int, name: str, *, prefer: Path, margin: int = 40 << 30) -> Path:
    """NVMe (`prefer`) when it can hold `need_bytes` plus a margin, else the Seagate (decision 4:
    the Seagate is staging space for write-once artifacts, never for serving)."""
    if disk_free(prefer) - need_bytes > margin:
        d = Path(prefer) / name
    elif SEAGATE.exists() and disk_free(SEAGATE) - need_bytes > margin:
        d = SEAGATE / "v41_staging" / name
    else:
        raise SystemExit(f"no disk for {name}: need {need_bytes / 2**30:.1f} GiB + {margin / 2**30:.0f} GiB margin")
    d.mkdir(parents=True, exist_ok=True)
    return d
