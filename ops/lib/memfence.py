# Vendored unchanged from ~/glm-5.3-reap/scripts/memfence.py (2026-10-19).
"""Per-step memory admission control, inside the process, before each allocation.

WHY THIS AND NOT THE THINGS WE ALREADY HAVE. Four box resets on 2026-09-26/27 established
that every external control fails on this hardware:

  * `drop_caches` cannot reclaim a live allocation, and on Tegra NvMap cannot reclaim
    page-cached pages the way MemAvailable implies -- a small allocation can fail with GiB
    "available" (vllm-project/vllm#35920, jarida-io/goose-in-a-pond#370).
  * SIGKILL does not land on a process blocked in the GPU driver. memguard killed the correct
    PID twice, 39 s apart, and MemAvailable fell 125MB -> 97MB -> 49MB regardless.
  * cgroup `MemoryMax` does not bind Tegra unified allocations; they are not charged to the
    cgroup that requested them.

If nothing outside the process can stop it, the process has to stop itself. `require()` is
called BEFORE each large allocation with that allocation's predicted size. It reads live
MemAvailable and raises rather than allocating. An exception unwinds Python and frees tensors;
a wedged box does not.

This is deliberately a RUNTIME check, not the static model in mem_lookahead.py. The static model
bounds the configuration before launch; this bounds each step during it. MEASURED: the static
model was right about S=16384-eager and still missed the S=16384-sparse failure, because peak
depends on driver-pool state that no static model can see.
"""
from __future__ import annotations

import os
import sys

GIB = 2 ** 30
# Abort with this much still free. It is large on purpose: the failure mode is not "allocation
# returns NULL" but "the box stops scheduling", and NvMap can fail well above zero.
RESERVE_GIB = float(os.environ.get("MEMFENCE_RESERVE_GIB", "18"))
_log = []


def available_gib() -> float:
    with open("/proc/meminfo") as fh:
        for line in fh:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1048576
    raise RuntimeError("MemAvailable missing from /proc/meminfo")


class MemFence(Exception):
    """Raised instead of making an allocation that would take the box below the reserve."""


def require(gib: float, label: str, *, reserve: float | None = None) -> None:
    """Refuse to proceed unless `gib` can be allocated and still leave the reserve free."""
    res = RESERVE_GIB if reserve is None else reserve
    avail = available_gib()
    if avail - gib < res:
        raise MemFence(
            f"{label}: needs {gib:.1f} GiB, only {avail:.1f} GiB available, "
            f"reserve is {res:.1f} GiB -> would leave {avail - gib:.1f} GiB. Refusing.")


def checkpoint(label: str, *, quiet: bool = False) -> float:
    """Record MemAvailable at a named point, for the allocation profile."""
    a = available_gib()
    _log.append((label, a))
    if not quiet:
        print(f"[memfence] {label:34s} avail {a:7.2f} GiB", flush=True)
    return a


def profile() -> list[tuple[str, float]]:
    return list(_log)


def report() -> None:
    if not _log:
        return
    print("\n[memfence] allocation profile (delta = memory consumed by that step):", flush=True)
    prev = None
    for lab, a in _log:
        d = "" if prev is None else f"  delta {a - prev:+7.2f}"
        print(f"    {lab:34s} {a:7.2f} GiB{d}", flush=True)
        prev = a
    lo = min(a for _, a in _log)
    print(f"    {'MINIMUM':34s} {lo:7.2f} GiB", flush=True)
