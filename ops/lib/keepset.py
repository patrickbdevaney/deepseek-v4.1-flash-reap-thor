"""Keep-sets and their fingerprints. numpy-free on purpose: every stage imports this.

A keep-set is {"<layer>": [kept routed-expert ids, ascending]} for all 40 decoder layers. Its
fingerprint is the sha256 of the canonical JSON (sorted layer keys as ints, sorted ids, no
whitespace). Every artifact derived from a keep-set records the fingerprint, and every consumer
calls `require()` with the fingerprint it was built against. A mismatch is FATAL everywhere.

That is the GLM pass-3 lesson (PLAN.md, defect 1): a keep-set guard that raised correctly but sat
inside `try/except Exception` was downgraded to a WARN and the stage fell back to a gain derived
for a different mask. Nothing in this pipeline catches KeepsetMismatch.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


class KeepsetMismatch(SystemExit):
    """Subclasses SystemExit so a broad `except Exception` cannot swallow it."""


def canonical(mask: dict, *, n_layers: int | None = None, n_experts: int | None = None,
              n_keep: int | None = None) -> dict[str, list[int]]:
    out = {}
    for k, v in mask.items():
        ids = sorted(int(i) for i in v)
        if len(set(ids)) != len(ids):
            raise ValueError(f"layer {k}: duplicate expert ids")
        if n_experts is not None and ids and not (0 <= ids[0] and ids[-1] < n_experts):
            raise ValueError(f"layer {k}: id out of range [0, {n_experts})")
        if n_keep is not None and len(ids) != n_keep:
            raise ValueError(f"layer {k}: {len(ids)} kept, expected {n_keep}")
        out[str(int(k))] = ids
    if n_layers is not None and sorted(int(k) for k in out) != list(range(n_layers)):
        raise ValueError(f"keep-set must cover layers 0..{n_layers - 1}, has {sorted(int(k) for k in out)[:5]}...")
    return {k: out[k] for k in sorted(out, key=int)}


def fingerprint(mask: dict) -> str:
    c = canonical(mask)
    blob = json.dumps([[int(k), v] for k, v in c.items()], separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def short(fp: str) -> str:
    return fp[:16]


def save(path: Path, mask: dict, **meta) -> str:
    from .env import atomic_json
    c = canonical(mask)
    fp = fingerprint(c)
    atomic_json(Path(path), {"sha256": fp, "n_keep": sorted({len(v) for v in c.values()}), **meta, "keep": c})
    return fp


def load(path: Path, expect: str | None = None) -> tuple[dict[str, list[int]], str]:
    """-> (keep, fingerprint). Re-derives the fingerprint from the ids and refuses a file whose
    stored sha does not match its own content, or (with `expect`) a different keep-set."""
    d = json.load(open(path))
    keep = canonical(d["keep"])
    fp = fingerprint(keep)
    if d.get("sha256") != fp:
        raise KeepsetMismatch(f"{path}: stored sha {d.get('sha256')} != content sha {fp} (file edited by hand?)")
    if expect is not None:
        require(fp, expect, where=str(path))
    return keep, fp


def require(got: str, expect: str, *, where: str = "") -> None:
    if got != expect:
        raise KeepsetMismatch(f"KEEP-SET MISMATCH{' at ' + where if where else ''}: built against "
                              f"{short(expect)}, found {short(got)}. Fatal by design; re-run the stage on the "
                              f"current keep-set instead of reusing a stale artifact.")


def remap(keep_ids: list[int]) -> dict[int, int]:
    """old expert id -> new contiguous id (surgery renumbers kept experts 0..n-1 in ascending order)."""
    return {old: new for new, old in enumerate(sorted(keep_ids))}
