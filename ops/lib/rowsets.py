"""Which corpus rows each post-saliency stage forwards (numpy only).

s02 froze a held-out split (rows.jsonl "heldout": true, ~10% per bucket). It is halved here into
  screen   - every TUNING decision: s04 mask arbitration, the s10 heal gate
  confirm  - the RELEASE gates only (s10 final panel), never looked at while tuning
so the numbers on the model cards were not used to choose the model (PLAN.md: "Tune on a screening
split. Freeze the finalists, hash the confirmation split").

The halves are a function of (corpus spec hash, row index) only: rebuilding the same corpus gives
the same split, a different corpus gives a different one, and no RNG state is involved.

Subsets under a token budget are stratified by bucket in proportion to the corpus token targets,
with at least one row per bucket when the split has one, and long-context rows are admitted even
when one row exceeds its bucket's share (without them the position bins past 8k are empty).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def load_rows(corpus_dir: Path) -> tuple[list[dict], dict]:
    rows = [json.loads(l) for l in open(Path(corpus_dir) / "rows.jsonl")]
    man = json.load(open(Path(corpus_dir) / "manifest.json"))
    return rows, man


def split_of(spec_hash: str, row: int) -> str:
    h = hashlib.sha256(f"{spec_hash}:{row}".encode()).digest()[0]
    return "screen" if h < 128 else "confirm"


def split_rows(rows: list[dict], spec_hash: str, which: str) -> list[dict]:
    """which: screen | confirm | heldout (both halves) | calib (non-held-out)."""
    if which == "calib":
        return [r for r in rows if not r["heldout"]]
    held = [r for r in rows if r["heldout"]]
    if which == "heldout":
        return held
    return [r for r in held if split_of(spec_hash, r["row"]) == which]


def subset(rows: list[dict], budget_tokens: int, shares: dict[str, float], *,
           long_bucket: str = "longctx", min_per_bucket: int = 1) -> list[int]:
    """Deterministic stratified subset. Within a bucket rows are taken in visit order (s02 already
    interleaved and shuffled them), shortest-first only for the long bucket so the budget buys the
    most long rows. Returns sorted row ids."""
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(r["bucket"], []).append(r)
    tot_share = sum(shares.get(b, 0.0) for b in by) or 1.0
    picked: list[int] = []
    for b, rs in sorted(by.items()):
        if b == long_bucket:
            rs = sorted(rs, key=lambda r: (r["length"], r["row"]))
        cap = budget_tokens * shares.get(b, 0.0) / tot_share
        used, n = 0, 0
        for r in rs:
            if n >= min_per_bucket and used + r["length"] > cap:
                break
            picked.append(r["row"])
            used += r["length"]
            n += 1
    return sorted(picked)


def rows_hash(row_ids: list[int], spec_hash: str) -> str:
    return hashlib.sha256(json.dumps([spec_hash, sorted(row_ids)]).encode()).hexdigest()
