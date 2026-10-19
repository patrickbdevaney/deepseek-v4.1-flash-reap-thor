#!/usr/bin/env python3
"""Long-context mask guard: the position-binned routing-drift test of
notes/long_context_routing_drift.md (KQ4), run on the teacher routing s03 banked.

Two uses, both CPU-only and safe beside a GPU job:

  protect   before selection: per layer, experts that are sink-coupled, Super-Expert candidates
            routed in the first 8 positions, or whose gate-mass share rises with log-position.
            s04 passes them to HOPE's `protect` so no candidate mask can prune them.
                python3 ops/drift_guard.py protect --out $V41_WORK/s04/position_protect.json

  check     after selection: the full report for one candidate mask (slot / token / gate-mass
            rates per log-position bin with row-cluster bootstrap CIs, paired within-row drift
            vs [256,1k), across-row control, per-expert share-vs-position) and the KQ4 verdict.
            Exit 2 on BLOCK, 3 on POSITION_SENSITIVE.
                python3 ops/drift_guard.py check --keep $V41_WORK/s04/mask_hope.json

The mask file is {"<layer>": [kept expert ids, ...], ...} (s04 writes this shape).
Long-context rows (32k-128k) are what make the bins past 8k exist at all; without them the test
can only speak to [0, 8k), which the report states.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ops.lib import drift, env  # noqa: E402

D = env.WORK / "s03"
CFG = json.load(open(env.INFERENCE / "config.json"))
E, N_LAYERS = CFG["n_routed_experts"], CFG["n_layers"]


def load_rows(which: str):
    rows = [json.loads(l) for l in open(env.WORK / "corpus" / "rows.jsonl")]
    keep = [r for r in rows if which == "all" or (which == "heldout") == r["heldout"]]
    return keep


def layer_rows(L: int, rows):
    idx = np.load(D / "routing" / f"L{L}_idx.npy", mmap_mode="r")
    w = np.load(D / "routing" / f"L{L}_w.npy", mmap_mode="r")
    return [(np.asarray(idx[r["offset"]:r["offset"] + r["length"]], dtype=np.int64),
             np.asarray(w[r["offset"]:r["offset"] + r["length"]], dtype=np.float64)) for r in rows]


def super_candidates() -> np.ndarray:
    fmax = np.stack([np.load(D / "bank" / f"L{L}.npz")["fmax"] for L in range(N_LAYERS)])
    return drift.super_expert_candidates(fmax)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["protect", "check"])
    ap.add_argument("--keep", type=Path, help="mask json for `check`")
    ap.add_argument("--rows", choices=["all", "calib", "heldout"], default="all")
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    rows = load_rows(a.rows)
    maxlen = max(r["length"] for r in rows)
    sc = super_candidates()
    reports = {}
    for L in range(N_LAYERS):
        if a.cmd == "protect":
            pruned = np.ones(E, bool)          # evaluate every expert as if it were pruned
        else:
            keep = json.load(open(a.keep))[str(L)]
            pruned = np.ones(E, bool); pruned[np.asarray(keep, int)] = False
        reports[L] = drift.layer_report(layer_rows(L, rows), pruned, super_cand=sc[L], n_draws=a.draws)
        env.log(f"L{L}: sink {reports[L]['block_sink'][:8]} rising {reports[L]['block_rising'][:8]} "
                f"mass slope {reports[L]['mass_slope']['est']:+.2e}", "drift")
    meta = {"rows": len(rows), "max_row_len": maxlen,
            "note": None if maxlen > 8192 else "no rows beyond 8k: bins past [4k,8k) are empty, long-context drift untested"}
    if a.cmd == "protect":
        prot = {str(L): sorted(set(r["block_sink"]) | set(r["block_rising"])) for L, r in reports.items()}
        out = a.out or env.WORK / "s04" / "position_protect.json"
        env.atomic_json(out, {"protect": prot, "meta": meta})
        env.log(f"protect lists -> {out}: {sum(len(v) for v in prot.values())} experts over {N_LAYERS} layers", "drift")
        return
    verdict = drift.mask_verdict(reports)
    out = a.out or a.keep.with_suffix(".drift.json")
    env.atomic_json(out, {"verdict": verdict, "meta": meta, "layers": {str(k): v for k, v in reports.items()}})
    env.log(f"{a.keep.name}: {verdict['verdict']} -> {out}", "drift")
    raise SystemExit({"BLOCK": 2, "POSITION_SENSITIVE": 3}.get(verdict["verdict"], 0))


if __name__ == "__main__":
    main()
