#!/usr/bin/env python3
"""Pick the next zenith lever from docs/zenith/GRAPH.tsv (docs/ZENITH_LOOP.md section 3).

    ops/zenith/next.py [--goal ar_ceiling|head_fp4|spec|draft_ft] [--all]

IN_PROGRESS first (crash resume); else TODO nodes whose deps are DONE, ranked by
expected_ms * p / cost_h. Information nodes (expected_ms 0) rank by 1 / cost_h * 0.5.
"""
import argparse, csv, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
rows = list(csv.DictReader(open(ROOT / "docs/zenith/GRAPH.tsv"), delimiter="\t"))
by = {r["id"]: r for r in rows}
ap = argparse.ArgumentParser(); ap.add_argument("--goal"); ap.add_argument("--all", action="store_true")
a = ap.parse_args()
sel = [r for r in rows if not a.goal or r["goal"] in (a.goal, "all")]
def ready(r):
    return all(by[d]["status"] == "DONE" for d in r["deps"].split(",") if d and d != "-")
def score(r):
    e, p, c = float(r["expected_ms"]), float(r["p"]), max(float(r["cost_h"]), 0.5)
    return (e * p if e > 0 else 0.5) / c
prog = [r for r in sel if r["status"] == "IN_PROGRESS"]
todo = sorted((r for r in sel if r["status"] == "TODO" and ready(r)), key=score, reverse=True)
blocked = [r for r in sel if r["status"] == "TODO" and not ready(r)]
for r in prog: print(f"IN_PROGRESS {r['id']:4s} [{r['goal']}] {r['notes']}")
for r in todo[: None if a.all else 5]: print(f"READY {score(r):6.2f} {r['id']:4s} [{r['goal']}] -{r['expected_ms']} ms  {r['notes']}")
if a.all:
    for r in blocked: print(f"BLOCKED on {r['deps']:8s} {r['id']:4s} [{r['goal']}] {r['notes']}")
# goal rollup: remaining expected ms per goal (open nodes)
tot = {}
for r in rows:
    if r["status"] != "DONE": tot[r["goal"]] = tot.get(r["goal"], 0) + float(r["expected_ms"]) * float(r["p"])
print("open expected (ms x p):", ", ".join(f"{g} {v:.1f}" for g, v in sorted(tot.items())))
