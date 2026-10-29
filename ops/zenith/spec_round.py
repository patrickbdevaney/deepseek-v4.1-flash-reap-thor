#!/usr/bin/env python3
"""Spec-round ledger: an nsys trace of v41_decode --spec K -> per-round draft vs verify time by kernel.

    ops/zenith/spec_round.py ops/logs/prof/spec2b.sqlite [--top 14] [--skip 5]

A round starts at k_draft_inputs; the draft ends 4 kernels after the last k_mk_gather (Markov
gather, linear, add, argmax). Everything after that until the next round is the verify (+ rollback).
"""
import argparse
import collections
import sqlite3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite")
    ap.add_argument("--top", type=int, default=14)
    ap.add_argument("--skip", type=int, default=5, help="warm-up rounds to drop (graph capture)")
    a = ap.parse_args()
    db = sqlite3.connect(a.sqlite)
    rows = list(db.execute("select s.value, k.start, k.end from CUPTI_ACTIVITY_KIND_KERNEL k "
                           "join StringIds s on k.demangledName = s.id order by k.start"))
    starts = [i for i, r in enumerate(rows) if "k_draft_inputs" in r[0]]
    short = lambda n: n.split("(")[0].replace("void ", "")[:100]
    dr, vr = collections.defaultdict(float), collections.defaultdict(float)
    dw = vw = 0.0
    n = 0
    for s0, s1 in zip(starts[a.skip:-1], starts[a.skip + 1:]):
        seg = rows[s0:s1]
        last = max(i for i, r in enumerate(seg) if "k_mk_gather" in r[0])
        d, v = seg[:last + 4], seg[last + 4:]
        dw += d[-1][2] - d[0][1]
        vw += v[-1][2] - v[0][1]
        for r in d: dr[short(r[0])] += r[2] - r[1]
        for r in v: vr[short(r[0])] += r[2] - r[1]
        n += 1
    rt = (rows[starts[-1]][1] - rows[starts[a.skip]][1]) / (len(starts) - a.skip - 1)
    print(f"{n} rounds: round {rt / 1e6:.1f} ms = draft wall {dw / n / 1e6:.1f} + verify wall {vw / n / 1e6:.1f} + host gaps")
    for name, dd in (("DRAFT", dr), ("VERIFY", vr)):
        print(f"{name} busy {sum(dd.values()) / n / 1e6:.1f} ms")
        for k, t in sorted(dd.items(), key=lambda x: -x[1])[:a.top]:
            print(f"  {t / n / 1e6:7.2f}  {k}")


if __name__ == "__main__":
    main()
