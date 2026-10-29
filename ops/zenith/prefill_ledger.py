#!/usr/bin/env python3
"""Prefill ledger: an nsys trace of v41_decode on a long prompt -> kernel time by name (top N) and by family.

    ops/zenith/prefill_ledger.py ops/logs/prof/pf1.sqlite [--top 25]
"""
import argparse
import collections
import sqlite3
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import ledger  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite")
    ap.add_argument("--top", type=int, default=25)
    a = ap.parse_args()
    db = sqlite3.connect(a.sqlite)
    rows = list(db.execute("select s.value, k.gridX, k.end - k.start, k.start, k.end from CUPTI_ACTIVITY_KIND_KERNEL k "
                           "join StringIds s on k.shortName = s.id order by k.start"))
    t0, t1 = rows[0][3], rows[-1][4]
    by_name, by_fam, cnt = collections.defaultdict(float), collections.defaultdict(float), collections.Counter()
    for n, gx, d, s, e in rows:
        by_name[n] += d / 1e9; cnt[n] += 1
        by_fam[ledger.family(n, gx)] += d / 1e9
    tot = sum(by_name.values())
    print(f"trace span {(t1 - t0) / 1e9:.1f} s, kernel time {tot:.1f} s")
    for f, v in sorted(by_fam.items(), key=lambda kv: -kv[1]):
        print(f"  {f:16s} {v:8.2f} s  {100 * v / tot:5.1f}%")
    print()
    for n, v in sorted(by_name.items(), key=lambda kv: -kv[1])[:a.top]:
        print(f"  {v:8.2f} s  {cnt[n]:8d} x {1e6 * v / cnt[n]:9.1f} us  {n}")


if __name__ == "__main__":
    main()
