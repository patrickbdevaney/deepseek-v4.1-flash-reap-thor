#!/usr/bin/env python3
"""S5b ceiling from a held-out --spec-sweep 1,2,3,4 json: per-prompt oracle K vs fixed K and per-domain K.

    ops/zenith/s5b_ceiling.py ops/logs/m3/held_ksweep.json

Speedups are per-prompt tok/s over that prompt's no-spec run, averaged (as v41_decode reports).
Also prints the implied round cost T(K) in AR steps = tok_per_verify / speedup.
"""
import collections, json, sys
import numpy as np
rows = json.load(open(sys.argv[1]))["rows"]
Ks = [s["k"] for s in rows[0]["spec"]]
def sp(r, k): return 1.0 if k == 0 else next(s for s in r["spec"] if s["k"] == k)["tok_s"] / r["tok_s"]
dom = collections.defaultdict(list)
for r in rows: dom[r["domain"]].append(r)
print("domain   n  " + "  ".join(f"K={k}" for k in [0] + Ks) + "   dom-best  oracle/prompt   T(K) AR steps")
allr = {k: [] for k in [0] + Ks}; dbest = []; orc = []
for d, rs in sorted(dom.items()):
    m = {k: np.mean([sp(r, k) for r in rs]) for k in [0] + Ks}
    kb = max(m, key=m.get); o = np.mean([max(sp(r, k) for k in [0] + Ks) for r in rs])
    T = [np.mean([next(s for s in r["spec"] if s["k"] == k)["tok_per_verify"] / sp(r, k) for r in rs]) for k in Ks]
    print(f"{d:8s} {len(rs):2d}  " + "  ".join(f"{m[k]:.3f}" for k in [0] + Ks) + f"   K={kb} {m[kb]:.3f}  {o:.3f}        " + " ".join(f"{t:.2f}" for t in T))
    for k in allr: allr[k] += [sp(r, k) for r in rs]
    dbest += [sp(r, kb) for r in rs]; orc += [max(sp(r, k) for k in [0] + Ks) for r in rs]
print("ALL      " + f"{len(rows):2d}  " + "  ".join(f"{np.mean(allr[k]):.3f}" for k in allr) + f"   per-domain K {np.mean(dbest):.3f}  oracle {np.mean(orc):.3f}")
