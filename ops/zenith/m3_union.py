#!/usr/bin/env python3
"""M3: unique routed experts per verify block, from a V41_ROUTE_DUMP of v41_decode --route-trace.

    ops/zenith/m3_union.py ops/logs/m3/route_train.bin ~/models/v41-work/d2/trace_train/seqs.jsonl

A verify of K drafts runs K+1 rows; its routed-expert bytes scale with the union of their experts per
layer. Reported per domain: mean union/6 over all MoE layers (= expert bytes vs one AR step), for
K = 0..4, over generated positions only (rows >= gen_start), plus the ratio per accepted token.
"""
import collections, json, sys
import numpy as np

N_ACT = 6
dump, seqs = sys.argv[1], sys.argv[2]
raw = np.fromfile(dump, dtype=np.int32)
meta = [json.loads(l) for l in open(seqs) if l.strip()]
recs, i = [], 0
while i < len(raw):
    L, M = int(raw[i]), int(raw[i + 1]); i += 2
    recs.append((L, raw[i:i + M * N_ACT].reshape(M, N_ACT))); i += M * N_ACT
# split into sequences: layer index restarts at the first MoE layer per prefill
first = recs[0][0]
groups, cur = [], []
for L, r in recs:
    if L == first and cur: groups.append(cur); cur = []
    cur.append((L, r))
groups.append(cur)
assert len(groups) == len(meta), (len(groups), len(meta))
Ks = [0, 1, 2, 3, 4]
acc = collections.defaultdict(lambda: collections.defaultdict(list))
layers = None
for m, g in zip(meta, groups):
    layers = layers or [L for L, _ in g]
    gs, n = m["gen_start"], m["length"]
    for L, r in g:
        assert len(r) == n, (m["id"], L, len(r), n)
        for K in Ks:
            u = [len(np.unique(r[t:t + K + 1])) for t in range(gs, n - K)]
            acc[m["domain"]][K] += u
            acc["ALL"][K] += u
print(f"{len(meta)} seqs, MoE layers {len(layers)} ({layers[0]}..{layers[-1]})")
print("union/6 = routed-expert bytes of one verify vs one AR step (perfectly cached reuse within the block)")
print(f"{'domain':13s}" + "".join(f"  K={K}  " for K in Ks) + "   K=3 per-row/6")
for d in sorted(acc, key=lambda d: (d == "ALL", d)):
    row = [np.mean(acc[d][K]) / N_ACT for K in Ks]
    print(f"{d:13s}" + "".join(f" {x:6.2f} " for x in row) + f"   {row[3] / 4:.3f}")
