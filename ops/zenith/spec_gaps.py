#!/usr/bin/env python3
"""S10: where a spec round's wall time goes. GPU busy time by kernel family, and GPU-idle gaps
(host syncs, Engram reads, draft D2H) labelled by the kernels around them, over the decode phase.

    ops/zenith/spec_gaps.py ops/logs/prof/sp3.sqlite DECODE_MS
"""
import collections, sqlite3, sys
db = sqlite3.connect(sys.argv[1])
k = db.execute("select k.start, k.end, s.value, k.gridX from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName order by k.start").fetchall()
# decode phase: the last DECODE_MS (argv[2], from the run's log) of the trace
t1 = k[-1][1]; k = [x for x in k if x[0] >= t1 - float(sys.argv[2]) * 1e6]
wall = (k[-1][1] - k[0][0]) / 1e6
busy = collections.Counter(); gaps = collections.Counter(); gapn = collections.Counter()
for i, (s, e, n, g) in enumerate(k):
    busy[n] += e - s
    if i + 1 < len(k):
        d = k[i + 1][0] - e
        if d > 20_000:  # > 20 us idle
            key = f"{n[:28]} -> {k[i+1][2][:28]}"
            gaps[key] += d; gapn[key] += 1
tb = sum(busy.values()) / 1e6; tg = sum(gaps.values()) / 1e6
print(f"decode window {wall:.0f} ms: kernels {tb:.0f} ms ({tb/wall:.0%}), gaps>20us {tg:.0f} ms ({tg/wall:.0%})")
print("top gaps (total ms, count, mean us):")
for key, v in gaps.most_common(12): print(f"  {v/1e6:8.1f} {gapn[key]:5d} {v/gapn[key]/1e3:8.0f}  {key}")
print("top kernels (ms):")
for n, v in busy.most_common(15): print(f"  {v/1e6:8.1f}  {n[:80]}")
