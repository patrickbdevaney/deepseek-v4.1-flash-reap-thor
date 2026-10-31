#!/usr/bin/env python3
"""Max-only banking of headline numbers (H5): docs/zenith/RATCHET.tsv keeps the best value ever measured
per metric, with its evidence. A new value is banked only when it beats the bank in the metric's
direction; a regression is reported (exit 1) and never overwrites.
  ratchet.py show
  ratchet.py bank <metric> <value> "<evidence: log path / commit>"     [--dir max|min for a new metric]"""
import sys, os, datetime
P = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'docs', 'zenith', 'RATCHET.tsv')
HDR = ['metric', 'dir', 'value', 'date', 'evidence']

def load():
    if not os.path.exists(P): return {}
    rows = [l.rstrip('\n').split('\t') for l in open(P) if l.strip()]
    return {r[0]: dict(zip(HDR, r)) for r in rows[1:]}

def save(b):
    with open(P, 'w') as f:
        f.write('\t'.join(HDR) + '\n')
        for m in sorted(b): f.write('\t'.join(b[m][h] for h in HDR) + '\n')

def main(a):
    b = load()
    if not a or a[0] == 'show':
        for m in sorted(b): print(f"{m:28s} {b[m]['dir']}  {b[m]['value']:>10s}  {b[m]['date']}  {b[m]['evidence']}")
        return 0
    if a[0] != 'bank' or len(a) < 4: print(__doc__); return 2
    m, v, ev = a[1], float(a[2]), a[3]
    d = a[a.index('--dir') + 1] if '--dir' in a else None
    today = datetime.date.today().isoformat()
    if m not in b:
        if d not in ('max', 'min'): print(f"new metric {m}: pass --dir max|min"); return 2
        b[m] = dict(metric=m, dir=d, value=f'{v:g}', date=today, evidence=ev); save(b)
        print(f"[ratchet] {m} opened at {v:g}"); return 0
    old = float(b[m]['value']); better = v > old if b[m]['dir'] == 'max' else v < old
    if not better:
        print(f"[ratchet] {m} {v:g} does not beat bank {old:g} ({b[m]['evidence']})"); return 1
    b[m].update(value=f'{v:g}', date=today, evidence=ev); save(b)
    print(f"[ratchet] {m} {old:g} -> {v:g}"); return 0

if __name__ == '__main__': sys.exit(main(sys.argv[1:]))
