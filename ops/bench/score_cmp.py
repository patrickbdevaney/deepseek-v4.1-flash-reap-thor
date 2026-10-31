#!/usr/bin/env python3
"""KL(arm0 || armN), top-1 agreement and forced-token NLL from score_arms.sh logits files.
   score_cmp.py <forced_ids.txt> arm0.f32 armN.f32 ..."""
import sys, numpy as np
V = 129280
forced = np.array(open(sys.argv[1]).read().split(), dtype=np.int64)
def load(p):
    x = np.fromfile(p, dtype=np.float32).reshape(-1, V).astype(np.float64)
    x -= x.max(1, keepdims=True); lp = x - np.log(np.exp(x).sum(1, keepdims=True)); return lp
ref = load(sys.argv[2]); n = ref.shape[0]; pr = np.exp(ref)
print(f"ref {sys.argv[2]}: {n} positions, NLL {-ref[np.arange(n), forced[:n]].mean():.5f}")
for p in sys.argv[3:]:
    lp = load(p)
    kl = (pr * (ref - lp)).sum(1)
    top = (lp.argmax(1) == ref.argmax(1)).mean()
    nll = -lp[np.arange(n), forced[:n]].mean()
    print(f"{p}: KL mean {kl.mean():.3e} max {kl.max():.3e}  top1 {top:.4f}  NLL {nll:.5f}")
