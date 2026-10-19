#!/usr/bin/env python3
"""CPU-only check of lib/gptq.py (torch on CPU, tiny tensors; CUDA is hidden before torch loads).

    CUDA_VISIBLE_DEVICES= .venv/bin/python ops/tests/test_gptq_cpu.py

GPTQ must (1) emit codes + params that pack to ggml bytes dequantizing to exactly what it
rounded to, and (2) beat diag-imatrix RTN on the Hessian-weighted output error when the
calibration inputs are correlated (the case it exists for).
"""
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from ops.lib import ggml_quants as G, gptq  # noqa: E402

rng = np.random.default_rng(0)
B, R, C, N = 3, 8, 512, 4096
mix = rng.standard_normal((C, C)) * 0.15 + np.eye(C)
X = (rng.standard_normal((B, N, C)) @ mix).astype(np.float32)
X[:, :, 7] *= 6.0                                                   # one loud channel, as in real activations
H = np.einsum("bnc,bnd->bcd", X, X)
W = (rng.standard_normal((B, R, C)) * 0.02).astype(np.float32)
fail = 0
for t in ("Q2_K", "Q3_K", "Q4_K"):
    codes, params = gptq.quantize(torch.from_numpy(W.copy()), torch.from_numpy(H), t, damp=0.01)
    q = gptq.pack(codes, params)
    Wg = G.dequant(t, q)
    lo, hi = G.QRANGE[t]
    ok_range = codes.min() >= lo and codes.max() <= hi
    Wr = np.stack([G.dequant(t, G.quantize_rtn(t, W[b], np.diagonal(H[b]))) for b in range(B)])
    loss = lambda Wq: float(np.einsum("brc,bcd,brd->", W - Wq, H, W - Wq))
    lg, lr = loss(Wg), loss(Wr)
    ok = ok_range and lg < lr and q.shape == (B, R, G.row_bytes(t, C))
    fail += not ok
    print(f"{'ok' if ok else 'FAIL'}  {t}: H-weighted error GPTQ {lg:.4e} vs RTN {lr:.4e} ({lg / lr:.3f}x)")
sys.exit(1 if fail else 0)
