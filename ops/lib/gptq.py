"""GPTQ error-feedback rounding into ggml K-quant blocks, batched over experts (torch, GPU).

Standard GPTQ (Frantar et al.) with the column order left natural and the block parameters chosen
at each 256-column super-block from the CURRENT (error-updated) weights - the "groups decided when
reached" variant, which is what a K-quant super-block is: scales/mins per 16 or 32 columns, an
f16 super-scale, all along the input dimension. Parameters are fitted by ggml_quants.FIT with the
Hessian diagonal as per-column importance (the imatrix weighting llama.cpp uses), then every
column is rounded against those fixed parameters and its error is pushed onto the remaining
columns through the upper Cholesky factor of H^-1. Rows are independent given H, so w1 and w3
(same input) are quantized together, and B experts run as one batch.

The output is (codes int8 [B, R, C], params per super-block) and ggml_quants.PACK turns them into
the exact ggml bytes; nothing here invents a format.

Maya (GLM-5.3-Flash) reports ~25% lower expert output error from this at equal size, unverified;
s10's thor panel measures what it buys here.
"""
from __future__ import annotations

import numpy as np

from . import ggml_quants as G


def _hinv_upper(H, damp: float):
    import torch
    B, C, _ = H.shape
    H = H.clone()
    d = torch.diagonal(H, dim1=-2, dim2=-1)
    dead = d <= 0
    d[dead] = 1.0
    mean = d.mean(-1, keepdim=True)
    H += torch.diag_embed((damp * mean).expand(B, C))
    for attempt in range(6):
        L, info = torch.linalg.cholesky_ex(H)
        if not bool(info.any()):
            break
        H += torch.diag_embed((damp * mean * 10 ** attempt).expand(B, C))
    else:
        raise RuntimeError("H not positive definite after damping")
    Hinv = torch.cholesky_inverse(L)
    return torch.linalg.cholesky(Hinv, upper=True), torch.diagonal(H, dim1=-2, dim2=-1)


def quantize(W, H, qtype: str, *, damp: float = 0.01):
    """W [B, R, C] float32 cuda (modified in place), H [B, C, C] float32 cuda (sum of x x^T).
    -> (codes int8 numpy [B, R, C], params dict of numpy arrays with leading [B, R, C/256])."""
    import torch
    B, R, C = W.shape
    assert C % G.QK_K == 0
    U, diagH = _hinv_upper(H.float(), damp)
    lo, hi = G.QRANGE[qtype]
    sub = G.SUB[qtype]
    Q = torch.empty(B, R, C, dtype=torch.int8, device=W.device)
    plist = []
    imp = diagH.cpu().numpy()                                  # [B, C] importance (imatrix-like)
    for c0 in range(0, C, G.QK_K):
        c1 = c0 + G.QK_K
        Wb = W[:, :, c0:c1].float().cpu().numpy()
        p = G.FIT[qtype](Wb, imp[:, None, c0:c1])
        plist.append(p)
        S = torch.from_numpy(np.repeat(p["S"], sub, -1).astype(np.float32)).to(W.device)   # [B, R, 256]
        Mn = torch.from_numpy(np.repeat(p["M"], sub, -1).astype(np.float32)).to(W.device)
        Err = torch.empty(B, R, G.QK_K, dtype=torch.float32, device=W.device)
        for k in range(G.QK_K):
            c = c0 + k
            w = W[:, :, c]
            s, m = S[:, :, k], Mn[:, :, k]
            q = torch.where(s != 0, torch.round((w + m) / torch.where(s != 0, s, torch.ones_like(s))), torch.zeros_like(s))
            q = q.clamp(lo, hi)
            Q[:, :, c] = q.to(torch.int8)
            err = (w - (s * q - m)) / U[:, c, c][:, None]
            Err[:, :, k] = err
            if k + 1 < G.QK_K:
                W[:, :, c + 1:c1] -= err[:, :, None] * U[:, None, c, c + 1:c1]
        if c1 < C:
            W[:, :, c1:] -= torch.bmm(Err, U[:, c0:c1, c1:])
    params = {"type": qtype}
    for key in plist[0]:
        if key != "type":
            params[key] = np.stack([p[key] for p in plist], axis=2)        # [B, R, n_sb, ...]
    return Q.cpu().numpy(), params


def pack(codes: np.ndarray, params: dict) -> np.ndarray:
    """codes [B, R, C], params [B, R, n_sb, ...] -> ggml bytes uint8 [B, R, row_bytes]."""
    B, R, C = codes.shape
    nsb = C // G.QK_K
    flat = {k: (v.reshape(B * R * nsb, *v.shape[3:]) if isinstance(v, np.ndarray) else v) for k, v in params.items()}
    b = G.PACK[params["type"]](codes.reshape(B * R * nsb, G.QK_K), flat)
    return b.reshape(B, R, -1)


def rtn_pack(W: np.ndarray, A: np.ndarray | None, qtype: str) -> np.ndarray:
    """No-Hessian fallback (experts with too few calibration tokens): imatrix-weighted RTN."""
    return np.stack([G.quantize_rtn(qtype, w, A if A is None or A.ndim == 1 else A[i]) for i, w in enumerate(W)])
