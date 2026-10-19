"""Paired statistics for arm-vs-teacher and arm-vs-arm comparisons (numpy only).

Tokens are not independent: dNLL's mean is ~160x its median on GLM and errors cluster within a
document. Every interval here is a CLUSTER bootstrap over rows (documents), never a per-token
Gaussian SE (PLAN.md, "Proving best agreement"). Top-1 comparisons between two arms on the same
tokens are McNemar-paired.
"""
from __future__ import annotations

import math

import numpy as np


def cluster_sums(values: np.ndarray, clusters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """-> (per-cluster sum, per-cluster count), clusters relabelled 0..C-1 in order of appearance."""
    _, inv = np.unique(clusters, return_inverse=True)
    C = inv.max() + 1 if inv.size else 0
    s = np.bincount(inv, weights=values.astype(np.float64), minlength=C)
    n = np.bincount(inv, minlength=C).astype(np.float64)
    return s, n


def cluster_bootstrap_mean(values: np.ndarray, clusters: np.ndarray, *, n_draws: int = 2000,
                           seed: int = 0, alpha: float = 0.05) -> dict:
    """Ratio-of-sums mean (token-weighted, as the point estimate) with a percentile CI from
    resampling whole clusters."""
    values = np.asarray(values, np.float64)
    ok = np.isfinite(values)
    s, n = cluster_sums(values[ok], np.asarray(clusters)[ok])
    if n.sum() == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n_tokens": 0, "n_clusters": 0}
    rng = np.random.default_rng(seed)
    C = s.size
    draws = np.empty(n_draws)
    for d in range(n_draws):
        i = rng.integers(0, C, C)
        draws[d] = s[i].sum() / max(n[i].sum(), 1.0)
    return {"mean": float(s.sum() / n.sum()), "lo": float(np.percentile(draws, 100 * alpha / 2)),
            "hi": float(np.percentile(draws, 100 * (1 - alpha / 2))), "n_tokens": int(n.sum()), "n_clusters": int(C)}


def paired_diff(a: np.ndarray, b: np.ndarray, clusters: np.ndarray, **kw) -> dict:
    """mean(a - b) over tokens where both are finite, cluster-bootstrapped. Negative = a lower."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    r = cluster_bootstrap_mean(a[ok] - b[ok], np.asarray(clusters)[ok], **kw)
    r["excludes_zero"] = bool(r["hi"] < 0 or r["lo"] > 0)
    return r


def mcnemar(a_ok: np.ndarray, b_ok: np.ndarray) -> dict:
    """Paired top-1 test. a_ok/b_ok: bool per token (arm agrees with the teacher's top-1).
    z > 0 means arm a agrees more often. Continuity-corrected two-sided p."""
    a_ok, b_ok = np.asarray(a_ok, bool), np.asarray(b_ok, bool)
    b_only = int((a_ok & ~b_ok).sum())    # a right, b wrong
    c_only = int((~a_ok & b_ok).sum())
    n = b_only + c_only
    z = (b_only - c_only) / math.sqrt(n) if n else 0.0
    zc = max(abs(b_only - c_only) - 1, 0) / math.sqrt(n) if n else 0.0
    p = math.erfc(zc / math.sqrt(2))
    return {"a_only": b_only, "b_only": c_only, "discordant": n, "z": z, "p": p,
            "delta": (b_only - c_only) / max(a_ok.size, 1)}


def model_confidence_set(losses: dict[str, np.ndarray], clusters: np.ndarray, *, alpha: float = 0.10,
                         n_draws: int = 1000, seed: int = 0) -> dict:
    """Hansen-Lunde-Nason MCS with the T_max statistic over cluster means of per-token loss.
    losses: arm -> per-token loss (same tokens, same order). Returns the surviving set and the
    elimination order with p-values. "Best" may be claimed only if one arm survives."""
    names = list(losses)
    L = np.stack([np.asarray(losses[k], np.float64) for k in names])          # [m, T]
    ok = np.isfinite(L).all(0)
    L, cl = L[:, ok], np.asarray(clusters)[ok]
    _, inv = np.unique(cl, return_inverse=True)
    C = inv.max() + 1
    Lc = np.stack([np.bincount(inv, weights=l, minlength=C) / np.bincount(inv, minlength=C) for l in L])  # [m, C]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, C, (n_draws, C))
    alive = list(range(len(names)))
    elim, p_run = [], 0.0
    while len(alive) > 1:
        X = Lc[alive]                                   # [k, C]
        dbar = X.mean(1) - X.mean()                     # each arm vs the average of the alive set
        boot = np.stack([X[:, i].mean(1) - X[:, i].mean() for i in idx])      # [draws, k]
        se = np.sqrt(((boot - dbar) ** 2).mean(0)) + 1e-300
        t = dbar / se
        tmax = t.max()
        tb = ((boot - dbar) / se).max(1)
        p = float((tb >= tmax).mean())
        p_run = max(p_run, p)
        if p_run >= alpha:
            break
        worst = alive[int(np.argmax(t))]
        elim.append({"arm": names[worst], "p": p_run})
        alive.remove(worst)
    return {"alpha": alpha, "survivors": [names[i] for i in alive], "eliminated": elim, "clusters": int(C)}
