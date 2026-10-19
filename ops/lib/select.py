"""Candidate keep-sets from the s03 bank (numpy only; s04 runs it on the CPU).

Inputs per layer L (exactly what s03 wrote):
  bank/L{L}.npz   sum/sq/cnt/nrm/nsq/gat/gsq [n_buckets, E] (held-out rows and pads excluded),
                  F_sum/F_cnt [E, E], fmax [E], buckets [n_buckets] (names, s02 BUCKETS order)
  routing/L{L}_idx.npy [T, 6] uint16, routing/L{L}_w.npy [T, 6] float16 (renormalised x 1.5,
                  bias_vl on image spans) - used here only for the per-modality image ranking
  corpus types.npy [T] int8 (-1 text, >= 0 image span), valid.npy, rows.jsonl (heldout flag)

Criteria (all "higher = keep"):
  reap_sum   sum over pools of g*||f||  (REAP-sum; GLM evidence: sum beats mean, 0.73 vs 0.85 KL)
  mass       sum of g                   (routing mass)
  freq       routed-slot count          (frequency)
  hope_sum   argmin p^T F p with F = F_sum (co-activation SUMS: the REAP-sum diagonal, "use sum, not
             mean"), p^T F p = sum over tokens of (sum of pruned g*||f||)^2
  hope_cond  the vendored conditional-mean F (F_sum / F_cnt), HOPE paper form
  random     uniform over the unprotected experts (anchor; never eligible to win)

Protection (never pruned, every candidate including random):
  * drift_guard protect lists (sink-coupled, Super-Expert in the sink, position-rising),
  * every Super-Expert candidate (max down_proj output above P99.5 and >= max/10, model-wide),
  * HOPE candidates additionally protect the top `pf` fraction of EVERY pool separately, where the
    pools are the s02 buckets plus "image" (image-span tokens ranked on their own, bias_vl routing):
    the per-modality vision ranking. pf is swept and the knee read off (not monotone near 0).
The shared expert, the ViT and the aligner are not routed experts and never enter a keep-set.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .hope import solve_prune_set

PF_GRID = [0.0, 0.005, 0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.12, 0.16]


def load_bank(d: Path, L: int) -> dict:
    z = np.load(Path(d) / "bank" / f"L{L}.npz", allow_pickle=False)
    return {k: z[k] for k in z.files}


def criteria(bank: dict) -> dict[str, np.ndarray]:
    return {"reap_sum": bank["sum"].sum(0), "mass": bank["gat"].sum(0), "freq": bank["cnt"].sum(0)}


def mean_norm(bank: dict, bucket: int | None) -> np.ndarray:
    """mean ||f|| per expert in one bucket, falling back to all buckets where the bucket never
    routed to the expert."""
    allm = bank["nrm"].sum(0) / np.maximum(bank["cnt"].sum(0), 1)
    if bucket is None:
        return allm
    c = bank["cnt"][bucket]
    return np.where(c > 0, bank["nrm"][bucket] / np.maximum(c, 1), allm)


def image_saliency(idx: np.ndarray, w: np.ndarray, token_mask: np.ndarray, norm: np.ndarray) -> np.ndarray:
    """REAP-sum restricted to image-span tokens: sum of g over image tokens routed to e (bias_vl
    selection, as banked) times e's mean ||f|| on vision rows. The bank does not split ||f|| by
    modality inside a row, so the norm is the vision-bucket mean - recorded as an approximation."""
    E = norm.size
    i = np.asarray(idx[token_mask], np.int64).ravel()
    g = np.asarray(w[token_mask], np.float64).ravel()
    return np.bincount(i, weights=g, minlength=E)[:E] * norm


def pools(bank: dict, img: np.ndarray | None) -> tuple[list[str], np.ndarray]:
    names = [str(b) for b in bank["buckets"]]
    P = bank["sum"].astype(np.float64)
    if img is not None:
        names = names + ["image"]
        P = np.vstack([P, img[None]])
    return names, P


def hope_F(bank: dict, form: str) -> np.ndarray:
    Fs = bank["F_sum"].astype(np.float64)
    if form == "sum":
        return Fs
    if form == "cond":
        return Fs / np.maximum(bank["F_cnt"], 1.0)
    raise ValueError(form)


def pool_protect(P: np.ndarray, pf: float, live: np.ndarray) -> np.ndarray:
    E = P.shape[1]
    if pf <= 0:
        return np.zeros(0, int)
    k = max(1, int(E * pf))
    return np.unique(np.concatenate([np.argsort(-P[b])[:k] for b in range(P.shape[0]) if live[b]]))


def retention(P: np.ndarray, pruned: np.ndarray) -> np.ndarray:
    keep = np.ones(P.shape[1], bool)
    keep[pruned] = False
    return P[:, keep].sum(1) / np.maximum(P.sum(1), 1e-30)


def icost(F: np.ndarray, pruned: np.ndarray) -> float:
    p = np.zeros(F.shape[0]); p[pruned] = 1.0
    den = float(F.sum())
    return float(p @ F @ p) / den if den > 0 else 0.0


class Infeasible(ValueError):
    pass


def prune_set(mode: str, *, n_keep: int, protect: np.ndarray, crit: dict | None = None,
              F: np.ndarray | None = None, seed: int = 0, E: int | None = None) -> np.ndarray:
    """-> sorted PRUNED ids. mode: hope (needs F) | reap_sum | mass | freq (need crit) | random."""
    E = E or (F.shape[0] if F is not None else next(iter(crit.values())).size)
    n_prune = E - n_keep
    prot = np.zeros(E, bool)
    prot[np.asarray(protect, int)] = True
    if prot.sum() > n_keep:
        raise Infeasible(f"{int(prot.sum())} protected experts > {n_keep} slots")
    if mode == "hope":
        return np.sort(solve_prune_set(F, n_prune, protect=np.flatnonzero(prot)))
    if mode == "random":
        rng = np.random.default_rng(seed)
        return np.sort(rng.choice(np.flatnonzero(~prot), n_prune, replace=False))
    s = crit[mode].astype(np.float64).copy()
    s[prot] = np.inf
    return np.sort(np.argsort(s, kind="stable")[:n_prune])


def keep_from_pruned(pruned: np.ndarray, E: int) -> list[int]:
    m = np.ones(E, bool)
    m[pruned] = False
    return np.flatnonzero(m).tolist()


def sweep_pf(layers: list[dict], n_keep: int, form: str, grid=PF_GRID) -> dict:
    """layers: per layer {"F", "P" [pools, E], "live" [pools], "guard" [ids]}. HOPE at every pf;
    worst-pool retention (mean over layers, worst pool) against interaction cost. Knee = the last
    grid step whose marginal gain per unit cost is >= 1 (GLM sweep_protect_frac: marginal, not
    cumulative, which would always name the smallest nonzero pf)."""
    rows = []
    for pf in grid:
        R, ic = [], []
        try:
            for d in layers:
                prot = np.union1d(d["guard"], pool_protect(d["P"], pf, d["live"]))
                pr = prune_set("hope", n_keep=n_keep, protect=prot, F=d["F"])
                R.append(retention(d["P"], pr)); ic.append(icost(d["F"], pr))
        except Infeasible as e:
            rows.append({"pf": pf, "infeasible": str(e)})
            break
        R = np.stack(R).mean(0)
        live = layers[0]["live"]
        w = np.where(live, R, np.inf)
        rows.append({"pf": pf, "worst": float(w.min()), "worst_pool": int(np.argmin(w)),
                     "mean": float(R[live].mean()), "icost": float(np.mean(ic)), "retention": R.tolist()})
    ok = [r for r in rows if "infeasible" not in r]
    knee = ok[0]["pf"] if ok else 0.0
    steps = []
    for a, b in zip(ok, ok[1:]):
        dw, di = b["worst"] - a["worst"], b["icost"] - a["icost"]
        ratio = dw / di if di > 0 else (float("inf") if dw > 0 else 0.0)
        steps.append({"from": a["pf"], "to": b["pf"], "d_worst": dw, "d_icost": di, "ratio": ratio})
        if ratio >= 1.0:
            knee = b["pf"]
    return {"n_keep": n_keep, "form": form, "rows": rows, "steps": steps, "knee": knee}
