"""Position-binned routing-drift test and the long-context mask guard.

Method: notes/long_context_routing_drift.md, KQ3/KQ4. Pure numpy; it reads the teacher routing
s03 banks (per layer: top-6 expert ids + renormalised x route_scale weights per token) and a
candidate prune set, so every candidate mask can be checked offline without another forward.

Per (layer, log-position bin, row) three rates of the candidate PRUNE set:
  slot   = pruned slots / (K x tokens)
  token  = tokens with >= 1 pruned slot / tokens
  mass   = teacher gate weight on pruned slots / all gate weight      <- primary (REAP-aligned)
plus the rank-stratified top-1 rate (the token's highest-weight expert was pruned).

The row is the independence unit (tokens in a document are not independent), so intervals are a
cluster bootstrap over rows, drift is tested paired within row against the reference bin
[256, 1k), and the across-position spread is compared with the across-row same-position control.

Decision rules (KQ4), all reported, the first two blocking:
  (a) SINK: a pruned expert is sink-coupled (its mass share in bin [0, 8) is far above its share
      elsewhere) or a Super-Expert candidate (max down_proj output, s03 `fmax`) that the teacher
      routes in [0, 8) -> BLOCK the mask.
  (a') RISING: a pruned expert whose gate-mass share rises with log-position: row-bootstrap CI of
      the within-row slope (bins past [0,8)) above zero AND an effect size (share in the two
      furthest populated bins >= 1.5x its share in [8,1k), on pooled mass) AND a non-trivial late share -> BLOCK.
  (b) the mask's mass-rate slope vs log-position has a CI excluding zero (positive) in >= 25% of
      layers -> POSITION_SENSITIVE: extend long-context calibration before using the mask.
  (c) otherwise FLAT.
"""
from __future__ import annotations

import math
import warnings

import numpy as np

BIN_EDGES = np.array([0, 8, 64, 256, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 1 << 40])
BIN_NAMES = ["[0,8)", "[8,64)", "[64,256)", "[256,1k)", "[1k,2k)", "[2k,4k)", "[4k,8k)",
             "[8k,16k)", "[16k,32k)", "[32k,64k)", "[64k,128k)", "[128k,)"]
REF_BIN = 3                      # [256, 1k)
SINK_BIN = 0                     # [0, 8)
LONG_FROM = 7                    # bins >= 8k: only populated by s02 longctx rows


def pos_bin(pos: np.ndarray) -> np.ndarray:
    return np.searchsorted(BIN_EDGES, pos, side="right") - 1


def _bin_centres() -> np.ndarray:
    lo = BIN_EDGES[:-1].astype(float)
    hi = np.minimum(BIN_EDGES[1:], 1 << 18).astype(float)
    return np.log2(np.sqrt(np.maximum(lo, 1) * hi))


def row_rates(idx: np.ndarray, w: np.ndarray, pruned: np.ndarray) -> dict[str, np.ndarray]:
    """idx [n, K] expert ids and w [n, K] weights of ONE row, in position order.
    pruned: bool [E]. Returns per-bin arrays (nan where the bin is empty)."""
    n, K = idx.shape
    b = pos_bin(np.arange(n))
    hit = pruned[idx]                                    # [n, K]
    top1 = hit[np.arange(n), w.argmax(1)]
    nb = len(BIN_NAMES)
    cnt = np.bincount(b, minlength=nb).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = {
            "slot": np.bincount(b, hit.sum(1), nb) / (K * cnt),
            "token": np.bincount(b, hit.any(1), nb) / cnt,
            "mass": np.bincount(b, (w * hit).sum(1), nb) / np.bincount(b, w.sum(1), nb),
            "top1": np.bincount(b, top1, nb) / cnt,
        }
    for v in out.values():
        v[cnt == 0] = np.nan
    return out


def bootstrap(R: np.ndarray, fn, n_draws: int = 2000, seed: int = 0) -> np.ndarray:
    """Resample rows of R [rows, ...] with replacement; returns fn applied per draw."""
    rng = np.random.default_rng(seed)
    n = R.shape[0]
    return np.stack([fn(R[rng.integers(0, n, n)]) for _ in range(n_draws)])


def within_row_slope(R: np.ndarray, first_bin: int = 0) -> float:
    """Pooled OLS slope of R [rows, bins] on log2 bin centre with row fixed effects (demeaned
    within row), ignoring empty cells. nan if fewer than two bins carry data."""
    x = np.broadcast_to(_bin_centres()[first_bin: first_bin + R.shape[1]], R.shape)
    m = np.isfinite(R)
    if m.sum() < 2:
        return float("nan")
    rows = m.sum(1) >= 2
    if not rows.any():
        return float("nan")
    Rm, xm, mm = R[rows], x[rows], m[rows]
    xbar = np.nansum(np.where(mm, xm, 0), 1, keepdims=True) / mm.sum(1, keepdims=True)
    ybar = np.nansum(np.where(mm, Rm, 0), 1, keepdims=True) / mm.sum(1, keepdims=True)
    dx = np.where(mm, xm - xbar, 0.0)
    dy = np.where(mm, Rm - ybar, 0.0)
    den = (dx * dx).sum()
    return float((dx * dy).sum() / den) if den > 0 else float("nan")


def sign_test(d: np.ndarray) -> float:
    """Two-sided exact sign test p-value on the non-zero, finite entries of d."""
    d = d[np.isfinite(d) & (d != 0)]
    n, k = d.size, int((d > 0).sum())
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def expert_mass(idx: np.ndarray, w: np.ndarray, E: int) -> np.ndarray:
    """Gate mass of every expert per bin for ONE row: [E, bins]."""
    n, K = idx.shape
    b = np.repeat(pos_bin(np.arange(n)), K)
    mass = np.zeros((E, len(BIN_NAMES)))
    np.add.at(mass, (idx.reshape(-1), b), w.reshape(-1))
    return mass


def expert_share(idx: np.ndarray, w: np.ndarray, E: int) -> np.ndarray:
    """Per-bin gate-mass share of every expert for ONE row: [E, bins] (nan for empty bins)."""
    mass = expert_mass(idx, w, E)
    tot = mass.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        s = mass / tot
    s[:, tot == 0] = np.nan
    return s


def super_expert_candidates(fmax: np.ndarray, pct: float = 99.5) -> np.ndarray:
    """fmax [layers, E] max |down_proj output| (s03 bank). The note's rule: above the 99.5th
    percentile across all experts AND >= 1/10 of the model-wide max. Returns bool [layers, E]."""
    thr = max(np.percentile(fmax, pct), fmax.max() / 10)
    return fmax >= thr


def layer_report(rows: list[tuple[np.ndarray, np.ndarray]], pruned: np.ndarray, *,
                 super_cand: np.ndarray | None = None, n_draws: int = 2000, seed: int = 0,
                 sink_ratio: float = 10.0, sink_floor: float = 1e-3, rise_floor: float = 1e-3,
                 rise_ratio: float = 1.5) -> dict:
    """rows: per calibration row (idx [n, K], w [n, K]) of one layer. pruned: bool [E]."""
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)   # empty bins are nan by design
        return _layer_report(rows, pruned, super_cand, n_draws, seed, sink_ratio, sink_floor, rise_floor, rise_ratio)


def _layer_report(rows, pruned, super_cand, n_draws, seed, sink_ratio, sink_floor, rise_floor, rise_ratio) -> dict:
    E = pruned.size
    rr = [row_rates(i, w, pruned) for i, w in rows]
    rep = {"bins": BIN_NAMES, "rows": len(rows)}
    for key in ("mass", "slot", "token", "top1"):
        R = np.stack([r[key] for r in rr])
        draws = bootstrap(R, lambda X: np.nanmean(X, 0), n_draws, seed)
        rep[key] = {"mean": np.nanmean(R, 0).tolist(),
                    "lo": np.nanpercentile(draws, 2.5, 0).tolist(), "hi": np.nanpercentile(draws, 97.5, 0).tolist()}
    M = np.stack([r["mass"] for r in rr])
    d = M - M[:, [REF_BIN]]
    rep["paired_vs_ref"] = {"median_d": np.nanmedian(d, 0).tolist(),
                            "sign_p": [sign_test(d[:, k]) for k in range(d.shape[1])]}
    sd = bootstrap(M, within_row_slope, n_draws, seed)
    rep["mass_slope"] = {"est": within_row_slope(M), "lo": float(np.nanpercentile(sd, 2.5)),
                         "hi": float(np.nanpercentile(sd, 97.5))}
    # control: across-row spread at the same bin vs within-row spread across bins
    rep["control"] = {"across_row_same_bin_var": float(np.nanmean(np.nanvar(M, 0))),
                      "within_row_across_bin_var": float(np.nanmean(np.nanvar(M, 1)))}

    # per-expert share vs position
    Ms = np.stack([expert_mass(i, w, E) for i, w in rows])          # [rows, E, bins]
    with np.errstate(invalid="ignore", divide="ignore"):
        Sh = Ms / Ms.sum(1, keepdims=True)
    rest = np.nanmean(Sh[:, :, SINK_BIN + 1:], (0, 2))             # share away from the sink bin
    sink_share = np.nanmean(Sh[:, :, SINK_BIN], 0)
    sink_coupled = (sink_share >= sink_floor) & (sink_share >= sink_ratio * np.maximum(rest, 1e-12))
    # RISING. The sink bin is excluded (8 tokens, handled above). With hundreds of experts a CI
    # alone flags a few flat experts by chance, so a rising expert must also show an effect on
    # POOLED mass (per-row shares in the small early bins are too noisy): its pooled share in the
    # two furthest populated bins >= rise_ratio x its pooled share in [8, 1k) OF THE SAME ROWS.
    # Matched rows matter: only the few long rows reach the far bins, so an [8, 1k) baseline pooled
    # over every row compares long-document domains (books, code) against the whole corpus mix and
    # flags domain specialists as position-rising (2026-10-20: 35-48% of the flags on L2/L5/L13).
    pm = Ms.sum(0)                                                  # [E, bins] pooled over rows
    pop = np.flatnonzero(pm.sum(0) > 0)
    far = pop[-2:] if pop.size >= 2 else pop
    reach = Ms[:, :, far].sum((1, 2)) > 0                           # rows that populate the far bins
    pe = Ms[reach].sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        early = pe[:, 1:REF_BIN + 1].sum(1) / pe[:, 1:REF_BIN + 1].sum()
        late = pm[:, far].sum(1) / pm[:, far].sum()
    rising = np.zeros(E, bool)
    slopes = np.full(E, np.nan)
    cand = np.flatnonzero(pruned & (late >= rise_floor) & (late >= rise_ratio * np.maximum(early, 1e-12)))
    for e in cand:   # bootstrap only where it can change a decision
        Re = Sh[:, e, 1:]
        slopes[e] = within_row_slope(Re, first_bin=1)
        if np.isfinite(slopes[e]) and slopes[e] > 0:
            lo = np.nanpercentile(bootstrap(Re, lambda X: within_row_slope(X, first_bin=1), min(n_draws, 500), seed), 2.5)
            rising[e] = lo > 0
    routed_in_sink = np.zeros(E, bool)
    for i, _ in rows:
        routed_in_sink[np.unique(i[: BIN_EDGES[1]])] = True
    sc = super_cand if super_cand is not None else np.zeros(E, bool)
    rep["block_sink"] = sorted(int(e) for e in np.flatnonzero(pruned & (sink_coupled | (sc & routed_in_sink))))
    rep["block_rising"] = sorted(int(e) for e in np.flatnonzero(rising))
    rep["sink_coupled"] = sorted(int(e) for e in np.flatnonzero(sink_coupled))
    rep["rising_slope"] = {int(e): float(slopes[e]) for e in cand}
    return rep


def mask_verdict(layer_reports: dict[int, dict], sensitive_frac: float = 0.25) -> dict:
    """Combine per-layer reports into the KQ4 decision."""
    blocks = {L: {"sink": r["block_sink"], "rising": r["block_rising"]}
              for L, r in layer_reports.items() if r["block_sink"] or r["block_rising"]}
    pos = [L for L, r in layer_reports.items() if r["mass_slope"]["lo"] > 0]
    frac = len(pos) / max(1, len(layer_reports))
    verdict = "BLOCK" if blocks else ("POSITION_SENSITIVE" if frac >= sensitive_frac else "FLAT")
    return {"verdict": verdict, "blocking_experts": blocks, "rising_mass_layers": pos, "rising_mass_frac": frac}
