# Vendored from ~/glm-5.3-reap/scripts/hope_fmatrix.py (2026-10-19); only change: torch is imported lazily so the numpy solver is testable without torch. V4.1: E=384, K=6, 40 layers -> 40*384*384*8 B*2 = 94 MiB.
"""HOPE: the one calibration statistic that cannot be recovered offline.

arXiv 2609.18916 shows REAP is a special case of HOPE with the interaction terms zeroed, and at
40-50% pruning -- our forced ratio -- HOPE beat REAP in EVERY agentic experiment (mean +2.8%, up
to +6.1%) on a model with MiMo's exact shape (256 experts, top-8, 48 MoE layers).

    F[i,j] = (1/|X_ij|) * sum over tokens where BOTH i,j routed of  g_i*||f_i|| * g_j*||f_j||
    prune-set = argmin_p  p^T F p   s.t.  p in {0,1}^E,  sum(p) = n_prune

WHY THIS FILE EXISTS SEPARATELY FROM THE CRITERION WORK. Every criterion in the unified family
of arXiv 2606.15716 -- Frequency (0,0,0), SEER (0,1,0), EAN (0,0,1), REAP (1,1,1), MAN (1,0,1),
MSAN (1,0,2), and the task-specific winners (0,1,1) and (0,2,2) -- is already computable offline
from the per-expert accumulators the saliency pass keeps (`sum`, `sq`, `nrm`, `nsq`, `gat`,
`cnt`). They can all be re-derived after the fact, for free, as often as we like.

The off-diagonal of F cannot. It needs the per-token CO-ACTIVATION structure, which no
per-expert accumulator retains. Miss it in the pass and the only way back is another full
forward pass over the whole calibration corpus. That is what makes this the single step with a
deadline.

Cost: for MiMo (47 MoE layers, 256 experts) the accumulators are 47*256*256*8 B * 2 = 49.3 MB,
and the per-token work is one 8x8 outer product of values the expert loop already computes.
"""
from __future__ import annotations

import numpy as np


class FAccumulator:
    """Per-layer E x E co-activation accumulators, float64 on CPU.

    float64 because these are long running sums over millions of tokens and the whole point is a
    second-order term; float32 drift here would be indistinguishable from the signal we are
    trying to measure.

    ON THE COMPUTE DEVICE by default in the pass. The first version kept these on CPU, reasoning
    that 47 MiB had no business competing with the model for device memory. That was the wrong
    trade and it was measured, not argued: a per-batch device-to-host copy of the [N,8,8] pair
    tensor plus a CPU index_add_ cost 0.15 s on top of a 0.20 s layer forward -- 75% overhead --
    while the 47 MiB it saved is 0.04% of the envelope.
    """

    def __init__(self, n_layers: int, n_experts: int, device="cpu"):
        import torch
        self.E = n_experts
        self.device = device
        self.sum = torch.zeros(n_layers, n_experts, n_experts, dtype=torch.float64, device=device)
        self.cnt = torch.zeros(n_layers, n_experts, n_experts, dtype=torch.float64, device=device)

    def update(self, layer: int, topk_idx: torch.Tensor, s: torch.Tensor) -> None:
        """topk_idx [N, K] expert ids per token; s [N, K] = g * ||f|| for those same slots.

        Accumulates BOTH orders (i,j) and (j,i) and the diagonal, so F comes out symmetric with
        F[k,k] = sum over the expert's own tokens of (g*||f||)^2 -- which is exactly the
        second-moment form the paper gives for the diagonal, and differs from the squared REAP
        score by the variance of the expert's contribution.
        """
        if topk_idx.numel() == 0:
            return
        import torch
        idx = topk_idx.to(torch.long).to(self.device)
        v = s.to(torch.float64).to(self.device)
        N, K = idx.shape
        # [N,K,K] outer products; pair (a,b) of slots -> experts (idx[:,a], idx[:,b]).
        pair = v[:, :, None] * v[:, None, :]
        flat = (idx[:, :, None] * self.E + idx[:, None, :]).reshape(-1)
        self.sum[layer].view(-1).index_add_(0, flat, pair.reshape(-1))
        self.cnt[layer].view(-1).index_add_(0, flat, torch.ones_like(pair).reshape(-1))

    def finalize(self, layer: int) -> np.ndarray:
        """CONDITIONAL average, dividing by |X_ij| not by N.

        This follows REAP's own design choice, which the paper is explicit about: it stops pairs
        that are rarely co-selected but contribute strongly whenever they ARE from being
        undervalued. It also means F is not a Gram matrix and need not be PSD, so the QP below is
        a local search with a good init rather than a convex solve.
        """
        c = self.cnt[layer].clone()
        c[c == 0] = 1.0
        return (self.sum[layer] / c).cpu().numpy()


def _project_capped_simplex(p: np.ndarray, k: float) -> np.ndarray:
    """Project onto {0 <= p <= 1, sum p = k} by bisection on the dual variable."""
    lo, hi = (p - 1.0).min(), p.max()
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if np.clip(p - mid, 0.0, 1.0).sum() > k:
            lo = mid
        else:
            hi = mid
    return np.clip(p - 0.5 * (lo + hi), 0.0, 1.0)


def _objective(F: np.ndarray, P: np.ndarray) -> float:
    return float(F[np.ix_(P, P)].sum())


def _swap_refine(F: np.ndarray, P: np.ndarray, banned: np.ndarray) -> np.ndarray:
    """Greedy 1-swap to a local optimum of p^T F p, with O(1) incremental evaluation.

    The projected-gradient relaxation alone reached the brute-force optimum on only 27 of 40
    random dense 8x8 problems. Real F matrices are far more benign -- the paper reports
    continuous solutions concentrating near 0 and 1 -- but "benign in their experiments" is not
    a property we can assert about MiMo's F, and this step is cheap enough that we need not.
    At E=256, |P|=128 a sweep is 128*128 swap evaluations, each O(1) given the cached row sums.

    r[x] = sum over i in P of F[x,i], so removing `a` then adding `b` changes the objective by
    (-2 r[a] + F[a,a]) + (2 (r[b] - F[b,a]) + F[b,b]).
    """
    E = F.shape[0]
    P = np.sort(P.copy())
    inP = np.zeros(E, dtype=bool); inP[P] = True
    r = F[:, P].sum(1)
    for _ in range(200):
        outs = np.flatnonzero(~inP & ~banned)
        if outs.size == 0:
            break
        best = (0.0, -1, -1)
        for a in P:
            d_rm = -2.0 * r[a] + F[a, a]
            d_add = 2.0 * (r[outs] - F[outs, a]) + F[outs, outs]
            j = int(np.argmin(d_add))
            delta = d_rm + d_add[j]
            if delta < best[0] - 1e-12:
                best = (delta, int(a), int(outs[j]))
        if best[1] < 0:
            break
        a, b = best[1], best[2]
        inP[a] = False; inP[b] = True
        r += F[:, b] - F[:, a]
        P = np.flatnonzero(inP)
    return np.sort(P)


def solve_prune_set(F: np.ndarray, n_prune: int, iters: int = 300, protect=None,
                    restarts: int = 4, seed: int = 0) -> np.ndarray:
    """Return the indices to PRUNE: argmin p^T F p over {0,1}^E with sum(p) = n_prune.

    Relaxation with a REAP init, rounded, then refined by greedy swap; repeated from a few random
    inits and the best kept. With the off-diagonals zeroed this returns the REAP prune-set
    unchanged, which is both the paper's reported control (Figure S4) and the gate's.
    `protect` marks experts that must never be pruned -- MiMo has no shared expert
    (n_shared_experts is null) but sibling architectures do, and those are never pruned.
    """
    E = F.shape[0]
    assert F.shape == (E, E), "F must be square"
    assert 0 < n_prune < E, f"n_prune {n_prune} out of range for {E} experts"
    banned = np.zeros(E, dtype=bool)
    if protect is not None and len(protect):
        banned[np.asarray(protect, dtype=int)] = True

    d = np.diag(F).copy()
    d[banned] = np.inf
    reap = np.sort(np.argsort(d)[:n_prune])
    if not np.any(F - np.diag(np.diag(F))):          # no interaction -> REAP is exactly optimal
        return reap

    rng = np.random.default_rng(seed)
    step = 1.0 / (np.abs(F).sum(1).max() + 1e-12)
    best_P, best_v = None, np.inf
    for t in range(max(1, restarts)):
        p = np.zeros(E)
        if t == 0:
            p[reap] = 1.0
        else:
            p[rng.choice(np.flatnonzero(~banned), n_prune, replace=False)] = 1.0
        for _ in range(iters):
            p = _project_capped_simplex(p - step * (2.0 * F @ p), float(n_prune))
        p[banned] = -np.inf
        P = _swap_refine(F, np.sort(np.argsort(-p)[:n_prune]), banned)
        v = _objective(F, P)
        if v < best_v:
            best_P, best_v = P, v
    return best_P
