"""Per-token arm metrics from a replay run, against the s03 teacher cache (numpy only).

Scored tokens: position t of a row is scored when it is a real token (valid), t+1 exists in the
row, and t+1 is text (types < 0). Predicting an image-span token is meaningless (the model never
emits them), so those positions are excluded - GLM's s09 had to bolt this on after its vision
bucket came out meaningless.

Per scored token:  kl32 (GLM `topk_KL` definition, teacher top-32 support), agree (student top-1
== teacher top-1), dnll = student NLL - teacher NLL of the actual next token, confident (teacher
p_max >= 0.9). Grouped by row (the bootstrap cluster), bucket and log-position bin (drift.BIN_EDGES).
"caption" tokens are text tokens after the first image span of a vision-bucket row: caption
perplexity is exp(mean NLL) over them.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import drift, stats


def corpus(corpus_dir: Path):
    rows = [json.loads(l) for l in open(Path(corpus_dir) / "rows.jsonl")]
    return {"rows": rows, "tokens": np.load(Path(corpus_dir) / "tokens.npy", mmap_mode="r"),
            "types": np.load(Path(corpus_dir) / "types.npy", mmap_mode="r"),
            "valid": np.load(Path(corpus_dir) / "valid.npy", mmap_mode="r")}


def teacher(s03_dir: Path):
    d = Path(s03_dir) / "teacher"
    return {"top1": np.load(d / "top32_ids.npy", mmap_mode="r"), "lp": np.load(d / "top32_logprob.npy", mmap_mode="r"),
            "nll": np.load(d / "nll_next.npy", mmap_mode="r")}


def row_frame(C: dict, T: dict, r: int) -> dict:
    row = C["rows"][r]
    g = slice(row["offset"], row["offset"] + row["length"])
    n = row["length"]
    ty = np.asarray(C["types"][g])
    ok = np.asarray(C["valid"][g]).astype(bool).copy()
    ok[-1] = False
    ok[:-1] &= ty[1:] < 0
    first_img = int(np.argmax(ty >= 0)) if (ty >= 0).any() else n
    cap = ok & (np.arange(n) > first_img) & (ty < 0) & (row["bucket"] == "vision")
    return {"g": g, "n": n, "scored": ok, "caption": cap, "pos": np.arange(n), "bucket": row["bucket"],
            "t_top1": np.asarray(T["top1"][g, 0]), "t_nll": np.asarray(T["nll"][g], np.float64),
            "t_pmax": np.exp(np.asarray(T["lp"][g, 0], np.float64))}


def load_arm(run_root: Path, arm: str, rows: list[int], C: dict, T: dict) -> dict:
    cols = {k: [] for k in ("kl32", "agree", "dnll", "s_nll", "t_nll", "row", "bucket", "pos", "caption", "confident")}
    for r in rows:
        f = row_frame(C, T, r)
        m = np.load(Path(run_root) / arm / "metrics" / f"r{r:05d}.npz")
        s = f["scored"]
        for k in ("kl32", "nll"):
            if not np.isfinite(m[k][s]).all():
                raise ValueError(f"{arm} r{r}: {int((~np.isfinite(m[k][s])).sum())} non-finite scored {k} values")
        cols["kl32"].append(m["kl32"][s].astype(np.float64))
        cols["agree"].append(m["top1"][s] == f["t_top1"][s])
        cols["s_nll"].append(m["nll"][s].astype(np.float64))
        cols["t_nll"].append(f["t_nll"][s])
        cols["dnll"].append(m["nll"][s].astype(np.float64) - f["t_nll"][s])
        cols["row"].append(np.full(int(s.sum()), r))
        cols["bucket"].append(np.full(int(s.sum()), f["bucket"], dtype=object))
        cols["pos"].append(f["pos"][s])
        cols["caption"].append(f["caption"][s])
        cols["confident"].append(f["t_pmax"][s] >= 0.9)
    return {k: np.concatenate(v) if v else np.zeros(0) for k, v in cols.items()}


def _block(a: dict, sel: np.ndarray, n_draws: int, seed: int) -> dict:
    cl = a["row"][sel]
    out = {"kl32": stats.cluster_bootstrap_mean(a["kl32"][sel], cl, n_draws=n_draws, seed=seed),
           "top1": stats.cluster_bootstrap_mean(a["agree"][sel].astype(np.float64), cl, n_draws=n_draws, seed=seed),
           "dnll": stats.cluster_bootstrap_mean(a["dnll"][sel], cl, n_draws=n_draws, seed=seed)}
    k = a["kl32"][sel]
    out["kl32_p99"] = float(np.nanpercentile(k, 99)) if k.size else float("nan")
    c = sel & a["confident"]
    out["confident_flip_rate"] = float(1 - a["agree"][c].mean()) if c.any() else float("nan")
    out["confident_tokens"] = int(c.sum())
    return out


def summarize(a: dict, *, n_draws: int = 2000, seed: int = 0, min_tokens: int = 2000) -> dict:
    allm = np.ones(a["kl32"].size, bool)
    out = {"overall": _block(a, allm, n_draws, seed), "by_bucket": {}, "by_pos_bin": {}}
    for b in sorted(set(a["bucket"].tolist())):
        sel = a["bucket"] == b
        out["by_bucket"][b] = {**_block(a, sel, n_draws, seed), "under_sampled": bool(sel.sum() < min_tokens)}
    bins = drift.pos_bin(a["pos"])
    for k in sorted(set(bins.tolist())):
        sel = bins == k
        out["by_pos_bin"][drift.BIN_NAMES[k]] = {**_block(a, sel, n_draws, seed), "under_sampled": bool(sel.sum() < min_tokens)}
    cap = a["caption"]
    out["caption"] = {"tokens": int(cap.sum()),
                      "student_ppl": float(np.exp(np.nanmean(a["s_nll"][cap]))) if cap.any() else float("nan"),
                      "teacher_ppl": float(np.exp(np.nanmean(a["t_nll"][cap]))) if cap.any() else float("nan")}
    out["caption"]["ppl_ratio"] = out["caption"]["student_ppl"] / out["caption"]["teacher_ppl"] if cap.any() else float("nan")
    return out


def without_buckets(a: dict, buckets: list[str]) -> dict:
    """The same arm restricted to tokens outside `buckets` (every per-token column filtered alike)."""
    keep = ~np.isin(a["bucket"], list(buckets))
    return {k: v[keep] for k, v in a.items()}


def compare(a: dict, b: dict, *, n_draws: int = 2000, seed: int = 0) -> dict:
    """Paired arm a vs arm b on identical tokens (same rows): KL / dNLL differences (a - b, cluster
    bootstrap) and McNemar on top-1 agreement (z > 0: a agrees more)."""
    if a["row"].size != b["row"].size or not np.array_equal(a["row"], b["row"]) or not np.array_equal(a["pos"], b["pos"]):
        raise ValueError("arms were not scored on identical tokens")
    return {"d_kl32": stats.paired_diff(a["kl32"], b["kl32"], a["row"], n_draws=n_draws, seed=seed),
            "d_dnll": stats.paired_diff(a["dnll"], b["dnll"], a["row"], n_draws=n_draws, seed=seed),
            "top1": stats.mcnemar(a["agree"], b["agree"])}
