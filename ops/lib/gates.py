"""Pre-registered gate evaluation (numpy-free logic over metrics.summarize / compare outputs).

The thresholds live in ops/s10_gates.json and are locked by sha256 at the first s10 result; this
module only applies them. Every verdict carries its reasons so a FAIL is never a bare boolean.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import drift, env

GATES_FILE = env.OPS / "s10_gates.json"


def load() -> tuple[dict, str]:
    raw = GATES_FILE.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def lock(work_dir: Path) -> dict:
    """First call writes the lock; later calls require the same sha. Fatal on change."""
    g, sha = load()
    p = Path(work_dir) / "prereg_lock.json"
    if p.exists():
        lk = json.load(open(p))
        if lk["sha256"] != sha:
            raise SystemExit(f"PRE-REGISTRATION VIOLATED: {GATES_FILE} changed after results were produced "
                             f"(locked {lk['sha256'][:16]} at {lk['ts']}, now {sha[:16]}). Revert it.")
        return g
    env.atomic_json(p, {"sha256": sha, "ts": env.now(), "file": str(GATES_FILE)})
    return g


def control(summary: dict, g: dict) -> dict:
    c = g["control"]
    o = summary["overall"]
    why = []
    if not o["kl32"]["mean"] <= c["max_kl32"]:
        why.append(f"control kl32 {o['kl32']['mean']:.5f} > {c['max_kl32']}")
    if not o["top1"]["mean"] >= c["min_top1"]:
        why.append(f"control top1 {o['top1']['mean']:.5f} < {c['min_top1']}")
    return {"pass": not why, "why": why}


def heal(cmp_healed_vs_unhealed: dict, g: dict) -> dict:
    d, t = cmp_healed_vs_unhealed["d_kl32"], cmp_healed_vs_unhealed["top1"]
    ok = d["hi"] < 0 and t["z"] > -1.96
    return {"adopt": bool(ok), "d_kl32": d, "top1_z": t["z"],
            "why": [] if ok else [f"d_kl32 CI hi {d['hi']:+.5f} (must be < 0) or top1 z {t['z']:+.2f} <= -1.96"]}


def pruned_floor(summary: dict, g: dict, *, vs_random: dict | None = None, drift_verdict: str | None = None) -> dict:
    f = g["release"]["pruned_floor"]
    o, why = summary["overall"], []
    if not o["kl32"]["mean"] <= f["max_kl32"]:
        why.append(f"kl32 {o['kl32']['mean']:.4f} > {f['max_kl32']}")
    if not o["top1"]["mean"] >= f["min_top1"]:
        why.append(f"top1 {o['top1']['mean']:.4f} < {f['min_top1']}")
    notes = []
    for b, s in summary["by_bucket"].items():
        ov = f.get("bucket_overrides", {}).get(b)
        lo = ov["min_top1"] if ov else f["min_bucket_top1"]
        if not s["under_sampled"] and not s["top1"]["mean"] >= lo:
            why.append(f"bucket {b} top1 {s['top1']['mean']:.4f} < {lo}")
        elif ov and not s["top1"]["mean"] >= f["min_bucket_top1"]:
            notes.append(f"bucket {b} top1 {s['top1']['mean']:.4f} is below min_bucket_top1 {f['min_bucket_top1']}: "
                         f"passes only under its override floor {lo} (known limitation)")
    cap = summary["caption"]
    if cap["tokens"] and not cap["ppl_ratio"] <= f["max_caption_ppl_ratio"]:
        why.append(f"caption ppl ratio {cap['ppl_ratio']:.3f} > {f['max_caption_ppl_ratio']}")
    pb = summary["by_pos_bin"]
    ref = pb.get(f["ref_bin"])
    first = drift.BIN_NAMES.index(f["long_bins_from"])
    long_checked = []
    for name in drift.BIN_NAMES[first:]:
        s = pb.get(name)
        if s is None or s["under_sampled"] or ref is None:
            continue
        drop = ref["top1"]["mean"] - s["top1"]["mean"]
        long_checked.append(name)
        if not drop <= f["max_long_bin_top1_drop"]:
            why.append(f"bin {name} top1 {s['top1']['mean']:.4f} is {drop:.4f} below {f['ref_bin']}")
    if f.get("must_beat_random_kl32") and vs_random is not None and not vs_random["d_kl32"]["hi"] < 0:
        why.append(f"does not beat the best random anchor (d_kl32 CI hi {vs_random['d_kl32']['hi']:+.4f})")
    if drift_verdict is not None and drift_verdict in f["drift_verdict_not"]:
        why.append(f"drift verdict {drift_verdict}")
    if not long_checked and f.get("long_bins_required", True):
        why.append("no adequately sampled bin past 8k and long_bins_required: long-context retention untested")
    if not long_checked:
        notes.append("no adequately sampled bin past 8k: long-context retention UNTESTED")
    return {"pass": not why, "why": why, "notes": notes, "long_bins_checked": long_checked}


def thor_quant(summary: dict, cmp_vs_mxfp4: dict, g: dict, **kw) -> dict:
    t = g["release"]["thor_quant"]
    r = pruned_floor(summary, g, **kw) if t.get("apply_pruned_floor") else {"pass": True, "why": [], "notes": []}
    d = cmp_vs_mxfp4["d_kl32"]["mean"]
    if d > t["max_d_kl32_vs_mxfp4"]:
        r["why"].append(f"quantization adds kl32 {d:.4f} > {t['max_d_kl32_vs_mxfp4']}")
    r["pass"] = not r["why"]
    return r


def overlay(cmp_overlay_vs_base: dict, base: dict, ov: dict, g: dict) -> dict:
    t = g["release"]["overlay"]
    why = []
    hi = cmp_overlay_vs_base["d_kl32"]["hi"]
    if hi > t["max_d_kl32_upper"]:
        why.append(f"d_kl32 CI hi {hi:.4f} > {t['max_d_kl32_upper']}")
    dc = ov["overall"]["confident_flip_rate"] - base["overall"]["confident_flip_rate"]
    if dc > t["max_confident_flip_increase"]:
        why.append(f"confident flips +{dc:.4f} > {t['max_confident_flip_increase']}")
    return {"adopt": not why, "why": why, "d_confident_flip": dc}


def arm_decision(cmp_A_vs_B: dict, g: dict) -> dict:
    d, t = cmp_A_vs_B["d_kl32"], cmp_A_vs_B["top1"]
    if d["excludes_zero"]:
        win, why = ("A" if d["mean"] < 0 else "B"), f"paired d_kl32 {d['mean']:+.4f} CI excludes 0"
    elif t["z"] != 0:
        win, why = ("A" if t["z"] > 0 else "B"), f"kl32 tie (CI includes 0); McNemar z {t['z']:+.2f}"
    else:
        win, why = "B", "exact tie: B (bit-exact, proven hardware FP4 GEMV)"
    return {"winner": win, "why": why, "d_kl32": d, "top1": t}


def gemv_time(alloc: dict, bench: dict, M: int = 1) -> float:
    """Modelled routed-expert GEMV time per token (us, relative): for every layer, the gate|up pass
    in its w13 format plus the down pass in its w2 format, from s08 gemv_bench rows at M. Both builds
    run the same experts per token, so the ratio of two builds is the ratio of their expert decode time."""
    us = {(r["format"], r["shape"]): r["us"] for r in bench["rows"] if r["M"] == M}
    t = 0.0
    for L, a in alloc.items():
        t += us[(a["w13"], "gate|up")] + us[(a["w2"], "down")]
    return t


def trellis(cmp_kt_vs_kq: dict, g: dict, *, kt_bytes: int, kq_bytes: int, t_kt: float | None, t_kq: float | None) -> dict:
    t = g["trellis"]
    d, tp = cmp_kt_vs_kq["d_kl32"], cmp_kt_vs_kq["top1"]
    why = []
    if not d["hi"] < t["max_d_kl32_hi"]:
        why.append(f"d_kl32 CI hi {d['hi']:+.5f} (must be < {t['max_d_kl32_hi']})")
    if not tp["z"] > t["min_top1_z"]:
        why.append(f"top-1 McNemar z {tp['z']:+.2f} (must be > {t['min_top1_z']})")
    if kt_bytes > kq_bytes:
        why.append(f"KT expert bytes {kt_bytes} > K-quant {kq_bytes}: not a matched-bytes comparison")
    ratio = None
    if t_kt is None or t_kq is None:
        why.append("no s08 gemv_bench result: decode speed unproven")
    else:
        ratio = t_kt / t_kq
        if ratio > t["max_gemv_time_ratio"]:
            why.append(f"modelled expert GEMV time {ratio:.3f}x the K-quant build's (max {t['max_gemv_time_ratio']})")
    return {"adopt": not why, "why": why, "d_kl32": d, "top1": tp, "gemv_time_ratio": ratio,
            "kt_expert_bytes": kt_bytes, "kq_expert_bytes": kq_bytes}
