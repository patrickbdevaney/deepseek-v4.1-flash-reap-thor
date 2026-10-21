#!/usr/bin/env python3
"""s09 release gate: which GGUF levels may be published (ops/s10_gates.json "gguf", pre-registered).

A level passes when
  * its teacher_kld.json exists (v41-teacher-kld) and was measured on the GGUF the ladder wrote (same keep-set
    fingerprint as the others), on the pre-registered split (gguf.split, "confirm"; a result that does not
    record its split fails);
  * kl32 <= max_kl32 and top-1 >= min_top1 (the pruned floor);
  * for L1..L3, when L0 was measured: kl32 - kl32(L0) <= max_d_kl32_vs_L0 (the requantization's own cost);
  * its llama-server smoke passed (ops/logs/s09_smoke_<LV>.log carries "smoke ok").
A level whose measurement is missing FAILS (never published unmeasured).

  python ops/s09_gate.py [--levels L0 L1 L2 L3]  -> $V41_WORK/s09/gates.json; prints the passing levels
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ops.lib import env, gates as G  # noqa: E402

OUT = env.WORK / "s09"


def evaluate(kld: dict, smoke_ok: dict, g: dict) -> dict:
    """kld: {LV: teacher_kld.json or None}; smoke_ok: {LV: bool}. -> {levels: {LV: {pass, why, ...}}, passing}."""
    res, fps = {}, {}
    l0 = kld.get("L0")
    for lv, k in kld.items():
        why, rec = [], {}
        if k is None:
            why.append("not measured (no teacher_kld.json)")
        else:
            s = k["stats"]
            rec = {"kl32": s["kl_top32"]["mean"], "kl32_ci95": s["kl_top32"]["ci95"], "top1": s["top1"]["mean"],
                   "top1_ci95": s["top1"]["ci95"], "dnll": s["dnll"]["mean"], "ngl": k.get("ngl"),
                   "keep_set_sha256": k["identity"]["keep_set_sha256"]}
            fps[lv] = rec["keep_set_sha256"]
            if k.get("split", g["split"]) != g["split"]:
                why.append(f"measured on split {k.get('split')!r}, the gate pre-registers {g['split']!r}")
            if rec["kl32"] > g["max_kl32"]:
                why.append(f"kl32 {rec['kl32']:.4f} > {g['max_kl32']}")
            if rec["top1"] < g["min_top1"]:
                why.append(f"top-1 {rec['top1']:.4f} < {g['min_top1']}")
            if lv in g["d_vs_L0_applies_to"] and l0 is not None:
                d = rec["kl32"] - l0["stats"]["kl_top32"]["mean"]
                rec["d_kl32_vs_L0"] = d
                if d > g["max_d_kl32_vs_L0"]:
                    why.append(f"kl32 vs L0 +{d:.4f} > {g['max_d_kl32_vs_L0']}")
        if g["smoke_required"] and not smoke_ok.get(lv):
            why.append("llama-server smoke not passed")
        res[lv] = {**rec, "pass": not why, "why": why}
    if len(set(fps.values())) > 1:
        for lv in res:
            res[lv]["pass"] = False
            res[lv]["why"].append(f"levels disagree on the keep-set: {sorted(set(fps.values()))}")
    return {"levels": res, "passing": [lv for lv, r in res.items() if r["pass"]]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", nargs="+", default=["L0", "L1", "L2", "L3"])
    a = ap.parse_args()
    allg, sha = G.load()
    g = allg["gguf"]
    kld, smoke = {}, {}
    for lv in a.levels:
        f = OUT / lv / "teacher_eval" / "teacher_kld.json"
        kld[lv] = json.load(open(f)) if f.exists() else None
        if kld[lv] is not None:
            kld[lv].setdefault("split", "unrecorded")       # pre-fix results scored both held-out halves
        if kld[lv] is not None and (OUT / lv / "teacher_eval" / "ngl.txt").exists():
            kld[lv]["ngl"] = (OUT / lv / "teacher_eval" / "ngl.txt").read_text().strip()
        lg = env.LOGS / f"s09_smoke_{lv}.log"
        smoke[lv] = lg.exists() and "smoke ok" in lg.read_text(errors="replace")
    out = evaluate(kld, smoke, g)
    out.update({"gates_sha256": sha, "gate": g, "ts": env.now()})
    env.atomic_json(OUT / "gates.json", out)
    for lv, r in out["levels"].items():
        env.log(f"{lv}: {'PASS' if r['pass'] else 'FAIL'} {'; '.join(r['why'])}", "s09_gate")
    print(" ".join(out["passing"]))


if __name__ == "__main__":
    main()
