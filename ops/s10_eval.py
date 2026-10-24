#!/usr/bin/env python3
"""s10: paired evaluation gates against the cached s03 teacher, thresholds pre-registered in
ops/s10_gates.json (sha256-locked at the first result).

  heal   SCREEN split (tuning). Arms: unhealed (keep-set A) vs healed (A + s06 gains). Paired
         d_kl32 and McNemar -> s10/heal/decision.json {adopt, keepset_sha256}. s07 reads it.
  final  CONFIRM split (release; never used for a tuning decision). Arms:
           control       all 384 experts (replay must reproduce the teacher)
           mxfp4_A       keep-set A unhealed            = the MXFP4 master
           healed_A      keep-set A + adopted gains     = the NVFP4 artifact (lossless repack + gain)
           thorA         Thor build A (requantized experts, gain folded)
           b128          keep-set B bit-exact           = arm B (and the Thor B build)
           thorA_overlay Thor build A + NVFP4 dense overlay (only if s07 overlay exists)
           random0       the s04 random anchor at 192 (must be beaten)
         Gates: control, pruned floors, Thor quantization cost, the pre-registered A/B arm decision
         at matched expert bytes, the overlay gate, and a 90% Model Confidence Set over the Thor
         candidates. -> s10/final/results.json and s10/gates.json (what s12 may publish).
  trellis  CONFIRM split, pre-registered sub-arm (s10_gates.json "trellis"), after s07 thor-trellis and
         s08 gemv_bench. Arms: the shipped MXFP4/NVFP4 arm, thorA (K-quant), thorA_kt (KT2/KT3 at <= the
         same expert bytes), b128. KT is adopted only if it beats KQ on paired kl32 AND top-1 and its
         modelled expert decode time is within the bound; then thor_quant and the A/B decision are re-run
         with it as arm A and s10/gates.json gains thor_a_build / thor_build_dir. -> s10/trellis/results.json

Per arm and gate: overall / per-bucket / per-log-position-bin kl32, top-1, dNLL with row-cluster
bootstrap CIs, p99 kl32, confident-flip rate, vision caption perplexity. These files are the only
source of numbers for the model cards.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ops import corpus_spec as spec  # noqa: E402
from ops.lib import env, gates, keepset, metrics, orchestrate, rowsets, stats  # noqa: E402

STAGE = "s10"
OUT = env.WORK / "s10"
S03 = env.WORK / "s03"
S04 = env.WORK / "s04"
CORPUS = env.WORK / "corpus"
N_LAYERS = json.load(open(env.INFERENCE / "config.json"))["n_layers"]


def keep_A():
    return keepset.load(S04 / "keepset.json")


def thor_build(n: int, suffix: str, fp: str, dec: dict | None = None) -> Path:
    thor = env.WORK / "thor" / f"A{n}{suffix}"
    tj = json.load(open(thor / "thor.json"))
    if not tj.get("complete"):
        raise SystemExit(f"{thor} is not a complete Thor build")
    keepset.require(tj["keepset_sha256"], fp, where=str(thor))
    if dec is not None:
        # the gain folded into w2 at s07 build time must be the heal decision in force now (GLM stale-gain class)
        want = {str(L): (float(dec["gains"][str(L)]) if dec["adopt"] else 1.0) for L in range(N_LAYERS)}
        have = {str(k): float(v) for k, v in (tj.get("heal") or {}).get("gains", {}).items()}
        if have != want:
            raise SystemExit(f"{thor}: folded heal gains differ from s10/heal/decision.json (adopt={dec['adopt']}); "
                             "rebuild s07 thor")
    return thor


def arms_for(panel: str) -> list[dict]:
    kA, fA = keep_A()
    if panel == "heal":
        h = json.load(open(env.WORK / "s06" / "heal_gain.json"))
        keepset.require(h["keepset_sha256"], fA, where="s06/heal_gain.json")
        return [{"name": "unhealed", "keep": kA}, {"name": "healed", "keep": kA, "gain": h["gains"]}]
    dec = json.load(open(OUT / "heal" / "decision.json"))
    keepset.require(dec["keepset_sha256"], fA, where="s10/heal/decision.json")
    kB, fB = keepset.load(S04 / "keepset_128.json")
    n = len(kA["0"])
    thor = thor_build(n, "", fA, dec)
    if panel == "trellis":
        base = {"name": "healed_A", "keep": kA, "gain": dec["gains"]} if dec["adopt"] else {"name": "mxfp4_A", "keep": kA}
        return [base, {"name": "thorA", "keep": kA, "thor": str(thor)},
                {"name": "thorA_kt", "keep": kA, "thor": str(thor_build(n, "kt", fA, dec))}, {"name": "b128", "keep": kB}]
    arms = [{"name": "control", "keep": None}, {"name": "mxfp4_A", "keep": kA}]
    if dec["adopt"]:
        arms.append({"name": "healed_A", "keep": kA, "gain": dec["gains"]})
    arms += [{"name": "thorA", "keep": kA, "thor": str(thor)}, {"name": "b128", "keep": kB}]
    ov = env.WORK / "thor" / "overlay_nvfp4" / "overlay.json"
    if ov.exists():
        arms.append({"name": "thorA_overlay", "keep": kA, "thor": str(thor), "overlay": str(ov.parent)})
    rnd = S04 / "masks" / "random0.json"
    if not rnd.exists():
        raise SystemExit(f"{rnd} missing: the pre-registered must_beat_random_kl32 floor needs the random anchor")
    arms.append({"name": "random0", "keep": keepset.load(rnd)[0]})
    return arms


def run_panel(a, panel: str):
    from ops.lib import replay
    g = gates.lock(OUT)
    rows, man = rowsets.load_rows(CORPUS)
    split = g["heal"]["split"] if panel == "heal" else g["release"]["split"]
    pool = rowsets.split_rows(rows, man["spec_hash"], split)
    ids = rowsets.subset(pool, a.tokens or (196_608 if panel == "heal" else 393_216), spec.TOKEN_TARGET)
    root = OUT / panel / "replay"
    run = replay.prepare(root, ids, arms_for(panel), man["spec_hash"])
    env.log(f"{panel}: {split} split, {len(ids)} rows, {run.T} tokens, arms {[x['name'] for x in run.arms]}", STAGE)
    names = [x["name"] for x in run.arms if not run.led.done(f"{x['name']}/head")]
    per = replay.bytes_per_arm(run.T)
    me = Path(__file__)
    common = ["--panel", panel, "--kernels", a.kernels, "--token-chunk", str(a.token_chunk)]
    if not run.led.done("_embed"):
        orchestrate.run_unit(me, ["unit", "embed", *common], stage=STAGE, gpu=True, min_avail_gib=a.min_avail_gib)
    for b in orchestrate.batches(names, per, orchestrate.disk_free(root), 60 << 30):
        al = ",".join(b)
        for L in range(N_LAYERS):
            if all(replay.Run(root).led.done(f"{n}/L{L}") for n in b):
                continue
            orchestrate.run_unit(me, ["unit", "layer", "--layer", str(L), "--arms", al, *common], stage=STAGE, gpu=True,
                                 min_avail_gib=a.min_avail_gib)
        orchestrate.run_unit(me, ["unit", "head", "--arms", al, *common], stage=STAGE, gpu=True, min_avail_gib=a.min_avail_gib)
        r = replay.Run(root)
        for n in b:
            replay.drop_streams(r, n)
    return root, ids


def analyse(root: Path, ids: list[int], draws: int) -> tuple[dict, dict]:
    spec_run = json.load(open(root / "run.json"))
    C, T = metrics.corpus(CORPUS), metrics.teacher(S03)
    data = {x["name"]: metrics.load_arm(root, x["name"], ids, C, T) for x in spec_run["arms"]}
    summ = {n: metrics.summarize(d, n_draws=draws) for n, d in data.items()}
    return data, summ


def heal_panel(a):
    root, ids = run_panel(a, "heal")
    g = gates.lock(OUT)
    data, summ = analyse(root, ids, a.draws)
    cmp = metrics.compare(data["healed"], data["unhealed"], n_draws=a.draws)
    dec = gates.heal(cmp, g)
    kA, fA = keep_A()
    h = json.load(open(env.WORK / "s06" / "heal_gain.json"))
    out = {"keepset_sha256": fA, "adopt": dec["adopt"], "why": dec["why"], "gains": h["gains"] if dec["adopt"] else None,
           "comparison": cmp, "summaries": summ, "split": "screen", "tokens": int(data["healed"]["kl32"].size),
           "gates_sha256": gates.load()[1], "ts": env.now()}
    env.atomic_json(OUT / "heal" / "decision.json", out)
    env.log(f"heal: adopt={dec['adopt']} d_kl32 {cmp['d_kl32']['mean']:+.5f} [{cmp['d_kl32']['lo']:+.5f}, "
            f"{cmp['d_kl32']['hi']:+.5f}] top1 z {cmp['top1']['z']:+.2f}", STAGE)


def final_panel(a):
    root, ids = run_panel(a, "final")
    g = gates.lock(OUT)
    data, summ = analyse(root, ids, a.draws)
    res = {"split": "confirm", "rows": len(ids), "tokens": int(data["control"]["kl32"].size),
           "gates_sha256": gates.load()[1], "summaries": summ, "comparisons": {}, "gates": {}}
    cmpd = lambda x, y: metrics.compare(data[x], data[y], n_draws=a.draws)
    # beat-random comparisons honour the 2026-10-24 amendment (buckets excluded); the as-registered all-bucket
    # comparison is still computed and recorded beside it
    excl = g["release"]["pruned_floor"].get("must_beat_random_exclude_buckets", [])
    rnd = lambda x: metrics.compare(*(metrics.without_buckets(data[n], excl) for n in (x, "random0")), n_draws=a.draws)
    res["gates"]["control"] = gates.control(summ["control"], g)
    if not res["gates"]["control"]["pass"]:
        env.atomic_json(OUT / "final" / "results.json", res)
        raise SystemExit(f"control failed on the confirm split: {res['gates']['control']['why']}")
    arb = json.load(open(S04 / "arbitration.json"))
    winner = arb["winners"]["A"]["name"]
    dv = arb["candidates"][winner].get("drift_verdict")
    vs_r = rnd("mxfp4_A") if "random0" in data else None
    res["comparisons"]["mxfp4_A_vs_random0" + ("_excl_" + "_".join(excl) if excl else "")] = vs_r
    if excl and vs_r is not None:
        res["comparisons"]["mxfp4_A_vs_random0"] = cmpd("mxfp4_A", "random0")   # as registered, all buckets
    res["gates"]["mxfp4_master"] = gates.pruned_floor(summ["mxfp4_A"], g, vs_random=vs_r, drift_verdict=dv)
    healed = "healed_A" if "healed_A" in data else "mxfp4_A"
    res["gates"]["nvfp4"] = {**gates.pruned_floor(summ[healed], g, vs_random=rnd(healed) if vs_r else None,
                                                   drift_verdict=dv), "arm": healed}
    res["comparisons"]["thorA_vs_" + healed] = cmpd("thorA", healed)
    res["gates"]["thorA"] = gates.thor_quant(summ["thorA"], res["comparisons"]["thorA_vs_" + healed], g,
                                             vs_random=rnd("thorA") if vs_r else None, drift_verdict=dv)
    b_dv = arb["candidates"][arb["winners"]["B"]["name"]].get("drift_verdict")
    res["gates"]["b128"] = gates.pruned_floor(summ["b128"], g, drift_verdict=b_dv)
    res["comparisons"]["thorA_vs_b128"] = cmpd("thorA", "b128")
    dec = gates.arm_decision(res["comparisons"]["thorA_vs_b128"], g)
    res["gates"]["arm_decision"] = dec
    if "thorA_overlay" in data:
        res["comparisons"]["overlay_vs_thorA"] = cmpd("thorA_overlay", "thorA")
        res["gates"]["overlay"] = gates.overlay(res["comparisons"]["overlay_vs_thorA"], summ["thorA"], summ["thorA_overlay"], g)
    finalists = [n for n in ("thorA", "b128", "thorA_overlay") if n in data]
    res["mcs"] = stats.model_confidence_set({n: data[n]["kl32"] for n in finalists}, data["thorA"]["row"],
                                            alpha=0.10, n_draws=min(a.draws, 1000))
    thor_arm = dec["winner"]
    thor_pass = res["gates"]["thorA" if thor_arm == "A" else "b128"]["pass"]
    publish = {"mxfp4_master": res["gates"]["mxfp4_master"]["pass"], "nvfp4": res["gates"]["nvfp4"]["pass"],
               "thor": thor_pass, "thor_arm": thor_arm,
               "overlay_adopted": bool(res["gates"].get("overlay", {}).get("adopt", False))}
    res["publish"] = publish
    env.atomic_json(OUT / "final" / "results.json", res)
    kA, fA = keep_A()
    env.atomic_json(OUT / "gates.json", {**publish, "keepset_A_sha256": fA,
                                         "keepset_B_sha256": keepset.load(S04 / "keepset_128.json")[1],
                                         "heal_adopted": json.load(open(OUT / "heal" / "decision.json"))["adopt"],
                                         "gates_sha256": res["gates_sha256"], "ts": env.now()})
    env.log(f"final: {publish}", STAGE)


def trellis_panel(a):
    root, ids = run_panel(a, "trellis")
    g = gates.lock(OUT)
    data, summ = analyse(root, ids, a.draws)
    kA, fA = keep_A()
    n = len(kA["0"])
    kq = json.load(open(thor_build(n, "", fA) / "thor.json"))
    kt = json.load(open(thor_build(n, "kt", fA) / "thor.json"))
    bp = env.WORK / "s08" / "gemv_bench.json"
    t_kq = t_kt = None
    if bp.exists():
        bench = json.load(open(bp))
        t_kq, t_kt = gates.gemv_time(kq["alloc"], bench), gates.gemv_time(kt["alloc"], bench)
    cmpd = lambda x, y: metrics.compare(data[x], data[y], n_draws=a.draws)
    res = {"split": "confirm", "rows": len(ids), "tokens": int(data["thorA"]["kl32"].size), "gates_sha256": gates.load()[1],
           "summaries": summ, "comparisons": {"thorA_kt_vs_thorA": cmpd("thorA_kt", "thorA")}, "gates": {},
           "gemv_time_us": {"kq": t_kq, "kt": t_kt, "bench": str(bp) if bp.exists() else None}}
    dec = gates.trellis(res["comparisons"]["thorA_kt_vs_thorA"], g, kt_bytes=int(kt["expert_bytes"]),
                        kq_bytes=int(kq["expert_bytes"]), t_kt=t_kt, t_kq=t_kq)
    res["gates"]["trellis"] = dec
    gp = OUT / "gates.json"
    gj = json.load(open(gp))
    if gj.get("keepset_A_sha256") != fA:
        raise keepset.KeepsetMismatch("s10/gates.json was written for a different keep-set A")
    upd = {"trellis_adopted": dec["adopt"], "thor_a_build": "kt" if dec["adopt"] else "kq"}
    if dec["adopt"]:
        base = "healed_A" if "healed_A" in data else "mxfp4_A"
        arb = json.load(open(S04 / "arbitration.json"))
        dv = arb["candidates"][arb["winners"]["A"]["name"]].get("drift_verdict")
        res["comparisons"]["thorA_kt_vs_" + base] = cmpd("thorA_kt", base)
        res["gates"]["thorA_kt"] = gates.thor_quant(summ["thorA_kt"], res["comparisons"]["thorA_kt_vs_" + base], g,
                                                    drift_verdict=dv)
        res["comparisons"]["thorA_kt_vs_b128"] = cmpd("thorA_kt", "b128")
        ad = gates.arm_decision(res["comparisons"]["thorA_kt_vs_b128"], g)
        res["gates"]["arm_decision"] = ad
        b_pass = json.load(open(OUT / "final" / "results.json"))["gates"]["b128"]["pass"]
        upd.update({"thor_arm": ad["winner"], "thor": res["gates"]["thorA_kt"]["pass"] if ad["winner"] == "A" else b_pass})
    arm = upd.get("thor_arm", gj["thor_arm"])
    nB = len(keepset.load(S04 / "keepset_128.json")[0]["0"])
    if arm == "B" and not (env.WORK / "thor" / f"B{nB}" / "thor.json").exists():
        raise SystemExit(f"trellis re-decision picked arm B but thor/B{nB} was never built "
                         "(run `s07_quant.py thor-b`, then re-run `s10_eval.py trellis`)")
    upd["thor_build_dir"] = str(env.WORK / "thor" / (f"A{n}{'kt' if upd['thor_a_build'] == 'kt' else ''}" if arm == "A" else f"B{nB}"))
    res["publish_update"] = upd
    env.atomic_json(OUT / "trellis" / "results.json", res)
    env.atomic_json(gp, {**gj, **upd, "ts_trellis": env.now()})
    env.log(f"trellis: adopt={dec['adopt']} {dec['why']} -> {upd}", STAGE)


def unit(a):
    from ops.lib import replay
    run = replay.Run(OUT / a.panel / "replay")
    arms = a.arms.split(",") if a.arms else None
    if a.unit == "embed":
        replay.unit_embed(run, a.kernels)
    elif a.unit == "layer":
        replay.unit_layer(run, a.layer, a.kernels, a.token_chunk, arms)
    else:
        replay.unit_head(run, arms)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["heal", "final", "trellis", "unit"])
    ap.add_argument("unit", nargs="?", choices=["embed", "layer", "head"])
    ap.add_argument("--panel", choices=["heal", "final", "trellis"])
    ap.add_argument("--layer", type=int)
    ap.add_argument("--arms")
    ap.add_argument("--kernels", choices=["tilelang", "torch"], default="tilelang")
    ap.add_argument("--token-chunk", type=int, default=8192)
    ap.add_argument("--tokens", type=int)
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--min-avail-gib", type=float, default=100.0)
    a = ap.parse_args()
    (OUT / "heal").mkdir(parents=True, exist_ok=True)
    (OUT / "final").mkdir(parents=True, exist_ok=True)
    (OUT / "trellis").mkdir(parents=True, exist_ok=True)
    if a.cmd == "unit":
        return unit(a)
    env.assert_no_gpu_server(STAGE)
    {"heal": heal_panel, "final": final_panel, "trellis": trellis_panel}[a.cmd](a)


if __name__ == "__main__":
    main()
