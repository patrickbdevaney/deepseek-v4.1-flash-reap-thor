#!/usr/bin/env python3
"""s04: candidate keep-sets from the s03 bank, then arbitration by held-out teacher KL with proper
re-routing over the survivors, evaluated in memory (layer-streamed replay, no checkpoint written).

  select     CPU. Guards (drift_guard protect lists + every Super-Expert candidate), per-pool
             protection with the image pool ranked on image-span tokens (bias_vl routing), the
             protect_frac sweep and its knee for each arm size, the candidate masks, and
             `drift_guard.py check` on every candidate (BLOCK -> ineligible).
  arbitrate  GPU. lib/replay over the SCREEN half of the held-out rows: a control arm (all 384
             experts: must reproduce the cached teacher, gate `control` in s10_gates.json) and every
             candidate, arm-batched to fit the disk. One process per (layer, batch).
  decide     CPU. Per-arm kl32 / top-1 / dNLL with row-cluster CIs, paired comparisons, the winner
             per arm size -> keepset.json (192, arm A: feeds s05-s07) and keepset_128.json (arm B).
  run        all three, resumable.

Candidates (pre-registered arms, README "s04"):
  arm A (192): hope_sum@knee, hope_cond@knee, hope_sum@pf0 (does pool protection earn its place?),
               reap_sum, mass, freq, random x3 (anchors, never eligible)
  arm B (128): hope_sum@knee128, reap_sum
  round 2 (owner decision 2026-10-22, after round 1 lost multilingual on screen: winner bucket top1
  0.52 < the 0.65 release floor, random0 0.57; the winner kept 54% of the top-96 multilingual
  experts by gate mass, random keeps 50%): hope_sum@pf0 + the top-k multilingual experts per layer
  by gate mass protected, k in ML_LADDER. Ranked by gate mass, not REAP sum: the sum pools are
  dominated by guarded super-experts (one expert held 96% of a pool's mass at L20).
Winner rule (round 2, declared before any round-2 arm was scored): among eligible candidates, the
lowest screen kl32 of those passing gates.pruned_floor on screen; arm A with none passing stops.
Round 2 outcome: ml144 alone reached the multilingual floor (0.657) but lost general/finance/science
and did not beat random on kl32; ml48/ml96 stayed under the floor. The owner then chose (2026-10-22)
to ship the all-round keep-set: s10_gates.json pruned_floor.bucket_overrides gives multilingual a
0.40 collapse guard (every other floor unchanged; the cards state the limitation), and the same
winner rule re-run picks hope_sum_pf0.
Every candidate, random included, keeps the guard set. The shared expert, ViT and aligner are not
routed experts and are never in a keep-set. Allocation stays uniform per layer (one scalar
n_routed_experts for vLLM, GGUF and the CUDA server).

Outputs ($V41_WORK/s04): candidates.json, masks/<name>.json (+ .drift.json), sweep_<n>.json,
replay/ (run dir), arbitration.json, keepset.json, keepset_128.json.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ops import corpus_spec as spec  # noqa: E402
from ops.lib import drift, env, gates, keepset, metrics, orchestrate, rowsets  # noqa: E402
from ops.lib import select as SEL  # noqa: E402

STAGE = "s04"
OUT = env.WORK / "s04"
S03 = env.WORK / "s03"
CORPUS = env.WORK / "corpus"
CFG = json.load(open(env.INFERENCE / "config.json"))
N_LAYERS, E = CFG["n_layers"], CFG["n_routed_experts"]
ARMS = {"A": 192, "B": 128}
ML_LADDER = (48, 96, 144)


# ----------------------------------------------------------------------------- select
def _layer_inputs(L: int, img_mask: np.ndarray, protect: dict, superc: np.ndarray) -> dict:
    bank = SEL.load_bank(S03, L)
    vb = [str(b) for b in bank["buckets"]].index("vision") if "vision" in [str(b) for b in bank["buckets"]] else None
    idx = np.load(S03 / "routing" / f"L{L}_idx.npy", mmap_mode="r")
    w = np.load(S03 / "routing" / f"L{L}_w.npy", mmap_mode="r")
    img = SEL.image_saliency(idx, w, img_mask, SEL.mean_norm(bank, vb)) if img_mask.any() else None
    names, P = SEL.pools(bank, img)
    live = P.sum(1) > 0
    guard = np.union1d(np.asarray(protect.get(str(L), []), int), np.flatnonzero(superc[L]))
    ml = [str(b) for b in bank["buckets"]].index("multilingual")
    return {"L": L, "pools": names, "P": P, "live": live, "guard": guard.astype(int), "gat_ml": bank["gat"][ml],
            "F_sum": SEL.hope_F(bank, "sum"), "F_cond": SEL.hope_F(bank, "cond"), "crit": SEL.criteria(bank)}


def _solve(job):
    d, mode, n_keep, pf, form, seed, ml_k = job
    prot = d["guard"] if pf is None else np.union1d(d["guard"], SEL.pool_protect(d["P"], pf, d["live"]))
    if ml_k:
        prot = np.union1d(prot, np.argsort(-d["gat_ml"], kind="stable")[:ml_k])
    F = d["F_sum"] if form == "sum" else d["F_cond"]
    pr = SEL.prune_set(mode, n_keep=n_keep, protect=prot, crit=d["crit"], F=F if mode == "hope" else None,
                       seed=seed, E=E)
    return d["L"], pr, SEL.retention(d["P"], pr), SEL.icost(d["F_sum"], pr)


def _mask(pool, layers, mode, n_keep, pf=None, form="sum", seed=0, ml_k=0):
    out = pool.map(_solve, [(d, mode, n_keep, pf, form, seed + d["L"], ml_k) for d in layers])
    keep = {str(L): SEL.keep_from_pruned(pr, E) for L, pr, _, _ in out}
    R = np.stack([r for _, _, r, _ in out]).mean(0)
    return keep, R, float(np.mean([c for _, _, _, c in out]))


def _sweep(pool, layers, n_keep):
    rows = []
    for pf in SEL.PF_GRID:
        try:
            _, R, ic = _mask(pool, layers, "hope", n_keep, pf=pf)
        except SEL.Infeasible as e:
            rows.append({"pf": pf, "infeasible": str(e)})
            break
        live = layers[0]["live"]
        wv = np.where(live, R, np.inf)
        rows.append({"pf": pf, "worst": float(wv.min()), "worst_pool": layers[0]["pools"][int(np.argmin(wv))],
                     "mean": float(R[live].mean()), "icost": ic, "retention": dict(zip(layers[0]["pools"], R.tolist()))})
        env.log(f"n_keep {n_keep} pf {pf}: worst {rows[-1]['worst']:.4f} ({rows[-1]['worst_pool']}) icost {ic:.5f}", STAGE)
    ok = [r for r in rows if "infeasible" not in r]
    knee, steps = (ok[0]["pf"] if ok else 0.0), []
    for a, b in zip(ok, ok[1:]):
        dw, di = b["worst"] - a["worst"], b["icost"] - a["icost"]
        ratio = dw / di if di > 0 else (float("inf") if dw > 0 else 0.0)
        steps.append({"from": a["pf"], "to": b["pf"], "d_worst": dw, "d_icost": di, "ratio": ratio})
        if ratio >= 1.0:
            knee = b["pf"]
    return {"n_keep": n_keep, "rows": rows, "steps": steps, "knee": knee,
            "rule": "knee = last grid step with marginal d_worst / d_icost >= 1 (GLM sweep_protect_frac)"}


def select(a):
    led = env.Ledger(env.LOGS / "s04_ledger.json")
    if led.done("select"):
        return
    s03led = env.Ledger(env.LOGS / "s03_ledger.json")
    if not s03led.done("head"):
        raise SystemExit("s03 is not complete (ledger has no finished `head`)")
    prot_f = OUT / "position_protect.json"
    if not prot_f.exists():
        raise SystemExit(f"{prot_f} missing: run `drift_guard.py protect` first (chain does, after s03)")
    protect = json.load(open(prot_f))["protect"]
    fmax = np.stack([SEL.load_bank(S03, L)["fmax"] for L in range(N_LAYERS)])
    superc = drift.super_expert_candidates(fmax)
    rows, man = rowsets.load_rows(CORPUS)
    types = np.load(CORPUS / "types.npy", mmap_mode="r")
    valid = np.load(CORPUS / "valid.npy", mmap_mode="r")
    img = np.zeros(types.shape[0], bool)
    for r in rows:
        if not r["heldout"] and r["images"]:
            s = slice(r["offset"], r["offset"] + r["length"])
            img[s] = (np.asarray(types[s]) >= 0) & np.asarray(valid[s]).astype(bool)
    env.log(f"image-span calibration tokens: {int(img.sum())}", STAGE)
    layers = [_layer_inputs(L, img, protect, superc) for L in range(N_LAYERS)]
    guard_n = [len(d["guard"]) for d in layers]
    env.log(f"guards per layer: max {max(guard_n)}, total {sum(guard_n)} "
            f"(super-expert candidates {int(superc.sum())})", STAGE)
    (OUT / "masks").mkdir(parents=True, exist_ok=True)
    cands = []
    with mp.Pool(a.procs) as pool:
        knees = {}
        for arm, n_keep in ARMS.items():
            sw = _sweep(pool, layers, n_keep)
            env.atomic_json(OUT / f"sweep_{n_keep}.json", sw)
            knees[arm] = sw["knee"]
        plan = [("A", "hope_sum_knee", "hope", knees["A"], "sum", 0, True),
                ("A", "hope_cond_knee", "hope", knees["A"], "cond", 0, True),
                ("A", "hope_sum_pf0", "hope", 0.0, "sum", 0, True),
                ("A", "reap_sum", "reap_sum", None, "sum", 0, True),
                ("A", "mass", "mass", None, "sum", 0, True),
                ("A", "freq", "freq", None, "sum", 0, True),
                ("A", "random0", "random", None, "sum", 1000, False),
                ("A", "random1", "random", None, "sum", 2000, False),
                ("A", "random2", "random", None, "sum", 3000, False),
                ("B", "b128_hope_sum_knee", "hope", knees["B"], "sum", 0, True),
                ("B", "b128_reap_sum", "reap_sum", None, "sum", 0, True)]
        for arm, name, mode, pf, form, seed, eligible in plan:
            n_keep = ARMS[arm]
            keep, R, ic = _mask(pool, layers, mode, n_keep, pf=pf, form=form, seed=seed)
            for L in range(N_LAYERS):
                assert set(layers[L]["guard"].tolist()) <= set(keep[str(L)]), (name, L)
            fp = keepset.save(OUT / "masks" / f"{name}.json", keep, arm=arm, criterion=mode, pf=pf, form=form,
                              seed=seed, eligible=eligible, retention=dict(zip(layers[0]["pools"], R.tolist())), icost=ic)
            cands.append({"name": name, "arm": arm, "n_keep": n_keep, "criterion": mode, "pf": pf, "form": form,
                          "eligible": eligible, "sha256": fp, "icost": ic,
                          "worst_retention": float(R[layers[0]["live"]].min())})
            env.log(f"{name}: sha {fp[:16]} worst retention {cands[-1]['worst_retention']:.4f} icost {ic:.5f}", STAGE)
    for c in cands:
        _drift_check(c)
    env.atomic_json(OUT / "candidates.json", {"knees": knees, "candidates": cands,
                                              "guards": {"protect_file": str(prot_f), "super_expert_candidates":
                                                         {str(L): np.flatnonzero(superc[L]).tolist() for L in range(N_LAYERS)}}})
    led.finish("select", n=len(cands))


def _drift_check(c: dict) -> None:
    m = OUT / "masks" / f"{c['name']}.json"
    flat = m.with_suffix(".keep.json")
    env.atomic_json(flat, keepset.load(m)[0])
    rc = subprocess.run([sys.executable, str(env.OPS / "drift_guard.py"), "check", "--keep", str(flat),
                         "--rows", "calib", "--out", str(m.with_suffix(".drift.json"))]).returncode
    if rc not in (0, 2, 3):
        raise SystemExit(f"drift check failed for {c['name']} rc={rc}")
    c["drift_verdict"] = json.load(open(m.with_suffix(".drift.json")))["verdict"]["verdict"]
    g, _ = gates.load()
    if c["drift_verdict"] in g["arbitration"]["ineligible_drift_verdicts"]:
        c["eligible"] = False
        c["ineligible_why"] = f"drift {c['drift_verdict']}"


def select_ml(a):
    """Round 2: hope_sum@pf0 with the top-k multilingual experts (gate mass) protected, k in ML_LADDER."""
    led = env.Ledger(env.LOGS / "s04_ledger.json")
    if led.done("select_ml"):
        return
    if not led.done("select"):
        raise SystemExit("select_ml needs round 1 (`select`) first")
    cj = json.load(open(OUT / "candidates.json"))
    g = cj["guards"]
    protect = json.load(open(g["protect_file"]))["protect"]
    superc = np.zeros((N_LAYERS, E), bool)
    for L, ids in g["super_expert_candidates"].items():
        superc[int(L), ids] = True
    layers = [_layer_inputs(L, np.zeros(1, bool), protect, superc) for L in range(N_LAYERS)]
    have = {c["name"] for c in cj["candidates"]}
    with mp.Pool(a.procs) as pool:
        for k in ML_LADDER:
            name = f"hope_sum_ml{k}"
            if name in have:
                continue
            keep, R, ic = _mask(pool, layers, "hope", ARMS["A"], pf=None, form="sum", ml_k=k)
            for L in range(N_LAYERS):
                assert set(layers[L]["guard"].tolist()) <= set(keep[str(L)]), (name, L)
            fp = keepset.save(OUT / "masks" / f"{name}.json", keep, arm="A", criterion="hope", pf=None, form="sum",
                              seed=0, eligible=True, ml_k=k, retention=dict(zip(layers[0]["pools"], R.tolist())),
                              icost=ic)
            c = {"name": name, "arm": "A", "n_keep": ARMS["A"], "criterion": "hope", "pf": None, "form": "sum",
                 "ml_k": k, "round": 2, "eligible": True, "sha256": fp, "icost": ic,
                 "worst_retention": float(R[layers[0]["live"]].min())}
            _drift_check(c)
            cj["candidates"].append(c)
            env.atomic_json(OUT / "candidates.json", cj)
            env.log(f"{name}: sha {fp[:16]} ml retention {R[layers[0]['pools'].index('multilingual')]:.4f} "
                    f"icost {ic:.5f} drift {c['drift_verdict']}", STAGE)
    led.finish("select_ml", n=len(ML_LADDER))


# ----------------------------------------------------------------------------- arbitrate
def _run_dir():
    return OUT / "replay"


def _prepare_run(a):
    from ops.lib import replay
    rows, man = rowsets.load_rows(CORPUS)
    screen = rowsets.split_rows(rows, man["spec_hash"], "screen")
    ids = rowsets.subset(screen, a.tokens, spec.TOKEN_TARGET)
    cands = json.load(open(OUT / "candidates.json"))["candidates"]
    arms = [{"name": "control", "keep": None}]
    for c in cands:
        keep, fp = keepset.load(OUT / "masks" / f"{c['name']}.json", expect=c["sha256"])
        arms.append({"name": c["name"], "keep": keep})
    run = replay.prepare(_run_dir(), ids, arms, man["spec_hash"])
    env.log(f"screen subset: {len(ids)} rows, {run.T} tokens, rows sha {rowsets.rows_hash(ids, man['spec_hash'])[:16]}", STAGE)
    return run


def arbitrate(a):
    from ops.lib import park, replay
    # s03 parked the decoder-layer shards on the Seagate while its stream filled the NVMe; bring them
    # back (dropping the dead stream if needed) before any replay reads the source
    if (k := park.restore()):
        env.log(f"restored {k} parked source shards before arbitration", STAGE)
    run = _prepare_run(a)
    names = [x["name"] for x in run.arms]
    todo = [n for n in names if not run.led.done(f"{n}/head")]
    per = replay.bytes_per_arm(run.T)
    if a.drop_s03_stream and orchestrate.disk_free(OUT) < per * min(len(todo), 3) + (60 << 30):
        if env.Ledger(env.LOGS / "s03_ledger.json").done("head"):
            for p in ("h.npy", "pre_mix.npy"):
                if (S03 / p).exists():
                    env.log(f"dropping dead s03 stream {p} (teacher cache complete; nothing downstream reads it)", STAGE)
                    (S03 / p).unlink()
    free = orchestrate.disk_free(OUT)
    bs = orchestrate.batches(todo, per, free, 60 << 30, cap=a.max_batch)
    env.log(f"{len(todo)} arms to score, {per / 2**30:.1f} GiB stream each, batches {[len(b) for b in bs]}", STAGE)
    me = Path(__file__)
    if not run.led.done("_embed"):
        orchestrate.run_unit(me, ["unit", "embed", "--kernels", a.kernels], stage=STAGE, gpu=True, min_avail_gib=a.min_avail_gib)
    for b in bs:
        al = ",".join(b)
        for L in range(N_LAYERS):
            if all(run.led.done(f"{n}/L{L}") for n in b):
                continue
            orchestrate.run_unit(me, ["unit", "layer", "--layer", str(L), "--arms", al, "--kernels", a.kernels,
                                      "--token-chunk", str(a.token_chunk)], stage=STAGE, gpu=True, min_avail_gib=a.min_avail_gib)
        orchestrate.run_unit(me, ["unit", "head", "--arms", al], stage=STAGE, gpu=True, min_avail_gib=a.min_avail_gib)
        run = replay.Run(_run_dir())
        for n in b:
            replay.drop_streams(run, n)


def unit(a):
    from ops.lib import replay
    run = replay.Run(_run_dir())
    arms = a.arms.split(",") if a.arms else None
    if a.unit == "embed":
        replay.unit_embed(run, a.kernels)
    elif a.unit == "layer":
        replay.unit_layer(run, a.layer, a.kernels, a.token_chunk, arms)
    else:
        replay.unit_head(run, arms)


# ----------------------------------------------------------------------------- decide
def decide(a):
    from ops.lib import replay  # noqa: F401  (Run is numpy-only at this point)
    g = gates.load()[0]
    spec_run = json.load(open(_run_dir() / "run.json"))
    rows = spec_run["rows"]
    C, T = metrics.corpus(CORPUS), metrics.teacher(S03)
    cands = {c["name"]: c for c in json.load(open(OUT / "candidates.json"))["candidates"]}
    data = {x["name"]: metrics.load_arm(_run_dir(), x["name"], rows, C, T) for x in spec_run["arms"]}
    summ = {n: metrics.summarize(d, n_draws=a.draws) for n, d in data.items()}
    ctl = gates.control(summ["control"], g)
    res = {"rows": len(rows), "tokens": int(data["control"]["kl32"].size), "control": ctl,
           "summaries": summ, "candidates": cands, "winners": {}}
    if not ctl["pass"]:
        env.atomic_json(OUT / "arbitration.json", res)
        raise SystemExit(f"CONTROL FAILED, replay machinery does not reproduce the teacher: {ctl['why']}")
    for arm, n_keep in ARMS.items():
        el = [n for n, c in cands.items() if c["arm"] == arm and c["eligible"]]
        if not el:
            raise SystemExit(f"arm {arm}: no eligible candidate")
        order = sorted(el, key=lambda n: summ[n]["overall"]["kl32"]["mean"])
        rnd = [n for n, c in cands.items() if c["arm"] == arm and c["criterion"] == "random"]
        best_r = min(rnd, key=lambda n: summ[n]["overall"]["kl32"]["mean"]) if rnd else None
        floor = {n: gates.pruned_floor(summ[n], g, drift_verdict=cands[n].get("drift_verdict"),
                                       vs_random=metrics.compare(data[n], data[best_r], n_draws=a.draws)
                                       if best_r else None) for n in order}
        passing = [n for n in order if floor[n]["pass"]]
        res.setdefault("screen_floor", {}).update(floor)
        if passing:
            order = passing + [n for n in order if n not in passing]
        elif arm == "A":
            env.atomic_json(OUT / "arbitration.json", res)
            raise SystemExit("arm A: no eligible candidate passes the release floor gate on screen: " +
                             "; ".join(f"{n}: {floor[n]['why']}" for n in order))
        win = order[0]
        w = {"name": win, "rule": "lowest screen kl32 among eligible candidates passing pruned_floor on screen"
                               + ("" if passing else " (none passed: lowest kl32, arm B only)"),
             "sha256": cands[win]["sha256"], "kl32": summ[win]["overall"]["kl32"],
             "top1": summ[win]["overall"]["top1"], "ranking": [(n, summ[n]["overall"]["kl32"]["mean"]) for n in order]}
        if len(order) > 1:
            w["vs_runner_up"] = metrics.compare(data[win], data[order[1]], n_draws=a.draws)
        if rnd:
            w["vs_best_random"] = {"arm": best_r, **metrics.compare(data[win], data[best_r], n_draws=a.draws)}
        if "hope_sum_pf0" in cands and arm == "A" and win != "hope_sum_pf0":
            w["vs_pf0"] = metrics.compare(data[win], data["hope_sum_pf0"], n_draws=a.draws)
        res["winners"][arm] = w
        keep, fp = keepset.load(OUT / "masks" / f"{win}.json", expect=cands[win]["sha256"])
        dst = OUT / ("keepset.json" if arm == "A" else "keepset_128.json")
        keepset.save(dst, keep, arm=arm, n_keep=n_keep, winner=win, screen_kl32=w["kl32"],
                     screen_top1=w["top1"], drift_verdict=cands[win].get("drift_verdict"),
                     corpus_spec_hash=json.load(open(CORPUS / "manifest.json"))["spec_hash"])
        env.log(f"arm {arm} ({n_keep}): winner {win} kl32 {w['kl32']['mean']:.4f} top1 {w['top1']['mean']:.4f} "
                f"sha {fp[:16]}", STAGE)
    env.atomic_json(OUT / "arbitration.json", res)
    env.Ledger(env.LOGS / "s04_ledger.json").finish("decide", winner_A=res["winners"]["A"]["sha256"],
                                                    winner_B=res["winners"]["B"]["sha256"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["select", "select_ml", "arbitrate", "decide", "run", "unit"])
    ap.add_argument("unit", nargs="?", choices=["embed", "layer", "head"])
    ap.add_argument("--layer", type=int)
    ap.add_argument("--arms")
    ap.add_argument("--kernels", choices=["tilelang", "torch"], default="tilelang")
    ap.add_argument("--token-chunk", type=int, default=8192)
    ap.add_argument("--tokens", type=int, default=196_608, help="screen-split token budget for arbitration")
    ap.add_argument("--procs", type=int, default=8, help="CPU workers for the HOPE solves")
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--max-batch", type=int, default=13)
    ap.add_argument("--min-avail-gib", type=float, default=100.0)
    ap.add_argument("--drop-s03-stream", action="store_true",
                    help="delete s03/h.npy + pre_mix.npy (dead after the s03 head unit) when disk is short")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.cmd == "unit":
        return unit(a)
    if a.cmd in ("select", "run"):
        select(a)
    if a.cmd in ("select_ml", "run"):
        select_ml(a)
    if a.cmd in ("arbitrate", "run"):
        env.assert_no_gpu_server(STAGE)
        arbitrate(a)
    if a.cmd in ("decide", "run"):
        decide(a)


if __name__ == "__main__":
    main()
