#!/usr/bin/env python3
"""s07: quantization, last and once, always FROM THE MXFP4 SOURCE (lte-ai: requantizing from a
low-bit intermediate nearly doubled KL).

  nvfp4    (a) CPU, streamed like s05: the arm-A experts losslessly repacked MXFP4 -> NVFP4
           (packed E2M1 bytes unchanged, E8M0 -> e4m3 block scales, per-tensor FP32 global scale
           carrying the adopted heal gain on w2), input scales from the s06 post-prune amax,
           dense FP8 untouched. Every tensor is checked lossless (zero underflowing blocks and an
           exact dequant comparison on sampled rows). w1/w3 of an expert share one global scale (fused
           w13 loaders keep one). Kept on the Seagate; --upload streams to a PRIVATE repo instead.
  plan     CPU. Per-layer bit allocation for the Thor build: RTN-imatrix error of Q2_K/Q3_K/Q4_K on
           sampled experts of every layer, weighted by the layer's kept routed mass, greedy upgrades
           under a byte budget equal to arm B's experts (128 x MXFP4 = 2.833 bpw at 192). Protected
           layers (0-1, Engram 1/14, KV sources 2/8/14/20, DSpark taps 37-39) start at Q3_K.
  thor     (b) GPU, one two-phase unit per layer, sequential: the calibration subset is pushed
           through the already-quantized layers, each kept expert's input Hessian (w1|w3) and, after
           those are quantized, its w2-input Hessian are accumulated from cached FFN inputs, and
           every expert is GPTQ-rounded into ggml K-quant blocks (lib/gptq.py), heal gain folded
           into w2 before rounding. Then the layer runs with its quantized experts and the stream
           advances. Dense tensors pass through bit-exact. Output: the Thor build directory
           (ops/lib/thor_format.md), consumed by the s08 CUDA server (s09 converts from the master).
  thor-b   CPU. Arm B's Thor build: 128 kept experts copied bit-exact as MXFP4_SRC, dense the same.
  thor-trellis  (b') pre-registered sub-arm of A (s10 `trellis` panel decides): the same keep-set,
           heal and two-phase GPU units as `thor`, but experts in the KT2/KT3 bitshift-trellis format
           (ops/lib/kt.py: RHT + block LDLQ + tail-biting Viterbi on the GPU, cuda/build/libkt_viterbi.so)
           under a byte budget equal to the finished K-quant build's expert bytes. Allocation first
           (CPU, sampled rows, alloc_kt.json), then the build in $V41_WORK/thor/A<n>kt. CUDA-server
           only: no ggml type carries it (s09 quantizes from the master, so the GGUF ladder is unaffected).
  overlay  (c) CPU. NVFP4 dense overlay (FP8 Linear weights -> NVFP4 RTN) in its own directory,
           adopted only through the s10 overlay gate.

Inputs: $V41_WORK/s04/keepset.json (+ keepset_128.json), s06/heal_gain.json, s10/heal/decision.json,
s06/bank/L*.npz, s03/bank/L*.npz, the source shards (sha256-pinned). Outputs under $V41_WORK/s07 and
$V41_WORK/thor/<build>.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ops import corpus_spec as spec  # noqa: E402
from ops.lib import env, ggml_quants as G, keepset, orchestrate, rowsets, st_io  # noqa: E402
from ops.lib import kt as KT, thor_format as TF  # noqa: E402

STAGE = "s07"
OUT = env.WORK / "s07"
THOR = env.WORK / "thor"
CORPUS = env.WORK / "corpus"
CFG = json.load(open(env.INFERENCE / "config.json"))
N_LAYERS, E, DIM, INTER = CFG["n_layers"], CFG["n_routed_experts"], CFG["dim"], CFG["moe_inter_dim"]
PROTECTED = sorted({0, 1, *CFG["engram_layer_ids"], *CFG["kv_source_layers"], *CFG["dspark_target_layer_ids"]})
LADDER = ["Q2_K", "Q3_K", "Q4_K"]
KT_LADDER = ["KT2", "KT3"]
EXPERT = re.compile(r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.(weight|scale)$")


def s05():
    import ops.s05_surgery as m
    return m


def heal_decision(fp: str) -> dict:
    """{L: gain} if the s10 heal panel adopted the s06 gains for THIS keep-set, else all 1.0."""
    p = env.WORK / "s10" / "heal" / "decision.json"
    if not p.exists():
        raise SystemExit("s10 heal panel has not run: the heal decision is required before any requant")
    d = json.load(open(p))
    keepset.require(d["keepset_sha256"], fp, where=str(p))
    if not d["adopt"]:
        return {str(L): 1.0 for L in range(N_LAYERS)}
    h = json.load(open(env.WORK / "s06" / "heal_gain.json"))
    keepset.require(h["keepset_sha256"], fp, where="s06/heal_gain.json")
    return {k: float(v) for k, v in h["gains"].items()}


def src_reader(L: int) -> st_io.Reader:
    return st_io.Reader(env.SRC / s05().shard_name(L + 3))


def expert_src(rd: st_io.Reader, L: int, e: int, p: str):
    return rd.get(f"layers.{L}.ffn.experts.{e}.{p}.weight"), rd.get(f"layers.{L}.ffn.experts.{e}.{p}.scale")


# ----------------------------------------------------------------------------- (a) NVFP4
def input_scales(L: int, keep: list[int]) -> dict[str, np.ndarray]:
    """amax / (6 * 448) per kept expert, from the s06 post-prune bank (the model these scales run
    in); an expert the s06 subset never routed to falls back to the teacher's s03 amax (w1|w3) or
    the layer max (ungated w2 input, which s03 did not bank)."""
    b6 = np.load(env.WORK / "s06" / "bank" / f"L{L}.npz")
    b3 = np.load(env.WORK / "s03" / "bank" / f"L{L}.npz")
    if not np.array_equal(b6["keep"], np.asarray(keep)):
        raise keepset.KeepsetMismatch(f"s06 bank L{L} is for a different keep-set")
    seen = b6["n_tok"] > 0
    a_in = np.where(seen, b6["amax_in"], b3["amax_in"][keep])
    a_w2 = np.where(seen, b6["amax_mid_ungated"], b6["amax_mid_ungated"][seen].max() if seen.any() else 1.0)
    return {"w1": a_in / 2688.0, "w3": a_in / 2688.0, "w2": a_w2 / 2688.0}


def w13_emax(rd: st_io.Reader, L: int, e: int) -> int | None:
    """One exponent for w1 AND w3 of an expert: vLLM's ModelOpt NVFP4 MoE fuses them into w13 and keeps
    only w1's weight_scale_2 (modelopt.py, "w1_weight_scale_2 must match w3_weight_scale_2"), so both
    global scales must be equal. The shared value is the larger emax, which keeps the repack exact."""
    em = [G.mxfp4_emax(*expert_src(rd, L, e, q)) for q in ("w1", "w3")]
    em = [x for x in em if x is not None]
    return max(em) if em else None


def nvfp4_layer(rd: st_io.Reader, L: int, keep: list[int], gain: float) -> list:
    out = []
    remap = keepset.remap(keep)
    isc = input_scales(L, keep)
    lossy, e13 = [], {}
    for n in sorted(rd.names(), key=rd.offset):
        m = EXPERT.match(n)
        if m:
            e, p, part = int(m.group(2)), m.group(3), m.group(4)
            if e not in remap or part == "scale":
                continue
            w, s = expert_src(rd, L, e, p)
            if p in ("w1", "w3") and e not in e13:
                e13[e] = w13_emax(rd, L, e)
            r = G.mxfp4_to_nvfp4(w, s, gain if p == "w2" else 1.0, emax=e13[e] if p in ("w1", "w3") else None)
            if r["lossy_blocks"]:
                lossy.append({"expert": e, "proj": p, "lossy_blocks": r["lossy_blocks"]})   # report the whole layer
                continue
            rows = slice(0, 8)
            if not np.array_equal(G.nvfp4_dequant(r["weight"][rows], r["weight_scale"][rows], r["weight_scale_2"]),
                                  G.src_dequant(w[rows], s[rows]) * np.float32(gain if p == "w2" else 1.0)):
                raise SystemExit(f"L{L} e{e} {p}: NVFP4 dequant differs from the MXFP4 source")
            j = remap[e]
            base = f"layers.{L}.ffn.experts.{j}.{p}"
            out += [(base + ".weight", "U8", r["weight"]), (base + ".weight_scale", "F8_E4M3", r["weight_scale"]),
                    (base + ".weight_scale_2", "F32", np.asarray(r["weight_scale_2"], np.float32)),
                    (base + ".input_scale", "F32", np.asarray(isc[p][j], np.float32))]
        elif n in (f"layers.{L}.ffn.gate.weight", f"layers.{L}.ffn.gate.bias", f"layers.{L}.ffn.gate.bias_vl"):
            out.append((n, rd.dtype(n), np.ascontiguousarray(rd.get(n)[keep])))
        else:
            out.append((n, rd.dtype(n), rd.get(n)))
    if lossy:
        rp = OUT / f"nvfp4_lossy_L{L}.json"
        env.atomic_json(rp, {"layer": L, "tensors": lossy, "rule": "nonzero 32-block > 17 octaves below the "
                              "(w1|w3 shared) emax", "ts": env.now()})
        raise SystemExit(f"L{L}: {len(lossy)} expert tensors have blocks not representable in NVFP4 "
                         f"(refusing a lossy repack); full list in {rp}")
    return out


def cmd_nvfp4(a):
    S5 = s05()
    keep, fp = keepset.load(Path(a.keepset))
    n_keep = len(keep["0"])
    gains = heal_decision(fp)
    if all(g == 1.0 for g in gains.values()):
        # Without an adopted heal gain the repack is the published MXFP4 master re-encoded, value for value: a second
        # ~150 GB repository of the same model. Skip it; s12 then publishes the MXFP4 master and the GGUF only.
        env.atomic_json(OUT / f"manifest_nvfp4_REAP{n_keep}.json", {
            "keepset_sha256": fp, "skipped": "heal gain not adopted: NVFP4 would equal the MXFP4 master", "ts": env.now()})
        env.log("nvfp4: heal gain not adopted, so the repack would duplicate the MXFP4 master; skipped", STAGE)
        return
    led = env.Ledger(env.LOGS / f"s07_nvfp4_REAP{n_keep}_ledger.json")
    if led.get("keepset"):
        keepset.require(fp, led.get("keepset")["sha256"], where="s07 nvfp4 ledger")
    led.mark("keepset", sha256=fp)
    from huggingface_hub import HfApi
    # same quota rule as s05: kept on the Seagate, uploaded by s12 to a PUBLIC repo only if the nvfp4 gate passed
    repo = a.repo or f"{HfApi().whoami()['name']}/DeepSeek-V4.1-Flash-1017-REAP{n_keep}-NVFP4"
    up = S5.Uploader(repo, a.upload)
    stage = orchestrate.staging_dir(16 << 30, f"s07_nvfp4_REAP{n_keep}", prefer=env.WORK)
    out_dir = OUT / f"nvfp4_REAP{n_keep}" if up.enabled else S5.local_master_dir(f"nvfp4_REAP{n_keep}")
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = json.load(open(env.SRC / "model.safetensors.index.json"))["weight_map"]
    new_map, quantized = {}, []
    for i in range(1, S5.N_SHARDS + 1):
        name = S5.shard_name(i)
        if i in S5.ENGRAM_SHARDS:
            new_map.update({k: v for k, v in idx.items() if v == name})
            continue
        u = f"shard/{name}"
        if led.done(u):
            new_map.update(led.get(u)["weight_map"])
            quantized += led.get(u).get("quantized", [])
            continue
        src = env.SRC / name
        dst = stage / name
        L = S5.layer_of(i)
        rd = st_io.Reader(src)
        if L is None:
            shutil.copyfile(src, dst)
            wm, q = {n: name for n in rd.names()}, []
        else:
            t = nvfp4_layer(rd, L, keep[str(L)], gains[str(L)])
            st_io.write(dst, t, metadata={"format": "pt"}, align=1)
            wm = {n: name for n, _, _ in t}
            q = sorted({n.rsplit(".", 1)[0] for n, _, _ in t if n.endswith(".weight_scale_2")})
        sha = S5.sha256_file(dst)
        up.put(dst, name)
        size = dst.stat().st_size
        if up.enabled:
            dst.unlink()
        else:
            shutil.move(str(dst), out_dir / name)
            if S5.sha256_file(out_dir / name) != sha:
                (out_dir / name).unlink()
                raise SystemExit(f"{name}: sha256 of the copy on {out_dir} differs from the staged file")
        led.finish(u, sha256=sha, bytes=size, weight_map=wm, quantized=q)
        new_map.update(wm)
        quantized += q
        env.log(f"{name}: {size / 2**30:.2f} GiB", STAGE)
    small = stage / "_small"
    shutil.rmtree(small, ignore_errors=True)
    shutil.copytree(env.SRC, small, ignore=shutil.ignore_patterns("*.safetensors", ".cache", "__pycache__", "*.pyc",
                                                                   "model.safetensors.index.json", "README.md"))
    env.atomic_json(small / "model.safetensors.index.json", {"metadata": {}, "weight_map": new_map})
    cfg = json.load(open(small / "config.json"))
    cfg.setdefault("text_config", {})["n_routed_experts"] = n_keep
    env.atomic_json(small / "config.json", cfg)
    icfg = json.load(open(small / "inference" / "config.json"))
    icfg["n_routed_experts"] = n_keep
    env.atomic_json(small / "inference" / "config.json", icfg)
    env.atomic_json(small / "hf_quant_config.json", {
        "producer": {"name": "deepseek-v4.1-flash-reap-thor/ops/s07_quant.py nvfp4"},
        "note": "routed experts NVFP4 (ModelOpt tensor layout: weight U8 [out,in/2] even element in the low nibble, "
                "weight_scale e4m3 [out,in/16], weight_scale_2 FP32 global (equal for w1 and w3 of an expert), "
                "input_scale FP32); every other linear keeps the source FP8 block-32 ue8m0 format of config.json "
                "quantization_config, and the MTP layers' experts stay MXFP4 (not listed here). NO LOADER IS VERIFIED: "
                "vLLM reads config.json quantization_config first and only falls back to this file when it is absent, "
                "and its ModelOpt MIXED_PRECISION path (quant_method=modelopt) can express per-layer NVFP4 but not the "
                "block-32 ue8m0 FP8 dense layers, so config.json is deliberately left as the source fp8 config "
                "(this artifact does not fit Thor); see the model card.",
        "quantization": {"quant_algo": "MIXED_PRECISION", "kv_cache_quant_algo": None,
                         "quantized_layers": {n: {"quant_algo": "NVFP4", "group_size": 16} for n in quantized}}})
    env.atomic_json(small / "engram_pointer.json", S5.engram_pointer(new_map, S5.tree_sha()))
    keepset.save(small / "reap_keepset.json", keep, heal_gains=gains)
    for p in sorted(x for x in small.rglob("*") if x.is_file()):
        rel = str(p.relative_to(small))
        sha = S5.sha256_file(p)
        if not led.done(f"small/{rel}") or led.get(f"small/{rel}").get("sha256") != sha:   # changed -> re-put
            up.put(p, rel)
            led.finish(f"small/{rel}", sha256=sha)
    if not up.enabled:
        shutil.copytree(small, out_dir, dirs_exist_ok=True)
    shutil.rmtree(small)
    env.atomic_json(OUT / f"manifest_nvfp4_REAP{n_keep}.json", {
        "keepset_sha256": fp, "repo": up.repo, "private": True, "uploaded": up.enabled,
        # no loader has loaded these bytes (config.json still declares the source fp8/ue8m0 config; the s10
        # nvfp4 gate measures the in-memory function): publishing must keep the repo private while False
        "loader_verified": False,
        "local_dir": None if up.enabled else str(out_dir), "heal_gains": gains, "quantized_tensors": len(quantized),
        "files": {k[6:]: {"sha256": v["sha256"], "bytes": v["bytes"]} for k, v in led.data.items() if k.startswith("shard/")},
        "ts": env.now()})
    led.finish("done", repo=up.repo)


# ----------------------------------------------------------------------------- plan
def _importance(L: int, keep: list[int]) -> tuple[np.ndarray, np.ndarray]:
    b6 = np.load(env.WORK / "s06" / "bank" / f"L{L}.npz")
    n = np.maximum(b6["n_tok"], 1)[:, None]
    return b6["x2_in"] / n + 1e-12, b6["x2_mid"] / n + 1e-12


def layer_errors(L: int, keep: list[int], n_sample: int, seed: int) -> dict:
    """imatrix-weighted relative error of each ladder type for w1|w3 and w2 on sampled experts."""
    rd = src_reader(L)
    b3 = np.load(env.WORK / "s03" / "bank" / f"L{L}.npz")
    sal = b3["sum"].sum(0)[keep]
    order = np.argsort(-sal)
    rng = np.random.default_rng(seed + L)
    pick = sorted(set(order[: n_sample // 2].tolist()) | set(rng.choice(len(keep), n_sample - n_sample // 2, replace=False).tolist()))
    a_in, a_mid = _importance(L, keep)
    err = {"w13": {t: [] for t in LADDER}, "w2": {t: [] for t in LADDER}}
    for j in pick:
        e = keep[j]
        for part, projs, A in (("w13", ("w1", "w3"), a_in[j]), ("w2", ("w2",), a_mid[j])):
            for p in projs:
                W = G.src_dequant(*expert_src(rd, L, e, p))
                den = float((W * W * A).sum())
                for t in LADDER:
                    Wq = G.dequant(t, G.quantize_rtn(t, W, A))
                    err[part][t].append(float(((W - Wq) ** 2 * A).sum()) / max(den, 1e-30))
    return {part: {t: float(np.mean(v)) for t, v in d.items()} for part, d in err.items()} | \
        {"mass": float(b3["sum"].sum(0)[keep].sum()), "sampled": [int(keep[j]) for j in pick]}


def row_bytes(t: str, n_cols: int) -> int:
    """stored bytes of one expert row, codes plus any per-block scale tensor."""
    if t in KT.QTYPES:
        return KT.row_bytes(t, n_cols) + KT.scale_bytes(n_cols)
    return G.row_bytes(t, n_cols)


def expert_bytes(t: str, part: str, n_keep: int) -> int:
    if part == "w13":
        return 2 * n_keep * INTER * row_bytes(t, DIM) + (DIM // 8 if t in KT.QTYPES else 0)
    return n_keep * DIM * row_bytes(t, INTER) + (INTER // 8 if t in KT.QTYPES else 0)


def allocate(errs: dict, n_keep: int, budget: int, protected: list[int], ladder: list[str] = LADDER) -> dict:
    """Greedy: start at the ladder bottom (protected layers one rung up: Q3_K, or KT3 on the trellis
    ladder), then repeatedly take the upgrade with the largest drop in mass-weighted error per extra
    byte that still fits the budget."""
    LADDER = ladder
    lo, pr = LADDER[0], LADDER[1]
    alloc = {L: {"w13": pr if L in protected else lo, "w2": pr if L in protected else lo} for L in errs}
    cost = lambda L, part, t: errs[L]["mass"] * errs[L][part][t]
    used = sum(expert_bytes(alloc[L][p], p, n_keep) for L in alloc for p in ("w13", "w2"))
    if used > budget:
        raise SystemExit(f"protected minimum alone ({used / 2**30:.2f} GiB) exceeds the budget ({budget / 2**30:.2f} GiB)")
    while True:
        best = None
        for L in alloc:
            for part in ("w13", "w2"):
                k = LADDER.index(alloc[L][part])
                if k + 1 >= len(LADDER):
                    continue
                t = LADDER[k + 1]
                db = expert_bytes(t, part, n_keep) - expert_bytes(alloc[L][part], part, n_keep)
                if used + db > budget:
                    continue
                gain = (cost(L, part, alloc[L][part]) - cost(L, part, t)) / db
                if best is None or gain > best[0]:
                    best = (gain, L, part, t, db)
        if best is None or best[0] <= 0:
            break
        _, L, part, t, db = best
        alloc[L][part] = t
        used += db
    return {"alloc": {str(L): v for L, v in alloc.items()}, "bytes": used, "budget": budget,
            "avg_bpw": used * 8 / (n_keep * 3 * DIM * INTER * len(alloc)), "ladder": list(LADDER)}


def cmd_plan(a):
    keep, fp = keepset.load(Path(a.keepset))
    n_keep = len(keep["0"])
    nb = len(keepset.load(env.WORK / "s04" / "keepset_128.json")[0]["0"]) if (env.WORK / "s04" / "keepset_128.json").exists() else 128
    budget = a.budget_bytes or int(nb * 3 * DIM * INTER * G.bpw("MXFP4") / 8 * N_LAYERS)
    led = env.Ledger(env.LOGS / "s07_plan_ledger.json")
    errs = {}
    for L in range(N_LAYERS):
        u = f"L{L}"
        if not led.done(u):
            led.finish(u, keepset=fp, **layer_errors(L, keep[str(L)], a.sample, 0))
        r = led.get(u)
        keepset.require(r["keepset"], fp, where="s07 plan ledger")
        errs[L] = r
        env.log(f"L{L}: Q2_K w13 {r['w13']['Q2_K']:.4f} w2 {r['w2']['Q2_K']:.4f}", STAGE)
    res = allocate(errs, n_keep, budget, PROTECTED)
    res.update({"keepset_sha256": fp, "protected": PROTECTED, "ladder": LADDER, "errors": {str(k): v for k, v in errs.items()},
                "budget_rule": f"arm B experts: {nb} x MXFP4 (4.25 bpw) = matched resident expert bytes"})
    env.atomic_json(OUT / "alloc.json", res)
    env.log(f"alloc: {res['avg_bpw']:.3f} bpw, {res['bytes'] / 2**30:.2f} / {budget / 2**30:.2f} GiB", STAGE)


# ----------------------------------------------------------------------------- plan (trellis ladder)
def kt_signs(L: int) -> dict[str, np.ndarray]:
    """The layer's two RHT sign vectors: w13 (input DIM, shared by w1 and w3) and w2 (input INTER)."""
    return {"w13": KT.signs(f"L{L}.w13", DIM), "w2": KT.signs(f"L{L}.w2", INTER)}


def layer_errors_kt(L: int, keep: list[int], n_sample: int, seed: int, rows: int) -> dict:
    """imatrix-weighted relative error of KT2/KT3 (no-feedback encode, numpy Viterbi) on `rows`
    sampled rows of the same sampled experts `layer_errors` uses. The rotated-domain column weight is
    the 256-block mean of the imatrix: the diagonal of R diag(A) R^T for a blockwise Hadamard."""
    rd = src_reader(L)
    b3 = np.load(env.WORK / "s03" / "bank" / f"L{L}.npz")
    sal = b3["sum"].sum(0)[keep]
    order = np.argsort(-sal)
    rng = np.random.default_rng(seed + L)
    pick = sorted(set(order[: n_sample // 2].tolist()) | set(rng.choice(len(keep), n_sample - n_sample // 2, replace=False).tolist()))
    a_in, a_mid = _importance(L, keep)
    sg = kt_signs(L)
    err = {"w13": {t: [] for t in KT_LADDER}, "w2": {t: [] for t in KT_LADDER}}
    for j in pick:
        e = keep[j]
        for part, projs, A in (("w13", ("w1", "w3"), a_in[j]), ("w2", ("w2",), a_mid[j])):
            wt = np.repeat(A.reshape(-1, KT.RHT).mean(1), KT.RHT).astype(np.float32)
            for p in projs:
                W = G.src_dequant(*expert_src(rd, L, e, p))
                W = W[np.sort(rng.choice(W.shape[0], min(rows, W.shape[0]), replace=False))]
                den = float((W * W * A).sum())
                for t in KT_LADDER:
                    Wq = KT.dequant(t, *KT.quantize_rows(W, t, sg[part], wt), W.shape[1], sg[part])
                    err[part][t].append(float(((W - Wq) ** 2 * A).sum()) / max(den, 1e-30))
    return {part: {t: float(np.mean(v)) for t, v in d.items()} for part, d in err.items()} | \
        {"mass": float(b3["sum"].sum(0)[keep].sum()), "sampled": [int(keep[j]) for j in pick], "rows": rows}


def plan_kt(a, keep: dict, fp: str, n_keep: int) -> dict:
    """alloc_kt.json: KT2/KT3 per (layer, pass) under a budget equal to the K-quant build's expert bytes."""
    p = OUT / "alloc_kt.json"
    if p.exists():
        r = json.load(open(p))
        keepset.require(r["keepset_sha256"], fp, where=str(p))
        return r
    if a.budget_bytes:
        budget, rule = a.budget_bytes, "--budget-bytes"
    else:
        kq = TF.Build(build_dir(n_keep, "A"), keepset_sha256=fp, n_routed_experts=n_keep)
        if not kq.m.get("complete"):
            raise SystemExit("thor-trellis matches the K-quant build's bytes: run `s07 thor` first")
        # recomputed from the tensor list (decoder-layer routed experts only), not the stored field
        budget = kq.expert_bytes()
        bpw = budget * 8 / (n_keep * 3 * DIM * INTER * N_LAYERS)
        if kq.m.get("expert_bytes") not in (None, budget):
            env.log(f"thor/A thor.json expert_bytes {kq.m['expert_bytes']} != recomputed {budget} (MTP experts "
                    f"were counted by an older Build.expert_bytes); using {budget}", STAGE)
        budget, rule = int(budget), f"<= the K-quant Thor build's expert bytes ({bpw:.4f} bpw)"
    led = env.Ledger(env.LOGS / "s07_plan_kt_ledger.json")
    errs = {}
    for L in range(N_LAYERS):
        u = f"L{L}"
        if not led.done(u):
            led.finish(u, keepset=fp, **layer_errors_kt(L, keep[str(L)], a.sample, 0, a.kt_rows))
        r = led.get(u)
        keepset.require(r["keepset"], fp, where="s07 plan-kt ledger")
        errs[L] = r
        env.log(f"L{L}: KT2 w13 {r['w13']['KT2']:.4f} w2 {r['w2']['KT2']:.4f}", STAGE)
    res = allocate(errs, n_keep, budget, PROTECTED, ladder=KT_LADDER)
    res.update({"keepset_sha256": fp, "protected": PROTECTED, "errors": {str(k): v for k, v in errs.items()},
                "budget_rule": rule})
    env.atomic_json(p, res)
    env.log(f"alloc_kt: {res['avg_bpw']:.3f} bpw, {res['bytes'] / 2**30:.2f} / {budget / 2**30:.2f} GiB", STAGE)
    return res


# ----------------------------------------------------------------------------- shared Thor pieces
def dense_items(rd: st_io.Reader, L: int, keep: list[int]) -> list[dict]:
    """Every non-expert tensor of a layer shard, bit-exact, router sliced to the keep-set."""
    out = []
    for n in sorted(rd.names(), key=rd.offset):
        if EXPERT.match(n):
            continue
        a = rd.get(n)
        if n in (f"layers.{L}.ffn.gate.weight", f"layers.{L}.ffn.gate.bias", f"layers.{L}.ffn.gate.bias_vl"):
            a = np.ascontiguousarray(a[keep])
        out.append({"name": n, "dtype": rd.dtype(n), "data": a})
    return out


def engram_items(L: int) -> list[dict]:
    """The Engram layer's small tensors (gate/key projections, wkv) from shards 47/48; the 98 GB
    tables are not part of the build (the row store is built separately, see thor_format.md)."""
    if L not in CFG["engram_layer_ids"]:
        return []
    shard = env.SRC / s05().shard_name(47 + CFG["engram_layer_ids"].index(L))
    if not shard.exists():   # shards 47/48 may be symlinks onto the Seagate
        raise SystemExit(f"Engram shard {shard} is not readable (-> {shard.resolve()}): is the Seagate mounted?")
    rd = st_io.Reader(shard)
    return [{"name": n, "dtype": rd.dtype(n), "data": np.asarray(rd.get(n))} for n in rd.names() if ".engram.embed." not in n]


def write_globals(b: TF.Build) -> None:
    S5 = s05()
    groups = {"globals.safetensors": [2, 43], "vision.safetensors": [1],
              "mtp-0.safetensors": [44], "mtp-1.safetensors": [45], "mtp-2.safetensors": [46]}
    for fname, shards in groups.items():
        if b.has(fname):
            continue
        items = []
        for i in shards:
            rd = st_io.Reader(env.SRC / S5.shard_name(i))
            items += [{"name": n, "dtype": rd.dtype(n), "data": np.asarray(rd.get(n))} for n in sorted(rd.names(), key=rd.offset)]
        b.write_file(fname, items)


def finish_build(b: TF.Build, n_keep: int) -> None:
    S5 = s05()
    shas = S5.tree_sha()
    b.m["engram_tables"] = {"base_repo": S5.BASE_REPO, "revision": env.REVISION,
                            "tables": [{"layer": L, "file": S5.shard_name(47 + i), "tensor": f"layers.{L}.engram.embed.weight",
                                        "scale": f"layers.{L}.engram.embed.scale", **shas[S5.shard_name(47 + i)]}
                                       for i, L in enumerate(CFG["engram_layer_ids"])]}
    missing = [L for L in range(N_LAYERS) if not b.has(TF.layer_file(L))]
    if missing:
        raise SystemExit(f"build incomplete, missing layers {missing}")
    b.m["expert_bytes"] = int(b.expert_bytes())
    b.m["expert_bpw"] = b.m["expert_bytes"] * 8 / (n_keep * 3 * DIM * INTER * N_LAYERS)
    b.m["complete"] = True
    b.save()
    bad = b.verify()
    if bad:
        raise SystemExit(f"sha256 mismatch in {bad}")


# ----------------------------------------------------------------------------- disk: preflight + parking
GIB = 1 << 30
# Everything in a Thor build except the routed experts, from the source headers (audit s07 RISK-2): 40 decoder
# layers' dense tensors (~170.3 MB each, 6.3 GiB) + globals/vision/mtp (~10.7 GiB) + Engram small tensors (0.3 GiB).
NONEXPERT_BYTES = 18 * GIB
# Replay peak per calibration token: `_embed/h.npy` + the arm stream + the per-layer cache (x_mid + ffn_in), each
# [T, 4, DIM] bf16 -> 1 + 1 + 1.25 streams (10 + 10 + 12.5 GiB at T = 262,144).
REPLAY_STREAMS = 3.25
THOR_MARGIN = 20 * GIB          # thor / thor-trellis: on top of the remaining build + replay
THOR_B_MARGIN = 40 * GIB        # thor-b: free minus 40 GiB must hold the rest of the build
PARK_MARGIN = 20 * GIB
# chain2.sh park_build/unpark_build convention: $SEAW/thor/<name>, NVMe thor/<name> replaced by a symlink to it
PARK_ROOT = orchestrate.SEAGATE / "v41_staging" / "thor"


def _du(p: Path) -> int:
    """bytes of the regular files under p, not following symlinks (0 if p is missing or a symlink)."""
    p = Path(p)
    if not p.exists() or p.is_symlink():
        return 0
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file() and not f.is_symlink())


def kq_build_estimate(alloc_map: dict, n_keep: int) -> int:
    """final size of a K-quant / KT Thor build from its allocation (expert bytes are exact)."""
    return sum(expert_bytes(v["w13"], "w13", n_keep) + expert_bytes(v["w2"], "w2", n_keep)
               for v in alloc_map.values()) + NONEXPERT_BYTES


def mxfp4_build_estimate(n_keep: int) -> int:
    """arm B: n_keep experts bit-exact MXFP4 (in/2 code bytes + in/32 E8M0 bytes = 17/32 B per weight)."""
    return n_keep * 3 * DIM * INTER * 17 // 32 * N_LAYERS + NONEXPERT_BYTES


def replay_peak(tokens: int) -> int:
    return int(tokens * 4 * DIM * 2 * REPLAY_STREAMS)


def _nvme_builds() -> str:
    if not THOR.exists():
        return "none"
    xs = [f"thor/{d.name} {_du(d) / GIB:.0f} GiB" for d in sorted(THOR.iterdir()) if d.is_dir() and not d.is_symlink()]
    return ", ".join(xs) or "none"


def disk_preflight(what: str, need: int, margin: int, path: Path | None = None) -> None:
    """Fail fast, before any GPU-hour, if `path` (env.WORK) cannot take `need` more bytes plus `margin`."""
    path = Path(path or env.WORK)
    free = orchestrate.disk_free(path)
    if free - need < margin:
        raise SystemExit(
            f"s07 {what}: disk preflight failed on {path}: {free / GIB:.0f} GiB free, needs {need / GIB:.0f} GiB more "
            f"+ {margin / GIB:.0f} GiB margin. Thor builds on NVMe: {_nvme_builds()}. Free space by parking a "
            f"finished losing build (chain2.sh park_build: verified copy to {PARK_ROOT}/<name>, NVMe dir -> symlink) "
            f"or deleting dead replay streams (s04/s07 *_replay), then re-run (it resumes). Never move a build a "
            f"GPU step will load next.")
    env.log(f"df ok for {what}: {free / GIB:.0f} GiB free on {path}, need {need / GIB:.0f} + {margin / GIB:.0f} GiB", STAGE)


def _copy_fsync(src: Path, dst: Path) -> None:
    tmp = dst.with_name(dst.name + ".part")
    with open(src, "rb") as fi, open(tmp, "wb") as fo:
        shutil.copyfileobj(fi, fo, 64 << 20)
        fo.flush()
        os.fsync(fo.fileno())
    os.replace(tmp, dst)


def park_build(name: str, *, park_root: Path | None = None, seagate: Path | None = None) -> bool:
    """Move a FINISHED build thor/<name> off the NVMe, exactly as chain2.sh park_build leaves it (so its
    unpark_build / move_thor_* steps find it): copy to <park_root>/<name>, sha256 of every file checked
    against thor.json, marker thor/<name>.parked.json, then the NVMe dir is replaced by a symlink to the copy
    and deleted. Readers of thor.json keep working through the symlink; a GPU step that loads the experts
    must unpark first (the Seagate is NTFS over FUSE: mmap faults crawl). Returns True if it moved bytes."""
    park_root = Path(park_root or PARK_ROOT)
    seagate = Path(seagate or orchestrate.SEAGATE)
    src, dst = THOR / name, park_root / name
    gone, marker = THOR / f".{name}.parking", THOR / f"{name}.parked.json"
    if gone.exists():                                       # crashed between the rename and the rmtree
        if not src.exists() and not src.is_symlink() and marker.exists():
            os.symlink(json.load(open(marker))["location"], src)
        if src.is_symlink():
            shutil.rmtree(gone)
    if src.is_symlink() or not (src / "thor.json").exists():
        return False
    b = TF.Build(src)
    if not b.m.get("complete"):
        raise SystemExit(f"refusing to park thor/{name}: build is not complete (it would be resumed, not parked)")
    if not os.path.ismount(seagate):
        raise SystemExit(f"cannot park thor/{name}: the Seagate is not mounted at {seagate}")
    size = _du(src)
    if orchestrate.disk_free(seagate) - max(size - _du(dst), 0) < PARK_MARGIN:
        raise SystemExit(f"cannot park thor/{name}: {orchestrate.disk_free(seagate) / GIB:.0f} GiB free on the Seagate, "
                         f"needs {size / GIB:.0f} GiB + {PARK_MARGIN / GIB:.0f} GiB margin")
    dst.mkdir(parents=True, exist_ok=True)
    files = sorted(b.m["files"])
    for f in files:
        if not ((dst / f).exists() and (dst / f).stat().st_size == (src / f).stat().st_size):
            _copy_fsync(src / f, dst / f)
    _copy_fsync(src / "thor.json", dst / "thor.json")
    bad = TF.Build(dst).verify()
    if bad:
        for f in bad:
            (dst / f).unlink(missing_ok=True)
        raise SystemExit(f"parking thor/{name}: the Seagate copy of {bad} does not match thor.json sha256 (removed; re-run)")
    env.atomic_json(marker, {"name": name, "location": str(dst), "from": str(src), "bytes": size, "files": len(files),
                             "verified": "sha256 of every file against thor.json", "convention": "chain2.sh park_build",
                             "ts": env.now()})
    src.rename(gone)
    os.symlink(dst, src)
    shutil.rmtree(gone)
    env.log(f"thor/{name} parked on {dst} (verified, {size / GIB:.1f} GiB)", STAGE)
    return True


# ----------------------------------------------------------------------------- (b) Thor A
def build_dir(n_keep: int, arm: str) -> Path:
    return THOR / f"{arm}{n_keep}"


def _quantize(W, H, t: str, a, sgn):
    """-> (stored bytes uint8 [B, R, row_bytes], fp16 kt_scale [B, R, in/128] or None)."""
    from ops.lib import gptq
    if t in KT.QTYPES:
        return KT.quantize_ldlq(W, H, t, sgn, damp=a.damp)
    codes, params = gptq.quantize(W, H, t, damp=a.damp)
    return gptq.pack(codes, params), None


def _dequant_bf16(t: str, q, sc, in_f: int, sgn):
    """the logical (un-rotated) weights the next phase runs with, bf16 on the GPU."""
    import torch
    if t in KT.QTYPES:
        return KT.dequant_t(t, q, sc, in_f, sgn).to(torch.bfloat16)
    return torch.from_numpy(G.dequant(t, q)).to("cuda", torch.bfloat16)


def gptq_layer(M, blk, args, run, cache, L, keep, fp, alloc, gain, a, b: TF.Build):
    """decide() of the two-phase unit: Hessians from the cached FFN inputs, GPTQ w1|w3, then w2 on
    the quantized w1|w3 activations; write the layer file; return the quantized arm layer. The same
    unit builds the trellis arm when the allocation names KT qtypes (block LDLQ instead of GPTQ)."""
    import torch
    from ops.lib import replay
    fname = TF.layer_file(L)
    if b.has(fname):
        return replay.ArmLayer(E, keep, 1.0, replay.thor_experts(M, args, b.root, L, keep, fp))
    rd = src_reader(L)
    rows = run.rows
    keep_t = sorted(keep)
    n = len(keep_t)
    t13, t2 = alloc["w13"], alloc["w2"]
    sg = kt_signs(L)
    b3 = np.load(env.WORK / "s03" / "bank" / f"L{L}.npz")
    g = blk.ffn.gate

    def routed(r):
        c = np.load(cache / f"r{r:05d}.npz")
        h = torch.from_numpy(c["ffn_in"]).cuda().view(torch.bfloat16)
        im = replay.image_mask(run, r)
        w, i = g(h, None if im is None else im.flatten())
        return h, w, i

    rb = lambda t, c: KT.row_bytes(t, c) if t in KT.QTYPES else G.row_bytes(t, c)
    q13 = np.empty((n, 2 * INTER, rb(t13, DIM)), np.uint8)
    q2 = np.empty((n, DIM, rb(t2, INTER)), np.uint8)
    s13 = np.empty((n, 2 * INTER, DIM // KT.T), np.float16) if t13 in KT.QTYPES else None
    s2 = np.empty((n, DIM, INTER // KT.T), np.float16) if t2 in KT.QTYPES else None
    ntok = np.zeros(n, np.int64)
    for g0 in range(0, n, a.gptq_batch):
        grp = keep_t[g0:g0 + a.gptq_batch]
        B = len(grp)
        pos = {e: k for k, e in enumerate(grp)}
        Hin = torch.zeros(B, DIM, DIM, dtype=torch.float32, device="cuda")
        with torch.inference_mode():
            for r in rows:
                h, w, i = routed(r)
                x = replay.fake_act_quant(h).float()
                for e in grp:
                    idx = torch.where((i == e).any(-1))[0]
                    if idx.numel():
                        xe = x[idx]
                        Hin[pos[e]] += xe.T @ xe
                        ntok[g0 + pos[e]] += idx.numel()
        W13 = torch.stack([torch.from_numpy(np.concatenate([G.src_dequant(*expert_src(rd, L, e, "w1")),
                                                            G.src_dequant(*expert_src(rd, L, e, "w3"))])) for e in grp]).cuda()
        Hin = _fallback_hessian(Hin, ntok[g0:g0 + B], b3["x2_in"][grp], b3["n_tok"][grp], a.min_tokens)
        q13[g0:g0 + B], sc = _quantize(W13, Hin, t13, a, sg["w13"])
        if s13 is not None:
            s13[g0:g0 + B] = sc
        del Hin, W13
        deq = _dequant_bf16(t13, q13[g0:g0 + B], None if s13 is None else s13[g0:g0 + B], DIM, sg["w13"])   # [B, 2*INTER, DIM]
        fq = {e: replay.FQExpert(deq[k, :INTER], None, deq[k, INTER:], args.swiglu_limit) for e, k in pos.items()}
        Hmid = torch.zeros(B, INTER, INTER, dtype=torch.float32, device="cuda")
        with torch.inference_mode():
            for r in rows:
                h, w, i = routed(r)
                for e in grp:
                    idx, top = torch.where(i == e)
                    if idx.numel():
                        mid = _mid(fq[e], h[idx], w[idx, top, None])
                        mq = replay.fake_act_quant(mid).float()
                        Hmid[pos[e]] += mq.T @ mq
        W2 = torch.stack([torch.from_numpy(G.src_dequant(*expert_src(rd, L, e, "w2"))) for e in grp]).cuda() * float(gain)
        Hmid = _fallback_hessian(Hmid, ntok[g0:g0 + B], b3["x2_mid"][grp], b3["n_tok"][grp], a.min_tokens)
        q2[g0:g0 + B], sc = _quantize(W2, Hmid, t2, a, sg["w2"])
        if s2 is not None:
            s2[g0:g0 + B] = sc
        del Hmid, W2, deq, fq
        torch.cuda.empty_cache()
        env.log(f"L{L}: GPTQ experts {g0}..{g0 + B - 1} ({t13}/{t2})", STAGE)
    items = dense_items(rd, L, keep_t) + engram_items(L)
    items += [{"name": TF.expert_name(L, "w1"), "dtype": "U8", "data": np.ascontiguousarray(q13[:, :INTER]), "qtype": t13, "shape": [n, INTER, DIM]},
              {"name": TF.expert_name(L, "w3"), "dtype": "U8", "data": np.ascontiguousarray(q13[:, INTER:]), "qtype": t13, "shape": [n, INTER, DIM]},
              {"name": TF.expert_name(L, "w2"), "dtype": "U8", "data": q2, "qtype": t2, "shape": [n, DIM, INTER]}]
    if s13 is not None:
        items += [{"name": TF.expert_name(L, "w1", "kt_scale"), "dtype": "F16", "data": np.ascontiguousarray(s13[:, :INTER]), "qtype": t13},
                  {"name": TF.expert_name(L, "w3", "kt_scale"), "dtype": "F16", "data": np.ascontiguousarray(s13[:, INTER:]), "qtype": t13},
                  {"name": TF.expert_name(L, "w13", "rht_sign"), "dtype": "U8", "data": KT.pack_signs(sg["w13"]), "qtype": t13}]
    if s2 is not None:
        items += [{"name": TF.expert_name(L, "w2", "kt_scale"), "dtype": "F16", "data": s2, "qtype": t2},
                  {"name": TF.expert_name(L, "w2", "rht_sign"), "dtype": "U8", "data": KT.pack_signs(sg["w2"]), "qtype": t2}]
    b.write_file(fname, items, meta={"layer": str(L), "keepset_sha256": fp, "heal_gain_folded": str(gain),
                                     "calib_tokens_per_expert": json.dumps(ntok.tolist())})
    return replay.ArmLayer(E, keep, 1.0, replay.thor_experts(M, args, b.root, L, keep, fp))


def _mid(fq, x, w):
    import torch
    import torch.nn.functional as F
    xq = replay_fq(x)
    gate = F.linear(xq, fq.w1).float()
    up = F.linear(xq, fq.w3).float()
    if fq.swiglu_limit > 0:
        up = torch.clamp(up, min=-fq.swiglu_limit, max=fq.swiglu_limit)
        gate = torch.clamp(gate, max=fq.swiglu_limit)
    return (w * (F.silu(gate) * up)).to(x.dtype)


def replay_fq(x):
    from ops.lib import replay
    return replay.fake_act_quant(x)


def _fallback_hessian(H, ntok, x2_teacher, n_teacher, min_tokens):
    """Experts with fewer than `min_tokens` calibration tokens get a diagonal Hessian from the
    teacher's s03 imatrix (many more tokens): weighted RTN instead of an under-determined GPTQ."""
    import torch
    for k in range(H.shape[0]):
        if ntok[k] < min_tokens:
            d = torch.from_numpy(x2_teacher[k] / max(n_teacher[k], 1) + 1e-8).float().cuda()
            H[k] = torch.diag(d)
    return H


def _variant(trellis: bool) -> dict:
    """the K-quant Thor A build and its pre-registered trellis sub-arm share every step but these."""
    if trellis:
        return {"suffix": "kt", "alloc": OUT / "alloc_kt.json", "replay": OUT / "thor_kt_replay", "stream": "thorAkt",
                "folded": "w2 before LDLQ"}
    return {"suffix": "", "alloc": OUT / "alloc.json", "replay": OUT / "thor_replay", "stream": "thorA",
            "folded": "w2 before GPTQ"}


def cmd_thor(a, trellis: bool = False):
    from ops.lib import replay
    V = _variant(trellis)
    keep, fp = keepset.load(Path(a.keepset))
    n_keep = len(keep["0"])
    if trellis:
        if not KT._LIB_PATH.exists():
            raise SystemExit(f"{KT._LIB_PATH} missing: cuda/scripts/build_v41.sh libkt_viterbi")
        plan_kt(a, keep, fp, n_keep)
    alloc = json.load(open(V["alloc"]))
    keepset.require(alloc["keepset_sha256"], fp, where=str(V["alloc"]))
    gains = heal_decision(fp)
    extra = {"kt": {"window_bits": KT.L, "block": KT.T, "rht_block": KT.RHT, "codebook": "3INST (QTIP)",
                    "sign_rule": "sha256('v41-kt-rht/L<L>.<w13|w2>') seeds numpy default_rng; stored as rht_sign",
                    "encoder": "block LDLQ + tail-biting Viterbi (ops/lib/kt.py)"}} if trellis else {}
    b = TF.Build(build_dir(n_keep, "A" + V["suffix"]), create=True, keepset_sha256=fp, n_routed_experts=n_keep, arm="A",
                 source_revision=env.REVISION, heal={"gains": gains, "folded_into": V["folded"]},
                 alloc=alloc["alloc"], protected=PROTECTED, **extra)
    if not b.m.get("complete"):
        need = max(kq_build_estimate(alloc["alloc"], n_keep) - b.stored_bytes(), 0) + \
            max(replay_peak(a.tokens) - _du(V["replay"]), 0)
        disk_preflight("thor-trellis" if trellis else "thor", need, THOR_MARGIN)
    rows, man = rowsets.load_rows(CORPUS)
    ids = rowsets.subset(rowsets.split_rows(rows, man["spec_hash"], "calib"), a.tokens, spec.TOKEN_TARGET)
    run = replay.prepare(V["replay"], ids, [{"name": V["stream"], "keep": keep}], man["spec_hash"])
    me = Path(__file__)
    common = ["--keepset", a.keepset, "--kernels", a.kernels, "--token-chunk", str(a.token_chunk),
              "--gptq-batch", str(a.gptq_batch), "--damp", str(a.damp), "--min-tokens", str(a.min_tokens)]
    if trellis:
        common.append("--trellis")
    if not run.led.done("_embed"):
        orchestrate.run_unit(me, ["unit", "embed", *common], stage=STAGE, gpu=True, min_avail_gib=a.min_avail_gib)
    for L in range(N_LAYERS):
        if not replay.Run(V["replay"]).led.done(f"{V['stream']}/L{L}"):
            orchestrate.run_unit(me, ["unit", "layer", "--layer", str(L), *common], stage=STAGE, gpu=True,
                                 min_avail_gib=a.min_avail_gib)
    # the layer units rewrote thor.json from their own processes: re-open it, or the parent's stale copy would
    # erase every layer entry on the next save (and finish_build would never see the layers)
    b = TF.Build(build_dir(n_keep, "A" + V["suffix"]), keepset_sha256=fp, n_routed_experts=n_keep)
    write_globals(b)
    finish_build(b, n_keep)
    run = replay.Run(V["replay"])
    replay.drop_streams(run, V["stream"])
    for p in ("h.npy", "pre_mix.npy"):
        (run.root / "_embed" / p).unlink(missing_ok=True)
    env.log(f"Thor A{n_keep}{V['suffix']} complete: {b.m['expert_bpw']:.3f} expert bpw, "
            f"{b.m['expert_bytes'] / 2**30:.2f} GiB experts", STAGE)


def unit(a):
    from ops.lib import replay
    V = _variant(a.trellis)
    run = replay.Run(V["replay"])
    if a.unit == "embed":
        return replay.unit_embed(run, a.kernels)
    keep, fp = keepset.load(Path(a.keepset))
    n_keep = len(keep["0"])
    alloc = json.load(open(V["alloc"]))["alloc"][str(a.layer)]
    gain = heal_decision(fp)[str(a.layer)]
    b = TF.Build(build_dir(n_keep, "A" + V["suffix"]), keepset_sha256=fp, n_routed_experts=n_keep)
    L = a.layer

    def decide(M, blk, args, run_, cache):
        return gptq_layer(M, blk, args, run_, cache, L, keep[str(L)], fp, alloc, gain, a, b)

    replay.unit_layer_two_phase(run, L, a.kernels, a.token_chunk, V["stream"], measure=lambda *x: None, decide=decide,
                                cache_ffn_in=True)


# ----------------------------------------------------------------------------- Thor B, overlay
def cmd_thor_b(a):
    keep, fp = keepset.load(Path(a.keepset))
    n_keep = len(keep["0"])
    b = TF.Build(build_dir(n_keep, "B"), create=True, keepset_sha256=fp, n_routed_experts=n_keep, arm="B",
                 source_revision=env.REVISION, heal={"gains": None, "note": "arm B ships unhealed (s06 measured arm A's keep-set)"},
                 alloc={str(L): {"w13": "MXFP4_SRC", "w2": "MXFP4_SRC"} for L in range(N_LAYERS)})
    if not b.m.get("complete"):
        make_room_for_b(max(mxfp4_build_estimate(n_keep) - b.stored_bytes(), 0))
    for L in range(N_LAYERS):
        fname = TF.layer_file(L)
        if b.has(fname):
            continue
        rd = src_reader(L)
        kt = sorted(keep[str(L)])
        items = dense_items(rd, L, kt) + engram_items(L)
        for p, (o, i) in (("w1", (INTER, DIM)), ("w2", (DIM, INTER)), ("w3", (INTER, DIM))):
            w = np.stack([np.asarray(expert_src(rd, L, e, p)[0]).view(np.uint8) for e in kt])
            s = np.stack([np.asarray(expert_src(rd, L, e, p)[1]).view(np.uint8) for e in kt])
            items += [{"name": TF.expert_name(L, p, "weight"), "dtype": "U8", "data": w, "qtype": "MXFP4_SRC", "shape": [len(kt), o, i]},
                      {"name": TF.expert_name(L, p, "scale"), "dtype": "F8_E8M0", "data": s, "qtype": "MXFP4_SRC"}]
        b.write_file(fname, items, meta={"layer": str(L), "keepset_sha256": fp})
        env.log(f"thor B L{L} written", STAGE)
    write_globals(b)
    finish_build(b, n_keep)


def make_room_for_b(need: int) -> None:
    """thor-b runs (chain.sh) right after s10 picked arm B, with the losing arm-A build(s) still on NVMe and
    not enough room for B. Park the losers on the Seagate (chain2 convention, verified) until B fits;
    B itself is always built on NVMe (it is served and GPU-loaded)."""
    nA = len(keepset.load(env.WORK / "s04" / "keepset.json")[0]["0"])
    for name in (f"A{nA}", f"A{nA}kt"):
        if orchestrate.disk_free(env.WORK) - need >= THOR_B_MARGIN:
            break
        if (THOR / name / "thor.json").exists() and not (THOR / name).is_symlink():
            env.log(f"thor-b: {orchestrate.disk_free(env.WORK) / GIB:.0f} GiB free, B needs {need / GIB:.0f} + "
                    f"{THOR_B_MARGIN / GIB:.0f}: parking the losing build thor/{name}", STAGE)
            park_build(name)
    disk_preflight("thor-b", need, THOR_B_MARGIN)


def cmd_overlay(a):
    """FP8 dense Linears of every decoder layer -> NVFP4 (RTN), named by module path in the block."""
    od = THOR / "overlay_nvfp4"
    od.mkdir(parents=True, exist_ok=True)
    led = env.Ledger(env.LOGS / "s07_overlay_ledger.json")
    for L in range(N_LAYERS):
        if led.done(f"L{L}"):
            continue
        rd = src_reader(L)
        items = []
        for n in sorted(rd.names(), key=rd.offset):
            if EXPERT.match(n) or not n.endswith(".weight") or rd.dtype(n) != "F8_E4M3":
                continue
            w = rd.get(n).view(np.uint8)
            s = rd.get(n[:-len("weight")] + "scale")
            W = G.e4m3_decode(w) * np.repeat(np.repeat(G.e8m0(s), 32, 0), 32, 1)[: w.shape[0], : w.shape[1]]
            q = G.nvfp4_quantize(W)
            mod = n[len(f"layers.{L}."):-len(".weight")]
            items += [(mod + ".weight", "U8", q["weight"]), (mod + ".weight_scale", "F8_E4M3", q["weight_scale"]),
                      (mod + ".weight_scale_2", "F32", np.asarray([q["weight_scale_2"]], np.float32))]
        st_io.write(od / f"L{L:02d}.safetensors", items, metadata={"format": "thor-overlay-nvfp4", "layer": str(L)})
        led.finish(f"L{L}", tensors=len(items) // 3)
        env.log(f"overlay L{L}: {len(items) // 3} linears", STAGE)
    env.atomic_json(od / "overlay.json", {"format": "thor-overlay-nvfp4", "layers": N_LAYERS,
                                          "rule": "every F8_E4M3 *.weight of a decoder-layer shard except routed experts",
                                          "ts": env.now()})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["nvfp4", "plan", "thor", "thor-b", "thor-trellis", "overlay", "unit"])
    ap.add_argument("unit", nargs="?", choices=["embed", "layer"])
    ap.add_argument("--layer", type=int)
    ap.add_argument("--keepset", default=str(env.WORK / "s04" / "keepset.json"))
    ap.add_argument("--repo")
    ap.add_argument("--upload", action="store_true", help="nvfp4: upload to a PRIVATE repo as it goes (needs PRO quota)")
    ap.add_argument("--no-upload", action="store_true", help="default; kept for compatibility")
    ap.add_argument("--kernels", choices=["tilelang", "torch"], default="tilelang")
    ap.add_argument("--token-chunk", type=int, default=8192)
    ap.add_argument("--tokens", type=int, default=262_144, help="calibration tokens for the GPTQ Hessians")
    ap.add_argument("--gptq-batch", type=int, default=32, help="experts per GPTQ batch (~17 GB peak at 32)")
    ap.add_argument("--damp", type=float, default=0.01)
    ap.add_argument("--min-tokens", type=int, default=256, help="fewer routed calibration tokens -> diagonal (imatrix) Hessian")
    ap.add_argument("--sample", type=int, default=4, help="experts per layer sampled by `plan`")
    ap.add_argument("--kt-rows", type=int, default=16, help="rows per sampled projection in the trellis plan (numpy Viterbi)")
    ap.add_argument("--trellis", action="store_true", help="(unit) build the KT trellis sub-arm")
    ap.add_argument("--budget-bytes", type=int)
    ap.add_argument("--min-avail-gib", type=float, default=100.0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.cmd == "thor-b" and a.keepset == str(env.WORK / "s04" / "keepset.json"):
        a.keepset = str(env.WORK / "s04" / "keepset_128.json")
    {"nvfp4": cmd_nvfp4, "plan": cmd_plan, "thor": cmd_thor, "thor-b": cmd_thor_b, "overlay": cmd_overlay,
     "thor-trellis": lambda x: cmd_thor(x, trellis=True), "unit": unit}[a.cmd](a)


if __name__ == "__main__":
    main()
