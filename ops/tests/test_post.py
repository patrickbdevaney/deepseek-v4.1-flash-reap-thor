#!/usr/bin/env python3
"""Pure numpy tests for the post-saliency stages (s04-s12). No torch, no model, no GPU.

    python3 ops/tests/test_post.py

ggml block layouts are checked against llama.cpp's gguf-py (pure numpy) when the fork checkout at
llamacpp/llama.cpp is present; those tests report `skip` otherwise.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

TMP = Path(tempfile.mkdtemp(prefix="v41post_"))
os.environ["V41_WORK"] = str(TMP / "work")
sys.modules["torch"] = None
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from ops.lib import (cards, drift, env, gates, ggml_quants as G, keepset, metrics, rowsets,  # noqa: E402
                     select as SEL, st_io, stats, thor_format as TF)

GGUF_PY = REPO / "llamacpp" / "llama.cpp" / "gguf-py"
RESULTS = []


def test(fn):
    try:
        r = fn()
        RESULTS.append((fn.__name__, "skip" if r == "skip" else "ok", ""))
    except Exception as e:
        RESULTS.append((fn.__name__, "FAIL", f"{type(e).__name__}: {e}"))
    return fn


def gguf_quants():
    if not GGUF_PY.exists():
        return None
    sys.path.insert(0, str(GGUF_PY))
    try:
        from gguf import quants
        from gguf.constants import GGMLQuantizationType as T
        return quants, T
    except Exception:
        return None


# ----------------------------------------------------------------------------- keep-sets
@test
def keepset_fingerprint_is_canonical_and_fatal():
    a = {"1": [5, 3, 9], "0": [2, 1, 0]}
    b = {"0": [0, 1, 2], "1": [3, 5, 9]}
    assert keepset.fingerprint(a) == keepset.fingerprint(b)
    assert keepset.fingerprint(a) != keepset.fingerprint({"0": [0, 1, 3], "1": [3, 5, 9]})
    p = TMP / "k.json"
    fp = keepset.save(p, a, arm="A")
    k, fp2 = keepset.load(p, expect=fp)
    assert fp == fp2 and k["0"] == [0, 1, 2]
    d = json.load(open(p)); d["keep"]["0"] = [0, 1, 4]; json.dump(d, open(p, "w"))
    try:
        keepset.load(p)
        raise AssertionError("hand-edited keep-set accepted")
    except keepset.KeepsetMismatch:
        pass
    # GLM v3 defect: a broad `except Exception` swallowed the guard. This one cannot be swallowed.
    assert not issubclass(keepset.KeepsetMismatch, Exception)
    try:
        try:
            keepset.require("a" * 64, "b" * 64)
        except Exception:
            raise AssertionError("caught by except Exception")
    except keepset.KeepsetMismatch:
        pass
    assert keepset.remap([7, 3, 9]) == {3: 0, 7: 1, 9: 2}
    try:
        keepset.canonical({"0": [1, 1]})
        raise AssertionError("duplicate ids accepted")
    except ValueError:
        pass


# ----------------------------------------------------------------------------- row sets
@test
def rowsets_split_and_subset():
    rows, off = [], 0
    rng = np.random.default_rng(0)
    for r in range(400):
        b = ["general", "code", "longctx"][r % 3]
        n = 40000 if b == "longctx" else 8192
        rows.append({"row": r, "bucket": b, "heldout": r % 5 == 0, "offset": off, "length": n})
        off += n
    sc = rowsets.split_rows(rows, "abc", "screen")
    cf = rowsets.split_rows(rows, "abc", "confirm")
    assert sc and cf and not ({r["row"] for r in sc} & {r["row"] for r in cf})
    assert len(sc) + len(cf) == sum(r["heldout"] for r in rows)
    assert rowsets.split_rows(rows, "abc", "screen") == sc          # deterministic
    assert {r["row"] for r in rowsets.split_rows(rows, "xyz", "screen")} != {r["row"] for r in sc}
    ids = rowsets.subset(sc, 100_000, {"general": .45, "code": .45, "longctx": .10})
    by = {rows[i]["bucket"] for i in ids}
    assert by == {"general", "code", "longctx"}, by                  # long row admitted despite its share
    assert ids == sorted(ids)
    assert rowsets.subset(sc, 100_000, {"general": .45, "code": .45, "longctx": .10}) == ids


# ----------------------------------------------------------------------------- stats
@test
def stats_bootstrap_mcnemar_mcs():
    rng = np.random.default_rng(1)
    cl = np.repeat(np.arange(200), 50)
    v = rng.normal(1.0, 1.0, cl.size) + rng.normal(0, 0.5, 200)[cl]
    r = stats.cluster_bootstrap_mean(v, cl, n_draws=500)
    assert r["lo"] < 1.0 < r["hi"] and r["lo"] < r["mean"] < r["hi"]
    d = stats.paired_diff(v - 0.3, v, cl, n_draws=300)
    assert d["excludes_zero"] and abs(d["mean"] + 0.3) < 1e-9
    a = rng.random(5000) < 0.9
    b = a.copy(); b[:200] = False
    m = stats.mcnemar(a, b)
    assert m["z"] > 5 and m["p"] < 1e-6 and stats.mcnemar(b, a)["z"] == -m["z"]
    L = {"good": rng.normal(0.30, 0.1, cl.size), "bad": rng.normal(0.60, 0.1, cl.size), "worse": rng.normal(0.9, 0.1, cl.size)}
    mcs = stats.model_confidence_set(L, cl, n_draws=300)
    assert mcs["survivors"] == ["good"], mcs
    L2 = {"x": L["good"], "y": L["good"] + rng.normal(0, 1e-4, cl.size)}
    assert len(stats.model_confidence_set(L2, cl, n_draws=300)["survivors"]) == 2


# ----------------------------------------------------------------------------- selection
def _bank(rng, E=32, nb=3):
    s = rng.random((nb, E)) * 10
    c = rng.integers(1, 100, (nb, E)).astype(float)
    Fs = rng.random((E, E)); Fs = Fs + Fs.T
    return {"sum": s, "cnt": c, "gat": rng.random((nb, E)), "nrm": s * 2, "F_sum": Fs, "F_cnt": np.ones((E, E)),
            "buckets": np.array(["general", "code", "vision"])}


@test
def select_masks_respect_guards_and_sizes():
    rng = np.random.default_rng(2)
    bank = _bank(rng)
    crit = SEL.criteria(bank)
    prot = np.array([0, 1, 2])
    for mode in ("reap_sum", "mass", "freq", "random"):
        pr = SEL.prune_set(mode, n_keep=16, protect=prot, crit=crit, seed=3, E=32)
        assert pr.size == 16 and not set(prot) & set(pr.tolist()), mode
    pr = SEL.prune_set("hope", n_keep=16, protect=prot, F=SEL.hope_F(bank, "sum"))
    assert pr.size == 16 and not set(prot) & set(pr.tolist())
    s = crit["reap_sum"].copy(); s[prot] = np.inf
    assert np.array_equal(SEL.prune_set("reap_sum", n_keep=16, protect=prot, crit=crit, E=32), np.sort(np.argsort(s)[:16]))
    try:
        SEL.prune_set("reap_sum", n_keep=2, protect=prot, crit=crit, E=32)
        raise AssertionError("infeasible protection accepted")
    except SEL.Infeasible:
        pass
    assert SEL.keep_from_pruned(pr, 32) == sorted(set(range(32)) - set(pr.tolist()))
    # per-pool protection reaches the top experts of a small pool
    P = np.zeros((2, 32)); P[0] = np.arange(32); P[1, 5] = 1.0
    assert 5 in SEL.pool_protect(P, 0.04, np.array([True, True])).tolist()
    assert SEL.retention(P, np.array([5]))[1] == 0.0


@test
def select_image_saliency_and_sweep():
    rng = np.random.default_rng(3)
    idx = rng.integers(0, 32, (1000, 6)); w = np.full((1000, 6), 0.25)
    mask = np.zeros(1000, bool); mask[:100] = True
    idx[:100, 0] = 7
    img = SEL.image_saliency(idx, w, mask, np.ones(32))
    assert img.argmax() == 7 and abs(img.sum() - 100 * 6 * 0.25) < 1e-9
    bank = _bank(rng)
    names, P = SEL.pools(bank, img)
    assert names[-1] == "image" and P.shape == (4, 32)
    layers = [{"F": SEL.hope_F(bank, "sum"), "P": P, "live": P.sum(1) > 0, "guard": np.array([0])}]
    sw = SEL.sweep_pf(layers, 16, "sum", grid=[0.0, 0.03, 0.06, 0.5])
    assert sw["rows"][-1].get("infeasible") and sw["knee"] in (0.0, 0.03, 0.06)


# ----------------------------------------------------------------------------- ggml blocks
@test
def ggml_layouts_match_gguf_py():
    q = gguf_quants()
    if q is None:
        return "skip"
    quants, T = q
    rng = np.random.default_rng(4)
    W = (rng.standard_normal((16, 512)) * 0.02).astype(np.float32); W[2, 9] = 0.4
    A = rng.random(512) + 0.05
    errs = {}
    for t in ("Q2_K", "Q3_K", "Q4_K"):
        b = G.quantize_rtn(t, W, A)
        assert b.shape == (16, G.row_bytes(t, 512))
        ref = quants.dequantize(b, getattr(T, t))
        mine = G.dequant(t, b)
        assert np.array_equal(ref, mine), t
        errs[t] = float(np.linalg.norm(mine - W) / np.linalg.norm(W))
    assert errs["Q2_K"] > errs["Q3_K"] > errs["Q4_K"] and errs["Q4_K"] < 0.12, errs
    wb = rng.integers(0, 256, (8, 64), dtype=np.uint8); sc = rng.integers(115, 135, (8, 4), dtype=np.uint8)
    gb = G.src_to_ggml_mxfp4(wb, sc)
    assert np.array_equal(quants.dequantize(gb, T.MXFP4), G.src_dequant(wb, sc))


@test
def ggml_codes_pack_identity():
    rng = np.random.default_rng(5)
    W = (rng.standard_normal((6, 256)) * 0.05).astype(np.float32)
    for t in ("Q2_K", "Q3_K", "Q4_K"):
        p = G.FIT[t](W, np.ones_like(W))
        q = G.codes(W, p)
        lo, hi = G.QRANGE[t]
        assert q.min() >= lo and q.max() <= hi
        assert np.allclose(G.dequant(t, G.PACK[t](q, p)), G.values(q, p), atol=0, rtol=1e-6), t


@test
def nvfp4_repack_lossless_and_refuses_lossy():
    rng = np.random.default_rng(6)
    wb = rng.integers(0, 256, (16, 128), dtype=np.uint8)
    sc = rng.integers(118, 130, (16, 8), dtype=np.uint8)
    for gain in (1.0, 0.9111):
        r = G.mxfp4_to_nvfp4(wb, sc, gain)
        assert r["lossy_blocks"] == 0 and np.array_equal(r["weight"], wb)
        assert np.array_equal(G.nvfp4_dequant(r["weight"], r["weight_scale"], r["weight_scale_2"]),
                              G.src_dequant(wb, sc) * np.float32(gain))
    sc2 = sc.copy(); sc2[0, 0] = 90
    assert G.mxfp4_to_nvfp4(wb, sc2)["lossy_blocks"] >= 1
    wz = wb.copy(); wz[0, :16] = 0; sc3 = sc.copy(); sc3[0, 0] = 1        # an all-zero block at a tiny scale is fine
    assert G.mxfp4_to_nvfp4(wz, sc3)["lossy_blocks"] == 0
    k = np.arange(-9, 9)
    assert np.array_equal(G.e4m3_decode(G.e4m3_pow2(k)), np.ldexp(1.0, k).astype(np.float32))
    o = G.nvfp4_quantize((rng.standard_normal((32, 64)) * 0.1).astype(np.float32))
    assert o["weight"].shape == (32, 32) and o["weight_scale"].shape == (32, 4)


# ----------------------------------------------------------------------------- files
@test
def st_io_alignment_and_roundtrip():
    p = TMP / "a.safetensors"
    t = [("x", "F32", np.arange(3, dtype=np.float32)), ("y", "U8", np.arange(7, dtype=np.uint8)),
         ("z", "BF16", np.arange(10, dtype=np.uint16).reshape(2, 5)), ("s", "F32", np.asarray(2.5, np.float32))]
    offs = st_io.write(p, t, metadata={"k": {"a": 1}})
    rd = st_io.Reader(p)
    assert rd.names() == ["x", "y", "z", "s"]
    assert all(offs[n][0] % st_io.ALIGN == 0 for n in rd.names())
    assert any(n.startswith("__pad.") for n in rd.h)
    for n, d, a in t:
        assert rd.dtype(n) == d and np.array_equal(rd.get(n), a)
    assert json.loads(rd.meta["k"]) == {"a": 1}
    p2 = TMP / "b.safetensors"
    st_io.write(p2, t, align=1)
    r2 = st_io.Reader(p2)
    assert not any(n.startswith("__pad.") for n in r2.h) and np.array_equal(r2.get("z"), t[2][2])
    with open(p2, "rb") as fh:
        n = int.from_bytes(fh.read(8), "little")
    assert (8 + n) % 8 == 0


@test
def thor_build_roundtrip():
    rng = np.random.default_rng(7)
    b = TF.Build(TMP / "thor", create=True, keepset_sha256="f" * 64, n_routed_experts=2)
    W = (rng.standard_normal((2, 4, 512)) * 0.02).astype(np.float32)
    qb = np.stack([G.quantize_rtn("Q3_K", w) for w in W])
    wb = rng.integers(0, 256, (2, 4, 256), dtype=np.uint8); sc = rng.integers(120, 130, (2, 4, 16), dtype=np.uint8)
    b.write_file("layer-00.safetensors", [
        {"name": TF.expert_name(0, "w1"), "dtype": "U8", "data": qb, "qtype": "Q3_K", "shape": [2, 4, 512]},
        {"name": TF.expert_name(0, "w2", "weight"), "dtype": "U8", "data": wb, "qtype": "MXFP4_SRC", "shape": [2, 4, 512]},
        {"name": TF.expert_name(0, "w2", "scale"), "dtype": "F8_E8M0", "data": sc, "qtype": "MXFP4_SRC"},
        {"name": "layers.0.ffn.gate.bias", "dtype": "F32", "data": np.zeros(2, np.float32)}])
    assert b.has("layer-00.safetensors") and b.verify() == []
    rd = b.reader("layer-00.safetensors")
    d = TF.dequant_stack("Q3_K", rd.get(TF.expert_name(0, "w1")), 4, 512, idx=1)
    assert np.array_equal(d, G.dequant("Q3_K", qb[1]))
    m = TF.dequant_stack("MXFP4_SRC", rd.get(TF.expert_name(0, "w2", "weight")), 4, 512,
                         rd.get(TF.expert_name(0, "w2", "scale")), idx=0)
    assert np.array_equal(m, G.src_dequant(wb[0], sc[0]))
    assert b.expert_bytes() == qb.nbytes + wb.nbytes + sc.nbytes
    try:
        TF.Build(TMP / "thor", keepset_sha256="e" * 64, n_routed_experts=2)
        raise AssertionError("foreign keep-set accepted")
    except keepset.KeepsetMismatch:
        pass


# ----------------------------------------------------------------------------- s05 surgery
@test
def s05_surgery_is_bit_exact_and_checked():
    from ops import s05_surgery as S5
    rng = np.random.default_rng(8)
    L, E = 3, 6
    t = []
    for e in range(E):
        for p in ("w1", "w2", "w3"):
            t.append((f"layers.{L}.ffn.experts.{e}.{p}.weight", "I8", rng.integers(-128, 128, (4, 16), dtype=np.int8)))
            t.append((f"layers.{L}.ffn.experts.{e}.{p}.scale", "F8_E8M0", rng.integers(100, 140, (4, 1), dtype=np.uint8)))
    t += [(f"layers.{L}.ffn.gate.weight", "BF16", rng.integers(0, 60000, (E, 8), dtype=np.uint16)),
          (f"layers.{L}.ffn.gate.bias", "F32", rng.random(E).astype(np.float32)),
          (f"layers.{L}.ffn.gate.bias_vl", "F32", rng.random(E).astype(np.float32)),
          (f"layers.{L}.attn.wq_a.weight", "F8_E4M3", rng.integers(0, 255, (8, 8), dtype=np.uint8))]
    src = TMP / "src_shard.safetensors"
    st_io.write(src, t, align=1)
    keep = [1, 4, 5]
    out, src_of = S5.prune_tensors(st_io.Reader(src), L, keep)
    dst = TMP / "dst_shard.safetensors"
    st_io.write(dst, out, align=1)
    info = S5.verify_layer(st_io.Reader(src), st_io.Reader(dst), L, keep, src_of)
    assert info["experts"] == 3
    d = st_io.Reader(dst)
    assert np.array_equal(d.get(f"layers.{L}.ffn.experts.1.w2.weight"), st_io.Reader(src).get(f"layers.{L}.ffn.experts.4.w2.weight"))
    assert np.array_equal(d.get(f"layers.{L}.ffn.gate.bias"), st_io.Reader(src).get(f"layers.{L}.ffn.gate.bias")[keep])
    bad = [(n, dt, (a.copy() if n != f"layers.{L}.ffn.experts.2.w1.weight" else a ^ 1)) for n, dt, a in out]
    st_io.write(dst, bad, align=1)
    try:
        S5.verify_layer(st_io.Reader(src), st_io.Reader(dst), L, keep, src_of)
        raise AssertionError("corrupted expert passed verification")
    except SystemExit:
        pass


# ----------------------------------------------------------------------------- s07 allocation + NVFP4 layer
@test
def s07_allocation_budget_and_protection():
    from ops import s07_quant as S7
    errs = {L: {"w13": {"Q2_K": 0.30, "Q3_K": 0.12, "Q4_K": 0.05}, "w2": {"Q2_K": 0.28, "Q3_K": 0.11, "Q4_K": 0.05},
                "mass": 1.0 + (L == 5) * 9.0} for L in range(10)}
    nb = 128
    budget = int(nb * 3 * S7.DIM * S7.INTER * G.bpw("MXFP4") / 8 * 10)
    r = S7.allocate(errs, 192, budget, [0, 1])
    assert r["bytes"] <= budget and r["avg_bpw"] <= 8 * budget / (192 * 3 * S7.DIM * S7.INTER * 10) + 1e-9
    assert r["alloc"]["0"]["w13"] != "Q2_K" and r["alloc"]["1"]["w2"] != "Q2_K"
    up = [L for L in r["alloc"] if r["alloc"][L] != {"w13": "Q2_K", "w2": "Q2_K"} and int(L) not in (0, 1)]
    assert "5" in up, r["alloc"]                       # the heavy layer is upgraded first
    try:
        S7.allocate(errs, 192, budget // 4, [0, 1])
        raise AssertionError("infeasible budget accepted")
    except SystemExit:
        pass
    assert abs(nb * G.bpw("MXFP4") / 192 - 2.8333) < 1e-3           # matched-bytes target of the A/B


@test
def s07_nvfp4_layer_lossless():
    from ops import s07_quant as S7
    rng = np.random.default_rng(9)
    L, E, keep = 4, 4, [0, 2, 3]
    t = []
    for e in range(E):
        for p in ("w1", "w2", "w3"):
            t.append((f"layers.{L}.ffn.experts.{e}.{p}.weight", "I8", rng.integers(-128, 128, (4, 32), dtype=np.int8)))
            t.append((f"layers.{L}.ffn.experts.{e}.{p}.scale", "F8_E8M0", rng.integers(118, 130, (4, 2), dtype=np.uint8)))
    t.append((f"layers.{L}.ffn.gate.bias", "F32", np.arange(E, dtype=np.float32)))
    src = TMP / "nv_src.safetensors"
    st_io.write(src, t, align=1)
    for d in ("s06", "s03"):
        (env.WORK / d / "bank").mkdir(parents=True, exist_ok=True)
    np.savez(env.WORK / "s06" / "bank" / f"L{L}.npz", keep=np.array(keep), n_tok=np.array([5, 0, 3]),
             amax_in=np.array([1., 2., 3.]), amax_mid_ungated=np.array([4., 5., 6.]))
    np.savez(env.WORK / "s03" / "bank" / f"L{L}.npz", amax_in=np.arange(E) * 10.0)
    out = {n: (dt, a) for n, dt, a in S7.nvfp4_layer(st_io.Reader(src), L, keep, 0.9)}
    assert f"layers.{L}.ffn.experts.2.w2.weight" in out and f"layers.{L}.ffn.experts.3.w2.weight" not in out
    assert np.array_equal(out[f"layers.{L}.ffn.gate.bias"][1], np.array([0., 2., 3.], np.float32))
    s_in = float(out[f"layers.{L}.ffn.experts.1.w1.input_scale"][1])
    assert abs(s_in - 20.0 / 2688.0) < 1e-9                      # unseen expert -> teacher amax fallback
    w, s16, g = (out[f"layers.{L}.ffn.experts.1.w2.{k}"][1] for k in ("weight", "weight_scale", "weight_scale_2"))
    rd = st_io.Reader(src)
    ref = G.src_dequant(rd.get(f"layers.{L}.ffn.experts.2.w2.weight"), rd.get(f"layers.{L}.ffn.experts.2.w2.scale")) * np.float32(0.9)
    assert np.array_equal(G.nvfp4_dequant(w, s16, float(g)), ref)


# ----------------------------------------------------------------------------- metrics + gates + cards
def _fake_run(rng, arms, n_rows=12, n=300):
    cdir, sdir, rdir = TMP / "corpus", TMP / "s03", TMP / "run"
    (cdir).mkdir(exist_ok=True); (sdir / "teacher").mkdir(parents=True, exist_ok=True)
    rows, T = [], n_rows * n
    for r in range(n_rows):
        b = ["general", "vision", "code"][r % 3]
        rows.append({"row": r, "bucket": b, "offset": r * n, "length": n, "heldout": True})
    with open(cdir / "rows.jsonl", "w") as fh:
        for x in rows:
            fh.write(json.dumps(x) + "\n")
    types = np.full(T, -1, np.int8)
    for r in range(1, n_rows, 3):
        types[r * n + 10:r * n + 50] = 0
    np.save(cdir / "tokens.npy", rng.integers(0, 1000, T).astype(np.uint32))
    np.save(cdir / "types.npy", types)
    np.save(cdir / "valid.npy", np.ones(T, np.uint8))
    tid = rng.integers(0, 1000, (T, 32)).astype(np.int32)
    lp = np.sort(np.log(rng.dirichlet(np.ones(32), T)), 1)[:, ::-1].astype(np.float32)
    np.save(sdir / "teacher" / "top32_ids.npy", tid)
    np.save(sdir / "teacher" / "top32_logprob.npy", lp)
    T_nll = rng.random(T).astype(np.float32) + 1.0
    np.save(sdir / "teacher" / "nll_next.npy", T_nll)
    for arm, (kl, agree) in arms.items():
        md = rdir / arm / "metrics"; md.mkdir(parents=True, exist_ok=True)
        for r in range(n_rows):
            g = slice(r * n, (r + 1) * n)
            top1 = np.where(rng.random(n) < agree, tid[g, 0], -1)
            np.savez(md / f"r{r:05d}.npz", kl32=np.abs(rng.normal(kl, kl / 3, n)).astype(np.float32), top1=top1,
                     nll=(T_nll[g] + kl / 4).astype(np.float32))
    return cdir, sdir, rdir, list(range(n_rows))


@test
def metrics_gates_end_to_end():
    rng = np.random.default_rng(10)
    cdir, sdir, rdir, rows = _fake_run(rng, {"control": (0.0005, 0.999), "good": (0.40, 0.86), "random0": (0.9, 0.7)})
    C, T = metrics.corpus(cdir), metrics.teacher(sdir)
    data = {a: metrics.load_arm(rdir, a, rows, C, T) for a in ("control", "good", "random0")}
    f = metrics.row_frame(C, T, 1)
    assert not f["scored"][9] and f["scored"][8] is not None and not f["scored"][-1]   # next token is image / end of row
    assert f["caption"][60] and not f["caption"][5]
    S = {a: metrics.summarize(d, n_draws=200, min_tokens=100) for a, d in data.items()}
    g, _ = gates.load()
    assert gates.control(S["control"], g)["pass"] and not gates.control(S["good"], g)["pass"]
    vs = metrics.compare(data["good"], data["random0"], n_draws=200)
    r = gates.pruned_floor(S["good"], g, vs_random=vs, drift_verdict="FLAT")
    assert r["pass"], r
    assert not gates.pruned_floor(S["random0"], g)["pass"]
    assert not gates.pruned_floor(S["good"], g, drift_verdict="BLOCK")["pass"]
    dec = gates.arm_decision(metrics.compare(data["good"], data["random0"], n_draws=200), g)
    assert dec["winner"] == "A"
    h = gates.heal(metrics.compare(data["good"], data["random0"], n_draws=200), g)
    assert h["adopt"]
    assert not gates.heal(metrics.compare(data["random0"], data["good"], n_draws=200), g)["adopt"]
    tie = {"d_kl32": {"mean": 0.0, "lo": -1, "hi": 1, "excludes_zero": False}, "top1": {"z": 0.0}}
    assert gates.arm_decision(tie, g)["winner"] == "B"


@test
def gates_prereg_lock():
    d = TMP / "s10lock"; d.mkdir()
    gates.lock(d)
    gates.lock(d)                                      # same file: fine
    lk = json.load(open(d / "prereg_lock.json")); lk["sha256"] = "0" * 64; json.dump(lk, open(d / "prereg_lock.json", "w"))
    try:
        gates.lock(d)
        raise AssertionError("changed gates accepted")
    except SystemExit:
        pass


@test
def cards_render_and_lint():
    assert cards.lint("kl32 was 0.65 and 12% better") == ["0.65", "12%"]
    assert cards.lint("V4.1 Q8_0 1017 top-32 {{ a.b | f4 }}") == []
    ctx = {"a": {"b": 0.123456, "c": [1, 2]}}
    assert cards.render("x {{ a.b | f3 }} {{ a.c.1 }}", ctx) == "x 0.123 2"
    for bad in ("{{ a.zz }}", "kl 0.5"):
        try:
            cards.render(bad, ctx)
            raise AssertionError(bad)
        except cards.CardError:
            pass
    parts = {p.name[: -len(".md.part")]: p.read_text() for p in (REPO / "ops" / "cards").glob("*.md.part")}
    for t in (REPO / "ops" / "cards").glob("*.md.tmpl"):
        assert cards.lint(cards.expand(t.read_text(), parts)) == [], t.name


if __name__ == "__main__":
    w = max(len(n) for n, _, _ in RESULTS)
    for n, st, msg in RESULTS:
        print(f"{st:4s}  {n:<{w}}  {msg}")
    bad = [r for r in RESULTS if r[1] == "FAIL"]
    print(f"\n{len(RESULTS) - len(bad)}/{len(RESULTS)} passed ({sum(r[1] == 'skip' for r in RESULTS)} skipped)")
    sys.exit(1 if bad else 0)
