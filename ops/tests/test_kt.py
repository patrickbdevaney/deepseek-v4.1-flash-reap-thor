#!/usr/bin/env python3
"""CPU checks of the KT2/KT3 trellis format (ops/lib/kt.py). numpy for the format, torch on CPU for
the LDLQ driver (CUDA hidden before torch loads). No model, no GPU.

    CUDA_VISIBLE_DEVICES= .venv/bin/python ops/tests/test_kt.py

Prints the synthetic MSE table (KT vs ggml K-quants, Gaussian and heavy-tailed rows) that the
pre-registration and thor_format.md quote. Thresholds are loose sanity bounds, not the gate: the
gate is s10's paired KL against the teacher at matched bytes.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
TMP = Path(tempfile.mkdtemp(prefix="v41kt_"))
os.environ.setdefault("V41_WORK", str(TMP / "work"))
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from ops.lib import ggml_quants as G, kt, thor_format as TF  # noqa: E402

RESULTS = []
TABLE = []


def test(fn):
    try:
        r = fn()
        RESULTS.append((fn.__name__, "skip" if r == "skip" else "ok", ""))
    except Exception as e:
        import traceback
        RESULTS.append((fn.__name__, "FAIL", f"{type(e).__name__}: {e} @ {traceback.extract_tb(e.__traceback__)[-1].lineno}"))
    return fn


rng = np.random.default_rng(1017)


@test
def codebook_is_3inst():
    c = kt.codebook()
    assert c.dtype == np.float32 and c.shape == (4096,)
    # spot values computed by hand from the 3INST definition
    for s in (0, 1, 4095):
        x = (s * 89226354 + 64248484) & 0xFFFFFFFF
        m = (x & 0x8FFF8FFF) ^ 0x3B603B60
        v = np.float32(np.uint16(m & 0xFFFF).view(np.float16)) + np.float32(np.uint16(m >> 16).view(np.float16))
        assert c[s] == v, (s, c[s], v)
    assert 1.1 < kt.CB_RMS < 1.4 and abs(float(c.mean())) < 0.02


@test
def stream_roundtrip():
    for b in (2, 3):
        X = rng.standard_normal((40, kt.T)).astype(np.float32)
        st = kt.viterbi(X, np.ones_like(X), b)
        w = kt.states_to_words(st, b)
        assert w.shape == (40, 4 * b) and w.dtype == np.uint32
        assert np.array_equal(kt.words_to_windows(w, b), st), "states are not a valid circular stream"


@test
def viterbi_beats_greedy_and_meets_rate():
    for b, bound in ((2, 0.075), (3, 0.021)):
        X = rng.standard_normal((256, kt.T)).astype(np.float32) * kt.CB_RMS
        st = kt.viterbi(X, np.ones_like(X), b)
        mse = float(((X - kt.CB[st]) ** 2).mean()) / kt.CB_RMS ** 2
        assert mse < bound, (b, mse)
        # any valid stream is a feasible point: random streams must never beat the Viterbi path
        rnd = rng.integers(0, 2 ** 32, (256, 4 * b), dtype=np.uint64).astype(np.uint32)
        mse_r = float(((X - kt.CB[kt.words_to_windows(rnd, b)]) ** 2).mean()) / kt.CB_RMS ** 2
        assert mse < mse_r


@test
def viterbi_is_optimal_on_its_trellis():
    # perturb the optimum: re-run with one target nudged far away must not lower the original cost
    b = 2
    X = rng.standard_normal((8, kt.T)).astype(np.float32)
    W = np.ones_like(X)
    st = kt.viterbi(X, W, b)
    cost = ((X - kt.CB[st]) ** 2).sum(1)
    X2 = X.copy()
    X2[:, 50] += 5
    st2 = kt.viterbi(X2, W, b)
    cost2 = ((X - kt.CB[st2]) ** 2).sum(1)
    assert (cost <= cost2 + 1e-4).all()


@test
def rht_is_orthonormal_and_matches_matmul():
    import torch
    v = rng.standard_normal((5, 768)).astype(np.float32)
    s = kt.signs("L3.w13", 768)
    r = kt.rht_rows(v, s)
    assert np.allclose(kt.unrht_rows(r, s), v, atol=1e-5)
    assert np.allclose((r ** 2).sum(1), (v ** 2).sum(1), rtol=1e-5)
    rt = kt.rotate_t(torch.from_numpy(v), torch.from_numpy(s), kt.hadamard256()).numpy()
    assert np.allclose(rt, r, atol=1e-5)
    assert np.array_equal(kt.unpack_signs(kt.pack_signs(s), 768), s)
    assert kt.pack_signs(s).shape == (96,)


@test
def encode_decode_roundtrip():
    for q in ("KT2", "KT3"):
        b = kt.bits_of(q)
        W = (rng.standard_normal((16, 2304)) * 0.02).astype(np.float32)
        s = kt.signs("L0.w2", 2304)
        qw, sc = kt.quantize_rows(W, q, s)
        assert qw.dtype == np.uint8 and qw.shape == (16, kt.row_bytes(q, 2304)) and sc.shape == (16, 18)
        rot = kt.dequant(q, qw, sc, 2304, None)
        words = qw.reshape(-1, 4 * b * 4).view(np.uint32)
        ref = (kt.CB[kt.words_to_windows(words, b)] * sc.reshape(-1).astype(np.float32)[:, None]).reshape(16, 2304)
        assert np.array_equal(rot, ref.astype(np.float32))
        full = kt.dequant(q, qw, sc, 2304, s)
        rel = float(((W - full) ** 2).sum() / (W ** 2).sum())
        assert rel < (0.08 if b == 2 else 0.025), (q, rel)


@test
def byte_counts():
    assert kt.bpw("KT2") == 2.125 and kt.bpw("KT3") == 3.125
    assert kt.row_bytes("KT2", 5120) == 40 * 32 and kt.row_bytes("KT3", 2304) == 18 * 48
    n, I, D = 192, 2304, 5120
    for q in ("KT2", "KT3"):
        b13 = 2 * n * I * (kt.row_bytes(q, D) + kt.scale_bytes(D))
        b2 = n * D * (kt.row_bytes(q, I) + kt.scale_bytes(I))
        bpw = (b13 + b2) * 8 / (3 * n * I * D)
        assert abs(bpw - kt.bpw(q)) < 1e-9, (q, bpw)
    # arm-B matched budget: 128 x MXFP4 (4.25 bpw) over 192 experts = 2.8333 bpw, between KT2 and KT3
    target = 128 * 4.25 / 192
    assert kt.bpw("KT2") < target < kt.bpw("KT3")


def _rel(W, Wq):
    return float(((W - Wq) ** 2).sum() / (W ** 2).sum())


def _kq(W, t):
    return G.dequant(t, G.quantize_rtn(t, W)).reshape(W.shape)


def _kt(W, q):
    s = kt.signs("bench", W.shape[1])
    qw, sc = kt.quantize_rows(W, q, s)
    return kt.dequant(q, qw, sc, W.shape[1], s)


@test
def mse_vs_kquants():
    R, C = 24, 2304
    dists = {
        "gaussian": lambda: rng.standard_normal((R, C)),
        "student-t3": lambda: rng.standard_t(3, (R, C)),
        "t3+outlier-cols": lambda: rng.standard_t(3, (R, C)) * np.where(rng.random(C) < 0.01, 8.0, 1.0)[None, :],
    }
    target = 128 * 4.25 / 192
    for name, gen in dists.items():
        W = (gen() * rng.uniform(0.005, 0.03, (R, 1))).astype(np.float32)
        e = {t: _rel(W, _kq(W, t)) for t in ("Q2_K", "Q3_K")}
        e.update({q: _rel(W, _kt(W, q)) for q in ("KT2", "KT3")})
        f_kq = (target - G.bpw("Q2_K")) / (G.bpw("Q3_K") - G.bpw("Q2_K"))
        f_kt = (target - kt.bpw("KT2")) / (kt.bpw("KT3") - kt.bpw("KT2"))
        mix_kq = (1 - f_kq) * e["Q2_K"] + f_kq * e["Q3_K"]
        mix_kt = (1 - f_kt) * e["KT2"] + f_kt * e["KT3"]
        TABLE.append((name, e, mix_kq, mix_kt, target))
        assert e["KT2"] < e["Q2_K"], (name, e)
        assert mix_kt < mix_kq, (name, mix_kt, mix_kq)


@test
def ldlq_beats_no_feedback_on_correlated_inputs():
    import torch
    B, R, C, N = 2, 6, 512, 3072
    mix = rng.standard_normal((C, C)) * 0.15 + np.eye(C)
    X = (rng.standard_normal((B, N, C)) @ mix).astype(np.float32)
    X[:, :, 7] *= 6.0
    H = np.einsum("bnc,bnd->bcd", X, X)
    W = (rng.standard_normal((B, R, C)) * 0.02).astype(np.float32)
    s = kt.signs("ldlq", C)
    for q in ("KT2", "KT3"):
        qw, sc = kt.quantize_ldlq(torch.from_numpy(W.copy()), torch.from_numpy(H), q, s)
        assert qw.shape == (B, R, kt.row_bytes(q, C)) and sc.shape == (B, R, C // kt.T)
        Wl = kt.dequant(q, qw, sc, C, s)
        Wn = np.stack([kt.dequant(q, *kt.quantize_rows(W[i], q, s), C, s) for i in range(B)])
        err = lambda Wq: float(sum(np.einsum("rc,cd,rd->", W[i] - Wq[i], H[i], W[i] - Wq[i]) for i in range(B)))
        el, en = err(Wl), err(Wn)
        TABLE.append((f"LDLQ {q}", {"H-err LDLQ": el, "H-err no-feedback": en}, None, None, None))
        assert el < en, (q, el, en)


@test
def thor_format_dequant_stack_kt():
    n, out, inn = 3, 8, 512
    s = kt.signs("L5.w2", inn)
    W = (rng.standard_normal((n, out, inn)) * 0.02).astype(np.float32)
    qs = [kt.quantize_rows(W[i], "KT3", s) for i in range(n)]
    qw = np.stack([a for a, _ in qs])
    sc = np.stack([b for _, b in qs])
    full = TF.dequant_stack("KT3", qw, out, inn, sc, sign=kt.pack_signs(s))
    one = TF.dequant_stack("KT3", qw, out, inn, sc, idx=1, sign=kt.pack_signs(s))
    assert full.shape == (n, out, inn) and np.array_equal(one, full[1])
    assert _rel(W, full) < 0.025


@test
def s07_trellis_allocation_matched_bytes():
    from ops import s07_quant as S7
    n = 192
    errs = {L: {"mass": 1.0 + (L % 5), "w13": {"KT2": 0.073, "KT3": 0.0195}, "w2": {"KT2": 0.074, "KT3": 0.0196}}
            for L in range(S7.N_LAYERS)}
    kq_budget = int(n * 3 * S7.DIM * S7.INTER * (128 * 4.25 / 192) / 8 * S7.N_LAYERS)
    r = S7.allocate(errs, n, kq_budget, [0, 1, 14], ladder=S7.KT_LADDER)
    assert r["bytes"] <= kq_budget and r["ladder"] == ["KT2", "KT3"]
    assert all(r["alloc"][str(L)][p] == "KT3" for L in (0, 1, 14) for p in ("w13", "w2"))
    assert kq_budget - r["bytes"] < S7.expert_bytes("KT3", "w13", n), "greedy left a whole upgrade unspent"
    assert abs(r["avg_bpw"] - 128 * 4.25 / 192) < 0.05
    # the K-quant ladder is unchanged by the generalisation
    rk = S7.allocate({L: {"mass": 1.0, "w13": {t: 0.1 / (i + 1) for i, t in enumerate(S7.LADDER)},
                          "w2": {t: 0.1 / (i + 1) for i, t in enumerate(S7.LADDER)}} for L in range(4)},
                     n, 10 ** 12, [0])
    assert rk["alloc"]["0"]["w13"] in ("Q3_K", "Q4_K") and rk["ladder"] == S7.LADDER


@test
def trellis_gate_rules():
    from ops.lib import gates
    g = gates.load()[0]
    bench = {"rows": [{"format": f, "shape": sh, "M": 1, "us": us} for f, sh, us in
                      (("Q2_K", "gate|up", 100.0), ("Q2_K", "down", 50.0), ("Q3_K", "gate|up", 120.0), ("Q3_K", "down", 60.0),
                       ("KT2", "gate|up", 105.0), ("KT2", "down", 52.0), ("KT3", "gate|up", 125.0), ("KT3", "down", 64.0))]}
    t_kq = gates.gemv_time({"0": {"w13": "Q2_K", "w2": "Q3_K"}, "1": {"w13": "Q2_K", "w2": "Q2_K"}}, bench)
    t_kt = gates.gemv_time({"0": {"w13": "KT3", "w2": "KT3"}, "1": {"w13": "KT2", "w2": "KT3"}}, bench)
    assert t_kq == 100 + 60 + 100 + 50 and t_kt == 125 + 64 + 105 + 64
    win = {"d_kl32": {"mean": -0.05, "lo": -0.07, "hi": -0.03, "excludes_zero": True}, "top1": {"z": 3.1}}
    ok = gates.trellis(win, g, kt_bytes=99, kq_bytes=100, t_kt=t_kt, t_kq=t_kq)
    assert ok["adopt"] == (t_kt / t_kq <= g["trellis"]["max_gemv_time_ratio"]), ok
    assert not gates.trellis(win, g, kt_bytes=101, kq_bytes=100, t_kt=1, t_kq=1)["adopt"]
    assert not gates.trellis(win, g, kt_bytes=99, kq_bytes=100, t_kt=None, t_kq=None)["adopt"]
    assert not gates.trellis(win, g, kt_bytes=99, kq_bytes=100, t_kt=2.0, t_kq=1.0)["adopt"]
    tie = {"d_kl32": {"mean": -0.01, "lo": -0.03, "hi": 0.01, "excludes_zero": False}, "top1": {"z": 2.0}}
    assert not gates.trellis(tie, g, kt_bytes=99, kq_bytes=100, t_kt=1, t_kq=1)["adopt"]
    lose_t1 = {"d_kl32": {"mean": -0.05, "lo": -0.07, "hi": -0.03, "excludes_zero": True}, "top1": {"z": -0.5}}
    assert not gates.trellis(lose_t1, g, kt_bytes=99, kq_bytes=100, t_kt=1, t_kq=1)["adopt"]


@test
def dequant_t_matches_numpy():
    import torch
    for q in ("KT2", "KT3"):
        b = kt.bits_of(q)
        qw = rng.integers(0, 256, (3, 5, 2304 // kt.T * 16 * b), dtype=np.uint8)
        sc = (rng.uniform(0.001, 0.02, (3, 5, 2304 // kt.T))).astype(np.float16)
        s = kt.signs("t", 2304)
        a = kt.dequant(q, qw, sc, 2304, s)
        t = kt.dequant_t(q, qw, sc, 2304, s, device="cpu").numpy()
        assert np.array_equal(a, t), (q, float(np.abs(a - t).max()))
        assert np.array_equal(kt.dequant(q, qw, sc, 2304, None), kt.dequant_t(q, qw, sc, 2304, None, device="cpu").numpy())


if __name__ == "__main__":
    w = max(len(n) for n, _, _ in RESULTS)
    for n, st, msg in RESULTS:
        print(f"{st:4s}  {n:<{w}}  {msg}")
    print("\nrelative MSE sum(err^2)/sum(w^2), RTN-style (no Hessian), 24 x 2304 rows; mixes at the arm-B budget")
    for name, e, mkq, mkt, tgt in TABLE:
        cells = "  ".join(f"{k} {v:.5f}" for k, v in e.items())
        extra = "" if mkq is None else f"  | @{tgt:.4f} bpw: K-quant mix {mkq:.5f}  KT mix {mkt:.5f}  (KT/KQ {mkt / mkq:.3f})"
        print(f"  {name:<16s} {cells}{extra}")
    sys.exit(1 if any(st == "FAIL" for _, st, _ in RESULTS) else 0)
