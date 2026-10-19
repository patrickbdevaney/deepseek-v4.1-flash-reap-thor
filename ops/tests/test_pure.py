#!/usr/bin/env python3
"""Pure-python / numpy tests for the ops library. No torch, no model, no GPU: safe beside any job.

    python3 ops/tests/test_pure.py
"""
from __future__ import annotations

import io
import itertools
import json
import os
import struct
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

TMP = Path(tempfile.mkdtemp(prefix="v41ops_"))
os.environ["V41_WORK"] = str(TMP / "work")          # keep env.py's mkdir out of ~/models
sys.modules["torch"] = None                           # any torch import in these paths is a bug
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ops import corpus_spec as spec                   # noqa: E402
from ops.lib import drift, engram_rows, env, hope, imatrix, imatrix_verify  # noqa: E402

RESULTS = []


def test(fn):
    try:
        fn()
        RESULTS.append((fn.__name__, "ok", ""))
    except Exception as e:  # report every failure, keep going
        RESULTS.append((fn.__name__, "FAIL", f"{type(e).__name__}: {e}"))
    return fn


# ----------------------------------------------------------------------------- HOPE
@test
def hope_matches_bruteforce_small():
    rng = np.random.default_rng(1)
    worst = 0
    for trial in range(20):
        E, n = 8, 4
        A = rng.random((E, E)); F = (A + A.T) / 2
        best = min(itertools.combinations(range(E), n), key=lambda P: F[np.ix_(P, P)].sum())
        got = hope.solve_prune_set(F, n, restarts=6, seed=trial)
        gap = F[np.ix_(got, got)].sum() - F[np.ix_(best, best)].sum()
        worst = max(worst, gap)
    assert worst < 1e-9, f"solver missed the brute-force optimum by {worst}"


@test
def hope_diagonal_is_reap():
    d = np.random.default_rng(2).random(384)
    got = hope.solve_prune_set(np.diag(d), 192)
    assert np.array_equal(got, np.sort(np.argsort(d)[:192]))


@test
def hope_protect_respected():
    rng = np.random.default_rng(3)
    A = rng.random((32, 32)); F = (A + A.T) / 2
    prot = [int(i) for i in np.argsort(np.diag(F))[:5]]   # the cheapest experts, i.e. first to go
    got = hope.solve_prune_set(F, 16, protect=prot)
    assert not set(prot) & set(got.tolist())


# ----------------------------------------------------------------------------- imatrix
@test
def imatrix_roundtrip_and_verify():
    E, C, Cm = 6, 64, 16
    rng = np.random.default_rng(4)
    bank = {"x2_in": rng.random((E, C)) * 10, "x2_mid": rng.random((E, Cm)), "n_tok": np.array([5, 0, 3, 7, 1, 2.]),
            "x2_in_sh": rng.random(C), "x2_mid_sh": rng.random(Cm), "n_tok_sh": np.array([18.])}
    keep = np.array([3, 0, 1])
    ent = imatrix.layer_entries(7, bank, keep, ncall=11)
    p = TMP / "t.dat"
    imatrix.write_dat(p, ent, "v41-calib", 11)
    back, last, ds = imatrix.read_dat(p)
    assert last == 11 and ds == "v41-calib" and list(back) == list(ent)
    gate = back["blk.7.ffn_gate_exps.weight"][1].reshape(3, C)
    np.testing.assert_allclose(gate[0], (bank["x2_in"][3] / 7 * 11).astype(np.float32), rtol=1e-6)
    assert not gate[2].any(), "an expert with zero tokens must be written as zeros"
    with redirect_stdout(io.StringIO()):
        assert imatrix_verify.main(str(p)) == 0
    p.write_bytes(p.read_bytes()[:-3])
    with redirect_stdout(io.StringIO()):
        assert imatrix_verify.main(str(p)) != 0, "truncated file must fail verification"


# ----------------------------------------------------------------------------- Engram rows
def _write_safetensors(path: Path, tensors: dict[str, tuple[str, np.ndarray]]):
    hdr, blobs, off = {}, [], 0
    for name, (dt, arr) in tensors.items():
        b = arr.tobytes()
        hdr[name] = {"dtype": dt, "shape": list(arr.shape), "data_offsets": [off, off + len(b)]}
        blobs.append(b); off += len(b)
    h = json.dumps(hdr).encode()
    h += b" " * (-len(h) % 8)
    path.write_bytes(struct.pack("<Q", len(h)) + h + b"".join(blobs))


@test
def engram_gather_paths_agree():
    rng = np.random.default_rng(5)
    n, w = 1000, 256
    vals = rng.integers(0, 256, (n, w), dtype=np.uint8)
    scl = rng.integers(0, 256, (n, 8), dtype=np.uint8)
    p = TMP / "eng.safetensors"
    _write_safetensors(p, {"layers.1.engram.embed.weight": ("F8_E4M3", vals),
                           "layers.1.engram.embed.scale": ("F8_E8M0", scl)})
    t = engram_rows.EngramTables(p, 1)
    ids = rng.integers(0, n, (3, 50, 24))
    v, s = t.gather(ids)
    assert np.array_equal(v, vals[ids]) and np.array_equal(s, scl[ids])
    # windowed sweep, interrupted after window 2 and resumed
    out = np.zeros((ids.size, w), np.uint8)
    seen = []
    class Stop(Exception):
        pass
    def hook(k):
        seen.append(k)
        if k == 2:
            raise Stop
    try:
        engram_rows.gather_windowed(t.values, ids, out, window_rows=128, on_window=hook)
    except Stop:
        pass
    engram_rows.gather_windowed(t.values, ids, out, window_rows=128, start_window=3)
    assert np.array_equal(out, vals[ids.reshape(-1)])
    try:
        engram_rows.gather(t.values, np.array([n]))
        raise AssertionError("out-of-range id accepted")
    except IndexError:
        pass


@test
def engram_bytes_per_token_matches_shard_size():
    cfg = json.load(open(env.INFERENCE / "config.json"))
    rows = cfg["engram_num_embeddings"][0]
    per_row = cfg["engram_head_dim"] + cfg["engram_head_dim"] // 32       # fp8 + E8M0 per 32
    assert abs(rows * per_row / 1e9 - 101.4) < 0.2, rows * per_row / 1e9
    assert 24 * per_row == 6336


# ----------------------------------------------------------------------------- ledger
@test
def ledger_atomic_resume():
    p = TMP / "led.json"
    L = env.Ledger(p); L.mark("a", n=1); L.finish("b", rows=3)
    L2 = env.Ledger(p)
    assert not L2.done("a") and L2.done("b") and L2.get("b")["rows"] == 3
    assert not list(TMP.glob("led.json.tmp"))


# ----------------------------------------------------------------------------- corpus spec
@test
def corpus_mix_and_order():
    t = spec.TOKEN_TARGET
    assert abs(sum(t.values()) - 1) < 1e-9 and min(t.values()) >= 0.10 and abs(t["vision"] - 0.14) < 1e-9
    assert {"multilingual", "longctx"} <= set(t)
    rows = spec.rows_per_bucket()
    order = spec.interleave(rows)
    assert sorted(order) == sorted(b for b, c in rows.items() for _ in range(c))
    n = len(order)
    for k in range(20, n + 1, 20):                     # every prefix stays near the mix
        for b in spec.BUCKETS:
            assert abs(order[:k].count(b) - k * rows[b] / n) <= 1.0 + 1e-9, (k, b)


@test
def corpus_provenance_rules():
    all_src = [s for v in spec.SOURCES.values() for s in v] + spec.VISION_SOURCES + spec.LONG_SOURCES
    for s in all_src:
        ok, _ = spec.usable(s, allow_unverified=True)
        assert not (ok and "claude" in s["generator"].lower())
        if ok:
            assert spec.licence_ok(s["licence"]) is not False, s["hf_id"]
    for h in spec.EXCLUDED["claude_generated"]:
        assert all(s["hf_id"] != h or not spec.usable(s, True)[0] for s in all_src)
    fake = dict(all_src[0], generator="Claude 3.7 Sonnet")
    assert not spec.usable(fake, True)[0]
    r1 = next(s for s in all_src if "R1" in s["generator"])
    assert spec.usable(r1, False, "no-claude")[0] and not spec.usable(r1, False, "human-or-self")[0]
    selfgen = next(s for s in all_src if s["generator"] == spec.SELF_GEN)
    assert spec.usable(selfgen, False, "human-or-self")[0]
    assert spec.licence_ok("cc-by-nc-sa-4.0") is False and spec.licence_ok("cc-by-sa-3.0") is False
    assert spec.licence_ok("apache-2.0") and spec.licence_ok("other") is None


@test
def s02_dry_run_lists_everything():
    from ops import s02_calib_corpus as s02
    buf = io.StringIO()
    with redirect_stdout(buf):
        s02.dry_run(False, "no-claude", spec.TOTAL_TOKENS)
    out = buf.getvalue()
    assert "longctx" in out and "multilingual" in out and "SKIP" in out and "USE" in out
    assert "SWE-smith-trajectories" in out                 # listed as excluded, visibly


# ----------------------------------------------------------------------------- s03 topology
@test
def s03_cross_layer_state_topology():
    from ops import s03_saliency as s03
    assert s03.sources_for(0) == {} and s03.sources_for(1) == {}
    assert s03.sources_for(3) == {"kv": 2, "topk": 2}
    assert s03.sources_for(8) == {}                        # source of everything it reads
    assert s03.sources_for(24) == {"kv": 20, "ik": 20, "cand": 20}
    assert s03.sources_for(37) == {"kv": 20, "topk": 36}
    assert s03.published_by(20) == ["kv", "ik", "topk", "cand"]
    assert s03.last_consumer("kv", 20) == 39 and s03.last_consumer("ik", 20) == 36
    assert s03.last_consumer("topk", 2) == 7 and s03.last_consumer("cand", 20) == 36


# ----------------------------------------------------------------------------- drift
def _rows(n_rows, n_tok, E, K, rng, rising=None, sink=None):
    rows = []
    for _ in range(n_rows):
        idx = np.stack([rng.choice(E, K, replace=False) for _ in range(n_tok)])
        w = rng.random((n_tok, K)) + 0.1
        pos = np.arange(n_tok)
        if rising is not None:   # expert `rising` takes slot 0 with probability growing in log-position
            p = (np.log2(pos + 2) / np.log2(n_tok + 2)) ** 3
            on = rng.random(n_tok) < 0.6 * p
            idx[on, 0] = rising
            idx[on, 1:] = np.where(idx[on, 1:] == rising, (rising + 1) % E, idx[on, 1:])
        if sink is not None:
            idx[:4, 0] = sink; w[:4, 0] = 5.0
        rows.append((idx, w))
    return rows


@test
def drift_flat_mask_is_flat():
    rng = np.random.default_rng(6)
    rows = _rows(12, 4096, 32, 4, rng)
    pruned = np.zeros(32, bool); pruned[:8] = True
    rep = drift.layer_report(rows, pruned, n_draws=200)
    assert rep["mass_slope"]["lo"] < 0 < rep["mass_slope"]["hi"] or abs(rep["mass_slope"]["est"]) < 2e-3
    assert drift.mask_verdict({0: rep})["verdict"] == "FLAT"


@test
def drift_blocks_rising_and_sink_experts():
    rng = np.random.default_rng(7)
    rows = _rows(12, 4096, 32, 4, rng, rising=5, sink=9)
    pruned = np.zeros(32, bool); pruned[[5, 9, 20]] = True
    rep = drift.layer_report(rows, pruned, n_draws=200)
    assert 5 in rep["block_rising"], rep["rising_slope"]
    assert 9 in rep["block_sink"] and 20 not in rep["block_sink"] + rep["block_rising"]
    assert drift.mask_verdict({0: rep})["verdict"] == "BLOCK"
    keep_safe = np.zeros(32, bool); keep_safe[[20, 21]] = True
    rep2 = drift.layer_report(rows, keep_safe, n_draws=200)
    assert not rep2["block_sink"] and not rep2["block_rising"]


@test
def drift_no_false_rising_at_v41_scale():
    rng = np.random.default_rng(0)
    rows = [(np.argsort(rng.random((8192, 384)), 1)[:, :6], rng.random((8192, 6)) + 0.1) for _ in range(16)]
    rep = drift.layer_report(rows, np.ones(384, bool), n_draws=300)
    assert not rep["block_rising"] and not rep["block_sink"], (rep["block_rising"], rep["block_sink"])


@test
def drift_helpers():
    assert drift.pos_bin(np.array([0, 7, 8, 1023, 1024, 8191, 8192, 131071, 131072])).tolist() == [0, 0, 1, 3, 4, 6, 7, 10, 11]
    R = np.array([[0.0, 1, 2, 3], [5, 6, 7, 8]]) + 0
    x = np.log2(np.sqrt(np.maximum(drift.BIN_EDGES[:4], 1) * drift.BIN_EDGES[1:5]))
    slope = drift.within_row_slope(np.stack([np.r_[R[0], [np.nan] * 8], np.r_[R[1], [np.nan] * 8]]))
    assert abs(slope - np.polyfit(x, R[0], 1)[0]) < 1e-9
    assert drift.sign_test(np.ones(10)) < 0.01 and drift.sign_test(np.array([1, -1, 1, -1.])) == 1.0
    sc = drift.super_expert_candidates(np.r_[np.ones(999), 100.0].reshape(1, -1))
    assert sc.sum() == 1 and sc[0, -1]


# ----------------------------------------------------------------------------- tokenizer
@test
def tokenizer_shim_vocab():
    try:
        import tokenizers  # noqa: F401
    except ImportError:
        RESULTS.append(("tokenizer_shim_vocab", "skip", "tokenizers not installed"))
        return
    from ops.lib.tok import Tok
    t = Tok(env.SRC)
    cfg = json.load(open(env.INFERENCE / "config.json"))
    assert len(t) <= cfg["vocab_size"], (len(t), cfg["vocab_size"])
    assert t.convert_tokens_to_ids("<｜deepseek_image｜>") in (None, cfg["image_token_id"])
    bos = t.encode("<｜begin▁of▁sentence｜>hello")
    assert len(bos) >= 2 and t.decode(bos[:1]) == "<｜begin▁of▁sentence｜>"


if __name__ == "__main__":
    w = max(len(n) for n, _, _ in RESULTS)
    for n, st, msg in RESULTS:
        print(f"{st:4s}  {n:<{w}}  {msg}")
    bad = [r for r in RESULTS if r[1] == "FAIL"]
    print(f"\n{len(RESULTS) - len(bad)}/{len(RESULTS)} passed")
    sys.exit(1 if bad else 0)
