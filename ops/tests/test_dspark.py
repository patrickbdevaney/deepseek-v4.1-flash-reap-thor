#!/usr/bin/env python3
"""s09 DSpark sidecar tests: the source comparison on tiny synthetic GGUFs, level picking, the gate. CPU only,
no checkpoint, no GPU.

    .venv/bin/python ops/tests/test_dspark.py
"""
from __future__ import annotations

import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from ops import s09_dspark as D  # noqa: E402

TESTS = []
GIB = 1 << 30
CFG = {"source_q8_max_rel_rmse": 0.001, "min_identical_frac": 0.5, "min_common_prefix_tokens": 32,
       "min_accept_rate": 0.1, "reserve_gib": 10}


def test(fn):
    TESTS.append(fn)
    return fn


def write(path: Path, *, layers=(37, 38, 39), q8_noise=0.0, f32_bump=False, extra_key=False):
    import gguf
    from gguf.quants import quantize
    rng = np.random.default_rng(0)
    w = gguf.GGUFWriter(str(path), "dflash")
    w.add_block_size(5)
    w.add_target_layers(list(layers))
    w.add_sample_from_anchor(True)
    if extra_key:
        w.add_uint32("dflash.indexer.head_count", 64)
    base = rng.standard_normal((8, 64)).astype(np.float32)
    q = base + q8_noise * rng.standard_normal(base.shape).astype(np.float32)
    w.add_tensor("blk.0.attn_q_a.weight", quantize(q, gguf.GGMLQuantizationType.Q8_0),
                 raw_dtype=gguf.GGMLQuantizationType.Q8_0)
    n = np.arange(16, dtype=np.float32)
    w.add_tensor("blk.0.attn_norm.weight", n + (1 if f32_bump else 0))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    return path


@test
def identical_files_pass():
    with tempfile.TemporaryDirectory() as d:
        a, b = write(Path(d) / "a.gguf"), write(Path(d) / "b.gguf")
        r = D.compare_files(a, b, 0.001)
        assert r["pass"] and r["byte_identical"] == r["tensors"] == 2, r


@test
def v4_layer_plus_one_fails():
    with tempfile.TemporaryDirectory() as d:
        a, b = write(Path(d) / "a.gguf", layers=(38, 39, 40)), write(Path(d) / "b.gguf")
        r = D.compare_files(a, b, 0.001)
        assert not r["pass"] and any("target_layers" in w for w in r["why"]), r


@test
def f32_difference_fails_q8_tolerance_passes():
    with tempfile.TemporaryDirectory() as d:
        r = D.compare_files(write(Path(d) / "a.gguf", f32_bump=True), write(Path(d) / "b.gguf"), 0.001)
        assert not r["pass"] and any("attn_norm" in w for w in r["why"]), r
        r = D.compare_files(write(Path(d) / "c.gguf", q8_noise=1e-5), write(Path(d) / "e.gguf"), 0.001)
        assert r["pass"], r
        r = D.compare_files(write(Path(d) / "f.gguf", q8_noise=0.1), write(Path(d) / "g.gguf"), 0.001)
        assert not r["pass"] and any("rel RMSE" in w for w in r["why"]), r


@test
def extra_keys_allowed():
    with tempfile.TemporaryDirectory() as d:
        r = D.compare_files(write(Path(d) / "a.gguf", extra_key=True), write(Path(d) / "b.gguf"), 0.001)
        assert r["pass"] and r["extra_keys"] == ["dflash.indexer.head_count"], r


@test
def pick_largest_level_that_fits():
    sizes = {"L2": 101 * GIB, "L3": 88 * GIB}
    assert D.pick(["L2", "L3"], 117.0, int(7.4 * GIB), 10, sizes) == "L3"
    assert D.pick(["L2", "L3"], 125.0, int(7.4 * GIB), 10, sizes) == "L2"
    assert D.pick(["L2"], 117.0, int(7.4 * GIB), 10, sizes) is None
    assert D.pick(["L3"], 117.0, int(7.4 * GIB), 10, sizes) == "L3"


def bench(per):
    return {"level": "L3", "level_gib": 88.0, "drafter_gib": 7.4, "n_predict": 256, "draft_n_max": 3, "ctx": 8192,
            "per_prompt": per}


def row(ident, pre, tp=10.0, ts=12.0, dn=100, da=60):
    return {"identical": ident, "common_prefix": pre, "tps_plain": tp, "tps_spec": ts, "draft_n": dn, "draft_accepted": da}


@test
def gate_passes_and_reports_speed():
    r = D.evaluate({"pass": True, "why": []}, bench([row(True, 256), row(False, 200)]), CFG)
    assert r["pass"], r
    assert abs(r["speedup"] - 1.2) < 1e-9 and abs(r["accept_rate"] - 0.6) < 1e-9


@test
def gate_slower_still_passes():
    r = D.evaluate({"pass": True, "why": []}, bench([row(True, 256, ts=8.0)] * 2), CFG)
    assert r["pass"] and r["speedup"] < 1, r


@test
def gate_failures():
    ok = {"pass": True, "why": []}
    assert not D.evaluate(ok, bench([row(True, 256), row(False, 5)]), CFG)["pass"]          # early divergence
    assert not D.evaluate(ok, bench([row(False, 100)] * 3 + [row(True, 256)]), CFG)["pass"]  # mostly different
    assert not D.evaluate(ok, bench([row(True, 256, da=1)] * 2), CFG)["pass"]               # drafter rejected
    assert not D.evaluate({"pass": False, "why": ["x"]}, bench([row(True, 256)] * 2), CFG)["pass"]
    r = D.evaluate(ok, None, CFG)
    assert not r["pass"] and "not benched" in r["why"][0]


@test
def card_section_pass_and_fail():
    from ops import s12_publish as S12
    ok = {"pass": True, "why": []}
    g = D.evaluate(ok, bench([row(True, 256), row(False, 200)]), CFG)
    t = S12.dspark_section(g, D.FILE, "u/r")
    assert D.FILE in t and "1.20x" in t and "60.0%" in t and "net slowdown" not in t, t
    g = D.evaluate(ok, bench([row(True, 256, ts=8.0)] * 2), CFG)
    assert "net slowdown" in S12.dspark_section(g, D.FILE, "u/r")
    f = S12.dspark_section(D.evaluate(ok, None, CFG), D.FILE, "u/r")
    assert "No DSpark drafter is published" in f and "not benched" in f, f
    assert S12.dspark_section(None, D.FILE, "u/r") == ""


@test
def common_prefix_counts():
    assert D.common_prefix([1, 2, 3], [1, 2, 4]) == 2 and D.common_prefix([1], [1]) == 1


if __name__ == "__main__":
    ok = fail = 0
    for fn in TESTS:
        try:
            fn()
            print(f"ok    {fn.__name__}")
            ok += 1
        except Exception:
            print(f"FAIL  {fn.__name__}")
            traceback.print_exc()
            fail += 1
    print(f"\n{ok}/{ok + fail} passed")
    sys.exit(1 if fail else 0)
