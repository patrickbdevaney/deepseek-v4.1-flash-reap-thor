#!/usr/bin/env python3
"""CPU checks for the s07 audit fixes (ops/logs/audit/s07.md). numpy only, synthetic tensors, temp dirs.

    .venv/bin/python ops/tests/test_s07_format.py

  - BLOCKER-1  thor.json exists right after Build(create=True), and a child process can open it
  - BLOCKER-2  a parent that re-opens the Build after the units keeps their layer entries
  - CORRUPT-1  expert_bytes counts decoder-layer routed experts only (not mtp.*)
  - CORRUPT-2  NVFP4 w1/w3 share weight_scale_2, and the repack still dequantizes bit-exactly
  - RISK-2     disk preflight fails fast; thor-b parks the losing A build (verified, chain2 layout)
  - NIT        resuming a build with a different recipe is refused
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

TMP = Path(tempfile.mkdtemp(prefix="v41s07_"))
os.environ["V41_WORK"] = str(TMP / "work")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from ops.lib import env, ggml_quants as G, keepset, orchestrate, st_io, thor_format as TF  # noqa: E402

RESULTS = []


def test(fn):
    try:
        fn()
        RESULTS.append((fn.__name__, "ok", ""))
    except BaseException as e:      # SystemExit included: a fix that exits where it should not is a failure
        import traceback
        RESULTS.append((fn.__name__, "FAIL", f"{type(e).__name__}: {e} @ {traceback.extract_tb(e.__traceback__)[-1].lineno}"))
    return fn


def expect_exit(f, *a, **k) -> str:
    try:
        f(*a, **k)
    except SystemExit as e:
        return str(e)
    raise AssertionError(f"{f.__name__} did not raise SystemExit")


def _layer_items(L: int, n: int = 2) -> list[dict]:
    rng = np.random.default_rng(L)
    return [{"name": TF.expert_name(L, "w1"), "dtype": "U8", "data": rng.integers(0, 256, (n, 4, 110), dtype=np.uint8),
             "qtype": "Q3_K", "shape": [n, 4, 256]},
            {"name": f"layers.{L}.ffn.gate.bias", "dtype": "F32", "data": np.zeros(n, np.float32)}]


# ----------------------------------------------------------------------------- BLOCKER-1 / BLOCKER-2
@test
def create_persists_manifest_for_child_units():
    root = TMP / "b1"
    TF.Build(root, create=True, keepset_sha256="a" * 64, n_routed_experts=2, arm="A")
    assert (root / "thor.json").exists(), "thor.json not written at create"
    child = ("import sys; sys.path.insert(0, sys.argv[1]); from ops.lib import thor_format as TF; "
             "TF.Build(sys.argv[2], keepset_sha256='a' * 64, n_routed_experts=2)")
    r = subprocess.run([sys.executable, "-c", child, str(REPO), str(root)], capture_output=True, text=True,
                       env={**os.environ})
    assert r.returncode == 0, r.stderr[-500:]


@test
def parent_reopen_keeps_unit_layers():
    root = TMP / "b2"
    parent = TF.Build(root, create=True, keepset_sha256="a" * 64, n_routed_experts=2, arm="A")
    unit = TF.Build(root, keepset_sha256="a" * 64, n_routed_experts=2)       # what a layer unit does
    unit.write_file(TF.layer_file(0), _layer_items(0))
    # the bug: the stale parent saving would drop the unit's entry
    assert not parent.has(TF.layer_file(0))
    # the fix (cmd_thor): re-open, then write globals
    b = TF.Build(root, keepset_sha256="a" * 64, n_routed_experts=2)
    b.write_file("globals.safetensors", [{"name": "norm.weight", "dtype": "F32", "data": np.ones(4, np.float32)}])
    again = TF.Build(root, keepset_sha256="a" * 64, n_routed_experts=2)
    assert again.has(TF.layer_file(0)) and again.has("globals.safetensors") and again.verify() == []
    src = (REPO / "ops" / "s07_quant.py").read_text()
    i = src.index("def cmd_thor(")
    body = src[i:src.index("\ndef unit(", i)]
    assert body.index("b = TF.Build(build_dir(n_keep, \"A\" + V[\"suffix\"]), keepset_sha256=fp") < body.index("write_globals(b)"), \
        "cmd_thor does not re-open the Build before write_globals"


@test
def recreate_with_other_recipe_refused():
    root = TMP / "b3"
    TF.Build(root, create=True, keepset_sha256="a" * 64, n_routed_experts=2, alloc={"0": {"w13": "Q2_K", "w2": "Q3_K"}},
             heal={"gains": {"0": 0.93}})
    TF.Build(root, create=True, keepset_sha256="a" * 64, n_routed_experts=2, alloc={"0": {"w13": "Q2_K", "w2": "Q3_K"}},
             heal={"gains": {"0": 0.93}})                                      # same recipe resumes
    msg = expect_exit(TF.Build, root, create=True, keepset_sha256="a" * 64, n_routed_experts=2,
                      alloc={"0": {"w13": "Q3_K", "w2": "Q3_K"}}, heal={"gains": {"0": 0.93}})
    assert "alloc" in msg


# ----------------------------------------------------------------------------- CORRUPT-1
@test
def expert_bytes_excludes_mtp():
    root = TMP / "c1"
    b = TF.Build(root, create=True, keepset_sha256="a" * 64, n_routed_experts=2)
    it = _layer_items(0)
    b.write_file(TF.layer_file(0), it)
    mtp = np.zeros((4, 64), np.uint8)
    b.write_file("mtp-0.safetensors", [{"name": "mtp.0.ffn.experts.0.w1.weight", "dtype": "U8", "data": mtp},
                                       {"name": "mtp.0.ffn.experts.0.w1.scale", "dtype": "F8_E8M0", "data": mtp[:, :4]}])
    assert b.expert_bytes() == it[0]["data"].nbytes, (b.expert_bytes(), it[0]["data"].nbytes)


# ----------------------------------------------------------------------------- CORRUPT-2
def _mx(rng, R, C, lo, hi):
    w = rng.integers(0, 256, (R, C // 2), dtype=np.uint8)
    s = rng.integers(lo, hi + 1, (R, C // 32), dtype=np.uint8)
    return w, s


@test
def shared_emax_lossless_and_equal_global():
    rng = np.random.default_rng(3)
    for lo1, hi1, lo3, hi3 in ((110, 120, 113, 127), (120, 127, 111, 118), (5, 18, 5, 18)):
        w1, s1 = _mx(rng, 8, 256, lo1, hi1)
        w3, s3 = _mx(rng, 8, 256, lo3, hi3)
        em = max(G.mxfp4_emax(w1, s1), G.mxfp4_emax(w3, s3))
        r1, r3 = G.mxfp4_to_nvfp4(w1, s1, emax=em), G.mxfp4_to_nvfp4(w3, s3, emax=em)
        assert r1["weight_scale_2"] == r3["weight_scale_2"]
        assert r1["lossy_blocks"] == 0 and r3["lossy_blocks"] == 0
        for r, w, s in ((r1, w1, s1), (r3, w3, s3)):
            assert np.array_equal(G.nvfp4_dequant(r["weight"], r["weight_scale"], float(r["weight_scale_2"])),
                                  G.src_dequant(w, s))
            assert np.array_equal(r["weight"], w)
    # an 18-octave spread across w1/w3 is caught, not silently rounded
    w1, s1 = _mx(rng, 4, 64, 127, 127)
    w3, s3 = _mx(rng, 4, 64, 109, 109)
    w3[:] = 0x11                                                     # every block nonzero
    r3 = G.mxfp4_to_nvfp4(w3, s3, emax=127)
    assert r3["lossy_blocks"] == s3.size
    try:
        G.mxfp4_to_nvfp4(w1, s1, emax=120)
        raise AssertionError("emax below the tensor's own maximum accepted")
    except ValueError:
        pass


@test
def nvfp4_layer_w13_share_global_scale():
    from ops import s07_quant as S7
    rng = np.random.default_rng(11)
    L, E, keep, C = 5, 3, [0, 2], 64
    t = []
    for e in range(E):
        for p, (lo, hi) in (("w1", (110, 118)), ("w2", (115, 125)), ("w3", (114, 126))):   # w1/w3 emax differ
            t.append((f"layers.{L}.ffn.experts.{e}.{p}.weight", "I8", rng.integers(-128, 128, (4, C // 2), dtype=np.int8)))
            t.append((f"layers.{L}.ffn.experts.{e}.{p}.scale", "F8_E8M0", rng.integers(lo, hi + 1, (4, C // 32), dtype=np.uint8)))
    src = TMP / "nv13.safetensors"
    st_io.write(src, t, align=1)
    for d in ("s06", "s03"):
        (env.WORK / d / "bank").mkdir(parents=True, exist_ok=True)
    np.savez(env.WORK / "s06" / "bank" / f"L{L}.npz", keep=np.array(keep), n_tok=np.array([5, 3]),
             amax_in=np.array([1., 2.]), amax_mid_ungated=np.array([4., 5.]))
    np.savez(env.WORK / "s03" / "bank" / f"L{L}.npz", amax_in=np.arange(E) * 10.0)
    out = {n: a for n, _, a in S7.nvfp4_layer(st_io.Reader(src), L, keep, 0.9)}
    rd = st_io.Reader(src)
    for j, e in enumerate(keep):
        b = f"layers.{L}.ffn.experts.{j}"
        g1, g3 = float(out[f"{b}.w1.weight_scale_2"]), float(out[f"{b}.w3.weight_scale_2"])
        assert g1 == g3, (e, g1, g3)
        for p, gain in (("w1", 1.0), ("w3", 1.0), ("w2", 0.9)):
            ref = G.src_dequant(rd.get(f"layers.{L}.ffn.experts.{e}.{p}.weight"),
                                rd.get(f"layers.{L}.ffn.experts.{e}.{p}.scale")) * np.float32(gain)
            got = G.nvfp4_dequant(out[f"{b}.{p}.weight"], out[f"{b}.{p}.weight_scale"], float(out[f"{b}.{p}.weight_scale_2"]))
            assert np.array_equal(got, ref), (e, p)


# ----------------------------------------------------------------------------- RISK-2
class _Free:
    """patch orchestrate.disk_free: a fixed number for the NVMe, plus what parking released."""
    def __init__(self, base: int, seagate: int = 10 << 40):
        self.base, self.seagate, self.orig, self.bonus = base, seagate, orchestrate.disk_free, 200 << 30

    def __call__(self, p):
        from ops import s07_quant as S7
        if str(p).startswith(str(TMP / "seagate")):
            return self.seagate
        return self.base + self.bonus * sum(x.is_symlink() for x in S7.THOR.glob("A*"))

    def __enter__(self):
        orchestrate.disk_free = self
        return self

    def __exit__(self, *a):
        orchestrate.disk_free = self.orig


@test
def disk_preflight_fails_fast():
    from ops import s07_quant as S7
    with _Free(50 << 30):
        msg = expect_exit(S7.disk_preflight, "thor", 140 << 30, S7.THOR_MARGIN)
        assert "disk preflight failed" in msg and "GiB free" in msg
    with _Free(500 << 30):
        S7.disk_preflight("thor", 140 << 30, S7.THOR_MARGIN)
    full = S7.kq_build_estimate({str(L): {"w13": "Q3_K", "w2": "Q3_K"} for L in range(S7.N_LAYERS)}, 192)
    assert 100 << 30 < full < 130 << 30, full / 2**30
    b = S7.mxfp4_build_estimate(128)
    assert abs(b - (89.65 * 2**30 + S7.NONEXPERT_BYTES)) < 0.1 * 2**30, b / 2**30
    assert abs(S7.replay_peak(262_144) - 32.5 * 2**30) < 2**28


def _fake_A(name: str) -> Path:
    from ops import s07_quant as S7
    b = TF.Build(S7.THOR / name, create=True, keepset_sha256="a" * 64, n_routed_experts=2)
    b.write_file(TF.layer_file(0), _layer_items(0))
    b.write_file("globals.safetensors", [{"name": "norm.weight", "dtype": "F32", "data": np.ones(64, np.float32)}])
    b.m["complete"] = True
    b.save()
    return S7.THOR / name


@test
def park_build_verified_symlink_and_resume():
    from ops import s07_quant as S7
    sea = TMP / "seagate"
    sea.mkdir(exist_ok=True)
    root = sea / "v41_staging" / "thor"
    src = _fake_A("A9")
    real_ismount = os.path.ismount
    os.path.ismount = lambda p: True if str(p).startswith(str(sea)) else real_ismount(p)
    try:
        assert "not mounted" in expect_exit(S7.park_build, "A9", park_root=root, seagate=TMP / "nowhere")
        # a stale same-size wrong copy on the Seagate is detected by sha256, removed, and redone on re-run
        (root / "A9").mkdir(parents=True)
        bad = (src / "globals.safetensors").read_bytes()
        (root / "A9" / "globals.safetensors").write_bytes(bytes(len(bad)))
        with _Free(1 << 40):
            assert "does not match" in expect_exit(S7.park_build, "A9", park_root=root, seagate=sea)
            assert src.is_dir() and not src.is_symlink()
            assert S7.park_build("A9", park_root=root, seagate=sea)
        assert src.is_symlink() and src.resolve() == (root / "A9").resolve()
        m = json.load(open(S7.THOR / "A9.parked.json"))
        assert m["location"] == str(root / "A9")
        assert TF.Build(src).verify() == [] and TF.Build(src).has(TF.layer_file(0))
        assert not S7.park_build("A9", park_root=root, seagate=sea)               # idempotent
        # crash between rename and symlink: recovery restores the symlink and drops the leftover
        os.unlink(src)
        (S7.THOR / ".A9.parking").mkdir()
        S7.park_build("A9", park_root=root, seagate=sea)
        assert src.is_symlink() and not (S7.THOR / ".A9.parking").exists()
        # an incomplete build is never parked
        b = TF.Build(S7.THOR / "A8", create=True, keepset_sha256="a" * 64, n_routed_experts=2)
        assert "not complete" in expect_exit(S7.park_build, "A8", park_root=root, seagate=sea)
        import shutil
        shutil.rmtree(b.root)
    finally:
        os.path.ismount = real_ismount


@test
def thor_b_parks_losing_A_when_short():
    from ops import s07_quant as S7
    sea = TMP / "seagate"
    nA = 6
    for x in S7.THOR.glob("A*"):                     # earlier tests' parked builds would count as freed space
        if x.is_symlink():
            x.unlink()
    keepset.save(env.WORK / "s04" / "keepset.json", {str(L): list(range(nA)) for L in range(S7.N_LAYERS)})
    src = _fake_A(f"A{nA}")
    real_ismount, real_root, real_sea = os.path.ismount, S7.PARK_ROOT, orchestrate.SEAGATE
    os.path.ismount = lambda p: True if str(p).startswith(str(sea)) else real_ismount(p)
    S7.PARK_ROOT, orchestrate.SEAGATE = sea / "v41_staging" / "thor", sea
    try:
        need = 107 << 30
        with _Free(61 << 30):                      # 61 GiB free (ops audit blocker 3): park A, then B fits
            S7.make_room_for_b(need)
        assert src.is_symlink() and (S7.THOR / f"A{nA}.parked.json").exists()
        with _Free(61 << 30) as f:                  # nothing left to park and still short -> clear SystemExit
            f.base, f.bonus = 10 << 30, 0
            assert "disk preflight failed" in expect_exit(S7.make_room_for_b, need)
    finally:
        os.path.ismount, S7.PARK_ROOT, orchestrate.SEAGATE = real_ismount, real_root, real_sea


if __name__ == "__main__":
    w = max(len(n) for n, _, _ in RESULTS)
    for n, st, msg in RESULTS:
        print(f"{st:4s}  {n:<{w}}  {msg}")
    bad = [r for r in RESULTS if r[1] == "FAIL"]
    print(f"\n{len(RESULTS) - len(bad)}/{len(RESULTS)} passed")
    sys.exit(1 if bad else 0)
