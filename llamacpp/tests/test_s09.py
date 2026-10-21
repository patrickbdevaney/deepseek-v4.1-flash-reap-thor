"""CPU-only tests for the s09 GGUF path. Run: .venv/bin/python -m pytest -q llamacpp/tests/test_s09.py

No model weights and no GPU: synthetic shards, a fake master and a tiny fake GGUF stand in.
The real config.json from the checkpoint provides the geometry and protected layers.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(1, str(REPO / "llamacpp" / "llama.cpp" / "gguf-py"))


def write_safetensors(path: Path, tensors: dict[str, tuple[str, list[int], bytes]]) -> None:
    hdr, blobs, off = {}, [], 0
    for name, (dtype, shape, data) in tensors.items():
        hdr[name] = {"dtype": dtype, "shape": shape, "data_offsets": [off, off + len(data)]}
        blobs.append(data)
        off += len(data)
    h = json.dumps(hdr).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h + b"".join(blobs))


# ---------------------------------------------------------------- keep-set
def test_s09_uses_the_pipeline_fingerprint(tmp_path):
    from ops.lib import keepset
    from ops import s09_keyset
    keep = {str(L): [L, L + 1] for L in range(3)}
    fp = keepset.save(tmp_path / "k.json", keep)
    got, fp2 = s09_keyset.check_master_keepset(tmp_path / "k.json", n_layers=3, n_experts=2, n_source=8, expect_sha=fp)
    assert fp2 == fp and got == keepset.canonical(keep)
    with pytest.raises(SystemExit, match="MISMATCH"):
        s09_keyset.check_master_keepset(tmp_path / "k.json", n_layers=3, n_experts=2, n_source=8, expect_sha="0" * 64)
    with pytest.raises(SystemExit, match="expected 3"):
        s09_keyset.check_master_keepset(tmp_path / "k.json", n_layers=3, n_experts=3, n_source=8, expect_sha=fp)


def fake_master(d: Path, n_layers=2, n_exp=4, n_src=8, stale=False, bad_gate=False):
    from ops.lib import keepset
    keep = {str(L): list(range(L, L + n_exp)) for L in range(n_layers)}
    (d / "config.json").write_text(json.dumps({"text_config": {"n_routed_experts": n_exp, "num_hidden_layers": n_layers}}))
    fp = keepset.save(d / "reap_keepset.json", keep)
    wmap, tens = {}, {}
    for L in range(n_layers):
        for e in range(n_exp + (1 if stale and L == 1 else 0)):
            for w in ("w1", "w2", "w3"):
                wmap[f"layers.{L}.ffn.experts.{e}.{w}.weight"] = "s.safetensors"
        for s, shape in (("gate.weight", [n_exp + (1 if bad_gate else 0), 8]), ("gate.bias", [n_exp])):
            n = f"layers.{L}.ffn.{s}"
            wmap[n] = "s.safetensors"
            tens[n] = ("F32", shape, bytes(4 * int(np.prod(shape))))
    (d / "model.safetensors.index.json").write_text(json.dumps({"weight_map": wmap}))
    write_safetensors(d / "s.safetensors", tens)
    return fp


def test_converter_master_checks(tmp_path):
    sys.path.insert(0, str(REPO / "llamacpp"))
    conv = importlib.import_module("convert_v41_reap")
    fp = fake_master(tmp_path)
    assert conv.check_master(tmp_path, fp, 8)["fingerprint"] == fp
    with pytest.raises(SystemExit, match="MISMATCH"):
        conv.check_master(tmp_path, "0" * 64, 8)
    for kw, msg in ((dict(stale=True), "stale"), (dict(bad_gate=True), "rows")):
        d = tmp_path / msg
        d.mkdir()
        fp2 = fake_master(d, **kw)
        with pytest.raises(SystemExit, match=msg):
            conv.check_master(d, fp2, 8)
    k = json.loads((tmp_path / "reap_keepset.json").read_text())
    k["keep"]["0"][0] = 7                                        # hand edit: stored sha no longer matches
    (tmp_path / "reap_keepset.json").write_text(json.dumps(k))
    with pytest.raises(SystemExit, match="edited by hand"):
        conv.check_master(tmp_path, fp, 8)
    (tmp_path / "reap_keepset.json").unlink()
    with pytest.raises(SystemExit, match="missing"):
        conv.check_master(tmp_path, fp, 8)


# ---------------------------------------------------------------- engram row file
def test_rowfile_roundtrip(tmp_path):
    src, work = tmp_path / "src", tmp_path / "work"
    src.mkdir()
    rng = np.random.default_rng(0)
    n = 3000
    vals = rng.integers(0, 256, (n, 256), dtype=np.uint8)
    vals[vals == 0x7F] = 0x7E                                     # keep NaN codes out of the dequant check
    vals[vals == 0xFF] = 0xFE
    scl = rng.integers(110, 140, (n, 8), dtype=np.uint8)
    write_safetensors(src / "model-00047-of-00048.safetensors", {
        "layers.1.engram.embed.weight": ("F8_E4M3", [n, 256], vals.tobytes()),
        "layers.1.engram.embed.scale": ("F8_E8M0", [n, 8], scl.tobytes())})
    (src / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
        "layers.1.engram.embed.weight": "model-00047-of-00048.safetensors",
        "layers.1.engram.embed.scale": "model-00047-of-00048.safetensors"}}))
    (src / "config.json").write_text(json.dumps({"text_config": {"engram_layer_ids": [1]}}))
    sha = hashlib.sha256((src / "model-00047-of-00048.safetensors").read_bytes()).hexdigest()
    (tmp_path / "v41_1017_tree.json").write_text(json.dumps([{"path": "model-00047-of-00048.safetensors",
                                                              "lfs": {"oid": sha}}]))
    env = {**os.environ, "V41_SRC": str(src), "V41_WORK": str(work)}
    script = REPO / "ops" / "s09_engram_rowfile.py"
    for args in (["build"], ["verify", "--full", "--sample", "500"]):
        r = subprocess.run([sys.executable, str(script), *args, "--dst", str(tmp_path / "rows")], env=env,
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
    assert "PASS L1" in r.stdout
    out = tmp_path / "rows" / "engram_L1.v41rows"
    raw = out.read_bytes()
    assert len(raw) == 4096 + n * 264 and raw[:8] == b"V41ENGR1"
    assert struct.unpack_from("<Q", raw, 44)[0] == 4096
    r5 = raw[4096 + 5 * 264: 4096 + 6 * 264]
    assert r5[:256] == vals[5].tobytes() and r5[256:] == scl[5].tobytes()
    # flip one byte: verify must fail
    with open(out, "r+b") as f:
        f.seek(4096 + 17 * 264)
        f.write(bytes([raw[4096 + 17 * 264] ^ 1]))
    r = subprocess.run([sys.executable, str(script), "verify", "--full", "--sample", "10",
                        "--dst", str(tmp_path / "rows")], env=env, capture_output=True, text=True)
    assert r.returncode == 1


def test_rowfile_dequant_matches_reference():
    """The C++ reader's fp8 decode + bf16 rounding against torch's float8_e4m3fn."""
    torch = pytest.importorskip("torch")
    from ops.lib.engram_rows import safetensors_header  # noqa: F401  (import check only)
    codes = torch.arange(256, dtype=torch.uint8)
    ref = codes.view(torch.float8_e4m3fn).float()

    def fp8(v):                                                   # mirrors deepseek41-engram-rowstore.h
        s, e, m = v >> 7, (v >> 3) & 0xF, v & 7
        if e == 0xF and m == 7:
            r = float("nan")
        elif e == 0:
            r = (m / 8.0) * 2.0 ** -6
        else:
            r = (1 + m / 8.0) * 2.0 ** (e - 7)
        return -r if s else r
    mine = torch.tensor([fp8(int(c)) for c in codes])
    assert torch.equal(torch.isnan(mine), torch.isnan(ref))
    ok = ~torch.isnan(ref)
    assert torch.equal(mine[ok], ref[ok])


# ---------------------------------------------------------------- ladder planner
def fake_source_gguf(path: Path, n_layer: int):
    import gguf
    w = gguf.GGUFWriter(str(path), "deepseek41")
    w.add_string("deepseek41.reap.keep_set_sha256", "f" * 64)
    blk = np.zeros((1, 256), np.float32)
    for L in range(n_layer):
        for p in ("gate", "up", "down"):
            w.add_tensor(f"blk.{L}.ffn_{p}_exps.weight", np.zeros((4, 512), np.float32))
        for nm in ("attn_q_a", "attn_kv", "ffn_gate_shexp", "hc_attn_fn", "indexer.proj"):
            w.add_tensor(f"blk.{L}.{nm}.weight", blk)
        w.add_tensor(f"blk.{L}.attn_norm.weight", np.zeros(256, np.float32))
    w.add_tensor("token_embd.weight", blk)
    w.add_tensor("output.weight", blk)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


@pytest.mark.parametrize("level", ["L0", "L1", "L2", "L3"])
def test_planner_hits_target_and_never_uses_forbidden(tmp_path, level):
    from ops import s09_gguf_ladder as lad
    n_layer = int(lad.text_config()["num_hidden_layers"])
    src = tmp_path / "src.gguf"
    fake_source_gguf(src, n_layer)
    scores = {L: float((L * 7) % 11) for L in range(n_layer)}
    p = lad.plan(level, scores, src)
    spec = lad.LEVELS[level]
    if spec["target"]:
        assert p["expert_bpw"] <= spec["target"] + 1e-6
        assert p["expert_bpw"] >= spec["target"] - 0.12, p["expert_bpw"]   # within one promotion step
    prot = set(p["protected_layers"])
    assert {0, 1, 2, 8, 14, 20, 37, 38, 39} <= prot
    for k, t in p["expert_types"].items():
        assert t not in lad.FORBIDDEN_EXPERT
        L = int(k.split(".")[0])
        if L in prot:
            assert t == spec["protected"]
    rules = dict(p["rules"])
    assert any("shexp" in r and t == "Q6_K" for r, t in rules.items())
    assert any("attn_" in r and t == "Q8_0" for r, t in rules.items())
    assert any("hc_" in r and t == "F32" for r, t in rules.items())


# ---------------------------------------------------------------- model card
def test_card_renders_only_from_artifacts(tmp_path, monkeypatch):
    from ops import s09_card as card
    from ops import s09_gguf_ladder as lad
    n_layer = int(lad.text_config()["num_hidden_layers"])
    src = tmp_path / "src.gguf"
    fake_source_gguf(src, n_layer)
    p = lad.plan("L2", None, src)
    monkeypatch.setattr(lad, "OUT", tmp_path)
    d = tmp_path / "L2"
    (d / "teacher_eval").mkdir(parents=True)
    (d / "plan.json").write_text(json.dumps(p))
    split = d / f"{lad.NAME}-L2-00001-of-00001.gguf"
    fake_source_gguf(split, 2)                                  # carries keep_set_sha256 = f*64
    kld = {"identity": {"gguf": str(split), "keep_set_sha256": "f" * 64}, "tokens": 1234,
           "stats": {k: {"mean": 0.1, "ci95": [0.09, 0.11]} for k in ("kl_top32", "top1", "dnll")},
           "measured": "2026-10-19T00:00:00"}
    (d / "teacher_eval" / "teacher_kld.json").write_text(json.dumps(kld))
    with pytest.raises(SystemExit, match="--s10"):
        card.render("L2", None, False)
    txt = card.render("L2", None, True)
    assert "DRAFT" in txt and "{{" not in txt and "0.1000" in txt
    s10 = tmp_path / "s10.json"
    s10.write_text(json.dumps({"verdict": {"L2": "FAIL"}, "summary": {"mmlu": 0.7}}))
    with pytest.raises(SystemExit, match="failed gate"):
        card.render("L2", s10, False)
    s10.write_text(json.dumps({"verdict": {"L2": "PASS"}, "summary": {"mmlu": 0.7}}))
    txt = card.render("L2", s10, False)
    assert "| mmlu | 0.7000 |" in txt and "DRAFT" not in txt
    kld["identity"]["keep_set_sha256"] = "0" * 64
    (d / "teacher_eval" / "teacher_kld.json").write_text(json.dumps(kld))
    with pytest.raises(SystemExit, match="keep-set"):
        card.render("L2", s10, False)


# ---------------------------------------------------------------- s12 manifest
def test_manifest_matches_s12_contract(tmp_path, monkeypatch):
    from ops import s09_gguf_ladder as lad
    monkeypatch.setattr(lad, "OUT", tmp_path)
    monkeypatch.setattr(lad, "MANIFEST", tmp_path / "manifest.json")
    import gguf
    for lv in ("L2", "L3"):
        (tmp_path / lv).mkdir()
        f = tmp_path / lv / f"{lad.NAME}-{lv}-00001-of-00001.gguf"
        w = gguf.GGUFWriter(str(f), "deepseek41")
        w.add_string("deepseek41.reap.keep_set_sha256", "a" * 64)
        w.add_uint32("tokenizer.ggml.eos_token_id", 1)
        w.add_uint32("tokenizer.ggml.eot_token_id", 1)
        w.add_tensor("x.weight", np.zeros((1, 32), np.float32))
        w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    m = lad.write_manifest(["L1", "L2", "L3"], None)
    on_disk = json.loads((tmp_path / "manifest.json").read_text())
    assert on_disk == json.loads(json.dumps(m))
    # exactly what ops/s12_publish.py reads
    assert "repo" not in m and m["dense_precision"]
    paths = [Path(f["path"]) for f in m["files"]]
    assert all(p.exists() for p in paths) and len({p.name for p in paths}) == len(paths)
    for f in m["files"]:
        assert f["sha256"] == hashlib.sha256(Path(f["path"]).read_bytes()).hexdigest()
    assert sorted(m["levels"]) == ["L2", "L3"] and m["keepset_sha256"] == "a" * 64


# ---------------------------------------------------------------- heal gain (deepseek41.reap.routed_gain)
def _write_kv_only(path: Path, reap: dict):
    import gguf
    sys.path.insert(0, str(REPO / "llamacpp"))
    conv = importlib.import_module("convert_v41_reap")
    w = gguf.GGUFWriter(str(path), "deepseek41")
    conv.write_reap_kv(w, "deepseek41", reap)
    w.add_tensor("x.weight", np.zeros((1, 32), np.float32))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    from ops import s09_gguf_ladder as lad
    return lad.gguf_kv(path)


def test_converter_routed_gain_kv(tmp_path):
    sys.path.insert(0, str(REPO / "llamacpp"))
    conv = importlib.import_module("convert_v41_reap")
    from ops.lib import keepset
    from ops import s09_gguf_ladder as lad
    n = 40
    keep = {str(L): [L, L + 1] for L in range(n)}
    fp = keepset.fingerprint(keep)
    gains = {str(L): 1.0 + 0.003 * L for L in range(n)}
    gains["7"] = 1.0                                             # a layer at exactly 1 is still written
    (tmp_path / "g.json").write_text(json.dumps({"keepset_sha256": fp, "source": "t", "gains": gains}))
    g = conv.load_heal_gains(tmp_path / "g.json", n, fp)
    assert g is not None and len(g) == n and g[7] == 1.0
    reap = {"fingerprint": fp, "n_source": 384, "keep": keep, "n_layers": n, "routed_gain": g}
    kv = _write_kv_only(tmp_path / "healed.gguf", reap)
    got = kv[lad.GAIN_KEY]
    assert lad.GAIN_KEY == "deepseek41." + conv.ROUTED_GAIN_KEY
    assert len(got) == n and np.array_equal(np.asarray(got, np.float32), np.asarray(g, np.float32))
    from gguf import GGUFReader, GGUFValueType
    f = GGUFReader(str(tmp_path / "healed.gguf")).fields[lad.GAIN_KEY]
    assert f.types == [GGUFValueType.ARRAY, GGUFValueType.FLOAT32]     # the fork reads std::vector<float>

    # all 1.0 (heal not adopted, or arm B): no key at all, the graph stays unchanged
    (tmp_path / "ones.json").write_text(json.dumps({str(L): 1.0 for L in range(n)}))
    assert conv.load_heal_gains(tmp_path / "ones.json", n) is None
    kv = _write_kv_only(tmp_path / "plain.gguf", {**reap, "routed_gain": None})
    assert lad.GAIN_KEY not in kv and kv["deepseek41.reap.keep_set_sha256"] == fp

    # malformed or foreign gains are fatal
    for bad, msg in (({str(L): 1.0 for L in range(n - 1)}, "cover"), ({**gains, "3": 0.0}, "positive"),
                     ({**gains, "3": float("nan")}, "positive")):
        (tmp_path / "b.json").write_text(json.dumps(bad))
        with pytest.raises(SystemExit, match=msg):
            conv.load_heal_gains(tmp_path / "b.json", n)
    with pytest.raises(SystemExit, match="MISMATCH"):
        conv.load_heal_gains(tmp_path / "g.json", n, "0" * 64)


def test_manifest_and_card_carry_routed_gain(tmp_path, monkeypatch):
    from ops import s09_gguf_ladder as lad, s12_publish as P
    import gguf
    monkeypatch.setattr(lad, "OUT", tmp_path)
    monkeypatch.setattr(lad, "MANIFEST", tmp_path / "manifest.json")
    g = [1.0 + 0.01 * L for L in range(40)]
    for lv in ("L2", "L3"):
        (tmp_path / lv).mkdir()
        f = tmp_path / lv / f"{lad.NAME}-{lv}-00001-of-00001.gguf"
        w = gguf.GGUFWriter(str(f), "deepseek41")
        w.add_string("deepseek41.reap.keep_set_sha256", "a" * 64)
        w.add_array(lad.GAIN_KEY, [float(v) for v in g])
        w.add_uint32("tokenizer.ggml.eos_token_id", 1)
        w.add_uint32("tokenizer.ggml.eot_token_id", 1)
        w.add_tensor("x.weight", np.zeros((1, 32), np.float32))
        w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    m = lad.write_manifest(["L2", "L3"], None)
    assert np.allclose(m["routed_gain"], g) and len(m["routed_gain"]) == 40
    s = P.gguf_heal_sentence(m)
    assert "`deepseek41.reap.routed_gain`" in s and "median 1.2000" in s and "range 1.0000-1.3900" in s
    heal = {"keepset_sha256": "a" * 64, "adopt": True, "gains": {str(L): v for L, v in enumerate(g)}}
    assert P.gguf_heal_sentence(m, heal) == s
    with pytest.raises(SystemExit, match="heal decision"):
        P.gguf_heal_sentence(m, {**heal, "adopt": False, "gains": None})
    with pytest.raises(SystemExit, match="heal decision"):
        P.gguf_heal_sentence({**m, "routed_gain": None}, heal)
    assert "No heal gain" in P.gguf_heal_sentence({**m, "routed_gain": None}, {**heal, "adopt": False})
    # a different keep-set (arm B) is not checked against arm A's decision
    assert "No heal gain" in P.gguf_heal_sentence({**m, "routed_gain": None}, {**heal, "keepset_sha256": "b" * 64})
    # levels that disagree on the gain are fatal (L3 rewritten without the key)
    f = tmp_path / "L3" / f"{lad.NAME}-L3-00001-of-00001.gguf"
    w = gguf.GGUFWriter(str(f), "deepseek41")
    w.add_string("deepseek41.reap.keep_set_sha256", "a" * 64)
    w.add_uint32("tokenizer.ggml.eos_token_id", 1)
    w.add_uint32("tokenizer.ggml.eot_token_id", 1)
    w.add_tensor("x.weight", np.zeros((1, 32), np.float32))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    with pytest.raises(SystemExit, match="disagree"):
        lad.write_manifest(["L2", "L3"], None)
