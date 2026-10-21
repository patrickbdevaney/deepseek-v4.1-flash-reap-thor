#!/usr/bin/env python3
"""REAP-aware DeepSeek-V4.1 -> GGUF converter (wraps the pinned fork's convert_hf_to_gguf.py).

This produces the *source* GGUF that every s09 level is quantized from, exactly once. It is a
lossless re-encoding of the MXFP4 master:
  * routed experts are repacked bit-exact into GGML MXFP4 (the PR's own path; E2M1 + E8M0 map 1:1);
  * FP8 [32,32]-block weights are dequantized to BF16, which is exact because e4m3 x 2^k always fits
    bf16. The PR writes Q8_0 here, and every level that wants Q6_K/Q8_0 would then be a requant
    of a requant;
  * BF16 and F32 tensors are kept as stored (hyper-connection, norms and sinks stay F32);
  * the Engram tables are dropped. They are served from the row files (ENGRAM_ROWFILE.md), so the
    converter's 2 x 100 GB Q8_0 re-encode of them is skipped;
  * the vision tower goes to mmproj (the PR's filter), and DSpark/MTP follows the usual --no-mtp/--dspark.

REAP checks, all fatal (KeepsetMismatch is a SystemExit, nothing can swallow it):
  * <master>/reap_keepset.json (written by s05 via ops/lib/keepset.save) must exist, hash to its own
    stored sha256, and equal --expect-keep-sha (the arbitrated s04 keep-set);
  * every layer keeps exactly config n_routed_experts experts out of --n-source-experts;
  * the safetensors index must hold experts 0..N-1 for w1/w2/w3 on every layer, and no expert >= N;
  * gate.weight / gate.bias (and bias_vl if present) must have N rows (from the shard headers;
    no weights are read).
The fingerprint, the source expert count and the per-layer kept ids are written into the GGUF as
deepseek41.reap.*, so every derived level can be checked again without the master.

Heal gain: the MXFP4 master ships unhealed (E8M0 cannot carry an arbitrary scale). --heal-gains
takes the adopted per-layer gains ({"0": g, ..., "<n_layers-1>": g}, as ops/s07_quant.py
heal_decision() returns them) and, if any gain != 1.0, writes them as the f32 array
deepseek41.reap.routed_gain. The fork's patch 0004 scales layer il's routed-expert sum by
gain[il] before the shared expert is added, the same as the gain folded into every routed w2
in the NVFP4 and Thor builds. All gains 1.0 (heal not adopted): no key, graph unchanged.

Usage:
  python llamacpp/convert_v41_reap.py MASTER_DIR --outfile SRC.gguf --expect-keep-sha SHA [--heal-gains J] [--no-mtp]
  python llamacpp/convert_v41_reap.py MASTER_DIR --check-only --expect-keep-sha SHA
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import re
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LCPP = HERE / "llama.cpp"
sys.path.insert(0, str(HERE.parent))
from ops.s09_keyset import KEEPSET_FILE, KeepsetMismatch, check_master_keepset  # noqa: E402

KeySetError = KeepsetMismatch

log = logging.getLogger("convert_v41_reap")

ROUTED_GAIN_KEY = "reap.routed_gain"        # prefixed with the arch name: deepseek41.reap.routed_gain


def load_heal_gains(path: Path, n_layers: int, fingerprint: str | None = None) -> list[float] | None:
    """{"0": g, ..., "<n_layers-1>": g} (or {"keepset_sha256", "gains": {...}}, as CHAIN_STEPS writes it)
    -> [g_0 .. g_{n-1}], or None when every gain is exactly 1.0 (nothing to write). Every layer must be
    present, every gain finite and positive, and a recorded keep-set must be the master's."""
    d = json.loads(Path(path).read_text())
    if isinstance(d, dict) and isinstance(d.get("gains"), dict):
        if fingerprint is not None and "keepset_sha256" in d:
            from ops.lib import keepset
            keepset.require(d["keepset_sha256"], fingerprint, where=str(path))
        d = d["gains"]
    if sorted(d, key=int) != [str(L) for L in range(n_layers)]:
        raise SystemExit(f"{path}: heal gains must cover layers 0..{n_layers - 1} exactly, has {sorted(d, key=int)[:5]}...")
    g = [float(d[str(L)]) for L in range(n_layers)]
    for L, v in enumerate(g):
        if not math.isfinite(v) or v <= 0.0:
            raise SystemExit(f"{path}: layer {L} gain {v} is not a positive finite number")
    return None if all(v == 1.0 for v in g) else g


def write_reap_kv(w, arch: str, reap: dict) -> None:
    """The deepseek41.reap.* keys on a GGUFWriter. routed_gain only when the heal was adopted."""
    w.add_string(f"{arch}.reap.keep_set_sha256", reap["fingerprint"])
    w.add_uint32(f"{arch}.reap.source_expert_count", reap["n_source"])
    flat = [int(i) for L in range(reap["n_layers"]) for i in reap["keep"][str(L)]]
    w.add_array(f"{arch}.reap.keep_ids", flat)
    w.add_string(f"{arch}.reap.source_revision", "1bae0e1088993b9b82de08f59de5be5a34e42170")
    gain = reap.get("routed_gain")
    if gain is not None:
        if len(gain) != reap["n_layers"]:
            raise SystemExit(f"routed_gain has {len(gain)} entries for {reap['n_layers']} layers")
        w.add_array(f"{arch}.{ROUTED_GAIN_KEY}", [float(v) for v in gain])     # python floats -> FLOAT32


def _st_header(path: Path) -> dict:
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(n))


def check_master(master: Path, expect_sha: str | None, n_source: int) -> dict:
    cfg = json.loads((master / "config.json").read_text())
    tc = {**cfg, **(cfg.get("text_config") or {})}
    n_exp, n_layers = int(tc["n_routed_experts"]), int(tc["num_hidden_layers"])
    keep, fp = check_master_keepset(master / KEEPSET_FILE, n_layers=n_layers, n_experts=n_exp,
                                    n_source=n_source, expect_sha=expect_sha)

    wmap = json.loads((master / "model.safetensors.index.json").read_text())["weight_map"]
    pat = re.compile(r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.weight$")
    seen: dict[int, set] = {}
    for name in wmap:
        m = pat.match(name)
        if m:
            L, e = int(m.group(1)), int(m.group(2))
            if L >= n_layers:
                continue                     # MTP / DSpark blocks keep their own 128 experts, unpruned
            if e >= n_exp:
                raise KeySetError(f"{name}: expert id >= n_routed_experts {n_exp} (surgery left a stale expert)")
            seen.setdefault(L, set()).add((e, m.group(3)))
    want = {(e, w) for e in range(n_exp) for w in ("w1", "w2", "w3")}
    for L in range(n_layers):
        if seen.get(L, set()) != want:
            missing = sorted(want - seen.get(L, set()))[:4]
            raise KeySetError(f"layer {L}: routed expert tensors incomplete (missing e.g. {missing})")

    headers: dict[str, dict] = {}
    for L in range(n_layers):
        for suffix in ("gate.weight", "gate.bias", "gate.bias_vl"):
            name = f"layers.{L}.ffn.{suffix}"
            if name not in wmap:
                if suffix == "gate.bias_vl":
                    continue
                raise KeySetError(f"{name} missing from the index")
            shard = wmap[name]
            if shard not in headers:
                headers[shard] = _st_header(master / shard)
            rows = headers[shard][name]["shape"][0]
            if rows != n_exp:
                raise KeySetError(f"{name} has {rows} rows, keep-set has {n_exp}")
    log.info("keep-set %s OK: %d layers x %d / %d experts", fp, n_layers, n_exp, n_source)
    return {"fingerprint": fp, "n_source": n_source, "keep": keep, "n_layers": n_layers}


def patch_converter(reap: dict, lossless: bool) -> None:
    sys.path.insert(1, str(LCPP / "gguf-py"))
    sys.path.insert(0, str(LCPP))
    import gguf
    from conversion import deepseek as ds

    cls = ds.DeepseekV41Model

    orig_index = cls.index_tensors

    def index_tensors(self, remote_hf_model_id=None):
        out = orig_index(self, remote_hf_model_id=remote_hf_model_id)
        dropped = [n for n in self.model_tensors if ".engram.embed." in n]
        for n in dropped:
            del self.model_tensors[n]
        log.info("engram tables dropped (%d tensors): served from the .v41rows files", len(dropped))
        return out

    orig_force = cls.tensor_force_quant

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        if lossless and name in self._dsv4_fp8_dequantized and n_dims >= 2:
            return gguf.GGMLQuantizationType.BF16
        return orig_force(self, name, new_name, bid, n_dims)

    orig_params = cls.set_gguf_parameters

    def set_gguf_parameters(self):
        orig_params(self)
        arch = gguf.MODEL_ARCH_NAMES[self.model_arch]
        write_reap_kv(self.gguf_writer, arch, reap)

    cls.index_tensors = index_tensors
    cls.tensor_force_quant = tensor_force_quant
    cls.set_gguf_parameters = set_gguf_parameters


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("master", type=Path)
    ap.add_argument("--expect-keep-sha", required=True)
    ap.add_argument("--n-source-experts", type=int, default=384)
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--q8-dense", action="store_true", help="PR behaviour: FP8 dense -> Q8_0 (not lossless)")
    ap.add_argument("--heal-gains", type=Path, help='JSON {"0": g, ..., "39": g}: the adopted per-layer heal gains; '
                    "written as deepseek41.reap.routed_gain only if any gain != 1.0")
    a, rest = ap.parse_known_args()
    try:
        reap = check_master(a.master, a.expect_keep_sha, a.n_source_experts)
    except KeySetError as e:
        log.error("FATAL keep-set: %s", e)
        return 2
    log.info("keep-set %s", reap["fingerprint"])
    reap["routed_gain"] = load_heal_gains(a.heal_gains, reap["n_layers"], reap["fingerprint"]) if a.heal_gains else None
    if reap["routed_gain"] is None:
        log.info("heal gain: %s", "all 1.0, no routed_gain key" if a.heal_gains else "not given, no routed_gain key")
    else:
        g = sorted(reap["routed_gain"])
        log.info("heal gain: writing %s (median %.4f, range %.4f-%.4f)", ROUTED_GAIN_KEY, g[len(g) // 2], g[0], g[-1])
    if a.check_only:
        return 0
    patch_converter(reap, lossless=not a.q8_dense)
    import convert_hf_to_gguf
    sys.argv = [str(LCPP / "convert_hf_to_gguf.py"), str(a.master), "--outtype", "bf16", *rest]
    convert_hf_to_gguf.main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
