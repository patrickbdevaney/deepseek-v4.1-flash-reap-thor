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
  * the vision tower goes to mmproj (the PR's filter), and DSpark/MTP follows the usual --no-mtp/--dspark;
  * the chat template is ALWAYS the fork's V4.1 one (models/templates/deepseek-ai-DeepSeek-V4.1-Flash.jinja).
    The PR picks it from the directory NAME ("V4.1" in it), and master_REAP<n> / s05/REAP<n> would get the V4
    template, whose DSML tags differ: llama.cpp then detects the V4 parser and V4.1 tool calls come back as
    plain content. ops/s09_gguf_ladder.py verify_whole refuses a level whose template is not this file.

Engram shards: s05 does not rehost base shards 47/48, but the master's index still maps every
layers.{1,14}.engram.* tensor to them, and the PR's index_tensors opens every shard the index names.
--engram-src DIR builds a symlink view (--view-dir, default <outfile dir>/<NAME>-view): every top-level
entry of the master plus each index shard the master lacks, taken from DIR (the base checkpoint). A
borrowed shard must hold exactly the tensors the index maps to it, all of them `.engram.` tensors (shard
headers only), so nothing but Engram can come from outside the master. embed/scale are then dropped and
wkv/q_weight/k_weight are converted (the fork's deepseek41 needs them).

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
        [--engram-src BASE_DIR [--view-dir DIR]]
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
V41_TEMPLATE = LCPP / "models" / "templates" / "deepseek-ai-DeepSeek-V4.1-Flash.jinja"


def v41_chat_template() -> str:
    """The one chat template every GGUF of this release carries (ops/s09_gguf_ladder.py checks equality)."""
    t = V41_TEMPLATE.read_text(encoding="utf-8")
    if "' calls>" not in t:          # the V4.1 DSML tag; the V4 template has 'tool_calls>' with no space
        raise SystemExit(f"FATAL: {V41_TEMPLATE} is not the V4.1 template (no ' calls>' DSML tag)")
    return t


def index_shards(master: Path) -> dict[str, list[str]]:
    """{shard file name: [tensor names the index maps to it]}."""
    wmap = json.loads((master / "model.safetensors.index.json").read_text())["weight_map"]
    out: dict[str, list[str]] = {}
    for name, shard in wmap.items():
        out.setdefault(shard, []).append(name)
    return out


def build_view(master: Path, engram_src: Path, view: Path) -> list[str]:
    """Symlink view = master + the index shards the master lacks (from engram_src). Returns the borrowed names.
    Fatal unless every borrowed shard exists in engram_src and its header holds exactly the index's tensors for
    it, all `.engram.`; an existing view is rebuilt (it holds only symlinks)."""
    master, engram_src, view = Path(master).resolve(), Path(engram_src), Path(view)
    if view.resolve() == master:
        raise SystemExit("FATAL: --view-dir must not be the master itself")
    shards = index_shards(master)
    borrowed = sorted(s for s in shards if not (master / s).exists())
    for s in borrowed:
        f = engram_src / s
        if not f.exists():
            raise SystemExit(f"FATAL: the master index maps {len(shards[s])} tensors to {s}, which is neither in "
                             f"{master} nor in --engram-src {engram_src}")
        bad = [n for n in shards[s] if ".engram." not in n]
        if bad:
            raise SystemExit(f"FATAL: {s} is not in the master and the index maps non-Engram tensors to it "
                             f"(e.g. {bad[:3]}); only the unpruned Engram shards may come from the base checkpoint")
        have = {k for k in _st_header(f) if k != "__metadata__"}
        if have != set(shards[s]):
            raise SystemExit(f"FATAL: {f} header {sorted(have)[:4]}... != the index's tensors for it "
                             f"{sorted(shards[s])[:4]}... (wrong file?)")
    if view.exists():
        for e in view.iterdir():
            if not e.is_symlink():
                raise SystemExit(f"FATAL: {view} holds a real file ({e.name}); refusing to reuse it as a view")
            e.unlink()
    view.mkdir(parents=True, exist_ok=True)
    for e in master.iterdir():
        (view / e.name).symlink_to(e.resolve())
    for s in borrowed:
        (view / s).symlink_to((engram_src / s).resolve())
    log.info("view %s: master %s + %s from %s", view, master, borrowed, engram_src)
    return borrowed


def require_index_shards(model_dir: Path) -> None:
    missing = sorted(s for s in index_shards(model_dir) if not (model_dir / s).exists())
    if missing:
        raise SystemExit(f"FATAL: {model_dir} lacks index shards {missing} (s05 does not rehost the Engram shards "
                         f"47/48): pass --engram-src <base checkpoint dir>")


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

    orig_vocab = cls.set_vocab
    template = v41_chat_template()

    def set_vocab(self):
        # set_vocab is the last metadata step (after __init__'s name-based default and set_gguf_parameters), so a
        # tokenizer_config / chat_template.jinja in the master cannot override this either
        orig_vocab(self)
        self.gguf_writer.add_chat_template(template)
        log.info("chat template: forced %s", V41_TEMPLATE.name)

    cls.index_tensors = index_tensors
    cls.set_vocab = set_vocab
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
    ap.add_argument("--engram-src", type=Path, help="base checkpoint dir holding the index shards the master lacks "
                    "(Engram 47/48); the converter then reads a symlink view")
    ap.add_argument("--view-dir", type=Path, help="where to build that view (default: <outfile dir>/<master name>-view)")
    a, rest = ap.parse_known_args()
    model_dir = a.master
    if a.engram_src:
        if a.view_dir:
            model_dir = a.view_dir
        else:
            out = next((Path(rest[i + 1]) for i, x in enumerate(rest[:-1]) if x == "--outfile"), None)
            model_dir = (out.parent if out else Path(".")) / f"{a.master.resolve().name}-view"
        build_view(a.master, a.engram_src, model_dir)
    require_index_shards(model_dir)
    v41_chat_template()
    try:
        reap = check_master(model_dir, a.expect_keep_sha, a.n_source_experts)
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
    sys.argv = [str(LCPP / "convert_hf_to_gguf.py"), str(model_dir), "--outtype", "bf16", *rest]
    convert_hf_to_gguf.main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
