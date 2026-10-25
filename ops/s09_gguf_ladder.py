#!/usr/bin/env python3
"""s09: the community GGUF ladder for the V4.1-1017 REAP. Every level is quantized once, from one source.

The source is llamacpp/convert_v41_reap.py's lossless GGUF of the MXFP4 master. In it the routed
experts are bit-exact GGML MXFP4, FP8 dense weights are exact BF16, and HC/norm/sink tensors are F32.
No level is ever made from another level.

Levels (target = average bpw over the routed-expert tensors; the rest of the file is fixed):
  L0  experts MXFP4, copied bit for bit (lossless, ~152 GiB; the reference GGUF, does not fit Thor)
  L1  ~3.40   IQ3_XXS / IQ3_S, protected layers MXFP4
  L2  ~2.85   IQ2_XS .. IQ3_S, protected layers IQ3_S (flagship, 128 GB devices)
  L3  ~2.46   IQ2_XXS .. IQ3_XXS, protected layers IQ3_XXS (optional fallback)
Fixed for every level:
  * attention, compressor, indexer and engram_wkv: Q8_0;
  * shared experts, token_embd and output: Q6_K;
  * router (ffn_gate_inp): kept as stored (BF16);
  * hyper-connection: F32, forced by the fork's llama-quant patch;
  * norms, sinks and biases: kept (F32);
  * any 2-D weight no rule names: Q8_0, never the expert type.
Never Q4_K/Q5_K/IQ4 for experts. The source is FP4, and a 4-bit non-FP4 grid costs bytes and still
loses information; at 4.25 bpw the only right answer is MXFP4 copied as-is.
Protected layers: 0-1, Engram 1/14, KV sources 2/8/14/20, DSpark taps 37-39 (from config).

  plan    LEVEL [--layer-scores J]   tensor-type file + exact predicted size (reads GGUF headers only)
  imatrix [--check]                  legacy .dat from the s03 bank, remapped to the keep-set, then imatrix_verify;
                                     the keep-set fingerprint is recorded in the .dat's dataset string, and quant
                                     refuses an imatrix built for another keep-set (--check: exit 3 if stale)
  quant   LEVEL                      llama-quantize (+ optional --ef error feedback) -> gguf-split (<50 GB) -> verify
  verify  LEVEL                      types, eos/eot, keep-set fingerprint and split sizes, from the written files
  manifest [--levels ..] [--repo R]  $V41_WORK/s09/manifest.json for ops/s12_publish.py:
                                     {"repo"?, "files": [{"path", "sha256"}], "dense_precision", "routed_gain", ...}
                                     (routed_gain: the heal gains read back from split 1, or null)
                                     only levels whose splits re-verify; sha256 cached by (size, mtime)

CPU and disk only, no GPU. llama-quantize runs at -t 4 unless S09_THREADS says otherwise.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ops.lib import env  # noqa: E402

LCPP = env.REPO / "llamacpp"
BIN = LCPP / "llama.cpp" / "build-thor" / "bin"
sys.path.insert(1, str(LCPP / "llama.cpp" / "gguf-py"))

OUT = env.WORK / "s09"
NKEEP = int(os.environ.get("V41_NKEEP", 192))      # the arm s10 chose; the chain exports it
SRC_GGUF = OUT / "src" / f"V4.1-1017-REAP{NKEEP}-src.gguf"
IMATRIX = OUT / "imatrix" / f"v41-1017-reap{NKEEP}.imatrix.dat"
NAME = f"DeepSeek-V4.1-Flash-1017-REAP{NKEEP}"
MANIFEST = OUT / "manifest.json"
DENSE_PRECISION = {"attention, compressor, indexer, engram_wkv": "Q8_0",
                   "shared expert, token_embd, output": "Q6_K", "router (ffn_gate_inp)": "BF16 (as stored)",
                   "hyper-connection": "F32", "norms, sinks, biases": "F32",
                   "engram tables": "not in the GGUF (O_DIRECT row files, llamacpp/ENGRAM_ROWFILE.md)"}
SPLIT_MAX = "45G"          # HF's per-file cap is 50 GB; leave headroom for the split header
EOS_ID = 1                 # <｜end▁of▁sentence｜> ends every assistant turn (encoding.py); there is no other EOT

BPW = {"MXFP4": 4.25, "Q3_K": 3.4375, "IQ3_S": 3.4375, "IQ3_XXS": 3.0625, "Q2_K": 2.625,
       "IQ2_S": 2.5, "IQ2_XS": 2.3125, "IQ2_XXS": 2.0625, "Q6_K": 6.5625, "Q8_0": 8.5, "BF16": 16.0,
       "F16": 16.0, "F32": 32.0}
FORBIDDEN_EXPERT = {"Q4_K", "Q5_K", "Q4_0", "Q4_1", "Q5_0", "Q5_1", "IQ4_XS", "IQ4_NL", "Q6_K", "Q8_0"}

LEVELS = {
    "L0": {"target": None, "ladder": ["MXFP4"], "protected": "MXFP4"},
    "L1": {"target": 3.40, "ladder": ["IQ3_XXS", "IQ3_S", "MXFP4"], "protected": "MXFP4"},
    "L2": {"target": 2.85, "ladder": ["IQ2_XS", "IQ2_S", "IQ3_XXS", "IQ3_S"], "protected": "IQ3_S"},
    "L3": {"target": 2.46, "ladder": ["IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ3_XXS"], "protected": "IQ3_XXS"},
}
V41_TEMPLATE = LCPP / "llama.cpp" / "models" / "templates" / "deepseek-ai-DeepSeek-V4.1-Flash.jinja"
TEMPLATE_KEY = "tokenizer.chat_template"
GAIN_KEY = "deepseek41.reap.routed_gain"   # per-layer heal gain on the routed sum (fork patch 0004); absent = 1.0
EXPERT_RE = re.compile(r"^blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$")


def log(msg: str) -> None:
    print(f"[{env.now()}] {msg}", flush=True)


def text_config() -> dict:
    c = json.loads((env.SRC / "config.json").read_text())
    return {**c, **(c.get("text_config") or {})}


def protected_layers(tc: dict) -> list[int]:
    p = {0, 1} | set(tc["engram_layer_ids"]) | set(tc["kv_source_layer_ids"]) | set(tc["dspark_target_layer_ids"])
    return sorted(p)


def fixed_rules() -> list[tuple[str, str]]:
    """(regex, type) for everything that is not a routed expert. First match wins in llama-quantize."""
    return [
        (r"\.hc_|^output_hc_", "F32"),
        (r"_ape\.weight$", "F32"),
        (r"\.ffn_(gate|up|down)_shexp\.weight$", "Q6_K"),
        (r"\.(attn_|indexer)", "Q8_0"),
        (r"\.engram_wkv\.weight$", "Q8_0"),
        (r"\.nextn\.", "Q8_0"),
    ]


def read_source(path: Path) -> list[tuple[str, str, int]]:
    """(name, type, n_elements) per tensor, from the GGUF header (no tensor data is read)."""
    from gguf import GGUFReader
    r = GGUFReader(str(path), "r")
    return [(t.name, t.tensor_type.name, int(t.n_elements)) for t in r.tensors]


def plan(level: str, scores: dict[int, float] | None, src: Path = SRC_GGUF) -> dict:
    spec, tc = LEVELS[level], text_config()
    n_layer = int(tc["num_hidden_layers"])
    prot = protected_layers(tc)
    tensors = read_source(src)
    exp = {}                                  # (layer, group) -> n_elements; group gu = gate+up, d = down
    for name, _, n in tensors:
        m = EXPERT_RE.match(name)
        if m:
            g = "d" if m.group(2) == "down" else "gu"
            exp[(int(m.group(1)), g)] = exp.get((int(m.group(1)), g), 0) + n
    if sorted({L for L, _ in exp}) != list(range(n_layer)):
        raise SystemExit(f"FATAL: source has expert tensors for layers {sorted({L for L, _ in exp})}")
    total = sum(exp.values())
    ladder = spec["ladder"]
    assign = {}
    for (L, g) in exp:
        assign[(L, g)] = ladder.index(spec["protected"]) if L in prot else 0

    def avg() -> float:
        return sum(BPW[ladder[assign[u]]] * exp[u] for u in exp) / total

    if spec["target"] is not None:
        if avg() > spec["target"] + 1e-9:
            raise SystemExit(f"FATAL {level}: protected layers alone exceed the target ({avg():.3f} bpw)")
        # promote one step at a time, most salient first; down_proj ranks just above its own gate/up
        s = scores or {L: 1.0 for L in range(n_layer)}
        order = sorted((u for u in exp if u[0] not in prot), key=lambda u: (-s[u[0]] * (1.1 if u[1] == "d" else 1.0), u))
        top = ladder.index(spec["protected"])
        changed = True
        while changed:
            changed = False
            for u in order:
                if assign[u] + 1 >= top:      # unprotected layers stay strictly below the protected type
                    continue
                assign[u] += 1
                if avg() > spec["target"]:
                    assign[u] -= 1
                    continue
                changed = True
    rules = []
    for (L, g), k in sorted(assign.items()):
        t = ladder[k]
        if t in FORBIDDEN_EXPERT:
            raise SystemExit(f"FATAL: {t} for experts")
        pat = rf"^blk\.{L}\.ffn_(gate|up)_exps\.weight$" if g == "gu" else rf"^blk\.{L}\.ffn_down_exps\.weight$"
        rules.append((pat, t))
    rules += fixed_rules()

    # exact size: every tensor through the same first-match rules llama-quantize applies
    size_bits, by_type = 0.0, {}
    for name, cur, n in tensors:
        t = None
        for pat, typ in rules:
            if re.search(pat, name):
                t = typ
                break
        if t is None:
            if name == "token_embd.weight" or name == "output.weight":
                t = "Q6_K"
            elif not name.endswith(".weight") or "_norm" in name or "ffn_gate_inp" in name or cur in ("F32",) \
                    or "engram_q" in name or "engram_k" in name or "ffn_gate_tid2eid" in name:
                t = cur                       # llama-quant leaves these as stored
            else:
                t = "Q8_0"                    # default ftype: no unnamed weight ever takes an expert type
        size_bits += BPW.get(t, BPW.get(cur, 16.0)) * n
        by_type[t] = by_type.get(t, 0) + n
    out = {
        "level": level, "target_expert_bpw": spec["target"], "expert_bpw": round(avg(), 4),
        "protected_layers": prot, "predicted_bytes": int(size_bits / 8), "predicted_gib": round(size_bits / 8 / 2**30, 2),
        "expert_types": {f"{L}.{g}": ladder[k] for (L, g), k in sorted(assign.items())},
        "elements_by_type": by_type, "rules": rules, "layer_scores": "given" if scores else "uniform",
    }
    return out


def write_rules(p: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{pat}={t.lower()}\n" for pat, t in p["rules"]))


def build_imatrix(keepset_path: Path, bank: Path, out: Path) -> None:
    from ops.lib import imatrix as im
    from ops.lib import imatrix_verify
    from ops.lib import keepset
    keep, fp = keepset.load(keepset_path)     # ascending ids = the master's new expert order (keepset.remap)
    n_layer = int(text_config()["num_hidden_layers"])
    entries, ncall = {}, 1
    for L in range(n_layer):
        f = bank / f"L{L}.npz"
        if not f.exists():
            raise SystemExit(f"FATAL: {f} missing (s03 bank incomplete)")
        b = dict(np.load(f))
        ncall = int(b.get("rows", 1)) or 1
        entries.update(im.layer_entries(L, b, np.asarray(keep[str(L)], dtype=np.int64), ncall))
    out.parent.mkdir(parents=True, exist_ok=True)
    im.write_dat(out, entries, dataset=f"v41-1017 s03 calibration (held-out excluded) keepset={fp}", last_call=ncall)
    if imatrix_verify.main(str(out)) != 0:
        raise SystemExit(f"FATAL: imatrix_verify failed on {out}")
    log(f"imatrix: {len(entries)} entries -> {out}")


def imatrix_keepset(path: Path) -> str | None:
    """The keep-set fingerprint build_imatrix recorded in the .dat's dataset string (None: not recorded)."""
    from ops.lib import imatrix as im
    ds = im.read_dat(path)[2]
    m = re.search(r"keepset=([0-9a-f]{64})", ds)
    return m.group(1) if m else None


def check_imatrix(path: Path, fp: str) -> None:
    got = imatrix_keepset(path) if path.exists() else "missing"
    if got != fp:
        raise SystemExit(f"FATAL: {path} was built for keep-set {got}, the source GGUF is {fp}: rebuild the imatrix "
                         f"(rm ops/logs/done.s09_imatrix)")


def template_problem(kv: dict) -> str | None:
    """None if the GGUF carries exactly the fork's V4.1 chat template."""
    t = kv.get(TEMPLATE_KEY)
    if t == V41_TEMPLATE.read_text(encoding="utf-8"):
        return None
    if not t:
        return f"{TEMPLATE_KEY} missing"
    kind = "the V4 template ('tool_calls>' DSML tags)" if "tool_calls>" in t else "not the V4.1 template"
    return f"{TEMPLATE_KEY} is {kind}: V4.1 tool calls would come back as content (llamacpp/convert_v41_reap.py forces it)"


def gguf_kv(path: Path) -> dict:
    from gguf import GGUFReader
    r = GGUFReader(str(path), "r")
    out = {}
    for k, f in r.fields.items():
        if k.startswith(("tokenizer.ggml.eos", "tokenizer.ggml.eot", "tokenizer.ggml.eom", "general.file_type")) \
                or k == TEMPLATE_KEY \
                or ".reap." in k:
            try:
                out[k] = f.contents()
            except Exception:
                out[k] = None
    return out


def quant(level: str, p: dict, ef_hessians: Path | None, threads: int) -> Path:
    d = OUT / level
    d.mkdir(parents=True, exist_ok=True)
    rules = d / "tensor_types.txt"
    write_rules(p, rules)
    (d / "plan.json").write_text(json.dumps(p, indent=1))
    whole = d / f"{NAME}-{level}.gguf"
    kv = env.Ledger(d / "ledger.json")
    if not kv.done("quantize"):
        if not SRC_GGUF.exists():
            raise SystemExit(f"FATAL: source GGUF {SRC_GGUF} is missing (dropped after the quants?): re-run "
                             f"`llamacpp/CHAIN_STEPS.sh convert` first")
        src_kv = gguf_kv(SRC_GGUF)
        if level != "L0":
            check_imatrix(IMATRIX, src_kv.get("deepseek41.reap.keep_set_sha256"))
        if template_problem(src_kv):
            raise SystemExit(f"FATAL: source GGUF: {template_problem(src_kv)}; re-convert")
        cmd = [str(BIN / "llama-quantize"), "--allow-requantize", "--tensor-type-file", str(rules),
               "--token-embedding-type", "q6_k", "--output-tensor-type", "q6_k",
               "--override-kv", f"tokenizer.ggml.eos_token_id=int:{EOS_ID}",
               "--override-kv", f"tokenizer.ggml.eot_token_id=int:{EOS_ID}"]
        if level != "L0":
            cmd += ["--imatrix", str(IMATRIX)]
        cmd += [str(SRC_GGUF), str(whole), "Q8_0", str(threads)]
        log(" ".join(cmd))
        subprocess.run(cmd, check=True)
        kv.finish("quantize")
    if ef_hessians is not None and level != "L0" and not kv.done("ef"):
        subprocess.run([sys.executable, str(Path(__file__).with_name("s09_ef.py")), "apply", str(SRC_GGUF), str(whole),
                        "--hessians", str(ef_hessians), "--imatrix", str(IMATRIX), "--threads", str(threads)], check=True)
        kv.finish("ef")
    verify_whole(level, whole, p)
    if not kv.done("split"):
        for f in d.glob(f"{NAME}-{level}-0*-of-*.gguf"):
            f.unlink()
        # a level on the USB disk splits onto NVMe and is copied back: gguf-split reading and writing the same
        # ntfs-3g disk ran L2's last split at ~1.5 MB/s. The splits go home to d, so NVMe stays free for rowfile/stage.
        sd = d
        if str(d.resolve()).startswith("/media/"):
            sd = OUT / "splitwork" / level
            sd.mkdir(parents=True, exist_ok=True)
            free = shutil.disk_usage(sd).free
            if free < whole.stat().st_size + 20 * 2**30:
                log(f"split {level}: only {free / 2**30:.0f} GiB free on NVMe, splitting in place")
                sd = d
        for f in sd.glob(f"{NAME}-{level}-0*-of-*.gguf"):
            f.unlink()
        subprocess.run([str(BIN / "llama-gguf-split"), "--split", "--split-max-size", SPLIT_MAX,
                        str(whole), str(sd / f"{NAME}-{level}")], check=True)
        if sd != d:
            verify_splits(level, sd)
            for f in sorted(sd.glob(f"{NAME}-{level}-0*-of-*.gguf")):
                shutil.copyfile(f, d / f".{f.name}.part")
                if (d / f".{f.name}.part").stat().st_size != f.stat().st_size:
                    raise SystemExit(f"FATAL: {f.name}: size differs after the copy to {d}")
                (d / f".{f.name}.part").rename(d / f.name)
                f.unlink()
                log(f"split {level}: {f.name} -> {d}")
            sd.rmdir()
        kv.finish("split")
    verify_splits(level)
    return d


def verify_whole(level: str, path: Path, p: dict) -> None:
    from gguf import GGUFReader
    r = GGUFReader(str(path), "r")
    bad = []
    for t in r.tensors:
        m = EXPERT_RE.match(t.name)
        typ = t.tensor_type.name
        if m:
            want = p["expert_types"][f"{m.group(1)}.{'d' if m.group(2) == 'down' else 'gu'}"]
            if typ != want or typ in FORBIDDEN_EXPERT:
                bad.append(f"{t.name}: {typ} != {want}")
        elif (".hc_" in t.name or t.name.startswith("output_hc_")) and typ != "F32":
            bad.append(f"{t.name}: {typ} != F32")
        elif "_shexp" in t.name and typ != "Q6_K":
            bad.append(f"{t.name}: {typ} != Q6_K")
        elif re.search(r"\.(attn_|indexer)", t.name) and t.name.endswith(".weight") and "_norm" not in t.name \
                and "_ape" not in t.name and "_sinks" not in t.name and typ != "Q8_0":
            bad.append(f"{t.name}: {typ} != Q8_0")
        elif "_sinks" in t.name and typ != "F32":     # 1-D: llama-quantize keeps them F32 (design: sinks kept)
            bad.append(f"{t.name}: {typ} != F32")
        elif "engram_embd" in t.name:
            bad.append(f"{t.name}: engram tables must not be in the GGUF (row files)")
    kv = gguf_kv(path)
    src = gguf_kv(SRC_GGUF)
    for k in ("tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id"):
        if kv.get(k) != EOS_ID:
            bad.append(f"{k} = {kv.get(k)} (want {EOS_ID}): the multi-EOS trap")
    fp_key = "deepseek41.reap.keep_set_sha256"
    if not kv.get(fp_key) or kv.get(fp_key) != src.get(fp_key):
        bad.append(f"keep-set fingerprint {kv.get(fp_key)} != source {src.get(fp_key)}")
    if template_problem(kv):
        bad.append(template_problem(kv))
    if kv.get(GAIN_KEY) != src.get(GAIN_KEY):
        bad.append(f"{GAIN_KEY} {kv.get(GAIN_KEY)} != source {src.get(GAIN_KEY)} (the heal gain must survive quantize)")
    if bad:
        raise SystemExit("FATAL verify:\n  " + "\n  ".join(bad[:40]))
    log(f"verify {level}: {path.name} types, eos/eot={EOS_ID}, keep-set {kv[fp_key][:12]} OK "
        f"({path.stat().st_size / 2**30:.2f} GiB, predicted {p['predicted_gib']})")


def verify_splits(level: str, d: Path | None = None) -> None:
    d = d or OUT / level
    parts = sorted(d.glob(f"{NAME}-{level}-0*-of-*.gguf"))
    if not parts:
        raise SystemExit(f"FATAL: no splits in {d}")
    for f in parts:
        if f.stat().st_size >= 50 * 10**9:
            raise SystemExit(f"FATAL: {f.name} is {f.stat().st_size / 1e9:.1f} GB (HF limit 50 GB)")
    kv = gguf_kv(parts[0])
    if kv.get("tokenizer.ggml.eos_token_id") != EOS_ID or kv.get("tokenizer.ggml.eot_token_id") != EOS_ID:
        raise SystemExit(f"FATAL: split 1 lost the eos/eot override: { {k: v for k, v in kv.items() if k != TEMPLATE_KEY} }")
    if template_problem(kv):
        raise SystemExit(f"FATAL: {parts[0].name}: {template_problem(kv)}")
    log(f"splits {level}: {len(parts)} files, max {max(f.stat().st_size for f in parts) / 1e9:.1f} GB")


def sha256_cached(f: Path, cache: dict) -> str:
    import hashlib
    st = f.stat()
    key = f"{f.name}:{st.st_size}:{int(st.st_mtime)}"
    if key not in cache:
        h = hashlib.sha256()
        with open(f, "rb") as fh:
            while b := fh.read(1 << 24):
                h.update(b)
                os.posix_fadvise(fh.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        cache[key] = h.hexdigest()
    return cache[key]


def write_manifest(levels: list[str], repo: str | None) -> dict:
    """The s12 contract: {"repo"?, "files": [{"path", "sha256"}], "dense_precision"?}. s12 uploads the files
    flat by name, so the split names (which carry the level) are unique. Only levels whose splits exist and
    re-verify are listed; a level without a teacher_kld.json is listed with kld = null."""
    cache_f = OUT / "sha256_cache.json"
    cache = json.loads(cache_f.read_text()) if cache_f.exists() else {}
    files, lv_out, fps, gains = [], {}, set(), set()
    for lv in levels:
        parts = sorted((OUT / lv).glob(f"{NAME}-{lv}-0*-of-*.gguf"))
        if not parts:
            log(f"manifest: {lv} has no splits, not listed")
            continue
        verify_splits(lv)
        kv1 = gguf_kv(parts[0])
        fp = kv1.get("deepseek41.reap.keep_set_sha256")
        if not fp:
            raise SystemExit(f"FATAL: {parts[0].name} carries no keep-set fingerprint")
        fps.add(fp)
        g = kv1.get(GAIN_KEY)
        gains.add(None if g is None else tuple(float(x) for x in g))
        for f in parts:
            files.append({"path": str(f), "sha256": sha256_cached(f, cache), "bytes": f.stat().st_size, "level": lv})
            cache_f.write_text(json.dumps(cache))
        plan_f, kld_f = OUT / lv / "plan.json", OUT / lv / "teacher_eval" / "teacher_kld.json"
        pl = json.loads(plan_f.read_text()) if plan_f.exists() else {}
        lv_out[lv] = {"expert_bpw": pl.get("expert_bpw"), "bytes": sum(f.stat().st_size for f in parts),
                      "first_split": parts[0].name, "n_splits": len(parts),
                      "kld": json.loads(kld_f.read_text())["stats"] if kld_f.exists() else None}
    if not files:
        raise SystemExit("FATAL: no verified GGUF level to put in the manifest")
    if len(fps) != 1:
        raise SystemExit(f"FATAL: levels disagree on the keep-set: {sorted(fps)}")
    if len(gains) != 1:
        raise SystemExit(f"FATAL: levels disagree on {GAIN_KEY}: {sorted(gains, key=str)}")
    gain = gains.pop()
    fork = subprocess.run(["git", "-C", str(LCPP / "llama.cpp"), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    m = {"files": files, "dense_precision": DENSE_PRECISION, "keepset_sha256": fps.pop(), "n_routed_experts": NKEEP,
         "levels": lv_out, "llama_cpp": {"pr": 28696, "pr_head": "2076412598e99127ea2554467b2b7530856895d3",
                                         "fork_head": fork, "patches": sorted(p.name for p in (LCPP / "patches").glob("*.patch"))},
         "routed_gain": None if gain is None else list(gain),
         "eos_token_id": EOS_ID, "eot_token_id": EOS_ID, "engram": "llamacpp/ENGRAM_ROWFILE.md", "ts": env.now()}
    if repo:
        m["repo"] = repo
    tmp = MANIFEST.with_suffix(".tmp")
    tmp.write_text(json.dumps(m, indent=1))
    os.replace(tmp, MANIFEST)
    log(f"manifest: {len(files)} files over {sorted(lv_out)} -> {MANIFEST}")
    return m


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["plan", "imatrix", "quant", "verify", "manifest"])
    ap.add_argument("level", nargs="?", choices=sorted(LEVELS))
    ap.add_argument("--src", type=Path, default=SRC_GGUF)
    ap.add_argument("--layer-scores", type=Path, help="JSON {layer: sensitivity}; higher keeps more bits")
    ap.add_argument("--keepset", type=Path, help="the master's reap_keepset.json (imatrix)")
    ap.add_argument("--levels", nargs="+", default=["L0", "L1", "L2", "L3"])
    ap.add_argument("--repo", help="HF repo id for s12 (default: s12 derives it)")
    ap.add_argument("--bank", type=Path, default=env.WORK / "s03" / "bank")
    ap.add_argument("--ef-hessians", type=Path, help="enable GPTQ-style error feedback (s09_ef.py)")
    ap.add_argument("--check", action="store_true", help="imatrix: exit 0 if the existing .dat matches --keepset, 3 if not")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("S09_THREADS", 4)))
    a = ap.parse_args()
    scores = None
    if a.layer_scores:
        scores = {int(k): float(v) for k, v in json.loads(a.layer_scores.read_text()).items()}
    if a.cmd == "imatrix":
        if not a.keepset:
            ap.error("imatrix needs --keepset")
        if a.check:
            from ops.lib import keepset
            fp = keepset.load(a.keepset)[1]
            got = imatrix_keepset(IMATRIX) if IMATRIX.exists() else None
            log(f"imatrix check: {IMATRIX} keep-set {got}, master {fp}: {'OK' if got == fp else 'STALE'}")
            return 0 if got == fp else 3
        build_imatrix(a.keepset, a.bank, IMATRIX)
        return 0
    if a.cmd == "manifest":
        write_manifest(a.levels, a.repo)
        return 0
    if not a.level:
        ap.error("level required")
    if not a.src.exists():
        raise SystemExit(f"FATAL: source GGUF {a.src} is missing (CHAIN_STEPS drop_source removes it once the quants "
                         f"are done): re-run `llamacpp/CHAIN_STEPS.sh convert` first")
    p = plan(a.level, scores, a.src)
    if a.cmd == "plan":
        p2 = dict(p)
        p2.pop("rules")
        print(json.dumps(p2, indent=1))
        write_rules(p, OUT / a.level / "tensor_types.txt")
        return 0
    if a.cmd == "quant":
        quant(a.level, p, a.ef_hessians, a.threads)
        return 0
    verify_whole(a.level, OUT / a.level / f"{NAME}-{a.level}.gguf", p)
    verify_splits(a.level)
    return 0


if __name__ == "__main__":
    sys.exit(main())
