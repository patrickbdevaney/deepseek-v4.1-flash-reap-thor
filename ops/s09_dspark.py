#!/usr/bin/env python3
"""s09 DSpark sidecar: the checkpoint's own drafter (mtp.0..2) as an opt-in llama.cpp `dspark-*.gguf` at the GGUF
repo root, where `-hf ... --spec-type draft-dspark` finds it (common/download.cpp find_best_dspark). Gate:
ops/s10_gates.json "dspark_gguf" (pre-registered 2026-10-22).

The drafter is not pruned: it has its own 128 routed experts (top-3) and borrows the target's token embeddings and
output head, which REAP leaves alone. It reads the attention input of target layers 37-39 (inference/model.py), so
llamacpp patch 0005 writes target_layers as configured, not as V4's layer + 1.

  convert            CPU       base checkpoint mtp.0..2 (shards 44-46) -> $V41_WORK/s09/dspark/FILE (patch 0005)
  verify             CPU+net   tensors and drafter keys vs the pinned reference drafter GGUF -> source_check.json
  pick               CPU       the level to bench: largest passing level that fits beside the drafter -> prints it
  bench LEVEL        GPU       LEVEL served without and with the drafter, greedy, on confirm-split prompts -> bench.json
  gate               CPU       dspark_gguf over source_check.json + bench.json -> gate.json (exit 0 either way)

Nothing here runs beside another model loader: bench is a chain2 soft step under quiet/reclaim, and it refuses to
start a server unless level + drafter + reserve fit MemAvailable (an overcommitted Thor hard-reboots).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ops.lib import env, gates as G  # noqa: E402

LCPP = env.REPO / "llamacpp"
BIN = LCPP / "llama.cpp" / "build-thor" / "bin"
sys.path.insert(1, str(LCPP / "llama.cpp" / "gguf-py"))

STAGE = "s09_dspark"
S09 = env.WORK / "s09"
OUT = S09 / "dspark"
FILE = "dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf"   # unmodified drafter: no REAP tag in the name
NKEEP = int(os.environ.get("V41_NKEEP", 192))
NAME = f"DeepSeek-V4.1-Flash-1017-REAP{NKEEP}"
# the reference drafter, pinned (repo revision and LFS sha256 checked 2026-10-22)
REF_REPO = "unsloth/DeepSeek-V4.1-Flash-1017-GGUF"
REF_REV = "60ad3651641b5c97679c807e064ab1d7a3ca1455"
REF_SHA = "6ba6ac1b331c540691a209e7fc97dec2a24e295acb1e755c84c7d24a4373927a"
REF_DIR = Path(os.environ.get("S09_DSPARK_REF_DIR", "/media/patrickd/Seagate Portable Drive/v41_staging/dspark_ref"))
PROMPT_TOKENS = 1024
PORT = 18088
GIB = 1 << 30


def log(msg: str) -> None:
    env.log(msg, STAGE)


def gate_cfg() -> dict:
    return G.load()[0]["dspark_gguf"]


# ------------------------------------------------------------------------------------------------ convert
def convert() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / FILE
    if out.exists() and (OUT / "convert.json").exists():
        log(f"convert: {out} already written"); return
    tmp = OUT / (FILE + ".tmp")
    cmd = [sys.executable, str(LCPP / "llama.cpp" / "convert_hf_to_gguf.py"), str(env.SRC), "--dspark",
           "--target-model-dir", str(env.SRC), "--outfile", str(tmp)]
    log("convert: " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    os.replace(tmp, out)
    env.atomic_json(OUT / "convert.json", {"file": str(out), "bytes": out.stat().st_size, "source": str(env.SRC),
                                           "revision": env.REVISION, "ts": env.now()})
    log(f"convert: {out} ({out.stat().st_size / GIB:.2f} GiB)")


# ------------------------------------------------------------------------------------------------ verify
def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        while b := fh.read(1 << 24):
            h.update(b)
    return h.hexdigest()


def fetch_ref() -> Path:
    from huggingface_hub import hf_hub_download
    REF_DIR.mkdir(parents=True, exist_ok=True)
    p = Path(hf_hub_download(REF_REPO, FILE, revision=REF_REV, local_dir=str(REF_DIR)))
    got = _sha256(p)
    if got != REF_SHA:
        raise SystemExit(f"reference drafter sha256 {got[:16]} != pinned {REF_SHA[:16]}")
    return p


def _kv(reader) -> dict:
    out = {}
    for name, f in reader.fields.items():
        if not f.types:
            continue
        try:
            v = f.contents()
        except Exception:  # noqa: BLE001 - an unreadable field is compared by its raw parts
            v = [bytes(p) for p in f.parts]
        out[name] = v
    return out


def bytes_equal(x, y, chunk: int = 1 << 26) -> bool:
    """Byte equality of two (memmapped) tensors, chunked so no full-size temporary is built."""
    a, b = np.asarray(x).reshape(-1).view(np.uint8), np.asarray(y).reshape(-1).view(np.uint8)
    if a.size != b.size:
        return False
    return all(np.array_equal(a[i:i + chunk], b[i:i + chunk]) for i in range(0, a.size, chunk))


def compare_files(ours: Path, ref: Path, max_rel_rmse: float) -> dict:
    """Every reference tensor: same type and shape; bytes equal, or (Q8_0 only) dequantized rel RMSE <= max. Every
    reference key outside general.* and the chat template: equal in ours (extra keys of ours are allowed)."""
    from gguf import GGUFReader, GGMLQuantizationType
    from gguf.quants import dequantize
    a, b = GGUFReader(str(ours)), GGUFReader(str(ref))
    ta, tb = {t.name: t for t in a.tensors}, {t.name: t for t in b.tensors}
    why, rows = [], []
    for name in sorted(set(ta) | set(tb)):
        x, y = ta.get(name), tb.get(name)
        if x is None or y is None:
            why.append(f"tensor {name} only in {'reference' if x is None else 'ours'}"); continue
        rec = {"name": name, "type": y.tensor_type.name, "shape": [int(d) for d in y.shape]}
        if x.tensor_type != y.tensor_type or list(x.shape) != list(y.shape):
            why.append(f"tensor {name}: ours {x.tensor_type.name} {list(x.shape)} vs ref {y.tensor_type.name} {list(y.shape)}")
            rows.append({**rec, "equal": False}); continue
        eq = bytes_equal(x.data, y.data)
        rec["equal"] = eq
        if not eq:
            if y.tensor_type != GGMLQuantizationType.Q8_0:
                why.append(f"tensor {name} ({y.tensor_type.name}) differs from the reference bytes")
            else:
                dx = dequantize(np.asarray(x.data), x.tensor_type).astype(np.float64)
                dy = dequantize(np.asarray(y.data), y.tensor_type).astype(np.float64)
                rel = float(np.sqrt(np.mean((dx - dy) ** 2)) / max(np.sqrt(np.mean(dy ** 2)), 1e-30))
                rec["rel_rmse"] = rel
                if not rel <= max_rel_rmse:
                    why.append(f"tensor {name} Q8_0 rel RMSE {rel:.2e} > {max_rel_rmse}")
        rows.append(rec)
    ka, kb = _kv(a), _kv(b)
    skip = ("general.", "GGUF.", "tokenizer.chat_template")
    for k, v in kb.items():
        if k.startswith(skip):
            continue
        if k not in ka:
            why.append(f"key {k} missing (reference {str(v)[:80]})")
        elif ka[k] != v:
            why.append(f"key {k}: ours {str(ka[k])[:80]} vs reference {str(v)[:80]}")
    return {"pass": not why, "why": why, "tensors": len(tb), "byte_identical": sum(r.get("equal", False) for r in rows),
            "rows": rows, "extra_keys": sorted(k for k in ka if k not in kb and not k.startswith(skip))}


def verify() -> None:
    ours = OUT / FILE
    if not ours.exists():
        raise SystemExit(f"no {ours}: run convert first")
    ref = fetch_ref()
    res = compare_files(ours, ref, gate_cfg()["source_q8_max_rel_rmse"])
    res.update({"ours": str(ours), "ours_sha256": _sha256(ours), "reference": f"{REF_REPO}@{REF_REV}/{FILE}",
                "reference_sha256": REF_SHA, "ts": env.now()})
    env.atomic_json(OUT / "source_check.json", res)
    log(f"verify: {'PASS' if res['pass'] else 'FAIL'}: {res['byte_identical']}/{res['tensors']} tensors byte-identical"
        + ("" if res["pass"] else f"; {res['why'][:3]}"))


# ------------------------------------------------------------------------------------------------ pick + bench
def level_files(lv: str) -> list[Path]:
    return sorted((S09 / lv).glob(f"{NAME}-{lv}-0*-of-*.gguf"))


def level_bytes(lv: str) -> int:
    return sum(f.stat().st_size for f in level_files(lv))


def pick(passing: list[str], avail_gib: float, draft_bytes: int, reserve_gib: float,
         sizes: dict[str, int]) -> str | None:
    """Largest passing level (lowest L-number) whose bytes + drafter + reserve fit avail."""
    for lv in sorted(passing):
        if sizes.get(lv) and sizes[lv] + draft_bytes + reserve_gib * GIB <= avail_gib * GIB:
            return lv
    return None


def _post(path: str, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def _get(path: str, timeout: float = 10) -> dict:
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}{path}", timeout=timeout) as r:
        return json.load(r)


def prompts(n: int) -> list[list[int]]:
    """n confirm-split rows (s09 teacher_eval prep, held out from every tuning decision), evenly spaced over the
    row list so the domain buckets are mixed, each cut to its first PROMPT_TOKENS tokens."""
    prep = S09 / "teacher_eval"
    rows = np.fromfile(prep / "rows.i64", np.int64).reshape(-1, 2)
    tok = np.fromfile(prep / "tokens.i32", np.int32)
    ok = [i for i in range(len(rows)) if rows[i, 1] >= PROMPT_TOKENS]
    if len(ok) < n:
        raise SystemExit(f"only {len(ok)} confirm rows have {PROMPT_TOKENS} tokens; need {n}")
    pick_ = [ok[int(round(j * (len(ok) - 1) / max(n - 1, 1)))] for j in range(n)]
    return [tok[rows[i, 0]: rows[i, 0] + PROMPT_TOKENS].tolist() for i in pick_]


def serve_and_run(model: Path, extra: list[str], ps: list[list[int]], c: dict, tag: str) -> list[dict]:
    log_path = env.LOGS / f"s09_dspark_bench_{tag}.log"
    cmd = [str(BIN / "llama-server"), "-m", str(model), "-c", str(c["ctx"]), "-np", "1", "-ngl", "999",
           "--port", str(PORT), "--host", "127.0.0.1", *extra]
    log(f"bench {tag}: " + " ".join(cmd))
    with open(log_path, "w") as lf:
        # exec'd directly (no shell), so the pid we kill IS the server (eval-orphaned-server lesson)
        # the GGUF has no Engram tables: the server reads them from the NVMe row files (s09_engram_rowfile build)
        srv_env = {**os.environ, "LLAMA_ENGRAM_ROWFILE_DIR": os.environ.get("LLAMA_ENGRAM_ROWFILE_DIR",
                                                                           str(env.WORK / "engram_rows"))}
        srv = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True,
                               env=srv_env)
    try:
        for _ in range(180):
            if srv.poll() is not None:
                raise SystemExit(f"bench {tag}: server exited rc={srv.returncode} (see {log_path})")
            try:
                if _get("/health").get("status") == "ok":
                    break
            except Exception:  # noqa: BLE001 - not up yet
                pass
            time.sleep(10)
        else:
            raise SystemExit(f"bench {tag}: server not healthy after 30 min")
        served = _get("/props", 60)["model_path"]
        if Path(served).resolve() != model.resolve():
            raise SystemExit(f"bench {tag}: served {served}, expected {model}")
        out = []
        for i, p in enumerate(ps):
            r = _post("/completion", {"prompt": p, "n_predict": c["n_predict"], "temperature": 0, "cache_prompt": False,
                                      "ignore_eos": True, "return_tokens": True}, timeout=1800)
            t = r.get("timings", {})
            out.append({"tokens": r["tokens"], "tps": t.get("predicted_per_second"), "n": t.get("predicted_n"),
                        "draft_n": t.get("draft_n", 0), "draft_accepted": t.get("draft_n_accepted", 0)})
            log(f"bench {tag}: prompt {i + 1}/{len(ps)} {out[-1]['tps']:.2f} tok/s"
                + (f", accepted {out[-1]['draft_accepted']}/{out[-1]['draft_n']}" if out[-1]["draft_n"] else ""))
        return out
    finally:
        srv.terminate()
        try:
            srv.wait(60)
        except subprocess.TimeoutExpired:
            srv.kill(); srv.wait()


def common_prefix(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def bench(lv: str) -> None:
    c = gate_cfg()
    draft = OUT / FILE
    model = level_files(lv)[0]
    if "/media/" in str(model.resolve()):
        raise SystemExit(f"bench {lv}: the level is on the USB disk; stage it to NVMe first")
    need = level_bytes(lv) + draft.stat().st_size + c["reserve_gib"] * GIB
    avail = env.reclaim(STAGE)
    if need > avail * GIB:
        raise SystemExit(f"bench {lv}: needs {need / GIB:.1f} GiB (level + drafter + reserve), {avail:.1f} available")
    ps = prompts(c["n_prompts"])
    plain = serve_and_run(model, [], ps, c, f"{lv}_plain")
    env.reclaim(STAGE, expect_gib=100)
    spec = serve_and_run(model, ["-md", str(draft), "-ngld", "999", "--spec-type", "draft-dspark",
                                 "--spec-draft-n-max", str(c["draft_n_max"])], ps, c, f"{lv}_spec")
    env.reclaim(STAGE, expect_gib=100)
    per = []
    for a, b in zip(plain, spec):
        per.append({"identical": a["tokens"] == b["tokens"], "common_prefix": common_prefix(a["tokens"], b["tokens"]),
                    "tps_plain": a["tps"], "tps_spec": b["tps"], "draft_n": b["draft_n"], "draft_accepted": b["draft_accepted"]})
    env.atomic_json(OUT / "bench.json", {"level": lv, "model": str(model), "drafter": str(draft),
                                         "level_gib": level_bytes(lv) / GIB, "drafter_gib": draft.stat().st_size / GIB,
                                         "prompt_tokens": PROMPT_TOKENS, **{k: c[k] for k in ("n_prompts", "n_predict",
                                         "ctx", "draft_n_max")}, "per_prompt": per, "ts": env.now()})
    log(f"bench {lv}: written {OUT / 'bench.json'}")


# ------------------------------------------------------------------------------------------------ gate
def gmean(xs: list[float]) -> float:
    return float(np.exp(np.mean(np.log(xs))))


def evaluate(src: dict | None, b: dict | None, c: dict) -> dict:
    why = []
    if src is None:
        why.append("no source check")
    elif not src["pass"]:
        why += [f"source check: {w}" for w in src["why"][:5]]
    if b is None:
        why.append("not benched (no level fits beside the drafter, or the bench failed)")
        return {"pass": False, "why": why}
    per = b["per_prompt"]
    ident = sum(p["identical"] for p in per) / len(per)
    minpre = min(p["common_prefix"] for p in per)
    dn, da = sum(p["draft_n"] for p in per), sum(p["draft_accepted"] for p in per)
    acc = da / dn if dn else 0.0
    if not ident >= c["min_identical_frac"]:
        why.append(f"identical outputs {ident:.2f} < {c['min_identical_frac']}")
    if not minpre >= c["min_common_prefix_tokens"]:
        why.append(f"shortest common prefix {minpre} tokens < {c['min_common_prefix_tokens']}")
    if not acc >= c["min_accept_rate"]:
        why.append(f"draft acceptance {acc:.3f} < {c['min_accept_rate']}")
    tp, ts = [p["tps_plain"] for p in per], [p["tps_spec"] for p in per]
    return {"pass": not why, "why": why, "level": b["level"], "identical_frac": ident, "min_common_prefix": minpre,
            "accept_rate": acc, "tps_plain": gmean(tp), "tps_spec": gmean(ts), "speedup": gmean(ts) / gmean(tp),
            "level_gib": b["level_gib"], "drafter_gib": b["drafter_gib"], "n_prompts": len(per),
            "n_predict": b["n_predict"], "draft_n_max": b["draft_n_max"], "ctx": b["ctx"]}


def gate() -> None:
    j = lambda p: json.loads(p.read_text()) if p.exists() else None  # noqa: E731
    res = evaluate(j(OUT / "source_check.json"), j(OUT / "bench.json"), gate_cfg())
    res["gates_sha256"] = G.load()[1]
    env.atomic_json(OUT / "gate.json", res)
    log(f"gate: {'PASS' if res['pass'] else 'FAIL'} {res.get('why') or ''}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("convert"); sp.add_parser("verify"); sp.add_parser("gate")
    p = sp.add_parser("pick"); p.add_argument("--passing", nargs="+", required=True)
    p = sp.add_parser("bench"); p.add_argument("level")
    a = ap.parse_args()
    if a.cmd == "convert":
        convert()
    elif a.cmd == "verify":
        verify()
    elif a.cmd == "gate":
        gate()
    elif a.cmd == "pick":
        sizes = {lv: level_bytes(lv) for lv in a.passing}
        lv = pick(a.passing, env.reclaim(STAGE), (OUT / FILE).stat().st_size, gate_cfg()["reserve_gib"], sizes)
        print(lv or "")
    elif a.cmd == "bench":
        bench(a.level)


if __name__ == "__main__":
    main()
