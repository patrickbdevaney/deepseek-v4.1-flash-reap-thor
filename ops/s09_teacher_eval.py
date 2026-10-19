#!/usr/bin/env python3
"""s09c: paired GGUF-vs-teacher statistics on the frozen held-out split, scored against s03's cached teacher.

  prep       [--max-tokens N]  held-out, text-only rows -> raw arrays for llamacpp/tools/v41-teacher-kld (CPU)
  summarize  LEVEL GGUF        per-token kl/top1/s_nll -> teacher_kld.json, with row-bootstrap 95% CIs

The GPU run between the two is a CHAIN_STEPS.sh step. These two subcommands load no model.

Measures (pre-registered in PLAN):
  * top-32 KL with a tail bucket;
  * top-1 agreement with the teacher's argmax;
  * dNLL = student NLL - teacher NLL on the next token.
All are paired per token, with CIs from resampling rows. The served identity goes into the result:
GGUF path, size, keep-set fingerprint and level. One result therefore cannot silently stand in for
another arm (eval-orphaned-server lesson).
Vision rows are excluded because this harness has no mmproj path. Their numbers come from s10.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ops.lib import env  # noqa: E402

CORPUS = env.WORK / "corpus"
TEACHER = env.WORK / "s03" / "teacher"
PREP = env.WORK / "s09" / "teacher_eval"


def prep(max_tokens: int | None) -> None:
    rows = [json.loads(l) for l in open(CORPUS / "rows.jsonl")]
    tokens = np.load(CORPUS / "tokens.npy", mmap_mode="r")
    types = np.load(CORPUS / "types.npy", mmap_mode="r")
    valid = np.load(CORPUS / "valid.npy", mmap_mode="r")
    lp = np.load(TEACHER / "top32_logprob.npy", mmap_mode="r")
    ti = np.load(TEACHER / "top32_ids.npy", mmap_mode="r")
    nll = np.load(TEACHER / "nll_next.npy", mmap_mode="r")
    keep, n = [], 0
    for r in rows:
        if not r["heldout"]:
            continue
        s = slice(r["offset"], r["offset"] + r["length"])
        if (types[s] != -1).any() or not valid[s].all():
            continue
        if not np.isfinite(lp[s.start]).all():          # the head pass never reached this row
            raise SystemExit(f"FATAL: teacher cache has no logits for held-out row {r['row']}")
        keep.append(r)
        n += r["length"]
        if max_tokens and n >= max_tokens:
            break
    if not keep:
        raise SystemExit("FATAL: no text-only held-out rows")
    PREP.mkdir(parents=True, exist_ok=True)
    N = sum(r["length"] for r in keep)
    out_rows = np.empty((len(keep), 2), np.int64)
    tok = np.empty(N, np.int32)
    tlp = np.empty((N, 32), np.float32)
    tid = np.empty((N, 32), np.int32)
    tnll = np.empty(N, np.float32)
    o = 0
    for i, r in enumerate(keep):
        s = slice(r["offset"], r["offset"] + r["length"])
        L = r["length"]
        out_rows[i] = (o, L)
        tok[o:o + L], tlp[o:o + L], tid[o:o + L], tnll[o:o + L] = tokens[s], lp[s], ti[s], nll[s]
        o += L
    out_rows.tofile(PREP / "rows.i64")
    tok.tofile(PREP / "tokens.i32")
    tlp.tofile(PREP / "t_lp.f32")
    tid.tofile(PREP / "t_id.i32")
    tnll.tofile(PREP / "t_nll.f32")
    (PREP / "meta.json").write_text(json.dumps({
        "rows": [r["row"] for r in keep], "buckets": [r["bucket"] for r in keep], "tokens": int(N),
        "corpus_manifest": json.loads((CORPUS / "manifest.json").read_text()).get("spec_hash")}, indent=1))
    print(f"[{env.now()}] prep: {len(keep)} held-out text rows, {N:,} tokens -> {PREP}")


def boot(per_row: np.ndarray, weights: np.ndarray, n: int = 2000) -> list[float]:
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(per_row), (n, len(per_row)))
    est = (per_row[idx] * weights[idx]).sum(1) / weights[idx].sum(1)
    return [float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5))]


def summarize(level: str, gguf: Path, run_dir: Path) -> dict:
    from gguf import GGUFReader
    meta = json.loads((PREP / "meta.json").read_text())
    rows = np.fromfile(PREP / "rows.i64", np.int64).reshape(-1, 2)
    kl = np.fromfile(run_dir / "kl.f32", np.float32)
    top1 = np.fromfile(run_dir / "top1.u8", np.uint8).astype(np.float64)
    snll = np.fromfile(run_dir / "s_nll.f32", np.float32)
    tnll = np.fromfile(PREP / "t_nll.f32", np.float32)
    if not (len(kl) == len(top1) == len(snll) == len(tnll) == meta["tokens"]):
        raise SystemExit("FATAL: run output does not match the prepared token count")
    if not np.isfinite(kl).all():
        raise SystemExit(f"FATAL: {int((~np.isfinite(kl)).sum())} tokens have no KL (incomplete run)")
    r = GGUFReader(str(gguf), "r")
    fp = r.fields.get("deepseek41.reap.keep_set_sha256")
    ident = {"level": level, "gguf": str(gguf), "bytes": gguf.stat().st_size,
             "keep_set_sha256": fp.contents() if fp else None}
    if not ident["keep_set_sha256"]:
        raise SystemExit("FATAL: served GGUF carries no keep-set fingerprint")
    stats = {}
    for name, v in (("kl_top32", kl.astype(np.float64)), ("top1", top1), ("dnll", (snll - tnll).astype(np.float64))):
        per_row, w = [], []
        for s, L in rows:
            x = v[s:s + L]
            x = x[np.isfinite(x)]
            per_row.append(x.mean() if len(x) else 0.0)
            w.append(len(x))
        per_row, w = np.array(per_row), np.array(w, np.float64)
        stats[name] = {"mean": float((per_row * w).sum() / w.sum()), "ci95": boot(per_row, w)}
    by_bucket = {}
    for b in sorted(set(meta["buckets"])):
        sel = [i for i, x in enumerate(meta["buckets"]) if x == b]
        t = np.concatenate([np.arange(rows[i, 0], rows[i, 0] + rows[i, 1]) for i in sel])
        d = (snll[t] - tnll[t])
        by_bucket[b] = {"tokens": int(len(t)), "kl_top32": float(kl[t].mean()), "top1": float(top1[t].mean()),
                        "dnll": float(np.nanmean(d))}
    out = {"identity": ident, "tokens": meta["tokens"], "rows": len(rows), "stats": stats, "by_bucket": by_bucket,
           "corpus_spec_hash": meta["corpus_manifest"], "measured": env.now()}
    (run_dir / "teacher_kld.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["stats"], indent=1))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["prep", "summarize"])
    ap.add_argument("level", nargs="?")
    ap.add_argument("gguf", nargs="?", type=Path)
    ap.add_argument("--run-dir", type=Path)
    ap.add_argument("--max-tokens", type=int)
    a = ap.parse_args()
    if a.cmd == "prep":
        prep(a.max_tokens)
        return 0
    if not (a.level and a.gguf):
        ap.error("summarize LEVEL GGUF")
    summarize(a.level, a.gguf, a.run_dir or env.WORK / "s09" / a.level / "teacher_eval")
    return 0


if __name__ == "__main__":
    sys.exit(main())
