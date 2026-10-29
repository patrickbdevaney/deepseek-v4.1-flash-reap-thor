#!/usr/bin/env python3
"""D2 drafter-training prompts from the calibration corpus -> v41_decode --trace-gen jsonl.

    .venv/bin/python ops/zenith/d2_prompts.py OUTDIR [--train 480] [--eval 48] [--seed 0]

One prompt per corpus DOCUMENT (rows are packed docs), never across a doc boundary, from permissively
licensed sources only (odc-by / MIT / Apache-2.0 / CC-BY-4.0 / BSD; no share-alike, no unknown or custom
terms), text buckets only (vision rows carry image tokens). Eval prompts come from the corpus's held-out
rows, train from the rest, so the two never share a document. Balanced across buckets; modes:
  raw       the doc's first 128..1024 tokens as ids (base-LM continuation)
  chat      the doc's first <= 1024 tokens decoded, as the user message (the served chat template)
  thinking  the same under the thinking template
The targets are the served model's own greedy continuations (v41_decode), never text from this script.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CORPUS = Path.home() / "models/v41-work/corpus"
TOKENIZER = Path.home() / "models/DeepSeek-V4.1-Flash-1017/tokenizer.json"
OK_LICENCES = {"odc-by", "mit", "apache-2.0", "cc-by-4.0", "bsd-3-clause", "mit (model output)"}
BUCKETS = ["general", "multilingual", "agentic", "code", "math", "science", "finance", "longctx"]
MODES = [("raw", 0.25), ("chat", 0.55), ("thinking", 0.20)]


def docs(rows, sources):
    """(bucket, heldout, source, start, length) for every doc of every text row with an OK licence."""
    for r in rows:
        if r["bucket"] not in BUCKETS or sum(d["tokens"] for d in r["docs"]) != r["length"]:
            continue
        o = r["offset"]
        for d in r["docs"]:
            lic = str(sources.get(d["source"], {}).get("licence_live"))
            if lic in OK_LICENCES:
                yield r["bucket"], r["heldout"], d["source"], o, d["tokens"]
            o += d["tokens"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--train", type=int, default=480)
    ap.add_argument("--eval", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(str(TOKENIZER))
    rows = [json.loads(l) for l in open(CORPUS / "rows.jsonl")]
    sources = json.load(open(CORPUS / "sources.json"))
    tokens = np.load(CORPUS / "tokens.npy", mmap_mode="r")
    types = np.load(CORPUS / "types.npy", mmap_mode="r")
    rng = np.random.default_rng(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    pool = {}
    for b, held, src, s, n in docs(rows, sources):
        if n >= 48 and not (types[s:s + min(n, 1024)] != -1).any():
            pool.setdefault((held, b), []).append((src, s, n))
    for split, held, total in (("train", False, a.train), ("eval", True, a.eval)):
        lines, per = [], {}
        quota = {b: total // len(BUCKETS) + (i < total % len(BUCKETS)) for i, b in enumerate(BUCKETS)}
        for b in BUCKETS:
            cand = pool.get((held, b), [])
            idx = rng.permutation(len(cand))
            k = 0
            for j in idx:
                if k >= quota[b]:
                    break
                src, s, n = cand[j]
                mode = MODES[int(np.searchsorted(np.cumsum([w for _, w in MODES]), rng.random()))][0]
                ids = np.asarray(tokens[s:s + min(n, 1024)]).astype(np.int64)
                if mode == "raw":
                    if n < 136:
                        mode = "chat"
                    else:
                        L = int(rng.integers(128, min(1024, n - 8) + 1))
                        line = {"id": f"{split}-{b}-{k}", "domain": b, "mode": "raw", "source": src, "ids": ids[:L].tolist()}
                if mode != "raw":
                    text = tk.decode(ids.tolist(), skip_special_tokens=True).strip()
                    if len(text) < 40:
                        continue
                    line = {"id": f"{split}-{b}-{k}", "domain": b, "mode": mode, "source": src, "content": text}
                lines.append(line); k += 1
                per[(b, mode)] = per.get((b, mode), 0) + 1
        rng.shuffle(lines)                    # interleave domains: a partial run is still balanced
        with open(out / f"prompts_{split}.jsonl", "w") as f:
            for l in lines:
                f.write(json.dumps(l, ensure_ascii=False) + "\n")
        print(f"{split}: {len(lines)} prompts", {f"{b}/{m}": c for (b, m), c in sorted(per.items())})


if __name__ == "__main__":
    main()
