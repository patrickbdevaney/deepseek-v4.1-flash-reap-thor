#!/usr/bin/env python3
"""s02: build the token-balanced calibration corpus (spec and final mix: ops/corpus_spec.py).

Rows. Eight pools are packed into 8,192-token rows, one pool per row. The `longctx` pool holds
whole documents of 32k-128k tokens, one per row and never concatenated, so rows are RAGGED: all
per-token arrays are flat over the corpus and rows.jsonl gives each row's offset and length.

Output, under $V41_WORK/corpus/ (default ~/models/v41-work/corpus):
  tokens.npy   [T] uint32   token ids (vocab 129,280 does not fit uint16)
  types.npy    [T] int8     -1 text, 0..3 image_processor IMAGE_START/IMAGE/NEW_LINE/END
  valid.npy    [T] uint8    1 for real tokens, 0 for the tail pad of a vision row
  rows.jsonl   one record per row IN VISIT ORDER: row, bucket, offset, length, heldout, docs, images
  sources.json per source: declared + live licence, generator, status, rows read / kept, tokens
  manifest.json achieved token mix, counts, spec hash; written last (it marks completion)
  images/      original image bytes for vision rows (s03 re-reads them with load_image)

Rows are written in the smooth interleaved order of corpus_spec.interleave(). The last 10% of each
pool's rows are the frozen held-out split: s03 runs them through the teacher for top-32 logits
and routing but never lets them touch a saliency accumulator.

Packing. Text documents are packed back to back in 8k pools (a document that overflows continues
on the next row of the same pool). Vision samples are never split: an image span must sit inside
one prefill, so a sample that does not fit pads the row out (valid=0) and starts the next.

Sources. `hf_id` is a HF dataset, `local:<path>` a jsonl under $V41_WORK (s02b self-generated
traces), or `git:<repo>@<ref>` an upstream git ref read with `git show` (whole-repo long rows).

Modes:
  --dry-run   list sources, usability, declared licences and the row/token plan; no network
  --probe     per source: live card licence + a small streamed read to measure yield; no corpus
  (default)   build; resumable per pool through ops/logs/s02_ledger.json

RUN (CPU and network only; detach it):
    python3 ops/s02_calib_corpus.py --dry-run
    setsid nohup python3 ops/s02_calib_corpus.py --probe > ops/logs/s02_probe.log 2>&1 &
    setsid nohup python3 ops/s02_calib_corpus.py > ops/logs/s02.log 2>&1 &
Needs: datasets, tokenizers, pillow, huggingface_hub; torch on CPU only because the vision pool
calls the reference inference/image_processor.plan_image_grid.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ops import corpus_spec as spec  # noqa: E402
from ops.lib import env  # noqa: E402

STAGE = "s02"
OUT = env.WORK / "corpus"
LEDGER = env.LOGS / "s02_ledger.json"
TEXT, IMAGE_START, IMAGE, IMAGE_NEW_LINE, IMAGE_END = -1, 0, 1, 2, 3
CFG = json.load(open(env.INFERENCE / "config.json"))


def pool_sources(bucket: str) -> list[dict]:
    if bucket == "vision":
        return spec.VISION_SOURCES
    if bucket == "longctx":
        return spec.LONG_SOURCES
    return spec.SOURCES[bucket]


def spec_hash() -> str:
    blob = json.dumps({"target": spec.TOKEN_TARGET, "seq": spec.SEQ_LEN, "total": spec.TOTAL_TOKENS,
                       "long": [spec.LONG_MIN, spec.LONG_MAX],
                       "sources": {b: [(s["hf_id"], s["config"], s["split"], s["weight"]) for s in pool_sources(b)]
                                   for b in spec.BUCKETS}}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def dry_run(allow_unverified: bool, policy: str, total: int) -> None:
    rows = spec.rows_per_bucket(total)
    print(f"spec {spec_hash()}: ~{total:,} tokens; generator policy {policy}")
    print("pool plan (8k rows; longctx = token budget in 8k units, filled with whole 32k-128k docs):")
    for b in spec.BUCKETS:
        print(f"  {b:12s} share {spec.TOKEN_TARGET[b]:.2f}  {rows[b]:4d} x 8k = {rows[b] * spec.SEQ_LEN:>10,} tokens")
    for b in spec.BUCKETS:
        for s in pool_sources(b):
            ok, why = spec.usable(s, allow_unverified, policy)
            print(f"  {b:12s} {'USE ' if ok else 'SKIP'} w={s['weight']:.2f} {s['hf_id']}"
                  f"{'/' + s['config'] if s['config'] else ''}{':' + s['split'] if s['split'] else ''}"
                  f"  licence={s['licence']}  generator={s['generator']}  status={s['status']}"
                  + ("" if ok else f"  ({why})"))
    print("excluded:", json.dumps(spec.EXCLUDED, indent=1))


def live_licence(src: dict) -> str | None:
    if not src["hf_id"].startswith(("local:", "git:")):
        try:
            from huggingface_hub import HfApi
            card = HfApi().dataset_info(src["hf_id"]).card_data or {}
            lic = card.get("license")
            return ",".join(lic) if isinstance(lic, list) else lic
        except Exception as e:
            return f"error: {type(e).__name__}"
    if src["hf_id"].startswith("git:"):
        repo, ref = git_ref(src)
        for name in ("LICENSE", "LICENSE.txt", "LICENSE.md"):
            r = subprocess.run(["git", "-C", repo, "show", f"{ref}:{name}"], capture_output=True, text=True)
            if r.returncode == 0:
                head = r.stdout[:400].lower()
                return "mit" if "mit license" in head or "permission is hereby granted" in head else \
                    "bsd-3-clause" if "redistribution and use" in head else f"see {name}"
        return None
    return src["licence"]


def git_ref(src: dict) -> tuple[str, str]:
    repo, ref = src["hf_id"][4:].split("@")
    return os.path.expanduser(repo), ref


def stream(src: dict):
    hid = src["hf_id"]
    if hid.startswith("local:"):
        p = env.WORK / hid[6:]
        if not p.exists():
            return iter(())
        return (json.loads(l) for l in open(p))
    if hid.startswith("git:"):
        return git_repo_docs(src)
    from datasets import load_dataset
    kw = {"split": src["split"], "streaming": True}
    return iter(load_dataset(hid, src["config"], **kw) if src["config"] else load_dataset(hid, **kw))


def git_repo_docs(src: dict):
    """Yield whole-repo text slices of the upstream ref: files grouped by top-level directory,
    each file prefixed with its path, so one row is one coherent subtree."""
    repo, ref = git_ref(src)
    files = subprocess.run(["git", "-C", repo, "ls-tree", "-r", "--name-only", ref],
                           capture_output=True, text=True, check=True).stdout.split()
    files = [f for f in files if f.endswith(spec.REPO_EXTS)]
    groups: dict[str, list[str]] = {}
    for f in files:
        groups.setdefault(f.split("/")[0] if "/" in f else ".", []).append(f)
    for top, fs in sorted(groups.items()):
        parts = []
        for f in sorted(fs):
            r = subprocess.run(["git", "-C", repo, "show", f"{ref}:{f}"], capture_output=True)
            if r.returncode == 0 and b"\0" not in r.stdout[:4096]:
                parts.append(f"// ===== {f} =====\n" + r.stdout.decode("utf-8", "replace"))
        if parts:
            yield {"text": "\n".join(parts), "_doc": f"{src['hf_id']}:{top}"}


class Renderer:
    """Document -> token ids through the checkpoint's own encoder and tokenizer."""

    def __init__(self):
        sys.path.insert(0, str(env.ENCODING))
        import encoding as E
        from ops.lib.tok import Tok
        self.E, self.tok = E, Tok(env.SRC)
        self.image_token_id = CFG["image_token_id"]
        self.pad_id = self.tok.convert_tokens_to_ids(E.eos_token)   # pad slots carry valid=0

    def text_ids(self, doc, thinking: bool) -> list[int] | None:
        if doc is None:
            return None
        if isinstance(doc, str):
            return self.tok.encode(self.E.bos_token + doc)
        prompt = self.E.encode_messages(doc, thinking_mode="thinking" if thinking else "chat")
        return self.tok.encode(prompt)

    def vision_ids(self, messages) -> tuple[list[int], list[int], list[dict]]:
        """Expand each image placeholder into its span exactly as image_processor.prepare_vl_inputs
        lays it out, using the reference plan_image_grid (s03 re-derives it with load_image and
        asserts equality)."""
        from PIL import Image
        prompt, mm = self.E.encode_messages(messages, thinking_mode="chat", return_multi_modal_data=True)
        ids = self.tok.encode(prompt)
        imgs = iter(mm["images"])
        out, types, spans = [], [], []
        for t in ids:
            if t != self.image_token_id:
                out.append(t); types.append(TEXT)
                continue
            rec = next(imgs)
            with Image.open(rec["url"]) as im:
                w, h = im.size
            n_h, n_w = plan_grid(w, h)
            ty = [IMAGE_START] + ([IMAGE] * n_w + [IMAGE_NEW_LINE]) * n_h + [IMAGE_END]
            spans.append({"start": len(out), "path": rec["url"], "n_llm_h": n_h, "n_llm_w": n_w, "n": len(ty)})
            out += [self.image_token_id] * len(ty); types += ty
        return out, types, spans


class _VisionArgs:
    vision_patch_size = CFG["vision_patch_size"]
    vision_downsample_ratio = CFG["vision_downsample_ratio"]
    vision_max_n_token = CFG["vision_max_n_token"]
    vision_min_pixels = CFG["vision_min_pixels"]
    vision_max_wh_ratio = CFG["vision_max_wh_ratio"]


def plan_grid(w, h):
    if str(env.INFERENCE) not in sys.path:
        sys.path.insert(0, str(env.INFERENCE))
    from image_processor import plan_image_grid  # the reference function, not a copy
    n_h, n_w, _, _ = plan_image_grid(w, h, _VisionArgs)
    return n_h, n_w


def vision_messages(row: dict, img_dir: Path, key: str) -> list[dict] | None:
    """the_cauldron / Docmatix: images + texts[{user, assistant}];
    LLaVA-OneVision: image + conversations[{from, value}] with an <image> marker."""
    from PIL import Image
    imgs = row.get("images") or ([row["image"]] if row.get("image") is not None else [])
    imgs = [im for im in imgs if im is not None]
    if not imgs:
        return None
    paths = []
    for k, im in enumerate(imgs[:4]):
        if isinstance(im, dict):          # undecoded Image feature: {"bytes": ..., "path": ...}
            im = Image.open(io.BytesIO(im["bytes"]))
        buf = io.BytesIO()
        im.convert("RGB").save(buf, format="PNG")
        p = img_dir / f"{key}_{k}.png"
        p.write_bytes(buf.getvalue())
        paths.append(str(p))
    blocks = [{"type": "image", "url": p} for p in paths]
    if isinstance(row.get("texts"), list) and row["texts"]:
        msgs = []
        for i, t in enumerate(row["texts"][:6]):
            u, a = t.get("user"), t.get("assistant")
            if not u or not a:
                continue
            content = (blocks if not msgs else []) + [{"type": "text", "text": u}]
            msgs += [{"role": "user", "content": content}, {"role": "assistant", "content": a}]
        return msgs or None
    conv = spec._messages(row, "conversations")
    if conv and conv[0]["role"] == "user":
        conv[0] = {"role": "user", "content": blocks + [{"type": "text", "text": conv[0]["content"].replace("<image>", "").strip()}]}
        return conv
    return None


def url_filter(src: dict):
    st = src["status"]
    if not st.startswith("filtered:"):
        return None
    pat = spec.SCIENCE_URL if "SCIENCE" in st else spec.FINANCE_URL if "FINANCE" in st else spec.KNOWLEDGE_URL
    return re.compile(pat)


def build_pool(bucket: str, budget: int, R: Renderer, a, rep: dict) -> list[dict]:
    """Fill one pool to `budget` tokens. Returns rows: {tokens, types, valid, docs, images}."""
    S = spec.SEQ_LEN
    long_pool = bucket == "longctx"
    srcs = [s for s in pool_sources(bucket) if spec.usable(s, a.allow_unverified, a.generator_policy)[0]]
    wsum = sum(s["weight"] for s in srcs)
    rows: list[dict] = []
    cur = None
    img_dir = OUT / "images"; img_dir.mkdir(parents=True, exist_ok=True)

    def new_row():
        r = {"tokens": [], "types": [], "valid": [], "docs": [], "images": []}
        rows.append(r)
        return r

    def filled():
        return sum(sum(r["valid"]) for r in rows)

    deficit = 0
    for si, s in enumerate(srcs):
        name = s["hf_id"] + (f"/{s['config']}" if s["config"] else "") + (f":{s['split']}" if s["split"] else "")
        # a short source's shortfall is spread over the sources after it by weight (2026-10-19: rolling it all
        # into the next one gave SWE-rebench 524k of the ~600k agentic pool when selfgen was empty)
        w_left = sum(x["weight"] for x in srcs[si:])
        quota = int(budget * s["weight"] / wsum + deficit * s["weight"] / w_left)
        deficit -= deficit * s["weight"] / w_left
        live = live_licence(s)
        rec = rep.setdefault(name, {"bucket": bucket, "licence_declared": s["licence"], "licence_live": live,
                                    "licence_live_ok": spec.licence_ok(live), "generator": s["generator"],
                                    "status": s["status"], "read": 0, "kept": 0, "tokens": 0})
        if spec.licence_ok(live) is False:
            rec["skipped"] = f"live licence {live} is NC/share-alike"
            deficit += quota
            env.log(f"{bucket}: SKIP {name}: live licence {live}", STAGE, "WARN")
            continue
        flt = url_filter(s)
        got, t0 = 0, time.time()
        try:
            for row in stream(s):
                if got >= quota:
                    break
                rec["read"] += 1
                if flt is not None and not flt.search(str(row.get("url", ""))):
                    continue
                try:
                    if bucket == "vision":
                        msgs = vision_messages(row, img_dir, f"{si}_{rec['read']}")
                        ids, ty, spans = R.vision_ids(msgs) if msgs else (None, None, [])
                    elif s["fn"] is None or s["hf_id"].startswith("git:"):
                        ids, ty, spans = R.text_ids(row.get("text"), False), None, []
                    else:
                        doc = s["fn"](row)
                        th = s["thinking"] if not isinstance(row, dict) or "thinking" not in row else bool(row["thinking"])
                        ids, ty, spans = R.text_ids(doc, th), None, []
                except Exception:
                    ids = None
                if not ids:
                    if rec["read"] > 2000 and rec["kept"] == 0:
                        break  # dead source: 2,000 rows, zero usable (GLM: media sources die silently)
                    continue
                if long_pool:
                    if len(ids) < spec.LONG_MIN:
                        continue
                    ids = ids[: max(spec.LONG_MIN, min(spec.LONG_MAX, quota - got))]   # don't overshoot the quota
                    rows.append({"tokens": ids, "types": [TEXT] * len(ids), "valid": [1] * len(ids),
                                 "docs": [{"source": name, "tokens": len(ids), "doc": row.get("_doc")}], "images": []})
                    got += len(ids)
                elif bucket == "vision":
                    if len(ids) > S:
                        continue                                  # an image span would be cut
                    if cur is None or len(cur["tokens"]) + len(ids) > S:
                        if cur is not None:                       # pad this row out (valid=0)
                            pad = S - len(cur["tokens"])
                            cur["tokens"] += [R.pad_id] * pad; cur["types"] += [TEXT] * pad; cur["valid"] += [0] * pad
                        cur = new_row()
                    cur["images"] += [{**sp, "start": sp["start"] + len(cur["tokens"])} for sp in spans]
                    cur["tokens"] += ids; cur["types"] += ty; cur["valid"] += [1] * len(ids)
                    cur["docs"].append({"source": name, "tokens": len(ids)})
                    got += len(ids)
                else:
                    # one document may not dominate a source: cap at two rows and at the quota left
                    # (2026-10-19: whole SWE-rebench patches took 524k against a 224k quota)
                    ids = ids[: min(2 * S, quota - got)]
                    pos = 0
                    while pos < len(ids):
                        if cur is None or len(cur["tokens"]) == S:
                            cur = new_row()
                        k = min(S - len(cur["tokens"]), len(ids) - pos)
                        cur["tokens"] += ids[pos:pos + k]; cur["types"] += [TEXT] * k; cur["valid"] += [1] * k
                        cur["docs"].append({"source": name, "tokens": k})
                        pos += k; got += k
                rec["kept"] += 1
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        rec["tokens"] += got
        rec["seconds"] = round(time.time() - t0, 1)
        rec["yield"] = rec["kept"] / max(1, rec["read"])
        deficit += max(0, quota - got)
        env.log(f"{bucket}: {name} +{got:,} tok (quota {quota:,}), kept {rec['kept']}/{rec['read']}"
                f"{' ERROR ' + rec['error'] if 'error' in rec else ''}", STAGE)
    if not long_pool:
        # 8k pools: whole rows only; drop a trailing partial text row, pad a trailing vision row
        if rows and len(rows[-1]["tokens"]) < S:
            if bucket == "vision":
                r = rows[-1]; pad = S - len(r["tokens"])
                r["tokens"] += [R.pad_id] * pad; r["types"] += [TEXT] * pad; r["valid"] += [0] * pad
            else:
                rows.pop()
        rows = rows[: budget // S]
    if filled() < 0.9 * budget:
        raise RuntimeError(f"{bucket}: only {filled():,}/{budget:,} tokens; sources too thin. Fix the spec rather "
                           "than letting another pool absorb the share")
    return rows


def probe(a) -> None:
    out = {}
    for b in spec.BUCKETS:
        for s in pool_sources(b):
            name = s["hf_id"] + (f"/{s['config']}" if s["config"] else "") + (f":{s['split']}" if s["split"] else "")
            rec = {"bucket": b, "usable": spec.usable(s, a.allow_unverified, a.generator_policy),
                   "licence_declared": s["licence"], "licence_live": live_licence(s), "generator": s["generator"],
                   "status": s["status"]}
            try:
                ok = seen = 0
                row = {}
                for row in stream(s):
                    seen += 1
                    if b == "vision":
                        ok += bool(row.get("images") or row.get("image") is not None)
                    elif s["fn"] is not None:
                        ok += bool(s["fn"](row))
                    else:
                        ok += bool(row.get("text"))
                    if seen >= a.probe_rows:
                        break
                rec.update(seen=seen, yield_=ok / max(1, seen), columns=sorted(row.keys()) if seen else [])
            except Exception as e:
                rec["error"] = f"{type(e).__name__}: {str(e)[:300]}"
            out[name] = rec
            env.log(f"probe {name}: {json.dumps({k: v for k, v in rec.items() if k != 'columns'}, default=str)}", STAGE)
    env.atomic_json(env.LOGS / "s02_probe.json", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--probe-rows", type=int, default=64)
    ap.add_argument("--allow-unverified", action="store_true",
                    help="include sources whose generator is 'unverified' (only after checking their cards)")
    ap.add_argument("--generator-policy", choices=spec.GENERATOR_POLICIES, default="no-claude")
    ap.add_argument("--total-tokens", type=int, default=spec.TOTAL_TOKENS)
    a = ap.parse_args()
    if a.dry_run:
        return dry_run(a.allow_unverified, a.generator_policy, a.total_tokens)
    if a.probe:
        return probe(a)

    OUT.mkdir(parents=True, exist_ok=True)
    led = env.Ledger(LEDGER)
    h = spec_hash()
    if led.get("spec", {}).get("hash") not in (None, h):
        raise SystemExit(f"ledger was built with spec {led.get('spec')['hash']}, current {h}; move {OUT} aside first")
    led.mark("spec", hash=h, generator_policy=a.generator_policy, allow_unverified=a.allow_unverified)
    plan = spec.rows_per_bucket(a.total_tokens)
    R = Renderer()
    rep = json.load(open(OUT / "sources.json")) if (OUT / "sources.json").exists() else {}
    for b in spec.BUCKETS:
        part = OUT / f"pool_{b}.json"
        if led.done(f"pool:{b}") and part.exists():
            continue
        rows = build_pool(b, plan[b] * spec.SEQ_LEN, R, a, rep)
        env.atomic_json(part, rows)
        env.atomic_json(OUT / "sources.json", rep)
        led.finish(f"pool:{b}", rows=len(rows), tokens=sum(sum(r["valid"]) for r in rows))

    pools = {b: json.load(open(OUT / f"pool_{b}.json")) for b in spec.BUCKETS}
    order = spec.interleave({b: len(pools[b]) for b in spec.BUCKETS})
    T = sum(len(r["tokens"]) for b in pools for r in pools[b])
    mm = {k: np.lib.format.open_memmap(OUT / f"{k}.npy", mode="w+", dtype=d, shape=(T,))
          for k, d in (("tokens", np.uint32), ("types", np.int8), ("valid", np.uint8))}
    nxt = {b: 0 for b in spec.BUCKETS}
    off = 0
    tok_by = {b: 0 for b in spec.BUCKETS}
    with open(OUT / "rows.jsonl.tmp", "w") as fh:
        for row, b in enumerate(order):
            i = nxt[b]; nxt[b] += 1
            r = pools[b][i]
            n = len(r["tokens"])
            mm["tokens"][off:off + n] = r["tokens"]; mm["types"][off:off + n] = r["types"]; mm["valid"][off:off + n] = r["valid"]
            n_held = max(1, round(len(pools[b]) * spec.HELDOUT_ROWS_FRACTION))
            fh.write(json.dumps({"row": row, "bucket": b, "bucket_id": spec.BUCKETS.index(b), "offset": off, "length": n,
                                 "heldout": i >= len(pools[b]) - n_held, "n_valid": int(sum(r["valid"])),
                                 "docs": r["docs"], "images": r["images"]}) + "\n")
            tok_by[b] += int(sum(r["valid"]))
            off += n
    os.replace(OUT / "rows.jsonl.tmp", OUT / "rows.jsonl")
    for m in mm.values():
        m.flush()
    total = sum(tok_by.values())
    lens = [len(pools["longctx"][i]["tokens"]) for i in range(len(pools["longctx"]))]
    manifest = {"spec_hash": h, "rows": len(order), "tokens_total": T, "tokens_valid": total, "order": order,
                "achieved_share": {b: tok_by[b] / total for b in spec.BUCKETS}, "target_share": spec.TOKEN_TARGET,
                "longctx_lengths": lens, "max_row_len": max(len(r["tokens"]) for b in pools for r in pools[b]),
                "generator_policy": a.generator_policy, "revision": env.REVISION, "ts": env.now()}
    env.atomic_json(OUT / "manifest.json", manifest)
    env.log(f"corpus done: {len(order)} rows, {total:,} tokens, achieved "
            + ", ".join(f"{b} {v:.3f}" for b, v in manifest["achieved_share"].items()), STAGE)


if __name__ == "__main__":
    main()
