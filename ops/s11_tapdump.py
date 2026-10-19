#!/usr/bin/env python3
"""s11 tap dump: the SERVED REAP arm's own DSpark inputs over the s11 traces.

The reference (`Transformer.forward`) feeds the head `h.mean(dim=2)` of the stream ENTERING each tap
layer (37, 38, 39), i.e. the hc-mean of layers 36/37/38's outputs. This runs the winning Thor arm
(router pruned to its keep-set, experts from the Thor build, heal gain folded: the machinery of
cuda/tests/golden/ref_dump.py) over each trace (prompt + the served model's own greedy continuation),
layers 0..38, one layer resident at a time, and keeps only the three taps.

  out $V41_WORK/s11/taps_<set>/: tokens.npy int64 [T]; L37.npy L38.npy L39.npy uint16 (bf16 bits)
  [T, dim], flat over traces; meta.json {seqs: [{row, offset, length, gen_start}], taps, served, complete}.

GPU. NEVER beside another GPU job. Resumable only as a whole (a partial dump is redone).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ops.lib import env, memfence  # noqa: E402
from ops import s03_saliency as s03  # noqa: E402

TAPS = (37, 38, 39)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=["eval", "train"])
    ap.add_argument("--kernels", default="tilelang", choices=["torch", "tilelang"])
    ap.add_argument("--token-chunk", type=int, default=4096)
    a = ap.parse_args()
    import torch
    from transformers import AutoTokenizer
    from ops.lib import replay, shards, thor_format as TF
    from ops.lib.engram_rows import EngramTables
    from ops.s11_spec import OUT, served

    env.assert_no_gpu_server("s11_tapdump")
    srv = served()
    out = OUT / f"taps_{a.set}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "meta.json").exists() and json.load(open(out / "meta.json")).get("complete"):
        print(f"[tapdump] {out} complete")
        return
    traces = [json.loads(l) for l in open(OUT / f"traces_{a.set}.jsonl")]
    seqs, toks, off = [], [], 0
    for t in traces:
        ids = t["prompt_ids"] + t["completion_ids"]
        seqs.append({"row": t["row"], "offset": off, "length": len(ids), "gen_start": len(t["prompt_ids"])})
        toks.append(np.asarray(ids, np.int64))
        off += len(ids)
    tokens = np.concatenate(toks)
    np.save(out / "tokens.npy", tokens)
    M = replay.load_reference(a.kernels, a.token_chunk)
    tok = AutoTokenizer.from_pretrained(env.SRC)
    args = env.model_args(M, max_batch_size=1, max_seq_len=max(s["length"] for s in seqs) + 8)
    keep, ks_sha = srv["keep"], srv["fp"]
    build = TF.Build(srv["dir"])
    taps = {L: np.lib.format.open_memmap(out / f"L{L}.npy", mode="w+", dtype=np.uint16, shape=(off, args.dim))
            for L in TAPS}

    import engram as E   # inference/engram.py
    layout = E.EngramLayout.from_args(args)
    hstate = E.NgramHashState(args, layout, tok)
    idx = shards.ShardIndex()
    Wemb = idx.get("embed.weight").cuda()
    tabs = {L: EngramTables(env.SRC / f"model-{47 + li:05d}-of-00048.safetensors", L) for li, L in enumerate((1, 14))}
    hashes = []
    for s, ids in zip(seqs, toks):
        hstate.cache.zero_()
        hashes.append(hstate(torch.tensor([ids.tolist()]), 0, None)[0].numpy())   # [T, 2, 24]
    rows = {}                                                     # one sorted pass over each table
    for li, L in enumerate((1, 14)):
        allh = np.concatenate([h[:, li, :] for h in hashes]).reshape(-1)
        v, sc = tabs[L].gather(np.ascontiguousarray(allh))
        rows[L] = np.concatenate([v.reshape(off, 24, 256), sc.reshape(off, 24, 8)], -1).astype(np.uint8)
    streams = []
    for s, ids in zip(seqs, toks):
        x = torch.nn.functional.embedding(torch.from_numpy(ids).cuda(), Wemb)
        H = x.unsqueeze(1).repeat(1, 4, 1).to(torch.bfloat16).cpu()
        PM = torch.zeros(len(ids), 4, dtype=torch.float32); PM[:, 0] = 1.0
        streams.append([H, PM])
    del Wemb
    state = [{} for _ in seqs]
    for L in range(0, max(TAPS)):
        t0 = time.time()
        memfence.require(12.0, f"s11 tapdump L{L}")
        blk = shards.build_block(M, args, L, idx, "cuda", with_engram=True)
        experts = replay.thor_experts(M, args, srv["dir"], L, keep[str(L)], ks_sha)
        try:
            gain = float(np.asarray(build.reader(TF.layer_file(L)).get(f"layers.{L}.ffn.heal_gain")).reshape(-1)[0])
        except Exception:
            gain = 1.0
        replay.configure(blk, replay.ArmLayer(args.n_routed_experts, keep[str(L)], gain=gain, experts=experts))
        torch.set_default_device("cuda")
        need = s03.sources_for(L)
        for k, s in enumerate(seqs):
            H, PM = streams[k]
            T = H.shape[0]
            x = H.cuda()[None]; pm = PM.cuda()[None]
            sa = M.shared_attn
            for kind, src in need.items():
                arr = state[k][(kind, src)]
                if kind in ("kv", "ik"):
                    t = torch.from_numpy(arr).cuda().view(torch.bfloat16)[None]
                    if kind == "kv": sa.compress_kv = t
                    else: sa.index_k = t
                elif kind == "topk":
                    sa.topk_idxs = torch.from_numpy(arr).cuda()[None]
                else:
                    sa.candidate_bits = None if arr.size == 0 else torch.from_numpy(arr).cuda()[None]
            with torch.inference_mode():
                for mod in blk.modules():
                    if isinstance(mod, M.Compressor) and mod.compress_ratio > 1:
                        mod.kv_state.zero_(); mod.score_state.fill_(float("-inf"))
                if blk.engram is not None:
                    r_all = rows[L][s["offset"]:s["offset"] + T]
                    o = torch.empty_like(x)
                    for c0 in range(0, T, a.token_chunk):
                        c1 = min(T, c0 + a.token_chunk)
                        r = torch.from_numpy(np.ascontiguousarray(r_all[c0:c1])).cuda()
                        blk.engram.embed.rows_fp8 = r[..., :256].contiguous().view(torch.float8_e4m3fn)[None]
                        blk.engram.embed.rows_scale = r[..., 256:].contiguous().view(torch.float8_e8m0fnu)[None]
                        dummy = torch.zeros(1, c1 - c0, 24, dtype=torch.long, device="cuda")
                        o[:, c0:c1] = blk.engram(x[:, c0:c1], dummy, None)
                    blk.engram.embed.rows_fp8 = blk.engram.embed.rows_scale = None
                    x = o
                y, ffn_pre = blk(x, 0, pm, None)
                if L + 1 in TAPS:                        # the stream entering layer L+1, hc-mean in bf16
                    taps[L + 1][s["offset"]:s["offset"] + T] = y[0].mean(dim=1).contiguous().view(torch.int16).cpu().numpy().view(np.uint16)
            for kind in s03.published_by(L):
                state[k][(kind, L)] = s03.grab_state(M, blk, kind, T)
            streams[k] = [y[0].cpu(), ffn_pre[0].float().cpu()]
            del x, y, ffn_pre
        for st in state:
            for key in list(st):
                if s03.last_consumer(*key) <= L:
                    del st[key]
        del blk
        torch.cuda.empty_cache()
        print(f"[tapdump] {a.set} L{L} {time.time() - t0:.0f}s", flush=True)
    idx.close()
    for m in taps.values():
        m.flush()
    env.atomic_json(out / "meta.json", {"seqs": seqs, "taps": list(TAPS), "served": srv["name"], "keepset_sha256": ks_sha,
                                        "kernels": a.kernels, "tokens": int(off), "complete": True, "ts": env.now()})
    print(f"[tapdump] done -> {out} ({off} tokens, {len(seqs)} traces)")


if __name__ == "__main__":
    main()
