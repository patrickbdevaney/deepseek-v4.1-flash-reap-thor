#!/usr/bin/env python3
"""Golden reference for tests/gate_v41_golden.cu: the checkpoint's own inference/model.py, one Block
at a time (ops' machinery: shards.build_block, s03's cross-layer state protocol, replay's arm MoE),
over a few prompts, dumping every layer's output stream so the CUDA server is compared layer by layer.

GPU, one layer resident at a time. NEVER run beside another GPU job (Thor unified memory): the chain
runs this, then the CUDA gate, sequentially (cuda/CHAIN_STEPS.sh).

  python -I tests/golden/ref_dump.py --out $V41_WORK/golden/src            # source checkpoint, 384 experts
  python -I tests/golden/ref_dump.py --out ... --keepset $V41_WORK/s04/keepset.json --thor $V41_WORK/thor/A192
        # the REAP arm: router pruned to the keep-set, experts from the thor build (replay's FQExpert:
        # dequantised to bf16, activations fake-quantised) -- what the CUDA server serves

Per prompt <p> (out/<p>/):
  tokens.npy int32 [T]; engram_L{1,14}.npy uint8 [T, 24, 264] (the gathered rows, for the CUDA side)
  L{LL}_h.npy uint16 (bf16 bits) [S, 4, 5120] and L{LL}_pre.npy float32 [S, 4]: layer output rows
      [T-S, T) (S = --save-rows, all rows if T is smaller)
  head_argmax.npy int32 [S]; head_last_logits.npy float32 [vocab]
meta.json: prompts, T, saved range, layers, arm, kernels.

Prompts: a short chat turn, a ~1k-token document (decoder-span replay exercised), and with --long a
~20k-token document (> 16,384 compressed rows at layer 20: the candidate-block path).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from ops.lib import env, memfence  # noqa: E402
from ops import s03_saliency as s03  # noqa: E402  (pure functions only: sources_for, published_by, grab_state)


def prompts(tok, long: bool) -> dict:
    sys.path.insert(0, str(env.SRC / "encoding"))
    import encoding as enc
    out = {}
    chat = enc.encode_messages([{"role": "user", "content": "Explain why the sky is blue in two sentences."}],
                               thinking_mode="chat")
    out["chat"] = tok.encode(chat, add_special_tokens=False)
    doc = (env.SRC / "README.md").read_text(errors="replace")
    ids = tok.encode(doc, add_special_tokens=False)
    out["doc1k"] = [0] + ids[:1100]
    if long:
        lic = (env.SRC / "LICENSE").read_text(errors="replace")
        src = ids + tok.encode(lic, add_special_tokens=False)
        rep = (20500 // len(src) + 1)
        out["long20k"] = [0] + (src * rep)[:20500]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--kernels", default="torch", choices=["torch", "tilelang"])
    ap.add_argument("--long", action="store_true")
    ap.add_argument("--save-rows", type=int, default=1024)
    ap.add_argument("--layers", default="0-39")
    ap.add_argument("--keepset", default=None)
    ap.add_argument("--thor", default=None)
    ap.add_argument("--token-chunk", type=int, default=4096)
    # Host memory grows ~7 GiB per layer in one process (native, not Python-visible; malloc_trim does not return it,
    # measured 2026-10-24), so a 40-layer run hits the memfence near L8. --layers-per-proc N saves the carried state
    # (residual streams + published attention states) to <out>/_carry.pt after N layers and exits 0; the next run
    # resumes from it. The caller loops while _carry.pt exists (cuda/CHAIN_STEPS.sh golden_ref).
    ap.add_argument("--layers-per-proc", type=int, default=0)
    a = ap.parse_args()
    import torch
    from transformers import AutoTokenizer
    from ops.lib import replay, shards
    from ops.lib import keepset as KS
    from ops.lib.engram_rows import EngramTables

    env.assert_no_gpu_server("golden_ref_dump")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    lo, hi = (int(x) for x in a.layers.split("-"))
    M = replay.load_reference(a.kernels, a.token_chunk)
    tok = AutoTokenizer.from_pretrained(env.SRC)
    P = prompts(tok, a.long)
    maxT = max(len(v) for v in P.values())
    args = env.model_args(M, max_batch_size=1, max_seq_len=maxT + 8)
    keep, ks_sha, build = None, None, None
    if a.keepset:
        keep, ks_sha = KS.load(Path(a.keepset))
    if a.thor:
        from ops.lib import thor_format as TF
        build = TF.Build(Path(a.thor))
        if not keep:
            raise SystemExit("--thor needs --keepset (the build's experts are renumbered keep-set order)")
    meta = {"prompts": {}, "layers": [lo, hi], "kernels": a.kernels, "keepset_sha256": ks_sha,
            "thor": a.thor, "save_rows": a.save_rows, "started": time.strftime("%F %T")}

    # ---- hashes, Engram rows, embeddings: per prompt
    import engram as E   # inference/engram.py (import_reference put inference/ on sys.path)
    idx = shards.ShardIndex()
    carry_p = out / "_carry.pt"
    carry = torch.load(carry_p, weights_only=False) if carry_p.exists() else None
    if carry is not None:
        if carry["args"] != [a.layers, a.keepset, a.thor, a.long, a.save_rows]:
            raise SystemExit(f"{carry_p} was written by a run with different arguments; remove it to start over")
        streams, state, meta, lo = carry["streams"], carry["state"], carry["meta"], carry["next"]
        print(f"[golden] resuming at L{lo} from {carry_p}", flush=True)
    P_items = [] if carry is not None else P.items()
    if carry is None:
        layout = E.EngramLayout.from_args(args)
        hstate = E.NgramHashState(args, layout, tok)
        W = idx.get("embed.weight").cuda()
        streams = {}
    for name, ids in P_items:
        T = len(ids); d = out / name; d.mkdir(exist_ok=True)
        np.save(d / "tokens.npy", np.asarray(ids, np.int32))
        hstate.cache.zero_()
        h = hstate(torch.tensor([ids]), 0, None)[0].numpy()                 # [T, 2, 24]
        for li, L in enumerate((1, 14)):
            tabs = EngramTables(env.SRC / f"model-{47 + li:05d}-of-00048.safetensors", L)
            v, s = tabs.gather(np.ascontiguousarray(h[:, li, :]).reshape(-1))
            rows = np.concatenate([v.reshape(T, 24, 256), s.reshape(T, 24, 8)], -1)
            np.save(d / f"engram_L{L}.npy", rows.astype(np.uint8))
        x = torch.nn.functional.embedding(torch.tensor(ids).cuda(), W)
        H = x.unsqueeze(1).repeat(1, 4, 1).to(torch.bfloat16)
        PM = torch.zeros(T, 4, dtype=torch.float32, device="cuda"); PM[:, 0] = 1.0
        streams[name] = [H.cpu(), PM.cpu()]
        S = min(T, a.save_rows)
        meta["prompts"][name] = {"T": T, "save_from": T - S}
    if carry is None:
        del W
        state = {name: {} for name in P}                                # (kind, src) -> array, per prompt
    n_done = 0
    if carry is not None:
        # A single run sets the CUDA default device during its first layer, so every later block (and its RoPE
        # tables, rebuilt by shards._reinit_buffers with torch.arange) is built under it. Match that, or the
        # resumed run's freqs come from CPU math and drift with position (measured 2026-10-24).
        torch.set_default_device("cuda")

    for L in range(lo, hi + 1):
        t0 = time.time()
        memfence.require(12.0, f"golden L{L}")
        blk = shards.build_block(M, args, L, idx, "cuda", with_engram=True)
        if keep is not None:
            experts = None
            gain = 1.0
            if build is not None:
                experts = replay.thor_experts(M, args, Path(a.thor), L, keep[str(L)], ks_sha)
                rd = build.reader(TF.layer_file(L))
                try:
                    gain = float(np.asarray(rd.get(f"layers.{L}.ffn.heal_gain")).reshape(-1)[0])
                except Exception:
                    gain = 1.0
            replay.configure(blk, replay.ArmLayer(args.n_routed_experts, keep[str(L)], gain=gain, experts=experts))
        torch.set_default_device("cuda")
        need = s03.sources_for(L)
        for name in P:
            H, PM = streams[name]
            T = H.shape[0]
            x = H.cuda()[None]; pm = PM.cuda()[None]
            sa = M.shared_attn
            for kind, src in need.items():
                arr = state[name][(kind, src)]
                if kind in ("kv", "ik"):
                    t = torch.from_numpy(arr).cuda().view(torch.bfloat16)[None]
                    if kind == "kv": sa.compress_kv = t
                    else: sa.index_k = t
                elif kind == "topk":
                    sa.topk_idxs = torch.from_numpy(arr).cuda()[None]
                else:
                    sa.candidate_bits = None if arr.size == 0 else torch.from_numpy(arr).cuda()[None]
            with torch.inference_mode():
                for mod in blk.modules():                                   # fresh caches per prompt
                    if isinstance(mod, M.Compressor) and mod.compress_ratio > 1:
                        mod.kv_state.zero_(); mod.score_state.fill_(float("-inf"))
                if blk.engram is not None:
                    rows = np.load(out / name / f"engram_L{L}.npy")
                    o = torch.empty_like(x)
                    for c0 in range(0, T, a.token_chunk):
                        c1 = min(T, c0 + a.token_chunk)
                        r = torch.from_numpy(np.ascontiguousarray(rows[c0:c1])).cuda()
                        blk.engram.embed.rows_fp8 = r[..., :256].contiguous().view(torch.float8_e4m3fn)[None]
                        blk.engram.embed.rows_scale = r[..., 256:].contiguous().view(torch.float8_e8m0fnu)[None]
                        dummy = torch.zeros(1, c1 - c0, 24, dtype=torch.long, device="cuda")
                        o[:, c0:c1] = blk.engram(x[:, c0:c1], dummy, None)
                    blk.engram.embed.rows_fp8 = blk.engram.embed.rows_scale = None
                    x = o
                y, ffn_pre = blk(x, 0, pm, None)
            for kind in s03.published_by(L):
                state[name][(kind, L)] = s03.grab_state(M, blk, kind, T)
            streams[name] = [y[0].cpu(), ffn_pre[0].float().cpu()]
            f = meta["prompts"][name]["save_from"]
            np.save(out / name / f"L{L:02d}_h.npy", y[0, f:].contiguous().view(torch.int16).cpu().numpy().view(np.uint16))
            np.save(out / name / f"L{L:02d}_pre.npy", ffn_pre[0, f:].float().cpu().numpy())
            del x, y, ffn_pre
        # free states nobody later reads
        for name in P:
            for (kind, src) in list(state[name]):
                if s03.last_consumer(kind, src) <= L:
                    del state[name][(kind, src)]
        del blk
        torch.cuda.empty_cache()
        print(f"[golden] L{L} {time.time() - t0:.0f}s", flush=True)
        json.dump(meta, open(out / "meta.json", "w"), indent=1)
        n_done += 1
        if a.layers_per_proc and n_done >= a.layers_per_proc and L < hi:
            tmp = carry_p.with_suffix(".tmp")
            torch.save({"args": [a.layers, a.keepset, a.thor, a.long, a.save_rows], "next": L + 1,
                        "streams": streams, "state": state, "meta": meta}, tmp)
            os.replace(tmp, carry_p)
            print(f"[golden] carried state saved after L{L}; re-run to continue", flush=True)
            idx.close()
            return

    if hi == 39:                                                      # head, as s03 unit_head does
        norm_w = idx.get("norm.weight").float().cuda()
        Wh = idx.get("head.weight").float().cuda()
        for name in P:
            H, PM = streams[name]
            f = meta["prompts"][name]["save_from"]
            x = H[f:].cuda(); pm = PM[f:].cuda()
            h = torch.sum(pm.unsqueeze(-1) * x.float(), dim=1).to(torch.bfloat16).float()
            hn = (norm_w * (h * torch.rsqrt(h.square().mean(-1, keepdim=True) + args.norm_eps))).to(torch.bfloat16)
            logits = torch.nn.functional.linear(hn.float(), Wh)
            np.save(out / name / "head_argmax.npy", logits.argmax(-1).int().cpu().numpy())
            np.save(out / name / "head_last_logits.npy", logits[-1].cpu().numpy())
    idx.close()
    carry_p.unlink(missing_ok=True)
    meta["finished"] = time.strftime("%F %T")
    json.dump(meta, open(out / "meta.json", "w"), indent=1)
    print(f"[golden] done -> {out}")


if __name__ == "__main__":
    main()
