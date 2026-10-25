#!/usr/bin/env python3
"""One-shot diagnostic (2026-10-25): are golden A192's low isolated rows MoE routing near-ties?

Golden A192 (fixed harness, 60ef2c8) fails condition 2 of the drift amendment on doc1k pos 850: isolated row cos
0.9765 at L6, >= 0.9995 at every other layer. Other single low isolated rows: L9/877, L33/988, L39/180,530,721.
This re-runs the torch reference (ref_dump.py's arithmetic, doc1k only, layers 0..max target) and at each target
(layer, pos) reports:
  - sanity: cos(rerun output, saved reference row) -- must be ~1 or the rerun is not the reference
  - the biased router scores around the top-6 boundary (6th vs 7th margin, ids)
  - counterfactual: the same layer with the 6th expert swapped for the 7th on that row only, its cos to the
    reference. If it lands near the CUDA isolated cos, the CUDA row is a routing flip at a near-tie.
GPU, one layer resident. NEVER beside another GPU/model-loading job (run with chain2 held).

  .venv/bin/python -I ops/diag_route_flip.py --ref $V41_WORK/golden/A192 --keepset $V41_WORK/s04/keepset.json \
      --thor $V41_WORK/thor/A192 --targets 6:850 9:877 33:988 39:180 39:530 39:721 --cuda-cos ops/logs/diag_golden_a192/row_cos.txt
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "cuda" / "tests" / "golden"))
from ops.lib import env, memfence  # noqa: E402
from ops import s03_saliency as s03  # noqa: E402
import ref_dump  # noqa: E402


def bf16_to_f32(u16: np.ndarray) -> np.ndarray:
    return (u16.astype(np.uint32) << 16).view(np.float32)


def cos(a, b) -> float:
    a = a.reshape(-1).astype(np.float64); b = b.reshape(-1).astype(np.float64)
    return float(a @ b / np.sqrt((a @ a) * (b @ b) + 1e-300))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--keepset", required=True)
    ap.add_argument("--thor", required=True)
    ap.add_argument("--targets", nargs="+", required=True, help="LAYER:POS ...")
    ap.add_argument("--cuda-cos", default=None, help="gate row_cos.txt: prints the CUDA isolated cos beside each target")
    ap.add_argument("--out", default=str(REPO / "ops/logs/diag_route_flip.json"))
    a = ap.parse_args()

    import torch
    from transformers import AutoTokenizer
    from ops.lib import replay, shards, thor_format as TF
    from ops.lib import keepset as KS

    env.assert_no_gpu_server("diag_route_flip")
    T_ = {}
    for t in a.targets:
        L, p = (int(x) for x in t.split(":")); T_.setdefault(L, []).append(p)
    hi = max(T_)
    cuda_cos = {}
    if a.cuda_cos:
        for line in open(a.cuda_cos):
            f = line.split()
            if f[1] == "doc1k" and f[2] == "1" and int(f[0]) in T_ and int(f[3]) in T_[int(f[0])]:
                cuda_cos[(int(f[0]), int(f[3]))] = float(f[4])

    ref = Path(a.ref)
    meta = json.load(open(ref / "meta.json"))
    ids = np.load(ref / "doc1k" / "tokens.npy").tolist()
    T = len(ids); f0 = meta["prompts"]["doc1k"]["save_from"]
    tok = AutoTokenizer.from_pretrained(env.SRC)
    assert ref_dump.prompts(tok, False)["doc1k"] == ids, "doc1k tokens differ from the reference's"

    M = replay.load_reference(meta["kernels"], 4096)
    args = env.model_args(M, max_batch_size=1, max_seq_len=T + 8)
    keep, ks_sha = KS.load(Path(a.keepset))
    build = TF.Build(Path(a.thor))
    idx = shards.ShardIndex()
    W = idx.get("embed.weight").cuda()
    H = torch.nn.functional.embedding(torch.tensor(ids).cuda(), W).unsqueeze(1).repeat(1, 4, 1).to(torch.bfloat16)
    PM = torch.zeros(T, 4, dtype=torch.float32, device="cuda"); PM[:, 0] = 1.0
    del W
    state = {}
    results = []

    for L in range(0, hi + 1):
        t0 = time.time()
        memfence.require(12.0, f"diag L{L}")
        blk = shards.build_block(M, args, L, idx, "cuda", with_engram=True)
        experts = replay.thor_experts(M, args, Path(a.thor), L, keep[str(L)], ks_sha)
        rd = build.reader(TF.layer_file(L))
        try:
            gain = float(np.asarray(rd.get(f"layers.{L}.ffn.heal_gain")).reshape(-1)[0])
        except Exception:
            gain = 1.0
        cap = {}

        class Obs:   # replay's observer protocol; only route() is used
            def route(self, moe, x, im, weights, indices):
                cap["x"] = x.detach().clone(); cap["idx"] = indices.detach().clone()
            def expert(self, *a_): pass
            def routed(self, y): pass

        arm = replay.ArmLayer(args.n_routed_experts, keep[str(L)], gain=gain, experts=experts,
                              observer=Obs() if L in T_ else None)
        replay.configure(blk, arm)
        torch.set_default_device("cuda")      # where ref_dump sets it: after layer 0's block is built
        need = s03.sources_for(L)
        sa = M.shared_attn

        def set_shared():
            for kind, src in need.items():
                arr = state[(kind, src)]
                if kind in ("kv", "ik"):
                    t = torch.from_numpy(arr).cuda().view(torch.bfloat16)[None]
                    if kind == "kv": sa.compress_kv = t
                    else: sa.index_k = t
                elif kind == "topk":
                    sa.topk_idxs = torch.from_numpy(arr).cuda()[None]
                else:
                    sa.candidate_bits = None if arr.size == 0 else torch.from_numpy(arr).cuda()[None]

        def fresh():
            for mod in blk.modules():
                if isinstance(mod, M.Compressor) and mod.compress_ratio > 1:
                    mod.kv_state.zero_(); mod.score_state.fill_(float("-inf"))

        x = H[None]; pm = PM[None]
        with torch.inference_mode():
            set_shared(); fresh()
            if blk.engram is not None:
                rows = torch.from_numpy(np.load(ref / "doc1k" / f"engram_L{L}.npy")).cuda()
                blk.engram.embed.rows_fp8 = rows[..., :256].contiguous().view(torch.float8_e4m3fn)[None]
                blk.engram.embed.rows_scale = rows[..., 256:].contiguous().view(torch.float8_e8m0fnu)[None]
                x = blk.engram(x, torch.zeros(1, T, 24, dtype=torch.long, device="cuda"), None)
                blk.engram.embed.rows_fp8 = blk.engram.embed.rows_scale = None
            y, ffn_pre = blk(x, 0, pm, None)
        for kind in s03.published_by(L):
            state[(kind, L)] = s03.grab_state(M, blk, kind, T)

        if L in T_:
            rh = bf16_to_f32(np.load(ref / "doc1k" / f"L{L:02d}_h.npy"))       # [S, 4, 5120]
            g = blk.ffn.gate
            xs = cap["x"]
            with torch.inference_mode():
                sc = torch.nn.functional.softplus(torch.nn.functional.linear(xs.float(), g.weight.float()) / g.gate_temp).sqrt()
                biased = sc + g.bias
            for pos in T_[L]:
                b = biased[pos]; order = torch.argsort(b, descending=True)[:8].tolist()
                top = cap["idx"][pos].tolist()
                k = len(top)
                margin = float(b[order[k - 1]] - b[order[k]])
                sanity = cos(y[0, pos].float().cpu().numpy(), rh[pos - f0])
                # counterfactual: swap the k-th pick for the (k+1)-th on this row only
                swap_to = order[k]; swap_from = order[k - 1]
                orig_fwd = g.forward

                def fwd(xx, im=None, _o=orig_fwd, _p=pos, _f=swap_from, _t=swap_to):
                    w, ind = _o(xx, im)
                    if xx.shape[0] > _p:
                        r = ind[_p].clone(); r[r == _f] = _t; ind = ind.clone(); ind[_p] = r
                        s_ = torch.nn.functional.softplus(torch.nn.functional.linear(xx.float(), g.weight.float()) / g.gate_temp).sqrt()
                        ww = s_[_p].gather(0, ind[_p]); ww = (ww / (ww.sum() + 1e-20) if g.norm_topk_prob and g.topk > 1 else ww) * g.route_scale
                        w = w.clone(); w[_p] = ww
                    return w, ind
                g.forward = fwd
                with torch.inference_mode():
                    set_shared(); fresh()
                    y2, _ = blk(x, 0, pm, None)
                g.forward = orig_fwd
                cf = cos(y2[0, pos].float().cpu().numpy(), rh[pos - f0])
                rec = {"layer": L, "pos": pos, "sanity_cos_vs_ref": sanity, "top_k_ids": top,
                       "boundary_ids": [swap_from, swap_to], "margin_k_vs_k1": margin,
                       "biased_top8": [round(float(b[i]), 5) for i in order],
                       "counterfactual_cos_vs_ref": cf, "cuda_isolated_cos": cuda_cos.get((L, pos))}
                results.append(rec)
                print(f"[diag] L{L} pos {pos}: sanity {sanity:.6f} | 6/7 margin {margin:.5f} ids {swap_from}->{swap_to} "
                      f"| counterfactual cos {cf:.5f} | CUDA isolated {cuda_cos.get((L, pos))}", flush=True)
        H, PM = y[0].detach(), ffn_pre[0].float().detach()
        for (kind, src) in list(state):
            if s03.last_consumer(kind, src) <= L:
                del state[(kind, src)]
        del blk, experts, y, ffn_pre, x
        torch.cuda.empty_cache()
        print(f"[diag] L{L} {time.time() - t0:.0f}s", flush=True)
    idx.close()
    json.dump(results, open(a.out, "w"), indent=1)
    print(f"[diag] done -> {a.out}")


if __name__ == "__main__":
    main()
