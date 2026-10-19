#!/usr/bin/env python3
"""s01: run decoder layer 3 of DeepSeek-V4.1-Flash-1017 in the reference numerics on sm_110a and
check it against a CPU fp32 oracle. This is the gate in front of the whole saliency pass: it
answers whether the reference TileLang kernels build and agree on Thor, how fast one layer is,
and how much memory one layer costs.

What runs. Layer 3 is a compress-ratio-2 layer that is neither a KV source nor an index source,
so its attention reads compress_kv, index_k and topk_idxs published by layer 2. The probe builds
layers 2 and 3 from their shards (5, 6), embeds a fixed prompt with the real embedding (shard 2),
expands it to the 4 hyper-connection streams with the identity pre_mix, and runs layer 2 then
layer 3, so the cross-layer state is produced by the real source layer rather than faked.
Layers 0-1 (and so Engram) are skipped: the input to layer 2 is the embedding, not the true
layer-2 input. That is irrelevant to a kernel-agreement probe and is stated in the report.
With --hidden FILE the probe instead reads a recorded [S, 4, 5120] bf16 stream and [S, 4] fp32
pre_mix (e.g. an s03 row after layer 2) and runs layer 3 from it.

Arms (each reported):
  gpu    reference model.py + the selected kernels (tilelang | torch) on cuda, bf16
  oracle reference model.py + ops/lib/torch_kernels.py on CPU, bf16 params promoted to fp32
The GPU arm uses s03's query-chunked Indexer (ops/lib/hooks.py) unless --reference-indexer, so
the oracle comparison also validates that patch. Report: cosine and relative L2 of the layer-3 stream output and of ffn_pre vs the oracle,
top-6 routing set agreement, layer-3 tok/s (median of --reps, after one warm-up), peak CUDA
allocation, and the MemAvailable drop. Written to ops/logs/s01_layer_probe.json.

RUN (only when no model server is up; the stage refuses otherwise):
    setsid nohup python3 ops/s01_layer_probe.py --tokens 2048 > ops/logs/s01.log 2>&1 &
    # if tilelang fails to build for sm_110a, measure the torch fallback instead:
    setsid nohup python3 ops/s01_layer_probe.py --kernels torch > ops/logs/s01_torch.log 2>&1 &
Afterwards: python3 -c 'from ops.lib.env import reclaim; reclaim("s01")'
Needs: torch>=2.10 with float4_e2m1fn_x2 + float8_e8m0fnu, safetensors, tokenizers, sympy,
and for --kernels tilelang, tilelang==0.1.12 (inference/requirements.txt).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ops.lib import env  # noqa: E402

PROMPT = ("The mitochondrion is the site of oxidative phosphorylation. In the inner membrane, the electron "
          "transport chain pumps protons into the intermembrane space, and ATP synthase uses the gradient. ")


def promote_fp32(mod):
    """bf16 params/buffers -> fp32; fp8/fp4 weights and their e8m0 scales stay as stored."""
    import torch
    for p in mod.parameters():
        if p.dtype == torch.bfloat16:
            p.data = p.data.float()
    for name, b in list(mod.named_buffers()):
        if b.dtype == torch.bfloat16:
            owner = mod.get_submodule(name.rpartition(".")[0]) if "." in name else mod
            setattr(owner, name.rpartition(".")[2], b.float())


def build_input(M, args, idx, n_tok, hidden: Path | None):
    import torch
    if hidden is not None:
        blob = torch.load(hidden, map_location="cpu")
        return blob["h"][:n_tok].unsqueeze(0), blob["pre_mix"][:n_tok].unsqueeze(0), None
    from ops.lib.tok import Tok
    ids = Tok(env.SRC).encode(PROMPT * (n_tok // 30 + 2))[:n_tok]
    assert len(ids) == n_tok
    ids = torch.tensor([ids])
    emb = torch.nn.functional.embedding(ids, idx.get("embed.weight"))
    h = emb.unsqueeze(2).repeat(1, 1, args.hc_mult, 1)
    return h, M.make_identity_pre_mix(h, args.hc_mult), ids


def run_arm(M, args, idx, layers, h, pre_mix, device, *, oracle: bool, reps: int):
    """Build `layers` on device, run them in order, return the last layer's outputs and timing."""
    import torch
    from ops.lib import shards
    torch.set_default_dtype(torch.bfloat16)
    blocks = [shards.build_block(M, args, L, idx, device, with_engram=False) for L in layers]
    if oracle:
        for b in blocks:
            promote_fp32(b)
        torch.set_default_dtype(torch.float32)
        h, pre_mix = h.float(), pre_mix.float()
    routes = {}

    def cap(mod, inp, out):
        routes[mod.layer_id] = out[1].detach().cpu()
    hooks = []
    for b in blocks:
        b.ffn.gate.layer_id = b.layer_id
        hooks.append(b.ffn.gate.register_forward_hook(cap))
    h, pre_mix = h.to(device), pre_mix.to(device)
    # model.py makes some tensors without a device (get_window_topk_idxs, lru-cached): run with
    # the arm's device as default, as generate.py does, and never reuse the other arm's cache
    torch.set_default_device(device)
    M.get_window_topk_idxs.cache_clear()
    times = []
    with torch.inference_mode():
        for rep in range(1 if oracle else reps + 1):
            x, pm = h, pre_mix
            for b in blocks[:-1]:
                x, pm = b(x, 0, pm, None)
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            y, ffn_pre = blocks[-1](x, 0, pm, None)
            if device == "cuda":
                torch.cuda.synchronize()
            if rep > 0 or oracle:
                times.append(time.perf_counter() - t0)
    routes["_memavail_gib"] = env.mem_available_gib()     # layers + last activations still resident
    for hk in hooks:
        hk.remove()
    torch.set_default_device("cpu")
    out = (y.float().cpu(), ffn_pre.float().cpu(), routes, times)
    del blocks
    torch.set_default_dtype(torch.bfloat16)
    return out


def compare(a, b):
    import torch
    a, b = a.flatten().double(), b.flatten().double()
    return {"cos": float(torch.nn.functional.cosine_similarity(a, b, dim=0)),
            "rel_l2": float((a - b).norm() / b.norm().clamp_min(1e-30)),
            "max_abs": float((a - b).abs().max()), "finite": bool(torch.isfinite(a).all())}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--tokens", type=int, default=2048, help="sequence length; > --block-chunk exercises the long-row path")
    ap.add_argument("--kernels", choices=["tilelang", "torch"], default="tilelang")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--hidden", type=Path, help="recorded {'h': [S,4,5120], 'pre_mix': [S,4]} input")
    ap.add_argument("--reference-indexer", action="store_true",
                    help="GPU arm uses the dense reference Indexer instead of s03's query-chunked one")
    ap.add_argument("--block-chunk", type=int, default=8192,
                    help="GPU arm runs s03's token-chunked Block when tokens > this (e.g. --tokens 4096 --block-chunk 1024)")
    ap.add_argument("--no-oracle", action="store_true", help="skip the CPU fp32 oracle arm")
    ap.add_argument("--out", type=Path, default=env.LOGS / "s01_layer_probe.json")
    a = ap.parse_args()
    env.assert_no_gpu_server("s01")

    import torch
    try:
        M = env.import_reference(a.kernels)
    except Exception as e:  # TileLang missing or failing to import is itself the s01 answer
        env.atomic_json(a.out, {"stage": "s01_layer_probe", "ts": env.now(), "kernels": a.kernels,
                                "import_error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-4000:]})
        env.log(f"reference import failed with kernels={a.kernels}: {e}; rerun with --kernels torch", "s01", "ERROR")
        raise SystemExit(3)
    from ops.lib import shards, torch_kernels
    args = env.model_args(M, max_batch_size=1, max_seq_len=max(8192, a.tokens))
    L = a.layer
    src_kv = max(s for s in args.kv_source_layers if s <= L)
    src_ix = max(s for s in args.index_source_layers if s <= L)
    layers = sorted({src_kv, src_ix, L}) if a.hidden is None else [L]
    assert args.compress_ratios[L] and a.tokens % args.compress_ratios[L] == 0
    rep = {"stage": "s01_layer_probe", "ts": env.now(), "layer": L, "run_layers": layers, "tokens": a.tokens,
           "kernels": a.kernels, "input": str(a.hidden) if a.hidden else "embed(prompt) -> 4 streams, layers 0-1 skipped",
           "torch": torch.__version__, "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None}
    idx = shards.ShardIndex()
    h, pre_mix, _ = build_input(M, args, idx, a.tokens, a.hidden)
    mem0 = env.mem_available_gib()

    ref_indexer, ref_block = M.Indexer.forward, M.Block.forward
    try:
        torch_kernels.install(M, a.kernels)
        if not a.reference_indexer:
            from ops.lib.hooks import chunked_indexer
            chunked_indexer(M)
        rep["gpu_indexer"] = "reference" if a.reference_indexer else "chunked (s03)"
        from ops.lib.hooks import chunked_block
        chunked_block(M, a.block_chunk)
        rep["gpu_block"] = f"chunked at {a.block_chunk} (active: {a.tokens > a.block_chunk})"
        torch.cuda.reset_peak_memory_stats()
        y, fp, routes, times = run_arm(M, args, idx, layers, h, pre_mix, "cuda", oracle=False, reps=a.reps)
        rep["gpu"] = {"layer_s": statistics.median(times), "tok_per_s": a.tokens / statistics.median(times),
                      "times": times, "peak_cuda_gib": torch.cuda.max_memory_allocated() / 2**30,
                      "memavail_drop_gib": mem0 - routes.pop("_memavail_gib")}
        torch.cuda.empty_cache()
    except Exception as e:
        rep["gpu"] = {"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-4000:]}
        env.atomic_json(a.out, rep)
        env.log(f"GPU arm failed ({a.kernels}): {e}", "s01", "ERROR")
        raise SystemExit(2)

    if not a.no_oracle:
        torch_kernels.install(M, "torch")
        M.Indexer.forward, M.Block.forward = ref_indexer, ref_block  # the oracle is the unmodified reference
        yo, fpo, ro, to = run_arm(M, args, idx, layers, h, pre_mix, "cpu", oracle=True, reps=1)
        ro.pop("_memavail_gib")
        rep["oracle"] = {"layer_s": to[0]}
        rep["vs_oracle"] = {"stream": compare(y, yo), "ffn_pre": compare(fp, fpo)}
        agree = {}
        for lid in routes:
            g, o = routes[lid].sort(-1).values, ro[lid].sort(-1).values
            agree[lid] = {"exact_set": float((g == o).all(-1).float().mean()),
                          "slot_overlap": float((g.unsqueeze(-1) == o.unsqueeze(-2)).any(-1).float().mean())}
        rep["routing_agreement"] = agree
        c = rep["vs_oracle"]["stream"]["cos"]
        rep["verdict"] = "PASS" if c >= 0.999 and rep["vs_oracle"]["stream"]["finite"] else "INVESTIGATE"
    env.atomic_json(a.out, rep)
    env.log(json.dumps({k: rep.get(k) for k in ("gpu", "vs_oracle", "verdict")}, default=str)[:2000], "s01")


if __name__ == "__main__":
    main()
