"""raw_loader.py — load raw HF safetensors shards into the reference model modules.

The reference `generate.py` loads a convert.py-produced mp1 file; we instead read the raw
0xSero/DeepSeek-V4-Flash-180B shards directly and populate module params in the dtypes
`deepseek_v4_ref.py` expects:
  - fp8 linears (wq_a/wq_b/wkv/wo_b/e_proj/h_proj/shared_experts): weight float8_e4m3fn + .scale
  - fp4 routed experts (w1/w2/w3): weight float4_e2m1fn_x2 (view of I8) + .scale (e8m0)
  - wo_a: raw is fp8; the model declares bf16 -> DEQUANT to bf16 on load (matches convert.py)
  - bf16 (embed/head/norms/gate/indexer.weights_proj/compressor.*): as-is
  - f32 (hc_*/attn_sink/gate.bias/ape): as-is

Only loads the module names present in the constructed model, so single-layer / single-block
reference runs touch just a few GB of weights.
"""
import os, json, glob, struct
import torch
from safetensors import safe_open

DEQUANT_TO_BF16 = ("wo_a",)   # names whose raw fp8 must be dequantized to bf16 for the ref module


def _build_index(ckpt_dir):
    idx = json.load(open(os.path.join(ckpt_dir, "model.safetensors.index.json")))
    return idx["weight_map"]           # name -> shard file


def _dequant_fp8_to_bf16(w, scale, block=128):
    out, inn = w.shape
    wf = w.float().view(out // block, block, inn // block, block)
    s = scale.float().view(out // block, 1, inn // block, 1)
    return (wf * s).view(out, inn).bfloat16()


def load_state_into(model, ckpt_dir, only_prefixes=None, verbose=True):
    """Assign raw checkpoint tensors to model params/buffers by name.
    only_prefixes: iterable of name prefixes to load (None = all present in model).
    Returns (loaded, skipped_missing)."""
    wm = _build_index(ckpt_dir)
    # group wanted names by shard to open each shard once
    model_names = dict(model.named_parameters())
    model_names.update(dict(model.named_buffers()))
    want = set()
    for n in model_names:
        if only_prefixes and not any(n.startswith(p) for p in only_prefixes):
            continue
        want.add(n)

    # a param at "<mod>.weight" quantized has a sibling scale at "<mod>.scale" in the ckpt
    by_shard = {}
    for n in list(want):
        for key in (n, n.replace(".weight", ".scale")):
            if key in wm:
                by_shard.setdefault(wm[key], []).append(key)

    loaded, missing = [], []
    scales = {}
    tensors = {}
    for shard, keys in by_shard.items():
        with safe_open(os.path.join(ckpt_dir, shard), framework="pt", device="cpu") as f:
            avail = set(f.keys())
            for k in keys:
                if k in avail:
                    tensors[k] = f.get_tensor(k)

    with torch.no_grad():
        for n in want:
            if n not in tensors:
                missing.append(n); continue
            w = tensors[n]
            scale = tensors.get(n.replace(".weight", ".scale"))
            leaf = n.split(".")[-2] if n.endswith(".weight") else n.split(".")[-1]
            tgt = model_names[n]
            if any(d in n for d in DEQUANT_TO_BF16) and scale is not None:
                val = _dequant_fp8_to_bf16(w, scale)
            elif w.dtype == torch.int8:
                # packed FP4 experts: reinterpret I8 bytes as float4_e2m1fn_x2
                val = w.view(torch.float4_e2m1fn_x2)
            else:
                val = w
            if val.shape != tgt.shape and val.numel() == tgt.numel():
                val = val.view(tgt.shape)
            tgt.copy_(val.to(tgt.dtype) if tgt.dtype not in
                      (torch.float8_e4m3fn, torch.float4_e2m1fn_x2) else val)
            if scale is not None and hasattr(tgt, "scale"):
                tgt.scale.copy_(scale.to(tgt.scale.dtype))
            elif scale is not None:
                # attach as attribute so linear() can find weight.scale
                tgt.scale = scale
            loaded.append(n)
    if verbose:
        print(f"[raw_loader] loaded={len(loaded)} missing={len(missing)} (of {len(want)} wanted)")
        for m in missing[:20]:
            print("   MISSING", m)
    return loaded, missing
