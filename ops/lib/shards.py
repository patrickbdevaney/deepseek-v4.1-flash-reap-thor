"""Read 1017 tensors straight from the HF shards and build one reference Block at a time.

Layout (model.safetensors.index.json at the pinned revision): shard 1 vision+aligner, shard 2
embed + image delimiters, shard L+3 = decoder layer L (0..39), shard 43 head+norm, 44-46 the three
DSpark/MTP layers, 47 and 48 the Engram tables of layers 1 and 14. Names are already in the
inference module's naming, so the only transforms convert.py applies at mp=1 are reproduced here:
  * `attn.wo_a` is dequantised to bf16 (the reference uses it in an einsum, not a Linear);
  * routed-expert int8 tensors are reinterpreted as float4_e2m1fn_x2.
The Engram `embed` table (98 GB per layer) is never loaded: Blocks are built with a row source
that serves pre-gathered rows (see engram_rows.py).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import torch
from safetensors import safe_open

from .env import SRC


class ShardIndex:
    def __init__(self, src: Path = SRC):
        self.src = Path(src)
        self.map = json.load(open(self.src / "model.safetensors.index.json"))["weight_map"]
        self._open = {}

    def shard_of(self, name: str) -> str:
        return self.map[name]

    def names(self, prefix: str, *, exclude: tuple[str, ...] = ()) -> list[str]:
        return sorted(n for n in self.map if n.startswith(prefix) and not any(e in n for e in exclude))

    def shards_for(self, prefix: str) -> set[str]:
        return {self.map[n] for n in self.names(prefix)}

    def _f(self, shard: str):
        if shard not in self._open:
            self._open[shard] = safe_open(str(self.src / shard), framework="pt", device="cpu")
        return self._open[shard]

    def get(self, name: str) -> torch.Tensor:
        return self._f(self.map[name]).get_tensor(name)

    def close(self):
        self._open.clear()

    def state_dict(self, prefix: str, *, exclude: tuple[str, ...] = ()) -> dict[str, torch.Tensor]:
        """All tensors under `prefix`, renamed relative to it, with convert.py's mp=1 transforms."""
        sd = {}
        for n in self.names(prefix, exclude=exclude):
            sd[n[len(prefix):]] = self.get(n)
        for k in [k for k in sd if k.endswith("wo_a.weight")]:
            w, s = sd[k], sd.pop(k.replace("weight", "scale"))
            bo, bi = w.size(0) // s.size(0), w.size(1) // s.size(1)
            assert (bo, bi) in ((32, 32), (128, 128)), (k, w.shape, s.shape)
            w = w.unflatten(0, (-1, bo)).unflatten(-1, (-1, bi)).float() * s[:, None, :, None].float()
            sd[k] = w.flatten(2, 3).flatten(0, 1).bfloat16()
        for k in list(sd):
            if re.search(r"(^|\.)experts\.\d+\.w[123]\.weight$", k) and sd[k].dtype == torch.int8:
                sd[k] = sd[k].view(torch.float4_e2m1fn_x2)
        return sd


def shard_for_layer(layer: int) -> str:
    return f"model-{layer + 3:05d}-of-00048.safetensors"


class RowSource(torch.nn.Module):
    """Stands in for ParallelEngramEmbedding: returns the bf16 rows for the current batch,
    dequantised exactly as the reference does (fp8 values x per-32 E8M0 scales)."""

    def __init__(self, block_size: int = 32):
        super().__init__()
        self.block_size = block_size
        self.rows_fp8 = None     # [B, L, n_cols, head_dim] float8_e4m3fn (uint8 view on disk)
        self.rows_scale = None   # [B, L, n_cols, head_dim // block] float8_e8m0fnu

    def forward(self, indices):  # indices are ignored: the rows were gathered from them already
        v = self.rows_fp8.float().unflatten(-1, (-1, self.block_size)) * self.rows_scale.float().unsqueeze(-1)
        return v.flatten(-2).to(torch.bfloat16)


def build_block(M, args, layer: int, idx: ShardIndex, device: str, *, with_engram: bool = True):
    """Construct reference Block `layer` on `device` with its weights loaded (strict)."""
    layout = M.EngramLayout.from_args(args) if with_engram else None
    torch.set_default_dtype(torch.bfloat16)
    # precompute_freqs_cis is lru_cached in model.py: without this a CPU block built after a CUDA one
    # registers the CUDA freqs_cis tensor as its buffer (s01 oracle crash, 2026-10-19)
    M.precompute_freqs_cis.cache_clear()
    if layout is not None and layer in layout.layer_ids:
        with torch.device("meta"):
            blk = M.Block(layer, args, layout)
        blk.engram.embed = RowSource(M.fp8_block_size)
        blk = blk.to_empty(device=device)
        _reinit_buffers(M, blk, args, device)
    else:
        with torch.device(device):
            blk = M.Block(layer, args, None)
    sd = idx.state_dict(f"layers.{layer}.", exclude=(".engram.embed.",))
    missing, unexpected = blk.load_state_dict(sd, strict=False)
    missing = [m for m in missing if not m.startswith("engram.embed.")]
    if missing or unexpected:
        raise RuntimeError(f"layer {layer}: missing {missing[:8]} unexpected {unexpected[:8]}")
    _reattach_scales(blk)
    blk.eval()
    return blk


def _reattach_scales(mod):
    """Linear stores `weight.scale = self.scale`; load_state_dict keeps the Parameter objects, but
    to_empty() replaces them, so re-link every quantized Linear's weight to its scale."""
    for m in mod.modules():
        if hasattr(m, "weight") and getattr(m, "scale", None) is not None and isinstance(m.scale, torch.nn.Parameter):
            m.weight.scale = m.scale


def _reinit_buffers(M, blk, args, device):
    """to_empty() leaves non-persistent buffers uninitialised; rebuild the ones whose initial
    values matter (freqs_cis, compressor score_state = -inf, zeroed caches)."""
    M.precompute_freqs_cis.cache_clear()   # construction ran under the meta device and cached meta freqs
    for mod in blk.modules():
        if isinstance(mod, M.Attention):
            ref_freqs = M.precompute_freqs_cis(
                mod.rope_head_dim, args.max_seq_len,
                args.original_seq_len if mod.compress_ratio else 0,
                args.compress_rope_theta if mod.compress_ratio else args.rope_theta,
                args.rope_factor, args.beta_fast, args.beta_slow)
            mod.freqs_cis = ref_freqs.to(device)
            mod.window_kv_cache.zero_()
            if hasattr(mod, "compress_kv_cache"):
                mod.compress_kv_cache.zero_()
        if isinstance(mod, M.Compressor) and mod.compress_ratio > 1:
            mod.kv_state.zero_()
            mod.score_state.fill_(float("-inf"))
        if isinstance(mod, M.Indexer) and mod.owns_k:
            mod.k_cache.zero_()
