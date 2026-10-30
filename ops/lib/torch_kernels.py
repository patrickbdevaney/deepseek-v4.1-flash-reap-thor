"""Pure-torch equivalents of the reference TileLang kernels (inference/kernel.py).

Two jobs:
  1. The oracle. s01 runs a layer once with these on CPU in fp32 and once with TileLang on the
     GPU, and compares. A kernel that miscompiles for sm_110a shows up as a cosine gap.
  2. The fallback. If tilelang==0.1.12 does not build for sm_110a (LibertAI needed kernel patches
     on GB10), `install(M, "torch")` swaps these into the reference module so calibration can run.

Each function reproduces the kernel's arithmetic, including its clamps, scale rounding and the
bf16 cast of the unnormalised softmax weights in sparse_attn. Differences that remain are fp32
summation order (tiled vs. whole-K) and the fp4 round-to-nearest tie rule, which s01 measures.
"""
from __future__ import annotations

import sys

import torch
import torch.nn.functional as F

FP8_MAX = 448.0
FP4_MAX = 6.0
FP4_TABLE = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                          -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=torch.float32)


def pow2_ceil(v: torch.Tensor) -> torch.Tensor:
    """2 ** ceil(log2(v)) by the kernel's bit trick: exponent + 1 if any mantissa bit is set."""
    bits = v.float().contiguous().view(torch.int32)
    exp = ((bits >> 23) & 0xFF) - 127
    man = bits & ((1 << 23) - 1)
    e = exp + (man != 0).to(torch.int32)
    return torch.ldexp(torch.ones_like(v, dtype=torch.float32), e)


def _groups(x: torch.Tensor, block: int) -> torch.Tensor:
    return x.unflatten(-1, (-1, block))


def act_quant(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
    g = _groups(x.float(), block_size)
    amax = g.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    s = pow2_ceil(amax / FP8_MAX) if scale_fmt is not None else amax / FP8_MAX
    q = (g / s).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    if inplace:
        x.copy_((q.float() * s).flatten(-2).to(x.dtype))
        return x
    return q.flatten(-2), s.squeeze(-1).to(scale_dtype)


def fp4_round(v: torch.Tensor) -> torch.Tensor:
    """Round to the nearest e2m1 value, ties to the even code (the PTX cvt.rn rule)."""
    a = v.abs()
    # code boundaries; at an exact midpoint the even code wins
    out = torch.zeros_like(a)
    out = torch.where(a > 0.25, torch.full_like(a, 0.5), out)
    out = torch.where(a >= 0.75, torch.full_like(a, 1.0), out)
    out = torch.where(a > 1.25, torch.full_like(a, 1.5), out)
    out = torch.where(a >= 1.75, torch.full_like(a, 2.0), out)
    out = torch.where(a > 2.5, torch.full_like(a, 3.0), out)
    out = torch.where(a >= 3.5, torch.full_like(a, 4.0), out)
    out = torch.where(a > 5.0, torch.full_like(a, 6.0), out)
    return torch.copysign(out, v)


def fp4_act_quant(x, block_size=32, inplace=False, scale_dtype=torch.float8_e8m0fnu):
    g = _groups(x.float(), block_size)
    amax = g.abs().amax(-1, keepdim=True)
    if scale_dtype == torch.float8_e4m3fn:
        amax = amax.clamp_min(6 * 2 ** -9)
        s = (amax / FP4_MAX).to(torch.float8_e4m3fn).float()
    else:
        amax = amax.clamp_min(6 * 2 ** -126)
        s = pow2_ceil(amax / FP4_MAX)
    q = fp4_round((g / s).clamp(-FP4_MAX, FP4_MAX))
    if inplace:
        x.copy_((q * s).flatten(-2).to(x.dtype))
        return x
    raise NotImplementedError("packed fp4 output is not used by the forward pass")


def unpack_fp4(w: torch.Tensor) -> torch.Tensor:
    """[N, K//2] float4_e2m1fn_x2 (low nibble first, as convert.py packs) -> [N, K] fp32."""
    b = w.view(torch.uint8)
    t = FP4_TABLE.to(b.device)
    return torch.stack([t[(b & 0x0F).long()], t[(b >> 4).long()]], dim=-1).flatten(-2)


def dequant_fp4(w, scale, block=32):
    v = unpack_fp4(w)
    return (_groups(v, block) * scale.float().unsqueeze(-1)).flatten(-2)


def dequant_fp8(w, scale):
    """Block-scaled fp8 weight [N, K] with scale [ceil(N/bo), ceil(K/bi)] -> fp32."""
    N, K = w.shape
    bo = -(-N // scale.shape[0])
    bi = -(-K // scale.shape[1])
    s = scale.float().repeat_interleave(bo, 0)[:N].repeat_interleave(bi, 1)[:, :K]
    return w.float() * s


def _deq_act(a, a_s, K):
    blk = K // a_s.shape[-1]
    return (_groups(a.float(), blk) * a_s.float().unsqueeze(-1)).flatten(-2)


def fp8_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, block_size=128):
    K = a.size(-1)
    out = _deq_act(a, a_s.view(*a.shape[:-1], -1), K) @ dequant_fp8(b, b_s).t()
    return out.to(torch.get_default_dtype())


def fp4_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, act_block_size=128):
    K = a.size(-1)
    out = _deq_act(a, a_s.view(*a.shape[:-1], -1), K) @ dequant_fp4(b, b_s).t()
    return out.to(torch.get_default_dtype())


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale, q_chunk: int = 256):
    """o[b,m,h,:] = softmax over the gathered top-k kv rows plus a per-head sink logit.

    Matches the kernel: masked slots score -inf, the running max starts at -1e30 so an all-masked
    row returns zeros, the sink enters the denominator only, and the exp weights are cast to bf16
    before the PV product. Chunked over queries so an 8k prefill does not gather 5 GiB of kv."""
    b, m, h, d = q.shape
    out = torch.empty_like(q)
    for s0 in range(0, m, q_chunk):
        s1 = min(m, s0 + q_chunk)
        idx = topk_idxs[:, s0:s1].to(kv.device).long()          # [b, c, k]; index helpers build on cpu
        valid = idx >= 0
        g = torch.gather(kv.unsqueeze(1).expand(b, s1 - s0, *kv.shape[1:]), 2,
                         idx.clamp_min(0).unsqueeze(-1).expand(-1, -1, -1, d))  # [b, c, k, d]
        g = torch.where(valid.unsqueeze(-1), g, torch.zeros_like(g))
        sc = torch.einsum("bchd,bckd->bchk", q[:, s0:s1].float(), g.float()) * softmax_scale
        sc = sc.masked_fill(~valid.unsqueeze(2), float("-inf"))
        mx = sc.amax(-1, keepdim=True).clamp_min(-1e30)
        p = torch.exp(sc - mx)
        den = p.sum(-1) + torch.exp(attn_sink.float().view(1, 1, h) - mx.squeeze(-1))
        o = torch.einsum("bchk,bckd->bchd", p.to(torch.bfloat16).float(), g.float())
        out[:, s0:s1] = (o / den.unsqueeze(-1)).to(q.dtype)
    return out


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    hc = hc_mult
    m = mixes.float()
    pre = torch.sigmoid(m[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * torch.sigmoid(m[..., hc:2 * hc] * hc_scale[1] + hc_base[hc:2 * hc])
    comb = (m[..., 2 * hc:] * hc_scale[2] + hc_base[2 * hc:]).unflatten(-1, (hc, hc))
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


NAMES = ("act_quant", "fp4_act_quant", "fp4_gemm", "fp8_gemm", "hc_split_sinkhorn", "sparse_attn")


def install(M, which: str) -> None:
    """Point the reference module's kernel names at TileLang ("tilelang") or at this file
    ("torch"). model.py binds them with `from kernel import ...`, so patch the module globals."""
    if which == "tilelang":
        import kernel as K
        src = K
    elif which == "torch":
        src = sys.modules[__name__]
    else:
        raise ValueError(which)
    for n in NAMES:
        setattr(M, n, getattr(src, n))

