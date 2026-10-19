"""kernel.py — pure-PyTorch replacements for the reference tilelang kernels.

Shadows the reference `inference/kernel.py` so `deepseek_v4_ref.py` (a copy of the reference
model.py) runs WITHOUT tilelang / fast_hadamard_transform, which don't build on Thor sm_110a.

These implement the SAME math as the tilelang kernels (kernel.py in the DSpark repo), computed
in fp32/bf16, so their outputs are the numerical golden reference the CUDA kernels gate against.
Faithful to the documented op (see reference/DEEPSEEK_V4_MODELING_NOTES.md §1,§2,§4).
"""
import torch
import torch.nn.functional as F
from typing import Optional

FP8_MAX = 448.0
FP4_MAX = 6.0
# E2M1 codebook (convert.py FP4_TABLE): nibble -> value. High bit = sign.
_E2M1 = torch.tensor([0.,0.5,1.,1.5,2.,3.,4.,6., 0.,-0.5,-1.,-1.5,-2.,-3.,-4.,-6.], dtype=torch.float32)


def _round_pow2(x: torch.Tensor) -> torch.Tensor:
    """Round scale up to the next power of two (ue8m0 / MXFP behaviour)."""
    return torch.pow(2.0, torch.ceil(torch.log2(x.clamp_min(1e-30))))


def _blocks(x: torch.Tensor, block: int):
    N = x.size(-1)
    assert N % block == 0, (N, block)
    return x.unflatten(-1, (N // block, block))


def act_quant(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
    """Block-wise FP8 activation quant (kernel.py:40-125).
    inplace=True -> fused quant+dequant back to input dtype (QAT sim), returns x.
    else -> (y_fp8, scale)."""
    dtype = x.dtype
    xb = _blocks(x.float(), block_size)                         # [...,nb,block]
    amax = xb.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)  # [...,nb,1]
    s = amax * (1.0 / FP8_MAX)
    if scale_fmt is not None:
        s = _round_pow2(s)
    q = torch.clamp(xb / s, -FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    if inplace:
        deq = (q.float() * s).flatten(-2).to(dtype)
        x.copy_(deq.view_as(x))
        return x
    y = q.flatten(-2)
    return y, s.squeeze(-1).to(scale_dtype)


def fp4_act_quant(x, block_size=32, inplace=False):
    """Block-wise FP4 activation quant (kernel.py:128-200). Rounds to E2M1 grid.
    inplace=True -> fused quant+dequant back to bf16."""
    dtype = x.dtype
    xb = _blocks(x.float(), block_size)
    amax = xb.abs().amax(dim=-1, keepdim=True).clamp_min(6 * (2 ** -126))
    s = _round_pow2(amax * (1.0 / FP4_MAX))
    scaled = torch.clamp(xb / s, -FP4_MAX, FP4_MAX)
    q = _round_e2m1(scaled)
    if inplace:
        deq = (q * s).flatten(-2).to(dtype)
        x.copy_(deq.view_as(x))
        return x
    # non-inplace packing path not needed by the reference forward
    return (q * s).flatten(-2).to(dtype), s.squeeze(-1)


def _round_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Round to nearest representable E2M1 value (grid ±{0,.5,1,1.5,2,3,4,6})."""
    grid = _E2M1.to(x.device)
    # nearest-neighbour on the sorted unique magnitudes
    d = (x.unsqueeze(-1) - grid.view(*([1] * x.ndim), -1)).abs()
    return grid[d.argmin(dim=-1)]


def _dequant_fp8(w_fp8: torch.Tensor, scale: torch.Tensor, block=128) -> torch.Tensor:
    """Dequant a [out,in] fp8 weight with per-[128,128]-block e8m0/f32 scale -> fp32."""
    out, inn = w_fp8.shape
    w = w_fp8.float().view(out // block, block, inn // block, block)
    s = scale.float().view(out // block, 1, inn // block, 1)
    return (w * s).view(out, inn)


def _dequant_fp4(w_packed: torch.Tensor, scale: torch.Tensor, block=32) -> torch.Tensor:
    """Dequant [out,in//2] packed-fp4 (uint8: 2 nibbles) with per-32 e8m0 scale -> fp32 [out,in]."""
    b = w_packed.view(torch.uint8)
    lo = _E2M1.to(b.device)[(b & 0x0F).long()]
    hi = _E2M1.to(b.device)[((b >> 4) & 0x0F).long()]
    w = torch.stack([lo, hi], dim=-1).flatten(-2)              # [out,in]
    out, inn = w.shape
    s = scale.float().view(out, inn // block, 1)
    return (w.view(out, inn // block, block) * s).view(out, inn)


def fp8_gemm(a_fp8, a_s, b_fp8, b_s, scale_dtype=torch.float32):
    """C[M,N] = A[M,K] @ B[N,K]^T, per-128 block fp8 scales (kernel.py:203-273)."""
    K = a_fp8.size(-1)
    a = a_fp8.float().unflatten(-1, (K // 128, 128)) * a_s.float().unsqueeze(-1)
    a = a.flatten(-2)                                           # [...,K]
    b = _dequant_fp8(b_fp8, b_s)                                # [N,K]
    return (a @ b.t().to(a.dtype)).to(torch.get_default_dtype())


def fp4_gemm(a_fp8, a_s, b_fp4, b_s, scale_dtype=torch.float32):
    """C[M,N] = A_fp8[M,K] @ B_fp4[N,K]^T (kernel.py:441-536)."""
    K = a_fp8.size(-1)
    a = a_fp8.float().unflatten(-1, (K // 128, 128)) * a_s.float().unsqueeze(-1)
    a = a.flatten(-2)
    b = _dequant_fp4(b_fp4, b_s)                                # [N,K]
    return (a @ b.t().to(a.dtype)).to(torch.get_default_dtype())


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    """Gathered sparse attention with a learnable sink (kernel.py:276-368).
    q:[b,m,h,d] kv:[b,n,d] attn_sink:[h] topk_idxs:[b,m,topk] (-1 => masked). o:[b,m,h,d]."""
    b, m, h, d = q.shape
    topk = topk_idxs.size(-1)
    idx = topk_idxs.clamp_min(0)                                # gather-safe
    valid = (topk_idxs >= 0)                                    # [b,m,topk]
    kg = torch.gather(kv.unsqueeze(1).expand(b, m, kv.size(1), d), 2,
                      idx.unsqueeze(-1).expand(b, m, topk, d).long())  # [b,m,topk,d]
    scores = torch.einsum("bmhd,bmkd->bmhk", q.float(), kg.float()) * softmax_scale
    scores = scores.masked_fill(~valid.unsqueeze(2), float("-inf"))
    # sink: an extra column with logit = attn_sink[h] (its "key" contributes 0 to the output)
    sink = attn_sink.float().view(1, 1, h, 1).expand(b, m, h, 1)
    alls = torch.cat([scores, sink], dim=-1)
    p = torch.softmax(alls, dim=-1)[..., :topk]                 # drop sink prob from the value mix
    o = torch.einsum("bmhk,bmkd->bmhd", p, kg.float())
    return o.to(q.dtype)


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    """Hyper-Connections pre/post/comb from 24 mixes/token, comb made doubly-stochastic
    via Sinkhorn (kernel.py:371-438)."""
    *lead, mh = mixes.shape
    hc = hc_mult
    m = mixes.float()
    pre = torch.sigmoid(m[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * torch.sigmoid(m[..., hc:2 * hc] * hc_scale[1] + hc_base[hc:2 * hc])
    comb = m[..., 2 * hc:] * hc_scale[2] + hc_base[2 * hc:]
    comb = comb.view(*lead, hc, hc)
    # row-softmax + eps
    comb = torch.softmax(comb, dim=-1) + eps
    # col-normalize
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb
