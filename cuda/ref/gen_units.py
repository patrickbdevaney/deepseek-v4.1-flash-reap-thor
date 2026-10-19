"""gen_units.py — micro-op golden generators that directly gate individual CUDA kernels.

Deterministic small inputs -> reference outputs, saved as safetensors the host CUDA gate reads.
Covers the two hardest net-new primitives first: FP8 128-block GEMM and HC/Sinkhorn.

  docker run --rm --network none -e CUDA_VISIBLE_DEVICES="" -v $PWD/ref:/ref \
    vllm-dflash-thor:sglang python3 /ref/gen_units.py --out /ref/goldens
"""
import os, sys, argparse
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kernel as K
from safetensors.torch import save_file


def _round_pow2(x):
    return torch.pow(2.0, torch.ceil(torch.log2(x.clamp_min(1e-30))))


def weight_quant_fp8(W, block=128):
    """Quantize [out,in] bf16 weight to fp8 e4m3 with per-128x128-block power-of-2 scale (e8m0-style)."""
    out, inn = W.shape
    Wb = W.float().view(out // block, block, inn // block, block)
    amax = Wb.abs().amax(dim=(1, 3), keepdim=True).clamp_min(1e-6)
    s = _round_pow2(amax / K.FP8_MAX)                               # [out/128,1,in/128,1]
    q = torch.clamp(Wb / s, -K.FP8_MAX, K.FP8_MAX).to(torch.float8_e4m3fn)
    return q.view(out, inn), s.view(out // block, inn // block)


def gen_fp8_gemm(out_dir, M=48, N=256, K_=512):
    torch.manual_seed(7)
    A = torch.randn(M, K_)                                          # bf16 activation
    A_fp8, a_s = K.act_quant(A, 128, None, torch.float32)          # [M,K] fp8, [M,K/128] f32
    W = torch.randn(N, K_) * 0.05
    b_fp8, b_s = weight_quant_fp8(W, 128)                          # [N,K] fp8, [N/128,K/128] f32
    C = K.fp8_gemm(A_fp8, a_s, b_fp8, b_s)                         # reference [M,N] (default bf16)
    save_file({
        "A_fp8": A_fp8.contiguous(), "a_s": a_s.contiguous().float(),
        "B_fp8": b_fp8.contiguous(), "b_s": b_s.contiguous().float(),
        "C_ref": C.contiguous().float(),
        "dims": torch.tensor([M, N, K_], dtype=torch.int32),
    }, os.path.join(out_dir, "unit_fp8_gemm.safetensors"))
    print(f"[fp8_gemm] M={M} N={N} K={K_}  C {tuple(C.shape)} |C|max={C.abs().max():.4f}")


def gen_hc_sinkhorn(out_dir, n=64, hc_mult=4, iters=20, eps=1e-6):
    torch.manual_seed(11)
    mh = (2 + hc_mult) * hc_mult                                    # 24
    mixes = torch.randn(n, mh)
    hc_scale = torch.randn(3) * 0.5
    hc_base = torch.randn(mh) * 0.5
    pre, post, comb = K.hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult, iters, eps)
    save_file({
        "mixes": mixes.contiguous().float(), "hc_scale": hc_scale.contiguous().float(),
        "hc_base": hc_base.contiguous().float(),
        "pre": pre.contiguous().float(), "post": post.contiguous().float(),
        "comb": comb.contiguous().float(),
        "params": torch.tensor([n, hc_mult, iters], dtype=torch.int32),
    }, os.path.join(out_dir, "unit_hc_sinkhorn.safetensors"))
    # sanity: comb rows/cols should be ~doubly-stochastic
    print(f"[hc_sinkhorn] n={n} comb row_sum~{comb.sum(-1).mean():.4f} col_sum~{comb.sum(-2).mean():.4f} "
          f"pre {tuple(pre.shape)} post {tuple(post.shape)} comb {tuple(comb.shape)}")


def gen_sparse_attn(out_dir, b=1, m=8, h=4, d=128, n=24, topk=16):
    """Gathered top-k attention with a learnable sink (kernel.py:276-368). Single latent KV shared
    across heads (MLA: 1 kv, h q-heads). Some topk_idxs = -1 (masked)."""
    torch.manual_seed(21)
    q = torch.randn(b, m, h, d, dtype=torch.bfloat16)
    kv = torch.randn(b, n, d, dtype=torch.bfloat16)
    attn_sink = torch.randn(h, dtype=torch.float32)
    # per (b,m): topk indices into [0,n), with a few -1 masks
    idx = torch.randint(0, n, (b, m, topk), dtype=torch.int32)
    mask = torch.rand(b, m, topk) < 0.25
    idx = torch.where(mask, torch.full_like(idx, -1), idx)
    scale = d ** -0.5
    o = K.sparse_attn(q, kv, attn_sink, idx, scale)          # ref -> [b,m,h,d] bf16
    save_file({
        "q": q.contiguous().float(), "kv": kv.contiguous().float(),
        "attn_sink": attn_sink.contiguous(), "topk_idxs": idx.contiguous(),
        "o_ref": o.contiguous().float(),
        "dims": torch.tensor([b, m, h, d, n, topk], dtype=torch.int32),
        "scale": torch.tensor([scale], dtype=torch.float32),
    }, os.path.join(out_dir, "unit_sparse_attn.safetensors"))
    print(f"[sparse_attn] b={b} m={m} h={h} d={d} n={n} topk={topk}  o {tuple(o.shape)} |o|max={o.abs().max():.4f}")


def _rope_ref(x, cos, sin, inverse=False):
    # x:[n,D] (D even), cos/sin:[n,D/2]; matches apply_rotary_emb (interleaved pairs, view_as_complex)
    xc = x.float().reshape(x.shape[0], -1, 2)
    xr, xi = xc[..., 0], xc[..., 1]
    s = -sin if inverse else sin
    yr = xr * cos - xi * s
    yi = xr * s + xi * cos
    return torch.stack([yr, yi], dim=-1).reshape(x.shape)


def gen_rope(out_dir, n=32, rope_dim=64):
    torch.manual_seed(31)
    x = torch.randn(n, rope_dim)
    ang = torch.randn(n, rope_dim // 2)          # arbitrary per-pos angles
    cos, sin = torch.cos(ang), torch.sin(ang)
    y_fwd = _rope_ref(x, cos, sin, False)
    y_inv = _rope_ref(x, cos, sin, True)
    save_file({
        "x": x.contiguous(), "cos": cos.contiguous(), "sin": sin.contiguous(),
        "y_fwd": y_fwd.contiguous(), "y_inv": y_inv.contiguous(),
        "dims": torch.tensor([n, rope_dim], dtype=torch.int32),
    }, os.path.join(out_dir, "unit_rope.safetensors"))
    print(f"[rope] n={n} rope_dim={rope_dim}  fwd|max={y_fwd.abs().max():.4f}")


def gen_rmsnorm(out_dir, n=32, dim=512, eps=1e-6):
    torch.manual_seed(41)
    x = torch.randn(n, dim)
    w = torch.randn(dim) * 0.1 + 1.0
    yw = (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps) * w)
    yn = (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps))   # no-weight (per-head q norm)
    save_file({
        "x": x.contiguous(), "weight": w.contiguous(),
        "y_w": yw.contiguous(), "y_now": yn.contiguous(),
        "dims": torch.tensor([n, dim], dtype=torch.int32),
        "eps": torch.tensor([eps], dtype=torch.float32),
    }, os.path.join(out_dir, "unit_rmsnorm.safetensors"))
    print(f"[rmsnorm] n={n} dim={dim}")


def gen_act_quant(out_dir, n=32, dim=512, block=64):
    """Fused FP8 QAT-sim (quant->dequant back to input dtype), ue8m0 pow2 scale (kernel.py act_quant
    inplace=True; used on KV NoPE dims model.py:512)."""
    torch.manual_seed(51)
    x = torch.randn(n, dim, dtype=torch.bfloat16)
    y = K.act_quant(x.clone(), block, "ue8m0", torch.float8_e8m0fnu, inplace=True)   # in-place -> bf16
    save_file({
        "x": x.contiguous().float(), "y_ref": y.contiguous().float(),
        "dims": torch.tensor([n, dim, block], dtype=torch.int32),
    }, os.path.join(out_dir, "unit_act_quant.safetensors"))
    print("[act_quant] n=%d dim=%d block=%d  maxdiff=%.4f" % (n, dim, block, (x.float()-y.float()).abs().max().item()))


def gen_act_quant_fp4(out_dir, n=32, dim=256, block=32):
    """Fused FP4 QAT-sim (kernel.py fp4_act_quant inplace=True) — indexer q + rotate-compressor."""
    torch.manual_seed(116)
    x = torch.randn(n, dim)
    y = K.fp4_act_quant(x.clone(), block, inplace=True)
    save_file({"x": x.contiguous(), "y_ref": y.contiguous(), "dims": torch.tensor([n, dim, block], dtype=torch.int32)},
              os.path.join(out_dir, "unit_act_quant_fp4.safetensors"))
    print("[act_quant_fp4] n=%d dim=%d block=%d  maxdiff=%.4f" % (n, dim, block, (x - y).abs().max().item()))


_E2M1_MAG = torch.tensor([0., 0.5, 1., 1.5, 2., 3., 4., 6.])   # nibble&7 -> magnitude


def weight_quant_fp4(W, block=32):
    """[N,K] bf16 -> packed fp4 [N,K/2] uint8 (2 nibbles/byte) + per-32 pow2 scale [N,K/32] f32.
    Nibble = sign(bit3) | nearest-E2M1-magnitude-index. Packing: element k even -> low nibble."""
    N, K = W.shape
    Wb = W.float().view(N, K // block, block)
    amax = Wb.abs().amax(-1, keepdim=True).clamp_min(6 * 2 ** -126)
    s = _round_pow2(amax / 6.0)
    scaled = torch.clamp((Wb / s).view(N, K), -6.0, 6.0)
    mags = _E2M1_MAG.to(scaled.device)
    idx = (scaled.abs().unsqueeze(-1) - mags).abs().argmin(-1).to(torch.uint8)     # [N,K]
    nib = idx | ((scaled < 0).to(torch.uint8) << 3)
    packed = (nib[:, 0::2] | (nib[:, 1::2] << 4)).contiguous()                      # [N,K/2]
    return packed, s.squeeze(-1).contiguous()


def gen_fp4_gemm(out_dir, M=32, N=128, K_=256):
    """FP8-act x FP4-weight GEMM (kernel.py fp4_gemm) — the MoE expert GEMM."""
    torch.manual_seed(71)
    A = torch.randn(M, K_)
    A_fp8, a_s = K.act_quant(A, 128, None, torch.float32)
    W = torch.randn(N, K_) * 0.1
    b_packed, b_s = weight_quant_fp4(W, 32)
    C = K.fp4_gemm(A_fp8, a_s, b_packed.view(torch.float4_e2m1fn_x2), b_s)
    save_file({
        "A_fp8": A_fp8.contiguous(), "a_s": a_s.contiguous().float(),
        "B_fp4": b_packed, "b_s": b_s.contiguous().float(), "C_ref": C.contiguous().float(),
        "dims": torch.tensor([M, N, K_], dtype=torch.int32),
    }, os.path.join(out_dir, "unit_fp4_gemm.safetensors"))
    print("[fp4_gemm] M=%d N=%d K=%d  |C|max=%.4f" % (M, N, K_, C.abs().max().item()))


def gen_router(out_dir, n=16, dim=256, n_routed=8, topk=2, route_scale=1.5):
    """noaux_tc score router (sqrtsoftplus, bias for selection only, renorm, *route_scale)."""
    torch.manual_seed(81)
    x = torch.randn(n, dim)
    gate_w = torch.randn(n_routed, dim) * 0.1
    bias = torch.randn(n_routed) * 0.1
    scores = (x.float() @ gate_w.float().t())
    scores = F.softplus(scores).sqrt()
    orig = scores
    sel = scores + bias
    idx = sel.topk(topk, dim=-1)[1]
    wts = orig.gather(1, idx)
    wts = wts / wts.sum(-1, keepdim=True) * route_scale
    save_file({
        "x": x.contiguous(), "gate_w": gate_w.contiguous(), "bias": bias.contiguous(),
        "weights_ref": wts.contiguous().float(), "indices_ref": idx.contiguous().int(),
        "dims": torch.tensor([n, dim, n_routed, topk], dtype=torch.int32),
        "route_scale": torch.tensor([route_scale], dtype=torch.float32),
    }, os.path.join(out_dir, "unit_router.safetensors"))
    print("[router] n=%d dim=%d n_routed=%d topk=%d" % (n, dim, n_routed, topk))


def gen_compressor(out_dir, bs=8, dim=256, d=128, ratio=4):
    """KV Compressor gated-pooling core (non-overlap prefill; model.py:329-348)."""
    torch.manual_seed(111)
    x = torch.randn(bs, dim); wkv = torch.randn(d, dim) * 0.1; wgate = torch.randn(d, dim) * 0.1
    ape = torch.randn(ratio, d) * 0.1
    kv = x @ wkv.t(); score = x @ wgate.t()
    kv_g = kv.reshape(bs // ratio, ratio, d); score_g = score.reshape(bs // ratio, ratio, d) + ape
    pooled = (kv_g * score_g.softmax(dim=1)).sum(1)                 # [groups, d]
    save_file({
        "x": x.contiguous(), "wkv": wkv.contiguous(), "wgate": wgate.contiguous(), "ape": ape.contiguous(),
        "pooled": pooled.contiguous(), "dims": torch.tensor([bs, dim, d, ratio], dtype=torch.int32),
    }, os.path.join(out_dir, "unit_compressor.safetensors"))
    print("[compressor] bs=%d dim=%d d=%d ratio=%d  |pooled|max=%.4f" % (bs, dim, d, ratio, pooled.abs().max().item()))


def _overlap_transform(t, value, groups, ratio, d):
    # t:[groups,ratio,2d] -> [groups,2*ratio,d]  (model.py Compressor.overlap_transform)
    new = torch.full((groups, 2 * ratio, d), value, dtype=t.dtype)
    new[:, ratio:] = t[:, :, d:]
    new[1:, :ratio] = t[:-1, :, :d]
    return new


def gen_compressor_overlap(out_dir, bs=8, d=64, ratio=4):
    torch.manual_seed(112)
    s = bs; groups = s // ratio; twod = 2 * d
    kv = torch.randn(s, twod); score = torch.randn(s, twod); ape = torch.randn(ratio, twod) * 0.1
    kv_g = kv.reshape(groups, ratio, twod); score_g = score.reshape(groups, ratio, twod) + ape
    kv_o = _overlap_transform(kv_g, 0.0, groups, ratio, d)
    score_o = _overlap_transform(score_g, float("-inf"), groups, ratio, d)
    pooled = (kv_o * score_o.softmax(dim=1)).sum(dim=1)             # [groups, d]
    save_file({"kv": kv.contiguous(), "score": score.contiguous(), "ape": ape.contiguous(),
               "pooled": pooled.contiguous(), "dims": torch.tensor([s, d, ratio], dtype=torch.int32)},
              os.path.join(out_dir, "unit_compressor_overlap.safetensors"))
    print("[compressor_overlap] s=%d d=%d ratio=%d  |pooled|max=%.4f" % (s, d, ratio, pooled.abs().max().item()))


def gen_compressor_full(out_dir, bs=8, dim=256, d=128, ratio=4, rope_dim=64, eps=1e-6, rotate=False):
    """Full Compressor fwd (overlap): gemm -> overlap-pool -> norm -> RoPE -> [rotate: hadamard(full)+fp4-sim
    | else fp8-sim NoPE]. rotate=True is the DSA indexer's compressor (computed in fp32 to match CUDA)."""
    torch.manual_seed(113 + int(rotate)); coff = 2; s = bs; groups = s // ratio; od = coff * d
    x = torch.randn(s, dim); wkv = torch.randn(od, dim) * 0.1; wgate = torch.randn(od, dim) * 0.1
    ape = torch.randn(ratio, od) * 0.1; norm_w = torch.randn(d) * 0.1 + 1.0
    ang = torch.randn(groups, rope_dim // 2); cos, sin = torch.cos(ang), torch.sin(ang)
    kv = x @ wkv.t(); score = x @ wgate.t()
    kv_g = kv.reshape(groups, ratio, od); score_g = score.reshape(groups, ratio, od) + ape
    kv_o = _overlap_transform(kv_g, 0.0, groups, ratio, d); score_o = _overlap_transform(score_g, float("-inf"), groups, ratio, d)
    pooled = (kv_o * score_o.softmax(dim=1)).sum(dim=1)             # [groups, d]
    pooled = pooled.float() * torch.rsqrt(pooled.float().square().mean(-1, keepdim=True) + eps) * norm_w
    out = pooled.clone()
    out[:, -rope_dim:] = _rope_ref(pooled[:, -rope_dim:], cos, sin, False)
    if rotate:
        from fast_hadamard_transform import hadamard_transform
        out = hadamard_transform(out, scale=d ** -0.5)             # full-d Hadamard
        out = K.fp4_act_quant(out, 32, inplace=True)               # fp4-sim full
    else:
        out[:, :-rope_dim] = K.act_quant(out[:, :-rope_dim].clone(), 64, "ue8m0", torch.float8_e8m0fnu, inplace=True)
    fn = "unit_compressor_full_rotate.safetensors" if rotate else "unit_compressor_full.safetensors"
    save_file({"x": x.contiguous(), "wkv": wkv.contiguous(), "wgate": wgate.contiguous(), "ape": ape.contiguous(),
               "norm_w": norm_w.contiguous(), "cos": cos.contiguous(), "sin": sin.contiguous(),
               "out": out.contiguous(), "dims": torch.tensor([s, dim, d, ratio, rope_dim, coff], dtype=torch.int32)},
              os.path.join(out_dir, fn))
    print("[compressor_full rotate=%d] s=%d dim=%d d=%d  |out|max=%.4f" % (int(rotate), s, dim, d, out.abs().max().item()))


def _compressor_ref(x, wkv, wgate, ape, norm_w, cos, sin, ratio, d, rope_dim, eps, rotate):
    """fp32 reference matching CUDA compressor_forward (overlap)."""
    s = x.shape[0]; groups = s // ratio; od = 2 * d
    kv = x @ wkv.t(); score = x @ wgate.t()
    kv_g = kv.reshape(groups, ratio, od); score_g = score.reshape(groups, ratio, od) + ape
    kv_o = _overlap_transform(kv_g, 0.0, groups, ratio, d); score_o = _overlap_transform(score_g, float("-inf"), groups, ratio, d)
    pooled = (kv_o * score_o.softmax(dim=1)).sum(dim=1)
    pooled = pooled.float() * torch.rsqrt(pooled.float().square().mean(-1, keepdim=True) + eps) * norm_w
    out = pooled.clone(); out[:, -rope_dim:] = _rope_ref(pooled[:, -rope_dim:], cos, sin, False)
    if rotate:
        from fast_hadamard_transform import hadamard_transform
        out = hadamard_transform(out, scale=d ** -0.5); out = K.fp4_act_quant(out, 32, inplace=True)
    else:
        out[:, :-rope_dim] = K.act_quant(out[:, :-rope_dim].clone(), 64, "ue8m0", torch.float8_e8m0fnu, inplace=True)
    return out


def gen_indexer(out_dir, s=16, dim=256, q_lora=128, n_heads=8, idx_hd=128, rd=64, ratio=4, index_topk=512, offset=100, eps=1e-6):
    """Full DSA Indexer forward (replicated via gated primitives). Gates CUDA indexer_forward's index_score."""
    from fast_hadamard_transform import hadamard_transform
    torch.manual_seed(117); T = s // ratio; QD = n_heads * idx_hd; od = 2 * idx_hd
    x = torch.randn(s, dim); qr = torch.randn(s, q_lora)
    Wqb = torch.randn(QD, q_lora) * 0.1; wqb_p, wqb_s = weight_quant_fp8(Wqb, 128)
    Wproj = torch.randn(n_heads, dim) * 0.1
    c_wkv = torch.randn(od, dim) * 0.1; c_wgate = torch.randn(od, dim) * 0.1
    c_ape = torch.randn(ratio, od) * 0.1; c_norm = torch.randn(idx_hd) * 0.1 + 1.0
    q_cos, q_sin = torch.cos(torch.randn(s, rd // 2)), torch.sin(torch.randn(s, rd // 2))
    c_cos, c_sin = torch.cos(torch.randn(T, rd // 2)), torch.sin(torch.randn(T, rd // 2))
    wscale = idx_hd ** -0.5 * n_heads ** -0.5
    # q = wq_b(qr) -> rope -> hadamard -> fp4
    qrq, qrs = K.act_quant(qr, 128, "ue8m0", torch.float8_e8m0fnu)
    q = K.fp8_gemm(qrq, qrs, wqb_p, wqb_s).reshape(s, n_heads, idx_hd)
    cos_e = q_cos.unsqueeze(1).expand(s, n_heads, rd // 2).reshape(s * n_heads, rd // 2)
    sin_e = q_sin.unsqueeze(1).expand(s, n_heads, rd // 2).reshape(s * n_heads, rd // 2)
    q[:, :, -rd:] = _rope_ref(q[:, :, -rd:].reshape(s * n_heads, rd), cos_e, sin_e, False).reshape(s, n_heads, rd)
    q = hadamard_transform(q, scale=idx_hd ** -0.5); q = K.fp4_act_quant(q, 32, inplace=True)
    ckv = _compressor_ref(x, c_wkv, c_wgate, c_ape, c_norm, c_cos, c_sin, ratio, idx_hd, rd, eps, True)   # [T, idx_hd]
    weights = (x @ Wproj.t()) * wscale
    isc = torch.einsum("shd,td->sht", q, ckv)
    isc = (F.relu(isc) * weights.unsqueeze(-1)).sum(dim=1)                          # [s,T]
    thr = (torch.arange(1, s + 1) // ratio).unsqueeze(1)
    isc = isc.masked_fill(torch.arange(T).unsqueeze(0) >= thr, -1e30)               # match CUDA -inf sentinel
    save_file({"x": x.contiguous(), "qr": qr.contiguous(), "wq_b": wqb_p.contiguous(), "wq_b_s": wqb_s.contiguous().float(),
               "weights_proj": Wproj.contiguous(), "c_wkv": c_wkv.contiguous(), "c_wgate": c_wgate.contiguous(),
               "c_ape": c_ape.contiguous(), "c_norm": c_norm.contiguous(),
               "q_cos": q_cos.contiguous(), "q_sin": q_sin.contiguous(), "c_cos": c_cos.contiguous(), "c_sin": c_sin.contiguous(),
               "index_score": isc.contiguous(),
               "dims": torch.tensor([s, dim, q_lora, n_heads, idx_hd, rd, ratio, index_topk, offset], dtype=torch.int32)},
              os.path.join(out_dir, "unit_indexer.safetensors"))
    print("[indexer] s=%d T=%d n_heads=%d idx_hd=%d  valid|score|max=%.4f" % (s, T, n_heads, idx_hd, isc[isc > -1e29].abs().max().item()))


def gen_yarn(out_dir, seqlen=64, dim=64):
    """RoPE/YaRN freq precompute (model.py precompute_freqs_cis): compressed (YaRN on) + sliding (off)."""
    import deepseek_v4_ref as MM
    def cs(orig, base):
        fc = MM.precompute_freqs_cis(dim, seqlen, orig, base, 16, 32, 1)
        r = torch.view_as_real(fc); return r[..., 0].contiguous(), r[..., 1].contiguous()
    cy, sy = cs(65536, 160000.0)          # compressed (YaRN)
    co, so = cs(0, 10000.0)               # pure-sliding (no YaRN)
    save_file({"cos_yarn": cy, "sin_yarn": sy, "cos_off": co, "sin_off": so,
               "dims": torch.tensor([seqlen, dim], dtype=torch.int32)},
              os.path.join(out_dir, "unit_yarn.safetensors"))
    print("[yarn] seqlen=%d dim=%d" % (seqlen, dim))


def gen_hadamard(out_dir, rows=8, D=128):
    from fast_hadamard_transform import hadamard_transform
    torch.manual_seed(114)
    x = torch.randn(rows, D)
    y = hadamard_transform(x, scale=D ** -0.5)
    save_file({"x": x.contiguous(), "y": y.contiguous(), "dims": torch.tensor([rows, D], dtype=torch.int32)},
              os.path.join(out_dir, "unit_hadamard.safetensors"))
    print("[hadamard] rows=%d D=%d  |y|max=%.4f" % (rows, D, y.abs().max().item()))


def gen_index_score(out_dir, S=8, H=8, d=128, T=6):
    torch.manual_seed(115)
    q = torch.randn(S, H, d); kv = torch.randn(T, d); weights = torch.randn(S, H) * 0.1
    score = torch.einsum("shd,td->sht", q, kv)
    score = (F.relu(score) * weights.unsqueeze(-1)).sum(dim=1)      # [S,T]
    save_file({"q": q.contiguous(), "kv": kv.contiguous(), "weights": weights.contiguous(),
               "score": score.contiguous(), "dims": torch.tensor([S, H, d, T], dtype=torch.int32)},
              os.path.join(out_dir, "unit_index_score.safetensors"))
    print("[index_score] S=%d H=%d d=%d T=%d  |score|max=%.4f" % (S, H, d, T, score.abs().max().item()))


def gen_hc(out_dir, bs=8, hc=4, d=64, eps=1e-6, iters=20):
    """Hyper-Connections pre + post (model.py:680-693)."""
    torch.manual_seed(101)
    x = torch.randn(bs, hc, d); hc_fn = torch.randn((2 + hc) * hc, hc * d) * 0.1
    scale = torch.randn(3) * 0.5; base = torch.randn((2 + hc) * hc) * 0.5
    xf = x.reshape(bs, hc * d)
    rsqrt = torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
    mixes = (xf @ hc_fn.t()) * rsqrt
    pre, post, comb = K.hc_split_sinkhorn(mixes, scale, base, hc, iters, eps)
    y_pre = (pre.unsqueeze(-1) * x).sum(1)                          # [bs,d]
    x_new = torch.randn(bs, d)
    # model.py hc_post sums comb's FIRST index: y[j,e] = post_j*x_new_e + Σ_i comb[i,j]*x[i,e]
    y_post = post.unsqueeze(-1) * x_new.unsqueeze(1) + torch.einsum("bij,bie->bje", comb, x)   # [bs,hc,d]
    save_file({
        "x": x.contiguous(), "hc_fn": hc_fn.contiguous(), "hc_scale": scale.contiguous(), "hc_base": base.contiguous(),
        "x_new": x_new.contiguous(), "y_pre": y_pre.contiguous(), "post": post.contiguous(),
        "comb": comb.contiguous(), "y_post": y_post.contiguous(),
        "dims": torch.tensor([bs, hc, d, iters], dtype=torch.int32),
    }, os.path.join(out_dir, "unit_hc.safetensors"))
    print("[hc] bs=%d hc=%d d=%d  |y_pre|max=%.4f |y_post|max=%.4f" % (bs, hc, d, y_pre.abs().max().item(), y_post.abs().max().item()))


def _fp4x2(t): return t.view(torch.float4_e2m1fn_x2)


def gen_moe(out_dir, n=8, dim=256, inter=128, nr=8, na=2, rs=1.5, lim=10.0):
    """Synthetic score-routed MoE (fp4 experts + fp8 shared) computed via the gated primitives, matching
    model.py Expert/MoE semantics (routing weight applied to intermediate BEFORE w2's act-quant)."""
    torch.manual_seed(91)
    x = torch.randn(n, dim); gate_w = torch.randn(nr, dim) * 0.1; bias = torch.randn(nr) * 0.1
    W1 = [torch.randn(inter, dim) * 0.1 for _ in range(nr)]
    W2 = [torch.randn(dim, inter) * 0.1 for _ in range(nr)]
    W3 = [torch.randn(inter, dim) * 0.1 for _ in range(nr)]
    q1 = [weight_quant_fp4(W1[e]) for e in range(nr)]; q2 = [weight_quant_fp4(W2[e]) for e in range(nr)]
    q3 = [weight_quant_fp4(W3[e]) for e in range(nr)]
    S1, S2, S3 = torch.randn(inter, dim)*0.1, torch.randn(dim, inter)*0.1, torch.randn(inter, dim)*0.1
    from gen_units import weight_quant_fp8   # local helper
    s1p, s1s = weight_quant_fp8(S1); s2p, s2s = weight_quant_fp8(S2); s3p, s3s = weight_quant_fp8(S3)

    scores = F.softplus(x.float() @ gate_w.float().t()).sqrt()
    idx = (scores + bias).topk(na, dim=-1)[1]
    wts = scores.gather(1, idx); wts = wts / wts.sum(-1, keepdim=True) * rs

    def clampswi(g, u):
        g = torch.clamp(g, max=lim); u = torch.clamp(u, -lim, lim); return F.silu(g) * u
    def exp_fp4(xr, e, weight):
        xq, xs = K.act_quant(xr.unsqueeze(0), 128, "ue8m0", torch.float32)
        g = K.fp4_gemm(xq, xs, _fp4x2(q1[e][0]), q1[e][1]); u = K.fp4_gemm(xq, xs, _fp4x2(q3[e][0]), q3[e][1])
        h = weight * clampswi(g, u)
        hq, hs = K.act_quant(h, 128, "ue8m0", torch.float32)
        return K.fp4_gemm(hq, hs, _fp4x2(q2[e][0]), q2[e][1])[0]
    def exp_shared(xr):
        xq, xs = K.act_quant(xr.unsqueeze(0), 128, "ue8m0", torch.float32)
        g = K.fp8_gemm(xq, xs, s1p, s1s); u = K.fp8_gemm(xq, xs, s3p, s3s)
        h = clampswi(g, u); hq, hs = K.act_quant(h, 128, "ue8m0", torch.float32)
        return K.fp8_gemm(hq, hs, s2p, s2s)[0]
    y = torch.zeros(n, dim)
    for t in range(n):
        for s in range(na): y[t] += exp_fp4(x[t], int(idx[t, s]), float(wts[t, s]))
        y[t] += exp_shared(x[t])

    d = {"x": x.contiguous(), "gate_w": gate_w.contiguous(), "bias": bias.contiguous(),
         "y_ref": y.contiguous().float(),
         "w1": torch.stack([q1[e][0] for e in range(nr)]), "w1s": torch.stack([q1[e][1] for e in range(nr)]).float(),
         "w2": torch.stack([q2[e][0] for e in range(nr)]), "w2s": torch.stack([q2[e][1] for e in range(nr)]).float(),
         "w3": torch.stack([q3[e][0] for e in range(nr)]), "w3s": torch.stack([q3[e][1] for e in range(nr)]).float(),
         "sw1": s1p, "sw1s": s1s.float(), "sw2": s2p, "sw2s": s2s.float(), "sw3": s3p, "sw3s": s3s.float(),
         "dims": torch.tensor([n, dim, inter, nr, na], dtype=torch.int32),
         "rs": torch.tensor([rs], dtype=torch.float32), "lim": torch.tensor([lim], dtype=torch.float32)}
    save_file({k: v.contiguous() for k, v in d.items()}, os.path.join(out_dir, "unit_moe.safetensors"))
    print("[moe] n=%d dim=%d inter=%d nr=%d na=%d  |y|max=%.4f" % (n, dim, inter, nr, na, y.abs().max().item()))


def gen_ogroup_gemm(out_dir, bs=8, G=2, R=16, Kd=128):
    """Grouped o-LoRA einsum: out[bs,G,R] = sum_d o[bs,G,d]*wo_a[G,R,d]  (model.py:543-546, bf16)."""
    torch.manual_seed(61)
    o = torch.randn(bs, G, Kd, dtype=torch.bfloat16)
    wo_a = (torch.randn(G, R, Kd, dtype=torch.bfloat16) * 0.05)
    out = torch.einsum("bgd,grd->bgr", o.float(), wo_a.float())
    save_file({
        "o": o.contiguous().float(), "wo_a": wo_a.contiguous().float(),
        "out_ref": out.contiguous().float(),
        "dims": torch.tensor([bs, G, R, Kd], dtype=torch.int32),
    }, os.path.join(out_dir, "unit_ogroup_gemm.safetensors"))
    print(f"[ogroup_gemm] bs={bs} G={G} R={R} Kd={Kd}  |out|max={out.abs().max():.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="goldens")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    gen_fp8_gemm(a.out)
    gen_hc_sinkhorn(a.out)
    gen_sparse_attn(a.out)
    gen_rope(a.out)
    gen_rmsnorm(a.out)
    gen_act_quant(a.out)
    gen_ogroup_gemm(a.out)
    gen_fp4_gemm(a.out)
    gen_router(a.out)
    gen_moe(a.out)
    gen_hc(a.out)
    gen_compressor(a.out)
    gen_compressor_overlap(a.out)
    gen_compressor_full(a.out)
    gen_compressor_full(a.out, rotate=True)
    gen_hadamard(a.out)
    gen_index_score(a.out)
    gen_act_quant_fp4(a.out)
    gen_indexer(a.out)
    gen_yarn(a.out)
    print("units written to", a.out)
