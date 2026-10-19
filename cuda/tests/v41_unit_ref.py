#!/usr/bin/env python3
"""Reference vectors for tests/gate_v41_units.cu, computed by the ops oracles (ops/lib/ggml_quants.py,
ops/lib/torch_kernels.py) and the reference Gate / FQExpert formulas. CPU only, no checkpoint.

  python -I tests/v41_unit_ref.py --out build/units      (then: build/gate_v41_units build/units)

Writes .npy files (little-endian, C order). Cases:
  dec_<qt>_{w,ref}   expert decoders: thor-v1 blocks -> fp32, must match ggml_quants.dequant BIT-EXACTLY
  aq_{x,q,s}         act_quant (block 32, ue8m0): e4m3 bytes + scales, exact
  lin_*              fp8_linear vs act_quant + fp8_gemm -> bf16 (<= 1 bf16 ulp, summation order)
  blin_*             bf16_linear vs x @ W^T -> bf16
  rt_*               router (sqrtsoftplus, bias selects, bias_vl on image rows, x1.5)
  moe_*              moe_routed with Q3_K gate/up and Q4_K down over 8 experts, vs FQExpert-style math
  dec_KT{2,3}_*      trellis decoders: random code words (every bit pattern is a valid stream) + fp16
                     scales -> rotated-domain fp32, must match ops/lib/kt.dequant BIT-EXACTLY
  rht_*              KT activation rotation, both pass widths: must match kt.rht_rows BIT-EXACTLY
  vit_*              the CUDA Viterbi encoder: must return kt.viterbi's states exactly
  moekt_*            moe_routed with KT3 gate/up and KT2 down (rotated passes), cos >= 0.99999
"""
import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from ops.lib import ggml_quants as G  # noqa: E402
from ops.lib import kt  # noqa: E402
from ops.lib import torch_kernels as tk  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="build/units")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
rng = np.random.default_rng(1017)
torch.manual_seed(1017)


def save(name, arr):
    np.save(os.path.join(a.out, name + ".npy"), np.ascontiguousarray(arr))


def bf16(t):
    return t.to(torch.bfloat16).float()


# ---------------------------------------------------------------- decoders
R, C = 64, 2304                                     # 9 superblocks: the w2 input width
for qt in ("Q2_K", "Q3_K", "Q4_K"):
    W = (rng.standard_normal((R, C)) * rng.uniform(0.002, 0.05, (R, 1))).astype(np.float32)
    blocks = G.quantize_rtn(qt, W)
    save(f"dec_{qt}_w", blocks.astype(np.uint8))
    save(f"dec_{qt}_ref", G.dequant(qt, blocks).astype(np.float32))
wb = rng.integers(0, 256, (R, C // 2), dtype=np.uint8)
sb = rng.integers(115, 135, (R, C // 32), dtype=np.uint8)
save("dec_MXFP4_w", wb); save("dec_MXFP4_s", sb)
save("dec_MXFP4_ref", G.src_dequant(wb, sb).astype(np.float32))

# ---------------------------------------------------------------- act_quant
x = torch.randn(4, 5120) * torch.logspace(-3, 1, 4).unsqueeze(1)
x[1, :32] = 0.0                                      # amax clamp 1e-4
x = bf16(x)
q, s = tk.act_quant(x.clone(), 32, "ue8m0")
save("aq_x", x.numpy()); save("aq_q", q.view(torch.uint8).numpy()); save("aq_s", s.float().numpy())

# ---------------------------------------------------------------- fp8 linear
M, N, K = 3, 256, 1024
Wf = torch.randn(N, K) * 0.05
Ws_e = torch.randint(120, 128, (N // 32, K // 32), dtype=torch.uint8)
Ws = torch.ldexp(torch.ones(N // 32, K // 32), Ws_e.int() - 127)
Wq = (Wf / Ws.repeat_interleave(32, 0).repeat_interleave(32, 1)).clamp(-448, 448).to(torch.float8_e4m3fn)
x = bf16(torch.randn(M, K))
xq, xs = tk.act_quant(x.clone(), 32, "ue8m0")
ref = tk._deq_act(xq, xs, K) @ tk.dequant_fp8(Wq, Ws).t()
save("lin_x", x.numpy()); save("lin_w", Wq.view(torch.uint8).numpy()); save("lin_ws", Ws_e.numpy())
save("lin_ref", bf16(ref).numpy())

# ---------------------------------------------------------------- bf16 linear
Wb = torch.randn(512, 5120).to(torch.bfloat16)
x = bf16(torch.randn(2, 5120))
save("blin_x", x.numpy()); save("blin_w", Wb.view(torch.int16).numpy().view(np.uint16))
save("blin_ref", bf16(x @ Wb.float().t()).numpy())

# ---------------------------------------------------------------- router
E, M = 192, 5
x = bf16(torch.randn(M, 5120))
Wg = (torch.randn(E, 5120) * 0.02).to(torch.bfloat16)
bias = torch.randn(E) * 0.1
bias_vl = torch.randn(E) * 0.1
img = torch.tensor([0, 1, 0, 1, 0], dtype=torch.uint8)
sc = torch.nn.functional.softplus(x @ Wg.float().t()).sqrt()
b = torch.where(img.bool().unsqueeze(1), bias_vl.unsqueeze(0), bias.unsqueeze(0))
idx = (sc + b).topk(6, dim=-1)[1]
w = sc.gather(1, idx)
w = w / (w.sum(-1, keepdim=True) + 1e-20) * 1.5
save("rt_x", x.numpy()); save("rt_w", Wg.view(torch.int16).numpy().view(np.uint16))
save("rt_bias", bias.numpy()); save("rt_bias_vl", bias_vl.numpy()); save("rt_img", img.numpy())
save("rt_idx", idx.int().numpy()); save("rt_wt", w.numpy())

# ---------------------------------------------------------------- moe_routed (Q3_K w1/w3, Q4_K w2)
E, M, TOPK, D, I = 8, 3, 6, 5120, 2304
Wd = {}
for p, (o, i, qt) in {"w1": (I, D, "Q3_K"), "w3": (I, D, "Q3_K"), "w2": (D, I, "Q4_K")}.items():
    blk = np.stack([G.quantize_rtn(qt, (rng.standard_normal((o, i)) * 0.02).astype(np.float32)) for _ in range(E)])
    save(f"moe_{p}", blk)
    Wd[p] = torch.from_numpy(G.dequant(qt, blk).reshape(E, o, i))
x = bf16(torch.randn(M, D))
idx = torch.stack([torch.randperm(E)[:TOPK] for _ in range(M)]).int()
wts = torch.rand(M, TOPK) * 0.5
xq = tk.act_quant(x.clone(), 32, "ue8m0", inplace=True).float()
y = torch.zeros(M, D)
for m in range(M):
    for k in sorted(range(TOPK), key=lambda k: int(idx[m, k])):          # ascending expert order
        e = int(idx[m, k])
        g = bf16(xq[m] @ Wd["w1"][e].t()); u = bf16(xq[m] @ Wd["w3"][e].t())
        u = u.clamp(-10, 10); g = g.clamp(max=10)
        mid = bf16(wts[m, k] * (torch.nn.functional.silu(g) * u))
        midq = tk.act_quant(mid.clone()[None], 32, "ue8m0", inplace=True)[0].float()
        y[m] += bf16(midq @ Wd["w2"][e].t())
save("moe_x", x.numpy()); save("moe_idx", idx.numpy()); save("moe_wts", wts.numpy()); save("moe_ref", y.numpy())
# ---------------------------------------------------------------- KT (trellis) decoders
for qt in ("KT2", "KT3"):
    b = kt.bits_of(qt)
    words = rng.integers(0, 2 ** 32, (R, C // kt.T, 4 * b), dtype=np.uint64).astype(np.uint32)
    sc = (rng.uniform(0.002, 0.05, (R, C // kt.T)) / kt.CB_RMS).astype(np.float16)
    qw = words.view(np.uint8).reshape(R, -1)
    save(f"dec_{qt}_w", qw); save(f"dec_{qt}_s", sc.view(np.uint16))
    save(f"dec_{qt}_ref", kt.dequant(qt, qw, sc, C, None))

# ---------------------------------------------------------------- KT activation rotation
for K in (5120, 2304):
    xr = (rng.standard_normal((3, K)) * rng.uniform(0.1, 3.0, (3, 1))).astype(np.float32)
    sg = kt.signs(f"unit{K}", K)
    save(f"rht{K}_x", xr); save(f"rht{K}_sign", kt.pack_signs(sg)); save(f"rht{K}_ref", kt.rht_rows(xr, sg))

# ---------------------------------------------------------------- KT Viterbi encoder
NV = 192
vx = (rng.standard_normal((NV, kt.T)) * kt.CB_RMS).astype(np.float32)
vx[:8] *= 4.0                                        # far outside the codebook: saturating paths
vw = rng.uniform(0.2, 3.0, (NV, kt.T)).astype(np.float32)
vw[8:16] = 1.0
save("vit_x", vx); save("vit_w", vw)
for b in (2, 3):
    save(f"vit_b{b}_states", kt.viterbi(vx, vw, b).astype(np.int32))

# ---------------------------------------------------------------- moe_routed, KT3 w1/w3 + KT2 w2
E, M, TOPK = 6, 2, 3
s13, s2 = kt.signs("unit.w13", D), kt.signs("unit.w2", I)
save("moekt_s13", kt.pack_signs(s13)); save("moekt_s2", kt.pack_signs(s2))
Wk = {}
for p, (o, i, qt) in {"w1": (I, D, "KT3"), "w3": (I, D, "KT3"), "w2": (D, I, "KT2")}.items():
    b = kt.bits_of(qt)
    words = rng.integers(0, 2 ** 32, (E, o, i // kt.T, 4 * b), dtype=np.uint64).astype(np.uint32)
    sc = (np.full((E, o, i // kt.T), 0.02) / kt.CB_RMS).astype(np.float16)
    qw = words.view(np.uint8).reshape(E, o, -1)
    save(f"moekt_{p}", qw); save(f"moekt_{p}_s", sc.view(np.uint16))
    sg = s13 if p != "w2" else s2
    Wk[p] = torch.from_numpy(np.stack([kt.dequant(qt, qw[e], sc[e], i, sg) for e in range(E)]))
x = bf16(torch.randn(M, D))
idx = torch.stack([torch.randperm(E)[:TOPK] for _ in range(M)]).int()
wts = torch.rand(M, TOPK) * 0.5
xq = tk.act_quant(x.clone(), 32, "ue8m0", inplace=True).float()
y = torch.zeros(M, D)
for m in range(M):
    for k in sorted(range(TOPK), key=lambda k: int(idx[m, k])):
        e = int(idx[m, k])
        g = bf16(xq[m] @ Wk["w1"][e].t()); u = bf16(xq[m] @ Wk["w3"][e].t())
        u = u.clamp(-10, 10); g = g.clamp(max=10)
        mid = bf16(wts[m, k] * (torch.nn.functional.silu(g) * u))
        midq = tk.act_quant(mid.clone()[None], 32, "ue8m0", inplace=True)[0].float()
        y[m] += bf16(midq @ Wk["w2"][e].t())
save("moekt_x", x.numpy()); save("moekt_idx", idx.numpy()); save("moekt_wts", wts.numpy()); save("moekt_ref", y.numpy())
print(f"unit reference vectors -> {a.out}")
