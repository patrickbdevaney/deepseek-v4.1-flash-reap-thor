"""The 1017 checkpoint's own speculative head for s11: the three `mtp.*` blocks (DSpark), on the
reference module.

V4.1 has no separate classical MTP head: `mtp.0..2` IS the DSpark drafter (3 blocks, own 128-expert
top-3 MoE, taps 37-39, PLAN.md). The two s11 residency arms are the same blocks served two ways:
  inmodel   the blocks' routed experts REAP-pruned with the backbone's criterion and K-quantized like
            the backbone's experts, resident inside the Thor build (preferred: no extra model);
  separate  the blocks unpruned at source precision (MXFP4 experts), a separate drafter file.

What the reference fixes (inference/model.py, `Transformer.forward` / `forward_spec`):
  * `main_hidden` = concat over the tap layers (37, 38, 39) of the hc-mean of the stream ENTERING
    each tap layer, [b, s, 3 * dim];
  * a draft step at `start_pos = p` takes the hidden of position p and the committed token at p+1
    (block slot 0) and proposes the 5 tokens at p+2 .. p+6 (slots 1..5, Markov-chained);
  * each DSparkAttention attends over a 128-slot window of `main_kv` (one per earlier position,
    seeded by a start_pos=0 call) plus the 5 block slots. RoPE scores depend on relative position
    only, so an anchor at any p is replayed with its window shifted to 0..W-1.

`step(...)` with targets=None is the reference forward (greedy Markov chain): acceptance. With
targets it is teacher-forced (the Markov head reads the target tokens): training. With targets equal
to the greedy drafts the two are the same function (ops/tests/test_s11.py checks it).
"""
from __future__ import annotations

import re

import numpy as np

TAPS = (37, 38, 39)
FP8_MAX = 448.0


class Drafter:
    def __init__(self, M, args, blocks, embed, head):
        self.M, self.args, self.blocks, self.embed, self.head = M, args, blocks, embed, head
        self.S = args.dspark_block_size

    def parameters(self):
        for b in self.blocks:
            yield from b.parameters()


def mp1(sd: dict) -> dict:
    """convert.py's mp=1 transforms on one block's raw tensors (as ops/lib/shards.ShardIndex.state_dict):
    `attn.wo_a` dequantized to bf16, routed-expert int8 viewed as float4_e2m1fn_x2."""
    import torch
    sd = dict(sd)
    for k in [k for k in sd if k.endswith("wo_a.weight") and (k[:-6] + "scale") in sd]:
        w, s = sd[k], sd.pop(k[:-6] + "scale")
        bo, bi = w.size(0) // s.size(0), w.size(1) // s.size(1)
        w = w.unflatten(0, (-1, bo)).unflatten(-1, (-1, bi)).float() * s[:, None, :, None].float()
        sd[k] = w.flatten(2, 3).flatten(0, 1).bfloat16()
    for k in list(sd):
        if re.search(r"(^|\.)experts\.\d+\.w[123]\.weight$", k) and sd[k].dtype == torch.int8:
            sd[k] = sd[k].view(torch.float4_e2m1fn_x2)
    return sd


def build(M, args, state_dicts: list[dict], embed_w, head_w, device: str) -> Drafter:
    """state_dicts[i]: mtp.i.* tensors relative to the block, after mp1(). Strict loading."""
    import torch
    blocks = []
    with torch.device(device):
        for i, sd in enumerate(state_dicts):
            b = M.DSparkBlock(args.n_layers + i, args)
            missing, unexpected = b.load_state_dict(sd, strict=False)
            if missing or unexpected:
                raise RuntimeError(f"mtp.{i}: missing {missing[:8]} unexpected {unexpected[:8]}")
            for m in b.modules():
                if hasattr(m, "weight") and getattr(m, "scale", None) is not None:
                    m.weight.scale = m.scale
            blocks.append(b.eval())
        embed = M.ParallelEmbedding(args.vocab_size, args.dim)
        head = M.ParallelHead(args.vocab_size, args.dim, args.norm_eps, args.hc_eps)
    with torch.no_grad():
        embed.weight.copy_(embed_w)
        head.weight.copy_(head_w.float())
    embed.weight.requires_grad_(False)
    head.weight.requires_grad_(False)
    for b in blocks:
        b.embed, b.head = embed, head
    return Drafter(M, args, blocks, embed, head)


def reset(d: Drafter) -> None:
    """Fresh window caches (also drops the autograd history a training step left in them)."""
    import torch
    for b in d.blocks:
        b.attn.window_kv_cache = torch.zeros_like(b.attn.window_kv_cache.detach())


def step(d: Drafter, main_hidden, tok_in, targets=None):
    """main_hidden [b, W, n_taps*dim] (positions p-W+1 .. p, W >= 2), tok_in [b] (the token at p+1).
    -> (logits [b, S, V] fp32, drafts [b, S], confidence [b, S])."""
    M, blocks = d.M, d.blocks
    W = main_hidden.size(1)
    if W < 2:
        raise ValueError("a draft step needs an earlier position (the reference asserts start_pos > 0)")
    reset(d)
    x, mx = blocks[0].forward_embed(main_hidden[:, :W - 1], tok_in)
    pm = M.make_identity_pre_mix(x, d.args.hc_mult)
    for b in blocks:                                  # start_pos 0 seeds the window caches only
        b(x, 0, pm, mx)
    x, mx = blocks[0].forward_embed(main_hidden[:, W - 1:], tok_in)
    pm = M.make_identity_pre_mix(x, d.args.hc_mult)
    for b in blocks:
        x, pm = b(x, W - 1, pm, mx)
    last = blocks[-1]
    if targets is None:
        out, logits, conf = last.forward_head(x, pm, tok_in)
        return logits, out[:, 1:], conf
    import torch
    h = last.hc_pre(x, pm)
    logits = last.head(last.norm(h), full_logits=True)
    prev = torch.cat([tok_in[:, None], targets[:, :-1]], 1)
    bias, emb = last.markov_head(prev)
    logits = logits + bias
    conf = last.confidence_head(h, emb)
    return logits, logits.argmax(-1), conf


def accept_len(drafts: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Leading matches per row: how many drafts a greedy verify accepts (0..S)."""
    ok = np.asarray(drafts) == np.asarray(targets)
    return np.cumprod(ok, axis=1).sum(axis=1)


def trace_anchors(n: int, gen_start: int, W: int, S: int) -> np.ndarray:
    """Anchors p of one trace whose committed token p+1 and all S targets p+2..p+S+1 are GENERATED
    (the served model's own greedy tokens), with a full W-window behind p."""
    lo, hi = max(W - 1, gen_start - 1), n - S - 2
    return np.arange(lo, hi + 1, dtype=np.int64) if hi >= lo else np.zeros(0, np.int64)


# ----------------------------------------------------------------------------- saliency (REAP on the drafter)
class MoEBank:
    """Per drafter block: the s03 bank fields s04's criteria read (cnt, gat, sum = g*||f||, nrm),
    the co-activation sum F_sum (HOPE), and the per-column input energy each expert saw (imatrix:
    A13 over dim for w1/w3, A2 over inter for w2)."""

    def __init__(self, E: int, dim: int, inter: int):
        z = lambda *s: np.zeros(s, np.float64)
        self.cnt, self.gat, self.sum, self.nrm = z(E), z(E), z(E), z(E)
        self.F_sum, self.F_cnt = z(E, E), z(E, E)
        self.A13, self.A2 = z(E, dim), z(E, inter)

    def save(self, path) -> None:
        np.savez(path, cnt=self.cnt, gat=self.gat, sum=self.sum, nrm=self.nrm, F_sum=self.F_sum, F_cnt=self.F_cnt,
                 A13=self.A13, A2=self.A2)


def instrument(M, d: Drafter) -> list[MoEBank]:
    """Replace each drafter block's MoE.forward with an observing copy (same output)."""
    import torch
    import torch.nn.functional as F
    banks = []
    for b in d.blocks:
        moe = b.ffn
        ex = next(e for e in moe.experts if e is not None)
        bank = MoEBank(moe.n_routed_experts, moe.dim, ex.w2.in_features)
        banks.append(bank)

        def fwd(x, image_mask=None, moe=moe, bank=bank):
            shape = x.size()
            x = x.view(-1, moe.dim)
            weights, indices = moe.gate(x, None if image_mask is None else image_mask.flatten())
            y = torch.zeros_like(x, dtype=torch.float32)
            sal = torch.zeros(x.size(0), moe.n_routed_experts, dtype=torch.float64, device=x.device)
            for i in range(moe.n_routed_experts):
                idx, top = torch.where(indices == i)
                if idx.numel() == 0:
                    continue
                e = moe.experts[i]
                xi = x[idx]
                gate = e.w1(xi).float()
                up = e.w3(xi).float()
                if e.swiglu_limit > 0:
                    up = torch.clamp(up, min=-e.swiglu_limit, max=e.swiglu_limit)
                    gate = torch.clamp(gate, max=e.swiglu_limit)
                act = F.silu(gate) * up
                w = weights[idx, top, None]
                out = e.w2(act.to(xi.dtype)).float()                 # unweighted expert output f_j(x)
                y[idx] += e.w2((w * act).to(xi.dtype)).float()       # exactly Expert.forward(x, w)
                nrm = out.norm(dim=-1).double()
                g = w[:, 0].double()
                bank.cnt[i] += idx.numel()
                bank.gat[i] += float(g.sum())
                bank.sum[i] += float((g * nrm).sum())
                bank.nrm[i] += float(nrm.sum())
                bank.A13[i] += xi.float().square().sum(0).double().cpu().numpy()
                bank.A2[i] += act.square().sum(0).double().cpu().numpy()
                sal[idx, i] = g * nrm
            bank.F_sum += (sal.T @ sal).cpu().numpy()
            hit = (sal > 0).double()
            bank.F_cnt += (hit.T @ hit).cpu().numpy()
            y += moe.shared_experts(x)
            return y.type_as(x).view(shape)

        moe.forward = fwd
    return banks


def select_keep(banks: list[dict], criterion: str, n_keep: int) -> dict[str, list[int]]:
    """The backbone's s04 criterion applied to each drafter block (uniform n_keep, no protection:
    the drift guard has no drafter statistics). Keys "0".."2" are the mtp stages."""
    from . import select
    keep = {}
    for i, z in enumerate(banks):
        E = z["cnt"].size
        if criterion.startswith("hope"):
            F = z["F_sum"] if criterion in ("hope", "hope_sum") else z["F_sum"] / np.maximum(z["F_cnt"], 1.0)
            pr = select.prune_set("hope", n_keep=n_keep, protect=np.zeros(0, int), F=F)
        else:
            crit = {"reap_sum": z["sum"], "mass": z["gat"], "freq": z["cnt"]}
            pr = select.prune_set(criterion if criterion in crit else "reap_sum", n_keep=n_keep,
                                  protect=np.zeros(0, int), crit=crit)
        keep[str(i)] = select.keep_from_pruned(pr, E)
    return keep


def apply_keep(M, d: Drafter, keep: dict, experts_bf16: dict | None = None) -> None:
    """Prune each block's router to the keep-set (proper re-routing: top-3 over the survivors,
    renormalised by the unmodified Gate) and renumber the experts 0..n-1. experts_bf16[i][j] =
    (w1, w2, w3) tensors replace the kept experts' weights (the quantized arm, dequantized)."""
    import torch
    for i, b in enumerate(d.blocks):
        ids = keep[str(i)]
        moe, g = b.ffn, b.ffn.gate
        sel = torch.tensor(ids, device=g.weight.device)
        with torch.no_grad():
            g.weight = torch.nn.Parameter(g.weight.data[sel].clone(), requires_grad=g.weight.requires_grad)
            g.bias = torch.nn.Parameter(g.bias.data[sel].clone(), requires_grad=False)
            if g.bias_vl is not None:
                g.bias_vl = torch.nn.Parameter(g.bias_vl.data[sel].clone(), requires_grad=False)
        kept = [moe.experts[j] for j in ids]
        if experts_bf16 is not None:
            for n, e in enumerate(kept):
                for proj, w in zip(("w1", "w2", "w3"), experts_bf16[i][n]):
                    lin = getattr(e, proj)
                    lin.weight = torch.nn.Parameter(w.to(torch.bfloat16).to(g.weight.device), requires_grad=False)
                    lin.register_parameter("scale", None)
        moe.experts = torch.nn.ModuleList(kept)
        moe.n_routed_experts = moe.n_local_experts = moe.experts_end_idx = len(ids)
        moe.experts_start_idx = 0


# ----------------------------------------------------------------------------- training
def ste_act_quant(tk):
    """Activation fake-quant as a straight-through estimator: the forward value is the kernel's (the
    in-place variant rewrites values through .data, invisible to autograd), the gradient identity."""
    import torch

    def act_quant(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
        with torch.no_grad():
            q = tk.act_quant(x.detach().clone(), block_size, scale_fmt, scale_dtype, True)
        if inplace:
            x.data.copy_(q)
            return x
        return tk.act_quant(x, block_size, scale_fmt, scale_dtype, False)   # fp8/fp4 Linears (frozen paths)
    return act_quant


def make_trainable(d: Drafter, tk) -> dict:
    """Dequantize every quantized Linear to a bf16 Parameter; routed experts and the shared
    embed/head FROZEN, everything else trains. -> {source name: numel} of what trains."""
    import torch
    M = d.M
    M.act_quant = ste_act_quant(tk)
    trained = {}
    for i, b in enumerate(d.blocks):
        for _, mod in b.named_modules():
            if not isinstance(mod, M.Linear):
                continue
            w = mod.weight
            if w.dtype == torch.float4_e2m1fn_x2:
                v = tk.dequant_fp4(w.data, mod.scale.data, 32)
            elif w.dtype == torch.float8_e4m3fn:
                v = tk.dequant_fp8(w.data, mod.scale.data)
            else:
                continue
            mod.weight = torch.nn.Parameter(v.to(torch.bfloat16))
            mod.register_parameter("scale", None)
        for name, p in b.named_parameters():
            routed = re.search(r"(^|\.)ffn\.experts\.\d+\.", name) is not None
            frozen = routed or name.startswith(("embed.", "head.")) or name.endswith((".bias", "bias_vl")) and ".gate." in name
            p.requires_grad_(not frozen)
            if not frozen:
                trained[f"mtp.{i}.{name}"] = p.numel()
    d.embed.weight.requires_grad_(False)
    d.head.weight.requires_grad_(False)
    return trained


def loss_fn(logits, conf, targets):
    """Cross-entropy of every block slot against the served model's greedy token, plus a BCE that
    teaches the confidence head whether the slot survives a greedy verify (prefix-correct)."""
    import torch
    import torch.nn.functional as F
    S = targets.size(1)
    ce = F.cross_entropy(logits.float().flatten(0, 1), targets.flatten(), reduction="none").view(-1, S)
    with torch.no_grad():
        ok = torch.cumprod((logits.argmax(-1) == targets).float(), dim=1)
    bce = F.binary_cross_entropy_with_logits(conf.float(), ok, reduction="none")
    return ce.mean() + 0.1 * bce.mean(), ce.mean(0).detach()


# ----------------------------------------------------------------------------- export
def quant_fp8(w, bo: int, bi: int):
    """[N, K] -> (fp8 e4m3 [N, K], e8m0 scale [ceil(N/bo), ceil(K/bi)]): power-of-two (ue8m0) block
    scales, the format the checkpoint stores and torch_kernels.dequant_fp8 reads."""
    import torch
    N, K = w.shape
    gN, gK = -(-N // bo), -(-K // bi)
    x = torch.zeros(gN * bo, gK * bi, dtype=torch.float32, device=w.device)
    x[:N, :K] = w.float()
    blk = x.view(gN, bo, gK, bi)
    amax = blk.abs().amax(dim=(1, 3)).clamp_min(1e-30)
    e = torch.ceil(torch.log2(amax / FP8_MAX))
    s = torch.ldexp(torch.ones_like(amax), e.to(torch.int32))
    q = (blk / s[:, None, :, None]).clamp(-FP8_MAX, FP8_MAX).view(gN * bo, gK * bi)[:N, :K]
    return q.to(torch.float8_e4m3fn), s.to(torch.float8_e8m0fnu)


def export_dense(d: Drafter, raw: list[dict], trained: set | None = None) -> list[list[tuple[str, str, np.ndarray]]]:
    """raw[i]: {source name: (safetensors dtype, np array)} of shard 44+i, untransformed, in source
    order. Every TRAINED tensor is written back in its source dtype (FP8 E4M3 + E8M0 at the source's
    scale block shape; BF16; F32); every other tensor is byte-identical. Routed experts are frozen,
    so the separate arm's experts pass through untouched. `trained` (full source names, as
    make_trainable returns them) defaults to every parameter that requires grad; an empty set writes
    the source verbatim (the native head)."""
    import torch
    out = []
    for i, b in enumerate(d.blocks):
        params = {n: p for n, p in b.named_parameters()
                  if (f"mtp.{i}.{n}" in trained if trained is not None else p.requires_grad)}
        pre = f"mtp.{i}."
        items, done = [], set()
        for name, (dt, arr) in raw[i].items():
            if name in done:
                continue
            rel = name[len(pre):]
            if rel.endswith(".scale") and rel[:-6] + ".weight" in params:
                continue                                  # written next to its weight
            p = params.get(rel)
            if p is None:
                items.append((name, dt, arr))
                continue
            v = p.detach().float().cpu()
            if dt == "F8_E4M3":
                sname = name[:-7] + ".scale"
                sdt, sarr = raw[i][sname]
                bo, bi = -(-v.shape[0] // sarr.shape[0]), -(-v.shape[1] // sarr.shape[1])
                q, s = quant_fp8(v, bo, bi)
                items.append((name, dt, q.view(torch.uint8).numpy()))
                items.append((sname, sdt, s.view(torch.uint8).numpy()))
                done.add(sname)
            elif dt == "BF16":
                items.append((name, dt, v.to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)))
            elif dt == "F32":
                items.append((name, dt, v.numpy().astype(np.float32)))
            else:
                raise TypeError(f"{name}: trained tensor stored as {dt}; no export rule")
        out.append(items)
    return out
