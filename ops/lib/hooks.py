"""Instrumented replacements for reference forwards: MoE (statistics), Indexer and Block (memory).

MoE.forward is re-implemented line for line, with every statistic taken from tensors the
reference computes anyway. Its input `x` is ffn_norm(hc_pre(stream, attn_pre)), i.e. the REAP
hook sits after hc_pre as PLAN.md requires, and routing uses bias_vl on image spans because the
image mask is passed through exactly as Block.forward passes it.

The reference applies the routing weight INSIDE the expert, before w2 (Expert.forward:
`x = weights * x; return self.w2(x)`), and w2 quantizes its input per 32-block with a
power-of-two scale. So the expert's true contribution is w2(g * act); we record its norm as the
REAP term g*||f|| and report ||f|| as that norm / g. The two differ from a separately computed
ungated w2(act) only through the fp8 activation grid, and the recorded value is the one the
model actually adds to the stream.

Indexer.forward is re-implemented with the query dimension chunked: the reference materialises
index_score [b, s, h, t] (32 heads x 8k x 8k x 4 B = 8 GiB per row before the head sum). The
chunked version keeps the per-chunk tensor at `chunk` x 32 x t and is otherwise identical.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

HIST_LO, HIST_HI, HIST_BINS = -6.0, 3.0, 64


class FFNStats:
    """Everything one layer's MoE contributes to the saliency bank, float64 where it is a sum.

    Per bucket b and expert e (REAP family, as in ~/glm-5.3-reap/scripts/stream_saliency.py):
      sum/sq  of s = g*||f||;  nrm/nsq of ||f||;  gat/gsq of g;  cnt
    Per expert: hist of log10 s; osum = vector sum of the gated output (heal-gain measurement);
      fmax = max |f| element of the ungated output (Super-Expert guard), gfmax the gated one;
      amax_in / amax_mid = input amax of w1|w3 and of w2 (NVFP4 activation scales);
      x2_in [E, 5120] / x2_mid [E, 2304] = per-column sum x^2 (imatrix), n_tok.
    Shared expert: the same amax / x2 terms with suffix _sh.
    HOPE: F sum/cnt [E, E] (vendored FAccumulator, layer slot 0).
    """

    def __init__(self, n_buckets: int, E: int, dim: int, inter: int, device):
        from .hope import FAccumulator
        z = lambda *s: torch.zeros(*s, dtype=torch.float64, device=device)
        self.t = {k: z(n_buckets, E) for k in ("sum", "sq", "cnt", "nrm", "nsq", "gat", "gsq")}
        self.t.update(
            hist=torch.zeros(E, HIST_BINS, dtype=torch.int64, device=device),
            osum=z(E, dim), fmax=z(E), gfmax=z(E), amax_in=z(E), amax_mid=z(E),
            x2_in=z(E, dim), x2_mid=z(E, inter), n_tok=z(E),
            amax_in_sh=z(1), amax_mid_sh=z(1), x2_in_sh=z(dim), x2_mid_sh=z(inter), n_tok_sh=z(1),
        )
        self.F = FAccumulator(1, E, device=device)
        self.bucket = None  # LongTensor [S] bucket id per token of the current row
        self.valid = None   # BoolTensor [S]: tokens that count (pads and held-out rows excluded)
        self.cursor = 0     # MoE.forward may see the row in token chunks; this is where the next starts
        self.route = []     # per forward: (indices int16 [n, k], weights fp32 [n, k]), in token order

    def start_row(self, bucket, valid) -> None:
        self.bucket, self.valid, self.cursor, self.route = bucket, valid, 0, []

    def state(self) -> dict[str, torch.Tensor]:
        return {**self.t, "F_sum": self.F.sum[0], "F_cnt": self.F.cnt[0]}

    def load(self, st: dict[str, torch.Tensor]) -> None:
        for k, v in st.items():
            if k == "F_sum":
                self.F.sum[0].copy_(v)
            elif k == "F_cnt":
                self.F.cnt[0].copy_(v)
            else:
                self.t[k].copy_(v)


def _amax(t: torch.Tensor) -> torch.Tensor:
    return t.detach().abs().amax().double() if t.numel() else torch.zeros((), dtype=torch.float64, device=t.device)


def _expert_parts(expert, x, w, act_dtype):
    """Expert.forward split open: returns (w1|w3 input, w2 input, gated output)."""
    gate = expert.w1(x).float()
    up = expert.w3(x).float()
    if expert.swiglu_limit > 0:
        up = torch.clamp(up, min=-expert.swiglu_limit, max=expert.swiglu_limit)
        gate = torch.clamp(gate, max=expert.swiglu_limit)
    mid = F.silu(gate) * up
    if w is not None:
        mid = w * mid
    mid = mid.to(act_dtype)
    return mid, expert.w2(mid)


def instrument_moe(M, stats_of: dict) -> None:
    """Replace M.MoE.forward. `stats_of[id(moe)]` -> FFNStats for MoEs being measured; others
    run the plain reference arithmetic (same code path, no statistics)."""

    def forward(self, x, image_mask=None):
        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x, None if image_mask is None else image_mask.flatten())
        st = stats_of.get(id(self))
        y = torch.zeros_like(x, dtype=torch.float32)
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
        if st is not None:
            n = x.size(0)
            st.route.append((indices.to(torch.int16), weights.float()))
            valid = st.valid[st.cursor:st.cursor + n]
            bucket = st.bucket[st.cursor:st.cursor + n]
            st.cursor += n
            assert valid.numel() == n, "token count out of step with the row's valid mask"
            S_buf = torch.zeros_like(weights, dtype=torch.float32)
            T = st.t
        for i in range(self.experts_start_idx, self.experts_end_idx):
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            idx, top = torch.where(indices == i)
            xi, gi = x[idx], weights[idx, top, None]
            mid, out = _expert_parts(expert, xi, gi, x.dtype)
            y[idx] += out
            if st is None:
                continue
            with torch.no_grad():
                keep = valid[idx]
                if not bool(keep.any()):
                    continue
                o, g, b = out[keep].float(), gi[keep, 0].float(), bucket[idx[keep]]
                xk, mk = xi[keep].float(), mid[keep].float()
                gn = o.norm(dim=-1)                       # g * ||f||
                n = gn / g.clamp_min(1e-20)               # ||f||
                S_buf[idx[keep], top[keep]] = gn
                for name, v in (("sum", gn), ("sq", gn * gn), ("nrm", n), ("nsq", n * n), ("gat", g), ("gsq", g * g),
                                ("cnt", torch.ones_like(gn))):
                    T[name][:, i].index_add_(0, b, v.double())
                lg = torch.log10(gn.clamp_min(1e-12))
                bins = ((lg - HIST_LO) / (HIST_HI - HIST_LO) * HIST_BINS).clamp_(0, HIST_BINS - 1).long()
                T["hist"][i].scatter_add_(0, bins, torch.ones_like(bins))
                T["osum"][i] += o.sum(0).double()
                T["gfmax"][i] = torch.maximum(T["gfmax"][i], _amax(o))
                T["fmax"][i] = torch.maximum(T["fmax"][i], (o.abs().amax(-1) / g.clamp_min(1e-20)).max().double())
                T["amax_in"][i] = torch.maximum(T["amax_in"][i], _amax(xk))
                T["amax_mid"][i] = torch.maximum(T["amax_mid"][i], _amax(mk))
                T["x2_in"][i] += (xk * xk).sum(0).double()
                T["x2_mid"][i] += (mk * mk).sum(0).double()
                T["n_tok"][i] += xk.size(0)
        if st is not None:
            with torch.no_grad():
                st.F.update(0, indices[valid], S_buf[valid])
        sh = self.shared_experts
        mid, out = _expert_parts(sh, x, None, x.dtype)
        y += out
        if st is not None:
            with torch.no_grad():
                xs, ms = x[valid].float(), mid[valid].float()
                T["amax_in_sh"][0] = torch.maximum(T["amax_in_sh"][0], _amax(xs))
                T["amax_mid_sh"][0] = torch.maximum(T["amax_mid_sh"][0], _amax(ms))
                T["x2_in_sh"] += (xs * xs).sum(0).double()
                T["x2_mid_sh"] += (ms * ms).sum(0).double()
                T["n_tok_sh"] += xs.size(0)
        return y.type_as(x).view(shape)

    M.MoE.forward = forward


def pack_bits(m: torch.Tensor) -> torch.Tensor:
    """bool [..., n] -> uint8 [..., ceil(n/8)], MSB first (numpy.packbits order)."""
    n = m.size(-1)
    m = F.pad(m.to(torch.uint8), (0, -n % 8))
    w = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1], dtype=torch.uint8, device=m.device)
    return (m.unflatten(-1, (-1, 8)) * w).sum(-1).to(torch.uint8)


def unpack_bits(b: torch.Tensor, n: int) -> torch.Tensor:
    w = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1], dtype=torch.uint8, device=b.device)
    return ((b.unsqueeze(-1) & w) != 0).flatten(-2)[..., :n]


def chunked_indexer(M, budget_bytes: int = 2 << 30) -> None:
    """Replace M.Indexer.forward with a query-chunked prefill (decode is not used by any stage).

    Candidates (layer 20 -> index sources 24..36). The reference keeps, per query, the best
    candidate_topk_blocks x candidate_block_size = 16,384 compressed positions of layer 20's score.
    Up to that many reachable positions this keeps every reachable one, i.e. it is the identity
    on top of the causal mask, and nothing is stored. Beyond it (long rows at ratio 1, > 16k
    tokens) the source layer computes the reference select_candidate_blocks per query chunk and
    publishes the block-level keep mask bit-packed in `shared_attn.candidate_bits` [b, s, nb/8]
    (2 KiB per token at 128k); the consuming layers unpack it per chunk. s03 persists those bits
    across layers like the other cross-layer state."""
    ref = M.Indexer.forward

    def forward(self, x, qr, latent, start_pos, offset):
        if start_pos != 0:
            return ref(self, x, qr, latent, start_pos, offset)
        sa = M.shared_attn
        bsz, seqlen, _ = x.size()
        ratio, rd = self.compress_ratio, self.rope_head_dim
        if self.owns_k and latent is not None:
            freqs = self.freqs_cis[: seqlen - seqlen % ratio : ratio]
            k = self.k_norm(self.wk(latent))
            M.apply_rotary_emb(k[..., -rd:], freqs)
            M.fp4_act_quant(k, M.fp4_block_size, True)
            self.k_cache[:bsz, : k.size(1)] = k
        if self.owns_k:
            sa.index_k = self.k_cache
        q = self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.index_head_dim))
        M.apply_rotary_emb(q[..., -rd:], self.freqs_cis[0:seqlen])
        M.fp4_act_quant(q, M.fp4_block_size, True)
        n_t = seqlen // ratio
        index_k = sa.index_k[:bsz, :n_t]
        weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads ** -0.5)
        topk = min(self.index_topk, n_t)
        out = torch.empty(bsz, seqlen, topk, dtype=torch.int32, device=x.device)
        t_ar = torch.arange(n_t, device=x.device)
        blk = self.candidate_block_size
        nb = -(-n_t // blk)
        need_cand = n_t > self.candidate_topk_blocks * blk
        if self.is_candidate_source:
            sa.candidate_bits = (torch.empty(bsz, seqlen, -(-nb // 8), dtype=torch.uint8, device=x.device)
                                 if need_cand else None)
        chunk = int(max(16, min(1024, budget_bytes // max(1, self.n_local_heads * n_t * 4))))
        for s0 in range(0, seqlen, chunk):
            s1 = min(seqlen, s0 + chunk)
            sc = torch.einsum("bshd,btd->bsht", q[:, s0:s1], index_k)
            sc = (sc.relu_() * weights[:, s0:s1].unsqueeze(-1)).sum(dim=2)
            lens = (torch.arange(s0 + 1, s1 + 1, device=x.device) // ratio).unsqueeze(-1)
            sc.masked_fill_(t_ar >= lens, -torch.inf)
            if need_cand and self.is_candidate_source:
                keep = M.select_candidate_blocks(sc, lens, self.candidate_topk_blocks, blk)
                sa.candidate_bits[:, s0:s1] = pack_bits(keep[..., ::blk])
            elif need_cand and self.uses_candidates:
                keep = unpack_bits(sa.candidate_bits[:, s0:s1], nb).repeat_interleave(blk, dim=-1)[..., :n_t]
                sc.masked_fill_(~keep, -torch.inf)
            idxs = sc.topk(topk, dim=-1, sorted=False).indices.sort(dim=-1).values
            out[:, s0:s1] = torch.where(idxs < lens, idxs + offset, -1).int()
            del sc
        return out

    M.Indexer.forward = forward


def chunked_block(M, chunk: int = 8192) -> None:
    """Replace M.Block.forward with one that runs every per-token part (hc_mixes, hc_pre, norms,
    the MoE, hc_post) over token chunks and only attention over the whole sequence. Each of those
    parts is a function of its own token alone, so the result equals the reference; what changes
    is peak memory: the reference hc_post materialises [s, 4, 4, 5120] fp32 (43 GB at 128k).
    Rows up to `chunk` tokens take the reference path untouched."""
    ref = M.Block.forward

    def forward(self, x, start_pos, pre_mix, image_mask, *attn_args):
        b, S = x.shape[:2]
        if S <= chunk or start_pos != 0:
            return ref(self, x, start_pos, pre_mix, image_mask, *attn_args)
        x = x.clone()
        spans = [(s, min(S, s + chunk)) for s in range(0, S, chunk)]
        attn_in = torch.empty(b, S, x.size(-1), dtype=x.dtype, device=x.device)
        attn_pre = torch.empty(b, S, self.hc_mult, dtype=torch.float32, device=x.device)
        post = torch.empty_like(attn_pre)
        comb = torch.empty(b, S, self.hc_mult, self.hc_mult, dtype=torch.float32, device=x.device)
        for s0, s1 in spans:
            p, q, c = self.hc_mixes(x[:, s0:s1], self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
            attn_pre[:, s0:s1], post[:, s0:s1], comb[:, s0:s1] = p, q, c
            attn_in[:, s0:s1] = self.attn_norm(self.hc_pre(x[:, s0:s1], pre_mix[:, s0:s1]))
        a = self.attn(attn_in, start_pos, *attn_args)
        del attn_in
        for s0, s1 in spans:
            x[:, s0:s1] = self.hc_post(a[:, s0:s1], x[:, s0:s1], post[:, s0:s1], comb[:, s0:s1])
        del a
        ffn_pre = torch.empty_like(attn_pre)
        for s0, s1 in spans:
            p, q, c = self.hc_mixes(x[:, s0:s1], self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
            ffn_pre[:, s0:s1] = p
            h = self.ffn_norm(self.hc_pre(x[:, s0:s1], attn_pre[:, s0:s1]))
            f = self.ffn(h, None if image_mask is None else image_mask[:, s0:s1])
            x[:, s0:s1] = self.hc_post(f, x[:, s0:s1], q, c)
        return x, ffn_pre

    M.Block.forward = forward


def topk_logprobs(logits: torch.Tensor, k: int = 32) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """logits [n, V] fp32 -> (top-k log-probs [n, k], ids int32 [n, k], logsumexp [n])."""
    lse = torch.logsumexp(logits, dim=-1)
    v, i = logits.topk(k, dim=-1)
    return (v - lse[:, None]), i.int(), lse

