"""Layer-streamed replay of the teacher forward for a SET OF ARMS over a SUBSET OF ROWS.

This is how every post-saliency decision is measured without writing a checkpoint (PLAN.md stage
3: "evaluated in memory"). One process loads ONE decoder layer with all 384 routed experts from the
source shards (exactly as s03 did) and pushes every arm's stream through it, each arm configured
in memory:

  keep     pruning = the router can no longer select the pruned experts. Their bias and bias_vl
           entries become -inf, so the top-6 is taken over the survivors and the weights are
           renormalised over the new top-6 by the unmodified reference Gate: PROPER RE-ROUTING,
           not "drop the slot" (lapadatbogdan: mask search without re-routing reached KL 1.52,
           worse than frequency).
  gain     per-layer scalar on the routed-expert sum (the s06 heal), shared expert untouched.
  thor     the kept experts replaced by the s07 Thor build, dequantized to bf16 with the
           activation fake-quantized to the reference fp8 grid, so a quantized arm differs from
           its MXFP4 arm by its weights only.
  overlay  the layer's FP8 dense Linears replaced by an NVFP4 overlay (same fake-quant rule).

Every arm starts from the same embedding (computed once) and carries its own residual stream,
pre_mix and cross-layer attention state (KV / index keys / top-k / candidates, s03's scheme) on
disk, so arms never contaminate each other. The head unit scores each arm against the cached
teacher of s03 (top-32 log-probs, NLL), per token.

LAYOUT  root/run.json            rows, arms (with keep-set fingerprints), corpus spec hash
        root/_embed/{h,pre_mix}.npy
        root/<arm>/{h,pre_mix}.npy, state/, journal/, metrics/r<row>.npz, obs/L<L>/r<row>.npz
        root/ledger.json
Streams are indexed by LOCAL token offsets (rows concatenated in ascending row order); Engram rows
and the teacher cache are read at GLOBAL offsets from s03.

CRASH SAFETY. As s03: a row's output goes to a journal (atomic rename) before the stream file is
overwritten in place, and the ledger marks a row applied only after the stream is flushed. A row
that crashed mid-write is re-applied from its journal; a row without a journal is untouched and is
recomputed. Two-phase layer units (s06, s07) cache the post-attention stream and never touch the
arm's stream until phase B, whose rows follow the same journal protocol.

Nothing here runs at import; torch is imported inside the functions.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import numpy as np

from . import env, keepset


def s03mod():
    from ops import s03_saliency as s03
    return s03


# ----------------------------------------------------------------------------- run description
class Run:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.spec = json.load(open(self.root / "run.json"))
        self.rows: list[int] = self.spec["rows"]
        self.arms: list[dict] = self.spec["arms"]
        s03 = s03mod()
        self.C = s03.Corpus()
        self.loc, off = {}, 0
        for r in self.rows:
            n = self.C.rows[r]["length"]
            self.loc[r] = slice(off, off + n)
            off += n
        self.T = off
        self.led = env.Ledger(self.root / "ledger.json")

    def arm(self, name: str) -> dict:
        return next(a for a in self.arms if a["name"] == name)

    def gsl(self, r: int) -> slice:
        return self.C.sl(r)

    def files(self, arm: str):
        s03 = s03mod()
        d = self.root / arm
        return (s03.mm(d / "h.npy", np.uint16, (self.T, s03.HC, s03.DIM)),
                s03.mm(d / "pre_mix.npy", np.float32, (self.T, s03.HC)))

    def state_path(self, arm: str, kind: str, src: int, r: int) -> Path:
        return self.root / arm / "state" / f"{kind}_L{src}" / f"r{r:05d}.npy"


def bytes_per_arm(T: int) -> int:
    return T * (4 * 5120 * 2 + 4 * 4)


def prepare(root: Path, rows: list[int], arms: list[dict], spec_hash: str) -> Run:
    """Create (or re-open and verify) a run. Arms are compared by name AND keep-set fingerprint, so
    a re-used run directory can never silently score a different mask under an old name."""
    root = Path(root)
    for a in arms:
        if a.get("keep") is not None:
            a["keep"] = keepset.canonical(a["keep"])
            a["keep_sha256"] = keepset.fingerprint(a["keep"])
    spec = {"rows": sorted(int(r) for r in rows), "arms": arms, "corpus_spec_hash": spec_hash}
    p = root / "run.json"
    if p.exists():
        old = json.load(open(p))
        if old["rows"] != spec["rows"] or old["corpus_spec_hash"] != spec_hash:
            raise SystemExit(f"{root}: existing run has different rows/corpus; use a fresh directory")
        have = {a["name"]: a for a in old["arms"]}
        for a in arms:
            if a["name"] in have and have[a["name"]].get("keep_sha256") != a.get("keep_sha256"):
                raise keepset.KeepsetMismatch(f"{root}: arm {a['name']} exists with a different keep-set")
            if a["name"] in have and {k: v for k, v in have[a["name"]].items() if k != "keep"} != \
                    {k: v for k, v in a.items() if k != "keep"}:
                raise SystemExit(f"{root}: arm {a['name']} exists with a different configuration")
        spec["arms"] = old["arms"] + [a for a in arms if a["name"] not in have]
    env.atomic_json(p, spec)
    return Run(root)


# ----------------------------------------------------------------------------- model plumbing
def load_reference(kernels: str, token_chunk: int):
    import torch
    from .hooks import chunked_block, chunked_indexer
    M = env.import_reference(kernels)
    if kernels == "torch":
        from . import torch_kernels
        torch_kernels.install(M, "torch")
    chunked_indexer(M)
    chunked_block(M, token_chunk)
    install_arm_moe(M)
    torch.set_default_dtype(torch.bfloat16)
    torch.cuda.memory._set_allocator_settings("expandable_segments:True")
    return M


def fake_act_quant(x):
    """The reference's fp8 activation grid (block 32, power-of-two ue8m0 scale), applied in place
    on a copy: what a quantized-weight arm's matmul input would have been in the MXFP4 teacher."""
    from . import torch_kernels as tk
    return tk.act_quant(x.clone(), 32, "ue8m0", inplace=True)


class FQExpert:
    """Expert.forward with bf16 weights and fake-quantized activations (Thor / overlay arms)."""

    def __init__(self, w1, w2, w3, swiglu_limit):
        self.w1, self.w2, self.w3, self.swiglu_limit = w1, w2, w3, swiglu_limit

    def parts(self, x, weights):
        import torch
        import torch.nn.functional as F
        dtype = x.dtype
        xq = fake_act_quant(x)
        gate = F.linear(xq, self.w1).float()
        up = F.linear(xq, self.w3).float()
        if self.swiglu_limit > 0:
            up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
            gate = torch.clamp(gate, max=self.swiglu_limit)
        mid = F.silu(gate) * up
        if weights is not None:
            mid = weights * mid
        mid = mid.to(dtype)
        return mid, F.linear(fake_act_quant(mid), self.w2)

    def __call__(self, x, weights=None):
        return self.parts(x, weights)[1]


class ArmLayer:
    """Per-(arm, layer) MoE configuration read by the patched MoE.forward."""

    def __init__(self, E: int, keep_ids=None, gain: float = 1.0, experts: dict | None = None, observer=None):
        import torch
        self.gain = float(gain)
        self.experts = experts             # original id -> callable expert (thor arms), else None
        self.observer = observer
        self.keep_ids = None if keep_ids is None else sorted(int(i) for i in keep_ids)
        add = torch.zeros(E, dtype=torch.float32, device="cuda")
        if keep_ids is not None:
            m = torch.ones(E, dtype=torch.bool, device="cuda")
            m[torch.as_tensor(self.keep_ids, device="cuda")] = False
            add[m] = float("-inf")
        self.bias_add = add


def configure(blk, arm_layer) -> None:
    """Point the block's router at this arm's survivors. Originals are kept on the module."""
    g = blk.ffn.gate
    if not hasattr(g, "_bias0"):
        g._bias0 = g.bias.data.clone()
        g._bias_vl0 = None if g.bias_vl is None else g.bias_vl.data.clone()
    g.bias.data = g._bias0 + arm_layer.bias_add
    if g.bias_vl is not None:
        g.bias_vl.data = g._bias_vl0 + arm_layer.bias_add
    blk.ffn._arm = arm_layer


def expert_parts(expert, x, w):
    """(w2 input, output) of one expert, matching Expert.forward (routing weight before w2)."""
    if isinstance(expert, FQExpert):
        return expert.parts(x, w)
    from .hooks import _expert_parts
    return _expert_parts(expert, x, w, x.dtype)


def install_arm_moe(M) -> None:
    """MoE.forward for arms: the reference arithmetic (same loop, same accumulation order), plus the
    arm's gain on the routed sum, expert overrides, and observer callbacks."""
    import torch

    def forward(self, x, image_mask=None):
        arm = getattr(self, "_arm", None)
        shape = x.size()
        x = x.view(-1, self.dim)
        im = None if image_mask is None else image_mask.flatten()
        weights, indices = self.gate(x, im)
        obs = arm.observer if arm is not None else None
        if obs is not None:
            obs.route(self, x, im, weights, indices)
        y = torch.zeros_like(x, dtype=torch.float32)
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
        table = arm.experts if (arm is not None and arm.experts is not None) else None
        for i in range(self.experts_start_idx, self.experts_end_idx):
            if counts[i] == 0:
                continue
            expert = table[i] if table is not None else self.experts[i]
            idx, top = torch.where(indices == i)
            gi = weights[idx, top, None]
            if obs is not None:
                mid, out = expert_parts(expert, x[idx], gi)
                obs.expert(i, idx, x[idx], mid, gi, out)
            else:
                out = expert(x[idx], gi)
            y[idx] += out
        if arm is not None and arm.gain != 1.0:
            y *= arm.gain
        if obs is not None:
            obs.routed(y)
        y += self.shared_experts(x)
        return y.type_as(x).view(shape)

    M.MoE.forward = forward


def build_layer(M, L: int, max_len: int):
    from . import shards
    args = env.model_args(M, max_batch_size=1, max_seq_len=max_len)
    idx = shards.ShardIndex()
    blk = shards.build_block(M, args, L, idx, "cuda", with_engram=True)
    idx.close()
    return blk, args


def thor_experts(M, args, build_dir: Path, L: int, keep_ids: list[int], expect_sha: str) -> dict:
    """Dequantize one layer's Thor experts to bf16 on the GPU -> {original id: FQExpert}."""
    import torch
    from . import thor_format as TF
    b = TF.Build(build_dir)
    keepset.require(b.m["keepset_sha256"], expect_sha, where=str(build_dir))
    rd = b.reader(TF.layer_file(L))
    dims = {"w1": (args.moe_inter_dim, args.dim), "w3": (args.moe_inter_dim, args.dim), "w2": (args.dim, args.moe_inter_dim)}
    W = {}
    for p in TF.PROJ:
        qt = b.m["tensors"].get(TF.expert_name(L, p, "qweight"))
        if qt is not None and qt["qtype"] in ("KT2", "KT3"):
            from . import kt
            sg = rd.get(TF.expert_name(L, "w2" if p == "w2" else "w13", "rht_sign"))
            W[p] = (qt["qtype"], rd.get(TF.expert_name(L, p, "qweight")), rd.get(TF.expert_name(L, p, "kt_scale")),
                    kt.unpack_signs(sg, dims[p][1]))
        elif qt is not None:
            arr = rd.get(TF.expert_name(L, p, "qweight"))
            W[p] = (qt["qtype"], arr, None, None)
        else:
            W[p] = ("MXFP4_SRC", rd.get(TF.expert_name(L, p, "weight")), rd.get(TF.expert_name(L, p, "scale")), None)
    out = {}
    for j, orig in enumerate(sorted(keep_ids)):
        ws = []
        for p in TF.PROJ:
            qt, arr, sc, sg = W[p]
            o, i = dims[p]
            if sg is not None:                       # trellis: GPU decode + un-rotation, bit-exact with numpy
                from . import kt
                ws.append(kt.dequant_t(qt, arr[j], sc[j], i, sg).to(torch.bfloat16))
                continue
            ws.append(torch.from_numpy(TF.dequant_stack(qt, arr, o, i, sc, idx=j)).to("cuda", torch.bfloat16))
        out[orig] = FQExpert(ws[0], ws[1], ws[2], args.swiglu_limit)
    return out


class Overlay:
    """Swap a block's FP8 dense Linear weights for an NVFP4 overlay (bf16 dequant) and back. The
    overlay file L<LL>.safetensors names tensors by module path inside the block (attn.wq_a.weight,
    .weight_scale, .weight_scale_2), as s07 `overlay` writes them."""

    def __init__(self, blk, overlay_dir: Path, L: int):
        import torch
        from . import ggml_quants as G, st_io
        self.saved = []
        rd = st_io.Reader(Path(overlay_dir) / f"L{L:02d}.safetensors")
        names = sorted({n.rsplit(".", 1)[0] for n in rd.names()})
        mods = dict(blk.named_modules())
        for n in names:
            w = G.nvfp4_dequant(rd.get(n + ".weight"), rd.get(n + ".weight_scale"),
                                float(rd.get(n + ".weight_scale_2").reshape(-1)[0]))
            mod = mods[n]                              # module path inside the block, e.g. attn.wq_a
            self.saved.append((mod, mod.weight, getattr(mod, "scale", None)))
            mod.weight = torch.nn.Parameter(torch.from_numpy(w).to("cuda", torch.bfloat16), requires_grad=False)
            if hasattr(mod, "scale"):
                mod.scale = None
            mod._fq = True
        self.names = names

    def restore(self):
        for mod, w, s in self.saved:
            mod.weight = w
            if s is not None:
                mod.scale = s
                mod.weight.scale = s
            mod._fq = False


def install_fq_linear(M) -> None:
    """Linear.forward (and the parallel subclasses, which override it) fake-quantizes its input when
    an overlay swapped in a bf16 weight. wo_a is read directly as a weight by an einsum and gets the
    overlay weight without an activation fake-quant (the reference also feeds it bf16)."""
    for cls in (M.Linear, M.ColumnParallelLinear, M.RowParallelLinear):
        ref = cls.forward

        def forward(self, x, _ref=ref):
            if getattr(self, "_fq", False):
                import torch.nn.functional as F
                return F.linear(fake_act_quant(x), self.weight)
            return _ref(self, x)

        cls.forward = forward


# ----------------------------------------------------------------------------- per-row helpers
def row_inputs(run: Run, arm: str, L: int, r: int):
    """(x [1,n,4,dim] bf16 cuda, pre_mix [1,n,4] f32 cuda, image_mask) for arm/row at layer L."""
    import torch
    src = "_embed" if L == 0 else arm
    H, PM = run.files(src)
    s = run.loc[r]
    x = torch.from_numpy(np.ascontiguousarray(H[s])).cuda().view(torch.bfloat16)[None]
    pm = torch.from_numpy(np.ascontiguousarray(PM[s])).cuda()[None]
    return x, pm, image_mask(run, r)


def image_mask(run: Run, r: int):
    import torch
    ty = run.C.types[run.gsl(r)]
    return torch.from_numpy(ty >= 0).cuda()[None] if (ty >= 0).any() else None


def install_states(M, run: Run, arm: str, L: int, r: int) -> None:
    import torch
    s03 = s03mod()
    for kind, src in s03.sources_for(L).items():
        a = np.load(run.state_path(arm, kind, src, r))
        sa = M.shared_attn
        if kind in ("kv", "ik"):
            t = torch.from_numpy(a).cuda().view(torch.bfloat16)[None]
            if kind == "kv":
                sa.compress_kv = t
            else:
                sa.index_k = t
        elif kind == "topk":
            sa.topk_idxs = torch.from_numpy(a).cuda()[None]
        else:
            sa.candidate_bits = None if a.size == 0 else torch.from_numpy(a).cuda()[None]


def journal_row(run: Run, arm: str, L: int, r: int, payload: dict) -> Path:
    jd = run.root / arm / "journal" / f"L{L}"
    jd.mkdir(parents=True, exist_ok=True)
    jf = jd / f"r{r:05d}.npz"
    with open(jf.with_suffix(".tmp"), "wb") as fh:
        np.savez(fh, **payload)
        fh.flush(); os.fsync(fh.fileno())
    os.replace(jf.with_suffix(".tmp"), jf)
    return jf


def apply_row(run: Run, arm: str, L: int, r: int, jf: Path) -> None:
    j = np.load(jf)
    H, PM = run.files(arm)
    s = run.loc[r]
    H[s] = j["h"].view(np.uint16)
    PM[s] = j["pre_mix"]
    H.flush(); PM.flush()
    for k in j.files:
        if k.startswith("state_"):
            p = run.state_path(arm, k[6:], L, r)
            p.parent.mkdir(parents=True, exist_ok=True)
            np.save(p, j[k])
        elif k.startswith("obs_"):
            p = run.root / arm / "obs" / f"L{L}" / f"r{r:05d}.npz"
            p.parent.mkdir(parents=True, exist_ok=True)
            np.savez(p, **{kk[4:]: j[kk] for kk in j.files if kk.startswith("obs_")})
    run.led.finish(f"{arm}/L{L}/r{r}")
    jf.unlink()


def reconcile(run: Run, arm: str, L: int) -> None:
    jd = run.root / arm / "journal" / f"L{L}"
    if not jd.exists():
        return
    for f in jd.glob("*.tmp"):
        f.unlink()
    for f in sorted(jd.glob("r*.npz")):
        r = int(f.stem[1:])
        if run.led.done(f"{arm}/L{L}/r{r}"):
            f.unlink()
        else:
            apply_row(run, arm, L, r, f)


def finish_layer(run: Run, arm: str, L: int) -> None:
    s03 = s03mod()
    for src in range(L + 1):
        for kind in s03.published_by(src):
            d = run.root / arm / "state" / f"{kind}_L{src}"
            if s03.last_consumer(kind, src) <= L and d.exists():
                shutil.rmtree(d)
    run.led.finish(f"{arm}/L{L}")


def forward_row(M, blk, run: Run, arm: str, L: int, r: int, token_chunk: int, x=None, pm=None, im=None,
                ffn_only_from=None):
    """One row through the block for one arm. Returns the journal payload (not yet applied)."""
    import torch
    s03 = s03mod()
    from . import memfence
    n = run.loc[r].stop - run.loc[r].start
    memfence.require(n * s03.BYTES_PER_TOKEN_PEAK / 2**30 + 2.0, f"{arm} L{L} r{r} ({n} tok)")
    if ffn_only_from is None:
        if x is None:
            x, pm, im = row_inputs(run, arm, L, r)
        install_states(M, run, arm, L, r)
    with torch.inference_mode():
        if ffn_only_from is None:
            if blk.engram is not None:
                x = s03.run_engram(blk, x, L, run.gsl(r), im, token_chunk)
            y, ffn_pre = blk(x, 0, pm, im)
        else:
            x_mid, attn_pre = ffn_only_from
            y, ffn_pre = ffn_half(blk, x_mid, attn_pre, im, token_chunk)
    payload = {"h": y[0].contiguous().view(torch.int16).cpu().numpy(), "pre_mix": ffn_pre[0].float().cpu().numpy()}
    if ffn_only_from is None:
        for kind in s03.published_by(L):
            payload["state_" + kind] = s03.grab_state(M, blk, kind, n)
    return payload


def attn_half(blk, x, pre_mix, chunk):
    """Block.forward up to and including the attention hc_post (hooks.chunked_block, first half)."""
    import torch
    b, S = x.shape[:2]
    x = x.clone()
    spans = [(s, min(S, s + chunk)) for s in range(0, S, chunk)]
    attn_in = torch.empty(b, S, x.size(-1), dtype=x.dtype, device=x.device)
    attn_pre = torch.empty(b, S, blk.hc_mult, dtype=torch.float32, device=x.device)
    post = torch.empty_like(attn_pre)
    comb = torch.empty(b, S, blk.hc_mult, blk.hc_mult, dtype=torch.float32, device=x.device)
    for s0, s1 in spans:
        p, q, c = blk.hc_mixes(x[:, s0:s1], blk.hc_attn_fn, blk.hc_attn_scale, blk.hc_attn_base)
        attn_pre[:, s0:s1], post[:, s0:s1], comb[:, s0:s1] = p, q, c
        attn_in[:, s0:s1] = blk.attn_norm(blk.hc_pre(x[:, s0:s1], pre_mix[:, s0:s1]))
    a = blk.attn(attn_in, 0)
    del attn_in
    for s0, s1 in spans:
        x[:, s0:s1] = blk.hc_post(a[:, s0:s1], x[:, s0:s1], post[:, s0:s1], comb[:, s0:s1])
    return x, attn_pre


def ffn_input(blk, x_mid, attn_pre, s0, s1):
    return blk.ffn_norm(blk.hc_pre(x_mid[:, s0:s1], attn_pre[:, s0:s1]))


def ffn_half(blk, x, attn_pre, image_mask, chunk):
    import torch
    S = x.shape[1]
    x = x.clone()
    ffn_pre = torch.empty_like(attn_pre)
    for s0 in range(0, S, chunk):
        s1 = min(S, s0 + chunk)
        p, q, c = blk.hc_mixes(x[:, s0:s1], blk.hc_ffn_fn, blk.hc_ffn_scale, blk.hc_ffn_base)
        ffn_pre[:, s0:s1] = p
        h = ffn_input(blk, x, attn_pre, s0, s1)
        f = blk.ffn(h, None if image_mask is None else image_mask[:, s0:s1])
        x[:, s0:s1] = blk.hc_post(f, x[:, s0:s1], q, c)
    return x, ffn_pre


# ----------------------------------------------------------------------------- units
def unit_embed(run: Run, kernels: str) -> None:
    import torch
    s03 = s03mod()
    from . import shards
    env.assert_no_gpu_server("replay")
    M = env.import_reference(kernels)
    idx = shards.ShardIndex()
    torch.set_default_dtype(torch.bfloat16)
    W = idx.get("embed.weight").cuda()
    H, PM = run.files("_embed")
    vis = None
    for r in run.rows:
        u = f"_embed/r{r}"
        if run.led.done(u):
            continue
        row = run.C.rows[r]
        g = run.gsl(r)
        ids = torch.from_numpy(run.C.tokens[g].astype(np.int64)).cuda()
        h = torch.nn.functional.embedding(ids, W)
        if row["images"]:
            vis = vis or s03.build_vision(M, idx)
            s03.merge_images(M, vis, row, h, run.C.types[g])
        s = run.loc[r]
        H[s] = h.unsqueeze(1).repeat(1, s03.HC, 1).view(torch.int16).cpu().numpy().view(np.uint16)
        pm = np.zeros((h.size(0), s03.HC), np.float32); pm[:, 0] = 1.0
        PM[s] = pm
        H.flush(); PM.flush()
        run.led.finish(u)
    run.led.finish("_embed")


def arm_layer_for(M, args, run: Run, arm: dict, L: int, observer=None):
    E = args.n_routed_experts
    keep = None if arm.get("keep") is None else arm["keep"][str(L)]
    if arm.get("gain") and str(L) not in arm["gain"]:
        raise KeyError(f"heal gain for layer {L} missing from arm {arm.get('name', '?')}")
    gain = 1.0 if not arm.get("gain") else float(arm["gain"][str(L)])
    experts = None
    if arm.get("thor"):
        experts = thor_experts(M, args, Path(arm["thor"]), L, keep, arm["keep_sha256"])
    return ArmLayer(E, keep, gain, experts, observer)


def unit_layer(run: Run, L: int, kernels: str, token_chunk: int, arms: list[str] | None = None,
               observer_factory=None) -> None:
    """Single-phase layer unit: every selected arm, every row, through layer L."""
    import torch
    env.assert_no_gpu_server("replay")
    M = load_reference(kernels, token_chunk)
    install_fq_linear(M)
    blk, args = build_layer(M, L, run.C.max_len)
    torch.set_default_device("cuda")
    for arm in run.arms:
        name = arm["name"]
        if (arms and name not in arms) or run.led.done(f"{name}/L{L}"):
            continue
        if L > 0 and not run.led.done(f"{name}/L{L - 1}"):
            raise SystemExit(f"{name}: layer {L - 1} not finished")
        obs = observer_factory(name, L) if observer_factory else None
        al = arm_layer_for(M, args, run, arm, L, obs)
        configure(blk, al)
        ov = Overlay(blk, Path(arm["overlay"]), L) if arm.get("overlay") else None
        reconcile(run, name, L)
        t0, ntok = time.time(), 0
        for r in run.rows:
            if run.led.done(f"{name}/L{L}/r{r}"):
                continue
            if obs is not None:
                obs.start_row(r)
            payload = forward_row(M, blk, run, name, L, r, token_chunk)
            if obs is not None:
                for k, v in (obs.end_row(r) or {}).items():
                    payload["obs_" + k] = v
            apply_row(run, name, L, r, journal_row(run, name, L, r, payload))
            ntok += run.loc[r].stop - run.loc[r].start
        if ov is not None:
            ov.restore()
        al.experts = None
        torch.cuda.empty_cache()
        finish_layer(run, name, L)
        env.log(f"{name} L{L}: {ntok} tok in {time.time() - t0:.0f}s", "replay")


def unit_layer_two_phase(run: Run, L: int, kernels: str, token_chunk: int, arm_name: str, *,
                         measure, decide, cache_ffn_in: bool = True) -> None:
    """Phase A: attention half for every row (cached to disk) and `measure(blk, h, im, r, s0)` on
    the FFN inputs of each token chunk under this arm's routing, then `measure.finish()` persists
    what it accumulated (phase A is only marked done after that); then `decide(M, blk, args, run,
    cache) -> ArmLayer` builds the layer's final configuration (a measured gain, freshly quantized
    experts); phase B runs the FFN half with it and advances the stream. Phase A is idempotent
    (it never touches the arm's stream), so a crash restarts the layer from phase A unless phase B
    already began, in which case the decision is reloaded through `decide` (it must be
    deterministic from its own persisted outputs)."""
    import torch
    s03 = s03mod()
    env.assert_no_gpu_server("replay")
    M = load_reference(kernels, token_chunk)
    install_fq_linear(M)
    blk, args = build_layer(M, L, run.C.max_len)
    torch.set_default_device("cuda")
    arm = run.arm(arm_name)
    if run.led.done(f"{arm_name}/L{L}"):
        return
    if L > 0 and not run.led.done(f"{arm_name}/L{L - 1}"):
        raise SystemExit(f"{arm_name}: layer {L - 1} not finished")
    cache = run.root / arm_name / "cache" / f"L{L}"
    cache.mkdir(parents=True, exist_ok=True)
    base = arm_layer_for(M, args, run, arm, L)          # routing for phase A: this arm's survivors
    configure(blk, base)
    phase_b_started = any(run.led.done(f"{arm_name}/L{L}/r{r}") for r in run.rows) or \
        any((run.root / arm_name / "journal" / f"L{L}").glob("r*.npz"))
    if not run.led.done(f"{arm_name}/L{L}/phaseA"):
        if phase_b_started:
            raise SystemExit(f"{arm_name} L{L}: phase B rows exist without a finished phase A")
        for r in run.rows:
            n = run.loc[r].stop - run.loc[r].start
            x, pm, im = row_inputs(run, arm_name, L, r)
            install_states(M, run, arm_name, L, r)
            with torch.inference_mode():
                if blk.engram is not None:
                    x = s03.run_engram(blk, x, L, run.gsl(r), im, token_chunk)
                x_mid, attn_pre = attn_half(blk, x, pm, token_chunk)
                payload = {"x_mid": x_mid[0].contiguous().view(torch.int16).cpu().numpy(),
                           "attn_pre": attn_pre[0].cpu().numpy()}
                for kind in s03.published_by(L):
                    payload["state_" + kind] = s03.grab_state(M, blk, kind, n)
                hs = []
                for s0 in range(0, n, token_chunk):
                    s1 = min(n, s0 + token_chunk)
                    h = ffn_input(blk, x_mid, attn_pre, s0, s1)
                    measure(blk, h, None if im is None else im[:, s0:s1], r, s0)
                    if cache_ffn_in:
                        hs.append(h[0].contiguous().view(torch.int16).cpu().numpy())
                if cache_ffn_in:
                    payload["ffn_in"] = np.concatenate(hs)
            with open(cache / f"r{r:05d}.tmp", "wb") as fh:
                np.savez(fh, **payload)
            os.replace(cache / f"r{r:05d}.tmp", cache / f"r{r:05d}.npz")
        if hasattr(measure, "finish"):
            measure.finish()               # persist what phase A measured BEFORE marking it done
        run.led.finish(f"{arm_name}/L{L}/phaseA")
    final = decide(M, blk, args, run, cache)
    configure(blk, final)
    reconcile(run, arm_name, L)
    for r in run.rows:
        if run.led.done(f"{arm_name}/L{L}/r{r}"):
            continue
        c = np.load(cache / f"r{r:05d}.npz")
        x_mid = torch.from_numpy(c["x_mid"]).cuda().view(torch.bfloat16)[None]
        attn_pre = torch.from_numpy(c["attn_pre"]).cuda()[None]
        payload = forward_row(M, blk, run, arm_name, L, r, token_chunk, ffn_only_from=(x_mid, attn_pre),
                              im=image_mask(run, r))
        for k in c.files:
            if k.startswith("state_"):
                payload[k] = c[k]
        apply_row(run, arm_name, L, r, journal_row(run, arm_name, L, r, payload))
    final.experts = None
    finish_layer(run, arm_name, L)
    shutil.rmtree(cache)


def unit_head(run: Run, arms: list[str] | None = None) -> None:
    """Final hc_pre + norm + head per arm -> per-token metrics against the s03 teacher cache:
    kl32 = sum over the teacher's top-32 of p_t (log p_t - log p_s) (the GLM `topk_KL`, student
    log-probs gathered exactly from the full student softmax), student top-1, student NLL."""
    import torch
    s03 = s03mod()
    from . import shards
    env.assert_no_gpu_server("replay")
    idx = shards.ShardIndex()
    norm_w = idx.get("norm.weight").float().cuda()
    W = idx.get("head.weight").float().cuda()
    D = s03.D
    t_lp = np.load(D / "teacher" / "top32_logprob.npy", mmap_mode="r")
    t_id = np.load(D / "teacher" / "top32_ids.npy", mmap_mode="r")
    eps = s03.CFG["norm_eps"]
    for arm in run.arms:
        name = arm["name"]
        if (arms and name not in arms) or run.led.done(f"{name}/head"):
            continue
        if not run.led.done(f"{name}/L{s03.N_LAYERS - 1}"):
            raise SystemExit(f"{name}: layers not finished")
        H, PM = run.files(name)
        md = run.root / name / "metrics"
        md.mkdir(parents=True, exist_ok=True)
        for r in run.rows:
            if run.led.done(f"{name}/head/r{r}"):
                continue
            s, g = run.loc[r], run.gsl(r)
            n = s.stop - s.start
            kl = np.empty(n, np.float32); top1 = np.empty(n, np.int32); nll = np.empty(n, np.float32)
            for c0 in range(0, n, 1024):
                c1 = min(n, c0 + 1024)
                x = torch.from_numpy(np.ascontiguousarray(H[s.start + c0:s.start + c1])).cuda().view(torch.bfloat16)
                pm = torch.from_numpy(np.ascontiguousarray(PM[s.start + c0:s.start + c1])).cuda()
                h = torch.sum(pm.unsqueeze(-1) * x.float(), dim=1).to(torch.bfloat16)
                hf = h.float()
                hn = (norm_w * (hf * torch.rsqrt(hf.square().mean(-1, keepdim=True) + eps))).to(torch.bfloat16)
                logits = torch.nn.functional.linear(hn.float(), W)
                lse = torch.logsumexp(logits, -1)
                ti = torch.from_numpy(np.ascontiguousarray(t_id[g.start + c0:g.start + c1]).astype(np.int64)).cuda()
                tl = torch.from_numpy(np.ascontiguousarray(t_lp[g.start + c0:g.start + c1])).cuda()
                sl = logits.gather(1, ti) - lse[:, None]
                kl[c0:c1] = (tl.exp() * (tl - sl)).sum(-1).cpu().numpy()
                top1[c0:c1] = logits.argmax(-1).int().cpu().numpy()
                nxt = np.full(c1 - c0, -1, np.int64)
                m = min(c1, n - 1) - c0
                if m > 0:
                    nxt[:m] = run.C.tokens[g.start + c0 + 1:g.start + c0 + 1 + m]
                tg = torch.from_numpy(np.maximum(nxt, 0)).cuda()
                nl = (lse - logits.gather(1, tg[:, None])[:, 0]).cpu().numpy()
                nll[c0:c1] = np.where(nxt >= 0, nl, np.nan)
            np.savez(md / f"r{r:05d}.npz", kl32=kl, top1=top1, nll=nll)
            run.led.finish(f"{name}/head/r{r}")
        run.led.finish(f"{name}/head")


def drop_streams(run: Run, arm: str) -> None:
    """After the head: the arm's stream and state are dead weight (metrics are kept)."""
    for p in ("h.npy", "pre_mix.npy"):
        (run.root / arm / p).unlink(missing_ok=True)
    for d in ("state", "journal", "cache"):
        shutil.rmtree(run.root / arm / d, ignore_errors=True)
