#!/usr/bin/env python3
"""s03 vectorized MoE statistics == the per-expert reference (vendored below, verbatim but for the
module prefix), on a tiny random MoE built from the REFERENCE model.py with the torch kernels. CPU only.

    .venv/bin/python ops/tests/test_s03_vectorized.py [path/to/hooks.py]
"""
from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: F401  (used by the vendored reference)

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from ops.tests.test_s11 import ref, tiny_args  # noqa: E402

HOOKS = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "ops/lib/hooks.py"
S03 = Path(sys.argv[2]) if len(sys.argv) > 2 else REPO / "ops/s03_saliency.py"
ENV = Path(sys.argv[3]) if len(sys.argv) > 3 else REPO / "ops/lib/env.py"
spec = importlib.util.spec_from_file_location(
    "ops.lib.hooks_under_test", HOOKS, loader=importlib.machinery.SourceFileLoader("ops.lib.hooks_under_test", str(HOOKS)))
import ops.lib  # noqa: E402,F401  (parent package for the relative import of .hope)
H = importlib.util.module_from_spec(spec); spec.loader.exec_module(H)


def _instrument_moe_reference(M, stats_of: dict) -> None:
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
            mid, out = H._expert_parts(expert, xi, gi, x.dtype)
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
                bins = ((lg - H.HIST_LO) / (H.HIST_HI - H.HIST_LO) * H.HIST_BINS).clamp_(0, H.HIST_BINS - 1).long()
                T["hist"][i].scatter_add_(0, bins, torch.ones_like(bins))
                T["osum"][i] += o.sum(0).double()
                T["gfmax"][i] = torch.maximum(T["gfmax"][i], H._amax(o))
                T["fmax"][i] = torch.maximum(T["fmax"][i], (o.abs().amax(-1) / g.clamp_min(1e-20)).max().double())
                T["amax_in"][i] = torch.maximum(T["amax_in"][i], H._amax(xk))
                T["amax_mid"][i] = torch.maximum(T["amax_mid"][i], H._amax(mk))
                T["x2_in"][i] += (xk * xk).sum(0).double()
                T["x2_mid"][i] += (mk * mk).sum(0).double()
                T["n_tok"][i] += xk.size(0)
        if st is not None:
            with torch.no_grad():
                st.F.update(0, indices[valid], S_buf[valid])
        sh = self.shared_experts
        mid, out = H._expert_parts(sh, x, None, x.dtype)
        y += out
        if st is not None:
            with torch.no_grad():
                xs, ms = x[valid].float(), mid[valid].float()
                T["amax_in_sh"][0] = torch.maximum(T["amax_in_sh"][0], H._amax(xs))
                T["amax_mid_sh"][0] = torch.maximum(T["amax_mid_sh"][0], H._amax(ms))
                T["x2_in_sh"] += (xs * xs).sum(0).double()
                T["x2_mid_sh"] += (ms * ms).sum(0).double()
                T["n_tok_sh"] += xs.size(0)
        return y.type_as(x).view(shape)

    M.MoE.forward = forward




def tiny_moe(M, args, seed):
    g = torch.Generator().manual_seed(seed)
    moe = M.MoE(1, args)
    sd = {}
    for name, p in moe.state_dict().items():
        if name.endswith(".scale"):
            continue
        if p.dtype == torch.float4_e2m1fn_x2:
            sd[name] = torch.randint(0, 256, tuple(p.shape), generator=g, dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
            sd[name[:-7] + ".scale"] = torch.full((p.shape[0], p.shape[1] * 2 // 32), 2.0 ** -4).to(torch.float8_e8m0fnu)
        elif p.dtype == torch.float8_e4m3fn:
            sd[name] = (torch.randn(tuple(p.shape), generator=g) * 0.5).to(torch.float8_e4m3fn)
            sd[name[:-7] + ".scale"] = torch.full(moe.state_dict()[name[:-7] + ".scale"].shape, 0.05).to(
                moe.state_dict()[name[:-7] + ".scale"].dtype)
        elif name.endswith(("bias", "bias_vl")):
            sd[name] = (torch.randn(tuple(p.shape), generator=g) * 0.1).to(p.dtype)
        else:
            sd[name] = (torch.randn(tuple(p.shape), generator=g) * 0.05).to(p.dtype)
    missing, unexpected = moe.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected
    for m in moe.modules():
        if hasattr(m, "weight") and isinstance(getattr(m, "scale", None), torch.nn.Parameter):
            m.weight.scale = m.scale
    return moe.eval()


def run(M, install, moe, x, bucket, valid, chunks, image_mask=None):
    stats_of = {}
    install(M, stats_of)
    st = H.FFNStats(5, moe.n_routed_experts, moe.dim, moe.experts[0].w2.weight.shape[1] * (
        2 if moe.experts[0].w2.weight.dtype == torch.float4_e2m1fn_x2 else 1), "cpu")
    stats_of[id(moe)] = st
    st.start_row(bucket, valid)
    ys = []
    with torch.inference_mode():
        for a, b in chunks:
            ys.append(moe(x[a:b], None if image_mask is None else image_mask[a:b]))
    return torch.cat(ys), st


def check(seed, n=96, chunks=((0, 40), (40, 96)), all_invalid=False):
    M, _ = ref()
    args = tiny_args(M)
    moe = tiny_moe(M, args, seed)
    g = torch.Generator().manual_seed(seed + 1)
    x = (torch.randn(n, args.dim, generator=g)).bfloat16()
    bucket = torch.randint(0, 5, (n,), generator=g)
    valid = torch.zeros(n, dtype=torch.bool) if all_invalid else torch.rand(n, generator=g) > 0.3
    # force empty experts: steer every token away from experts 0 and 1
    moe.gate.bias.data[:2] = -100.0
    ya, sa = run(M, _instrument_moe_reference, moe, x, bucket, valid, chunks)
    yb, sb = run(M, lambda M_, so: H.instrument_moe(M_, so, group_rows=7), moe, x, bucket, valid, chunks)
    assert torch.equal(ya, yb), "forward output changed"
    for (ia, wa), (ib, wb) in zip(sa.route, sb.route):
        assert torch.equal(ia, ib) and torch.equal(wa, wb)
    exact = ("cnt", "hist", "n_tok", "amax_in", "amax_mid", "fmax", "gfmax", "amax_in_sh", "amax_mid_sh", "n_tok_sh")
    worst = {}
    for k, va in sa.state().items():
        vb = sb.state()[k]
        if k in exact or k == "F_cnt":
            assert torch.equal(va, vb), (k, (va - vb).abs().max())
        else:
            den = va.abs().max().clamp_min(1e-300)
            worst[k] = float((va - vb).abs().max() / den)
    return worst


def _load(name, path):
    sp = importlib.util.spec_from_file_location(name, path, loader=importlib.machinery.SourceFileLoader(name, str(path)))
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
    return m


def journal_writer_and_ledger():
    import tempfile
    import numpy as np
    s03 = _load("s03_under_test", S03)
    envm = _load("ops.lib.env_under_test", ENV)
    d = Path(tempfile.mkdtemp())
    jw = s03.JournalWriter()
    pls = {r: {"h": np.full((3, 4), r, np.int16), "rw": np.arange(r + 1, dtype=np.float16)} for r in range(5)}
    for r, pl in pls.items():
        jw.put(d / f"r{r:05d}.npz", pl)
    jw.drain()
    for r, pl in pls.items():
        with np.load(d / f"r{r:05d}.npz") as j:
            assert all(np.array_equal(j[k], v) for k, v in pl.items())
    assert not list(d.glob("*.tmp"))
    jw.put(d / "missing_dir" / "r9.npz", pls[0])          # write fails -> drain raises, later puts refuse
    try:
        jw.drain(); raise AssertionError("drain did not surface the writer error")
    except RuntimeError:
        pass
    led = envm.Ledger(d / "led.json")
    led.finish("a"); led.finish_many(["r1", "r2", "r3"])
    led2 = envm.Ledger(d / "led.json")
    assert all(led2.done(u) for u in ("a", "r1", "r2", "r3")) and not led2.done("r4")
    print("journal writer (order, durability, error surfacing) and Ledger.finish_many ok")


def main():
    torch.set_num_threads(4)
    journal_writer_and_ledger()
    w = {}
    for seed in range(4):
        for k, v in check(seed).items():
            w[k] = max(w.get(k, 0.0), v)
    check(7, all_invalid=True)
    # fp64-accumulated sums: bucket x expert terms are fp64 sums of identical fp32 values (order
    # only); osum / x2_* the reference summed in fp32 per expert per call, so they differ at fp32
    fp64_order = ("sum", "sq", "nrm", "nsq", "gat", "gsq", "F_sum", "x2_in_sh", "x2_mid_sh")
    for k, v in sorted(w.items()):
        tol = 1e-9 if k in fp64_order else 1e-5
        print(f"  {k:10s} max rel diff {v:.2e} (tol {tol:.0e})")
        assert v <= tol, (k, v)
    print("exact: forward output, route, cnt, hist, n_tok, amax*, fmax, gfmax, F_cnt")
    bench()
    print("OK")


def bench():
    """CPU micro-benchmark: wall time and kernel-ish op count (torch profiler events) per forward."""
    M, _ = ref()
    args = tiny_args(M)
    import dataclasses
    args = dataclasses.replace(args, n_routed_experts=64, n_activated_experts=6)
    moe = tiny_moe(M, args, 11)
    g = torch.Generator().manual_seed(5)
    n = 2048
    x = torch.randn(n, args.dim, generator=g).bfloat16()
    bucket = torch.randint(0, 5, (n,), generator=g); valid = torch.rand(n, generator=g) > 0.1
    from torch.profiler import profile, ProfilerActivity
    for name, inst in (("reference", _instrument_moe_reference), ("vectorized", H.instrument_moe)):
        run(M, inst, moe, x, bucket, valid, ((0, n),))
        t0 = time.time()
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            run(M, inst, moe, x, bucket, valid, ((0, n),))
        ops = sum(1 for e in prof.events() if e.name.startswith("aten::") and e.cpu_parent is None)
        print(f"  bench {name:10s}: {time.time() - t0:.3f} s, {ops} top-level aten ops "
              f"(64 experts, top-6, {n} tokens)")


if __name__ == "__main__":
    main()
