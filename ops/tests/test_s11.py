#!/usr/bin/env python3
"""s11 tests on a tiny random DSpark drafter built from the REFERENCE module (inference/model.py with
the pure-torch kernels), CPU only. No checkpoint weights, no GPU.

    .venv/bin/python ops/tests/test_s11.py
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

TESTS = []


def test(fn):
    TESTS.append(fn)
    return fn


_CACHE = {}


def ref():
    if "M" not in _CACHE:
        import torch
        from ops.lib import env, torch_kernels
        M = env.import_reference("torch")
        torch_kernels.install(M, "torch")
        _CACHE["M"], _CACHE["tk"] = M, torch_kernels
    return _CACHE["M"], _CACHE["tk"]


def tiny_args(M, dtype="fp8"):
    import torch
    args = M.ModelArgs(max_batch_size=4, max_seq_len=256, dtype=dtype, expert_dtype="fp4", vocab_size=320, dim=256,
                       moe_inter_dim=256, n_layers=2, n_mtp_layers=3, n_heads=4, n_routed_experts=8,
                       n_activated_experts=2, q_lora_rank=64, head_dim=64, rope_head_dim=32, o_groups=2,
                       o_lora_rank=64, window_size=16, compress_ratios=(0, 0, 0, 0, 0), kv_source_layers=(),
                       index_source_layers=(), index_n_heads=4, index_head_dim=32, index_topk=8,
                       dspark_block_size=5, dspark_noise_token_id=7, dspark_target_layer_ids=(0, 1),
                       dspark_markov_rank=32, dspark_n_routed_experts=8, dspark_n_activated_experts=2,
                       temperature=0.0, norm_eps=1e-6, swiglu_limit=10.0, route_scale=1.5)
    M.default_dtype = torch.float8_e4m3fn if dtype == "fp8" else torch.bfloat16
    torch.set_default_dtype(torch.bfloat16)
    return args


def raw_block(M, args, i, seed):
    """Random source-format tensors for mtp.i (the dtypes the checkpoint stores)."""
    import torch
    from ops.lib import dspark
    g = torch.Generator().manual_seed(seed)
    b = M.DSparkBlock(args.n_layers + i, args)    # cpu, not meta: precompute_freqs_cis is lru_cached
    sd = {}
    for name, p in b.state_dict().items():
        if name.endswith(".scale"):
            continue
        shape = tuple(p.shape)
        if p.dtype == torch.float8_e4m3fn:
            w = torch.randn(shape, generator=g) * 0.05
            q, s = dspark.quant_fp8(w, 32, 32)
            sd[name], sd[name[:-7] + ".scale"] = q, s
        elif p.dtype == torch.float4_e2m1fn_x2:
            sd[name] = torch.randint(0, 256, shape, generator=g, dtype=torch.uint8).view(torch.int8)
            sd[name[:-7] + ".scale"] = torch.full((shape[0], shape[1] * 2 // 32), 2.0 ** -6).to(torch.float8_e8m0fnu)
        elif "norm" in name and name.endswith("weight"):
            sd[name] = torch.ones(shape, dtype=p.dtype)
        elif name.endswith(("hc_attn_scale", "hc_ffn_scale")):
            sd[name] = torch.ones(shape, dtype=p.dtype)
        elif name.endswith(("hc_attn_base", "hc_ffn_base", "gate.bias", "gate.bias_vl", "attn_sink")):
            sd[name] = torch.zeros(shape, dtype=p.dtype)
        elif "window_kv_cache" in name or "freqs_cis" in name:
            continue
        else:
            sd[name] = (torch.randn(shape, generator=g) * 0.05).to(p.dtype)
    return sd


def tiny_drafter(seed=0):
    import torch
    from ops.lib import dspark
    M, tk = ref()
    args = tiny_args(M)
    raws = [raw_block(M, args, i, seed + i) for i in range(3)]
    sds = [dspark.mp1({k: v.clone() for k, v in r.items()}) for r in raws]
    g = torch.Generator().manual_seed(seed + 99)
    emb = torch.randn(args.vocab_size, args.dim, generator=g).bfloat16()
    head = (torch.randn(args.vocab_size, args.dim, generator=g) * 0.05).float()
    d = dspark.build(M, args, sds, emb, head, "cpu")
    return d, raws, args


def inputs(args, b, W, seed=1):
    import torch
    g = torch.Generator().manual_seed(seed)
    mh = torch.randn(b, W, len(args.dspark_target_layer_ids) * args.dim, generator=g).bfloat16()
    tok = torch.randint(0, args.vocab_size, (b,), generator=g)
    return mh, tok


@test
def greedy_equals_teacher_forced_on_its_own_drafts():
    import torch
    from ops.lib import dspark
    d, _, args = tiny_drafter()
    mh, tok = inputs(args, 3, 12)
    with torch.inference_mode():
        lg, dr, cf = dspark.step(d, mh, tok)
        lt, dt, ct = dspark.step(d, mh, tok, targets=dr)
    assert dr.shape == (3, args.dspark_block_size)
    assert torch.allclose(lg, lt, atol=1e-5, rtol=0), (lg - lt).abs().max()
    assert torch.equal(dt, dr)
    assert torch.allclose(cf, ct, atol=1e-5)


@test
def shifted_window_equals_absolute_position():
    """RoPE relative invariance: an anchor replayed at W-1 equals the same anchor at a later absolute
    start (window filled by a longer prefill whose first rows fall out of the window)."""
    import torch
    from ops.lib import dspark
    d, _, args = tiny_drafter()
    W = args.window_size
    mh, tok = inputs(args, 2, W + 9)
    with torch.inference_mode():
        lg1, dr1, _ = dspark.step(d, mh[:, -W:], tok)
        lg2, dr2, _ = dspark.step(d, mh, tok)       # prefill W+8 rows: the cache keeps the last W-ish slots
    # same window content, different absolute offset: equal up to the fp8 activation grid, which the
    # rotated kv lands on differently (e4m3 keeps 3 mantissa bits); the slot-0 logits agree to a few %
    rel = ((lg1[:, 0] - lg2[:, 0]).norm() / lg2[:, 0].norm()).item()
    assert rel < 0.10, rel


@test
def backward_reaches_dense_not_experts():
    import torch
    from ops.lib import dspark
    d, _, args = tiny_drafter()
    M, tk = ref()
    trained = dspark.make_trainable(d, tk)
    assert trained and not any(".ffn.experts." in n and ".shared_experts." not in n for n in trained), list(trained)[:5]
    mh, tok = inputs(args, 2, 10)
    tgt = torch.randint(0, args.vocab_size, (2, args.dspark_block_size))
    for b in d.blocks:
        b.train()
    lg, _, cf = dspark.step(d, mh, tok, targets=tgt)
    loss, per = dspark.loss_fn(lg, cf, tgt)
    loss.backward()
    got = [n for n, p in d.blocks[0].named_parameters() if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0]
    assert any("main_proj" in n for n in got) and any("attn.wq_a" in n for n in got), got
    for n, p in d.blocks[2].named_parameters():
        if ".experts." in n and "shared" not in n:
            assert p.grad is None, n
    assert per.shape == (args.dspark_block_size,)
    # a second step must not backprop through the first step's window cache
    lg, _, cf = dspark.step(d, mh, tok, targets=tgt)
    dspark.loss_fn(lg, cf, tgt)[0].backward()


@test
def export_keeps_untrained_bytes_and_requantizes_trained():
    import torch
    from ops.lib import dspark
    d, raws, args = tiny_drafter()
    M, tk = ref()
    dspark.make_trainable(d, tk)
    with torch.no_grad():
        d.blocks[0].attn.wq_a.weight.mul_(1.5)
    to_np = lambda t: (t.view(torch.uint8).numpy() if t.dtype in (torch.float8_e4m3fn, torch.float8_e8m0fnu, torch.int8)
                       else t.view(torch.int16).numpy().view(np.uint16) if t.dtype == torch.bfloat16 else t.numpy())
    dts = {torch.float8_e4m3fn: "F8_E4M3", torch.float8_e8m0fnu: "F8_E8M0", torch.int8: "I8", torch.bfloat16: "BF16",
           torch.float32: "F32"}
    raw = [{f"mtp.{i}.{k}": (dts[v.dtype], to_np(v)) for k, v in r.items()} for i, r in enumerate(raws)]
    out = dspark.export_dense(d, raw)
    o0 = {n: (dt, a) for n, dt, a in out[0]}
    assert set(o0) == set(raw[0]), set(o0) ^ set(raw[0])
    ex = [n for n in raw[0] if ".ffn.experts." in n and "shared" not in n]
    assert ex and all(np.array_equal(o0[n][1], raw[0][n][1]) for n in ex)
    w = torch.from_numpy(o0["mtp.0.attn.wq_a.weight"][1]).view(torch.float8_e4m3fn)
    s = torch.from_numpy(o0["mtp.0.attn.wq_a.scale"][1]).view(torch.float8_e8m0fnu)
    back = tk.dequant_fp8(w, s)
    ref_w = d.blocks[0].attn.wq_a.weight.float()
    rel = ((back - ref_w).norm() / ref_w.norm()).item()
    assert rel < 0.08, rel
    w0 = tk.dequant_fp8(torch.from_numpy(raw[0]["mtp.0.attn.wq_a.weight"][1]).view(torch.float8_e4m3fn),
                        torch.from_numpy(raw[0]["mtp.0.attn.wq_a.scale"][1]).view(torch.float8_e8m0fnu))
    assert ((back - 1.5 * w0).norm() / (1.5 * w0).norm()).item() < 0.08


@test
def instrumented_moe_is_identical_and_banks_feed_selection():
    import torch
    from ops.lib import dspark
    d0, _, args = tiny_drafter()
    d1, _, _ = tiny_drafter()
    M, _ = ref()
    banks = dspark.instrument(M, d1)
    mh, tok = inputs(args, 3, 10)
    with torch.inference_mode():
        a = dspark.step(d0, mh, tok)
        b = dspark.step(d1, mh, tok)
    assert torch.allclose(a[0], b[0], atol=1e-4), (a[0] - b[0]).abs().max()
    z = [{k: getattr(bk, k) for k in ("cnt", "gat", "sum", "nrm", "F_sum", "F_cnt", "A13", "A2")} for bk in banks]
    assert all(x["cnt"].sum() > 0 for x in z)
    keep = dspark.select_keep(z, "reap_sum", 4)
    assert sorted(keep) == ["0", "1", "2"] and all(len(v) == 4 for v in keep.values())
    for c in ("hope_sum", "hope_cond", "mass", "freq"):
        assert all(len(v) == 4 for v in dspark.select_keep(z, c, 4).values()), c
    full = {k: list(range(8)) for k in ("0", "1", "2")}
    d2, _, _ = tiny_drafter()
    dspark.apply_keep(M, d2, full)
    with torch.inference_mode():
        c = dspark.step(d2, mh, tok)
    assert torch.allclose(a[0], c[0], atol=1e-4)
    d3, _, _ = tiny_drafter()
    dspark.apply_keep(M, d3, keep)
    with torch.inference_mode():
        e = dspark.step(d3, mh, tok)
    assert e[1].shape == a[1].shape and d3.blocks[0].ffn.gate.weight.shape[0] == 4


@test
def kquant_experts_roundtrip_through_the_quantized_arm():
    import torch
    from ops.lib import dspark, ggml_quants as G
    from ops import s11_spec as S
    d, raws, args = tiny_drafter()
    M, tk = ref()
    keep = {k: [0, 2, 5, 7] for k in ("0", "1", "2")}
    A = [{"A13": np.ones((8, args.dim)), "A2": np.ones((8, args.moe_inter_dim))} for _ in range(3)]
    q = S.quantize_experts(d, keep, A, "Q4_K", tk)
    assert len(q) == 3 and len(q[0]) == 4
    w1 = q[0][0]["w1"]
    assert w1.dtype == np.uint8 and w1.shape == (args.moe_inter_dim, G.row_bytes("Q4_K", args.dim))
    src = tk.dequant_fp4(d.blocks[0].ffn.experts[0].w1.weight.data, d.blocks[0].ffn.experts[0].w1.scale.data, 32)
    back = G.dequant("Q4_K", w1).reshape(args.moe_inter_dim, args.dim)
    rel = np.linalg.norm(back - src.float().numpy()) / np.linalg.norm(src.float().numpy())
    assert rel < 0.15, rel
    bf = S.dequantize_experts(q, args)
    dspark.apply_keep(M, d, keep, bf)
    mh, tok = inputs(args, 2, 10)
    with torch.inference_mode():
        lg, dr, _ = dspark.step(d, mh, tok)
    assert torch.isfinite(lg).all()


@test
def inmodel_head_file_roundtrip_and_fit():
    import tempfile
    from ops.lib import dspark, st_io
    from ops import s11_spec as S
    d, raws, args = tiny_drafter()
    M, tk = ref()
    keep = {k: [1, 3, 4, 6] for k in ("0", "1", "2")}
    A = [{"A13": np.ones((8, args.dim)), "A2": np.ones((8, args.moe_inter_dim))} for _ in range(3)]
    q = S.quantize_experts(d, keep, A, "Q3_K", tk)
    to_np = lambda t: (t.view(torch.uint8).numpy() if t.dtype in (torch.float8_e4m3fn, torch.float8_e8m0fnu, torch.int8)
                       else t.view(torch.int16).numpy().view(np.uint16) if t.dtype == torch.bfloat16 else t.numpy())
    import torch
    dts = {torch.float8_e4m3fn: "F8_E4M3", torch.float8_e8m0fnu: "F8_E8M0", torch.int8: "I8", torch.bfloat16: "BF16",
           torch.float32: "F32"}
    raw = [{f"mtp.{i}.{k}": (dts[v.dtype], to_np(v)) for k, v in r.items()} for i, r in enumerate(raws)]
    dense = dspark.export_dense(d, raw, trained=set())
    for i in range(3):
        assert all(np.array_equal(a, raw[i][n][1]) for n, _, a in dense[i])
    with tempfile.TemporaryDirectory() as td:
        from pathlib import Path
        hd = Path(td) / "inmodel_native"
        files = [S.inmodel_items(i, dense[i], keep[str(i)], q[i], "Q3_K") for i in range(3)]
        S.write_head(hd, files, {"arm": "inmodel", "head": "native", "qtype": "Q3_K"})
        rd = st_io.Reader(hd / "mtp-0.safetensors")
        assert not any(".experts.0." in n for n in rd.names())
        assert np.asarray(rd.get("mtp.0.ffn.gate.weight")).shape[0] == 4
        back = S.read_qexperts(hd, keep, "Q3_K")
        assert np.array_equal(back[2][3]["w2"], q[2][3]["w2"])
        GI = 1 << 30
        srv = {"name": "A192", "thor": {"files": {"layer-00.safetensors": {"bytes": 105 * GI}, "mtp-0.safetensors": {"bytes": 3 * GI}}}}
        rep = S.fit_report(srv, int(1.5 * GI), int(7.4 * GI), {"fit": {"budget_gib": 117.0, "workspace_gib": 6.0,
                                                                       "main_kv_1m_gib_default": 2.0}})
        assert rep["arms"]["inmodel"]["fits"] and not rep["arms"]["separate"]["fits"], rep
        assert abs(rep["arms"]["base"]["resident_gib"] - 113.0) < 1e-6


@test
def acceptance_bookkeeping():
    from ops.lib import dspark
    from ops import s11_spec as S
    dr = np.array([[1, 2, 3, 4, 5], [1, 9, 3, 4, 5], [0, 2, 3, 4, 5]])
    tg = np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
    assert dspark.accept_len(dr, tg).tolist() == [5, 1, 0]
    a = dspark.trace_anchors(n=40, gen_start=20, W=16, S=5)
    assert a[0] == 19 and a[-1] == 40 - 7
    st = S.accept_stats(np.array([5, 1, 0, 3]), np.array([0, 0, 1, 1]), depths=(1, 2, 3), draws=200)
    assert abs(st["alpha"]["1"]["mean"] - 0.75) < 1e-9 and abs(st["alpha"]["2"]["mean"] - 0.5) < 1e-9
    assert abs(st["mean_accepted"]["mean"] - 2.25) < 1e-9


@test
def decision_rules():
    from ops import s11_spec as S
    g = S.gates()
    base = {"tok_s_128": 10.0, "tok_s_8k": 8.0}
    arms = {"inmodel/tuned": {"fits": True, "e2e": {"tok_s_128": 13.0, "tok_s_8k": 9.0, "identical": True}},
            "separate/tuned": {"fits": True, "e2e": {"tok_s_128": 14.0, "tok_s_8k": 8.4, "identical": True}},
            "inmodel/native": {"fits": True, "e2e": {"tok_s_128": 11.0, "tok_s_8k": 8.1, "identical": False}}}
    dec = S.choose(arms, base, g)
    assert dec["ship"] == "inmodel/tuned", dec
    assert "inmodel/native" in dec["rejected"] and "identical" in " ".join(dec["rejected"]["inmodel/native"])
    none = S.choose({"separate/tuned": {"fits": True, "e2e": {"tok_s_128": 9.0, "tok_s_8k": 7.0, "identical": True}}}, base, g)
    assert none["ship"] is None
    blocked = S.choose({"inmodel/tuned": {"fits": True, "e2e": None}}, base, g)
    assert blocked["ship"] is None and blocked["status"] == "blocked"


@test
def gguf_gate_and_card():
    from ops import s09_gate, s12_publish as P
    from ops.lib import cards
    g = json.load(open(REPO / "ops" / "s10_gates.json"))["gguf"]
    k = lambda kl, t1, fp="f": {"stats": {"kl_top32": {"mean": kl, "ci95": [kl - .01, kl + .01]},
                                          "top1": {"mean": t1, "ci95": [t1 - .01, t1 + .01]},
                                          "dnll": {"mean": .1, "ci95": [.09, .11]}},
                                "identity": {"keep_set_sha256": fp}, "tokens": 1000}
    kld = {"L0": k(.30, .88), "L1": k(.40, .86), "L2": k(.60, .82), "L3": k(.90, .75), "L4": None}
    out = s09_gate.evaluate(kld, {"L0": True, "L1": True, "L2": False, "L3": True, "L4": True}, g)
    assert out["passing"] == ["L0", "L1"], out
    assert any("smoke" in w for w in out["levels"]["L2"]["why"])
    assert any("kl32" in w for w in out["levels"]["L3"]["why"]) and out["levels"]["L4"]["why"]
    bad = s09_gate.evaluate({"L0": k(.3, .9, "a"), "L1": k(.3, .9, "b")}, {"L0": True, "L1": True}, g)
    assert bad["passing"] == []
    m = {"levels": {"L0": {"expert_bpw": 4.25, "bytes": 140 << 30, "kld": kld["L0"]["stats"]},
                    "L1": {"expert_bpw": 3.4, "bytes": 115 << 30, "kld": kld["L1"]["stats"]}},
         "dense_precision": {"attention": "Q8_0", "shared expert": "Q6_K"}, "n_routed_experts": 192,
         "keepset_sha256": "ab" * 32, "engram": "llamacpp/ENGRAM_ROWFILE.md", "files": []}
    out["levels"]["L0"]["ngl"] = "40"
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        try:
            P.gguf_context(m, out, Path(td))
            raise AssertionError("token sets unchecked")
        except SystemExit:
            pass
        for lv in ("L0", "L1"):
            (Path(td) / lv / "teacher_eval").mkdir(parents=True)
            (Path(td) / lv / "teacher_eval" / "teacher_kld.json").write_text(json.dumps(kld[lv]))
        ctx = P.gguf_context(m, out, Path(td))
    assert "| L0 | 4.25 | 140.0 |" in ctx["levels_table"] and "| 40 |" in ctx["levels_table"]
    assert "Q6_K" in ctx["dense_table"] and ctx["gate_status"].startswith("PASS")
    parts = {p.name[: -len(".md.part")]: p.read_text() for p in (REPO / "ops" / "cards").glob("*.md.part")}
    o = {"kl32": {"mean": .1, "lo": .09, "hi": .11}, "top1": {"mean": .9, "lo": .89, "hi": .91},
         "dnll": {"mean": .1, "lo": .09, "hi": .11}, "confident_flip_rate": .01}
    full = {"base": {"repo": "x/y", "revision": "r"}, "gguf": ctx, "artifact": {"name": "n", "n_routed_base": 384},
            "s10": {"tokens": 5, "rows": 2, "gates_sha256": "s" * 64,
                    "summaries": {a: {"overall": o} for a in ("control", "mxfp4_A", "thorA", "b128")}},
            "engram": {"list": "- e"}, "spec": P.spec_context(Path("/nonexistent"))}
    text = cards.render((REPO / "ops" / "cards" / "gguf.md.tmpl").read_text(), full, parts)
    assert "Not measured yet" in text and "| L1 | 3.40 |" in text and "not requantized" not in text


if __name__ == "__main__":
    ok = fail = 0
    for fn in TESTS:
        try:
            fn()
            print(f"ok    {fn.__name__}")
            ok += 1
        except Exception:
            print(f"FAIL  {fn.__name__}")
            traceback.print_exc()
            fail += 1
    print(f"\n{ok}/{ok + fail} passed")
    sys.exit(1 if fail else 0)
