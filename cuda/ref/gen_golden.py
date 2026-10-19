"""gen_golden.py — reference golden-activation generator for the CUDA port.

Modes:
  smoke  : tiny random model (dtype=bf16, no real weights) — validates that the pure-torch
           kernel replacements + the full deepseek_v4 architecture run end-to-end
           (HC/Sinkhorn, MLA, sparse_attn, compressor, indexer, MoE, DSpark). No GPU RAM for weights.
  block  : real config from config.json; instantiate embed + ONE Block at --layer, load that
           block's real weights, feed a FIXED-seed input hidden, dump boundary tensors that the
           CUDA kernels gate against (block-in, attn_norm, attn-out, ffn-out, block-out).

  usage:
    python gen_golden.py smoke
    python gen_golden.py block --ckpt /model --config /model/config.json --layer 2 --out goldens/
"""
import os, sys, json, argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # local kernel.py / fast_hadamard shadow
import deepseek_v4_ref as M


def init_random_(model, n_routed):
    with torch.no_grad():
        for name, t in list(model.named_parameters()) + list(model.named_buffers()):
            if "tid2eid" in name:                       # hash-route table: valid expert ids
                t.random_(0, n_routed)
            elif t.dtype in (torch.float32, torch.bfloat16, torch.float16):
                t.normal_(0, 0.02)


def smoke():
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device(dev)
    print(f"[smoke] device={dev}")
    args = M.ModelArgs(
        dtype="bf16", expert_dtype=None, scale_dtype="fp32", scale_fmt=None,
        # head_dim-rope_head_dim and index_head_dim-rope_head_dim must be multiples of 64 (act_quant block)
        dim=512, n_layers=4, n_heads=8, head_dim=128, rope_head_dim=64,
        q_lora_rank=128, o_lora_rank=128, o_groups=4,
        n_routed_experts=8, n_activated_experts=2, n_shared_experts=1,
        moe_inter_dim=256, n_hash_layers=1, window_size=16,
        # length must cover layers 0..n_layers-1 PLUS the MTP layer index (n_layers); MTP => ratio 0
        compress_ratios=(0, 0, 4, 128, 0), index_n_heads=8, index_head_dim=128,
        index_topk=16, max_seq_len=256, max_batch_size=1,
        vocab_size=1024, score_func="sqrtsoftplus", route_scale=1.5, swiglu_limit=10.0,
        dspark_block_size=4, dspark_target_layer_ids=(2, 3), dspark_noise_token_id=1000,
        compress_rope_theta=160000.0, original_seq_len=0, rope_theta=10000.0, rope_factor=16,
    )
    model = M.Transformer(args)
    init_random_(model, args.n_routed_experts)
    x = torch.randint(0, args.vocab_size, (1, 64), device=dev)

    out_ids, logits, main_hidden = model(x[:, :48])
    print(f"[smoke] prefill: logits {tuple(logits.shape)} finite={torch.isfinite(logits).all().item()} "
          f"main_hidden {None if main_hidden is None else tuple(main_hidden.shape)}")
    model.forward_spec(out_ids, main_hidden)                      # prefill spec (builds draft KV)
    for i in range(48, 52):
        out_ids, logits, main_hidden = model(x[:, i:i + 1], i)
        spec = model.forward_spec(out_ids, main_hidden, i)
        if spec is not None:
            oids, slog, conf = spec
    print(f"[smoke] decode+spec: draft_ids {tuple(oids.shape)} conf {tuple(conf.shape)} "
          f"finite={torch.isfinite(slog).all().item()}")
    assert torch.isfinite(logits).all(), "non-finite logits"
    print("[smoke] PASS — architecture + swapped kernels run end-to-end.")


_HF2MA = {"hidden_size": "dim", "moe_intermediate_size": "moe_inter_dim", "num_hidden_layers": "n_layers",
          "num_hash_layers": "n_hash_layers", "num_nextn_predict_layers": "n_mtp_layers",
          "num_attention_heads": "n_heads", "num_experts_per_tok": "n_activated_experts",
          "scoring_func": "score_func", "routed_scaling_factor": "route_scale",
          "qk_rope_head_dim": "rope_head_dim", "rms_norm_eps": "norm_eps", "sliding_window": "window_size"}


def build_args(cfg):
    """Build ModelArgs from an HF config.json, mapping HF key names -> ModelArgs fields (they differ!)."""
    fields = M.ModelArgs.__dataclass_fields__; kw = {}
    for k, v in cfg.items():
        key = _HF2MA.get(k, k)
        if key in fields and not isinstance(v, dict):
            kw[key] = v
    rs = cfg.get("rope_scaling") or {}
    for hk, mk in [("factor", "rope_factor"), ("original_max_position_embeddings", "original_seq_len"),
                   ("beta_fast", "beta_fast"), ("beta_slow", "beta_slow")]:
        if hk in rs: kw[mk] = rs[hk]
    return M.ModelArgs(**kw)


def block(ckpt, config, layer, seq, out_dir):
    """Full-Block real-weights golden (pure-sliding, hash-routed layer 1), fp32. Dumps block_in +
    input_ids + ALL weights (attn + 160 fp4 experts + fp8 shared + gate + hc + norms) + block_out."""
    from safetensors.torch import save_file
    import raw_loader
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_dtype(torch.float32); torch.set_default_device(dev)
    args = build_args(json.load(open(config)))
    args.max_batch_size = 1; args.max_seq_len = max(256, seq * 2)
    assert args.compress_ratios[layer] == 0, f"block gate needs pure-sliding layer (got ratio {args.compress_ratios[layer]})"
    M.world_size = 1; M.rank = 0
    M.default_dtype = torch.float8_e4m3fn if args.dtype == "fp8" else torch.bfloat16
    M.scale_fmt = "ue8m0" if args.scale_dtype == "fp8" else args.scale_fmt
    M.scale_dtype = torch.float8_e8m0fnu if args.scale_dtype == "fp8" else torch.float32
    print(f"[block] layer={layer} dim={args.dim} inter={args.moe_inter_dim} nr={args.n_routed_experts} "
          f"na={args.n_activated_experts} hash_layers={args.n_hash_layers} route_scale={args.route_scale}")

    blk = M.Block(layer, args)
    class Wrap(torch.nn.Module):
        def __init__(s): super().__init__(); s.layers = torch.nn.ModuleList()
    w = Wrap()
    for _ in range(layer): w.layers.append(torch.nn.Module())
    w.layers.append(blk)
    raw_loader.load_state_into(w, ckpt, only_prefixes=[f"layers.{layer}."])
    blk.attn.wo_a.weight.data = blk.attn.wo_a.weight.data.float()

    torch.manual_seed(1234)
    x = torch.randn(1, seq, args.hc_mult, args.dim)
    ids = torch.randint(0, args.vocab_size, (1, seq))
    taps = {}
    def hook(name):
        def fn(m, i, o): taps[name] = (o if isinstance(o, torch.Tensor) else o[0]).detach().clone()
        return fn
    blk.attn.register_forward_hook(hook("attn_out"))          # mla output
    blk.ffn.register_forward_hook(hook("moe_out"))            # moe output
    out = blk(x, 0, ids)                                       # [1,seq,hc,dim]
    # recompute the two HC-block boundaries via the reference's own methods (for stage isolation)
    x1a, post_a, comb_a = blk.hc_pre(x, blk.hc_attn_fn, blk.hc_attn_scale, blk.hc_attn_base)   # [1,s,d]
    res2 = blk.hc_post(taps["attn_out"], x, post_a, comb_a)   # attn-block output [1,s,hc,d]
    x1f, _, _ = blk.hc_pre(res2, blk.hc_ffn_fn, blk.hc_ffn_scale, blk.hc_ffn_base)

    a, f = blk.attn, blk.ffn; nr = args.n_routed_experts
    def u8(p): return p.detach().view(torch.uint8).contiguous().cpu()
    def ff(p): return p.detach().float().contiguous().cpu()
    fc = a.freqs_cis[:seq]
    g = {
        "block_in": ff(x[0]), "block_out": ff(out[0]), "input_ids": ids[0].int().contiguous().cpu(),
        "wq_a": u8(a.wq_a.weight), "wq_a_s": ff(a.wq_a.scale), "wq_b": u8(a.wq_b.weight), "wq_b_s": ff(a.wq_b.scale),
        "wkv": u8(a.wkv.weight), "wkv_s": ff(a.wkv.scale), "wo_b": u8(a.wo_b.weight), "wo_b_s": ff(a.wo_b.scale),
        "q_norm": ff(a.q_norm.weight), "kv_norm": ff(a.kv_norm.weight),
        "wo_a": ff(a.wo_a.weight.view(args.o_groups, args.o_lora_rank, -1)), "attn_sink": ff(a.attn_sink),
        "cos": torch.view_as_real(fc)[..., 0].contiguous().cpu(), "sin": torch.view_as_real(fc)[..., 1].contiguous().cpu(),
        "attn_norm": ff(blk.attn_norm.weight), "ffn_norm": ff(blk.ffn_norm.weight),
        "hc_attn_fn": ff(blk.hc_attn_fn), "hc_attn_scale": ff(blk.hc_attn_scale), "hc_attn_base": ff(blk.hc_attn_base),
        "hc_ffn_fn": ff(blk.hc_ffn_fn), "hc_ffn_scale": ff(blk.hc_ffn_scale), "hc_ffn_base": ff(blk.hc_ffn_base),
        "gate_w": ff(f.gate.weight), "tid2eid": f.gate.tid2eid.detach().long().contiguous().cpu(),
        "w1": torch.stack([u8(f.experts[e].w1.weight) for e in range(nr)]), "w1s": torch.stack([ff(f.experts[e].w1.scale) for e in range(nr)]),
        "w2": torch.stack([u8(f.experts[e].w2.weight) for e in range(nr)]), "w2s": torch.stack([ff(f.experts[e].w2.scale) for e in range(nr)]),
        "w3": torch.stack([u8(f.experts[e].w3.weight) for e in range(nr)]), "w3s": torch.stack([ff(f.experts[e].w3.scale) for e in range(nr)]),
        "sw1": u8(f.shared_experts.w1.weight), "sw1s": ff(f.shared_experts.w1.scale),
        "sw2": u8(f.shared_experts.w2.weight), "sw2s": ff(f.shared_experts.w2.scale),
        "sw3": u8(f.shared_experts.w3.weight), "sw3s": ff(f.shared_experts.w3.scale),
        "dims": torch.tensor([seq, args.hc_mult, args.dim, args.moe_inter_dim, nr, args.n_activated_experts, args.vocab_size], dtype=torch.int32),
        "route_scale": torch.tensor([args.route_scale], dtype=torch.float32),
        "swiglu_limit": torch.tensor([args.swiglu_limit], dtype=torch.float32),
        # stage-isolation taps
        "tap_hc_attn_out": ff(x1a[0]), "tap_attn_out": ff(taps["attn_out"][0]), "tap_res2": ff(res2[0]),
        "tap_hc_ffn_out": ff(x1f[0]), "tap_moe_out": ff(taps["moe_out"][0]),
        "tap_post_a": ff(post_a[0]), "tap_comb_a": ff(comb_a[0]),
    }
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"block_layer{layer}_seq{seq}.safetensors")
    save_file({k: v.contiguous() for k, v in g.items()}, path)
    print(f"[block] wrote {path} ({os.path.getsize(path)/1e9:.2f} GB)  |out|max={out.abs().max():.4f}")


def mla(ckpt, config, layer, seq, out_dir):
    """Real-weights golden for a pure-sliding MLA layer (compress_ratio=0). Runs the reference Attention
    in fp32 (default dtype) so the gate isolates composition error, not bf16 noise. Dumps input + all
    weights (fp8 bytes+scale, bf16 wo_a, norms, sink) + freqs so tests/gate_mla.cu can replay it."""
    from safetensors.torch import save_file
    import raw_loader
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_dtype(torch.float32)          # fp32 activations to match the CUDA path
    torch.set_default_device(dev)
    args = build_args(json.load(open(config)))
    args.max_batch_size = 1; args.max_seq_len = max(256, seq * 2)
    assert args.compress_ratios[layer] == 0, f"layer {layer} is not pure-sliding (ratio={args.compress_ratios[layer]})"
    M.world_size = 1; M.rank = 0
    M.default_dtype = torch.float8_e4m3fn if args.dtype == "fp8" else torch.bfloat16
    M.scale_fmt = "ue8m0" if args.scale_dtype == "fp8" else args.scale_fmt
    M.scale_dtype = torch.float8_e8m0fnu if args.scale_dtype == "fp8" else torch.float32

    attn = M.Attention(layer, args)
    class Wrap(torch.nn.Module):
        def __init__(s): super().__init__(); s.layers = torch.nn.ModuleList()
    w = Wrap()
    for _ in range(layer): w.layers.append(torch.nn.Module())
    holder = torch.nn.Module(); holder.attn = attn; w.layers.append(holder)
    raw_loader.load_state_into(w, ckpt, only_prefixes=[f"layers.{layer}.attn."])
    # wo_a is declared bf16 but used in a raw einsum with fp32 activations -> cast to fp32 (matches CUDA path)
    attn.wo_a.weight.data = attn.wo_a.weight.data.float()

    torch.manual_seed(1234)
    x = torch.randn(1, seq, args.dim)
    o = attn(x, 0)                                   # [1,seq,dim]

    def u8(p): return p.detach().view(torch.uint8).contiguous().cpu()
    def f(p):  return p.detach().float().contiguous().cpu()
    fc = attn.freqs_cis[:seq]                         # complex [seq, rope_dim/2]
    g = {
        "x": f(x[0]), "o_ref": f(o[0]),
        "wq_a": u8(attn.wq_a.weight),   "wq_a_s": f(attn.wq_a.scale),
        "wq_b": u8(attn.wq_b.weight),   "wq_b_s": f(attn.wq_b.scale),
        "wkv":  u8(attn.wkv.weight),    "wkv_s":  f(attn.wkv.scale),
        "wo_b": u8(attn.wo_b.weight),   "wo_b_s": f(attn.wo_b.scale),
        "q_norm": f(attn.q_norm.weight), "kv_norm": f(attn.kv_norm.weight),
        "wo_a": f(attn.wo_a.weight.view(args.o_groups, args.o_lora_rank, -1)),
        "attn_sink": f(attn.attn_sink),
        "cos": torch.view_as_real(fc)[..., 0].contiguous().cpu(),
        "sin": torch.view_as_real(fc)[..., 1].contiguous().cpu(),
        "dims": torch.tensor([1, seq], dtype=torch.int32),
    }
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"mla_layer{layer}_seq{seq}.safetensors")
    save_file(g, path)
    print(f"[mla] layer={layer} seq={seq} dev={dev} wrote {path}")
    print("   |o_ref|max=%.4f  x%s wq_b%s wo_a%s" % (o.abs().max().item(), tuple(x.shape), tuple(attn.wq_b.weight.shape), tuple(g['wo_a'].shape)))


def cmla(ckpt, config, layer, seq, out_dir):
    """Real-weights golden for a COMPRESSED+indexer MLA layer (ratio-4). fp32; dumps input + all attn +
    main-compressor + indexer weights + query/compressed freqs so gate_cmla.cu replays compressed_attn_forward."""
    from safetensors.torch import save_file
    import raw_loader
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_dtype(torch.float32); torch.set_default_device(dev)
    args = build_args(json.load(open(config)))
    args.max_batch_size = 1; args.max_seq_len = max(256, seq * 2)
    assert args.compress_ratios[layer] in (4, 128), f"cmla needs a compressed layer (got {args.compress_ratios[layer]})"
    M.world_size = 1; M.rank = 0
    M.default_dtype = torch.float8_e4m3fn if args.dtype == "fp8" else torch.bfloat16
    M.scale_fmt = "ue8m0" if args.scale_dtype == "fp8" else args.scale_fmt
    M.scale_dtype = torch.float8_e8m0fnu if args.scale_dtype == "fp8" else torch.float32
    a = M.Attention(layer, args)
    class Wrap(torch.nn.Module):
        def __init__(s): super().__init__(); s.layers = torch.nn.ModuleList()
    w = Wrap()
    for _ in range(layer): w.layers.append(torch.nn.Module())
    holder = torch.nn.Module(); holder.attn = a; w.layers.append(holder)
    raw_loader.load_state_into(w, ckpt, only_prefixes=[f"layers.{layer}.attn."])
    a.wo_a.weight.data = a.wo_a.weight.data.float()
    if a.indexer is not None:
        a.indexer.weights_proj.weight.data = a.indexer.weights_proj.weight.data.float()   # bf16 -> fp32

    torch.manual_seed(1234)
    x = torch.randn(1, seq, args.dim)
    o = a(x, 0)                                          # full compressed-path forward
    mc, idx = a.compressor, a.indexer; ic = idx.compressor if idx is not None else None
    def u8(p): return p.detach().view(torch.uint8).contiguous().cpu()
    def ff(p): return p.detach().float().contiguous().cpu()
    fq = a.freqs_cis[:seq]; fc = a.freqs_cis[:seq:args.compress_ratios[layer]]
    g = {
        "x": ff(x[0]), "o_ref": ff(o[0]),
        "wq_a": u8(a.wq_a.weight), "wq_a_s": ff(a.wq_a.scale), "wq_b": u8(a.wq_b.weight), "wq_b_s": ff(a.wq_b.scale),
        "wkv": u8(a.wkv.weight), "wkv_s": ff(a.wkv.scale), "wo_b": u8(a.wo_b.weight), "wo_b_s": ff(a.wo_b.scale),
        "q_norm": ff(a.q_norm.weight), "kv_norm": ff(a.kv_norm.weight),
        "wo_a": ff(a.wo_a.weight.view(args.o_groups, args.o_lora_rank, -1)), "attn_sink": ff(a.attn_sink),
        "cos": torch.view_as_real(fq)[..., 0].contiguous().cpu(), "sin": torch.view_as_real(fq)[..., 1].contiguous().cpu(),
        "cc_cos": torch.view_as_real(fc)[..., 0].contiguous().cpu(), "cc_sin": torch.view_as_real(fc)[..., 1].contiguous().cpu(),
        "mc_wkv": ff(mc.wkv.weight), "mc_wgate": ff(mc.wgate.weight), "mc_ape": ff(mc.ape), "mc_norm": ff(mc.norm.weight),
        "dims": torch.tensor([seq, args.dim, args.q_lora_rank, args.window_size, args.compress_ratios[layer],
                              args.index_n_heads, args.index_head_dim, args.index_topk], dtype=torch.int32),
    }
    if idx is not None:
        g.update({"idx_wq_b": u8(idx.wq_b.weight), "idx_wq_b_s": ff(idx.wq_b.scale), "idx_weights_proj": ff(idx.weights_proj.weight),
                  "idx_c_wkv": ff(ic.wkv.weight), "idx_c_wgate": ff(ic.wgate.weight), "idx_c_ape": ff(ic.ape), "idx_c_norm": ff(ic.norm.weight)})
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"cmla_layer{layer}_seq{seq}.safetensors")
    save_file({k: v.contiguous() for k, v in g.items()}, path)
    print(f"[cmla] layer={layer} seq={seq} wrote {path} ({os.path.getsize(path)/1e9:.3f} GB)  |o|max={o.abs().max():.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    sub.add_parser("smoke")
    cm = sub.add_parser("cmla")
    cm.add_argument("--ckpt", required=True); cm.add_argument("--config", required=True)
    cm.add_argument("--layer", type=int, default=2); cm.add_argument("--seq", type=int, default=16)
    cm.add_argument("--out", default="goldens")
    b = sub.add_parser("block")
    b.add_argument("--ckpt", required=True); b.add_argument("--config", required=True)
    b.add_argument("--layer", type=int, default=2); b.add_argument("--seq", type=int, default=32)
    b.add_argument("--out", default="goldens")
    ml = sub.add_parser("mla")
    ml.add_argument("--ckpt", required=True); ml.add_argument("--config", required=True)
    ml.add_argument("--layer", type=int, default=1); ml.add_argument("--seq", type=int, default=16)
    ml.add_argument("--out", default="goldens")
    a = ap.parse_args()
    if a.mode == "smoke": smoke()
    elif a.mode == "block": block(a.ckpt, a.config, a.layer, a.seq, a.out)
    elif a.mode == "cmla": cmla(a.ckpt, a.config, a.layer, a.seq, a.out)
    else: mla(a.ckpt, a.config, a.layer, a.seq, a.out)
