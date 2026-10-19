#!/usr/bin/env python3
"""s03: layer-major saliency pass over the s02 corpus. One pass banks every statistic later stages
need, so the 475 GiB teacher is never forwarded twice (PLAN.md stage 2).

ORDER. Layer-major, the opposite of GLM's chunk-major order, because the backbone is 3x the free
disk: each unit loads ONE decoder layer (one ~7.4 GB shard) and pushes every corpus row through
it. The residual stream (4 hyper-connection copies, bf16, 40,960 B/token) and pre_mix (fp32
[4]) live in one on-disk array updated in place, so the corpus size is bounded by NVMe, not RAM.

UNITS (each its own process; `run` executes them in order with a verified reclaim between):
  hash          token ids -> Engram hash ids [T, 2, 24] (reference NgramHashState, CPU)
  engram L      rows of the layer-L Engram table for those ids, one sequential sweep of shard
                47 (L=1) or 48 (L=14); fp8 values [T, 24, 256] + E8M0 scales [T, 24, 8]. CPU/IO
                only; the tables can live on another drive (--engram-dir).
  embed         embedding (+ ViT/aligner for vision rows) -> stream, identity pre_mix
  layer L       one decoder layer over every row (0..39): Engram first for L in {1, 14}, then
                the reference Block with the instrumented MoE; banks the statistics below
  head          final hc_pre + norm + head -> teacher top-32 log-probs, logsumexp, next-token NLL

CRASH SAFETY (layer units). A row's outputs go to a journal file first; the stream file is only
overwritten after the accumulator checkpoint that already includes the row is on disk:
  compute row -> journal (atomic) -> [every --ckpt-rows rows] accumulators + covered-row list
  (atomic) -> apply journals to the stream / routing / state files -> mark applied -> delete
On restart, rows in the checkpoint are re-applied from their journals (idempotent); rows after
it are recomputed from the untouched stream, and their stale journals are discarded. Nothing is
counted twice and nothing is lost (GLM lesson: a (chunk, block) ledger against double counting).

CROSS-LAYER STATE. Attention in V4.1 is shared down the stack: KV-source layers 2/8/14/20 publish
the compressed KV and index keys, index-source layers 2,8,...,36 publish top-k indices, layer 20
publishes candidate blocks. Under layer-major order those are saved per row by the publishing
layer (state/<kind>_L<src>/r<row>.npy) and re-installed into model.shared_attn before each row of
a consuming layer; a state directory is deleted once its last consumer has finished.

STATISTICS (bank/L<L>.npz + routing/L<L>_*.npy), held-out rows excluded from every accumulator:
  REAP family per pool and expert (sum/sq of g*||f||, nrm/nsq, gat/gsq, cnt), log-saliency hist,
  gated-output vector sum (heal gain), HOPE F (384 x 384, sum and count), Super-Expert max
  down_proj output, input amax of w1|w3 and w2 (NVFP4 activation scales), per-column sum x^2 of
  both expert inputs (imatrix), the same for the shared expert; teacher routing (top-6 ids and
  weights) for every token of every row, held-out included (heal replay, drift test, s04 KL).

MEMORY. One layer resident (~7.4 GB MXFP4/FP8) plus one row's activations. Rows are 8k except the
long-context pool (32k-128k); attention runs over the whole row, every per-token part runs in
8k token chunks (hooks.chunked_block), the indexer is query-chunked (hooks.chunked_indexer), and
memfence.require() is called with the row's predicted peak before each row.

RUN (only with no model server up; the stage refuses otherwise):
    setsid nohup python3 ops/s03_saliency.py run > ops/logs/s03.log 2>&1 &
    python3 ops/s03_saliency.py status
Options of note: --kernels torch (if TileLang fails s01), --engram-dir /mnt/seagate/v41,
--fetch-shards / --drop-shards (stream one layer shard at a time when the disk cannot hold 475 GiB).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ops import corpus_spec as spec  # noqa: E402
from ops.lib import env  # noqa: E402

STAGE = "s03"
CORPUS = env.WORK / "corpus"
D = env.WORK / "s03"
LEDGER = env.LOGS / "s03_ledger.json"
CFG = json.load(open(env.INFERENCE / "config.json"))
N_LAYERS, HC, DIM, INTER, E, K = CFG["n_layers"], CFG["hc_mult"], CFG["dim"], CFG["moe_inter_dim"], CFG["n_routed_experts"], CFG["n_activated_experts"]
ENGRAM_LAYERS = CFG["engram_layer_ids"]                    # [1, 14]
N_COLS = (CFG["engram_max_ngram_size"] - 1) * CFG["engram_n_heads"]   # 24
EHD = CFG["engram_head_dim"]                               # 256
TOPK_LOGITS = 32
BYTES_PER_TOKEN_PEAK = 240_000       # s01 must replace this estimate with the measured value


# ----------------------------------------------------------------------------- corpus layout
class Corpus:
    def __init__(self):
        man = json.load(open(CORPUS / "manifest.json"))
        self.rows = [json.loads(l) for l in open(CORPUS / "rows.jsonl")]
        self.T = man["tokens_total"]
        self.max_len = man["max_row_len"]
        self.tokens = np.load(CORPUS / "tokens.npy", mmap_mode="r")
        self.types = np.load(CORPUS / "types.npy", mmap_mode="r")
        self.valid = np.load(CORPUS / "valid.npy", mmap_mode="r")
        assert self.tokens.shape[0] == self.T

    def sl(self, r: int) -> slice:
        o = self.rows[r]["offset"]
        return slice(o, o + self.rows[r]["length"])


def mm(path: Path, dtype, shape, mode="r+"):
    """Open (creating if missing) a .npy memmap."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
    a = np.load(path, mmap_mode=mode)
    assert a.shape == tuple(shape) and a.dtype == np.dtype(dtype), (path, a.shape, shape, a.dtype, dtype)
    return a


def stream_files(T):
    return mm(D / "h.npy", np.uint16, (T, HC, DIM)), mm(D / "pre_mix.npy", np.float32, (T, HC))


# ----------------------------------------------------------------------------- shared state
def sources_for(L: int) -> dict:
    """Which earlier layers' published state layer L reads (kind -> source layer)."""
    r = CFG["compress_ratios"][L]
    if not r:
        return {}
    kv = max(s for s in CFG["kv_source_layers"] if s <= L)
    ix = max(s for s in CFG["index_source_layers"] if s <= L)
    need = {}
    if kv != L:
        need["kv"] = kv
    if L in CFG["index_source_layers"] and L not in CFG["kv_source_layers"]:
        need["ik"] = kv
    if L not in CFG["index_source_layers"]:
        need["topk"] = ix
    cs = CFG["candidate_source_layer"]
    if L in CFG["index_source_layers"] and 0 <= cs < L:
        need["cand"] = cs
    return need


def published_by(L: int) -> list[str]:
    out = []
    if CFG["compress_ratios"][L] and L in CFG["kv_source_layers"]:
        out += ["kv", "ik"]
    if CFG["compress_ratios"][L] and L in CFG["index_source_layers"]:
        out += ["topk"]
    if L == CFG["candidate_source_layer"]:
        out += ["cand"]
    return out


def last_consumer(kind: str, src: int) -> int:
    users = [L for L in range(N_LAYERS) if sources_for(L).get(kind) == src]
    return max(users) if users else src


def state_path(kind: str, src: int, r: int) -> Path:
    return D / "state" / f"{kind}_L{src}" / f"r{r:05d}.npy"


# ----------------------------------------------------------------------------- units
def unit_hash(a):
    import torch
    M = env.import_reference(a.kernels)
    from ops.lib.tok import Tok
    C = Corpus()
    args = env.model_args(M, max_batch_size=1, max_seq_len=C.max_len)
    layout = M.EngramLayout.from_args(args)
    hs = M.NgramHashState(args, layout, Tok(env.SRC))      # asserts the 99,092 compressed vocab
    out = mm(D / "engram" / "hash.npy", np.int32, (C.T, len(ENGRAM_LAYERS), N_COLS))
    led = env.Ledger(LEDGER)
    for r in range(len(C.rows)):
        u = f"hash/r{r}"
        if led.done(u):
            continue
        s = C.sl(r)
        ids = torch.from_numpy(C.tokens[s].astype(np.int64))[None]
        text = torch.from_numpy(C.types[s] < 0)[None]
        mask = None if bool(text.all()) else text          # Transformer: engram_mask = ~image_mask
        h = hs(ids, 0, mask)[0]                             # [n, 2, 24] int64
        assert int(h.max()) < 2 ** 31
        out[s] = h.numpy().astype(np.int32)
        if r % 16 == 15 or r == len(C.rows) - 1:
            out.flush()
            for rr in range(r - (r % 16), r + 1):
                led.finish(f"hash/r{rr}")
    out.flush()
    led.finish("hash", tokens=C.T)


def unit_engram(a):
    from ops.lib.engram_rows import tensor_memmap, gather_windowed
    L = a.layer
    li = ENGRAM_LAYERS.index(L)
    C = Corpus()
    shard = Path(a.engram_dir) / f"model-{47 + li:05d}-of-00048.safetensors"
    vals_t = tensor_memmap(shard, f"layers.{L}.engram.embed.weight")
    scl_t = tensor_memmap(shard, f"layers.{L}.engram.embed.scale")
    hashes = np.load(D / "engram" / "hash.npy", mmap_mode="r")[:, li, :]            # [T, 24]
    ids = np.ascontiguousarray(hashes).reshape(-1)
    led = env.Ledger(LEDGER)
    for name, table, width in (("vals", vals_t, EHD), ("scales", scl_t, scl_t.shape[1])):
        u = f"engram/L{L}/{name}"
        if led.done(u):
            continue
        out = mm(D / "engram" / f"L{L}_{name}.npy", np.uint8, (C.T * N_COLS, width))
        start = led.get(u, {}).get("next_window", 0)

        def on_window(w, out=out, u=u):
            out.flush()
            led.mark(u, next_window=w + 1)
        t0 = time.time()
        gather_windowed(table, ids, out, start_window=start, on_window=on_window)
        out.flush()
        led.finish(u, seconds=round(time.time() - t0, 1), shard=str(shard))


def unit_embed(a):
    import torch
    M = env.import_reference(a.kernels)
    from ops.lib import shards
    C = Corpus()
    env.assert_no_gpu_server(STAGE)
    idx = shards.ShardIndex()
    torch.set_default_dtype(torch.bfloat16)
    W = idx.get("embed.weight").cuda()
    H, PM = stream_files(C.T)
    led = env.Ledger(LEDGER)
    vis = None
    for r, row in enumerate(C.rows):
        u = f"embed/r{r}"
        if led.done(u):
            continue
        s = C.sl(r)
        ids = torch.from_numpy(C.tokens[s].astype(np.int64)).cuda()
        h = torch.nn.functional.embedding(ids, W)                      # [n, dim]
        if row["images"]:
            vis = vis or build_vision(M, idx)
            merge_images(M, vis, row, h, C.types[s])
        H[s] = h.unsqueeze(1).repeat(1, HC, 1).view(torch.int16).cpu().numpy().view(np.uint16)
        pm = np.zeros((h.size(0), HC), np.float32); pm[:, 0] = 1.0     # make_identity_pre_mix
        PM[s] = pm
        H.flush(); PM.flush()
        led.finish(u)
    led.finish("embed", tokens=C.T)


def build_vision(M, idx):
    import torch
    args = env.model_args(M, max_batch_size=1)
    from ops.lib import shards  # noqa: F401
    with torch.device("cuda"):
        vit, aligner = M.ViT(args), M.Aligner(args)
    for mod, pre in ((vit, "vision."), (aligner, "aligner.")):
        sd = {n[len(pre):]: idx.get(n) for n in idx.names(pre)}
        missing, unexpected = mod.load_state_dict(sd, strict=False)
        assert not missing and not unexpected, (pre, missing[:5], unexpected[:5])
        mod.eval()
    delim = {k: idx.get(k).cuda() for k in ("image_start", "image_end", "image_newline")}
    return {"args": args, "vit": vit, "aligner": aligner, **delim}


def merge_images(M, vis, row, h, types_np):
    """Transformer.merge_image_embeddings for one row, with a cross-check that s02's span layout
    equals what the reference load_image produces for the same file."""
    import torch
    from image_processor import IMAGE, IMAGE_END, IMAGE_NEW_LINE, IMAGE_START, load_image
    types = torch.from_numpy(types_np.astype(np.int64)).cuda()
    for im in row["images"]:
        patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = load_image({"url": im["path"]}, vis["args"])
        assert (n_llm_h, n_llm_w) == (im["n_llm_h"], im["n_llm_w"]), ("s02/s03 image grid mismatch", im)
        sl = slice(im["start"], im["start"] + im["n"])
        t = types[sl]
        span = h[sl]
        span[t == IMAGE_START] = vis["image_start"].to(h.dtype)
        span[t == IMAGE_END] = vis["image_end"].to(h.dtype)
        span[t == IMAGE_NEW_LINE] = vis["image_newline"].to(h.dtype)
        with torch.inference_mode():
            emb = vis["aligner"](vis["vit"](patches.cuda(), n_vit_h, n_vit_w), n_vit_h, n_vit_w)
        span[t == IMAGE] = emb.to(h.dtype)
        h[sl] = span


def unit_layer(a):
    import torch
    from ops.lib import memfence, shards
    from ops.lib.hooks import FFNStats, chunked_block, chunked_indexer, instrument_moe
    L = a.layer
    env.assert_no_gpu_server(STAGE)
    M = env.import_reference(a.kernels)
    if a.kernels == "torch":
        from ops.lib import torch_kernels
        torch_kernels.install(M, "torch")
    chunked_indexer(M)
    chunked_block(M, a.token_chunk)
    stats_of: dict = {}
    instrument_moe(M, stats_of)
    C = Corpus()
    args = env.model_args(M, max_batch_size=1, max_seq_len=C.max_len)
    torch.set_default_dtype(torch.bfloat16)
    torch.cuda.memory._set_allocator_settings("expandable_segments:True")
    idx = shards.ShardIndex()
    blk = shards.build_block(M, args, L, idx, "cuda", with_engram=True)
    idx.close()
    st = FFNStats(len(spec.BUCKETS), E, DIM, INTER, "cuda")
    stats_of[id(blk.ffn)] = st
    # model.py creates some tensors without a device (get_window_topk_idxs); generate.py runs with
    # cuda as the default device, so do the same for the forward
    torch.set_default_device("cuda")

    H, PM = stream_files(C.T)
    rid = mm(D / "routing" / f"L{L}_idx.npy", np.uint16, (C.T, K))
    rw = mm(D / "routing" / f"L{L}_w.npy", np.float16, (C.T, K))
    jdir = D / "journal" / f"L{L}"; jdir.mkdir(parents=True, exist_ok=True)
    ck = D / "bank" / f"L{L}.ckpt.pt"
    lst = env.Ledger(D / "bank" / f"L{L}.rows.json")       # per-layer applied-row ledger
    covered: set[int] = set()
    if ck.exists():
        blob = torch.load(ck, map_location="cuda")
        st.load(blob["state"]); covered = set(blob["rows"])
    # reconcile: re-apply covered-but-unapplied journals, drop stale ones
    for f in sorted(jdir.glob("r*.npz")):
        r = int(f.stem[1:])
        if r in covered and not lst.done(f"r{r}"):
            apply_journal(C, r, f, H, PM, rid, rw, L); lst.finish(f"r{r}"); f.unlink()
        elif r not in covered:
            f.unlink()
    for f in jdir.glob("*.tmp"):
        f.unlink()
    need = sources_for(L)
    pending: list[int] = []
    taps = L in CFG["dspark_target_layer_ids"] and a.taps != "none"
    t_start, n_done = time.time(), 0
    for r, row in enumerate(C.rows):
        if r in covered:
            continue
        s = C.sl(r)
        n = s.stop - s.start
        memfence.require(n * BYTES_PER_TOKEN_PEAK / 2**30 + 2.0, f"L{L} r{r} ({n} tok)")
        x = torch.from_numpy(np.ascontiguousarray(H[s])).cuda().view(torch.bfloat16)[None]   # [1, n, 4, dim]
        pm = torch.from_numpy(np.ascontiguousarray(PM[s])).cuda()[None]
        ty = C.types[s]
        image_mask = torch.from_numpy(ty >= 0).cuda()[None] if (ty >= 0).any() else None
        for kind, src in need.items():
            install_state(M, kind, src, r)
        valid = torch.from_numpy(C.valid[s].astype(bool)).cuda()
        if row["heldout"]:
            valid = torch.zeros_like(valid)
        st.start_row(torch.full((n,), row["bucket_id"], dtype=torch.long, device="cuda"), valid)
        with torch.inference_mode():
            if blk.engram is not None:
                x = run_engram(blk, x, L, s, image_mask, a.token_chunk)
            tap = x.mean(dim=2)[0] if taps and (a.taps == "all" or row["heldout"]) else None
            y, ffn_pre = blk(x, 0, pm, image_mask)
        ridx = torch.cat([i for i, _ in st.route]).cpu().numpy().astype(np.uint16)
        rwt = torch.cat([w for _, w in st.route]).cpu().numpy().astype(np.float16)
        assert ridx.shape == (n, K)
        payload = {"h": y[0].contiguous().view(torch.int16).cpu().numpy(), "pre_mix": ffn_pre[0].float().cpu().numpy(),
                   "ridx": ridx, "rw": rwt}
        for kind in published_by(L):
            payload["state_" + kind] = grab_state(M, blk, kind, n)
        if tap is not None:
            payload["tap"] = tap.contiguous().view(torch.int16).cpu().numpy()
        jf = jdir / f"r{r:05d}.npz"
        with open(jf.with_suffix(".tmp"), "wb") as fh:
            np.savez(fh, **payload)
            fh.flush(); os.fsync(fh.fileno())
        os.replace(jf.with_suffix(".tmp"), jf)
        pending.append(r)
        del x, y, ffn_pre, payload
        n_done += n
        if len(pending) >= a.ckpt_rows or r == len(C.rows) - 1:
            covered |= set(pending)
            tmp = ck.with_suffix(".tmp")
            torch.save({"state": {k: v.clone() for k, v in st.state().items()}, "rows": sorted(covered)}, tmp)
            os.replace(tmp, ck)
            for pr in pending:
                f = jdir / f"r{pr:05d}.npz"
                apply_journal(C, pr, f, H, PM, rid, rw, L); lst.finish(f"r{pr}"); f.unlink()
            pending = []
            el = time.time() - t_start
            env.log(f"L{L}: {len(covered)}/{len(C.rows)} rows, {n_done / max(el, 1e-9):.0f} tok/s, "
                    f"cuda peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", STAGE)
    finalize_layer(L, st, C)


def run_engram(blk, x, L, s, image_mask, chunk):
    """Engram over token chunks (it is per-token), with rows served from the s03 engram files."""
    import torch
    vals = np.load(D / "engram" / f"L{L}_vals.npy", mmap_mode="r")
    scl = np.load(D / "engram" / f"L{L}_scales.npy", mmap_mode="r")
    n = x.size(1)
    out = torch.empty_like(x)
    for c0 in range(0, n, chunk):
        c1 = min(n, c0 + chunk)
        a0, a1 = (s.start + c0) * N_COLS, (s.start + c1) * N_COLS
        v = torch.from_numpy(np.ascontiguousarray(vals[a0:a1])).cuda().view(torch.float8_e4m3fn)
        sc = torch.from_numpy(np.ascontiguousarray(scl[a0:a1])).cuda().view(torch.float8_e8m0fnu)
        blk.engram.embed.rows_fp8 = v.view(1, c1 - c0, N_COLS, EHD)
        blk.engram.embed.rows_scale = sc.view(1, c1 - c0, N_COLS, -1)
        dummy = torch.zeros(1, c1 - c0, N_COLS, dtype=torch.long, device=x.device)
        mask = None if image_mask is None else ~image_mask[:, c0:c1]
        out[:, c0:c1] = blk.engram(x[:, c0:c1], dummy, mask)
    blk.engram.embed.rows_fp8 = blk.engram.embed.rows_scale = None
    return out


def grab_state(M, blk, kind, n):
    import torch
    r = CFG["compress_ratios"][blk.layer_id]
    if kind == "kv":
        t = blk.attn.compress_kv_cache[0, : n // r]
    elif kind == "ik":
        t = blk.attn.indexer.k_cache[0, : n // r]
    elif kind == "topk":
        return M.shared_attn.topk_idxs[0].cpu().numpy()
    else:  # cand: None when the candidate mask is the identity at this length
        b = M.shared_attn.candidate_bits
        return np.zeros((0,), np.uint8) if b is None else b[0].cpu().numpy()
    return t.contiguous().view(torch.int16).cpu().numpy()


def install_state(M, kind, src, r):
    import torch
    a = np.load(state_path(kind, src, r))
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


def apply_journal(C, r, f, H, PM, rid, rw, L):
    j = np.load(f)
    s = C.sl(r)
    H[s] = j["h"].view(np.uint16)
    PM[s] = j["pre_mix"]
    rid[s] = j["ridx"]
    rw[s] = j["rw"]
    for k in j.files:
        if k.startswith("state_"):
            p = state_path(k[6:], L, r)
            p.parent.mkdir(parents=True, exist_ok=True)
            np.save(p, j[k])
        elif k == "tap":
            p = D / "taps" / f"L{L}" / f"r{r:05d}.npy"
            p.parent.mkdir(parents=True, exist_ok=True)
            np.save(p, j[k].view(np.uint16))
    H.flush(); PM.flush(); rid.flush(); rw.flush()


def finalize_layer(L, st, C):
    import torch
    out = {k: v.double().cpu().numpy() if v.dtype.is_floating_point else v.cpu().numpy() for k, v in st.state().items()}
    np.savez(D / "bank" / f"L{L}.npz", **out, buckets=np.array(spec.BUCKETS),
             rows=np.array(len(C.rows)), heldout_rows=np.array(sum(r["heldout"] for r in C.rows)))
    led = env.Ledger(LEDGER)
    led.finish(f"layer/L{L}", tokens_counted=float(out["n_tok_sh"][0]))
    # retire published state whose last consumer is this layer
    for src in range(L + 1):
        for kind in published_by(src):
            if last_consumer(kind, src) <= L and (D / "state" / f"{kind}_L{src}").exists():
                shutil.rmtree(D / "state" / f"{kind}_L{src}")
    (D / "bank" / f"L{L}.ckpt.pt").unlink(missing_ok=True)
    torch.cuda.empty_cache()


def unit_head(a):
    import torch
    from ops.lib import shards
    from ops.lib.hooks import topk_logprobs
    env.assert_no_gpu_server(STAGE)
    C = Corpus()
    idx = shards.ShardIndex()
    norm_w = idx.get("norm.weight").float().cuda()
    W = idx.get("head.weight").float().cuda()                     # fp32, as ParallelHead keeps it
    H, PM = stream_files(C.T)
    lp = mm(D / "teacher" / "top32_logprob.npy", np.float32, (C.T, TOPK_LOGITS))
    ti = mm(D / "teacher" / "top32_ids.npy", np.int32, (C.T, TOPK_LOGITS))
    lse = mm(D / "teacher" / "lse.npy", np.float32, (C.T,))
    nll = mm(D / "teacher" / "nll_next.npy", np.float32, (C.T,))
    led = env.Ledger(LEDGER)
    eps = CFG["norm_eps"]
    for r in range(len(C.rows)):
        u = f"head/r{r}"
        if led.done(u):
            continue
        s = C.sl(r)
        for c0 in range(s.start, s.stop, 1024):
            c1 = min(s.stop, c0 + 1024)
            x = torch.from_numpy(np.ascontiguousarray(H[c0:c1])).cuda().view(torch.bfloat16)
            pm = torch.from_numpy(np.ascontiguousarray(PM[c0:c1])).cuda()
            h = torch.sum(pm.unsqueeze(-1) * x.float(), dim=1).to(torch.bfloat16)       # Block.hc_pre
            hf = h.float()
            hn = (norm_w * (hf * torch.rsqrt(hf.square().mean(-1, keepdim=True) + eps))).to(torch.bfloat16)
            logits = torch.nn.functional.linear(hn.float(), W)
            v, i, z = topk_logprobs(logits, TOPK_LOGITS)
            lp[c0:c1], ti[c0:c1], lse[c0:c1] = v.cpu().numpy(), i.cpu().numpy(), z.cpu().numpy()
            nxt = np.full(c1 - c0, -1, np.int64)
            m = min(c1, s.stop - 1) - c0
            nxt[:m] = C.tokens[c0 + 1:c0 + 1 + m]
            have = torch.from_numpy(nxt >= 0).cuda()
            tgt = torch.from_numpy(np.maximum(nxt, 0)).cuda()
            nl = z - logits.gather(1, tgt[:, None])[:, 0]
            nll[c0:c1] = torch.where(have, nl, torch.full_like(nl, float("nan"))).cpu().numpy()
        for f in (lp, ti, lse, nll):
            f.flush()
        led.finish(u)
    led.finish("head", tokens=C.T)


# ----------------------------------------------------------------------------- orchestration
def units(a) -> list[tuple[str, list[str], bool]]:
    """(ledger key, argv, uses_gpu)."""
    u = [("hash", ["unit", "hash"], False)]
    u += [(f"engram/L{L}/scales", ["unit", "engram", "--layer", str(L)], False) for L in ENGRAM_LAYERS]
    u += [("embed", ["unit", "embed"], True)]
    u += [(f"layer/L{L}", ["unit", "layer", "--layer", str(L)], True) for L in range(N_LAYERS)]
    u += [("head", ["unit", "head"], True)]
    return u


def run(a):
    env.assert_no_gpu_server(STAGE)
    if not (CORPUS / "manifest.json").exists():
        raise SystemExit("no s02 corpus (manifest.json missing)")
    common = ["--kernels", a.kernels, "--engram-dir", a.engram_dir, "--ckpt-rows", str(a.ckpt_rows),
              "--token-chunk", str(a.token_chunk), "--taps", a.taps]
    for key, argv, gpu in units(a):
        led = env.Ledger(LEDGER)
        if led.done(key):
            continue
        if key.startswith("layer/") and a.fetch_shards:
            fetch_shard(int(key.split("L")[-1]))
        env.log(f"start {key}", STAGE)
        t0 = time.time()
        rc = subprocess.run([sys.executable, __file__, *argv, *common]).returncode
        if gpu:
            env.reclaim(STAGE, expect_gib=a.min_avail_gib)
        if rc != 0:
            raise SystemExit(f"{key} failed rc={rc}; fix and re-run `run` (it resumes)")
        env.log(f"done {key} in {time.time() - t0:.0f}s", STAGE)
        if key.startswith("layer/") and a.drop_shards:
            p = env.SRC / f"model-{int(key.split('L')[-1]) + 3:05d}-of-00048.safetensors"
            p.unlink(missing_ok=True)          # inside this project's own download; re-fetchable
    env.log("s03 complete", STAGE)


def fetch_shard(L: int):
    name = f"model-{L + 3:05d}-of-00048.safetensors"
    if (env.SRC / name).exists():
        return
    from huggingface_hub import hf_hub_download
    hf_hub_download("deepseek-ai/DeepSeek-V4.1-Flash-1017", name, revision=env.REVISION, local_dir=env.SRC)


def status(a):
    led = env.Ledger(LEDGER)
    for key, _, _ in units(a):
        print(f"{'done' if led.done(key) else '----'}  {key}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run", "status", "unit"])
    ap.add_argument("unit", nargs="?", choices=["hash", "engram", "embed", "layer", "head"])
    ap.add_argument("--layer", type=int)
    ap.add_argument("--kernels", choices=["tilelang", "torch"], default="tilelang")
    ap.add_argument("--engram-dir", default=str(env.SRC), help="directory holding shards 47/48")
    ap.add_argument("--ckpt-rows", type=int, default=8)
    ap.add_argument("--token-chunk", type=int, default=8192)
    ap.add_argument("--taps", choices=["none", "heldout", "all"], default="heldout",
                    help="DSpark target-layer taps (mean over hc copies at layers 37-39) for s11")
    ap.add_argument("--min-avail-gib", type=float, default=100.0, help="reclaim must restore at least this")
    ap.add_argument("--fetch-shards", action="store_true")
    ap.add_argument("--drop-shards", action="store_true")
    a = ap.parse_args()
    if a.cmd == "run":
        return run(a)
    if a.cmd == "status":
        return status(a)
    {"hash": unit_hash, "engram": unit_engram, "embed": unit_embed, "layer": unit_layer, "head": unit_head}[a.unit](a)


if __name__ == "__main__":
    main()
