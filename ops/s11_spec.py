#!/usr/bin/env python3
"""s11: the checkpoint's own speculative head (the 3 `mtp.*` DSpark blocks), refit to the REAP.

User direction (2026-10-19), pre-registered in ops/s10_gates.json "s11":
  * ALWAYS fine-tune. Measure three things: base (no speculation), the native head untuned, the
    tuned head.
  * Two residency arms of the same head (ops/lib/dspark.py): `inmodel` (preferred: the head's own
    routed experts REAP-pruned with the backbone's s04 criterion and ratio, K-quantized like the
    backbone's experts, resident in the Thor build) and `separate` (fallback: unpruned, source MXFP4
    experts). Resident GiB per arm, the head's bytes and KV counted against the reserve.
  * Train on the SERVED REAP's own hidden states (taps 37-39 from the winning Thor arm, not the
    teacher's: every quant build needs its own head), with self-distilled traces as targets: the
    served model's greedy continuations of corpus prompts (the corpus generator policy excludes
    Claude-written text; s02b's V4.1 self-traces are used as prompts where they exist).
  * Gates: acceptance at depths 1-3; end-to-end tok/s through the s08 server vs no speculation at
    128 and 8k context; identical greedy output. Ship the head of the arm that wins the tok/s gate;
    if none beats 1.0x, publish nothing and record the result in the cards (s12 reads decision.json).

Subcommands, in chain order (ops/chain2.sh):
  traces  --set eval|train    GPU  v41_server greedy continuations, exact ids (/v1/completions return_ids)
  tapdump --set eval|train    GPU  ops/s11_tapdump.py: the served arm's taps 37-39 over the traces
  bank                        GPU  the native head over train anchors, instrumented MoE -> REAP bank
  prune                       CPU  s04's winning criterion, the backbone's keep ratio -> mtp_keepset.json
  quant                       CPU  in-model experts: imatrix-RTN K-quant (Q3_K, Q2_K if Q3_K does not fit)
                                   -> heads/inmodel_native, fit.json (resident GiB per arm)
  accept  --arm A --head H    GPU  acceptance on the eval traces (exact: targets are generated tokens)
  refit   --arm A             GPU  fine-tune the dense head on the train traces -> heads/A_tuned
  curve                       GPU  diagnostic: K-wide forward cost (prefill chunk K) vs decode step
  e2e                         GPU  through the s08 decoder: no-spec vs each head at 128 and 8k, identity
  decide                      CPU  decision.json (ship rule)
  publish                     net  the shipped head only, after decide
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ops import corpus_spec as spec  # noqa: E402
from ops.lib import env, keepset, orchestrate, rowsets  # noqa: E402

STAGE = "s11"
OUT = env.WORK / "s11"
CORPUS = env.WORK / "corpus"
S08 = env.WORK / "s08"
ROWS = env.WORK / "engram_rows"
CUDA = env.REPO / "cuda" / "build"
CFG = json.load(open(env.INFERENCE / "config.json"))
N_STAGES = CFG["n_mtp_layers"]
E_HEAD = CFG["dspark_n_routed_experts"]
W = CFG["window_size"]
S = CFG["dspark_block_size"]
GIB = 1 << 30


def gates() -> dict:
    return json.load(open(env.OPS / "s10_gates.json"))["s11"]


# ----------------------------------------------------------------------------- the served arm
def served() -> dict:
    """The Thor arm s10 chose: build dir, keep-set, keep-set path (A192 or B128)."""
    g = json.load(open(env.WORK / "s10" / "gates.json"))
    if g["thor_arm"] == "A":
        kp = env.WORK / "s04" / "keepset.json"
    else:
        kp = env.WORK / "s04" / "keepset_128.json"
    keep, fp = keepset.load(kp)
    n = len(keep["0"])
    d = env.WORK / "thor" / f"{g['thor_arm']}{n}"
    tj = json.load(open(d / "thor.json"))
    keepset.require(tj["keepset_sha256"], fp, where=str(d))
    if not tj.get("complete"):
        raise SystemExit(f"{d} is not a complete Thor build")
    return {"arm": g["thor_arm"], "name": f"{g['thor_arm']}{n}", "dir": d, "keep": keep, "keep_path": kp, "fp": fp,
            "n": n, "thor": tj}


# ----------------------------------------------------------------------------- statistics
def accept_stats(acc: np.ndarray, rows: np.ndarray, depths=(1, 2, 3), draws: int = 2000) -> dict:
    """alpha_k = P(accepted >= k) and mean accepted drafts, with row-cluster bootstrap 95% CIs."""
    acc, rows = np.asarray(acc, np.float64), np.asarray(rows)
    ur, inv = np.unique(rows, return_inverse=True)
    rng = np.random.default_rng(0)
    pick = rng.integers(0, len(ur), (draws, len(ur)))

    def ci(v):
        s = np.bincount(inv, weights=v, minlength=len(ur))
        n = np.bincount(inv, minlength=len(ur)).astype(np.float64)
        est = s[pick].sum(1) / np.maximum(n[pick].sum(1), 1)
        return {"mean": float(v.mean()), "lo": float(np.percentile(est, 2.5)), "hi": float(np.percentile(est, 97.5))}

    return {"anchors": int(acc.size), "rows": int(len(ur)), "alpha": {str(k): ci((acc >= k).astype(np.float64)) for k in depths},
            "mean_accepted": ci(acc), "hist": np.bincount(acc.astype(int), minlength=S + 1).tolist()}


def choose(arms: dict, base: dict, g: dict) -> dict:
    """arms: {"arm/head": {"fits": bool, "e2e": {"tok_s_128", "tok_s_8k", "identical"} | None}}.
    A candidate passes when it fits, its greedy output is identical to no speculation, and it beats
    min_speedup at every context. Ship the passing in-model head if any (preferred: no separate
    model), else the passing separate head; within an arm the higher geometric-mean speedup."""
    rej, ok, spd = {}, [], {}
    measured = False
    for k, a in arms.items():
        why = []
        if not a.get("fits"):
            why.append("does not fit the Thor budget")
        e = a.get("e2e")
        if e is None:
            why.append("not measured end to end")
        else:
            measured = True
            s = {c: e[f"tok_s_{c}"] / base[f"tok_s_{c}"] for c in ("128", "8k")}
            spd[k] = s
            if not e.get("identical"):
                why.append("greedy output not identical to no speculation")
            for c, v in s.items():
                if not v > g["min_speedup"]:
                    why.append(f"speedup {v:.3f}x at {c} <= {g['min_speedup']}")
        (rej.__setitem__(k, why) if why else ok.append(k))
    gm = lambda k: float(np.sqrt(spd[k]["128"] * spd[k]["8k"]))
    ship = None
    for arm in ("inmodel", "separate"):
        c = sorted((k for k in ok if k.startswith(arm + "/")), key=gm, reverse=True)
        if c:
            ship = c[0]
            break
    status = "shipped" if ship else ("blocked" if not measured else "none")
    return {"status": status, "ship": ship, "speedups": spd, "rejected": rej, "passing": ok}


# ----------------------------------------------------------------------------- traces (GPU, s08 server)
def trace_prompts(which: str, n: int, seed: int) -> list[dict]:
    rows, man = rowsets.load_rows(CORPUS)
    pool = rowsets.split_rows(rows, man["spec_hash"], "calib" if which == "train" else "screen")
    types = np.load(CORPUS / "types.npy", mmap_mode="r")
    tokens = np.load(CORPUS / "tokens.npy", mmap_mode="r")
    lo, hi = gates()["traces"]["prompt_tokens"]
    ids = rowsets.subset(pool, n * hi, spec.TOKEN_TARGET)
    by = {r["row"]: r for r in rows}
    out = []
    rng = np.random.default_rng(seed)
    for rid in ids:
        r = by[rid]
        if r["length"] < lo + 8:
            continue
        k = int(rng.integers(lo, min(hi, r["length"] - 8) + 1))
        s = slice(r["offset"], r["offset"] + k)
        if (types[s] != -1).any():
            continue
        t = np.asarray(tokens[s]).astype(np.int64)
        out.append({"row": rid, "bucket": r["bucket"], "ids": t[1:].tolist() if t[0] == 0 else t.tolist()})
        if len(out) >= n:
            break
    return out


def cmd_traces(a):
    from transformers import AutoTokenizer
    srv = served()
    g = gates()["traces"]
    n, gen = (g["train_prompts"], g["gen_tokens_train"]) if a.set == "train" else (g["eval_prompts"], g["gen_tokens_eval"])
    f = OUT / f"traces_{a.set}.jsonl"
    done = {json.loads(l)["row"] for l in open(f)} if f.exists() else set()
    P = [p for p in trace_prompts(a.set, n, 11 if a.set == "train" else 12) if p["row"] not in done]
    if not P:
        env.log(f"traces {a.set}: complete ({len(done)})", STAGE)
        return
    env.assert_no_gpu_server(STAGE)
    tok = AutoTokenizer.from_pretrained(env.SRC)
    port = 18111
    log = open(env.LOGS / f"s11_server_{a.set}.log", "ab")
    sp = subprocess.Popen([str(CUDA / "v41_server"), "--ckpt", str(srv["dir"]), "--keepset", str(srv["keep_path"]),
                           "--engram", str(ROWS), "--port", str(port), "--host", "127.0.0.1"],
                          stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    import urllib.request

    def post(body):
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        return json.load(urllib.request.urlopen(req, timeout=3600))
    try:
        for _ in range(360):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5)
                break
            except Exception:
                if sp.poll() is not None:
                    raise SystemExit(f"v41_server exited rc={sp.returncode} (see {log.name})")
                time.sleep(10)
        with open(f, "a") as fh:
            for i, p in enumerate(P):
                text = tok.decode(p["ids"], skip_special_tokens=False)
                r = post({"prompt": text, "max_tokens": gen, "temperature": 0, "return_ids": True})
                if "completion_ids" not in r:
                    raise SystemExit("v41_server did not return ids (cuda/server/server.cpp return_ids)")
                fh.write(json.dumps({"row": p["row"], "bucket": p["bucket"], "prompt_ids": r["prompt_ids"],
                                     "completion_ids": r["completion_ids"], "timings": r.get("timings")}) + "\n")
                fh.flush()
                if i % 10 == 0:
                    env.log(f"traces {a.set}: {i + 1}/{len(P)} ({len(r['completion_ids'])} tok)", STAGE)
    finally:
        sp.terminate()
        try:
            sp.wait(60)
        except subprocess.TimeoutExpired:
            sp.kill()
    env.reclaim(STAGE, expect_gib=a.min_avail_gib)


def cmd_tapdump(a):
    orchestrate.run_unit(Path(__file__).with_name("s11_tapdump.py"), ["--set", a.set, "--kernels", a.kernels],
                         stage=STAGE, gpu=True, min_avail_gib=a.min_avail_gib)


class Taps:
    """s11_tapdump output: tokens [T], seq offsets, gen_start per seq, taps L37/38/39 bf16 [T, dim]."""

    def __init__(self, which: str):
        d = OUT / f"taps_{which}"
        self.meta = json.load(open(d / "meta.json"))
        if not self.meta.get("complete"):
            raise SystemExit(f"{d}: tapdump not complete")
        self.tokens = np.load(d / "tokens.npy")
        self.taps = [np.load(d / f"L{L}.npy", mmap_mode="r") for L in self.meta["taps"]]
        self.seqs = self.meta["seqs"]                     # [{offset, length, gen_start, row}]

    def anchors(self, stride: int = 1):
        from ops.lib import dspark
        out = []
        for si, s in enumerate(self.seqs):
            a = dspark.trace_anchors(s["length"], s["gen_start"], W, S)[::stride]
            out += [(si, int(p)) for p in a]
        return out

    def batch(self, items, device):
        """-> main_hidden [b, W, 3*dim] bf16, tok_in [b], targets [b, S] (all generated tokens)."""
        import torch
        mh, ti, tg = [], [], []
        for si, p in items:
            o = self.seqs[si]["offset"]
            win = slice(o + p - W + 1, o + p + 1)
            mh.append(np.concatenate([t[win] for t in self.taps], -1))
            ti.append(self.tokens[o + p + 1])
            tg.append(self.tokens[o + p + 2:o + p + 2 + S])
        mh = torch.from_numpy(np.stack(mh).view(np.int16)).to(device).view(torch.bfloat16)
        return mh, torch.tensor(ti, device=device), torch.from_numpy(np.stack(tg).astype(np.int64)).to(device)


# ----------------------------------------------------------------------------- head loading
def raw_source(i: int) -> dict:
    """mtp.i tensors from source shard 44+i, untransformed: {name: (dtype, np array)} in file order."""
    from ops.lib import st_io
    rd = st_io.Reader(env.SRC / f"model-{44 + i:05d}-of-00048.safetensors")
    return {n: (rd.dtype(n), np.asarray(rd.get(n))) for n in sorted(rd.names(), key=rd.offset) if n.startswith(f"mtp.{i}.")}


def to_torch(dt: str, arr: np.ndarray):
    """st_io storage array (bf16 as uint16, 8-bit floats as uint8) -> torch tensor of the real dtype."""
    import torch
    a = np.ascontiguousarray(arr)
    if dt == "BF16":
        return torch.from_numpy(a.view(np.int16)).view(torch.bfloat16)
    m = {"F8_E4M3": torch.float8_e4m3fn, "F8_E8M0": torch.float8_e8m0fnu, "I8": torch.int8}
    if dt in m:
        return torch.from_numpy(a.view(np.uint8)).view(m[dt])
    return torch.from_numpy(a)


def head_dir(arm: str, head: str) -> Path:
    return OUT / "heads" / f"{arm}_{head}"


def load_head(M, args, arm: str, head: str, device: str):
    """The drafter as served: source mtp.* blocks; tuned dense tensors from heads/<arm>_tuned;
    the in-model arm pruned to mtp_keepset.json with its K-quant experts dequantized."""
    import torch
    from ops.lib import dspark, shards, st_io
    raws = [raw_source(i) for i in range(N_STAGES)]
    if head == "tuned" or arm == "inmodel":
        hd = head_dir(arm, head)
        man = json.load(open(hd / "head.json"))
        for i in range(N_STAGES):
            rd = st_io.Reader(hd / f"mtp-{i}.safetensors")
            for n in rd.names():
                if n.startswith("__pad.") or ".ffn.experts." in n and n.endswith(".qweight"):
                    continue
                if n in raws[i] and not (".ffn.gate." in n and arm == "inmodel"):
                    raws[i][n] = (rd.dtype(n), np.asarray(rd.get(n)))
    sds = []
    for i in range(N_STAGES):
        sd = {k[len(f"mtp.{i}."):]: to_torch(dt, a) for k, (dt, a) in raws[i].items()}
        sds.append(dspark.mp1(sd))
    idx = shards.ShardIndex()
    d = dspark.build(M, args, sds, idx.get("embed.weight"), idx.get("head.weight"), device)
    idx.close()
    if arm == "inmodel":
        keep, fp = keepset.load(OUT / "mtp_keepset.json")
        keepset.require(man["mtp_keepset_sha256"], fp, where=str(hd))
        q = read_qexperts(head_dir("inmodel", "native"), keep, man["qtype"])
        dspark.apply_keep(M, d, keep, dequantize_experts(q, args))
        if head == "tuned":                      # the tuned router rows (already in keep order)
            for i, b in enumerate(d.blocks):
                rd = st_io.Reader(hd / f"mtp-{i}.safetensors")
                w = to_torch(rd.dtype(f"mtp.{i}.ffn.gate.weight"), np.asarray(rd.get(f"mtp.{i}.ffn.gate.weight")))
                with torch.no_grad():
                    b.ffn.gate.weight.copy_(w.to(b.ffn.gate.weight.dtype))
    return d, raws


# ----------------------------------------------------------------------------- in-model experts (K-quant)
def _wfloat(lin, tk) -> np.ndarray:
    import torch
    w = lin.weight.data
    if w.dtype == torch.float4_e2m1fn_x2:
        w = tk.dequant_fp4(w, lin.scale.data, 32)
    elif w.dtype == torch.float8_e4m3fn:
        w = tk.dequant_fp8(w, lin.scale.data)
    return w.float().cpu().numpy()


def quantize_experts(d, keep: dict, A: list[dict], qtype: str, tk) -> list[list[dict]]:
    """imatrix-weighted RTN into ggml K-quant blocks (the backbone's format, ops/lib/thor_format.md) for
    every kept expert of every head block. A[i]: the bank's per-expert column energies (A13, A2)."""
    from ops.lib import ggml_quants as G
    out = []
    for i, b in enumerate(d.blocks):
        blk = []
        for j in keep[str(i)]:
            e = b.ffn.experts[j]
            blk.append({p: G.quantize_rtn(qtype, _wfloat(getattr(e, p), tk), (A[i]["A2"] if p == "w2" else A[i]["A13"])[j])
                        for p in ("w1", "w2", "w3")} | {"qtype": qtype})
        out.append(blk)
    return out


def dequantize_experts(q: list[list[dict]], args) -> list[list[tuple]]:
    """K-quant experts -> bf16-ready float tensors (w1, w2, w3), the arm's served values."""
    import torch
    from ops.lib import ggml_quants as G
    dim, inter = args.dim, args.moe_inter_dim
    out = []
    for blk in q:
        rows = []
        for e in blk:
            f = lambda p, o, c: torch.from_numpy(G.dequant(e["qtype"], e[p]).reshape(o, c))
            rows.append((f("w1", inter, dim), f("w2", dim, inter), f("w3", inter, dim)))
        out.append(rows)
    return out


def read_qexperts(hd: Path, keep: dict, qtype: str) -> list[list[dict]]:
    from ops.lib import st_io
    out = []
    for i in range(N_STAGES):
        rd = st_io.Reader(hd / f"mtp-{i}.safetensors")
        st = {p: np.asarray(rd.get(f"mtp.{i}.ffn.experts.{p}.qweight")) for p in ("w1", "w2", "w3")}
        if st["w1"].shape[0] != len(keep[str(i)]):
            raise keepset.KeepsetMismatch(f"{hd}: mtp.{i} has {st['w1'].shape[0]} experts, keep-set {len(keep[str(i)])}")
        out.append([{p: st[p][n] for p in st} | {"qtype": qtype} for n in range(st["w1"].shape[0])])
    return out


def inmodel_items(i: int, dense: list, keep: list[int], q: list[dict], qtype: str) -> list[dict]:
    """One in-model head file: the dense tensors (routed-expert tensors dropped, router rows sliced to
    the keep-set unless already sliced) + the stacked K-quant experts (thor-v1 naming)."""
    pre = f"mtp.{i}."
    items = []
    sel = np.asarray(keep)
    for n, dt, a in dense:
        r = n[len(pre):]
        if r.startswith("ffn.experts."):
            continue
        if r in ("ffn.gate.weight", "ffn.gate.bias", "ffn.gate.bias_vl") and a.shape[0] != len(sel):   # not yet sliced
            a = a[sel]
        items.append({"name": n, "dtype": dt, "data": np.ascontiguousarray(a)})
    for p in ("w1", "w2", "w3"):
        items.append({"name": f"{pre}ffn.experts.{p}.qweight", "dtype": "U8", "qtype": qtype,
                      "data": np.stack([e[p] for e in q])})
    return items


def write_head(hd: Path, files: list[list[dict]], meta: dict) -> dict:
    from ops.lib import st_io, thor_format as TF
    hd.mkdir(parents=True, exist_ok=True)
    rec = {}
    for i, items in enumerate(files):
        f = hd / f"mtp-{i}.safetensors"
        st_io.write(f, [(x["name"], x["dtype"], x["data"]) for x in items],
                    metadata={"format": TF.FORMAT, "s11": json.dumps({k: v for k, v in meta.items() if k != "files"})})
        rec[f.name] = {"bytes": f.stat().st_size, "sha256": TF.sha256_file(f)}
    env.atomic_json(hd / "head.json", {**meta, "files": rec, "ts": env.now()})
    return rec


# ----------------------------------------------------------------------------- GPU units
def reference(kernels: str, device: str = "cuda"):
    import torch
    from ops.lib import torch_kernels
    M = env.import_reference(kernels)
    if kernels == "torch":
        torch_kernels.install(M, "torch")
    torch.set_default_dtype(torch.bfloat16)
    args = env.model_args(M, max_batch_size=gates()["train"]["batch"], max_seq_len=W + S + 8, temperature=0.0)
    return M, args, torch_kernels


def cmd_bank(a):
    import torch
    from ops.lib import dspark
    env.assert_no_gpu_server(STAGE)
    M, args, tk = reference(a.kernels)
    d, _ = load_head(M, args, "separate", "native", "cuda")
    banks = dspark.instrument(M, d)
    T = Taps("train")
    items = T.anchors(stride=a.stride)
    bs = gates()["train"]["batch"]
    with torch.inference_mode():
        for k in range(0, len(items), bs):
            dspark.step(d, *T.batch(items[k:k + bs], "cuda")[:2])
    (OUT / "bank").mkdir(parents=True, exist_ok=True)
    for i, bk in enumerate(banks):
        bk.save(OUT / "bank" / f"mtp{i}.npz")
    env.log(f"bank: {len(items)} anchors over {N_STAGES} head blocks", STAGE)


def cmd_prune(a):
    arb = json.load(open(env.WORK / "s04" / "arbitration.json"))
    win = arb["candidates"][arb["winners"]["A"]["name"]]
    crit = win["criterion"]
    if crit == "hope":
        crit = f"hope_{win.get('form') or 'sum'}"
    srv = served()
    n_keep = int(round(E_HEAD * srv["n"] / CFG["n_routed_experts"]))
    from ops.lib import dspark
    banks = [dict(np.load(OUT / "bank" / f"mtp{i}.npz")) for i in range(N_STAGES)]
    keep = dspark.select_keep(banks, crit, n_keep)
    fp = keepset.save(OUT / "mtp_keepset.json", keep, criterion=crit, n_keep_head=n_keep, base_keepset=srv["fp"],
                      note="keys are the mtp stages 0..2, not decoder layers")
    env.log(f"prune: {crit}, {n_keep}/{E_HEAD} experts per head block, keep-set {fp[:16]}", STAGE)


def bytes_of(items) -> int:
    return int(sum(np.asarray(x["data"] if isinstance(x, dict) else x[2]).nbytes for x in items))


def fit_report(srv: dict, inmodel_bytes: int, separate_bytes: int, g: dict) -> dict:
    """Resident GiB per arm: Thor build without its mtp files + head + head KV + main 1M-context KV
    + workspace, against the budget (s10_gates.json s11.fit)."""
    f = g["fit"]
    build = sum(r["bytes"] for n, r in srv["thor"]["files"].items() if not n.startswith("mtp-"))
    kv1m = f["main_kv_1m_gib_default"] * GIB
    c = S08 / f"ctx1m_{srv['name']}.json"
    if c.exists():
        j = json.load(open(c))
        kv1m = j.get("state_bytes", kv1m)
    head_kv = N_STAGES * W * CFG["head_dim"] * 4               # one serving sequence, fp32 window cache
    rep = {"budget_gib": f["budget_gib"], "build_gib": build / GIB, "main_kv_1m_gib": kv1m / GIB,
           "workspace_gib": f["workspace_gib"], "head_kv_gib": head_kv / GIB, "arms": {}}
    for arm, hb in (("inmodel", inmodel_bytes), ("separate", separate_bytes)):
        tot = build + hb + head_kv + kv1m + f["workspace_gib"] * GIB
        rep["arms"][arm] = {"head_gib": hb / GIB, "resident_gib": tot / GIB, "fits": tot <= f["budget_gib"] * GIB}
    rep["arms"]["base"] = {"head_gib": 0.0, "resident_gib": (build + kv1m + f["workspace_gib"] * GIB) / GIB,
                           "fits": build + kv1m + f["workspace_gib"] * GIB <= f["budget_gib"] * GIB}
    return rep


def cmd_quant(a):
    from ops.lib import dspark, torch_kernels as tk
    import torch
    srv = served()
    g = gates()
    keep, fp = keepset.load(OUT / "mtp_keepset.json")
    M = env.import_reference("torch")
    tk.install(M, "torch")
    torch.set_default_dtype(torch.bfloat16)
    args = env.model_args(M, max_batch_size=1, max_seq_len=W + S + 8)
    d, raws = load_head(M, args, "separate", "native", "cpu")
    A = [dict(np.load(OUT / "bank" / f"mtp{i}.npz")) for i in range(N_STAGES)]
    dense = dspark.export_dense(d, raws, trained=set())
    sep_bytes = sum(int(arr.nbytes) for r in raws for _, arr in r.values())
    chosen = None
    for qt in g["inmodel_qtypes"]:
        from ops.lib import ggml_quants as G
        eb = N_STAGES * len(keep["0"]) * (2 * CFG["moe_inter_dim"] * G.row_bytes(qt, CFG["dim"])
                                          + CFG["dim"] * G.row_bytes(qt, CFG["moe_inter_dim"]))
        db = sum(int(arr.nbytes) for i in range(N_STAGES) for n, _, arr in dense[i]
                 if not n[len(f"mtp.{i}."):].startswith("ffn.experts."))
        rep = fit_report(srv, eb + db, sep_bytes, g)
        if rep["arms"]["inmodel"]["fits"] or qt == g["inmodel_qtypes"][-1]:
            chosen = qt
            break
    q = quantize_experts(d, keep, A, chosen, tk)
    files = [inmodel_items(i, dense[i], keep[str(i)], q[i], chosen) for i in range(N_STAGES)]
    write_head(head_dir("inmodel", "native"), files, {"arm": "inmodel", "head": "native", "qtype": chosen,
                                                      "mtp_keepset_sha256": fp, "base_keepset_sha256": srv["fp"],
                                                      "served": srv["name"]})
    real = sum(bytes_of(f) for f in files)
    rep = fit_report(srv, real, sep_bytes, g)
    rep["inmodel_qtype"] = chosen
    env.atomic_json(OUT / "fit.json", rep)
    env.log(f"quant: in-model head {chosen}, {real / GIB:.2f} GiB; resident "
            + ", ".join(f"{k} {v['resident_gib']:.1f} GiB ({'fits' if v['fits'] else 'DOES NOT FIT'})"
                        for k, v in rep["arms"].items()), STAGE)


def cmd_accept(a):
    import torch
    from ops.lib import dspark
    env.assert_no_gpu_server(STAGE)
    M, args, tk = reference(a.kernels)
    d, _ = load_head(M, args, a.arm, a.head, "cuda")
    T = Taps("eval")
    items = T.anchors()
    acc, rws = [], []
    bs = gates()["train"]["batch"]
    with torch.inference_mode():
        for k in range(0, len(items), bs):
            it = items[k:k + bs]
            mh, ti, tg = T.batch(it, "cuda")
            _, dr, _ = dspark.step(d, mh, ti)
            acc.append(dspark.accept_len(dr.cpu().numpy(), tg.cpu().numpy()))
            rws += [T.seqs[si]["row"] for si, _ in it]
    st = accept_stats(np.concatenate(acc), np.asarray(rws), tuple(gates()["depths"]))
    st.update({"arm": a.arm, "head": a.head, "served": served()["name"], "kernels": a.kernels, "ts": env.now()})
    env.atomic_json(OUT / "accept" / f"{a.arm}_{a.head}.json", st)
    env.log(f"accept {a.arm}/{a.head}: mean {st['mean_accepted']['mean']:.3f}, "
            + " ".join(f"a{k} {v['mean']:.3f}" for k, v in st["alpha"].items()), STAGE)


def cmd_refit(a):
    import torch
    from ops.lib import dspark
    env.assert_no_gpu_server(STAGE)
    g = gates()["train"]
    torch.manual_seed(g["seed"])
    M, args, tk = reference("torch")                    # differentiable kernels (STE fake-quant)
    d, raws = load_head(M, args, a.arm, "native", "cuda")
    trained = dspark.make_trainable(d, tk)
    T = Taps("train")
    items = T.anchors()
    rng = np.random.default_rng(g["seed"])
    seqs = np.unique([si for si, _ in items])
    val_seqs = set(rng.choice(seqs, max(1, int(len(seqs) * g["val_frac"])), replace=False).tolist())
    tr = [x for x in items if x[0] not in val_seqs]
    va = [x for x in items if x[0] in val_seqs][: g["val_anchors"]]
    params = [p for b in d.blocks for p in b.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=g["lr"], weight_decay=0.0, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / g["warmup"]))
    log = env.Ledger(OUT / f"refit_{a.arm}.json")

    def val():
        for b in d.blocks:
            b.eval()
        acc = []
        with torch.no_grad():
            for k in range(0, len(va), g["batch"]):
                mh, ti, tg = T.batch(va[k:k + g["batch"]], "cuda")
                _, dr, _ = dspark.step(d, mh, ti)
                acc.append(dspark.accept_len(dr.cpu().numpy(), tg.cpu().numpy()))
        for b in d.blocks:
            b.train()
        return float(np.concatenate(acc).mean())

    best, best_state, bad = val(), None, 0
    log.mark("val/0", mean_accepted=best)
    env.log(f"refit {a.arm}: {sum(trained.values()) / 1e6:.1f}M trainable params, {len(tr)} train / {len(va)} val "
            f"anchors, native val accepted {best:.3f}", STAGE)
    for b in d.blocks:
        b.train()
    step = 0
    while step < g["steps"]:
        for k in rng.permutation(len(tr) // g["batch"]):
            mh, ti, tg = T.batch(tr[k * g["batch"]:(k + 1) * g["batch"]], "cuda")
            lg, _, cf = dspark.step(d, mh, ti, targets=tg)
            loss, per = dspark.loss_fn(lg, cf, tg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, g["clip"])
            opt.step()
            sched.step()
            step += 1
            if step % g["val_every"] == 0 or step == g["steps"]:
                v = val()
                log.mark(f"val/{step}", mean_accepted=v, loss=float(loss), ce=per.tolist())
                env.log(f"refit {a.arm} step {step}: loss {float(loss):.4f} val accepted {v:.3f} (best {best:.3f})", STAGE)
                if v > best:
                    best, bad = v, 0
                    best_state = {n: p.detach().clone() for b in d.blocks for n, p in [(f"{id(b)}.{n}", p) for n, p in b.named_parameters() if p.requires_grad]}
                else:
                    bad += 1
                    if bad >= g["early_stop_patience"]:
                        step = g["steps"]
            if step >= g["steps"]:
                break
    if best_state is not None:
        with torch.no_grad():
            for b in d.blocks:
                for n, p in b.named_parameters():
                    if p.requires_grad:
                        p.copy_(best_state[f"{id(b)}.{n}"])
    dense = dspark.export_dense(d, raws, trained=set(trained))
    meta = {"arm": a.arm, "head": "tuned", "trained_params": int(sum(trained.values())), "val_mean_accepted": best,
            "native_val_mean_accepted": log.get("val/0")["mean_accepted"], "steps": step, "served": served()["name"],
            "base_keepset_sha256": served()["fp"], "train_targets": "the served model's greedy continuations (s11 traces)"}
    if a.arm == "inmodel":
        keep, fp = keepset.load(OUT / "mtp_keepset.json")
        man = json.load(open(head_dir("inmodel", "native") / "head.json"))
        q = read_qexperts(head_dir("inmodel", "native"), keep, man["qtype"])
        files = [inmodel_items(i, dense[i], keep[str(i)], q[i], man["qtype"]) for i in range(N_STAGES)]
        meta.update(qtype=man["qtype"], mtp_keepset_sha256=fp)
    else:
        files = [[{"name": n, "dtype": dt, "data": arr} for n, dt, arr in dense[i]] for i in range(N_STAGES)]
    write_head(head_dir(a.arm, "tuned"), files, meta)
    env.log(f"refit {a.arm}: exported, val accepted {meta['native_val_mean_accepted']:.3f} -> {best:.3f}", STAGE)


def run_decode(srv, extra: list[str], out: Path, ids: Path | None, chat: str | None, n: int, seqmax: int) -> dict:
    cmd = [str(CUDA / "v41_decode"), "--ckpt", str(srv["dir"]), "--keepset", str(srv["keep_path"]), "--engram", str(ROWS),
           "--n", str(n), "--seqmax", str(seqmax), "--json", str(out), *extra]
    cmd += ["--ids-file", str(ids)] if ids else ["--chat", chat]
    with open(env.LOGS / f"s11_{out.stem}.log", "w") as lf:
        rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True).returncode
    env.reclaim(STAGE, expect_gib=100)
    if rc != 0:
        raise SystemExit(f"v41_decode {' '.join(extra)} failed rc={rc} (see {env.LOGS / f's11_{out.stem}.log'})")
    return json.load(open(out))


def spec_capable() -> bool:
    """The s08 decoder must expose a draft path: `--draft DIR` (an s11 head dir) and `--draft-depth K`.
    The V4.1 engine was written plain-autoregressive (cuda/src/v41_engine.cu)."""
    b = CUDA / "v41_decode"
    return b.exists() and b"--draft" in b.read_bytes()


def cmd_curve(a):
    srv = served()
    d = OUT / "curve"
    d.mkdir(parents=True, exist_ok=True)
    ids = d / "ids512.txt"
    if not ids.exists():
        T = Taps("eval")
        ids.write_text(" ".join(map(str, T.tokens[:512].tolist())))
    res = {}
    for K in (1, 2, 3, 4, 6, 8):
        f = d / f"chunk{K}.json"
        j = json.load(open(f)) if f.exists() else run_decode(srv, ["--chunk", str(K)], f, ids, None, 32, 1024)
        res[K] = {"prefill_ms": j["prefill_ms"], "steps": -(-512 // K), "decode_tok_s": j["decode_tok_s"]}
    t1 = res[1]["prefill_ms"] / res[1]["steps"]
    dec = 1000.0 / float(np.median([r["decode_tok_s"] for r in res.values()]))
    for K, r in res.items():
        tk_ = r["prefill_ms"] / r["steps"]
        r["vs_chunk1"], r["vs_decode"] = tk_ / t1, tk_ / dec
        r["perfect_drafter_bound"] = K / r["vs_chunk1"]
    env.atomic_json(d / "curve.json", {"served": srv["name"], "curve": res, "decode_ms": dec, "ts": env.now()})
    env.log("curve: " + ", ".join(f"K{K} {r['vs_chunk1']:.2f}x (bound {r['perfect_drafter_bound']:.2f})" for K, r in res.items()), STAGE)


def cmd_e2e(a):
    srv = served()
    g = gates()
    d = OUT / "e2e"
    d.mkdir(parents=True, exist_ok=True)
    chat = "Write a short paragraph about the history of the printing press."
    ids8k = S08 / "ids8k.txt"
    if not ids8k.exists():
        raise SystemExit(f"{ids8k} missing (s08 decode step writes it)")
    base = {}
    for c, (ids, ch, n, sm) in {"128": (None, chat, 256, 4096), "8k": (ids8k, None, 128, 16384)}.items():
        f = d / f"base_{c}.json"
        base[c] = json.load(open(f)) if f.exists() else run_decode(srv, [], f, ids, ch, n, sm)
    env.atomic_json(d / "base.json", {"tok_s_128": base["128"]["decode_tok_s"], "tok_s_8k": base["8k"]["decode_tok_s"]})
    if not spec_capable():
        msg = ("BLOCKED: the s08 decoder has no draft path (cuda/build/v41_decode lacks --draft; the V4.1 engine is "
               "plain autoregressive). Acceptance is measured offline; the end-to-end gate needs the DSpark verify "
               "path in cuda/ (draft head forward + K-wide verify + rollback).")
        env.atomic_json(d / "status.json", {"status": "blocked", "why": msg, "ts": env.now()})
        env.log(msg, STAGE, "WARN")
        return
    for arm in ("inmodel", "separate"):
        for head in ("native", "tuned"):
            hd = head_dir(arm, head)
            if arm == "separate" and head == "native":
                hd = OUT / "heads" / "separate_native"
                if not hd.exists():
                    from ops.lib import st_io  # the source head verbatim, so the decoder reads one format
                    write_head(hd, [[{"name": n, "dtype": dt, "data": arr} for n, (dt, arr) in raw_source(i).items()]
                                    for i in range(N_STAGES)], {"arm": "separate", "head": "native"})
            res = {}
            for c, (ids, ch, n, sm) in {"128": (None, chat, 256, 4096), "8k": (ids8k, None, 128, 16384)}.items():
                best = None
                for K in g["e2e_depths"]:
                    f = d / f"{arm}_{head}_{c}_K{K}.json"
                    j = json.load(open(f)) if f.exists() else run_decode(
                        srv, ["--draft", str(hd), "--draft-depth", str(K)], f, ids, ch, n, sm)
                    j["identical"] = j["ids"] == base[c]["ids"]
                    if best is None or j["decode_tok_s"] > best["decode_tok_s"]:
                        best = {**j, "K": K}
                res[c] = best
            env.atomic_json(d / f"{arm}_{head}.json", {"tok_s_128": res["128"]["decode_tok_s"], "tok_s_8k": res["8k"]["decode_tok_s"],
                                                       "K_128": res["128"]["K"], "K_8k": res["8k"]["K"],
                                                       "identical": bool(res["128"]["identical"] and res["8k"]["identical"])})
    env.atomic_json(d / "status.json", {"status": "measured", "ts": env.now()})


def cmd_decide(a):
    g = gates()
    fit = json.load(open(OUT / "fit.json"))
    e2e_dir = OUT / "e2e"
    status = json.load(open(e2e_dir / "status.json")) if (e2e_dir / "status.json").exists() else {"status": "not run"}
    base = json.load(open(e2e_dir / "base.json")) if (e2e_dir / "base.json").exists() else None
    arms, acc = {}, {}
    for arm in ("inmodel", "separate"):
        for head in ("native", "tuned"):
            k = f"{arm}/{head}"
            f = e2e_dir / f"{arm}_{head}.json"
            arms[k] = {"fits": fit["arms"][arm]["fits"], "e2e": json.load(open(f)) if f.exists() else None}
            af = OUT / "accept" / f"{arm}_{head}.json"
            acc[k] = json.load(open(af)) if af.exists() else None
    dec = choose(arms, base or {"tok_s_128": 1.0, "tok_s_8k": 1.0}, g)
    if status.get("status") == "blocked":
        dec["status"], dec["why"] = "blocked", status["why"]
    curve = OUT / "curve" / "curve.json"
    out = {**dec, "base": base, "fit": fit, "acceptance": acc, "curve": json.load(open(curve)) if curve.exists() else None,
           "served": served()["name"], "gates": g, "ts": env.now()}
    env.atomic_json(OUT / "decision.json", out)
    env.log(f"decide: {dec['status']} ship={dec['ship']}", STAGE)


def cmd_publish(a):
    dec = json.load(open(OUT / "decision.json"))
    if dec["status"] != "shipped" or not dec["ship"]:
        env.log(f"publish: nothing to ship ({dec['status']}); the result is recorded in the cards via s12", STAGE)
        return
    if not json.load(open(env.WORK / "s10" / "gates.json"))["thor"]:
        env.log("publish: the served Thor build failed its s10 release gate; its head is not published", STAGE, "WARN")
        return
    arm, head = dec["ship"].split("/")
    hd = head_dir(arm, head) if not (arm == "separate" and head == "native") else OUT / "heads" / "separate_native"
    os.environ.pop("HF_XET_HIGH_PERFORMANCE", None)
    from huggingface_hub import HfApi
    h = HfApi()
    srv = served()
    repo = f"{h.whoami()['name']}/DeepSeek-V4.1-Flash-1017-REAP{srv['n']}-DSpark-{arm}"
    h.create_repo(repo, private=True, exist_ok=True, repo_type="model")
    for f in sorted(hd.iterdir()):
        h.upload_file(path_or_fileobj=str(f), path_in_repo=f.name, repo_id=repo, repo_type="model")
    card = OUT / "README.md"
    card.write_text(head_card(dec, arm, head, repo))
    h.upload_file(path_or_fileobj=str(card), path_in_repo="README.md", repo_id=repo)
    h.upload_file(path_or_fileobj=str(env.SRC / "LICENSE"), path_in_repo="LICENSE", repo_id=repo)
    if hasattr(h, "update_repo_settings"):
        h.update_repo_settings(repo_id=repo, private=False)
    else:
        h.update_repo_visibility(repo_id=repo, private=False)
    env.atomic_json(OUT / "published.json", {"repo": repo, "ship": dec["ship"], "ts": env.now()})
    env.log(f"publish: {repo} is PUBLIC", STAGE)


def summary(dec: dict) -> str:
    """One paragraph for the model cards, every number from decision.json."""
    acc = dec.get("acceptance") or {}
    parts = []
    for k, v in acc.items():
        if v:
            parts.append(f"{k}: mean accepted {v['mean_accepted']['mean']:.2f} of {S}, "
                         + ", ".join(f"depth {d} {x['mean']:.3f}" for d, x in v["alpha"].items()))
    fit = dec.get("fit", {}).get("arms", {})
    res = ", ".join(f"{k} {v['resident_gib']:.1f} GiB" for k, v in fit.items())
    if dec["status"] == "shipped":
        sp = dec["speedups"][dec["ship"]]
        head = (f"The fine-tuned speculative head ({dec['ship']}) passed its pre-registered gates: "
                f"{sp['128']:.2f}x at 128 tokens of context and {sp['8k']:.2f}x at 8k, greedy output identical.")
    elif dec["status"] == "blocked":
        head = ("Speculative decoding is not shipped: the end-to-end gate could not be measured because the CUDA "
                "server has no draft path yet. Offline acceptance on the served model's own greedy traces:")
    else:
        head = "Speculative decoding is not shipped: no head beat no-speculation decoding at every context."
    return " ".join([head, "; ".join(parts) + "." if parts else "", f"Resident memory per arm: {res}." if res else ""]).strip()


def head_card(dec: dict, arm: str, head: str, repo: str) -> str:
    srv = served()
    return (f"---\nlicense: mit\nbase_model: deepseek-ai/DeepSeek-V4.1-Flash-1017\ntags: [deepseek, dspark, speculative-decoding, reap]\n---\n\n"
            f"# {repo.split('/')[-1]}\n\nThe `mtp.*` DSpark head of DeepSeek-V4.1-Flash-1017, {'fine-tuned' if head == 'tuned' else 'unchanged'} "
            f"for the {srv['name']} REAP build (keep-set `{srv['fp']}`). {'Its routed experts are REAP-pruned and K-quantized to live inside the Thor build.' if arm == 'inmodel' else 'Its experts are the unpruned source MXFP4 experts.'}\n\n"
            f"{summary(dec)}\n\nIt is trained on that build's own hidden states and on its own greedy continuations; "
            f"a head is specific to the quantized build it was fitted to.\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["traces", "tapdump", "bank", "prune", "quant", "accept", "refit", "curve", "e2e",
                                    "decide", "publish"])
    ap.add_argument("--set", choices=["eval", "train"])
    ap.add_argument("--arm", choices=["inmodel", "separate"])
    ap.add_argument("--head", choices=["native", "tuned"])
    ap.add_argument("--kernels", choices=["tilelang", "torch"], default="tilelang")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--min-avail-gib", type=float, default=100.0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"traces": cmd_traces, "tapdump": cmd_tapdump, "bank": cmd_bank, "prune": cmd_prune, "quant": cmd_quant,
     "accept": cmd_accept, "refit": cmd_refit, "curve": cmd_curve, "e2e": cmd_e2e, "decide": cmd_decide,
     "publish": cmd_publish}[a.cmd](a)


if __name__ == "__main__":
    main()
