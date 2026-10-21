#!/usr/bin/env python3
"""s06: measured per-layer heal gain by router replay over the survivors (GLM v2 recipe), plus the
post-prune activation bank the quantizers need. GPU, one two-phase unit per layer.

WHY A GAIN. Pruning changes the top-6 support, and norm_topk_prob renormalises the weights over
whatever survives, so a surviving expert's weight after pruning is strictly larger than before -
by an amount that depends, per token, on which competitors died. The routed sum comes out
inflated. The heal is ONE scalar per layer on the routed sum (GLM: the per-expert least-squares
heal lowered reconstruction residual in 41/42 layers and still LOST top-1 by 0.00556, z=11.8, so
there is no per-expert option here), and it must be MEASURED on the final keep-set: the
first-moment derivation was 30.8% off on GLM, and a stale gain is fatal (keep-set fingerprint).

PER LAYER, phase A (lib/replay two-phase unit, calibration rows only, never held-out):
  on the same FFN input x, route twice - pre (all 384) and post (survivors, bias -inf) - and
  * replay   (GLM heal_refit, the shipped v2 estimator): e = sum_k w_k^2 * m_k^2 with m the s03
             bank's mean ||f|| per expert; gain = 1 / mean_tok sqrt(e_post / e_pre)
  * direct   y_pre / y_post = the actual routed sums (all 384 experts are resident), gain =
             1 / mean_tok ||y_post|| / ||y_pre||
  * ls       sum <y_post, y_pre> / sum ||y_post||^2
  and bank, per kept expert, the post-prune activation statistics (input amax of w1|w3, w2-input
  amax gated and ungated, per-column sum x^2 for both, token count): NVFP4 input scales and the
  imatrix for the pruned model come from the model they will run in.
phase B: the layer runs with gain = replay estimate and the stream advances, so layer L+1 is
measured on healed inputs (sequential, unlike GLM's offline replay of the teacher's own scores).

The gain is applied only if the s10 `heal` panel adopts it (paired arm on the screen split).
The MXFP4 master ships unhealed regardless (E8M0 cannot carry it).

Outputs ($V41_WORK/s06): heal_gain.json {keepset_sha256, gains{L}, estimators{L}}, bank/L<L>.npz,
replay/ (run dir; streams dropped at the end).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ops import corpus_spec as spec  # noqa: E402
from ops.lib import env, keepset, orchestrate, rowsets  # noqa: E402

STAGE = "s06"
OUT = env.WORK / "s06"
S03 = env.WORK / "s03"
CORPUS = env.WORK / "corpus"
CFG = json.load(open(env.INFERENCE / "config.json"))
N_LAYERS, E = CFG["n_layers"], CFG["n_routed_experts"]


class Measure:
    def __init__(self, run, L, keep, m_norm, dim, inter):
        import torch
        self.run, self.L = run, L
        self.keep = torch.as_tensor(sorted(keep), device="cuda")
        self.kpos = {int(e): j for j, e in enumerate(sorted(keep))}
        self.m = torch.as_tensor(m_norm, dtype=torch.float32, device="cuda")
        z = lambda *s: torch.zeros(s, dtype=torch.float64, device="cuda")   # z() -> 0-dim scalar (zeros(*()) raises)
        n = len(keep)
        self.acc = {"replay_ratio": z(), "direct_ratio": z(), "ls_num": z(), "ls_den": z(), "tokens": z(),
                    "mass_pre": z(), "mass_post": z(), "slots_hit": z(),
                    "amax_in": z(n), "amax_mid": z(n), "amax_mid_ungated": z(n), "x2_in": z(n, dim),
                    "x2_mid": z(n, inter), "n_tok": z(n)}

    def __call__(self, blk, h, im, r, s0):
        import torch
        from ops.lib.replay import expert_parts
        g = blk.ffn.gate
        x = h.view(-1, h.size(-1))
        n = x.size(0)
        gs = self.run.gsl(r)
        valid = torch.from_numpy(np.asarray(self.run.C.valid[gs.start + s0:gs.start + s0 + n]).astype(bool)).cuda()
        imf = None if im is None else im.flatten()
        arm_bias, arm_vl = g.bias.data, (None if g.bias_vl is None else g.bias_vl.data)
        g.bias.data = g._bias0
        if g.bias_vl is not None:
            g.bias_vl.data = g._bias_vl0
        w_pre, i_pre = g(x, imf)
        g.bias.data = arm_bias
        if g.bias_vl is not None:
            g.bias_vl.data = arm_vl
        w_post, i_post = g(x, imf)
        A = self.acc
        e_pre = (w_pre.float() ** 2 * self.m[i_pre] ** 2).sum(-1)
        e_post = (w_post.float() ** 2 * self.m[i_post] ** 2).sum(-1)
        y_pre = torch.zeros(n, x.size(1), dtype=torch.float32, device="cuda")
        y_post = torch.zeros_like(y_pre)
        experts = blk.ffn.experts
        for i in torch.unique(i_pre).tolist():
            idx, top = torch.where(i_pre == i)
            y_pre[idx] += experts[i](x[idx], w_pre[idx, top, None]).float()
        for i in torch.unique(i_post).tolist():
            idx, top = torch.where(i_post == i)
            gi = w_post[idx, top, None]
            mid, out = expert_parts(experts[i], x[idx], gi)
            y_post[idx] += out.float()
            kv = valid[idx]
            if not bool(kv.any()):
                continue
            j = self.kpos[i]
            xk, mk, gk = x[idx][kv].float(), mid[kv].float(), gi[kv].float()
            A["amax_in"][j] = torch.maximum(A["amax_in"][j], xk.abs().amax().double())
            A["amax_mid"][j] = torch.maximum(A["amax_mid"][j], mk.abs().amax().double())
            A["amax_mid_ungated"][j] = torch.maximum(A["amax_mid_ungated"][j], (mk / gk.clamp_min(1e-20)).abs().amax().double())
            A["x2_in"][j] += (xk * xk).sum(0).double()
            A["x2_mid"][j] += (mk * mk).sum(0).double()
            A["n_tok"][j] += xk.size(0)
        v = valid
        A["replay_ratio"] += (e_post[v].clamp_min(1e-20) / e_pre[v].clamp_min(1e-20)).sqrt().double().sum()
        npre, npost = y_pre[v].norm(dim=-1), y_post[v].norm(dim=-1)
        A["direct_ratio"] += (npost / npre.clamp_min(1e-20)).double().sum()
        A["ls_num"] += (y_post[v] * y_pre[v]).sum().double()
        A["ls_den"] += (y_post[v] * y_post[v]).sum().double()
        A["tokens"] += v.sum().double()
        A["mass_pre"] += w_pre[v].float().sum().double()
        A["mass_post"] += w_post[v].float().sum().double()
        A["slots_hit"] += (~torch.isin(i_pre[v], self.keep)).sum().double()

    def finish(self):
        out = {k: v.cpu().numpy() for k, v in self.acc.items()}
        (OUT / "measure").mkdir(parents=True, exist_ok=True)
        tmp = OUT / "measure" / f"L{self.L}.tmp.npz"
        np.savez(tmp, **out, keep=self.keep.cpu().numpy())
        tmp.replace(OUT / "measure" / f"L{self.L}.npz")


def estimators(z) -> dict:
    t = float(z["tokens"])
    rr, dr = float(z["replay_ratio"]) / t, float(z["direct_ratio"]) / t
    return {"replay": 1.0 / rr, "direct": 1.0 / dr, "ls": float(z["ls_num"]) / float(z["ls_den"]),
            "inflation_replay": rr, "inflation_direct": dr, "tokens": t,
            "gate_mass_pre": float(z["mass_pre"]) / t, "gate_mass_post": float(z["mass_post"]) / t,
            "slots_hit_rate": float(z["slots_hit"]) / (6 * t)}


def unit_layer(a):
    import torch  # noqa: F401
    from ops.lib import replay
    run = replay.Run(OUT / "replay")
    keep, fp = keepset.load(Path(a.keepset))
    L = a.layer
    bank = np.load(S03 / "bank" / f"L{L}.npz")
    m_norm = bank["nrm"].sum(0) / np.maximum(bank["cnt"].sum(0), 1)
    meas = Measure(run, L, keep[str(L)], m_norm, CFG["dim"], CFG["moe_inter_dim"])

    def decide(M, blk, args, run_, cache):
        z = np.load(OUT / "measure" / f"L{L}.npz")
        est = estimators(z)
        return replay.ArmLayer(E, keep[str(L)], est["replay"])

    replay.unit_layer_two_phase(run, L, a.kernels, a.token_chunk, "heal", measure=meas, decide=decide,
                                cache_ffn_in=False)


def finalize(keep, fp, run):
    gains, est = {}, {}
    (OUT / "bank").mkdir(parents=True, exist_ok=True)
    for L in range(N_LAYERS):
        z = np.load(OUT / "measure" / f"L{L}.npz")
        if not np.array_equal(z["keep"], np.asarray(keep[str(L)])):
            raise keepset.KeepsetMismatch(f"s06 measure L{L} was taken on a different keep-set")
        est[str(L)] = estimators(z)
        gains[str(L)] = est[str(L)]["replay"]
        np.savez(OUT / "bank" / f"L{L}.npz", **{k: z[k] for k in ("amax_in", "amax_mid", "amax_mid_ungated",
                                                                     "x2_in", "x2_mid", "n_tok", "keep")})
    rows, man = rowsets.load_rows(CORPUS)
    env.atomic_json(OUT / "heal_gain.json", {
        "keepset_sha256": fp, "estimator": "replay", "gains": gains, "estimators": est,
        "rows_hash": rowsets.rows_hash(run.rows, man["spec_hash"]), "tokens": run.T,
        "note": "one scalar per layer on the routed-expert sum; adopted only if the s10 heal panel passes",
        "ts": env.now()})
    g = np.array(list(gains.values()))
    env.log(f"heal gains: median {np.median(g):.4f} range {g.min():.4f}-{g.max():.4f}", STAGE)


def run_all(a):
    from ops.lib import replay
    keep, fp = keepset.load(Path(a.keepset))
    rows, man = rowsets.load_rows(CORPUS)
    calib = rowsets.split_rows(rows, man["spec_hash"], "calib")
    ids = rowsets.subset(calib, a.tokens, spec.TOKEN_TARGET)
    run = replay.prepare(OUT / "replay", ids, [{"name": "heal", "keep": keep}], man["spec_hash"])
    keepset.require(run.arm("heal")["keep_sha256"], fp, where="s06 run dir")
    me = Path(__file__)
    common = ["--kernels", a.kernels, "--token-chunk", str(a.token_chunk), "--keepset", a.keepset]
    if not run.led.done("_embed"):
        orchestrate.run_unit(me, ["unit", "embed", *common], stage=STAGE, gpu=True, min_avail_gib=a.min_avail_gib)
    for L in range(N_LAYERS):
        if not replay.Run(OUT / "replay").led.done(f"heal/L{L}"):
            orchestrate.run_unit(me, ["unit", "layer", "--layer", str(L), *common], stage=STAGE, gpu=True,
                                 min_avail_gib=a.min_avail_gib)
    run = replay.Run(OUT / "replay")
    finalize(keep, fp, run)
    replay.drop_streams(run, "heal")
    for p in ("h.npy", "pre_mix.npy"):
        (run.root / "_embed" / p).unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run", "unit"])
    ap.add_argument("unit", nargs="?", choices=["embed", "layer"])
    ap.add_argument("--layer", type=int)
    ap.add_argument("--keepset", default=str(env.WORK / "s04" / "keepset.json"))
    ap.add_argument("--kernels", choices=["tilelang", "torch"], default="tilelang")
    ap.add_argument("--token-chunk", type=int, default=8192)
    ap.add_argument("--tokens", type=int, default=262_144, help="calibration tokens: the gain converges on one chunk, but the "
                    "post-prune bank feeds s07 input scales and the imatrix (131k left 6 kept experts at L39 under 32 tokens)")
    ap.add_argument("--min-avail-gib", type=float, default=100.0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.cmd == "run":
        env.assert_no_gpu_server(STAGE)
        return run_all(a)
    from ops.lib import replay
    if a.unit == "embed":
        replay.unit_embed(replay.Run(OUT / "replay"), a.kernels)
    else:
        unit_layer(a)


if __name__ == "__main__":
    main()
