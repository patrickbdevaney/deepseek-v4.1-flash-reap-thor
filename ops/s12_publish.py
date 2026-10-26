#!/usr/bin/env python3
"""s12: model cards rendered from the s10 artifacts (never literals), then each repo made public
only if its pre-registered release gate passed.

Artifacts:
  mxfp4   the s05 master            ($V41_WORK/s05/manifest_REAP<n>.json)   gate: gates.json mxfp4_master
  nvfp4   the s07 NVFP4 repack      ($V41_WORK/s07/manifest_nvfp4_REAP<n>.json) gate: nvfp4
  gguf    the s09 GGUF files        ($V41_WORK/s09/manifest.json, written by the llama.cpp/GGUF
                                     owner: {"repo"?, "files": [{"path", "sha256"}], "dense_precision"?})
          gate: $V41_WORK/s09/gates.json (ops/s09_gate.py, s10_gates.json "gguf"): at least one
          level passed; the manifest lists only passing levels. The card is filled from each level's
          teacher_kld.json and the writer's dense precision (Q8_0/Q6_K), not from the Thor build.
  thor    the served Thor-format build ($V41_WORK/thor/<s10 gates.json thor_build_dir>: A<n> K-quant,
          A<n>kt trellis or B<n> bit-exact MXFP4), uploaded with thor.json and its keep-set.
          gate: gates.json thor (the pre-registered thor_quant gate of the arm that ships) AND the build
          fits the Thor budget. Published only after chain2's s08 serving gates ran on that exact build
          (golden_store + decode + server done markers); before that the repo is not created.
          The CUDA server source is bundled into the repo only with V41_PUBLISH_RUNTIME=1.
Every card carries the s11 speculative-decoding result ($V41_WORK/s11/decision.json, or "not
measured yet").
Every card states the teacher revision, corpus spec hash, keep-set fingerprint and the sha256 of
the pre-registered gates file; Engram instructions come from the pinned HF tree. A repo whose gate
failed keeps its card but stays PRIVATE.

    python3 ops/s12_publish.py [--only mxfp4,nvfp4,gguf,thor] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ops.lib import cards, env, gates as GT, keepset  # noqa: E402

STAGE = "s12"
OUT = env.WORK / "s12"
CARDS = env.OPS / "cards"
BASE_REPO = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
N_BASE = json.load(open(env.INFERENCE / "config.json"))["n_routed_experts"]


def jload(p: Path, what: str):
    if not p.exists():
        raise SystemExit(f"missing {what}: {p}")
    return json.load(open(p))


def context() -> dict:
    W = env.WORK
    res = jload(W / "s10" / "final" / "results.json", "s10 final results")
    g = jload(W / "s10" / "gates.json", "s10 gates")
    kA, fA = keepset.load(W / "s04" / "keepset.json")
    kB, fB = keepset.load(W / "s04" / "keepset_128.json")
    keepset.require(g["keepset_A_sha256"], fA, where="s10/gates.json")
    keepset.require(g["keepset_B_sha256"], fB, where="s10/gates.json")
    metaA, metaB = json.load(open(W / "s04" / "keepset.json")), json.load(open(W / "s04" / "keepset_128.json"))
    cands = jload(W / "s04" / "candidates.json", "s04 candidates")
    heal = jload(W / "s10" / "heal" / "decision.json", "heal decision")
    if heal["adopt"]:
        gs = sorted(heal["gains"].values())
        status = (f"measured and adopted (paired gate passed; median {gs[len(gs) // 2]:.4f}, "
                  f"range {gs[0]:.4f}-{gs[-1]:.4f})")
    else:
        status = "measured but not adopted (it did not pass its paired gate), so every gain is 1"
    tree = {x["path"]: x for x in json.load(open(env.SRC.parent / "v41_1017_tree.json")) if x.get("type") == "file"}
    eng = [tree[f"model-{i:05d}-of-00048.safetensors"] for i in (47, 48)]
    elist = "\n".join(f"- `{e['path']}` ({e['lfs']['size'] / 1e9:.1f} GB, sha256 `{e['lfs']['oid']}`)" for e in eng)
    elist += (f"\n\n```\nhf download {BASE_REPO} {eng[0]['path']} {eng[1]['path']} --revision {env.REVISION} "
              f"--local-dir .\n```")
    am = GT.load()[0]["release"]["pruned_floor"].get("must_beat_random_amended")
    pf = lambda b: (f"PASS under the gate as amended {am['date']} (FAIL as originally registered; see the evaluation)"
                    if am else "PASS") if b else "FAIL (repository kept private)"
    ctx = {
        "base": {"repo": BASE_REPO, "revision": env.REVISION},
        "keepset": {"A": {**{k: v for k, v in metaA.items() if k != "keep"}, "sha256": fA},
                    "B": {**{k: v for k, v in metaB.items() if k != "keep"}, "sha256": fB}},
        "corpus": jload(W / "corpus" / "manifest.json", "corpus manifest"),
        "arb": {"knee_A": cands["knees"]["A"], "knee_B": cands["knees"]["B"]},
        "heal": {**heal, "status": status},
        "s10": res,
        "gates": {"mxfp4_master": pf(g["mxfp4_master"]), "nvfp4": pf(g["nvfp4"]), "thor": pf(g["thor"]),
                  "thor_arm": g["thor_arm"]},
        "engram": {"list": elist},
        "limitations": {"text": limitations_text(res, W)},
        "spec": spec_context(W),
    }
    m5 = W / "s05" / f"manifest_REAP{len(kA['0'])}.json"
    if m5.exists():
        ctx["master"] = json.load(open(m5))
    arm = g["thor_arm"]
    # s10 trellis writes the served build dir (A<n>kt when the KT sub-arm was adopted); before it, the arm names it
    tb = Path(g["thor_build_dir"]) if g.get("thor_build_dir") else \
        W / "thor" / (f"A{len(kA['0'])}" if arm == "A" else f"B{len(kB['0'])}")
    if (tb / "thor.json").exists():
        tj = json.load(open(tb / "thor.json"))
        counts = {}
        for v in tj["alloc"].values():
            for part, t in v.items():
                counts[f"{part}:{t}"] = counts.get(f"{part}:{t}", 0) + 1
        desc = ("Experts were requantized once from the MXFP4 source with GPTQ error-feedback rounding into ggml K-quant blocks "
                "(per-layer mix, protected layers higher; counts of layer x projection-group: "
                + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())) + "). The heal gain was folded into w2 before rounding."
                if arm == "A" else "Experts are the bit-exact MXFP4 blocks of the smaller keep-set (no requantization).")
        if tj.get("kt"):
            desc = ("Experts were requantized once from the MXFP4 source into KT2/KT3 bitshift-trellis blocks (QTIP-style, "
                    f"{tj['kt']['window_bits']}-bit window, {tj['kt']['block']}-weight blocks, a randomized Hadamard rotation of "
                    f"width {tj['kt']['rht_block']} along the input dimension; encoder: {tj['kt']['encoder']}); counts of layer x "
                    "projection-group: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))
                    + ". The heal gain was folded into w2 before rounding.")
        ctx["thor"] = {"n_routed": tj["n_routed_experts"], "expert_bpw": tj["expert_bpw"], "expert_bytes": tj["expert_bytes"],
                       "description": desc, "keepset_sha256": tj["keepset_sha256"],
                       "arm_name": ("thorA_kt" if tj.get("kt") else "thorA") if arm == "A" else "b128"}
        ctx["thor_build"] = thor_context(W, tb, tj, ctx["thor"]["arm_name"], g)
    return ctx


def limitations_text(res: dict, W: Path) -> str:
    """Known weaknesses, from the gates file's per-bucket overrides and the measured confirm-split numbers
    (plus the s04 screen numbers that motivated the override). Empty when there is no override."""
    f = GT.load()[0]["release"]["pruned_floor"]
    S, out = res["summaries"], []
    arb = W / "s04" / "arbitration.json"
    scr = json.load(open(arb))["summaries"] if arb.exists() else {}
    for b, ov in f.get("bucket_overrides", {}).items():
        meas = ", ".join(f"{a} {S[a]['by_bucket'][b]['top1']['mean']:.4f}" for a in ("mxfp4_A", "thorA", "thorA_kt", "b128")
                         if a in S and b in S[a].get("by_bucket", {}))
        rnd = [scr[r]["by_bucket"][b]["top1"]["mean"] for r in scr if r.startswith("random") and b in scr[r]["by_bucket"]]
        line = (f"**Known limitation: {b}.** Top-1 agreement with the unpruned model on the held-out {b} documents is "
                f"lower than in every other domain (confirmation split: {meas or 'not measured'}). The release floor "
                f"for every other domain is {f['min_bucket_top1']}; for {b} it was lowered to a collapse guard of "
                f"{ov['min_top1']} ({ov['amended']}). ")
        if "hope_sum_pf0" in scr and rnd:
            line += (f"During expert selection, the shipped keep-set scored {scr['hope_sum_pf0']['by_bucket'][b]['top1']['mean']:.4f} "
                     f"on the screen split against {min(rnd):.4f}-{max(rnd):.4f} for random keep-sets of the same size. "
                     f"Keep-sets that protected more {b} experts recovered {b} only by losing general, finance and science "
                     f"text, so the better all-round keep-set ships. For {b} use, prefer the unpruned model.")
        out.append(line)
    ex = f.get("must_beat_random_exclude_buckets")
    C = res.get("comparisons", {})
    if ex and "mxfp4_A_vs_random0" in C:
        key = "mxfp4_A_vs_random0_excl_" + "_".join(ex)
        d0, d1 = C["mxfp4_A_vs_random0"]["d_kl32"], C[key]["d_kl32"]
        am = f["must_beat_random_amended"]
        out.append(f"**Gate amended after the result ({am['date']}).** The pre-registered rule required the pruned "
                   f"model to beat a random keep-set of the same size on kl32. Over all domains it did not reach "
                   f"significance (paired difference {d0['mean']:+.4f}, interval {d0['lo']:+.4f} to {d0['hi']:+.4f}), "
                   f"so this release FAILED its gate as originally registered. The cause is {', '.join(ex)} alone. "
                   f"After seeing that result, the owner amended the rule to exclude {', '.join(ex)}, whose weakness "
                   f"is already documented above. Without it the difference is {d1['mean']:+.4f} "
                   f"(interval {d1['lo']:+.4f} to {d1['hi']:+.4f}). Reason: {am['why']}")
    wv = GT.load()[0]["release"]["thor_quant"].get("waived_min_top1")
    if wv and "thorA" in S:
        o, m = S["thorA"]["overall"]["top1"], S["mxfp4_A"]["overall"]["top1"]
        out.append(f"**Thor build: top-1 floor waived ({wv['date']}, after the result).** The low-bit Thor build scored "
                   f"top-1 {o['mean']:.4f} (interval {o['lo']:.4f} to {o['hi']:.4f}) against a pre-registered floor of "
                   f"{f['min_top1']}, so it FAILED that floor; the MXFP4 master scores {m['mean']:.4f}. The owner waived "
                   f"this one clause for the Thor build. Every other clause still applies and passed. Reason: {wv['why']}")
    return "\n\n".join(out)


def thor_context(W: Path, tb: Path, tj: dict, arm_name: str, g: dict) -> dict:
    """The Thor-build card's values: format, bytes, resident GiB on Thor (build + main 1M-context state +
    workspace, s10_gates.json s11.fit), paired metrics of the arm that ships, and whether chain2's s08
    serving gates ran on this exact build."""
    res = jload(W / "s10" / ("trellis" if arm_name == "thorA_kt" else "final") / "results.json", "s10 results for the Thor arm")
    S = {"s10": {"summaries": res["summaries"]}}
    fit = GT.load()[0]["s11"]["fit"]
    name = tb.name
    build = sum(r["bytes"] for r in tj["files"].values())
    kv1m, kv_src = fit["main_kv_1m_gib_default"] * 2**30, "the pre-registered default (not measured yet)"
    c = W / "s08" / f"ctx1m_{name}.json"
    if c.exists() and json.load(open(c)).get("state_bytes"):
        kv1m, kv_src = json.load(open(c))["state_bytes"], f"measured by the s08 1M-context run (`{c.name}`)"
    resident = build + kv1m + fit["workspace_gib"] * 2**30
    dec, d8 = W / "s08" / f"decode_{name}.json", W / "s08" / f"decode8k_{name}.json"
    marks = [env.LOGS / f"done.s08_{m}_{name}" for m in ("golden_store", "decode", "server")]
    verified = dec.exists() and all(m.exists() for m in marks)
    speed = "not measured yet"
    if dec.exists():
        speed = f"{json.load(open(dec))['decode_tok_s']:.2f} tok/s at a short context"
        if d8.exists():
            speed += f", {json.load(open(d8))['decode_tok_s']:.2f} tok/s after an 8k-token prompt"
        speed += " (greedy, batch 1, Engram rows on NVMe; measured by the s08 decode gate on this build)"
    bundle = os.environ.get("V41_PUBLISH_RUNTIME") == "1"
    return {"name": name, "format": tj["format"], "arm": arm_name, "results_file": f"s10/{'trellis' if arm_name == 'thorA_kt' else 'final'}/results.json",
            "build_bytes": build, "kv1m_bytes": kv1m, "kv_src": kv_src, "workspace_bytes": fit["workspace_gib"] * 2**30,
            "resident_bytes": resident, "budget_bytes": fit["budget_gib"] * 2**30, "fits": resident <= fit["budget_gib"] * 2**30,
            "fit_text": "fits" if resident <= fit["budget_gib"] * 2**30 else "DOES NOT FIT",
            "overall_table": cards.table(S, "table_arms", arm_name), "bucket_table": cards.table(S, "table_buckets", arm_name),
            "speed": speed, "verified": verified, "golden_note": golden_note(name),
            "runtime_note": ("The CUDA server source used to verify this build is bundled here as `runtime/v41-thor-runtime.tar.gz` "
                             "(the `cuda/` tree and the Engram row-file builder of the build pipeline, at the commit named in the "
                             "archive name inside it)." if bundle else
                             "The CUDA server that runs this format (`v41_server`) is not public yet; the files here are complete "
                             "and verified, but you need that runtime to load them.")}


def golden_note(name: str) -> str:
    """The CUDA-vs-torch golden gates as they actually ran (numbers from the s08 logs) and how each was resolved."""
    import re
    def pooled(log: Path):
        if not log.exists():
            return None
        t = log.read_text()
        m = re.findall(r"pooled\s+(\d+) disagreements, (\d+) near-ties", t)
        v = re.findall(r"^Gate V41 GOLDEN: (PASS|FAIL)$", t, re.M)
        return (int(m[-1][0]), int(m[-1][1]), v[-1]) if m and v else None
    src, arm = pooled(env.LOGS / "s08_golden_src.log"), pooled(env.LOGS / f"s08_golden_{name}.log")
    def fmt(r):
        return "not run" if r is None else (f"{r[1]} of {r[0]} teacher-forced next-token disagreements were near-ties "
                                            f"({100 * r[1] / max(r[0], 1):.1f}%), {r[2]} as registered (rule: at least 90%)")
    out = ("**CUDA correctness gates (layer by layer against the torch reference, `cuda/tests/gate_v41_golden.cu`).** "
           f"Source checkpoint: {fmt(src)}")
    if src and src[2] == "FAIL":
        out += ("; passed under an owner amendment made before its diagnostic was read, for chained bf16 drift "
                "(`ops/decisions/2026-10-24-golden-src-drift-amendment.md`)")
    out += f". This build: {fmt(arm)}"
    if arm and arm[2] == "FAIL":
        out += ("; the owner waived it and released the build on the serving gates (Engram row store, decode determinism, 1M context, server) "
                "(`ops/decisions/2026-10-25-golden-a192-drift-amendment.md`, which also records the per-row diagnostics)")
    return out + ". Every hidden-state layer check and the agreement and final-cosine rules passed in both runs."


def overlay_context(W: Path, g: dict, tc: dict | None) -> dict | None:
    """The NVFP4 dense-overlay card: s10's paired overlay-vs-thorA numbers and the s08 decode runs of the same build
    with and without the overlay. None until every piece exists (the overlay ships only on top of the published,
    served Thor build)."""
    led = jload(env.LOGS / "s12_ledger.json", "s12 ledger") if (env.LOGS / "s12_ledger.json").exists() else {}
    if tc is None or not g.get("thor") or not g.get("overlay_adopted") or not (led.get("thor") or {}).get("public"):
        return None
    name = tc["name"]
    dec = {k: W / "s08" / f"decode{s}_{name}{k}.json" for k in ("", "_overlay") for s in ("", "8k")}
    marks = [env.LOGS / f"done.s08_{m}_{name}_overlay" for m in ("decode", "server")]
    if not all(p.exists() for p in dec.values()) or not all(m.exists() for m in marks):
        return None
    tok = lambda s, k: json.load(open(W / "s08" / f"decode{s}_{name}{k}.json"))["decode_tok_s"]
    res = jload(W / "s10" / "final" / "results.json", "s10 final results")
    cmp, S = res["comparisons"]["overlay_vs_thorA"], res["summaries"]
    oj = json.load(open(W / "thor" / "overlay_nvfp4" / "overlay.json"))
    ob = sum(p.stat().st_size for p in (W / "thor" / "overlay_nvfp4").iterdir() if p.is_file())
    return {"build": name, "bytes_gib": f"{ob / 2**30:.2f}", "layers": oj.get("layers"),
            "tok_base": f"{tok('', ''):.2f}", "tok_ovl": f"{tok('', '_overlay'):.2f}",
            "tok8k_base": f"{tok('8k', ''):.2f}", "tok8k_ovl": f"{tok('8k', '_overlay'):.2f}",
            "d_kl32": f"{cmp['d_kl32']['mean']:+.4f} (interval {cmp['d_kl32']['lo']:+.4f} to {cmp['d_kl32']['hi']:+.4f})",
            "top1_base": f"{S['thorA']['overall']['top1']['mean']:.4f}",
            "top1_ovl": f"{S['thorA_overlay']['overall']['top1']['mean']:.4f}",
            "flips": f"{(S['thorA_overlay']['overall']['confident_flip_rate'] - S['thorA']['overall']['confident_flip_rate']) * 100:+.2f}",
            "gate": res["gates"]["overlay"]}


def runtime_bundle(out: Path) -> Path:
    """git archive of the runtime (cuda/ without build products, the Engram row-file builder and its lib)."""
    import subprocess
    sha = subprocess.check_output(["git", "rev-parse", "--short=12", "HEAD"], cwd=env.REPO, text=True).strip()
    f = out / "v41-thor-runtime.tar.gz"
    subprocess.check_call(["git", "archive", "--format=tar.gz", f"--prefix=v41-thor-runtime-{sha}/", "-o", str(f), "HEAD",
                           "cuda", "ops/s09_engram_rowfile.py", "ops/lib", "llamacpp/ENGRAM_ROWFILE.md"], cwd=env.REPO)
    return f


def spec_context(W: Path) -> dict:
    f = W / "s11" / "decision.json"
    if not f.exists():
        return {"summary": "Not measured yet. The checkpoint's own DSpark head (the three `mtp` layers) is "
                           "being refit to this keep-set; this section is filled from that result."}
    from ops import s11_spec
    return {"summary": s11_spec.summary(json.load(open(f)))}


def _ci(m: dict, f: str) -> str:
    return f"{cards.fmt(m['mean'], f)} [{cards.fmt(m['ci95'][0], f)}, {cards.fmt(m['ci95'][1], f)}]"


def gguf_heal_sentence(m: dict, heal: dict | None = None) -> str:
    """One card sentence on the heal gain, from the gains s09 read back out of the published files
    (manifest routed_gain). With the s10 heal decision for the same keep-set, the two must agree."""
    g = m.get("routed_gain")
    if heal is not None and heal.get("keepset_sha256") == m["keepset_sha256"]:
        want = [float(heal["gains"][str(L)]) for L in range(len(heal["gains"]))] if heal["adopt"] else None
        if want is not None and all(v == 1.0 for v in want):
            want = None
        if (g is None) != (want is None) or (g is not None and (len(g) != len(want) or any(
                abs(a - b) > 1e-6 * max(1.0, abs(b)) for a, b in zip(g, want)))):
            raise SystemExit("gguf: the files' deepseek41.reap.routed_gain does not match the s10 heal decision "
                             f"(adopt={heal['adopt']}); rebuild s09 convert after the decision")
    if g is None:
        return ("No heal gain is applied: the files carry no `deepseek41.reap.routed_gain` key, so the routed experts' "
                "summed output is used exactly as converted from the unhealed MXFP4 master.")
    s = sorted(g)
    return (f"The per-layer heal gain is carried in the files as `deepseek41.reap.routed_gain` ({len(g)} layers, median "
            f"{s[len(s) // 2]:.4f}, range {s[0]:.4f}-{s[-1]:.4f}), and the fork's patch 0004 multiplies each layer's "
            "routed-expert sum by it before the shared expert is added, the same gain the NVFP4 build folds into `w2`.")


def dspark_section(g: dict | None, file: str, repo: str) -> str:
    """The GGUF card's speculative-decoding section, from s09_dspark gate.json (pass or fail); empty before it ran."""
    if g is None:
        return ""
    if not g["pass"]:
        return ("## Speculative decoding\n\nNo DSpark drafter is published with this release. Its pre-registered gate "
                "(`dspark_gguf`) did not pass: " + "; ".join(g["why"]) + ".")
    faster = g["speedup"] >= 1.0
    return (
        "## Speculative decoding (optional)\n\n"
        f"`{file}` in this repository is the base checkpoint's own DSpark drafter (`mtp.0`-`mtp.2`, not pruned), "
        "converted with the fork's patch 0005. Every tensor was checked against an independent conversion of the same "
        "drafter. It is loaded only when you ask for it:\n\n"
        "```bash\n"
        f"llamacpp/llama.cpp/build-thor/bin/llama-server -m gguf/<first split of the level> -md gguf/{file} -ngld 999 \\\n"
        f"    --spec-type draft-dspark --spec-draft-n-max {g['draft_n_max']} -ngl 999 -c {g['ctx']} --jinja --port 8080\n"
        "```\n\n"
        f"With `-hf {repo}` and `--spec-type draft-dspark`, llama.cpp finds it on its own.\n\n"
        f"The drafter adds {g['drafter_gib']:.1f} GiB. Measured on the Jetson AGX Thor beside `{g['level']}` "
        f"({g['level_gib']:.1f} GiB), the largest level it fits next to with a safety margin, greedy decoding of "
        f"{g['n_predict']} tokens after {g['n_prompts']} held-out document prompts: "
        f"{g['tps_plain']:.2f} tok/s without the drafter, {g['tps_spec']:.2f} with it "
        f"({g['speedup']:.2f}x, geometric means), {100 * g['accept_rate']:.1f}% of drafted tokens accepted. "
        + ("" if faster else "On this machine the drafter is a net slowdown: leave it off on Thor. It may still help "
               "on hardware where verifying a few tokens at once costs little more than decoding one. ")
        + f"{100 * g['identical_frac']:.0f}% of the outputs were token-identical to decoding without it; the rest "
        "diverged at a near-tie, where batched verification on CUDA can pick the other token.")


def ctx_cap() -> int:
    """The fork's per-sequence context clamp (llamacpp patch 0006): V4.1's candidate-block span, from the config."""
    c = json.load(open(env.INFERENCE / "config.json"))
    return int(c["candidate_topk_blocks"]) * int(c["candidate_block_size"])


def gguf_context(m: dict, gate: dict, root: Path | None = None, heal: dict | None = None) -> dict:
    """The GGUF card's values: one row per published level from its own teacher_kld.json stats (carried
    in the s09 manifest) and the s09 gate; the dense precision the writer used."""
    rows = ["| level | expert bpw | size (GiB) | kl32 (95% CI) | top-1 agreement | dNLL | GPU layers in the eval |",
            "|---|---|---|---|---|---|---|"]
    partial, tokens = [], set()
    for lv, r in sorted(m["levels"].items()):
        k = r["kld"]
        if k is None:
            raise SystemExit(f"gguf: {lv} is in the manifest without a teacher_kld.json")
        ngl = gate["levels"][lv].get("ngl") or "all"
        if ngl != "all":
            partial.append(lv)
        rows.append(f"| {lv} | {cards.fmt(r['expert_bpw'], 'f2')} | {cards.fmt(r['bytes'], 'gib')} | {_ci(k['kl_top32'], 'f4')} "
                    f"| {_ci(k['top1'], 'f4')} | {_ci(k['dnll'], 'f4')} | {ngl} |")
    for lv in m["levels"]:
        f = (root or env.WORK / "s09") / lv / "teacher_eval" / "teacher_kld.json"
        if f.exists():
            tokens.add(json.load(open(f))["tokens"])
    if len(tokens) != 1:
        raise SystemExit(f"gguf: levels were scored on different token sets: {sorted(tokens)}")
    note = ("Levels marked with a GPU-layer count were too large for full offload on the 128 GB Thor and were scored "
            "with that many layers on the GPU and the rest on the CPU; the numbers are the same model, only slower."
            if partial else "Every level was scored fully offloaded to the GPU.")
    dense = ["| tensors | precision |", "|---|---|"] + [f"| {k} | {v} |" for k, v in m["dense_precision"].items()]
    return {"manifest_name": "gguf_manifest.json", **m, "levels_table": "\n".join(rows), "offload_note": note,
            "dense_table": "\n".join(dense), "eval_tokens": tokens.pop(), "engram_doc": m.get("engram", ""),
            "heal_sentence": gguf_heal_sentence(m, heal),
            "dspark_section": "",          # filled by main() from s09_dspark gate.json, once the repo name is known
            "ctx_cap": f"{ctx_cap():,}",
            # the level the 128 GB recipe runs: the lowest-numbered passing level (L2 unless only the L3 fallback passed)
            "run_level": min(m["levels"]),
            "run_file": _run_file(m),
            "run_gib": cards.fmt(m["levels"][min(m["levels"])]["bytes"], "gib"),
            "gate_status": (f"FAIL on the pre-registered floor ({'; '.join(gate['owner_release']['why_failed'])}); "
                            f"{gate['owner_release']['level']} released by owner decision (llama-server smoke passed)")
                           if gate.get("owner_release") else
                           "PASS (" + ", ".join(gate["passing"]) + ")" if gate["passing"] else "FAIL (repository kept private)"}


def _run_file(m: dict) -> str:
    """The first split of the run level (the lowest-numbered passing level): the file the 128 GB recipe points at."""
    lv = min(m["levels"])
    names = sorted(Path(f["path"]).name for f in m["files"] if f"-{lv}-" in Path(f["path"]).name)
    if not names:
        raise SystemExit(f"gguf: manifest lists no split file for run level {lv}")
    return names[0]


def local_files(m: dict, kind: str) -> list[tuple[Path, str]]:
    """Every file of a locally kept s05/s07 artifact (Seagate), as (path, path_in_repo). Each weight shard is re-hashed
    against the manifest first: the copy crossed a USB disk and will be published. An artifact that was already
    uploaded by the old private-repo flow (uploaded=True) has nothing local to send."""
    if m.get("uploaded", True) and not m.get("local_dir"):
        return []
    root = Path(m["local_dir"])
    if not root.is_dir():
        raise SystemExit(f"{kind}: local artifact {root} is missing (Seagate not mounted?)")
    import ops.s05_surgery as S5
    for name, info in m["files"].items():
        got = S5.sha256_file(root / name)
        if got != info["sha256"]:
            raise SystemExit(f"{kind}: {root / name} sha256 {got[:16]} != manifest {info['sha256'][:16]}")
    out = [(p, str(p.relative_to(root))) for p in sorted(root.rglob("*")) if p.is_file() and p.name != "README.md"]
    names = {dst for _, dst in out}
    missing = sorted(set(m["files"]) - names)
    if missing:
        raise SystemExit(f"{kind}: local artifact lacks {missing[:3]}")
    return out


def retry(fn, what: str):
    from ops.s05_surgery import retry as r
    return r(fn, what)


def api():
    os.environ.pop("HF_XET_HIGH_PERFORMANCE", None)
    os.environ["HF_XET_FIXED_UPLOAD_CONCURRENCY"] = "8"
    from huggingface_hub import HfApi
    return HfApi()


def make_public(h, repo: str) -> None:
    if hasattr(h, "update_repo_settings"):
        h.update_repo_settings(repo_id=repo, private=False)
    else:
        h.update_repo_visibility(repo_id=repo, private=False)


def publish(a):
    ctx = context()
    parts = {p.stem.replace(".md", ""): p.read_text() for p in CARDS.glob("*.md.part")}
    g = json.load(open(env.WORK / "s10" / "gates.json"))
    only = set(a.only.split(",")) if a.only else {"mxfp4", "nvfp4", "gguf", "thor", "overlay"}
    led = env.Ledger(env.LOGS / "s12_ledger.json")
    h = None if a.dry_run else api()
    nA = len(keepset.load(env.WORK / "s04" / "keepset.json")[0]["0"])
    plan = []
    fpA = keepset.load(env.WORK / "s04" / "keepset.json")[1]
    if "mxfp4" in only:
        m = json.load(open(env.WORK / "s05" / f"manifest_REAP{nA}.json"))
        keepset.require(m["keepset_sha256"], fpA, where="s05 master manifest (s04 re-decided after s05?)")
        plan.append(("mxfp4", m["repo"], g["mxfp4_master"], {"name": m["repo"].split("/")[-1], "n_routed": nA},
                     local_files(m, "mxfp4")))
    if "nvfp4" in only:
        m = json.load(open(env.WORK / "s07" / f"manifest_nvfp4_REAP{nA}.json"))
        if m.get("skipped"):
            env.log(f"nvfp4: not built ({m['skipped']}); no NVFP4 repo", STAGE)
        elif not m.get("loader_verified", False):
            # no known loader reads ModelOpt NVFP4 experts beside block-32 ue8m0 FP8 dense layers (s07 RISK-1): an
            # unloadable 150 GB repo is not a release. It stays on the Seagate; the owner decides.
            env.log(f"nvfp4: loader_verified is false; not published (kept at {m.get('local_dir')})", STAGE, "WARN")
            with open(env.LOGS / "OWNER_FOLLOWUP", "a") as f:
                f.write(f"[{env.now()}] s12: NVFP4 REAP{nA} built at {m.get('local_dir')} but not published: no verified "
                        "loader for NVFP4 experts + FP8 block-32 dense (see ops/logs/audit/s07.md RISK-1)\n")
        else:
            keepset.require(m["keepset_sha256"], fpA, where="s07 nvfp4 manifest")
            plan.append(("nvfp4", m["repo"], g["nvfp4"], {"name": m["repo"].split("/")[-1], "n_routed": nA},
                         local_files(m, "nvfp4")))
    if "gguf" in only:
        mf = env.WORK / "s09" / "manifest.json"
        if not mf.exists():
            env.log("gguf: s09 manifest not present yet; skipping the GGUF repo (re-run s12 after s09)", STAGE)
        else:
            m = json.load(open(mf))
            gg = jload(env.WORK / "s09" / "gates.json", "s09 GGUF gate (ops/s09_gate.py)")
            stray = sorted(set(m["levels"]) - set(gg["passing"]))
            if stray:
                raise SystemExit(f"gguf: manifest lists levels that did not pass the s09 gate: {stray}")
            user = "<user>" if a.dry_run else h.whoami()["name"]
            repo = m.get("repo") or f"{user}/DeepSeek-V4.1-Flash-1017-REAP{m['n_routed_experts']}-GGUF"
            ctx["gguf"] = gguf_context(m, gg, heal=ctx["heal"])
            files = [(Path(f["path"]), Path(f["path"]).name) for f in m["files"]]
            import ops.s09_dspark as SD
            dg = SD.OUT / "gate.json"
            dg = json.load(open(dg)) if dg.exists() else None
            ctx["gguf"]["dspark_section"] = dspark_section(dg, SD.FILE, repo)
            if dg is not None and dg["pass"]:
                files.append((SD.OUT / SD.FILE, SD.FILE))     # repo root: where -hf looks for dspark-*.gguf
            # The "gguf" gate is defined as the ladder of the PUBLISHED master and has no beat-random or bucket
            # clause of its own; a GGUF of a master that failed its release gate must not go public through it.
            gguf_ok = bool(gg["passing"]) and bool(g["mxfp4_master"])
            if gg["passing"] and not g["mxfp4_master"]:
                env.log("gguf: levels passed s09 but the MXFP4 master failed its release gate; kept private", STAGE, "WARN")
            plan.append(("gguf", repo, gguf_ok, {"name": repo.split("/")[-1], "repo": repo, "n_routed": m["n_routed_experts"]},
                         files))
    if "thor" in only:
        tc = ctx.get("thor_build")
        if tc is None:
            env.log("thor: no complete served Thor build; skipping the Thor repo", STAGE, "WARN")
        elif not tc["verified"]:
            env.log(f"thor: {tc['name']} has not passed chain2's s08 serving gates yet (golden_store/decode/server); "
                    "skipping the Thor repo (chain2 re-runs s12 after them)", STAGE)
        else:
            tb = Path(g["thor_build_dir"]) if g.get("thor_build_dir") else env.WORK / "thor" / tc["name"]
            tj = json.load(open(tb / "thor.json"))
            kp = env.WORK / "s04" / ("keepset_128.json" if g["thor_arm"] == "B" else "keepset.json")
            keepset.require(tj["keepset_sha256"], keepset.load(kp)[1], where=str(tb))
            d = OUT / "thor"
            d.mkdir(parents=True, exist_ok=True)
            (d / "reap_keepset.json").write_bytes(kp.read_bytes())
            files = [(p, p.name) for p in [tb / f for f in sorted(tj["files"])] + [tb / "thor.json", d / "reap_keepset.json"]]
            if os.environ.get("V41_PUBLISH_RUNTIME") == "1":
                rb = runtime_bundle(d)
                files.append((rb, f"runtime/{rb.name}"))
            user = "<user>" if a.dry_run else h.whoami()["name"]
            repo = f"{user}/DeepSeek-V4.1-Flash-1017-REAP{tj['n_routed_experts']}-Thor"
            plan.append(("thor", repo, bool(g["thor"]) and tc["fits"],
                         {"name": repo.split("/")[-1], "n_routed": tj["n_routed_experts"]}, files))
    if "overlay" in only:
        oc = overlay_context(env.WORK, g, ctx.get("thor_build"))
        if oc is None:
            env.log("overlay: not ready (needs the published Thor build, the overlay gate and the s08 overlay "
                    "decode + server runs); skipping", STAGE)
        else:
            ctx["overlay"] = oc
            user = "<user>" if a.dry_run else h.whoami()["name"]
            repo = f"{user}/DeepSeek-V4.1-Flash-1017-REAP{nA}-Thor-NVFP4-overlay"
            ctx["overlay"]["thor_repo"] = f"{user}/DeepSeek-V4.1-Flash-1017-REAP{nA}-Thor"
            od = env.WORK / "thor" / "overlay_nvfp4"
            plan.append(("overlay", repo, True, {"name": repo.split("/")[-1], "n_routed": nA},
                         [(p, p.name) for p in sorted(od.iterdir()) if p.is_file()]))
    for kind, repo, passed, art, files in plan:
        c = dict(ctx, artifact={**art, "n_routed_base": N_BASE, "kind": kind})
        text = cards.render((CARDS / f"{kind}.md.tmpl").read_text(), c, parts)
        d = OUT / kind
        d.mkdir(parents=True, exist_ok=True)
        (d / "README.md").write_text(text)
        env.log(f"{kind}: card rendered -> {d / 'README.md'} (gate {'PASS' if passed else 'FAIL'})", STAGE)
        if a.dry_run:
            continue
        if not passed:
            # The free account's 100 GB private quota cannot hold these artifacts, and a failed gate means they are not
            # released: nothing is uploaded; the files and the rendered card stay local for the owner.
            env.log(f"{kind}: gate FAILED - not uploaded; card at {d / 'README.md'}, files stay local "
                    f"({len(files)} files)", STAGE, "WARN")
            led.finish(kind, repo=repo, public=False, uploaded=False)
            continue
        # Gate passed: the repo is PUBLIC before the large files go up, so they never count against the private quota.
        if not h.repo_exists(repo, repo_type="model"):
            h.create_repo(repo, private=False, repo_type="model")
        elif h.model_info(repo).private:
            make_public(h, repo)
        for f, dst in files:
            st = f.stat()
            sig = {"bytes": st.st_size, "mtime": int(st.st_mtime)}
            prev = led.get(f"{kind}/file/{dst}") or {}
            # a rebuilt file (new size or mtime) is uploaded again; a name alone does not prove the bytes are current
            if not (led.done(f"{kind}/file/{dst}") and prev.get("bytes") == sig["bytes"] and prev.get("mtime") == sig["mtime"]):
                retry(lambda: h.upload_file(path_or_fileobj=str(f), path_in_repo=dst, repo_id=repo, repo_type="model",
                                            commit_message=f"s12: {dst}"), what=f"{kind}/{dst}")
                led.finish(f"{kind}/file/{dst}", **sig)
        if kind == "gguf":
            env.atomic_json(d / "gguf_manifest.json", ctx["gguf"])
            retry(lambda: h.upload_file(path_or_fileobj=str(d / "gguf_manifest.json"), path_in_repo="gguf_manifest.json",
                                        repo_id=repo), what="gguf_manifest.json")
            retry(lambda: h.upload_file(path_or_fileobj=str(env.SRC / "LICENSE"), path_in_repo="LICENSE", repo_id=repo),
                  what="LICENSE")
        retry(lambda: h.upload_file(path_or_fileobj=str(d / "README.md"), path_in_repo="README.md", repo_id=repo,
                                    commit_message="s12: model card generated from s10 results"), what="README.md")
        env.log(f"{kind}: {repo} is PUBLIC ({len(files)} files)", STAGE)
        led.finish(kind, repo=repo, public=True, uploaded=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only")
    ap.add_argument("--dry-run", action="store_true", help="render the cards only")
    a = ap.parse_args()
    publish(a)


if __name__ == "__main__":
    main()
