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
    pf = lambda b: "PASS" if b else "FAIL (repository kept private)"
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
            "speed": speed, "verified": verified,
            "runtime_note": ("The CUDA server source used to verify this build is bundled here as `runtime/v41-thor-runtime.tar.gz` "
                             "(the `cuda/` tree and the Engram row-file builder of the build pipeline, at the commit named in the "
                             "archive name inside it)." if bundle else
                             "The CUDA server that runs this format (`v41_server`) is not public yet; the files here are complete "
                             "and verified, but you need that runtime to load them.")}


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
            # the level the 128 GB recipe runs: the lowest-numbered passing level (L2 unless only the L3 fallback passed)
            "run_level": min(m["levels"]),
            "run_file": _run_file(m),
            "run_gib": cards.fmt(m["levels"][min(m["levels"])]["bytes"], "gib"),
            "gate_status": "PASS (" + ", ".join(gate["passing"]) + ")" if gate["passing"] else "FAIL (repository kept private)"}


def _run_file(m: dict) -> str:
    """The first split of the run level (the lowest-numbered passing level): the file the 128 GB recipe points at."""
    lv = min(m["levels"])
    names = sorted(Path(f["path"]).name for f in m["files"] if f"-{lv}-" in Path(f["path"]).name)
    if not names:
        raise SystemExit(f"gguf: manifest lists no split file for run level {lv}")
    return names[0]


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
    only = set(a.only.split(",")) if a.only else {"mxfp4", "nvfp4", "gguf", "thor"}
    led = env.Ledger(env.LOGS / "s12_ledger.json")
    h = None if a.dry_run else api()
    nA = len(keepset.load(env.WORK / "s04" / "keepset.json")[0]["0"])
    plan = []
    if "mxfp4" in only:
        m = json.load(open(env.WORK / "s05" / f"manifest_REAP{nA}.json"))
        plan.append(("mxfp4", m["repo"], g["mxfp4_master"], {"name": m["repo"].split("/")[-1], "n_routed": nA}, []))
    if "nvfp4" in only:
        m = json.load(open(env.WORK / "s07" / f"manifest_nvfp4_REAP{nA}.json"))
        if m.get("skipped"):
            env.log(f"nvfp4: not built ({m['skipped']}); no NVFP4 repo", STAGE)
        else:
            plan.append(("nvfp4", m["repo"], g["nvfp4"], {"name": m["repo"].split("/")[-1], "n_routed": nA}, []))
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
            plan.append(("gguf", repo, bool(gg["passing"]), {"name": repo.split("/")[-1], "repo": repo, "n_routed": m["n_routed_experts"]},
                         [Path(f["path"]) for f in m["files"]]))
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
            files = [tb / f for f in sorted(tj["files"])] + [tb / "thor.json", d / "reap_keepset.json"]
            if os.environ.get("V41_PUBLISH_RUNTIME") == "1":
                files.append(runtime_bundle(d))
            user = "<user>" if a.dry_run else h.whoami()["name"]
            repo = f"{user}/DeepSeek-V4.1-Flash-1017-REAP{tj['n_routed_experts']}-Thor"
            plan.append(("thor", repo, bool(g["thor"]) and tc["fits"],
                         {"name": repo.split("/")[-1], "n_routed": tj["n_routed_experts"]}, files))
    for kind, repo, passed, art, files in plan:
        c = dict(ctx, artifact={**art, "n_routed_base": N_BASE, "kind": kind})
        text = cards.render((CARDS / f"{kind}.md.tmpl").read_text(), c, parts)
        d = OUT / kind
        d.mkdir(parents=True, exist_ok=True)
        (d / "README.md").write_text(text)
        env.log(f"{kind}: card rendered -> {d / 'README.md'} (gate {'PASS' if passed else 'FAIL'})", STAGE)
        if a.dry_run:
            continue
        h.create_repo(repo, private=True, exist_ok=True, repo_type="model")
        for f in files:
            dst = f"runtime/{f.name}" if f.name.endswith(".tar.gz") else f.name
            if not led.done(f"{kind}/file/{dst}"):
                h.upload_file(path_or_fileobj=str(f), path_in_repo=dst, repo_id=repo, repo_type="model")
                led.finish(f"{kind}/file/{dst}")
        if kind == "gguf":
            env.atomic_json(d / "gguf_manifest.json", ctx["gguf"])
            h.upload_file(path_or_fileobj=str(d / "gguf_manifest.json"), path_in_repo="gguf_manifest.json", repo_id=repo)
            h.upload_file(path_or_fileobj=str(env.SRC / "LICENSE"), path_in_repo="LICENSE", repo_id=repo)
        h.upload_file(path_or_fileobj=str(d / "README.md"), path_in_repo="README.md", repo_id=repo,
                      commit_message="s12: model card generated from s10 results")
        if passed:
            make_public(h, repo)
            env.log(f"{kind}: {repo} is now PUBLIC", STAGE)
        else:
            env.log(f"{kind}: gate failed, {repo} stays PRIVATE", STAGE, "WARN")
        led.finish(kind, repo=repo, public=bool(passed))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only")
    ap.add_argument("--dry-run", action="store_true", help="render the cards only")
    a = ap.parse_args()
    publish(a)


if __name__ == "__main__":
    main()
