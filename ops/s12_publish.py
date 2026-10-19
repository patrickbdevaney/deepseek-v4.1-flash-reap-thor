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
Every card carries the s11 speculative-decoding result ($V41_WORK/s11/decision.json, or "not
measured yet").
Every card states the teacher revision, corpus spec hash, keep-set fingerprint and the sha256 of
the pre-registered gates file; Engram instructions come from the pinned HF tree. A repo whose gate
failed keeps its card but stays PRIVATE.

    python3 ops/s12_publish.py [--only mxfp4,nvfp4,gguf] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ops.lib import cards, env, keepset  # noqa: E402

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
    tb = W / "thor" / (f"A{len(kA['0'])}" if arm == "A" else f"B{len(kB['0'])}")
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
        ctx["thor"] = {"n_routed": tj["n_routed_experts"], "expert_bpw": tj["expert_bpw"], "expert_bytes": tj["expert_bytes"],
                       "description": desc, "keepset_sha256": tj["keepset_sha256"], "arm_name": "thorA" if arm == "A" else "b128"}
    return ctx


def spec_context(W: Path) -> dict:
    f = W / "s11" / "decision.json"
    if not f.exists():
        return {"summary": "Not measured yet. The checkpoint's own DSpark head (the three `mtp` layers) is "
                           "being refit to this keep-set; this section is filled from that result."}
    from ops import s11_spec
    return {"summary": s11_spec.summary(json.load(open(f)))}


def _ci(m: dict, f: str) -> str:
    return f"{cards.fmt(m['mean'], f)} [{cards.fmt(m['ci95'][0], f)}, {cards.fmt(m['ci95'][1], f)}]"


def gguf_context(m: dict, gate: dict, root: Path | None = None) -> dict:
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
            "gate_status": "PASS (" + ", ".join(gate["passing"]) + ")" if gate["passing"] else "FAIL (repository kept private)"}


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
    only = set(a.only.split(",")) if a.only else {"mxfp4", "nvfp4", "gguf"}
    led = env.Ledger(env.LOGS / "s12_ledger.json")
    h = None if a.dry_run else api()
    nA = len(keepset.load(env.WORK / "s04" / "keepset.json")[0]["0"])
    plan = []
    if "mxfp4" in only:
        m = json.load(open(env.WORK / "s05" / f"manifest_REAP{nA}.json"))
        plan.append(("mxfp4", m["repo"], g["mxfp4_master"], {"name": m["repo"].split("/")[-1], "n_routed": nA}, []))
    if "nvfp4" in only:
        m = json.load(open(env.WORK / "s07" / f"manifest_nvfp4_REAP{nA}.json"))
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
            ctx["gguf"] = gguf_context(m, gg)
            plan.append(("gguf", repo, bool(gg["passing"]), {"name": repo.split("/")[-1], "n_routed": m["n_routed_experts"]},
                         [Path(f["path"]) for f in m["files"]]))
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
            if not led.done(f"{kind}/file/{f.name}"):
                h.upload_file(path_or_fileobj=str(f), path_in_repo=f.name, repo_id=repo, repo_type="model")
                led.finish(f"{kind}/file/{f.name}")
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
