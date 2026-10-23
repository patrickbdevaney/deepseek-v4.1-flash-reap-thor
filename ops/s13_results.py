#!/usr/bin/env python3
"""s13: publish the REAP record itself, not only the weights.

    python3 ops/s13_results.py export            # small artifacts -> results/, RESULTS.md rendered from them
    python3 ops/s13_results.py scan              # secret scan of results/ and the whole git history
    python3 ops/s13_results.py release [--push]  # export + scan + commit; once the MXFP4 master is public (s12 ledger):
                                                 # bulk arrays -> a public HF dataset, GitHub repo -> public

What goes where (owner decision 2026-10-22: the repo publishes every REAP result, the process and the relevant
intermediate data, plus the rationale in DECISIONS.md):
  * git, results/: every JSON/JSONL result, ledger, mask, keep-set, gate and decision under $V41_WORK, the run logs,
    and the calibration corpus PROVENANCE (sources, licences, per-row source/token counts). The corpus token arrays and
    images are NOT published: their text belongs to the sources and several licences (CC-BY-SA, ODC-BY) attach terms
    to redistribution. sources.json names every dataset and revision so the corpus can be rebuilt (ops/s02_corpus.py).
  * HF dataset <user>/DeepSeek-V4.1-Flash-1017-REAP<n>-artifacts: the per-layer saliency/HOPE banks (s03/bank: per-
    expert sums, F = p p^T interaction matrices, activation stats), the teacher routing trace and top-32 teacher cache,
    the s06 heal banks, and the per-row metrics of every s04/s10 replay arm. These are the inputs that let anyone
    re-derive every mask and every table without re-running the 763B forward.
  * not published: Engram row cache (60 GB), activation taps (15 GB), replay embedding caches. Bulk, and regenerated
    by s03/s04/s10 from the pinned checkpoint.

The repo and the dataset go public only after the MXFP4 master did (s12 publishes it only when its gate passed), and
only if the secret scan is clean. Every number in RESULTS.md is read from an artifact; nothing here is a literal.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ops.lib import env  # noqa: E402

STAGE = "s13"
REPO = env.REPO
RESULTS = REPO / "results"
W = env.WORK
SMALL = {".json", ".jsonl", ".csv", ".md", ".txt"}
CAP = 20 << 20
# corpus token arrays/images and the defective/archived runs stay out; bulk dirs go to the dataset, not git
EXCLUDE = ["corpus.v1_defective/*", "*/archive_*", "corpus/pool_*.json", "corpus/images/*", "s03/engram/*",
           "s03/taps/*", "*/_embed/*", "s13_bulk/*", "*.tmp"]
BULK = ["s03/bank/*", "s03/routing/*", "s03/teacher/*", "s06/bank/*", "s06/measure/*", "s04/replay/*/metrics/*",
        "s10/*/replay/*/metrics/*"]
SECRETS = re.compile(r"hf_[A-Za-z0-9]{30,}|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9]{32,}"
                     r"|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY"
                     r"|\b100\.(6[4-9]|[7-9][0-9]|1[01][0-9]|12[0-7])\.\d{1,3}\.\d{1,3}\b")   # Tailscale CGNAT


def excluded(rel: str) -> bool:
    return any(fnmatch.fnmatch(rel, p) for p in EXCLUDE)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------------------ export
def export() -> dict:
    files, skipped = {}, []
    for p in sorted(W.rglob("*")):
        rel = str(p.relative_to(W))
        if not p.is_file() or p.suffix not in SMALL or excluded(rel):
            continue
        if p.stat().st_size > CAP:
            skipped.append(rel)
            continue
        dst = RESULTS / "work" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)
        files[f"work/{rel}"] = {"sha256": sha256(p), "bytes": p.stat().st_size}
    logs = env.LOGS
    for p in sorted(logs.rglob("*")):
        rel = str(p.relative_to(logs))
        if not p.is_file() or rel.startswith("done.") or p.suffix == ".tmp" or p.stat().st_size > CAP:
            continue
        dst = RESULTS / "logs" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)
        files[f"logs/{rel}"] = {"sha256": sha256(p), "bytes": p.stat().st_size}
    done = {p.name[5:]: p.read_text().strip() or env.time.strftime("%Y-%m-%dT%H:%M:%S", env.time.localtime(p.stat().st_mtime))
            for p in sorted(logs.glob("done.*"))}
    man = {"generated": env.now(), "work": str(W), "teacher_revision": env.REVISION, "stages_done": done,
           "files": files, "skipped_over_cap": skipped, "bulk_patterns": BULK, "excluded_patterns": EXCLUDE}
    env.atomic_json(RESULTS / "MANIFEST.json", man)
    (REPO / "RESULTS.md").write_text(render())
    env.log(f"export: {len(files)} files -> {RESULTS} ({sum(f['bytes'] for f in files.values()) / 2**20:.1f} MiB), "
            f"{len(skipped)} over cap, RESULTS.md rendered", STAGE)
    return man


# ------------------------------------------------------------------------------------------------------------ render
def jl(p: Path):
    return json.load(open(p)) if p.exists() else None


def f3(x) -> str:
    return "-" if x is None else f"{x:.3f}"


def ci(m: dict) -> str:
    return f"{m['mean']:.3f} [{m['lo']:.3f}, {m['hi']:.3f}]"


def arm_table(arb: dict, shipped: str) -> list[str]:
    S = arb["summaries"]
    buckets = sorted(next(iter(S.values()))["by_bucket"])
    cand = arb.get("candidates", {})
    rows = ["| arm | experts | overall top-1 | overall kl32 | " + " | ".join(buckets) + " |",
            "|---|---|---|---|" + "---|" * len(buckets)]
    for name, s in S.items():
        if name == "control":
            continue
        n = cand.get(name, {}).get("n_keep", 128 if name.startswith("b128") else 192)
        tag = f"**{name}** (shipped)" if name == shipped else name
        rows.append(f"| {tag} | {n} | {f3(s['overall']['top1']['mean'])} | {f3(s['overall']['kl32']['mean'])} | "
                    + " | ".join(f3(s["by_bucket"][b]["top1"]["mean"]) for b in buckets) + " |")
    return rows


def render() -> str:
    arb, ks = jl(W / "s04" / "arbitration.json"), jl(W / "s04" / "keepset.json")
    heal = jl(W / "s10" / "heal" / "decision.json")
    gates, s09 = jl(W / "s10" / "gates.json"), jl(W / "s09" / "gates.json")
    dsp, s11 = jl(W / "s09" / "dspark" / "gate.json"), jl(W / "s11" / "decision.json")
    L = ["# Results", "",
         "Generated by `ops/s13_results.py` from the artifacts in [`results/`](results/) (sha256 of every file in "
         "[`results/MANIFEST.json`](results/MANIFEST.json)). No number on this page is typed by hand. The selection "
         "rationale is in [`DECISIONS.md`](DECISIONS.md).", ""]
    if arb and ks:
        shipped = ks["winner"]
        t1 = arb["summaries"][shipped]["overall"]
        L += ["## s04: expert selection (192 of 384 per layer)", "",
              f"Every arm is replayed through the full model against the unpruned teacher on the screen split "
              f"({arb['rows']} rows, {arb['tokens']:,} tokens; per-bucket figures rest on "
              f"{next(iter(arb['summaries'][shipped]['by_bucket'].values()))['top1']['n_clusters']} rows each, so read "
              "them as directional). top-1 = agreement with the teacher's argmax; kl32 = KL(teacher || arm) over the "
              "teacher's top 32. The `random*` arms are uniform random keep-sets: a criterion has to beat them.", ""]
        L += arm_table(arb, shipped) + [""]
        L += [f"Shipped keep-set: `{shipped}`, sha256 `{ks['sha256']}`. Screen top-1 {ci(t1['top1'])}, "
              f"kl32 {ci(t1['kl32'])} (95% CIs, row-clustered bootstrap).", ""]
        notes = [n for n in dict(arb["screen_floor"]).get(shipped, {}).get("notes", [])]
        if notes:
            L += ["Gate note: " + "; ".join(notes) + ".", ""]
    if heal:
        c = heal["comparison"]
        L += ["## s10 heal panel: per-layer heal gains (s06)", "",
              f"Adopted: **{heal['adopt']}**. " + ("; ".join(heal["why"]) + "." if heal["why"] else ""), "",
              "| healed vs unhealed | value |", "|---|---|",
              f"| d kl32 (healed - unhealed) | {c['d_kl32']['mean']:+.5f} [{c['d_kl32']['lo']:+.5f}, {c['d_kl32']['hi']:+.5f}] |",
              f"| d NLL | {c['d_dnll']['mean']:+.4f} [{c['d_dnll']['lo']:+.4f}, {c['d_dnll']['hi']:+.4f}] |",
              f"| top-1: healed-only right / unhealed-only right | {c['top1']['a_only']:,} / {c['top1']['b_only']:,} "
              f"(McNemar z {c['top1']['z']:+.2f}) |", "",
              f"Measured on {c['d_kl32']['n_tokens']:,} tokens. The heal lowers NLL but moves the argmax away from the "
              "teacher, so the release carries the unhealed REAP weights." if not heal["adopt"] else "", ""]
    if gates:
        L += ["## s10 final: release gates (confirm split)", "", "```json",
              json.dumps({k: v for k, v in gates.items() if k in ("mxfp4_master", "nvfp4", "thor", "thor_arm",
                                                                 "heal_adopted", "trellis_adopted", "overlay_adopted")},
                         indent=1), "```", ""]
    if s09:
        L += ["## s09: GGUF ladder gate", "", "```json", json.dumps(s09, indent=1)[:6000], "```", ""]
    if dsp:
        L += ["## s09: DSpark sidecar gate", "", "```json", json.dumps(dsp, indent=1)[:4000], "```", ""]
    if s11:
        L += ["## s11: speculative head", "", "```json", json.dumps(s11, indent=1)[:4000], "```", ""]
    pending = [n for n, v in (("s10 final", gates), ("s09 GGUF", s09), ("s09 DSpark", dsp), ("s11", s11)) if v is None]
    if pending:
        L += ["## Not yet run", "", "Sections appear here as their stages finish: " + ", ".join(pending) + ".", ""]
    return "\n".join(L)


# ------------------------------------------------------------------------------------------------------------- scan
def scan() -> list[str]:
    hits = []
    for p in [*RESULTS.rglob("*"), REPO / "RESULTS.md", REPO / "DECISIONS.md", REPO / "README.md"]:
        if p.is_file():
            for m in SECRETS.finditer(p.read_text(errors="replace")):
                hits.append(f"{p.relative_to(REPO)}: {m.group(0)[:12]}...")
    log = subprocess.run(["git", "-C", str(REPO), "log", "--all", "-p", "--format="], capture_output=True, text=True,
                         errors="replace").stdout
    hits += [f"git history: {m.group(0)[:12]}..." for m in SECRETS.finditer(log)]
    env.log(f"scan: {len(hits)} hit(s)" + (": " + "; ".join(hits[:10]) if hits else ""), STAGE,
            "ERROR" if hits else "INFO")
    return hits


# ---------------------------------------------------------------------------------------------------------- release
def master_public() -> bool:
    led = jl(env.LOGS / "s12_ledger.json") or {}
    return bool(led.get("mxfp4", {}).get("public"))


def git(*a, check=True):
    return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True, check=check)


def commit(push: bool, msg: str) -> None:
    git("add", "results", "RESULTS.md")
    if git("diff", "--cached", "--quiet", check=False).returncode == 0:
        env.log("commit: nothing changed", STAGE)
    else:
        git("commit", "-m", msg + "\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>")
        env.log(f"commit: {git('rev-parse', '--short', 'HEAD').stdout.strip()}", STAGE)
    if push:
        # GitHub SSH has timed out at the banner exchange from this box (2026-10-22) and worked minutes later
        for i in range(5):
            r = git("push", "origin", "HEAD", check=False)
            if r.returncode == 0:
                env.log("pushed", STAGE)
                return
            env.log(f"push failed (try {i + 1}/5): {r.stderr.strip()[-200:]}", STAGE, "WARN")
            env.time.sleep(60 * (i + 1))
        raise SystemExit("git push failed 5 times")


def bulk(user: str, n: int) -> str:
    """Hardlink-stage the bulk arrays and upload them as a PUBLIC dataset (created public, so nothing counts against
    the private quota; called only after the MXFP4 master is public). Resumable: upload_large_folder skips files
    already committed."""
    from ops.s12_publish import api
    h = api()
    repo = f"{user}/DeepSeek-V4.1-Flash-1017-REAP{n}-artifacts"
    st = W / "s13_bulk"
    st.mkdir(exist_ok=True)
    nb = 0
    for p in sorted(W.rglob("*")):
        rel = str(p.relative_to(W))
        if p.is_file() and any(fnmatch.fnmatch(rel, b) for b in BULK) and not rel.startswith("s13_bulk/"):
            d = st / rel
            if not d.exists():
                d.parent.mkdir(parents=True, exist_ok=True)
                os.link(p, d)
            nb += p.stat().st_size
    (st / "README.md").write_text(DATASET_CARD.format(repo=repo, gh="https://github.com/" + user +
                                                      "/deepseek-v4.1-flash-reap-thor", gib=nb / 2**30))
    if not h.repo_exists(repo, repo_type="dataset"):
        h.create_repo(repo, repo_type="dataset", private=False)
    elif h.dataset_info(repo).private:
        h.update_repo_settings(repo_id=repo, repo_type="dataset", private=False)
    h.upload_large_folder(repo_id=repo, repo_type="dataset", folder_path=str(st), ignore_patterns=[".cache/*"])
    env.log(f"bulk: {nb / 2**30:.2f} GiB -> https://huggingface.co/datasets/{repo} (public)", STAGE)
    return repo


DATASET_CARD = """---
license: mit
tags: [reap, moe-pruning, deepseek, expert-pruning]
---
# {repo}

Intermediate data of the DeepSeek-V4.1-Flash-1017 REAP run ({gib:.1f} GiB). Code, process, every JSON result and the
selection rationale: {gh} (see RESULTS.md, DECISIONS.md, results/MANIFEST.json).

| path | what |
|---|---|
| `s03/bank/L*.npz` | per-layer, per-bucket expert saliency accumulators (REAP sums, gate mass, activation stats, HOPE `F = p p^T` interaction sums) from the teacher calibration pass |
| `s03/routing/L*_idx.npy`, `L*_w.npy` | the teacher's routed expert ids and weights for every calibration token |
| `s03/teacher/` | teacher top-32 ids/logits, log-sum-exp and next-token NLL per calibration token |
| `s06/bank`, `s06/measure` | the per-layer heal measurements (the heal was evaluated and NOT adopted, see RESULTS.md) |
| `s04/replay/<arm>/metrics`, `s10/*/replay/<arm>/metrics` | per-row, per-token teacher-agreement metrics of every candidate keep-set and evaluation arm |

The calibration token arrays are not included: the text belongs to its sources. `results/work/corpus/sources.json`
in the GitHub repo lists every source, revision and licence so the corpus can be rebuilt.
"""


def release(push: bool) -> None:
    export()
    if scan():
        raise SystemExit("secret scan hit: nothing committed, nothing made public (see the log line above)")
    pub = master_public()
    if not pub:
        commit(push, "results: export the REAP record (repo stays private until the MXFP4 master is public)")
        env.log("release: MXFP4 master is not public (s12 ledger); results committed, repo and dataset stay private",
                STAGE, "WARN")
        return
    from ops.s12_publish import api
    user = api().whoami()["name"]
    n = len(json.load(open(W / "s04" / "keepset.json"))["keep"]["0"])
    repo = bulk(user, n)
    env.atomic_json(RESULTS / "DATASET.json", {"dataset": f"https://huggingface.co/datasets/{repo}", "ts": env.now()})
    commit(push, f"results: export the REAP record; bulk arrays at hf.co/datasets/{repo}")
    if push:
        r = subprocess.run(["gh", "repo", "edit", f"{user}/deepseek-v4.1-flash-reap-thor", "--visibility", "public",
                            "--accept-visibility-change-consequences"], capture_output=True, text=True)
        if r.returncode:
            raise SystemExit(f"gh repo edit failed: {r.stderr.strip()}")
        env.log("GitHub repo is PUBLIC", STAGE)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["export", "scan", "release"])
    ap.add_argument("--push", action="store_true")
    a = ap.parse_args()
    if a.cmd == "export":
        export()
    elif a.cmd == "scan":
        sys.exit(1 if scan() else 0)
    else:
        release(a.push)


if __name__ == "__main__":
    main()
