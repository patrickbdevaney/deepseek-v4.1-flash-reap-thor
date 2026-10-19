#!/usr/bin/env python3
"""s09d: render a GGUF level's model card from llamacpp/MODEL_CARD_TEMPLATE.md. Every number comes from an artifact.

  python ops/s09_card.py LEVEL --s10 S10_RESULT.json [--out README.md]
  python ops/s09_card.py LEVEL --draft            (s10 not in yet: DRAFT banner, never publishable)

Sources:
  * $V41_WORK/s09/LEVEL/plan.json: types, bpw and protected layers;
  * the split files: size and count;
  * $V41_WORK/s09/LEVEL/teacher_eval/teacher_kld.json: paired KL / top-1 / dNLL, plus served identity;
  * the s10 result JSON. If it has a "verdict" for this level, the verdict must be PASS. Its numeric leaves
    under "summary" (or the top level) become the s10 table.
The renderer refuses to write when:
  * any {{field}} is unresolved;
  * the KL result's keep-set differs from the GGUF's;
  * the s10 verdict for this level is not PASS.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ops.lib import env  # noqa: E402
from ops import s09_gguf_ladder as lad  # noqa: E402

TEMPLATE = env.REPO / "llamacpp" / "MODEL_CARD_TEMPLATE.md"


def fmt(v):
    if isinstance(v, float):
        return f"{v:.4f}"
    if isinstance(v, list) and all(isinstance(x, float) for x in v):
        return "[" + ", ".join(f"{x:.4f}" for x in v) + "]"
    return str(v)


def leaves(d: dict, prefix=""):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from leaves(v, f"{prefix}{k}.")
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            yield f"{prefix}{k}", v


def render(level: str, s10: Path | None, draft: bool) -> str:
    d = lad.OUT / level
    plan = json.loads((d / "plan.json").read_text())
    kld = json.loads((d / "teacher_eval" / "teacher_kld.json").read_text())
    parts = sorted(d.glob(f"{lad.NAME}-{level}-0*-of-*.gguf"))
    if not parts:
        raise SystemExit(f"FATAL: no splits for {level}")
    gkv = lad.gguf_kv(parts[0])
    fp = gkv.get("deepseek41.reap.keep_set_sha256")
    if kld["identity"]["keep_set_sha256"] != fp:
        raise SystemExit(f"FATAL: KL result measured keep-set {kld['identity']['keep_set_sha256']}, GGUF has {fp}")
    if Path(kld["identity"]["gguf"]).name != parts[0].name:
        raise SystemExit(f"FATAL: KL result measured {kld['identity']['gguf']}, not {parts[0].name}")

    prot = plan["protected_layers"]
    unprot = collections.Counter(t for k, t in plan["expert_types"].items() if int(k.split(".")[0]) not in prot)
    prot_t = sorted({t for k, t in plan["expert_types"].items() if int(k.split(".")[0]) in prot})
    s10_table = "_s10 results pending: DRAFT, not for publication._"
    if s10 is not None:
        r = json.loads(s10.read_text())
        verdict = (r.get("verdict") or {}).get(level) if isinstance(r.get("verdict"), dict) else r.get("verdict")
        if verdict is not None and verdict != "PASS":
            raise SystemExit(f"FATAL: s10 verdict for {level} is {verdict}; no card for a failed gate")
        rows = list(leaves(r.get("summary", r)))
        if not rows:
            raise SystemExit("FATAL: s10 result has no numeric fields")
        s10_table = "### s10 evaluation\n\n| metric | value |\n|---|---|\n" + "".join(f"| {k} | {fmt(v)} |\n" for k, v in rows)
    elif not draft:
        raise SystemExit("FATAL: --s10 RESULT required (or --draft)")

    remote = subprocess.run(["git", "-C", str(env.REPO), "remote", "get-url", "origin"], capture_output=True, text=True).stdout.strip()
    repo_url = re.sub(r"^git@github\.com:", "https://github.com/", remote).removesuffix(".git")
    fork_sha = subprocess.run(["git", "-C", str(env.REPO / "llamacpp" / "llama.cpp"), "rev-parse", "--short=12", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
    ctx = {
        "level": level, "keep_set_sha256": fp,
        "size_gib": f"{sum(p.stat().st_size for p in parts) / 2**30:.1f}", "n_splits": len(parts),
        "expert_bpw": f"{plan['expert_bpw']:.2f}",
        "expert_type_summary": ", ".join(f"{t} ({n} tensor groups)" for t, n in unprot.most_common()) or "-",
        "protected_layers": ", ".join(map(str, prot)), "protected_type": ", ".join(prot_t),
        "kld": kld, "s10_table": s10_table, "repo_url": repo_url, "first_split": parts[0].name,
        "source_revision": gkv.get("deepseek41.reap.source_revision") or env.REVISION, "fork_sha": fork_sha,
    }

    def sub(m):
        cur = ctx
        for part in m.group(1).split("."):
            if not isinstance(cur, dict) or part not in cur:
                raise SystemExit(f"FATAL: unresolved card field {{{{{m.group(1)}}}}}")
            cur = cur[part]
        return fmt(cur)

    out = re.sub(r"\{\{([a-zA-Z0-9_.]+)\}\}", sub, TEMPLATE.read_text())
    if draft and s10 is None:
        out = out.replace("---\n\n", "---\n\n> **DRAFT: s10 gates not run. Do not publish.**\n\n", 1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("level", choices=sorted(lad.LEVELS))
    ap.add_argument("--s10", type=Path)
    ap.add_argument("--draft", action="store_true")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    text = render(a.level, a.s10, a.draft)
    out = a.out or lad.OUT / a.level / "README.md"
    out.write_text(text)
    print(f"[{env.now()}] card -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
