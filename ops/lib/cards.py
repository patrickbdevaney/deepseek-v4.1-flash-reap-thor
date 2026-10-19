"""Model-card rendering from pipeline artifacts. No number on a card is typed by hand.

Template syntax (ops/cards/*.md.tmpl):
  {{ dotted.path }}            value from the context (dict keys / list indices); missing -> error
  {{ dotted.path | f4 }}       filters: f2 f3 f4 (fixed decimals), pct (x100, 2 decimals, %), int,
                               short (first 16 chars), gib (bytes -> GiB, 1 decimal), json
  {{ table_buckets:ARM }}      per-bucket kl32 / top-1 / dNLL table of a final-panel arm
  {{ table_bins:ARM }}         the same per log-position bin
  {{ table_arms:A,B,... }}     overall table over several arms
  {{ table_buckets:@path }}    the arm name itself read from the context
  {{ include:NAME }}           the text of ops/cards/NAME.md.part (expanded before the lint)
LINT: the template's literal text may not contain a decimal number or a percentage. Every
measured quantity must arrive through a placeholder, so a stale literal cannot survive a re-run
(PLAN.md: "The card's numbers are filled from artifacts, never from literals").
"""
from __future__ import annotations

import json
import re

PH = re.compile(r"\{\{\s*([^}|]+?)\s*(?:\|\s*(\w+)\s*)?\}\}")
LITERAL_NUMBER = re.compile(r"(?<![\w.])\d+\.\d+(?![\w.])|\d+(?:\.\d+)?\s*%")


class CardError(ValueError):
    pass


def lint(template: str) -> list[str]:
    text = PH.sub("", template)
    return sorted(set(m.group(0) for m in LITERAL_NUMBER.finditer(text)))


def lookup(ctx: dict, path: str):
    cur = ctx
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        elif isinstance(cur, list) and part.lstrip("-").isdigit() and -len(cur) <= int(part) < len(cur):
            cur = cur[int(part)]
        else:
            raise CardError(f"card placeholder {path!r}: no {part!r}")
    return cur


def fmt(v, f: str | None) -> str:
    if v is None:
        raise CardError("placeholder resolved to None")
    if f is None:
        return str(v)
    if f in ("f2", "f3", "f4"):
        return f"{float(v):.{int(f[1])}f}"
    if f == "pct":
        return f"{100 * float(v):.2f}%"
    if f == "int":
        return str(int(v))
    if f == "short":
        return str(v)[:16]
    if f == "gib":
        return f"{float(v) / 2**30:.1f}"
    if f == "json":
        return json.dumps(v)
    raise CardError(f"unknown filter {f}")


def _ci(m: dict, f: str = "f4") -> str:
    return f"{fmt(m['mean'], f)} [{fmt(m['lo'], f)}, {fmt(m['hi'], f)}]"


def table(ctx: dict, kind: str, arg: str) -> str:
    S = lookup(ctx, "s10.summaries")
    if kind == "table_arms":
        rows = ["| arm | kl32 (95% CI) | top-1 agreement | dNLL | confident-flip rate |", "|---|---|---|---|---|"]
        for arm in arg.split(","):
            o = S[arm]["overall"]
            rows.append(f"| {arm} | {_ci(o['kl32'])} | {_ci(o['top1'])} | {_ci(o['dnll'])} | {fmt(o['confident_flip_rate'], 'f4')} |")
        return "\n".join(rows)
    key = {"table_buckets": "by_bucket", "table_bins": "by_pos_bin"}[kind]
    rows = [f"| {'domain' if key == 'by_bucket' else 'position'} | tokens | kl32 | top-1 agreement | dNLL |", "|---|---|---|---|---|"]
    for name, s in S[arg][key].items():
        flag = " (under-sampled)" if s["under_sampled"] else ""
        rows.append(f"| {name}{flag} | {s['kl32']['n_tokens']} | {_ci(s['kl32'])} | {_ci(s['top1'])} | {_ci(s['dnll'])} |")
    return "\n".join(rows)


INCLUDE = re.compile(r"\{\{\s*include:(\w+)\s*\}\}")


def expand(template: str, parts: dict[str, str]) -> str:
    return INCLUDE.sub(lambda m: parts[m.group(1)], template)


def render(template: str, ctx: dict, parts: dict[str, str] | None = None) -> str:
    template = expand(template, parts or {})
    bad = lint(template)
    if bad:
        raise CardError(f"template contains literal numbers {bad}; route them through the context")

    def sub(m):
        expr, f = m.group(1), m.group(2)
        if ":" in expr:
            kind, arg = expr.split(":", 1)
            arg = arg.strip()
            if arg.startswith("@"):
                arg = str(lookup(ctx, arg[1:]))
            return table(ctx, kind.strip(), arg)
        return fmt(lookup(ctx, expr), f)

    return PH.sub(sub, template)
