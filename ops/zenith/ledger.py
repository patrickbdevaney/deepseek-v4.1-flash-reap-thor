#!/usr/bin/env python3
"""ZENITH ledger: an nsys trace of v41_decode -> per-family decode-step time, bytes, roofline floor, gap.

    ops/zenith/ledger.py ops/logs/prof/ov2.sqlite [--overlay] [--bw 228] [--out docs/zenith/LEDGER.md]

Decode steps are counted as M=1 head launches (k_dense<DecBF16P|DecBF16, 1, ..> with the vocab grid);
prefill kernels (MT>1 instantiations, M>1 grids) are reported separately and excluded from the step.
Bytes per family are exact from the model shapes (docs/DECODE_PARITY_0731.md §3).
"""
import argparse
import collections
import sqlite3
import sys

# bytes/token by family, MB (DECODE_PARITY_0731.md §3; overlay values where the overlay replaces FP8)
BYTES = {
    "nvfp4_dense": 3672.0,  # MLA 2848.8 + shared 796.3 + indexer 26.7
    "fp8_dense": 315.0,  # Engram wkv (+ MLA/shared when no overlay, see --no-overlay)
    "kquant_experts": 3002.6,
    "bf16_head": 1323.8,
    "sparse_attn": 25.0,
    "glue": 200.0,  # HC fn 157 + router 79 + norms (rough)
    "topk_index": 30.0,
}
BYTES_NO_OVERLAY = dict(BYTES, nvfp4_dense=0.0, fp8_dense=315.0 + 5069.6 + 1417.0 + 45.1)


def family(name: str) -> str:
    if "k_dense" in name:
        if "DecNVFP4" in name:
            return "nvfp4_dense"
        if "DecFP8" in name:
            return "fp8_dense"
        if "DecBF16" in name:
            return "bf16_head"
        return "dense_other"
    if "k_kq_soa" in name:
        return "load"
    if "k_kq" in name or "k_experts" in name:
        return "kquant_experts"
    if "k_sa_" in name or "k_sparse_attn" in name:
        return "sparse_attn"
    if "k_topk" in name or "k_index" in name or "k_cand" in name:
        return "topk_index"
    return "glue"


def template_args(name: str) -> list:
    """top-level template arguments of the kernel name (k_dense<Dec, MT, RB, LPR> -> [Dec, MT, RB, LPR])"""
    i = name.find("<")
    if i < 0:
        return []
    out, depth, cur = [], 0, ""
    for ch in name[i + 1:]:
        if ch == "<":
            depth += 1
        elif ch == ">":
            if depth == 0:
                out.append(cur.strip())
                break
            depth -= 1
        elif ch == "," and depth == 0:
            out.append(cur.strip()); cur = ""; continue
        cur += ch
    return out


def is_prefill(name: str, grid_y: int) -> bool:
    # MT > 1 instantiations of k_dense / k_kq / k_experts serve M > 1 (prefill or verify); decode uses MT = 1
    if not any(k in name for k in ("k_dense<", "k_kq<", "k_experts<")):
        return False
    a = template_args(name)
    return len(a) >= 2 and a[1].replace("(int)", "").strip() != "1"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite")
    ap.add_argument("--bw", type=float, default=228.0, help="measured streaming GB/s")
    ap.add_argument("--no-overlay", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()
    db = sqlite3.connect(a.sqlite)
    q = ("select s.value, k.gridX, k.gridY, k.end - k.start, k.start, k.end from CUPTI_ACTIVITY_KIND_KERNEL k "
         "join StringIds s on k.demangledName = s.id order by k.start")
    rows = list(db.execute(q))
    # steps: M=1 head launches (vocab 129280 / 64 rows per CTA = 2020 CTAs)
    steps = sum(1 for n, gx, gy, d, s, e in rows if "DecBF16" in n and gx == 2020 and gy == 1)
    if not steps:
        print("no M=1 head launches found", file=sys.stderr)
        return 1
    t = collections.defaultdict(float)
    pre = collections.defaultdict(float)
    for n, gx, gy, d, s, e in rows:
        f = family(n)
        if f != "load":
            (pre if is_prefill(n, gy) else t)[f] += d / 1e6
    # wall: first to last M=1 head launch spans the decode
    heads = [(s, e) for n, gx, gy, d, s, e in rows if "DecBF16" in n and gx == 2020 and gy == 1]
    wall_ms = (heads[-1][1] - heads[0][1]) / 1e6 / max(1, steps - 1)
    B = BYTES_NO_OVERLAY if a.no_overlay else BYTES
    lines = [f"# ZENITH ledger ({a.sqlite})", "",
             f"{steps} decode steps; wall {wall_ms:.2f} ms/step = {1000 / wall_ms:.2f} tok/s; BW_peak {a.bw:.0f} GB/s", "",
             "| family | t ms/step | MB/step | floor ms | gap ms | achieved GB/s |", "|---|---:|---:|---:|---:|---:|"]
    tot_t = tot_b = tot_gap = 0.0
    for f in sorted(t, key=lambda f: -t[f]):
        ms = t[f] / steps
        mb = B.get(f, 0.0)
        floor = mb / a.bw
        gap = max(0.0, ms - floor)
        tot_t += ms; tot_b += mb; tot_gap += gap
        gbs = mb / ms if ms > 0 else 0.0
        lines.append(f"| {f} | {ms:.2f} | {mb:.0f} | {floor:.2f} | {gap:.2f} | {gbs:.0f} |")
    idle = max(0.0, wall_ms - tot_t)
    lines.append(f"| launch/sync idle | {idle:.2f} | | | {idle:.2f} | |")
    floor_all = tot_b / a.bw
    lines += [f"| **total** | **{wall_ms:.2f}** | **{tot_b:.0f}** | **{floor_all:.2f}** | **{tot_gap + idle:.2f}** | |", "",
              f"AR zenith at these bytes: {floor_all:.1f} ms = {1000 / floor_all:.1f} tok/s; "
              f"today {100 * floor_all / wall_ms:.0f}% of roofline.", "",
              "Prefill/verify (MT>1) kernel time in the trace, not in the step: " +
              ", ".join(f"{f} {v:.0f} ms" for f, v in sorted(pre.items(), key=lambda kv: -kv[1]))]
    out = "\n".join(lines) + "\n"
    if a.out:
        open(a.out, "w").write(out)
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
