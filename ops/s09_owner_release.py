#!/usr/bin/env python3
"""Owner rule (2026-10-25/26): "if it's a near tie failing the floor, probably still upload it; else debug then upload",
and "make sure that we proceed for certain to ... gguf release". Run by chain2 only when NO level passed s09_gate.
Releases the best level that failed ONLY on the KL/top-1 floors (measured, smoke passed, keep-set consistent): those are
the near-misses. A level whose llama-server smoke failed (broken generation) or that was never measured is never
released. Writes passing=[level] plus an "owner_release" record into gates.json; s12_publish renders the card's gate
status as FAIL-on-floor/released-by-owner, never as PASS. Exit 0 with the level on stdout, exit 1 if none qualifies."""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ops.lib import env

p = env.WORK / "s09" / "gates.json"
g = json.load(open(p))
if g["passing"]:
    print(" ".join(g["passing"])); sys.exit(0)
floor = ("kl32 ", "top-1 ")
ok = {lv: r for lv, r in g["levels"].items()
      if r.get("kl32") is not None and r["why"] and all(w.startswith(floor) for w in r["why"])}
if not ok:
    print("no level qualifies (every failing level failed smoke, measurement or keep-set, not just the floor)", file=sys.stderr)
    sys.exit(1)
lv = min(ok, key=lambda k: ok[k]["kl32"])
g["passing"] = [lv]
g["owner_release"] = {"level": lv, "why_failed": ok[lv]["why"], "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                      "rule": "owner 2026-10-25/26: a near-miss on the pre-registered floor is released anyway; "
                              "smoke (generation) must still pass"}
json.dump(g, open(p, "w"), indent=1)
print(lv)
