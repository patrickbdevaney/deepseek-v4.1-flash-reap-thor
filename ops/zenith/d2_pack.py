#!/usr/bin/env python3
"""D2: v41_decode --trace-gen output -> s11's tap layout (ops/s11_spec.py Taps).

    python3 ops/zenith/d2_pack.py ~/models/v41-work/d2/trace_train ~/models/v41-work/d2/taps_train

IN : taps.bin (bf16 [T, 3, dim]), tokens.bin (int32 [T]), seqs.jsonl {id, offset, length, gen_start, ...}
OUT: L37.npy L38.npy L39.npy (uint16 bf16 bits [T, dim]), tokens.npy (int32 [T]),
     meta.json {complete, taps, seqs: [{offset, length, gen_start, row, id, domain, mode}], source}
Only complete seqs (meta line present) are packed; a run still in progress packs what it has.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

DIM, TAPS = 5120, [37, 38, 39]


def main() -> None:
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    seqs = [json.loads(l) for l in open(src / "seqs.jsonl") if l.strip()]
    T = max((s["offset"] + s["length"] for s in seqs), default=0)
    taps = np.memmap(src / "taps.bin", np.uint16, "r", shape=(T, 3, DIM))
    toks = np.fromfile(src / "tokens.bin", np.int32, count=T)
    if toks.size != T:
        raise SystemExit(f"tokens.bin has {toks.size} rows, seqs say {T}")
    dst.mkdir(parents=True, exist_ok=True)
    for k, L in enumerate(TAPS):
        out = np.lib.format.open_memmap(dst / f"L{L}.npy", "w+", np.uint16, (T, DIM))
        for a in range(0, T, 4096):
            out[a:a + 4096] = taps[a:a + 4096, k]
        out.flush(); del out
    np.save(dst / "tokens.npy", toks)
    meta = {"complete": True, "taps": TAPS, "source": str(src), "rows": T,
            "seqs": [{"offset": s["offset"], "length": s["length"], "gen_start": s["gen_start"], "row": i,
                      "id": s["id"], "domain": s.get("domain"), "mode": s.get("mode")} for i, s in enumerate(seqs)]}
    json.dump(meta, open(dst / "meta.json", "w"), indent=0)
    print(f"[d2_pack] {len(seqs)} seqs, {T} rows -> {dst}")


if __name__ == "__main__":
    main()
