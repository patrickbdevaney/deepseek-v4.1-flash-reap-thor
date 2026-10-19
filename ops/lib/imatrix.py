"""Write a llama.cpp legacy (.dat) imatrix from the streamed per-expert column sums.

The saliency pass banks, per layer and per routed expert, sum(x^2) over every column of the input
to gate/up (5120 wide, shared by both) and to down (2304 wide), plus the token count. That is
everything llama-imatrix would have collected, without running llama-imatrix over a 475 GiB model.

Legacy layout (what ~/glm-5.3-reap/scripts/imatrix_verify.py checks):
    int32 n_entries
    n_entries x { int32 len, name bytes, int32 ncall, int32 nval, float32[nval] }
    int32 last_call, int32 len, dataset bytes
For a 3-D expert tensor nval = n_expert * n_per_row, experts concatenated in tensor order, and
each value is mean(x^2) over that expert's tokens times ncall: llama.cpp divides by ncall on load,
and an expert that saw no tokens is written as zeros, as llama-imatrix itself does.

`keep` maps the pruned tensor's expert order onto source expert ids, so the same bank serves the
192-, 128- or any other keep-set without a second pass.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np


def expert_entry(sumsq: np.ndarray, count: np.ndarray, keep: np.ndarray | None, ncall: int) -> np.ndarray:
    """sumsq [E, C] float64, count [E] -> flat float32 [len(keep) * C] of mean(x^2) * ncall."""
    if keep is not None:
        sumsq, count = sumsq[keep], count[keep]
    mean = np.divide(sumsq, count[:, None], out=np.zeros_like(sumsq, dtype=np.float64), where=count[:, None] > 0)
    return (mean * ncall).astype(np.float32).reshape(-1)


def write_dat(path: Path, entries: dict[str, tuple[int, np.ndarray]], dataset: str, last_call: int) -> None:
    """entries: gguf tensor name -> (ncall, float32 values). Written tmp + rename."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as fh:
        fh.write(struct.pack("<i", len(entries)))
        for name, (ncall, vals) in entries.items():
            nb = name.encode()
            v = np.ascontiguousarray(vals, dtype="<f4")
            fh.write(struct.pack("<i", len(nb)) + nb + struct.pack("<ii", ncall, v.size))
            fh.write(v.tobytes())
        db = dataset.encode()
        fh.write(struct.pack("<ii", last_call, len(db)) + db)
    tmp.replace(path)


def read_dat(path: Path) -> tuple[dict[str, tuple[int, np.ndarray]], int, str]:
    data = Path(path).read_bytes()
    off = 0

    def take(fmt):
        nonlocal off
        v = struct.unpack_from(fmt, data, off)
        off += struct.calcsize(fmt)
        return v

    (n,) = take("<i")
    out = {}
    for _ in range(n):
        (ln,) = take("<i")
        name = data[off:off + ln].decode(); off += ln
        ncall, nval = take("<ii")
        out[name] = (ncall, np.frombuffer(data, dtype="<f4", count=nval, offset=off).copy())
        off += 4 * nval
    last_call, ln = take("<ii")
    ds = data[off:off + ln].decode(); off += ln
    assert off == len(data), f"{len(data) - off} trailing bytes"
    return out, last_call, ds


def layer_entries(layer: int, bank: dict[str, np.ndarray], keep: np.ndarray | None, ncall: int) -> dict:
    """GGUF names for one layer's routed + shared experts from an s03 bank (see s03 `ffn_stats`)."""
    b = f"blk.{layer}."
    e = {
        b + "ffn_gate_exps.weight": (ncall, expert_entry(bank["x2_in"], bank["n_tok"], keep, ncall)),
        b + "ffn_up_exps.weight": (ncall, expert_entry(bank["x2_in"], bank["n_tok"], keep, ncall)),
        b + "ffn_down_exps.weight": (ncall, expert_entry(bank["x2_mid"], bank["n_tok"], keep, ncall)),
    }
    if "x2_in_sh" in bank:
        n = np.asarray(bank["n_tok_sh"], dtype=np.float64).reshape(1)
        e[b + "ffn_gate_shexp.weight"] = (ncall, expert_entry(bank["x2_in_sh"][None], n, None, ncall))
        e[b + "ffn_up_shexp.weight"] = (ncall, expert_entry(bank["x2_in_sh"][None], n, None, ncall))
        e[b + "ffn_down_shexp.weight"] = (ncall, expert_entry(bank["x2_mid_sh"][None], n, None, ncall))
    return e
