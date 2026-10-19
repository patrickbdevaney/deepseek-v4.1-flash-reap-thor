"""Reader/writer for the Thor build directory (spec: ops/lib/thor_format.md). numpy only."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from . import ggml_quants as G
from . import st_io

FORMAT = "thor-v1"
PROJ = ("w1", "w2", "w3")


def layer_file(L: int) -> str:
    return f"layer-{L:02d}.safetensors"


def expert_name(L: int, proj: str, part: str = "qweight") -> str:
    """Stacked routed experts of one layer and projection: part = qweight (ggml blocks, U8
    [n_experts, out, row_bytes]) or weight / scale (MXFP4_SRC: U8 [n, out, in/2] / [n, out, in/32])."""
    return f"layers.{L}.ffn.experts.{proj}.{part}"


def sha256_file(p: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


class Build:
    def __init__(self, root: Path, *, create: bool = False, **meta):
        self.root = Path(root)
        self.mpath = self.root / "thor.json"
        if self.mpath.exists():
            self.m = json.load(open(self.mpath))
            for k, v in meta.items():
                if k in ("keepset_sha256", "n_routed_experts") and self.m.get(k) != v:
                    from .keepset import KeepsetMismatch
                    raise KeepsetMismatch(f"{self.root}: built for {k}={self.m.get(k)}, asked for {v}")
        elif create:
            self.root.mkdir(parents=True, exist_ok=True)
            self.m = {"format": FORMAT, **meta, "files": {}, "tensors": {}, "complete": False}
        else:
            raise FileNotFoundError(self.mpath)

    def save(self) -> None:
        from .env import atomic_json
        atomic_json(self.mpath, self.m)

    def write_file(self, fname: str, items: list[dict], meta: dict | None = None) -> None:
        """items: {name, dtype (safetensors), data (np array), qtype (optional), shape (logical,
        optional)}. Replaces the file and its manifest entries atomically w.r.t. the manifest."""
        out = self.root / fname
        st_io.write(out, [(i["name"], i["dtype"], i["data"]) for i in items],
                    metadata={"format": FORMAT, **(meta or {})})
        for name in [n for n, t in self.m["tensors"].items() if t["file"] == fname]:
            del self.m["tensors"][name]
        for i in items:
            self.m["tensors"][i["name"]] = {"file": fname, "dtype": i["dtype"], "qtype": i.get("qtype"),
                                            "stored_shape": list(i["data"].shape),
                                            "shape": list(i.get("shape") or i["data"].shape)}
        self.m["files"][fname] = {"bytes": out.stat().st_size, "sha256": sha256_file(out)}
        self.save()

    def has(self, fname: str) -> bool:
        f = self.m["files"].get(fname)
        return bool(f) and (self.root / fname).exists() and (self.root / fname).stat().st_size == f["bytes"]

    def reader(self, fname: str) -> st_io.Reader:
        return st_io.Reader(self.root / fname)

    def expert_bytes(self) -> int:
        return sum(np.prod(t["stored_shape"]) * np.dtype(st_io.DT[t["dtype"]]).itemsize
                   for n, t in self.m["tensors"].items() if ".ffn.experts." in n)

    def verify(self) -> list[str]:
        """sha256 of every file against the manifest -> list of bad files."""
        return [f for f, r in self.m["files"].items()
                if not (self.root / f).exists() or sha256_file(self.root / f) != r["sha256"]]


def dequant_stack(qtype: str, arr: np.ndarray, out_f: int, in_f: int, scale: np.ndarray | None = None,
                  idx: int | None = None, sign: np.ndarray | None = None) -> np.ndarray:
    """One expert (idx) or all of a stacked tensor -> float32 [.., out, in]. KT2/KT3 take the
    `kt_scale` tensor as `scale` and the pass's packed `rht_sign` as `sign` (un-rotated output)."""
    a = arr if idx is None else arr[idx]
    if qtype in ("KT2", "KT3"):
        from . import kt
        if scale is None or sign is None:
            raise ValueError(f"{qtype} needs its kt_scale and rht_sign tensors")
        s = scale if idx is None else scale[idx]
        return kt.dequant(qtype, np.asarray(a), np.asarray(s), in_f, kt.unpack_signs(sign, in_f)).reshape(*a.shape[:-1], in_f)
    if qtype == "MXFP4_SRC":
        s = scale if idx is None else scale[idx]
        lead = a.shape[:-2]
        w = a.reshape(-1, a.shape[-1])
        y = G.src_dequant(w, s.reshape(-1, s.shape[-1]))
        return y.reshape(*lead, out_f, in_f)
    return G.dequant(qtype, np.asarray(a)).reshape(*a.shape[:-1], in_f)
