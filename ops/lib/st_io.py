"""Minimal safetensors reader/writer in numpy (no torch, no safetensors package needed).

Why not the safetensors package: the writers here stream tensors that are already raw bytes (FP8,
E8M0, packed FP4, ggml blocks), the tests must run with system python (no safetensors there), and
the s08 CUDA server needs ALIGNED tensors. On this project a safetensors file whose data started
at a 4-byte boundary broke vectorised loads (memory: glm53-cuda-server-build). The format forbids
holes in the byte buffer, so alignment is done with explicit padding tensors named
`__pad.<n>` (U8 [k]) that every consumer must skip, and with a header padded by spaces.
Every tensor here starts at a multiple of ALIGN bytes from the start of the FILE.
"""
from __future__ import annotations

import json
import os
import struct
from pathlib import Path

import numpy as np

ALIGN = 256
# safetensors dtype -> numpy storage dtype (8-bit floats and bf16 are carried as raw integers)
DT = {"F64": np.float64, "F32": np.float32, "F16": np.float16, "BF16": np.uint16, "F8_E4M3": np.uint8,
      "F8_E8M0": np.uint8, "I64": np.int64, "I32": np.int32, "I16": np.int16, "I8": np.int8, "U8": np.uint8,
      "U16": np.uint16, "U32": np.uint32}


def write(path: Path, tensors: list[tuple[str, str, np.ndarray]], metadata: dict | None = None, *,
          align: int = ALIGN) -> dict:
    """tensors: (name, safetensors dtype, array whose bytes are the tensor). Returns {name: (start, end)}
    as absolute file offsets. Atomic (tmp + rename + fsync).

    align=1 writes a plain safetensors file with no padding tensors: use it for HF artifacts loaded
    by vLLM/transformers, whose loaders reject unknown tensor names. The header is always padded to
    8 bytes, as the format recommends."""
    path = Path(path)
    names = [n for n, _, _ in tensors]
    if len(set(names)) != len(names):
        raise ValueError("duplicate tensor names")
    entries, layout, off, pad_i = {}, [], 0, 0
    # pass 1: offsets relative to the data section; the header length is fixed afterwards and the
    # data section start is padded to ALIGN, so relative alignment == absolute alignment.
    for name, dt, arr in tensors:
        a = np.ascontiguousarray(arr).reshape(np.shape(arr))      # ascontiguousarray makes 0-d arrays 1-d
        if a.dtype != np.dtype(DT[dt]) and not (dt in ("F8_E4M3", "F8_E8M0", "U8") and a.dtype == np.int8):
            raise TypeError(f"{name}: array dtype {a.dtype} does not carry safetensors {dt}")
        if align > 1 and off % align:
            k = align - off % align
            pn = f"__pad.{pad_i}"; pad_i += 1
            entries[pn] = {"dtype": "U8", "shape": [k], "data_offsets": [off, off + k]}
            layout.append((pn, np.zeros(k, np.uint8)))
            off += k
        nb = a.nbytes
        entries[name] = {"dtype": dt, "shape": list(a.shape), "data_offsets": [off, off + nb]}
        layout.append((name, a))
        off += nb
    hdr = dict(entries)
    if metadata:
        hdr["__metadata__"] = {k: (v if isinstance(v, str) else json.dumps(v)) for k, v in metadata.items()}
    hj = json.dumps(hdr, separators=(",", ":")).encode()
    total = 8 + len(hj)
    hj += b" " * ((-total) % max(align, 8))
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as fh:
        fh.write(struct.pack("<Q", len(hj)))
        fh.write(hj)
        for _, a in layout:
            fh.write(memoryview(a).cast("B") if a.flags.c_contiguous else a.tobytes())
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    base = 8 + len(hj)
    return {n: (base + e["data_offsets"][0], base + e["data_offsets"][1]) for n, e in entries.items()}


def header(path: Path) -> tuple[dict, int]:
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        h = json.loads(fh.read(n))
    return h, 8 + n


class Reader:
    """Memory-mapped tensors of one file. get(name) -> numpy array in the storage dtype of DT."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.h, self.base = header(self.path)
        self.meta = self.h.pop("__metadata__", {})
        self._mm = np.memmap(self.path, dtype=np.uint8, mode="r")

    def names(self) -> list[str]:
        return [n for n in self.h if not n.startswith("__pad.")]

    def dtype(self, name: str) -> str:
        return self.h[name]["dtype"]

    def get(self, name: str) -> np.ndarray:
        e = self.h[name]
        a, b = e["data_offsets"]
        raw = self._mm[self.base + a:self.base + b]
        return raw.view(DT[e["dtype"]]).reshape(e["shape"])

    def offset(self, name: str) -> int:
        return self.base + self.h[name]["data_offsets"][0]
