"""Gather Engram table rows for the calibration tokens without loading a 98 GB table.

Engram is a pure function of token ids: NgramHashState maps every position to 24 row ids per
Engram layer (3 n-gram sizes x 8 heads), and Engram.forward only ever reads those rows. So the
rows a calibration corpus needs can be gathered once, up front, by memory-mapping shards 47/48
at the tensor's safetensors offset and reading the sorted unique row ids. The saliency forward
then serves them from a small per-row file and never touches the table.

Per token per Engram layer: 24 rows x (256 B fp8 values + 8 B E8M0 scales, one per 32) = 6,336 B.
(Shard check: 384,006,168 rows x 264 B = 101.4 GB, the size of shard 47.)

No torch here: this file is imported by the pure tests and can run beside a GPU job.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import os

import numpy as np

ST_DTYPES = {"F8_E4M3": 1, "F8_E8M0": 1, "U8": 1, "I8": 1, "BF16": 2, "F16": 2, "F32": 4, "I32": 4, "I64": 8}


def safetensors_header(path: Path) -> tuple[dict, int]:
    """(header dict, absolute byte offset where tensor data begins)."""
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
    return hdr, 8 + n


def tensor_memmap(path: Path, name: str) -> np.memmap:
    """A read-only uint8 memmap [rows, row_bytes] over one 2-D tensor in a safetensors file."""
    hdr, base = safetensors_header(path)
    t = hdr[name]
    rows, cols = t["shape"]
    width = ST_DTYPES[t["dtype"]]
    a, b = t["data_offsets"]
    assert b - a == rows * cols * width, (name, t, b - a)
    return np.memmap(path, dtype=np.uint8, mode="r", offset=base + a, shape=(rows, cols * width))


def gather(table: np.memmap, ids: np.ndarray, *, chunk: int = 1 << 20) -> np.ndarray:
    """table[ids] for an int64 id array of any shape, reading each distinct row once in ascending
    order (the access pattern a spinning disk or USB drive serves best). Returns
    ids.shape + (row_bytes,) uint8."""
    flat = ids.reshape(-1)
    if flat.size and (flat.min() < 0 or flat.max() >= table.shape[0]):
        raise IndexError(f"row id out of range [0, {table.shape[0]}): {flat.min()}..{flat.max()}")
    uniq, inv = np.unique(flat, return_inverse=True)
    rows = np.empty((uniq.size, table.shape[1]), dtype=np.uint8)
    for s in range(0, uniq.size, chunk):
        rows[s:s + chunk] = table[uniq[s:s + chunk]]
    return rows[inv].reshape(*ids.shape, table.shape[1])


class EngramTables:
    """The two arrays one Engram layer reads: fp8 values [n, 256] and E8M0 scales [n, 8]."""

    def __init__(self, shard: Path, layer: int):
        p = f"layers.{layer}.engram.embed."
        self.values = tensor_memmap(shard, p + "weight")
        self.scales = tensor_memmap(shard, p + "scale")
        assert self.values.shape[0] == self.scales.shape[0]

    def gather(self, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return gather(self.values, ids), gather(self.scales, ids)


def read_rows(table: np.ndarray, a: int, b: int, *, chunk: int = 64 << 20) -> np.ndarray:
    """table[a:b] as an in-memory array. For a memmap this reads the byte range with large pread()s
    instead of faulting it in 4 KiB pages: over the NTFS USB drive (ntfs-3g, FUSE) page faults ran at
    ~8 MB/s, a 64 MiB pread loop at the drive's sequential rate (2026-10-19)."""
    if not isinstance(table, np.memmap) or table.filename is None:
        return np.asarray(table[a:b])
    width = table.shape[1] * table.itemsize
    buf = np.empty((b - a, table.shape[1]), dtype=table.dtype)
    mv = memoryview(buf.reshape(-1).view(np.uint8))
    off, n = table.offset + a * width, (b - a) * width
    fd = os.open(table.filename, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, off, n, os.POSIX_FADV_SEQUENTIAL)
        done = 0
        while done < n:
            k = os.preadv(fd, [mv[done:done + min(chunk, n - done)]], off + done)
            if k <= 0:
                raise IOError(f"short read at {off + done} in {table.filename}")
            done += k
    finally:
        os.close(fd)
    return buf


def gather_windowed(table: np.memmap, ids: np.ndarray, out: np.ndarray, *, window_rows: int = 1 << 22,
                    start_window: int = 0, on_window=None) -> None:
    """out[i] = table[ids[i]] for a flat id array, reading the table ONCE, sequentially, in windows
    of `window_rows` rows. Calibration ids are hashes, i.e. uniformly scattered over a 384M-row
    table, so a sorted random read would still touch most pages; one sequential sweep is the
    fastest access pattern on NVMe and the only sane one over USB. `on_window(w)` is called after
    window w is written (the caller's ledger hook), and `start_window` resumes after a crash."""
    flat = ids.reshape(-1)
    n_rows = table.shape[0]
    if flat.size and (flat.min() < 0 or flat.max() >= n_rows):
        raise IndexError(f"row id out of range [0, {n_rows}): {flat.min()}..{flat.max()}")
    n_win = -(-n_rows // window_rows)
    for w in range(start_window, n_win):
        a, b = w * window_rows, min(n_rows, (w + 1) * window_rows)
        sel = np.flatnonzero((flat >= a) & (flat < b))
        if sel.size:
            block = read_rows(table, a, b)
            out[sel] = block[flat[sel] - a]
        if on_window is not None:
            on_window(w)
