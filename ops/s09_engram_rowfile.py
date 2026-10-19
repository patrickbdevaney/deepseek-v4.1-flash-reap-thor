#!/usr/bin/env python3
"""s09a: build the Engram O_DIRECT row files (llamacpp/ENGRAM_ROWFILE.md) from base shards 47/48.

The output has one `engram_L<layer>.v41rows` per Engram layer. Each row is 264 B: 256 FP8 values
copied byte for byte, then 8 E8M0 scales. The llama.cpp fork reads these files through
LLAMA_ENGRAM_ROWFILE_DIR, and so does the CUDA server. Engram is never pruned or quantized, so these
files serve every GGUF level and every keep-set.

CPU and disk only. Peak RSS is about two chunks (2 x 264 MB). Both input and output use
posix_fadvise(DONTNEED), so the 200 GB stream does not fill the unified-memory page cache next to a
GPU job.

  build   [--dst DIR] [--layers 1 14]    resumable from the output size; header (with data sha) written last
  verify  [--dst DIR] [--sample N] [--full]  header, size, N random rows vs the shards, and optionally the data sha256
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ops.lib import env  # noqa: E402
from ops.lib.engram_rows import safetensors_header, tensor_memmap  # noqa: E402

MAGIC = b"V41ENGR1"
VERSION = 1
DATA_OFF = 4096
CHUNK_ROWS = 1 << 20
DEFAULT_DST = env.WORK / "engram_rows"
STAGE = "s09_engram_rowfile"


def log(msg: str) -> None:
    print(f"[{env.now()}] {msg}", flush=True)


def locate(layer: int) -> tuple[Path, str, str]:
    """(shard path, weight name, scale name) for one Engram layer, from the checkpoint index."""
    idx = json.loads((env.SRC / "model.safetensors.index.json").read_text())["weight_map"]
    w, s = f"layers.{layer}.engram.embed.weight", f"layers.{layer}.engram.embed.scale"
    if w not in idx or s not in idx:
        raise SystemExit(f"FATAL: {w} / {s} not in the checkpoint index")
    if idx[w] != idx[s]:
        raise SystemExit(f"FATAL: {w} and {s} live in different shards ({idx[w]}, {idx[s]})")
    return env.SRC / idx[w], w, s


def shard_sha(shard: Path) -> bytes:
    """The pinned sha256 of a shard, from the s00 tree listing (never recomputed from a USB copy)."""
    tree = json.loads((env.SRC.parent / "v41_1017_tree.json").read_text())
    for x in tree:
        if x["path"] == shard.name:
            return bytes.fromhex(x["lfs"]["oid"])
    raise SystemExit(f"FATAL: {shard.name} missing from v41_1017_tree.json")


def header(layer: int, n_rows: int, src_sha: bytes, tensor: str, data_sha: bytes) -> bytes:
    h = bytearray(DATA_OFF)
    h[0:8] = MAGIC
    struct.pack_into("<IIQIIIIIQ", h, 8, VERSION, layer, n_rows, 256, 8, 264, 1, 1, DATA_OFF)
    h[52:84] = src_sha
    h[84:84 + len(tensor)] = tensor.encode()
    h[148:148 + len(env.REVISION)] = env.REVISION.encode()
    h[188:220] = data_sha
    return bytes(h)


def parse_header(buf: bytes) -> dict:
    if buf[:8] != MAGIC:
        raise ValueError("bad magic")
    version, layer, n_rows, key_len, n_scales, row_bytes, vdt, sdt, data_off = struct.unpack_from("<IIQIIIIIQ", buf, 8)
    return dict(version=version, layer=layer, n_rows=n_rows, key_len=key_len, n_scales=n_scales,
                row_bytes=row_bytes, value_dtype=vdt, scale_dtype=sdt, data_off=data_off,
                src_sha=buf[52:84].hex(), tensor=buf[84:148].rstrip(b"\0").decode(),
                revision=buf[148:188].rstrip(b"\0").decode(), data_sha=buf[188:220].hex())


def check_dst(dst: Path, allow_usb: bool) -> None:
    if str(dst.resolve()).startswith("/media/") and not allow_usb:
        raise SystemExit("FATAL: row files are served with O_DIRECT from NVMe; refusing a /media path "
                         "(pass --allow-usb for a convenience copy)")


def build_one(layer: int, dst: Path) -> None:
    shard, wname, sname = locate(layer)
    hdr, base = safetensors_header(shard)
    tw, ts = hdr[wname], hdr[sname]
    n_rows, cols = tw["shape"]
    if tw["dtype"] != "F8_E4M3" or cols != 256 or ts["shape"] != [n_rows, 8] or ts["dtype"] != "F8_E8M0":
        raise SystemExit(f"FATAL: unexpected engram layout {tw} / {ts}")
    out = dst / f"engram_L{layer}.v41rows"
    part = out.with_suffix(".v41rows.partial")
    if out.exists():
        log(f"L{layer}: {out} exists, skipping (run verify)")
        return
    src_sha = shard_sha(shard)

    # resume at a whole chunk; the data hash is re-fed from what is already on disk
    done = 0
    if part.exists():
        done = max(0, (part.stat().st_size - DATA_OFF) // 264 // CHUNK_ROWS * CHUNK_ROWS)
    sha = hashlib.sha256()
    fd_out = os.open(part, os.O_WRONLY | os.O_CREAT, 0o644)
    os.ftruncate(fd_out, DATA_OFF + done * 264)
    if done:
        log(f"L{layer}: resuming at row {done:,}, re-hashing {done * 264 / 1e9:.1f} GB")
        with open(part, "rb") as fh:
            fh.seek(DATA_OFF)
            left = done * 264
            while left:
                b = fh.read(min(left, 1 << 26))
                sha.update(b)
                left -= len(b)
                os.posix_fadvise(fh.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    else:
        os.pwrite(fd_out, bytes(DATA_OFF), 0)   # placeholder header, rewritten last

    fd_in = os.open(shard, os.O_RDONLY)
    w0, s0 = base + tw["data_offsets"][0], base + ts["data_offsets"][0]
    row = np.empty((CHUNK_ROWS, 264), dtype=np.uint8)
    for r0 in range(done, n_rows, CHUNK_ROWS):
        n = min(CHUNK_ROWS, n_rows - r0)
        v = os.pread(fd_in, n * 256, w0 + r0 * 256)
        s = os.pread(fd_in, n * 8, s0 + r0 * 8)
        if len(v) != n * 256 or len(s) != n * 8:
            raise SystemExit(f"FATAL: short read from {shard} at row {r0}")
        row[:n, :256] = np.frombuffer(v, np.uint8).reshape(n, 256)
        row[:n, 256:] = np.frombuffer(s, np.uint8).reshape(n, 8)
        blob = row[:n].tobytes()
        sha.update(blob)
        off = DATA_OFF + r0 * 264
        if os.pwrite(fd_out, blob, off) != len(blob):
            raise SystemExit(f"FATAL: short write to {part}")
        os.posix_fadvise(fd_in, 0, 0, os.POSIX_FADV_DONTNEED)
        os.fdatasync(fd_out)
        os.posix_fadvise(fd_out, 0, 0, os.POSIX_FADV_DONTNEED)
        if (r0 // CHUNK_ROWS) % 32 == 0:
            log(f"L{layer}: {r0 + n:,} / {n_rows:,} rows")
    os.close(fd_in)
    os.pwrite(fd_out, header(layer, n_rows, src_sha, wname, sha.digest()), 0)
    os.fsync(fd_out)
    os.close(fd_out)
    os.rename(part, out)
    log(f"L{layer}: wrote {out} ({n_rows:,} rows, data sha256 {sha.hexdigest()})")


def verify_one(layer: int, dst: Path, sample: int, full: bool) -> bool:
    out = dst / f"engram_L{layer}.v41rows"
    if not out.exists():
        log(f"FAIL L{layer}: {out} missing")
        return False
    with open(out, "rb") as fh:
        h = parse_header(fh.read(DATA_OFF))
    shard, wname, sname = locate(layer)
    ok = (h["version"] == VERSION and h["layer"] == layer and h["row_bytes"] == 264 and h["key_len"] == 256
          and h["n_scales"] == 8 and h["data_off"] == DATA_OFF and h["tensor"] == wname
          and h["revision"] == env.REVISION and h["src_sha"] == shard_sha(shard).hex())
    size_ok = out.stat().st_size == DATA_OFF + h["n_rows"] * 264
    if not (ok and size_ok):
        log(f"FAIL L{layer}: header/size mismatch {h} size={out.stat().st_size}")
        return False
    # random rows against the shard (the shard may be the Seagate copy: a few thousand reads is fine)
    W, S = tensor_memmap(shard, wname), tensor_memmap(shard, sname)
    if W.shape[0] != h["n_rows"]:
        log(f"FAIL L{layer}: n_rows {h['n_rows']} != source {W.shape[0]}")
        return False
    rng = np.random.default_rng(layer)
    ids = np.unique(np.concatenate([[0, h["n_rows"] - 1], rng.integers(0, h["n_rows"], sample)]))
    fd = os.open(out, os.O_RDONLY)
    for r in ids:
        got = os.pread(fd, 264, DATA_OFF + int(r) * 264)
        want = bytes(W[r]) + bytes(S[r])
        if got != want:
            log(f"FAIL L{layer}: row {r} differs from the source")
            return False
    os.close(fd)
    if full:
        sha = hashlib.sha256()
        with open(out, "rb") as fh:
            fh.seek(DATA_OFF)
            while b := fh.read(1 << 26):
                sha.update(b)
                os.posix_fadvise(fh.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        if sha.hexdigest() != h["data_sha"]:
            log(f"FAIL L{layer}: data sha256 {sha.hexdigest()} != header {h['data_sha']}")
            return False
    log(f"PASS L{layer}: {h['n_rows']:,} rows, {len(ids)} sampled rows bit-exact"
        + (", data sha256 ok" if full else ""))
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["build", "verify"])
    ap.add_argument("--dst", type=Path, default=DEFAULT_DST)
    ap.add_argument("--layers", type=int, nargs="+", default=None)
    ap.add_argument("--sample", type=int, default=4096)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--allow-usb", action="store_true")
    a = ap.parse_args()
    layers = a.layers or json.loads((env.SRC / "config.json").read_text())["text_config"]["engram_layer_ids"]
    check_dst(a.dst, a.allow_usb)
    a.dst.mkdir(parents=True, exist_ok=True)
    if a.cmd == "build":
        for L in layers:
            build_one(L, a.dst)
        return 0
    return 0 if all(verify_one(L, a.dst, a.sample, a.full) for L in layers) else 1


if __name__ == "__main__":
    sys.exit(main())
