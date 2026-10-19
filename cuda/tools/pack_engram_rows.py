#!/usr/bin/env python3
"""Optional LOSSY 6-bit Engram rows: convert the FP8 .v41rows files (llamacpp/ENGRAM_ROWFILE.md,
built by ops/s09_engram_rowfile.py -- the default, bit-exact, and shared with llama.cpp) into the
same container with value_dtype = 2:

  row = 192 B of MXFP6 E3M2 codes (4 codes per 3 bytes, little-endian 24-bit groups, sign at bit 5)
        + 8 E8M0 scales re-derived per 32 (pow2-ceil so the group max lands at or below 28)
  -> 200 B/row, 2 x 76.8 GB instead of 2 x 101.4 GB (-24%); every O_DIRECT read is still one 1 KiB.

Only the CUDA server reads value_dtype 2 (engram_store.cpp); llama.cpp's reader refuses it because
row_bytes != key_len + n_scales, which is the intended failure. The converter prints rel-L2 and cosine
against the FP8 rows on a running sample; serving it needs the s10 Engram-ablation gate first.

Refuses to start without the disk space (+2% margin). Sequential reads and writes, posix_fadvise
DONTNEED on both, peak RSS ~ two chunks. Header data sha256 written last (resumable = rerun).

Usage: python -I tools/pack_engram_rows.py --src <dir of engram_L*.v41rows> --out <nvme dir> [--verify 20000]
"""
import argparse
import hashlib
import os
import shutil
import struct
import sys
import time

import numpy as np

HDR = 4096


def e4m3_table():
    t = np.zeros(256, np.float32)
    for b in range(256):
        s = -1.0 if b & 0x80 else 1.0
        e = (b >> 3) & 0xF
        m = b & 7
        if e == 0xF and m == 7:
            t[b] = np.nan
        elif e == 0:
            t[b] = s * (m / 8.0) * 2.0 ** -6
        else:
            t[b] = s * (1 + m / 8.0) * 2.0 ** (e - 7)
    return t


E4M3 = e4m3_table()
# E3M2 positive grid (bias 3, no inf/nan): codes 0..31
E3M2 = np.array([(m / 4.0) * 2.0 ** -2 if e == 0 else (1 + m / 4.0) * 2.0 ** (e - 3)
                 for e in range(8) for m in range(4)], np.float32)
E3M2_MAX = float(E3M2[-1])   # 28


def to_e3m2(x):
    """round-to-nearest-even onto E3M2 codes (6 bit, sign at bit 5)"""
    a = np.abs(x)
    i = np.searchsorted(E3M2, a)                       # first grid value >= a
    i = np.clip(i, 1, 31)
    lo, hi = E3M2[i - 1], E3M2[i]
    pick_hi = (a - lo > hi - a) | ((a - lo == hi - a) & ((i & 1) == 0))
    code = np.where(pick_hi, i, i - 1)
    code = np.where(a >= E3M2_MAX, 31, code)
    return (code | ((x < 0) << 5)).astype(np.uint32)


def mxfp6_rows(w8, s8):
    """w8 [n,256] uint8 e4m3, s8 [n,8] uint8 e8m0 -> [n,200] uint8"""
    n = w8.shape[0]
    v = E4M3[w8].reshape(n, 8, 32) * np.ldexp(1.0, s8.astype(np.int32) - 127)[..., None].astype(np.float32)
    amax = np.abs(v).max(-1)
    # pow2-ceil scale so the group max lands at or below 28
    with np.errstate(divide="ignore"):
        e = np.ceil(np.log2(np.maximum(amax, 1e-38) / E3M2_MAX)).astype(np.int32)
    e = np.clip(e + 127, 0, 254)
    sc = np.ldexp(1.0, e - 127).astype(np.float32)
    codes = to_e3m2(v / sc[..., None]).reshape(n, 64, 4)
    word = codes[..., 0] | (codes[..., 1] << 6) | (codes[..., 2] << 12) | (codes[..., 3] << 18)
    out = np.zeros((n, 200), np.uint8)
    out[:, 0:192:3] = word & 0xFF
    out[:, 1:192:3] = (word >> 8) & 0xFF
    out[:, 2:192:3] = (word >> 16) & 0xFF
    out[:, 192:] = e.astype(np.uint8)
    return out, v, sc


def unpack_mxfp6(r):
    n = r.shape[0]
    b = r[:, :192].astype(np.uint32)
    word = b[:, 0::3] | (b[:, 1::3] << 8) | (b[:, 2::3] << 16)
    codes = np.stack([(word >> (6 * k)) & 63 for k in range(4)], -1).reshape(n, 256)
    mag = E3M2[codes & 31]
    val = np.where(codes & 32, -mag, mag).reshape(n, 8, 32)
    return val * np.ldexp(1.0, r[:, 192:].astype(np.int32) - 127)[..., None]


def read_header(path):
    with open(path, "rb") as f:
        h = f.read(HDR)
    magic, ver, layer, n_rows, key_len, n_sc, rb, vdt, sdt, doff = struct.unpack_from("<8sIIQIIIIIQ", h, 0)
    assert magic == b"V41ENGR1" and ver == 1 and vdt == 1 and rb == 264 and doff == HDR, (path, magic, ver, vdt, rb, doff)
    return h, layer, n_rows


def convert(L, src, out, chunk, verify):
    sp = os.path.join(src, f"engram_L{L}.v41rows")
    h, layer, n = read_header(sp)
    assert layer == L
    dst = os.path.join(out, f"engram_L{L}.v41rows")
    tmp = dst + ".partial"
    sha = hashlib.sha256()
    err_num = err_den = dot = n1 = n2 = 0.0
    t0 = time.time()
    hdr = bytearray(h)
    struct.pack_into("<II", hdr, 32, 200, 2)                 # row_bytes, value_dtype = MXFP6
    hdr[188:220] = b"\0" * 32                                # data sha: written last
    with open(sp, "rb", buffering=0) as fi, open(tmp, "wb") as fo:
        fo.write(bytes(hdr))
        for r0 in range(0, n, chunk):
            m = min(chunk, n - r0)
            fi.seek(HDR + r0 * 264)
            raw = np.frombuffer(fi.read(m * 264), np.uint8).reshape(m, 264)
            os.posix_fadvise(fi.fileno(), HDR + r0 * 264, m * 264, os.POSIX_FADV_DONTNEED)
            rows, v, _ = mxfp6_rows(raw[:, :256], raw[:, 256:])
            if (r0 // chunk) % 50 == 0:
                d = unpack_mxfp6(rows[:4096]).reshape(-1); ref = v[:4096].reshape(-1)
                err_num += float(((d - ref) ** 2).sum()); err_den += float((ref ** 2).sum())
                dot += float((d * ref).sum()); n1 += float((d * d).sum()); n2 += float((ref * ref).sum())
            buf = rows.tobytes(); fo.write(buf); sha.update(buf)
            fo.flush(); os.posix_fadvise(fo.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
            if (r0 // chunk) % 20 == 0:
                print(f"L{L}: {r0 + m:,}/{n:,} rows  {(r0 + m) * 264 / max(time.time() - t0, 1e-9) / 1e6:.0f} MB/s", flush=True)
        fo.seek(188); fo.write(sha.digest())
    os.replace(tmp, dst)
    meta = {"rows": n, "seconds": round(time.time() - t0, 1)}
    if err_den:
        meta["mxfp6_rel_l2"] = (err_num / err_den) ** 0.5
        meta["mxfp6_cos"] = dot / (n1 * n2) ** 0.5
    if verify:
        rng = np.random.default_rng(L)
        bad = 0
        with open(dst, "rb") as fo, open(sp, "rb") as fi:
            for r in np.sort(rng.integers(0, n, verify)):
                fo.seek(HDR + r * 200); got = np.frombuffer(fo.read(200), np.uint8)
                fi.seek(HDR + r * 264); raw = np.frombuffer(fi.read(264), np.uint8)
                ref, _, _ = mxfp6_rows(raw[None, :256], raw[None, 256:])
                bad += not np.array_equal(got, ref[0])
        meta["verify_rows"], meta["verify_bad"] = int(verify), int(bad)
        if bad:
            raise SystemExit(f"L{L}: {bad}/{verify} rows mismatch after packing -- refusing")
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="dir with the FP8 engram_L1/L14.v41rows (s09)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk", type=int, default=1 << 20)
    ap.add_argument("--verify", type=int, default=20000)
    ap.add_argument("--layers", default="1,14")
    a = ap.parse_args()
    if os.path.realpath(a.src) == os.path.realpath(a.out):
        raise SystemExit("--out must differ from --src (same file names)")
    os.makedirs(a.out, exist_ok=True)
    layers = [int(x) for x in a.layers.split(",")]
    need = sum(HDR + read_header(os.path.join(a.src, f"engram_L{L}.v41rows"))[2] * 200 for L in layers)
    free = shutil.disk_usage(a.out).free
    if free < need * 1.02:
        raise SystemExit(f"need {need / 1e9:.1f} GB (+2%) on {a.out}, have {free / 1e9:.1f} GB -- refusing")
    for L in layers:
        print(L, convert(L, a.src, a.out, a.chunk, a.verify), flush=True)


if __name__ == "__main__":
    main()
