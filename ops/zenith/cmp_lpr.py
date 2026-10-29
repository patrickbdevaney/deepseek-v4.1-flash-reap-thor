#!/usr/bin/env python3
"""cmp_lpr.py A.bin B.bin: per (shape, M) max |a-b| / rms(a) between two gate_lpr dumps."""
import struct, sys, numpy as np
def read(p):
    out, b = [], open(p, "rb").read(); o = 0
    while o < len(b):
        kind, N, K, M = struct.unpack_from("4i", b, o); o += 16
        name = b[o:o + 16].split(b"\0")[0].decode(); o += 16
        y = np.frombuffer(b, np.float32, M * N, o); o += 4 * M * N
        out.append((name, kind, N, K, M, y))
    return out
for (n, k, N, K, M, a), (*_, b) in zip(read(sys.argv[1]), read(sys.argv[2])):
    rms = float(np.sqrt(np.mean(a.astype(np.float64) ** 2))) or 1.0
    d = float(np.max(np.abs(a - b))) / rms
    print(f"{'BAD ' if d > 1e-3 else '    '}{n:10s} kind {k} N {N:5d} K {K:5d} M {M:3d}  max|d|/rms {d:.2e}")
