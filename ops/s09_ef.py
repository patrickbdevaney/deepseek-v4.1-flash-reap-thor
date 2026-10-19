#!/usr/bin/env python3
"""s09b: GPTQ-style error-feedback rounding for the routed experts of one GGUF level. Optional.

Plain llama-quantize rounds each 256-wide superblock on its own, with the imatrix as diagonal
weights. This pass quantizes the columns of every expert matrix in superblock order, left to right.
After each block it pushes that block's rounding error onto the columns not yet quantized, using
the input second moment H:

    W[:, rest] -= (W_b - Q_b) @ inv(U_bb) @ U_b,rest          (U = upper Cholesky of H^-1)

This is the block-OBS form of GPTQ; it reduces to the per-column update when the block width is 1.
Q_b comes from ggml_quantize_chunk, the same codec, imatrix weights and code path that
llama-quantize uses. So the tensor this pass writes has the identical type and byte size, and
it is patched in place into the level's whole (pre-split) GGUF.

Inputs:
  SRC.gguf     the lossless source; experts are MXFP4 and get dequantized here (exact)
  LEVEL.gguf   llama-quantize's output; must already hold the planned type per expert tensor
  --hessians D one npz per layer, D/L<L>.npz, holding:
                 H_in [C_in, C_in]   pooled E[x x^T] of the routed-expert inputs (gate/up), or H_in_e<e>
                 H_mid [C_mid, C_mid] pooled for down_proj, or H_mid_e<e>
               Per-expert keys take precedence. s03 banks only the diagonal (the imatrix), so these
               come from a separate calibration pass. A layer without its file keeps its plain
               rounding, and that is logged.
  --imatrix    the level's legacy .dat, so Q_b is rounded with the same weights llama-quantize used

  apply SRC LEVEL --hessians D --imatrix I [--layers ...]
  selftest   synthetic check: EF output error must beat plain rounding by a clear margin for each ladder type
"""
from __future__ import annotations

import argparse
import ctypes
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", os.environ.get("S09_THREADS", "4"))   # before numpy loads BLAS
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ops.lib import env  # noqa: E402

LCPP = env.REPO / "llamacpp" / "llama.cpp"
sys.path.insert(1, str(LCPP / "gguf-py"))
LIB = Path(os.environ.get("GGML_LIB", LCPP / "build-thor" / "bin" / "libggml-base.so"))
EXPERT_RE = re.compile(r"^blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$")
QK = 256


class _Traits(ctypes.Structure):
    _fields_ = [("type_name", ctypes.c_char_p), ("blck_size", ctypes.c_int64),
                ("blck_size_interleave", ctypes.c_int64), ("type_size", ctypes.c_size_t),
                ("is_quantized", ctypes.c_bool), ("to_float", ctypes.c_void_p), ("from_float_ref", ctypes.c_void_p)]


class Codec:
    """ggml's own quantize/dequantize for one type."""
    _lib = None

    def __init__(self, qtype: int):
        if Codec._lib is None:
            lib = ctypes.CDLL(str(LIB))
            lib.ggml_quantize_chunk.restype = ctypes.c_size_t
            lib.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64,
                                                ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
            lib.ggml_get_type_traits.restype = ctypes.POINTER(_Traits)
            lib.ggml_get_type_traits.argtypes = [ctypes.c_int]
            lib.ggml_quantize_init.argtypes = [ctypes.c_int]
            lib.ggml_row_size.restype = ctypes.c_size_t
            lib.ggml_row_size.argtypes = [ctypes.c_int, ctypes.c_int64]
            Codec._lib = lib
        self.t = qtype
        self.lib = Codec._lib
        self.lib.ggml_quantize_init(qtype)
        tr = self.lib.ggml_get_type_traits(qtype).contents
        self.to_float = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64)(tr.to_float)

    def row_bytes(self, n: int) -> int:
        return int(self.lib.ggml_row_size(self.t, n))

    def quant(self, x: np.ndarray, imat: np.ndarray | None) -> np.ndarray:
        """x [rows, n] float32 -> bytes [rows, row_bytes(n)]"""
        x = np.ascontiguousarray(x, dtype=np.float32)
        rows, n = x.shape
        out = np.empty((rows, self.row_bytes(n)), dtype=np.uint8)
        im = None if imat is None else np.ascontiguousarray(imat, dtype=np.float32)
        self.lib.ggml_quantize_chunk(self.t, x.ctypes.data, out.ctypes.data, 0, rows, n,
                                     None if im is None else im.ctypes.data)
        return out

    def dequant(self, q: np.ndarray, n: int) -> np.ndarray:
        q = np.ascontiguousarray(q)
        rows = q.shape[0]
        out = np.empty((rows, n), dtype=np.float32)
        self.to_float(q.ctypes.data, out.ctypes.data, rows * n)
        return out


def gptq(W: np.ndarray, H: np.ndarray, codec: Codec, imat: np.ndarray | None, damp: float = 0.01) -> np.ndarray:
    """W [rows, cols] float -> quantized bytes [rows, row_bytes(cols)] with block error feedback."""
    rows, cols = W.shape
    assert cols % QK == 0 and H.shape == (cols, cols)
    W = W.astype(np.float64).copy()
    H = H.astype(np.float64).copy()
    dead = np.diag(H) <= 0
    H[dead, dead] = 1.0
    W[:, dead] = 0.0
    H[np.diag_indices(cols)] += damp * float(np.mean(np.diag(H)))
    Hinv = np.linalg.inv(np.linalg.cholesky(H))
    Hinv = Hinv.T @ Hinv                          # H^-1
    U = np.linalg.cholesky(Hinv).T                # upper: H^-1 = U^T U
    nb = cols // QK
    rb = codec.row_bytes(QK)
    out = np.empty((rows, nb * rb), dtype=np.uint8)
    for b in range(nb):
        s, e = b * QK, (b + 1) * QK
        q = codec.quant(W[:, s:e].astype(np.float32), None if imat is None else imat[s:e])
        out[:, b * rb:(b + 1) * rb] = q
        if e < cols:
            err = W[:, s:e] - codec.dequant(q, QK).astype(np.float64)
            W[:, e:] -= err @ np.linalg.solve(U[s:e, s:e], U[s:e, e:])
    return out


def load_h(hdir: Path, L: int, kind: str, e: int, cache: dict) -> np.ndarray | None:
    f = hdir / f"L{L}.npz"
    if not f.exists():
        return None
    if f not in cache:
        cache.clear()
        cache[f] = np.load(f)
    z = cache[f]
    for k in (f"{kind}_e{e}", kind):
        if k in z.files:
            return np.asarray(z[k], dtype=np.float64)
    return None


def apply(src: Path, level: Path, hdir: Path, imatrix: Path, layers: list[int] | None) -> None:
    from gguf import GGUFReader, GGMLQuantizationType
    from ops.lib.imatrix import read_dat
    imat, _, _ = read_dat(imatrix)
    S, D = GGUFReader(str(src), "r"), GGUFReader(str(level), "r")
    dst_t = {t.name: t for t in D.tensors}
    mx = Codec(int(GGMLQuantizationType.MXFP4))
    fd = os.open(level, os.O_WRONLY)
    patched = skipped = 0
    hcache: dict = {}
    for t in S.tensors:
        m = EXPERT_RE.match(t.name)
        if not m or (layers and int(m.group(1)) not in layers):
            continue
        L, proj = int(m.group(1)), m.group(2)
        d = dst_t[t.name]
        if t.tensor_type != GGMLQuantizationType.MXFP4:
            raise SystemExit(f"FATAL: source {t.name} is {t.tensor_type.name}, not MXFP4")
        if d.tensor_type == GGMLQuantizationType.MXFP4:
            continue                                  # lossless copy, nothing to round
        cols, rows, n_exp = (int(x) for x in t.shape[:3])
        codec = Codec(int(d.tensor_type))
        ebytes = rows * codec.row_bytes(cols)
        assert ebytes * n_exp == d.n_bytes, (t.name, ebytes, n_exp, d.n_bytes)
        ncall, vals = imat.get(t.name, (1, None))
        src_raw = np.asarray(t.data).reshape(n_exp, -1)
        for e in range(n_exp):
            H = load_h(hdir, L, "H_mid" if proj == "down" else "H_in", e, hcache)
            if H is None:
                skipped += 1
                continue
            W = mx.dequant(src_raw[e].reshape(rows, -1), cols)
            im = None if vals is None else vals[e * cols:(e + 1) * cols] / max(ncall, 1)
            q = gptq(W, H, codec, im)
            os.pwrite(fd, q.tobytes(), d.data_offset + e * ebytes)
            patched += 1
        print(f"[{env.now()}] {t.name}: {d.tensor_type.name} error feedback done", flush=True)
    os.fsync(fd)
    os.close(fd)
    print(f"[{env.now()}] EF patched {patched} expert matrices, {skipped} kept plain rounding (no Hessian)", flush=True)
    if patched == 0:
        raise SystemExit("FATAL: --ef given but no Hessians matched; the level is plain rounding")


def selftest() -> int:
    from gguf import GGMLQuantizationType as T
    rng = np.random.default_rng(0)
    rows, cols, n = 256, 1024, 4096
    A = rng.standard_normal((cols, cols)) / np.sqrt(cols)
    X = rng.standard_normal((n, cols)) @ (np.eye(cols) + 0.6 * A)     # correlated inputs
    X *= rng.lognormal(0, 0.5, cols)                                  # and uneven channel scales
    H = X.T @ X / n
    W = rng.standard_normal((rows, cols)).astype(np.float32) * 0.02
    ok = True
    for name in ("IQ3_S", "IQ3_XXS", "IQ2_S", "IQ2_XS", "IQ2_XXS", "Q3_K"):
        c = Codec(int(getattr(T, name)))
        im = np.diag(H).astype(np.float32)
        rtn = c.dequant(c.quant(W, im), cols)
        ef = c.dequant(gptq(W, H, c, im), cols)
        e_rtn = np.linalg.norm(X @ (W - rtn).T)
        e_ef = np.linalg.norm(X @ (W - ef).T)
        gain = 1 - e_ef / e_rtn
        print(f"{name:8s} output error  plain {e_rtn:9.4f}  EF {e_ef:9.4f}  ({gain:+.1%})")
        ok &= gain > 0.05
    print("selftest", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["apply", "selftest"])
    ap.add_argument("src", nargs="?", type=Path)
    ap.add_argument("level", nargs="?", type=Path)
    ap.add_argument("--hessians", type=Path)
    ap.add_argument("--imatrix", type=Path)
    ap.add_argument("--layers", type=int, nargs="*")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    if a.cmd == "selftest":
        return selftest()
    if not (a.src and a.level and a.hessians and a.imatrix):
        ap.error("apply needs SRC LEVEL --hessians --imatrix")
    apply(a.src, a.level, a.hessians, a.imatrix, a.layers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
