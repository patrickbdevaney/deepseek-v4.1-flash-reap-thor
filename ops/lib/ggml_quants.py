"""Bit-exact ggml block formats for the Thor build, plus the MXFP4 / NVFP4 conversions (numpy only).

Why ggml blocks: the s07 Thor build is consumed by two runtimes, our CUDA server (s08) and the
lean llama.cpp fork (s09). Storing every requantized expert as the exact byte layout of a ggml
block type means the GGUF writer copies bytes and never re-quantizes (lte-ai: requantizing from
a low-bit intermediate nearly doubled KL), and the CUDA server can port llama.cpp's MIT dequant
code for the same bytes. Layouts below follow ggml-common.h / ggml-quants.c and are verified
against llama.cpp's gguf-py `dequantize_blocks` in ops/tests/test_pure.py.

  Q2_K  84 B / 256  (2.625 bpw)  scales[16] (lo4 scale, hi4 min) | qs[64] | d f16 | dmin f16
  Q3_K 110 B / 256  (3.4375)     hmask[32] | qs[64] | scales[12] (6-bit signed, -32) | d f16
  Q4_K 144 B / 256  (4.5)        d f16 | dmin f16 | scales[12] (6-bit scale+min) | qs[128]
  MXFP4 17 B / 32   (4.25)       e (E8M0) | qs[16], element i low nibble, element i+16 high nibble

Quantization here is split in two so GPTQ can use it: `fit_*` chooses the block parameters from
(current) weights and per-column importance, `codes_*` rounds weights against FIXED parameters,
and `pack_*` writes the bytes. RTN = fit + codes + pack.

The source checkpoint's MXFP4 (float4_e2m1fn_x2 [out, in/2] + E8M0 [out, in/32]) packs element 2k
in the LOW nibble and 2k+1 in the high nibble of byte k (assumption shared with
torch_kernels.unpack_fp4; s01 checks it against the TileLang kernel). ggml's MXFP4 block uses a
different nibble order, so `src_to_ggml_mxfp4` reorders nibbles; codes and scales are unchanged.
"""
from __future__ import annotations

import numpy as np

QK_K = 256
# name -> (ggml type id, block elements, block bytes)
TYPES = {
    "F32": (0, 1, 4), "F16": (1, 1, 2), "BF16": (30, 1, 2), "Q8_0": (8, 32, 34),
    "Q2_K": (10, 256, 84), "Q3_K": (11, 256, 110), "Q4_K": (12, 256, 144), "Q5_K": (13, 256, 176),
    "Q6_K": (14, 256, 210), "MXFP4": (39, 32, 17),
}
E2M1 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], np.float32)


def bpw(t: str) -> float:
    _, n, b = TYPES[t]
    return 8.0 * b / n


def row_bytes(t: str, n_cols: int) -> int:
    _, n, b = TYPES[t]
    assert n_cols % n == 0, (t, n_cols)
    return n_cols // n * b


def _f16(x):
    return np.asarray(x, np.float32).astype(np.float16)


# ----------------------------------------------------------------------------- fitting helpers
def _wls_affine(w, a, q):
    """min_{s,b} sum a (s q + b - w)^2 with b <= 0, s >= 0, over the last axis."""
    Sa, Sq, Sqq = a.sum(-1), (a * q).sum(-1), (a * q * q).sum(-1)
    Sw, Sqw = (a * w).sum(-1), (a * q * w).sum(-1)
    det = Sa * Sqq - Sq * Sq
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(det > 0, (Sa * Sqw - Sq * Sw) / det, 0.0)
        b = np.where(det > 0, (Sqq * Sw - Sq * Sqw) / det, 0.0)
        pos = b > 0
        s = np.where(pos, np.where(Sqq > 0, Sqw / np.maximum(Sqq, 1e-30), 0.0), s)
        b = np.where(pos, 0.0, b)
    return np.maximum(s, 0.0), b


def _fit_affine(w, a, nmax, iters=6):
    """Sub-block affine fit value = s*q + b (b <= 0), q in [0, nmax], weighted. w/a [..., n]."""
    lo = np.minimum(w.min(-1), 0.0)
    hi = w.max(-1)
    s = np.maximum(hi - lo, 1e-30) / nmax
    b = lo
    for _ in range(iters):
        q = np.clip(np.rint((w - b[..., None]) / s[..., None]), 0, nmax)
        s_new, b_new = _wls_affine(w, a, q)
        ok = s_new > 0
        s, b = np.where(ok, s_new, s), np.where(ok, b_new, b)
    return s, b


def _fit_sym(w, a, iters=6):
    """Sub-block symmetric fit value = s*q, q in [-4, 3]; s may be negative (the extreme maps to -4
    on whichever side has the larger magnitude, as quantize_row_q3_K does)."""
    i = np.abs(w).argmax(-1)
    ext = np.take_along_axis(w, i[..., None], -1)[..., 0]
    best_s, best_e = None, None
    for s0 in (ext / -4.0, ext / 3.0):
        s = np.where(s0 == 0, 1e-30, s0)
        for _ in range(iters):
            q = np.clip(np.rint(w / s[..., None]), -4, 3)
            num, den = (a * q * w).sum(-1), (a * q * q).sum(-1)
            s = np.where(den > 0, num / np.maximum(den, 1e-30), s)
            s = np.where(s == 0, 1e-30, s)
        q = np.clip(np.rint(w / s[..., None]), -4, 3)
        e = (a * (s[..., None] * q - w) ** 2).sum(-1)
        if best_s is None:
            best_s, best_e = s, e
        else:
            take = e < best_e
            best_s, best_e = np.where(take, s, best_s), np.where(take, e, best_e)
    return best_s


def _superquant(x, levels):
    """Quantize non-negative per-sub-block values x [..., n] to integers 0..levels with an f16
    super-scale. -> (d f16 [...], codes uint8 [..., n], dequantized x)."""
    mx = x.max(-1)
    d = _f16(np.where(mx > 0, mx / levels, 0.0)).astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        c = np.where(d[..., None] > 0, np.clip(np.rint(x / d[..., None]), 0, levels), 0)
    return d, c.astype(np.uint8), d[..., None] * c


# ----------------------------------------------------------------------------- K-quant params
def fit_q2k(W, A):
    """W, A [..., 256] -> params dict (S, M effective per sub-block of 16; d, dmin, sc, mc)."""
    w = W.reshape(*W.shape[:-1], 16, 16).astype(np.float64)
    a = np.broadcast_to(A, W.shape).reshape(w.shape).astype(np.float64) + 1e-12
    s, b = _fit_affine(w, a, 3)
    d, sc, S = _superquant(s, 15)
    dmin, mc, M = _superquant(-b, 15)
    return {"type": "Q2_K", "S": S, "M": M, "d": d, "dmin": dmin, "sc": sc, "mc": mc}


def fit_q4k(W, A):
    w = W.reshape(*W.shape[:-1], 8, 32).astype(np.float64)
    a = np.broadcast_to(A, W.shape).reshape(w.shape).astype(np.float64) + 1e-12
    s, b = _fit_affine(w, a, 15)
    d, sc, S = _superquant(s, 63)
    dmin, mc, M = _superquant(-b, 63)
    return {"type": "Q4_K", "S": S, "M": M, "d": d, "dmin": dmin, "sc": sc, "mc": mc}


def fit_q3k(W, A):
    w = W.reshape(*W.shape[:-1], 16, 16).astype(np.float64)
    a = np.broadcast_to(A, W.shape).reshape(w.shape).astype(np.float64) + 1e-12
    s = _fit_sym(w, a)
    i = np.abs(s).argmax(-1)
    ext = np.take_along_axis(s, i[..., None], -1)[..., 0]
    d = _f16(-ext / 32.0).astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        sc = np.where(d[..., None] != 0, np.clip(np.rint(s / d[..., None]), -32, 31), 0)
    return {"type": "Q3_K", "S": d[..., None] * sc, "M": np.zeros_like(s), "d": d, "sc": (sc + 32).astype(np.uint8)}


FIT = {"Q2_K": fit_q2k, "Q3_K": fit_q3k, "Q4_K": fit_q4k}
QRANGE = {"Q2_K": (0, 3), "Q3_K": (-4, 3), "Q4_K": (0, 15)}
SUB = {"Q2_K": 16, "Q3_K": 16, "Q4_K": 32}


def expand(p: dict, key: str) -> np.ndarray:
    """Per-sub-block value -> per-element [..., 256]."""
    return np.repeat(p[key], SUB[p["type"]], axis=-1)


def codes(W, p):
    """Round W [..., 256] against fixed params. value = S*q - M (M = 0 for Q3_K)."""
    lo, hi = QRANGE[p["type"]]
    S, M = expand(p, "S"), expand(p, "M")
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(S != 0, np.rint((W + M) / S), 0)
    return np.clip(q, lo, hi).astype(np.int8)


def values(q, p):
    return expand(p, "S") * q.astype(np.float32) - expand(p, "M")


# ----------------------------------------------------------------------------- packers
def pack_q2k(q, p):
    n = q.reshape(-1, 256).shape[0]
    q = q.reshape(n, 2, 4, 32).astype(np.uint8)          # chunk, shift, byte
    qs = (q[:, :, 0] | (q[:, :, 1] << 2) | (q[:, :, 2] << 4) | (q[:, :, 3] << 6)).reshape(n, 64)
    scales = (p["sc"].reshape(n, 16) | (p["mc"].reshape(n, 16) << 4)).astype(np.uint8)
    d = _f16(p["d"]).reshape(n, 1).view(np.uint8)
    dm = _f16(p["dmin"]).reshape(n, 1).view(np.uint8)
    return np.concatenate([scales, qs, d, dm], 1)


def pack_q3k(q, p):
    n = q.reshape(-1, 256).shape[0]
    u = (q.reshape(n, 256).astype(np.int16) + 4).astype(np.uint8)       # 0..7
    lo = (u & 3).reshape(n, 2, 4, 32)
    qs = (lo[:, :, 0] | (lo[:, :, 1] << 2) | (lo[:, :, 2] << 4) | (lo[:, :, 3] << 6)).reshape(n, 64)
    hb = (u >> 2).reshape(n, 8, 32)                                      # bit index = element // 32
    hmask = np.zeros((n, 32), np.uint8)
    for b in range(8):
        hmask |= hb[:, b] << b
    sc = p["sc"].reshape(n, 16).astype(np.uint8)                         # 0..63
    scales = np.zeros((n, 12), np.uint8)
    scales[:, :8] = (sc[:, :8] & 0xF) | ((sc[:, 8:] & 0xF) << 4)
    hi = sc >> 4
    for j in range(16):
        scales[:, 8 + j % 4] |= hi[:, j] << (2 * (j // 4))
    d = _f16(p["d"]).reshape(n, 1).view(np.uint8)
    return np.concatenate([hmask, qs, scales, d], 1)


def pack_q4k(q, p):
    n = q.reshape(-1, 256).shape[0]
    u = q.reshape(n, 4, 2, 32).astype(np.uint8)
    qs = (u[:, :, 0] | (u[:, :, 1] << 4)).reshape(n, 128)
    sc, mc = p["sc"].reshape(n, 8).astype(np.uint8), p["mc"].reshape(n, 8).astype(np.uint8)
    s = np.zeros((n, 12), np.uint8)
    s[:, 0:4] = (sc[:, 0:4] & 63) | ((sc[:, 4:8] >> 4) << 6)
    s[:, 4:8] = (mc[:, 0:4] & 63) | ((mc[:, 4:8] >> 4) << 6)
    s[:, 8:12] = (sc[:, 4:8] & 0xF) | ((mc[:, 4:8] & 0xF) << 4)
    d = _f16(p["d"]).reshape(n, 1).view(np.uint8)
    dm = _f16(p["dmin"]).reshape(n, 1).view(np.uint8)
    return np.concatenate([d, dm, s, qs], 1)


PACK = {"Q2_K": pack_q2k, "Q3_K": pack_q3k, "Q4_K": pack_q4k}


# ----------------------------------------------------------------------------- dequant
def dequant(t: str, blocks: np.ndarray) -> np.ndarray:
    """blocks uint8 [..., k * block_bytes] -> float32 [..., k * block_elements]."""
    _, ne, nb = TYPES[t]
    assert blocks.shape[-1] % nb == 0, (t, blocks.shape)
    lead = blocks.shape[:-1]
    b = np.ascontiguousarray(blocks).view(np.uint8).reshape(-1, nb)
    n = b.shape[0]
    if t == "Q2_K":
        sc, qs, d, dm = b[:, :16], b[:, 16:80], b[:, 80:82].copy().view(np.float16), b[:, 82:84].copy().view(np.float16)
        q = ((qs.reshape(n, 2, 1, 32) >> np.array([0, 2, 4, 6], np.uint8).reshape(1, 1, 4, 1)) & 3).reshape(n, 16, 16)
        y = d.astype(np.float32).reshape(n, 1, 1) * (sc & 0xF).reshape(n, 16, 1) * q \
            - dm.astype(np.float32).reshape(n, 1, 1) * (sc >> 4).reshape(n, 16, 1)
    elif t == "Q3_K":
        hm, qs, s12, d = b[:, :32], b[:, 32:96], b[:, 96:108], b[:, 108:110].copy().view(np.float16)
        lo = np.concatenate([s12[:, :8] & 0xF, s12[:, :8] >> 4], 1)
        hi = ((s12[:, 8:12].reshape(n, 1, 4) >> np.array([0, 2, 4, 6], np.uint8).reshape(1, 4, 1)) & 3).reshape(n, 16)
        sc = (lo | (hi << 4)).astype(np.int16) - 32
        ql = ((qs.reshape(n, 2, 1, 32) >> np.array([0, 2, 4, 6], np.uint8).reshape(1, 1, 4, 1)) & 3).reshape(n, 256)
        qh = ((hm.reshape(n, 1, 32) >> np.arange(8, dtype=np.uint8).reshape(1, 8, 1)) & 1).reshape(n, 256)
        q = ql.astype(np.int16) - np.where(qh == 1, 0, 4)
        y = d.astype(np.float32).reshape(n, 1, 1) * sc.reshape(n, 16, 1) * q.reshape(n, 16, 16)
    elif t == "Q4_K":
        d, dm = b[:, 0:2].copy().view(np.float16), b[:, 2:4].copy().view(np.float16)
        s, qs = b[:, 4:16], b[:, 16:144]
        sc = np.concatenate([s[:, 0:4] & 63, (s[:, 8:12] & 0xF) | ((s[:, 0:4] >> 6) << 4)], 1)
        mn = np.concatenate([s[:, 4:8] & 63, (s[:, 8:12] >> 4) | ((s[:, 4:8] >> 6) << 4)], 1)
        q = ((qs.reshape(n, 4, 1, 32) >> np.array([0, 4], np.uint8).reshape(1, 1, 2, 1)) & 0xF).reshape(n, 8, 32)
        y = d.astype(np.float32).reshape(n, 1, 1) * sc.reshape(n, 8, 1) * q - dm.astype(np.float32).reshape(n, 1, 1) * mn.reshape(n, 8, 1)
    elif t == "Q8_0":
        d, qs = b[:, :2].copy().view(np.float16), b[:, 2:].view(np.int8)
        y = d.astype(np.float32) * qs
    elif t == "MXFP4":
        e, qs = b[:, 0], b[:, 1:17]
        codes_ = np.concatenate([qs & 0xF, qs >> 4], 1)
        y = E2M1[codes_] * e8m0(e).reshape(n, 1)
    else:
        raise ValueError(t)
    return y.astype(np.float32).reshape(*lead, -1)


def quantize_rtn(t: str, W: np.ndarray, A: np.ndarray | None = None) -> np.ndarray:
    """W [R, C] float, A importance [C] or [R, C] (imatrix) -> uint8 [R, C/256*bytes]."""
    R, C = W.shape
    Wb = W.reshape(R, C // QK_K, QK_K).astype(np.float32)
    Ab = np.ones_like(Wb) if A is None else np.broadcast_to(A, W.shape).reshape(Wb.shape)
    p = FIT[t](Wb, Ab)
    q = codes(Wb, p)
    return PACK[t](q.reshape(-1, QK_K), {k: (v.reshape(-1, *v.shape[2:]) if isinstance(v, np.ndarray) else v)
                                         for k, v in p.items()}).reshape(R, -1)


# ----------------------------------------------------------------------------- MXFP4 / NVFP4
def e8m0(e: np.ndarray) -> np.ndarray:
    """E8M0 byte -> float32 2^(e-127); 0xFF (NaN in OCP) is refused, not decoded."""
    e = np.asarray(e, np.uint8)
    if (e == 0xFF).any():
        raise ValueError("E8M0 NaN scale (0xFF) in source")
    return np.ldexp(np.float32(1.0), e.astype(np.int32) - 127).astype(np.float32)


def src_codes(wbytes: np.ndarray) -> np.ndarray:
    """source float4_e2m1fn_x2 bytes [R, C/2] -> nibble codes [R, C] (low nibble = even element)."""
    b = np.asarray(wbytes).view(np.uint8)
    out = np.empty((*b.shape[:-1], b.shape[-1] * 2), np.uint8)
    out[..., 0::2] = b & 0xF
    out[..., 1::2] = b >> 4
    return out


def src_dequant(wbytes: np.ndarray, scale: np.ndarray, block: int = 32) -> np.ndarray:
    c = src_codes(wbytes)
    R, C = c.shape
    s = e8m0(np.asarray(scale).view(np.uint8))
    return (E2M1[c].reshape(R, C // block, block) * s.reshape(R, C // block, 1)).reshape(R, C)


def src_to_ggml_mxfp4(wbytes: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Lossless reorder to ggml block_mxfp4 [R, C/32 * 17]: same codes, same E8M0 exponent."""
    c = src_codes(wbytes)
    R, C = c.shape
    c = c.reshape(R, C // 32, 32)
    qs = c[..., :16] | (c[..., 16:] << 4)
    e = np.asarray(scale).view(np.uint8).reshape(R, C // 32, 1)
    return np.concatenate([e, qs], -1).reshape(R, -1)


def e4m3_pow2(k: np.ndarray) -> np.ndarray:
    """2^k for integer k in [-9, 8] as float8_e4m3fn bytes (exact); other k raise."""
    k = np.asarray(k, np.int32)
    if ((k < -9) | (k > 8)).any():
        raise ValueError("power of two outside float8_e4m3fn range")
    return np.where(k >= -6, (k + 7) << 3, 1 << (k + 9)).astype(np.uint8)


def e4m3_decode(b: np.ndarray) -> np.ndarray:
    b = np.asarray(b, np.uint8).astype(np.int32)
    s = np.where(b & 0x80, -1.0, 1.0)
    e, m = (b >> 3) & 0xF, b & 7
    v = np.where(e == 0, m / 8.0 * 2.0 ** -6, (1 + m / 8.0) * np.ldexp(1.0, e - 7))
    return (s * v).astype(np.float32)


def mxfp4_to_nvfp4(wbytes: np.ndarray, scale: np.ndarray, gain: float = 1.0) -> dict:
    """Lossless repack of one MXFP4 tensor to NVFP4 (ModelOpt layout): the packed E2M1 bytes are
    unchanged, each 32-block's E8M0 2^e becomes two 16-block float8_e4m3fn scales 2^(e - emax + 8),
    and the per-tensor FP32 global scale 2^(emax - 135) * gain carries the rest - including the
    heal gain, exactly, since every block scale stays a power of two.

    A block whose codes are all zero is lossless at any scale and gets byte 0. A nonzero block more
    than 17 octaves below the tensor's largest scale cannot be represented: reported in
    `lossy_blocks`, and the caller refuses the tensor (nvidia reports zero such blocks in 1017)."""
    wb = np.asarray(wbytes).view(np.uint8)
    e = np.asarray(scale).view(np.uint8).astype(np.int32)
    R, nb32 = e.shape
    c = src_codes(wb).reshape(R, nb32, 32)
    nz = (c & 7).any(-1)                                   # any nonzero magnitude in the block
    emax = int(e[nz].max()) if nz.any() else 127
    k = e - emax + 8
    lossy = int((nz & (k < -9)).sum())
    k = np.clip(k, -9, 8)
    b16 = np.where(nz, e4m3_pow2(k), 0).astype(np.uint8)
    return {"weight": wb.copy(), "weight_scale": np.repeat(b16, 2, axis=1),
            "weight_scale_2": np.float32(np.ldexp(1.0, emax - 127 - 8) * gain), "lossy_blocks": lossy}


def nvfp4_dequant(w: np.ndarray, s16: np.ndarray, g: float) -> np.ndarray:
    c = src_codes(w)
    R, C = c.shape
    return (E2M1[c].reshape(R, C // 16, 16) * e4m3_decode(s16).reshape(R, C // 16, 1) * np.float32(g)).reshape(R, C)


def nvfp4_quantize(W: np.ndarray) -> dict:
    """RTN NVFP4 of a float matrix (dense overlay): 16-blocks, e4m3 block scale, FP32 global."""
    R, C = W.shape
    Wb = W.reshape(R, C // 16, 16).astype(np.float32)
    amax = np.abs(Wb).max(-1)
    g = float(amax.max()) / (448.0 * 6.0) if amax.max() > 0 else 1.0
    s = amax / 6.0 / g
    sb = _e4m3_round(s)
    sv = e4m3_decode(sb) * g
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.where(sv[..., None] > 0, Wb / sv[..., None], 0)
    mid = np.array([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.0], np.float32)     # midpoints of the E2M1 magnitudes
    idx = np.searchsorted(mid, np.abs(x)).astype(np.uint8)
    code = (idx | np.where(x < 0, 8, 0).astype(np.uint8)).reshape(R, C)
    packed = code[:, 0::2] | (code[:, 1::2] << 4)
    return {"weight": packed, "weight_scale": sb, "weight_scale_2": np.float32(g)}


def _e4m3_round(x: np.ndarray) -> np.ndarray:
    """nearest float8_e4m3fn byte for non-negative x (saturating at 448). Codes 0x00..0x7E are the
    finite non-negative values in increasing order, so a sorted search finds the neighbours."""
    tab = e4m3_decode(np.arange(0x7F, dtype=np.uint8))
    x = np.minimum(np.asarray(x, np.float32), tab[-1])
    hi = np.clip(np.searchsorted(tab, x), 1, tab.size - 1)
    lo = hi - 1
    return np.where(np.abs(x - tab[lo]) <= np.abs(tab[hi] - x), lo, hi).astype(np.uint8)
