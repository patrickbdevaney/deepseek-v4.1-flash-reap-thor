"""KT2 / KT3: QTIP-style bitshift-trellis expert format (thor-v1 qtypes "KT2", "KT3"), numpy oracle,
encoder and the torch LDLQ driver. Byte layout: ops/lib/thor_format.md "Trellis experts (KT2/KT3)".

FORMAT. A routed-expert weight row of `in` values is first rotated along the input dimension by a
blockwise randomized Hadamard transform (RHT, 256-wide blocks, one +-1 sign vector per layer and pass,
shared by every expert of the layer: w1 and w3 share one because they read the same activations).
The rotated row is cut into 128-weight blocks. Each block is a tail-biting bitshift trellis walk:
a circular stream of 128*b bits, and weight i's code is the L = 12-bit window starting at bit i*b
(mod 128*b). A window maps to a value through the "3INST" computed codebook of QTIP
(Tseng et al. 2024): x = w*89226354 + 64248484 (mod 2^32), h = (x & 0x8fff8fff) ^ 0x3b603b60, value =
fp16(h low) + fp16(h high), summed in fp32. One fp16 scale per block. Value = codebook(window) * scale.
Stored: b + 0.125 bpw (KT2 = 2.125, KT3 = 3.125), plus 1 bit per input column per layer for the signs.

WHY L = 12 (not QTIP's 16). The decoder costs the same for any L (the window is masked), but the
Viterbi encoder visits 2^L states per weight. Measured on unit Gaussians (tests/test_kt.py prints
it): L=12 gives MSE 0.069 (b=2) / 0.0186 (b=3) against L=16's ~0.063 / 0.016 -- and scalar
Lloyd-Max's 0.118 / 0.035 -- at 1/16 of the encode work. At L=12 the per-sequence state (4096 fp32
costs + packed backpointers) fits in shared memory, which is what makes the ~2.7e11-weight encode
a few GPU-hours on Thor (cuda/kernels/v41/kt_viterbi.cu) instead of days.

ENCODER. Block LDLQ (QuIP#/QTIP): columns of the rotated weights are taken 128 at a time; every
row's 128-block is encoded by a tail-biting Viterbi that minimizes the column-weighted squared error
(weights 1/U_kk^2, U = upper Cholesky of H^-1); the block's error E is propagated onto the remaining
columns as W[:, c1:] -= (E U_gg^-1) U[g, c1:], the exact vector-quantization analogue of GPTQ's
column update. H is the same per-expert input Hessian s07's K-quant GPTQ uses, rotated: R H R^T.

TAIL-BITING. Exact tail-biting Viterbi is 2^(L-b) runs. As in QTIP, a first unconstrained pass over
the 64 weights around the wrap point (positions 96..127, 0..31) fixes the shared overlap bits
(the low L-b bits of the first window), then the full pass is constrained to start on states with
those low bits and end on states whose high L-b bits equal them. Every path is a valid stream.

EXACTNESS CONTRACT (held by cuda/tests/gate_kt_host.cu on the CPU and gate_v41_units on the GPU):
  * codebook(), dequant_blocks(): float32 arithmetic, (lo + hi) then * scale -- bit-exact with DecKT.
  * rht_rows(): the in-place radix-2 butterfly (h = 1, 2, .., 128), then * 1/16 -- bit-exact with the
    CUDA rotation kernel.
  * viterbi(): float32 ((x - c)^2 * w then + m), first-minimum tie rule -- the CUDA encoder must
    return the same states.
"""
from __future__ import annotations

import ctypes
import hashlib
from pathlib import Path

import numpy as np

L = 12                     # trellis window bits
T = 128                    # weights per trellis block
RHT = 256                  # Hadamard block along the input dimension
H_WRAP = 32                # pass-1 half window around the wrap point
QTYPES = {"KT2": 2, "KT3": 3}
NS = 1 << L


def bits_of(qtype: str) -> int:
    return QTYPES[qtype]


def bpw(qtype: str) -> float:
    return bits_of(qtype) + 16.0 / T


def row_bytes(qtype: str, n_cols: int) -> int:
    """code bytes of one row (the fp16 scales live in a separate tensor)."""
    assert n_cols % T == 0
    return n_cols // T * 16 * bits_of(qtype)


def scale_bytes(n_cols: int) -> int:
    return n_cols // T * 2


# ----------------------------------------------------------------------------- codebook
def codebook(nbits: int = L) -> np.ndarray:
    """3INST value of every window 0..2^nbits-1, float32, exactly as kt_decode computes it."""
    s = np.arange(1 << nbits, dtype=np.uint64)
    x = (s * np.uint64(89226354) + np.uint64(64248484)) & np.uint64(0xFFFFFFFF)
    m = ((x & np.uint64(0x8FFF8FFF)) ^ np.uint64(0x3B603B60)).astype(np.uint32)
    lo = (m & 0xFFFF).astype(np.uint16).view(np.float16).astype(np.float32)
    hi = (m >> 16).astype(np.uint16).view(np.float16).astype(np.float32)
    return lo + hi


CB = codebook()
CB_RMS = float(np.sqrt(np.mean(CB.astype(np.float64) ** 2)))


# ----------------------------------------------------------------------------- RHT
def signs(tag: str, n: int) -> np.ndarray:
    """Deterministic +-1 (float32) for a (layer, pass) tag, e.g. "L7.w13". Stored in the build, so the
    seed rule only has to be stable within one build; it is recorded in thor.json anyway."""
    seed = int.from_bytes(hashlib.sha256(f"v41-kt-rht/{tag}".encode()).digest()[:8], "little")
    return np.where(np.random.default_rng(seed).integers(0, 2, n) == 1, -1.0, 1.0).astype(np.float32)


def pack_signs(s: np.ndarray) -> np.ndarray:
    """+-1 [n] -> U8 [n/8], bit k of byte j = 1 iff element 8j+k is negative."""
    return np.packbits((s < 0).astype(np.uint8), bitorder="little")


def unpack_signs(b: np.ndarray, n: int) -> np.ndarray:
    return np.where(np.unpackbits(np.asarray(b, np.uint8), bitorder="little")[:n] == 1, -1.0, 1.0).astype(np.float32)


def _fwht256(v: np.ndarray) -> np.ndarray:
    """Radix-2 butterflies on the last axis (length a multiple of 256), per 256-block, float32,
    stage order h = 1, 2, 4, .., 128; pair (j, j+h) -> (a+b, a-b). Unscaled."""
    shp = v.shape
    x = v.reshape(-1, RHT).astype(np.float32, copy=True)
    h = 1
    while h < RHT:
        y = x.reshape(-1, RHT // (2 * h), 2, h)
        a, b = y[:, :, 0, :].copy(), y[:, :, 1, :].copy()
        y[:, :, 0, :] = a + b
        y[:, :, 1, :] = a - b
        x = y.reshape(-1, RHT)
        h *= 2
    return x.reshape(shp)


def rht_rows(v: np.ndarray, sgn: np.ndarray) -> np.ndarray:
    """R v for every row: FWHT_256(v * sign) / 16 (orthonormal). Weights rotate the same way as
    activations: W' = W R^T has rows R w, and W' (R x) = W x."""
    return (_fwht256(np.asarray(v, np.float32) * sgn) * np.float32(1.0 / 16)).astype(np.float32)


def unrht_rows(v: np.ndarray, sgn: np.ndarray) -> np.ndarray:
    """R^T v = sign * FWHT_256(v) / 16."""
    return (_fwht256(np.asarray(v, np.float32)) * np.float32(1.0 / 16) * sgn).astype(np.float32)


# ----------------------------------------------------------------------------- streams
def states_to_words(states: np.ndarray, b: int) -> np.ndarray:
    """states [N, 128] (windows) -> uint32 [N, 4b]: chunk i (the low b bits of window i) at stream
    bits [i*b, i*b+b), little-endian bit order across words."""
    N = states.shape[0]
    chunks = (states & ((1 << b) - 1)).astype(np.uint8)                      # [N, T]
    bits = ((chunks[:, :, None] >> np.arange(b, dtype=np.uint8)) & 1).reshape(N, T * b)
    return np.packbits(bits, axis=1, bitorder="little").view(np.uint32).reshape(N, 4 * b)


def words_to_windows(words: np.ndarray, b: int) -> np.ndarray:
    """uint32 [N, 4b] -> windows [N, 128] (L-bit, circular)."""
    wd = np.ascontiguousarray(words).astype(np.uint64)
    nw = 4 * b
    out = np.empty((wd.shape[0], T), np.int64)
    for i in range(T):                                                        # a 12-bit window spans <= 2 words
        bit = i * b
        wi, sh = bit >> 5, bit & 31
        two = wd[:, wi] | (wd[:, (wi + 1) % nw] << np.uint64(32))
        out[:, i] = ((two >> np.uint64(sh)) & np.uint64(NS - 1)).astype(np.int64)
    return out


def dequant_blocks(words: np.ndarray, scales: np.ndarray, b: int) -> np.ndarray:
    """words uint32 [N, 4b], scales fp16 [N] -> float32 [N, 128], rotated domain (no un-RHT)."""
    v = CB[words_to_windows(words, b)]
    return (v * np.asarray(scales, np.float16).astype(np.float32)[:, None]).astype(np.float32)


def dequant(qtype: str, qweight: np.ndarray, scale: np.ndarray, in_f: int, sgn: np.ndarray | None) -> np.ndarray:
    """Stacked [.., out, row_bytes] codes + fp16 [.., out, in/128] scales -> float32 [.., out, in].
    sgn given -> un-rotated (the logical weight); None -> rotated domain (what DecKT emits)."""
    b = bits_of(qtype)
    lead = qweight.shape[:-1]
    nblk = in_f // T
    words = np.ascontiguousarray(qweight).view(np.uint32).reshape(-1, 4 * b)
    sc = np.asarray(scale).view(np.float16).reshape(-1)
    W = dequant_blocks(words, sc, b).reshape(*lead, nblk * T)
    return W if sgn is None else unrht_rows(W, sgn)


def _fwht256_t(x):
    """torch twin of _fwht256 (same butterflies, same float32 roundings)."""
    shp = x.shape
    y = x.reshape(-1, RHT).float().clone()
    h = 1
    while h < RHT:
        v = y.view(-1, RHT // (2 * h), 2, h)
        a, b = v[:, :, 0, :].clone(), v[:, :, 1, :].clone()
        v[:, :, 0, :] = a + b
        v[:, :, 1, :] = a - b
        h *= 2
    return y.reshape(shp)


def dequant_t(qtype: str, qweight, scale, in_f: int, sgn: np.ndarray | None, device="cuda"):
    """torch twin of dequant() for the replay and s07's w2 Hessian pass: [.., out, row_bytes] codes
    (numpy or torch uint8) + [.., out, in/128] fp16 scales -> float32 [.., out, in] on `device`.
    Bit-exact with dequant() (integer window extraction, the same float32 multiply and butterflies)."""
    import torch
    b = bits_of(qtype)
    nw = 4 * b
    q = torch.as_tensor(np.ascontiguousarray(qweight)).to(device)
    lead = tuple(q.shape[:-1])
    qb = q.reshape(-1, nw, 4).long()
    words = qb[..., 0] | (qb[..., 1] << 8) | (qb[..., 2] << 16) | (qb[..., 3] << 24)       # [N, nw]
    cbt = torch.from_numpy(CB).to(device)
    vals = torch.empty(words.shape[0], T, dtype=torch.float32, device=device)
    for i in range(T):
        bit = i * b
        wi, sh = bit >> 5, bit & 31
        if sh + L <= 32:
            win = (words[:, wi] >> sh) & (NS - 1)
        else:
            win = ((words[:, wi] >> sh) | (words[:, (wi + 1) % nw] << (32 - sh))) & (NS - 1)
        vals[:, i] = cbt[win]
    s = torch.as_tensor(np.ascontiguousarray(np.asarray(scale).view(np.float16))).to(device).float().reshape(-1)
    W = (vals * s[:, None]).reshape(*lead, in_f)
    if sgn is None:
        return W
    return _fwht256_t(W) * (1.0 / 16) * torch.from_numpy(np.asarray(sgn, np.float32)).to(device)


# ----------------------------------------------------------------------------- Viterbi (oracle)
def _dist(x, w):
    """[N] targets, [N] weights -> [N, NS] float32 ((x - c)^2 * w)."""
    d = x[:, None] - CB[None, :]
    return (d * d) * w[:, None]


def _run(X, Wt, b, order, init_mask=None):
    """Viterbi over steps `order` (indices into the 128 columns). -> (cost [N, NS], bp [len, N, 2^(L-b)])."""
    N = X.shape[0]
    nl = L - b
    cost = _dist(X[:, order[0]], Wt[:, order[0]])
    if init_mask is not None:
        cost = np.where(init_mask, cost, np.float32(np.inf)).astype(np.float32)
    bp = np.zeros((len(order), N, 1 << nl), np.uint8)
    for k in range(1, len(order)):
        r = cost.reshape(N, 1 << nl, 1 << b)
        p = r.argmin(2)
        m = np.take_along_axis(r, p[..., None], 2)[..., 0]
        bp[k] = p
        i = order[k]
        cost = np.broadcast_to(m[:, None, :], (N, 1 << b, 1 << nl)).reshape(N, NS) + _dist(X[:, i], Wt[:, i])
    return cost, bp


def _trace(bp, last, b):
    N = last.shape[0]
    nl = L - b
    n = bp.shape[0]
    st = np.empty((N, n), np.int64)
    st[:, n - 1] = last
    ar = np.arange(N)
    for k in range(n - 1, 0, -1):
        low = st[:, k] & ((1 << nl) - 1)
        st[:, k - 1] = (low << b) | bp[k][ar, low]
    return st


def viterbi(X: np.ndarray, Wt: np.ndarray, b: int) -> np.ndarray:
    """Tail-biting trellis encode. X [N, 128] targets (already divided by the scale), Wt [N, 128]
    nonnegative column weights. -> windows int64 [N, 128] forming valid circular streams."""
    X = np.asarray(X, np.float32)
    Wt = np.asarray(Wt, np.float32)
    N = X.shape[0]
    nl = L - b
    lowmask = (1 << nl) - 1
    order1 = list(range(T - H_WRAP, T)) + list(range(H_WRAP))
    cost, bp = _run(X, Wt, b, order1)
    st1 = _trace(bp, cost.argmin(1), b)
    o = st1[:, H_WRAP] & lowmask                                              # overlap bits at position 0
    init = (np.arange(NS)[None, :] & lowmask) == o[:, None]
    cost, bp = _run(X, Wt, b, list(range(T)), init_mask=init)
    cand = (o[:, None] << b) | np.arange(1 << b)[None, :]                     # states with high bits == o
    last = np.take_along_axis(cand, np.take_along_axis(cost, cand, 1).argmin(1)[:, None], 1)[:, 0]
    return _trace(bp, last, b)


# ----------------------------------------------------------------------------- block encode
def encode_blocks(Wb: np.ndarray, wt: np.ndarray, b: int, viterbi_fn=None) -> tuple[np.ndarray, np.ndarray]:
    """Wb [N, 128] rotated weights, wt [N, 128] column weights -> (words uint32 [N, 4b], scales fp16 [N]).
    Scale: init rms(w)/rms(codebook) (weighted), Viterbi, then the weighted least-squares refit
    sum(w c x)/sum(w c^2) rounded to fp16 (never worse than the init for the chosen codes)."""
    vf = viterbi_fn or viterbi
    Wb = np.asarray(Wb, np.float32)
    wt = np.asarray(wt, np.float32)
    rms = np.sqrt((wt * Wb * Wb).sum(1) / np.maximum(wt.sum(1), 1e-30))
    s0 = np.where(rms > 0, rms / CB_RMS, 1.0).astype(np.float16).astype(np.float32)
    s0 = np.where(s0 > 0, s0, np.float32(1.0))
    st = vf((Wb / s0[:, None]).astype(np.float32), wt, b)
    c = CB[st]
    num = (wt * c * Wb).sum(1)
    den = (wt * c * c).sum(1)
    s = np.where(den > 0, num / np.maximum(den, 1e-30), s0).astype(np.float16)
    s = np.where(np.isfinite(s.astype(np.float32)) & (s != 0), s, np.float16(0))
    return states_to_words(st, b), s


def quantize_rows(W: np.ndarray, qtype: str, sgn: np.ndarray, wt_cols: np.ndarray | None = None,
                  viterbi_fn=None) -> tuple[np.ndarray, np.ndarray]:
    """No-feedback encode (RTN analogue): W [R, C] logical -> (qweight U8 [R, row_bytes], scale fp16 [R, C/128])."""
    b = bits_of(qtype)
    R, C = W.shape
    Wr = rht_rows(W, sgn).reshape(R * (C // T), T)
    wt = np.ones_like(Wr) if wt_cols is None else np.broadcast_to(np.asarray(wt_cols, np.float32).reshape(C // T, T),
                                                                  (R, C // T, T)).reshape(-1, T)
    words, s = encode_blocks(Wr, wt, b, viterbi_fn)
    return words.view(np.uint8).reshape(R, -1), s.reshape(R, C // T)


# ----------------------------------------------------------------------------- CUDA Viterbi (ctypes)
_LIB = None
_LIB_PATH = Path(__file__).resolve().parents[2] / "cuda" / "build" / "libkt_viterbi.so"


def viterbi_cuda(X, Wt, b: int):
    """torch float32 cuda [N, 128] targets and weights -> int32 cuda [N, 128] windows. Same contract
    (and the same states) as viterbi(); one CTA per sequence, cuda/kernels/v41/kt_viterbi.cu."""
    import torch
    global _LIB
    if _LIB is None:
        if not _LIB_PATH.exists():
            raise FileNotFoundError(f"{_LIB_PATH} (build: cuda/scripts/build_v41.sh libkt_viterbi)")
        _LIB = ctypes.CDLL(str(_LIB_PATH))
        _LIB.kt_viterbi.restype = ctypes.c_int
        _LIB.kt_viterbi.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                    ctypes.c_void_p]
    X = X.contiguous().float()
    Wt = Wt.contiguous().float()
    out = torch.empty(X.shape, dtype=torch.int32, device=X.device)
    rc = _LIB.kt_viterbi(X.data_ptr(), Wt.data_ptr(), out.data_ptr(), int(X.shape[0]), int(b),
                         ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
    if rc != 0:
        raise RuntimeError(f"kt_viterbi failed: cuda error {rc}")
    return out


# ----------------------------------------------------------------------------- LDLQ (torch)
def hadamard256(device=None, dtype=None):
    import torch
    Hd = torch.ones(1, 1, dtype=torch.float32)
    while Hd.shape[0] < RHT:
        Hd = torch.cat([torch.cat([Hd, Hd], 1), torch.cat([Hd, -Hd], 1)], 0)
    return Hd.to(device=device, dtype=dtype or torch.float32) / 16.0


def rotate_t(W, sgn_t, Hd):
    """torch: rows of W [.., C] -> R w (matmul form of rht_rows; same math, not bit-exact)."""
    shp = W.shape
    x = (W * sgn_t).reshape(*shp[:-1], shp[-1] // RHT, RHT)
    return (x @ Hd).reshape(shp)                      # Hd symmetric: row-vector x @ Hd == (Hd x)^T


def quantize_ldlq(W, H, qtype: str, sgn: np.ndarray, *, damp: float = 0.01, viterbi_fn=None):
    """Block LDLQ into KT blocks. W [B, R, C] float32 (logical, heal gain already folded), H [B, C, C]
    float32 (sum of x x^T over the routed calibration tokens, logical domain). Both torch, any device.
    -> (qweight uint8 numpy [B, R, row_bytes], scale fp16 numpy [B, R, C/128])."""
    import torch
    from .gptq import _hinv_upper
    b = bits_of(qtype)
    B, R, C = W.shape
    dev = W.device
    Hd = hadamard256(dev)
    st = torch.from_numpy(sgn).to(dev)
    Wr = rotate_t(W.float(), st, Hd)                                          # [B, R, C]
    Hr = torch.empty_like(H, dtype=torch.float32)
    for k in range(B):                                                        # R H R^T, one expert at a time:
        Hr[k] = rotate_t(rotate_t(H[k].float(), st, Hd).T, st, Hd)            # no batch-sized temporaries
    U, _ = _hinv_upper(Hr, damp)
    del Hr
    vf = viterbi_fn or (viterbi_cuda if dev.type == "cuda" else None)
    words = np.empty((B, R, C // T, 4 * b), np.uint32)
    scales = np.empty((B, R, C // T), np.float16)
    for g in range(C // T):
        c0, c1 = g * T, (g + 1) * T
        Ug = U[:, c0:c1, c0:c1]
        wt = (1.0 / torch.diagonal(Ug, dim1=-2, dim2=-1) ** 2)                # [B, 128]
        Wg = Wr[:, :, c0:c1]
        rms = torch.sqrt((wt[:, None, :] * Wg * Wg).sum(-1) / wt.sum(-1)[:, None].clamp_min(1e-30))
        s0 = torch.where(rms > 0, rms / CB_RMS, torch.ones_like(rms)).half().float()
        s0 = torch.where(s0 > 0, s0, torch.ones_like(s0))
        X = (Wg / s0[..., None]).reshape(-1, T)
        Wt = wt[:, None, :].expand(B, R, T).reshape(-1, T)
        if vf is viterbi_cuda:
            stt = viterbi_cuda(X, Wt, b).long()
        else:
            stt = torch.from_numpy(viterbi(X.cpu().numpy(), Wt.cpu().numpy(), b)).to(dev)
        c = torch.from_numpy(CB).to(dev)[stt].reshape(B, R, T)
        num = (wt[:, None, :] * c * Wg).sum(-1)
        den = (wt[:, None, :] * c * c).sum(-1)
        s = torch.where(den > 0, num / den.clamp_min(1e-30), s0).half()
        s = torch.where(torch.isfinite(s.float()), s, torch.zeros_like(s))
        Q = c * s.float()[..., None]
        E = Wg - Q
        E1 = torch.linalg.solve_triangular(Ug, E, upper=True, left=False)    # E U_gg^-1
        if c1 < C:
            Wr[:, :, c1:] -= torch.bmm(E1, U[:, c0:c1, c1:])
        words[:, :, g] = states_to_words(stt.cpu().numpy(), b).reshape(B, R, 4 * b)
        scales[:, :, g] = s.cpu().numpy()
    return words.view(np.uint8).reshape(B, R, -1), scales
