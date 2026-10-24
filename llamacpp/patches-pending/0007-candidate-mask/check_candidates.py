#!/usr/bin/env python3
"""NumPy reference for the DeepSeek-V4.1 candidate-block mask, checked against the ggml CPU dump.

Transcribed from DeepSeek-V4.1-Flash-1017/inference/model.py:
  Indexer.forward            (l.527-582): index score, compress_lens mask, candidates, top-k
  select_candidate_blocks    (l.584-612): block max, pin the newest block, top-k blocks, drop -inf
  sparse_attn (kernel.py l.311-402): softmax over the picked rows plus the attention sink

For each case of test-dsv41-candidates.cpp: one candidate source (level one) and one later index
source (level two) whose top-k feeds one reader attention. Everything here runs in float64.

Optionally (--torch) also runs model.py's own select_candidate_blocks, extracted from the file with
ast and executed on the CPU (CUDA hidden), on the same scores, to check this transcription.

Usage: check_candidates.py DUMP_DIR [--torch MODEL_PY]
"""
import argparse
import ast
import os
import sys

import numpy as np


def load(path, dtype, shape):
    a = np.fromfile(path, dtype=dtype)
    assert a.size == int(np.prod(shape)), (path, a.size, shape)
    return a.reshape(shape)


# ---------------------------------------------------------------- model.py, transcribed

def index_score(q, k, w):
    """einsum("bshd,btd->bsht") -> relu -> * weights -> sum over heads. q [Q,H,D], k [T,D], w [Q,H]."""
    s = np.einsum("qhd,td->qht", q, k)
    return (np.maximum(s, 0.0) * w[:, :, None]).sum(axis=1)          # [Q, T]


def select_candidate_blocks(logits, compress_lens, topk_blocks, block_size):
    """model.py l.584-612. logits [Q, width] with unreachable at -inf; compress_lens [Q, 1] or int."""
    width = logits.shape[-1]
    pad = (-width) % block_size
    scores = np.pad(logits, [(0, 0), (0, pad)], constant_values=-np.inf)
    scores = scores.reshape(scores.shape[0], -1, block_size).max(axis=-1)
    num_blocks = scores.shape[-1]
    last = (np.asarray(compress_lens) - 1) // block_size                # floor division, like torch
    scores = np.where(np.arange(num_blocks)[None, :] == last, np.inf, scores)
    k = min(topk_blocks, num_blocks)
    top_idx = np.argsort(-scores, axis=-1, kind="stable")[:, :k]
    top_val = np.take_along_axis(scores, top_idx, axis=-1)
    keep = np.zeros_like(scores, dtype=bool)
    np.put_along_axis(keep, top_idx, top_val > -np.inf, axis=-1)
    return keep.repeat(block_size, axis=-1)[:, :width], keep, scores


def indexer_reader_topk(score2, compress_lens, candidates, index_topk, end_pos):
    """level two (l.574-582): mask to the candidates, top-k, sort, -1 for unreachable."""
    s = np.where(candidates, score2, -np.inf) if candidates is not None else score2
    topk = min(index_topk, end_pos)
    idx = np.sort(np.argsort(-s, axis=-1, kind="stable")[:, :topk], axis=-1)
    return np.where(idx < np.asarray(compress_lens), idx, -1)


def sparse_attn(q, kv, sink, idxs, scale):
    """kernel.py sparse_attn: q [Q,H,D], kv [T,D] (k = v), sink [H], idxs [Q,K] with -1 = none."""
    Q, H, D = q.shape
    o = np.zeros((Q, H, D))
    for i in range(Q):
        sel = idxs[i][idxs[i] >= 0]
        kk = kv[sel]                                          # [K, D]
        s = (q[i] @ kk.T) * scale                             # [H, K]
        m = s.max(axis=1, initial=-1e30)
        e = np.exp(s - m[:, None])
        den = e.sum(axis=1) + np.exp(sink - m)
        o[i] = (e @ kk) / den[:, None]
    return o


# ---------------------------------------------------------------- per case

def run_case(d, line, torch_fn):
    name, S, Q, N, IH, ID, AH, AD, BS, TOPKB, ITOPK, K = line.split()
    S, Q, N, IH, ID, AH, AD, BS, TOPKB, ITOPK, K = map(int, (S, Q, N, IH, ID, AH, AD, BS, TOPKB, ITOPK, K))
    p = os.path.join(d, name)
    f = lambda n, shape: load(f"{p}.{n}.f32", np.float32, shape).astype(np.float64)
    idx_k = f("idx_k", (S, N, ID))
    q1, q2 = f("q1", (S, Q, IH, ID)), f("q2", (S, Q, IH, ID))
    w1, w2 = f("w1", (S, Q, IH)), f("w2", (S, Q, IH))
    kv = f("kv", (S, N, AD))
    qa = f("qa", (S, Q, AH, AD))
    sinks = f("sinks", (AH,))
    n_vis = load(f"{p}.n_vis.i32", np.int32, (S, Q))
    g_score1 = load(f"{p}.score1.f32", np.float32, (S, Q, N))
    nb = N // BS
    g_cand = np.fromfile(f"{p}.cand.f32", dtype=np.float32)
    g_cand = g_cand.reshape(S, Q, nb) if g_cand.size else None
    g_top = load(f"{p}.top.i32", np.int32, (S, Q, K))
    g_out = load(f"{p}.out.f32", np.float32, (S, Q, AH, AD)).astype(np.float64)

    res = dict(name=name, blocks_equal=True, n_rows_blocks=0, topk_equal=True, extra_unreachable_kept=0,
               max_err_out=0.0, max_err_score=0.0, min_boundary_gap=np.inf, torch_equal=None, mask_active=False)

    for s in range(S):
        nv = n_vis[s]                                    # compress_lens per query, ratio 1
        pos0 = int(nv[0]) - 1
        if pos0 == 0 and Q > 1:
            # prefill, start_pos == 0: width = seqlen // ratio, causal mask by compress_lens
            width = Q
            k_rows = idx_k[s, :width]
            cl = nv[:, None].astype(np.int64)
            sc1 = index_score(q1[s], k_rows, w1[s])
            sc1 = np.where(np.arange(width)[None, :] >= cl, -np.inf, sc1)
            sc2 = index_score(q2[s], k_rows, w2[s])
            sc2 = np.where(np.arange(width)[None, :] >= cl, -np.inf, sc2)
            cand, keep_b, bscore = select_candidate_blocks(sc1, cl, TOPKB, BS)
            idxs = indexer_reader_topk(sc2, cl, cand, ITOPK, width)
            rows = [(qq, cl[qq, 0]) for qq in range(Q)]
            per_q = [(qq, sc1[qq], cand[qq], keep_b[qq], bscore[qq], idxs[qq]) for qq in range(Q)]
            torch_in = (sc1, cl)
        else:
            # decode (start_pos > 0, one query); a mid-sequence ubatch is the same per query
            per_q, torch_in = [], []
            for qq in range(Q):
                width = int(nv[qq])                        # end_pos // ratio == compress_lens
                k_rows = idx_k[s, :width]
                sc1 = index_score(q1[s, qq:qq+1], k_rows, w1[s, qq:qq+1])
                sc2 = index_score(q2[s, qq:qq+1], k_rows, w2[s, qq:qq+1])
                cand, keep_b, bscore = select_candidate_blocks(sc1, width, TOPKB, BS)
                idxs = indexer_reader_topk(sc2, width, cand, ITOPK, width)
                per_q.append((qq, sc1[0], cand[0], keep_b[0], bscore[0], idxs[0]))
                torch_in.append((sc1, width))

        if torch_fn is not None:
            ok = True
            import torch
            items = [torch_in] if isinstance(torch_in, tuple) else torch_in
            for sc, cl in items:
                t_logits = torch.from_numpy(sc)
                t_cl = torch.from_numpy(cl) if isinstance(cl, np.ndarray) else cl
                t_keep = torch_fn(t_logits, t_cl, TOPKB, BS).numpy()
                mine = select_candidate_blocks(sc, cl, TOPKB, BS)[0]
                ok &= bool(np.array_equal(t_keep, mine))
            res["torch_equal"] = ok if res["torch_equal"] is None else (res["torch_equal"] and ok)

        for qq, sc1, cand, keep_b, bscore, idxs in per_q:
            v = int(nv[qq])
            # level-one score agreement (finite entries)
            g = g_score1[s, qq, :len(sc1)].astype(np.float64)
            fin = np.isfinite(sc1)
            assert np.array_equal(fin, np.isfinite(g)), (name, s, qq)
            if fin.any():
                res["max_err_score"] = max(res["max_err_score"], float(np.abs(g[fin] - sc1[fin]).max()))
            # block ids: compare every block holding a reachable row
            nb_ref = keep_b.shape[0]
            if nb_ref > TOPKB:
                res["mask_active"] = True
            n_reach_b = (v + BS - 1) // BS
            if g_cand is not None:
                g_keep = g_cand[s, qq] == 0.0
                same = np.array_equal(g_keep[:n_reach_b], keep_b[:n_reach_b])
                res["blocks_equal"] &= bool(same)
                res["extra_unreachable_kept"] += int(g_keep[n_reach_b:].sum())
            else:
                # no mask built: the reference must keep every reachable block
                res["blocks_equal"] &= bool(keep_b[:n_reach_b].all())
            res["n_rows_blocks"] += 1
            # how close the K-th and (K+1)-th unpinned reachable block scores are (tie risk)
            fb = np.sort(bscore[:n_reach_b][np.isfinite(bscore[:n_reach_b])])[::-1]
            kk = TOPKB - 1                                   # one slot goes to the pinned block
            if fb.size > kk:
                res["min_boundary_gap"] = min(res["min_boundary_gap"], float(fb[kk - 1] - fb[kk]))
            # reader top-k, as a set of reachable rows
            ref_set = set(int(x) for x in idxs if x >= 0)
            g_set = set(int(x) for x in g_top[s, qq] if 0 <= x < v)
            res["topk_equal"] &= ref_set == g_set
            # reader attention
            o = sparse_attn(qa[s, qq:qq+1], kv[s], sinks, idxs[None, :], 1.0 / np.sqrt(AD))
            res["max_err_out"] = max(res["max_err_out"], float(np.abs(o[0] - g_out[s, qq]).max()))
    return res


def load_torch_select(model_py):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""          # CPU only; never touch the GPU
    import torch
    import torch.nn.functional as F
    src = open(model_py).read()
    tree = ast.parse(src)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "select_candidate_blocks")
    code = ast.get_source_segment(src, fn)
    ns = {"torch": torch, "F": F}
    exec(compile("from __future__ import annotations\n" + code, model_py, "exec"), ns)
    return ns["select_candidate_blocks"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--torch", dest="model_py", default=None)
    args = ap.parse_args()
    torch_fn = load_torch_select(args.model_py) if args.model_py else None

    fails = 0
    worst_out = worst_score = 0.0
    with open(os.path.join(args.dump, "cases.txt")) as fh:
        for line in fh:
            r = run_case(args.dump, line, torch_fn)
            ok = r["blocks_equal"] and r["topk_equal"] and r["max_err_out"] < 1e-4 and r["torch_equal"] is not False
            fails += not ok
            worst_out = max(worst_out, r["max_err_out"])
            worst_score = max(worst_score, r["max_err_score"])
            print(f"{r['name']:<20} mask_active={int(r['mask_active'])} queries={r['n_rows_blocks']:3d} "
                  f"blocks_equal={int(r['blocks_equal'])} topk_equal={int(r['topk_equal'])} "
                  f"max|out err|={r['max_err_out']:.3e} max|score err|={r['max_err_score']:.3e} "
                  f"min_boundary_gap={r['min_boundary_gap']:.3e} unreachable_kept={r['extra_unreachable_kept']}"
                  + ("" if r["torch_equal"] is None else f" model.py_select_equal={int(r['torch_equal'])}")
                  + ("" if ok else "  FAIL"))
    print(f"{'FAIL' if fails else 'OK'}: {fails} failing case(s); worst attention max|err| {worst_out:.3e}, "
          f"worst level-one score max|err| {worst_score:.3e}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
