#!/usr/bin/env python3
"""Export the Engram hashing tables for the CUDA server, and a hash fixture to gate the C++ port.

Everything is produced BY THE REFERENCE ITSELF (inference/engram.py imported from the checkpoint),
not re-derived, so the C++ hasher is checked against the code that trained the tables:

  engram_tokmap.bin   int32[129280] compressed token map, then int64: pad id, multipliers [2][4],
                      primes [2][24], offsets [2][24]   (layout read by EngramHashTables::load)
  engram_hash_fixture.bin   cases for tests/gate_engram_hash.cpp:
                      int32 n_cases; per case: int32 n, int32 tokens[n], uint8 is_image[n],
                      int64 ids[n][2][24]  (prefill of the whole case, start_pos 0)

CPU only, no weights: needs the checkpoint's tokenizer + inference/ code and sympy.
Usage: python tools/engram_meta.py --ckpt ~/models/DeepSeek-V4.1-Flash-1017 --out <dir>
"""
import argparse
import json
import os
import struct
import sys
import types

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.expanduser("~/models/DeepSeek-V4.1-Flash-1017"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sys.path.insert(0, os.path.join(a.ckpt, "inference"))
    import engram as E  # the reference module
    from transformers import AutoTokenizer

    cfg = json.load(open(os.path.join(a.ckpt, "inference", "config.json")))
    args = types.SimpleNamespace(**cfg)
    args.max_batch_size, args.max_seq_len = 1, 4096
    args.engram_pad_id = cfg["engram_pad_id"]
    tok = AutoTokenizer.from_pretrained(a.ckpt)
    layout = E.EngramLayout.from_args(args)
    assert tuple(layout.num_embeddings) == (384006168, 384016682)
    st = E.NgramHashState(args, layout, tok)

    with open(os.path.join(a.out, "engram_tokmap.bin"), "wb") as f:
        tm = st.token_map.to(torch.int32).numpy()
        assert tm.shape[0] == cfg["vocab_size"], tm.shape
        f.write(tm.tobytes())
        f.write(struct.pack("<q", int(st.pad_id)))
        f.write(st.multipliers.to(torch.int64).numpy().tobytes())              # [2][4]
        f.write(st.primes.to(torch.int64).reshape(2, 24).numpy().tobytes())    # [2][3][8]
        f.write(st.offsets.to(torch.int64).reshape(2, 24).numpy().tobytes())   # [2][24]

    # fixture: plain text, a long run, a case with an image span, and random ids
    rng = np.random.default_rng(7)
    texts = ["The quick brown fox jumps over the lazy dog.", "def f(x):\n    return x * 2\n" * 3,
             "深度求索 DeepSeek-V4.1 发布 1M context " * 2]
    cases = []
    for t in texts:
        ids = tok.encode(t, add_special_tokens=False)
        cases.append((ids, [0] * len(ids)))
    ids = rng.integers(0, cfg["vocab_size"], 300).tolist()
    img = [0] * 300
    for i in range(100, 140): img[i] = 1
    cases.append((ids, img))
    with open(os.path.join(a.out, "engram_hash_fixture.bin"), "wb") as f:
        f.write(struct.pack("<i", len(cases)))
        for ids, img in cases:
            st.cache.zero_()
            x = torch.tensor([ids])
            mask = torch.tensor([[not v for v in img]])
            h = st(x, 0, mask if any(img) else None)          # [1, n, 2, 24]
            f.write(struct.pack("<i", len(ids)))
            f.write(np.asarray(ids, np.int32).tobytes())
            f.write(np.asarray(img, np.uint8).tobytes())
            f.write(h[0].to(torch.int64).numpy().tobytes())
    print(f"wrote engram_tokmap.bin + engram_hash_fixture.bin ({len(cases)} cases) to {a.out}")


if __name__ == "__main__":
    main()
