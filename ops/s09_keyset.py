#!/usr/bin/env python3
"""Keep-set checks for the GGUF path (s09). There is one fingerprint scheme, ops/lib/keepset.py; this
file only adds the checks the converter needs. A mismatch is fatal; it is never a warning.

The contract with s05: the master directory carries `reap_keepset.json`, written by keepset.save()
as {"sha256", "n_keep", "base_repo", "revision", "keep": {"<layer>": [source ids, ascending]}}.
The kept experts are renumbered 0..n-1 in ascending source-id order (keepset.remap), so
new expert j of layer L is keep[L][j].

  python ops/s09_keyset.py KEEPSET.json      prints the fingerprint (refuses a hand-edited file)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ops.lib import keepset  # noqa: E402
from ops.lib.keepset import KeepsetMismatch  # noqa: E402,F401

KEEPSET_FILE = "reap_keepset.json"


def check_master_keepset(path: Path, *, n_layers: int, n_experts: int, n_source: int,
                         expect_sha: str | None) -> tuple[dict[str, list[int]], str]:
    """Load the master's keep-set (self-consistency checked by keepset.load), require the arbitrated
    fingerprint, and check its shape against the master's config. -> (keep, fingerprint)."""
    if not path.exists():
        raise KeepsetMismatch(f"{path} is missing: refusing to convert an un-fingerprinted checkpoint")
    keep, fp = keepset.load(path, expect=expect_sha)
    try:
        keepset.canonical(keep, n_layers=n_layers, n_experts=n_source, n_keep=n_experts)
    except ValueError as e:
        raise KeepsetMismatch(f"{path}: {e}") from None
    return keep, fp


if __name__ == "__main__":
    print(keepset.load(Path(sys.argv[1]))[1])
