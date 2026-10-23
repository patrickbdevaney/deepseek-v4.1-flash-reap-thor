#!/usr/bin/env python3
"""s13 results export: what stays out of git, and the secret pattern. CPU only.

    .venv/bin/python ops/tests/test_s13.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ops import s13_results as R  # noqa: E402


def test_excluded():
    for rel in ["corpus/pool_finance.json", "corpus/images/a.png", "s03/engram/x.json", "s04/archive_v2_rowmismatch/m.json",
                "corpus.v1_defective/rows.jsonl", "s04/replay/_embed/meta.json", "s13_bulk/README.md"]:
        assert R.excluded(rel), rel
    for rel in ["corpus/sources.json", "corpus/rows.jsonl", "s04/keepset.json", "s04/masks/hope_sum_pf0.json",
                "s10/heal/decision.json"]:
        assert not R.excluded(rel), rel


def test_secret_pattern():
    bad = ["hf_" + "a" * 34, "ghp_" + "B" * 36, "github_pat_" + "c" * 30, "AKIA" + "Q" * 16, "100.116.9.21",
           "-----BEGIN OPENSSH PRIVATE KEY"]
    good = ["100.0.0.1", "192.168.1.5", "hf_hub_download", "sha256 1ace9b33", "ghp_short"]
    assert all(R.SECRETS.search(s) for s in bad), [s for s in bad if not R.SECRETS.search(s)]
    assert not any(R.SECRETS.search(s) for s in good), [s for s in good if R.SECRETS.search(s)]


if __name__ == "__main__":
    test_excluded(); test_secret_pattern()
    print("2/2 passed")
