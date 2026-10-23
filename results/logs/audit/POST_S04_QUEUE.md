# Edits deferred until s04 finishes (s04 re-imports ops/lib/{gates,metrics,orchestrate,replay,...} per layer)
1. ops/lib/gates.py pruned_floor: fail closed when f.get("long_bins_required", True) and no long bin sampled (s06_s10.md C1).
2. ops/lib/metrics.py: refuse non-finite scored kl32/nll (s06_s10.md R2); gates point checks NaN-safe (use `not x <= max`).
3. ops/lib/orchestrate.py batching: refuse k<1 (s06_s10.md R6).
4. ops/lib/replay.py:531 missing-layer gain -> KeyError (N1); phase-A memfence (N3).
5. Verify s04/arbitration.json: every arm's kl32 mean finite and token counts equal (s04_s05.md R5) before s05 consumes keepset.
6. Delete dead _embed replay files (s04 ~7.6 GiB) — ops.md disk plan.
