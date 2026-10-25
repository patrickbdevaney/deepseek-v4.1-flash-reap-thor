# Owner decision 2026-10-25: golden A192 under the same drift rule (decided BEFORE the diagnostic result)

golden A192 (s08 golden A192, 2026-10-25 03:12) FAILED the pooled near-tie rule as registered (1c54996 criteria):
46 disagreements, 41 near-ties (89.1% < 90%); all 5 confident disagreements on doc1k (978/1024); long20k 1024/1024,
chat passes. The isolated head reports 0 confident disagreements (all 5 "isolated head agrees").

The golden_src amendment (2026-10-24-golden-src-drift-amendment.md) explicitly left golden A192 unchanged, so it was
put to the owner. Owner choice, 2026-10-25: "Same rule, test first" -- pause the GGUF ladder for the per-row
diagnostic, then apply the SAME two conditions:

1. at every confident chained disagreement the isolated head is "agrees" or "near-tie", and
2. at that position the ISOLATED per-row cos (row_cos.txt pass 1) is >= 0.98 at every layer 0..39.

Both hold -> DRIFT: golden A192 passes under this amendment; continue golden_store, decode, ctx1m, server (all
unchanged), Thor publish, overlay. Either fails -> BUG: no amendment, debug, re-run unchanged.
The Thor card states both golden gates failed the near-tie rule as registered and passed under these amendments.

Diagnostic: ops/diag_golden_a192.sh -> ops/logs/diag_golden_a192/{gate.log,row_cos.txt}.

## Result (2026-10-25 06:53): BUG, in the gate harness, not drift

The doc1k-only diagnostic (`ops/logs/diag_golden_a192/`) did not reproduce the 03:12 run's head numbers, while
every per-layer hidden-state statistic matched it exactly (chained final mean cos 0.966632 in both; same
per-layer min/mean):

| run | chained argmax (doc1k) | pooled near-ties | isolated head agreement | isolated confident |
|---|---|---|---|---|
| 03:12 chain (chat, doc1k, long20k) | 978/1024 | 41/46 (89.1%) FAIL | 1021/1024 | 0 |
| 06:53 diagnostic (doc1k only) | 989/1024 | 33/35 (94.3%) "PASS" | 844/1024 | 178 |

Identical hidden states with different argmaxes means the head check, not the model. Condition (1) of this
amendment also fails (pos 850: "isolated head confident flip"), so by the registered rule the verdict is BUG, and
the diagnostic's "PASS" is not used.

Cause: `cuda/tests/gate_v41_golden.cu` `head_at` copied each row into `p.sc.h`/`p.sc.pre` with a synchronous-API
D2D `cudaMemcpy`, which may return before the copy completes and (built `--default-stream per-thread`) is not
ordered against stream `st`, where `head_row` then ran. The head could read part of the previously copied row
(the other stream's row r, or chained row r-1), so argmaxes varied with timing. The served model path does not use
this pattern (no D2D `cudaMemcpy` in `cuda/src` V4.1 code or `cuda/server`). Fix: both copies are
`cudaMemcpyAsync(..., st)`.

Action: no amendment applies. `soft_fail.golden_arm` is renamed with the fix commit and chain2 re-runs golden A192
exactly as registered (rule 1c54996, all three prompts, same reference). Its result decides; if it fails, it is
escalated again with this harness bug ruled out.
