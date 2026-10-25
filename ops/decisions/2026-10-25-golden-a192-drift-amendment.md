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
