# Owner decision 2026-10-24: golden_src drift amendment (decided BEFORE the diagnostic result)

Owner, verbatim: "yes, record the amendment if it's drift and publish; fix the bug though"

Context: golden_src run 4 (commit 1c54996 criteria, fixed before that run) FAILED the pooled near-tie rule:
41 disagreements, 33 near-ties (80% < 90%); all 8 confident disagreements on doc1k. Agreement and final-cos
rules passed on every prompt with >= 100 rows (doc1k 985/1024, long20k 1024/1024).

The diagnostic is ops/diag_golden_doc1k.sh (gate commit 793b5af: isolated head + per-row trajectories), run at
boot after the GPU-wedge reboot. Its output is ops/logs/diag_golden_doc1k/{gate.log,row_cos.txt}.

## Rule fixed now, before reading the diagnostic

It is DRIFT (amend + continue to golden A192, serving gates, Thor + overlay publish) only if BOTH hold for every
confident chained disagreement:
1. the isolated head (CUDA L39 + head on the reference's L38 rows) is "agrees" or "near-tie" at that position, and
2. at that position the ISOLATED per-row cos (row_cos.txt, pass 1) is >= 0.98 at every layer 0..39
   (each layer computes that row as the reference does when given the reference's input).

Otherwise it is a BUG: no amendment; debug the layer/row that fails (1) or (2), fix, re-run golden_src under
the unchanged 1c54996 rule.

Even under the amendment the release still requires every downstream gate unchanged: golden A192, golden_store,
decode determinism/consistency, server, ctx1m. The Thor card states that golden_src failed the near-tie rule as
registered and passed under this owner amendment, with the diagnostic numbers.
