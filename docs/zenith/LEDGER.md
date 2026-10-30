# ZENITH ledger (ops/logs/prof/ar8.sqlite)

90 decode steps; wall 48.23 ms/step = 20.73 tok/s; BW_peak 228 GB/s

| family | t ms/step | MB/step | floor ms | gap ms | achieved GB/s |
|---|---:|---:|---:|---:|---:|
| nvfp4_dense | 21.38 | 3672 | 16.11 | 5.28 | 172 |
| kquant_experts | 16.49 | 3003 | 13.17 | 3.32 | 182 |
| glue | 6.18 | 200 | 0.88 | 5.30 | 32 |
| bf16_head | 5.73 | 1324 | 5.81 | 0.00 | 231 |
| sparse_attn | 2.16 | 25 | 0.11 | 2.06 | 12 |
| fp8_dense | 1.36 | 315 | 1.38 | 0.00 | 232 |
| topk_index | 0.26 | 30 | 0.13 | 0.12 | 117 |
| launch/sync idle | 0.00 | | | 0.00 | |
| **total** | **48.23** | **8568** | **37.58** | **16.07** | |

AR zenith at these bytes: 37.6 ms = 26.6 tok/s; today 78% of roofline.

Prefill/verify (MT>1) kernel time in the trace, not in the step: kquant_experts 122 ms, nvfp4_dense 113 ms, fp8_dense 6 ms, bf16_head 1 ms
