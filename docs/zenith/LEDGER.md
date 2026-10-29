# ZENITH ledger (ops/logs/prof/ov2.sqlite)

105 decode steps; wall 55.69 ms/step = 17.96 tok/s; BW_peak 228 GB/s

| family | t ms/step | MB/step | floor ms | gap ms | achieved GB/s |
|---|---:|---:|---:|---:|---:|
| kquant_experts | 21.18 | 3003 | 13.17 | 8.01 | 142 |
| nvfp4_dense | 20.55 | 3672 | 16.11 | 4.45 | 179 |
| glue | 7.58 | 200 | 0.88 | 6.70 | 26 |
| bf16_head | 6.07 | 1324 | 5.81 | 0.27 | 218 |
| sparse_attn | 2.23 | 25 | 0.11 | 2.12 | 11 |
| fp8_dense | 1.42 | 315 | 1.38 | 0.03 | 223 |
| topk_index | 0.27 | 30 | 0.13 | 0.14 | 112 |
| launch/sync idle | 0.00 | | | 0.00 | |
| **total** | **55.69** | **8568** | **37.58** | **21.71** | |

AR zenith at these bytes: 37.6 ms = 26.6 tok/s; today 67% of roofline.

Prefill/verify (MT>1) kernel time in the trace, not in the step: kquant_experts 322 ms, nvfp4_dense 186 ms, fp8_dense 7 ms, bf16_head 1 ms
