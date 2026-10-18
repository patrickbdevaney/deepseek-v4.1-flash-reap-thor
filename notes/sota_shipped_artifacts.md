# Shipped compressed artifacts of 400B–1T+ open-weight MoE models that approach or fit ≤128 GB (2025–2026)

Research date: 2026-10-17. Scope: real, downloadable artifacts only (HF/GitHub), with method, bytes, ratio, measured evals, hardware and author. "GB" = decimal (HF file listing), "GiB" = binary (llama.cpp/ik_llama.cpp tables). Sizes are copied as the source reports them; the unit is kept per row. "UNEVALUATED" = the artifact ships with no benchmark/PPL/KLD against its source.

## Landscape table of real artifacts (all families, ordered by source model)

### Takeaway
Nothing from a ≥500B source ships at ≤128 GB with a benchmark table against the unpruned model. The only ≤~130 GB artifacts of ≥500B sources are (a) ubergarm's 1.66-bpw DeepSeek-R1-0528 (130.2 GiB, PPL +52%), (b) 0xSero's REAP-50% + IQ2_M GLM-5 (130 GB, pilot eval only vs. a sibling quant), and (c) 1-bit Unsloth/ubergarm quants of 400B-class models (Qwen3.5-397B smol-IQ1_KT at 88.8 GiB, PPL +22%). Everything with a real eval table (Cerebras REAP, 0xSero Router-KD, kwanhee Kimi-K2.5, Altar-1) lands at 140–330 GB.

### Cited Findings (one row per artifact; link = repo)

**Source-size anchors (bytes), needed for ratios**
- DeepSeek-R1-0528 Q8_0 reference = 666 GiB (ubergarm's baseline) — [ubergarm/DeepSeek-R1-0528-GGUF](https://huggingface.co/ubergarm/DeepSeek-R1-0528-GGUF)
- Kimi-K2 (1T) full = 1.09 TB per Unsloth; Kimi-K2-Thinking native-INT4 "Q4_X" = 543.6 GiB — [Unsloth K2 guide](https://docs.unsloth.ai/basics/kimi-k2-how-to-run-locally); [ubergarm/Kimi-K2-Thinking-GGUF](https://huggingface.co/ubergarm/Kimi-K2-Thinking-GGUF)
- Kimi-K2.5 full ≈ 600–630 GB — [Unsloth K2.5 guide](https://unsloth.ai/docs/models/tutorials/kimi-k2.5)
- Qwen3.5-397B-A17B BF16 = 738.5 GiB; Q8_0 = 392.5 GiB — [ubergarm/Qwen3.5-397B-A17B-GGUF](https://huggingface.co/ubergarm/Qwen3.5-397B-A17B-GGUF)
- Ling-1T Q8_0 = 989.7 GiB — [ubergarm2/Ling-1T-GGUF](https://huggingface.co/ubergarm2/Ling-1T-GGUF)
- Qwen3-Coder-480B: the 25%-pruned 363B BF16 GGUF is 726 GB, so unpruned BF16 ≈ 960 GB (inference) — [unsloth/Qwen3-Coder-REAP-363B-A35B-GGUF](https://huggingface.co/unsloth/Qwen3-Coder-REAP-363B-A35B-GGUF)
- GLM-4.6 (355B): 25%-pruned 268B BF16 GGUF = 538 GB — [unsloth/GLM-4.6-REAP-268B-A32B-GGUF](https://huggingface.co/unsloth/GLM-4.6-REAP-268B-A32B-GGUF)
- GLM-5 = 744B total, 256 routed experts — [0xSero/GLM-5-REAP-50pct-UD-IQ2_M-GGUF](https://huggingface.co/0xSero/GLM-5-REAP-50pct-UD-IQ2_M-GGUF)
- GLM-5.2 ≈ 744–763B (card) / ~753B; 78 layers (3 dense + 75 MoE), 256 routed experts top-8 + 1 shared — [0xSero/GLM-5.2-504B](https://huggingface.co/0xSero/GLM-5.2-504B)
- GLM-5.3 = 753B, 8-of-256 experts, ~40B active — [AikidoSec/altar-1](https://huggingface.co/AikidoSec/altar-1)
- GLM-5.3-Flash = 321.3B, 328 GB on disk (FP8), 288 routed experts/layer — [OpenMOSE/GLM-5.3-Flash-REAP-250B-A18B](https://huggingface.co/OpenMOSE/GLM-5.3-Flash-REAP-250B-A18B)
- DeepSeek-V4-Flash = 284B/13B active, official checkpoint 156.4 GB; DeepSeek-V4-Pro = 1.6T/49B active — [Unsloth V4 docs](https://unsloth.ai/docs/models/deepseek-v4)
- DeepSeek-V4-Pro-0813 = 1.57T, 384 routed experts top-6, upstream weights FP4, F16 GGUF 812.7 GiB (experts = 96.4% of bytes) — [6block/DeepSeek-V4-Pro-0813-GGUF](https://huggingface.co/6block/DeepSeek-V4-Pro-0813-GGUF)
- Kimi-K3 = 2.8T total; UD-IQ1_S = 634.6 GB (1.83 bpw) — [Classmethod Pulsar article](https://dev.classmethod.jp/en/articles/dgx-spark-pulsar-ssd-expert-streaming/)

**DeepSeek-V3/R1 family (671B)**
- Unsloth UD-IQ1_S DeepSeek-R1 (Jan 2025): 1.58-bit, 131 GB; MoE at 1.56-bit, MoE down_proj mix 2.06/1.56-bit; UD-IQ1_M 158 GB, UD-IQ2_XXS 183 GB, UD-Q2_K_XL 212 GB. Quality evidence = a Flappy-Bird prompt GIF only; recommends --min-p 0.05 for the 1.58-bit. Offload guidance for 24 GB GPU (7 layers) and 128 GB Mac (~59 layers). UNEVALUATED beyond anecdote — [Unsloth R1 guide](https://unsloth.ai/docs/basics/tutorials-how-to-fine-tune-and-run-llms/deepseek-r1-how-to-run-locally)
- ubergarm IQ1_S_R4 DeepSeek-R1-0528: 130.203 GiB, 1.664 bpw, PPL 4.8805 vs Q8_0 3.2130 (+51.9%); ratio vs Q8_0 = 666/130.2 = 5.1x; ran on an AM5 rig with 2x64 GB DDR5 + one GPU (~116 GiB RAM + ~22 GB VRAM); author: "world's smallest working R1-0528 quant", "only for the desperate". Ladder: IQ2_K_R4 220 GiB 2.799 bpw PPL 3.5069 (+9.1%, ~15 tok/s early test); IQ3_KS 281.5 GiB PPL 3.2983; IQ3_K_R4 301 GiB PPL 3.2730 (~12 tok/s on 256 GB RAM + 2x A6000); IQ4_KS_R4 368 GiB PPL 3.2286. ik_llama.cpp only — [ubergarm/DeepSeek-R1-0528-GGUF](https://huggingface.co/ubergarm/DeepSeek-R1-0528-GGUF)
- Unsloth Dynamic DeepSeek-V3.1 GGUFs, Aider Polyglot pass-2 (community-run, ~3 runs averaged): thinking mode 1-bit 57.8 / 2-bit 66.7 / 3-bit 75.6 vs full 76.1; non-thinking 1-bit 55.7 (IQ1_M, 206 GB) / 2-bit 65.8 / 3-bit 68.4 / 4-bit 69.7 / 5-bit 70.7 (Q5_K_XL, 484 GB) vs full 71.6; 1-bit stated as 192 GB. Non-Unsloth comparison: IQ2_XXS 164 GB 43.6, IQ2_M 215 GB 56.6, Q2_K_L 239 GB 64.0 vs Unsloth IQ2_XXS 225 GB 61.2 — [Unsloth Aider page](https://unsloth.ai/docs/basics/dynamic-3.0-ggufs/unsloth-dynamic-ggufs-on-aider-polyglot)
- Intel OPEA DeepSeek-R1 INT2-mixed (AutoRound): routed experts int2 g64 sym, non-expert/shared/first-3-layers int4 g128, lm_head 16-bit; ~200 GB per the AutoRound PR; MMLU 0.8514→0.8302, ARC-C 0.6212→0.6084, HellaSwag 0.6935→0.6657, Winogrande 0.7932→0.7940; quantized on 5x80 GB GPUs + 1.4–1.6 TB RAM; inference 4x80 GB via transformers only ("could NOT run on other serving frameworks"); int2 slower than int4 on CUDA — [OPEA/DeepSeek-R1-int2-mixed-sym-inc](https://huggingface.co/OPEA/DeepSeek-R1-int2-mixed-sym-inc); [auto-round PR #962](https://github.com/intel/auto-round/pull/962.diff)
- OPEA DeepSeek-R1-0528-int2-mixed-AutoRound: same recipe (pile-10k, 512 samples, 400 iters); no eval table, only sample generations, one of which answered an unrelated question — [OPEA/DeepSeek-R1-0528-int2-mixed-AutoRound](https://huggingface.co/OPEA/DeepSeek-R1-0528-int2-mixed-AutoRound)
- NVIDIA DeepSeek-V3.2-NVFP4 (modelopt v0.41, linear ops in transformer blocks, ~1.66x smaller than FP8): MMLU-Pro 0.802→0.799, GPQA-D 0.849→0.835, LCB-v6 0.756→0.756, SciCode 0.391→0.401, AIME25 0.934→0.923; tested on B200 (8x for deployment) — [nvidia/DeepSeek-V3.2-NVFP4](https://huggingface.co/nvidia/DeepSeek-V3.2-NVFP4)
- Cerebras DeepSeek-V3.2-REAP-345B-A37B: 50% uniform expert prune (256→128), FP8 tensors, no fine-tune; card reports REAP-only scores (LCB 64.64, HumanEval 93.29, HE+ 89.02, MBPP 91.01, MBPP+ 75.13) with no baseline column; also a 508B (25%) variant — [cerebras/DeepSeek-V3.2-REAP-345B-A37B](https://huggingface.co/cerebras/DeepSeek-V3.2-REAP-345B-A37B); [collection](https://huggingface.co/collections/cerebras/cerebras-reap)
- 0xSero DeepSeek-V3.2-345B-W3A16: REAP (calibration 700 evol-codealpaca / 330 xlam / 330 SWE-smith) + 3-bit weights (auto-round tag); 138 GB on disk; UNEVALUATED; vLLM/SGLang instructions, no hardware — [0xSero/DeepSeek-V3.2-345B-W3A16](https://huggingface.co/0xSero/DeepSeek-V3.2-345B-W3A16)

**DeepSeek-V4 family**
- 6block DeepSeek-V4-Pro-0813 GGUF (1.57T): Q3_K_M 711.3 GiB 3.88 bpw PPL 1.6217; IQ3_XXS 577.0 GiB; Q2_K 547.0 GiB PPL 1.7621; IQ1_M 346.6 GiB 1.89 bpw PPL 3.6966; IQ1_S 314.1 GiB 1.72 bpw PPL 4.1095 (12-chunk wikitext). Rejected: IQ2_XS 445 GiB PPL 4.47, IQ2_XXS 401 GiB PPL 21.68 ("IQ2 expert path breaks down on this sparse-routing MoE"). Cliff is Q2_K→IQ1_M: ~200 GiB saved for +1.93 PPL vs +0.14 for the whole Q3_K_M→Q2_K range. No tok/s — [6block/DeepSeek-V4-Pro-0813-GGUF](https://huggingface.co/6block/DeepSeek-V4-Pro-0813-GGUF)
- Unsloth DeepSeek-V4-Flash (284B) GGUF with KLD table (4x B200, wikitext-2 ctx 512): official 156.4 GB PPL 4.5319; UD-Q8_K_XL 161.9 GB KLD ~0; UD-Q4_K_XL 155.1 GB KLD 0.0102; bartowski MXFP4 156.0 GB KLD 0.0105; antirez Q4KExperts 164.6 GB KLD 0.0290; antirez IQ2XXS 86.7 GB PPL 6.1518 KLD 0.4207; UD-IQ3_XXS 103 GB recommended for ~110 GB RAM — [Unsloth V4 docs](https://unsloth.ai/docs/models/deepseek-v4)
- antirez deepseek-v4-gguf (ds4 engine): q2 = 80.8 GiB (gate/up IQ2_XXS, down Q2_K, Q8_0 attention/shared/output, F16 router/embed/indexer), recommended for 128 GB Macs; q4 = 153.3 GiB; no tok/s or evals on the card — [antirez/deepseek-v4-gguf](https://huggingface.co/antirez/deepseek-v4-gguf)
- 0xSero DeepSeek-V4-Flash-0731-REAP K160: 256→160 experts (37.5%), 193B, MXFP4 experts, tensor payload 107.8 GB (100.4 GiB), smoke-tested on one DGX Spark/GB10 (Anemll vLLM preview); rankings transferred from an earlier V4-Flash observation run; UNEVALUATED ("structural and smoke validation do not establish benchmark parity") — [0xSero/DeepSeek-V4-Flash-0731-REAP](https://huggingface.co/0xSero/DeepSeek-V4-Flash-0731-REAP)
- Measured on one DGX Spark (121 GiB), llama.cpp, V4-Flash-0731: UD-IQ2_M 84.7 GiB → 448/444/401 tok/s prefill and 17.35/17.05/16.05 tok/s decode at 2K/8K/32K; UD-IQ3_XXS 97.1 GiB → 16.56/16.21/15.49 tok/s decode, ran to 131K ctx (103 GiB used), 6/6 tool calls, 5/5 code fix; UD-IQ1_S 76.9 GiB and UD-Q2_K_XL 90.2 GiB fit but untested; UD-IQ3_S 108.1 GiB "almost no room for KV"; ≥119 GiB impossible; ~6.5 min load; KV ≈ 29 KB/token — [Classmethod V4-Flash on Spark](https://dev.classmethod.jp/en/articles/dgx-spark-deepseek-v4-flash-0731-llama-cpp/)

**Kimi-K2 family (1T)**
- Unsloth UD-TQ1_0 Kimi-K2 1.8-bit = 245 GB vs 1.09 TB full (4.4x); UD-Q2_K_XL 381 GB; no KLD/PPL — [Unsloth K2 guide](https://docs.unsloth.ai/basics/kimi-k2-how-to-run-locally)
- Unsloth Kimi-K2.5 UD-TQ1_0 ~240 GB, UD-Q2_K_XL 375 GB; ~10 tok/s on 1x24 GB GPU + ~256 GB RAM; >40 tok/s on a B200; no KLD table — [Unsloth K2.5 guide](https://unsloth.ai/docs/models/tutorials/kimi-k2.5)
- Unsloth Kimi-K2.7-Code UD-Q2_K_XL 339 GB: PPL ~2.4131 vs ~1.8419 at Q8_K_XL (+31%); ~55 tok/s on 4x RTX PRO 6000 (DataCamp) — [Unsloth K2.7-Code docs](https://unsloth.ai/docs/models/kimi-k2.7-code); [DataCamp](https://www.datacamp.com/tutorial/how-to-run-kimi-k2-7-code-locally)
- ubergarm Kimi-K2-Thinking (ik_llama.cpp): Q4_X 543.6 GiB PPL 2.0818 (reference); smol-IQ3_KS 388.3 GiB +7.4% (Aider Polyglot 77.3%); IQ2_KL 348.9 GiB +14.0%; smol-IQ2_KS 270.1 GiB 2.261 bpw +41.0%; smol-IQ1_KT 218.9 GiB 1.832 bpw PPL 3.5931 (+72.6%, "only for the desperate", KT trellis slower at TG) — [ubergarm/Kimi-K2-Thinking-GGUF](https://huggingface.co/ubergarm/Kimi-K2-Thinking-GGUF)
- gghfexp Kimi-K2.7-Code ik quants (1T): IQ2_KT 264.5 GiB PPL 2.8960 (+44.1%); IQ2_KS 270.9 GiB PPL 2.9740 (+48.0%); IQ2_KL 329.7 GiB +21.5%; IQ3_KT 381.8 GiB untested — [gghfexp/Kimi-K2.7-Code-GGUF](https://huggingface.co/gghfexp/Kimi-K2.7-Code-GGUF)
- kwanhee Kimi-K2.5-REAP50-INT4-W4A16: REAP via llm-compressor REAPPruningModifier, 384→192 experts, top-8 unchanged; INT4 g32 sym expert weights inherited from Moonshot's native release (not requantized); attention/shared/dense/lm_head/vision BF16; 288.4 GB in 58 shards; text-only vs dense K2.5: OpenLLM-v1 avg 80.77 vs 82.43 (97.99%), AIME25 0.910 vs 0.957 (95.1%), GPQA-D 0.788 vs 0.895 (88.0%), MATH500 0.978 vs 0.964; same prune at NVFP4 W4A4 = 80.00 (97.06%); run on 4x B200, ctx 73,728; needs author's vLLM fork (34,560 weight_shape tensors break stock vLLM); nested quantization_config silently loads experts packed and gives wrong results — [kwanhee/Kimi-K2.5-REAP50-INT4-W4A16](https://huggingface.co/kwanhee/Kimi-K2.5-REAP50-INT4-W4A16)
- No Cerebras Kimi-K2 checkpoint exists; the collection's only Kimi entry is Kimi-Linear-REAP-35B — [Cerebras REAP collection](https://huggingface.co/collections/cerebras/cerebras-reap)

**Qwen3-Coder-480B / Qwen3.5-397B**
- Cerebras Qwen3-Coder-REAP-246B-A35B-FP8 (50%, 160→80 experts) vs full 480B vs REAP-363B (25%): HumanEval 95.1/95.7/93.9; HE+ 89.0/89.0/87.2; MBPP 92.3/91.7/91.0; MBPP+ 79.1/77.2/77.2; LCB 43.1/41.6/41.5; SWE-Bench-Verified (mini-swe-agent) 54.0/54.0/52.2; BFCL-v3 overall 69.0/69.8/67.4 (multi-turn 38.0/39.2/37.1); τ²-bench airline 46.0/48.7/44.7, retail 64.3/66.1/63.2, telecom 50.0/52.9/47.1; TerminalBench 30.5/30.5/30.0; calibration evol-codealpaca + xlam-function-calling-60k + SWE-smith-trajectories — [cerebras/Qwen3-Coder-REAP-246B-A35B-FP8](https://huggingface.co/cerebras/Qwen3-Coder-REAP-246B-A35B-FP8)
- Unsloth GGUFs of REAP-363B (25%): UD-TQ1_0 101 GB, UD-IQ1_S 104 GB, UD-IQ1_M 116 GB, Q2_K 132 GB, UD-Q2_K_XL 139 GB, UD-IQ3_XXS 155 GB, UD-Q4_K_XL 209 GB, BF16 726 GB; evals "TBD" (UNEVALUATED for the GGUFs) — [unsloth/Qwen3-Coder-REAP-363B-A35B-GGUF](https://huggingface.co/unsloth/Qwen3-Coder-REAP-363B-A35B-GGUF)
- ubergarm Qwen3.5-397B-A17B (wiki.test, 580 chunks): BF16 738.5 GiB PPL 3.4852; Q8_0 392.5 GiB 3.4862; IQ4_KSS 194.1 GiB 3.5102; Q3_K 180.0 GiB 3.5409 (mainline); IQ2_KL 138.1 GiB 2.994 bpw 3.6536 (+4.8%); smol-IQ2_XS 113.4 GiB 2.46 bpw 3.8717 (+11.1%, mainline-compatible); smol-IQ2_KS 108.1 GiB 2.344 bpw 3.9153 (+12.3%); smol-IQ1_KT 88.8 GiB 1.925 bpw 4.2523 (+22.0%). Ratio BF16→IQ1_KT = 8.3x. Example config for "96GB VRAM" hybrid; no tok/s — [ubergarm/Qwen3.5-397B-A17B-GGUF](https://huggingface.co/ubergarm/Qwen3.5-397B-A17B-GGUF)
- Qwen3.5-397B NVFP4 on 4x RTX PRO 6000 (TP=4, Marlin W4A16 because native NVFP4 MoE kernels fail on SM120): ~50.5 tok/s — [vLLM forum](https://discuss.vllm.ai/t/sm120-rtx-pro-6000-nvfp4-moe-performance-report-qwen3-5-397b/2536/2)

**GLM-4.6 / 4.7 (355B)**
- Cerebras GLM-4.6-REAP FP8 ladder vs GLM-4.6-FP8 full: HumanEval 96.3 / 96.3 (25%) / 95.7 (30%) / 95.1 (40%); HE+ 93.3/91.5/90.9/90.2; MBPP 87.6/89.9/89.9/89.4; MBPP+ 73.5/74.9/73.5/73.8; GPQA-Diamond (thinking) 78.8/76.8/75.8/69.7; AIME25 90.0/93.3/90.0/90.0; MATH-500 95.5/97.0/94.8/93.3; BFCL-v3 78.4/77.3/76.8/74.2 — [cerebras/GLM-4.6-REAP-218B-A32B-FP8](https://huggingface.co/cerebras/GLM-4.6-REAP-218B-A32B-FP8)
- Unsloth GGUFs of GLM-4.6-REAP-268B (25%, 160→120): UD-TQ1_0 64.8 GB, UD-IQ1_S 74.2 GB, UD-IQ1_M 82.4 GB, UD-IQ2_XXS 89.1 GB, UD-IQ2_M 93.7 GB, UD-Q2_K_XL 103 GB, UD-IQ3_XXS 110 GB, UD-Q3_K_XL 120 GB, UD-Q4_K_XL 154 GB, BF16 538 GB; evals TBD (UNEVALUATED) — [unsloth/GLM-4.6-REAP-268B-A32B-GGUF](https://huggingface.co/unsloth/GLM-4.6-REAP-268B-A32B-GGUF)

**GLM-5 / 5.1 / 5.2 / 5.3 (744–753B)**
- 0xSero GLM-5-REAP-50pct-UD-IQ2_M-GGUF: 744B → 50% REAP (256→128 experts) → imatrix IQ2_M at 2.72 bpw; 130 GB on disk (card also says ~121 GB); needs ~125 GB VRAM with 8K KV; pilot eval (10 samples/category) vs sibling Q3_K_M (170 GB): 40/50 (80%) vs 44/50 (88%); GSM8K 6/10, BBH 7/10, HumanEval 8/10, SWE-bench 10/10, terminal-bench 9/10; no vs-full-GLM-5 numbers; UD-IQ2_XXS and Q3_K_M siblings exist; only the IQ2_M link worked for a forum poster — [0xSero/GLM-5-REAP-50pct-UD-IQ2_M-GGUF](https://huggingface.co/0xSero/GLM-5-REAP-50pct-UD-IQ2_M-GGUF); [NVIDIA forum thread](https://forums.developer.nvidia.com/t/new-quantized-models-drop-glm-5-reap-50-how-many-dgx-sparks-do-you-need/364965)
- 0xSero GLM-5-381B-W3A16: 50% REAP (refusal_contrast_reap variant, router renorm on) + AutoRound W3A16 g128 (pile-10k, 128 samples, seq 1024, 50 iters/block, 4549 s tuning); lm_head, first-3-layer MLPs and indexer weights_proj unquantized; 154 GB; UNEVALUATED — [Friendli mirror of 0xSero/GLM-5-381B-W3A16](https://friendli.ai/models/0xSero/GLM-5-381B-W3A16)
- sombra-x GLM-5.1-REAP-Solidity-MXFP8: 128-of-256 experts kept, domain-specific; no details retrieved — [sombra-x/GLM-5.1-REAP-Solidity-MXFP8](https://huggingface.co/sombra-x/GLM-5.1-REAP-Solidity-MXFP8)
- 0xSero GLM-5.2-504B (REAP keep-168 = 34% pruned, applied to all MoE layers + MTP; Router-KD: network frozen, only 75 router gate matrices ≈0.016% of params trained, KL to teacher, lr 5e-5; a 6x larger KD set did not close the gap; NVFP4 modelopt on routed experts, BF16 router/attention/shared; compute sponsored by Lambda, 8x B200): Aider Polyglot aggregate 90.1 (Python 97.1→94.1, JS 93.9→93.9, Java 85.1→85.1, C++ 92.3→96.2, Go 89.7→87.2, Rust 86.7→83.3); Terminal-Bench 2.1 full-89 = 70.5%; loop/attractor rate 0.072 vs teacher 0.036 on n=2000 held-out coding-agent prompts; natural-EOS 0.928 vs 0.965; distinct-4 0.880 vs 0.921; agentic sources (codex/opencode) loop ~13% vs ~6%; min_p 0.05 + rep-penalty 1.05 → 4.9% loops, 1.10 → 2.3%; disk size not stated — [0xSero/GLM-5.2-504B](https://huggingface.co/0xSero/GLM-5.2-504B)
- 0xSero GLM-5.2-NVFP4-REAP-469B build: ~313 GB NVFP4, ~78.6 GB/GPU on 4x RTX PRO 6000 (PCIe, NCCL_P2P_DISABLE=1); forum user: "only 35 ish tokens/sec" before MTP — [0xSero/glm-5.2-sm120](https://github.com/0xSero/glm-5.2-sm120); [Level1Techs GLM 5.2 REAP](https://forum.level1techs.com/t/glm-5-2-reap/251814)
- madeby561 GLM-5.2-Int8Mix-NVFP4-REAP-594B (~22% pruned): GPQA-Diamond 86.87 vs 89.39 (full NVFP4) / 89.52 (FP8); SciCode 47.77 vs 49.04/49.85; REAP-504B (34%) SciCode 44.67; IFBench and τ²-telecom pending — [madeby561/GLM-5.2-Int8Mix-NVFP4-REAP-594B](https://huggingface.co/madeby561/GLM-5.2-Int8Mix-NVFP4-REAP-594B)
- pipenetwork GLM-5.2-REAP50-Q3_K_M-GGUF: 50% REAP (256→128, MTP dropped), ~394B/381B, Q3_K_M ~169 GB (182 GB file metadata); "~+37.5% perplexity" vs full GLM-5.2; sibling MLX REAP-25 "+2.3%"; intended for 2x 96 GB; Apple Metal smoke test ~17 tok/s; stock llama.cpp cannot load any GLM-5.2 GGUF without the bundled DSA-indexer patch — [pipenetwork/GLM-5.2-REAP50-Q3_K_M-GGUF](https://huggingface.co/pipenetwork/GLM-5.2-REAP50-Q3_K_M-GGUF)
- AikidoSec Altar-1 (GLM-5.3 753B → REAP keep-168, 34%, domain-max-share expert scoring, no retraining, prune by 0xSero on 8x RTX PRO 6000; INT4 W4A16 AWQ/compressed-tensors on routed experts only, from cyankiwi/GLM-5.3-AWQ-INT4): 328 GB; KL vs BF16 = 0.506 nats on a sealed 25-prompt panel; EXL3 build of the same cut = 0.511; serve on 4x H200 TP=4, 131k ctx; no named benchmarks — [AikidoSec/altar-1](https://huggingface.co/AikidoSec/altar-1); 0xSero's own W4A16 copy: [0xSero/GLM-5.3-504B-W4A16](https://huggingface.co/0xSero/GLM-5.3-504B-W4A16)
- OpenMOSE GLM-5.3-Flash-REAP-250B-A18B (321B → 288→228 experts, 20.8%, domain-aware REAP with per-domain top-64 always kept, FP8 block-wise copied without requant): 265 GB; teacher-forced on ~130k held-out tokens: KL 0.453, top-1 agreement 86.0% (92.4% on confident tokens), held-out PPL 4.181 vs 4.068 (+2.8%); uniform-224 single-calibration reference KL 0.476, +5.8% PPL; a chat-only selection "lost most of its coding ability despite only modest per-layer error" — [OpenMOSE/GLM-5.3-Flash-REAP-250B-A18B](https://huggingface.co/OpenMOSE/GLM-5.3-Flash-REAP-250B-A18B)

**MiniMax-M2.x (230B) — included only because it is the one REAP family with single-128 GB-device measurements**
- Cerebras MiniMax-M2-REAP-162B (30%, 256→180) vs full: HumanEval 93.9→93.3, HE+ 89.0→86.6, MBPP 87.6→86.5, MBPP+ 73.0→73.0, AIME25 76.7→73.3, MATH-500 91.6→89.4, τ²-telecom 59.1→59.1, BFCLv3 62.6→59.9; cyankiwi AWQ-4bit weights ≈86.6 GB — [cyankiwi/MiniMax-M2-REAP-162B-A10B-AWQ-4bit](https://huggingface.co/cyankiwi/MiniMax-M2-REAP-162B-A10B-AWQ-4bit)
- Single DGX Spark, vLLM, MiniMax-M2.5-REAP NVFP4 (40%): tg32 16.71 tok/s, 15.85 @8K, 12.95 @32K, 10.49 @64K; pp2048 3342 tok/s; saricles/MiniMax-M2.5-REAP-172B-A10B-NVFP4-GB10 runs to 128k ctx with KV fp8; i1-Q4_K_S GGUF of REAP-139B: 20.4 → 26.15 tok/s after parallel=1/ctx 80k tuning; two users reported "gibberish" output with the NVFP4 REAP, one calling 40% "too strongly REAPed … 20–25% would be better" — [NVIDIA forum MiniMax REAP on Spark](https://forums.developer.nvidia.com/t/minimax-2-5-reap-nvfp4-on-single-dgx-spark/361248)

**Ling-1T / Ring-1T**
- ubergarm2 Ling-1T (ik_llama.cpp): Q8_0 989.7 GiB PPL 1.9859; IQ5_K 689.9 GiB; smol-IQ4_KSS 471.9 GiB 2.0176; smol-IQ3_KS 378.9 GiB 2.0770; IQ2_K 330.9 GiB 2.2169; smol-IQ2_KS 265.0 GiB 2.277 bpw 2.4429; smol-IQ2_XXS 249.9 GiB 2.5870 (mainline test build); smol-IQ1_KT 215.4 GiB 1.851 bpw 2.8581 (+43.9%); no tok/s — [ubergarm2/Ling-1T-GGUF](https://huggingface.co/ubergarm2/Ling-1T-GGUF)
- DevQuasar Ling-1T IQ1_M (17 shards) and IQ2_XXS (20 shards) exist; total size and evals not retrieved — [DevQuasar/inclusionAI.Ling-1T-GGUF](https://huggingface.co/DevQuasar/inclusionAI.Ling-1T-GGUF)
- No REAP/HOPE prune of Ling-1T or Ring-1T found.

**Step-3 / Hunyuan-Large / MiMo-V2**
- Cerebras ships Step-3.5-Flash-REAP-149B and -121B (source <400B); no Step-3 (321B) or Hunyuan-Large (389B) compressed artifact with evals was found; no MiMo-V2 REAP found — [Cerebras REAP collection](https://huggingface.co/collections/cerebras/cerebras-reap)
- OpenMOSE Hy3-REAP-200B-21B GGUF exists (Hunyuan 3 family; source size unknown, not fetched) — [OpenMOSE/Hy3-REAP-200B-21B-GGUF](https://huggingface.co/OpenMOSE/Hy3-REAP-200B-21B-GGUF)

### Inferences
- Computable ratios (source bytes / artifact bytes): R1-0528 Q8_0→IQ1_S_R4 = 5.1x (vs BF16 ≈ 1.34 TB it is ~10x); Qwen3.5-397B BF16→smol-IQ1_KT = 8.3x; Kimi-K2-Thinking Q4_X→smol-IQ1_KT = 2.5x (native INT4 source); Ling-1T Q8_0→smol-IQ1_KT = 4.6x; GLM-5 BF16 (≈1.49 TB, inferred from 744B x 2 B) → REAP50+IQ2_M 130 GB ≈ 11.4x; GLM-5.3 BF16 (≈1.51 TB) → Altar-1 328 GB = 4.6x; Qwen3-Coder-480B BF16 (≈960 GB) → REAP-363B UD-IQ1_S 104 GB ≈ 9.2x; V4-Pro F16-GGUF 812.7 GiB → IQ1_S 314.1 GiB = 2.6x (source already FP4).
- The field converged on one hardware target: 4x RTX PRO 6000 (384 GB) or 4x H200 for the pruned+4-bit 500B-class artifacts (0xSero, Altar-1, kwanhee), and a single DGX Spark/Mac 128 GB only for ≤300B sources (V4-Flash, MiniMax REAP). No one ships a ≤128 GB artifact of a ≥500B source with a benchmark table.

### Gaps
- Disk sizes are not stated on the cards for: cerebras DeepSeek-V3.2-REAP-345B, cerebras GLM-4.6-REAP-218B, 0xSero GLM-5.2-504B (NVFP4), madeby561 594B, OPEA R1-0528 int2.
- 0xSero/GLM-5-REAP-50pct-UD-IQ2_XXS-GGUF returned HTTP 401 (gated/private); its size is unknown (would be the smallest 744B-source artifact).
- ubergarm Qwen3-235B-A22B, Kimi-K2.6, DeepSeek-V3.2 ik-quant tables were not retrieved (search did not surface the tables).
- No Cerebras/0xSero REAP of Kimi-K2 (original), Ling/Ring-1T, Hunyuan-Large, Step-3, or MiMo-V2 was found on HF.

## Which artifacts combine expert pruning WITH sub-4-bit quantization, and what did the combination cost vs. either alone?

### Takeaway
Five shipped artifacts stack REAP with ≤3-bit weights (GLM-5 REAP50+IQ2_M 130 GB, GLM-5 REAP50+W3A16 154 GB, DeepSeek-V3.2 REAP+W3A16 138 GB, GLM-5.2 REAP50+Q3_K_M 169 GB, and the REAP-363B/268B Unsloth 1–2-bit GGUF ladders), but only one reports a number against the unpruned model (+37.5% PPL for GLM-5.2 REAP50+Q3_K_M) and none decomposes the cost; the only decomposable evidence is at 4-bit, where REAP-50%+INT4 Kimi-K2.5 keeps 98% of OpenLLM but only 88% of GPQA.

### Cited Findings
- pipenetwork GLM-5.2 REAP-50% + Q3_K_M (~169 GB): "~+37.5% perplexity" vs full GLM-5.2, attributed to pruning plus Q3_K; its REAP-25 MLX sibling is +2.3% — [pipenetwork/GLM-5.2-REAP50-Q3_K_M-GGUF](https://huggingface.co/pipenetwork/GLM-5.2-REAP50-Q3_K_M-GGUF)
- 0xSero GLM-5 REAP-50% + UD-IQ2_M 2.72 bpw (130 GB): 80% vs 88% for the same prune at Q3_K_M on a 10-sample/category pilot; math and reasoning (6/10, 7/10) were hit hardest, coding/agentic 8–10/10; no vs-unpruned comparison — [0xSero/GLM-5-REAP-50pct-UD-IQ2_M-GGUF](https://huggingface.co/0xSero/GLM-5-REAP-50pct-UD-IQ2_M-GGUF)
- 0xSero GLM-5-381B-W3A16 (REAP-50% + AutoRound 3-bit, 154 GB) and DeepSeek-V3.2-345B-W3A16 (REAP + 3-bit, 138 GB): both UNEVALUATED — [Friendli GLM-5-381B-W3A16](https://friendli.ai/models/0xSero/GLM-5-381B-W3A16); [0xSero/DeepSeek-V3.2-345B-W3A16](https://huggingface.co/0xSero/DeepSeek-V3.2-345B-W3A16)
- Unsloth 1-bit/2-bit GGUFs of Cerebras REAP-363B (Qwen3-Coder) and REAP-268B (GLM-4.6): sizes 101–139 GB and 64.8–103 GB respectively, evals "TBD" — [unsloth/Qwen3-Coder-REAP-363B-A35B-GGUF](https://huggingface.co/unsloth/Qwen3-Coder-REAP-363B-A35B-GGUF); [unsloth/GLM-4.6-REAP-268B-A32B-GGUF](https://huggingface.co/unsloth/GLM-4.6-REAP-268B-A32B-GGUF)
- Pruning + 4-bit (decomposable): Kimi-K2.5 REAP-50% + native INT4: OpenLLM 97.99%, AIME25 95.1%, GPQA-D 88.0% of dense; same prune at NVFP4 W4A4 = 97.06% — [kwanhee/Kimi-K2.5-REAP50-INT4-W4A16](https://huggingface.co/kwanhee/Kimi-K2.5-REAP50-INT4-W4A16)
- Quantization alone (4-bit) on the same families is ~lossless: DeepSeek-V3.2 NVFP4 GPQA-D 0.849→0.835, LCB flat; GLM-5.2 NVFP4 GPQA 89.52→89.39, SciCode 49.85→49.04 — [nvidia/DeepSeek-V3.2-NVFP4](https://huggingface.co/nvidia/DeepSeek-V3.2-NVFP4); [madeby561 card](https://huggingface.co/madeby561/GLM-5.2-Int8Mix-NVFP4-REAP-594B)
- Pruning alone (34%) on GLM-5.2 NVFP4: SciCode 49.04→44.67 (−8.9% rel.); 22%: GPQA 89.39→86.87, SciCode →47.77 — [madeby561 card](https://huggingface.co/madeby561/GLM-5.2-Int8Mix-NVFP4-REAP-594B)
- Pruning alone (34%, no quant) on GLM-5.3-Flash: KL 0.453, top-1 86%, PPL +2.8% — [OpenMOSE card](https://huggingface.co/OpenMOSE/GLM-5.3-Flash-REAP-250B-A18B); pruning (34%) + INT4 on GLM-5.3: KL 0.506 (Altar-1) — [AikidoSec/altar-1](https://huggingface.co/AikidoSec/altar-1)
- 2-bit alone on 671B (no pruning): V3.1 UD 2-bit Aider thinking 66.7 vs 76.1 (−12% rel.), 1-bit 57.8 (−24% rel.) — [Unsloth Aider page](https://unsloth.ai/docs/basics/dynamic-3.0-ggufs/unsloth-dynamic-ggufs-on-aider-polyglot); R1-0528 IQ2_K_R4 PPL +9.1%, IQ1_S_R4 +51.9% — [ubergarm R1-0528](https://huggingface.co/ubergarm/DeepSeek-R1-0528-GGUF)
- The REAP paper's own Kimi-K2 experiments were run on Kimi-K2-Instruct-W4A16 (already 4-bit), i.e. pruning stacked on quantization at 50%: Eval+ 0.828→0.819, LCB 0.434→0.429, SWE-V 0.554→0.576, but BFCL overall 0.666→0.564 and BFCL multi-turn 0.355→0.164 — [REAP paper Table 3](https://arxiv.org/html/2510.13999v2); Cerebras states REAP "can be applied on top of … quantization" — [Cerebras blog](https://www.cerebras.ai/blog/reap)

### Inferences
- Stacking 50% REAP with ~2.7-bit costs at least the sum of the parts on reasoning/math (GLM-5 pilot: −8 pts vs Q3_K_M on top of whatever REAP50 cost), while coding/agentic (the calibration domain) survives both; this matches the GLM-4.6 ladder (code flat, GPQA −9 pts at 40%).
- At equal bytes, the evidence favours quant-only over prune+quant for general tasks: R1-0528 IQ2_K_R4 (220 GiB, +9% PPL) vs GLM-5.2 REAP50+Q3_K_M (169 GB, +37.5% PPL) — different models, so this is directional, not a controlled comparison.

### Gaps
- No artifact ships a 2x2 decomposition (pruned-only, quant-only, both) on the same model with the same benchmark at ≤3 bits.
- GLM-5 REAP50 IQ2_M pilot has no unpruned-GLM-5 column, so its absolute cost is unknown.

## Highest expert-pruning ratio actually shipped on a ≥400B model, and its measured evals

### Takeaway
50% is the ceiling anyone has shipped on a ≥400B model (Cerebras Qwen3-Coder-246B, Cerebras DeepSeek-V3.2-345B, kwanhee Kimi-K2.5-REAP50, 0xSero GLM-5-REAP50, pipenetwork GLM-5.2-REAP50); the only 50% artifacts with a full baseline table are Qwen3-Coder-246B (−1 to −3 points on code/tool benchmarks) and Kimi-K2.5-REAP50 (−2% OpenLLM, −12% GPQA).

### Cited Findings
- Qwen3-Coder-480B → 246B (50%): HumanEval 95.1→93.9, LCB 43.1→41.5, SWE-Bench-V 54.0→52.2, BFCL overall 69.0→67.4, τ²-telecom 50.0→47.1, TerminalBench 30.5→30.0 — [cerebras/Qwen3-Coder-REAP-246B-A35B-FP8](https://huggingface.co/cerebras/Qwen3-Coder-REAP-246B-A35B-FP8); Cerebras summarises as 97.6% of non-agentic coding and 96.7% of SWE-Bench retained — [Cerebras blog](https://www.cerebras.ai/blog/reap)
- Kimi-K2.5 → REAP50 (384→192): OpenLLM-v1 avg 80.77 vs 82.43, AIME25 0.910 vs 0.957, GPQA-D 0.788 vs 0.895, MATH500 0.978 vs 0.964 — [kwanhee card](https://huggingface.co/kwanhee/Kimi-K2.5-REAP50-INT4-W4A16)
- Kimi-K2-Instruct-W4A16 50% REAP (paper, no shipped checkpoint): code ~flat, BFCL multi-turn 0.355→0.164, MC avg 0.780→0.643; frequency pruning at 50% collapses entirely (Eval+ 0.112, SWE 0.000) — [REAP paper](https://arxiv.org/html/2510.13999v2)
- DeepSeek-V3.2 → 345B (50%): LCB 64.64, HumanEval 93.29 with no baseline column — [cerebras/DeepSeek-V3.2-REAP-345B-A37B](https://huggingface.co/cerebras/DeepSeek-V3.2-REAP-345B-A37B)
- GLM-5.2 → REAP50 + Q3_K_M: +37.5% PPL — [pipenetwork](https://huggingface.co/pipenetwork/GLM-5.2-REAP50-Q3_K_M-GGUF); GLM-5 → REAP50 + IQ2_M/W3A16: pilot/UNEVALUATED — [0xSero IQ2_M](https://huggingface.co/0xSero/GLM-5-REAP-50pct-UD-IQ2_M-GGUF)
- Intermediate ratios with full tables: GLM-4.6 40% (GPQA-D 78.8→69.7, BFCL 78.4→74.2, code within 1–3 pts) — [cerebras GLM-4.6-REAP-218B-FP8](https://huggingface.co/cerebras/GLM-4.6-REAP-218B-A32B-FP8); GLM-5.2 34% with Router-KD (Aider 90.1 agg., loop rate 2x teacher) — [0xSero/GLM-5.2-504B](https://huggingface.co/0xSero/GLM-5.2-504B); GLM-5.2 22% (GPQA −2.5 pts) — [madeby561](https://huggingface.co/madeby561/GLM-5.2-Int8Mix-NVFP4-REAP-594B)
- Method updates: Cerebras added router-logit renormalisation on 2026-03-11 (mean accuracy drop 1.9% vs 2.6% on 5 small models; all HF checkpoints calibrated with it) and a layer-wise observer (2026-03-30) for single-GPU pruning of large models; calibration mix for the agentic checkpoints = 24,576 samples at 16,384 max length — [CerebrasResearch/reap](https://github.com/CerebrasResearch/reap)
- Successors with no shipped ≥400B checkpoints: HOPE (second-order, REAP is its diagonal; avg rank 1.58 vs REAP 2.42 at 50%, up to +6.1% agentic coding, largest model tested 122B) — [arXiv 2609.18916](https://arxiv.org/abs/2609.18916); REAM (merging, Samsung SAIL Montreal, code only) — [arXiv 2604.04356](https://arxiv.org/abs/2604.04356); REAPPruningModifier now in vLLM's llm-compressor (worked example on 30B/16B models) — [llm-compressor docs](https://docs.vllm.ai/projects/llm-compressor/en/latest/examples/reap_expert_pruning/)

### Inferences
- The reproducible pattern across every ladder (Qwen3-Coder, GLM-4.6, GLM-5.2, Kimi-K2.5) is that the calibration domain (code/tool) loses 1–3 points at 50% while out-of-domain knowledge (GPQA) loses 9–12 points, i.e. the loss is domain-selective, not uniform.

### Gaps
- No one has shipped >50% expert pruning on a ≥400B model.
- The DeepSeek-V3.2-REAP cards omit the baseline column, so the 50% cost on DeepSeek is unmeasured on HF.

## Lowest bits/weight actually shipped on a ≥400B MoE with evals, and PPL/KLD vs BF16

### Takeaway
The floor is ~1.66–1.85 bpw via ik_llama.cpp trellis/`_R4` quants (ubergarm R1-0528 IQ1_S_R4 1.664 bpw PPL +52%; Kimi-K2-Thinking smol-IQ1_KT 1.832 bpw +73%; Ling-1T smol-IQ1_KT 1.851 bpw +44%; Qwen3.5-397B smol-IQ1_KT 1.925 bpw +22%) and 1.72 bpw mainline IQ1_S on V4-Pro; no 1-bit artifact of a ≥400B model ships a KLD number, and the only task-level 1-bit eval is Unsloth's V3.1 (Aider 57.8 vs 76.1).

### Cited Findings
- 1.664 bpw: R1-0528 IQ1_S_R4 130.2 GiB, PPL 4.8805 vs 3.2130 Q8_0 (KLD chart only) — [ubergarm](https://huggingface.co/ubergarm/DeepSeek-R1-0528-GGUF)
- 1.72 bpw: V4-Pro IQ1_S 314.1 GiB PPL 4.11 (12-chunk; F16 220-chunk baseline 4.08 not comparable; Q3_K_M at 12 chunks 1.62) — [6block](https://huggingface.co/6block/DeepSeek-V4-Pro-0813-GGUF)
- 1.832 bpw: Kimi-K2-Thinking smol-IQ1_KT 218.9 GiB PPL 3.5931 vs 2.0818 (+72.6%) — [ubergarm](https://huggingface.co/ubergarm/Kimi-K2-Thinking-GGUF)
- 1.851 bpw: Ling-1T smol-IQ1_KT 215.4 GiB PPL 2.8581 vs 1.9859 (+43.9%) — [ubergarm2](https://huggingface.co/ubergarm2/Ling-1T-GGUF)
- 1.925 bpw: Qwen3.5-397B smol-IQ1_KT 88.8 GiB PPL 4.2523 vs 3.4852 BF16 (+22.0%) — [ubergarm](https://huggingface.co/ubergarm/Qwen3.5-397B-A17B-GGUF)
- 1.58-bit (Unsloth naming) R1 131 GB and 1.8-bit Kimi-K2 245 GB / K2.5 240 GB: no PPL/KLD published — [Unsloth R1](https://unsloth.ai/docs/basics/tutorials-how-to-fine-tune-and-run-llms/deepseek-r1-how-to-run-locally); [Unsloth K2](https://docs.unsloth.ai/basics/kimi-k2-how-to-run-locally); [Unsloth K2.5](https://unsloth.ai/docs/models/tutorials/kimi-k2.5)
- Task-level 1-bit: V3.1 1-bit 192 GB Aider thinking 57.8 / non-thinking 55.7 (IQ1_M 206 GB) vs 76.1 / 71.6 — [Unsloth Aider](https://unsloth.ai/docs/basics/dynamic-3.0-ggufs/unsloth-dynamic-ggufs-on-aider-polyglot)
- 2-bit with evals: K2.7-Code IQ2_KT 264.5 GiB +44.1%, IQ2_KS 270.9 GiB +48.0% — [gghfexp](https://huggingface.co/gghfexp/Kimi-K2.7-Code-GGUF); Unsloth K2.7-Code UD-Q2_K_XL 339 GB PPL 2.4131 vs 1.8419 (+31%) — [Unsloth](https://unsloth.ai/docs/models/kimi-k2.7-code); Kimi-K2-Thinking IQ2_KL 348.9 GiB +14.0% — [ubergarm](https://huggingface.co/ubergarm/Kimi-K2-Thinking-GGUF); R1-0528 IQ2_K_R4 220 GiB +9.1% — [ubergarm](https://huggingface.co/ubergarm/DeepSeek-R1-0528-GGUF)
- INT2 with task evals: OPEA R1 int2-mixed MMLU 0.8514→0.8302 (−2.1 pts) — [OPEA](https://huggingface.co/OPEA/DeepSeek-R1-int2-mixed-sym-inc)
- KLD on a 284B source: antirez IQ2XXS 86.7 GB KLD 0.4207 vs UD-Q4_K_XL 0.0102 — [Unsloth V4 docs](https://unsloth.ai/docs/models/deepseek-v4)
- EXL3: exllamav3 lists DeepSeek V3/V4, GLM 4.6/5.2/5.3-Flash, Qwen3-MoE, MiniMax-M2 and Kimi-Linear (not Kimi-K2) with CPU offload for large MoE; no 2-bit PPL table on the README; the only ≥400B EXL3 artifact found is Altar-1's EXL3 build (KL 0.511) — [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3); [AikidoSec/altar-1](https://huggingface.co/AikidoSec/altar-1)
- Unsloth's public KLD tables cover dense models (Gemma 3, Qwen3.8-27B), not the 1T MoEs — [Unsloth Dynamic 2.0](https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs)

### Inferences
- Relative PPL cost at ~1.85 bpw scales with how "dense-ish" the model is: Qwen3.5-397B (+22%) < Ling-1T (+44%) < R1-0528 (+52%) < Kimi-K2-Thinking (+73%, from a native-INT4 base, so its reference is already lossy).
- ik_llama.cpp `smol-IQ1_KT`/`IQ1_S_R4` are consistently ~1.5–2x better PPL-per-bit than mainline IQ1_S/IQ1_M at the same size (6block's IQ1_M/IQ1_S vs its own Q2_K shows the mainline cliff), but trellis KT types are slower at token generation on CPU.

### Gaps
- No KLD numbers for any 1-bit quant of a ≥400B model (only PPL and one chart).
- turboderp has not published EXL3 quants of DeepSeek/Kimi; the EXL3 2-bit PPL chart is an image not read here.

## Artifacts that target a single ~128 GB device (DGX Spark, Mac Studio 128–192 GB, Jetson Thor, RTX PRO 6000 96 GB): what fit, at what context, at what tok/s

### Takeaway
Measured single-128 GB-device runs exist only for ≤300B sources: DeepSeek-V4-Flash UD-IQ3_XXS (97.1 GiB, 16.5 tok/s, 131K ctx, tool calls pass) and MiniMax-M2.5-REAP NVFP4/Q4 (16–26 tok/s, to 128K) on a DGX Spark, and Qwen3.5-122B AWQ on a Jetson Thor (~4.2 tok/s); for ≥500B sources the candidates (R1-0528 IQ1_S_R4 130 GiB, GLM-5 REAP50 IQ2_M 130 GB) have only been run on CPU+GPU rigs, not on a Spark/Mac/Thor.

### Cited Findings
- DGX Spark (121 GiB usable), llama.cpp, V4-Flash-0731: UD-IQ2_M 84.7 GiB 17.35→16.05 tok/s (2K→32K), UD-IQ3_XXS 97.1 GiB 16.56→15.49 tok/s, 131K ctx at 103 GiB; UD-IQ3_S 108.1 GiB leaves almost no KV room; ≥119 GiB impossible — [Classmethod](https://dev.classmethod.jp/en/articles/dgx-spark-deepseek-v4-flash-0731-llama-cpp/)
- DGX Spark, vLLM, MiniMax-M2.5-REAP NVFP4: 16.7 tok/s short ctx → 10.5 tok/s at 64K; GB10-targeted NVFP4 of REAP-172B runs to 128K ctx with fp8 KV, max-num-seqs 4; i1-Q4_K_S REAP-139B GGUF 26.15 tok/s tuned — [NVIDIA forum](https://forums.developer.nvidia.com/t/minimax-2-5-reap-nvfp4-on-single-dgx-spark/361248)
- DGX Spark, 0xSero V4-Flash-0731 REAP K160 MXFP4 (100.4 GiB): smoke-tested on one GB10 with Anemll vLLM preview, no tok/s — [0xSero card](https://huggingface.co/0xSero/DeepSeek-V4-Flash-0731-REAP)
- GLM-5 REAP50 UD-IQ2_M (130 GB): forum answer that it "won't fit comfortably" on one Spark, two needed; no Spark test — [NVIDIA forum](https://forums.developer.nvidia.com/t/new-quantized-models-drop-glm-5-reap-50-how-many-dgx-sparks-do-you-need/364965)
- Mac 128 GB: antirez recommends his 80.8 GiB V4-Flash q2 for 128 GB Macs (untested on the card); Unsloth's R1 guide offloads ~59 layers of UD-IQ1_S on a 128 GB Mac; Mac 512 GB M3 Ultra runs DeepSeek-V3-0324 4-bit at >20 tok/s (Awni Hannun) — [antirez](https://huggingface.co/antirez/deepseek-v4-gguf); [Unsloth R1](https://unsloth.ai/docs/basics/tutorials-how-to-fine-tune-and-run-llms/deepseek-r1-how-to-run-locally); [Slashdot/Hannun](https://apple.slashdot.org/story/25/03/25/2054214/deepseek-v3-now-runs-at-20-tokens-per-second-on-mac-studio)
- Jetson Thor 128 GB: Qwen3.5-122B-A10B AWQ-4bit at ~4.2 tok/s with tool calling (Hackster write-up; page returned 403, figure from search summary, unverified) — [Hackster](https://www.hackster.io/jenfi59/brainiak-at-home-claude-is-left-2ea8e7)
- RTX PRO 6000 96 GB: no single-card run of a ≥400B artifact found; cyankiwi MiniMax-M2-REAP-162B AWQ weights ≈86.6 GB are the only 96 GB-targeted REAP artifact; 4-card runs: GLM-5.2-REAP-469B NVFP4 ~35 tok/s, Qwen3.5-397B NVFP4 ~50.5 tok/s, K2.7-Code UD-Q2_K_XL ~55 tok/s — [cyankiwi](https://huggingface.co/cyankiwi/MiniMax-M2-REAP-162B-A10B-AWQ-4bit); [Level1Techs](https://forum.level1techs.com/t/glm-5-2-reap/251814); [vLLM forum](https://discuss.vllm.ai/t/sm120-rtx-pro-6000-nvfp4-moe-performance-report-qwen3-5-397b/2536/2); [DataCamp](https://www.datacamp.com/tutorial/how-to-run-kimi-k2-7-code-locally)
- 128 GiB RAM + 24 GB GPU desktop: R1-0528 IQ1_S_R4 ran at ~116 GiB RAM + 22 GB VRAM, 32k+ ctx, no tok/s stated; IQ2_K_R4 ~15 tok/s on a bigger rig — [ubergarm](https://huggingface.co/ubergarm/DeepSeek-R1-0528-GGUF)
- Multi-Spark: Kimi-K3 UD-IQ1_S on 3x Spark via SparkInfer ≈6.59 tok/s (unsourced per the page) — [ContextStudios](https://www.contextstudios.ai/local-ai/vcruz305--kimi-k3-neuron-iq1s-3spark-sparkinfer); V4-Flash FP8 on 2x Spark TP=2 ~25 tok/s with 200K ctx — [NVIDIA forum](https://forums.developer.nvidia.com/t/deepseek-v4-flash-on-two-dgx-sparks-realistic-tok-sec/370924)

### Inferences
- The practical weight budget on a 128 GB unified-memory device is ~97–108 GiB (Spark: 121 GiB minus KV and runtime), so the ≥500B candidates at 130 GB/GiB (R1 IQ1_S_R4, GLM-5 REAP50 IQ2_M) do not fit without mmap paging; the next size down for a ≥500B source does not exist yet.
- Decode on a Spark is ~16–17 tok/s for a ~13B-active model at ~90–100 GiB of 2–3-bit weights, i.e. the ceiling for a hypothetical ≤108 GiB 1T-source artifact with ~32–40B active would be roughly 5–8 tok/s by bandwidth scaling (my estimate).

### Gaps
- No measured run of any ≥500B-source artifact on a single DGX Spark, Mac 128 GB, Jetson Thor, or RTX PRO 6000 was found.

## Recipes and numbers from Cerebras, Unsloth, ubergarm, turboderp, Lambda, NVIDIA, Intel, RedHat/vLLM

### Takeaway
Each producer has one reproducible recipe: Cerebras = one-shot REAP (gate x activation-norm, renormalised logits, code/tool calibration, FP8); Unsloth = per-layer dynamic imatrix GGUF with MoE experts at 1.5–2.5 bit and attention higher, now with KLD tables only for ≤300B models; ubergarm = ik_llama.cpp mixed "smol" recipes (q8/iq4 attention+shared, IQ1_KT/IQ2_KS routed experts) with wiki.test PPL ladders; NVIDIA = modelopt NVFP4 with 5-benchmark tables; Intel = AutoRound int2-mixed with lm-eval tables; RedHat = REAP inside llm-compressor; 0xSero (Lambda-sponsored) = REAP + Router-KD + NVFP4/AutoRound/UD-GGUF.

### Cited Findings
- Cerebras: saliency = gate weight x expert output norm; prune lowest per layer, no fine-tune; calibration 12,228 samples at 16,384 length for 110B+ models (paper) and 24,576 samples for the released agentic checkpoints; releases are FP8 and BF16 on HF, run on vanilla vLLM TP=8 — [REAP paper](https://arxiv.org/html/2510.13999v2); [GitHub](https://github.com/CerebrasResearch/reap); [GLM-4.5-Air card](https://gitcode.com/hf_mirrors/cerebras/GLM-4.5-Air-REAP-82B-A12B/blob/main/README.md)
- Unsloth: R1 UD recipe = MoE at 1.56-bit, down_proj mixed 2.06/1.56, --min-p 0.05; Dynamic 2.0/3.0 = per-layer, per-model type selection, >1.5M-token curated imatrix set, no QAT; K2.7-Code 2-bit uses jukofyork's scale tweak (max/-7) on MoE layers; V4-Flash KLD run on 4x B200 — [Unsloth R1](https://unsloth.ai/docs/basics/tutorials-how-to-fine-tune-and-run-llms/deepseek-r1-how-to-run-locally); [Dynamic 2.0](https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs); [K2.7-Code](https://unsloth.ai/docs/models/kimi-k2.7-code); [V4 docs](https://unsloth.ai/docs/models/deepseek-v4)
- ubergarm: ik_llama.cpp-only types (IQ1_KT, IQ2_KS, IQ2_KL, IQ4_KSS, _R4 repacks), -mla 3, -amb 512 for MLA models, q8_0 KV; PPL over wiki.test.raw at n_ctx=512; mainline-compatible variants released for a subset (Q3_K, smol-IQ2_XS, smol-IQ2_XXS) — [Qwen3.5-397B](https://huggingface.co/ubergarm/Qwen3.5-397B-A17B-GGUF); [Kimi-K2-Thinking](https://huggingface.co/ubergarm/Kimi-K2-Thinking-GGUF); [Ling-1T](https://huggingface.co/ubergarm2/Ling-1T-GGUF)
- turboderp/EXL3: QTIP-derived trellis format, DeepSeek V3/V4 + GLM 5.2/5.3-Flash + MiniMax-M2 + Qwen3-MoE supported, CPU offload for large MoE, MoE kernel ticket scheduler in v1.0.0; no 2-bit PPL numbers in text — [exllamav3](https://github.com/turboderp-org/exllamav3)
- Lambda: compute sponsor for 0xSero's GLM-5.2-504B (8x B200 for prune, KD, eval); 0xSero's full recipe = REAP keep-168, Router-KD (gates only, KL to teacher, lr 5e-5), NVFP4 (modelopt) on routed experts — [0xSero/GLM-5.2-504B](https://huggingface.co/0xSero/GLM-5.2-504B)
- NVIDIA: modelopt v0.41 NVFP4, cnn_dailymail + Nemotron-Post-Training-v2 calibration, ~1.66x smaller than FP8, ≤1.4-pt deltas on V3.2; official GLM-5.2 NVFP4 within 0.2–0.8 pts of FP8 — [nvidia/DeepSeek-V3.2-NVFP4](https://huggingface.co/nvidia/DeepSeek-V3.2-NVFP4); [madeby561 table](https://huggingface.co/madeby561/GLM-5.2-Int8Mix-NVFP4-REAP-594B)
- Intel AutoRound: int2 g64 sym experts, int4 g128 non-experts/shared/first 3 layers, 400 iters, pile-10k; needs 5x80 GB + 1.4–1.6 TB RAM to quantize; transformers-only inference — [OPEA R1 int2](https://huggingface.co/OPEA/DeepSeek-R1-int2-mixed-sym-inc)
- RedHat/vLLM llm-compressor: `REAPPruningModifier` (sparsity param, hooks collect saliency during sequential calibration, config updated per layer); kwanhee's Kimi-K2.5-REAP50 was built with it — [llm-compressor docs](https://docs.vllm.ai/projects/llm-compressor/en/latest/examples/reap_expert_pruning/); [kwanhee](https://huggingface.co/kwanhee/Kimi-K2.5-REAP50-INT4-W4A16)
- 0xSero also ships REAP of GLM-4.7 (185B/218B W4A16), MiniMax-M2.1 (139B/162B), Qwen3-Coder (57B/64B), Gemma-4-21B, DeepSeek-V4-Flash K160 — [0xSero search hits](https://huggingface.co/0xSero/DeepSeek-V4-Flash-0731-REAP); [Friendli listings](https://friendli.ai/models/0xSero/GLM-4.7-218B-W4A16)

### Inferences
- Every ≥500B prune in the wild traces back to two toolchains (CerebrasResearch/reap and llm-compressor's port) and one practitioner (0xSero) plus Cerebras itself; the quant side is split between Unsloth (mainline GGUF, broad but thin on 1T evals) and ubergarm (ik_llama.cpp, best PPL ladders).

### Gaps
- Bartowski has no ≥400B 1–2-bit quant with a PPL table in the results (only V4-Flash MXFP4 at KLD 0.0105).
- Lambda publishes no recipe of its own; its role is compute sponsorship.

## Community-reported failure modes at high compression on giant MoE

### Takeaway
The documented breakages are: (1) looping/attractor states after REAP (0xSero measured 7.2% vs 3.6% teacher, 13% on agentic prompts); (2) multi-turn tool-calling collapse at 50% REAP on Kimi-K2 (BFCL multi-turn 0.355→0.164) while single-turn code stays flat; (3) out-of-calibration-domain knowledge loss (GPQA −9 pts at 40% on GLM-4.6; chat-only calibration "lost most coding ability" on GLM-5.3-Flash); (4) 1-bit gibberish/looping for non-selective quants and a hard IQ2 cliff on sparse-routing V4-Pro (PPL 21.7); (5) anecdotal "gibberish"/"can't hold a chain of thought" reports for 40–50% REAP.

### Cited Findings
- Looping: GLM-5.2 REAP-504B loops on 7.2% of 2000 held-out agent prompts vs 3.6% for the teacher; codex/opencode-style prompts ~13% vs ~6%; `</think>`-restart loops; min_p 0.05 + rep-penalty 1.05/1.10 cuts to 4.9%/2.3%; the -K variant (full-data Router-KD) is 7.8% — [0xSero/GLM-5.2-504B](https://huggingface.co/0xSero/GLM-5.2-504B); [0xSero/GLM-5.2-504B-K](https://huggingface.co/0xSero/GLM-5.2-504B-K)
- Tool calling: Kimi-K2-W4A16 at 50% REAP keeps Eval+ (0.828→0.819) but BFCL multi-turn falls 0.355→0.164 and BFCL overall 0.666→0.564; at 25% multi-turn already 0.355→0.263 — [REAP paper Table 3](https://arxiv.org/html/2510.13999v2)
- Thin-domain loss: GLM-4.6 REAP 40% GPQA-Diamond 78.8→69.7 while HumanEval 96.3→95.1 — [cerebras GLM-4.6-REAP-218B-FP8](https://huggingface.co/cerebras/GLM-4.6-REAP-218B-A32B-FP8); GLM-5.2 REAP 34% SciCode 49.04→44.67 — [madeby561](https://huggingface.co/madeby561/GLM-5.2-Int8Mix-NVFP4-REAP-594B); Kimi-K2.5 REAP50 GPQA-D −12% vs OpenLLM −2% — [kwanhee](https://huggingface.co/kwanhee/Kimi-K2.5-REAP50-INT4-W4A16)
- Calibration-domain dependence: a chat-only expert selection of GLM-5.3-Flash "lost most of its coding ability despite only modest per-layer error"; per-token metrics may understate task damage; quality loss expected to grow with generation length; tool use and other languages absent from calibration may degrade — [OpenMOSE](https://huggingface.co/OpenMOSE/GLM-5.3-Flash-REAP-250B-A18B); 0xSero: "pruning without code calibration destroys coding ability" — [0xSero V3.2 card](https://huggingface.co/0xSero/DeepSeek-V3.2-345B-W3A16)
- Frequency-based pruning at 50% collapses completely (Qwen3-Coder Eval+ 0.007, Kimi-K2 SWE 0.000) — [REAP paper](https://arxiv.org/html/2510.13999v2)
- Forum anecdotes: "REAP50 and the model is pretty much useless, it can't hold a chain of thought … 5–15% may be tolerable"; a video claiming REAP "lobotomizes the model for tasks other than coding" — [Level1Techs GLM 5.2 REAP](https://forum.level1techs.com/t/glm-5-2-reap/251814); two Spark users got "gibberish" from MiniMax-M2.5-REAP NVFP4 ("mostly my experience using REAP models"), another: "40% is somewhat too much removed … 20–25% would be better" — [NVIDIA forum](https://forums.developer.nvidia.com/t/minimax-2-5-reap-nvfp4-on-single-dgx-spark/361248)
- 1-bit quant failures: non-Unsloth 1-/2-bit V3.1 quants "either failed to load or produced gibberish and looping outputs"; naive all-layer 1.5-bit R1 gave "gibberish and infinite repetitions" (fix: experts only at 1.5-bit, attention 4–6-bit) — [Unsloth Aider](https://unsloth.ai/docs/basics/dynamic-3.0-ggufs/unsloth-dynamic-ggufs-on-aider-polyglot); [r/LocalLLaMA 1.58-bit thread](https://lr.psf.lt/r/LocalLLaMA/comments/1ibbloy/158bit_deepseek_r1_131gb_dynamic_gguf/m9s3p8z/?context=3)
- Sparse-routing cliff: V4-Pro IQ2_XXS PPL 21.68 and IQ2_XS 4.47 rejected ("the IQ2 expert-quantization path reportedly breaks down on this sparse-routing MoE"); IQ1_S untested for instruction following, long context or agentic use — [6block](https://huggingface.co/6block/DeepSeek-V4-Pro-0813-GGUF)
- Int2 misfires: OPEA R1-0528 int2 answered an unrelated question in a sample; int2 kernels slower than int4 — [OPEA R1-0528](https://huggingface.co/OPEA/DeepSeek-R1-0528-int2-mixed-AutoRound)
- Long context: REAP V4-Flash user reported 4/4 needle-in-haystack to 3.5K tokens but a 2–3% "derailment" tail (essay/CJK on terse prompts) with no unpruned comparison; a REAP Qwen3.6 coding checkpoint's pass rate "barely moved" with a higher context ceiling (34% vs 36%) — [search summary of community cards](https://huggingface.co/0xSero/Gemma-4-21B); Spark UD-IQ3_XXS V4-Flash summarised a 66K-token doc with no factual errors — [Classmethod](https://dev.classmethod.jp/en/articles/dgx-spark-deepseek-v4-flash-0731-llama-cpp/)
- Loader/format traps: GLM-5.2 GGUFs fail in stock llama.cpp (indexer tensors on some layers only); REAP-228 experts not a multiple of 8/16 may break grouped-GEMM kernels; MTP layer left unpruned breaks engines that build it from config; kwanhee's weight_shape tensors break stock vLLM; nested quantization_config silently gives wrong results — [pipenetwork](https://huggingface.co/pipenetwork/GLM-5.2-REAP50-Q3_K_M-GGUF); [OpenMOSE](https://huggingface.co/OpenMOSE/GLM-5.3-Flash-REAP-250B-A18B); [kwanhee](https://huggingface.co/kwanhee/Kimi-K2.5-REAP50-INT4-W4A16)

### Inferences
- The failure signature of pruning (loops, multi-turn tool collapse, out-of-domain knowledge loss) is different from that of ≤2-bit quantization (uniform PPL rise, rare derailment, gibberish when attention is quantized too hard), so stacking both produces two independent failure tails.
- Router-KD recovers only part of the loop rate (7.2% vs 3.6%) and a 6x larger KD set did not help, implying the attractor behaviour is in the removed experts, not the router.

### Gaps
- No systematic RULER/long-context or BFCL/τ²-bench study of 1–2-bit quants of a ≥400B model was found; all long-context evidence is single-run.
- No independent (non-author) benchmark of any 0xSero or Cerebras ≥400B prune was found.

## Any 1T-class model run on a 128 GB device at all, and how

### Takeaway
Yes, once: Kimi-K3 (2.8T) UD-IQ1_S at 634.6 GB was run on a single DGX Spark (121 GiB) by streaming routed experts from NVMe with the Pulsar engine at 0.70 tok/s decode (23-token prefill in 39 s); every other 1T-class run is multi-device or ≥256 GB.

### Cited Findings
- Pulsar SSD expert streaming on one DGX Spark: Kimi-K3 UD-IQ1_S 634.6 GB (1.83 bpw), decode 0.70 tok/s (64 tokens in 91.8 s), 0.63 tok/s with a 40 GB host cache, VRAM cache 0% hits, host cache 37% hits, 100 GB cache budget made the machine unresponsive (practical ~80 GB), one crash, output "somewhat thin" — [Classmethod](https://dev.classmethod.jp/en/articles/dgx-spark-pulsar-ssd-expert-streaming/); engine: [giannisanni/pulsar](https://github.com/giannisanni/pulsar)
- colibri streams GLM-5.2 (744B) experts from disk; estimated to need ~6x RTX 5090; not a 128 GB single-device result — [Level1Techs](https://forum.level1techs.com/t/glm-5-2-reap/251814)
- Unsloth: Kimi-K2.5 1.8-bit can run via mmap below 240 GB "but speed may fall to under 2 tokens/s"; ~10 tok/s needs 24 GB GPU + ~256 GB RAM — [Unsloth K2.5](https://unsloth.ai/docs/models/tutorials/kimi-k2.5); ubergarm: without --no-mmap Kimi-K2-Thinking pages from disk at "a couple tok/sec maybe" — [ubergarm](https://huggingface.co/ubergarm/Kimi-K2-Thinking-GGUF)
- Kimi-K2-Thinking on 4x M3 Ultra ~28 tok/s (Geerling, via secondary source); K3 UD-IQ1_S on 3x Spark ~6.6 tok/s (unsourced) — [pinggy](https://pinggy.io/blog/self_hosting_llms_on_512gb_m5_ultra_mac_studio/); [ContextStudios](https://www.contextstudios.ai/local-ai/vcruz305--kimi-k3-neuron-iq1s-3spark-sparkinfer)
- The madebyagents Thor model directory claims Kimi-K2-0905 "fits well" on a 128 GB Thor; that is a calculator estimate, contradicted by every size above — [madebyagents](https://www.madebyagents.com/hardware/nvidia-jetson-agx-thor-developer-kit/models)

### Inferences
- The smallest in-memory 1T-source artifacts are 215–219 GiB (Ling-1T and Kimi-K2-Thinking smol-IQ1_KT), ~1.7–1.8x over a 128 GB device; a REAP-50% of Kimi-K2.x at ~1.85 bpw would land near 110–130 GiB, but no one has shipped that combination.

### Gaps
- No in-memory run of a ≥1T model on any single 128 GB device exists in the sources found.

## Single best quality-per-GiB artifact in the ≤128 GB class, and the recipe behind it

### Takeaway
Among ≥400B sources, the best measured quality-per-GiB at ≤128 GiB is ubergarm's Qwen3.5-397B-A17B smol-IQ2_KS (108.1 GiB, 2.34 bpw, PPL +12.3% vs BF16) with smol-IQ1_KT (88.8 GiB, +22%) as the smallest functional option; it is quant-only (no pruning), mixed-type ik_llama.cpp with imatrix. For strictly ≥500B sources nothing fits with a vs-full eval; the nearest are ubergarm R1-0528 IQ1_S_R4 (130.2 GiB, +52% PPL) and 0xSero GLM-5 REAP50 UD-IQ2_M (130 GB, pilot only).

### Cited Findings
- Qwen3.5-397B ladder at ≤128 GiB: smol-IQ2_XS 113.4 GiB +11.1% (mainline llama.cpp), smol-IQ2_KS 108.1 GiB +12.3%, smol-IQ1_KT 88.8 GiB +22.0%; next size up IQ2_KL 138.1 GiB +4.8% — [ubergarm/Qwen3.5-397B-A17B-GGUF](https://huggingface.co/ubergarm/Qwen3.5-397B-A17B-GGUF)
- 284B-source alternative with task evals on a Spark: V4-Flash UD-IQ3_XXS 97.1 GiB, 16.5 tok/s, 131K ctx, 6/6 tool calls, 5/5 code fixes; KLD for the sibling UD-Q4_K_XL is 0.0102 — [Classmethod](https://dev.classmethod.jp/en/articles/dgx-spark-deepseek-v4-flash-0731-llama-cpp/); [Unsloth V4](https://unsloth.ai/docs/models/deepseek-v4)
- ≥500B near-misses: R1-0528 IQ1_S_R4 130.2 GiB PPL 4.88 vs 3.21 — [ubergarm](https://huggingface.co/ubergarm/DeepSeek-R1-0528-GGUF); GLM-5 REAP50 IQ2_M 130 GB, 80% of a 50-item pilot — [0xSero](https://huggingface.co/0xSero/GLM-5-REAP-50pct-UD-IQ2_M-GGUF); Unsloth R1 UD-IQ1_S 131 GB, anecdote only — [Unsloth](https://unsloth.ai/docs/basics/tutorials-how-to-fine-tune-and-run-llms/deepseek-r1-how-to-run-locally)
- Best ≥500B artifacts with real tables all exceed 128 GB: GLM-5.2-504B NVFP4 Router-KD (~313 GB class, Aider 90.1, TB2.1 70.5) — [0xSero](https://huggingface.co/0xSero/GLM-5.2-504B); Kimi-K2.5 REAP50 INT4 288 GB (98% OpenLLM) — [kwanhee](https://huggingface.co/kwanhee/Kimi-K2.5-REAP50-INT4-W4A16); Qwen3-Coder-REAP-246B FP8 (~246 GB class, ≥96% retained) — [Cerebras](https://huggingface.co/cerebras/Qwen3-Coder-REAP-246B-A35B-FP8)
- Recipe of the winner: ik_llama.cpp quant with attention/shared-expert/dense tensors at higher types and routed experts at IQ2_KS or IQ1_KT (trellis), imatrix calibrated, PPL measured on wiki.test.raw; mainline-compatible mixes (Q3_K, smol-IQ2_XS) released alongside — [ubergarm](https://huggingface.co/ubergarm/Qwen3.5-397B-A17B-GGUF)

### Inferences
- Quality-per-GiB ranking among shipped ≤~130 GiB artifacts of ≥400B sources (relative PPL rise per GiB saved, computed from cited rows): Qwen3.5-397B smol-IQ2_KS (+12% at 108 GiB from 738 GiB) >> R1-0528 IQ1_S_R4 (+52% at 130 GiB from 666 GiB Q8_0) > GLM-5.2 REAP50+Q3_K_M (+37.5% at 169 GB, does not fit) ; the pruned+2–3-bit artifacts are not competitive with quant-only at equal bytes on the metrics that exist.
- For a 1T-class source the record shows the floor of in-memory quant-only is ~215 GiB at ~1.85 bpw; reaching ≤108 GiB therefore requires ≥50% expert pruning on top of ~1.85 bpw, a combination no one has shipped or evaluated, and whose two failure tails (loops/tool-collapse from pruning, +40–70% PPL from 1.8-bit) are each individually documented above.

### Gaps
- No ≤128 GB artifact of a ≥500B model has a benchmark table against its unpruned source; "best" above rests on wiki.test PPL and a 50-item pilot.
- Qwen3.5-397B is 397B, marginally under the ≥400B bar; if the bar is strict there is no evaluated ≤128 GiB artifact at all.
