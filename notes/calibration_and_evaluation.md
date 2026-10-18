# Calibration corpus design and mask-ranking evaluation for MoE expert pruning

Scope: (A) how to build the calibration corpus for REAP/HOPE-style one-shot expert pruning so general "ballast" plus code/math/agentic/finance/science/vision survive, under licence constraints; (B) evaluation methodology sensitive enough to rank candidate prune masks. Internal measurements are cited to the local memory notes in `/home/patrickd/.claude/projects/-home-patrickd/memory/`. Licences were verified on 2026-10-17 via the HF Hub API (`/api/datasets/<id>` cardData.license + tags) and dataset-card README text; "size" is the HF `size_categories` tag unless stated.

## KQ1. Corpus design: token budget / saturation, domain weighting, stratified vs blocked order

### Takeaway
Published one-shot expert-pruning pipelines calibrate on 10^2–10^4 prompts (REAP: 1,024 × 2,048 tokens for ≤110B, 12,228 prompts × ≤16,384 tokens for ≥110B; HOPE: prune-set agreement with the final mask saturates within ~10k prompts); dense-pruning studies show saliency is "robust to data amount" past ~128 × 2,048 tokens (0.1–0.2% fluctuation) but that *composition* moves results by up to 18.8 points at high sparsity. No paper tests stratified-vs-blocked chunk order; the only evidence is ours (blocked order left media domains at 74–85% of a 33 h pass).

### Cited Findings
- REAP calibration: "1,024 samples are randomly selected and packed to 2,048 sequence length" for models up to 110B; for ≥110B "we select 12,228 samples with a maximum sequence length of 16,384 tokens without truncation or packing." Datasets: C4 and evol-codealpaca for MC/code; xlam-function-calling and SWE-smith-trajectories added for ≥110B; WritingPrompts (curated) for creative writing; tulu-3-sft-personas-math for math. No ablation on calibration set size is reported; the only composition ablation is C4 vs domain-specific. — [REAP, arXiv 2510.13999v2](https://arxiv.org/html/2510.13999v2)
- REAP reasoning-model recipe: 12,228 samples "including tool calling and agentic trajectory data" plus 4,096 samples from Open-R1 Mixture-of-Thoughts Code/Math/Science, max seq len 16,384. Kimi-Linear-48B-A3B used 24,414 samples (evol-codealpaca, xlam-function-calling, SWE-smith-trajectories, WildChat, NuminaMath, Aya); no per-source counts given. — [REAP appendix](https://arxiv.org/html/2510.13999v2)
- REAP saliency is the mean of g_j(x)·‖f_j(x)‖₂ over tokens where expert j is active (conditional on activation), which "decouples the score from usage frequency"; upstream repo renormalises top-k router logits to sum to 1 since 2026-03-11 (mean accuracy drop 1.9% with fix vs 2.6% without). — [REAP](https://arxiv.org/html/2510.13999v2); [CerebrasResearch/reap](https://github.com/CerebrasResearch/reap); [memory: reap-logit-renorm-fix](/home/patrickd/.claude/projects/-home-patrickd/memory/reap-logit-renorm-fix.md)
- HOPE (Higher-Order Pruning of Experts; REAP is the special case with interaction terms zeroed) calibrates on all 111k Evol-CodeAlpaca-v1 prompts generated through the unpruned model, or on SWE-bench Verified trajectories (500 instances, traces >100k tokens truncated), served at max 8,192 tokens with a 2,048-token generation budget. Convergence (Fig. S6, Qwen3.5-35B-A3B, 50%): "Both HOPE and REAP stabilize within 10k prompts, with HOPE requiring marginally more data to converge due to estimating pairwise (rather than per-expert) statistics." Cost: REAP 1.20 h / 1k prompts, HOPE 1.27 h on Qwen-35B; 1.48 h vs 1.55 h on Qwen-122B. — [HOPE, arXiv 2609.18916v3](https://arxiv.org/html/2609.18916v3)
- HOPE calibration-domain sensitivity: across the two calibration domains the prune sets differ substantially — "Average calibration Jaccard index: HOPE 0.60, REAP 0.61" — whereas three random trials of the same set give ">0.95 Jaccard similarity on average". — [HOPE](https://arxiv.org/html/2609.18916v3)
- HOPE results at 50% on Qwen3.5-122B-A10B (Evol-CA calibration): HOPE mean 43.75±1.62 vs REAP 40.18±2.86 — the gap is driven by MMLU (48.93 vs 20.87), BBH (50.31 vs 38.65) and PopQA (8.90 vs 1.68), i.e. *general-knowledge* retention, with code-only calibration. On GLM-4.5-Air REAP beats HOPE in 3 of 4 cells at 40/50%. — [HOPE](https://arxiv.org/html/2609.18916v3)
- Dense-pruning saturation: with DCLM-7B (Wanda/DSnoT, 2:4), calibration sizes 64–2048 samples (× 2,048 tokens) give "fluctuations of only 0.1%-0.2%"; "more data does not close the gap between sources"; seed std falls as data grows. Source choice matters more: Wikipedia trails DCLM by 1.3–1.8%; the gap grows "from a 0.5% difference at 50% sparsity to 2.3% at 60% sparsity", and at 60% Wikipedia/SlimPajama calibration was worse than magnitude pruning with no calibration. — [Beware of Calibration Data, ICLR 2025, arXiv 2410.17711v2](https://arxiv.org/html/2410.17711v2)
- SliceGPT accuracy rises sharply from 128 to 512 calibration examples then levels off beyond 1024. — [From LLMs to LRMs, arXiv 2601.18091](https://arxiv.org/pdf/2601.18091)
- Calibration-data source and format (Llama-2-Chat-7B, 50% Wanda, 128 × 2,048): Pile 0.3819 > OSCAR 0.3671 > C4 0.3657 > RedPajama 0.3622; at 70% Pile 0.1713 vs C4 0.1275. Few-shot/ICL formatting of downstream data raised GSM8K-calibration from 0.2049 (zero-shot) to 0.3803 (ICL); nonsense calibration (ellipses) 0.2241. — [Is C4 Optimal for Pruning?, arXiv 2410.07461](https://arxiv.org/html/2410.07461)
- Multi-source mixing: "across 15 sources no single one preserves all capabilities"; at 60% sparsity strategies within 2.85 points on commonsense accuracy differed by 51.9 points in code retention; a uniform four-way "Balanced" mixture under a fixed budget "outperforms every evaluated single source", margin over C4 growing from +4.6 points at 30% to +18.8 at 60%; "Capability coverage is a first-order design variable for high-sparsity pruning." — [Calibration Data Trade-offs Across Capability Dimensions, arXiv 2606.03328](https://arxiv.org/abs/2606.03328)
- MoE few-shot domain calibration (DeepSeek-R1/V3-0324, EASY-EP): 25 samples/domain; "25 demonstrations are sufficient to capture all domain-specific experts" (99% overlap); single-sample calibration is unstable. Math-only calibration on R1 kept AIME24 79.17 but dropped Agent-OS to 3.47 (full 40.51), USMLE to 46.43 (full 92.66), FinanceIQ to 58.20 (full 82.1); mixed-domain R1 average 61.47 vs full 67.20 vs domain-specific 66.55. — [Domain-Specific Pruning of Large MoE with Few-shot Demonstrations, arXiv 2504.06792](https://arxiv.org/html/2504.06792)
- Generic TB-Coverage: calibrates on "512 sequences of length 1024 tokens from each corpus" (WikiText2 and C4), ranks experts separately per corpus per layer, and protects a fixed budget of top experts (20 for Qwen1.5-MoE, 24 for DeepSeek-MoE-16B) by round-robin across corpora before filling the rest. At 50% retention: Qwen 0.598 vs REAP 0.552 (W.PPL 16.76 vs 23.77); DeepSeek 0.606 vs REAP 0.542. No ablation on protection budget ("important direction for future work"). — [arXiv 2607.01710](https://arxiv.org/html/2607.01710)
- Minitron (dense): importance estimated on "1024 samples drawn randomly from the full dataset"; aggregation (batch=L2, seq=mean) chosen; aggregation choice alone swings zero-shot 8T LM loss from 7.18 to 10.55. — [Minitron, arXiv 2407.14679](https://arxiv.org/html/2407.14679)
- Internal: on GLM-5.3-Flash REAP50, `general` was 4.76% of calibration routed tokens (agentic 28.4%, math 26.0%); retained saliency mass general 0.4865 vs agentic 0.7466/code 0.7283/math 0.7133/science 0.7204; a bucket-balanced re-mask recovered only +0.028 for general — "re-masking cannot create information the calibration never collected"; the `bio` bucket got 0 tokens (corpus bug). — [memory: reap-criterion-and-corpus-audit](/home/patrickd/.claude/projects/-home-patrickd/memory/reap-criterion-and-corpus-audit.md)
- Internal: dropping REAP's 1/N_j (rank by saliency *mass*) was a Pareto win on GLM-5.3-Flash (recon 0.2697→0.1959, all seven domains up) but "makes the mask more corpus-dependent than the published mean, so it INTERACTS with the pass-3 corpus rebalance". — [memory: reap-criterion-and-corpus-audit](/home/patrickd/.claude/projects/-home-patrickd/memory/reap-criterion-and-corpus-audit.md)
- Internal: the MiMo corpus builder emitted 27 chunks grouped by bucket; the pass walked agentic 0-5, code 6-10, math 11-14, ballast 15-19, image 20-21, audio 22, video 23, science 24-25, finance 26, putting media at 74–85% of a 33 h pass; sorting chunks by fractional position within bucket moved image 74%→11%, audio 81%→0%, video 85%→4%, all nine domains represented within the first nine chunks; worst-domain objectives are identically zero when any domain is empty. — [memory: calibration-corpus-visit-order](/home/patrickd/.claude/projects/-home-patrickd/memory/calibration-corpus-visit-order.md)

### Inferences
- Token budget: for a frequency-invariant criterion (REAP mean), ~2M tokens (1,024 × 2,048) is the published floor and ~10k prompts (HOPE's convergence point; at 8k context that is ≤80M tokens) is the published ceiling; dense-pruning evidence says the saliency *estimate* saturates by ~0.25–1M tokens per domain. The binding constraint is therefore per-domain coverage, not total tokens: every domain whose experts you want protected needs its own ≥25-demonstration / ≥~0.5M-token slice, and with a mass-weighted criterion the per-domain token share directly sets expert survival, so the mix must be chosen deliberately (see KQ2).
- Weighting: the published evidence (IGSP "Balanced" uniform mix beating every single source by up to 18.8 points; GEC's per-corpus protection beating pooled REAP by 4.6–6.4 points at 50%) supports *equal-by-domain* budgets plus a per-domain protected top slice, not usage-proportional weighting. Inverse-frequency weighting has no published test; it is equivalent to equal weighting after rebalancing, so treat "equal + protect" as the default and usage-proportional as the thing to avoid.
- Chunk order: no external evidence; our own measurement is the only data, and it says per-chunk domain stratification (or at least round-robin chunk order) is required for any partial pass to be non-degenerate. Since HOPE/REAP accumulators are sums, stratified order costs nothing.
- The HOPE Jaccard of 0.60 across calibration domains means ~40% of the mask is set by the corpus, which matches our finding that the criterion gap (MAN/MSAN vs REAP: 89.4% mask agreement) is smaller than the corpus gap.

### Gaps
- No paper reports a saturation curve for *MoE expert* saliency in tokens (HOPE reports prompts only; no token counts). No paper tests stratified-vs-blocked order.
- No published test of inverse-frequency weighting or of what happens when per-domain budgets are unequal by design.
- REAP's Figures A8/A9 (C4 vs domain-specific vs "general" mix) are image-only; exact numbers for the general-mix condition are not extractable.

## KQ2. Ballast protection: what fraction should be general text?

### Takeaway
No paper measures a dose-response curve for the general-text fraction; the closest evidence is (a) our own pipeline where general at 4.76% of tokens collapsed to 48.7% retained mass and dNLL 1.0027 on ballast while code stayed at 0.057, (b) HOPE at 50% with code-only calibration where REAP's MMLU fell to 20.87 and PopQA to 1.68, and (c) GEC/IGSP showing per-corpus protection or a uniform mix recovers general accuracy. The evidence supports "general ≥ the share of any single specialised domain, plus a protected per-domain top slice", but the exact fraction is unmeasured.

### Cited Findings
- Internal: general 4.76% of routed calibration tokens → retained saliency mass 0.4865 vs 0.71–0.75 for the specialised domains; eval by_domain: ballast dNLL 1.0027 / top-1 agreement 0.579 vs code 0.057 / 0.916. — [memory: reap-criterion-and-corpus-audit](/home/patrickd/.claude/projects/-home-patrickd/memory/reap-criterion-and-corpus-audit.md)
- REAP on C4-only calibration collapses code: Qwen3-30B-A3B REAP-50% Eval+ 0.329 / LiveCode 0.104 (C4) vs 0.780 / 0.302 (evol-codealpaca), with many C4-calibrated rows at 0.000; but C4 "slightly improved accuracies for all compression methods" on MC. — [REAP](https://arxiv.org/html/2510.13999v2)
- Code-only calibration at 50% on Qwen3.5-122B: REAP MMLU 20.87±0.02, PopQA 1.68±1.54, BBH 38.65; HOPE 48.93, 8.90, 50.31 (baseline MMLU not shown in the excerpt). — [HOPE](https://arxiv.org/html/2609.18916v3)
- Math-only calibration on DeepSeek-R1 (128/256 experts): Agent-OS 3.47 vs 40.51 full; USMLE 46.43 vs 92.66; FinanceIQ 58.20 vs 82.1. — [arXiv 2504.06792](https://arxiv.org/html/2504.06792)
- Uniform four-way mix beats every single source by +4.6 (30% sparsity) to +18.8 points (60%); IGSP self-generated mixture beats prior self-generation by 7.23–12.79 points. — [arXiv 2606.03328](https://arxiv.org/abs/2606.03328)
- Per-corpus protected top experts (20–24 of 60/64) on generic corpora only: +4.6 to +6.4 average accuracy over pooled REAP at 50% retention; W.PPL 16.76 vs 23.77. — [arXiv 2607.01710](https://arxiv.org/html/2607.01710)
- Calibration closest to pre-training distribution is best for dense pruning (DCLM calibration best on DCLM-7B; Wikipedia worst by 1.3–1.8%). — [arXiv 2410.17711v2](https://arxiv.org/html/2410.17711v2)
- Pruning is ~14x more damaging per unit compression than quantisation on our stack (REAP50 +21.0% ppl vs a GGUF bit-step +1.46% / NVFP4 +1.56%). — [memory: reap-criterion-and-corpus-audit](/home/patrickd/.claude/projects/-home-patrickd/memory/reap-criterion-and-corpus-audit.md)

### Inferences
- With a mass-weighted criterion, retained mass per domain tracks token share almost linearly (0.0476 share → 0.49 retained vs 0.26–0.28 share → 0.71–0.75). Extrapolating that relation, general needs at least parity with the largest specialised bucket (~25–30% of tokens in a 6–7 domain mix) to reach the same ~0.72 retained mass; the published "Balanced" result says equal shares is the safe default. [Extrapolation.]
- Because the best single calibration source for *general* retention is pre-training-like web text (DCLM/FineWeb-style, not Wikipedia), the ballast slice should be web+book pre-training text in raw format, while specialised slices should be in chat/ICL format (REAP applies the chat template to SFT data; ICL format gave +0.18 on GSM8K calibration).
- Protect-then-fill (GEC's fixed per-domain budget) is the only published mechanism that *guarantees* thin domains keep experts regardless of mix; it is the published analogue of our "per-domain top slice" and should remain in place even after rebalancing.

### Gaps
- No measured dose-response (general fraction vs general-NLL or MMLU) exists in any source found; our own pass-3 rebalance is the first such measurement available.
- The protection-budget size (how many experts per domain per layer) has no sensitivity analysis anywhere (GEC explicitly defers it).

## KQ3. Sequence length: does 2k–8k calibration generalise to 32k–128k routing?

### Takeaway
No source measures routing drift or affected-rate vs token position after expert pruning; published MoE pruning calibrates at 1k–16k (REAP ≥110B: 16,384; HOPE: 8,192 serving cap with trajectories up to 100k) and evaluates at short context. Indirect evidence: routers are largely position-agnostic (router can only tell first vs second half of a sequence), REAP's JSD vs completion position grows faster for merged than pruned models, and OpenMoE found later tokens more likely to be dropped under capacity limits.

### Cited Findings
- REAP calibrates ≥110B models at "maximum sequence length of 16,384 tokens without truncation or packing"; ≤110B at 2,048 packed. — [REAP](https://arxiv.org/html/2510.13999v2)
- HOPE: vLLM max 8,192 tokens with 2,048 generation budget for Evol-CA calibration; SWE-bench trajectories truncated only above 100k tokens. — [HOPE](https://arxiv.org/html/2609.18916v3)
- GEC calibrates at 1,024-token sequences. — [arXiv 2607.01710](https://arxiv.org/html/2607.01710)
- REAP Fig. 4c: "Initially, all compressed models share close alignment with the baseline model"; "The merged model logits diverge from the baseline more rapidly than the pruned model" as completion token position increases (100 evol-codealpaca prompts, 50%, Qwen3-30B). No numeric values. — [REAP](https://arxiv.org/html/2510.13999v2)
- Router position-sensitivity probe: the MoE router "is independent of token position"; a classifier on router inputs "fails to predict parity but can successfully determine whether a token belongs to the first or second half of a sequence". — [On the Spatial Structure of MoE, arXiv 2504.04444](https://arxiv.org/pdf/2504.04444)
- OpenMoE: routing depends mostly on token IDs with little context sensitivity; later tokens in a sequence are more likely to be dropped ("drop-towards-the-end"), hurting multi-turn. — [OpenMoE, arXiv 2402.01739](https://arxiv.org/abs/2402.01739v1)
- Dense/SSM pruning: GHOST evaluates models calibrated at ≤2,048 on longer sequences; Taylor-based pruning degrades badly as evaluation length grows (~317 ppl at 128 to 1613 at 2048) while SparseGPT tracks the dense baseline. SlimGPT varies calibration length 64–2048 and sees diminishing returns like sample count. Nothing tested at 8,192+. — [GHOST, arXiv 2602.11408](https://arxiv.org/pdf/2602.11408); [SlimGPT, arXiv 2412.18110](https://arxiv.org/pdf/2412.18110)
- Only long-context number for expert-pruned MoE found: dynamic pruning (ODP) on Mixtral 8x7B at ~14.9% parameter reduction scored 99.26 on needle-in-a-haystack vs ~93–96 for token-statistic baselines; no context-length sweep. — [Mixture Compressor, arXiv 2410.06270](https://arxiv.org/pdf/2410.06270)
- Internal: a 16,384-token calibration forward needs >93 GiB on Thor so calibration is capped near 8,192. — [task brief / memory: thor-unified-memory-oom](/home/patrickd/.claude/projects/-home-patrickd/memory/thor-unified-memory-oom.md)

### Inferences
- Since routers are near position-agnostic and routing is token-ID dominated, expert saliency at 8k should transfer to 32k–128k *for the same token distribution*; the risk is distributional (long-context inputs contain token types—repeated tool outputs, long code files—under-represented in 8k windows), not positional. Mitigation: include long-document *content* (tool-call trajectories, long code files, papers) chunked into 8k windows rather than trying to calibrate at 16k. [Inference.]
- A cheap diagnostic we can run: bin our routing-fidelity metric (fraction of tokens whose top-8 all survived) by absolute position on 32k–128k held-out sequences; if it is flat, drift is not a concern. No published equivalent exists.

### Gaps
- No measurement anywhere of affected-rate vs token position after expert pruning, nor of calibration-length vs long-context eval for MoE expert pruning; RULER/LongBench on expert-pruned models not found.

## KQ4. Licence-clean sources per domain (HF-verified 2026-10-17)

### Takeaway
Clean (Apache/MIT/CC-BY/ODC-By/CC0/CDLA-permissive) sources exist at scale for every domain; the traps are SciQ (CC-BY-NC-3.0), ShareGPT4V and APIGen-MT (CC-BY-NC-4.0), R1-Distill-SFT (CC-BY-NC-SA), Wikipedia (CC-BY-SA), Nemotron-Math-v2 (partly CC-BY-SA), ChartQA (GPL-3.0), Cauldron/FineVision (per-sub-dataset licences), and gated NVIDIA Data Access Agreement sets (Nemotron-CC-v2/v2.1, Pretraining-Code-v1) which permit commercial training but are not a standard open licence.

### Cited Findings
General web/book ballast:
- HuggingFaceFW/fineweb — ODC-By 1.0, 10B–100B rows, not gated ("full dataset under the ODC-By 1.0 license"). — [card](https://huggingface.co/datasets/HuggingFaceFW/fineweb)
- HuggingFaceFW/fineweb-edu — ODC-By, 1B–10B rows. — [card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
- mlfoundations/dclm-baseline-1.0 — CC-BY-4.0, 1B–10B rows, not gated. — [card](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0)
- HuggingFaceFW/finepdfs — ODC-By, 100M–1B rows. — [card](https://huggingface.co/datasets/HuggingFaceFW/finepdfs)
- allenai/dolma — ODC-By, >1T tokens. — [card](https://huggingface.co/datasets/allenai/dolma)
- tiiuae/falcon-refinedweb — ODC-By, 100M–1B rows. — [card](https://huggingface.co/datasets/tiiuae/falcon-refinedweb)
- allenai/c4 — ODC-By, 10B–100B rows. — [card](https://huggingface.co/datasets/allenai/c4)
- Skylion007/openwebtext — CC0-1.0, 1M–10M docs. — [card](https://huggingface.co/datasets/Skylion007/openwebtext)
- sedthh/gutenberg_english — MIT (public-domain books), 10K–100K books. — [card](https://huggingface.co/datasets/sedthh/gutenberg_english)
- allenai/WildChat-1M (real chat) — ODC-By since 2024-06-26, retroactive. — [card](https://huggingface.co/datasets/allenai/WildChat-1M)
- allenai/tulu-3-sft-mixture — ODC-By, 100K–1M; allenai/Dolci-Instruct-SFT — ODC-By, 1M–10M. — [tulu](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture); [Dolci](https://huggingface.co/datasets/allenai/Dolci-Instruct-SFT)
- AVOID: wikimedia/wikipedia — CC-BY-SA-3.0 + GFDL (share-alike). — [card](https://huggingface.co/datasets/wikimedia/wikipedia)
- CONDITIONAL: nvidia/Nemotron-CC-v2 — "NVIDIA Data Agreement for Model Training", gated (manual), ~5.9T tokens; "enables training of any AI model, including models released under proprietary or open source licenses", but flow-down of Qwen/DeepSeek licence terms for synthetic portions; requires agreeing to use "for model training purposes only". — [card](https://huggingface.co/datasets/nvidia/Nemotron-CC-v2)
- No licence field: EleutherAI/pile ("other"), cerebras/SlimPajama-627B, togethercomputer/RedPajama-Data-V2, PleIAs/common_corpus (API returns no licence tag; card claims public-domain content but unverified here), bookcorpus ("unknown"). — [HF API query results, this note]

Code:
- bigcode/the-stack-v2 — tag "other"; "all files contained in the dataset are licensed with one of the permissive licenses" or no licence; gated; bulk download "requires a an agreement with SoftwareHeritage and INRIA"; ~900B tokens training split; users must honour removal requests. — [card](https://huggingface.co/datasets/bigcode/the-stack-v2)
- nvidia/OpenCodeReasoning — CC-BY-4.0, 100K–1M; nvidia/OpenCodeInstruct — CC-BY-4.0, 1M–10M. — [OCR](https://huggingface.co/datasets/nvidia/OpenCodeReasoning); [OCI](https://huggingface.co/datasets/nvidia/OpenCodeInstruct)
- bigcode/commitpackft — MIT; bigcode/self-oss-instruct-sc2-exec-filter-50k — ODC-By. — [commitpackft](https://huggingface.co/datasets/bigcode/commitpackft); [self-oss](https://huggingface.co/datasets/bigcode/self-oss-instruct-sc2-exec-filter-50k)
- CONDITIONAL: nvidia/Nemotron-Pretraining-Code-v1 — NVIDIA Data Access Agreement, gated manual, 100M–1B. — [Specialized-v1 card licence section](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Specialized-v1)

Math:
- nvidia/OpenMathInstruct-2 — CC-BY-4.0, 10M–100M rows; nvidia/OpenMathReasoning — CC-BY-4.0, 1M–10M. — [OMI2](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2); [OMR](https://huggingface.co/datasets/nvidia/OpenMathReasoning)
- open-r1/OpenR1-Math-220k — Apache-2.0; AI-MO/NuminaMath-CoT and NuminaMath-1.5 — Apache-2.0 (100K–1M); TIGER-Lab/MathInstruct — MIT. — [OpenR1](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k); [Numina](https://huggingface.co/datasets/AI-MO/NuminaMath-1.5); [MathInstruct](https://huggingface.co/datasets/TIGER-Lab/MathInstruct)
- HuggingFaceTB/finemath — ODC-By, 10M–100M (pre-training-style math web text). — [card](https://huggingface.co/datasets/HuggingFaceTB/finemath)
- CAUTION: nvidia/Nemotron-Math-v2 — dual CC-BY-4.0 / CC-BY-SA-4.0 (subset share-alike). — [card](https://huggingface.co/datasets/nvidia/Nemotron-Math-v2)
- No licence tag: open-web-math, proof-pile-2. — [HF API]

Agentic / tool-use:
- nvidia/Nemotron-Agentic-v1 — CC-BY-4.0 (plus Apache-2.0 for its glaive portion). — [card](https://huggingface.co/datasets/nvidia/Nemotron-Agentic-v1)
- glaiveai/glaive-function-calling-v2 — Apache-2.0, 100K–1M; Team-ACE/ToolACE — Apache-2.0; NousResearch/hermes-function-calling-v1 — Apache-2.0; interstellarninja/hermes_reasoning_tool_use — Apache-2.0. — [glaive](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2); [ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE); [hermes](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1)
- Salesforce/xlam-function-calling-60k — CC-BY-4.0, gated (auto). — [card](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k)
- SWE trajectories: SWE-bench/SWE-smith-trajectories — MIT, 10K–100K; nebius/SWE-agent-trajectories — CC-BY-4.0; nebius/SWE-rebench — CC-BY-4.0; nvidia/Nemotron-SWE-v1 — CC-BY-4.0; SWE-Gym/SWE-Gym — MIT. — [SWE-smith](https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories); [nebius](https://huggingface.co/datasets/nebius/SWE-agent-trajectories); [Nemotron-SWE](https://huggingface.co/datasets/nvidia/Nemotron-SWE-v1)
- microsoft/orca-agentinstruct-1M-v1 — CDLA-Permissive-2.0, 1M–10M. — [card](https://huggingface.co/datasets/microsoft/orca-agentinstruct-1M-v1)
- AVOID: Salesforce/APIGen-MT-5k — CC-BY-NC-4.0. — [card](https://huggingface.co/datasets/Salesforce/APIGen-MT-5k)
- Not found on HF: nvidia/Nemotron-Agentic-v2 (404).

Finance:
- ibm-research/finqa — CC-BY-4.0. — [card](https://huggingface.co/datasets/ibm-research/finqa)
- Josephgflowers/Finance-Instruct-500k — Apache-2.0, 100K–1M; sujet-ai/Sujet-Finance-Instruct-177k — Apache-2.0; gbharti/finance-alpaca — MIT; Duxiaoman-DI/FinCorpus — Apache-2.0 (100K–1M); eloukas/edgar-corpus — Apache-2.0 (SEC 10-K filings, 100K–1M). — [Finance-Instruct](https://huggingface.co/datasets/Josephgflowers/Finance-Instruct-500k); [Sujet](https://huggingface.co/datasets/sujet-ai/Sujet-Finance-Instruct-177k); [edgar](https://huggingface.co/datasets/eloukas/edgar-corpus); [FinCorpus](https://huggingface.co/datasets/Duxiaoman-DI/FinCorpus)
- No licence tag: FinGPT/fingpt-sentiment-train, TheFinAI/fiqa, AdaptLLM/finance-tasks, ChanceFocus/flare-finqa, dreamerdeo/finqa — do not use without upstream check (financial_phrasebank upstream is CC-BY-NC-SA). — [HF API]

Science:
- nvidia/Nemotron-Science-v1 — CC-BY-4.0 (100K–1M); nvidia/OpenScienceReasoning-2 — CC-BY-4.0; nvidia/OpenScience — CC-BY-4.0 (1M–10M). — [Science-v1](https://huggingface.co/datasets/nvidia/Nemotron-Science-v1); [OSR2](https://huggingface.co/datasets/nvidia/OpenScienceReasoning-2)
- allenai/peS2o — ODC-By, 10B–100B tokens (academic papers). — [card](https://huggingface.co/datasets/allenai/peS2o)
- qiaojin/PubMedQA — MIT (abstracts, not clinical records). — [card](https://huggingface.co/datasets/qiaojin/PubMedQA)
- nvidia/Nemotron-Pretraining-Specialized-v1 — CC-BY-4.0 except Wiki-Rewrite and Scientific-Coding subsets (CC-BY-SA-4.0/GFDL). — [card](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Specialized-v1)
- AVOID: allenai/sciq — CC-BY-NC-3.0; armanc/scientific_papers and allenai/scitldr — "unknown". — [sciq](https://huggingface.co/datasets/allenai/sciq)

Vision-language:
- lmms-lab/LLaVA-OneVision-Data — Apache-2.0, 1M–10M. — [card](https://huggingface.co/datasets/lmms-lab/LLaVA-OneVision-Data)
- liuhaotian/LLaVA-Instruct-150K — CC-BY-4.0 (note: images are COCO, text generated by GPT-4; check OpenAI-terms exposure separately). — [card](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K)
- nvidia/Llama-Nemotron-VLM-Dataset-v1 and nvidia/Nemotron-VLM-Dataset-v2 — CC-BY-4.0, 1M–10M each. — [v1](https://huggingface.co/datasets/nvidia/Llama-Nemotron-VLM-Dataset-v1); [v2](https://huggingface.co/datasets/nvidia/Nemotron-VLM-Dataset-v2)
- allenai/pixmo-cap, pixmo-docs, pixmo-ask-model-anything, pixmo-points — ODC-By. — [pixmo-cap](https://huggingface.co/datasets/allenai/pixmo-cap); [pixmo-docs](https://huggingface.co/datasets/allenai/pixmo-docs)
- HuggingFaceM4/Docmatix — MIT, 1M–10M; allenai/olmOCR-mix-0225 — ODC-By; BAAI/DenseFusion-1M — CC-BY-4.0; MMMU/MMMU — Apache-2.0 (eval, 10K–100K). — [Docmatix](https://huggingface.co/datasets/HuggingFaceM4/Docmatix); [olmOCR](https://huggingface.co/datasets/allenai/olmOCR-mix-0225); [DenseFusion](https://huggingface.co/datasets/BAAI/DenseFusion-1M)
- CAUTION: HuggingFaceM4/the_cauldron and FineVision — no top-level licence; "Each of the publicly available sub-datasets ... are governed by specific licensing conditions"; prompts CC-BY-4.0 — filter sub-datasets individually. — [Cauldron](https://huggingface.co/datasets/HuggingFaceM4/the_cauldron); [FineVision](https://huggingface.co/datasets/HuggingFaceM4/FineVision)
- AVOID: Lin-Chen/ShareGPT4V — CC-BY-NC-4.0; HuggingFaceM4/ChartQA — GPL-3.0. — [ShareGPT4V](https://huggingface.co/datasets/Lin-Chen/ShareGPT4V); [ChartQA](https://huggingface.co/datasets/HuggingFaceM4/ChartQA)

Other traps found: ServiceNow-AI/R1-Distill-SFT — CC-BY-NC-SA-4.0; lmsys/lmsys-chat-1m — custom licence, gated, no tag. — [HF API]

### Inferences
- A fully clean 7-domain mix is feasible without any NC/SA source: FineWeb/DCLM + Gutenberg (ballast), The Stack v2 permissive + OpenCodeInstruct (code), OpenMathInstruct-2 + finemath (math), Nemotron-Agentic-v1 + SWE-smith/nebius trajectories + ToolACE (agentic), FinQA + Finance-Instruct-500k + EDGAR (finance), Nemotron-Science/OpenScience + peS2o (science), LLaVA-OneVision + Nemotron-VLM-v2 + PixMo (vision).
- Synthetic sets generated by DeepSeek/Qwen/GPT models carry flow-down model-licence terms even when tagged CC-BY (explicit in the NVIDIA cards); for a redistributed pruned checkpoint this is a documentation obligation, not a block.

### Gaps
- HF `license` tags are self-declared; I did not audit per-sub-dataset licences inside Cauldron/FineVision, nor the per-file licence list of The Stack v2.
- CC-BY-4.0 sets whose text was produced by GPT-4 (LLaVA-Instruct-150K) may conflict with OpenAI terms for a competing model; not resolved here.

## KQ5. Evaluation metrics sensitive enough to rank masks

### Takeaway
Token-level distributional metrics against a cached teacher (per-token KL / top-1 agreement / flips) are 1–2 orders of magnitude more sensitive than benchmark accuracy: flip rates of 3–7% appear where MMLU accuracy moved <1 point (Spearman 0.981 between KL and flips), llama.cpp reports KLD to ±6e-6 on Wikitext, and our own top-1 agreement has SE 0.00075 on 241k tokens. But our v2 pass showed reconstruction residual (−2.5%) does not predict agreement (−0.0001); only per-domain held-out metrics exposed the ballast collapse (ballast top-1 0.579 vs code 0.916).

### Cited Findings
- Flips vs accuracy (MMLU 5-shot): Llama2-70b-chat GPTQ W4A16 accuracy −0.45 with 4.26% flips; AWQ W4A16 −0.76 / 4.05%; BnB W4A4 −0.78 / 5.65%; Yi-34b BnB W4A4 −1.57 / 7.44%; max 13.62% (Llama2-7b SQ W8A8). "Spearman correlation on the MMLU benchmark is 0.981" between KL divergence and flips; "KL-Divergence on a given dataset can better identify the differences created due to various compression techniques". Compressed models with high flips scored worse on MT-Bench (80 two-turn questions). — [Accuracy is Not All You Need, NeurIPS 2024, arXiv 2407.09141](https://arxiv.org/html/2407.09141v1)
- llama.cpp `--kl-divergence` against saved FP16 logits reports mean KLD, 99th-percentile KLD, "same top p" (share of positions where both models' argmax agree, with binomial SE), mean/RMS Δp of the correct token; LLaMA-3-8B Wikitext: q8_0 KLD 0.001355 ± 0.000006, q6_K 0.005452 ± 0.000035. Logit cache is 11 GiB (LLaMA-2) / 37 GiB (LLaMA-3) for Wikitext-2 test. — [llama.cpp perplexity README](https://docs.rs/crate/llama-cpp-sys-4/0.4.4/source/llama.cpp/tools/perplexity/README.md); [PR #5076](https://github.com/ggerganov/llama.cpp/pull/5076)
- REAP's distribution-level diagnostics: Jensen-Shannon divergence of compressed vs baseline logits by completion position, cross-perplexity, n-gram diversity (100 prompts); REAP's benchmarks: lm-eval MC avg, EvalPlus/LiveCodeBench, SWE-Bench Verified, WildBench, GSM8K/MATH-500, BFCLv3. — [REAP](https://arxiv.org/html/2510.13999v2)
- HOPE evaluates on Tulu3 dev suite (GSM8K, MATH, IFEval, MMLU, BBH, TruthfulQA, PopQA, LiveCodeBench) with mean±std over 3 calibration seeds; per-benchmark stds at 50% range 0.00–10.45 (LCB REAP 34.09±10.45), i.e. benchmark noise often exceeds the method gap. — [HOPE](https://arxiv.org/html/2609.18916v3)
- Minitron ranks width candidates before retraining by zero-shot LM loss on an 8T validation split and WikiText2 loss; depth candidates by per-layer PPL and Block Importance (cosine distance). — [Minitron](https://arxiv.org/html/2407.14679)
- Puzzle-75B (NVIDIA) ablation ranks latent-pruning candidates by zero-shot LM loss and top-1 accuracy on a held-out validation split; healing KD matches logits at 32K sequence length. — [arXiv 2607.04371](https://arxiv.org/html/2607.04371)
- Perplexity misses capability loss: LLM-KICK found pruning degraded knowledge-intensive tasks "sometimes at 25–30% sparsity" while in-context retrieval/summarisation held at ≥50%. — [Compressing LLMs: The Truth is Rarely Pure, ICLR 2024, arXiv 2310.01382](https://arxiv.org/abs/2310.01382v2)
- Internal: s09_eval by_domain on GLM-5.3-Flash REAP50 — ballast dNLL 1.0027 / top-1 0.579 vs code 0.057 / 0.916; overall dNLL 0.1909; teacher FP8 logits are banked (`teacher_from_cache: true`), so arms re-score without re-running the 321 GB teacher. — [memory: reap-criterion-and-corpus-audit](/home/patrickd/.claude/projects/-home-patrickd/memory/reap-criterion-and-corpus-audit.md)
- Internal: v2 pass improved reconstruction residual by 2.5% while end-to-end top-1 agreement moved −0.0001 (SE 0.00075 on 241,516 tokens) — proxies did not predict the outcome. — [task brief]
- Internal: reconstruction error is REAP's own, frequency-weighted objective and "would be circular on its own"; per-domain retained saliency mass is the load-bearing offline evidence. — [memory: reap-criterion-and-corpus-audit](/home/patrickd/.claude/projects/-home-patrickd/memory/reap-criterion-and-corpus-audit.md)
- Internal (quant, not prune): IQ3_M/IQ4_XS/NVFP4 hidden states sit ~0.89–0.93 cosine apart vs 0.999 same-quant control — hidden-state cosine resolves quant-level differences. — [memory: glm53-quant-hidden-state-drift](/home/patrickd/.claude/projects/-home-patrickd/memory/glm53-quant-hidden-state-drift.md)

### Inferences
- Rank order of sensitivity for mask ranking, from the evidence: per-token KL (continuous, SE ~1e-5 at 300k tokens per llama.cpp) > top-1 agreement / flips (Bernoulli, SE ~7.5e-4 at 240k tokens) > ΔNLL (same tokens, but dominated by a few high-loss positions) >> benchmark accuracy (SE 1–3 points at 500–1000 questions). Use KL as the primary ranking statistic, top-1 agreement as the interpretable secondary, and report both per domain.
- Routing fidelity (fraction of held-out tokens whose top-8 experts all survived, or retained gate mass) is a zero-GPU proxy computable from the banked router cache; our data show it tracks per-domain damage (general 0.49 retained mass ↔ top-1 0.579) but, like reconstruction residual, it is a proxy — gate it against the end-to-end metric, never substitute.
- The "proxy did not predict outcome" result argues that reconstruction residual should be dropped as a selection criterion in favour of per-domain held-out KL against the cached teacher; the residual is computed on the calibration distribution it was optimised on and is weighted by the same imbalanced frequencies.

### Gaps
- No paper reports standard errors for token-level KL or agreement between *pruned* MoE masks; the llama.cpp numbers are for quantisation and dense models.
- No published routing-fidelity metric for expert pruning was found (our definition appears to be novel).

## KQ6. Experimental design: paired eval vs cached teacher, sample size to resolve 0.5%

### Takeaway
Miller's paired-difference formula applied to our measured variance says 0.5% in top-1 agreement is resolvable (80% power, α=0.05) with ≈43k held-out tokens; 0.2% needs ≈270k; 0.1% needs ≈1.1M; our current 241,516-token set has an MDE of ≈0.21% — before within-document clustering, which Miller measured at up to 3x SE inflation (i.e. 4–9x more tokens). Pair every arm on the same tokens against the same cached teacher, cluster SEs by document, and spend extra tokens on the thin domains.

### Cited Findings
- Paired SE: SE_{A−B,paired} = sqrt(Var(s_{A−B})/n); Var(paired) = Var(unpaired) − 2Cov(x_A,x_B)/n; "paired differences represent a 'free' reduction in estimator variance when comparing two models". Sample size (Eq. 9): n = (z_{α/2}+z_β)²(ω² + σ_A²/K_A + σ_B²/K_B)/δ²; worked example ω²=1/9, δ=0.03 → n ≈ 969 questions; MDE inversion (Eq. 10). Clustered SEs on real Anthropic evals were up to 3.05x the naive SE (DROP). — [Miller, Adding Error Bars to Evals, arXiv 2411.00640](https://arxiv.org/pdf/2411.00640)
- Wang 2025: HumanEval (164 q) at 50% accuracy — unpaired SE(A−B) 6% / detectable difference 11%; paired 4% / 8%; paired + averaging 1–2% / 2–4%; "paired prediction noise typically exceeds paired data noise, which means reducing prediction noise by averaging can significantly increase statistical power"; "Leading agentic evals often have fewer than 1000 questions". Recommends measuring data/prediction/total noise with all-pairs comparison and releasing question-level results. — [Measuring all the noises of LLM Evals, arXiv 2512.21326](https://arxiv.org/pdf/2512.21326)
- Internal measurement: top-1 agreement SE = 0.00075 on 241,516 held-out tokens, paired against cached FP8 teacher logits. — [task brief; memory: reap-criterion-and-corpus-audit](/home/patrickd/.claude/projects/-home-patrickd/memory/reap-criterion-and-corpus-audit.md)
- Internal eval hygiene already learned: assert served-model identity in every result (2 of 3 arms silently re-measured the same model); pin generation config; a changed `--seqmax` re-routes greedy decoding, so pin arena size across compared arms. — [memory: eval-orphaned-server-trap](/home/patrickd/.claude/projects/-home-patrickd/memory/eval-orphaned-server-trap.md); [memory: eval-seqmax-perturbs-greedy-output](/home/patrickd/.claude/projects/-home-patrickd/memory/eval-seqmax-perturbs-greedy-output.md)
- Prune-set seed stability is >0.95 Jaccard (3 trials), so mask-level noise from calibration sampling is small relative to the 0.60 Jaccard across domains. — [HOPE](https://arxiv.org/html/2609.18916v3)

### Inferences
- [Extrapolation from Miller Eq. 9 with our numbers] Per-token paired variance = SE²·n = (0.00075)²×241,516 = 0.136. With deterministic (greedy) scoring σ_A=σ_B=0: n(δ=0.005) = 7.849×0.136/2.5e-5 ≈ 42,700 tokens; n(δ=0.002) ≈ 267k; n(δ=0.001) ≈ 1.07M; n(δ=0.0005) ≈ 4.3M. MDE at 241,516 tokens = 2.80×0.00075 ≈ 0.0021. Our v2 delta of −0.0001 is 0.13 SE — indistinguishable from zero, as reported.
- The implied per-token correlation between two masks' agreement indicators is only ρ ≈ 0.25 (0.136 ≈ 2×0.09×(1−ρ) at p≈0.9), so pairing buys ~25% variance reduction at token level — less than the 2–4x Wang reports for question-level benchmarks; comparing masks that share most experts should raise ρ and tighten the SE further.
- Tokens inside one document are not independent; Miller's 1.1–3x cluster inflation implies the 43k-token figure for 0.5% should be budgeted as 0.2–0.4M tokens, and per-domain resolution at 0.5% needs that many tokens *per domain* (~1.5–3M total for 7 domains). All affordable with a cached teacher.
- Design: fixed held-out token set, stratified by domain in equal shares (not by usage), teacher logits cached once (FP8 teacher cache already exists), every arm scored on identical tokens with identical seqmax, bootstrap over documents (not tokens) for CIs, report per-domain KL and top-1 agreement with paired differences vs the incumbent mask. Benchmarks only as a final gate, since their MDE (~2–4 points even paired) is above the effects between masks.

### Gaps
- No published power analysis exists for token-level distributional metrics on pruned models; all numbers above are our own extrapolation from Miller's formula.
- The within-document cluster inflation factor for token-level KL/agreement is unmeasured in our pipeline; it should be estimated from the existing 241k-token run before fixing the budget.

## KQ7. Benchmarks that detect ballast collapse vs domain-specific loss

### Takeaway
General-knowledge/recall benchmarks (MMLU, PopQA, BBH, factoid QA in LLM-KICK) are the published detectors of ballast collapse — they moved 20–30 points in HOPE/REAP at 50% with code-only calibration while GSM8K stayed ~95 — whereas perplexity and in-context tasks stay flat; domain loss is caught by EvalPlus/LiveCodeBench, SWE-Bench Verified, BFCLv3, MATH-500 and domain QA (FinanceIQ, USMLE, GPQA), and only per-domain held-out NLL/KL exposes both simultaneously at low cost.

### Cited Findings
- Qwen3.5-122B at 50%, code calibration: REAP MMLU 20.87, PopQA 1.68, BBH 38.65 while GSM8K 94.70 — general knowledge collapsed, math held. — [HOPE](https://arxiv.org/html/2609.18916v3)
- Math-only calibration on DeepSeek-R1: GPQA 46.91 (full 70.91), USMLE 46.43 (92.66), FinanceIQ 58.20 (82.1), Agent-OS 3.47 (40.51); AIME24 kept at 79.17. — [arXiv 2504.06792](https://arxiv.org/html/2504.06792)
- LLM-KICK: factoid QA and MC reasoning QA degrade at 25–30% sparsity; in-context retrieval and summarisation robust at ≥50%; perplexity "can miss subtle changes in a compressed model's real capabilities". — [arXiv 2310.01382](https://arxiv.org/abs/2310.01382v2)
- REAP: C4 calibration gives slightly better MC averages but collapses code to 0% in several instances — MC and code benchmarks respond to opposite calibration choices. — [REAP](https://arxiv.org/html/2510.13999v2)
- Internal: letter-only MC scoring gave 0.48 vs 0.967 reasoned on Belebele-en for Gemma-4-26B — MC format artefacts can masquerade as capability loss. — [memory: letter-only-mc-format-artifact](/home/patrickd/.claude/projects/-home-patrickd/memory/letter-only-mc-format-artifact.md)
- Internal: a shared max_tokens scores tokens-to-answer, not capability; pin generation config across arms. — [memory: eval-fixed-budget-measures-economy](/home/patrickd/.claude/projects/-home-patrickd/memory/eval-fixed-budget-measures-economy.md); [memory: eval-pin-generation-config](/home/patrickd/.claude/projects/-home-patrickd/memory/eval-pin-generation-config.md)

### Inferences
- Ballast-collapse panel: MMLU (or MMLU-Pro) + PopQA + BBH + a WildChat/WildBench-style open-ended check; these are the benchmarks that moved most under code-only calibration. Domain panel: EvalPlus + LiveCodeBench (code), SWE-Bench Verified subset (agentic code), BFCLv3 (tool-use), GSM8K/MATH-500 (math), GPQA/PubMedQA (science), FinQA/FinanceIQ (finance), MMMU/DocVQA (vision).
- Because benchmark MDEs are 2–4 points paired, they cannot rank masks that differ by 0.1–0.5% top-1 agreement; use them to detect collapse (effects of 10+ points), and use per-domain token metrics to rank.

### Gaps
- No source quantifies a benchmark's sensitivity to *expert* pruning specifically (all stds found are across calibration seeds, not across masks).
- "Do Compressed LLMs Forget Knowledge?" (NeurIPS 2024 workshop) was surfaced but not read.
