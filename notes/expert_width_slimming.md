# Expert-width slimming and top-k reduction as alternatives/complements to whole-expert deletion

Scope: pruning the intermediate (FFN) dimension inside every expert ("slimming"), reducing top-k at inference, how both compose with whole-expert deletion (REAP-style), whether slimming can be done as block-aligned surgery on MXFP4/NVFP4/GGUF weights, and runtime support. Results are labelled **[training-free]**, **[mask-training only, experts frozen]**, or **[distilled, N tokens]**. Extrapolations are labelled **[inference]**.

Headline for the report writer: the 10-point SlimMoE gap (63.8 vs 53.4) is measured under SlimMoE's own gradient-based criterion with distillation in the loop; the closest **training-free** evidence (TENP, DERN, FlexMoE-on-Qwen2) says neuron-level slimming beats whole-expert deletion by a comparable or larger margin at 50-60% expert-parameter reduction, but every one of those results is on <=80B models and none of them is a clean "equal bytes, same criterion, zero training" comparison on a 500B-1T model. Block-aligned surgery on already-quantized experts is exact for gate/up and exact for down_proj only when channels are removed in whole aligned blocks (32 for MXFP4, 16 for NVFP4, 256 for K-quants). Top-k reduction is nearly free *if* the router renormalisation denominator is kept at the trained k (Qwen3.5-397B top-10 -> top-5 at -0.55 MMLU), but is a -2 to -5 point loss if done naively.

---

## KQ1. SlimMoE ablation in full: what was compared, with/without distillation, and is there a training-free version?

### Takeaway
SlimMoE's Table 4 compares expert *slimming* (gradient-sensitivity criterion) with whole-expert *pruning* (KL- or frequency-based) and M-SMoE merging on Phi-3.5-MoE at two prune ratios: at "50%" slimming scores 63.76 (KL) / 59.30 (CLM) vs 53.41 / 48.59 for expert pruning; at "25%" 43.25 / 37.52 vs 30.38 / 31.87. SlimMoE's criterion needs a backward pass (|grad x weight| from a KD loss), and the paper does not report any zero-training numbers, so the 10-point gap is *not* a training-free measurement. No paper reproduces the comparison training-free on the same model; the nearest substitutes are TENP, DERN and FlexMoE (KQ2).

### Cited Findings
- SlimMoE compresses Phi-3.5-MoE (41.9B total / 6.6B active) to Phi-mini-MoE (7.6B / 2.4B) and Phi-tiny-MoE (3.8B / 1.1B) using 400B tokens, <10% of the original training data — [SlimMoE abstract](https://arxiv.org/abs/2506.18349)
- Table 4 (benchmark not named in the extracted text; indexed by "prune ratio", parameter parity implied but not stated): at 50%: Slimming-KL 63.76, Slimming-CLM 59.30, Prune-Expert-KL 53.41, Prune-Expert-freq 48.59, M-SMoE 38.54. At 25%: 43.25 / 37.52 / 30.38 / 31.87 / 25.50 — [SlimMoE HTML](https://arxiv.org/html/2506.18349). Note: scores at the "25%" row are *lower* than at "50%", which only makes sense if the label means fraction of parameters *retained* (25% kept = more aggressive); treat the 63.8-vs-53.4 pair as the milder, 50%-kept setting.
- The text does not state whether Table 4 numbers are after a short distillation or zero-shot, nor the token budget of the ablation — [SlimMoE HTML](https://arxiv.org/html/2506.18349)
- Neuron importance criterion: first-order sensitivity |gradient x weight| computed from a top-8 teacher-logit KD loss plus the load-balancing auxiliary loss; per neuron, the l2 norm of sensitivities across its row of the down-projection W3; lowest-scoring neurons removed together with matching rows/columns of W1, W2, W3. Attention heads scored the same way on W_O, averaged per GQA group. Calibration: 16K uniformly sampled training examples — [SlimMoE HTML](https://arxiv.org/html/2506.18349)
- Sizes: Phi-3.5-MoE has 16 experts, top-2, expert intermediate 6400, 32 layers. Phi-mini-MoE keeps 16 experts / top-2 / all 32 layers and slims expert intermediate to 960 (15%); Phi-tiny-MoE to 448 (7%) plus 50% of GQA groups pruned. Expert count and depth are **never** reduced — [SlimMoE HTML](https://arxiv.org/html/2506.18349)
- Multi-stage schedule matters at this prune depth: Phi-mini one-stage vs multi-stage MMLU 68.58 vs 69.87, GSM8K 74.52 vs 77.48; Phi-tiny 58.40 vs 60.08; reduced 90B-token budget: 64.11 vs 65.59 MMLU. One-stage pruning "drops sharply at the first pruning step" ("model collapse"), shown only in figures — [SlimMoE HTML](https://arxiv.org/html/2506.18349)
- Post-trained MMLU: Phi-3.5-MoE 78.36 -> Phi-mini-MoE 70.68 (18% of params) -> Phi-tiny-MoE 60.83 (9%) — [SlimMoE HTML](https://arxiv.org/html/2506.18349)

### Inferences
- SlimMoE's slimming-vs-deletion gap is measured with (a) a gradient criterion and (b) distillation in the loop; neither applies directly to a forward-pass-only pipeline. The *direction* of the gap is corroborated training-free by TENP/DERN/FlexMoE (KQ2), but the *size* on a 500B+ model is unmeasured.
- SlimMoE went to 15% and 7% expert width with the expert count fixed, i.e. 85-93% width reduction. Our target (30-50% width) is far milder than anything SlimMoE tested; the collapse they observe at one-stage pruning happens at 85% removal, not 30%.
- Their "Prune-Expert-KL" baseline is a KL-reconstruction expert-deletion criterion, not REAP; REAP's own paper argues it beats earlier deletion criteria, so SlimMoE's deletion baseline is probably weaker than our REAP pipeline. The true training-free gap vs REAP is plausibly smaller than 10 points. **[inference]**

### Gaps
- Which benchmark Table 4 reports, and whether the numbers are before or after distillation, could not be confirmed from the HTML extract (the table caption was not captured). Recommend reading Table 4's caption in the PDF.
- No source reproduces the deletion-vs-slimming comparison training-free on any model larger than ~80B.

---

## KQ2. Training-free structured width pruning per expert: methods, criteria, measured results at 25-60% with no fine-tuning

### Takeaway
Three 2025-2026 papers do neuron-level pruning inside MoE experts without weight updates and all beat whole-expert deletion by 5-13 average points at 50-60% expert-parameter reduction: TENP (projection-onto-expert-output criterion, forward pass only, 128 samples/domain) keeps Qwen1.5-MoE at 59.06 vs 61.16 full and DeepSeek-V2-Lite at 51.98 vs 52.98 at 60% routed-expert sparsity, where whole-expert deletion (EASY-EP) gets 48.02 / 46.18; DERN (delete experts, recombine their neurons into survivors) gets Qwen2-57B 32/64 to 83.56 vs 70.71 for deletion. FlexMoE reaches 99.8% of Qwen2-57B at 50% expert-channel removal but requires gradient-trained per-expert width masks. The criteria all need per-expert intermediate activations from a calibration forward pass, i.e. the same pass REAP already runs.

### Cited Findings
**TENP (arXiv 2606.09885) [training-free; no fine-tuning step described]**
- Models: Qwen1.5-MoE-A2.7B (60 routed + 4 shared, top-4), DeepSeek-V2-Lite (64 routed + 2 shared, top-6), Qwen3-Next-80B-A3B-Instruct (appendix, 60% sparsity) — [TENP](https://arxiv.org/html/2606.09885v1)
- Two stages: (1) expert importance = routing-weighted output norm x (1 - cos(input, output)), summed over tokens, L2-normalised across domains; top 20-30% experts kept intact; (2) remaining experts are pruned at the *neuron* level: each neuron scored by the projection of its output onto the expert's full output averaged over tokens; top-K neurons kept; matching rows/cols of up, gate, down removed; router untouched. Trapezoidal allocation: shallow layers compressed most, deep layers keep most capacity — [TENP](https://arxiv.org/html/2606.09885v1)
- Calibration: 128 validation samples per domain; one sample already retains most performance, gains plateau at 8 — [TENP](https://arxiv.org/html/2606.09885v1)
- Table 1 averages (GSM8K, MBPP, HumanEval, ARC-E, ARC-C), 60% routed-expert sparsity: Qwen1.5-MoE full 61.16; EASY-EP (whole-expert) 48.02; Gating-score deletion 43.14; ENP (neuron-only) 55.88; TENP 59.06 at 61.38% activated params. DeepSeek-V2-Lite full 52.98; Frequency deletion 41.18; EASY-EP 46.18; TENP 51.98 at 63.76% activated params — [TENP](https://arxiv.org/html/2606.09885v1)
- At 30% sparsity the whole-expert baselines collapse much harder than TENP: Qwen 43.00 (TENP) vs 21.14 (EASY-EP); DeepSeek 39.72 vs 15.26 (these "30%" rows look inverted relative to the 60% rows; the extract flags the abstract/table labelling as inconsistent) — [TENP](https://arxiv.org/html/2606.09885v1)
- Ablation: projection-based scoring (ENP-COS 51.00) beats plain L2-norm neuron scoring (ENP-L2 48.65) — [TENP](https://arxiv.org/html/2606.09885v1)
- Generalisation (MMLU, no MMLU calibration): TENP 54.81 vs gating-score deletion 49.13, EASY-EP 47.49, full 61.05 — [TENP](https://arxiv.org/html/2606.09885v1)
- Qwen3-Next-80B at 60% sparsity: TENP 87.50 vs full 88.64 vs EASY-EP 87.12 (benchmark set for this appendix table not captured) — [TENP](https://arxiv.org/html/2606.09885v1)
- Throughput at 50% params (Qwen): neuron pruning 1.47x vs whole-expert pruning 1.41x total throughput; static memory 32.95 -> 16.37 GB — [TENP](https://arxiv.org/html/2606.09885v1)

**DERN (EMNLP 2025 Findings, arXiv 2509.10377) [training-free]**
- Drops experts by a routing-based importance (selection frequency x activation strength), then splits dropped experts into neuron "segments" (gate row + up row + down column), assigns each to the most cosine-similar retained expert above threshold alpha, and re-clusters each retained expert's segments with spherical weighted k-means to a target hidden dimension — i.e. survivors' intermediate size *changes* — [DERN](https://arxiv.org/html/2509.10377)
- Calibration: 128 x 2048-token C4 sequences; baselines NAEE, M-SMoE, LLM-Pruner (REAP not compared) — [DERN](https://arxiv.org/html/2509.10377)
- Average (MMLU in parentheses): Mixtral 4/8: DERN 70.79 (54.91) vs NAEE 65.69 (51.89); Mixtral 6/8: 80.04 vs 77.25 (dense 80.88). Qwen2-57B 32/64: 83.56 (71.17) vs 70.71 (59.61). DeepSeek-MoE-16B 48/64: 55.50 (44.17) vs 46.67 (34.60) — [DERN](https://arxiv.org/html/2509.10377)
- A third-party reviewer notes the >5% margin may be optimistic because the mixing hyperparameter was tuned on the evaluation benchmarks — [search summary citing Pith review](https://pith.science/paper/2509.10377)

**FlexMoE (arXiv 2606.27866) [mask training only; expert weights frozen unless the optional LoRA recovery is used]**
- Channel group = matching gate row + up row + down column; saliency = first-order Taylor score sum of (theta x dL/dtheta)^2 over the group, averaged over Zyda-2 calibration batches; 100 samples give most of the gain, 5k add little — [FlexMoE](https://arxiv.org/html/2606.27866v1)
- Per-expert retention from a *discrete* action set {0.1, 0.4, 0.7, 1.0}, chosen by Gumbel-softmax-trained logits (48 h Mixtral, 30 h Phi-3.5-MoE, 7 h Qwen2 on 2xH200) — [FlexMoE](https://arxiv.org/html/2606.27866v1)
- Qwen2-57B-A14B (base 58.71 avg over 7 zero-shot tasks), **no recovery fine-tuning**: 20% 58.43; 40% 58.86; 50% 58.57 (99.8%); 60% 58.57; 80% 54.57 (92.9%). Whole-expert NAEE: 20% 55.86; 40% 53.14; 60% 44.00 — [FlexMoE](https://arxiv.org/html/2606.27866v1)
- Mixtral (63.29), with a one-off LoRA recovery at the 40% mask: 20% 61.86 vs NAEE 57.71; 40% 58.00 vs 48.71; 60% 49.57 vs 36.86. Phi-3.5-MoE (62.00): 40% 59.43 vs NAEE 57.57 — [FlexMoE](https://arxiv.org/html/2606.27866v1)

**Other training-free width work (dense baselines, transferable criteria)**
- FLAP: retraining-free structured pruning that "significantly outperforms" LLM-Pruner and the structured extension of Wanda, evaluated on dense LLaMA models only — [FLAP via SEAP search](https://arxiv.org/pdf/2503.07605); Wanda-sp scores whole weight groups with the l2 norm of (weight x activation) groups — [SEAP](https://arxiv.org/pdf/2503.07605)
- Minitron: activation-based importance needs only forward passes over a 1024-sample calibration set; aggregation choice matters (batch=L2, seq=mean gave zero-shot LM loss 7.18 vs 8.73 for L2/L2); iterative importance gives no lasting benefit after retraining — [Minitron](https://arxiv.org/html/2407.14679)
- SlimQwen width pruning of the *hidden* size (2048 -> 1536) uses mean absolute RMSNorm-output activation averaged over batch/seq/layers; it does **not** slim expert intermediate size (stays 512) — [SlimQwen](https://arxiv.org/html/2605.08738)
- MoE-Pruner (2410.12013) is unstructured (per-output-neuron weight x activation x router weight), one-shot — [MoE-Pruner](https://arxiv.org/pdf/2410.12013)

### Inferences
- Criterion choice: TENP's "projection of neuron output onto expert output" (forward-only) beat plain activation-norm by 2.35 points; FlexMoE's Taylor score needs gradients. For a forward-pass-only pipeline the TENP criterion is the strongest measured option, and it needs exactly the per-token per-expert intermediate activations (act(gate)*up, and the down-proj column contribution) that a REAP calibration pass already touches — compute per neuron j in expert e: s_j = mean_t [ h_j(t) * < W2[:,j], y_e(t) > / ||y_e(t)|| ] (reconstructed from the text; verify against TENP eq.). **[inference]**
- The whole-expert baselines in these papers (EASY-EP, NAEE, frequency) are generally weaker than REAP. The deletion-vs-slimming margin against REAP is likely several points smaller than the 11-13 points shown vs EASY-EP/NAEE. **[inference]**
- Every training-free slimming result lives on 14-80B models with 8-64 experts (fine-grained Qwen3-Next-80B is the only 500-expert case, at 60% with a near-saturated benchmark). Extrapolating to a 700B-1T model with 256-512 experts of width 2048-2688 is unmeasured. **[inference]**

### Gaps
- No training-free comparison of per-expert width pruning against REAP specifically.
- No per-expert width-pruning result on DeepSeek-V3/GLM-5-class sigmoid-router models (TENP's DeepSeek-V2-Lite is softmax).
- FlexMoE's striking Qwen2 numbers (99.8% at 50%) depend on learned non-uniform widths; a uniform training-free 50% slim on the same model is not reported.

---

## KQ3. Width vs depth vs experts: which axis is cheapest in quality per byte for MoE?

### Takeaway
No paper runs a clean three-way comparison. What the large-scale pipelines actually chose is informative: SlimMoE (Phi-3.5-MoE) slimmed width and touched neither experts nor depth; Nemotron Puzzle (120.7B -> 75.3B) slimmed expert width (mean 59.9%) and halved top-k per layer but kept every expert and every layer; SlimQwen (80B -> 23B) deleted/merged 50% of experts, shrank hidden size, dropped 25% of layers, and left expert width untouched. Minitron's dense finding (width > depth after retraining; depth pruning looks better *before* retraining then flips) and SlimQwen's one-shot depth numbers (last-layer pruning: MMLU -1.8 but GSM8K 82 -> 2) show depth is the worst axis for generative/reasoning quality before healing.

### Cited Findings
- Nemotron-Labs-3-Puzzle-75B-A9B: routed expert count unchanged; routed expert intermediate 2688 -> 1280-2688 (mean 59.9%); top-k 22 -> 4-18 (mean 50%); active routed-expert capacity 8.7-62.3% (mean 30.9%); total 120.7B -> 75.3B, active 12.8B -> 9.3B; attention untouched; depth untouched; Mamba state 128 -> 96 — [Puzzle](https://arxiv.org/html/2607.04371v2)
- Puzzle channel ranking: "based on their contribution to the expert's output", the same contribution criterion Puzzle uses for dense FFN pruning; per-layer allocation via mixed-integer program; no pre-distillation accuracy reported; ~120B tokens of KD (24B + 43.2B + 52.8B), with a separate figure of up to 100B tokens per sequence-length stage that the paper does not reconcile — [Puzzle](https://arxiv.org/html/2607.04371v2)
- Puzzle quality (BF16, child vs parent): MMLU-Pro 82.4 vs 83.8; GPQA 78.6 vs 80.5; LiveCodeBench 81.1 vs 82.1; SWE-Bench 56.9 vs 59.5; Arena-Hard-V2 68.6 vs 72.8; RULER 1M 92.2 vs 93.9; the paper says >97% of parent across most categories; ~2x server throughput at matched user throughput on 8xB200 — [Puzzle](https://arxiv.org/html/2607.04371v2)
- Puzzle iterative (3-step compress-and-heal) vs single-step: 69.05 vs 68.48 average — [Puzzle](https://arxiv.org/html/2607.04371v2)
- SlimQwen 80A3B -> 23A2B: hidden 2048 -> 1536; routed experts 512 -> 256 (prune or merge); top-k 10 -> 8; expert intermediate 512 unchanged; 12 of 48 layers removed — [SlimQwen](https://arxiv.org/html/2605.08738)
- SlimQwen one-shot depth pruning (15A2B, 4 of 24 layers): last-layer pruning MMLU 73.86 vs teacher 75.62 but GSM8K 2.05 vs 82.41; activation-similarity selection MMLU 41.95, GSM8K 11.22. After 120B tokens: 73.02 / 77.86 (last-layer) — [SlimQwen](https://arxiv.org/html/2605.08738)
- SlimQwen expert compression (24A2B -> 6A1B, after 400B tokens): all one-shot criteria converge to MMLU 68.7-69.7 (REAP prune 69.11, soft-logit prune 68.74, partial-preservation merges 68.7-69.7); the authors conclude one-shot expert compression differences wash out after continued pretraining — [SlimQwen](https://arxiv.org/html/2605.08738)
- SlimQwen: pruned init + KD (120B tokens) reaches 73.45 avg vs 61.66 for random init + KD, i.e. ~86.5% of the teacher's 82.68 — [SlimQwen](https://arxiv.org/html/2605.08738)
- Minitron (dense, <=15B): width-only pruning reached 7.74B at LM loss 2.049 vs depth-only 9.39B at 2.155 after ~1.8B retraining tokens; depth+width had lower loss *before* retraining but the ranking flipped by ~200 steps; best practice "prefer width over depth for model scales <=15B" — [Minitron](https://arxiv.org/html/2407.14679)
- NVIDIA ModelOpt's `mcore_minitron` can prune "MoE experts and number of layers" plus MLP FFN hidden size, heads, GQA groups, Mamba dims, using activation magnitudes; 0.45.0 added `active_params` (MoE) and `memory_mb` constraints; a `gpt_oss_pruned_to_mxfp4` module exists under Puzzletron with helpers `convert_moe_packed_tensors` / `copy_non_moe_weights` but no usage docs — [ModelOpt pruning guide](https://nvidia.github.io/Model-Optimizer/guides/3_pruning.html); [ModelOpt 0.45.0 notes](https://newreleases.io/project/github/NVIDIA/Model-Optimizer/release/0.45.0); [gpt_oss_pruned_to_mxfp4 API stub](https://nvidia.github.io/Model-Optimizer/pr-preview/pr-1190/_sources/reference/generated/modelopt.torch.puzzletron.anymodel.models.gpt_oss.gpt_oss_pruned_to_mxfp4.rst.txt)

### Inferences
- The two NVIDIA-lineage pipelines (SlimMoE, Puzzle) chose width+top-k over expert deletion whenever they had a distillation budget; the Qwen pipeline (SlimQwen) chose expert deletion/merging and left width alone. Both camps had >=120B tokens, so the choice reflects engineering preference and target shape, not a measured per-byte quality ranking. **[inference]**
- Puzzle's per-layer allocation (some layers at 8.7% active capacity, others at 62%) is the same lever as TENP's trapezoid and the layer-sensitivity findings in KQ8; uniform slimming is the wrong default. **[inference]**
- For the bytes-first Thor problem the ranking by *training-free* damage is: depth (catastrophic on GSM8K one-shot) > hidden-size width (unmeasured one-shot, needs re-projection of every layer) > whole-expert deletion (REAP, ~5% at 50%, see other notes) ~ expert-width slimming (TENP/FlexMoE, smaller gaps at 50-60%). **[inference]**

### Gaps
- No three-way (width / depth / expert-count) comparison at equal bytes exists in any source found.
- Puzzle reports no pre-distillation accuracy at all, so the training-free value of width slimming at 60% is unknown there.
- ModelOpt's gpt-oss MXFP4 pruning path has no documentation beyond a module name; whether it prunes in bf16 and re-packs to MXFP4 (likely, given `convert_moe_packed_tensors`) is unconfirmed.

---

## KQ4. Quantized-weight surgery: removing intermediate channels on MXFP4 / NVFP4 / GGUF experts in block multiples

### Takeaway
Block scales run along the *input* (K) dimension of each projection. For gate_proj/up_proj (K = hidden), an intermediate channel is a whole *row* of the packed tensor and of the scale tensor, so any subset of channels can be removed exactly, no requantisation. For down_proj (K = intermediate), the blocks run *along* the dimension being pruned: removal is exact only if the pruned channels form whole aligned blocks (32 elements = 16 packed bytes + 1 E8M0 scale for MXFP4; 16 elements = 8 bytes + 1 E4M3 scale for NVFP4; 256 for K-quant/IQ GGUF, 32 for Q4_0/Q8_0/IQ4_NL). Arbitrary channel choice forces re-blocking, hence requantising, down_proj. Per-tensor FP32 global scales (NVFP4 `weight_scale_2`, `input_scale`) stay valid because deleting elements can only lower the amax. The practical answer is to score channels in groups of 16/32/256 and remove whole groups, or to prune before quantising (which is the order our REAP pipeline already uses).

### Cited Findings
- vLLM MXFP4 MoE weight shapes (N experts, I intermediate, H hidden): `w13_weight (N, 2I, H//2) uint8`, `w13_weight_scale (N, 2I, H//32) uint8`; `w2_weight (N, H, I//2) uint8`, `w2_weight_scale (N, H, I//32) uint8`; the 32-element block runs along H for w13 and along I for w2; `create_weights` asserts no divisibility, but `maybe_roundup_sizes` -> `mxfp4_round_up_hidden_size_and_intermediate_size` rounds sizes up per backend, and a comment says only the SITU path handles "non-256-aligned" intermediate sizes without prior round-up; Triton/AITER backends store *swizzled* scales in precision configs — [vLLM mxfp4 docs](https://docs.vllm.ai/en/latest/api/vllm/model_executor/layers/quantization/mxfp4/)
- gpt-oss checkpoints name the tensors `gate_up_proj_blocks` / `down_proj_blocks` and `_scales`; vLLM maps them to w13/w2; NVIDIA Megatron Bridge notes the MXFP4 checkpoint orders gate/up differently from BF16 and stores down_proj transposed vs Megatron — [vLLM gpt_oss loader](https://docs.vllm.ai/en/v0.11.1/api/vllm/model_executor/models/gpt_oss/); [Megatron Bridge gpt_oss bridge](https://docs.nvidia.com/nemo/megatron-bridge/0.4.1/apidocs/bridge/bridge.models.gpt_oss.gpt_oss_bridge.html)
- NVFP4: 16-element blocks with an E4M3 block scale plus an FP32 per-tensor global scale; Transformer Engine rowwise layout is data [A, B] with scales [A, B/16] (block along the last/input dim), scale tensors padded to multiples of 128 x 4 for hardware alignment; TE optionally uses 2D 16x16 weight scaling for training, which is *not* the inference checkpoint format — [TE NVFP4 docs](https://nvidia.github.io/TransformerEngine/features/low_precision_training/nvfp4/nvfp4.html); [packet.ai NVFP4 summary](https://packet.ai/blog/nvfp4)
- ModelOpt NVFP4 checkpoint layout (third-party conversion write-up for DeepSeek-V4-Flash): packed `weight` uint8 (out, in/2); `weight_scale` E4M3 (out, in/16); `weight_scale_2` and `input_scale` per-tensor FP32 scalars; config group size 16 — [friendli.ai DSV4-Flash NVFP4 notes](https://friendli.ai/models/Rarri/DeepSeek-V4-Flash-0731-NVFP4); llm-compressor maps ModelOpt `weight_scale_2` -> `weight_global_scale` and `input_scale` -> `input_global_scale` (inverted) — [llm-compressor convert guide](https://docs.vllm.ai/projects/llm-compressor/en/latest/guides/entrypoints/convert/)
- llama.cpp quantisation fallback rule (local checkout at commit 465e49b9c, 2026-09-07, `src/llama-quant.cpp` `tensor_type_fallback`): the check is `ncols = t->ne[0]; if (ncols % ggml_blck_size(target_type) != 0)` then IQ1/IQ2/IQ3/IQ4_XS fall back to IQ4_NL, Q2_K/Q3_K -> Q4_0, Q4_K -> Q5_0, Q5_K -> Q5_1, Q6_K -> Q8_0, and if still not divisible by 32, F16 — [llama.cpp llama-quant.cpp](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-quant.cpp) (verified locally: /home/patrickd/llama.cpp/src/llama-quant.cpp lines 371-424)
- llama.cpp expert FFN width is a *per-layer* hparam array `n_ff_exp_arr[LLAMA_MAX_LAYERS]` accessed via `hparams.n_ff_exp(il)`, and the fused gate/up tensor is asserted to be exactly `2*n_ff_exp` along its axis (`GGML_ASSERT(tensor->ne[axis] == 2*n_ff_exp)`) — [llama.cpp llama-model.cpp](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-model.cpp) (verified locally: /home/patrickd/llama.cpp/src/llama-model.cpp lines 641-666, /home/patrickd/llama.cpp/src/llama-hparams.h line 103)
- GGUF converter writes `moe_intermediate_size` from the HF config as `{arch}.expert_feed_forward_length` and `shared_expert_intermediate_size` as `expert_shared_feed_forward_length` (Qwen2-MoE fix #7835) — [llama.cpp commit a94e6ff](https://cdn04132025.gitlink.org.cn/replica/llama.cpp/commit/a94e6ff8774b7c9f950d9545baf0ce35e8d1ed2f)

### Per-projection feasibility table **[inference from the layouts above]**

| Projection | Shape (out, in) | Block axis | Removing intermediate channel j | Exact without requant? |
|---|---|---|---|---|
| gate_proj (W1) | (I, H) | along H | delete row j of packed data and row j of scales (and bias[j]) | Yes, any subset |
| up_proj (W3) | (I, H) | along H | same as gate; in gpt-oss/vLLM fused w13 the gate and up rows are interleaved or stacked, so delete both rows for j | Yes, any subset |
| down_proj (W2) | (H, I) | along I | j lives inside block floor(j/B); must delete the whole block (B = 32 MXFP4, 16 NVFP4, 256 K-quant/IQ GGUF, 32 Q4_0/Q8_0/IQ4_NL) to delete the 16/8/128... packed bytes and the one scale cleanly | Only if the removed set is a union of aligned blocks; otherwise requantise W2 (re-block + new scales) |
| NVFP4 `weight_scale_2` (per-tensor) | scalar | - | unchanged; amax of the remaining elements <= old amax, so the old global scale still represents every value exactly | Yes (slightly suboptimal headroom) |
| NVFP4 `input_scale` for down_proj | scalar | - | activation is the pruned intermediate vector; its amax cannot grow | Yes |
| MXFP4 E8M0 block scale | per block | - | dropped with the block | Yes |
| GGUF ffn_down_exps | ne[0] = n_ff_exp | along n_ff_exp | same as W2: remove aligned 256-groups for K/IQ types; else llama-quantize falls back (Q4_K -> Q5_0, IQ4_XS -> IQ4_NL), which costs ~0.5-1 bit/weight | Only aligned; otherwise re-quantise from bf16 with imatrix |
| GGUF ffn_gate_exps / ffn_up_exps | ne[0] = n_embd | along n_embd | delete rows along ne[1] | Yes |

### Inferences
- Byte accounting per removed intermediate channel at NVFP4: 2 x (H/2 + H/16) bytes from gate/up + (H/2 + H/16)/... per the block for down; with H = 2048-7168 and 256-512 experts, 30% slimming of every expert removes ~30% of expert bytes, same as deleting 30% of experts, with *no* routing change. **[inference]**
- Do the surgery on the *checkpoint* layout (linear, pre-swizzle), never on vLLM's kernel-format tensors; vLLM re-swizzles at load. **[inference from the swizzle note]**
- Block-granular selection for down_proj: score each aligned block of 16/32 channels by the sum of per-channel saliencies and remove whole blocks. The intermediate index order is arbitrary at training time, so a block is a random bag of channels; block-level pruning therefore loses some selectivity vs channel-level. The loss is bounded: with B = 16 and 30% removal, you keep blocks whose summed score is in the top 70%, which at random channel order is only slightly worse than per-channel. For B = 256 (GGUF K-quants) the granularity is coarse enough that you should prune in bf16 and requantise instead. **[inference]**
- Mixed-granularity alternative: prune gate/up at channel level (exact) but keep down_proj unpruned columns for channels that are not block-aligned - not valid: gate/up removal zeroes h_j, so W2[:, j] contributes nothing and its bytes are dead weight but harmless; this gives the FLOP saving on gate/up only and *no* byte saving on down until the block is fully dead. Not worth it. **[inference]**
- For our GLM-5.3 pipeline (prune bf16 -> quantise, per memory notes), block constraints only matter if we want to *skip* requantising already-shipped NVFP4/MXFP4 experts (gpt-oss, MiMo-V2.6-Flash MXFP4, V4.1-Flash pre-FP4). For those, channel-level on gate/up + block-level on down is the exact path. **[inference]**
- Re-running NVFP4 quantisation of W2 alone after arbitrary channel selection is cheap (per-block amax, no Hessian) but changes the global scale and all block scales; the quality delta of a fresh NVFP4 quantisation vs the shipped one is small but non-zero and un-measurable without the bf16 original. **[inference]**

### Gaps
- vLLM's exact round-up multiples for MXFP4/NVFP4 MoE intermediate sizes per backend (`mxfp4_round_up_hidden_size_and_intermediate_size`) were not visible in the docs; expect 256 for Marlin/CUTLASS paths. Check the function body in the installed vLLM.
- gpt-oss gate/up row interleave order (even = gate, odd = up vs stacked halves) not confirmed from a primary source; verify by reading one expert's tensor before surgery.

---

## KQ5. Composition with whole-expert deletion: any paper that deletes experts and slims survivors; what the error models predict

### Takeaway
Three papers do a two-stage inter-expert + intra-expert compression: TENP (keep 20-30% experts intact, neuron-prune the rest), DERN (delete experts, fold their neurons into survivors), and MoE-I2 (delete 25% experts, then low-rank the survivors ~38% for ~52-54% expert-parameter reduction). None isolates whether the two errors add. MoE-I2's numbers are the only ones that show each stage separately: pruning alone costs 2.0-3.2 points, adding intra-expert low-rank without fine-tuning costs a further 2.0-11.2 points, and the combined loss is larger than the sum of the individual losses would predict from pruning alone, i.e. roughly additive-to-superadditive before healing; after LoRA healing the combination recovers to within 2-3 points of base.

### Cited Findings
- MoE-I2: 25% experts removed (layer-wise genetic search) + intra-expert SVD at ~37.5% (Mixtral) / 38.6% (Qwen, DeepSeek) -> ~51.8% / 54.0% expert-parameter reduction; calibration 2048 C4 samples for pruning, 2048 Alpaca for decomposition; healing = LoRA on ~50k Alpaca samples, 2 epochs — [MoE-I2](https://arxiv.org/html/2411.01016)
- MoE-I2 Table 1 averages (7 zero-shot tasks): Mixtral base 67.53 -> P 65.57 -> P+D 54.33 -> P+D+F 64.55; Qwen1.5-MoE 60.25 -> 57.16 -> 55.13 -> 57.82; DeepSeek-V2-Lite 61.49 -> 58.34 -> 54.65 -> 59.62; P+F alone lifts DeepSeek to 63.86 (above base). The paper does not model error interaction — [MoE-I2](https://arxiv.org/html/2411.01016)
- TENP keeps the top 20-30% experts fully intact and neuron-prunes the remaining 70-80%; the mix is justified by ablation (random expert selection hurts), and the final activated-parameter fraction is 61-64% at "60% sparsity" — [TENP](https://arxiv.org/html/2606.09885v1)
- DERN deletes experts then *adds* neurons from deleted experts to survivors (survivors can grow or shrink to a target width), and attributes its weaker DeepSeek result to low inter-expert similarity making experts less interchangeable — [DERN](https://arxiv.org/html/2509.10377)
- FlexMoE's cost term is weighted by expert routing load, so lightly-routed experts get slimmer; retention logits are per expert with widths from {0.1, 0.4, 0.7, 1.0} — [FlexMoE](https://arxiv.org/html/2606.27866v1)
- SlimQwen: after 400B tokens, prune-vs-merge criteria for experts converge (68.7-69.7 MMLU), implying distillation erases one-shot criterion differences — [SlimQwen](https://arxiv.org/html/2605.08738)

### Inferences (error model) **[inference]**
- Deletion error: for a token whose top-k set included deleted expert e, the token is re-routed to the next-best survivor, so the error is `g_e * (y_e(x) - y_e'(x))` where e' is the replacement; it is concentrated on the subset of tokens that used e (REAP's criterion minimises `g_e * ||y_e||` mass). Routing distribution shifts for every token (renormalisation over a smaller set).
- Slimming error: for every token and every active expert, `y_e(x)` is replaced by `y_e^(slim)(x) = sum_{j in kept} h_j(x) W2[:,j]`; the error is `sum_{j in dropped} h_j(x) W2[:,j]`, small per expert if the dropped channels have small `h_j * ||W2[:,j]||` on calibration data, but it hits *every* token and *every* layer. Routing is unchanged (router sees the residual stream, which is only perturbed downstream).
- The two errors are on different tokens/experts to first order: deletion error is nonzero only on tokens that routed to deleted experts; slimming error is nonzero on all tokens but only through surviving experts. Cross-term: tokens re-routed from a deleted expert now also receive the slim approximation of the replacement expert, and the replacement's channels were scored on *its own* calibration tokens, not on the re-routed ones. So the composition is approximately additive in squared output error, with a positive cross-term proportional to the re-routed token mass. This matches MoE-I2's observed pattern (P+D worse than P by more than D alone would suggest on Mixtral: -11.2 vs -2.0).
- Ordering: delete first, then re-run calibration through the *pruned* model so that survivor channels are scored on the post-deletion routing (including re-routed tokens). This removes the cross-term at the cost of a second forward pass. Scoring both from one pass of the unpruned model is cheaper but leaves the cross-term.
- Budget arithmetic: delete fraction d of experts and slim survivors by fraction s -> expert bytes kept = (1-d)(1-s); d = s = 0.30 gives 0.49 (51% removed); d = 0.25, s = 0.33 gives 0.50. Given the measured curves (REAP ~ -5% at d = 0.5 per our other notes; TENP/FlexMoE ~ -2 to -4% at s = 0.5-0.6 training-free), splitting the budget plausibly beats either axis alone at 50%, since each axis is operating on the flatter part of its curve. Not measured anywhere.
- Allocation across experts: TENP's "keep the most important 20-30% intact, slim the rest" and FlexMoE's load-weighted cost both say the slim fraction should be *non-uniform*: high-mass experts (large REAP score) keep full width, low-mass experts that survive deletion get slimmed hardest. The REAP score already ranks experts for this.

### Gaps
- No paper reports deletion + width slimming at matched bytes vs either alone, training-free, on any model.
- No measurement of the cross-term (re-routed tokens through slimmed survivors).

---

## KQ6. Composition with ExFold-style routing recovery and with LoRA healing (attention/shared expert, experts frozen)

### Takeaway
ExFold (Aug 2026) is a training-free *skipping* method: it executes a constrained expert set and folds excluded experts' contributions into retained ones through a calibrated pairwise scalar-projector matrix, retaining ~99% quality at up to 2.45x TPOT; its abstract does not say experts are removed from memory. Router-only KD ("router calibration", ~0.04% of parameters, ~2 h on Qwen3-30B) improves most benchmarks after REAP/merging on fine-grained MoEs and barely helps on Mixtral; it has not been tested after width slimming. Both are compatible with slimming in principle because neither touches expert weights, but no source measures the combination.

### Cited Findings
- ExFold: "training-free framework"; runs only a "phase-specific constrained expert set"; token-level Top-K folding for prefill, batch-level expert-pool folding for decode; contributions of excluded experts projected onto retained experts via a pairwise scalar-projector matrix calibrated on unlabeled data; vLLM plugin + CUDA kernel; up to 1.41x TTFT and 2.45x TPOT at ~99% of average quality; models and quality-vs-k curves not in the abstract; whether excluded experts are deleted from memory is not stated — [ExFold abstract](https://arxiv.org/abs/2608.24938)
- Router KD: updates only the router (0.04% of Qwen3-30B-A3B-Instruct-2507, 0.002% of Mixtral), distilling the original model's next-token distribution on C4; ~2 h (Qwen3) / ~40 min (Mixtral); applied after REAP, CFES, MoBE, TD-MoE, HC-SMoE, M-SMoE at 37.5% expert removal (plus a 25% setting on Qwen3); "improved most Qwen3 benchmarks across all six methods" with marginal Mixtral gains; width pruning not tested; exact numbers are in figures/appendix not captured — [Router calibration](https://arxiv.org/html/2603.02217v1)
- FlexMoE's optional recovery is a single LoRA run on the frozen ranked base with CE + KL distillation, then merged; it lifts Mixtral at 40% from (unreported) to 58.00 and is reused at other budgets — [FlexMoE](https://arxiv.org/html/2606.27866v1)
- MoE-I2 LoRA healing on ~50k Alpaca samples recovers Mixtral P+D from 54.33 to 64.55 (base 67.53) — [MoE-I2](https://arxiv.org/html/2411.01016)
- SlimMoE and Puzzle both do full-model distillation (400B and ~120B tokens) rather than LoRA; Puzzle's "heal" is between iterative compression rounds — [SlimMoE](https://arxiv.org/html/2506.18349); [Puzzle](https://arxiv.org/html/2607.04371v2)

### Inferences **[inference]**
- Slimming leaves the router untouched but changes every expert's output slightly; router-only KD should be *more* useful after slimming than after deletion-only, because with slimming all experts drift and the router's gate values (not just its top-k set) can be re-tuned to compensate; this is untested.
- ExFold's folding matrix assumes a retained expert can absorb another's contribution via a scalar projector; slimmed experts have a reduced output subspace, so the calibrated projectors should be recomputed on the slimmed model. Cost: one calibration pass, no training.
- LoRA on attention + shared expert with routed experts frozen: the shared expert sees every token and is a natural place to absorb the mean slimming residual (sum over dropped channels), which is a dense, token-dependent function the shared expert can partially learn. This is the cheapest healing target that does not require touching the quantized routed experts, but there is no measurement of how much of the slimming error it recovers.

### Gaps
- No source combines width slimming with router-only KD, ExFold, or attention/shared-expert LoRA.
- ExFold full text (method details, models, whether weights are dropped) not retrieved; only the abstract.

---

## KQ7. Runtime support: llama.cpp / vLLM / SGLang with non-standard expert intermediate size

### Takeaway
A *uniform* per-layer slim is a config-only change in all three stacks: llama.cpp reads `expert_feed_forward_length` into a per-layer array `n_ff_exp_arr` (so per-layer varying widths are also supported), vLLM reads `moe_intermediate_size` and sizes fused-MoE weights from it, and SGLang served FlexMoE's static pruned checkpoints with 1.15-1.75x throughput gains. Per-*expert* varying widths within a layer are not supported by any stock fused-expert kernel: FlexMoE measured that naive per-expert slicing *slows* Mixtral to 0.85-0.94x and had to write a width-bucketed batched-GEMM kernel to get 1.12-1.47x. The real constraints are quantisation block alignment (llama.cpp demotes any tensor whose ne[0] is not a multiple of the type's block size; vLLM MXFP4 rounds intermediate up to 256 on most backends) and keeping the width a multiple of the kernel's tile.

### Cited Findings
- llama.cpp: `hparams.n_ff_exp(il)` per-layer array; the fused gate/up expert tensor must be exactly `2*n_ff_exp` on its axis; converter writes the value from HF `moe_intermediate_size` — local checkout /home/patrickd/llama.cpp (llama-hparams.h:103, llama-model.cpp:641-666) and [commit a94e6ff](https://cdn04132025.gitlink.org.cn/replica/llama.cpp/commit/a94e6ff8774b7c9f950d9545baf0ce35e8d1ed2f)
- llama.cpp quantise fallback when `ne[0] % blck_size != 0` (256 for K/IQ types): Q4_K -> Q5_0, Q6_K -> Q8_0, IQ4_XS -> IQ4_NL etc.; only `ffn_down_exps` has ne[0] = n_ff_exp — local /home/patrickd/llama.cpp/src/llama-quant.cpp lines 371-424 ([upstream](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-quant.cpp))
- vLLM MXFP4 MoE: weight tensors are allocated from `intermediate_size_per_partition`; `maybe_roundup_sizes` rounds hidden/intermediate per backend; SITU path tolerates non-256-aligned intermediate; FlashInfer TRTLLM backends have padding behaviour; a Gemma-4 PR shows the fused_moe tuning path reads `moe_intermediate_size` from the config and breaks when a model names it differently — [vLLM mxfp4](https://docs.vllm.ai/en/latest/api/vllm/model_executor/layers/quantization/mxfp4/); [vLLM PR #40181](https://github.com/vllm-project/vllm/pull/40181)
- FlexMoE on SGLang (H200): static pruned checkpoints: Mixtral 1.27x / 1.56x / 1.75x at 20/40/60%; Phi-3.5-MoE 1.15x / 1.70x / 1.73x; Qwen2-57B 3.18x / 4.52x / 4.57x (base 1228 tok/s). Online per-expert clipping in Python: 0.94x / 0.91x / 0.85x (slower than unpruned); with their custom width-bucketed kernel 1.12x / 1.25x / 1.47x. Causes: variable widths break shape-regular batched GEMMs (degrade to per-expert GEMMs); concatenated gate-up layout forces extra slices; fixes were bucketing experts by width, rounding each width up to a hardware-friendly size, interleaved gate-up layout, fused gating on the smaller activation — [FlexMoE](https://arxiv.org/html/2606.27866v1)
- TENP: neuron-pruned Qwen at 50% params gives 1.47x total throughput vs 1.41x for expert pruning at the same param count — [TENP](https://arxiv.org/html/2606.09885v1)
- vLLM fused_moe Triton tuning configs are keyed by (E, N = intermediate, dtype); an untuned N falls back to default block configs — [vLLM fused_moe layer docs](https://docs.vllm.ai/en/stable/api/vllm/model_executor/layers/fused_moe/layer/) (keying inferred from the Gemma-4 PR's need for per-model handlers)

### Inferences **[inference]**
- For Thor (our own CUDA server for GLM-5.3, plus llama.cpp/vLLM fallbacks): choose the slimmed width as a multiple of 256 (satisfies K-quant superblocks, vLLM MXFP4/NVFP4 round-up, and typical 64/128-wide GEMM tiles simultaneously). GLM-5.3-Flash expert width 1536 -> 1024 (-33%) or 1280 (-17%); a 2048-wide expert -> 1536 (-25%) or 1024 (-50%); Puzzle's 2688 -> 1280 is itself a 256-multiple.
- Per-layer widths are free in llama.cpp (array hparam) and in our own server; vLLM/SGLang HF configs are a single `moe_intermediate_size`, so per-layer widths need a model-class patch there. Per-expert widths within a layer: avoid; pad to a bucket or treat as not supported.
- Decode on Thor is weight-bandwidth-bound on the *active* experts (memory notes: activation traffic and weight bytes). Slimming by s cuts active-expert weight bytes per token by s *and* the intermediate activation vector by s; deletion cuts neither per token. So slimming is the only one of the two that speeds decode, consistent with TENP's 1.47x vs 1.41x.

### Gaps
- No llama.cpp or vLLM GitHub issue found that discusses pruned expert widths specifically; the config-only claim rests on reading the loaders.
- vLLM's exact per-backend round-up multiples not confirmed.

---

## KQ8. Top-k reduction: measured quality vs k, with/without renormalisation, and interaction with expert deletion

### Takeaway
Naive top-k halving costs 2-5 MMLU points and 4-7 GSM8K points on Qwen3.5/3.6 (and 8.5% GSM8K / 11.6% HumanEval on an older Qwen MoE); keeping the renormalisation denominator at the *trained* k (or 2k) while activating k/2 shrinks the loss to -0.35 to -0.8 points, with no weight change and no bytes saved. DeepSeek-V2-Lite tolerates ~3.3 of 6 experts within 1%; DeepSeek-V3-style sigmoid routers with polarised weights are reported as more fragile, and GLM-style `routed_scaling_factor` is a fixed gain that would need the same recalibration. No source measures whether a lower k lets you delete more experts; the mechanics argue the opposite: a lower k reduces the chance a deleted expert's slot is backfilled by a good substitute.

### Cited Findings
- Halving activated experts training-free (arXiv 2609.04575): w_i = p_i / sum_{j in T_k2} p_j for i in T_k1; k2 = k1 is naive renorm, k2 = E is no renorm, best k2 in [k, 2k]. Qwen3.6-35B-A3B (k = 8, 256 experts), MMLU 5-shot: native 81.65; 4/4 naive 77.00 (-4.65, p = 1.7e-9); 4/16 81.30 (-0.35, n.s.); 6/8 80.55 (-1.10); 6/16 82.40 (+0.75); 6/256 (no renorm) 54.20 (-27.45). Qwen3.5-397B-A17B (k = 10, 512 experts): native 89.45; 5/5 87.35 (-2.10); 5/10 88.90 (-0.55, n.s.); 8/8 89.60; 5/512 70.90 — [Halving](https://arxiv.org/html/2609.04575)
- GSM8K 0-shot: Qwen3.6 95.00 -> 4/4 88.40 (-6.60, 10.4% truncation) -> 4/16 94.40 (-0.60); Qwen3.5-397B 96.60 -> 5/5 92.40 -> 5/10 95.80 (-0.80). WikiText ppl Qwen3.6: 6/6 +4.08%, 6/8 -1.61%, no renorm +356%. Perplexity and MMLU pick different k2 (k vs 2k). C-Eval on 397B: 8/8 -2.92, 8/10 -1.00, 5/10 -0.77 — [Halving](https://arxiv.org/html/2609.04575)
- Compute: routed-expert compute halved; activated expert params 1.132B -> 0.629B on the 35B (-44% of the expert term, ~-17% end-to-end of ~3.0B/token); memory unchanged; no wall-clock reported; GLM-style configs ship `routed_scaling_factor` as a fixed post-renorm scalar; DeepSeek/GLM not tested; per-layer k2 left as future work; cross-domain routing divergence grows with depth (0.135 -> 0.338 nats, layer 0 -> 39) — [Halving](https://arxiv.org/html/2609.04575)
- "Faster MoE LLM Inference for Extremely Large Models" (DeepSeek-V2-Lite, softmax): reducing to 2 active experts -> ~-7.5% avg (best config -6%); keeping an average of 3.3 experts stays within 1%; throughput 479 -> 631 tok/s at concurrency 2 (k 6 -> 2, +32%), +17% at concurrency 512. Explanation: "weights of lower-ranked experts are significantly smaller than the top-ranked expert" in V2-Lite; in V3 "expert weights tend to polarize toward 0 or 1" so skipping an expert whose weight is near 1 can cause a substantial drop, and V3 was less stable across strategies — [Faster MoE](https://arxiv.org/html/2505.03531v1)
- MoBiLE: naively activating fewer experts on Qwen MoE costs ~8.5% GSM8K and ~11.6% HumanEval; a big/little fallback recovers much of it — [MoBiLE](https://arxiv.org/pdf/2510.12357)
- PCoMoE: top-k -> top-(k-1) in a single layer hurts some layers far more than others, and losses compound when applied globally — [PCoMoE](https://arxiv.org/pdf/2609.01024)
- Mixtral-8x7B (coarse, top-2), no training: expert pruning 8 -> 6 -> 4: 67.58 -> 64.22 -> 59.57 avg; dynamic skipping (top-2 -> conditional top-1, threshold = median w_e1/w_e0 on calibration) adds a further -1.2 to -1.7 points for +0.04-0.08x speed; GSM8K is more fragile: 58.61 -> 54.28 with skipping alone, 51.25 -> 47.16 at 6 experts; Instruct 4 experts + skipping 40.33 vs 63.46 — [Not All Experts are Equal](https://arxiv.org/html/2402.14800)
- Alloc-MoE: layer-level DP allocation of activation budget + token-level redistribution; DeepSeek-V2-Lite at half budget 1.15x prefill / 1.34x decode; no accuracy numbers in the abstract — [Alloc-MoE](https://arxiv.org/abs/2604.08133)
- ZEDA (post-trained MoE skips half its experts via self-distillation with zero-output experts): Qwen3-30B-A3B and GLM-4.7-Flash, >50% expert FLOPs removed "at marginal accuracy loss", +6.1 / +4.0 points over the best dynamic baseline, ~1.20x end-to-end; training cost not in the abstract — [ZEDA](https://arxiv.org/abs/2605.18643)
- Puzzle reduced top-k from 22 to 4-18 per layer (mean 50%) alongside width slimming, with ~120B tokens of KD, as part of a >97%-of-parent model — [Puzzle](https://arxiv.org/html/2607.04371v2)

### Inferences **[inference]**
- The mechanism in the halving paper is a *gain* miscalibration: with k' < k the renormalised weights sum to 1 over fewer experts, so each surviving expert's contribution is inflated by roughly sum(top-k)/sum(top-k'). The same inflation happens after *expert deletion* with renormalisation (the deleted expert's mass is redistributed), so the k2 trick is directly relevant to REAP: after deleting experts, renormalise over the original top-k *of the unpruned router* (or keep the pre-deletion denominator) instead of over the surviving top-k. Cheap to test on our GLM-5.3 REAP50 model.
- For sigmoid routers (DeepSeek-V3, GLM-4.5/5 with `norm_topk_prob` + `routed_scaling_factor`), the "denominator" is the sum of sigmoid scores over the selected set times the scaling factor; the analogous fix is to rescale `routed_scaling_factor` by the ratio of mean selected-mass at k' vs k measured on calibration data, i.e. a per-layer scalar. Not measured by any source.
- Top-k reduction and expert deletion compete for the same slack: deletion relies on the (k+1)-th... (k+m)-th experts being decent substitutes; lowering k removes the lowest-weight slots, which are exactly where substitutes land. Expect lower k to allow *fewer* deletions at equal quality, not more. Conversely, lowering k makes slimming *less* damaging per token in absolute terms (fewer slimmed experts summed), but the per-expert approximation error is unchanged.
- Serve-time value on Thor: top-k reduction cuts active-expert weight bytes per token proportionally (the binding term in our decode roofline), so k 8 -> 6 with the k2 fix is a ~25% cut in expert traffic at -0.5 to -1 MMLU with zero bytes saved; it stacks with slimming (further -s on the remaining active bytes).

### Gaps
- No measured top-k-vs-quality curve on DeepSeek-V3/V4, GLM-4.5/5, or Kimi-K2-class sigmoid routers with the renormalisation fix; only Qwen (softmax) and DeepSeek-V2-Lite (softmax).
- No source tests whether lower k changes the deletion budget.
- ZEDA and Alloc-MoE full tables not retrieved.

---

## Summary table of measured numbers (for the report writer)

| Method | Regime | Model | Reduction | Metric | Result vs full | Whole-expert baseline | Source |
|---|---|---|---|---|---|---|---|
| SlimMoE Table 4 | gradient criterion, distillation in loop (budget not captured) | Phi-3.5-MoE | "50%" row | unnamed | 63.76 (slim-KL) | 53.41 (prune-KL) / 48.59 (freq) | [2506.18349](https://arxiv.org/html/2506.18349) |
| SlimMoE | distilled 400B | Phi-3.5-MoE 42B -> 7.6B | width 6400 -> 960 | MMLU | 70.68 vs 78.36 | - | same |
| Puzzle | distilled ~120B | Nemotron-3-Super 120.7B -> 75.3B | width mean 59.9%, top-k mean 50% | MMLU-Pro | 82.4 vs 83.8 | - | [2607.04371](https://arxiv.org/html/2607.04371v2) |
| TENP | training-free | Qwen1.5-MoE-A2.7B | 60% routed-expert params | 5-task avg | 59.06 vs 61.16 | 48.02 (EASY-EP) | [2606.09885](https://arxiv.org/html/2606.09885v1) |
| TENP | training-free | DeepSeek-V2-Lite | 60% | 5-task avg | 51.98 vs 52.98 | 46.18 | same |
| TENP | training-free | Qwen3-Next-80B-A3B | 60% | avg | 87.50 vs 88.64 | 87.12 | same |
| DERN | training-free | Qwen2-57B-A14B | 32/64 experts (neurons recombined) | avg | 83.56 | 70.71 (NAEE) | [2509.10377](https://arxiv.org/html/2509.10377) |
| FlexMoE | mask training, no recovery FT | Qwen2-57B-A14B | 50% expert channels | 7-task avg | 58.57 vs 58.71 | 53.14 (NAEE @40%) | [2606.27866](https://arxiv.org/html/2606.27866v1) |
| MoE-I2 | training-free | Mixtral-8x7B | 25% experts + ~38% low-rank (~52% expert params) | 7-task avg | 54.33 vs 67.53 (65.57 prune-only) | - | [2411.01016](https://arxiv.org/html/2411.01016) |
| MoE-I2 | + LoRA 50k Alpaca | Mixtral-8x7B | same | avg | 64.55 | - | same |
| Not All Experts Equal | training-free | Mixtral-8x7B | 8 -> 4 experts | avg | 59.57 vs 67.58 | (is the baseline) | [2402.14800](https://arxiv.org/html/2402.14800) |
| Halving | training-free | Qwen3.5-397B-A17B | top-10 -> 5, k2 = 10 | MMLU | 88.90 vs 89.45 (naive 87.35) | - | [2609.04575](https://arxiv.org/html/2609.04575) |
| Halving | training-free | Qwen3.6-35B-A3B | top-8 -> 4, k2 = 16 | GSM8K | 94.40 vs 95.00 (naive 88.40) | - | same |
| Faster MoE | training-free | DeepSeek-V2-Lite | avg 3.3 of 6 experts | avg | within 1% | - | [2505.03531](https://arxiv.org/html/2505.03531v1) |
