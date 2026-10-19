#!/usr/bin/env bash
# Build the Gate-K unit test (host CUDA). Run: ./build/gate_units ref/goldens
set -e
cd "$(dirname "$0")/.."

# Gate TOPK_RADIX — the single-CTA radix select (DECODE_LADDER 1.2) must be BIT-IDENTICAL to the warp
# selection sort it replaces, not merely select the same set: sparse_attn sums the selected rows in
# order, so fp32 association makes the ORDER load-bearing. Six distributions including exact ties,
# signed zeros and floor-straddling rows, on all four kernel shapes. Two seconds, no checkpoint.
nvcc -O2 -std=c++17 -gencode arch=compute_110a,code=sm_110a -I include \
  tests/gate_topk_radix.cu -o build/gate_topk_radix
echo "built build/gate_topk_radix"

# Gate TOPK_SMEM_CTX (DECODE_LADDER 1.4) -- the ENGINE'S decode step above the dynamic-shared-memory
# ceiling. gate_topk_radix launches the scan kernels directly; this drives
# `compressed_decode_step_indexer` itself at context 49,207 and 200,003, over pre-filled caches and
# no checkpoint, and reproduces the pre-fix defect in the same binary via DSV4_TOPK_SMEM_OPTIN=0.
# Driver: scripts/gate_topk_smem_ctx.sh. A gate that lives only in a comment rots; this is why it is
# in the build script.
nvcc -O2 -std=c++17 -gencode arch=compute_110a,code=sm_110a -I include \
  tests/gate_topk_smem_ctx.cu kernels/compressed_decode.cu kernels/compressed_attn.cu \
  kernels/compressor.cu kernels/indexer.cu kernels/mla_attn.cu kernels/fp8_block_gemm.cu \
  kernels/tc_fp8_gemm.cu kernels/dscratch.cu kernels/dprof.cu kernels/nvfp4_dense.cu \
  -o build/gate_topk_smem_ctx
echo "built build/gate_topk_smem_ctx"

# Gate SDPA_SMEM (DECODE_LADDER 1.4) -- the third context-sized dynamic-shared launch, on the one
# path in kernels/ that nothing links (tests/test_attention.cu is its only caller and its golden
# dirs are absent). Fixing an unexercised path and calling it fixed is how a wiki page becomes
# confidently wrong, so the fix gets a leg: below the ceiling and above it, same finite output.
nvcc -O2 -std=c++17 -gencode arch=compute_110a,code=sm_110a -I include \
  tests/gate_sdpa_smem.cu kernels/attention.cu -o build/gate_sdpa_smem
echo "built build/gate_sdpa_smem"

# Gate INDEX_SCORE (DECODE_LADDER 1.5) -- `index_score` now has FOUR implementations and TWO
# separate bit-exactness claims: IXS_TILED == IXS_WARP (memory placement only) and IXS_GEMM ==
# IXS_SCALAR (the reference order the warp kernel walked away from in Finding 68). gate_units next
# door is a COSINE gate against the golden and would pass a reduction-order change; this is memcmp,
# and it also PRINTS the deviation IXS_GEMM spends against the shipped warp kernel per input
# distribution, so the number behind the LOSSLESS gate is measured and not asserted.
nvcc -O2 -std=c++17 -gencode arch=compute_110a,code=sm_110a -I include \
  tests/gate_index_score.cu kernels/indexer.cu kernels/compressor.cu kernels/mla_attn.cu \
  kernels/fp8_block_gemm.cu kernels/tc_fp8_gemm.cu kernels/dscratch.cu kernels/nvfp4_dense.cu \
  -o build/gate_index_score
echo "built build/gate_index_score"

# `kernels/nvfp4_dense.cu` ADDED HERE 2026-08-20 (DECODE_LADDER 1.5). It was missing, so gate_units
# had not LINKED since nvfp4 landed -- and because this line is a bare `nvcc` under `set -e`, the
# whole script died here and NONE of the twelve gates below it rebuilt either. That is the failure
# mode the "THE OTHER EIGHT" block at the bottom of this file was written to prevent, reappearing
# above the block it was written in: `bg` records failures and keeps going, the raw `nvcc` lines up
# here do not. Found by 1.5 running the script, not by reading it.
nvcc -O2 -std=c++17 -gencode arch=compute_110a,code=sm_110a -I include \
  tests/gate_units.cu kernels/fp8_block_gemm.cu kernels/hc_sinkhorn.cu kernels/mla_attn.cu kernels/moe.cu kernels/hc.cu kernels/compressor.cu kernels/indexer.cu kernels/tc_moe_gemm.cu kernels/tc_fp8_gemm.cu kernels/dscratch.cu kernels/dprof.cu kernels/nvfp4_dense.cu \
  -o build/gate_units
echo "built build/gate_units"
# CPU-only gates (no GPU): chat encoder byte-exactness + OpenAI API shaping.
g++ -O1 -std=c++17 -I include tests/gate_encoding.cpp -o build/gate_encoding && echo "built build/gate_encoding"
nvcc -O3 -std=c++17 -gencode arch=compute_110a,code=sm_110a -I include tests/gate_bf16w.cu kernels/compressor.cu kernels/dscratch.cu kernels/mla_attn.cu kernels/indexer.cu kernels/fp8_block_gemm.cu kernels/tc_fp8_gemm.cu kernels/nvfp4_dense.cu -o build/gate_bf16w && echo "built build/gate_bf16w"
g++ -O1 -std=c++17 -I include tests/gate_api.cpp      -o build/gate_api      && echo "built build/gate_api"
# Gate SUFFIX_DRAFT — the S6 candidate drafter's matcher, used by the DSV4_SUFFIXPROBE
# counterfactual in src/decode.cu. That probe prices a lever from ONE checkpoint load, so a matcher
# that silently proposes nothing would retire S6 on a bug rather than on a measurement.
g++ -O1 -std=c++17 -I include tests/gate_suffix_draft.cpp -o build/gate_suffix_draft && echo "built build/gate_suffix_draft"
# Gate FORKJOIN_GRAPH — the side-stream fork/join (Finding 55) must survive CUDA-graph capture, 43
# times on one event pair, or the base-AR graph that is worth 1.26x silently stops capturing. Two
# seconds here instead of a 15-minute checkpoint load to find out.
nvcc -O2 -std=c++17 -gencode arch=compute_110a,code=sm_110a -I include \
  tests/gate_forkjoin_graph.cu kernels/dscratch.cu -o build/gate_forkjoin_graph
echo "built build/gate_forkjoin_graph"
# Gate OGGEMV — the M=1 fp8 ogroup GEMV (Finding 35). Not covered by gate_ogroup, which tests the
# f32 bs=8 tensor-core path; this is a different kernel carrying ~22% of the decode step.
nvcc -O2 -std=c++17 -gencode arch=compute_110a,code=sm_110a -I include \
  tests/gate_ogroup_gemv.cu kernels/mla_attn.cu kernels/dscratch.cu kernels/nvfp4_dense.cu -o build/gate_ogroup_gemv
echo "built build/gate_ogroup_gemv"
# Gate TC_FP8_SMEM — the smem-staged FP8 tensor-core GEMM (Finding 41). gate_units checks
# tc_fp8_gemm at ONE (M,N,K); this sweeps every shape and every M the verify path issues, plus the
# N%64 and M%16 tails that the tile mapping gets wrong independently of each other.
nvcc -O2 -std=c++17 -gencode arch=compute_110a,code=sm_110a -I include \
  tests/gate_tc_fp8_smem.cu kernels/fp8_block_gemm.cu kernels/tc_fp8_gemm.cu kernels/dscratch.cu kernels/nvfp4_dense.cu -o build/gate_tc_fp8_smem
echo "built build/gate_tc_fp8_smem"
# Gate PREFILL_LEN — prefix-invariance of the prefill attention chain across prompt LENGTHS, plus a
# drain of the CUDA last-error slot after every stage (Finding 53). Nothing else in the suite varies
# s, which is how an undersized dynamic-shared-memory request survived at T<3 and how a ratio-128
# layer issued five gridDim-0 launches on every prompt this project has ever run. Worth running under
# `compute-sanitizer --tool memcheck` too: that is what found the shared over-read.
nvcc -O2 -std=c++17 -gencode arch=compute_110a,code=sm_110a -I include -lineinfo \
  tests/gate_prefill_len.cu kernels/compressed_decode.cu kernels/compressed_attn.cu kernels/compressor.cu \
  kernels/indexer.cu kernels/mla_attn.cu kernels/mla_forward.cu kernels/mla_decode.cu kernels/hc.cu \
  kernels/hc_sinkhorn.cu kernels/fp8_block_gemm.cu kernels/tc_fp8_gemm.cu kernels/dscratch.cu kernels/dprof.cu \
  kernels/nvfp4_dense.cu -o build/gate_prefill_len
echo "built build/gate_prefill_len"
# Gate MOE_SCAN — the parallel MoE grouping scans (k_moe_prefix_par / k_build_tiles_par) must be
# BIT-IDENTICAL to the <<<1,1>>> kernels they replaced. Equality, not cosine: these are integer
# scans. Sweeps nr below/at/above the 256-thread block width plus empty, exactly-16 and multi-tile
# experts, because per LEVERS.md trap 9 a harness that cannot express the regime confirms itself.
nvcc -O2 -std=c++17 -gencode arch=compute_110a,code=sm_110a -I include \
  tests/gate_moe_scan.cu kernels/moe.cu kernels/tc_moe_gemm.cu kernels/dscratch.cu kernels/dprof.cu \
  kernels/fp8_block_gemm.cu kernels/tc_fp8_gemm.cu kernels/mla_attn.cu kernels/nvfp4_dense.cu -o build/gate_moe_scan
echo "built build/gate_moe_scan"
# Gate TC_FP8_KC — the K-chunked staging (Finding 74) must be BIT-IDENTICAL to KC=1. gate_tc_fp8_smem
# next door is a cosine gate and would pass a reduction-order change; Finding 68 is the reason that
# distinction gets its own binary.
nvcc -O2 -std=c++17 -gencode arch=compute_110a,code=sm_110a -I include \
  tests/gate_tc_fp8_kc.cu kernels/fp8_block_gemm.cu kernels/tc_fp8_gemm.cu kernels/dscratch.cu \
  kernels/nvfp4_dense.cu -o build/gate_tc_fp8_kc
echo "built build/gate_tc_fp8_kc"
# Gate OG_WS1 — the ogroup GEMV scale-hoist (Finding 76) must be BIT-IDENTICAL to the shipped default. Same
# reasoning as gate_tc_fp8_kc: the claim is value equality, so the instrument is memcmp, and
# gate_ogroup_gemv next door is a cosine gate on a DIFFERENT (M=1) kernel.
nvcc -O2 -std=c++17 -gencode arch=compute_110a,code=sm_110a -I include \
  tests/gate_og_ws1.cu kernels/mla_attn.cu kernels/dscratch.cu kernels/nvfp4_dense.cu \
  -o build/gate_og_ws1
echo "built build/gate_og_ws1"

# Gate SPARSE_HPB (DECODE_LADDER 1.7) -- `sparse_attn` now has THREE launch shapes (per-warp block,
# HPB-heads-per-block, and HPB + shared-memory row staging) and the claim binding them is VALUE
# EQUALITY: the online softmax is not associative and the gathered rows are summed IN ORDER, so a
# cosine gate passes exactly the reordering this must catch (the same lesson as gate_topk_radix,
# gate_tc_fp8_kc and gate_og_ws1). memcmp of the whole output buffer against the pre-1.7 launch, at
# the six shapes the engine issues -- m=1/2/6 verify, the pre-knee topk=320, and both prefill widths
# -- and it TIMES them, which is what chose the default. `--control` bumps one gathered row by one
# ulp and the memcmp must fail; that control caught the gate being blind at topk=320 on its first run.
nvcc -O2 -std=c++17 -gencode arch=compute_110a,code=sm_110a -I include \
  tests/gate_sparse_hpb.cu kernels/mla_attn.cu kernels/dscratch.cu kernels/nvfp4_dense.cu \
  -o build/gate_sparse_hpb
echo "built build/gate_sparse_hpb"

# Gate HADAMARD_ALIAS (DECODE_LADDER 1.10) -- the read/write race that made `build/decode`'s prefill
# nondeterministic from 192 positions up. `hadamard_kernel` gives every thread of a row the whole row
# and one element of it to overwrite, which is correct only while y != x, and three call sites pass
# the same pointer twice. BOTH ARMS RUN IN ONE PROCESS via hadamard_set_stage(), and the gate returns
# non-zero if the OFF arm ever STOPS reproducing the defect -- a gate that can no longer see its own
# bug is not a passing gate. Sweeps rows across the 20-block boundary the defect fires at.
nvcc -O2 -std=c++17 -gencode arch=compute_110a,code=sm_110a -I include \
  tests/gate_hadamard_alias.cu kernels/indexer.cu kernels/compressor.cu kernels/mla_attn.cu \
  kernels/fp8_block_gemm.cu kernels/tc_fp8_gemm.cu kernels/dscratch.cu kernels/nvfp4_dense.cu \
  -o build/gate_hadamard_alias
echo "built build/gate_hadamard_alias"

# Gate KV_PACK (DECODE_LADDER 1b.2) -- the packed FP8+UE8M0 main-KV cache. TWO memcmp claims and an
# idempotence one: unpack(pack(x)) == what act_quant_fp8sim stores, on all 512 dims and five
# distributions; and `sparse_attn` over the packed cache == `sparse_attn` over the FP32 cache
# holding the same values, at all 11 (hpb,smem) launches and the shapes the engine issues. Same
# reasoning as gate_idx_pack next door: a TOLERANCE gate passes a dropped sign bit at |delta| = 0,
# which is exactly the bug 1b.1's first implementation had. `--control` flips one code byte and one
# scale byte and both memcmps must fail.
nvcc -O2 -std=c++17 -gencode arch=compute_110a,code=sm_110a -I include \
  tests/gate_kv_pack.cu kernels/mla_attn.cu kernels/dscratch.cu kernels/nvfp4_dense.cu \
  -o build/gate_kv_pack
echo "built build/gate_kv_pack"

# Gate KV_PACK_E2E (DECODE_LADDER 1b.2) -- the WIRING, which gate_kv_pack says nothing about. Drives
# compressed_attn_cache[_r4] + compressed_decode_step_{strided,indexer} + compressed_verify_step_*
# on synthetic weights, once FP32 and once packed, and memcmps the outputs. A wrong row stride in
# one of the ~30 touched call sites produces a plausible number, not a crash, and the only other
# instrument that would catch it is a 15-minute checkpoint load. `--swap` reverses the arm order,
# which is the control for state left in a static or an allocator between arms.
nvcc -O2 -std=c++17 -gencode arch=compute_110a,code=sm_110a -I include \
  tests/gate_kv_pack_e2e.cu kernels/compressed_decode.cu kernels/compressed_attn.cu kernels/compressor.cu \
  kernels/indexer.cu kernels/mla_attn.cu kernels/fp8_block_gemm.cu kernels/tc_fp8_gemm.cu \
  kernels/dscratch.cu kernels/dprof.cu kernels/nvfp4_dense.cu -o build/gate_kv_pack_e2e
echo "built build/gate_kv_pack_e2e"

# Gate JOIN_DEFER (DECODE_LADDER 1.11) -- the deferred ATTN_SPLIT join, bit for bit. This is the
# FIRST gate in this repo that calls arena_init() before driving these kernels, which is what makes
# `g_side` non-null and the fork/join path reachable at all: every other gate here links them
# without an arena, so `asplit` is false and the side stream that 1.8 measured at 0.81 ms/forward
# has never been under test. Two arms in one process (NO_JOIN_DEFER set / unset), memcmp, `--swap`
# for arm order and `--negctl` to prove the memcmp is live.
nvcc -O2 -std=c++17 -gencode arch=compute_110a,code=sm_110a -I include \
  tests/gate_join_defer.cu kernels/compressed_decode.cu kernels/compressed_attn.cu kernels/compressor.cu \
  kernels/indexer.cu kernels/mla_attn.cu kernels/fp8_block_gemm.cu kernels/tc_fp8_gemm.cu \
  kernels/dscratch.cu kernels/dprof.cu kernels/nvfp4_dense.cu -o build/gate_join_defer
echo "built build/gate_join_defer"

# f32mk_bench (DECODE_LADDER 1.12) -- it is a BENCH and a GATE in one binary: it times all 17
# instantiated (MM,NN) warp tiles of `gemm_fp32` at the four shapes the engine actually issues, and
# it returns NON-ZERO if any tile differs from the M=1 warp-per-output path by a single bit. It
# lives in tools/ rather than tests/ because it chose the shipped default, but scripts/f32mkn_ab_run.sh
# runs it in its gate phase and a missing binary there is a failed gate, so it is built here. The
# baseline arm `8x0` is the pre-1.12 host-side chunk loop, so the before-arm is gated too.
nvcc -O3 -std=c++17 -gencode arch=compute_110a,code=sm_110a -I include \
  tools/f32mk_bench.cu kernels/compressor.cu kernels/mla_attn.cu kernels/indexer.cu \
  kernels/fp8_block_gemm.cu kernels/tc_fp8_gemm.cu kernels/dscratch.cu kernels/nvfp4_dense.cu \
  -o build/f32mk_bench
echo "built build/f32mk_bench"

# ---------------------------------------------------------------------------------------------
# THE OTHER EIGHT (Finding 76). Before this block, build_gate.sh built 10 of the 18 gate binaries
# and the rest went stale SILENTLY: F74 found gate_compressed_decode and gate_indexer_decode a day
# out of date, and gate_fp4_gemv turned out to be two engine parameters behind — it had not linked,
# and therefore had not tested anything, since F65/F72 added `rows_hint`/`align8`, while a binary
# from 2026-08-06 sat in build/ looking like a passing gate. A gate that is not in the build script
# is a gate that measures the past. Sources are listed here rather than in each file's header
# comment so there is exactly one place that can drift.
#
# `nvcc ... && echo built` DOES NOT FAIL THE SCRIPT even under `set -e`: the failure is a non-final
# command of an && list, so bash exempts it, the "built" line simply never prints, and the run ends
# 0 with one binary silently missing. That is how gate_grouped_moe went missing on the first pass of
# this very block. `bg` below builds, reports, and records — and the script exits non-zero at the end
# if anything failed, because "a MISSING binary is a FAILURE, not a pass".
GA="-O2 -std=c++17 -gencode arch=compute_110a,code=sm_110a -I include"
GATE_FAILED=""
bg(){ local t="$1"; shift
      if nvcc $GA "tests/$t.cu" "$@" -o "build/$t"; then echo "built build/$t"
      else echo "FAILED TO BUILD build/$t" >&2; GATE_FAILED="$GATE_FAILED $t"; fi; }
trap '[ -n "$GATE_FAILED" ] && { echo "GATE BUILD FAILURES:$GATE_FAILED" >&2; exit 1; }; exit 0' EXIT
CDEC="kernels/compressed_decode.cu kernels/compressed_attn.cu kernels/compressor.cu kernels/indexer.cu \
  kernels/mla_attn.cu kernels/fp8_block_gemm.cu kernels/tc_fp8_gemm.cu kernels/dscratch.cu kernels/dprof.cu \
  kernels/nvfp4_dense.cu"
FULL="$CDEC kernels/mla_forward.cu kernels/mla_decode.cu kernels/hc.cu kernels/hc_sinkhorn.cu \
  kernels/moe.cu kernels/tc_moe_gemm.cu"
for t in gate_compressed_decode gate_indexer_decode; do bg $t $CDEC; done
for t in gate_scratch_init gate_compressed_graph gate_indexer_graph; do bg $t $FULL; done
bg gate_compressor_emit kernels/compressor.cu kernels/mla_attn.cu kernels/indexer.cu \
  kernels/dscratch.cu kernels/fp8_block_gemm.cu kernels/tc_fp8_gemm.cu kernels/nvfp4_dense.cu
# moe.cu, for `fp4_gemm` — the oracle this gate diffs the grouped GEMV against. Its own header build
# line lists only tc_moe_gemm.cu and has not linked since the oracle moved.
bg gate_grouped_moe kernels/tc_moe_gemm.cu kernels/moe.cu kernels/mla_attn.cu kernels/fp8_block_gemm.cu \
  kernels/tc_fp8_gemm.cu kernels/dscratch.cu kernels/dprof.cu kernels/nvfp4_dense.cu
# Gate MAINKV_INCR (ladder 1.0) — the incremental main-KV must be BIT-IDENTICAL to the from-scratch
# one at every split point. Same reasoning as gate_tc_fp8_kc and gate_og_ws1: the claim is value
# equality, so the instrument is memcmp. Seconds here instead of a 10-minute checkpoint load.
#
# BOTH NEGATIVE CONTROLS WERE RUN, 2026-08-20, not just asserted (an earlier version of this comment
# claimed them without having done it). Each was a one-line edit to a scratch copy of
# kernels/dspark_attn.cu, rebuilt and run:
#   A. rope offset dropped (cosT/sinT not advanced by r0):  FAIL on BOTH GEMM settings,
#      130,624 of 1,048,576 floats differ, first at row 7 col 448.
#   B. GEMM pin dropped (delta goes back through fp8_block_gemm's M-dependent dispatch):
#      PASS at g_tc_fp8=0 -- the oracle really is M-independent -- and **FAIL at g_tc_fp8=1 with
#      377 of 1,048,576 floats differing at the last ulp** (0.209164858 vs 0.209164843).
# B is the one that justifies the gate's two design choices. 0.036 % of floats at the last ulp is
# exactly what a tolerance-based check calls "close enough", and it appears ONLY on the tensor-core
# path -- which is the path decode uses and the one a gate running only the default would skip.
#
# Both controls put the first differing float at col 448 = NOPE_DIM, and that is mechanism, not
# coincidence: act_quant_fp8sim re-quantises columns [0, NOPE_DIM) and absorbs sub-ulp differences,
# while the ROPE half above it stays fp32 and preserves them. The NOPE half hides small errors.
bg gate_mainkv_incr kernels/dspark_attn.cu kernels/dspark.cu kernels/block.cu \
  kernels/compressed_block.cu kernels/block_decode.cu $FULL
# -lcublasLt: fp4_gemm.cu carries a cublasLt reference path next to the hand-written kernel.
bg gate_fp4_gemv kernels/fp4_gemm.cu kernels/moe.cu kernels/tc_moe_gemm.cu \
  kernels/dscratch.cu kernels/dprof.cu kernels/mla_attn.cu kernels/fp8_block_gemm.cu kernels/tc_fp8_gemm.cu \
  kernels/nvfp4_dense.cu -lcublasLt -lcublas