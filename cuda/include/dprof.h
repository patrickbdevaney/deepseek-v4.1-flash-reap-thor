// dprof.h — named-phase GPU timing, for attributing a composite step to its sub-operations.
//
// Built for LOOP_LOG Finding 15: ~70 ms of the M>=2 verify step is unattributed after three
// refuted hypotheses (MoE expert-union, DSA, ogroup/HC). The layer-flavour split narrowed it to
// "something every layer does"; this narrows it to WHICH sub-op, by timing all 8 phases of
// block_verify_step across all 43 layers and comparing K=1 against K=5.
//
// Design notes:
//  - Off by default and compiled to nothing but a branch; enabled with DSV4_DPROF=1.
//  - Events are recorded into a preallocated pool and NOT synchronised until dprof_report(), so
//    the instrumented path stays asynchronous (a sync per phase would itself create the stalls we
//    are hunting, which is exactly the trap dspark_forward_head fell into).
//  - Pool is fixed-size; overflow stops recording rather than reallocating mid-measurement.
#pragma once
#include <cuda_runtime.h>

enum DProfId {
    DP_HC_PRE_ATTN = 0, DP_RMSNORM_ATTN, DP_ATTN, DP_HC_POST_ATTN,
    DP_HC_PRE_FFN, DP_RMSNORM_FFN, DP_MOE, DP_HC_POST_FFN, DP_KV_XIN,
    // OUTSIDE the layer loop. Everything above sums to the "TOTAL" the report prints, and that
    // total has been ~20 ms short of the measured step for the whole optimisation loop — the
    // largest single unexamined block in the engine, simply because nothing timed it.
    DP_EMBED, DP_HEAD_HC, DP_LM_HEAD, DP_ARGMAX,
    // second level: inside the attention phase (Finding 15 narrowed the M>=2 step to ATTENTION)
    // NOTE: these live in mla_decode.cu, which serves only the 2 pure-sliding (ratio==0) layers.
    // The other 41 go through compressed_decode.cu and were invisible here — extrapolating a
    // per-layer cost from these two overstated `ogroup` by ~5x and sent Finding 35 chasing 12 ms
    // that was worth 2.4. The DP_C_* ids below close that hole.
    DP_A_QPROJ, DP_A_KV, DP_A_SPARSE, DP_A_OGROUP, DP_A_MISC,
    // compressed-attention layers (the other 41)
    DP_C_QPROJ, DP_C_COMPRESS, DP_C_INDEXER, DP_C_SPARSE, DP_C_OGROUP,
    // inside MoE — 44% of the step and, until now, entirely unattributed
    DP_M_ROUTER, DP_M_GROUP, DP_M_W13, DP_M_ACT, DP_M_W2, DP_M_COMBINE, DP_M_SHARED,
    // Third level, inside cattn:q_proj and cattn:ogroup — the two worst-efficiency big items left
    // (1.63 GB in 19.85 ms = 35% of roofline, and 2.75 GB in 21.52 = 55%). Each bundles 4-6 kernels
    // plus, for q_proj, the C1 kv-fork join, so the region total says nothing about which part is
    // slow. 22.6 ms of the 144.6 ms verify sits behind these marks.
    DP_Q_AQX, DP_Q_WQA, DP_Q_RMSAQ, DP_Q_WQB, DP_Q_TAIL, DP_Q_KVJOIN,
    DP_O_ROPE, DP_O_WOA, DP_O_WOB,
    DP_I_QIDX, DP_I_IW, DP_I_SCORE, DP_I_TOPK,
    // Fourth level, inside moe:group. 2.66 ms of the K=5 verify sits in a region that moves almost
    // no bytes (gather+quant of maxm=30 rows is ~0.9 MB), so it is launch/latency, and TWO of its
    // six launches are <<<1,1>>> serial scans over nr=160 (k_moe_prefix, k_build_tiles). Finding 71
    // is the precedent for not guessing which: mark them.
    DP_MG_COUNT, DP_MG_PREFIX, DP_MG_SCATTER, DP_MG_TILES,
    // THE DRAFT SIDE, which nothing has ever timed. Every mark above lives in the VERIFY stack, so
    // a dprof TOTAL has only ever described part of a decode step -- and `dspark_main_kv` is called
    // NSTAGE times per step over the FULL `ctxlen` (src/engine.cu), which makes it O(context) and a
    // first-class suspect for ladder item 0.4's residual 7.36 ms/1000. Attributing a context slope
    // with the one unmarked O(context) call left out is how 0.2 happened; mark it.
    // These sit AFTER DP_ARGMAX so they print as their own rows and do not enter TOTAL, which keeps
    // every historical table comparable.
    DP_D_MAINKV, DP_D_BLOCK, DP_D_HEAD,
    DP_N
};

extern bool g_dprof_on;

void dprof_init(int max_marks = 8192);
void dprof_begin(int id, cudaStream_t s = 0);
void dprof_end(int id, cudaStream_t s = 0);
void dprof_reset();
// Sync, sum elapsed per id, print, and reset. `tag` labels the row block (e.g. "K=1").
void dprof_report(const char* tag);
