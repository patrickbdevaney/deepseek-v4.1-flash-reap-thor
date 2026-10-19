// v41_config.h — DeepSeek-V4.1-Flash-1017 (and its REAP derivatives) geometry + per-layer roles.
//
// Ground truth: ~/models/DeepSeek-V4.1-Flash-1017/inference/config.json and the 48 safetensors
// headers copied verbatim into docs/hdrs/ (fetched by HTTP Range at revision 1bae0e10). Every
// constant below was read from those files. What changed from the 0731 engine this tree was
// copied from (include/deepseek_v4.h) is listed in docs/V41_DELTA.md; the short version:
//
//   * dim 4096 -> 5120, 43 -> 40 layers, q_lora 1024 -> 1280, moe_inter 2048 -> 2304
//   * fp8 blocks 128x128 -> 32x32 (weights) and 128 -> 32 (activations); MXFP4 experts unchanged
//   * no hash-routed layers; router adds bias_vl for image-span tokens
//   * compress ratios 4/128 -> 2 (layers 2..19) and 1 (layers 20..39), and CSA2 SHARING: only
//     kv_source layers compress, only index_source layers run an indexer, everyone else reuses
//   * indexer: 32 heads (was 64), no Hadamard, fp4-sim keys; candidate pre-filter from layer 20
//   * compressed KV is fp4-sim with E4M3 scales per 16 (was fp8-sim); index keys fp4-sim per 32
//   * HC pre-mix is DEFERRED: a sublayer collapses with the mix the PREVIOUS sublayer produced,
//     and the final head collapses with the last FFN's mix (no separate hc_head_fn)
//   * Engram at layers 1 and 14, applied to the 4-stream residual BEFORE the block
#pragma once
#include <cstdint>
#include <string>
#include <stdexcept>

namespace v41 {

// ---- core dims ----
static const int DIM        = 5120;
static const int N_LAYERS   = 40;
static const int VOCAB      = 129280;
static const int N_HEADS    = 64;
static const int HEAD_DIM   = 512;
static const int ROPE_DIM   = 64;
static const int NOPE_DIM   = HEAD_DIM - ROPE_DIM;      // 448
static const int Q_LORA     = 1280;
static const int O_LORA     = 1024;
static const int O_GROUPS   = 8;
static const int O_GROUP_IN = N_HEADS * HEAD_DIM / O_GROUPS;   // 4096
static const int WINDOW     = 128;
static const float NORM_EPS = 1e-20f;                   // config norm_eps (NOT 1e-6)

// ---- quantisation geometry (quantization_config.weight_block_size = [32, 32]) ----
static const int FP8_BLK    = 32;                       // weight block (both dims) AND activation block
static const int FP4_BLK    = 32;                       // MXFP4 expert scale group along K
static const int CKV_FP4_GRP = 16;                      // compressed KV: fp4 values, E4M3 scale per 16
static const int IDX_FP4_GRP = 32;                      // index keys/queries: fp4, E8M0 scale per 32

// ---- MoE ----
static const int N_ROUTED_TEACHER = 384;                // teacher; REAP checkpoints are DISCOVERED
static const int N_ACT      = 6;
static const int MOE_INTER  = 2304;
static const float ROUTE_SCALE  = 1.5f;
static const float SWIGLU_LIMIT = 10.0f;
// score_func sqrtsoftplus; bias SELECTS (noaux_tc), unbiased scores WEIGHT; renorm + 1e-20; x1.5.

// ---- Hyper-Connections ----
static const int HC_MULT = 4;
static const int HC_MIX  = (2 + HC_MULT) * HC_MULT;     // 24
static const int HC_DIM  = HC_MULT * DIM;               // 20480
static const int HC_SINKHORN_ITERS = 20;
static const float HC_EPS = 1e-6f;

// ---- CSA2 (compressed sparse attention, shared) ----
static const int INDEX_N_HEADS  = 32;
static const int INDEX_HEAD_DIM = 128;
static const int INDEX_TOPK     = 512;
static const int CAND_SOURCE    = 20;
static const int CAND_TOPK_BLOCKS = 2048;
static const int CAND_BLOCK     = 8;
static const int KV_SOURCES[4]  = {2, 8, 14, 20};
static const int INDEX_SOURCES[8] = {2, 8, 14, 20, 24, 28, 32, 36};

// compress_ratios[43] (40 backbone + 3 MTP): 0,0, 2 x18, 1 x20, 0,0,0
static inline int compress_ratio(int L) {
    if (L < 2 || L >= N_LAYERS) return 0;
    return L < 20 ? 2 : 1;
}
static inline bool is_kv_source(int L)    { for (int s : KV_SOURCES) if (s == L) return true; return false; }
static inline bool is_index_source(int L) { for (int s : INDEX_SOURCES) if (s == L) return true; return false; }
// The layer whose compressed KV (and index keys) layer L reads: the latest kv_source <= L. Layers
// sharing a ratio share the cache; the reference relies on layer order (`shared_attn`), so do we.
static inline int kv_owner(int L) { int o = -1; for (int s : KV_SOURCES) if (s <= L) o = s; return o; }
static inline int index_owner(int L) { int o = -1; for (int s : INDEX_SOURCES) if (s <= L) o = s; return o; }
static inline bool uses_candidates(int L) { return CAND_SOURCE >= 0 && CAND_SOURCE < L; }

// ---- encoder / decoder split ----
// Layers 21..39 read ALL compressed KV from layer 20 (ratio 1 = one latent per token), so their own
// per-token KV exists only in a 128-token window. A prompt token older than the window therefore
// never needs to pass through layers 21..39 -- "causal encoder-decoder". But the replay is NOT 128
// tokens, as PLAN.md states: decoder layer L's window KV of token t depends on layer L-1's OUTPUT
// at t, which attends layer L-1's window back to t-127. Exactness needs the decoder to run over
//      WINDOW + (N_DEC - 1) * (WINDOW - 1)   tokens  = 128 + 18*127 = 2414
// where N_DEC = 19 (layers 21..39). docs/V41_DELTA.md has the derivation; gate_v41_encdec checks it.
static const int ENC_LAST   = 20;                       // last layer that must see every prompt token
static const int N_DEC      = N_LAYERS - 1 - ENC_LAST;  // 19
static const int DEC_REPLAY = WINDOW + (N_DEC - 1) * (WINDOW - 1);   // 2414
// The span shrinks layer by layer: layer L in 21..39 must process the last dec_span(L) prompt
// tokens (layer 39: 128, layer 21: 2414). Layers <= 20 process all of them.
static inline int dec_span(int L) {
    return L <= ENC_LAST ? 0x7fffffff : WINDOW + (N_LAYERS - 1 - L) * (WINDOW - 1);
}

// ---- RoPE / YaRN ----
static const float ROPE_THETA          = 10000.0f;      // ratio-0 layers (0, 1, MTP): YaRN off
static const float COMPRESS_ROPE_THETA = 160000.0f;     // ratio 1/2 layers: YaRN on
static const int   YARN_ORIG_MAXPOS    = 65536;
static const float YARN_FACTOR         = 16.0f;
static const int   YARN_BETA_FAST      = 32;
static const int   YARN_BETA_SLOW      = 1;
static const int   MAX_CTX             = YARN_ORIG_MAXPOS * 16;   // 1,048,576

// ---- Engram ----
static const int ENGRAM_LAYERS[2] = {1, 14};
static const long long ENGRAM_ROWS[2] = {384006168LL, 384016682LL};
static const int ENGRAM_MAX_NGRAM = 4;
static const int ENGRAM_N_HEADS   = 8;
static const int ENGRAM_HEAD_DIM  = 256;
static const int ENGRAM_COLS      = (ENGRAM_MAX_NGRAM - 1) * ENGRAM_N_HEADS;   // 24 rows / token / layer
static const int ENGRAM_KV_IN     = ENGRAM_COLS * ENGRAM_HEAD_DIM;            // 6144
static const int ENGRAM_KV_OUT    = DIM * (HC_MULT + 1);                      // 25600
static const int ENGRAM_PAD_ID    = 2;
static const int ENGRAM_CVOCAB    = 99092;
static const long long ENGRAM_BUCKET_START = 16000000;
static inline int engram_slot(int L) { return L == 1 ? 0 : (L == 14 ? 1 : -1); }

// ---- DSpark / MTP ----
static const int N_MTP = 3;
static const int DSPARK_BLOCK = 5;
static const int DSPARK_NOISE_TID = 128799;
static const int DSPARK_MARKOV_RANK = 256;
static const int DSPARK_N_ROUTED = 128;
static const int DSPARK_N_ACT = 3;
static const int DSPARK_TAP_LAYERS[3] = {37, 38, 39};

static const int IMAGE_TOKEN_ID = 129264;

template <class WMap>
inline int discover_n_routed(const WMap& W, const std::string& expert_pfx) {
    int n = 0;
    while (W.has(expert_pfx + std::to_string(n) + ".w1.weight")) ++n;
    if (n == 0) throw std::runtime_error("no routed experts under '" + expert_pfx + "'");
    return n;
}

} // namespace v41
