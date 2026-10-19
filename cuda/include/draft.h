// draft.h — DFlash draft model (qwen3-style, 5 layers) forward + propose.
// Self-contained module: BF16 weights decoded on the fly, fp32 activations.
// See reference/DFLASH_SPEC.md (Q3 propose, Q6 draft KV, Q7 exact forward).
#pragma once
#include <cstdint>

struct DraftModel;  // opaque

// Load the DFlash draft checkpoint. cap = max number of *context* positions the
// draft KV cache must hold (the per-layer cache is sized [cap+16] slots).
DraftModel* draft_load(const char* path, int cap);

// Propose k draft token ids (k <= 15) for one block, single forward (parallel).
//   taps_dev   : device fp32 [C*6*2816], residual after target layers
//                [1,6,11,17,22,27] for the C context positions (concat order = that list).
//   ctx_pos    : host int[C], absolute sequence positions of the context tokens.
//   C          : number of context positions.
//   next_token : the bonus token the target just sampled (query slot 0), at P0=ctx_pos[C-1]+1.
//   embed_bf16 : device bf16 [262144*2816], shared target embed_tokens table.
//   lmhead_bf16: device bf16 [262144*2816], shared target lm_head table (== embed for tied).
//   out_ids    : host int[k], filled with the k draft token ids (argmax, greedy).
void draft_propose(DraftModel* d, const float* taps_dev, const int* ctx_pos, int C,
                   int next_token, const uint16_t* embed_bf16, const uint16_t* lmhead_bf16,
                   const uint8_t* ewp, const uint8_t* ews, float egs,
                   int* out_ids, int k);

void draft_free(DraftModel* d);
