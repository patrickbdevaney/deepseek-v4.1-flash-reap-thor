// v41_vision.h — DeepSeek-V4.1 vision tower (inference/vision.py ViT + Aligner) on CUTLASS tensor cores (V1).
//
// ViT: patch_embed (Linear 588 -> 1024), 32 blocks of [RMSNorm, attention with 2D RoPE (16 heads x 64,
// bidirectional over the whole image), RMSNorm, SwiGLU MLP 1024 -> 2 x 2816 -> 1024], final RMSNorm.
// Aligner: 3x3 unfold of the patch grid (zero padded to multiples of 3; feature index c * 9 + ky * 3 + kx),
// Linear 9216 -> 5120, exact GELU, Linear 5120 -> 5120. Weights are the checkpoint's bf16; every GEMM takes
// bf16 operands (activations rounded at its input, as the model's own bf16 path does) with an fp32 sum and an
// fp32 output; the residual stream stays fp32.
#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <cuda_runtime.h>

namespace v41m { class Store; }
namespace v41v {

constexpr int PATCH = 14, VDIM = 1024, VHEADS = 16, VHD = 64, VINTER = 2816, VLAYERS = 32, DOWN = 3;
constexpr int PATCH_IN = 3 * PATCH * PATCH, PATCH_IN_PAD = 592;   // 588 padded to a multiple of 8 (16-byte rows)
constexpr int AL_IN = VDIM * DOWN * DOWN, LLM_DIM = 5120;

struct VBlock {
    const uint16_t *norm1 = nullptr, *norm2 = nullptr, *wqkv = nullptr, *wo = nullptr, *w1 = nullptr, *w2 = nullptr;
    float *bqkv = nullptr, *bo = nullptr;          // biases as fp32 (GEMM epilogue C, broadcast)
};
struct Vision {
    std::unique_ptr<v41m::Store> store;
    uint16_t* patch_w = nullptr; float* patch_b = nullptr;     // [1024, 592] zero-padded copy
    VBlock blk[VLAYERS];
    const uint16_t* norm = nullptr;
    const uint16_t *al_w1 = nullptr, *al_w2 = nullptr; float *al_b1 = nullptr, *al_b2 = nullptr;
    const uint16_t *img_start = nullptr, *img_end = nullptr, *img_newline = nullptr;   // [5120] bf16
    // scratch, sized for max_patches
    int max_patches = 0;
    float *x = nullptr, *t = nullptr, *big = nullptr;          // [P,1024] residual, [P,3072] qkv / wo out, [P,5632] mlp
    uint16_t *xb = nullptr, *qkvb = nullptr, *ab = nullptr;    // bf16 operands
    std::vector<void*> allocs;
    double gib = 0;
    Vision();
    ~Vision();
};
// dir: the thor build (vision.safetensors, globals.safetensors) or the HF checkpoint. Throws on a missing tensor.
void load_vision(Vision& V, const std::string& dir, int max_patches = 9216);
// patches: device bf16 [n_h * n_w, 3, 14, 14] (image_processor.load_image's layout and normalisation).
// out: device fp32 [ceil(n_h / 3) * ceil(n_w / 3), 5120], aligner rows in reading order.
void encode_image(Vision& V, const uint16_t* patches, int n_h, int n_w, float* out, cudaStream_t st);
// debug: after the ViT (before the aligner), the [n_h * n_w, 1024] fp32 output of the final norm in V.x
inline int llm_rows(int n_h, int n_w) { return ((n_h + DOWN - 1) / DOWN) * ((n_w + DOWN - 1) / DOWN); }

} // namespace v41v
