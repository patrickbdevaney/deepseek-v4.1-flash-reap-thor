// v41_image.h — V2 host-side image input for the V4.1 vision tower: the checkpoint's inference/image_processor.py
// in C++ (header-only, no CUDA). load_image(): decode (stb_image, any 8-bit format it reads, converted to RGB),
// plan_image_grid (min-pixel upscale, 14-pixel rounding, the 1024-token cap), PIL's ImageOps.pad onto a
// (127,127,127) canvas with PIL's own fixed-point BICUBIC resample (Resample.c: separable, 22-bit coefficients,
// horizontal pass then vertical, uint8 between passes), then (x / 255 - 0.5) / 0.5 rounded to bf16 in the
// [P, 3, 14, 14] patch layout. The span layout is image_token_types(). JPEG decoding is stb's, not libjpeg's,
// so decoded pixels may differ by a level or two from PIL's; everything after decoding follows PIL exactly.
#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef V41_IMAGE_NO_STB_IMPL
#define STB_IMAGE_IMPLEMENTATION
#endif
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#define STBI_ONLY_BMP
#define STBI_ONLY_GIF
#include "third_party/stb_image.h"

namespace v41img {

constexpr int PATCH = 14, DOWN = 3, MAX_TOKENS = 1024, MIN_PIXELS = 295936;   // config.json vision_*
enum Type { TEXT = -1, IMAGE_START = 0, IMAGE = 1, IMAGE_NEW_LINE = 2, IMAGE_END = 3 };

struct Image { int w = 0, h = 0; std::vector<uint8_t> rgb; };   // row-major RGB
struct Prepared {
    std::vector<uint16_t> patches;       // bf16 [n_vit_h * n_vit_w, 3, 14, 14]
    int n_vit_h = 0, n_vit_w = 0, n_llm_h = 0, n_llm_w = 0;
    std::vector<int> types;              // the span, image_token_types()
    uint64_t hash = 0;                   // FNV-1a of the decoded pixels (prefix-cache identity)
};

inline int num_image_tokens(int lh, int lw) { return lh * (lw + 1) + 2; }
inline void llm_grid(int bh, int bw, int& lh, int& lw) {
    lh = (int)std::ceil((double)(bh / PATCH) / DOWN); lw = (int)std::ceil((double)(bw / PATCH) / DOWN);
}
inline void solve_resize_ratio(int height, int width, int max_n_token, int& bh, int& bw) {
    const double r = (double)height / width;
    const double max_w = std::sqrt((max_n_token - 2) / r + 0.25) - 0.5, max_h = max_w * r;
    const int cell = PATCH * DOWN;
    if (max_w < 1.0) { bh = (max_n_token - 2) / 2 * cell; bw = cell; return; }
    if (max_h < 1.0) { bh = cell; bw = (max_n_token - 3) * cell; return; }
    const double beta = std::min(std::floor(max_w) * cell / width, std::floor(max_h) * cell / height);
    bh = (int)std::floor(height * beta / PATCH) * PATCH; bw = (int)std::floor(width * beta / PATCH) * PATCH;
}
// plan_image_grid(width, height) with vision_max_wh_ratio = None
inline void plan_image_grid(int width0, int height0, int& lh, int& lw, int& bh, int& bw, int max_n_token = MAX_TOKENS) {
    double width = width0, height = height0;
    int iw = width0, ih = height0;
    if (0 < width * height && width * height < MIN_PIXELS) {
        const double ratio = std::sqrt(MIN_PIXELS / (width * height));
        iw = (int)(width * ratio); ih = (int)(height * ratio);
    }
    bw = (int)std::ceil((double)iw / PATCH) * PATCH; bh = (int)std::ceil((double)ih / PATCH) * PATCH;
    llm_grid(bh, bw, lh, lw);
    if (num_image_tokens(lh, lw) > max_n_token) {
        solve_resize_ratio(ih, iw, max_n_token, bh, bw);
        llm_grid(bh, bw, lh, lw);
        if (num_image_tokens(lh, lw) > max_n_token) throw std::runtime_error("image: token plan over budget");
    }
}

// ---- PIL Resample.c, BICUBIC (a = -0.5, support 2), 8 bpc
namespace detail {
constexpr int PB = 32 - 8 - 2;
inline double bicubic(double x) {
    const double a = -0.5;
    if (x < 0.0) x = -x;
    if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1;
    if (x < 2.0) return (((x - 5) * x + 8) * x - 4) * a;
    return 0.0;
}
inline int coeffs(int inSize, double in0, double in1, int outSize, std::vector<int>& bounds, std::vector<int32_t>& kk) {
    const double scale = (in1 - in0) / outSize;
    double fs = scale < 1.0 ? 1.0 : scale;
    const double support = 2.0 * fs;
    const int ksize = (int)std::ceil(support) * 2 + 1;
    std::vector<double> pre((size_t)outSize * ksize);
    bounds.assign((size_t)outSize * 2, 0);
    for (int xx = 0; xx < outSize; ++xx) {
        const double center = in0 + (xx + 0.5) * scale, ss = 1.0 / fs;
        double ww = 0.0;
        int xmin = (int)(center - support + 0.5); if (xmin < 0) xmin = 0;
        int xmax = (int)(center + support + 0.5); if (xmax > inSize) xmax = inSize;
        xmax -= xmin;
        double* k = &pre[(size_t)xx * ksize];
        int x = 0;
        for (; x < xmax; ++x) { const double w = bicubic((x + xmin - center + 0.5) * ss); k[x] = w; ww += w; }
        for (x = 0; x < xmax; ++x) if (ww != 0.0) k[x] /= ww;
        for (; x < ksize; ++x) k[x] = 0;
        bounds[xx * 2] = xmin; bounds[xx * 2 + 1] = xmax;
    }
    kk.resize(pre.size());
    for (size_t i = 0; i < pre.size(); ++i)
        kk[i] = pre[i] < 0 ? (int32_t)(-0.5 + pre[i] * (1 << PB)) : (int32_t)(0.5 + pre[i] * (1 << PB));
    return ksize;
}
inline uint8_t clip8(int v) { v >>= PB; return (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v); }
}  // namespace detail

// Image.resize((ow, oh), BICUBIC) for RGB
inline Image resize_bicubic(const Image& in, int ow, int oh) {
    using namespace detail;
    if (ow == in.w && oh == in.h) return in;
    std::vector<int> bh_, bv; std::vector<int32_t> kh, kv;
    const int ksh = coeffs(in.w, 0, in.w, ow, bh_, kh), ksv = coeffs(in.h, 0, in.h, oh, bv, kv);
    const bool need_h = ow != in.w, need_v = oh != in.h;
    const Image* src = &in;
    Image tmp;
    const int yfirst = bv[0], ylast = bv[oh * 2 - 2] + bv[oh * 2 - 1];
    if (need_h) {
        for (int i = 0; i < oh; ++i) bv[i * 2] -= yfirst;
        tmp.w = ow; tmp.h = ylast - yfirst; tmp.rgb.resize((size_t)tmp.w * tmp.h * 3);
        for (int y = 0; y < tmp.h; ++y)
            for (int xx = 0; xx < ow; ++xx) {
                const int xmin = bh_[xx * 2], xmax = bh_[xx * 2 + 1];
                const int32_t* k = &kh[(size_t)xx * ksh];
                for (int c = 0; c < 3; ++c) {
                    int ss = 1 << (PB - 1);
                    for (int x = 0; x < xmax; ++x) ss += in.rgb[((size_t)(y + yfirst) * in.w + x + xmin) * 3 + c] * k[x];
                    tmp.rgb[((size_t)y * ow + xx) * 3 + c] = clip8(ss);
                }
            }
        src = &tmp;
    }
    if (!need_v) return tmp.w ? tmp : in;
    Image out; out.w = ow; out.h = oh; out.rgb.resize((size_t)ow * oh * 3);
    for (int yy = 0; yy < oh; ++yy) {
        const int ymin = bv[yy * 2], ymax = bv[yy * 2 + 1];
        const int32_t* k = &kv[(size_t)yy * ksv];
        for (int xx = 0; xx < ow; ++xx)
            for (int c = 0; c < 3; ++c) {
                int ss = 1 << (PB - 1);
                for (int y = 0; y < ymax; ++y) ss += src->rgb[((size_t)(y + ymin) * src->w + xx) * 3 + c] * k[y];
                out.rgb[((size_t)yy * ow + xx) * 3 + c] = clip8(ss);
            }
    }
    return out;
}

// ImageOps.pad(image, (W, H), color=(127,127,127)), centering 0.5 (Python round: half to even)
inline Image pad_to(const Image& im, int W, int H) {
    const double ir = (double)im.w / im.h, dr = (double)W / H;
    if (ir == dr) return resize_bicubic(im, W, H);
    Image out; out.w = W; out.h = H; out.rgb.assign((size_t)W * H * 3, 127);
    Image r; int x0 = 0, y0 = 0;
    if (ir > dr) {
        const int nh = (int)std::nearbyint((double)im.h / im.w * W);
        r = nh != H ? resize_bicubic(im, W, nh) : im;
        y0 = (int)std::nearbyint((H - nh) * 0.5);
    } else {
        const int nw = (int)std::nearbyint((double)im.w / im.h * H);
        r = nw != W ? resize_bicubic(im, nw, H) : im;
        x0 = (int)std::nearbyint((W - nw) * 0.5);
    }
    for (int y = 0; y < r.h; ++y) {                    // paste, clipped to the canvas
        const int oy = y + y0; if (oy < 0 || oy >= H) continue;
        for (int x = 0; x < r.w; ++x) {
            const int ox = x + x0; if (ox < 0 || ox >= W) continue;
            memcpy(&out.rgb[((size_t)oy * W + ox) * 3], &r.rgb[((size_t)y * r.w + x) * 3], 3);
        }
    }
    return out;
}

inline uint16_t f2bf(float f) {                      // round to nearest even
    uint32_t u; memcpy(&u, &f, 4);
    u += 0x7fff + ((u >> 16) & 1);
    return (uint16_t)(u >> 16);
}
inline Image decode(const std::string& bytes) {
    int w, h, n;
    stbi_uc* p = stbi_load_from_memory((const stbi_uc*)bytes.data(), (int)bytes.size(), &w, &h, &n, 3);
    if (!p) throw std::runtime_error(std::string("image: cannot decode (") + stbi_failure_reason() + ")");
    Image im; im.w = w; im.h = h; im.rgb.assign(p, p + (size_t)w * h * 3);
    stbi_image_free(p);
    return im;
}
inline std::vector<int> image_token_types(int lh, int lw) {
    std::vector<int> t{IMAGE_START};
    for (int r = 0; r < lh; ++r) { for (int c = 0; c < lw; ++c) t.push_back(IMAGE); t.push_back(IMAGE_NEW_LINE); }
    t.push_back(IMAGE_END);
    return t;
}
inline Prepared prepare(const Image& im, int max_n_token = MAX_TOKENS) {
    Prepared P;
    int bh, bw;
    plan_image_grid(im.w, im.h, P.n_llm_h, P.n_llm_w, bh, bw, max_n_token);
    P.n_vit_h = bh / PATCH; P.n_vit_w = bw / PATCH;
    const Image c = pad_to(im, bw, bh);
    const int nh = P.n_vit_h, nw = P.n_vit_w;
    P.patches.resize((size_t)nh * nw * 3 * PATCH * PATCH);
    for (int py = 0; py < nh; ++py)
        for (int px = 0; px < nw; ++px)
            for (int ch = 0; ch < 3; ++ch)
                for (int y = 0; y < PATCH; ++y)
                    for (int x = 0; x < PATCH; ++x) {
                        const uint8_t v = c.rgb[((size_t)(py * PATCH + y) * bw + px * PATCH + x) * 3 + ch];
                        const float f = ((float)v / 255.f - 0.5f) / 0.5f;
                        P.patches[((((size_t)py * nw + px) * 3 + ch) * PATCH + y) * PATCH + x] = f2bf(f);
                    }
    P.types = image_token_types(P.n_llm_h, P.n_llm_w);
    uint64_t hsh = 1469598103934665603ull;
    for (uint8_t b : im.rgb) { hsh ^= b; hsh *= 1099511628211ull; }
    hsh ^= (uint64_t)im.w << 32 | (uint32_t)im.h; hsh *= 1099511628211ull;
    P.hash = hsh | 1;                                  // nonzero: 0 marks text rows
    return P;
}

}  // namespace v41img
