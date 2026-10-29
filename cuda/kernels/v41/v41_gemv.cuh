// v41_gemv.cuh — the one GEMV engine every V4.1 weight format plugs into.
//
// SHAPE OF THE WORK. A CTA owns 64 consecutive output rows (8 warps x R=8 rows) and up to MT
// activation rows. It stages its activations ONCE in shared memory as fp32 (dequantised: e4m3 code
// x pow2 scale is exact in fp32, so nothing is lost), and every warp reuses them across its 8
// rows. That is the fix from the GLM-5.3 activation-traffic finding: a per-row GEMV re-reads x for
// every weight row, which is 4 B of activation traffic per weight byte read -- more than the
// weights themselves once weights go below 4 bits.
//
// THE UNIT OF WORK is a (row, 32-element K chunk) pair. Every format below decodes one chunk into
// 32 fp32 weights with the scale already applied (exact for every format here: pow2 x e2m1/e4m3,
// e4m3 x fp16 LUT entry, fp16 x fp16-pair sum), then dots it against each staged activation row.
// Lanes stride over chunks of one row at a time, so a warp's 32 lanes read 32 consecutive chunks:
// coalesced for every format because every format stores a row's chunks contiguously.
//
// SHARED-MEMORY LAYOUT. x is stored in 16-byte granules, granule j of chunk c at c*8 + (j ^ (c&7)).
// Eight lanes of a 128-bit load phase read eight consecutive chunks, so the XOR puts them in eight
// different bank groups. Without it they all hit the same 4 banks (8-way conflict).
#pragma once
#include "v41_dev.cuh"

namespace v41g {

using namespace v41d;

static constexpr int ROWS_PER_WARP = 8;
static constexpr int WARPS = 8;
static constexpr int ROWS_PER_CTA = ROWS_PER_WARP * WARPS;   // 64

// ---------------------------------------------------------------- activation staging
// mode 0: x fp32 [M, ldx] ; mode 1: e4m3 bytes [M, ldx] + f32 scale per 32 [M, ldx/32]
struct XSrc {
    const float* xf = nullptr;
    const uint8_t* xq = nullptr;
    const float* xs = nullptr;
    int ldx = 0;
};
// stage K elements starting at column k0 of activation row `row` into smem slot (float4 granules)
__device__ __forceinline__ void stage_row(float4* dst, const XSrc& X, int row, int k0, int K) {
    const int nch = K / 32;
    for (int g = threadIdx.x; g < nch * 8; g += blockDim.x) {
        const int c = g >> 3, j = g & 7;
        const int k = k0 + c * 32 + j * 4;
        float4 v;
        if (X.xf) {
            v = *reinterpret_cast<const float4*>(X.xf + (size_t)row * X.ldx + k);
        } else {
            const uint8_t* q = X.xq + (size_t)row * X.ldx + k;
            const float s = X.xs[(size_t)row * (X.ldx / 32) + k / 32];
            uint32_t w = *reinterpret_cast<const uint32_t*>(q);
            float2 a = e4m3x2f((uint16_t)(w & 0xFFFF)), b = e4m3x2f((uint16_t)(w >> 16));
            v = make_float4(a.x * s, a.y * s, b.x * s, b.y * s);
        }
        dst[c * 8 + (j ^ (c & 7))] = v;
    }
}

template <int MT>
__device__ __forceinline__ void dot_chunk(float (&acc)[MT], const float (&w)[32], const float4* xs,
                                          int c, int Kc) {
    #pragma unroll
    for (int t = 0; t < MT; ++t) {
        const float4* xc = xs + (size_t)t * Kc * 8 + c * 8;
        float p = 0.f;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float4 v = xc[j ^ (c & 7)];
            p = fmaf(w[4 * j + 0], v.x, p); p = fmaf(w[4 * j + 1], v.y, p);
            p = fmaf(w[4 * j + 2], v.z, p); p = fmaf(w[4 * j + 3], v.w, p);
        }
        acc[t] += p;
    }
}

// ---------------------------------------------------------------- weight decoders
// Each: __device__ void load(int row, int c, float (&w)[32]) const ; `row` is the GLOBAL output row.

struct DecFP8 {                 // e4m3 [N,K], e8m0 per 32x32 block [ceil(N/32), K/32]
    const uint8_t* W; const uint8_t* Ws; int K;
    __device__ __forceinline__ void load(int row, int c, float (&w)[32]) const {
        const uint4* p = reinterpret_cast<const uint4*>(W + (size_t)row * K + c * 32);
        const float s = e8m0f(Ws[(size_t)(row >> 5) * (K >> 5) + c]);
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            const uint4 u = p[h];
            const uint32_t v[4] = {u.x, u.y, u.z, u.w};
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                float2 a = e4m3x2f((uint16_t)(v[i] & 0xFFFF)), b = e4m3x2f((uint16_t)(v[i] >> 16));
                w[16 * h + 4 * i + 0] = a.x * s; w[16 * h + 4 * i + 1] = a.y * s;
                w[16 * h + 4 * i + 2] = b.x * s; w[16 * h + 4 * i + 3] = b.y * s;
            }
        }
    }
};

struct DecBF16 {                // bf16 [N,K]
    const uint16_t* W; int K;
    __device__ __forceinline__ void load(int row, int c, float (&w)[32]) const {
        const uint4* p = reinterpret_cast<const uint4*>(W + (size_t)row * K + c * 32);
        #pragma unroll
        for (int h = 0; h < 4; ++h) {
            const uint4 u = p[h];
            const uint32_t v[4] = {u.x, u.y, u.z, u.w};
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                w[8 * h + 2 * i + 0] = __uint_as_float(v[i] << 16);
                w[8 * h + 2 * i + 1] = __uint_as_float(v[i] & 0xFFFF0000u);
            }
        }
    }
};

// MXFP4: codes [N, K/2] (low nibble first) + e8m0 per 32 along K [N, K/32]. Rows split across two
// tensors (w1 | w3) at `split` so gate and up run as one 4608-row GEMV; split = N for one tensor.
template <bool HW>
struct DecMXFP4 {
    const uint8_t* W0; const uint8_t* S0; const uint8_t* W1; const uint8_t* S1; int K; int split;
    __device__ __forceinline__ void load(int row, int c, float (&w)[32]) const {
        const uint8_t* W = W0; const uint8_t* S = S0;
        if (row >= split) { W = W1; S = S1; row -= split; }
        const uint4 u = *reinterpret_cast<const uint4*>(W + (size_t)row * (K / 2) + c * 16);
        const float s = e8m0f(S[(size_t)row * (K / 32) + c]);
        const uint32_t v[4] = {u.x, u.y, u.z, u.w};
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            #pragma unroll
            for (int b = 0; b < 4; ++b) {
                const uint8_t byte = (uint8_t)(v[i] >> (8 * b));
                if (HW) {
                    const float2 f = __half22float2(e2m1x2_h2(byte));
                    w[8 * i + 2 * b] = f.x * s; w[8 * i + 2 * b + 1] = f.y * s;
                } else {
                    // e2m1 by bit construction: no table, no divergent constant-cache access
                    #pragma unroll
                    for (int n = 0; n < 2; ++n) {
                        const uint32_t code = (byte >> (4 * n)) & 15u, m = code & 7u;
                        float mag = (m < 2u) ? 0.5f * (float)m
                                             : __uint_as_float(((m >> 1) + 126u) << 23 | (m & 1u) << 22);
                        w[8 * i + 2 * b + n] = (code & 8u ? -mag : mag) * s;
                    }
                }
            }
        }
    }
};

// NVFP4 dense overlay (thor-overlay-nvfp4, ops/lib/ggml_quants.nvfp4_dequant): codes [N, K/2] (low
// nibble = even element), e4m3 scale per 16 along K [N, K/16], one fp32 global g per tensor.
// w = (e2m1 * e4m3) * g: the product in parentheses is exact in fp32, then one rounding by g -- the
// numpy reference's float32 order, so bit-identical to it.
struct DecNVFP4 {
    const uint8_t* W; const uint8_t* S; float g; int K;
    __device__ __forceinline__ void load(int row, int c, float (&w)[32]) const {
        const uint4 u = *reinterpret_cast<const uint4*>(W + (size_t)row * (K / 2) + c * 16);
        const uint8_t* sp = S + (size_t)row * (K / 16) + 2 * c;
        const float s0 = e4m3f(sp[0]), s1 = e4m3f(sp[1]);
        const uint32_t v[4] = {u.x, u.y, u.z, u.w};
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float s = i < 2 ? s0 : s1;
            #pragma unroll
            for (int b = 0; b < 4; ++b) {
                const float2 f = __half22float2(e2m1x2_h2((uint8_t)(v[i] >> (8 * b))));
                w[8 * i + 2 * b] = (f.x * s) * g; w[8 * i + 2 * b + 1] = (f.y * s) * g;
            }
        }
    }
};

// LUT-b (FORMAT_PROPOSAL.md §2): b-bit codes into a 2^b-entry fp16 codebook, E4M3 scale per 32.
// Chunk record = 4*b bytes: b=2 -> 2 words of 2-bit codes; b=3 -> 2 words of low 2 bits + 1 word of
// high bits; b=4 -> 4 words of nibbles. The codebook lives in REGISTERS, one entry per lane, and is
// read with a warp shuffle: no shared or constant memory touched per weight.
template <int B>
struct DecLUT {
    const uint8_t* W0; const uint8_t* S0; const uint8_t* W1; const uint8_t* S1; int K; int split;
    float lut0, lut1;           // this lane's codebook entry for each of the two tensors
    __device__ __forceinline__ void load(int row, int c, float (&w)[32]) const {
        const uint8_t* W = W0; const uint8_t* S = S0; float lut = lut0;
        if (row >= split) { W = W1; S = S1; row -= split; lut = lut1; }
        const uint32_t* p = reinterpret_cast<const uint32_t*>(W + (size_t)row * (K / 32) * (4 * B) + (size_t)c * 4 * B);
        const float s = e4m3f(S[(size_t)row * (K / 32) + c]);
        uint32_t wd[B];
        #pragma unroll
        for (int i = 0; i < B; ++i) wd[i] = p[i];
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            uint32_t code;
            if (B == 2) code = (wd[i >> 4] >> (2 * (i & 15))) & 3u;
            else if (B == 3) code = ((wd[i >> 4] >> (2 * (i & 15))) & 3u) | (((wd[2] >> i) & 1u) << 2);
            else code = (wd[i >> 3] >> (4 * (i & 7))) & 15u;
            w[i] = __shfl_sync(0xffffffffu, lut, (int)code) * s;
        }
    }
};

// ggml K-quants (ops/lib/thor_format.md "Routed experts", arm A): rows are whole 256-element
// superblocks, BYTE-IDENTICAL to ggml block_q{2,3,4}_K so the GGUF writer copies them verbatim.
// A 32-element chunk c is chunk j = c&7 of superblock c>>3. Arithmetic follows the numpy reference
// (ops/lib/ggml_quants.dequant) operation by operation -- (d*sc)*q - dmin*m with separate
// roundings, no fma -- so a decoded weight is bit-identical to the reference's float32.
// Alignment: a Q2_K superblock is 84 B (4-aligned), Q3_K 110 B (2-aligned!), Q4_K 144 B (16-aligned).
// __host__ too, so tests/gate_kq_host.cu can run the SAME decode on the CPU (built with
// -ffp-contract=off so the host compiler does not fuse the multiply-subtract either).
__host__ __device__ __forceinline__ float f16_at(const uint8_t* p) {
    return __half2float(__ushort_as_half((unsigned short)(p[0] | (p[1] << 8))));
}
__host__ __device__ __forceinline__ float mul_rn(float a, float b) {
#ifdef __CUDA_ARCH__
    return __fmul_rn(a, b);
#else
    return a * b;
#endif
}
__host__ __device__ __forceinline__ float sub_rn(float a, float b) {
#ifdef __CUDA_ARCH__
    return __fsub_rn(a, b);
#else
    return a - b;
#endif
}
template <int QT>              // 2, 3, 4
struct DecKQ {
    const uint8_t* W0; const uint8_t* W1; int K; int split;
    static constexpr int BB = QT == 2 ? 84 : QT == 3 ? 110 : 144;
    __host__ __device__ __forceinline__ void load(int row, int c, float (&w)[32]) const {
        const uint8_t* W = W0;
        if (row >= split) { W = W1; row -= split; }
        const uint8_t* b = W + ((size_t)row * (K / 256) + (c >> 3)) * BB;
        const int j = c & 7;
        if (QT == 4) {
            const float d = f16_at(b), dm = f16_at(b + 2);
            const uint8_t* s = b + 4;
            int sc, mn;
            if (j < 4) { sc = s[j] & 63; mn = s[j + 4] & 63; }
            else { sc = (s[j + 4] & 0xF) | ((s[j - 4] >> 6) << 4); mn = (s[j + 4] >> 4) | ((s[j] >> 6) << 4); }
            const float a = mul_rn(d, (float)sc), m = mul_rn(dm, (float)mn);
            const uint4* q = reinterpret_cast<const uint4*>(b + 16 + (j >> 1) * 32);
            const int sh = (j & 1) * 4;
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                const uint4 u = q[h]; const uint32_t v[4] = {u.x, u.y, u.z, u.w};
                #pragma unroll
                for (int i = 0; i < 16; ++i)
                    w[16 * h + i] = sub_rn(mul_rn(a, (float)((v[i >> 2] >> (8 * (i & 3) + sh)) & 15u)), m);
            }
        } else if (QT == 2) {
            const float d = f16_at(b + 80), dm = f16_at(b + 82);
            const int h = j >> 2, sh = 2 * (j & 3);
            const uint32_t* q = reinterpret_cast<const uint32_t*>(b + 16 + h * 32);
            #pragma unroll
            for (int half = 0; half < 2; ++half) {
                const uint8_t sc = b[2 * j + half];
                const float a = mul_rn(d, (float)(sc & 0xF)), m = mul_rn(dm, (float)(sc >> 4));
                #pragma unroll
                for (int i4 = 0; i4 < 4; ++i4) {
                    const uint32_t v = q[half * 4 + i4];
                    #pragma unroll
                    for (int k = 0; k < 4; ++k)
                        w[16 * half + 4 * i4 + k] = sub_rn(mul_rn(a, (float)((v >> (8 * k + sh)) & 3u)), m);
                }
            }
        } else {
            const float d = f16_at(b + 108);
            const uint8_t* hm = b; const uint8_t* qs = b + 32 + (j >> 2) * 32; const uint8_t* s12 = b + 96;
            const int sh = 2 * (j & 3);
            #pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int k = 2 * j + half;
                const int lo = k < 8 ? (s12[k] & 0xF) : (s12[k - 8] >> 4);
                const int hi = (s12[8 + (k & 3)] >> (2 * (k >> 2))) & 3;
                const float a = mul_rn(d, (float)((lo | (hi << 4)) - 32));
                #pragma unroll
                for (int l = 0; l < 16; ++l) {
                    const int e = 16 * half + l;
                    const int q = (int)((qs[e] >> sh) & 3u) - (((hm[e] >> j) & 1u) ? 0 : 4);
                    w[e] = mul_rn(a, (float)q);
                }
            }
        }
    }
};


// KT-b, thor-v1 qtypes KT2 / KT3 (ops/lib/thor_format.md "Trellis experts", encoder ops/lib/kt.py):
// tail-biting bitshift trellis over 128-weight blocks of the RHT-rotated row, L = 12-bit windows,
// QTIP's 3INST computed codebook. Block = 4*b uint32 words; weight i's window is the 12 bits at
// stream bit i*b (mod 128*b), little-endian across words. One fp16 scale per block [N, K/128].
// A 32-weight chunk q of a block starts exactly at word q*b, so a lane needs only words
// q*b .. q*b+3 (wrapping inside the block) and every shift is a compile-time constant.
// The ROTATION is not here: the activations are rotated once per pass (k_rht_rows in v41_moe.cu),
// so this decoder emits the rotated-domain weight, exactly kt.dequant(.., sgn=None) in numpy:
// value = (fp32(fp16 lo) + fp32(fp16 hi)) * fp32(scale), separate roundings. __host__ too, for
// tests/gate_kt_host.cu.
static constexpr int KT_L = 12;
__host__ __device__ __forceinline__ float add_rn(float a, float b) {
#ifdef __CUDA_ARCH__
    return __fadd_rn(a, b);
#else
    return a + b;
#endif
}
__host__ __device__ __forceinline__ uint32_t fshr32(uint32_t lo, uint32_t hi, int sh) {
#ifdef __CUDA_ARCH__
    return __funnelshift_r(lo, hi, sh);
#else
    return (uint32_t)((((uint64_t)hi << 32) | lo) >> sh);
#endif
}
__host__ __device__ __forceinline__ float kt_value(uint32_t win) {
    const uint32_t x = win * 89226354u + 64248484u;
    const uint32_t m = (x & 0x8fff8fffu) ^ 0x3b603b60u;
    return add_rn(__half2float(__ushort_as_half((unsigned short)(m & 0xFFFFu))),
                  __half2float(__ushort_as_half((unsigned short)(m >> 16))));
}
template <int B>               // 2, 3
struct DecKT {
    const uint8_t* W0; const uint8_t* S0; const uint8_t* W1; const uint8_t* S1; int K; int split;
    static_assert(B == 2 || B == 3, "KT2 / KT3 only: a 32-weight chunk's windows must fit 4 words");
    __host__ __device__ __forceinline__ void load(int row, int c, float (&w)[32]) const {
        const uint8_t* W = W0; const uint8_t* S = S0;
        if (row >= split) { W = W1; S = S1; row -= split; }
        constexpr int NW = 4 * B;                       // words per 128-weight block
        const int blk = c >> 2, q = c & 3;
        const size_t bi = (size_t)row * (K / 128) + blk;
        const uint32_t* p = reinterpret_cast<const uint32_t*>(W) + bi * NW;
        const float s = f16_at(S + bi * 2);
        uint32_t wd[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) wd[i] = p[(q * B + i) % NW];
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            const int bit = i * B, wi = bit >> 5, sh = bit & 31;     // wi + 1 <= 3 for B <= 3
            const uint32_t win = fshr32(wd[wi], wd[wi + 1], sh) & ((1u << KT_L) - 1u);
            w[i] = mul_rn(kt_value(win), s);
        }
    }
};

} // namespace v41g
