// v41_kq_mma.cuh — routed-expert GEMV for ggml Q2_K / Q3_K on fp8 tensor cores (mma.m16n8k16 e4m3).
//
// WHY. k_kq (v41_kq_fast.cuh) is issue-bound: ncu 2026-10-29, Q2_K gate|up at M=1 runs ~4.5 thread
// instructions per weight (code extraction, HFMA2, per-chunk scaling) at 0.66 eligible warps per
// scheduler, so the experts stream at ~164 GB/s against ~210 for plain reads.
//
// HOW. A K-quant code (Q2_K q in 0..3; Q3_K c = q | hbit << 2 in 0..7) read as an e4m3 BYTE is the
// subnormal c * 2^-9, so the masked code bytes are mma operands with no conversion at all. Expert
// inputs are e4m3 codes x an fp32 scale per 32 (act_quant_fp8), so B is the staged x bytes. One
// mma.m16n8k16 covers 16 rows x one 16-weight sub-block (one K-quant scale) x up to 8 tokens, with
// every product exact and an fp32 sum (the half2 path rounds an 8-term fp16 sum). Per sub-block:
//     out[r][n] += (d*sc)[r][sb] * (D[r][n] * xs[n][chunk])                      (code term, x 512)
//     out[r][n] -= (dmin*m)[r][sb] * xm[n][sb]   Q2_K;   -= 4 (d*sc)[r][sb] * xm[n][sb]   Q3_K
// with xm = (sum of the sub-block's e4m3 values) * xs. The code-free term is summed per lane over a
// quarter of the sub-blocks and reduced over the quad at the end. NOT bit-identical to k_kq (different,
// tighter rounding); every output element's arithmetic is independent of how many tokens share the
// unit, so verify rows equal AR decode and prefill equals decode.
//
// LAYOUT (kq_mma_repack, in place, same bytes as the ggml blocks). Rows in tiles of 16; per (tile,
// superblock) idx, planes in this order, each over all idx:
//   codes  1024 B : lane l = 4g + t holds 32 B = 8 words; word sb>>1, 2-bit field 2*(sb&1)+h, byte b
//                   is the low-2 code of row g + 8h, weight 16 sb + 4t + b (the mma A fragment)
//   hmask   512 B : (Q3_K) lane l 16 B = 4 words; word sb>>2, bit 2*(sb&3)+h, byte b, same weight
//   scales 16 x 16 B (Q2_K) / 16 x 12 B (Q3_K), one row each, ggml order
//   d      16 x 4 B (Q2_K d, dmin) / 16 x 2 B (Q3_K d)
#pragma once
#include "v41_gemv.cuh"
#include "v41_kq_fast.cuh"

namespace v41g {

constexpr int KQM_ROWS_PER_CTA = 128;                 // 8 warps x one 16-row tile

// per staged token: x bytes in B-fragment order, then K/32 scales, then K/16 xm
__host__ __device__ constexpr size_t kqm_xb_bytes(int K) { return (size_t)K + 16; }
__host__ __device__ constexpr size_t kqm_stage_bytes(int K) { return kqm_xb_bytes(K) + (size_t)K / 32 * 4 + (size_t)K / 16 * 4; }

// x bytes of superblock s, lane t, sub-block sb at word (s*4 + t)*16 + sb
__device__ __forceinline__ void kqm_stage_row(uint8_t* dst, const XSrc& X, int row, int K) {
    if (X.stg) {                                      // prestaged by k_kqm_prep: the same bytes, copied
        const uint4* src = reinterpret_cast<const uint4*>(X.stg + (size_t)row * kqm_stage_bytes(K));
        uint4* d = reinterpret_cast<uint4*>(dst);
        for (int i = threadIdx.x; i < (int)(kqm_stage_bytes(K) / 16); i += blockDim.x) d[i] = src[i];
        return;
    }
    uint32_t* xb = reinterpret_cast<uint32_t*>(dst);
    float* xs = reinterpret_cast<float*>(dst + kqm_xb_bytes(K));
    float* xm = xs + K / 32;
    const uint32_t* src = reinterpret_cast<const uint32_t*>(X.xq + (size_t)row * X.ldx);
    for (int w = threadIdx.x; w < K / 4; w += blockDim.x) {
        const int k0 = 4 * w, s = k0 >> 8, r = k0 & 255, sb = r >> 4, t = (r >> 2) & 3;
        xb[(s * 4 + t) * 16 + sb] = src[w];
    }
    for (int c = threadIdx.x; c < K / 32; c += blockDim.x) xs[c] = X.xs[(size_t)row * (X.ldx / 32) + c];
    for (int g = threadIdx.x; g < K / 16; g += blockDim.x) {
        const uint8_t* q = X.xq + (size_t)row * X.ldx + g * 16;
        float s = 0.f;
        #pragma unroll
        for (int i = 0; i < 16; ++i) s += e4m3f(q[i]);
        xm[g] = s * X.xs[(size_t)row * (X.ldx / 32) + g / 2];
    }
}

// one CTA per row: the staged layout once into global memory, so k_kqm's ceil(N / 128) CTAs per unit
// copy it instead of each re-deriving the B-fragment order and the xm sums (prefill)
static __global__ void k_kqm_prep(uint8_t* dst, XSrc X, int K) {
    kqm_stage_row(dst + (size_t)blockIdx.x * kqm_stage_bytes(K), X, blockIdx.x, K);
}

__device__ __forceinline__ void mma_e4m3(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t b0) {
    d[0] = d[1] = d[2] = d[3] = 0.f;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3},{%4,%5},{%6},{%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a0), "r"(a1), "r"(b0));
}

// code-free term: m[n] += c[i] * xm[n][i] for i = 0..3 in order (one float4 of 4 consecutive sub-blocks per token)
template <int MT>
__device__ __forceinline__ void kqm_mt_acc(float (&m)[MT], const float (&c)[4], const float* xm, int stride) {
    #pragma unroll
    for (int n = 0; n < MT; ++n) {
        const float4 v = *reinterpret_cast<const float4*>(xm + n * stride);
        m[n] = fmaf(c[0], v.x, m[n]); m[n] = fmaf(c[1], v.y, m[n]); m[n] = fmaf(c[2], v.z, m[n]); m[n] = fmaf(c[3], v.w, m[n]);
    }
}

// grid (ceil(N / 128), max units); same unit contract as k_kq (MT = most tokens a unit may hold, <= 8,
// or a multiple of 8: NT = MT / 8 n8 tiles share each A fragment -- prefill units read an expert once
// for up to MT tokens; every column's fmaf sequence is the MT <= 8 one, so rows stay bitwise)
// RT: each warp runs RT 16-row tiles (grid x = ceil(N / (128 RT))), so prefill stages its MT token rows once
// per 128 RT rows (the 16-token stage is 113 KB at K = 5120, half the bytes of a 128-row weight slice).
// Per-element arithmetic is unchanged: bitwise for every RT.
// A1b: one superblock's weight words for one lane (codes, Q3_K high bits, this lane's two rows' scale words)
struct KqmLd { uint32_t cw[8], hw[4], sa[2], sb[2], sc[2]; };
template <int QT>
__device__ __forceinline__ void kqm_load(KqmLd& L, const uint8_t* W, size_t idx, size_t cnt, int lane, int g, int t) {
    const uint4* c4 = reinterpret_cast<const uint4*>(W + idx * 1024 + lane * 32);
    const uint4 a = c4[0], b = c4[1];
    L.cw[0] = a.x; L.cw[1] = a.y; L.cw[2] = a.z; L.cw[3] = a.w; L.cw[4] = b.x; L.cw[5] = b.y; L.cw[6] = b.z; L.cw[7] = b.w;
    if (QT == 3) {
        const uint4 h = *reinterpret_cast<const uint4*>(W + 1024 * cnt + idx * 512 + lane * 16);
        L.hw[0] = h.x; L.hw[1] = h.y; L.hw[2] = h.z; L.hw[3] = h.w;
    }
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
        const int r = g + 8 * h;
        if (QT == 2) {
            L.sa[h] = *reinterpret_cast<const uint32_t*>(W + 1024 * cnt + idx * 256 + r * 16 + 4 * t);
            L.sb[h] = *reinterpret_cast<const uint32_t*>(W + 1280 * cnt + idx * 64 + r * 4);
        } else {
            const uint32_t* s12 = reinterpret_cast<const uint32_t*>(W + 1536 * cnt + idx * 192 + r * 12);
            L.sa[h] = s12[t & 1]; L.sb[h] = s12[2];
            L.sc[h] = *reinterpret_cast<const unsigned short*>(W + 1728 * cnt + idx * 32 + r * 2);
        }
    }
}

// PF (A1b): the next superblock's weight words are loaded before this one's math (register double buffer);
// the arithmetic is the same instruction sequence on the same values, so PF changes no output bit.
template <int QT, int MT, class UnitT, int RT = 1, bool PF = false>
__global__ void __launch_bounds__(256) k_kqm(float* __restrict__ out, int ldo, XSrc X, bool x_by_pair,
                                             const UnitT* units, const int* units_n, const int* pair_tok,
                                             KQTab tab, int N, int K, int cmin = 1, int cmax = 1 << 30) {
    static_assert(QT == 2 || QT == 3, "Q2_K / Q3_K");
    static_assert(MT >= 1 && (MT <= 8 || MT % 8 == 0), "one n8 tile or whole n8 tiles");
    constexpr int NT = MT <= 8 ? 1 : MT / 8;
    const int u = blockIdx.y;
    if (u >= *units_n) return;
    const UnitT U = units[u];
    if (U.cnt < cmin || U.cnt > cmax) return;
    extern __shared__ __align__(16) uint8_t kqm_smem[];
    const size_t SB = kqm_stage_bytes(K);
    for (int n = 0; n < MT; ++n) {
        const int p = U.start + min(n, U.cnt - 1);
        kqm_stage_row(kqm_smem + n * SB, X, x_by_pair ? p : pair_tok[p], K);
    }
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    for (int rr = 0; rr < RT; ++rr) {
    const int tile = (blockIdx.x * RT + rr) * 8 + warp;
    if (tile * 16 >= N) return;
    const uint8_t* W; int lt, mrows;
    if (tab.pass == 0) {
        const bool up = tile * 16 >= v41::MOE_INTER;
        W = up ? tab.w3[U.e] : tab.w1[U.e]; lt = up ? tile - v41::MOE_INTER / 16 : tile; mrows = v41::MOE_INTER;
    } else { W = tab.w2[U.e]; lt = tile; mrows = N; }
    const int nsb = K / 256;
    const size_t cnt = (size_t)(mrows / 16) * nsb;
    // B source per tile j: token 8j + g for the mma (clamped), columns 8j + 2t (+1) for the scaling (clamped)
    const uint8_t* xg[NT]; const float* xsn[NT][2];
    #pragma unroll
    for (int j = 0; j < NT; ++j) {
        xg[j] = kqm_smem + min(8 * j + g, MT - 1) * SB;
        #pragma unroll
        for (int e = 0; e < 2; ++e)
            xsn[j][e] = reinterpret_cast<const float*>(kqm_smem + min(8 * j + 2 * t + e, MT - 1) * SB + kqm_xb_bytes(K));
    }
    const float* xm0 = reinterpret_cast<const float*>(kqm_smem + kqm_xb_bytes(K)) + K / 32;   // token n at + n * SB / 4
    constexpr int NE = MT == 1 ? 1 : 2;                  // C-fragment columns this lane can own per tile
    float acc[NT][2][NE], mt[2][MT];
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
        #pragma unroll
        for (int j = 0; j < NT; ++j)
            #pragma unroll
            for (int e = 0; e < NE; ++e) acc[j][h][e] = 0.f;
        #pragma unroll
        for (int n = 0; n < MT; ++n) mt[h][n] = 0.f;
    }
    const int src_q = lane & ~3;
    KqmLd cur, nxt;
    if (PF) kqm_load<QT>(cur, W, (size_t)lt * nsb, cnt, lane, g, t);
    for (int s = 0; s < nsb; ++s) {
        const size_t idx = (size_t)lt * nsb + s;
        if (PF) { if (s + 1 < nsb) kqm_load<QT>(nxt, W, idx + 1, cnt, lane, g, t); }
        else kqm_load<QT>(cur, W, idx, cnt, lane, g, t);
        const uint32_t* cw = cur.cw; const uint32_t* hw = cur.hw;
        // this lane's scales: sub-blocks 4t + i, rows g + 8h
        float scf[2][4];
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            if (QT == 2) {
                const uint32_t sw = cur.sa[h];
                const uint32_t dd = cur.sb[h];
                const float d = f16_lo(dd), dm = f16_lo(dd >> 16);
                float mm[4];
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    scf[h][i] = d * (float)((sw >> (8 * i)) & 0xF);
                    mm[i] = dm * (float)((sw >> (8 * i + 4)) & 0xF);
                }
                kqm_mt_acc<MT>(mt[h], mm, xm0 + s * 16 + 4 * t, (int)(SB / 4));
            } else {
                const uint32_t lo = cur.sa[h] >> (4 * (t >> 1)), hi = cur.sb[h] >> (2 * t);
                const float d = __half2float(__ushort_as_half((unsigned short)cur.sc[h]));
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int sc = (int)(((lo >> (8 * i)) & 0xF) | (((hi >> (8 * i)) & 3) << 4)) - 32;
                    scf[h][i] = d * (float)sc;
                }
                kqm_mt_acc<MT>(mt[h], scf[h], xm0 + s * 16 + 4 * t, (int)(SB / 4));
            }
        }
        uint32_t xb[NT][16];
        float xs[NT][2][8];
        #pragma unroll
        for (int j = 0; j < NT; ++j) {
            const uint4* x4 = reinterpret_cast<const uint4*>(xg[j] + (size_t)(s * 4 + t) * 64);
            #pragma unroll
            for (int i = 0; i < 4; ++i) { const uint4 v = x4[i]; xb[j][4 * i] = v.x; xb[j][4 * i + 1] = v.y; xb[j][4 * i + 2] = v.z; xb[j][4 * i + 3] = v.w; }
            #pragma unroll
            for (int e = 0; e < NE; ++e) {
                const float4* p = reinterpret_cast<const float4*>(xsn[j][e] + s * 8);
                const float4 a = p[0], b = p[1];
                xs[j][e][0] = a.x; xs[j][e][1] = a.y; xs[j][e][2] = a.z; xs[j][e][3] = a.w; xs[j][e][4] = b.x; xs[j][e][5] = b.y; xs[j][e][6] = b.z; xs[j][e][7] = b.w;
            }
        }
        #pragma unroll
        for (int sb = 0; sb < 16; ++sb) {
            const uint32_t w = cw[sb >> 1];
            uint32_t a0 = (w >> (4 * (sb & 1))) & 0x03030303u, a1 = (w >> (4 * (sb & 1) + 2)) & 0x03030303u;
            if (QT == 3) {
                const uint32_t hv = hw[sb >> 2] >> (2 * (sb & 3));
                a0 |= (hv & 0x01010101u) << 2; a1 |= ((hv >> 1) & 0x01010101u) << 2;
            }
            const float s0 = __shfl_sync(0xffffffffu, scf[0][sb & 3], src_q | (sb >> 2));
            const float s1 = __shfl_sync(0xffffffffu, scf[1][sb & 3], src_q | (sb >> 2));
            #pragma unroll
            for (int j = 0; j < NT; ++j) {
                float D[4];
                mma_e4m3(D, a0, a1, xb[j][sb]);
                #pragma unroll
                for (int e = 0; e < NE; ++e) {
                    acc[j][0][e] = fmaf(s0, D[e] * xs[j][e][sb >> 1], acc[j][0][e]);
                    acc[j][1][e] = fmaf(s1, D[2 + e] * xs[j][e][sb >> 1], acc[j][1][e]);
                }
            }
        }
        if (PF) cur = nxt;
    }
    // code-free term: quad reduction (fixed order), then every lane holds all MT columns
    #pragma unroll
    for (int h = 0; h < 2; ++h)
        #pragma unroll
        for (int n = 0; n < MT; ++n) {
            float v = mt[h][n];
            v += __shfl_xor_sync(0xffffffffu, v, 1);
            v += __shfl_xor_sync(0xffffffffu, v, 2);
            mt[h][n] = v;
        }
    #pragma unroll
    for (int j = 0; j < NT; ++j)
    #pragma unroll
    for (int h = 0; h < 2; ++h)
        #pragma unroll
        for (int e = 0; e < NE; ++e) {
            const int n = 8 * j + 2 * t + e, row = tile * 16 + g + 8 * h;
            if (n >= MT || n >= U.cnt || row >= N) continue;
            float m = 0.f;
            #pragma unroll
            for (int q = 0; q < MT; ++q) if (q == n) m = mt[h][q];
            const float v = QT == 2 ? fmaf(512.f, acc[j][h][e], -m) : fmaf(512.f, acc[j][h][e], -4.f * m);
            out[(size_t)(U.start + n) * ldo + row] = bf16r(v);
        }
    }
}

// ggml AoS blocks -> the layout above; one thread per (tile, superblock, lane)
static __global__ void k_kq_mma_repack(uint8_t* dst, const uint8_t* src, int rows, int nsb, int bits) {
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x, cnt = (size_t)(rows / 16) * nsb;
    if (i >= cnt * 32) return;
    const int l = (int)(i & 31), g = l >> 2, t = l & 3;
    const size_t idx = i >> 5;
    const int T = (int)(idx / nsb), s = (int)(idx % nsb);
    const int BB = bits == 2 ? 84 : 110, qoff = bits == 2 ? 16 : 32;
    uint32_t cw[8] = {0, 0, 0, 0, 0, 0, 0, 0}, hw[4] = {0, 0, 0, 0};
    for (int sb = 0; sb < 16; ++sb)
        for (int h = 0; h < 2; ++h) {
            const uint8_t* b = src + ((size_t)(T * 16 + g + 8 * h) * nsb + s) * BB;
            for (int bb = 0; bb < 4; ++bb) {
                const int k = sb * 16 + 4 * t + bb;
                const uint32_t q = (b[qoff + (k >> 7) * 32 + (k & 31)] >> (2 * ((k & 127) >> 5))) & 3;
                cw[sb >> 1] |= q << (8 * bb + 2 * (2 * (sb & 1) + h));
                if (bits == 3) hw[sb >> 2] |= (uint32_t)((b[k & 31] >> (k >> 5)) & 1) << (8 * bb + 2 * (sb & 3) + h);
            }
        }
    uint32_t* c = reinterpret_cast<uint32_t*>(dst + idx * 1024 + l * 32);
    for (int w = 0; w < 8; ++w) c[w] = cw[w];
    if (bits == 3) { uint32_t* hp = reinterpret_cast<uint32_t*>(dst + 1024 * cnt + idx * 512 + l * 16); for (int w = 0; w < 4; ++w) hp[w] = hw[w]; }
    if (l < 16) {
        const uint8_t* b = src + ((size_t)(T * 16 + l) * nsb + s) * BB;
        if (bits == 2) {
            for (int j = 0; j < 16; ++j) dst[1024 * cnt + idx * 256 + l * 16 + j] = b[j];
            for (int j = 0; j < 4; ++j) dst[1280 * cnt + idx * 64 + l * 4 + j] = b[80 + j];
        } else {
            for (int j = 0; j < 12; ++j) dst[1536 * cnt + idx * 192 + l * 12 + j] = b[96 + j];
            for (int j = 0; j < 2; ++j) dst[1728 * cnt + idx * 32 + l * 2 + j] = b[108 + j];
        }
    }
}

} // namespace v41g
