// v41_sa_mma.cuh — sparse attention on bf16 tensor cores (mma.m16n8k16), every M.
//
// WHY. Per row, V4.1 attention is MQA: 64 query heads x 512 dims against ONE 512-wide kv row per entry
// (128 window slots + up to 512 compressed). k_sparse_attn_c / the split form run that as per-head fp32
// dot products with a warp_sum per entry: 53% of an 8k prefill (k_sparse_attn_c ~0.6 TFLOPS).
// As GEMMs per row: S = Q[64x512] . E^T[512xne], O = bf16(P)[64xne] . E[nex512].
//
// EXACT PRODUCTS. q is bf16 (wq_b output rounded, rope rounded); a window kv row is fp8_sim'd to bf16;
// a compressed entry is fp4 x e4m3 (<= 5 significant bits); p is cast to bf16 before the PV product,
// as the reference does. So every product is exact in the fp32 accumulator and only the summation
// order differs from k_sparse_attn_c (T1). The softmax keeps the reference's two passes (p = exp(s -
// FINAL max), den over unrounded p). Every (row, head, dim) runs the same instructions at any M, so
// decode == verify == prefill stays bit for bit.
//
// K1 k_sam_gather: valid-entry list (same construction as k_sparse_attn_c), rows -> Eb[m][k][512] bf16
// K2 k_sam_scores: CTA (64-entry tile, row), 4 warps x 16 heads: S[m][h][k] = (Q . E^T) * scale
// K3 k_sam_out:    CTA (64-dim tile, row), 4 warps x 16 heads: softmax stats, P bf16 in smem, O = P . E
#pragma once
#include <cuda_bf16.h>

namespace v41g {

constexpr int SAM_ROWS = 64;                                    // rows per pass (scratch bound)
constexpr int SAM_E = v41::WINDOW + v41::INDEX_TOPK;            // 640 entries max
constexpr int SAM_D = v41::HEAD_DIM;                            // 512
constexpr int SAM_QS = SAM_D + 8;                               // smem row stride (bf16): 1040 B, ldmatrix conflict-free
constexpr int SAM_PS = SAM_E + 8;                               // P row stride (bf16)
constexpr int SAM_K3D = 64;                                     // K3 dims per CTA
constexpr int SAM_K3S = SAM_K3D + 8;                            // K3 E-chunk row stride (bf16)

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void mma_bf16(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint16_t bf16_bits(float v) { return __bfloat16_as_ushort(__float2bfloat16_rn(v)); }

// grid (SAM_E / SAM_GE, rows): SAM_GE entries per CTA (each CTA rebuilds the row's list with one warp)
#ifndef SAM_GE_DEF
#define SAM_GE_DEF 16
#endif
constexpr int SAM_GE = SAM_GE_DEF;
__global__ void __launch_bounds__(256) k_sam_gather(uint16_t* Eb, int* nent_out, const float* ring_buf, int ring,
                                                    const uint8_t* cc, const uint8_t* cs, const int* cidx, int ncomp,
                                                    const int* d_pos) {
    const int m = blockIdx.y, k0 = blockIdx.x * SAM_GE, pos = d_pos[m];
    __shared__ short ent[SAM_E];
    __shared__ int nent;
    const int* ci = cidx ? cidx + (size_t)m * ncomp : nullptr;
    if (threadIdx.x < 32) {
        const int lane = threadIdx.x;
        int n = 0;
        const int w0 = max(0, (v41::WINDOW - 1) - pos);
        for (int e = w0 + lane; e < v41::WINDOW; e += 32) ent[n + (e - w0)] = (short)e;
        n = v41::WINDOW - w0;
        for (int b = 0; b < ncomp; b += 32) {
            const int e = b + lane;
            const bool ok = e < ncomp && ci[e] >= 0;
            const unsigned bal = __ballot_sync(0xffffffffu, ok);
            if (ok) ent[n + __popc(bal & ((1u << lane) - 1u))] = (short)(v41::WINDOW + e);
            n += __popc(bal);
        }
        if (lane == 0) { nent = n; if (blockIdx.x == 0) nent_out[m] = n; }
    }
    __syncthreads();
    const int ne = nent;
    if (k0 >= ne) return;                                        // rows past ne are never read (K2/K3 zero-fill them)
    // resolve this tile's sources once (the copy below was latency-bound on ent -> cidx -> codes chains)
    __shared__ const void* srcp[SAM_GE];
    __shared__ const uint8_t* scp[SAM_GE];
    if (threadIdx.x < SAM_GE && k0 + (int)threadIdx.x < ne) {
        const int e = ent[k0 + threadIdx.x];
        if (e < v41::WINDOW) { srcp[threadIdx.x] = ring_buf + (size_t)((pos - (v41::WINDOW - 1) + e) % ring) * SAM_D; scp[threadIdx.x] = nullptr; }
        else { const int t = ci[e - v41::WINDOW]; srcp[threadIdx.x] = cc + (size_t)t * 256; scp[threadIdx.x] = cs + (size_t)t * 32; }
    }
    __syncthreads();
    uint16_t* Em = Eb + (size_t)m * SAM_E * SAM_D;
    // 8 values per item: two float4 of a window row, or 4 code bytes (one 16-value fp4 scale group spans 2 items)
    for (int i = threadIdx.x; i < SAM_GE * SAM_D / 8; i += blockDim.x) {
        const int rr = i / (SAM_D / 8), j8 = i % (SAM_D / 8), r = k0 + rr;
        if (r >= ne) continue;
        float v[8];
        if (!scp[rr]) {
            const float4* src = reinterpret_cast<const float4*>(srcp[rr]) + 2 * j8;
            const float4 x = src[0], y = src[1];
            v[0] = x.x; v[1] = x.y; v[2] = x.z; v[3] = x.w; v[4] = y.x; v[5] = y.y; v[6] = y.z; v[7] = y.w;
        } else {
            const uint32_t w = reinterpret_cast<const uint32_t*>(srcp[rr])[j8];
            const float sc = e4m3f(scp[rr][(8 * j8) / v41::CKV_FP4_GRP]);
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                const uint32_t byte = (w >> (8 * k)) & 0xff;
                v[2 * k] = fp4f(byte & 15) * sc; v[2 * k + 1] = fp4f(byte >> 4) * sc;
            }
        }
        uint4 o;
        o.x = (uint32_t)bf16_bits(v[0]) | ((uint32_t)bf16_bits(v[1]) << 16); o.y = (uint32_t)bf16_bits(v[2]) | ((uint32_t)bf16_bits(v[3]) << 16);
        o.z = (uint32_t)bf16_bits(v[4]) | ((uint32_t)bf16_bits(v[5]) << 16); o.w = (uint32_t)bf16_bits(v[6]) | ((uint32_t)bf16_bits(v[7]) << 16);
        *reinterpret_cast<uint4*>(Em + (size_t)r * SAM_D + 8 * j8) = o;
    }
}

// grid (SAM_E / 64, rows), 128 threads, dyn smem 2 * 64 * SAM_QS * 2 B
__global__ void __launch_bounds__(128) k_sam_scores(float* S, const uint16_t* Eb, const int* nent, const float* q, float scale) {
    const int m = blockIdx.y, k0 = blockIdx.x * 64, ne = nent[m];
    if (k0 >= ne) return;
    extern __shared__ __align__(16) uint16_t sam_smem[];
    uint16_t* Qs = sam_smem;                                     // [64 heads][SAM_QS]
    uint16_t* Es = sam_smem + 64 * SAM_QS;                       // [64 entries][SAM_QS]
    const float* qm = q + (size_t)m * v41::N_HEADS * SAM_D;
    for (int i = threadIdx.x; i < 64 * SAM_D / 4; i += blockDim.x) {
        const int h = i / (SAM_D / 4), c = (i % (SAM_D / 4)) * 4;
        const float4 v = *reinterpret_cast<const float4*>(qm + (size_t)h * SAM_D + c);
        uint2 u; u.x = (uint32_t)bf16_bits(v.x) | ((uint32_t)bf16_bits(v.y) << 16); u.y = (uint32_t)bf16_bits(v.z) | ((uint32_t)bf16_bits(v.w) << 16);
        *reinterpret_cast<uint2*>(Qs + h * SAM_QS + c) = u;
    }
    const uint16_t* Em = Eb + (size_t)m * SAM_E * SAM_D;
    for (int i = threadIdx.x; i < 64 * SAM_D / 8; i += blockDim.x) {
        const int r = i / (SAM_D / 8), c = (i % (SAM_D / 8)) * 8;
        uint4 v = make_uint4(0, 0, 0, 0);
        if (k0 + r < ne) v = *reinterpret_cast<const uint4*>(Em + (size_t)(k0 + r) * SAM_D + c);
        *reinterpret_cast<uint4*>(Es + r * SAM_QS + c) = v;
    }
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    float acc[8][4];
    #pragma unroll
    for (int j = 0; j < 8; ++j) { acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f; }
    const uint16_t* qa = Qs + (16 * warp + (lane & 15)) * SAM_QS + 8 * (lane >> 4);
    #pragma unroll 2
    for (int kk = 0; kk < SAM_D / 16; ++kk) {
        uint32_t a[4];
        ldsm_x4(a, qa + kk * 16);
        #pragma unroll
        for (int j = 0; j < 8; j += 2) {
            // matrices: (tile j, k lo), (tile j, k hi), (tile j+1, k lo), (tile j+1, k hi)
            const uint16_t* eb = Es + (8 * j + (lane & 7) + 8 * (lane >> 4)) * SAM_QS + kk * 16 + 8 * ((lane >> 3) & 1);
            uint32_t b[4];
            ldsm_x4(b, eb);
            mma_bf16(acc[j], a, b[0], b[1]);
            mma_bf16(acc[j + 1], a, b[2], b[3]);
        }
    }
    float* Sm = S + (size_t)m * v41::N_HEADS * SAM_E;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int k = k0 + 8 * j + 2 * t, h = 16 * warp + g;
        *reinterpret_cast<float2*>(Sm + (size_t)h * SAM_E + k) = make_float2(acc[j][0] * scale, acc[j][1] * scale);
        *reinterpret_cast<float2*>(Sm + (size_t)(h + 8) * SAM_E + k) = make_float2(acc[j][2] * scale, acc[j][3] * scale);
    }
}

// grid (SAM_D / SAM_K3D, rows), 128 threads, dyn smem 64 * SAM_PS * 2 + 64 * SAM_K3S * 2 B
__global__ void __launch_bounds__(128) k_sam_out(float* o, const float* S, const uint16_t* Eb, const int* nent, const float* sink) {
    const int m = blockIdx.y, d0 = blockIdx.x * SAM_K3D, ne = nent[m], nep = (ne + 15) & ~15;
    extern __shared__ __align__(16) uint16_t sam_smem[];
    uint16_t* Ps = sam_smem;                                     // [64 heads][SAM_PS] bf16(p)
    uint16_t* Ec = sam_smem + 64 * SAM_PS;                       // [64 entries][SAM_K3S]
    __shared__ float dens[64];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const float* Sm = S + (size_t)m * v41::N_HEADS * SAM_E;
    // softmax stats: warp w owns heads 16w..16w+15 (the heads its mma rows cover)
    for (int hh = 0; hh < 16; ++hh) {
        const int h = 16 * warp + hh;
        const float* sr = Sm + (size_t)h * SAM_E;
        float mx = -INFINITY;
        for (int k = lane; k < ne; k += 32) mx = fmaxf(mx, sr[k]);
        mx = fmaxf(warp_max(mx), -1e30f);
        float den = 0.f;
        for (int k = lane; k < nep; k += 32) {
            float p = 0.f;
            if (k < ne) { p = expf(sr[k] - mx); den += p; }
            Ps[h * SAM_PS + k] = bf16_bits(p);
        }
        den = warp_sum(den);
        if (lane == 0) dens[h] = den + expf(sink[h] - mx);
    }
    float acc[8][4];
    #pragma unroll
    for (int j = 0; j < 8; ++j) { acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f; }
    const uint16_t* Em = Eb + (size_t)m * SAM_E * SAM_D + d0;
    for (int c0 = 0; c0 < nep; c0 += 64) {
        __syncthreads();
        for (int i = threadIdx.x; i < 64 * SAM_K3D / 8; i += blockDim.x) {
            const int r = i / (SAM_K3D / 8), c = (i % (SAM_K3D / 8)) * 8;
            uint4 v = make_uint4(0, 0, 0, 0);
            if (c0 + r < ne) v = *reinterpret_cast<const uint4*>(Em + (size_t)(c0 + r) * SAM_D + c);
            *reinterpret_cast<uint4*>(Ec + r * SAM_K3S + c) = v;
        }
        __syncthreads();
        const int kn = min(64, nep - c0);
        for (int ks = 0; ks < kn; ks += 16) {
            uint32_t a[4];
            ldsm_x4(a, Ps + (16 * warp + (lane & 15)) * SAM_PS + c0 + ks + 8 * (lane >> 4));
            #pragma unroll
            for (int j = 0; j < 8; j += 2) {
                // .trans over [entry][dim]: matrices (k lo, tile j), (k hi, tile j), (k lo, tile j+1), (k hi, tile j+1)
                const uint16_t* eb = Ec + (ks + (lane & 7) + 8 * ((lane >> 3) & 1)) * SAM_K3S + 8 * j + 8 * (lane >> 4);
                uint32_t b[4];
                ldsm_x4_t(b, eb);
                mma_bf16(acc[j], a, b[0], b[1]);
                mma_bf16(acc[j + 1], a, b[2], b[3]);
            }
        }
    }
    __syncthreads();
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int d = d0 + 8 * j + 2 * t, h = 16 * warp + g;
        const float i0 = dens[h], i1 = dens[h + 8];
        float* o0 = o + ((size_t)m * v41::N_HEADS + h) * SAM_D + d;
        float* o1 = o + ((size_t)m * v41::N_HEADS + h + 8) * SAM_D + d;
        *reinterpret_cast<float2*>(o0) = make_float2(bf16r(acc[j][0] / i0), bf16r(acc[j][1] / i0));
        *reinterpret_cast<float2*>(o1) = make_float2(bf16r(acc[j][2] / i1), bf16r(acc[j][3] / i1));
    }
}

// ---- prefill form (P7a, bitwise with K2 + K3 above). Per row K2 restages the fp32 q in all SAM_E / 64
// entry CTAs and K3 re-reads the whole fp32 S row in all SAM_D / 64 dim CTAs: ~2.6 of the ~4.9 MB a row
// moves. K2R: one CTA per row stages q once, loops the entry tiles, writes S, then runs K3's softmax
// (same per-lane order, same warp_max / warp_sum) and writes bf16 P + den. K3P: K3 reading P + den.
// Every S element is the same mma chain over kk; every P, den and O element the same instructions.

// grid (rows), 256 threads: warp w = heads 16 (w & 3) .. +15, entries 32 (w >> 2) .. +31 of each 64-tile;
// dyn smem 2 * 64 * SAM_QS * 2 B (q, entry tile)
__global__ void __launch_bounds__(256) k_sam_scores_row(float* S, uint16_t* P, float* den_out, const uint16_t* Eb,
                                                        const int* nent, const float* q, const float* sink, float scale) {
    const int m = blockIdx.x, ne = nent[m], nep = (ne + 15) & ~15;
    extern __shared__ __align__(16) uint16_t sam_smem[];
    uint16_t* Qs = sam_smem;
    uint16_t* Es = sam_smem + 64 * SAM_QS;
    const float* qm = q + (size_t)m * v41::N_HEADS * SAM_D;
    for (int i = threadIdx.x; i < 64 * SAM_D / 4; i += blockDim.x) {
        const int h = i / (SAM_D / 4), c = (i % (SAM_D / 4)) * 4;
        const float4 v = *reinterpret_cast<const float4*>(qm + (size_t)h * SAM_D + c);
        uint2 u; u.x = (uint32_t)bf16_bits(v.x) | ((uint32_t)bf16_bits(v.y) << 16); u.y = (uint32_t)bf16_bits(v.z) | ((uint32_t)bf16_bits(v.w) << 16);
        *reinterpret_cast<uint2*>(Qs + h * SAM_QS + c) = u;
    }
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3, hg = warp & 3, eh = warp >> 2;
    const uint16_t* Em = Eb + (size_t)m * SAM_E * SAM_D;
    float* Sm = S + (size_t)m * v41::N_HEADS * SAM_E;
    const uint16_t* qa = Qs + (16 * hg + (lane & 15)) * SAM_QS + 8 * (lane >> 4);
    for (int k0 = 0; k0 < ne; k0 += 64) {
        __syncthreads();                                  // q staged / previous tile consumed
        for (int i = threadIdx.x; i < 64 * SAM_D / 8; i += blockDim.x) {
            const int r = i / (SAM_D / 8), c = (i % (SAM_D / 8)) * 8;
            uint4 v = make_uint4(0, 0, 0, 0);
            if (k0 + r < ne) v = *reinterpret_cast<const uint4*>(Em + (size_t)(k0 + r) * SAM_D + c);
            *reinterpret_cast<uint4*>(Es + r * SAM_QS + c) = v;
        }
        __syncthreads();
        float acc[4][4];
        #pragma unroll
        for (int j = 0; j < 4; ++j) { acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f; }
        #pragma unroll 2
        for (int kk = 0; kk < SAM_D / 16; ++kk) {
            uint32_t a[4];
            ldsm_x4(a, qa + kk * 16);
            #pragma unroll
            for (int j = 0; j < 4; j += 2) {
                const uint16_t* eb = Es + (32 * eh + 8 * j + (lane & 7) + 8 * (lane >> 4)) * SAM_QS + kk * 16 + 8 * ((lane >> 3) & 1);
                uint32_t b[4];
                ldsm_x4(b, eb);
                mma_bf16(acc[j], a, b[0], b[1]);
                mma_bf16(acc[j + 1], a, b[2], b[3]);
            }
        }
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int k = k0 + 32 * eh + 8 * j + 2 * t, h = 16 * hg + g;
            *reinterpret_cast<float2*>(Sm + (size_t)h * SAM_E + k) = make_float2(acc[j][0] * scale, acc[j][1] * scale);
            *reinterpret_cast<float2*>(Sm + (size_t)(h + 8) * SAM_E + k) = make_float2(acc[j][2] * scale, acc[j][3] * scale);
        }
    }
    __syncthreads();                                      // the block's S writes are visible to the block
    uint16_t* Pm = P + (size_t)m * v41::N_HEADS * SAM_PS;
    for (int h = warp; h < v41::N_HEADS; h += 8) {        // K3's softmax, verbatim
        const float* sr = Sm + (size_t)h * SAM_E;
        float mx = -INFINITY;
        for (int k = lane; k < ne; k += 32) mx = fmaxf(mx, sr[k]);
        mx = fmaxf(warp_max(mx), -1e30f);
        float den = 0.f;
        for (int k = lane; k < nep; k += 32) {
            float p = 0.f;
            if (k < ne) { p = expf(sr[k] - mx); den += p; }
            Pm[(size_t)h * SAM_PS + k] = bf16_bits(p);
        }
        den = warp_sum(den);
        if (lane == 0) den_out[(size_t)m * v41::N_HEADS + h] = den + expf(sink[h] - mx);
    }
}

// grid (SAM_D / SAM_K3PD, rows), 256 threads: warp w = heads 16 (w & 3) .., dims 64 (w >> 2) .. of the CTA's 128;
// dyn smem SAM_K3P_SMEM. K3 with P + den from k_sam_scores_row; P staged once per 128 dims (K3: per 64)
constexpr int SAM_K3PD = 128, SAM_K3PS = SAM_K3PD + 8;
constexpr int SAM_K3P_SMEM = 64 * SAM_PS * 2 + 64 * SAM_K3PS * 2;
__global__ void __launch_bounds__(256) k_sam_out_p(float* o, const uint16_t* P, const float* den_in, const uint16_t* Eb,
                                                   const int* nent) {
    const int m = blockIdx.y, d0 = blockIdx.x * SAM_K3PD, ne = nent[m], nep = (ne + 15) & ~15;
    extern __shared__ __align__(16) uint16_t sam_smem[];
    uint16_t* Ps = sam_smem;
    uint16_t* Ec = sam_smem + 64 * SAM_PS;
    const uint16_t* Pm = P + (size_t)m * v41::N_HEADS * SAM_PS;
    for (int i = threadIdx.x; i < 64 * (nep / 8); i += blockDim.x) {
        const int h = i / (nep / 8), c = (i % (nep / 8)) * 8;
        *reinterpret_cast<uint4*>(Ps + h * SAM_PS + c) = *reinterpret_cast<const uint4*>(Pm + (size_t)h * SAM_PS + c);
    }
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3, hg = warp & 3, dh = warp >> 2;
    float acc[8][4];
    #pragma unroll
    for (int j = 0; j < 8; ++j) { acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f; }
    const uint16_t* Em = Eb + (size_t)m * SAM_E * SAM_D + d0;
    for (int c0 = 0; c0 < nep; c0 += 64) {
        __syncthreads();
        for (int i = threadIdx.x; i < 64 * SAM_K3PD / 8; i += blockDim.x) {
            const int r = i / (SAM_K3PD / 8), c = (i % (SAM_K3PD / 8)) * 8;
            uint4 v = make_uint4(0, 0, 0, 0);
            if (c0 + r < ne) v = *reinterpret_cast<const uint4*>(Em + (size_t)(c0 + r) * SAM_D + c);
            *reinterpret_cast<uint4*>(Ec + r * SAM_K3PS + c) = v;
        }
        __syncthreads();
        const int kn = min(64, nep - c0);
        for (int ks = 0; ks < kn; ks += 16) {
            uint32_t a[4];
            ldsm_x4(a, Ps + (16 * hg + (lane & 15)) * SAM_PS + c0 + ks + 8 * (lane >> 4));
            #pragma unroll
            for (int j = 0; j < 8; j += 2) {
                const uint16_t* eb = Ec + (ks + (lane & 7) + 8 * ((lane >> 3) & 1)) * SAM_K3PS + 64 * dh + 8 * j + 8 * (lane >> 4);
                uint32_t b[4];
                ldsm_x4_t(b, eb);
                mma_bf16(acc[j], a, b[0], b[1]);
                mma_bf16(acc[j + 1], a, b[2], b[3]);
            }
        }
    }
    const float* dm = den_in + (size_t)m * v41::N_HEADS;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int d = d0 + 64 * dh + 8 * j + 2 * t, h = 16 * hg + g;
        const float i0 = dm[h], i1 = dm[h + 8];
        float* o0 = o + ((size_t)m * v41::N_HEADS + h) * SAM_D + d;
        float* o1 = o + ((size_t)m * v41::N_HEADS + h + 8) * SAM_D + d;
        *reinterpret_cast<float2*>(o0) = make_float2(bf16r(acc[j][0] / i0), bf16r(acc[j][1] / i0));
        *reinterpret_cast<float2*>(o1) = make_float2(bf16r(acc[j][2] / i1), bf16r(acc[j][3] / i1));
    }
}

} // namespace v41g
