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

// grid (SAM_E / 16, rows): 16 entries per CTA
__global__ void __launch_bounds__(256) k_sam_gather(uint16_t* Eb, int* nent_out, const float* ring_buf, int ring,
                                                    const uint8_t* cc, const uint8_t* cs, const int* cidx, int ncomp,
                                                    const int* d_pos) {
    const int m = blockIdx.y, k0 = blockIdx.x * 16, pos = d_pos[m];
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
    uint16_t* Em = Eb + (size_t)m * SAM_E * SAM_D;
    for (int i = threadIdx.x; i < 16 * SAM_D / 2; i += blockDim.x) {
        const int r = k0 + i / (SAM_D / 2), j = i % (SAM_D / 2);
        float a = 0.f, b = 0.f;                                  // rows past ne are zero (K3 multiplies them by p = 0)
        if (r < ne) {
            const int e = ent[r];
            if (e < v41::WINDOW) {
                const float* src = ring_buf + (size_t)((pos - (v41::WINDOW - 1) + e) % ring) * SAM_D;
                a = src[2 * j]; b = src[2 * j + 1];
            } else {
                const int t = ci[e - v41::WINDOW];
                const uint8_t byte = cc[(size_t)t * 256 + j];
                const float s = e4m3f(cs[(size_t)t * 32 + (2 * j) / v41::CKV_FP4_GRP]);
                a = fp4f(byte & 15) * s; b = fp4f(byte >> 4) * s;
            }
        }
        reinterpret_cast<uint32_t*>(Em + (size_t)r * SAM_D)[j] = (uint32_t)bf16_bits(a) | ((uint32_t)bf16_bits(b) << 16);
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

} // namespace v41g
