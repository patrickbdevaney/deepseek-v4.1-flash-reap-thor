// v41_attn.cu — CSA2: compressor pooling, packed fp4 caches, indexer scores, candidate blocks,
// top-k selection and the sparse MLA attention over window + compressed rows.
//
// CACHE FORMATS. The reference stores its compressed KV and index keys in bf16 buffers, but every
// value it writes there has just been through fp4_act_quant (inplace): an e2m1 code times a scale.
// So storing the code + scale is EXACT -- the same numbers, 288 B per 512-dim compressed row
// (vs 1 KiB bf16) and 68 B per 128-dim index key. At 1M context that is the difference between
// ~2.1 GiB and ~7 GiB of cache (docs/V41_DELTA.md has the table).
#include "v41_kernels.h"
#include "v41_config.h"
#include "v41_dev.cuh"
#include <cstdio>

using namespace v41d;

namespace v41k {

// --------------------------------------------------------------------------- compressor, ratio 2
// One CTA (512 threads = head_dim), rows in order: the 2-slot state is carried row to row.
__global__ void k_pool2(float* latent, int* valid, float* kvs, float* scs, const float* kv, const float* score,
                        const uint16_t* norm_w, const int* d_pos, int M, float* hist) {
    const int d = threadIdx.x;                           // 0..511
    __shared__ float red[16];
    for (int m = 0; m < M; ++m) {
        const int pos = d_pos[m], slot = pos & 1;
        kvs[slot * 512 + d] = kv[(size_t)m * 512 + d];
        scs[slot * 512 + d] = score[(size_t)m * 512 + d];
        if (hist) { hist[(size_t)m * 1024 + d] = kv[(size_t)m * 512 + d]; hist[(size_t)m * 1024 + 512 + d] = score[(size_t)m * 512 + d]; }
        const bool done = ((pos + 1) & 1) == 0;
        if (!done) { if (d == 0) valid[m] = 0; continue; }
        const float s0 = scs[d], s1 = scs[512 + d], mx = fmaxf(s0, s1);
        const float e0 = expf(s0 - mx), e1 = expf(s1 - mx), z = e0 + e1;
        const float pooled = bf16r(kvs[d] * (e0 / z) + kvs[512 + d] * (e1 / z));   // .to(bf16)
        float ss = warp_sum(pooled * pooled);
        if ((d & 31) == 0) red[d >> 5] = ss;
        __syncthreads();
        if (d < 32) { float v = d < 16 ? red[d] : 0.f; v = warp_sum(v); if (d == 0) red[0] = v; }
        __syncthreads();
        const float r = rsqrtf(red[0] / 512.f + v41::NORM_EPS);
        latent[(size_t)m * 512 + d] = bf16r(bf16f(norm_w[d]) * (pooled * r));
        if (d == 0) valid[m] = 1;
        __syncthreads();
    }
}
void compressor_pool2(float* latent, int* valid, float* kv_state, float* score_state, const float* kv,
                      const float* score, const uint16_t* norm_w, const int* d_pos, int M, cudaStream_t st, float* hist) {
    k_pool2<<<1, 512, 0, st>>>(latent, valid, kv_state, score_state, kv, score, norm_w, d_pos, M, hist);
}
// Speculative rollback: the 2-slot state as it was right after the row whose (kv, score) `hist_row`
// holds, at an EVEN position: slot 0 = that row, slot 1 is rewritten by the next (odd) row before use.
__global__ void k_pool2_restore(float* kvs, float* scs, const float* hist_row) {
    const int d = threadIdx.x;
    kvs[d] = hist_row[d];
    scs[d] = hist_row[512 + d];
}
void pool2_restore_even(float* kv_state, float* score_state, const float* hist_row, cudaStream_t st) {
    k_pool2_restore<<<1, 512, 0, st>>>(kv_state, score_state, hist_row);
}

// --------------------------------------------------------------------------- packed cache store
__global__ void k_store_packed(uint8_t* cc, uint8_t* cs, const uint8_t* codes, const uint8_t* scales, const int* valid,
                               const int* d_pos, int ratio, int cb, int sb) {
    const int m = blockIdx.x;
    if (valid && !valid[m]) return;
    const size_t slot = (size_t)(d_pos[m] / ratio);
    for (int i = threadIdx.x; i < cb; i += blockDim.x) cc[slot * cb + i] = codes[(size_t)m * cb + i];
    for (int i = threadIdx.x; i < sb; i += blockDim.x) cs[slot * sb + i] = scales[(size_t)m * sb + i];
}
void store_packed_rows(uint8_t* cache_codes, uint8_t* cache_scales, const uint8_t* codes, const uint8_t* scales,
                       const int* valid, const int* d_pos, int ratio, int M, int code_bytes, int scale_bytes,
                       cudaStream_t st) {
    k_store_packed<<<M, 64, 0, st>>>(cache_codes, cache_scales, codes, scales, valid, d_pos, ratio, code_bytes, scale_bytes);
}
__global__ void k_ring_store(float* ring_buf, int ring, const float* kv, const int* d_pos) {
    const int m = blockIdx.x;
    float* dst = ring_buf + (size_t)(d_pos[m] % ring) * v41::HEAD_DIM;
    for (int i = threadIdx.x; i < v41::HEAD_DIM; i += blockDim.x) dst[i] = kv[(size_t)m * v41::HEAD_DIM + i];
}
void ring_store(float* ring_buf, int ring, const float* kv, const int* d_pos, int M, cudaStream_t st) {
    k_ring_store<<<M, 128, 0, st>>>(ring_buf, ring, kv, d_pos);
}

// --------------------------------------------------------------------------- index scores
// index_score[t] = bf16( sum_h bf16( relu(bf16(q_h . k_t)) * w_h ) ) -- the rounding of the
// reference's bf16 einsum, relu_, bf16 multiply and bf16 sum. Positions >= n or outside a kept
// candidate block are -inf. grid (M, ceil(ld/32)); a CTA decodes 32 keys into smem.
static constexpr int IS_TILE = 32;
__global__ void __launch_bounds__(256) k_index_scores(float* score, const float* q, const float* w, const uint8_t* kc,
                                                      const uint8_t* ks, const uint8_t* cand, const int* d_pos, int ratio,
                                                      int ld, int ldc) {
    const int m = blockIdx.x, t0 = blockIdx.y * IS_TILE;
    if (t0 >= ld) return;
    const int n = (d_pos[m] + 1) / ratio;
    float* srow = score + (size_t)m * ld;
    __shared__ float qs[v41::INDEX_N_HEADS * v41::INDEX_HEAD_DIM];   // 16 KB
    __shared__ float kt[IS_TILE][v41::INDEX_HEAD_DIM + 1];            // 16.5 KB
    __shared__ float wsm[v41::INDEX_N_HEADS];
    if (t0 >= n) { for (int i = threadIdx.x; i < IS_TILE && t0 + i < ld; i += blockDim.x) srow[t0 + i] = -INFINITY; return; }
    bool any = true;
    if (cand) {                                          // skip tiles with no kept block (8 blocks/tile)
        any = false;
        for (int b = 0; b < IS_TILE / v41::CAND_BLOCK; ++b) {
            const int bi = t0 / v41::CAND_BLOCK + b;
            if (bi < ldc) any |= cand[(size_t)m * ldc + bi] != 0;
        }
    }
    if (!any) { for (int i = threadIdx.x; i < IS_TILE && t0 + i < ld; i += blockDim.x) srow[t0 + i] = -INFINITY; return; }
    for (int i = threadIdx.x; i < v41::INDEX_N_HEADS * v41::INDEX_HEAD_DIM; i += blockDim.x)
        qs[i] = q[(size_t)m * v41::INDEX_N_HEADS * v41::INDEX_HEAD_DIM + i];
    if (threadIdx.x < v41::INDEX_N_HEADS) wsm[threadIdx.x] = w[m * v41::INDEX_N_HEADS + threadIdx.x];
    for (int i = threadIdx.x; i < IS_TILE * 64; i += blockDim.x) {      // 64 code bytes per key
        const int r = i >> 6, b = i & 63, t = min(t0 + r, n - 1);
        const uint8_t byte = kc[(size_t)t * 64 + b];
        const float s = e8m0f(ks[(size_t)t * 4 + (b >> 4)]);
        kt[r][2 * b] = fp4f(byte & 15) * s; kt[r][2 * b + 1] = fp4f(byte >> 4) * s;
    }
    __syncthreads();
    // thread -> (key r = tid/8, heads hq*4 .. hq*4+3)
    const int r = threadIdx.x >> 3, hq = threadIdx.x & 7;
    float part = 0.f;
    for (int h = hq * 4; h < hq * 4 + 4; ++h) {
        const float* qh = qs + h * v41::INDEX_HEAD_DIM;
        float d = 0.f;
        #pragma unroll 8
        for (int k = 0; k < v41::INDEX_HEAD_DIM; ++k) d = fmaf(qh[k], kt[r][k], d);
        d = fmaxf(bf16r(d), 0.f);
        part += bf16r(d * wsm[h]);
    }
    part += __shfl_xor_sync(0xffffffffu, part, 1);
    part += __shfl_xor_sync(0xffffffffu, part, 2);
    part += __shfl_xor_sync(0xffffffffu, part, 4);
    const int t = t0 + r;
    if (hq == 0 && t < ld) {
        float v = t < n ? bf16r(part) : -INFINITY;
        if (cand && t < n && !cand[(size_t)m * ldc + t / v41::CAND_BLOCK]) v = -INFINITY;
        srow[t] = v;
    }
}
void index_scores(float* score, const float* q, const float* w, const uint8_t* kcodes, const uint8_t* kscales,
                  const uint8_t* cand, const int* d_pos, int ratio, int M, int ld_score, cudaStream_t st) {
    dim3 grid(M, (ld_score + IS_TILE - 1) / IS_TILE);
    const int ldc = (ld_score + v41::CAND_BLOCK - 1) / v41::CAND_BLOCK;
    k_index_scores<<<grid, 256, 0, st>>>(score, q, w, kcodes, kscales, cand, d_pos, ratio, ld_score, ldc);
}

// --------------------------------------------------------------------------- radix top-k
// Order-preserving float -> uint key (larger float -> larger key). -inf maps below every finite.
__device__ __forceinline__ uint32_t fkey(float f) {
    uint32_t b = __float_as_uint(f);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}
// One CTA per row: 3 passes of 11 bits find the k-th largest key T, then an ordered compaction
// emits every index with key > T plus the lowest-index ties at T. Ascending output order.
static constexpr int TK_THREADS = 1024;
__device__ void radix_threshold(const float* v, int n, int k, uint32_t& T, int& need_eq, uint32_t* hist) {
    uint32_t prefix = 0, pmask = 0; int remaining = k;
    const int shifts[3] = {21, 10, 0}, widths[3] = {11, 11, 10};
    for (int pass = 0; pass < 3; ++pass) {
        const int sh = shifts[pass], nb = 1 << widths[pass];
        for (int i = threadIdx.x; i < nb; i += blockDim.x) hist[i] = 0;
        __syncthreads();
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            const uint32_t key = fkey(v[i]);
            if ((key & pmask) == prefix) atomicAdd(&hist[(key >> sh) & (nb - 1)], 1u);   // integer: deterministic
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            int acc = 0, b = nb - 1;
            for (; b > 0; --b) { if (acc + (int)hist[b] >= remaining) break; acc += hist[b]; }
            remaining -= acc;
            hist[nb] = (uint32_t)b; hist[nb + 1] = (uint32_t)remaining;
        }
        __syncthreads();
        const uint32_t b = hist[nb];
        remaining = (int)hist[nb + 1];
        prefix |= b << sh; pmask |= (uint32_t)(nb - 1) << sh;
        __syncthreads();
    }
    T = prefix; need_eq = remaining;               // elements equal to T still to take
}
__device__ void ordered_emit(int* out, const float* v, int n, int k, uint32_t T, int need_eq, int offset,
                             uint32_t* sm) {
    // per-thread contiguous segments -> exclusive scans of (gt, eq) -> write
    const int seg = (n + blockDim.x - 1) / blockDim.x, lo = threadIdx.x * seg, hi = min(n, lo + seg);
    int gt = 0, eq = 0;
    for (int i = lo; i < hi; ++i) { const uint32_t key = fkey(v[i]); gt += key > T; eq += key == T; }
    sm[threadIdx.x] = gt; sm[blockDim.x + threadIdx.x] = eq;
    __syncthreads();
    if (threadIdx.x == 0) {
        uint32_t a = 0, b = 0;
        for (int t = 0; t < (int)blockDim.x; ++t) { uint32_t x = sm[t], y = sm[blockDim.x + t]; sm[t] = a; sm[blockDim.x + t] = b; a += x; b += y; }
    }
    __syncthreads();
    int eq_before = sm[blockDim.x + threadIdx.x];
    // position in output = (#gt before) + (#taken-eq before)
    int pos = sm[threadIdx.x] + min(eq_before, need_eq);
    for (int i = lo; i < hi; ++i) {
        const uint32_t key = fkey(v[i]);
        if (key > T) out[pos++] = i + offset;
        else if (key == T) { if (eq_before < need_eq) out[pos++] = i + offset; ++eq_before; }
    }
    __syncthreads();
}
__global__ void __launch_bounds__(TK_THREADS) k_topk(int* idxs, const float* score, const int* d_pos, int ratio, int ld,
                                                     int kmax, int offset) {
    const int m = blockIdx.x;
    __shared__ uint32_t sm[2 * TK_THREADS + 2050];
    const int n = (d_pos[m] + 1) / ratio;
    const int k = min(kmax, n);
    int* out = idxs + (size_t)m * kmax;
    if (k <= 0) { for (int i = threadIdx.x; i < kmax; i += blockDim.x) out[i] = -1; return; }
    uint32_t T; int need;
    radix_threshold(score + (size_t)m * ld, n, k, T, need, sm);
    __syncthreads();
    ordered_emit(out, score + (size_t)m * ld, n, k, T, need, offset, sm);
    for (int i = k + threadIdx.x; i < kmax; i += blockDim.x) out[i] = -1;
}
size_t topk_scratch_bytes(int M) { (void)M; return 0; }
void topk_rows(int* idxs, const float* score, const int* d_pos, int ratio, int M, int ld_score, int k, int offset,
               void* scratch, cudaStream_t st) {
    (void)scratch;
    k_topk<<<M, TK_THREADS, 0, st>>>(idxs, score, d_pos, ratio, ld_score, k, offset);
}

// --------------------------------------------------------------------------- candidate blocks
// select_candidate_blocks: block score = max over its 8 positions; the block holding the newest
// position is pinned (+inf); keep the top 2048 blocks whose score is > -inf.
__global__ void k_block_max(float* bs, const float* score, const int* d_pos, int ratio, int ld, int ldb) {
    const int m = blockIdx.y; const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= ldb) return;
    const int n = (d_pos[m] + 1) / ratio, nb = (n + v41::CAND_BLOCK - 1) / v41::CAND_BLOCK;
    float v = -INFINITY;
    if (b < nb) {
        for (int i = 0; i < v41::CAND_BLOCK; ++i) { const int t = b * v41::CAND_BLOCK + i; if (t < n) v = fmaxf(v, score[(size_t)m * ld + t]); }
        if (b == (n - 1) / v41::CAND_BLOCK) v = INFINITY;
    }
    bs[(size_t)m * ldb + b] = v;
}
__global__ void __launch_bounds__(TK_THREADS) k_cand_select(uint8_t* cand, const float* bs, const int* d_pos, int ratio,
                                                            int ldb) {
    const int m = blockIdx.x;
    __shared__ uint32_t sm[2 * TK_THREADS + 2050];
    const int n = (d_pos[m] + 1) / ratio, nb = (n + v41::CAND_BLOCK - 1) / v41::CAND_BLOCK;
    uint8_t* row = cand + (size_t)m * ldb;
    const float* b = bs + (size_t)m * ldb;
    if (nb <= v41::CAND_TOPK_BLOCKS) {
        for (int i = threadIdx.x; i < ldb; i += blockDim.x) row[i] = (i < nb && b[i] > -INFINITY) ? 1 : 0;
        return;
    }
    uint32_t T; int need;
    radix_threshold(b, nb, v41::CAND_TOPK_BLOCKS, T, need, sm);
    __syncthreads();
    // mark: key > T, or the lowest-index `need` ties; never a -inf block
    const int seg = (nb + blockDim.x - 1) / blockDim.x, lo = threadIdx.x * seg, hi = min(nb, lo + seg);
    int eq = 0; for (int i = lo; i < hi; ++i) eq += fkey(b[i]) == T;
    sm[threadIdx.x] = eq; __syncthreads();
    if (threadIdx.x == 0) { uint32_t a = 0; for (int t = 0; t < (int)blockDim.x; ++t) { uint32_t x = sm[t]; sm[t] = a; a += x; } }
    __syncthreads();
    int eb = sm[threadIdx.x];
    for (int i = lo; i < hi; ++i) {
        const uint32_t key = fkey(b[i]); uint8_t keep = 0;
        if (key > T) keep = 1; else if (key == T) { keep = eb < need; ++eb; }
        row[i] = keep && b[i] > -INFINITY;
    }
    for (int i = nb + threadIdx.x; i < ldb; i += blockDim.x) row[i] = 0;
}
size_t candidate_scratch_bytes(int M, int max_blocks) { return (size_t)M * max_blocks * sizeof(float); }
void candidate_blocks(uint8_t* cand, const float* score, const int* d_pos, int ratio, int M, int ld_score, int ld_cand,
                      void* scratch, cudaStream_t st) {
    float* bs = (float*)scratch;
    k_block_max<<<dim3((ld_cand + 255) / 256, M), 256, 0, st>>>(bs, score, d_pos, ratio, ld_score, ld_cand);
    k_cand_select<<<M, TK_THREADS, 0, st>>>(cand, bs, d_pos, ratio, ld_cand);
}

// --------------------------------------------------------------------------- sparse attention
// One CTA per (row m, 8 heads); warp = head. Entries: 128 window slots (positions pos-127..pos,
// ring slot pos % ring) then ncomp compressed indices. Two passes so that p = exp(s - max) uses the
// FINAL max before its bf16 cast, as the reference does (online rescaling would round differently).
static constexpr int SA_TILE = 8;
static constexpr int SA_MAXE = v41::WINDOW + v41::INDEX_TOPK;     // 640
__device__ __forceinline__ void sa_stage(float (*kv)[v41::HEAD_DIM], int e0, int ne, int pos, const float* ring_buf, int ring,
                                         const uint8_t* cc, const uint8_t* cs, const int* cidx, bool* ok) {
    for (int i = threadIdx.x; i < SA_TILE * v41::HEAD_DIM / 2; i += blockDim.x) {
        const int r = i / (v41::HEAD_DIM / 2), j = i % (v41::HEAD_DIM / 2), e = e0 + r;
        float a = 0.f, b = 0.f; bool valid = false;
        if (e < ne) {
            if (e < v41::WINDOW) {
                const int p = pos - (v41::WINDOW - 1) + e;
                if (p >= 0) { const float* src = ring_buf + (size_t)(p % ring) * v41::HEAD_DIM; a = src[2 * j]; b = src[2 * j + 1]; valid = true; }
            } else {
                const int t = cidx[e - v41::WINDOW];
                if (t >= 0) {
                    const uint8_t byte = cc[(size_t)t * 256 + j];
                    const float s = e4m3f(cs[(size_t)t * 32 + (2 * j) / v41::CKV_FP4_GRP]);
                    a = fp4f(byte & 15) * s; b = fp4f(byte >> 4) * s; valid = true;
                }
            }
        }
        kv[r][2 * j] = a; kv[r][2 * j + 1] = b;
        if (j == 0) ok[r] = valid;
    }
}
__global__ void __launch_bounds__(256) k_sparse_attn(float* o, const float* q, const float* ring_buf, int ring,
                                                     const uint8_t* cc, const uint8_t* cs, const int* cidx, int ncomp,
                                                     const float* sink, const int* d_pos, float scale) {
    const int m = blockIdx.x, h = blockIdx.y * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    const int pos = d_pos[m];
    const int ne = v41::WINDOW + ncomp;
    __shared__ float kv[SA_TILE][v41::HEAD_DIM];                       // 16 KB
    __shared__ float sc[8][SA_MAXE];                                   // 20 KB
    __shared__ bool ok[SA_TILE];
    float qr[16];
    const float* qh = q + ((size_t)m * v41::N_HEADS + h) * v41::HEAD_DIM;
    #pragma unroll
    for (int i = 0; i < 16; ++i) qr[i] = qh[lane + 32 * i];
    const int* ci = cidx ? cidx + (size_t)m * ncomp : nullptr;
    float mx = -INFINITY;
    for (int e0 = 0; e0 < ne; e0 += SA_TILE) {
        __syncthreads();
        sa_stage(kv, e0, ne, pos, ring_buf, ring, cc, cs, ci, ok);
        __syncthreads();
        for (int r = 0; r < SA_TILE && e0 + r < ne; ++r) {
            float d = 0.f;
            #pragma unroll
            for (int i = 0; i < 16; ++i) d = fmaf(qr[i], kv[r][lane + 32 * i], d);
            d = warp_sum(d) * scale;
            const float s = ok[r] ? d : -INFINITY;
            if (lane == 0) sc[threadIdx.x >> 5][e0 + r] = s;
            mx = fmaxf(mx, s);
        }
    }
    mx = fmaxf(mx, -1e30f);
    float acc[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) acc[i] = 0.f;
    float den = 0.f;
    for (int e0 = 0; e0 < ne; e0 += SA_TILE) {
        __syncthreads();
        sa_stage(kv, e0, ne, pos, ring_buf, ring, cc, cs, ci, ok);
        __syncthreads();
        for (int r = 0; r < SA_TILE && e0 + r < ne; ++r) {
            const float p = expf(sc[threadIdx.x >> 5][e0 + r] - mx);
            den += p;
            const float pb = bf16r(p);
            #pragma unroll
            for (int i = 0; i < 16; ++i) acc[i] = fmaf(pb, kv[r][lane + 32 * i], acc[i]);
        }
    }
    den += expf(sink[h] - mx);
    float* oh = o + ((size_t)m * v41::N_HEADS + h) * v41::HEAD_DIM;
    #pragma unroll
    for (int i = 0; i < 16; ++i) oh[lane + 32 * i] = bf16r(acc[i] / den);
}
void sparse_attn(float* o, const float* q, const float* win_ring, int ring, const uint8_t* ccodes, const uint8_t* cscales,
                 const int* cidx, int ncomp, int coff, const float* sink, const int* d_pos, int M, float scale,
                 cudaStream_t st) {
    (void)coff;
    k_sparse_attn<<<dim3(M, v41::N_HEADS / 8), 256, 0, st>>>(o, q, win_ring, ring, ccodes, cscales, cidx, ncomp, sink,
                                                            d_pos, scale);
}

} // namespace v41k
