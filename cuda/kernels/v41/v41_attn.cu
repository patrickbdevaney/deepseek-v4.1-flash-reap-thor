// v41_attn.cu — CSA2: compressor pooling, packed fp4 caches, indexer scores, candidate blocks,
// top-k selection and the sparse MLA attention over window + compressed rows.
//
// CACHE FORMATS. The reference stores its compressed KV and index keys in bf16 buffers, but every
// value it writes there has just been through fp4_act_quant (inplace): an e2m1 code times a scale.
// So storing the code + scale is EXACT -- the same numbers, 288 B per 512-dim compressed row
// (vs 1 KiB bf16) and 68 B per 128-dim index key. At 1M context that is the difference between
// ~2.1 GiB and ~7 GiB of cache (docs/V41_DELTA.md has the table).
#include <cstdlib>
#include <cub/block/block_scan.cuh>
#include "v41_kernels.h"
#include "v41_config.h"
#include "v41_dev.cuh"
namespace v41g { using namespace v41d; }
#include "v41_sa_mma.cuh"
#include "v41_is_mma.cuh"
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
bool is_mma_on() { static const bool on = [] { const char* e = getenv("V41_IS_MMA"); return !(e && *e == '0'); }(); return on; }   // P3 default (bitwise); V41_IS_MMA=0 = scalar
void index_scores(float* score, const float* q, const float* w, const uint8_t* kcodes, const uint8_t* kscales,
                  const uint8_t* cand, const int* d_pos, int ratio, int M, int ld_score, cudaStream_t st,
                  const uint8_t* qcodes, const uint8_t* qscales) {
    const int ldc = (ld_score + v41::CAND_BLOCK - 1) / v41::CAND_BLOCK;
    if (qcodes && is_mma_on()) {
        dim3 g((M + v41g::IM_ROWS - 1) / v41g::IM_ROWS, (ld_score + v41g::IM_KEYS - 1) / v41g::IM_KEYS);
        v41g::k_index_mma<v41::INDEX_N_HEADS, v41::INDEX_HEAD_DIM><<<g, 256, 0, st>>>(score, qcodes, qscales, w, kcodes, kscales,
                                                                                    cand, d_pos, ratio, M, ld_score, ldc, v41::CAND_BLOCK);
        return;
    }
    dim3 grid(M, (ld_score + IS_TILE - 1) / IS_TILE);
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
using TKScan = cub::BlockScan<int, TK_THREADS>;
static int g_topk_serial = -1;                  // V41_TOPK_OLD=1: thread-0 scans (the reference form)
bool topk_serial() { if (g_topk_serial < 0) { const char* e = getenv("V41_TOPK_OLD"); g_topk_serial = (e && *e == '1') ? 1 : 0; } return g_topk_serial; }
void topk_serial_set(bool on) { g_topk_serial = on ? 1 : 0; }
__device__ void radix_threshold(const float* v, int n, int k, uint32_t& T, int& need_eq, uint32_t* hist,
                                bool serial, typename TKScan::TempStorage* scan) {
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
        if (serial) {
            if (threadIdx.x == 0) {
                int acc = 0, b = nb - 1;
                for (; b > 0; --b) { if (acc + (int)hist[b] >= remaining) break; acc += hist[b]; }
                remaining -= acc;
                hist[nb] = (uint32_t)b; hist[nb + 1] = (uint32_t)remaining;
            }
        } else {
            // the same search as a block scan: bins in descending order, `per` per thread; the bin
            // where the running count first reaches `remaining` (b >= 1), else b = 0
            const int per = nb / TK_THREADS;
            int loc[2] = {0, 0}, sum = 0;
            for (int q = 0; q < per; ++q) { loc[q] = (int)hist[nb - 1 - (threadIdx.x * per + q)]; sum += loc[q]; }
            int acc;
            TKScan(*scan).ExclusiveSum(sum, acc);
            for (int q = 0; q < per; ++q) {
                const int b = nb - 1 - (threadIdx.x * per + q);
                if (b > 0 && acc < remaining && acc + loc[q] >= remaining) { hist[nb] = (uint32_t)b; hist[nb + 1] = (uint32_t)(remaining - acc); }
                if (b == 0 && acc < remaining) { hist[nb] = 0u; hist[nb + 1] = (uint32_t)(remaining - acc); }
                acc += loc[q];
            }
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
                             uint32_t* sm, bool serial, typename TKScan::TempStorage* scan) {
    // per-thread contiguous segments -> exclusive scans of (gt, eq) -> write
    const int seg = (n + blockDim.x - 1) / blockDim.x, lo = threadIdx.x * seg, hi = min(n, lo + seg);
    int gt = 0, eq = 0;
    for (int i = lo; i < hi; ++i) { const uint32_t key = fkey(v[i]); gt += key > T; eq += key == T; }
    sm[threadIdx.x] = gt; sm[blockDim.x + threadIdx.x] = eq;
    __syncthreads();
    if (serial) {
        if (threadIdx.x == 0) {
            uint32_t a = 0, b = 0;
            for (int t = 0; t < (int)blockDim.x; ++t) { uint32_t x = sm[t], y = sm[blockDim.x + t]; sm[t] = a; sm[blockDim.x + t] = b; a += x; b += y; }
        }
    } else {
        int a, b;
        TKScan(*scan).ExclusiveSum(gt, a);
        __syncthreads();
        TKScan(*scan).ExclusiveSum(eq, b);
        sm[threadIdx.x] = a; sm[blockDim.x + threadIdx.x] = b;
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
                                                     int kmax, int offset, bool serial) {
    const int m = blockIdx.x;
    __shared__ uint32_t sm[2 * TK_THREADS + 2050];
    const int n = (d_pos[m] + 1) / ratio;
    const int k = min(kmax, n);
    int* out = idxs + (size_t)m * kmax;
    if (k <= 0) { for (int i = threadIdx.x; i < kmax; i += blockDim.x) out[i] = -1; return; }
    if (k == n && !serial) {                       // every candidate is taken: ascending 0..n-1
        for (int i = threadIdx.x; i < n; i += blockDim.x) out[i] = i + offset;
        for (int i = k + threadIdx.x; i < kmax; i += blockDim.x) out[i] = -1;
        return;
    }
    __shared__ typename TKScan::TempStorage scan;
    uint32_t T; int need;
    radix_threshold(score + (size_t)m * ld, n, k, T, need, sm, serial, &scan);
    __syncthreads();
    ordered_emit(out, score + (size_t)m * ld, n, k, T, need, offset, sm, serial, &scan);
    for (int i = k + threadIdx.x; i < kmax; i += blockDim.x) out[i] = -1;
}
size_t topk_scratch_bytes(int M) { (void)M; return 0; }
void topk_rows(int* idxs, const float* score, const int* d_pos, int ratio, int M, int ld_score, int k, int offset,
               void* scratch, cudaStream_t st) {
    (void)scratch;
    k_topk<<<M, TK_THREADS, 0, st>>>(idxs, score, d_pos, ratio, ld_score, k, offset, topk_serial());
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
                                                            int ldb, bool serial) {
    const int m = blockIdx.x;
    __shared__ uint32_t sm[2 * TK_THREADS + 2050];
    const int n = (d_pos[m] + 1) / ratio, nb = (n + v41::CAND_BLOCK - 1) / v41::CAND_BLOCK;
    uint8_t* row = cand + (size_t)m * ldb;
    const float* b = bs + (size_t)m * ldb;
    if (nb <= v41::CAND_TOPK_BLOCKS) {
        for (int i = threadIdx.x; i < ldb; i += blockDim.x) row[i] = (i < nb && b[i] > -INFINITY) ? 1 : 0;
        return;
    }
    __shared__ typename TKScan::TempStorage scan;
    uint32_t T; int need;
    radix_threshold(b, nb, v41::CAND_TOPK_BLOCKS, T, need, sm, serial, &scan);
    __syncthreads();
    // mark: key > T, or the lowest-index `need` ties; never a -inf block
    const int seg = (nb + blockDim.x - 1) / blockDim.x, lo = threadIdx.x * seg, hi = min(nb, lo + seg);
    int eq = 0; for (int i = lo; i < hi; ++i) eq += fkey(b[i]) == T;
    if (serial) {
        sm[threadIdx.x] = eq; __syncthreads();
        if (threadIdx.x == 0) { uint32_t a = 0; for (int t = 0; t < (int)blockDim.x; ++t) { uint32_t x = sm[t]; sm[t] = a; a += x; } }
    } else {
        int a; TKScan(scan).ExclusiveSum(eq, a); sm[threadIdx.x] = a;
    }
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
    k_cand_select<<<M, TK_THREADS, 0, st>>>(cand, bs, d_pos, ratio, ld_cand, topk_serial());
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
// Compacted form, bit-identical to k_sparse_attn: an invalid entry (window slot before position 0,
// compressed index -1) scores -inf, so its p is exactly 0 and it adds exactly nothing to den or acc.
// Each CTA first lists the valid entries IN ORDER (one warp, ballot + popc), then runs the same two
// passes over that list only. At short context k_sparse_attn walked all 640 entries (~80% masked);
// HPC heads per CTA (4) give 16 CTAs at M=1 instead of 8 (the kv tile is staged per CTA either way).
template <int HPC>
__global__ void __launch_bounds__(HPC * 32) k_sparse_attn_c(float* o, const float* q, const float* ring_buf, int ring,
                                                          const uint8_t* cc, const uint8_t* cs, const int* cidx, int ncomp,
                                                          const float* sink, const int* d_pos, float scale) {
    const int m = blockIdx.x, h = blockIdx.y * HPC + (threadIdx.x >> 5), lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
    const int pos = d_pos[m];
    __shared__ float kv[SA_TILE][v41::HEAD_DIM];
    __shared__ float sc[HPC][SA_MAXE];
    __shared__ short ent[SA_MAXE];
    __shared__ int nent;
    const int* ci = cidx ? cidx + (size_t)m * ncomp : nullptr;
    if (threadIdx.x < 32) {
        int n = 0;
        const int w0 = max(0, (v41::WINDOW - 1) - pos);              // window slots with p = pos-127+e >= 0
        for (int e = w0 + lane; e < v41::WINDOW; e += 32) ent[n + (e - w0)] = (short)e;
        n = v41::WINDOW - w0;
        for (int b = 0; b < ncomp; b += 32) {
            const int e = b + lane;
            const bool ok = e < ncomp && ci[e] >= 0;
            const unsigned bal = __ballot_sync(0xffffffffu, ok);
            if (ok) ent[n + __popc(bal & ((1u << lane) - 1u))] = (short)(v41::WINDOW + e);
            n += __popc(bal);
        }
        if (lane == 0) nent = n;
    }
    __syncthreads();
    const int ne = nent;
    float qr[16];
    const float* qh = q + ((size_t)m * v41::N_HEADS + h) * v41::HEAD_DIM;
    #pragma unroll
    for (int i = 0; i < 16; ++i) qr[i] = qh[lane + 32 * i];
    auto stage = [&](int k0) {
        for (int i = threadIdx.x; i < SA_TILE * v41::HEAD_DIM / 2; i += blockDim.x) {
            const int r = i / (v41::HEAD_DIM / 2), j = i % (v41::HEAD_DIM / 2);
            float a = 0.f, b = 0.f;
            if (k0 + r < ne) {
                const int e = ent[k0 + r];
                if (e < v41::WINDOW) {
                    const float* src = ring_buf + (size_t)((pos - (v41::WINDOW - 1) + e) % ring) * v41::HEAD_DIM;
                    a = src[2 * j]; b = src[2 * j + 1];
                } else {
                    const int t = ci[e - v41::WINDOW];
                    const uint8_t byte = cc[(size_t)t * 256 + j];
                    const float s = e4m3f(cs[(size_t)t * 32 + (2 * j) / v41::CKV_FP4_GRP]);
                    a = fp4f(byte & 15) * s; b = fp4f(byte >> 4) * s;
                }
            }
            kv[r][2 * j] = a; kv[r][2 * j + 1] = b;
        }
    };
    float mx = -INFINITY;
    for (int k0 = 0; k0 < ne; k0 += SA_TILE) {
        __syncthreads(); stage(k0); __syncthreads();
        for (int r = 0; r < SA_TILE && k0 + r < ne; ++r) {
            float d = 0.f;
            #pragma unroll
            for (int i = 0; i < 16; ++i) d = fmaf(qr[i], kv[r][lane + 32 * i], d);
            d = warp_sum(d) * scale;
            if (lane == 0) sc[wp][k0 + r] = d;
            mx = fmaxf(mx, d);
        }
    }
    mx = fmaxf(mx, -1e30f);
    float acc[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) acc[i] = 0.f;
    float den = 0.f;
    for (int k0 = 0; k0 < ne; k0 += SA_TILE) {
        __syncthreads(); stage(k0); __syncthreads();
        for (int r = 0; r < SA_TILE && k0 + r < ne; ++r) {
            const float p = expf(sc[wp][k0 + r] - mx);
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
// Split form for the decode/verify widths (M <= SA_SPLIT_M), bit-identical to k_sparse_attn_c. All 64
// heads read the same 512-wide kv row (one kv head), so the rows are decoded ONCE into an fp32 scratch
// and the two passes are spread over (head, entry) and (head, dim) instead of 16 CTAs each restaging
// everything twice. Order is kept where it matters: a score is the same per-lane fma chain + warp_sum;
// max is order-free; den and every acc[h][d] are sums over the entries in list order.
static constexpr int SA_SPLIT_M = 8;
static float* g_sa_e = nullptr;  static float* g_sa_s = nullptr;  static int* g_sa_n = nullptr;  static int g_sa_m = 0;
// tensor-core path scratch (v41_sa_mma.cuh): SAM_ROWS rows of bf16 entries + fp32 scores
static uint16_t* g_sam_e = nullptr; static float* g_sam_s = nullptr; static int* g_sam_n = nullptr;
// P7a row form (k_sam_scores_row + k_sam_out_p, bitwise with K2 + K3) for passes of >= g_sam_row_min rows:
// one CTA per row is too few CTAs for decode / verify widths, which keep K2 + K3
static uint16_t* g_sam_p = nullptr; static float* g_sam_d = nullptr;
static int g_sam_row_min = [] { const char* e = getenv("V41_SAM_ROW_MIN"); return e ? atoi(e) : 16; }();
void sam_row_set(int min_m) { g_sam_row_min = min_m; }
static constexpr int SAM_K2_SMEM = 2 * 64 * v41g::SAM_QS * 2;
static constexpr int SAM_K3_SMEM = 64 * v41g::SAM_PS * 2 + 64 * v41g::SAM_K3S * 2;
void sparse_attn_reserve(int max_m) {
    if (!g_sam_e) {
        const int R = v41g::SAM_ROWS;
        if (cudaMalloc(&g_sam_e, (size_t)R * v41g::SAM_E * v41g::SAM_D * 2) == cudaSuccess &&
            cudaMalloc(&g_sam_s, (size_t)R * v41::N_HEADS * v41g::SAM_E * 4) == cudaSuccess &&
            cudaMalloc(&g_sam_n, (size_t)R * 4) == cudaSuccess) {
            cudaMemset(g_sam_e, 0, (size_t)R * v41g::SAM_E * v41g::SAM_D * 2);
            if (cudaMalloc(&g_sam_p, (size_t)R * v41::N_HEADS * v41g::SAM_PS * 2) != cudaSuccess ||
                cudaMalloc(&g_sam_d, (size_t)R * v41::N_HEADS * 4) != cudaSuccess) { g_sam_p = nullptr; (void)cudaGetLastError(); }
            cudaFuncSetAttribute(v41g::k_sam_scores_row, cudaFuncAttributeMaxDynamicSharedMemorySize, SAM_K2_SMEM);
            cudaFuncSetAttribute(v41g::k_sam_out_p, cudaFuncAttributeMaxDynamicSharedMemorySize, v41g::SAM_K3P_SMEM);
            cudaFuncSetAttribute(v41g::k_sam_scores, cudaFuncAttributeMaxDynamicSharedMemorySize, SAM_K2_SMEM);
            cudaFuncSetAttribute(v41g::k_sam_out, cudaFuncAttributeMaxDynamicSharedMemorySize, SAM_K3_SMEM);
        } else { g_sam_e = nullptr; (void)cudaGetLastError(); }
    }
    max_m = std::min(max_m, SA_SPLIT_M);
    if (max_m <= g_sa_m) return;
    if (g_sa_e) { cudaFree(g_sa_e); cudaFree(g_sa_s); cudaFree(g_sa_n); }
    g_sa_m = 0;
    if (cudaMalloc(&g_sa_e, (size_t)max_m * SA_MAXE * v41::HEAD_DIM * 4) != cudaSuccess) { g_sa_e = nullptr; return; }
    if (cudaMalloc(&g_sa_s, (size_t)max_m * v41::N_HEADS * SA_MAXE * 4) != cudaSuccess) { cudaFree(g_sa_e); g_sa_e = nullptr; return; }
    if (cudaMalloc(&g_sa_n, (size_t)max_m * 4) != cudaSuccess) { cudaFree(g_sa_e); cudaFree(g_sa_s); g_sa_e = nullptr; return; }
    g_sa_m = max_m;
}
static int g_sa_split = -1;                              // V41_SA_SPLIT=0 -> k_sparse_attn_c
void sparse_attn_split_set(bool on) { g_sa_split = on ? 1 : 0; }
static constexpr int SA_K1_ROWS = 16, SA_K2_ECH = 32, SA_K3_HG = 8, SA_K3_D = 64;
// K1: valid-entry list (same construction as k_sparse_attn_c), rows decoded to E[m][k][512]
__global__ void __launch_bounds__(256) k_sa_gather(float* E, int* nent_out, const float* ring_buf, int ring,
                                                   const uint8_t* cc, const uint8_t* cs, const int* cidx, int ncomp,
                                                   const int* d_pos) {
    const int m = blockIdx.y, k0 = blockIdx.x * SA_K1_ROWS, pos = d_pos[m];
    __shared__ short ent[SA_MAXE];
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
    float* Em = E + (size_t)m * SA_MAXE * v41::HEAD_DIM;
    for (int i = threadIdx.x; i < SA_K1_ROWS * v41::HEAD_DIM / 2; i += blockDim.x) {
        const int r = k0 + i / (v41::HEAD_DIM / 2), j = i % (v41::HEAD_DIM / 2);
        if (r >= ne) break;
        const int e = ent[r];
        float a, b;
        if (e < v41::WINDOW) {
            const float* src = ring_buf + (size_t)((pos - (v41::WINDOW - 1) + e) % ring) * v41::HEAD_DIM;
            a = src[2 * j]; b = src[2 * j + 1];
        } else {
            const int t = ci[e - v41::WINDOW];
            const uint8_t byte = cc[(size_t)t * 256 + j];
            const float s = e4m3f(cs[(size_t)t * 32 + (2 * j) / v41::CKV_FP4_GRP]);
            a = fp4f(byte & 15) * s; b = fp4f(byte >> 4) * s;
        }
        Em[(size_t)r * v41::HEAD_DIM + 2 * j] = a; Em[(size_t)r * v41::HEAD_DIM + 2 * j + 1] = b;
    }
}
// K2: one warp per head over SA_K2_ECH entries; S[m][h][k] = warp_sum(fma chain) * scale
__global__ void __launch_bounds__(256) k_sa_scores(float* S, const float* E, const int* nent, const float* q, float scale) {
    const int m = blockIdx.z, h = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    const int ne = nent[m], k0 = blockIdx.y * SA_K2_ECH;
    if (k0 >= ne) return;
    const float* qh = q + ((size_t)m * v41::N_HEADS + h) * v41::HEAD_DIM;
    float qr[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) qr[i] = qh[lane + 32 * i];
    const float* Em = E + (size_t)m * SA_MAXE * v41::HEAD_DIM;
    float* Sh = S + ((size_t)m * v41::N_HEADS + h) * SA_MAXE;
    const int k1 = min(ne, k0 + SA_K2_ECH);
    for (int k = k0; k < k1; ++k) {
        const float* row = Em + (size_t)k * v41::HEAD_DIM;
        float d = 0.f;
        #pragma unroll
        for (int i = 0; i < 16; ++i) d = fmaf(qr[i], row[lane + 32 * i], d);
        d = warp_sum(d) * scale;
        if (lane == 0) Sh[k] = d;
    }
}
// K3: SA_K3_HG heads x SA_K3_D dims per CTA; p, den and acc in entry order as in k_sparse_attn_c
__global__ void __launch_bounds__(SA_K3_D) k_sa_out(float* o, const float* S, const float* E, const int* nent,
                                                    const float* sink) {
    const int m = blockIdx.z, h0 = blockIdx.x * SA_K3_HG, d = blockIdx.y * SA_K3_D + threadIdx.x;
    const int ne = nent[m];
    __shared__ float pb[SA_K3_HG][SA_MAXE];
    __shared__ float pp[SA_K3_HG][SA_MAXE];
    __shared__ float mxs[SA_K3_HG], dens[SA_K3_HG];
    const float* Sm = S + (size_t)m * v41::N_HEADS * SA_MAXE;
    // max per head (order-free), one warp-strided scan per head
    for (int hh = 0; hh < SA_K3_HG; ++hh) {
        float mx = -INFINITY;
        for (int k = threadIdx.x; k < ne; k += blockDim.x) mx = fmaxf(mx, Sm[(size_t)(h0 + hh) * SA_MAXE + k]);
        mx = warp_max(mx);
        __shared__ float red[SA_K3_D / 32];
        if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = mx;
        __syncthreads();
        if (threadIdx.x == 0) { float v = red[0]; for (int w = 1; w < SA_K3_D / 32; ++w) v = fmaxf(v, red[w]); mxs[hh] = fmaxf(v, -1e30f); }
        __syncthreads();
    }
    for (int i = threadIdx.x; i < SA_K3_HG * ne; i += blockDim.x) {
        const int hh = i / ne, k = i % ne;
        const float p = expf(Sm[(size_t)(h0 + hh) * SA_MAXE + k] - mxs[hh]);
        pp[hh][k] = p; pb[hh][k] = bf16r(p);
    }
    __syncthreads();
    if (threadIdx.x < SA_K3_HG) {
        const int hh = threadIdx.x;
        float den = 0.f;
        for (int k = 0; k < ne; ++k) den += pp[hh][k];
        dens[hh] = den + expf(sink[h0 + hh] - mxs[hh]);
    }
    float acc[SA_K3_HG];
    #pragma unroll
    for (int hh = 0; hh < SA_K3_HG; ++hh) acc[hh] = 0.f;
    const float* Em = E + (size_t)m * SA_MAXE * v41::HEAD_DIM + d;
    for (int k = 0; k < ne; ++k) {
        const float v = Em[(size_t)k * v41::HEAD_DIM];
        #pragma unroll
        for (int hh = 0; hh < SA_K3_HG; ++hh) acc[hh] = fmaf(pb[hh][k], v, acc[hh]);
    }
    __syncthreads();
    #pragma unroll
    for (int hh = 0; hh < SA_K3_HG; ++hh)
        o[((size_t)m * v41::N_HEADS + h0 + hh) * v41::HEAD_DIM + d] = bf16r(acc[hh] / dens[hh]);
}
void sparse_attn(float* o, const float* q, const float* win_ring, int ring, const uint8_t* ccodes, const uint8_t* cscales,
                 const int* cidx, int ncomp, int coff, const float* sink, const int* d_pos, int M, float scale,
                 cudaStream_t st) {
    (void)coff;
    static const bool old = [] { const char* e = getenv("V41_SA_OLD"); return e && *e == '1'; }();
    if (old) {
        k_sparse_attn<<<dim3(M, v41::N_HEADS / 8), 256, 0, st>>>(o, q, win_ring, ring, ccodes, cscales, cidx, ncomp, sink,
                                                                d_pos, scale);
        return;
    }
    // tensor-core path for every M (V41_SA_MMA=0: the fp32 kernels below)
    static const bool sam = [] { const char* e = getenv("V41_SA_MMA"); return !(e && *e == '0'); }();
    if (sam && g_sam_e) {
        for (int a = 0; a < M; a += v41g::SAM_ROWS) {
            const int mm = std::min(v41g::SAM_ROWS, M - a);
            v41g::k_sam_gather<<<dim3(v41g::SAM_E / v41g::SAM_GE, mm), 256, 0, st>>>(g_sam_e, g_sam_n, win_ring, ring, ccodes, cscales,
                                                                          cidx ? cidx + (size_t)a * ncomp : nullptr, ncomp, d_pos + a);
            if (g_sam_p && mm >= g_sam_row_min) {             // P7a row form
                v41g::k_sam_scores_row<<<mm, 256, SAM_K2_SMEM, st>>>(g_sam_s, g_sam_p, g_sam_d, g_sam_e, g_sam_n,
                                                                     q + (size_t)a * v41::N_HEADS * v41::HEAD_DIM, sink, scale);
                v41g::k_sam_out_p<<<dim3(v41g::SAM_D / v41g::SAM_K3PD, mm), 256, v41g::SAM_K3P_SMEM, st>>>(
                    o + (size_t)a * v41::N_HEADS * v41::HEAD_DIM, g_sam_p, g_sam_d, g_sam_e, g_sam_n);
                continue;
            }
            v41g::k_sam_scores<<<dim3(v41g::SAM_E / 64, mm), 128, SAM_K2_SMEM, st>>>(g_sam_s, g_sam_e, g_sam_n,
                                                                                    q + (size_t)a * v41::N_HEADS * v41::HEAD_DIM, scale);
            v41g::k_sam_out<<<dim3(v41g::SAM_D / v41g::SAM_K3D, mm), 128, SAM_K3_SMEM, st>>>(
                o + (size_t)a * v41::N_HEADS * v41::HEAD_DIM, g_sam_s, g_sam_e, g_sam_n, sink);
        }
        return;
    }
    if (g_sa_split < 0) { const char* e = getenv("V41_SA_SPLIT"); g_sa_split = (e && *e == '0') ? 0 : 1; }
    if (g_sa_split && M <= g_sa_m) {
        k_sa_gather<<<dim3(SA_MAXE / SA_K1_ROWS, M), 256, 0, st>>>(g_sa_e, g_sa_n, win_ring, ring, ccodes, cscales, cidx,
                                                                   ncomp, d_pos);
        k_sa_scores<<<dim3(v41::N_HEADS / 8, SA_MAXE / SA_K2_ECH, M), 256, 0, st>>>(g_sa_s, g_sa_e, g_sa_n, q, scale);
        k_sa_out<<<dim3(v41::N_HEADS / SA_K3_HG, v41::HEAD_DIM / SA_K3_D, M), SA_K3_D, 0, st>>>(o, g_sa_s, g_sa_e, g_sa_n,
                                                                                               sink);
        return;
    }
    constexpr int HPC = 4;
    k_sparse_attn_c<HPC><<<dim3(M, v41::N_HEADS / HPC), HPC * 32, 0, st>>>(o, q, win_ring, ring, ccodes, cscales, cidx,
                                                                         ncomp, sink, d_pos, scale);
}

} // namespace v41k
