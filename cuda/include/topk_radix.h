// topk_radix.h — single-CTA radix select, the exact-equivalent replacement for the four warp
// selection-sort top-k kernels. DECODE_LADDER.md item 1.2.
//
// WHAT IT REPLACES AND WHY. `k_topk_decode`, `k_topk_verify`, `k_topk_masked` (compressed_decode.cu)
// and `k_topk_offset` (indexer.cu) all run the same shape: `topk` sequential passes over the score
// row, one warp argmax per output slot, marking the winner `-1e30f` between passes. Item 1.1 spread
// that argmax across the 32 lanes that were already launched (14-28x), but the algorithm is still
// O(topk x T) work in T/32 x topk SEQUENTIAL steps -- at topk=512 and T=3072 (ctx 12,288, ratio 4)
// that is 512 dependent rounds of 96 strided loads each. 0.4 measured the survivor, `i:topk`, at
// **13.47 ms at ctx 12,288, slope 0.872 +/- 0.021 ms per 1000 context, 12.5 % of the context term**.
//
// This is O(T) work in a constant number of passes: an MSB-first 8-bit radix select finds the
// threshold, one gather collects the winners, one bitonic sort puts them back in order.
//
// BIT-EXACTNESS IS THE WHOLE DESIGN, not a property checked afterwards. `sparse_attn` sums the
// selected rows IN ORDER, so fp32 association makes the ORDER load-bearing, not just the set. Four
// things carry that:
//
//   1. THE COMPOSITE KEY. `comp(v,t) = (ord(v) << 32) | ~t`, where `ord` is the standard
//      order-preserving float->uint32 map. Sorting `comp` DESCENDING is exactly (value descending,
//      index ascending) -- which is what a serial ascending scan with a strict `>` produces, because
//      a later equal never displaces an earlier one. Because `~t` makes every composite distinct,
//      the radix select has NO ties to break and needs no equal-key special case: exactly `k_eff`
//      elements satisfy `comp >= threshold`, always.
//   2. ADMISSION IS THE ORIGINAL'S FLOAT COMPARE, `v > floorv`, evaluated in float and never through
//      the key. That matters for NaN: `NaN > best` is false so the originals can never select one,
//      but `ord(NaN)` is larger than every finite key and a key-space test WOULD select it. Scores
//      are relu sums and should not go NaN; "should not" is not a gate.
//   3. SIGNED ZERO IS CANONICALISED. `-0.0f == +0.0f` in the originals' compare, so they are a tie
//      and the lower index wins; but `ord(-0.0) < ord(+0.0)` as raw bits. `index_score` can emit
//      `-0.0` (relu gives exactly 0, times a negative head weight), so `ord` maps both to `+0.0`
//      first. Covered by a dedicated distribution in tests/gate_topk_radix.cu.
//   4. THE SENTINEL CONVENTION IS PRESERVED AT THE CALLER. This returns raw source indices, or -1
//      for a slot with no candidate left; each caller applies its own offset/threshold rule to that
//      exactly as before.
//
// It also drops the dynamic-shared-memory request entirely (item 1.4): the originals asked for
// ~4T bytes against a 48 KiB default limit and silently returned garbage above ~49k context. This
// uses a fixed ~7 KiB of STATIC shared memory whatever T is.
#pragma once
#include <cuda_runtime.h>
#include <cstdlib>

#define TOPK_RADIX_CAP 512      // max `topk` handled; INDEX_TOPK is 512. Callers fall back above it.
#define TOPK_RADIX_NT  512      // block size every caller launches with; must match the template arg.
                                // 128/256/512/1024 measured at T=3072: 53.3 / 38.9 / 32.0 / 31.6 us.

struct TopkRadixSmem {
    unsigned int       hist[256];
    unsigned long long buf[TOPK_RADIX_CAP];
    unsigned int       nsel, keff, need;
    unsigned long long prefix;
    int                done;
};

// order-preserving float -> uint32, with -0.0 folded onto +0.0 (see note 3 above).
__device__ __forceinline__ unsigned int topk_ord(float f){
    float g = (f == 0.0f) ? 0.0f : f;
    unsigned int u = __float_as_uint(g);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
__device__ __forceinline__ unsigned long long topk_comp(float v, int t){
    return ((unsigned long long)topk_ord(v) << 32) | (unsigned long long)(~(unsigned int)t);
}

// Block-wide. Every thread of the block must call it; NT must equal blockDim.x.
// Candidates are `t` in [0,lim) with `score[t] > floorv`. Writes the source index of the j-th best
// into sel[j] for j < min(topk, #candidates), and -1 into every remaining slot. `sel` may be global.
template<int NT>
__device__ void topk_radix_select(int* sel, const float* score, int lim, int topk,
                                  float floorv, TopkRadixSmem& S, bool early_out)
{
    const int tid = threadIdx.x;
    const int limpad = ((lim + NT - 1) / NT) * NT;            // uniform trip count, see the histogram
    if(tid == 0){ S.prefix = 0ull; S.need = 0u; S.keff = 0u; S.done = 0; S.nsel = 0u; }
    __syncthreads();

    // ---- ITEM 1.3: lim <= topk means the threshold search cannot exclude anything ---------------
    // Every candidate satisfies `#candidates <= lim <= topk`, so `k_eff == #candidates` and the
    // selected SET is "all of them" before a single score has been read. The radix loop below still
    // has to discover that: it runs one full level (clear 256 bins, one strided pass over the row
    // with a shared atomicAdd per surviving element, then a 512-iteration SERIAL scan on thread 0 to
    // find the lowest non-empty bucket) and only then concludes `hist[d] == need` and stops. That
    // level is pure discovery -- its threshold admits every candidate the gather would have taken
    // anyway -- so skipping it is a strict work reduction with an identical output, not an
    // approximation. `lim <= topk` holds on the decode side below ctx 2048 (T = ctx/ratio, ratio 4,
    // INDEX_TOPK 512) and on the verify side for every query whose causal limit is under 512.
    // BIT-EXACTNESS: the gather admits `comp >= thr`; the skipped level would have set `thr` to
    // `lowest_non_empty_top_byte << 56`, which is <= every candidate's composite, so `thr = 0`
    // selects the identical set. The bitonic sort that orders it is untouched.
    const bool early = early_out && (lim <= topk);

    // ---- MSB-first 8-bit radix select over the 64-bit composite --------------------------------
    // Invariant at the top of each level: `S.prefix` holds the bits of the threshold composite above
    // `shift+8` and `S.need` is how many of the top-k are still to be found among the elements
    // carrying that prefix. Levels below the first cost a scan and almost no atomics, because only
    // the surviving bucket matches. The `hist[d] == need` early exit is what makes the common case
    // (no exact float ties) stop after ~4 levels rather than 8.
    if(!early)
    for(int shift = 56; shift >= 0; shift -= 8){
        for(int b = tid; b < 256; b += NT) S.hist[b] = 0u;
        __syncthreads();
        const unsigned long long hmask = (shift == 56) ? 0ull : (~0ull << (shift + 8));   // <<64 is UB
        const unsigned long long pref  = S.prefix;
        // WARP-AGGREGATED HISTOGRAM. The naive `atomicAdd(&hist[d],1)` was measured at 20.7 us for
        // T=3072 against a 2.1 us empty-kernel floor -- not bandwidth, CONTENTION: real score rows
        // are exponentially distributed, so at the top byte most of the block lands on a handful of
        // bins and the shared atomic serialises. One atomicAdd per (warp, distinct digit) instead.
        // The trip count is padded to a whole number of block strides so every lane of every warp
        // reaches __match_any_sync -- the "no contribution" lanes park on key 256.
        for(int t = tid; t < limpad; t += NT){
            bool ok = false; unsigned int d = 0u;
            if(t < lim){
                float v = score[t];
                if(v > floorv){                               // note 2: the original's float compare
                    unsigned long long c = topk_comp(v, t);
                    if((c & hmask) == pref){ ok = true; d = (unsigned int)((c >> shift) & 0xffu); }
                }
            }
            if(ok) atomicAdd(&S.hist[d], 1u);
        }
        __syncthreads();
        if(tid == 0){
            unsigned int need = S.need;
            if(shift == 56){                                   // the first level also fixes k_eff
                unsigned int tot = 0;
                for(int b = 0; b < 256; ++b) tot += S.hist[b];
                need = ((unsigned)topk < tot) ? (unsigned)topk : tot;
                S.keff = need;
            }
            if(need == 0) S.done = 1;                          // no candidate at all -> all -1
            else {
                unsigned int acc = 0; int d = 255;
                for(; d > 0; --d){ if(acc + S.hist[d] >= need) break; acc += S.hist[d]; }
                S.prefix = pref | ((unsigned long long)(unsigned)d << shift);
                S.need   = need - acc;
                if(S.hist[d] == S.need) S.done = 1;            // the whole bucket is in: threshold settled
            }
        }
        __syncthreads();
        if(S.done) break;
    }

    for(int j = tid; j < topk; j += NT) sel[j] = -1;
    if(early){
        // One unconditional gather instead of (one histogram level + one gather). No `c < thr`
        // test, because the threshold that was never computed would have admitted everything.
        for(int t = tid; t < lim; t += NT){
            float v = score[t];
            if(!(v > floorv)) continue;
            unsigned int p = atomicAdd(&S.nsel, 1u);
            if(p < TOPK_RADIX_CAP) S.buf[p] = topk_comp(v, t);      // p < lim <= topk <= CAP
        }
        __syncthreads();
        if(tid == 0) S.keff = S.nsel;                               // k_eff IS the candidate count
        __syncthreads();
    }
    const unsigned int keff = S.keff;
    if(keff){
        // ---- gather: exactly keff elements satisfy comp >= threshold (composites are distinct) --
        if(!early){
        if(tid == 0) S.nsel = 0u;
        __syncthreads();
        const unsigned long long thr = S.prefix;
        for(int t = tid; t < lim; t += NT){
            float v = score[t];
            if(!(v > floorv)) continue;
            unsigned long long c = topk_comp(v, t);
            if(c < thr) continue;
            unsigned int p = atomicAdd(&S.nsel, 1u);
            if(p < TOPK_RADIX_CAP) S.buf[p] = c;
        }
        }
        int P = 1; while(P < (int)keff) P <<= 1;               // sort length, uniform across the block
        __syncthreads();
        for(int b = tid; b < P; b += NT) if(b >= (int)keff) S.buf[b] = 0ull;   // pad below everything
        __syncthreads();
        // ---- bitonic sort ASCENDING, read back reversed ----------------------------------------
        for(int k = 2; k <= P; k <<= 1){
            for(int j = k >> 1; j > 0; j >>= 1){
                for(int i = tid; i < P; i += NT){
                    int ixj = i ^ j;
                    if(ixj > i){
                        unsigned long long a = S.buf[i], b2 = S.buf[ixj];
                        bool up = ((i & k) == 0);
                        if(up ? (a > b2) : (a < b2)){ S.buf[i] = b2; S.buf[ixj] = a; }
                    }
                }
                __syncthreads();
            }
        }
        for(int j = tid; j < (int)keff; j += NT)
            sel[j] = (int)(~(unsigned int)(S.buf[P - 1 - j] & 0xffffffffu));
    }
    __syncthreads();
}

// The A/B arm. DSV4_TOPK_RADIX=0 restores the warp selection sort on the same binary.
static inline bool topk_radix_on(){
    static const bool on = [](){ const char* e = getenv("DSV4_TOPK_RADIX"); return !e || atoi(e) != 0; }();
    return on;
}
// Item 1.3's arm. DSV4_TOPK_EARLY=0 restores the full threshold search on the same binary, so both
// legs of the A/B are one build, one corpus, one server start each -- the only difference is this
// flag. It is a HOST predicate turned into a kernel ARGUMENT rather than a getenv inside the device
// function, because `topk_radix_select` is also reached from the CUDA-graph-captured `_dp` path.
static inline bool topk_early_on(){
    static const bool on = [](){ const char* e = getenv("DSV4_TOPK_EARLY"); return !e || atoi(e) != 0; }();
    return on;
}
