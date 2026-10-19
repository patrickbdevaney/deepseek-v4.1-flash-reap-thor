// moe.cu — MoE primitives, correctness-first (Gate K oracle: ref/gen_units.py).
#include "moe.h"
#include "dscratch.h"
#include "dprof.h"
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cstdlib>

__constant__ float E2M1_MAG[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};

__device__ __forceinline__ float dec_e4m3(uint8_t b) {
    __half_raw r = __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)b, __NV_E4M3);
    return __half2float(*reinterpret_cast<__half*>(&r));
}
__device__ __forceinline__ float dec_fp4(uint8_t nib) {   // sign(bit3) | mag-index(bits0-2)
    float m = E2M1_MAG[nib & 7];
    return (nib & 8) ? -m : m;
}

// ---------------- fp4_gemm (fp8 act x fp4 weight) ----------------
// One warp per (m,n). Walk K; per K-block accumulate raw dot then apply act(per-128) & weight(per-32) scales.
__global__ void fp4_gemm_kernel(float* __restrict__ C, const uint8_t* __restrict__ A, const float* __restrict__ as,
                                const uint8_t* __restrict__ B, const float* __restrict__ bs,
                                int M, int N, int K) {
    int n = blockIdx.x, m = blockIdx.y; if (m >= M || n >= N) return;
    int lane = threadIdx.x & 31;
    int KBa = K / 128, KBw = K / 32;
    const uint8_t* Arow = A + (size_t)m * K;
    const uint8_t* Bpack = B + (size_t)n * (K / 2);          // packed nibbles
    const float* asr = as + (size_t)m * KBa;
    const float* bsr = bs + (size_t)n * KBw;
    float acc = 0.f;
    for (int kb = 0; kb < KBw; ++kb) {                       // per 32-weight-block (constant weight scale)
        float sub = 0.f; int base = kb * 32;
        for (int j = lane; j < 32; j += 32) {                // 1 iter (32 lanes cover 32)
            int k = base + j;
            float av = dec_e4m3(Arow[k]) * asr[k / 128];
            uint8_t byte = Bpack[k >> 1];
            uint8_t nib = (k & 1) ? (byte >> 4) & 0xF : byte & 0xF;
            sub += av * dec_fp4(nib);
        }
        acc += sub * bsr[kb];
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffff, acc, o);
    if (lane == 0) C[(size_t)m * N + n] = acc;
}
void fp4_gemm(float* C, const uint8_t* A_fp8, const float* a_s, const uint8_t* B_fp4, const float* b_s,
              int M, int N, int K, cudaStream_t stream) {
    dim3 grid(N, M); fp4_gemm_kernel<<<grid, 32, 0, stream>>>(C, A_fp8, a_s, B_fp4, b_s, M, N, K);
}

// ---------------- moe_router_score ----------------
// One block per token. Compute n_routed scores (sqrtsoftplus), pick top-k of (score+bias) by iterative
// max, gather the PRE-bias scores, renormalize, scale.
__global__ void router_kernel(float* __restrict__ weights, int* __restrict__ indices,
                              const float* __restrict__ x, const float* __restrict__ gate_w,
                              const float* __restrict__ bias, int n, int dim, int n_routed, int topk,
                              float route_scale) {
    int tok = blockIdx.x; if (tok >= n) return;
    extern __shared__ float sh[];                 // [n_routed] orig scores + [n_routed] sel scores
    float* orig = sh; float* sel = sh + n_routed;
    const float* xr = x + (size_t)tok * dim;
    for (int e = threadIdx.x; e < n_routed; e += blockDim.x) {
        const float* gw = gate_w + (size_t)e * dim;
        float d = 0.f; for (int j = 0; j < dim; ++j) d += xr[j] * gw[j];
        float sp = (d > 20.f) ? d : log1pf(expf(d));          // softplus (stable)
        float s = sqrtf(sp);
        orig[e] = s; sel[e] = s + (bias ? bias[e] : 0.f);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float wsum = 0.f;
        for (int t = 0; t < topk; ++t) {
            float best = -1e30f; int bi = -1;
            for (int e = 0; e < n_routed; ++e) if (sel[e] > best) { best = sel[e]; bi = e; }
            sel[bi] = -1e30f;                                  // remove
            indices[(size_t)tok * topk + t] = bi;
            weights[(size_t)tok * topk + t] = orig[bi];
            wsum += orig[bi];
        }
        for (int t = 0; t < topk; ++t)
            weights[(size_t)tok * topk + t] = weights[(size_t)tok * topk + t] / wsum * route_scale;
    }
}
void moe_router_score(float* weights, int* indices, const float* x, const float* gate_w,
                      const float* bias, int n, int dim, int n_routed, int topk,
                      float route_scale, cudaStream_t stream) {
    router_kernel<<<n, 64, 2 * n_routed * sizeof(float), stream>>>(weights, indices, x, gate_w, bias,
                                                                   n, dim, n_routed, topk, route_scale);
}

// ================= MoE forward composition =================
#include "fp8_block_gemm.h"
#include "mla_attn.h"
#include <vector>
#include <unordered_map>
#include <cstdio>
void tc_fp4_gemm(float*, const uint8_t*, const float*, const uint8_t*, const float*, int, int, int, cudaStream_t); // Marlin TC W4A8 (tc_moe_gemm.cu)
bool g_moe_grouped=false;   // STRUCTURAL_PLAN Step 1b: zero-sync grouped-GEMM MoE (removes off[] D2H, graph-capturable)
bool g_moe_gemv=false;      // decode: M=1 fp4 GEMV on ORIGINAL fp4 (no repack, act stays fp8) — bandwidth-bound at small M
#define CU(x) do{cudaError_t e=(x); if(e){fprintf(stderr,"cuda %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)
long long g_moe_union_sum = 0;
int       g_moe_union_calls = 0;
long long g_moe_rows_sum = 0;
long long g_moe_tiles_sum = 0;
long long g_moe_chunks_sum = 0;
int       g_moe_rows_max = 0;
long long g_moe_rows_hist[10] = {0,0,0,0,0,0,0,0,0,0};

__global__ void compute_scores_kernel(float* sc, const float* x, const float* gw, int bs, int dim, int nr){
    int i = blockIdx.x*blockDim.x+threadIdx.x; if(i>=bs*nr) return;
    int t=i/nr, e=i%nr; const float* xr=x+(size_t)t*dim; const float* gr=gw+(size_t)e*dim;
    float d=0.f; for(int j=0;j<dim;++j) d+=xr[j]*gr[j];
    float sp = d>20.f ? d : log1pf(expf(d));
    sc[i]=sqrtf(sp);
}
// warp-per-(token,expert): coalesced row read + shuffle-reduce (vs the serial per-thread dot above).
// Finding 34/38: <<<bs*nr, 32>>> is 160 ONE-WARP blocks at decode (50% occupancy ceiling) running a
// scalar ILP=1 dot. Same fix as gemm_fp32: 4 warps/block, float4, 4 accumulator chains.
__device__ __forceinline__ float rscore_dot(const float* __restrict__ xr, const float* __restrict__ gr,
                                            int dim, int lane, int vec4){
    if(vec4){
        const float4* x4=(const float4*)xr; const float4* g4=(const float4*)gr; const int n4=dim>>2;
        float a0=0.f,a1=0.f,a2=0.f,a3=0.f;
        for(int j=lane;j<n4;j+=32){ const float4 a=x4[j], b=g4[j];
            a0=fmaf(a.x,b.x,a0); a1=fmaf(a.y,b.y,a1); a2=fmaf(a.z,b.z,a2); a3=fmaf(a.w,b.w,a3); }
        return (a0+a1)+(a2+a3);
    }
    float d=0.f; for(int j=lane;j<dim;j+=32) d+=xr[j]*gr[j]; return d;
}
__device__ __forceinline__ float rscore_fin(float d){    // sqrtsoftplus
    #pragma unroll
    for(int o=16;o>0;o>>=1) d+=__shfl_down_sync(0xffffffff,d,o);
    return sqrtf(d>20.f?d:log1pf(expf(d)));
}
__global__ void compute_scores_warp(float* sc, const float* x, const float* gw, int bs, int dim, int nr, int vec4){
    int gid=blockIdx.x*(blockDim.x>>5)+(threadIdx.x>>5); if(gid>=bs*nr) return;
    int t=gid/nr, e=gid%nr; int lane=threadIdx.x&31;
    float d=rscore_dot(x+(size_t)t*dim, gw+(size_t)e*dim, dim, lane, vec4);
    float v=rscore_fin(d); if(lane==0) sc[gid]=v;
}
// HASH layers only (IMPLEMENTATION_PLAN Tier-1 #7). The first 3 layers route by `tid2eid`, a pure
// function of the token id, so the expert set is known before any arithmetic happens — yet the
// router still scored all 160 experts and then read exactly `na`=6 of them. Score only the 6.
// 26x less router work and 26x fewer gate_w bytes on those layers; no behaviour change, because
// the discarded 154 scores never reached an output.
__global__ void compute_scores_sel(float* scsel, const float* x, const float* gw, const int* idx,
                                   int bs, int dim, int na, int vec4){
    int gid=blockIdx.x*(blockDim.x>>5)+(threadIdx.x>>5); if(gid>=bs*na) return;
    int t=gid/na, lane=threadIdx.x&31; int e=idx[gid];
    float d=rscore_dot(x+(size_t)t*dim, gw+(size_t)e*dim, dim, lane, vec4);
    float v=rscore_fin(d); if(lane==0) scsel[gid]=v;
}
// gather_scale over the COMPACT [bs,na] scores (identical math to gather_scale_kernel, which indexed
// the full [bs,nr] score table by idx).
__global__ void gather_scale_sel_kernel(float* w, const float* scsel, int bs, int na, float rs){
    int t=blockIdx.x; if(t>=bs) return; if(threadIdx.x) return;
    float sum=0.f; for(int s=0;s<na;++s) sum+=scsel[(size_t)t*na+s];
    for(int s=0;s<na;++s) w[(size_t)t*na+s]=scsel[(size_t)t*na+s]/sum*rs;
}
__global__ void gather_hash_kernel(int* idx, const long* tid2eid, const int* ids, int bs, int na){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=bs*na) return;
    int t=i/na, s=i%na; idx[i]=(int)tid2eid[(size_t)ids[t]*na + s];
}
__global__ void gather_scale_kernel(float* w, const float* sc, const int* idx, int bs, int na, int nr, float rs){
    int t=blockIdx.x; if(t>=bs) return; if(threadIdx.x) return;
    float sum=0.f; for(int s=0;s<na;++s){ float v=sc[(size_t)t*nr+idx[(size_t)t*na+s]]; w[(size_t)t*na+s]=v; sum+=v; }
    for(int s=0;s<na;++s) w[(size_t)t*na+s]=w[(size_t)t*na+s]/sum*rs;
}
__global__ void router_topk_kernel(float* w, int* idx, const float* sc, const float* bias,
                                   int bs, int nr, int na, float rs){
    int t=blockIdx.x; if(t>=bs||threadIdx.x) return;
    extern __shared__ float sel[];
    for(int e=0;e<nr;++e) sel[e]=sc[(size_t)t*nr+e]+(bias?bias[e]:0.f);
    float sum=0.f;
    for(int s=0;s<na;++s){ float best=-1e30f; int bi=-1;
        for(int e=0;e<nr;++e) if(sel[e]>best){best=sel[e];bi=e;}
        sel[bi]=-1e30f; idx[(size_t)t*na+s]=bi; float o=sc[(size_t)t*nr+bi];
        w[(size_t)t*na+s]=o; sum+=o; }
    for(int s=0;s<na;++s) w[(size_t)t*na+s]=w[(size_t)t*na+s]/sum*rs;
}
__global__ void swiglu_kernel(float* h, const float* g, const float* u, int n, float lim, float weight){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    float gg=g[i], uu=u[i];
    if(lim>0.f){ gg=fminf(gg,lim); uu=fminf(fmaxf(uu,-lim),lim); }
    float s = gg/(1.f+expf(-gg));               // silu
    h[i]= weight * s * uu;
}
__global__ void accum_kernel(float* y, const float* v, int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) y[i]+=v[i];
}
// --- batched/grouped-dispatch helpers ---
__global__ void k_gather_x(float* Xe, const float* x, const int* tok, int me, int dim){
    long i=(long)blockIdx.x*blockDim.x+threadIdx.x; if(i>=(long)me*dim) return; int r=i/dim, c=i%dim;
    Xe[i]=x[(long)tok[r]*dim+c];
}
__global__ void k_scatter_add(float* out, const float* OE, const int* tok, int me, int dim){
    long i=(long)blockIdx.x*blockDim.x+threadIdx.x; if(i>=(long)me*dim) return; int r=i/dim, c=i%dim;
    atomicAdd(&out[(long)tok[r]*dim+c], OE[i]);
}
// swiglu with per-row (per-token) routing weight; gate+up already share the quantized input tile.
__global__ void swiglu_wrow(float* h, const float* g, const float* u, const float* wrow, int me, int inter, float lim){
    long i=(long)blockIdx.x*blockDim.x+threadIdx.x; if(i>=(long)me*inter) return; int r=i/inter;
    float gg=g[i], uu=u[i]; if(lim>0.f){ gg=fminf(gg,lim); uu=fminf(fmaxf(uu,-lim),lim); }
    h[i]= wrow[r] * (gg/(1.f+expf(-gg))) * uu;
}

// device-side MoE grouping (Step 1 -> CUDA graphs): counting-sort tokens by expert on the GPU (no host vector
// work / big copies). Within-expert order is nondeterministic (atomic) but each token's expert output is scattered
// back independently, so the final result is order-invariant -> gates cosine 1.0 vs the host grouping.
__global__ void k_moe_count(int* counts, const int* idx, int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) atomicAdd(&counts[idx[i]],1); }
__global__ void k_moe_prefix(int* off, const int* counts, int nr){   // nr<=~160: single-thread exclusive scan
    if(threadIdx.x||blockIdx.x) return; off[0]=0; for(int e=0;e<nr;++e) off[e+1]=off[e]+counts[e]; }
// LAUNCH GEOMETRY (LOOP_LOG Finding 71's class). The scan above is ONE thread walking nr=160
// entries, once per layer, 43 layers per verify step and again per token in base AR. It is not a
// bytes problem — 640 B — it is 160 serial iterations on a 20-SM device inside a region
// (`moe:group`) whose entire content is launch latency. Same shape as `index_score`: the work is
// tiny, but nothing ever checked the geometry against the decode-sized input.
//
// Block-wide Hillis-Steele scan, one block, 256 threads, chunked so nr>256 still works. The result
// is BIT-IDENTICAL to the serial version (integer addition, same values, same destinations), so
// this is gated by equality, not tolerance.
#define MOE_SCAN_T 256
__global__ void k_moe_prefix_par(int* off, const int* __restrict__ counts, int nr){
    __shared__ int s[MOE_SCAN_T]; __shared__ int carry;
    if(threadIdx.x==0){ off[0]=0; carry=0; }
    __syncthreads();
    for(int base=0; base<nr; base+=MOE_SCAN_T){
        int e = base + threadIdx.x;
        s[threadIdx.x] = (e<nr) ? counts[e] : 0;
        __syncthreads();
        for(int d=1; d<MOE_SCAN_T; d<<=1){
            int t = (threadIdx.x>=d) ? s[threadIdx.x-d] : 0;
            __syncthreads();
            s[threadIdx.x] += t;
            __syncthreads();
        }
        if(e<nr) off[e+1] = carry + s[threadIdx.x];      // inclusive scan of counts == exclusive off[e+1]
        __syncthreads();
        if(threadIdx.x==MOE_SCAN_T-1) carry += s[MOE_SCAN_T-1];
        __syncthreads();
    }
}
// DSV4_SERIAL_SCAN=1 restores the <<<1,1>>> kernels so the A/B stays reachable (LEVERS.md rule).
static inline bool moe_serial_scan(){ static int v=-1; if(v<0) v = getenv("DSV4_SERIAL_SCAN")!=nullptr; return v; }
__global__ void k_moe_scatter(int* alltok, float* allwt, int* allslot, int* cursor, const int* idx, const float* wt,
                              const int* off, int bs, int na){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=bs*na) return; int t=i/na, e=idx[i];
    int pos=atomicAdd(&cursor[e],1); alltok[off[e]+pos]=t; allwt[off[e]+pos]=wt[i]; allslot[off[e]+pos]=i%na; }
// DETERMINISTIC MoE combine: place each grouped expert-output row at its unique (token,slot) slot (no atomics),
// then sum the na slots per token in FIXED order -> run-to-run reproducible (atomicAdd scatter was the only
// non-determinism; near-tie argmax flips from it were rejecting valid spec-decode drafts).
__global__ void k_scatter_ts(float* OEbts, const float* OEb, const int* alltok, const int* allslot, int maxm, int dim, int na){
    long i=(long)blockIdx.x*blockDim.x+threadIdx.x; if(i>=(long)maxm*dim) return; int r=i/dim, c=i%dim;
    OEbts[((long)alltok[r]*na+allslot[r])*dim + c] = OEb[i]; }
__global__ void k_reduce_ts(float* out, const float* OEbts, int bs, int dim, int na){
    long i=(long)blockIdx.x*blockDim.x+threadIdx.x; if(i>=(long)bs*dim) return; int t=i/dim, c=i%dim;
    float acc=0.f; for(int s=0;s<na;++s) acc += OEbts[((long)t*na+s)*dim + c]; out[i]=acc; }

void moe_forward(float* out, const float* x, const int* input_ids, const MoEWeights& w, int bs, cudaStream_t stream){
    const int dim=w.dim, inter=w.inter, nr=w.n_routed, na=w.n_act;
    float *sc,*wt,*g,*u,*h,*hs,*xs,*oe; uint8_t *xq,*hq; int *idx;
    sc=(decltype(sc))dmalloc((size_t)bs*nr*4); wt=(decltype(wt))dmalloc((size_t)bs*na*4); idx=(decltype(idx))dmalloc((size_t)bs*na*4);
    xq=(decltype(xq))dmalloc(dim); xs=(decltype(xs))dmalloc((dim/128)*4);
    g=(decltype(g))dmalloc(inter*4); u=(decltype(u))dmalloc(inter*4); h=(decltype(h))dmalloc(inter*4);
    hq=(decltype(hq))dmalloc(inter); hs=(decltype(hs))dmalloc((inter/128)*4); oe=(decltype(oe))dmalloc(dim*4);
    CU(cudaMemsetAsync(out,0,(size_t)bs*dim*4,stream));

    // float4 needs 16-byte alignment on both operands; `gate_w` is a mapped tensor (Finding 25).
    dprof_begin(DP_M_ROUTER,stream);
    const int rvec4 = ((dim & 3)==0) && ((((uintptr_t)x)&15)==0) && ((((uintptr_t)w.gate_w)&15)==0);
    if(w.is_hash){
        // Route FIRST (it needs no scores at all), then score only the selected experts — Finding 38.
        gather_hash_kernel<<<(bs*na+63)/64,64,0,stream>>>(idx,w.tid2eid,input_ids,bs,na);
        compute_scores_sel<<<(bs*na+3)/4,128,0,stream>>>(sc,x,w.gate_w,idx,bs,dim,na,rvec4);
        gather_scale_sel_kernel<<<bs,32,0,stream>>>(wt,sc,bs,na,w.route_scale);
    } else {
        compute_scores_warp<<<(bs*nr+3)/4,128,0,stream>>>(sc,x,w.gate_w,bs,dim,nr,rvec4);
        router_topk_kernel<<<bs,32,nr*sizeof(float),stream>>>(wt,idx,sc,w.gate_bias,bs,nr,na,w.route_scale);
    }
    dprof_end(DP_M_ROUTER,stream);
    // ================= GROUPED zero-sync MoE (STRUCTURAL_PLAN Step 1b) =================
    // ONE grouped GEMM per stage (gate/up/down) over ALL experts, expert map built on-device from off[] — the
    // last per-layer host sync (the off[] D2H copy) is GONE, so this path is CUDA-graph-capturable. Gated
    // cosine-1.0 vs the per-expert path (gate_grouped_moe + the full-forward argmax gate).
    extern bool g_moe_grouped;
    if(w.batched && w.device_route && g_moe_grouped && w.w1p){
        void tc_ensure_repacked(uint8_t*, int, int, cudaStream_t);
        void tc_a_to_fp16(__half*, const uint8_t*, const float*, int, int, cudaStream_t);
        void tc_build_tiles(int*, int*, int*, const int*, int, cudaStream_t);
        void tc_build_tiles_rg(int*, int*, int*, const int*, int, cudaStream_t, int);
        int  tcm_rowg();
        void tc_fp4_grouped_gemm(float*, const __half*, const uint8_t* const*, const float* const*,
                                 const int*, const int*, const int*, const int*, int, int, int, cudaStream_t);
        void tc_fp4_grouped_gemm_e8m0(float*, const __half*, const uint8_t* const*, const uint8_t* const*,
                                 const int*, const int*, const int*, const int*, int, int, int, cudaStream_t);
        void tc_fp4_grouped_gemv_e8m0(float*, const uint8_t*, const float*, const uint8_t* const*, const uint8_t* const*,
                                 const int*, const int*, const int*, const int*, int, int, int, cudaStream_t, int, int);
        extern bool g_moe_gemv;
        const int maxm = bs*na;
        // ALIGN8 for the GEMV's uint2 path (Finding 72). PER WEIGHTS STRUCT, not process-wide.
        // The first version cached this in a `static` computed from whichever MoEWeights called
        // moe_forward first — the 43 main layers, whose expert tensors are all at data_offset%16 == 8
        // and so 8-byte aligned. The DSpark MTP draft blocks are a DIFFERENT struct with different
        // tensors, and applying the main layers' answer to them issued a uint2 load at a 4-byte
        // aligned address: `cuda kernels/dspark_attn.cu:85 misaligned address`, i.e. the first real
        // sync after the draft's MoE (Finding 58's rule about line numbers again).
        // 480 host pointer compares per call is nothing against a 1.6 ms kernel; a cache keyed on the
        // struct would be correct too, but this cannot go stale.
        // Cached PER STRUCT (keyed on the expert-pointer table), computed once. Doing it per call was
        // correct but cost 3.05 ms: `moe:group` went 2.74 -> 5.79. Those 480 host dereferences per
        // call sit between dprof_begin and the first kernel launch, so the GPU idles through them and
        // the region absorbs the stall — a host-side cost showing up in a GPU mark. There are only
        // ~46 distinct MoEWeights (43 layers + 3 MTP blocks), so one entry each is free and, unlike a
        // process-wide static, cannot answer for a struct it never examined.
        static std::unordered_map<const void*,int> a8cache;
        int align8_cache;
        { auto it = a8cache.find((const void*)w.w1p);
          if(it != a8cache.end()) align8_cache = it->second;
          else {
              int ok = (getenv("NO_MOE_A8") == nullptr);
              for(int e=0;e<nr && ok;++e){
                  if(w.w1p && (((uintptr_t)w.w1p[e]) & 7)) ok = 0;
                  if(w.w3p && (((uintptr_t)w.w3p[e]) & 7)) ok = 0;
                  if(w.w2p && (((uintptr_t)w.w2p[e]) & 7)) ok = 0;
              }
              a8cache[(const void*)w.w1p] = ok; align8_cache = ok;
              if(getenv("DSV4_MOEUNION"))
                  printf("[moe] nr=%d expert table %p: 8B-aligned %s\n", nr, (const void*)w.w1p, ok?"yes -> uint2":"no -> funnel");
          } }
        // -- device counting-sort grouping (off_d KEPT on device; NO D2H) --
        int *counts,*off_d,*cursor,*alltok_d,*allslot_d; float* allwt_d;
        counts=(decltype(counts))dmalloc(nr*4); off_d=(decltype(off_d))dmalloc((nr+1)*4); cursor=(decltype(cursor))dmalloc(nr*4);
        alltok_d=(decltype(alltok_d))dmalloc((size_t)maxm*4); allwt_d=(decltype(allwt_d))dmalloc((size_t)maxm*4); allslot_d=(decltype(allslot_d))dmalloc((size_t)maxm*4);
        CU(cudaMemsetAsync(counts,0,nr*4,stream)); CU(cudaMemsetAsync(cursor,0,nr*4,stream));
        dprof_begin(DP_M_GROUP,stream);
        dprof_begin(DP_MG_COUNT,stream);
        k_moe_count<<<(bs*na+63)/64,64,0,stream>>>(counts,idx,bs*na);
        dprof_end(DP_MG_COUNT,stream);
        dprof_begin(DP_MG_PREFIX,stream);
        if(moe_serial_scan()) k_moe_prefix<<<1,1,0,stream>>>(off_d,counts,nr);
        else                  k_moe_prefix_par<<<1,MOE_SCAN_T,0,stream>>>(off_d,counts,nr);
        dprof_end(DP_MG_PREFIX,stream);
        dprof_begin(DP_MG_SCATTER,stream);
        k_moe_scatter<<<(bs*na+63)/64,64,0,stream>>>(alltok_d,allwt_d,allslot_d,cursor,idx,wt,off_d,bs,na);
        dprof_end(DP_MG_SCATTER,stream);
        // EXPERT UNION (DSV4_MOEUNION=1). The claim "the routed MoE is at the memory roofline" — the
        // single most load-bearing fact in the priority model, because it is 50% of the verify — rests
        // on a MODELLED union of 29.9 distinct experts at K=5 (ROOFLINE.md's c_v), never a measured
        // one. The measured TIME only scales 1.90/2.58/3.28/3.95 from K=1..5 where a no-overlap union
        // would scale 2/3/4/5, so either the union is ~23.7 and the MoE is at 72% of roofline with
        // real headroom, or it is ~30 and efficiency rises with K. Those differ by ~22 ms of the
        // verify. `off_d` is the exclusive scan of per-expert row counts, so the number of e with
        // off[e+1]>off[e] IS the union. One D2H per call, debug only.
        if(getenv("DSV4_MOEUNION")){
            std::vector<int> ho(nr+1); CU(cudaStreamSynchronize(stream));
            CU(cudaMemcpy(ho.data(), off_d, (size_t)(nr+1)*4, cudaMemcpyDeviceToHost));
            int u=0, mx=0; long long rows=0;
            for(int e=0;e<nr;++e){ int me_=ho[e+1]-ho[e]; if(me_>0){ ++u; rows+=me_; if(me_>mx) mx=me_; } }
            g_moe_union_sum += u; g_moe_union_calls += 1;
            g_moe_rows_sum += rows; if(mx>g_moe_rows_max) g_moe_rows_max = mx;
            // Exact tile count: ceil(rows_e/16) per touched expert. This is the redundancy factor
            // against the ideal "read each expert once" — measured, not modelled.
            { long long tiles=0, chunks=0;
              // Mirror tc_fp4_grouped_gemv_e8m0's own RB rule exactly (rows_hint == bs).
              const char* rbe=getenv("MOE_RB");
              int RB = rbe ? atoi(rbe) : (bs<=1 ? 1 : bs<=2 ? 2 : 4);
              if(RB!=1&&RB!=2&&RB!=4&&RB!=8) RB=8;
              for(int e2=0;e2<nr;++e2){ int m2=ho[e2+1]-ho[e2]; if(m2<=0) continue;
                  int nt2=(m2+15)/16; tiles += nt2;
                  for(int t2=0;t2<nt2;++t2){ int me2=m2-16*t2; if(me2>16) me2=16; chunks += (me2+RB-1)/RB; } }
              g_moe_tiles_sum += tiles; g_moe_chunks_sum += chunks; }
            // Which RB the row-amortised GEMV should use is decided by rows-per-EXPERT, not by bs:
            // acc[RB][BN] is live regardless of the real row count (the Finding 28 occupancy trap).
            for(int e=0;e<nr;++e){ int me_=ho[e+1]-ho[e]; if(me_>0 && me_<=8) g_moe_rows_hist[me_]++; else if(me_>8) g_moe_rows_hist[9]++; }
        }
        // -- tile descriptors (device) --
        int *tile_e,*tile_row0,*ntiles_d; tile_e=(decltype(tile_e))dmalloc(maxm*4); tile_row0=(decltype(tile_row0))dmalloc(maxm*4); ntiles_d=(decltype(ntiles_d))dmalloc(4);
        dprof_begin(DP_MG_TILES,stream);
        // The mma path can pack RG row-groups per tile (tc_build_tiles_rg); the GEMV path reads
        // me<=16 per tile and would silently drop rows if handed a wider one, so it stays at 16.
        if(!g_moe_gemv) tc_build_tiles_rg(tile_e,tile_row0,ntiles_d,off_d,nr,stream,tcm_rowg());
        else            tc_build_tiles(tile_e,tile_row0,ntiles_d,off_d,nr,stream);
        dprof_end(DP_MG_TILES,stream);
        // -- repack every expert weight in place once (idempotent) + upload per-expert ptr tables to device --
        // GEMV path reads ORIGINAL fp4 (no repack).
        if(!g_moe_gemv) for(int e=0;e<nr;++e){ tc_ensure_repacked((uint8_t*)w.w1p[e],inter,dim,stream);
            tc_ensure_repacked((uint8_t*)w.w3p[e],inter,dim,stream); tc_ensure_repacked((uint8_t*)w.w2p[e],dim,inter,stream); }
        // Finding 37: uploaded ONCE per layer struct, not once per layer per token. These tables are
        // constant; the only reason they were re-copied every step is that they lived in the arena,
        // which resets. One persistent cudaMalloc holding all six, filled on first use.
        const uint8_t **w1d,**w3d,**w2d; void **s1d,**s3d,**s2d;   // s*d hold float* (dequant) OR uint8_t* (e8m0)
        const void *hs1 = w.e8m0_scales?(const void*)w.w1sp8:(const void*)w.w1sp;
        const void *hs3 = w.e8m0_scales?(const void*)w.w3sp8:(const void*)w.w3sp;
        const void *hs2 = w.e8m0_scales?(const void*)w.w2sp8:(const void*)w.w2sp;
        if(!w.dev_ptr_tables){
            void* p=nullptr; CU(cudaMalloc(&p, (size_t)6*nr*sizeof(void*)));
            const void* src[6] = {(const void*)w.w1p,(const void*)w.w3p,(const void*)w.w2p, hs1,hs3,hs2};
            for(int i=0;i<6;++i) CU(cudaMemcpy((char*)p + (size_t)i*nr*sizeof(void*), src[i], nr*sizeof(void*), cudaMemcpyHostToDevice));
            w.dev_ptr_tables = p;
        }
        char* tb = (char*)w.dev_ptr_tables; const size_t tsz = (size_t)nr*sizeof(void*);
        w1d=(const uint8_t**)(tb+0*tsz); w3d=(const uint8_t**)(tb+1*tsz); w2d=(const uint8_t**)(tb+2*tsz);
        s1d=(void**)(tb+3*tsz); s3d=(void**)(tb+4*tsz); s2d=(void**)(tb+5*tsz);
        // -- scratch (maxm rows) --
        float *Xe,*Xes,*Gb,*Ub,*Hb,*Hsb,*OEb; uint8_t *Xeq,*Hqb; __half *x16,*h16;
        Xe=(decltype(Xe))dmalloc((size_t)maxm*dim*4); Xeq=(decltype(Xeq))dmalloc((size_t)maxm*dim); Xes=(decltype(Xes))dmalloc((size_t)maxm*(dim/128)*4);
        x16=(decltype(x16))dmalloc((size_t)maxm*dim*2); h16=(decltype(h16))dmalloc((size_t)maxm*inter*2);
        Gb=(decltype(Gb))dmalloc((size_t)maxm*inter*4); Ub=(decltype(Ub))dmalloc((size_t)maxm*inter*4); Hb=(decltype(Hb))dmalloc((size_t)maxm*inter*4);
        Hqb=(decltype(Hqb))dmalloc((size_t)maxm*inter); Hsb=(decltype(Hsb))dmalloc((size_t)maxm*(inter/128)*4); OEb=(decltype(OEb))dmalloc((size_t)maxm*dim*4);
        float* OEbts=(float*)dmalloc((size_t)bs*na*dim*4);   // deterministic per-(token,slot) combine buffer

        // -- shared expert, FORKED BEFORE the routed path (LOOP_LOG Finding 55) --
        //
        // The shared expert reads only `x` and writes only its own buffers, so it never depended on
        // the routed experts — it was merely issued after them on the same stream, and paid the
        // full price of its bytes: 10.37 ms for 574 MB = 55 GB/s in situ at K=5, alongside a routed
        // grouped GEMV running 17.18 GB at 241 GB/s, i.e. AT the roofline. A kernel that slow is
        // latency-starved, not bandwidth-starved (overlap_probe), so folding its traffic into an
        // already-saturated stream should cost ~2.4 ms instead of 10.37.
        //
        // THE FORK MUST BE RECORDED HERE, not next to the shared code. The first version of this
        // change recorded g_side_fork immediately before the shared chain — which is *after* the
        // whole routed path is already enqueued on `stream`, so the side stream dutifully waited
        // for all of it and overlapped nothing. Measured: moe:shared 10.37 -> 10.48 ms, TOTAL
        // 163.95 -> 166.51, i.e. exactly no effect. An event records a POINT IN THE STREAM, and the
        // point has to precede the work you want to overlap with.
        //
        // Separate buffers are the other precondition: the shared path used to reuse the routed
        // path's Xeq/Xes/Gb/Ub/Hb/Hqb/Hsb/OEb, and running both on those concurrently is a race
        // that still returns plausible numbers. These are bs rows, not maxm -- ~236 KB at bs=5.
        const bool split = g_side && !getenv("NOSHARED") && !getenv("NO_MOESPLIT");
        float *sXes=nullptr,*sGb,*sUb,*sHb,*sHsb,*sOEb=nullptr; uint8_t *sXeq,*sHqb;
        if(split){
            sXeq=(decltype(sXeq))dmalloc((size_t)bs*dim); sXes=(decltype(sXes))dmalloc((size_t)bs*(dim/128)*4);
            sGb=(decltype(sGb))dmalloc((size_t)bs*inter*4); sUb=(decltype(sUb))dmalloc((size_t)bs*inter*4);
            sHb=(decltype(sHb))dmalloc((size_t)bs*inter*4); sHqb=(decltype(sHqb))dmalloc((size_t)bs*inter);
            sHsb=(decltype(sHsb))dmalloc((size_t)bs*(inter/128)*4); sOEb=(decltype(sOEb))dmalloc((size_t)bs*dim*4);
            cudaEventRecord(g_side_fork,stream); cudaStreamWaitEvent(g_side,g_side_fork,0);
            act_quant_fp8(sXeq,sXes,x,bs,dim,128,g_side);
            fp8_block_gemm(sGb,sXeq,sXes,w.sw1,w.sw1s,bs,inter,dim,g_side);
            fp8_block_gemm(sUb,sXeq,sXes,w.sw3,w.sw3s,bs,inter,dim,g_side);
            swiglu_kernel<<<((size_t)bs*inter+63)/64,64,0,g_side>>>(sHb,sGb,sUb,bs*inter,w.swiglu_limit,1.f);
            act_quant_fp8(sHqb,sHsb,sHb,bs,inter,128,g_side);
            fp8_block_gemm(sOEb,sHqb,sHsb,w.sw2,w.sw2s,bs,dim,inter,g_side);
            cudaEventRecord(g_side_join,g_side);
        }
        // -- routed experts: gather -> quant -> fp16 -> grouped gate/up -> swiglu -> quant -> fp16 -> grouped down -> scatter --
        k_gather_x<<<((size_t)maxm*dim+255)/256,256,0,stream>>>(Xe,x,alltok_d,maxm,dim);
        act_quant_fp8(Xeq,Xes,Xe,maxm,dim,128,stream);
        dprof_end(DP_M_GROUP,stream);
        // NOTE (LOOP_LOG Finding 31): the GEMV and the m16 mma need MUTUALLY EXCLUSIVE weight
        // layouts. `tc_ensure_repacked` rewrites each expert IN PLACE into mma-fragment order and
        // is skipped when the GEMV is active; the GEMV reads the ORIGINAL packed fp4. So the two
        // cannot be mixed per-M within one run — attempting it made prefill (M=5) read unrepacked
        // weights through the mma and produced argmax 260 instead of 11111. Selecting per-M would
        // need a second weight copy (~86 GiB — impossible here) or a non-mutating repack.
        // The GEMV reads scale tables as e8m0 BYTES, unconditionally. It has no f32-scale variant,
        // and the `(const uint8_t* const*)s1d` cast below will happily accept `const float*`
        // pointers and read the low byte of each float as an exponent. That is exactly what
        // happened to the DSpark draft blocks for eleven runs (Finding 39): correct-looking main
        // output, garbage draft, acceptance pinned at 0/4. Make the precondition part of the
        // predicate so the two can never disagree again.
        const bool use_gemv = g_moe_gemv && w.e8m0_scales;
        dprof_begin(DP_M_W13,stream);
        if(use_gemv){                                          // fp4 GEMV on ORIGINAL fp4 (act stays fp8)
            tc_fp4_grouped_gemv_e8m0(Gb,Xeq,Xes,w1d,(const uint8_t* const*)s1d,off_d,tile_e,tile_row0,ntiles_d,maxm,inter,dim, stream, bs, align8_cache);
            tc_fp4_grouped_gemv_e8m0(Ub,Xeq,Xes,w3d,(const uint8_t* const*)s3d,off_d,tile_e,tile_row0,ntiles_d,maxm,inter,dim, stream, bs, align8_cache);
        } else if(w.e8m0_scales){
            tc_a_to_fp16(x16,Xeq,Xes,maxm,dim,stream);
            tc_fp4_grouped_gemm_e8m0(Gb,x16,w1d,(const uint8_t* const*)s1d,off_d,tile_e,tile_row0,ntiles_d,maxm,inter,dim,stream);
            tc_fp4_grouped_gemm_e8m0(Ub,x16,w3d,(const uint8_t* const*)s3d,off_d,tile_e,tile_row0,ntiles_d,maxm,inter,dim,stream);
        } else {
            tc_a_to_fp16(x16,Xeq,Xes,maxm,dim,stream);
            tc_fp4_grouped_gemm(Gb,x16,w1d,(const float* const*)s1d,off_d,tile_e,tile_row0,ntiles_d,maxm,inter,dim,stream);
            tc_fp4_grouped_gemm(Ub,x16,w3d,(const float* const*)s3d,off_d,tile_e,tile_row0,ntiles_d,maxm,inter,dim,stream);
        }
        dprof_end(DP_M_W13,stream);
        dprof_begin(DP_M_ACT,stream);
        swiglu_wrow<<<((size_t)maxm*inter+255)/256,256,0,stream>>>(Hb,Gb,Ub,allwt_d,maxm,inter,w.swiglu_limit);
        act_quant_fp8(Hqb,Hsb,Hb,maxm,inter,128,stream);
        dprof_end(DP_M_ACT,stream);
        dprof_begin(DP_M_W2,stream);
        if(use_gemv)           tc_fp4_grouped_gemv_e8m0(OEb,Hqb,Hsb,w2d,(const uint8_t* const*)s2d,off_d,tile_e,tile_row0,ntiles_d,maxm,dim,inter, stream, bs, align8_cache);
        else if(w.e8m0_scales){tc_a_to_fp16(h16,Hqb,Hsb,maxm,inter,stream); tc_fp4_grouped_gemm_e8m0(OEb,h16,w2d,(const uint8_t* const*)s2d,off_d,tile_e,tile_row0,ntiles_d,maxm,dim,inter,stream);}
        else                  {tc_a_to_fp16(h16,Hqb,Hsb,maxm,inter,stream); tc_fp4_grouped_gemm(OEb,h16,w2d,(const float* const*)s2d,off_d,tile_e,tile_row0,ntiles_d,maxm,dim,inter,stream);}
        dprof_end(DP_M_W2,stream);
        dprof_begin(DP_M_COMBINE,stream);
        // DETERMINISTIC combine: scatter to unique (token,slot) slots (no atomics) then sum na slots in fixed order
        k_scatter_ts<<<((size_t)maxm*dim+255)/256,256,0,stream>>>(OEbts,OEb,alltok_d,allslot_d,maxm,dim,na);
        k_reduce_ts<<<((size_t)bs*dim+255)/256,256,0,stream>>>(out,OEbts,bs,dim,na);
        dprof_end(DP_M_COMBINE,stream);
        // -- shared expert (fp8, all bs tokens, weight 1) --
        //
        // CONCURRENT WITH THE ROUTED PATH (LOOP_LOG Finding 55). The shared expert reads only `x`
        // and writes only its own buffers, so it never depended on the routed experts — it was
        // merely issued after them on the same stream. That cost the full price of its bytes:
        // measured in situ at K=5, moe:shared is 10.37 ms for 574 MB = 55 GB/s, while the routed
        // grouped GEMV alongside it runs 17.18 GB at 241 GB/s, i.e. AT the roofline. Folding the
        // shared expert's traffic into a stream that is already saturated costs ~2.4 ms of extra
        // time instead of 10.37, because the memory system was never the thing the shared expert
        // was waiting on — latency was.
        //
        // The buffers below are SEPARATE allocations, which is the whole reason this could not be
        // done before: the shared path used to reuse Xeq/Xes/Gb/Ub/Hb/Hqb/Hsb/OEb, the routed
        // path's scratch, and running the two concurrently on those would be a race that still
        // returns plausible numbers. They cost bs (not maxm) rows -- ~236 KB at bs=5 against a
        // 512 MB arena.
        //
        // Arithmetic, operand order and accumulation order are untouched, and `accum` still runs on
        // the main stream AFTER k_scatter_ts/k_reduce_ts have written `out`, so the result is
        // bit-identical to the serial version. The join is what guarantees that, and it is also
        // what keeps the side stream's writes from outliving the layer's arena_reset.
        // JOIN. `out` already holds the routed sum on the main stream; add the shared branch there,
        // in the same order as the serial version, so the result is bit-identical. The join is also
        // what keeps the side stream's writes from outliving this layer's arena_reset.
        dprof_begin(DP_M_SHARED,stream);
        if(split){
            cudaStreamWaitEvent(stream,g_side_join,0);
            accum_kernel<<<((size_t)bs*dim+63)/64,64,0,stream>>>(out,sOEb,bs*dim);
        } else if(!getenv("NOSHARED")){
            act_quant_fp8(Xeq,Xes,x,bs,dim,128,stream);
            fp8_block_gemm(Gb,Xeq,Xes,w.sw1,w.sw1s,bs,inter,dim,stream);
            fp8_block_gemm(Ub,Xeq,Xes,w.sw3,w.sw3s,bs,inter,dim,stream);
            swiglu_kernel<<<((size_t)bs*inter+63)/64,64,0,stream>>>(Hb,Gb,Ub,bs*inter,w.swiglu_limit,1.f);
            act_quant_fp8(Hqb,Hsb,Hb,bs,inter,128,stream);
            fp8_block_gemm(OEb,Hqb,Hsb,w.sw2,w.sw2s,bs,dim,inter,stream);
            accum_kernel<<<((size_t)bs*dim+63)/64,64,0,stream>>>(out,OEb,bs*dim);
        }
        dprof_end(DP_M_SHARED,stream);
        dsync(stream);
        dfree(counts);dfree(off_d);dfree(cursor);dfree(alltok_d);dfree(allwt_d);
        dfree(tile_e);dfree(tile_row0);dfree(ntiles_d);
        // w1d..s2d point into the persistent dev_ptr_tables cache — NOT arena scratch. Never freed here.
        dfree(Xe);dfree(Xeq);dfree(Xes);dfree(x16);dfree(h16);
        dfree(Gb);dfree(Ub);dfree(Hb);dfree(Hqb);dfree(Hsb);dfree(OEb);
        dfree(sc);dfree(wt);dfree(idx);dfree(xq);dfree(xs);dfree(g);dfree(u);dfree(h);dfree(hq);dfree(hs);dfree(oe);
        return;
    }

    std::vector<int> hidx; std::vector<float> hw;
    if(!(w.batched && w.device_route)){                       // device_route does the grouping on-GPU (no host copy)
        hidx.resize((size_t)bs*na); hw.resize((size_t)bs*na);
        dsync(stream);
        CU(cudaMemcpy(hidx.data(),idx,(size_t)bs*na*4,cudaMemcpyDeviceToHost));
        CU(cudaMemcpy(hw.data(),wt,(size_t)bs*na*4,cudaMemcpyDeviceToHost));
    }

    size_t w13n=(size_t)inter*(dim/2), w13s=(size_t)inter*(dim/32);
    size_t w2n=(size_t)dim*(inter/2),  w2s=(size_t)dim*(inter/32);
    void tc_fp4_gemm_pp_auto(float*, const uint8_t*, const float*, const uint8_t*, const float*, int,int,int, cudaStream_t);
    auto GEMM = w.use_tc_pp ? tc_fp4_gemm_pp_auto : (w.use_tc ? tc_fp4_gemm : fp4_gemm);
    if(w.batched){
        // Scratch must hold the LARGEST per-expert group: a token can route to the same expert in multiple na
        // slots (esp. hash layers), so me can exceed bs — up to bs*na total assignments. Size for bs*na.
        const int maxm = bs*na;
        float *Xe,*Xes2,*Gb,*Ub,*Hb,*Hsb,*OEb; uint8_t *Xeq,*Hqb;
        Xe=(decltype(Xe))dmalloc((size_t)maxm*dim*4); Xeq=(decltype(Xeq))dmalloc((size_t)maxm*dim); Xes2=(decltype(Xes2))dmalloc((size_t)maxm*(dim/128)*4);
        Gb=(decltype(Gb))dmalloc((size_t)maxm*inter*4); Ub=(decltype(Ub))dmalloc((size_t)maxm*inter*4); Hb=(decltype(Hb))dmalloc((size_t)maxm*inter*4);
        Hqb=(decltype(Hqb))dmalloc((size_t)maxm*inter); Hsb=(decltype(Hsb))dmalloc((size_t)maxm*(inter/128)*4); OEb=(decltype(OEb))dmalloc((size_t)maxm*dim*4);
        std::vector<int> off(nr+1,0); int* alltok_d; float* allwt_d;
        if(w.device_route){
            // DEVICE grouping: counting-sort on GPU (no host vector build / big idx-wt-alltok copies).
            int *counts,*off_d,*cursor,*allslot_d;
            counts=(decltype(counts))dmalloc(nr*4); off_d=(decltype(off_d))dmalloc((nr+1)*4); cursor=(decltype(cursor))dmalloc(nr*4);
            alltok_d=(decltype(alltok_d))dmalloc((size_t)maxm*4); allwt_d=(decltype(allwt_d))dmalloc((size_t)maxm*4); allslot_d=(decltype(allslot_d))dmalloc((size_t)maxm*4);
            CU(cudaMemsetAsync(counts,0,nr*4,stream)); CU(cudaMemsetAsync(cursor,0,nr*4,stream));
            k_moe_count<<<(bs*na+63)/64,64,0,stream>>>(counts,idx,bs*na);
            if(moe_serial_scan()) k_moe_prefix<<<1,1,0,stream>>>(off_d,counts,nr);
            else                  k_moe_prefix_par<<<1,MOE_SCAN_T,0,stream>>>(off_d,counts,nr);
            k_moe_scatter<<<(bs*na+63)/64,64,0,stream>>>(alltok_d,allwt_d,allslot_d,cursor,idx,wt,off_d,bs,na);
            CU(cudaMemcpy(off.data(),off_d,(nr+1)*4,cudaMemcpyDeviceToHost));   // small sync (grid sizes); zero-sync = Step 1b
            dfree(counts); dfree(off_d); dfree(cursor);
        } else {
            // HOST grouping (oracle): build per-expert lists on host, upload flat arrays + offsets.
            std::vector<std::vector<int>> etok(nr); std::vector<std::vector<float>> ewt(nr);
            for(int t=0;t<bs;++t) for(int s=0;s<na;++s){ int e=hidx[(size_t)t*na+s]; etok[e].push_back(t); ewt[e].push_back(hw[(size_t)t*na+s]); }
            std::vector<int> alltok; std::vector<float> allwt;
            for(int e=0;e<nr;++e){ for(int t:etok[e]) alltok.push_back(t); for(float wv:ewt[e]) allwt.push_back(wv); off[e+1]=alltok.size(); }
            alltok_d=(decltype(alltok_d))dmalloc((size_t)alltok.size()*4); allwt_d=(decltype(allwt_d))dmalloc((size_t)allwt.size()*4);
            CU(cudaMemcpy(alltok_d,alltok.data(),(size_t)alltok.size()*4,cudaMemcpyHostToDevice));
            CU(cudaMemcpy(allwt_d,allwt.data(),(size_t)allwt.size()*4,cudaMemcpyHostToDevice));
        }
        for(int e=0;e<nr;++e){ int me=off[e+1]-off[e]; if(!me) continue;
            const int* tok_d = alltok_d+off[e]; const float* wrow = allwt_d+off[e];
            const uint8_t *W1=w.w1p?w.w1p[e]:w.w1+(size_t)e*w13n, *W3=w.w3p?w.w3p[e]:w.w3+(size_t)e*w13n, *W2=w.w2p?w.w2p[e]:w.w2+(size_t)e*w2n;
            const float *W1s=w.w1sp?w.w1sp[e]:w.w1s+(size_t)e*w13s, *W3s=w.w3sp?w.w3sp[e]:w.w3s+(size_t)e*w13s, *W2s=w.w2sp?w.w2sp[e]:w.w2s+(size_t)e*w2s;
            k_gather_x<<<((size_t)me*dim+255)/256,256,0,stream>>>(Xe,x,tok_d,me,dim);
            act_quant_fp8(Xeq,Xes2,Xe,me,dim,128,stream);
            GEMM(Gb,Xeq,Xes2, W1,W1s, me,inter,dim,stream);   // gate+up share the quantized input tile Xeq
            GEMM(Ub,Xeq,Xes2, W3,W3s, me,inter,dim,stream);
            swiglu_wrow<<<((size_t)me*inter+255)/256,256,0,stream>>>(Hb,Gb,Ub,wrow,me,inter,w.swiglu_limit);
            act_quant_fp8(Hqb,Hsb,Hb,me,inter,128,stream);
            GEMM(OEb,Hqb,Hsb, W2,W2s, me,dim,inter,stream);
            k_scatter_add<<<((size_t)me*dim+255)/256,256,0,stream>>>(out,OEb,tok_d,me,dim);
        }
        // shared expert, all bs tokens (fp8, weight 1)
        if(!getenv("NOSHARED")){
        act_quant_fp8(Xeq,Xes2,x,bs,dim,128,stream);
        fp8_block_gemm(Gb,Xeq,Xes2, w.sw1,w.sw1s, bs,inter,dim,stream);
        fp8_block_gemm(Ub,Xeq,Xes2, w.sw3,w.sw3s, bs,inter,dim,stream);
        swiglu_kernel<<<((size_t)bs*inter+63)/64,64,0,stream>>>(Hb,Gb,Ub,bs*inter,w.swiglu_limit,1.f);
        act_quant_fp8(Hqb,Hsb,Hb,bs,inter,128,stream);
        fp8_block_gemm(OEb,Hqb,Hsb, w.sw2,w.sw2s, bs,dim,inter,stream);
        accum_kernel<<<((size_t)bs*dim+63)/64,64,0,stream>>>(out,OEb,bs*dim);
        }
        dsync(stream);
        dfree(Xe);dfree(Xeq);dfree(Xes2);dfree(Gb);dfree(Ub);dfree(Hb);dfree(Hqb);dfree(Hsb);dfree(OEb);dfree(alltok_d);dfree(allwt_d);
        dfree(sc);dfree(wt);dfree(idx);dfree(xq);dfree(xs);dfree(g);dfree(u);dfree(h);dfree(hq);dfree(hs);dfree(oe);
        return;
    }
    for(int t=0;t<bs;++t){
        const float* xr=x+(size_t)t*dim;
        // routed experts (fp4)
        for(int s=0;s<na;++s){
            int e=hidx[(size_t)t*na+s]; float wgt=hw[(size_t)t*na+s];
            // stacked base+stride, OR per-expert pointer table (real checkpoint) when w1p != null.
            const uint8_t *W1 = w.w1p? w.w1p[e] : w.w1+(size_t)e*w13n, *W3 = w.w3p? w.w3p[e] : w.w3+(size_t)e*w13n,
                          *W2 = w.w2p? w.w2p[e] : w.w2+(size_t)e*w2n;
            const float *W1s = w.w1sp? w.w1sp[e] : w.w1s+(size_t)e*w13s, *W3s = w.w3sp? w.w3sp[e] : w.w3s+(size_t)e*w13s,
                        *W2s = w.w2sp? w.w2sp[e] : w.w2s+(size_t)e*w2s;
            act_quant_fp8(xq,xs,xr,1,dim,128,stream);
            GEMM(g,xq,xs, W1, W1s, 1,inter,dim,stream);
            GEMM(u,xq,xs, W3, W3s, 1,inter,dim,stream);
            swiglu_kernel<<<(inter+63)/64,64,0,stream>>>(h,g,u,inter,w.swiglu_limit,wgt);
            act_quant_fp8(hq,hs,h,1,inter,128,stream);
            GEMM(oe,hq,hs, W2, W2s, 1,dim,inter,stream);
            accum_kernel<<<(dim+63)/64,64,0,stream>>>(out+(size_t)t*dim,oe,dim);
        }
        // shared expert (fp8), no routing weight
        if(getenv("NOSHARED")) continue;
        act_quant_fp8(xq,xs,xr,1,dim,128,stream);
        fp8_block_gemm(g,xq,xs, w.sw1, w.sw1s, 1,inter,dim,stream);
        fp8_block_gemm(u,xq,xs, w.sw3, w.sw3s, 1,inter,dim,stream);
        swiglu_kernel<<<(inter+63)/64,64,0,stream>>>(h,g,u,inter,w.swiglu_limit,1.f);
        act_quant_fp8(hq,hs,h,1,inter,128,stream);
        fp8_block_gemm(oe,hq,hs, w.sw2, w.sw2s, 1,dim,inter,stream);
        accum_kernel<<<(dim+63)/64,64,0,stream>>>(out+(size_t)t*dim,oe,dim);
    }
    dsync(stream);
    dfree(sc);dfree(wt);dfree(idx);dfree(xq);dfree(xs);dfree(g);dfree(u);
    dfree(h);dfree(hq);dfree(hs);dfree(oe);
}
