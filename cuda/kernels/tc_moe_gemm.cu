// tc_moe_gemm.cu — Marlin-class tensor-core GEMM for fp8-act × fp4-weight (W4A8), our MoE experts.
// Adapted from gemma-cuda-hybrid/kernels/tc_verify_gemm.cu (raw mma.sync.m16n8k16, 1 warp = 8 N-cols,
// weight-repack + __ldcs coalesced loads + in-register FP4→fp16 dequant).
// OUR adaptation: (a) fp8-e4m3 act → fp16 with per-128 act scale folded in; (b) fp4 weight path unchanged;
// (c) gemma's per-k-tile fp8 weight-scale → our per-32 e8m0 (exp2(byte-127)), pre-expanded to fp16 per-k-tile.
// *** UNGATED — must pass bit-exact/cosine vs fp4_gemm (tests/gate_tc_moe.cu) before it is trusted / used. ***
#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <cstdio>
#include "moe.h"   // fp4_gemm signature parity

__device__ __forceinline__ __half2 tcm_fp4x2(unsigned char b){
    __half2_raw r=__nv_cvt_fp4x2_to_halfraw2((__nv_fp4x2_storage_t)b,__NV_E2M1); return *reinterpret_cast<__half2*>(&r); }
__device__ __forceinline__ float tcm_e4m3(uint8_t b){
    __half_raw r=__nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)b,__NV_E4M3); return __half2float(*reinterpret_cast<__half*>(&r)); }
__device__ __forceinline__ void mma_m16n8k16(float* c, const unsigned* a, const unsigned* b){
    asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3])
      : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]), "r"(b[0]),"r"(b[1]));
}

// A fp8[M,K] + a_s[M,K/128] (f32) -> x16[M,K] fp16, with act scale folded (matches fp4_gemm's dec_e4m3(A)*as).
__global__ void k_a_to_fp16(__half* x16, const uint8_t* A, const float* a_s, int M, int K){
    long i=(long)blockIdx.x*blockDim.x+threadIdx.x; if(i>=(long)M*K) return; int m=i/K, k=i%K;
    x16[i]=__float2half(tcm_e4m3(A[i]) * a_s[(long)m*(K/128)+k/128]);
}
// weight repack (fp4 [N,K/2] -> [N/8][K/128][32 lane][16B]), same as gemma k_tc_repack_w.
__global__ void k_repack_w(uint8_t* wpr, const uint8_t* wp, int N, int K){
    long idx=(long)blockIdx.x*blockDim.x+threadIdx.x; int kg8=K/128; long tot=(long)(N/8)*kg8*32*8; if(idx>=tot)return;
    int kl=idx&7; long r=idx>>3; int lane=r&31; long r2=r>>5; int g=r2%kg8, n_block=r2/kg8, gid=lane>>2, t4=lane&3;
    int k_tile=g*8+kl; long src=(long)(n_block*8+gid)*(K/2) + (long)k_tile*8;
    long dst=((long)n_block*kg8 + g)*512 + (long)lane*16 + 2*kl;
    wpr[dst]=wp[src+t4]; wpr[dst+1]=wp[src+t4+4];
}
// weight scale b_s[N,K/32] (f32, already-dequantized pow2 — parity with fp4_gemm) -> wsr[N/8][K/16][8] fp16
// (per-k-tile, per-n): one 32-block scale covers 2 k-tiles (32/16).
__global__ void k_repack_s(__half* wsr, const float* bs, int N, int K){
    long idx=(long)blockIdx.x*blockDim.x+threadIdx.x; int kt=K/16; long tot=(long)(N/8)*kt*8; if(idx>=tot)return;
    int g=idx&7; long r=idx>>3; int k_tile=r%kt, n_block=r/kt; int n=n_block*8+g;
    wsr[((long)n_block*kt + k_tile)*8 + g] = __float2half(bs[(long)n*(K/32) + k_tile/2]);
}
#ifndef MOE_BN
#define MOE_BN 2
#endif
#ifndef TCM_WARPS
#define TCM_WARPS 1
#endif
__global__ void tc_w4a8_kernel(float* out, const uint8_t* wpr, const __half* wsr, const __half* x16, int M, int N, int K){
    int lane=threadIdx.x&31, gid=lane>>2, t4=lane&3;
    int warp=threadIdx.x>>5; int n_block=blockIdx.x*TCM_WARPS+warp; if((long)n_block*8>=N) return; int n0=n_block*8;
    float c[4]={0,0,0,0}; int kg8=K/128, kt=K/16;
    const uint8_t* wb = wpr + (long)n_block*kg8*512;
    const __half*  sb = wsr + (long)n_block*kt*8;
    const __half* xg0 = x16 + (size_t)gid*K, *xg8 = x16 + (size_t)(gid+8)*K;
    bool m0=gid<M, m8=(gid+8)<M;
    for(int g=0; g<kg8; ++g){
        uint4 w16 = __ldcs((const uint4*)(wb + (long)g*512 + lane*16));
        const uint8_t* wby=(const uint8_t*)&w16;
        #pragma unroll
        for(int kl=0; kl<8; ++kl){ int k_tile=g*8+kl, k0=k_tile*16;
            unsigned a[4];
            a[0]=m0? *(const unsigned*)(xg0+k0+2*t4)   : 0u;
            a[1]=m8? *(const unsigned*)(xg8+k0+2*t4)   : 0u;
            a[2]=m0? *(const unsigned*)(xg0+k0+2*t4+8) : 0u;
            a[3]=m8? *(const unsigned*)(xg8+k0+2*t4+8) : 0u;
            __half2 sc2 = __half2half2(sb[(long)k_tile*8 + gid]);
            __half2 b0 = __hmul2(tcm_fp4x2(wby[2*kl]),   sc2);
            __half2 b1 = __hmul2(tcm_fp4x2(wby[2*kl+1]), sc2);
            unsigned bb[2]; bb[0]=*(unsigned*)&b0; bb[1]=*(unsigned*)&b1;
            mma_m16n8k16(c, a, bb);
        }
    }
    int cn=2*t4;
    if(gid<M   && n0+cn  <N) out[(size_t)gid*N   + n0+cn  ]=c[0];
    if(gid<M   && n0+cn+1<N) out[(size_t)gid*N   + n0+cn+1]=c[1];
    if(gid+8<M && n0+cn  <N) out[(size_t)(gid+8)*N + n0+cn ]=c[2];
    if(gid+8<M && n0+cn+1<N) out[(size_t)(gid+8)*N + n0+cn+1]=c[3];
}

#define CT(x) do{cudaError_t e=(x); if(e){fprintf(stderr,"cuda %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)
#include <unordered_map>
// weight-repack cache: keyed by (B_fp4 ptr, b_s ptr). Repacked ONCE (lazy warm-up), reused across all decode
// forwards — this is the decode win. NOTE: repacked ≈ same bytes as the original fp4 layout, so caching every
// routed expert doubles expert-weight memory (~82 GB -> OOM on the full model). For the full model, repack at
// LOAD storing repacked in place of the original (loader change), or scope per-layer. Fine for gates/tests here.
struct TcW { uint8_t* wpr; __half* wsr; };
static std::unordered_map<const void*, TcW> g_tc_cache;
static TcW tc_get_weight(const uint8_t* B_fp4, const float* b_s, int N, int K, cudaStream_t s){
    auto it=g_tc_cache.find(B_fp4); if(it!=g_tc_cache.end()) return it->second;
    TcW w; CT(cudaMalloc(&w.wpr,(size_t)(N/8)*(K/128)*512)); CT(cudaMalloc(&w.wsr,(size_t)(N/8)*(K/16)*8*2));
    long tw=(long)(N/8)*(K/128)*32*8, ts=(long)(N/8)*(K/16)*8;
    k_repack_w<<<(tw+255)/256,256,0,s>>>(w.wpr, B_fp4, N, K);
    k_repack_s<<<(ts+255)/256,256,0,s>>>(w.wsr, b_s, N, K);
    g_tc_cache.emplace(B_fp4, w); return w;
}
// Same signature as fp4_gemm (drop-in). Weight repack cached by ptr; only the fp8->fp16 act convert is per-call.
void tc_fp4_gemm(float* C, const uint8_t* A_fp8, const float* a_s, const uint8_t* B_fp4, const float* b_s,
                 int M, int N, int K, cudaStream_t s){
    __half* x16; CT(cudaMalloc(&x16,(size_t)M*K*2));
    k_a_to_fp16<<<((long)M*K+255)/256,256,0,s>>>(x16, A_fp8, a_s, M, K);
    TcW w = tc_get_weight(B_fp4, b_s, N, K, s);
    dim3 grid((N/8 + TCM_WARPS-1)/TCM_WARPS); tc_w4a8_kernel<<<grid, TCM_WARPS*32, 0, s>>>(C, w.wpr, w.wsr, x16, M, N, K);
    CT(cudaStreamSynchronize(s)); cudaFree(x16);
}

// Free the repack cache — call per-layer in forward.cu so the full model doesn't accumulate ~82GB of repacked
// expert weights (the cache is a decode-across-forwards optimization; per single forward it's per-layer scoped).
void tc_moe_clear_cache(){ for(auto& kv : g_tc_cache){ cudaFree(kv.second.wpr); cudaFree(kv.second.wsr); } g_tc_cache.clear(); }

// ===================== REPACK-AT-LOAD (zero extra memory) =====================
// The repacked layout is the SAME byte-size as the fp4 weight (N*K/2), so we repack IN PLACE at load and the
// kernel reads the ORIGINAL scale b_s[N,K/32] directly (no wsr). Result: no 82GB doubling, no per-layer repack.
// Funnel-combine two 16B-aligned uint4 loads straddling an unaligned 16B read. off = base&15 (uniform across the
// kernel: every g*512/lane*16 is a multiple of 16). k0=off>>2, sh=(off&3)*8. Register-based, uniform branch on k0.
__device__ __forceinline__ uint4 tcm_funnel16(uint4 A, uint4 B, int k0, unsigned sh){
    uint4 r;
    if(k0==0){r.x=__funnelshift_r(A.x,A.y,sh);r.y=__funnelshift_r(A.y,A.z,sh);r.z=__funnelshift_r(A.z,A.w,sh);r.w=__funnelshift_r(A.w,B.x,sh);}
    else if(k0==1){r.x=__funnelshift_r(A.y,A.z,sh);r.y=__funnelshift_r(A.z,A.w,sh);r.z=__funnelshift_r(A.w,B.x,sh);r.w=__funnelshift_r(B.x,B.y,sh);}
    else if(k0==2){r.x=__funnelshift_r(A.z,A.w,sh);r.y=__funnelshift_r(A.w,B.x,sh);r.z=__funnelshift_r(B.x,B.y,sh);r.w=__funnelshift_r(B.y,B.z,sh);}
    else          {r.x=__funnelshift_r(A.w,B.x,sh);r.y=__funnelshift_r(B.x,B.y,sh);r.z=__funnelshift_r(B.y,B.z,sh);r.w=__funnelshift_r(B.z,B.w,sh);}
    return r;   // off==0 (k0=0,sh=0) -> r=A (identity)
}
// Pre-packed kernel: weight already in wpr layout; scale read from original b_s (per-32, one 32-block=2 k-tiles).
__global__ void tc_w4a8_pp_kernel(float* out, const uint8_t* wpr, const float* b_s, const __half* x16, int M, int N, int K, int off){
    int lane=threadIdx.x&31, gid=lane>>2, t4=lane&3;
    int warp=threadIdx.x>>5; int n_block=blockIdx.x*TCM_WARPS+warp; if((long)n_block*8>=N) return; int n0=n_block*8;
    float c[4]={0,0,0,0}; int kg8=K/128, Ks32=K/32; int k0f=off>>2; unsigned shf=(off&3)*8;
    const uint8_t* wb = wpr + (long)n_block*kg8*512;
    const __half* xg0 = x16 + (size_t)gid*K, *xg8 = x16 + (size_t)(gid+8)*K;
    const float* bsr = b_s + (long)(n0+gid)*Ks32;   // original per-32 scale for this lane's weight row n0+gid
    bool m0=gid<M, m8=(gid+8)<M;
    for(int g=0; g<kg8; ++g){
        // ALIGNED coalesced load via funnel-shift: the in-place repacked weight is at an arbitrary WeightStore byte
        // offset. Load the two 16B-aligned uint4 straddling this lane's 16B, funnel-combine by the constant `off`.
        // Recovers coalescing (vs the byte-load fallback) while staying alignment-correct.
        const uint8_t* wa = wb + (long)g*512 + lane*16 - off;
        uint4 A=__ldcs((const uint4*)wa), B=__ldcs((const uint4*)(wa+16));
        uint4 W=tcm_funnel16(A,B,k0f,shf); const uint8_t* wby=(const uint8_t*)&W;
        #pragma unroll
        for(int kl=0; kl<8; ++kl){ int k_tile=g*8+kl, k0=k_tile*16;
            unsigned a[4];
            a[0]=m0? *(const unsigned*)(xg0+k0+2*t4)   : 0u;
            a[1]=m8? *(const unsigned*)(xg8+k0+2*t4)   : 0u;
            a[2]=m0? *(const unsigned*)(xg0+k0+2*t4+8) : 0u;
            a[3]=m8? *(const unsigned*)(xg8+k0+2*t4+8) : 0u;
            __half2 sc2 = __half2half2(__float2half(bsr[k_tile/2]));  // 32-block = k_tile/2 (16 per k-tile)
            __half2 b0 = __hmul2(tcm_fp4x2(wby[2*kl]),   sc2);
            __half2 b1 = __hmul2(tcm_fp4x2(wby[2*kl+1]), sc2);
            unsigned bb[2]; bb[0]=*(unsigned*)&b0; bb[1]=*(unsigned*)&b1;
            mma_m16n8k16(c, a, bb);
        }
    }
    int cn=2*t4;
    if(gid<M   && n0+cn  <N) out[(size_t)gid*N   + n0+cn  ]=c[0];
    if(gid<M   && n0+cn+1<N) out[(size_t)gid*N   + n0+cn+1]=c[1];
    if(gid+8<M && n0+cn  <N) out[(size_t)(gid+8)*N + n0+cn ]=c[2];
    if(gid+8<M && n0+cn+1<N) out[(size_t)(gid+8)*N + n0+cn+1]=c[3];
}
// Repack one fp4 weight IN PLACE (via a reused temp of size N*K/2). Call once per routed expert weight at load.
void tc_repack_weight_inplace(uint8_t* w_fp4, int N, int K, uint8_t* temp, cudaStream_t s){
    long tw=(long)(N/8)*(K/128)*32*8; size_t bytes=(size_t)(N/8)*(K/128)*512;   // == N*K/2
    k_repack_w<<<(tw+255)/256,256,0,s>>>(temp, w_fp4, N, K);
    CT(cudaMemcpyAsync(w_fp4, temp, bytes, cudaMemcpyDeviceToDevice, s));
}
// Drop-in for fp4_gemm, but B_fp4 is ALREADY repacked (via tc_repack_weight_inplace) and b_s is the ORIGINAL scale.
void tc_fp4_gemm_pp(float* C, const uint8_t* A_fp8, const float* a_s, const uint8_t* Bpacked, const float* b_s,
                    int M, int N, int K, cudaStream_t s){
    __half* x16; CT(cudaMalloc(&x16,(size_t)M*K*2));
    k_a_to_fp16<<<((long)M*K+255)/256,256,0,s>>>(x16, A_fp8, a_s, M, K);
    int off = (int)((uintptr_t)Bpacked & 15);   // constant per weight; funnel-shift makes loads aligned+coalesced
    dim3 grid((N/8 + TCM_WARPS-1)/TCM_WARPS); tc_w4a8_pp_kernel<<<grid, TCM_WARPS*32, 0, s>>>(C, Bpacked, b_s, x16, M, N, K, off);
    CT(cudaStreamSynchronize(s)); cudaFree(x16);
}
// Auto variant (drop-in for tc_fp4_gemm): repack B IN PLACE the first time it's seen (zero extra mem), then run
// the pre-packed GEMM. First forward warms up (repacks); every later forward is pure pp GEMM. Reused temp buffer.
#include <unordered_set>
static std::unordered_set<const void*> g_pp_done;
static uint8_t* g_pp_tmp=nullptr; static size_t g_pp_tmpsz=0;
void tc_fp4_gemm_pp_auto(float* C, const uint8_t* A_fp8, const float* a_s, const uint8_t* B_fp4, const float* b_s,
                         int M, int N, int K, cudaStream_t s){
    if(g_pp_done.find(B_fp4)==g_pp_done.end()){
        size_t bytes=(size_t)(N/8)*(K/128)*512;                 // == N*K/2
        if(bytes>g_pp_tmpsz){ if(g_pp_tmp) cudaFree(g_pp_tmp); CT(cudaMalloc(&g_pp_tmp,bytes)); g_pp_tmpsz=bytes; }
        tc_repack_weight_inplace((uint8_t*)B_fp4, N, K, g_pp_tmp, s);   // overwrite the fp4 weight with its repacked layout
        g_pp_done.insert(B_fp4);
    }
    tc_fp4_gemm_pp(C, A_fp8, a_s, B_fp4, b_s, M, N, K, s);
}

// Idempotently repack one expert weight IN PLACE (shared g_pp_done set with pp_auto, so a weight already warmed
// by either path is never re-repacked -> bytes stay identical -> grouped path is cosine-1.0 with the pp path).
void tc_ensure_repacked(uint8_t* B_fp4, int N, int K, cudaStream_t s){
    if(g_pp_done.find(B_fp4)!=g_pp_done.end()) return;
    size_t bytes=(size_t)(N/8)*(K/128)*512;
    if(bytes>g_pp_tmpsz){ if(g_pp_tmp) cudaFree(g_pp_tmp); CT(cudaMalloc(&g_pp_tmp,bytes)); g_pp_tmpsz=bytes; }
    tc_repack_weight_inplace(B_fp4, N, K, g_pp_tmp, s);
    g_pp_done.insert(B_fp4);
}
// fp8[M,K]+a_s -> fp16[M,K] (act-scale folded), exposed so the grouped path converts ALL gathered rows once.
void tc_a_to_fp16(__half* x16, const uint8_t* A_fp8, const float* a_s, int M, int K, cudaStream_t s){
    k_a_to_fp16<<<((long)M*K+255)/256,256,0,s>>>(x16, A_fp8, a_s, M, K);
}

// ===================== GROUPED (zero-sync) W4A8 GEMM — STRUCTURAL_PLAN Step 1b =====================
// ONE launch over ALL experts. A "tile" = up to 16 rows of ONE expert (its own repacked weight+scale). The
// tile->expert map is built ON DEVICE from off[] (k_build_tiles) so the host never needs off[] -> removes the
// last per-layer host sync (the off[] D2H copy) that blocked CUDA-graph capture. Per-expert byte alignment of
// the in-place repacked weight is handled by funnel-shift, computed per tile (uniform across the block).
// Weights & scales are the SAME bytes the pp path uses -> identical mma -> cosine 1.0 vs the per-expert loop.

// For each expert e, emit ceil(me/16) tiles at rows off[e], off[e]+16, ...  (single thread; nr<=~160).
__global__ void k_build_tiles(int* tile_e, int* tile_row0, int* ntiles, const int* off, int nr, int step){
    if(threadIdx.x||blockIdx.x) return;
    int nt=0;
    for(int e=0;e<nr;++e){ int r0=off[e], r1=off[e+1];
        for(int r=r0;r<r1;r+=step){ tile_e[nt]=e; tile_row0[nt]=r; ++nt; } }
    *ntiles=nt;
}

// gridDim.x = N/8 (n-blocks); gridDim.y = maxtiles (host UPPER BOUND = total gathered rows; extra tiles early-exit).
__global__ void k_grouped_w4a8_kernel(float* out, const uint8_t* const* wptr, const float* const* sptr,
        const int* __restrict__ tile_e, const int* __restrict__ tile_row0, const int* __restrict__ ntiles,
        const int* __restrict__ off, const __half* x16all, int N, int K){
    int tile = blockIdx.y; if(tile >= *ntiles) return;
    int e = tile_e[tile]; int row0 = tile_row0[tile];
    int me = off[e+1]-row0; if(me>16) me=16;                    // rows this tile owns (<=16)
    const uint8_t* wprE = wptr[e]; const float* b_s = sptr[e];
    int off_b=(int)((uintptr_t)wprE & 15); int k0f=off_b>>2; unsigned shf=(off_b&3)*8;   // per-expert alignment
    int lane=threadIdx.x&31, gid=lane>>2, t4=lane&3;
    // OCCUPANCY (LOOP_LOG Finding 21): this launched <<<grid,32>>> — one warp per block. A
    // 32-thread block still consumes a whole block slot, so half the SM's warp slots were unusable
    // (ncu: theoretical occupancy 50%, achieved 46%, Compute 24.7% / Memory 37.3% = latency-bound,
    // not saturated). The warps are fully independent here (no __shared__, no __syncthreads), so
    // pack several per block, each taking its own n-block. Pure launch geometry: the per-warp math
    // is byte-identical, so this gates cosine-1.0 against the previous output.
    int n_block = blockIdx.x*(blockDim.x>>5) + (threadIdx.x>>5);
    if((long)n_block*8>=N) return; int n0=n_block*8;
    float c[4]={0,0,0,0}; int kg8=K/128, Ks32=K/32;
    const uint8_t* wb = wprE + (long)n_block*kg8*512;
    const __half* xg0 = x16all + (size_t)(row0+gid)*K, *xg8 = x16all + (size_t)(row0+gid+8)*K;
    const float* bsr = b_s + (long)(n0+gid)*Ks32;
    bool m0=gid<me, m8=(gid+8)<me;
    for(int g=0; g<kg8; ++g){
        const uint8_t* wa = wb + (long)g*512 + lane*16 - off_b;             // funnel-aligned coalesced load
        uint4 A=__ldcs((const uint4*)wa), B=__ldcs((const uint4*)(wa+16));
        uint4 W=tcm_funnel16(A,B,k0f,shf); const uint8_t* wby=(const uint8_t*)&W;
        #pragma unroll
        for(int kl=0; kl<8; ++kl){ int k_tile=g*8+kl, k0=k_tile*16;
            unsigned a[4];
            a[0]=m0? *(const unsigned*)(xg0+k0+2*t4)   : 0u;
            a[1]=m8? *(const unsigned*)(xg8+k0+2*t4)   : 0u;
            a[2]=m0? *(const unsigned*)(xg0+k0+2*t4+8) : 0u;
            a[3]=m8? *(const unsigned*)(xg8+k0+2*t4+8) : 0u;
            __half2 sc2 = __half2half2(__float2half(bsr[k_tile/2]));
            __half2 b0 = __hmul2(tcm_fp4x2(wby[2*kl]),   sc2);
            __half2 b1 = __hmul2(tcm_fp4x2(wby[2*kl+1]), sc2);
            unsigned bb[2]; bb[0]=*(unsigned*)&b0; bb[1]=*(unsigned*)&b1;
            mma_m16n8k16(c, a, bb);
        }
    }
    int cn=2*t4;
    if(gid<me   && n0+cn  <N) out[(size_t)(row0+gid)*N   + n0+cn  ]=c[0];
    if(gid<me   && n0+cn+1<N) out[(size_t)(row0+gid)*N   + n0+cn+1]=c[1];
    if(gid+8<me && n0+cn  <N) out[(size_t)(row0+gid+8)*N + n0+cn ]=c[2];
    if(gid+8<me && n0+cn+1<N) out[(size_t)(row0+gid+8)*N + n0+cn+1]=c[3];
}
// LAUNCH GEOMETRY (Finding 71's class, same fix as k_moe_prefix). The scan above is one thread
// walking nr=160 experts, once per layer, and it sits in `moe:group` — a region that moves ~0.9 MB
// and costs 2.66 ms of the K=5 verify, i.e. pure latency. One block, 256 threads, block scan of the
// per-expert tile counts. Emission order is unchanged (expert-ascending, then row-ascending), so
// tile_e[]/tile_row0[]/*ntiles are BIT-IDENTICAL to the serial version.
#define TCM_SCAN_T 256
__global__ void k_build_tiles_par(int* tile_e, int* tile_row0, int* ntiles, const int* __restrict__ off, int nr, int step){
    __shared__ int s[TCM_SCAN_T]; __shared__ int carry;
    if(threadIdx.x==0) carry=0;
    __syncthreads();
    for(int base=0; base<nr; base+=TCM_SCAN_T){
        int e = base + threadIdx.x;
        int r0=0, nt=0;
        if(e<nr){ r0=off[e]; nt=(off[e+1]-r0+step-1)/step; }   // ceil(me/step) tiles for this expert
        s[threadIdx.x]=nt;
        __syncthreads();
        for(int d=1; d<TCM_SCAN_T; d<<=1){
            int t = (threadIdx.x>=d) ? s[threadIdx.x-d] : 0;
            __syncthreads();
            s[threadIdx.x] += t;
            __syncthreads();
        }
        int excl = carry + s[threadIdx.x] - nt;          // this expert's first tile index
        for(int j=0;j<nt;++j){ tile_e[excl+j]=e; tile_row0[excl+j]=r0+step*j; }
        __syncthreads();
        if(threadIdx.x==TCM_SCAN_T-1) carry += s[TCM_SCAN_T-1];
        __syncthreads();
    }
    if(threadIdx.x==0) *ntiles = carry;
}
// Build tile descriptors on device from off[] (no host sync).
// DSV4_SERIAL_SCAN=1 restores the <<<1,1>>> kernel so the A/B stays reachable.
static void tc_build_tiles_step(int* tile_e, int* tile_row0, int* ntiles_d, const int* off_d, int nr,
                                cudaStream_t s, int step){
    static int serial = -1; if(serial<0) serial = getenv("DSV4_SERIAL_SCAN")!=nullptr;
    if(serial) k_build_tiles    <<<1,1,0,s>>>(tile_e, tile_row0, ntiles_d, off_d, nr, step);
    else       k_build_tiles_par<<<1,TCM_SCAN_T,0,s>>>(tile_e, tile_row0, ntiles_d, off_d, nr, step);
}
void tc_build_tiles(int* tile_e, int* tile_row0, int* ntiles_d, const int* off_d, int nr, cudaStream_t s){
    tc_build_tiles_step(tile_e, tile_row0, ntiles_d, off_d, nr, s, 16);
}
// ROW-GROUPS PER TILE (2026-08-26). include/moe.h has recorded since B9 that "EVERY tile re-reads its
// expert's whole weight matrix, so weight traffic is ntiles x (w1+w3+w2)". At decode that is invisible
// -- one tile per expert. At a PREFILL it is the whole cost: PS=845 puts 845*6/160 = ~32 rows on every
// expert, i.e. TWO 16-row tiles, i.e. every expert weight read TWICE. Measured on tools/moe_gemv_bench
// at the prefill grouping, the mma path holds a flat 87 GB/s of REAL traffic from 16 rows upward, so
// the tiles are the bytes and the bytes are the time: 160x16 = 16.3 ms, 160x32 = 32.8, 160x64 = 65.5,
// exactly linear in ceil(R/16) while the IDEAL traffic never changes.
//
// A tile that owns RG row-groups loads and dequantises the B fragment ONCE for all of them. The
// K-reduction order of every output element is untouched -- the same mma over the same k_tile
// sequence -- so this is bit-exact, and gate_grouped_moe proves it by memcmp rather than by this
// paragraph. RG=1 dispatches to the ORIGINAL kernel, byte for byte, so the default cannot regress.
// RG*NB > 4 IS REFUSED, and the reason is a trap worth naming. RG=2,NB=4 benched 6.74x faster than
// baseline -- and wrote 1/8 of the output: 5120 rows x 256 of 2048 columns. It was faster because it
// was doing an eighth of the work. ptxas reports no spills, so it is a launch that fails to cover its
// grid, not register pressure. A checksum across configs (MOE_BENCH_SUM=1 in tools/moe_gemv_bench)
// caught it; speed alone never would have.
//
// The clamp lives HERE, in the accessor, and not at the launch site. Clamping at the launch site was
// the first attempt and it was WORSE: tools/moe_gemv_bench builds its tiles from tcm_rowg() while the
// kernel used the clamped value, so tiles were strided 32 and read as 16 and exactly half the rows
// vanished. One value, read by both, or they disagree.
int tcm_rowg();
int tcm_nblk(){ static int nb=-1;
    if(nb<0){ const char* e=getenv("MOE_NBLK"); nb = e?atoi(e):8;   // DEFAULT 8: w1w3 2005->1809 ms end to end, bit-exact if(nb!=1&&nb!=2&&nb!=4&&nb!=8) nb=1;
              // VALID COMBOS ARE ENUMERATED, NOT DERIVED FROM A PRODUCT. The first guard was
              // "RG*NB <= 4", which reads like a register-budget rule and is not one: RG=1,NB=8
              // (product 8) is CORRECT and RG=2,NB=4 (product 8) writes 1/8 of the output. Raising
              // the cap to 8 to admit the first silently re-admitted the second. The failure is
              // specific to RG>=2 combined with NB>=4, so that is what the guard says.
              // Measured, combination by combination, against a full-output checksum:
              //   RG=1: NB 1,2,4,8 all correct        RG=2: NB 1,2 correct, NB=4 writes 1/8
              //   RG=4: NB=1 correct, NB=2 writes 1/4
              // So the rule is "RG=1 scales to NB=8; for RG>=2 the product must stay <= 4" -- which is
              // NOT the same as a flat product cap, since RG=1/NB=8 has product 8 and is correct.
              const int rgv = tcm_rowg();
              const bool ok = (rgv==1) ? (nb==1||nb==2||nb==4||nb==8)
                                       : (rgv*nb <= 4);
              if(!ok){
                  fprintf(stderr,"[moe] MOE_ROWG=%d x MOE_NBLK=%d is not a verified combination; NB forced to 1\n", rgv, nb);
                  nb = 1; } }
    return nb; }
int tcm_rowg(){ static int rg=-1;
    if(rg<0){ const char* e=getenv("MOE_ROWG"); rg = e?atoi(e):1; if(rg!=1&&rg!=2&&rg!=4) rg=1; }
    return rg; }
void tc_build_tiles_rg(int* tile_e, int* tile_row0, int* ntiles_d, const int* off_d, int nr,
                       cudaStream_t s, int rowg){
    tc_build_tiles_step(tile_e, tile_row0, ntiles_d, off_d, nr, s, 16*rowg);
}

// ===================== M=1 fp4 GEMV (small-M decode, ORIGINAL fp4 layout, no repack) =====================
// One warp per (output n, tile); loops the tile's <=16 rows. Reads the ORIGINAL packed fp4 weight row
// (uint4-vectorized, coalesced) + fp8 act (uint4) + e8m0 per-32 scale in-register. Bandwidth-bound — beats the
// m16-tile mma at small M (which is mma-latency bound). Gated cosine vs fp4_gemm (tests/gate_fp4_gemv).
__constant__ float GEMV_E2M1[8] = {0.f,0.5f,1.f,1.5f,2.f,3.f,4.f,6.f};
__device__ __forceinline__ float gv_fp4(uint8_t nib){ float m=GEMV_E2M1[nib&7]; return (nib&8)?-m:m; }
__device__ __forceinline__ float gv_e4m3(uint8_t b){ __half_raw r=__nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)b,__NV_E4M3); return __half2float(*reinterpret_cast<__half*>(&r)); }
template<int RB, bool ALIGN8>
__global__ void k_grouped_fp4_gemv_e8m0(float* out, const uint8_t* const* wptr, const uint8_t* const* sptr,
        const int* __restrict__ tile_e, const int* __restrict__ tile_row0, const int* __restrict__ ntiles,
        const int* __restrict__ off, const uint8_t* Xq, const float* Xs, int N, int K){
    int tile=blockIdx.y; if(tile>=*ntiles) return;
    int e=tile_e[tile], row0=tile_row0[tile]; int me=off[e+1]-row0; if(me>16)me=16;
    // OUTPUT-COLUMN BLOCKING, BN=2 (IMPLEMENTATION_PLAN Tier-1 #1).
    // This used one warp per output n, so the ACTIVATION uint4 pair was re-loaded for every output
    // column even though every column at the same k reads the SAME activation. A rebuild of our
    // exact MoE shape measured BN=1 at 155-160 GB/s and BN=2 at 242-249 (101-104% of achievable):
    // at BN=1 each 16-byte weight load is paired with a fresh activation fetch, and the inner loop
    // becomes issue-rate bound (20 SMs x 4 sched x 1.575 GHz = 126 G warp-inst/s, and a BN=1 body
    // burns ~50 of them per 512 B of weight). At BN>=2 the activation registers are reused and the
    // instruction count per weight byte roughly halves.
    // Per-n accumulation order over kb is UNCHANGED -> bit-exact.
    // BN is a COMPILE-TIME knob so the register/occupancy trade can be A/B'd without editing the
    // kernel (`-DMOE_BN=1`). Default 2 = shipped, byte-for-byte.
    const int BN = MOE_BN;
    int warp=(blockIdx.x*blockDim.x+threadIdx.x)>>5; int nbase=warp*BN; if(nbase>=N) return; int lane=threadIdx.x&31;
    const int nact = (nbase+BN<=N) ? BN : (N-nbase);
    const uint8_t* Wn0 = wptr[e] + (size_t)nbase*(K/2);
    int nb32 = K/32;                                     // 32-weight blocks = 16 bytes each
    int off_b=(int)((uintptr_t)Wn0 & 15); int k0f=off_b>>2; unsigned shf=(off_b&3)*8;   // funnel-align
    // ROW AMORTISATION (LOOP_LOG Finding 64). The weight loads used to sit INSIDE this row loop, so
    // an expert serving `me` rows had its whole weight matrix re-read `me` times. Actual traffic was
    // therefore rows x expert_bytes, not union x expert_bytes: at K=5 that is 17.25 GB where 10.08
    // would do, and it is why the MoE looked "at the roofline" (217 GB/s) while moving 1.71x the
    // bytes it needs. The measured union is 6.00/9.67/12.58/15.16/17.53 at K=1..5 (DSV4_MOEUNION=1),
    // NOT the 29.9 the ROOFLINE.md byte model assumed — the five block positions share most of their
    // experts, and the model that said they do not is what hid this for the whole project.
    //
    // This is the same "read B once, dot it against all M rows" transformation Findings 40/42/43
    // applied to ogroup, lm_head and gemm_fp32; the MoE grouped GEMV never got it. Weights and their
    // scales are hoisted out of the row loop and RB rows are accumulated against one load. RB is a
    // template parameter so RB=1 reproduces the old kernel exactly for A/B (MOE_RB=1).
    //
    // Registers are why RB is chunked rather than "all me rows": me can be up to 16 and a fixed
    // acc[16][BN] would cost 32 accumulators live regardless of the real row count, which is exactly
    // the occupancy trap Finding 28 hit on the fp8 GEMV. RB=8 covers the verify (me<=5 at K=5) in one
    // chunk while holding 16 accumulators.
    //
    // Per-(row,n) accumulation order over kb is UNCHANGED and the inner half2 block math is
    // untouched, so this is BIT-EXACT with the previous kernel, not merely close.
    for(int rb=0; rb<me; rb+=RB){
        const int rn = (me-rb < RB) ? (me-rb) : RB;
        float acc[RB][BN];
        #pragma unroll
        for(int r=0;r<RB;++r){
            #pragma unroll
            for(int u=0;u<BN;++u) acc[r][u]=0.f; }
        for(int kb=lane; kb<nb32; kb+=32){              // lane -> whole 32-weight block (16B), coalesced
            // FUNNEL vs TWO uint2 (LOOP_LOG Finding 72). The funnel exists because expert weight
            // pointers are misaligned, and F66 measured that they are misaligned by a CONSTANT 8
            // bytes: 43,470 of 44,436 expert tensors at data_offset%16 == 8, 966 at 12, none at 0.
            // Residue 8 means the address is not 16-byte aligned but IS 8-byte aligned — so two
            // `uint2` loads fetch exactly the 16 bytes wanted, with no second 16-byte load, no
            // funnel shift, and half the weight registers (2 uint2 = 4 regs vs 2 uint4 = 8, per BN).
            // Same instruction count, half the bytes requested, and this kernel is occupancy-limited
            // at 69 registers / 54.7% (ncu, RB=4).
            //
            // ALIGN8 is decided on the HOST by checking every expert pointer, not per block: the
            // residue is a property of the shard and 966 tensors really are at 12, so a kernel-side
            // branch would keep both paths live and give back the registers it was trying to save.
            uint4 WAv[BN], WBv[BN]; float wsv[BN];
            #pragma unroll
            for(int u=0; u<BN; ++u){
                if(u>=nact) break;
                const uint8_t* Wn = wptr[e] + (size_t)(nbase+u)*(K/2);
                const uint8_t* Sn = sptr[e] + (size_t)(nbase+u)*(K/32);
                if(ALIGN8){
                    const uint8_t* wa=Wn+(size_t)kb*16;                 // == off_b (mod 16) -> 8B aligned
                    uint2 lo=__ldcs((const uint2*)wa), hi=__ldcs((const uint2*)(wa+8));
                    WAv[u]=make_uint4(lo.x,lo.y,hi.x,hi.y);
                } else {
                    const uint8_t* wa=Wn+(size_t)kb*16-off_b;
                    WAv[u]=__ldcs((const uint4*)wa); WBv[u]=__ldcs((const uint4*)(wa+16));
                }
                wsv[u]=exp2f((float)Sn[kb]-127.f);
            }
            for(int r=0;r<rn;++r){
                const uint8_t* Aq = Xq + (size_t)(row0+rb+r)*K;
                const float*  As = Xs + (size_t)(row0+rb+r)*(K/128);
                const float asc=As[kb/4];                // act scale per-128 = per 4 of the 32-blocks
                uint4 a0 =*(const uint4*)(Aq+(size_t)kb*32);
                uint4 a1 =*(const uint4*)(Aq+(size_t)kb*32+16);
                const uint8_t* ab0=(const uint8_t*)&a0; const uint8_t* ab1=(const uint8_t*)&a1;
                #pragma unroll
                for(int u=0; u<BN; ++u){
                    if(u>=nact) break;
                    const float ws=wsv[u];
                    uint4 w16 = ALIGN8 ? WAv[u] : tcm_funnel16(WAv[u],WBv[u],k0f,shf);
                    const uint8_t* wb=(const uint8_t*)&w16;
                    // Dequant is REDONE per row rather than cached: it is ALU on a memory-bound
                    // kernel, and holding the dequantised half2 values across the row loop is the
                    // register blow-up Finding 43 already measured as a loss on the ogroup twin.
                    __half2 acc2 = __floats2half2_rn(0.f, 0.f);
                    #pragma unroll
                    for(int j=0;j<16;++j){
                        const uint8_t byte = wb[j];
                        const uint8_t lo = (j<8)? ab0[2*j]   : ab1[2*(j-8)];
                        const uint8_t hi = (j<8)? ab0[2*j+1] : ab1[2*(j-8)+1];
                        __half2 w2 = tcm_fp4x2(byte);
                        __half2_raw ar = __nv_cvt_fp8x2_to_halfraw2(
                            (__nv_fp8x2_storage_t)((unsigned short)lo | ((unsigned short)hi << 8)), __NV_E4M3);
                        acc2 = __hfma2(*reinterpret_cast<__half2*>(&ar), w2, acc2);
                    }
                    const float sub = __low2float(acc2) + __high2float(acc2);
                    acc[r][u] += sub * asc * ws;
                }
            }
        }
        // Store all BN columns. This used to be hardcoded to two, which silently made BN a lie: at
        // BN=4 the compiler dead-codes accumulators 2..3 and with them HALF the weight loads, so the
        // kernel "ran" 1.65x faster while writing half its outputs. Any BN sweep against the old
        // store was measuring work deletion, not blocking.
        #pragma unroll
        for(int r=0;r<RB;++r){ if(r>=rn) break;
            #pragma unroll
            for(int u=0;u<BN;++u){ if(u>=nact) break;
                float av=acc[r][u];
                #pragma unroll
                for(int o=16;o>0;o>>=1) av+=__shfl_down_sync(0xffffffff,av,o);
                if(lane==0) out[(size_t)(row0+rb+r)*N + nbase+u] = av;
            }
        }
    }
}

void tc_fp4_grouped_gemv_e8m0(float* out, const uint8_t* Xq, const float* Xs, const uint8_t* const* wptr_d,
        const uint8_t* const* sptr_d, const int* off_d, const int* tile_e, const int* tile_row0, const int* ntiles_d,
        int maxtiles, int N, int K, cudaStream_t s, int rows_hint, int align8){
    // BN=2 output columns per warp -> half the warps. 128 threads/block matched the measured
    // optimum (BN=2 @128 thr = 242-249 GB/s; 256 thr was consistently worse at every BN).
    const int BN=MOE_BN, warps_needed=(N+BN-1)/BN;
    int threads=128; dim3 grid((warps_needed*32+threads-1)/threads, maxtiles);
    // RB = rows accumulated against ONE weight load (Finding 64). MOE_RB=1 restores the pre-Finding-64
    // kernel, which re-read each expert's weights once per row it served; it is kept reachable because
    // it is the A/B arm and the fallback, not because it is ever preferable.
    // RB must follow the ROWS PER EXPERT, which is bounded by the batch. At M=1 decode every expert
    // serves exactly one row, so there is nothing to amortise and a large RB only costs registers —
    // measured as a 2.9% base-AR regression (12.56 -> 12.19 tok/s) when RB was pinned at 8. Pick the
    // smallest power of two that covers `rows_hint` (= bs), so base AR gets the original kernel
    // byte-for-byte and the K=5 verify gets one weight load per expert.
    // RB follows ROWS PER EXPERT, which is NOT bs. Measured with DSV4_MOEUNION=1 at K=5: the union is
    // 17.53 experts over 30 rows and the distribution is heavily skewed — ~70% of experts serve
    // exactly ONE row, ~88% serve <=2, and the maximum is 5. So the rows an expert actually has are
    // far below bs, and sizing RB from bs over-allocates acc[RB][BN], which is live regardless of the
    // real row count (the Finding 28 occupancy trap). ncu on the measured grouping, 1.67 rows/expert:
    //     RB=1  441.9 us  55 reg  71.4% occ      RB=4  434.4 us  68 reg  55.3% occ
    //     RB=2  423.8 us  62 reg  63.1% occ      RB=8  497.3 us  85 reg  39.4% occ   <- what F64 shipped
    // and in situ the K=5 verify goes 141.53 -> 139.72 ms moving RB 8 -> 2 (moe:w2 23.88 -> 21.33).
    // At bs=1 every expert has exactly one row, so RB=1 is both optimal and the original kernel.
    // F65 picked RB=2 from a probe whose grouping clamped rows-per-expert at 2, so no tile ever
    // needed chunking and the sweep could not see what RB is FOR. On the MEASURED histogram
    // (12x1 + 3x2 + 1x3 + 1x4 + 1x5 over 18 experts, DSV4_MOEUNION=1) an expert with me=5 costs
    // THREE weight reads at RB=2 and one at RB=4, and the ranking inverts:
    //     capped-at-2 probe:  RB=1 441.9  RB=2 423.8  RB=4 434.4  RB=8 497.3   -> RB=2
    //     measured histogram: RB=1 559.1  RB=2 534.7  RB=4 499.8  RB=8 530.6   -> RB=4
    // RB=8 loses at both because its 16 live accumulators cost more occupancy than the last few
    // re-reads are worth. At bs=1 every expert has exactly one row, so RB=1 is optimal and is also
    // the original kernel byte-for-byte.
    const char* e_=getenv("MOE_RB"); int RB = e_ ? atoi(e_) : (rows_hint<=1 ? 1 : rows_hint<=2 ? 2 : 4);
    if(RB!=1&&RB!=2&&RB!=4&&RB!=8) RB=8;
    #define GV_LAUNCH(RBV) { if(align8) k_grouped_fp4_gemv_e8m0<RBV,true ><<<grid,threads,0,s>>>(out,wptr_d,sptr_d,tile_e,tile_row0,ntiles_d,off_d,Xq,Xs,N,K); \
                             else       k_grouped_fp4_gemv_e8m0<RBV,false><<<grid,threads,0,s>>>(out,wptr_d,sptr_d,tile_e,tile_row0,ntiles_d,off_d,Xq,Xs,N,K); }
    switch(RB){ case 1: GV_LAUNCH(1) break; case 2: GV_LAUNCH(2) break;
                case 4: GV_LAUNCH(4) break; default: GV_LAUNCH(8) break; }
    #undef GV_LAUNCH
}

// NATIVE-e8m0 grouped GEMM: scale ptrs point to the ORIGINAL e8m0 scale BYTES (F8_E8M0) in the WeightStore —
// exp2f(byte-127) is computed in-register (bit-identical to the pre-dequanted f32 pow2). This removes the
// per-layer-per-token scale dequant (160x3 mallocs+kernels/layer) AND keeps the scale pointers persistent
// (no dequant buffer) -> the dominant decode cost. Only the scale read differs from k_grouped_w4a8_kernel.
__global__ void k_grouped_w4a8_e8m0_kernel(float* out, const uint8_t* const* wptr, const uint8_t* const* sptr,
        const int* __restrict__ tile_e, const int* __restrict__ tile_row0, const int* __restrict__ ntiles,
        const int* __restrict__ off, const __half* x16all, int N, int K){
    int tile = blockIdx.y; if(tile >= *ntiles) return;
    int e = tile_e[tile]; int row0 = tile_row0[tile];
    int me = off[e+1]-row0; if(me>16) me=16;
    const uint8_t* wprE = wptr[e]; const uint8_t* b_s = sptr[e];       // b_s = e8m0 bytes [N, K/32]
    int off_b=(int)((uintptr_t)wprE & 15); int k0f=off_b>>2; unsigned shf=(off_b&3)*8;
    int lane=threadIdx.x&31, gid=lane>>2, t4=lane&3;
    // OCCUPANCY (LOOP_LOG Finding 21): this launched <<<grid,32>>> — one warp per block. A
    // 32-thread block still consumes a whole block slot, so half the SM's warp slots were unusable
    // (ncu: theoretical occupancy 50%, achieved 46%, Compute 24.7% / Memory 37.3% = latency-bound,
    // not saturated). The warps are fully independent here (no __shared__, no __syncthreads), so
    // pack several per block, each taking its own n-block. Pure launch geometry: the per-warp math
    // is byte-identical, so this gates cosine-1.0 against the previous output.
    int n_block = blockIdx.x*(blockDim.x>>5) + (threadIdx.x>>5);
    if((long)n_block*8>=N) return; int n0=n_block*8;
    float c[4]={0,0,0,0}; int kg8=K/128, Ks32=K/32;
    const uint8_t* wb = wprE + (long)n_block*kg8*512;
    const __half* xg0 = x16all + (size_t)(row0+gid)*K, *xg8 = x16all + (size_t)(row0+gid+8)*K;
    const uint8_t* bsr = b_s + (long)(n0+gid)*Ks32;
    bool m0=gid<me, m8=(gid+8)<me;
    // NEGATIVE RESULT (IMPLEMENTATION_PLAN Tier-1 #2, reverted). Software-pipelining this loop —
    // prefetching iteration g+1's two uint4 weight loads while consuming g — measured WORSE:
    // 101.7 -> 102.9 ms/tok on the full model, and 121.6 -> 118.0 GB/s on the bench. The loop
    // already issues A and B together (ILP=2), and the prefetch costs 8 extra registers against a
    // Block-Limit-Registers of 48, which loses more in occupancy than it gains in latency hiding.
    // The ILP lever that DID pay was on the dense GEMV (Opt #7), which was genuinely at ILP=1.
    for(int g=0; g<kg8; ++g){
        const uint8_t* wa = wb + (long)g*512 + lane*16 - off_b;
        uint4 A=__ldcs((const uint4*)wa), B=__ldcs((const uint4*)(wa+16));
        uint4 W=tcm_funnel16(A,B,k0f,shf); const uint8_t* wby=(const uint8_t*)&W;
        #pragma unroll
        for(int kl=0; kl<8; ++kl){ int k_tile=g*8+kl, k0=k_tile*16;
            unsigned a[4];
            a[0]=m0? *(const unsigned*)(xg0+k0+2*t4)   : 0u;
            a[1]=m8? *(const unsigned*)(xg8+k0+2*t4)   : 0u;
            a[2]=m0? *(const unsigned*)(xg0+k0+2*t4+8) : 0u;
            a[3]=m8? *(const unsigned*)(xg8+k0+2*t4+8) : 0u;
            __half2 sc2 = __half2half2(__float2half(exp2f((float)bsr[k_tile/2]-127.f)));  // e8m0 -> pow2 in-register
            __half2 b0 = __hmul2(tcm_fp4x2(wby[2*kl]),   sc2);
            __half2 b1 = __hmul2(tcm_fp4x2(wby[2*kl+1]), sc2);
            unsigned bb[2]; bb[0]=*(unsigned*)&b0; bb[1]=*(unsigned*)&b1;
            mma_m16n8k16(c, a, bb);
        }
    }
    int cn=2*t4;
    if(gid<me   && n0+cn  <N) out[(size_t)(row0+gid)*N   + n0+cn  ]=c[0];
    if(gid<me   && n0+cn+1<N) out[(size_t)(row0+gid)*N   + n0+cn+1]=c[1];
    if(gid+8<me && n0+cn  <N) out[(size_t)(row0+gid+8)*N + n0+cn ]=c[2];
    if(gid+8<me && n0+cn+1<N) out[(size_t)(row0+gid+8)*N + n0+cn+1]=c[2+1];
}
// ============ RG row-groups x NB n-blocks per warp: load once, use many ==========================
// TWO redundancies, measured at the PREFILL grouping on tools/moe_gemv_bench (160 experts):
//
//   WEIGHTS. include/moe.h has recorded since B9 that every 16-row tile re-reads its expert's whole
//   matrix. PS=845 puts 845*6/160 = ~32 rows on each expert = TWO tiles = every weight read twice.
//   RG row-groups in one tile load and dequantise the B fragment ONCE for all of them.
//
//   ACTIVATIONS, and this one is bigger. A warp owned 8 N-columns, i.e. ONE mma per A-fragment load,
//   so the activation tensor is re-read once per n-block: 42 MB x 256 = 10.7 GB against the weights'
//   1.43 GB. The A fragment depends on (row,k) and NOT on n, so NB n-blocks per warp issue NB mma
//   against one A load and divide that traffic by NB. Same lever MOE_BN=2 already took on the GEMV
//   path in this file (155 -> 242 GB/s); the mma path never got it.
//
// Every output element still accumulates over the same k_tile sequence in the same order, so this is
// bit-exact. RG=1,NB=1 dispatches to the ORIGINAL kernel so the default cannot regress.
template<int RG, int NB>
__global__ void k_grouped_w4a8_e8m0_kernel_rg(float* out, const uint8_t* const* wptr, const uint8_t* const* sptr,
        const int* __restrict__ tile_e, const int* __restrict__ tile_row0, const int* __restrict__ ntiles,
        const int* __restrict__ off, const __half* x16all, int N, int K){
    int tile = blockIdx.y; if(tile >= *ntiles) return;
    int e = tile_e[tile]; int row0 = tile_row0[tile];
    int me_tot = off[e+1]-row0; if(me_tot > 16*RG) me_tot = 16*RG;
    const uint8_t* wprE = wptr[e]; const uint8_t* b_s = sptr[e];
    int off_b=(int)((uintptr_t)wprE & 15); int k0f=off_b>>2; unsigned shf=(off_b&3)*8;
    int lane=threadIdx.x&31, gid=lane>>2, t4=lane&3;
    int wbase = (blockIdx.x*(blockDim.x>>5) + (threadIdx.x>>5))*NB;
    if((long)wbase*8>=N) return;
    float c[4*RG*NB];
    #pragma unroll
    for(int i=0;i<4*RG*NB;++i) c[i]=0.f;
    int kg8=K/128, Ks32=K/32;
    const uint8_t* wbn[NB]; const uint8_t* bsrn[NB]; bool nok[NB];
    #pragma unroll
    for(int q=0;q<NB;++q){ int nb=wbase+q; nok[q] = ((long)nb*8 < N); int nu = nok[q]?nb:0;
        wbn[q]  = wprE + (long)nu*kg8*512;
        bsrn[q] = b_s  + (long)(nu*8+gid)*Ks32; }
    const __half* xg0[RG]; const __half* xg8[RG]; int mer[RG];
    #pragma unroll
    for(int r=0;r<RG;++r){ int base = row0 + r*16;
        xg0[r] = x16all + (size_t)(base+gid)*K;
        xg8[r] = x16all + (size_t)(base+gid+8)*K;
        int m = me_tot - r*16; if(m<0) m=0; if(m>16) m=16; mer[r]=m; }
    for(int g=0; g<kg8; ++g){
        uint4 W[NB];
        #pragma unroll
        for(int q=0;q<NB;++q){ const uint8_t* wa = wbn[q] + (long)g*512 + lane*16 - off_b;
            uint4 A=__ldcs((const uint4*)wa), B=__ldcs((const uint4*)(wa+16));
            W[q]=tcm_funnel16(A,B,k0f,shf); }
        #pragma unroll
        for(int kl=0; kl<8; ++kl){ int k_tile=g*8+kl, k0=k_tile*16;
            unsigned bb[NB][2];
            #pragma unroll
            for(int q=0;q<NB;++q){ const uint8_t* wby=(const uint8_t*)&W[q];
                __half2 sc2 = __half2half2(__float2half(exp2f((float)bsrn[q][k_tile/2]-127.f)));
                __half2 b0 = __hmul2(tcm_fp4x2(wby[2*kl]),   sc2);
                __half2 b1 = __hmul2(tcm_fp4x2(wby[2*kl+1]), sc2);
                bb[q][0]=*(unsigned*)&b0; bb[q][1]=*(unsigned*)&b1; }
            #pragma unroll
            for(int r=0;r<RG;++r){
                unsigned a[4];
                bool m0 = gid < mer[r], m8 = (gid+8) < mer[r];
                a[0]=m0? *(const unsigned*)(xg0[r]+k0+2*t4)   : 0u;
                a[1]=m8? *(const unsigned*)(xg8[r]+k0+2*t4)   : 0u;
                a[2]=m0? *(const unsigned*)(xg0[r]+k0+2*t4+8) : 0u;
                a[3]=m8? *(const unsigned*)(xg8[r]+k0+2*t4+8) : 0u;
                #pragma unroll
                for(int q=0;q<NB;++q) mma_m16n8k16(c+4*(r*NB+q), a, bb[q]);
            }
        }
    }
    int cn=2*t4;
    #pragma unroll
    for(int r=0;r<RG;++r){ int base = row0 + r*16, m = mer[r];
        #pragma unroll
        for(int q=0;q<NB;++q){
            if(!nok[q]) continue;
            int n0q=(wbase+q)*8, i=4*(r*NB+q);
            if(gid<m   && n0q+cn  <N) out[(size_t)(base+gid)*N   + n0q+cn  ]=c[i+0];
            if(gid<m   && n0q+cn+1<N) out[(size_t)(base+gid)*N   + n0q+cn+1]=c[i+1];
            if(gid+8<m && n0q+cn  <N) out[(size_t)(base+gid+8)*N + n0q+cn  ]=c[i+2];
            if(gid+8<m && n0q+cn+1<N) out[(size_t)(base+gid+8)*N + n0q+cn+1]=c[i+3];
        }
    }
}
// Warps per block for the grouped MoE GEMMs. 1 (the old value) caps theoretical occupancy at 50%.
// Env-overridable so the A/B is reproducible: MOE_WPB=1 restores the previous behaviour exactly.
static int moe_wpb(){
    static int w = [](){ const char* e=getenv("MOE_WPB"); int v = e?atoi(e):4;
                         if(v<1) v=1; if(v>8) v=8; return v; }();
    return w;
}
void tc_fp4_grouped_gemm_e8m0(float* out, const __half* x16all, const uint8_t* const* wptr_d, const uint8_t* const* sptr_d,
        const int* off_d, const int* tile_e, const int* tile_row0, const int* ntiles_d,
        int maxtiles, int N, int K, cudaStream_t s){
    const int wpb = moe_wpb(), nb = N/8;
    dim3 grid((nb + wpb - 1)/wpb, maxtiles);
    // RG=1 keeps the ORIGINAL kernel so the shipped default is byte-for-byte what it always was.
    const int rg = tcm_rowg(), nbk = tcm_nblk();
    if(rg==1 && nbk==1){
        k_grouped_w4a8_e8m0_kernel<<<grid, 32*wpb, 0, s>>>(out, wptr_d, sptr_d, tile_e, tile_row0, ntiles_d, off_d, x16all, N, K);
        return;
    }
    dim3 gridn((nb + nbk*wpb - 1)/(nbk*wpb), maxtiles);
    #define TCM_LAUNCH(R,B) k_grouped_w4a8_e8m0_kernel_rg<R,B><<<gridn, 32*wpb, 0, s>>>(out, wptr_d, sptr_d, tile_e, tile_row0, ntiles_d, off_d, x16all, N, K)
    if(rg==1 && nbk==8) TCM_LAUNCH(1,8);
    else if(rg==1 && nbk==2) TCM_LAUNCH(1,2);
    else if(rg==1 && nbk==4) TCM_LAUNCH(1,4);
    else if(rg==2 && nbk==1) TCM_LAUNCH(2,1);
    else if(rg==2 && nbk==2) TCM_LAUNCH(2,2);
    else if(rg==4 && nbk==1) TCM_LAUNCH(4,1);
    else TCM_LAUNCH(1,1);
    #undef TCM_LAUNCH
}
// Grouped W4A8: out[total,N] = per-tile (expert wptr[e]) mma over x16all rows. maxtiles = host upper bound on tiles.
void tc_fp4_grouped_gemm(float* out, const __half* x16all, const uint8_t* const* wptr_d, const float* const* sptr_d,
        const int* off_d, const int* tile_e, const int* tile_row0, const int* ntiles_d,
        int maxtiles, int N, int K, cudaStream_t s){
    const int wpb = moe_wpb(), nb = N/8;
    dim3 grid((nb + wpb - 1)/wpb, maxtiles);
    k_grouped_w4a8_kernel<<<grid, 32*wpb, 0, s>>>(out, wptr_d, sptr_d, tile_e, tile_row0, ntiles_d, off_d, x16all, N, K);
}
