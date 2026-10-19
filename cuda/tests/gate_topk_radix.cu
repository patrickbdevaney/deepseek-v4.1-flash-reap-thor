// gate_topk_radix.cu — prove the single-CTA radix select is BIT-IDENTICAL to the shipped warp
// selection sort on all four kernel shapes, then time both. DECODE_LADDER.md item 1.2.
//
// The shipped kernels do `topk` SEQUENTIAL argmax rounds over the row (item 1.1 spread each round
// across 32 lanes; the rounds themselves are still serial). At topk=512, T=3072 that is 512
// dependent rounds -- measured by 0.4 as 13.47 ms at ctx 12,288 and 12.5 % of the whole context
// term. The replacement is O(T) in a constant number of passes.
//
// WHAT THIS GATE HAS TO CATCH, and why each distribution is here:
//   mode 0  exponential decay          the realistic shape
//   mode 1  pseudo-uniform             generic
//   mode 2  HEAVY EXACT TIES (t*7%13)  the only place the ORDER can diverge while the SET agrees,
//                                      and order is load-bearing: sparse_attn sums selected rows in
//                                      order, so fp32 association changes if two equal scores swap.
//   mode 3  SIGNED ZEROS               `-0.0f == +0.0f` for the originals' `>` compare, but
//                                      ord(-0.0) < ord(+0.0) as raw bits. index_score CAN emit -0.0
//                                      (relu gives exactly 0, times a negative head weight).
//   mode 4  FLOOR-STRADDLING           values at and just under the -1e30f/-1e29f floor, plus rows
//                                      that are entirely floored, so `keff == 0` and `keff < topk`
//                                      are both exercised -- the two paths where the slot count and
//                                      the -1 padding can go wrong.
//   mode 5  NEGATIVE ONLY              all scores < 0, so the key's sign-flip branch is the live one.
//
//   nvcc -O2 -std=c++17 -gencode arch=compute_110a,code=sm_110a -I include tests/gate_topk_radix.cu -o build/gate_topk_radix
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <unistd.h>
#include <csignal>
#include <sys/wait.h>
#include "topk_radix.h"
#include "indexer.h"     // item 1.4: topk_scan_smem_optin / TOPK_SMEM / TOPK_LAUNCHED

#define CU_(x) do{ cudaError_t e=(x); if(e){ printf("CUDA %s @%d\n", cudaGetErrorString(e), __LINE__); exit(1);} }while(0)

// ---- shipped (item 1.1 warp scan), verbatim ---------------------------------------------------
__device__ __forceinline__ void warp_argmax_(float& best, int& bi){
    #pragma unroll
    for(int o=16;o>0;o>>=1){
        float ob = __shfl_down_sync(0xffffffffu, best, o);
        int   oi = __shfl_down_sync(0xffffffffu, bi,   o);
        if(ob > best || (ob == best && oi < bi)){ best = ob; bi = oi; }
    }
    bi = __shfl_sync(0xffffffffu, bi, 0); best = __shfl_sync(0xffffffffu, best, 0);
}
__global__ void old_masked(int* out, const float* score, int Tmax, int topk, int winmax){
    if(blockIdx.x) return; extern __shared__ float sh[]; const int L=threadIdx.x;
    for(int t=L;t<Tmax;t+=32) sh[t]=score[t];
    __syncwarp();
    for(int k=0;k<topk;++k){ float best=-1e29f; int bi=Tmax;
        for(int t=L;t<Tmax;t+=32) if(sh[t]>best){best=sh[t];bi=t;}
        warp_argmax_(best,bi);
        if(L==0){ if(bi<Tmax){ sh[bi]=-1e30f; out[k]=winmax+bi; } else out[k]=-1; }
        __syncwarp(); } }
__global__ void old_decode(int* out, const float* score, int T, int topk, int offset){
    if(blockIdx.x) return; extern __shared__ float sh[]; const int L=threadIdx.x;
    for(int t=L;t<T;t+=32) sh[t]=score[t];
    __syncwarp();
    for(int k=0;k<topk;++k){ float best=-1e30f; int bi=T;
        for(int t=L;t<T;t+=32) if(sh[t]>best){best=sh[t];bi=t;}
        warp_argmax_(best,bi);
        if(L==0){ if(bi<T) sh[bi]=-1e30f; out[k]=(bi>=T)? -1 : bi+offset; }
        __syncwarp(); } }
__global__ void old_verify(int* dtop, const float* score, int K, int Tf, int topkc, int pos, int ratio, int nwin){
    int i=blockIdx.x; if(i>=K) return; extern __shared__ float sh[]; const int L=threadIdx.x;
    const float* s=score+(size_t)i*Tf;
    for(int t=L;t<Tf;t+=32) sh[t]=s[t];
    __syncwarp();
    const int thr=(pos+i+1)/ratio; const int lim = thr<Tf? thr : Tf;
    for(int k=0;k<topkc;++k){ float best=-1e30f; int bi=Tf;
        for(int t=L;t<lim;t+=32) if(sh[t]>best){best=sh[t];bi=t;}
        warp_argmax_(best,bi);
        if(L==0){ if(bi<Tf){ sh[bi]=-1e30f; dtop[(size_t)i*topkc+k]=nwin+bi; } else dtop[(size_t)i*topkc+k]=-1; }
        __syncwarp(); } }
__global__ void old_offset(int* out, const float* score, int s, int T, int topk, int ratio, int offset){
    int si=blockIdx.x; if(si>=s) return; extern __shared__ float sh[]; const int L=threadIdx.x;
    for(int t=L;t<T;t+=32) sh[t]=score[(size_t)si*T+t];
    __syncwarp();
    int thr=(si+1)/ratio;
    for(int k=0;k<topk;++k){ float best=-1e30f; int bi=T;
        for(int t=L;t<T;t+=32) if(sh[t]>best){best=sh[t];bi=t;}
        warp_argmax_(best,bi);
        if(L==0){ if(bi<T) sh[bi]=-1e30f; out[(size_t)si*topk+k] = (bi>=T || bi>=thr) ? -1 : bi+offset; }
        __syncwarp(); } }

// ---- the replacements, identical bodies to the shipped ones ------------------------------------
__global__ void new_masked(int* out, const float* score, int Tmax, int topk, int winmax, bool early){
    if(blockIdx.x) return; __shared__ TopkRadixSmem S;
    topk_radix_select<TOPK_RADIX_NT>(out, score, Tmax, topk, -1e29f, S, early);
    for(int k=threadIdx.x;k<topk;k+=TOPK_RADIX_NT){ int b=out[k]; out[k]=(b<0)? -1 : winmax+b; } }
__global__ void new_decode(int* out, const float* score, int T, int topk, int offset, bool early){
    if(blockIdx.x) return; __shared__ TopkRadixSmem S;
    topk_radix_select<TOPK_RADIX_NT>(out, score, T, topk, -1e30f, S, early);
    for(int k=threadIdx.x;k<topk;k+=TOPK_RADIX_NT){ int b=out[k]; out[k]=(b<0)? -1 : b+offset; } }
__global__ void new_verify(int* dtop, const float* score, int K, int Tf, int topkc, int pos, int ratio, int nwin, bool early){
    int i=blockIdx.x; if(i>=K) return; __shared__ TopkRadixSmem S;
    const int thr=(pos+i+1)/ratio; const int lim = thr<Tf? thr : Tf;
    int* out = dtop + (size_t)i*topkc;
    topk_radix_select<TOPK_RADIX_NT>(out, score+(size_t)i*Tf, lim, topkc, -1e30f, S, early);
    for(int k=threadIdx.x;k<topkc;k+=TOPK_RADIX_NT){ int b=out[k]; out[k]=(b<0)? -1 : nwin+b; } }
__global__ void new_offset(int* out, const float* score, int s, int T, int topk, int ratio, int offset, bool early){
    int si=blockIdx.x; if(si>=s) return; __shared__ TopkRadixSmem S;
    int* o = out + (size_t)si*topk;
    topk_radix_select<TOPK_RADIX_NT>(o, score+(size_t)si*T, T, topk, -1e30f, S, early);
    const int thr=(si+1)/ratio;
    for(int k=threadIdx.x;k<topk;k+=TOPK_RADIX_NT){ int b=o[k]; o[k]=(b<0 || b>=thr)? -1 : b+offset; } }

static size_t smem(int n){ int m=n+3; m=(m+3)&~3; return (size_t)m*sizeof(float); }

static void fill(std::vector<float>& h, int mode, int T){
    for(int t=0;t<T;++t){
        switch(mode){
        case 0: h[t]=expf(-3.f*t/T)*(1.f+0.1f*sinf(t*0.7f)); break;
        case 1: h[t]=(float)((t*2654435761u)%10007)/10007.f; break;
        case 2: h[t]=(float)((t*7u)%13); break;                                    // heavy exact ties
        case 3: h[t]=((t%3)==0)? -0.0f : (((t%3)==1)? 0.0f : (float)((t*7u)%5)); break;  // signed zeros
        case 4: h[t]=((t%5)==0)? -1e30f : (((t%5)==1)? -1e29f : (((t%5)==2)? -1e31f
                                        : (float)((t*2654435761u)%97)/97.f)); break;     // floor straddle
        case 5: h[t]=-1.f-(float)((t*2654435761u)%10007)/10007.f; break;           // all negative
        }
    }
}
static bool cmp(const char* what, const std::vector<int>& a, const std::vector<int>& b, int T, int mode){
    for(size_t i=0;i<a.size();++i) if(a[i]!=b[i]){
        printf("   MISMATCH %s T=%d mode=%d slot %zu: shipped %d vs radix %d\n",what,T,mode,i,a[i],b[i]);
        return false; }
    return true;
}

static float bench_old(int* o, const float* s, int T, int topk, int iters){
    cudaEvent_t a,b; CU_(cudaEventCreate(&a)); CU_(cudaEventCreate(&b));
    old_masked<<<1,32,smem(T)>>>(o,s,T,topk,0); CU_(cudaDeviceSynchronize());
    CU_(cudaEventRecord(a));
    for(int i=0;i<iters;++i) old_masked<<<1,32,smem(T)>>>(o,s,T,topk,0);
    CU_(cudaEventRecord(b)); CU_(cudaEventSynchronize(b));
    float ms; CU_(cudaEventElapsedTime(&ms,a,b)); return ms*1000.f/iters; }
static float bench_new(int* o, const float* s, int T, int topk, int iters, bool early=true){
    cudaEvent_t a,b; CU_(cudaEventCreate(&a)); CU_(cudaEventCreate(&b));
    new_masked<<<1,TOPK_RADIX_NT>>>(o,s,T,topk,0,early); CU_(cudaDeviceSynchronize());
    CU_(cudaEventRecord(a));
    for(int i=0;i<iters;++i) new_masked<<<1,TOPK_RADIX_NT>>>(o,s,T,topk,0,early);
    CU_(cudaEventRecord(b)); CU_(cudaEventSynchronize(b));
    float ms; CU_(cudaEventElapsedTime(&ms,a,b)); return ms*1000.f/iters; }

// ---- item 1.3 timer. A BAND, NOT A POINT: the whole claim is a few microseconds on a kernel whose
// own run-to-run spread is of that order, so a single event pair would be indistinguishable from
// noise in either direction. `reps` independent event pairs, min/median/max reported.
static float bench_decode(int* o, const float* s, int T, int topk, int iters, bool early){
    cudaEvent_t a,b; CU_(cudaEventCreate(&a)); CU_(cudaEventCreate(&b));
    new_decode<<<1,TOPK_RADIX_NT>>>(o,s,T,topk,0,early); CU_(cudaDeviceSynchronize());
    CU_(cudaEventRecord(a));
    for(int i=0;i<iters;++i) new_decode<<<1,TOPK_RADIX_NT>>>(o,s,T,topk,0,early);
    CU_(cudaEventRecord(b)); CU_(cudaEventSynchronize(b));
    float ms; CU_(cudaEventElapsedTime(&ms,a,b));
    CU_(cudaEventDestroy(a)); CU_(cudaEventDestroy(b));
    return ms*1000.f/iters; }
static void band(int* o, const float* s, int T, int topk, int iters, int reps, bool early,
                 float& lo, float& med, float& hi){
    std::vector<float> v; v.reserve(reps);
    for(int r=0;r<reps;++r) v.push_back(bench_decode(o,s,T,topk,iters,early));
    std::sort(v.begin(), v.end());
    lo=v.front(); med=v[v.size()/2]; hi=v.back(); }

int main(){
    const int TOPK=512, WINMAX=8194;
    // 3072 is ctx 12,288 at ratio 4 -- the point 0.4 attributed 13.47 ms to. 4096 is seqmax 16384.
    int Ts[] = {1, 3, 17, 64, 128, 512, 1024, 1648, 2048, 3072, 4096, 6000, 8192};
    bool all_ok = true;
    printf("== k_topk_masked: correctness on 6 distributions, then timing on mode 0 ==\n");
    printf("%8s %8s %12s %12s %9s   %s\n","T","topk","shipped us","radix us","speedup","all modes");
    printf("%s\n","---------------------------------------------------------------------------");
    for(int T : Ts){
        int topk = TOPK<T? TOPK : T;
        bool ok_all = true; float t_old=0, t_new=0;
        for(int mode=0; mode<6; ++mode){
            std::vector<float> h(T); fill(h,mode,T);
            float* ds; int *o1,*o2;
            CU_(cudaMalloc(&ds,(size_t)T*4)); CU_(cudaMemcpy(ds,h.data(),(size_t)T*4,cudaMemcpyHostToDevice));
            CU_(cudaMalloc(&o1,(size_t)topk*4)); CU_(cudaMalloc(&o2,(size_t)topk*4));
            old_masked<<<1,32,smem(T)>>>(o1,ds,T,topk,WINMAX);
            std::vector<int> a(topk),b(topk);
            CU_(cudaDeviceSynchronize()); CU_(cudaGetLastError());
            CU_(cudaMemcpy(a.data(),o1,(size_t)topk*4,cudaMemcpyDeviceToHost));
            // ITEM 1.3: BOTH arms are compared against the shipped warp scan, at every T and every
            // distribution -- including T <= topk where the early-out fires and T > topk where it
            // must be inert. An arm that is only exercised in the regime it changes proves nothing
            // about the regime it must not change.
            for(int ea=0; ea<2; ++ea){
                new_masked<<<1,TOPK_RADIX_NT>>>(o2,ds,T,topk,WINMAX,ea!=0);
                CU_(cudaDeviceSynchronize()); CU_(cudaGetLastError());
                CU_(cudaMemcpy(b.data(),o2,(size_t)topk*4,cudaMemcpyDeviceToHost));
                ok_all &= cmp(ea? "masked/early" : "masked/full",a,b,T,mode);
            }
            if(mode==0){ int it = T>4096? 20 : 100; t_old=bench_old(o1,ds,T,topk,it); t_new=bench_new(o2,ds,T,topk,it); }
            CU_(cudaFree(ds)); CU_(cudaFree(o1)); CU_(cudaFree(o2));
        }
        all_ok &= ok_all;
        printf("%8d %8d %12.1f %12.1f %8.1fx   %s\n",T,topk,t_old,t_new,t_old/t_new, ok_all?"identical":"DIFFER");
    }

    printf("\n== k_topk_verify (K queries, per-query causal limit) ==\n");
    for(int mode=0; mode<6; ++mode){
        for(int Tf : {1648, 3072}){
            const int K=5, topkc=512, pos=Tf*4-40, ratio=4, nwin=8194;
            std::vector<float> h((size_t)K*Tf), row(Tf);
            for(int i=0;i<K;++i){ fill(row,mode,Tf); for(int t=0;t<Tf;++t) h[(size_t)i*Tf+t]=row[t]*(1.f+0.01f*i); }
            float* ds; int *o1,*o2;
            CU_(cudaMalloc(&ds,h.size()*4)); CU_(cudaMemcpy(ds,h.data(),h.size()*4,cudaMemcpyHostToDevice));
            CU_(cudaMalloc(&o1,(size_t)K*topkc*4)); CU_(cudaMalloc(&o2,(size_t)K*topkc*4));
            old_verify<<<K,32,smem(Tf)>>>(o1,ds,K,Tf,topkc,pos,ratio,nwin);
            std::vector<int> a((size_t)K*topkc),b((size_t)K*topkc);
            CU_(cudaDeviceSynchronize()); CU_(cudaGetLastError());
            CU_(cudaMemcpy(a.data(),o1,a.size()*4,cudaMemcpyDeviceToHost));
            bool ok = true;
            for(int ea=0; ea<2; ++ea){
                new_verify<<<K,TOPK_RADIX_NT>>>(o2,ds,K,Tf,topkc,pos,ratio,nwin,ea!=0);
                CU_(cudaDeviceSynchronize()); CU_(cudaGetLastError());
                CU_(cudaMemcpy(b.data(),o2,b.size()*4,cudaMemcpyDeviceToHost));
                ok &= cmp(ea? "verify/early" : "verify/full",a,b,Tf,mode);
            }
            all_ok &= ok;
            printf("  Tf=%-5d mode=%d K=%d topkc=%d : %s\n",Tf,mode,K,topkc, ok?"identical":"DIFFER");
            CU_(cudaFree(ds));CU_(cudaFree(o1));CU_(cudaFree(o2));
        }
    }
    // A short causal limit is the case where keff < topk and the tail must be -1.
    {
        const int K=4, Tf=2048, topkc=512, pos=800, ratio=4, nwin=77;
        std::vector<float> h((size_t)K*Tf), row(Tf);
        for(int i=0;i<K;++i){ fill(row,1,Tf); for(int t=0;t<Tf;++t) h[(size_t)i*Tf+t]=row[t]; }
        float* ds; int *o1,*o2;
        CU_(cudaMalloc(&ds,h.size()*4)); CU_(cudaMemcpy(ds,h.data(),h.size()*4,cudaMemcpyHostToDevice));
        CU_(cudaMalloc(&o1,(size_t)K*topkc*4)); CU_(cudaMalloc(&o2,(size_t)K*topkc*4));
        old_verify<<<K,32,smem(Tf)>>>(o1,ds,K,Tf,topkc,pos,ratio,nwin);
        std::vector<int> a((size_t)K*topkc),b((size_t)K*topkc);
        CU_(cudaDeviceSynchronize());
        CU_(cudaMemcpy(a.data(),o1,a.size()*4,cudaMemcpyDeviceToHost));
        bool ok = true;
        for(int ea=0; ea<2; ++ea){
            new_verify<<<K,TOPK_RADIX_NT>>>(o2,ds,K,Tf,topkc,pos,ratio,nwin,ea!=0);
            CU_(cudaDeviceSynchronize()); CU_(cudaGetLastError());
            CU_(cudaMemcpy(b.data(),o2,b.size()*4,cudaMemcpyDeviceToHost));
            ok &= cmp(ea? "verify-short/early" : "verify-short/full",a,b,Tf,1);
        }
        all_ok &= ok;
        printf("  short causal limit (lim=%d < topkc=%d)          : %s\n",(pos+1)/ratio,topkc, ok?"identical":"DIFFER");
        CU_(cudaFree(ds));CU_(cudaFree(o1));CU_(cudaFree(o2));
    }

    printf("\n== k_topk_decode / k_topk_offset ==\n");
    for(int mode=0; mode<6; ++mode){
        const int T=1648, topk=512, off=8194;
        std::vector<float> h(T); fill(h,mode,T);
        float* ds; int *o1,*o2;
        CU_(cudaMalloc(&ds,(size_t)T*4)); CU_(cudaMemcpy(ds,h.data(),(size_t)T*4,cudaMemcpyHostToDevice));
        CU_(cudaMalloc(&o1,(size_t)topk*4)); CU_(cudaMalloc(&o2,(size_t)topk*4));
        old_decode<<<1,32,smem(T)>>>(o1,ds,T,topk,off);
        std::vector<int> a(topk),b(topk);
        CU_(cudaDeviceSynchronize()); CU_(cudaGetLastError());
        CU_(cudaMemcpy(a.data(),o1,(size_t)topk*4,cudaMemcpyDeviceToHost));
        bool ok=true;
        for(int ea=0; ea<2; ++ea){
            new_decode<<<1,TOPK_RADIX_NT>>>(o2,ds,T,topk,off,ea!=0);
            CU_(cudaDeviceSynchronize()); CU_(cudaGetLastError());
            CU_(cudaMemcpy(b.data(),o2,(size_t)topk*4,cudaMemcpyDeviceToHost));
            ok &= cmp(ea? "decode/early" : "decode/full",a,b,T,mode);
        }
        all_ok &= ok;
        printf("  k_topk_decode  T=%d mode=%d : %s\n",T,mode, ok?"identical":"DIFFER");
        CU_(cudaFree(ds));CU_(cudaFree(o1));CU_(cudaFree(o2));
    }
    for(int mode=0; mode<6; ++mode){
        const int s=64, ratio=4, T=s/ratio>0? 400:400, topk=(512<T?512:T), off=31;
        std::vector<float> h((size_t)s*T), row(T);
        for(int i=0;i<s;++i){ fill(row,mode,T); for(int t=0;t<T;++t) h[(size_t)i*T+t]=row[t]; }
        // the prefill path masks first; reproduce that so the floored region is real.
        for(int i=0;i<s;++i){ int thr=(i+1)/ratio; for(int t=thr;t<T;++t) h[(size_t)i*T+t]=-1e30f; }
        float* ds; int *o1,*o2;
        CU_(cudaMalloc(&ds,h.size()*4)); CU_(cudaMemcpy(ds,h.data(),h.size()*4,cudaMemcpyHostToDevice));
        CU_(cudaMalloc(&o1,(size_t)s*topk*4)); CU_(cudaMalloc(&o2,(size_t)s*topk*4));
        old_offset<<<s,32,smem(T)>>>(o1,ds,s,T,topk,ratio,off);
        std::vector<int> a((size_t)s*topk),b((size_t)s*topk);
        CU_(cudaDeviceSynchronize()); CU_(cudaGetLastError());
        CU_(cudaMemcpy(a.data(),o1,a.size()*4,cudaMemcpyDeviceToHost));
        bool ok=true;
        for(int ea=0; ea<2; ++ea){
            new_offset<<<s,TOPK_RADIX_NT>>>(o2,ds,s,T,topk,ratio,off,ea!=0);
            CU_(cudaDeviceSynchronize()); CU_(cudaGetLastError());
            CU_(cudaMemcpy(b.data(),o2,b.size()*4,cudaMemcpyDeviceToHost));
            ok &= cmp(ea? "offset/early" : "offset/full",a,b,T,mode);
        }
        all_ok &= ok;
        printf("  k_topk_offset  s=%d T=%d mode=%d : %s\n",s,T,mode, ok?"identical":"DIFFER");
        CU_(cudaFree(ds));CU_(cudaFree(o1));CU_(cudaFree(o2));
    }

    // ================= ITEM 1.3: the early-out, timed as a band ==================================
    // `lim <= topk` is the decode-side condition below ctx 2048 (T = ctx/ratio, ratio 4, topk 512)
    // and the verify-side condition for any query whose causal limit is under 512. Above it the arm
    // must be INERT, which is why 1024/2048/3072 are timed here too and not just asserted.
    printf("\n== item 1.3: full threshold search vs `lim <= topk` early-out (k_topk_decode shape) ==\n");
    printf("%6s %6s %5s   %-22s %-22s %9s\n","T","topk","fires","full us (min/med/max)","early us (min/med/max)","saving us");
    printf("%s\n","------------------------------------------------------------------------------------------");
    {
        const int REPS=9;
        for(int T : {32, 64, 128, 256, 384, 512, 1024, 2048, 3072}){
            const int topk = TOPK<T? TOPK : T;
            std::vector<float> h(T); fill(h,0,T);
            float* ds; int* o;
            CU_(cudaMalloc(&ds,(size_t)T*4)); CU_(cudaMemcpy(ds,h.data(),(size_t)T*4,cudaMemcpyHostToDevice));
            CU_(cudaMalloc(&o,(size_t)topk*4));
            const int it = 200;
            float f_lo,f_md,f_hi,e_lo,e_md,e_hi;
            band(o,ds,T,topk,it,REPS,false,f_lo,f_md,f_hi);
            band(o,ds,T,topk,it,REPS,true, e_lo,e_md,e_hi);
            char fb[64], eb[64];
            snprintf(fb,sizeof fb,"%.2f/%.2f/%.2f",f_lo,f_md,f_hi);
            snprintf(eb,sizeof eb,"%.2f/%.2f/%.2f",e_lo,e_md,e_hi);
            printf("%6d %6d %5s   %-22s %-22s %+9.2f\n", T, topk, (T<=topk)?"yes":"no", fb, eb, f_md-e_md);
            CU_(cudaFree(ds)); CU_(cudaFree(o));
        }
    }

    // ================= ITEM 1.4: ABOVE THE 48 KiB CEILING ========================================
    // The whole point of the item. The four warp kernels stage the score row in DYNAMIC shared
    // memory, so they ask for ~4T bytes against a 49,152 B per-block default -- T 12,288, context
    // 49,152 at ratio 4. Nothing in this repo has ever run the engine that high (`seqmax` is
    // 16,384, T 4,096), which is exactly why the defect survived: it is unreachable from the
    // engine and therefore untested by every gate that goes through the engine. This leg reaches
    // it directly, because a ceiling you have not crossed is a ceiling you have not measured.
    //
    // THREE THINGS ARE PROVED HERE, IN THIS ORDER, AND THE FIRST IS THE ONE THAT MATTERS:
    //   1. WITHOUT the opt-in the launch fails, the following cudaDeviceSynchronize returns
    //      SUCCESS, and the output buffer is returned to the caller UNCHANGED. That is the silent
    //      garbage-return. It is asserted, not narrated: if a future CUDA release starts making
    //      this launch work, or starts making the sync report it, this gate fails and says so.
    //   2. WITH the opt-in (`TOPK_SMEM`, the engine's own macro, from include/indexer.h -- not a
    //      copy) the same launch runs and is BIT-IDENTICAL to the radix select at every T above the
    //      ceiling, on all six distributions.
    //   3. The opt-in survives CUDA-GRAPH CAPTURE. `k_topk_masked` is reached from
    //      `compressed_decode_step_indexer_dp`, which is captured; `cudaFuncSetAttribute` is a host
    //      call and is documented as not stream-ordered, but "documented" is not "measured", and
    //      the failure mode would be a capture that aborts the whole decode path.
    printf("\n== item 1.4: the four warp kernels above the default dynamic-shared-memory ceiling ==\n");
    {
        int dev=0; CU_(cudaGetDevice(&dev));
        int lim_def=0, lim_opt=0;
        CU_(cudaDeviceGetAttribute(&lim_def, cudaDevAttrMaxSharedMemoryPerBlock,      dev));
        CU_(cudaDeviceGetAttribute(&lim_opt, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
        const int T_def = (int)(lim_def/4) - 3, T_opt = (int)(lim_opt/4) - 3;
        printf("  default limit %d B -> T <= %d (context <= %d at ratio 4)\n", lim_def, T_def, T_def*4);
        printf("  opt-in  limit %d B -> T <= %d (context <= %d at ratio 4), %.1fx more headroom\n",
               lim_opt, T_opt, T_opt*4, (double)lim_opt/lim_def);

        // ---- 1. the defect, demonstrated on the shipped kernel BEFORE anything opts it in -------
        const int Tbad = T_def + 4096;                     // comfortably over, still far under opt-in
        const int topkb = TOPK;
        std::vector<float> hb(Tbad); fill(hb,0,Tbad);
        float* dsb; int* ob;
        CU_(cudaMalloc(&dsb,(size_t)Tbad*4)); CU_(cudaMemcpy(dsb,hb.data(),(size_t)Tbad*4,cudaMemcpyHostToDevice));
        CU_(cudaMalloc(&ob,(size_t)topkb*4));
        CU_(cudaMemset(ob,0xA5,(size_t)topkb*4));          // poison: an un-run kernel leaves this intact
        std::vector<int> before(topkb), after(topkb);
        CU_(cudaMemcpy(before.data(),ob,(size_t)topkb*4,cudaMemcpyDeviceToHost));
        cudaGetLastError();
        old_masked<<<1,32,smem(Tbad)>>>(ob,dsb,Tbad,topkb,WINMAX);      // RAW size, no opt-in
        cudaError_t le = cudaGetLastError();
        cudaError_t se = cudaDeviceSynchronize();
        CU_(cudaMemcpy(after.data(),ob,(size_t)topkb*4,cudaMemcpyDeviceToHost));
        bool untouched = (before == after);
        bool defect_ok = (le == cudaErrorInvalidValue) && (se == cudaSuccess) && untouched;
        printf("  no opt-in, T=%d (%zu B): launch=%s  sync=%s  output=%s  -> %s\n",
               Tbad, smem(Tbad), cudaGetErrorName(le), cudaGetErrorName(se),
               untouched? "UNTOUCHED" : "written",
               defect_ok? "defect reproduced (launch fails, sync says success, buffer is stale)"
                        : "UNEXPECTED - the failure mode has changed, re-read item 1.4");
        all_ok &= defect_ok;
        cudaGetLastError();

        // ---- 2. the fix: identical results at every T above the ceiling -------------------------
        // T_opt-64 is one block under the hardware opt-in maximum: the last T that can work at all.
        int Thi[] = { T_def+1, T_def+4096, 2*T_def, T_opt-64 };
        for(int T : Thi){
            int topk = TOPK<T? TOPK : T;
            bool ok_all = true;
            for(int mode=0; mode<6; ++mode){
                std::vector<float> h(T); fill(h,mode,T);
                float* ds; int *o1,*o2;
                CU_(cudaMalloc(&ds,(size_t)T*4)); CU_(cudaMemcpy(ds,h.data(),(size_t)T*4,cudaMemcpyHostToDevice));
                CU_(cudaMalloc(&o1,(size_t)topk*4)); CU_(cudaMalloc(&o2,(size_t)topk*4));
                old_masked<<<1,32,TOPK_SMEM(old_masked,T)>>>(o1,ds,T,topk,WINMAX);
                TOPK_LAUNCHED(old_masked,T);
                CU_(cudaDeviceSynchronize());
                std::vector<int> a(topk),b(topk);
                CU_(cudaMemcpy(a.data(),o1,(size_t)topk*4,cudaMemcpyDeviceToHost));
                for(int ea=0; ea<2; ++ea){
                    new_masked<<<1,TOPK_RADIX_NT>>>(o2,ds,T,topk,WINMAX,ea!=0);
                    CU_(cudaDeviceSynchronize()); CU_(cudaGetLastError());
                    CU_(cudaMemcpy(b.data(),o2,(size_t)topk*4,cudaMemcpyDeviceToHost));
                    ok_all &= cmp(ea? "over-ceiling/early" : "over-ceiling/full",a,b,T,mode);
                }
                CU_(cudaFree(ds)); CU_(cudaFree(o1)); CU_(cudaFree(o2));
            }
            all_ok &= ok_all;
            printf("  opted in,  T=%6d (%7zu B, context %7d): %s\n",
                   T, smem(T), T*4, ok_all? "identical to radix on all 6 distributions" : "DIFFER");
        }

        // ---- 3. the opt-in under CUDA-graph capture ---------------------------------------------
        {
            const int T = T_def + 4096, topk = TOPK;
            std::vector<float> h(T); fill(h,0,T);
            float* ds; int *o1,*o2;
            CU_(cudaMalloc(&ds,(size_t)T*4)); CU_(cudaMemcpy(ds,h.data(),(size_t)T*4,cudaMemcpyHostToDevice));
            CU_(cudaMalloc(&o1,(size_t)topk*4)); CU_(cudaMalloc(&o2,(size_t)topk*4));
            cudaStream_t st; CU_(cudaStreamCreate(&st));
            cudaGraph_t g; cudaGraphExec_t ge;
            CU_(cudaStreamBeginCapture(st, cudaStreamCaptureModeRelaxed));
            old_masked<<<1,32,TOPK_SMEM(old_masked,T),st>>>(o1,ds,T,topk,WINMAX);
            cudaError_t ce = cudaStreamEndCapture(st,&g);
            bool cap_ok = (ce == cudaSuccess);
            if(cap_ok){
                CU_(cudaGraphInstantiate(&ge,g,nullptr,nullptr,0));
                CU_(cudaGraphLaunch(ge,st)); CU_(cudaStreamSynchronize(st));
                new_masked<<<1,TOPK_RADIX_NT,0,st>>>(o2,ds,T,topk,WINMAX,true);
                CU_(cudaStreamSynchronize(st));
                std::vector<int> a(topk),b(topk);
                CU_(cudaMemcpy(a.data(),o1,(size_t)topk*4,cudaMemcpyDeviceToHost));
                CU_(cudaMemcpy(b.data(),o2,(size_t)topk*4,cudaMemcpyDeviceToHost));
                cap_ok = cmp("graph-captured over-ceiling",a,b,T,0);
                CU_(cudaGraphExecDestroy(ge)); CU_(cudaGraphDestroy(g));
            } else printf("   capture FAILED: %s\n", cudaGetErrorName(ce));
            all_ok &= cap_ok;
            printf("  graph capture, T=%d: %s\n", T,
                   cap_ok? "captured, replayed, identical to radix" : "FAILED");
            CU_(cudaStreamDestroy(st)); CU_(cudaFree(ds)); CU_(cudaFree(o1)); CU_(cudaFree(o2));
        }
        // ---- 4. NEGATIVE CONTROL: above the OPT-IN maximum it must ABORT, not launch --------------
        // Untested error handling is decoration. This forks a child, asks the engine's own helper
        // for a T that no device can satisfy, and requires the child to die on SIGABRT -- i.e. that
        // `topk_scan_smem_optin` refuses rather than handing back a size whose launch will fail and
        // leave the buffer stale, which is the exact failure this item exists to remove.
        {
            fflush(stdout); fflush(stderr);
            pid_t pid = fork();
            if(pid == 0){
                if(!freopen("/dev/null","w",stderr)) _exit(120);
                volatile size_t r = topk_scan_smem_optin((const void*)old_masked, T_opt + 1024, "old_masked");
                _exit((int)(r & 0x7f));            // reached only if the helper failed to refuse
            }
            int st = 0; waitpid(pid, &st, 0);
            bool aborted = WIFSIGNALED(st) && WTERMSIG(st) == SIGABRT;
            all_ok &= aborted;
            printf("  above the opt-in max, T=%d: %s\n", T_opt + 1024,
                   aborted ? "child aborted (helper refuses to size an impossible launch)"
                           : "NO ABORT - the helper returned a size that cannot be launched");
        }

        CU_(cudaFree(dsb)); CU_(cudaFree(ob));
    }

    printf("\nGATE: %s\n", all_ok? "PASS — radix select is bit-identical to the shipped warp scan on every shape and distribution,\n        with the item-1.3 early-out both ON and OFF"
                                 : "FAIL — outputs differ, do NOT ship");
    return all_ok?0:1;
}
