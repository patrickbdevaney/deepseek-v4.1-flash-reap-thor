// gate_topk_warp.cu — prove the warp-parallel top-k is BIT-IDENTICAL to the shipped one, then time both.
//
// The shipped kernels (`k_topk_decode`, `k_topk_verify`, `k_topk_masked`) open with
// `if(threadIdx.x||blockIdx.x) return;` and then run an O(topk x T) selection sort on ONE thread of
// the 32 that were launched, on 1 of 20 SMs. At 24k context that is 512 x 6000 serial comparisons
// per ratio-4 layer, 21 layers, every target forward -- and it is the whole of the context-linear
// term in `tools/decode_model.py`'s fit.
//
// THE MINIMUM-RISK FIX IS NOT A NEW ALGORITHM. It is to use the 32 threads that are already there:
// each lane strides the row, then one warp argmax reduction per output slot. Same selection sort,
// same order, same ties -- 1/32 of the serial work. A radix select is worth far more and comes
// next, but this needs no shared-memory redesign and no new failure modes, and it is the change
// that can be proven identical by construction rather than by argument.
//
// TIE-BREAKING IS THE WHOLE CORRECTNESS ARGUMENT. The originals scan ascending with a STRICT `>`,
// so on equal scores the LOWEST index wins. The reduction must reproduce that exactly or the
// selected set is the same but its order is not -- and `sparse_attn` sums the selected rows in
// order, so a reordering changes fp32 association and breaks the lossless gate. The sentinel is
// therefore `T` (one past any valid index) rather than -1, so that "nothing found" loses every
// comparison and lower indices win ties for free.
//
//   nvcc -O2 -gencode arch=compute_110a,code=sm_110a -I include tests/gate_topk_warp.cu -o build/gate_topk_warp
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

#define CU_(x) do{ cudaError_t e=(x); if(e){ printf("CUDA %s @%d\n", cudaGetErrorString(e), __LINE__); exit(1);} }while(0)

// ------------------------- shipped, verbatim -------------------------
__global__ void old_topk_masked(int* out, const float* score, int Tmax, int topk, int winmax){
    if(threadIdx.x||blockIdx.x) return; extern __shared__ float sh[]; for(int t=0;t<Tmax;++t) sh[t]=score[t];
    for(int k=0;k<topk;++k){ float best=-1e29f; int bi=-1; for(int t=0;t<Tmax;++t) if(sh[t]>best){best=sh[t];bi=t;}
        if(bi>=0){ sh[bi]=-1e30f; out[k]=winmax+bi; } else out[k]=-1; } }

__global__ void old_topk_decode(int* out, const float* score, int T, int topk, int offset){
    if(threadIdx.x||blockIdx.x) return;
    extern __shared__ float sh[];
    for(int t=0;t<T;++t) sh[t]=score[t];
    for(int k=0;k<topk;++k){ float best=-1e30f; int bi=-1;
        for(int t=0;t<T;++t) if(sh[t]>best){best=sh[t];bi=t;}
        if(bi>=0) sh[bi]=-1e30f;
        out[k] = (bi<0)? -1 : bi+offset; }
}

__global__ void old_topk_verify(int* dtop, const float* score, int K, int Tf, int topkc, int pos, int ratio, int nwin){
    int i=blockIdx.x; if(i>=K||threadIdx.x) return; extern __shared__ float sh[]; const float* s=score+(size_t)i*Tf;
    for(int t=0;t<Tf;++t) sh[t]=s[t]; int thr=(pos+i+1)/ratio;
    for(int k=0;k<topkc;++k){ float best=-1e30f;int bi=-1; for(int t=0;t<thr&&t<Tf;++t) if(sh[t]>best){best=sh[t];bi=t;}
        if(bi>=0){ sh[bi]=-1e30f; dtop[(size_t)i*topkc+k]=nwin+bi; } else dtop[(size_t)i*topkc+k]=-1; } }

// ------------------------- warp-parallel -------------------------
// One argmax reduction over 32 lanes. `bi` carries T as its "nothing here" sentinel so that a lane
// which found nothing loses to every real index, and equal scores resolve to the lower index --
// which is exactly what the serial ascending scan with a strict `>` does.
__device__ __forceinline__ void warp_argmax(float& best, int& bi){
    #pragma unroll
    for(int o=16;o>0;o>>=1){
        float ob = __shfl_down_sync(0xffffffffu, best, o);
        int   oi = __shfl_down_sync(0xffffffffu, bi,   o);
        if(ob > best || (ob == best && oi < bi)){ best = ob; bi = oi; }
    }
    bi   = __shfl_sync(0xffffffffu, bi,   0);
    best = __shfl_sync(0xffffffffu, best, 0);
}

__global__ void new_topk_masked(int* out, const float* score, int Tmax, int topk, int winmax){
    if(blockIdx.x) return; extern __shared__ float sh[]; const int L=threadIdx.x;
    for(int t=L;t<Tmax;t+=32) sh[t]=score[t];
    __syncwarp();
    for(int k=0;k<topk;++k){
        float best=-1e29f; int bi=Tmax;
        for(int t=L;t<Tmax;t+=32) if(sh[t]>best){best=sh[t];bi=t;}
        warp_argmax(best,bi);
        if(L==0){ if(bi<Tmax){ sh[bi]=-1e30f; out[k]=winmax+bi; } else out[k]=-1; }
        __syncwarp();
    }
}

__global__ void new_topk_decode(int* out, const float* score, int T, int topk, int offset){
    if(blockIdx.x) return; extern __shared__ float sh[]; const int L=threadIdx.x;
    for(int t=L;t<T;t+=32) sh[t]=score[t];
    __syncwarp();
    for(int k=0;k<topk;++k){
        float best=-1e30f; int bi=T;
        for(int t=L;t<T;t+=32) if(sh[t]>best){best=sh[t];bi=t;}
        warp_argmax(best,bi);
        if(L==0){ if(bi<T) sh[bi]=-1e30f; out[k]=(bi>=T)? -1 : bi+offset; }
        __syncwarp();
    }
}

__global__ void new_topk_verify(int* dtop, const float* score, int K, int Tf, int topkc, int pos, int ratio, int nwin){
    int i=blockIdx.x; if(i>=K) return; extern __shared__ float sh[]; const int L=threadIdx.x;
    const float* s=score+(size_t)i*Tf;
    for(int t=L;t<Tf;t+=32) sh[t]=s[t];
    __syncwarp();
    const int thr=(pos+i+1)/ratio; const int lim = thr<Tf? thr : Tf;
    for(int k=0;k<topkc;++k){
        float best=-1e30f; int bi=Tf;
        for(int t=L;t<lim;t+=32) if(sh[t]>best){best=sh[t];bi=t;}
        warp_argmax(best,bi);
        if(L==0){ if(bi<Tf){ sh[bi]=-1e30f; dtop[(size_t)i*topkc+k]=nwin+bi; } else dtop[(size_t)i*topkc+k]=-1; }
        __syncwarp();
    }
}

static size_t smem(int n){ int m=n+3; m=(m+3)&~3; return (size_t)m*sizeof(float); }

static float bench(void(*launch)(int*,const float*,int,int,int,cudaStream_t), int* o, const float* s,
                   int T, int topk, int off, int iters){
    cudaEvent_t a,b; CU_(cudaEventCreate(&a)); CU_(cudaEventCreate(&b));
    launch(o,s,T,topk,off,0); CU_(cudaDeviceSynchronize());
    CU_(cudaEventRecord(a));
    for(int i=0;i<iters;++i) launch(o,s,T,topk,off,0);
    CU_(cudaEventRecord(b)); CU_(cudaEventSynchronize(b));
    float ms; CU_(cudaEventElapsedTime(&ms,a,b)); return ms*1000.f/iters;   // us
}
static void L_old(int* o,const float* s,int T,int k,int off,cudaStream_t st){ old_topk_masked<<<1,32,smem(T),st>>>(o,s,T,k,off); }
static void L_new(int* o,const float* s,int T,int k,int off,cudaStream_t st){ new_topk_masked<<<1,32,smem(T),st>>>(o,s,T,k,off); }

int main(){
    const int TOPK=512, WINMAX=8194;
    int Ts[] = {64, 128, 512, 1024, 1648, 2048, 4096, 6000, 8192};
    printf("%8s %8s %12s %12s %9s   %s\n","T","topk","shipped us","warp us","speedup","indices");
    printf("%s\n","------------------------------------------------------------------------------");
    bool all_ok = true;
    for(int T : Ts){
        int topk = TOPK<T? TOPK : T;
        std::vector<float> h(T);
        // Three distributions: exponential decay (realistic), uniform, and one with MANY EXACT TIES
        // -- ties are the only place the two implementations could diverge, so they get their own run.
        for(int mode=0; mode<3; ++mode){
            for(int t=0;t<T;++t){
                if(mode==0) h[t] = expf(-3.f*t/T)*(1.f + 0.1f*sinf(t*0.7f));
                else if(mode==1) h[t] = (float)((t*2654435761u)%10007)/10007.f;
                else h[t] = (float)((t*7u)%13);                 // heavy exact ties
            }
            float *ds; int *o1,*o2;
            CU_(cudaMalloc(&ds,(size_t)T*4)); CU_(cudaMemcpy(ds,h.data(),(size_t)T*4,cudaMemcpyHostToDevice));
            CU_(cudaMalloc(&o1,(size_t)topk*4)); CU_(cudaMalloc(&o2,(size_t)topk*4));
            old_topk_masked<<<1,32,smem(T)>>>(o1,ds,T,topk,WINMAX);
            new_topk_masked<<<1,32,smem(T)>>>(o2,ds,T,topk,WINMAX);
            CU_(cudaDeviceSynchronize());
            std::vector<int> a(topk),b(topk);
            CU_(cudaMemcpy(a.data(),o1,(size_t)topk*4,cudaMemcpyDeviceToHost));
            CU_(cudaMemcpy(b.data(),o2,(size_t)topk*4,cudaMemcpyDeviceToHost));
            bool ok=true; for(int i=0;i<topk;++i) if(a[i]!=b[i]){ ok=false;
                if(mode==2||T<=1024) printf("   MISMATCH T=%d mode=%d slot %d: shipped %d vs warp %d\n",T,mode,i,a[i],b[i]);
                break; }
            all_ok &= ok;
            if(mode==0){
                int it = T>4096? 20 : 100;
                float t_old = bench(L_old,o1,ds,T,topk,WINMAX,it);
                float t_new = bench(L_new,o2,ds,T,topk,WINMAX,it);
                printf("%8d %8d %12.1f %12.1f %8.1fx   %s\n",T,topk,t_old,t_new,t_old/t_new, ok?"identical":"DIFFER");
            } else if(!ok) printf("        (mode %d: DIFFER)\n",mode);
            CU_(cudaFree(ds)); CU_(cudaFree(o1)); CU_(cudaFree(o2));
        }
    }
    // The other two kernel shapes, correctness only.
    {
        const int K=5, Tf=1648, topkc=512, pos=6000, ratio=4, nwin=8194;
        std::vector<float> h((size_t)K*Tf);
        for(size_t i=0;i<h.size();++i) h[i]=(float)((i*2654435761u)%9973)/9973.f;
        float* ds; int *o1,*o2;
        CU_(cudaMalloc(&ds,h.size()*4)); CU_(cudaMemcpy(ds,h.data(),h.size()*4,cudaMemcpyHostToDevice));
        CU_(cudaMalloc(&o1,(size_t)K*topkc*4)); CU_(cudaMalloc(&o2,(size_t)K*topkc*4));
        old_topk_verify<<<K,32,smem(Tf)>>>(o1,ds,K,Tf,topkc,pos,ratio,nwin);
        new_topk_verify<<<K,32,smem(Tf)>>>(o2,ds,K,Tf,topkc,pos,ratio,nwin);
        CU_(cudaDeviceSynchronize());
        std::vector<int> a((size_t)K*topkc),b((size_t)K*topkc);
        CU_(cudaMemcpy(a.data(),o1,a.size()*4,cudaMemcpyDeviceToHost));
        CU_(cudaMemcpy(b.data(),o2,b.size()*4,cudaMemcpyDeviceToHost));
        bool ok = a==b; all_ok &= ok;
        printf("\nk_topk_verify  K=%d Tf=%d topkc=%d : %s\n",K,Tf,topkc, ok?"identical":"DIFFER");
        CU_(cudaFree(ds));CU_(cudaFree(o1));CU_(cudaFree(o2));
    }
    {
        const int T=1648, topk=512, off=8194;
        std::vector<float> h(T); for(int t=0;t<T;++t) h[t]=(float)((t*40503u)%7919)/7919.f;
        float* ds; int *o1,*o2;
        CU_(cudaMalloc(&ds,(size_t)T*4)); CU_(cudaMemcpy(ds,h.data(),(size_t)T*4,cudaMemcpyHostToDevice));
        CU_(cudaMalloc(&o1,(size_t)topk*4)); CU_(cudaMalloc(&o2,(size_t)topk*4));
        old_topk_decode<<<1,32,smem(T)>>>(o1,ds,T,topk,off);
        new_topk_decode<<<1,32,smem(T)>>>(o2,ds,T,topk,off);
        CU_(cudaDeviceSynchronize());
        std::vector<int> a(topk),b(topk);
        CU_(cudaMemcpy(a.data(),o1,(size_t)topk*4,cudaMemcpyDeviceToHost));
        CU_(cudaMemcpy(b.data(),o2,(size_t)topk*4,cudaMemcpyDeviceToHost));
        bool ok=a==b; all_ok &= ok;
        printf("k_topk_decode  T=%d topk=%d          : %s\n",T,topk, ok?"identical":"DIFFER");
        CU_(cudaFree(ds));CU_(cudaFree(o1));CU_(cudaFree(o2));
    }
    printf("\nGATE: %s\n", all_ok? "PASS — warp-parallel output is bit-identical on every shape and distribution"
                                 : "FAIL — outputs differ, do NOT ship");
    return all_ok?0:1;
}
