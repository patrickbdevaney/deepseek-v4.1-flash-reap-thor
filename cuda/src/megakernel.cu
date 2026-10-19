// megakernel.cu — persistent counter-synced fused kernels (milestone M1).
// The champion per-kernel path stays default; this runs only under MEGAKERNEL=1.
// M1 goal: prove the persistent-grid + global-counter-sync pattern by fusing
// input_rmsnorm + Q/K/V projection into ONE kernel, bit-close to the current path.
#include "megakernel.h"
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <cstdio>
#include <cstring>

#define MCU(x) do{cudaError_t e=(x); if(e){fprintf(stderr,"mega %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));}}while(0)

__device__ __forceinline__ float mbf2f(uint16_t h){ unsigned u=(unsigned)h<<16; float f; memcpy(&f,&u,4); return f; }
__device__ __forceinline__ __half2 mdec_fp4x2(unsigned char b){
    __half2_raw r=__nv_cvt_fp4x2_to_halfraw2((__nv_fp4x2_storage_t)b,__NV_E2M1); return *reinterpret_cast<__half2*>(&r); }
__device__ __forceinline__ float mhw_e4m3(uint8_t b){
    __half_raw r=__nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)b,__NV_E4M3); return __half2float(*reinterpret_cast<__half*>(&r)); }

// warp GEMV dot: y[n] = sum_k dec(Wp[n][k]) * e4m3(Ws[n][k/16]) * wg_inv * x16[k]. Result valid on lane 0.
__device__ __forceinline__ float mega_dot(const uint8_t* Wp,const uint8_t* Ws,float wg_inv,const __half* x16,int n,int K,int lane){
    const unsigned* wpn=(const unsigned*)(Wp+(size_t)n*(K/2)); const uint8_t* wsn=Ws+(size_t)n*(K/16);
    float acc=0.f; int nu=K/8;
    for(int vi=lane; vi<nu; vi+=32){
        unsigned w=__ldcs(&wpn[vi]); int k=vi*8; float sc=mhw_e4m3(__ldcs(&wsn[k>>4]))*wg_inv;
        uint4 xpk=*(const uint4*)(x16+k); const __half2* xh2=(const __half2*)&xpk; const unsigned char* wb=(const unsigned char*)&w;
        __half2 a2=__float2half2_rn(0.f);
        #pragma unroll
        for(int b=0;b<4;++b) a2=__hfma2(mdec_fp4x2(wb[b]),xh2[b],a2);
        acc += sc*(__half2float(__low2half(a2))+__half2float(__high2half(a2)));
    }
    #pragma unroll
    for(int o=16;o>0;o>>=1) acc+=__shfl_down_sync(0xffffffffu,acc,o);
    return acc;
}

// Persistent M1 kernel. Phase A (block 0): RMSNorm h -> hnorm16, signal counter. All blocks spin, then Phase B: GEMV.
__global__ void mega_qkv_kernel(float* q,float* k,float* v,
    const float* h,const uint16_t* g,__half* hnorm16,int* counter,
    const uint8_t* qp,const uint8_t* qs,float qwi,
    const uint8_t* kp,const uint8_t* ks,float kwi,
    const uint8_t* vp,const uint8_t* vs,float vwi,
    int H,int qd,int kd,float eps){
    int tid=threadIdx.x, lane=tid&31, warp=tid>>5, wpb=blockDim.x>>5;
    if(blockIdx.x==0){                                   // Phase A: produce hnorm16 (fp16), FlashNorm-style not yet (kept simple/correct)
        __shared__ float ss[256]; float loc=0.f;
        for(int i=tid;i<H;i+=blockDim.x){ float x=h[i]; loc+=x*x; }
        ss[tid]=loc; __syncthreads();
        for(int s=blockDim.x/2;s>0;s>>=1){ if(tid<s)ss[tid]+=ss[tid+s]; __syncthreads(); }
        float rinv=rsqrtf(ss[0]/H+eps);
        for(int i=tid;i<H;i+=blockDim.x) hnorm16[i]=__float2half(h[i]*rinv*mbf2f(g[i]));
        __threadfence();
        if(tid==0) atomicExch(counter,1);
    }
    if(tid==0){ while(*(volatile int*)counter==0){} }    // sentinel spin (all blocks resident -> no deadlock)
    __syncthreads(); __threadfence();
    int nv_out = vp? kd : 0;                              // full layer: vp null, v=k copied by host
    long total = (long)qd + kd + nv_out;
    for(long o=(long)blockIdx.x*wpb+warp; o<total; o+=(long)gridDim.x*wpb){
        if(o<qd){ float r=mega_dot(qp,qs,qwi,hnorm16,(int)o,H,lane); if(lane==0)q[o]=r; }
        else if(o<qd+kd){ int n=o-qd; float r=mega_dot(kp,ks,kwi,hnorm16,n,H,lane); if(lane==0)k[n]=r; }
        else { int n=o-qd-kd; float r=mega_dot(vp,vs,vwi,hnorm16,n,H,lane); if(lane==0)v[n]=r; }
    }
}

// ================= FULL-STEP architecture (grid-cooperative, single persistent kernel) =================
// grid-wide barrier: all NB resident blocks must reach bar[phase] before any proceeds. NB<=residency => no deadlock.
__device__ __forceinline__ void grid_bar(int* barp,int NB){
    __syncthreads(); __threadfence();
    if(threadIdx.x==0) atomicAdd(barp,1);
    if(threadIdx.x==0){ while(*(volatile int*)barp < NB){} }
    __syncthreads();
}
// Stage-1 instruction block: grid-cooperative input_rmsnorm + Q/K/V proj (foundation for the full-step layer loop).
// Phase0: grid-reduce sum(h^2). bar. Phase1: hnorm16 slices. bar. Phase2: QKV GEMV.
__global__ void mega_af_kernel(float* q,float* k,float* v,const float* h,const uint16_t* g,
    __half* hnorm16,float* sumsq,int* bar,
    const uint8_t* qp,const uint8_t* qs,float qwi,
    const uint8_t* kp,const uint8_t* ks,float kwi,
    const uint8_t* vp,const uint8_t* vs,float vwi,
    int H,int qd,int kd,float eps,int NB){
    int tid=threadIdx.x, lane=tid&31, warp=tid>>5, wpb=blockDim.x>>5;
    long gtid=(long)blockIdx.x*blockDim.x+tid, gstride=(long)gridDim.x*blockDim.x;
    __shared__ float ss[256]; float loc=0.f;                       // Phase 0: partial sum(h^2)
    for(long i=gtid;i<H;i+=gstride){ float x=h[i]; loc+=x*x; }
    ss[tid]=loc; __syncthreads();
    for(int s=128;s>0;s>>=1){ if(tid<s)ss[tid]+=ss[tid+s]; __syncthreads(); }
    if(tid==0) atomicAdd(sumsq,ss[0]);
    grid_bar(&bar[0],NB);
    float rinv=rsqrtf(*sumsq/H+eps);                               // Phase 1: hnorm16 (all blocks, own slice)
    for(long i=gtid;i<H;i+=gstride) hnorm16[i]=__float2half(h[i]*rinv*mbf2f(g[i]));
    grid_bar(&bar[1],NB);
    int nv_out=vp?kd:0; long total=(long)qd+kd+nv_out;             // Phase 2: QKV GEMV
    for(long o=(long)blockIdx.x*wpb+warp; o<total; o+=(long)gridDim.x*wpb){
        if(o<qd){ float r=mega_dot(qp,qs,qwi,hnorm16,(int)o,H,lane); if(lane==0)q[o]=r; }
        else if(o<qd+kd){ int n=o-qd; float r=mega_dot(kp,ks,kwi,hnorm16,n,H,lane); if(lane==0)k[n]=r; }
        else { int n=o-qd-kd; float r=mega_dot(vp,vs,vwi,hnorm16,n,H,lane); if(lane==0)v[n]=r; }
    }
}
extern "C" void mega_attn_front(float* q,float* k,float* v,const float* h,const uint16_t* g,
    const uint8_t* qp,const uint8_t* qs,float qwg, const uint8_t* kp,const uint8_t* ks,float kwg,
    const uint8_t* vp,const uint8_t* vs,float vwg, int H,int qd,int kd,float eps,cudaStream_t s){
    static __half* hnorm16=nullptr; static float* sumsq=nullptr; static int* bar=nullptr; static int NB=0;
    if(!hnorm16){ MCU(cudaMalloc(&hnorm16,(size_t)16384*sizeof(__half))); MCU(cudaMalloc(&sumsq,4)); MCU(cudaMalloc(&bar,2*sizeof(int)));
        int maxb=0; cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxb,mega_af_kernel,256,0);
        cudaDeviceProp p; cudaGetDeviceProperties(&p,0); NB=maxb*p.multiProcessorCount; if(NB<1)NB=20; }
    MCU(cudaMemsetAsync(sumsq,0,4,s)); MCU(cudaMemsetAsync(bar,0,2*sizeof(int),s));
    mega_af_kernel<<<NB,256,0,s>>>(q,k,v,h,g,hnorm16,sumsq,bar,
        qp,qs,1.f/qwg, kp,ks,1.f/kwg, vp,vp?vs:nullptr,vp?1.f/vwg:0.f, H,qd,kd,eps,NB);
}

// M2: fused o_proj + post_attn_norm + residual. GRID-BARRIER class (produce all op, then reduce+add).
// Launched with NB = resident-capacity blocks so the arrival barrier can't deadlock.
__global__ void mega_oproj_kernel(float* hres,const float* ao,const uint16_t* g,
    __half* ao16,float* op,int* cnt,const uint8_t* op_p,const uint8_t* op_s,float owi,int H,int qd,float eps,int NB){
    int tid=threadIdx.x, lane=tid&31, warp=tid>>5, wpb=blockDim.x>>5;
    if(blockIdx.x==0){                                   // Phase 0: convert ao -> ao16 (producer)
        for(int i=tid;i<qd;i+=blockDim.x) ao16[i]=__float2half(ao[i]);
        __threadfence(); if(tid==0) atomicExch(&cnt[0],1);
    }
    if(tid==0){ while(*(volatile int*)&cnt[0]==0){} } __syncthreads(); __threadfence();
    for(long n=(long)blockIdx.x*wpb+warp; n<H; n+=(long)gridDim.x*wpb){    // Phase A: op = Wo·ao16 (all blocks)
        float r=mega_dot(op_p,op_s,owi,ao16,(int)n,qd,lane); if(lane==0) op[n]=r;
    }
    __syncthreads(); __threadfence();
    if(tid==0) atomicAdd(&cnt[1],1);                     // arrival barrier
    if(blockIdx.x==0){                                   // Phase B: block0 reduces op -> rinv, then residual add
        if(tid==0){ while(*(volatile int*)&cnt[1]<NB){} } __syncthreads(); __threadfence();
        __shared__ float ss[256]; float loc=0.f;
        for(int i=tid;i<H;i+=blockDim.x){ float x=op[i]; loc+=x*x; }
        ss[tid]=loc; __syncthreads();
        for(int s=blockDim.x/2;s>0;s>>=1){ if(tid<s)ss[tid]+=ss[tid+s]; __syncthreads(); }
        float rinv=rsqrtf(ss[0]/H+eps);
        for(int i=tid;i<H;i+=blockDim.x) hres[i]+=op[i]*rinv*mbf2f(g[i]);
    }
}
extern "C" void mega_oproj(float* hres,const float* ao,const uint16_t* g,
    const uint8_t* op_p,const uint8_t* op_s,float owg,int H,int qd,float eps,cudaStream_t s){
    static __half* ao16=nullptr; static float* op=nullptr; static int* cnt=nullptr; static int NB=0;
    if(!ao16){ MCU(cudaMalloc(&ao16,(size_t)16384*sizeof(__half))); MCU(cudaMalloc(&op,(size_t)16384*4)); MCU(cudaMalloc(&cnt,2*sizeof(int)));
        int maxb=0; cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxb,mega_oproj_kernel,256,0);
        cudaDeviceProp p; cudaGetDeviceProperties(&p,0); NB=maxb*p.multiProcessorCount; if(NB<1)NB=20; }
    MCU(cudaMemsetAsync(cnt,0,2*sizeof(int),s));
    mega_oproj_kernel<<<NB,256,0,s>>>(hres,ao,g,ao16,op,cnt,op_p,op_s,1.f/owg,H,qd,eps,NB);
}

extern "C" void mega_qkv(float* q,float* k,float* v,const float* h,const uint16_t* g,
    const uint8_t* qp,const uint8_t* qs,float qwg,
    const uint8_t* kp,const uint8_t* ks,float kwg,
    const uint8_t* vp,const uint8_t* vs,float vwg,
    int H,int qd,int kd,float eps,cudaStream_t s){
    static __half* hnorm16=nullptr; static int* counter=nullptr;
    if(!hnorm16){ MCU(cudaMalloc(&hnorm16,(size_t)16384*sizeof(__half))); MCU(cudaMalloc(&counter,sizeof(int))); }
    MCU(cudaMemsetAsync(counter,0,sizeof(int),s));
    int total_out = qd + kd + (vp? kd:0);
    int nblocks = (total_out + 7)/8;   // 8 warps/block, ~1 output/warp -> match champion's per-GEMV concurrency (latency hiding)
    mega_qkv_kernel<<<nblocks,256,0,s>>>(q,k,v,h,g,hnorm16,counter,
        qp,qs,1.f/qwg, kp,ks,1.f/kwg, vp,vp?vs:nullptr,vp?1.f/vwg:0.f, H,qd,kd,eps);
    if(!vp) MCU(cudaMemcpyAsync(v,k,(size_t)kd*sizeof(float),cudaMemcpyDeviceToDevice,s));  // full layer: v=k
}
