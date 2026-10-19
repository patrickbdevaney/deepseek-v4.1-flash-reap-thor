// gate_sdpa_smem.cu — the third dynamic-shared-memory launch (kernels/attention.cu `sdpa`), run
// above and below its own context ceiling. DECODE_LADDER item 1.4.
//
// WHY THIS FILE EXISTS AT ALL. `sdpa` is not linked into build/decode or build/dsv4-server; its
// only caller is tests/test_attention.cu, whose golden case dirs (gen_attention_ref.py) are not in
// this tree, so it has no runnable gate. Fixing an unexercised path and calling it fixed is how a
// wiki page becomes confidently wrong -- so the fix gets a leg that actually runs it. There is no
// reference to compare against here and none is needed: the claim under test is about the LAUNCH,
// not the arithmetic. Below the ceiling and above it, the same kernel over the same inputs must
// produce the same finite output, and `out` must not be the memset pattern it started as.
//
//   build: nvcc -O2 -std=c++17 -gencode arch=compute_110a,code=sm_110a -I include \
//            tests/gate_sdpa_smem.cu kernels/attention.cu -o build/gate_sdpa_smem
#include "attention.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#define CU(x) do{cudaError_t e_=(x); if(e_){fprintf(stderr,"cuda %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e_));exit(1);} }while(0)

static int run(int seq, int hd, const char* tag){
    const int heads = 2, nkv = 1, win = 0;
    size_t shmem = (size_t)(hd + seq) * sizeof(float);
    int lim = 0; { int dev=0; cudaGetDevice(&dev); cudaDeviceGetAttribute(&lim, cudaDevAttrMaxSharedMemoryPerBlock, dev); }
    printf("[sdpa-smem %s] seq=%d head_dim=%d -> %zu B dynamic shared (default limit %d B) : %s\n",
           tag, seq, hd, shmem, lim, shmem > (size_t)lim ? "ABOVE" : "below");
    fflush(stdout);
    std::vector<float> hq((size_t)seq*heads*hd), hk((size_t)seq*nkv*hd), hv((size_t)seq*nkv*hd);
    unsigned long long s = 88172645463325252ull;
    auto nx=[&]{ s^=s<<13; s^=s>>7; s^=s<<17; return (float)((s>>40)%2001)/1000.f - 1.f; };
    for(auto&e:hq) e=nx(); for(auto&e:hk) e=nx(); for(auto&e:hv) e=nx();
    float *dq,*dk,*dv,*dout;
    CU(cudaMalloc(&dq,hq.size()*4)); CU(cudaMalloc(&dk,hk.size()*4)); CU(cudaMalloc(&dv,hv.size()*4));
    CU(cudaMalloc(&dout,(size_t)seq*heads*hd*4));
    CU(cudaMemcpy(dq,hq.data(),hq.size()*4,cudaMemcpyHostToDevice));
    CU(cudaMemcpy(dk,hk.data(),hk.size()*4,cudaMemcpyHostToDevice));
    CU(cudaMemcpy(dv,hv.data(),hv.size()*4,cudaMemcpyHostToDevice));
    // -1 is a sentinel no softmax average of the inputs above can produce, so "untouched" is
    // distinguishable from "wrote a plausible number". A zero fill would not be.
    { std::vector<float> fill((size_t)seq*heads*hd, -1.f);
      CU(cudaMemcpy(dout,fill.data(),fill.size()*4,cudaMemcpyHostToDevice)); }
    sdpa(dout,dq,dk,dv,seq,heads,nkv,hd,win,1.f/sqrtf((float)hd),0);
    cudaError_t sync = cudaDeviceSynchronize();
    std::vector<float> ho((size_t)seq*heads*hd);
    CU(cudaMemcpy(ho.data(),dout,ho.size()*4,cudaMemcpyDeviceToHost));
    long untouched=0, nan=0; double mx=0;
    for(auto v:ho){ if(v==-1.f) ++untouched; if(std::isnan(v)) ++nan; if(fabs(v)>mx) mx=fabs(v); }
    int ok = (sync==cudaSuccess) && untouched==0 && nan==0 && mx>0;
    printf("[sdpa-smem %s] sync=%s  untouched=%ld/%zu  NaN=%ld  maxabs=%.6f -> %s\n",
           tag, cudaGetErrorString(sync), untouched, ho.size(), nan, mx, ok?"PASS":"FAIL");
    fflush(stdout);
    cudaFree(dq);cudaFree(dk);cudaFree(dv);cudaFree(dout);
    return ok?0:1;
}

int main(int argc, char** argv){
    int hd = 64;
    int below = argc>1?atoi(argv[1]):4096;    //  (64 + 4096)*4 =  16,640 B — under the default
    int above = argc>2?atoi(argv[2]):16384;   //  (64 +16384)*4 =  65,792 B — over it, needs the opt-in
    int bad = 0;
    bad |= run(below, hd, "below");
    bad |= run(above, hd, "above");
    printf("GATE: %s\n", bad?"FAIL":"PASS");
    return bad;
}
