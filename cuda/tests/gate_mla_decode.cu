// gate_mla_decode.cu — EQUIVALENCE gate for the M=1 sliding-window decode step (Step 4, milestone 1).
// Compares two code paths on IDENTICAL (synthetic) weights: (A) mla_forward prefill over s tokens ->
// out_pref[s]; (B) mla_cache_kv over the first s-1 tokens + mla_decode_step for token s-1 -> out_dec[1].
// Per-row math is identical, so out_dec must reproduce out_pref[s-1]. No golden needed (path-vs-path).
//   build: nvcc -O2 -std=c++17 -arch=sm_110a -I include tests/gate_mla_decode.cu kernels/mla_forward.cu \
//          kernels/mla_decode.cu kernels/fp8_block_gemm.cu kernels/mla_attn.cu kernels/tc_fp8_gemm.cu -o build/gate_mla_decode
#include "mla_forward.h"
#include "mla_decode.h"
#include "deepseek_v4.h"
#include <cuda_runtime.h>
#include <vector>
#include <cstdio>
#include <cmath>
#include <cstdlib>
using namespace dsv4;
#define CU(x) do{cudaError_t e=(x); if(e){fprintf(stderr,"cuda %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)

static uint8_t rfp8(){ return (uint8_t)((rand()%0x40) | ((rand()&1)<<7)); }   // finite e4m3, both signs (no 0x7F/0xFF NaN)
static const uint8_t* upW(int n){ std::vector<uint8_t> h(n); for(auto&v:h)v=rfp8();
    uint8_t* d; CU(cudaMalloc(&d,n)); CU(cudaMemcpy(d,h.data(),n,cudaMemcpyHostToDevice)); return d; }
static const float* upS(int n){ std::vector<float> h(n); for(auto&v:h)v=0.3f+0.01f*(rand()%40);
    float* d; CU(cudaMalloc(&d,n*4)); CU(cudaMemcpy(d,h.data(),n*4,cudaMemcpyHostToDevice)); return d; }
static const float* upF(std::vector<float>& h){ float* d; CU(cudaMalloc(&d,h.size()*4));
    CU(cudaMemcpy(d,h.data(),h.size()*4,cudaMemcpyHostToDevice)); return d; }

int main(int argc, char** argv){
    const int s = argc>1?atoi(argv[1]):16, half=ROPE_DIM/2, Kd=N_HEADS*HEAD_DIM, GKd=Kd/O_GROUPS, OB=O_GROUPS*O_LORA;
    srand(7);
    MLAWeights w{};
    w.wq_a=upW(Q_LORA*DIM);   w.wq_a_s=upS((Q_LORA/128)*(DIM/128));
    w.wq_b=upW(Kd*Q_LORA);    w.wq_b_s=upS((Kd/128)*(Q_LORA/128));
    w.wkv =upW(HEAD_DIM*DIM); w.wkv_s =upS((HEAD_DIM/128)*(DIM/128));
    w.wo_b=upW(DIM*OB);       w.wo_b_s=upS((DIM/128)*(OB/128));
    { std::vector<float> qn(Q_LORA); for(auto&v:qn)v=0.5f+0.01f*(rand()%100); w.q_norm=upF(qn); }
    { std::vector<float> kn(HEAD_DIM); for(auto&v:kn)v=0.5f+0.01f*(rand()%100); w.kv_norm=upF(kn); }
    { std::vector<float> wa((size_t)O_GROUPS*O_LORA*GKd); for(auto&v:wa)v=0.02f*((rand()%200)-100)/100.f; w.wo_a=upF(wa); }
    { std::vector<float> sk(N_HEADS); for(auto&v:sk)v=0.01f*(rand()%100); w.attn_sink=upF(sk); }
    std::vector<float> cosh((size_t)s*half), sinh((size_t)s*half);
    for(int p=0;p<s;++p) for(int j=0;j<half;++j){ float a=p*0.017f*(j+1); cosh[p*half+j]=cosf(a); sinh[p*half+j]=sinf(a); }
    w.cosT=upF(cosh); w.sinT=upF(sinh);
    std::vector<float> xh((size_t)s*DIM); for(auto&v:xh)v=0.1f*((rand()%200)-100)/100.f;
    const float* x=upF(xh);

    // (A) prefill
    float* out_pref; CU(cudaMalloc(&out_pref,(size_t)s*DIM*4));
    mla_forward(out_pref, x, w, 1, s); CU(cudaDeviceSynchronize());

    // (B) cache first s-1 tokens, decode token s-1
    float* kvcache; CU(cudaMalloc(&kvcache,(size_t)s*HEAD_DIM*4));
    mla_cache_kv(kvcache, x, w, s-1);
    float* out_dec; CU(cudaMalloc(&out_dec,(size_t)DIM*4));
    mla_decode_step(out_dec, x + (size_t)(s-1)*DIM, w, kvcache, s-1);
    CU(cudaDeviceSynchronize());

    std::vector<float> op((size_t)DIM), od((size_t)DIM);
    CU(cudaMemcpy(op.data(), out_pref+(size_t)(s-1)*DIM, DIM*4, cudaMemcpyDeviceToHost));
    CU(cudaMemcpy(od.data(), out_dec, DIM*4, cudaMemcpyDeviceToHost));
    double dot=0,np=0,nd=0,sd=0,sr=0,mx=0,ma=0;
    for(int i=0;i<DIM;++i){ double a=op[i],b=od[i]; dot+=a*b; np+=a*a; nd+=b*b; sd+=(a-b)*(a-b); sr+=a*a; mx=fmax(mx,fabs(a)); ma=fmax(ma,fabs(a-b)); }
    double cosine=dot/(sqrt(np)*sqrt(nd)+1e-30), rms=sqrt(sd/(sr+1e-30)), absr=ma/(mx+1e-30);
    bool ok = cosine>0.99999 && rms<1e-3;
    printf("[mla_decode equiv] s=%d pos=%d dim=%d\n", s, s-1, DIM);
    printf("[mla_decode equiv] cosine=%.8f rms_rel=%.2e max_abs/|o|max=%.2e -> %s\n", cosine, rms, absr, ok?"PASS":"FAIL");
    return ok?0:1;
}
