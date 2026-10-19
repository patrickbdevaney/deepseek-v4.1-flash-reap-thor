// gate_mla_verify.cu — M=K verify step ≡ K sequential M=1 decode steps (spec-decode verify primitive).
// Same synthetic weights; both start from a PS-token cache, then process the next K tokens. out must match.
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
static uint8_t rfp8(){ return (uint8_t)((rand()%0x40)|((rand()&1)<<7)); }
static const uint8_t* upW(int n){ std::vector<uint8_t> h(n); for(auto&v:h)v=rfp8(); uint8_t*d; CU(cudaMalloc(&d,n)); CU(cudaMemcpy(d,h.data(),n,cudaMemcpyHostToDevice)); return d; }
static const float* upS(int n){ std::vector<float> h(n); for(auto&v:h)v=0.3f+0.01f*(rand()%40); float*d; CU(cudaMalloc(&d,n*4)); CU(cudaMemcpy(d,h.data(),n*4,cudaMemcpyHostToDevice)); return d; }
static const float* upF(std::vector<float>&h){ float*d; CU(cudaMalloc(&d,h.size()*4)); CU(cudaMemcpy(d,h.data(),h.size()*4,cudaMemcpyHostToDevice)); return d; }
int main(int argc,char**argv){
    const int s=12, K=4, PS=s-K, half=ROPE_DIM/2, Kd=N_HEADS*HEAD_DIM, GKd=Kd/O_GROUPS, OB=O_GROUPS*O_LORA; srand(7);
    MLAWeights w{};
    w.wq_a=upW(Q_LORA*DIM);w.wq_a_s=upS((Q_LORA/128)*(DIM/128));
    w.wq_b=upW(Kd*Q_LORA);w.wq_b_s=upS((Kd/128)*(Q_LORA/128));
    w.wkv=upW(HEAD_DIM*DIM);w.wkv_s=upS((HEAD_DIM/128)*(DIM/128));
    w.wo_b=upW(DIM*OB);w.wo_b_s=upS((DIM/128)*(OB/128));
    {std::vector<float>v(Q_LORA);for(auto&e:v)e=0.5f+0.01f*(rand()%100);w.q_norm=upF(v);}
    {std::vector<float>v(HEAD_DIM);for(auto&e:v)e=0.5f+0.01f*(rand()%100);w.kv_norm=upF(v);}
    {std::vector<float>v((size_t)O_GROUPS*O_LORA*GKd);for(auto&e:v)e=0.02f*((rand()%200)-100)/100.f;w.wo_a=upF(v);}
    {std::vector<float>v(N_HEADS);for(auto&e:v)e=0.01f*(rand()%100);w.attn_sink=upF(v);}
    std::vector<float> cc((size_t)s*half),ss((size_t)s*half);
    for(int p=0;p<s;++p)for(int j=0;j<half;++j){float a=p*0.017f*(j+1);cc[p*half+j]=cosf(a);ss[p*half+j]=sinf(a);}
    w.cosT=upF(cc);w.sinT=upF(ss);
    std::vector<float> xh((size_t)s*DIM);for(auto&e:xh)e=0.1f*((rand()%200)-100)/100.f; const float* x=upF(xh);
    // Path A: cache PS, then K sequential decode steps
    float *cA,*outA; CU(cudaMalloc(&cA,(size_t)s*HEAD_DIM*4)); CU(cudaMalloc(&outA,(size_t)K*DIM*4));
    mla_cache_kv(cA,x,w,PS);
    for(int i=0;i<K;++i) mla_decode_step(outA+(size_t)i*DIM, x+(size_t)(PS+i)*DIM, w, cA, PS+i);
    // Path B: fresh cache PS, then ONE M=K verify
    float *cB,*outB; CU(cudaMalloc(&cB,(size_t)s*HEAD_DIM*4)); CU(cudaMalloc(&outB,(size_t)K*DIM*4));
    mla_cache_kv(cB,x,w,PS);
    mla_verify_step(outB, x+(size_t)PS*DIM, w, cB, PS, K);
    CU(cudaDeviceSynchronize());
    std::vector<float> a((size_t)K*DIM),b((size_t)K*DIM);
    CU(cudaMemcpy(a.data(),outA,a.size()*4,cudaMemcpyDeviceToHost)); CU(cudaMemcpy(b.data(),outB,b.size()*4,cudaMemcpyDeviceToHost));
    double dot=0,na=0,nb=0,md=0; for(size_t i=0;i<a.size();++i){dot+=a[i]*b[i];na+=a[i]*a[i];nb+=b[i]*b[i];md=fmax(md,fabs(a[i]-b[i]));}
    double cos=dot/(sqrt(na)*sqrt(nb)+1e-30); bool ok=cos>0.999999&&md<1e-2;
    printf("[mla_verify] s=%d PS=%d K=%d cosine=%.7f maxabs=%.2e -> %s\n",s,PS,K,cos,md,ok?"PASS":"FAIL"); return ok?0:1;
}
