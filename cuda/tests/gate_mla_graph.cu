// gate_mla_graph.cu — CUDA-GRAPH capture of the device-pos sliding decode. Capture mla_decode_step_dp once,
// replay it advancing d_pos, and confirm it equals K sequential mla_decode_step calls (cosine 1.0). Also times
// graph-replay vs un-captured step -> the launch-overhead saving. Synthetic weights, no big-model run.
#include "mla_forward.h"
#include "mla_decode.h"
#include "mla_attn.h"
#include "deepseek_v4.h"
#include "dscratch.h"
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
int main(){
    const int s=20,K=12,PS=s-K,half=ROPE_DIM/2,Kd=N_HEADS*HEAD_DIM,GKd=Kd/O_GROUPS,OB=O_GROUPS*O_LORA; srand(7);
    MLAWeights w{};
    w.wq_a=upW(Q_LORA*DIM);w.wq_a_s=upS((Q_LORA/128)*(DIM/128)); w.wq_b=upW(Kd*Q_LORA);w.wq_b_s=upS((Kd/128)*(Q_LORA/128));
    w.wkv=upW(HEAD_DIM*DIM);w.wkv_s=upS((HEAD_DIM/128)*(DIM/128)); w.wo_b=upW(DIM*OB);w.wo_b_s=upS((DIM/128)*(OB/128));
    {std::vector<float>v(Q_LORA);for(auto&e:v)e=0.5f+0.01f*(rand()%100);w.q_norm=upF(v);}
    {std::vector<float>v(HEAD_DIM);for(auto&e:v)e=0.5f+0.01f*(rand()%100);w.kv_norm=upF(v);}
    {std::vector<float>v((size_t)O_GROUPS*O_LORA*GKd);for(auto&e:v)e=0.02f*((rand()%200)-100)/100.f;w.wo_a=upF(v);}
    {std::vector<float>v(N_HEADS);for(auto&e:v)e=0.01f*(rand()%100);w.attn_sink=upF(v);}
    std::vector<float> cc((size_t)s*half),ss((size_t)s*half);
    for(int p=0;p<s;++p)for(int j=0;j<half;++j){float a=p*0.017f*(j+1);cc[p*half+j]=cosf(a);ss[p*half+j]=sinf(a);}
    w.cosT=upF(cc);w.sinT=upF(ss);
    std::vector<float> xh((size_t)s*DIM);for(auto&e:xh)e=0.1f*((rand()%200)-100)/100.f; const float* x=upF(xh);
    // reference: K sequential mla_decode_step on cache cA
    float *cA,*outA; CU(cudaMalloc(&cA,(size_t)s*HEAD_DIM*4)); CU(cudaMalloc(&outA,(size_t)K*DIM*4));
    mla_cache_kv(cA,x,w,PS); for(int i=0;i<K;++i) mla_decode_step(outA+(size_t)i*DIM,x+(size_t)(PS+i)*DIM,w,cA,PS+i);
    CU(cudaDeviceSynchronize());
    // graph: device-pos step on cache cB, captured once
    arena_init((size_t)64<<20);
    float *cB,*xbuf,*outg; int* d_pos; CU(cudaMalloc(&cB,(size_t)s*HEAD_DIM*4)); CU(cudaMalloc(&xbuf,(size_t)DIM*4)); CU(cudaMalloc(&outg,(size_t)DIM*4)); CU(cudaMalloc(&d_pos,4));
    mla_cache_kv(cB,x,w,PS); CU(cudaDeviceSynchronize());
    cudaStream_t cap; CU(cudaStreamCreate(&cap));
    int p0=PS; CU(cudaMemcpy(d_pos,&p0,4,cudaMemcpyHostToDevice));
    CU(cudaMemcpy(xbuf,x+(size_t)PS*DIM,DIM*4,cudaMemcpyDeviceToDevice));
    arena_reset(); mla_decode_step_dp(outg,xbuf,w,cB,d_pos,s,cap); CU(cudaStreamSynchronize(cap));  // warm (also appends cB[PS])
    // re-seed cB (warm appended PS) and capture
    mla_cache_kv(cB,x,w,PS); CU(cudaDeviceSynchronize());
    arena_reset();
    CU(cudaStreamBeginCapture(cap,cudaStreamCaptureModeThreadLocal));
    mla_decode_step_dp(outg,xbuf,w,cB,d_pos,s,cap);
    cudaGraph_t g; CU(cudaStreamEndCapture(cap,&g)); cudaGraphExec_t exec; CU(cudaGraphInstantiate(&exec,g,0));
    printf("[graph] captured mla_decode_step_dp OK\n");
    std::vector<float> og((size_t)K*DIM);
    for(int i=0;i<K;++i){ int p=PS+i; CU(cudaMemcpy(d_pos,&p,4,cudaMemcpyHostToDevice));
        CU(cudaMemcpy(xbuf,x+(size_t)p*DIM,DIM*4,cudaMemcpyDeviceToDevice));
        CU(cudaGraphLaunch(exec,cap)); CU(cudaStreamSynchronize(cap));
        CU(cudaMemcpy(&og[(size_t)i*DIM],outg,DIM*4,cudaMemcpyDeviceToHost)); }
    std::vector<float> oa((size_t)K*DIM); CU(cudaMemcpy(oa.data(),outA,oa.size()*4,cudaMemcpyDeviceToHost));
    double dot=0,na=0,ng=0,md=0; for(size_t i=0;i<oa.size();++i){dot+=oa[i]*og[i];na+=oa[i]*oa[i];ng+=og[i]*og[i];md=fmax(md,fabs(oa[i]-og[i]));}
    double cos=dot/(sqrt(na)*sqrt(ng)+1e-30); bool ok=cos>0.99999&&md<1e-2;
    printf("[graph] replay vs sequential: cosine=%.7f maxabs=%.2e -> %s\n",cos,md,ok?"PASS":"FAIL");
    // timing: graph replay vs un-captured step
    mla_cache_kv(cB,x,w,PS); CU(cudaDeviceSynchronize()); int IT=60; cudaEvent_t a,b; cudaEventCreate(&a);cudaEventCreate(&b);
    for(int i=0;i<5;++i){int p=PS;CU(cudaMemcpy(d_pos,&p,4,cudaMemcpyHostToDevice));CU(cudaGraphLaunch(exec,cap));} CU(cudaStreamSynchronize(cap));
    cudaEventRecord(a,cap); for(int i=0;i<IT;++i){CU(cudaGraphLaunch(exec,cap));} cudaEventRecord(b,cap); CU(cudaStreamSynchronize(cap));
    float tg=0; cudaEventElapsedTime(&tg,a,b);
    cudaEventRecord(a); for(int i=0;i<IT;++i){arena_reset(); mla_decode_step_dp(outg,xbuf,w,cB,d_pos,s,0);} cudaEventRecord(b); CU(cudaEventSynchronize(b));
    float tn=0; cudaEventElapsedTime(&tn,a,b);
    printf("[graph] per-step: graph %.3f ms | un-captured %.3f ms -> %.2fx (launch-overhead saved)\n", tg/IT, tn/IT, tn/tg);
    return ok?0:1;
}
