// gate_indexer_graph.cu — CUDA-graph capture of the device-pos INDEXER (ratio-4) compressed decode vs sequential.
#include "compressed_attn.h"
#include "compressed_decode.h"
#include "deepseek_v4.h"
#include "dscratch.h"
#include "kv_pack.h"
#include <cuda_runtime.h>
#include <vector>
#include <cstdio>
#include <cmath>
#include <cstdlib>
using namespace dsv4;
#define CU(x) do{cudaError_t e=(x); if(e){fprintf(stderr,"cuda %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)
void compressed_decode_step_indexer_dp(float*, const float*, const float*, const CompressedAttnWeights&, float*, float*, const int*, int*, int*, int, int, int, float, cudaStream_t);
static uint8_t rfp8(){ return (uint8_t)((rand()%0x40)|((rand()&1)<<7)); }
static const uint8_t* upW(size_t n){ std::vector<uint8_t> h(n); for(auto&v:h)v=rfp8(); uint8_t*d; CU(cudaMalloc(&d,n)); CU(cudaMemcpy(d,h.data(),n,cudaMemcpyHostToDevice)); return d; }
static const float* upS(size_t n){ std::vector<float> h(n); for(auto&v:h)v=0.3f+0.01f*(rand()%40); float*d; CU(cudaMalloc(&d,n*4)); CU(cudaMemcpy(d,h.data(),n*4,cudaMemcpyHostToDevice)); return d; }
static const float* upFv(std::vector<float>&h){ float*d; CU(cudaMalloc(&d,h.size()*4)); CU(cudaMemcpy(d,h.data(),h.size()*4,cudaMemcpyHostToDevice)); return d; }
static const float* upR(size_t n,float sc){ std::vector<float> h(n); for(auto&v:h)v=sc*((rand()%200)-100)/100.f; return upFv(h); }
static const float* upN(int n){ std::vector<float> v(n); for(auto&e:v)e=0.5f+0.01f*(rand()%100); return upFv(v); }
int main(){
    // DECODE_LADDER 1b.2. The device-pos graph path has its OWN KV store kernels (kv_store_at and
    // kv_store_comp replace k_append_at2 and k_commit_comp), and nothing else exercises them.
    // DSV4_KV_PACK=1 runs this whole gate on the packed layout, so the graph-vs-sequential
    // comparison below covers those two kernels against kv_stage/kv_commit.
    kv_pack_init();
    printf("[graph-gate] KV layout: %s, %d B/row\n", g_kv_pack?"PACKED (1b.2)":"fp32", g_kv_rowf*4);
    const int ratio=4, s=24, K=16, PS=s-K, half=ROPE_DIM/2, Kd=N_HEADS*HEAD_DIM, GKd=Kd/O_GROUPS, OB=O_GROUPS*O_LORA;
    const int nH=INDEX_N_HEADS, ihd=INDEX_HEAD_DIM, QD=nH*ihd, iod=2*ihd; srand(31);
    CompressedAttnWeights w{}; MLAWeights& a=w.attn;
    a.wq_a=upW((size_t)Q_LORA*DIM);a.wq_a_s=upS((size_t)(Q_LORA/128)*(DIM/128)); a.wq_b=upW((size_t)Kd*Q_LORA);a.wq_b_s=upS((size_t)(Kd/128)*(Q_LORA/128));
    a.wkv=upW((size_t)HEAD_DIM*DIM);a.wkv_s=upS((size_t)(HEAD_DIM/128)*(DIM/128)); a.wo_b=upW((size_t)DIM*OB);a.wo_b_s=upS((size_t)(DIM/128)*(OB/128));
    a.q_norm=upN(Q_LORA);a.kv_norm=upN(HEAD_DIM); a.wo_a=upR((size_t)O_GROUPS*O_LORA*GKd,0.02f); a.attn_sink=upR(N_HEADS,0.1f);
    std::vector<float> cq((size_t)s*half),sq((size_t)s*half); for(int p=0;p<s;++p)for(int j=0;j<half;++j){float ang=p*0.011f*(j+1);cq[p*half+j]=cosf(ang);sq[p*half+j]=sinf(ang);} a.cosT=upFv(cq);a.sinT=upFv(sq);
    w.mc_wkv=upR((size_t)2*HEAD_DIM*DIM,0.02f);w.mc_wgate=upR((size_t)2*HEAD_DIM*DIM,0.02f);w.mc_ape=upR((size_t)ratio*2*HEAD_DIM,0.1f);w.mc_norm=upN(HEAD_DIM);
    int T=s/ratio; std::vector<float> cc((size_t)T*half),cs((size_t)T*half); for(int t=0;t<T;++t)for(int j=0;j<half;++j){float ang=t*0.019f*(j+1);cc[t*half+j]=cosf(ang);cs[t*half+j]=sinf(ang);} w.cc_cos=upFv(cc);w.cc_sin=upFv(cs);
    w.idx_wq_b=upW((size_t)QD*Q_LORA);w.idx_wq_b_s=upS((size_t)(QD/128)*(Q_LORA/128)); w.idx_weights_proj=upR((size_t)nH*DIM,0.02f);
    w.idx_c_wkv=upR((size_t)iod*DIM,0.02f);w.idx_c_wgate=upR((size_t)iod*DIM,0.02f);w.idx_c_ape=upR((size_t)ratio*iod,0.1f);w.idx_c_norm=upN(ihd);
    w.index_n_heads=nH;w.index_head_dim=ihd;w.index_topk=INDEX_TOPK;
    std::vector<float> xh((size_t)s*DIM);for(auto&e:xh)e=0.1f*((rand()%200)-100)/100.f; const float* x=upFv(xh);
    int winmax=s, Tmax=s/ratio+2;
    float *win_kv,*comp_kv,*idx_ckv,*outS; CU(cudaMalloc(&win_kv,kv_rows_bytes(s))); CU(cudaMalloc(&comp_kv,kv_rows_bytes(Tmax))); CU(cudaMalloc(&idx_ckv,(size_t)Tmax*ihd*4)); CU(cudaMalloc(&outS,(size_t)K*DIM*4));
    int Th=0; compressed_attn_cache_r4(win_kv,comp_kv,idx_ckv,&Th,x,w,PS,ratio,EPS); int T0=Th;
    for(int i=0;i<K;++i) compressed_decode_step_indexer(outS+(size_t)i*DIM,x+(size_t)(PS+i)*DIM,x,PS+i,w,win_kv,comp_kv,idx_ckv,&Th,ratio,EPS);
    CU(cudaDeviceSynchronize());
    arena_init((size_t)256<<20);
    float *kvc,*ikvc,*xbuf,*outg; int *d_pos,*d_T,*d_g; CU(cudaMalloc(&kvc,kv_rows_bytes((size_t)winmax+Tmax))); CU(cudaMalloc(&ikvc,(size_t)Tmax*ihd*4)); CU(cudaMalloc(&xbuf,(size_t)DIM*4)); CU(cudaMalloc(&outg,(size_t)DIM*4));
    CU(cudaMalloc(&d_pos,4)); CU(cudaMalloc(&d_T,4)); CU(cudaMalloc(&d_g,4));
    auto seed=[&](){ int Tp=0; float *wk,*ck,*ik; CU(cudaMalloc(&wk,kv_rows_bytes(s))); CU(cudaMalloc(&ck,kv_rows_bytes(Tmax))); CU(cudaMalloc(&ik,(size_t)Tmax*ihd*4));
        compressed_attn_cache_r4(wk,ck,ik,&Tp,x,w,PS,ratio,EPS); CU(cudaDeviceSynchronize());
        CU(cudaMemcpy(kvc,wk,kv_rows_bytes(PS),cudaMemcpyDeviceToDevice)); CU(cudaMemcpy(kv_row(kvc,winmax),ck,kv_rows_bytes(Tp),cudaMemcpyDeviceToDevice));
        CU(cudaMemcpy(ikvc,ik,(size_t)Tp*ihd*4,cudaMemcpyDeviceToDevice)); CU(cudaMemcpy(d_T,&Tp,4,cudaMemcpyHostToDevice)); cudaFree(wk);cudaFree(ck);cudaFree(ik); };
    seed(); cudaStream_t cap; CU(cudaStreamCreate(&cap));
    int p0=PS; CU(cudaMemcpy(d_pos,&p0,4,cudaMemcpyHostToDevice)); CU(cudaMemcpy(xbuf,x+(size_t)PS*DIM,DIM*4,cudaMemcpyDeviceToDevice));
    arena_reset(); compressed_decode_step_indexer_dp(outg,xbuf,x,w,kvc,ikvc,d_pos,d_T,d_g,winmax,Tmax,ratio,EPS,cap); CU(cudaStreamSynchronize(cap));
    seed(); arena_reset(); CU(cudaStreamBeginCapture(cap,cudaStreamCaptureModeThreadLocal));
    compressed_decode_step_indexer_dp(outg,xbuf,x,w,kvc,ikvc,d_pos,d_T,d_g,winmax,Tmax,ratio,EPS,cap);
    cudaGraph_t g; CU(cudaStreamEndCapture(cap,&g)); cudaGraphExec_t exec; CU(cudaGraphInstantiate(&exec,g,0));
    printf("[igraph] captured compressed_decode_step_indexer_dp OK\n");
    std::vector<float> og((size_t)K*DIM);
    for(int i=0;i<K;++i){ int p=PS+i; CU(cudaMemcpy(d_pos,&p,4,cudaMemcpyHostToDevice)); CU(cudaMemcpy(xbuf,x+(size_t)p*DIM,DIM*4,cudaMemcpyDeviceToDevice));
        CU(cudaGraphLaunch(exec,cap)); CU(cudaStreamSynchronize(cap)); CU(cudaMemcpy(&og[(size_t)i*DIM],outg,DIM*4,cudaMemcpyDeviceToHost)); }
    std::vector<float> os((size_t)K*DIM); CU(cudaMemcpy(os.data(),outS,os.size()*4,cudaMemcpyDeviceToHost));
    double dot=0,na=0,ng=0,md=0,mx=0; for(size_t i=0;i<os.size();++i){dot+=os[i]*og[i];na+=os[i]*os[i];ng+=og[i]*og[i];md=fmax(md,fabs(os[i]-og[i]));mx=fmax(mx,fabs(os[i]));}
    double cos=dot/(sqrt(na)*sqrt(ng)+1e-30); bool ok=cos>0.9999&&md/(mx+1e-30)<2e-2;
    printf("[igraph] indexer_dp graph replay vs sequential: cosine=%.7f maxabs/|o|=%.2e -> %s (T0=%d)\n",cos,md/(mx+1e-30),ok?"PASS":"FAIL",T0);
    return ok?0:1;
}
