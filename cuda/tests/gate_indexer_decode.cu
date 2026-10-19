// gate_indexer_decode.cu — EQUIVALENCE gate for the ratio-4 DSA-indexer compressed decode step (Step 4 m2b).
// prefill compressed_attn_forward(ratio=4) vs compressed_attn_cache_r4(0..s-2)+compressed_decode_step_indexer(s-1).
// Exercises: overlap main compressor + indexer overlap/rotate compressor + DSA scoring + top-k selection +
// sparse_attn over [window ⊕ indexer-selected compressed]. Synthetic weights, path-vs-path -> expect bit-exact.
//   build: nvcc -O2 -std=c++17 -arch=sm_110a -I include tests/gate_indexer_decode.cu kernels/compressed_decode.cu \
//     kernels/compressed_attn.cu kernels/compressor.cu kernels/indexer.cu kernels/mla_attn.cu \
//     kernels/fp8_block_gemm.cu kernels/tc_fp8_gemm.cu -o build/gate_indexer_decode
#include "compressed_attn.h"
#include "compressed_decode.h"
#include "deepseek_v4.h"
#include <cuda_runtime.h>
#include <vector>
#include <cstdio>
#include <cmath>
#include <cstdlib>
using namespace dsv4;
#define CU(x) do{cudaError_t e=(x); if(e){fprintf(stderr,"cuda %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)
static uint8_t rfp8(){ return (uint8_t)((rand()%0x40) | ((rand()&1)<<7)); }
static const uint8_t* upW(size_t n){ std::vector<uint8_t> h(n); for(auto&v:h)v=rfp8();
    uint8_t* d; CU(cudaMalloc(&d,n)); CU(cudaMemcpy(d,h.data(),n,cudaMemcpyHostToDevice)); return d; }
static const float* upS(size_t n){ std::vector<float> h(n); for(auto&v:h)v=0.3f+0.01f*(rand()%40);
    float* d; CU(cudaMalloc(&d,n*4)); CU(cudaMemcpy(d,h.data(),n*4,cudaMemcpyHostToDevice)); return d; }
static const float* upFv(std::vector<float>& h){ float* d; CU(cudaMalloc(&d,h.size()*4));
    CU(cudaMemcpy(d,h.data(),h.size()*4,cudaMemcpyHostToDevice)); return d; }
static const float* upR(size_t n, float sc){ std::vector<float> h(n); for(auto&v:h)v=sc*((rand()%200)-100)/100.f; return upFv(h); }
static const float* upNorm(int n){ std::vector<float> v(n); for(auto&e:v)e=0.5f+0.01f*(rand()%100); return upFv(v); }

int main(int argc, char** argv){
    const int s = argc>1?atoi(argv[1]):16, ratio=4, win=WINDOW, half=ROPE_DIM/2;
    const int Kd=N_HEADS*HEAD_DIM, GKd=Kd/O_GROUPS, OB=O_GROUPS*O_LORA, T=s/ratio;
    const int nH=INDEX_N_HEADS, ihd=INDEX_HEAD_DIM, QD=nH*ihd, iod=2*ihd;
    srand(31);
    CompressedAttnWeights w{}; MLAWeights& a=w.attn;
    a.wq_a=upW((size_t)Q_LORA*DIM);   a.wq_a_s=upS((size_t)(Q_LORA/128)*(DIM/128));
    a.wq_b=upW((size_t)Kd*Q_LORA);    a.wq_b_s=upS((size_t)(Kd/128)*(Q_LORA/128));
    a.wkv =upW((size_t)HEAD_DIM*DIM); a.wkv_s =upS((size_t)(HEAD_DIM/128)*(DIM/128));
    a.wo_b=upW((size_t)DIM*OB);       a.wo_b_s=upS((size_t)(DIM/128)*(OB/128));
    a.q_norm=upNorm(Q_LORA); a.kv_norm=upNorm(HEAD_DIM);
    a.wo_a=upR((size_t)O_GROUPS*O_LORA*GKd,0.02f); a.attn_sink=upR(N_HEADS,0.1f);
    std::vector<float> cq((size_t)s*half), sq((size_t)s*half);
    for(int p=0;p<s;++p) for(int j=0;j<half;++j){ float ang=p*0.011f*(j+1); cq[p*half+j]=cosf(ang); sq[p*half+j]=sinf(ang); }
    a.cosT=upFv(cq); a.sinT=upFv(sq);
    // main compressor (overlap, d=HEAD_DIM) -> od=2*HEAD_DIM
    w.mc_wkv=upR((size_t)2*HEAD_DIM*DIM,0.02f); w.mc_wgate=upR((size_t)2*HEAD_DIM*DIM,0.02f);
    w.mc_ape=upR((size_t)ratio*2*HEAD_DIM,0.1f); w.mc_norm=upNorm(HEAD_DIM);
    std::vector<float> cc((size_t)T*half), cs((size_t)T*half);
    for(int t=0;t<T;++t) for(int j=0;j<half;++j){ float ang=t*0.019f*(j+1); cc[t*half+j]=cosf(ang); cs[t*half+j]=sinf(ang); }
    w.cc_cos=upFv(cc); w.cc_sin=upFv(cs);
    // indexer weights
    w.idx_wq_b=upW((size_t)QD*Q_LORA); w.idx_wq_b_s=upS((size_t)(QD/128)*(Q_LORA/128));
    w.idx_weights_proj=upR((size_t)nH*DIM,0.02f);
    w.idx_c_wkv=upR((size_t)iod*DIM,0.02f); w.idx_c_wgate=upR((size_t)iod*DIM,0.02f);
    w.idx_c_ape=upR((size_t)ratio*iod,0.1f); w.idx_c_norm=upNorm(ihd);
    w.index_n_heads=nH; w.index_head_dim=ihd; w.index_topk=INDEX_TOPK;
    std::vector<float> xh((size_t)s*DIM); for(auto&e:xh)e=0.1f*((rand()%200)-100)/100.f;
    const float* x=upFv(xh);

    float* out_pref; CU(cudaMalloc(&out_pref,(size_t)s*DIM*4));
    compressed_attn_forward(out_pref, x, w, s, win, ratio, EPS); CU(cudaDeviceSynchronize());

    float *win_kv,*comp_kv,*idx_ckv;
    CU(cudaMalloc(&win_kv,(size_t)s*HEAD_DIM*4)); CU(cudaMalloc(&comp_kv,(size_t)T*HEAD_DIM*4)); CU(cudaMalloc(&idx_ckv,(size_t)T*ihd*4));
    int Th=0;
    compressed_attn_cache_r4(win_kv, comp_kv, idx_ckv, &Th, x, w, s-1, ratio, EPS);
    float* out_dec; CU(cudaMalloc(&out_dec,(size_t)DIM*4));
    compressed_decode_step_indexer(out_dec, x+(size_t)(s-1)*DIM, x, s-1, w, win_kv, comp_kv, idx_ckv, &Th, ratio, EPS);
    CU(cudaDeviceSynchronize());

    std::vector<float> op(DIM), od(DIM);
    CU(cudaMemcpy(op.data(), out_pref+(size_t)(s-1)*DIM, DIM*4, cudaMemcpyDeviceToHost));
    CU(cudaMemcpy(od.data(), out_dec, DIM*4, cudaMemcpyDeviceToHost));
    double dot=0,np=0,nd=0,sd=0,sr=0,mx=0,ma=0;
    for(int i=0;i<DIM;++i){ double p=op[i],q=od[i]; dot+=p*q; np+=p*p; nd+=q*q; sd+=(p-q)*(p-q); sr+=p*p; mx=fmax(mx,fabs(p)); ma=fmax(ma,fabs(p-q)); }
    double cosine=dot/(sqrt(np)*sqrt(nd)+1e-30), rms=sqrt(sd/(sr+1e-30));
    bool ok = cosine>0.99999 && rms<1e-3;
    printf("[indexer_decode ratio=%d s=%d pos=%d T=%d topk=%d] cosine=%.8f rms=%.2e maxabs=%.2e -> %s\n",
           ratio, s, s-1, Th, (INDEX_TOPK<Th?INDEX_TOPK:Th), cosine, rms, ma/(mx+1e-30), ok?"PASS":"FAIL");
    return ok?0:1;
}
