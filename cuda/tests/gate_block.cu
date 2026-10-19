// gate_block.cu — Gate K for the full Block forward on REAL layer-1 REAP weights (attn + MoE + HC).
//   build: scripts/build_block.sh ; run: ./build/gate_block ref/goldens/block_layer1_seq16.safetensors
#include "safetensors.h"
#include "block.h"
#include <cstdio>
#include <vector>
#include <cmath>
#include <string>
#define CU(x) do{cudaError_t e=(x); if(e){fprintf(stderr,"cuda %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)

static int i32(const st::Tensor& t,int i){ return ((const int*)t.data)[i]; }
static const float* F(const st::Tensor& t){ return (const float*)t.data; }
template<class T> static const T* up(const st::Tensor& t){
    void* d; CU(cudaMalloc(&d,t.nbytes)); CU(cudaMemcpy(d,t.data,t.nbytes,cudaMemcpyHostToDevice)); return (const T*)d;
}

int main(int argc, char** argv){
    std::string path = argc>1 ? argv[1] : "ref/goldens/block_layer1_seq16.safetensors";
    st::SafeTensors S(path);
    const auto& dm=S.get("dims");
    int s=i32(dm,0), hc=i32(dm,1), dim=i32(dm,2), inter=i32(dm,3), nr=i32(dm,4), na=i32(dm,5), vocab=i32(dm,6);

    BlockWeights w{}; w.dim=dim; w.hc=hc;
    auto& a=w.attn;
    a.wq_a=up<uint8_t>(S.get("wq_a")); a.wq_a_s=up<float>(S.get("wq_a_s"));
    a.wq_b=up<uint8_t>(S.get("wq_b")); a.wq_b_s=up<float>(S.get("wq_b_s"));
    a.wkv=up<uint8_t>(S.get("wkv"));   a.wkv_s=up<float>(S.get("wkv_s"));
    a.wo_b=up<uint8_t>(S.get("wo_b")); a.wo_b_s=up<float>(S.get("wo_b_s"));
    a.q_norm=up<float>(S.get("q_norm")); a.kv_norm=up<float>(S.get("kv_norm"));
    a.wo_a=up<float>(S.get("wo_a")); a.attn_sink=up<float>(S.get("attn_sink"));
    a.cosT=up<float>(S.get("cos")); a.sinT=up<float>(S.get("sin"));
    auto& f=w.ffn;
    f.gate_w=up<float>(S.get("gate_w")); f.gate_bias=nullptr; f.tid2eid=up<long>(S.get("tid2eid")); f.is_hash=true;
    f.w1=up<uint8_t>(S.get("w1")); f.w1s=up<float>(S.get("w1s"));
    f.w2=up<uint8_t>(S.get("w2")); f.w2s=up<float>(S.get("w2s"));
    f.w3=up<uint8_t>(S.get("w3")); f.w3s=up<float>(S.get("w3s"));
    f.sw1=up<uint8_t>(S.get("sw1")); f.sw1s=up<float>(S.get("sw1s"));
    f.sw2=up<uint8_t>(S.get("sw2")); f.sw2s=up<float>(S.get("sw2s"));
    f.sw3=up<uint8_t>(S.get("sw3")); f.sw3s=up<float>(S.get("sw3s"));
    f.n_routed=nr; f.n_act=na; f.dim=dim; f.inter=inter; f.vocab=vocab;
    f.route_scale=F(S.get("route_scale"))[0]; f.swiglu_limit=F(S.get("swiglu_limit"))[0];
    w.attn_norm=up<float>(S.get("attn_norm")); w.ffn_norm=up<float>(S.get("ffn_norm"));
    w.hc_attn_fn=up<float>(S.get("hc_attn_fn")); w.hc_attn_scale=up<float>(S.get("hc_attn_scale")); w.hc_attn_base=up<float>(S.get("hc_attn_base"));
    w.hc_ffn_fn=up<float>(S.get("hc_ffn_fn")); w.hc_ffn_scale=up<float>(S.get("hc_ffn_scale")); w.hc_ffn_base=up<float>(S.get("hc_ffn_base"));

    const float* x=up<float>(S.get("block_in"));
    const int* ids=up<int>(S.get("input_ids"));
    float* out; CU(cudaMalloc(&out,(size_t)s*hc*dim*4));
    block_forward(out, x, ids, w, s, 20, 1e-6f);
    CU(cudaDeviceSynchronize());

    std::vector<float> o((size_t)s*hc*dim); CU(cudaMemcpy(o.data(),out,o.size()*4,cudaMemcpyDeviceToHost));
    const float* oref=F(S.get("block_out"));
    double mx=0,mabs=0,mrel=0;
    for(size_t i=0;i<o.size();++i) mx=fmax(mx,fabs((double)oref[i]));
    for(size_t i=0;i<o.size();++i){ double d=fabs((double)o[i]-oref[i]); mabs=fmax(mabs,d); mrel=fmax(mrel,d/(fabs((double)oref[i])+0.01*mx)); }
    bool ok = mrel < 3e-2;
    printf("[block_forward] s=%d hc=%d dim=%d nr=%d na=%d  |o|max=%.4f max_abs=%.5f max_rel=%.5f -> %s\n",
           s,hc,dim,nr,na,mx,mabs,mrel, ok?"PASS":"FAIL");
    printf("\nGate K (Block): %s\n", ok?"PASS":"FAIL");
    return ok?0:1;
}
