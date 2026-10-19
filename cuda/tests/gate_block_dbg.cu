// gate_block_dbg.cu — staged Block debug: run each stage, compare to golden taps, print where it diverges.
#include "safetensors.h"
#include "block.h"
#include "hc.h"
#include "mla_attn.h"     // rmsnorm
#include <cstdio>
#include <vector>
#include <cmath>
#include <string>
#define CU(x) do{cudaError_t e=(x); if(e){fprintf(stderr,"cuda %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)
static int i32(const st::Tensor& t,int i){ return ((const int*)t.data)[i]; }
static const float* F(const st::Tensor& t){ return (const float*)t.data; }
template<class T> static const T* up(const st::Tensor& t){ void* d; CU(cudaMalloc(&d,t.nbytes)); CU(cudaMemcpy(d,t.data,t.nbytes,cudaMemcpyHostToDevice)); return (const T*)d; }

static void cmp(const char* name, const float* dev, const st::Tensor& ref, int n){
    std::vector<float> h(n); CU(cudaMemcpy(h.data(),dev,(size_t)n*4,cudaMemcpyDeviceToHost));
    const float* r=F(ref); double mx=0,ma=0; for(int i=0;i<n;++i) mx=fmax(mx,fabs((double)r[i]));
    for(int i=0;i<n;++i) ma=fmax(ma,fabs((double)h[i]-r[i]));
    printf("  %-16s max_abs=%.5f  (|ref|max=%.4f  rel=%.4f)\n", name, ma, mx, ma/(mx+1e-9));
}

int main(int argc, char** argv){
    std::string path = argc>1?argv[1]:"ref/goldens/block_layer1_seq16.safetensors";
    st::SafeTensors S(path);
    const auto& dm=S.get("dims");
    int s=i32(dm,0),hc=i32(dm,1),dim=i32(dm,2),inter=i32(dm,3),nr=i32(dm,4),na=i32(dm,5),vocab=i32(dm,6);
    int bs=s;
    BlockWeights w{}; w.dim=dim; w.hc=hc; auto&a=w.attn; auto&f=w.ffn;
    a.wq_a=up<uint8_t>(S.get("wq_a")); a.wq_a_s=up<float>(S.get("wq_a_s")); a.wq_b=up<uint8_t>(S.get("wq_b")); a.wq_b_s=up<float>(S.get("wq_b_s"));
    a.wkv=up<uint8_t>(S.get("wkv")); a.wkv_s=up<float>(S.get("wkv_s")); a.wo_b=up<uint8_t>(S.get("wo_b")); a.wo_b_s=up<float>(S.get("wo_b_s"));
    a.q_norm=up<float>(S.get("q_norm")); a.kv_norm=up<float>(S.get("kv_norm")); a.wo_a=up<float>(S.get("wo_a")); a.attn_sink=up<float>(S.get("attn_sink"));
    a.cosT=up<float>(S.get("cos")); a.sinT=up<float>(S.get("sin"));
    f.gate_w=up<float>(S.get("gate_w")); f.gate_bias=nullptr; f.tid2eid=up<long>(S.get("tid2eid")); f.is_hash=true;
    f.w1=up<uint8_t>(S.get("w1")); f.w1s=up<float>(S.get("w1s")); f.w2=up<uint8_t>(S.get("w2")); f.w2s=up<float>(S.get("w2s")); f.w3=up<uint8_t>(S.get("w3")); f.w3s=up<float>(S.get("w3s"));
    f.sw1=up<uint8_t>(S.get("sw1")); f.sw1s=up<float>(S.get("sw1s")); f.sw2=up<uint8_t>(S.get("sw2")); f.sw2s=up<float>(S.get("sw2s")); f.sw3=up<uint8_t>(S.get("sw3")); f.sw3s=up<float>(S.get("sw3s"));
    f.n_routed=nr; f.n_act=na; f.dim=dim; f.inter=inter; f.vocab=vocab; f.route_scale=F(S.get("route_scale"))[0]; f.swiglu_limit=F(S.get("swiglu_limit"))[0];
    w.attn_norm=up<float>(S.get("attn_norm")); w.ffn_norm=up<float>(S.get("ffn_norm"));
    w.hc_attn_fn=up<float>(S.get("hc_attn_fn")); w.hc_attn_scale=up<float>(S.get("hc_attn_scale")); w.hc_attn_base=up<float>(S.get("hc_attn_base"));
    w.hc_ffn_fn=up<float>(S.get("hc_ffn_fn")); w.hc_ffn_scale=up<float>(S.get("hc_ffn_scale")); w.hc_ffn_base=up<float>(S.get("hc_ffn_base"));
    const float* x=up<float>(S.get("block_in")); const int* ids=up<int>(S.get("input_ids"));

    float *x1,*post,*comb,*sub,*res2,*out;
    CU(cudaMalloc(&x1,(size_t)bs*dim*4)); CU(cudaMalloc(&post,(size_t)bs*hc*4)); CU(cudaMalloc(&comb,(size_t)bs*hc*hc*4));
    CU(cudaMalloc(&sub,(size_t)bs*dim*4)); CU(cudaMalloc(&res2,(size_t)bs*hc*dim*4)); CU(cudaMalloc(&out,(size_t)bs*hc*dim*4));

    hc_pre(x1,post,comb,x,w.hc_attn_fn,w.hc_attn_scale,w.hc_attn_base,bs,hc,dim,20,1e-6f);
    cmp("hc_attn_out", x1, S.get("tap_hc_attn_out"), bs*dim);
    cmp("post", post, S.get("tap_post_a"), bs*hc);
    cmp("comb", comb, S.get("tap_comb_a"), bs*hc*hc);
    rmsnorm(x1,x1,w.attn_norm,bs,dim,1e-6f,true); CU(cudaDeviceSynchronize());
    mla_forward(sub,x1,w.attn,1,s); CU(cudaDeviceSynchronize());
    cmp("attn_out(mla)", sub, S.get("tap_attn_out"), bs*dim);
    hc_post(res2,sub,x,post,comb,bs,hc,dim); CU(cudaDeviceSynchronize());
    cmp("res2", res2, S.get("tap_res2"), bs*hc*dim);
    hc_pre(x1,post,comb,res2,w.hc_ffn_fn,w.hc_ffn_scale,w.hc_ffn_base,bs,hc,dim,20,1e-6f);
    cmp("hc_ffn_out", x1, S.get("tap_hc_ffn_out"), bs*dim);
    rmsnorm(x1,x1,w.ffn_norm,bs,dim,1e-6f,true); CU(cudaDeviceSynchronize());
    moe_forward(sub,x1,ids,w.ffn,bs); CU(cudaDeviceSynchronize());
    cmp("moe_out", sub, S.get("tap_moe_out"), bs*dim);
    hc_post(out,sub,res2,post,comb,bs,hc,dim); CU(cudaDeviceSynchronize());
    cmp("block_out", out, S.get("block_out"), bs*hc*dim);
    return 0;
}
