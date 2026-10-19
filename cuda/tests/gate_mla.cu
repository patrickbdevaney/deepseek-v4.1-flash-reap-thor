// gate_mla.cu — Gate K for the full MLA forward composition on REAL layer weights.
//   build: nvcc -O2 -arch=sm_110a -I include tests/gate_mla.cu kernels/mla_forward.cu \
//          kernels/fp8_block_gemm.cu kernels/mla_attn.cu -o build/gate_mla
//   run:   ./build/gate_mla ref/goldens/mla_layer1_seq16.safetensors
#include "safetensors.h"
#include "mla_forward.h"
#include <cstdio>
#include <vector>
#include <cmath>
#include <string>

#define CU(x) do{cudaError_t e=(x); if(e){fprintf(stderr,"cuda %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)

static const uint8_t* U8(const st::Tensor& t){ return t.data; }
static const float*   F(const st::Tensor& t){ return (const float*)t.data; }
static int   i32(const st::Tensor& t,int i){ return ((const int*)t.data)[i]; }

template<class T> static const T* up(const st::Tensor& t){
    void* d; CU(cudaMalloc(&d,t.nbytes)); CU(cudaMemcpy(d,t.data,t.nbytes,cudaMemcpyHostToDevice)); return (const T*)d;
}

int main(int argc, char** argv){
    std::string path = argc>1 ? argv[1] : "ref/goldens/mla_layer1_seq16.safetensors";
    st::SafeTensors S(path);
    int b=i32(S.get("dims"),0), s=i32(S.get("dims"),1);

    MLAWeights w{};
    w.wq_a=up<uint8_t>(S.get("wq_a")); w.wq_a_s=up<float>(S.get("wq_a_s"));
    w.wq_b=up<uint8_t>(S.get("wq_b")); w.wq_b_s=up<float>(S.get("wq_b_s"));
    w.wkv =up<uint8_t>(S.get("wkv"));  w.wkv_s =up<float>(S.get("wkv_s"));
    w.wo_b=up<uint8_t>(S.get("wo_b")); w.wo_b_s=up<float>(S.get("wo_b_s"));
    w.q_norm=up<float>(S.get("q_norm")); w.kv_norm=up<float>(S.get("kv_norm"));
    w.wo_a=up<float>(S.get("wo_a")); w.attn_sink=up<float>(S.get("attn_sink"));
    w.cosT=up<float>(S.get("cos")); w.sinT=up<float>(S.get("sin"));
    const float* x=up<float>(S.get("x"));

    int dim = 4096;
    float* out; CU(cudaMalloc(&out,(size_t)b*s*dim*4));
    mla_forward(out, x, w, b, s);
    CU(cudaDeviceSynchronize());

    std::vector<float> o((size_t)b*s*dim); CU(cudaMemcpy(o.data(),out,o.size()*4,cudaMemcpyDeviceToHost));
    const float* oref=F(S.get("o_ref"));
    double mx=0,mabs=0,mrel=0;
    for(size_t i=0;i<o.size();++i){ double r=fabs((double)oref[i]); mx=fmax(mx,r);
        double d=fabs((double)o[i]-oref[i]); mabs=fmax(mabs,d); }
    for(size_t i=0;i<o.size();++i){ double d=fabs((double)o[i]-oref[i]); mrel=fmax(mrel,d/(fabs((double)oref[i])+0.01*mx)); }
    bool ok = mrel < 2e-2;
    printf("[mla_forward] b=%d s=%d dim=%d  |o|max=%.4f  max_abs=%.5f  max_rel=%.5f  -> %s\n",
           b,s,dim,mx,mabs,mrel, ok?"PASS":"FAIL");
    printf("\nGate K (MLA): %s\n", ok?"PASS":"FAIL");
    return ok?0:1;
}
