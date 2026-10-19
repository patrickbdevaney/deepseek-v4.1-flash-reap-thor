// gate_compressor_emit.cu — EQUIVALENCE gate for the incremental compressor (Step 4 decode, milestone 2 crux).
// compressor_emit_group(g) must reproduce compressor_forward's out[g] for every g (append-only compressed KV).
// Covers non-overlap (ratio-128 strided layer) AND overlap (ratio-4 indexer layer). Synthetic weights, no golden.
//   build: nvcc -O2 -std=c++17 -arch=sm_110a -I include tests/gate_compressor_emit.cu kernels/compressor.cu \
//          kernels/mla_attn.cu kernels/indexer.cu -o build/gate_compressor_emit
#include "compressor.h"
#include "deepseek_v4.h"
#include <cuda_runtime.h>
#include <vector>
#include <cstdio>
#include <cmath>
#include <cstdlib>
using namespace dsv4;
#define CU(x) do{cudaError_t e=(x); if(e){fprintf(stderr,"cuda %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)
static const float* upF(std::vector<float>& h){ float* d; CU(cudaMalloc(&d,h.size()*4)); CU(cudaMemcpy(d,h.data(),h.size()*4,cudaMemcpyHostToDevice)); return d; }
static std::vector<float> rnd(size_t n, float sc){ std::vector<float> h(n); for(auto&v:h)v=sc*((rand()%200)-100)/100.f; return h; }

static bool run(int ratio, bool overlap, bool rotate, int d, int s){
    const int dim=DIM, rope_dim=ROPE_DIM, half=rope_dim/2, od=(overlap?2:1)*d, T=s/ratio;
    std::vector<float> xh=rnd((size_t)s*dim,0.1f), wkvh=rnd((size_t)od*dim,0.02f), wgh=rnd((size_t)od*dim,0.02f);
    std::vector<float> apeh=rnd((size_t)ratio*od,0.1f), nwh(d,1.f);
    for(auto&v:nwh) v=0.5f+0.01f*(rand()%100);
    std::vector<float> cch((size_t)T*half), csh((size_t)T*half);
    for(int t=0;t<T;++t) for(int j=0;j<half;++j){ float a=t*0.013f*(j+1); cch[t*half+j]=cosf(a); csh[t*half+j]=sinf(a); }
    const float *x=upF(xh),*wkv=upF(wkvh),*wgate=upF(wgh),*ape=upF(apeh),*nw=upF(nwh),*cc=upF(cch),*cs=upF(csh);

    float* out_pref; CU(cudaMalloc(&out_pref,(size_t)T*d*4));
    compressor_forward(out_pref, x, wkv, wgate, ape, nw, cc, cs, s, dim, d, ratio, overlap, rope_dim, EPS, rotate, 0);
    CU(cudaDeviceSynchronize());
    float* row; CU(cudaMalloc(&row,(size_t)d*4));
    std::vector<float> op((size_t)T*d), allrow((size_t)T*d);
    CU(cudaMemcpy(op.data(), out_pref, op.size()*4, cudaMemcpyDeviceToHost));
    for(int g=0; g<T; ++g){
        compressor_emit_group(row, x, g, ratio, wkv, wgate, ape, nw, cc, cs, dim, d, overlap, rope_dim, EPS, rotate, 0);
        CU(cudaDeviceSynchronize());
        CU(cudaMemcpy(&allrow[(size_t)g*d], row, d*4, cudaMemcpyDeviceToHost));
    }
    double dot=0,np=0,nd=0,sd=0,sr=0,ma=0,mx=0;
    for(size_t i=0;i<op.size();++i){ double a=op[i],b=allrow[i]; dot+=a*b; np+=a*a; nd+=b*b; sd+=(a-b)*(a-b); sr+=a*a; ma=fmax(ma,fabs(a-b)); mx=fmax(mx,fabs(a)); }
    double cosine=dot/(sqrt(np)*sqrt(nd)+1e-30), rms=sqrt(sd/(sr+1e-30));
    bool ok = cosine>0.99999 && rms<1e-4;
    printf("[compressor_emit ratio=%d overlap=%d rotate=%d d=%d s=%d T=%d] cosine=%.8f rms=%.2e maxabs=%.2e -> %s\n",
           ratio, overlap, rotate, d, s, T, cosine, rms, ma/(mx+1e-30), ok?"PASS":"FAIL");
    return ok;
}

int main(){
    srand(11);
    bool ok=true;
    ok &= run(128, false, false, HEAD_DIM, 256);   // strided layer (ratio-128) main compressor
    ok &= run(4,   true,  false, HEAD_DIM, 16);     // indexer layer (ratio-4) main compressor (overlap)
    ok &= run(4,   true,  true,  INDEX_HEAD_DIM, 16);// indexer's OWN compressor (rotate=hadamard+fp4sim)
    printf("\ncompressor_emit equivalence: %s\n", ok?"PASS":"FAIL");
    return ok?0:1;
}
