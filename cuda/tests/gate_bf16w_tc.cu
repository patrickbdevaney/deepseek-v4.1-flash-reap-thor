// gate_bf16w_tc.cu — equivalence gate for the bf16 TENSOR-CORE gemm_bf16w at PREFILL shapes.
//
// WHY THIS FILE EXISTS. `gate_bf16w` only exercises M=1 (the lm_head shapes), so when the
// tensor-core path was added for M>=64 there was NO gate in the project that could reach it. Five
// existing gates were run against it and all five passed — because none of them executed a single
// instruction of the new kernel. Three of those PASSes reported `rms=0.00e+00` on a change that
// rounds fp32 activations to bf16, which is impossible, and that number is the only reason the gap
// was noticed. A gate that cannot fail is not evidence.
//
// WHAT IT CHECKS. The TC kernel against the project's trusted scalar `gemm_bf16w_kernel` (itself
// gated at M=1 against the PyTorch oracle by gate_bf16w), on the real compressor shapes. This is a
// NUMERICS change, not a bit-exact one — the fp32 A operand is rounded to bf16 before the mma — so
// the bar is cosine/rms, not memcmp. The tolerance is set for bf16's 8 mantissa bits.
//
//   nvcc -O3 -arch=sm_110a -I include tests/gate_bf16w_tc.cu kernels/compressor.cu \
//        kernels/dscratch.cu kernels/mla_attn.cu kernels/indexer.cu kernels/fp8_block_gemm.cu \
//        kernels/tc_fp8_gemm.cu -o build/gate_bf16w_tc
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

void gemm_bf16w(float* C, const float* A, const void* B, int M, int N, int K, cudaStream_t s);

#define CU(x) do{ cudaError_t e=(x); if(e){ printf("CUDA %s @%d\n", cudaGetErrorString(e), __LINE__); return 1; } }while(0)

static int run_shape(int M, int N, int K){
    std::vector<float> hA((size_t)M*K);
    std::vector<__nv_bfloat16> hB((size_t)N*K);
    // Deterministic, no RNG: a fixed mixed-sign pattern with realistic magnitude spread.
    for(size_t i=0;i<hA.size();++i) hA[i] = (float)(((i*1103515245u+12345u)>>16)%2000 - 1000) / 500.0f;
    for(size_t i=0;i<hB.size();++i) hB[i] = __float2bfloat16((float)(((i*22695477u+1u)>>16)%2000 - 1000) / 1000.0f);

    float *dA,*dCa,*dCb; __nv_bfloat16* dB;
    CU(cudaMalloc(&dA,hA.size()*4)); CU(cudaMalloc(&dB,hB.size()*2));
    CU(cudaMalloc(&dCa,(size_t)M*N*4)); CU(cudaMalloc(&dCb,(size_t)M*N*4));
    CU(cudaMemcpy(dA,hA.data(),hA.size()*4,cudaMemcpyHostToDevice));
    CU(cudaMemcpy(dB,hB.data(),hB.size()*2,cudaMemcpyHostToDevice));

    unsetenv("TC_BF16W");                       // trusted scalar arm
    gemm_bf16w(dCa,dA,dB,M,N,K,0); CU(cudaDeviceSynchronize());
    setenv("TC_BF16W","1",1);                   // tensor-core arm
    gemm_bf16w(dCb,dA,dB,M,N,K,0); CU(cudaDeviceSynchronize());
    unsetenv("TC_BF16W");

    std::vector<float> a((size_t)M*N), b((size_t)M*N);
    CU(cudaMemcpy(a.data(),dCa,a.size()*4,cudaMemcpyDeviceToHost));
    CU(cudaMemcpy(b.data(),dCb,b.size()*4,cudaMemcpyDeviceToHost));

    double dot=0, na=0, nb=0, se=0, mx=0, amax=0;
    for(size_t i=0;i<a.size();++i){
        dot += (double)a[i]*b[i]; na += (double)a[i]*a[i]; nb += (double)b[i]*b[i];
        double d = fabs((double)a[i]-b[i]); se += d*d; if(d>mx) mx=d;
        if(fabs((double)a[i])>amax) amax=fabs((double)a[i]);
    }
    double cos = dot/(sqrt(na)*sqrt(nb)+1e-30);
    double rms_rel = sqrt(se/a.size())/(amax+1e-30);
    // bf16 carries 8 mantissa bits -> ~4e-3 relative per element; a K=4096 dot averages error down.
    bool ok = (cos > 0.99999) && (rms_rel < 5e-3);
    printf("[bf16w_tc] M=%d N=%d K=%d  cosine=%.8f  rms_rel=%.2e  max_abs/|c|max=%.2e -> %s\n",
           M,N,K,cos,rms_rel,mx/(amax+1e-30), ok?"PASS":"FAIL");
    cudaFree(dA);cudaFree(dB);cudaFree(dCa);cudaFree(dCb);
    return ok?0:1;
}

int main(){
    int bad=0;
    bad |= run_shape(1022, 1024, 4096);   // main compressor, overlap: od = 2*512
    bad |= run_shape(1022,  512, 4096);   // main compressor, non-overlap
    bad |= run_shape(1022,  256, 4096);   // indexer compressor: od = 2*128
    bad |= run_shape(  64,  128,  512);   // smallest shape the dispatch admits (M>=64)
    bad |= run_shape(1000, 1024, 4096);   // M not a multiple of BM=64 -> tail masking
    bad |= run_shape(1022, 1000, 4096);   // N not a multiple of BN=64 -> tail masking
    printf("gate_bf16w_tc: %s\n", bad?"FAIL":"PASS");
    return bad;
}
