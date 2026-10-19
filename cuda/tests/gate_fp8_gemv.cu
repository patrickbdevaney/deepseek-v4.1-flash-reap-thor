// gate_fp8_gemv.cu — M=1 fp8 GEMV vs fp8_block_gemm oracle (bit-exact: same dec_e4m3 + per-128 scales).
#include "fp8_block_gemm.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#define CU(x) do{cudaError_t e=(x); if(e){fprintf(stderr,"cuda %s\n",cudaGetErrorString(e));exit(1);} }while(0)
extern bool g_tc_fp8;
int main(){
    int M=1,N=2048,K=1024; srand(7);
    std::vector<uint8_t> A(M*K), B((size_t)N*K); std::vector<float> as(M*(K/128)), bs((size_t)(N/128)*(K/128));
    for(auto&v:A)v=rand()%0x40; for(auto&v:B)v=rand()%0x40; for(auto&v:as)v=0.5f+0.01f*(rand()%50); for(auto&v:bs)v=0.5f+0.01f*(rand()%50);
    uint8_t *dA,*dB; float *das,*dbs,*Cr,*Cg;
    CU(cudaMalloc(&dA,A.size()));CU(cudaMalloc(&dB,B.size()));CU(cudaMalloc(&das,as.size()*4));CU(cudaMalloc(&dbs,bs.size()*4));
    CU(cudaMalloc(&Cr,(size_t)M*N*4));CU(cudaMalloc(&Cg,(size_t)M*N*4));
    CU(cudaMemcpy(dA,A.data(),A.size(),cudaMemcpyHostToDevice));CU(cudaMemcpy(dB,B.data(),B.size(),cudaMemcpyHostToDevice));
    CU(cudaMemcpy(das,as.data(),as.size()*4,cudaMemcpyHostToDevice));CU(cudaMemcpy(dbs,bs.data(),bs.size()*4,cudaMemcpyHostToDevice));
    g_tc_fp8=false; fp8_block_gemm(Cr,dA,das,dB,dbs,M,N,K,0); CU(cudaDeviceSynchronize());  // oracle
    g_tc_fp8=true;  fp8_block_gemm(Cg,dA,das,dB,dbs,M,N,K,0); CU(cudaDeviceSynchronize());  // GEMV (M=1 branch)
    std::vector<float> cr(N),cg(N); CU(cudaMemcpy(cr.data(),Cr,N*4,cudaMemcpyDeviceToHost)); CU(cudaMemcpy(cg.data(),Cg,N*4,cudaMemcpyDeviceToHost));
    double dot=0,nr=0,ng=0,md=0; for(int i=0;i<N;++i){dot+=cr[i]*cg[i];nr+=cr[i]*cr[i];ng+=cg[i]*cg[i];md=fmax(md,fabs(cr[i]-cg[i]));}
    double cos=dot/(sqrt(nr)*sqrt(ng)+1e-30); bool ok=cos>0.999999 && md<1e-2;
    printf("[fp8_gemv M=1] N=%d K=%d cosine=%.7f maxabs=%.2e -> %s\n",N,K,cos,md,ok?"PASS":"FAIL"); return ok?0:1;
}
