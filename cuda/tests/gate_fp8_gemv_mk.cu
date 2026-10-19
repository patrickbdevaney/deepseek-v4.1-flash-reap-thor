#include "fp8_block_gemm.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#define CU(x) do{cudaError_t e=(x); if(e){fprintf(stderr,"cuda %s\n",cudaGetErrorString(e));exit(1);} }while(0)
extern bool g_tc_fp8;
int main(int c,char**v){ int M=c>1?atoi(v[1]):5,N=2048,K=1024; srand(7);
    std::vector<uint8_t> A((size_t)M*K),B((size_t)N*K); std::vector<float> as((size_t)M*(K/128)),bs((size_t)(N/128)*(K/128));
    for(auto&x:A)x=rand()%0x40; for(auto&x:B)x=rand()%0x40; for(auto&x:as)x=0.5f+0.01f*(rand()%50); for(auto&x:bs)x=0.5f+0.01f*(rand()%50);
    uint8_t*dA,*dB; float*das,*dbs,*Cr,*Cg;
    CU(cudaMalloc(&dA,A.size()));CU(cudaMalloc(&dB,B.size()));CU(cudaMalloc(&das,as.size()*4));CU(cudaMalloc(&dbs,bs.size()*4));CU(cudaMalloc(&Cr,(size_t)M*N*4));CU(cudaMalloc(&Cg,(size_t)M*N*4));
    CU(cudaMemcpy(dA,A.data(),A.size(),cudaMemcpyHostToDevice));CU(cudaMemcpy(dB,B.data(),B.size(),cudaMemcpyHostToDevice));CU(cudaMemcpy(das,as.data(),as.size()*4,cudaMemcpyHostToDevice));CU(cudaMemcpy(dbs,bs.data(),bs.size()*4,cudaMemcpyHostToDevice));
    g_tc_fp8=false; fp8_block_gemm(Cr,dA,das,dB,dbs,M,N,K,0); CU(cudaDeviceSynchronize());
    g_tc_fp8=true;  fp8_block_gemm(Cg,dA,das,dB,dbs,M,N,K,0); CU(cudaDeviceSynchronize());
    std::vector<float> cr((size_t)M*N),cg((size_t)M*N); CU(cudaMemcpy(cr.data(),Cr,cr.size()*4,cudaMemcpyDeviceToHost));CU(cudaMemcpy(cg.data(),Cg,cg.size()*4,cudaMemcpyDeviceToHost));
    double dot=0,nr=0,ng=0,md=0; for(size_t i=0;i<cr.size();++i){dot+=cr[i]*cg[i];nr+=cr[i]*cr[i];ng+=cg[i]*cg[i];md=fmax(md,fabs(cr[i]-cg[i]));}
    double cos=dot/(sqrt(nr)*sqrt(ng)+1e-30); bool ok=cos>0.999999&&md<1e-2;
    printf("[fp8_gemv M=%d] N=%d K=%d cosine=%.7f maxabs=%.2e -> %s\n",M,N,K,cos,md,ok?"PASS":"FAIL"); return ok?0:1; }
