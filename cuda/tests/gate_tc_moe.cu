// gate_tc_moe.cu — validate tc_fp4_gemm (Marlin TC) vs fp4_gemm (gated oracle) on identical bytes.
// Both decode the same fp8/fp4 bytes + scales; TC uses fp16 acts so expect cosine>0.999 (not bit-exact).
#include "moe.h"
#include <cuda_runtime.h>
#include <vector>
#include <cstdio>
#include <cmath>
#include <cstdlib>
void tc_fp4_gemm(float*, const uint8_t*, const float*, const uint8_t*, const float*, int, int, int, cudaStream_t);
#define CU(x) do{cudaError_t e=(x); if(e){fprintf(stderr,"cuda %s\n",cudaGetErrorString(e));exit(1);} }while(0)

int main(int argc, char** argv){
    int M = argc>1?atoi(argv[1]):8, N=2048, K=4096;          // decode-ish M; MoE w1 shape [inter,dim]
    srand(1234);
    std::vector<uint8_t> A(M*K), B((size_t)N*(K/2));
    std::vector<float> as((size_t)M*(K/128)), bs((size_t)N*(K/32));
    for(auto& v:A) v=rand()%0x40;                            // fp8 e4m3: sign 0, exp 0..7 -> finite (avoid 0x7f NaN)
    for(auto& v:B) v=rand()&0xff;                            // fp4 packed
    for(auto& v:as) v=0.5f+0.01f*(rand()%50);
    for(auto& v:bs) v=0.5f+0.01f*(rand()%50);
    uint8_t *dA,*dB; float *das,*dbs,*Cr,*Ct;
    CU(cudaMalloc(&dA,A.size())); CU(cudaMalloc(&dB,B.size())); CU(cudaMalloc(&das,as.size()*4)); CU(cudaMalloc(&dbs,bs.size()*4));
    CU(cudaMalloc(&Cr,(size_t)M*N*4)); CU(cudaMalloc(&Ct,(size_t)M*N*4));
    CU(cudaMemcpy(dA,A.data(),A.size(),cudaMemcpyHostToDevice)); CU(cudaMemcpy(dB,B.data(),B.size(),cudaMemcpyHostToDevice));
    CU(cudaMemcpy(das,as.data(),as.size()*4,cudaMemcpyHostToDevice)); CU(cudaMemcpy(dbs,bs.data(),bs.size()*4,cudaMemcpyHostToDevice));

    fp4_gemm(Cr, dA, das, dB, dbs, M, N, K, 0); CU(cudaDeviceSynchronize());
    tc_fp4_gemm(Ct, dA, das, dB, dbs, M, N, K, 0); CU(cudaDeviceSynchronize());

    std::vector<float> cr((size_t)M*N), ct((size_t)M*N);
    CU(cudaMemcpy(cr.data(),Cr,cr.size()*4,cudaMemcpyDeviceToHost)); CU(cudaMemcpy(ct.data(),Ct,ct.size()*4,cudaMemcpyDeviceToHost));
    double dot=0,nr=0,nt=0,sd=0,sr=0,mx=0,ma=0;
    for(size_t i=0;i<cr.size();++i){ double r=cr[i],t=ct[i]; dot+=r*t; nr+=r*r; nt+=t*t; sd+=(r-t)*(r-t); sr+=r*r; mx=fmax(mx,fabs(r)); ma=fmax(ma,fabs(r-t)); }
    double cosine=dot/(sqrt(nr)*sqrt(nt)+1e-30), rms=sqrt(sd/sr), absr=ma/(mx+1e-30);
    bool ok = cosine>0.999 && rms<3e-2;
    printf("[tc_moe W4A8] M=%d N=%d K=%d  cosine=%.6f rms_rel=%.5f max_abs/|c|max=%.5f -> %s\n",
           M,N,K,cosine,rms,absr, ok?"PASS":"FAIL");
    // --- REPACK-AT-LOAD (pre-packed) variant: repack B in place, run tc_fp4_gemm_pp with ORIGINAL scale, vs oracle ---
    extern void tc_repack_weight_inplace(uint8_t*, int, int, uint8_t*, cudaStream_t);
    extern void tc_fp4_gemm_pp(float*, const uint8_t*, const float*, const uint8_t*, const float*, int,int,int, cudaStream_t);
    uint8_t *dBp,*tmp; CU(cudaMalloc(&dBp,B.size())); CU(cudaMalloc(&tmp,B.size()));
    CU(cudaMemcpy(dBp,dB,B.size(),cudaMemcpyDeviceToDevice));
    tc_repack_weight_inplace(dBp,N,K,tmp,0); CU(cudaDeviceSynchronize());   // in-place, same as load-time
    float* Ctp; CU(cudaMalloc(&Ctp,(size_t)M*N*4));
    tc_fp4_gemm_pp(Ctp,dA,das,dBp,dbs,M,N,K,0); CU(cudaDeviceSynchronize());
    std::vector<float> ctp((size_t)M*N); CU(cudaMemcpy(ctp.data(),Ctp,ctp.size()*4,cudaMemcpyDeviceToHost));
    double dp=0,np=0,nq=0; for(size_t i=0;i<cr.size();++i){ dp+=cr[i]*ctp[i]; np+=cr[i]*cr[i]; nq+=ctp[i]*ctp[i]; }
    double cospp=dp/(sqrt(np)*sqrt(nq)+1e-30); bool okpp=cospp>0.999; ok=ok&&okpp;
    printf("[tc_moe PP repack-at-load] cosine vs fp4_gemm oracle=%.6f -> %s\n", cospp, okpp?"PASS":"FAIL");
    // funnel-shift path: repack at an UNALIGNED (off=8) pointer, verify aligned coalesced loads still correct
    uint8_t* dBu; CU(cudaMalloc(&dBu,B.size()+32)); uint8_t* Bu=dBu+8;
    CU(cudaMemcpy(Bu,dB,B.size(),cudaMemcpyDeviceToDevice));
    tc_repack_weight_inplace(Bu,N,K,tmp,0); CU(cudaDeviceSynchronize());
    float* Ctu; CU(cudaMalloc(&Ctu,(size_t)M*N*4));
    tc_fp4_gemm_pp(Ctu,dA,das,Bu,dbs,M,N,K,0); CU(cudaDeviceSynchronize());
    std::vector<float> ctu((size_t)M*N); CU(cudaMemcpy(ctu.data(),Ctu,ctu.size()*4,cudaMemcpyDeviceToHost));
    double du=0,nu1=0,nu2=0; for(size_t i=0;i<cr.size();++i){du+=cr[i]*ctu[i];nu1+=cr[i]*cr[i];nu2+=ctu[i]*ctu[i];}
    double cosu=du/(sqrt(nu1)*sqrt(nu2)+1e-30); bool oku=cosu>0.999; ok=ok&&oku;
    printf("[tc_moe PP funnel off=8] cosine vs oracle=%.6f -> %s\n", cosu, oku?"PASS":"FAIL");
    cudaFree(dBp);cudaFree(tmp);cudaFree(Ctp);cudaFree(dBu);cudaFree(Ctu);
    // --- A/B timing (full calls; tc includes per-call repack — caching that is a further win) ---
    int IT=30; cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
    for(int i=0;i<3;++i){ fp4_gemm(Cr,dA,das,dB,dbs,M,N,K,0); tc_fp4_gemm(Ct,dA,das,dB,dbs,M,N,K,0);} CU(cudaDeviceSynchronize());
    cudaEventRecord(a); for(int i=0;i<IT;++i) fp4_gemm(Cr,dA,das,dB,dbs,M,N,K,0); cudaEventRecord(b); cudaEventSynchronize(b);
    float t_ref=0; cudaEventElapsedTime(&t_ref,a,b);
    cudaEventRecord(a); for(int i=0;i<IT;++i) tc_fp4_gemm(Ct,dA,das,dB,dbs,M,N,K,0); cudaEventRecord(b); cudaEventSynchronize(b);
    float t_tc=0; cudaEventElapsedTime(&t_tc,a,b);
    printf("[tc_moe A/B] fp4_gemm %.3f ms/call  |  tc_fp4_gemm %.3f ms/call  -> %.2fx %s\n",
           t_ref/IT, t_tc/IT, t_ref/t_tc, t_ref>t_tc?"FASTER":"SLOWER (repack overhead — cache it)");
    return ok?0:1;
}
