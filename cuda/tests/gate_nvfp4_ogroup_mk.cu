// gate_nvfp4_ogroup_mk.cu — does the NVFP4 ogroup M=K kernel agree with the M=1 kernel?
//
// The M=1 ogroup path passes LOSSLESS end to end, so it is the reference. The M=K path changed the
// emitted tokens and failed LOSSLESS at token 3. Running M=1 M separate times must, by definition,
// equal one M=K call -- they are the same function. This isolates the disagreement to a kernel
// instead of to a 2-minute engine run.
#include "nvfp4_dense.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#define CU(x) do{cudaError_t e=(x); if(e){printf("CUDA %s @%d\n",cudaGetErrorString(e),__LINE__);return 1;}}while(0)

int main(){
    const int G=8, R=1024, Kd=4096, M=5;          // the real wo_a shape, and the spec verify width
    const size_t rows=(size_t)G*R, wbytes=rows*(Kd/2), sbytes=rows*(Kd/16);
    std::vector<uint8_t> hP(wbytes), hS(sbytes);
    std::vector<float> hO((size_t)M*G*Kd);
    srand(1234);
    for(auto& v: hP) v=(uint8_t)(rand()&0xff);     // arbitrary E2M1 code pairs
    for(auto& v: hS) v=(uint8_t)(0x30 + (rand()&0x7));   // small positive e4m3 scales
    for(auto& v: hO) v=(float)((rand()%2001)-1000)/1000.f;

    uint8_t *dP,*dS; float *dO,*dRef,*dMK;
    CU(cudaMalloc(&dP,wbytes)); CU(cudaMalloc(&dS,sbytes));
    CU(cudaMalloc(&dO,hO.size()*4));
    CU(cudaMalloc(&dRef,(size_t)M*rows*4)); CU(cudaMalloc(&dMK,(size_t)M*rows*4));
    CU(cudaMemcpy(dP,hP.data(),wbytes,cudaMemcpyHostToDevice));
    CU(cudaMemcpy(dS,hS.data(),sbytes,cudaMemcpyHostToDevice));
    CU(cudaMemcpy(dO,hO.data(),hO.size()*4,cudaMemcpyHostToDevice));

    // register under a dummy fp8 pointer so the launchers' lookup finds it
    NvFp4Weight w{}; w.packed=dP; w.scale=dS; w.global=1.0f; w.N=(int)rows; w.K=Kd;
    const uint8_t* key=(const uint8_t*)0x1000;
    nvfp4_register(key,w); nvfp4_force_enable(true);

    // REFERENCE: the M=1 kernel, once per activation row (its activation base is o + g*Kd, so row m
    // starts at o + m*G*Kd -- exactly the slice the M=K kernel should read for that m)
    for(int m=0;m<M;++m)
        if(!nvfp4_ogroup_gemv(dRef+(size_t)m*rows, dO+(size_t)m*G*Kd, key, G,R,Kd, 0)){
            printf("M=1 launcher refused\n"); return 1; }
    CU(cudaDeviceSynchronize());
    if(!nvfp4_ogroup_mk(dMK, dO, key, M, G,R,Kd, 0)){ printf("M=K launcher refused\n"); return 1; }
    CU(cudaDeviceSynchronize());

    std::vector<float> a((size_t)M*rows), b((size_t)M*rows);
    CU(cudaMemcpy(a.data(),dRef,a.size()*4,cudaMemcpyDeviceToHost));
    CU(cudaMemcpy(b.data(),dMK ,b.size()*4,cudaMemcpyDeviceToHost));
    double num=0,da=0,db=0; int bad=0, firstm=-1,firstr=-1;
    for(size_t i=0;i<a.size();++i){
        num+=(double)a[i]*b[i]; da+=(double)a[i]*a[i]; db+=(double)b[i]*b[i];
        if(fabs(a[i]-b[i])>1e-3*(fabs(a[i])+1e-6)){ if(!bad){firstm=(int)(i/rows);firstr=(int)(i%rows);} ++bad; }
    }
    double cos = num/(sqrt(da)*sqrt(db)+1e-30);
    printf("  M=1 x %d  vs  M=K : cosine %.8f, %d/%zu elements differ\n", M, cos, bad, a.size());
    if(bad) printf("  first mismatch at m=%d row=%d : ref %.6f  mk %.6f\n", firstm, firstr,
                   a[(size_t)firstm*rows+firstr], b[(size_t)firstm*rows+firstr]);
    // per-m breakdown localises a layout error immediately
    for(int m=0;m<M;++m){ int c=0;
        for(size_t r=0;r<rows;++r){ size_t i=(size_t)m*rows+r;
            if(fabs(a[i]-b[i])>1e-3*(fabs(a[i])+1e-6)) ++c; }
        printf("    m=%d: %d/%zu differ\n", m, c, rows); }
    printf(cos>0.9999 && bad==0 ? "  GATE PASS\n" : "  GATE FAIL\n");
    return (cos>0.9999 && bad==0)?0:1;
}
