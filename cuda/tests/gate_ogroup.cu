#include <cuda_runtime.h>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#define CU(x) do{auto e=(x); if(e){printf("ERR %d\n",(int)e);return 1;}}while(0)
void ogroup_gemm(float*, const float*, const float*, int,int,int,int, cudaStream_t);
extern bool g_tc_ogroup;
int main(){
    int bs=8,G=8,R=1024,Kd=4096;
    std::vector<float> o((size_t)bs*G*Kd), w((size_t)G*R*Kd);
    srand(7); for(auto&x:o)x=(rand()%2000-1000)/1000.f; for(auto&x:w)x=(rand()%2000-1000)/1000.f;
    float *doo,*dw,*Cr,*Ct;
    CU(cudaMalloc(&doo,o.size()*4)); CU(cudaMalloc(&dw,w.size()*4));
    CU(cudaMalloc(&Cr,(size_t)bs*G*R*4)); CU(cudaMalloc(&Ct,(size_t)bs*G*R*4));
    CU(cudaMemcpy(doo,o.data(),o.size()*4,cudaMemcpyHostToDevice));
    CU(cudaMemcpy(dw,w.data(),w.size()*4,cudaMemcpyHostToDevice));
    g_tc_ogroup=false; ogroup_gemm(Cr,doo,dw,bs,G,R,Kd,0); CU(cudaDeviceSynchronize());
    g_tc_ogroup=true;  ogroup_gemm(Ct,doo,dw,bs,G,R,Kd,0); CU(cudaDeviceSynchronize());
    std::vector<float> cr((size_t)bs*G*R), ct((size_t)bs*G*R);
    CU(cudaMemcpy(cr.data(),Cr,cr.size()*4,cudaMemcpyDeviceToHost));
    CU(cudaMemcpy(ct.data(),Ct,ct.size()*4,cudaMemcpyDeviceToHost));
    double d=0,a=0,b=0,mx=0,ma=0; for(size_t i=0;i<cr.size();++i){d+=cr[i]*ct[i];a+=cr[i]*cr[i];b+=ct[i]*ct[i];mx=fmax(mx,fabs(cr[i]));ma=fmax(ma,fabs(cr[i]-ct[i]));}
    double cos=d/(sqrt(a)*sqrt(b)+1e-30);
    printf("[tc_ogroup] bs=%d G=%d R=%d Kd=%d  cosine=%.6f max_abs/|c|max=%.5f -> %s\n",bs,G,R,Kd,cos,ma/(mx+1e-30),(cos>0.999)?"PASS":"FAIL");
    return cos>0.999?0:1;
}
