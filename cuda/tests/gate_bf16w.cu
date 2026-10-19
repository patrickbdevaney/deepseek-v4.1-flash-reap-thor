// gate_bf16w.cu — gemm_bf16w (native BF16 weights) vs gemm_fp32 on the SAME weights dequantised to
// f32, which is exactly what the engine did before Finding 26. BF16->F32 is lossless, so the only
// admissible difference is fp32 accumulation reassociation (the bf16x2 path pairs elements).
//   build: nvcc -O3 -std=c++17 -arch=sm_110a -I include tests/gate_bf16w.cu kernels/compressor.cu \
//            kernels/dscratch.cu -o build/gate_bf16w
#include "compressor.h"
#include "dscratch.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <cmath>
#include <cstdlib>
#define CU(x) do{cudaError_t e=(x); if(e){printf("cuda %s\n",cudaGetErrorString(e));return 2;} }while(0)

__global__ void k_bf2f(float* o, const __nv_bfloat16* i, size_t n){
    size_t k=blockIdx.x*(size_t)blockDim.x+threadIdx.x; if(k<n) o[k]=__bfloat162float(i[k]); }

int main(){
    arena_init((size_t)64<<20);
    int fail=0;
    struct Case{ const char* n; int M,N,K; };
    // the shapes that actually matter: lm_head at decode (M=1) and verify (M=5), and markov.
    const Case cs[] = {{"lm_head M=1",1,129280,4096},{"lm_head M=5",5,129280,4096},
                       {"markov  M=1",1,129280,256},{"markov  M=5",5,129280,256}};
    for(const auto& c : cs){
        std::vector<__nv_bfloat16> hb((size_t)c.N*c.K);
        std::vector<float> ha((size_t)c.M*c.K);
        for(size_t i=0;i<hb.size();++i) hb[i]=__float2bfloat16(((float)rand()/RAND_MAX-0.5f)*0.1f);
        for(size_t i=0;i<ha.size();++i) ha[i]=((float)rand()/RAND_MAX-0.5f);
        __nv_bfloat16* B; float *Bf,*A,*C1,*C2;
        CU(cudaMalloc(&B,hb.size()*2)); CU(cudaMalloc(&Bf,hb.size()*4));
        CU(cudaMalloc(&A,ha.size()*4)); CU(cudaMalloc(&C1,(size_t)c.M*c.N*4)); CU(cudaMalloc(&C2,(size_t)c.M*c.N*4));
        CU(cudaMemcpy(B,hb.data(),hb.size()*2,cudaMemcpyHostToDevice));
        CU(cudaMemcpy(A,ha.data(),ha.size()*4,cudaMemcpyHostToDevice));
        k_bf2f<<<(hb.size()+255)/256,256>>>(Bf,B,hb.size());          // what Loader::bf16 used to do
        CU(cudaDeviceSynchronize());
        gemm_fp32 (C1,A,Bf,c.M,c.N,c.K,0);                            // old path
        gemm_bf16w(C2,A,(const void*)B,c.M,c.N,c.K,0);                // new path
        CU(cudaDeviceSynchronize());
        std::vector<float> r1((size_t)c.M*c.N), r2((size_t)c.M*c.N);
        CU(cudaMemcpy(r1.data(),C1,r1.size()*4,cudaMemcpyDeviceToHost));
        CU(cudaMemcpy(r2.data(),C2,r2.size()*4,cudaMemcpyDeviceToHost));
        double dot=0,n1=0,n2=0,mx=0,amx=0; int argmax1=0,argmax2=0;
        for(size_t i=0;i<r1.size();++i){
            dot+=(double)r1[i]*r2[i]; n1+=(double)r1[i]*r1[i]; n2+=(double)r2[i]*r2[i];
            mx=fmax(mx,fabs(r1[i]-r2[i])); amx=fmax(amx,fabs((double)r1[i]));
            if(r1[i]>r1[argmax1]) argmax1=(int)i;
            if(r2[i]>r2[argmax2]) argmax2=(int)i;
        }
        const double cos=dot/sqrt(n1*n2), rel=mx/amx;
        const bool ok = cos>0.9999999 && rel<1e-5 && argmax1==argmax2;
        printf("[%-12s] cosine=%.9f max_abs=%.3e rel=%.2e argmax %s -> %s\n",
               c.n, cos, mx, rel, argmax1==argmax2?"MATCH":"DIFFER", ok?"PASS":"FAIL");
        if(!ok) ++fail;
        cudaFree(B);cudaFree(Bf);cudaFree(A);cudaFree(C1);cudaFree(C2);
    }
    // gemm_fp32's own M=K path (Finding 43). It is reached by compressor_emit_group at M=2*ratio,
    // and by gemm_bf16w's reference above, so a silent error here would be invisible in the
    // comparison it feeds. Chunked in 8s, so M=8 (one full chunk), M=13 (chunk + 5-row tail) and
    // M=128 (sixteen chunks, the ratio-128 layers) are the three shapes that fail differently.
    //
    // LADDER 1.12 widened that path to an (MM,NN) warp tile, so the claim is no longer about ONE
    // kernel: it is that EVERY tile the dispatcher can select produces the identical bytes. The
    // whole argument for widening the tile is that it changes which warp owns an output and how
    // many outputs it owns, and NOT the order of any single dot product -- so max|diff| == 0 is the
    // statement, not a tolerance. `8x0` is the pre-1.12 host-side chunk loop and is swept too, so
    // the before-arm of 1.12's A/B is gated as well as the after-arm. N=254 and M=13 exist to hit
    // the tile tails (N % (NN*wpb) != 0, M % MM != 0), which is where a guarded store gets it wrong.
    {
        printf("\n-- gemm_fp32 M=K vs the warp-per-output-element path, every (MM,NN) tile --\n");
        const int K_GATE = 4096;
        const int TL[][2] = {{8,0},{8,1},{8,2},{8,4},{4,4},{4,8},{2,8},{16,1},{16,2},{32,1},
                             {6,4},{4,6},{6,6},{5,5},{4,2},{2,4},{4,1},{2,2}};
        const int NTL = (int)(sizeof(TL)/sizeof(TL[0]));
        for (int N : {1024, 254}) {
        std::vector<float> hb((size_t)N*K_GATE); for(auto&x:hb) x=(rand()%2000-1000)/1000.f;
        float* dB; CU(cudaMalloc(&dB,hb.size()*4)); CU(cudaMemcpy(dB,hb.data(),hb.size()*4,cudaMemcpyHostToDevice));
        for (int M : {1,2,5,8,13,16,128}) {
            std::vector<float> ha((size_t)M*K_GATE); for(auto&x:ha) x=(rand()%2000-1000)/1000.f;
            float *dA,*C1,*C2; CU(cudaMalloc(&dA,ha.size()*4));
            CU(cudaMemcpy(dA,ha.data(),ha.size()*4,cudaMemcpyHostToDevice));
            CU(cudaMalloc(&C1,(size_t)M*N*4)); CU(cudaMalloc(&C2,(size_t)M*N*4));
            setenv("NO_FP32MK","1",1); gemm_fp32(C1,dA,dB,M,N,K_GATE,0);
            unsetenv("NO_FP32MK");
            std::vector<float> v1((size_t)M*N), v2((size_t)M*N);
            CU(cudaDeviceSynchronize());
            CU(cudaMemcpy(v1.data(),C1,(size_t)M*N*4,cudaMemcpyDeviceToHost));
            double worst=0; int bad=0;
            for (int t=0;t<NTL;++t){
                gemm_fp32_set_tile(TL[t][0],TL[t][1]);
                CU(cudaMemset(C2,0,(size_t)M*N*4));
                gemm_fp32(C2,dA,dB,M,N,K_GATE,0);
                CU(cudaDeviceSynchronize());
                CU(cudaMemcpy(v2.data(),C2,(size_t)M*N*4,cudaMemcpyDeviceToHost));
                double md=0; for(size_t i=0;i<v1.size();++i) md=fmax(md,fabs((double)v1[i]-v2[i]));
                if(md!=0.0){ ++bad; printf("   tile %dx%d: max|diff| = %.3e\n", TL[t][0],TL[t][1],md); }
                worst=fmax(worst,md);
            }
            gemm_fp32_set_tile(4,4);   // restore the shipped default
            printf("[gemm_fp32 M=%-3d N=%-4d] %2d tiles, worst max|diff| = %.3e -> %s\n",
                   M, N, NTL, worst, bad?"FAIL":"PASS");
            if(bad) ++fail;
            cudaFree(dA); cudaFree(C1); cudaFree(C2);
        }
        cudaFree(dB);
        }
    }
    printf("\nGate BF16W: %s\n", fail?"FAIL":"PASS");
    return fail?1:0;
}
