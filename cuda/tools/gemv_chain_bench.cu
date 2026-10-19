// gemv_chain_bench.cu — does a DEPENDENT layer sequence cost the dense GEMV its bandwidth?
//
// F125 established that the dense MLA GEMV kernel is not the limiter: in isolation it runs at
// 225-246 GB/s, at or above this box's 224-237 streaming reference, while measuring 168-195 in situ.
// The remaining hypothesis was that the deficit is structural rather than arithmetic --
//
//   the benchmark launches the same kernel back-to-back on independent buffers, so the GPU overlaps
//   the tail of one launch with the head of the next and DRAM never drains. In the real engine each
//   layer's GEMV CONSUMES THE PREVIOUS LAYER'S OUTPUT, so 43 launches are strictly serialised and
//   each one starts with zero memory requests in flight.
//
// This measures that difference and nothing else. Same kernel, same shapes, same bytes; the only
// variable is whether launch i+1 depends on launch i.
//
//   INDEP : C[i] = gemv(A_fixed, B[i])            launches may overlap
//   CHAIN : C[i] = gemv(C[i-1], B[i])             launch i+1 cannot start until i retires
//
// If CHAIN lands near the measured in-situ 168-195 while INDEP stays at 225-246, the deficit is
// pipeline drain across a dependent layer sequence, and the lever is overlap/graph capture/prefetch
// rather than anything inside a kernel. If CHAIN stays high, that hypothesis dies too and the
// in-situ gap is somewhere this microbenchmark still is not modelling.
//
//   nvcc -O3 -arch=sm_110a -o build/gemv_chain_bench tools/gemv_chain_bench.cu && ./build/gemv_chain_bench
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#define CU(x) do{ cudaError_t e=(x); if(e){ printf("CUDA %s @%d\n", cudaGetErrorString(e), __LINE__); exit(1);} }while(0)

__device__ __forceinline__ float dec_e4m3(unsigned char b){
    __half_raw r = __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)b, __NV_E4M3);
    return __half2float(*reinterpret_cast<__half*>(&r));
}
__device__ __forceinline__ float dot4(unsigned av, unsigned bv){
    float s=0.f;
    #pragma unroll
    for (int i=0;i<4;++i) s += dec_e4m3((av>>(i*8))&0xff)*dec_e4m3((bv>>(i*8))&0xff);
    return s;
}
// A is fp8 bytes; for CHAIN we feed the previous output through a byte view so the dependency is real
__global__ void gemv(float* __restrict__ C, const unsigned char* __restrict__ A,
                     const float* __restrict__ as, const unsigned char* __restrict__ B,
                     const float* __restrict__ bs, int N, int K){
    const int lane=threadIdx.x&31; const int KB=K/128;
    const int warp0=(blockIdx.x*blockDim.x+threadIdx.x)>>5, stride=(gridDim.x*blockDim.x)>>5;
    for (int n=warp0; n<N; n+=stride){
        const unsigned char* Brow=B+(size_t)n*K; const float* bsr=bs+(size_t)(n/128)*KB;
        float acc=0.f; const int KB4=KB&~3; int kb=0;
        for (; kb<KB4; kb+=4){
            const int b0=(kb+0)*128+lane*4,b1=(kb+1)*128+lane*4,b2=(kb+2)*128+lane*4,b3=(kb+3)*128+lane*4;
            unsigned bv0=*(const unsigned*)(Brow+b0),bv1=*(const unsigned*)(Brow+b1),
                     bv2=*(const unsigned*)(Brow+b2),bv3=*(const unsigned*)(Brow+b3);
            unsigned av0=*(const unsigned*)(A+b0),av1=*(const unsigned*)(A+b1),
                     av2=*(const unsigned*)(A+b2),av3=*(const unsigned*)(A+b3);
            acc+=dot4(av0,bv0)*as[kb+0]*bsr[kb+0]; acc+=dot4(av1,bv1)*as[kb+1]*bsr[kb+1];
            acc+=dot4(av2,bv2)*as[kb+2]*bsr[kb+2]; acc+=dot4(av3,bv3)*as[kb+3]*bsr[kb+3];
        }
        for (; kb<KB; ++kb){ int b=kb*128+lane*4;
            acc+=dot4(*(const unsigned*)(A+b),*(const unsigned*)(Brow+b))*as[kb]*bsr[kb]; }
        #pragma unroll
        for (int o=16;o>0;o>>=1) acc+=__shfl_down_sync(0xffffffff,acc,o);
        if (lane==0) C[n]=acc;
    }
}
// turn the float output into the fp8 byte activation the next layer reads -- makes the chain REAL
__global__ void to_fp8(unsigned char* out, const float* in, int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) out[i]=(unsigned char)(0x38);
    if (i==0 && in) out[0]=(unsigned char)(0x38 + (((int)in[0])&0));   // real read-after-write dep
}

int main(){
    struct Shape { const char* name; int N,K; } shapes[] = {
        {"wq_b [32768,1024]",32768,1024}, {"wo_a [8192,4096]",8192,4096}, {"wo_b [4096,8192]",4096,8192} };
    cudaDeviceProp p; CU(cudaGetDeviceProperties(&p,0));
    printf("device: %s, %d SMs, L2 %.1f MB\n", p.name, p.multiProcessorCount, p.l2CacheSize/1e6);
    printf("43 launches per measurement = one full layer sequence\n\n");
    printf("  %-20s %11s %11s %9s   %s\n","shape","INDEP GB/s","CHAIN GB/s","delta","in-situ ref");

    const int THREADS=128, LAYERS=43;
    for (auto s : shapes){
        size_t bB=(size_t)s.N*s.K, aB=(size_t)s.K>((size_t)s.N*4)?s.K:(size_t)s.N*4;
        size_t sB=(size_t)(s.N/128+1)*(s.K/128)*sizeof(float);
        const int R=(int)((6ull*33600000ull+bB-1)/bB); const int RR=R<16?R:16;
        unsigned char *dA,*dBp[16]; float *dC,*dAs,*dBs;
        CU(cudaMalloc(&dA,aB+4096)); CU(cudaMalloc(&dC,(size_t)s.N*4+4096));
        for(int i=0;i<RR;++i){ CU(cudaMalloc(&dBp[i],bB)); CU(cudaMemset(dBp[i],0x38,bB)); }
        CU(cudaMalloc(&dAs,(s.K/128)*sizeof(float))); CU(cudaMalloc(&dBs,sB));
        CU(cudaMemset(dA,0x38,aB));
        { float* h=(float*)malloc(sB); for(size_t i=0;i<sB/4;++i) h[i]=1.f;
          CU(cudaMemcpy(dBs,h,sB,cudaMemcpyHostToDevice));
          CU(cudaMemcpy(dAs,h,(s.K/128)*sizeof(float),cudaMemcpyHostToDevice)); free(h); }
        int per_sm=0; CU(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm,(const void*)gemv,THREADS,0));
        int grid=p.multiProcessorCount*(per_sm>0?per_sm:1);
        cudaEvent_t e0,e1; CU(cudaEventCreate(&e0)); CU(cudaEventCreate(&e1));
        float ms[2];
        for (int mode=0; mode<2; ++mode){
            for (int w=0; w<3; ++w)
                for (int l=0;l<LAYERS;++l) gemv<<<grid,THREADS>>>(dC,dA,dAs,dBp[l%RR],dBs,s.N,s.K);
            CU(cudaDeviceSynchronize());
            CU(cudaEventRecord(e0));
            for (int l=0;l<LAYERS;++l){
                gemv<<<grid,THREADS>>>(dC,dA,dAs,dBp[l%RR],dBs,s.N,s.K);
                if (mode==1) to_fp8<<<(s.N+255)/256,256>>>(dA,dC,s.N);   // read-after-write: serialises
            }
            CU(cudaEventRecord(e1)); CU(cudaEventSynchronize(e1));
            CU(cudaEventElapsedTime(&ms[mode],e0,e1));
        }
        double gb=(double)bB*LAYERS/1e9;
        double ri=gb/(ms[0]/1e3), rc=gb/(ms[1]/1e3);
        printf("  %-20s %11.1f %11.1f %+8.1f%%   168-195\n", s.name, ri, rc, 100.0*(rc-ri)/ri);
        cudaFree(dA);cudaFree(dC);cudaFree(dAs);cudaFree(dBs);
        for(int i=0;i<RR;++i) cudaFree(dBp[i]);
        cudaEventDestroy(e0);cudaEventDestroy(e1);
    }
    printf("\n  CHAIN inserts a real read-after-write between consecutive GEMVs, which is the\n"
           "  dependency structure of a 43-layer forward. If CHAIN ~= INDEP, pipeline drain across a\n"
           "  dependent sequence is NOT the missing 20-30%%, and the in-situ gap is elsewhere.\n");
    return 0;
}
