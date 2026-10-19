// gemv_accum_bench.cu — is the accumulator dependency chain what limits the dense MLA GEMV?
//
// THE QUESTION, and nothing else. `fp8_gemv_m1_kernel` already fixed its ILP problem: it unrolls by
// 4 and issues 8 loads before consuming any. Yet the three big MLA weights measure 168-195 GB/s
// against a streaming benchmark that sustains 224-237 on this box, with 25-205 warps/SM, so
// parallelism is not the limiter either. The remaining suspect is the accumulator:
//
//     acc += dot4(av0,bv0) * as[kb+0] * bsr[kb+0];      four STRICTLY SERIAL FP adds per
//     acc += dot4(av1,bv1) * as[kb+1] * bsr[kb+1];      iteration, each waiting on its dot4,
//     acc += dot4(av2,bv2) * as[kb+2] * bsr[kb+2];      and dot4 is itself a chain of 4 adds
//     acc += dot4(av3,bv3) * as[kb+3] * bsr[kb+3];
//
// This is a THROWAWAY. It answers one question -- does breaking the chain reach the streaming rate,
// or does it stall at ~195 anyway -- before anyone touches a shipped kernel. If it stalls, the
// hypothesis dies here for the cost of one afternoon and the limiter is somewhere else.
//
// WHY IT CANNOT SIMPLY BE ADOPTED IF IT WINS. Four accumulators REASSOCIATE the sum. Floating-point
// addition is not associative, so the result changes in the last ulp, and this engine gates on a
// PyTorch oracle and on `LOSSLESS: first 8 tokens match base AR`. A last-ulp change can flip an
// argmax and therefore a token. So this measures the CEILING; what to trade for it is a separate
// decision (see wiki/dense-mla-gemv.md -- the draft-path-only option is the interesting one).
//
// L2 TRAP, found the hard way on the first run. Thor's L2 is **33.6 MB** and these weights are
// 33.6 MB, so a loop that re-reads ONE buffer measures L2, not DRAM: the first version reported
// 227-242 GB/s for kernels that measure 168-195 in situ. The fix is a POOL of R weight copies cycled
// across reps, so the working set is R x 33.6 MB and every rep starts cold in DRAM. Any GEMV
// benchmark on this box that does not do this is measuring the wrong memory.
//
//   nvcc -O3 -arch=sm_110a -o build/gemv_accum_bench tools/gemv_accum_bench.cu
//   ./build/gemv_accum_bench
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
    float s = 0.f;
    #pragma unroll
    for (int i = 0; i < 4; ++i) s += dec_e4m3((av>>(i*8))&0xff) * dec_e4m3((bv>>(i*8))&0xff);
    return s;
}

// ---- SHIPPED SHAPE: one accumulator, order 0,1,2,3,... preserved (bit-exact) -------------------
__global__ void gemv_1acc(float* __restrict__ C, const unsigned char* __restrict__ A,
                          const float* __restrict__ as, const unsigned char* __restrict__ B,
                          const float* __restrict__ bs, int N, int K){
    const int lane = threadIdx.x & 31; const int KB = K/128;
    const int warp0 = (blockIdx.x*blockDim.x + threadIdx.x) >> 5;
    const int stride = (gridDim.x*blockDim.x) >> 5;
    for (int n = warp0; n < N; n += stride){
        const unsigned char* Brow = B + (size_t)n*K; const float* bsr = bs + (size_t)(n/128)*KB;
        float acc = 0.f;
        const int KB4 = KB & ~3; int kb = 0;
        for (; kb < KB4; kb += 4){
            const int b0=(kb+0)*128+lane*4, b1=(kb+1)*128+lane*4, b2=(kb+2)*128+lane*4, b3=(kb+3)*128+lane*4;
            unsigned bv0=*(const unsigned*)(Brow+b0), bv1=*(const unsigned*)(Brow+b1),
                     bv2=*(const unsigned*)(Brow+b2), bv3=*(const unsigned*)(Brow+b3);
            unsigned av0=*(const unsigned*)(A+b0), av1=*(const unsigned*)(A+b1),
                     av2=*(const unsigned*)(A+b2), av3=*(const unsigned*)(A+b3);
            acc += dot4(av0,bv0)*as[kb+0]*bsr[kb+0];
            acc += dot4(av1,bv1)*as[kb+1]*bsr[kb+1];
            acc += dot4(av2,bv2)*as[kb+2]*bsr[kb+2];
            acc += dot4(av3,bv3)*as[kb+3]*bsr[kb+3];
        }
        for (; kb < KB; ++kb){ int b=kb*128+lane*4;
            acc += dot4(*(const unsigned*)(A+b), *(const unsigned*)(Brow+b))*as[kb]*bsr[kb]; }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffff, acc, o);
        if (lane == 0) C[n] = acc;
    }
}

// ---- CANDIDATE: four accumulators, chain broken (REASSOCIATED, not bit-exact) ------------------
__global__ void gemv_4acc(float* __restrict__ C, const unsigned char* __restrict__ A,
                          const float* __restrict__ as, const unsigned char* __restrict__ B,
                          const float* __restrict__ bs, int N, int K){
    const int lane = threadIdx.x & 31; const int KB = K/128;
    const int warp0 = (blockIdx.x*blockDim.x + threadIdx.x) >> 5;
    const int stride = (gridDim.x*blockDim.x) >> 5;
    for (int n = warp0; n < N; n += stride){
        const unsigned char* Brow = B + (size_t)n*K; const float* bsr = bs + (size_t)(n/128)*KB;
        float a0=0.f, a1=0.f, a2=0.f, a3=0.f;
        const int KB4 = KB & ~3; int kb = 0;
        for (; kb < KB4; kb += 4){
            const int b0=(kb+0)*128+lane*4, b1=(kb+1)*128+lane*4, b2=(kb+2)*128+lane*4, b3=(kb+3)*128+lane*4;
            unsigned bv0=*(const unsigned*)(Brow+b0), bv1=*(const unsigned*)(Brow+b1),
                     bv2=*(const unsigned*)(Brow+b2), bv3=*(const unsigned*)(Brow+b3);
            unsigned av0=*(const unsigned*)(A+b0), av1=*(const unsigned*)(A+b1),
                     av2=*(const unsigned*)(A+b2), av3=*(const unsigned*)(A+b3);
            a0 += dot4(av0,bv0)*as[kb+0]*bsr[kb+0];      // four INDEPENDENT chains
            a1 += dot4(av1,bv1)*as[kb+1]*bsr[kb+1];
            a2 += dot4(av2,bv2)*as[kb+2]*bsr[kb+2];
            a3 += dot4(av3,bv3)*as[kb+3]*bsr[kb+3];
        }
        float acc = (a0+a1)+(a2+a3);
        for (; kb < KB; ++kb){ int b=kb*128+lane*4;
            acc += dot4(*(const unsigned*)(A+b), *(const unsigned*)(Brow+b))*as[kb]*bsr[kb]; }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffff, acc, o);
        if (lane == 0) C[n] = acc;
    }
}

struct Shape { const char* name; int N, K; };

int main(){
    // the three that carry 94 % of the MLA bytes, plus the parallelism-starved one for contrast
    Shape shapes[] = { {"wq_b  [32768,1024]",32768,1024}, {"wo_a  [8192,4096]",8192,4096},
                       {"wo_b  [4096,8192]",4096,8192},   {"wq_a  [1024,4096]",1024,4096} };
    int dev=0; cudaDeviceProp p; CU(cudaGetDeviceProperties(&p,dev));
    printf("device: %s, %d SMs, L2 %.1f MB\n", p.name, p.multiProcessorCount, p.l2CacheSize/1e6);
    printf("weights are cycled over a pool > L2 so every rep starts cold in DRAM\n\n");
    printf("  %-22s %7s %10s %10s %9s   %s\n","shape","warps/SM","1acc GB/s","4acc GB/s","delta","verdict");

    const int THREADS=128, REPS=30;
    for (Shape s : shapes){
        size_t bBytes=(size_t)s.N*s.K, aBytes=s.K, sBytes=(size_t)(s.N/128+1)*(s.K/128)*sizeof(float);
        // POOL: enough copies that R*bBytes comfortably exceeds the 33.6 MB L2
        const int R = (int)((6ull*33600000ull + bBytes - 1) / bBytes);
        unsigned char *dA,*dBp[16]; float *dC,*dAs,*dBs;
        CU(cudaMalloc(&dA,aBytes)); CU(cudaMalloc(&dC,(size_t)s.N*4));
        for (int i=0;i<R && i<16;++i){ CU(cudaMalloc(&dBp[i],bBytes)); CU(cudaMemset(dBp[i],0x38,bBytes)); }
        unsigned char* dB = dBp[0];
        CU(cudaMalloc(&dAs,(s.K/128)*sizeof(float))); CU(cudaMalloc(&dBs,sBytes));
        CU(cudaMemset(dA,0x38,aBytes));   // e4m3 ~0.5
        // scales = 1.0f
        { float* h=(float*)malloc(sBytes); for(size_t i=0;i<sBytes/4;++i) h[i]=1.f;
          CU(cudaMemcpy(dBs,h,sBytes,cudaMemcpyHostToDevice));
          CU(cudaMemcpy(dAs,h,(s.K/128)*sizeof(float),cudaMemcpyHostToDevice)); free(h); }

        int per_sm=0; CU(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm,(const void*)gemv_1acc,THREADS,0));
        int grid = p.multiProcessorCount * (per_sm>0?per_sm:1);
        double warps_per_sm = (double)(s.N/8) / p.multiProcessorCount;

        cudaEvent_t e0,e1; CU(cudaEventCreate(&e0)); CU(cudaEventCreate(&e1));
        float ms[2];
        for (int variant=0; variant<2; ++variant){
            for (int w=0; w<5; ++w)  // warm-up, discarded (F120: the first run of a batch is slow)
                (variant? gemv_4acc : gemv_1acc)<<<grid,THREADS>>>(dC,dA,dAs,dBp[w%(R<16?R:16)],dBs,s.N,s.K);
            CU(cudaDeviceSynchronize());
            CU(cudaEventRecord(e0));
            for (int r=0; r<REPS; ++r)   // cycle the pool: every rep starts cold in DRAM
                (variant? gemv_4acc : gemv_1acc)<<<grid,THREADS>>>(dC,dA,dAs,dBp[r%(R<16?R:16)],dBs,s.N,s.K);
            CU(cudaEventRecord(e1)); CU(cudaEventSynchronize(e1));
            CU(cudaEventElapsedTime(&ms[variant],e0,e1));
        }
        double gb = (double)bBytes/1e9;
        double r1 = gb*REPS/(ms[0]/1e3), r4 = gb*REPS/(ms[1]/1e3);
        const char* v = (r4 > r1*1.05) ? "chain WAS limiting" : "chain is NOT the limiter";
        printf("  %-22s %7.1f %10.1f %10.1f %+8.1f%%   %s\n", s.name, warps_per_sm, r1, r4,
               100.0*(r4-r1)/r1, v);
        cudaFree(dA);cudaFree(dC);cudaFree(dAs);cudaFree(dBs);
        for (int i=0;i<R && i<16;++i) cudaFree(dBp[i]);
        cudaEventDestroy(e0);cudaEventDestroy(e1);
    }
    printf("\n  reference: streaming benchmark on this box sustains 224-237 GB/s at ILP>=2;\n"
           "  the shipped kernel measures 168-195 on these shapes.\n"
           "  If 4acc does not approach 224+, the accumulator chain is not what costs the gap.\n");
    return 0;
}
