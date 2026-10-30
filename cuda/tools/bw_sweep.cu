// bw_sweep.cu — M2: what read bandwidth is actually reachable on Thor's unified pool?
// bw_probe.cu measures one shape (float4, 16 blocks/SM, 1 load in flight per thread) and gave the
// 228 GB/s the ledger uses. This sweeps the knobs that set bytes in flight:
//   ld   : 128-bit loads with U independent loads per thread per iteration (U = 1, 2, 4, 8)
//   nc   : same, ld.global.nc.L1::no_allocate (streaming hint)
//   v8   : 256-bit ld.global.v8.f32 (sm_100+), U = 1, 2, 4
//   tma  : cp.async.bulk global->shared, CHUNK bytes per stage, S stages per block
// over blocks/SM in {2,4,8,16,32}, and prints the best GB/s per method.
//   nvcc -O3 -arch=sm_110a tools/bw_sweep.cu -o build/bw_sweep && build/bw_sweep [MiB=4096] [iters=10]
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

#define CU(x) do{ cudaError_t e=(x); if(e){ printf("CUDA %s @%d\n", cudaGetErrorString(e), __LINE__); return 1; } }while(0)

__device__ __forceinline__ float4 ld_nc(const float4* p){
    float4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.f32 {%0,%1,%2,%3}, [%4];"
                 : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"(p));
    return v;
}

template<int U, bool NC>
__global__ void k_ld(const float4* __restrict__ p, size_t n4, float* out){
    float acc = 0.f;
    const size_t T = (size_t)gridDim.x * blockDim.x;
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for(; i + (U-1)*T < n4; i += U*T){
        float4 v[U];
        #pragma unroll
        for(int u = 0; u < U; u++) v[u] = NC ? ld_nc(p + i + u*T) : p[i + u*T];
        #pragma unroll
        for(int u = 0; u < U; u++) acc += v[u].x + v[u].y + v[u].z + v[u].w;
    }
    for(; i < n4; i += T){ float4 v = p[i]; acc += v.x + v.y + v.z + v.w; }
    if(acc == 1234.5678f) out[0] = acc;
}

template<int U>
__global__ void k_v8(const float* __restrict__ p, size_t n8, float* out){
    float acc = 0.f;
    const size_t T = (size_t)gridDim.x * blockDim.x;
    for(size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i + (U-1)*T < n8; i += U*T){
        float r[U][8];
        #pragma unroll
        for(int u = 0; u < U; u++){
            const float* q = p + (i + u*T) * 8;
            asm volatile("ld.global.nc.L1::no_allocate.v8.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                : "=f"(r[u][0]), "=f"(r[u][1]), "=f"(r[u][2]), "=f"(r[u][3]),
                  "=f"(r[u][4]), "=f"(r[u][5]), "=f"(r[u][6]), "=f"(r[u][7]) : "l"(q));
        }
        #pragma unroll
        for(int u = 0; u < U; u++)
            #pragma unroll
            for(int j = 0; j < 8; j++) acc += r[u][j];
    }
    if(acc == 1234.5678f) out[0] = acc;
}

// TMA bulk: each block owns S smem stages of CHUNK bytes; one elected thread issues
// cp.async.bulk for chunk c into stage c%S, everyone waits on the stage mbarrier and touches 1 word.
__global__ void k_tma(const char* __restrict__ p, size_t nchunks, int chunk, int S, float* out){
    extern __shared__ __align__(128) char smem[];
    uint64_t* bar = (uint64_t*)smem;                 // S mbarriers
    char* buf = smem + 128;
    const uint32_t bar0 = (uint32_t)__cvta_generic_to_shared(bar);
    const uint32_t buf0 = (uint32_t)__cvta_generic_to_shared(buf);
    if(threadIdx.x == 0){
        for(int s = 0; s < S; s++) asm volatile("mbarrier.init.shared.b64 [%0], 1;" :: "r"(bar0 + 8*s));
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    __syncthreads();
    float acc = 0.f;
    // chunks for this block: blockIdx.x, +gridDim.x, ...
    size_t mine = nchunks > blockIdx.x ? (nchunks - blockIdx.x + gridDim.x - 1) / gridDim.x : 0;
    auto issue = [&](size_t j){
        size_t c = blockIdx.x + j * gridDim.x;
        int s = (int)(j % S);
        asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" :: "r"(bar0 + 8*s), "r"(chunk));
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                     :: "r"(buf0 + s*chunk), "l"(p + c*(size_t)chunk), "r"(chunk), "r"(bar0 + 8*s) : "memory");
    };
    if(threadIdx.x == 0) for(size_t j = 0; j < (size_t)S && j < mine; j++) issue(j);
    for(size_t j = 0; j < mine; j++){
        int s = (int)(j % S);
        uint32_t phase = (uint32_t)((j / S) & 1);
        uint32_t done = 0;
        while(!done) asm volatile("{ .reg .pred P; mbarrier.try_wait.parity.shared.b64 P, [%1], %2; selp.u32 %0,1,0,P; }"
                                  : "=r"(done) : "r"(bar0 + 8*s), "r"(phase));
        acc += ((const float*)(buf + s*chunk))[threadIdx.x];
        __syncthreads();                                  // stage s free
        if(threadIdx.x == 0 && j + S < mine) issue(j + S);
    }
    if(acc == 1234.5678f) out[0] = acc;
}

static float timeit(cudaEvent_t a, cudaEvent_t b){ float ms; cudaEventSynchronize(b); cudaEventElapsedTime(&ms, a, b); return ms; }

int main(int argc, char** argv){
    size_t mb = argc > 1 ? atol(argv[1]) : 4096;
    int iters = argc > 2 ? atoi(argv[2]) : 10;
    size_t bytes = mb << 20;
    char* d; float* out;
    CU(cudaMalloc(&d, bytes)); CU(cudaMalloc(&out, 4)); CU(cudaMemset(d, 0, bytes));
    cudaDeviceProp pr; CU(cudaGetDeviceProperties(&pr, 0));
    const int sms = pr.multiProcessorCount;
    printf("device %s  SMs %d  buffer %zu MiB  iters %d\n", pr.name, sms, mb, iters);
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    const int bps[] = {2, 4, 8, 16, 32};
    const size_t n4 = bytes / 16, n8 = bytes / 32;

#define RUN(label, launch) do{ double best = 0; int bb = 0; for(int bp : bps){ int g = sms*bp; \
        launch; CU(cudaDeviceSynchronize()); cudaEventRecord(a); for(int it=0; it<iters; it++){ launch; } cudaEventRecord(b); \
        CU(cudaGetLastError()); double gbs = (double)bytes*iters / (timeit(a,b)*1e-3) / 1e9; \
        printf("  %-12s blocks/SM %2d : %6.1f GB/s\n", label, bp, gbs); if(gbs > best){ best = gbs; bb = bp; } } \
        printf("BEST %-12s %6.1f GB/s @ %d blocks/SM\n", label, best, bb); }while(0)

    RUN("ld U1",   (k_ld<1,false><<<g,256>>>((const float4*)d, n4, out)));
    RUN("ld U2",   (k_ld<2,false><<<g,256>>>((const float4*)d, n4, out)));
    RUN("ld U4",   (k_ld<4,false><<<g,256>>>((const float4*)d, n4, out)));
    RUN("ld U8",   (k_ld<8,false><<<g,256>>>((const float4*)d, n4, out)));
    RUN("nc U4",   (k_ld<4,true ><<<g,256>>>((const float4*)d, n4, out)));
    RUN("nc U8",   (k_ld<8,true ><<<g,256>>>((const float4*)d, n4, out)));
    RUN("v8 U1",   (k_v8<1><<<g,256>>>((const float*)d, n8, out)));
    RUN("v8 U2",   (k_v8<2><<<g,256>>>((const float*)d, n8, out)));
    RUN("v8 U4",   (k_v8<4><<<g,256>>>((const float*)d, n8, out)));
    for(int chunk : {16384, 32768}) for(int S : {2, 4, 6}){
        size_t sm = 128 + (size_t)S*chunk;
        if(sm > (size_t)pr.sharedMemPerBlockOptin) continue;
        CU(cudaFuncSetAttribute(k_tma, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm));
        char label[32]; snprintf(label, sizeof label, "tma %dK x%d", chunk/1024, S);
        size_t nch = bytes / chunk;
        double best = 0; int bb = 0;
        for(int bp : {1, 2, 4}){
            if(sm * bp > (size_t)pr.sharedMemPerMultiprocessor) continue;
            int g = sms*bp;
            k_tma<<<g,256,sm>>>(d, nch, chunk, S, out); CU(cudaDeviceSynchronize());
            cudaEventRecord(a); for(int it=0; it<iters; it++) k_tma<<<g,256,sm>>>(d, nch, chunk, S, out); cudaEventRecord(b);
            CU(cudaGetLastError());
            double gbs = (double)bytes*iters / (timeit(a,b)*1e-3) / 1e9;
            printf("  %-12s blocks/SM %2d : %6.1f GB/s\n", label, bp, gbs);
            if(gbs > best){ best = gbs; bb = bp; }
        }
        printf("BEST %-12s %6.1f GB/s @ %d blocks/SM\n", label, best, bb);
    }
    return 0;
}
