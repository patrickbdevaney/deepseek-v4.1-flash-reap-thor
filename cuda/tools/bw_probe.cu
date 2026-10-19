// bw_probe.cu — streaming-bandwidth microbenchmark + an ncu smoke target.
// Measures achievable read bandwidth on the unified pool, which ROOFLINE.md currently
// carries as an INHERITED ~200 GB/s planning figure rather than a measurement.
//   nvcc -O3 -arch=sm_110a tools/bw_probe.cu -o build/bw_probe
#include <cstdio>
#include <cuda_runtime.h>

#define CU(x) do{ cudaError_t e=(x); if(e){ printf("CUDA %s @%d\n", cudaGetErrorString(e), __LINE__); return 1; } }while(0)

// Pure streaming read: grid-stride over float4, reduce so nothing is optimised away.
__global__ void stream_read(const float4* __restrict__ p, size_t n4, float* __restrict__ out){
    float acc = 0.f;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for(size_t i = (size_t)blockIdx.x*blockDim.x + threadIdx.x; i < n4; i += stride){
        float4 v = p[i];
        acc += v.x + v.y + v.z + v.w;
    }
    if(acc == 1234.5678f) out[0] = acc;   // never true; defeats DCE without a real write
}

int main(int argc, char** argv){
    size_t mb = (argc > 1) ? atol(argv[1]) : 4096;      // buffer size, MiB
    int iters  = (argc > 2) ? atoi(argv[2]) : 20;
    size_t bytes = mb << 20, n4 = bytes / sizeof(float4);

    float4* d; float* out;
    CU(cudaMalloc(&d, bytes));
    CU(cudaMalloc(&out, sizeof(float)));
    CU(cudaMemset(d, 0, bytes));

    int dev; CU(cudaGetDevice(&dev));
    cudaDeviceProp prop; CU(cudaGetDeviceProperties(&prop, dev));
    int blocks = prop.multiProcessorCount * 16;
    printf("device %s  SMs %d  buffer %zu MiB  blocks %d\n", prop.name, prop.multiProcessorCount, mb, blocks);

    stream_read<<<blocks, 256>>>(d, n4, out);           // warm
    CU(cudaDeviceSynchronize());

    cudaEvent_t a, b; CU(cudaEventCreate(&a)); CU(cudaEventCreate(&b));
    CU(cudaEventRecord(a));
    for(int i = 0; i < iters; ++i) stream_read<<<blocks, 256>>>(d, n4, out);
    CU(cudaEventRecord(b));
    CU(cudaEventSynchronize(b));

    float ms; CU(cudaEventElapsedTime(&ms, a, b));
    double gbs = (double)bytes * iters / (ms * 1e-3) / 1e9;
    printf("read  %.1f GB/s  (%.3f ms/pass, %d passes)\n", gbs, ms/iters, iters);
    printf("  vs 273 GB/s spec peak: %.0f%%\n", 100.0 * gbs / 273.0);
    CU(cudaFree(d)); CU(cudaFree(out));
    return 0;
}
