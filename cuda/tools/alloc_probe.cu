// alloc_probe.cu — does the ALLOCATOR the weights live in cost bandwidth?
//
// Every byte-carrying kernel in the engine lands in a narrow band well under the roofline, and the
// band is the same for kernels that have nothing else in common (measured at K=1, in situ):
//
//   cattn:ogroup 170 GB/s | moe w13+w2 168 | lm_head 183 | cattn:q_proj 142
//
// A uniform ~70% across unrelated kernels is not a kernel property. The one thing they share is
// where the weights live: `WeightStore` preads each shard into `cudaHostAlloc(cudaHostAllocMapped)`
// and hands the kernels `cudaHostGetDevicePointer`. That is zero-copy MAPPED HOST memory — chosen
// because the model is 100.4 GiB in a 122 GiB unified pool and a device copy would double it.
//
// ROOFLINE.md's 240 GB/s came from `bw_probe`, which measures a `cudaMalloc` buffer. If mapped-host
// reads are slower than device reads on this part, then the engine has been measured against a
// roofline it structurally cannot reach, and every "% of achievable" in the log is wrong.
//
// Same streaming kernel, same size, three allocators. No model, no weights, ~30 seconds.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <vector>

#define CU(x) do{ cudaError_t e=(x); if(e){ printf("CUDA %s @%d\n", cudaGetErrorString(e), __LINE__); return 1; } }while(0)

__global__ void stream_read(const float4* __restrict__ p, size_t n4, float* __restrict__ out){
    float acc = 0.f;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for(size_t i = (size_t)blockIdx.x*blockDim.x + threadIdx.x; i < n4; i += stride){
        float4 v = p[i];
        acc += v.x + v.y + v.z + v.w;
    }
    if(acc == 1234.5678f) out[0] = acc;
}
// The engine does not stream contiguously — it reads a 128-byte line per warp per K-block with the
// rows of a weight matrix strided. Same total bytes, but the address stream jumps, which is where a
// page-table or SMMU difference between allocators would actually show up.
__global__ void strided_read(const float4* __restrict__ p, size_t n4, size_t rowf4, float* __restrict__ out){
    float acc = 0.f;
    const size_t rows = n4 / rowf4;
    for(size_t r = blockIdx.x; r < rows; r += gridDim.x){
        const float4* row = p + r*rowf4;
        for(size_t i = threadIdx.x; i < rowf4; i += blockDim.x){
            float4 v = row[i];
            acc += v.x + v.y + v.z + v.w;
        }
    }
    if(acc == 1234.5678f) out[0] = acc;
}

int main(int argc, char** argv){
    size_t mb   = (argc > 1) ? atol(argv[1]) : 4096;
    int    iters= (argc > 2) ? atoi(argv[2]) : 10;
    size_t bytes = mb << 20, n4 = bytes / sizeof(float4);
    int dev; CU(cudaGetDevice(&dev));
    cudaDeviceProp prop; CU(cudaGetDeviceProperties(&prop, dev));
    const int blocks = prop.multiProcessorCount * 16;
    printf("%s  SMs %d  buffer %zu MiB  iters %d\n\n", prop.name, prop.multiProcessorCount, mb, iters);
    float* out; CU(cudaMalloc(&out, sizeof(float)));
    const size_t rowf4 = 4096 / sizeof(float4);           // 4096-byte rows, like a 4096-col fp8 weight

    printf("%-34s %12s %12s\n", "allocator", "stream GB/s", "strided GB/s");
    for (int mode = 0; mode < 4; ++mode){
        void* p = nullptr; const char* name = "";
        switch(mode){
            case 0: name = "cudaMalloc (device)";
                    if(cudaMalloc(&p, bytes) != cudaSuccess) p = nullptr; break;
            case 1: name = "cudaHostAlloc Mapped  <- weights";
                    if(cudaHostAlloc(&p, bytes, cudaHostAllocMapped) == cudaSuccess){
                        void* d; if(cudaHostGetDevicePointer(&d, p, 0) == cudaSuccess){
                            cudaMemset(d, 0, bytes); p = d; } else p = nullptr;
                    } else p = nullptr; break;
            case 2: name = "cudaMallocManaged";
                    if(cudaMallocManaged(&p, bytes) != cudaSuccess) p = nullptr; break;
            case 3: name = "cudaMallocManaged + PreferredLoc";
                    if(cudaMallocManaged(&p, bytes) == cudaSuccess){
                        // CUDA 13 takes a cudaMemLocation, not a bare device ordinal.
                        cudaMemLocation loc{}; loc.type = cudaMemLocationTypeDevice; loc.id = dev;
                        cudaMemAdvise(p, bytes, cudaMemAdviseSetPreferredLocation, loc);
                        cudaMemPrefetchAsync(p, bytes, loc, 0, 0); cudaDeviceSynchronize();
                    } else p = nullptr; break;
        }
        if(!p){ printf("%-34s %12s %12s\n", name, "n/a", "n/a"); continue; }
        if(mode != 1) cudaMemset(p, 0, bytes);
        cudaDeviceSynchronize();
        double gbs[2];
        for(int k = 0; k < 2; ++k){
            if(k==0) stream_read <<<blocks,256>>>((const float4*)p, n4, out);
            else     strided_read<<<blocks,256>>>((const float4*)p, n4, rowf4, out);
            CU(cudaDeviceSynchronize());
            cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
            cudaEventRecord(a);
            for(int i=0;i<iters;++i){
                if(k==0) stream_read <<<blocks,256>>>((const float4*)p, n4, out);
                else     strided_read<<<blocks,256>>>((const float4*)p, n4, rowf4, out);
            }
            cudaEventRecord(b); CU(cudaEventSynchronize(b));
            float ms=0; cudaEventElapsedTime(&ms,a,b);
            gbs[k] = (double)bytes*iters / (ms*1e-3) / 1e9;
            cudaEventDestroy(a); cudaEventDestroy(b);
        }
        printf("%-34s %12.1f %12.1f\n", name, gbs[0], gbs[1]);
        if(mode==1){ /* free the HOST pointer, not the device alias — leaked deliberately, we exit */ }
        else cudaFree(p);
    }
    printf("\nIf row 1 is materially below row 0, the engine's weights are in the slow allocator and\n"
           "every '%% of achievable' recorded against the 240 GB/s cudaMalloc roofline is overstated.\n");

    // FRAGMENTATION. The probe above reads ONE contiguous allocation. The engine reads 45,821
    // tensors scattered across ~20 managed shard allocations, one weight matrix per kernel, each
    // kernel touching a different region. If page-table locality is what separates the probe's 233
    // GB/s from the engine's ~155, then splitting the same bytes across many allocations and
    // walking them in sequence should reproduce the engine's number on a kernel that is otherwise
    // identical. Same total bytes, same kernel, only the allocation pattern differs.
    printf("\n--- same bytes, split across N allocations, read in sequence ---\n");
    printf("%-10s %14s %14s\n", "N allocs", "each (MiB)", "GB/s");
    for (int nalloc : {1, 4, 16, 64, 256}){
        const size_t per = bytes / nalloc;
        if (per < (1u<<20)) continue;
        std::vector<void*> ps(nalloc, nullptr);
        bool ok = true;
        for (int i=0;i<nalloc;++i){
            if (cudaMallocManaged(&ps[i], per) != cudaSuccess){ ok=false; break; }
            cudaMemLocation loc{}; loc.type = cudaMemLocationTypeDevice; loc.id = dev;
            cudaMemAdvise(ps[i], per, cudaMemAdviseSetPreferredLocation, loc);
            cudaMemPrefetchAsync(ps[i], per, loc, 0, 0);
        }
        cudaDeviceSynchronize();
        if(!ok){ for(auto p2:ps) if(p2) cudaFree(p2); printf("%-10d %14s %14s\n", nalloc, "-", "alloc failed"); continue; }
        const size_t pn4 = per/sizeof(float4);
        for(int i=0;i<nalloc;++i) stream_read<<<blocks,256>>>((const float4*)ps[i], pn4, out);
        CU(cudaDeviceSynchronize());
        cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
        cudaEventRecord(a);
        for(int k=0;k<iters;++k) for(int i=0;i<nalloc;++i) stream_read<<<blocks,256>>>((const float4*)ps[i], pn4, out);
        cudaEventRecord(b); CU(cudaEventSynchronize(b));
        float ms=0; cudaEventElapsedTime(&ms,a,b);
        printf("%-10d %14.1f %14.1f\n", nalloc, per/1048576.0, (double)bytes*iters/(ms*1e-3)/1e9);
        cudaEventDestroy(a); cudaEventDestroy(b);
        for(auto p2:ps) cudaFree(p2);
    }
    return 0;
}
