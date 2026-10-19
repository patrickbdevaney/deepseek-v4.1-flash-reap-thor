// footprint_probe.cu — does the SIZE of the resident working set cost bandwidth?
//
// Why this exists. Every byte-carrying kernel in the engine measures ~1.8-2.4x slower in situ than
// the same kernel on the same shape in gemm_bench, and the gap is uniform across kernels that have
// nothing in common (ogroup fp8 GEMV, the m16 fp8 tile, the shared-expert GEMV, lm_head). Finding
// 47 wrote that off as "the bench relaunches on rotating weights so launches overlap", and the
// verify-graph experiment (1.05x, ~2788 nodes) proved launch overhead is NOT the difference. So the
// gap is still unexplained, and one variable has never been tested:
//
//   gemm_bench streams out of a ~400 MB cudaMalloc pool.
//   The engine streams out of a 111.5 GiB cudaMallocManaged pool.
//
// alloc_probe.cu answered "which ALLOCATOR" (mapped-host vs managed vs cudaMalloc) at one small
// size. It never varied the SIZE. If achieved bandwidth falls as the resident footprint grows —
// SMMU/TLB reach, page-table walk depth, memory-controller page locality — then the 240 GB/s
// roofline every "% of achievable" in this project is measured against is a number the engine
// structurally cannot reach, and the uniform ~2x is one bug, not fifteen.
//
// Two questions, separated, because they are different mechanisms:
//   Q1 SIZE OF THE REGION  : read a FIXED 1 GiB, but out of regions of growing size. Same bytes
//                            read, same kernel, same launch config — only the address span differs.
//                            This isolates translation reach from anything to do with volume.
//   Q2 VOLUME             : read the WHOLE region. This is what the engine actually does per token
//                            (it walks all 12.26 GB of live weights every step) and folds in any
//                            effect that only appears once you have streamed past cache capacity.
//
// Access pattern matches the engine, not a memcpy: a warp reads 128 contiguous bytes per row and
// rows are strided, because a page-table or memory-controller effect shows up in the address
// stream's jumpiness, not in a perfectly sequential walk. `stream_read` is kept as the control.
//
//   build: nvcc -O3 -std=c++17 -gencode arch=compute_110a,code=sm_110a -I include \
//            tools/footprint_probe.cu -o build/footprint_probe
//   run:   ./build/footprint_probe            (skips footprints that do not fit)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

#define CU(x) do{ cudaError_t e=(x); if(e){ printf("CUDA %s @%d\n", cudaGetErrorString(e), __LINE__); exit(1);} }while(0)

// Contiguous control: every warp walks a contiguous run. TLB-friendliest possible pattern.
__global__ void stream_read(const float4* __restrict__ p, size_t n4, float* __restrict__ out){
    float acc = 0.f;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for(size_t i = (size_t)blockIdx.x*blockDim.x + threadIdx.x; i < n4; i += stride){
        float4 v = p[i]; acc += v.x + v.y + v.z + v.w;
    }
    if(acc == 1234.5678f) out[0] = acc;
}

// Engine-shaped: one block per weight row, 128 contiguous bytes per warp per K-block, rows strided.
// `rowf4` is the row length in float4 (a 4096-byte weight row = 256 float4).
__global__ void strided_read(const float4* __restrict__ p, size_t n4, size_t rowf4, float* __restrict__ out){
    float acc = 0.f;
    const size_t rows = n4 / rowf4;
    for(size_t r = blockIdx.x; r < rows; r += gridDim.x){
        const float4* row = p + r*rowf4;
        for(size_t i = threadIdx.x; i < rowf4; i += blockDim.x){
            float4 v = row[i]; acc += v.x + v.y + v.z + v.w;
        }
    }
    if(acc == 1234.5678f) out[0] = acc;
}

// Read a fixed VOLUME out of a region of size n4, striding across the whole span so the address
// stream covers the entire region. `skip` rows between consecutive reads spreads the reads out.
__global__ void strided_read_span(const float4* __restrict__ p, size_t rows_total, size_t rowf4,
                                  size_t rows_to_read, float* __restrict__ out){
    float acc = 0.f;
    const size_t skip = rows_total / rows_to_read;      // >=1: spread the reads over the whole span
    for(size_t j = blockIdx.x; j < rows_to_read; j += gridDim.x){
        const float4* row = p + (j*skip)*rowf4;
        for(size_t i = threadIdx.x; i < rowf4; i += blockDim.x){
            float4 v = row[i]; acc += v.x + v.y + v.z + v.w;
        }
    }
    if(acc == 1234.5678f) out[0] = acc;
}

static float* g_sink = nullptr;
static const size_t ROWB = 4096;                        // 4 KB weight row, as in wo_a / wq_b
static const size_t ROWF4 = ROWB / sizeof(float4);      // 256

// Median of `reps` timings, so one scheduling hiccup cannot create a finding.
static double time_ms(void(*launch)(const float4*,size_t,size_t,size_t), const float4* p,
                      size_t a, size_t b, size_t c, int reps){
    std::vector<double> t;
    for(int i=0;i<reps;++i){
        cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        cudaEventRecord(e0); launch(p,a,b,c); cudaEventRecord(e1);
        CU(cudaEventSynchronize(e1));
        float ms=0; cudaEventElapsedTime(&ms,e0,e1); t.push_back(ms);
        cudaEventDestroy(e0); cudaEventDestroy(e1);
    }
    std::sort(t.begin(), t.end());
    return t[t.size()/2];
}

static void L_stream (const float4* p,size_t n4,size_t,size_t){ stream_read <<<2048,256>>>(p,n4,g_sink); }
static void L_strided(const float4* p,size_t n4,size_t rf,size_t){ strided_read<<<2048,256>>>(p,n4,rf,g_sink); }
static void L_span   (const float4* p,size_t rows,size_t rf,size_t rd){ strided_read_span<<<2048,256>>>(p,rows,rf,rd,g_sink); }

int main(int argc, char** argv){
    CU(cudaMalloc(&g_sink, 4));
    int dev=0; CU(cudaGetDevice(&dev));
    size_t freeb=0, totalb=0; CU(cudaMemGetInfo(&freeb,&totalb));
    printf("device %d: %.1f GiB free of %.1f GiB\n", dev, freeb/1073741824.0, totalb/1073741824.0);

    // The fixed volume for Q1. 1 GiB is large enough to be far past any cache and small enough that
    // it is the SAME work at every footprint.
    const size_t FIXED = (size_t)1<<30;
    const int REPS = 3;

    // Footprints to sweep. The engine's is 111.5 GiB; the bench's is ~0.4 GiB.
    const double GiB[] = {0.5, 2, 8, 32, 64, 96};
    const int NG = sizeof(GiB)/sizeof(GiB[0]);

    for(int alloc=0; alloc<2; ++alloc){
        const char* aname = alloc==0 ? "cudaMalloc" : "managed+advise+prefetch";
        printf("\n=== %s ===\n", aname);
        printf("%10s %14s %14s %14s %14s\n", "footprint",
               "Q1 stream", "Q1 strided", "Q2 stream", "Q2 strided");
        printf("%10s %14s %14s %14s %14s\n", "(GiB)",
               "GB/s (1GiB)", "GB/s (1GiB)", "GB/s (all)", "GB/s (all)");
        for(int gi=0; gi<NG; ++gi){
            size_t bytes = (size_t)(GiB[gi]*1073741824.0);
            bytes -= bytes % ROWB;
            CU(cudaMemGetInfo(&freeb,&totalb));
            if(bytes + ((size_t)6<<30) > freeb){ printf("%10.1f   (skipped: needs %.1f GiB, %.1f free)\n",
                                                        GiB[gi], bytes/1073741824.0, freeb/1073741824.0); continue; }
            void* p=nullptr;
            if(alloc==0){
                if(cudaMalloc(&p,bytes)!=cudaSuccess){ printf("%10.1f   (cudaMalloc failed)\n",GiB[gi]); cudaGetLastError(); continue; }
            } else {
                if(cudaMallocManaged(&p,bytes)!=cudaSuccess){ printf("%10.1f   (managed alloc failed)\n",GiB[gi]); cudaGetLastError(); continue; }
                // Exactly what include/weight_store.h does for the real weights.
                cudaMemLocation loc{}; loc.type=cudaMemLocationTypeDevice; loc.id=dev;
                cudaMemAdvise(p,bytes,cudaMemAdviseSetPreferredLocation,loc);
                CU(cudaMemPrefetchAsync(p,bytes,loc,0,0));
            }
            CU(cudaMemset(p,0x11,bytes));            // make every page resident before timing
            CU(cudaDeviceSynchronize());

            const size_t n4 = bytes/sizeof(float4);
            const size_t rows = bytes/ROWB;
            const size_t rows_fixed = FIXED/ROWB < rows ? FIXED/ROWB : rows;
            const double fixed_gb = rows_fixed*(double)ROWB/1e9;
            const double all_gb   = bytes/1e9;

            // Q1: same 1 GiB of reads, spread across the whole region.
            double q1s = time_ms(L_span, (const float4*)p, rows, ROWF4, rows_fixed, REPS);
            double q1t = q1s;   // span kernel is the strided one; contiguous control below
            // contiguous control over the first 1 GiB only
            double q1c = time_ms(L_stream, (const float4*)p, FIXED/sizeof(float4) < n4 ? FIXED/sizeof(float4) : n4, 0, 0, REPS);
            // Q2: read everything.
            double q2c = time_ms(L_stream,  (const float4*)p, n4, 0, 0, REPS);
            double q2t = time_ms(L_strided, (const float4*)p, n4, ROWF4, 0, REPS);

            printf("%10.1f %14.1f %14.1f %14.1f %14.1f\n", GiB[gi],
                   fixed_gb/(q1c/1e3), fixed_gb/(q1t/1e3), all_gb/(q2c/1e3), all_gb/(q2t/1e3));
            fflush(stdout);
            CU(cudaFree(p));
        }
    }
    printf("\nread: Q1 columns hold the READ VOLUME CONSTANT at 1 GiB and grow only the address span.\n"
           "If Q1 falls with footprint, it is translation reach (TLB/SMMU), not cache capacity.\n"
           "If Q1 is flat and only Q2 falls, it is volume, and the roofline is fine.\n");
    return 0;
}
