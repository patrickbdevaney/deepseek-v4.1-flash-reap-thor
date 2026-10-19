// flops_probe.cu — achievable COMPUTE peak on this box, which the project has never measured.
//
// WHY. Every roofline in this repo is a BANDWIDTH roofline (tools/bw_probe.cu, 233 GB/s). Findings
// 84-86 all lean on statements like "cattn:sparse runs at 0.72 TFLOPS, which is low" — and "low"
// there is a spec-sheet-class assumption, not a measurement. That assumption decides whether the
// remaining prefill work is a tensor-core rewrite (order of magnitude) or fine tuning (percent), so
// it is worth one probe.
//
// WHAT IT MEASURES, in the order that matters for the decision:
//   1. FP32 FFMA  — the regime sparse_attn/compressor actually run in today (fp32 dot products).
//   2. FP16/BF16 mma.sync m16n8k16 — the Ampere-lineage tensor-core path, what a flash-attention
//      style score kernel would use.
//   3. FP8 e4m3 mma.sync m16n8k32 — used by the fp8_block_gemm path.
// FP4 is deliberately NOT probed here: HARDWARE.md records that mma.sync FP4 is BLOCKED on sm_110a
// and that Thor reaches FP4 only through tcgen05, which needs tensor-memory allocation and matrix
// descriptors — a probe of its own, not a line in this one.
//
//   nvcc -O3 -arch=sm_110a tools/flops_probe.cu -o build/flops_probe
#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CU(x) do{ cudaError_t e=(x); if(e){ printf("CUDA %s @%d\n", cudaGetErrorString(e), __LINE__); return 1; } }while(0)

// ---- FP32 FFMA. NACC independent chains per thread so the measurement is throughput-bound rather
// than latency-bound: one dependent chain would measure FFMA latency (~4 cycles) and report ~1/4 of
// peak, which is the classic way this microbenchmark lies.
template<int NACC>
__global__ void ffma_kernel(float* out, int iters, float seed){
    float a[NACC], c[NACC];
    #pragma unroll
    for(int j=0;j<NACC;++j){ a[j] = seed + j; c[j] = seed * (j+1); }
    float b = seed * 1.000001f;
    for(int i=0;i<iters;++i){
        #pragma unroll
        for(int j=0;j<NACC;++j) c[j] = fmaf(a[j], b, c[j]);
    }
    float s=0.f;
    #pragma unroll
    for(int j=0;j<NACC;++j) s += c[j];
    if(s == 1234.5678f) out[0] = s;          // never true; defeats DCE without a real store
}

// ---- FP16 tensor core, mma.sync.m16n8k16.f32.f16.f16.f32 (one per warp = 16*8*16*2 = 4096 FLOP)
__global__ void mma_f16_kernel(float* out, int iters){
    unsigned a0=0x3c003c00u, a1=0x3c003c00u, a2=0x3c003c00u, a3=0x3c003c00u;
    unsigned b0=0x3c003c00u, b1=0x3c003c00u;
    float c0=0.f,c1=0.f,c2=0.f,c3=0.f;
    for(int i=0;i<iters;++i){
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                     "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                     : "+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3)
                     : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
    }
    if(c0+c1+c2+c3 == 1234.5678f) out[0] = c0;
}

// ---- BF16 tensor core, same shape
__global__ void mma_bf16_kernel(float* out, int iters){
    unsigned a0=0x3f803f80u, a1=0x3f803f80u, a2=0x3f803f80u, a3=0x3f803f80u;
    unsigned b0=0x3f803f80u, b1=0x3f803f80u;
    float c0=0.f,c1=0.f,c2=0.f,c3=0.f;
    for(int i=0;i<iters;++i){
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                     "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                     : "+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3)
                     : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
    }
    if(c0+c1+c2+c3 == 1234.5678f) out[0] = c0;
}

// ---- FP8 e4m3 tensor core, m16n8k32 (16*8*32*2 = 8192 FLOP per warp-instruction)
__global__ void mma_fp8_kernel(float* out, int iters){
    unsigned a0=0x38383838u,a1=0x38383838u,a2=0x38383838u,a3=0x38383838u;
    unsigned b0=0x38383838u,b1=0x38383838u;
    float c0=0.f,c1=0.f,c2=0.f,c3=0.f;
    for(int i=0;i<iters;++i){
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                     "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                     : "+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3)
                     : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
    }
    if(c0+c1+c2+c3 == 1234.5678f) out[0] = c0;
}

static float time_ms(void(*launch)(float*,int,int,int), float* out, int grid, int block, int iters){
    cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
    launch(out, grid, block, iters);                 // warm
    cudaDeviceSynchronize();
    cudaEventRecord(a);
    launch(out, grid, block, iters);
    cudaEventRecord(b); cudaEventSynchronize(b);
    float ms=0.f; cudaEventElapsedTime(&ms,a,b);
    cudaEventDestroy(a); cudaEventDestroy(b);
    return ms;
}

static const int NACC = 32;
static void L_ffma(float* o,int g,int b,int it){ ffma_kernel<NACC><<<g,b>>>(o,it,1.0001f); }
static void L_f16 (float* o,int g,int b,int it){ mma_f16_kernel<<<g,b>>>(o,it); }
static void L_bf16(float* o,int g,int b,int it){ mma_bf16_kernel<<<g,b>>>(o,it); }
static void L_fp8 (float* o,int g,int b,int it){ mma_fp8_kernel<<<g,b>>>(o,it); }

int main(){
    cudaDeviceProp p; CU(cudaGetDeviceProperties(&p,0));
    // cudaDeviceProp::clockRate was REMOVED in CUDA 13; the attribute API still carries it.
    int clk_khz = 0; cudaDeviceGetAttribute(&clk_khz, cudaDevAttrClockRate, 0);
    printf("device: %s  SMs=%d  clock=%.3f GHz  cc=%d.%d\n",
           p.name, p.multiProcessorCount, clk_khz/1e6, p.major, p.minor);
    // Theoretical FP32 FFMA for a 128-core/SM Blackwell SM, stated as a REFERENCE not a claim:
    double th = 2.0 * 128.0 * p.multiProcessorCount * (clk_khz/1e6) / 1e3;
    printf("reference FP32 FFMA peak if 128 cores/SM: %.2f TFLOPS "
           "(cores/SM is NOT reported by the runtime; treat as a sanity anchor, not a measurement)\n\n", th);

    float* out; CU(cudaMalloc(&out,sizeof(float)));
    const int block = 256;
    const int grid  = p.multiProcessorCount * 8;     // 8 blocks/SM: enough to cover, cheap to run
    const int iters = 100000;
    const long warps = (long)grid * (block/32);

    printf("%-26s %10s %12s\n","kernel","ms","TFLOPS");
    {   float ms = time_ms(L_ffma,out,grid,block,iters);
        double fl = 2.0*(double)grid*block*NACC*iters;
        printf("%-26s %10.2f %12.2f\n","FP32 FFMA", ms, fl/(ms/1e3)/1e12); }
    {   float ms = time_ms(L_f16,out,grid,block,iters);
        double fl = 4096.0*(double)warps*iters;
        printf("%-26s %10.2f %12.2f\n","FP16 mma m16n8k16", ms, fl/(ms/1e3)/1e12); }
    {   float ms = time_ms(L_bf16,out,grid,block,iters);
        double fl = 4096.0*(double)warps*iters;
        printf("%-26s %10.2f %12.2f\n","BF16 mma m16n8k16", ms, fl/(ms/1e3)/1e12); }
    {   float ms = time_ms(L_fp8,out,grid,block,iters);
        double fl = 8192.0*(double)warps*iters;
        printf("%-26s %10.2f %12.2f\n","FP8 e4m3 mma m16n8k32", ms, fl/(ms/1e3)/1e12); }
    cudaError_t e = cudaGetLastError();
    if(e) printf("\nNOTE: last CUDA error = %s (a row above may be an unsupported instruction)\n",
                 cudaGetErrorString(e));
    return 0;
}
