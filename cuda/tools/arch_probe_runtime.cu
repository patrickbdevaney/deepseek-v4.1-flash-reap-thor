#include <cstdio>
#include <cuda_runtime.h>
__global__ void t5(unsigned* out){
  __shared__ alignas(16) unsigned smem[4];
  if(threadIdx.x==0) smem[0]=0;
  __syncthreads();
  // allocate 32 columns of tensor memory, read the assigned base address back, free it.
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\n"::"l"(smem),"r"(32));
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n"::);
  __syncthreads();
  unsigned base = smem[0];
  if(threadIdx.x==0) out[0]=base;
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;\n"::"r"(base),"r"(32));
}
__global__ void bulk(unsigned* out, const unsigned* g){
  __shared__ alignas(128) unsigned smem[64];
  __shared__ alignas(8) unsigned long long bar;
  if(threadIdx.x==0){ asm volatile("mbarrier.init.shared.b64 [%0], 1;\n"::"l"(&bar)); }
  __syncthreads();
  if(threadIdx.x==0){
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
                 ::"l"(smem),"l"(g),"r"(256),"l"(&bar));
    asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n"::"l"(&bar),"r"(256));
  }
  __syncthreads();
  if(threadIdx.x==0){ unsigned st;
    do { asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], 0; selp.u32 %0,1,0,p; }\n"
                      :"=r"(st):"l"(&bar)); } while(!st); }
  __syncthreads();
  if(threadIdx.x==0) out[0]=smem[0]+smem[63];
}
int main(){
  unsigned *d,*g; cudaMalloc(&d,16); cudaMalloc(&g,4096); cudaMemset(g,1,4096);
  t5<<<1,32>>>(d); cudaError_t e1=cudaDeviceSynchronize();
  printf("tcgen05 alloc/dealloc RUNTIME : %s\n", e1?cudaGetErrorString(e1):"OK");
  cudaGetLastError();
  bulk<<<1,32>>>(d,g); cudaError_t e2=cudaDeviceSynchronize();
  printf("cp.async.bulk (TMA) RUNTIME   : %s\n", e2?cudaGetErrorString(e2):"OK");
  return 0;
}
