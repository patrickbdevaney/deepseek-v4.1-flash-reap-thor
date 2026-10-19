// gate_forkjoin_graph.cu — is the side-stream fork/join capturable into a CUDA graph, 43 times?
//
// Finding 55 puts the shared expert on a secondary stream (include/dscratch.h, g_side). The base-AR
// decode path captures the WHOLE step — all 43 layers — into one CUDA graph (src/decode.cu), which
// is worth 1.26x and must not break. Three things could break it, and none of them fail loudly at
// build time:
//
//   1. cudaStreamCreate / cudaEventCreate during capture is illegal. (Handled by creating both in
//      arena_init, but that is a claim about call order, so gate it.)
//   2. A timing-enabled event cannot be captured; the events must be cudaEventDisableTiming.
//   3. The SAME event object is recorded and waited on once per layer — 43 record/wait pairs on two
//      event handles inside a single capture. That is the part with no precedent in this codebase.
//
// This gate reproduces exactly that pattern with trivial kernels, so a capture failure is reported
// in two seconds instead of after a 15-minute checkpoint load. It also checks the RESULT, not just
// that capture succeeded: a fork/join that captured but lost its dependency edge would replay with
// the join reading a stale value, and the accumulator would come out short.
//
//   build: nvcc -O2 -std=c++17 -gencode arch=compute_110a,code=sm_110a -I include \
//            tests/gate_forkjoin_graph.cu kernels/dscratch.cu -o build/gate_forkjoin_graph
#include "dscratch.h"
#include <cuda_runtime.h>
#include <cstdio>

#define CU(x) do{cudaError_t e=(x); if(e){fprintf(stderr,"cuda %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));return 1;} }while(0)

__global__ void k_add(float* a, float v, int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) a[i]+=v; }
__global__ void k_copy(float* d, const float* s, int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) d[i]=s[i]; }

int main(){
    const int N=4096, LAYERS=43;
    arena_init((size_t)8<<20);                     // creates g_side / g_side_fork / g_side_join
    if(!g_side){ printf("[forkjoin] g_side is null — arena_init did not create the side stream\n");
                 printf("\nGate FORKJOIN_GRAPH: FAIL\n"); return 1; }

    float *main_buf,*side_buf;
    CU(cudaMalloc(&main_buf,N*4)); CU(cudaMalloc(&side_buf,N*4));

    cudaStream_t st; CU(cudaStreamCreate(&st));

    // The shape of one layer: main stream does work, side stream does independent work off a fork,
    // main stream joins and consumes it. Repeated LAYERS times on ONE pair of event handles.
    auto body=[&](cudaStream_t s){
        for(int L=0;L<LAYERS;++L){
            k_add<<<(N+255)/256,256,0,s>>>(main_buf,1.f,N);                 // main branch
            cudaEventRecord(g_side_fork,s); cudaStreamWaitEvent(g_side,g_side_fork,0);
            k_copy<<<(N+255)/256,256,0,g_side>>>(side_buf,main_buf,N);      // side branch reads main's result
            k_add <<<(N+255)/256,256,0,g_side>>>(side_buf,1.f,N);
            cudaEventRecord(g_side_join,g_side); cudaStreamWaitEvent(s,g_side_join,0);
            k_copy<<<(N+255)/256,256,0,s>>>(main_buf,side_buf,N);           // join: consume it
        }
    };

    // --- eager reference ---
    CU(cudaMemset(main_buf,0,N*4));
    body(st); CU(cudaStreamSynchronize(st));
    float eager=0; CU(cudaMemcpy(&eager,main_buf,4,cudaMemcpyDeviceToHost));

    // --- captured ---
    CU(cudaMemset(main_buf,0,N*4));
    cudaGraph_t g; cudaGraphExec_t ex;
    cudaError_t cerr = cudaStreamBeginCapture(st,cudaStreamCaptureModeThreadLocal);
    if(cerr==cudaSuccess){ body(st); cerr = cudaStreamEndCapture(st,&g); }
    if(cerr!=cudaSuccess){
        printf("[forkjoin] capture FAILED: %s\n", cudaGetErrorString(cerr));
        printf("[forkjoin] the base-AR CUDA graph (worth 1.26x) would break. Keep NO_MOESPLIT=1.\n");
        printf("\nGate FORKJOIN_GRAPH: FAIL\n"); return 1;
    }
    size_t nnodes=0; cudaGraphGetNodes(g,nullptr,&nnodes);
    CU(cudaGraphInstantiate(&ex,g,0));
    CU(cudaGraphLaunch(ex,st)); CU(cudaStreamSynchronize(st));
    float graphed=0; CU(cudaMemcpy(&graphed,main_buf,4,cudaMemcpyDeviceToHost));

    // Every layer adds 1 on the main branch and 1 on the side branch, and the join takes the side
    // value — so the fixed point is exactly 2*LAYERS. A dropped dependency edge shows up here.
    const float want = 2.f*LAYERS;
    bool ok = (eager==want) && (graphed==want);
    printf("[forkjoin] %zu graph nodes over %d layers on one event pair\n", nnodes, LAYERS);
    printf("[forkjoin] eager=%.1f graph=%.1f expected=%.1f -> %s\n", eager, graphed, want, ok?"PASS":"FAIL");
    printf("\nGate FORKJOIN_GRAPH: %s\n", ok?"PASS":"FAIL");
    return ok?0:1;
}
