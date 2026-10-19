// overlap_probe.cu — can ONE decode-sized kernel saturate memory, or does it take two at once?
//
// The question this settles. After this cycle's dprof re-measurement the verify splits cleanly:
//
//   routed MoE (w1w3 + w2)   71.4 ms carrying 17.18 GB  = 241 GB/s   <- AT the roofline
//   everything else          92.6 ms carrying  8.81 GB  =  95 GB/s   <- 41% of it
//
// Every kernel in the second group is small: one weight matrix, M=1..5 rows of activation, a few
// hundred microseconds. The MoE kernel is one launch with ~60k warps of work. The obvious
// hypothesis is that a decode-sized kernel simply does not have enough concurrent misses in flight
// to cover DRAM latency on its own, and that the engine's serialised layer chain therefore leaves
// half the memory system idle at all times — not because any kernel is badly written, but because
// there is only ever one of them running.
//
// Two explanations were already eliminated, so this is not a fishing trip:
//   - working-set size          — tools/footprint_probe.cu: 230-246 GB/s at 0.5 GiB and at 64 GiB,
//                                 both allocators. Streaming out of a 111 GiB pool is free.
//   - kernel-launch overhead    — the VERIFYGRAPH experiment: 2788 graph nodes, 1.05x.
// And note gemm_bench does NOT overlap anything: `timeit` launches reps back-to-back on stream 0,
// which is stream-ordered, so its numbers are serialised too. Whatever the gap is, it is not that.
//
// The measurement. Take the real decode shapes, run W independent copies of the same GEMM, and
// compare putting them on ONE stream (what the engine does today) against W streams (what an
// intra-layer multi-stream or a fused grouped launch would buy). Aggregate bandwidth is the answer:
//
//   if 1 stream and W streams reach the same GB/s -> the kernels already saturate; there is no
//        concurrency lever, and 95 GB/s in situ is a cache-residency story, not a scheduling one.
//   if W streams reach materially more            -> the engine is leaving that factor on the floor
//        every layer, and independent chains within a layer (q vs kv vs compressor) are worth
//        running concurrently.
//
// Weights rotate over >=1 GB per stream so nothing is served from L2 — the in-situ condition.
//
//   build: nvcc -O3 -std=c++17 -gencode arch=compute_110a,code=sm_110a -I include \
//            tools/overlap_probe.cu kernels/fp8_block_gemm.cu kernels/tc_fp8_gemm.cu \
//            kernels/mla_attn.cu kernels/dscratch.cu -o build/overlap_probe
#include "deepseek_v4.h"
#include "fp8_block_gemm.h"
#include "dscratch.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

using namespace dsv4;
#define CU(x) do{cudaError_t e=(x); if(e){fprintf(stderr,"cuda %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)
extern bool g_tc_fp8;

struct Shape { const char* name; int N, K; };

// One independent "lane" of work: its own weights (rotating), its own activation, its own output.
struct Lane {
    std::vector<uint8_t*> W; std::vector<float*> S;
    uint8_t* A; float* As; float* C;
    int ncopy, cur=0;
};

static Lane make_lane(int N, int K, int M, size_t rot_bytes){
    Lane L;
    const size_t wb=(size_t)N*K;
    L.ncopy = (int)std::max<size_t>(2, rot_bytes/wb);
    for(int i=0;i<L.ncopy;++i){
        uint8_t* w; float* s;
        CU(cudaMalloc(&w,wb)); CU(cudaMemset(w,0x3c,wb));
        CU(cudaMalloc(&s,(size_t)(N/128+1)*(K/128)*4)); CU(cudaMemset(s,0,(size_t)(N/128+1)*(K/128)*4));
        L.W.push_back(w); L.S.push_back(s);
    }
    CU(cudaMalloc(&L.A,(size_t)M*K));      CU(cudaMemset(L.A,0x3c,(size_t)M*K));
    CU(cudaMalloc(&L.As,(size_t)M*(K/128)*4)); CU(cudaMemset(L.As,0,(size_t)M*(K/128)*4));
    CU(cudaMalloc(&L.C,(size_t)M*N*4));
    return L;
}
static void free_lane(Lane& L){
    for(auto p:L.W) cudaFree(p); for(auto p:L.S) cudaFree(p);
    cudaFree(L.A); cudaFree(L.As); cudaFree(L.C);
}

int main(int argc, char** argv){
    g_tc_fp8 = true;
    arena_init((size_t)256<<20);
    const int reps = argc>1 ? atoi(argv[1]) : 20;
    const size_t ROT = (size_t)1<<30;              // >=1 GB of rotating weights per lane

    const Shape shapes[] = {
        {"wq_a  [1024,4096]",  1024, 4096},
        {"wkv   [512,4096]",    512, 4096},
        {"wq_b  [32768,1024]",32768, 1024},
        {"wo_b  [4096,8192]",  4096, 8192},
    };
    const int NS = sizeof(shapes)/sizeof(shapes[0]);

    printf("overlap_probe: reps=%d, >=1 GB rotating weights per lane (cold)\n", reps);
    printf("Aggregate GB/s moving the SAME total bytes, 1 stream (engine today) vs W streams.\n\n");

    for(int M : {1,5}){
        printf("=== M=%d ===\n", M);
        printf("%-22s %5s %14s %14s %8s\n", "shape", "lanes", "1 stream", "W streams", "speedup");
        for(int si=0; si<NS; ++si){
            const Shape& sh = shapes[si];
            for(int W : {2,4}){
                std::vector<Lane> lanes; std::vector<cudaStream_t> st(W);
                for(int i=0;i<W;++i){ lanes.push_back(make_lane(sh.N,sh.K,M,ROT/W)); CU(cudaStreamCreate(&st[i])); }
                const double gb = (double)W*sh.N*sh.K/1e9;      // bytes moved per iteration (fp8 = 1 B)

                auto run = [&](bool concurrent)->double{
                    std::vector<double> t;
                    for(int trial=0; trial<5; ++trial){
                        cudaEvent_t a,b; CU(cudaEventCreate(&a)); CU(cudaEventCreate(&b));
                        CU(cudaDeviceSynchronize()); CU(cudaEventRecord(a));
                        for(int r=0;r<reps;++r)
                            for(int i=0;i<W;++i){
                                Lane& L=lanes[i]; int c=(L.cur++)%L.ncopy;
                                fp8_block_gemm(L.C, L.A, L.As, L.W[c], L.S[c], M, sh.N, sh.K,
                                               concurrent ? st[i] : st[0]);
                            }
                        CU(cudaEventRecord(b)); CU(cudaEventSynchronize(b));
                        float ms=0; CU(cudaEventElapsedTime(&ms,a,b));
                        CU(cudaDeviceSynchronize());
                        t.push_back(ms/reps);
                        cudaEventDestroy(a); cudaEventDestroy(b);
                    }
                    std::sort(t.begin(),t.end()); return t[t.size()/2];
                };
                // Warm the code paths, then measure. Serial first so any clock ramp favours the
                // concurrent arm's rival, not the hypothesis.
                run(false);
                double s1 = run(false), sw = run(true);
                printf("%-22s %5d %10.1f GB/s %10.1f GB/s %7.2fx\n",
                       si==0||true ? sh.name : "", W, gb/(s1/1e3), gb/(sw/1e3), s1/sw);
                fflush(stdout);
                for(int i=0;i<W;++i){ free_lane(lanes[i]); CU(cudaStreamDestroy(st[i])); }
            }
        }
        printf("\n");
    }
    printf("If W streams >> 1 stream, the engine's serialised layer chain is leaving that factor on\n"
           "the floor every layer, and independent chains within a layer should run concurrently.\n");
    return 0;
}
