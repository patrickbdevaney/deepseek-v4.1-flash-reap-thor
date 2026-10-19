// ncu_target.cu — a minimal, single-purpose profiling target for the two kernels that are now the
// whole gap to the roofline (LOOP_LOG Finding 46): the MoE grouped FP4 GEMV and the ogroup FP8 GEMV.
//
// gemm_bench launches these hundreds of times across an M x NR sweep, which makes `ncu -k ... -c N`
// land on an arbitrary configuration. This runs ONE launch of each shape we care about, in a fixed
// order, so the ncu report maps to a named row:
//
//   0  moe   M=1   (base AR)        3  ogroup M=1   (base AR)
//   1  moe   M=5   (verify)         4  ogroup M=5   (verify, NR=8)
//
// Weights are rotated over enough copies that nothing is served from cache, matching the in-situ
// condition (every layer's weights come from a 100 GiB working set, always cold).
//   build: see scripts/ncu_probe.sh
#include "deepseek_v4.h"
#include "dscratch.h"
#include "mla_attn.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace dsv4;
#define CU(x) do{cudaError_t e=(x); if(e){fprintf(stderr,"cuda %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)
extern bool g_tc_fp8;
void tc_fp4_grouped_gemm_e8m0(float*, const __half*, const uint8_t* const*, const uint8_t* const*,
        const int*, const int*, const int*, const int*, int, int, int, cudaStream_t);
// The path the ENGINE takes: moe_forward picks the GEMV whenever the expert scales are native e8m0
// (g_moe_gemv defaults on). Profiling only the mma kernel would have measured a path decode does
// not use -- the same class of mistake as the stale GEMV_MK bench row in Finding 41.
void tc_fp4_grouped_gemv_e8m0(float*, const uint8_t*, const float*, const uint8_t* const*, const uint8_t* const*,
        const int*, const int*, const int*, const int*, int, int, int, cudaStream_t, int, int);
__global__ void k_build_tiles(int*, int*, int*, const int*, int);

static void* dalloc(size_t n){ void* p; CU(cudaMalloc(&p,n)); CU(cudaMemset(p,0x11,n)); return p; }

int main(int argc, char** argv){
    const int which = argc>1 ? atoi(argv[1]) : -1;    // -1 = all
    g_tc_fp8 = true;
    arena_init((size_t)512<<20);

    // ---------------- MoE: w1 [inter, dim] fp4, 160 experts, top-6 ----------------
    if(which<0 || which==0 || which==1){
        const int nr=N_ROUTED, inter=MOE_INTER, dim=DIM;
        const size_t w13n=(size_t)inter*(dim/2), w13s=(size_t)inter*(dim/32);
        std::vector<uint8_t*> hw(nr), hs(nr);
        for(int ex=0;ex<nr;++ex){ hw[ex]=(uint8_t*)dalloc(w13n); hs[ex]=(uint8_t*)dalloc(w13s); }
        const uint8_t **wptr,**sptr;
        CU(cudaMalloc(&wptr,nr*sizeof(void*))); CU(cudaMalloc(&sptr,nr*sizeof(void*)));
        CU(cudaMemcpy(wptr,hw.data(),nr*sizeof(void*),cudaMemcpyHostToDevice));
        CU(cudaMemcpy(sptr,hs.data(),nr*sizeof(void*),cudaMemcpyHostToDevice));
        for(int M : {1,5}){
            if(which>=0 && which!=(M==1?0:1)) continue;
            // MEASURED grouping, not the worst case (LOOP_LOG Finding 64). DSV4_MOEUNION=1 says the
            // K=5 verify activates 17.53 distinct experts over 30 rows, i.e. ~1.71 rows each — not 30
            // experts of 1 row. That distinction is the whole point now: with the row-amortised kernel
            // the weight is read once per EXPERT, so profiling 1-row experts measures a shape the
            // engine does not run and hides the amortisation entirely.
            const int ROWS = 6*M;                                 // bs*na rows to place
            const int U    = (M==1) ? 6 : 18;                     // measured distinct experts
            std::vector<int> hoff(nr+1,0);
            // The MEASURED histogram, not a flat spread (LOOP_LOG Finding 70). DSV4_MOEUNION=1 reports
            // rows/expert at K=5 as ~70% me=1, 18% me=2, 7% me=3, 2.4% me=4, 2.6% me=5 — max 5. The
            // previous version clamped `take` at 2, so no tile ever had me>2 and the RB sweep could
            // not see the chunking cost that RB is FOR: at RB=2 an expert with me=5 needs three
            // weight reads. Profiling a distribution the engine does not have is how F65 happened.
            // 18 experts / 30 rows: 12x1 + 3x2 + 1x3 + 1x4 + 1x5 = 30.
            { const int shape[18] = {1,1,1,1,1,1,1,1,1,1,1,1, 2,2,2, 3,4,5};
              int placed=0;
              for(int ex=0; ex<=nr; ++ex){
                  hoff[ex] = placed;
                  if(ex<U) placed += (M==1) ? 1 : shape[ex<18?ex:17]; }
              hoff[nr] = placed; }
            int *off_d,*tile_e,*tile_row0,*ntiles_d;
            CU(cudaMalloc(&off_d,(nr+1)*4)); CU(cudaMemcpy(off_d,hoff.data(),(nr+1)*4,cudaMemcpyHostToDevice));
            CU(cudaMalloc(&tile_e,(U+8)*4)); CU(cudaMalloc(&tile_row0,(U+8)*4)); CU(cudaMalloc(&ntiles_d,4));
            k_build_tiles<<<1,256>>>(tile_e,tile_row0,ntiles_d,off_d,nr);
            __half* x16=(__half*)dalloc((size_t)U*dim*2);
            uint8_t* xq=(uint8_t*)dalloc((size_t)U*dim);
            float* xs=(float*)dalloc((size_t)U*(dim/128)*4);
            float* out=(float*)dalloc((size_t)U*inter*4);
            CU(cudaDeviceSynchronize());
            printf("[ncu] moe M=%d U=%d rows=%d w=%.1f MB (rows/expert %.2f)\n", M, U, ROWS, U*(double)w13n/1e6, (double)ROWS/U);
            tc_fp4_grouped_gemv_e8m0(out,xq,xs,wptr,sptr,off_d,tile_e,tile_row0,ntiles_d,U+8,inter,dim, 0, M, getenv("A8")?atoi(getenv("A8")):1);  // engine default
            CU(cudaDeviceSynchronize());
            tc_fp4_grouped_gemm_e8m0(out,x16,wptr,sptr,off_d,tile_e,tile_row0,ntiles_d,U+8,inter,dim,0);    // mma, for comparison
            CU(cudaDeviceSynchronize());
            CU(cudaFree(xq)); CU(cudaFree(xs));
            CU(cudaFree(off_d)); CU(cudaFree(tile_e)); CU(cudaFree(tile_row0)); CU(cudaFree(ntiles_d));
            CU(cudaFree(x16)); CU(cudaFree(out));
        }
        for(int ex=0;ex<nr;++ex){ CU(cudaFree(hw[ex])); CU(cudaFree(hs[ex])); }
        CU(cudaFree(wptr)); CU(cudaFree(sptr));
    }

    // ---------------- ogroup: wo_a [G*R, Kd] fp8 ----------------
    if(which<0 || which==3 || which==4){
        const int G=O_GROUPS, R=O_LORA, Kd=(N_HEADS*HEAD_DIM)/O_GROUPS;   // [8, 1024, 4096] = 33.5 MB
        const size_t wb=(size_t)G*R*Kd;
        uint8_t* W=(uint8_t*)dalloc(wb);
        uint8_t* SC=(uint8_t*)dalloc((size_t)(G*R/128)*(Kd/128));
        for(int M : {1,5}){
            if(which>=0 && which!=(M==1?3:4)) continue;
            float* o=(float*)dalloc((size_t)M*G*Kd*4);
            float* out=(float*)dalloc((size_t)M*G*R*4);
            CU(cudaDeviceSynchronize());
            printf("[ncu] ogroup M=%d w=%.1f MB act=%.1f MB\n", M, wb/1e6, (double)M*G*Kd*4/1e6);
            ogroup_gemm_fp8(out,o,W,SC,M,G,R,Kd,0);
            CU(cudaDeviceSynchronize());
            CU(cudaFree(o)); CU(cudaFree(out));
        }
        CU(cudaFree(W)); CU(cudaFree(SC));
    }
    return 0;
}
