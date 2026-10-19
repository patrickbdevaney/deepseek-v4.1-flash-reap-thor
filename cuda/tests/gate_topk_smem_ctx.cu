// gate_topk_smem_ctx.cu — DECODE_LADDER item 1.4's "one leg above the ceiling", run through the
// ENGINE'S OWN decode entry point rather than through an isolated kernel.
//
// WHY THIS EXISTS AND WHY IT IS NOT tests/gate_topk_radix.cu. That gate launches the four scan
// kernels directly, which proves the kernels tolerate T above the 49,152 B default. It does not
// prove the thing the item is actually about: that `compressed_decode_step_indexer` -- the function
// the server calls once per decode step, with its arena, its dmalloc'd (UNZEROED) index buffer, its
// sparse_attn consumer downstream -- survives a context past the ceiling. Those are different
// claims, and the second one is the one a user of this engine cares about.
//
// WHY IT IS NOT THE SERVER EITHER, which would be the strongest form and is not available. Reaching
// context 49,140 in `build/dsv4-server` means `--seqmax 49152`, and the engine's seqmax-scaling
// allocations are 134,276 B/token (win_kv 88,064 + comp_kv 20,992 + main_x 16,384 + mkv 6,144 +
// idx_ckv 2,688 + d_ids 4; src/engine.cu:332-433 against the constants in deepseek_v4.h). That is
// 6.15 GiB at seqmax 49,152 against 2.05 GiB at the 16,384 the engine runs today -- 4.10 GiB more,
// on a box whose engine reports `mem 119.1/122.8 GiB` at seqmax 16,384, i.e. 3.7 GiB free. It does
// not fit, and this box does not OOM gracefully (scripts/memguard.sh: two whole-machine takedowns).
// So the leg is run here, where the same function is driven at the same T for ~140 MB of KV and no
// checkpoint at all.
//
// WHAT IS SYNTHETIC AND WHAT IS NOT. The weights and the three KV caches are seeded pseudo-random,
// so the OUTPUT VALUES are meaningless. That is fine and it is the point: both arms see byte-
// identical inputs, so the comparison is exact, and every kernel, allocation and launch
// configuration on the path is the real one. `pos` is chosen with (pos+1) % ratio != 0 so no group
// emit fires and the compressor never reads `x_full` -- the leg is the decode step, not the
// compressor. What this leg tests is the top-k selection at T > 12,285 and everything downstream of
// it consuming the result.
//
//   build: nvcc -O2 -std=c++17 -gencode arch=compute_110a,code=sm_110a -I include \
//     tests/gate_topk_smem_ctx.cu kernels/compressed_decode.cu kernels/compressed_attn.cu \
//     kernels/compressor.cu kernels/indexer.cu kernels/mla_attn.cu kernels/fp8_block_gemm.cu \
//     kernels/tc_fp8_gemm.cu kernels/dscratch.cu kernels/dprof.cu -o build/gate_topk_smem_ctx
//
//   run:   build/gate_topk_smem_ctx <context> <tag>
//   arms are selected by environment, one arm per process, because DSV4_* are cached in statics:
//     DSV4_TOPK_RADIX=1                        the shipped path (no dynamic shared memory at all)
//     DSV4_TOPK_RADIX=0                        the fallback scan arm -- the one 1.4 repairs
//     DSV4_TOPK_RADIX=1 DSV4_TOPK_GATE=1       shipped path + in-situ reference (also repaired)
//     DSV4_TOPK_SMEM_OPTIN=0                   pre-1.4 behaviour restored, for the before-arm
#include "compressed_attn.h"
#include "compressed_decode.h"
#include "deepseek_v4.h"
#include "indexer.h"
#include <cuda_runtime.h>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
using namespace dsv4;
#define CU(x) do{cudaError_t e_=(x); if(e_){fprintf(stderr,"cuda %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e_));exit(1);} }while(0)

// One deterministic stream for every buffer in both arms. Not rand(): rand() is global state and a
// gate whose inputs depend on how many times it was called earlier is a gate that cannot be
// compared across arms.
struct Rng { unsigned long long s; explicit Rng(unsigned long long seed):s(seed){}
    unsigned next(){ s ^= s<<13; s ^= s>>7; s ^= s<<17; return (unsigned)(s>>32); }
    float unit(){ return (float)(next() % 200001) / 100000.f - 1.f; } };            // [-1,1]

static const uint8_t* upW(size_t n, Rng& r){ std::vector<uint8_t> h(n);
    for(auto&v:h) v = (uint8_t)((r.next()%0x40) | ((r.next()&1)<<7));
    uint8_t* d; CU(cudaMalloc(&d,n)); CU(cudaMemcpy(d,h.data(),n,cudaMemcpyHostToDevice)); return d; }
static const float* upFv(std::vector<float>& h){ float* d; CU(cudaMalloc(&d,h.size()*4));
    CU(cudaMemcpy(d,h.data(),h.size()*4,cudaMemcpyHostToDevice)); return d; }
static const float* upS(size_t n, Rng& r){ std::vector<float> h(n); for(auto&v:h)v=0.3f+0.2f*(r.unit()+1.f); return upFv(h); }
static const float* upR(size_t n, float sc, Rng& r){ std::vector<float> h(n); for(auto&v:h)v=sc*r.unit(); return upFv(h); }
static const float* upNorm(size_t n, Rng& r){ std::vector<float> h(n); for(auto&v:h)v=0.5f+0.5f*(r.unit()+1.f); return upFv(h); }

static unsigned long long fnv(const void* p, size_t n){
    const unsigned char* b=(const unsigned char*)p; unsigned long long h=1469598103934665603ull;
    for(size_t i=0;i<n;++i){ h ^= b[i]; h *= 1099511628211ull; } return h; }

int main(int argc, char** argv){
    const int s     = argc>1 ? atoi(argv[1]) : 49208;
    const char* tag = argc>2 ? argv[2] : "arm";
    const int ratio = 4, half = ROPE_DIM/2;
    const int Kd = N_HEADS*HEAD_DIM;
    const int nH = 32, ihd = INDEX_HEAD_DIM, QD = nH*ihd, iod = 2*ihd;

    // pos is the decode position; T is how many compressed rows the caches already hold. Both arms
    // get the same pair. (pos+1)%ratio must be nonzero so compressor_emit_group never runs.
    int pos = s - 1; if((pos+1) % ratio == 0) --pos;
    int T   = (pos+1) / ratio;
    const size_t need = topk_scan_smem(T);
    int lim_default = 0, lim_optin = 0; { int dev=0; cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&lim_default, cudaDevAttrMaxSharedMemoryPerBlock, dev);
        cudaDeviceGetAttribute(&lim_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev); }
    printf("[smem-ctx %s] context=%d pos=%d T=%d  scan smem would be %zu B "
           "(device default %d B, opt-in %d B) -> %s the default ceiling\n",
           tag, pos+1, pos, T, need, lim_default, lim_optin,
           need > (size_t)lim_default ? "ABOVE" : "below");
    fflush(stdout);

    Rng r(0x9E3779B97F4A7C15ull);
    CompressedAttnWeights w{}; MLAWeights& a = w.attn;
    const int GKd = Kd/O_GROUPS, OB = O_GROUPS*O_LORA;
    a.wq_a=upW((size_t)Q_LORA*DIM,r);   a.wq_a_s=upS((size_t)(Q_LORA/128)*(DIM/128),r);
    a.wq_b=upW((size_t)Kd*Q_LORA,r);    a.wq_b_s=upS((size_t)(Kd/128)*(Q_LORA/128),r);
    a.wkv =upW((size_t)HEAD_DIM*DIM,r); a.wkv_s =upS((size_t)(HEAD_DIM/128)*(DIM/128),r);
    a.wo_b=upW((size_t)DIM*OB,r);       a.wo_b_s=upS((size_t)(DIM/128)*(OB/128),r);
    a.q_norm=upNorm(Q_LORA,r); a.kv_norm=upNorm(HEAD_DIM,r);
    a.wo_a=upR((size_t)O_GROUPS*O_LORA*GKd,0.02f,r);
    a.attn_sink=upR(N_HEADS,0.1f,r);
    // Position tables must cover pos (query) and T (compressed). Full-length, as the engine has them.
    { std::vector<float> cq((size_t)(pos+1)*half), sq((size_t)(pos+1)*half);
      for(int p=0;p<=pos;++p) for(int j=0;j<half;++j){ float ang=p*0.011f*(j+1); cq[(size_t)p*half+j]=cosf(ang); sq[(size_t)p*half+j]=sinf(ang); }
      a.cosT=upFv(cq); a.sinT=upFv(sq); }
    { std::vector<float> cc((size_t)(T+2)*half), cs((size_t)(T+2)*half);
      for(int t=0;t<T+2;++t) for(int j=0;j<half;++j){ float ang=t*0.019f*(j+1); cc[(size_t)t*half+j]=cosf(ang); cs[(size_t)t*half+j]=sinf(ang); }
      w.cc_cos=upFv(cc); w.cc_sin=upFv(cs); }
    w.mc_wkv=upR((size_t)2*HEAD_DIM*DIM,0.02f,r); w.mc_wgate=upR((size_t)2*HEAD_DIM*DIM,0.02f,r);
    w.mc_ape=upR((size_t)ratio*2*HEAD_DIM,0.1f,r); w.mc_norm=upNorm(HEAD_DIM,r);
    w.idx_wq_b=upW((size_t)QD*Q_LORA,r); w.idx_wq_b_s=upS((size_t)(QD/128)*(Q_LORA/128),r);
    w.idx_weights_proj=upR((size_t)nH*DIM,0.02f,r);
    w.idx_c_wkv=upR((size_t)iod*DIM,0.02f,r); w.idx_c_wgate=upR((size_t)iod*DIM,0.02f,r);
    w.idx_c_ape=upR((size_t)ratio*iod,0.1f,r); w.idx_c_norm=upNorm(ihd,r);
    w.index_n_heads=nH; w.index_head_dim=ihd; w.index_topk=INDEX_TOPK;

    // The three caches, pre-filled. This is the substitute for a 49k-token prefill: the decode step
    // reads them, it does not care how they were produced, and both arms read identical bytes.
    float *win_kv=nullptr, *comp_kv=nullptr, *idx_ckv=nullptr;
    CU(cudaMalloc(&win_kv, (size_t)(pos+1)*HEAD_DIM*4));
    CU(cudaMalloc(&comp_kv,(size_t)(T+2)*HEAD_DIM*4));
    CU(cudaMalloc(&idx_ckv,(size_t)(T+2)*ihd*4));
    { // only the sliding window [pos-WINDOW+1 .. pos] of win_kv is ever read; fill it all anyway
      std::vector<float> h((size_t)(pos+1)*HEAD_DIM); for(auto&e:h) e=0.05f*r.unit();
      CU(cudaMemcpy(win_kv,h.data(),h.size()*4,cudaMemcpyHostToDevice)); }
    { std::vector<float> h((size_t)(T+2)*HEAD_DIM); for(auto&e:h) e=0.05f*r.unit();
      CU(cudaMemcpy(comp_kv,h.data(),h.size()*4,cudaMemcpyHostToDevice)); }
    { std::vector<float> h((size_t)(T+2)*ihd); for(auto&e:h) e=0.05f*r.unit();
      CU(cudaMemcpy(idx_ckv,h.data(),h.size()*4,cudaMemcpyHostToDevice)); }
    // x_cur is the one attention-input row the step actually reads. x_full is passed as the same
    // pointer and is never dereferenced, because no group emit fires at this pos.
    std::vector<float> xh(DIM); for(auto&e:xh) e=0.1f*r.unit();
    const float* x_cur = upFv(xh);

    float* out; CU(cudaMalloc(&out,(size_t)DIM*4)); CU(cudaMemset(out,0,(size_t)DIM*4));
    int Th = T;
    compressed_decode_step_indexer(out, x_cur, x_cur, pos, w, win_kv, comp_kv, idx_ckv, &Th, ratio, EPS);
    cudaError_t sync = cudaDeviceSynchronize();

    std::vector<float> ho(DIM);
    CU(cudaMemcpy(ho.data(), out, (size_t)DIM*4, cudaMemcpyDeviceToHost));
    int nz=0, nnan=0; for(int i=0;i<DIM;++i){ if(ho[i]!=0.f) ++nz; if(std::isnan(ho[i])) ++nnan; }
    printf("[smem-ctx %s] sync=%s  out: %d/%d nonzero, %d NaN, hash=%016llx\n",
           tag, cudaGetErrorString(sync), nz, DIM, nnan, fnv(ho.data(), (size_t)DIM*4));
    fflush(stdout);
    return sync == cudaSuccess ? 0 : 1;
}
