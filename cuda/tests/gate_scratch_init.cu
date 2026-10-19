// gate_scratch_init.cu — does the prefill attention chain read scratch it never wrote, and WHICH one?
//
// Finding 60. The engine is nondeterministic on byte-identical input: 36 identical sweep points give
// 36 distinct first-verify margin vectors. Zeroing the prefill's raw cudaMalloc scratch moves that to
// 5/8 on an 8-point run, so those buffers are a real contributor — but zeroing MASKS the defect
// instead of naming it, and a 15-minute checkpoint load per hypothesis is no way to find a kernel.
//
// This finds it at unit level in seconds. `compressed_attn_forward` allocates its scratch through
// `zalloc`, which can fill each allocation with an arbitrary byte pattern. Run the same forward on
// the same weights and the same input twice, with two DIFFERENT poison patterns:
//
//   outputs bitwise equal      -> nothing in the chain depends on uninitialised scratch
//   outputs differ             -> something does, and the run is not reproducible
//
// Then localise: poison exactly ONE allocation (by its sequence number within the call) and leave the
// rest zeroed. The allocations whose poison changes the output are the ones being read before being
// written. That names the buffer, and the buffer names the kernel.
//
// Weights are synthetic and the setup mirrors tests/gate_prefill_len.cu, including the +4 byte
// alignment offset (Finding 41) — real weights are 4-byte-aligned pointers into a mapped file and
// cudaMalloc is always 256, so a gate that allocates its own inputs cannot see alignment bugs.
//
//   build: see scripts/build_gate.sh
//   run:   ./build/gate_scratch_init
#include "compressed_attn.h"
#include "mla_attn.h"
#include "deepseek_v4.h"
#include "dscratch.h"
#include <cuda_runtime.h>
#include <vector>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <cstdlib>
using namespace dsv4;
#define CU(x) do{cudaError_t e=(x); if(e){fprintf(stderr,"cuda %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)

static int g_off = 0;
static void* upBytes(const void* h, size_t n){
    char* d; CU(cudaMalloc((void**)&d, n + 8)); CU(cudaMemcpy(d + g_off, h, n, cudaMemcpyHostToDevice));
    return (void*)(d + g_off);
}
static uint8_t rfp8(){ return (uint8_t)((rand()%0x40) | ((rand()&1)<<7)); }
static const uint8_t* upW(size_t n){ std::vector<uint8_t> h(n); for(auto&v:h)v=rfp8(); return (const uint8_t*)upBytes(h.data(),n); }
static const float* upS(size_t n){ std::vector<float> h(n); for(auto&v:h)v=0.3f+0.01f*(rand()%40); return (const float*)upBytes(h.data(),n*4); }
static const float* upFv(std::vector<float>& h){ return (const float*)upBytes(h.data(),h.size()*4); }
static const float* upR(size_t n, float sc){ std::vector<float> h(n); for(auto&v:h)v=sc*((rand()%200)-100)/100.f; return upFv(h); }
static const float* upNorm(size_t n){ std::vector<float> h(n); for(auto&v:h)v=0.8f+0.004f*(rand()%100); return upFv(h); }

static std::vector<float> down(const float* d, size_t n){
    std::vector<float> h(n); CU(cudaMemcpy(h.data(), d, n*4, cudaMemcpyDeviceToHost)); return h;
}
// Bitwise: the claim is reproducibility, not closeness. NaN must compare equal to NaN here, so
// memcmp, not ==.
static size_t ndiff(const std::vector<float>& a, const std::vector<float>& b, size_t& first){
    size_t n=0; first=(size_t)-1;
    for(size_t i=0;i<a.size();++i) if(memcmp(&a[i],&b[i],4)!=0){ if(!n) first=i; ++n; }
    return n;
}

int main(int argc, char** argv){
    const int s = argc>1 ? atoi(argv[1]) : 17;        // PSp of the canonical 18-id prompt
    const int ratio4 = 4, half = ROPE_DIM/2;
    const int Kd = N_HEADS*HEAD_DIM, GKd = Kd/O_GROUPS, OB = O_GROUPS*O_LORA;
    const int nH = INDEX_N_HEADS, ihd = INDEX_HEAD_DIM, QD = nH*ihd, iod = 2*ihd;
    const int d = DIM, T4 = s/ratio4;
    int fails = 0;

    g_off = 4;                                        // real weights are 4-byte aligned, not 256
    srand(31);
    CompressedAttnWeights w{}; MLAWeights& a = w.attn;
    a.wq_a=upW((size_t)Q_LORA*DIM);   a.wq_a_s=upS((size_t)(Q_LORA/128)*(DIM/128));
    a.wq_b=upW((size_t)Kd*Q_LORA);    a.wq_b_s=upS((size_t)(Kd/128)*(Q_LORA/128));
    a.wkv =upW((size_t)HEAD_DIM*DIM); a.wkv_s =upS((size_t)(HEAD_DIM/128)*(DIM/128));
    a.wo_b=upW((size_t)DIM*OB);       a.wo_b_s=upS((size_t)(DIM/128)*(OB/128));
    a.q_norm=upNorm(Q_LORA); a.kv_norm=upNorm(HEAD_DIM);
    a.wo_a=upR((size_t)O_GROUPS*O_LORA*GKd,0.02f); a.attn_sink=upR(N_HEADS,0.1f);
    std::vector<float> cq((size_t)s*half), sq((size_t)s*half);
    for(int p=0;p<s;++p) for(int j=0;j<half;++j){ float ang=p*0.011f*(j+1); cq[p*half+j]=cosf(ang); sq[p*half+j]=sinf(ang); }
    a.cosT=upFv(cq); a.sinT=upFv(sq);
    w.mc_wkv=upR((size_t)2*HEAD_DIM*DIM,0.02f); w.mc_wgate=upR((size_t)2*HEAD_DIM*DIM,0.02f);
    w.mc_ape=upR((size_t)ratio4*2*HEAD_DIM,0.1f); w.mc_norm=upNorm(HEAD_DIM);
    std::vector<float> cc((size_t)(T4?T4:1)*half), cs((size_t)(T4?T4:1)*half);
    for(int t=0;t<(T4?T4:1);++t) for(int j=0;j<half;++j){ float ang=t*0.019f*(j+1); cc[t*half+j]=cosf(ang); cs[t*half+j]=sinf(ang); }
    w.cc_cos=upFv(cc); w.cc_sin=upFv(cs);
    w.idx_wq_b=upW((size_t)QD*Q_LORA); w.idx_wq_b_s=upS((size_t)(QD/128)*(Q_LORA/128));
    w.idx_weights_proj=upR((size_t)nH*DIM,0.02f);
    w.idx_c_wkv=upR((size_t)iod*DIM,0.02f); w.idx_c_wgate=upR((size_t)iod*DIM,0.02f);
    w.idx_c_ape=upR((size_t)ratio4*iod,0.1f); w.idx_c_norm=upNorm(ihd);
    w.index_n_heads=nH; w.index_head_dim=ihd; w.index_topk=INDEX_TOPK;

    std::vector<float> xh((size_t)s*d); for(auto&e:xh) e = 0.05f*((rand()%200)-100)/100.f;
    const float* x = upFv(xh);
    float* out; CU(cudaMalloc(&out,(size_t)s*DIM*4));

    // THE ARENA IS PART OF THE PREFILL PATH TOO, and the first version of this gate missed it.
    // `compressor_forward` — called by both compressed_attn_forward and indexer_forward — allocates
    // its `kv`/`score` through dmalloc, not through the raw cudaMalloc that `zalloc` wraps. With no
    // arena_init the gate left g_arena_on false, so dmalloc fell back to a FRESH cudaMalloc and the
    // gate exercised an allocation path the engine does not take. Turn the arena on and poison it
    // too, so "scratch" here means every buffer the engine's prefill actually reuses.
    arena_init((size_t)64<<20);

    // One forward under a chosen poison regime.
    auto run = [&](int idx, unsigned char val)->std::vector<float>{
        CU(cudaMemset(out, 0xA5, (size_t)s*DIM*4));       // so an unwritten OUTPUT is also caught
        if(idx == -1 && g_arena) CU(cudaMemset(g_arena, val, g_arena_cap));   // arena carries the same poison
        arena_reset();
        g_scratch_poison_idx = idx; g_scratch_poison_val = val; g_scratch_alloc_seq = 0;
        compressed_attn_forward(out, x, w, s, WINDOW, ratio4, EPS, 0);
        CU(cudaDeviceSynchronize());
        g_scratch_poison_idx = -2;
        return down(out,(size_t)s*DIM);
    };

    printf("gate_scratch_init: compressed_attn_forward, s=%d, ratio=4, weight offset +%d\n", s, g_off);

    // --- 1. control: the same poison twice must give the same answer ---
    std::vector<float> c1 = run(-1, 0x00), c2 = run(-1, 0x00);
    size_t at; size_t nd = ndiff(c1,c2,at);
    printf("  control  (poison 0x00 twice)      : %s (%zu diffs)\n", nd?"FAIL — nondeterministic even at fixed poison":"ok", nd);
    if(nd) ++fails;

    // --- 2. the question: does the poison VALUE change the answer? ---
    std::vector<float> pA = run(-1, 0x00), pB = run(-1, 0xFF), pC = run(-1, 0x3C);
    size_t nAB = ndiff(pA,pB,at); size_t atAB = at;
    size_t nAC = ndiff(pA,pC,at);
    const int total = g_scratch_alloc_seq;   // allocations the last call made
    printf("  poison 0x00 vs 0xFF              : %zu of %zu outputs differ%s\n",
           nAB, pA.size(), nAB?"":"  -> no dependence");
    printf("  poison 0x00 vs 0x3C              : %zu of %zu outputs differ\n", nAC, pA.size());
    if(nAB) printf("  first differing output element   : %zu (%.9g vs %.9g)\n", atAB, pA[atAB], pB[atAB]);

    if(!nAB && !nAC){
        printf("\n  Nothing in this chain reads uninitialised scratch at s=%d.\n", s);
        printf("gate_scratch_init: %s\n", fails?"GATE FAIL":"GATE PASS (no uninitialised read here)");
        return fails?1:0;
    }

    // --- 3. localise: poison ONE allocation at a time, zero the rest ---
    ++fails;
    printf("\n  localising over %d scratch allocations in the call (zeroed except the named one):\n", total);
    for(int k=0; k<total; ++k){
        std::vector<float> zk = run(k, 0x00), fk = run(k, 0xFF);
        size_t w2; size_t n = ndiff(zk,fk,w2);
        if(n) printf("    alloc #%-2d : %6zu outputs change  <== READ BEFORE WRITTEN (first at %zu)\n", k, n, w2);
    }
    printf("\n  Map the index to the buffer by reading the zalloc() call order in\n"
           "  kernels/compressed_attn.cu (then kernels/indexer.cu, which allocates inside this call).\n");
    printf("gate_scratch_init: GATE FAIL (uninitialised read present)\n");
    return 1;
}
