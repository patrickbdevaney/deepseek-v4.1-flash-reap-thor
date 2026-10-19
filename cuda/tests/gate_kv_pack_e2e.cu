// gate_kv_pack_e2e.cu — DECODE_LADDER 1b.2, END TO END, with NO checkpoint.
//
// `gate_kv_pack` next door proves the FORMAT (round trip) and the READER (`sparse_attn` over a
// packed cache). Neither says anything about the WIRING -- the thirty-odd call sites where a row
// stride, a memcpy length, a staging buffer or a device-index append had to change. A wrong stride
// in one of them produces a plausible number, not a crash, and the only instrument that would have
// caught it was a 15-minute checkpoint load.
//
// So this drives the REAL decode functions on synthetic weights -- `compressed_attn_cache[_r4]`,
// `compressed_decode_step_{strided,indexer}`, `compressed_verify_step_{strided,indexer}`, which
// between them cover `build_qKV`, `compressor_emit_group`, the kv_all concatenation and
// `sparse_attn_kv` -- ONCE with the FP32 cache and ONCE with the packed cache, in the same process
// on the same inputs, and memcmps the outputs. Equality, not cosine: the claim is that packing is a
// storage refactor, so a single differing bit is a failure.
//
// The two arms run in the order (fp32, packed) and then AGAIN as (packed, fp32) under --swap, which
// is the control for state left behind in a static or an allocator between arms.
#include "compressed_attn.h"
#include "compressed_decode.h"
#include "deepseek_v4.h"
#include "kv_pack.h"
#include <cuda_runtime.h>
#include <vector>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdlib>
using namespace dsv4;
#define CU(x) do{cudaError_t e=(x); if(e){fprintf(stderr,"cuda %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)
static uint8_t rfp8(){ return (uint8_t)((rand()%0x40) | ((rand()&1)<<7)); }
static const uint8_t* upW(size_t n){ std::vector<uint8_t> h(n); for(auto&v:h)v=rfp8();
    uint8_t* d; CU(cudaMalloc(&d,n)); CU(cudaMemcpy(d,h.data(),n,cudaMemcpyHostToDevice)); return d; }
static const float* upS(size_t n){ std::vector<float> h(n); for(auto&v:h)v=0.3f+0.01f*(rand()%40);
    float* d; CU(cudaMalloc(&d,n*4)); CU(cudaMemcpy(d,h.data(),n*4,cudaMemcpyHostToDevice)); return d; }
static const float* upFv(std::vector<float>& h){ float* d; CU(cudaMalloc(&d,h.size()*4));
    CU(cudaMemcpy(d,h.data(),h.size()*4,cudaMemcpyHostToDevice)); return d; }
static const float* upR(size_t n, float sc){ std::vector<float> h(n); for(auto&v:h)v=sc*((rand()%200)-100)/100.f; return upFv(h); }
static const float* upNorm(int n){ std::vector<float> v(n); for(auto&e:v)e=0.5f+0.01f*(rand()%100); return upFv(v); }

struct Case { int ratio, s0, ndec, K; };

// One arm: build the caches from scratch, prefill s0, run `ndec` decode steps, then one verify step
// of width K. Returns every output row it produced, concatenated.
static std::vector<float> run_arm(bool packed, const CompressedAttnWeights& w, const float* x,
                                  const Case& C, int seqmax) {
    g_kv_pack = packed ? 1 : 0;
    g_kv_rowf = packed ? dsv4kv::KVP_ROWF : HEAD_DIM;
    const int ihd = w.index_head_dim, Tmax = seqmax / C.ratio + 2;
    float *win_kv, *comp_kv, *idx_ckv, *out;
    CU(cudaMalloc(&win_kv, kv_rows_bytes(seqmax)));
    CU(cudaMalloc(&comp_kv, kv_rows_bytes(Tmax)));
    CU(cudaMalloc(&idx_ckv, (size_t)Tmax*ihd*4));
    CU(cudaMalloc(&out, (size_t)(C.ndec + C.K)*DIM*4));
    CU(cudaMemset(win_kv, 0, kv_rows_bytes(seqmax)));
    CU(cudaMemset(comp_kv, 0, kv_rows_bytes(Tmax)));
    CU(cudaMemset(idx_ckv, 0, (size_t)Tmax*ihd*4));
    int Th = 0;
    if (C.ratio == 4) compressed_attn_cache_r4(win_kv, comp_kv, idx_ckv, &Th, x, w, C.s0, C.ratio, EPS);
    else              compressed_attn_cache   (win_kv, comp_kv,          &Th, x, w, C.s0, C.ratio, EPS);
    for (int i = 0; i < C.ndec; ++i) {
        const int pos = C.s0 + i;
        if (C.ratio == 4) compressed_decode_step_indexer(out + (size_t)i*DIM, x + (size_t)pos*DIM, x, pos, w,
                                                         win_kv, comp_kv, idx_ckv, &Th, C.ratio, EPS);
        else              compressed_decode_step_strided(out + (size_t)i*DIM, x + (size_t)pos*DIM, x, pos, w,
                                                         win_kv, comp_kv,          &Th, C.ratio, EPS);
    }
    {   // one M=K verify block, which is the shape the engine spends most of its time in
        const int pos = C.s0 + C.ndec;
        if (C.ratio == 4) compressed_verify_step_indexer(out + (size_t)C.ndec*DIM, x + (size_t)pos*DIM, x, pos, C.K, w,
                                                         win_kv, comp_kv, idx_ckv, &Th, C.ratio, EPS);
        else              compressed_verify_step_strided(out + (size_t)C.ndec*DIM, x + (size_t)pos*DIM, x, pos, C.K, w,
                                                         win_kv, comp_kv,          &Th, C.ratio, EPS);
    }
    CU(cudaDeviceSynchronize());
    std::vector<float> h((size_t)(C.ndec + C.K)*DIM);
    CU(cudaMemcpy(h.data(), out, h.size()*4, cudaMemcpyDeviceToHost));
    CU(cudaFree(win_kv)); CU(cudaFree(comp_kv)); CU(cudaFree(idx_ckv)); CU(cudaFree(out));
    return h;
}

int main(int argc, char** argv){
    bool swap = false;
    for (int i=1;i<argc;++i) if(!strcmp(argv[i],"--swap")) swap = true;
    const int half = ROPE_DIM/2, Kd = N_HEADS*HEAD_DIM, GKd = Kd/O_GROUPS, OB = O_GROUPS*O_LORA;
    const int nH = INDEX_N_HEADS, ihd = INDEX_HEAD_DIM, QD = nH*ihd, iod = 2*ihd;
    // s must clear WINDOW so the window is saturated and the compressed rows are actually selected.
    const int seqmax = 640, s = 512;
    Case cases[] = { {4, 480, 6, 2}, {4, 480, 6, 6}, {128, 480, 6, 2} };

    int fails = 0;
    for (const Case& C : cases) {
        srand(31 + C.ratio + C.K);
        const int ratio = C.ratio, T = seqmax/ratio + 2;
        CompressedAttnWeights w{}; MLAWeights& a = w.attn;
        a.wq_a=upW((size_t)Q_LORA*DIM);   a.wq_a_s=upS((size_t)(Q_LORA/128)*(DIM/128));
        a.wq_b=upW((size_t)Kd*Q_LORA);    a.wq_b_s=upS((size_t)(Kd/128)*(Q_LORA/128));
        a.wkv =upW((size_t)HEAD_DIM*DIM); a.wkv_s =upS((size_t)(HEAD_DIM/128)*(DIM/128));
        a.wo_b=upW((size_t)DIM*OB);       a.wo_b_s=upS((size_t)(DIM/128)*(OB/128));
        a.q_norm=upNorm(Q_LORA); a.kv_norm=upNorm(HEAD_DIM);
        a.wo_a=upR((size_t)O_GROUPS*O_LORA*GKd,0.02f); a.attn_sink=upR(N_HEADS,0.1f);
        std::vector<float> cq((size_t)seqmax*half), sq((size_t)seqmax*half);
        for(int p=0;p<seqmax;++p) for(int j=0;j<half;++j){ float ang=p*0.011f*(j+1); cq[p*half+j]=cosf(ang); sq[p*half+j]=sinf(ang); }
        a.cosT=upFv(cq); a.sinT=upFv(sq);
        const int mcoff = (ratio==4) ? 2 : 1;             // overlap only for ratio 4
        w.mc_wkv=upR((size_t)mcoff*HEAD_DIM*DIM,0.02f); w.mc_wgate=upR((size_t)mcoff*HEAD_DIM*DIM,0.02f);
        w.mc_ape=upR((size_t)ratio*mcoff*HEAD_DIM,0.1f); w.mc_norm=upNorm(HEAD_DIM);
        std::vector<float> cc((size_t)T*half), cs((size_t)T*half);
        for(int t=0;t<T;++t) for(int j=0;j<half;++j){ float ang=t*0.019f*(j+1); cc[t*half+j]=cosf(ang); cs[t*half+j]=sinf(ang); }
        w.cc_cos=upFv(cc); w.cc_sin=upFv(cs);
        w.idx_wq_b=upW((size_t)QD*Q_LORA); w.idx_wq_b_s=upS((size_t)(QD/128)*(Q_LORA/128));
        w.idx_weights_proj=upR((size_t)nH*DIM,0.02f);
        w.idx_c_wkv=upR((size_t)iod*DIM,0.02f); w.idx_c_wgate=upR((size_t)iod*DIM,0.02f);
        w.idx_c_ape=upR((size_t)ratio*iod,0.1f); w.idx_c_norm=upNorm(ihd);
        w.index_n_heads=nH; w.index_head_dim=ihd; w.index_topk=INDEX_TOPK;
        std::vector<float> xh((size_t)seqmax*DIM); for(auto&e:xh)e=0.1f*((rand()%200)-100)/100.f;
        const float* x = upFv(xh);
        (void)s;

        std::vector<float> A, B;
        if (swap) { B = run_arm(true, w, x, C, seqmax);  A = run_arm(false, w, x, C, seqmax); }
        else      { A = run_arm(false, w, x, C, seqmax); B = run_arm(true, w, x, C, seqmax); }
        size_t diff = 0, first = (size_t)-1;
        for (size_t i=0;i<A.size();++i) if (memcmp(&A[i],&B[i],4)) { if(first==(size_t)-1) first=i; ++diff; }
        printf("ratio=%-3d s0=%d decode=%d verify_K=%d  %zu / %zu floats differ%s\n",
               C.ratio, C.s0, C.ndec, C.K, diff, A.size(), diff? "   <<<< NOT BIT-EXACT":"");
        if (diff) { printf("   first at %zu (row %zu col %zu): fp32 %.9g  packed %.9g\n",
                           first, first/DIM, first%DIM, A[first], B[first]); ++fails; }
    }
    // Reset so a later caller in the same process is not surprised by a static.
    g_kv_pack = 0; g_kv_rowf = HEAD_DIM;
    printf("%s  (arm mismatches %d)%s\n", fails? "GATE KV_PACK_E2E: FAIL":"GATE KV_PACK_E2E: PASS",
           fails, swap? "  [--swap arm order]":"");
    return fails ? 1 : 0;
}
