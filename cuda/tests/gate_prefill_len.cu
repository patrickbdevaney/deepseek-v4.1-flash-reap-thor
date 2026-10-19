// gate_prefill_len.cu — PREFIX-INVARIANCE gate for the prefill attention chain across prompt LENGTHS.
//
// Why this gate exists (LOOP_LOG Finding 52, lever I2). Cycle 3's multi-prompt sweep reproduced
// byte-identically on three of four prompts and NOT on the fourth — the only one whose length equals
// SMAX (s=18, so the prefill runs at PSp=17) — and then died with
//     cuda kernels/indexer.cu:91 an illegal memory access
// inside that prompt's 12th prefill. Nothing in the engine asserted anything about prefill LENGTH,
// so the whole length axis was untested.
//
// The invariant. Every prefill path here is causal and every prompt in the sweep is shorter than the
// sliding WINDOW, so row i of the output is a function of x[0..i] ONLY. Therefore, for two prefill
// lengths s < S over the SAME x, the first s rows must agree — and BIT-EXACTLY, not approximately:
// every GEMM here is per-output-row (gemm_fp32's M=K variant chunks rows in 8s but each row keeps the
// same accumulation order), rmsnorm/act_quant/rope are per row, the compressor pools per group, and
// sparse_attn is per query. There is no cross-row reduction anywhere in the chain. So any difference
// at all is a defect: a length-dependent buffer overrun, a stale index, or a mis-sized launch.
//
// Coverage: the sliding layer (hc_pre -> rmsnorm -> mla_cache_kv -> mla_forward -> hc_post), the
// ratio-4 indexer layer (compressed_attn_cache_r4 + compressed_attn_forward) and the ratio-128
// strided layer (compressed_attn_cache + compressed_attn_forward). Those are exactly the kernels the
// engine runs between the prefill's last real cudaDeviceSynchronize and indexer.cu:91's sync — the
// window the fault has to live in. It checks the CACHES too, not just the output: win_kv, comp_kv and
// idx_ckv are what a wrong-length prefill leaves behind for the decode loop to read.
//
// Alignment (Finding 41 / standing caveat 2): real weights are 4-byte-aligned pointers into a mapped
// file, cudaMalloc is always 256. Every weight blob here is allocated with 8 spare bytes and handed
// out at base+`off`, and the whole sweep runs at off=0 AND off=4.
//
// Run it under compute-sanitizer too — that is what turns "the outputs agree" into "and nothing read
// or wrote out of bounds at any length".
//   build: see scripts/build_gate.sh (gate_prefill_len)
//   run:   ./build/gate_prefill_len
//          compute-sanitizer --tool memcheck ./build/gate_prefill_len
#include "compressed_attn.h"
#include "compressed_decode.h"
#include "mla_forward.h"
#include "mla_decode.h"
#include "mla_attn.h"      // rmsnorm
#include "hc.h"
#include "deepseek_v4.h"
#include <cuda_runtime.h>
#include <vector>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <cstdlib>
using namespace dsv4;
#define CU(x) do{cudaError_t e=(x); if(e){fprintf(stderr,"cuda %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)

// ---- weight upload, all at base+off so the gate sees the real 4-byte alignment, not cudaMalloc's 256
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
static const float* upNorm(int n){ std::vector<float> v(n); for(auto&e:v)e=0.5f+0.01f*(rand()%100); return upFv(v); }

// Every CUDA runtime call returns "the last error produced by any preceding runtime call in this
// host thread", so an error left behind by stage N is reported by stage N+k's CU() at a line that has
// nothing to do with it. That is the attribution mechanism this whole gate exists to pin down, so the
// gate drains and names the error after EVERY stage rather than once at the end.
static int g_errs = 0;
static void chkpt(const char* what, int s){
    cudaError_t e = cudaGetLastError();
    if(e != cudaSuccess){ printf("  [s=%2d] %-22s left error: %s\n", s, what, cudaGetErrorString(e)); ++g_errs; }
}

// device buffer -> host, for a bitwise compare
static std::vector<float> down(const float* d, size_t n){
    std::vector<float> h(n); CU(cudaMemcpy(h.data(), d, n*4, cudaMemcpyDeviceToHost)); return h;
}
// bitwise, because the invariant is bit-exactness. Reports the first differing element.
static bool same(const std::vector<float>& a, const std::vector<float>& b, size_t n, size_t& where){
    for(size_t i=0;i<n;++i) if(memcmp(&a[i],&b[i],4)!=0){ where=i; return false; }
    return true;
}

struct Run {                       // everything one prefill length produces
    std::vector<float> sl_out, sl_win;                 // sliding layer: block output, window KV
    std::vector<float> r4_out, r4_win, r4_comp, r4_idx;
    std::vector<float> r128_out, r128_win;
    int r4_T = -1, r128_T = -1;
};

int main(int argc, char** argv){
    // Lengths the sweep actually prefills at are PSp = len(prompt)-1: 5, 10, 14, 17. 17 is the one
    // that failed. The neighbours are here so a failure at 17 can be read as "SMAX" or "that length".
    std::vector<int> LENS = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20};
    if(argc>1){                                   // "5,10,14,17,20" — a short list for sanitizer runs
        LENS.clear(); for(char* t=strtok(argv[1],","); t; t=strtok(nullptr,",")) LENS.push_back(atoi(t)); }
    const int SREF = LENS.back();
    const int ratio4 = 4, ratio128 = 128, half = ROPE_DIM/2;
    const int Kd = N_HEADS*HEAD_DIM, GKd = Kd/O_GROUPS, OB = O_GROUPS*O_LORA;
    const int nH = INDEX_N_HEADS, ihd = INDEX_HEAD_DIM, QD = nH*ihd, iod = 2*ihd;
    const int hc = HC_MULT, d = DIM, mix_hc = (2+hc)*hc;
    const int Tref4 = SREF/ratio4;
    int nfail = 0;

    for (int pass = 0; pass < 2; ++pass) {
        g_off = pass ? 4 : 0;
        srand(31);                                    // same weights in both passes
        CompressedAttnWeights w{}; MLAWeights& a = w.attn;
        a.wq_a=upW((size_t)Q_LORA*DIM);   a.wq_a_s=upS((size_t)(Q_LORA/128)*(DIM/128));
        a.wq_b=upW((size_t)Kd*Q_LORA);    a.wq_b_s=upS((size_t)(Kd/128)*(Q_LORA/128));
        a.wkv =upW((size_t)HEAD_DIM*DIM); a.wkv_s =upS((size_t)(HEAD_DIM/128)*(DIM/128));
        a.wo_b=upW((size_t)DIM*OB);       a.wo_b_s=upS((size_t)(DIM/128)*(OB/128));
        a.q_norm=upNorm(Q_LORA); a.kv_norm=upNorm(HEAD_DIM);
        a.wo_a=upR((size_t)O_GROUPS*O_LORA*GKd,0.02f); a.attn_sink=upR(N_HEADS,0.1f);
        // rope tables sized at the LONGEST length and shared by every shorter run — exactly how the
        // engine sizes them (seqmax), so a short run reading past its own s would still be in-bounds
        // here and would show up as a prefix mismatch rather than a segfault.
        std::vector<float> cq((size_t)SREF*half), sq((size_t)SREF*half);
        for(int p=0;p<SREF;++p) for(int j=0;j<half;++j){ float ang=p*0.011f*(j+1); cq[p*half+j]=cosf(ang); sq[p*half+j]=sinf(ang); }
        a.cosT=upFv(cq); a.sinT=upFv(sq);
        w.mc_wkv=upR((size_t)2*HEAD_DIM*DIM,0.02f); w.mc_wgate=upR((size_t)2*HEAD_DIM*DIM,0.02f);
        w.mc_ape=upR((size_t)ratio4*2*HEAD_DIM,0.1f); w.mc_norm=upNorm(HEAD_DIM);
        std::vector<float> cc((size_t)Tref4*half), cs((size_t)Tref4*half);
        for(int t=0;t<Tref4;++t) for(int j=0;j<half;++j){ float ang=t*0.019f*(j+1); cc[t*half+j]=cosf(ang); cs[t*half+j]=sinf(ang); }
        w.cc_cos=upFv(cc); w.cc_sin=upFv(cs);
        w.idx_wq_b=upW((size_t)QD*Q_LORA); w.idx_wq_b_s=upS((size_t)(QD/128)*(Q_LORA/128));
        w.idx_weights_proj=upR((size_t)nH*DIM,0.02f);
        w.idx_c_wkv=upR((size_t)iod*DIM,0.02f); w.idx_c_wgate=upR((size_t)iod*DIM,0.02f);
        w.idx_c_ape=upR((size_t)ratio4*iod,0.1f); w.idx_c_norm=upNorm(ihd);
        w.index_n_heads=nH; w.index_head_dim=ihd; w.index_topk=INDEX_TOPK;

        // ratio-128 twin: non-overlap compressor (od = HEAD_DIM), ape [128, HEAD_DIM]
        CompressedAttnWeights w128 = w;
        w128.mc_wkv=upR((size_t)HEAD_DIM*DIM,0.02f); w128.mc_wgate=upR((size_t)HEAD_DIM*DIM,0.02f);
        w128.mc_ape=upR((size_t)ratio128*HEAD_DIM,0.1f);

        // HC weights (the sliding block's pre/post around attention)
        const float* hc_fn   = upR((size_t)mix_hc*hc*d, 0.01f);
        const float* hc_scl  = upR(3, 1.0f);
        const float* hc_base = upR(mix_hc, 0.5f);

        // the one input every length is a prefix of
        std::vector<float> xh((size_t)SREF*hc*d); for(auto&e:xh) e = 0.05f*((rand()%200)-100)/100.f;
        const float* xstate = upFv(xh);            // HC state [SREF, hc, d]

        std::vector<Run> runs(LENS.size());
        for(size_t li=0; li<LENS.size(); ++li){
            const int s = LENS[li]; Run& R = runs[li];
            const int T4 = s/ratio4, T128 = s/ratio128;
            float *x1,*post,*comb,*sub,*res2;
            CU(cudaMalloc(&x1,(size_t)s*d*4)); CU(cudaMalloc(&post,(size_t)s*hc*4));
            CU(cudaMalloc(&comb,(size_t)s*hc*hc*4)); CU(cudaMalloc(&sub,(size_t)s*d*4));
            CU(cudaMalloc(&res2,(size_t)s*hc*d*4));
            // ---- sliding layer (ratio 0) ----
            cudaGetLastError();                                // start each length from a clean slate
            hc_pre(x1,post,comb,xstate,hc_fn,hc_scl,hc_base,s,hc,d,HC_SINKHORN_ITERS,EPS,0);
            chkpt("hc_pre", s);
            rmsnorm(x1,x1,nullptr,s,d,EPS,false,0);           // stand-in attn_norm (unweighted)
            float* win_kv; CU(cudaMalloc(&win_kv,(size_t)s*HEAD_DIM*4));
            mla_cache_kv(win_kv, x1, a, s, 0);                 chkpt("mla_cache_kv", s);
            mla_forward(sub, x1, a, 1, s, 0);                  chkpt("mla_forward", s);
            hc_post(res2, sub, xstate, post, comb, s, hc, d, 0);
            CU(cudaDeviceSynchronize());                       chkpt("hc_post", s);
            R.sl_out = down(res2,(size_t)s*hc*d); R.sl_win = down(win_kv,(size_t)s*HEAD_DIM);

            // ---- ratio-4 indexer layer ----
            float *w4,*c4,*i4,*o4;
            CU(cudaMalloc(&w4,(size_t)s*HEAD_DIM*4)); CU(cudaMalloc(&c4,(size_t)(T4?T4:1)*HEAD_DIM*4));
            CU(cudaMalloc(&i4,(size_t)(T4?T4:1)*ihd*4)); CU(cudaMalloc(&o4,(size_t)s*DIM*4));
            int T4o=0; compressed_attn_cache_r4(w4,c4,i4,&T4o,x1,w,s,ratio4,EPS,0);
            chkpt("cache_r4", s);
            compressed_attn_forward(o4,x1,w,s,WINDOW,ratio4,EPS,0);
            CU(cudaDeviceSynchronize());                       chkpt("attn_forward r4", s);
            R.r4_T=T4o; R.r4_out=down(o4,(size_t)s*DIM); R.r4_win=down(w4,(size_t)s*HEAD_DIM);
            if(T4){ R.r4_comp=down(c4,(size_t)T4*HEAD_DIM); R.r4_idx=down(i4,(size_t)T4*ihd); }

            // ---- ratio-128 strided layer (T is 0 at every one of these lengths: that is the point) ----
            float *w1x,*c1x,*o1x;
            CU(cudaMalloc(&w1x,(size_t)s*HEAD_DIM*4)); CU(cudaMalloc(&c1x,(size_t)(T128?T128:1)*HEAD_DIM*4));
            CU(cudaMalloc(&o1x,(size_t)s*DIM*4));
            int T128o=0; compressed_attn_cache(w1x,c1x,&T128o,x1,w128,s,ratio128,EPS,0);
            chkpt("cache_r128", s);
            // groups = s/128 == 0 at every prompt this project runs, so every launch the ratio-128
            // compressor issues has gridDim 0. Semantically a no-op (there is no compressed row to
            // emit yet), but it sets the thread's last-error, and the engine's dsync() is a no-op
            // under the decode arena — so the code is carried forward and reported by whatever CU()
            // runs next. Check for it explicitly, here.
            compressed_attn_forward(o1x,x1,w128,s,WINDOW,ratio128,EPS,0);
            CU(cudaDeviceSynchronize());                       chkpt("attn_forward r128", s);
            R.r128_T=T128o; R.r128_out=down(o1x,(size_t)s*DIM); R.r128_win=down(w1x,(size_t)s*HEAD_DIM);

            cudaFree(x1);cudaFree(post);cudaFree(comb);cudaFree(sub);cudaFree(res2);cudaFree(win_kv);
            cudaFree(w4);cudaFree(c4);cudaFree(i4);cudaFree(o4);cudaFree(w1x);cudaFree(c1x);cudaFree(o1x);
        }

        // ---- the invariant: every shorter run is a bitwise prefix of the longest one ----
        const Run& REF = runs.back();
        printf("[prefill_len off=%d] reference s=%d (T4=%d, T128=%d)\n", g_off, SREF, REF.r4_T, REF.r128_T);
        for(size_t li=0; li<LENS.size(); ++li){
            const int s = LENS[li]; const Run& R = runs[li]; const int T4 = s/ratio4;
            size_t at; bool ok = true; char detail[256]; detail[0]=0;
            struct { const char* nm; const std::vector<float>*a; const std::vector<float>*b; size_t n; } chk[] = {
                {"sliding.out",  &R.sl_out,   &REF.sl_out,   (size_t)s*hc*d},
                {"sliding.win",  &R.sl_win,   &REF.sl_win,   (size_t)s*HEAD_DIM},
                {"r4.out",       &R.r4_out,   &REF.r4_out,   (size_t)s*DIM},
                {"r4.win_kv",    &R.r4_win,   &REF.r4_win,   (size_t)s*HEAD_DIM},
                {"r4.comp_kv",   &R.r4_comp,  &REF.r4_comp,  (size_t)T4*HEAD_DIM},
                {"r4.idx_ckv",   &R.r4_idx,   &REF.r4_idx,   (size_t)T4*ihd},
                {"r128.out",     &R.r128_out, &REF.r128_out, (size_t)s*DIM},
                {"r128.win_kv",  &R.r128_win, &REF.r128_win, (size_t)s*HEAD_DIM},
            };
            for(auto& c : chk){
                if(!c.n) continue;
                if(!same(*c.a,*c.b,c.n,at)){ ok=false;
                    snprintf(detail,sizeof detail,"%s[%zu]: %.9g vs %.9g", c.nm, at, (*c.a)[at], (*c.b)[at]);
                    break; }
            }
            if(R.r4_T != T4){ ok=false; snprintf(detail,sizeof detail,"r4 emitted T=%d, expected %d", R.r4_T, T4); }
            if(!ok) ++nfail;
            printf("  s=%2d T4=%d %s%s%s\n", s, R.r4_T, ok?"PASS":"FAIL", detail[0]?"  ":"", detail);
        }
    }
    printf("[prefill_len] %s (%d prefix mismatches, %d stages leaving a CUDA error)\n",
           (nfail||g_errs)?"GATE FAIL":"GATE PASS", nfail, g_errs);
    return (nfail||g_errs) ? 1 : 0;
}
