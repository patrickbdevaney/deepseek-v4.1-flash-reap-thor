// gate_index_score.cu — DECODE_LADDER 1.5. Two DIFFERENT bit-exactness claims, and the whole point
// of this file is that they are different:
//
//   IXS_TILED  is BIT-IDENTICAL to IXS_WARP   — the shipped kernel. Memory placement only.
//   IXS_GEMM   is BIT-IDENTICAL to IXS_SCALAR — the correctness-first REFERENCE kernel that
//              tests/gate_units.cu checks against ref/goldens/unit_index_score.safetensors.
//
// IXS_GEMM is therefore NOT bit-identical to what ships today, and this gate MEASURES that gap
// rather than asserting it is small: it prints max_abs / max_rel / #differing between GEMM and WARP
// on every shape. The direction matters — LOOP_LOG Finding 68 adopted IXS_WARP as a deviation FROM
// the reference order (serial-in-d became a shuffle tree) behind the LOSSLESS gate; 1.5 hands that
// deviation back, because a register-tiled GEMM accumulates serially in k, which is the reference's
// own order.
//
// THE CLAIM IS VALUE EQUALITY, SO THE INSTRUMENT IS memcmp. 1.5 changes where the operands live
// (q -> shared, kv[t] -> registers, d -> template so the e-loop unrolls) and changes NOTHING about
// the arithmetic: same lane->element mapping (lane, lane+32, ...), same serial `dot +=` order, same
// 5-step __shfl_down_sync tree, same fmaxf, same serial accumulation over h. If any of that were
// reassociated the outputs would differ in the last mantissa bit and a cosine gate would call it
// fine — which is precisely LOOP_LOG Finding 68, where a reduction-order change to THIS kernel
// bought +28 % tok/s by degrading the output past every tolerance gate in the project. The top-k
// downstream is a SELECTION: one last-ulp flip near the k-th boundary changes which rows attention
// sees, and `sparse_attn` then sums them in index order, so the error is not bounded by the ulp.
// tests/gate_units.cu next door is the cosine gate against the golden and would pass all of that.
//
// SWEEP, per LEVERS.md trap 9 (a harness that cannot express the regime just confirms itself):
//   * S  = 1 (the dp decode step, one query), 5 and 6 (the verify widths the engine issues), 2, 17
//     (not a multiple of anything), and 129 (prefill-shaped, forces gridDim.y past a wave)
//   * T  = 1, 7, 31, 32, 33, 63, 129, 384, 3072 (ctx 12,288 at ratio 4), 6001 — including T below
//     the warps-per-block count, where the grid-stride loop's first iteration is already past T
//   * H  = 64 (the model), 1, 3 (H not a multiple of a warp)
//   * d  = 128 (the model) and 64 — both templated; 96 checks that an untemplated d falls back to
//     the warp kernel instead of silently writing nothing
//   * distributions: uniform +/-, all-positive (relu never fires), all-negative (relu always
//     fires, so every score is exactly 0 and a kernel that writes nothing would PASS — which is
//     why the output is poisoned to 0xEE before every launch), a near-tie row, and one with -0.0
//     and denormals, because ord(-0.0) < ord(+0.0) as raw bits downstream (include/topk_radix.h).
//
// TIMING BAND is printed for the shipped shape, min/median/max over repeats, both arms, same
// process, same buffers, interleaved arm order so drift cannot favour one.
//
//   build: see scripts/build_gate.sh      run: ./build/gate_index_score
#include "indexer.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#define CU(x) do{cudaError_t e=(x); if(e){printf("ERR %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));return 1;}}while(0)

static unsigned rngs = 12345u;
static inline unsigned r32(){ rngs = rngs*1664525u + 1013904223u; return rngs; }
static inline float ru(){ return (float)(r32() >> 8) * (1.0f/16777216.0f); }

// dist: 0 uniform+/-, 1 all-positive, 2 all-negative, 3 near-tie, 4 signed zeros + denormals
static void fill(std::vector<float>& v, int dist, float sc){
    for(size_t i=0;i<v.size();++i){
        float x;
        switch(dist){
            case 1: x = ru()*sc + 0.01f; break;
            case 2: x = -(ru()*sc + 0.01f); break;
            case 3: x = ((i%7)==0) ? sc : sc*(1.0f + 1e-7f*(float)(i%3)); break;
            case 4: { unsigned k=r32()%4;
                      x = (k==0)? -0.0f : (k==1)? 0.0f : (k==2)? 1.4e-45f*(float)(1+(r32()%8))
                                                             : (ru()-0.5f)*sc; break; }
            default: x = (ru()-0.5f)*2.0f*sc;
        }
        v[i]=x;
    }
}

struct Shape { int S,T,H,d; };

int main(){
    const int Ss[] = {1,2,5,6,17,129};
    const int Ts[] = {1,7,31,32,33,63,129,384,3072,6001};
    const int Hs[] = {64,1,3};
    const int ds[] = {128,64};

    long checked=0, badshapes=0, badfloats=0; int shapes=0;
    double dev_abs[5]={0,0,0,0,0}, dev_rel[5]={0,0,0,0,0}; long dev_n[5]={0,0,0,0,0}, dev_tot[5]={0,0,0,0,0};
    for(int di=0; di<2; ++di) for(int hi=0; hi<3; ++hi) for(int si=0; si<6; ++si) for(int ti=0; ti<10; ++ti){
        const int S=Ss[si], T=Ts[ti], H=Hs[hi], d=ds[di];
        // Keep the sweep to seconds: only the model head count gets the full T ladder.
        if(H!=64 && (T>129 || S>6)) continue;
        if(d!=128 && (T>3072)) continue;
        for(int dist=0; dist<5; ++dist){
            std::vector<float> q((size_t)S*H*d), kv((size_t)T*d), w((size_t)S*H);
            rngs = 999u + 7*dist + 13*T + 31*S + 101*H + d;
            fill(q,dist,1.0f); fill(kv,dist,1.0f); fill(w,(dist==2?0:dist),0.5f);
            float *dq,*dkv,*dw,*sa,*sb,*sc,*sg;
            CU(cudaMalloc(&dq,q.size()*4)); CU(cudaMalloc(&dkv,kv.size()*4)); CU(cudaMalloc(&dw,w.size()*4));
            CU(cudaMalloc(&sa,(size_t)S*T*4)); CU(cudaMalloc(&sb,(size_t)S*T*4));
            CU(cudaMalloc(&sc,(size_t)S*T*4)); CU(cudaMalloc(&sg,(size_t)S*T*4));
            CU(cudaMemcpy(dq,q.data(),q.size()*4,cudaMemcpyHostToDevice));
            CU(cudaMemcpy(dkv,kv.data(),kv.size()*4,cudaMemcpyHostToDevice));
            CU(cudaMemcpy(dw,w.data(),w.size()*4,cudaMemcpyHostToDevice));
            // POISON both outputs. dist==2 makes every true score exactly 0.f, so an unwritten
            // buffer that happened to be zero would pass; 0xEE is a large negative float.
            CU(cudaMemset(sa,0xEE,(size_t)S*T*4)); CU(cudaMemset(sb,0xEE,(size_t)S*T*4));
            CU(cudaMemset(sc,0xEE,(size_t)S*T*4)); CU(cudaMemset(sg,0xEE,(size_t)S*T*4));
            bool okw = index_score_impl(IXS_WARP  , sa,dq,dkv,dw,S,T,H,d);
            bool okt = index_score_impl(IXS_TILED , sb,dq,dkv,dw,S,T,H,d);
            bool oks = index_score_impl(IXS_SCALAR, sc,dq,dkv,dw,S,T,H,d);
            bool okg = index_score_impl(IXS_GEMM  , sg,dq,dkv,dw,S,T,H,d);
            CU(cudaDeviceSynchronize());
            CU(cudaGetLastError());
            const bool gemm_expected = (d % 32)==0 && (H % 8)==0;
            if(!okw || !okt || !oks || okg != gemm_expected){
                printf("  S=%d T=%d H=%d d=%d dist=%d: warp=%d tiled=%d scalar=%d gemm=%d (gemm expected %d) LAUNCH\n",
                       S,T,H,d,dist,(int)okw,(int)okt,(int)oks,(int)okg,(int)gemm_expected); ++badshapes; }
            if(okw && okt && oks){
                const size_t N=(size_t)S*T;
                std::vector<float> a(N), b(N), r(N), g(N);
                CU(cudaMemcpy(a.data(),sa,N*4,cudaMemcpyDeviceToHost));
                CU(cudaMemcpy(b.data(),sb,N*4,cudaMemcpyDeviceToHost));
                CU(cudaMemcpy(r.data(),sc,N*4,cudaMemcpyDeviceToHost));
                if(okg) CU(cudaMemcpy(g.data(),sg,N*4,cudaMemcpyDeviceToHost));
                unsigned poison; { float f; memset(&f,0xEE,4); memcpy(&poison,&f,4); }
                long np=0;
                for(size_t i=0;i<N;++i){ unsigned u;
                    memcpy(&u,&a[i],4); if(u==poison) ++np;
                    memcpy(&u,&b[i],4); if(u==poison) ++np;
                    memcpy(&u,&r[i],4); if(u==poison) ++np;
                    if(okg){ memcpy(&u,&g[i],4); if(u==poison) ++np; } }
                if(np){ printf("  S=%d T=%d H=%d d=%d dist=%d: %ld UNWRITTEN slots\n",S,T,H,d,dist,np); ++badshapes; }
                long nbt=0, nbg=0; size_t ft=0, fg=0;
                for(size_t i=0;i<N;++i){ unsigned ua,ub; memcpy(&ua,&a[i],4); memcpy(&ub,&b[i],4);
                    if(ua!=ub){ if(!nbt) ft=i; ++nbt; } }
                if(okg) for(size_t i=0;i<N;++i){ unsigned ur,ug; memcpy(&ur,&r[i],4); memcpy(&ug,&g[i],4);
                    if(ur!=ug){ if(!nbg) fg=i; ++nbg; } }
                checked += (long)N*2; badfloats += nbt + nbg;
                if(nbt){ ++badshapes;
                    printf("  TILED!=WARP  S=%d T=%d H=%d d=%d dist=%d: %ld/%zu differ, first %zu (%.9g vs %.9g)\n",
                           S,T,H,d,dist,nbt,N,ft,a[ft],b[ft]); }
                if(nbg){ ++badshapes;
                    printf("  GEMM!=SCALAR S=%d T=%d H=%d d=%d dist=%d: %ld/%zu differ, first %zu (%.9g vs %.9g)\n",
                           S,T,H,d,dist,nbg,N,fg,r[fg],g[fg]); }
                // The MEASURED deviation of the adopted kernel from the one that ships today.
                if(okg) for(size_t i=0;i<N;++i){
                    float x=a[i], y=g[i];
                    if(x!=y){ ++dev_n[dist]; double ab=fabs((double)x-(double)y);
                              double rel = ab / (fabs((double)x)>1e-30 ? fabs((double)x) : 1e-30);
                              if(ab>dev_abs[dist]) dev_abs[dist]=ab;
                              if(rel>dev_rel[dist]) dev_rel[dist]=rel; }
                    ++dev_tot[dist]; }
            }
            cudaFree(dq);cudaFree(dkv);cudaFree(dw);cudaFree(sa);cudaFree(sb);cudaFree(sc);cudaFree(sg);
            ++shapes;
        }
    }

    // THE DEVIATION OF THE ADOPTED KERNEL FROM THE ONE THAT SHIPS TODAY, per distribution, because
    // one aggregated max_rel is unreadable: dist 3 (near-ties) and dist 4 (signed zeros/denormals)
    // are CONSTRUCTED so the sum cancels to near zero, and a relative error against a near-zero
    // denominator is arbitrarily large by construction and says nothing about the model's inputs.
    // dist 0 (uniform +/-) is the one that resembles a real score row.
    {
        static const char* dn[5] = {"uniform+/-","all-positive","all-negative","near-tie","+/-0,denorm"};
        printf("\n[gemm vs shipped warp] the deviation this change SPENDS (bit-exact vs SCALAR, not vs WARP)\n");
        long tn=0, tt2=0;
        for(int k=0;k<5;++k){
            printf("   %-12s %8ld/%-8ld differ   max_abs=%.3g  max_rel=%.3g\n",
                   dn[k], dev_n[k], dev_tot[k], dev_abs[k], dev_rel[k]);
            tn+=dev_n[k]; tt2+=dev_tot[k];
        }
        printf("   %-12s %8ld/%-8ld\n","ALL",tn,tt2);
    }

    // --- fallback controls: an impl that cannot serve a shape must REFUSE, not silently do
    // something else. d=96 is untemplated for TILED (it templates d) but perfectly legal for GEMM
    // (96 % IXG_KC == 0), and H=12 is legal for TILED but not for GEMM (12 % IXG_TH != 0). Getting
    // this expectation wrong is how a gate reports a pass for a shape nobody ran: the FIRST version
    // of this control asserted GEMM would refuse d=96 and "failed" on a kernel that was correct.
    {
        const int S=2,T=64,H=8,d=96;
        std::vector<float> q((size_t)S*H*d,0.5f), kv((size_t)T*d,0.25f), w((size_t)S*H,1.f);
        float *dq,*dkv,*dw,*sa; CU(cudaMalloc(&dq,q.size()*4)); CU(cudaMalloc(&dkv,kv.size()*4));
        CU(cudaMalloc(&dw,w.size()*4)); CU(cudaMalloc(&sa,(size_t)S*T*4));
        CU(cudaMemcpy(dq,q.data(),q.size()*4,cudaMemcpyHostToDevice));
        CU(cudaMemcpy(dkv,kv.data(),kv.size()*4,cudaMemcpyHostToDevice));
        CU(cudaMemcpy(dw,w.data(),w.size()*4,cudaMemcpyHostToDevice));
        bool okt = index_score_impl(IXS_TILED, sa,dq,dkv,dw,S,T,H,d);
        bool okg = index_score_impl(IXS_GEMM , sa,dq,dkv,dw,S,T,H,d);
        CU(cudaDeviceSynchronize());
        printf("[control d=96 H=8 ] tiled refused = %s (want yes), gemm refused = %s (want no)\n",
               okt?"NO":"yes", okg?"NO":"yes");
        if(okt || !okg) ++badshapes;
        cudaFree(dq);cudaFree(dkv);cudaFree(dw);cudaFree(sa);
    }
    {
        const int S=2,T=64,H=12,d=128;
        std::vector<float> q((size_t)S*H*d,0.5f), kv((size_t)T*d,0.25f), w((size_t)S*H,1.f);
        float *dq,*dkv,*dw,*sa; CU(cudaMalloc(&dq,q.size()*4)); CU(cudaMalloc(&dkv,kv.size()*4));
        CU(cudaMalloc(&dw,w.size()*4)); CU(cudaMalloc(&sa,(size_t)S*T*4));
        CU(cudaMemcpy(dq,q.data(),q.size()*4,cudaMemcpyHostToDevice));
        CU(cudaMemcpy(dkv,kv.data(),kv.size()*4,cudaMemcpyHostToDevice));
        CU(cudaMemcpy(dw,w.data(),w.size()*4,cudaMemcpyHostToDevice));
        bool okt = index_score_impl(IXS_TILED, sa,dq,dkv,dw,S,T,H,d);
        bool okg = index_score_impl(IXS_GEMM , sa,dq,dkv,dw,S,T,H,d);
        CU(cudaDeviceSynchronize());
        printf("[control d=128 H=12] tiled refused = %s (want no), gemm refused = %s (want yes)\n",
               okt?"NO":"yes", okg?"NO":"yes");
        if(!okt || okg) ++badshapes;
        cudaFree(dq);cudaFree(dkv);cudaFree(dw);cudaFree(sa);
    }

    printf("[gate_index_score] %d shapes, %ld floats compared, %ld differing, %ld bad shapes -> %s\n",
           shapes, checked, badfloats, badshapes, (badshapes==0 && badfloats==0)?"PASS":"FAIL");

    // ================= TIMING BAND, shipped shape ===============================================
    // S=6 (the K=6 verify) and S=1 (the dp decode step) at the compressed lengths ctx 3k..24k maps
    // to at ratio 4. Arms interleaved per repeat so thermal drift cannot favour either.
    {
        const int H=64,d=128;
        struct P { int S,T; } pts[] = { {6,768},{6,1536},{6,3072},{6,6144},{1,3072},{1,6144} };
        const int REP = 30;
        printf("\n[band] H=%d d=%d, %d repeats, arms interleaved   (us: min/median/max)\n",H,d,REP);
        printf("   S     T        warp                  tiled                 gemm          tiled/gemm vs warp\n");
        for(P p : pts){
            std::vector<float> q((size_t)p.S*H*d), kv((size_t)p.T*d), w((size_t)p.S*H);
            rngs=4242u; fill(q,0,1.f); fill(kv,0,1.f); fill(w,1,0.5f);
            float *dq,*dkv,*dw,*so; CU(cudaMalloc(&dq,q.size()*4)); CU(cudaMalloc(&dkv,kv.size()*4));
            CU(cudaMalloc(&dw,w.size()*4)); CU(cudaMalloc(&so,(size_t)p.S*p.T*4));
            CU(cudaMemcpy(dq,q.data(),q.size()*4,cudaMemcpyHostToDevice));
            CU(cudaMemcpy(dkv,kv.data(),kv.size()*4,cudaMemcpyHostToDevice));
            CU(cudaMemcpy(dw,w.data(),w.size()*4,cudaMemcpyHostToDevice));
            cudaEvent_t e0,e1; CU(cudaEventCreate(&e0)); CU(cudaEventCreate(&e1));
            std::vector<float> tw, tt, tg;
            for(int r=0;r<REP+3;++r){
                for(int arm=0; arm<3; ++arm){
                    CU(cudaDeviceSynchronize()); CU(cudaEventRecord(e0));
                    index_score_impl(arm==0?IXS_WARP:arm==1?IXS_TILED:IXS_GEMM, so,dq,dkv,dw,p.S,p.T,H,d);
                    CU(cudaEventRecord(e1)); CU(cudaEventSynchronize(e1));
                    float ms; CU(cudaEventElapsedTime(&ms,e0,e1));
                    if(r>=3){ (arm==0?tw:arm==1?tt:tg).push_back(ms*1000.f); }   // us
                }
            }
            std::sort(tw.begin(),tw.end()); std::sort(tt.begin(),tt.end()); std::sort(tg.begin(),tg.end());
            float wm=tw[tw.size()/2], tm=tt[tt.size()/2], gm=tg[tg.size()/2];
            printf("  %2d %5d  %7.1f/%7.1f/%7.1f  %6.1f/%6.1f/%6.1f  %6.1f/%6.1f/%6.1f  %6.2fx %6.2fx\n",
                   p.S,p.T, tw.front(),wm,tw.back(), tt.front(),tm,tt.back(),
                   tg.front(),gm,tg.back(), wm/tm, wm/gm);
            cudaEventDestroy(e0);cudaEventDestroy(e1);
            cudaFree(dq);cudaFree(dkv);cudaFree(dw);cudaFree(so);
        }
    }
    return (badshapes==0 && badfloats==0) ? 0 : 1;
}
