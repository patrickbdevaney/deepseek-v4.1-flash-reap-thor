// gate_join_defer.cu — DECODE_LADDER 1.11: the deferred ATTN_SPLIT join is BIT-EXACT.
//
// WHY A NEW GATE AND NOT AN EXISTING ONE. Every gate that links these kernels does so WITHOUT
// calling arena_init(), so `g_side` is null, `asplit` is false, and the fork/join path is never
// taken -- gate_kv_pack_e2e drives `compressed_verify_step_indexer` and still runs it single
// stream. That means the ATTN_SPLIT overlap 1.8 measured at 0.81 ms/forward has never had a gate,
// and 1.11 moves the join inside it. A gate that cannot reach the code it is named after is the
// "green audit that proves nothing" failure, so this one calls arena_init() FIRST and REFUSES to
// report PASS if `g_side` came back null.
//
// WHAT IT PROVES. `compressed_verify_step_indexer` forks the two `compressor_emit_group` calls onto
// `g_side`. 1.11 leaves the `cudaEventRecord` where the emits end but moves the
// `cudaStreamWaitEvent` from immediately after `build_qKV` down to `index_score`, the first reader
// of `idx_ckv`. The claim is that nothing in between (`i:qidx`, `i:iw`) reads what the emits write,
// so the outputs must be IDENTICAL BIT FOR BIT -- not close. Both arms run in the same process, on
// the same synthetic weights and the same freshly zeroed caches, and the arm is chosen by
// setenv/unsetenv of NO_JOIN_DEFER, which the kernel reads with getenv per call.
//
// The two arms run (deferred, joined) and then AGAIN as (joined, deferred) under --swap: the
// control for state left behind in a static or in the arena between arms.
//
// --negctl is the control on the GATE ITSELF. It perturbs one input float between the arms; the
// memcmp must then FAIL. A bit-exactness gate that has never been seen to fail is not evidence.
//
//   build: see scripts/build_gate.sh
#include "compressed_attn.h"
#include "compressed_decode.h"
#include "deepseek_v4.h"
#include "dscratch.h"
#include "kv_pack.h"
#include <cuda_runtime.h>
#include <vector>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdlib>
using namespace dsv4;
#define CU(x) do{cudaError_t e=(x); if(e){fprintf(stderr,"cuda %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)
static const float JEPS = 1e-6f;   // EPS is ambiguous here: deepseek_v4.h and dsv4:: both define one
static uint8_t rfp8(){ return (uint8_t)((rand()%0x40) | ((rand()&1)<<7)); }
static const uint8_t* upW(size_t n){ std::vector<uint8_t> h(n); for(auto&v:h)v=rfp8();
    uint8_t* d; CU(cudaMalloc(&d,n)); CU(cudaMemcpy(d,h.data(),n,cudaMemcpyHostToDevice)); return d; }
static const float* upS(size_t n){ std::vector<float> h(n); for(auto&v:h)v=0.3f+0.01f*(rand()%40);
    float* d; CU(cudaMalloc(&d,n*4)); CU(cudaMemcpy(d,h.data(),n*4,cudaMemcpyHostToDevice)); return d; }
static const float* upFv(std::vector<float>& h){ float* d; CU(cudaMalloc(&d,h.size()*4));
    CU(cudaMemcpy(d,h.data(),h.size()*4,cudaMemcpyHostToDevice)); return d; }
static const float* upR(size_t n, float sc){ std::vector<float> h(n); for(auto&v:h)v=sc*((rand()%200)-100)/100.f; return upFv(h); }
static const float* upNorm(int n){ std::vector<float> v(n); for(auto&e:v)e=0.5f+0.01f*(rand()%100); return upFv(v); }

struct Case { int s0, ndec, K; };

// One arm. `jdefer` picks the join position; everything else is identical, including the caches,
// which are reallocated and zeroed so neither arm can inherit the other's state.
static std::vector<float> run_arm(bool jdefer, const CompressedAttnWeights& w, const float* x,
                                  const Case& C, int seqmax, int ratio) {
    if (jdefer) unsetenv("NO_JOIN_DEFER"); else setenv("NO_JOIN_DEFER","1",1);
    const int ihd = w.index_head_dim, Tmax = seqmax/ratio + 2;
    float *win_kv, *comp_kv, *idx_ckv, *out;
    CU(cudaMalloc(&win_kv, kv_rows_bytes(seqmax)));
    CU(cudaMalloc(&comp_kv, kv_rows_bytes(Tmax)));
    CU(cudaMalloc(&idx_ckv, (size_t)Tmax*ihd*4));
    CU(cudaMalloc(&out, (size_t)(C.ndec + C.K)*DIM*4));
    CU(cudaMemset(win_kv, 0, kv_rows_bytes(seqmax)));
    CU(cudaMemset(comp_kv, 0, kv_rows_bytes(Tmax)));
    CU(cudaMemset(idx_ckv, 0, (size_t)Tmax*ihd*4));
    CU(cudaMemset(out, 0, (size_t)(C.ndec + C.K)*DIM*4));
    int Th = 0;
    arena_reset();
    compressed_attn_cache_r4(win_kv, comp_kv, idx_ckv, &Th, x, w, C.s0, ratio, JEPS);
    for (int i = 0; i < C.ndec; ++i) {                      // M=1 steps: unchanged by 1.11, carried for state
        arena_reset();
        compressed_decode_step_indexer(out + (size_t)i*DIM, x + (size_t)(C.s0+i)*DIM, x, C.s0+i, w,
                                       win_kv, comp_kv, idx_ckv, &Th, ratio, JEPS);
    }
    arena_reset();                                          // THE SUBJECT: one M=K verify block
    compressed_verify_step_indexer(out + (size_t)C.ndec*DIM, x + (size_t)(C.s0+C.ndec)*DIM, x,
                                   C.s0+C.ndec, C.K, w, win_kv, comp_kv, idx_ckv, &Th, ratio, JEPS);
    CU(cudaDeviceSynchronize());
    std::vector<float> h((size_t)(C.ndec + C.K)*DIM);
    CU(cudaMemcpy(h.data(), out, h.size()*4, cudaMemcpyDeviceToHost));
    CU(cudaFree(win_kv)); CU(cudaFree(comp_kv)); CU(cudaFree(idx_ckv)); CU(cudaFree(out));
    return h;
}

int main(int argc, char** argv){
    bool swap=false, negctl=false; int same=-1;
    for (int i=1;i<argc;++i){ if(!strcmp(argv[i],"--swap")) swap=true; if(!strcmp(argv[i],"--negctl")) negctl=true;
        // NULL CONTROL. --same-defer / --same-join run BOTH arms at the SAME join position, so any
        // difference they report cannot be 1.11 -- it is state the harness itself leaks between the
        // two arms (a static, the arena, a pinned buffer). Without this control a harness bug and a
        // kernel bug are indistinguishable, and the first version of this gate reported one row
        // differing that turned out to belong here.
        if(!strcmp(argv[i],"--same-defer")) same=1; if(!strcmp(argv[i],"--same-join")) same=0; }
    // The arena MUST come first: it is what creates g_side and the two events, and without it the
    // whole subject of this gate compiles to the single-stream fallback.
    arena_init((size_t)2048<<20);
    kv_pack_init();
    if (!g_side) { printf("[joindefer] g_side is NULL -- arena_init made no side stream, so this gate\n"
                          "            would have run the single-stream path and proved nothing.\n"
                          "\nGATE JOIN_DEFER: FAIL\n"); return 1; }
    if (getenv("NO_ATTN_SPLIT")) { printf("[joindefer] NO_ATTN_SPLIT is set in the environment; the fork is off\n"
                                          "            and there is no join to defer.\n\nGATE JOIN_DEFER: FAIL\n"); return 1; }
    printf("[joindefer] arena on, g_side=%p, ATTN_SPLIT active. KV layout: %s\n",
           (void*)g_side, g_kv_pack ? "packed" : "fp32");

    const int half = ROPE_DIM/2, Kd = N_HEADS*HEAD_DIM, GKd = Kd/O_GROUPS, OB = O_GROUPS*O_LORA;
    const int nH = INDEX_N_HEADS, ihd = INDEX_HEAD_DIM, QD = nH*ihd, iod = 2*ihd;
    const int seqmax = 640, ratio = 4;
    // s0 clears WINDOW so the window is saturated and compressed rows are actually selected.
    // K=2 puts ONE ratio-4 group boundary inside the verify block, K=6 puts TWO, K=1 puts NONE --
    // the last is the control for "no emit, therefore no fork", which must also be identical.
    Case cases[] = { {480, 6, 2}, {480, 6, 6}, {480, 6, 1}, {184, 3, 5} };

    int fails = 0, emitless = 0;
    for (const Case& C : cases) {
        srand(31 + C.K + C.s0);
        const int T = seqmax/ratio + 2;
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
        w.mc_wkv=upR((size_t)2*HEAD_DIM*DIM,0.02f); w.mc_wgate=upR((size_t)2*HEAD_DIM*DIM,0.02f);
        w.mc_ape=upR((size_t)ratio*2*HEAD_DIM,0.1f); w.mc_norm=upNorm(HEAD_DIM);
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

        // How many emits the verify block actually contains, so a case that forks nothing cannot
        // be counted as evidence that the fork is bit-exact.
        int nemit=0; for(int j=C.s0+C.ndec; j<C.s0+C.ndec+C.K; ++j) if((j+1)%ratio==0) ++nemit;
        if(!nemit) ++emitless;

        const bool armA = (same>=0) ? (same==1) : true;      // deferred unless a null control pins both
        const bool armB = (same>=0) ? (same==1) : false;
        // WARM ARM, DISCARDED. The FIRST run_arm() in a process disagrees with every later one, in
        // ONE row -- row 0, the first M=1 decode step -- and it does so with BOTH arms pinned to the
        // SAME join position and with identical weights, i.e. with 1.11 out of the picture entirely.
        // Duplicating a case proved it: instance 1 differs, instance 2 of the same case is clean.
        // The arena slab is cudaMalloc'd and never zeroed, so on the first pass some scratch is read
        // before it is written and sees driver garbage; on every later pass it sees the previous
        // pass's bytes, which repeat. That is a pre-existing uninitialised read in the M=1 step and
        // it is NOT what this gate is for -- but it MUST be burned off here, because a gate that
        // reports a difference it cannot attribute is worse than no gate. Recorded in
        // wiki/measurement-and-traps.md; the M=1 read itself belongs on the ladder, not in 1.11.
        (void)run_arm(armA, w, x, C, seqmax, ratio);
        std::vector<float> A, B;
        if (swap) { B = run_arm(armB, w, x, C, seqmax, ratio); A = run_arm(armA, w, x, C, seqmax, ratio); }
        else      { A = run_arm(armA, w, x, C, seqmax, ratio); B = run_arm(armB, w, x, C, seqmax, ratio); }
        if (negctl) {   // prove the memcmp is live: one perturbed float must be caught
            std::vector<float> xn = xh; xn[(size_t)(C.s0+1)*DIM+7] += 1e-3f;
            const float* x2 = upFv(xn);
            B = run_arm(false, w, x2, C, seqmax, ratio);
        }
        size_t diff = 0, first = (size_t)-1;
        for (size_t i=0;i<A.size();++i) if (memcmp(&A[i],&B[i],4)) { if(first==(size_t)-1) first=i; ++diff; }
        printf("s0=%d decode=%d verify_K=%d emits_in_block=%d  %zu / %zu floats differ%s\n",
               C.s0, C.ndec, C.K, nemit, diff, A.size(), diff? "   <<<< NOT BIT-EXACT":"");
        if (diff) { printf("   first at %zu (row %zu col %zu): deferred %.9g  joined %.9g\n",
                           first, first/DIM, first%DIM, A[first], B[first]); }
        if (negctl) { if(!diff){ printf("   NEGATIVE CONTROL DID NOT FIRE — the memcmp is not live\n"); ++fails; } }
        else if (diff) ++fails;
    }
    unsetenv("NO_JOIN_DEFER");
    if (emitless == (int)(sizeof(cases)/sizeof(cases[0]))) {
        printf("[joindefer] NO case contained an emit — this gate exercised no fork at all.\n"
               "\nGATE JOIN_DEFER: FAIL\n"); return 1; }
    if (negctl) printf("%s  [--negctl: every arm pair MUST differ]\n",
                       fails? "GATE JOIN_DEFER NEGCTL: FAIL":"GATE JOIN_DEFER NEGCTL: PASS");
    else if (same>=0) printf("%s  (arm mismatches %d)  [NULL CONTROL: both arms %s -- any\n"
                             "    difference here is the HARNESS, not 1.11]\n",
                             fails? "GATE JOIN_DEFER NULLCTL: DIFFERS":"GATE JOIN_DEFER NULLCTL: CLEAN",
                             fails, same? "deferred":"joined");
    else        printf("%s  (arm mismatches %d)%s\n", fails? "GATE JOIN_DEFER: FAIL":"GATE JOIN_DEFER: PASS",
                       fails, swap? "  [--swap arm order]":"");
    return fails ? 1 : 0;
}
