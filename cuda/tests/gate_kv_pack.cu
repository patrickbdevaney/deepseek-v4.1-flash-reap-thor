// gate_kv_pack.cu — DECODE_LADDER 1b.2. THREE claims about the packed KV cache, in one binary, in
// seconds, with no checkpoint:
//
//   1. STORAGE ROUND TRIP. `unpack(k_kv_pack(x))` must be BIT-IDENTICAL to what `act_quant_fp8sim`
//      writes into an FP32 cache for the same x, on all 512 dims -- the 448 quantised ones AND the
//      64 RoPE ones that are only copied. memcmp, not a tolerance: this is the whole basis for
//      1b.2 being a storage refactor rather than a precision change, and a tolerance gate would
//      pass a dropped sign bit at |delta| = 0. That is not hypothetical -- it is exactly the bug
//      `gate_idx_pack` caught in 1b.1's first implementation (16,011 mismatches, worst |delta| 0).
//
//   2. READER EQUIVALENCE. `sparse_attn` reading the packed cache must produce the BYTE-IDENTICAL
//      output buffer to `sparse_attn` reading the FP32 cache holding the same values, at every
//      (hpb, smem) launch and at the shapes the engine issues. The online softmax is not
//      associative and the gathered rows are summed IN ORDER, so this is a value-equality claim and
//      cosine would pass the reordering it exists to catch (gate_topk_radix, gate_tc_fp8_kc,
//      gate_og_ws1, gate_sparse_hpb -- same reasoning, four times over).
//
//   3. IDEMPOTENCE. Packing a row that is ALREADY on the E4M3 grid must not move it. The
//      compressed-KV emit path quantises before it stores, so the pack kernel sees pre-quantised
//      input there, and re-quantisation picks a SMALLER-or-equal scale: the codes change, the
//      values must not.
//
// NEGATIVE CONTROLS ARE RUN, not asserted (`--control`): one bumped code byte and one bumped scale
// byte must each make the memcmp FAIL. A gate that has never failed has not been shown able to.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include "mla_attn.h"
#include "kv_pack.h"
#include "deepseek_v4.h"
using namespace dsv4;
using namespace dsv4kv;

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ printf("CUDA %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e)); exit(2);} }while(0)

static uint32_t rng_s = 0x9e3779b9u;
static float frand(){ rng_s ^= rng_s<<13; rng_s ^= rng_s>>17; rng_s ^= rng_s<<5;
                      return ((float)(rng_s>>8) / 8388608.0f) - 1.0f; }

struct Dist { const char* name; float (*gen)(int j); };
static float d_uniform (int)  { return frand(); }
static float d_tiny    (int)  { return frand() * 1e-6f; }
static float d_huge    (int)  { return frand() * 1e5f;  }
// Zero-heavy AND sign-mixed: the case where a NEGATIVE value rounds to zero, which is where 1b.1's
// first implementation lost the sign bit at |delta| = 0.
static float d_zeroish (int j) { float v = frand(); return ((j % 3) == 0) ? 0.f : v * 1e-7f * ((j&1)?-1.f:1.f); }
// One huge outlier per 64-wide group forces the group scale to the top of the E4M3 range, so every
// other element in that group lands near the bottom of it -- including the subnormal codes.
static float d_outlier (int j) { return ((j % 64) == 17) ? 4.4e4f : frand() * 1e-2f; }

int main(int argc, char** argv){
    bool control = false, bench = false;
    for(int i=1;i<argc;++i){ if(!strcmp(argv[i],"--control")) control = true;
                             if(!strcmp(argv[i],"--bench"))   bench   = true; }
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,0));
    printf("device %s  SMs %d\n", p.name, p.multiProcessorCount);
    printf("layout: %d B payload in a %d B stride (%d floats/row); FP32 row is %d B -> %.3fx\n\n",
           KVP_PAYLOAD, KVP_ROWB, KVP_ROWF, HEAD_DIM*4, (double)(HEAD_DIM*4)/KVP_ROWB);

    int fails = 0, ctlfails = 0;

    // ---------------- 1 + 3: storage round trip, per distribution ----------------
    {
        const int rows = 1024;
        const size_t fn = (size_t)rows * HEAD_DIM;
        std::vector<float> hx(fn), href(fn);
        float *dref, *dstage; uint8_t* dpk;
        CK(cudaMalloc(&dref, fn*4)); CK(cudaMalloc(&dstage, fn*4));
        CK(cudaMalloc(&dpk, (size_t)rows*KVP_ROWB));
        std::vector<uint8_t> hpk((size_t)rows*KVP_ROWB);
        Dist dists[] = {{"uniform",d_uniform},{"tiny 1e-6",d_tiny},{"large 1e5",d_huge},
                        {"zero-heavy signed",d_zeroish},{"one outlier per group",d_outlier}};
        for (Dist D : dists) {
            for (size_t i=0;i<fn;++i) hx[i] = D.gen((int)(i % HEAD_DIM));
            // reference: the FP32 cache exactly as the engine writes it today
            CK(cudaMemcpy(dref, hx.data(), fn*4, cudaMemcpyHostToDevice));
            act_quant_fp8sim(dref, rows, NOPE_DIM, 64, HEAD_DIM);
            CK(cudaDeviceSynchronize());
            CK(cudaMemcpy(href.data(), dref, fn*4, cudaMemcpyDeviceToHost));
            for (int pass = 0; pass < 2; ++pass) {
                // pass 0: pack the RAW row.  pass 1: pack the ALREADY-QUANTISED row (claim 3).
                CK(cudaMemcpy(dstage, pass ? href.data() : hx.data(), fn*4, cudaMemcpyHostToDevice));
                CK(cudaMemset(dpk, 0xCD, (size_t)rows*KVP_ROWB));
                g_kv_pack = 1; g_kv_rowf = KVP_ROWF;
                kv_commit((float*)dpk, dstage, rows, 0);
                CK(cudaDeviceSynchronize());
                CK(cudaMemcpy(hpk.data(), dpk, (size_t)rows*KVP_ROWB, cudaMemcpyDeviceToHost));
                size_t diff = 0; int first = -1;
                for (int r=0;r<rows;++r) for (int j=0;j<HEAD_DIM;++j) {
                    float got = kv_unpack(hpk.data() + (size_t)r*KVP_ROWB, j);
                    float want = href[(size_t)r*HEAD_DIM + j];
                    if (memcmp(&got,&want,4)) { if(first<0) first = r*HEAD_DIM+j; ++diff; }
                }
                printf("  %-24s %-14s  %zu / %zu floats differ%s%s\n", D.name,
                       pass ? "[requantised]" : "[raw]", diff, fn,
                       first>=0 ? "   first at " : "", "");
                if (first >= 0) printf("      first mismatch row %d col %d: want %.9g got %.9g\n",
                                       first/HEAD_DIM, first%HEAD_DIM,
                                       href[first], kv_unpack(hpk.data()+(size_t)(first/HEAD_DIM)*KVP_ROWB, first%HEAD_DIM));
                if (diff) ++fails;
            }
        }
        if (control) {
            // one code byte, and one scale byte. Both must be visible to the memcmp above.
            for (int which=0; which<2; ++which) {
                CK(cudaMemcpy(hpk.data(), dpk, (size_t)rows*KVP_ROWB, cudaMemcpyDeviceToHost));
                const int r = 7; const int off = which ? (KVP_OFF_SCALE+2) : 33;
                uint8_t save = hpk[(size_t)r*KVP_ROWB + off];
                hpk[(size_t)r*KVP_ROWB + off] = (uint8_t)(save ^ 1);
                size_t diff = 0;
                for (int j=0;j<HEAD_DIM;++j) {
                    float got = kv_unpack(hpk.data() + (size_t)r*KVP_ROWB, j);
                    float want = href[(size_t)r*HEAD_DIM + j];
                    if (memcmp(&got,&want,4)) ++diff;
                }
                printf("  [control] flip 1 bit of the %s byte -> %zu of %d floats differ %s\n",
                       which?"SCALE":"CODE", diff, HEAD_DIM,
                       diff ? "(gate can fail: good)" : "  <<<< GATE IS BLIND");
                if (!diff) ++ctlfails;
                hpk[(size_t)r*KVP_ROWB + off] = save;
            }
        }
        CK(cudaFree(dref)); CK(cudaFree(dstage)); CK(cudaFree(dpk));
        printf("\n");
    }

    // ---------------- 2: reader equivalence, at the engine's shapes ----------------
    struct Shape { const char* name; int m, topk, n; };
    Shape shapes[] = {
        {"m=1  topk=640 (base AR, ctx>=2048)", 1, 640, 20480},
        {"m=2  topk=640 (mean verify width)",  2, 640, 20480},
        {"m=6  topk=640 (max verify width)",   6, 640, 20480},
        {"m=2  topk=320 (ctx 768, pre-knee)",  2, 320,  5120},
        {"m=256 topk=320 (short prefill)",   256,  320,   320},
    };
    struct Cfg { int hpb, smem; };
    Cfg cfgs[] = {{1,0},{2,0},{4,0},{8,0},{1,1},{2,1},{4,1},{8,1},{2,2},{4,2},{8,2}};
    const int NC = (int)(sizeof(cfgs)/sizeof(cfgs[0]));
    const int b=1, h=N_HEADS, d=HEAD_DIM;

    for (Shape S : shapes) {
        const size_t qn=(size_t)b*S.m*h*d, kvn=(size_t)b*S.n*d, on=qn, in_=(size_t)b*S.m*S.topk;
        std::vector<float> hq(qn), hkv(kvn), hs(h);
        std::vector<int> hidx(in_);
        for (auto& v : hq)  v = frand()*0.5f;
        for (auto& v : hkv) v = frand();
        for (auto& v : hs)  v = frand();
        for (int mi=0; mi<S.m; ++mi) for (int k=0;k<S.topk;++k) {
            int v = (k<128) ? k : (128 + ((k*37 + mi*11) % (S.n-128)));
            if ((k % 97) == 96) v = -1;                       // masked slots: the uniform branch
            hidx[(size_t)mi*S.topk + k] = v;
        }
        float *dq,*dkv,*dsink,*dout,*dref,*dstage; int* didx; uint8_t* dpk;
        CK(cudaMalloc(&dq,qn*4)); CK(cudaMalloc(&dkv,kvn*4)); CK(cudaMalloc(&dsink,h*4));
        CK(cudaMalloc(&dout,on*4)); CK(cudaMalloc(&dref,on*4)); CK(cudaMalloc(&didx,in_*4));
        CK(cudaMalloc(&dstage,kvn*4)); CK(cudaMalloc(&dpk,(size_t)S.n*KVP_ROWB));
        CK(cudaMemcpy(dq,hq.data(),qn*4,cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dsink,hs.data(),h*4,cudaMemcpyHostToDevice));
        CK(cudaMemcpy(didx,hidx.data(),in_*4,cudaMemcpyHostToDevice));
        // FP32 cache = exactly what the engine stores today; packed cache = the same raw rows.
        CK(cudaMemcpy(dkv,hkv.data(),kvn*4,cudaMemcpyHostToDevice));
        act_quant_fp8sim(dkv, S.n, NOPE_DIM, 64, HEAD_DIM);
        CK(cudaMemcpy(dstage,hkv.data(),kvn*4,cudaMemcpyHostToDevice));
        g_kv_pack = 1; g_kv_rowf = KVP_ROWF;
        kv_commit((float*)dpk, dstage, S.n, 0);
        CK(cudaDeviceSynchronize());
        const float scale = 1.f/sqrtf((float)d);

        sparse_attn_launch(dref,dq,dkv,dsink,didx,b,S.m,h,d,S.n,S.topk,scale,0,1,0,false);
        CK(cudaDeviceSynchronize());
        std::vector<float> href(on); CK(cudaMemcpy(href.data(),dref,on*4,cudaMemcpyDeviceToHost));
        std::vector<float> hout(on);
        printf("=== %s ===\n", S.name);
        for (int c=0;c<NC;++c) {
            CK(cudaMemset(dout,0xCD,on*4));
            sparse_attn_launch(dout,dq,(const float*)dpk,dsink,didx,b,S.m,h,d,S.n,S.topk,scale,0,
                               cfgs[c].hpb,cfgs[c].smem,true);
            CK(cudaDeviceSynchronize());
            CK(cudaMemcpy(hout.data(),dout,on*4,cudaMemcpyDeviceToHost));
            size_t diff=0; for(size_t z=0;z<on;++z) if(memcmp(&hout[z],&href[z],4)) ++diff;
            printf("  packed hpb=%d smem=%d  bytediff=%zu / %zu%s\n",
                   cfgs[c].hpb, cfgs[c].smem, diff, on, diff? "   <<<< NOT BIT-EXACT":"");
            if (diff) ++fails;
        }
        if (control) {
            uint8_t save, bump;
            CK(cudaMemcpy(&save, dpk + (size_t)5*KVP_ROWB + 7, 1, cudaMemcpyDeviceToHost));
            bump = (uint8_t)(save ^ 1);
            CK(cudaMemcpy(dpk + (size_t)5*KVP_ROWB + 7, &bump, 1, cudaMemcpyHostToDevice));
            sparse_attn_launch(dout,dq,(const float*)dpk,dsink,didx,b,S.m,h,d,S.n,S.topk,scale,0,4,2,true);
            CK(cudaDeviceSynchronize());
            CK(cudaMemcpy(hout.data(),dout,on*4,cudaMemcpyDeviceToHost));
            size_t diff=0; for(size_t z=0;z<on;++z) if(memcmp(&hout[z],&href[z],4)) ++diff;
            printf("  [control] 1 bit on packed row 5 -> %zu of %zu floats differ %s\n", diff, on,
                   diff? "(gate can fail: good)":"  <<<< GATE IS BLIND");
            if (!diff) ++ctlfails;
            CK(cudaMemcpy(dpk + (size_t)5*KVP_ROWB + 7, &save, 1, cudaMemcpyHostToDevice));
        }
        printf("\n");
        CK(cudaFree(dq));CK(cudaFree(dkv));CK(cudaFree(dsink));CK(cudaFree(dout));
        CK(cudaFree(dref));CK(cudaFree(didx));CK(cudaFree(dstage));CK(cudaFree(dpk));
    }

    // ---------------- 4: THE BAND, at the shapes the engine issues ----------------
    // Added after 1b.2's first paired sweep measured +5.00 +/- 0.37 ms/forward of TERM A against
    // -0.465 +/- 0.053 per 1000 context. That is a fixed cost, so it is a per-call cost, so it is
    // visible here in seconds instead of in two checkpoint loads. Two candidates and this
    // separates them: the READER (`sparse_attn` staging a 720 B row instead of a 2048 B one) and
    // the WRITER (`k_kv_pack` replacing `act_quant_fp8sim`, one block per row running 7 sequential
    // 64-wide reductions instead of 7 blocks running one each).
    if (bench) {
        printf("=== BAND: reader, default launch (hpb=4, smem = total<=1024 ? 2 : 1) ===\n");
        for (Shape S : shapes) {
            const size_t qn=(size_t)b*S.m*h*d, kvn=(size_t)b*S.n*d, on=qn, in_=(size_t)b*S.m*S.topk;
            std::vector<float> hq(qn), hkv(kvn), hs(h); std::vector<int> hidx(in_);
            for (auto& v : hq) v = frand()*0.5f;
            for (auto& v : hkv) v = frand();
            for (auto& v : hs) v = frand();
            for (int mi=0; mi<S.m; ++mi) for (int k=0;k<S.topk;++k) {
                int v = (k<128) ? k : (128 + ((k*37 + mi*11) % (S.n-128)));
                hidx[(size_t)mi*S.topk + k] = v; }
            float *dq,*dkv,*dsink,*dout,*dstage; int* didx; uint8_t* dpk;
            CK(cudaMalloc(&dq,qn*4)); CK(cudaMalloc(&dkv,kvn*4)); CK(cudaMalloc(&dsink,h*4));
            CK(cudaMalloc(&dout,on*4)); CK(cudaMalloc(&didx,in_*4));
            CK(cudaMalloc(&dstage,kvn*4)); CK(cudaMalloc(&dpk,(size_t)S.n*KVP_ROWB));
            CK(cudaMemcpy(dq,hq.data(),qn*4,cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dsink,hs.data(),h*4,cudaMemcpyHostToDevice));
            CK(cudaMemcpy(didx,hidx.data(),in_*4,cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dkv,hkv.data(),kvn*4,cudaMemcpyHostToDevice));
            act_quant_fp8sim(dkv, S.n, NOPE_DIM, 64, HEAD_DIM);
            CK(cudaMemcpy(dstage,hkv.data(),kvn*4,cudaMemcpyHostToDevice));
            g_kv_pack = 1; g_kv_rowf = KVP_ROWF; kv_commit((float*)dpk, dstage, S.n, 0);
            CK(cudaDeviceSynchronize());
            const float scale = 1.f/sqrtf((float)d);
            const long total = (long)b*S.m*h;
            const int hpb0 = 4, smem0 = (total <= 1024L) ? 2 : 1;
            cudaEvent_t e0,e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
            const int ITER = (S.m >= 256) ? 10 : 200, WARM = (S.m >= 256) ? 3 : 20;
            auto time_one = [&](bool pk, int hpb, int smem)->float {
                const float* src = pk ? (const float*)dpk : (const float*)dkv;
                for(int w=0;w<WARM;++w) sparse_attn_launch(dout,dq,src,dsink,didx,b,S.m,h,d,S.n,S.topk,scale,0,hpb,smem,pk);
                CK(cudaDeviceSynchronize()); CK(cudaEventRecord(e0));
                for(int w=0;w<ITER;++w) sparse_attn_launch(dout,dq,src,dsink,didx,b,S.m,h,d,S.n,S.topk,scale,0,hpb,smem,pk);
                CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
                float t=0; CK(cudaEventElapsedTime(&t,e0,e1)); return t/(float)ITER; };
            const float f32 = time_one(false, hpb0, smem0);
            const float pkd = time_one(true,  hpb0, smem0);
            printf("  %-38s fp32 %8.4f ms   packed %8.4f ms   %5.3fx   delta %+7.4f ms/call\n",
                   S.name, f32, pkd, f32/pkd, pkd-f32);
            // A PACKED ROW HAS A DIFFERENT COST STRUCTURE FROM AN FP32 ONE, so the (hpb, smem)
            // default 1.7 tuned on the FP32 layout is not automatically right for it. The unpack is
            // paid ONCE PER BLOCK PER GATHERED ROW, and hpb is exactly how many heads share a
            // block, so raising hpb halves the unpack work per query while costing occupancy.
            printf("      packed sweep:");
            for (int hpb : {2,4,8}) for (int sm : {1,2})
                printf("  h%d/s%d %6.4f", hpb, sm, time_one(true, hpb, sm));
            printf("\n      fp32   sweep:");
            for (int hpb : {2,4,8}) for (int sm : {1,2})
                printf("  h%d/s%d %6.4f", hpb, sm, time_one(false, hpb, sm));
            printf("\n");
            CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
            CK(cudaFree(dq));CK(cudaFree(dkv));CK(cudaFree(dsink));CK(cudaFree(dout));
            CK(cudaFree(didx));CK(cudaFree(dstage));CK(cudaFree(dpk));
        }
        printf("\n=== BAND: writer, act_quant_fp8sim vs k_kv_pack ===\n");
        for (int rows : {1, 2, 6, 64}) {
            const size_t fn = (size_t)rows*HEAD_DIM;
            float* dsrc; uint8_t* dpk;
            CK(cudaMalloc(&dsrc,fn*4)); CK(cudaMalloc(&dpk,(size_t)rows*KVP_ROWB));
            CK(cudaMemset(dsrc,0x3C,fn*4));
            cudaEvent_t e0,e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
            const int ITER=2000, WARM=200; float ms[2];
            for (int pk=0; pk<2; ++pk) {
                g_kv_pack = pk; g_kv_rowf = pk ? KVP_ROWF : HEAD_DIM;
                for(int w=0;w<WARM;++w) kv_commit(pk?(float*)dpk:dsrc, dsrc, rows, 0);
                CK(cudaDeviceSynchronize()); CK(cudaEventRecord(e0));
                for(int w=0;w<ITER;++w) kv_commit(pk?(float*)dpk:dsrc, dsrc, rows, 0);
                CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
                CK(cudaEventElapsedTime(&ms[pk],e0,e1)); ms[pk] /= (float)ITER;
            }
            printf("  rows=%-4d  act_quant_fp8sim %8.5f ms   k_kv_pack %8.5f ms   delta %+8.5f ms/call\n",
                   rows, ms[0], ms[1], ms[1]-ms[0]);
            CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
            CK(cudaFree(dsrc)); CK(cudaFree(dpk));
        }
        printf("\n");
    }

    printf("%s  (bit-exactness failures %d, blind-control failures %d)\n",
           (fails||ctlfails)?"GATE KV_PACK: FAIL":"GATE KV_PACK: PASS", fails, ctlfails);
    return (fails||ctlfails) ? 1 : 0;
}
