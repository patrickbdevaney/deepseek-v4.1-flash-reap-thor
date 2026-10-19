// decode.cu — full 43-layer M=1 KV-cache DECODE driver for DeepSeek-V4-Flash-0731-REAP.
// Prefill-populates per-layer KV caches over [id0..id_{PS-1}], then autoregressively decodes M=1
// tokens and measures decode tok/s.
//
// GATE: the first decoded token must argmax == the expected id, supplied via the DSV4_EXPECT env
// var (default 11111 = " Paris" for the canonical probe "The capital of France is" =
// ids 0,671,6102,294,8760,344). The old hardcoded 270 belonged to the 180B project's DIFFERENT
// probe prompt ("...is the powerhouse of" -> " the") and reported a spurious GATE FAIL on a
// numerically correct 0731 run — a prompt-specific answer must not live in a global constant.
//
// Memory-safe: weights load native (WeightStore, zero-copy on Thor's unified memory);
// scales/norms/wo_a re-dequant PER LAYER with release().
//
//   build: bash scripts/build_decode.sh -> build/decode
//   run:   ./build/decode <ckpt_dir> "0,671,6102,294,8760,344" 8        (base AR)
//          DSV4_KSWEEP=1 ./build/decode ... 8                           (+ verify K-sweep)
//   args:  <dir> <ids> <NDEC> [headdir] [NGEN0]
//          headdir defaults to <dir> when the checkpoint has embedded mtp.* (0731 does).
#include <unordered_map>
#include <chrono>
#include <deque>
#include "weight_store.h"
#include "nvfp4_dense.h"
#include "deepseek_v4.h"
#include "block.h"
#include "suffix_draft.h"   // S6 counterfactual probe (DSV4_SUFFIXPROBE=1); gated by gate_suffix_draft
#include "compressed_block.h"
#include "block_decode.h"
#include "hc.h"
#include "mla_attn.h"
#include "compressor.h"
#include "dscratch.h"
#include "dprof.h"
// void-returning CU for use inside lambdas that cannot return an int
#define CU_V(x) do{ cudaError_t e_=(x); if(e_){ fprintf(stderr,"cuda %s @%d\n",cudaGetErrorString(e_),__LINE__); abort(); } }while(0)
#include "dspark_real.h"   // DSpark head: main_x, tap_pool, forward_head, markov
#include "dspark_attn.h"   // dspark_main_kv, dspark_block_forward
#include "kv_pack.h"       // DECODE_LADDER 1b.2 packed KV storage
#include "yarn.h"
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <vector>
#include <string>
#include <cstdio>
#include <cstring>
#include <memory>
#include <cmath>
using namespace dsv4;
#define CU(x) do{cudaError_t e=(x); if(e){fprintf(stderr,"cuda %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)

static std::string key_map(const std::string& in){ std::string s=in; if(s.rfind("model.",0)==0) s=s.substr(6); return s; }

__global__ void k_deq_e8m0(float* o, const uint8_t* in, size_t n){ size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x; if(i<n) o[i]=exp2f((float)in[i]-127.f); }
__global__ void k_deq_bf16(float* o, const __nv_bfloat16* in, size_t n){ size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x; if(i<n) o[i]=__bfloat162float(in[i]); }
__global__ void k_deq_fp8_blk(float* o, const uint8_t* w, const uint8_t* sc, int rows, int cols, int blk){
    size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x; if(i>=(size_t)rows*cols) return; int r=i/cols, c=i%cols;
    __half_raw hr=__nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)w[i], __NV_E4M3);
    float wv=__half2float(*reinterpret_cast<__half*>(&hr));
    int scw=cols/blk; float sv=exp2f((float)sc[(size_t)(r/blk)*scw + c/blk]-127.f);
    o[i]=wv*sv;
}
__global__ void k_embed(float* h, const __nv_bfloat16* emb, const int* ids, int s, int dim){
    size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x; if(i>=(size_t)s*dim) return; int t=i/dim, j=i%dim;
    h[i]=__bfloat162float(emb[(size_t)ids[t]*dim + j]);
}
__global__ void k_hc_expand(float* out, const float* h, int s, int hc, int dim){
    size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x; if(i>=(size_t)s*hc*dim) return; int t=i/(hc*dim), j=i%dim;
    out[i]=h[(size_t)t*dim+j];
}

struct Loader {
    st::WeightStore& W; std::vector<void*> allocs;
    Loader(st::WeightStore& w):W(w){}
    ~Loader(){ for(void*p:allocs) cudaFree(p); }
    size_t mark(){ return allocs.size(); }
    void release(size_t m){ for(size_t i=m;i<allocs.size();++i) cudaFree(allocs[i]); allocs.resize(m); }
    const uint8_t* raw(const std::string& n){ return W.dev<uint8_t>(n); }
    const float* f32(const std::string& n){ return W.dev<float>(n); }
    float* alloc(size_t nb){ void* p; CU(cudaMalloc(&p,nb)); allocs.push_back(p); return (float*)p; }
    const float* scale(const std::string& n){ auto& t=W.get(n); size_t ne=t.numel(); float* o=alloc(ne*4);
        k_deq_e8m0<<<(ne+255)/256,256>>>(o,(const uint8_t*)t.dev,ne); return o; }
    const float* bf16(const std::string& n){ auto& t=W.get(n); size_t ne=t.numel(); float* o=alloc(ne*4);
        k_deq_bf16<<<(ne+255)/256,256>>>(o,(const __nv_bfloat16*)t.dev,ne); return o; }
    const float* wo_a(const std::string& wn, const std::string& sn){ auto& t=W.get(wn);
        int rows=t.shape[0], cols=t.shape[1]; size_t ne=(size_t)rows*cols; float* o=alloc(ne*4);
        k_deq_fp8_blk<<<(ne+255)/256,256>>>(o,(const uint8_t*)t.dev,(const uint8_t*)W.get(sn).dev,rows,cols,128); return o; }
};
static const float* up_f(const std::vector<float>& v, std::vector<void*>& keep){
    void* d; CU(cudaMalloc(&d,v.size()*4)); CU(cudaMemcpy(d,v.data(),v.size()*4,cudaMemcpyHostToDevice)); keep.push_back(d); return (const float*)d; }
static std::vector<float> stride_rows(const std::vector<float>& in, int s, int half, int ratio){
    int ng=s/ratio; if(ng<1) ng=1;                        // >=1 row: the device-pos emit reads cc[d_g] every step (uncommitted too)
    std::vector<float> o((size_t)ng*half); for(int g=0; g<ng; ++g) for(int j=0;j<half;++j) o[(size_t)g*half+j]=in[(size_t)(g*ratio)*half+j]; return o; }

int main(int argc, char** argv){
    setvbuf(stdout, nullptr, _IONBF, 0);
    const char* dir = argc>1?argv[1]:"/home/patrickd/models/DeepSeek-V4-Flash-180B";
    std::vector<int> ids;
    if(argc>2 && strchr(argv[2],',')){ char* tok=strtok(argv[2],","); while(tok){ ids.push_back(atoi(tok)); tok=strtok(nullptr,","); } }
    else { for(int i=0;i<6;++i) ids.push_back((int[]){0,671,6102,294,8760,344}[i]); }  // BOS + "The capital of France is"
    int s = ids.size();
    int NDEC = argc>3?atoi(argv[3]):6;                 // tokens to decode (autoregressive) after prefill
    int NGEN0 = argc>5?atoi(argv[5]):24;               // spec-decode tokens (if head given)
    int PS = s-1;                                      // prefill positions 0..PS-1; decode starts at pos PS (=s-1)
    // PROMPT SET. argv[2] is prompt 0 and stays the gate prompt: the base-AR path below is
    // unchanged and still returns nonzero unless it emits 11111. DSV4_PROMPTS appends prompts
    // 1..N as ';'-separated id lists, which a sweep entry then selects by index. This exists
    // because the adaptive-verify threshold (Finding 49) was fitted on 18 verifies of ONE prompt,
    // and re-fitting it across prompts otherwise costs one 15-minute checkpoint load per prompt.
    // The ids must come from the checkpoint's own tokenizer (`tools/encode_prompt.py`, which gates
    // itself on reproducing prompt 0); inventing them is the mistake invariant "no invented model
    // constants" exists to stop.
    std::vector<std::vector<int>> prompts{ids};
    // DSV4_PROMPTS_FILE: one ';'-free id list per LINE. DSV4_PROMPTS is an environment variable and
    // 500 prompts x ~200 ids is ~500 KB, past the usual per-variable limit -- the S5 sessions need
    // hundreds of prompts in one checkpoint load, so they cannot go through the environment.
    if(const char* pf=getenv("DSV4_PROMPTS_FILE")){
        FILE* f=fopen(pf,"r");
        if(!f){ fprintf(stderr,"[parse] cannot open DSV4_PROMPTS_FILE %s\n",pf); return 2; }
        char* line=nullptr; size_t cap=0; ssize_t n; int nl=0;
        while((n=getline(&line,&cap,f))>0){
            std::vector<int> cur; const char* q=line;
            while(*q){ if(*q==','||*q==' '||*q=='\n'||*q=='\r'){ ++q; }
                       else { cur.push_back(atoi(q)); while(*q && *q!=',' && *q!=' ' && *q!='\n' && *q!='\r') ++q; } }
            if(cur.size()>=2){ prompts.push_back(cur); ++nl; }
        }
        free(line); fclose(f);
        printf("[parse] DSV4_PROMPTS_FILE %s -> %d prompt(s)\n", pf, nl);
    }
    if(const char* pv=getenv("DSV4_PROMPTS")){ std::vector<int> cur; const char* q=pv;
        while(*q){ if(*q==';'){ if(cur.size()>=2) prompts.push_back(cur); cur.clear(); ++q; }
                   else if(*q==','||*q==' '){ ++q; }
                   else { cur.push_back(atoi(q)); while(*q && *q!=',' && *q!=';' && *q!=' ') ++q; } }
        if(cur.size()>=2) prompts.push_back(cur); }
    // DSV4_BLKSWEEP="5,8,12,16" runs the spec-decode loop once per block size in ONE process. Each
    // full-model measurement otherwise costs a ~10-minute cold load of a 100 GiB checkpoint, which
    // makes a four-point sweep of the single most consequential speculation parameter a whole
    // afternoon. The sweep re-prefills between points, so the runs are independent.
    // Entries are "BLK" or "BLK:passes", e.g. "5:1,5:2" to A/B draft refinement at one block size.
    // Entries are "BLK[:passes[:adaptK[:prompt]]]", e.g. "5:1:0,5:1:0.5,5:1:1.5" to sweep the
    // adaptive-verify margin threshold at one block size, or "5:1:1.5:2" to run that setting on
    // prompt 2 of DSV4_PROMPTS. Each point re-prefills, so they are independent — one checkpoint
    // load answers what would otherwise be one 15-minute run per (threshold, prompt) pair.
    std::vector<int> blkSweep, passSweep, promptSweep; std::vector<float> adaptSweep;
    if(const char* bsz=getenv("DSV4_BLKSWEEP")){ const char* q=bsz;
        while(*q){ int v=atoi(q); int np=1; float ak=0.f; int pi=0;
                   const char* c=q; while(*c && *c!=',' && *c!=':') ++c;
                   if(*c==':'){ np=atoi(c+1); const char* c2=c+1; while(*c2 && *c2!=',' && *c2!=':') ++c2;
                                if(*c2==':'){ ak=(float)atof(c2+1); const char* c3=c2+1;
                                              while(*c3 && *c3!=',' && *c3!=':') ++c3;
                                              if(*c3==':') pi=atoi(c3+1); } }
                   if(v>=2&&v<=16){ blkSweep.push_back(v); passSweep.push_back(np<1?1:np); adaptSweep.push_back(ak);
                                    promptSweep.push_back(pi); }
                   while(*q && *q!=',') ++q; if(*q==',') ++q; } }
    // DEFAULT SPECULATION BLOCK = 6, and it is deliberately NOT `DSPARK_BLOCK`.
    //
    // `DSPARK_BLOCK` is 5 because `config.json` says `dspark_block_size: 5` -- that is the width the
    // head was TRAINED at, a model constant, and it stays untouched. The number of proposals the
    // ENGINE asks for is a serving decision, and it was 6 from F94 until LADDER 2.1 re-measured it
    // against a verify that is no longer free.
    //
    // F94's 6 was correct FOR ITS ENGINE. `k_topk_verify<<<K,32>>>` ran one active thread per
    // block, so a verified position cost almost nothing in the context term and the only thing
    // width bought was acceptance. Ladder 1.1/1.2 replaced that kernel; 2.1 then priced both sides
    // out of 13,392 verify rounds: one more DRAFTED position (verified or not) costs
    // +3.324 +/- 0.281 ms and one more VERIFIED position +15.184 +/- 0.396 ms. The draft forward
    // runs at M=BLK whatever adaptK later decides, so every proposal the verify never reaches is
    // paid for in full -- and at BLK=6 the mean realised verify width is 3.71 of a ceiling of 7,
    // i.e. 3.29 proposals per round, 10.7 ms, are drafted and thrown away.
    //
    // MEASURED, 32 prompts, one load, palindromic order, drift 0.000 +/- 0.002 tok/s:
    //   BLK=5 vs 6  +3.91 +/- 1.65 % tok/s, 26/32 legs, d tau -0.052 +/- 0.084  (tau band covers 0)
    //   BLK=4 vs 6  +3.78 +/- 2.15 % tok/s, 23/32 legs, d tau -0.294 +/- 0.147  (tau really falls)
    //   BLK=4 vs 5  -0.12 +/- 1.43 % tok/s               d tau -0.242 +/- 0.102
    // So 5 is an INTERIOR optimum -- 4 buys no throughput over it and does cost acceptance -- and
    // it is exactly the width the head was trained at. 7..12 were measured in the first sweep and
    // fall monotonically to -10.5 %.
    //
    // F93 is still the prerequisite: the verify checks all BLK proposals (VB = BLK+1), so a block
    // of 5 has a ceiling of 6 accepted tokens, not 5.
    // DSV4_BLK overrides; DSV4_BLK=6 restores the pre-2.1 default for the A/B.
    if(blkSweep.empty()){
        const char* be = getenv("DSV4_BLK");
        int defblk = be ? atoi(be) : 5;
        if(defblk < 2 || defblk > 16) defblk = 5;
        blkSweep.push_back(defblk); passSweep.push_back(1); adaptSweep.push_back(0.f); promptSweep.push_back(0); }
    // A sweep point naming a prompt that was not supplied would silently run prompt 0 and report a
    // number against the wrong sequence. Refuse the whole run instead — a 10-minute load producing
    // a mislabelled table is the expensive failure here.
    for(size_t i2=0;i2<promptSweep.size();++i2)
        if(promptSweep[i2]<0 || promptSweep[i2]>=(int)prompts.size()){
            fprintf(stderr,"[decode] SWEEP FAIL: entry %zu names prompt %d but only %zu prompt(s) supplied "
                           "(argv[2] is prompt 0; DSV4_PROMPTS adds the rest)\n", i2, promptSweep[i2], prompts.size());
            return 2; }
    int BLKMAX=0; for(int v:blkSweep) if(v>BLKMAX) BLKMAX=v;
    // DSV4_ADAPTBLK can GROW the block past what the sweep asked for, and every draft-side device
    // buffer (dbid, dmarg, xemb, xa, xb) is sized from BLKMAX. A sweep at BLK0=5 sets BLKMAX=5, so
    // the controller stepping to 6 would overrun all five -- and, exactly like F93's verify-side
    // buffers, it would NOT crash: it would corrupt the last drafted proposal. Raise the ceiling
    // here, where the buffers are sized, not where the controller is written.
    if(getenv("DSV4_ADAPTBLK") && BLKMAX < 6) BLKMAX = 6;
    // VERIFY WIDTH vs DRAFT WIDTH -- they are NOT the same number, and conflating them cost us a
    // whole token of ceiling. The verify spends vtok[0] on the ALREADY-KNOWN token and fills
    // vtok[1..VB-1] from draft[], so verifying all BLK proposals needs VB = BLK+1. The drafter has
    // always produced BLK proposals (hmarg[] carries BLK margins) but the verify capped at VB=BLK,
    // so draft[BLK-1] was computed every round and thrown away, and tau could never exceed BLK.
    // That is why the DSpark paper reports 6.11 accepted length at gamma=5 while our ceiling read 5.
    // Every VERIFY-side buffer must be sized from this, not from BLKMAX; the DRAFT-side ones
    // (dbid, dmarg, xemb, xa, xb) stay at BLKMAX because the draft really is BLK wide.
    int VBMAX = BLKMAX + 1;
    int SMAX=0; for(const auto& p:prompts) if((int)p.size()>SMAX) SMAX=(int)p.size();
    int seqmax = SMAX + (NDEC>NGEN0?NDEC:NGEN0) + VBMAX + 8;   // longest prompt + room for spec block overshoot
    // Parse-only self-gate: the sweep table is the thing most likely to be wrong, and checking it
    // after a 10-minute checkpoint load is checking it too late. DSV4_PARSE_ONLY=1 prints what was
    // parsed and exits before the load, so the table can be gated in milliseconds.
    if(getenv("DSV4_PARSE_ONLY")){
        printf("[parse] %zu prompt(s), SMAX=%d, BLKMAX=%d, VBMAX=%d, seqmax=%d\n", prompts.size(), SMAX, BLKMAX, VBMAX, seqmax);
        for(size_t i2=0;i2<prompts.size();++i2){ printf("[parse] prompt %zu (%zu ids):", i2, prompts[i2].size());
            for(int v:prompts[i2]) printf(" %d", v); printf("\n"); }
        for(size_t i2=0;i2<blkSweep.size();++i2)
            printf("[parse] point %zu: BLK=%d passes=%d adaptK=%.2f prompt=%d\n",
                   i2, blkSweep[i2], passSweep[i2], adaptSweep[i2], promptSweep[i2]);
        return 0; }
    printf("[decode] loading %s ... s=%d NDEC=%d seqmax=%d\n", dir, s, NDEC, seqmax);
    st::WeightStore W(dir, key_map); Loader L(W);
    printf("[decode] loaded %.2f GiB, %zu tensors  (weights in %s memory)\n",
           W.loadedGiB(), W.count(), W.managed() ? "MANAGED device-preferred" : "mapped-host");
    const int half=ROPE_DIM/2, hc=HC_MULT, d=DIM;
    // OPTIONAL NVFP4 DENSE OVERLAY (DSV4_NVFP4_OVERLAY=<dir>). Registers an NVFP4 replacement for
    // each dense/MLA weight against the FP8 pointer the engine already holds, so the single dispatch
    // in fp8_block_gemm picks it up at M=1. Unset -> nothing is registered, nothing is looked up, and
    // the FP8 path is byte-identical to before. Untouched copies of every file this feature reaches
    // live in original-fp8-path/.
    if (const char* ov = getenv("DSV4_NVFP4_OVERLAY")) {
        struct Ctx { st::WeightStore* w; };
        Ctx cx{&W};
        nvfp4_load_overlay(ov, [](const char* name, void* c) -> const uint8_t* {
            auto* k = static_cast<Ctx*>(c);
            try { return (const uint8_t*)k->w->get(name).dev; } catch (...) { return nullptr; }
        }, &cx);
    }
    extern bool g_tc_fp8; g_tc_fp8=true; extern bool g_tc_ogroup; g_tc_ogroup=true;
    extern bool g_moe_grouped; g_moe_grouped=true; extern void tc_moe_clear_cache();
    // MoE fp4 GEMV is now the DEFAULT (LOOP_LOG Finding 31). It was off because its scalar
    // nibble-decode inner loop made it compute-bound and slower than the m16 mma. After the half2
    // dequant rewrite (cvt.f16x2.e2m1x2 + cvt.f16x2.e4m3x2 + __hfma2) it beats the mma path at
    // every M measured: 314.6 vs 121.5 GB/s at M=1, 232.2 vs 125.7 at M=2, 230.3 vs 130.9 at M=8.
    // MOE_MMA=1 restores the m16 tile for A/B.
    extern bool g_moe_gemv; g_moe_gemv=(getenv("MOE_MMA")==nullptr);
    extern bool g_compressor_bf16; g_compressor_bf16=(getenv("COMP_BF16")!=nullptr);  // Finding 32: OFF by default — measured SLOWER

    // freqs over seqmax
    std::vector<void*> keep;
    std::vector<float> ssc,sss; yarn::freqs(ssc,sss,seqmax,ROPE_DIM,0,ROPE_THETA,YARN_FACTOR,YARN_BETA_FAST,YARN_BETA_SLOW);
    const float *slide_cos=up_f(ssc,keep), *slide_sin=up_f(sss,keep);
    std::vector<float> cqc_h,cqs_h; yarn::freqs(cqc_h,cqs_h,seqmax,ROPE_DIM,YARN_ORIG_MAXPOS,COMPRESS_ROPE_THETA,YARN_FACTOR,YARN_BETA_FAST,YARN_BETA_SLOW);
    const float *cqc=up_f(cqc_h,keep), *cqs=up_f(cqs_h,keep);
    const float *cc4c=up_f(stride_rows(cqc_h,seqmax,half,4),keep), *cc4s=up_f(stride_rows(cqs_h,seqmax,half,4),keep);
    const float *cc128c=up_f(stride_rows(cqc_h,seqmax,half,128),keep), *cc128s=up_f(stride_rows(cqs_h,seqmax,half,128),keep);

    // DECODE_LADDER 1b.2. Must run BEFORE the first KV allocation: it sets the row stride every
    // allocation, memcpy and kernel below reads out of `g_kv_rowf`. See include/kv_pack.h.
    kv_pack_init();
    printf("[decode] KV layout: %s, %d B/row, %.3f GiB for %d rows x %d layers\n",
           g_kv_pack ? "PACKED fp8+ue8m0 (1b.2)" : "fp32",
           g_kv_rowf*4, (double)kv_rows_bytes((size_t)seqmax*N_LAYERS)/1073741824.0, seqmax, N_LAYERS);
    // per-layer KV caches
    std::vector<LayerKV> KV(N_LAYERS);
    for(int Lyr=0; Lyr<N_LAYERS; ++Lyr){ int ratio=compress_ratio(Lyr);
        CU(cudaMalloc(&KV[Lyr].win_kv,kv_rows_bytes(seqmax)));
        if(ratio){ CU(cudaMalloc(&KV[Lyr].xin,(size_t)seqmax*DIM*4));
            CU(cudaMalloc(&KV[Lyr].comp_kv,kv_rows_bytes(seqmax/ratio+2)));
            if(ratio==4) CU(cudaMalloc(&KV[Lyr].idx_ckv,(size_t)(seqmax/ratio+2)*INDEX_HEAD_DIM*4)); }
    }
    int* d_ids; CU(cudaMalloc(&d_ids,seqmax*4));
    float *h0,*h,*h2,*collapsed,*logits;
    CU(cudaMalloc(&h0,(size_t)seqmax*d*4)); CU(cudaMalloc(&h,(size_t)seqmax*hc*d*4)); CU(cudaMalloc(&h2,(size_t)seqmax*hc*d*4));
    CU(cudaMalloc(&collapsed,(size_t)d*4)); CU(cudaMalloc(&logits,(size_t)VOCAB*4));
    // head weights (persistent)
    // lm_head is BF16 on disk; do NOT dequantise it (LOOP_LOG Finding 26). Loader::bf16 would
    // materialise 2.118 GB of f32 and read all of it every step. gemm_bf16w reads the bf16 natively.
    const void  *head_bf = (const void*)W.get("head.weight").dev;
    const float *norm_w  = L.bf16("norm.weight");
    const float *hc_fn=L.f32("hc_head_fn"), *hc_sc=L.f32("hc_head_scale"), *hc_bs=L.f32("hc_head_base");
    size_t head_mark=L.mark();                                   // keep head + freqs; per-layer dequant is above this

    std::vector<std::vector<const uint8_t*>> P1(N_LAYERS),P2(N_LAYERS),P3(N_LAYERS);
    std::vector<std::vector<const uint8_t*>> S18(N_LAYERS),S28(N_LAYERS),S38(N_LAYERS);  // NATIVE e8m0 expert scales
    auto fill_moe=[&](const std::string& pfx, bool is_hash, MoEWeights& m, int Lyr){
        std::string p=pfx+"ffn."; auto& p1=P1[Lyr];auto&p2=P2[Lyr];auto&p3=P3[Lyr];auto&s1=S18[Lyr];auto&s2=S28[Lyr];auto&s3=S38[Lyr];
        p1.clear();p2.clear();p3.clear();s1.clear();s2.clear();s3.clear();
        m.gate_w=L.bf16(p+"gate.weight"); m.is_hash=is_hash;
        m.gate_bias=is_hash?nullptr:(W.has(p+"gate.bias")?L.f32(p+"gate.bias"):nullptr);
        m.tid2eid=is_hash?(const long*)W.get(p+"gate.tid2eid").dev:nullptr;
        for(int e=0;e<N_ROUTED;++e){ std::string ep=p+"experts."+std::to_string(e)+".";
            p1.push_back(L.raw(ep+"w1.weight")); p2.push_back(L.raw(ep+"w2.weight")); p3.push_back(L.raw(ep+"w3.weight"));
            s1.push_back(L.raw(ep+"w1.scale")); s2.push_back(L.raw(ep+"w2.scale")); s3.push_back(L.raw(ep+"w3.scale")); }  // e8m0 bytes, persistent (no dequant)
        m.w1p=p1.data();m.w2p=p2.data();m.w3p=p3.data();
        m.e8m0_scales=true; m.w1sp8=s1.data();m.w2sp8=s2.data();m.w3sp8=s3.data();
        std::string sp=p+"shared_experts.";
        m.sw1=L.raw(sp+"w1.weight");m.sw2=L.raw(sp+"w2.weight");m.sw3=L.raw(sp+"w3.weight");
        m.sw1s=L.scale(sp+"w1.scale");m.sw2s=L.scale(sp+"w2.scale");m.sw3s=L.scale(sp+"w3.scale");
        m.n_routed=N_ROUTED;m.n_act=N_ACT;m.dim=DIM;m.inter=MOE_INTER;m.vocab=VOCAB;m.route_scale=ROUTE_SCALE;m.swiglu_limit=SWIGLU_LIMIT;
        m.use_tc_pp=true;m.batched=true;m.device_route=true; };
    auto fill_attn=[&](const std::string& pfx, MLAWeights& a, bool compressed){
        std::string p=pfx+"attn.";
        a.wq_a=L.raw(p+"wq_a.weight");a.wq_a_s=L.scale(p+"wq_a.scale");a.wq_b=L.raw(p+"wq_b.weight");a.wq_b_s=L.scale(p+"wq_b.scale");
        a.wkv=L.raw(p+"wkv.weight");a.wkv_s=L.scale(p+"wkv.scale");a.wo_b=L.raw(p+"wo_b.weight");a.wo_b_s=L.scale(p+"wo_b.scale");
        a.q_norm=L.bf16(p+"q_norm.weight");a.kv_norm=L.bf16(p+"kv_norm.weight");
        a.wo_a_native=true; a.wo_a_fp8=L.raw(p+"wo_a.weight"); a.wo_a_sc=L.raw(p+"wo_a.scale");  // native fp8 (no dequant)
        a.attn_sink=L.f32(p+"attn_sink");
        a.cosT=compressed?cqc:slide_cos;a.sinT=compressed?cqs:slide_sin; };

    // build one layer's weights (dequant), run either prefill_cache (bs=PS) or a decode step (pos), then it's the
    // caller's job to L.release(mk). Returns via x_out.
    // Build EVERY layer's weight struct ONCE (persistent — experts + wo_a are native so residual dequant is
    // ~2 GB, fits). The decode loop then does zero per-token Loader work (no dequant, no cudaMalloc, no struct
    // rebuild). Memory-neutral enough: ~112 GiB peak.
    std::vector<BlockWeights> BW(N_LAYERS); std::vector<CompressedBlockWeights> CW(N_LAYERS);
    auto build_layer=[&](int Lyr){
        int ratio=compress_ratio(Lyr); std::string lp="layers."+std::to_string(Lyr)+".";
        if(ratio==0){
            BlockWeights& b=BW[Lyr]; fill_attn(lp,b.attn,false); fill_moe(lp,is_hash_layer(Lyr),b.ffn,Lyr);
            b.attn_norm=L.bf16(lp+"attn_norm.weight");b.ffn_norm=L.bf16(lp+"ffn_norm.weight");
            b.hc_attn_fn=L.f32(lp+"hc_attn_fn");b.hc_attn_scale=L.f32(lp+"hc_attn_scale");b.hc_attn_base=L.f32(lp+"hc_attn_base");
            b.hc_ffn_fn=L.f32(lp+"hc_ffn_fn");b.hc_ffn_scale=L.f32(lp+"hc_ffn_scale");b.hc_ffn_base=L.f32(lp+"hc_ffn_base");
            b.dim=DIM;b.hc=HC_MULT;
        } else {
            CompressedBlockWeights& b=CW[Lyr]; fill_attn(lp,b.attn.attn,true);
            std::string p=lp+"attn.";
            // Finding 32/33: f32-dequantised. The BF16-native variant is SLOWER with the current
            // gemm and — critically — `emit_group_dp` reaches these same pointers through
            // `gemm_fp32_cond`, which has no bf16 variant, so a bf16 pointer here is silently
            // misread on the device-pos decode path regardless of COMP_BF16. One pointer type.
            b.attn.mc_wkv=L.bf16(p+"compressor.wkv.weight");
            b.attn.mc_wgate=L.bf16(p+"compressor.wgate.weight");
            b.attn.mc_ape=L.f32(p+"compressor.ape");b.attn.mc_norm=L.bf16(p+"compressor.norm.weight");
            b.attn.cc_cos=(ratio==4)?cc4c:cc128c;b.attn.cc_sin=(ratio==4)?cc4s:cc128s;
            if(ratio==4){
                b.attn.idx_wq_b=L.raw(p+"indexer.wq_b.weight");b.attn.idx_wq_b_s=L.scale(p+"indexer.wq_b.scale");
                b.attn.idx_weights_proj=L.bf16(p+"indexer.weights_proj.weight");
                b.attn.idx_c_wkv=L.bf16(p+"indexer.compressor.wkv.weight");   // see above (Finding 33)
                b.attn.idx_c_wgate=L.bf16(p+"indexer.compressor.wgate.weight");
                b.attn.idx_c_ape=L.f32(p+"indexer.compressor.ape");b.attn.idx_c_norm=L.bf16(p+"indexer.compressor.norm.weight");
            }
            b.attn.index_n_heads=INDEX_N_HEADS;b.attn.index_head_dim=INDEX_HEAD_DIM;b.attn.index_topk=INDEX_TOPK;
            fill_moe(lp,is_hash_layer(Lyr),b.ffn,Lyr);
            b.attn_norm=L.bf16(lp+"attn_norm.weight");b.ffn_norm=L.bf16(lp+"ffn_norm.weight");
            b.hc_attn_fn=L.f32(lp+"hc_attn_fn");b.hc_attn_scale=L.f32(lp+"hc_attn_scale");b.hc_attn_base=L.f32(lp+"hc_attn_base");
            b.hc_ffn_fn=L.f32(lp+"hc_ffn_fn");b.hc_ffn_scale=L.f32(lp+"hc_ffn_scale");b.hc_ffn_base=L.f32(lp+"hc_ffn_base");
            b.dim=DIM;b.hc=HC_MULT;b.win=WINDOW;b.ratio=ratio;
        }
    };
    // `npre` = prefill positions, EXPLICIT (ignored when !prefill). It used to be read from the
    // captured outer `PS`, and LOOP_LOG Finding 52 is what that cost: the multi-prompt sweep loop
    // declares its own `const int PS = s-1` for the point's prompt, but shadowing is LEXICAL, so
    // this body kept binding to line 96's argv-prompt PS. Every sweep point on a non-argv prompt
    // prefilled the KV caches to the WRONG length while everything else used the right one. A
    // parameter cannot be shadowed out from under the callee; a capture can.
    auto run_layer=[&](int Lyr, bool prefill, int pos, const float* x_in, float* x_out, const int* ids_dev, int npre, cudaStream_t st=0){
        int ratio=compress_ratio(Lyr);
        if(ratio==0){
            if(prefill) block_prefill_cache(x_out,x_in,ids_dev,BW[Lyr],npre,HC_SINKHORN_ITERS,EPS,KV[Lyr],st);
            else        block_decode_step (x_out,x_in,ids_dev,BW[Lyr],pos,HC_SINKHORN_ITERS,EPS,KV[Lyr],st);
        } else {
            if(prefill) cblock_prefill_cache(x_out,x_in,ids_dev,CW[Lyr],npre,HC_SINKHORN_ITERS,EPS,KV[Lyr],st);
            else        cblock_decode_step  (x_out,x_in,ids_dev,CW[Lyr],pos,HC_SINKHORN_ITERS,EPS,KV[Lyr],st);
        }
    };
    auto head_fwd=[&](const float* hstate, int* out_am){       // hc_head->norm->lm_head->argmax (1 token)
        hc_head(collapsed,hstate,hc_fn,hc_sc,hc_bs,1,hc,d,HC_EPS);
        rmsnorm(collapsed,collapsed,norm_w,1,d,EPS,true,0);
        gemm_bf16w(logits,collapsed,head_bf,1,VOCAB,d,0); CU(cudaDeviceSynchronize());
        std::vector<float> lg(VOCAB); CU(cudaMemcpy(lg.data(),logits,VOCAB*4,cudaMemcpyDeviceToHost));
        int am=0; for(int v=1;v<VOCAB;++v) if(lg[v]>lg[am]) am=v; *out_am=am; };

    // ---------------- PREFILL: populate caches over [id0..id_{PS-1}] ----------------
    CU(cudaMemcpy(d_ids,ids.data(),s*4,cudaMemcpyHostToDevice));
    k_embed<<<((size_t)PS*d+255)/256,256>>>(h0,(const __nv_bfloat16*)W.get("embed.weight").dev,d_ids,PS,d);
    k_hc_expand<<<((size_t)PS*hc*d+255)/256,256>>>(h,h0,PS,hc,d); CU(cudaDeviceSynchronize());
    printf("[decode] building 43 layer structs once (persistent)...\n");
    for(int Lyr=0; Lyr<N_LAYERS; ++Lyr) build_layer(Lyr);       // all dequant done ONCE, resident (~2 GB)
    { size_t fb,tb; cudaMemGetInfo(&fb,&tb); printf("[decode] structs built. mem %.1f/%.1f GiB\n",(tb-fb)/1073741824.0,tb/1073741824.0); }
    // I3 STRESS KNOB (DSV4_BALLAST_GB=n). The run that faulted (cycle 5) sat at 111.5/122.8 GiB after
    // building its structs; a faithful re-run of its sweep on the current tree sits at 108.8 and does
    // NOT fault, twice, with and without the sync probe. Headroom is the one variable that differs and
    // it is the exact variable I3's hypothesis is about, so make it settable instead of hoping the
    // environment reproduces it. This reserves device memory and never frees it.
    if(const char* bg = getenv("DSV4_BALLAST_GB")){
        double gb = atof(bg);
        if(gb > 0){ void* bp=nullptr; size_t nb=(size_t)(gb*1073741824.0);
            cudaError_t be = cudaMalloc(&bp, nb);
            size_t fb,tb; cudaMemGetInfo(&fb,&tb);
            printf("[ballast] reserved %.2f GiB -> %s; mem now %.1f/%.1f GiB (free %.2f)\n",
                   gb, be==cudaSuccess?"ok":cudaGetErrorString(be),
                   (tb-fb)/1073741824.0, tb/1073741824.0, fb/1073741824.0);
            if(be!=cudaSuccess) cudaGetLastError(); }
    }
    // Decode scratch arena (bump, reset per layer). 512 MB was sized for the 6-token gate prompt and
    // silently became a prompt-length ceiling: a PS=1023 prefill asks for 538451712 bytes and the
    // allocator reports `arena overflow 538451712>536870912`, killing the run. The arena holds
    // per-position intermediates, so its high-water mark scales with the longest prompt, not with a
    // constant. Sized from SMAX; the box has 114 GiB free, so the headroom costs nothing and a
    // too-small ceiling costs the whole run.
    size_t arena_bytes = (size_t)512<<20;
    if(SMAX > 512) arena_bytes = (size_t)(512 + (size_t)SMAX*2) << 20;
    arena_init(arena_bytes);
    // dprof_init() used to be called ONLY inside the DSV4_KSWEEP branch, which runs after the
    // prefill -- so g_dprof_on was still false while the prefill executed and every mark in it was
    // silently dropped. The first B9 profiling run cost a full 20-minute checkpoint load and
    // produced a log with no [dprof] line at all, which is the cheap version of this failure: the
    // expensive version is a report that looks complete because the marks it is missing never
    // announce themselves. init here, before any model work; it is idempotent and compiles to a
    // branch when DSV4_DPROF is unset.
    dprof_init();
    printf("[decode] prefill %d positions...\n", PS);
    // Timed because prefill throughput -- not the M=1 decode rate -- is what sizes a teacher-forced
    // hidden-state capture for a draft-head fine-tune (LEVERS.md S5). One batched forward over PS
    // positions, so tok/s here is the corpus-ingest rate. Synchronise on both sides: the loop is
    // async and an untimed launch queue would report the enqueue cost, which is trap 3.
    CU(cudaDeviceSynchronize());
    auto pf_t0 = std::chrono::steady_clock::now();
    for(int Lyr=0; Lyr<N_LAYERS; ++Lyr){ arena_reset();
        run_layer(Lyr,true,0,h,h2,d_ids,PS); std::swap(h,h2);     // structs prebuilt -> no per-token Loader work
    }
    CU(cudaDeviceSynchronize());
    double pf_ms = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-pf_t0).count();
    printf("[decode] PREFILL: %d positions in %.1f ms = %.1f tok/s (%.3f ms/tok)\n",
           PS, pf_ms, PS*1000.0/pf_ms, pf_ms/PS);
    printf("[decode] prefill done. caches populated. starting decode.\n");

    // ---------------- DECODE: autoregressive M=1 ----------------
    cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    int cur = ids[s-1]; int first_am=-1; std::vector<int> gen;
    float total_ms=0;
    float *hd, *hd2; CU(cudaMalloc(&hd,(size_t)hc*d*4)); CU(cudaMalloc(&hd2,(size_t)hc*d*4));
    for(int step=0; step<NDEC; ++step){
        int pos = (s-1) + step;                                 // decode token `cur` at absolute position pos
        int* cur_dev; cur_dev=d_ids+pos; CU(cudaMemcpy(cur_dev,&cur,4,cudaMemcpyHostToDevice));
        cudaEventRecord(t0);
        k_embed<<<((size_t)d+255)/256,256>>>(h0,(const __nv_bfloat16*)W.get("embed.weight").dev,cur_dev,1,d);
        k_hc_expand<<<((size_t)hc*d+255)/256,256>>>(hd,h0,1,hc,d);
        float* xin=hd; float* xout=hd2;
        for(int Lyr=0; Lyr<N_LAYERS; ++Lyr){ arena_reset();
            run_layer(Lyr,false,pos,xin,xout,cur_dev,0); std::swap(xin,xout);
        }
        int am; head_fwd(xin,&am);
        cudaEventRecord(t1); cudaEventSynchronize(t1); float ms=0; cudaEventElapsedTime(&ms,t0,t1);
        if(step>0) total_ms+=ms;                                // exclude step 0 (warmup: repack + first dequant)
        if(step==0) first_am=am;
        gen.push_back(am); cur=am;
        printf("  step %d pos %d -> token %d  (%.1f ms%s)\n", step, pos, am, ms, step==0?" warmup":"");
    }
    double warm_ms = NDEC>1 ? total_ms/(NDEC-1) : total_ms;
    const char* ev=getenv("DSV4_EXPECT"); const int EXPECT = ev?atoi(ev):11111;   // 11111 = " Paris"
    printf("\n[decode] first decoded token argmax = %d  (expect %d)  -> %s\n", first_am, EXPECT, first_am==EXPECT?"GATE PASS":"GATE FAIL");
    printf("[decode] generated:"); for(int g:gen) printf(" %d",g); printf("\n");
    printf("[decode] WARM decode: %.1f ms/tok = %.2f tok/s  (M=1 steady state, %d-step avg)\n", warm_ms, 1000.0/warm_ms, NDEC-1);

    // ================= FULL-STEP CUDA GRAPH (device-pos) — capture once, replay per token =================
    if(getenv("GRAPH")){
        const int winmax=seqmax, Tmax=seqmax/4+2;
        int *d_pos,*d_g,*d_curid; CU(cudaMalloc(&d_pos,4)); CU(cudaMalloc(&d_g,4)); CU(cudaMalloc(&d_curid,4));
        for(int L=0;L<N_LAYERS;++L){ int ratio=compress_ratio(L); if(!ratio) continue;
            CU(cudaMalloc(&KV[L].kvc,kv_rows_bytes((size_t)winmax+Tmax))); CU(cudaMalloc(&KV[L].d_T,4));
            if(ratio==4) CU(cudaMalloc(&KV[L].idx_kvc,(size_t)Tmax*INDEX_HEAD_DIM*4)); }
        // dp prefill: reset + re-run normal prefill, then copy separate caches -> combined kvc/idx_kvc + device d_T
        for(int L=0;L<N_LAYERS;++L) KV[L].T=0;
        k_embed<<<((size_t)PS*d+255)/256,256>>>(h0,(const __nv_bfloat16*)W.get("embed.weight").dev,d_ids,PS,d);
        k_hc_expand<<<((size_t)PS*hc*d+255)/256,256>>>(h,h0,PS,hc,d); CU(cudaDeviceSynchronize());
        for(int Lyr=0; Lyr<N_LAYERS; ++Lyr){ arena_reset(); run_layer(Lyr,true,0,h,h2,d_ids,PS); std::swap(h,h2); }
        for(int L=0;L<N_LAYERS;++L){ int ratio=compress_ratio(L); if(!ratio) continue;
            CU(cudaMemcpy(KV[L].kvc,KV[L].win_kv,kv_rows_bytes(PS),cudaMemcpyDeviceToDevice));
            CU(cudaMemcpy(kv_row(KV[L].kvc,winmax),KV[L].comp_kv,kv_rows_bytes(KV[L].T),cudaMemcpyDeviceToDevice));
            if(ratio==4) CU(cudaMemcpy(KV[L].idx_kvc,KV[L].idx_ckv,(size_t)KV[L].T*INDEX_HEAD_DIM*4,cudaMemcpyDeviceToDevice));
            CU(cudaMemcpy(KV[L].d_T,&KV[L].T,4,cudaMemcpyHostToDevice)); }
        // build + capture the 43-layer step (input hd, device-pos)
        cudaStream_t cap; CU(cudaStreamCreate(&cap));
        auto build_step=[&](cudaStream_t st)->float*{ float* vin=hd; float* vout=hd2;   // reuse decode ping-pong buffers
            for(int L=0;L<N_LAYERS;++L){ arena_reset(); int ratio=compress_ratio(L);
                if(ratio==0) block_decode_step_dp (vout,vin,d_curid,BW[L],d_pos,winmax,HC_SINKHORN_ITERS,EPS,KV[L],st);
                else         cblock_decode_step_dp(vout,vin,d_curid,CW[L],d_pos,d_g,winmax,Tmax,HC_SINKHORN_ITERS,EPS,KV[L],st);
                std::swap(vin,vout); } return vin; };
        int cur=ids[s-1]; int p=PS;
        k_embed<<<((size_t)d+255)/256,256>>>(h0,(const __nv_bfloat16*)W.get("embed.weight").dev,d_ids+PS,1,d); k_hc_expand<<<((size_t)hc*d+255)/256,256>>>(hd,h0,1,hc,d);
        CU(cudaMemcpy(d_pos,&p,4,cudaMemcpyHostToDevice)); CU(cudaMemcpy(d_curid,&cur,4,cudaMemcpyHostToDevice));
        build_step(cap); CU(cudaStreamSynchronize(cap));                 // warm (also advances caches for pos PS) — re-prefill after
        for(int L=0;L<N_LAYERS;++L) KV[L].T=0;
        k_embed<<<((size_t)PS*d+255)/256,256>>>(h0,(const __nv_bfloat16*)W.get("embed.weight").dev,d_ids,PS,d); k_hc_expand<<<((size_t)PS*hc*d+255)/256,256>>>(h,h0,PS,hc,d); CU(cudaDeviceSynchronize());
        for(int Lyr=0; Lyr<N_LAYERS; ++Lyr){ arena_reset(); run_layer(Lyr,true,0,h,h2,d_ids,PS); std::swap(h,h2); }
        for(int L=0;L<N_LAYERS;++L){ int ratio=compress_ratio(L); if(!ratio) continue;
            CU(cudaMemcpy(KV[L].kvc,KV[L].win_kv,kv_rows_bytes(PS),cudaMemcpyDeviceToDevice));
            CU(cudaMemcpy(kv_row(KV[L].kvc,winmax),KV[L].comp_kv,kv_rows_bytes(KV[L].T),cudaMemcpyDeviceToDevice));
            if(ratio==4) CU(cudaMemcpy(KV[L].idx_kvc,KV[L].idx_ckv,(size_t)KV[L].T*INDEX_HEAD_DIM*4,cudaMemcpyDeviceToDevice));
            CU(cudaMemcpy(KV[L].d_T,&KV[L].T,4,cudaMemcpyHostToDevice)); }
        arena_reset(); float* fbuf; CU(cudaStreamBeginCapture(cap,cudaStreamCaptureModeThreadLocal)); fbuf=build_step(cap);
        cudaGraph_t g; CU(cudaStreamEndCapture(cap,&g)); cudaGraphExec_t exec; CU(cudaGraphInstantiate(&exec,g,0));
        printf("\n[graph] captured FULL 43-layer decode step OK. replaying...\n");
        std::vector<int> ggen; int gm0=-1; cudaEvent_t g0,g1; cudaEventCreate(&g0); cudaEventCreate(&g1); float gms=0; int gt=0;
        cur=ids[s-1];
        for(int step=0; step<NDEC; ++step){ p=PS+step;
            cudaEventRecord(g0);
            CU(cudaMemcpy(d_curid,&cur,4,cudaMemcpyHostToDevice)); CU(cudaMemcpy(d_pos,&p,4,cudaMemcpyHostToDevice));
            k_embed<<<((size_t)d+255)/256,256,0,cap>>>(h0,(const __nv_bfloat16*)W.get("embed.weight").dev,d_curid,1,d); k_hc_expand<<<((size_t)hc*d+255)/256,256,0,cap>>>(hd,h0,1,hc,d);
            CU(cudaGraphLaunch(exec,cap)); CU(cudaStreamSynchronize(cap));
            int am; head_fwd(fbuf,&am);
            cudaEventRecord(g1); cudaEventSynchronize(g1); float ms=0; cudaEventElapsedTime(&ms,g0,g1);
            if(step>0){ gms+=ms; gt++; } if(step==0) gm0=am; ggen.push_back(am); cur=am; }
        double gwarm=gt?gms/gt:0;
        printf("[graph] first token argmax=%d (expect %d) -> %s\n", gm0, EXPECT, gm0==EXPECT?"GATE PASS":"GATE FAIL");
        printf("[graph] generated:"); for(int x:ggen) printf(" %d",x); printf("\n");
        printf("[graph] WARM: %.1f ms/tok = %.2f tok/s  vs non-graph %.1f ms/tok -> %.2fx\n", gwarm, 1000.0/gwarm, warm_ms, warm_ms/gwarm);
    }

    // ================= SPEC-DECODE M=K VERIFY equivalence gate + timing =================
    // Verify the SAME K tokens the autoregressive decode produced, in ONE M=K forward. Its per-position argmax
    // must equal the decode's tokens (gen), and it costs ~1 forward for K tokens (the spec-decode weight-share win).
    int VK = NDEC<DSPARK_BLOCK?NDEC:DSPARK_BLOCK;
    if(VK>=2){
        std::vector<int> vtok(VK); vtok[0]=ids[s-1]; for(int i=1;i<VK;++i) vtok[i]=gen[i-1];   // decode INPUTS at [PS..PS+VK-1]
        for(int L=0;L<N_LAYERS;++L) KV[L].T=0;                                                 // reset compressed caches
        k_embed<<<((size_t)PS*d+255)/256,256>>>(h0,(const __nv_bfloat16*)W.get("embed.weight").dev,d_ids,PS,d);
        k_hc_expand<<<((size_t)PS*hc*d+255)/256,256>>>(h,h0,PS,hc,d); CU(cudaDeviceSynchronize());
        for(int Lyr=0; Lyr<N_LAYERS; ++Lyr){ arena_reset(); run_layer(Lyr,true,0,h,h2,d_ids,PS); std::swap(h,h2); }  // re-prefill (reset window/comp caches)
        int* d_vtok; CU(cudaMalloc(&d_vtok,(size_t)VK*4)); CU(cudaMemcpy(d_vtok,vtok.data(),(size_t)VK*4,cudaMemcpyHostToDevice));
        CU(cudaMemcpy(d_ids+PS,vtok.data(),(size_t)VK*4,cudaMemcpyHostToDevice));               // ids for hash routing at [PS..]
        float *hv,*hv2,*collK,*logK; CU(cudaMalloc(&hv,(size_t)VK*hc*d*4)); CU(cudaMalloc(&hv2,(size_t)VK*hc*d*4));
        CU(cudaMalloc(&collK,(size_t)VK*d*4)); CU(cudaMalloc(&logK,(size_t)VK*VOCAB*4));
        k_embed<<<((size_t)VK*d+255)/256,256>>>(h0,(const __nv_bfloat16*)W.get("embed.weight").dev,d_vtok,VK,d);
        k_hc_expand<<<((size_t)VK*hc*d+255)/256,256>>>(hv,h0,VK,hc,d); CU(cudaDeviceSynchronize());
        cudaEventRecord(t0);
        float* vin=hv; float* vout=hv2;
        for(int Lyr=0; Lyr<N_LAYERS; ++Lyr){ arena_reset(); int ratio=compress_ratio(Lyr);
            if(ratio==0) block_verify_step (vout,vin,d_ids+PS,BW[Lyr],PS,VK,HC_SINKHORN_ITERS,EPS,KV[Lyr]);
            else         cblock_verify_step (vout,vin,d_ids+PS,CW[Lyr],PS,VK,HC_SINKHORN_ITERS,EPS,KV[Lyr]);
            std::swap(vin,vout); }
        hc_head(collK,vin,hc_fn,hc_sc,hc_bs,VK,hc,d,HC_EPS); rmsnorm(collK,collK,norm_w,VK,d,EPS,true,0);
        gemm_bf16w(logK,collK,head_bf,VK,VOCAB,d,0); CU(cudaDeviceSynchronize());
        cudaEventRecord(t1); cudaEventSynchronize(t1); float vms=0; cudaEventElapsedTime(&vms,t0,t1);
        std::vector<float> lg((size_t)VK*VOCAB); CU(cudaMemcpy(lg.data(),logK,(size_t)VK*VOCAB*4,cudaMemcpyDeviceToHost));
        std::vector<int> vam(VK); for(int t=0;t<VK;++t){const float*r=&lg[(size_t)t*VOCAB];int a=0;for(int v=1;v<VOCAB;++v)if(r[v]>r[a])a=v;vam[t]=a;}
        int match=0; for(int i=0;i<VK;++i) if(vam[i]==gen[i]) ++match;
        printf("\n[spec-verify] M=%d verify in ONE forward: %.1f ms (= %.1f ms/tok if all accepted vs %.1f M=1 -> %.2fx)\n", VK, vms, vms/VK, warm_ms, warm_ms/(vms/VK));
        printf("[spec-verify] verify argmax:"); for(int a:vam) printf(" %d",a); printf("\n");
        printf("[spec-verify] decode tokens:"); for(int i=0;i<VK;++i) printf(" %d",gen[i]); printf("\n");
        // match>=VK-1 tolerated: the only expected diffs are MoE-atomic near-ties (same tokens the decode flips
        // run-to-run). gate_mla_verify proves the M=K math bit-exact; full-model deterministic positions match.
        printf("[spec-verify] MATCH %d/%d -> %s  (M=K verify == K sequential decodes; diffs = MoE-atomic near-ties)\n",
               match, VK, match>=VK-1?"PASS":"FAIL");

        // ---- VERIFY GRAPH VALUE (VERIFYGRAPH=1) ----
        // Finding 47 left one question: how much of the verify is the exposed tail of ~600
        // serialised dependent launches? A reusable verify graph needs device-pos variants of every
        // verify kernel (~400 lines, mirroring the decode path's `_dp` family) because Tf, ntot,
        // topk and wmax all move with pos and T. But the QUESTION can be answered without any of
        // that: capture ONE verify at a fixed position and replay it. The replay recomputes the same
        // position, so the numbers it produces are meaningless — the TIME is exactly what a graph
        // would save. 40 lines to de-risk 400.
        if(getenv("VERIFYGRAPH")){
            cudaStream_t vs; CU(cudaStreamCreate(&vs));
            auto build_verify=[&](cudaStream_t st){ float* a=hv; float* b=hv2;
                for(int L=0;L<N_LAYERS;++L){ arena_reset(); int r=compress_ratio(L);
                    if(r==0) block_verify_step (b,a,d_ids+PS,BW[L],PS,VK,HC_SINKHORN_ITERS,EPS,KV[L],st);
                    else     cblock_verify_step(b,a,d_ids+PS,CW[L],PS,VK,HC_SINKHORN_ITERS,EPS,KV[L],st);
                    std::swap(a,b); }
                hc_head(collK,a,hc_fn,hc_sc,hc_bs,VK,hc,d,HC_EPS,st); rmsnorm(collK,collK,norm_w,VK,d,EPS,true,st);
                gemm_bf16w(logK,collK,head_bf,VK,VOCAB,d,st); };
            for(int L=0;L<N_LAYERS;++L) KV[L].T=0;                       // re-prefill: replay must not extend caches
            k_embed<<<((size_t)PS*d+255)/256,256>>>(h0,(const __nv_bfloat16*)W.get("embed.weight").dev,d_ids,PS,d);
            k_hc_expand<<<((size_t)PS*hc*d+255)/256,256>>>(h,h0,PS,hc,d); CU(cudaDeviceSynchronize());
            for(int L=0;L<N_LAYERS;++L){ arena_reset(); run_layer(L,true,0,h,h2,d_ids,PS); std::swap(h,h2); }
            std::vector<int> Tsnap(N_LAYERS); for(int L=0;L<N_LAYERS;++L) Tsnap[L]=KV[L].T;
            // ungraphed baseline on the same stream, same state
            const int VIT=5;
            arena_reset(); build_verify(vs); CU(cudaStreamSynchronize(vs));
            for(int L=0;L<N_LAYERS;++L) KV[L].T=Tsnap[L];
            cudaEvent_t a0,a1; cudaEventCreate(&a0); cudaEventCreate(&a1);
            cudaEventRecord(a0,vs);
            for(int i=0;i<VIT;++i){ build_verify(vs); for(int L=0;L<N_LAYERS;++L) KV[L].T=Tsnap[L]; }
            cudaEventRecord(a1,vs); CU(cudaStreamSynchronize(a1?vs:vs)); CU(cudaEventSynchronize(a1));
            float ums=0; cudaEventElapsedTime(&ums,a0,a1); ums/=VIT;
            for(int L=0;L<N_LAYERS;++L) KV[L].T=Tsnap[L];
            arena_reset();
            cudaGraph_t vg; cudaGraphExec_t vex;
            cudaError_t cerr = cudaStreamBeginCapture(vs, cudaStreamCaptureModeThreadLocal);
            if(cerr==cudaSuccess){ build_verify(vs); cerr = cudaStreamEndCapture(vs,&vg); }
            if(cerr!=cudaSuccess){
                printf("\n[vgraph] capture FAILED: %s — the verify path still contains something\n"
                       "[vgraph] uncapturable (pageable H2D, a sync, or a host callback).\n", cudaGetErrorString(cerr));
            } else {
                size_t nnodes=0; cudaGraphGetNodes(vg,nullptr,&nnodes);
                CU(cudaGraphInstantiate(&vex,vg,0));
                CU(cudaGraphLaunch(vex,vs)); CU(cudaStreamSynchronize(vs));
                cudaEventRecord(a0,vs);
                for(int i=0;i<VIT;++i) CU(cudaGraphLaunch(vex,vs));
                cudaEventRecord(a1,vs); CU(cudaEventSynchronize(a1));
                float gms=0; cudaEventElapsedTime(&gms,a0,a1); gms/=VIT;
                printf("\n[vgraph] M=%d verify, %zu graph nodes: ungraphed %.1f ms -> graph %.1f ms (%.2fx)\n",
                       VK, nnodes, ums, gms, ums/gms);
                printf("[vgraph] this replays ONE position, so the outputs are meaningless; the TIME is\n"
                       "[vgraph] what a device-pos verify graph would be worth per verify.\n");
                cudaGraphExecDestroy(vex); cudaGraphDestroy(vg);
            }
            for(int L=0;L<N_LAYERS;++L) KV[L].T=Tsnap[L];
            CU(cudaStreamDestroy(vs));
        }

        // ---- K-sweep + per-layer-flavour cost split (DSV4_KSWEEP=1) ----
        // Decisive test for LOOP_LOG Finding 13. The weight-traffic model in ROOFLINE.md §5 says
        // c_v(K) = (B_fixed + 43*|union|(K)*b_expert)/B_tok, i.e. 1.000/1.296/1.582/1.856/2.120
        // for K=1..5. Measured c_v(5) is 2.62. Finding 13 refuted the MoE explanation by code
        // inspection, leaving DSA (top-512 select + irregular gather PER QUERY POSITION) as the
        // suspect. If that is right, the excess must be concentrated in the 21 ratio-4 layers
        // (compressor + DSA indexer) and ABSENT from the 2 pure-sliding layers, which have neither.
        // Timing per flavour separates them directly: pure-sliding is the control.
        if(getenv("DSV4_KSWEEP")){
            dprof_init();
            std::vector<cudaEvent_t> ev(N_LAYERS+1);
            for(auto& e: ev) cudaEventCreate(&e);
            const double cv_model[6]={0,1.000,1.296,1.582,1.856,2.120};
            double TOT[6]={0},TS[6]={0},T128[6]={0},T4[6]={0};
            for(int K=1; K<=DSPARK_BLOCK; ++K){
                std::vector<int> kt(K); kt[0]=ids[s-1]; for(int i=1;i<K;++i) kt[i]=gen[i-1];
                for(int L=0;L<N_LAYERS;++L) KV[L].T=0;                       // reset caches, then re-prefill
                k_embed<<<((size_t)PS*d+255)/256,256>>>(h0,(const __nv_bfloat16*)W.get("embed.weight").dev,d_ids,PS,d);
                k_hc_expand<<<((size_t)PS*hc*d+255)/256,256>>>(h,h0,PS,hc,d); CU(cudaDeviceSynchronize());
                for(int L=0;L<N_LAYERS;++L){ arena_reset(); run_layer(L,true,0,h,h2,d_ids,PS); std::swap(h,h2); }
                CU(cudaMemcpy(d_vtok,kt.data(),(size_t)K*4,cudaMemcpyHostToDevice));
                CU(cudaMemcpy(d_ids+PS,kt.data(),(size_t)K*4,cudaMemcpyHostToDevice));
                k_embed<<<((size_t)K*d+255)/256,256>>>(h0,(const __nv_bfloat16*)W.get("embed.weight").dev,d_vtok,K,d);
                k_hc_expand<<<((size_t)K*hc*d+255)/256,256>>>(hv,h0,K,hc,d); CU(cudaDeviceSynchronize());
                float* a=hv; float* b=hv2;
                // The re-prefill above ALSO runs moe_forward / compressed attention, so their
                // sub-phase marks fire 43 times for prefill and 43 for the verify pass. Reporting
                // both made `moe:w1w3` (67 ms, mostly 5-token prefill) exceed its own parent
                // `MoE` (31 ms). Only the timed pass counts.
                dprof_reset();
                g_moe_union_sum=0; g_moe_union_calls=0; g_moe_rows_sum=0; g_moe_rows_max=0;
                for(int i_=0;i_<10;++i_) g_moe_rows_hist[i_]=0;   // verify-only: the re-prefill above is bs=PS, not bs=K
                cudaEventRecord(ev[0]);
                for(int L=0; L<N_LAYERS; ++L){ arena_reset(); int r=compress_ratio(L);
                    if(r==0) block_verify_step (b,a,d_ids+PS,BW[L],PS,K,HC_SINKHORN_ITERS,EPS,KV[L]);
                    else     cblock_verify_step(b,a,d_ids+PS,CW[L],PS,K,HC_SINKHORN_ITERS,EPS,KV[L]);
                    cudaEventRecord(ev[L+1]); std::swap(a,b); }
                // The head chain runs once per step and had never been timed. It is not in the
                // per-layer event split above (TOT stays layer-only, so the c_v column keeps
                // meaning what it has always meant); dprof reports it as its own top-level rows.
                dprof_begin(DP_HEAD_HC,0);
                hc_head(collK,a,hc_fn,hc_sc,hc_bs,K,hc,d,HC_EPS); rmsnorm(collK,collK,norm_w,K,d,EPS,true,0);
                dprof_end(DP_HEAD_HC,0);
                dprof_begin(DP_LM_HEAD,0);
                gemm_bf16w(logK,collK,head_bf,K,VOCAB,d,0);
                dprof_end(DP_LM_HEAD,0);
                CU(cudaDeviceSynchronize());
                { char tg[32]; snprintf(tg,sizeof tg,"K=%d",K); dprof_report(tg); }
                if(getenv("DSV4_MOEUNION") && g_moe_union_calls){
                    printf("[union] K=%d : mean distinct experts per layer = %.2f  (%d calls, top-%d of %d)\n",
                           K, (double)g_moe_union_sum/g_moe_union_calls, g_moe_union_calls, N_ACT, N_ROUTED);
                    printf("[union] K=%d : rows/expert mean %.2f max %d  hist(me=1..8,>8):",
                           K, (double)g_moe_rows_sum/g_moe_union_sum, g_moe_rows_max);
                    for(int i=1;i<=9;++i) printf(" %lld", g_moe_rows_hist[i]);
                    printf("\n"); fflush(stdout);
                    g_moe_union_sum=0; g_moe_union_calls=0; g_moe_rows_sum=0; g_moe_rows_max=0;
                    for(int i=0;i<10;++i) g_moe_rows_hist[i]=0; }
                double tot=0, ts=0, t128=0, t4=0;
                for(int L=0; L<N_LAYERS; ++L){ float dt; cudaEventElapsedTime(&dt,ev[L],ev[L+1]);
                    tot+=dt; int r=compress_ratio(L); (r==0?ts:(r==128?t128:t4)) += dt; }
                TOT[K]=tot; TS[K]=ts; T128[K]=t128; T4[K]=t4;
            }
            printf("\n[ksweep] 43-layer verify cost vs K (ms), split by layer flavour\n");
            printf("[ksweep]  K |   total | slide(2L) | r128(20L) |  r4(21L) | c_v meas | c_v model | excess\n");
            for(int K=1; K<=DSPARK_BLOCK; ++K)
                printf("[ksweep] %2d | %7.2f | %9.2f | %9.2f | %8.2f | %8.3f | %9.3f | %+6.3f\n",
                       K, TOT[K], TS[K], T128[K], T4[K], TOT[K]/TOT[1], cv_model[K], TOT[K]/TOT[1]-cv_model[K]);
            printf("\n[ksweep] per-flavour growth K=1 -> K=%d (the actual test):\n", DSPARK_BLOCK);
            printf("[ksweep]   slide (no compressor, no indexer) : %.3fx   <- control\n", TS[DSPARK_BLOCK]/TS[1]);
            printf("[ksweep]   r128  (compressor, NO indexer)    : %.3fx\n", T128[DSPARK_BLOCK]/T128[1]);
            printf("[ksweep]   r4    (compressor + DSA indexer)  : %.3fx\n", T4[DSPARK_BLOCK]/T4[1]);
            printf("[ksweep] DSA hypothesis holds IFF r4 grows materially faster than slide and r128.\n");
            for(auto& e: ev) cudaEventDestroy(e);
        }
    }
    // ================= DSpark SPEC-DECODE (draft head + accept loop) =================
    // On 0731-REAP the DSpark heads are EMBEDDED: mtp.0/1/2 ship inside the main checkpoint and are
    // already resident in W. Opening a second WeightStore for them (as the 180B project had to,
    // because its head lived in a separate repo) would duplicate ~6.5 GiB against ~16 GiB of
    // headroom — the exact memory-neutrality violation that hard-hung this box before. So: reuse
    // the main store by default, and only open a separate one if an explicit *different* head
    // directory is passed (kept for backward compatibility with an external head).
    const bool have_embedded_mtp = W.has("mtp.0.attn_norm.weight");
    // An EMPTY argv[4] means "no external head, I just want to reach argv[5]" — treat it as
    // absent rather than as a path, which threw `open failed: /model.safetensors.index.json`.
    const char* headdir = (argc>4 && argv[4][0]) ? argv[4] : (have_embedded_mtp ? dir : nullptr);
    const bool separate_head = (headdir && strcmp(headdir, dir) != 0);
    if(headdir){
        const int hf=ROPE_DIM/2;
        std::unique_ptr<st::WeightStore> WHp; std::unique_ptr<Loader> LHp;
        if(separate_head){
            printf("\n[spec] loading EXTERNAL DSpark head %s ...\n", headdir);
            WHp.reset(new st::WeightStore(headdir, key_map, "mtp.")); LHp.reset(new Loader(*WHp));
            printf("[spec] head loaded %.2f GiB, %zu mtp tensors (SEPARATE store)\n", WHp->loadedGiB(), WHp->count());
        } else {
            printf("\n[spec] using EMBEDDED DSpark heads from the main checkpoint (no extra memory)\n");
        }
        st::WeightStore& WH = separate_head ? *WHp : W;
        Loader&          LH = separate_head ? *LHp : L;
        std::vector<float> bc,bs2; yarn::freqs(bc,bs2,seqmax,ROPE_DIM,0,ROPE_THETA,YARN_FACTOR,YARN_BETA_FAST,YARN_BETA_SLOW);
        const float* blk_cos=up_f(bc,keep); const float* blk_sin=up_f(bs2,keep);
        const uint8_t* main_proj=LH.raw("mtp.0.main_proj.weight"); const float* main_proj_s=LH.scale("mtp.0.main_proj.scale");
        const float* main_norm=LH.bf16("mtp.0.main_norm.weight");
        int NSTAGE=0; while(WH.has("mtp."+std::to_string(NSTAGE)+".attn_norm.weight")) NSTAGE++;
        int NE=0; while(WH.has("mtp.0.ffn.experts."+std::to_string(NE)+".w1.weight")) NE++;
        printf("[spec] NSTAGE=%d head-experts=%d BLK sweep:", NSTAGE, NE);
        for(int v:blkSweep) printf(" %d",v); printf("\n");
        std::vector<BlockWeights> mb(NSTAGE); std::vector<float*> mkv(NSTAGE);
        std::vector<std::vector<const uint8_t*>> HP1(NSTAGE),HP2(NSTAGE),HP3(NSTAGE);
        // NATIVE e8m0 expert scales, exactly as fill_moe does for the main model. These used to be
        // `const float*` filled by LH.scale() (a dequant to f32) with `e8m0_scales` left false —
        // which the mma path handled correctly and the GEMV path silently misread as scale BYTES.
        // See LOOP_LOG Finding 39: that is why draft acceptance was 0/4 on every single verify.
        std::vector<std::vector<const uint8_t*>> HS1(NSTAGE),HS2(NSTAGE),HS3(NSTAGE);
        for(int st=0; st<NSTAGE; ++st){
            std::string b="mtp."+std::to_string(st)+".", p=b+"attn."; MLAWeights& a=mb[st].attn;
            a.wq_a=LH.raw(p+"wq_a.weight");a.wq_a_s=LH.scale(p+"wq_a.scale");a.wq_b=LH.raw(p+"wq_b.weight");a.wq_b_s=LH.scale(p+"wq_b.scale");
            a.wkv=LH.raw(p+"wkv.weight");a.wkv_s=LH.scale(p+"wkv.scale");a.wo_b=LH.raw(p+"wo_b.weight");a.wo_b_s=LH.scale(p+"wo_b.scale");
            a.q_norm=LH.bf16(p+"q_norm.weight");a.kv_norm=LH.bf16(p+"kv_norm.weight");
            a.wo_a=LH.wo_a(p+"wo_a.weight",p+"wo_a.scale");a.attn_sink=LH.f32(p+"attn_sink");a.cosT=blk_cos;a.sinT=blk_sin;
            mb[st].dim=DIM;mb[st].hc=HC_MULT;mb[st].attn_norm=LH.bf16(b+"attn_norm.weight");mb[st].ffn_norm=LH.bf16(b+"ffn_norm.weight");
            mb[st].hc_attn_fn=LH.f32(b+"hc_attn_fn");mb[st].hc_attn_scale=LH.f32(b+"hc_attn_scale");mb[st].hc_attn_base=LH.f32(b+"hc_attn_base");
            mb[st].hc_ffn_fn=LH.f32(b+"hc_ffn_fn");mb[st].hc_ffn_scale=LH.f32(b+"hc_ffn_scale");mb[st].hc_ffn_base=LH.f32(b+"hc_ffn_base");
            MoEWeights& m=mb[st].ffn; std::string fp=b+"ffn.";
            m.gate_w=LH.bf16(fp+"gate.weight");m.is_hash=false;m.gate_bias=WH.has(fp+"gate.bias")?LH.f32(fp+"gate.bias"):nullptr;m.tid2eid=nullptr;
            for(int e=0;e<NE;++e){ std::string ep=fp+"experts."+std::to_string(e)+".";
                HP1[st].push_back(LH.raw(ep+"w1.weight"));HP2[st].push_back(LH.raw(ep+"w2.weight"));HP3[st].push_back(LH.raw(ep+"w3.weight"));
                HS1[st].push_back(LH.raw(ep+"w1.scale"));HS2[st].push_back(LH.raw(ep+"w2.scale"));HS3[st].push_back(LH.raw(ep+"w3.scale")); }   // e8m0 bytes, no dequant
            m.w1p=HP1[st].data();m.w2p=HP2[st].data();m.w3p=HP3[st].data();
            m.e8m0_scales=true; m.w1sp8=HS1[st].data();m.w2sp8=HS2[st].data();m.w3sp8=HS3[st].data();
            std::string sp2=fp+"shared_experts."; m.sw1=LH.raw(sp2+"w1.weight");m.sw2=LH.raw(sp2+"w2.weight");m.sw3=LH.raw(sp2+"w3.weight");
            m.sw1s=LH.scale(sp2+"w1.scale");m.sw2s=LH.scale(sp2+"w2.scale");m.sw3s=LH.scale(sp2+"w3.scale");
            m.n_routed=NE;m.n_act=N_ACT;m.dim=DIM;m.inter=MOE_INTER;m.vocab=VOCAB;m.route_scale=ROUTE_SCALE;m.swiglu_limit=SWIGLU_LIMIT;
            m.use_tc_pp=true;m.batched=true;m.device_route=true; CU(cudaMalloc(&mkv[st],kv_rows_bytes(seqmax)));
        }
        std::string LS="mtp."+std::to_string(NSTAGE-1)+".";
        const float* hh_fn=LH.f32(LS+"hc_head_fn");const float* hh_sc=LH.f32(LS+"hc_head_scale");const float* hh_ba=LH.f32(LS+"hc_head_base");
        const float* hnorm=LH.bf16(LS+"norm.weight");
        // markov tables are BF16 too, and w2 is re-read once per block position in the draft's AR
        // loop (5x), so the f32 dequant cost 5 x 132 MB instead of 5 x 66 MB. Keep them native.
        const void* mw1=(const void*)WH.get(LS+"markov_head.markov_w1.weight").dev;
        const void* mw2=(const void*)WH.get(LS+"markov_head.markov_w2.weight").dev;
        const __nv_bfloat16* emb=(const __nv_bfloat16*)W.get("embed.weight").dev;
        { size_t fb,tb; cudaMemGetInfo(&fb,&tb); printf("[spec] head built. mem %.1f/%.1f GiB\n",(tb-fb)/1073741824.0,tb/1073741824.0); }

        // main_x accumulator + tapped re-prefill over [0..PS-1]
        // mh_pre is the only spec buffer not already sized by seqmax, and a multi-prompt sweep can
        // prefill a LONGER prompt than argv[2]: size it at the longest prompt, not this one.
        float *main_x,*mh_pre; CU(cudaMalloc(&main_x,(size_t)seqmax*d*4)); CU(cudaMalloc(&mh_pre,(size_t)(SMAX-1)*3*d*4));

        // buffers, sized once at the largest block in the sweep
        int *dbid,*dfid,*dout; CU(cudaMalloc(&dbid,BLKMAX*4)); CU(cudaMalloc(&dfid,4)); CU(cudaMalloc(&dout,(BLKMAX+1)*4));
        float* dmarg; CU(cudaMalloc(&dmarg,BLKMAX*4));       // per-proposal top1-top2 logit margin
        float *xemb,*xa,*xb; CU(cudaMalloc(&xemb,(size_t)BLKMAX*d*4)); CU(cudaMalloc(&xa,(size_t)BLKMAX*hc*d*4)); CU(cudaMalloc(&xb,(size_t)BLKMAX*hc*d*4));
        // VERIFY-side: sized VBMAX = BLKMAX+1. At VB=BLK+1 a BLKMAX-sized hv/collK/logK/mh_v
        // overruns by exactly one position -- which would not crash, it would corrupt the last
        // verified token. Same shape as Finding 62.
        float *hv,*hv2,*collK,*logK,*mh_v; CU(cudaMalloc(&hv,(size_t)VBMAX*hc*d*4)); CU(cudaMalloc(&hv2,(size_t)VBMAX*hc*d*4));
        CU(cudaMalloc(&collK,(size_t)VBMAX*d*4)); CU(cudaMalloc(&logK,(size_t)VBMAX*VOCAB*4)); CU(cudaMalloc(&mh_v,(size_t)VBMAX*3*d*4));
        std::vector<double> sweep_mstok(blkSweep.size()), sweep_acc(blkSweep.size());
        // The EFFECTIVE adaptK, which is not the requested one: a sweep entry of 0 falls through to
        // the 1.5 default unless NO_ADAPTK=1 is also in the environment, so a table printing the
        // requested value labels those rows 0.00 while they ran at 1.5. Cycle 2 lost a run to
        // exactly that. Print both until the semantics are fixed and re-gated.
        std::vector<float> sweep_akeff(blkSweep.size());
      // I3 INSTRUMENT (DSV4_MEMTRACE=1). The standing hypothesis for the fault that killed cycles 3
      // and 5 is that "something ACCUMULATES across sweep points and the longest prompt is merely
      // the first allocation big enough to fall off the end of it" — the suspect being the 8 raw
      // cudaMalloc/cudaFree per indexer layer per call inside a pool with only ~11 GiB headroom.
      // That is a claim about free memory over time, and nothing has ever printed free memory over
      // time. If this column is FLAT across points, the accumulation frame is dead and the next
      // suspect is arena_reset / the per-point head rebuild, not the allocator.
      const bool memtrace = getenv("DSV4_MEMTRACE")!=nullptr;
      for(size_t bsi=0; bsi<blkSweep.size(); ++bsi){
        if(memtrace){ size_t fb,tb; cudaMemGetInfo(&fb,&tb);
            printf("[memtrace] entering point %zu (prompt %d): free %.3f GiB of %.1f\n",
                   bsi, promptSweep[bsi], fb/1073741824.0, tb/1073741824.0); fflush(stdout); }
        const int BLK0 = blkSweep[bsi], NPASS = passSweep[bsi];
        // ADAPTIVE BLOCK SIZE (DSV4_ADAPTBLK=1). F94 measured that the right block is a property of
        // the WORKLOAD, not of the model: on realistic prompts BLK=6 beat 5 on 4 of 4, while the
        // canonical low-acceptance prompt regressed monotonically (24.38 -> 22.91 -> 20.87 for
        // 5 -> 6 -> 8) because its tau never moved off 3.61 and it paid a wider verify for width it
        // could not use. `adaptK` already adapts the width WITHIN a block from the draft's own
        // margins; the block itself was still a compile-time constant.
        //
        // The controller is deliberately dumb and hysteretic. The signal is the accepted-token
        // count, which is a COUNTED INTEGER and therefore immune to the ~1.5% cross-run timing floor
        // of trap 25 -- the same property that made S6's oracle resolvable. Grow when the drafter is
        // saturating the block it has, shrink when it is not using it, and never faster than one
        // step per window so a single lucky verify cannot move it.
        //
        // BLK=8 is NOT reachable by this controller and that is deliberate: F94 measured tau FALLING
        // 6 -> 8 on 3 of 4 prompts, because the block conditions on DSPARK_NOISE_TID at positions
        // 1..BLK-1 and a wider noise field degrades proposals the head was getting right. The head
        // was trained at 5 and generalises exactly one position. The cap is that measurement.
        static const bool adaptblk = getenv("DSV4_ADAPTBLK") != nullptr;
        const int ABLK_MIN = 3, ABLK_MAX = 6, ABLK_WIN = 8;
        int BLK = adaptblk ? BLK0 : BLK0;          // BLK is now MUTABLE when adaptblk is on
        std::deque<int> acc_hist;                  // recent accepted-token counts
        // adaptK = 0 means fixed width, i.e. exactly the previous behaviour. The default comes from
        // a within-run A/B (Finding 49): 3694 -> 3616 ms for the SAME 61 tokens, +2.1%. The
        // threshold is fitted on 18 verifies of one prompt, so it is deliberately on the permissive
        // side — a too-low threshold degrades to fixed width, a too-high one costs accepted tokens.
        // D1 FIXED. A sweep entry of `:0` USED to fall through to the 1.5 default, so fixed width was
        // not expressible per point and NO run since Finding 49 has contained a fixed-width control —
        // the +2.1% that justified adaptive verify width has never been re-checked on any prompt but
        // the one it was fitted on. A NEGATIVE adaptK now means "fixed width", e.g. `5:1:-1:2` runs
        // prompt 2 at a fixed verify width. `:0` keeps its old meaning (fall through to the default)
        // so every recorded sweep string still means what it meant when it was run.
        const float adaptK = adaptSweep[bsi] < 0.f ? 0.f
                           : adaptSweep[bsi] > 0.f ? adaptSweep[bsi]
                           : (getenv("NO_ADAPTK") ? 0.f : 1.5f);
        // STAYS 1.5. Shipping 2.0 was attempted and REVERTED (F130). The powered sweeps that
        // justified it were run on the s1 and s2 heads (+0.278 t=3.2, +0.320 t=4.75), and s3's own
        // re-fit was on the hold-out, which F129 shows is the wrong instrument. Measured on the
        // frozen suite with the SHIPPED s3 head, 2.0 is a REGRESSION: 24.73 -> 23.84, tok/verify
        // 3.736 -> 3.562. That is F111's coupling, alive after all -- a better drafter earns wider
        // verifies, so its optimal threshold is LOWER, and a threshold fitted on a weaker head does
        // not transfer to a stronger one.
        // This point's prompt. Named `pids`/`ps`/`PSp`, NOT `ids`/`s`/`PS`: the previous version
        // shadowed the outer names, and `run_layer` — a [&] lambda defined at outer scope — went on
        // binding the OUTER `PS`. See Finding 52. Every device buffer above is sized at the
        // longest prompt (SMAX), so any of these lengths is safe to use here.
        const std::vector<int>& pids = prompts[promptSweep[bsi]];
        const int ps = (int)pids.size(), PSp = ps-1;
        // Re-prefill so each point starts from the same state: the spec loop mutates the
        // window/compressed caches and main_x, and a sweep point that inherited them would be
        // measuring a different sequence, not a different setting. d_ids must be reloaded BEFORE
        // the prefill, not after it — the previous point may have left a different prompt there.
        CU(cudaMemcpy(d_ids,pids.data(),(size_t)ps*4,cudaMemcpyHostToDevice));
        for(int L=0;L<N_LAYERS;++L) KV[L].T=0;
        g_scratch_alloc_seq = 0;   // so scratch0 names layer 0's first prefill allocation
        // N1 DIAGNOSTIC (DSV4_ZERO_CACHES=1). Finding 60. A sweep point resets `T` and re-prefills,
        // but the KV caches, `xin`, `main_x` and `mh_pre` are persistent cudaMalloc buffers that are
        // never cleared — so every point starts holding the PREVIOUS point's tail, and a read past
        // the freshly written range sees different data at every point while being perfectly
        // deterministic within one. That is exactly the signature: identical inputs, different
        // first-verify margins, and no atomics anywhere in the path.
        // DSV4_ARENA_ZERO exonerated the arena; DSV4_ZERO_SCRATCH (prefill's raw cudaMalloc) moved
        // 8/8 distinct margin vectors to 5/8, so it is part of it. This covers the rest.
        if(getenv("DSV4_ZERO_CACHES")){
            for(int L=0;L<N_LAYERS;++L){ int r=compress_ratio(L);
                if(KV[L].win_kv) CU(cudaMemset(KV[L].win_kv,0,kv_rows_bytes(seqmax)));
                if(KV[L].xin)    CU(cudaMemset(KV[L].xin,0,(size_t)seqmax*DIM*4));
                if(r && KV[L].comp_kv) CU(cudaMemset(KV[L].comp_kv,0,kv_rows_bytes(seqmax/r+2)));
                if(r==4 && KV[L].idx_ckv) CU(cudaMemset(KV[L].idx_ckv,0,(size_t)(seqmax/r+2)*INDEX_HEAD_DIM*4)); }
            CU(cudaMemset(main_x,0,(size_t)seqmax*d*4));
            CU(cudaMemset(mh_pre,0,(size_t)(SMAX-1)*3*d*4));
            CU(cudaDeviceSynchronize());
        }
        k_embed<<<((size_t)PSp*d+255)/256,256>>>(h0,emb,d_ids,PSp,d); k_hc_expand<<<((size_t)PSp*hc*d+255)/256,256>>>(h,h0,PSp,hc,d); CU(cudaDeviceSynchronize());
        if(getenv("DSV4_HASH")){
            std::vector<float> hv0((size_t)PSp*hc*d), hv0b((size_t)PSp*d);
            CU(cudaMemcpy(hv0.data(),h,hv0.size()*4,cudaMemcpyDeviceToHost));
            CU(cudaMemcpy(hv0b.data(),h0,hv0b.size()*4,cudaMemcpyDeviceToHost));
            // LADDER 1.9: word-wise FNV-1a, not byte-wise. Same construction, 4x fewer rounds.
            // At real context this runs over PSp*hc*d = 25 M floats per layer x 43 layers, where the
            // byte loop is ~17e9 rounds and turns a 7-minute diagnostic into an hour. The values do
            // not compare against pre-1.9 logs; nothing does, they are only ever compared to another
            // hash from the same build.
            auto fnv=[](const std::vector<float>& v){ unsigned long long x=1469598103934665603ULL;
                for(float f: v){ unsigned u; memcpy(&u,&f,4); x^=(unsigned long long)u; x*=1099511628211ULL; } return x; };
            // WEIGHT INTEGRITY. Everything the prefill reads has now been shown bit-identical at
            // every point, yet layer 0's OUTPUT differs — and the effect needs the previous point to
            // have decoded far enough (NGEN0=60 reproduces, 20 does not). The remaining explanation
            // is that the decode writes out of bounds and corrupts persistent state that the next
            // prefill reads. Weights are the largest such state. Hash a slice of layer 0's own
            // weights, which is what layer 0 reads first.
            std::vector<uint8_t> wq((size_t)1<<20);
            CU(cudaMemcpy(wq.data(), (const void*)BW[0].attn.wq_a, wq.size(), cudaMemcpyDeviceToHost));
            unsigned long long wh=1469598103934665603ULL;
            for(uint8_t b: wq){ wh^=b; wh*=1099511628211ULL; }
            printf("[ihash] point %zu : h0(embed)=%016llx  h(hc_expand)=%016llx  hptr=%p  wq_a[0..1MB]=%016llx\n",
                   bsi, fnv(hv0b), fnv(hv0), (void*)h, wh); fflush(stdout);
        }
        // N1 INPUT HASH. Layer 0 of the prefill already differs between byte-identical points, so the
        // question is whether its INPUT differs (k_embed / k_hc_expand / d_ids) or whether the layer
        // itself is nondeterministic. Hash h immediately after k_hc_expand, before any layer runs.
        // N1 LAYER BISECTION (DSV4_HASH=2). The prefill's outputs cycle with period 5 across
        // byte-identical sweep points while being reproducible run to run, and the scratch address
        // is constant, so it is not the allocator. Hash the hidden state after EVERY layer: the
        // first layer whose hash differs between two identical points names the kernel.
        const int hashlvl = getenv("DSV4_HASH") ? atoi(getenv("DSV4_HASH")) : 0;
        // CHUNKED, and that is not a style choice. At PSp=3071 one layer's hidden state is
        // PSp*hc*d floats and a fresh std::vector per layer, 43 layers deep, drove MemAvailable to
        // 1192 MB and the memguard killed the run mid-prefill (stephash_H2.memguard.log). A 16 MB
        // reused staging buffer makes the diagnostic's host footprint independent of context, which
        // is the only way it can be pointed at the contexts where the defect lives.
        static std::vector<float> hstage;
        auto hlayer=[&](const float* dptr, size_t n)->unsigned long long {
            const size_t CH = 4u<<20;                       // 4 M floats = 16 MB
            if(hstage.size()<CH) hstage.resize(CH);
            unsigned long long v=1469598103934665603ULL;
            for(size_t off=0; off<n; off+=CH){
                const size_t m = (n-off<CH)?(n-off):CH;
                CU(cudaMemcpy(hstage.data(),dptr+off,m*4,cudaMemcpyDeviceToHost));
                const unsigned* uw=(const unsigned*)hstage.data();
                for(size_t i=0;i<m;++i){ v^=(unsigned long long)uw[i]; v*=1099511628211ULL; } }
            return v; };
        CU(cudaDeviceSynchronize());
        // B9. Prefill has never been profiled: F75 measured it end-to-end (48 tok/s, only 3.5x the
        // M=1 decode rate for ~1000x the work) and stopped there. The dprof marks already live
        // inside run_layer, so a reset/report pair around this loop attributes the whole prefill at
        // no cost -- and, more importantly, keeps prefill's marks OUT of the verify table. That
        // mixing is the reason nobody could read them before: a K=1 report over a run that also
        // prefilled 1023 positions is two regimes summed into one column.
        if(g_dprof_on) dprof_reset();
        // The MoE union/tile counters are cumulative and were never reset, so the first byte-count
        // run swept up the PS=5 prefill, the first-token gate and all 344 warm-decode calls at bs=1
        // alongside the 43 prefill calls. Those run one row per expert -- redundancy exactly 1.0 --
        // so they DILUTE the average toward 1 and the printed 2.27x was a lower bound, not the
        // measurement. Reset here so the [moebytes] line describes the prefill and nothing else.
        g_moe_tiles_sum = 0; g_moe_chunks_sum = 0; g_moe_union_sum = 0; g_moe_rows_sum = 0; g_moe_union_calls = 0;
        auto pfs_t0 = std::chrono::steady_clock::now();
        for(int Lyr=0; Lyr<N_LAYERS; ++Lyr){ arena_reset(); run_layer(Lyr,true,0,h,h2,d_ids,PSp); std::swap(h,h2);
            if(hashlvl>=2){ CU(cudaDeviceSynchronize());
                printf("[lhash] point %zu layer %2d ratio %3d : %016llx\n",
                       bsi, Lyr, compress_ratio(Lyr), hlayer(h,(size_t)PSp*hc*d)); fflush(stdout); }
            if(Lyr==40) dspark_tap_pool(mh_pre,h,PSp,hc,d,0,3); else if(Lyr==41) dspark_tap_pool(mh_pre,h,PSp,hc,d,1,3); else if(Lyr==42) dspark_tap_pool(mh_pre,h,PSp,hc,d,2,3); }
        CU(cudaDeviceSynchronize());
        { double ms = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-pfs_t0).count();
          printf("[decode] PREFILL: %d positions in %.1f ms = %.1f tok/s (%.3f ms/tok)\n",
                 PSp, ms, PSp*1000.0/ms, ms/PSp); fflush(stdout); }
        // Reported AFTER the wall-clock print above so dprof_report's own sync cannot land inside
        // the interval it is describing.
        if(g_dprof_on){ char tg[48]; snprintf(tg,sizeof tg,"PREFILL PS=%d",PSp); dprof_report(tg); }
        // B9 byte count. The prefill MoE achieved 10.1 GB/s against the IDEAL 92 GB (each expert read
        // once). That is 4.3% of roofline and reads as latency-bound — but the ideal is a LOWER bound,
        // and the ceiling (233 GB/s x 9.09 s = 2.1 TB) is 23x higher, so "latency-bound" and "moving
        // 23x too many bytes" were both consistent with the same measurement. Tiles settle it.
        if(getenv("DSV4_MOEUNION") && g_moe_union_calls){
            const double BPW = 0.5 + 1.0/32.0;                       // MXFP4: 4 bits + one e8m0 per 32
            double per_tile = 3.0*(double)MOE_INTER*(double)d*BPW;   // w1 + w3 + w2 for one expert
            double tiles = (double)g_moe_tiles_sum, ideal = (double)g_moe_union_sum;
            printf("[moebytes] PS=%d  calls %d  union/call %.1f  rows/call %.1f  tiles/call %.1f\n",
                   PSp, g_moe_union_calls, ideal/g_moe_union_calls,
                   (double)g_moe_rows_sum/g_moe_union_calls, tiles/g_moe_union_calls);
            double chunks = (double)g_moe_chunks_sum;
            printf("[moebytes] tiles %.0f  RB-chunks %.0f (= actual weight reads; a tile costs "
                   "ceil(me/RB) reads, NOT one)\n", tiles, chunks);
            printf("[moebytes] expert weight traffic: ACTUAL %.1f GB vs IDEAL %.1f GB"
                   "  -> redundancy %.2fx   (tiles-only would say %.2fx and be wrong)\n",
                   chunks*per_tile/1e9, ideal*per_tile/1e9, chunks/ideal, tiles/ideal);
            fflush(stdout);
        }
        // GATE (in-run, every point). A compressed layer emits exactly floor(PSp/ratio) rows during
        // prefill, so KV[L].T is a direct readout of the length the prefill ACTUALLY ran at. This is
        // the assertion whose absence cost cycle 2 its whole run: with the Finding-52 bug and
        // PSp=17 against an argv PS of 5, a ratio-4 layer reports T=1 where it must report 4.
        for(int L=0;L<N_LAYERS;++L){ int r=compress_ratio(L); if(!r) continue;
            if(KV[L].T != PSp/r){
                fprintf(stderr,"[spec] GATE FAIL: point %zu prompt %d PSp=%d ratio=%d -> KV[%d].T=%d, expected %d "
                               "(the prefill ran at the wrong length)\n", bsi, promptSweep[bsi], PSp, r, L, KV[L].T, PSp/r);
                return 3; } }
        // ---------------- S5 CAPTURE (DSV4_CAPTURE=<dir>) ----------------
        // Writes exactly what training needs and nothing else: the THREE raw layer taps per
        // position (`mh_pre`, layout mh[t*(3*d) + slot*d + j] per k_tap_pool) plus the token ids.
        //
        // WHY mh_pre AND NOT main_x, which is 3x smaller: `main_proj` is the "feature projection",
        // which NVIDIA's reference DSpark trainer lists as TRAINABLE. Capturing after it would bake
        // in weights the fine-tune is about to change. So capture upstream of the first trainable
        // parameter, which is the tap pool.
        //
        // The target distribution for the TV loss is NOT stored -- 129280 floats/token is 258 KB/tok
        // and impossible. It is recomputed during training from these taps through the frozen
        // lm_head (1.06 GB bf16, cheap to keep resident). The taps are the minimal sufficient
        // statistic.
        //
        // One file per sweep point, so DSV4_PROMPTS + DSV4_BLKSWEEP capture N samples per checkpoint
        // load -- the load is ~10 minutes and must be amortised, not paid per sample.
        if(const char* capdir = getenv("DSV4_CAPTURE")){
            // FORMAT v2 adds the LM-HEAD INPUT per position, and it is not optional.
            //
            // F99 recorded the plan as "reconstruct the target distribution from the taps through
            // the frozen lm_head". That is WRONG and the code says why: k_tap_pool writes
            // `h.mean(dim=hc)` -- an UNWEIGHTED mean -- while the real path is
            // `hc_head(h, hc_fn, hc_scale, hc_base)`, a LEARNED Sinkhorn-normalised combination
            // followed by rmsnorm. The tap is not the lm_head input and no amount of arithmetic
            // turns one into the other.
            //
            // Without it the TV term -- 90% of the loss weight -- has no target and the trainer
            // silently optimises CE alone (measured: tv=0.0000). So capture the real thing: run
            // the same hc_head+rmsnorm the head_fwd path runs, over ALL prefill positions. The
            // expensive part (the lm_head GEMM over 129280 columns) stays in training, where it is
            // one matmul against a frozen weight; here we only store its 4096-wide input.
            // Cost: +8 KB/token on top of 24, and storage was never the constraint (F90).
            float* lmin = nullptr;
            { CU(cudaMalloc(&lmin,(size_t)PSp*d*4));
              hc_head(lmin, h, hc_fn, hc_sc, hc_bs, PSp, hc, d, HC_EPS);
              rmsnorm(lmin, lmin, norm_w, PSp, d, EPS, true, 0);
              CU(cudaDeviceSynchronize()); }
            const size_t ntap=3, nvals=(size_t)PSp*ntap*d;
            std::vector<float> hf(nvals);
            CU(cudaMemcpy(hf.data(), mh_pre, nvals*4, cudaMemcpyDeviceToHost));
            std::vector<uint16_t> hb(nvals);
            for(size_t i2=0;i2<nvals;++i2){ uint32_t u; memcpy(&u,&hf[i2],4); hb[i2]=(uint16_t)(u>>16); } // bf16 = fp32 top half (truncate)
            char path[1024]; snprintf(path,sizeof path,"%s/shard_p%02d_s%06d.dspc", capdir, promptSweep[bsi], PSp);
            FILE* f=fopen(path,"wb");
            if(!f){ fprintf(stderr,"[capture] FAIL: cannot open %s\n", path); return 4; }
            const std::vector<int>& pids = prompts[promptSweep[bsi]];
            // v2: header word 7 (was reserved) is now 1 = an lm_head-input block follows the taps.
            uint32_t hdr[8] = {0x43505344u, 2u, (uint32_t)PSp, (uint32_t)ntap, (uint32_t)d, 1u,
                               (uint32_t)pids.size(), 1u};
            std::vector<float> hf2((size_t)PSp*d);
            CU(cudaMemcpy(hf2.data(), lmin, hf2.size()*4, cudaMemcpyDeviceToHost));
            std::vector<uint16_t> hb2(hf2.size());
            for(size_t i2=0;i2<hf2.size();++i2){ uint32_t u; memcpy(&u,&hf2[i2],4); hb2[i2]=(uint16_t)(u>>16); }
            size_t w = 0;
            w += fwrite(hdr,4,8,f);
            w += fwrite(pids.data(),4,pids.size(),f);
            w += fwrite(hb.data(),2,nvals,f);
            w += fwrite(hb2.data(),2,hb2.size(),f);
            long bytes = ftell(f);
            fclose(f);
            if(w != 8+pids.size()+nvals+(size_t)PSp*d){ fprintf(stderr,"[capture] FAIL: short write to %s\n", path); return 4; }
            // Manifest is append-only and one line per sample, so a killed run resumes from the last
            // COMPLETE line rather than re-deriving progress from directory listings.
            char mpath[1024]; snprintf(mpath,sizeof mpath,"%s/manifest.jsonl", capdir);
            if(FILE* mf=fopen(mpath,"a")){
                fprintf(mf,"{\"file\":\"%s\",\"prompt\":%d,\"n_tok\":%d,\"n_taps\":%zu,\"d\":%d,"
                           "\"dtype\":\"bf16\",\"n_ids\":%zu,\"bytes\":%ld}\n",
                        strrchr(path,'/')?strrchr(path,'/')+1:path, promptSweep[bsi], PSp, ntap, d,
                        pids.size(), bytes);
                fclose(mf);
            }
            cudaFree(lmin);
            printf("[capture] wrote %s  %d tok x (%zu taps + 1 lm_in) x %d = %.1f MB (%.1f KB/tok)\n",
                   path, PSp, ntap, d, bytes/1048576.0, bytes/1024.0/PSp); fflush(stdout);
        }
        dspark_main_x(main_x, mh_pre, main_proj, main_proj_s, main_norm, PSp, d, EPS); CU(cudaDeviceSynchronize());
        // LADDER 1.0. Rows [0, mkv_valid) of mkv[st] are built from the CURRENT main_x. main_x was
        // just rewritten wholesale for this sweep point, so nothing carries over from the last one.
        // Below the committed position main_x is write-once (the accept path at the bottom of the
        // spec loop only ever appends acc+1 rows), so the cache only ever grows from here.
        int mkv_valid = 0;
        // ---------------- S5 DRAFT PROBE (DSV4_CAPTURE) ----------------
        // THE EQUIVALENCE ORACLE. F101: the PyTorch port agreed with the engine on 1.5% of draft
        // positions where the engine itself gets ~30%, so training on it would have aligned the
        // head to a function the server does not run. The gate that catches this needs the ENGINE's
        // own proposals at the SAME positions the port is fed.
        //
        // It has to be a probe, not a log of the decode loop: the capture covers prefill positions
        // 0..PSp-1, while spec decode visits positions AFTER PSp. Logging the decode would produce
        // drafts at positions we hold no taps for -- the two would never line up.
        //
        // So: reuse the already-built KV (filled to PSp by the prefill above) and run the SAME
        // draft chain the spec loop runs, with ctx = t+1 and cur = ids[t], for a spread of probe
        // positions inside the captured range. This is the engine's answer to exactly the question
        // the port is asked.
        if(getenv("DSV4_CAPTURE")){
            const int NPROBE = getenv("DSV4_NPROBE") ? atoi(getenv("DSV4_NPROBE")) : 16;
            char ppath[1024]; snprintf(ppath,sizeof ppath,"%s/drafts_p%02d_s%06d.txt",
                                       getenv("DSV4_CAPTURE"), promptSweep[bsi], PSp);
            FILE* pf2 = fopen(ppath,"w");
            std::vector<int> pbid(BLK,DSPARK_NOISE_TID), poo(BLK+1); std::vector<float> pmarg(BLK,0.f);
            const std::vector<int>& pids2 = prompts[promptSweep[bsi]];
            int nprobed = 0;
            for(int pi=0; pi<NPROBE; ++pi){
                int t = 8 + (int)((long)pi*(PSp-10)/ (NPROBE>1?NPROBE-1:1));   // spread, skip the warm-up
                if(t<1 || t>=PSp) continue;
                int pctx = t+1, panchor = t, pcur = pids2[t];   // spec loop: anchor=cpos-1, ctx=cpos
                arena_reset();
                for(int st=0;st<NSTAGE;++st) dspark_main_kv(mkv[st], main_x, mb[st].attn, pctx, EPS);
                pbid.assign(BLK,DSPARK_NOISE_TID); pbid[0]=pcur;
                CU(cudaMemcpy(dbid,pbid.data(),BLK*4,cudaMemcpyHostToDevice));
                k_embed<<<((size_t)BLK*d+255)/256,256>>>(xemb,emb,dbid,BLK,d);
                k_hc_expand<<<((size_t)BLK*hc*d+255)/256,256>>>(xa,xemb,BLK,hc,d); CU(cudaDeviceSynchronize());
                // INTERMEDIATE DUMP (DSV4_PROBE_DUMP=<file>), first probe position only.
                // F102: the port scores 0/288 and four guesses at the cause were all wrong. Dumping
                // the tensors the port must reproduce turns "which of four candidates" into "the
                // first one that diverges", in one run instead of four.
                FILE* dmp = (pi==0 && getenv("DSV4_PROBE_DUMP")) ? fopen(getenv("DSV4_PROBE_DUMP"),"wb") : nullptr;
                auto dump = [&](const char* tag, const float* dev, size_t n){
                    if(!dmp) return; std::vector<float> hv(n);
                    CU_V(cudaMemcpy(hv.data(),dev,n*4,cudaMemcpyDeviceToHost));
                    uint32_t L=(uint32_t)strlen(tag), N=(uint32_t)n;
                    fwrite(&L,4,1,dmp); fwrite(tag,1,L,dmp); fwrite(&N,4,1,dmp); fwrite(hv.data(),4,n,dmp); };
                if(dmp){ uint32_t meta[3]={(uint32_t)t,(uint32_t)pcur,(uint32_t)BLK};
                         fwrite(meta,4,3,dmp);
                         dump("main_x_t", main_x+(size_t)t*d, d); }
                float *pc=xa,*pn=xb;
                dump("after_embed", pc, (size_t)BLK*hc*d);
                { extern FILE* g_dspark_dump; g_dspark_dump = dmp; }   // stage dumps for block 0 only
                for(int st=0;st<NSTAGE;++st){ dspark_block_forward(pn,pc,dbid,mkv[st],panchor,mb[st],
                        blk_cos+(size_t)pctx*hf, blk_sin+(size_t)pctx*hf, BLK,WINDOW,HC_SINKHORN_ITERS,EPS);
                    std::swap(pc,pn);
                    { extern FILE* g_dspark_dump; g_dspark_dump = nullptr; }  // only the FIRST block
                    char tg[24]; snprintf(tg,sizeof tg,"after_block%d",st); dump(tg, pc, (size_t)BLK*hc*d); }
                CU(cudaMemcpy(dfid,&pcur,4,cudaMemcpyHostToDevice));
                dspark_forward_head(dout,pc,dfid,hh_fn,hh_sc,hh_ba,hnorm,head_bf,mw1,mw2,1,BLK,hc,d,
                                    VOCAB,DSPARK_MARKOV_RANK,EPS,dmarg); CU(cudaDeviceSynchronize());
                CU(cudaMemcpy(poo.data(),dout,(BLK+1)*4,cudaMemcpyDeviceToHost));
                CU(cudaMemcpy(pmarg.data(),dmarg,BLK*4,cudaMemcpyDeviceToHost));
                if(dmp){ fclose(dmp); printf("[probe] intermediates dumped at t=%d -> %s\n",
                                             t, getenv("DSV4_PROBE_DUMP")); fflush(stdout); }
                if(pf2){ fprintf(pf2,"%d %d", t, pcur);
                         for(int k=0;k<BLK;++k) fprintf(pf2," %d", poo[k+1]);
                         for(int k=0;k<BLK;++k) fprintf(pf2," %.4f", pmarg[k]);
                         fprintf(pf2,"\n"); }
                ++nprobed;
            }
            if(pf2) fclose(pf2);
            printf("[probe] wrote %d draft probes (block=%d) to %s\n", nprobed, BLK, ppath); fflush(stdout);
        }
        // N1 BISECTION (DSV4_HASH=1). Finding 60 says the FIRST verify of every sweep point produces
        // a different margin vector on byte-identical input. Unit gates have since cleared the two
        // obvious suspects — tests/gate_scratch_init shows the prefill attention chain is
        // poison-independent and reproducible at every length 1..29, and gate_units shows
        // moe_forward bit-identical over 32 repeats of the decode configuration. So the divergence is
        // either in the in-situ prefill (which those gates do not run: 43 real layers, real weights,
        // hc, MoE and attention composed) or downstream in the draft.
        //
        // Hashing the prefill's own outputs separates those two cases in ONE run, which is the whole
        // point: `main_x` and `mh_pre` are everything the draft consumes from the prefill. Identical
        // hashes at identical points => the prefill is fine and the draft is the source; differing
        // hashes => the composition is nondeterministic even though its parts gate clean.
        if(getenv("DSV4_HASH")){
            auto h64=[&](const float* dptr, size_t n)->unsigned long long {
                const size_t CH = 4u<<20; if(hstage.size()<CH) hstage.resize(CH);
                unsigned long long x=1469598103934665603ULL;                 // FNV-1a over the raw bits, word-wise
                for(size_t off=0; off<n; off+=CH){
                    const size_t m=(n-off<CH)?(n-off):CH;
                    CU(cudaMemcpy(hstage.data(),dptr+off,m*4,cudaMemcpyDeviceToHost));
                    const unsigned* uw=(const unsigned*)hstage.data();
                    for(size_t i=0;i<m;++i){ x^=(unsigned long long)uw[i]; x*=1099511628211ULL; } }
                return x; };
            printf("[hash] point %zu prompt %d PSp=%d : main_x=%016llx mh_pre=%016llx scratch0=%016llx\n",
                   bsi, promptSweep[bsi], PSp,
                   h64(main_x,(size_t)PSp*d), h64(mh_pre,(size_t)PSp*3*d), g_scratch_first_addr);
            fflush(stdout);
        }
        int NGEN=NGEN0;
        int cur=pids[ps-1], cpos=PSp;              // cur = token at position cpos (=PSp=ps-1), not yet in cache
        std::vector<int> sgen; int nverify=0, timed_tok=0; float spec_ms=0; cudaEvent_t s0,s1; cudaEventCreate(&s0); cudaEventCreate(&s1);
        // Per-phase attribution of the spec step (LOOP_LOG Finding 17: the draft is ~6x off its
        // roofline and it is what keeps speculation at parity). DSV4_SPECPROF=1.
        const bool specprof = getenv("DSV4_SPECPROF")!=nullptr;
        static bool draft_hwm_shown = false;      // lever B10 capacity print, once per process
        cudaEvent_t p0,p1,p2,p3,p4; for(auto e:{&p0,&p1,&p2,&p3,&p4}) cudaEventCreate(e);
        double acc_kv=0, acc_blk=0, acc_head=0, acc_ver=0; int nprof=0;
        // Finding 82: the draft path's RAW allocator cost, split draft-half vs rest-of-round. See
        // the instrument in dscratch.h — the verify path is on the arena, the DSpark draft path
        // never was, and nobody has ever measured what that costs at the current baseline.
        double acc_dral=0, acc_drsy=0, acc_rral=0, acc_rrsy=0; long long acc_dn=0, acc_dsn=0, acc_rn=0, acc_rsn=0;
        double raw0=0, rsy0=0, rawd=0, rsyd=0; long long rawn0=0, rsyn0=0, rawnd=0, rsynd=0;
        // ---- S6 COUNTERFACTUAL SUFFIX-DRAFT PROBE (DSV4_SUFFIXPROBE=1, default OFF) ----
        // LEVERS.md S6 proposes a suffix-automaton / prompt-lookup drafter ahead of the MTP, and its
        // own falsification says the win is NOT the skipped draft head (13 ms of a 151 ms cycle) but
        // any ACCEPTANCE gained at the same K. Acceptance is a counted integer, so it is immune to
        // the ~1.5 % cross-run timing floor Finding 79 measured — which is the only reason S6 is
        // resolvable on this box at all. This probe prices it WITHOUT building the cascade: at every
        // verify it computes what a suffix drafter would have proposed from the committed sequence
        // and how much of that the target would have accepted, alongside what the MTP actually got.
        //
        // It is READ-ONLY. It touches no device buffer and no engine state, so the emitted sequence,
        // the GATE and the LOSSLESS GATE are bit-identical to a run without it. Finding 67 killed a
        // lever this way before it was built; this is the same move.
        //
        // EXACTNESS, stated precisely, because the counterfactual is only partly observable.
        // tam[i] is the target's argmax at position cpos+1+i GIVEN the verify input
        // vtok[0..i] = [cur, draft[0..i-1]] — i.e. given the MTP's draft. It is therefore a valid
        // ground truth for a different draft only while that draft agrees with the MTP's, which
        // holds for i <= acc (both equal tam there). So a counterfactual acceptance <= acc is EXACT,
        // and a value of acc+1 is EXACT as "at least acc+1"; anything beyond that is an
        // extrapolation. Both are reported: `lb` is the sound number, `raw` the optimistic one.
        const bool sfxprobe = getenv("DSV4_SUFFIXPROBE")!=nullptr;
        const int SFX_MAXNG = getenv("DSV4_SUFFIX_MAXNG") ? atoi(getenv("DSV4_SUFFIX_MAXNG")) : 32;
        static const int SFX_NTHR = 5; const int sfx_thr[SFX_NTHR] = {1,2,3,4,6};
        long sfx_mtp=0, sfx_lb=0, sfx_raw=0, sfx_orc=0, sfx_casc[SFX_NTHR]={0,0,0,0,0};
        int sfx_n=0, sfx_hit=0, sfx_win=0, sfx_lose=0;
        printf("[spec] decoding %d tokens (block=%d, draft passes=%d, adaptK=%.2f, prompt=%d s=%d)...\n",
               NGEN, BLK, NPASS, adaptK, promptSweep[bsi], ps);
        // ---- LADDER 1.9: PER-STEP STATE HASH (DSV4_STEPHASH=<path>, default OFF) ----
        // The engine stops reproducing itself part-way through a long run: point 0 of the 1.0 sweep
        // is identical across three runs, points 1 and 2 diverge at generated token 25 and 43. The
        // divergence is autoregressive, so ONE flipped token hides everything after it and a
        // token-id diff can only name the step, never the quantity. This names the quantity.
        //
        // It writes one line per verify with the whole causal chain of that step, in dataflow
        // order, so a plain `diff` of two runs points at the FIRST link that differs:
        //   mkv/mainx -> draft input -> draft ids+margins -> verify logits -> acc/correction
        // If `mkv` and `mainx` match and `dh` (draft out) differs, the draft chain is the source.
        // If everything up to `dh` matches and `lg` differs, the target verify is. If `lg` matches
        // and `acc` differs, it is the host-side accept logic (i.e. not the GPU at all).
        //
        // READ-ONLY: device-to-host copies and host arithmetic only, no device write, no engine
        // state touched. Off by default so the shipped path is byte-identical.
        // Level 1 = the cheap half (draft out, logits, accept) -- everything already on the host.
        // Level 2 (default when set) adds the two big persistent inputs, `mkv[0]` and `main_x`,
        // which is what makes "the input was identical" a claim rather than an assumption.
        const char* stephash_path = getenv("DSV4_STEPHASH");
        const int stephash_lvl = stephash_path ? (getenv("DSV4_STEPHASH_LVL") ? atoi(getenv("DSV4_STEPHASH_LVL")) : 2) : 0;
        FILE* shf = nullptr;
        if(stephash_path){ shf = fopen(stephash_path, bsi==0 ? "w" : "a");
            if(!shf) printf("[stephash] WARN: cannot open %s\n", stephash_path); else setvbuf(shf,nullptr,_IOLBF,0); }
        // FNV-1a over the raw bits, one 32-bit word at a time. Not byte-at-a-time: this runs over
        // main_x (up to 25 MB per step) and the byte loop would cost more than the forward.
        auto fnvw=[](const void* p2, size_t nwords, unsigned long long x=1469598103934665603ULL)->unsigned long long{
            const unsigned* u=(const unsigned*)p2;
            for(size_t i=0;i<nwords;++i){ x ^= (unsigned long long)u[i]; x *= 1099511628211ULL; }
            return x; };
        // Chunked for the same reason as `hlayer` above: at ctx 12k this is 100 MB per verify.
        static std::vector<float> shbuf;
        auto dhash=[&](const float* dptr, size_t n)->unsigned long long{
            if(!n) return 0ULL;
            const size_t CH = 4u<<20; if(shbuf.size()<CH) shbuf.resize(CH);
            unsigned long long x=1469598103934665603ULL;
            for(size_t off=0; off<n; off+=CH){
                const size_t m=(n-off<CH)?(n-off):CH;
                CU(cudaMemcpy(shbuf.data(), dptr+off, m*4, cudaMemcpyDeviceToHost));
                x = fnvw(shbuf.data(), m, x); }
            return x; };
        while((int)sgen.size()<NGEN && cpos+BLK+1<seqmax){
            cudaEventRecord(s0);
            int anchor=cpos-1, ctx=cpos;           // main context [0..cpos-1]
            if(specprof){ cudaEventRecord(p0); raw0=g_raw_ms; rsy0=g_rawsync_ms; rawn0=g_raw_n; rsyn0=g_rawsync_n; }
            // LADDER 1.0. Was a full O(ctx) rebuild of the head main-KV, three stages, every token
            // -- 55 % of the whole context term (0.4). Now only the rows committed since the last
            // step. Byte-identical; DSV4_MAINKV_CACHE=0 restores the from-scratch path for the A/B,
            // and this binary is the one that carries the LOSSLESS gate, so the two arms of that
            // env var are what proves the emitted sequence is unchanged.
            static const bool mainkv_cache = !(getenv("DSV4_MAINKV_CACHE") && atoi(getenv("DSV4_MAINKV_CACHE"))==0);
            if(mainkv_cache){
                for(int st=0;st<NSTAGE;++st){ int v=mkv_valid; dspark_main_kv_upto(mkv[st], main_x, mb[st].attn, ctx, &v, EPS); }
                mkv_valid = ctx;
            } else {
                for(int st=0;st<NSTAGE;++st) dspark_main_kv(mkv[st], main_x, mb[st].attn, ctx, EPS);
            }
            // LADDER 1.9. The draft's two persistent inputs, hashed BEFORE the draft reads them.
            unsigned long long sh_mkv=0, sh_mx=0, sh_din=0;
            if(stephash_lvl>=2){ CU(cudaDeviceSynchronize());
                sh_mkv = dhash(mkv[0], (size_t)ctx*(size_t)g_kv_rowf);
                sh_mx  = dhash(main_x, (size_t)ctx*d); }
            if(specprof) cudaEventRecord(p1);
            // DRAFT: block [cur, noise x (BLK-1)].
            //
            // REFINEMENT (NPASS>1). The MTP blocks see DSPARK_NOISE_TID at positions 1..BLK-1, so
            // they condition the whole block on a token that carries no information; only the
            // markov head at the output does any sequencing. Pass 2 re-runs the same three blocks
            // with pass 1's own proposals in those slots, so the blocks condition on something
            // plausible. It is free of correctness risk by construction — the draft is only a
            // proposal and the verify is unchanged — and costs one more block chain + head, so it
            // pays iff mean tokens/verify rises by more than that fraction of the cycle.
            std::vector<int> bid(BLK,DSPARK_NOISE_TID); bid[0]=cur;
            std::vector<int> oo(BLK+1), draft(BLK); std::vector<float> hmarg(BLK,0.f);
            float *cb=nullptr,*nb=nullptr;
            for(int pass=0; pass<NPASS; ++pass){
                arena_reset();     // each pass re-dmallocs the whole block chain; 3 passes overflow without this
                CU(cudaMemcpy(dbid,bid.data(),BLK*4,cudaMemcpyHostToDevice));
                k_embed<<<((size_t)BLK*d+255)/256,256>>>(xemb,emb,dbid,BLK,d); k_hc_expand<<<((size_t)BLK*hc*d+255)/256,256>>>(xa,xemb,BLK,hc,d); CU(cudaDeviceSynchronize());
                if(stephash_lvl && pass==0) sh_din = dhash(xa, (size_t)BLK*hc*d);
                cb=xa; nb=xb;
                for(int st=0;st<NSTAGE;++st){ dspark_block_forward(nb,cb,dbid,mkv[st],anchor,mb[st],blk_cos+(size_t)ctx*hf,blk_sin+(size_t)ctx*hf,BLK,WINDOW,HC_SINKHORN_ITERS,EPS); std::swap(cb,nb); }
                if(specprof && pass==NPASS-1) cudaEventRecord(p2);
                CU(cudaMemcpy(dfid,&cur,4,cudaMemcpyHostToDevice));
                dspark_forward_head(dout,cb,dfid,hh_fn,hh_sc,hh_ba,hnorm,head_bf,mw1,mw2,1,BLK,hc,d,VOCAB,DSPARK_MARKOV_RANK,EPS,dmarg); CU(cudaDeviceSynchronize());
                CU(cudaMemcpy(oo.data(),dout,(BLK+1)*4,cudaMemcpyDeviceToHost));
                CU(cudaMemcpy(hmarg.data(),dmarg,BLK*4,cudaMemcpyDeviceToHost));
                for(int i=0;i<BLK;++i) draft[i]=oo[1+i];   // proposals for cpos+1..cpos+BLK
                for(int i=1;i<BLK;++i) bid[i]=draft[i-1];  // feed them back for the next pass
            }
            // LADDER 1.9. The draft's whole output: proposed ids and the margins the width
            // controller reads. Both are already on the host, so this costs nothing.
            unsigned long long sh_draft=0;
            if(stephash_lvl){ sh_draft = fnvw(oo.data(), oo.size());
                              sh_draft = fnvw(hmarg.data(), hmarg.size(), sh_draft); }
            if(specprof){ cudaEventRecord(p3);
                rawd=g_raw_ms-raw0; rsyd=g_rawsync_ms-rsy0; rawnd=g_raw_n-rawn0; rsynd=g_rawsync_n-rsyn0; }
            // Lever B10 / Finding 83 capacity check, printed ONCE and in a clean run too. The draft
            // now dmallocs its whole 3-stage chain + head out of the arena with dfree a no-op and
            // only one arena_reset() per pass, so the draft's peak is additive on top of whatever
            // the prefill set. The arena ABORTS on overflow and that kills a 15-minute run, so the
            // margin is evidence, not an assumption. Printed after the FIRST draft of the point,
            // when the draft's own contribution is at its high-water mark for the round.
            if(!draft_hwm_shown){ draft_hwm_shown=true;
                // g_arena_off here is the draft chain's OWN footprint: the last arena_reset() was at
                // the top of this pass, so everything above 0 was allocated by the 3 block forwards
                // (attn + hc + MoE) and the head. g_arena_hwm is the global peak, which the prefill
                // usually owns.
                printf("[arena] first draft: draft footprint %.2f MB, global hwm %.2f MB / cap %.2f MB"
                       " (%.1f%%), draft path = %s\n",
                       g_arena_off/1048576.0, g_arena_hwm/1048576.0, g_arena_cap/1048576.0,
                       100.0*g_arena_hwm/g_arena_cap,
                       g_draft_raw ? "RAW cudaMalloc (DSV4_DRAFT_RAW=1)" : "ARENA"); fflush(stdout); }
            // ---- ADAPTIVE VERIFY WIDTH (LOOP_LOG Finding 49; EVICT, arXiv 2605.00342) ----
            // Verifying the k-th block position is only worth it if the probability it is accepted
            // exceeds its marginal cost in tokens. On a dense model that cost is ~0 and you always
            // take the widest block; on a sparse MoE it is dominated by the EXPANDING UNION OF
            // EXPERTS the extra position activates, and our own ksweep prices it:
            //
            //   cycle(K) = 106.9 / 137.9 / 161.3 / 176.6 / 198.5 ms for K=1..5
            //   break-even P(accept) = marginal_ms / ms_per_tok = 0.47 / 0.35 / 0.23 / 0.33
            //
            // The measured marginal acceptance rate is 0.33 — straddling those thresholds, which is
            // exactly the regime where a per-verify signal decides rather than a fixed width. The
            // signal is the draft head's own top1-top2 logit margin, already computed while it takes
            // the argmax. Truncating is LOSSLESS: verifying fewer proposals cannot change what the
            // target emits, only how many tokens one verify can commit.
            // VKPLUS: verify all BLK proposals (VB = BLK+1), not BLK-1 of them. DSV4_NOVKPLUS=1
            // restores the old cap for the A/B. The extension is gated on the LAST proposal's own
            // margin, hmarg[BLK-1], which is in bounds precisely because hmarg carries BLK entries.
            static const int novk = getenv("DSV4_NOVKPLUS") != nullptr;
            const int VKCAP = novk ? BLK : BLK + 1;
            int VK = VKCAP;
            if(adaptK > 0.f){
                VK = 2;                                        // always verify at least cur + one proposal
                // VK=k verifies [cur, draft[0..k-2]], so extending to k+1 ADDS draft[k-1]: the gate
                // is that proposal's own margin, hmarg[VK-1]. (hmarg[VK-2] gates a token already in
                // the block, which is the wrong one and always passes for a confident draft[0].)
                while(VK < VKCAP && hmarg[VK-1] >= adaptK) ++VK;
            }
            // Print every verify's margins: one adaptK=0 run is then a CALIBRATION SET (margin vs
            // whether that proposal was actually accepted), from which the threshold is fitted
            // offline instead of swept blind across 15-minute runs.
            printf("  [margins]"); for(int i=0;i<BLK;++i) printf(" %.2f",hmarg[i]); printf("  K=%d\n", VK);
            const int VB = VK;
            std::vector<int> vtok(VB); vtok[0]=cur; for(int i=1;i<VB;++i) vtok[i]=draft[i-1];
            std::vector<int> Tbefore(N_LAYERS); for(int L=0;L<N_LAYERS;++L) Tbefore[L]=KV[L].T;
            int* dvt; dvt=d_ids+cpos; CU(cudaMemcpy(dvt,vtok.data(),VB*4,cudaMemcpyHostToDevice));
            k_embed<<<((size_t)VB*d+255)/256,256>>>(h0,emb,dvt,VB,d); k_hc_expand<<<((size_t)VB*hc*d+255)/256,256>>>(hv,h0,VB,hc,d); CU(cudaDeviceSynchronize());
            float* vin=hv; float* vout=hv2;
            for(int Lyr=0; Lyr<N_LAYERS; ++Lyr){ arena_reset(); int ratio=compress_ratio(Lyr);
                if(ratio==0) block_verify_step (vout,vin,dvt,BW[Lyr],cpos,VB,HC_SINKHORN_ITERS,EPS,KV[Lyr]);
                else         cblock_verify_step (vout,vin,dvt,CW[Lyr],cpos,VB,HC_SINKHORN_ITERS,EPS,KV[Lyr]);
                std::swap(vin,vout);
                if(Lyr==40) dspark_tap_pool(mh_v,vin,VB,hc,d,0,3); else if(Lyr==41) dspark_tap_pool(mh_v,vin,VB,hc,d,1,3); else if(Lyr==42) dspark_tap_pool(mh_v,vin,VB,hc,d,2,3); }
            hc_head(collK,vin,hc_fn,hc_sc,hc_bs,VB,hc,d,HC_EPS); rmsnorm(collK,collK,norm_w,VB,d,EPS,true,0);
            gemm_bf16w(logK,collK,head_bf,VB,VOCAB,d,0); CU(cudaDeviceSynchronize());
            std::vector<float> lg((size_t)VB*VOCAB); CU(cudaMemcpy(lg.data(),logK,(size_t)VB*VOCAB*4,cudaMemcpyDeviceToHost));
            std::vector<int> tam(VB); for(int i=0;i<VB;++i){const float*r=&lg[(size_t)i*VOCAB];int aa=0;for(int v=1;v<VOCAB;++v)if(r[v]>r[aa])aa=v;tam[i]=aa;}
            // ACCEPT longest matching prefix: draft[i]==tam[i] (target's token for pos cpos+1+i)
            int acc=0; while(acc<VB-1 && draft[acc]==tam[acc]) ++acc;
            int correction=tam[acc];                        // target's token for pos cpos+acc+1
            // LADDER 1.9. One line per verify, in dataflow order. Two runs of the same binary
            // diff to the first differing FIELD, which names the link that broke, not just the step.
            if(shf){
                unsigned long long sh_lg = fnvw(lg.data(), lg.size());
                fprintf(shf, "p%zu v%d cpos=%d ctx=%d cur=%d VK=%d mkv=%016llx mx=%016llx din=%016llx "
                             "draft=%016llx lg=%016llx acc=%d corr=%d\n",
                        bsi, nverify, cpos, ctx, cur, VB, sh_mkv, sh_mx, sh_din, sh_draft, sh_lg, acc, correction);
            }
            if(sfxprobe){
                // S = the committed sequence, positions 0..cpos. pids covers 0..PSp and sgen covers
                // PSp+1..cpos, so S.back() == cur by construction. Built BEFORE sgen is extended.
                std::vector<int> S(pids.begin(), pids.end());
                S.insert(S.end(), sgen.begin(), sgen.end());
                const int n=(int)S.size();                  // == cpos+1
                std::vector<int> sdraft(BLK,-1);            // -1 never matches a real token id
                const int mlen = suffix_draft(S.data(), n, BLK, SFX_MAXNG, sdraft.data());
                int accs=0; while(accs<VB-1 && sdraft[accs]==tam[accs]) ++accs;
                const int accs_lb = accs < acc+1 ? accs : acc+1;   // sound; see the note above
                const int orc = acc > accs_lb ? acc : accs_lb;
                sfx_mtp += acc+1; sfx_lb += accs_lb+1; sfx_raw += accs+1; sfx_orc += orc+1;
                for(int t=0;t<SFX_NTHR;++t) sfx_casc[t] += (mlen>=sfx_thr[t] ? accs_lb : acc) + 1;
                ++sfx_n; if(mlen>0) ++sfx_hit;
                if(accs_lb>acc) ++sfx_win; else if(accs_lb<acc) ++sfx_lose;
                printf("  [sfx] mlen=%d sdraft:", mlen);
                for(int j=0;j<VB-1;++j) printf(" %d",sdraft[j]);
                printf("  target:"); for(int j=0;j<VB-1;++j) printf(" %d",tam[j]);
                printf("  acc_sfx=%d(lb %d) acc_mtp=%d\n", accs, accs_lb, acc);
            }
            for(int i=0;i<acc;++i) sgen.push_back(draft[i]); sgen.push_back(correction);
            // update main_x for the accepted range [cpos..cpos+acc] from verify taps; rollback compressor T
            dspark_main_x(main_x+(size_t)cpos*d, mh_v, main_proj, main_proj_s, main_norm, acc+1, d, EPS); CU(cudaDeviceSynchronize());
            for(int L=0;L<N_LAYERS;++L){ int ratio=compress_ratio(L); if(!ratio) continue; int valid=0;
                for(int j=cpos;j<=cpos+acc;++j) if((j+1)%ratio==0) ++valid; KV[L].T=Tbefore[L]+valid; }   // drop rows from rejected drafts
            cpos += acc+1; cur = correction;
            cudaEventRecord(s1); cudaEventSynchronize(s1); float ms=0; cudaEventElapsedTime(&ms,s0,s1);
            if(specprof){ cudaEventRecord(p4); CU(cudaDeviceSynchronize());
                float a,b,c,e; cudaEventElapsedTime(&a,p0,p1); cudaEventElapsedTime(&b,p1,p2);
                cudaEventElapsedTime(&c,p2,p3); cudaEventElapsedTime(&e,p3,p4);
                if(nverify>0){ acc_kv+=a; acc_blk+=b; acc_head+=c; acc_ver+=e; ++nprof;
                    acc_dral+=rawd; acc_drsy+=rsyd; acc_dn+=rawnd; acc_dsn+=rsynd;
                    // the remainder of the round: dspark_main_x is the only instrumented call after
                    // p3, so this isolates the commit half from the draft half.
                    acc_rral+=(g_raw_ms-raw0)-rawd;   acc_rn +=(g_raw_n-rawn0)-rawnd;
                    acc_rrsy+=(g_rawsync_ms-rsy0)-rsyd; acc_rsn+=(g_rawsync_n-rsyn0)-rsynd; } }
            if(nverify>0){ spec_ms+=ms; timed_tok+=acc+1; } ++nverify;   // exclude round 0 (warmup: head repack)
            printf("  verify %d: accepted %d/%d (K=%d) + correction -> +%d tokens (%.1f ms)  cpos=%d\n", nverify, acc, VB-1, VB, acc+1, ms, cpos);
            if(adaptblk){
                acc_hist.push_back(acc);
                if((int)acc_hist.size() > ABLK_WIN) acc_hist.pop_front();
                if((int)acc_hist.size() == ABLK_WIN){
                    double mtok = 0; for(int a2 : acc_hist) mtok += a2 + 1;
                    mtok /= ABLK_WIN;                                   // mean tokens/verify == tau
                    int nb = BLK;
                    // Saturating: tau within 0.75 of the ceiling this block can deliver (BLK), so
                    // the drafter is being cut off rather than running out of confidence.
                    if(mtok >= BLK - 0.75 && BLK < ABLK_MAX) nb = BLK + 1;
                    // Idle width: tau more than 1.75 below the ceiling means we are paying for
                    // verify positions the adaptive gate is declining anyway.
                    else if(mtok <= BLK - 1.75 && BLK > ABLK_MIN) nb = BLK - 1;
                    if(nb != BLK){
                        printf("  [adaptblk] tau=%.2f over last %d -> BLK %d -> %d\n", mtok, ABLK_WIN, BLK, nb);
                        BLK = nb; acc_hist.clear();                    // re-measure at the new width
                    }
                }
            }
        }
        double avg_acc=(double)sgen.size()/nverify;
        double ms_per_tok = timed_tok>0 ? spec_ms/timed_tok : 0;
        if(specprof && nprof){
            const double tot=(acc_kv+acc_blk+acc_head+acc_ver)/nprof;
            printf("\n[specprof] per verify round, mean of %d (ms):\n", nprof);
            printf("[specprof]   draft: main_kv  %7.2f  (%4.1f%%)\n", acc_kv/nprof,  100*acc_kv /nprof/tot);
            printf("[specprof]   draft: 3 blocks %7.2f  (%4.1f%%)\n", acc_blk/nprof, 100*acc_blk/nprof/tot);
            printf("[specprof]   draft: fwd_head %7.2f  (%4.1f%%)  <- DEVICE-side AR over %d positions (F27)\n",
                   acc_head/nprof, 100*acc_head/nprof/tot, BLK);
            printf("[specprof]   verify 43 layer %7.2f  (%4.1f%%)\n", acc_ver/nprof, 100*acc_ver/nprof/tot);
            printf("[specprof]   TOTAL           %7.2f\n", tot);
            // Finding 82. HOST time inside raw cudaMalloc/cudaFree/cudaStreamSynchronize on the
            // DSpark draft path, which never got the arena the verify path runs on. Counts are
            // integers and are the part of this that no variance floor can touch.
            const double dral=acc_dral/nprof, drsy=acc_drsy/nprof, rral=acc_rral/nprof, rrsy=acc_rrsy/nprof;
            printf("[specprof]   -- raw allocator inside the DRAFT half (host time, not device) --\n");
            printf("[specprof]   cudaMalloc+Free %7.2f  (%4.1f%% of round)   %.1f calls/round\n",
                   dral, 100*dral/tot, (double)acc_dn/nprof);
            printf("[specprof]   cudaStreamSync  %7.2f  (%4.1f%% of round)   %.1f calls/round\n",
                   drsy, 100*drsy/tot, (double)acc_dsn/nprof);
            printf("[specprof]   draft raw TOTAL %7.2f  (%4.1f%% of round, %4.1f%% of the draft half)\n",
                   dral+drsy, 100*(dral+drsy)/tot,
                   (acc_kv+acc_blk+acc_head)>0 ? 100*(dral+drsy)/((acc_kv+acc_blk+acc_head)/nprof) : 0.0);
            printf("[specprof]   rest-of-round   %7.2f malloc/free + %.2f sync  (%.1f + %.1f calls)  <- dspark_main_x\n",
                   rral, rrsy, (double)acc_rn/nprof, (double)acc_rsn/nprof);
        }
        if(sfxprobe && sfx_n){
            const double N=(double)sfx_n, base=(double)sfx_mtp;
            printf("\n[sfx] S6 counterfactual over %d verifies, max n-gram %d (probe is READ-ONLY;\n"
                   "[sfx] the emitted sequence, GATE and LOSSLESS GATE are unaffected)\n", sfx_n, SFX_MAXNG);
            printf("[sfx]   MTP draft (SHIPPED)        : %4ld tokens  %.3f tok/verify\n", sfx_mtp, sfx_mtp/N);
            printf("[sfx]   suffix only, SOUND lower bd: %4ld tokens  %.3f tok/verify  (%+.1f%%)\n",
                   sfx_lb, sfx_lb/N, 100.0*(sfx_lb-base)/base);
            printf("[sfx]   suffix only, optimistic    : %4ld tokens  %.3f tok/verify  (%+.1f%%)\n",
                   sfx_raw, sfx_raw/N, 100.0*(sfx_raw-base)/base);
            printf("[sfx]   ORACLE max(MTP,suffix)     : %4ld tokens  %.3f tok/verify  (%+.1f%%)  <- S6 CEILING\n",
                   sfx_orc, sfx_orc/N, 100.0*(sfx_orc-base)/base);
            for(int t=0;t<SFX_NTHR;++t)
                printf("[sfx]   cascade: suffix if mlen>=%d  : %4ld tokens  %.3f tok/verify  (%+.1f%%)\n",
                       sfx_thr[t], sfx_casc[t], sfx_casc[t]/N, 100.0*(sfx_casc[t]-base)/base);
            printf("[sfx]   a suffix match existed in %d/%d verifies; suffix beat MTP in %d, lost in %d\n",
                   sfx_hit, sfx_n, sfx_win, sfx_lose);
        }
        printf("\n[spec] generated %d tokens over %d verifies: mean tokens/verify = %.2f (block=%d, max %d)\n", (int)sgen.size(), nverify, avg_acc, BLK, getenv("DSV4_NOVKPLUS")?BLK:BLK+1);
        printf("[spec] tokens:"); for(int i=0;i<(int)sgen.size() && i<40;++i) printf(" %d",sgen[i]); printf("\n");
        // DSV4_GENOUT=<file>: append the FULL prompt+generated id sequence, one line per sweep
        // point, in the exact ';'-separated form DSV4_PROMPTS accepts. This is what makes the
        // two-pass S5 capture possible: pass 1 regenerates responses with the target (which is what
        // every source in S5_RECIPE demands), pass 2 feeds prompt+response back as a prompt and
        // captures the taps over the whole sequence. The 40-token console dump is for eyeballing;
        // it cannot be parsed back because it is truncated, which is exactly the trap.
        if(const char* go = getenv("DSV4_GENOUT")){
            if(FILE* gf = fopen(go, "a")){
                const std::vector<int>& pr = prompts[promptSweep[bsi]];
                for(size_t i=0;i<pr.size();++i) fprintf(gf, "%s%d", i?",":"", pr[i]);
                for(size_t i=0;i<sgen.size();++i) fprintf(gf, ",%d", sgen[i]);
                fprintf(gf, "\n"); fclose(gf);
                printf("[genout] wrote %zu prompt + %zu generated ids to %s\n",
                       pr.size(), sgen.size(), go); fflush(stdout);
            } else fprintf(stderr, "[genout] cannot open %s\n", go);
        }
        // SPEC-vs-BASE EQUIVALENCE GATE (LOOP_LOG Finding 68). Speculative decoding is supposed to be
        // LOSSLESS: the verify corrects every draft, so the emitted sequence must equal what base AR
        // emits from the same prompt. Nothing checked that. The existing gates check the FIRST token
        // (argmax 11111) and MATCH 5/5 at one position, and split-K passed both while sending the
        // model into a degenerate repeating loop from token 6 onward — which RAISED acceptance 2.90
        // -> 3.86 and "improved" tok/s by 28%, because a repetitive sequence is trivially predictable.
        //
        // A quality regression that inflates the headline number is the worst failure mode this
        // engine has, and it costs one comparison to catch: base AR already generated `gen` from the
        // same prompt in the same process. Only the canonical prompt's points can be checked (other
        // sweep prompts have no base-AR run), which is enough — the failure was visible at token 6.
        if(promptSweep[bsi]==0 && !gen.empty()){
            size_t n = gen.size() < sgen.size() ? gen.size() : sgen.size();
            size_t bad = (size_t)-1;
            for(size_t i=0;i<n;++i) if(gen[i]!=sgen[i]) { bad=i; break; }
            if(bad==(size_t)-1)
                printf("[spec] LOSSLESS GATE: first %zu tokens match base AR -> PASS\n", n);
            else
                printf("[spec] LOSSLESS GATE: diverges from base AR at token %zu (%d vs %d) -> GATE FAIL\n",
                       bad, gen[bad], sgen[bad]);
            fflush(stdout);
        }
        printf("[spec] SPEC-DECODE: %.1f ms/tok = %.2f tok/s  (vs base M=1 %.1f ms/tok = %.2f tok/s -> %.2fx)\n",
               ms_per_tok, ms_per_tok>0?1000.0/ms_per_tok:0, warm_ms, 1000.0/warm_ms, ms_per_tok>0?warm_ms/ms_per_tok:0);
        if(shf){ fclose(shf); printf("[stephash] point %zu written to %s\n", bsi, stephash_path); fflush(stdout); }
        sweep_mstok[bsi]=ms_per_tok; sweep_acc[bsi]=avg_acc; sweep_akeff[bsi]=adaptK;
      }
      if(blkSweep.size()>1){
        printf("\n[blksweep]  BLK passes adaptK->eff prompt | mean tok/verify | ms/tok | tok/s | vs base %.2f tok/s\n", 1000.0/warm_ms);
        for(size_t i2=0;i2<blkSweep.size();++i2)
            printf("[blksweep] %4d %6d %6.2f %6.2f %6d | %15.2f | %6.1f | %5.2f | %.2fx\n", blkSweep[i2], passSweep[i2], adaptSweep[i2],
                   sweep_akeff[i2], promptSweep[i2], sweep_acc[i2], sweep_mstok[i2], 1000.0/sweep_mstok[i2], warm_ms/sweep_mstok[i2]);
      }
    }
    size_t fb,tb; cudaMemGetInfo(&fb,&tb); printf("[decode] mem %.1f/%.1f GiB\n",(tb-fb)/1073741824.0,tb/1073741824.0);
    return first_am==EXPECT?0:1;
}
