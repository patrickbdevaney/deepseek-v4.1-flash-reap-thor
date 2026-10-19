// forward.cu — full DeepSeek-V4-Flash-180B-REAP forward on Thor. embed -> HC-expand -> 43 blocks
// (block_forward L0-1 / compressed_block_forward L2-42) -> hc_head -> norm -> lm_head -> logits.
// Weights load zero-copy via WeightStore; dtypes the kernels need as fp32 (e8m0 scales, wo_a fp8, bf16
// norms/compressor/head) are dequantized at load. See ROADMAP Phase A. Gate 1 = this runs on real weights.
//   build: nvcc -O2 -std=c++17 -arch=sm_110a -I include src/forward.cu kernels/*.cu -o build/forward
#include <unordered_map>
#include "weight_store.h"
#include "deepseek_v4.h"
#include "block.h"
#include "compressed_block.h"
#include "dspark.h"        // DSpark MTP draft head (proxy)
#include "dspark_real.h"   // real DSpark head pieces (main_x, markov, tap_pool, forward_head)
#include "dspark_attn.h"   // dspark_main_kv, dspark_block_forward
#include "hc.h"            // hc_head
#include "mla_attn.h"      // rmsnorm
#include "compressor.h"    // gemm_fp32
#include "yarn.h"
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <vector>
#include <string>
#include <cstdio>
#include <cstring>
#include <cmath>
using namespace dsv4;
#define CU(x) do{cudaError_t e=(x); if(e){fprintf(stderr,"cuda %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)

static std::string key_map(const std::string& in){ std::string s=in; if(s.rfind("model.",0)==0) s=s.substr(6); return s; }

// ---------------- dequant kernels ----------------
__global__ void k_deq_e8m0(float* o, const uint8_t* in, size_t n){ size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x; if(i<n) o[i]=exp2f((float)in[i]-127.f); }
__global__ void k_deq_bf16(float* o, const __nv_bfloat16* in, size_t n){ size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x; if(i<n) o[i]=__bfloat162float(in[i]); }
// fp8(e4m3) weight * e8m0 block scale -> fp32. rows x cols, scale [rows/blk, cols/blk].
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
__global__ void k_hc_expand(float* out, const float* h, int s, int hc, int dim){   // [s,dim]->[s,hc,dim] repeat
    size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x; if(i>=(size_t)s*hc*dim) return; int t=i/(hc*dim), j=i%dim;
    out[i]=h[(size_t)t*dim+j];
}

// ---------------- dequant-caching loader ----------------
// NOTE: dequant-at-load CACHING was REVERTED — persisting scales (+5.5GB) on top of the 112.9GB forward
// footprint starved this shared unified-memory device (user had to power-cycle). On Thor, unified RAM is shared
// with the host/OS/Claude Code; the forward already uses ~92% of it, so NO persistent additions are safe here.
// Dequant stays per-layer with release() (memory-safe). Killing the re-dequant needs a memory-NEUTRAL scheme.
struct Loader {
    st::WeightStore& W; std::vector<void*> allocs;
    Loader(st::WeightStore& w):W(w){}
    ~Loader(){ for(void*p:allocs) cudaFree(p); }
    size_t mark(){ return allocs.size(); }                           // per-layer dequant scoping: free
    void release(size_t m){ for(size_t i=m;i<allocs.size();++i) cudaFree(allocs[i]); allocs.resize(m); }  // buffers after the block runs (synced)
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
    std::vector<float> o((size_t)(s/ratio)*half); for(int g=0; g<s/ratio; ++g) for(int j=0;j<half;++j) o[(size_t)g*half+j]=in[(size_t)(g*ratio)*half+j]; return o; }

int main(int argc, char** argv){
    setvbuf(stdout, nullptr, _IONBF, 0);          // unbuffered: see progress live (esp. if killed)
    const char* dir = argc>1?argv[1]:"/home/patrickd/models/DeepSeek-V4-Flash-180B";
    std::vector<int> ids; int s;
    if(argc>2 && strchr(argv[2],',')){                       // explicit comma-separated token ids
        char* tok=strtok(argv[2],","); while(tok){ ids.push_back(atoi(tok)); tok=strtok(nullptr,","); } s=ids.size();
    } else { s = argc>2?atoi(argv[2]):8; ids.resize(s); for(int i=0;i<s;++i) ids[i]=(i*131+7)%VOCAB; }
    printf("[forward] loading %s (single-tenant, ~96 GiB)...  s=%d ids:", dir, s); for(int v:ids) printf(" %d",v); printf("\n");
    st::WeightStore W(dir, key_map);
    printf("[forward] loaded %.2f GiB, %zu tensors. prefill s=%d\n", W.loadedGiB(), W.count(), s);
    Loader L(W);
    const int half=ROPE_DIM/2, hc=HC_MULT, d=DIM;
    extern bool g_tc_fp8; g_tc_fp8 = true;                 // WIN: dense/attn fp8 GEMMs -> tc_fp8_gemm (no repack, no OOM) = 559.8 ms/tok (1.23x)
    extern bool g_tc_ogroup; g_tc_ogroup = true;           // WIN: attn-output grouped GEMM -> fp16 tensor core (cosine 1.0), ~13% warp-per-output -> TC
    extern bool g_moe_grouped; g_moe_grouped = (getenv("NOGROUPED")==nullptr);  // Step 1b: zero-sync grouped-GEMM MoE (off[] D2H gone). NOGROUPED=1 -> old per-expert path (A/B).
    extern void tc_moe_clear_cache();

    std::vector<void*> keep;
    std::vector<float> ssc,sss; yarn::freqs(ssc,sss,s,ROPE_DIM,0,ROPE_THETA,YARN_FACTOR,YARN_BETA_FAST,YARN_BETA_SLOW);
    const float *slide_cos=up_f(ssc,keep), *slide_sin=up_f(sss,keep);
    std::vector<float> cqc_h,cqs_h; yarn::freqs(cqc_h,cqs_h,s,ROPE_DIM,YARN_ORIG_MAXPOS,COMPRESS_ROPE_THETA,YARN_FACTOR,YARN_BETA_FAST,YARN_BETA_SLOW);
    const float *cqc=up_f(cqc_h,keep), *cqs=up_f(cqs_h,keep);
    const float *cc4c=up_f(stride_rows(cqc_h,s,half,4),keep), *cc4s=up_f(stride_rows(cqs_h,s,half,4),keep);
    const float *cc128c=(s>=128)?up_f(stride_rows(cqc_h,s,half,128),keep):cqc, *cc128s=(s>=128)?up_f(stride_rows(cqs_h,s,half,128),keep):cqs;

    int* d_ids; CU(cudaMalloc(&d_ids,s*4)); CU(cudaMemcpy(d_ids,ids.data(),s*4,cudaMemcpyHostToDevice));
    float *h0, *h, *h2, *main_hidden;
    CU(cudaMalloc(&h0,(size_t)s*d*4)); CU(cudaMalloc(&h,(size_t)s*hc*d*4)); CU(cudaMalloc(&h2,(size_t)s*hc*d*4));
    CU(cudaMalloc(&main_hidden,(size_t)s*3*d*4));          // DSpark taps: mean-pool of L40/41/42 output -> [s,3d]
    k_embed<<<((size_t)s*d+255)/256,256>>>(h0,(const __nv_bfloat16*)W.get("embed.weight").dev,d_ids,s,d);
    k_hc_expand<<<((size_t)s*hc*d+255)/256,256>>>(h,h0,s,hc,d);
    CU(cudaDeviceSynchronize());

    auto fill_moe=[&](const std::string& pfx, bool is_hash, MoEWeights& m, std::vector<const uint8_t*>& p1,std::vector<const uint8_t*>& p2,std::vector<const uint8_t*>& p3,
                      std::vector<const float*>& s1,std::vector<const float*>& s2,std::vector<const float*>& s3){
        p1.clear();p2.clear();p3.clear();s1.clear();s2.clear();s3.clear();   // idempotent: safe to re-run (2x warmup)
        std::string p=pfx+"ffn.";
        m.gate_w=L.bf16(p+"gate.weight"); m.is_hash=is_hash;
        m.gate_bias=m.is_hash?nullptr:(W.has(p+"gate.bias")?L.f32(p+"gate.bias"):nullptr);
        m.tid2eid=m.is_hash?(const long*)W.get(p+"gate.tid2eid").dev:nullptr;
        for(int e=0;e<N_ROUTED;++e){ std::string ep=p+"experts."+std::to_string(e)+".";
            p1.push_back(L.raw(ep+"w1.weight")); p2.push_back(L.raw(ep+"w2.weight")); p3.push_back(L.raw(ep+"w3.weight"));
            s1.push_back(L.scale(ep+"w1.scale")); s2.push_back(L.scale(ep+"w2.scale")); s3.push_back(L.scale(ep+"w3.scale")); }
        m.w1p=p1.data(); m.w2p=p2.data(); m.w3p=p3.data(); m.w1sp=s1.data(); m.w2sp=s2.data(); m.w3sp=s3.data();
        std::string sp=p+"shared_experts.";
        m.sw1=L.raw(sp+"w1.weight"); m.sw2=L.raw(sp+"w2.weight"); m.sw3=L.raw(sp+"w3.weight");
        m.sw1s=L.scale(sp+"w1.scale"); m.sw2s=L.scale(sp+"w2.scale"); m.sw3s=L.scale(sp+"w3.scale");
        m.n_routed=N_ROUTED; m.n_act=N_ACT; m.dim=DIM; m.inter=MOE_INTER; m.vocab=VOCAB; m.route_scale=ROUTE_SCALE; m.swiglu_limit=SWIGLU_LIMIT;
        m.use_tc_pp=true; m.batched=true; m.device_route=true; }; // TC pp MoE + batched + DEVICE-SIDE grouping (Step 1 -> graphs; gated cosine 1.0)
    auto fill_attn=[&](const std::string& pfx, MLAWeights& a, bool compressed){
        std::string p=pfx+"attn.";
        a.wq_a=L.raw(p+"wq_a.weight"); a.wq_a_s=L.scale(p+"wq_a.scale"); a.wq_b=L.raw(p+"wq_b.weight"); a.wq_b_s=L.scale(p+"wq_b.scale");
        a.wkv=L.raw(p+"wkv.weight"); a.wkv_s=L.scale(p+"wkv.scale"); a.wo_b=L.raw(p+"wo_b.weight"); a.wo_b_s=L.scale(p+"wo_b.scale");
        a.q_norm=L.bf16(p+"q_norm.weight"); a.kv_norm=L.bf16(p+"kv_norm.weight");
        a.wo_a=L.wo_a(p+"wo_a.weight",p+"wo_a.scale"); a.attn_sink=L.f32(p+"attn_sink");
        a.cosT=compressed?cqc:slide_cos; a.sinT=compressed?cqs:slide_sin; };

    std::vector<std::vector<const uint8_t*>> P1(N_LAYERS),P2(N_LAYERS),P3(N_LAYERS);
    std::vector<std::vector<const float*>> S1(N_LAYERS),S2(N_LAYERS),S3(N_LAYERS);

    cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1); float ms=0;
    // Head weights + head buffers dequanted/allocated ONCE (reused both passes; head.weight fp32 ~2.7GB — must
    // NOT re-alloc per pass or memory blows up). MEMORY-NEUTRAL 2x warmup: no per-pass persistent allocations.
    const float *head_w=L.bf16("head.weight"), *norm_w=L.bf16("norm.weight");
    const float *hc_fn=L.f32("hc_head_fn"), *hc_sc=L.f32("hc_head_scale"), *hc_bs=L.f32("hc_head_base");
    float* collapsed; CU(cudaMalloc(&collapsed,(size_t)s*d*4)); float* logits; CU(cudaMalloc(&logits,(size_t)s*VOCAB*4));
    // pass 0 = warmup (in-place repack via g_pp_done); pass 1 = steady-state decode (repack cached, no one-time cost).
    for(int pass=0; pass<2; ++pass){
    k_hc_expand<<<((size_t)s*hc*d+255)/256,256>>>(h,h0,s,hc,d); CU(cudaDeviceSynchronize());  // reset hidden from embedding
    cudaEventRecord(t0);
    for(int Lyr=0; Lyr<N_LAYERS; ++Lyr){
        int ratio=compress_ratio(Lyr);
        std::string lp="layers."+std::to_string(Lyr)+".";
        size_t mk=L.mark();                                   // free this layer's dequant buffers after the block
        if(ratio==0){
            BlockWeights b{}; fill_attn(lp,b.attn,false);
            fill_moe(lp,is_hash_layer(Lyr),b.ffn,P1[Lyr],P2[Lyr],P3[Lyr],S1[Lyr],S2[Lyr],S3[Lyr]);
            b.attn_norm=L.bf16(lp+"attn_norm.weight"); b.ffn_norm=L.bf16(lp+"ffn_norm.weight");
            b.hc_attn_fn=L.f32(lp+"hc_attn_fn"); b.hc_attn_scale=L.f32(lp+"hc_attn_scale"); b.hc_attn_base=L.f32(lp+"hc_attn_base");
            b.hc_ffn_fn=L.f32(lp+"hc_ffn_fn"); b.hc_ffn_scale=L.f32(lp+"hc_ffn_scale"); b.hc_ffn_base=L.f32(lp+"hc_ffn_base");
            b.dim=DIM; b.hc=HC_MULT;
            block_forward(h2,h,d_ids,b,s,HC_SINKHORN_ITERS,EPS);
        } else {
            CompressedBlockWeights b{}; fill_attn(lp,b.attn.attn,true);
            std::string p=lp+"attn.";
            b.attn.mc_wkv=L.bf16(p+"compressor.wkv.weight"); b.attn.mc_wgate=L.bf16(p+"compressor.wgate.weight");
            b.attn.mc_ape=L.f32(p+"compressor.ape"); b.attn.mc_norm=L.bf16(p+"compressor.norm.weight");
            b.attn.cc_cos=(ratio==4)?cc4c:cc128c; b.attn.cc_sin=(ratio==4)?cc4s:cc128s;
            if(ratio==4){
                b.attn.idx_wq_b=L.raw(p+"indexer.wq_b.weight"); b.attn.idx_wq_b_s=L.scale(p+"indexer.wq_b.scale");
                b.attn.idx_weights_proj=L.bf16(p+"indexer.weights_proj.weight");
                b.attn.idx_c_wkv=L.bf16(p+"indexer.compressor.wkv.weight"); b.attn.idx_c_wgate=L.bf16(p+"indexer.compressor.wgate.weight");
                b.attn.idx_c_ape=L.f32(p+"indexer.compressor.ape"); b.attn.idx_c_norm=L.bf16(p+"indexer.compressor.norm.weight");
            }
            b.attn.index_n_heads=INDEX_N_HEADS; b.attn.index_head_dim=INDEX_HEAD_DIM; b.attn.index_topk=INDEX_TOPK;
            fill_moe(lp,is_hash_layer(Lyr),b.ffn,P1[Lyr],P2[Lyr],P3[Lyr],S1[Lyr],S2[Lyr],S3[Lyr]);
            b.attn_norm=L.bf16(lp+"attn_norm.weight"); b.ffn_norm=L.bf16(lp+"ffn_norm.weight");
            b.hc_attn_fn=L.f32(lp+"hc_attn_fn"); b.hc_attn_scale=L.f32(lp+"hc_attn_scale"); b.hc_attn_base=L.f32(lp+"hc_attn_base");
            b.hc_ffn_fn=L.f32(lp+"hc_ffn_fn"); b.hc_ffn_scale=L.f32(lp+"hc_ffn_scale"); b.hc_ffn_base=L.f32(lp+"hc_ffn_base");
            b.dim=DIM; b.hc=HC_MULT; b.win=WINDOW; b.ratio=ratio;
            compressed_block_forward(h2,h,d_ids,b,s,HC_SINKHORN_ITERS,EPS);
        }
        std::swap(h,h2);
        if(Lyr==40) dspark_tap_pool(main_hidden,h,s,hc,d,0,3);   // DSpark taps: mean-pool output of L40/41/42
        else if(Lyr==41) dspark_tap_pool(main_hidden,h,s,hc,d,1,3);
        else if(Lyr==42) dspark_tap_pool(main_hidden,h,s,hc,d,2,3);
        L.release(mk);                                        // block synced internally -> safe to free layer dequant
        tc_moe_clear_cache();                                 // free this layer's repacked TC experts (avoid ~82GB accumulation)
        if(Lyr%4==0){ size_t fb,tb; cudaMemGetInfo(&fb,&tb);
            printf("  layer %d/%d done (ratio %d)  mem %.1f/%.1f GiB\n",Lyr,N_LAYERS,ratio,(tb-fb)/1073741824.0,tb/1073741824.0); }
    }

    // --- head: hc_head (4->1) -> final norm -> lm_head -> logits[s,vocab] --- (weights hoisted, reused both passes)
    hc_head(collapsed, h, hc_fn, hc_sc, hc_bs, s, hc, d, HC_EPS);
    rmsnorm(collapsed, collapsed, norm_w, s, d, EPS, true, 0);
    gemm_fp32(logits, collapsed, head_w, s, VOCAB, d, 0);   // [s,vocab]
    CU(cudaDeviceSynchronize()); cudaEventRecord(t1); cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms,t0,t1);
    printf("[pass %d] %.1f ms (%.1f ms/tok)%s\n", pass, ms, ms/s, pass==0?"  [warmup: one-time repack]":"  [WARM steady-state]");
    }  // end 2x warmup loop; ms = pass-1 (warm) time

    std::vector<float> lg((size_t)VOCAB); CU(cudaMemcpy(lg.data(),logits+(size_t)(s-1)*VOCAB,VOCAB*4,cudaMemcpyDeviceToHost));
    int am=0; for(int v=1;v<VOCAB;++v) if(lg[v]>lg[am]) am=v;
    size_t freeb,totb; cudaMemGetInfo(&freeb,&totb);
    printf("\n[Gate 1] prefill s=%d: %.1f ms (%.1f ms/tok). last-token argmax=%d logit=%.3f\n", s, ms, ms/s, am, lg[am]);
    printf("[Gate 1] GPU mem: %.1f/%.1f GiB used\n", (totb-freeb)/1073741824.0, totb/1073741824.0);
    printf("[Gate 1] FULL 180B FORWARD RAN ON THOR.\n");

    // ================= GATE 2: DSpark draft head + single-token acceptance tau =================
    std::vector<float> ml((size_t)s*VOCAB); CU(cudaMemcpy(ml.data(),logits,(size_t)s*VOCAB*4,cudaMemcpyDeviceToHost));
    std::vector<int> main_am(s);
    for(int t=0;t<s;++t){ const float* r=&ml[(size_t)t*VOCAB]; int a=0; for(int v=1;v<VOCAB;++v) if(r[v]>r[a])a=v; main_am[t]=a; }

    DSparkWeights dw{}; BlockWeights& mb=dw.block;
    std::vector<const uint8_t*> mp1,mp2,mp3; std::vector<const float*> ms1,ms2,ms3;
    fill_attn("mtp.0.", mb.attn, false);                                     // pure-sliding block
    fill_moe("mtp.0.", false, mb.ffn, mp1,mp2,mp3,ms1,ms2,ms3);
    mb.attn_norm=L.bf16("mtp.0.attn_norm.weight"); mb.ffn_norm=L.bf16("mtp.0.ffn_norm.weight");
    mb.hc_attn_fn=L.f32("mtp.0.hc_attn_fn"); mb.hc_attn_scale=L.f32("mtp.0.hc_attn_scale"); mb.hc_attn_base=L.f32("mtp.0.hc_attn_base");
    mb.hc_ffn_fn=L.f32("mtp.0.hc_ffn_fn"); mb.hc_ffn_scale=L.f32("mtp.0.hc_ffn_scale"); mb.hc_ffn_base=L.f32("mtp.0.hc_ffn_base");
    mb.dim=DIM; mb.hc=HC_MULT;
    dw.e_proj=L.raw("mtp.0.e_proj.weight"); dw.e_proj_s=L.scale("mtp.0.e_proj.scale");
    dw.h_proj=L.raw("mtp.0.h_proj.weight"); dw.h_proj_s=L.scale("mtp.0.h_proj.scale");
    dw.enorm=L.bf16("mtp.0.enorm.weight"); dw.hnorm=L.bf16("mtp.0.hnorm.weight"); dw.norm=L.bf16("mtp.0.norm.weight");
    dw.hc_head_fn=L.f32("mtp.0.hc_head_fn"); dw.hc_head_scale=L.f32("mtp.0.hc_head_scale"); dw.hc_head_base=L.f32("mtp.0.hc_head_base");
    dw.lm_head=L.bf16("head.weight"); dw.embed=(const __nv_bfloat16*)W.get("embed.weight").dev;
    dw.dim=DIM; dw.hc=HC_MULT; dw.vocab=VOCAB;

    float* draft; CU(cudaMalloc(&draft,(size_t)s*VOCAB*4));
    dspark_head_forward(draft, h, d_ids, dw, s, EPS);                        // tapped [s,hc,d] state
    std::vector<float> dl((size_t)s*VOCAB); CU(cudaMemcpy(dl.data(),draft,(size_t)s*VOCAB*4,cudaMemcpyDeviceToHost));
    std::vector<int> draft_am(s);
    for(int t=0;t<s;++t){ const float* r=&dl[(size_t)t*VOCAB]; int a=0; for(int v=1;v<VOCAB;++v) if(r[v]>r[a])a=v; draft_am[t]=a; }
    int m0=0,m1=0; for(int t=0;t<s;++t) if(draft_am[t]==main_am[t]) m0++;
    for(int t=0;t<s-1;++t) if(draft_am[t]==main_am[t+1]) m1++;
    printf("\n[Gate 2] DSpark draft head single-token acceptance (unfine-tuned on REAP):\n");
    printf("   tau@0  (draft[t]==main[t])   = %.3f  (%d/%d)\n", (double)m0/s, m0, s);
    printf("   tau@+1 (draft[t]==main[t+1]) = %.3f  (%d/%d)\n", (double)m1/(s-1), m1, s-1);
    printf("[Gate 2] main  argmax:"); for(int t=0;t<s;++t) printf(" %d",main_am[t]); printf("\n");
    printf("[Gate 2] draft argmax:"); for(int t=0;t<s;++t) printf(" %d",draft_am[t]); printf("\n");

    // ============ GATE 2-REAL: DSpark block-diffusion head -> block acceptance ============
    if(getenv("DSPARK_REAL") && argc>3){
        const char* headdir=argv[3]; const int BLK=DSPARK_BLOCK;
        printf("\n[Gate 2-real] loading DSpark head mtp.* from %s ...\n", headdir);
        st::WeightStore WH(headdir, key_map, "mtp."); Loader LH(WH);
        printf("[Gate 2-real] head loaded %.2f GiB, %zu mtp tensors\n", WH.loadedGiB(), WH.count());
        std::vector<float> bc,bs2; yarn::freqs(bc,bs2,s+BLK,ROPE_DIM,0,ROPE_THETA,YARN_FACTOR,YARN_BETA_FAST,YARN_BETA_SLOW);
        const float* blk_cos=up_f(bc,keep); const float* blk_sin=up_f(bs2,keep); const int hf=ROPE_DIM/2;
        float* main_x; CU(cudaMalloc(&main_x,(size_t)s*d*4));
        dspark_main_x(main_x, main_hidden, LH.raw("mtp.0.main_proj.weight"), LH.scale("mtp.0.main_proj.scale"), LH.bf16("mtp.0.main_norm.weight"), s, d, EPS);
        // ---- 3-stage DSpark chain (repo ships mtp.0/1/2; forward_spec chains all, head on mtp.2) ----
        int NSTAGE=0; while(WH.has("mtp."+std::to_string(NSTAGE)+".attn_norm.weight")) NSTAGE++;
        int NE=0; while(WH.has("mtp.0.ffn.experts."+std::to_string(NE)+".w1.weight")) NE++;
        printf("[Gate 2-real] DSpark stages=%d, head MoE experts=%d\n", NSTAGE, NE);
        std::vector<BlockWeights> mb(NSTAGE); std::vector<float*> mkv(NSTAGE);
        std::vector<std::vector<const uint8_t*>> HP1(NSTAGE),HP2(NSTAGE),HP3(NSTAGE);
        std::vector<std::vector<const float*>> HS1(NSTAGE),HS2(NSTAGE),HS3(NSTAGE);
        for(int st=0; st<NSTAGE; ++st){
            std::string b="mtp."+std::to_string(st)+".", p=b+"attn.";
            MLAWeights& a=mb[st].attn;
            a.wq_a=LH.raw(p+"wq_a.weight"); a.wq_a_s=LH.scale(p+"wq_a.scale"); a.wq_b=LH.raw(p+"wq_b.weight"); a.wq_b_s=LH.scale(p+"wq_b.scale");
            a.wkv=LH.raw(p+"wkv.weight"); a.wkv_s=LH.scale(p+"wkv.scale"); a.wo_b=LH.raw(p+"wo_b.weight"); a.wo_b_s=LH.scale(p+"wo_b.scale");
            a.q_norm=LH.bf16(p+"q_norm.weight"); a.kv_norm=LH.bf16(p+"kv_norm.weight");
            a.wo_a=LH.wo_a(p+"wo_a.weight",p+"wo_a.scale"); a.attn_sink=LH.f32(p+"attn_sink"); a.cosT=blk_cos; a.sinT=blk_sin;
            mb[st].dim=DIM; mb[st].hc=HC_MULT;
            mb[st].attn_norm=LH.bf16(b+"attn_norm.weight"); mb[st].ffn_norm=LH.bf16(b+"ffn_norm.weight");
            mb[st].hc_attn_fn=LH.f32(b+"hc_attn_fn"); mb[st].hc_attn_scale=LH.f32(b+"hc_attn_scale"); mb[st].hc_attn_base=LH.f32(b+"hc_attn_base");
            mb[st].hc_ffn_fn=LH.f32(b+"hc_ffn_fn"); mb[st].hc_ffn_scale=LH.f32(b+"hc_ffn_scale"); mb[st].hc_ffn_base=LH.f32(b+"hc_ffn_base");
            MoEWeights& m=mb[st].ffn; std::string fp=b+"ffn.";
            m.gate_w=LH.bf16(fp+"gate.weight"); m.is_hash=false; m.gate_bias=WH.has(fp+"gate.bias")?LH.f32(fp+"gate.bias"):nullptr; m.tid2eid=nullptr;
            for(int e=0;e<NE;++e){ std::string ep=fp+"experts."+std::to_string(e)+".";
                HP1[st].push_back(LH.raw(ep+"w1.weight")); HP2[st].push_back(LH.raw(ep+"w2.weight")); HP3[st].push_back(LH.raw(ep+"w3.weight"));
                HS1[st].push_back(LH.scale(ep+"w1.scale")); HS2[st].push_back(LH.scale(ep+"w2.scale")); HS3[st].push_back(LH.scale(ep+"w3.scale")); }
            m.w1p=HP1[st].data(); m.w2p=HP2[st].data(); m.w3p=HP3[st].data(); m.w1sp=HS1[st].data(); m.w2sp=HS2[st].data(); m.w3sp=HS3[st].data();
            std::string sp2=fp+"shared_experts.";
            m.sw1=LH.raw(sp2+"w1.weight"); m.sw2=LH.raw(sp2+"w2.weight"); m.sw3=LH.raw(sp2+"w3.weight");
            m.sw1s=LH.scale(sp2+"w1.scale"); m.sw2s=LH.scale(sp2+"w2.scale"); m.sw3s=LH.scale(sp2+"w3.scale");
            m.n_routed=NE; m.n_act=N_ACT; m.dim=DIM; m.inter=MOE_INTER; m.vocab=VOCAB; m.route_scale=ROUTE_SCALE; m.swiglu_limit=SWIGLU_LIMIT;
            CU(cudaMalloc(&mkv[st],(size_t)s*HEAD_DIM*4)); dspark_main_kv(mkv[st], main_x, a, s, EPS);   // each stage its own main-KV
        }
        std::string LS="mtp."+std::to_string(NSTAGE-1)+".";                                    // head params on LAST stage
        const float* hh_fn=LH.f32(LS+"hc_head_fn"); const float* hh_sc=LH.f32(LS+"hc_head_scale"); const float* hh_ba=LH.f32(LS+"hc_head_base");
        const float* hnorm=LH.bf16(LS+"norm.weight"); const float* lm=L.bf16("head.weight");
        const float* mw1=LH.bf16(LS+"markov_head.markov_w1.weight"); const float* mw2=LH.bf16(LS+"markov_head.markov_w2.weight");
        const __nv_bfloat16* emb=(const __nv_bfloat16*)W.get("embed.weight").dev;
        int nanchor=s-BLK-1, matched=0, total=0; std::vector<int> blkacc;
        float *xa,*xb2,*xemb; int *dblkids,*dfid,*dout1;
        CU(cudaMalloc(&xa,(size_t)BLK*hc*d*4)); CU(cudaMalloc(&xb2,(size_t)BLK*hc*d*4)); CU(cudaMalloc(&xemb,(size_t)BLK*d*4));
        CU(cudaMalloc(&dblkids,(size_t)BLK*4)); CU(cudaMalloc(&dfid,4)); CU(cudaMalloc(&dout1,(size_t)(BLK+1)*4));
        for(int t=0;t<nanchor;++t){
            std::vector<int> bid(BLK,DSPARK_NOISE_TID); bid[0]=ids[t+1];
            CU(cudaMemcpy(dblkids,bid.data(),(size_t)BLK*4,cudaMemcpyHostToDevice));
            k_embed<<<((size_t)BLK*d+255)/256,256>>>(xemb, emb, dblkids, BLK, d);
            k_hc_expand<<<((size_t)BLK*hc*d+255)/256,256>>>(xa, xemb, BLK, hc, d); CU(cudaDeviceSynchronize());
            float *cur=xa, *nxt=xb2;
            for(int st=0; st<NSTAGE; ++st){                                                    // chain h through all stages
                dspark_block_forward(nxt, cur, dblkids, mkv[st], t, mb[st], blk_cos+(size_t)(t+1)*hf, blk_sin+(size_t)(t+1)*hf, BLK, WINDOW, HC_SINKHORN_ITERS, EPS);
                float* tmp=cur; cur=nxt; nxt=tmp;
            }
            CU(cudaMemcpy(dfid,&ids[t+1],4,cudaMemcpyHostToDevice));
            dspark_forward_head(dout1, cur, dfid, hh_fn,hh_sc,hh_ba, hnorm, lm, mw1, mw2, 1, BLK, hc, d, VOCAB, DSPARK_MARKOV_RANK, EPS, nullptr);
            std::vector<int> oo(BLK+1); CU(cudaMemcpy(oo.data(),dout1,(size_t)(BLK+1)*4,cudaMemcpyDeviceToHost));
            int acc=0; for(int i=0;i<BLK && t+2+i<=s-1; ++i){ if(oo[1+i]==ids[t+2+i]) acc++; else break; }
            blkacc.push_back(acc); matched+=acc; total++;
        }
        printf("\n[Gate 2-real] DSpark BLOCK acceptance (unfine-tuned, vs prompt continuation):\n");
        printf("   mean accepted prefix = %.3f / %d  over %d anchors\n", total?(double)matched/total:0.0, BLK, total);
        printf("[Gate 2-real] per-anchor:"); for(int a:blkacc) printf(" %d",a); printf("\n");
    }
    return 0;
}
