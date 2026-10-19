// draft.cu — DFlash draft model forward + propose (qwen3-style, 5 layers).
// Self-contained: BF16 weights decoded via (uint16<<16), fp32 activations,
// simple correctness-first kernels. Follows reference/DFLASH_SPEC.md exactly.
//
// Algorithm (single forward, parallel block-diffusion drafting):
//   (A) context K/V:  fused = fc @ concat6(taps); fused_n = rmsnorm(fused, hidden_norm);
//                     for each layer: K/V = k/v_proj @ fused_n; k_norm; RoPE(K) (V not roped);
//                     store into draft KV cache at ctx_pos.  (Every layer's K/V from same fused_n.)
//   (B) query forward: qids=[next_token, 4x15]; h=embed(qids) UNSCALED;
//                     5 qwen3 decoder layers (input_ln; q/k/v; q/k_norm; RoPE; append query K/V;
//                     NON-CAUSAL attn over context ∪ block, scale 1/sqrt(128); o_proj; resid;
//                     post_attn_ln; SiLU MLP; resid); final norm.
//   (C) logits = lm_head @ h[mask slots 1..15]; argmax -> draft ids.
#include "draft.h"
#include "fp4_gemm.h"
extern "C" void tc_w4a16_gemm(float*,const uint8_t*,const uint8_t*,float,const void*,int,int,int,cudaStream_t);
extern "C" void tc_bf16_gemm(float*,const void*,const void*,int,int,int,cudaStream_t);
#include "safetensors.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>

#define CU(x) do{cudaError_t e=(x); if(e){fprintf(stderr,"cuda %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)

// ---- qwen3 draft config (DFLASH_SPEC.md §0) ----
static const int H=2816, NL=5, NH=32, HD=128, NKV=8;
static const int QD=NH*HD /*4096*/, KD=NKV*HD /*1024*/, FFN=5632, FCIN=6*H /*16896*/;
static const int VOCAB=262144, MASK=4, BLK=16, NDRAFT=15;
static const float EPS=1e-6f, THETA=1e6f;
static const float SCALE=0.08838834764831845f;  // 1/sqrt(128)

__device__ __forceinline__ float bf2f(uint16_t h){ unsigned u=(unsigned)h<<16; float f; memcpy(&f,&u,4); return f; }

// ---------------- kernels ----------------
// out[M,N] = x[M,K] @ W_bf16[N,K]^T   (one thread per output element, loop over K)
// warp-per-output: out[m,n]=dot(x[m],W[n]) over K, uint2 bf16 loads (4/load) + shfl reduce
// warp-per-(m,n): handles ANY M (used for the large-M context forward, M=valid_len). Weight re-read per m
// but context weights (k/v_proj) are small so that's fine.
__global__ void k_linear_bf16_bigM(float* out,const float* x,const uint16_t* W,int M,int N,int K){
    int lane=threadIdx.x&31; long o=(long)blockIdx.x*(blockDim.x>>5)+(threadIdx.x>>5);
    if(o>=(long)M*N) return; int n=o%N,m=o/N;
    const float* xr=x+(size_t)m*K; const uint2* wv=(const uint2*)(W+(size_t)n*K);
    float acc=0.f; int nv=K/4;
    for(int vi=lane; vi<nv; vi+=32){ uint2 ww=wv[vi]; const uint16_t* b=(const uint16_t*)&ww; const float* xv=xr+vi*4;
        acc += xv[0]*bf2f(b[0])+xv[1]*bf2f(b[1])+xv[2]*bf2f(b[2])+xv[3]*bf2f(b[3]); }
    #pragma unroll
    for(int s=16;s>0;s>>=1) acc+=__shfl_down_sync(0xffffffffu,acc,s);
    if(lane==0) out[o]=acc;
}
// warp-per-output-COLUMN (M<=16): weight row W[n] read ONCE, reused across all M tokens (was M re-reads;
// critical for the M=15 lm_head over the 262144 vocab and the M=16 block query linears).
__global__ void k_df32to16(__half* o,const float* x,long n){ long i=(long)blockIdx.x*blockDim.x+threadIdx.x; if(i<n)o[i]=__float2half(x[i]); }
__global__ void k_linear_bf16(float* out,const __half* x16,const uint16_t* W,int M,int N,int K){
    int lane=threadIdx.x&31; long n=(long)blockIdx.x*(blockDim.x>>5)+(threadIdx.x>>5);
    if(n>=N) return;
    const uint2* wv=(const uint2*)(W+(size_t)n*K);
    float acc[16]; for(int m=0;m<M;++m) acc[m]=0.f; int nv=K/4;
    for(int vi=lane; vi<nv; vi+=32){ uint2 ww=__ldcs(&wv[vi]); const uint16_t* b=(const uint16_t*)&ww; int k=vi*4;
        __half2 w01=__floats2half2_rn(bf2f(b[0]),bf2f(b[1])), w23=__floats2half2_rn(bf2f(b[2]),bf2f(b[3]));
        for(int m=0;m<M;++m){ const __half2* xh=(const __half2*)(x16+(size_t)m*K+k);
            __half2 a=__hfma2(w23,xh[1],__hmul2(w01,xh[0]));
            acc[m]+=__half2float(__low2half(a))+__half2float(__high2half(a)); } }
    for(int m=0;m<M;++m){
        #pragma unroll
        for(int s=16;s>0;s>>=1) acc[m]+=__shfl_down_sync(0xffffffffu,acc[m],s);
        if(lane==0) out[(size_t)m*N+n]=acc[m];
    }
}
// RMSNorm: out = x*rsqrt(mean(x^2)+eps) [* bf16 weight if w!=null]. One block per row over `dim`.
__global__ void k_rmsnorm(float* out,const float* x,const uint16_t* w,int dim,float eps){
    int row=blockIdx.x; const float* xr=x+(size_t)row*dim; float* o=out+(size_t)row*dim;
    __shared__ float ss[256]; float local=0.f;
    for(int i=threadIdx.x;i<dim;i+=blockDim.x){ float v=xr[i]; local+=v*v; }
    ss[threadIdx.x]=local; __syncthreads();
    for(int s=blockDim.x/2;s>0;s>>=1){ if(threadIdx.x<s)ss[threadIdx.x]+=ss[threadIdx.x+s]; __syncthreads(); }
    float inv=rsqrtf(ss[0]/dim+eps);
    for(int i=threadIdx.x;i<dim;i+=blockDim.x){ float v=xr[i]*inv; if(w)v*=bf2f(w[i]); o[i]=v; }
}
// fused residual add: resid += h  (resid becomes the new residual stream r_new)
__global__ void k_add(float* acc,const float* in,int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n)acc[i]+=in[i]; }
// embed query ids, UNSCALED (no sqrt(2816)); see DFLASH_SPEC.md Q7 Q-embed note
__global__ void k_embed_unscaled(float* out,const uint16_t* emb,const int* ids,int seq,float esc){
    int t=blockIdx.x; for(int j=threadIdx.x;j<H;j+=blockDim.x) out[(size_t)t*H+j]=bf2f(emb[(size_t)ids[t]*H+j])*esc;
}
// rotate-half (neox) RoPE: x[rows,nh,hd], pairs dim i with i+hd/2. cos/sin: [rows, hd/2].
__global__ void k_rope(float* x,const float* cosT,const float* sinT,int nh,int hd){
    int r=blockIdx.y,h=blockIdx.x,half=hd/2,i=threadIdx.x; if(i>=half)return;
    float* xh=x+(((size_t)r*nh+h)*hd);
    const float* c=cosT+(size_t)r*half; const float* sn=sinT+(size_t)r*half;
    float x0=xh[i],x1=xh[i+half],cc=c[i],ss=sn[i];
    xh[i]=x0*cc-x1*ss; xh[i+half]=x1*cc+x0*ss;
}
// store K/V (fp32 [m,nkv,hd]) into contiguous cache at slot positions slots[t]
__global__ void k_store_kv(float* cache,const float* src,const int* slots,int m,int nkv,int hd){
    int i=blockIdx.x*blockDim.x+threadIdx.x; int n=m*nkv*hd; if(i>=n)return;
    int t=i/(nkv*hd), rest=i%(nkv*hd);
    cache[(size_t)slots[t]*nkv*hd + rest]=src[i];
}
// NON-CAUSAL attention: query i (0..BLK-1), head h. Attends to all `nkeys` cache slots in kslots.
// Q[BLK,nh,hd] fp32, cache K/V [.,nkv,hd] fp32. Online softmax. block=hd threads.
// blkstart = index in kslots where block keys begin (= context length). causal=1 -> block query i
// attends only to block keys 0..i (the 4 sliding draft layers); causal=0 -> bidirectional (full layer).
__global__ void k_attn(float* out,const float* Q,const float* Kc,const float* Vc,
                       const int* kslots,int nkeys,int nh,int nkv,int hd,float scale,int blkstart,int causal){
    int i=blockIdx.x,h=blockIdx.y,d=threadIdx.x; int kvh=h/(nh/nkv);
    extern __shared__ float sh[]; float* Qs=sh; float* acc=sh+hd; float* red=sh+2*hd;
    Qs[d]=Q[((size_t)i*nh+h)*hd+d]; acc[d]=0.f; __syncthreads();
    float m_run=-1e30f,l_run=0.f;
    for(int j=0;j<nkeys;++j){
        if(causal && j>=blkstart && (j-blkstart)>i) continue;   // causal within block (uniform across block threads)
        int slot=kslots[j];
        red[d]=Qs[d]*Kc[((size_t)slot*nkv+kvh)*hd+d]; __syncthreads();
        for(int s=hd/2;s>0;s>>=1){ if(d<s)red[d]+=red[d+s]; __syncthreads(); }
        float score=red[0]*scale;
        float nm=fmaxf(m_run,score), corr=__expf(m_run-nm), pj=__expf(score-nm);
        l_run=l_run*corr+pj;
        acc[d]=acc[d]*corr+pj*Vc[((size_t)slot*nkv+kvh)*hd+d]; m_run=nm; __syncthreads();
    }
    out[((size_t)i*nh+h)*hd+d]=acc[d]/l_run;
}
// SiLU-gate: g = silu(g) * u   (silu(x)=x*sigmoid(x))
__global__ void k_silu_mul(float* g,const float* u,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return; float x=g[i]; g[i]=(x/(1.f+__expf(-x)))*u[i];
}

// ---------------- model ----------------
struct DraftModel {
    st::SafeTensors* st; uint8_t* draw; const uint8_t* h0; int cap, slots;
    float* Kc[NL]; float* Vc[NL]; int ctx_done=0;   // # context positions already projected into the draft KV cache
    DraftModel(const char* path,int cap_):cap(cap_){
        st=new st::SafeTensors(path); h0=st->dataStart();
        CU(cudaMalloc(&draw, st->dataBytes()));
        CU(cudaMemcpy(draw, h0, st->dataBytes(), cudaMemcpyHostToDevice));
        slots=cap+BLK;
        for(int l=0;l<NL;++l){ size_t sz=(size_t)slots*NKV*HD*4;
            CU(cudaMalloc(&Kc[l],sz)); CU(cudaMalloc(&Vc[l],sz)); }
        printf("draft: uploaded %.2f GB weights, KV cache %d slots x %d layers\n",
               st->dataBytes()/1e9, slots, NL);
    }
    template<class P> P dptr(const std::string& n){ return (P)(draw + (st->get(n).data - h0)); }
};

DraftModel* draft_load(const char* path,int cap){ return new DraftModel(path,cap); }
void draft_free(DraftModel* d){ if(d){ delete d->st; delete d; } }

// ---- helpers ----
static void linbf(DraftModel* d,float* out,const float* x,const std::string& wname,int M,int N,int K){
    const uint16_t* W=d->dptr<const uint16_t*>(wname);
    if(M<=16){                                                                          // half2: fp16 acts, reuse W across M
        static __half* xf16=nullptr; if(!xf16) CU(cudaMalloc(&xf16,(size_t)16*FCIN*sizeof(__half)));   // K up to FCIN=16896 (fc)
        k_df32to16<<<(unsigned)(((long)M*K+255)/256),256>>>(xf16,x,(long)M*K);
        static bool noBTC=getenv("NOBF16TC")!=nullptr;
        if(!noBTC && (N%8==0) && (K%32==0)) tc_bf16_gemm(out, W, xf16, M, N, K, 0);          // Marlin-class bf16 TC (draft numerics only affect tau, verify keeps output bit-exact)
        else k_linear_bf16<<<(unsigned)(((long)N+7)/8),256>>>(out,xf16,W,M,N,K); }
    else      k_linear_bf16_bigM<<<(unsigned)(((long)M*N+7)/8),256>>>(out,x,W,M,N,K);   // large-M context
}
static void rmsn(DraftModel* d,float* out,const float* x,const std::string& wname,int rows,int dim){
    const uint16_t* w = wname.empty()? nullptr : d->dptr<const uint16_t*>(wname);
    k_rmsnorm<<<rows,256>>>(out,x,w,dim,EPS);
}
// RoPE cos/sin tables from explicit absolute positions (theta=1e6, head_dim=128, all dims rotated)
static void rope_tables(const int* pos,int n,float** dc,float** ds){
    int half=HD/2; std::vector<float> c((size_t)n*half), s((size_t)n*half);
    for(int r=0;r<n;++r) for(int j=0;j<half;++j){
        double inv=1.0/pow((double)THETA,(2.0*j)/HD); double a=(double)pos[r]*inv;
        c[(size_t)r*half+j]=(float)cos(a); s[(size_t)r*half+j]=(float)sin(a);
    }
    CU(cudaMalloc(dc,(size_t)n*half*4)); CU(cudaMalloc(ds,(size_t)n*half*4));
    CU(cudaMemcpy(*dc,c.data(),(size_t)n*half*4,cudaMemcpyHostToDevice));
    CU(cudaMemcpy(*ds,s.data(),(size_t)n*half*4,cudaMemcpyHostToDevice));
}
static std::string LP(int l,const char* s){ return "layers."+std::to_string(l)+"."+s; }

// device argmax over VOCAB for each of R rows (replaces 15.7MB D2H copy + serial host argmax)
__global__ void k_argmax_draft(int* out,const float* lg,int V){
    int r=blockIdx.x; __shared__ float sv[256]; __shared__ int si[256];
    float bv=-1e30f; int bi=0;
    for(int v=threadIdx.x;v<V;v+=256){ float x=lg[(size_t)r*V+v]; if(x>bv){bv=x;bi=v;} }
    sv[threadIdx.x]=bv; si[threadIdx.x]=bi; __syncthreads();
    for(int s=128;s>0;s>>=1){ if(threadIdx.x<s && sv[threadIdx.x+s]>sv[threadIdx.x]){sv[threadIdx.x]=sv[threadIdx.x+s];si[threadIdx.x]=si[threadIdx.x+s];} __syncthreads(); }
    if(threadIdx.x==0) out[r]=si[0];
}
// ---------------- propose ----------------
void draft_propose(DraftModel* d, const float* taps_dev, const int* ctx_pos, int C,
                   int next_token, const uint16_t* embed_bf16, const uint16_t* lmhead_bf16,
                   const uint8_t* ewp, const uint8_t* ews, float egs,
                   int* out_ids, int k){
    if(k>NDRAFT) k=NDRAFT;
    int P0 = ctx_pos[C-1]+1;

    // ---- (A) context K/V into draft cache — INCREMENTAL: only project the newly-committed positions
    int c0=d->ctx_done; if(c0>C)c0=0; int Cn=C-c0;   // new context positions [c0..C-1]; c0>C guards a fresh sequence
    if(Cn>0){
        float *fused,*fused_n; CU(cudaMalloc(&fused,(size_t)Cn*H*4)); CU(cudaMalloc(&fused_n,(size_t)Cn*H*4));
        linbf(d, fused, taps_dev+(size_t)c0*FCIN, "fc.weight", Cn, H, FCIN);   // taps for new positions
        rmsn(d, fused_n, fused, "hidden_norm.weight", Cn, H);
        int* ctx_dev; CU(cudaMalloc(&ctx_dev,Cn*4)); CU(cudaMemcpy(ctx_dev,ctx_pos+c0,Cn*4,cudaMemcpyHostToDevice));
        float *cdc,*cds; rope_tables(ctx_pos+c0,Cn,&cdc,&cds);
        float *Kctx,*Vctx; CU(cudaMalloc(&Kctx,(size_t)Cn*KD*4)); CU(cudaMalloc(&Vctx,(size_t)Cn*KD*4));
        for(int l=0;l<NL;++l){
            linbf(d, Kctx, fused_n, LP(l,"self_attn.k_proj.weight"), Cn, KD, H);
            linbf(d, Vctx, fused_n, LP(l,"self_attn.v_proj.weight"), Cn, KD, H);
            rmsn(d, Kctx, Kctx, LP(l,"self_attn.k_norm.weight"), Cn*NKV, HD);
            { dim3 g(NKV,Cn); k_rope<<<g,HD/2>>>(Kctx,cdc,cds,NKV,HD); }
            k_store_kv<<<(Cn*KD+255)/256,256>>>(d->Kc[l],Kctx,ctx_dev,Cn,NKV,HD);
            k_store_kv<<<(Cn*KD+255)/256,256>>>(d->Vc[l],Vctx,ctx_dev,Cn,NKV,HD);
        }
        CU(cudaDeviceSynchronize());
        cudaFree(fused);cudaFree(fused_n);cudaFree(ctx_dev);cudaFree(cdc);cudaFree(cds);cudaFree(Kctx);cudaFree(Vctx);
    }
    d->ctx_done=C;

    // ---- (B) query forward over BLK tokens ----
    // qids = [next_token, MASK x15];  qpos = [P0 .. P0+15]
    std::vector<int> qids(BLK), qpos(BLK), qslot(BLK);
    qids[0]=next_token; for(int i=1;i<BLK;++i) qids[i]=MASK;
    for(int i=0;i<BLK;++i){ qpos[i]=P0+i; qslot[i]=P0+i; }
    int *qid_dev,*qslot_dev; CU(cudaMalloc(&qid_dev,BLK*4)); CU(cudaMalloc(&qslot_dev,BLK*4));
    CU(cudaMemcpy(qid_dev,qids.data(),BLK*4,cudaMemcpyHostToDevice));
    CU(cudaMemcpy(qslot_dev,qslot.data(),BLK*4,cudaMemcpyHostToDevice));
    // key slot list for attention = context positions ∪ block positions  (non-causal)
    std::vector<int> kslots; kslots.reserve(C+BLK);
    for(int i=0;i<C;++i) kslots.push_back(ctx_pos[i]);
    for(int i=0;i<BLK;++i) kslots.push_back(P0+i);
    int nkeys=kslots.size(); int* kslot_dev; CU(cudaMalloc(&kslot_dev,nkeys*4));
    CU(cudaMemcpy(kslot_dev,kslots.data(),nkeys*4,cudaMemcpyHostToDevice));
    float *qdc,*qds; rope_tables(qpos.data(),BLK,&qdc,&qds);

    float *h,*resid,*hn,*q,*kk,*vv,*ao,*attn,*g,*u;
    CU(cudaMalloc(&h,(size_t)BLK*H*4));    CU(cudaMalloc(&resid,(size_t)BLK*H*4));
    CU(cudaMalloc(&hn,(size_t)BLK*H*4));   CU(cudaMalloc(&q,(size_t)BLK*QD*4));
    CU(cudaMalloc(&kk,(size_t)BLK*KD*4));  CU(cudaMalloc(&vv,(size_t)BLK*KD*4));
    CU(cudaMalloc(&ao,(size_t)BLK*QD*4));  CU(cudaMalloc(&attn,(size_t)BLK*H*4));
    CU(cudaMalloc(&g,(size_t)BLK*FFN*4));  CU(cudaMalloc(&u,(size_t)BLK*FFN*4));

    // EMPIRICAL: draft wants SCALED embeddings (sqrt(2816)) — tau 4.0 vs 2.67 unscaled, despite
    // DFLASH_SPEC.md Q7's UNSCALED guess. Default scaled; DEMB_OFF=1 reverts to unscaled.
    float esc = getenv("DEMB_OFF") ? 1.0f : 53.06599664f;
    k_embed_unscaled<<<BLK,256>>>(h, embed_bf16, qid_dev, BLK, esc);
    CU(cudaMemset(resid,0,(size_t)BLK*H*4));   // running residual stream (r_new = resid + h)
    int shmem=3*HD*4; dim3 agrid(BLK,NH);

    for(int l=0;l<NL;++l){
        // input_layernorm fused: resid += h; hn = rmsnorm(resid)
        k_add<<<(BLK*H+255)/256,256>>>(resid,h,BLK*H);
        rmsn(d, hn, resid, LP(l,"input_layernorm.weight"), BLK, H);
        // attention
        linbf(d, q,  hn, LP(l,"self_attn.q_proj.weight"), BLK, QD, H);
        linbf(d, kk, hn, LP(l,"self_attn.k_proj.weight"), BLK, KD, H);
        linbf(d, vv, hn, LP(l,"self_attn.v_proj.weight"), BLK, KD, H);
        rmsn(d, q,  q,  LP(l,"self_attn.q_norm.weight"), BLK*NH,  HD);   // per-head q_norm
        rmsn(d, kk, kk, LP(l,"self_attn.k_norm.weight"), BLK*NKV, HD);   // per-head k_norm
        k_rope<<<dim3(NH,BLK),HD/2>>>(q, qdc,qds,NH, HD);
        k_rope<<<dim3(NKV,BLK),HD/2>>>(kk,qdc,qds,NKV,HD);
        // append query K/V to cache at P0..P0+15 (V not roped)
        k_store_kv<<<(BLK*KD+255)/256,256>>>(d->Kc[l],kk,qslot_dev,BLK,NKV,HD);
        k_store_kv<<<(BLK*KD+255)/256,256>>>(d->Vc[l],vv,qslot_dev,BLK,NKV,HD);
        // sliding layers 0-3: CAUSAL within block; full layer 4: bidirectional (per DFlash layer_types)
        k_attn<<<agrid,HD,shmem>>>(ao,q,d->Kc[l],d->Vc[l],kslot_dev,nkeys,NH,NKV,HD,SCALE, nkeys-BLK, (l<4)?1:0);
        linbf(d, attn, ao, LP(l,"self_attn.o_proj.weight"), BLK, H, QD);
        // post_attention_layernorm fused: resid += attn; hn = rmsnorm(resid)
        k_add<<<(BLK*H+255)/256,256>>>(resid,attn,BLK*H);
        rmsn(d, hn, resid, LP(l,"post_attention_layernorm.weight"), BLK, H);
        // SiLU MLP: down(silu(gate@hn) * (up@hn))
        linbf(d, g, hn, LP(l,"mlp.gate_proj.weight"), BLK, FFN, H);
        linbf(d, u, hn, LP(l,"mlp.up_proj.weight"),   BLK, FFN, H);
        k_silu_mul<<<(BLK*FFN+255)/256,256>>>(g,u,BLK*FFN);
        linbf(d, h, g, LP(l,"mlp.down_proj.weight"), BLK, H, FFN);  // h = mlp output -> next layer
    }
    // final norm: resid += h; out = rmsnorm(resid, norm)
    k_add<<<(BLK*H+255)/256,256>>>(resid,h,BLK*H);
    float* hfin; CU(cudaMalloc(&hfin,(size_t)BLK*H*4));
    rmsn(d, hfin, resid, "norm.weight", BLK, H);
    CU(cudaDeviceSynchronize());

    // ---- (C) logits for the 15 MASK positions (query slots 1..15) -> argmax ----
    float* hmask = hfin + (size_t)1*H;   // rows 1..15
    float* logits; CU(cudaMalloc(&logits,(size_t)NDRAFT*VOCAB*4));
    { static __half* lmxf=nullptr; if(!lmxf) CU(cudaMalloc(&lmxf,(size_t)16*H*sizeof(__half)));   // fp16 acts for half2 lm_head
      k_df32to16<<<(unsigned)(((long)NDRAFT*H+255)/256),256>>>(lmxf,hmask,(long)NDRAFT*H);
      tc_w4a16_gemm(logits, ewp, ews, egs, lmxf, NDRAFT, VOCAB, H, 0); }   // NVFP4 draft lm_head (4x lighter than bf16)
    int* did; CU(cudaMalloc(&did,k*4));               // device argmax: only need k token ids (not 15.7MB logits)
    k_argmax_draft<<<k,256>>>(did,logits,VOCAB);
    CU(cudaMemcpy(out_ids,did,k*4,cudaMemcpyDeviceToHost)); CU(cudaFree(did));

    if(getenv("DBG_DRAFT")){ static int done=0; if(!done){ done=1;   // dump 1 propose call for PyTorch ref compare
        FILE* f=fopen("/tmp/dbg_meta.txt","w"); fprintf(f,"%d %d %d\n",C,BLK,k);
        for(int i=0;i<C;++i)fprintf(f,"%d ",ctx_pos[i]); fprintf(f,"\n");
        for(int i=0;i<BLK;++i)fprintf(f,"%d ",qids[i]); fprintf(f,"\n");
        for(int i=0;i<BLK;++i)fprintf(f,"%d ",qpos[i]); fprintf(f,"\n");
        for(int i=0;i<k;++i)fprintf(f,"%d ",out_ids[i]); fprintf(f,"\n"); fclose(f);
        std::vector<float> th((size_t)C*6*H); CU(cudaMemcpy(th.data(),taps_dev,(size_t)C*6*H*4,cudaMemcpyDeviceToHost));
        FILE* g=fopen("/tmp/dbg_taps.bin","wb"); fwrite(th.data(),4,(size_t)C*6*H,g); fclose(g);
        std::vector<float> hh((size_t)BLK*H); CU(cudaMemcpy(hh.data(),hfin,(size_t)BLK*H*4,cudaMemcpyDeviceToHost));
        FILE* h2=fopen("/tmp/dbg_hfin.bin","wb"); fwrite(hh.data(),4,(size_t)BLK*H,h2); fclose(h2);
        fprintf(stderr,"[DBG_DRAFT dumped C=%d k=%d]\n",C,k); } }

    cudaFree(qid_dev);cudaFree(qslot_dev);cudaFree(kslot_dev);   // phase-A buffers freed in the incremental block
    cudaFree(qdc);cudaFree(qds);cudaFree(h);cudaFree(resid);cudaFree(hn);cudaFree(q);
    cudaFree(kk);cudaFree(vv);cudaFree(ao);cudaFree(attn);cudaFree(g);cudaFree(u);
    cudaFree(hfin);cudaFree(logits);
}
