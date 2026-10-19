// gate_units.cu — Gate K: run the CUDA kernels on the reference unit goldens and compare within tolerance.
//   build: nvcc -O2 -arch=sm_110a -I include tests/gate_units.cu kernels/fp8_block_gemm.cu \
//          kernels/hc_sinkhorn.cu -o build/gate_units
//   run:   ./build/gate_units ref/goldens
#include "safetensors.h"
#include "fp8_block_gemm.h"
#include "hc_sinkhorn.h"
#include "mla_attn.h"
#include "moe.h"
#include "hc.h"
#include "compressor.h"
#include "indexer.h"
#include "yarn.h"
#include <cstdio>
#include <vector>
#include <cmath>
#include <string>

#define CU(x) do{cudaError_t e=(x); if(e){fprintf(stderr,"cuda %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)

static void* up(const st::Tensor& t) {                     // upload raw bytes to device
    void* d; CU(cudaMalloc(&d, t.nbytes)); CU(cudaMemcpy(d, t.data, t.nbytes, cudaMemcpyHostToDevice)); return d;
}
static const float* f32(const st::Tensor& t){ return (const float*)t.data; }
static int i32(const st::Tensor& t, int i){ return ((const int*)t.data)[i]; }

struct Err { double max_abs=0, max_rel=0; };
static Err compare(const std::vector<float>& got, const float* ref, int n, double denom) {
    Err e;
    for (int i=0;i<n;++i){ double d=fabs((double)got[i]-ref[i]); e.max_abs=fmax(e.max_abs,d);
        e.max_rel=fmax(e.max_rel, d/(fabs((double)ref[i])+0.01*denom)); }
    return e;
}

static bool gate_fp8_gemm(const std::string& dir) {
    st::SafeTensors S(dir + "/unit_fp8_gemm.safetensors");
    const auto& dims = S.get("dims"); int M=i32(dims,0), N=i32(dims,1), K=i32(dims,2);
    void *dA=up(S.get("A_fp8")), *dB=up(S.get("B_fp8"));
    void *das=up(S.get("a_s")),  *dbs=up(S.get("b_s"));
    float* dC; CU(cudaMalloc(&dC, (size_t)M*N*4));
    fp8_block_gemm(dC, (const uint8_t*)dA, (const float*)das, (const uint8_t*)dB, (const float*)dbs, M,N,K);
    CU(cudaDeviceSynchronize());
    std::vector<float> C(M*N); CU(cudaMemcpy(C.data(), dC, (size_t)M*N*4, cudaMemcpyDeviceToHost));
    const float* Cref = f32(S.get("C_ref"));
    double mx=0; for(int i=0;i<M*N;++i) mx=fmax(mx,fabs((double)Cref[i]));
    Err e = compare(C, Cref, M*N, mx);
    bool ok = e.max_rel < 2e-2;
    printf("[fp8_block_gemm] M=%d N=%d K=%d  |C|max=%.4f  max_abs=%.5f  max_rel=%.5f  -> %s\n",
           M,N,K,mx,e.max_abs,e.max_rel, ok?"PASS":"FAIL");
    // native FP8 tensor-core GEMM (mma.sync.m16n8k32.e4m3) vs oracle — the 2x-fp16 compute path
    extern void tc_fp8_gemm(float*,const uint8_t*,const float*,const uint8_t*,const float*,int,int,int,cudaStream_t);
    float* dCt; CU(cudaMalloc(&dCt,(size_t)M*N*4));
    tc_fp8_gemm(dCt,(const uint8_t*)dA,(const float*)das,(const uint8_t*)dB,(const float*)dbs,M,N,K,0); CU(cudaDeviceSynchronize());
    std::vector<float> Ct((size_t)M*N); CU(cudaMemcpy(Ct.data(),dCt,(size_t)M*N*4,cudaMemcpyDeviceToHost));
    Err et=compare(Ct,Cref,M*N,mx); double dt=0,n1=0,n2=0; for(int i=0;i<M*N;++i){dt+=(double)Ct[i]*Cref[i];n1+=(double)Ct[i]*Ct[i];n2+=(double)Cref[i]*Cref[i];}
    double cos8=dt/(sqrt(n1)*sqrt(n2)+1e-30); bool ok8=cos8>0.999 && et.max_rel<5e-2;
    printf("[fp8_gemm TC m16n8k32] cosine=%.6f max_rel=%.5f -> %s\n", cos8, et.max_rel, ok8?"PASS":"FAIL"); ok=ok&&ok8;
    { cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b); int IT=50;
      for(int i=0;i<5;++i){ fp8_block_gemm(dC,(const uint8_t*)dA,(const float*)das,(const uint8_t*)dB,(const float*)dbs,M,N,K); tc_fp8_gemm(dCt,(const uint8_t*)dA,(const float*)das,(const uint8_t*)dB,(const float*)dbs,M,N,K,0);} CU(cudaDeviceSynchronize());
      cudaEventRecord(a); for(int i=0;i<IT;++i) fp8_block_gemm(dC,(const uint8_t*)dA,(const float*)das,(const uint8_t*)dB,(const float*)dbs,M,N,K); cudaEventRecord(b); cudaEventSynchronize(b); float t0=0; cudaEventElapsedTime(&t0,a,b);
      cudaEventRecord(a); for(int i=0;i<IT;++i) tc_fp8_gemm(dCt,(const uint8_t*)dA,(const float*)das,(const uint8_t*)dB,(const float*)dbs,M,N,K,0); cudaEventRecord(b); cudaEventSynchronize(b); float t1=0; cudaEventElapsedTime(&t1,a,b);
      printf("[fp8_gemm A/B] fp8_block_gemm %.4f ms | tc_fp8_gemm %.4f ms -> %.2fx\n", t0/IT, t1/IT, t0/t1); }
    cudaFree(dCt);
    cudaFree(dA);cudaFree(dB);cudaFree(das);cudaFree(dbs);cudaFree(dC);
    return ok;
}

static bool gate_hc(const std::string& dir) {
    st::SafeTensors S(dir + "/unit_hc_sinkhorn.safetensors");
    const auto& pp = S.get("params"); int n=i32(pp,0), hc=i32(pp,1), iters=i32(pp,2);
    void *dm=up(S.get("mixes")), *dsc=up(S.get("hc_scale")), *dba=up(S.get("hc_base"));
    float *dpre,*dpost,*dcomb;
    CU(cudaMalloc(&dpre,(size_t)n*hc*4)); CU(cudaMalloc(&dpost,(size_t)n*hc*4)); CU(cudaMalloc(&dcomb,(size_t)n*hc*hc*4));
    hc_sinkhorn(dpre,dpost,dcomb,(const float*)dm,(const float*)dsc,(const float*)dba,n,hc,iters,1e-6f);
    CU(cudaDeviceSynchronize());
    std::vector<float> pre(n*hc),post(n*hc),comb(n*hc*hc);
    CU(cudaMemcpy(pre.data(),dpre,(size_t)n*hc*4,cudaMemcpyDeviceToHost));
    CU(cudaMemcpy(post.data(),dpost,(size_t)n*hc*4,cudaMemcpyDeviceToHost));
    CU(cudaMemcpy(comb.data(),dcomb,(size_t)n*hc*hc*4,cudaMemcpyDeviceToHost));
    Err ep=compare(pre,f32(S.get("pre")),n*hc,1.0), eq=compare(post,f32(S.get("post")),n*hc,1.0),
        ec=compare(comb,f32(S.get("comb")),n*hc*hc,1.0);
    double mabs=fmax(ep.max_abs,fmax(eq.max_abs,ec.max_abs));
    bool ok = mabs < 1e-3;
    printf("[hc_sinkhorn] n=%d hc=%d iters=%d  max_abs pre=%.2e post=%.2e comb=%.2e  -> %s\n",
           n,hc,iters,ep.max_abs,eq.max_abs,ec.max_abs, ok?"PASS":"FAIL");
    cudaFree(dm);cudaFree(dsc);cudaFree(dba);cudaFree(dpre);cudaFree(dpost);cudaFree(dcomb);
    return ok;
}

static bool gate_sparse_attn(const std::string& dir) {
    st::SafeTensors S(dir + "/unit_sparse_attn.safetensors");
    const auto& dm = S.get("dims");
    int b=i32(dm,0), m=i32(dm,1), h=i32(dm,2), d=i32(dm,3), n=i32(dm,4), topk=i32(dm,5);
    float scale = f32(S.get("scale"))[0];
    void *dq=up(S.get("q")), *dkv=up(S.get("kv")), *dsink=up(S.get("attn_sink")), *didx=up(S.get("topk_idxs"));
    float* d_o; CU(cudaMalloc(&d_o,(size_t)b*m*h*d*4));
    sparse_attn(d_o,(const float*)dq,(const float*)dkv,(const float*)dsink,(const int*)didx,b,m,h,d,n,topk,scale);
    CU(cudaDeviceSynchronize());
    std::vector<float> o((size_t)b*m*h*d); CU(cudaMemcpy(o.data(),d_o,o.size()*4,cudaMemcpyDeviceToHost));
    const float* oref=f32(S.get("o_ref"));
    double mx=0; for(size_t i=0;i<o.size();++i) mx=fmax(mx,fabs((double)oref[i]));
    Err e=compare(o,oref,o.size(),mx);
    bool ok=e.max_rel<1e-2;
    printf("[sparse_attn] b=%d m=%d h=%d d=%d n=%d topk=%d  |o|max=%.4f max_abs=%.5f max_rel=%.5f -> %s\n",
           b,m,h,d,n,topk,mx,e.max_abs,e.max_rel, ok?"PASS":"FAIL");
    cudaFree(dq);cudaFree(dkv);cudaFree(dsink);cudaFree(didx);cudaFree(d_o); return ok;
}

static bool gate_rope(const std::string& dir) {
    st::SafeTensors S(dir + "/unit_rope.safetensors");
    const auto& dm=S.get("dims"); int rows=i32(dm,0), rd=i32(dm,1);
    void *dc=up(S.get("cos")), *ds=up(S.get("sin"));
    auto run=[&](const char* refname, bool inv)->Err{
        void* dx=up(S.get("x"));
        rope_interleaved((float*)dx,(const float*)dc,(const float*)ds,rows,rd,inv);
        CU(cudaDeviceSynchronize());
        std::vector<float> y((size_t)rows*rd); CU(cudaMemcpy(y.data(),dx,y.size()*4,cudaMemcpyDeviceToHost));
        Err e=compare(y,f32(S.get(refname)),y.size(),1.0); cudaFree(dx); return e;
    };
    Err ef=run("y_fwd",false), ei=run("y_inv",true);
    bool ok=fmax(ef.max_abs,ei.max_abs)<1e-4;
    printf("[rope] rows=%d rope_dim=%d  fwd_abs=%.2e inv_abs=%.2e -> %s\n",rows,rd,ef.max_abs,ei.max_abs,ok?"PASS":"FAIL");
    cudaFree(dc);cudaFree(ds); return ok;
}

static bool gate_rmsnorm(const std::string& dir) {
    st::SafeTensors S(dir + "/unit_rmsnorm.safetensors");
    const auto& dm=S.get("dims"); int rows=i32(dm,0), dim=i32(dm,1); float eps=f32(S.get("eps"))[0];
    void *dx=up(S.get("x")), *dw=up(S.get("weight"));
    float* dy; CU(cudaMalloc(&dy,(size_t)rows*dim*4));
    rmsnorm(dy,(const float*)dx,(const float*)dw,rows,dim,eps,true); CU(cudaDeviceSynchronize());
    std::vector<float> yw((size_t)rows*dim); CU(cudaMemcpy(yw.data(),dy,yw.size()*4,cudaMemcpyDeviceToHost));
    rmsnorm(dy,(const float*)dx,nullptr,rows,dim,eps,false); CU(cudaDeviceSynchronize());
    std::vector<float> yn((size_t)rows*dim); CU(cudaMemcpy(yn.data(),dy,yn.size()*4,cudaMemcpyDeviceToHost));
    Err ew=compare(yw,f32(S.get("y_w")),yw.size(),1.0), en=compare(yn,f32(S.get("y_now")),yn.size(),1.0);
    bool ok=fmax(ew.max_abs,en.max_abs)<1e-4;
    printf("[rmsnorm] rows=%d dim=%d  w_abs=%.2e now_abs=%.2e -> %s\n",rows,dim,ew.max_abs,en.max_abs,ok?"PASS":"FAIL");
    cudaFree(dx);cudaFree(dw);cudaFree(dy); return ok;
}

static bool gate_act_quant(const std::string& dir) {
    st::SafeTensors S(dir + "/unit_act_quant.safetensors");
    const auto& dm=S.get("dims"); int rows=i32(dm,0), dim=i32(dm,1), block=i32(dm,2);
    void* dx=up(S.get("x"));
    act_quant_fp8sim((float*)dx, rows, dim, block); CU(cudaDeviceSynchronize());
    std::vector<float> y((size_t)rows*dim); CU(cudaMemcpy(y.data(),dx,y.size()*4,cudaMemcpyDeviceToHost));
    const float* yr=f32(S.get("y_ref"));
    double mx=0; for(size_t i=0;i<y.size();++i) mx=fmax(mx,fabs((double)yr[i]));
    Err e=compare(y,yr,y.size(),mx);
    bool ok=e.max_rel<1e-3;
    printf("[act_quant] rows=%d dim=%d block=%d  max_abs=%.5f max_rel=%.5f -> %s\n",rows,dim,block,e.max_abs,e.max_rel,ok?"PASS":"FAIL");
    cudaFree(dx); return ok;
}

static bool gate_ogroup(const std::string& dir) {
    st::SafeTensors S(dir + "/unit_ogroup_gemm.safetensors");
    const auto& dm=S.get("dims"); int bs=i32(dm,0), G=i32(dm,1), R=i32(dm,2), Kd=i32(dm,3);
    void *doo=up(S.get("o")), *dw=up(S.get("wo_a"));
    float* dout; CU(cudaMalloc(&dout,(size_t)bs*G*R*4));
    ogroup_gemm(dout,(const float*)doo,(const float*)dw,bs,G,R,Kd); CU(cudaDeviceSynchronize());
    std::vector<float> out((size_t)bs*G*R); CU(cudaMemcpy(out.data(),dout,out.size()*4,cudaMemcpyDeviceToHost));
    const float* oref=f32(S.get("out_ref"));
    double mx=0; for(size_t i=0;i<out.size();++i) mx=fmax(mx,fabs((double)oref[i]));
    Err e=compare(out,oref,out.size(),mx);
    bool ok=e.max_rel<1e-2;
    printf("[ogroup_gemm] bs=%d G=%d R=%d Kd=%d  |out|max=%.4f max_abs=%.5f max_rel=%.5f -> %s\n",bs,G,R,Kd,mx,e.max_abs,e.max_rel,ok?"PASS":"FAIL");
    cudaFree(doo);cudaFree(dw);cudaFree(dout); return ok;
}

static bool gate_fp4_gemm(const std::string& dir) {
    st::SafeTensors S(dir + "/unit_fp4_gemm.safetensors");
    const auto& dm=S.get("dims"); int M=i32(dm,0), N=i32(dm,1), K=i32(dm,2);
    void *dA=up(S.get("A_fp8")), *das=up(S.get("a_s")), *dB=up(S.get("B_fp4")), *dbs=up(S.get("b_s"));
    float* dC; CU(cudaMalloc(&dC,(size_t)M*N*4));
    fp4_gemm(dC,(const uint8_t*)dA,(const float*)das,(const uint8_t*)dB,(const float*)dbs,M,N,K);
    CU(cudaDeviceSynchronize());
    std::vector<float> C(M*N); CU(cudaMemcpy(C.data(),dC,(size_t)M*N*4,cudaMemcpyDeviceToHost));
    const float* Cref=f32(S.get("C_ref")); double mx=0; for(int i=0;i<M*N;++i) mx=fmax(mx,fabs((double)Cref[i]));
    Err e=compare(C,Cref,M*N,mx); bool ok=e.max_rel<2e-2;
    printf("[fp4_gemm] M=%d N=%d K=%d  |C|max=%.4f max_abs=%.5f max_rel=%.5f -> %s\n",M,N,K,mx,e.max_abs,e.max_rel,ok?"PASS":"FAIL");
    cudaFree(dA);cudaFree(das);cudaFree(dB);cudaFree(dbs);cudaFree(dC); return ok;
}

static bool gate_router(const std::string& dir) {
    st::SafeTensors S(dir + "/unit_router.safetensors");
    const auto& dm=S.get("dims"); int n=i32(dm,0), dim=i32(dm,1), nr=i32(dm,2), topk=i32(dm,3);
    float rs=f32(S.get("route_scale"))[0];
    void *dx=up(S.get("x")), *dgw=up(S.get("gate_w")), *dbias=up(S.get("bias"));
    float* dw; int* di; CU(cudaMalloc(&dw,(size_t)n*topk*4)); CU(cudaMalloc(&di,(size_t)n*topk*4));
    moe_router_score(dw,di,(const float*)dx,(const float*)dgw,(const float*)dbias,n,dim,nr,topk,rs);
    CU(cudaDeviceSynchronize());
    std::vector<float> wv(n*topk); std::vector<int> iv(n*topk);
    CU(cudaMemcpy(wv.data(),dw,(size_t)n*topk*4,cudaMemcpyDeviceToHost));
    CU(cudaMemcpy(iv.data(),di,(size_t)n*topk*4,cudaMemcpyDeviceToHost));
    const float* wref=f32(S.get("weights_ref")); const int* iref=(const int*)S.get("indices_ref").data;
    Err e=compare(wv,wref,n*topk,1.0); int idx_mism=0; for(int i=0;i<n*topk;++i) if(iv[i]!=iref[i]) idx_mism++;
    bool ok = e.max_abs<1e-3 && idx_mism==0;
    printf("[router] n=%d dim=%d n_routed=%d topk=%d  w_abs=%.2e idx_mismatch=%d -> %s\n",n,dim,nr,topk,e.max_abs,idx_mism,ok?"PASS":"FAIL");
    cudaFree(dx);cudaFree(dgw);cudaFree(dbias);cudaFree(dw);cudaFree(di); return ok;
}

static bool gate_moe(const std::string& dir) {
    st::SafeTensors S(dir + "/unit_moe.safetensors");
    const auto& dm=S.get("dims"); int n=i32(dm,0), dim=i32(dm,1), inter=i32(dm,2), nr=i32(dm,3), na=i32(dm,4);
    MoEWeights w{};
    w.gate_w=(const float*)up(S.get("gate_w")); w.gate_bias=(const float*)up(S.get("bias")); w.tid2eid=nullptr; w.is_hash=false;
    w.w1=(const uint8_t*)up(S.get("w1")); w.w1s=(const float*)up(S.get("w1s"));
    w.w2=(const uint8_t*)up(S.get("w2")); w.w2s=(const float*)up(S.get("w2s"));
    w.w3=(const uint8_t*)up(S.get("w3")); w.w3s=(const float*)up(S.get("w3s"));
    w.sw1=(const uint8_t*)up(S.get("sw1")); w.sw1s=(const float*)up(S.get("sw1s"));
    w.sw2=(const uint8_t*)up(S.get("sw2")); w.sw2s=(const float*)up(S.get("sw2s"));
    w.sw3=(const uint8_t*)up(S.get("sw3")); w.sw3s=(const float*)up(S.get("sw3s"));
    w.n_routed=nr; w.n_act=na; w.dim=dim; w.inter=inter; w.vocab=0;
    w.route_scale=f32(S.get("rs"))[0]; w.swiglu_limit=f32(S.get("lim"))[0];
    const float* x=(const float*)up(S.get("x"));
    float* out; CU(cudaMalloc(&out,(size_t)n*dim*4));
    moe_forward(out, x, nullptr, w, n);
    CU(cudaDeviceSynchronize());
    std::vector<float> y((size_t)n*dim); CU(cudaMemcpy(y.data(),out,y.size()*4,cudaMemcpyDeviceToHost));
    const float* yref=f32(S.get("y_ref")); double mx=0; for(size_t i=0;i<y.size();++i) mx=fmax(mx,fabs((double)yref[i]));
    Err e=compare(y,yref,y.size(),mx); bool ok=e.max_rel<2e-2;
    // batched/grouped dispatch vs per-token oracle (scatter atomicAdd reorders -> cosine, not bit-exact)
    w.batched=true; float* out2; CU(cudaMalloc(&out2,(size_t)n*dim*4));
    moe_forward(out2, x, nullptr, w, n); CU(cudaDeviceSynchronize());
    std::vector<float> y2((size_t)n*dim); CU(cudaMemcpy(y2.data(),out2,y2.size()*4,cudaMemcpyDeviceToHost));
    double dt=0,n1=0,n2=0; for(size_t i=0;i<y.size();++i){dt+=y[i]*y2[i];n1+=y[i]*y[i];n2+=y2[i]*y2[i];}
    double cosb=dt/(sqrt(n1)*sqrt(n2)+1e-30);
    // INFORMATIONAL (not gating Gate K): batched dispatch is WIP — gate caught a correctness bug (cosine<1).
    // Per-token oracle (above) stays bit-exact and is the default. Do NOT enable MoEWeights.batched until this PASSES.
    bool okb=cosb>0.9999; printf("[moe batched] cosine vs per-token oracle=%.7f -> %s\n", cosb, okb?"PASS":"FAIL"); ok=ok&&okb;
    // device-side grouping (Step 1 -> graphs): GPU counting-sort, order-invariant -> cosine 1.0 vs oracle
    w.device_route=true; float* outd; CU(cudaMalloc(&outd,(size_t)n*dim*4));
    moe_forward(outd, x, nullptr, w, n); CU(cudaDeviceSynchronize());
    std::vector<float> yd((size_t)n*dim); CU(cudaMemcpy(yd.data(),outd,yd.size()*4,cudaMemcpyDeviceToHost));
    double dd=0,nd1=0,nd2=0; for(size_t i=0;i<y.size();++i){dd+=(double)y[i]*yd[i];nd1+=(double)y[i]*y[i];nd2+=(double)yd[i]*yd[i];}
    double cosd=dd/(sqrt(nd1)*sqrt(nd2)+1e-30); bool okd=cosd>0.9999; ok=ok&&okd;
    printf("[moe device_route] cosine vs per-token oracle=%.7f -> %s\n", cosd, okd?"PASS":"FAIL");
    w.device_route=false; cudaFree(outd);
    // compounded fast path: batched dispatch + TC GEMM (use_tc) — the real decode win. cosine (batched reorder + fp16-act).
    w.use_tc=true; float* out3; CU(cudaMalloc(&out3,(size_t)n*dim*4));
    moe_forward(out3, x, nullptr, w, n); CU(cudaDeviceSynchronize());
    std::vector<float> y3((size_t)n*dim); CU(cudaMemcpy(y3.data(),out3,y3.size()*4,cudaMemcpyDeviceToHost));
    double d3=0,a3=0,b3=0; for(size_t i=0;i<y.size();++i){d3+=y[i]*y3[i];a3+=y[i]*y[i];b3+=y3[i]*y3[i];}
    double cosc=d3/(sqrt(a3)*sqrt(b3)+1e-30); bool okc=cosc>0.999;
    printf("[moe batched+TC] cosine vs oracle=%.7f -> %s\n", cosc, okc?"PASS":"FAIL"); ok=ok&&okc; cudaFree(out3);
    // A/B timing (measured delta per Constitution VI.5): per-token vs batched vs batched+TC. Note: TINY gate
    // shapes (dim=256,inter=128) — shows DISPATCH amortization (fewer launches), NOT the real-model TC 19.7x.
    { cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b); int IT=20;
      auto tm=[&](bool bat,bool tc)->float{ w.batched=bat; w.use_tc=tc; for(int i=0;i<3;++i) moe_forward(out,x,nullptr,w,n); CU(cudaDeviceSynchronize());
        cudaEventRecord(a); for(int i=0;i<IT;++i) moe_forward(out,x,nullptr,w,n); cudaEventRecord(b); cudaEventSynchronize(b); float ms=0; cudaEventElapsedTime(&ms,a,b); return ms/IT; };
      float t0=tm(false,false), t1=tm(true,false), t2=tm(true,true);
      printf("[moe A/B] per-token %.3f ms | batched %.3f ms (%.2fx) | batched+TC %.3f ms (%.2fx)\n", t0,t1,t0/t1,t2,t0/t2);
      w.batched=false; w.use_tc=false; }
    // DETERMINISM (LOOP_LOG Finding 60). The engine produces different numbers on byte-identical
    // input — 36 identical sweep points give 36 distinct first-verify margin vectors — and the
    // prefill attention chain has been cleared by tests/gate_scratch_init at every length 1..29. The
    // MoE is the only kernel on the decode path with an atomic: k_moe_scatter takes each (token,
    // expert) pair's position within its expert's run from atomicAdd(&cursor[e],1), so the ORDER of
    // rows inside a group is race-determined. The argument that this is still bit-exact is that
    // `allslot` is i%na, not the atomic position, so k_scatter_ts writes a fixed (token,slot)
    // destination and k_reduce_ts sums the slots in a fixed order. That argument has never been
    // TESTED, and it is exactly the kind of claim that is true of the code someone read and false of
    // the code that runs. Run the decode configuration repeatedly on one input and compare bitwise.
    { w.batched=true; w.device_route=true; w.use_tc=true;      // the configuration decode actually sets
      float* od; CU(cudaMalloc(&od,(size_t)n*dim*4));
      std::vector<float> ref((size_t)n*dim), cur((size_t)n*dim);
      moe_forward(od,x,nullptr,w,n); CU(cudaDeviceSynchronize());
      CU(cudaMemcpy(ref.data(),od,ref.size()*4,cudaMemcpyDeviceToHost));
      int ndet=0; size_t worst=0;
      for(int it=0; it<32; ++it){
          CU(cudaMemset(od,0xA5,(size_t)n*dim*4));
          moe_forward(od,x,nullptr,w,n); CU(cudaDeviceSynchronize());
          CU(cudaMemcpy(cur.data(),od,cur.size()*4,cudaMemcpyDeviceToHost));
          size_t nd=0; for(size_t i=0;i<ref.size();++i) if(memcmp(&ref[i],&cur[i],4)!=0) ++nd;
          if(nd){ ++ndet; if(nd>worst) worst=nd; }
      }
      bool detok = (ndet==0);
      printf("[moe determinism] 32 repeats of the DECODE config on one input: %d differ (worst %zu/%zu elements) -> %s\n",
             ndet, worst, ref.size(), detok?"PASS":"FAIL — moe_forward is not reproducible");
      ok = ok && detok;
      w.batched=false; w.device_route=false; w.use_tc=false; cudaFree(od); }
    printf("[moe] n=%d dim=%d inter=%d nr=%d na=%d  |y|max=%.4f max_abs=%.5f max_rel=%.5f -> %s\n",
           n,dim,inter,nr,na,mx,e.max_abs,e.max_rel,ok?"PASS":"FAIL");
    return ok;
}

static bool gate_hc(const std::string& dir, int) {
    st::SafeTensors S(dir + "/unit_hc.safetensors");
    const auto& dm=S.get("dims"); int bs=i32(dm,0), hc=i32(dm,1), d=i32(dm,2), iters=i32(dm,3);
    void *dx=up(S.get("x")), *dfn=up(S.get("hc_fn")), *dsc=up(S.get("hc_scale")), *dba=up(S.get("hc_base"));
    void *dxn=up(S.get("x_new"));
    float *y,*post,*comb; CU(cudaMalloc(&y,(size_t)bs*d*4)); CU(cudaMalloc(&post,(size_t)bs*hc*4)); CU(cudaMalloc(&comb,(size_t)bs*hc*hc*4));
    hc_pre(y,post,comb,(const float*)dx,(const float*)dfn,(const float*)dsc,(const float*)dba,bs,hc,d,iters,1e-6f);
    CU(cudaDeviceSynchronize());
    std::vector<float> vy(bs*d),vp(bs*hc),vc(bs*hc*hc);
    CU(cudaMemcpy(vy.data(),y,vy.size()*4,cudaMemcpyDeviceToHost));
    CU(cudaMemcpy(vp.data(),post,vp.size()*4,cudaMemcpyDeviceToHost));
    CU(cudaMemcpy(vc.data(),comb,vc.size()*4,cudaMemcpyDeviceToHost));
    Err ey=compare(vy,f32(S.get("y_pre")),vy.size(),1.0), ep=compare(vp,f32(S.get("post")),vp.size(),1.0), ec=compare(vc,f32(S.get("comb")),vc.size(),1.0);
    // hc_post using CUDA-computed post/comb, residual = x
    float* y2; CU(cudaMalloc(&y2,(size_t)bs*hc*d*4));
    hc_post(y2,(const float*)dxn,(const float*)dx,post,comb,bs,hc,d); CU(cudaDeviceSynchronize());
    std::vector<float> vy2((size_t)bs*hc*d); CU(cudaMemcpy(vy2.data(),y2,vy2.size()*4,cudaMemcpyDeviceToHost));
    Err e2=compare(vy2,f32(S.get("y_post")),vy2.size(),1.0);
    double m=fmax(fmax(ey.max_abs,ep.max_abs),fmax(ec.max_abs,e2.max_abs));
    bool ok=m<1e-3;
    printf("[hc] bs=%d hc=%d d=%d  pre=%.2e post=%.2e comb=%.2e postout=%.2e -> %s\n",bs,hc,d,ey.max_abs,ep.max_abs,ec.max_abs,e2.max_abs,ok?"PASS":"FAIL");
    cudaFree(dx);cudaFree(dfn);cudaFree(dsc);cudaFree(dba);cudaFree(dxn);cudaFree(y);cudaFree(post);cudaFree(comb);cudaFree(y2); return ok;
}

static bool gate_compressor(const std::string& dir) {
    st::SafeTensors S(dir + "/unit_compressor.safetensors");
    const auto& dm=S.get("dims"); int bs=i32(dm,0), dim=i32(dm,1), d=i32(dm,2), ratio=i32(dm,3);
    int groups=bs/ratio;
    void *dx=up(S.get("x")), *dwkv=up(S.get("wkv")), *dwg=up(S.get("wgate")), *dape=up(S.get("ape"));
    float *kv,*score,*pooled;
    CU(cudaMalloc(&kv,(size_t)bs*d*4)); CU(cudaMalloc(&score,(size_t)bs*d*4)); CU(cudaMalloc(&pooled,(size_t)groups*d*4));
    gemm_fp32(kv,(const float*)dx,(const float*)dwkv,bs,d,dim);
    gemm_fp32(score,(const float*)dx,(const float*)dwg,bs,d,dim);
    compressor_pool(pooled,kv,score,(const float*)dape,groups,ratio,d); CU(cudaDeviceSynchronize());
    std::vector<float> p((size_t)groups*d); CU(cudaMemcpy(p.data(),pooled,p.size()*4,cudaMemcpyDeviceToHost));
    const float* pr=f32(S.get("pooled")); double mx=0; for(size_t i=0;i<p.size();++i) mx=fmax(mx,fabs((double)pr[i]));
    Err e=compare(p,pr,p.size(),mx); bool ok=e.max_rel<1e-3;
    printf("[compressor] bs=%d d=%d ratio=%d  |pooled|max=%.4f max_abs=%.5f max_rel=%.5f -> %s\n",bs,d,ratio,mx,e.max_abs,e.max_rel,ok?"PASS":"FAIL");
    cudaFree(dx);cudaFree(dwkv);cudaFree(dwg);cudaFree(dape);cudaFree(kv);cudaFree(score);cudaFree(pooled); return ok;
}

static bool gate_compressor_overlap(const std::string& dir) {
    st::SafeTensors S(dir + "/unit_compressor_overlap.safetensors");
    const auto& dm=S.get("dims"); int s=i32(dm,0), d=i32(dm,1), ratio=i32(dm,2); int groups=s/ratio;
    void *dkv=up(S.get("kv")), *dsc=up(S.get("score")), *dape=up(S.get("ape"));
    float* pooled; CU(cudaMalloc(&pooled,(size_t)groups*d*4));
    compressor_pool_overlap(pooled,(const float*)dkv,(const float*)dsc,(const float*)dape,groups,ratio,d);
    CU(cudaDeviceSynchronize());
    std::vector<float> p((size_t)groups*d); CU(cudaMemcpy(p.data(),pooled,p.size()*4,cudaMemcpyDeviceToHost));
    const float* pr=f32(S.get("pooled")); double mx=0; for(size_t i=0;i<p.size();++i) mx=fmax(mx,fabs((double)pr[i]));
    Err e=compare(p,pr,p.size(),mx); bool ok=e.max_rel<1e-3;
    printf("[compressor_overlap] s=%d d=%d ratio=%d  |p|max=%.4f max_abs=%.5f max_rel=%.5f -> %s\n",s,d,ratio,mx,e.max_abs,e.max_rel,ok?"PASS":"FAIL");
    cudaFree(dkv);cudaFree(dsc);cudaFree(dape);cudaFree(pooled); return ok;
}

static bool gate_compressor_full(const std::string& dir, bool rotate) {
    std::string fn = rotate ? "/unit_compressor_full_rotate.safetensors" : "/unit_compressor_full.safetensors";
    st::SafeTensors S(dir + fn);
    const auto& dm=S.get("dims"); int s=i32(dm,0), dim=i32(dm,1), d=i32(dm,2), ratio=i32(dm,3), rd=i32(dm,4); int groups=s/ratio;
    void *dx=up(S.get("x")), *dwkv=up(S.get("wkv")), *dwg=up(S.get("wgate")), *dape=up(S.get("ape"));
    void *dn=up(S.get("norm_w")), *dc=up(S.get("cos")), *dsin=up(S.get("sin"));
    float* out; CU(cudaMalloc(&out,(size_t)groups*d*4));
    compressor_forward(out,(const float*)dx,(const float*)dwkv,(const float*)dwg,(const float*)dape,
                       (const float*)dn,(const float*)dc,(const float*)dsin,s,dim,d,ratio,true,rd,1e-6f,rotate);
    CU(cudaDeviceSynchronize());
    std::vector<float> o((size_t)groups*d); CU(cudaMemcpy(o.data(),out,o.size()*4,cudaMemcpyDeviceToHost));
    const float* orf=f32(S.get("out")); double mx=0; for(size_t i=0;i<o.size();++i) mx=fmax(mx,fabs((double)orf[i]));
    Err e=compare(o,orf,o.size(),mx); bool ok=e.max_rel< (rotate?2e-2:5e-3);
    printf("[compressor_full rotate=%d] s=%d dim=%d d=%d  |o|max=%.4f max_abs=%.5f max_rel=%.5f -> %s\n",(int)rotate,s,dim,d,mx,e.max_abs,e.max_rel,ok?"PASS":"FAIL");
    cudaFree(dx);cudaFree(dwkv);cudaFree(dwg);cudaFree(dape);cudaFree(dn);cudaFree(dc);cudaFree(dsin);cudaFree(out); return ok;
}

static bool gate_hadamard(const std::string& dir) {
    st::SafeTensors S(dir + "/unit_hadamard.safetensors");
    const auto& dm=S.get("dims"); int rows=i32(dm,0), D=i32(dm,1);
    void* dx=up(S.get("x")); float* dy; CU(cudaMalloc(&dy,(size_t)rows*D*4));
    hadamard(dy,(const float*)dx,rows,D); CU(cudaDeviceSynchronize());
    std::vector<float> y((size_t)rows*D); CU(cudaMemcpy(y.data(),dy,y.size()*4,cudaMemcpyDeviceToHost));
    const float* yr=f32(S.get("y")); double mx=0; for(size_t i=0;i<y.size();++i) mx=fmax(mx,fabs((double)yr[i]));
    Err e=compare(y,yr,y.size(),mx); bool ok=e.max_rel<1e-3;
    printf("[hadamard] rows=%d D=%d  |y|max=%.4f max_abs=%.5f max_rel=%.5f -> %s\n",rows,D,mx,e.max_abs,e.max_rel,ok?"PASS":"FAIL");
    cudaFree(dx);cudaFree(dy); return ok;
}

static bool gate_index_score(const std::string& dir) {
    st::SafeTensors S(dir + "/unit_index_score.safetensors");
    const auto& dm=S.get("dims"); int Sn=i32(dm,0), H=i32(dm,1), d=i32(dm,2), T=i32(dm,3);
    void *dq=up(S.get("q")), *dkv=up(S.get("kv")), *dw=up(S.get("weights"));
    float* sc; CU(cudaMalloc(&sc,(size_t)Sn*T*4));
    index_score(sc,(const float*)dq,(const float*)dkv,(const float*)dw,Sn,T,H,d); CU(cudaDeviceSynchronize());
    std::vector<float> v((size_t)Sn*T); CU(cudaMemcpy(v.data(),sc,v.size()*4,cudaMemcpyDeviceToHost));
    const float* r=f32(S.get("score")); double mx=0; for(size_t i=0;i<v.size();++i) mx=fmax(mx,fabs((double)r[i]));
    Err e=compare(v,r,v.size(),mx); bool ok=e.max_rel<1e-3;
    printf("[index_score] S=%d H=%d d=%d T=%d  |sc|max=%.4f max_abs=%.5f max_rel=%.5f -> %s\n",Sn,H,d,T,mx,e.max_abs,e.max_rel,ok?"PASS":"FAIL");
    cudaFree(dq);cudaFree(dkv);cudaFree(dw);cudaFree(sc); return ok;
}

static bool gate_act_quant_fp4(const std::string& dir) {
    st::SafeTensors S(dir + "/unit_act_quant_fp4.safetensors");
    const auto& dm=S.get("dims"); int n=i32(dm,0), dim=i32(dm,1), block=i32(dm,2);
    void* dx=up(S.get("x"));
    act_quant_fp4sim((float*)dx, n, dim, block); CU(cudaDeviceSynchronize());
    std::vector<float> y((size_t)n*dim); CU(cudaMemcpy(y.data(),dx,y.size()*4,cudaMemcpyDeviceToHost));
    const float* yr=f32(S.get("y_ref")); double mx=0; for(size_t i=0;i<y.size();++i) mx=fmax(mx,fabs((double)yr[i]));
    Err e=compare(y,yr,y.size(),mx); bool ok=e.max_rel<1e-3;
    printf("[act_quant_fp4] n=%d dim=%d block=%d  max_abs=%.5f max_rel=%.5f -> %s\n",n,dim,block,e.max_abs,e.max_rel,ok?"PASS":"FAIL");
    cudaFree(dx); return ok;
}

static bool gate_indexer(const std::string& dir) {
    st::SafeTensors S(dir + "/unit_indexer.safetensors");
    const auto& dm=S.get("dims");
    int s=i32(dm,0),dim=i32(dm,1),q_lora=i32(dm,2),nh=i32(dm,3),ihd=i32(dm,4),rd=i32(dm,5),ratio=i32(dm,6),itk=i32(dm,7),off=i32(dm,8);
    int T=s/ratio;
    void *dx=up(S.get("x")),*dqr=up(S.get("qr")),*dwqb=up(S.get("wq_b")),*dwqbs=up(S.get("wq_b_s")),*dwp=up(S.get("weights_proj"));
    void *dcwkv=up(S.get("c_wkv")),*dcwg=up(S.get("c_wgate")),*dcape=up(S.get("c_ape")),*dcn=up(S.get("c_norm"));
    void *dqc=up(S.get("q_cos")),*dqs=up(S.get("q_sin")),*dcc=up(S.get("c_cos")),*dcs=up(S.get("c_sin"));
    float* isc; int* tk; CU(cudaMalloc(&isc,(size_t)s*T*4)); CU(cudaMalloc(&tk,(size_t)s*T*4));
    indexer_forward(isc, tk, (const float*)dx,(const float*)dqr,(const unsigned char*)dwqb,(const float*)dwqbs,(const float*)dwp,
                    (const float*)dcwkv,(const float*)dcwg,(const float*)dcape,(const float*)dcn,
                    (const float*)dqc,(const float*)dqs,(const float*)dcc,(const float*)dcs,
                    s,dim,q_lora,nh,ihd,rd,ratio,itk,off,1e-6f);
    CU(cudaDeviceSynchronize());
    std::vector<float> v((size_t)s*T); CU(cudaMemcpy(v.data(),isc,v.size()*4,cudaMemcpyDeviceToHost));
    const float* r=f32(S.get("index_score"));
    double mx=0,ma=0; for(size_t i=0;i<v.size();++i) if(fabs((double)r[i])<1e29) mx=fmax(mx,fabs((double)r[i]));
    for(size_t i=0;i<v.size();++i) if(fabs((double)r[i])<1e29) ma=fmax(ma,fabs((double)v[i]-r[i]));
    bool ok = ma/(mx+1e-9) < 1e-2;
    printf("[indexer] s=%d T=%d nh=%d idx_hd=%d  valid|sc|max=%.4f max_abs=%.5f rel=%.5f -> %s\n",s,T,nh,ihd,mx,ma,ma/(mx+1e-9),ok?"PASS":"FAIL");
    return ok;
}

static bool gate_yarn(const std::string& dir) {
    st::SafeTensors S(dir + "/unit_yarn.safetensors");
    const auto& dm=S.get("dims"); int seqlen=i32(dm,0), dim=i32(dm,1); int half=dim/2;
    std::vector<float> cy,sy,co,so;
    yarn::freqs(cy,sy,seqlen,dim,65536,160000.0,16,32,1);   // compressed
    yarn::freqs(co,so,seqlen,dim,0,10000.0,16,32,1);        // sliding
    auto cmp2=[&](const std::vector<float>& g,const char* name)->double{
        const float* r=f32(S.get(name)); double m=0; for(size_t i=0;i<g.size();++i) m=fmax(m,fabs((double)g[i]-r[i])); return m; };
    double e = fmax(fmax(cmp2(cy,"cos_yarn"),cmp2(sy,"sin_yarn")), fmax(cmp2(co,"cos_off"),cmp2(so,"sin_off")));
    bool ok = e < 1e-5;
    printf("[yarn] seqlen=%d dim=%d  max_abs=%.2e (yarn+off) -> %s\n",seqlen,dim,e,ok?"PASS":"FAIL");
    return ok;
}

int main(int argc, char** argv) {
    std::string dir = argc>1 ? argv[1] : "ref/goldens";
    bool ok = true;
    ok &= gate_yarn(dir);
    ok &= gate_indexer(dir);
    ok &= gate_compressor(dir);
    ok &= gate_compressor_overlap(dir);
    ok &= gate_compressor_full(dir, false);
    ok &= gate_compressor_full(dir, true);
    ok &= gate_hadamard(dir);
    ok &= gate_index_score(dir);
    ok &= gate_act_quant_fp4(dir);
    ok &= gate_fp8_gemm(dir);
    ok &= gate_hc(dir);
    ok &= gate_sparse_attn(dir);
    ok &= gate_rope(dir);
    ok &= gate_rmsnorm(dir);
    ok &= gate_act_quant(dir);
    ok &= gate_ogroup(dir);
    ok &= gate_fp4_gemm(dir);
    ok &= gate_router(dir);
    ok &= gate_moe(dir);
    ok &= gate_hc(dir, 0);
    printf("\nGate K (units): %s\n", ok?"PASS":"FAIL");
    return ok?0:1;
}
