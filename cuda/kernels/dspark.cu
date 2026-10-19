// dspark.cu — DSpark MTP draft-head forward. See dspark.h.
#include "dspark.h"
#include "fp8_block_gemm.h"
#include "mla_attn.h"      // rmsnorm, act_quant_fp8
#include "hc.h"            // hc_head
#include "compressor.h"    // gemm_fp32 (lm_head)
#include "deepseek_v4.h"
#include <cstdio>
#include "dscratch.h"
#define CU(x) do{cudaError_t e=(x); if(e){fprintf(stderr,"cuda %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)

__global__ void k_embed_bf16(float* o, const __nv_bfloat16* emb, const int* ids, int s, int dim){
    size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x; if(i>=(size_t)s*dim) return; int t=i/dim, j=i%dim;
    o[i]=__bfloat162float(emb[(size_t)ids[t]*dim+j]);
}
// x'[t,c,j] = eproj[t,j] + hproj[t,c,j]   (eproj broadcast over hc)
__global__ void k_fuse(float* xp, const float* eproj, const float* hproj, int s, int hc, int dim){
    size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x; if(i>=(size_t)s*hc*dim) return;
    int t=i/(hc*dim), j=i%dim;
    xp[i]=eproj[(size_t)t*dim+j]+hproj[i];
}

void dspark_head_forward(float* logits, const float* x, const int* input_ids, const DSparkWeights& w,
                         int s, float eps, cudaStream_t stream){
    using namespace dsv4; const int d=w.dim, hc=w.hc;
    float *ebuf, *xh, *eproj, *hproj, *xp, *collapsed;
    uint8_t *eq,*hq; float *es,*hs;
    CU(rmalloc(&ebuf,(size_t)s*d*4)); CU(rmalloc(&xh,(size_t)s*hc*d*4));
    CU(rmalloc(&eproj,(size_t)s*d*4)); CU(rmalloc(&hproj,(size_t)s*hc*d*4));
    CU(rmalloc(&xp,(size_t)s*hc*d*4)); CU(rmalloc(&collapsed,(size_t)s*d*4));
    CU(rmalloc(&eq,(size_t)s*hc*d)); CU(rmalloc(&es,(size_t)s*hc*(d/128)*4));
    CU(rmalloc(&hq,(size_t)s*hc*d)); CU(rmalloc(&hs,(size_t)s*hc*(d/128)*4));

    // e = enorm(embed(ids)) ; xh = hnorm(x)
    k_embed_bf16<<<((size_t)s*d+255)/256,256,0,stream>>>(ebuf,w.embed,input_ids,s,d);
    rmsnorm(ebuf, ebuf, w.enorm, s, d, eps, true, stream);
    rmsnorm(xh, x, w.hnorm, s*hc, d, eps, true, stream);
    // eproj = e_proj(e) [s,d] ; hproj = h_proj(xh) [s*hc,d]  (fp8 act x fp8 weight)
    act_quant_fp8(eq, es, ebuf, s, d, 128, stream);   fp8_block_gemm(eproj, eq, es, w.e_proj, w.e_proj_s, s, d, d, stream);
    act_quant_fp8(hq, hs, xh, s*hc, d, 128, stream);  fp8_block_gemm(hproj, hq, hs, w.h_proj, w.h_proj_s, s*hc, d, d, stream);
    // x' = eproj[:,None,:] + hproj
    k_fuse<<<((size_t)s*hc*d+255)/256,256,0,stream>>>(xp, eproj, hproj, s, hc, d);
    // x' = Block(x') [pure-sliding, mtp weights]
    float* xb; CU(rmalloc(&xb,(size_t)s*hc*d*4));
    block_forward(xb, xp, input_ids, w.block, s, HC_SINKHORN_ITERS, eps, stream);
    // logits = hc_head(xb) -> norm -> lm_head
    hc_head(collapsed, xb, w.hc_head_fn, w.hc_head_scale, w.hc_head_base, s, hc, d, HC_EPS, stream);
    rmsnorm(collapsed, collapsed, w.norm, s, d, eps, true, stream);
    gemm_fp32(logits, collapsed, w.lm_head, s, w.vocab, d, stream);

    CU(rsync(stream));
    rfree(ebuf);rfree(xh);rfree(eproj);rfree(hproj);rfree(xp);rfree(collapsed);
    rfree(eq);rfree(es);rfree(hq);rfree(hs);rfree(xb);
}
