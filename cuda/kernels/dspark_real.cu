// dspark_real.cu — real DSpark head composable pieces. See dspark_real.h / DSPARK_HEAD_BUILD.md.
#include "dspark_real.h"
#include <cuda_bf16.h>
#include "fp8_block_gemm.h"
#include "mla_attn.h"      // rmsnorm, act_quant_fp8
#include "compressor.h"    // gemm_fp32
#include "hc.h"            // hc_head
#include <cstdio>
#include "dscratch.h"
#include <vector>
#define CU(x) do{cudaError_t e=(x); if(e){fprintf(stderr,"cuda %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)

// main_x = main_norm( main_proj(main_hidden) ) : fp8 gemm (3d->d) then RMSNorm.
void dspark_main_x(float* main_x, const float* main_hidden, const uint8_t* main_proj, const float* main_proj_s,
                   const float* main_norm, int s, int dim, float eps, cudaStream_t stream){
    const int K = 3 * dim;                                   // main_hidden is [s, 3d]
    uint8_t* xq; float* xs;
    CU(dkmalloc(&xq,(size_t)s*K)); CU(dkmalloc(&xs,(size_t)s*(K/128)*4));
    act_quant_fp8(xq, xs, main_hidden, s, K, 128, stream);
    fp8_block_gemm(main_x, xq, xs, main_proj, main_proj_s, s, dim, K, stream);   // [s, dim]
    rmsnorm(main_x, main_x, main_norm, s, dim, eps, true, stream);
    dksync(stream); dkfree(xq); dkfree(xs);
}

// gather markov_w1[token] -> markov_embed[n, rank]  (rows of the rank-256 embedding table)
// BF16 table variant — the markov embedding rows are read natively (Finding 26).
__global__ void k_gather_rows_bf16(float* out, const __nv_bfloat16* table, const int* ids, int n, int rank){
    size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x; if(i>=(size_t)n*rank) return; int t=i/rank, j=i%rank;
    out[i]=__bfloat162float(table[(size_t)ids[t]*rank + j]);
}
__global__ void k_gather_rows(float* out, const float* table, const int* ids, int n, int rank){
    size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x; if(i>=(size_t)n*rank) return;
    int t=i/rank, r=i%rank; out[i]=table[(size_t)ids[t]*rank + r];
}

// Markov head: embed = markov_w1[token] (rank); logits_bias = embed @ markov_w2^T  ([n,rank]x[vocab,rank]->[n,vocab]).
void dspark_markov(float* logits_bias, float* markov_embed, const int* token_ids,
                   const void* markov_w1, const void* markov_w2, int n, int vocab, int rank,
                   cudaStream_t stream){
    k_gather_rows_bf16<<<((size_t)n*rank+255)/256,256,0,stream>>>(markov_embed, (const __nv_bfloat16*)markov_w1, token_ids, n, rank);
    gemm_bf16w(logits_bias, markov_embed, markov_w2, n, vocab, rank, stream);      // C[n,vocab] = E[n,rank] @ W2[vocab,rank]^T
}

// ---- Piece 2: tap mean-pool over hc -> main_hidden[:, slot*d:] ----
__global__ void k_tap_pool(float* mh, const float* h, int s, int hc, int d, int slot, int n_taps){
    size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x; if(i>=(size_t)s*d) return; int t=i/d, j=i%d;
    float acc=0.f; for(int c=0;c<hc;++c) acc+=h[((size_t)t*hc+c)*d+j];
    mh[(size_t)t*(n_taps*d) + slot*d + j] = acc/(float)hc;                 // h.mean(dim=hc)
}
void dspark_tap_pool(float* main_hidden, const float* h, int s, int hc, int d, int slot, int n_taps, cudaStream_t stream){
    k_tap_pool<<<((size_t)s*d+255)/256,256,0,stream>>>(main_hidden,h,s,hc,d,slot,n_taps);
}

// ---- Piece 3: forward_head — block hidden -> greedy proposed block (with Markov bias) ----
__global__ void k_add_bias(float* logits, const float* bias, int n, int vocab){    // logits[t,:]+=bias[t,:]
    size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x; if(i<(size_t)n*vocab) logits[i]+=bias[i];
}
// Device-side greedy argmax over one logits row. One block per row; grid-stride + shared reduce.
//
// `mo` (optional) also returns the TOP-1 MINUS TOP-2 logit margin — the drafter confidence signal
// that adaptive verification needs (LOOP_LOG Finding 49 / EVICT, arXiv 2605.00342). The second
// maximum costs one more shared array and one more reduction over an already-loaded row; the row
// scan itself, which is the expensive part at vocab=129,280, is unchanged.
__global__ void k_argmax_row(int* __restrict__ out, const float* __restrict__ logits,
                             int vocab, int row_stride, const int* __restrict__ rows, int nrow,
                             float* __restrict__ mo){
    const int r = blockIdx.x; if (r >= nrow) return;
    const float* lg = logits + (size_t)rows[r] * row_stride;
    __shared__ float sv[256]; __shared__ int si[256]; __shared__ float s2[256];
    float bv = -1e30f, b2 = -1e30f; int bi = 0;
    for (int v = threadIdx.x; v < vocab; v += blockDim.x) {
        const float x = lg[v];
        if (x > bv) { b2 = bv; bv = x; bi = v; } else if (x > b2) { b2 = x; }
    }
    sv[threadIdx.x] = bv; si[threadIdx.x] = bi; s2[threadIdx.x] = b2; __syncthreads();
    for (int k = blockDim.x >> 1; k > 0; k >>= 1) {
        if (threadIdx.x < k) {
            // merging two (max, second) pairs: the loser's max becomes a second-place candidate
            const float ov = sv[threadIdx.x + k], o2 = s2[threadIdx.x + k];
            if (ov > sv[threadIdx.x]) {
                s2[threadIdx.x] = fmaxf(sv[threadIdx.x], o2);
                sv[threadIdx.x] = ov; si[threadIdx.x] = si[threadIdx.x + k];
            } else {
                s2[threadIdx.x] = fmaxf(s2[threadIdx.x], ov);
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) { out[r] = si[0]; if (mo) mo[r] = sv[0] - s2[0]; }
}
// out_ids[t,i+1] <- argmax; also copy first_ids into out_ids[t,0]
__global__ void k_seed_first(int* out_ids, const int* first_ids, int s, int block){
    int t = blockIdx.x*blockDim.x + threadIdx.x; if (t < s) out_ids[(size_t)t*(block+1)] = first_ids[t];
}
// gather out_ids[t,i] -> cur[t]; and the flat row index of logits[t,i]
__global__ void k_pick(int* cur, int* rows, const int* out_ids, int s, int block, int i){
    int t = blockIdx.x*blockDim.x + threadIdx.x; if (t >= s) return;
    cur[t]  = out_ids[(size_t)t*(block+1) + i];
    rows[t] = t*block + i;
}
__global__ void k_store(int* out_ids, const int* am, int s, int block, int i){
    int t = blockIdx.x*blockDim.x + threadIdx.x; if (t < s) out_ids[(size_t)t*(block+1) + i + 1] = am[t];
}

// `margins` (optional, device, [block][s]) receives the top1-top2 logit margin per proposal.
void dspark_forward_head(int* output_ids, const float* x_block, const int* first_ids,
                         const float* hc_head_fn, const float* hc_head_scale, const float* hc_head_base,
                         const float* norm, const void* lm_head, const void* markov_w1, const void* markov_w2,
                         int s, int block, int hc, int d, int vocab, int rank, float eps,
                         float* margins, cudaStream_t stream){
    const int N=s*block;
    float *collapsed,*logits,*bias,*membed; int *cur,*rows,*am;
    CU(dkmalloc(&collapsed,(size_t)N*d*4)); CU(dkmalloc(&logits,(size_t)N*vocab*4));
    CU(dkmalloc(&bias,(size_t)s*vocab*4)); CU(dkmalloc(&membed,(size_t)s*rank*4));
    CU(dkmalloc(&cur,(size_t)s*4)); CU(dkmalloc(&rows,(size_t)s*4)); CU(dkmalloc(&am,(size_t)s*4));
    // hc_head (hc 4->1) -> norm -> lm_head  over all N=s*block block-positions
    hc_head(collapsed, x_block, hc_head_fn, hc_head_scale, hc_head_base, N, hc, d, 1e-6f, stream);
    rmsnorm(collapsed, collapsed, norm, N, d, eps, true, stream);
    gemm_bf16w(logits, collapsed, lm_head, N, vocab, d, stream);            // [s,block,vocab]

    // DEVICE-SIDE greedy AR over the block (LOOP_LOG Finding 27). This loop used to run on the
    // HOST: per position it synced, copied the whole 129,280-float logits row D2H, and scanned it
    // on the CPU — 5 syncs, 2.6 MB of D2H and 5 host scans per draft, on the critical path of a
    // kernel chain that is otherwise fully asynchronous. The dependency (position i+1 needs the
    // argmax of position i) is real, but it is a DEVICE dependency; nothing needs to reach the host
    // until the block is finished.
    k_seed_first<<<(s+63)/64,64,0,stream>>>(output_ids, first_ids, s, block);
    for(int i=0;i<block;++i){
        k_pick<<<(s+63)/64,64,0,stream>>>(cur, rows, output_ids, s, block, i);
        dspark_markov(bias, membed, cur, markov_w1, markov_w2, s, vocab, rank, stream);
        for(int t=0;t<s;++t) k_add_bias<<<(vocab+255)/256,256,0,stream>>>(logits+((size_t)t*block+i)*vocab, bias+(size_t)t*vocab, 1, vocab);
        k_argmax_row<<<s,256,0,stream>>>(am, logits, vocab, vocab, rows, s, margins? margins+(size_t)i*s : nullptr);
        k_store<<<(s+63)/64,64,0,stream>>>(output_ids, am, s, block, i);
    }
    dkfree(collapsed);dkfree(logits);dkfree(bias);dkfree(membed);dkfree(cur);dkfree(rows);dkfree(am);
}
