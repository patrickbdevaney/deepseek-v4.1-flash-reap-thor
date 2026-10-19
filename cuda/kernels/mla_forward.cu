// mla_forward.cu — full MLA forward (pure-sliding layer, prefill). See mla_forward.h.
#include "mla_forward.h"
#include "fp8_block_gemm.h"
#include "mla_attn.h"
#include "deepseek_v4.h"
#include <vector>
#include <cmath>
#include <cstdio>

#define CU(x) do{cudaError_t e=(x); if(e){fprintf(stderr,"cuda %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)

using namespace dsv4;

// sliding-window top-k indices at prefill start_pos=0 (model.py get_window_topk_idxs, seqlen<=win branch
// generalized): query i attends keys [max(0,i-W+1) .. i], width = min(s, W), pad with -1.
static std::vector<int> window_idxs(int b, int s, int W, int& width) {
    width = s < W ? s : W;
    std::vector<int> idx((size_t)b * s * width, -1);
    for (int bi = 0; bi < b; ++bi)
        for (int i = 0; i < s; ++i) {
            int base = i - W + 1; if (base < 0) base = 0;
            for (int k = 0; k < width; ++k) {
                int v = base + k;
                idx[((size_t)bi * s + i) * width + k] = (v > i) ? -1 : v;
            }
        }
    return idx;
}

void mla_forward(float* out, const float* x, const MLAWeights& w, int b, int s, cudaStream_t stream) {
    const int bs = b * s;
    const int Kd = N_HEADS * HEAD_DIM;                 // 32768
    const int GKd = Kd / O_GROUPS;                     // 4096 per group (o reshaped [bs,G,GKd])
    const int OB = O_GROUPS * O_LORA;                  // 8192 (wo_b input)
    const float scale = 1.f / sqrtf((float)HEAD_DIM);

    // workspace
    uint8_t *xq, *qrq, *ogq; float *xs, *qrs, *ogs;
    float *qr, *q, *kv, *o, *og;
    CU(cudaMalloc(&xq, (size_t)bs * DIM));         CU(cudaMalloc(&xs, (size_t)bs * (DIM/128) * 4));
    CU(cudaMalloc(&qr, (size_t)bs * Q_LORA * 4));
    CU(cudaMalloc(&qrq,(size_t)bs * Q_LORA));      CU(cudaMalloc(&qrs,(size_t)bs * (Q_LORA/128) * 4));
    CU(cudaMalloc(&q,  (size_t)bs * Kd * 4));
    CU(cudaMalloc(&kv, (size_t)bs * HEAD_DIM * 4));
    CU(cudaMalloc(&o,  (size_t)bs * Kd * 4));
    CU(cudaMalloc(&og, (size_t)bs * OB * 4));
    CU(cudaMalloc(&ogq,(size_t)bs * OB));          CU(cudaMalloc(&ogs,(size_t)bs * (OB/128) * 4));

    // 1. quantize x once (shared by wq_a and wkv)
    act_quant_fp8(xq, xs, x, bs, DIM, 128, stream);

    // 2. qr = q_norm(wq_a(x))
    fp8_block_gemm(qr, xq, xs, w.wq_a, w.wq_a_s, bs, Q_LORA, DIM, stream);
    rmsnorm(qr, qr, w.q_norm, bs, Q_LORA, EPS, true, stream);

    // 3. q = wq_b(qr) ; per-head RMS ; RoPE(last 64 per head)
    act_quant_fp8(qrq, qrs, qr, bs, Q_LORA, 128, stream);
    fp8_block_gemm(q, qrq, qrs, w.wq_b, w.wq_b_s, bs, Kd, Q_LORA, stream);
    rmsnorm(q, q, nullptr, bs * N_HEADS, HEAD_DIM, EPS, false, stream);          // per-head, no weight
    rope_interleaved(q + NOPE_DIM, w.cosT, w.sinT, bs * N_HEADS, ROPE_DIM, false, HEAD_DIM, N_HEADS, stream);

    // 4. kv = kv_norm(wkv(x)) ; RoPE ; fp8-sim NoPE dims
    fp8_block_gemm(kv, xq, xs, w.wkv, w.wkv_s, bs, HEAD_DIM, DIM, stream);
    rmsnorm(kv, kv, w.kv_norm, bs, HEAD_DIM, EPS, true, stream);
    rope_interleaved(kv + NOPE_DIM, w.cosT, w.sinT, bs, ROPE_DIM, false, HEAD_DIM, 1, stream);
    act_quant_fp8sim(kv, bs, NOPE_DIM, 64, HEAD_DIM, stream);                    // NoPE dims only

    // 5. sparse attention over the sliding window
    int width; std::vector<int> hidx = window_idxs(b, s, WINDOW, width);
    int* didx; CU(cudaMalloc(&didx, hidx.size() * 4));
    CU(cudaMemcpyAsync(didx, hidx.data(), hidx.size() * 4, cudaMemcpyHostToDevice, stream));
    sparse_attn(o, q, kv, w.attn_sink, didx, b, s, N_HEADS, HEAD_DIM, s, width, scale, stream);

    // 6. de-rotate o, grouped o-LoRA, wo_b
    rope_interleaved(o + NOPE_DIM, w.cosT, w.sinT, bs * N_HEADS, ROPE_DIM, true, HEAD_DIM, N_HEADS, stream);
    if(w.wo_a_native) ogroup_gemm_fp8(og, o, w.wo_a_fp8, w.wo_a_sc, bs, O_GROUPS, O_LORA, GKd, stream);
    else              ogroup_gemm    (og, o, w.wo_a,                bs, O_GROUPS, O_LORA, GKd, stream);   // o [bs,G,GKd]
    act_quant_fp8(ogq, ogs, og, bs, OB, 128, stream);
    fp8_block_gemm(out, ogq, ogs, w.wo_b, w.wo_b_s, bs, DIM, OB, stream);

    CU(cudaStreamSynchronize(stream));
    cudaFree(xq);cudaFree(xs);cudaFree(qr);cudaFree(qrq);cudaFree(qrs);cudaFree(q);
    cudaFree(kv);cudaFree(o);cudaFree(og);cudaFree(ogq);cudaFree(ogs);cudaFree(didx);
}
