// v41_draft.cu — the DSpark drafter (include/v41_draft.h). Follows inference/model.py DSparkBlock,
// DSparkAttention and forward_spec line by line; the block structure is the decoder Block of
// v41_model.cu with the attention swapped and no compressor, indexer or Engram.
#include "v41_draft.h"
#include "v41_dev.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include "third_party/json.hpp"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) throw std::runtime_error(std::string(#x) + ": " + cudaGetErrorString(e_)); } while (0)

using namespace v41;
using namespace v41d;

namespace v41m {

static constexpr int NB = DSPARK_BLOCK;      // 5 block slots

// =========================================================================== kernels
// taps[r][k][:] = bf16( mean over the 4 hc copies of h[r] )   (h.mean(dim=2) of a bf16 stream)
__global__ static void k_tap_mean(float* taps, int k, const float* h, int M) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i >= (size_t)M * DIM) return;
    const int m = (int)(i / DIM), d = (int)(i % DIM);
    const float* hr = h + (size_t)m * HC_DIM + d;
    const float s = ((hr[0] + hr[DIM]) + hr[2 * DIM]) + hr[3 * DIM];
    taps[(size_t)m * 3 * DIM + (size_t)k * DIM + d] = bf16r(s * 0.25f);
}

// draft inputs from the device anchor: ids = [tok, noise x4], positions p+1 .. p+5
__global__ static void k_draft_inputs(int* ids, int* pos, int* out0, const int* anchor, const int* tok) {
    const int i = threadIdx.x;
    if (i >= NB) return;
    ids[i] = i == 0 ? *tok : DSPARK_NOISE_TID;
    pos[i] = *anchor + 1 + i;
    if (i == 0) *out0 = *tok;
}

// DSparkAttention: row m (block slot, position p+1+m) attends the window positions max(0,p-127)..p
// (ring slot pos % ring) and all NB block-slot kvs, with the sink. One CTA per (row, 8 heads), warp =
// head; two passes so p = exp(s - max) uses the final max before its bf16 cast (as k_sparse_attn).
static constexpr int DA_TILE = 8;
static constexpr int DA_MAXE = WINDOW + NB;
__device__ __forceinline__ void da_stage(float (*kv)[HEAD_DIM], int e0, int ne, int nwin, int p, const float* ring, int rs,
                                         const float* blk, bool* ok) {
    for (int i = threadIdx.x; i < DA_TILE * HEAD_DIM; i += blockDim.x) {
        const int r = i / HEAD_DIM, j = i % HEAD_DIM, e = e0 + r;
        float v = 0.f; bool valid = false;
        if (e < ne) {
            if (e < nwin) { const int t = p - nwin + 1 + e; v = ring[(size_t)(t % rs) * HEAD_DIM + j]; }
            else v = blk[(size_t)(e - nwin) * HEAD_DIM + j];
            valid = true;
        }
        kv[r][j] = v;
        if (j == 0) ok[r] = valid;
    }
}
__global__ void __launch_bounds__(256) k_dspark_attn(float* o, const float* q, const float* ring, int rs, const float* blk,
                                                     const float* sink, const int* anchor, float scale) {
    const int m = blockIdx.x, h = blockIdx.y * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    const int p = *anchor;
    const int nwin = min(WINDOW, p + 1), ne = nwin + NB;
    __shared__ float kv[DA_TILE][HEAD_DIM];
    __shared__ float sc[8][DA_MAXE];
    __shared__ bool ok[DA_TILE];
    float qv[16];
    const float* qh = q + ((size_t)m * N_HEADS + h) * HEAD_DIM;
    #pragma unroll
    for (int i = 0; i < 16; ++i) qv[i] = qh[lane + 32 * i];
    float mx = -INFINITY;
    for (int e0 = 0; e0 < ne; e0 += DA_TILE) {
        __syncthreads();
        da_stage(kv, e0, ne, nwin, p, ring, rs, blk, ok);
        __syncthreads();
        for (int r = 0; r < DA_TILE && e0 + r < ne; ++r) {
            float d = 0.f;
            #pragma unroll
            for (int i = 0; i < 16; ++i) d = fmaf(qv[i], kv[r][lane + 32 * i], d);
            d = warp_sum(d) * scale;
            const float s = ok[r] ? d : -INFINITY;
            if (lane == 0) sc[threadIdx.x >> 5][e0 + r] = s;
            mx = fmaxf(mx, s);
        }
    }
    mx = fmaxf(mx, -1e30f);
    float acc[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) acc[i] = 0.f;
    float den = 0.f;
    for (int e0 = 0; e0 < ne; e0 += DA_TILE) {
        __syncthreads();
        da_stage(kv, e0, ne, nwin, p, ring, rs, blk, ok);
        __syncthreads();
        for (int r = 0; r < DA_TILE && e0 + r < ne; ++r) {
            const float pr = expf(sc[threadIdx.x >> 5][e0 + r] - mx);
            den += pr;
            const float pb = bf16r(pr);
            #pragma unroll
            for (int i = 0; i < 16; ++i) acc[i] = fmaf(pb, kv[r][lane + 32 * i], acc[i]);
        }
    }
    den += expf(sink[h] - mx);
    float* oh = o + ((size_t)m * N_HEADS + h) * HEAD_DIM;
    #pragma unroll
    for (int i = 0; i < 16; ++i) oh[lane + 32 * i] = bf16r(acc[i] / den);
}

// Markov head step i: e = embed[out[i]] (bf16, rank 256) -> fp32
__global__ static void k_mk_gather(float* e, const uint16_t* emb, const int* out, int i) {
    const int k = threadIdx.x;
    if (k < DSPARK_MARKOV_RANK) e[k] = bf16f(emb[(size_t)out[i] * DSPARK_MARKOV_RANK + k]);
}
// logits[i] += bias (fp32, in place, as logits[:, i].add_(logits_bias))
__global__ static void k_add(float* y, const float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) y[i] += b[i];
}

// =========================================================================== load
Drafter::~Drafter() { for (void* p : allocs) cudaFree(p); }

void load_drafter(Drafter& D, const std::string& ckpt, const std::string& head_dir, bool hw) {
    std::string qtype;
    nlohmann::json hj;
    if (!head_dir.empty()) {
        std::ifstream f(head_dir + "/head.json");
        if (!f) throw std::runtime_error(head_dir + ": no head.json (an s11 head dir is expected)");
        std::stringstream ss; ss << f.rdbuf();
        hj = nlohmann::json::parse(ss.str());
        qtype = hj.value("qtype", std::string());
        D.head.reset(new Store(head_dir, [](const std::string& n) { return n.compare(0, 4, "mtp.") == 0; }));
    }
    const Store* H = D.head.get();
    // routed experts come wholly from one place per block: the head dir when it carries any (stacked
    // keep-set or per-expert), else the checkpoint. Never load the checkpoint's 128 experts under an
    // in-model head: that is the memory the in-model arm exists to save.
    std::vector<std::string> head_ex;
    for (int i = 0; i < N_MTP; ++i) {
        const std::string ep = "mtp." + std::to_string(i) + ".ffn.experts.";
        if (H && (H->has(ep + "w1.qweight") || H->has(ep + "w1.weight") || H->has(ep + "0.w1.weight"))) head_ex.push_back(ep);
    }
    D.base.reset(new Store(ckpt, [H, head_ex](const std::string& n) {
        if (n.compare(0, 4, "mtp.") != 0 || (H && H->has(n))) return false;
        for (const auto& ep : head_ex) if (n.compare(0, ep.size(), ep) == 0) return false;
        return true;
    }));
    auto has = [&](const std::string& n) { return (H && H->has(n)) || D.base->has(n); };
    auto get = [&](const std::string& n) -> const DevT& { return H && H->has(n) ? H->get(n) : D.base->get(n); };
    auto u8 = [&](const std::string& n) { return (const uint8_t*)get(n).p; };
    auto b16 = [&](const std::string& n) { return (const uint16_t*)get(n).p; };
    auto f32 = [&](const std::string& n) { return (const float*)get(n).p; };
    // one Store view for bind_experts: stacked experts live in the head (in-model arm), per-expert
    // ones in the base (separate arm); never a mix within one block
    for (int i = 0; i < N_MTP; ++i) {
        DBlock& w = D.b[i];
        const std::string p = "mtp." + std::to_string(i) + ".";
        w.attn_sink = f32(p + "attn.attn_sink");
        w.attn_norm = b16(p + "attn_norm.weight"); w.ffn_norm = b16(p + "ffn_norm.weight");
        w.q_norm = b16(p + "attn.q_norm.weight"); w.kv_norm = b16(p + "attn.kv_norm.weight");
        w.wq_a = u8(p + "attn.wq_a.weight"); w.wq_a_s = u8(p + "attn.wq_a.scale");
        w.wq_b = u8(p + "attn.wq_b.weight"); w.wq_b_s = u8(p + "attn.wq_b.scale");
        w.wkv = u8(p + "attn.wkv.weight"); w.wkv_s = u8(p + "attn.wkv.scale");
        w.wo_a = u8(p + "attn.wo_a.weight"); w.wo_a_s = u8(p + "attn.wo_a.scale");
        w.wo_b = u8(p + "attn.wo_b.weight"); w.wo_b_s = u8(p + "attn.wo_b.scale");
        if (get(p + "attn.wo_a.weight").dtype != "F8_E4M3") throw std::runtime_error(p + "attn.wo_a is not FP8");
        w.hc_attn_fn = f32(p + "hc_attn_fn"); w.hc_attn_base = f32(p + "hc_attn_base"); w.hc_attn_scale = f32(p + "hc_attn_scale");
        w.hc_ffn_fn = f32(p + "hc_ffn_fn"); w.hc_ffn_base = f32(p + "hc_ffn_base"); w.hc_ffn_scale = f32(p + "hc_ffn_scale");
        w.gate_w = b16(p + "ffn.gate.weight"); w.gate_bias = f32(p + "ffn.gate.bias");
        w.sh_w1 = u8(p + "ffn.shared_experts.w1.weight"); w.sh_w1s = u8(p + "ffn.shared_experts.w1.scale");
        w.sh_w2 = u8(p + "ffn.shared_experts.w2.weight"); w.sh_w2s = u8(p + "ffn.shared_experts.w2.scale");
        w.sh_w3 = u8(p + "ffn.shared_experts.w3.weight"); w.sh_w3s = u8(p + "ffn.shared_experts.w3.scale");
        const std::string ep = p + "ffn.experts.";
        const bool in_head = std::find(head_ex.begin(), head_ex.end(), ep) != head_ex.end();
        std::function<std::string(const std::string&)> qtype_of;
        if (in_head && H->has(ep + "w1.qweight")) {               // stacked K-quant keep-set (s11 inmodel)
            if (qtype.empty()) throw std::runtime_error(head_dir + ": stacked head experts but head.json has no qtype");
            qtype_of = [qtype](const std::string&) { return qtype; };
        } else if (!in_head && !D.base->thor_json().empty()) {    // a thor build's own mtp files
            auto j = std::make_shared<nlohmann::json>(nlohmann::json::parse(D.base->thor_json()));
            qtype_of = [j](const std::string& n) {
                const auto& T = (*j)["tensors"];
                return T.contains(n) ? T[n].value("qtype", std::string()) : std::string();
            };
        }
        const int E = bind_experts(in_head ? *H : *D.base, ep, qtype_of, hw, MOE_INTER, w.ex);
        if ((int)get(p + "ffn.gate.weight").shape[0] != E)
            throw std::runtime_error(p + "router rows != expert count (the in-model keep-set must slice the router)");
        if (i == 0) D.n_routed = E;
        else if (E != D.n_routed) throw std::runtime_error(p + ": blocks disagree on the expert count");
    }
    D.main_proj = u8("mtp.0.main_proj.weight"); D.main_proj_s = u8("mtp.0.main_proj.scale");
    D.main_norm = b16("mtp.0.main_norm.weight");
    const std::string last = "mtp." + std::to_string(N_MTP - 1) + ".";
    D.norm = b16(last + "norm.weight");
    D.mk_embed = b16(last + "markov_head.embed.weight"); D.mk_head = b16(last + "markov_head.head.weight");
    if (get(last + "markov_head.embed.weight").shape[1] != DSPARK_MARKOV_RANK)
        throw std::runtime_error("markov_head rank != DSPARK_MARKOV_RANK");
    if (!has("mtp.0.main_proj.weight")) throw std::runtime_error("no mtp.0.main_proj");
    D.gib = (H ? H->gib() : 0.0) + D.base->gib();
    char buf[512];
    snprintf(buf, sizeof buf, "%s (%s, %d experts/block, %s, %.2f GiB)", head_dir.empty() ? (ckpt + " mtp.*").c_str() : head_dir.c_str(),
             hj.value("arm", std::string(head_dir.empty() ? "native" : "?")).c_str(), D.n_routed,
             qtype.empty() ? "MXFP4" : qtype.c_str(), D.gib);
    D.desc = buf;
}

template <class T> static T* dal(Drafter& D, size_t n) {
    void* p = nullptr; CK(cudaMalloc(&p, n * sizeof(T) + 256)); CK(cudaMemset(p, 0, n * sizeof(T) + 256));
    D.allocs.push_back(p); D.bytes += n * sizeof(T); return (T*)p;
}
void alloc_drafter(Drafter& D, int max_rows) {
    D.max_rows = max_rows;
    for (int i = 0; i < N_MTP; ++i) D.ring[i] = dal<float>(D, (size_t)Drafter::RING * HEAD_DIM);
    const int R = std::max(max_rows, NB);
    D.taps = dal<float>(D, (size_t)max_rows * 3 * DIM);
    D.mainx = dal<float>(D, (size_t)max_rows * DIM);
    D.kvt = dal<float>(D, (size_t)max_rows * HEAD_DIM);
    const size_t xk = std::max((size_t)R * 3 * DIM, (size_t)NB * N_HEADS * HEAD_DIM);   // taps | o
    D.xq = dal<uint8_t>(D, xk); D.xs = dal<float>(D, xk / 32);
    D.d_ids = dal<int>(D, NB); D.d_pos = dal<int>(D, NB); D.d_anchor = dal<int>(D, 1); D.d_tok = dal<int>(D, 1);
    D.d_out = dal<int>(D, NB + 1);
    D.h = dal<float>(D, (size_t)NB * HC_DIM); D.h2 = dal<float>(D, (size_t)NB * HC_DIM);
    D.pre = dal<float>(D, NB * 4); D.post = dal<float>(D, NB * 4); D.comb = dal<float>(D, NB * 16); D.pre_next = dal<float>(D, NB * 4);
    D.x = dal<float>(D, (size_t)NB * DIM); D.xa = dal<float>(D, (size_t)NB * DIM); D.y = dal<float>(D, (size_t)NB * DIM);
    D.qa = dal<float>(D, (size_t)NB * Q_LORA); D.qr = dal<float>(D, (size_t)NB * Q_LORA);
    D.q = dal<float>(D, (size_t)NB * N_HEADS * HEAD_DIM); D.o = dal<float>(D, (size_t)NB * N_HEADS * HEAD_DIM);
    D.kv = dal<float>(D, (size_t)NB * HEAD_DIM); D.og = dal<float>(D, (size_t)NB * O_GROUPS * O_LORA);
    D.rw = dal<float>(D, (size_t)NB * DSPARK_N_ACT); D.ridx = dal<int>(D, (size_t)NB * DSPARK_N_ACT);
    D.moe_scratch = dal<uint8_t>(D, v41k::moe_scratch_bytes(NB, DSPARK_N_ACT));
    D.logits = dal<float>(D, (size_t)NB * VOCAB);
    D.mk_e = dal<float>(D, DSPARK_MARKOV_RANK); D.mk_bias = dal<float>(D, VOCAB);
    v41k::hc_reserve(R);
    CK(cudaDeviceSynchronize());
}
void reset_drafter(Drafter& D, cudaStream_t st) {
    for (int i = 0; i < N_MTP; ++i) cudaMemsetAsync(D.ring[i], 0, (size_t)Drafter::RING * HEAD_DIM * 4, st);
}

// =========================================================================== taps -> window
// main_kv for rows [a, a+n) of the main call (positions d_pos[a..a+n)): main_x, then per block
// kv_norm(wkv(main_x)), RoPE at the row's position (base theta, no YaRN), fp8 QAT-sim, ring store.
static void write_window(Drafter& D, const int* d_pos, int a, int n, cudaStream_t st) {
    v41k::act_quant_fp8(D.xq, D.xs, D.taps + (size_t)a * 3 * DIM, n, 3 * DIM, st);
    v41k::fp8_linear(D.mainx + (size_t)a * DIM, D.xq, D.xs, D.main_proj, D.main_proj_s, n, DIM, 3 * DIM, true, st);
    v41k::rmsnorm(D.mainx + (size_t)a * DIM, D.mainx + (size_t)a * DIM, D.main_norm, n, DIM, NORM_EPS, st);
    v41k::act_quant_fp8(D.xq, D.xs, D.mainx + (size_t)a * DIM, n, DIM, st);
    for (int i = 0; i < N_MTP; ++i) {
        const DBlock& w = D.b[i];
        v41k::fp8_linear(D.kvt, D.xq, D.xs, w.wkv, w.wkv_s, n, HEAD_DIM, DIM, true, st);
        v41k::rmsnorm(D.kvt, D.kvt, w.kv_norm, n, HEAD_DIM, NORM_EPS, st);
        v41k::rope(D.kvt, n, 1, HEAD_DIM, NOPE_DIM, d_pos + a, 0, false, false, st);
        v41k::fp8_sim(D.kvt, n, HEAD_DIM, st);
        v41k::ring_store(D.ring[i], Drafter::RING, D.kvt, d_pos + a, n, st);
    }
}

LayerHook tap_hook(Drafter& D, State& s, int M_total) {
    if (M_total > D.max_rows) throw std::runtime_error("tap_hook: rows > drafter max_rows");
    const int* d_pos = s.d_pos;
    return [&D, d_pos, M_total](int L, const float* h, const float*, int M, cudaStream_t st) {
        // the stream ENTERING tap layer t is the output of layer t-1
        int k = -1;
        for (int j = 0; j < 3; ++j) if (L + 1 == DSPARK_TAP_LAYERS[j]) k = j;
        if (k < 0) return;
        const int a = M_total - M;                       // the hook's rows are the LAST M of the call
        const size_t n = (size_t)M * DIM;
        k_tap_mean<<<(unsigned)((n + 255) / 256), 256, 0, st>>>(D.taps + (size_t)a * 3 * DIM, k, h, M);
        if (k == 2) write_window(D, d_pos, a, M, st);
    };
}

// =========================================================================== draft
// DSparkAttention up to the attention itself: q (block positions p+1..p+5) and the 5 block-slot kvs
static void dspark_qkv(const DBlock& w, Drafter& D, const float* x, cudaStream_t st) {
    const int M = NB;
    v41k::act_quant_fp8(D.xq, D.xs, x, M, DIM, st);
    v41k::fp8_linear(D.qa, D.xq, D.xs, w.wq_a, w.wq_a_s, M, Q_LORA, DIM, true, st);
    v41k::rmsnorm(D.qr, D.qa, w.q_norm, M, Q_LORA, NORM_EPS, st);
    v41k::fp8_linear(D.kv, D.xq, D.xs, w.wkv, w.wkv_s, M, HEAD_DIM, DIM, true, st);
    v41k::rmsnorm(D.kv, D.kv, w.kv_norm, M, HEAD_DIM, NORM_EPS, st);
    v41k::rope(D.kv, M, 1, HEAD_DIM, NOPE_DIM, D.d_pos, 0, false, false, st);
    v41k::fp8_sim(D.kv, M, HEAD_DIM, st);
    v41k::act_quant_fp8(D.xq, D.xs, D.qr, M, Q_LORA, st);
    v41k::fp8_linear(D.q, D.xq, D.xs, w.wq_b, w.wq_b_s, M, N_HEADS * HEAD_DIM, Q_LORA, true, st);
    v41k::rope(D.q, M * N_HEADS, N_HEADS, HEAD_DIM, NOPE_DIM, D.d_pos, 0, false, false, st);
}

static void dblock(const DBlock& w, int bi, Drafter& D, cudaStream_t st) {
    const int M = NB;
    // attention, collapsing with the previous block's FFN mix (identity for block 0)
    v41k::hc_mixes(D.pre_next, D.post, D.comb, D.h, w.hc_attn_fn, w.hc_attn_scale, w.hc_attn_base, M, st);
    v41k::hc_collapse(D.x, D.h, D.pre, M, st);
    v41k::rmsnorm(D.x, D.x, w.attn_norm, M, DIM, NORM_EPS, st);
    dspark_qkv(w, D, D.x, st);
    k_dspark_attn<<<dim3(M, N_HEADS / 8), 256, 0, st>>>(D.o, D.q, D.ring[bi], Drafter::RING, D.kv, w.attn_sink, D.d_anchor,
                                                        1.0f / std::sqrt((float)HEAD_DIM));
    v41k::rope(D.o, M * N_HEADS, N_HEADS, HEAD_DIM, NOPE_DIM, D.d_pos, 0, true, false, st);
    v41k::act_quant_fp8(D.xq, D.xs, D.o, M, N_HEADS * HEAD_DIM, st);
    v41k::fp8_linear(D.og, D.xq, D.xs, w.wo_a, w.wo_a_s, M, O_GROUPS * O_LORA, O_GROUP_IN, true, st, O_GROUPS);
    v41k::act_quant_fp8(D.xq, D.xs, D.og, M, O_GROUPS * O_LORA, st);
    v41k::fp8_linear(D.xa, D.xq, D.xs, w.wo_b, w.wo_b_s, M, DIM, O_GROUPS * O_LORA, true, st);
    v41k::hc_post(D.h2, D.xa, D.h, D.post, D.comb, M, st);
    // FFN: own mixes from h2, collapses with the attention's pre
    v41k::hc_mixes(D.pre, D.post, D.comb, D.h2, w.hc_ffn_fn, w.hc_ffn_scale, w.hc_ffn_base, M, st);
    v41k::hc_collapse(D.x, D.h2, D.pre_next, M, st);
    v41k::rmsnorm(D.x, D.x, w.ffn_norm, M, DIM, NORM_EPS, st);
    v41k::router(D.rw, D.ridx, D.x, w.gate_w, w.gate_bias, nullptr, nullptr, M, w.ex.n_routed, DSPARK_N_ACT, ROUTE_SCALE, st);
    v41k::act_quant_fp8(D.xq, D.xs, D.x, M, DIM, st);
    cudaMemsetAsync(D.y, 0, (size_t)M * DIM * 4, st);
    v41k::moe_routed(D.y, D.xq, D.xs, D.rw, D.ridx, w.ex.tab, M, DSPARK_N_ACT, D.moe_scratch, st);
    v41k::moe_shared_add(D.y, D.xq, D.xs, w.sh_w1, w.sh_w1s, w.sh_w3, w.sh_w3s, w.sh_w2, w.sh_w2s, M, D.moe_scratch, st);
    v41k::hc_post(D.h, D.y, D.h2, D.post, D.comb, M, st);
}

void draft_body(const Model& m, Drafter& D, cudaStream_t st) {
    k_draft_inputs<<<1, 32, 0, st>>>(D.d_ids, D.d_pos, D.d_out, D.d_anchor, D.d_tok);
    v41k::embed_expand(D.h, m.embed, D.d_ids, NB, st);
    v41k::hc_identity_pre(D.pre, NB, st);
    for (int i = 0; i < N_MTP; ++i) dblock(D.b[i], i, D, st);
    // forward_head: hc_pre with the last FFN's mix, mtp.2.norm, the main head for every slot
    v41k::hc_collapse(D.x, D.h, D.pre, NB, st);
    v41k::rmsnorm(D.x, D.x, D.norm, NB, DIM, NORM_EPS, st);
    v41k::bf16_linear(D.logits, D.x, m.head, NB, VOCAB, DIM, false, st);
    for (int i = 0; i < NB; ++i) {                       // the Markov chain, greedy
        k_mk_gather<<<1, DSPARK_MARKOV_RANK, 0, st>>>(D.mk_e, D.mk_embed, D.d_out, i);
        v41k::bf16_linear(D.mk_bias, D.mk_e, D.mk_head, 1, VOCAB, DSPARK_MARKOV_RANK, false, st);
        k_add<<<(VOCAB + 255) / 256, 256, 0, st>>>(D.logits + (size_t)i * VOCAB, D.mk_bias, VOCAB);
        v41k::argmax(D.d_out + i + 1, D.logits + (size_t)i * VOCAB, VOCAB, st);
    }
}

void draft(const Model& m, Drafter& D, int anchor_pos, int tok, cudaStream_t st) {
    CK(cudaMemcpyAsync(D.d_anchor, &anchor_pos, 4, cudaMemcpyHostToDevice, st));
    CK(cudaMemcpyAsync(D.d_tok, &tok, 4, cudaMemcpyHostToDevice, st));
    draft_body(m, D, st);
}

} // namespace v41m
