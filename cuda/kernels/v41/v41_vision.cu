// v41_vision.cu — the V4.1 vision tower (include/v41_vision.h): CUTLASS 2.x bf16 tensor-core GEMMs (mma.sync,
// Sm80 kernels run unchanged on sm_110a) with a broadcast-bias epilogue, the CUTLASS example 41 fused attention
// (xformers memory-efficient forward, bf16, head dim 64, non-causal), and small elementwise kernels between them.
#include "v41_vision.h"
#include "v41_model.h"
#include "v41_dev.cuh"
#include <cuda_bf16.h>
#include <stdexcept>
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "kernel_forward.h"                     // ~/cutlass/examples/41_fused_multi_head_attention

namespace v41v {
using namespace v41d;

#define VCK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) throw std::runtime_error(std::string("vision: ") + #x + ": " + cudaGetErrorString(e_)); } while (0)

using bf16 = cutlass::bfloat16_t;
using Gemm = cutlass::gemm::device::Gemm<
    bf16, cutlass::layout::RowMajor,                 // A: activations [M, K]
    bf16, cutlass::layout::ColumnMajor,              // B: K x N column-major == W [N, K] row-major
    float, cutlass::layout::RowMajor,                // C (bias, ldc = 0: broadcast) and D [M, N]
    float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>, cutlass::gemm::GemmShape<64, 64, 32>, cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<float, 4, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 4>;

// D[M, N] = A[M, K] W[N, K]^T (+ bias[N])
static void gemm(float* D, const uint16_t* A, const uint16_t* W, const float* bias, int M, int N, int K, cudaStream_t st) {
    Gemm op;
    Gemm::Arguments args({M, N, K}, {(const bf16*)A, K}, {(const bf16*)W, K}, {bias ? bias : D, bias ? 0 : N}, {D, N},
                         {1.f, bias ? 1.f : 0.f});
    if (op.can_implement(args) != cutlass::Status::kSuccess) throw std::runtime_error("vision gemm: shape not supported");
    if (op(args, nullptr, st) != cutlass::Status::kSuccess) throw std::runtime_error("vision gemm: launch failed");
}

using Attn = AttentionKernel<bf16, cutlass::arch::Sm80, true, 64, 64, 64, false, false>;
// o[P, 16 x 64] = softmax(q k^T / 8) v per head; q, k, v interleaved in qkv [P, 3072] (q | k | v, head-major)
static void attention(uint16_t* o, const uint16_t* qkv, int P, cudaStream_t st) {
    Attn::Params p;
    p.query_ptr = (bf16*)qkv; p.key_ptr = (bf16*)qkv + VDIM; p.value_ptr = (bf16*)qkv + 2 * VDIM;
    p.output_ptr = (bf16*)o; p.output_accum_ptr = nullptr; p.logsumexp_ptr = nullptr;
    p.scale = 0.125f;
    p.num_heads = VHEADS; p.num_batches = 1; p.head_dim = VHD; p.head_dim_value = VHD;
    p.num_queries = P; p.num_keys = P; p.num_keys_absolute = P;
    p.q_strideH = p.k_strideH = p.v_strideH = VHD;
    p.q_strideM = p.k_strideM = p.v_strideM = 3 * VDIM;
    p.q_strideB = p.k_strideB = p.v_strideB = (int64_t)3 * VDIM * P;
    p.o_strideM = VDIM;
    if (!Attn::check_supported(p)) throw std::runtime_error("vision attention: unsupported shape");
    constexpr auto kf = attention_kernel_batched_impl<Attn>;
    const int smem = (int)sizeof(typename Attn::SharedStorage);
    static bool attr = false;
    if (!attr && smem > 0xc000) { VCK(cudaFuncSetAttribute(kf, cudaFuncAttributeMaxDynamicSharedMemorySize, smem)); attr = true; }
    kf<<<p.getBlocksGrid(), p.getThreadsGrid(), smem, st>>>(p);
}

__device__ __forceinline__ uint16_t f2bf(float v) { return __bfloat16_as_ushort(__float2bfloat16_rn(v)); }

// y = bf16(w * (x * rsqrt(mean(x^2) + eps))), one 256-thread block per row of 1024
__global__ void k_vnorm(uint16_t* y, const float* x, const uint16_t* w) {
    const float* xr = x + (size_t)blockIdx.x * VDIM;
    float v[4], ss = 0.f;
    #pragma unroll
    for (int j = 0; j < 4; ++j) { v[j] = xr[threadIdx.x + 256 * j]; ss += v[j] * v[j]; }
    __shared__ float red[8];
    ss = warp_sum(ss);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
    __syncthreads();
    float tot = 0.f;
    #pragma unroll
    for (int i = 0; i < 8; ++i) tot += red[i];
    const float r = rsqrtf(tot / VDIM + 1e-6f);
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int i = threadIdx.x + 256 * j;
        y[(size_t)blockIdx.x * VDIM + i] = f2bf(bf16f(w[i]) * (v[j] * r));
    }
}
// 2D RoPE on q and k (rotate-half over the head's 64: x1 = [0, 32), x2 = [32, 64)), v copied; bf16 out.
// Frequency i in [0, 32): position = row / n_w for i < 16, row % n_w above; inv = 1 / 10000^(2 (i % 16) / 32).
__global__ void k_vrope(uint16_t* out, const float* in, int n_w) {
    const int row = blockIdx.x;
    const float* ir = in + (size_t)row * 3 * VDIM;
    uint16_t* orow = out + (size_t)row * 3 * VDIM;
    const float hp = (float)(row / n_w), wp = (float)(row % n_w);
    for (int c = threadIdx.x; c < 3 * VDIM; c += blockDim.x) {
        if (c >= 2 * VDIM) { orow[c] = f2bf(ir[c]); continue; }
        const int d = c & 63, base = c - d, i = d & 31;
        const float inv = 1.f / powf(10000.f, (float)(2 * (i & 15)) / 32.f);
        const float f = (i < 16 ? hp : wp) * inv;
        const float cs = cosf(f), sn = sinf(f);
        const float x1 = ir[base + i], x2 = ir[base + i + 32];
        orow[c] = f2bf(d < 32 ? x1 * cs - x2 * sn : x2 * cs + x1 * sn);
    }
}
__global__ void k_vswiglu(uint16_t* out, const float* in, size_t n) {
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i >= n) return;
    const size_t r = i / VINTER, k = i % VINTER;
    const float g = in[r * 2 * VINTER + k], u = in[r * 2 * VINTER + VINTER + k];
    out[i] = f2bf(g / (1.f + expf(-g)) * u);
}
__global__ void k_vadd(float* x, const float* t, size_t n) {
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i < n) x[i] += t[i];
}
__global__ void k_vgelu(uint16_t* out, const float* in, size_t n) {
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i >= n) return;
    const float v = in[i];
    out[i] = f2bf(0.5f * v * (1.f + erff(v * 0.70710678118654752f)));
}
// patches [P, 588] -> [P, 592] (zero tail)
__global__ void k_vpad(uint16_t* out, const uint16_t* in, int P) {
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i >= (size_t)P * PATCH_IN_PAD) return;
    const size_t r = i / PATCH_IN_PAD, c = i % PATCH_IN_PAD;
    out[i] = c < PATCH_IN ? in[r * PATCH_IN + c] : 0;
}
// F.unfold(kernel 3, stride 3) of the zero-padded [1024, n_h, n_w] grid, transposed: out[l][c * 9 + ky * 3 + kx]
__global__ void k_vunfold(uint16_t* out, const uint16_t* x, int n_h, int n_w, int lw, size_t n) {
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i >= n) return;
    const int l = (int)(i / AL_IN), f = (int)(i % AL_IN);
    const int c = f / 9, ky = (f % 9) / 3, kx = f % 3;
    const int y = (l / lw) * DOWN + ky, xx = (l % lw) * DOWN + kx;
    out[i] = (y < n_h && xx < n_w) ? x[((size_t)y * n_w + xx) * VDIM + c] : 0;
}
__global__ void k_vbf2f(float* o, const uint16_t* in, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) o[i] = bf16f(in[i]);
}

static unsigned blocks(size_t n) { return (unsigned)((n + 255) / 256); }

Vision::Vision() = default;
Vision::~Vision() { for (void* p : allocs) cudaFree(p); }

void load_vision(Vision& V, const std::string& dir, int max_patches) {
    V.store.reset(new v41m::Store(dir, [](const std::string& n) {
        return n.compare(0, 7, "vision.") == 0 || n.compare(0, 8, "aligner.") == 0 || n.compare(0, 6, "image_") == 0;
    }));
    v41m::Store& S = *V.store;
    auto bf = [&](const std::string& n) { return S.p<uint16_t>(n); };
    auto dalloc = [&](size_t bytes) { void* p; VCK(cudaMalloc(&p, bytes)); V.allocs.push_back(p); return p; };
    auto bias = [&](const std::string& n, int len) {
        float* o = (float*)dalloc((size_t)len * 4);
        k_vbf2f<<<blocks(len), 256>>>(o, bf(n), len);
        return o;
    };
    V.patch_w = (uint16_t*)dalloc((size_t)VDIM * PATCH_IN_PAD * 2);
    VCK(cudaMemset(V.patch_w, 0, (size_t)VDIM * PATCH_IN_PAD * 2));
    VCK(cudaMemcpy2D(V.patch_w, PATCH_IN_PAD * 2, bf("vision.patch_embed.proj.weight"), PATCH_IN * 2, PATCH_IN * 2, VDIM,
                     cudaMemcpyDeviceToDevice));
    V.patch_b = bias("vision.patch_embed.proj.bias", VDIM);
    for (int L = 0; L < VLAYERS; ++L) {
        const std::string b = "vision.blocks." + std::to_string(L) + ".";
        VBlock& k = V.blk[L];
        k.norm1 = bf(b + "norm1.weight"); k.norm2 = bf(b + "norm2.weight");
        k.wqkv = bf(b + "attn.wqkv.weight"); k.bqkv = bias(b + "attn.wqkv.bias", 3 * VDIM);
        k.wo = bf(b + "attn.wo.weight"); k.bo = bias(b + "attn.wo.bias", VDIM);
        k.w1 = bf(b + "mlp.w1.weight"); k.w2 = bf(b + "mlp.w2.weight");
    }
    V.norm = bf("vision.norm.weight");
    V.al_w1 = bf("aligner.w1.weight"); V.al_b1 = bias("aligner.w1.bias", LLM_DIM);
    V.al_w2 = bf("aligner.w2.weight"); V.al_b2 = bias("aligner.w2.bias", LLM_DIM);
    V.img_start = bf("image_start"); V.img_end = bf("image_end"); V.img_newline = bf("image_newline");
    const size_t P = (size_t)max_patches;
    V.max_patches = max_patches;
    V.x = (float*)dalloc(P * VDIM * 4);
    V.t = (float*)dalloc(P * 3 * VDIM * 4);
    V.big = (float*)dalloc(P * 2 * VINTER * 4);
    V.xb = (uint16_t*)dalloc(P * VINTER * 2);
    V.qkvb = (uint16_t*)dalloc(P * 3 * VDIM * 2);
    V.ab = (uint16_t*)dalloc(P * VDIM * 2);
    VCK(cudaDeviceSynchronize());
    V.gib = (S.gib() * 1073741824.0 + P * (VDIM * 4 + 3 * VDIM * 4 + 2 * VINTER * 4 + VINTER * 2 + 3 * VDIM * 2 + VDIM * 2)) / 1073741824.0;
}

void encode_image(Vision& V, const uint16_t* patches, int n_h, int n_w, float* out, cudaStream_t st) {
    const int P = n_h * n_w;
    if (P > V.max_patches) throw std::runtime_error("vision: " + std::to_string(P) + " patches > max " + std::to_string(V.max_patches));
    const int lh = (n_h + DOWN - 1) / DOWN, lw = (n_w + DOWN - 1) / DOWN, Lr = lh * lw;
    // the aligner's unfold [Lr, 9216] and hidden [Lr, 5120] reuse big / t
    if ((size_t)Lr * AL_IN > (size_t)V.max_patches * 2 * VINTER * 2 || (size_t)Lr * LLM_DIM > (size_t)V.max_patches * 3 * VDIM)
        throw std::runtime_error("vision: aligner scratch too small");
    k_vpad<<<blocks((size_t)P * PATCH_IN_PAD), 256, 0, st>>>(V.xb, patches, P);
    gemm(V.x, V.xb, V.patch_w, V.patch_b, P, VDIM, PATCH_IN_PAD, st);
    for (int L = 0; L < VLAYERS; ++L) {
        const VBlock& k = V.blk[L];
        k_vnorm<<<P, 256, 0, st>>>(V.xb, V.x, k.norm1);
        gemm(V.t, V.xb, k.wqkv, k.bqkv, P, 3 * VDIM, VDIM, st);
        k_vrope<<<P, 256, 0, st>>>(V.qkvb, V.t, n_w);
        attention(V.ab, V.qkvb, P, st);
        gemm(V.t, V.ab, k.wo, k.bo, P, VDIM, VDIM, st);
        k_vadd<<<blocks((size_t)P * VDIM), 256, 0, st>>>(V.x, V.t, (size_t)P * VDIM);
        k_vnorm<<<P, 256, 0, st>>>(V.xb, V.x, k.norm2);
        gemm(V.big, V.xb, k.w1, nullptr, P, 2 * VINTER, VDIM, st);
        k_vswiglu<<<blocks((size_t)P * VINTER), 256, 0, st>>>(V.xb, V.big, (size_t)P * VINTER);
        gemm(V.t, V.xb, k.w2, nullptr, P, VDIM, VINTER, st);
        k_vadd<<<blocks((size_t)P * VDIM), 256, 0, st>>>(V.x, V.t, (size_t)P * VDIM);
    }
    k_vnorm<<<P, 256, 0, st>>>(V.ab, V.x, V.norm);                 // ViT output, bf16 [P, 1024]
    uint16_t* uf = reinterpret_cast<uint16_t*>(V.big);
    k_vunfold<<<blocks((size_t)Lr * AL_IN), 256, 0, st>>>(uf, V.ab, n_h, n_w, lw, (size_t)Lr * AL_IN);
    gemm(V.t, uf, V.al_w1, V.al_b1, Lr, LLM_DIM, AL_IN, st);
    k_vgelu<<<blocks((size_t)Lr * LLM_DIM), 256, 0, st>>>(V.xb, V.t, (size_t)Lr * LLM_DIM);
    gemm(out, V.xb, V.al_w2, V.al_b2, Lr, LLM_DIM, LLM_DIM, st);
    VCK(cudaGetLastError());
}

} // namespace v41v
