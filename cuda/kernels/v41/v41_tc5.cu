// v41_tc5.cu — P8 stage 1: prefill NVFP4-overlay GEMMs on tcgen05 (CUTLASS SM100 mixed-input, sm_110a).
//
// y[M, N] = g * x_bf16[M, K] . (e2m1[N, K] * scale_bf16[N, K/16])^T, fp32 accumulate in TMEM.
// Every operand is the decode operand exactly: x = e4m3 x pow2 (act_quant_fp8) and e2m1 x e4m3 scale are
// exact in bf16, g is the epilogue alpha -- only the accumulation order differs from k_nvm (T1). Decode and
// verify stay on k_nvm. The k_nvm-repacked codes / scales are unpacked per call into scratch (row-major
// codes, [K/16][N] bf16 scales): no second resident copy of the overlay.
#include "v41_kernels.h"
#include "v41_config.h"
#include "v41_dev.cuh"
#include <cstdio>
#include <cstdlib>
#include <unordered_map>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/detail/collective/mixed_input_utils.hpp"

using namespace cute;

namespace v41tc5 {

using ElementX = cutlass::bfloat16_t;            // activations (the wide operand)
using ElementW = cutlass::float_e2m1_t;          // weights (narrow, upcast in the mainloop)
using ElementScale = cutlass::bfloat16_t;
using LayoutX = cutlass::layout::RowMajor;
using LayoutW = cutlass::layout::ColumnMajor;    // K contiguous per weight row
using LayoutX_T = typename cutlass::layout::LayoutTranspose<LayoutX>::type;
using LayoutW_T = typename cutlass::layout::LayoutTranspose<LayoutW>::type;
using LayoutD = cutlass::layout::RowMajor;
constexpr int AlignX = 128 / cutlass::sizeof_bits<ElementX>::value;
constexpr int AlignW = 128 / cutlass::sizeof_bits<ElementW>::value;
constexpr int AlignD = 128 / cutlass::sizeof_bits<float>::value;
using MmaTileShape = Shape<_256, _128, _128>;    // (weights N, tokens M, K): operands swapped as in CUTLASS ex. 86
using ClusterShape = Shape<_2, _1, _1>;
constexpr int ScaleGranularityN = 1, ScaleGranularityK = 16;
using ScaleConfig = cutlass::detail::Sm100MixedInputBlockwiseScaleConfig<ScaleGranularityN, ScaleGranularityK>;
using LayoutScale = decltype(ScaleConfig::deduce_layout_scale());

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, MmaTileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto, float, float,
    void, typename cutlass::layout::LayoutTranspose<LayoutD>::type, AlignD,
    float, typename cutlass::layout::LayoutTranspose<LayoutD>::type, AlignD,
    cutlass::epilogue::TmaWarpSpecialized2Sm>::CollectiveOp;
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    cute::tuple<ElementW, ElementScale>, cute::tuple<LayoutW_T, LayoutScale>, AlignW,
    ElementX, LayoutX_T, AlignX, float, MmaTileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecialized2SmMixedInputSm100>::CollectiveOp;
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
using StrideX = cutlass::detail::TagToStrideA_t<LayoutX>;
using StrideW = cutlass::detail::TagToStrideB_t<LayoutW>;
using StrideD = typename GemmKernel::StrideD;

// k_nvm layout (v41_nv_mma.cuh) -> row-major codes [N][K/2] (low nibble = even k). One thread per output
// word (8 codes of one row), so stores are coalesced; the two source words it needs sit in L2.
__global__ void k_tc5_unpack_codes(uint32_t* __restrict__ dst, const uint8_t* __restrict__ src, int N, int K) {
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x, wpr = K / 8;
    if (i >= (size_t)N * wpr) return;
    const int row = (int)(i / wpr), k0 = (int)(i % wpr) * 8, nsb = K / 256;
    const int T = row >> 4, rr = row & 15, gq = rr & 7, hi = rr >> 3, s = k0 >> 8, sb = (k0 & 255) >> 4, t0 = (k0 & 15) >> 2;
    const size_t base = ((size_t)T * nsb + s) * 2048 + (sb >> 2) * 512 + (sb & 3) * 4;
    uint32_t out = 0;
    #pragma unroll
    for (int h = 0; h < 2; ++h) {                       // codes k0 + 4h .. +3 from lane 4 gq + t0 + h
        const uint32_t w = *reinterpret_cast<const uint32_t*>(src + base + (size_t)(4 * gq + t0 + h) * 16);
        #pragma unroll
        for (int b = 0; b < 4; ++b) out |= ((w >> (8 * b + 4 * hi)) & 15u) << (4 * (4 * h + b));
    }
    dst[i] = out;
}
// k_nvm scale tiles (16 rows x 16 e4m3 per (tile, superblock)) -> bf16 [K/16][N], one thread per output
__global__ void k_tc5_unpack_scales(__nv_bfloat16* __restrict__ dst, const uint8_t* __restrict__ src, int N, int K) {
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)N * (K / 16)) return;
    const int n = (int)(i % N), kb = (int)(i / N), nsb = K / 256;
    dst[i] = __float2bfloat16(v41d::e4m3f(src[((size_t)(n >> 4) * nsb + (kb >> 4)) * 256 + (n & 15) * 16 + (kb & 15)]));
}
// act_quant_fp8 output -> bf16 (exact: e4m3 x pow2)
__global__ void k_tc5_act(__nv_bfloat16* __restrict__ dst, const uint8_t* __restrict__ xq, const float* __restrict__ xs, size_t n) {
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2bfloat16(v41d::e4m3f(xq[i]) * xs[i / 32]);
}
__global__ void k_tc5_round(float* y, size_t n) {
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) y[i] = __bfloat162float(__float2bfloat16(y[i]));
}

struct Scratch { void* p = nullptr; size_t cap = 0; };
void* grow(Scratch& s, size_t bytes) {
    if (bytes <= s.cap) return s.p;
    if (s.p) cudaFree(s.p);
    if (cudaMalloc(&s.p, bytes) != cudaSuccess) { fprintf(stderr, "[tc5] scratch alloc %zu\n", bytes); abort(); }
    s.cap = bytes;
    return s.p;
}
// one set per stream: the engine runs linears concurrently on its KV-branch and shared-expert streams
struct Bufs { Scratch codes, scales, act, ws, sgrp; };
Bufs& bufs(cudaStream_t st) { static std::unordered_map<cudaStream_t, Bufs> m; return m[st]; }

} // namespace v41tc5

namespace v41k {
using namespace v41tc5;

bool tc5_on() { static const bool on = [] { const char* e = getenv("V41_TC5"); return e && *e == '1'; }(); return on; }

// x rows: act_quant_fp8 codes/scales with row stride ldx (elements); group gi of G reads columns [gi K, (gi+1) K)
bool tc5_nvfp4(float* y, const uint8_t* xq, const float* xs, int ldx, const uint8_t* W, const uint8_t* S, float g,
               int M, int N, int K, int G, bool round, cudaStream_t st) {
    const int Ng = N / G;
    if (K % 256 || Ng % 16 || M < 1) return false;
    Bufs& B = bufs(st);
    Scratch &g_act = B.act, &g_codes = B.codes, &g_scales = B.scales, &g_ws = B.ws, &g_sgrp = B.sgrp;
    __nv_bfloat16* xb = (__nv_bfloat16*)grow(g_act, (size_t)M * ldx * 2);
    const size_t nx = (size_t)M * ldx;
    k_tc5_act<<<(unsigned)((nx + 255) / 256), 256, 0, st>>>(xb, xq, xs, nx);
    uint8_t* wc = (uint8_t*)grow(g_codes, (size_t)N * K / 2);
    __nv_bfloat16* sc = (__nv_bfloat16*)grow(g_scales, (size_t)N * (K / 16) * 2);
    {
        const size_t th = (size_t)N * (K / 8);
        k_tc5_unpack_codes<<<(unsigned)((th + 255) / 256), 256, 0, st>>>(reinterpret_cast<uint32_t*>(wc), W, N, K);
        const size_t ts = (size_t)N * (K / 16);
        // scales of group gi go to their own [K/16][Ng] block (CUTLASS sees one problem per group)
        k_tc5_unpack_scales<<<(unsigned)((ts + 255) / 256), 256, 0, st>>>(sc, S, N, K);
    }
    for (int gi = 0; gi < G; ++gi) {
        StrideX sx = make_stride((int64_t)ldx, Int<1>{}, (int64_t)0);
        StrideW sw = make_stride((int64_t)K, Int<1>{}, (int64_t)0);
        StrideD sd = make_stride(Int<1>{}, (int64_t)N, (int64_t)0);
        auto layout_S = ScaleConfig::tile_atom_to_shape_scale(make_shape(Ng, K, 1));
        // the [K/16][N] scale array viewed per group: column offset gi Ng, row stride N
        // (tile_atom_to_shape_scale assumes row stride Ng; G > 1 uses a per-group copy below)
        const __nv_bfloat16* scg = sc + (size_t)gi * Ng;
        if (G > 1) {
            __nv_bfloat16* dst = (__nv_bfloat16*)grow(g_sgrp, (size_t)Ng * (K / 16) * 2);
            cudaMemcpy2DAsync(dst, (size_t)Ng * 2, scg, (size_t)N * 2, (size_t)Ng * 2, K / 16, cudaMemcpyDeviceToDevice, st);
            scg = dst;
        }
        typename Gemm::Arguments args{
            cutlass::gemm::GemmUniversalMode::kGemm,
            {Ng, M, K, 1},
            {reinterpret_cast<const ElementW*>(wc + (size_t)gi * Ng * (K / 2)), sw,
             reinterpret_cast<const ElementX*>(xb + (size_t)gi * K), sx,
             reinterpret_cast<const ElementScale*>(scg), layout_S},
            {{g, 0.f}, nullptr, sd, y + (size_t)gi * Ng, sd}};
        Gemm gemm;
        if (gemm.can_implement(args) != cutlass::Status::kSuccess) {
            static bool warned = false;
            if (!warned) { fprintf(stderr, "[tc5] can_implement failed M=%d N=%d K=%d\n", M, Ng, K); warned = true; }
            return false;
        }
        void* ws = grow(g_ws, std::max<size_t>(Gemm::get_workspace_size(args), 16));
        if (gemm.initialize(args, ws, st) != cutlass::Status::kSuccess || gemm.run(st) != cutlass::Status::kSuccess) {
            fprintf(stderr, "[tc5] gemm failed M=%d N=%d K=%d\n", M, Ng, K);
            abort();
        }
    }
    if (round) k_tc5_round<<<(unsigned)(((size_t)M * N + 255) / 256), 256, 0, st>>>(y, (size_t)M * N);
    return true;
}

} // namespace v41k
