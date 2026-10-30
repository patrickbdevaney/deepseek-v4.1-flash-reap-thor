// v41_kqt.cu — host side of the tcgen05 Q2_K / Q3_K prefill experts (v41_kqt.cuh). P8 stage 2b. Built with the
// cutlass flags (scripts/build_v41.sh), so v41_moe.cu reaches it through v41_kernels.h only.
#include "v41_kernels.h"
#include "v41_config.h"
#include "v41_kqt.cuh"
#include <cstdlib>

namespace v41k {
namespace {
struct KqtUnit { int e, start, cnt; };                 // = v41_moe.cu Unit
constexpr int KQT_NT = 64;
}

size_t kqt_stage_bytes(int K) { return v41g::kqt_stage_bytes(K); }

void kqt_prep(int bits, uint8_t* dst, const uint8_t* xq, const float* xs, int ldx, int rows, int K, cudaStream_t st) {
    v41g::XSrc X; X.xq = xq; X.xs = xs; X.ldx = ldx;
    if (bits == 2) v41g::k_kqt_prep<2><<<rows, 256, 0, st>>>(dst, X, K);
    else v41g::k_kqt_prep<3><<<rows, 256, 0, st>>>(dst, X, K);
}

template <int QT>
static bool kqt_run_(float* out, int ldo, const uint8_t* stg, bool by_pair, const void* units, const int* units_n, const int* pair_tok,
                     const uint8_t* const* w1, const uint8_t* const* w3, const uint8_t* const* w2, int pass, int N, int K, int cmin,
                     cudaStream_t st) {
    if (N % 128 || K % 256) return false;
    auto kf = v41g::k_kqt<QT, KQT_NT, KqtUnit>;
    const size_t smem = sizeof(v41g::KqtSmem<KQT_NT>) + 1024;
    static const bool attr = [&] { return cudaFuncSetAttribute(kf, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem) == cudaSuccess; }();
    if (!attr) return false;
    // persistent unit rows: N / 128 x GY CTAs, one resident per SM (smem)
    static const int gy = [] { const char* e = getenv("V41_KQT_GY"); const int v = e ? atoi(e) : 5; return v < 1 ? 1 : v; }();
    v41g::XSrc X; X.stg = stg;
    v41g::KQTab tab{w1, w3, w2, pass};
    kf<<<dim3(N / 128, gy), v41g::KQT_THREADS, smem, st>>>(out, ldo, X, by_pair, (const KqtUnit*)units, units_n, pair_tok,
                                                           tab, N, K, cmin, KQT_NT);
    return true;
}
bool kqt_run(int bits, float* out, int ldo, const uint8_t* stg, bool by_pair, const void* units, const int* units_n, const int* pair_tok,
             const uint8_t* const* w1, const uint8_t* const* w3, const uint8_t* const* w2, int pass, int N, int K, int cmin,
             cudaStream_t st) {
    if (bits == 2) return kqt_run_<2>(out, ldo, stg, by_pair, units, units_n, pair_tok, w1, w3, w2, pass, N, K, cmin, st);
    if (bits == 3) return kqt_run_<3>(out, ldo, stg, by_pair, units, units_n, pair_tok, w1, w3, w2, pass, N, K, cmin, st);
    return false;
}

} // namespace v41k
