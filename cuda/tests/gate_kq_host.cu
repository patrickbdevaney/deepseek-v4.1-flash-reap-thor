// gate_kq_host.cu — the thor-v1 K-quant decoders (DecKQ in v41_gemv.cuh) run on the CPU against
// ops/lib/ggml_quants.dequant. Same source as the device path; host build uses -ffp-contract=off.
// Needs no GPU: python -I tests/v41_unit_ref.py --out build/units && build/gate_kq_host build/units
#include "../kernels/v41/v41_gemv.cuh"
#include "npy.h"
#include <cmath>

template <int QT>
static int check(const std::string& dir, const char* qt) {
    const Npy w = npy_load(dir + "/dec_" + qt + "_w.npy"), ref = npy_load(dir + "/dec_" + qt + "_ref.npy");
    const int R = (int)ref.shape[0], C = (int)ref.shape[1];
    v41g::DecKQ<QT> d; d.W0 = d.W1 = w.as<uint8_t>(); d.K = C; d.split = 1 << 30;
    long bad = 0; double maxd = 0;
    for (int r = 0; r < R; ++r)
        for (int c = 0; c < C / 32; ++c) {
            float v[32]; d.load(r, c, v);
            for (int i = 0; i < 32; ++i) {
                const float g = v[i], e = ref.as<float>()[(size_t)r * C + c * 32 + i];
                if (g != e) { ++bad; maxd = std::max(maxd, (double)std::fabs(g - e)); }
            }
        }
    printf("[%s] %d x %d decoded on host: %ld mismatches (max |d| %.3g) -> %s\n", qt, R, C, bad, maxd, bad ? "FAIL" : "PASS");
    return bad ? 1 : 0;
}
int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : "build/units";
    const int f = check<2>(dir, "Q2_K") + check<3>(dir, "Q3_K") + check<4>(dir, "Q4_K");
    return f ? 1 : 0;
}
