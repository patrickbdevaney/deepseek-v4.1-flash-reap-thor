// gate_kt_host.cu — the thor-v1 trellis decoders (DecKT in v41_gemv.cuh) run on the CPU against
// ops/lib/kt.dequant. Same source as the device path; host build uses -ffp-contract=off.
// Needs no GPU: python -I tests/v41_unit_ref.py --out build/units && build/gate_kt_host build/units
#include "../kernels/v41/v41_gemv.cuh"
#include "npy.h"
#include <cmath>

template <int B>
static int check(const std::string& dir, const char* qt) {
    const Npy w = npy_load(dir + "/dec_" + qt + "_w.npy"), s = npy_load(dir + "/dec_" + qt + "_s.npy"),
              ref = npy_load(dir + "/dec_" + qt + "_ref.npy");
    const int R = (int)ref.shape[0], C = (int)ref.shape[1];
    v41g::DecKT<B> d; d.W0 = d.W1 = w.as<uint8_t>(); d.S0 = d.S1 = s.as<uint8_t>(); d.K = C; d.split = 1 << 30;
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
    return check<2>(dir, "KT2") + check<3>(dir, "KT3") ? 1 : 0;
}
