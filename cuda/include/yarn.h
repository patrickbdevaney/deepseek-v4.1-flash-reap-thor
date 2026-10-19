// yarn.h — RoPE/YaRN frequency precompute (host, model.py:205-235 precompute_freqs_cis).
// Compressed layers: original_seq_len=65536, base=compress_rope_theta (YaRN on).
// Pure-sliding layers: original_seq_len=0, base=rope_theta (YaRN off).
// Fills cos[seqlen, dim/2], sin[seqlen, dim/2] with cos/sin(t * freq_j). One-time host precompute.
#pragma once
#include <vector>
#include <cmath>

namespace yarn {

inline double _corr_dim(double num_rot, int dim, double base, double max_seq) {
    return dim * std::log(max_seq / (num_rot * 2.0 * M_PI)) / (2.0 * std::log(base));
}

// cos/sin sized [seqlen * (dim/2)] row-major (per position, dim/2 freqs).
inline void freqs(std::vector<float>& cos, std::vector<float>& sin, int seqlen, int dim,
                  int original_seq_len, double base, double factor, int beta_fast, int beta_slow) {
    int half = dim / 2;
    std::vector<double> f(half);
    for (int j = 0; j < half; ++j) f[j] = 1.0 / std::pow(base, (double)(2 * j) / dim);

    if (original_seq_len > 0) {                                    // YaRN
        double lo = std::floor(_corr_dim(beta_fast, dim, base, original_seq_len));
        double hi = std::ceil (_corr_dim(beta_slow, dim, base, original_seq_len));
        double low = lo < 0 ? 0 : lo, high = hi > dim - 1 ? dim - 1 : hi;
        if (low == high) high += 0.001;
        for (int j = 0; j < half; ++j) {
            double ramp = ((double)j - low) / (high - low);       // linear_ramp_factor over dim/2
            ramp = ramp < 0 ? 0 : (ramp > 1 ? 1 : ramp);
            double smooth = 1.0 - ramp;
            f[j] = f[j] / factor * (1.0 - smooth) + f[j] * smooth;
        }
    }
    cos.resize((size_t)seqlen * half); sin.resize((size_t)seqlen * half);
    for (int t = 0; t < seqlen; ++t)
        for (int j = 0; j < half; ++j) {
            double a = (double)t * f[j];
            cos[(size_t)t * half + j] = (float)std::cos(a);
            sin[(size_t)t * half + j] = (float)std::sin(a);
        }
}

} // namespace yarn
