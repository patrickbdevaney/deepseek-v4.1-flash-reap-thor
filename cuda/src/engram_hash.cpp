// engram_hash.cpp — host n-gram hashing (engram.py NgramHashState). No CUDA: the CPU gate
// (tests/gate_engram_hash.cpp) links this file alone.
#include "engram_store.h"
#include "v41_config.h"
#include <cstdio>

// ------------------------------------------------------------------ tables
bool EngramHashTables::load(const std::string& path, std::string* err) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { if (err) *err = "cannot open " + path; return false; }
    tokmap.resize(v41::VOCAB);
    bool ok = fread(tokmap.data(), 4, v41::VOCAB, f) == (size_t)v41::VOCAB
           && fread(&pad, 8, 1, f) == 1
           && fread(mult, 8, 8, f) == 8 && fread(prime, 8, 48, f) == 48 && fread(offset, 8, 48, f) == 48;
    fclose(f);
    if (!ok && err) *err = "short read in " + path;
    return ok;
}

// ------------------------------------------------------------------ hashing (engram.py)
// tokens[s] for lookback s = 0..3 (s=0 is the position itself); blocked once any earlier lookback
// ran off the sequence start or hit a DEAD (image) token -> pad id from there on. Then the rolling
// XOR of tokens[s] * multiplier[layer][s]; after step i it is the (i+1)-gram hash, reduced modulo
// each head's prime and shifted into that head's row range.
void EngramHasher::hash(const int* tokens, const uint8_t* is_image, int n, int pos0, int64_t* ids) {
    if ((int)hist_.size() < pos0 + n) hist_.resize(pos0 + n);
    for (int i = 0; i < n; ++i)
        hist_[pos0 + i] = (is_image && is_image[i]) ? -1 : (int64_t)t_.tokmap[tokens[i]];
    for (int i = 0; i < n; ++i) {
        const int p = pos0 + i;
        int64_t tok[v41::ENGRAM_MAX_NGRAM]; bool blocked = false;
        for (int s = 0; s < v41::ENGRAM_MAX_NGRAM; ++s) {
            const int64_t src = hist_[p - s >= 0 ? p - s : 0];
            blocked = blocked || (p < s) || (src == -1);
            tok[s] = blocked ? t_.pad : src;
        }
        for (int l = 0; l < 2; ++l) {
            int64_t* out = ids + ((size_t)i * 2 + l) * v41::ENGRAM_COLS;
            int64_t rolling = tok[0] * t_.mult[l][0];
            for (int k = 1; k < v41::ENGRAM_MAX_NGRAM; ++k) {
                rolling ^= tok[k] * t_.mult[l][k];
                for (int h = 0; h < v41::ENGRAM_N_HEADS; ++h) {
                    const int c = (k - 1) * v41::ENGRAM_N_HEADS + h;
                    // torch `%` on int64 is floored; rolling >= 0 here (non-negative ids x odd
                    // positive multipliers bounded against overflow), so C++ % agrees.
                    out[c] = rolling % t_.prime[l][c] + t_.offset[l][c];
                }
            }
        }
    }
}

