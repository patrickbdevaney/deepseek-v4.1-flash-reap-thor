// gate_engram_hash — the C++ hasher against hash ids produced by the reference engram.py
// (tools/engram_meta.py writes the fixture by running NgramHashState itself). CPU only.
// Cases: plain text, code, CJK + Latin, and 300 random ids with a 40-token image span (DEAD).
// Each case is hashed twice: as one prefill, and token by token (the decode path), which must
// agree -- that is the prefill/decode split the host state carries.
#include "engram_store.h"
#include "v41_config.h"
#include <cstdio>
#include <cstring>
#include <vector>

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : "tests/data";
    EngramHashTables T; std::string err;
    if (!T.load(dir + "/engram_tokmap.bin", &err)) { fprintf(stderr, "FAIL %s\n", err.c_str()); return 1; }
    FILE* f = fopen((dir + "/engram_hash_fixture.bin").c_str(), "rb");
    if (!f) { fprintf(stderr, "FAIL no fixture\n"); return 1; }
    int nc = 0; if (fread(&nc, 4, 1, f) != 1) return 1;
    int bad = 0; long total = 0;
    for (int c = 0; c < nc; ++c) {
        int n = 0; if (fread(&n, 4, 1, f) != 1) return 1;
        std::vector<int> tok(n); std::vector<uint8_t> img(n);
        std::vector<int64_t> want((size_t)n * 2 * v41::ENGRAM_COLS), got(want.size()), got2(want.size());
        if (fread(tok.data(), 4, n, f) != (size_t)n || fread(img.data(), 1, n, f) != (size_t)n ||
            fread(want.data(), 8, want.size(), f) != want.size()) return 1;
        EngramHasher H(T); H.hash(tok.data(), img.data(), n, 0, got.data());
        EngramHasher D(T);
        for (int i = 0; i < n; ++i) D.hash(&tok[i], &img[i], 1, i, got2.data() + (size_t)i * 2 * v41::ENGRAM_COLS);
        int cb = 0;
        for (size_t i = 0; i < want.size(); ++i) { cb += got[i] != want[i]; cb += got2[i] != want[i]; }
        for (size_t i = 0; i < want.size(); ++i) {
            const int l = (int)((i / v41::ENGRAM_COLS) % 2);
            if (want[i] < 0 || want[i] >= v41::ENGRAM_ROWS[l]) { ++cb; break; }
        }
        printf("case %d: n=%d  mismatches=%d\n", c, n, cb);
        bad += cb; total += (long)want.size();
    }
    fclose(f);
    printf("%s: %ld hash ids, %d mismatches\n", bad ? "FAIL" : "PASS", total, bad);
    return bad ? 1 : 0;
}
