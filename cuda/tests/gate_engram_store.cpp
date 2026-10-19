// gate_engram_store.cpp — the Engram row store's I/O path, on the CPU (no CUDA calls: host_only).
//
// Builds two SPARSE full-size .v41rows files (llamacpp/ENGRAM_ROWFILE.md header, real n_rows, so the
// bucket-coverage check is the real one; holes cost no disk), plants random bytes at every row the
// hash of a test prompt selects, then fetches through the real io_uring + O_DIRECT path and checks
// every slot byte-for-byte against the planted row, plus the flag protocol and the short-read check.
//
//   g++ -O2 -std=c++17 -I include tests/gate_engram_store.cpp src/engram_store.cpp src/engram_hash.cpp \
//       -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart -o build/gate_engram_store
//   build/gate_engram_store <scratch dir on the NVMe> tests/data/engram_tokmap.bin
#include "engram_store.h"
#include "v41_config.h"
#include <fcntl.h>
#include <unistd.h>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static void write_file(const std::string& p, int layer, int64_t n_rows, int rb, int vdt) {
    int fd = ::open(p.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0644);
    std::vector<uint8_t> h(4096, 0);
    memcpy(h.data(), "V41ENGR1", 8);
    uint32_t u; uint64_t q;
    u = 1; memcpy(&h[8], &u, 4); u = layer; memcpy(&h[12], &u, 4); q = n_rows; memcpy(&h[16], &q, 8);
    u = 256; memcpy(&h[24], &u, 4); u = 8; memcpy(&h[28], &u, 4); u = rb; memcpy(&h[32], &u, 4);
    u = vdt; memcpy(&h[36], &u, 4); u = 1; memcpy(&h[40], &u, 4); q = 4096; memcpy(&h[44], &q, 8);
    if (pwrite(fd, h.data(), 4096, 0) != 4096) { perror("pwrite"); exit(2); }
    if (ftruncate(fd, 4096 + n_rows * rb)) { perror("ftruncate"); exit(2); }
    close(fd);
}

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : "build/engram_gate";
    const std::string tokmap = argc > 2 ? argv[2] : "tests/data/engram_tokmap.bin";
    int fails = 0;
    for (int fmt = 0; fmt < 2; ++fmt) {
        const int rb = fmt ? 200 : 264, vdt = fmt ? 2 : 1;
        for (int l = 0; l < 2; ++l)
            write_file(dir + "/engram_L" + std::to_string(v41::ENGRAM_LAYERS[l]) + ".v41rows", v41::ENGRAM_LAYERS[l],
                       v41::ENGRAM_ROWS[l], rb, vdt);
        EngramStoreConfig c; c.dir = dir; c.tokmap = tokmap; c.max_tokens = 128; c.host_only = true;
        EngramStore S; std::string err;
        if (!S.open(c, &err)) { printf("open: %s -> FAIL\n", err.c_str()); return 1; }
        EngramHasher H(S.tables());
        std::mt19937 rng(41 + fmt);
        std::vector<int> toks(128); for (auto& t : toks) t = rng() % v41::VOCAB;
        std::vector<int64_t> ids(128 * 2 * v41::ENGRAM_COLS);
        H.hash(toks.data(), nullptr, 128, 0, ids.data());
        // plant a distinct random row at every selected id
        int fds[2];
        for (int l = 0; l < 2; ++l) fds[l] = ::open((dir + "/engram_L" + std::to_string(v41::ENGRAM_LAYERS[l]) + ".v41rows").c_str(), O_WRONLY);
        std::vector<std::vector<uint8_t>> want(2 * 128 * v41::ENGRAM_COLS, std::vector<uint8_t>(rb));
        for (int i = 0; i < 128; ++i)
            for (int l = 0; l < 2; ++l)
                for (int k = 0; k < v41::ENGRAM_COLS; ++k) {
                    const int64_t r = ids[((size_t)i * 2 + l) * v41::ENGRAM_COLS + k];
                    auto& w = want[((size_t)l * 128 + i) * v41::ENGRAM_COLS + k];
                    for (auto& b : w) b = rng() & 0xFF;
                    if (pwrite(fds[l], w.data(), rb, 4096 + r * rb) != rb) { perror("plant"); return 2; }
                }
        // duplicates: a later plant overwrites an earlier one at the same row -> re-read the truth
        for (int l = 0; l < 2; ++l) { fsync(fds[l]); close(fds[l]); }
        for (int l = 0; l < 2; ++l) {
            int fd = ::open((dir + "/engram_L" + std::to_string(v41::ENGRAM_LAYERS[l]) + ".v41rows").c_str(), O_RDONLY);
            for (int i = 0; i < 128; ++i) for (int k = 0; k < v41::ENGRAM_COLS; ++k) {
                const int64_t r = ids[((size_t)i * 2 + l) * v41::ENGRAM_COLS + k];
                (void)!pread(fd, want[((size_t)l * 128 + i) * v41::ENGRAM_COLS + k].data(), rb, 4096 + r * rb);
            }
            close(fd);
        }
        for (int rep = 0; rep < 3; ++rep) {
            const int n = rep == 0 ? 128 : rep == 1 ? 1 : 77;
            if (!S.fetch(ids.data(), n, &err)) { printf("fetch: %s -> FAIL\n", err.c_str()); return 1; }
            long bad = 0;
            for (int l = 0; l < 2; ++l) {
                const uint8_t* A = S.arena_host(l);
                const uint16_t* off = (const uint16_t*)(A + (size_t)128 * v41::ENGRAM_COLS * 1024);
                for (int s = 0; s < n * v41::ENGRAM_COLS; ++s)
                    bad += memcmp(A + (size_t)s * 1024 + off[s], want[(size_t)l * 128 * v41::ENGRAM_COLS + s].data(), rb) != 0;
                if (*S.flag_host(l) != 1) { printf("  flag[%d] not set\n", l); ++bad; }
            }
            printf("[fmt %s, %3d tokens] %ld bad slots, %.0f us -> %s\n", fmt ? "mxfp6" : "fp8  ", n, bad, S.last_us(), bad ? "FAIL" : "PASS");
            fails += bad != 0;
        }
    }
    // a truncated file must be refused
    {
        write_file(dir + "/engram_L1.v41rows", 1, 1000, 264, 1);
        EngramStoreConfig c; c.dir = dir; c.tokmap = tokmap; c.host_only = true;
        EngramStore S; std::string err;
        const bool ok = !S.open(c, &err);
        printf("[n_rows below the hash buckets] refused (%s) -> %s\n", err.c_str(), ok ? "PASS" : "FAIL");
        fails += !ok;
    }
    for (int l = 0; l < 2; ++l) unlink((dir + "/engram_L" + std::to_string(v41::ENGRAM_LAYERS[l]) + ".v41rows").c_str());
    printf("Gate ENGRAM STORE: %s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
