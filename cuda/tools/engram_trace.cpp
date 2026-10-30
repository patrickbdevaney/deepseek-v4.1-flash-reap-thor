// engram_trace.cpp — M3: Engram row ids for a D2 token trace, host only (no CUDA, no row files).
//   g++ -O2 -std=c++17 -I include tools/engram_trace.cpp src/engram_hash.cpp -o build/engram_trace
//   build/engram_trace TOKMAP TRACE_DIR OUT   -> OUT: int64 [T, 2, ENGRAM_COLS] row ids, each seq hashed from pos 0
#include "engram_store.h"
#include "v41_config.h"
#include <cstdio>
#include <fstream>
#include <regex>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 4) { fprintf(stderr, "usage: %s TOKMAP TRACE_DIR OUT\n", argv[0]); return 2; }
    EngramHashTables t; std::string err;
    if (!t.load(argv[1], &err)) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
    const std::string dir = argv[2];
    std::ifstream fk(dir + "/tokens.bin", std::ios::binary);
    fk.seekg(0, std::ios::end); std::vector<int> all(fk.tellg() / 4); fk.seekg(0); fk.read((char*)all.data(), all.size() * 4);
    std::ifstream fs(dir + "/seqs.jsonl"); std::string l;
    FILE* fo = fopen(argv[3], "wb");
    const std::regex ro("\"offset\":(\\d+)"), rl("\"length\":(\\d+)");
    long T = 0; int n = 0;
    while (std::getline(fs, l)) {
        std::smatch a, b;
        if (!std::regex_search(l, a, ro) || !std::regex_search(l, b, rl)) continue;
        const long o = std::stol(a[1]), len = std::stol(b[1]);
        EngramHasher h(t);
        std::vector<int64_t> ids((size_t)len * 2 * v41::ENGRAM_COLS);
        h.hash(all.data() + o, nullptr, (int)len, 0, ids.data());
        fwrite(ids.data(), 8, ids.size(), fo); T += len; ++n;
    }
    fclose(fo);
    printf("%d seqs, %ld tokens, %d cols\n", n, T, v41::ENGRAM_COLS);
    return 0;
}
