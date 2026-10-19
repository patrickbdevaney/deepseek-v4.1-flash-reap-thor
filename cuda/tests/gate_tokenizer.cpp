// gate_tokenizer.cpp — Gate S1a. The C++ tokenizer must agree with HF token-for-token.
//
// A tokenizer that is 99 % right is not 99 % of a server: one wrong split at position 3 makes the
// model answer a different question, silently, with no error anywhere. So this gate is exact and
// it is over a corpus chosen to break ByteLevel-BPE ports (see tools/gen_tokenizer_vectors.py),
// not over a handful of greetings.
//
//   python3 tools/gen_tokenizer_vectors.py            # reference ids from HF
//   g++ -O2 -std=c++17 -I include tests/gate_tokenizer.cpp -o build/gate_tokenizer
//   ./build/gate_tokenizer
#include "tokenizer_dsv4.h"
#include <cstdio>
#include <fstream>
#include <chrono>

static std::string vis(const std::string& s, size_t cap = 60) {
    std::string o;
    for (size_t i = 0; i < s.size() && o.size() < cap; ++i) {
        const unsigned char c = s[i];
        if (c == '\n') o += "\\n";
        else if (c == '\r') o += "\\r";
        else if (c == '\t') o += "\\t";
        else if (c < 0x20) { char b[8]; snprintf(b, sizeof b, "\\x%02x", c); o += b; }
        else o.push_back((char)c);
    }
    if (o.size() >= cap) o += "...";
    return o;
}

int main(int argc, char** argv) {
    const char* ck = argc > 1 ? argv[1] : "/home/patrickd/models/DeepSeek-V4.1-Flash-1017";
    const char* vecs = argc > 2 ? argv[2] : "tests/data/tokenizer_vectors.json";

    dsv4tok::Tokenizer tok;
    tok.load(std::string(ck) + "/tokenizer.json");
    printf("[tok] vocab %zu  merges %zu  added %zu  bos %d eos %d\n",
           tok.vocab.size(), tok.merges.size(), tok.added.size(), tok.bos_id, tok.eos_id);

    // The gate the rest of the repo already uses, kept here so it fails loudly and first.
    {
        const std::vector<int> want{671, 6102, 294, 8760, 344};
        const auto got = tok.encode("The capital of France is");
        if (got != want) {
            printf("  [FAIL] canonical gate: got");
            for (int x : got) printf(" %d", x);
            printf("\n");
            return 1;
        }
        printf("  [PASS] canonical gate 'The capital of France is'\n");
    }

    std::ifstream f(vecs);
    if (!f) { printf("  [FAIL] cannot open %s (run tools/gen_tokenizer_vectors.py)\n", vecs); return 1; }
    nlohmann::json j; f >> j;

    int n = 0, bad = 0, shown = 0;
    long ntok = 0;
    for (const auto& c : j) {
        const std::string t = c["t"].get<std::string>();
        std::vector<int> want = c["ids"].get<std::vector<int>>();
        const auto got = tok.encode(t);
        ++n; ntok += (long)want.size();
        if (got != want) {
            ++bad;
            if (shown++ < 8) {
                printf("  [FAIL] %s\n", vis(t).c_str());
                size_t k = 0;
                while (k < got.size() && k < want.size() && got[k] == want[k]) ++k;
                printf("         first divergence at %zu: got", k);
                for (size_t x = k; x < got.size() && x < k + 6; ++x) printf(" %d[%s]", got[x], vis(tok.decode({got[x]}, false), 12).c_str());
                printf("\n                                  want");
                for (size_t x = k; x < want.size() && x < k + 6; ++x) printf(" %d[%s]", want[x], vis(tok.decode({want[x]}, false), 12).c_str());
                printf("\n");
            }
        }
    }
    printf("  encode: %d/%d cases exact (%ld reference tokens)\n", n - bad, n, ntok);

    // Round-trip. decode(encode(t)) must reproduce t for every case that is valid UTF-8 text --
    // ByteLevel is lossless by construction, so any failure here is a decoder bug.
    int rt_bad = 0;
    for (const auto& c : j) {
        const std::string t = c["t"].get<std::string>();
        if (tok.decode(tok.encode(t), false) != t) {
            if (rt_bad++ < 4) printf("  [FAIL] roundtrip: %s\n", vis(t).c_str());
        }
    }
    printf("  roundtrip: %d/%d cases lossless\n", n - rt_bad, n);

    // Streaming decode must never emit a partial UTF-8 character, and must converge to the same
    // bytes as a one-shot decode. This is what the SSE path relies on.
    int st_bad = 0;
    for (const auto& c : j) {
        const std::string t = c["t"].get<std::string>();
        const auto ids = tok.encode(t);
        std::string acc;
        std::vector<int> seen;
        size_t emitted = 0;
        for (int id : ids) {
            seen.push_back(id);
            size_t nb = 0;
            const std::string piece = tok.decode_stream(seen, 0, nb);
            if (nb > emitted) { acc += piece.substr(emitted); emitted = nb; }
        }
        if (acc != tok.decode(ids, true)) { if (st_bad++ < 4) printf("  [FAIL] stream: %s\n", vis(t).c_str()); }
    }
    printf("  stream:    %d/%d cases match one-shot decode\n", n - st_bad, n);

    // The markers the server's streaming splitter and tool parser hunt for are ADDED TOKENS, so a
    // decode that skips specials deletes them and both features silently stop working -- reasoning
    // arrives glued into content, tool calls arrive as unparseable markup, and nothing errors.
    // Found by scripts/smoke_server.sh; pinned here so it stays fixed.
    {
        int marker_fail = 0;
        for (const char* m : { "</think>", "<｜DSML｜tool_calls>", "<｜DSML｜invoke name=\"f\">" }) {
            const auto ids = tok.encode(m);
            size_t nb = 0;
            const std::string kept = tok.decode_stream(ids, 0, nb, /*skip_special=*/false);
            const std::string dropped = tok.decode_stream(ids, 0, nb, /*skip_special=*/true);
            if (kept.find("</think>") == std::string::npos && kept.find("DSML") == std::string::npos) {
                printf("  [FAIL] marker vanished with skip_special=false: %s -> \"%s\"\n", m, kept.c_str());
                ++marker_fail;
            }
            if (dropped == kept && kept.find("DSML") != std::string::npos) {
                printf("  [FAIL] skip_special=true did not strip specials for %s\n", m);
                ++marker_fail;
            }
        }
        printf("  [%s] </think> and DSML markers survive decode_stream(skip_special=false)\n",
               marker_fail ? "FAIL" : "PASS");
        if (marker_fail) return 1;
    }

    // Throughput. Not a correctness gate, but the number that decides whether tokenisation shows up
    // in time-to-first-token once the prefix cache has removed the GPU work from a cached turn.
    {
        std::string big;
        for (const auto& c : j) { big += c["t"].get<std::string>(); if (big.size() > (1u << 20)) break; }
        const auto t0 = std::chrono::steady_clock::now();
        const auto ids = tok.encode(big);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        printf("  speed:     %.1f MB/s  (%zu bytes -> %zu tokens in %.1f ms)\n",
               big.size() / 1e6 / (ms / 1e3), big.size(), ids.size(), ms);
    }

    const int fails = bad + rt_bad + st_bad;
    printf("\nGate TOKENIZER: %s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
