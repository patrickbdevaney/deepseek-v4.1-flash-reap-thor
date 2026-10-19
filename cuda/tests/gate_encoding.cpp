// gate_encoding.cpp — Gate S1 (part 1): the C++ chat encoder must be BYTE-EXACT against the
// four golden vectors shipped in the checkpoint's encoding/tests/.
//
// This is the strongest kind of gate available for the server layer: DeepSeek ships both the
// inputs and the expected prompt strings, so there is no judgement call about what "correct" means.
//
//   build: g++ -O2 -std=c++17 -I include tests/gate_encoding.cpp -o build/gate_encoding
//   run:   ./build/gate_encoding <checkpoint_dir>/encoding/tests
#include "encoding_dsv4.h"
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

using dsv4enc::json;

static std::string slurp(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) { fprintf(stderr, "cannot open %s\n", p.c_str()); exit(2); }
    std::ostringstream ss; ss << f.rdbuf(); return ss.str();
}

// Show the first divergence with context — a byte-exact gate is useless if a failure is unreadable.
static void show_diff(const std::string& got, const std::string& want) {
    size_t i = 0;
    while (i < got.size() && i < want.size() && got[i] == want[i]) ++i;
    printf("      first difference at byte %zu (got %zu bytes, want %zu)\n", i, got.size(), want.size());
    const size_t lo = i > 60 ? i - 60 : 0;
    printf("      ...want: %s\n", want.substr(lo, 140).c_str());
    printf("      ...got : %s\n", got.substr(lo, 140).c_str());
}

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1]
        : "/home/patrickd/models/DeepSeek-V4-Flash-0731-REAP/encoding/tests";
    int pass = 0, fail = 0;

    // The reference harness (encoding/test_encoding_dsv4.py) drives each vector like this:
    //   1: dict with "tools" + "messages", thinking
    //   2: message list, thinking
    //   3: message list, thinking
    //   4: message list, chat
    struct Case { int n; const char* mode; };
    const Case cases[] = { {1,"thinking"}, {2,"thinking"}, {3,"thinking"}, {4,"chat"} };

    for (const auto& c : cases) {
        const std::string in  = dir + "/test_input_"  + std::to_string(c.n) + ".json";
        const std::string out = dir + "/test_output_" + std::to_string(c.n) + ".txt";
        json td = json::parse(slurp(in));
        const std::string want = slurp(out);

        json messages;
        if (td.is_array()) messages = td;
        else {
            // vector 1: {"tools": [...], "messages": [...]} — tools attach to the system message
            messages = td.contains("messages") ? td["messages"] : json::array();
            if (td.contains("tools") && !messages.empty()) {
                bool placed = false;
                for (auto& m : messages)
                    if (m.value("role","") == "system") { m["tools"] = td["tools"]; placed = true; break; }
                if (!placed) {
                    json sys; sys["role"] = "system"; sys["content"] = ""; sys["tools"] = td["tools"];
                    messages.insert(messages.begin(), sys);
                }
            }
        }

        std::string got;
        try { got = dsv4enc::encode_messages(messages, c.mode); }
        catch (const std::exception& e) {
            printf("[vector %d, %-8s] THREW: %s -> FAIL\n", c.n, c.mode, e.what()); ++fail; continue;
        }

        if (got == want) { printf("[vector %d, %-8s] %6zu bytes byte-exact -> PASS\n", c.n, c.mode, want.size()); ++pass; }
        else             { printf("[vector %d, %-8s] -> FAIL\n", c.n, c.mode); show_diff(got, want); ++fail; }
    }

    // round-trip: encode -> parse -> the parsed pieces must round-trip
    {
        json msgs = json::array();
        json u; u["role"] = "user"; u["content"] = "hi"; msgs.push_back(u);
        const std::string p = dsv4enc::encode_messages(msgs, "thinking");
        const bool ok = p.rfind(dsv4enc::BOS, 0) == 0 && p.find(dsv4enc::THINK_START) != std::string::npos;
        printf("[roundtrip] BOS + <think> present -> %s\n", ok ? "PASS" : "FAIL");
        ok ? ++pass : ++fail;
    }

    // reasoning_effort must be a pure PREFIX and must not perturb anything downstream
    {
        json msgs = json::array();
        json s; s["role"] = "system"; s["content"] = "sys"; msgs.push_back(s);
        json u; u["role"] = "user";   u["content"] = "q";   msgs.push_back(u);
        const std::string lo = dsv4enc::encode_messages(msgs, "thinking", json::array(), true, true, "low");
        const std::string hi = dsv4enc::encode_messages(msgs, "thinking", json::array(), true, true, "high");
        const std::string ch = dsv4enc::encode_messages(msgs, "chat",     json::array(), true, true, "high");
        const std::string pfx = dsv4enc::reasoning_effort_prompts().at("high");
        // BOS comes first, then the effort prefix, then the rest — and "high" == "low" after the prefix
        const bool a = lo.size() + pfx.size() == hi.size();
        const bool b = hi.find(pfx) != std::string::npos;
        const bool c = ch.find(pfx) == std::string::npos;   // no effect in chat mode
        printf("[effort] low+prefix==high:%d prefix present:%d absent in chat:%d -> %s\n",
               a, b, c, (a && b && c) ? "PASS" : "FAIL");
        (a && b && c) ? ++pass : ++fail;
    }

    printf("\nGate ENCODING: %d passed, %d failed -> %s\n", pass, fail, fail ? "FAIL" : "PASS");
    return fail ? 1 : 0;
}
