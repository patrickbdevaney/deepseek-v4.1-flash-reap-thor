// gate_encoding_v41.cpp — the V4.1 chat encoder must be BYTE-EXACT against the golden vectors in the
// checkpoint's encoding/tests (driven exactly as test_encoding.py drives them: load_cases + encode_case
// with default thinking_mode "chat"), and the strict parser must accept the reference's tool-call output.
//
//   build: g++ -O2 -std=c++17 -I include tests/gate_encoding_v41.cpp -o build/gate_encoding_v41
//   run:   ./build/gate_encoding_v41 ~/models/DeepSeek-V4.1-Flash-1017/encoding/tests
#include "encoding_dsv41.h"
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

using dsv41enc::json;

static std::string slurp(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) { fprintf(stderr, "cannot open %s\n", p.c_str()); exit(2); }
    std::ostringstream ss; ss << f.rdbuf(); return ss.str();
}
static void show_diff(const std::string& got, const std::string& want) {
    size_t i = 0;
    while (i < got.size() && i < want.size() && got[i] == want[i]) ++i;
    printf("      first difference at byte %zu (got %zu bytes, want %zu)\n", i, got.size(), want.size());
    const size_t lo = i > 60 ? i - 60 : 0;
    printf("      ...want: %s\n      ...got : %s\n", want.substr(lo, 140).c_str(), got.substr(lo, 140).c_str());
}

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : "/home/patrickd/models/DeepSeek-V4.1-Flash-1017/encoding/tests";
    int pass = 0, fail = 0;
    for (int n = 1;; ++n) {
        std::ifstream probe(dir + "/test_input_" + std::to_string(n) + ".json");
        if (!probe) break;
        json d = json::parse(slurp(dir + "/test_input_" + std::to_string(n) + ".json"));
        const std::string want = slurp(dir + "/test_output_" + std::to_string(n) + ".txt");
        // load_cases
        json c = d.is_object() ? d : json::object();
        if (d.is_array()) c["messages"] = d;
        json msgs = c["messages"];
        if (c.contains("tools")) msgs[0]["tools"] = c["tools"];
        const std::string mode = c.contains("thinking_mode") && c["thinking_mode"].is_string() ? c["thinking_mode"].get<std::string>() : "chat";
        const json effort = c.contains("reasoning_effort") ? c["reasoning_effort"] : json();
        const json ctx = c.contains("context") && c["context"].is_array() ? c["context"] : json::array();
        std::string got; json imgs;
        try { got = dsv41enc::encode_messages(msgs, mode, ctx, true, true, effort, &imgs); }
        catch (const std::exception& e) { printf("[vector %d %-8s] THREW %s -> FAIL\n", n, mode.c_str(), e.what()); ++fail; continue; }
        if (got == want) { printf("[vector %d %-8s] %6zu bytes byte-exact, %zu image(s) -> PASS\n", n, mode.c_str(), want.size(), imgs.size()); ++pass; }
        else { printf("[vector %d %-8s] -> FAIL\n", n, mode.c_str()); show_diff(got, want); ++fail; }
    }
    // parse: the reference test's V41_TOOL_CALL_OUTPUT
    {
        const std::string out =
            "  reason  </think>summary\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"lookup\">\n"
            "<｜DSML｜ parameter name=\"query\" string=\"true\">value</｜DSML｜ parameter>\n"
            "<｜DSML｜ parameter name=\"limit\" string=\"false\">2</｜DSML｜ parameter>\n"
            "</｜DSML｜ invoke>\n</｜DSML｜ calls><｜end▁of▁sentence｜>";
        bool ok = false;
        try {
            json p = dsv41enc::parse_message_from_completion_text(out, "thinking", true);
            ok = p["reasoning_content"] == "  reason  " && p["content"] == "summary" && p["tool_calls"].size() == 1 &&
                 p["tool_calls"][0]["function"]["name"] == "lookup" &&
                 p["tool_calls"][0]["function"]["arguments"] == "{\"query\": \"value\", \"limit\": 2}";
            // round trip: re-render the parsed message as an assistant turn and get the same bytes back
            json m = p; m["tool_calls"][0]["function"]["arguments"] = json::parse(p["tool_calls"][0]["function"]["arguments"].get<std::string>());
            json conv = json::array({json{{"role", "user"}, {"content", "q"}}, m});
            const std::string r = dsv41enc::render_message(1, conv, "thinking");
            ok = ok && r == out;
            if (!ok) show_diff(r, out);
        } catch (const std::exception& e) { printf("  parse threw: %s\n", e.what()); }
        printf("[parse tool-call output strict + round-trip] -> %s\n", ok ? "PASS" : "FAIL");
        ok ? ++pass : ++fail;
    }
    // differential: random conversations encoded by the reference (tools/encoding_v41_fixtures.py)
    {
        std::string fz = argc > 2 ? argv[2] : "tests/data/encoding_v41_fuzz.json";
        std::ifstream probe(fz);
        if (probe) {
            json cs = json::parse(slurp(fz));
            int ok = 0, bad = 0;
            for (size_t k = 0; k < cs.size(); ++k) {
                const json& c = cs[k];
                std::string got;
                try {
                    got = dsv41enc::encode_messages(c["messages"], c["thinking_mode"].get<std::string>(),
                                                    c["context"].is_array() ? c["context"] : json::array(),
                                                    c["drop_thinking"].get<bool>(), true, c["reasoning_effort"]);
                } catch (const std::exception& e) { got = std::string("THREW ") + e.what(); }
                if (got == c["expected"].get<std::string>()) ++ok;
                else { if (bad < 3) { printf("  fuzz case %zu differs\n", k); show_diff(got, c["expected"].get<std::string>()); } ++bad; }
            }
            printf("[reference fuzz %zu cases] %d byte-exact -> %s\n", cs.size(), ok, bad ? "FAIL" : "PASS");
            bad ? ++fail : ++pass;
        } else printf("[reference fuzz] %s missing -- SKIPPED (run tools/encoding_v41_fixtures.py)\n", fz.c_str());
    }
    printf("%d pass, %d fail\n", pass, fail);
    return fail ? 1 : 0;
}
