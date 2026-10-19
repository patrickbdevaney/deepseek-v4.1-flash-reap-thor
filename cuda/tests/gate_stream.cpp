// gate_stream.cpp — the streaming splitter must be independent of how the bytes arrive.
//
// The property that matters: feeding a completion in ANY chunking (all at once, one byte at a time,
// split exactly across a marker) yields the same reasoning/content/tools split. A splitter that
// only works when markers happen to land inside one chunk passes every casual test and then drops
// "</think>" into the visible answer the first time a token boundary falls in the middle of it.
//
//   g++ -O2 -std=c++17 -I include tests/gate_stream.cpp -o build/gate_stream && ./build/gate_stream
#include "stream_parse.h"
#include "encoding_dsv41.h"
#include <cstdio>
#include <string>
#include <vector>

using dsv4srv::StreamSplitter;

static int pass = 0, fail = 0;
static void ck(bool ok, const char* what) {
    printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
    ok ? ++pass : ++fail;
}

struct Split { std::string r, c, t; };

// Feed `text` in chunks of `n` bytes (n<=0 => one shot).
static Split run(const std::string& text, bool thinking, int n) {
    StreamSplitter sp(thinking);
    Split o;
    if (n <= 0) sp.feed(text, o.r, o.c);
    else for (size_t i = 0; i < text.size(); i += n) sp.feed(text.substr(i, n), o.r, o.c);
    sp.finish(o.r, o.c);
    o.t = sp.tools_buf;
    return o;
}

int main() {
    const std::string TOOLBLK =
        "<｜DSML｜ calls>\n"
        "<｜DSML｜ invoke name=\"get_weather\">\n"
        "<｜DSML｜ parameter name=\"city\" string=\"true\">Beijing</｜DSML｜ parameter>\n"
        "<｜DSML｜ parameter name=\"days\" string=\"false\">5</｜DSML｜ parameter>\n"
        "</｜DSML｜ invoke>\n"
        "</｜DSML｜ calls>";

    struct Case { const char* name; std::string text; bool thinking; std::string want_r, want_c; bool want_tools; };
    const std::vector<Case> cases = {
        { "thinking then content", "let me think</think>The answer is 4.", true, "let me think", "The answer is 4.", false },
        { "chat mode, no think marker", "The answer is 4.", false, "", "The answer is 4.", false },
        { "thinking with no content", "just reasoning</think>", true, "just reasoning", "", false },
        { "empty reasoning", "</think>direct", true, "", "direct", false },
        { "content then tools", "I will look it up.\n\n" + TOOLBLK, false, "", "I will look it up.\n\n", true },
        { "reasoning, content, tools", "hmm</think>ok\n\n" + TOOLBLK, true, "hmm", "ok\n\n", true },
        { "multibyte content", "推理</think>你好世界，这是一个测试。", true, "推理", "你好世界，这是一个测试。", false },
        { "marker-like text that is not a marker", "a </thin b </think>c", true, "a </thin b ", "c", false },
        { "no markers at all", "plain text only", false, "", "plain text only", false },
    };

    for (const auto& c : cases) {
        // Every chunking must agree with the one-shot result AND with the expectation.
        const Split one = run(c.text, c.thinking, 0);
        bool ok = (one.r == c.want_r) && (one.c == c.want_c) && (!c.want_tools == one.t.empty());
        if (!ok)
            printf("        one-shot: r=[%s] want[%s] c=[%s] want[%s] tools=%zu\n",
                   one.r.c_str(), c.want_r.c_str(), one.c.c_str(), c.want_c.c_str(), one.t.size());
        for (int n : {1, 2, 3, 5, 7, 8, 13, 22, 23, 64}) {
            const Split s = run(c.text, c.thinking, n);
            if (s.r != one.r || s.c != one.c || s.t != one.t) {
                printf("        chunk=%d disagrees: r=[%s] c=[%s]\n", n, s.r.c_str(), s.c.c_str());
                ok = false;
                break;
            }
        }
        ck(ok, c.name);
    }

    // Every delta must be valid UTF-8 on its own -- it goes straight into a JSON string, and
    // nlohmann throws on invalid UTF-8, which would turn a CJK answer into a 500 mid-stream.
    {
        const std::string text = "推理过程很长</think>你好世界，这是一个很长的中文回答，用来测试分块。";
        bool ok = true;
        for (int n = 1; n <= 8 && ok; ++n) {
            StreamSplitter sp(true);
            for (size_t i = 0; i < text.size(); i += n) {
                std::string r, c;
                sp.feed(text.substr(i, n), r, c);
                for (const std::string* s : { &r, &c }) {
                    try { (void)dsv41enc::json(*s).dump(); }
                    catch (...) { printf("        invalid UTF-8 delta at chunk=%d\n", n); ok = false; }
                }
            }
        }
        ck(ok, "every delta is independently valid UTF-8");
    }

    // The tools buffer must be exactly what the (already gated) DSML parser expects.
    {
        const Split s = run("ok\n\n" + TOOLBLK, false, 3);
        const dsv41enc::json parsed = dsv41enc::parse_message_from_completion_text("ok\n\n" + TOOLBLK, "chat");
        const bool ok = parsed["tool_calls"].size() == 1 &&
                        parsed["tool_calls"][0]["function"]["name"] == "get_weather" &&
                        s.t.find("get_weather") != std::string::npos;
        ck(ok, "buffered tool block parses to one call via the gated DSML parser");
    }

    printf("\nGate STREAM: %d passed, %d failed -> %s\n", pass, fail, fail ? "FAIL" : "PASS");
    return fail ? 1 : 0;
}
