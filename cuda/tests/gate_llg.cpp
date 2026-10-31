// gate_llg.cpp — C3 constrained output, CPU only: the llguidance vocabulary over our tokenizer, request
// parsing, and a token walk through each constraint (every token of a valid answer allowed, a wrong first
// token masked, EOS allowed only once the answer is complete, free text while thinking).
//   g++ -O2 -std=c++17 -I include tests/gate_llg.cpp third_party/llg/target/release/libllg_ffi.a -ldl -lpthread -lm
//   ./gate_llg <tokenizer.json>
#include "llg_constraint.h"
#include "dsv4_engine.h"
#include <chrono>
#include <cstdio>

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { ++fails; printf("FAIL %s:%d ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

static bool allowed(const std::vector<uint32_t>& b, int t) { return (b[t >> 5] >> (t & 31)) & 1u; }

// walk `text` (tokenised) through a matcher; return false at the first masked token
static bool walk(dsv4llg::Matcher& m, const dsv4tok::Tokenizer& tok, const std::string& text, bool want_eos) {
    std::vector<uint32_t> bits(dsv4srv::kMaskWords);
    for (int t : tok.encode(text)) {
        std::fill(bits.begin(), bits.end(), 0u);
        if (m.mask(bits.data(), bits.size()) && !allowed(bits, t)) { printf("  masked token %d in \"%s\"\n", t, text.c_str()); return false; }
        m.commit(t);
    }
    std::fill(bits.begin(), bits.end(), 0u);
    const bool constrained = m.mask(bits.data(), bits.size());
    if (!constrained) return !want_eos;
    return allowed(bits, tok.eos_id) == want_eos;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: gate_llg <tokenizer.json>\n"); return 2; }
    dsv4tok::Tokenizer tok; tok.load(argv[1]);
    dsv4llg::Vocab v(argv[1], &tok);
    printf("think_end id %d\n", v.think_end_id());
    CHECK(v.think_end_id() > 0, "</think> is not one token");
    using dsv4llg::json;

    // request parsing
    auto c = dsv4llg::request_constraint(json::parse(R"({"response_format":{"type":"json_object"}})"));
    CHECK(c.first == "json", "json_object -> %s", c.first.c_str());
    CHECK(dsv4llg::request_constraint(json::parse(R"({"response_format":{"type":"text"}})")).first.empty(), "text");
    CHECK(dsv4llg::request_constraint(json::parse(R"({})")).first.empty(), "none");
    bool threw = false;
    try { dsv4llg::request_constraint(json::parse(R"({"response_format":{"type":"json_schema"}})")); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw, "json_schema without schema accepted");
    threw = false;
    try { dsv4llg::Matcher bad(v, "lark", "start: (", false); } catch (const std::invalid_argument& e) { threw = true; printf("bad lark -> %s\n", e.what()); }
    CHECK(threw, "bad lark accepted");

    // json_schema walk
    const auto sc = dsv4llg::request_constraint(json::parse(R"({"response_format":{"type":"json_schema","json_schema":{"name":"x","schema":
        {"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}},"required":["name","age"],"additionalProperties":false}}}})"));
    { dsv4llg::Matcher m(v, sc.first, sc.second, false);
      CHECK(walk(m, tok, R"({"name": "Ada", "age": 36})", true), "valid schema answer rejected or EOS not allowed at end"); }
    { dsv4llg::Matcher m(v, sc.first, sc.second, false);
      CHECK(walk(m, tok, R"({"name": "Ada")", false), "EOS allowed mid-object"); }
    { dsv4llg::Matcher m(v, sc.first, sc.second, false);
      std::vector<uint32_t> bits(dsv4srv::kMaskWords);
      CHECK(m.mask(bits.data(), bits.size()), "not constrained at start");
      const int hello = tok.encode("Hello")[0];
      CHECK(!allowed(bits, hello), "prose token allowed at start of a JSON answer");
      CHECK(!allowed(bits, tok.eos_id), "EOS allowed before any JSON");
      int n = 0; for (auto w : bits) n += __builtin_popcount(w);
      printf("tokens allowed at start: %d\n", n); }
    // a wrong key is refused somewhere along the walk
    { dsv4llg::Matcher m(v, sc.first, sc.second, false);
      CHECK(!walk(m, tok, R"({"nom": "Ada", "age": 36})", true), "wrong key accepted"); }

    // thinking: free until </think>, then constrained
    { dsv4llg::Matcher m(v, "regex", "(yes|no)", true);
      std::vector<uint32_t> bits(dsv4srv::kMaskWords);
      CHECK(!m.mask(bits.data(), bits.size()), "constrained while thinking");
      for (int t : tok.encode("Let me think about it carefully.")) m.commit(t);
      CHECK(!m.mask(bits.data(), bits.size()), "constrained before </think>");
      m.commit(v.think_end_id());
      CHECK(walk(m, tok, "yes", true), "regex answer after </think> rejected"); }

    // lark extension
    { const auto lc = dsv4llg::request_constraint(json::parse(R"({"grammar":{"type":"lark","data":"start: \"A\" | \"B\" | \"C\""}})"));
      dsv4llg::Matcher m(v, lc.first, lc.second, false);
      CHECK(walk(m, tok, "B", true), "lark choice rejected");
      dsv4llg::Matcher m2(v, lc.first, lc.second, false);
      CHECK(!walk(m2, tok, "D", true), "lark accepted D"); }

    // mask latency over a long schema answer (budget: < 0.4% of a 40 ms decode step = 160 us)
    { const auto ac = dsv4llg::request_constraint(json::parse(R"({"response_format":{"type":"json_schema","json_schema":{"schema":
          {"type":"object","properties":{"items":{"type":"array","items":{"type":"object","properties":{"id":{"type":"integer"},
          "label":{"type":"string"},"score":{"type":"number"}},"required":["id","label","score"]}}},"required":["items"]}}}})"));
      std::string ans = R"({"items": [)";
      for (int i = 0; i < 40; ++i) ans += (i ? ", " : "") + std::string(R"({"id": )") + std::to_string(i) + R"(, "label": "item number )" + std::to_string(i) + R"( of the list", "score": 0.)" + std::to_string(i * 7 % 100) + "}";
      ans += "]}";
      dsv4llg::Matcher m(v, ac.first, ac.second, false);
      std::vector<uint32_t> bits(dsv4srv::kMaskWords);
      const auto ids = tok.encode(ans);
      double tot = 0, mx = 0; bool ok = true;
      for (int t : ids) {
          std::fill(bits.begin(), bits.end(), 0u);
          const auto t0 = std::chrono::steady_clock::now();
          m.mask(bits.data(), bits.size());
          const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
          tot += us; mx = std::max(mx, us);
          if (!allowed(bits, t)) ok = false;
          m.commit(t);
      }
      CHECK(ok, "long schema answer masked");
      printf("mask latency over %zu tokens: mean %.1f us, max %.1f us\n", ids.size(), tot / ids.size(), mx);
      CHECK(tot / ids.size() < 400, "mean mask latency %.1f us", tot / ids.size()); }

    printf(fails ? "GATE FAIL (%d)\n" : "GATE PASS\n", fails);
    return fails ? 1 : 0;
}
