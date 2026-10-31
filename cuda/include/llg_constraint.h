// llg_constraint.h — C3 grammar-constrained output for the v41 server, on llguidance (MIT; built as
// third_party/llg/target/release/libllg_ffi.a by scripts/build_v41.sh).
//
// Request surface (OpenAI-compatible, plus one extension):
//   response_format {"type":"json_object"}                         -> any JSON object
//   response_format {"type":"json_schema","json_schema":{"schema":S}} -> JSON matching S
//   grammar {"type":"lark"|"regex"|"json","data":...}               -> extension (llguidance syntax)
// The constraint covers the answer only: in thinking mode it starts after the model's `</think>`, so the
// reasoning is free text. The engine applies the mask on the GPU (GenParams::mask) and runs plain AR for
// the request.
#pragma once
#include "third_party/llguidance.h"
#include "third_party/json.hpp"
#include "tokenizer_dsv4.h"
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace dsv4llg {

using json = nlohmann::ordered_json;

// The vocabulary, once per process: tokenizer.json for the token bytes (llguidance marks the added tokens
// special), our own BPE for re-tokenising forced text.
class Vocab {
public:
    Vocab(const std::string& tokenizer_json_path, const dsv4tok::Tokenizer* tok) : tok_(tok) {
        std::ifstream f(tokenizer_json_path, std::ios::binary);
        if (!f) throw std::runtime_error("llg: cannot read " + tokenizer_json_path);
        std::stringstream ss; ss << f.rdbuf(); tj_ = ss.str();
        LlgTokenizerInit init{};
        init.tok_eos = (LlgToken)tok->eos_id;
        init.tokenizer_json = tj_.c_str();
        init.tokenize_fn = &Vocab::tokenize;
        init.tokenize_user_data = this;
        char err[1024] = {0};
        t_ = llg_new_tokenizer(&init, err, sizeof err);
        if (!t_) throw std::runtime_error(std::string("llg tokenizer: ") + err);
    }
    ~Vocab() { if (t_) llg_free_tokenizer(t_); }
    Vocab(const Vocab&) = delete; Vocab& operator=(const Vocab&) = delete;
    const LlgTokenizer* get() const { return t_; }
    int think_end_id() const {
        const auto ids = tok_->encode("</think>");
        return ids.size() == 1 ? ids[0] : -1;
    }

private:
    static size_t tokenize(const void* ud, const uint8_t* b, size_t n, uint32_t* out, size_t cap) {
        std::vector<int> ids;
        static_cast<const Vocab*>(ud)->tok_->encode_run(std::string((const char*)b, n), ids);
        for (size_t i = 0; i < ids.size() && i < cap; ++i) out[i] = (uint32_t)ids[i];
        return ids.size();
    }
    const dsv4tok::Tokenizer* tok_;
    std::string tj_;
    LlgTokenizer* t_ = nullptr;
};

// Pick the constraint a request asks for: {} (none) or {type, data}. Throws std::invalid_argument on a
// malformed spec (-> 400).
inline std::pair<std::string, std::string> request_constraint(const json& body) {
    if (body.contains("grammar") && !body["grammar"].is_null()) {
        const json& g = body["grammar"];
        const std::string t = g.value("type", "lark");
        if (t != "lark" && t != "regex" && t != "json" && t != "llguidance")
            throw std::invalid_argument("grammar.type must be lark, regex, json or llguidance");
        if (!g.contains("data")) throw std::invalid_argument("grammar.data missing");
        return {t, g["data"].is_string() ? g["data"].get<std::string>() : g["data"].dump()};
    }
    if (!body.contains("response_format") || !body["response_format"].is_object()) return {};
    const json& rf = body["response_format"];
    const std::string t = rf.value("type", "text");
    if (t == "text") return {};
    if (t == "json_object") return {"json", R"({"type":"object"})"};
    if (t == "json_schema") {
        const json& js = rf.contains("json_schema") ? rf["json_schema"] : json();
        if (!js.is_object() || !js.contains("schema")) throw std::invalid_argument("response_format.json_schema.schema missing");
        return {"json", js["schema"].dump()};
    }
    throw std::invalid_argument("response_format.type must be text, json_object or json_schema");
}

// One request's matcher. mask() fills the engine's bit buffer; commit() follows every emitted token.
class Matcher {
public:
    Matcher(const Vocab& v, const std::string& type, const std::string& data, bool thinking)
        : active_(!thinking), think_end_(v.think_end_id()) {
        LlgConstraintInit init;
        llg_constraint_init_set_defaults(&init, v.get());
        init.ff_tokens_ok = false; init.backtrack_ok = false;
        m_ = llg_new_matcher(&init, type.c_str(), data.c_str());
        if (!m_ || llg_matcher_is_error(m_)) {
            const std::string e = m_ ? llg_matcher_get_error(m_) : "null matcher";
            if (m_) llg_free_matcher(m_);
            m_ = nullptr;
            throw std::invalid_argument("grammar: " + e);
        }
        buf_.resize(llg_matcher_get_mask_byte_size(m_) / 4);
    }
    ~Matcher() { if (m_) llg_free_matcher(m_); }
    Matcher(const Matcher&) = delete; Matcher& operator=(const Matcher&) = delete;

    // false = leave this token free (still thinking, or the matcher failed: the error is reported)
    bool mask(uint32_t* bits, size_t words) {
        if (!active_ || failed_) return false;
        if (llg_matcher_compute_mask_into(m_, buf_.data(), buf_.size() * 4) != 0) { fail(); return false; }
        std::copy(buf_.begin(), buf_.begin() + std::min(words, buf_.size()), bits);
        return true;
    }
    void commit(int tok) {
        if (!active_) { if (tok == think_end_) active_ = true; return; }
        if (failed_ || llg_matcher_is_stopped(m_)) return;
        if (llg_matcher_consume_token(m_, (uint32_t)tok) != 0) fail();
    }
    bool accepting() { return m_ && !failed_ && llg_matcher_is_accepting(m_); }
    const std::string& error() const { return err_; }

private:
    void fail() { failed_ = true; const char* e = llg_matcher_get_error(m_); err_ = e ? e : "matcher error"; }
    LlgMatcher* m_ = nullptr;
    std::vector<uint32_t> buf_;
    bool active_, failed_ = false;
    int think_end_;
    std::string err_;
};

} // namespace dsv4llg
