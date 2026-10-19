// tokenizer_dsv4.h — pure-C++ DeepSeek-V4 tokenizer. Loads the checkpoint's own tokenizer.json.
//
// WHY NOT REUSE include/tokenizer.h. That file is gemma's, and gemma's tokenizer is a different
// animal: SentencePiece-style, ' ' -> U+2581, byte_fallback for unknown chars, a no-op
// pre-tokenizer. DeepSeek-V4 is **ByteLevel BPE** (GPT-2 lineage): every byte maps into a private
// unicode alphabet, byte_fallback is off because it cannot be needed, and a four-stage
// pre-tokenizer splits the text before BPE ever runs. Adapting gemma's by habit produces token ids
// that are wrong but plausible -- the worst possible failure for a decode server. Written against
// tokenizer.json, gated against HF in tests/gate_tokenizer.cpp.
//
// PIPELINE (tokenizer.json, verbatim):
//   normalizer   Sequence[]                                   -- empty, a genuine no-op
//   pre_tokenizer Sequence[
//      Split(Regex "\p{N}{1,3}",              Isolated)        -- digits in groups of <=3
//      Split(Regex "[CJK/kana ranges]+",      Isolated)
//      Split(Regex <the big alternation>,     Isolated)
//      ByteLevel(add_prefix_space=false, use_regex=false)      -- byte -> unicode, no splitting
//   ]
//   model        BPE(dropout=null, unk=null, byte_fallback=false)
//   decoder      ByteLevel
// added_tokens (1283) are matched literally on the RAW text, before any of the above.
#pragma once
#include "third_party/json.hpp"
#include "unicode_cat.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <fstream>
#include <cstdint>
#include <climits>

namespace dsv4tok {

// ---- UTF-8 <-> codepoints -------------------------------------------------------------------
inline int u8len(unsigned char c) {
    return c < 0x80 ? 1 : (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xE ? 3 : (c >> 3) == 0x1E ? 4 : 1;
}
inline void u8_decode(const std::string& s, std::vector<uint32_t>& cp, std::vector<int>& off) {
    cp.clear(); off.clear();
    for (size_t i = 0; i < s.size();) {
        const int L = u8len((unsigned char)s[i]);
        uint32_t c = 0;
        if (L == 1) c = (unsigned char)s[i];
        else if (L == 2) c = ((s[i] & 0x1F) << 6) | (s[i+1] & 0x3F);
        else if (L == 3) c = ((s[i] & 0x0F) << 12) | ((s[i+1] & 0x3F) << 6) | (s[i+2] & 0x3F);
        else             c = ((s[i] & 0x07) << 18) | ((s[i+1] & 0x3F) << 12) | ((s[i+2] & 0x3F) << 6) | (s[i+3] & 0x3F);
        cp.push_back(c); off.push_back((int)i);
        i += L;
    }
    off.push_back((int)s.size());
}
inline void u8_append(std::string& s, uint32_t c) {
    if (c < 0x80) s.push_back((char)c);
    else if (c < 0x800) { s.push_back((char)(0xC0 | (c >> 6))); s.push_back((char)(0x80 | (c & 0x3F))); }
    else if (c < 0x10000) { s.push_back((char)(0xE0 | (c >> 12))); s.push_back((char)(0x80 | ((c >> 6) & 0x3F))); s.push_back((char)(0x80 | (c & 0x3F))); }
    else { s.push_back((char)(0xF0 | (c >> 18))); s.push_back((char)(0x80 | ((c >> 12) & 0x3F))); s.push_back((char)(0x80 | ((c >> 6) & 0x3F))); s.push_back((char)(0x80 | (c & 0x3F))); }
}

// ---- character classes used by the pre-tokenizer ---------------------------------------------
// \s. Unicode White_Space=Yes, spelled out: the set is small, fixed, and hardcoding it is more
// honest than pulling a sixth generated table for six ranges.
inline bool is_space(uint32_t c) {
    return (c >= 0x09 && c <= 0x0D) || c == 0x20 || c == 0x85 || c == 0xA0 || c == 0x1680 ||
           (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 || c == 0x202F ||
           c == 0x205F || c == 0x3000;
}
inline bool is_nl(uint32_t c) { return c == '\r' || c == '\n'; }
inline bool is_alpha_ascii(uint32_t c) { return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z'); }
// The literal ASCII punctuation class that opens alternative A.
inline bool is_punct_ascii(uint32_t c) {
    return (c >= '!' && c <= '/') || (c >= ':' && c <= '@') || (c >= '[' && c <= '`') || (c >= '{' && c <= '~');
}
inline bool is_cjk(uint32_t c) {
    return (c >= 0x4E00 && c <= 0x9FA5) || (c >= 0x3040 && c <= 0x309F) || (c >= 0x30A0 && c <= 0x30FF);
}

// ---- the GPT-2 byte <-> unicode alphabet ------------------------------------------------------
struct ByteLevel {
    uint32_t b2u[256];
    std::unordered_map<uint32_t, int> u2b;
    ByteLevel() {
        std::vector<int> bs;
        for (int b = 33; b < 127; ++b) bs.push_back(b);
        for (int b = 161; b < 173; ++b) bs.push_back(b);
        for (int b = 174; b < 256; ++b) bs.push_back(b);
        std::vector<uint32_t> cs(bs.begin(), bs.end());
        int n = 0;
        for (int b = 0; b < 256; ++b)
            if (std::find(bs.begin(), bs.end(), b) == bs.end()) { bs.push_back(b); cs.push_back(256 + n++); }
        for (size_t i = 0; i < bs.size(); ++i) { b2u[bs[i]] = cs[i]; u2b[cs[i]] = bs[i]; }
    }
    std::string encode(const std::string& raw) const {
        std::string o;
        for (unsigned char ch : raw) u8_append(o, b2u[ch]);
        return o;
    }
    // Inverse. Unmapped codepoints cannot occur in vocab strings, so they are dropped rather than
    // guessed at.
    std::string decode(const std::string& mapped) const {
        std::vector<uint32_t> cp; std::vector<int> off;
        u8_decode(mapped, cp, off);
        std::string o;
        for (uint32_t c : cp) { auto it = u2b.find(c); if (it != u2b.end()) o.push_back((char)it->second); }
        return o;
    }
};

// ---- pre-tokenizer ----------------------------------------------------------------------------
// HF's SplitDelimiterBehavior::Isolated keeps BOTH the gaps and the matches as separate pieces.
// A Sequence of pre-tokenizers applies each stage to every piece the previous stage produced.

// Stage 3's alternation, tried in source order at each position with ordinary backtracking
// semantics. Returns the match length in CODEPOINTS, or -1.
//   A  [ascii punct][A-Za-z]+
//   B  [^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+
//   C   ?[\p{P}\p{S}]+[\r\n]*
//   D  \s*[\r\n]+
//   E  \s+(?!\S)
//   F  \s+
inline int stage3_match(const std::vector<uint32_t>& c, int i) {
    const int n = (int)c.size();
    // A
    if (is_punct_ascii(c[i])) {
        int j = i + 1;
        while (j < n && is_alpha_ascii(c[j])) ++j;
        if (j > i + 1) return j - i;
    }
    // B — the optional leading char is greedy, so try it consumed first, then zero-width.
    for (int opt = 1; opt >= 0; --opt) {
        int j = i;
        if (opt) {
            if (j >= n || is_nl(c[j]) || uc_is_L(c[j]) || uc_is_P(c[j]) || uc_is_S(c[j])) continue;
            ++j;
        }
        int k = j;
        while (k < n && (uc_is_L(c[k]) || uc_is_M(c[k]))) ++k;
        if (k > j) return k - i;
    }
    // C
    {
        int j = i;
        if (c[j] == ' ') ++j;
        int k = j;
        while (k < n && (uc_is_P(c[k]) || uc_is_S(c[k]))) ++k;
        if (k > j) {
            while (k < n && is_nl(c[k])) ++k;
            return k - i;
        }
    }
    // D — \s* is greedy, then backtrack to the last position where [\r\n]+ can start.
    if (is_space(c[i])) {
        int run = i;
        while (run < n && is_space(c[run])) ++run;      // one past the whitespace run
        for (int s = run; s >= i; --s) {
            if (s < run && !is_nl(c[s])) continue;      // \s* must be followed by [\r\n]
            if (s >= n || !is_nl(c[s])) continue;
            int k = s;
            while (k < n && is_nl(c[k])) ++k;
            return k - i;
        }
        // E — \s+ greedy, backtracked until the next char is not \S.
        const int k = run - i;                          // maximal \s+ length, >= 1
        if (run >= n) return k;                         // at EOF the lookahead is satisfied
        if (k >= 2) return k - 1;                       // else give the last space back
        // F
        return k;
    }
    return -1;
}

// Split `s` on every match of `match(cp,i)`, keeping gaps and matches (Isolated).
template <typename F>
inline void split_isolated(const std::string& s, F&& match, std::vector<std::string>& out) {
    std::vector<uint32_t> cp; std::vector<int> off;
    u8_decode(s, cp, off);
    const int n = (int)cp.size();
    int gap = 0;                                   // codepoint index where the current gap started
    for (int i = 0; i < n;) {
        const int L = match(cp, i);
        if (L <= 0) { ++i; continue; }
        if (i > gap) out.push_back(s.substr(off[gap], off[i] - off[gap]));
        out.push_back(s.substr(off[i], off[i + L] - off[i]));
        i += L; gap = i;
    }
    if (gap < n) out.push_back(s.substr(off[gap]));
}

inline int stage1_match(const std::vector<uint32_t>& c, int i) {   // \p{N}{1,3}
    int j = i, n = (int)c.size();
    while (j < n && j - i < 3 && uc_is_N(c[j])) ++j;
    return j - i;
}
inline int stage2_match(const std::vector<uint32_t>& c, int i) {   // [CJK]+
    int j = i, n = (int)c.size();
    while (j < n && is_cjk(c[j])) ++j;
    return j - i;
}

// ---- the tokenizer -----------------------------------------------------------------------------
struct Tokenizer {
    std::unordered_map<std::string, int> vocab;                 // ByteLevel-space token -> id
    std::vector<std::string> id2tok;
    std::vector<char> id_is_added;                              // added tokens decode verbatim
    std::unordered_map<uint64_t, std::pair<int,int>> merges;    // (a,b) -> (rank, merged)
    std::vector<std::pair<std::string,int>> added;              // (content,id), longest-first
    // Added tokens bucketed by first byte. Scanning all 1283 patterns at every offset is O(1283*n)
    // per segment, and once the prefix cache removes the GPU work from a cached turn, tokenisation
    // IS the latency floor -- so the scan checks only the patterns that could start here.
    std::vector<std::vector<int>> added_by_first{256};
    ByteLevel bl;

    int bos_id = 0, eos_id = 1, pad_id = 2;

    static uint64_t pk(int a, int b) { return ((uint64_t)(uint32_t)a << 32) | (uint32_t)b; }

    void load(const std::string& path) {
        std::ifstream f(path);
        if (!f) throw std::runtime_error("tokenizer: cannot open " + path);
        nlohmann::json j; f >> j;

        auto& v = j["model"]["vocab"];
        int maxid = 0;
        for (auto it = v.begin(); it != v.end(); ++it) maxid = std::max(maxid, it.value().get<int>());
        if (j.contains("added_tokens"))
            for (auto& a : j["added_tokens"]) maxid = std::max(maxid, a["id"].get<int>());
        id2tok.assign(maxid + 1, std::string());
        id_is_added.assign(maxid + 1, 0);
        for (auto it = v.begin(); it != v.end(); ++it) {
            const int id = it.value().get<int>();
            vocab[it.key()] = id;
            id2tok[id] = it.key();
        }
        int rank = 0;
        for (auto& m : j["model"]["merges"]) {
            std::string a, b;
            if (m.is_array()) { a = m[0].get<std::string>(); b = m[1].get<std::string>(); }
            else { const std::string s = m.get<std::string>(); const size_t sp = s.find(' '); a = s.substr(0, sp); b = s.substr(sp + 1); }
            auto ia = vocab.find(a), ib = vocab.find(b), ic = vocab.find(a + b);
            if (ia != vocab.end() && ib != vocab.end() && ic != vocab.end())
                merges[pk(ia->second, ib->second)] = { rank, ic->second };
            ++rank;
        }
        if (j.contains("added_tokens")) for (auto& a : j["added_tokens"]) {
            const std::string c = a["content"].get<std::string>();
            const int id = a["id"].get<int>();
            added.push_back({ c, id });
            id2tok[id] = c;
            id_is_added[id] = 1;
            if (c == "<｜begin▁of▁sentence｜>") bos_id = id;
            else if (c == "<｜end▁of▁sentence｜>") eos_id = id;
            else if (c == "<｜▁pad▁｜>") pad_id = id;
        }
        // Longest-first so a literal match at a position takes the longest added token there.
        std::sort(added.begin(), added.end(),
                  [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });
        for (int i = 0; i < (int)added.size(); ++i)
            if (!added[i].first.empty())
                added_by_first[(unsigned char)added[i].first[0]].push_back(i);
    }

    // BPE over one pre-token, already in ByteLevel space.
    void bpe(const std::string& piece, std::vector<int>& out) const {
        std::vector<uint32_t> cp; std::vector<int> off;
        u8_decode(piece, cp, off);
        std::vector<int> s; s.reserve(cp.size());
        for (size_t i = 0; i < cp.size(); ++i) {
            const std::string ch = piece.substr(off[i], off[i+1] - off[i]);
            auto it = vocab.find(ch);
            if (it != vocab.end()) s.push_back(it->second);   // ByteLevel guarantees this hits
        }
        while (s.size() >= 2) {
            int bestK = -1, bestRank = INT_MAX, bestMerged = -1;
            for (size_t k = 0; k + 1 < s.size(); ++k) {
                auto m = merges.find(pk(s[k], s[k+1]));
                if (m != merges.end() && m->second.first < bestRank) {
                    bestRank = m->second.first; bestK = (int)k; bestMerged = m->second.second;
                }
            }
            if (bestK < 0) break;
            s[bestK] = bestMerged;
            s.erase(s.begin() + bestK + 1);
        }
        out.insert(out.end(), s.begin(), s.end());
    }

    // One added-token-free text run: the four pre-tokenizer stages, then BPE.
    void encode_run(const std::string& text, std::vector<int>& out) const {
        if (text.empty()) return;
        std::vector<std::string> a{ text }, b;
        for (auto& p : a) split_isolated(p, stage1_match, b);
        a.swap(b); b.clear();
        for (auto& p : a) split_isolated(p, stage2_match, b);
        a.swap(b); b.clear();
        for (auto& p : a) split_isolated(p, stage3_match, b);
        for (auto& p : b) bpe(bl.encode(p), out);
    }

    std::vector<int> encode(const std::string& text, bool add_bos = false) const {
        std::vector<int> out;
        if (add_bos) out.push_back(bos_id);
        size_t run = 0;                                      // start of the pending plain-text run
        for (size_t i = 0; i < text.size(); ++i) {
            const auto& cand = added_by_first[(unsigned char)text[i]];
            if (cand.empty()) continue;
            // `added` is sorted longest-first, so the first hit at this position is the longest.
            for (int ai : cand) {
                const std::string& s = added[ai].first;
                if (i + s.size() > text.size()) continue;
                if (text.compare(i, s.size(), s) != 0) continue;
                if (i > run) encode_run(text.substr(run, i - run), out);
                out.push_back(added[ai].second);
                i += s.size() - 1;                           // -1: the loop's ++i lands after it
                run = i + 1;
                break;
            }
        }
        if (run < text.size()) encode_run(text.substr(run), out);
        return out;
    }

    std::string decode(const std::vector<int>& ids, bool skip_special = true) const {
        std::string mapped;                                  // ByteLevel-space run
        std::string out;
        for (int id : ids) {
            if (id < 0 || id >= (int)id2tok.size()) continue;
            if (id_is_added[id]) {
                out += bl.decode(mapped); mapped.clear();    // flush before a verbatim token
                if (!skip_special) out += id2tok[id];
                continue;
            }
            mapped += id2tok[id];
        }
        out += bl.decode(mapped);
        return out;
    }

    // Streaming-safe: decode ids[from..] but return only the prefix that is complete UTF-8, so a
    // multi-byte character split across two tokens is never emitted as a broken byte.
    //
    // `skip_special` defaults to true, but a SERVER must pass false. `</think>` and the `｜DSML｜`
    // tool markers are added tokens, so skipping specials deletes precisely the markers the
    // streaming splitter and the tool-call parser exist to find -- reasoning then arrives glued
    // into content and tool calls arrive as unparseable markup. Nothing errors; it just silently
    // stops working. The engine never delivers EOS to its callback, so keeping specials does not
    // leak a stop token into the output.
    std::string decode_stream(const std::vector<int>& ids, size_t from, size_t& consumed_bytes,
                              bool skip_special = true) const {
        std::vector<int> tail(ids.begin() + from, ids.end());
        const std::string s = decode(tail, skip_special);
        size_t end = s.size();
        while (end > 0) {                                    // back off any incomplete final char
            size_t st = end - 1;
            while (st > 0 && ((unsigned char)s[st] & 0xC0) == 0x80) --st;
            const int need = u8len((unsigned char)s[st]);
            if (st + need <= end) break;
            end = st;
        }
        consumed_bytes = end;
        return s.substr(0, end);
    }
};

} // namespace dsv4tok
