// server.cpp — OpenAI-compatible HTTP server for DeepSeek-V4.1-Flash-1017 REAP on Jetson Thor
// (copied from the 0731 server; encoder = encoding_dsv41.h, engine = src/v41_engine.cu).
//
// One process, one binary, no Python anywhere on the request path: httplib and json.hpp are
// header-only and vendored, the chat encoder is include/encoding_dsv4.h (byte-exact against the
// checkpoint's own encoder), the tokenizer is include/tokenizer_dsv4.h (exact against HF over
// 173k reference tokens), and the model is src/engine.cu.
//
//   GET  /                      the web UI (self-contained, no CDN)
//   GET  /health                readiness + resident context length
//   GET  /metrics               cumulative counters, Prometheus text format
//   GET  /v1/models
//   POST /v1/chat/completions   streaming and non-streaming; tools; thinking blocks
//   POST /v1/completions        raw prompt in, text out
//
// CONCURRENCY IS DELIBERATELY ONE-AT-A-TIME. The engine owns a single KV cache sized for one
// context, and on this box that is the right shape: a 122 GiB unified-memory part running a 12.26
// GB/token weight read has no headroom for a second context, and batching a decode that is
// bandwidth-bound at M=1 buys nothing. Requests serialise on `g_lock`; the queue depth is reported
// in /metrics rather than hidden.
#include "third_party/httplib.h"
#include "openai_api.h"
#include "encoding_dsv41.h"
#include "tokenizer_dsv4.h"
#include "stream_parse.h"
#include "webui_dsv4.h"
#include "dsv4_engine.h"
#include "v41_image.h"
#include "llg_constraint.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <mutex>
#include <random>
#include <string>

using dsv41enc::json;

// tokenizer.json ships in the thor build; a build without one falls back to the source checkpoint ($V41_SRC).
static std::string tokenizer_path(const std::string& ckpt) {
    const std::string a = ckpt + "/tokenizer.json";
    if (FILE* f = fopen(a.c_str(), "rb")) { fclose(f); return a; }
    const char* src = getenv("V41_SRC");
    return src ? std::string(src) + "/tokenizer.json" : a;
}

static dsv4tok::Tokenizer g_tok;
static dsv4srv::Engine*   g_eng = nullptr;
static dsv4llg::Vocab*    g_llg = nullptr;   // C3: grammar vocabulary (null: constrained requests get a 400)
static std::mutex         g_lock;
static std::string        g_model_name = "deepseek-v4.1-flash-reap";

static std::atomic<long> m_requests{0}, m_prompt_tok{0}, m_cached_tok{0}, m_completion_tok{0},
                         m_verifies{0}, m_queued{0}, m_errors{0};
static std::atomic<long long> m_decode_us{0}, m_prefill_us{0};

static long now_s() { return (long)std::time(nullptr); }

// ---- V2 image input ------------------------------------------------------------------------------
static int v41img_token_id() { return 129264; }        // <｜deepseek_image｜> (v41_config.h IMAGE_TOKEN_ID)
static std::string b64decode(const std::string& in) {
    static int T[256]; static bool init = false;
    if (!init) {
        for (int& t : T) t = -1;
        const char* a = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        for (int i = 0; i < 64; ++i) T[(unsigned char)a[i]] = i;
        T['-'] = 62; T['_'] = 63; init = true;
    }
    std::string out; int val = 0, bits = -8;
    for (unsigned char c : in) {
        if (T[c] < 0) { if (c == '=') break; continue; }
        val = (val << 6) + T[c]; bits += 6;
        if (bits >= 0) { out.push_back(char((val >> bits) & 0xFF)); bits -= 8; }
    }
    return out;
}
// an encoder image record -> bytes: data: URLs (base64), {"data"} / {"source": {"data"}} base64, and local paths
// only with V41_IMAGE_FILES=1 (a server should not read its own disk for a remote client by default)
static std::string image_bytes(const json& rec) {
    auto from_url = [](const std::string& u) -> std::string {
        if (u.rfind("data:", 0) == 0) {
            const size_t c = u.find(',');
            if (c == std::string::npos || u.substr(0, c).find(";base64") == std::string::npos)
                throw std::runtime_error("image data URL must be base64");
            return b64decode(u.substr(c + 1));
        }
        if (u.rfind("http://", 0) == 0 || u.rfind("https://", 0) == 0)
            throw std::runtime_error("remote image URLs are not fetched; send the image as a data: URL");
        const char* e = getenv("V41_IMAGE_FILES");
        if (!(e && *e == '1')) throw std::runtime_error("local image paths are disabled (V41_IMAGE_FILES=1 enables them)");
        FILE* f = fopen(u.c_str(), "rb");
        if (!f) throw std::runtime_error("cannot open image " + u);
        std::string b; char buf[65536]; size_t n;
        while ((n = fread(buf, 1, sizeof buf, f)) > 0) b.append(buf, n);
        fclose(f);
        return b;
    };
    if (rec.contains("data") && rec["data"].is_string()) return b64decode(rec["data"].get<std::string>());
    if (rec.contains("source") && rec["source"].is_object()) {
        const json& s = rec["source"];
        if (s.contains("data") && s["data"].is_string()) return b64decode(s["data"].get<std::string>());
        if (s.contains("url") && s["url"].is_string()) return from_url(s["url"].get<std::string>());
    }
    if (rec.contains("url") && rec["url"].is_string()) return from_url(rec["url"].get<std::string>());
    throw std::runtime_error("image record has no usable source");
}
// expand every image placeholder token into its span; fills gp.images (start, patches, types, hash)
static void expand_images(std::vector<int>& ids, const json& images, dsv4srv::GenParams& gp) {
    if (images.empty()) return;
    if (!g_eng->has_vision()) throw std::runtime_error("this server has no vision tower (the build has no vision.safetensors)");
    std::vector<int> out; size_t k = 0;
    for (int t : ids) {
        if (t != v41img_token_id()) { out.push_back(t); continue; }
        if (k >= images.size()) throw std::runtime_error("more image placeholders than images");
        const v41img::Prepared P = v41img::prepare(v41img::decode(image_bytes(images[k++])));
        dsv4srv::ImageSpan sp;
        sp.start = (int)out.size(); sp.patches = P.patches; sp.n_vit_h = P.n_vit_h; sp.n_vit_w = P.n_vit_w;
        sp.types = P.types; sp.hash = P.hash;
        out.insert(out.end(), P.types.size(), v41img_token_id());
        gp.images.push_back(std::move(sp));
    }
    if (k != images.size()) throw std::runtime_error("image placeholders and images differ in number");
    ids.swap(out);
}

static std::string rand_id() {
    static std::mt19937_64 rng{ (uint64_t)std::chrono::steady_clock::now().time_since_epoch().count() };
    static const char* hexd = "0123456789abcdef";
    std::string s(24, '0');
    for (auto& c : s) c = hexd[rng() & 15];
    return s;
}

// Stop strings are a text-level concept; the engine works in tokens. Rather than pretend otherwise,
// the check runs on the decoded text as it accumulates and truncates there.
static bool hit_stop(const std::string& text, const std::vector<std::string>& stops, size_t& cut) {
    for (const auto& s : stops) {
        if (s.empty()) continue;
        const size_t p = text.find(s);
        if (p != std::string::npos) { cut = p; return true; }
    }
    return false;
}

struct RunResult {
    std::string raw;                 // the full completion text
    dsv4srv::GenStats stats;
    std::vector<int> ids;            // the generated token ids (s11 traces: exact self-model greedy tokens)
};

// One generation, with the token->text->stop-string plumbing shared by both endpoints.
// `on_delta(reasoning, content)` is called as text becomes final; return false to stop.
static RunResult run_generation(const std::vector<int>& ids, const dsv4srv::GenParams& gp,
                                bool thinking, const std::vector<std::string>& stops,
                                const std::function<bool(const std::string&, const std::string&)>& on_delta,
                                dsv4llg::Matcher* gm = nullptr, const std::function<bool()>& cancelled = nullptr) {
    RunResult out;
    dsv4srv::StreamSplitter sp(thinking);
    std::vector<int> gen;
    size_t emitted = 0;
    bool stopped = false;

    out.stats = g_eng->generate(ids, gp, [&](int tok) -> bool {
        if (gm) gm->commit(tok);                        // before the engine asks for the next mask
        if (cancelled && cancelled()) return false;     // non-streaming client hung up
        // EOS ends the turn: the final parse wants it in `raw`, but it is not text -- fed to the splitter it
        // went out as the last content delta of every stream ("...}<｜end▁of▁sentence｜>")
        if (tok == g_eng->eos_id()) { out.ids.push_back(tok); sp.raw += dsv41enc::EOS; return true; }
        gen.push_back(tok);
        size_t nb = 0;
        // skip_special=FALSE on purpose: `</think>` and the DSML tool markers are added tokens, and
        // the splitter downstream is looking for exactly those.
        const std::string all = g_tok.decode_stream(gen, 0, nb, /*skip_special=*/false);
        if (nb <= emitted) return true;                 // incomplete UTF-8 so far; wait for more
        const std::string chunk = all.substr(emitted);
        emitted = nb;

        std::string r, c;
        sp.feed(chunk, r, c);

        size_t cut = 0;
        if (!stops.empty() && hit_stop(sp.raw, stops, cut)) stopped = true;
        if ((!r.empty() || !c.empty()) && on_delta && !on_delta(r, c)) stopped = true;
        return !stopped;
    });

    { std::string r, c;
      sp.finish(r, c);
      if ((!r.empty() || !c.empty()) && on_delta) on_delta(r, c); }

    out.raw = sp.raw;
    out.ids.insert(out.ids.begin(), gen.begin(), gen.end());   // + the EOS pushed above, last
    size_t cut = 0;
    if (!stops.empty() && hit_stop(out.raw, stops, cut)) out.raw = out.raw.substr(0, cut);
    return out;
}

// Serialise the per-request speculation profile as a sparse list of [width, accepted, count].
//
// The dense 16x16 is 256 cells of which a real request touches a handful -- emitting it whole
// would put a wall of zeros in every response for no information. Sparse keeps the field small
// enough to leave on by default, which is the point: this is the counter that lets a corpus
// recover the conditional accept hazard h(j) per workload, and a telemetry field nobody enables
// measures nothing. Also carries the two marginals so a consumer that only wants tau's
// decomposition does not have to reduce the joint itself.
static json spec_profile_json(const dsv4srv::SpecProfile& s) {
    json cells = json::array();
    long long tot = 0, w_sum = 0, a_sum = 0;
    for (int w = 0; w < dsv4srv::SpecProfile::MAXW; ++w)
        for (int a = 0; a < dsv4srv::SpecProfile::MAXW; ++a) {
            const int c = s.accept_joint[w][a];
            if (!c) continue;
            cells.push_back(json::array({w, a, c}));
            tot += c; w_sum += (long long)w * c; a_sum += (long long)a * c;
        }
    return json{{"accept_joint", cells},
                {"verifies", tot},
                {"mean_width", tot ? (double)w_sum / tot : 0.0},
                {"mean_accepted", tot ? (double)a_sum / tot : 0.0}};
}

static void account(const dsv4srv::GenStats& s) {
    m_prompt_tok += s.prompt_tokens;
    m_cached_tok += s.cached_tokens;
    m_completion_tok += s.completion_tokens;
    m_verifies += s.verifies;
    m_decode_us += (long long)(s.decode_ms * 1000);
    m_prefill_us += (long long)(s.prefill_ms * 1000);
}

// Dump a response that carries MODEL-GENERATED TEXT.
//
// nlohmann's default dump() THROWS type_error.316 on invalid UTF-8, and this model can produce it:
// the tokenizer is byte-level BPE, so a generation can contain byte sequences that are not valid
// UTF-8 on their own. Reproduced on GPQA-Diamond item 0 --
// "[json.exception.type_error.316] invalid UTF-8 byte at index 7053: 0x70" -- which surfaced as a
// bare 500 and destroyed two whole benchmarks before the handler was made to catch and report.
//
// `error_handler_t::replace` substitutes U+FFFD for the offending bytes instead of throwing. A
// response is not the place to be strict: the alternative to a replacement character is no answer
// at all. Valid output is byte-for-byte unaffected, since the handler only fires on bytes that
// could not have been serialised anyway.
static inline std::string dump_lossy(const json& j) {
    return j.dump(-1, ' ', false, json::error_handler_t::replace);
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    std::string ckpt = getenv("V41_CKPT") ? getenv("V41_CKPT") : ".";
    std::string host = "0.0.0.0";
    int port = 8080;
    dsv4srv::EngineConfig ec;
    ec.seqmax = 8192; ec.ext_chunk = 1024;           // prefill chunk, as v41_decode (8k: 64 303 tok/s, 1024 576; ids identical)

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "--ckpt")        ckpt = next();
        else if (a == "--host")   host = next();
        else if (a == "--port")   port = atoi(next().c_str());
        else if (a == "--seqmax") ec.seqmax = atoi(next().c_str());
        else if (a == "--engram") ec.engram_dir = next();
        else if (a == "--keepset") ec.keepset = next();
        else if (a == "--no-graph") ec.graph = false;
        else if (a == "--blk")    ec.blk = atoi(next().c_str());
        else if (a == "--adaptk") ec.adaptK = (float)atof(next().c_str());
        // Prefill now runs through the same chunked forward as a cache extension, so this is the
        // batch width of a PREFILL as well as of an extension. It has to be settable before load:
        // it sizes hv/hv2/collK/logK/mh_v and the arena, and set_ext_chunk() can only ever clamp
        // DOWN to what those were allocated for.
        else if (a == "--ext-chunk") ec.ext_chunk = atoi(next().c_str());
        else if (a == "--spec")   ec.spec_k = atoi(next().c_str());       // V4.1 DSpark drafts per verify
        else if (a == "--draft")  ec.draft_dir = next();                  // an s11 head dir ("" = ckpt mtp.*)
        else if (a == "--model")  g_model_name = next();
        else if (a == "--help") {
            printf("usage: %s [--ckpt DIR] [--host H] [--port P] [--seqmax N] [--blk N] [--adaptk F]"
                   " [--ext-chunk N] [--engram DIR] [--no-graph] [--spec K [--draft DIR]]\n", argv[0]);
            return 0;
        }
    }
    ec.ckpt_dir = ckpt;

    printf("[server] tokenizer %s\n", tokenizer_path(ckpt).c_str());
    g_tok.load(tokenizer_path(ckpt));
    try { g_llg = new dsv4llg::Vocab(tokenizer_path(ckpt), &g_tok); }
    catch (const std::exception& e) { fprintf(stderr, "[server] grammar disabled: %s\n", e.what()); }
    { // The same gate the rest of the repo uses. If this ever fails, every prompt is silently wrong.
        const std::vector<int> want{671, 6102, 294, 8760, 344};
        if (g_tok.encode("The capital of France is") != want) {
            fprintf(stderr, "[server] FATAL: tokenizer gate failed\n"); return 1; }
    }
    printf("[server] tokenizer ok (vocab %zu, %zu added)\n", g_tok.vocab.size(), g_tok.added.size());

    dsv4srv::Engine eng(ec);
    g_eng = &eng;
    eng.load();

    httplib::Server srv;
    srv.new_task_queue = [] { return new httplib::ThreadPool(4); };

    auto cors = [](httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization");
        res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    };
    srv.Options(".*", [&](const httplib::Request&, httplib::Response& res) { cors(res); });

    srv.Get("/", [&](const httplib::Request&, httplib::Response& res) {
        res.set_content(WEBUI_DSV4_HTML, "text/html; charset=utf-8");
    });

    srv.Get("/health", [&](const httplib::Request&, httplib::Response& res) {
        cors(res);
        json o;
        o["status"] = "ok";
        o["model"] = g_model_name;
        o["context_len"] = g_eng->context_len();
        o["seqmax"] = g_eng->seqmax();
        o["busy"] = m_queued.load() > 0;
        res.set_content(o.dump(), "application/json");
    });

    srv.Get("/metrics", [&](const httplib::Request&, httplib::Response& res) {
        cors(res);
        char buf[2048];
        const double dsec = m_decode_us.load() / 1e6;
        snprintf(buf, sizeof buf,
            "# HELP dsv4_requests_total Completed requests.\n# TYPE dsv4_requests_total counter\n"
            "dsv4_requests_total %ld\n"
            "dsv4_errors_total %ld\n"
            "dsv4_prompt_tokens_total %ld\n"
            "dsv4_cached_prompt_tokens_total %ld\n"
            "dsv4_completion_tokens_total %ld\n"
            "dsv4_verifies_total %ld\n"
            "dsv4_decode_seconds_total %.3f\n"
            "dsv4_prefill_seconds_total %.3f\n"
            "dsv4_decode_tokens_per_second %.3f\n"
            "dsv4_tokens_per_verify %.3f\n"
            "dsv4_prefix_cache_hit_ratio %.4f\n"
            "dsv4_queue_depth %ld\n",
            m_requests.load(), m_errors.load(), m_prompt_tok.load(), m_cached_tok.load(),
            m_completion_tok.load(), m_verifies.load(), dsec, m_prefill_us.load() / 1e6,
            dsec > 0 ? m_completion_tok.load() / dsec : 0.0,
            m_verifies.load() > 0 ? (double)m_completion_tok.load() / m_verifies.load() : 0.0,
            m_prompt_tok.load() > 0 ? (double)m_cached_tok.load() / m_prompt_tok.load() : 0.0,
            m_queued.load());
        res.set_content(buf, "text/plain; version=0.0.4");
    });

    srv.Get("/v1/models", [&](const httplib::Request&, httplib::Response& res) {
        cors(res);
        res.set_content(dsv4api::models_response(g_model_name, now_s()).dump(), "application/json");
    });

    // ---- POST /v1/chat/completions ----------------------------------------------------------
    srv.Post("/v1/chat/completions", [&](const httplib::Request& req, httplib::Response& res) {
        cors(res);
        json body;
        try { body = json::parse(req.body); }
        catch (const std::exception& e) {
            ++m_errors;
            res.status = 400;
            res.set_content(json{{"error", {{"message", std::string("invalid JSON: ") + e.what()},
                                            {"type", "invalid_request_error"}}}}.dump(), "application/json");
            return;
        }

        dsv4api::ChatRequest cr;
        std::string prompt;
        json images = json::array();
        try {
            cr = dsv4api::parse_chat_request(body);
            prompt = dsv4api::build_prompt(cr, &images);
        } catch (const std::exception& e) {
            ++m_errors;
            res.status = 400;
            res.set_content(json{{"error", {{"message", e.what()}, {"type", "invalid_request_error"}}}}.dump(),
                            "application/json");
            return;
        }

        std::vector<int> ids = g_tok.encode(prompt);
        const bool thinking = (cr.thinking_mode == "thinking");
        const std::string id = rand_id();
        const long created = now_s();

        dsv4srv::GenParams gp;
        try { expand_images(ids, images, gp); }
        catch (const std::exception& e) {
            ++m_errors;
            res.status = 400;
            res.set_content(json{{"error", {{"message", e.what()}, {"type", "invalid_request_error"}}}}.dump(), "application/json");
            return;
        }
        gp.temperature = (float)cr.sampling.temperature;
        gp.top_p = (float)cr.sampling.top_p;
        gp.max_tokens = cr.sampling.max_tokens;
        gp.seed = cr.sampling.seed;
        gp.has_seed = cr.sampling.has_seed;

        // C3: response_format json_object / json_schema, or the `grammar` extension (lark / regex / json)
        std::shared_ptr<dsv4llg::Matcher> gm;
        try {
            const auto c = dsv4llg::request_constraint(body);
            if (!c.first.empty()) {
                if (!g_llg) throw std::invalid_argument("constrained output is unavailable (llguidance vocabulary failed to load)");
                gm = std::make_shared<dsv4llg::Matcher>(*g_llg, c.first, c.second, thinking);
                gp.mask = [gm](uint32_t* bits) { return gm->mask(bits, dsv4srv::kMaskWords); };
            }
        } catch (const std::exception& e) {
            ++m_errors;
            res.status = 400;
            res.set_content(json{{"error", {{"message", e.what()}, {"type", "invalid_request_error"}}}}.dump(), "application/json");
            return;
        }

        if ((int)ids.size() + gp.max_tokens + 8 > g_eng->seqmax()) {
            ++m_errors;
            res.status = 400;
            char m[256];
            snprintf(m, sizeof m, "prompt (%zu tokens) + max_tokens (%d) exceeds context %d",
                     ids.size(), gp.max_tokens, g_eng->seqmax());
            res.set_content(json{{"error", {{"message", m}, {"type", "context_length_exceeded"}}}}.dump(),
                            "application/json");
            return;
        }

        if (!cr.stream) {
            std::lock_guard<std::mutex> lk(g_lock);
            ++m_requests;
            // THIS PATH HAD NO try/catch AND THE STREAMING ONE DID. Any throw out of
            // run_generation or parse_message_from_completion_text therefore escaped the handler
            // and httplib turned it into a bare 500 with an EMPTY BODY -- no message, no log line,
            // nothing to debug from. GPQA-Diamond item 0 hit it reproducibly: not context (122
            // prompt tokens), not concurrency (a parallel request succeeds), not the prompt (the
            // same text at max_tokens=64 is fine), only at max_tokens >= 2000, i.e. once the
            // generation is long enough to emit whatever construct the parser rejects. A whole
            // 198-item benchmark was lost to it, and the empty body is why it took three sessions
            // to find. Catch it, say what it was, and log it.
            try {
                const RunResult r = run_generation(ids, gp, thinking, cr.sampling.stop, nullptr, gm.get(),
                                                   [&req] { return req.is_connection_closed(); });
                account(r.stats);
                const json parsed = dsv41enc::parse_message_from_completion_text(r.raw, cr.thinking_mode);
                json out = dsv4api::chat_completion_response(id, cr.model, parsed,
                                                            r.stats.prompt_tokens, r.stats.completion_tokens, created);
                out["usage"]["prompt_tokens_details"] = json{{"cached_tokens", r.stats.cached_tokens}};
                out["timings"] = json{{"prefill_ms", r.stats.prefill_ms},
                                      {"decode_ms", r.stats.decode_ms},
                                      {"tokens_per_second", r.stats.tok_per_s},
                                      {"tokens_per_verify", r.stats.tok_per_verify}};
                out["spec_profile"] = spec_profile_json(r.stats.spec);
                if (gm && !gm->error().empty()) out["grammar_error"] = gm->error();
                res.set_content(dump_lossy(out), "application/json");
            } catch (const std::exception& e) {
                ++m_errors;
                fprintf(stderr, "[server] generation failed (%zu prompt tokens, max_tokens %d): %s\n",
                        ids.size(), gp.max_tokens, e.what());
                fflush(stderr);
                res.status = 500;
                res.set_content(json{{"error", {{"message", e.what()},
                                                {"type", "generation_error"}}}}.dump(),
                                "application/json");
            }
            return;
        }

        // ---- streaming (SSE) ----
        res.set_header("Cache-Control", "no-cache");
        res.set_header("Connection", "keep-alive");
        res.set_header("X-Accel-Buffering", "no");
        res.set_chunked_content_provider("text/event-stream",
            [id, created, ids, gp, thinking, cr, gm](size_t, httplib::DataSink& sink) -> bool {
                std::lock_guard<std::mutex> lk(g_lock);
                ++m_requests;
                bool alive = true;
                auto send = [&](const std::string& s) {
                    if (!alive) return;
                    if (!sink.write(s.data(), s.size())) alive = false;   // client hung up
                };
                // A first chunk carrying only the role is what OpenAI clients expect, and it also
                // flushes headers so a slow first token does not look like a dead connection.
                { json d{{"role", "assistant"}};
                  json ch{{"index", 0}, {"delta", d}, {"finish_reason", nullptr}};
                  json o{{"id", "chatcmpl-" + id}, {"object", "chat.completion.chunk"},
                         {"created", created}, {"model", cr.model}, {"choices", json::array({ch})}};
                  send("data: " + dump_lossy(o) + "\n\n"); }

                RunResult r;
                try {
                    r = run_generation(ids, gp, thinking, cr.sampling.stop,
                        [&](const std::string& rr, const std::string& cc) -> bool {
                            if (!rr.empty() || !cc.empty())
                                send(dsv4api::sse_chunk(id, cr.model, created, cc, rr, nullptr));
                            return alive;
                        }, gm.get());
                } catch (const std::exception& e) {
                    ++m_errors;
                    send(std::string("data: ") + json{{"error", {{"message", e.what()}}}}.dump() + "\n\n");
                    sink.done();
                    return true;
                }
                account(r.stats);

                // Tool calls arrive as one final delta: each DSML parameter's string="true|false"
                // decides its JSON type, so the block cannot be interpreted until it is complete.
                const json parsed = dsv41enc::parse_message_from_completion_text(r.raw, cr.thinking_mode);
                const bool has_tc = parsed.contains("tool_calls") && !parsed["tool_calls"].empty();
                if (has_tc) {
                    json tcs = json::array();
                    int i = 0;
                    for (const auto& tc : parsed["tool_calls"]) {
                        json o = tc;
                        o["index"] = i;
                        o["id"] = "call_" + id.substr(0, 8) + "_" + std::to_string(i);
                        ++i;
                        tcs.push_back(o);
                    }
                    json ch{{"index", 0}, {"delta", json{{"tool_calls", tcs}}}, {"finish_reason", nullptr}};
                    json o{{"id", "chatcmpl-" + id}, {"object", "chat.completion.chunk"},
                           {"created", created}, {"model", cr.model}, {"choices", json::array({ch})}};
                    send("data: " + dump_lossy(o) + "\n\n");
                }
                send(dsv4api::sse_chunk(id, cr.model, created, "", "", has_tc ? "tool_calls" : "stop"));

                // usage on the final chunk (OpenAI's stream_options.include_usage shape), plus the
                // timings a local server is actually useful for.
                { json o{{"id", "chatcmpl-" + id}, {"object", "chat.completion.chunk"},
                         {"created", created}, {"model", cr.model}, {"choices", json::array()},
                         {"usage", json{{"prompt_tokens", r.stats.prompt_tokens},
                                        {"completion_tokens", r.stats.completion_tokens},
                                        {"total_tokens", r.stats.prompt_tokens + r.stats.completion_tokens},
                                        {"prompt_tokens_details", json{{"cached_tokens", r.stats.cached_tokens}}}}},
                         {"timings", json{{"prefill_ms", r.stats.prefill_ms},
                                          {"decode_ms", r.stats.decode_ms},
                                          {"tokens_per_second", r.stats.tok_per_s},
                                          {"tokens_per_verify", r.stats.tok_per_verify}}},
                         {"spec_profile", spec_profile_json(r.stats.spec)}};
                  send("data: " + dump_lossy(o) + "\n\n"); }
                send("data: [DONE]\n\n");
                sink.done();
                return true;
            });
    });

    // ---- POST /v1/completions ---------------------------------------------------------------
    srv.Post("/v1/completions", [&](const httplib::Request& req, httplib::Response& res) {
        cors(res);
        json body;
        try { body = json::parse(req.body); }
        catch (const std::exception& e) {
            ++m_errors; res.status = 400;
            res.set_content(json{{"error", {{"message", std::string("invalid JSON: ") + e.what()}}}}.dump(),
                            "application/json");
            return;
        }
        // EVERYTHING past this point is wrapped. The non-streaming chat handler learned this the
        // expensive way: an exception escaping the lambda closes the connection with an empty body,
        // the client sees an unexplained HTTP 500, and a benchmark scores every remaining item
        // wrong -- that bug cost an entire 198-item GPQA run. This path had never been given the
        // same treatment, and the budget-extension runs (tools/eval_extend.py) drive their whole
        // ~20-hour workload through it.
        try {
        std::string prompt = body.value("prompt", "");
        dsv4srv::GenParams gp;
        gp.temperature = body.contains("temperature") && body["temperature"].is_number()
                       ? (float)body["temperature"].get<double>() : 1.0f;
        gp.top_p = body.contains("top_p") && body["top_p"].is_number() ? (float)body["top_p"].get<double>() : 1.0f;
        gp.max_tokens = body.contains("max_tokens") && body["max_tokens"].is_number() ? body["max_tokens"].get<int>() : 128;
        if (body.contains("seed") && body["seed"].is_number()) { gp.seed = body["seed"].get<uint64_t>(); gp.has_seed = true; }
        std::vector<std::string> stops;
        if (body.contains("stop")) {
            if (body["stop"].is_string()) stops.push_back(body["stop"].get<std::string>());
            else if (body["stop"].is_array()) for (auto& x : body["stop"]) if (x.is_string()) stops.push_back(x.get<std::string>());
        }
        // A raw completion is a raw completion: BOS, then the prompt, no chat template at all.
        std::vector<int> ids = g_tok.encode(prompt);
        ids.insert(ids.begin(), g_tok.bos_id);

        std::lock_guard<std::mutex> lk(g_lock);
        ++m_requests;
        const RunResult r = run_generation(ids, gp, false, stops, nullptr);
        account(r.stats);
        json choice{{"index", 0}, {"text", r.raw}, {"finish_reason", "stop"}, {"logprobs", nullptr}};
        json out{{"id", "cmpl-" + rand_id()}, {"object", "text_completion"}, {"created", now_s()},
                 {"model", g_model_name}, {"choices", json::array({choice})},
                 {"usage", json{{"prompt_tokens", r.stats.prompt_tokens},
                                {"completion_tokens", r.stats.completion_tokens},
                                {"total_tokens", r.stats.prompt_tokens + r.stats.completion_tokens}}}};
        // TIMINGS ON THIS PATH ARE NOT COSMETIC -- THEY ARE THE ONLY WINDOW INTO DEEP CONTEXT.
        // /v1/chat/completions has always returned them; this endpoint never did, and it is the
        // endpoint tools/eval_extend.py drives. Every continuation leg the project has ever run --
        // the ONLY decode above the 8k base budget, i.e. the entire 8k-24k regime -- therefore
        // recorded nothing, and tools/decode_model.py can fit decode cost against context only out
        // to ~6.6k. The context-linear term is already 59% of a forward there and rising, so the
        // regime that matters most is the one that was invisible. Same fields as the chat path so
        // the two are directly comparable.
        out["timings"] = json{{"prefill_ms", r.stats.prefill_ms},
                              {"decode_ms", r.stats.decode_ms},
                              {"tokens_per_second", r.stats.tok_per_s},
                              {"tokens_per_verify", r.stats.tok_per_verify}};
        out["spec_profile"] = spec_profile_json(r.stats.spec);
        // opt-in: the exact prompt and generated ids (ops/s11_spec.py traces). Text round-trips are
        // not id-exact, and a draft head is scored on ids.
        if (body.value("return_ids", false)) { out["prompt_ids"] = ids; out["completion_ids"] = r.ids; }
        res.set_content(dump_lossy(out), "application/json");
        } catch (const std::exception& e) {
            ++m_errors;
            res.status = 500;
            res.set_content(json{{"error", {{"message", std::string("completion failed: ") + e.what()},
                                            {"type", "server_error"}}}}.dump(), "application/json");
        } catch (...) {
            ++m_errors;
            res.status = 500;
            res.set_content(json{{"error", {{"message", "completion failed: unknown exception"},
                                            {"type", "server_error"}}}}.dump(), "application/json");
        }
    });

    srv.set_pre_routing_handler([](const httplib::Request&, httplib::Response&) {
        ++m_queued;
        return httplib::Server::HandlerResponse::Unhandled;
    });
    srv.set_post_routing_handler([](const httplib::Request&, httplib::Response&) { --m_queued; });

    printf("\n[server] listening on http://%s:%d   (web UI at /)\n", host.c_str(), port);
    printf("[server] model %s, context %d, block %d, adaptK %.2f, spec K %d%s%s\n",
           g_model_name.c_str(), ec.seqmax, ec.blk, ec.adaptK, ec.spec_k,
           ec.draft_dir.empty() ? "" : ", draft ", ec.draft_dir.c_str());
    if (!srv.listen(host.c_str(), port)) {
        fprintf(stderr, "[server] FATAL: cannot bind %s:%d\n", host.c_str(), port);
        return 1;
    }
    return 0;
}
