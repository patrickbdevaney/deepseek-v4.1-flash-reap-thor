// v41_decode.cu — command-line greedy generation and tok/s bench over the V4.1 engine.
//
//   build/v41_decode --ckpt DIR [--keepset s04/keepset.json] [--engram DIR] [--prompt TEXT | --chat TEXT [--thinking] | --ids 1,2,3 | --ids-file F]
//                    [--n 128] [--seqmax 8192] [--chunk 1024] [--no-graph] [--sw-mxfp4] [--temp T --seed S]
//                    [--repeat R] [--json OUT] [--logits OUT]
//                    [--spec K [--draft HEAD_DIR] [--spec-sweep 1,2,3,4]]
//
// --json writes {"prompt_ids", "ids", "prefill_ms", "decode_tok_s", ...}: the chain's greedy gate
// compares "ids" with the reference model.py run on the same prompt (tests/golden/ref_dump.py).
// --logits writes the first-step logits (fp32, 129280) for the teacher-forced comparison.
// --repeat R re-runs the same prompt R times WITHOUT the prefix cache: decode tok/s per run and the
// spread, and a determinism check (every run must emit the same ids at temperature 0).
// --consistency K: for K positions k spread over the greedy output, re-prefill prompt + out[:k] from
// scratch and require argmax == out[k]. The decode path (M=1, graph, incremental caches) must agree
// with the prefill path (chunked, decoder-span skipping) -- the free-running half of the golden gate.
// A mismatch prints both logits' top-2 margin: a near-tie flip is reassociation, a gross one is a bug.
//
// --spec K: DSpark speculation with K drafts per verify (the checkpoint's own mtp.* blocks, or the s11
// head dir --draft; --draft-depth K is the same flag). The reported run is the spec run.
// --spec-sweep 1,2,3,4 (needs --spec >= the largest): one load, then a no-spec run and one run per K on
// the same prompt. Each K's greedy ids must equal the no-spec ids ("identity"); the first mismatch is
// re-prefilled and classified like --consistency (near-tie = reassociation at verify width M=K+1 vs
// M=1, GROSS = a bug). JSON "spec": {"nospec": {...}, "k": [{k, tok_s, tok_per_verify, accept_hist,
// identical, first_diff, diff_class, speedup}], "identity_gross"}. Exit 1 on any GROSS mismatch.
// --trace-gen F --trace-dir DIR --spec K (drafter training data, D2): per jsonl line {id, domain, mode
// raw|chat|thinking, ids | content}, the greedy continuation (--n tokens, speculative K: same ids), then
// one teacher-forced prefill of prompt + continuation for the drafter taps of rows [gen_start - 128, end).
// Appends DIR/taps.bin (bf16 [T, 3, dim]), DIR/tokens.bin (int32 [T]), DIR/seqs.jsonl {id, domain, mode,
// offset, length, gen_start, prompt_tokens}; resumes after the last complete seqs line. ops/zenith/d2_pack.py
// turns it into s11's taps_<set>/ (L37/L38/L39.npy, tokens.npy, meta.json).
// --chat-file F (ops/bench/heldout_prompts.jsonl): every line's prompt in one load -- greedy no-spec, then
// each --spec-sweep K; per-prompt and per-domain tok/s and tok/verify, rows to --json. The held-out set
// for acceptance and head-shortlist work; never the speed prompt.
#include "dsv4_engine.h"
#include "v41_config.h"
#include "encoding_dsv41.h"
#include "tokenizer_dsv4.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <map>
#include <unistd.h>
#include <sstream>
#include <string>
#include <vector>

using dsv41enc::json;

// tokenizer.json ships in the thor build; a build without one falls back to the source checkpoint ($V41_SRC).
static std::string tokenizer_path(const std::string& ckpt) {
    const std::string a = ckpt + "/tokenizer.json";
    if (FILE* f = fopen(a.c_str(), "rb")) { fclose(f); return a; }
    const char* src = getenv("V41_SRC");
    return src ? std::string(src) + "/tokenizer.json" : a;
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    dsv4srv::EngineConfig ec;
    ec.ckpt_dir = getenv("V41_CKPT") ? getenv("V41_CKPT") : ".";
    ec.seqmax = 8192; ec.ext_chunk = 1024;           // prefill chunk (8k, P8: 512 349, 1024 458 @ 1.97 GiB state, 2048 476 @ 3.76 GiB; ids identical)
    std::string prompt, chat, ids_arg, json_out, logits_out, score_in, score_out, score_nll, chat_file, trace_gen, trace_dir, route_trace;
    bool thinking = false;
    int n = 128, repeat = 1, consistency = 0;
    std::vector<int> sweep;
    dsv4srv::GenParams gp;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "--ckpt") ec.ckpt_dir = next();
        else if (a == "--engram") ec.engram_dir = next();
        else if (a == "--keepset") ec.keepset = next();
        else if (a == "--prompt") prompt = next();
        else if (a == "--chat") chat = next();
        else if (a == "--chat-file") chat_file = next();
        else if (a == "--trace-gen") trace_gen = next();
        else if (a == "--trace-dir") trace_dir = next();
        else if (a == "--route-trace") route_trace = next();
        else if (a == "--thinking") thinking = true;
        else if (a == "--ids") ids_arg = next();
        else if (a == "--ids-file") { std::ifstream f(next()); std::stringstream ss; ss << f.rdbuf(); ids_arg = ss.str(); }
        else if (a == "--n") n = atoi(next().c_str());
        else if (a == "--seqmax") ec.seqmax = atoi(next().c_str());
        else if (a == "--chunk") ec.ext_chunk = atoi(next().c_str());
        else if (a == "--no-graph") ec.graph = false;
        else if (a == "--sw-mxfp4") ec.mxfp4_hw = false;
        else if (a == "--temp") gp.temperature = (float)atof(next().c_str());
        else if (a == "--top-p") gp.top_p = (float)atof(next().c_str());
        else if (a == "--seed") { gp.seed = strtoull(next().c_str(), nullptr, 10); gp.has_seed = true; }
        else if (a == "--repeat") repeat = atoi(next().c_str());
        else if (a == "--consistency") consistency = atoi(next().c_str());
        else if (a == "--json") json_out = next();
        else if (a == "--spec" || a == "--draft-depth") ec.spec_k = std::max(ec.spec_k, atoi(next().c_str()));
        else if (a == "--draft") ec.draft_dir = next();
        else if (a == "--spec-sweep") {
            std::stringstream ss(next()); std::string t;
            while (std::getline(ss, t, ',')) if (!t.empty()) sweep.push_back(atoi(t.c_str()));
            for (int k : sweep) ec.spec_k = std::max(ec.spec_k, k);
        }
        else if (a == "--logits") logits_out = next();
        else if (a == "--score") score_in = next();
        else if (a == "--score-out") score_out = next();
        else if (a == "--score-nll") score_nll = next();   // with --score: -log p per forced token (f32) instead of logits
        else { fprintf(stderr, "unknown arg %s (see the header of src/v41_decode.cu)\n", a.c_str()); return 2; }
    }
    gp.max_tokens = n;

    dsv4tok::Tokenizer tok;
    tok.load(tokenizer_path(ec.ckpt_dir));
    std::vector<int> ids;
    if (!ids_arg.empty()) {
        std::stringstream ss(ids_arg); std::string t;
        while (std::getline(ss, t, ',')) {
            std::stringstream w(t); std::string u;
            while (w >> u) ids.push_back(atoi(u.c_str()));          // commas and/or whitespace
        }
    } else {
        std::string text = prompt;
        if (!chat.empty()) {
            json msgs = json::array({json{{"role", "user"}, {"content", chat}}});
            text = dsv41enc::encode_messages(msgs, thinking ? "thinking" : "chat");
            ids = tok.encode(text);                  // BOS is a literal added token in the encoded text
        } else {
            if (text.empty()) text = "The capital of France is";
            ids.push_back(0);                         // BOS
            for (int t : tok.encode(text)) ids.push_back(t);
        }
    }
    printf("[v41] prompt %zu tokens\n", ids.size());

    dsv4srv::Engine eng(ec);
    eng.load();
    if (!route_trace.empty()) {                       // M3 probe: prefill every D2 trace seq with V41_ROUTE_DUMP set
        if (!getenv("V41_ROUTE_DUMP")) { fprintf(stderr, "--route-trace needs V41_ROUTE_DUMP=FILE\n"); return 2; }
        std::ifstream fk(route_trace + "/tokens.bin", std::ios::binary), fs(route_trace + "/seqs.jsonl");
        fk.seekg(0, std::ios::end); std::vector<int> all(fk.tellg() / 4); fk.seekg(0); fk.read((char*)all.data(), all.size() * 4);
        std::string l; int n = 0;
        while (std::getline(fs, l)) {
            if (l.empty()) continue;
            const json j = json::parse(l);
            const long o = j["offset"], len = j["length"];
            std::vector<int> sid(all.begin() + o, all.begin() + o + len);
            eng.debug_prefix_logits(sid); eng.reset();
            printf("[route] %s %s %ld\n", j["id"].get<std::string>().c_str(), j["domain"].get<std::string>().c_str(), len);
            fflush(stdout); ++n;
        }
        printf("[route] %d seqs\n", n);
        return 0;
    }
    if (!score_in.empty() && !score_nll.empty()) {
        std::ifstream fi(score_in); std::vector<int> forced; int v; while (fi >> v) forced.push_back(v);
        std::vector<float> tlp; std::vector<int> tid;
        const std::vector<float> nl = eng.debug_forced_nll(ids, forced, &tlp, &tid);
        std::ofstream f(score_nll, std::ios::binary);
        f.write((const char*)nl.data(), nl.size() * 4);
        std::ofstream(score_nll + ".lp", std::ios::binary).write((const char*)tlp.data(), tlp.size() * 4);   // [n, 32] f32
        std::ofstream(score_nll + ".id", std::ios::binary).write((const char*)tid.data(), tid.size() * 4);   // [n, 32] i32
        double s = 0; for (float x : nl) s += x;
        printf("[v41] forced %zu tokens: mean NLL %.5f -> %s\n", nl.size(), s / std::max<size_t>(nl.size(), 1), score_nll.c_str());
        return 0;
    }
    if (!score_in.empty()) {                          // whitespace-separated forced ids -> raw f32 logits
        std::ifstream fi(score_in); std::vector<int> forced; int v; while (fi >> v) forced.push_back(v);
        const std::vector<float> lg = eng.debug_forced_logits(ids, forced);
        std::ofstream f(score_out, std::ios::binary);
        f.write((const char*)lg.data(), lg.size() * 4);
        printf("[v41] forced %zu tokens -> %s\n", forced.size(), score_out.c_str());
        return 0;
    }
    if (!logits_out.empty()) {
        const std::vector<float> lg = eng.debug_prefix_logits(ids);
        std::ofstream f(logits_out, std::ios::binary);
        f.write((const char*)lg.data(), lg.size() * 4);
        printf("[v41] first-step logits -> %s\n", logits_out.c_str());
        eng.reset();
    }
    eng.set_prefix_cache(false);
    if (!trace_gen.empty()) {
        if (trace_dir.empty() || ec.spec_k < 1) { fprintf(stderr, "--trace-gen needs --trace-dir and --spec K\n"); return 2; }
        const int row = 3 * v41::DIM, W = v41::WINDOW;
        std::map<std::string, int> done; long T = 0;
        { std::ifstream fs(trace_dir + "/seqs.jsonl"); std::string l;
          while (std::getline(fs, l)) { if (l.empty()) continue; const json j = json::parse(l); done[j["id"]] = 1;
                                        T = std::max(T, j["offset"].get<long>() + j["length"].get<long>()); } }
        // drop a partly written tail (a seq whose meta line never landed)
        if (truncate((trace_dir + "/taps.bin").c_str(), (off_t)T * row * 2) && T) { perror("truncate taps.bin"); return 2; }
        if (truncate((trace_dir + "/tokens.bin").c_str(), (off_t)T * 4) && T) { perror("truncate tokens.bin"); return 2; }
        std::ofstream ft(trace_dir + "/taps.bin", std::ios::binary | std::ios::app), fk(trace_dir + "/tokens.bin", std::ios::binary | std::ios::app),
                      fm(trace_dir + "/seqs.jsonl", std::ios::app);
        std::ifstream fi(trace_gen); std::string line; int cnt = 0; double gen_tok = 0, gen_s = 0;
        while (std::getline(fi, line)) {
            if (line.empty()) continue;
            const json q = json::parse(line);
            const std::string id = q["id"], mode = q.value("mode", std::string("chat"));
            if (done.count(id)) continue;
            std::vector<int> pid;
            if (q.contains("ids")) pid = q["ids"].get<std::vector<int>>();
            else pid = tok.encode(dsv41enc::encode_messages(json::array({json{{"role", "user"}, {"content", q["content"]}}}),
                                                            mode == "thinking" ? "thinking" : "chat"));
            std::vector<int> out;
            eng.set_spec_k(ec.spec_k); eng.reset();
            const dsv4srv::GenStats g = eng.generate(pid, gp, [&](int t) { out.push_back(t); return true; });
            std::vector<int> all(pid); all.insert(all.end(), out.begin(), out.end());
            const int gs = (int)pid.size(), from = std::max(0, gs - W);
            const std::vector<uint16_t> taps = eng.debug_taps(all, from);
            const int len = (int)all.size() - from;
            ft.write(reinterpret_cast<const char*>(taps.data()), (std::streamsize)taps.size() * 2);
            fk.write(reinterpret_cast<const char*>(all.data() + from), (std::streamsize)len * 4);
            ft.flush(); fk.flush();
            if (!ft || !fk) { fprintf(stderr, "[trace] write failed\n"); return 2; }
            fm << json{{"id", id}, {"domain", q.value("domain", std::string("?"))}, {"mode", mode}, {"offset", T}, {"length", len},
                       {"gen_start", gs - from}, {"prompt_tokens", gs}, {"gen_tokens", (int)out.size()}, {"tok_s", g.tok_per_s},
                       {"tok_per_verify", g.tok_per_verify}}.dump() << "\n";
            fm.flush();
            T += len; ++cnt; gen_tok += out.size(); gen_s += g.decode_ms / 1e3;
            if (cnt % 10 == 1) printf("[trace] %d done (+%zu resumed): %s %s %d+%zu tok, %.1f tok/s, %.2f/verify, T=%ld\n", cnt, done.size(),
                                      id.c_str(), mode.c_str(), gs, out.size(), g.tok_per_s, g.tok_per_verify, T), fflush(stdout);
        }
        printf("[trace] %d new seqs, %.0f gen tok at %.1f tok/s; T=%ld rows\n", cnt, gen_tok, gen_s > 0 ? gen_tok / gen_s : 0.0, T);
        return 0;
    }
    if (!chat_file.empty()) {
        // held-out set: one jsonl line per prompt {id, domain, content[, file]} ("file" is prepended, relative
        // to the cwd); greedy no-spec, then each --spec-sweep K on the same prompt; per-prompt rows to --json
        std::ifstream fi(chat_file); std::string line; json rows = json::array();
        std::map<std::string, std::vector<double>> dom_rate;
        std::map<std::string, std::map<int, std::pair<double, double>>> dom_k;   // domain -> K -> (sum tok/s ratio, sum tok/verify)
        int gross = 0, cnt = 0;
        while (std::getline(fi, line)) {
            if (line.empty()) continue;
            const json q = json::parse(line);
            std::string content = q["content"];
            if (q.contains("file")) { std::ifstream ff(q["file"].get<std::string>()); std::stringstream ss; ss << ff.rdbuf(); content = ss.str() + "\n\n" + content; }
            const std::vector<int> pid = tok.encode(dsv41enc::encode_messages(json::array({json{{"role", "user"}, {"content", content}}}), thinking ? "thinking" : "chat"));
            const std::string dom = q["domain"];
            std::vector<int> ref;
            eng.set_spec_k(0); eng.reset();
            const dsv4srv::GenStats g0 = eng.generate(pid, gp, [&](int t) { ref.push_back(t); return true; });
            json row{{"id", q["id"]}, {"domain", dom}, {"prompt_tokens", (int)pid.size()}, {"tok_s", g0.tok_per_s}, {"ids", ref},
                     {"head2_fallbacks", g0.head2_fallbacks}};
            dom_rate[dom].push_back(g0.tok_per_s);
            std::string ks_txt;
            json ks = json::array();
            for (int k : sweep) {
                std::vector<int> out;
                eng.set_spec_k(k); eng.reset();
                const dsv4srv::GenStats g = eng.generate(pid, gp, [&](int t) { out.push_back(t); return true; });
                int diff = -1;
                for (size_t i = 0; i < std::max(out.size(), ref.size()); ++i)
                    if (i >= out.size() || i >= ref.size() || out[i] != ref[i]) { diff = (int)i; break; }
                if (diff >= 0) ++gross;                     // near-tie classification lives in --spec-sweep
                const double ratio = g0.tok_per_s > 0 ? g.tok_per_s / g0.tok_per_s : 0.0;
                dom_k[dom][k].first += ratio; dom_k[dom][k].second += g.tok_per_verify;
                json kh = json::array();                    // rounds per realised width (adaptive K)
                for (int w = 0; w < dsv4srv::SpecProfile::MAXW; ++w) { int c = 0; for (int a = 0; a < dsv4srv::SpecProfile::MAXW; ++a) c += g.spec.accept_joint[w][a]; kh.push_back(c); }
                ks.push_back(json{{"k", k}, {"tok_s", g.tok_per_s}, {"tok_per_verify", g.tok_per_verify}, {"first_diff", diff}, {"width_rounds", kh}});
                char b[96]; snprintf(b, sizeof b, "  K=%d %.2f (x%.3f, %.2f/verify%s)", k, g.tok_per_s, ratio, g.tok_per_verify, diff >= 0 ? ", DIFF" : "");
                ks_txt += b;
            }
            row["spec"] = ks; rows.push_back(row); ++cnt;
            printf("[held] %-9s %5zu tok  no-spec %.2f tok/s%s%s\n", q["id"].get<std::string>().c_str(), pid.size(), g0.tok_per_s,
                   g0.head2_fallbacks ? (" (head2 fallbacks " + std::to_string(g0.head2_fallbacks) + ")").c_str() : "", ks_txt.c_str());
        }
        for (auto& [d, v] : dom_rate) {
            double m = 0; for (double x : v) m += x; m /= v.size();
            std::string t;
            for (auto& [k, pr] : dom_k[d]) { char b[64]; snprintf(b, sizeof b, "  K=%d x%.3f %.2f/verify", k, pr.first / v.size(), pr.second / v.size()); t += b; }
            printf("[held] domain %-8s n=%zu  no-spec %.2f tok/s%s\n", d.c_str(), v.size(), m, t.c_str());
        }
        printf("[held] %d prompts, %d spec divergence(s)\n", cnt, gross);
        if (!json_out.empty()) std::ofstream(json_out) << json{{"rows", rows}, {"n", n}, {"spec_divergences", gross}}.dump(1);
        return 0;
    }
    std::vector<int> first;
    std::vector<double> rates;
    bool deterministic = true;
    dsv4srv::GenStats last;
    for (int r = 0; r < repeat; ++r) {
        std::vector<int> out;
        last = eng.generate(ids, gp, [&](int t) { out.push_back(t); return true; });
        rates.push_back(last.tok_per_s);
        if (r == 0) {
            first = out;
            printf("[v41] %s\n", tok.decode(out).c_str());
        } else if (gp.temperature <= 0.f && out != first) deterministic = false;
        printf("[v41] run %d: prefill %d tok %.1f ms (%.1f tok/s)  decode %d tok %.1f ms = %.2f tok/s%s\n", r,
               last.prompt_tokens, last.prefill_ms, last.prompt_tokens / (last.prefill_ms / 1e3),
               last.completion_tokens, last.decode_ms, last.tok_per_s,
               last.head2_fallbacks ? ("  head2 fallbacks " + std::to_string(last.head2_fallbacks)).c_str() : "");
    }
    double mean = 0, mn = 1e30, mx = 0;
    for (double v : rates) { mean += v; mn = std::min(mn, v); mx = std::max(mx, v); }
    mean /= rates.size();
    printf("[v41] decode tok/s mean %.2f  min %.2f  max %.2f over %d run(s)%s\n", mean, mn, mx, repeat,
           repeat > 1 ? (deterministic ? "  -- greedy output identical across runs" : "  -- NONDETERMINISTIC (FAIL)") : "");
    int incons = 0, near_tie = 0;
    if (consistency > 0 && gp.temperature <= 0.f && !first.empty()) {
        // k = 0 included: the first token comes from the prefill's argmax, not a decode step (a stale
        // d_tok there once made every run start from the previous run's last token)
        for (int j = 0; j <= consistency; ++j) {
            const int k = (int)((long)j * (first.size() - 1) / consistency);
            std::vector<int> pre(ids); pre.insert(pre.end(), first.begin(), first.begin() + k);
            eng.reset();
            const std::vector<float> lg = eng.debug_prefix_logits(pre);
            int a1 = 0; for (int i = 1; i < (int)lg.size(); ++i) if (lg[i] > lg[a1]) a1 = i;
            if (a1 != first[k]) {
                const float margin = lg[a1] - lg[first[k]];
                const bool tie = margin < 0.05f;
                printf("[v41] consistency: position %d decode %d, prefill %d (logit gap %.4f)%s\n", k, first[k], a1, margin,
                       tie ? " near-tie" : " GROSS");
                tie ? ++near_tie : ++incons;
            }
        }
        printf("[v41] consistency over %d positions: %d gross, %d near-tie mismatches -> %s\n", consistency + 1, incons, near_tie,
               incons ? "FAIL" : "PASS");
    }
    // speculation sweep: no-spec reference, then each K on the same prompt, same process
    json spec_j;
    int spec_gross = 0;
    if (!sweep.empty()) {
        auto run = [&](int k, std::vector<int>& out) {
            eng.set_spec_k(k); eng.reset();
            out.clear();
            return eng.generate(ids, gp, [&](int t) { out.push_back(t); return true; });
        };
        std::vector<int> ref;
        const dsv4srv::GenStats g0 = run(0, ref);
        printf("[v41] spec sweep: no-spec %.2f tok/s (%d tok)\n", g0.tok_per_s, g0.completion_tokens);
        spec_j["nospec"] = json{{"tok_s", g0.tok_per_s}, {"tokens", g0.completion_tokens}, {"ids", ref}};
        spec_j["drafter"] = eng.drafter_desc();
        json ks = json::array();
        for (int k : sweep) {
            std::vector<int> out;
            const dsv4srv::GenStats g = run(k, out);
            json hist = json::array();
            for (int a = 0; a <= std::min(k, dsv4srv::SpecProfile::MAXW - 1); ++a) hist.push_back(g.spec.accept_joint[std::min(k, dsv4srv::SpecProfile::MAXW - 1)][a]);
            int diff = -1;
            for (size_t i = 0; i < std::max(out.size(), ref.size()); ++i)
                if (i >= out.size() || i >= ref.size() || out[i] != ref[i]) { diff = (int)i; break; }
            std::string cls = "identical";
            if (diff >= 0) {
                cls = "length";
                if (diff < (int)out.size() && diff < (int)ref.size()) {
                    std::vector<int> pre(ids); pre.insert(pre.end(), ref.begin(), ref.begin() + diff);
                    eng.set_spec_k(0); eng.reset();
                    const std::vector<float> lg = eng.debug_prefix_logits(pre);
                    const float gap = std::fabs(lg[ref[diff]] - lg[out[diff]]);
                    cls = gap < 0.05f ? "near-tie" : "GROSS";
                    printf("[v41] spec K=%d: first diff at %d: no-spec %d, spec %d (logit gap %.4f) %s\n", k, diff, ref[diff],
                           out[diff], gap, cls.c_str());
                }
                if (cls != "near-tie") ++spec_gross;
            }
            printf("[v41] spec K=%d: %.2f tok/s (x%.3f), %.3f tok/verify, %s\n", k, g.tok_per_s,
                   g0.tok_per_s > 0 ? g.tok_per_s / g0.tok_per_s : 0.0, g.tok_per_verify, cls.c_str());
            ks.push_back(json{{"k", k}, {"tok_s", g.tok_per_s}, {"tokens", g.completion_tokens}, {"verifies", g.verifies},
                              {"tok_per_verify", g.tok_per_verify}, {"accept_hist", hist}, {"identical", diff < 0},
                              {"first_diff", diff}, {"diff_class", cls},
                              {"speedup", g0.tok_per_s > 0 ? g.tok_per_s / g0.tok_per_s : 0.0}});
        }
        spec_j["k"] = ks;
        spec_j["identity_gross"] = spec_gross;
        printf("[v41] spec identity: %d GROSS mismatch(es) over %zu K -> %s\n", spec_gross, sweep.size(), spec_gross ? "FAIL" : "PASS");
    }
    if (!json_out.empty()) {
        json o;
        o["prompt_ids"] = ids; o["ids"] = first; o["text"] = tok.decode(first);
        o["prefill_ms"] = last.prefill_ms; o["decode_tok_s"] = mean; o["decode_tok_s_min"] = mn; o["decode_tok_s_max"] = mx;
        o["runs"] = repeat; o["deterministic"] = deterministic; o["graph"] = ec.graph; o["mxfp4_hw"] = ec.mxfp4_hw;
        o["ckpt"] = ec.ckpt_dir; o["seqmax"] = ec.seqmax;
        o["consistency_checked"] = consistency; o["consistency_gross"] = incons; o["consistency_near_tie"] = near_tie;
        o["spec_k"] = ec.spec_k; o["draft"] = ec.draft_dir;
        o["verifies"] = last.verifies; o["tok_per_verify"] = last.tok_per_verify;
        if (!sweep.empty()) o["spec"] = spec_j;
        std::ofstream(json_out) << o.dump(1);
    }
    return deterministic && !incons && !spec_gross ? 0 : 1;
}
