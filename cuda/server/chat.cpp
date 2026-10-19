// chat.cpp — terminal chat client. Speaks the OpenAI endpoint over HTTP, so it exercises exactly
// what an external agentic harness would: no shared memory with the server, no private hooks.
//
//   build/dsv4-chat                          talk to http://127.0.0.1:8080
//   build/dsv4-chat --host h --port 8080
//   build/dsv4-chat --effort high --mode thinking --temp 0.6
//   echo "2+2?" | build/dsv4-chat --once     one-shot, for scripting
//
// Slash commands: /new /think /chat /effort <low|high|max> /temp <f> /sys <text> /stats /quit
#define CPPHTTPLIB_NO_EXCEPTIONS
#include "third_party/httplib.h"
#include "third_party/json.hpp"
#include <cstdio>
#include <iostream>
#include <string>

using json = nlohmann::ordered_json;

static const char* DIM = "\033[2m";
static const char* CYA = "\033[36m";
static const char* MAG = "\033[35m";
static const char* YEL = "\033[33m";
static const char* RST = "\033[0m";
static bool g_color = true;
static const char* C(const char* c) { return g_color ? c : ""; }

int main(int argc, char** argv) {
    std::string host = "127.0.0.1", sys, mode = "thinking", effort = "low";
    int port = 8080, maxt = 1024;
    double temp = 1.0, top_p = 1.0;
    bool once = false, show_think = true;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto nx = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "--host") host = nx();
        else if (a == "--port") port = atoi(nx().c_str());
        else if (a == "--sys") sys = nx();
        else if (a == "--mode") mode = nx();
        else if (a == "--effort") effort = nx();
        else if (a == "--temp") temp = atof(nx().c_str());
        else if (a == "--top-p") top_p = atof(nx().c_str());
        else if (a == "--max-tokens") maxt = atoi(nx().c_str());
        else if (a == "--once") once = true;
        else if (a == "--no-think") show_think = false;
        else if (a == "--no-color") g_color = false;
        else if (a == "--help") {
            printf("usage: %s [--host H] [--port P] [--sys TEXT] [--mode chat|thinking]\n"
                   "          [--effort low|high|max] [--temp F] [--top-p F] [--max-tokens N]\n"
                   "          [--once] [--no-think] [--no-color]\n", argv[0]);
            return 0;
        }
    }

    httplib::Client cli(host, port);
    cli.set_read_timeout(3600, 0);
    cli.set_write_timeout(60, 0);

    if (auto r = cli.Get("/health")) {
        if (r->status == 200) {
            const json h = json::parse(r->body, nullptr, false);
            if (!h.is_discarded())
                printf("%sconnected: %s (context %d)%s\n", C(DIM),
                       h.value("model", "?").c_str(), h.value("seqmax", 0), C(RST));
        }
    } else {
        fprintf(stderr, "cannot reach http://%s:%d — is the server running?\n", host.c_str(), port);
        return 1;
    }
    if (!once)
        printf("%s/new /think /chat /effort <low|high|max> /temp <f> /sys <text> /stats /quit%s\n\n",
               C(DIM), C(RST));

    json history = json::array();

    for (;;) {
        std::string line;
        if (!once) { printf("%s>%s ", C(CYA), C(RST)); fflush(stdout); }
        if (!std::getline(std::cin, line)) break;
        if (line.empty()) continue;

        if (line[0] == '/' && !once) {
            const size_t sp = line.find(' ');
            const std::string cmd = line.substr(0, sp);
            const std::string arg = sp == std::string::npos ? "" : line.substr(sp + 1);
            if (cmd == "/quit" || cmd == "/exit") break;
            if (cmd == "/new") { history = json::array(); printf("%s(cleared)%s\n", C(DIM), C(RST)); continue; }
            if (cmd == "/think") { mode = "thinking"; printf("%s(thinking)%s\n", C(DIM), C(RST)); continue; }
            if (cmd == "/chat")  { mode = "chat";     printf("%s(chat)%s\n", C(DIM), C(RST)); continue; }
            if (cmd == "/effort") { effort = arg; printf("%s(effort %s)%s\n", C(DIM), effort.c_str(), C(RST)); continue; }
            if (cmd == "/temp") { temp = atof(arg.c_str()); printf("%s(temp %.2f)%s\n", C(DIM), temp, C(RST)); continue; }
            if (cmd == "/sys")  { sys = arg; printf("%s(system prompt set)%s\n", C(DIM), C(RST)); continue; }
            if (cmd == "/stats") {
                if (auto r = cli.Get("/metrics")) printf("%s%s%s", C(DIM), r->body.c_str(), C(RST));
                continue;
            }
            printf("%s(unknown command)%s\n", C(DIM), C(RST));
            continue;
        }

        history.push_back(json{{"role", "user"}, {"content", line}});
        json msgs = json::array();
        if (!sys.empty()) msgs.push_back(json{{"role", "system"}, {"content", sys}});
        for (const auto& m : history) msgs.push_back(m);

        json body{{"messages", msgs}, {"stream", true}, {"temperature", temp}, {"top_p", top_p},
                  {"max_tokens", maxt}, {"thinking_mode", mode}, {"reasoning_effort", effort}};

        // SSE frames can split across TCP reads, so buffer and consume complete "\n\n" records.
        std::string buf, answer, reasoning, stat;
        bool in_think = false;
        auto on_chunk = [&](const char* data, size_t len) -> bool {
            buf.append(data, len);
            size_t i;
            while ((i = buf.find("\n\n")) != std::string::npos) {
                const std::string frame = buf.substr(0, i);
                buf.erase(0, i + 2);
                if (frame.rfind("data: ", 0) != 0) continue;
                const std::string p = frame.substr(6);
                if (p == "[DONE]") continue;
                const json j = json::parse(p, nullptr, false);
                if (j.is_discarded()) continue;
                if (j.contains("error")) {
                    printf("\n%s[error: %s]%s\n", C(YEL),
                           j["error"].value("message", "unknown").c_str(), C(RST));
                    continue;
                }
                if (j.contains("choices") && !j["choices"].empty()) {
                    const auto& d = j["choices"][0]["delta"];
                    if (d.contains("reasoning_content")) {
                        const std::string t = d["reasoning_content"].get<std::string>();
                        reasoning += t;
                        if (show_think) {
                            if (!in_think) { printf("%s[thinking] ", C(MAG)); in_think = true; }
                            fputs(t.c_str(), stdout);
                        }
                    }
                    if (d.contains("content")) {
                        const std::string t = d["content"].get<std::string>();
                        if (in_think) { printf("%s\n\n", C(RST)); in_think = false; }
                        answer += t;
                        fputs(t.c_str(), stdout);
                    }
                    if (d.contains("tool_calls")) {
                        printf("\n%s[tool_calls] %s%s\n", C(YEL), d["tool_calls"].dump(2).c_str(), C(RST));
                    }
                }
                if (j.contains("usage")) {
                    const auto& u = j["usage"];
                    const auto& tm = j.contains("timings") ? j["timings"] : json::object();
                    char b[256];
                    const long cached = u.contains("prompt_tokens_details")
                                      ? u["prompt_tokens_details"].value("cached_tokens", 0L) : 0L;
                    snprintf(b, sizeof b, "%ld tok, %.1f tok/s, %.2f tok/verify, prefill %.0f ms (%ld/%ld cached)",
                             u.value("completion_tokens", 0L), tm.value("tokens_per_second", 0.0),
                             tm.value("tokens_per_verify", 0.0), tm.value("prefill_ms", 0.0),
                             cached, u.value("prompt_tokens", 0L));
                    stat = b;
                }
            }
            fflush(stdout);
            return true;
        };

        // The 6-arg overload is the streaming one: a ContentReceiver is only accepted alongside an
        // explicit Headers and a progress slot.
        if (!cli.Post("/v1/chat/completions", httplib::Headers(), body.dump(), "application/json",
                      on_chunk, httplib::DownloadProgress())) {
            printf("%s[request failed]%s\n", C(YEL), C(RST));
            history.erase(history.end() - 1);
            continue;
        }
        if (in_think) printf("%s", C(RST));
        printf("\n");
        if (!stat.empty()) printf("%s%s%s\n\n", C(DIM), stat.c_str(), C(RST));

        json am{{"role", "assistant"}, {"content", answer}};
        if (!reasoning.empty()) am["reasoning_content"] = reasoning;
        history.push_back(am);
        if (once) break;
    }
    return 0;
}
