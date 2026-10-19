// tokprobe.cpp — offline token accounting for the budget-extension prefix rebuild.
//
// WHY IT EXISTS. eval_extend.py continues a truncated trace from its stored prefix and REFUSES to
// proceed unless the rebuilt prefix tokenizes to exactly prompt_tokens + completion_tokens of the
// base record. On gpqa-0010 it refused: 8325 rebuilt against 8327 fed. Two tokens is not a rounding
// error, it is a structural difference between what /v1/chat/completions encodes and what the
// rebuild assumes -- and finding it by sending probes at the live engine would mean generating
// while a benchmark is scoring, which is the one thing the eval programme must never do.
//
// So this links the SAME encoder and the SAME tokenizer the server links, and answers the question
// on the CPU with the GPU untouched.
//
//   build/tokprobe --ckpt <dir> --prompt-file q.txt --content-file c.txt
//
// Prints the chat-path token count, the raw-completion rebuild count, and the delta, plus a
// piece-by-piece breakdown so the missing tokens can be named rather than guessed at.
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include "tokenizer_dsv4.h"
#include "encoding_dsv4.h"
#include "openai_api.h"
#include <vector>

static std::string slurp(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// BATCH MODE. One item proves the mechanism; the tolerance has to come from the whole population,
// because a gate set from a single observation is a guess wearing a number. The manifest is one
// JSON object per line carrying id, prompt, content and the base record's own token counts, so the
// entire truncated set of every task can be audited on the CPU before the engine is asked for
// anything.
static int run_manifest(dsv4tok::Tokenizer& tok, const std::string& path) {
    std::ifstream f(path);
    std::string line;
    int n = 0, exact = 0, worst = 0;
    printf("%-24s %8s %8s %8s %7s\n", "id", "fed", "rebuilt", "delta", "verdict");
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        const auto j = nlohmann::json::parse(line);
        const std::string id = j.value("id", "");
        const std::string q = j.value("prompt", ""), c = j.value("content", "");
        const int pt = j.value("prompt_tokens", 0), ct = j.value("completion_tokens", 0);
        auto ids = tok.encode(std::string(dsv4enc::USER_SP) + q + dsv4enc::ASSISTANT_SP + c);
        const int rebuilt = (int)ids.size() + 1;              // /v1/completions prepends BOS
        const int delta = rebuilt - (pt + ct);
        if (!delta) exact++;
        if (delta < worst) worst = delta;
        if (delta > -worst) worst = delta;
        printf("%-24s %8d %8d %+8d %7s\n", id.c_str(), pt + ct, rebuilt, delta,
               delta ? "drift" : "exact");
        n++;
    }
    printf("\n%d items: %d exact, %d drifted, worst |delta| %d\n", n, exact, n - exact, worst < 0 ? -worst : worst);
    return 0;
}

int main(int argc, char** argv) {
    std::string ckpt, pf, cf, manifest, idsfile, effort = "low", mode = "thinking";
    int ntok = 0;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--ckpt" && i + 1 < argc) ckpt = argv[++i];
        else if (a == "--prompt-file" && i + 1 < argc) pf = argv[++i];
        else if (a == "--content-file" && i + 1 < argc) cf = argv[++i];
        else if (a == "--effort" && i + 1 < argc) effort = argv[++i];
        else if (a == "--mode" && i + 1 < argc) mode = argv[++i];
        else if (a == "--manifest" && i + 1 < argc) manifest = argv[++i];
        // --ids FILE [--ntok N]: encode a text file and print comma-separated token ids, optionally
        // truncated to N tokens. This is how DSV4_PROMPTS_FILE gets prompts whose ids come from the
        // CHECKPOINT'S OWN tokenizer rather than from a guess -- the repo's standing constraint.
        else if (a == "--ids" && i + 1 < argc) idsfile = argv[++i];
        else if (a == "--ntok" && i + 1 < argc) ntok = atoi(argv[++i]);
    }
    if (ckpt.empty() || (pf.empty() && manifest.empty() && idsfile.empty())) {
        fprintf(stderr, "usage: tokprobe --ckpt DIR --prompt-file F [--content-file F]"
                        " [--effort low|high|max] [--mode thinking|chat]\n");
        return 2;
    }
    dsv4tok::Tokenizer tok;
    tok.load(ckpt + "/tokenizer.json");
    if (!manifest.empty()) return run_manifest(tok, manifest);
    if (!idsfile.empty()) {
        auto ids = tok.encode(slurp(idsfile));
        size_t n = (ntok > 0 && (size_t)ntok < ids.size()) ? (size_t)ntok : ids.size();
        for (size_t i = 0; i < n; i++) printf("%d%s", ids[i], i + 1 < n ? "," : "\n");
        fprintf(stderr, "%zu tokens\n", n);
        return 0;
    }

    const std::string question = slurp(pf);
    const std::string content = cf.empty() ? std::string() : slurp(cf);

    // 1. Exactly what /v1/chat/completions would have encoded for this item.
    dsv4api::ChatRequest cr;
    cr.messages = nlohmann::json::array({{{"role", "user"}, {"content", question}}});
    cr.thinking_mode = mode;
    cr.reasoning_effort = effort;
    const std::string chat = dsv4api::build_prompt(cr);
    const auto chat_ids = tok.encode(chat);

    // 2. Exactly what eval_extend.py rebuilds and sends to /v1/completions, which prepends BOS.
    const std::string rebuilt = std::string(dsv4enc::USER_SP) + question +
                                dsv4enc::ASSISTANT_SP + content;
    auto rebuilt_ids = tok.encode(rebuilt);
    rebuilt_ids.insert(rebuilt_ids.begin(), tok.bos_id);

    // 3. The same rebuild without the stored content, so the prompt legs can be compared directly.
    const std::string head = std::string(dsv4enc::USER_SP) + question + dsv4enc::ASSISTANT_SP;
    auto head_ids = tok.encode(head);
    head_ids.insert(head_ids.begin(), tok.bos_id);

    printf("chat prompt        %zu tokens\n", chat_ids.size());
    printf("rebuilt head       %zu tokens   (BOS + User + question + Assistant)\n", head_ids.size());
    printf("delta head         %+d tokens   (chat - rebuilt head)\n",
           (int)chat_ids.size() - (int)head_ids.size());
    if (!cf.empty()) {
        printf("rebuilt full       %zu tokens   (head + stored content)\n", rebuilt_ids.size());
        printf("content alone      %zu tokens\n", tok.encode(content).size());
    }
    // Name the difference rather than leave it as a number: dump the chat prompt's tail so the
    // tokens the rebuild is missing can be read off directly.
    printf("\nchat prompt tail   %s\n", chat.size() > 220
           ? ("..." + chat.substr(chat.size() - 200)).c_str() : chat.c_str());
    printf("rebuilt head tail  %s\n", head.size() > 220
           ? ("..." + head.substr(head.size() - 200)).c_str() : head.c_str());
    printf("\nchat head          %s\n", chat.substr(0, 160).c_str());
    printf("rebuilt head start %s\n", head.substr(0, 160).c_str());
    return 0;
}
