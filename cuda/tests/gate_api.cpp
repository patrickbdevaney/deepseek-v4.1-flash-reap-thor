// gate_api.cpp — Gate S1 (part 2): OpenAI request/response shaping, no GPU required.
//
// Covers the model-specific behaviour that is easy to get wrong by porting habits from the other
// servers in this family (CHAT_FORMAT.md): the two orthogonal thinking controls, the tools ->
// system-message fold, the DSML tool-call round-trip including the string="true|false" typing
// signal, and the sampling defaults (temperature 1.0, top_p 0.95 only when tools are present).
//
//   build: g++ -O1 -std=c++17 -I include tests/gate_api.cpp -o build/gate_api
#include "openai_api.h"
#include <cstdio>
#include <string>

using dsv4api::json;
static int pass = 0, fail = 0;
static void ck(bool ok, const char* what, const std::string& detail = "") {
    printf("[%s] %s%s\n", ok ? "PASS" : "FAIL", what, detail.empty() ? "" : ("  " + detail).c_str());
    ok ? ++pass : ++fail;
}

int main() {
    // ---- thinking_mode: native field, enable_thinking, chat_template_kwargs, and the default ----
    {
        auto mode = [](json b){ b["messages"] = json::array(); return dsv4api::parse_chat_request(b).thinking_mode; };
        ck(mode(json::object()) == "thinking", "default thinking_mode is 'thinking'");
        ck(mode(json{{"thinking_mode","chat"}}) == "chat", "explicit thinking_mode honoured");
        ck(mode(json{{"enable_thinking",false}}) == "chat", "enable_thinking=false -> chat");
        ck(mode(json{{"chat_template_kwargs", json{{"thinking",false}}}}) == "chat",
           "chat_template_kwargs.thinking=false -> chat");
        ck(mode(json{{"thinking_mode","nonsense"}}) == "thinking", "invalid thinking_mode falls back safely");
    }

    // ---- reasoning_effort is independent of thinking_mode, and validated ----
    {
        json b; b["messages"] = json::array(); b["reasoning_effort"] = "max";
        auto r = dsv4api::parse_chat_request(b);
        ck(r.reasoning_effort == "max" && r.thinking_mode == "thinking",
           "reasoning_effort is orthogonal to thinking_mode");
        b["reasoning_effort"] = 37;                             // V4.1: numeric 1..100
        ck(dsv4api::parse_chat_request(b).reasoning_effort == 37, "numeric reasoning_effort accepted");
        b["reasoning_effort"] = 500;
        ck(dsv4api::parse_chat_request(b).reasoning_effort.is_null(),
           "out-of-range reasoning_effort falls back to the default (high = 75)");
    }

    // ---- sampling defaults: temperature 1.0, top_p 0.95 ONLY with tools ----
    {
        json b; b["messages"] = json::array({json{{"role","user"},{"content","hi"}}});
        auto plain = dsv4api::parse_chat_request(b);
        ck(plain.sampling.temperature == 1.0, "default temperature is 1.0 (NOT gemma/Laguna's 0.7)");
        ck(plain.sampling.top_p == 1.0, "default top_p is 1.0 without tools");

        b["tools"] = json::array({json{{"type","function"},{"function",json{{"name","f"},{"parameters",json::object()}}}}});
        auto agentic = dsv4api::parse_chat_request(b);
        ck(agentic.sampling.top_p == 0.95, "top_p becomes 0.95 when tools are present (agentic)");
        ck(agentic.has_tools, "has_tools detected");

        b["top_p"] = 0.5;
        ck(dsv4api::parse_chat_request(b).sampling.top_p == 0.5, "explicit top_p overrides the default");
    }

    // ---- tools fold onto the system message (the encoder expects them there, not on the request) ----
    {
        json b;
        b["messages"] = json::array({json{{"role","user"},{"content","hi"}}});
        b["tools"] = json::array({json{{"type","function"},{"function",json{{"name","f"},{"parameters",json::object()}}}}});
        auto r = dsv4api::parse_chat_request(b);
        ck(r.messages.size() == 2 && r.messages[0].value("role","") == "system" && r.messages[0].contains("tools"),
           "tools with no system message -> a system message is created to carry them");

        json b2;
        b2["messages"] = json::array({json{{"role","system"},{"content","s"}}, json{{"role","user"},{"content","hi"}}});
        b2["tools"] = b["tools"];
        auto r2 = dsv4api::parse_chat_request(b2);
        ck(r2.messages.size() == 2 && r2.messages[0].contains("tools"),
           "tools with an existing system message -> attached, no message added");
    }

    // ---- prompt building actually reaches the encoder, and the effort prefix leads ----
    {
        json b;
        b["messages"] = json::array({json{{"role","user"},{"content","q"}}});
        b["reasoning_effort"] = "high";
        const std::string p = dsv4api::build_prompt(dsv4api::parse_chat_request(b));
        const std::string pfx = std::string(dsv41enc::SYSTEM_SP) + "Reasoning Effort: 75 ";
        ck(p.rfind(dsv41enc::BOS, 0) == 0, "prompt starts with BOS");
        ck(p.find(pfx) == std::string::npos ? false : p.find(pfx) < 40,
           "reasoning_effort prefix sits at the FRONT (prefix-cache hazard, CHAT_FORMAT §2.2)");
    }

    // ---- DSML tool-call round trip, including the string="true|false" typing signal ----
    {
        json tc; tc["type"] = "function";
        tc["function"] = json{{"name","get_weather"},
                              {"arguments","{\"location\": \"Beijing\", \"days\": 5, \"metric\": true}"}};
        json asst; asst["role"] = "assistant"; asst["content"] = ""; asst["tool_calls"] = json::array({tc});
        json msgs = json::array({json{{"role","user"},{"content","w?"}}, asst});
        const std::string enc = dsv41enc::encode_messages(msgs, "chat");

        ck(enc.find("string=\"true\">Beijing") != std::string::npos, "string param encoded with string=\"true\"");
        ck(enc.find("string=\"false\">5")      != std::string::npos, "number param encoded with string=\"false\"");
        ck(enc.find("string=\"false\">true")   != std::string::npos, "bool param encoded with string=\"false\"");

        // parse it back out of the assistant turn
        const size_t a = enc.find(dsv41enc::ASSISTANT_SP);
        const json parsed = dsv41enc::parse_message_from_completion_text(
            enc.substr(a + std::string(dsv41enc::ASSISTANT_SP).size()), "chat");
        ck(parsed["tool_calls"].size() == 1, "one tool call parsed back");
        if (parsed["tool_calls"].size() == 1) {
            const std::string args = parsed["tool_calls"][0]["function"]["arguments"].get<std::string>();
            json a2; bool ok = true;
            try { a2 = json::parse(args); } catch (...) { ok = false; }
            ck(ok && a2.value("location", "") == "Beijing", "string arg round-trips as a string", args);
            ck(ok && a2["days"].is_number() && a2["days"] == 5, "number arg round-trips as a NUMBER, not \"5\"");
            ck(ok && a2["metric"].is_boolean() && a2["metric"] == true, "bool arg round-trips as a bool");
            ck(parsed["tool_calls"][0]["function"].value("name","") == "get_weather", "tool name round-trips");
        }
    }

    // ---- response shaping ----
    {
        json parsed = json::object();
        parsed["content"] = "hello";
        parsed["reasoning_content"] = "thinking...";
        parsed["tool_calls"] = json::array();
        const json r = dsv4api::chat_completion_response("abcdef123456", "m", parsed, 10, 3, 1700000000);
        ck(r["choices"][0]["message"]["reasoning_content"] == "thinking...",
           "reasoning_content surfaced as its own field");
        ck(r["choices"][0]["finish_reason"] == "stop", "finish_reason 'stop' with no tool calls");
        ck(r["usage"]["total_tokens"] == 13, "usage totals add up");

        json tc; tc["type"]="function"; tc["function"]=json{{"name","f"},{"arguments","{}"}};
        parsed["tool_calls"] = json::array({tc});
        const json r2 = dsv4api::chat_completion_response("abcdef123456", "m", parsed, 10, 3, 1700000000);
        ck(r2["choices"][0]["finish_reason"] == "tool_calls", "finish_reason 'tool_calls' when a call is present");
        ck(r2["choices"][0]["message"]["tool_calls"][0].contains("id"), "tool calls get ids");
    }

    // ---- SSE framing ----
    {
        const std::string c = dsv4api::sse_chunk("id","m",0,"tok","",nullptr);
        ck(c.rfind("data: ",0) == 0 && c.size() > 8 && c.compare(c.size()-2,2,"\n\n") == 0,
           "SSE chunk is 'data: {...}\\n\\n'");
    }

    printf("\nGate API: %d passed, %d failed -> %s\n", pass, fail, fail ? "FAIL" : "PASS");
    return fail ? 1 : 0;
}
