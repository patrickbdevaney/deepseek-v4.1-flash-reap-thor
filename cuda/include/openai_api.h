// openai_api.h — OpenAI-compatible request/response shaping for DeepSeek-V4.1-Flash-1017 (REAP).
//
// Kept separate from the HTTP transport and from the engine so it can be unit-gated without a GPU
// (tests/gate_api.cpp). The engine-facing surface is deliberately tiny: a prompt string in,
// generated text out.
//
// Model-specific behaviour that differs from the other servers in this family — do not carry
// gemma's or Laguna's defaults over by habit (CHAT_FORMAT.md §4):
//   temperature 1.0 (NOT 0.7), top_p 0.95 for agentic/tool use, 1.0 otherwise.
//   thinking_mode {chat, thinking} AND reasoning_effort (1..100 or low/high/max, default high = 75)
//   are INDEPENDENT controls; the effort prefix renders in thinking mode only (encoding_dsv41.h).
#pragma once
#include "encoding_dsv41.h"
#include <string>
#include <vector>
#include <cstdint>

namespace dsv4api {

using dsv41enc::json;

struct SamplingParams {
    double temperature = 1.0;      // DeepSeek's own recommendation; NOT 0.7
    double top_p       = 1.0;      // 0.95 when tools are present (agentic use)
    int    max_tokens  = 512;
    std::vector<std::string> stop;
    uint64_t seed = 0;
    bool  has_seed = false;
};

struct ChatRequest {
    std::string model = "deepseek-v4.1-flash-reap";
    json messages = json::array();
    std::string thinking_mode   = "thinking";   // {chat, thinking}
    json reasoning_effort;                      // null (= "high") | 1..100 | low/high/max; thinking-mode only
    bool stream = false;
    bool has_tools = false;
    SamplingParams sampling;
};

inline bool truthy(const json& v, bool dflt) {
    if (v.is_boolean()) return v.get<bool>();
    if (v.is_number())  return v.get<double>() != 0.0;
    return dflt;
}

// Parse an OpenAI /v1/chat/completions body. Tolerant of the usual client variations.
inline ChatRequest parse_chat_request(const json& b) {
    ChatRequest r;
    if (b.contains("model") && b["model"].is_string()) r.model = b["model"].get<std::string>();
    if (b.contains("messages")) r.messages = b["messages"];
    if (b.contains("stream"))   r.stream   = truthy(b["stream"], false);

    // Tools live on the request in the OpenAI schema; the DeepSeek encoder expects them attached to
    // the system message, so fold them in here (creating a system message if there isn't one).
    if (b.contains("tools") && b["tools"].is_array() && !b["tools"].empty()) {
        r.has_tools = true;
        bool placed = false;
        for (auto& m : r.messages)
            if (m.value("role","") == "system") { m["tools"] = b["tools"]; placed = true; break; }
        if (!placed) {
            json sys; sys["role"] = "system"; sys["content"] = ""; sys["tools"] = b["tools"];
            r.messages.insert(r.messages.begin(), sys);
        }
    }

    // Thinking control. Accept both our native field and the common `chat_template_kwargs`
    // / `enable_thinking` spellings clients use.
    if (b.contains("thinking_mode") && b["thinking_mode"].is_string())
        r.thinking_mode = b["thinking_mode"].get<std::string>();
    else if (b.contains("enable_thinking"))
        r.thinking_mode = truthy(b["enable_thinking"], true) ? "thinking" : "chat";
    else if (b.contains("chat_template_kwargs") && b["chat_template_kwargs"].contains("thinking"))
        r.thinking_mode = truthy(b["chat_template_kwargs"]["thinking"], true) ? "thinking" : "chat";
    if (r.thinking_mode != "chat" && r.thinking_mode != "thinking") r.thinking_mode = "thinking";

    // V4.1: int 1..100 or low/high/max ("medium" from OpenAI clients maps to the default); an invalid
    // value falls back to the default rather than failing the request.
    if (b.contains("reasoning_effort")) {
        json e = b["reasoning_effort"];
        if (e.is_number_float()) e = (long long)e.get<double>();
        if (e.is_string() && e.get<std::string>() == "medium") e = json();
        try { (void)dsv41enc::effort_budget(e); r.reasoning_effort = e; } catch (...) { r.reasoning_effort = json(); }
    }

    auto& s = r.sampling;
    if (b.contains("temperature") && b["temperature"].is_number()) s.temperature = b["temperature"].get<double>();
    // top_p default depends on whether this is an agentic turn — DeepSeek recommends 0.95 with tools.
    s.top_p = r.has_tools ? 0.95 : 1.0;
    if (b.contains("top_p") && b["top_p"].is_number()) s.top_p = b["top_p"].get<double>();
    if (b.contains("max_tokens") && b["max_tokens"].is_number())            s.max_tokens = b["max_tokens"].get<int>();
    else if (b.contains("max_completion_tokens") && b["max_completion_tokens"].is_number())
                                                                            s.max_tokens = b["max_completion_tokens"].get<int>();
    if (b.contains("seed") && b["seed"].is_number()) { s.seed = b["seed"].get<uint64_t>(); s.has_seed = true; }
    if (b.contains("stop")) {
        if (b["stop"].is_string()) s.stop.push_back(b["stop"].get<std::string>());
        else if (b["stop"].is_array()) for (auto& x : b["stop"]) if (x.is_string()) s.stop.push_back(x.get<std::string>());
    }
    return r;
}

inline std::string build_prompt(const ChatRequest& r, json* images = nullptr) {
    return dsv41enc::encode_messages(r.messages, r.thinking_mode, json::array(), true, true, r.reasoning_effort, images);
}

// Non-streaming response. `parsed` comes from dsv41enc::parse_message_from_completion_text.
inline json chat_completion_response(const std::string& id, const std::string& model,
                                     const json& parsed, int prompt_tokens, int completion_tokens,
                                     long created) {
    json msg = json::object();
    msg["role"] = "assistant";
    msg["content"] = parsed.value("content", "");
    const std::string rc = parsed.value("reasoning_content", "");
    if (!rc.empty()) msg["reasoning_content"] = rc;      // surfaced separately, as vLLM/SGLang do
    const bool has_tc = parsed.contains("tool_calls") && !parsed["tool_calls"].empty();
    if (has_tc) {
        json tcs = json::array();
        int i = 0;
        for (const auto& tc : parsed["tool_calls"]) {
            json o = tc;
            o["id"] = "call_" + id.substr(0,8) + "_" + std::to_string(i++);
            tcs.push_back(o);
        }
        msg["tool_calls"] = tcs;
    }
    json choice = json::object();
    choice["index"] = 0;
    choice["message"] = msg;
    choice["finish_reason"] = has_tc ? "tool_calls" : "stop";

    json out = json::object();
    out["id"] = "chatcmpl-" + id;
    out["object"] = "chat.completion";
    out["created"] = created;
    out["model"] = model;
    out["choices"] = json::array({choice});
    json usage = json::object();
    usage["prompt_tokens"] = prompt_tokens;
    usage["completion_tokens"] = completion_tokens;
    usage["total_tokens"] = prompt_tokens + completion_tokens;
    out["usage"] = usage;
    return out;
}

// One SSE chunk of a streamed completion.
inline std::string sse_chunk(const std::string& id, const std::string& model, long created,
                             const std::string& delta_content, const std::string& delta_reasoning,
                             const char* finish_reason) {
    json d = json::object();
    if (!delta_content.empty())   d["content"] = delta_content;
    if (!delta_reasoning.empty()) d["reasoning_content"] = delta_reasoning;
    json choice = json::object();
    choice["index"] = 0;
    choice["delta"] = d;
    if (finish_reason) choice["finish_reason"] = finish_reason; else choice["finish_reason"] = nullptr;
    json o = json::object();
    o["id"] = "chatcmpl-" + id;
    o["object"] = "chat.completion.chunk";
    o["created"] = created;
    o["model"] = model;
    o["choices"] = json::array({choice});
    // dump with the replace handler, not the throwing default: deltas carry model-generated text
    // and this model's byte-level BPE can emit sequences that are not valid UTF-8 on their own.
    // See dump_lossy() in server.cpp for the incident this comes from.
    return "data: " + o.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
}

inline json models_response(const std::string& model, long created) {
    json m = json::object();
    m["id"] = model; m["object"] = "model"; m["created"] = created; m["owned_by"] = "local";
    json o = json::object();
    o["object"] = "list";
    o["data"] = json::array({m});
    return o;
}

} // namespace dsv4api
