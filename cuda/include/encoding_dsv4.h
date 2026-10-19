// encoding_dsv4.h — C++ port of the checkpoint's encoding/encoding_dsv4.py.
//
// DeepSeek-V4 ships NO Jinja chat template; the prompt format is defined by that Python file.
// This is a faithful port of its logic, gated byte-exact against the four golden vectors in
// encoding/tests/test_input_N.json -> test_output_N.txt (see tests/gate_encoding.cpp).
//
// Two independent controls — see CHAT_FORMAT.md:
//   thinking_mode    {chat, thinking}          binary; decides <think> vs </think> after <|Assistant|>
//   reasoning_effort {low, high, max}          text prefix BEFORE the system message; thinking-mode only
// They are orthogonal. `low` is the default and emits nothing.
//
// No Python on the hot path: this header is the server's encoder.
#pragma once
#include "third_party/json.hpp"
#include <string>
#include <vector>
#include <map>
#include <set>
#include <stdexcept>

namespace dsv4enc {

// ordered_json, NOT json: nlohmann's default `json` is backed by std::map and therefore SORTS
// object keys, while Python's json.dumps preserves insertion order. The tool schemas are embedded
// in the prompt verbatim, so key order is load-bearing — sorting them fails the byte-exact gate.
using json = nlohmann::ordered_json;

// ---- special tokens (verbatim from encoding_dsv4.py) ----
inline constexpr const char* BOS               = "<｜begin▁of▁sentence｜>";
inline constexpr const char* EOS               = "<｜end▁of▁sentence｜>";
inline constexpr const char* THINK_START       = "<think>";
inline constexpr const char* THINK_END         = "</think>";
inline constexpr const char* DSML              = "｜DSML｜";
inline constexpr const char* USER_SP           = "<｜User｜>";
inline constexpr const char* ASSISTANT_SP      = "<｜Assistant｜>";
inline constexpr const char* LATEST_REMINDER_SP= "<｜latest_reminder｜>";

// Task tokens drive DeepSeek's internal search-agent pipeline. Tokenised correctly, not surfaced
// as server features (CHAT_FORMAT.md §5).
inline const std::map<std::string,std::string>& task_tokens() {
    static const std::map<std::string,std::string> m = {
        {"action","<｜action｜>"}, {"query","<｜query｜>"}, {"authority","<｜authority｜>"},
        {"domain","<｜domain｜>"}, {"title","<｜title｜>"},  {"read_url","<｜read_url｜>"},
    };
    return m;
}

// Verbatim from REASONING_EFFORT_PROMPTS. Do NOT paraphrase — a reworded prefix is a different
// prompt, and this one sits at position 0 where it also keys the prefix cache.
inline const std::map<std::string,std::string>& reasoning_effort_prompts() {
    static const std::map<std::string,std::string> m = {
        {"low", ""},
        {"high",
         "Reasoning Effort: Absolute maximum with no shortcuts permitted.\n"
         "You MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.\n"
         "Explicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.\n\n"},
        {"max",
         "Reasoning Effort: Beyond maximum — exhaustive, relentless, and uncompromising.\n"
         "You MUST reason with the utmost depth and rigor, leaving absolutely nothing to chance: exhaustively decompose the problem into its most fundamental components, trace every causal chain to its root, and resolve the underlying cause rather than any surface symptom.\n"
         "Do not stop reasoning until you have independently verified the solution from multiple angles and are certain that no assumption remains unchecked and no error remains undiscovered.\n\n"},
    };
    return m;
}
inline constexpr const char* DEFAULT_REASONING_EFFORT = "low";

// ---- helpers ----

// Mimic Python's json.dumps(value, ensure_ascii=False) EXACTLY.
// Two differences from nlohmann's dump() that both break the byte-exact gate:
//   1. Python's default separators are ", " and ": " (with spaces); nlohmann emits "," and ":".
//   2. Key order — handled by using ordered_json above.
// Scalars are delegated to nlohmann's dump(), which already escapes correctly and emits raw UTF-8
// (i.e. ensure_ascii=False).
inline std::string to_json(const json& v) {
    if (v.is_object()) {
        std::string s = "{";
        bool first = true;
        for (auto it = v.begin(); it != v.end(); ++it) {
            if (!first) s += ", ";
            first = false;
            s += json(it.key()).dump() + ": " + to_json(it.value());
        }
        return s + "}";
    }
    if (v.is_array()) {
        std::string s = "[";
        for (size_t i = 0; i < v.size(); ++i) { if (i) s += ", "; s += to_json(v[i]); }
        return s + "]";
    }
    return v.dump();
}

inline std::string replace_all(std::string s, const std::string& from, const std::string& to) {
    if (from.empty()) return s;
    size_t p = 0;
    while ((p = s.find(from, p)) != std::string::npos) { s.replace(p, from.size(), to); p += to.size(); }
    return s;
}

inline std::string render_tools(const json& tools_fns) {
    std::string schemas;
    for (size_t i = 0; i < tools_fns.size(); ++i) {
        if (i) schemas += "\n";
        schemas += to_json(tools_fns[i]);
    }
    std::string t =
        "## Tools\n\nYou have access to a set of tools to help answer the user's question. "
        "You can invoke tools by writing a \"<" + std::string(DSML) + "tool_calls>\" block like the following:\n\n"
        "<" + DSML + "tool_calls>\n"
        "<" + DSML + "invoke name=\"$TOOL_NAME\">\n"
        "<" + DSML + "parameter name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</" + DSML + "parameter>\n"
        "...\n"
        "</" + DSML + "invoke>\n"
        "<" + DSML + "invoke name=\"$TOOL_NAME2\">\n"
        "...\n"
        "</" + DSML + "invoke>\n"
        "</" + DSML + "tool_calls>\n\n"
        "String parameters should be specified as is and set `string=\"true\"`. For all other types "
        "(numbers, booleans, arrays, objects), pass the value in JSON format and set `string=\"false\"`.\n\n"
        "If thinking_mode is enabled (triggered by " + THINK_START + "), you MUST output your complete "
        "reasoning inside " + THINK_START + "..." + THINK_END + " BEFORE any tool calls or final response.\n\n"
        "Otherwise, output directly after " + THINK_END + " with tool calls or final response.\n\n"
        "### Available Tool Schemas\n\n" + schemas + "\n\n"
        "You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.\n";
    return t;
}

// OpenAI tools -> the bare function objects the template renders.
inline json tools_from_openai(const json& tools) {
    json out = json::array();
    for (const auto& t : tools) if (t.contains("function")) out.push_back(t["function"]);
    return out;
}

// encode_arguments_to_dsml: each parameter carries string="true|false". That attribute is the ONLY
// signal separating the literal string "5" from the number 5 — see CHAT_FORMAT.md §3.
inline std::string encode_arguments_to_dsml(const std::string& arguments_json) {
    json args;
    bool ok = true;
    try { args = json::parse(arguments_json); } catch (...) { ok = false; }
    if (!ok || !args.is_object()) { args = json::object(); args["arguments"] = arguments_json; }

    std::string out;
    bool first = true;
    for (auto it = args.begin(); it != args.end(); ++it) {
        if (!first) out += "\n";
        first = false;
        const bool is_str = it.value().is_string();
        out += "<" + std::string(DSML) + "parameter name=\"" + it.key() + "\" string=\"" +
               (is_str ? "true" : "false") + "\">" +
               (is_str ? it.value().get<std::string>() : to_json(it.value())) +
               "</" + DSML + "parameter>";
    }
    return out;
}

inline int find_last_user_index(const json& msgs) {
    for (int i = (int)msgs.size() - 1; i >= 0; --i) {
        const std::string r = msgs[i].value("role", "");
        if (r == "user" || r == "developer") return i;
    }
    return -1;
}

// merge_tool_messages: DeepSeek-V4 has no standalone "tool" role — results fold into the preceding
// user message as content_blocks wrapped in <tool_result>.
inline json merge_tool_messages(const json& messages) {
    json out = json::array();
    for (const auto& m : messages) {
        const std::string role = m.value("role", "");
        if (role != "tool") { out.push_back(m); continue; }
        json block = json::object();
        block["type"] = "tool_result";
        block["content"] = m.value("content", "");
        if (m.contains("tool_call_id")) block["tool_call_id"] = m["tool_call_id"];

        if (!out.empty() && out.back().value("role", "") == "user") {
            json& prev = out.back();
            if (!prev.contains("content_blocks")) {
                json blocks = json::array();
                const std::string c = prev.value("content", "");
                if (!c.empty()) { json tb; tb["type"] = "text"; tb["text"] = c; blocks.push_back(tb); }
                prev["content_blocks"] = blocks;
            }
            prev["content_blocks"].push_back(block);
        } else {
            json um = json::object();
            um["role"] = "user";
            um["content_blocks"] = json::array({block});
            out.push_back(um);
        }
    }
    return out;
}

inline std::string tool_result_wrap(const json& content) {
    std::string c;
    if (content.is_string()) c = content.get<std::string>();
    else if (content.is_array()) {
        std::vector<std::string> parts;
        for (const auto& b : content) {
            if (b.value("type", "") == "text") parts.push_back(b.value("text", ""));
            else parts.push_back("[Unsupported " + b.value("type", "") + "]");
        }
        for (size_t i = 0; i < parts.size(); ++i) { if (i) c += "\n\n"; c += parts[i]; }
    } else c = to_json(content);
    return "<tool_result>" + c + "</tool_result>";
}

// ---- render_message ----
inline std::string render_message(int index, const json& messages, const std::string& thinking_mode,
                                  bool drop_thinking = true, const std::string& reasoning_effort_in = "") {
    if (index < 0 || index >= (int)messages.size()) throw std::runtime_error("render_message: index out of range");
    if (thinking_mode != "chat" && thinking_mode != "thinking")
        throw std::runtime_error("Invalid thinking_mode `" + thinking_mode + "`");

    std::string prompt;
    const json& msg = messages[index];
    const int last_user_idx = find_last_user_index(messages);

    const std::string role = msg.value("role", "");
    const std::string content = msg.contains("content") && msg["content"].is_string() ? msg["content"].get<std::string>() : "";
    const bool has_tools = msg.contains("tools") && !msg["tools"].is_null() && !msg["tools"].empty();
    const bool wo_eos = msg.value("wo_eos", false);

    std::string effort = reasoning_effort_in.empty() ? DEFAULT_REASONING_EFFORT : reasoning_effort_in;
    if (!reasoning_effort_prompts().count(effort))
        throw std::runtime_error("Invalid reasoning effort: " + effort);
    if (index == 0 && thinking_mode == "thinking") prompt += reasoning_effort_prompts().at(effort);

    if (role == "system") {
        prompt += content;
        if (has_tools) prompt += "\n\n" + render_tools(tools_from_openai(msg["tools"]));
        if (msg.contains("response_format") && !msg["response_format"].is_null())
            prompt += "\n\n## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n"
                    + to_json(msg["response_format"]);
    } else if (role == "developer") {
        std::string c = std::string(USER_SP) + content;
        if (has_tools) c += "\n\n" + render_tools(tools_from_openai(msg["tools"]));
        if (msg.contains("response_format") && !msg["response_format"].is_null())
            c += "\n\n## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n"
               + to_json(msg["response_format"]);
        prompt += c;
    } else if (role == "user") {
        prompt += USER_SP;
        if (msg.contains("content_blocks") && msg["content_blocks"].is_array()) {
            std::vector<std::string> parts;
            for (const auto& b : msg["content_blocks"]) {
                const std::string bt = b.value("type", "");
                if (bt == "text") parts.push_back(b.value("text", ""));
                else if (bt == "tool_result") parts.push_back(tool_result_wrap(b["content"]));
                else parts.push_back("[Unsupported " + bt + "]");
            }
            for (size_t i = 0; i < parts.size(); ++i) { if (i) prompt += "\n\n"; prompt += parts[i]; }
        } else prompt += content;
    } else if (role == "latest_reminder") {
        prompt += std::string(LATEST_REMINDER_SP) + content;
    } else if (role == "tool") {
        throw std::runtime_error("tool role must be merged first — call merge_tool_messages()");
    } else if (role == "assistant") {
        std::string thinking_part, tc_content;
        if (msg.contains("tool_calls") && msg["tool_calls"].is_array() && !msg["tool_calls"].empty()) {
            std::string list;
            const auto& tcs = msg["tool_calls"];
            for (size_t i = 0; i < tcs.size(); ++i) {
                if (i) list += "\n";
                const json& fn = tcs[i].contains("function") ? tcs[i]["function"] : tcs[i];
                list += "<" + std::string(DSML) + "invoke name=\"" + fn.value("name", "") + "\">\n"
                      + encode_arguments_to_dsml(fn.value("arguments", "{}"))
                      + "\n</" + DSML + "invoke>";
            }
            tc_content += "\n\n<" + std::string(DSML) + "tool_calls>\n" + list + "\n</" + DSML + "tool_calls>";
        }
        const std::string rc = msg.contains("reasoning_content") && msg["reasoning_content"].is_string()
                             ? msg["reasoning_content"].get<std::string>() : "";
        const bool prev_has_task = index - 1 >= 0 && messages[index-1].contains("task")
                                   && !messages[index-1]["task"].is_null();
        if (thinking_mode == "thinking" && !prev_has_task) {
            if (!drop_thinking || index > last_user_idx) thinking_part = rc + THINK_END;
        }
        prompt += thinking_part + content + tc_content;
        if (!wo_eos) prompt += EOS;
    } else {
        throw std::runtime_error("Unknown role: " + role);
    }

    // transition tokens
    if (index + 1 < (int)messages.size()) {
        const std::string nr = messages[index+1].value("role", "");
        if (nr != "assistant" && nr != "latest_reminder") return prompt;
    }

    if (msg.contains("task") && !msg["task"].is_null()) {
        const std::string task = msg["task"].get<std::string>();
        if (!task_tokens().count(task)) throw std::runtime_error("Invalid task: '" + task + "'");
        const std::string tok = task_tokens().at(task);
        if (task != "action") prompt += tok;
        else {
            prompt += ASSISTANT_SP;
            prompt += (thinking_mode != "thinking") ? THINK_END : THINK_START;
            prompt += tok;
        }
    } else if (role == "user" || role == "developer") {
        prompt += ASSISTANT_SP;
        if (!drop_thinking && thinking_mode == "thinking")                         prompt += THINK_START;
        else if (drop_thinking && thinking_mode == "thinking" && index >= last_user_idx) prompt += THINK_START;
        else                                                                       prompt += THINK_END;
    }
    return prompt;
}

// _drop_thinking_messages
inline json drop_thinking_messages(const json& messages) {
    const int last_user = find_last_user_index(messages);
    json out = json::array();
    for (int i = 0; i < (int)messages.size(); ++i) {
        const std::string r = messages[i].value("role", "");
        const bool always_keep = (r == "user" || r == "system" || r == "tool" || r == "latest_reminder");
        if (always_keep || i >= last_user) { out.push_back(messages[i]); continue; }
        if (r == "developer") continue;                       // dropped entirely before the last user
        json m = messages[i];
        m.erase("reasoning_content");
        out.push_back(m);
    }
    return out;
}

// ---- encode_messages ----
inline std::string encode_messages(const json& messages_in, const std::string& thinking_mode,
                                   const json& context_in = json::array(),
                                   bool drop_thinking = true, bool add_default_bos_token = true,
                                   const std::string& reasoning_effort = "") {
    json context  = context_in.is_array() ? context_in : json::array();
    json messages = merge_tool_messages(messages_in);
    if (!context.empty()) context = merge_tool_messages(context);

    json full = json::array();
    for (const auto& m : context)  full.push_back(m);
    for (const auto& m : messages) full.push_back(m);

    std::string prompt = (add_default_bos_token && context.empty()) ? BOS : "";

    // if ANY message defines tools, thinking is never dropped (tool conversations need full context)
    bool effective_drop = drop_thinking;
    for (const auto& m : full)
        if (m.contains("tools") && !m["tools"].is_null() && !m["tools"].empty()) { effective_drop = false; break; }

    int num_to_render, context_len;
    if (thinking_mode == "thinking" && effective_drop) {
        full = drop_thinking_messages(full);
        const json dropped_ctx = drop_thinking_messages(context);
        num_to_render = (int)full.size() - (int)dropped_ctx.size();
        context_len   = (int)full.size() - num_to_render;
    } else {
        num_to_render = (int)messages.size();
        context_len   = (int)context.size();
    }

    for (int i = 0; i < num_to_render; ++i)
        prompt += render_message(i + context_len, full, thinking_mode, effective_drop, reasoning_effort);
    return prompt;
}

// ---- parse_message_from_completion_text ----
// Handles well-formed model output (the reference makes the same caveat).
inline json parse_message_from_completion_text(const std::string& text_in, const std::string& thinking_mode) {
    std::string text = text_in;
    const std::string eos = EOS;
    if (text.size() >= eos.size() && text.compare(text.size()-eos.size(), eos.size(), eos) == 0)
        text = text.substr(0, text.size()-eos.size());

    json out = json::object();
    out["role"] = "assistant";
    std::string reasoning, body = text;

    if (thinking_mode == "thinking") {
        const size_t e = text.find(THINK_END);
        if (e != std::string::npos) {
            reasoning = text.substr(0, e);
            const std::string st = THINK_START;
            if (reasoning.rfind(st, 0) == 0) reasoning = reasoning.substr(st.size());
            body = text.substr(e + std::string(THINK_END).size());
        }
    }

    // DSML tool_calls block
    json tool_calls = json::array();
    const std::string open_tc  = "<" + std::string(DSML) + "tool_calls>";
    const std::string close_tc = "</" + std::string(DSML) + "tool_calls>";
    const size_t tb = body.find(open_tc);
    if (tb != std::string::npos) {
        const size_t te = body.find(close_tc, tb);
        const std::string block = body.substr(tb + open_tc.size(),
                                              te == std::string::npos ? std::string::npos : te - tb - open_tc.size());
        if (te != std::string::npos) body = body.substr(0, tb) + body.substr(te + close_tc.size());
        else                        body = body.substr(0, tb);

        const std::string open_iv = "<" + std::string(DSML) + "invoke name=\"";
        const std::string close_iv = "</" + std::string(DSML) + "invoke>";
        size_t p = 0;
        while ((p = block.find(open_iv, p)) != std::string::npos) {
            const size_t nstart = p + open_iv.size();
            const size_t nend   = block.find('"', nstart);
            if (nend == std::string::npos) break;
            const std::string name = block.substr(nstart, nend - nstart);
            const size_t ivend = block.find(close_iv, nend);
            const std::string inner = block.substr(nend, (ivend == std::string::npos ? block.size() : ivend) - nend);

            // rebuild the arguments JSON, honouring string="true|false" per parameter
            std::string args = "{";
            const std::string open_p = "<" + std::string(DSML) + "parameter name=\"";
            const std::string close_p = "</" + std::string(DSML) + "parameter>";
            size_t q = 0; bool first = true;
            while ((q = inner.find(open_p, q)) != std::string::npos) {
                const size_t ks = q + open_p.size();
                const size_t ke = inner.find('"', ks);
                if (ke == std::string::npos) break;
                const std::string key = inner.substr(ks, ke - ks);
                const size_t ss = inner.find("string=\"", ke);
                bool is_str = true;
                size_t gt;
                if (ss != std::string::npos) {
                    const size_t se = inner.find('"', ss + 8);
                    is_str = inner.substr(ss + 8, se - ss - 8) == "true";
                    gt = inner.find('>', se);
                } else gt = inner.find('>', ke);
                if (gt == std::string::npos) break;
                const size_t ve = inner.find(close_p, gt);
                const std::string val = inner.substr(gt + 1, (ve == std::string::npos ? inner.size() : ve) - gt - 1);
                if (!first) args += ", ";
                first = false;
                args += to_json(json(key)) + ": " + (is_str ? to_json(json(val)) : val);
                q = (ve == std::string::npos) ? inner.size() : ve + close_p.size();
            }
            args += "}";

            json tc = json::object();
            tc["type"] = "function";
            tc["function"] = json::object();
            tc["function"]["name"] = name;
            tc["function"]["arguments"] = args;
            tool_calls.push_back(tc);
            p = (ivend == std::string::npos) ? block.size() : ivend + close_iv.size();
        }
    }

    // the reference strips the "\n\n" that joins content and the tool_calls block
    while (body.size() >= 2 && body.compare(body.size()-2, 2, "\n\n") == 0) body = body.substr(0, body.size()-2);

    out["reasoning_content"] = reasoning;
    out["content"] = body;
    out["tool_calls"] = tool_calls;
    return out;
}

} // namespace dsv4enc
