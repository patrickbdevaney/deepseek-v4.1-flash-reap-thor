// encoding_dsv41.h — C++ port of the V4.1 checkpoint's encoding/encoding.py (self-contained; it
// does NOT build on encoding_dsv4.h because nearly every rule moved).
//
// V4.1 vs V4 (0731), each one load-bearing for the byte-exact gate (tests/gate_encoding_v41.cpp):
//   1. DSML tag names carry a LEADING SPACE: "<｜DSML｜ calls>", "<｜DSML｜ invoke", "<｜DSML｜ parameter".
//   2. reasoning_effort is numeric 1..100 (or "low"/"high"/"max" = 50/75/100), DEFAULT "high", and is
//      rendered as "Reasoning Effort: N (range 1-100, ...)" behind a <｜System｜> token -- thinking
//      mode, message 0 only. V4's long prose prefixes are gone.
//   3. <｜System｜> leads message 0 when it is a system message or carries the effort prefix, and a
//      mid-conversation system message is "<｜System｜>content" and opens the assistant turn like a user.
//   4. There is no "developer" role. Every user message becomes content_blocks, and CONSECUTIVE user
//      messages merge into one (joined by "\n\n"); tool results become tool_result blocks sorted by
//      the preceding assistant's call order.
//   5. Image blocks become the <｜deepseek_image｜> placeholder (records returned in prompt order).
//   6. Tool names may be namespace-qualified ("ns::name").
#pragma once
#include "third_party/json.hpp"
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace dsv41enc {

using json = nlohmann::ordered_json;    // key order is part of the prompt (see encoding_dsv4.h)

inline constexpr const char* BOS = "<｜begin▁of▁sentence｜>";
inline constexpr const char* EOS = "<｜end▁of▁sentence｜>";
inline constexpr const char* THINK_START = "<think>";
inline constexpr const char* THINK_END = "</think>";
inline constexpr const char* DSML = "｜DSML｜";
inline constexpr const char* USER_SP = "<｜User｜>";
inline constexpr const char* ASSISTANT_SP = "<｜Assistant｜>";
inline constexpr const char* SYSTEM_SP = "<｜System｜>";
inline constexpr const char* LATEST_REMINDER_SP = "<｜latest_reminder｜>";
inline constexpr const char* IMAGE_PLACEHOLDER = "<｜deepseek_image｜>";
inline constexpr const char* CALLS = " calls";
inline constexpr const char* INVOKE = " invoke";
inline constexpr const char* PARAM = " parameter";

inline const std::map<std::string, std::string>& task_tokens() {
    static const std::map<std::string, std::string> m = {
        {"action", "<｜action｜>"}, {"query", "<｜query｜>"}, {"authority", "<｜authority｜>"},
        {"domain", "<｜domain｜>"}, {"title", "<｜title｜>"}, {"read_url", "<｜read_url｜>"},
    };
    return m;
}

// ---- helpers ----
// json.dumps(v, ensure_ascii=False): ", " / ": " separators, insertion order (ordered_json).
inline std::string to_json(const json& v) {
    if (v.is_object()) {
        std::string s = "{"; bool first = true;
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
inline std::string str_or(const json& m, const char* k, const std::string& d = "") {
    return m.contains(k) && m[k].is_string() ? m[k].get<std::string>() : d;
}
inline bool truthy(const json& m, const char* k) {
    if (!m.contains(k) || m[k].is_null()) return false;
    const json& v = m[k];
    if (v.is_array() || v.is_object() || v.is_string()) return !v.empty();
    if (v.is_boolean()) return v.get<bool>();
    return true;
}
inline std::string join(const std::vector<std::string>& v, const std::string& sep) {
    std::string s; for (size_t i = 0; i < v.size(); ++i) { if (i) s += sep; s += v[i]; } return s;
}
// Python str() of a non-list tool_result content, for the cases json can carry
inline std::string py_str(const json& v) {
    if (v.is_string()) return v.get<std::string>();
    if (v.is_null()) return "None";
    if (v.is_boolean()) return v.get<bool>() ? "True" : "False";
    return to_json(v);
}

// ---- tool names: "ns::name" ----
inline void split_tool_name(const std::string& qualified, const json& ns_in, json& ns_out, std::string& name) {
    ns_out = ns_in; name = qualified;
    const size_t p = qualified.find("::");
    if (p != std::string::npos) {
        const std::string prefix = qualified.substr(0, p);
        if (!ns_in.is_null() && !(ns_in.is_string() && ns_in.get<std::string>() == prefix))
            throw std::runtime_error("Conflicting tool namespaces");
        ns_out = prefix; name = qualified.substr(p + 2);
    }
    if (name.find("::") != std::string::npos) throw std::runtime_error("Tool name must not contain '::': " + name);
}
inline std::string tool_name_for_encoding(const json& tool) {
    json ns = tool.contains("namespace") ? tool["namespace"] : json();
    if (ns.is_object()) ns = ns["name"];
    json o; std::string name;
    split_tool_name(str_or(tool, "name"), ns, o, name);
    return o.is_null() ? name : o.get<std::string>() + "::" + name;
}
inline json tools_from_openai(const json& tools) {
    json out = json::array();
    for (const auto& t : tools) {
        json f = t["function"];
        if (t.contains("namespace") && !t["namespace"].is_null()) f["namespace"] = t["namespace"];
        f["name"] = tool_name_for_encoding(f);
        json ns = f.contains("namespace") ? f["namespace"] : json();
        f.erase("namespace");
        if (ns.is_object() && truthy(ns, "description"))
            f["description"] = ns["description"].get<std::string>() + "\n" + str_or(f, "description");
        out.push_back(f);
    }
    return out;
}

inline std::string render_tools(const json& fns) {
    std::vector<std::string> schemas; for (const auto& f : fns) schemas.push_back(to_json(f));
    const std::string D = DSML;
    return "## Tools\n\nYou have access to a set of tools to help answer the user's question. "
           "You can invoke tools by writing a \"<" + D + CALLS + ">\" block like the following:\n\n"
           "<" + D + CALLS + ">\n"
           "<" + D + INVOKE + " name=\"$TOOL_NAME\">\n"
           "<" + D + PARAM + " name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</" + D + PARAM + ">\n"
           "...\n"
           "</" + D + INVOKE + ">\n"
           "<" + D + INVOKE + " name=\"$TOOL_NAME2\">\n"
           "...\n"
           "</" + D + INVOKE + ">\n"
           "</" + D + CALLS + ">\n\n"
           "String parameters should be specified as is and set `string=\"true\"`. For all other types "
           "(numbers, booleans, arrays, objects), pass the value in JSON format and set `string=\"false\"`.\n\n"
           "If thinking_mode is enabled (triggered by " + THINK_START + "), you MUST output your complete "
           "reasoning inside " + THINK_START + "..." + THINK_END + " BEFORE any tool calls or final response.\n\n"
           "Otherwise, output directly after " + THINK_END + " with tool calls or final response.\n\n"
           "### Available Tool Schemas\n\n" + join(schemas, "\n") + "\n\n"
           "You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.\n";
}

// arguments: a dict, a JSON string, or a doubly-encoded JSON string; anything else -> {"arguments": raw}
inline std::string encode_arguments_to_dsml(const json& raw) {
    json a = raw;
    if (!a.is_object())
        for (int i = 0; i < 2 && a.is_string(); ++i) {
            try { a = json::parse(a.get<std::string>()); } catch (...) { break; }
        }
    if (!a.is_object()) { json o = json::object(); o["arguments"] = raw; a = o; }
    std::vector<std::string> ps;
    const std::string D = DSML;
    for (auto it = a.begin(); it != a.end(); ++it) {
        const bool s = it.value().is_string();
        ps.push_back("<" + D + PARAM + " name=\"" + it.key() + "\" string=\"" + (s ? "true" : "false") + "\">" +
                     (s ? it.value().get<std::string>() : to_json(it.value())) + "</" + D + PARAM + ">");
    }
    return join(ps, "\n");
}

// numeric effort: int in [1,100] or low/high/max. `effort` null -> "high".
inline int effort_budget(const json& effort) {
    if (effort.is_null()) return 75;
    if (effort.is_number_integer()) {
        const long long v = effort.get<long long>();
        if (v >= 1 && v <= 100) return (int)v;
    } else if (effort.is_string()) {
        const std::string s = effort.get<std::string>();
        if (s == "low") return 50;
        if (s == "high") return 75;
        if (s == "max") return 100;
    }
    throw std::runtime_error("Invalid reasoning effort for deepseek_v41: " + effort.dump() +
                             ", should be int within [1,100] or ['low', 'high', 'max']");
}
inline std::string render_reasoning_effort(int index, const std::string& mode, const json& effort) {
    const int b = effort_budget(effort);
    if (index == 0 && mode == "thinking")
        return "Reasoning Effort: " + std::to_string(b) +
               " (range 1-100, the higher the value, the more thorough the reasoning)\n\n";
    return "";
}

// ---- preprocessing ----
inline void validate_no_image_tokens(const json& m) {
    if (str_or(m, "content").find(IMAGE_PLACEHOLDER) != std::string::npos)
        throw std::runtime_error("Message content contains image special token");
    if (str_or(m, "reasoning_content").find(IMAGE_PLACEHOLDER) != std::string::npos)
        throw std::runtime_error("reasoning_content contains image special token");
}

inline json merge_tool_messages(const json& messages) {
    json merged = json::array();
    for (json msg : messages) {
        const std::string role = str_or(msg, "role");
        if (role == "tool") {
            json content = msg.contains("content") ? msg["content"] : json("");
            if (truthy(msg, "content_blocks") && !(msg.contains("content") && msg["content"].is_string()))
                content = msg["content_blocks"];
            json b = json::object();
            b["type"] = "tool_result"; b["tool_use_id"] = msg.contains("tool_call_id") ? msg["tool_call_id"] : json("");
            b["content"] = content;
            if (!merged.empty() && str_or(merged.back(), "role") == "user" && merged.back().contains("content_blocks"))
                merged.back()["content_blocks"].push_back(b);
            else { json u = json::object(); u["role"] = "user"; u["content_blocks"] = json::array({b}); merged.push_back(u); }
        } else if (role == "user") {
            json blocks = msg.contains("content_blocks") ? msg["content_blocks"] : json();
            if (blocks.is_null()) {
                json c = msg.contains("content") ? msg["content"] : json("");
                if (c.is_array()) blocks = c;
                else { json t = json::object(); t["type"] = "text"; t["text"] = c; blocks = json::array({t}); }
            }
            if (!merged.empty() && str_or(merged.back(), "role") == "user" && merged.back().contains("content_blocks") &&
                (!merged.back().contains("task") || merged.back()["task"].is_null())) {
                for (auto& b : blocks) merged.back()["content_blocks"].push_back(b);
            } else { msg["content_blocks"] = blocks; merged.push_back(msg); }
        } else merged.push_back(msg);
    }
    return merged;
}

inline json sort_tool_results_by_call_order(json messages) {
    std::map<std::string, int> order;
    for (auto& msg : messages) {
        const std::string role = str_or(msg, "role");
        if (role == "assistant" && truthy(msg, "tool_calls")) {
            order.clear();
            int i = 0;
            for (const auto& tc : msg["tool_calls"]) {
                std::string id = str_or(tc, "id");
                if (id.empty() && tc.contains("function")) id = str_or(tc["function"], "id");
                if (!id.empty()) order[id] = i;
                ++i;
            }
        } else if (role == "user" && truthy(msg, "content_blocks")) {
            std::vector<json> tb;
            for (const auto& b : msg["content_blocks"]) if (str_or(b, "type") == "tool_result") tb.push_back(b);
            if (tb.size() > 1 && !order.empty()) {
                auto key = [&](const json& b) { auto it = order.find(str_or(b, "tool_use_id")); return it == order.end() ? 0 : it->second; };
                std::stable_sort(tb.begin(), tb.end(), [&](const json& a, const json& b) { return key(a) < key(b); });
                size_t k = 0;
                for (auto& b : msg["content_blocks"]) if (str_or(b, "type") == "tool_result") b = tb[k++];
            }
        }
    }
    return messages;
}

inline bool is_image_block(const json& b) {
    const std::string t = str_or(b, "type"); return b.is_object() && (t == "image" || t == "image_url");
}
inline json extract_image(const json& b) {
    json r = json::object(); r["type"] = "image";
    if (str_or(b, "type") == "image_url") {
        const json& iu = b.contains("image_url") ? b["image_url"] : json();
        r["url"] = iu.is_string() ? iu : (iu.is_object() && iu.contains("url") ? iu["url"] : json(""));
    } else for (const char* k : {"source", "url", "data"}) if (b.contains(k)) r[k] = b[k];
    if (!truthy(r, "source") && !truthy(r, "url") && !truthy(r, "data"))
        throw std::runtime_error("Image block does not contain a valid source");
    return r;
}
inline json process_image_blocks(const json& blocks, json& images) {
    json out = json::array();
    for (const auto& b : blocks) {
        if (!b.is_object()) { out.push_back(b); continue; }
        if (is_image_block(b)) {
            json t = json::object(); t["type"] = "text"; t["text"] = IMAGE_PLACEHOLDER;
            out.push_back(t); images.push_back(extract_image(b));
        } else if (str_or(b, "type") == "tool_result" && b.contains("content") && b["content"].is_array()) {
            json c = b; c["content"] = process_image_blocks(b["content"], images); out.push_back(c);
        } else {
            if (str_or(b, "type") == "text" && str_or(b, "text").find(IMAGE_PLACEHOLDER) != std::string::npos)
                throw std::runtime_error("Text block contains image placeholder; images should be separate content blocks");
            out.push_back(b);
        }
    }
    return out;
}
inline json process_image_messages(const json& messages, json& images) {
    json out = json::array();
    for (json m : messages) {
        validate_no_image_tokens(m);
        if (m.contains("content") && m["content"].is_array() && !m.contains("content_blocks")) {
            m["content_blocks"] = m["content"]; m.erase("content");
        }
        if (truthy(m, "content_blocks")) {
            m["content_blocks"] = process_image_blocks(m["content_blocks"], images);
            if (!(m.contains("content") && m["content"].is_string())) {
                std::vector<std::string> t;
                for (const auto& b : m["content_blocks"])
                    if (b.is_object() && str_or(b, "type") == "text") t.push_back(str_or(b, "text"));
                m["content"] = join(t, "\n\n");
            }
        }
        out.push_back(m);
    }
    return out;
}

// a user message holding only tool results does not start a turn; mid-conversation systems do
inline int find_last_user_index(const json& msgs) {
    for (int i = (int)msgs.size() - 1; i >= 0; --i) {
        const std::string r = str_or(msgs[i], "role");
        if (r == "user" || (r == "system" && i > 0)) {
            if (truthy(msgs[i], "content_blocks")) {
                bool any = false;
                for (const auto& b : msgs[i]["content_blocks"]) {
                    const std::string t = str_or(b, "type");
                    any |= t == "text" || t == "image" || t == "image_url";
                }
                if (!any) continue;
            }
            return i;
        }
    }
    return -1;
}

// ---- render_message ----
inline std::string render_message(int index, const json& messages, const std::string& mode, bool drop_thinking = true,
                                  const json& reasoning_effort = json()) {
    if (index < 0 || index >= (int)messages.size()) throw std::runtime_error("render_message: index out of range");
    if (mode != "chat" && mode != "thinking") throw std::runtime_error("Invalid thinking_mode `" + mode + "`");
    const json& msg = messages[index];
    const int last_user = find_last_user_index(messages);
    const std::string role = str_or(msg, "role");
    const std::string content = str_or(msg, "content");
    const std::string D = DSML;

    const std::string effort = render_reasoning_effort(index, mode, reasoning_effort);
    std::string prompt = (index == 0 && (!effort.empty() || role == "system")) ? SYSTEM_SP : "";
    prompt += effort;

    if (role == "system") {
        if (index > 0) prompt += SYSTEM_SP;
        prompt += content;
        if (truthy(msg, "tools")) prompt += "\n\n" + render_tools(tools_from_openai(msg["tools"]));
        if (truthy(msg, "response_format"))
            prompt += "\n\n## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n" +
                      to_json(msg["response_format"]);
    } else if (role == "user") {
        prompt += USER_SP;
        if (truthy(msg, "content_blocks")) {
            std::vector<std::string> parts;
            for (const auto& b : msg["content_blocks"]) {
                const std::string bt = str_or(b, "type");
                if (bt == "text") parts.push_back(str_or(b, "text"));
                else if (bt == "tool_result") {
                    const json tc = b.contains("content") ? b["content"] : json("");
                    std::string c;
                    if (tc.is_array()) {
                        std::vector<std::string> tp;
                        for (const auto& x : tc)
                            tp.push_back(str_or(x, "type") == "text" ? str_or(x, "text") : "[Unsupported " + py_str(x.contains("type") ? x["type"] : json()) + "]");
                        c = join(tp, "\n\n");
                    } else c = py_str(tc);
                    parts.push_back("<tool_result>" + c + "</tool_result>");
                } else parts.push_back("[Unsupported " + py_str(b.contains("type") ? b["type"] : json()) + "]");
            }
            prompt += join(parts, "\n\n");
        } else prompt += content;
    } else if (role == "latest_reminder") {
        prompt += std::string(LATEST_REMINDER_SP) + content;
    } else if (role == "tool") {
        throw std::runtime_error("deepseek_v41 merges tool messages into user; call merge_tool_messages()");
    } else if (role == "assistant") {
        std::string tc_content;
        if (truthy(msg, "tool_calls")) {
            std::vector<std::string> list;
            for (const auto& tc : msg["tool_calls"]) {
                const json& fn = tc["function"];
                json ns_in = tc.contains("namespace") && !tc["namespace"].is_null() ? tc["namespace"]
                           : (fn.contains("namespace") ? fn["namespace"] : json());
                json ns; std::string name;
                split_tool_name(str_or(fn, "name"), ns_in, ns, name);
                const std::string qn = ns.is_null() ? name : ns.get<std::string>() + "::" + name;
                list.push_back("<" + D + INVOKE + " name=\"" + qn + "\">\n" +
                               encode_arguments_to_dsml(fn.contains("arguments") ? fn["arguments"] : json("{}")) +
                               "\n</" + D + INVOKE + ">");
            }
            tc_content = "\n\n<" + D + CALLS + ">\n" + join(list, "\n") + "\n</" + D + CALLS + ">";
        }
        const bool prev_task = index >= 1 && messages[index - 1].contains("task") && !messages[index - 1]["task"].is_null();
        std::string thinking;
        if (mode == "thinking" && !prev_task && (!drop_thinking || index > last_user))
            thinking = str_or(msg, "reasoning_content") + THINK_END;
        prompt += thinking + content + tc_content;
        if (!(msg.contains("wo_eos") && msg["wo_eos"].is_boolean() && msg["wo_eos"].get<bool>())) prompt += EOS;
    } else throw std::runtime_error("Unknown role: " + role);

    if (index + 1 < (int)messages.size()) {
        const std::string nr = str_or(messages[index + 1], "role");
        if (nr != "assistant" && nr != "latest_reminder") return prompt;
    }
    if (msg.contains("task") && !msg["task"].is_null()) {
        const std::string task = msg["task"].get<std::string>();
        if (!task_tokens().count(task)) throw std::runtime_error("Invalid task: '" + task + "'");
        const std::string tok = task_tokens().at(task);
        if (task != "action") prompt += tok;
        else { prompt += ASSISTANT_SP; prompt += mode != "thinking" ? THINK_END : THINK_START; prompt += tok; }
    } else if (role == "user" || (role == "system" && index > 0)) {
        prompt += ASSISTANT_SP;
        if (!drop_thinking && mode == "thinking") prompt += THINK_START;
        else if (drop_thinking && mode == "thinking" && index >= last_user) prompt += THINK_START;
        else prompt += THINK_END;
    }
    return prompt;
}

inline json drop_thinking_messages(const json& messages) {
    const int last_user = find_last_user_index(messages);
    json out = json::array();
    for (int i = 0; i < (int)messages.size(); ++i) {
        const std::string r = str_or(messages[i], "role");
        const bool keep = r == "user" || r == "system" || r == "tool" || r == "latest_reminder" || r == "direct_search_results";
        if (keep || i >= last_user) out.push_back(messages[i]);
        else if (r == "assistant") { json m = messages[i]; m.erase("reasoning_content"); out.push_back(m); }
    }
    return out;
}

// encode_messages; `images` (optional) receives the current-turn image records in prompt order.
inline std::string encode_messages(const json& messages_in, const std::string& mode, const json& context_in = json::array(),
                                   bool drop_thinking = true, bool add_bos = true, const json& reasoning_effort = json(),
                                   json* images = nullptr) {
    json context = context_in.is_array() ? context_in : json::array();
    for (const auto& m : context) validate_no_image_tokens(m);
    for (const auto& m : messages_in) validate_no_image_tokens(m);
    context = merge_tool_messages(context);
    json messages = merge_tool_messages(messages_in);
    {
        json all = context; for (const auto& m : messages) all.push_back(m);
        all = sort_tool_results_by_call_order(all);
        messages = json::array();
        for (size_t i = context.size(); i < all.size(); ++i) messages.push_back(all[i]);
    }
    json ctx_imgs = json::array(), imgs = json::array();
    if (!context.empty()) context = process_image_messages(context, ctx_imgs);
    messages = process_image_messages(messages, imgs);
    if (images) *images = imgs;

    json full = context; for (const auto& m : messages) full.push_back(m);
    std::string prompt = (add_bos && context.empty()) ? BOS : "";
    bool drop = drop_thinking;
    for (const auto& m : full) if (truthy(m, "tools")) { drop = false; break; }
    int n_render, ctx_len;
    if (mode == "thinking" && drop) {
        full = drop_thinking_messages(full);
        n_render = (int)full.size() - (int)drop_thinking_messages(context).size();
        ctx_len = (int)full.size() - n_render;
    } else { n_render = (int)messages.size(); ctx_len = (int)context.size(); }
    for (int i = 0; i < n_render; ++i) prompt += render_message(i + ctx_len, full, mode, drop, reasoning_effort);
    return prompt;
}

// ---- parse_message_from_completion_text ----
// strict = the reference's asserts (gate); lenient (server) accepts a leading <think>, a missing
// </think> or EOS (max_tokens cut) and returns whatever parsed.
inline size_t read_until(size_t i, const std::string& t, const std::vector<std::string>& stops, std::string& got, std::string& hit) {
    size_t best = t.size(); hit.clear();
    for (const auto& s : stops) { const size_t p = t.find(s, i); if (p != std::string::npos && p < best) { best = p; hit = s; } }
    got = t.substr(i, best - i);
    return hit.empty() ? t.size() : best + hit.size();
}

inline json parse_message_from_completion_text(const std::string& text_in, const std::string& mode, bool strict = false) {
    std::string text = text_in;
    if (!strict && text.rfind(THINK_START, 0) == 0) text = text.substr(std::string(THINK_START).size());
    const std::string D = DSML;
    const std::string tc_start = "\n\n<" + D + CALLS;
    std::string reasoning, content, got, hit;
    size_t i = 0;
    auto fail = [&](const std::string& why) { if (strict) throw std::runtime_error(why); };
    if (mode == "thinking") {
        i = read_until(i, text, {THINK_END, tc_start}, got, hit);
        if (hit != THINK_END) { fail("Invalid thinking format: missing </think>");
            if (hit.empty()) { json o; o["role"] = "assistant"; o["content"] = ""; o["reasoning_content"] = got;
                               o["tool_calls"] = json::array(); return o; } }
        reasoning = got;
        if (hit == tc_start) i -= tc_start.size();
    }
    i = read_until(i, text, {EOS, tc_start}, got, hit);
    content = got;
    json calls = json::array();
    if (hit == tc_start) {
        const std::string calls_end = "</" + D + CALLS + ">", inv_s = "<" + D + INVOKE, inv_e = "</" + D + INVOKE,
                          par_s = "<" + D + PARAM, par_e = "/" + D + PARAM;
        while (i < text.size()) {
            i = read_until(i, text, {inv_s, calls_end}, got, hit);
            if (got != ">\n") { fail("Tool call format error"); break; }
            if (hit == calls_end || hit.empty()) { if (hit.empty()) fail("Missing special token in tool calls"); break; }
            i = read_until(i, text, {par_s, inv_e}, got, hit);
            const size_t q0 = got.find("name=\""), q1 = got.rfind("\">\n");
            if (q0 == std::string::npos || q1 == std::string::npos || q1 < q0 + 6) { fail("Tool name format error"); break; }
            const std::string qname = got.substr(q0 + 6, q1 - q0 - 6);
            std::vector<std::pair<std::string, std::pair<std::string, bool>>> args;
            bool bad = false;
            while (hit == par_s) {
                i = read_until(i, text, {par_e}, got, hit);
                // ' name="K" string="true|false">V<'
                const std::string pre = " name=\"";
                const size_t ke = got.find("\" string=\"");
                if (got.rfind(pre, 0) != 0 || ke == std::string::npos || got.empty() || got.back() != '<') { bad = true; break; }
                const std::string key = got.substr(pre.size(), ke - pre.size());
                const size_t vs = ke + 10;
                bool is_str;
                if (got.compare(vs, 6, "true\">") == 0) { is_str = true; }
                else if (got.compare(vs, 7, "false\">") == 0) { is_str = false; }
                else { bad = true; break; }
                const size_t v0 = vs + (is_str ? 6 : 7);
                for (auto& a : args) if (a.first == key) { bad = true; }
                args.push_back({key, {got.substr(v0, got.size() - 1 - v0), is_str}});
                i = read_until(i, text, {par_s, inv_e}, got, hit);
                if (got != ">\n") { bad = true; break; }
            }
            if (bad) { fail("Parameter format error"); break; }
            std::string a = "{"; bool first = true;
            for (auto& kv : args) {
                if (!first) a += ", ";
                first = false;
                a += to_json(json(kv.first)) + ": " + (kv.second.second ? to_json(json(kv.second.first)) : kv.second.first);
            }
            a += "}";
            json ns; std::string name; split_tool_name(qname, json(), ns, name);
            json c = json::object(); c["type"] = "function";
            c["function"] = json::object(); c["function"]["name"] = name; c["function"]["arguments"] = a;
            if (!ns.is_null()) c["namespace"] = ns;
            calls.push_back(c);
        }
        i = read_until(i, text, {EOS}, got, hit);
        if (!got.empty()) fail("Unexpected content after tool calls");
    } else if (hit != EOS) fail("Invalid format: missing EOS token");
    if (strict) {
        if (i != text.size()) throw std::runtime_error("Unexpected content at end");
        for (const char* sp : {BOS, EOS, THINK_START, THINK_END, DSML})
            if (content.find(sp) != std::string::npos || reasoning.find(sp) != std::string::npos)
                throw std::runtime_error(std::string("Unexpected special token '") + sp + "' in content");
    }
    json o = json::object();
    o["role"] = "assistant"; o["content"] = content; o["reasoning_content"] = reasoning; o["tool_calls"] = calls;
    return o;
}

} // namespace dsv41enc
