#!/usr/bin/env python3
"""C3 + V3 end to end against a running v41_server (stdlib only).
  c3_http.py <port> <image.jpg|-> <out.json>
C3: json_schema (chat + thinking, non-streaming + streaming), json_object, lark grammar, malformed spec -> 400,
an unconstrained control, and non-streaming cancellation (a client that hangs up must free the server).
V3: a two-turn image conversation; turn 2 must reuse the prompt prefix (cached_tokens) and answer."""
import base64, json, sys, time, urllib.request, urllib.error, socket

port, image, outp = int(sys.argv[1]), sys.argv[2], sys.argv[3]
URL = f"http://127.0.0.1:{port}/v1/chat/completions"
res, fails = {}, []

def post(body, timeout=900):
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read()), time.time() - t0
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}"), time.time() - t0

def post_stream(body):
    body = dict(body, stream=True)
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    content, reasoning = "", ""
    with urllib.request.urlopen(req, timeout=900) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]": continue
            o = json.loads(line[6:])
            for ch in o.get("choices", []):
                d = ch.get("delta", {})
                content += d.get("content") or ""; reasoning += d.get("reasoning_content") or ""
    return content, reasoning

def check(name, ok, info):
    res[name] = {"ok": bool(ok), **info}
    print(f"[c3] {name}: {'OK' if ok else 'FAIL'} {json.dumps(info)[:300]}", flush=True)
    if not ok: fails.append(name)

SCHEMA = {"type": "object", "properties": {"city": {"type": "string"}, "country": {"type": "string"},
          "population_millions": {"type": "number"}, "landmarks": {"type": "array", "items": {"type": "string"}, "maxItems": 3}},
          "required": ["city", "country", "population_millions", "landmarks"], "additionalProperties": False}
def valid(o):
    return (isinstance(o, dict) and set(o) == set(SCHEMA["required"]) and isinstance(o["city"], str)
            and isinstance(o["country"], str) and isinstance(o["population_millions"], (int, float))
            and isinstance(o["landmarks"], list) and len(o["landmarks"]) <= 3 and all(isinstance(x, str) for x in o["landmarks"]))
MSG = [{"role": "user", "content": "Tell me about Lisbon."}]
RF = {"type": "json_schema", "json_schema": {"name": "city", "schema": SCHEMA, "strict": True}}

# thinking: the constraint engages after </think> (as vLLM's), so the prompt must say what it wants, like real clients
MSG_T = [{"role": "user", "content": "Tell me about Lisbon. Answer as a JSON object with city, country, population_millions and up to 3 landmarks."}]
for mode, mt in (("chat", 256), ("thinking", 8192)):
    # thinking at T=0.6: greedy reasoning loops on this model ("Let's use: ...? No. ... Hmm." to any budget, q16),
    # which is the model's chain of thought, not the constraint (inactive until </think>); DeepSeek samples thinking
    samp = {"temperature": 0} if mode == "chat" else {"temperature": 0.6, "top_p": 0.95, "seed": 7}
    st, o, dt = post({"messages": MSG if mode == "chat" else MSG_T, "response_format": RF, "max_tokens": mt,
                      "thinking_mode": mode, **samp})
    c = o.get("choices", [{}])[0].get("message", {}).get("content", "") if st == 200 else ""
    try: parsed = json.loads(c); ok = valid(parsed)
    except Exception: parsed, ok = None, False
    check(f"schema_{mode}", st == 200 and ok, {"status": st, "content": c[:200], "s": round(dt, 1),
          "reasoning_chars": len(o.get("choices", [{}])[0].get("message", {}).get("reasoning_content", "") or "") if st == 200 else 0,
          "reasoning_tail": (o.get("choices", [{}])[0].get("message", {}).get("reasoning_content", "") or "")[-160:] if st == 200 else "",
          "tok_s": o.get("timings", {}).get("tokens_per_second"), "grammar_error": o.get("grammar_error")})

c, r = post_stream({"messages": MSG, "response_format": RF, "temperature": 0, "max_tokens": 256, "thinking_mode": "chat"})
try: ok = valid(json.loads(c))
except Exception: ok = False
check("schema_stream", ok, {"content": c[:200]})

c, r = post_stream({"messages": [{"role": "user", "content": "Say hello in one short sentence."}], "temperature": 0,
                    "max_tokens": 40, "thinking_mode": "chat"})
check("stream_control_no_eos_text", len(c) > 0 and "end\u2581of" not in c and "<\uff5c" not in c, {"content": c[:120]})

st, o, dt = post({"messages": [{"role": "user", "content": "Give me a JSON object describing a cat."}],
                  "response_format": {"type": "json_object"}, "temperature": 0, "max_tokens": 200, "thinking_mode": "chat"})
c = o.get("choices", [{}])[0].get("message", {}).get("content", "") if st == 200 else ""
try: ok = isinstance(json.loads(c), dict)
except Exception: ok = False
check("json_object", st == 200 and ok, {"status": st, "content": c[:200]})

st, o, dt = post({"messages": [{"role": "user", "content": "Is the Pacific the largest ocean? Answer."}],
                  "grammar": {"type": "lark", "data": 'start: "Yes" | "No"'}, "temperature": 0, "max_tokens": 16, "thinking_mode": "chat"})
c = o.get("choices", [{}])[0].get("message", {}).get("content", "") if st == 200 else ""
check("lark_choice", st == 200 and c in ("Yes", "No"), {"status": st, "content": c})

st, o, dt = post({"messages": MSG, "grammar": {"type": "lark", "data": "start: ("}, "max_tokens": 8})
check("bad_grammar_400", st == 400, {"status": st, "msg": str(o)[:160]})

st, o, dt = post({"messages": [{"role": "user", "content": "Say hello in one short sentence."}], "temperature": 0,
                  "max_tokens": 40, "thinking_mode": "chat"})
c = o.get("choices", [{}])[0].get("message", {}).get("content", "") if st == 200 else ""
check("control", st == 200 and len(c) > 0, {"status": st, "content": c[:120], "tok_s": o.get("timings", {}).get("tokens_per_second")})

# cancellation: hang up after 4 s on a 3000-token request; the next request must not wait for it to finish
long_body = {"messages": [{"role": "user", "content": "Write a very long, detailed history of the Roman Empire."}],
             "temperature": 0, "max_tokens": 3000, "thinking_mode": "chat"}
try:
    post(long_body, timeout=4)
except (socket.timeout, TimeoutError, urllib.error.URLError):
    pass
st, o, dt = post({"messages": [{"role": "user", "content": "Say OK."}], "temperature": 0, "max_tokens": 8, "thinking_mode": "chat"})
check("cancel_nonstream", st == 200 and dt < 30, {"status": st, "next_request_s": round(dt, 1)})

# V3: two turns over one image; turn 2 must hit the prefix cache and answer
if image != "-":
    url = "data:image/jpeg;base64," + base64.b64encode(open(image, "rb").read()).decode()
    m1 = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                       {"type": "text", "text": "What is the name of this newspaper?"}]}]
    st, o, dt = post({"messages": m1, "temperature": 0, "max_tokens": 64, "thinking_mode": "chat"})
    a1 = o.get("choices", [{}])[0].get("message", {}).get("content", "") if st == 200 else ""
    check("v3_turn1", st == 200 and len(a1) > 0, {"status": st, "content": a1[:160], "s": round(dt, 1),
          "prompt_tokens": o.get("usage", {}).get("prompt_tokens"), "prefill_ms": o.get("timings", {}).get("prefill_ms")})
    m2 = m1 + [{"role": "assistant", "content": a1}, {"role": "user", "content": "And what date is printed on it?"}]
    st, o, dt = post({"messages": m2, "temperature": 0, "max_tokens": 64, "thinking_mode": "chat"})
    a2 = o.get("choices", [{}])[0].get("message", {}).get("content", "") if st == 200 else ""
    cached = o.get("usage", {}).get("prompt_tokens_details", {}).get("cached_tokens", 0) if st == 200 else 0
    check("v3_turn2_cached", st == 200 and len(a2) > 0 and cached > 0,
          {"status": st, "content": a2[:160], "cached_tokens": cached, "prompt_tokens": o.get("usage", {}).get("prompt_tokens"),
           "prefill_ms": o.get("timings", {}).get("prefill_ms"), "s": round(dt, 1)})

json.dump(res, open(outp, "w"), indent=1)
print(f"[c3] {len(res) - len(fails)}/{len(res)} passed" + (f"; FAILED {fails}" if fails else ""))
sys.exit(1 if fails else 0)
