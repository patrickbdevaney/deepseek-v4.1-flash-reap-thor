#!/usr/bin/env python3
"""C3 control: does the constraint touch the reasoning? Greedy thinking with and without response_format must give
token-identical reasoning (the matcher is inactive until </think>); plus one sampled thinking run with no schema.
  c3_think_ctl.py <port> <out.json>"""
import json, sys, urllib.request
port, outp = int(sys.argv[1]), sys.argv[2]
URL = f"http://127.0.0.1:{port}/v1/chat/completions"
def post(body):
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        m = json.loads(r.read())["choices"][0]["message"]
        return m.get("reasoning_content") or "", m.get("content") or ""
MSG = [{"role": "user", "content": "Tell me about Lisbon. Answer as a JSON object with city, country, population_millions and up to 3 landmarks."}]
SCHEMA = {"type": "object", "properties": {"city": {"type": "string"}, "country": {"type": "string"},
          "population_millions": {"type": "number"}, "landmarks": {"type": "array", "items": {"type": "string"}, "maxItems": 3}},
          "required": ["city", "country", "population_millions", "landmarks"], "additionalProperties": False}
RF = {"type": "json_schema", "json_schema": {"name": "city", "schema": SCHEMA, "strict": True}}
base = {"messages": MSG, "max_tokens": 3000, "thinking_mode": "thinking", "temperature": 0}
r_s, c_s = post(dict(base, response_format=RF))
r_n, c_n = post(base)
same = r_s == r_n
div = next((i for i, (a, b) in enumerate(zip(r_s, r_n)) if a != b), min(len(r_s), len(r_n)))
r_t, c_t = post({"messages": MSG, "max_tokens": 8192, "thinking_mode": "thinking", "temperature": 0.6, "top_p": 0.95, "seed": 7})
loop = lambda r: r[-400:].count(r[-60:-20]) > 2 if len(r) > 400 else False
res = {"greedy_reasoning_identical": same, "first_diff_char": div, "len_schema": len(r_s), "len_free": len(r_n),
       "greedy_free_tail": r_n[-200:], "greedy_free_content": c_n[:200], "greedy_looping": loop(r_n),
       "sampled_free_reasoning_chars": len(r_t), "sampled_free_content": c_t[:300], "sampled_free_tail": r_t[-200:], "sampled_looping": loop(r_t)}
for k, v in res.items(): print(f"[ctl] {k}: {json.dumps(v)}")
json.dump(res, open(outp, "w"), indent=1)
