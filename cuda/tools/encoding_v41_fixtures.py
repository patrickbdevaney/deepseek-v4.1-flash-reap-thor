#!/usr/bin/env python3
"""Differential fixtures for the C++ V4.1 encoder: random conversations encoded BY THE REFERENCE
(encoding/encoding.py from the checkpoint) -> tests/data/encoding_v41_fuzz.json, which
tests/gate_encoding_v41.cpp replays. Covers what the 5 shipped vectors do not: mid-conversation
systems, tool results out of call order, namespaces, double-encoded arguments, numeric efforts,
tasks, context=, drop_thinking across turns, consecutive user merges.
Usage: python -I tools/encoding_v41_fixtures.py --ckpt ~/models/DeepSeek-V4.1-Flash-1017 --out tests/data
"""
import argparse, copy, json, os, random, sys

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=os.path.expanduser("~/models/DeepSeek-V4.1-Flash-1017"))
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=400)
a = ap.parse_args()
sys.path.insert(0, os.path.join(a.ckpt, "encoding"))
import encoding as enc  # noqa: E402

R = random.Random(41)
WORDS = ["alpha", "Beta", "γάμμα", "深度", "x=1", "tab\there", "line\nbreak", "\"quoted\"", "emoji 🙂", ""]
def txt(): return " ".join(R.choice(WORDS) for _ in range(R.randint(0, 5)))
TOOLS = [
    {"type": "function", "function": {"name": "lookup", "description": "Look up", "parameters": {"type": "object", "properties": {"q": {"type": "string"}, "k": {"type": "integer"}}}}},
    {"type": "function", "namespace": "web", "function": {"name": "search", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "namespace": {"name": "fs", "description": "Files"}, "function": {"name": "read", "description": "Read a file", "parameters": {}}},
]
def args():
    d = {"q": txt(), "k": R.randint(0, 9), "f": R.choice([1.5, True, None, [1, "a"], {"z": 1, "a": 2}])}
    return R.choice([d, json.dumps(d, ensure_ascii=False), json.dumps(json.dumps(d)), "not json"])
def call(i):
    name = R.choice(["lookup", "web::search", "fs::read"])
    return {"id": f"c{i}", "type": "function", "function": {"name": name, "arguments": args()}}

def conv():
    msgs = []
    if R.random() < 0.6: msgs.append({"role": "system", "content": txt()})
    if R.random() < 0.2: msgs.append({"role": "latest_reminder", "content": txt()})
    tools = R.random() < 0.4
    for t in range(R.randint(1, 4)):
        r = R.random()
        if r < 0.15 and msgs: msgs.append({"role": "system", "content": txt()})
        else:
            m = {"role": "user", "content": txt()}
            if R.random() < 0.15: m["content"] = [{"type": "text", "text": txt()}, {"type": "text", "text": txt()}]
            msgs.append(m)
            if R.random() < 0.1: msgs.append({"role": "user", "content": txt()})
        if t == 0 and R.random() < 0.1: msgs[-1]["task"] = R.choice(["query", "action", "title"])
        a_ = {"role": "assistant", "content": txt()}
        if R.random() < 0.6: a_["reasoning_content"] = txt()
        if tools and R.random() < 0.6:
            n = R.randint(1, 3); a_["tool_calls"] = [call(i) for i in range(n)]
            msgs.append(a_)
            ids = [f"c{i}" for i in range(n)]; R.shuffle(ids)
            for i in ids: msgs.append({"role": "tool", "tool_call_id": i, "content": R.choice([txt(), [{"type": "text", "text": txt()}]])})
            continue
        if t < 3: msgs.append(a_)
    if tools: msgs[0]["tools"] = R.sample(TOOLS, R.randint(1, 3))
    if R.random() < 0.1: msgs[0]["response_format"] = {"type": "json_object", "b": 1}
    return msgs

cases = []
while len(cases) < a.n:
    msgs = conv()
    mode = R.choice(["chat", "thinking"])
    eff = R.choice([None, None, "low", "high", "max", 1, 37, 100])
    ctx = None
    if R.random() < 0.2 and len(msgs) > 2:
        k = R.randint(1, len(msgs) - 1); ctx, msgs = msgs[:k], msgs[k:]
    drop = R.random() < 0.8
    try:
        out = enc.encode_messages(copy.deepcopy(msgs), thinking_mode=mode, context=copy.deepcopy(ctx),
                                  drop_thinking=drop, reasoning_effort=eff)
    except Exception as e:  # the reference rejects some random shapes; keep only valid ones
        continue
    cases.append({"messages": msgs, "context": ctx, "thinking_mode": mode, "reasoning_effort": eff,
                  "drop_thinking": drop, "expected": out})
os.makedirs(a.out, exist_ok=True)
json.dump(cases, open(os.path.join(a.out, "encoding_v41_fuzz.json"), "w"), ensure_ascii=False)
print(f"{len(cases)} cases -> {a.out}/encoding_v41_fuzz.json")
