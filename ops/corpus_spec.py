"""Calibration corpus for the V4.1-Flash-1017 saliency pass (s02 builds it, s03 consumes it).

Mixture (PLAN.md stage 2), in ROUTED-TOKEN space, not sample space: GLM pass-2 showed a sample
quota delivered code at 9.3% of tokens against 21% specified, because sources differ 4x in
length. Here every row is a full 8,192-token sequence packed from ONE bucket, so the token
share of a bucket is exactly its share of rows and needs no length-ratio correction.

Provenance rule (user, standing): no Claude-written text, ever. Each source carries the generator
of its assistant/answer text as its card states it. "human" means the text was written by people
(web, papers, issues, patches, exam questions). A source whose generator is unknown, mixed with
undisclosed models, or "unverified" (named here from memory and not yet checked against the card)
is excluded unless `--allow-unverified` is passed after a human has checked the card and updated
the entry. Known Claude-generated sets are listed in EXCLUDED so the exclusion stays visible.

Licence: the derivative must stay MIT-compatible (GLM policy, 2026-08-27). Permissive or
attribution licences only; NC and share-alike sources are excluded. `licence` is the card's
declared licence as we last knew it; s02 re-reads `cardData.license` from the hub at runtime and
records both, and refuses a source whose live licence is NC or share-alike.

Text shape: `fn(row)` returns either a str (raw document: BOS + text) or a list of OpenAI-format
messages, rendered by the checkpoint's own encoding/encoding.py so chat, reasoning and DSML
tool-call structure reach the router exactly as V4.1 sees them in serving.
"""
from __future__ import annotations

import json
import re

SEQ_LEN = 8192
TOTAL_TOKENS = 5_000_000           # >=500k routed tokens per pool (dose-response note; GLM pass-1 under-calibration
                                   # was its main defect). Hidden states page to NVMe (~40 KB/token, ~205 GB): Engram
                                   # shards 47/48 read from the Seagate to make room (decision 2026-10-19)
HELDOUT_ROWS_FRACTION = 0.10       # frozen held-out split for s04 arbitration (top-32 KL), per bucket

# PLAN.md gave general 20 / agentic 15 / code 15 / vision 14 / math 12 / science 12 / finance 10
# (which sums to 98%). The user then added (2026-10-19): a long-context pool (~10% of tokens in
# 32k-128k rows), a multilingual pool (~8-10%, Chinese first), a long-tail world-knowledge slice
# inside general, and every pool >= 10% where possible with vision held at ~14%. Nine pools at
# >= 10% leave exactly 10 points to place; they go to vision (+4) and to general, agentic and
# code (+2 each), the PLAN's three largest pools. FINAL MIX (token shares):
TOKEN_TARGET = {
    "general": 0.12,        # web prose + long-tail world knowledge (encyclopedic + knowledge QA)
    "multilingual": 0.10,   # Chinese first (half the pool), then ru/ja/ko/de/fr/es/ar/pt/...
    "agentic": 0.12,
    "code": 0.12,
    "vision": 0.14,         # real images through the reference image_processor; bias_vl routing
    "math": 0.10,
    "science": 0.10,
    "finance": 0.10,
    "longctx": 0.10,        # 32k-128k rows: whole books, whole code repos, long agentic traces
}
assert abs(sum(TOKEN_TARGET.values()) - 1.0) < 1e-9
assert min(TOKEN_TARGET.values()) >= 0.10, "every pool at least 10%"
LONG_MIN, LONG_MAX = 32_768, 65_536    # long-context row lengths (tokens); 64k cap (was 128k, 2026-10-19): the
                                        # 10% pool held only 6 docs at 128k, so diversity beats length
# Frozen order: bucket ids are written into every s02 row and every s03 accumulator.
BUCKETS = list(TOKEN_TARGET)


def _first(row, *keys):
    for k in keys:
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return None


def _messages(row, *keys):
    """OpenAI {role, content} or ShareGPT {from, value}; also columns holding a JSON string."""
    for key in keys or ("messages", "conversations", "conversation"):
        msgs = row.get(key)
        if isinstance(msgs, str):
            try:
                msgs = json.loads(msgs)
            except Exception:
                continue
        if not isinstance(msgs, list):
            continue
        out = []
        role_map = {"human": "user", "gpt": "assistant", "user": "user", "assistant": "assistant",
                    "system": "system", "tool": "tool", "function": "tool"}
        for m in msgs:
            if not isinstance(m, dict):
                continue
            role = role_map.get(m.get("role", m.get("from")), None)
            c = m.get("content", m.get("value"))
            if isinstance(c, list):
                c = "\n".join(str(x.get("text", "")) for x in c if isinstance(x, dict))
            if role and isinstance(c, str) and c.strip():
                out.append({"role": role, "content": c})
        if len(out) >= 2:
            return out
    return None


def _qa(q_keys, a_keys, reasoning_keys=()):
    """A user question and an assistant answer; a reasoning trace goes to reasoning_content
    so the thinking-mode template wraps it in <think> as V4.1 would."""
    def fn(row):
        q, a = _first(row, *q_keys), _first(row, *a_keys)
        if not q or not a:
            return None
        msg = {"role": "assistant", "content": a}
        r = _first(row, *reasoning_keys) if reasoning_keys else None
        if r:
            msg["reasoning_content"] = r
        return [{"role": "user", "content": q}, msg]
    return fn


def _text(*keys):
    return lambda row: _first(row, *keys)


def _trivia(row):
    """TriviaQA (human trivia questions) with one evidence passage when present."""
    q, a = row.get("question"), (row.get("answer") or {}).get("value")
    if not q or not a:
        return None
    ctx = ""
    for key in ("search_results", "entity_pages"):
        v = row.get(key) or {}
        texts = v.get("search_context") or v.get("wiki_context") or []
        if texts:
            ctx = texts[0][:6000] + "\n\n"
            break
    return [{"role": "user", "content": ctx + q}, {"role": "assistant", "content": a}]


def _mmlu_aux(row):
    q, ch, a = row.get("question"), row.get("choices"), row.get("answer")
    if isinstance(row.get("train"), dict):           # auxiliary_train nests the record
        r = row["train"]; q, ch, a = r.get("question"), r.get("choices"), r.get("answer")
    if not q or not ch or a is None:
        return None
    opts = "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(ch))
    return [{"role": "user", "content": f"{q}\n{opts}"}, {"role": "assistant", "content": f"{chr(65 + int(a))}. {ch[int(a)]}"}]


def _obqa(row):
    q, ch, key, fact = row.get("question_stem"), row.get("choices") or {}, row.get("answerKey"), row.get("fact1")
    if not q or not key or "text" not in ch:
        return None
    opts = "\n".join(f"{l}. {t}" for l, t in zip(ch["label"], ch["text"]))
    ans = dict(zip(ch["label"], ch["text"])).get(key, "")
    return [{"role": "user", "content": f"{q}\n{opts}"},
            {"role": "assistant", "content": f"{key}. {ans}" + (f"\n\n{fact}" if fact else "")}]


def _xlam(row):
    """APIGen-style function calling -> tools + assistant tool_calls, rendered as DSML."""
    try:
        tools = json.loads(row["tools"]) if isinstance(row.get("tools"), str) else row.get("tools")
        calls = json.loads(row["answers"]) if isinstance(row.get("answers"), str) else row.get("answers")
    except Exception:
        return None
    if not row.get("query") or not tools or not calls:
        return None
    oa_tools = [{"type": "function", "function": t} for t in tools]
    tcs = [{"type": "function", "function": {"name": c["name"], "arguments": json.dumps(c.get("arguments", {}))}}
           for c in calls]
    return [{"role": "user", "content": row["query"], "tools": oa_tools},
            {"role": "assistant", "content": "", "tool_calls": tcs}]


def S(hf_id, config, split, weight, fn, *, licence, generator, status="reused-glm", thinking=False):
    return dict(hf_id=hf_id, config=config, split=split, weight=weight, fn=fn, licence=licence,
                generator=generator, status=status, thinking=thinking)


# status: "reused-glm" = loaded with this hf_id/config/split by ~/glm-5.3-reap s02 (2026-09),
#         "new"        = not yet loaded by any of our pipelines; s02 --probe must show yield.
SOURCES: dict[str, list[dict]] = {
    "general": [
        S("HuggingFaceFW/fineweb-edu", "sample-10BT", "train", 0.43, _text("text"), licence="odc-by", generator="human"),
        # encyclopedic breadth (CC-BY-SA allowed for calibration statistics, decision 2026-10-19)
        S("wikimedia/wikipedia", "20231101.en", "train", 0.12, _text("text"), licence="cc-by-sa-3.0", generator="human", status="new"),
        # long-tail world knowledge: rare-fact experts fire rarely, so saliency undercounts them
        # unless the corpus asks many different factual questions across many topics.
        S("HuggingFaceFW/fineweb-edu", "sample-10BT", "train", 0.15, _text("text"), licence="odc-by", generator="human",
          status="filtered: s02 keeps rows whose url matches KNOWLEDGE_URL (reference/encyclopedic sites)"),
        S("mandarjoshi/trivia_qa", "rc", "train", 0.12, _trivia, licence="apache-2.0", generator="human", status="new"),
        S("cais/mmlu", "auxiliary_train", "train", 0.08, _mmlu_aux, licence="mit", generator="human", status="new"),
        S("allenai/openbookqa", "additional", "train", 0.05, _obqa, licence="apache-2.0", generator="human", status="new"),
        S("HuggingFaceFW/fineweb", "sample-10BT", "train", 0.05, _text("text"), licence="odc-by", generator="human", status="new"),
    ],
    "multilingual": [
        S("HuggingFaceFW/fineweb-2", "cmn_Hani", "train", 0.50, _text("text"), licence="odc-by", generator="human", status="new"),
        S("HuggingFaceFW/fineweb-2", "rus_Cyrl", "train", 0.08, _text("text"), licence="odc-by", generator="human", status="new"),
        S("HuggingFaceFW/fineweb-2", "jpn_Jpan", "train", 0.07, _text("text"), licence="odc-by", generator="human", status="new"),
        S("HuggingFaceFW/fineweb-2", "kor_Hang", "train", 0.06, _text("text"), licence="odc-by", generator="human", status="new"),
        S("HuggingFaceFW/fineweb-2", "deu_Latn", "train", 0.05, _text("text"), licence="odc-by", generator="human", status="new"),
        S("HuggingFaceFW/fineweb-2", "fra_Latn", "train", 0.05, _text("text"), licence="odc-by", generator="human", status="new"),
        S("HuggingFaceFW/fineweb-2", "spa_Latn", "train", 0.05, _text("text"), licence="odc-by", generator="human", status="new"),
        S("HuggingFaceFW/fineweb-2", "arb_Arab", "train", 0.04, _text("text"), licence="odc-by", generator="human", status="new"),
        S("HuggingFaceFW/fineweb-2", "por_Latn", "train", 0.04, _text("text"), licence="odc-by", generator="human", status="new"),
        S("HuggingFaceFW/fineweb-2", "vie_Latn", "train", 0.03, _text("text"), licence="odc-by", generator="human", status="new"),
        S("HuggingFaceFW/fineweb-2", "ind_Latn", "train", 0.03, _text("text"), licence="odc-by", generator="human", status="new"),
    ],
    "agentic": [
        S("nebius/SWE-rebench", None, "test", 0.30, _qa(["problem_statement"], ["patch"]), licence="cc-by-4.0", generator="human"),
        S("SWE-Gym/SWE-Gym", None, "train", 0.20, _qa(["problem_statement"], ["patch"]), licence="mit", generator="human"),
        S("SWE-bench/SWE-bench", None, "dev", 0.15, _qa(["problem_statement"], ["patch"]), licence="mit", generator="human", status="new"),
        S("Salesforce/xlam-function-calling-60k", None, "train", 0.20, _xlam, licence="cc-by-4.0",
          generator="unverified", status="new"),
        S("nvidia/Nemotron-Post-Training-Dataset-v1", None, "tool_calling", 0.15, _messages,   # card read 2026-10-19:
          licence="cc-by-4.0", generator="DeepSeek-R1-0528 / Qwen3-235B-A22B", status="new"),  # responses from these two only
    ],
    "code": [
        S("nvidia/OpenCodeReasoning-2", "train", "python", 0.30, _qa(["question", "input"], ["solution"], ["r1_generation"]),
          licence="cc-by-4.0", generator="DeepSeek-R1-0528 / QwQ-32B", thinking=True),
        S("nvidia/OpenCodeReasoning-2", "train", "cpp", 0.15, _qa(["question", "input"], ["solution"], ["r1_generation"]),
          licence="cc-by-4.0", generator="DeepSeek-R1-0528 / QwQ-32B", thinking=True),
        S("nvidia/OpenCodeInstruct", None, "train", 0.25, _qa(["input", "instruction"], ["output"]),
          licence="cc-by-4.0", generator="Llama-3.3-70B-Instruct"),
        S("GPUMODE/KernelBook", None, "train", 0.20, _text("python_code", "triton_code"), licence="apache-2.0", generator="human + torch inductor"),
        S("SakanaAI/AI-CUDA-Engineer-Archive", None, "level_1", 0.10, _text("CUDA_Code"), licence="cc-by-4.0",
          generator="Sakana AI CUDA Engineer (agent; LLMs not named on the card)"),   # user 2026-10-19: keep
    ],
    "math": [
        S("nvidia/OpenMathReasoning", "default", "cot", 0.35, _qa(["problem"], ["generated_solution"]),
          licence="cc-by-4.0", generator="DeepSeek-R1 / QwQ-32B", thinking=True),
        S("zwhe99/DeepMath-103K", None, "train", 0.25, _qa(["question"], ["final_answer"], ["r1_solution_1"]),
          licence="mit", generator="DeepSeek-R1", thinking=True),
        S("open-r1/OpenR1-Math-220k", None, "train", 0.15, _qa(["problem"], ["answer"], ["solution"]),
          licence="apache-2.0", generator="DeepSeek-R1", thinking=True),
        S("HuggingFaceTB/finemath", "finemath-3plus", "train", 0.15, _text("text"), licence="odc-by", generator="human"),
        S("internlm/Lean-Workbook", None, "train", 0.10, _qa(["natural_language_statement"], ["formal_statement"]),
          licence="apache-2.0", generator="InternLM autoformalizer"),
    ],
    "science": [
        S("open-thoughts/OpenThoughts3-1.2M", None, "train", 0.45, _messages, licence="apache-2.0",
          generator="QwQ-32B", thinking=True),
        S("TIGER-Lab/MMLU-Pro", None, "test", 0.15, _qa(["question"], ["answer"], ["cot_content"]),
          licence="mit", generator="human"),
        # ChemBench dropped 2026-10-19: rows carry a "never in training corpora" canary
        S("HuggingFaceFW/fineweb-edu", "sample-10BT", "train", 0.30, _text("text"), licence="odc-by", generator="human",
          status="filtered: s02 keeps rows whose url matches SCIENCE_URL"),
    ],
    "finance": [
        S("G4KMU/t2-ragbench", "FinQA", "train", 0.22, _qa(["pre_text", "context", "question"], ["original_answer", "program_answer"]),
          licence="cc-by-4.0", generator="human"),
        S("G4KMU/t2-ragbench", "ConvFinQA", "turn_0", 0.15, _qa(["pre_text", "context", "question"], ["original_answer", "program_answer"]),
          licence="cc-by-4.0", generator="human"),
        S("G4KMU/t2-ragbench", "TAT-DQA", "train", 0.13, _qa(["pre_text", "context", "question"], ["original_answer", "program_answer"]),
          licence="cc-by-4.0", generator="human"),
        S("kensho/bizbench", "default", "train", 0.20, _qa(["question", "context"], ["answer", "program"]),
          licence="cc-by-4.0", generator="human"),
        S("next-tat/tat-llm-instructions", None, "train", 0.15, _qa(["user_prompt"], ["resp"]),
          licence="cc-by-4.0", generator="human (TAT-QA derived)"),
        S("HuggingFaceFW/fineweb-edu", "sample-10BT", "train", 0.15, _text("text"), licence="odc-by", generator="human",
          status="filtered: s02 keeps rows whose url matches FINANCE_URL"),
    ],
}

# Teacher-SELF-generated traces (user, 2026-10-19): calibrate in V4.1's own chat/thinking/DSML
# format with text V4.1-Flash-1017 itself wrote. Produced later by s02b (README) from a running
# 1017 teacher (rented vLLM node or the equality-checked official API, PLAN stage 1 sources B/C);
# never Claude-written, never another model's text. Each file is jsonl of
# {"messages": [...], "thinking": bool, "generator": "DeepSeek-V4.1-Flash-1017", "teacher_check": {...}}.
# Until the files exist these sources yield nothing and their quota flows to the next source.
SELF_GEN = "DeepSeek-V4.1-Flash-1017 (self)"
for _b, _w in (("agentic", 0.25), ("code", 0.15), ("math", 0.15), ("science", 0.10)):
    SOURCES[_b].insert(0, S(f"local:selfgen/{_b}.jsonl", None, None, _w, _messages, licence="mit (model output)",
                            generator=SELF_GEN, status="s02b output; absent until generated", thinking=True))

# Long-context pool: whole documents of 32k-128k tokens, one per row (never concatenated across
# documents: the point is real long-range structure). Upstream git repos are read from the
# upstream ref only (`git show <ref>:<path>`), never from a working tree with local commits,
# because our own repos contain Claude-written code.
LONG_SOURCES = [
    S("local:selfgen/longctx.jsonl", None, None, 0.20, _messages, licence="mit (model output)", generator=SELF_GEN,
      status="s02b output: long multi-turn agentic traces; absent until generated", thinking=True),
    S("emozilla/pg19", None, "train", 0.40, _text("text"), licence="apache-2.0 (public-domain books)", generator="human", status="new"),
    S("git:~/llama.cpp@origin/master", None, None, 0.20, None, licence="mit", generator="human (upstream ggml-org/llama.cpp)", status="local"),
    S("git:~/cutlass@origin/main", None, None, 0.15, None, licence="bsd-3-clause", generator="human (upstream NVIDIA/cutlass)", status="local"),
    S("git:~/TensorRT-Edge-LLM@origin/main", None, None, 0.10, None, licence="apache-2.0",
      generator="human (upstream NVIDIA/TensorRT-Edge-LLM)", status="local"),
]
REPO_EXTS = (".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp", ".py", ".md", ".txt", ".cmake", ".sh", ".metal", ".comp")

# Vision rows carry REAL images through the checkpoint's own image_processor; a text description
# of an image routes like text and protects nothing. Configs are explicit (Nemotron-VLM's first
# config alphabetically is German Wikipedia text). Media sources fail silently (GLM: 9 of 15
# yielded zero images), so s02 probes image yield per source and reports the achieved mix.
# `img` / `fn` pull the image(s) and the conversation from a row.
VISION_SOURCES = [
    S("HuggingFaceM4/the_cauldron", "chartqa", "train", 0.15, None, licence="cc-by-4.0 (per subset)", generator="human"),
    S("HuggingFaceM4/the_cauldron", "ai2d", "train", 0.10, None, licence="cc-by-4.0 (per subset)", generator="human"),
    S("HuggingFaceM4/the_cauldron", "docvqa", "train", 0.15, None, licence="cc-by-4.0 (per subset)", generator="human"),
    S("HuggingFaceM4/the_cauldron", "textvqa", "train", 0.10, None, licence="cc-by-4.0 (per subset)", generator="human", status="new"),
    S("HuggingFaceM4/the_cauldron", "infographic_vqa", "train", 0.10, None, licence="cc-by-4.0 (per subset)", generator="human", status="new"),
    S("HuggingFaceM4/the_cauldron", "screen2words", "train", 0.10, None, licence="cc-by-4.0 (per subset)", generator="human", status="new"),
    S("HuggingFaceM4/Docmatix", "images", "train", 0.15, None, licence="mit", generator="Phi-3-small (Q/A over PDFA text)"),
    S("lmms-lab/LLaVA-OneVision-Data", "CLEVR-Math(MathV360K)", "train", 0.15, None, licence="apache-2.0",
      generator="templated (CLEVR-Math)"),
]

SCIENCE_URL = r"(arxiv|nature\.com|science|nih\.gov|ncbi|chem|physics|biolog|astro|geolog|\.edu/)"
KNOWLEDGE_URL = r"(wikipedia\.org|britannica\.com|encyclopedia|wiktionary|plato\.stanford\.edu|history\.|\.gov/|museum|archive\.org)"
FINANCE_URL = r"(sec\.gov|investopedia|federalreserve|imf\.org|worldbank|bis\.org|finance|bank|invest|econom|ecb\.europa)"

EXCLUDED = {
    "claude_generated": [
        "SWE-bench/SWE-smith-trajectories",          # Claude 3.7 Sonnet trajectories
        "SWE-Gym/OpenHands-Sampled-Trajectories",    # includes Claude 3.5 Sonnet rollouts
        "allenai/pixmo-docs",                         # documents rendered from Claude-written code
    ],
    "generator_unknown_or_mixed": [
        "togethercomputer/CoderForge-Preview", "open-thoughts/AgentTrove", "arcee-ai/agent-data",
        "allenai/tulu-3-sft-mixture", "nvidia/sft_datablend_v1", "nvidia/Nemotron-VLM-Dataset-v2",
        "xlangai/aguvis-stage2", "ServiceNow/BigDocs-Bench",
    ],
    "licence_nc": ["EricLu/SCP-116K", "camel-ai/physics", "camel-ai/chemistry", "camel-ai/biology",
                   "osunlp/UGround-V1-Data", "allenai/sciq"],
    # share-alike (Wikipedia, Natural Questions, tattabio/OG) is ALLOWED (decision 2026-10-19): calibration
    # yields saliency statistics, not a derivative text, and nothing from these sources is redistributed.
    "gated_401": ["nvidia/Nemotron-CC-Math", "TheFinAI/MultiFinBen", "mlfoundations/MINT-1T"],
}

def licence_ok(lic: str | None) -> bool | None:
    """False for an explicit NC / GPL licence, True for a named permissive or share-alike one (SA allowed for
    calibration statistics, decision 2026-10-19),
    None when the card says nothing usable ("other", missing): recorded as needs-review."""
    if not lic or lic.strip().lower() in ("other", "unknown"):
        return None
    low = lic.lower()
    toks = set(re.split(r"[^a-z0-9.]+", low))
    flat = low.replace("-", "").replace("_", "")
    if toks & {"nc", "gpl", "agpl", "lgpl"} or "noncommercial" in flat:
        return False
    return True


GENERATOR_POLICIES = ("no-claude", "human-or-self")


def usable(src: dict, allow_unverified: bool, policy: str = "no-claude") -> tuple[bool, str]:
    """no-claude (default): any disclosed non-Claude generator. human-or-self: only human text and
    V4.1-1017's own output; switch to it once s02b has produced self-generated traces."""
    g = src["generator"].lower()
    if "claude" in g or "anthropic" in g:
        return False, "claude-generated"
    if g in ("unknown", "mixed") or (g == "unverified" and not allow_unverified):
        return False, f"generator {g}"
    if policy == "human-or-self" and not (g.startswith("human") or src["generator"] == SELF_GEN):
        return False, f"policy human-or-self excludes generator {src['generator']}"
    excluded = {h for v in EXCLUDED.values() for h in v}
    if src["hf_id"] in excluded:
        return False, "listed in EXCLUDED"
    return True, "ok"


def rows_per_bucket(total_tokens: int = TOTAL_TOKENS, seq_len: int = SEQ_LEN) -> dict[str, int]:
    """Largest-remainder apportionment of whole 8k rows to buckets by TOKEN_TARGET. For
    `longctx` the value is that pool's token budget in 8k units; s02 fills it with whole
    32k-128k documents, one per row."""
    n = max(len(BUCKETS), round(total_tokens / seq_len))
    raw = {b: TOKEN_TARGET[b] * n for b in BUCKETS}
    out = {b: int(v) for b, v in raw.items()}
    for b in sorted(BUCKETS, key=lambda b: raw[b] - out[b], reverse=True)[: n - sum(out.values())]:
        out[b] += 1
    return out


def interleave(counts: dict[str, int]) -> list[str]:
    """Smooth weighted round-robin: every prefix of the visit order is as close to the target mix
    as whole rows allow (GLM's grouped order put all media at 85% of the run, which made partial
    passes and worst-domain objectives degenerate)."""
    total = sum(counts.values())
    credit = {b: 0.0 for b in counts}
    left = dict(counts)
    order = []
    for _ in range(total):
        for b in counts:
            if left[b]:
                credit[b] += counts[b] / total
        b = max((b for b in counts if left[b]), key=lambda b: (credit[b], -BUCKETS.index(b)))
        credit[b] -= 1.0
        left[b] -= 1
        order.append(b)
    return order
