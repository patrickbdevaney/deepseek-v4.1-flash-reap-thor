#!/usr/bin/env python3
"""gen_tokenizer_vectors.py — reference (text -> ids) pairs from HF, for tests/gate_tokenizer.cpp.

The C++ tokenizer is the one that will actually serve, so it has to agree with HF everywhere the
model was trained, not just on "Hello, world!". The corpus below is chosen to hit the places a
ByteLevel-BPE port usually breaks:

  * the pre-tokenizer's digit grouping (\\p{N}{1,3}) at 1/2/3/4/5+ digit runs
  * CJK / kana isolation, and CJK adjacent to latin with no space
  * the \\s+(?!\\S) alternative: runs of spaces before a word vs at end-of-string
  * \\s*[\\r\\n]+ : CRLF, blank lines, trailing newlines, indentation
  * combining marks (\\p{M}) after letters -- alternative B's second half
  * symbol/punctuation runs (\\p{P}\\p{S}) including emoji and math
  * the ASCII-punct-then-letters alternative A, which fires in code (`.foo`, `->bar`)
  * added tokens embedded mid-text, adjacent, and at both ends
  * bytes that are not valid UTF-8 on their own -- the ByteLevel alphabet's whole purpose
  * real prompts: the repo's own suite, the chat encoder's golden outputs, GSM8K

Writes tests/data/tokenizer_vectors.json = [{"t": <text>, "ids": [...]}, ...].
"""
import glob, json, os, sys

CK = '/home/patrickd/models/DeepSeek-V4.1-Flash-1017'
OUT = 'tests/data/tokenizer_vectors.json'

CASES = [
    "", " ", "  ", "   ", "\n", "\n\n", "\r\n", " \n ", "\t", "\t\t x",
    "a", "The capital of France is", "Hello, world!", "hello world  ",
    "trailing spaces   ", "   leading spaces", "a  b   c    d",
    "1", "12", "123", "1234", "12345", "1234567890", "007", "3.14159", "1,000,000",
    "v1.2.3-rc4", "0x1F", "2026-08-12T00:00:00Z", "99 bottles, 100 bottles",
    "def fib(n):\n    if n < 2: return n\n    return fib(n-1)+fib(n-2)\n",
    "x->y", "obj.attr", "a::b", "foo(bar, baz)", "#include <stdio.h>",
    "/* comment */ int x = 5; // end", "printf(\"%d\\n\", x);",
    "SELECT * FROM t WHERE a>=1 AND b<>2;", "{\"k\": [1,2,3], \"b\": true}",
    "你好世界", "你好, world", "world你好", "こんにちは", "カタカナ", "漢字とかな混じり",
    "中文123英文", "北京市朝阳区", "🙂", "a🙂b", "🇺🇸🇯🇵", "👨‍👩‍👧‍👦",
    "naïve café", "résumé", "Ωμέγα", "Привет мир", "مرحبا بالعالم", "שלום עולם",
    "é", "à́̂", "kãna",
    " nbsp", "a b", " line sep", "　ideographic space",
    "!!!", "???", "...", "---", "===", "+++", "***", "~~~", "<<>>", "@#$%^&*",
    "a...b", "e.g.", "i.e., that", "(a)(b)", "[1][2]", "{x}{y}",
    "€100", "£5.50", "$1,234.56", "50%", "a±b", "x≤y", "∀x∈S", "α+β=γ",
    "<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>",
    "<｜begin▁of▁sentence｜>hi", "hi<｜end▁of▁sentence｜>",
    "<｜User｜>hello<｜Assistant｜>", "<think>reasoning</think>answer",
    "<｜User｜><｜Assistant｜>", "a<｜User｜>b<｜Assistant｜>c",
    "｜DSML｜tool_calls", "<｜DSML｜invoke name=\"f\">",
    "\x00\x01\x02", "\x7f", "raw \xc3\x28 bytes",
    "MixedCASE_snake_case-kebab-case.dotted",
    "  \n  \n  ", "line1\n\nline2\n\n\nline3", "trailing\n", "\n\nleading",
    "a" * 200, "ab " * 50, "1 " * 40,
]


def main():
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(CK, 'tokenizer.json'))

    cases = list(CASES)

    # Real prompts the server will actually see.
    p = 'protocol/suite_prompts.txt'
    if os.path.exists(p):
        cases += [l.rstrip('\n') for l in open(p, encoding='utf-8') if l.strip()]
    for f in sorted(glob.glob(os.path.join(CK, 'encoding/tests/test_output_*.txt'))):
        cases.append(open(f, encoding='utf-8').read())
    # Whole source files: the densest source of odd byte sequences we have on disk.
    for f in ['include/encoding_dsv4.h', 'CHAT_FORMAT.md', 'README.md']:
        if os.path.exists(f):
            cases.append(open(f, encoding='utf-8').read())
    # GSM8K, which is what capability was measured on.
    try:
        import pyarrow as pa
        base = os.path.expanduser('~/.cache/huggingface/datasets/openai___gsm8k')
        n = 0
        for af in sorted(glob.glob(os.path.join(base, '**', '*.arrow'), recursive=True)):
            with pa.memory_map(af, 'rb') as src:
                for b in pa.ipc.open_stream(src):
                    for r in b.to_pylist():
                        if r.get('question'):
                            cases.append(r['question']); n += 1
                        if r.get('answer'):
                            cases.append(r['answer']); n += 1
                    if n > 400: break
            if n > 400: break
    except Exception as e:
        print(f'[vec] gsm8k skipped: {e}', file=sys.stderr)

    out = []
    for t in cases:
        out.append({'t': t, 'ids': tok.encode(t, add_special_tokens=False).ids})
    os.makedirs('tests/data', exist_ok=True)
    json.dump(out, open(OUT, 'w'), ensure_ascii=False)
    ntok = sum(len(o['ids']) for o in out)
    print(f'[vec] wrote {len(out)} cases, {ntok} reference tokens -> {OUT}')


if __name__ == '__main__':
    main()
