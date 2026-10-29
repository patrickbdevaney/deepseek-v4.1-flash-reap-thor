#!/usr/bin/env python3
"""ab_ids.py REF.json NEW.json -> ids identical? (first diff index); the exactness check for T0 changes."""
import json, sys
a, b = (json.load(open(f)) for f in sys.argv[1:3])
d = next((i for i, (x, y) in enumerate(zip(a["ids"], b["ids"])) if x != y), None)
same = a["ids"] == b["ids"] and a["prompt_ids"] == b["prompt_ids"]
print(f"ids {'IDENTICAL' if same else 'DIFFER'} (first diff {d}, len {len(a['ids'])} vs {len(b['ids'])}); "
      f"tok/s {a['decode_tok_s']:.2f} -> {b['decode_tok_s']:.2f}")
sys.exit(0 if same else 1)
