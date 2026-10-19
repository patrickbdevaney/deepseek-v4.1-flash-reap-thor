"""Tokenizer via the `tokenizers` library only (no transformers anywhere in ops/).

`Tok` exposes the two things the reference code asks of a tokenizer:
  * engram.build_compressed_token_map wants `.backend_tokenizer` and `len()`;
  * image_processor.prepare_vl_inputs wants `.encode`, `.convert_tokens_to_ids`, `.unk_token_id`.
tokenizer_config.json sets add_bos_token false, and encoding.encode_messages writes the BOS
itself, so `encode` never adds special tokens.
"""
from __future__ import annotations

from pathlib import Path


class Tok:
    def __init__(self, path: Path):
        from tokenizers import Tokenizer
        self.backend_tokenizer = Tokenizer.from_file(str(Path(path) / "tokenizer.json"))
        self.unk_token_id = None

    def __len__(self) -> int:
        return self.backend_tokenizer.get_vocab_size(with_added_tokens=True)

    def encode(self, text: str) -> list[int]:
        return self.backend_tokenizer.encode(text, add_special_tokens=False).ids

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        return self.backend_tokenizer.decode(list(ids), skip_special_tokens=skip_special_tokens)

    def convert_tokens_to_ids(self, token: str):
        return self.backend_tokenizer.token_to_id(token)
