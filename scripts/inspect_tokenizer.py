"""Inspect tokenizer IDs and round-trip behavior for sample text."""

from __future__ import annotations

import argparse

from configs.config import load_config
from data.tokenizer import load_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a trained EdgeGPT tokenizer.")
    parser.add_argument("text", nargs="?", default="Hello, EdgeGPT!", help="Text to encode and decode.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--tokenizer", default=None, help="Artifact directory or tokenizer.json path.")
    args = parser.parse_args()

    config = load_config(args.config)
    tokenizer = load_tokenizer(config, args.tokenizer)
    ids = tokenizer.encode(args.text, add_bos=True, add_eos=True)
    tokens = [tokenizer.id_to_token(int(token_id)) for token_id in ids.tolist()]

    print("Text:")
    print(args.text)
    print("\nIDs:")
    print(ids.tolist())
    print("\nTokens:")
    print(tokens)
    print("\nDecoded without special tokens:")
    print(tokenizer.decode(ids, skip_special_tokens=True))


if __name__ == "__main__":
    main()
