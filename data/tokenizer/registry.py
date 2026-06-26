"""Tokenizer backend registry."""

from __future__ import annotations

from configs.config import TokenizerConfig
from data.tokenizer.base import TokenizerBackend
from data.tokenizer.byte_bpe import ByteBPETokenizerBackend


def build_tokenizer(config: TokenizerConfig) -> TokenizerBackend:
    """Create a tokenizer backend from config.

    This is the only switch statement the rest of the codebase should need.
    Future algorithms can register here while keeping the public tokenizer API
    unchanged for the data pipeline and model code.
    """

    if config.type == "byte_bpe":
        return ByteBPETokenizerBackend(config)
    raise ValueError(f"Unsupported tokenizer type: {config.type}")
