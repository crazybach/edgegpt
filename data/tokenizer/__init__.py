"""Public tokenizer entrypoints for EdgeGPT."""

from __future__ import annotations

from pathlib import Path

from configs.config import Config, TokenizerConfig
from data.tokenizer.base import TokenizerBackend
from data.tokenizer.registry import build_tokenizer


def load_tokenizer(config: Config | TokenizerConfig, path: str | Path | None = None) -> TokenizerBackend:
    """Build and load the configured tokenizer backend.

    Callers pass the full project config in normal use. Tests may pass a
    `TokenizerConfig` directly when isolating the tokenizer module.
    """

    tokenizer_config = config.tokenizer if isinstance(config, Config) else config
    tokenizer = build_tokenizer(tokenizer_config)
    tokenizer.load(Path(path or tokenizer_config.artifact_dir))
    return tokenizer


__all__ = ["TokenizerBackend", "build_tokenizer", "load_tokenizer"]
