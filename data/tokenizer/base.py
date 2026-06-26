"""Tokenizer backend interface.

EdgeGPT code should depend on this small interface instead of a concrete BPE
class. That keeps Phase 2 data packing and later model code independent from
the tokenizer algorithm, so a future SentencePiece/Unigram backend can plug in
without rewriting callers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence

import torch


class TokenizerBackend(ABC):
    """Common API for every tokenizer implementation."""

    @abstractmethod
    def train(self, files: list[Path], output_dir: Path) -> None:
        """Fit tokenizer artifacts from text files and save them."""

    @abstractmethod
    def load(self, path: Path) -> None:
        """Load tokenizer artifacts from a directory or tokenizer JSON file."""

    @abstractmethod
    def save(self, output_dir: Path) -> None:
        """Persist tokenizer artifacts."""

    @abstractmethod
    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> torch.LongTensor:
        """Convert text to token IDs."""

    @abstractmethod
    def decode(self, ids: Sequence[int] | torch.Tensor, skip_special_tokens: bool = False) -> str:
        """Convert token IDs back to text."""

    @abstractmethod
    def encode_batch(
        self,
        texts: list[str],
        padding: bool = True,
        max_length: int | None = None,
    ) -> dict[str, torch.LongTensor]:
        """Encode a batch, optionally right-padding to a rectangular tensor."""

    @abstractmethod
    def token_to_id(self, token: str) -> int:
        """Return the integer ID for one token string."""

    @abstractmethod
    def id_to_token(self, token_id: int) -> str:
        """Return the token string for one integer ID."""

    @abstractmethod
    def vocab_size(self) -> int:
        """Return the final tokenizer vocab size."""
