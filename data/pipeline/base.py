"""Stable interfaces for Phase 2 data pipeline implementations.

The data pipeline has several parts that are likely to change over the life of
the project: local files today, Hugging Face streaming later; simple memmap
shards today, indexed shards later. These small interfaces define the contracts
that later implementations must preserve so the training loop can stay boring.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch


@dataclass(frozen=True)
class Document:
    """One raw training document before tokenization.

    `doc_id` is deliberately opaque. A source backend can use a path, JSONL row
    number, dataset ID, or future remote shard coordinate without changing the
    tokenizer/packing stages.
    """

    doc_id: str
    text: str


class DocumentSource(ABC):
    """Yields raw documents in a deterministic order for a fixed config."""

    @abstractmethod
    def documents(self) -> Iterable[Document]:
        """Stream documents without requiring the whole corpus in memory."""


class TokenShardWriter(ABC):
    """Writes split-specific token shards and metadata.

    Alternate storage backends must keep the same semantic contract: each split
    is a flat stream of token IDs with EOS already inserted between documents.
    """

    @abstractmethod
    def write(self, split_tokens: dict[str, list[int]]) -> dict[str, object]:
        """Persist token IDs and return serializable metadata."""


class TokenBlockDataset(torch.utils.data.Dataset):
    """Dataset contract for fixed-length causal-LM blocks.

    Implementations return `T + 1` adjacent tokens internally so callers can
    produce the training pair `input_ids = block[:-1]` and
    `targets = block[1:]`.
    """

    @abstractmethod
    def __getitem__(self, index: int) -> dict[str, torch.LongTensor]:
        """Return one shifted training example."""


class BatchProvider(ABC):
    """Factory for batches consumed by the future training loop."""

    @abstractmethod
    def loader(self) -> torch.utils.data.DataLoader:
        """Build a PyTorch DataLoader for one split."""


def resolve_source_paths(data_dir: str, source_paths: list[str]) -> list[Path]:
    """Resolve configured paths while preserving explicit source order."""

    if source_paths:
        return [Path(path) for path in source_paths]

    root = Path(data_dir)
    train_txt = root / "train.txt"
    if train_txt.exists():
        return [train_txt]
    if root.exists():
        return sorted(root.glob("*.txt"))
    return []
