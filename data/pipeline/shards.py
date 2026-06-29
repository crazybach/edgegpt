"""Memmap-backed token shard storage for Phase 2.

The first implementation follows the nanoGPT-style flat binary shard approach:
pre-tokenize once, write compact integer arrays, and let training read cheap
contiguous slices. That tradeoff is especially good on a modest CPU machine.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from configs.config import Config
from data.pipeline.base import TokenShardWriter


TOKEN_DTYPE = np.uint16


def split_path(cache_dir: str | Path, split: str) -> Path:
    """Return the canonical shard path for one split."""

    return Path(cache_dir) / f"{split}.bin"


def metadata_path(cache_dir: str | Path) -> Path:
    """Return the metadata sidecar path."""

    return Path(cache_dir) / "metadata.json"


class MemmapTokenShardWriter(TokenShardWriter):
    """Write split token streams as compact `uint16` binary shards."""

    def __init__(self, config: Config, eos_id: int, block_size: int):
        self.config = config
        self.eos_id = eos_id
        self.block_size = block_size

    def write(self, split_tokens: dict[str, list[int]]) -> dict[str, Any]:
        cache_dir = Path(self.config.data.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        if self.config.model.vocab_size > np.iinfo(TOKEN_DTYPE).max + 1:
            raise ValueError("uint16 shards only support vocab_size <= 65536.")

        token_counts: dict[str, int] = {}
        for split, tokens in split_tokens.items():
            # `uint16` keeps shards half the size of int32 while still covering
            # the current 16k vocab. Future larger vocabs can swap this writer.
            array = np.asarray(tokens, dtype=TOKEN_DTYPE)
            array.tofile(split_path(cache_dir, split))
            token_counts[split] = int(array.size)

        metadata: dict[str, Any] = {
            "dataset": self.config.data.dataset,
            "source_type": self.config.data.source_type,
            "storage_type": self.config.data.storage_type,
            "dtype": "uint16",
            "vocab_size": self.config.model.vocab_size,
            "tokenizer_artifact_dir": self.config.tokenizer.artifact_dir,
            "eos_id": self.eos_id,
            "block_size": self.block_size,
            "seed": self.config.data.seed,
            "val_split": self.config.data.val_split,
            "token_counts": token_counts,
        }
        metadata_path(cache_dir).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return metadata


class StreamingTokenShardWriter:
    """Append tokenized documents directly to split shards.

    This writer keeps memory bounded by one encoded document at a time. It is
    the production-lite path used by `prepare_data`: tokenize a document, append
    EOS, write `uint16` bytes immediately, and keep only counters in memory.
    """

    def __init__(self, config: Config, eos_id: int, block_size: int):
        self.config = config
        self.eos_id = eos_id
        self.block_size = block_size
        self.cache_dir = Path(config.data.cache_dir)
        self.token_counts: dict[str, int] = {"train": 0, "val": 0}
        self.document_counts: dict[str, int] = {"train": 0, "val": 0}

        if self.config.model.vocab_size > np.iinfo(TOKEN_DTYPE).max + 1:
            raise ValueError("uint16 shards only support vocab_size <= 65536.")

    def __enter__(self) -> "StreamingTokenShardWriter":
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._handles = {
            split: split_path(self.cache_dir, split).open("wb")
            for split in ("train", "val")
        }
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        for handle in self._handles.values():
            handle.close()

    def append(self, split: str, token_ids: list[int]) -> None:
        """Write one tokenized document plus EOS to its split shard."""

        if split not in self._handles:
            raise ValueError(f"Unsupported split: {split}")
        if token_ids:
            max_id = max(token_ids)
            min_id = min(token_ids)
            if max_id >= self.config.model.vocab_size or min_id < 0:
                raise ValueError(f"{split} split contains token IDs outside the configured vocab.")

        # EOS boundaries prevent the packed stream from inventing transitions
        # between unrelated source documents.
        array = np.asarray([*token_ids, self.eos_id], dtype=TOKEN_DTYPE)
        array.tofile(self._handles[split])
        self.token_counts[split] += int(array.size)
        self.document_counts[split] += 1

    def metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "dataset": self.config.data.dataset,
            "source_type": self.config.data.source_type,
            "storage_type": self.config.data.storage_type,
            "dtype": "uint16",
            "vocab_size": self.config.model.vocab_size,
            "tokenizer_artifact_dir": self.config.tokenizer.artifact_dir,
            "eos_id": self.eos_id,
            "block_size": self.block_size,
            "seed": self.config.data.seed,
            "val_split": self.config.data.val_split,
            "token_counts": dict(self.token_counts),
            "document_counts": dict(self.document_counts),
        }
        metadata_path(self.cache_dir).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return metadata


def load_metadata(cache_dir: str | Path) -> dict[str, Any]:
    """Load shard metadata and fail clearly if preparation has not run."""

    path = metadata_path(cache_dir)
    if not path.exists():
        raise FileNotFoundError(f"Data metadata not found: {path}. Run scripts/prepare_data.py first.")
    return json.loads(path.read_text(encoding="utf-8"))
