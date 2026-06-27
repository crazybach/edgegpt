"""PyTorch dataset and loader builders for token shards."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from configs.config import Config
from data.pipeline.base import BatchProvider, TokenBlockDataset
from data.pipeline.shards import TOKEN_DTYPE, load_metadata, split_path


class MemmapTokenBlockDataset(TokenBlockDataset):
    """Read fixed-length causal-LM examples from a flat token shard."""

    def __init__(self, shard_path: str | Path, block_size: int):
        self.shard_path = Path(shard_path)
        self.block_size = block_size
        if not self.shard_path.exists():
            raise FileNotFoundError(f"Token shard not found: {self.shard_path}")

        # Memmap lets many batches slice the same file without loading the
        # entire corpus into RAM, which is the right default for laptop scale.
        self.tokens = np.memmap(self.shard_path, dtype=TOKEN_DTYPE, mode="r")
        if self.tokens.size < block_size + 1:
            raise ValueError(
                f"{self.shard_path} has {self.tokens.size} tokens, but at least {block_size + 1} "
                "are required for one shifted training block."
            )

    def __len__(self) -> int:
        # Each item samples T + 1 adjacent tokens. The last valid start is
        # `size - (T + 1)`, so the count is `size - T`.
        return int(self.tokens.size - self.block_size)

    def __getitem__(self, index: int) -> dict[str, torch.LongTensor]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        # Causal LM training needs the next token as the target at every
        # position, so we read one extra token and shift by one.
        block = np.asarray(self.tokens[index : index + self.block_size + 1], dtype=np.int64)
        ids = torch.from_numpy(block)
        return {"input_ids": ids[:-1].long(), "targets": ids[1:].long()}


class MemmapBatchProvider(BatchProvider):
    """Build deterministic DataLoaders for prepared memmap shards."""

    def __init__(self, config: Config, split: str):
        self.config = config
        self.split = split

    def loader(self) -> torch.utils.data.DataLoader:
        metadata = load_metadata(self.config.data.cache_dir)
        block_size = int(metadata["block_size"])
        dataset = MemmapTokenBlockDataset(split_path(self.config.data.cache_dir, self.split), block_size)
        generator = torch.Generator()
        generator.manual_seed(self.config.data.seed)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.training.batch_size,
            shuffle=self.split == "train",
            num_workers=self.config.data.num_workers,
            generator=generator,
        )
