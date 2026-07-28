"""PyTorch dataset and loader builders for token shards."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from configs.config import Config
from data.pipeline.base import BatchProvider, TokenBlockDataset
from data.pipeline.shards import TOKEN_DTYPE, load_metadata, split_path

class BoundedRandomSampler(torch.utils.data.Sampler[int]):
    """Sample token-block starts without allocating a full dataset permutation.

    PyTorch's RandomSampler uses randperm for shuffle-without-replacement. A
    billion-position memmap would require roughly 8 GB just for those indices.
    This sampler draws with replacement in bounded chunks and exposes its
    buffered state so checkpoints can resume deterministically.
    """

    def __init__(
        self,
        data_source: torch.utils.data.Dataset,
        *,
        generator: torch.Generator,
        chunk_size: int = 10_000,
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        self.data_source = data_source
        self.generator = generator
        self.chunk_size = int(chunk_size)
        self._remaining = len(data_source)
        self._buffer = torch.empty(0, dtype=torch.long)
        self._cursor = 0

    def __len__(self) -> int:
        return len(self.data_source)

    def __iter__(self):
        if self._remaining <= 0:
            self._remaining = len(self.data_source)
            self._buffer = torch.empty(0, dtype=torch.long)
            self._cursor = 0

        while self._remaining > 0:
            if self._cursor >= self._buffer.numel():
                count = min(self.chunk_size, self._remaining)
                self._buffer = torch.randint(
                    high=len(self.data_source),
                    size=(count,),
                    generator=self.generator,
                    dtype=torch.long,
                )
                self._cursor = 0
            index = int(self._buffer[self._cursor].item())
            self._cursor += 1
            self._remaining -= 1
            yield index

    def state_dict(self) -> dict[str, object]:
        return {
            "generator": self.generator.get_state(),
            "remaining": self._remaining,
            "buffer": self._buffer[self._cursor :].clone(),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        generator_state = state.get("generator")
        if isinstance(generator_state, torch.Tensor):
            self.generator.set_state(generator_state.cpu())
        self._remaining = int(state.get("remaining", len(self.data_source)))
        buffer = state.get("buffer")
        self._buffer = buffer.cpu().long().clone() if isinstance(buffer, torch.Tensor) else torch.empty(0, dtype=torch.long)
        self._cursor = 0


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
        loader_generator = torch.Generator()
        loader_generator.manual_seed(self.config.data.seed + 1)
        sampler = None
        if self.split == "train":
            sampler = BoundedRandomSampler(
                dataset,
                generator=generator,
                chunk_size=self.config.data.shuffle_buffer_size,
            )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=self.config.data.num_workers,
            generator=loader_generator,
        )
