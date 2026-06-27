"""Public Phase 2 data pipeline API."""

from __future__ import annotations

from configs.config import Config
from data.pipeline.dataset import MemmapBatchProvider
from data.pipeline.prepare import prepare_data


def build_train_loader(config: Config, split: str = "train"):
    """Build a DataLoader for a prepared split."""

    if split not in {"train", "val"}:
        raise ValueError("split must be 'train' or 'val'.")
    if config.data.storage_type != "memmap_bin":
        raise ValueError(f"Unsupported data.storage_type: {config.data.storage_type}")
    return MemmapBatchProvider(config, split).loader()


__all__ = ["build_train_loader", "prepare_data"]
