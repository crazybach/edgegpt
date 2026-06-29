"""Tests for the Phase 2 data pipeline contracts.

These checks protect the training interface: every later phase expects batches
to be fixed-size causal-LM pairs where targets are inputs shifted by one token.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from configs.config import Config, DataConfig, ModelConfig, TokenizerConfig, TrainingConfig
from data.pipeline import build_train_loader, prepare_data
from data.pipeline.shards import TOKEN_DTYPE, split_path
from data.pipeline.sources import LocalTextDocumentSource
from data.tokenizer.registry import build_tokenizer


def _write_corpus(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "alpha beta gamma delta",
                "numbers 1 2 3 stay split",
                "中文 文本 也 要 可逆",
                "def add(a, b): return a + b",
            ]
        ),
        encoding="utf-8",
    )
    return path


def _build_config(tmp_path: Path, *, block_size: int = 8, val_split: float = 0.0) -> Config:
    tokenizer_config = TokenizerConfig(
        vocab_size=512,
        reserved_special_tokens=64,
        artifact_dir=str(tmp_path / "tokenizer"),
        train_files=[str(_write_corpus(tmp_path / "tokenizer_corpus.txt"))],
        min_frequency=1,
    )
    tokenizer = build_tokenizer(tokenizer_config)
    tokenizer.train([Path(tokenizer_config.train_files[0])], Path(tokenizer_config.artifact_dir))

    source_a = tmp_path / "doc_a.txt"
    source_b = tmp_path / "doc_b.txt"
    source_a.write_text("alpha beta gamma delta epsilon zeta eta theta iota kappa", encoding="utf-8")
    source_b.write_text("中文 文本 也 可以 进入 数据 管线 测试", encoding="utf-8")

    return Config(
        model=ModelConfig(vocab_size=512, max_seq_len=block_size),
        training=TrainingConfig(batch_size=2),
        data=DataConfig(
            dataset="tiny-test",
            data_dir=str(tmp_path),
            val_split=val_split,
            seed=123,
            source_type="local_text",
            source_paths=[str(source_a), str(source_b)],
            cache_dir=str(tmp_path / "cache"),
            storage_type="memmap_bin",
            block_size=block_size,
            num_workers=0,
        ),
        tokenizer=tokenizer_config,
        device="cpu",
    )


def test_prepare_data_writes_shards_and_metadata(tmp_path: Path):
    config = _build_config(tmp_path, block_size=8)
    metadata = prepare_data(config)

    train_path = split_path(config.data.cache_dir, "train")
    val_path = split_path(config.data.cache_dir, "val")
    assert train_path.exists()
    assert val_path.exists()
    assert metadata["dtype"] == "uint16"
    assert metadata["block_size"] == 8
    assert metadata["document_counts"]["train"] == 2

    train_tokens = np.fromfile(train_path, dtype=TOKEN_DTYPE)
    assert train_tokens.size == metadata["token_counts"]["train"]
    assert int(train_tokens.min()) >= 0
    assert int(train_tokens.max()) < config.model.vocab_size


def test_local_text_source_streams_one_document_per_non_empty_line(tmp_path: Path):
    source_path = tmp_path / "stories.txt"
    source_path.write_text("first story\n\nsecond story\nthird story\n", encoding="utf-8")

    docs = list(LocalTextDocumentSource([source_path]).documents())

    assert [doc.text for doc in docs] == ["first story", "second story", "third story"]
    assert [doc.doc_id for doc in docs] == [
        f"{source_path}:1",
        f"{source_path}:3",
        f"{source_path}:4",
    ]


def test_eos_is_inserted_between_documents(tmp_path: Path):
    config = _build_config(tmp_path, block_size=8)
    prepare_data(config)

    train_tokens = np.fromfile(split_path(config.data.cache_dir, "train"), dtype=TOKEN_DTYPE)
    eos_id = config.tokenizer.vocab_size - config.tokenizer.reserved_special_tokens + 3

    # EOS boundaries stop the packed stream from inventing a transition from
    # one source document directly into the next.
    assert np.count_nonzero(train_tokens == eos_id) == 2


def test_train_loader_returns_shifted_batches(tmp_path: Path):
    config = _build_config(tmp_path, block_size=8)
    prepare_data(config)

    batch = next(iter(build_train_loader(config, split="train")))
    assert batch["input_ids"].shape == (2, 8)
    assert batch["targets"].shape == (2, 8)
    assert batch["input_ids"].dtype == torch.long
    assert batch["targets"].dtype == torch.long
    assert torch.equal(batch["targets"][:, :-1], batch["input_ids"][:, 1:])


def test_train_loader_is_deterministic_for_same_seed(tmp_path: Path):
    config = _build_config(tmp_path, block_size=8)
    prepare_data(config)

    first = next(iter(build_train_loader(config, split="train")))
    second = next(iter(build_train_loader(config, split="train")))

    assert torch.equal(first["input_ids"], second["input_ids"])
    assert torch.equal(first["targets"], second["targets"])


def test_small_train_split_fails_clearly(tmp_path: Path):
    config = _build_config(tmp_path, block_size=1000)

    with pytest.raises(ValueError, match="required for one shifted training block"):
        prepare_data(config)
