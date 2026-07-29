"""Tests for the Improvement 1.0 training and corpus-building path."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from configs.config import Config, DataConfig, ModelConfig, TokenizerConfig, TrainingConfig, load_config
from data.pipeline.dataset import BoundedRandomSampler
from model import EdgeGPT

corpus_builder = importlib.import_module("scripts.prepare_improvement_1_corpus")


def _checkpoint_config() -> Config:
    return Config(
        model=ModelConfig(
            vocab_size=64,
            d_model=32,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            d_ff=88,
            max_seq_len=16,
        ),
        training=TrainingConfig(dtype="fp32", gradient_checkpointing=True),
        data=DataConfig(block_size=16),
        tokenizer=TokenizerConfig(vocab_size=64, reserved_special_tokens=16),
        device="cpu",
    )


def test_improvement_1_config_is_2k_and_8gb_safe_profile():
    config = load_config("configs/improvement_1_20m_2k.yaml")

    assert config.model.max_seq_len == 2048
    assert config.data.block_size == 2048
    assert config.data.source_paths == ["./data/improvement_1/train.jsonl"]
    assert config.tokenizer.train_files == ["./data/improvement_1/tokenizer_train.txt"]
    assert config.training.batch_size == 1
    assert config.training.gradient_accumulation_steps == 8
    assert config.training.chunked_loss is True
    assert config.training.gradient_checkpointing is True

    model = EdgeGPT(config)
    assert model.count_parameters()["total"] == 18_880_896


def test_config_rejects_training_blocks_larger_than_context():
    with pytest.raises(ValueError, match="block_size cannot exceed"):
        Config(
            model=ModelConfig(max_seq_len=16),
            data=DataConfig(block_size=32),
        )


def test_gradient_checkpointing_runs_each_block_and_backpropagates(monkeypatch):
    config = _checkpoint_config()
    model = EdgeGPT(config)
    model.train()
    model_module = importlib.import_module("model.model")
    real_checkpoint = model_module.checkpoint
    calls = 0

    def tracked_checkpoint(function, *args, **kwargs):
        nonlocal calls
        calls += 1
        return real_checkpoint(function, *args, **kwargs)

    monkeypatch.setattr(model_module, "checkpoint", tracked_checkpoint)
    input_ids = torch.randint(0, config.model.vocab_size, (2, 12), dtype=torch.long)
    targets = torch.randint(0, config.model.vocab_size, (2, 12), dtype=torch.long)

    _, loss = model(input_ids, targets)
    assert loss is not None
    loss.backward()

    assert calls == config.model.n_layers
    assert model.layers[0].attention.q_proj.weight.grad is not None


def test_bounded_sampler_resume_preserves_buffered_sequence():
    dataset = torch.utils.data.TensorDataset(torch.arange(100))
    first_generator = torch.Generator().manual_seed(123)
    first = BoundedRandomSampler(dataset, generator=first_generator, chunk_size=8)
    iterator = iter(first)
    for _ in range(5):
        next(iterator)
    state = first.state_dict()
    expected = [next(iterator) for _ in range(20)]

    restored = BoundedRandomSampler(
        dataset,
        generator=torch.Generator().manual_seed(999),
        chunk_size=8,
    )
    restored.load_state_dict(state)
    restored_iterator = iter(restored)
    actual = [next(restored_iterator) for _ in range(20)]

    assert actual == expected
    assert restored._buffer.numel() <= restored.chunk_size


class _FakeTokenizer:
    def encode_texts(self, texts: list[str]) -> list[list[int]]:
        return [list(range(len(text.split()))) for text in texts]


def test_corpus_builder_enforces_source_budgets_and_balances_tokenizer_sample(tmp_path, monkeypatch):
    source_a = corpus_builder.CorpusSource(name="a", kind="test")
    source_b = corpus_builder.CorpusSource(name="b", kind="test")
    monkeypatch.setitem(corpus_builder.PROFILES, "unit", {"a": 5, "b": 9})
    monkeypatch.setitem(corpus_builder.SOURCES, "a", source_a)
    monkeypatch.setitem(corpus_builder.SOURCES, "b", source_b)
    monkeypatch.setattr(
        corpus_builder,
        "load_config",
        lambda path: SimpleNamespace(tokenizer=SimpleNamespace(artifact_dir="fake-tokenizer")),
    )
    monkeypatch.setattr(corpus_builder, "load_tokenizer", lambda config: _FakeTokenizer())
    monkeypatch.setattr(
        corpus_builder,
        "iter_documents",
        lambda source, **kwargs: iter([f"{source.name} one two", f"{source.name} three four"] * 10),
    )

    manifest = corpus_builder.build_corpus(
        profile="unit",
        output_dir=tmp_path,
        tokenizer_config_path=Path("unused.yaml"),
        tokenizer_sample_tokens=8,
    )

    assert manifest["total_tokens"] >= 14
    assert manifest["stats"]["a"]["tokenizer_sample_tokens"] > 0
    assert manifest["stats"]["b"]["tokenizer_sample_tokens"] > 0
    rows = [json.loads(line) for line in (tmp_path / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {row["source"] for row in rows} == {"a", "b"}
    assert (tmp_path / "tokenizer_train.txt").read_text(encoding="utf-8").strip()

def test_corpus_builder_rejects_duplicate_sources(tmp_path):
    with pytest.raises(ValueError, match="cannot contain duplicates"):
        corpus_builder.build_corpus(
            profile="pilot",
            output_dir=tmp_path,
            tokenizer_config_path=Path("unused.yaml"),
            selected_sources=["tinystories", "tinystories"],
        )
