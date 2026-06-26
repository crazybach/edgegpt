"""Tests for the Phase 1 tokenizer module.

These tests protect the tokenizer/checkpoint contract: token IDs feed directly
into embedding rows, so silent ID drift is a real model-breaking bug.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from configs.config import Config, ModelConfig, TokenizerConfig
from data.tokenizer import load_tokenizer
from data.tokenizer.registry import build_tokenizer
from data.tokenizer.special_tokens import special_token_ids


def _write_corpus(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "Hello EdgeGPT. This is a tiny tokenizer corpus.",
                "Numbers should stay compositional: 100 and 3.1415.",
                "Code keeps indentation:\n    def add(a, b):\n        return a + b",
                "Unicode survives: 你好世界",
                "Emoji survives: 🙂‍↔️ 👨‍👩‍👧‍👦",
                "<|user|> is a special token string, not normal prose.",
            ]
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture()
def tiny_tokenizer(tmp_path: Path):
    """Train a small tokenizer so tests stay fast while exercising real BPE."""

    vocab_size = 512
    reserved = 64
    config = TokenizerConfig(
        vocab_size=vocab_size,
        reserved_special_tokens=reserved,
        artifact_dir=str(tmp_path / "tok"),
        train_files=[str(_write_corpus(tmp_path / "corpus.txt"))],
        min_frequency=1,
        split_digits=True,
    )
    tokenizer = build_tokenizer(config)
    tokenizer.train([Path(config.train_files[0])], Path(config.artifact_dir))
    return tokenizer, config


def test_config_rejects_vocab_size_mismatch():
    # The model's embedding table and tokenizer ID space must have identical
    # sizes; otherwise some token IDs would index missing or wrong rows.
    with pytest.raises(ValueError, match="model.vocab_size must match tokenizer.vocab_size"):
        Config(model=ModelConfig(vocab_size=16384), tokenizer=TokenizerConfig(vocab_size=8192))


def test_special_token_ids_are_stable(tiny_tokenizer):
    tokenizer, config = tiny_tokenizer
    expected = special_token_ids(config.vocab_size, config.reserved_special_tokens)

    for token, expected_id in expected.items():
        assert tokenizer.token_to_id(token) == expected_id

    assert tokenizer.token_to_id("<|pad|>") == config.vocab_size - config.reserved_special_tokens
    assert tokenizer.token_to_id("<|unk|>") == config.vocab_size - config.reserved_special_tokens + 1
    assert tokenizer.token_to_id("<|bos|>") == config.vocab_size - config.reserved_special_tokens + 2
    assert tokenizer.token_to_id("<|eos|>") == config.vocab_size - config.reserved_special_tokens + 3


@pytest.mark.parametrize(
    "text",
    [
        "plain ASCII text",
        "你好世界",
        "🙂‍↔️👨‍👩‍👧‍👦",
        "def f(x):\n    return x + 1\n",
        " spaces\t tabs\npunctuation!!! ",
    ],
)
def test_round_trip_text(tiny_tokenizer, text: str):
    tokenizer, _ = tiny_tokenizer
    ids = tokenizer.encode(text)
    assert tokenizer.decode(ids) == text


def test_bos_eos_insertion(tiny_tokenizer):
    tokenizer, _ = tiny_tokenizer
    ids = tokenizer.encode("hello", add_bos=True, add_eos=True)

    assert int(ids[0]) == tokenizer.token_to_id("<|bos|>")
    assert int(ids[-1]) == tokenizer.token_to_id("<|eos|>")
    assert tokenizer.decode(ids, skip_special_tokens=True) == "hello"


def test_special_tokens_are_not_split(tiny_tokenizer):
    tokenizer, _ = tiny_tokenizer
    ids = tokenizer.encode("<|user|>")

    assert ids.tolist() == [tokenizer.token_to_id("<|user|>")]


def test_batch_padding_and_attention_mask(tiny_tokenizer):
    tokenizer, _ = tiny_tokenizer
    batch = tokenizer.encode_batch(["a", "longer text"], padding=True)

    assert set(batch) == {"input_ids", "attention_mask"}
    assert batch["input_ids"].shape == batch["attention_mask"].shape
    assert batch["input_ids"].dtype == torch.long
    assert batch["attention_mask"].dtype == torch.long

    pad_id = tokenizer.token_to_id("<|pad|>")
    pad_positions = batch["attention_mask"] == 0
    assert torch.all(batch["input_ids"][pad_positions] == pad_id)


def test_save_load_parity(tiny_tokenizer):
    tokenizer, config = tiny_tokenizer
    text = "Reloaded tokenizer must produce identical IDs."
    before = tokenizer.encode(text)

    reloaded = load_tokenizer(config, config.artifact_dir)
    after = reloaded.encode(text)

    assert before.tolist() == after.tolist()
    assert reloaded.decode(after) == text


def test_encoded_ids_stay_in_vocab_bounds(tiny_tokenizer):
    tokenizer, config = tiny_tokenizer
    ids = tokenizer.encode("Bounds check: 你好 🙂 123", add_bos=True, add_eos=True)

    assert int(ids.min()) >= 0
    assert int(ids.max()) < config.vocab_size


def test_digit_split_keeps_digits_compositional(tiny_tokenizer):
    tokenizer, _ = tiny_tokenizer
    ids = tokenizer.encode("100")
    pieces = [tokenizer.decode([int(token_id)]) for token_id in ids]

    assert pieces == ["1", "0", "0"]
