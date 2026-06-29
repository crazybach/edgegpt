"""Tests for Phase 3 embedding contracts."""

from __future__ import annotations

import pytest
import torch

from configs.config import Config, ModelConfig, TokenizerConfig
from model import OutputProjection, TokenEmbedding, build_embedding_layers


def _config(
    *,
    vocab_size: int = 32,
    d_model: int = 8,
    embedding_scale: float | None = None,
    tie_embeddings: bool = True,
) -> Config:
    return Config(
        model=ModelConfig(
            vocab_size=vocab_size,
            d_model=d_model,
            embedding_scale=embedding_scale,
            tie_embeddings=tie_embeddings,
        ),
        tokenizer=TokenizerConfig(vocab_size=vocab_size, reserved_special_tokens=8),
        device="cpu",
    )


def test_token_embedding_shape():
    config = _config(vocab_size=40, d_model=12)
    embedding = TokenEmbedding(config)
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)

    hidden = embedding(input_ids)

    assert hidden.shape == (2, 3, 12)
    assert hidden.dtype == embedding.weight.dtype


def test_output_projection_shape_with_tied_weights():
    config = _config(vocab_size=40, d_model=12, tie_embeddings=True)
    embedding, projection = build_embedding_layers(config)
    hidden = embedding(torch.tensor([[1, 2, 3]], dtype=torch.long))

    logits = projection(hidden)

    assert logits.shape == (1, 3, 40)


def test_tied_output_projection_shares_embedding_storage():
    config = _config(tie_embeddings=True)
    embedding, projection = build_embedding_layers(config)

    assert projection.weight.data_ptr() == embedding.weight.data_ptr()


def test_untied_output_projection_allocates_separate_weights():
    config = _config(tie_embeddings=False)
    embedding = TokenEmbedding(config)
    projection = OutputProjection(config, embedding)

    assert projection.weight.data_ptr() != embedding.weight.data_ptr()


def test_embedding_scale_multiplies_output():
    base_config = _config(embedding_scale=None)
    scaled_config = _config(embedding_scale=2.0)
    base = TokenEmbedding(base_config)
    scaled = TokenEmbedding(scaled_config)
    scaled.embedding.weight.data.copy_(base.embedding.weight.data)
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

    assert torch.allclose(scaled(input_ids), base(input_ids) * 2.0)


def test_gradients_flow_to_tied_embedding_weights():
    config = _config(tie_embeddings=True)
    embedding, projection = build_embedding_layers(config)
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

    loss = projection(embedding(input_ids)).sum()
    loss.backward()

    assert embedding.weight.grad is not None
    assert torch.count_nonzero(embedding.weight.grad).item() > 0


def test_invalid_token_ids_fail_predictably():
    config = _config(vocab_size=16)
    embedding = TokenEmbedding(config)

    with pytest.raises((IndexError, RuntimeError)):
        embedding(torch.tensor([[16]], dtype=torch.long))


def test_embedding_rejects_non_long_inputs():
    embedding = TokenEmbedding(_config())

    with pytest.raises(TypeError, match="torch.long"):
        embedding(torch.tensor([[1.0, 2.0]]))


def test_no_learned_positional_embedding_parameters_exist():
    embedding, projection = build_embedding_layers(_config())
    names = [name for name, _ in list(embedding.named_parameters()) + list(projection.named_parameters())]

    assert all("pos" not in name.lower() and "position" not in name.lower() for name in names)
