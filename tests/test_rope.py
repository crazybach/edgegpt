"""Tests for Phase 4 Rotary Positional Embeddings."""

from __future__ import annotations

import pytest
import torch

from configs.config import Config, ModelConfig, TokenizerConfig, load_config
from model import RotaryEmbedding, apply_rotary_pos_emb, rotate_half


def _config(
    *,
    d_model: int = 32,
    n_heads: int = 4,
    max_seq_len: int = 16,
    rope_theta: float = 10000.0,
) -> Config:
    return Config(
        model=ModelConfig(
            vocab_size=64,
            d_model=d_model,
            n_heads=n_heads,
            max_seq_len=max_seq_len,
            rope_theta=rope_theta,
        ),
        tokenizer=TokenizerConfig(vocab_size=64, reserved_special_tokens=8),
        device="cpu",
    )


def test_rope_preserves_qk_shapes():
    config = _config(d_model=32, n_heads=4)
    rope = RotaryEmbedding(config)
    q = torch.randn(2, 4, 5, 8)
    k = torch.randn(2, 4, 5, 8)

    q_rot, k_rot = rope(q, k)

    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape


def test_rope_has_no_trainable_parameters():
    rope = RotaryEmbedding(_config())

    assert list(rope.parameters()) == []
    assert "cos_cached" in dict(rope.named_buffers())
    assert "sin_cached" in dict(rope.named_buffers())


def test_buffers_follow_module_dtype():
    rope = RotaryEmbedding(_config()).to(dtype=torch.float64)
    q = torch.randn(1, 4, 3, 8, dtype=torch.float64)
    k = torch.randn(1, 4, 3, 8, dtype=torch.float64)

    q_rot, k_rot = rope(q, k)

    assert rope.cos_cached.dtype == torch.float64
    assert rope.sin_cached.dtype == torch.float64
    assert q_rot.dtype == torch.float64
    assert k_rot.dtype == torch.float64


def test_rotation_preserves_vector_norm():
    rope = RotaryEmbedding(_config())
    q = torch.randn(2, 4, 7, 8)
    k = torch.randn(2, 4, 7, 8)

    q_rot, k_rot = rope(q, k)

    assert torch.allclose(q_rot.norm(dim=-1), q.norm(dim=-1), atol=1e-5)
    assert torch.allclose(k_rot.norm(dim=-1), k.norm(dim=-1), atol=1e-5)


def test_position_zero_leaves_vectors_unchanged():
    rope = RotaryEmbedding(_config())
    q = torch.randn(1, 4, 1, 8)
    k = torch.randn(1, 4, 1, 8)

    q_rot, k_rot = rope(q, k, position_offset=0)

    assert torch.allclose(q_rot, q, atol=1e-6)
    assert torch.allclose(k_rot, k, atol=1e-6)


def test_apply_rotary_pos_emb_rotates_q_and_k_only():
    rope = RotaryEmbedding(_config())
    q = torch.randn(1, 4, 4, 8)
    k = torch.randn(1, 4, 4, 8)
    cos, sin = rope.get_cos_sin(seq_len=4)

    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)

    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape
    assert not torch.allclose(q_rot[:, :, 1:], q[:, :, 1:])
    assert not torch.allclose(k_rot[:, :, 1:], k[:, :, 1:])


def test_relative_position_property_holds():
    rope = RotaryEmbedding(_config(max_seq_len=32))
    x = torch.randn(8)
    y = torch.randn(8)
    cos, sin = rope.get_cos_sin(seq_len=32)

    def rotate_at(vector: torch.Tensor, position: int) -> torch.Tensor:
        return (vector * cos[position]) + (rotate_half(vector) * sin[position])

    m, n, shift = 3, 9, 7
    first = torch.dot(rotate_at(x, m), rotate_at(y, n))
    shifted = torch.dot(rotate_at(x, m + shift), rotate_at(y, n + shift))

    assert torch.allclose(first, shifted, atol=1e-5)


def test_cache_extends_for_position_offset():
    rope = RotaryEmbedding(_config(max_seq_len=4))
    q = torch.randn(1, 4, 3, 8)
    k = torch.randn(1, 4, 3, 8)

    q_rot, k_rot = rope(q, k, position_offset=5)

    assert rope.max_seq_len_cached >= 8
    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape


def test_odd_head_dim_fails_clearly():
    with pytest.raises(ValueError, match="even head_dim"):
        RotaryEmbedding(_config(d_model=30, n_heads=2))


def test_real_cpu_config_compatibility():
    config = load_config("configs/cpu.yaml")
    rope = RotaryEmbedding(config)
    q = torch.randn(1, config.model.n_heads, 5, config.head_dim)
    k = torch.randn(1, config.model.n_heads, 5, config.head_dim)

    q_rot, k_rot = rope(q, k)

    assert q_rot.shape == (1, 4, 5, 64)
    assert k_rot.shape == (1, 4, 5, 64)
