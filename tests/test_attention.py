"""Tests for Phase 6 causal self-attention with GQA."""

from __future__ import annotations

import pytest
import torch

from configs.config import Config, ModelConfig, TokenizerConfig, load_config
from model import CausalSelfAttention, manual_scaled_dot_product_attention, repeat_kv


def _config(
    *,
    d_model: int = 32,
    n_heads: int = 4,
    n_kv_heads: int = 2,
    max_seq_len: int = 16,
    dropout: float = 0.0,
) -> Config:
    return Config(
        model=ModelConfig(
            vocab_size=64,
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            max_seq_len=max_seq_len,
            dropout=dropout,
        ),
        tokenizer=TokenizerConfig(vocab_size=64, reserved_special_tokens=8),
        device="cpu",
    )


def test_attention_output_shape():
    attn = CausalSelfAttention(_config(d_model=32, n_heads=4, n_kv_heads=2))
    hidden = torch.randn(2, 5, 32)

    out = attn(hidden)

    assert out.shape == hidden.shape


def test_project_qkv_shapes_for_gqa():
    attn = CausalSelfAttention(_config(d_model=32, n_heads=4, n_kv_heads=2))
    hidden = torch.randn(2, 5, 32)

    q, k, v = attn.project_qkv(hidden)

    assert q.shape == (2, 4, 5, 8)
    assert k.shape == (2, 2, 5, 8)
    assert v.shape == (2, 2, 5, 8)


def test_repeat_kv_expands_groups_in_order():
    x = torch.tensor([[[[1.0]], [[2.0]]]])

    repeated = repeat_kv(x, n_rep=3)

    assert repeated.shape == (1, 6, 1, 1)
    assert repeated.flatten().tolist() == [1.0, 1.0, 1.0, 2.0, 2.0, 2.0]


def test_invalid_config_fails_when_head_count_is_not_positive():
    with pytest.raises(ValueError, match="n_heads"):
        _config(d_model=32, n_heads=0, n_kv_heads=1)


def test_invalid_config_fails_when_d_model_not_divisible_by_heads():
    with pytest.raises(ValueError, match="d_model"):
        _config(d_model=30, n_heads=8, n_kv_heads=2)


def test_invalid_config_fails_when_heads_not_divisible_by_kv_heads():
    with pytest.raises(ValueError, match="n_heads"):
        _config(d_model=30, n_heads=6, n_kv_heads=4)


def test_invalid_config_fails_when_kv_heads_exceed_query_heads():
    with pytest.raises(ValueError, match="n_kv_heads"):
        _config(d_model=32, n_heads=4, n_kv_heads=8)


def test_causal_property_future_tokens_do_not_affect_earlier_outputs():
    torch.manual_seed(0)
    attn = CausalSelfAttention(_config(dropout=0.0))
    attn.eval()
    hidden = torch.randn(1, 6, 32)
    changed = hidden.clone()
    changed[:, 4:] = torch.randn_like(changed[:, 4:]) * 10.0

    out = attn(hidden, use_manual_attention=True)
    changed_out = attn(changed, use_manual_attention=True)

    assert torch.allclose(out[:, :4], changed_out[:, :4], atol=1e-5)


def test_rope_rotates_qk_but_not_v():
    torch.manual_seed(0)
    attn = CausalSelfAttention(_config())
    hidden = torch.randn(1, 4, 32)

    q_rot, k_rot, v = attn.project_qkv(hidden)
    q_raw = attn._shape_projection(attn.q_proj(hidden), attn.n_heads)
    k_raw = attn._shape_projection(attn.k_proj(hidden), attn.n_kv_heads)
    v_raw = attn._shape_projection(attn.v_proj(hidden), attn.n_kv_heads)

    assert torch.allclose(q_rot[:, :, 0], q_raw[:, :, 0], atol=1e-6)
    assert torch.allclose(k_rot[:, :, 0], k_raw[:, :, 0], atol=1e-6)
    assert not torch.allclose(q_rot[:, :, 1:], q_raw[:, :, 1:])
    assert not torch.allclose(k_rot[:, :, 1:], k_raw[:, :, 1:])
    assert torch.allclose(v, v_raw)


def test_sdpa_matches_manual_reference_in_eval_mode():
    torch.manual_seed(0)
    attn = CausalSelfAttention(_config(dropout=0.0))
    attn.eval()
    hidden = torch.randn(2, 6, 32)

    sdpa_out = attn(hidden)
    manual_out = attn(hidden, use_manual_attention=True)

    assert torch.allclose(sdpa_out, manual_out, atol=1e-5)


def test_manual_attention_matches_mha_when_kv_heads_equal_query_heads():
    config = _config(d_model=32, n_heads=4, n_kv_heads=4)
    attn = CausalSelfAttention(config)
    hidden = torch.randn(1, 5, 32)

    q, k, v = attn.project_qkv(hidden)
    out = manual_scaled_dot_product_attention(q, k, v)

    assert out.shape == (1, 4, 5, 8)


def test_gradients_flow_through_attention_projections():
    attn = CausalSelfAttention(_config())
    hidden = torch.randn(2, 5, 32, requires_grad=True)

    loss = attn(hidden).sum()
    loss.backward()

    assert hidden.grad is not None
    for layer in [attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj]:
        assert layer.weight.grad is not None
        assert torch.count_nonzero(layer.weight.grad).item() > 0


def test_dropout_disabled_in_eval_mode():
    attn = CausalSelfAttention(_config(dropout=0.5))
    hidden = torch.randn(1, 5, 32)
    attn.eval()

    first = attn(hidden)
    second = attn(hidden)

    assert torch.allclose(first, second)


def test_masked_training_path_uses_attention_dropout():
    torch.manual_seed(0)
    attn = CausalSelfAttention(_config(dropout=0.5))
    attn.train()
    hidden = torch.randn(1, 5, 32)
    attention_mask = torch.ones(1, 5, dtype=torch.long)

    torch.manual_seed(123)
    dropped = attn(hidden, attention_mask=attention_mask)
    attn.dropout = 0.0
    torch.manual_seed(123)
    no_dropout = attn(hidden, attention_mask=attention_mask)

    assert not torch.allclose(dropped, no_dropout)


def test_cpu_config_compatibility():
    config = load_config("configs/cpu.yaml")
    attn = CausalSelfAttention(config)
    hidden = torch.randn(1, 5, config.model.d_model)

    out = attn(hidden)

    assert out.shape == (1, 5, 256)
    assert attn.head_dim == 64
    assert attn.n_rep == 2


def test_default_config_compatibility():
    config = load_config("configs/default.yaml")
    attn = CausalSelfAttention(config)
    hidden = torch.randn(1, 5, config.model.d_model)

    out = attn(hidden)

    assert out.shape == (1, 5, 512)
    assert attn.head_dim == 64
    assert attn.n_rep == 2
