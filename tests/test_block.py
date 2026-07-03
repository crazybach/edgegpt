"""Tests for Phase 8 transformer block assembly."""

from __future__ import annotations

import pytest
import torch

from configs.config import Config, ModelConfig, TokenizerConfig, load_config
from model import CausalSelfAttention, RMSNorm, SwiGLUMLP, TransformerBlock


def _config(*, d_model: int = 32, d_ff: int = 80, dropout: float = 0.0) -> Config:
    return Config(
        model=ModelConfig(
            vocab_size=64,
            d_model=d_model,
            d_ff=d_ff,
            n_heads=4,
            n_kv_heads=2,
            max_seq_len=16,
            dropout=dropout,
        ),
        tokenizer=TokenizerConfig(vocab_size=64, reserved_special_tokens=8),
        device="cpu",
    )


def test_block_preserves_shape():
    block = TransformerBlock(_config())
    hidden = torch.randn(2, 5, 32)

    out = block(hidden)

    assert out.shape == hidden.shape


def test_block_contains_expected_submodules():
    block = TransformerBlock(_config())

    assert isinstance(block.attention_norm, RMSNorm)
    assert isinstance(block.mlp_norm, RMSNorm)
    assert isinstance(block.attention, CausalSelfAttention)
    assert isinstance(block.mlp, SwiGLUMLP)
    assert block.attention_norm.weight.data_ptr() != block.mlp_norm.weight.data_ptr()


def test_residual_path_changes_output_but_preserves_shape():
    torch.manual_seed(0)
    block = TransformerBlock(_config())
    hidden = torch.randn(2, 5, 32)

    out = block(hidden)

    assert out.shape == hidden.shape
    assert not torch.allclose(out, hidden)


def test_block_matches_explicit_pre_norm_residual_formula():
    torch.manual_seed(0)
    block = TransformerBlock(_config())
    block.eval()
    hidden = torch.randn(2, 5, 32)

    attn_input = block.attention_norm(hidden)
    attn_update = block.attention(attn_input, use_manual_attention=True)
    after_attn = hidden + attn_update
    mlp_input = block.mlp_norm(after_attn)
    expected = after_attn + block.mlp(mlp_input)

    actual = block(hidden, use_manual_attention=True)

    assert torch.allclose(actual, expected, atol=1e-6)


def test_attention_branch_receives_normalized_input(monkeypatch: pytest.MonkeyPatch):
    block = TransformerBlock(_config())
    hidden = torch.randn(2, 5, 32)
    captured = {}

    def fake_attention_forward(attn_input, **kwargs):
        captured["attn_input"] = attn_input.detach().clone()
        return torch.zeros_like(attn_input)

    monkeypatch.setattr(block.attention, "forward", fake_attention_forward)

    block(hidden)

    assert torch.allclose(captured["attn_input"], block.attention_norm(hidden))


def test_mlp_branch_receives_normalized_post_attention_residual(monkeypatch: pytest.MonkeyPatch):
    block = TransformerBlock(_config())
    hidden = torch.randn(2, 5, 32)
    attention_update = torch.full_like(hidden, 0.25)
    captured = {}

    def fake_attention_forward(attn_input, **kwargs):
        return attention_update

    def fake_mlp_forward(mlp_input):
        captured["mlp_input"] = mlp_input.detach().clone()
        return torch.zeros_like(mlp_input)

    monkeypatch.setattr(block.attention, "forward", fake_attention_forward)
    monkeypatch.setattr(block.mlp, "forward", fake_mlp_forward)

    block(hidden)

    expected = block.mlp_norm(hidden + attention_update)
    assert torch.allclose(captured["mlp_input"], expected)


def test_gradients_flow_to_all_block_components():
    block = TransformerBlock(_config())
    hidden = torch.randn(2, 5, 32, requires_grad=True)

    loss = block(hidden).sum()
    loss.backward()

    assert hidden.grad is not None
    assert block.attention_norm.weight.grad is not None
    assert block.mlp_norm.weight.grad is not None
    assert block.attention.q_proj.weight.grad is not None
    assert block.attention.o_proj.weight.grad is not None
    assert block.mlp.gate_proj.weight.grad is not None
    assert block.mlp.down_proj.weight.grad is not None


def test_manual_attention_flag_routes_through_block():
    torch.manual_seed(0)
    block = TransformerBlock(_config(dropout=0.0))
    block.eval()
    hidden = torch.randn(1, 5, 32)

    manual = block(hidden, use_manual_attention=True)
    sdpa = block(hidden, use_manual_attention=False)

    assert torch.allclose(manual, sdpa, atol=1e-5)


def test_attention_mask_passes_through_block():
    block = TransformerBlock(_config(dropout=0.0))
    hidden = torch.randn(1, 5, 32)
    attention_mask = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.long)

    out = block(hidden, attention_mask=attention_mask)

    assert out.shape == hidden.shape


def test_cpu_config_compatibility():
    config = load_config("configs/cpu.yaml")
    block = TransformerBlock(config)
    hidden = torch.randn(1, 5, config.model.d_model)

    out = block(hidden)

    assert out.shape == (1, 5, 256)


def test_default_config_compatibility():
    config = load_config("configs/default.yaml")
    block = TransformerBlock(config)
    hidden = torch.randn(1, 5, config.model.d_model)

    out = block(hidden)

    assert out.shape == (1, 5, 512)


def test_block_does_not_create_model_level_modules():
    block = TransformerBlock(_config())
    module_names = set(dict(block.named_modules()))

    forbidden = {"token_embedding", "final_norm", "output_projection", "loss", "optimizer", "data_loader"}
    assert forbidden.isdisjoint(module_names)


def test_wrong_hidden_dimension_fails_clearly():
    block = TransformerBlock(_config(d_model=32))

    with pytest.raises(ValueError, match="hidden shape"):
        block(torch.randn(2, 5, 31))
