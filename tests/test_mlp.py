"""Tests for Phase 7 SwiGLU MLP contracts."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from configs.config import Config, ModelConfig, TokenizerConfig, load_config
from model import SwiGLUMLP


def _config(*, d_model: int = 8, d_ff: int = 24) -> Config:
    return Config(
        model=ModelConfig(vocab_size=64, d_model=d_model, d_ff=d_ff, n_heads=4, n_kv_heads=2),
        tokenizer=TokenizerConfig(vocab_size=64, reserved_special_tokens=8),
        device="cpu",
    )


def test_swiglu_mlp_preserves_shape():
    mlp = SwiGLUMLP(_config(d_model=12, d_ff=32))
    hidden = torch.randn(2, 3, 12)

    out = mlp(hidden)

    assert out.shape == hidden.shape


def test_swiglu_mlp_supports_arbitrary_leading_dimensions():
    mlp = SwiGLUMLP(_config(d_model=8, d_ff=24))
    hidden = torch.randn(2, 3, 4, 8)

    out = mlp(hidden)

    assert out.shape == hidden.shape


def test_projection_shapes_are_llama_compatible():
    mlp = SwiGLUMLP(_config(d_model=12, d_ff=32))

    assert mlp.gate_proj.weight.shape == (32, 12)
    assert mlp.up_proj.weight.shape == (32, 12)
    assert mlp.down_proj.weight.shape == (12, 32)


def test_all_projections_are_bias_free():
    mlp = SwiGLUMLP(_config())

    assert mlp.gate_proj.bias is None
    assert mlp.up_proj.bias is None
    assert mlp.down_proj.bias is None


def test_swiglu_matches_direct_reference_formula():
    torch.manual_seed(0)
    mlp = SwiGLUMLP(_config(d_model=8, d_ff=24))
    hidden = torch.randn(2, 3, 8)

    expected = mlp.down_proj(F.silu(mlp.gate_proj(hidden)) * mlp.up_proj(hidden))

    assert torch.allclose(mlp(hidden), expected, atol=1e-6)


def test_gradients_flow_to_input_and_all_projection_weights():
    mlp = SwiGLUMLP(_config(d_model=8, d_ff=24))
    hidden = torch.randn(2, 3, 8, requires_grad=True)

    loss = mlp(hidden).sum()
    loss.backward()

    assert hidden.grad is not None
    for layer in [mlp.gate_proj, mlp.up_proj, mlp.down_proj]:
        assert layer.weight.grad is not None
        assert torch.count_nonzero(layer.weight.grad).item() > 0


def test_zero_gate_suppresses_output():
    mlp = SwiGLUMLP(_config(d_model=8, d_ff=24))
    hidden = torch.randn(2, 3, 8)
    mlp.gate_proj.weight.data.zero_()

    out = mlp(hidden)

    assert torch.allclose(out, torch.zeros_like(out), atol=1e-7)


def test_invalid_d_ff_fails_at_config_validation():
    with pytest.raises(ValueError, match="d_ff"):
        _config(d_model=8, d_ff=0)


def test_wrong_hidden_dimension_fails_clearly():
    mlp = SwiGLUMLP(_config(d_model=8, d_ff=24))

    with pytest.raises(ValueError, match="last dimension 8"):
        mlp(torch.randn(2, 3, 7))


def test_cpu_config_compatibility():
    config = load_config("configs/cpu.yaml")
    mlp = SwiGLUMLP(config)
    hidden = torch.randn(1, 5, config.model.d_model)

    out = mlp(hidden)

    assert out.shape == (1, 5, 256)
    assert mlp.d_ff == 704


def test_default_config_compatibility():
    config = load_config("configs/default.yaml")
    mlp = SwiGLUMLP(config)
    hidden = torch.randn(1, 5, config.model.d_model)

    out = mlp(hidden)

    assert out.shape == (1, 5, 512)
    assert mlp.d_ff == 1408


def test_mlp_has_no_norm_attention_or_output_projection_modules():
    mlp = SwiGLUMLP(_config())
    module_names = set(dict(mlp.named_modules()))

    assert "norm" not in module_names
    assert "attention" not in module_names
    assert "output_projection" not in module_names
