"""Tests for Phase 5 RMSNorm contracts."""

from __future__ import annotations

import pytest
import torch

from configs.config import Config, ModelConfig, TokenizerConfig, load_config
from model import RMSNorm


def _config(*, d_model: int = 8, norm_eps: float = 1e-5) -> Config:
    return Config(
        model=ModelConfig(vocab_size=64, d_model=d_model, norm_eps=norm_eps),
        tokenizer=TokenizerConfig(vocab_size=64, reserved_special_tokens=8),
        device="cpu",
    )


def _reference_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Direct formula used to catch accidental LayerNorm-style behavior."""

    variance = x.float().pow(2).mean(dim=-1, keepdim=True)
    return (x.float() * torch.rsqrt(variance + eps)).to(dtype=x.dtype) * weight.to(dtype=x.dtype)


def test_rms_norm_preserves_input_shape():
    norm = RMSNorm(_config(d_model=12))
    x = torch.randn(2, 3, 12)

    y = norm(x)

    assert y.shape == x.shape


def test_rms_norm_has_only_weight_parameter_and_no_bias():
    norm = RMSNorm(_config(d_model=12))

    params = dict(norm.named_parameters())

    assert list(params) == ["weight"]
    assert not hasattr(norm, "bias")


def test_rms_norm_weight_initializes_to_ones():
    norm = RMSNorm(10)

    assert torch.allclose(norm.weight, torch.ones(10))


def test_rms_output_has_unit_rms_when_weight_is_one():
    norm = RMSNorm(_config(d_model=16), eps=0.0)
    x = torch.randn(4, 5, 16)

    y = norm(x)
    rms = torch.sqrt(y.pow(2).mean(dim=-1))

    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-5)


def test_rms_norm_matches_reference_implementation():
    norm = RMSNorm(_config(d_model=8, norm_eps=1e-6))
    x = torch.randn(2, 3, 8)
    norm.weight.data = torch.linspace(0.5, 1.5, steps=8)

    expected = _reference_rms_norm(x, norm.weight, norm.eps)

    assert torch.allclose(norm(x), expected, atol=1e-6)


def test_rms_norm_preserves_float32_dtype():
    norm = RMSNorm(8)
    x = torch.randn(2, 3, 8, dtype=torch.float32)

    assert norm(x).dtype == torch.float32


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rms_norm_preserves_low_precision_dtype(dtype: torch.dtype):
    norm = RMSNorm(8)
    x = torch.randn(2, 3, 8, dtype=dtype)

    assert norm(x).dtype == dtype


def test_gradients_flow_to_input_and_weight():
    norm = RMSNorm(8)
    x = torch.randn(2, 3, 8, requires_grad=True)

    loss = norm(x).sum()
    loss.backward()

    assert x.grad is not None
    assert norm.weight.grad is not None
    assert torch.count_nonzero(norm.weight.grad).item() > 0


def test_norm_eps_changes_near_zero_behavior():
    x = torch.full((1, 2, 4), 1e-7)
    small_eps = RMSNorm(4, eps=1e-12)
    large_eps = RMSNorm(4, eps=1e-3)

    assert not torch.allclose(small_eps(x), large_eps(x))


def test_cpu_config_compatibility():
    config = load_config("configs/cpu.yaml")
    norm = RMSNorm(config)
    x = torch.randn(1, 5, config.model.d_model)

    y = norm(x)

    assert y.shape == (1, 5, 256)
    assert norm.eps == pytest.approx(1e-5)


def test_rms_norm_does_not_subtract_mean():
    norm = RMSNorm(4, eps=0.0)
    x = torch.tensor([[[1.0, 1.0, 1.0, 1.0]]])

    y = norm(x)

    # LayerNorm would output zeros for a constant vector. RMSNorm preserves the
    # direction and scale-normalizes it instead.
    assert torch.allclose(y, torch.ones_like(y))


def test_wrong_hidden_dimension_fails_clearly():
    norm = RMSNorm(8)

    with pytest.raises(ValueError, match="last dimension 8"):
        norm(torch.randn(2, 3, 7))
