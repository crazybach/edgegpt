"""Tests for Phase 11 inference and generation."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from configs.config import Config, DataConfig, ModelConfig, TokenizerConfig, TrainingConfig
from eval import GenerationConfig, generate_ids, sample_next_token
from model import EdgeGPT, build_kv_cache


def _small_config(
    *,
    vocab_size: int = 64,
    d_model: int = 32,
    n_layers: int = 2,
    n_heads: int = 4,
    n_kv_heads: int = 2,
    d_ff: int = 88,
    max_seq_len: int = 32,
) -> Config:
    return Config(
        model=ModelConfig(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            d_ff=d_ff,
            max_seq_len=max_seq_len,
        ),
        tokenizer=TokenizerConfig(vocab_size=vocab_size, reserved_special_tokens=16),
        training=TrainingConfig(dtype="fp32"),
        data=DataConfig(),
        device="cpu",
    )


def test_kv_cache_allocation_shape_and_zero_init():
    config = _small_config()
    cache = build_kv_cache(config, batch_size=3, max_seq_len=11, device="cpu", dtype=torch.float32)

    assert len(cache) == config.model.n_layers
    layer = cache[0]
    assert layer.k.shape == (3, config.model.n_kv_heads, 11, config.model.d_model // config.model.n_heads)
    assert layer.v.shape == layer.k.shape
    assert layer.k.dtype == torch.float32
    assert torch.count_nonzero(layer.k) == 0
    assert torch.count_nonzero(layer.v) == 0


def test_kv_cache_append_returns_visible_prefix():
    config = _small_config()
    cache = build_kv_cache(config, batch_size=1, max_seq_len=6, device="cpu", dtype=torch.float32)
    k = torch.ones(1, config.model.n_kv_heads, 2, config.model.d_model // config.model.n_heads)
    v = torch.full_like(k, 2.0)

    visible_k, visible_v = cache[0].append(k, v, cache_position=3)

    assert visible_k.shape[2] == 5
    assert torch.allclose(visible_k[:, :, 3:5, :], k)
    assert torch.allclose(visible_v[:, :, 3:5, :], v)
    assert torch.count_nonzero(visible_k[:, :, :3, :]) == 0


def test_gqa_cache_stores_kv_heads_only():
    config = _small_config(n_heads=4, n_kv_heads=1)
    cache = build_kv_cache(config, batch_size=1, max_seq_len=8, device="cpu", dtype=torch.float32)

    assert cache[0].k.shape[1] == 1
    assert cache[0].k.shape[1] != config.model.n_heads


def test_cached_next_token_logits_match_full_recompute():
    torch.manual_seed(123)
    config = _small_config()
    model = EdgeGPT(config)
    model.eval()
    prompt = torch.randint(0, config.model.vocab_size, (1, 5), dtype=torch.long)
    next_token = torch.randint(0, config.model.vocab_size, (1, 1), dtype=torch.long)

    cache = build_kv_cache(config, batch_size=1, max_seq_len=8, device="cpu", dtype=torch.float32)
    with torch.no_grad():
        model(prompt, kv_cache=cache, cache_position=0, position_offset=0)
        cached_logits, _ = model(
            next_token,
            kv_cache=cache,
            cache_position=prompt.shape[1],
            position_offset=prompt.shape[1],
        )
        full_logits, _ = model(torch.cat([prompt, next_token], dim=1))

    assert cached_logits is not None and full_logits is not None
    assert torch.allclose(cached_logits[:, -1, :], full_logits[:, -1, :], atol=1e-5, rtol=1e-4)


def test_cached_greedy_generation_matches_full_recompute():
    torch.manual_seed(123)
    config = _small_config()
    model = EdgeGPT(config)
    model.eval()
    prompt = torch.randint(0, config.model.vocab_size, (1, 6), dtype=torch.long)
    gen_config = GenerationConfig(max_new_tokens=5, do_sample=False)

    cached = generate_ids(model, prompt, gen_config, use_cache=True)
    full = generate_ids(model, prompt, gen_config, use_cache=False)

    assert cached.tolist() == full.tolist()


def test_sample_next_token_greedy_returns_argmax():
    logits = torch.tensor([[0.0, 2.0, 1.0]])

    token = sample_next_token(logits, GenerationConfig(do_sample=False))

    assert token.tolist() == [[1]]


def test_temperature_zero_returns_argmax_even_when_sampling_enabled():
    logits = torch.tensor([[0.0, 2.0, 1.0]])

    token = sample_next_token(logits, GenerationConfig(do_sample=True, temperature=0.0))

    assert token.tolist() == [[1]]


def test_top_k_sampling_excludes_tokens_outside_k():
    logits = torch.tensor([[0.0, 1.0, 5.0]])
    generator = torch.Generator().manual_seed(7)

    token = sample_next_token(logits, GenerationConfig(do_sample=True, top_k=1), generator)

    assert token.tolist() == [[2]]


def test_top_p_sampling_keeps_smallest_nucleus():
    logits = torch.tensor([[5.0, 1.0, 0.0]])
    generator = torch.Generator().manual_seed(7)

    token = sample_next_token(logits, GenerationConfig(do_sample=True, top_p=0.6), generator)

    assert token.tolist() == [[0]]


class _EOSModel(torch.nn.Module):
    def __init__(self, config: Config, eos_token_id: int):
        super().__init__()
        self.config = config
        self.eos_token_id = eos_token_id
        self.param = torch.nn.Parameter(torch.zeros(1))

    @property
    def device(self) -> torch.device:
        return self.param.device

    def forward(self, input_ids, **kwargs):  # noqa: ANN001, ANN003
        logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], self.config.model.vocab_size)
        logits[:, -1, self.eos_token_id] = 10.0
        return logits, None


def test_generation_stops_on_eos():
    config = _small_config(vocab_size=32)
    eos_id = 3
    model = _EOSModel(config, eos_id)
    prompt = torch.tensor([[1, 2]], dtype=torch.long)

    out = generate_ids(model, prompt, GenerationConfig(max_new_tokens=5, eos_token_id=eos_id), use_cache=False)

    assert out.tolist() == [[1, 2, eos_id]]


def test_generation_rejects_context_overflow():
    config = _small_config(max_seq_len=4)
    model = EdgeGPT(config)
    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)

    with pytest.raises(ValueError, match="exceeds model.max_seq_len"):
        generate_ids(model, prompt, GenerationConfig(max_new_tokens=2), use_cache=True)


@pytest.mark.skipif(
    not Path("artifacts/runs/tinystories_full_gpu_test_1000/latest.pt").exists(),
    reason="full TinyStories checkpoint artifact is not present",
)
def test_default_checkpoint_cli_smoke():
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "scripts/generate.py",
            "--prompt",
            "Once upon a time",
            "--max-new-tokens",
            "4",
            "--device",
            "cpu",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.stdout.strip()