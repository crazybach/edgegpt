"""Phase 11 generation utilities for EdgeGPT."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from model import EdgeGPT, build_kv_cache


@dataclass
class GenerationConfig:
    """Runtime controls for autoregressive decoding."""

    max_new_tokens: int = 64
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None
    do_sample: bool = False
    seed: int | None = None
    eos_token_id: int | None = None


def _validate_generation_config(config: GenerationConfig) -> None:
    if config.max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative.")
    if config.temperature < 0:
        raise ValueError("temperature must be non-negative.")
    if config.top_k is not None and config.top_k <= 0:
        raise ValueError("top_k must be positive when set.")
    if config.top_p is not None and not (0.0 < config.top_p <= 1.0):
        raise ValueError("top_p must be in (0, 1] when set.")


def _apply_top_k(logits: torch.Tensor, top_k: int | None) -> torch.Tensor:
    if top_k is None or top_k >= logits.shape[-1]:
        return logits
    values, _ = torch.topk(logits, k=top_k, dim=-1)
    threshold = values[..., -1, None]
    return logits.masked_fill(logits < threshold, torch.finfo(logits.dtype).min)


def _apply_top_p(logits: torch.Tensor, top_p: float | None) -> torch.Tensor:
    if top_p is None or top_p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative = sorted_probs.cumsum(dim=-1)
    remove_sorted = cumulative > top_p
    remove_sorted[..., 1:] = remove_sorted[..., :-1].clone()
    remove_sorted[..., 0] = False
    remove = torch.zeros_like(remove_sorted).scatter(dim=-1, index=sorted_indices, src=remove_sorted)
    return logits.masked_fill(remove, torch.finfo(logits.dtype).min)


def sample_next_token(
    logits: torch.Tensor,
    config: GenerationConfig,
    generator: torch.Generator | None = None,
) -> torch.LongTensor:
    """Select the next token from final-position logits."""

    _validate_generation_config(config)
    if logits.ndim == 3:
        logits = logits[:, -1, :]
    if logits.ndim != 2:
        raise ValueError(f"logits must have shape [B, V] or [B, T, V], got {logits.shape}.")

    if not config.do_sample or config.temperature == 0:
        return torch.argmax(logits, dim=-1, keepdim=True).long()

    scaled = logits.float() / float(config.temperature)
    scaled = _apply_top_k(scaled, config.top_k)
    scaled = _apply_top_p(scaled, config.top_p)
    probs = F.softmax(scaled, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator).long()


def generate_ids(
    model: EdgeGPT,
    input_ids: torch.LongTensor,
    config: GenerationConfig,
    *,
    use_cache: bool = True,
) -> torch.LongTensor:
    """Generate token IDs from a prompt tensor."""

    _validate_generation_config(config)
    if input_ids.ndim != 2 or input_ids.dtype != torch.long:
        raise ValueError("input_ids must be a torch.long tensor with shape [B, T].")
    if input_ids.shape[1] <= 0:
        raise ValueError("input_ids must contain at least one prompt token.")

    max_total_len = input_ids.shape[1] + config.max_new_tokens
    if max_total_len > model.config.model.max_seq_len:
        raise ValueError(
            f"prompt + generation length {max_total_len} exceeds model.max_seq_len={model.config.model.max_seq_len}."
        )

    was_training = model.training
    model.eval()
    generator = None
    if config.seed is not None:
        generator = torch.Generator(device=input_ids.device)
        generator.manual_seed(int(config.seed))

    generated = input_ids.to(device=model.device)
    cache = None
    with torch.no_grad():
        if use_cache:
            dtype = next(model.parameters()).dtype
            cache = build_kv_cache(
                model.config,
                batch_size=generated.shape[0],
                max_seq_len=max_total_len,
                device=generated.device,
                dtype=dtype,
            )
            logits, _ = model(generated, kv_cache=cache, cache_position=0, position_offset=0)
        else:
            logits, _ = model(generated)
        if logits is None:
            raise RuntimeError("generation requires logits, got None.")

        for step in range(config.max_new_tokens):
            next_token = sample_next_token(logits, config, generator=generator)
            generated = torch.cat([generated, next_token], dim=1)
            if config.eos_token_id is not None and torch.all(next_token.squeeze(-1) == int(config.eos_token_id)):
                break
            if step == config.max_new_tokens - 1:
                break
            if use_cache:
                cache_position = generated.shape[1] - 1
                logits, _ = model(
                    next_token,
                    kv_cache=cache,
                    cache_position=cache_position,
                    position_offset=cache_position,
                )
            else:
                logits, _ = model(generated)
            if logits is None:
                raise RuntimeError("generation requires logits, got None.")

    if was_training:
        model.train()
    return generated