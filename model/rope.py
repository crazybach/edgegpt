"""Rotary positional embeddings for Llama-style attention.

RoPE injects position by rotating query and key vectors before the attention
dot product. It has no trainable parameters and replaces learned positional
embedding tables in Llama-family models. Values are not rotated: only Q and K
need position information for attention scores.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from configs.config import Config


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last dimension using the Llama/HF half-split layout.

    If `x = [x1, x2]` across the last dimension, the rotated vector is
    `[-x2, x1]`. Combined with cos/sin this is a batch of independent 2D
    rotations.
    """

    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply precomputed RoPE cos/sin tables to query and key tensors.

    Args:
        q, k: tensors shaped `[B, H, T, D]`.
        cos, sin: either full caches `[max_seq_len, D]` or sliced tables
            `[T, D]`.
        position_offset: starting position when full caches are supplied.
    """

    if q.shape != k.shape:
        raise ValueError(f"q and k must have the same shape, got {q.shape} and {k.shape}.")
    if q.ndim != 4:
        raise ValueError(f"RoPE expects q/k shape [B, H, T, D], got {q.shape}.")

    seq_len = q.shape[-2]
    head_dim = q.shape[-1]
    if head_dim % 2 != 0:
        raise ValueError(f"RoPE requires an even head_dim, got {head_dim}.")

    if cos.shape[-1] != head_dim or sin.shape[-1] != head_dim:
        raise ValueError("cos/sin last dimension must match q/k head_dim.")
    if cos.shape[-2] != seq_len:
        cos = cos[position_offset : position_offset + seq_len]
    if sin.shape[-2] != seq_len:
        sin = sin[position_offset : position_offset + seq_len]
    if cos.shape[-2] != seq_len or sin.shape[-2] != seq_len:
        raise ValueError("cos/sin caches are too short for the requested position range.")

    # Broadcast `[T, D]` over batch and heads: `[1, 1, T, D]`.
    cos = cos.to(device=q.device, dtype=q.dtype).unsqueeze(0).unsqueeze(0)
    sin = sin.to(device=q.device, dtype=q.dtype).unsqueeze(0).unsqueeze(0)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class RotaryEmbedding(nn.Module):
    """Cache and apply standard 1D RoPE for Q/K tensors."""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.dim = config.model.d_model // config.model.n_heads
        self.max_seq_len_cached = int(config.model.max_seq_len)
        self.base = float(config.model.rope_theta)

        if self.dim % 2 != 0:
            raise ValueError(f"RoPE requires an even head_dim, got {self.dim}.")

        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(self.max_seq_len_cached, device=inv_freq.device, dtype=inv_freq.dtype)

    def _set_cos_sin_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
        """Build cos/sin tables up to `seq_len`.

        The duplicated frequency layout gives cos/sin shape `[seq_len, D]`,
        matching the Llama/HF half-rotation formulation.
        """

        positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(positions, self.inv_freq.to(device=device))
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype=dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype=dtype), persistent=False)
        self.max_seq_len_cached = int(seq_len)

    def get_cos_sin(
        self,
        seq_len: int,
        position_offset: int = 0,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return sliced cos/sin tables for a sequence and position offset."""

        required_len = int(position_offset + seq_len)
        device = device or self.inv_freq.device
        dtype = dtype or self.inv_freq.dtype
        if required_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(required_len, device=device, dtype=dtype)

        cos = self.cos_cached[position_offset:required_len].to(device=device, dtype=dtype)
        sin = self.sin_cached[position_offset:required_len].to(device=device, dtype=dtype)
        return cos, sin

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rotate Q/K tensors shaped `[B, H, T, D]`."""

        cos, sin = self.get_cos_sin(
            q.shape[-2],
            position_offset=position_offset,
            dtype=q.dtype,
            device=q.device,
        )
        return apply_rotary_pos_emb(q, k, cos, sin)
