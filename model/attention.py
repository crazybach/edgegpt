"""Phase 6 causal self-attention with Grouped-Query Attention.

This module is the first core transformer component. It follows the
Llama-family layout: project hidden states into Q/K/V, apply RoPE to Q/K only,
run causal scaled dot-product attention, then project back to `d_model`.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.config import Config
from model.cache import LayerKVCache
from model.rope import RotaryEmbedding


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads so they line up with query heads.

    GQA stores fewer K/V heads than query heads. During attention each KV head
    is shared by a group of query heads. This function expands `[B, H_kv, T, D]`
    to `[B, H_kv * n_rep, T, D]` without changing the learned projections.
    """

    if x.ndim != 4:
        raise ValueError(f"repeat_kv expects [B, H, T, D], got {x.shape}.")
    if n_rep <= 0:
        raise ValueError(f"n_rep must be positive, got {n_rep}.")
    if n_rep == 1:
        return x

    batch, n_kv_heads, seq_len, head_dim = x.shape
    x = x[:, :, None, :, :].expand(batch, n_kv_heads, n_rep, seq_len, head_dim)
    return x.reshape(batch, n_kv_heads * n_rep, seq_len, head_dim)


def _prepare_attention_mask(
    attention_mask: torch.Tensor | None,
    *,
    batch_size: int,
    query_len: int,
    key_len: int,
    device: torch.device,
) -> torch.Tensor | None:
    """Convert a token mask to an attention-compatible boolean mask."""

    if attention_mask is None:
        return None
    if attention_mask.shape != (batch_size, key_len):
        raise ValueError(
            "attention_mask must have shape [B, T_key], got "
            f"{attention_mask.shape} for batch={batch_size}, key_len={key_len}."
        )
    mask = attention_mask.to(device=device, dtype=torch.bool)
    return mask[:, None, None, :].expand(batch_size, 1, query_len, key_len)


def manual_scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool = True,
    attention_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    cache_position: int = 0,
) -> torch.Tensor:
    """Readable scaled dot-product attention used as a test oracle.

    Args:
        q: `[B, H_q, T_q, D]`
        k: `[B, H_kv, T_k, D]`
        v: `[B, H_kv, T_k, D]`
        attention_mask: optional token mask `[B, T_k]`, where 1 means visible.
        dropout_p: probability for attention-weight dropout during training.
        cache_position: absolute position of q[:, :, 0] in cached decoding.
    """

    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must all have shape [B, H, T, D].")
    if k.shape != v.shape:
        raise ValueError(f"k and v must have the same shape, got {k.shape} and {v.shape}.")
    if q.shape[0] != k.shape[0] or q.shape[-1] != k.shape[-1]:
        raise ValueError("q and k must share batch size and head_dim.")
    if q.shape[1] % k.shape[1] != 0:
        raise ValueError("query heads must be divisible by kv heads.")

    n_rep = q.shape[1] // k.shape[1]
    k = repeat_kv(k, n_rep)
    v = repeat_kv(v, n_rep)

    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.shape[-1])

    if is_causal:
        query_len, key_len = q.shape[-2], k.shape[-2]
        query_positions = torch.arange(
            int(cache_position),
            int(cache_position) + query_len,
            device=q.device,
        )
        key_positions = torch.arange(key_len, device=q.device)
        causal_mask = key_positions[None, :] <= query_positions[:, None]
        scores = scores.masked_fill(~causal_mask, torch.finfo(scores.dtype).min)

    prepared_mask = _prepare_attention_mask(
        attention_mask,
        batch_size=q.shape[0],
        query_len=q.shape[-2],
        key_len=k.shape[-2],
        device=q.device,
    )
    if prepared_mask is not None:
        scores = scores.masked_fill(~prepared_mask, torch.finfo(scores.dtype).min)

    weights = torch.softmax(scores.float(), dim=-1).to(dtype=q.dtype)
    if dropout_p > 0.0:
        weights = F.dropout(weights, p=dropout_p, training=True)
    return torch.matmul(weights, v)


class CausalSelfAttention(nn.Module):
    """Llama-style causal self-attention with GQA."""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.d_model = config.model.d_model
        self.n_heads = config.model.n_heads
        self.n_kv_heads = config.model.n_kv_heads
        self.dropout = float(config.model.dropout)

        if self.n_heads <= 0:
            raise ValueError("model.n_heads must be positive.")
        if self.d_model % self.n_heads != 0:
            raise ValueError("model.d_model must be divisible by model.n_heads.")
        if self.n_kv_heads <= 0:
            raise ValueError("model.n_kv_heads must be positive.")
        if self.n_kv_heads > self.n_heads:
            raise ValueError("model.n_kv_heads cannot exceed model.n_heads.")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("model.n_heads must be divisible by model.n_kv_heads.")

        self.head_dim = self.d_model // self.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads

        self.q_proj = nn.Linear(self.d_model, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, self.d_model, bias=False)
        self.rope = RotaryEmbedding(config)

    def _shape_projection(self, x: torch.Tensor, n_heads: int) -> torch.Tensor:
        """Reshape `[B, T, H * D]` into `[B, H, T, D]` for attention."""

        batch, seq_len, _ = x.shape
        return x.view(batch, seq_len, n_heads, self.head_dim).transpose(1, 2)

    def project_qkv(
        self,
        hidden: torch.Tensor,
        *,
        position_offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project hidden states and apply RoPE to Q/K only."""

        if hidden.ndim != 3 or hidden.shape[-1] != self.d_model:
            raise ValueError(f"hidden must have shape [B, T, {self.d_model}], got {hidden.shape}.")

        q = self._shape_projection(self.q_proj(hidden), self.n_heads)
        k = self._shape_projection(self.k_proj(hidden), self.n_kv_heads)
        v = self._shape_projection(self.v_proj(hidden), self.n_kv_heads)
        q, k = self.rope(q, k, position_offset=position_offset)
        return q, k, v

    def _sdpa_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: torch.Tensor | None,
        *,
        cache_position: int = 0,
    ) -> torch.Tensor:
        """Run PyTorch SDPA, falling back if the local version lacks GQA support."""

        prepared_mask = _prepare_attention_mask(
            attention_mask,
            batch_size=q.shape[0],
            query_len=q.shape[-2],
            key_len=k.shape[-2],
            device=q.device,
        )
        dropout_p = self.dropout if self.training else 0.0

        if prepared_mask is None and int(cache_position) == 0:
            try:
                return F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    dropout_p=dropout_p,
                    is_causal=True,
                    enable_gqa=self.n_rep > 1,
                )
            except TypeError:
                k = repeat_kv(k, self.n_rep)
                v = repeat_kv(v, self.n_rep)
                return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=True)

        return manual_scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            attention_mask=attention_mask,
            dropout_p=dropout_p,
            cache_position=cache_position,
        )

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        position_offset: int = 0,
        kv_cache: LayerKVCache | None = None,
        cache_position: int = 0,
        use_manual_attention: bool = False,
    ) -> torch.Tensor:
        """Apply causal self-attention to hidden states `[B, T, d_model]`."""

        q, k, v = self.project_qkv(hidden, position_offset=position_offset)
        if kv_cache is not None:
            k, v = kv_cache.append(k, v, cache_position=cache_position)
        if use_manual_attention:
            dropout_p = self.dropout if self.training else 0.0
            attn = manual_scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=True,
                attention_mask=attention_mask,
                dropout_p=dropout_p,
                cache_position=cache_position,
            )
        else:
            attn = self._sdpa_attention(q, k, v, attention_mask, cache_position=cache_position)

        attn = attn.transpose(1, 2).contiguous().view(hidden.shape[0], hidden.shape[1], self.d_model)
        return self.o_proj(attn)