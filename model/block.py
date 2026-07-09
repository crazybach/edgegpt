"""Phase 8 Llama-compatible transformer decoder block.

The block composes the standalone Phase 5/6/7 modules into one residual decoder
layer. It intentionally remains a single layer; stacking and final logits are
part of the next model-level phase.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from configs.config import Config
from model.attention import CausalSelfAttention
from model.cache import LayerKVCache
from model.mlp import SwiGLUMLP
from model.norm import RMSNorm


class TransformerBlock(nn.Module):
    """Serial pre-norm decoder block."""

    def __init__(self, config: Config, layer_idx: int | None = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.d_model = int(config.model.d_model)

        self.attention_norm = RMSNorm(config)
        self.attention = CausalSelfAttention(config)
        self.mlp_norm = RMSNorm(config)
        self.mlp = SwiGLUMLP(config)

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        position_offset: int = 0,
        layer_cache: LayerKVCache | None = None,
        cache_position: int = 0,
        use_manual_attention: bool = False,
    ) -> torch.Tensor:
        if hidden.ndim != 3 or hidden.shape[-1] != self.d_model:
            raise ValueError(f"TransformerBlock expected hidden shape [B, T, {self.d_model}], got {hidden.shape}.")

        attention_update = self.attention(
            self.attention_norm(hidden),
            attention_mask=attention_mask,
            position_offset=position_offset,
            kv_cache=layer_cache,
            cache_position=cache_position,
            use_manual_attention=use_manual_attention,
        )
        hidden = hidden + attention_update

        mlp_update = self.mlp(self.mlp_norm(hidden))
        return hidden + mlp_update