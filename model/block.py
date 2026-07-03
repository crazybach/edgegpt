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
from model.mlp import SwiGLUMLP
from model.norm import RMSNorm


class TransformerBlock(nn.Module):
    """Serial pre-norm decoder block.

    Shape contract:
        input:  `[B, T, d_model]`
        output: `[B, T, d_model]`
    """

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
        use_manual_attention: bool = False,
    ) -> torch.Tensor:
        if hidden.ndim != 3 or hidden.shape[-1] != self.d_model:
            raise ValueError(f"TransformerBlock expected hidden shape [B, T, {self.d_model}], got {hidden.shape}.")

        # Pre-norm residual branch 1: normalize the residual stream before
        # attention, then add the attention update back to the original stream.
        attention_update = self.attention(
            self.attention_norm(hidden),
            attention_mask=attention_mask,
            position_offset=position_offset,
            use_manual_attention=use_manual_attention,
        )
        hidden = hidden + attention_update

        # Pre-norm residual branch 2: the MLP sees the post-attention residual
        # stream, again normalized inside the residual branch.
        mlp_update = self.mlp(self.mlp_norm(hidden))
        return hidden + mlp_update
