"""Phase 7 SwiGLU feed-forward network.

The MLP is the second residual branch in a Llama-style decoder block. It expands
hidden states to a larger intermediate dimension, uses a SiLU-gated product,
and projects back to the residual-stream width.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.config import Config


class SwiGLUMLP(nn.Module):
    """Bias-free Llama-style SwiGLU MLP.

    Shape contract:
        input:  `[..., d_model]`
        output: `[..., d_model]`
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.d_model = int(config.model.d_model)
        self.d_ff = int(config.model.d_ff)

        if self.d_model <= 0:
            raise ValueError(f"model.d_model must be positive, got {self.d_model}.")
        if self.d_ff <= 0:
            raise ValueError(f"model.d_ff must be positive, got {self.d_ff}.")

        # Llama-compatible naming matters for future checkpoint export. The gate
        # branch decides which intermediate features pass through; the up branch
        # supplies the candidate values; the down branch returns to d_model.
        self.gate_proj = nn.Linear(self.d_model, self.d_ff, bias=False)
        self.up_proj = nn.Linear(self.d_model, self.d_ff, bias=False)
        self.down_proj = nn.Linear(self.d_ff, self.d_model, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.shape[-1] != self.d_model:
            raise ValueError(f"SwiGLUMLP expected last dimension {self.d_model}, got {hidden.shape[-1]}.")

        gate = F.silu(self.gate_proj(hidden))
        up = self.up_proj(hidden)
        return self.down_proj(gate * up)
