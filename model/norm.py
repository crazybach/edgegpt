"""Phase 5 RMSNorm module.

RMSNorm is the normalization used by Llama-style decoder models. It is simpler
than LayerNorm: instead of subtracting the mean and dividing by standard
deviation, it divides by the root mean square of the hidden vector and then
learns one gain value per hidden channel.
"""

from __future__ import annotations

from typing import Union

import torch
import torch.nn as nn

from configs.config import Config


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    The module is intentionally small and local instead of delegating to
    `torch.nn.RMSNorm`. EdgeGPT supports a broad PyTorch version range, and a
    custom implementation keeps the model math stable for tests, exports, and
    future checkpoint conversion.

    Shape contract:
        input:  `[..., d_model]`
        output: `[..., d_model]`
    """

    def __init__(self, config_or_dim: Union[Config, int], eps: float | None = None):
        super().__init__()

        if isinstance(config_or_dim, Config):
            dim = config_or_dim.model.d_model
            default_eps = config_or_dim.model.norm_eps
        else:
            dim = int(config_or_dim)
            default_eps = 1e-5

        if dim <= 0:
            raise ValueError(f"RMSNorm hidden dimension must be positive, got {dim}.")

        self.dim = dim
        self.eps = float(default_eps if eps is None else eps)

        # Llama RMSNorm has a learned scale/gain but no bias. Initializing the
        # gain to one makes the layer an RMS-only normalization at step zero.
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.dim:
            raise ValueError(f"RMSNorm expected last dimension {self.dim}, got {x.shape[-1]}.")

        # Compute the RMS statistic in fp32 even when activations are fp16/bf16.
        # This avoids avoidable overflow/underflow in the squared mean while
        # preserving the caller-visible dtype at the module boundary.
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = x.float() * torch.rsqrt(variance + self.eps)

        # Cast back before applying the gain so low-precision training keeps the
        # same activation dtype as the surrounding model layers.
        normalized = normalized.to(dtype=x.dtype)
        return normalized * self.weight.to(dtype=x.dtype)
