"""Phase 3 token embeddings and tied output projection.

Llama-style decoder models do not use a learned absolute position table here;
position information is added later by RoPE inside attention. This module only
maps token IDs to vectors and maps hidden states back to vocabulary logits.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.config import Config


class TokenEmbedding(nn.Module):
    """Convert token IDs `[B, T]` into dense vectors `[B, T, d_model]`."""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.model.vocab_size, config.model.d_model)
        self.scale = config.model.embedding_scale

    @property
    def weight(self) -> nn.Parameter:
        """Expose weights for output-projection tying."""

        return self.embedding.weight

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        if input_ids.dtype != torch.long:
            raise TypeError(f"TokenEmbedding expects torch.long input_ids, got {input_ids.dtype}.")

        hidden = self.embedding(input_ids)
        # Llama leaves token embeddings unscaled. Gemma-style variants can set
        # a numeric scale in config without changing the module contract.
        if self.scale is not None:
            hidden = hidden * float(self.scale)
        return hidden


class OutputProjection(nn.Module):
    """Project hidden states back to vocabulary logits.

    When weights are tied, this module reuses the exact input embedding matrix.
    That saves `vocab_size * d_model` parameters and keeps the input/output
    token spaces aligned, which is a strong default for this small baseline.
    """

    def __init__(self, config: Config, token_embedding: TokenEmbedding | None = None):
        super().__init__()
        self.config = config
        self.tie_embeddings = config.model.tie_embeddings
        self.token_embedding = token_embedding

        if self.tie_embeddings:
            if token_embedding is None:
                raise ValueError("A TokenEmbedding instance is required when tie_embeddings=True.")
            self.proj = None
        else:
            self.proj = nn.Linear(config.model.d_model, config.model.vocab_size, bias=False)

    @property
    def weight(self) -> nn.Parameter:
        if self.tie_embeddings:
            assert self.token_embedding is not None
            return self.token_embedding.weight
        assert self.proj is not None
        return self.proj.weight

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.tie_embeddings:
            return F.linear(hidden, self.weight)
        assert self.proj is not None
        return self.proj(hidden)


def build_embedding_layers(config: Config) -> tuple[TokenEmbedding, OutputProjection]:
    """Build the Phase 3 embedding stack from config."""

    token_embedding = TokenEmbedding(config)
    output_projection = OutputProjection(config, token_embedding)
    return token_embedding, output_projection
