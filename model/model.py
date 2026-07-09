"""Phase 9 full Llama-compatible decoder-only transformer.

Assembles every Phase 3–8 primitive into a complete end-to-end model:

    input_ids [B, T]
    → TokenEmbedding
    → (optional embed scale)
    → N × TransformerBlock
    → final RMSNorm
    → OutputProjection (tied to embeddings)
    → logits [B, T, vocab_size]
    → cross-entropy loss (when targets are provided)

Attribute naming follows the Llama / HuggingFace convention so that
the Phase 12 ``convert_hf_to_gguf.py`` converter can map state-dict
keys automatically:

    embed_tokens   — TokenEmbedding
    layers         — nn.ModuleList of TransformerBlock
    norm           — final RMSNorm (after last block)
    lm_head        — OutputProjection (weight-tied to embed_tokens)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.config import Config
from model.attention import CausalSelfAttention
from model.block import TransformerBlock
from model.cache import KVCache
from model.embeddings import OutputProjection, TokenEmbedding
from model.norm import RMSNorm


class EdgeGPT(nn.Module):
    """Llama-compatible decoder-only transformer.

    Shape contract:
        input_ids:   ``[B, T]``
        logits:      ``[B, T, vocab_size]``  (``None`` during chunked-loss training)
        loss:        scalar ``Tensor`` or ``None`` (when *targets* is ``None``)

    Returns:
        ``(logits, loss)`` — *logits* is ``None`` when ``chunked_loss`` is
        active and *targets* is provided (the full logits tensor is never
        materialised on that path).
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # ── Phase 3: embeddings ──────────────────────────────────────────
        self.embed_tokens = TokenEmbedding(config)

        # ── Phase 8: transformer blocks ───────────────────────────────────
        self.layers = nn.ModuleList(
            TransformerBlock(config, layer_idx=i) for i in range(config.model.n_layers)
        )

        # ── Phase 5: final norm (after last block, before output proj) ───
        self.norm = RMSNorm(config)

        # ── Phase 3: output projection (weight-tied by default) ───────────
        self.lm_head = OutputProjection(config, self.embed_tokens)

        # Apply the GPT-2 / Llama-style initialisation.
        self.configure_initialization()

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.LongTensor,
        targets: torch.LongTensor | None = None,
        *,
        attention_mask: torch.Tensor | None = None,
        position_offset: int = 0,
        kv_cache: KVCache | None = None,
        cache_position: int = 0,
        use_manual_attention: bool = False,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Run the full model forward pass.

        Args:
            input_ids: ``[B, T]`` token indices.
            targets: Optional ``[B, T]`` target token indices for loss.
            attention_mask: Optional ``[B, T]`` padding mask (1 = keep).
            position_offset: RoPE position offset (0 during training;
                incremented per generation step during inference).
            use_manual_attention: Route through the manual (non-SDPA)
                attention path for test-oracle comparisons.

        Returns:
            ``(logits, loss)`` where *logits* has shape
            ``[B, T, vocab_size]`` (``None`` during chunked-loss training)
            and *loss* is a scalar tensor or ``None``.
        """
        if input_ids.dtype != torch.long:
            raise TypeError(f"EdgeGPT expects torch.long input_ids, got {input_ids.dtype}.")

        # 1. Token embeddings  [B, T] → [B, T, d_model]
        hidden = self.embed_tokens(input_ids)

        # 2. Transformer blocks  [B, T, d_model] → [B, T, d_model]
        if kv_cache is not None and len(kv_cache) != len(self.layers):
            raise ValueError(f"kv_cache has {len(kv_cache)} layers, expected {len(self.layers)}.")
        for layer_idx, block in enumerate(self.layers):
            hidden = block(
                hidden,
                attention_mask=attention_mask,
                position_offset=position_offset,
                layer_cache=kv_cache[layer_idx] if kv_cache is not None else None,
                cache_position=cache_position,
                use_manual_attention=use_manual_attention,
            )

        # 3. Final norm  [B, T, d_model] → [B, T, d_model]
        hidden = self.norm(hidden)

        # 4. Output projection + optional loss
        if targets is not None:
            if self.config.training.chunked_loss:
                # Memory-efficient path: project hidden in chunks so the
                # full [B*T, V] logits tensor is never materialised.
                loss = self._chunked_cross_entropy(hidden, targets)
                return None, loss

            # Standard path: project all hidden states, then compute loss.
            logits = self.lm_head(hidden)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )
            return logits, loss

        # Inference / eval: logits only.
        logits = self.lm_head(hidden)
        return logits, None

    # ------------------------------------------------------------------
    # Loss helpers
    # ------------------------------------------------------------------

    def _chunked_cross_entropy(
        self,
        hidden: torch.Tensor,
        targets: torch.LongTensor,
        chunk_size: int = 1024,
    ) -> torch.Tensor:
        """Tile the output projection + CE over the batch×seq dimension.

        Each chunk projects ``[chunk, d_model] → [chunk, V]``, so peak
        memory is bounded by ``max(chunk_size * V)`` instead of the full
        ``B * T * V`` logits tensor.
        """
        B, T, D = hidden.shape
        flat_hidden = hidden.view(B * T, D)
        flat_targets = targets.view(-1)

        total_loss = torch.tensor(0.0, device=hidden.device, dtype=torch.float32)
        total_tokens = torch.tensor(0, device=hidden.device, dtype=torch.long)
        chunk_size = min(chunk_size, B * T)

        for start in range(0, B * T, chunk_size):
            end = min(start + chunk_size, B * T)
            chunk_hidden = flat_hidden[start:end]
            chunk_targets = flat_targets[start:end]

            # Project and compute loss per-chunk — the full [BT, V] tensor
            # is never allocated.
            chunk_logits = self.lm_head(chunk_hidden).float()
            chunk_loss = F.cross_entropy(
                chunk_logits, chunk_targets, reduction="sum",
            )
            total_loss = total_loss + chunk_loss
            total_tokens = total_tokens + (chunk_targets != -100).sum()

        # Match the denominator behaviour of F.cross_entropy with
        # reduction='mean': divide by the count of non-ignored tokens.
        n_non_ignored = total_tokens.clamp(min=1)
        return total_loss / n_non_ignored

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def configure_initialization(self) -> None:
        """Apply GPT-2 / Llama-style weight initialisation.

        .. list-table::
           :header-rows: 1

           * - Parameter
             - Scheme
           * - ``nn.Linear`` weight
             - ``N(0, initializer_range)``
           * - ``nn.Linear`` bias
             - zeros
           * - ``nn.Embedding`` weight
             - ``N(0, initializer_range)``
           * - Residual projections (``o_proj``, ``down_proj``)
             - ``N(0, initializer_range / √(2 × n_layers))``
           * - ``RMSNorm.weight``
             - ones (set at construction, left untouched)

        Residual projection scaling prevents the residual-stream variance
        from growing with depth.  For the default 8-layer config the
        effective std is ``0.02 / √16 = 0.005``.
        """
        self.apply(self._init_weights)
        self._rescale_residual_projections()

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        init_std = 0.02  # kept as a reasonable default; caller overrides for residuals

        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=init_std)
        # RMSNorm is intentionally skipped — its weight is already ones.

    def _rescale_residual_projections(self) -> None:
        """Re-initialise residual projection weights with depth-scaled std."""
        n_layers = self.config.model.n_layers
        if n_layers <= 0:
            return
        scaled_std = self.config.model.initializer_range / math.sqrt(2 * n_layers)

        for param_name, param in self.named_parameters():
            if param_name.endswith("o_proj.weight") or param_name.endswith("down_proj.weight"):
                nn.init.normal_(param, mean=0.0, std=scaled_std)

    # ------------------------------------------------------------------
    # Introspection utilities
    # ------------------------------------------------------------------

    def count_parameters(self) -> dict[str, int]:
        """Return parameter counts grouped by functional component.

        Returns a dict with keys like ``"embed_tokens"``, ``"layers"``,
        ``"norm"``, ``"lm_head"``, and ``"total"``.
        """
        counts: dict[str, int] = {}
        counts["embed_tokens"] = sum(
            p.numel() for p in self.embed_tokens.parameters()
        )
        counts["layers"] = sum(p.numel() for p in self.layers.parameters())
        counts["norm"] = sum(p.numel() for p in self.norm.parameters())

        if self.config.model.tie_embeddings:
            counts["lm_head"] = 0
        else:
            counts["lm_head"] = sum(
                p.numel() for p in self.lm_head.parameters()
            )

        counts["total"] = sum(p.numel() for p in self.parameters())
        return counts

    # Bytes per element for common training dtypes.
    _DTYPE_BYTES: dict[str, int] = {
        "fp32": 4,
        "fp16": 2,
        "bf16": 2,
    }

    def estimate_memory(
        self,
        batch_size: int,
        seq_len: int,
        *,
        param_dtype: str | None = None,
        act_dtype: str | None = None,
    ) -> dict[str, float]:
        """Estimate parameter + activation memory in MiB.

        Reads dtype sizes from ``config.training.dtype`` by default.
        Override with *param_dtype* / *act_dtype* to model different
        mixed-precision or inference scenarios.

        The activation estimate is a rough upper bound (dominated by
        QKV projections, attention scores, and MLP intermediates).
        """
        train_dtype = self.config.training.dtype
        param_dtype = param_dtype or train_dtype
        act_dtype = act_dtype or param_dtype

        param_bytes = self._DTYPE_BYTES.get(param_dtype, 4)
        act_bytes = self._DTYPE_BYTES.get(act_dtype, 2)

        n_params = sum(p.numel() for p in self.parameters())
        param_mem_mib = n_params * param_bytes / (1024**2)

        d_model = self.config.model.d_model
        n_layers = self.config.model.n_layers

        # Rough per-layer activation: 10 × B × T × d_model elements.
        act_per_layer_bytes = 10 * batch_size * seq_len * d_model * act_bytes
        act_mem_mib = n_layers * act_per_layer_bytes / (1024**2)

        return {
            "param_dtype": param_dtype,
            "act_dtype": act_dtype,
            "param_mem_mib": round(param_mem_mib, 2),
            "act_mem_mib": round(act_mem_mib, 2),
            "total_mib": round(param_mem_mib + act_mem_mib, 2),
        }

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        """Return the device of the first parameter."""
        return next(self.parameters()).device
