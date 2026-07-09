"""External KV-cache containers for autoregressive inference."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from configs.config import Config


@dataclass
class LayerKVCache:
    """Preallocated key/value cache for one transformer layer."""

    k: torch.Tensor
    v: torch.Tensor

    @property
    def max_seq_len(self) -> int:
        return int(self.k.shape[2])

    def append(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        cache_position: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Write new K/V states and return the visible cache prefix."""

        if k.shape != v.shape:
            raise ValueError(f"k and v must share shape, got {k.shape} and {v.shape}.")
        if k.ndim != 4:
            raise ValueError(f"k/v must have shape [B, H_kv, T, D], got {k.shape}.")
        if k.shape[0] != self.k.shape[0] or k.shape[1] != self.k.shape[1] or k.shape[3] != self.k.shape[3]:
            raise ValueError(f"k/v shape {k.shape} is incompatible with cache shape {self.k.shape}.")
        if k.device != self.k.device or v.device != self.v.device:
            raise ValueError("k/v tensors must be on the same device as the cache.")

        start = int(cache_position)
        end = start + int(k.shape[2])
        if start < 0 or end > self.max_seq_len:
            raise ValueError(f"cache write [{start}, {end}) exceeds max_seq_len={self.max_seq_len}.")

        self.k[:, :, start:end, :] = k.to(dtype=self.k.dtype)
        self.v[:, :, start:end, :] = v.to(dtype=self.v.dtype)
        return self.k[:, :, :end, :], self.v[:, :, :end, :]


@dataclass
class KVCache:
    """External per-layer cache used by generation code."""

    layers: list[LayerKVCache]

    def __len__(self) -> int:
        return len(self.layers)

    def __getitem__(self, index: int) -> LayerKVCache:
        return self.layers[index]


def build_kv_cache(
    config: Config,
    *,
    batch_size: int,
    max_seq_len: int | None = None,
    device: torch.device | str,
    dtype: torch.dtype,
) -> KVCache:
    """Allocate an empty KV cache for a model/config."""

    max_seq_len = int(max_seq_len or config.model.max_seq_len)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if max_seq_len <= 0:
        raise ValueError("max_seq_len must be positive.")

    shape = (
        int(batch_size),
        int(config.model.n_kv_heads),
        max_seq_len,
        int(config.model.d_model // config.model.n_heads),
    )
    layers = [
        LayerKVCache(
            k=torch.zeros(shape, device=device, dtype=dtype),
            v=torch.zeros(shape, device=device, dtype=dtype),
        )
        for _ in range(int(config.model.n_layers))
    ]
    return KVCache(layers)