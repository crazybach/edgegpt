"""Configuration system — single source of truth for all hyperparameters."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class ModelConfig:
    """Architecture hyperparameters (Llama-compatible)."""

    vocab_size: int = 16384
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 4
    d_ff: int = 1408  # ~8/3 * d_model
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    initializer_range: float = 0.02  # GPT-2/Llama weight init std
    embedding_scale: Optional[float] = None  # None = no scaling (Llama-style)
    tie_embeddings: bool = True
    dropout: float = 0.0


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 3.0e-4
    min_lr: float = 3.0e-5
    warmup_steps: int = 500
    max_steps: int = 50000
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    dtype: str = "bf16"
    chunked_loss: bool = False  # tile CE over seq dim to save memory on CPU
    eval_every: int = 500
    save_every: int = 2000
    log_every: int = 10
    output_dir: str = "./artifacts/runs"
    eval_iters: int = 20
    checkpoint_keep_last: int = 3
    always_save_checkpoint: bool = True


@dataclass
class DataConfig:
    """Data pipeline configuration."""

    dataset: str = "tinystories"
    data_dir: str = "./data/tinystories"
    val_split: float = 0.005
    seed: int = 42
    source_type: str = "local_text"
    source_paths: list[str] = field(default_factory=list)
    text_column: str = "text"
    cache_dir: str = "./artifacts/data/tinystories"
    storage_type: str = "memmap_bin"
    block_size: Optional[int] = None
    shuffle_buffer_size: int = 10000
    num_workers: int = 0


@dataclass
class TokenizerConfig:
    """Tokenizer hyperparameters.

    The tokenizer is part of the model contract: the embedding table has one
    row per tokenizer ID. If the tokenizer vocab changes after training starts,
    old checkpoints no longer line up with token IDs.
    """

    type: str = "byte_bpe"
    vocab_size: int = 16384
    reserved_special_tokens: int = 256
    artifact_dir: str = "./artifacts/tokenizer/main_16k"
    train_files: list[str] = field(default_factory=lambda: ["./data/tinystories/train.txt"])
    min_frequency: int = 2
    add_prefix_space: bool = False
    split_digits: bool = True


@dataclass
class Config:
    """Top-level configuration aggregating all sub-configs."""

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    device: str = "auto"

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate cross-section invariants that would break training later."""
        if self.model.vocab_size != self.tokenizer.vocab_size:
            raise ValueError(
                "model.vocab_size must match tokenizer.vocab_size "
                f"({self.model.vocab_size} != {self.tokenizer.vocab_size})."
            )
        if self.tokenizer.reserved_special_tokens <= 0:
            raise ValueError("tokenizer.reserved_special_tokens must be positive.")
        if self.tokenizer.reserved_special_tokens >= self.tokenizer.vocab_size:
            raise ValueError("tokenizer.reserved_special_tokens must be smaller than tokenizer.vocab_size.")
        if self.data.val_split < 0 or self.data.val_split >= 1:
            raise ValueError("data.val_split must be in the range [0, 1).")
        if self.data.block_size is not None and self.data.block_size <= 0:
            raise ValueError("data.block_size must be positive when set.")
        if self.model.d_ff <= 0:
            raise ValueError("model.d_ff must be positive.")
        if self.model.n_heads <= 0:
            raise ValueError("model.n_heads must be positive.")
        if self.model.d_model % self.model.n_heads != 0:
            raise ValueError("model.d_model must be divisible by model.n_heads.")
        if self.model.n_kv_heads <= 0:
            raise ValueError("model.n_kv_heads must be positive.")
        if self.model.n_kv_heads > self.model.n_heads:
            raise ValueError("model.n_kv_heads cannot exceed model.n_heads.")
        if self.model.n_heads % self.model.n_kv_heads != 0:
            raise ValueError("model.n_heads must be divisible by model.n_kv_heads.")
        if self.training.batch_size <= 0:
            raise ValueError("training.batch_size must be positive.")
        if self.training.gradient_accumulation_steps <= 0:
            raise ValueError("training.gradient_accumulation_steps must be positive.")
        if self.training.learning_rate <= 0:
            raise ValueError("training.learning_rate must be positive.")
        if self.training.min_lr < 0:
            raise ValueError("training.min_lr must be non-negative.")
        if self.training.warmup_steps < 0:
            raise ValueError("training.warmup_steps must be non-negative.")
        if self.training.max_steps <= 0:
            raise ValueError("training.max_steps must be positive.")
        if self.training.eval_every <= 0:
            raise ValueError("training.eval_every must be positive.")
        if self.training.save_every <= 0:
            raise ValueError("training.save_every must be positive.")
        if self.training.log_every <= 0:
            raise ValueError("training.log_every must be positive.")
        if self.training.eval_iters <= 0:
            raise ValueError("training.eval_iters must be positive.")
        if self.training.checkpoint_keep_last <= 0:
            raise ValueError("training.checkpoint_keep_last must be positive.")
        if self.training.dtype not in {"fp32", "bf16", "fp16"}:
            raise ValueError("training.dtype must be one of: fp32, bf16, fp16.")

    @property
    def head_dim(self) -> int:
        """Dimension per attention head."""
        return self.model.d_model // self.model.n_heads

    @property
    def kv_head_dim(self) -> int:
        """Dimension per KV head (same as query head_dim)."""
        return self.model.d_model // self.model.n_heads

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load config from a YAML file, falling back to defaults for missing keys."""
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        cfg = cls()
        if "model" in raw:
            cfg.model = ModelConfig(**{k: v for k, v in raw["model"].items() if hasattr(ModelConfig, k)})
        if "training" in raw:
            cfg.training = TrainingConfig(**{k: v for k, v in raw["training"].items() if hasattr(TrainingConfig, k)})
        if "data" in raw:
            cfg.data = DataConfig(**{k: v for k, v in raw["data"].items() if hasattr(DataConfig, k)})
        if "tokenizer" in raw:
            cfg.tokenizer = TokenizerConfig(
                **{k: v for k, v in raw["tokenizer"].items() if hasattr(TokenizerConfig, k)}
            )
        if "device" in raw:
            cfg.device = raw["device"]
        cfg.validate()
        return cfg

    def resolve_device(self) -> str:
        """Resolve 'auto' device to the best available: cuda > mps > cpu."""
        import torch

        if self.device != "auto":
            return self.device
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"


def load_config(path: str | Path = "configs/default.yaml") -> Config:
    """Load config; falls back to defaults if no YAML present."""
    if os.path.exists(path):
        return Config.from_yaml(path)
    return Config()


