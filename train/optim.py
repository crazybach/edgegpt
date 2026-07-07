"""Optimizer construction for Phase 10 training.

The first EdgeGPT training loop uses AdamW because it is the mature default for
small decoder-only language-model pretraining. The important detail is the
parameter grouping: decoupled weight decay should regularize large matrix
weights, but it should not shrink normalization gains or embedding rows that
act as learned token identities.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from configs.config import Config


def _module_lookup(model: nn.Module) -> dict[str, nn.Module]:
    """Return a map from module path to module instance for parameter owners."""

    return dict(model.named_modules())


def _owning_module_name(param_name: str) -> str:
    """Return the module path before the final parameter component."""

    if "." not in param_name:
        return ""
    return param_name.rsplit(".", 1)[0]


def should_apply_weight_decay(param_name: str, param: nn.Parameter, modules: dict[str, nn.Module]) -> bool:
    """Decide whether AdamW weight decay should touch one parameter.

    Matrix-shaped linear weights benefit from decay. One-dimensional gains,
    biases, RMSNorm weights, and embeddings are excluded because decaying them
    can destabilize scale-sensitive parts of a transformer and is not the
    common Llama/nanoGPT-style optimizer policy.
    """

    if not param.requires_grad:
        return False
    if param.ndim < 2:
        return False
    owner = modules.get(_owning_module_name(param_name))
    if isinstance(owner, nn.Embedding):
        return False
    return True


def build_weight_decay_groups(model: nn.Module, weight_decay: float) -> list[dict[str, object]]:
    """Split trainable parameters into decayed and non-decayed AdamW groups."""

    modules = _module_lookup(model)
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    decay_names: list[str] = []
    no_decay_names: list[str] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if should_apply_weight_decay(name, param, modules):
            decay.append(param)
            decay_names.append(name)
        else:
            no_decay.append(param)
            no_decay_names.append(name)

    # Store names in the param group for debugging and tests. PyTorch ignores
    # unknown optimizer-group keys, but keeping names makes the training report
    # explainable when we audit which tensors are regularized.
    return [
        {"params": decay, "weight_decay": weight_decay, "param_names": decay_names},
        {"params": no_decay, "weight_decay": 0.0, "param_names": no_decay_names},
    ]


def build_adamw_optimizer(model: nn.Module, config: Config) -> torch.optim.AdamW:
    """Construct the Phase 10 AdamW optimizer from config values."""

    groups = build_weight_decay_groups(model, config.training.weight_decay)
    return torch.optim.AdamW(
        groups,
        lr=config.training.learning_rate,
        betas=(config.training.beta1, config.training.beta2),
    )
