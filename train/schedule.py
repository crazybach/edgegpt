"""Learning-rate schedules for Phase 10."""

from __future__ import annotations

import math


def get_warmup_cosine_lr(
    step: int,
    *,
    learning_rate: float,
    min_lr: float,
    warmup_steps: int,
    max_steps: int,
) -> float:
    """Return the AdamW learning rate for one optimizer step.

    This mirrors the mature nanoGPT-style schedule: start with small updates
    during warmup, then decay smoothly to a floor. Keeping the function pure
    makes resume behavior easy to reason about: the LR is determined only by
    the integer global step and config.
    """

    if step < 0:
        raise ValueError("step must be non-negative.")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative.")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive.")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")
    if min_lr < 0:
        raise ValueError("min_lr must be non-negative.")

    if warmup_steps > 0 and step < warmup_steps:
        return learning_rate * float(step + 1) / float(warmup_steps)
    if step >= max_steps:
        return min_lr
    if max_steps <= warmup_steps:
        return min_lr

    decay_ratio = (step - warmup_steps) / float(max_steps - warmup_steps)
    decay_ratio = min(max(decay_ratio, 0.0), 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)
