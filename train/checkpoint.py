"""Checkpoint save/load helpers for resumable Phase 10 training."""

from __future__ import annotations

import random
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from configs.config import Config


@dataclass
class CheckpointState:
    """Small resume summary returned by ``load_checkpoint``."""

    global_step: int
    best_val_loss: float | None
    path: Path


def capture_rng_state() -> dict[str, Any]:
    """Capture RNG states needed to continue a run deterministically."""

    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any] | None) -> None:
    """Restore RNG states when present in a checkpoint."""

    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _checkpoint_payload(
    *,
    config: Config,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    global_step: int,
    best_val_loss: float | None,
) -> dict[str, Any]:
    """Build the serializable checkpoint dictionary."""

    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "config": asdict(config),
        "rng_state": capture_rng_state(),
    }


def save_checkpoint(
    *,
    run_dir: str | Path,
    config: Config,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    global_step: int,
    best_val_loss: float | None,
) -> Path:
    """Save ``latest.pt`` and a numbered step checkpoint.

    ``latest.pt`` is the stable resume target. Numbered checkpoints support
    quick rollback and are pruned according to ``checkpoint_keep_last``.
    """

    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    payload = _checkpoint_payload(
        config=config,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        global_step=global_step,
        best_val_loss=best_val_loss,
    )

    step_path = run_path / f"step_{global_step}.pt"
    latest_path = run_path / "latest.pt"
    torch.save(payload, step_path)
    shutil.copyfile(step_path, latest_path)
    prune_checkpoints(run_path, keep_last=config.training.checkpoint_keep_last)
    return latest_path


def prune_checkpoints(run_dir: str | Path, keep_last: int) -> None:
    """Keep only the newest numbered step checkpoints."""

    if keep_last <= 0:
        return
    run_path = Path(run_dir)
    checkpoints = sorted(
        run_path.glob("step_*.pt"),
        key=lambda p: int(p.stem.split("_", 1)[1]),
    )
    for old_path in checkpoints[:-keep_last]:
        old_path.unlink(missing_ok=True)


def load_checkpoint(
    *,
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: Any = None,
    map_location: str | torch.device = "cpu",
    restore_rng: bool = True,
) -> CheckpointState:
    """Load a checkpoint into model/optimizer/scaler and return resume state."""

    ckpt_path = Path(path)
    checkpoint = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    if restore_rng:
        restore_rng_state(checkpoint.get("rng_state"))
    return CheckpointState(
        global_step=int(checkpoint.get("global_step", 0)),
        best_val_loss=checkpoint.get("best_val_loss"),
        path=ckpt_path,
    )
