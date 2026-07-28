"""Checkpoint save/load helpers for resumable Phase 10 training."""

from __future__ import annotations

import json
import random
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from configs.config import Config


@dataclass
class CheckpointState:
    """Resume summary returned by ``load_checkpoint``."""

    global_step: int
    best_val_loss: float | None
    tokens_consumed: int
    dataloader_state: dict[str, Any]
    path: Path


# ── RNG state ──────────────────────────────────────────────────────────


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
        torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([rng.cpu() for rng in state["cuda"]])


# ── DataLoader state ────────────────────────────────────────────────────


def capture_dataloader_state(train_loader: Any) -> dict[str, Any]:
    """Capture the DataLoader's RandomSampler generator state.

    This lets resume continue from the same point in the shuffled dataset
    instead of restarting from the beginning.
    """
    if train_loader is None:
        return {}
    sampler = getattr(train_loader, "sampler", None)
    if sampler is None:
        return {}
    state_dict = getattr(sampler, "state_dict", None)
    if callable(state_dict):
        return {"sampler_state": state_dict()}
    generator = getattr(sampler, "generator", None)
    if generator is not None:
        return {"sampler_generator": generator.get_state()}
    return {}


def restore_dataloader_state(train_loader: Any, state: dict[str, Any]) -> None:
    """Re-seed the DataLoader's RandomSampler to continue a shuffle in-place."""

    if not state or train_loader is None:
        return
    sampler = getattr(train_loader, "sampler", None)
    if sampler is None:
        return
    load_state_dict = getattr(sampler, "load_state_dict", None)
    if callable(load_state_dict) and "sampler_state" in state:
        load_state_dict(state["sampler_state"])
        return
    generator = getattr(sampler, "generator", None)
    if generator is not None and "sampler_generator" in state:
        generator.set_state(state["sampler_generator"].cpu())


# ── Checkpoint payload ──────────────────────────────────────────────────


def _checkpoint_payload(
    *,
    config: Config,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    global_step: int,
    best_val_loss: float | None,
    tokens_consumed: int,
    dataloader_state: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build the serializable checkpoint dictionary."""

    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "tokens_consumed": tokens_consumed,
        "dataloader_state": dataloader_state or {},
        "config": asdict(config),
        "rng_state": capture_rng_state(),
    }


# ── Save ────────────────────────────────────────────────────────────────


def save_checkpoint(
    *,
    run_dir: str | Path,
    config: Config,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    global_step: int,
    best_val_loss: float | None,
    tokens_consumed: int = 0,
    dataloader_state: dict[str, Any] | None = None,
) -> Path:
    """Save ``latest.pt`` and a numbered step checkpoint.

    ``latest.pt`` is the stable resume target.  Numbered checkpoints
    support quick rollback and are pruned according to
    ``checkpoint_keep_last``.

    A companion ``step_{N}.json`` report is written alongside the ``.pt``
    file so users can inspect checkpoint contents without loading into
    PyTorch.
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
        tokens_consumed=tokens_consumed,
        dataloader_state=dataloader_state,
    )

    step_path = run_path / f"step_{global_step}.pt"
    latest_path = run_path / "latest.pt"
    torch.save(payload, step_path)
    shutil.copyfile(step_path, latest_path)
    prune_checkpoints(run_path, keep_last=config.training.checkpoint_keep_last)
    return step_path


def write_checkpoint_report(
    *,
    checkpoint_path: Path,
    global_step: int,
    max_steps: int,
    tokens_consumed: int,
    total_dataset_tokens: int | None,
    train_loss: float | None,
    val_loss: float | None,
    lr: float,
    elapsed_s: float,
) -> Path:
    """Write a human-readable JSON summary alongside the ``.pt`` file."""

    report_path = checkpoint_path.with_suffix(".json")
    dataset_pct = None
    if total_dataset_tokens and total_dataset_tokens > 0:
        dataset_pct = round(tokens_consumed / total_dataset_tokens * 100, 1)

    report: dict[str, Any] = {
        "global_step": global_step,
        "max_steps": max_steps,
        "progress_pct": round(global_step / max_steps * 100, 1) if max_steps > 0 else None,
        "tokens_consumed": tokens_consumed,
        "dataset_progress_pct": dataset_pct,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "perplexity": _safe_perplexity(train_loss),
        "lr": lr,
        "elapsed_s": round(elapsed_s, 1),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Also write a latest.json copy for quick inspection.
    latest_report = checkpoint_path.parent / "latest.json"
    shutil.copyfile(report_path, latest_report)
    return report_path


# ── Load ────────────────────────────────────────────────────────────────


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
        tokens_consumed=int(checkpoint.get("tokens_consumed", 0)),
        dataloader_state=checkpoint.get("dataloader_state", {}),
        path=ckpt_path,
    )


def _safe_perplexity(loss: float | None) -> float | None:
    if loss is None:
        return None
    if loss > 80:
        return None  # inf is not valid JSON
    return round(float(np.exp(loss)), 2)


# ── Prune ───────────────────────────────────────────────────────────────


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
        # Clean up the companion report if it exists.
        report = old_path.with_suffix(".json")
        report.unlink(missing_ok=True)
