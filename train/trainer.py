"""Single-node Phase 10 trainer for EdgeGPT.

Checkpoint behaviour is controlled by ``training.checkpoint_enabled``:

* ``True`` (default) — save numbered checkpoints at ``save_every``,
  on new best validation loss, and on graceful interrupt (Ctrl+C).
  Resume picks up the data-loader shuffle position so no tokens are
  re-consumed.
* ``False`` — no checkpoint files are written.  Training still logs
  to JSONL, but runs cannot be resumed.
"""

from __future__ import annotations

import json
import math
import signal
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch

from configs.config import Config
from data.pipeline import build_train_loader
from model import EdgeGPT
from train.checkpoint import (
    capture_dataloader_state,
    load_checkpoint,
    restore_dataloader_state,
    save_checkpoint,
    write_checkpoint_report,
)
from train.logging import JsonlEventLogger
from train.optim import build_adamw_optimizer
from train.schedule import get_warmup_cosine_lr


@dataclass
class TrainingResult:
    """Summary returned after a training or eval-only run."""

    run_dir: Path
    global_step: int
    best_val_loss: float | None
    last_train_loss: float | None
    tokens_consumed: int


class Trainer:
    """Own the Phase 10 training loop around the Phase 9 ``EdgeGPT`` model."""

    def __init__(
        self,
        config: Config,
        *,
        run_dir: str | Path | None = None,
        run_id: str | None = None,
        model: EdgeGPT | None = None,
    ):
        self.config = config
        self.device = torch.device(config.resolve_device())
        self.run_dir = Path(run_dir) if run_dir is not None else self._default_run_dir(run_id)
        self.run_id = run_id or self.run_dir.name
        self.logger = JsonlEventLogger(self.run_dir, self.run_id)
        self.model = model if model is not None else EdgeGPT(config)
        self.model.to(self.device)
        self.optimizer = build_adamw_optimizer(self.model, config)
        self.scaler = self._build_scaler()
        self.global_step = 0
        self.best_val_loss: float | None = None
        self.last_train_loss: float | None = None
        self.tokens_consumed: int = 0
        self.dataset_total_tokens: int | None = None
        self._run_start_s: float | None = None
        self._active_max_steps = config.training.max_steps
        self._train_iter: Iterable[dict[str, torch.Tensor]] | None = None
        self._train_loader = None
        self._val_loader = None
        self._interrupted = False

    # ── helpers ───────────────────────────────────────────────────────

    def _default_run_dir(self, run_id: str | None) -> Path:
        suffix = run_id or time.strftime("%Y%m%d_%H%M%S")
        return Path(self.config.training.output_dir) / suffix

    def _build_scaler(self) -> Any:
        """Use GradScaler only for CUDA fp16 training."""

        enabled = self.device.type == "cuda" and self.config.training.dtype == "fp16"
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            try:
                return torch.amp.GradScaler("cuda", enabled=enabled)
            except TypeError:
                pass
        return torch.cuda.amp.GradScaler(enabled=enabled)

    def _autocast_context(self):
        """Return the mixed-precision context appropriate for the device."""

        if self.device.type != "cuda":
            return nullcontext()
        if self.config.training.dtype == "bf16":
            return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if self.config.training.dtype == "fp16":
            return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

    def _memory_report(self) -> dict[str, float] | None:
        """Return lightweight memory info for logs when CUDA is available."""

        if self.device.type != "cuda":
            return None
        return {
            "allocated_mib": round(torch.cuda.memory_allocated(self.device) / (1024**2), 2),
            "reserved_mib": round(torch.cuda.memory_reserved(self.device) / (1024**2), 2),
        }

    def _current_lr(self, step: int) -> float:
        return get_warmup_cosine_lr(
            step,
            learning_rate=self.config.training.learning_rate,
            min_lr=self.config.training.min_lr,
            warmup_steps=self.config.training.warmup_steps,
            max_steps=self.config.training.max_steps,
        )

    def _set_lr(self, lr: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def _ensure_train_loader(self) -> None:
        """Lazily build the train DataLoader and discover dataset size."""
        if self._train_loader is not None:
            return
        self._train_loader = build_train_loader(self.config, "train")
        self._train_iter = iter(self._train_loader)
        # Cache total dataset size for progress reporting.
        try:
            dataset = self._train_loader.dataset
            tokens = getattr(dataset, "tokens", None)
            self.dataset_total_tokens = int(tokens.size) if tokens is not None else len(dataset) + self.config.data.block_size
        except Exception:
            pass

    def _next_train_batch(self) -> dict[str, torch.Tensor]:
        self._ensure_train_loader()
        assert self._train_iter is not None and self._train_loader is not None
        try:
            return next(self._train_iter)
        except StopIteration:
            self._train_iter = iter(self._train_loader)
            return next(self._train_iter)

    def _to_device(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {key: value.to(self.device, non_blocking=self.device.type == "cuda") for key, value in batch.items()}

    def _log_common(
        self,
        *,
        global_step: int,
        batch_size: int | None = None,
        seq_len: int | None = None,
    ) -> dict[str, Any]:
        progress = min(global_step / float(self._active_max_steps), 1.0)
        tokens_seen = None
        if batch_size is not None and seq_len is not None:
            tokens_seen = global_step * batch_size * seq_len * self.config.training.gradient_accumulation_steps
        return {
            "global_step": global_step,
            "max_steps": self._active_max_steps,
            "progress": progress,
            "tokens_seen": tokens_seen,
            "tokens_consumed": self.tokens_consumed,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "device": str(self.device),
            "dtype": self.config.training.dtype,
            "memory": self._memory_report(),
        }

    def _elapsed(self) -> float:
        if self._run_start_s is None:
            return 0.0
        return time.perf_counter() - self._run_start_s

    # ── training entry-point ───────────────────────────────────────────

    def train(
        self,
        *,
        max_steps: int | None = None,
        resume: str | Path | None = None,
        eval_only: bool = False,
    ) -> TrainingResult:
        """Run training until ``max_steps`` or ``config.training.max_steps``.

        When *resume* is provided the model, optimizer, and RNG state are
        restored from the checkpoint file and the data-loader shuffle
        position is re-seeded so training continues without re-consuming
        previously seen tokens.
        """

        target_steps = max_steps if max_steps is not None else self.config.training.max_steps
        if target_steps <= 0:
            raise ValueError("max_steps must be positive.")
        self._active_max_steps = target_steps
        self._run_start_s = time.perf_counter()

        if resume is not None:
            state = load_checkpoint(
                path=resume,
                model=self.model,
                optimizer=self.optimizer,
                scaler=self.scaler,
                map_location=self.device,
            )
            self.global_step = state.global_step
            self.best_val_loss = state.best_val_loss
            self.tokens_consumed = state.tokens_consumed
            # Rebuild the data loader and re-seed its sampler so we
            # continue from where the shuffle left off.
            self._train_loader = build_train_loader(self.config, "train")
            restorable = state.tokens_consumed > 0
            if restorable:
                restore_dataloader_state(self._train_loader, state.dataloader_state)
            self._train_iter = iter(self._train_loader)
            # Discover dataset size even on resume.
            try:
                dataset = self._train_loader.dataset
                tokens = getattr(dataset, "tokens", None)
                self.dataset_total_tokens = (
                    int(tokens.size) if tokens is not None else len(dataset) + self.config.data.block_size
                )
            except Exception:
                pass
            checkpoint_name = Path(resume).name
            self.logger.log(
                "resume",
                checkpoint=checkpoint_name,
                best_val_loss=self.best_val_loss,
                **self._log_common(global_step=self.global_step),
            )
            print(
                f"Loaded checkpoint: step={self.global_step} "
                f"tokens_consumed={self.tokens_consumed} "
                f"best_val_loss={self.best_val_loss}"
            )

        self._install_sigint_handler()
        self.logger.log("run_start", **self._log_common(global_step=self.global_step))
        try:
            if eval_only:
                self.evaluate()
                self.logger.log("run_end", **self._log_common(global_step=self.global_step))
                return TrainingResult(
                    self.run_dir, self.global_step, self.best_val_loss,
                    self.last_train_loss, self.tokens_consumed,
                )

            while self.global_step < target_steps:
                if self._interrupted:
                    self._save_checkpoint(kind="interrupt")
                    print(
                        f"\nInterrupted at step {self.global_step}. "
                        f"Checkpoint saved. Resume with: --resume {self.run_dir / 'latest.pt'}"
                    )
                    break
                self._train_one_optimizer_step(target_steps=target_steps)

            if not self._interrupted:
                self.logger.log("run_end", **self._log_common(global_step=self.global_step))
            return TrainingResult(
                self.run_dir, self.global_step, self.best_val_loss,
                self.last_train_loss, self.tokens_consumed,
            )
        except Exception as exc:
            self.logger.log(
                "error",
                global_step=self.global_step,
                max_steps=target_steps,
                progress=0.0,
                loss=None,
                message=str(exc),
            )
            raise

    # ── signal handling ────────────────────────────────────────────────

    def _install_sigint_handler(self) -> None:
        """Catch Ctrl+C once so checkpoints can save a clean snapshot."""

        def _handler(signum: int, frame: Any) -> None:  # noqa: ARG001
            self._interrupted = True

        try:
            signal.signal(signal.SIGINT, _handler)
        except ValueError:
            pass  # not in main thread — skip handler

    # ── single optimizer step ──────────────────────────────────────────

    def _train_one_optimizer_step(self, *, target_steps: int) -> None:
        """Run one accumulated optimizer step and emit structured events."""

        step_start = time.perf_counter()
        lr = self._current_lr(self.global_step)
        self._set_lr(lr)
        self.optimizer.zero_grad(set_to_none=True)
        self.model.train()

        self.logger.log("step_start", lr=lr, **self._log_common(global_step=self.global_step))
        total_loss = 0.0
        last_batch_size = None
        last_seq_len = None

        for micro_step in range(self.config.training.gradient_accumulation_steps):
            batch = self._to_device(self._next_train_batch())
            input_ids = batch["input_ids"]
            targets = batch["targets"]
            last_batch_size, last_seq_len = int(input_ids.shape[0]), int(input_ids.shape[1])
            self.tokens_consumed += int(input_ids.numel())

            with self._autocast_context():
                _, loss = self.model(input_ids, targets)
                if loss is None:
                    raise RuntimeError("Model did not return a training loss.")
                scaled_loss = loss / self.config.training.gradient_accumulation_steps

            self.scaler.scale(scaled_loss).backward()
            total_loss += float(loss.detach().cpu().item())
            self.logger.log(
                "micro_step",
                micro_step=micro_step,
                split="train",
                loss=float(loss.detach().cpu().item()),
                perplexity=self._safe_perplexity(float(loss.detach().cpu().item())),
                lr=lr,
                **self._log_common(global_step=self.global_step, batch_size=last_batch_size, seq_len=last_seq_len),
            )

        grad_norm = None
        if self.config.training.grad_clip > 0:
            self.scaler.unscale_(self.optimizer)
            grad_norm_tensor = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.grad_clip)
            grad_norm = float(grad_norm_tensor.detach().cpu().item())

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        self.global_step += 1

        elapsed = max(time.perf_counter() - step_start, 1e-12)
        tokens_this_step = (last_batch_size or 0) * (last_seq_len or 0) * self.config.training.gradient_accumulation_steps
        self.last_train_loss = total_loss / self.config.training.gradient_accumulation_steps
        should_log = self.global_step % self.config.training.log_every == 0 or self.global_step == 1
        if should_log:
            self.logger.log(
                "optimizer_step",
                split="train",
                loss=self.last_train_loss,
                perplexity=self._safe_perplexity(self.last_train_loss),
                lr=lr,
                grad_norm=grad_norm,
                tokens_per_sec=tokens_this_step / elapsed,
                **self._log_common(global_step=self.global_step, batch_size=last_batch_size, seq_len=last_seq_len),
            )

        if self.global_step % self.config.training.eval_every == 0 or self.global_step == target_steps:
            self.evaluate()
        self._maybe_save_checkpoint(target_steps=target_steps)

    # ── checkpoint save gating ─────────────────────────────────────────

    def _maybe_save_checkpoint(self, *, target_steps: int) -> None:
        """Save a checkpoint if the policy allows it and checkpointing is on."""

        if not self.config.training.checkpoint_enabled:
            return

        should_save_interval = self.global_step % self.config.training.save_every == 0
        should_save_final = (
            self.config.training.always_save_checkpoint and self.global_step == target_steps
        )
        if should_save_interval or should_save_final:
            self._save_checkpoint(kind="interval")

    def _save_checkpoint(self, *, kind: str = "interval") -> Path | None:
        """Save a checkpoint, its JSON report, and log the event."""

        if not self.config.training.checkpoint_enabled:
            return None

        dl_state = capture_dataloader_state(self._train_loader)
        step_path = save_checkpoint(
            run_dir=self.run_dir,
            config=self.config,
            model=self.model,
            optimizer=self.optimizer,
            scaler=self.scaler,
            global_step=self.global_step,
            best_val_loss=self.best_val_loss,
            tokens_consumed=self.tokens_consumed,
            dataloader_state=dl_state,
        )

        write_checkpoint_report(
            checkpoint_path=step_path,
            global_step=self.global_step,
            max_steps=self._active_max_steps,
            tokens_consumed=self.tokens_consumed,
            total_dataset_tokens=self.dataset_total_tokens,
            train_loss=self.last_train_loss,
            val_loss=self.best_val_loss,
            lr=self._current_lr(self.global_step),
            elapsed_s=self._elapsed(),
        )

        self.logger.log(
            "checkpoint",
            kind=kind,
            checkpoint_path=str(step_path),
            **self._log_common(global_step=self.global_step),
        )
        return step_path

    # ── evaluation ─────────────────────────────────────────────────────

    def evaluate(self) -> dict[str, float] | None:
        """Evaluate validation loss when a val shard is available."""

        try:
            if self._val_loader is None:
                self._val_loader = build_train_loader(self.config, "val")
        except (FileNotFoundError, ValueError):
            return None

        was_training = self.model.training
        self.model.eval()
        losses: list[float] = []
        with torch.no_grad():
            val_iter = iter(self._val_loader)
            for _ in range(self.config.training.eval_iters):
                try:
                    batch = next(val_iter)
                except StopIteration:
                    break
                batch = self._to_device(batch)
                with self._autocast_context():
                    _, loss = self.model(batch["input_ids"], batch["targets"])
                if loss is not None:
                    losses.append(float(loss.detach().cpu().item()))

        if was_training:
            self.model.train()
        if not losses:
            return None

        val_loss = sum(losses) / len(losses)
        if self.best_val_loss is None or val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            if self.config.training.checkpoint_enabled:
                self._save_checkpoint(kind="best")
        self.logger.log(
            "eval",
            split="val",
            loss=val_loss,
            perplexity=self._safe_perplexity(val_loss),
            **self._log_common(global_step=self.global_step),
        )
        return {"loss": val_loss, "perplexity": self._safe_perplexity(val_loss)}

    # ── utilities ──────────────────────────────────────────────────────

    @staticmethod
    def _safe_perplexity(loss: float | None) -> float | None:
        if loss is None:
            return None
        if loss > 80:
            return float("inf")
        return math.exp(loss)
