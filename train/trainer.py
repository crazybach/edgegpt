"""Single-node Phase 10 trainer for EdgeGPT."""

from __future__ import annotations

import math
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch

from configs.config import Config
from data.pipeline import build_train_loader
from model import EdgeGPT
from train.checkpoint import load_checkpoint, save_checkpoint
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
        self._active_max_steps = config.training.max_steps
        self._train_iter: Iterable[dict[str, torch.Tensor]] | None = None
        self._train_loader = None
        self._val_loader = None

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

    def _next_train_batch(self) -> dict[str, torch.Tensor]:
        if self._train_loader is None:
            self._train_loader = build_train_loader(self.config, "train")
            self._train_iter = iter(self._train_loader)
        assert self._train_iter is not None
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
            "batch_size": batch_size,
            "seq_len": seq_len,
            "device": str(self.device),
            "dtype": self.config.training.dtype,
            "memory": self._memory_report(),
        }

    def train(
        self,
        *,
        max_steps: int | None = None,
        resume: str | Path | None = None,
        eval_only: bool = False,
    ) -> TrainingResult:
        """Run training until ``max_steps`` or ``config.training.max_steps``."""

        target_steps = max_steps if max_steps is not None else self.config.training.max_steps
        if target_steps <= 0:
            raise ValueError("max_steps must be positive.")
        self._active_max_steps = target_steps
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

        self.logger.log("run_start", **self._log_common(global_step=self.global_step))
        try:
            if eval_only:
                self.evaluate()
                self.logger.log("run_end", **self._log_common(global_step=self.global_step))
                return TrainingResult(self.run_dir, self.global_step, self.best_val_loss, self.last_train_loss)

            while self.global_step < target_steps:
                self._train_one_optimizer_step(target_steps=target_steps)
            self.logger.log("run_end", **self._log_common(global_step=self.global_step))
            return TrainingResult(self.run_dir, self.global_step, self.best_val_loss, self.last_train_loss)
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
        should_save_interval = self.global_step % self.config.training.save_every == 0
        should_save_final = self.config.training.always_save_checkpoint and self.global_step == target_steps
        if should_save_interval or should_save_final:
            self._save_checkpoint()

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
            self._save_checkpoint()
        self.logger.log(
            "eval",
            split="val",
            loss=val_loss,
            perplexity=self._safe_perplexity(val_loss),
            **self._log_common(global_step=self.global_step),
        )
        return {"loss": val_loss, "perplexity": self._safe_perplexity(val_loss)}

    def _save_checkpoint(self) -> Path:
        path = save_checkpoint(
            run_dir=self.run_dir,
            config=self.config,
            model=self.model,
            optimizer=self.optimizer,
            scaler=self.scaler,
            global_step=self.global_step,
            best_val_loss=self.best_val_loss,
        )
        self.logger.log("checkpoint", checkpoint_path=str(path), **self._log_common(global_step=self.global_step))
        return path

    @staticmethod
    def _safe_perplexity(loss: float | None) -> float | None:
        if loss is None:
            return None
        if loss > 80:
            return float("inf")
        return math.exp(loss)
