"""Modular structured training events for local and remote monitoring."""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from configs.config import Config


TRAINING_EVENT_FIELDS = (
    "event",
    "time",
    "run_id",
    "global_step",
    "micro_step",
    "max_steps",
    "progress",
    "split",
    "loss",
    "perplexity",
    "lr",
    "grad_norm",
    "tokens_seen",
    "tokens_per_sec",
    "batch_size",
    "seq_len",
    "device",
    "dtype",
    "memory",
    "checkpoint_path",
    "status",
)


class EventSink(Protocol):
    """Destination for a normalized training event."""

    def emit(self, record: dict[str, Any]) -> None:
        """Persist one event."""


class JsonlEventSink:
    """Append normalized events to ``events.jsonl``."""

    def __init__(self, run_dir: str | Path, run_id: str):
        self.run_dir = Path(run_dir)
        self.run_id = run_id
        self.path = self.run_dir / "events.jsonl"
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def emit(self, record: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def _flatten_config(config: "Config") -> dict[str, str | float | int | bool]:
    """Flatten dataclass config into MLflow-compatible scalar parameters."""

    flattened: dict[str, str | float | int | bool] = {}

    def visit(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                visit(f"{prefix}.{key}" if prefix else str(key), child)
        elif isinstance(value, (list, tuple)):
            flattened[prefix] = json.dumps(value)
        elif value is None:
            flattened[prefix] = "null"
        elif isinstance(value, (str, float, int, bool)):
            flattened[prefix] = value
        else:
            flattened[prefix] = str(value)

    visit("", asdict(config))
    return flattened


class MlflowEventSink:
    """Mirror selected training events to MLflow through its fluent API."""

    _METRIC_EVENTS = {"optimizer_step", "eval"}
    _METRIC_FIELDS = (
        "loss",
        "perplexity",
        "lr",
        "grad_norm",
        "progress",
        "tokens_seen",
        "tokens_consumed",
        "tokens_per_sec",
        "batch_size",
        "seq_len",
    )

    def __init__(self, config: "Config", run_dir: str | Path, run_id: str):
        try:
            import mlflow
        except ImportError as exc:
            raise RuntimeError(
                "MLflow monitoring requires the optional dependencies in "
                "requirements-monitoring.txt."
            ) from exc

        self.mlflow = mlflow
        self.config = config
        self.run_dir = Path(run_dir)
        self.run_id = run_id
        self._ended = False
        mlflow.set_tracking_uri(config.monitoring.mlflow_tracking_uri)
        mlflow.set_experiment(config.monitoring.mlflow_experiment_name)

        persisted_id_path = self.run_dir / "mlflow_run_id.txt"
        persisted_id = (
            persisted_id_path.read_text(encoding="utf-8").strip()
            if persisted_id_path.exists()
            else None
        )
        active_run = mlflow.start_run(
            run_id=persisted_id or None,
            run_name=None if persisted_id else run_id,
        )
        self.mlflow_run_id = active_run.info.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        persisted_id_path.write_text(self.mlflow_run_id, encoding="utf-8")

        if persisted_id is None:
            mlflow.log_params(_flatten_config(config))
            mlflow.set_tags(
                {
                    "edgegpt.run_id": run_id,
                    "edgegpt.run_dir": str(self.run_dir.resolve()),
                    "edgegpt.device": config.resolve_device(),
                }
            )

    @staticmethod
    def _finite_number(value: Any) -> float | None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None

    def emit(self, record: dict[str, Any]) -> None:
        if self._ended:
            return

        event = str(record.get("event"))
        step = int(record.get("global_step") or 0)
        if event in self._METRIC_EVENTS:
            split = record.get("split") or "train"
            metrics: dict[str, float] = {}
            for field in self._METRIC_FIELDS:
                value = self._finite_number(record.get(field))
                if value is not None:
                    metrics[f"{split}/{field}"] = value
            memory = record.get("memory")
            if isinstance(memory, dict):
                for field, raw_value in memory.items():
                    value = self._finite_number(raw_value)
                    if value is not None:
                        metrics[f"system/cuda_{field}"] = value
            if metrics:
                self.mlflow.log_metrics(metrics, step=step)

        if event == "checkpoint":
            checkpoint_path = record.get("checkpoint_path")
            if checkpoint_path:
                self.mlflow.set_tag("edgegpt.latest_checkpoint", str(checkpoint_path))
                report_path = Path(str(checkpoint_path)).with_suffix(".json")
                if report_path.exists():
                    self.mlflow.log_artifact(str(report_path), artifact_path="checkpoint_reports")
                if self.config.monitoring.mlflow_log_checkpoints and Path(str(checkpoint_path)).exists():
                    self.mlflow.log_artifact(str(checkpoint_path), artifact_path="checkpoints")
        elif event == "error":
            self.mlflow.set_tag("edgegpt.error", str(record.get("message") or "unknown error"))
            self._end("FAILED")
        elif event == "run_end":
            self._end("KILLED" if record.get("status") == "interrupted" else "FINISHED")

    def _end(self, status: str) -> None:
        if not self._ended:
            self.mlflow.end_run(status=status)
            self._ended = True


class EventLogger:
    """Normalize an event once and fan it out to independent sinks."""

    def __init__(self, run_id: str, sinks: list[EventSink], *, fail_on_error: bool = False):
        self.run_id = run_id
        self.sinks = sinks
        self.fail_on_error = fail_on_error
        self._failed_sink_types: set[type[Any]] = set()

    def log(self, event: str, **payload: Any) -> dict[str, Any]:
        record: dict[str, Any] = {field: None for field in TRAINING_EVENT_FIELDS}
        record.update({"event": event, "time": time.time(), "run_id": self.run_id})
        record.update(payload)

        for sink in self.sinks:
            if type(sink) in self._failed_sink_types:
                continue
            try:
                sink.emit(record)
            except Exception as exc:
                if self.fail_on_error:
                    raise
                sink_type = type(sink)
                if sink_type not in self._failed_sink_types:
                    print(
                        f"warning: monitoring sink {sink_type.__name__} disabled after error: {exc}",
                        file=sys.stderr,
                    )
                    self._failed_sink_types.add(sink_type)
        return record


class JsonlEventLogger(EventLogger):
    """Backward-compatible JSONL-only event logger."""

    def __init__(self, run_dir: str | Path, run_id: str):
        sink = JsonlEventSink(run_dir, run_id)
        self.run_dir = sink.run_dir
        self.path = sink.path
        super().__init__(run_id, [sink])


def build_event_logger(config: "Config", run_dir: str | Path, run_id: str) -> EventLogger:
    """Construct configured sinks without coupling the trainer to a backend."""

    sinks: list[EventSink] = []
    for backend in config.monitoring.backends:
        if backend == "jsonl":
            sinks.append(JsonlEventSink(run_dir, run_id))
        elif backend == "mlflow":
            try:
                sinks.append(MlflowEventSink(config, run_dir, run_id))
            except Exception as exc:
                if config.monitoring.fail_on_error:
                    raise
                print(f"warning: MLflow monitoring unavailable: {exc}", file=sys.stderr)
    return EventLogger(run_id, sinks, fail_on_error=config.monitoring.fail_on_error)
