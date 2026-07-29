"""Tests for backend-independent training monitoring."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from configs.config import Config, MonitoringConfig
from train.logging import EventLogger, JsonlEventSink, MlflowEventSink


class _FailingSink:
    def __init__(self) -> None:
        self.calls = 0

    def emit(self, record: dict[str, object]) -> None:
        self.calls += 1
        raise RuntimeError("offline")


def test_event_logger_isolates_a_failed_optional_sink(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    failing = _FailingSink()
    logger = EventLogger("run", [failing, JsonlEventSink(tmp_path, "run")])

    logger.log("optimizer_step", global_step=1, loss=2.0)
    logger.log("optimizer_step", global_step=2, loss=1.5)

    records = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [record["global_step"] for record in records] == [1, 2]
    assert failing.calls == 1
    assert capsys.readouterr().err.count("monitoring sink _FailingSink disabled") == 1


class _FakeMlflow:
    def __init__(self) -> None:
        self.tracking_uri = None
        self.experiment = None
        self.params: dict[str, object] = {}
        self.tags: dict[str, str] = {}
        self.metrics: list[tuple[dict[str, float], int]] = []
        self.artifacts: list[tuple[str, str | None]] = []
        self.ended_status = None

    def set_tracking_uri(self, uri: str) -> None:
        self.tracking_uri = uri

    def set_experiment(self, name: str) -> None:
        self.experiment = name

    def start_run(self, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(info=SimpleNamespace(run_id=kwargs.get("run_id") or "fake-run-id"))

    def log_params(self, params: dict[str, object]) -> None:
        self.params.update(params)

    def set_tags(self, tags: dict[str, str]) -> None:
        self.tags.update(tags)

    def set_tag(self, key: str, value: str) -> None:
        self.tags[key] = value

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        self.metrics.append((metrics, step))

    def log_artifact(self, path: str, artifact_path: str | None = None) -> None:
        self.artifacts.append((path, artifact_path))

    def end_run(self, status: str) -> None:
        self.ended_status = status


def test_mlflow_sink_maps_events_without_trainer_coupling(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fake = _FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    config = Config(
        monitoring=MonitoringConfig(
            backends=["jsonl", "mlflow"],
            mlflow_tracking_uri="file:./test-mlruns",
            mlflow_experiment_name="unit",
        )
    )
    sink = MlflowEventSink(config, tmp_path, "monitoring-smoke")

    sink.emit(
        {
            "event": "optimizer_step",
            "global_step": 3,
            "split": "train",
            "loss": 1.25,
            "perplexity": 3.49,
            "lr": 0.001,
            "progress": 0.3,
            "memory": {"allocated_mib": 512.0},
        }
    )
    sink.emit({"event": "run_end", "global_step": 3})

    assert fake.tracking_uri == "file:./test-mlruns"
    assert fake.experiment == "unit"
    assert fake.params["model.n_layers"] == config.model.n_layers
    assert fake.metrics[0][0]["train/loss"] == pytest.approx(1.25)
    assert fake.metrics[0][0]["system/cuda_allocated_mib"] == pytest.approx(512.0)
    assert fake.metrics[0][1] == 3
    assert fake.ended_status == "FINISHED"
    assert (tmp_path / "mlflow_run_id.txt").read_text() == "fake-run-id"


def test_mlflow_sink_marks_interrupted_run_killed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fake = _FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    sink = MlflowEventSink(Config(), tmp_path, "interrupted")

    sink.emit({"event": "run_end", "global_step": 4, "status": "interrupted"})

    assert fake.ended_status == "KILLED"


def test_monitoring_config_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unsupported monitoring backends"):
        Config(monitoring=MonitoringConfig(backends=["unknown"]))
