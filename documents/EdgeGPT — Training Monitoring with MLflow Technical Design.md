# EdgeGPT Training Monitoring with MLflow

## Status

Implemented as an optional monitoring layer. JSONL remains the default and
training does not import MLflow unless the `mlflow` backend is enabled.

## Goals

- Show loss, perplexity, learning rate, progress, throughput, token counts,
  validation metrics, and CUDA allocator memory in a browser.
- Keep a durable local `events.jsonl` record that is independent of MLflow.
- Ensure a dashboard or network failure cannot stop a long training run.
- Preserve a path from one Windows GPU to a central multi-host service.
- Avoid copying large checkpoints unless explicitly requested.

## Architecture

```text
Trainer
  |
  | structured event
  v
EventLogger
  +-- JsonlEventSink  --> run_dir/events.jsonl
  +-- MlflowEventSink --> MLflow tracking API --> web dashboard
  +-- future sinks    --> notifications, Prometheus, or another tracker
```

`Trainer` depends only on `EventLogger`. It contains no MLflow calls. The
logger normalizes each event once and sends it to configured `EventSink`
implementations. A sink that raises an exception is disabled after its first
failure unless `monitoring.fail_on_error` is enabled.

The trainer also accepts an injected `event_logger`, which permits tests,
special deployments, and future distributed launchers to construct their own
sink set without changing the training loop.

## Event Mapping

| EdgeGPT event | MLflow behavior |
|---|---|
| `optimizer_step` | Logs train loss, perplexity, LR, gradient norm, progress, token count, throughput, batch/sequence size, and CUDA allocator memory |
| `eval` | Logs validation loss, perplexity, and progress |
| `checkpoint` | Updates the latest-checkpoint tag and uploads the small JSON report |
| `run_end` | Marks the MLflow run finished, or killed after a graceful interrupt |
| `error` | Stores the error tag and marks the run failed |

The `.pt` checkpoint is not uploaded by default because a 1B training
checkpoint can be several gigabytes and already exists in the run directory.
Set `mlflow_log_checkpoints: true` only when the artifact store has sufficient
capacity and checkpoint duplication is desired.

## Installation

MLflow is intentionally separate from the core requirements:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-monitoring.txt
```

Start the local server and web UI:

```powershell
.\scripts\start_mlflow.bat
```

The dashboard is available at `http://127.0.0.1:5001`. Closing the server
console or pressing Ctrl+C stops it.

## Starting a Monitored Run

CLI flags can enable monitoring without editing a model configuration:

```powershell
.\.venv\Scripts\python.exe scripts\train.py `
  --config configs\improvement_1_20m_2k.yaml `
  --run-name improvement_1_monitored `
  --monitor jsonl `
  --monitor mlflow
```

The equivalent YAML is:

```yaml
monitoring:
  backends: ["jsonl", "mlflow"]
  mlflow_tracking_uri: "http://127.0.0.1:5001"
  mlflow_experiment_name: "edgegpt"
  mlflow_log_checkpoints: false
  fail_on_error: false
```

Removing `mlflow` restores dependency-free JSONL-only monitoring. Removing all
backends disables event persistence entirely. For production training, keep
`jsonl` enabled even when MLflow is used.

## Resume Behavior

The adapter writes `mlflow_run_id.txt` beside the checkpoint and event log.
Resuming from that run directory reopens the same MLflow run rather than
creating a disconnected dashboard entry. Training state remains owned by the
PyTorch checkpoint; the MLflow run ID is monitoring metadata only.

## Local Storage and Growth Path

The supplied launcher uses SQLite for metadata and
`artifacts/mlflow-artifacts` for reports. Training processes communicate with
it through HTTP rather than opening the database directly. It uses one local
server worker, which is sufficient for a personal training machine and keeps
Windows shutdown behavior predictable.

For larger deployments:

1. Move MLflow metadata from SQLite to PostgreSQL.
2. Move artifacts to MinIO, S3, or equivalent object storage.
3. Point all training hosts at the central tracking URI.
4. In distributed training, let rank 0 own global run metrics and checkpoint
   metadata; add rank-prefixed hardware metrics only where useful.
5. Add a separate GPU telemetry sink using NVIDIA DCGM on Linux clusters.

## Current Limits and Next Extensions

- CUDA metrics currently come from PyTorch's allocator. GPU utilization,
  temperature, and power sampling can be added as a separate sink without
  touching `Trainer`.
- Notifications are not part of MLflow event tracking. A future notification
  sink should consume checkpoint, error, progress milestone, stall, and
  run-end events.
- The current trainer is single-process. Rank ownership must be added when DDP
  training is implemented.

## Verification

`tests/test_monitoring.py` verifies sink isolation, event-to-metric mapping,
run completion, configuration validation, and persisted MLflow run identity.
The existing training tests protect JSONL and checkpoint behavior.
