# EdgeGPT — Phase 10b Checkpointing System Technical Design

## Summary

Phase 10b extends the Phase 10 training loop with a production-grade
checkpointing system that supports pause/resume, data-position tracking,
multi-checkpoint rollback, and human-readable snapshot reports.  The system
is feature-gated: setting `training.checkpoint_enabled = false` disables all
checkpoint I/O and the trainer runs without writing any `.pt` or `.json`
files.

This document covers the what, why, and how.  For implementation-level
detail, see the inline docstrings in `train/checkpoint.py` and
`train/trainer.py`.

---

## Requirements Checklist

| # | Requirement | Status | Implementation |
|---|-------------|--------|----------------|
| 1 | Save a snapshot of current weights | ✅ Done | `save_checkpoint()` writes `model.state_dict()`, `optimizer.state_dict()`, `scaler.state_dict()`, `config`, and full RNG state |
| 2 | Save position in the training data stream | ✅ Done | `capture_dataloader_state()` snapshots the `RandomSampler.generator` so resume continues the same shuffle. `tokens_consumed` counter tracks unique tokens seen |
| 3 | Multiple checkpoints for rollback & fork | ✅ Done | `step_{N}.pt` numbered files pruned to `checkpoint_keep_last`. Any step checkpoint can be loaded via `--resume path/to/step_N.pt` |
| 4 | All state needed to restore training | ✅ Done | RNG (Python, numpy, torch, CUDA), DataLoader sampler, optimizer, scaler, config, `global_step`, `best_val_loss`, `tokens_consumed` |
| 5 | Human-readable snapshot report per checkpoint | ✅ Done | `step_{N}.json` companion files with loss, perplexity, progress %, LR, elapsed time, timestamp |

---

## Checkpoint File Layout

After a training run, the run directory contains:

```
artifacts/runs/20250101_120000/
├── events.jsonl          # Structured training log (Phase 10)
├── latest.pt             # Stable resume target (copy of newest step_N.pt)
├── latest.json           # Companion report (copy of newest step_N.json)
├── step_0.pt             # Initial checkpoint (optional, on first save)
├── step_0.json
├── step_2000.pt          # Interval checkpoint
├── step_2000.json
├── step_4000.pt          # Interval checkpoint (pruned when > keep_last)
├── step_4000.json
└── ...
```

`prune_checkpoints()` keeps the most recent N `step_*.pt` (and their `.json`
companions), controlled by `training.checkpoint_keep_last` (default 3).

### Checkpoint file contents (`step_N.pt`)

| Key | Type | Purpose |
|-----|------|---------|
| `model` | `OrderedDict` | `model.state_dict()` — all parameter tensors |
| `optimizer` | `dict` | `optimizer.state_dict()` — AdamW moment buffers |
| `scaler` | `dict` or `None` | `GradScaler.state_dict()` — fp16 scale factor |
| `global_step` | `int` | Optimizer step count at save time |
| `best_val_loss` | `float` or `None` | Best validation loss seen so far |
| `tokens_consumed` | `int` | Cumulative unique tokens consumed from dataset |
| `dataloader_state` | `dict` | `RandomSampler.generator` state for shuffle continuation |
| `config` | `dict` | Full serialized `Config` (`dataclasses.asdict`) |
| `rng_state` | `dict` | Python `random`, `numpy`, `torch`, and `cuda` RNG states |

### Report file contents (`step_N.json`)

```json
{
  "global_step": 2000,
  "max_steps": 50000,
  "progress_pct": 4.0,
  "tokens_consumed": 32768000,
  "dataset_progress_pct": 13.1,
  "train_loss": 4.521,
  "val_loss": 5.012,
  "perplexity": 91.93,
  "lr": 0.00028,
  "elapsed_s": 1423.5,
  "timestamp": "2025-01-01 12:23:45"
}
```

---

## Checkpoint Save Triggers

| Trigger | Config control | Kind label |
|---------|---------------|------------|
| Interval (every N steps) | `training.save_every` | `interval` |
| New best validation loss | Automatic during `evaluate()` | `best` |
| Final step reached | `training.always_save_checkpoint` | `interval` |
| Graceful interrupt (Ctrl+C) | Always on when checkpoints enabled | `interrupt` |

All triggers are gated by `training.checkpoint_enabled`.  When disabled,
none of these events write to disk.

---

## Data Position Tracking

### Problem

On resume, the DataLoader with `shuffle=True` uses a `RandomSampler` backed
by a `torch.Generator`.  Each call to `next(iter(loader))` advances the
generator's internal state, consuming pseudorandom bits that determine which
dataset indices are drawn.  Without saving this state, a resumed run starts
from index 0 of a fresh shuffle and re-consumes tokens already seen.

### Solution

`capture_dataloader_state(loader)` extracts `loader.sampler.generator.get_state()`.
This byte tensor is stored in the checkpoint under `dataloader_state`.

On resume, the trainer rebuilds the DataLoader, then calls
`restore_dataloader_state(loader, state.dataloader_state)` which calls
`sampler.generator.set_state(...)`.  The next `next(iter(loader))` returns
the batch that *would have been next* had training never stopped.

### In addition

The `tokens_consumed` counter (incremented by `B * T` per micro-batch)
provides a true cumulative count of unique dataset tokens consumed.
Combined with `dataset_total_tokens` (discovered from `len(dataset) * block_size`),
the report computes `dataset_progress_pct = tokens_consumed / total * 100`.

On resume, `tokens_consumed` is restored from the checkpoint and continues
accumulating — it is NOT reset.

---

## Disabling Checkpoints

Set `training.checkpoint_enabled: false` in the YAML config or pass a
`TrainingConfig(checkpoint_enabled=False)` programmatically.  When disabled:

- No `step_*.pt` or `latest.pt` files are written
- No `step_*.json` or `latest.json` reports are written
- The JSONL event log (`events.jsonl`) still records training progress
- The trainer cannot be resumed (no checkpoint to load from)

This is useful for quick experiments, debugging runs, or when disk I/O is
the bottleneck.

---

## Usage Examples

### Start a new training run

```bash
python scripts/train.py --config configs/default.yaml --run-name experiment-1
```

### Resume from the latest checkpoint

```bash
python scripts/train.py --config configs/default.yaml --resume artifacts/runs/experiment-1/latest.pt
```

### Fork from an earlier checkpoint (rollback)

```bash
python scripts/train.py --config configs/default.yaml \
    --resume artifacts/runs/experiment-1/step_2000.pt \
    --run-name experiment-1-fork
```

The `--run-name experiment-1-fork` ensures the fork writes to a different
run directory so the original checkpoint is not overwritten.

### Inspect a checkpoint without loading into PyTorch

```bash
cat artifacts/runs/experiment-1/step_2000.json
```

### Run without checkpoints

```yaml
# In config YAML:
training:
  checkpoint_enabled: false
```

```bash
python scripts/train.py --config configs/no_checkpoint.yaml
```

---

## Implementation Files

| File | Role |
|------|------|
| `train/checkpoint.py` | `save_checkpoint`, `load_checkpoint`, `write_checkpoint_report`, `capture_dataloader_state`, `restore_dataloader_state`, `prune_checkpoints`, `capture_rng_state`, `restore_rng_state` |
| `train/trainer.py` | `_maybe_save_checkpoint` (gating), `_save_checkpoint` (orchestration), `_install_sigint_handler` (SIGINT), `tokens_consumed` tracking, resume dataloader restore |
| `configs/config.py` | `TrainingConfig.checkpoint_enabled`, `checkpoint_keep_last`, `always_save_checkpoint` |
| `tests/test_training.py` | 5 checkpointing-specific tests: save/load roundtrip, JSON report, resume preserves tokens, checkpoint disabled, dataloader state capture/restore |

---

## Invariants

- `checkpoint_enabled = False` guarantees zero checkpoint I/O — no `.pt` or `.json` files are created.
- A checkpoint loaded with `load_checkpoint()` restores the model to the exact parameter values at save time (verified by weight equality test).
- `tokens_consumed` is monotonic across saves and resumes — it never decreases.
- Restoring a DataLoader sampler state produces the same next batch as would have been produced without the pause (verified by roundtrip test).
- `prune_checkpoints` never deletes `latest.pt` or `latest.json`.
- Checkpoint reports are valid JSON with stable keys so downstream tooling can parse them.
- The SIGINT handler saves exactly one emergency checkpoint before the process exits.
