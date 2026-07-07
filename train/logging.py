"""Structured JSONL training events for live reporting.

Phase 10 starts with file-based observability. Every event is one JSON object
on one line, so a future web dashboard can tail or ingest the same file without
changing the trainer loop.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

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
)


class JsonlEventLogger:
    """Append stable-schema training events to ``events.jsonl``."""

    def __init__(self, run_dir: str | Path, run_id: str):
        self.run_dir = Path(run_dir)
        self.run_id = run_id
        self.path = self.run_dir / "events.jsonl"
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def log(self, event: str, **payload: Any) -> dict[str, Any]:
        """Write one event and return the serializable record."""

        record: dict[str, Any] = {field: None for field in TRAINING_EVENT_FIELDS}
        record.update(
            {
                "event": event,
                "time": time.time(),
                "run_id": self.run_id,
            }
        )
        for key, value in payload.items():
            if key in record:
                record[key] = value
            else:
                record[key] = value

        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        return record
