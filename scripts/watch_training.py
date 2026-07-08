"""Print compact progress updates from a Phase 10 events.jsonl file."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def _format_pct(value: object) -> str:
    if not isinstance(value, (int, float)):
        return "n/a"
    return f"{value * 100:.1f}%"


def _format_float(value: object, digits: int = 4) -> str:
    if not isinstance(value, (int, float)):
        return "n/a"
    return f"{value:.{digits}f}"


def emit_existing(path: Path) -> int:
    """Print existing optimizer/checkpoint events and return file offset."""

    if not path.exists():
        return 0
    offset = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            offset += len(line.encode("utf-8"))
            emit_line(line)
    return offset


def emit_line(line: str) -> None:
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        return

    kind = event.get("event")
    if kind == "optimizer_step":
        print(
            "step "
            f"{event.get('global_step')}/{event.get('max_steps')} "
            f"progress={_format_pct(event.get('progress'))} "
            f"loss={_format_float(event.get('loss'))} "
            f"ppl={_format_float(event.get('perplexity'), 2)} "
            f"tps={_format_float(event.get('tokens_per_sec'), 1)} "
            f"tokens={event.get('tokens_consumed')}"
        )
    elif kind == "checkpoint":
        print(
            "checkpoint "
            f"step={event.get('global_step')} "
            f"path={event.get('checkpoint_path')}"
        )
    elif kind == "run_end":
        print(f"run_end step={event.get('global_step')} tokens={event.get('tokens_consumed')}")


def follow(path: Path, offset: int, interval_s: float) -> None:
    while True:
        if not path.exists():
            time.sleep(interval_s)
            continue
        with path.open("r", encoding="utf-8") as handle:
            handle.seek(offset)
            for line in handle:
                emit_line(line)
            offset = handle.tell()
        time.sleep(interval_s)


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch EdgeGPT JSONL training progress.")
    parser.add_argument("events", type=Path, help="Path to events.jsonl.")
    parser.add_argument("--follow", action="store_true", help="Keep watching for new events.")
    parser.add_argument("--interval", type=float, default=2.0, help="Polling interval when following.")
    args = parser.parse_args()

    offset = emit_existing(args.events)
    if args.follow:
        follow(args.events, offset, args.interval)


if __name__ == "__main__":
    main()
