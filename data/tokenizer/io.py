"""Tokenizer artifact IO helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


TOKENIZER_JSON = "tokenizer.json"
TOKENIZER_CONFIG_JSON = "tokenizer_config.json"
SPECIAL_TOKENS_MAP_JSON = "special_tokens_map.json"
METRICS_JSON = "metrics.json"


def ensure_output_dir(path: Path) -> Path:
    """Create an artifact directory and return it as a resolved path."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write deterministic JSON for tokenizer sidecar files."""

    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    """Read a tokenizer sidecar JSON file."""

    return json.loads(path.read_text(encoding="utf-8"))


def tokenizer_json_path(path: Path) -> Path:
    """Accept either an artifact directory or a direct tokenizer.json path."""

    return path / TOKENIZER_JSON if path.is_dir() else path
