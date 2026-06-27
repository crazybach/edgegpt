"""Prepare Phase 2 token shards from raw text documents."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.config import load_config
from data.pipeline import prepare_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare EdgeGPT Phase 2 data shards.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    args = parser.parse_args()

    metadata = prepare_data(load_config(args.config))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
