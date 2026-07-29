"""Train the EdgeGPT tokenizer from configured text files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.config import load_config  # noqa: E402
from data.tokenizer.io import METRICS_JSON, write_json  # noqa: E402
from data.tokenizer.metrics import compute_tokenizer_metrics  # noqa: E402
from data.tokenizer.registry import build_tokenizer  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train EdgeGPT's Phase 1 tokenizer.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--output-dir", default=None, help="Override tokenizer artifact directory.")
    parser.add_argument("--metrics-lines", type=int, default=10000, help="Number of text lines sampled for metrics.")
    args = parser.parse_args()

    config = load_config(args.config)
    tokenizer_config = config.tokenizer
    output_dir = Path(args.output_dir or tokenizer_config.artifact_dir)
    train_files = [Path(path) for path in tokenizer_config.train_files]

    tokenizer = build_tokenizer(tokenizer_config)
    tokenizer.train(train_files, output_dir)

    metrics = compute_tokenizer_metrics(tokenizer, train_files, max_lines=args.metrics_lines)
    write_json(output_dir / METRICS_JSON, metrics)
    print(f"Saved tokenizer to {output_dir}")
    print(metrics)


if __name__ == "__main__":
    main()
