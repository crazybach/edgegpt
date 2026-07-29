"""Run Phase 10 EdgeGPT training."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.config import load_config  # noqa: E402
from train import Trainer  # noqa: E402


def build_run_dir(output_dir: str, run_name: str | None, resume: str | None) -> Path:
    """Choose the run directory for a new or resumed training run."""

    if resume:
        return Path(resume).resolve().parent
    suffix = run_name or time.strftime("%Y%m%d_%H%M%S")
    return Path(output_dir) / suffix


def main() -> None:
    parser = argparse.ArgumentParser(description="Train EdgeGPT with Phase 10 trainer.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--run-name", default=None, help="Optional run directory name under training.output_dir.")
    parser.add_argument("--resume", default=None, help="Path to a checkpoint, usually latest.pt.")
    parser.add_argument("--max-steps", type=int, default=None, help="Override final global optimizer step.")
    parser.add_argument("--eval-only", action="store_true", help="Run validation once and exit.")
    parser.add_argument(
        "--monitor",
        action="append",
        choices=("jsonl", "mlflow"),
        help="Monitoring backend. Repeat for multiple backends; overrides config when supplied.",
    )
    parser.add_argument("--mlflow-tracking-uri", default=None, help="Override the MLflow tracking URI.")
    parser.add_argument("--mlflow-experiment", default=None, help="Override the MLflow experiment name.")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.monitor:
        config.monitoring.backends = list(dict.fromkeys(args.monitor))
    if args.mlflow_tracking_uri:
        config.monitoring.mlflow_tracking_uri = args.mlflow_tracking_uri
    if args.mlflow_experiment:
        config.monitoring.mlflow_experiment_name = args.mlflow_experiment
    config.validate()
    run_dir = build_run_dir(config.training.output_dir, args.run_name, args.resume)
    trainer = Trainer(config, run_dir=run_dir, run_id=run_dir.name)
    print(f"run_dir={trainer.run_dir}")
    print(f"events={trainer.run_dir / 'events.jsonl'}")

    result = trainer.train(max_steps=args.max_steps, resume=args.resume, eval_only=args.eval_only)
    print(
        "done "
        f"global_step={result.global_step} "
        f"last_train_loss={result.last_train_loss} "
        f"best_val_loss={result.best_val_loss}"
    )


if __name__ == "__main__":
    main()
