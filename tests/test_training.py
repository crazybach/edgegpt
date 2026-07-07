"""Tests for Phase 10 training loop, checkpoints, and JSONL reporting."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from configs.config import Config, DataConfig, ModelConfig, TokenizerConfig, TrainingConfig
from data.pipeline.shards import TOKEN_DTYPE
from model import EdgeGPT
from train import (
    JsonlEventLogger,
    Trainer,
    build_weight_decay_groups,
    capture_dataloader_state,
    get_warmup_cosine_lr,
    load_checkpoint,
    restore_dataloader_state,
    save_checkpoint,
    write_checkpoint_report,
)


def _write_shards(cache_dir: Path, *, vocab_size: int = 32, block_size: int = 8) -> None:
    """Write tiny prepared train/val shards matching the Phase 2 contract."""

    cache_dir.mkdir(parents=True, exist_ok=True)
    train_tokens = (np.arange(96, dtype=np.uint16) % vocab_size).astype(TOKEN_DTYPE)
    val_tokens = (np.arange(32, dtype=np.uint16) % vocab_size).astype(TOKEN_DTYPE)
    train_tokens.tofile(cache_dir / "train.bin")
    val_tokens.tofile(cache_dir / "val.bin")
    metadata = {
        "dataset": "unit",
        "source_type": "local_text",
        "storage_type": "memmap_bin",
        "dtype": "uint16",
        "vocab_size": vocab_size,
        "tokenizer_artifact_dir": str(cache_dir / "tokenizer"),
        "eos_id": 3,
        "block_size": block_size,
        "seed": 123,
        "val_split": 0.2,
        "token_counts": {"train": int(train_tokens.size), "val": int(val_tokens.size)},
        "document_counts": {"train": 1, "val": 1},
    }
    (cache_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


def _tiny_config(
    tmp_path: Path,
    *,
    grad_accum: int = 1,
    max_steps: int = 2,
    checkpoint_enabled: bool = True,
    save_every: int = 100,
) -> Config:
    cache_dir = tmp_path / "data"
    _write_shards(cache_dir)
    return Config(
        model=ModelConfig(
            vocab_size=32,
            d_model=16,
            n_layers=1,
            n_heads=4,
            n_kv_heads=2,
            d_ff=48,
            max_seq_len=8,
            dropout=0.0,
        ),
        tokenizer=TokenizerConfig(vocab_size=32, reserved_special_tokens=8, artifact_dir=str(tmp_path / "tok")),
        training=TrainingConfig(
            batch_size=2,
            gradient_accumulation_steps=grad_accum,
            learning_rate=1e-3,
            min_lr=1e-4,
            warmup_steps=1,
            max_steps=max_steps,
            weight_decay=0.1,
            dtype="fp32",
            eval_every=100,
            save_every=save_every,
            log_every=1,
            output_dir=str(tmp_path / "runs"),
            eval_iters=2,
            checkpoint_keep_last=2,
            checkpoint_enabled=checkpoint_enabled,
        ),
        data=DataConfig(
            dataset="unit",
            cache_dir=str(cache_dir),
            block_size=8,
            seed=123,
            val_split=0.2,
        ),
        device="cpu",
    )


# ═══════════════════════════════════════════════════════════════════════
# Weight decay groups
# ═══════════════════════════════════════════════════════════════════════


def test_adamw_groups_exclude_norms_and_embeddings_from_decay(tmp_path: Path):
    model = EdgeGPT(_tiny_config(tmp_path))

    groups = build_weight_decay_groups(model, weight_decay=0.1)
    decay_names = set(groups[0]["param_names"])
    no_decay_names = set(groups[1]["param_names"])

    assert "embed_tokens.embedding.weight" in no_decay_names
    assert "norm.weight" in no_decay_names
    assert "layers.0.attention.q_proj.weight" in decay_names
    assert "layers.0.mlp.down_proj.weight" in decay_names
    assert all("norm.weight" not in name for name in decay_names)


# ═══════════════════════════════════════════════════════════════════════
# LR schedule
# ═══════════════════════════════════════════════════════════════════════


def test_warmup_cosine_schedule_boundaries():
    lr0 = get_warmup_cosine_lr(0, learning_rate=1.0, min_lr=0.1, warmup_steps=4, max_steps=20)
    lr3 = get_warmup_cosine_lr(3, learning_rate=1.0, min_lr=0.1, warmup_steps=4, max_steps=20)
    lr_mid = get_warmup_cosine_lr(12, learning_rate=1.0, min_lr=0.1, warmup_steps=4, max_steps=20)
    lr_end = get_warmup_cosine_lr(20, learning_rate=1.0, min_lr=0.1, warmup_steps=4, max_steps=20)

    assert lr0 == pytest.approx(0.25)
    assert lr3 == pytest.approx(1.0)
    assert 0.1 < lr_mid < 1.0
    assert lr_end == pytest.approx(0.1)


# ═══════════════════════════════════════════════════════════════════════
# Training step
# ═══════════════════════════════════════════════════════════════════════


def test_one_training_step_updates_parameters(tmp_path: Path):
    config = _tiny_config(tmp_path, max_steps=1)
    trainer = Trainer(config, run_dir=tmp_path / "run")
    before = trainer.model.layers[0].attention.q_proj.weight.detach().clone()

    result = trainer.train(max_steps=1)

    after = trainer.model.layers[0].attention.q_proj.weight.detach()
    assert result.global_step == 1
    assert not torch.allclose(before, after)
    assert (tmp_path / "run" / "events.jsonl").exists()


def test_gradient_accumulation_controls_optimizer_steps(tmp_path: Path):
    config = _tiny_config(tmp_path, grad_accum=3, max_steps=1)
    trainer = Trainer(config, run_dir=tmp_path / "run")
    calls = {"count": 0}
    original_step = trainer.optimizer.step

    def counted_step(*args, **kwargs):
        calls["count"] += 1
        return original_step(*args, **kwargs)

    trainer.optimizer.step = counted_step  # type: ignore[method-assign]
    trainer.train(max_steps=1)

    assert calls["count"] == 1
    events = [json.loads(line) for line in (tmp_path / "run" / "events.jsonl").read_text().splitlines()]
    assert sum(1 for event in events if event["event"] == "micro_step") == 3


def test_training_tracks_tokens_consumed(tmp_path: Path):
    """tokens_consumed should increase with each training step."""
    config = _tiny_config(tmp_path, max_steps=2, checkpoint_enabled=False)
    trainer = Trainer(config, run_dir=tmp_path / "run")

    result = trainer.train(max_steps=2)

    assert result.tokens_consumed > 0
    # Each step: B=2, T=8, grad_accum=1 → 16 tokens per step → 32 for 2 steps.
    assert result.tokens_consumed == 2 * 2 * 8


# ═══════════════════════════════════════════════════════════════════════
# JSONL logging
# ═══════════════════════════════════════════════════════════════════════


def test_jsonl_logger_writes_required_fields(tmp_path: Path):
    logger = JsonlEventLogger(tmp_path, "unit")
    logger.log("optimizer_step", global_step=1, loss=2.0, progress=0.5)

    record = json.loads((tmp_path / "events.jsonl").read_text(encoding="utf-8").strip())
    for field in ("event", "time", "run_id", "global_step", "progress", "loss", "checkpoint_path"):
        assert field in record
    assert record["event"] == "optimizer_step"
    assert record["run_id"] == "unit"


# ═══════════════════════════════════════════════════════════════════════
# Checkpoint save / load
# ═══════════════════════════════════════════════════════════════════════


def test_checkpoint_save_load_restores_model_optimizer_and_step(tmp_path: Path):
    config = _tiny_config(tmp_path)
    model = EdgeGPT(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    with torch.no_grad():
        model.embed_tokens.weight[0, 0] = 123.0

    path = save_checkpoint(
        run_dir=tmp_path / "run",
        config=config,
        model=model,
        optimizer=optimizer,
        scaler=None,
        global_step=7,
        best_val_loss=1.5,
        tokens_consumed=1000,
    )

    restored = EdgeGPT(config)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    state = load_checkpoint(path=path, model=restored, optimizer=restored_optimizer, restore_rng=False)

    assert state.global_step == 7
    assert state.best_val_loss == pytest.approx(1.5)
    assert state.tokens_consumed == 1000
    assert restored.embed_tokens.weight[0, 0].item() == pytest.approx(123.0)


def test_checkpoint_writes_json_report(tmp_path: Path):
    """Each checkpoint should produce a companion .json summary."""
    config = _tiny_config(tmp_path, max_steps=1, save_every=1)
    trainer = Trainer(config, run_dir=tmp_path / "run")

    trainer.train(max_steps=1)

    report_path = tmp_path / "run" / "step_1.json"
    assert report_path.exists()
    report = json.loads(report_path.read_text())
    assert report["global_step"] == 1
    assert report["tokens_consumed"] > 0
    assert "train_loss" in report
    assert "lr" in report
    assert "elapsed_s" in report
    assert "timestamp" in report


def test_resume_continues_from_saved_step(tmp_path: Path):
    config = _tiny_config(tmp_path, max_steps=2, checkpoint_enabled=True, save_every=1)
    first = Trainer(config, run_dir=tmp_path / "run")
    first.train(max_steps=1)
    checkpoint = tmp_path / "run" / "latest.pt"
    assert checkpoint.exists()

    resumed = Trainer(config, run_dir=tmp_path / "run")
    result = resumed.train(max_steps=2, resume=checkpoint)

    assert result.global_step == 2


def test_resume_preserves_tokens_consumed(tmp_path: Path):
    """Resuming should add to tokens_consumed, not reset to 0."""
    config = _tiny_config(tmp_path, max_steps=3, checkpoint_enabled=True, save_every=1)
    first = Trainer(config, run_dir=tmp_path / "run")
    r1 = first.train(max_steps=2)
    tokens_after_first = r1.tokens_consumed

    checkpoint = tmp_path / "run" / "latest.pt"
    resumed = Trainer(config, run_dir=tmp_path / "run")
    r2 = resumed.train(max_steps=3, resume=checkpoint)

    # Second run should have consumed more tokens, not reset.
    assert r2.tokens_consumed > tokens_after_first


def test_checkpoint_disabled_writes_no_files(tmp_path: Path):
    """With checkpoint_enabled=False, no .pt or .json files are produced."""
    config = _tiny_config(tmp_path, max_steps=2, checkpoint_enabled=False, save_every=1)
    trainer = Trainer(config, run_dir=tmp_path / "run")

    trainer.train(max_steps=2)

    pt_files = list((tmp_path / "run").glob("step_*.pt"))
    json_files = list((tmp_path / "run").glob("step_*.json"))
    assert len(pt_files) == 0
    assert len(json_files) == 0


# ═══════════════════════════════════════════════════════════════════════
# DataLoader state
# ═══════════════════════════════════════════════════════════════════════


def test_dataloader_state_capture_restore_roundtrip(tmp_path: Path):
    """Restoring sampler state reproduces the shuffle position faithfully.

    We capture the state BEFORE consuming the first batch, then restore
    it onto a fresh loader.  Both should produce the same first batch
    because they start from the same shuffle position.
    """
    config = _tiny_config(tmp_path)
    from data.pipeline import build_train_loader

    loader1 = build_train_loader(config, "train")
    # Capture state at the initial position (before any consumption).
    state = capture_dataloader_state(loader1)
    iter1 = iter(loader1)
    batch1 = next(iter1)

    loader2 = build_train_loader(config, "train")
    restore_dataloader_state(loader2, state)
    iter2 = iter(loader2)
    batch2 = next(iter2)

    assert torch.equal(batch1["input_ids"], batch2["input_ids"])
    assert torch.equal(batch1["targets"], batch2["targets"])


# ═══════════════════════════════════════════════════════════════════════
# Eval
# ═══════════════════════════════════════════════════════════════════════


def test_eval_returns_metrics_and_restores_training_mode(tmp_path: Path):
    trainer = Trainer(_tiny_config(tmp_path), run_dir=tmp_path / "run")
    trainer.model.train()

    metrics = trainer.evaluate()

    assert metrics is not None
    assert metrics["loss"] > 0
    assert metrics["perplexity"] > 0
    assert trainer.model.training is True


# ═══════════════════════════════════════════════════════════════════════
# Script smoke test
# ═══════════════════════════════════════════════════════════════════════


def test_train_script_smoke_creates_events_and_checkpoint(tmp_path: Path):
    config = _tiny_config(tmp_path, max_steps=2)
    yaml_path = tmp_path / "tiny.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "model": vars(config.model),
                "training": vars(config.training),
                "data": vars(config.data),
                "tokenizer": vars(config.tokenizer),
                "device": "cpu",
            }
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [sys.executable, "scripts/train.py", "--config", str(yaml_path), "--run-name", "smoke", "--max-steps", "2"],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )

    run_dir = Path(config.training.output_dir) / "smoke"
    assert (run_dir / "events.jsonl").exists()
    assert (run_dir / "latest.pt").exists()
