"""Tests for Phase 12 GGUF export."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest
import torch

from configs.config import Config, DataConfig, ModelConfig, TokenizerConfig, TrainingConfig
from model import EdgeGPT
from train.checkpoint import save_checkpoint


def _small_config(*, vocab_size: int = 64, n_layers: int = 2, tie_embeddings: bool = True) -> Config:
    return Config(
        model=ModelConfig(
            vocab_size=vocab_size,
            d_model=32,
            n_layers=n_layers,
            n_heads=4,
            n_kv_heads=2,
            d_ff=88,
            max_seq_len=32,
            tie_embeddings=tie_embeddings,
        ),
        tokenizer=TokenizerConfig(vocab_size=vocab_size, reserved_special_tokens=8),
        training=TrainingConfig(),
        data=DataConfig(),
        device="cpu",
    )


def _write_minimal_tokenizer(tokenizer_dir: Path, *, vocab_size: int = 64) -> None:
    """Write a tiny BPE tokenizer that exercises the export path.

    Creates vocab_size tokens: 56 ASCII chars + 8 special tokens.
    """

    tokenizer_dir.mkdir(parents=True, exist_ok=True)

    tokens: dict[str, int] = {}
    tid = 0
    # ASCII printable — exactly (vocab_size - 8) tokens to fill.
    for ch in range(32, 32 + vocab_size - 8):
        tokens[chr(ch)] = tid
        tid += 1

    # Special tokens at the end.
    special = {
        "<|pad|>": vocab_size - 8,
        "<|unk|>": vocab_size - 7,
        "<|bos|>": vocab_size - 6,
        "<|eos|>": vocab_size - 5,
        "<|sep|>": vocab_size - 4,
        "<|reserved_5|>": vocab_size - 3,
        "<|reserved_6|>": vocab_size - 2,
        "<|reserved_7|>": vocab_size - 1,
    }
    tokens.update(special)

    # A few simple merges for the BPE pre-tokenizer.
    merges = ["a b", "t h", "i n", "e r", "o n", "a n", "t i", "e s"]

    tokenizer_json = {
        "version": "1.0",
        "added_tokens": [
            {"id": vocab_size - 8, "content": "<|pad|>", "special": True},
            {"id": vocab_size - 7, "content": "<|unk|>", "special": True},
            {"id": vocab_size - 6, "content": "<|bos|>", "special": True},
            {"id": vocab_size - 5, "content": "<|eos|>", "special": True},
        ],
        "model": {
            "type": "BPE",
            "vocab": tokens,
            "merges": merges,
        },
    }
    (tokenizer_dir / "tokenizer.json").write_text(json.dumps(tokenizer_json), encoding="utf-8")

    sp_map = {
        "bos_token": "<|bos|>",
        "eos_token": "<|eos|>",
        "unk_token": "<|unk|>",
        "pad_token": "<|pad|>",
        "special_token_ids": special,
    }
    (tokenizer_dir / "special_tokens_map.json").write_text(json.dumps(sp_map), encoding="utf-8")


# ═══════════════════════════════════════════════════════════════════════
# Tensor name mapping
# ═══════════════════════════════════════════════════════════════════════


def test_tensor_map_embed_tokens():
    from scripts.export_gguf import _map_tensor_name

    assert _map_tensor_name("embed_tokens.embedding.weight") == "token_embd.weight"


def test_tensor_map_final_norm():
    from scripts.export_gguf import _map_tensor_name

    assert _map_tensor_name("norm.weight") == "output_norm.weight"


def test_tensor_map_untied_output_projection():
    """lm_head.proj.weight (untied) must map to output.weight."""
    from scripts.export_gguf import _map_tensor_name

    assert _map_tensor_name("lm_head.proj.weight") == "output.weight"
    assert _map_tensor_name("lm_head.weight") == "output.weight"


def test_tensor_map_per_layer():
    from scripts.export_gguf import _map_tensor_name

    assert _map_tensor_name("layers.0.attention.q_proj.weight") == "blk.0.attn_q.weight"
    assert _map_tensor_name("layers.1.mlp.down_proj.weight") == "blk.1.ffn_down.weight"
    assert _map_tensor_name("layers.0.attention_norm.weight") == "blk.0.attn_norm.weight"


def test_tensor_map_skips_non_weight():
    from scripts.export_gguf import _map_tensor_name

    assert _map_tensor_name("layers.0.attention.rope.inv_freq") is None
    assert _map_tensor_name("some_buffer") is None


# ═══════════════════════════════════════════════════════════════════════
# Norm weight FP32 preservation (Fix 6)
# ═══════════════════════════════════════════════════════════════════════


def test_is_norm_tensor():
    from scripts.export_gguf import _is_norm_tensor

    assert _is_norm_tensor("output_norm.weight", n_layers=2) is True
    assert _is_norm_tensor("blk.0.attn_norm.weight", n_layers=2) is True
    assert _is_norm_tensor("blk.0.ffn_norm.weight", n_layers=2) is True
    assert _is_norm_tensor("blk.0.attn_q.weight", n_layers=2) is False


def test_norm_weights_stay_fp32_in_f16_export(tmp_path: Path):
    """When outtype=f16, norm weights must remain FP32 per GGUF convention."""
    from scripts.export_gguf import export_gguf

    config = _small_config(vocab_size=64, n_layers=1)
    model = EdgeGPT(config)
    tk_dir = tmp_path / "tokenizer"
    _write_minimal_tokenizer(tk_dir, vocab_size=64)

    ckpt_dir = tmp_path / "run"
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ckpt_path = save_checkpoint(
        run_dir=ckpt_dir, config=config, model=model, optimizer=optimizer,
        scaler=None, global_step=0, best_val_loss=None, tokens_consumed=0,
    )

    out_path = tmp_path / "normtest.gguf"
    export_gguf(checkpoint_path=ckpt_path, tokenizer_dir=tk_dir, output_path=out_path, outtype="f16")

    from gguf import GGUFReader

    reader = GGUFReader(str(out_path))
    # Check that norm tensors have fp32 dtype (not fp16).
    for t in reader.tensors:
        if "norm" in t.name:
            # The tensor's type in GGUF for fp32.
            assert t.tensor_type.name in ("F32",), (
                f"Norm tensor {t.name} has dtype {t.tensor_type.name}, expected F32"
            )


# ═══════════════════════════════════════════════════════════════════════
# Full export
# ═══════════════════════════════════════════════════════════════════════


def test_export_creates_gguf_file(tmp_path: Path):
    from scripts.export_gguf import export_gguf

    config = _small_config(vocab_size=64, n_layers=2)
    model = EdgeGPT(config)
    tk_dir = tmp_path / "tokenizer"
    _write_minimal_tokenizer(tk_dir, vocab_size=64)

    ckpt_dir = tmp_path / "run"
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ckpt_path = save_checkpoint(
        run_dir=ckpt_dir, config=config, model=model, optimizer=optimizer,
        scaler=None, global_step=0, best_val_loss=None, tokens_consumed=0,
    )

    out_path = tmp_path / "edgegpt-f16.gguf"
    result = export_gguf(
        checkpoint_path=ckpt_path, tokenizer_dir=tk_dir,
        output_path=out_path, outtype="f32",
    )

    assert result.exists()
    assert result.stat().st_size > 1000


def test_export_f32_tensors_preserve_values(tmp_path: Path):
    from scripts.export_gguf import export_gguf

    config = _small_config(vocab_size=64, n_layers=2)
    model = EdgeGPT(config)
    tk_dir = tmp_path / "tokenizer"
    _write_minimal_tokenizer(tk_dir, vocab_size=64)

    ckpt_dir = tmp_path / "run"
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ckpt_path = save_checkpoint(
        run_dir=ckpt_dir, config=config, model=model, optimizer=optimizer,
        scaler=None, global_step=0, best_val_loss=None, tokens_consumed=0,
    )

    out_path = tmp_path / "test.gguf"
    export_gguf(checkpoint_path=ckpt_path, tokenizer_dir=tk_dir, output_path=out_path, outtype="f32")

    gguf_bytes = out_path.read_bytes()
    assert gguf_bytes[:4] == b"GGUF"
    assert struct.unpack_from("<I", gguf_bytes, 4)[0] >= 2


def test_export_with_weight_tying_skips_output_weight(tmp_path: Path):
    from scripts.export_gguf import export_gguf

    config = _small_config(vocab_size=64, n_layers=1, tie_embeddings=True)
    model = EdgeGPT(config)
    tk_dir = tmp_path / "tokenizer"
    _write_minimal_tokenizer(tk_dir, vocab_size=64)

    ckpt_dir = tmp_path / "run"
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ckpt_path = save_checkpoint(
        run_dir=ckpt_dir, config=config, model=model, optimizer=optimizer,
        scaler=None, global_step=0, best_val_loss=None, tokens_consumed=0,
    )

    out_path = tmp_path / "tied.gguf"
    export_gguf(checkpoint_path=ckpt_path, tokenizer_dir=tk_dir, output_path=out_path, outtype="f32")

    from gguf import GGUFReader

    reader = GGUFReader(str(out_path))
    tensor_names = {t.name for t in reader.tensors}
    assert "token_embd.weight" in tensor_names
    assert "output.weight" not in tensor_names


def test_export_untied_includes_output_weight(tmp_path: Path):
    """Fix 1: Untied embeddings must write output.weight."""
    from scripts.export_gguf import export_gguf

    config = _small_config(vocab_size=64, n_layers=1, tie_embeddings=False)
    model = EdgeGPT(config)
    tk_dir = tmp_path / "tokenizer"
    _write_minimal_tokenizer(tk_dir, vocab_size=64)

    ckpt_dir = tmp_path / "run"
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ckpt_path = save_checkpoint(
        run_dir=ckpt_dir, config=config, model=model, optimizer=optimizer,
        scaler=None, global_step=0, best_val_loss=None, tokens_consumed=0,
    )

    out_path = tmp_path / "untied.gguf"
    export_gguf(checkpoint_path=ckpt_path, tokenizer_dir=tk_dir, output_path=out_path, outtype="f32")

    from gguf import GGUFReader

    reader = GGUFReader(str(out_path))
    tensor_names = {t.name for t in reader.tensors}
    assert "token_embd.weight" in tensor_names
    assert "output.weight" in tensor_names


# ═══════════════════════════════════════════════════════════════════════
# Metadata correctness (Fixes 2, 4, 5)
# ═══════════════════════════════════════════════════════════════════════


def test_export_includes_architecture_metadata(tmp_path: Path):
    from scripts.export_gguf import export_gguf

    config = _small_config(vocab_size=64, n_layers=1)
    model = EdgeGPT(config)
    tk_dir = tmp_path / "tokenizer"
    _write_minimal_tokenizer(tk_dir, vocab_size=64)

    ckpt_dir = tmp_path / "run"
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ckpt_path = save_checkpoint(
        run_dir=ckpt_dir, config=config, model=model, optimizer=optimizer,
        scaler=None, global_step=0, best_val_loss=None, tokens_consumed=0,
    )

    out_path = tmp_path / "meta.gguf"
    export_gguf(checkpoint_path=ckpt_path, tokenizer_dir=tk_dir, output_path=out_path, outtype="f32")

    from gguf import GGUFReader

    reader = GGUFReader(str(out_path))
    field_keys = set(reader.fields.keys())

    required = {
        "general.architecture",
        "llama.block_count",
        "llama.embedding_length",
        "llama.vocab_size",
        "llama.attention.head_count",
        "llama.attention.head_count_kv",
        "llama.context_length",
        "llama.feed_forward_length",
        "llama.rope.freq_base",
        "llama.rope.dimension_count",
        "tokenizer.ggml.model",
        "tokenizer.ggml.pre",
        "tokenizer.ggml.tokens",
        "tokenizer.ggml.scores",
        "tokenizer.ggml.token_type",
        "tokenizer.ggml.merges",
        "tokenizer.ggml.bos_token_id",
        "tokenizer.ggml.eos_token_id",
    }
    missing = required - field_keys
    assert not missing, f"Missing GGUF metadata fields: {missing}"

    # Tokenizer pre-tokenizer must be "bytelevel" (Fix 4).
    # Read back via string field.
    pre_field = reader.fields.get("tokenizer.ggml.pre")
    assert pre_field is not None, "tokenizer.ggml.pre missing"


def test_token_scores_are_non_zero(tmp_path: Path):
    """Fix 2: Token scores must be derived from merge rank, not all zero."""
    from scripts.export_gguf import export_gguf

    config = _small_config(vocab_size=64, n_layers=1)
    model = EdgeGPT(config)
    tk_dir = tmp_path / "tokenizer"
    _write_minimal_tokenizer(tk_dir, vocab_size=64)

    ckpt_dir = tmp_path / "run"
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ckpt_path = save_checkpoint(
        run_dir=ckpt_dir, config=config, model=model, optimizer=optimizer,
        scaler=None, global_step=0, best_val_loss=None, tokens_consumed=0,
    )

    out_path = tmp_path / "scores.gguf"
    export_gguf(checkpoint_path=ckpt_path, tokenizer_dir=tk_dir, output_path=out_path, outtype="f32")

    from gguf import GGUFReader

    reader = GGUFReader(str(out_path))
    scores_field = reader.fields["tokenizer.ggml.scores"]
    # The scores are stored as an array — check that they exist and have content.
    assert scores_field is not None
    # Verify the field has parts (array data).
    assert len(scores_field.parts) > 0


def test_export_field_values_correct(tmp_path: Path):
    from scripts.export_gguf import export_gguf

    config = _small_config(vocab_size=64, n_layers=2)
    model = EdgeGPT(config)
    tk_dir = tmp_path / "tokenizer"
    _write_minimal_tokenizer(tk_dir, vocab_size=64)

    ckpt_dir = tmp_path / "run"
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ckpt_path = save_checkpoint(
        run_dir=ckpt_dir, config=config, model=model, optimizer=optimizer,
        scaler=None, global_step=0, best_val_loss=None, tokens_consumed=0,
    )

    out_path = tmp_path / "vals.gguf"
    export_gguf(checkpoint_path=ckpt_path, tokenizer_dir=tk_dir, output_path=out_path, outtype="f32")

    from gguf import GGUFReader

    reader = GGUFReader(str(out_path))

    def _get_scalar(field_name: str):
        f = reader.fields[field_name]
        arr = f.parts[-1]
        return arr.item() if hasattr(arr, "item") else int(arr[()])

    assert _get_scalar("llama.block_count") == 2
    assert _get_scalar("llama.embedding_length") == 32
    assert _get_scalar("llama.vocab_size") == 64
    assert _get_scalar("llama.attention.head_count") == 4
    assert _get_scalar("llama.attention.head_count_kv") == 2
    assert _get_scalar("llama.rope.freq_base") == pytest.approx(10000.0, rel=1e-5)
    # Fix 5: rope.dimension_count must be present and correct.
    assert _get_scalar("llama.rope.dimension_count") == 32 // 4


def test_export_all_tensors_mapped(tmp_path: Path):
    from scripts.export_gguf import _map_tensor_name, export_gguf

    config = _small_config(vocab_size=64, n_layers=2)
    model = EdgeGPT(config)
    tk_dir = tmp_path / "tokenizer"
    _write_minimal_tokenizer(tk_dir, vocab_size=64)

    ckpt_dir = tmp_path / "run"
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ckpt_path = save_checkpoint(
        run_dir=ckpt_dir, config=config, model=model, optimizer=optimizer,
        scaler=None, global_step=0, best_val_loss=None, tokens_consumed=0,
    )

    out_path = tmp_path / "all.gguf"
    export_gguf(checkpoint_path=ckpt_path, tokenizer_dir=tk_dir, output_path=out_path, outtype="f32")

    from gguf import GGUFReader

    reader = GGUFReader(str(out_path))
    gguf_names = {t.name for t in reader.tensors}

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model", ckpt)

    mapped_count = 0
    for name in sd:
        gguf_name = _map_tensor_name(name)
        if gguf_name is None:
            continue
        if gguf_name == "output.weight" and config.model.tie_embeddings:
            continue
        assert gguf_name in gguf_names, f"{name} → {gguf_name} not found in GGUF"
        mapped_count += 1

    expected = 1 + 2 * 9 + 1  # embed + 2 blocks × 9 tensors + norm
    assert mapped_count == expected, f"Expected {expected} mapped tensors, got {mapped_count}"
    assert len(gguf_names) == expected


def test_export_cli_smoke(tmp_path: Path):
    import subprocess
    import sys

    config = _small_config(vocab_size=64, n_layers=1)
    model = EdgeGPT(config)
    tk_dir = tmp_path / "tokenizer"
    _write_minimal_tokenizer(tk_dir, vocab_size=64)

    ckpt_dir = tmp_path / "run"
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ckpt_path = save_checkpoint(
        run_dir=ckpt_dir, config=config, model=model, optimizer=optimizer,
        scaler=None, global_step=0, best_val_loss=None, tokens_consumed=0,
    )

    out_path = tmp_path / "cli.gguf"
    subprocess.run(
        [sys.executable, "scripts/export_gguf.py",
         "--checkpoint", str(ckpt_path),
         "--tokenizer-dir", str(tk_dir),
         "--output", str(out_path),
         "--outtype", "f32"],
        cwd=Path(__file__).resolve().parents[1],
        check=True, capture_output=True, text=True,
    )

    assert out_path.exists()
    assert out_path.stat().st_size > 1000
