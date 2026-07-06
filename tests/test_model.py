"""Tests for Phase 9 full model assembly, loss, and overfit milestone."""

from __future__ import annotations

import math

import pytest
import torch

from configs.config import Config, DataConfig, ModelConfig, TokenizerConfig, TrainingConfig
from model import EdgeGPT


def _small_config(
    *,
    vocab_size: int = 128,
    d_model: int = 64,
    n_layers: int = 2,
    n_heads: int = 4,
    n_kv_heads: int = 1,
    d_ff: int = 176,
    max_seq_len: int = 64,
    tie_embeddings: bool = True,
    chunked_loss: bool = False,
) -> Config:
    """Return a lightweight config that exercises the full architecture."""
    return Config(
        model=ModelConfig(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            d_ff=d_ff,
            max_seq_len=max_seq_len,
            tie_embeddings=tie_embeddings,
        ),
        tokenizer=TokenizerConfig(vocab_size=vocab_size, reserved_special_tokens=16),
        training=TrainingConfig(chunked_loss=chunked_loss),
        data=DataConfig(),
        device="cpu",
    )


# ═══════════════════════════════════════════════════════════════════════
# Shape contracts
# ═══════════════════════════════════════════════════════════════════════


def test_forward_logits_shape():
    model = EdgeGPT(_small_config())
    input_ids = torch.randint(0, 128, (2, 16), dtype=torch.long)

    logits, loss = model(input_ids)

    assert logits is not None
    assert logits.shape == (2, 16, 128)
    assert loss is None


def test_forward_with_targets_returns_scalar_loss():
    model = EdgeGPT(_small_config())
    input_ids = torch.randint(0, 128, (2, 16), dtype=torch.long)
    targets = torch.randint(0, 128, (2, 16), dtype=torch.long)

    logits, loss = model(input_ids, targets)

    assert logits is not None
    assert logits.shape == (2, 16, 128)
    assert loss is not None
    assert loss.ndim == 0  # scalar


def test_loss_is_none_when_targets_is_none():
    model = EdgeGPT(_small_config())
    input_ids = torch.randint(0, 128, (2, 16), dtype=torch.long)

    _, loss = model(input_ids, targets=None)

    assert loss is None


def test_chunked_loss_returns_none_logits():
    """When chunked_loss is active, logits should be None (never materialised)."""
    model = EdgeGPT(_small_config(chunked_loss=True))
    model.eval()
    input_ids = torch.randint(0, 128, (2, 16), dtype=torch.long)
    targets = torch.randint(0, 128, (2, 16), dtype=torch.long)

    with torch.no_grad():
        logits, loss = model(input_ids, targets)

    assert logits is None, "Chunked-loss path should not materialise logits"
    assert loss is not None


def test_forward_respects_position_offset():
    """position_offset should pass through to RoPE, rotating Q/K vectors.

    We verify at the intermediate level because tiny test models with
    random inputs may produce near-identical final logits even when
    RoPE is correctly rotating the per-position Q/K values.
    """
    model = EdgeGPT(_small_config())
    model.eval()
    input_ids = torch.randint(0, 128, (1, 8), dtype=torch.long)

    # Compare Q/K after the first block's attention projection.
    block = model.layers[0]
    normed = block.attention_norm(model.embed_tokens(input_ids))

    q0, k0, _ = block.attention.project_qkv(normed, position_offset=0)
    q5, k5, _ = block.attention.project_qkv(normed, position_offset=5)

    # Different position offsets produce different RoPE rotations.
    assert not torch.allclose(q0, q5), "position_offset=0 vs 5 should differ on Q"
    assert not torch.allclose(k0, k5), "position_offset=0 vs 5 should differ on K"


# ═══════════════════════════════════════════════════════════════════════
# Initial loss sanity — loss ≈ ln(vocab_size)
# ═══════════════════════════════════════════════════════════════════════


def test_initial_loss_close_to_ln_vocab():
    torch.manual_seed(42)
    model = EdgeGPT(_small_config(vocab_size=128))
    model.eval()
    input_ids = torch.randint(0, 128, (4, 32), dtype=torch.long)
    targets = torch.randint(0, 128, (4, 32), dtype=torch.long)

    with torch.no_grad():
        _, loss = model(input_ids, targets)

    expected = math.log(128)  # ≈ 4.852
    assert abs(loss.item() - expected) < 0.5, (
        f"Initial loss {loss.item():.4f} deviates too far from expected {expected:.4f}"
    )


def test_initial_loss_scales_with_vocab_size():
    torch.manual_seed(42)
    small = EdgeGPT(_small_config(vocab_size=64))
    large = EdgeGPT(_small_config(vocab_size=256))
    small.eval()
    large.eval()
    input_ids = torch.randint(0, 64, (2, 16), dtype=torch.long)
    targets = torch.randint(0, 64, (2, 16), dtype=torch.long)

    with torch.no_grad():
        _, loss_small = small(input_ids, targets)
    # For the larger model, use token ids valid for its vocab.
    input_ids_l = torch.randint(0, 256, (2, 16), dtype=torch.long)
    targets_l = torch.randint(0, 256, (2, 16), dtype=torch.long)
    with torch.no_grad():
        _, loss_large = large(input_ids_l, targets_l)

    # Larger vocab → higher initial loss
    assert loss_large.item() > loss_small.item()


# ═══════════════════════════════════════════════════════════════════════
# Overfit single batch — the definitive architecture correctness test
# ═══════════════════════════════════════════════════════════════════════


def test_overfit_single_batch():
    """Overfit a fixed batch to < 0.1 loss — proves the architecture works.

    This is the exit-criterion test from plan.md Phase 9.  If it fails,
    there is a fundamental bug in the model assembly, data flow, or
    gradient plumbing that no amount of hyper-parameter tuning will fix.
    """
    torch.manual_seed(42)

    # Use a small-but-real config so the test runs quickly on CPU.
    config = _small_config(
        vocab_size=128,
        d_model=96,
        n_layers=4,
        n_heads=4,
        n_kv_heads=2,
        d_ff=256,
    )
    model = EdgeGPT(config)
    model.train()

    # Fixed batch — intentionally not shuffled.
    B, T, V = 4, 32, config.model.vocab_size
    input_ids = torch.randint(0, V, (B, T), dtype=torch.long)
    targets = torch.randint(0, V, (B, T), dtype=torch.long)

    # ── Step 1: initial-loss sanity ──────────────────────────────────
    with torch.no_grad():
        _, init_loss = model(input_ids, targets)
    expected = math.log(V)
    assert abs(init_loss.item() - expected) < 0.5, (
        f"Initial loss {init_loss.item():.4f} not near ln({V}) = {expected:.4f}"
    )

    # ── Step 2: overfit training ─────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    num_steps = 600
    final_loss = float("inf")

    for step in range(num_steps):
        _, loss = model(input_ids, targets)
        assert loss is not None

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        final_loss = loss.item()

    assert final_loss < 0.10, (
        f"Overfit test FAILED: loss {final_loss:.4f} did not drop below 0.10 "
        f"after {num_steps} steps.  Check model assembly, residual paths, "
        f"and gradient flow."
    )


# ═══════════════════════════════════════════════════════════════════════
# Weight tying
# ═══════════════════════════════════════════════════════════════════════


def test_tied_weights_share_storage():
    config = _small_config(tie_embeddings=True)
    model = EdgeGPT(config)

    emb_ptr = model.embed_tokens.weight.data_ptr()
    out_ptr = model.lm_head.weight.data_ptr()

    assert emb_ptr == out_ptr


def test_untied_weights_allocate_separately():
    config = _small_config(tie_embeddings=False)
    model = EdgeGPT(config)

    emb_ptr = model.embed_tokens.weight.data_ptr()
    out_ptr = model.lm_head.weight.data_ptr()

    assert emb_ptr != out_ptr


# ═══════════════════════════════════════════════════════════════════════
# Gradient flow
# ═══════════════════════════════════════════════════════════════════════


def test_gradients_flow_to_all_parameters():
    model = EdgeGPT(_small_config())
    model.train()
    input_ids = torch.randint(0, 128, (2, 16), dtype=torch.long)
    targets = torch.randint(0, 128, (2, 16), dtype=torch.long)

    _, loss = model(input_ids, targets)
    assert loss is not None
    loss.backward()

    untrained = []
    for name, param in model.named_parameters():
        if param.grad is None:
            untrained.append(name)
        elif param.grad.abs().sum() == 0:
            untrained.append(name)

    assert len(untrained) == 0, (
        f"Parameters with zero/no gradient: {untrained}"
    )


# ═══════════════════════════════════════════════════════════════════════
# Initialisation
# ═══════════════════════════════════════════════════════════════════════


def test_configure_initialization_is_idempotent_in_distribution():
    """Calling configure_initialization twice produces the same distribution.

    Individual weights differ (random sampling), but the std of each
    residual projection should remain at the depth-scaled target both times.
    """
    n_layers = 4
    config = _small_config(n_layers=n_layers)
    model = EdgeGPT(config)
    scaled_std = config.model.initializer_range / math.sqrt(2 * n_layers)

    # First call (already applied in __init__).
    for pn, p in model.named_parameters():
        if pn.endswith("o_proj.weight") or pn.endswith("down_proj.weight"):
            assert abs(p.std().item() - scaled_std) < 0.002, (
                f"First init: {pn} std {p.std().item():.6f} not near {scaled_std:.6f}"
            )

    # Second call should produce the same distribution.
    model.configure_initialization()
    for pn, p in model.named_parameters():
        if pn.endswith("o_proj.weight") or pn.endswith("down_proj.weight"):
            assert abs(p.std().item() - scaled_std) < 0.002, (
                f"Second init: {pn} std {p.std().item():.6f} not near {scaled_std:.6f}"
            )


def test_residual_projections_have_depth_scaled_std():
    """o_proj and down_proj should start with std ≈ initializer_range / √(2N)."""
    n_layers = 8
    config = _small_config(n_layers=n_layers)
    model = EdgeGPT(config)

    scaled_std = config.model.initializer_range / math.sqrt(2 * n_layers)

    residual_stds = []
    for pn, p in model.named_parameters():
        if pn.endswith("o_proj.weight") or pn.endswith("down_proj.weight"):
            residual_stds.append(p.std().item())

    # Check every residual projection is close to the expected scaled std.
    for std_val in residual_stds:
        assert abs(std_val - scaled_std) < 0.002, (
            f"Residual projection std {std_val:.6f} not close to expected {scaled_std:.6f}"
        )


def test_non_residual_linears_have_base_std():
    """Q/K/V/gate/up projections should start with std ≈ initializer_range."""
    config = _small_config()
    model = EdgeGPT(config)
    base_std = config.model.initializer_range

    non_residual_stds = []
    for pn, p in model.named_parameters():
        if (
            pn.endswith(".weight")
            and "o_proj" not in pn
            and "down_proj" not in pn
            and "norm" not in pn
            and "embedding" not in pn
            and "embed_tokens" not in pn
        ):
            non_residual_stds.append(p.std().item())

    for std_val in non_residual_stds:
        assert abs(std_val - base_std) < 0.008, (
            f"Non-residual linear std {std_val:.6f} not close to {base_std}"
        )


def test_rmsnorm_gains_are_ones_at_init():
    model = EdgeGPT(_small_config())

    norm_weights = []
    for pn, p in model.named_parameters():
        if "norm" in pn and "weight" in pn:
            norm_weights.append(p)

    for w in norm_weights:
        assert torch.allclose(w, torch.ones_like(w)), (
            f"RMSNorm weight not initialised to ones"
        )


# ═══════════════════════════════════════════════════════════════════════
# Parameter counting
# ═══════════════════════════════════════════════════════════════════════


def test_count_parameters_is_deterministic():
    model = EdgeGPT(_small_config())
    first = model.count_parameters()
    second = model.count_parameters()

    assert first == second


def test_count_parameters_sums_to_total():
    model = EdgeGPT(_small_config(tie_embeddings=True))
    counts = model.count_parameters()

    component_sum = (
        counts["embed_tokens"]
        + counts["layers"]
        + counts["norm"]
        + counts["lm_head"]
    )
    assert component_sum == counts["total"]


def test_count_parameters_total_matches_pytorch():
    model = EdgeGPT(_small_config())
    expected = sum(p.numel() for p in model.parameters())

    counts = model.count_parameters()

    assert counts["total"] == expected


def test_tied_output_projection_is_zero_params():
    model = EdgeGPT(_small_config(tie_embeddings=True))
    assert model.count_parameters()["lm_head"] == 0


def test_untied_output_projection_has_params():
    model = EdgeGPT(_small_config(tie_embeddings=False))
    assert model.count_parameters()["lm_head"] > 0


# ═══════════════════════════════════════════════════════════════════════
# Memory estimation
# ═══════════════════════════════════════════════════════════════════════


def test_estimate_memory_returns_positive_values():
    model = EdgeGPT(_small_config())
    mem = model.estimate_memory(batch_size=4, seq_len=32)

    assert mem["param_mem_mib"] > 0
    assert mem["act_mem_mib"] > 0
    assert mem["total_mib"] > 0
    assert "param_dtype" in mem
    assert "act_dtype" in mem


def test_estimate_memory_grows_with_batch_size():
    model = EdgeGPT(_small_config())
    small = model.estimate_memory(batch_size=2, seq_len=32)
    large = model.estimate_memory(batch_size=8, seq_len=32)

    assert abs(small["param_mem_mib"] - large["param_mem_mib"]) < 1e-6  # params don't change
    assert large["act_mem_mib"] > small["act_mem_mib"]  # activations do


def test_estimate_memory_respects_dtype_override():
    """fp32 should report ~2× the memory of fp16/bf16 for the same model."""
    model = EdgeGPT(_small_config())
    mem_fp32 = model.estimate_memory(
        batch_size=4, seq_len=32, param_dtype="fp32", act_dtype="fp32"
    )
    mem_bf16 = model.estimate_memory(
        batch_size=4, seq_len=32, param_dtype="bf16", act_dtype="bf16"
    )

    # fp32 params ≈ 2× bf16 params.
    assert mem_fp32["param_mem_mib"] > mem_bf16["param_mem_mib"] * 1.5
    # Activations should also be ~2× for fp32 vs bf16.
    assert mem_fp32["act_mem_mib"] > mem_bf16["act_mem_mib"] * 1.5


def test_estimate_memory_reads_dtype_from_config():
    """When no override is given, estimate_memory uses config.training.dtype."""
    config_fp32 = _small_config()
    config_fp32.training.dtype = "fp32"
    model_fp32 = EdgeGPT(config_fp32)

    config_bf16 = _small_config()
    config_bf16.training.dtype = "bf16"
    model_bf16 = EdgeGPT(config_bf16)

    mem_fp32 = model_fp32.estimate_memory(batch_size=4, seq_len=32)
    mem_bf16 = model_bf16.estimate_memory(batch_size=4, seq_len=32)

    # fp32 config should report higher memory than bf16 config.
    assert mem_fp32["total_mib"] > mem_bf16["total_mib"]


# ═══════════════════════════════════════════════════════════════════════
# Chunked loss
# ═══════════════════════════════════════════════════════════════════════


def test_chunked_loss_close_to_standard():
    torch.manual_seed(42)
    config = _small_config()
    model_std = EdgeGPT(config)
    model_std.eval()

    # Clone weights so both models start with identical parameters.
    config_chunked = _small_config(chunked_loss=True)
    model_chunked = EdgeGPT(config_chunked)
    model_chunked.load_state_dict(model_std.state_dict())
    model_chunked.eval()

    input_ids = torch.randint(0, 128, (3, 20), dtype=torch.long)
    targets = torch.randint(0, 128, (3, 20), dtype=torch.long)

    with torch.no_grad():
        _, loss_std = model_std(input_ids, targets)
        _, loss_chunked = model_chunked(input_ids, targets)

    assert loss_std is not None
    assert loss_chunked is not None
    assert torch.allclose(loss_std, loss_chunked, atol=1e-5), (
        f"Standard {loss_std.item():.6f} vs chunked {loss_chunked.item():.6f}"
    )


def test_chunked_loss_handles_ignore_index():
    """Chunked loss denominator should exclude -100 padding tokens."""
    torch.manual_seed(42)
    config = _small_config(chunked_loss=True)
    model = EdgeGPT(config)
    model.eval()

    input_ids = torch.randint(0, 128, (2, 16), dtype=torch.long)
    targets = torch.randint(0, 128, (2, 16), dtype=torch.long)
    # Set half the targets to -100 (ignore).
    targets[:, 8:] = -100

    with torch.no_grad():
        _, loss_chunked = model(input_ids, targets)

    assert loss_chunked is not None
    # Loss should be finite — not nan or inf.
    assert torch.isfinite(loss_chunked)


def test_chunked_loss_matches_standard_with_ignore_index():
    """With ignore_index=-100 tokens, both paths should agree."""
    torch.manual_seed(42)
    config = _small_config()
    model_std = EdgeGPT(config)
    model_std.eval()

    config_chunked = _small_config(chunked_loss=True)
    model_chunked = EdgeGPT(config_chunked)
    model_chunked.load_state_dict(model_std.state_dict())
    model_chunked.eval()

    input_ids = torch.randint(0, 128, (3, 16), dtype=torch.long)
    targets = torch.randint(0, 128, (3, 16), dtype=torch.long)
    targets[:, -4:] = -100  # ignore last 4 positions

    with torch.no_grad():
        _, loss_std = model_std(input_ids, targets)
        _, loss_chunked = model_chunked(input_ids, targets)

    assert loss_std is not None and loss_chunked is not None
    assert torch.allclose(loss_std, loss_chunked, atol=1e-5), (
        f"Standard {loss_std.item():.6f} vs chunked {loss_chunked.item():.6f}"
    )


# ═══════════════════════════════════════════════════════════════════════
# Config compatibility
# ═══════════════════════════════════════════════════════════════════════


def test_cpu_config_compatibility():
    from configs.config import load_config

    config = load_config("configs/cpu.yaml")
    model = EdgeGPT(config)
    input_ids = torch.randint(0, config.model.vocab_size, (2, 16), dtype=torch.long)

    logits, _ = model(input_ids)

    assert logits is not None
    assert logits.shape == (2, 16, config.model.vocab_size)


def test_default_config_compatibility():
    from configs.config import load_config

    config = load_config("configs/default.yaml")
    model = EdgeGPT(config)
    input_ids = torch.randint(0, config.model.vocab_size, (2, 16), dtype=torch.long)

    logits, _ = model(input_ids)

    assert logits is not None
    assert logits.shape == (2, 16, config.model.vocab_size)


# ═══════════════════════════════════════════════════════════════════════
# Edge cases & validation
# ═══════════════════════════════════════════════════════════════════════


def test_model_rejects_non_long_inputs():
    model = EdgeGPT(_small_config())

    with pytest.raises(TypeError, match="torch.long"):
        model(torch.tensor([[1.0, 2.0]]))


def test_model_rejects_wrong_vocab_token():
    model = EdgeGPT(_small_config(vocab_size=64))

    with pytest.raises((IndexError, RuntimeError)):
        model(torch.tensor([[99]], dtype=torch.long))


def test_device_property():
    model = EdgeGPT(_small_config())
    assert isinstance(model.device, torch.device)


def test_attention_mask_passes_through():
    model = EdgeGPT(_small_config())
    input_ids = torch.randint(0, 128, (2, 16), dtype=torch.long)
    mask = torch.ones(2, 16, dtype=torch.long)
    mask[:, -4:] = 0  # mask last 4 positions

    logits, _ = model(input_ids, attention_mask=mask)

    assert logits is not None
    assert logits.shape == (2, 16, 128)


def test_model_does_not_create_training_modules():
    model = EdgeGPT(_small_config())
    names = set(dict(model.named_modules()))

    forbidden = {"optimizer", "scheduler", "data_loader", "tokenizer"}
    assert forbidden.isdisjoint(names)


def test_final_norm_is_independent_from_block_norms():
    model = EdgeGPT(_small_config(n_layers=2))

    # model.norm should not share storage with any block's norm.
    for block in model.layers:
        assert model.norm.weight.data_ptr() != block.attention_norm.weight.data_ptr()
        assert model.norm.weight.data_ptr() != block.mlp_norm.weight.data_ptr()


def test_layers_are_separate_instances():
    model = EdgeGPT(_small_config(n_layers=4))

    # Every block should be a distinct Module with its own parameters.
    for i in range(len(model.layers)):
        for j in range(len(model.layers)):
            if i != j:
                assert model.layers[i] is not model.layers[j]


# ═══════════════════════════════════════════════════════════════════════
# Device portability — CPU / CUDA / MPS
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_model_runs_on_cuda():
    """Forward + backward pass on CUDA with full default config."""
    from configs.config import load_config

    config = load_config("configs/default.yaml")
    model = EdgeGPT(config).cuda()
    model.train()

    input_ids = torch.randint(0, config.model.vocab_size, (2, 64), device="cuda")
    targets = torch.randint(0, config.model.vocab_size, (2, 64), device="cuda")

    logits, loss = model(input_ids, targets)
    assert logits is not None
    assert logits.shape == (2, 64, config.model.vocab_size)
    assert loss is not None
    assert loss.device.type == "cuda"

    loss.backward()
    # Spot-check: at least one parameter gradient is non-zero on CUDA.
    assert model.embed_tokens.weight.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_model_moves_to_cuda_and_back():
    """Model should survive device round-trips without corruption."""
    model = EdgeGPT(_small_config())
    model.train()

    # Forward on CPU first.
    input_ids = torch.randint(0, 128, (1, 8), dtype=torch.long)
    _, loss_cpu = model(input_ids, input_ids.clone())

    # Move to CUDA and back.
    model.cuda()
    model.cpu()

    _, loss_cpu2 = model(input_ids, input_ids.clone())
    assert loss_cpu is not None and loss_cpu2 is not None
    assert torch.allclose(loss_cpu, loss_cpu2, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_chunked_loss_parity_on_cuda():
    """Chunked loss must match standard loss on CUDA."""
    torch.manual_seed(42)
    config = _small_config()
    model = EdgeGPT(config).cuda()
    model.eval()

    config_chunked = _small_config(chunked_loss=True)
    model_chunked = EdgeGPT(config_chunked).cuda()
    model_chunked.load_state_dict(model.state_dict())
    model_chunked.eval()

    input_ids = torch.randint(0, 128, (3, 20), device="cuda")
    targets = torch.randint(0, 128, (3, 20), device="cuda")

    with torch.no_grad():
        _, loss_std = model(input_ids, targets)
        _, loss_chunked = model_chunked(input_ids, targets)

    assert loss_std is not None and loss_chunked is not None
    assert torch.allclose(loss_std, loss_chunked, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_default_config_runs_on_cuda():
    """Full default config model must run forward on CUDA."""
    from configs.config import load_config

    config = load_config("configs/default.yaml")
    model = EdgeGPT(config).cuda()
    model.eval()

    input_ids = torch.randint(0, config.model.vocab_size, (2, 64), device="cuda")

    with torch.no_grad():
        logits, _ = model(input_ids)

    assert logits is not None
    assert logits.shape == (2, 64, config.model.vocab_size)
    assert logits.device.type == "cuda"


@pytest.mark.skipif(torch.cuda.is_available(), reason="CPU-only test — skip when CUDA present")
def test_device_property_returns_cpu_when_no_gpu():
    """When CUDA is absent, model.device should be 'cpu'."""
    model = EdgeGPT(_small_config())
    assert model.device.type == "cpu"
