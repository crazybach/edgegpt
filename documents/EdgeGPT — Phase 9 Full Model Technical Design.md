# EdgeGPT Phase 9 — Full Model (Logits, Loss, Overfit Test) Technical Design

## Summary

Phase 9 assembles all previously built primitives (TokenEmbedding, RMSNorm,
CausalSelfAttention with RoPE, SwiGLU MLP, TransformerBlock, OutputProjection)
into a complete, end-to-end forward pass. The model takes token IDs `[B, T]` and
produces logits `[B, T, vocab_size]`, then computes scalar cross-entropy loss
against shifted targets.

This phase also implements the model-wide initialization scheme, a parameter
counting utility, and the two non-negotiable correctness milestones from
`plan.md`:

1. **Initial loss sanity check** — at random init, loss ≈ ln(vocab_size)
2. **Single-batch overfit to ~0 loss** — the definitive proof the whole
   architecture learns

---

## Direction Decision

### 1. Model Assembly Pattern

| Direction | Status | Decision |
| --- | --- | --- |
| Sequential composition (Llama/GPT-2) | Industry standard for dense decoder models | **Used now** |
| Parallel attention + MLP (PaLM / GPT-J) | Reduces per-block depth, trades throughput | Deferred |
| Pipeline / tensor-parallel assembly | Large-cluster multi-GPU training | Not applicable to laptop scale |

**Sequential composition** is the only choice that aligns with the Llama export
path. Parallel residual variants (PaLM-style) change the block semantics and
would break the `convert_hf_to_gguf.py` conversion path in Phase 12.

### 2. Weight Initialization Scheme

| Direction | Status | Decision |
| --- | --- | --- |
| GPT-2 / Llama: N(0, 0.02), c_proj / √(2N) | Llama, GPT-2, Mistral, Qwen standard | **Used now** |
| SmallInit (reduced attention std) | Post-norm transformers | Rejected — we use pre-norm |
| DeepNet: β = (8N)^(−1/4) | 100+ layer networks | Rejected — overkill for 8 layers |
| WeSaR: gate-parameterized uniform σ | 13B+ models with loss spike risk | Rejected — complexity not needed |
| Admin: ω_init = √(log N / N) | Deep transformers (30+ layers) | Rejected — overkill for 8 layers |
| DS-Init: per-layer variance scaling | Very deep (50+ layers) | Rejected — overkill for 8 layers |

**GPT-2 / Llama-style N(0, 0.02) with residual projection scaling by 1/√(2N)**
is the clear winner for EdgeGPT. This is the initialization used by Llama, GPT-2,
Mistral, Qwen, and nanoGPT. For an 8-layer model, the residual scaling factor is
`1/√(2 × 8) = 1/4 = 0.25`, which means `c_proj`/`o_proj`/`down_proj` std is
`0.02 × 0.25 = 0.005`.

The more advanced schemes (DeepNet, WeSaR, Admin, DS-Init) target 30–1000 layer
networks or multi-billion-parameter models. They add complexity with negligible
benefit at our scale. WeSaR is mathematically appealing but introduces gate
parameters that would need special handling during GGUF export.

### 3. Loss Computation

| Direction | Status | Decision |
| --- | --- | --- |
| Standard cross-entropy (full logits) | PyTorch `F.cross_entropy` | **Used now (default)** |
| Chunked cross-entropy (tiled over seq) | Memory-efficient for large vocab | Config-gated fallback for CPU |
| Fused linear + CE (Liger-Kernel) | LinkedIn, 2024 | Deferred — external dependency |
| Cut Cross-Entropy (Apple) | ICLR 2025 Oral, Triton kernel | Deferred — requires Triton + Ampere+ |
| Label smoothing | GPT/LLaMA training | Deferred — training-policy choice |

**Standard cross-entropy is the default** because our vocabulary (16,384) is
small enough that the full logits tensor is manageable:

| Scenario | Logits shape [BT, V] | Memory (fp32) | Memory (bf16) |
| --- | --- | --- | --- |
| Default (B=8, T=2048) | [16,384, 16,384] | 1.0 GB | 536 MB |
| CPU config (B=4, T=512) | [2,048, 16,384] | 134 MB | 67 MB |
| Overfit test (B=4, T=128) | [512, 16,384] | 33 MB | 17 MB |

The default scenario uses ~536 MB in bf16 mixed precision, which fits within an
8 GB laptop GPU alongside the model and optimizer states. The **chunked loss**
option gates a tiled computation that never materializes the full `[BT, V]`
tensor — essential for the CPU config or if we later expand vocabulary.

Apple's Cut Cross-Entropy and Liger-Kernel are excellent innovations but require
either Triton 3.0+ with Ampere-class GPUs or add external dependencies. We keep
them as deferred optimizations for Phase 10 (training loop).

### 4. Overfit Test Methodology

| Direction | Status | Decision |
| --- | --- | --- |
| Initial loss sanity: loss ≈ ln(vocab_size) | Karpathy recipe, universal practice | **Used now** |
| Single-batch overfit to ~0 loss | Karpathy recipe, plan.md milestone | **Used now** |
| Gradient flow check (all params) | Debugging utility | **Used now** |
| Automated checklist (neural_net_checklist) | External package | Rejected — keep tests in-project |

The overfit test is the definitive Phase 9 exit criterion. For an LLM, the
expected initial loss is `ln(16384) ≈ 9.70` nats. With sufficient training on one
fixed batch, the model should drive loss well below 1.0 (target: < 0.1). The
higher threshold compared to classification (which targets < 1e-4) accounts for
the inherent difficulty of exact next-token prediction over 16k classes.

---

## Production Context

All major open-source models follow a similar assembly pattern for Phase 9:

- **Llama 3 (Meta, 2024)**: `_init_weights` uses `normal_(mean=0.0,
  std=initializer_range)` where `initializer_range=0.02`. Residual projections
  (`o_proj`, `down_proj`) are NOT explicitly re-scaled in the HuggingFace
  implementation — Llama 3 relies on pre-norm architecture and the training
  schedule (linear warmup, AdamW) for stability. However, the original GPT-2 and
  nanoGPT patterns DO re-scale residual projections by `1/√(2N)`, which is the
  more conservative choice and what Llama 1/2 used.

- **GPT-2 / nanoGPT (Karpathy)**: All weights N(0, 0.02), biases zero. The
  `c_proj` (output projection) is re-initialized with std `0.02 / √(2 × n_layer)`.
  This is the pattern we adopt.

- **Mistral 7B**: Follows Llama initialization conventions. Uses
  `initializer_range=0.02` with RMSNorm gains initialized to 1.0.

- **Qwen 2.5**: Same Llama-family initialization. Confirms N(0, 0.02) + zero
  bias + ones-init for norm gains as the industry default for dense decoder
  models.

- **Gemma 2**: Uses `embedding_scale = √d_model` (multiplies token embeddings by
  `√d_model` before entering the transformer). EdgeGPT already supports this via
  the `embedding_scale` config field; the default is `null` (Llama-style no
  scaling).

- **DeepSeek V3**: Uses FP8 mixed precision with adaptive gradient scaling for
  the 671B MoE. The initialization complexity is driven by MoE routing and
  extreme depth — not applicable to our dense 8-layer baseline. Notably, they
  report zero irrecoverable loss spikes throughout training, attributed to
  careful initialization and pre-norm.

References:

- GPT-2 weight init / residual scaling:
  https://github.com/karpathy/nanoGPT/blob/master/model.py
- DeepNet / DeepNorm (very deep transformers):
  https://arxiv.org/abs/2203.00555
- SmallInit (post-norm transformers):
  https://arxiv.org/abs/1910.05895
- WeSaR (gate-parameterized init for uniform update ratios):
  https://aclanthology.org/2024.emnlp-main.1264
- DeepSeek V3 technical report:
  https://arxiv.org/abs/2412.19437
- Cut Cross-Entropy (Apple, memory-efficient loss):
  https://arxiv.org/abs/2411.09009
- Liger-Kernel (fused linear + CE):
  https://github.com/linkedin/Liger-Kernel
- Karpathy "A Recipe for Training Neural Networks":
  https://karpathy.github.io/2019/04/25/recipe/

---

## Implementation

### Files

- `model/model.py` — new. Exposes `EdgeGPT(config)`.
- `model/__init__.py` — updated. Re-exports `EdgeGPT`.
- `tests/test_model.py` — new. Full test suite including overfit test.

### `model/model.py` — `EdgeGPT` class

```python
class EdgeGPT(nn.Module):
    """Llama-compatible decoder-only transformer.

    Forward chain:
        input_ids [B, T]
        → TokenEmbedding
        → (optional embed scale)
        → N × TransformerBlock
        → final RMSNorm
        → OutputProjection (tied to embeddings)
        → logits [B, T, vocab_size]
        → cross-entropy loss (if targets provided)
    """

    def __init__(self, config: Config):
        ...

    def forward(
        self,
        input_ids: torch.LongTensor,
        targets: torch.LongTensor | None = None,
        *,
        attention_mask: torch.Tensor | None = None,
        use_manual_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return (logits, loss). Loss is None when targets is None."""
        ...

    def configure_initialization(self) -> None:
        """Apply GPT-2/Llama-style weight initialization."""
        ...

    def count_parameters(self) -> dict[str, int]:
        """Return parameter counts by component for capacity planning."""
        ...

    def estimate_memory(self, batch_size: int, seq_len: int) -> dict[str, float]:
        """Estimate activation and parameter memory in MiB."""
        ...
```

### Initialization Rules (`configure_initialization`)

| Parameter type | Initialization |
| --- | --- |
| `nn.Linear` weight (not residual proj) | `N(0, 0.02)` |
| `nn.Linear` bias (if present) | Zeros |
| `nn.Embedding` weight | `N(0, 0.02)` |
| Residual projection (`o_proj`, `down_proj`) | `N(0, 0.02 / √(2 × n_layers))` |
| RMSNorm `weight` (gain) | Ones (already set in `__init__`) |
| All other parameters | PyTorch defaults (no override) |

Residual projections are identified by parameter name suffix:
- `o_proj.weight` (attention output projection)
- `down_proj.weight` (MLP down projection)

These are the linear layers whose output feeds directly into a residual addition.
Scaling them down by `1/√(2N)` prevents the residual stream variance from growing
with depth, which is especially important at initialization before the optimizer
adapts.

For EdgeGPT's default config (N=8): `0.02 / √(16) = 0.02 / 4 = 0.005`.

### Forward Pass Pseudocode

```python
def forward(self, input_ids, targets=None, *, attention_mask=None,
            use_manual_attention=False):
    B, T = input_ids.shape

    # 1. Token embeddings: [B, T] → [B, T, d_model]
    hidden = self.token_embedding(input_ids)

    # 2. Position offset: accumulate per-layer across blocks
    position_offset = 0

    # 3. N transformer blocks: [B, T, d_model] → [B, T, d_model]
    for block in self.blocks:
        hidden = block(
            hidden,
            attention_mask=attention_mask,
            position_offset=position_offset,
            use_manual_attention=use_manual_attention,
        )

    # 4. Final norm: [B, T, d_model] → [B, T, d_model]
    hidden = self.final_norm(hidden)

    # 5. Output projection: [B, T, d_model] → [B, T, vocab_size]
    logits = self.output_projection(hidden)

    # 6. Loss (only if targets provided)
    loss = None
    if targets is not None:
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=IGNORE_INDEX,  # if padding support added
        )

    return logits, loss
```

Note: `position_offset` is not incremented block-by-block for training (all
blocks see the full sequence). It is designed for KV-cache inference (Phase 11)
where each generation step appends one token and the offset increases. For now it
stays at 0 during training.

### Chunked Loss (Config-Gated)

When `config.training.chunked_loss` is enabled, the loss computation is tiled
over the sequence dimension to avoid materializing the full `[B*T, V]` logits
tensor:

```python
def _chunked_cross_entropy(self, hidden, targets, chunk_size=1024):
    """Compute CE loss in chunks over the batch*seq dimension."""
    B, T, D = hidden.shape
    total_loss = 0.0
    total_tokens = 0
    for start in range(0, B * T, chunk_size):
        end = min(start + chunk_size, B * T)
        chunk_hidden = hidden.view(B * T, D)[start:end]
        chunk_logits = self.output_projection(chunk_hidden)  # [chunk, V]
        chunk_targets = targets.view(-1)[start:end]
        total_loss += F.cross_entropy(
            chunk_logits, chunk_targets, reduction='sum'
        ).item()
        total_tokens += (end - start)
    return total_loss / total_tokens
```

This is a simple Python-level tiling approach. For our 16k vocabulary this is
rarely needed on GPU, but the CPU config benefits significantly because the
full `[BT, V]` logits tensor at FP32 would dominate RAM.

### Parameter Count Utility

```python
def count_parameters(self) -> dict[str, int]:
    """Return dict of parameter counts by functional group."""
    return {
        "token_embedding": sum(p.numel() for p in self.token_embedding.parameters()),
        "blocks": sum(p.numel() for p in self.blocks.parameters()),
        "final_norm": sum(p.numel() for p in self.final_norm.parameters()),
        "output_projection": (
            0 if self.config.model.tie_embeddings
            else sum(p.numel() for p in self.output_projection.parameters())
        ),
        "total": sum(p.numel() for p in self.parameters()),
    }
```

For the default config (vocab=16384, d_model=512, n_layers=8, n_heads=8,
n_kv_heads=4, d_ff=1408):

| Component | Parameters |
| --- | --- |
| Token embedding | 8,388,608 |
| Per block (×8) | ~1,628,160 each |
| — Attention QKV+O | 786,432 |
| — MLP gate+up+down | 721,920 |
| — RMSNorm ×2 | 1,024 |
| All 8 blocks | 13,025,280 |
| Final RMSNorm | 512 |
| Output projection | 0 (tied to embedding) |
| **Total** | **~21.4M** |

This fits comfortably in laptop GPU memory (~172 MB in fp32, ~86 MB in bf16 for
parameters alone) and is suitable for phone deployment after quantization.

### Memory Estimate Utility

```python
def estimate_memory(self, batch_size: int, seq_len: int) -> dict[str, float]:
    """Estimate activation + parameter memory in MiB."""
    n_params = sum(p.numel() for p in self.parameters())
    param_mem = n_params * 4 / (1024 ** 2)  # fp32 params

    # Rough activation estimate (dominated by QKV + attention scores + MLP)
    # Per layer: ~B * T * d_model * 4 (QKV) + B * n_heads * T * T (attn scores)
    # Simplified: ~10 × B × T × d_model
    act_per_layer = 10 * batch_size * seq_len * self.config.model.d_model * 2  # bf16
    act_mem = self.config.model.n_layers * act_per_layer / (1024 ** 2)

    return {"param_mem_mib": param_mem, "act_mem_mib": act_mem, "total_mib": param_mem + act_mem}
```

---

## Invariants

- The model must not call back into tokenizer, data pipeline, optimizer,
  scheduler, or checkpointing code.
- Weight initialization must be idempotent (calling `configure_initialization`
  twice produces the same result if weights haven't been trained).
- When `targets is None`, the model returns `(logits, None)` — no loss
  computation.
- When `tie_embeddings=True`, `output_projection.weight.data_ptr() ==
  token_embedding.weight.data_ptr()`.
- The `final_norm` is a dedicated RMSNorm instance, not shared with any block.
- `position_offset` is not incremented during training (all blocks see offset 0).
- The model does not create or manage a KV cache. That is Phase 11.
- Dropout is already configured at 0.0 in the default config; the model does not
  override it.
- Initial loss (random weights, no training) must be approximately
  `ln(vocab_size)` within ±0.5 nats.
- After overfit training on one small batch, loss must drop below 0.1.
- Parameter counts are deterministic for a given config.
- Model passes `torch.jit.script` compatibility check (optional but aspirational).

---

## Testing

Tests cover:

1. **Shape contracts**: Forward pass produces correct shapes for logits and loss.
2. **Initial loss sanity**: `|loss_at_init − ln(vocab_size)| < 0.5`.
3. **Overfit single batch**: Train on B=4, T=128 for ~500 steps; loss < 0.1.
4. **Weight tying behavior**: Tied weights share storage; untied allocates
   separate.
5. **Gradient flow**: Every named parameter receives non-zero gradient.
6. **Initialization idempotency**: Calling `configure_initialization` twice gives
   the same residual-projection std.
7. **Parameter count determinism**: `count_parameters()` returns consistent
   values.
8. **CPU config compatibility**: Model instantiates with `configs/cpu.yaml`.
9. **Default config compatibility**: Model instantiates with
   `configs/default.yaml`.
10. **No data leakage**: The model does not import or reference data pipeline
    modules.
11. **Inference mode (targets=None)**: Returns `(logits, None)` without computing
    loss.
12. **Chunked loss parity**: When enabled, chunked loss ≈ standard CE within
    tolerance.
13. **Config validation**: Invalid configs (e.g., vocab_size mismatch with
    embedding) raise clear errors.

### Overfit Test Specification

The overfit test is the most important test in Phase 9. It validates the entire
architecture stack end-to-end:

```python
def test_overfit_single_batch():
    config = _small_config()
    model = EdgeGPT(config)
    model.configure_initialization()

    # Fixed small batch — no shuffling
    torch.manual_seed(42)
    input_ids = torch.randint(0, config.model.vocab_size, (4, 128))
    targets = torch.randint(0, config.model.vocab_size, (4, 128))

    # Initial loss sanity
    with torch.no_grad():
        _, init_loss = model(input_ids, targets)
    expected = math.log(config.model.vocab_size)
    assert abs(init_loss.item() - expected) < 0.5

    # Overfit training
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    num_steps = 500
    for step in range(num_steps):
        _, loss = model(input_ids, targets)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    assert loss.item() < 0.1, f"Loss {loss.item():.4f} did not drop below 0.1"
```

The test uses a tiny model config (e.g., d_model=64, n_layers=2, n_heads=2,
vocab_size=128) for speed while still exercising the full architecture stack.

---

## Implementation Checklist

1. **Add `chunked_loss` to `TrainingConfig`** in `configs/config.py`.
2. **Create `model/model.py`** with `EdgeGPT` class:
   - `__init__`: token_embedding + blocks (ModuleList) + final_norm + output_projection
   - `forward`: token embed → blocks → final norm → logits → optional loss
   - `configure_initialization`: N(0, 0.02) with residual scaling
   - `count_parameters`: component-level parameter counts
   - `estimate_memory`: rough MiB estimates for capacity planning
   - `_chunked_cross_entropy` (private): tiled loss for CPU/memory-constrained scenarios
3. **Update `model/__init__.py`** to re-export `EdgeGPT`.
4. **Create `tests/test_model.py`** with all test cases listed above.
5. **Run full test suite** to confirm no regressions in earlier phases.
6. **Exit criterion**: Overfit test passes (loss < 0.1 on single batch).

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- |
| Weight tying breaks gradient flow | Low | High | Already tested in Phase 3; test_model validates end-to-end |
| Chunked loss is numerically different from standard CE | Medium | Low | Test parity; gate behind config flag until proven |
| Overfit test takes too long (500 steps × model size) | Low | Medium | Use tiny config for the test (d_model=64, 2 layers) |
| Initial loss deviates from ln(vocab) due to embedding scale | Low | Low | Account for `embedding_scale` in the sanity check math |
| CUDA OOM during overfit test on laptop GPU | Low | Medium | Test uses tiny dims; overfit on CPU fallback if needed |
