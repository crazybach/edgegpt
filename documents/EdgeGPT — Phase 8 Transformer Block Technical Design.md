# EdgeGPT Phase 8 - Transformer Block Technical Design

## Summary

Phase 8 assembles the first complete Llama-compatible decoder block from the
existing standalone primitives: RMSNorm, causal GQA attention with RoPE, and the
SwiGLU MLP. The block uses serial pre-norm residual ordering:

```python
x = x + Attention(RMSNorm(x))
x = x + MLP(RMSNorm(x))
```

This phase builds one reusable decoder layer. It does not stack layers, add the
final model norm, produce logits, compute loss, initialize checkpoint-specific
scaling, or implement KV cache.

## Direction Decision

| Direction | Status | Decision |
| --- | --- | --- |
| Post-norm block | Original Transformer / older BERT-style | Rejected |
| Serial pre-norm block | Llama/GPT-style modern baseline | Used now |
| Parallel attention + MLP | PaLM/GPT-J/GPT-NeoX-style variants | Deferred |
| Sandwich or peri-norm | Modern research | Deferred |
| ReZero / gated residual | Research and specialized deep nets | Deferred |
| MoE block | Production large-scale variant | Deferred |

Serial pre-norm is the conservative baseline because it keeps the residual path
simple, is stable at initialization, and matches the Llama-compatible direction
used by earlier EdgeGPT phases.

## Production Context

- Modern decoder-only LLMs commonly use pre-norm blocks for stable gradient flow.
- Llama-style dense blocks apply RMSNorm inside each residual branch and use
  RoPE attention plus a SwiGLU feed-forward branch.
- Parallel attention/MLP blocks can reduce dependency depth inside a layer, but
  are less aligned with the target Llama export path.
- MoE replaces or augments the MLP branch in large-scale models, but it adds
  routing, expert capacity, auxiliary losses, and export risk.

References:

- Pre-LN analysis: https://arxiv.org/abs/2002.04745
- B2T / norm placement discussion: https://arxiv.org/abs/2206.00330
- Llama architecture overview: https://en.wikipedia.org/wiki/Llama_%28language_model%29
- Parallel attention and FFN design: https://arxiv.org/abs/2305.13297

## Implementation

- `model/block.py` exposes `TransformerBlock(config, layer_idx=None)`.
- Modules:
  - `attention_norm = RMSNorm(config)`
  - `attention = CausalSelfAttention(config)`
  - `mlp_norm = RMSNorm(config)`
  - `mlp = SwiGLUMLP(config)`
- Forward API:

```python
forward(
    hidden,
    *,
    attention_mask=None,
    position_offset=0,
    use_manual_attention=False,
)
```

- Shape contract:
  - input: `[B, T, d_model]`
  - output: `[B, T, d_model]`
- Residual order:

```python
hidden = hidden + attention(attention_norm(hidden), ...)
hidden = hidden + mlp(mlp_norm(hidden))
```

## Invariants

- The two RMSNorm modules are independent parameters.
- The block must not create token embeddings, final norm, output projection,
  loss objects, optimizers, data loaders, or a layer stack.
- Dropout remains only where existing submodules already implement it.
- KV cache and generation-specific APIs remain later-phase work.
- Parallel residual, gated residual, peri-norm, and MoE variants remain future
  config-gated alternatives.
