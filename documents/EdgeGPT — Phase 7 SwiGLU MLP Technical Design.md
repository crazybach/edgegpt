# EdgeGPT Phase 7 - SwiGLU MLP Technical Design

## Summary

Phase 7 adds the dense feed-forward network used inside each future decoder
block. EdgeGPT uses a Llama-style SwiGLU MLP: a gate projection and an up
projection expand the residual stream to `d_ff`, the gate is activated with
SiLU, the two expanded tensors are multiplied elementwise, and a down projection
returns to `d_model`.

This phase only provides the standalone MLP primitive. It does not add RMSNorm,
residual wiring, attention, block composition, logits, loss, or MoE routing.

## Direction Decision

| Direction | Status | Decision |
| --- | --- | --- |
| ReLU FFN | Original Transformer baseline | Rejected |
| GELU FFN | GPT/BERT-style mature baseline | Rejected for Llama compatibility |
| GEGLU | Strong gated FFN variant | Deferred |
| SwiGLU | Llama/PaLM/Mistral-style production default | Used now |
| SoLU | Research / interpretability direction | Deferred |
| Dense MoE replacement | Production large-model direction | Deferred |
| Fine-grained/shared-expert MoE | DeepSeek-style advanced direction | Deferred |

SwiGLU is the default because it is the dense MLP choice used by many modern
Llama-like decoder models and keeps the future export path simpler than using a
less common activation or an MoE router.

## Production Context

- Llama-family models use a gated MLP with gate/up/down projections.
- Mistral dense models follow the same Llama-like gated MLP direction; Mixtral
  replaces the dense MLP with MoE experts in a separate architecture choice.
- Qwen2.5 and Gemma-style models confirm that gated MLPs remain mainstream in
  modern decoder-only LLMs.
- DeepSeek-style MoE is a strong future direction, but it adds routing,
  auxiliary losses, expert capacity, and export concerns that should wait until
  the dense baseline trains correctly.

References:

- GLU variants / SwiGLU: https://arxiv.org/abs/2002.05202
- Mistral 7B: https://arxiv.org/abs/2310.06825
- Qwen2.5 technical report: https://arxiv.org/abs/2412.15115
- Mixtral sparse MoE: https://arxiv.org/abs/2401.04088

## Implementation

- `model/mlp.py` exposes `SwiGLUMLP(config)`.
- Shape contract:
  - input: `[B, T, d_model]` or any `[..., d_model]`
  - output: same leading dimensions with final dimension `d_model`
- Bias-free projections:
  - `gate_proj: d_model -> d_ff`
  - `up_proj: d_model -> d_ff`
  - `down_proj: d_ff -> d_model`
- Formula:

```python
down_proj(silu(gate_proj(x)) * up_proj(x))
```

No dropout is added in this module. Dropout, if used, should be a later block
or training-policy decision so the MLP remains a clean Llama-compatible
primitive.

## d_ff Sizing

For SwiGLU, `d_ff` is typically around `8/3 * d_model`, rounded to a practical
multiple. This keeps parameter count close to a classic `4 * d_model` GELU MLP
while benefiting from the gated activation. Current configs already encode the
chosen values:

- default: `d_model=512`, `d_ff=1408`
- CPU: `d_model=256`, `d_ff=704`

## Invariants

- `d_ff` must be positive.
- All three projections are bias-free.
- The MLP must not apply normalization or residual addition internally.
- The MLP must not call attention or output projection layers.
- MoE remains a future replacement path, not part of Phase 7.
