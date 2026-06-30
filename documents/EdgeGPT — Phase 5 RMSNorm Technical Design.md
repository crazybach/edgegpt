# EdgeGPT Phase 5 - RMSNorm Technical Design

## Summary

Phase 5 adds a standalone Root Mean Square Layer Normalization module
(`RMSNorm`) for the Llama-compatible model stack. RMSNorm normalizes each token
vector by its root mean square, then applies a learned gain vector. It does not
subtract the mean and does not use a bias parameter.

This phase only provides the reusable normalization primitive. Transformer
blocks, attention, MLP wiring, and final logits/loss integration remain later
phases.

## Direction Decision

| Direction | Status | Decision |
| --- | --- | --- |
| LayerNorm | Mature GPT-style baseline | Rejected for Llama compatibility |
| PyTorch `nn.RMSNorm` | Available in newer PyTorch versions | Deferred to avoid version/API drift |
| Custom RMSNorm | Production-compatible and export-friendly | Used now |
| QK-Norm | Modern attention stability variant | Deferred to the attention phase |
| Gemma-style variants | Production reference | Deferred until baseline behavior is stable |

## Implementation

- `model/norm.py` exposes `RMSNorm(config_or_dim, eps=None)`.
- `config_or_dim` may be a full `Config` object or an integer hidden size.
- When a `Config` is provided:
  - hidden size comes from `config.model.d_model`
  - epsilon comes from `config.model.norm_eps`, unless explicitly overridden
- Learned parameters:
  - `weight: [d_model]`
  - initialized to ones
  - no bias
- Formula:

```python
x * rsqrt(mean(x * x, dim=-1, keepdim=True) + eps) * weight
```

## Numerical Behavior

RMSNorm receives hidden states shaped `[B, T, d_model]`, but the implementation
supports any input whose final dimension is `d_model`.

The RMS statistic is computed in `float32` for numerical stability, then cast
back to the input dtype before applying the learned gain. This keeps fp16/bf16
training stable while preserving the module's input/output dtype contract.

## Placement In Later Phases

EdgeGPT uses pre-norm transformer blocks:

```python
x = x + Attention(RMSNorm(x))
x = x + MLP(RMSNorm(x))
```

A final RMSNorm is applied after the last transformer block and before the tied
output projection. Pre-norm keeps gradients better behaved at depth and matches
the Llama-family block layout.

## Invariants

- RMSNorm must not subtract the mean.
- RMSNorm must not add a bias parameter.
- The only trainable parameter is the learned gain vector.
- `norm_eps` must remain part of the model checkpoint/config contract.
- QK-Norm, LayerNorm, and long-context attention stability tricks are separate
  choices and must not be hidden inside this module.

## References

- RMSNorm paper: https://arxiv.org/abs/1910.07467
- PyTorch RMSNorm API reference: https://docs.pytorch.org/docs/stable/generated/torch.nn.RMSNorm.html
- Llama-style pre-norm transformer layout: https://arxiv.org/abs/2302.13971
