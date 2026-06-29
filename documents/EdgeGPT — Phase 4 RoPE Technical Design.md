# EdgeGPT Phase 4 - RoPE Technical Design

## Summary

Phase 4 adds standard one-dimensional Rotary Positional Embeddings (RoPE) as a
standalone module. RoPE is applied to attention queries and keys only. Values
are not rotated, and no learned positional embedding table is introduced.

This keeps EdgeGPT aligned with Llama-style decoder models and prepares the
attention phase to consume tensors shaped `[B, H, T, D]`.

## Direction Decision

| Direction | Status | Decision |
| --- | --- | --- |
| Learned absolute position embeddings | Older GPT-style | Rejected for Llama compatibility |
| ALiBi | Mature alternative | Deferred |
| Standard 1D RoPE | Llama/Qwen/Mistral-style production default | Used now |
| Larger `rope_theta` | Production long-context tuning | Config-supported, not changed |
| NTK/linear/YaRN scaling | Long-context retrofit/research | Deferred |
| Context-aware or nD RoPE | Research/multimodal direction | Deferred |

## Implementation

- `model/rope.py` exposes:
  - `rotate_half(x)`
  - `apply_rotary_pos_emb(q, k, cos, sin, position_offset=0)`
  - `RotaryEmbedding(config)`
- Shape contract:
  - input Q/K: `[B, H, T, D]`
  - output Q/K: `[B, H, T, D]`
- `head_dim = d_model // n_heads` and must be even.
- Frequencies are computed from `model.rope_theta`.
- Cosine and sine tables are cached as non-persistent buffers so they move with
  `.to(device)` but are not checkpoint parameters.
- Cache extension grows the table when `position_offset + T` exceeds the
  initial `max_seq_len`.

## Layout

The implementation uses the Llama/HuggingFace half-rotation layout:

```python
rotate_half([x1, x2]) = [-x2, x1]
rotated = x * cos + rotate_half(x) * sin
```

The cached frequency table duplicates frequencies so `cos` and `sin` have
shape `[max_seq_len, head_dim]`.

## Notes

RoPE gives attention scores relative-position structure without adding learned
position parameters. Long-context scaling methods such as YaRN or NTK-aware
scaling are left for a later phase because they introduce tuning decisions that
should be validated only after the dense baseline trains correctly.

References:

- RoFormer / original RoPE: https://arxiv.org/abs/2104.09864
- Qwen2.5 config example using RoPE: https://huggingface.co/Qwen/Qwen2.5-0.5B/blob/main/config.json
- YaRN long-context extension: https://arxiv.org/abs/2309.00071
