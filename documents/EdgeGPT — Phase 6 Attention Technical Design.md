# EdgeGPT Phase 6 - Attention Technical Design

## Summary

Phase 6 adds Llama-style causal self-attention with Grouped-Query Attention
(GQA). Attention is the first core transformer block component: it projects
hidden states into Q/K/V tensors, applies RoPE to Q and K, performs causal
scaled dot-product attention, then projects the result back to `d_model`.

The default implementation uses PyTorch scaled dot-product attention (SDPA)
when possible and keeps a manual attention path as a readable reference and
test oracle.

## Direction Decision

| Direction | Status | Decision |
| --- | --- | --- |
| Full multi-head attention | Older GPT-style baseline | Test-compatible, not default |
| Multi-query attention | Production inference-efficient | Deferred because quality tradeoff is larger |
| Grouped-query attention | Llama/Qwen/Mistral-style production default | Used now |
| Multi-head latent attention | DeepSeek-style advanced compression | Deferred |
| Sliding-window attention | Mistral/Gemma-style long-context variant | Deferred |
| Sparse or paged serving attention | Production serving optimization | Deferred |

GQA is the chosen baseline because it keeps separate query heads while sharing
each KV head across a group of query heads. This cuts KV-cache size versus full
MHA and remains compatible with Llama-family deployment paths.

## Production Context

- Llama-family models use RoPE, RMSNorm, causal decoder attention, and GQA in
  modern variants.
- Qwen2.5 uses fewer KV heads than query heads, confirming GQA as a production
  default for small and large decoder models.
- Mistral uses GQA and adds sliding-window attention; the sliding-window policy
  is intentionally deferred until the dense baseline trains.
- DeepSeek V2/V3 use MLA to compress KV state more aggressively, but that is a
  larger architecture decision than Phase 6 should make.
- vLLM-style paged attention optimizes serving-time KV-cache allocation and is
  a later inference system concern.

References:

- GQA paper: https://arxiv.org/abs/2305.13245
- PyTorch SDPA: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
- Qwen2.5 config example: https://huggingface.co/Qwen/Qwen2.5-0.5B/blob/main/config.json
- Mistral config example: https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/config.json
- FlashAttention-2: https://arxiv.org/abs/2307.08691
- vLLM / PagedAttention: https://arxiv.org/abs/2309.06180

## Implementation

- `model/attention.py` exposes:
  - `repeat_kv(x, n_rep)`
  - `manual_scaled_dot_product_attention(q, k, v, is_causal=True, attention_mask=None)`
  - `CausalSelfAttention(config)`
- Shape contract:
  - input hidden states: `[B, T, d_model]`
  - Q: `[B, n_heads, T, head_dim]`
  - K/V: `[B, n_kv_heads, T, head_dim]`
  - output: `[B, T, d_model]`
- Projection layers:
  - `q_proj: d_model -> n_heads * head_dim`
  - `k_proj: d_model -> n_kv_heads * head_dim`
  - `v_proj: d_model -> n_kv_heads * head_dim`
  - `o_proj: n_heads * head_dim -> d_model`
- RoPE is applied to Q and K only. V is never rotated.
- SDPA is the default attention path. The manual path exists for tests and
  debugging.

## Invariants

- `d_model` must divide evenly by `n_heads`.
- `n_heads` must divide evenly by `n_kv_heads`.
- `n_kv_heads` must be positive and cannot exceed `n_heads`.
- Attention is causal by default.
- GQA repeats KV heads only for the attention operation; the projected K/V
  tensors remain smaller than Q.
- Dropout is zero in eval mode.
- KV cache, sliding-window masks, MLA, QK-Norm, and attention logit
  soft-capping are separate future phases.

## Testing

Tests cover tensor shapes, GQA repeat behavior, config validation, causal
behavior, RoPE placement, SDPA/manual parity, MHA equivalence, gradient flow,
dropout behavior, and compatibility with both CPU and default configs.
