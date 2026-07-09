# EdgeGPT — Phase 11 Inference & Generation Technical Design

## Summary

Phase 11 adds the first real inference path for EdgeGPT: prompt encoding, autoregressive token generation, optional sampling controls, and a Llama-style external KV cache. The goal is correctness and architectural clarity before Phase 12 GGUF export.

The default demo checkpoint is the first full TinyStories run:

- Config: `configs/tinystories_full_gpu_test.yaml`
- Checkpoint: `artifacts/runs/tinystories_full_gpu_test_1000/latest.pt`
- Tokenizer: `artifacts/tokenizer/main_16k`

This phase intentionally does not introduce Hugging Face `GenerationMixin`, vLLM, a streaming server, beam search, or llama.cpp export. Those are useful later, but they would make the first inference implementation harder to validate.

## Direction Comparison

| Direction | Production / Research Status | Pros | Cons | Decision |
|---|---|---|---|---|
| Full-context recompute | Simple baseline used for tests | Easiest to reason about; no mutable cache | O(T^2) repeated work during generation | Keep as correctness oracle |
| Llama-style preallocated KV cache | Used by Llama-family reference code | Simple, fast token decode, maps to GQA and RoPE | Fixed max length and batch shape per cache | Use for Phase 11 |
| Gemma-style batched cache | Production-style PyTorch reference | Better path toward batched serving | More masking and position bookkeeping | Defer until single-prompt works |
| Hugging Face `generate` | Mature production API | Feature-rich sampling and stopping logic | Requires adapting model output/cache conventions | Defer |
| llama.cpp generation | Deployment target | Validates phone runtime behavior | Requires Phase 12 export first | Defer to Phase 12 |
| vLLM / PagedAttention | Serving-scale state of the art | Efficient many-request KV memory management | Overkill for local single-user inference | Defer |

References:

- Hugging Face generation strategies: https://huggingface.co/docs/transformers/main/en/generation_strategies
- Meta Llama 3 cache/start-position implementation: https://github.com/meta-llama/llama3/blob/main/llama/model.py
- Google Gemma PyTorch generation/cache implementation: https://github.com/google/gemma_pytorch/blob/main/gemma/model.py
- vLLM PagedAttention: https://arxiv.org/abs/2309.06180
- PyTorch SDPA: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html

## Architecture

The cache is external to the model modules. `model.cache.KVCache` owns one `LayerKVCache` per transformer block. Each layer stores:

```text
k: [batch, n_kv_heads, max_seq_len, head_dim]
v: [batch, n_kv_heads, max_seq_len, head_dim]
```

The cache stores only GQA K/V heads, not repeated query heads. Repetition still happens inside attention at score computation time.

The model forward path accepts optional cache arguments:

```python
EdgeGPT.forward(input_ids, kv_cache=None, cache_position=0, position_offset=0)
TransformerBlock.forward(hidden, layer_cache=None, cache_position=0)
CausalSelfAttention.forward(hidden, kv_cache=None, cache_position=0)
```

When no cache is passed, training and evaluation behavior remain unchanged. When a cache is passed, each attention layer writes the current K/V states at `cache_position` and attends over the visible prefix.

The cached causal mask is offset-aware. A query at absolute position `p` may attend to keys `0..p`. This differs from a naive `tril()` mask when decoding one token with many cached keys.

## Generation Flow

`eval.generation.generate_ids()` implements the first decode loop:

1. Encode the prompt to `[1, T]` token IDs.
2. If caching is enabled, allocate `KVCache` for `T + max_new_tokens`.
3. Prefill the prompt once and use the last-position logits to sample the first new token.
4. For each later token, pass only the previous token with `cache_position` equal to its absolute position.
5. Stop on EOS or `max_new_tokens`.
6. Decode token IDs back to text.

Sampling controls are intentionally minimal:

- Greedy argmax by default
- `temperature`
- `top_k`
- `top_p`
- optional deterministic seed for sampling

CLI usage:

```bash
python scripts/generate.py \
  --config configs/tinystories_full_gpu_test.yaml \
  --checkpoint artifacts/runs/tinystories_full_gpu_test_1000/latest.pt \
  --prompt "Once upon a time" \
  --max-new-tokens 64
```

## Acceptance Criteria

- Cached next-token logits match full-context recompute logits in eval mode.
- Cached greedy generation returns the same token sequence as full recompute.
- Cache tensors have the expected GQA shape and preserve appended K/V values.
- Greedy, temperature-zero, top-k, top-p, and EOS stopping behavior are tested.
- The default checkpoint can generate non-empty text from `scripts/generate.py`.
- Existing Phase 10 training tests keep passing.