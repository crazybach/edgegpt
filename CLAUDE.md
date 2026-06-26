# CLAUDE.md — EdgeGPT Project Guide

EdgeGPT is a laptop-scale, Llama-compatible decoder-only transformer built in PyTorch. The primary goal is to train a small LLM from scratch and deploy it to a phone via llama.cpp + GGUF.

## Build Order (critical path)

```
0 → 1 → 2 → 3 → 5 → 4 → 7 → 6 → 8 → 9 → 10 → 11 → 12, then 13/14
```

- **Phase 0**: Project foundation (configs/, test harness, env)
- **Phase 1**: Tokenizer (byte-level BPE)
- **Phase 2**: Data pipeline (streamed, packed, sharded)
- **Phase 3**: Embeddings (token + RoPE-in-attention, weight tying)
- **Phase 5**: RMSNorm (pre-norm, no bias)
- **Phase 4**: RoPE (position via rotation of Q/K)
- **Phase 7**: SwiGLU MLP (gate/up/down, no bias)
- **Phase 6**: Attention (single → multi → GQA → SDPA → KV cache)
- **Phase 8**: Transformer block (residual x2, pre-norm)
- **Phase 9**: Full model → forward → loss → overfit test
- **Phase 10**: Training loop (AdamW, cosine LR, grad accumulation, mixed precision)
- **Phase 11**: Inference (KV cache, sampling)
- **Phase 12**: Export to GGUF via llama.cpp converter
- **Phase 13/14**: MoE / Vision (future)

## Non-negotiable milestones

1. **Overfit one batch to ~0 loss** (end of Phase 9) — proves architecture correctness
2. **PyTorch output == llama.cpp output** (Phase 12) — proves deployment faithfulness

## Architecture (Llama-compatible)

- Decoder-only, RMSNorm, RoPE, SwiGLU, GQA
- No bias in linear projections
- Pre-norm (norm before attention/MLP, inside residual branch)
- Weight tying between input embeddings and output projection
- Default config: 512-dim, 8 layers, 8 heads, 4 KV heads, 2048 ctx

## Key files

- `configs/default.yaml` — All hyperparameters. No magic numbers in code.
- `configs/config.py` — Dataclass config system, YAML loader, device resolver
- `model/` — Model components (attention, MLP, norm, embedding, etc.)
- `data/` — Data ingestion, tokenization, packing, batching
- `train/` — Training loop, optimizer, LR schedule, checkpointing
- `eval/` — Loss eval, perplexity, sample generation
- `tests/` — Pytest suite, one test file per component

## Conventions

- Every component has a unit test with shape contracts
- Build vertically: implement + test each component before moving on
- Keep llama.cpp converter path in mind — match Llama naming/structure
- Use `configs/config.py` Config class; never hardcode dimensions
- Validate shape contracts at every stage: `[B, T, d_model]` in → out

## Environment

```bash
source .venv/Scripts/activate  # Windows
pip install -r requirements.txt
pytest tests/ -v
```
