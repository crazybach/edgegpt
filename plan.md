EdgeGPT — Build Plan (Zero → Deployable)
Guiding Principles
Framework: PyTorch. It's the right call. Every reference project you named (nanoGPT, Gemma, Llama) is PyTorch-first, the ecosystem (tokenizers, datasets, profiling) is richest there, and — critically — the llama.cpp convert_hf_to_gguf.py path expects a HuggingFace/PyTorch-style checkpoint. Build your model to mirror the Llama architecture's naming/structure and your conversion-to-GGUF step becomes nearly free.
Build vertically, test each layer in isolation. Every stage below has an explicit unit test and shape contract. Never move to the next stage until the current one passes a numerical sanity check.
Keep it Llama-compatible. Decoder-only, RMSNorm, RoPE, SwiGLU, GQA. This is the path of least resistance to llama.cpp. Add MoE and newer tricks after the dense baseline trains correctly.
Phase 0 — Project Foundation & Scaffolding
Goal: Repo, environment, and tooling before any model code.

Repo structure — separate model/, data/, train/, eval/, configs/, tests/, scripts/, notebooks/.
Environment — Python venv, pin PyTorch version, requirements.txt. Decide device strategy: CPU + MPS (Apple) or CUDA for the laptop GPU.
Config system — a single dataclass/YAML defining every hyperparameter (dims, layers, heads, vocab size, context length). Everything downstream reads from this. No magic numbers in code.
VS Code setup — launch.json for debugging train/eval entrypoints, settings.json for the interpreter, recommended extensions (Python, Pylance, Jupyter), and a .env for paths.
Logging & reproducibility — global seed control, deterministic flags, a run-logging choice (TensorBoard or Weights & Biases), checkpoint directory convention.
Test harness — pytest wired up so every stage gets a test from day one.
Exit criterion: pytest runs (empty), VS Code debugger hits a breakpoint in a dummy script, config loads.

Phase 1 — Tokenizer
Goal: Turn raw text ↔ integer token IDs.

Choose algorithm — Byte-level BPE (GPT/Llama style) or SentencePiece Unigram (Gemma style). Recommendation: byte-level BPE for robustness (no out-of-vocab, handles code/emoji/unicode by construction).
Special tokens — define and reserve IDs for BOS, EOS, PAD, UNK, and any chat/role markers you'll want later. Lock these early; changing them later invalidates checkpoints.
Vocab size decision — for a laptop-scale model, ~8k–32k. Smaller vocab = smaller embedding matrix = less memory, but longer sequences. This is a real tradeoff; pick and record it in config.
Training the tokenizer — fit BPE merges on a representative sample (mix of your web + code + math data so code symbols and math notation get good coverage).
Encode/decode operations — implement and verify round-trip: decode(encode(text)) == text for ASCII, unicode, emoji, and code snippets.
Serialization — save/load tokenizer artifacts; ensure the format is convertible for llama.cpp later.
Shape contract: encode: str → LongTensor[seq_len].
Unit test: round-trip fidelity, special-token insertion, batch encoding with padding.

Phase 2 — Data Pipeline
Goal: Streamed, tokenized, batched training data.

Ingestion — loaders for your staged datasets (TinyStories → SmolLM mix → math/code). Abstract the source so swapping datasets is a config change.
Tokenize-and-pack — convert documents to token streams, concatenate with EOS separators, and pack into fixed-length blocks of context_length (no wasted padding during pretraining).
Sharding & caching — write tokenized data to memory-mapped binary shards (nanoGPT-style .bin) so epochs are fast and RAM-light on a laptop.
Batch sampler — yields (input_ids, target_ids) where targets are inputs shifted by one position.
Train/val split — deterministic holdout for loss tracking.
Shape contract: one batch = input_ids: [B, T], targets: [B, T], both LongTensor.
Unit test: targets are inputs shifted by 1; no index exceeds vocab size; block boundaries respect EOS.

Phase 3 — Embeddings
Goal: Token IDs → continuous vectors.

Token embedding — lookup table [vocab_size, d_model]. Operation: gather rows by token ID → [B, T, d_model].
Positional information — decision point. Modern Llama-style models use RoPE applied inside attention, not a learned positional embedding table. So: no separate positional embedding layer. Document this; it's a common point of confusion.
Weight tying — tie the input embedding matrix with the output (logits) projection. Saves parameters — meaningful at laptop scale.
Embedding scaling — note whether you scale embeddings by √d_model (Gemma does; Llama doesn't). Pick one, record it.
Shape contract: [B, T] → [B, T, d_model].
Unit test: output shape, weight-tying actually shares storage, gradients flow.

Phase 4 — RoPE (Rotary Positional Embeddings)
Goal: Inject position via rotation of Q/K vectors. Build this as a standalone module before attention.

Precompute frequencies — inverse-frequency vector from a base (θ, typically 10000); build cos and sin tables for all positions up to max context.
Apply-rotation operation — split each head's Q and K into even/odd (or half/half) components and apply the 2D rotation per position.
Caching the tables — register as buffers so they move with the model to GPU/MPS and aren't trained.
Long-context hook (future) — leave room for RoPE scaling (NTK/linear/YaRN) so you can extend context later without re-architecting.
Shape contract: Q,K [B, n_heads, T, head_dim] → rotated, same shape.
Unit test: rotation preserves vector norm; relative-position property (dot product depends only on position difference) holds numerically.

Phase 5 — Normalization (RMSNorm)
Goal: Stabilize activations.

RMSNorm — normalize by root-mean-square (no mean-subtraction, no bias), then scale by a learned gain vector. Cheaper than LayerNorm and what Llama/Gemma use.
Placement: Pre-norm. Norm is applied before attention and before MLP (inside the residual branch), not after. This is essential for training stability at depth.
Final norm — one RMSNorm after the last block, before the output projection.
Shape contract: [B, T, d_model] → [B, T, d_model].
Unit test: output RMS ≈ 1 before the gain scale; numerical stability with the epsilon term.

Phase 6 — Attention (the core)
Goal: Causal self-attention with Grouped-Query Attention (GQA). Build incrementally.

Step 6a — Single-head causal attention first. Q·Kᵀ → scale by 1/√head_dim → causal mask (upper triangle = −∞) → softmax → weighted sum of V. Get this provably correct before adding heads.
Step 6b — Multi-head. Project to Q,K,V; reshape to [B, n_heads, T, head_dim]; attention per head in parallel; concat; output projection.
Step 6c — Insert RoPE (Phase 4) on Q and K after projection, before the score computation.
Step 6d — Grouped-Query Attention. Use n_heads query heads but fewer n_kv_heads; each KV head is shared across a group of query heads (repeat-KV operation). This is the single biggest memory saver for inference on phones — it shrinks the KV cache. Llama uses it; it's directly supported by llama.cpp.
Step 6e — Causal masking strategy — decide between an explicit additive mask and PyTorch's scaled_dot_product_attention (Flash path). Recommendation: use SDPA with is_causal=True for speed and memory, keep the manual version as a reference/test oracle.
Step 6f — KV cache (inference only). Design the interface now even if you implement it later: store past K,V per layer; at generation step append new K,V and attend over the full cache. Critical for fast on-device generation.
Shape contract: [B, T, d_model] → [B, T, d_model].
Unit tests: (a) causal property — output at position t is unaffected by tokens > t; (b) GQA output matches MHA when n_kv_heads == n_heads; (c) SDPA path matches the manual reference within tolerance.

Phase 7 — Feed-Forward Network (SwiGLU MLP)
Goal: Per-token nonlinearity.

SwiGLU — three projections: gate and up (both d_model → d_ff), elementwise SiLU(gate) * up, then down (d_ff → d_model). This gated variant outperforms plain GELU MLPs and is the Llama/Gemma standard.
Sizing d_ff — typically ~8/3 × d_model for SwiGLU (so param count matches a 4× GELU MLP), rounded to a hardware-friendly multiple. Record the rule in config.
No bias — Llama-style omits biases in these projections; keep it consistent for clean GGUF conversion.
Shape contract: [B, T, d_model] → [B, T, d_model].
Unit test: shape, gradient flow, SiLU correctness at a few sample points.

Phase 8 — Transformer Block (Assembly)
Goal: Compose one decoder layer with correct residuals.

Residual branch 1: x = x + Attention(RMSNorm(x)).
Residual branch 2: x = x + MLP(RMSNorm(x)).
Pre-norm ordering — confirm norms are inside the residual branches (Phase 5).
Parameterize — block takes the config so depth is just stacking N identical blocks.
Shape contract: [B, T, d_model] → [B, T, d_model] (residual-preserving).
Unit test: identity-ish behavior at init (residual dominates); gradients reach both sub-layers.

Phase 9 — Full Model (Logits)
Goal: End-to-end forward pass.

Forward chain: token embedding → (optional scale) → N transformer blocks → final RMSNorm → output projection (tied to embeddings) → logits [B, T, vocab_size].
Loss: cross-entropy over shifted targets, averaged over non-padded tokens. Report perplexity = exp(loss).
Parameter count utility — print params by component so you can tune model size to fit laptop training + phone inference budgets.
Init scheme — scaled init (e.g., normal with std tied to d_model, residual-projection scaling by 1/√(2N)) for stable deep training.
Shape contract: [B, T] → logits [B, T, vocab_size] → scalar loss.
Unit tests: (a) loss at init ≈ ln(vocab_size) (uniform-prediction sanity check); (b) overfit a single tiny batch to near-zero loss — the definitive proof the whole stack learns.

Phase 10 — Training Loop
Goal: Stable, resumable training on a laptop.

Optimizer — AdamW, decoupled weight decay (exclude norms and embeddings from decay).
LR schedule — linear warmup → cosine decay to a floor.
Gradient clipping — global-norm clip (e.g., 1.0).
Gradient accumulation — simulate a larger batch than VRAM allows. Essential on a laptop.
Mixed precision — bf16/fp16 autocast where supported; keep a master copy if needed.
Checkpointing — save/resume model + optimizer + scheduler + step + RNG state.
Eval hook — periodic val loss + a few sample generations to eyeball quality.
Throughput logging — tokens/sec, memory, so you can right-size the model.
Exit criterion: Validation loss decreases steadily on TinyStories; sampled text is coherent.

Phase 11 — Inference & Generation
Goal: Autoregressive sampling with the KV cache.

Decode loop — feed prompt, then generate token-by-token using the KV cache (Phase 6f).
Sampling controls — temperature, top-k, top-p; greedy as a baseline.
Stopping — EOS handling, max-tokens cap.
Correctness check — cached generation must match a non-cached full-recompute generation exactly (greedy).
Unit test: KV-cache output == no-cache output (greedy, same seed).

Phase 12 — Export to GGUF / llama.cpp
Goal: Deploy on phone.

Architecture parity check — confirm your layer naming and structure map onto a Llama-family architecture so convert_hf_to_gguf.py recognizes it. Build to this target from Phase 3 onward.
Write a HF-style checkpoint + config — the converter reads a config.json-style spec (dims, heads, kv-heads, rope base, norm eps, vocab).
Convert to GGUF then quantize (Q4_K_M / Q5_K_M are good phone tradeoffs).
Validate — run the GGUF in llama.cpp and compare a few greedy generations against your PyTorch model to confirm the conversion is faithful.
Exit criterion: Identical prompt produces matching output in PyTorch and llama.cpp; runs within phone memory budget.

Phase 13 — MoE & Modern Upgrades (after dense baseline works)
Only attempt these once Phases 1–12 are solid. Each is a swap-in, gated by config.

Mixture-of-Experts (MoE) — replace the MLP (Phase 7) in some/all blocks with N expert MLPs + a router (top-k gating). Add the auxiliary load-balancing loss. Note: DeepSeek-style fine-grained experts + shared experts is the current best practice. Caveat: llama.cpp supports several MoE architectures, but verify your specific variant converts before committing — this is the riskiest part for deployment.
Multi-head Latent Attention (MLA) — DeepSeek's KV-compression alternative to GQA. Bigger architectural change; only if you want to research it. GQA is the safer deployable default.
RoPE scaling for long context — YaRN/NTK to extend beyond trained context.
QK-norm / attention logit soft-capping — stability tricks from Gemma 2 / newer models.
Quantization-aware training — if post-training quant degrades quality too much on phone.
Rule: every upgrade keeps the same shape contracts so your tests from earlier phases still pass.

Phase 14 — Future: Vision (ViT extension)
Park this until text works end-to-end.

Patch embedding — image → patches → linear projection → token sequence.
Vision encoder — reuse your transformer block (bidirectional, no causal mask).
Multimodal fusion — project image tokens into the LLM embedding space and prepend to text tokens (LLaVA-style).
Note: llama.cpp multimodal support is separate and evolving — research current state before designing the deployment path.
Recommended Build Order (the critical path)
0 → 1 → 2 → 3 → 5 → 4 → 7 → 6 → 8 → 9 → 10 → 11 → 12, then 13/14.

Notice I build RMSNorm (5) and SwiGLU (7) before attention (6) — they're simpler, give you quick wins, and let you test the block skeleton. Attention is the hardest part, so you tackle it with all its dependencies (RoPE, norm) already verified.

The two non-negotiable milestones:

Overfit one batch to ~0 loss (end of Phase 9) — proves the architecture is correct.
PyTorch output == llama.cpp output (Phase 12) — proves deployment is faithful.
Hit those two and everything in between is tuning.