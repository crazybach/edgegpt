# EdgeGPT Phase 12 — Export to GGUF & Model Format Comparison

## Summary

Phase 12 exports a trained EdgeGPT checkpoint to the **GGUF** format and
validates that the resulting model produces **identical output** to the
original PyTorch model when run through **llama.cpp** on Windows. This is
the second non-negotiable milestone from `plan.md`:

> *"PyTorch output == llama.cpp output (Phase 12) — proves deployment is
> faithful."*

---

## 1. Model Format Comparison

Four families of model formats were evaluated for EdgeGPT's target
deployment scenarios: **PC Windows (llama.cpp)** for testing, and **mobile
phone** as the ultimate target.

### 1.1 Format Comparison Matrix

| Dimension | GGUF | Safetensors | ONNX | CoreML |
|-----------|------|-------------|------|--------|
| **Primary use case** | CPU/local inference | Model sharing, HF training | Cross-platform, browser | Apple devices |
| **Single-file** | ✅ Yes (weights + tokenizer + metadata) | ❌ Weights only (config separate) | ✅ Yes (graph + weights) | ✅ Yes (.mlpackage) |
| **Quantization quality** | ⭐ Best-in-class (Q4_K_M, IQ series) | 🟡 Via bitsandbytes (limited) | ❌ Weak (int+scale tensor split) | 🟡 Core ML Tools (good on ANE) |
| **CPU performance** | ⭐ Best (hand-tuned SIMD: AVX2, NEON) | 🟡 Via PyTorch CPU backend | 🟡 Via ONNX Runtime | 🟡 Apple CPU only |
| **GPU support** | ✅ CUDA, Metal, Vulkan, SYCL, ROCm | ✅ CUDA, MPS (via PyTorch) | ✅ CUDA, DirectML, OpenVINO | ✅ Apple GPU/ANE |
| **Loading speed** | ✅ `mmap()` sub-second | ✅ Lazy load, partial | 🟡 Moderate | ✅ Fast on Apple |
| **Mobile phone** | ✅ ARM NEON, works on Android/iOS via llama.cpp | ❌ No native mobile runtime | ✅ ONNX Runtime Mobile | ⭐ Best on iOS (ANE) |
| **Browser (WASM)** | ❌ | ❌ | ✅ (transformers.js) | ❌ |
| **Fine-tunable after export** | ❌ Not practical | ✅ Native PyTorch | ❌ Complex | ❌ Not practical |
| **Security** | 🟡 Binary format | ⭐ No code execution | 🟡 Good | ⭐ Apple sandbox |
| **Ecosystem** | llama.cpp, Ollama, LM Studio, GPT4All | HuggingFace (default) | ONNX Runtime, TF.js | Apple ecosystem only |

### 1.2 Per-Platform Winner

| Deployment Target | Best Format | Why |
|---|---|---|
| **PC Windows (testing)** | **GGUF** | llama.cpp already working on user's PC; fastest CPU inference; easy quant experimentation |
| **Android phone** | **GGUF** | llama.cpp ARM NEON kernels; 4-bit quant fits in phone RAM; single-file deployment |
| **iOS phone** | **GGUF** or **CoreML** | GGUF via llama.cpp Metal; CoreML if ANE acceleration is needed |
| **Browser demo** | **ONNX** via transformers.js | Only format with WASM/WebGPU runtime |
| **HuggingFace sharing** | **Safetensors** | Native HF ecosystem; can distribute alongside GGUF |

### 1.3 Decision: GGUF

GGUF is the clear choice for EdgeGPT for these reasons:

1. **Project goal alignment.** `plan.md` explicitly targets llama.cpp + GGUF as the deployment path
2. **Single workflow.** One converter script produces a file that works on PC (testing) and phone (deployment)
3. **Best quantization.** The K-quant family (Q4_K_M, Q5_K_M) offers compression ratios of 3–4× with <2% perplexity degradation — essential for fitting a model on a phone with limited RAM
4. **Tokenization is self-contained.** The GGUF file embeds the full tokenizer configuration (BPE merges, vocab, special tokens), so no separate tokenizer file is needed at inference time
5. **User's setup.** llama.cpp is already working on the user's Windows PC — zero additional tooling needed for testing

---

## 2. Quantization Type Comparison

Once the model is in GGUF format (FP16), llama.cpp's `llama-quantize` tool
compresses it. Below is the full quantization comparison, with special
attention to small models (<50M parameters) since EdgeGPT is ~21.4M params.

### 2.1 Quantization Reference Table (7B model, WikiText-2)

| Quant Type | Bits/Weight | Size (7B) | Perplexity | Degradation vs FP16 | Notes |
|------------|-------------|-----------|------------|---------------------|-------|
| **FP16** | 16.0 | 13.0 GB | 5.9565 | baseline | Original quality |
| **FP32** | 32.0 | 26.0 GB | 5.9565 | 0 (same) | For validation only |
| Q8_0 | 8.5 | 7.0 GB | 5.9584 | **+0.03%** | Nearly lossless |
| Q6_K | 6.56 | 5.5 GB | 5.9642 | **+0.13%** | Best quality/size ratio |
| **Q5_K_M** | 5.5 | 4.8 GB | 5.9796 | **+0.39%** | ⭐ Safe for small models |
| **Q4_K_M** | 4.5 | 4.1 GB | 6.0565 | **+1.68%** | ⭐ Community default |
| Q4_K_S | 4.5 | 3.9 GB | 6.1125 | +2.62% | Faster, slightly lower quality |
| Q4_0 (legacy) | 4.0 | 3.6 GB | ~5.82 | ~+2.2% | Older symmetric scheme |
| Q3_K_M | 3.44 | 3.3 GB | 6.3184 | +6.07% | ⚠️ Not for small models |
| Q3_K_S | 3.25 | 3.1 GB | 6.44 | +8.1% | ❌ Avoid |
| Q2_K | 2.56 | 2.7 GB | 6.8673 | +15.3% | ❌ Not recommended |

### 2.2 Small Model Consideration (<50M parameters)

Smaller models are **more sensitive to quantization error** than large ones.
A 7B model has enough redundancy to absorb 4-bit rounding noise; a 21M model
has proportionally less.

| Model Size | Q4_K_M Degradation | Q5_K_M Degradation | Recommendation |
|------------|-------------------|-------------------|----------------|
| 7B | ~1.7% | ~0.4% | Q4_K_M fine |
| 1-3B | ~2-4% | ~1-2% | Q5_K_M safer |
| **<100M (EdgeGPT)** | **~3-6%** | **~1.5-3%** | **Q5_K_M recommended** |

### 2.3 EdgeGPT-Specific Size Estimates

With default config: vocab=16384, d_model=512, n_layers=8, ~21.4M params:

| Quant Type | Model Size on Disk | RAM at Inference | Notes |
|------------|-------------------|-------------------|-------|
| FP32 | ~85 MB | ~90 MB | Reference only, too large for mobile |
| FP16 | ~43 MB | ~50 MB | Validation baseline |
| Q8_0 | ~23 MB | ~30 MB | Nearly lossless, good fallback |
| Q6_K | ~18 MB | ~25 MB | Excellent quality retention |
| **Q5_K_M** | **~15 MB** | **~22 MB** | ⭐ Recommended for EdgeGPT |
| Q4_K_M | ~12 MB | ~19 MB | Usable if size-constrained |
| Q3_K_M | ~9 MB | ~16 MB | ❌ Too degraded for 21M param model |
| Q2_K | ~7 MB | ~14 MB | ❌ Severe quality loss |

### 2.4 Decision: Three-tier Quantization Strategy

| Tier | Quant | Purpose |
|------|-------|---------|
| **Validation** | FP16 | Compare PyTorch FP32 vs llama.cpp FP16 greedy output — must be identical |
| **Primary** | **Q5_K_M** | Best quality/size tradeoff for a 21M model. ~15 MB file, fits in any phone's RAM |
| **Optional** | Q4_K_M | Smaller (~12 MB) alternative if size is critical. Accept ~1-3% extra degradation |

---

## 3. Tensor Name Mapping

EdgeGPT's attribute naming (established in Phase 9) follows the
Llama/HuggingFace convention. The `gguf-py` library expects GGUF-standard
tensor names. The mapping is:

### 3.1 Per-Layer Mapping

| EdgeGPT state_dict key | GGUF tensor name | Notes |
|------------------------|------------------|-------|
| `embed_tokens.embedding.weight` | `token_embd.weight` | `[vocab_size, d_model]` → GGUF stores same shape |
| `layers.{N}.attention_norm.weight` | `blk.{N}.attn_norm.weight` | RMSNorm gain, shape `[d_model]` |
| `layers.{N}.attention.q_proj.weight` | `blk.{N}.attn_q.weight` | `[n_heads×head_dim, d_model]` |
| `layers.{N}.attention.k_proj.weight` | `blk.{N}.attn_k.weight` | `[n_kv_heads×head_dim, d_model]` |
| `layers.{N}.attention.v_proj.weight` | `blk.{N}.attn_v.weight` | `[n_kv_heads×head_dim, d_model]` |
| `layers.{N}.attention.o_proj.weight` | `blk.{N}.attn_output.weight` | `[d_model, n_heads×head_dim]` |
| `layers.{N}.attention.rope.inv_freq` | (metadata only) | RoPE frequencies → `rope.freq_base` in metadata |
| `layers.{N}.mlp_norm.weight` | `blk.{N}.ffn_norm.weight` | RMSNorm gain, shape `[d_model]` |
| `layers.{N}.mlp.gate_proj.weight` | `blk.{N}.ffn_gate.weight` | `[d_ff, d_model]` |
| `layers.{N}.mlp.up_proj.weight` | `blk.{N}.ffn_up.weight` | `[d_ff, d_model]` |
| `layers.{N}.mlp.down_proj.weight` | `blk.{N}.ffn_down.weight` | `[d_model, d_ff]` (residual projection) |
| `norm.weight` | `output_norm.weight` | Final RMSNorm gain, shape `[d_model]` |
| `lm_head.weight` (untied only) | `output.weight` | `[vocab_size, d_model]` — when `tie_embeddings=true`, omitted (llama.cpp deduces from token_embd) |

### 3.2 Metadata Keys (GGUF KV pairs)

These must be written into the GGUF header so llama.cpp knows the
architecture parameters:

| GGUF Key | Source (EdgeGPT config) | Example value |
|----------|------------------------|---------------|
| `general.architecture` | hardcoded `"llama"` | `"llama"` |
| `llama.context_length` | `config.model.max_seq_len` | `2048` |
| `llama.block_count` | `config.model.n_layers` | `8` |
| `llama.embedding_length` | `config.model.d_model` | `512` |
| `llama.feed_forward_length` | `config.model.d_ff` | `1408` |
| `llama.attention.head_count` | `config.model.n_heads` | `8` |
| `llama.attention.head_count_kv` | `config.model.n_kv_heads` | `4` |
| `llama.rope.freq_base` | `config.model.rope_theta` | `10000.0` |
| `llama.attention.layer_norm_epsilon` | `config.model.norm_eps` | `1e-5` |
| `llama.vocab_size` | `config.model.vocab_size` | `16384` |
| `tokenizer.ggml.model` | hardcoded `"gpt2"` (BPE) | `"gpt2"` |

### 3.3 Tokenizer Serialization

The EdgeGPT tokenizer is a byte-level BPE tokenizer (Phase 1). GGUF
requires the tokenizer to be embedded as metadata so llama.cpp can
encode/decode without a separate tokenizer file.

For a BPE tokenizer (`tokenizer.ggml.model = "gpt2"`), the required
metadata includes:

| Key | Content |
|-----|---------|
| `tokenizer.ggml.model` | `"gpt2"` |
| `tokenizer.ggml.tokens` | List of token strings (vocab_size entries) |
| `tokenizer.ggml.scores` | List of token scores |
| `tokenizer.ggml.token_type` | List of token type integers (1=normal, 2=control, 3=user_defined, 5=unused) |
| `tokenizer.ggml.merges` | BPE merge rules as space-joined pairs |
| `tokenizer.ggml.bos_token_id` | BOS token ID |
| `tokenizer.ggml.eos_token_id` | EOS token ID |
| `tokenizer.ggml.add_bos_token` | `true` or `false` |

---

## 4. Conversion Architecture

### 4.1 Strategy: Direct gguf-py Writer

We will **not** use `convert_hf_to_gguf.py` because EdgeGPT is not a
HuggingFace model (no `config.json` in `transformers` format, no
`LlamaForCausalLM` class). Instead, we write a **custom converter script**
that uses the `gguf` Python library (`gguf-py`) directly:

```
scripts/export_gguf.py
```

This script:

1. Loads the trained EdgeGPT checkpoint (`.pt` file from Phase 10)
2. Reads the model config from the checkpoint
3. Creates a `GGUFWriter` with architecture `"llama"`
4. Writes all metadata KV pairs from config values
5. Maps each `state_dict` key to its GGUF tensor name and writes the tensor
6. Serializes the tokenizer vocabulary, merges, and special tokens
7. Produces `edgegpt-f16.gguf` (FP16 weights)

### 4.2 Data Flow

```
Phase 10 checkpoint (step_N.pt)
    │
    ├── model state_dict  ──→  tensor name mapper  ──→  GGUFWriter.add_tensor()
    ├── config (asdict)   ──→  metadata builder    ──→  GGUFWriter.add_uint32/float32()
    └── tokenizer artifacts ─→  tokenizer extractor ──→  GGUFWriter tokenizer metadata
                                    │
                                    ▼
                            edgegpt-f16.gguf
                                    │
                            llama-quantize.exe
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
            edgegpt-Q4_K_M.gguf  edgegpt-Q5_K_M.gguf  edgegpt-Q8_0.gguf
```

### 4.3 Validation Pipeline

```
1. Load PyTorch model, run greedy generation on N prompts
2. Load edgegpt-f16.gguf in llama.cpp, run greedy generation on same prompts
3. Assert: all tokens match exactly (↓ this is the milestone)
4. Quantize to Q4_K_M, Q5_K_M, Q8_0
5. Compare each quant's greedy output vs FP16
6. Report per-token match rate and perplexity difference
```

---

## 5. EdgeGPT Size Estimates for Mobile

This is a forward-looking estimate for when mobile deployment is needed.

### 5.1 Model occupies (after quantization)

| Quant | File Size | RAM at Runtime | Fits in 4GB phone? | Fits in 6GB phone? |
|-------|-----------|---------------|--------------------|--------------------|
| FP16 | ~43 MB | ~50 MB | ✅ Yes | ✅ Yes |
| Q5_K_M | ~15 MB | ~22 MB | ✅ Yes | ✅ Yes |
| Q4_K_M | ~12 MB | ~19 MB | ✅ Yes | ✅ Yes |

A typical Android flagship phone (2024–2025, e.g., Snapdragon 8 Gen 3)
has 12–16 GB RAM. Even a mid-range phone with 6 GB RAM has >100× headroom.
The limiting factor is **not RAM** but **inference speed** on mobile CPUs.

For reference:
- llama.cpp on Snapdragon 8 Gen 2 (ARM Cortex-X3): ~15–30 tokens/sec for a 7B Q4_K_M model
- For EdgeGPT's 21M params: expect ~200–500 tokens/sec on modern phone CPU

---

## 6. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| GGUF `"llama"` arch expects `bias` tensors EdgeGPT doesn't have | Low | Low | Llama-family models are bias-free; llama.cpp handles this |
| BPE tokenizer not recognized by GGUF's pre-tokenizer detector | Medium | High | Write tokenizer metadata manually via gguf-py API; test roundtrip |
| Weight tying: llama.cpp expects separate `output.weight` | Low | Medium | Check llama.cpp behavior; either duplicate the tensor or use flag |
| Float precision differences (PyTorch fp32 vs GGUF fp16) cause token mismatch | Medium | Medium | Use atol=1e-3 for logit comparison; confirm greedy paths match |
| GGUF `add_tensor()` expects different dimension ordering than PyTorch | Low | Medium | Verify GGUF dimension convention; transpose if needed |
| llama.cpp not installed or wrong version on user's PC | Low | Low | Document the exact llama.cpp build steps |

---

## 7. Implementation Checklist

1. **Install `gguf` Python library** — `pip install gguf` (part of llama.cpp repo or PyPI)
2. **Create `scripts/export_gguf.py`** — custom converter script
3. **Implement tensor name mapper** — EdgeGPT state_dict keys → GGUF names
4. **Implement metadata writer** — config values → GGUF KV pairs
5. **Implement tokenizer serialization** — extract BPE vocab, merges, special tokens
6. **Produce `edgegpt-f16.gguf`** — FP16 GGUF file from trained checkpoint
7. **Validate with llama.cpp** — greedy generation comparison (PyTorch vs llama.cpp)
8. **Quantize** — produce Q4_K_M, Q5_K_M, Q8_0 variants
9. **Compare quantized outputs** — measure degradation per quantization level
10. **Create test** — `tests/test_export.py` with token-match assertions

---

## 8. References

- GGUF specification: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
- llama.cpp HOWTO-add-model: https://github.com/ggml-org/llama.cpp/blob/master/docs/development/HOWTO-add-model.md
- gguf-py library (GGUFWriter): https://pypi.org/project/gguf/
- K-quant design: https://github.com/ggml-org/llama.cpp/pull/1684
- Quantization benchmark (7B model): HuggingFace skills quantization reference
- Small model quantization sensitivity: "Which Quantization Should I Use?" (Jan 2025), arXiv:2601.14277
- Q4_K_M vs Q5_K_M analysis: https://zeroentropy.dev/concepts/gguf/
- Model format comparison: HuggingFace "Common AI Model Formats" blog (2024)
- ONNX vs GGUF: Google Cloud Community "Choosing the right format" (2024)
- MobileQuant: Mobile-friendly Quantization for On-device Language Models (EMNLP 2024)
