# EdgeGPT Deployment & Testing Guide

How to export a trained EdgeGPT checkpoint to GGUF and run inference
with llama.cpp on Windows (CUDA).

## Prerequisites

### 1. llama.cpp with CUDA

Location on this machine:

```
D:\workspace\llamacpp\source\llama.cpp
```

Build binaries are in `build-cuda/bin/`:

| Binary | Purpose |
|--------|---------|
| `llama-cli.exe` | Interactive CLI inference |
| `llama-server.exe` | HTTP API server |
| `llama.dll` | Shared library |
| `ggml-cuda.dll` | CUDA backend (~130 MB) |

> **Build note** (if you need to rebuild):
> ```bash
> cd D:\workspace\llamacpp\source\llama.cpp
> cmake -B build-cuda -DGGML_CUDA=ON
> cmake --build build-cuda --config Release
> ```

### 2. EdgeGPT Project Environment

```bash
cd d:\workspacep\edgegpt
.venv\Scripts\activate
pip install gguf
```

---

## Quick Test — Already Working

The tinystories checkpoint has been successfully exported and tested:

```
Model:     artifacts/runs/tinystories_full_gpu_test_1000/latest.pt
Tokenizer: artifacts/tokenizer/main_16k/
GGUF:      deployment_test/edgegpt-tinystories-f16.gguf (5.3 MiB)
```

### Test Result

```
> Once upon a time

 and a lot of it was a little boy named Lily. He went home with a big voice.

[ Prompt: 13027.9 t/s | Generation: 3770.9 t/s ]
```

Model loaded successfully. Generation speed: **~3,770 tokens/sec** on
NVIDIA RTX 4060 (8 GB) with CUDA.

---

## Export a Checkpoint to GGUF

```bash
cd d:\workspacep\edgegpt
.venv\Scripts\activate

python scripts/export_gguf.py \
    --checkpoint artifacts/runs/<run_name>/latest.pt \
    --tokenizer-dir artifacts/tokenizer/main_16k \
    --output deployment_test/edgegpt-<name>-f16.gguf \
    --outtype f16
```

| `--outtype` | File Size | Quality |
|-------------|-----------|---------|
| `f16` | ~1/2 of f32 | Good (default) |
| `f32` | Full size | Reference/validation |

---

## Run Inference with llama.cpp

### Interactive chat (recommended for testing)

```bash
"D:\workspace\llamacpp\source\llama.cpp\build-cuda\bin\llama-cli.exe" \
    -m deployment_test/edgegpt-tinystories-f16.gguf \
    -p "Once upon a time" \
    -n 100 \
    -t 4 \
    --temp 0.8
```

| Flag | Purpose | Default |
|------|---------|---------|
| `-m` | Path to .gguf model file | (required) |
| `-p` | Prompt text | (required) |
| `-n` | Max tokens to generate | `-1` (unlimited) |
| `-t` | CPU thread count | `-1` (auto) |
| `--temp` | Sampling temperature | `0.8` |
| `-c` | Context size | model default |
| `-ngl` | GPU layers (offload to CUDA) | all layers |

> The CUDA build auto-offloads to GPU — no need for `-ngl` flag.

### HTTP API server + Web UI (one-click)

Use the provided launch scripts — they start the server, open the Web UI,
and handle clean shutdown:

```bash
# PowerShell (recommended — clean shutdown on window close)
powershell -ExecutionPolicy Bypass -File deployment_test/serve.ps1

# Or batch file (quick launch)
deployment_test\serve.bat
```

| Script | Best for |
|--------|----------|
| `serve.ps1` | Clean shutdown when closing the window (registered exit handler) |
| `serve.bat` | Quick launch, familiar `.bat` feel |

Both scripts accept optional arguments:

```bash
serve.bat "path\to\model.gguf" 9090
serve.ps1 -Model "path\to\model.gguf" -Port 9090 -Temp 1.0
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `Model` | `edgegpt-tinystories-f16.gguf` | Path to `.gguf` file |
| `Port` | `8080` | HTTP port |
| `Host` | `0.0.0.0` | Bind address |
| `Temp` | `0.8` | Sampling temperature |
| `CtxSize` | `2048` | Context window size |

**What you get:**
- **Web UI** at `http://localhost:8080` — llama.cpp's built-in chat interface
- **REST API** — full OpenAI-compatible endpoints (see below)
- **4 parallel slots** — handles up to 4 concurrent users/requests

**To stop:** Close the console window, press Ctrl+C, or press any key
(in the batch file). The PowerShell script registers an exit handler that
kills the server process cleanly even when the window is closed.

### Manual server launch (advanced)

```bash
"D:\workspace\llamacpp\source\llama.cpp\build-cuda\bin\llama-server.exe" \
    -m deployment_test/edgegpt-tinystories-f16.gguf \
    --host 0.0.0.0 \
    --port 8080 \
    -c 2048 \
    --temp 0.8
```

### REST API reference

```bash
# Completion
curl http://localhost:8080/completion \
    -H "Content-Type: application/json" \
    -d '{"prompt": "Once upon a time", "n_predict": 50, "temperature": 0.8}'

# OpenAI-compatible chat
curl http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"messages": [{"role": "user", "content": "Hello"}], "temperature": 0.8}'

# Health check
curl http://localhost:8080/health

# Server info
curl http://localhost:8080/v1/slots
```

---

## Quantization (if llama-quantize is available)

The current build doesn't include `llama-quantize.exe`. To add it, rebuild
with the quantize target:

```bash
cd D:\workspace\llamacpp\source\llama.cpp
cmake -B build-cuda -DGGML_CUDA=ON
cmake --build build-cuda --config Release --target llama-quantize
```

Then quantize:

```bash
# Q8_0 (nearly lossless, ~1/2 size of f16)
llama-quantize.exe edgegpt-f16.gguf edgegpt-Q8_0.gguf Q8_0

# Q5_K_M (good balance for small models)
llama-quantize.exe edgegpt-f16.gguf edgegpt-Q5_K_M.gguf Q5_K_M

# Q4_K_M (smallest usable)
llama-quantize.exe edgegpt-f16.gguf edgegpt-Q4_K_M.gguf Q4_K_M
```

| Quant | ~Size (21M model) | Quality |
|-------|--------------------|---------|
| FP16 | 43 MB | Baseline |
| Q8_0 | 23 MB | ~lossless |
| Q5_K_M | 15 MB | +0.4% perplexity |
| Q4_K_M | 12 MB | +1.7% perplexity |

---

## Troubleshooting

### "unknown pre-tokenizer type"

The export script no longer writes `tokenizer.ggml.pre`. llama.cpp
auto-detects BPE pre-tokenization from the tokenizer data.

### Tensor shape mismatch

The export script now includes `added_tokens` in the vocab (not just
`model.vocab` from tokenizer.json). Ensure the tokenizer directory
contains:
- `tokenizer.json` (vocab + merges + added_tokens)
- `special_tokens_map.json` (special token IDs)
- `tokenizer_config.json` (BOS/EOS behavior)

### CUDA fails to load model

- CUDA DLLs (`ggml-cuda.dll`, `ggml-cpu.dll`, `ggml-base.dll`, `ggml.dll`)
  must be in the same directory as `llama-cli.exe`, or in `PATH`.
- Ensure `llama.dll` is also present.

### Out of VRAM

The EdgeGPT model is tiny (~5-43 MiB) and should never exceed VRAM.
If you see OOM errors, check that no other CUDA process is running:
```bash
nvidia-smi
```

---

## Current Status

| Checkpoint | Config | Steps | Val Loss | GGUF | Tested |
|------------|--------|-------|----------|------|--------|
| `tinystories_full_gpu_test_1000` | d_model=128, n_layers=2 | 1000 | 3.887 | ✅ | ✅ |
| `tinystories_smoke_gpu_first` | d_model=128, n_layers=2 | ? | ? | — | — |
| `phase10_smoke` | d_model=256, n_layers=4 | 2 | 9.813 | — | — |
