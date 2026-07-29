# EdgeGPT Improvement 1.0: 1B Training Test Report

## Status

**Completed successfully.**

The Improvement 1.0 model finished one planned 1B-token training run on an
NVIDIA RTX 4060 8 GB. The final model is a working 18.88M-parameter base text
completion model with a 2,048-token context window. It generates readable
English and short stories, exports to GGUF, and now produces matching greedy
output in PyTorch and llama.cpp.

It is not an instruction-tuned assistant. Direct question answering, factual
recall, reasoning, and multi-turn chat remain unreliable.

## Model Identity

| Item | Value |
| --- | --- |
| Run name | `improvement_1_full_1b` |
| Final step | 61,036 |
| Parameters | 18,880,896 |
| Layers | 8 |
| Hidden size | 384 |
| Query heads | 6 |
| KV heads | 2 |
| SwiGLU intermediate size | 1,024 |
| Vocabulary | 16,384 |
| Context window | 2,048 |
| Weight tying | Enabled |
| Training precision | BF16 autocast |
| Deployment format | FP16 GGUF |

The model uses a Llama-style decoder architecture: RMSNorm, RoPE, grouped-query
attention, SwiGLU, causal attention, and tied input/output embeddings.

## Corpus

The corpus builder produced 1,000,001,641 counted tokens from four sources.
Source revisions were pinned in `data/improvement_1/manifest.json`.

| Source | Tokens | Documents | Share |
| --- | ---: | ---: | ---: |
| TinyStories | 150,000,056 | 696,516 | 15% |
| WikiText-103 raw | 100,000,039 | 706,730 | 10% |
| FineWeb-Edu deduplicated | 550,001,042 | 408,218 | 55% |
| Cosmopedia v2 | 200,000,504 | 201,514 | 20% |
| **Total** | **1,000,001,641** | **2,012,978** | **100%** |

The new 16K byte-level BPE tokenizer was trained from a proportional
20,001,776-token sample. Final packed data used the new tokenizer:

| Split | Tokens | Documents |
| --- | ---: | ---: |
| Train | 994,726,270 | 2,002,771 |
| Validation | 5,275,371 | 10,207 |

Token shards are `uint16` memory maps with EOS separators and 2,048-token
training blocks.

## Training Configuration

| Setting | Value |
| --- | ---: |
| Micro-batch | 1 sequence |
| Gradient accumulation | 8 |
| Effective tokens per optimizer step | 16,384 |
| Maximum steps | 61,036 |
| Peak learning rate | 3.0e-4 |
| Minimum learning rate | 3.0e-5 |
| Warmup | 500 steps |
| Weight decay | 0.1 |
| Gradient clipping | 1.0 |
| Activation checkpointing | Enabled |
| Chunked loss | Enabled |
| Evaluation interval | 10,000 steps |
| Final checkpoint interval | 500 steps |
| Retained checkpoints | Latest 3 |

The successful restarted run began on 2026-07-28 at 17:13 and completed on
2026-07-29 at 14:45. Recorded training time was 77,514.5 seconds, approximately
21 hours 32 minutes.

Typical observed throughput was 12,700-13,000 tokens/second. GPU utilization
was normally 95-98%, total GPU memory use was approximately 6.2-6.8 GB
including desktop applications, and temperature was normally 63-65 C.

## Final Results

| Metric | Result |
| --- | ---: |
| Optimizer steps | 61,036 / 61,036 |
| Tokens consumed | 1,000,013,824 |
| Last training loss | 2.5139 |
| Last training perplexity | 12.35 |
| Final validation loss | 1.8595 |
| Final validation perplexity | 6.42 |
| Best validation loss | 1.8426 |
| Final learning rate | 3.0e-5 |

The final step report stores the best validation loss in its `val_loss` field.
The terminal `eval` event in `events.jsonl` records the final validation loss
of 1.8595. Both values are retained here to avoid treating them as the same
measurement.

The scheduled step count consumed 1,000,013,824 tokens, slightly above the
nominal 1B target because training operates in complete 16,384-token optimizer
steps.

## Failure and Recovery Log

### 1. Hugging Face mirror failure

**Symptom:** Streaming corpus preparation failed while using the machine-level
`HF_ENDPOINT=https://hf-mirror.com`.

**Root cause:** The configured mirror did not reliably serve the pinned dataset
streaming requests.

**Fix:** The run pipeline set `HF_ENDPOINT=https://huggingface.co` explicitly.

**Lesson:** Record the effective dataset endpoint in run metadata and override
machine-global mirrors for reproducible long-running downloads.

### 2. Tokenizer script import failure

**Symptom:** Corpus preparation completed, but tokenizer startup failed with
`ModuleNotFoundError: No module named 'configs'`.

**Root cause:** `scripts/train_tokenizer.py` did not add the repository root to
`sys.path` when launched directly.

**Fix:** Added the same repository-root bootstrap used by the other command-line
scripts.

**Lesson:** Every directly executable repository script needs a consistent
import bootstrap or package-based entry point.

### 3. Wrong data source selected during shard preparation

**Symptom:** JSONL preparation tried to parse
`data/improvement_1/tokenizer_train.txt` and raised `JSONDecodeError`.

**Root cause:** YAML loading used `hasattr(DataConfig, key)`. Dataclass fields
created with `default_factory`, including `source_paths` and `train_files`, are
not discoverable through that check. The loader silently discarded both
configured lists and fell back to auto-discovery.

**Fix:** Configuration filtering now uses `__dataclass_fields__`. Regression
tests assert the exact Improvement 1 corpus and tokenizer paths.

**Lesson:** Never use class-level `hasattr` to enumerate dataclass fields.
Configuration loaders must reject or report ignored keys rather than silently
falling back.

### 4. CUDA out-of-memory at step 758

**Symptom:** The first training attempt stopped during backward propagation at
step 758 with `CUDA error: out of memory`.

**Root cause:** The RTX 4060 was shared with desktop and graphics applications,
and transient external GPU memory pressure exhausted the remaining capacity.
The model itself normally reserved about 904 MiB through PyTorch, but total
device use was much higher.

**Impact:** The initial checkpoint interval was 10,000 steps, so no recovery
checkpoint existed and the first 758 steps were lost.

**Fix:** Restarted training from step zero with checkpoints every 500 steps and
retention of the latest three. Corpus, tokenizer, and packed shards were reused.

**Lesson:** On a shared 8 GB GPU, the first recovery checkpoint must occur
within 10-15 minutes. Sparse user-facing progress reports do not require sparse
recovery checkpoints.

### 5. Unsupported CUDA allocator option

**Symptom:** PyTorch warned that `expandable_segments` was unsupported on this
Windows build.

**Fix:** The warning was treated as non-fatal; training continued normally.

**Lesson:** Do not rely on Linux allocator features for Windows resilience.
Regular checkpoints and GPU workload isolation are the reliable controls.

### 6. GGUF output directory missing

**Symptom:** Initial export failed with `FileNotFoundError` because
`artifacts/exports` did not exist.

**Fix:** Created the export directory before conversion.

**Lesson:** The exporter should create `output_path.parent` itself.

### 7. llama.cpp produced repeated `|` characters

**Symptom:** `llama-cli` and the web chat UI generated repeated vertical bars or
otherwise meaningless output for `Once upon a time`.

**Root cause:** The tokenizer reserves future ChatML tokens such as
`<|im_start|>` and `<|im_end|>`. llama.cpp inferred a ChatML template from those
token names, but the base model was never trained on ChatML-formatted examples.

**Fix:** Added `configs/llama_base_completion.jinja` and forced that template in
CLI validation and `serve_model.bat`.

**Lesson:** Reserved chat tokens do not imply chat-template training. Base
models must explicitly disable or override runtime chat formatting.

### 8. PyTorch and llama.cpp greedy outputs differed

**Symptom:** With identical prompt token IDs, PyTorch predicted comma after
`Once upon a time`, while llama.cpp predicted ` there`.

**Root cause:** The GGUF exporter copied Q and K projection weights directly.
EdgeGPT uses the Hugging Face/Llama split-half RoPE convention, while llama.cpp
expects Q/K rows permuted into its interleaved Llama layout.

**Fix:** Added a per-head RoPE permutation for `attn_q.weight` and
`attn_k.weight`, using query-head and KV-head counts respectively. Added focused
row-order regression tests and regenerated the FP16 GGUF.

**Result:** The corrected llama.cpp greedy continuation matches PyTorch for the
tested 64-token sequence.

**Lesson:** Matching tensor names and shapes is insufficient. Deployment parity
must include deterministic runtime generation, especially for RoPE models.

## Generation Evaluation

### Story completion

Greedy PyTorch generation:

```text
Once upon a time, there was a little girl named Lily. She loved to play
outside in the sunshine. One day, she saw a big, scary dog in the park...
```

The corrected llama.cpp FP16 GGUF produced the same tested continuation.

Sampled story generation was readable and structurally coherent:

```text
In a small village near the sea, Tom discovered a small, broken chest. He was
curious and wanted to find out what was inside the chest...
```

### Factual completion

```text
Prompt: The capital of France is

The capital of France is the capital of the United States...
```

This is fluent but factually wrong and repetitive.

### Explanatory completion

```text
Prompt: Plants need sunlight because

Plants need sunlight because they are not able to absorb water.
The plants need sunlight to grow...
```

The model learned part of the concept but mixes correct and incorrect claims.

### Question answering and chat

Question/answer and `User`/`Assistant` prompts produced unstable formats,
hallucinated facts, and repetition. Few-shot factual prompting did not reliably
recover the correct answer.

### Quality verdict

The model is useful as a small base-model and deployment-path test. It has:

- readable English syntax;
- short narrative structure;
- some broad topical associations;
- stable 2K-context inference;
- successful PyTorch KV-cache generation;
- successful FP16 GGUF and CUDA llama.cpp deployment.

It does not yet have:

- reliable factual recall;
- robust question answering;
- instruction following;
- multi-turn assistant behavior;
- dependable arithmetic or reasoning;
- strong repetition control.

## GGUF and llama.cpp Validation

| Item | Result |
| --- | --- |
| GGUF tensors | 74 |
| GGUF size | 36.6 MiB |
| GGUF weight type | Mostly FP16, with norm gains retained as FP32 |
| Tokenizer parity | Exact for tested prompt |
| Prompt IDs | `[432, 449, 258, 396]` for `Once upon a time` |
| Greedy generation parity | Exact for tested 64-token continuation |
| CUDA offload | All model layers |
| llama.cpp context | 2,048 |
| CLI generation | Approximately 1,960 tokens/second in the parity test |
| Server generation | Approximately 1,850 tokens/second in the UI endpoint smoke test |

BF16 training followed by FP16 GGUF export did not cause the malformed output.
The observed corruption came from chat-template injection and missing RoPE
weight permutation. After both fixes, FP16 output matched PyTorch for the fixed
greedy test.

The exporter relies on llama.cpp pre-tokenizer auto-detection rather than
writing a potentially incorrect `tokenizer.ggml.pre` value. Runtime tokenization
and greedy generation parity passed for the tested prompt.

## Artifacts

| Artifact | Path |
| --- | --- |
| Runtime configuration | `artifacts/runs/improvement_1_full_1b/config.yaml` |
| Event history | `artifacts/runs/improvement_1_full_1b/events.jsonl` |
| Final checkpoint | `artifacts/runs/improvement_1_full_1b/step_61036.pt` |
| Default checkpoint | `artifacts/runs/improvement_1_full_1b/latest.pt` |
| Final step report | `artifacts/runs/improvement_1_full_1b/step_61036.json` |
| Corpus manifest | `data/improvement_1/manifest.json` |
| Packed data metadata | `artifacts/data/improvement_1_mixed_2k/metadata.json` |
| Tokenizer | `artifacts/tokenizer/improvement_1_16k` |
| FP16 GGUF | `artifacts/exports/improvement_1_full_1b-f16.gguf` |
| Web launcher | `serve_model.bat` |
| Base completion template | `configs/llama_base_completion.jinja` |

## Reproduction

PyTorch generation:

```powershell
.\.venv\Scripts\python.exe scripts\generate.py `
  --config artifacts\runs\improvement_1_full_1b\config.yaml `
  --checkpoint artifacts\runs\improvement_1_full_1b\latest.pt `
  --prompt "Once upon a time" `
  --max-new-tokens 64 `
  --temperature 0 `
  --device cuda
```

Web serving:

```powershell
.\serve_model.bat
```

The launcher serves only on `http://127.0.0.1:8081/`, opens the built-in
llama.cpp UI, uses CUDA, and keeps the server attached to the console. Press
`Ctrl+C` or close the console to stop it.

## Recommendations for Improvement 1.1

1. **Instruction tune instead of repeating base pretraining immediately.**
   Use a filtered supervised mixture containing conversational answers,
   factual QA, explanations, summarization, and refusal/safety examples.
2. **Build a fixed evaluation suite.** Store prompt, seed, decoding settings,
   expected properties, and outputs for every candidate checkpoint.
3. **Measure factual accuracy separately from language-model loss.** Add small
   held-out QA, arithmetic, reading-comprehension, and repetition benchmarks.
4. **Add llama.cpp parity to automated release tests.** Compare prompt token
   IDs and greedy output after every GGUF exporter change.
5. **Make pipeline stages resumable by artifact validation.** Corpus,
   tokenizer, shard, and training stages should skip only when manifests and
   hashes match the requested configuration.
6. **Keep frequent recovery checkpoints on shared GPUs.** A 500-step interval
   with three retained files worked well for this machine.
7. **Improve progress notification.** Report stage transitions, checkpoints,
   failures, ETA, GPU memory, loss, and final evaluation without requiring
   repeated manual status requests.
8. **Expand tokenizer parity coverage.** Keep pre-tokenizer auto-detection, but
   add punctuation, whitespace, Unicode, and digit-heavy parity cases.

## Final Conclusion

Improvement 1.0 achieved its engineering objective: a complete 1B-token,
2K-context training run on an RTX 4060 8 GB, followed by correct PyTorch and
llama.cpp inference from the same trained weights.

The run also demonstrated that deployment correctness cannot be inferred from
fluent output alone. Configuration loading, checkpoint cadence, chat formatting,
tokenizer parity, and RoPE tensor layout all required explicit validation.

The model should be retained as the first serious EdgeGPT base-model baseline.
The next version should focus on supervised instruction tuning and measurable
task quality rather than another undirected increase in base-training tokens.
