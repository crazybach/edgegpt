# EdgeGPT Improvement 1.0: 20M / 2K Mixed-Corpus Training

## Status

Implemented as the first serious training profile after the TinyStories GPU smoke run.

The previous `tinystories_full_gpu_test_1000` run proved the tokenizer, model,
training loop, checkpointing, and generation path. It did not fully train the
corpus: it consumed 4,096,000 of 451,544,175 training tokens (about 0.9%). Its
2-layer, width-128 model has about 2.47M parameters and is intentionally too
small for useful general text generation.

Improvement 1.0 starts a new model. It does not mutate or resume the smoke
checkpoint because the architecture and tokenizer training corpus change.

## Goals

- Increase the trained context from 256 to 2,048 tokens.
- Increase model capacity to about 20M parameters while fitting an RTX 4060 8GB.
- Replace the TinyStories-only tokenizer and corpus with a mixed-domain version.
- Build data through bounded streaming instead of downloading complete web corpora.
- Preserve checkpoint reports and the existing progress watcher.
- Validate the complete path with a 100M-token pilot before committing to 1B tokens.

Instruction tuning is not part of Improvement 1.0. This stage produces a better
base completion model; assistant behavior is a later supervised fine-tuning stage.

## Architecture

Configuration: `configs/improvement_1_20m_2k.yaml`

| Setting | Value |
| --- | ---: |
| Parameters | 18,880,896 |
| Layers | 8 |
| Hidden size | 384 |
| Query heads | 6 |
| KV heads | 2 |
| SwiGLU intermediate | 1,024 |
| Vocabulary | 16,384 |
| Context | 2,048 |
| Weight tying | Enabled |

The profile uses standard RoPE at 2K from the beginning. No interpolation or
position scaling is needed. GQA stores two KV heads while six query heads share
them, reducing inference cache size.

## RTX 4060 Training Profile

| Setting | Value |
| --- | ---: |
| Micro-batch | 1 sequence |
| Sequence length | 2,048 tokens |
| Gradient accumulation | 8 |
| Effective tokens/update | 16,384 |
| Precision | BF16 |
| Chunked output loss | Enabled |
| Activation checkpointing | Enabled |
| Full-run optimizer steps | 61,036 |

The training loader uses a bounded random sampler instead of PyTorch's full
`randperm` shuffle. This prevents a 1B-position memmap from allocating an
approximately 8GB index permutation. Its generator and unconsumed index buffer
are included in checkpoint state for deterministic resume.

Activation checkpointing recomputes each transformer block during backward. It
reduces retained activation memory at the cost of additional compute. It is only
used during training without a KV cache; inference behavior is unchanged.

If BF16 is unsupported by the installed CUDA/PyTorch combination, use FP16. If
the profile still runs out of memory after other GPU processes are stopped, the
next adjustment is reducing `d_model` or `d_ff`; batch size is already one.

## Corpus

The corpus builder is `scripts/prepare_improvement_1_corpus.py`. Token budgets
are measured with the existing TinyStories tokenizer and include one EOS token
per document. The new tokenizer's final count can differ slightly.

### Pilot Profile

| Source | Tokens |
| --- | ---: |
| TinyStories | 20M |
| WikiText-103 raw | 20M |
| FineWeb-Edu deduplicated | 50M |
| Cosmopedia v2 | 10M |
| Total | 100M |

### Full Profile

| Source | Tokens |
| --- | ---: |
| TinyStories | 150M |
| WikiText-103 raw | 100M |
| FineWeb-Edu deduplicated | 550M |
| Cosmopedia v2 | 200M |
| Total | 1B |

TinyStories remains in the mixture to retain simple narrative fluency.
WikiText contributes clean encyclopedic prose. FineWeb-Edu supplies broad,
filtered educational web text. Cosmopedia contributes structured synthetic
explanations. The builder streams Hugging Face datasets with a deterministic
seed and bounded shuffle buffer, reports every 10M tokens, and stops each source
at its configured budget.

The output is:

- `data/improvement_1/train.jsonl`: full mixed documents with source labels.
- `data/improvement_1/tokenizer_train.txt`: proportional 20M-token tokenizer sample.
- `data/improvement_1/manifest.json`: requested and achieved counts plus source metadata.

Writes use `.partial` files and only replace final outputs after all sources
complete. Existing outputs require explicit `--force`.

Python-Edu is deferred because its file contents require a separate Software
Heritage S3 retrieval path. It should be introduced as a measured 5-10% source
once the prose pilot is stable.

## Execution

First inspect the bounded plan without network access:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_improvement_1_corpus.py --profile pilot --dry-run
```

Build the 100M-token pilot corpus:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_improvement_1_corpus.py --profile pilot
```

Train the new mixed-domain tokenizer:

```powershell
.\.venv\Scripts\python.exe scripts\train_tokenizer.py --config configs\improvement_1_20m_2k.yaml
```

Prepare 2K token shards:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_data.py --config configs\improvement_1_20m_2k.yaml
```

Run the pilot. A 100M-token pass is about 6,104 optimizer steps at 16,384
tokens/update:

```powershell
.\.venv\Scripts\python.exe scripts\train.py --config configs\improvement_1_20m_2k.yaml --run-name improvement_1_pilot --max-steps 6104
```

Watch progress and checkpoint events from another terminal:

```powershell
.\.venv\Scripts\python.exe scripts\watch_training.py artifacts\runs\improvement_1_pilot\events.jsonl --follow
```

After the pilot passes its evaluation gates, rebuild with `--profile full`,
retrain the tokenizer from that profile, rebuild shards, and train with the
configuration's 61,036-step target.

## Evaluation Gates

Do not proceed from pilot to full solely because training completes.

- Training and validation loss decrease without sustained divergence.
- Fixed prompts produce more coherent output than the 2.47M TinyStories model.
- The model handles prompts longer than 256 tokens without failure.
- Checkpoint resume produces continuous token and step counts.
- Peak allocated and reserved CUDA memory stay below the device limit.
- Generation remains stable with both cached and full-recompute inference.

Keep a fixed prompt suite covering stories, factual prose, explanation,
summarization-like continuation, and basic code. Save generations at every
checkpoint so quality changes are visible even when loss differences are small.

## References

- [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
- [SmolLM-Corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus)
- [SmolLM training report](https://huggingface.co/blog/smollm)
- [WikiText](https://huggingface.co/datasets/Salesforce/wikitext)
- [Chinchilla scaling study](https://arxiv.org/abs/2203.15556)