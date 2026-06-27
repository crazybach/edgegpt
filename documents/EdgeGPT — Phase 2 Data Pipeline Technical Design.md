# EdgeGPT Phase 2 - Data Pipeline Technical Design

## Summary

Phase 2 turns raw text documents into fixed-length causal-language-model
training batches:

```text
input_ids: LongTensor[B, T]
targets:   LongTensor[B, T]
```

The default architecture is offline tokenization plus local binary token shards.
That means the expensive tokenizer pass runs once, and later training reads
compact integer blocks from disk. This is the right default for the current
CPU-friendly laptop setup.

## Research Comparison

| Architecture | Pros | Cons | Decision |
| --- | --- | --- | --- |
| On-the-fly streaming tokenization | Flexible, minimal preprocessing | CPU bottleneck during training | Support later as an adapter |
| Hugging Face streaming datasets | Good for huge remote datasets and dataset interleaving | Network/cache complexity, harder deterministic local tests | Optional future source |
| nanoGPT-style `.bin` memmap | Simple, fast, proven for small LLMs | Minimal metadata | Default v1 |
| Megatron-style `.bin` + `.idx` | Scalable, rich indexing | More complexity than needed now | Defer |

References:

- nanoGPT OpenWebText preparation writes token IDs to `train.bin` / `val.bin`: https://github.com/karpathy/nanoGPT/blob/master/data/openwebtext/prepare.py
- Hugging Face streaming datasets: https://huggingface.co/docs/datasets/stream
- PyTorch map-style and iterable-style datasets: https://docs.pytorch.org/docs/2.12/data.html
- Megatron-LM indexed dataset design: https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/datasets/indexed_dataset.py

## Locked Decisions

- Default source backend: local UTF-8 text files.
- Optional v1 source backend: JSONL with configurable `text_column`.
- Default storage backend: flat `uint16` `.bin` shards read by NumPy memmap.
- EOS is appended after every document before concatenation.
- Pretraining batches use no padding; every sample is a fixed block.
- Each dataset item reads `T + 1` adjacent tokens, then returns shifted inputs
  and targets.
- Leftover tail tokens that cannot form one complete shifted block are dropped
  by the dataset length calculation.
- `train` must contain at least `block_size + 1` tokens after tokenization.

## Module Layout

- `data/pipeline/base.py`: swappable interfaces and shared document contracts.
- `data/pipeline/sources.py`: local text and JSONL document sources.
- `data/pipeline/shards.py`: memmap shard writer and metadata helpers.
- `data/pipeline/dataset.py`: memmap block dataset and DataLoader provider.
- `data/pipeline/prepare.py`: offline tokenization and shard preparation.
- `scripts/prepare_data.py`: CLI entrypoint.
- `tests/test_data_pipeline.py`: Phase 2 contract tests.

## Public API

```python
from data.pipeline import prepare_data, build_train_loader

prepare_data(config)
loader = build_train_loader(config, split="train")
batch = next(iter(loader))
```

Batch format:

```python
{
    "input_ids": LongTensor[B, T],
    "targets": LongTensor[B, T],
}
```

Training code should only depend on this public API. Future source or storage
backends can change internals without changing the batch contract.

## Config Contract

`DataConfig` includes Phase 2 controls:

```yaml
data:
  dataset: "tinystories"
  data_dir: "./data/tinystories"
  val_split: 0.005
  seed: 42
  source_type: "local_text"
  source_paths: []
  text_column: "text"
  cache_dir: "./artifacts/data/tinystories"
  storage_type: "memmap_bin"
  block_size: null
  shuffle_buffer_size: 10000
  num_workers: 0
```

`block_size: null` means use `model.max_seq_len`.

## Data Flow

1. Build a `DocumentSource` from config.
2. Load the Phase 1 tokenizer with `load_tokenizer(config)`.
3. Deterministically assign documents to train/val with `data.seed`.
4. Encode each document.
5. Append `<|eos|>` to mark the document boundary in the flat stream.
6. Validate token IDs are inside `[0, vocab_size)`.
7. Write `train.bin`, `val.bin`, and `metadata.json` under `data.cache_dir`.
8. Read shards with `MemmapTokenBlockDataset`.
9. Return shifted `input_ids` and `targets`.

## Concept Notes From Review

### What a shard is

A shard is a saved token-ID array on disk. It is not raw text anymore.

Example:

```text
doc1: "hello world" -> [101, 205, 99]
doc2: "I like AI"  -> [42, 88, 300]
```

After appending EOS between documents:

```python
eos_id = 16131
all_tokens = [101, 205, 99, 16131, 42, 88, 300, 16131]
```

`train.bin` stores those integers as compact `uint16` binary data. Later,
NumPy memmap treats the file like an array and reads only the window needed for
one batch, instead of loading the whole shard into RAM.

Shard size depends on total dataset tokens, not block size. Because each token
is `uint16`, storage is roughly:

```text
1,000 tokens       -> ~2 KB
1,000,000 tokens   -> ~2 MB
100,000,000 tokens -> ~200 MB
```

### Why `input_ids` and `targets` are shifted

For causal language modeling, `targets` is the answer key for next-token
prediction:

```python
block = [101, 205, 99, 16131, 42]
input_ids = [101, 205, 99, 16131]
targets   = [205, 99, 16131, 42]
```

The model sees each prefix and predicts the next token:

| Input context | Target |
| --- | --- |
| `101` | `205` |
| `101, 205` | `99` |
| `101, 205, 99` | `16131` |
| `101, 205, 99, 16131` | `42` |

`input_ids` alone is not enough during training because the loss needs an
answer key. During inference there is no `targets` array; the model predicts the
next token, appends it to the context, and repeats.

This shifted next-token objective remains the core pretraining pattern for
modern decoder-only LLMs. Some frontier systems add extra objectives, such as
multi-token prediction, but standard next-token prediction is still the base.

### Batch and block dimensions

`input_ids` is two-dimensional:

```text
[B, T]
```

- `B`: batch size, or how many independent token windows are processed at once.
- `T`: block size/context length, or how many tokens are in each window.

Examples:

| Shape | Meaning | Tokens per step |
| --- | --- | ---: |
| `[1, 512]` | 1 sequence, 512 tokens each | 512 |
| `[2, 512]` | 2 sequences, 512 tokens each | 1024 |
| `[8, 512]` | 8 sequences, 512 tokens each | 4096 |
| `[8, 2048]` | 8 sequences, 2048 tokens each | 16384 |

With `configs/cpu.yaml`, `batch_size = 1` and `max_seq_len = 512`, so batches
are `[1, 512]`. With `configs/default.yaml`, batches are `[8, 2048]`.

After Phase 3 embeddings, the tensor becomes:

```text
[B, T] -> [B, T, d_model]
```

For the CPU config:

```text
[1, 512] -> [1, 512, 256]
```

### Why train and validation shards both exist

`train.bin` is data the model learns from. `val.bin` is held-out data used to
check whether training generalizes:

| Step | Train loss | Val loss | Meaning |
| ---: | ---: | ---: | --- |
| 1,000 | 5.0 | 5.1 | learning normally |
| 10,000 | 3.0 | 3.2 | improving and generalizing |
| 50,000 | 1.0 | 4.5 | likely overfitting |

For tiny smoke tests, train-only data can be enough. For real training, even a
small validation split is important because train loss alone can hide
memorization.

### Parallelism decision

Phase 2 is single-process by default. That is intentional for the current
laptop-scale CPU setup:

```text
read text -> tokenize -> append EOS -> write token IDs
```

Large LLM pipelines usually parallelize or distribute preprocessing because
they process enormous corpora and must keep many training workers fed. EdgeGPT
does not need that complexity yet. The interface boundaries (`DocumentSource`,
`TokenShardWriter`, `TokenBlockDataset`, `BatchProvider`) are where future
parallel or distributed implementations can plug in without changing the
training API.

## Commenting Requirements

Phase 2 code uses docstrings and focused comments for invariants that later
phases depend on:

- why EOS is inserted between documents
- why shards use `uint16`
- why memmap is the default for laptop-scale training
- why batches sample `T + 1` tokens
- why targets are shifted by one
- why tiny leftover tails are dropped in v1

Comments should explain design constraints, not restate obvious Python syntax.

## Automated Tests

`tests/test_data_pipeline.py` covers:

- shard and metadata creation
- token ID bounds
- EOS insertion between documents
- fixed batch shape `[B, T]`
- shifted target contract: `targets[:, :-1] == input_ids[:, 1:]`
- deterministic loader order for the same seed
- clear failure when train data cannot produce one full block

Run:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_data.py --config configs/cpu.yaml
.\.venv\Scripts\python.exe -m pytest tests/test_data_pipeline.py -v
```

## Acceptance Criteria

- The design doc exists separately from `plan.md`.
- Phase 2 exposes `prepare_data(config)` and `build_train_loader(config, split)`.
- Default implementation prepares local text into memmap token shards.
- Batches return fixed-shape shifted `LongTensor` inputs and targets.
- The data module stays swappable through explicit source/storage interfaces.
- All Phase 1 and Phase 2 tests pass.
