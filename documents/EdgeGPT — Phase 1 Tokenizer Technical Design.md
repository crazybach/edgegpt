# EdgeGPT Phase 1 - Tokenizer Technical Design

## Summary

Phase 1 implements a modular tokenizer subsystem for EdgeGPT. The first backend is byte-level BPE with HuggingFace `tokenizers`, using a final vocabulary size of `16384`.

The tokenizer is a model contract, not a replaceable preprocessing detail. Token IDs index embedding rows, so `model.vocab_size`, tokenizer artifacts, and special-token IDs must stay aligned once training starts.

## Locked Decisions

- Algorithm: byte-level BPE.
- Library: HuggingFace `tokenizers`.
- Final vocab size: `16384`.
- Reserved special-token block: final `256` IDs.
- Normal BPE token budget: `16128`.
- Special-token formula: `special_id = vocab_size - reserved_special_tokens + offset`.
- With `vocab_size = 16384`, the special-token base ID is `16128`.
- The module exposes a backend interface so later tokenizer algorithms can be swapped without changing the data pipeline.

## Module Layout

- `data/tokenizer/base.py`: abstract tokenizer backend API.
- `data/tokenizer/byte_bpe.py`: byte-level BPE implementation.
- `data/tokenizer/special_tokens.py`: frozen special-token offsets and ID validation.
- `data/tokenizer/registry.py`: maps config value `byte_bpe` to the backend.
- `data/tokenizer/io.py`: tokenizer sidecar JSON helpers.
- `data/tokenizer/metrics.py`: compression and sampling metrics.
- `scripts/train_tokenizer.py`: trains and saves tokenizer artifacts.
- `scripts/inspect_tokenizer.py`: prints token IDs, token strings, and decoded text.

## Public API

```python
from data.tokenizer import load_tokenizer

tokenizer = load_tokenizer(config)
ids = tokenizer.encode("hello", add_bos=True, add_eos=True)
text = tokenizer.decode(ids)
batch = tokenizer.encode_batch(["a", "bb"], padding=True)
```

`encode_batch` returns:

```python
{
    "input_ids": LongTensor[B, T],
    "attention_mask": LongTensor[B, T],
}
```

Padding is right-padding with `<|pad|>`. The attention mask is `1` for real tokens and `0` for padding.

## Special Tokens

Special tokens are appended after normal BPE training so they occupy the top IDs.

| Token | Offset | ID |
| --- | ---: | ---: |
| `<|pad|>` | 0 | 16128 |
| `<|unk|>` | 1 | 16129 |
| `<|bos|>` | 2 | 16130 |
| `<|eos|>` | 3 | 16131 |
| `<|sep|>` | 4 | 16132 |
| `<|im_start|>` | 16 | 16144 |
| `<|im_end|>` | 17 | 16145 |
| `<|system|>` | 18 | 16146 |
| `<|user|>` | 19 | 16147 |
| `<|assistant|>` | 20 | 16148 |
| `<|tool|>` | 21 | 16149 |
| `<|tool_call|>` | 22 | 16150 |
| `<|tool_result|>` | 23 | 16151 |
| `<|img_start|>` | 32 | 16160 |
| `<|img_end|>` | 33 | 16161 |
| `<|img_pad|>` | 34 | 16162 |
| `<|img_row_sep|>` | 35 | 16163 |

The remaining slots are named `<|reserved_N|>` for future chat, tool, or multimodal features.

## Training Behavior

1. Build an untrained byte-level BPE tokenizer.
2. Seed the trainer with the full byte-level alphabet so arbitrary UTF-8 text can round-trip.
3. Train normal BPE tokens up to `16128`.
4. If the corpus is too small to fill that budget, add inert filler tokens before specials.
5. Append all 256 special tokens.
6. Assert the final vocab size is exactly `16384`.
7. Assert every special token has its expected top-ID value.
8. Save artifacts and metrics.

Artifacts:

```text
artifacts/tokenizer/main_16k/
  tokenizer.json
  tokenizer_config.json
  special_tokens_map.json
  README.md
  metrics.json
```

## Config Contract

`configs/config.py` defines `TokenizerConfig`. `Config.validate()` fails if `model.vocab_size != tokenizer.vocab_size`, because the model embedding table and tokenizer ID space must match.

Default config:

```yaml
tokenizer:
  type: "byte_bpe"
  vocab_size: 16384
  reserved_special_tokens: 256
  artifact_dir: "./artifacts/tokenizer/main_16k"
  train_files:
    - "./data/tinystories/train.txt"
  min_frequency: 2
  add_prefix_space: false
  split_digits: true
```

## Automated Tests

`tests/test_tokenizer.py` covers:

- Config validation and vocab mismatch failure.
- Special-token ID stability.
- ASCII, CJK, emoji, code, whitespace, and punctuation round-trips.
- BOS/EOS insertion.
- Batch padding and attention mask behavior.
- Save/load parity.
- Encoded ID bounds.
- Smoke training on a temporary tiny corpus.

Run:

```bash
pytest tests/test_tokenizer.py
```

## Acceptance Criteria

- The design doc exists separately from `plan.md`.
- Tokenizer module loads through `load_tokenizer(config)`.
- Byte-level BPE trains and saves artifacts.
- Reloaded tokenizer produces identical encodings.
- All tokenizer tests pass.
- Code comments explain the algorithm and invariants without restating obvious Python syntax.
