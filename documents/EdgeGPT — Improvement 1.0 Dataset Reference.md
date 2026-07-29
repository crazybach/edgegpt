# EdgeGPT Improvement 1.0 Dataset Reference

## Purpose

The generated Improvement 1.0 corpus is intentionally excluded from Git
because it is approximately 3.7 GB and can be reproduced from public sources.
This document records the exact source names, configurations, revisions,
mixture budgets, and commands needed to rebuild it.

## Sources

| Mixture name | Dataset and configuration | Split | Pinned revision | Full-run budget |
| --- | --- | --- | --- | ---: |
| `tinystories` | [roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories), converted locally to `data/tinystories/train.txt` | train | Local source | 150M tokens |
| `wikitext` | [Salesforce/wikitext](https://huggingface.co/datasets/Salesforce/wikitext), `wikitext-103-raw-v1` | train | `b08601e04326c79dfdd32d625aee71d232d685c3` | 100M tokens |
| `fineweb` | [HuggingFaceTB/smollm-corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus), `fineweb-edu-dedup` | train | `3ba9d605774198c5868892d7a8deda78031a781f` | 550M tokens |
| `cosmopedia` | [HuggingFaceTB/smollm-corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus), `cosmopedia-v2` | train | `3ba9d605774198c5868892d7a8deda78031a781f` | 200M tokens |

The authoritative source definitions and revisions live in
`scripts/prepare_improvement_1_corpus.py`. Update that file and this document
together when changing a dataset revision.

## Profiles

| Source | Pilot | Full |
| --- | ---: | ---: |
| TinyStories | 20M | 150M |
| WikiText-103 raw | 20M | 100M |
| FineWeb-Edu deduplicated | 50M | 550M |
| Cosmopedia v2 | 10M | 200M |
| **Total** | **100M** | **1B** |

Budgets are counted with `artifacts/tokenizer/main_16k` and include the EOS
token inserted per document. Counts with the newly trained tokenizer can differ
slightly.

## Generated Files

The builder writes these ignored files under `data/improvement_1/`:

- `train.jsonl`: mixed training documents with source labels.
- `tokenizer_train.txt`: proportional 20M-token tokenizer sample.
- `manifest.json`: achieved counts and source metadata.
- `*.partial`: incomplete files used for atomic generation.

Do not commit these files. Preserve a completed run's manifest with the run
artifacts when exact provenance is required.

## Rebuild

Prerequisites:

- Create `data/tinystories/train.txt` with one TinyStories document per
  non-empty line.
- Install the project dependencies, including Hugging Face `datasets`.
- Ensure `artifacts/tokenizer/main_16k` exists for bounded token counting.

Inspect the plan without downloading:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_improvement_1_corpus.py `
  --profile full `
  --dry-run
```

Build the full mixed corpus:

```powershell
$env:HF_ENDPOINT = "https://huggingface.co"
.\.venv\Scripts\python.exe scripts\prepare_improvement_1_corpus.py `
  --profile full
```

Regenerate existing outputs only when replacement is intentional:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_improvement_1_corpus.py `
  --profile full `
  --force
```

Then train the mixed tokenizer and pack 2K shards:

```powershell
.\.venv\Scripts\python.exe scripts\train_tokenizer.py `
  --config configs\improvement_1_20m_2k.yaml

.\.venv\Scripts\python.exe scripts\prepare_data.py `
  --config configs\improvement_1_20m_2k.yaml
```

## Reproducibility Notes

- Streaming order is deterministic only for a fixed dataset revision, seed,
  shuffle buffer, library behavior, and source availability.
- The builder uses seed `42` and a bounded shuffle buffer of `10,000`.
- Remote datasets are streamed and stop at the configured token budget; full
  upstream datasets are not downloaded.
- Set `HF_ENDPOINT` explicitly. A machine-level mirror caused failures during
  this run.
- Keep the generated `manifest.json` with experiment records if exact achieved
  token and document counts are needed later.
