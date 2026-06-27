"""Offline tokenization and shard preparation for Phase 2.

The default path tokenizes once and writes local shards. Training then reads
fixed-size integer blocks from disk, which keeps the future training loop from
being throttled by tokenizer CPU work.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from configs.config import Config
from data.pipeline.sources import build_document_source
from data.pipeline.shards import MemmapTokenShardWriter
from data.tokenizer import load_tokenizer


def resolve_block_size(config: Config) -> int:
    """Use the data override when present, otherwise match model context."""

    return config.data.block_size or config.model.max_seq_len


def _split_documents(document_count: int, val_split: float, seed: int) -> dict[int, str]:
    """Assign document indexes to train/val deterministically."""

    if document_count <= 0:
        raise ValueError("At least one document is required to prepare data.")
    indexes = list(range(document_count))
    rng = random.Random(seed)
    rng.shuffle(indexes)

    val_count = int(round(document_count * val_split))
    if val_split > 0 and document_count > 1:
        val_count = max(1, val_count)
    val_count = min(val_count, document_count - 1) if document_count > 1 else 0
    val_indexes = set(indexes[:val_count])
    return {index: ("val" if index in val_indexes else "train") for index in range(document_count)}


def prepare_data(config: Config) -> dict[str, object]:
    """Tokenize configured raw documents and write train/val token shards."""

    if config.data.storage_type != "memmap_bin":
        raise ValueError(f"Unsupported data.storage_type: {config.data.storage_type}")

    source = build_document_source(config.data)
    documents = list(source.documents())
    split_by_index = _split_documents(len(documents), config.data.val_split, config.data.seed)
    tokenizer = load_tokenizer(config)
    eos_id = tokenizer.token_to_id("<|eos|>")
    block_size = resolve_block_size(config)

    split_tokens: dict[str, list[int]] = {"train": [], "val": []}
    doc_counts: dict[str, int] = {"train": 0, "val": 0}
    for index, document in enumerate(documents):
        split = split_by_index[index]
        ids = tokenizer.encode(document.text).tolist()
        # EOS marks document boundaries in the flat stream. Without it, the
        # model would learn fake transitions from the end of one document into
        # the beginning of the next.
        split_tokens[split].extend(int(token_id) for token_id in ids)
        split_tokens[split].append(eos_id)
        doc_counts[split] += 1

    for split, tokens in split_tokens.items():
        if not tokens:
            continue
        if max(tokens) >= config.model.vocab_size or min(tokens) < 0:
            raise ValueError(f"{split} split contains token IDs outside the configured vocab.")

    # Tiny validation splits may not produce a full block; that is acceptable
    # for preparation, but train must be usable immediately by Phase 3+ tests.
    if len(split_tokens["train"]) < block_size + 1:
        raise ValueError(
            f"Train split has {len(split_tokens['train'])} tokens, but at least {block_size + 1} "
            "are required for one shifted training block."
        )

    writer = MemmapTokenShardWriter(config, eos_id=eos_id, block_size=block_size)
    metadata = writer.write(split_tokens)
    metadata["document_counts"] = doc_counts
    Path(config.data.cache_dir, "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
