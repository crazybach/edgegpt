"""Offline tokenization and shard preparation for Phase 2.

The default path tokenizes once and writes local shards. Training then reads
fixed-size integer blocks from disk, which keeps the future training loop from
being throttled by tokenizer CPU work.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

from configs.config import Config
from data.pipeline.sources import build_document_source
from data.pipeline.shards import StreamingTokenShardWriter
from data.tokenizer import load_tokenizer


PREPARE_ENCODE_BATCH_SIZE = 1024


def resolve_block_size(config: Config) -> int:
    """Use the data override when present, otherwise match model context."""

    return config.data.block_size or config.model.max_seq_len


def _split_for_document(doc_id: str, val_split: float, seed: int) -> str:
    """Assign one document to train/val without knowing corpus size.

    A stable hash gives deterministic splits while preserving streaming: we do
    not need to list all documents before deciding where each one goes.
    """

    if val_split <= 0:
        return "train"
    digest = hashlib.blake2b(f"{seed}:{doc_id}".encode("utf-8"), digest_size=8).digest()
    bucket = int.from_bytes(digest, byteorder="big") / float(1 << 64)
    return "val" if bucket < val_split else "train"


def prepare_data(config: Config) -> dict[str, object]:
    """Tokenize configured raw documents and write train/val token shards."""

    if config.data.storage_type != "memmap_bin":
        raise ValueError(f"Unsupported data.storage_type: {config.data.storage_type}")

    source = build_document_source(config.data)
    tokenizer = load_tokenizer(config)
    eos_id = tokenizer.token_to_id("<|eos|>")
    block_size = resolve_block_size(config)

    document_count = 0
    with StreamingTokenShardWriter(config, eos_id=eos_id, block_size=block_size) as writer:
        pending_splits: list[str] = []
        pending_texts: list[str] = []

        def flush_pending() -> None:
            if not pending_texts:
                return
            for split, ids in zip(pending_splits, tokenizer.encode_texts(pending_texts), strict=True):
                writer.append(split, ids)
            pending_splits.clear()
            pending_texts.clear()

        for document in source.documents():
            split = _split_for_document(document.doc_id, config.data.val_split, config.data.seed)
            pending_splits.append(split)
            pending_texts.append(document.text)
            document_count += 1
            if len(pending_texts) >= PREPARE_ENCODE_BATCH_SIZE:
                flush_pending()
        flush_pending()
        metadata = writer.metadata()

    if document_count <= 0:
        raise ValueError("At least one document is required to prepare data.")

    # Tiny validation splits may be empty; train must be immediately usable by
    # Phase 3+ tests and the future training loop.
    train_tokens = int(metadata["token_counts"]["train"])
    if train_tokens < block_size + 1:
        raise ValueError(
            f"Train split has {train_tokens} tokens, but at least {block_size + 1} "
            "are required for one shifted training block."
        )
    Path(config.data.cache_dir, "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
