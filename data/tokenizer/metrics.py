"""Small reporting helpers for tokenizer quality checks."""

from __future__ import annotations

from pathlib import Path


def iter_text_lines(files: list[Path], max_lines: int | None = None):
    """Yield lines from UTF-8 text files for metric sampling."""

    seen = 0
    for path in files:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                yield line.rstrip("\n")
                seen += 1
                if max_lines is not None and seen >= max_lines:
                    return


def compute_tokenizer_metrics(tokenizer, files: list[Path], max_lines: int | None = 10000) -> dict[str, float | int]:
    """Compute lightweight compression metrics.

    Bytes/token matters because training cost is paid per token. A smaller BPE
    vocab usually means more tokens per document; this report lets us quantify
    that tradeoff before committing to a long model run.
    """

    total_bytes = 0
    total_chars = 0
    total_tokens = 0
    max_sequence_tokens = 0
    sampled_lines = 0

    for text in iter_text_lines(files, max_lines=max_lines):
        ids = tokenizer.encode(text)
        token_count = int(ids.numel())
        total_bytes += len(text.encode("utf-8"))
        total_chars += len(text)
        total_tokens += token_count
        max_sequence_tokens = max(max_sequence_tokens, token_count)
        sampled_lines += 1

    return {
        "sampled_lines": sampled_lines,
        "total_bytes": total_bytes,
        "total_chars": total_chars,
        "total_tokens": total_tokens,
        "bytes_per_token": total_bytes / total_tokens if total_tokens else 0.0,
        "chars_per_token": total_chars / total_tokens if total_tokens else 0.0,
        "max_sequence_tokens": max_sequence_tokens,
    }
