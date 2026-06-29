"""Document sources for Phase 2.

The default source is intentionally humble: local text or JSONL files on disk.
That gives repeatable CPU-friendly tests now, while the `DocumentSource`
interface leaves room for future Hugging Face streaming or blended corpora.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from configs.config import DataConfig
from data.pipeline.base import Document, DocumentSource, resolve_source_paths


class LocalTextDocumentSource(DocumentSource):
    """Stream local UTF-8 text files as one document per non-empty line.

    TinyStories-style text files commonly store one story per line. Reading the
    whole file into one string made Phase 2 memory scale with the full corpus
    size, so the source now streams small document records. Later source
    backends can add paragraph-aware parsing without changing the pipeline API.
    """

    def __init__(self, paths: list[Path]):
        self.paths = paths

    def documents(self) -> Iterable[Document]:
        for path in self.paths:
            if not path.exists():
                raise FileNotFoundError(f"Data source does not exist: {path}")
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    text = line.rstrip("\r\n")
                    if not text.strip():
                        continue
                    yield Document(doc_id=f"{path}:{line_number}", text=text)


class JsonlDocumentSource(DocumentSource):
    """Read one document per JSONL row from a configured text column."""

    def __init__(self, paths: list[Path], text_column: str):
        self.paths = paths
        self.text_column = text_column

    def documents(self) -> Iterable[Document]:
        for path in self.paths:
            if not path.exists():
                raise FileNotFoundError(f"Data source does not exist: {path}")
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    text = row.get(self.text_column)
                    if not isinstance(text, str):
                        raise ValueError(f"{path}:{line_number} is missing text column {self.text_column!r}.")
                    yield Document(doc_id=f"{path}:{line_number}", text=text)


def build_document_source(config: DataConfig) -> DocumentSource:
    """Create the configured source backend."""

    paths = resolve_source_paths(config.data_dir, config.source_paths)
    if not paths:
        raise FileNotFoundError(
            "No data source files found. Set data.source_paths or place train.txt under data.data_dir."
        )
    if config.source_type == "local_text":
        return LocalTextDocumentSource(paths)
    if config.source_type == "jsonl":
        return JsonlDocumentSource(paths, config.text_column)
    raise ValueError(f"Unsupported data.source_type: {config.source_type}")

