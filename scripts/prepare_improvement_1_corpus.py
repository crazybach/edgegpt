"""Build the bounded mixed corpus for EdgeGPT Improvement 1.0.

The builder streams remote datasets and stops at explicit token budgets measured
with the existing TinyStories tokenizer. It writes JSONL for data preparation,
a representative plain-text sample for tokenizer training, and a manifest.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.config import load_config  # noqa: E402
from data.tokenizer import load_tokenizer  # noqa: E402

ENCODE_BATCH_SIZE = 128
REPORT_EVERY_TOKENS = 10_000_000


@dataclass(frozen=True)
class CorpusSource:
    name: str
    kind: str
    text_column: str = "text"
    local_path: str | None = None
    dataset_path: str | None = None
    dataset_name: str | None = None
    split: str = "train"
    revision: str = "main"


SOURCES = {
    "tinystories": CorpusSource(
        name="tinystories",
        kind="local_text",
        local_path="data/tinystories/train.txt",
    ),
    "wikitext": CorpusSource(
        name="wikitext",
        kind="huggingface",
        dataset_path="Salesforce/wikitext",
        dataset_name="wikitext-103-raw-v1",
        revision="b08601e04326c79dfdd32d625aee71d232d685c3",
    ),
    "fineweb": CorpusSource(
        name="fineweb",
        kind="huggingface",
        dataset_path="HuggingFaceTB/smollm-corpus",
        dataset_name="fineweb-edu-dedup",
        revision="3ba9d605774198c5868892d7a8deda78031a781f",
    ),
    "cosmopedia": CorpusSource(
        name="cosmopedia",
        kind="huggingface",
        dataset_path="HuggingFaceTB/smollm-corpus",
        dataset_name="cosmopedia-v2",
        revision="3ba9d605774198c5868892d7a8deda78031a781f",
    ),
}

# Pilot validates the complete pipeline before the much longer 1B-token build.
PROFILES = {
    "pilot": {
        "tinystories": 20_000_000,
        "wikitext": 20_000_000,
        "fineweb": 50_000_000,
        "cosmopedia": 10_000_000,
    },
    "full": {
        "tinystories": 150_000_000,
        "wikitext": 100_000_000,
        "fineweb": 550_000_000,
        "cosmopedia": 200_000_000,
    },
}


def iter_local_documents(path: Path) -> Iterator[str]:
    if not path.exists():
        raise FileNotFoundError(f"Local corpus does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.rstrip("\r\n")
            if text.strip():
                yield text


def iter_huggingface_documents(
    source: CorpusSource,
    *,
    seed: int,
    shuffle_buffer_size: int,
) -> Iterator[str]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Hugging Face streaming requires the 'datasets' package.") from exc

    dataset = load_dataset(
        source.dataset_path,
        source.dataset_name,
        split=source.split,
        revision=source.revision,
        streaming=True,
    )
    dataset = dataset.shuffle(seed=seed, buffer_size=shuffle_buffer_size)
    for row in dataset:
        text = row.get(source.text_column)
        if isinstance(text, str) and text.strip():
            yield text


def iter_documents(
    source: CorpusSource,
    *,
    seed: int,
    shuffle_buffer_size: int,
) -> Iterable[str]:
    if source.kind == "local_text":
        assert source.local_path is not None
        return iter_local_documents(ROOT / source.local_path)
    if source.kind == "huggingface":
        return iter_huggingface_documents(
            source,
            seed=seed,
            shuffle_buffer_size=shuffle_buffer_size,
        )
    raise ValueError(f"Unsupported corpus source kind: {source.kind}")


def normalized_tokenizer_line(text: str) -> str:
    return " ".join(text.split())


def build_corpus(
    *,
    profile: str,
    output_dir: Path,
    tokenizer_config_path: Path,
    selected_sources: list[str] | None = None,
    tokenizer_sample_tokens: int = 20_000_000,
    seed: int = 42,
    shuffle_buffer_size: int = 10_000,
    force: bool = False,
) -> dict[str, object]:
    if profile not in PROFILES:
        raise ValueError(f"Unknown profile: {profile}")
    source_names = selected_sources or list(PROFILES[profile])
    if len(source_names) != len(set(source_names)):
        raise ValueError("selected_sources cannot contain duplicates.")
    unknown = sorted(set(source_names) - set(SOURCES))
    if unknown:
        raise ValueError(f"Unknown sources: {unknown}")
    if tokenizer_sample_tokens <= 0:
        raise ValueError("tokenizer_sample_tokens must be positive.")

    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = output_dir / "train.jsonl"
    tokenizer_path = output_dir / "tokenizer_train.txt"
    manifest_path = output_dir / "manifest.json"
    existing = [path for path in (corpus_path, tokenizer_path, manifest_path) if path.exists()]
    if existing and not force:
        raise FileExistsError(f"Output already exists: {existing}. Pass --force to replace it.")

    config = load_config(tokenizer_config_path)
    tokenizer = load_tokenizer(config)
    partial_corpus = corpus_path.with_suffix(corpus_path.suffix + ".partial")
    partial_tokenizer = tokenizer_path.with_suffix(tokenizer_path.suffix + ".partial")
    stats: dict[str, dict[str, int]] = {}
    tokenizer_tokens = 0
    selected_total_tokens = sum(PROFILES[profile][name] for name in source_names)

    with partial_corpus.open("w", encoding="utf-8", newline="\n") as corpus_handle, partial_tokenizer.open(
        "w", encoding="utf-8", newline="\n"
    ) as tokenizer_handle:
        for source_name in source_names:
            source = SOURCES[source_name]
            target_tokens = PROFILES[profile][source_name]
            source_tokens = 0
            source_documents = 0
            source_tokenizer_tokens = 0
            source_tokenizer_target = max(
                1,
                round(tokenizer_sample_tokens * target_tokens / selected_total_tokens),
            )
            next_report = REPORT_EVERY_TOKENS
            pending: list[str] = []

            def flush() -> None:
                nonlocal source_tokens, source_documents
                nonlocal source_tokenizer_tokens, tokenizer_tokens, next_report
                if not pending:
                    return
                encoded = tokenizer.encode_texts(pending)
                for text, token_ids in zip(pending, encoded, strict=True):
                    if source_tokens >= target_tokens:
                        break
                    corpus_handle.write(json.dumps({"source": source_name, "text": text}, ensure_ascii=False) + "\n")
                    token_count = len(token_ids) + 1  # include EOS inserted by prepare_data
                    source_tokens += token_count
                    source_documents += 1
                    if source_tokenizer_tokens < source_tokenizer_target:
                        line = normalized_tokenizer_line(text)
                        if line:
                            tokenizer_handle.write(line + "\n")
                            tokenizer_tokens += token_count
                            source_tokenizer_tokens += token_count
                    if source_tokens >= next_report:
                        print(
                            f"source={source_name} tokens={source_tokens:,}/{target_tokens:,} "
                            f"documents={source_documents:,}",
                            flush=True,
                        )
                        next_report += REPORT_EVERY_TOKENS
                pending.clear()

            for text in iter_documents(source, seed=seed, shuffle_buffer_size=shuffle_buffer_size):
                pending.append(text)
                if len(pending) >= ENCODE_BATCH_SIZE:
                    flush()
                if source_tokens >= target_tokens:
                    break
            flush()
            if source_tokens < target_tokens:
                raise RuntimeError(
                    f"Source {source_name} ended at {source_tokens:,} tokens before target {target_tokens:,}."
                )
            stats[source_name] = {
                "target_tokens": target_tokens,
                "actual_tokens": source_tokens,
                "documents": source_documents,
                "tokenizer_sample_tokens": source_tokenizer_tokens,
            }
            print(f"completed source={source_name} tokens={source_tokens:,} documents={source_documents:,}")

    partial_corpus.replace(corpus_path)
    partial_tokenizer.replace(tokenizer_path)
    manifest: dict[str, object] = {
        "profile": profile,
        "seed": seed,
        "counting_tokenizer": str(config.tokenizer.artifact_dir),
        "tokenizer_sample_tokens": tokenizer_tokens,
        "sources": {name: asdict(SOURCES[name]) for name in source_names},
        "stats": stats,
        "total_tokens": sum(item["actual_tokens"] for item in stats.values()),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the EdgeGPT Improvement 1.0 mixed corpus.")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="pilot")
    parser.add_argument("--output-dir", type=Path, default=Path("data/improvement_1"))
    parser.add_argument(
        "--tokenizer-config",
        type=Path,
        default=Path("configs/tinystories_full_gpu_test.yaml"),
        help="Existing tokenizer used only to enforce source token budgets.",
    )
    parser.add_argument("--source", action="append", choices=sorted(SOURCES), dest="sources")
    parser.add_argument("--tokenizer-sample-tokens", type=int, default=20_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle-buffer-size", type=int, default=10_000)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    selected = args.sources or list(PROFILES[args.profile])
    if args.dry_run:
        plan = {name: PROFILES[args.profile][name] for name in selected}
        print(json.dumps({"profile": args.profile, "sources": plan, "total_tokens": sum(plan.values())}, indent=2))
        return

    manifest = build_corpus(
        profile=args.profile,
        output_dir=args.output_dir,
        tokenizer_config_path=args.tokenizer_config,
        selected_sources=args.sources,
        tokenizer_sample_tokens=args.tokenizer_sample_tokens,
        seed=args.seed,
        shuffle_buffer_size=args.shuffle_buffer_size,
        force=args.force,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()