"""Print a bilingual tokenizer sample report.

This script is intentionally small and deterministic: it is a quick manual
inspection tool for checking how the trained tokenizer handles English and
Simplified Chinese at different text lengths.
"""

from __future__ import annotations

from configs.config import load_config
from data.tokenizer import load_tokenizer


SAMPLES: list[tuple[str, str, str]] = [
    ("English", "single alphabet", "a"),
    ("English", "single alphabet", "Z"),
    ("English", "word", "tokenizer"),
    ("English", "word", "language"),
    ("English", "short phrase", "byte level BPE"),
    ("English", "short phrase", "small model"),
    ("English", "sentence", "The tokenizer turns text into token IDs."),
    ("English", "sentence", "EdgeGPT will train on packed token sequences."),
    (
        "English",
        "long sentence",
        "A byte-level BPE tokenizer starts from raw bytes, then learns frequent merges so common words become shorter token sequences.",
    ),
    (
        "English",
        "long sentence",
        "Keeping special token IDs frozen matters because every checkpoint learns embeddings at those exact integer positions.",
    ),
    ("Simplified Chinese", "single character", "你"),
    ("Simplified Chinese", "single character", "学"),
    ("Simplified Chinese", "word", "分词器"),
    ("Simplified Chinese", "word", "语言"),
    ("Simplified Chinese", "short phrase", "字节级 BPE"),
    ("Simplified Chinese", "short phrase", "小型模型"),
    ("Simplified Chinese", "sentence", "分词器把文本转换成整数 ID。"),
    ("Simplified Chinese", "sentence", "EdgeGPT 将使用打包后的 token 序列进行训练。"),
    (
        "Simplified Chinese",
        "long sentence",
        "字节级 BPE 从 UTF-8 字节开始，然后学习常见片段，这样既能处理中文，也能保持可逆解码。",
    ),
    (
        "Simplified Chinese",
        "long sentence",
        "特殊 token 的 ID 必须保持固定，因为模型检查点会把每个整数位置对应到已学到的向量。",
    ),
]


def cell(text: object) -> str:
    """Escape Markdown table separators."""

    return str(text).replace("|", "\\|")


def main() -> None:
    tokenizer = load_tokenizer(load_config())
    print("| # | Lang | Coverage | Text | Count | Token IDs | Token pieces |")
    print("|---:|---|---|---|---:|---|---|")
    for index, (lang, coverage, text) in enumerate(SAMPLES, start=1):
        ids = tokenizer.encode(text)
        id_list = ids.tolist()
        pieces = [repr(tokenizer.id_to_token(int(token_id))) for token_id in id_list]
        print(
            f"| {index} | {cell(lang)} | {cell(coverage)} | {cell(text)} | "
            f"{len(id_list)} | {cell(id_list)} | {cell(', '.join(pieces))} |"
        )


if __name__ == "__main__":
    main()
