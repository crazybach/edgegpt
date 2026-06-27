"""Print a bilingual tokenizer sample report.

This script is intentionally deterministic: it checks how the trained tokenizer
handles English and Simplified Chinese at several text lengths.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.config import load_config
from data.tokenizer import load_tokenizer


@dataclass(frozen=True)
class Sample:
    language: str
    category: str
    text: str


SAMPLE_GROUPS: dict[tuple[str, str], list[str]] = {
    ("English", "alphabet"): ["a", "b", "c", "x", "y", "z", "A", "B", "Y", "Z"],
    ("English", "word"): [
        "token",
        "model",
        "train",
        "vector",
        "prompt",
        "memory",
        "context",
        "gradient",
        "dataset",
        "language",
    ],
    ("English", "short phrase"): [
        "small model",
        "byte BPE",
        "training loop",
        "token IDs",
        "packed data",
        "CPU setup",
        "short context",
        "stable vocab",
        "edge device",
        "fast test",
    ],
    ("English", "short sentence"): [
        "The tokenizer keeps text reversible.",
        "EdgeGPT uses byte level BPE.",
        "A small batch fits better on CPU.",
        "Special token IDs must stay stable.",
        "The model learns from packed sequences.",
        "Tests catch silent tokenizer drift.",
        "Short prompts should decode cleanly.",
        "The config chooses the runtime device.",
        "CPU training needs smaller settings.",
        "Good reports make debugging easier.",
    ],
    ("English", "long sentence"): [
        "A byte-level BPE tokenizer starts from raw bytes, then learns frequent merges so common words become shorter token sequences.",
        "Keeping special token IDs frozen matters because every checkpoint learns embeddings at those exact integer positions.",
        "When a laptop has limited GPU memory, a smaller context length and batch size make experiments much easier to run.",
        "Tokenizer round trips are important because training data must decode back to the original text without hidden corruption.",
        "The sample report shows token counts for different text lengths, which helps estimate how much context a prompt will consume.",
        "English text often compresses into familiar subword pieces after BPE training, especially for common technical vocabulary.",
        "A CPU-only setup is slower than CUDA, but it is reliable for tokenizer checks, unit tests, and tiny overfit experiments.",
        "Before training a model from scratch, it is worth proving that the tokenizer, configuration, and data contracts agree.",
        "Short scripts like this are useful because they turn vague tokenizer quality concerns into concrete examples and counts.",
        "If a future tokenizer changes token IDs, the report should make the change visible before any checkpoint depends on it.",
    ],
    ("Chinese", "alphabet"): ["你", "好", "学", "习", "中", "文", "模", "型", "数", "据"],
    ("Chinese", "word"): [
        "模型",
        "训练",
        "数据",
        "分词",
        "语言",
        "向量",
        "上下文",
        "提示词",
        "检查点",
        "字节",
    ],
    ("Chinese", "short phrase"): [
        "小模型",
        "字节 BPE",
        "训练循环",
        "中文数据",
        "稳定词表",
        "CPU 设置",
        "短上下文",
        "特殊 token",
        "边缘设备",
        "快速测试",
    ],
    ("Chinese", "short sentence"): [
        "分词器需要保持可逆解码。",
        "EdgeGPT 使用字节级 BPE。",
        "小批量更适合 CPU 运行。",
        "特殊 token 的 ID 必须稳定。",
        "模型从打包后的序列中学习。",
        "测试可以发现分词器漂移。",
        "短提示词应该能干净解码。",
        "配置会选择运行设备。",
        "CPU 训练需要更小的设置。",
        "清晰的报告方便调试。",
    ],
    ("Chinese", "long sentence"): [
        "字节级 BPE 从 UTF-8 字节开始，然后学习常见片段，这样既能处理中文，也能保持可逆解码。",
        "特殊 token 的 ID 必须保持固定，因为模型检查点会把每个整数位置对应到已经学到的向量。",
        "当笔记本电脑的 GPU 显存有限时，降低上下文长度和批量大小会让实验更容易运行。",
        "分词器的往返测试很重要，因为训练数据应该能够还原成原始文本，而不是悄悄损坏。",
        "这个样例报告会展示不同长度文本的 token 数量，方便估计提示词会占用多少上下文。",
        "中文文本通常会被拆成较多字节或短片段，因此 token 数可能比表面字数看起来更多。",
        "纯 CPU 设置虽然比 CUDA 慢，但很适合检查分词器、运行单元测试以及做很小的过拟合实验。",
        "在从零训练模型之前，最好先证明分词器、配置和数据约定彼此一致。",
        "像这样的短脚本很有用，因为它能把模糊的分词质量问题变成具体的样例和计数。",
        "如果未来更换分词器导致 token ID 变化，这份报告应该能在检查点依赖它之前暴露差异。",
    ],
}


def samples() -> list[Sample]:
    return [
        Sample(language=language, category=category, text=text)
        for (language, category), texts in SAMPLE_GROUPS.items()
        for text in texts
    ]


def cell(text: object) -> str:
    """Escape Markdown table separators."""

    return str(text).replace("|", "\\|").replace("\n", "<br>")


def build_report(detail: bool) -> str:
    tokenizer = load_tokenizer(load_config())
    rows = []
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    failures: list[tuple[Sample, str]] = []

    for index, sample in enumerate(samples(), start=1):
        ids = tokenizer.encode(sample.text)
        id_list = ids.tolist()
        decoded = tokenizer.decode(ids)
        ok = decoded == sample.text
        grouped[(sample.language, sample.category)].append(len(id_list))
        if not ok:
            failures.append((sample, decoded))
        rows.append((index, sample, id_list, ok))

    lines = [
        "# Tokenizer Sample Report",
        "",
        f"- Samples: {len(rows)}",
        f"- Round-trip failures: {len(failures)}",
        "",
        "| Language | Category | Samples | Min tokens | Avg tokens | Max tokens |",
        "|---|---|---:|---:|---:|---:|",
    ]

    for (language, category), counts in grouped.items():
        lines.append(
            f"| {cell(language)} | {cell(category)} | {len(counts)} | "
            f"{min(counts)} | {sum(counts) / len(counts):.1f} | {max(counts)} |"
        )

    if failures:
        lines.extend(["", "## Round-trip Failures", ""])
        for sample, decoded in failures:
            lines.append(f"- {sample.language}/{sample.category}: {sample.text!r} -> {decoded!r}")

    if detail:
        lines.extend(
            [
                "",
                "## Details",
                "",
                "| # | Language | Category | Text | Token count | Round trip | Token IDs |",
                "|---:|---|---|---|---:|---|---|",
            ]
        )
        for index, sample, id_list, ok in rows:
            lines.append(
                f"| {index} | {cell(sample.language)} | {cell(sample.category)} | "
                f"{cell(sample.text)} | {len(id_list)} | {'ok' if ok else 'fail'} | {cell(id_list)} |"
            )

    return "\n".join(lines)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Print a bilingual tokenizer sample report.")
    parser.add_argument("--detail", action="store_true", help="Include every sample and token ID list.")
    parser.add_argument("--output", type=Path, default=None, help="Optional Markdown output path.")
    args = parser.parse_args()

    report = build_report(args.detail)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
