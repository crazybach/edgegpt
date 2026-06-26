"""Frozen special-token definitions for EdgeGPT.

Special-token IDs are part of the checkpoint ABI. The model learns embedding
rows by integer ID, so changing these IDs later would silently make checkpoints
interpret tokens incorrectly. We place the special block at the top of the
vocab so the normal BPE merge table can be retrained without moving reserved
IDs, as long as the final vocab size stays fixed.
"""

from __future__ import annotations

from dataclasses import dataclass


RESERVED_SPECIAL_TOKENS = 256


@dataclass(frozen=True)
class SpecialToken:
    """A token whose ID is computed from its offset in the top reserved block."""

    name: str
    offset: int


CORE_SPECIAL_TOKENS: tuple[SpecialToken, ...] = (
    SpecialToken("<|pad|>", 0),
    SpecialToken("<|unk|>", 1),
    SpecialToken("<|bos|>", 2),
    SpecialToken("<|eos|>", 3),
    SpecialToken("<|sep|>", 4),
    SpecialToken("<|im_start|>", 16),
    SpecialToken("<|im_end|>", 17),
    SpecialToken("<|system|>", 18),
    SpecialToken("<|user|>", 19),
    SpecialToken("<|assistant|>", 20),
    SpecialToken("<|tool|>", 21),
    SpecialToken("<|tool_call|>", 22),
    SpecialToken("<|tool_result|>", 23),
    SpecialToken("<|img_start|>", 32),
    SpecialToken("<|img_end|>", 33),
    SpecialToken("<|img_pad|>", 34),
    SpecialToken("<|img_row_sep|>", 35),
)


def special_token_names(reserved_count: int = RESERVED_SPECIAL_TOKENS) -> list[str]:
    """Return all reserved special tokens in deterministic offset order.

    Unused slots are still named and reserved. Those extra rows cost little, and
    they prevent a future chat/tool/vision feature from forcing a vocab resize.
    """

    by_offset = {token.offset: token.name for token in CORE_SPECIAL_TOKENS}
    names: list[str] = []
    for offset in range(reserved_count):
        names.append(by_offset.get(offset, f"<|reserved_{offset}|>"))
    return names


def special_token_ids(vocab_size: int, reserved_count: int = RESERVED_SPECIAL_TOKENS) -> dict[str, int]:
    """Compute concrete IDs for the reserved block."""

    base_id = vocab_size - reserved_count
    return {name: base_id + offset for offset, name in enumerate(special_token_names(reserved_count))}


def assert_special_token_ids(
    token_to_id: dict[str, int],
    vocab_size: int,
    reserved_count: int = RESERVED_SPECIAL_TOKENS,
) -> None:
    """Fail fast if any frozen special-token ID moved."""

    expected = special_token_ids(vocab_size, reserved_count)
    mismatches = {
        token: (expected_id, token_to_id.get(token))
        for token, expected_id in expected.items()
        if token_to_id.get(token) != expected_id
    }
    if mismatches:
        preview = ", ".join(f"{tok}: expected {exp}, got {got}" for tok, (exp, got) in list(mismatches.items())[:5])
        raise ValueError(f"Special-token IDs are not stable: {preview}")
