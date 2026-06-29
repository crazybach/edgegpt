"""Byte-level BPE tokenizer backend.

Byte-level BPE starts from bytes rather than Unicode words. Every UTF-8 string
can be represented as bytes, so the tokenizer never needs a true out-of-vocab
path for normal text. Training then learns frequent byte sequences ("merges")
such as common words, punctuation groups, or code fragments, making sequences
shorter while preserving exact decode round-trips.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from tokenizers import Regex, Tokenizer
from tokenizers import decoders, models, pre_tokenizers, trainers
from tokenizers.processors import TemplateProcessing

from configs.config import TokenizerConfig
from data.tokenizer.base import TokenizerBackend
from data.tokenizer.io import (
    SPECIAL_TOKENS_MAP_JSON,
    TOKENIZER_CONFIG_JSON,
    TOKENIZER_JSON,
    ensure_output_dir,
    tokenizer_json_path,
    write_json,
)
from data.tokenizer.special_tokens import assert_special_token_ids, special_token_ids, special_token_names


class ByteBPETokenizerBackend(TokenizerBackend):
    """HuggingFace `tokenizers` implementation of EdgeGPT byte-level BPE."""

    def __init__(self, config: TokenizerConfig):
        self.config = config
        self.tokenizer: Tokenizer | None = None

    @property
    def pad_token(self) -> str:
        return "<|pad|>"

    @property
    def unk_token(self) -> str:
        return "<|unk|>"

    @property
    def bos_token(self) -> str:
        return "<|bos|>"

    @property
    def eos_token(self) -> str:
        return "<|eos|>"

    @property
    def normal_vocab_size(self) -> int:
        """Number of non-special IDs before the frozen top special block."""

        return self.config.vocab_size - self.config.reserved_special_tokens

    def _require_tokenizer(self) -> Tokenizer:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer is not loaded. Call train(...) or load(...) first.")
        return self.tokenizer

    def _build_empty_tokenizer(self) -> Tokenizer:
        """Create an untrained byte-BPE tokenizer.

        The BPE model receives an `unk_token` for API completeness, but the full
        byte alphabet is seeded during training so ordinary UTF-8 text can be
        represented without using `<|unk|>`.
        """

        tokenizer = Tokenizer(models.BPE(unk_token=self.unk_token))

        pretok_steps = []
        if self.config.split_digits:
            # Splitting digits before BPE prevents the merge learner from making
            # "100" a single token. That keeps arithmetic-like strings
            # compositional for small models.
            pretok_steps.append(pre_tokenizers.Split(Regex(r"\d"), behavior="isolated"))
        pretok_steps.append(
            pre_tokenizers.ByteLevel(add_prefix_space=self.config.add_prefix_space, use_regex=True)
        )
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence(pretok_steps)
        tokenizer.decoder = decoders.ByteLevel()
        return tokenizer

    def train(self, files: list[Path], output_dir: Path) -> None:
        """Train normal BPE tokens, then append the frozen special block."""

        missing = [str(path) for path in files if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Tokenizer training files do not exist: {missing}")

        tokenizer = self._build_empty_tokenizer()

        trainer = trainers.BpeTrainer(
            vocab_size=self.normal_vocab_size,
            min_frequency=self.config.min_frequency,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=True,
        )
        tokenizer.train([str(path) for path in files], trainer=trainer)

        # Tiny smoke-test corpora may not produce enough learned merges to reach
        # `normal_vocab_size`. Add inert filler tokens before special tokens so
        # specials still land at the exact top IDs required by the checkpoint ABI.
        filler_index = 0
        while tokenizer.get_vocab_size(with_added_tokens=True) < self.normal_vocab_size:
            tokenizer.add_tokens([f"<|unused_token_{filler_index}|>"])
            filler_index += 1

        if tokenizer.get_vocab_size(with_added_tokens=True) != self.normal_vocab_size:
            raise ValueError(
                "Normal tokenizer vocab exceeded its budget before adding special tokens "
                f"({tokenizer.get_vocab_size(with_added_tokens=True)} != {self.normal_vocab_size})."
            )

        tokenizer.add_special_tokens(special_token_names(self.config.reserved_special_tokens))
        self.tokenizer = tokenizer
        self._configure_post_processor()
        self._validate_invariants()
        self.save(output_dir)

    def load(self, path: Path) -> None:
        """Load a tokenizer from an artifact directory or tokenizer JSON file."""

        self.tokenizer = Tokenizer.from_file(str(tokenizer_json_path(path)))
        self._configure_post_processor()
        self._validate_invariants()

    def save(self, output_dir: Path) -> None:
        """Save HF tokenizer JSON plus sidecars used by scripts and exporters."""

        tokenizer = self._require_tokenizer()
        output_dir = ensure_output_dir(output_dir)
        tokenizer.save(str(output_dir / TOKENIZER_JSON))

        special_ids = special_token_ids(self.config.vocab_size, self.config.reserved_special_tokens)
        write_json(
            output_dir / TOKENIZER_CONFIG_JSON,
            {
                "tokenizer_class": "EdgeGPTByteBPETokenizer",
                "model_max_length": None,
                "vocab_size": self.config.vocab_size,
                "reserved_special_tokens": self.config.reserved_special_tokens,
                "type": self.config.type,
                "add_prefix_space": self.config.add_prefix_space,
                "split_digits": self.config.split_digits,
                "pad_token": self.pad_token,
                "unk_token": self.unk_token,
                "bos_token": self.bos_token,
                "eos_token": self.eos_token,
            },
        )
        write_json(
            output_dir / SPECIAL_TOKENS_MAP_JSON,
            {
                "pad_token": self.pad_token,
                "unk_token": self.unk_token,
                "bos_token": self.bos_token,
                "eos_token": self.eos_token,
                "additional_special_tokens": [
                    token for token in special_ids if token not in {self.pad_token, self.unk_token, self.bos_token, self.eos_token}
                ],
                "special_token_ids": special_ids,
            },
        )
        self._write_readme(output_dir)

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> torch.LongTensor:
        """Encode one string as a 1-D LongTensor."""

        tokenizer = self._require_tokenizer()
        ids = list(tokenizer.encode(text, add_special_tokens=False).ids)
        if add_bos:
            ids.insert(0, self.token_to_id(self.bos_token))
        if add_eos:
            ids.append(self.token_to_id(self.eos_token))
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids: Sequence[int] | torch.Tensor, skip_special_tokens: bool = False) -> str:
        """Decode token IDs back to text."""

        tokenizer = self._require_tokenizer()
        if isinstance(ids, torch.Tensor):
            ids = ids.detach().cpu().tolist()
        return tokenizer.decode([int(token_id) for token_id in ids], skip_special_tokens=skip_special_tokens)

    def encode_batch(
        self,
        texts: list[str],
        padding: bool = True,
        max_length: int | None = None,
    ) -> dict[str, torch.LongTensor]:
        """Encode a batch and create a padding mask.

        Padding makes variable-length text fit into one tensor. The attention
        mask marks real tokens with 1 and pad tokens with 0 so training can
        ignore padding in loss/attention later.
        """

        encoded = [self.encode(text) for text in texts]
        if not encoded:
            return {
                "input_ids": torch.empty((0, 0), dtype=torch.long),
                "attention_mask": torch.empty((0, 0), dtype=torch.long),
            }

        lengths = [int(ids.numel()) for ids in encoded]
        target_len = max(lengths) if padding else lengths[0]
        if max_length is not None:
            target_len = min(target_len, max_length)

        rows: list[torch.LongTensor] = []
        masks: list[torch.LongTensor] = []
        pad_id = self.token_to_id(self.pad_token)
        for ids in encoded:
            ids = ids[:target_len]
            real_len = int(ids.numel())
            if padding:
                pad_len = target_len - real_len
                if pad_len > 0:
                    ids = torch.cat([ids, torch.full((pad_len,), pad_id, dtype=torch.long)])
                mask = torch.cat(
                    [torch.ones(real_len, dtype=torch.long), torch.zeros(target_len - real_len, dtype=torch.long)]
                )
            else:
                if real_len != target_len:
                    raise ValueError("encode_batch(padding=False) requires equal-length encoded rows.")
                mask = torch.ones(real_len, dtype=torch.long)
            rows.append(ids)
            masks.append(mask)

        return {"input_ids": torch.stack(rows), "attention_mask": torch.stack(masks)}

    def encode_texts(self, texts: list[str]) -> list[list[int]]:
        """Encode many documents without padding.

        Offline data preparation needs variable-length token lists, not padded
        tensors. HuggingFace tokenizers executes this batch path in native Rust,
        avoiding Python-call overhead for every single document line.
        """

        tokenizer = self._require_tokenizer()
        encodings = tokenizer.encode_batch(texts, add_special_tokens=False)
        return [[int(token_id) for token_id in encoding.ids] for encoding in encodings]

    def token_to_id(self, token: str) -> int:
        tokenizer = self._require_tokenizer()
        token_id = tokenizer.token_to_id(token)
        if token_id is None:
            raise KeyError(f"Unknown token: {token}")
        return int(token_id)

    def id_to_token(self, token_id: int) -> str:
        tokenizer = self._require_tokenizer()
        token = tokenizer.id_to_token(int(token_id))
        if token is None:
            raise KeyError(f"Unknown token ID: {token_id}")
        return token

    def vocab_size(self) -> int:
        return self._require_tokenizer().get_vocab_size(with_added_tokens=True)

    def _configure_post_processor(self) -> None:
        """Teach HF tokenizers how BOS/EOS would be inserted if requested.

        EdgeGPT's public `encode` inserts BOS/EOS manually so callers can choose
        either flag independently. The processor is still stored in tokenizer
        JSON for compatibility with tooling that expects special-token metadata.
        """

        tokenizer = self._require_tokenizer()
        tokenizer.post_processor = TemplateProcessing(
            single="$A",
            special_tokens=[
                (self.bos_token, self.token_to_id(self.bos_token)),
                (self.eos_token, self.token_to_id(self.eos_token)),
            ],
        )

    def _validate_invariants(self) -> None:
        tokenizer = self._require_tokenizer()
        actual_size = tokenizer.get_vocab_size(with_added_tokens=True)
        if actual_size != self.config.vocab_size:
            raise ValueError(f"Tokenizer vocab size must be {self.config.vocab_size}, got {actual_size}.")
        assert_special_token_ids(tokenizer.get_vocab(with_added_tokens=True), self.config.vocab_size, self.config.reserved_special_tokens)

    def _write_readme(self, output_dir: Path) -> None:
        special_ids = special_token_ids(self.config.vocab_size, self.config.reserved_special_tokens)
        rows = "\n".join(
            f"| `{token}` | {token_id} |" for token, token_id in special_ids.items() if not token.startswith("<|reserved_")
        )
        readme = f"""# EdgeGPT Byte-Level BPE Tokenizer

Algorithm: byte-level BPE via HuggingFace `tokenizers`.

Vocab size: `{self.config.vocab_size}`
Reserved special tokens: `{self.config.reserved_special_tokens}`
Normal BPE token budget: `{self.normal_vocab_size}`

Byte-level BPE learns frequent byte-sequence merges while preserving a path for
every UTF-8 input byte. The final 256 IDs are reserved for special tokens so
their embedding rows stay frozen across tokenizer retraining.

| Token | ID |
| --- | ---: |
{rows}
"""
        (output_dir / "README.md").write_text(readme, encoding="utf-8")
