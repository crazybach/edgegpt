"""Export a trained EdgeGPT checkpoint to GGUF format for llama.cpp.

Usage::

    python scripts/export_gguf.py \
        --checkpoint artifacts/runs/tinystories/latest.pt \
        --tokenizer-dir artifacts/tokenizer/main_16k \
        --output edgegpt-f16.gguf

Architecture: ``"llama"`` — EdgeGPT follows the Llama tensor naming
convention (embed_tokens, layers, norm, lm_head), which maps directly
to the GGUF ``blk.N.*`` tensor names that llama.cpp expects.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gguf import GGUFWriter  # type: ignore[import-untyped]


# ── tensor name mapping ────────────────────────────────────────────────

# EdgeGPT state_dict key suffix → GGUF tensor name format string.
# The layer index is substituted via ``format(i=N)``.
_TENSOR_MAP: dict[str, str] = {
    "attention_norm.weight":     "blk.{i}.attn_norm.weight",
    "attention.q_proj.weight":   "blk.{i}.attn_q.weight",
    "attention.k_proj.weight":   "blk.{i}.attn_k.weight",
    "attention.v_proj.weight":   "blk.{i}.attn_v.weight",
    "attention.o_proj.weight":   "blk.{i}.attn_output.weight",
    "mlp_norm.weight":           "blk.{i}.ffn_norm.weight",
    "mlp.gate_proj.weight":      "blk.{i}.ffn_gate.weight",
    "mlp.up_proj.weight":        "blk.{i}.ffn_up.weight",
    "mlp.down_proj.weight":      "blk.{i}.ffn_down.weight",
}

# Names of norm-weight tensors that must stay in FP32 even when
# outtype="f16", following the GGUF "mostly f16" convention that
# keeps norm gains at full precision for numerical stability.
_NORM_TENSOR_NAMES: set[str] = {
    "token_embd.weight",           # not a norm, but embedding → sensitive
    "output_norm.weight",
    "blk.{i}.attn_norm.weight",
    "blk.{i}.ffn_norm.weight",
}


def _map_tensor_name(edgegpt_name: str) -> str | None:
    """Map one EdgeGPT state-dict key to a GGUF tensor name.

    Returns ``None`` for keys that should be skipped (e.g. RoPE
    ``inv_freq`` buffers, which are derived from metadata).
    """
    if not edgegpt_name.endswith(".weight"):
        return None

    # Top-level tensors.
    if edgegpt_name == "embed_tokens.embedding.weight":
        return "token_embd.weight"
    if edgegpt_name == "norm.weight":
        return "output_norm.weight"
    # Handle both tied (lm_head.weight) and untied (lm_head.proj.weight)
    # output-projection state-dict keys.
    if edgegpt_name in ("lm_head.weight", "lm_head.proj.weight"):
        return "output.weight"

    # Per-block tensors:  layers.{N}.<submodule>.<param>
    prefix = "layers."
    if not edgegpt_name.startswith(prefix):
        return None
    rest = edgegpt_name[len(prefix):]  # e.g. "0.attention.q_proj.weight"
    dot = rest.find(".")
    if dot == -1:
        return None
    try:
        layer_idx = int(rest[:dot])
    except ValueError:
        return None
    suffix = rest[dot + 1:]  # e.g. "attention.q_proj.weight"

    gguf_fmt = _TENSOR_MAP.get(suffix)
    if gguf_fmt is not None:
        return gguf_fmt.format(i=layer_idx)

    return None


def _is_norm_tensor(gguf_name: str, n_layers: int) -> bool:
    """Return True if *gguf_name* refers to a norm-gain weight."""
    if gguf_name == "output_norm.weight":
        return True
    for i in range(n_layers):
        if gguf_name in (
            f"blk.{i}.attn_norm.weight",
            f"blk.{i}.ffn_norm.weight",
        ):
            return True
    return False


# ── metadata writers ───────────────────────────────────────────────────


def _write_llama_metadata(writer: GGUFWriter, config: dict[str, Any]) -> None:
    """Write ``llama.*`` architecture metadata into the GGUF header."""

    model = config.get("model", config)

    def _m(key: str, default: Any = None) -> Any:
        return model.get(key, default)

    d_model = int(_m("d_model", 512))
    n_heads = int(_m("n_heads", 8))

    writer.add_architecture()  # sets general.architecture = "llama"
    writer.add_context_length(int(_m("max_seq_len", 2048)))
    writer.add_block_count(int(_m("n_layers", 8)))
    writer.add_embedding_length(d_model)
    writer.add_feed_forward_length(int(_m("d_ff", 1408)))
    writer.add_head_count(n_heads)
    writer.add_head_count_kv(int(_m("n_kv_heads", 4)))
    writer.add_rope_freq_base(float(_m("rope_theta", 10000.0)))
    writer.add_rope_dimension_count(d_model // n_heads)
    writer.add_layer_norm_rms_eps(float(_m("norm_eps", 1e-5)))
    writer.add_vocab_size(int(_m("vocab_size", 16384)))

    # Llama does not use bias — signalled structurally by absence of
    # .bias tensors; these metadata fields are informational only.
    writer.add_bool("llama.attention.use_bias", False)
    writer.add_bool("llama.feed_forward.use_bias", False)


def _write_tokenizer_metadata(
    writer: GGUFWriter,
    tokenizer_dir: str | Path,
) -> None:
    """Read EdgeGPT BPE tokenizer artifacts and serialize into GGUF metadata."""

    tk_dir = Path(tokenizer_dir)

    # 1. tokenizer.json — vocab + merges
    tok_path = tk_dir / "tokenizer.json"
    if not tok_path.exists():
        raise FileNotFoundError(f"Tokenizer artifact not found: {tok_path}")
    with open(tok_path, "r", encoding="utf-8") as f:
        tk_data = json.load(f)

    model_data = tk_data.get("model", {})
    vocab: dict[str, int] = dict(model_data.get("vocab", {}))
    merges_raw = model_data.get("merges", [])

    # Include added_tokens in the vocab — they sit above the BPE range
    # and are stored separately in tokenizer.json.
    for at in tk_data.get("added_tokens", []):
        tid = at.get("id", -1)
        token_str = at.get("content", "")
        if tid >= 0 and token_str:
            vocab[token_str] = tid

    # Normalise merges to GGUF format: space-joined string pairs.
    # EdgeGPT stores them as ``["token_a", "token_b"]`` lists.
    merges: list[str] = []
    for m in merges_raw:
        if isinstance(m, list):
            merges.append(" ".join(m))
        elif isinstance(m, str):
            merges.append(m)
        else:
            continue

    # Build ordered token list (index → token string).
    # vocab now includes both BPE tokens and added_tokens.
    vocab_size = max(vocab.values()) + 1 if vocab else 0
    id_to_token: list[str] = [""] * vocab_size
    for token, tid in vocab.items():
        if 0 <= tid < vocab_size:
            id_to_token[tid] = token

    # Scores: derive from merge rank so llama.cpp can prioritise merges.
    # Earlier merges (lower index) have higher priority → higher score.
    scores: list[float] = [0.0] * vocab_size
    num_merges = len(merges)
    for rank, merge in enumerate(merges):
        parts = merge.split(" ")
        score = float(num_merges - rank)
        for part in parts:
            tid = vocab.get(part)
            if tid is not None and 0 <= tid < vocab_size:
                # Use the highest (earliest) merge score for this token.
                if scores[tid] == 0.0 or score > scores[tid]:
                    scores[tid] = score

    token_types: list[int] = [1] * vocab_size  # 1 = normal

    # 2. special_tokens_map.json — special token IDs.
    # Handle both EdgeGPT's flat format and standard HuggingFace dict format.
    sp_path = tk_dir / "special_tokens_map.json"
    if not sp_path.exists():
        raise FileNotFoundError(f"Special-tokens artifact not found: {sp_path}")
    with open(sp_path, "r", encoding="utf-8") as f:
        special = json.load(f)

    special_ids: dict[str, int] = {}
    if "special_token_ids" in special:
        # EdgeGPT format: flat dict with token-string values.
        special_ids = special["special_token_ids"]
        bos_id = special_ids.get(special.get("bos_token", ""), -1)
        eos_id = special_ids.get(special.get("eos_token", ""), -1)
        unk_id = special_ids.get(special.get("unk_token", ""), -1)
        pad_id = special_ids.get(special.get("pad_token", ""), -1)
    else:
        # Standard HuggingFace format: each token is a dict {"content": ...}.
        def _resolve_special(key: str) -> int:
            entry = special.get(key)
            if isinstance(entry, dict):
                token_str = entry.get("content", "")
            elif isinstance(entry, str):
                token_str = entry
            else:
                return -1
            return vocab.get(token_str, -1)

        bos_id = _resolve_special("bos_token")
        eos_id = _resolve_special("eos_token")
        unk_id = _resolve_special("unk_token")
        pad_id = _resolve_special("pad_token")
        # Build special_ids from the HF-format entries.
        for key in ("bos_token", "eos_token", "unk_token", "pad_token"):
            entry = special.get(key)
            if isinstance(entry, dict):
                token_str = entry.get("content", "")
            elif isinstance(entry, str):
                token_str = entry
            else:
                continue
            tid = vocab.get(token_str)
            if tid is not None:
                special_ids[token_str] = tid
        # Also collect from additional_special_tokens list.
        for entry in special.get("additional_special_tokens", []):
            token_str = entry if isinstance(entry, str) else entry.get("content", "")
            tid = vocab.get(token_str)
            if tid is not None:
                special_ids[token_str] = tid

    # Mark only core special tokens as type 3 (control).
    # All tokens including reserved placeholders must stay type 1
    # (normal) so that tokenizer.ggml.tokens includes every vocab
    # entry.  If tokens are marked type 3, gguf-py filters them out
    # of the token list, which makes n_vocab != embedding_rows.
    _core_special = {
        special.get("bos_token", ""),
        special.get("eos_token", ""),
        special.get("unk_token", ""),
        special.get("pad_token", ""),
    }
    _core_special.discard("")

    for token_str, tid in special_ids.items():
        if 0 <= tid < vocab_size and token_str in _core_special:
            token_types[tid] = 3

    for at in tk_data.get("added_tokens", []):
        tid = at.get("id", -1)
        token_str = at.get("content", "")
        if at.get("special") and 0 <= tid < vocab_size and token_str in _core_special:
            token_types[tid] = 3

    # 3. Read add_bos_token from tokenizer_config.json when available.
    add_bos = False
    tk_cfg_path = tk_dir / "tokenizer_config.json"
    if tk_cfg_path.exists():
        with open(tk_cfg_path, "r", encoding="utf-8") as f:
            tk_cfg = json.load(f)
        add_bos = bool(tk_cfg.get("add_bos_token", False))

    # 4. Write tokenizer metadata via gguf-py's typed helpers.
    writer.add_tokenizer_model("gpt2")
    # Tokenizer pre-tokenizer is auto-detected by llama.cpp for BPE models
    # (based on vocab + merges).  Do not set an explicit value — the
    # GPT-2 BPE pre-tokenizer fingerprint is the correct default.
    writer.add_token_list(id_to_token)
    writer.add_token_scores(scores)
    writer.add_token_types(token_types)
    writer.add_token_merges(merges)

    if bos_id >= 0:
        writer.add_bos_token_id(bos_id)
    if eos_id >= 0:
        writer.add_eos_token_id(eos_id)
    if unk_id >= 0:
        writer.add_unk_token_id(unk_id)
    if pad_id >= 0:
        writer.add_pad_token_id(pad_id)

    writer.add_add_bos_token(add_bos)


# ── main export ────────────────────────────────────────────────────────


def export_gguf(
    checkpoint_path: str | Path,
    tokenizer_dir: str | Path,
    output_path: str | Path,
    *,
    outtype: str = "f16",
) -> Path:
    """Convert an EdgeGPT checkpoint to a single GGUF file.

    Args:
        checkpoint_path: Path to ``*.pt`` checkpoint (Phase 10 format).
        tokenizer_dir: Directory containing ``tokenizer.json``,
            ``special_tokens_map.json``.
        output_path: Destination ``.gguf`` file.
        outtype: ``"f32"`` or ``"f16"``.

    Returns:
        The output path.
    """
    ckpt_path = Path(checkpoint_path)
    out_path = Path(output_path)

    # 1. Load checkpoint.
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict: dict[str, torch.Tensor] = ckpt.get("model", ckpt)
    config: dict[str, Any] = ckpt.get("config", {})

    # 2. Determine block count for tensor mapping.
    n_layers = 0
    if isinstance(config, dict):
        model_cfg = config.get("model", config)
        if isinstance(model_cfg, dict):
            n_layers = model_cfg.get("n_layers", 0)
    if n_layers == 0:
        n_layers = sum(
            1 for k in state_dict
            if k.startswith("layers.") and k.endswith(".attention_norm.weight")
        )
    if n_layers == 0:
        raise ValueError("Could not determine n_layers from checkpoint.")

    # 3. Create GGUF writer.
    writer = GGUFWriter(str(out_path), "llama")

    # 4. Write metadata.
    _write_llama_metadata(writer, config)
    _write_tokenizer_metadata(writer, tokenizer_dir)

    # Set file type.
    ftype = 1 if outtype == "f16" else 0  # 1=mostly f16, 0=all f32
    writer.add_file_type(ftype)

    # 5. Map and write tensors.
    tie = True
    if isinstance(config, dict):
        model_cfg = config.get("model", config)
        if isinstance(model_cfg, dict):
            tie = model_cfg.get("tie_embeddings", True)

    mapped_count = 0
    try:
        for edgegpt_name, param in state_dict.items():
            gguf_name = _map_tensor_name(edgegpt_name)
            if gguf_name is None:
                continue
            if gguf_name == "output.weight" and tie:
                continue

            arr = param.detach().cpu().numpy()
            # FP16 conversion: skip norm-gain tensors so they stay at
            # full precision, matching the GGUF "mostly f16" convention.
            if (
                outtype == "f16"
                and arr.dtype == np.float32
                and not _is_norm_tensor(gguf_name, n_layers)
            ):
                arr = arr.astype(np.float16)

            writer.add_tensor(gguf_name, arr)
            mapped_count += 1

        if mapped_count == 0:
            raise RuntimeError("No tensors were mapped. Check state_dict naming.")

        # 6. Finalize.
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
    except Exception:
        writer.close()
        raise
    else:
        writer.close()

    print(f"Exported {mapped_count} tensors → {out_path} ({out_path.stat().st_size / (1024**2):.1f} MiB)")
    return out_path


# ── CLI ────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Export EdgeGPT checkpoint to GGUF.")
    parser.add_argument("--checkpoint", required=True, help="Path to EdgeGPT .pt checkpoint.")
    parser.add_argument("--tokenizer-dir", required=True, help="Directory with tokenizer.json + special_tokens_map.json.")
    parser.add_argument("--output", default="edgegpt-f16.gguf", help="Output GGUF path (default: edgegpt-f16.gguf).")
    parser.add_argument("--outtype", choices=("f16", "f32"), default="f16", help="Output dtype (default: f16).")
    args = parser.parse_args()

    export_gguf(
        checkpoint_path=args.checkpoint,
        tokenizer_dir=args.tokenizer_dir,
        output_path=args.output,
        outtype=args.outtype,
    )


if __name__ == "__main__":
    main()
