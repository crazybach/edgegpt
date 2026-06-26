# EdgeGPT — Tokenizer Specification (Phase 1)

> **Status:** SPEC — lock before training any model weights.  
> **Algorithm:** Byte-level BPE (trained with HuggingFace `tokenizers`).  
> **Deploy target:** llama.cpp (`llama-bpe` pre-tokenizer family).  
> **Golden rule:** Once IDs in this doc are assigned, they are FROZEN. Changing any reserved ID invalidates every checkpoint trained against it.

* * *

## 1. Design Decisions (locked)

| Decision | Value | Rationale |
| --- | --- | --- |
| Algorithm | Byte-level BPE | Llama 3 / GPT-4 / DeepSeek lineage; no OOV; best llama.cpp reproduction path |
| Training library | HF `tokenizers` (Rust) | tiktoken cannot train; SentencePiece SPM has llama.cpp leading-space divergence |
| Main vocab size | **32,768** (2^15) | Sweet spot for a phone-deployable small model; embedding matrix stays affordable |
| Smoke-test vocab size | **8,192** (2^13) | Faster laptop iteration on TinyStories pipeline validation |
| Vocab divisibility | Multiple of **128** | GPU matmul efficiency + avoids k-quant tensor-divisibility issues in llama.cpp |
| Number handling | **Split every digit** | Highest-leverage free win for math accuracy |
| Normalizer | **None** | Preserves reversible round-trip; matches Llama 3 |
| Byte fallback | **Enabled** (inherent) | Guarantees zero OOV across emoji / unicode / code |
| Pre-tokenizer | Llama-3-style regex + digit split | Guarantees llama.cpp recognizes it as `llama-bpe` |

**Note on vocab budget:** with `d_model` at small scale, the tied embedding matrix = `vocab_size × d_model`. At 32,768 vocab this is a meaningful but acceptable fraction of params. Do NOT use a 128K vocab (Llama 3 scale) — wrong tradeoff for a phone model.

* * *

## 2. Special Token Table (FROZEN IDs)

Byte-level BPE reserves IDs **0–255** for raw bytes. Special tokens therefore start at the TOP of the vocab (Llama 3 convention: high IDs added after main tokens), so adding/removing normal merges never shifts special-token IDs.

### 2.1 Layout strategy

*   **Bytes:** IDs `0 … 255` (fixed, byte-level base alphabet).
    
*   **Learned BPE merges:** IDs `256 … (V − R − 1)` where `V` = vocab size, `R` = size of reserved special block.
    
*   **Special / reserved block:** the **last** `R` **IDs** of the vocab. We fix `R = 256`.
    

For the **main model** (`V = 32768`), the special block occupies IDs **32512 … 32767**.  
For the **smoke-test model** (`V = 8192`), the special block occupies IDs **7936 … 8191**.

> The OFFSET within the special block is what we freeze. Define each special token as `V − 256 + offset`. This keeps the table identical across both vocab sizes (only the base `V` changes).

### 2.2 Core special tokens (offsets within the 256-slot reserved block)

| Offset | Name | Purpose | Used in pretraining? |
| --- | --- | --- | --- |
| 0 | `< | pad | >` |
| 1 | `< | unk | >` |
| 2 | `< | bos | >` |
| 3 | `< | eos | >` |
| 4 | `< | sep | >` |

### 2.3 Chat / instruction tokens (reserved now, used at SFT stage)

| Offset | Name | Purpose |
| --- | --- | --- |
| 16 | `< | im_start |
| 17 | `< | im_end |
| 18 | `< | system |
| 19 | `< | user |
| 20 | `< | assistant |
| 21 | `< | tool |
| 22 | `< | tool_call |
| 23 | `< | tool_result |

### 2.4 Vision / multimodal tokens (reserved now, used at ViT stage — future)

| Offset | Name | Purpose |
| --- | --- | --- |
| 32 | `< | img_start |
| 33 | `< | img_end |
| 34 | `< | img_pad |
| 35 | `< | img_row_sep |

### 2.5 Generic reserved slots (future-proofing)

| Offset range | Name pattern | Purpose |
| --- | --- | --- |
| 64 … 255 | `< | reserved_N |

> **Why reserve 256 total:** generous headroom means you will never need to resize the vocab (which would force retraining). 256 special slots cost only 256 rows in the embedding matrix.

### 2.6 Concrete ID examples

| Token | Offset | Main model (V=32768) ID | Smoke model (V=8192) ID |
| --- | --- | --- | --- |
| `< | pad | >` | 0 |
| `< | unk | >` | 1 |
| `< | bos | >` | 2 |
| `< | eos | >` | 3 |
| `< | im_start | >` | 16 |
| `< | im_end | >` | 17 |
| `< | img_start | >` | 32 |

Formula: `ID = V - 256 + offset`.

* * *

## 3. Pre-tokenizer Regex (Llama-3 style + digit split)

The pre-tokenizer splits raw text into chunks BEFORE BPE merging. Getting this exactly right is what makes llama.cpp fingerprint the tokenizer as `llama-bpe`.

### 3.1 Base pattern (Llama 3 / GPT-4o lineage)

```
(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
```

This is the standard Llama 3 split pattern. Key behaviors:

*   Contractions (`'s`, `'re`, etc.) split off cleanly.
    
*   Letters group into words with an optional leading non-letter/non-number.
    
*   `\p{N}{1,3}` groups numbers into runs of up to 3 digits (this is the GPT-4 behavior we are about to OVERRIDE — see below).
    
*   Punctuation runs group together with optional leading space.
    
*   Whitespace/newlines handled by the trailing alternatives.
    

### 3.2 Digit-split modification (for math)

Change the number clause from `\p{N}{1,3}` to single-digit grouping:

```
(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
```

Only change: `\p{N}{1,3}` → `\p{N}`. Now every digit is its own pre-token, so `"100"` → `["1","0","0"]` and BPE will not merge digit pairs (digit pairs must be excluded from merges — see §4.3).

> **llama.cpp caveat:** changing the number clause changes the pre-tokenizer fingerprint hash. llama.cpp may not auto-recognize it. Mitigation in §6.

### 3.3 ByteLevel settings

*   `add_prefix_space = False` (Llama 3 does not add a leading space).
    
*   `trim_offsets = True`.
    
*   `use_regex = True` with the pattern above.
    
*   Byte-level alphabet maps all 256 bytes to printable unicode placeholders (standard GPT-2 byte mapping).
    

* * *

## 4. Training Recipe

### 4.1 Corpus sample (representative mix)

Fit merges on a blended sample covering ALL target domains so no domain is under-tokenized:

*   General English web (FineWeb-Edu / SmolLM mix) — bulk
    
*   Code (Stack-Edu / Python subset) — ensure `{`, `}`, `==`, `def`, indentation get merges
    
*   Math (OpenWebMath) — ensure LaTeX symbols, operators get coverage
    
*   A small emoji + multi-codepoint sample — verify byte fallback path
    

Use a representative SAMPLE (a few GB of text is plenty), not the full corpus. Tokenizer training is not improved by brute-force data volume past a point.

### 4.2 Trainer settings (HF `tokenizers` BpeTrainer)

*   `vocab_size`: 32768 (main) / 8192 (smoke) — INCLUDING the 256 special slots and 256 byte tokens.
    
*   `min_frequency`: 2
    
*   `special_tokens`: pass the full table from §2 so IDs are assigned deterministically.
    
*   `initial_alphabet`: full 256-byte ByteLevel alphabet (so every byte is guaranteed present).
    
*   `show_progress`: true
    

### 4.3 Digit-merge exclusion

Even with single-digit pre-tokenization, ensure no merge rule combines two digits. Two layers of defense:

1.  Pre-tokenizer already isolates each digit (§3.2), so adjacent digits are in separate pre-tokens and cannot merge.
    
2.  As a belt-and-suspenders check, post-training, scan the merge list and assert no merged token is purely `\p{N}\p{N}`.
    

* * *

## 5. Shape Contracts & API

```
encode(text: str, add_bos: bool, add_eos: bool) -> LongTensor[seq_len]
decode(ids: LongTensor[seq_len]) -> str
encode_batch(texts: list[str], pad: bool) -> LongTensor[B, T]   # right-padded with <|pad|>
```

*   Single sequence: `str → LongTensor[seq_len]`
    
*   Batch: `list[str] → LongTensor[B, T]`, padded with `<|pad|>`, plus an attention mask `[B, T]`.
    
*   `<|pad|>` positions MUST be masked out of the loss and (later) attention.
    

* * *

## 6. llama.cpp Conversion & De-risking (DO THIS IN PHASE 1)

**Do not wait until Phase 12.** Validate the tokenizer-only conversion before training any weights.

1.  Export `tokenizer.json` (HF format) + minimal `config.json` / `tokenizer_config.json`.
    
2.  Run llama.cpp `convert_hf_to_gguf.py` on the tokenizer.
    
3.  **If you get "BPE pre-tokenizer was not recognized":** the digit-split modification changed the fingerprint hash. Fix by mapping your hash to the closest known pre-tokenizer:
    *   Compute your tokenizer's SHA-256 fingerprint (the converter prints it).
        
    *   Add an entry in `convert_hf_to_gguf_update.py` / `convert_hf_to_gguf.py` mapping that hash to `res = "llama-bpe"`.
        
    *   (Alternative: keep the unmodified Llama-3 number clause and lose single-digit math benefit — NOT recommended; prefer the hash mapping.)
        
4.  **Parity test:** `HF encode(text) == llama.cpp llama-tokenize(text)` for the full test battery in §7. This catches the known SPM leading-space-after-special-token divergence and any pre-tokenizer mismatch.
    

* * *

## 7. Unit Test Battery (must all pass before Phase 2)

| Test | Assertion |
| --- | --- |
| Round-trip ASCII | `decode(encode(x)) == x` |
| Round-trip CJK | `decode(encode("你好世界")) == "你好世界"` |
| Round-trip emoji (multi-codepoint) | `decode(encode("🙂‍↔️👨‍👩‍👧‍👦")) == input` |
| Round-trip code | `decode(encode(python_snippet)) == python_snippet` (indentation preserved) |
| Digit split | `encode("100")` tokens decode to `["1","0","0"]` (no merged digit tokens) |
| Long number | `encode("3.14159265358979")` — each digit its own token, `.` separate |
| Special token insertion | `encode(x, add_bos=True, add_eos=True)` begins with `< |
| Special tokens not splittable | `< |
| Batch padding | `encode_batch([...])` right-pads with `< |
| ID stability | Every special token resolves to `V - 256 + offset` for both V=8192 and V=32768 |
| llama.cpp parity | HF encode == `llama-tokenize` for all rows above |
| No-digit-merge invariant | No merge rule produces a pure two-digit token |

* * *

## 8. Serialization Artifacts

Save to `tokenizer/` in the repo:

*   `tokenizer.json` — HF format (vocab + merges + pre-tokenizer + special tokens)
    
*   `tokenizer_config.json` — special-token map, model_max_length
    
*   `special_tokens_map.json` — bos/eos/pad/unk references
    
*   `SPEC.md` — a copy of this document (the source of truth for frozen IDs)
    

Two builds: `tokenizer/smoke_8k/` and `tokenizer/main_32k/`.

* * *

## 9. Open Items / Future (Phase 13+)

*   **Decoupled large input vocab** (over-tokenization, NeurIPS/ICML scaling-law work): can match double-sized baselines at no extra cost — but llama.cpp won't support a decoupled-vocab architecture out of the box. Research-only for now.
    
*   **Byte Latent Transformer (tokenizer-free dynamic patching):** frontier; not llama.cpp-deployable today. Watch, don't build.
    
*   **SuperBPE / BoundlessBPE (superword merges across whitespace):** up to ~15% bytes-per-token improvement; reconsider at next tokenizer revision if a retrain is on the table anyway.