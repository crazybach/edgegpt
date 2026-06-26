# EdgeGPT Byte-Level BPE Tokenizer

Algorithm: byte-level BPE via HuggingFace `tokenizers`.

Vocab size: `16384`
Reserved special tokens: `256`
Normal BPE token budget: `16128`

Byte-level BPE learns frequent byte-sequence merges while preserving a path for
every UTF-8 input byte. The final 256 IDs are reserved for special tokens so
their embedding rows stay frozen across tokenizer retraining.

| Token | ID |
| --- | ---: |
| `<|pad|>` | 16128 |
| `<|unk|>` | 16129 |
| `<|bos|>` | 16130 |
| `<|eos|>` | 16131 |
| `<|sep|>` | 16132 |
| `<|im_start|>` | 16144 |
| `<|im_end|>` | 16145 |
| `<|system|>` | 16146 |
| `<|user|>` | 16147 |
| `<|assistant|>` | 16148 |
| `<|tool|>` | 16149 |
| `<|tool_call|>` | 16150 |
| `<|tool_result|>` | 16151 |
| `<|img_start|>` | 16160 |
| `<|img_end|>` | 16161 |
| `<|img_pad|>` | 16162 |
| `<|img_row_sep|>` | 16163 |
