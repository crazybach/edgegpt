# EdgeGPT — Phase 10 Training Loop Technical Design

## Summary

Phase 10 adds a custom single-node PyTorch training subsystem around the Phase 9 `EdgeGPT` model. The decision is a nanoGPT-style loop: explicit AdamW, warmup/cosine learning rate, gradient accumulation, clipping, checkpointing, validation, and structured JSONL logging.

This is intentionally not Hugging Face Trainer, Lightning, DeepSpeed, or Megatron-LM. Those frameworks are mature, but they add API and distributed-training complexity before the dense laptop baseline has proven it can train.

## Direction Comparison

| Direction | Production / Research Status | Pros | Cons | Decision |
|---|---|---|---|---|
| Custom nanoGPT-style loop | Mature open-source baseline | Transparent, easy to debug, fits memmap shards | We own edge cases | Use |
| Hugging Face Trainer | Mature production library | Rich checkpoint/logging/eval features | Heavier model-output conventions | Defer |
| PyTorch Lightning | Mature framework | Less boilerplate | Adds another lifecycle layer | Defer |
| DeepSpeed / Megatron-LM | Production large-scale | ZeRO and model parallelism | Overkill for laptop Phase 10 | Defer |
| AdamW + warmup cosine | Mature LLM default | Stable and well understood | Not the newest schedule | Use |
| WSD / linear decay-to-zero | Active research | Can improve compute efficiency | Needs tuning after baseline | Defer |
| Muon and newer optimizers | Active research | Promising speed/efficiency | More risk and less export relevance | Defer |

References:
- nanoGPT training loop: https://github.com/karpathy/nanoGPT/blob/master/train.py
- PyTorch AdamW: https://docs.pytorch.org/docs/2.9/generated/torch.optim.AdamW.html
- PyTorch gradient clipping: https://docs.pytorch.org/docs/2.9/generated/torch.nn.utils.clip_grad_norm_.html
- PyTorch AMP: https://docs.pytorch.org/docs/2.9/amp.html
- Hugging Face Trainer: https://huggingface.co/docs/transformers/main_classes/trainer
- OLMo open training: https://arxiv.org/abs/2402.00838
- Linear decay-to-zero: https://arxiv.org/abs/2502.15938

## Architecture

The trainer lives outside `model/` so the Phase 9 model remains a pure neural network module. `train.trainer.Trainer` owns runtime state: model, optimizer, AMP scaler, data loaders, global step, best validation loss, and logger.

One optimizer step is:
1. Emit `step_start`.
2. Load `gradient_accumulation_steps` micro-batches.
3. Run forward under CUDA autocast when configured.
4. Divide loss by accumulation count and backpropagate.
5. Emit `micro_step` for each micro-batch.
6. Unscale gradients for fp16, clip global gradient norm, step AdamW, update scaler, and zero gradients.
7. Emit `optimizer_step` at `log_every` cadence.
8. Run eval/checkpoint hooks at configured intervals.

AdamW parameter grouping decays only matrix-like trainable weights. Norm gains, embeddings, tied output weights, biases, and one-dimensional parameters are excluded so scale-sensitive transformer parameters are not shrunk by weight decay.

## Live Reporting Contract

The first live-reporting backend is `events.jsonl` inside each run directory. Every line is one JSON object with stable fields:

`event`, `time`, `run_id`, `global_step`, `micro_step`, `max_steps`, `progress`, `split`, `loss`, `perplexity`, `lr`, `grad_norm`, `tokens_seen`, `tokens_per_sec`, `batch_size`, `seq_len`, `device`, `dtype`, `memory`, `checkpoint_path`.

Supported event types are `run_start`, `step_start`, `micro_step`, `optimizer_step`, `eval`, `checkpoint`, `run_end`, and `error`.

The future web dashboard should read or tail this file first. A later server can stream the same records over HTTP/WebSocket without changing trainer internals.

## Checkpointing And Resume

Checkpoints save:
- model state
- optimizer state
- AMP scaler state
- global step
- best validation loss
- config snapshot
- Python, NumPy, PyTorch, and CUDA RNG state when available

Each save writes `step_{global_step}.pt` and `latest.pt`. Numbered checkpoints are pruned by `training.checkpoint_keep_last`; `latest.pt` remains the stable resume path.

## Mixed Precision

CPU and MPS start in fp32. CUDA bf16 uses autocast without GradScaler. CUDA fp16 uses autocast plus GradScaler. This keeps the first implementation conservative while leaving the config path ready for GPU training.

## Acceptance Criteria

- `scripts/train.py --config configs/cpu.yaml --max-steps 2` runs after data preparation.
- Training resumes from `latest.pt`.
- `events.jsonl` reports progress, micro-step index, loss, perplexity, LR, grad norm, throughput, memory, eval, and checkpoint paths.
- Existing model and data tests continue to pass.
- No distributed training, web dashboard, generation loop, or HF export is introduced in Phase 10.
