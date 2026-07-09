"""Generate text from an EdgeGPT checkpoint."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.config import load_config
from data.tokenizer import load_tokenizer
from eval import GenerationConfig, generate_ids
from model import EdgeGPT
from train.checkpoint import load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text from an EdgeGPT checkpoint.")
    parser.add_argument("--config", default="configs/tinystories_full_gpu_test.yaml", help="Path to model config YAML.")
    parser.add_argument(
        "--checkpoint",
        default="artifacts/runs/tinystories_full_gpu_test_1000/latest.pt",
        help="Path to a Phase 10 checkpoint.",
    )
    parser.add_argument("--prompt", default="Once upon a time", help="Prompt text.")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Maximum tokens to generate.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature; 0 forces greedy.")
    parser.add_argument("--top-k", type=int, default=None, help="Optional top-k sampling cutoff.")
    parser.add_argument("--top-p", type=float, default=None, help="Optional nucleus sampling cutoff.")
    parser.add_argument("--sample", action="store_true", help="Sample instead of greedy argmax.")
    parser.add_argument("--seed", type=int, default=None, help="Optional sampling seed.")
    parser.add_argument("--device", default=None, help="Override config device, e.g. cuda or cpu.")
    parser.add_argument("--no-cache", action="store_true", help="Use full-context recompute instead of KV cache.")
    parser.add_argument("--stats", action="store_true", help="Print timing stats to stderr.")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.device is not None:
        config.device = args.device
    device = torch.device(config.resolve_device())

    tokenizer = load_tokenizer(config)
    model = EdgeGPT(config).to(device)
    load_checkpoint(
        path=args.checkpoint,
        model=model,
        optimizer=None,
        scaler=None,
        map_location=device,
        restore_rng=False,
    )

    prompt_ids = tokenizer.encode(args.prompt, add_bos=False, add_eos=False).unsqueeze(0).to(device)
    gen_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        do_sample=args.sample,
        seed=args.seed,
        eos_token_id=tokenizer.token_to_id("<|eos|>"),
    )

    start = time.perf_counter()
    output_ids = generate_ids(model, prompt_ids, gen_config, use_cache=not args.no_cache)
    elapsed = max(time.perf_counter() - start, 1e-12)

    print(tokenizer.decode(output_ids[0].detach().cpu(), skip_special_tokens=True))
    if args.stats:
        new_tokens = max(int(output_ids.shape[1] - prompt_ids.shape[1]), 0)
        print(
            f"generated_tokens={new_tokens} elapsed_s={elapsed:.3f} tokens_per_sec={new_tokens / elapsed:.2f}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()