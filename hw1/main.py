"""Train and sample a small TransformerLM on TinyStories."""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from modules.loss import cross_entropy
from modules.module import TransformerLM, softmax
from modules.optimizer import AdamW
from modules.scheduler import get_cosine_scheduler
from modules.tokenizer import Tokenizer
from modules.utils import clip_grad_norm_, get_batch


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = Path("/essfs10/daicheng/datasets/TinyStories")


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    context_length: int
    d_model: int
    num_layers: int
    num_heads: int
    d_ff: int
    theta: float = 10_000.0
    eps: float = 1e-5


def _device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def save_checkpoint(
    checkpoint_path: str | Path,
    model: TransformerLM,
    optimizer: AdamW,
    *,
    step: int,
    model_config: dict,
) -> None:
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model_config": model_config,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        checkpoint_path,
    )


def load_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: torch.device | str,
) -> tuple[TransformerLM, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = TransformerLM(**checkpoint["model_config"], device=device)
    model.load_state_dict(checkpoint["model_state"])
    return model, checkpoint


@torch.no_grad()
def estimate_loss(
    model: TransformerLM,
    tokens: np.ndarray,
    *,
    batch_size: int,
    context_length: int,
    device: torch.device | str,
    num_batches: int,
) -> float:
    was_training = model.training
    model.eval()
    losses: list[float] = []
    for _ in range(num_batches):
        x, y = get_batch(tokens, batch_size, context_length, device)
        losses.append(cross_entropy(model(x), y).item())
    model.train(was_training)
    return sum(losses) / len(losses)


def split_train_validation(
    tokens: np.ndarray,
    validation_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Reserve the final contiguous fraction of tokens for validation."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")

    split_index = int(len(tokens) * (1 - validation_fraction))
    if split_index <= 0 or split_index >= len(tokens):
        raise ValueError("validation split leaves an empty partition")
    return tokens[:split_index], tokens[split_index:]


@torch.no_grad()
def generate_text(
    model: TransformerLM,
    tokenizer: Tokenizer,
    prompt: str,
    *,
    max_new_tokens: int,
    device: torch.device | str,
    temperature: float = 0.8,
    top_k: int | None = 50,
) -> str:
    if not prompt:
        raise ValueError("prompt must not be empty")
    if max_new_tokens < 0 or temperature <= 0:
        raise ValueError("max_new_tokens must be non-negative and temperature positive")

    token_ids = tokenizer.encode(prompt)
    if not token_ids:
        raise ValueError("prompt produced no tokens")

    was_training = model.training
    model.eval()
    for _ in range(max_new_tokens):
        context = token_ids[-model.context_length :]
        x = torch.tensor(context, dtype=torch.long, device=device).unsqueeze(0)
        next_logits = model(x)[0, -1] / temperature

        if top_k is not None:
            if top_k <= 0:
                raise ValueError("top_k must be positive or None")
            k = min(top_k, next_logits.numel())
            threshold = torch.topk(next_logits, k).values[-1]
            next_logits = next_logits.masked_fill(next_logits < threshold, float("-inf"))

        next_token = torch.multinomial(softmax(next_logits, dim=-1), 1).item()
        token_ids.append(next_token)

    model.train(was_training)
    return tokenizer.decode(token_ids)


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = _device(args.device)
    tokens = np.load(args.data, mmap_mode="r")
    train_tokens, validation_tokens = split_train_validation(
        tokens, args.validation_fraction
    )

    if args.resume is not None:
        model, checkpoint = load_checkpoint(args.resume, device=device)
        model_config = ModelConfig(**checkpoint["model_config"])
        optimizer = AdamW(
            model.parameters(),
            lr=args.max_lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
        )
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_step = int(checkpoint["step"]) + 1
    else:
        model_config = ModelConfig(
            vocab_size=args.vocab_size,
            context_length=args.context_length,
            d_model=args.d_model,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            d_ff=args.d_ff,
        )
        model = TransformerLM(**asdict(model_config), device=device)
        optimizer = AdamW(
            model.parameters(),
            lr=args.max_lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
        )
        start_step = 0

    cosine_steps = args.cosine_steps or args.steps
    run_start = time.perf_counter()
    for step in range(start_step, args.steps):
        model.train()
        optimizer.lr = get_cosine_scheduler(
            step,
            args.max_lr,
            args.min_lr,
            args.warmup_steps,
            cosine_steps,
        )

        x, y = get_batch(train_tokens, args.batch_size, model.context_length, device)
        logits = model(x)
        loss = cross_entropy(logits, y)

        optimizer.zero_grad()
        loss.backward()
        grad_norm = clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if step % args.log_every == 0:
            elapsed = time.perf_counter() - run_start
            tokens_seen = (step - start_step + 1) * args.batch_size * model.context_length
            print(
                f"step={step:>6} loss={loss.item():.4f} lr={optimizer.lr:.2e} "
                f"grad_norm={grad_norm.item():.2f} tokens/s={tokens_seen / elapsed:,.0f}",
                flush=True,
            )

        if step > 0 and step % args.eval_every == 0:
            validation_loss = estimate_loss(
                model,
                validation_tokens,
                batch_size=args.batch_size,
                context_length=model.context_length,
                device=device,
                num_batches=args.eval_batches,
            )
            print(
                f"step={step:>6} validation_loss={validation_loss:.4f} "
                f"perplexity={math.exp(validation_loss):.2f}",
                flush=True,
            )

        if step > 0 and step % args.save_every == 0:
            save_checkpoint(
                args.checkpoint,
                model,
                optimizer,
                step=step,
                model_config=asdict(model_config),
            )

    save_checkpoint(
        args.checkpoint,
        model,
        optimizer,
        step=args.steps - 1,
        model_config=asdict(model_config),
    )
    print(f"saved checkpoint to {args.checkpoint}", flush=True)


def run_generate(args: argparse.Namespace) -> None:
    device = _device(args.device)
    model, _ = load_checkpoint(args.checkpoint, device=device)
    tokenizer = Tokenizer.from_pickle(
        args.vocab,
        args.merges,
        special_tokens=["<|endoftext|>"],
    )
    print(
        generate_text(
            model,
            tokenizer,
            args.prompt,
            max_new_tokens=args.max_new_tokens,
            device=device,
            temperature=args.temperature,
            top_k=args.top_k,
        )
    )


def add_train_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_ROOT / "pretrained_models/tokenizer.npy",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "pretrained_models/tinystories_lm.pt",
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cuda:6")
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=1_024)
    parser.add_argument("--max-lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--cosine-steps", type=int)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--validation-fraction", type=float, default=0.01)
    parser.add_argument("--save-every", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    add_train_arguments(train_parser)
    train_parser.set_defaults(func=train)

    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "pretrained_models/tinystories_lm.pt",
    )
    generate_parser.add_argument(
        "--vocab",
        type=Path,
        default=PROJECT_ROOT / "pretrained_models/vocab_10000.pkl",
    )
    generate_parser.add_argument(
        "--merges",
        type=Path,
        default=PROJECT_ROOT / "pretrained_models/merges_10000.pkl",
    )
    generate_parser.add_argument("--device", default="cuda:6")
    generate_parser.add_argument("--prompt", required=True)
    generate_parser.add_argument("--max-new-tokens", type=int, default=100)
    generate_parser.add_argument("--temperature", type=float, default=0.8)
    generate_parser.add_argument("--top-k", type=int, default=50)
    generate_parser.set_defaults(func=run_generate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
