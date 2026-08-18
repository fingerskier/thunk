"""CLI trainer for model #1 recursive reasoning diffuser."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config import tiny_config
from model import ReasoningDiffuser
from text_io import CharacterTokenizer, QADataset, load_records, move_batch
from train import EMA, deep_supervision_step


MODEL_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODEL_DIR.parents[1]
DEFAULT_DATA = REPO_ROOT / "data" / "model" / "1" / "combined.jsonl"
DEFAULT_CHECKPOINT = MODEL_DIR / "checkpoints" / "model1.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train model #1 on question/answer JSONL records.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="JSONL file with question/answer rows.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="Checkpoint path to write.")
    parser.add_argument("--resume", type=Path, help="Checkpoint to continue training from.")
    parser.add_argument("--reset-optimizer", action="store_true", help="Do not restore optimizer state when resuming.")
    parser.add_argument("--steps", type=int, default=200, help="Number of optimizer updates to run.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-records", type=int, help="Use only the first N records.")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=0, help="Write an intermediate checkpoint every N steps.")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=512)
    parser.add_argument("--recursion-depth", type=int, default=2)
    parser.add_argument("--warmup-recursions", type=int, default=1)
    parser.add_argument("--supervision-steps", type=int, default=8)
    parser.add_argument("--max-question-len", type=int, default=64)
    parser.add_argument("--answer-len", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    return parser.parse_args()


def pick_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def build_config(args: argparse.Namespace, vocab_size: int):
    return tiny_config(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        recursion_depth=args.recursion_depth,
        warmup_recursions=args.warmup_recursions,
        supervision_steps=args.supervision_steps,
        max_question_len=args.max_question_len,
        answer_len=args.answer_len,
        lr=args.lr,
        weight_decay=args.weight_decay,
        ema_decay=args.ema_decay,
        label_smoothing=args.label_smoothing,
        seed=args.seed,
    )


def save_checkpoint(path: Path, cfg, model, ema, optimizer, tokenizer, step: int, records: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": asdict(cfg),
            "model": model.state_dict(),
            "ema": ema.shadow.state_dict(),
            "optimizer": optimizer.state_dict(),
            "tokenizer": tokenizer.to_metadata(),
            "step": step,
            "records": records,
        },
        path,
    )


def load_training_state(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device)
    cfg = tiny_config(**checkpoint["config"])
    tokenizer = CharacterTokenizer.from_metadata(checkpoint["tokenizer"])
    model = ReasoningDiffuser(cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
    )
    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    ema = EMA(model, cfg.ema_decay)
    if "ema" in checkpoint:
        ema.shadow.load_state_dict(checkpoint["ema"])
    return cfg, tokenizer, model, optimizer, ema, int(checkpoint.get("step", 0))


def main() -> int:
    args = parse_args()
    if args.steps < 1:
        raise SystemExit("--steps must be at least 1")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1")
    if not args.data.exists():
        raise SystemExit(
            f"data file not found: {args.data}\n"
            "Generate one with: python script/glean_datasets.py --model 1 --offline --limit 20"
        )

    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    records = load_records(args.data)
    if args.max_records:
        records = records[: args.max_records]

    if args.resume:
        cfg, tokenizer, model, optimizer, ema, start_step = load_training_state(args.resume, device)
        if args.reset_optimizer:
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=cfg.lr,
                weight_decay=cfg.weight_decay,
                betas=(0.9, 0.95),
            )
    else:
        tokenizer = CharacterTokenizer.build(records)
        cfg = build_config(args, tokenizer.vocab_size)
        model = ReasoningDiffuser(cfg).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=(0.9, 0.95),
        )
        ema = EMA(model, cfg.ema_decay)
        start_step = 0

    dataset = QADataset(records, tokenizer, cfg)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=generator)

    print(
        f"training records={len(dataset)} vocab={cfg.vocab_size} device={device} "
        f"steps={args.steps} checkpoint={args.checkpoint}"
    )
    model.train()
    loader_iter = iter(loader)
    for local_step in range(1, args.steps + 1):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)
        global_step = start_step + local_step
        loss = deep_supervision_step(model, move_batch(batch, device), optimizer, ema)
        model.train()
        if local_step == 1 or (args.log_every and local_step % args.log_every == 0):
            print(f"step {global_step}: loss={loss.item():.4f}")
        if args.save_every and local_step % args.save_every == 0:
            save_checkpoint(args.checkpoint, cfg, model, ema, optimizer, tokenizer, global_step, len(dataset))

    save_checkpoint(args.checkpoint, cfg, model, ema, optimizer, tokenizer, start_step + args.steps, len(dataset))
    print(f"saved checkpoint: {args.checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
