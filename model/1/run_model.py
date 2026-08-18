"""CLI runner for model #1 recursive reasoning diffuser checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from config import tiny_config
from model import ReasoningDiffuser
from text_io import CharacterTokenizer, generate_answer_ids


MODEL_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = MODEL_DIR / "checkpoints" / "model1.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a trained model #1 checkpoint.")
    parser.add_argument("prompt", nargs="?", help="Question to answer.")
    parser.add_argument("--question", help="Question to answer. Overrides the positional prompt.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--weights", choices=("ema", "model"), default="ema")
    parser.add_argument("--steps", type=int, help="Override checkpoint supervision_steps at inference.")
    parser.add_argument("--halt-threshold", type=float, default=0.5)
    parser.add_argument("--show-ids", action="store_true", help="Print generated token ids after the decoded answer.")
    return parser.parse_args()


def pick_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def main() -> int:
    args = parse_args()
    question = args.question or args.prompt
    if not question:
        raise SystemExit("provide a question with --question or as a positional prompt")
    if not args.checkpoint.exists():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")

    device = pick_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    cfg = tiny_config(**checkpoint["config"])
    tokenizer = CharacterTokenizer.from_metadata(checkpoint["tokenizer"])
    model = ReasoningDiffuser(cfg).to(device)
    state_key = "ema" if args.weights == "ema" and "ema" in checkpoint else "model"
    model.load_state_dict(checkpoint[state_key])

    question_ids = torch.tensor(tokenizer.encode(question, cfg.max_question_len), dtype=torch.long, device=device)
    question_mask = question_ids.ne(cfg.pad_id)
    answer_ids, used_steps, halt_scores = generate_answer_ids(
        model,
        cfg,
        question_ids,
        question_mask,
        steps=args.steps,
        halt_threshold=args.halt_threshold,
    )
    answer = tokenizer.decode(answer_ids[0].tolist())
    print(answer)
    print(f"steps={used_steps} halt={halt_scores[0].item():.4f} weights={state_key}")
    if args.show_ids:
        print("ids=" + " ".join(str(token_id) for token_id in answer_ids[0].tolist()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
