"""Per-direction exact-match evaluation for a trained Thunk v0 checkpoint.

The aggregate accuracy printed during training is dominated by the number-
spelling direction. This script reports exact-match decoding accuracy for each
translation direction separately (SPEC.md Section 11: exact preservation of
identifiers and meaning), so weaker directions are visible.

Usage:  python eval_directions.py [--ckpt checkpoints/best.pt] [--per-dir 60]
"""

import argparse
from collections import defaultdict

import torch

from config import small_config
from data import build_corpus
from model import Thunk
from tokenizer import Tokenizer
from translate import load_model


def direction_of(src_text: str) -> str:
    src = src_text.split("<src:")[1].split(">")[0]
    tgt = src_text.split("<tgt:")[1].split(">")[0]
    return f"{src}->{tgt}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/best.pt")
    ap.add_argument("--tokenizer", default="tokenizer.model")
    ap.add_argument("--per-dir", type=int, default=60,
                    help="max held-out examples to decode per direction")
    ap.add_argument("--beam", type=int, default=1)
    args = ap.parse_args()

    model, cfg = load_model(args.ckpt)
    tok = Tokenizer(args.tokenizer)

    # rebuild the same val split the training used (same seed)
    _, val_pairs = build_corpus(seed=small_config().seed)
    by_dir = defaultdict(list)
    for src, tgt in val_pairs:
        by_dir[direction_of(src)].append((src, tgt))

    print(f"checkpoint: {args.ckpt}  beam={args.beam}\n")
    print(f"{'direction':22s} {'acc':>7s}  {'n':>4s}  example (pred vs gold)")
    print("-" * 78)
    total_correct = total_n = 0
    for direction in sorted(by_dir):
        pairs = by_dir[direction][: args.per_dir]
        correct = 0
        shown = ""
        for src, tgt in pairs:
            src_ids = torch.tensor([tok.encode(src)], dtype=torch.long)
            src_mask = torch.ones_like(src_ids, dtype=torch.bool)
            out = model.generate(src_ids, src_mask, bos_id=tok.bos_id,
                                 eos_id=tok.eos_id, max_new_tokens=64,
                                 beam_size=args.beam)
            pred = tok.decode(out).strip()
            ok = pred == tgt.strip()
            correct += ok
            if not shown:
                mark = "OK " if ok else "XX "
                shown = f"{mark}{pred!r} vs {tgt!r}"
        acc = correct / max(1, len(pairs))
        total_correct += correct
        total_n += len(pairs)
        print(f"{direction:22s} {acc:7.3f}  {len(pairs):4d}  {shown}")
    print("-" * 78)
    print(f"{'OVERALL':22s} {total_correct/max(1,total_n):7.3f}  {total_n:4d}")


if __name__ == "__main__":
    main()
