"""Inference for model #2: auto-halting generation and latent embedding.

Decodes token by token; each token loops the weight-tied core until the KL
between successive output distributions drops under ``--kl-threshold``
(Huginn's training-free exit), with the mod-k KV-cache budget from PLAN.md.
Prints the generated text plus the halt-depth metadata — the per-token loop
counts that downstream chain links read as a confidence signal.

    python generate.py --ckpt checkpoints/model2-smoke.pt --prompt "count: 1 2 3"
    python generate.py --ckpt ... --prompt "..." --embed   # latent vector hook
"""

import argparse
from collections import Counter

import torch

from config import RecurrentLMConfig
from data import load_tokenizer
from model import RecurrentLM


def load_model(ckpt_path: str, device: str, tokenizer_override=None):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    cfg = RecurrentLMConfig(**ckpt["config"])
    model = RecurrentLM(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    tok_info = ckpt.get("tokenizer", {"kind": "byte", "path": None})
    tok_path = tokenizer_override or tok_info.get("path")
    if tokenizer_override is None and tok_info.get("kind") == "byte":
        tok_path = None
    tokenizer = load_tokenizer(tok_path)
    if tokenizer.vocab_size != cfg.vocab_size:
        raise ValueError(
            f"Tokenizer vocab ({tokenizer.vocab_size}) does not match the "
            f"checkpoint vocab ({cfg.vocab_size}) — pass the same pinned "
            "artifact the model was trained with (--tokenizer)."
        )
    return model, cfg, tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--tokenizer", default=None,
                        help="pinned tokenizer artifact (defaults to the one "
                             "recorded in the checkpoint)")
    parser.add_argument("--prompt", default="count: 1 2 3")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--kl-threshold", type=float, default=None,
                        help="halting KL threshold (default: config, 5e-4)")
    parser.add_argument("--max-loops", type=int, default=None,
                        help="loop cap per token (default: config)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--embed", action="store_true",
                        help="print the mean-pooled converged latent instead "
                             "of generating (the chain/memory hook)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    model, cfg, tokenizer = load_model(args.ckpt, args.device, args.tokenizer)
    ids = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long,
                       device=args.device)
    if ids.numel() == 0:
        raise SystemExit("empty prompt after tokenization")

    if args.embed:
        vec = model.embed(ids)
        print(f"latent embedding (d={vec.size(-1)}, loops={round(cfg.mean_loops)}):")
        print(vec.squeeze(0).tolist())
        return

    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    out, depths = model.generate(
        ids, args.max_new_tokens,
        kl_threshold=args.kl_threshold, max_loops=args.max_loops,
        temperature=args.temperature, top_k=args.top_k,
        eos_id=tokenizer.eos_id, generator=generator,
    )
    text = tokenizer.decode(out[0].tolist())
    print(text)

    if depths:
        max_loops = args.max_loops or cfg.max_loops_infer
        mean_depth = sum(depths) / len(depths)
        hist = Counter(depths)
        bar = " ".join(f"{d}:{hist[d]}" for d in sorted(hist))
        saved = 1.0 - mean_depth / max_loops
        print(f"\nhalt depths ({len(depths)} tokens): mean {mean_depth:.2f} "
              f"of max {max_loops} ({saved:.0%} compute saved vs always-max)")
        print(f"depth histogram: {bar}")


if __name__ == "__main__":
    main()
