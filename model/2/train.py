"""Training loop for model #2, the auto-halting recurrent LM.

Per PLAN.md: plain next-token training with **randomized loop counts**
(log-normal Poisson, mean ``mean_loops``, capped at ``max_loops``) and
truncated backprop through the last ``bptt_k`` iterations — the Huginn
recipe, so the model is robust to whatever depth inference later chooses.
Evaluation sweeps validation loss over fixed test-time loop counts; the
Phase 0 exit gate is that more loops monotonically improve quality.

The ternary preset (``--config ternary``) additionally follows the b1.58
schedule: high peak LR with a two-stage cosine (abrupt mid-run drop) and
weight decay 0.1 in stage 1 -> 0 in stage 2.

Smoke test (CPU, no data or tokenizer needed):

    python train.py --config smoke --steps 30
"""

import argparse
import math
import os
import random
from dataclasses import asdict

import torch
import torch.nn.functional as F

from config import CONFIGS, RecurrentLMConfig
from data import LMDataset, load_lines, load_tokenizer, synthetic_corpus
from model import RecurrentLM, stablemax_cross_entropy


def sample_loops(cfg: RecurrentLMConfig, rng: random.Random) -> int:
    """Log-normal Poisson loop count (Huginn): r = 1 + Poisson(lambda),
    lambda ~ LogNormal tuned so E[r] ~= mean_loops, clipped to max_loops."""
    target = max(cfg.mean_loops - 1.0, 1e-3)
    lam = math.exp(rng.gauss(math.log(target) - cfg.loop_sigma ** 2 / 2,
                             cfg.loop_sigma))
    # Knuth Poisson sampler — deterministic under rng's seed.
    L, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= L:
            break
        k += 1
    return min(1 + k, cfg.max_loops)


def lr_at(cfg: RecurrentLMConfig, step: int) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    t = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    if not cfg.two_stage_lr:
        floor = cfg.lr * cfg.min_lr_ratio
        return floor + (cfg.lr - floor) * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))
    # b1.58 two-stage: cosine at high LR, abrupt drop at midpoint, low cosine.
    if t < 0.5:
        u = t / 0.5
        return cfg.lr * (0.5 + 0.5 * 0.5 * (1 + math.cos(math.pi * u)))  # lr -> 0.5*lr
    u = (t - 0.5) / 0.5
    peak2 = cfg.lr * cfg.stage2_lr_ratio
    floor = peak2 * cfg.min_lr_ratio
    return floor + (peak2 - floor) * 0.5 * (1 + math.cos(math.pi * min(u, 1.0)))


def wd_at(cfg: RecurrentLMConfig, step: int) -> float:
    if not cfg.two_stage_lr:
        return cfg.weight_decay
    t = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    return cfg.weight_decay if t < 0.5 else 0.0   # b1.58: WD 0.1 -> 0


def compute_loss(cfg: RecurrentLMConfig, logits, targets):
    if cfg.use_stablemax:
        return stablemax_cross_entropy(logits, targets,
                                       label_smoothing=cfg.label_smoothing)
    return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1),
                           label_smoothing=cfg.label_smoothing)


@torch.no_grad()
def evaluate(model: RecurrentLM, dataset: LMDataset, cfg: RecurrentLMConfig,
             device: str, r_values=None):
    """Validation loss at fixed test-time loop counts, measured with the
    configured training objective (stablemax by default) so the number is
    comparable to what training optimizes. The Phase 0 gate reads off this
    table: loss must improve monotonically with r (to ~2x the training mean)
    or recursion is buying nothing."""
    model.eval()
    r_values = r_values or sorted({1, 2, max(1, round(cfg.mean_loops)),
                                   cfg.max_loops, 2 * cfg.max_loops})
    losses = {}
    batches = [dataset.get_batch(cfg.batch_size, "val", device)
               for _ in range(cfg.eval_batches)]
    for r in r_values:
        total = 0.0
        for x, y in batches:
            logits, _ = model(x, loops=r)
            total += compute_loss(cfg, logits, y).item()
        losses[r] = total / len(batches)
    model.train()
    return losses


def format_eval(losses) -> str:
    cells = [f"r={r}: loss {l:.4f} ppl {math.exp(min(l, 20)):.1f}"
             for r, l in losses.items()]
    return " | ".join(cells)


def save_checkpoint(path: str, model: RecurrentLM, opt, cfg, step: int,
                    tokenizer):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "config": asdict(cfg),
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "step": step,
        "tokenizer": {"kind": tokenizer.kind,
                      "path": getattr(tokenizer, "path", None)},
    }, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", choices=sorted(CONFIGS), default="smoke")
    parser.add_argument("--data", nargs="*", default=[],
                        help="plain-text corpus files (one document per line); "
                             "omit for the deterministic synthetic smoke corpus")
    parser.add_argument("--tokenizer", default=None,
                        help="pinned SentencePiece artifact shared with model/0 "
                             "(never regenerated here); omit for byte fallback")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default="checkpoints")
    parser.add_argument("--resume", default=None, help="checkpoint to resume from")
    args = parser.parse_args()

    cfg = CONFIGS[args.config]()
    if args.steps is not None:
        cfg.max_steps = args.steps
        cfg.warmup_steps = min(cfg.warmup_steps, max(1, args.steps // 10))
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.seq_len is not None:
        cfg.seq_len = args.seq_len
        cfg.max_seq_len = max(cfg.max_seq_len, args.seq_len)

    tokenizer = load_tokenizer(args.tokenizer)
    cfg.vocab_size = tokenizer.vocab_size
    cfg.__post_init__()

    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)

    lines = load_lines(args.data) if args.data else synthetic_corpus(seed=cfg.seed)
    dataset = LMDataset(lines, tokenizer, cfg.seq_len, seed=cfg.seed)

    model = RecurrentLM(cfg).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.95),
                            weight_decay=cfg.weight_decay)
    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=args.device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]
        print(f"resumed from {args.resume} at step {start_step}")

    print(f"config={args.config} device={args.device} "
          f"params={model.num_params():,} "
          f"(non-embedding {model.num_params(non_embedding=True):,}) "
          f"vocab={cfg.vocab_size} quantize={cfg.quantize}")

    model.train()
    running = None
    for step in range(start_step, cfg.max_steps):
        lr = lr_at(cfg, step)
        wd = wd_at(cfg, step)
        for group in opt.param_groups:
            group["lr"] = lr
            group["weight_decay"] = wd

        x, y = dataset.get_batch(cfg.batch_size, "train", args.device)
        r = sample_loops(cfg, rng)
        logits, _ = model(x, loops=r, bptt_k=cfg.bptt_k)
        loss = compute_loss(cfg, logits, y)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        running = loss.item() if running is None else 0.95 * running + 0.05 * loss.item()
        if step % 10 == 0 or step == cfg.max_steps - 1:
            print(f"step {step:5d} r={r} lr={lr:.2e} wd={wd:.2f} "
                  f"loss {loss.item():.4f} (ema {running:.4f})")
        if (step + 1) % cfg.eval_interval == 0 or step == cfg.max_steps - 1:
            print("  val " + format_eval(evaluate(model, dataset, cfg, args.device)))
        if (step + 1) % cfg.save_interval == 0 or step == cfg.max_steps - 1:
            path = os.path.join(args.out, f"model2-{args.config}.pt")
            save_checkpoint(path, model, opt, cfg, step + 1, tokenizer)
            print(f"  saved {path}")

    print("final val " + format_eval(evaluate(model, dataset, cfg, args.device)))


if __name__ == "__main__":
    main()
