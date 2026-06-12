"""Base training for the Thunk v0 baseline translator.

Phase-2 supervised seq2seq translation (SPEC.md Section 6.1) on the offline
parallel corpus in data.py. Trains the shared tokenizer, then trains the
encoder-decoder with teacher forcing, label smoothing, gradient clipping, and a
warmup-stable-decay LR schedule (SPEC.md Section 6.4). Evaluation reports both
val loss and exact-match decoding accuracy, since every pair is verifiable.

Run:  python train.py            # small config, full base-training run
      python train.py --steps 800  # shorter run
"""

import argparse
import math
import os
from functools import partial

import torch
from torch.utils.data import DataLoader

from config import small_config
from data import (build_corpus, write_tokenizer_corpus, Seq2SeqDataset, collate)
from model import Thunk
from tokenizer import train_tokenizer, Tokenizer


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def lr_multiplier(step, cfg):
    if step < cfg.warmup_steps:
        return step / max(1, cfg.warmup_steps)
    decay_start = int(cfg.max_steps * (1 - cfg.decay_frac))
    if step < decay_start:
        return 1.0
    progress = (step - decay_start) / max(1, cfg.max_steps - decay_start)
    return cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * (1 - progress)


def build_optimizer(model, cfg):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or "tok_emb" in name:
            no_decay.append(p)         # norms + embeddings: no weight decay
        else:
            decay.append(p)
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=(0.9, 0.95), eps=1e-8)


@torch.no_grad()
def evaluate(model, loader, device, tok, max_decode=80):
    model.eval()
    total_loss, n_batches = 0.0, 0
    for batch in loader:
        src = batch["src"].to(device)
        tgt_in = batch["tgt_in"].to(device)
        labels = batch["labels"].to(device)
        _, loss = model(src, tgt_in, batch["src_mask"].to(device),
                        batch["tgt_mask"].to(device), labels=labels)
        total_loss += loss.item()
        n_batches += 1
    val_loss = total_loss / max(1, n_batches)

    # exact-match decoding accuracy on a fixed slice of the val set
    correct = 0
    ds = loader.dataset
    n = min(max_decode, len(ds))
    for i in range(n):
        ex = ds[i]
        src = ex["src"][None].to(device)
        src_mask = torch.ones_like(src, dtype=torch.bool)
        out = model.generate(src, src_mask, bos_id=tok.bos_id, eos_id=tok.eos_id,
                             max_new_tokens=64)
        pred = tok.decode(out)
        gold = tok.decode(ex["labels"].tolist())
        correct += int(pred.strip() == gold.strip())
    return val_loss, correct / max(1, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--max-number", type=int, default=4999)
    args = ap.parse_args()

    cfg = small_config()
    if args.steps is not None:
        cfg.max_steps = args.steps
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size

    torch.manual_seed(cfg.seed)
    device = get_device()
    torch.set_num_threads(os.cpu_count() or 1)
    print(f"Device: {device} | threads: {torch.get_num_threads()}")

    os.makedirs(cfg.data_dir, exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # ---- corpus + tokenizer ----
    train_pairs, val_pairs = build_corpus(max_number=args.max_number, seed=cfg.seed)
    print(f"Corpus: {len(train_pairs):,} train / {len(val_pairs):,} val pairs")

    prefix = cfg.tokenizer_path.replace(".model", "")
    if not os.path.exists(cfg.tokenizer_path):
        corpus_txt = os.path.join(cfg.data_dir, "tokenizer_corpus.txt")
        write_tokenizer_corpus(train_pairs, corpus_txt)
        print(f"Training tokenizer (vocab_size={cfg.vocab_size})...")
        train_tokenizer(corpus_txt, prefix, cfg.vocab_size, cfg.control_tags)
    tok = Tokenizer(cfg.tokenizer_path)
    cfg.vocab_size = tok.vocab_size
    print(f"Tokenizer vocab size: {cfg.vocab_size}")

    # ---- data ----
    train_ds = Seq2SeqDataset(train_pairs, tok, cfg.max_seq_len)
    val_ds = Seq2SeqDataset(val_pairs, tok, cfg.max_seq_len)
    collate_fn = partial(collate, pad_id=tok.pad_id)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            collate_fn=collate_fn)

    # ---- model + optimizer ----
    model = Thunk(cfg).to(device)
    print(f"Model parameters: {model.param_count():,}")
    optimizer = build_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: lr_multiplier(s, cfg))

    step, best_acc = 0, -1.0
    done = False
    while not done:
        model.train()
        for batch in train_loader:
            src = batch["src"].to(device)
            tgt_in = batch["tgt_in"].to(device)
            labels = batch["labels"].to(device)
            _, loss = model(src, tgt_in, batch["src_mask"].to(device),
                            batch["tgt_mask"].to(device), labels=labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()
            step += 1

            if step % 50 == 0:
                lr = scheduler.get_last_lr()[0]
                print(f"step {step:5d} | loss {loss.item():.4f} | lr {lr:.2e}")

            if step % cfg.eval_interval == 0:
                val_loss, acc = evaluate(model, val_loader, device, tok)
                print(f"  [eval @ {step}] val_loss={val_loss:.4f} "
                      f"exact_match={acc:.3f}")
                if acc > best_acc:
                    best_acc = acc
                    save_checkpoint(model, cfg, step, "checkpoints/best.pt")
                model.train()

            if step % cfg.save_interval == 0:
                save_checkpoint(model, cfg, step, f"checkpoints/step_{step}.pt")

            if step >= cfg.max_steps:
                done = True
                break

    val_loss, acc = evaluate(model, val_loader, device, tok)
    print(f"\nFinal: val_loss={val_loss:.4f} exact_match={acc:.3f} "
          f"(best exact_match={max(best_acc, acc):.3f})")
    save_checkpoint(model, cfg, step, "checkpoints/final.pt")


def save_checkpoint(model, cfg, step, path):
    torch.save({"model": model.state_dict(), "config": cfg, "step": step}, path)
    print(f"  saved {path}")


if __name__ == "__main__":
    main()
