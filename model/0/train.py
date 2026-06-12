"""Base training for the Thunk v0 baseline translator.

Two phases from SPEC.md Section 6.1:

  Phase 1 — denoising pretrain: UL2-style span corruption + prefix-LM on the
            monolingual side of the corpus, so the encoder learns
            representations before it sees any translation tag.
  Phase 2 — supervised seq2seq translation: teacher forcing with label
            smoothing on the parallel pairs.

Optimizer is Muon (2D weights) + AdamW (embeddings/norms/head) per SPEC.md
Section 6.4, with an AdamW-only fallback. Schedule is warmup-stable-decay.

Run:  python train.py                 # full run: pretrain + translate
      python train.py --no-pretrain   # skip Phase 1
      python train.py --optimizer adamw
"""

import argparse
import os
from functools import partial

import torch
from torch.utils.data import DataLoader

from config import small_config
from data import (build_corpus, write_tokenizer_corpus, monolingual_texts,
                  pack_tokens, DenoisingDataset, Seq2SeqDataset, collate)
from model import Thunk
from optim import build_optimizers
from tokenizer import train_tokenizer, Tokenizer


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_schedulers(optimizers, cfg, total_steps, warmup):
    def mult(step):
        if step < warmup:
            return step / max(1, warmup)
        decay_start = int(total_steps * (1 - cfg.decay_frac))
        if step < decay_start:
            return 1.0
        progress = (step - decay_start) / max(1, total_steps - decay_start)
        return cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * (1 - progress)

    return [torch.optim.lr_scheduler.LambdaLR(o, mult) for o in optimizers]


def run_phase(model, loader, cfg, total_steps, warmup, device, label,
              on_eval=None, eval_interval=250):
    optimizers = build_optimizers(model, cfg)
    schedulers = make_schedulers(optimizers, cfg, total_steps, warmup)
    opt_names = "+".join(type(o).__name__ for o in optimizers)
    print(f"\n=== {label} | {total_steps} steps | optim: {opt_names} ===")

    step, done = 0, False
    while not done:
        model.train()
        for batch in loader:
            src = batch["src"].to(device)
            tgt_in = batch["tgt_in"].to(device)
            labels = batch["labels"].to(device)
            _, loss = model(src, tgt_in, batch["src_mask"].to(device),
                            batch["tgt_mask"].to(device), labels=labels)

            for opt in optimizers:
                opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            for opt in optimizers:
                opt.step()
            for sch in schedulers:
                sch.step()
            step += 1

            if step % 50 == 0:
                lrs = " ".join(f"{s.get_last_lr()[0]:.2e}" for s in schedulers)
                print(f"  [{label}] step {step:5d} | loss {loss.item():.4f} | lr {lrs}")
            if on_eval is not None and step % eval_interval == 0:
                on_eval(step)
                model.train()
            if step >= total_steps:
                done = True
                break
    return step


@torch.no_grad()
def denoise_loss(model, loader, device, max_batches=20):
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        if n >= max_batches:
            break
        _, loss = model(batch["src"].to(device), batch["tgt_in"].to(device),
                        batch["src_mask"].to(device), batch["tgt_mask"].to(device),
                        labels=batch["labels"].to(device))
        total += loss.item()
        n += 1
    return total / max(1, n)


@torch.no_grad()
def evaluate(model, loader, device, tok, max_decode=80):
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        _, loss = model(batch["src"].to(device), batch["tgt_in"].to(device),
                        batch["src_mask"].to(device), batch["tgt_mask"].to(device),
                        labels=batch["labels"].to(device))
        total += loss.item()
        n += 1
    val_loss = total / max(1, n)

    ds = loader.dataset
    correct = 0
    m = min(max_decode, len(ds))
    for i in range(m):
        ex = ds[i]
        src = ex["src"][None].to(device)
        out = model.generate(src, torch.ones_like(src, dtype=torch.bool),
                             bos_id=tok.bos_id, eos_id=tok.eos_id, max_new_tokens=64)
        if tok.decode(out).strip() == tok.decode(ex["labels"].tolist()).strip():
            correct += 1
    return val_loss, correct / max(1, m)


def save_checkpoint(model, cfg, step, path):
    torch.save({"model": model.state_dict(), "config": cfg, "step": step}, path)
    print(f"  saved {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=None, help="phase-2 steps")
    ap.add_argument("--denoise-steps", type=int, default=None)
    ap.add_argument("--no-pretrain", action="store_true")
    ap.add_argument("--optimizer", choices=["muon", "adamw"], default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--max-number", type=int, default=4999)
    args = ap.parse_args()

    cfg = small_config()
    if args.steps is not None:
        cfg.max_steps = args.steps
    if args.denoise_steps is not None:
        cfg.denoise_steps = args.denoise_steps
    if args.optimizer is not None:
        cfg.optimizer = args.optimizer
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size

    torch.manual_seed(cfg.seed)
    device = get_device()
    torch.set_num_threads(os.cpu_count() or 1)
    print(f"Device: {device} | threads: {torch.get_num_threads()} | optim: {cfg.optimizer}")

    os.makedirs(cfg.data_dir, exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # ---- corpus + tokenizer (control tags + denoising sentinels) ----
    train_pairs, val_pairs = build_corpus(max_number=args.max_number, seed=cfg.seed)
    print(f"Corpus: {len(train_pairs):,} train / {len(val_pairs):,} val pairs")

    prefix = cfg.tokenizer_path.replace(".model", "")
    if not os.path.exists(cfg.tokenizer_path):
        corpus_txt = os.path.join(cfg.data_dir, "tokenizer_corpus.txt")
        write_tokenizer_corpus(train_pairs, corpus_txt)
        print(f"Training tokenizer (vocab_size={cfg.vocab_size})...")
        train_tokenizer(corpus_txt, prefix, cfg.vocab_size,
                        cfg.control_tags + cfg.sentinels())
    tok = Tokenizer(cfg.tokenizer_path)
    cfg.vocab_size = tok.vocab_size
    print(f"Tokenizer vocab size: {cfg.vocab_size}")

    model = Thunk(cfg).to(device)
    print(f"Model parameters: {model.param_count():,}")
    collate_fn = partial(collate, pad_id=tok.pad_id)

    # ---- Phase 1: denoising pretrain ----
    if not args.no_pretrain and cfg.denoise_steps > 0:
        mono = monolingual_texts(train_pairs)
        blocks = pack_tokens(mono, tok, cfg.denoise_block, seed=cfg.seed)
        n_val = max(20, len(blocks) // 20)
        tr_ds = DenoisingDataset(blocks[n_val:], tok, cfg, seed=cfg.seed)
        va_ds = DenoisingDataset(blocks[:n_val], tok, cfg, seed=cfg.seed + 1)
        print(f"Denoising: {len(mono):,} monolingual texts -> {len(blocks):,} blocks")
        tr_loader = DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True,
                               collate_fn=collate_fn, drop_last=True)
        va_loader = DataLoader(va_ds, batch_size=cfg.batch_size, shuffle=False,
                               collate_fn=collate_fn)

        def on_eval(step):
            print(f"  [pretrain eval @ {step}] "
                  f"denoise_loss={denoise_loss(model, va_loader, device):.4f}")

        run_phase(model, tr_loader, cfg, cfg.denoise_steps,
                  warmup=min(cfg.warmup_steps, cfg.denoise_steps // 10),
                  device=device, label="pretrain", on_eval=on_eval,
                  eval_interval=cfg.eval_interval)
        save_checkpoint(model, cfg, cfg.denoise_steps, "checkpoints/pretrain.pt")

    # ---- Phase 2: supervised seq2seq translation ----
    train_ds = Seq2SeqDataset(train_pairs, tok, cfg.max_seq_len)
    val_ds = Seq2SeqDataset(val_pairs, tok, cfg.max_seq_len)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            collate_fn=collate_fn)

    best = {"acc": -1.0}

    def on_eval(step):
        val_loss, acc = evaluate(model, val_loader, device, tok)
        print(f"  [translate eval @ {step}] val_loss={val_loss:.4f} exact_match={acc:.3f}")
        if acc > best["acc"]:
            best["acc"] = acc
            save_checkpoint(model, cfg, step, "checkpoints/best.pt")

    run_phase(model, train_loader, cfg, cfg.max_steps, warmup=cfg.warmup_steps,
              device=device, label="translate", on_eval=on_eval,
              eval_interval=cfg.eval_interval)

    val_loss, acc = evaluate(model, val_loader, device, tok)
    print(f"\nFinal: val_loss={val_loss:.4f} exact_match={acc:.3f} "
          f"(best exact_match={max(best['acc'], acc):.3f})")
    save_checkpoint(model, cfg, cfg.max_steps, "checkpoints/final.pt")


if __name__ == "__main__":
    main()
