import os
import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import ThunkConfig
from model import Thunk
from data import train_tokenizer, TinyStoriesDataset


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train(cfg: ThunkConfig):
    device = get_device()
    print(f"Device: {device}")

    # tokenizer
    train_tokenizer(cfg)

    # data
    train_ds = TinyStoriesDataset(cfg, split="train", max_examples=500_000)
    val_ds = TinyStoriesDataset(cfg, split="validation", max_examples=10_000)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # model
    model = Thunk(cfg).to(device)
    print(f"Parameters: {model.param_count():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # linear warmup then cosine decay
    def lr_schedule(step):
        if step < cfg.warmup_steps:
            return step / cfg.warmup_steps
        progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
        return 0.5 * (1.0 + __import__("math").cos(__import__("math").pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    os.makedirs("checkpoints", exist_ok=True)

    step = 0
    best_val_loss = float("inf")

    for epoch in range(100):  # will break on max_steps
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for x, y in pbar:
            if step >= cfg.max_steps:
                break

            x, y = x.to(device), y.to(device)
            logits, loss, depth = model(x, y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()

            pbar.set_postfix(loss=f"{loss.item():.3f}", depth=depth, lr=f"{scheduler.get_last_lr()[0]:.2e}")
            step += 1

            # eval
            if step % cfg.eval_interval == 0:
                val_loss = evaluate(model, val_loader, device)
                print(f"\n[Step {step}] val_loss={val_loss:.4f}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(model, optimizer, step, cfg, "checkpoints/best.pt")
                model.train()

            # periodic save
            if step % cfg.save_interval == 0:
                save_checkpoint(model, optimizer, step, cfg, f"checkpoints/step_{step}.pt")

        if step >= cfg.max_steps:
            break

    save_checkpoint(model, optimizer, step, cfg, "checkpoints/final.pt")
    print(f"Training complete. Best val loss: {best_val_loss:.4f}")


@torch.no_grad()
def evaluate(model, val_loader, device, max_batches=50):
    model.eval()
    total_loss = 0
    count = 0
    for x, y in val_loader:
        if count >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        _, loss, _ = model(x, y)
        total_loss += loss.item()
        count += 1
    return total_loss / max(count, 1)


def save_checkpoint(model, optimizer, step, cfg, path):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "config": cfg,
    }, path)
    print(f"Saved checkpoint: {path}")


if __name__ == "__main__":
    cfg = ThunkConfig()
    train(cfg)
