import os
import sentencepiece as spm
from datasets import load_dataset
from torch.utils.data import Dataset

from config import ThunkConfig


def train_tokenizer(cfg: ThunkConfig, text_file: str = "train_text.txt"):
    """Train a SentencePiece BPE tokenizer on the training corpus."""
    if os.path.exists(cfg.tokenizer_path):
        print(f"Tokenizer already exists at {cfg.tokenizer_path}")
        return

    print("Downloading dataset for tokenizer training...")
    ds = load_dataset(cfg.dataset_name, split="train", streaming=True)

    # write a subset to disk for tokenizer training
    print(f"Writing text to {text_file}...")
    with open(text_file, "w") as f:
        for i, row in enumerate(ds):
            if i >= 100_000:
                break
            f.write(row["text"].strip() + "\n")

    print(f"Training tokenizer (vocab_size={cfg.vocab_size})...")
    spm.SentencePieceTrainer.train(
        input=text_file,
        model_prefix=cfg.tokenizer_path.replace(".model", ""),
        vocab_size=cfg.vocab_size,
        model_type="bpe",
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        character_coverage=1.0,
        num_threads=os.cpu_count(),
    )
    print(f"Tokenizer saved to {cfg.tokenizer_path}")

    # clean up temp file
    os.remove(text_file)


def load_tokenizer(cfg: ThunkConfig) -> spm.SentencePieceProcessor:
    sp = spm.SentencePieceProcessor()
    sp.load(cfg.tokenizer_path)
    return sp


class TinyStoriesDataset(Dataset):
    """Loads TinyStories and tokenizes on-the-fly."""

    def __init__(self, cfg: ThunkConfig, split: str = "train", max_examples: int = 500_000):
        self.cfg = cfg
        self.sp = load_tokenizer(cfg)
        self.max_len = cfg.max_seq_len

        print(f"Loading {split} split...")
        ds = load_dataset(cfg.dataset_name, split=split)
        if max_examples and len(ds) > max_examples:
            ds = ds.select(range(max_examples))

        print(f"Tokenizing {len(ds)} examples...")
        self.tokens = []
        for row in ds:
            ids = self.sp.encode(row["text"])
            ids = [self.sp.bos_id()] + ids + [self.sp.eos_id()]
            self.tokens.extend(ids)

        print(f"Total tokens: {len(self.tokens):,}")

    def __len__(self):
        return (len(self.tokens) - 1) // self.max_len

    def __getitem__(self, idx):
        import torch

        start = idx * self.max_len
        chunk = self.tokens[start : start + self.max_len + 1]

        # pad if needed
        if len(chunk) < self.max_len + 1:
            chunk = chunk + [0] * (self.max_seq_len + 1 - len(chunk))

        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


if __name__ == "__main__":
    cfg = ThunkConfig()
    train_tokenizer(cfg)
    ds = TinyStoriesDataset(cfg, split="train", max_examples=10_000)
    print(f"Dataset size: {len(ds)} chunks of {cfg.max_seq_len} tokens")
    x, y = ds[0]
    print(f"x: {x.shape}, y: {y.shape}")
