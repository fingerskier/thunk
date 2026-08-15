"""LM data loading for model #2.

Tokenizer policy (PLAN.md composability requirement + the open tokenizer
issue): model #2 **loads** a pinned tokenizer artifact shared with model/0 —
it never trains or regenerates one. Point ``--tokenizer`` at the versioned
``tokenizer.model``; if the file is missing this module raises instead of
silently training a fresh vocab. The byte-level fallback exists only for
deterministic smoke tests with no external dependencies.

Data is plain text: one document per line. ``script/glean_datasets.py
--model 0`` already emits a suitable ``data/model/0/train_text.txt``; any
other text corpus works the same way.
"""

import os
from typing import List, Optional, Sequence

import torch

PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3   # fixed, matches model/0


class ByteTokenizer:
    """Deterministic byte-level fallback: 4 special ids + 256 byte ids."""

    kind = "byte"
    pad_id, unk_id, bos_id, eos_id = PAD_ID, UNK_ID, BOS_ID, EOS_ID

    @property
    def vocab_size(self) -> int:
        return 260

    def encode(self, text: str) -> List[int]:
        return [b + 4 for b in text.encode("utf-8")]

    def decode(self, ids: Sequence[int]) -> str:
        data = bytes(i - 4 for i in ids if i >= 4)
        return data.decode("utf-8", errors="replace")


class SentencePieceTokenizer:
    """Thin wrapper over a pinned SentencePiece artifact (model/0 format)."""

    kind = "sentencepiece"
    pad_id, unk_id, bos_id, eos_id = PAD_ID, UNK_ID, BOS_ID, EOS_ID

    def __init__(self, model_path: str):
        import sentencepiece as spm
        self.path = model_path
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)

    @property
    def vocab_size(self) -> int:
        return self.sp.get_piece_size()

    def encode(self, text: str) -> List[int]:
        return self.sp.encode(text, out_type=int)

    def decode(self, ids: Sequence[int]) -> str:
        ids = [i for i in ids if i not in (self.pad_id, self.bos_id, self.eos_id)]
        return self.sp.decode(ids)


def load_tokenizer(path: Optional[str]):
    """Load the pinned tokenizer, or the byte fallback when ``path`` is None."""
    if path is None:
        return ByteTokenizer()
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Pinned tokenizer artifact not found: {path}. Model #2 never "
            "trains its own tokenizer (shared-vocab composability with "
            "model/0); point --tokenizer at the versioned tokenizer.model, "
            "or omit it to use the byte-level smoke tokenizer."
        )
    return SentencePieceTokenizer(path)


def synthetic_corpus(n_lines: int = 2000, seed: int = 1337) -> List[str]:
    """Deterministic structured text for offline smoke runs: counting,
    letter patterns, and small sums — enough signal for loss to fall."""
    import random
    rng = random.Random(seed)
    lines = []
    for _ in range(n_lines):
        kind = rng.randrange(3)
        if kind == 0:
            start = rng.randrange(0, 40)
            n = rng.randrange(4, 9)
            seq = " ".join(str(start + i) for i in range(n))
            lines.append(f"count: {seq}")
        elif kind == 1:
            word = "".join(rng.choice("abcdef") for _ in range(rng.randrange(2, 5)))
            lines.append(" ".join([word] * rng.randrange(3, 7)))
        else:
            a, b = rng.randrange(0, 30), rng.randrange(0, 30)
            lines.append(f"{a} + {b} = {a + b}")
    return lines


class LMDataset:
    """Concatenated-token LM dataset with a train/val split.

    Documents are joined with EOS; batches are random ``seq_len + 1`` windows
    yielding (input, next-token target) pairs.
    """

    def __init__(self, lines: Sequence[str], tokenizer, seq_len: int,
                 val_fraction: float = 0.1, seed: int = 1337):
        self.seq_len = seq_len
        ids: List[int] = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            ids.extend(tokenizer.encode(line))
            ids.append(tokenizer.eos_id)
        if len(ids) < (seq_len + 1) * 2:
            raise ValueError(
                f"Corpus too small: {len(ids)} tokens for seq_len={seq_len}."
            )
        stream = torch.tensor(ids, dtype=torch.long)
        split = max(seq_len + 1, int(len(stream) * (1 - val_fraction)))
        self.train_stream = stream[:split]
        self.val_stream = stream[split:] if len(stream) - split > seq_len else stream[:split]
        self.generator = torch.Generator().manual_seed(seed)

    def get_batch(self, batch_size: int, split: str = "train",
                  device: str = "cpu"):
        stream = self.train_stream if split == "train" else self.val_stream
        max_start = len(stream) - self.seq_len - 1
        starts = torch.randint(0, max_start + 1, (batch_size,),
                               generator=self.generator)
        x = torch.stack([stream[s:s + self.seq_len] for s in starts])
        y = torch.stack([stream[s + 1:s + self.seq_len + 1] for s in starts])
        return x.to(device), y.to(device)


def load_lines(paths: Sequence[str]) -> List[str]:
    lines: List[str] = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as fh:
            lines.extend(fh.read().splitlines())
    return lines
