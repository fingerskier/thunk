"""Offline parallel corpus + seq2seq dataset for the Thunk v0 base training.

Network access is restricted in this environment, so the base-training corpus is
generated deterministically rather than downloaded. It is still a real, meaning-
preserving translation task across several SPEC.md Section 4 directions, which is
exactly what the v0 milestone calls for (English <-> code/script and English ->
English style transfer):

  * <src:digits>  <-> <tgt:english>     number spelling (compositional, the bulk)
  * <src:english> <-> <tgt:english>     comparison paraphrase (semantic rewrite)
  * <src:python>  <-> <tgt:english>     tiny code <-> description
  * <src:bash>    <-> <tgt:powershell>  shell command porting

Every pair is exact-match verifiable, so training progress can be measured by
decoding accuracy, not just loss.
"""

import os
import random
from typing import List, Tuple

import torch
from torch.utils.data import Dataset

from tokenizer import Tokenizer

Pair = Tuple[str, str]

_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
         "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
         "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
         "eighty", "ninety"]


def number_to_words(n: int) -> str:
    if n < 20:
        return _ONES[n]
    if n < 100:
        t, r = divmod(n, 10)
        return _TENS[t] + ("-" + _ONES[r] if r else "")
    if n < 1000:
        h, r = divmod(n, 100)
        return _ONES[h] + " hundred" + (" " + number_to_words(r) if r else "")
    if n < 1_000_000:
        th, r = divmod(n, 1000)
        return number_to_words(th) + " thousand" + (" " + number_to_words(r) if r else "")
    raise ValueError("only supports n < 1,000,000")


def _tag(src: str, tgt: str, text: str) -> str:
    return f"<src:{src}> <tgt:{tgt}> {text}"


def _number_pairs(max_n: int) -> List[Pair]:
    pairs = []
    for n in range(max_n + 1):
        words = number_to_words(n)
        pairs.append((_tag("digits", "english", str(n)), words))
        pairs.append((_tag("english", "digits", words), str(n)))
    return pairs


def _paraphrase_pairs(max_n: int = 50) -> List[Pair]:
    """English -> English semantic rewrite (SPEC.md Section 1): a comparison and
    its logically equivalent inverse. Deterministic, meaning-preserving, and
    fully lowercase so it tokenizes cleanly (unlike case transfer)."""
    pairs = []
    for a in range(max_n + 1):
        for b in range(a):  # b < a, so the comparison direction is unambiguous
            gt = f"{a} is greater than {b}"
            lt = f"{b} is less than {a}"
            pairs.append((_tag("english", "english", gt), lt))
            pairs.append((_tag("english", "english", lt), gt))
    return pairs


_VARS = ["x", "y", "z", "total", "count", "result", "value", "n", "acc", "out"]
_PY_OPS = [("+", "sum"), ("-", "difference"), ("*", "product"), ("/", "quotient")]


def _python_pairs() -> List[Pair]:
    pairs = []
    for v in _VARS:
        for a in range(0, 10):
            for op, name in _PY_OPS:
                b = a + 1
                code = f"{v} = {a} {op} {b}"
                desc = f"assign the {name} of {a} and {b} to {v}"
                pairs.append((_tag("python", "english", code), desc))
                pairs.append((_tag("english", "python", desc), code))
        code = f"print({v})"
        pairs.append((_tag("python", "english", code), f"print {v}"))
        pairs.append((_tag("english", "python", f"print {v}"), code))
    return pairs


_FILES = ["report.txt", "data.csv", "build", "logs", "main.py", "notes.md",
          "image.png", "archive.zip", "config.json", "tmp",
          "index.html", "server.js", "styles.css", "readme.md", "test.py",
          "backup.tar", "results.json", "draft.txt", "schema.sql", "app.log",
          "model.bin", "video.mp4", "cache", "secrets.env", "output.csv"]


def _shell_pairs() -> List[Pair]:
    pairs = []
    for f in _FILES:
        templates = [
            (f"ls -la {f}", f"Get-ChildItem -Force {f}"),
            (f"rm {f}", f"Remove-Item {f}"),
            (f"mkdir {f}", f"New-Item -ItemType Directory {f}"),
            (f"cat {f}", f"Get-Content {f}"),
            (f"cp {f} backup", f"Copy-Item {f} backup"),
            (f"mv {f} backup", f"Move-Item {f} backup"),
        ]
        for bash, ps in templates:
            pairs.append((_tag("bash", "powershell", bash), ps))
            pairs.append((_tag("powershell", "bash", ps), bash))
    return pairs


def _upsample(pairs: List[Pair], floor: int, rng: random.Random) -> List[Pair]:
    """Repeat a direction group up to ``floor`` examples (temperature-style
    upsampling of low-resource directions, SPEC.md Sections 5-6). Distinct pairs
    are preserved; only duplicates are added to reach the floor."""
    if len(pairs) >= floor:
        return list(pairs)
    out = list(pairs)
    while len(out) < floor:
        extra = list(pairs)
        rng.shuffle(extra)
        out += extra[: floor - len(out)]
    return out


def build_corpus(max_number: int = 4999, seed: int = 1337, floor: int = 2200
                 ) -> Tuple[List[Pair], List[Pair]]:
    """Build the full parallel corpus and return (train, val) splits.

    The number-spelling direction naturally dominates, so the smaller directions
    are upsampled to a per-group floor before the train/val split. This keeps the
    held-out evaluation honest (val is split from distinct pairs) while giving
    each direction enough gradient signal to actually learn.
    """
    rng = random.Random(seed)

    # split each group into distinct train/val before any upsampling, so val
    # never contains a pair that was duplicated into train.
    def split(pairs):
        pairs = list(pairs)
        rng.shuffle(pairs)
        k = max(20, len(pairs) // 20)
        return pairs[k:], pairs[:k]

    groups = [
        _number_pairs(max_number),
        _paraphrase_pairs(),
        _python_pairs(),
        _shell_pairs(),
    ]

    train, val = [], []
    for g in groups:
        tr, va = split(g)
        train += _upsample(tr, floor, rng)
        val += va

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def write_tokenizer_corpus(pairs: List[Pair], path: str):
    with open(path, "w") as f:
        for src, tgt in pairs:
            f.write(src + "\n")
            f.write(tgt + "\n")


class Seq2SeqDataset(Dataset):
    """Tokenizes (src, tgt) pairs into encoder/decoder tensors."""

    def __init__(self, pairs: List[Pair], tok: Tokenizer, max_len: int = 256):
        self.tok = tok
        self.max_len = max_len
        self.examples = []
        for src, tgt in pairs:
            src_ids = tok.encode(src)[: max_len]
            tgt_ids = tok.encode(tgt)[: max_len - 1]
            self.examples.append((src_ids, tgt_ids))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        src_ids, tgt_ids = self.examples[idx]
        tgt_in = [self.tok.bos_id] + tgt_ids
        labels = tgt_ids + [self.tok.eos_id]
        return {
            "src": torch.tensor(src_ids, dtype=torch.long),
            "tgt_in": torch.tensor(tgt_in, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def collate(batch, pad_id: int = 0):
    def pad(seqs, fill):
        m = max(len(s) for s in seqs)
        out = torch.full((len(seqs), m), fill, dtype=torch.long)
        for i, s in enumerate(seqs):
            out[i, : len(s)] = s
        return out

    src = pad([b["src"] for b in batch], pad_id)
    tgt_in = pad([b["tgt_in"] for b in batch], pad_id)
    labels = pad([b["labels"] for b in batch], pad_id)
    return {
        "src": src,
        "src_mask": src != pad_id,
        "tgt_in": tgt_in,
        "tgt_mask": tgt_in != pad_id,
        "labels": labels,
    }


if __name__ == "__main__":
    train, val = build_corpus()
    print(f"train pairs: {len(train):,}  val pairs: {len(val):,}")
    for src, tgt in train[:6]:
        print(f"  {src!r:60s} -> {tgt!r}")
    print("number_to_words(3247):", number_to_words(3247))
