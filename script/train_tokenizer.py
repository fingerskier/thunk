"""Train the **pinned shared tokenizer** once and version it.

Every model in the chain (model/0 seq2seq, model/2 recurrent LM, ...) must
speak the same token space, so the tokenizer is trained exactly once by this
script, committed under ``tokenizer/vN/``, and then only ever *loaded* by the
model training scripts. Regenerating it is a breaking change (bump ``vN``,
retrain every model that shares it).

Settings are model/0's ``train_tokenizer`` (BPE, byte fallback, digit
splitting, identity normalization so code whitespace survives, fixed special
ids pad=0/unk=1/bos=2/eos=3) plus the union of every control tag used by
model/0 and the model/2 translation set (``TRANSLATION_DATA.md``), each
reserved as a single symbol.

Usage (repo root):

    python script/train_tokenizer.py                       # default inputs
    python script/train_tokenizer.py --data a.txt b.txt --out tokenizer/v2/tokenizer.model

Defaults: inputs = whichever of data/model/0/train_text.txt and
data/model/2/train_text.txt exist; output tokenizer/v1/tokenizer.model;
vocab 8000. A MANIFEST.json next to the artifact records vocab, tags, and
input hashes so the artifact is reproducible.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from typing import Iterable, List

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_INPUTS = [
    os.path.join(REPO, "data", "model", "0", "train_text.txt"),
    os.path.join(REPO, "data", "model", "2", "train_text.txt"),
]
DEFAULT_OUT = os.path.join(REPO, "tokenizer", "v1", "tokenizer.model")
DEFAULT_VOCAB = 8000

# Languages of the model/2 translation set (TRANSLATION_DATA.md).
MODEL2_LANGS = ("english", "lean4", "python", "javascript", "java", "go",
                "php", "ruby", "c", "cpp", "rust", "csharp")
SEP_TAG = "<sep>"


def _load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def model0_control_tags() -> List[str]:
    cfg = _load_module("model0_config", os.path.join(REPO, "model", "0", "config.py"))
    return list(cfg.CONTROL_TAGS)


def model0_train_tokenizer():
    tok = _load_module("model0_tokenizer", os.path.join(REPO, "model", "0", "tokenizer.py"))
    return tok.train_tokenizer


def control_tags() -> List[str]:
    """Ordered, de-duplicated union of every control tag across models."""
    tags: List[str] = [SEP_TAG]
    tags += model0_control_tags()
    for kind in ("src", "tgt"):
        tags += [f"<{kind}:{lang}>" for lang in MODEL2_LANGS]
    seen, out = set(), []
    for t in tags:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _count_lines(path: str) -> int:
    with open(path, "rb") as fh:
        return sum(1 for _ in fh)


def train(inputs: Iterable[str], out: str, vocab_size: int = DEFAULT_VOCAB) -> dict:
    inputs = [os.path.abspath(p) for p in inputs]
    missing = [p for p in inputs if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"tokenizer corpus missing: {missing}")
    if not inputs:
        raise ValueError("no input corpora")
    out = os.path.abspath(out)
    if not out.endswith(".model"):
        raise ValueError("--out must end in .model")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    tags = control_tags()
    # model/0's trainer takes a single text path; concatenate inputs.
    corpus = out.replace(".model", ".corpus.tmp.txt")
    with open(corpus, "w", encoding="utf-8", newline="\n") as dst:
        for p in inputs:
            with open(p, "r", encoding="utf-8", errors="replace") as src:
                for line in src:
                    dst.write(line.rstrip("\r\n") + "\n")
    try:
        model0_train_tokenizer()(corpus, out[:-len(".model")], vocab_size, tags)
    finally:
        os.remove(corpus)

    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.load(out)
    # A reserved symbol encodes as exactly one piece; SentencePiece may emit
    # the bare whitespace marker "▁" in front of it (add_dummy_prefix).
    bad = [t for t in tags
           if [p for p in sp.encode(t, out_type=str) if p != "▁"] != [t]]
    if bad:
        raise RuntimeError(f"control tags not single tokens: {bad}")

    manifest = {
        "artifact": os.path.basename(out),
        "vocab_size": sp.get_piece_size(),
        "requested_vocab_size": vocab_size,
        "special_ids": {"pad": sp.pad_id(), "unk": sp.unk_id(),
                        "bos": sp.bos_id(), "eos": sp.eos_id()},
        "control_tags": tags,
        "inputs": [{"path": os.path.relpath(p, REPO).replace("\\", "/"),
                    "lines": _count_lines(p), "sha256": _sha256(p)} for p in inputs],
        "sha256": _sha256(out),
        "command": "python script/train_tokenizer.py --data "
                   + " ".join(os.path.relpath(p, REPO).replace("\\", "/") for p in inputs)
                   + f" --out {os.path.relpath(out, REPO).replace(chr(92), '/')}"
                   + f" --vocab-size {vocab_size}",
    }
    with open(os.path.join(os.path.dirname(out), "MANIFEST.json"), "w",
              encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", nargs="*", default=None,
                    help="plain-text corpora (default: existing data/model/{0,2}/train_text.txt)")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB)
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing artifact (breaking change: bump the version instead)")
    args = ap.parse_args(argv)

    inputs = args.data if args.data is not None else [p for p in DEFAULT_INPUTS if os.path.exists(p)]
    if not inputs:
        print("no input corpora found; run script/glean_datasets.py first", file=sys.stderr)
        return 2
    if os.path.exists(args.out) and not args.force:
        print(f"{args.out} exists; pinned tokenizers are never regenerated in place. "
              "Use a new --out version dir (or --force if you really mean it).",
              file=sys.stderr)
        return 3
    m = train(inputs, args.out, args.vocab_size)
    print(f"wrote {args.out}: vocab {m['vocab_size']} "
          f"({len(m['control_tags'])} control tags) from "
          f"{sum(i['lines'] for i in m['inputs'])} lines")
    return 0


if __name__ == "__main__":
    sys.exit(main())
