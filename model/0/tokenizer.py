"""Shared SentencePiece tokenizer for Thunk v0 (SPEC.md Section 5).

One shared source/target vocabulary with byte fallback, digit splitting, and the
SPEC.md Section 4 control tags reserved as single user-defined symbols (never
split into subwords). Special ids are fixed: pad=0, unk=1, bos=2, eos=3.
"""

import os
from typing import List

import sentencepiece as spm

PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3


def train_tokenizer(text_path: str, model_prefix: str, vocab_size: int,
                    control_tags: List[str], character_coverage: float = 1.0):
    """Train a SentencePiece BPE model with reserved control tags."""
    spm.SentencePieceTrainer.train(
        input=text_path,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        pad_id=PAD_ID,
        unk_id=UNK_ID,
        bos_id=BOS_ID,
        eos_id=EOS_ID,
        user_defined_symbols=control_tags,
        byte_fallback=True,
        split_digits=True,
        # small curated corpora may not support the full requested vocab;
        # take as many pieces as the data supports rather than erroring.
        hard_vocab_limit=False,
        character_coverage=character_coverage,
        normalization_rule_name="identity",  # preserve code whitespace exactly
        remove_extra_whitespaces=False,
        num_threads=os.cpu_count() or 1,
    )


class Tokenizer:
    """Thin wrapper exposing encode/decode and the fixed special ids."""

    def __init__(self, model_path: str):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
        self.pad_id = PAD_ID
        self.unk_id = UNK_ID
        self.bos_id = BOS_ID
        self.eos_id = EOS_ID

    @property
    def vocab_size(self) -> int:
        return self.sp.get_piece_size()

    def encode(self, text: str) -> List[int]:
        return self.sp.encode(text, out_type=int)

    def decode(self, ids: List[int]) -> str:
        # drop special tokens before detokenizing
        ids = [i for i in ids if i not in (self.pad_id, self.bos_id, self.eos_id)]
        return self.sp.decode(ids)

    def piece(self, idx: int) -> str:
        return self.sp.id_to_piece(idx)
