"""Text data helpers for model #1 train/run scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch.utils.data import Dataset


SPECIAL_IDS = {
    "mask": 0,
    "pad": 1,
    "bos": 2,
    "eos": 3,
    "unk": 4,
}


class CharacterTokenizer:
    """Small character-level tokenizer for fixed-canvas smoke training."""

    def __init__(self, chars: Sequence[str]):
        self.chars = list(chars)
        self.char_to_id = {char: idx + len(SPECIAL_IDS) for idx, char in enumerate(self.chars)}
        self.id_to_char = {idx: char for char, idx in self.char_to_id.items()}

    @classmethod
    def build(cls, records: Iterable[dict[str, str]]) -> "CharacterTokenizer":
        chars: set[str] = set()
        for record in records:
            chars.update(record["question"])
            chars.update(record["answer"])
        return cls(sorted(chars))

    @classmethod
    def from_metadata(cls, metadata: dict) -> "CharacterTokenizer":
        if metadata.get("type") != "character":
            raise ValueError(f"unsupported tokenizer type: {metadata.get('type')!r}")
        return cls(metadata["chars"])

    def to_metadata(self) -> dict:
        return {
            "type": "character",
            "chars": self.chars,
            "special_ids": SPECIAL_IDS,
        }

    @property
    def vocab_size(self) -> int:
        return len(SPECIAL_IDS) + len(self.chars)

    def encode(self, text: str, max_len: int) -> list[int]:
        if max_len < 2:
            raise ValueError("max_len must be at least 2 to hold BOS/EOS")
        ids = [SPECIAL_IDS["bos"]]
        ids.extend(self.char_to_id.get(char, SPECIAL_IDS["unk"]) for char in text)
        ids.append(SPECIAL_IDS["eos"])
        if len(ids) > max_len:
            ids = ids[:max_len]
            ids[-1] = SPECIAL_IDS["eos"]
        ids.extend([SPECIAL_IDS["pad"]] * (max_len - len(ids)))
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        chars: list[str] = []
        for raw_id in ids:
            token_id = int(raw_id)
            if token_id == SPECIAL_IDS["eos"]:
                break
            if token_id in (SPECIAL_IDS["mask"], SPECIAL_IDS["pad"], SPECIAL_IDS["bos"]):
                continue
            chars.append(self.id_to_char.get(token_id, "?"))
        return "".join(chars).strip()


def load_records(path: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            question = str(item.get("question", "")).strip()
            answer = str(item.get("answer", "")).strip()
            if not question or not answer:
                continue
            records.append({"question": question, "answer": answer})
    if not records:
        raise ValueError(f"no usable question/answer records found in {path}")
    return records


class QADataset(Dataset):
    def __init__(self, records: Sequence[dict[str, str]], tokenizer: CharacterTokenizer, cfg):
        self.records = list(records)
        self.tokenizer = tokenizer
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        question = torch.tensor(
            self.tokenizer.encode(record["question"], self.cfg.max_question_len),
            dtype=torch.long,
        )
        answer = torch.tensor(
            self.tokenizer.encode(record["answer"], self.cfg.answer_len),
            dtype=torch.long,
        )
        return {
            "question": question,
            "question_mask": question.ne(self.cfg.pad_id),
            "answer": answer,
        }


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: tensor.to(device) for name, tensor in batch.items()}


@torch.no_grad()
def generate_answer_ids(model, cfg, question_ids, question_mask, steps: int | None = None, halt_threshold: float = 0.5):
    model.eval()
    if question_ids.dim() == 1:
        question_ids = question_ids.unsqueeze(0)
    if question_mask.dim() == 1:
        question_mask = question_mask.unsqueeze(0)
    y = torch.full(
        (question_ids.size(0), cfg.answer_len),
        cfg.mask_id,
        dtype=torch.long,
        device=question_ids.device,
    )
    z = model.initial_state(question_ids.size(0), question_ids.device)
    final_halt = torch.zeros(question_ids.size(0), device=question_ids.device)
    used_steps = 0
    for used_steps in range(1, (steps or cfg.supervision_steps) + 1):
        logits, _, z, halt_logit = model(question_ids, y, z, question_mask)
        y = logits.argmax(dim=-1)
        final_halt = halt_logit.sigmoid()
        if bool((final_halt > halt_threshold).all()):
            break
    return y.detach().cpu(), used_steps, final_halt.detach().cpu()
