"""Run translation with a trained Thunk v0 checkpoint (SPEC.md Section 9).

Usage:
  python translate.py --src digits --tgt english "3247"
  python translate.py --src english --tgt digits "three thousand two hundred forty-seven"
  python translate.py --src bash --tgt powershell "rm report.txt"
  python translate.py --beam 4 --src python --tgt english "x = 3 + 4"
"""

import argparse

import torch

from model import Thunk
from tokenizer import Tokenizer


def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = Thunk(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


def translate(model, tok, src_lang, tgt_lang, text, beam=1, max_new_tokens=64):
    prompt = f"<src:{src_lang}> <tgt:{tgt_lang}> {text}"
    src_ids = torch.tensor([tok.encode(prompt)], dtype=torch.long)
    src_mask = torch.ones_like(src_ids, dtype=torch.bool)
    out = model.generate(src_ids, src_mask, bos_id=tok.bos_id, eos_id=tok.eos_id,
                         max_new_tokens=max_new_tokens, beam_size=beam)
    return tok.decode(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text")
    ap.add_argument("--src", required=True)
    ap.add_argument("--tgt", required=True)
    ap.add_argument("--ckpt", default="checkpoints/best.pt")
    ap.add_argument("--tokenizer", default="tokenizer.model")
    ap.add_argument("--beam", type=int, default=1)
    args = ap.parse_args()

    model, _ = load_model(args.ckpt)
    tok = Tokenizer(args.tokenizer)
    print(translate(model, tok, args.src, args.tgt, args.text, beam=args.beam))


if __name__ == "__main__":
    main()
