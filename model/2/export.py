"""Export model #2 to a packed low-bit artifact (Phase 3, packing step).

Only here do the bits leave the floats: every ``BitLinear`` master weight is
quantized once more and packed 4 ternary weights per byte alongside its FP32
scale; embeddings and any remaining FP linears are stored as per-row int8;
norms and the loop-state init stay FP16. The result is a single
``torch.save`` archive whose size is reported per section.

``--verify`` reloads the artifact into a fresh model and checks the logits
against the QAT model on a probe batch — ternary layers round-trip exactly
(the QAT forward re-quantizes to the same values), int8 embeddings add a
bounded delta.

A bitnet.cpp-style CPU lookup-table kernel is the remaining Phase 3 work;
this file defines the interchange format for it.
"""

import argparse
import os

import torch

from config import RecurrentLMConfig
from model import RecurrentLM
from quant import (BitLinear, pack_ternary, quantize_embedding_int8,
                   quantize_weights_ternary, unpack_ternary)


def export_model(model: RecurrentLM, cfg: RecurrentLMConfig):
    """Build a {name: tensor} payload + per-section byte counts."""
    payload = {"__config__": dict(cfg.__dict__)}
    sizes = {"ternary": 0, "int8": 0, "fp16": 0}
    for name, module in model.named_modules():
        if isinstance(module, BitLinear):
            _, q, gamma = quantize_weights_ternary(module.weight, module.scaling)
            packed = pack_ternary(q)
            payload[f"ternary.{name}.packed"] = packed
            payload[f"ternary.{name}.scale"] = gamma.float()
            payload[f"ternary.{name}.shape"] = torch.tensor(q.shape)
            sizes["ternary"] += packed.numel() + 4
        elif isinstance(module, torch.nn.Linear):
            if cfg.tie_embeddings and name == "lm_head":
                continue   # served by the int8 embedding
            q, scale = quantize_embedding_int8(module.weight)
            payload[f"int8.{name}.q"] = q
            payload[f"int8.{name}.scale"] = scale.float()
            sizes["int8"] += q.numel() + scale.numel() * 4
        elif isinstance(module, torch.nn.Embedding):
            q, scale = quantize_embedding_int8(module.weight)
            payload[f"int8.{name}.q"] = q
            payload[f"int8.{name}.scale"] = scale.float()
            sizes["int8"] += q.numel() + scale.numel() * 4
    # Norm gains (block norms, QK-norms, BitLinear pre-norms, out_norm) and
    # the loop-state init stay in higher precision, per the b1.58 recipe.
    for name, param in model.named_parameters():
        if "norm" in name or name == "s_init":
            payload[f"fp16.{name}"] = param.detach().half()
            sizes["fp16"] += param.numel() * 2
    return payload, sizes


def restore_model(payload, device: str = "cpu") -> RecurrentLM:
    """Rebuild a model whose BitLinear masters are the dequantized ternary
    weights — its QAT forward re-derives the identical ternary values."""
    cfg = RecurrentLMConfig(**payload["__config__"])
    model = RecurrentLM(cfg).to(device)
    state = model.state_dict()
    for name, module in model.named_modules():
        if isinstance(module, BitLinear):
            shape = tuple(payload[f"ternary.{name}.shape"].tolist())
            q = unpack_ternary(payload[f"ternary.{name}.packed"], shape).float()
            state[f"{name}.weight"] = q * payload[f"ternary.{name}.scale"]
        elif isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            key = f"int8.{name}.q"
            if key in payload:
                w = payload[key].float() * payload[f"int8.{name}.scale"].unsqueeze(-1)
                state[f"{name}.weight"] = w
    if cfg.tie_embeddings and "int8.tok_emb.q" in payload:
        state["lm_head.weight"] = state["tok_emb.weight"]
    for key, value in payload.items():
        if key.startswith("fp16."):
            state_key = key[len("fp16."):]
            if state_key in state:
                state[state_key] = value.float()
    model.load_state_dict(state)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", default="export/model2.pack.pt")
    parser.add_argument("--verify", action="store_true",
                        help="reload the artifact and compare logits against "
                             "the QAT model on a probe batch")
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    cfg = RecurrentLMConfig(**ckpt["config"])
    model = RecurrentLM(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()

    if not cfg.quantize:
        print("note: checkpoint is an FP config — no BitLinear layers to pack; "
              "linears and embeddings export as int8 only.")

    payload, sizes = export_model(model, cfg)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(payload, args.out)
    disk = os.path.getsize(args.out)
    total = sum(sizes.values())
    print(f"packed sections: ternary {sizes['ternary']:,} B | "
          f"int8 {sizes['int8']:,} B | fp16 {sizes['fp16']:,} B "
          f"-> payload {total / 1e6:.2f} MB (file {disk / 1e6:.2f} MB)")

    if args.verify:
        restored = restore_model(payload)
        torch.manual_seed(0)
        x = torch.randint(0, cfg.vocab_size, (2, min(32, cfg.max_seq_len)))
        with torch.no_grad():
            ref, _ = model(x, loops=2)
            got, _ = restored(x, loops=2)
        err = (ref - got).abs().max().item()
        print(f"verify: max |logit delta| after round-trip = {err:.4f}")


if __name__ == "__main__":
    main()
