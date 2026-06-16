"""Masked discrete-diffusion utilities for model #1."""

import math
import torch
import torch.nn.functional as F


def cosine_mask_probability(step: int, total_steps: int) -> float:
    """Monotone schedule from almost fully masked to unmasked."""
    if total_steps <= 1:
        return 0.0
    progress = step / (total_steps - 1)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def make_mask(labels: torch.Tensor, mask_prob: float, mask_id: int, pad_id: int, generator=None):
    valid = labels.ne(pad_id)
    rand = torch.rand(labels.shape, device=labels.device, generator=generator)
    masked = valid & (rand < mask_prob)
    if valid.any() and not masked.any():
        first = valid.flatten().nonzero(as_tuple=False)[0, 0]
        masked.flatten()[first] = True
    canvas = labels.masked_fill(masked, mask_id)
    return canvas, masked


def masked_diffusion_ce(logits, labels, masked_positions, pad_id: int, label_smoothing: float = 0.0):
    train_positions = masked_positions & labels.ne(pad_id)
    if not train_positions.any():
        return logits.sum() * 0.0
    return F.cross_entropy(
        logits[train_positions],
        labels[train_positions],
        label_smoothing=label_smoothing,
    )


def remask_from_logits(logits, labels, current_mask, next_mask_prob: float, mask_id: int, pad_id: int):
    """Fill current masks with argmax predictions, then remask low-confidence tokens."""
    probs = logits.softmax(dim=-1)
    conf, pred = probs.max(dim=-1)
    y = labels.clone()
    y[current_mask] = pred[current_mask]
    valid = labels.ne(pad_id)
    if next_mask_prob <= 0:
        return y.masked_fill(~valid, pad_id)
    n_valid = valid.sum(dim=1)
    for row in range(y.size(0)):
        budget = int(torch.ceil(n_valid[row].float() * next_mask_prob).item())
        if budget <= 0:
            continue
        row_valid = valid[row]
        idx = torch.argsort(conf[row].masked_fill(~row_valid, 2.0))[:budget]
        y[row, idx] = mask_id
    return y.masked_fill(~valid, pad_id)
