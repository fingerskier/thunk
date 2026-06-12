"""Optimizer construction for Thunk v0 (SPEC.md Section 6.4).

Muon on the hidden 2D weight matrices + AdamW on embeddings, norms, and the
output head (the nanochat / Moonlight split). ``torch.optim.Muon`` ships in
PyTorch 2.12; if it is unavailable, or ``optimizer="adamw"`` is requested, a
single well-tuned AdamW is used as the documented fallback.
"""

from typing import List

import torch


def _split_params(model):
    """Return (muon_params, adamw_params).

    Muon orthogonalizes 2D hidden weight matrices (attention and FFN
    projections). Embeddings (also the tied output head) and all 1D parameters
    (RMSNorm / QK-norm weights) go to AdamW.
    """
    muon, adamw = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_embedding = "tok_emb" in name
        if p.ndim == 2 and not is_embedding:
            muon.append(p)
        else:
            adamw.append(p)
    return muon, adamw


def _adamw_groups(model, cfg):
    """AdamW with decoupled weight decay only on 2D weights (the fallback)."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or "tok_emb" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def build_optimizers(model, cfg) -> List[torch.optim.Optimizer]:
    """Build the optimizer(s) for one training phase.

    Returns a list so the caller can step/zero several optimizers uniformly.
    Each optimizer carries its own base LR; a shared schedule multiplier is
    applied to all of them.
    """
    use_muon = (cfg.optimizer == "muon" and hasattr(torch.optim, "Muon"))
    if not use_muon:
        return [torch.optim.AdamW(_adamw_groups(model, cfg), lr=cfg.lr,
                                  betas=(0.9, 0.95), eps=1e-8)]

    muon_params, adamw_params = _split_params(model)
    muon_kwargs = dict(lr=cfg.muon_lr, weight_decay=cfg.weight_decay,
                       momentum=0.95, nesterov=True)
    # Keller's rectangular-RMS LR adjustment, if this build exposes it.
    try:
        muon = torch.optim.Muon(muon_params, adjust_lr_fn="original", **muon_kwargs)
    except (TypeError, ValueError):
        muon = torch.optim.Muon(muon_params, **muon_kwargs)
    adamw = torch.optim.AdamW(adamw_params, lr=cfg.lr, betas=(0.9, 0.95),
                              eps=1e-8, weight_decay=0.0)
    return [muon, adamw]
