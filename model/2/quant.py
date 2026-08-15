"""Ternary (b1.58) quantization-aware training primitives for model #2.

"Bits stored in the floats", made precise (PLAN.md Evaluation B): training
keeps latent FP master weights; every forward pass quantizes them on the fly
(absmean/median ternary weights, per-token absmax int8 activations) and the
backward pass uses the straight-through estimator. The bits have no
independent existence during training and there is no training-time saving —
the payoff is at export, where ``pack_ternary`` stores four weights per byte.

Recipe details follow the BitNet b1.58 line: an extra RMSNorm immediately
before every quantized linear, no biases anywhere, and (per BitNet b1.58
Reloaded, arXiv:2407.09527) a *median*-based weight scale for small models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


def ternary_weight_scale(w: torch.Tensor, scaling: str) -> torch.Tensor:
    """Per-tensor scale gamma: mean(|W|) for b1.58, median(|W|) for Reloaded."""
    a = w.abs()
    gamma = a.median() if scaling == "median" else a.mean()
    return gamma.clamp(min=1e-8)


def quantize_weights_ternary(w: torch.Tensor, scaling: str = "median"):
    """RoundClip(W/gamma, -1, 1). Returns (dequantized FP, int {-1,0,1}, gamma)."""
    gamma = ternary_weight_scale(w, scaling)
    q = (w / gamma).round().clamp_(-1, 1)
    return q * gamma, q.to(torch.int8), gamma


def quantize_activations_int8(x: torch.Tensor) -> torch.Tensor:
    """Per-token absmax int8 fake-quantization (dequantized back to FP)."""
    scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0
    return (x / scale).round().clamp_(-128, 127) * scale


class BitLinear(nn.Module):
    """b1.58 linear: pre-RMSNorm -> int8 activations -> ternary weights.

    Both quantizers run in the forward pass with STE backward
    (``x + (q(x) - x).detach()``), so the optimizer sees FP gradients on the
    latent master ``weight``. No bias, per the b1.58 stability recipe.
    """

    def __init__(self, in_features: int, out_features: int,
                 scaling: str = "median", eps: float = 1e-6):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scaling = scaling
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.norm = RMSNorm(in_features, eps)
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x):
        x = self.norm(x)
        x = x + (quantize_activations_int8(x) - x).detach()
        w_deq, _, _ = quantize_weights_ternary(self.weight, self.scaling)
        w = self.weight + (w_deq - self.weight).detach()
        return F.linear(x, w)

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}, scaling={self.scaling}"


def make_linear(in_features: int, out_features: int, quantized: bool,
                scaling: str = "median", eps: float = 1e-6) -> nn.Module:
    if quantized:
        return BitLinear(in_features, out_features, scaling, eps)
    return nn.Linear(in_features, out_features, bias=False)


# ---------------------------------------------------------------------------
# Export packing: 2 bits per ternary weight, 4 weights per byte.
# ---------------------------------------------------------------------------

def pack_ternary(q: torch.Tensor) -> torch.Tensor:
    """Pack an int tensor of {-1, 0, +1} into uint8, 4 values per byte."""
    flat = (q.flatten().to(torch.int16) + 1).to(torch.uint8)   # {0, 1, 2}
    pad = (-flat.numel()) % 4
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    flat = flat.view(-1, 4)
    return flat[:, 0] | (flat[:, 1] << 2) | (flat[:, 2] << 4) | (flat[:, 3] << 6)


def unpack_ternary(packed: torch.Tensor, shape) -> torch.Tensor:
    """Inverse of :func:`pack_ternary`; returns int8 {-1, 0, +1} of ``shape``."""
    parts = [((packed >> s) & 0x3).to(torch.int8) - 1 for s in (0, 2, 4, 6)]
    flat = torch.stack(parts, dim=1).flatten()
    numel = 1
    for d in shape:
        numel *= d
    return flat[:numel].view(shape)


def quantize_embedding_int8(w: torch.Tensor):
    """Per-row absmax int8 for embeddings / untied heads (export only)."""
    scale = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0
    q = (w / scale).round().clamp_(-128, 127).to(torch.int8)
    return q, scale.squeeze(-1)
