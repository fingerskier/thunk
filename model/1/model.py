"""Small recursive reasoning diffuser (model #1).

The network carries exactly two mutable features, an answer canvas ``y`` and a
latent scratchpad ``z``. Supervision steps double as denoising steps.
"""

import torch
import torch.nn as nn

from config import ReasoningDiffuserConfig


class StableMaxCrossEntropy(nn.Module):
    """Cross entropy with logits clamped before softmax for tiny-run stability."""

    def __init__(self, clamp: float = 30.0):
        super().__init__()
        self.clamp = clamp

    def forward(self, logits, targets):
        return nn.functional.cross_entropy(logits.clamp(-self.clamp, self.clamp), targets)


class ReasoningBlock(nn.Module):
    def __init__(self, cfg: ReasoningDiffuserConfig):
        super().__init__()
        self.z_to_y = nn.MultiheadAttention(cfg.d_model, cfg.n_heads, cfg.dropout, batch_first=True)
        self.y_to_z = nn.MultiheadAttention(cfg.d_model, cfg.n_heads, cfg.dropout, batch_first=True)
        self.x_to_z = nn.MultiheadAttention(cfg.d_model, cfg.n_heads, cfg.dropout, batch_first=True)
        self.norm_y1 = nn.LayerNorm(cfg.d_model)
        self.norm_y2 = nn.LayerNorm(cfg.d_model)
        self.norm_z1 = nn.LayerNorm(cfg.d_model)
        self.norm_z2 = nn.LayerNorm(cfg.d_model)
        self.ff_y = nn.Sequential(nn.Linear(cfg.d_model, cfg.d_ff), nn.GELU(), nn.Linear(cfg.d_ff, cfg.d_model))
        self.ff_z = nn.Sequential(nn.Linear(cfg.d_model, cfg.d_ff), nn.GELU(), nn.Linear(cfg.d_ff, cfg.d_model))

    def update_z(self, x, y, z, x_mask=None):
        zy, _ = self.y_to_z(self.norm_z1(z), y, y, need_weights=False)
        zx, _ = self.x_to_z(self.norm_z1(z), x, x, key_padding_mask=(~x_mask if x_mask is not None else None), need_weights=False)
        z = z + zy + zx
        z = z + self.ff_z(self.norm_z2(z))
        return z

    def update_y(self, y, z):
        yz, _ = self.z_to_y(self.norm_y1(y), z, z, need_weights=False)
        y = y + yz
        y = y + self.ff_y(self.norm_y2(y))
        return y


class ReasoningDiffuser(nn.Module):
    def __init__(self, cfg: ReasoningDiffuserConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_x = nn.Parameter(torch.zeros(1, cfg.max_question_len, cfg.d_model))
        self.pos_y = nn.Parameter(torch.zeros(1, cfg.answer_len, cfg.d_model))
        self.z_init = nn.Parameter(torch.zeros(1, cfg.answer_len, cfg.d_model))
        self.blocks = nn.ModuleList(ReasoningBlock(cfg) for _ in range(cfg.recursion_depth))
        self.out_norm = nn.LayerNorm(cfg.d_model)
        self.output_head = nn.Linear(cfg.d_model, cfg.vocab_size)
        self.halt_head = nn.Linear(cfg.d_model, 1)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if getattr(module, 'bias', None) is not None:
                nn.init.zeros_(module.bias)

    def embed_question(self, x_ids):
        return self.token_emb(x_ids) + self.pos_x[:, :x_ids.size(1)]

    def embed_answer(self, y_ids):
        return self.token_emb(y_ids) + self.pos_y[:, :y_ids.size(1)]

    def initial_state(self, batch_size, device):
        return self.z_init.expand(batch_size, -1, -1).to(device)

    def latent_recursion(self, x, y_ids, z, x_mask=None):
        y = self.embed_answer(y_ids)
        for block in self.blocks:
            z = block.update_z(x, y, z, x_mask)
            y = block.update_y(y, z)
        logits = self.output_head(self.out_norm(y))
        halt_logit = self.halt_head(y.mean(dim=1)).squeeze(-1)
        return logits, y, z, halt_logit

    def forward(self, x_ids, y_ids, z=None, x_mask=None):
        x = self.embed_question(x_ids)
        if z is None:
            z = self.initial_state(x_ids.size(0), x_ids.device)
        return self.latent_recursion(x, y_ids, z, x_mask)
