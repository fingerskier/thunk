"""Thunk v0 baseline translator.

Deep-encoder / shallow-decoder encoder-decoder Transformer following SPEC.md
Sections 2.2 and 3:

  * RMSNorm pre-norm everywhere; QK-norm (RMSNorm on per-head queries/keys).
  * RoPE on self-attention only (encoder + decoder), base configurable; no RoPE
    on cross-attention.
  * Encoder self-attention: full MHA, bidirectional.
  * Decoder self-attention: causal grouped-query attention (n_heads:kv_heads).
  * Cross-attention: full MHA over encoder memory.
  * SwiGLU feed-forward, no biases on any linear projection.
  * One shared token embedding for encoder input, decoder input, and the LM head.

The encoder also exposes a mean-pooled embedding (SPEC.md Section 8) so the same
model can serve semantic search / memory, not only generation.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ThunkConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


def build_rope_cache(head_dim: int, max_seq_len: int, base: float, device=None):
    """Precompute cos/sin tables of shape (max_seq_len, head_dim // 2)."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, inv_freq)
    return freqs.cos(), freqs.sin()


def apply_rotary(x, cos, sin):
    """x: (B, H, T, head_dim); cos/sin: (T, head_dim // 2)."""
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class MultiHeadAttention(nn.Module):
    """Configurable attention block.

    Supports self/cross attention, GQA (n_kv_heads < n_heads), optional RoPE,
    optional causal masking, and QK-norm. No biases on any projection.
    """

    def __init__(self, cfg: ThunkConfig, n_kv_heads: int, use_rope: bool,
                 causal: bool):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = cfg.head_dim
        self.use_rope = use_rope
        self.causal = causal
        self.dropout_p = cfg.dropout

        q_dim = self.n_heads * self.head_dim
        kv_dim = self.n_kv_heads * self.head_dim
        self.q_proj = nn.Linear(cfg.d_model, q_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, kv_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, kv_dim, bias=False)
        self.o_proj = nn.Linear(q_dim, cfg.d_model, bias=False)

        # QK-norm (SPEC.md Section 3.3): RMSNorm on queries and keys per head.
        self.q_norm = RMSNorm(self.head_dim, cfg.rms_eps)
        self.k_norm = RMSNorm(self.head_dim, cfg.rms_eps)

    def forward(self, x, kv_source=None, attn_mask=None, rope=None):
        """x: (B, Tq, C). kv_source: (B, Tk, C) for cross-attn (else self).

        attn_mask: additive float mask broadcastable to (B, n_heads, Tq, Tk).
        rope: (cos, sin) tables sliced to the relevant sequence length.
        """
        if kv_source is None:
            kv_source = x
        B, Tq, _ = x.shape
        Tk = kv_source.shape[1]

        q = self.q_proj(x).view(B, Tq, self.n_heads, self.head_dim)
        k = self.k_proj(kv_source).view(B, Tk, self.n_kv_heads, self.head_dim)
        v = self.v_proj(kv_source).view(B, Tk, self.n_kv_heads, self.head_dim)

        q = self.q_norm(q).transpose(1, 2)   # (B, n_heads, Tq, hd)
        k = self.k_norm(k).transpose(1, 2)   # (B, n_kv_heads, Tk, hd)
        v = v.transpose(1, 2)

        if self.use_rope and rope is not None:
            cos_q, sin_q = rope
            q = apply_rotary(q, cos_q[:Tq], sin_q[:Tq])
            k = apply_rotary(k, cos_q[:Tk], sin_q[:Tk])

        # Expand KV heads to match query heads for GQA (version-safe).
        if self.n_kv_heads != self.n_heads:
            rep = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        dropout_p = self.dropout_p if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=self.causal and attn_mask is None,
        )
        out = out.transpose(1, 2).reshape(B, Tq, self.n_heads * self.head_dim)
        return self.o_proj(out)


class SwiGLU(nn.Module):
    def __init__(self, cfg: ThunkConfig):
        super().__init__()
        self.w1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w3 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)  # gate
        self.w2 = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class EncoderBlock(nn.Module):
    def __init__(self, cfg: ThunkConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.attn = MultiHeadAttention(
            cfg, n_kv_heads=cfg.n_heads, use_rope=True, causal=False)
        self.ff_norm = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.ff = SwiGLU(cfg)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, attn_mask, rope):
        x = x + self.drop(self.attn(self.attn_norm(x), attn_mask=attn_mask, rope=rope))
        x = x + self.drop(self.ff(self.ff_norm(x)))
        return x


class DecoderBlock(nn.Module):
    def __init__(self, cfg: ThunkConfig):
        super().__init__()
        self.self_norm = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.self_attn = MultiHeadAttention(
            cfg, n_kv_heads=cfg.decoder_kv_heads, use_rope=True, causal=True)
        self.cross_norm = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.cross_attn = MultiHeadAttention(
            cfg, n_kv_heads=cfg.n_heads, use_rope=False, causal=False)
        self.ff_norm = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.ff = SwiGLU(cfg)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, memory, self_mask, cross_mask, rope):
        x = x + self.drop(self.self_attn(self.self_norm(x), attn_mask=self_mask, rope=rope))
        x = x + self.drop(self.cross_attn(
            self.cross_norm(x), kv_source=memory, attn_mask=cross_mask))
        x = x + self.drop(self.ff(self.ff_norm(x)))
        return x


class Thunk(nn.Module):
    def __init__(self, cfg: ThunkConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

        self.encoder = nn.ModuleList(EncoderBlock(cfg) for _ in range(cfg.encoder_layers))
        self.enc_norm = RMSNorm(cfg.d_model, cfg.rms_eps)

        self.decoder = nn.ModuleList(DecoderBlock(cfg) for _ in range(cfg.decoder_layers))
        self.dec_norm = RMSNorm(cfg.d_model, cfg.rms_eps)

        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        cos, sin = build_rope_cache(cfg.head_dim, cfg.max_seq_len, cfg.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # Scaled init on residual-output projections for depth stability.
        n_layers = cfg.encoder_layers + cfg.decoder_layers
        scale = (2 * n_layers) ** -0.5
        for name, p in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("w2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 * scale)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    # ---- masks ----
    def _padding_bias(self, pad_mask):
        """pad_mask: (B, Tk) bool, True where token is real. -> (B,1,1,Tk) additive."""
        bias = torch.zeros_like(pad_mask, dtype=torch.float32)
        bias = bias.masked_fill(~pad_mask, float("-inf"))
        return bias[:, None, None, :]

    def _causal_bias(self, T, device):
        m = torch.full((T, T), float("-inf"), device=device)
        return torch.triu(m, diagonal=1)[None, None, :, :]

    def encode(self, src_ids, src_mask=None):
        """Run the encoder; returns memory states (B, Ts, C)."""
        rope = (self.rope_cos.to(src_ids.device), self.rope_sin.to(src_ids.device))
        x = self.drop(self.tok_emb(src_ids))
        attn_mask = self._padding_bias(src_mask) if src_mask is not None else None
        for block in self.encoder:
            x = block(x, attn_mask, rope)
        return self.enc_norm(x)

    def embed(self, src_ids, src_mask=None):
        """Mean-pooled encoder embedding (SPEC.md Section 8 memory/search hook)."""
        memory = self.encode(src_ids, src_mask)
        if src_mask is None:
            return memory.mean(dim=1)
        w = src_mask.float().unsqueeze(-1)
        return (memory * w).sum(dim=1) / w.sum(dim=1).clamp_min(1.0)

    def decode(self, tgt_ids, memory, tgt_mask=None, src_mask=None):
        rope = (self.rope_cos.to(tgt_ids.device), self.rope_sin.to(tgt_ids.device))
        x = self.drop(self.tok_emb(tgt_ids))
        T = tgt_ids.shape[1]
        self_mask = self._causal_bias(T, tgt_ids.device)
        if tgt_mask is not None:
            self_mask = self_mask + self._padding_bias(tgt_mask)
        cross_mask = self._padding_bias(src_mask) if src_mask is not None else None
        for block in self.decoder:
            x = block(x, memory, self_mask, cross_mask, rope)
        return self.dec_norm(x)

    def forward(self, src_ids, tgt_ids, src_mask=None, tgt_mask=None, labels=None):
        memory = self.encode(src_ids, src_mask)
        h = self.decode(tgt_ids, memory, tgt_mask, src_mask)
        logits = self.lm_head(h)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=PAD_ID,
                label_smoothing=self.cfg.label_smoothing,
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, src_ids, src_mask=None, bos_id=2, eos_id=3,
                 max_new_tokens=64, beam_size=1):
        """Greedy (beam_size=1) or beam-search decoding for a single example.

        src_ids: (1, Ts). Returns a list of generated target token ids
        (excluding BOS, up to and excluding EOS).
        """
        self.eval()
        assert src_ids.shape[0] == 1, "generate handles one example at a time"
        device = src_ids.device
        memory = self.encode(src_ids, src_mask)

        if beam_size == 1:
            ys = torch.tensor([[bos_id]], device=device)
            for _ in range(max_new_tokens):
                h = self.decode(ys, memory, src_mask=src_mask)
                next_id = self.lm_head(h[:, -1]).argmax(-1).item()
                if next_id == eos_id:
                    break
                ys = torch.cat([ys, torch.tensor([[next_id]], device=device)], dim=1)
            return ys[0, 1:].tolist()

        # Beam search.
        beams = [([bos_id], 0.0)]
        finished = []
        for _ in range(max_new_tokens):
            candidates = []
            for seq, score in beams:
                if seq[-1] == eos_id:
                    finished.append((seq, score))
                    continue
                ys = torch.tensor([seq], device=device)
                h = self.decode(ys, memory, src_mask=src_mask)
                logp = F.log_softmax(self.lm_head(h[:, -1]), dim=-1)[0]
                topv, topi = logp.topk(beam_size)
                for v, i in zip(topv.tolist(), topi.tolist()):
                    candidates.append((seq + [i], score + v))
            if not candidates:
                break
            candidates.sort(key=lambda x: x[1] / max(len(x[0]) - 1, 1), reverse=True)
            beams = candidates[:beam_size]
            if all(seq[-1] == eos_id for seq, _ in beams):
                finished.extend(beams)
                break
        finished = finished or beams
        best = max(finished, key=lambda x: x[1] / max(len(x[0]) - 1, 1))[0]
        out = best[1:]
        if eos_id in out:
            out = out[:out.index(eos_id)]
        return out

    def param_count(self, trainable_only=True):
        ps = (p for p in self.parameters() if p.requires_grad or not trainable_only)
        return sum(p.numel() for p in ps)


# PAD token id; kept in sync with the tokenizer (pad_id=0).
PAD_ID = 0


if __name__ == "__main__":
    from config import reference_config, small_config

    for name, cfg in [("reference", reference_config()), ("small", small_config())]:
        model = Thunk(cfg)
        emb = cfg.vocab_size * cfg.d_model
        print(f"[{name}] total params: {model.param_count():,} "
              f"(embedding/tied head: {emb:,})")

    # forward + generate smoke test on the small config
    cfg = small_config(vocab_size=512)
    model = Thunk(cfg)
    src = torch.randint(4, cfg.vocab_size, (2, 10))
    tgt = torch.randint(4, cfg.vocab_size, (2, 8))
    src_mask = torch.ones(2, 10, dtype=torch.bool)
    tgt_mask = torch.ones(2, 8, dtype=torch.bool)
    logits, loss = model(src, tgt, src_mask, tgt_mask, labels=tgt)
    print("forward logits:", tuple(logits.shape), "loss:", round(loss.item(), 3))
    out = model.generate(src[:1], src_mask[:1], max_new_tokens=12)
    print("greedy generate length:", len(out))
    print("encoder embedding:", tuple(model.embed(src, src_mask).shape))
