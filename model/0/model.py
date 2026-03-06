import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ThunkConfig


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=256):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos_cached", freqs.cos())
        self.register_buffer("sin_cached", freqs.sin())

    def forward(self, seq_len):
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def apply_rotary(x, cos, sin):
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class Attention(nn.Module):
    def __init__(self, cfg: ThunkConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.rotary = RotaryEmbedding(self.head_dim, cfg.max_seq_len)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, mask=None):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each: (B, T, n_heads, head_dim)
        q = q.transpose(1, 2)  # (B, n_heads, T, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        cos, sin = self.rotary(T)
        cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, head_dim//2)
        sin = sin.unsqueeze(0).unsqueeze(0)
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out(out)


class FeedForward(nn.Module):
    def __init__(self, cfg: ThunkConfig):
        super().__init__()
        self.w1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w2 = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)
        self.w3 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)  # SwiGLU gate

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ThunkConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.ff_norm = RMSNorm(cfg.d_model)
        self.ff = FeedForward(cfg)

    def forward(self, x, mask=None):
        x = x + self.attn(self.attn_norm(x), mask)
        x = x + self.ff(self.ff_norm(x))
        return x


class Thunk(nn.Module):
    def __init__(self, cfg: ThunkConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # tie weights
        self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward_once(self, x, mask=None):
        """Single forward pass through all transformer blocks."""
        for block in self.blocks:
            x = block(x, mask)
        return x

    def forward(self, token_ids, targets=None):
        B, T = token_ids.shape
        mask = torch.tril(torch.ones(T, T, device=token_ids.device)).unsqueeze(0).unsqueeze(0)

        x = self.drop(self.tok_emb(token_ids))

        # recursive pass with stability detection
        recurse_count = 0
        for _ in range(self.cfg.max_recurse):
            x_prev = x
            x = self.forward_once(x, mask)
            recurse_count += 1

            # check stability via cosine similarity
            if recurse_count >= 2:
                cos_sim = F.cosine_similarity(
                    x_prev.detach().reshape(-1, self.cfg.d_model),
                    x.detach().reshape(-1, self.cfg.d_model),
                    dim=-1,
                ).mean()
                if cos_sim > self.cfg.stability_threshold:
                    break

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss, recurse_count

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    cfg = ThunkConfig()
    model = Thunk(cfg)
    print(f"Parameters: {model.param_count():,}")
    x = torch.randint(0, cfg.vocab_size, (2, 64))
    logits, loss, depth = model(x, x)
    print(f"Logits: {logits.shape}, Loss: {loss:.4f}, Recursion depth: {depth}")
