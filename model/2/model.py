"""Model #2: auto-halting depth-recurrent language model (Huginn-style).

Decoder-only causal LM with three sections (PLAN.md reference configuration):

  * prelude  -- ``n_prelude`` blocks embedding tokens into latent space.
  * core     -- ``n_core`` blocks, **genuinely weight-tied**: the same blocks
    are applied r times, each iteration re-injecting the prelude output
    through a linear adapter (recurrent depth = test-time compute).
  * coda     -- ``n_coda`` blocks decoding the converged state into logits.

Conventions shared with model/0: RMSNorm pre-norm, QK-norm, RoPE on
self-attention, SwiGLU feed-forward (ReLU^2 in the ternary phase, per the
b1.58 recipe), no biases anywhere, one tied token embedding.

Training runs full sequences at a sampled loop count with truncated backprop
through the last ``bptt_k`` iterations. Inference (see ``generate``) decodes
token by token, loops the core until the KL between successive output
distributions falls under ``kl_threshold``, and keeps the loop's KV cache to a
fixed ``kv_budget`` of slots where iteration i writes slot ``i % k`` (Huginn).
"""

import math
from contextlib import nullcontext
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import RecurrentLMConfig
from quant import RMSNorm, make_linear


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def stablemax_cross_entropy(logits, targets, ignore_index: int = -100,
                            label_smoothing: float = 0.0):
    """Cross entropy under the *actual* stablemax transform (Prieto et al.,
    arXiv:2501.04697): s(x) = x + 1 for x >= 0, 1 / (1 - x) for x < 0,
    normalized instead of exp-softmax. Computed in log space:
    log s(x) = log1p(x) for x >= 0, -log1p(-x) for x < 0.

    This replaces the clamped exp-softmax that model/1 shipped under the same
    name (PLAN.md "Carry-over defects", item 1).
    """
    x = logits.float()
    log_s = torch.where(x >= 0, torch.log1p(x.clamp(min=0)),
                        -torch.log1p((-x).clamp(min=0)))
    logp = log_s - log_s.logsumexp(dim=-1, keepdim=True)
    logp = logp.view(-1, logp.size(-1))
    targets = targets.reshape(-1)
    if label_smoothing > 0.0:
        n = logp.size(-1)
        keep = targets != ignore_index
        smooth = -logp[keep].mean(dim=-1)
        nll = F.nll_loss(logp, targets, ignore_index=ignore_index)
        return (1 - label_smoothing) * nll + label_smoothing * smooth.mean()
    return F.nll_loss(logp, targets, ignore_index=ignore_index)


# ---------------------------------------------------------------------------
# Rotary embeddings (mirrors model/0)
# ---------------------------------------------------------------------------

def build_rope_cache(head_dim: int, max_seq_len: int, base: float):
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


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """Causal MHA with QK-norm and RoPE; supports cached single-step decode."""

    def __init__(self, cfg: RecurrentLMConfig, quantized: bool):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.dropout_p = cfg.dropout
        q_dim = cfg.n_heads * cfg.head_dim
        lin = lambda i, o: make_linear(i, o, quantized, cfg.quant_scaling, cfg.rms_eps)
        self.q_proj = lin(cfg.d_model, q_dim)
        self.k_proj = lin(cfg.d_model, q_dim)
        self.v_proj = lin(cfg.d_model, q_dim)
        self.o_proj = lin(q_dim, cfg.d_model)
        self.q_norm = RMSNorm(cfg.head_dim, cfg.rms_eps)
        self.k_norm = RMSNorm(cfg.head_dim, cfg.rms_eps)

    def project(self, x, cos, sin):
        """Rotated q/k and v for the positions covered by cos/sin rows."""
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)
        return q, k, v

    def forward(self, x, rope):
        """Full-sequence causal attention (training / fixed-r evaluation)."""
        B, T, _ = x.shape
        cos, sin = rope
        q, k, v = self.project(x, cos[:T], sin[:T])
        dropout_p = self.dropout_p if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p,
                                             is_causal=True)
        out = out.transpose(1, 2).reshape(B, T, -1)
        return self.o_proj(out)

    def forward_step(self, x, pos: int, rope, kv_ctx):
        """Single-position decode. x: (B, 1, C); kv_ctx: rotated (K, V) over
        positions < pos, each (B, H, T_prev, hd), or None. Returns the output
        and this position's rotated (k, v) for the caller's cache."""
        cos, sin = rope
        q, k, v = self.project(x, cos[pos:pos + 1], sin[pos:pos + 1])
        if kv_ctx is not None:
            K = torch.cat([kv_ctx[0], k], dim=2)
            V = torch.cat([kv_ctx[1], v], dim=2)
        else:
            K, V = k, v
        out = F.scaled_dot_product_attention(q, K, V)   # q is the last position
        out = out.transpose(1, 2).reshape(x.size(0), 1, -1)
        return self.o_proj(out), (k, v)


class FeedForward(nn.Module):
    def __init__(self, cfg: RecurrentLMConfig, quantized: bool):
        super().__init__()
        self.kind = cfg.ffn
        lin = lambda i, o: make_linear(i, o, quantized, cfg.quant_scaling, cfg.rms_eps)
        if self.kind == "swiglu":
            self.w1 = lin(cfg.d_model, cfg.d_ff)
            self.w3 = lin(cfg.d_model, cfg.d_ff)
            self.w2 = lin(cfg.d_ff, cfg.d_model)
        else:  # relu2 (b1.58 recipe for the ternary phase)
            self.w1 = lin(cfg.d_model, cfg.d_ff)
            self.w2 = lin(cfg.d_ff, cfg.d_model)

    def forward(self, x):
        if self.kind == "swiglu":
            return self.w2(F.silu(self.w1(x)) * self.w3(x))
        return self.w2(F.relu(self.w1(x)).square())


class Block(nn.Module):
    def __init__(self, cfg: RecurrentLMConfig, quantized: bool):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.attn = CausalSelfAttention(cfg, quantized)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.ffn = FeedForward(cfg, quantized)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, rope):
        x = x + self.dropout(self.attn(self.attn_norm(x), rope))
        x = x + self.dropout(self.ffn(self.ffn_norm(x)))
        return x

    def forward_step(self, x, pos: int, rope, kv_ctx):
        h, kv = self.attn.forward_step(self.attn_norm(x), pos, rope, kv_ctx)
        x = x + h
        x = x + self.ffn(self.ffn_norm(x))
        return x, kv


# ---------------------------------------------------------------------------
# The recurrent LM
# ---------------------------------------------------------------------------

class RecurrentLM(nn.Module):
    def __init__(self, cfg: RecurrentLMConfig):
        super().__init__()
        self.cfg = cfg
        quant_pc = cfg.quantize and cfg.quantize_prelude_coda
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.prelude = nn.ModuleList(Block(cfg, quant_pc) for _ in range(cfg.n_prelude))
        # The adapter re-injects the prelude embedding e into the loop state
        # each iteration: s <- A([s; e]) (Huginn). It runs inside the loop, so
        # it is quantized together with the core.
        self.adapter = make_linear(2 * cfg.d_model, cfg.d_model, cfg.quantize,
                                   cfg.quant_scaling, cfg.rms_eps)
        self.core = nn.ModuleList(Block(cfg, cfg.quantize) for _ in range(cfg.n_core))
        self.coda = nn.ModuleList(Block(cfg, quant_pc) for _ in range(cfg.n_coda))
        self.out_norm = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.s_init = nn.Parameter(torch.zeros(1, 1, cfg.d_model))

        cos, sin = build_rope_cache(cfg.head_dim, cfg.max_seq_len, cfg.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        nn.init.normal_(self.s_init, std=0.02)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    @property
    def rope(self):
        return self.rope_cos, self.rope_sin

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()
        return n

    # -- full-sequence paths (training and fixed-r evaluation) --------------

    def encode(self, ids):
        """Prelude: token ids -> injected embedding e, shape (B, T, C)."""
        e = self.tok_emb(ids)
        for block in self.prelude:
            e = block(e, self.rope)
        return e

    def core_step(self, s, e):
        """One weight-tied core iteration: s <- core(A([s; e]))."""
        s = self.adapter(torch.cat([s, e], dim=-1))
        for block in self.core:
            s = block(s, self.rope)
        return s

    def decode(self, s):
        """Coda: converged state -> logits."""
        for block in self.coda:
            s = block(s, self.rope)
        return self.lm_head(self.out_norm(s))

    def forward(self, ids, loops: int, bptt_k: Optional[int] = None):
        """Run the full sequence at a fixed loop count.

        ``bptt_k`` enables truncated backprop: the first ``loops - k``
        iterations run without gradient (Huginn's training recipe). Returns
        (logits, final core state).
        """
        e = self.encode(ids)
        s = self.s_init.expand(ids.size(0), ids.size(1), -1)
        no_grad_until = 0
        if bptt_k is not None and self.training:
            no_grad_until = max(0, loops - bptt_k)
        for i in range(loops):
            ctx = torch.no_grad() if i < no_grad_until else nullcontext()
            with ctx:
                s = self.core_step(s, e)
        logits = self.decode(s)
        return logits, s

    @torch.no_grad()
    def embed(self, ids, loops: Optional[int] = None, pad_id: Optional[int] = None):
        """Mean-pooled converged core state — the chain/memory-store hook
        (composability requirement in PLAN.md, mirroring model/0's embed())."""
        loops = loops or round(self.cfg.mean_loops)
        _, s = self.forward(ids, loops=loops)
        if pad_id is not None:
            mask = ids.ne(pad_id).unsqueeze(-1).float()
            return (s * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return s.mean(dim=1)

    # -- auto-halting token-by-token decode ---------------------------------

    @torch.no_grad()
    def generate(self, ids, max_new_tokens: int,
                 kl_threshold: Optional[float] = None,
                 min_loops: Optional[int] = None,
                 max_loops: Optional[int] = None,
                 temperature: float = 0.0, top_k: Optional[int] = None,
                 eos_id: Optional[int] = None,
                 generator: Optional[torch.Generator] = None):
        """Autoregressive decode with per-token KL-convergence halting.

        Every token loops the core until KL(p_i || p_{i-1}) between successive
        probe distributions drops under ``kl_threshold`` (Huginn's
        training-free exit), up to ``max_loops``. The core keeps a mod-k KV
        cache: iteration i writes slot ``i % kv_budget``; earlier positions
        are read from slot ``min(i, depth_t - 1) % kv_budget``, so a token
        that halted early serves its last-written slots (the Huginn budget
        approximation). Prelude and coda keep ordinary one-entry caches; the
        coda cache is committed from the final probe only.

        Returns (ids, depths): the extended ids and a list with the loop
        count spent on each generated token — the halt-depth metadata that
        downstream chain links read as a confidence signal.
        """
        cfg = self.cfg
        kl_threshold = cfg.kl_threshold if kl_threshold is None else kl_threshold
        min_loops = min_loops or cfg.min_loops_infer
        max_loops = max(1, max_loops or cfg.max_loops_infer)
        k_budget = cfg.kv_budget
        B, T0 = ids.shape
        device = ids.device
        H, hd = cfg.n_heads, cfg.head_dim

        prelude_kv: List[Optional[Tuple]] = [None] * cfg.n_prelude
        coda_kv: List[Optional[Tuple]] = [None] * cfg.n_coda
        # Core cache per layer: (K, V) of shape (B, k_budget, H, T, hd).
        core_kv: List[Optional[Tuple]] = [None] * cfg.n_core
        depths = torch.zeros(0, dtype=torch.long, device=device)

        def cache_append(store, layer, kv):
            k, v = kv
            if store[layer] is None:
                store[layer] = (k, v)
            else:
                K, V = store[layer]
                store[layer] = (torch.cat([K, k], dim=2), torch.cat([V, v], dim=2))

        def core_ctx(layer, i):
            """Gather the (K, V) context for iteration i over past positions."""
            if core_kv[layer] is None:
                return None
            K, V = core_kv[layer]
            it = torch.full_like(depths, i)
            slot = (torch.minimum(it, depths - 1) % k_budget)
            idx = slot.view(1, 1, 1, -1, 1).expand(B, 1, H, -1, hd)
            return (K.gather(1, idx).squeeze(1), V.gather(1, idx).squeeze(1))

        def run_position(pos: int, tok):
            """Full prelude/loop/coda pass for one position. Returns
            (final probe logits, loop depth)."""
            nonlocal depths
            # Prelude (commit its cache immediately).
            e = self.tok_emb(tok)
            for li, block in enumerate(self.prelude):
                e, kv = block.forward_step(e, pos, self.rope, prelude_kv[li])
                cache_append(prelude_kv, li, kv)
            # Core loop with mod-k slots for the current token.
            s = self.s_init.expand(B, 1, -1)
            cur_slots = [[None] * k_budget for _ in range(cfg.n_core)]
            prev_logp = None
            final_logits, depth = None, max_loops
            for i in range(max_loops):
                s = self.adapter(torch.cat([s, e], dim=-1))
                for li, block in enumerate(self.core):
                    s, kv = block.forward_step(s, pos, self.rope, core_ctx(li, i))
                    cur_slots[li][i % k_budget] = kv
                # Probe the coda (without committing its cache) for the exit
                # test. A non-positive threshold disables halting, so then we
                # only probe the final iteration for its logits.
                if kl_threshold <= 0 and i < max_loops - 1:
                    continue
                h = s
                probe_kv = []
                for li, block in enumerate(self.coda):
                    h, kv = block.forward_step(h, pos, self.rope, coda_kv[li])
                    probe_kv.append(kv)
                logits = self.lm_head(self.out_norm(h))
                logp = F.log_softmax(logits.float(), dim=-1)
                if prev_logp is not None and i + 1 >= min_loops:
                    kl = (logp.exp() * (logp - prev_logp)).sum(-1).clamp(min=0).max()
                    if kl < kl_threshold:
                        final_logits, depth = logits, i + 1
                        break
                prev_logp = logp
                final_logits = logits
            # Commit: coda cache from the final probe, core slots for this token.
            for li, kv in enumerate(probe_kv):
                cache_append(coda_kv, li, kv)
            for li in range(cfg.n_core):
                written = [kv for kv in cur_slots[li] if kv is not None]
                slots_k, slots_v = [], []
                for j in range(k_budget):
                    kv = cur_slots[li][j] or written[-1]
                    slots_k.append(kv[0])
                    slots_v.append(kv[1])
                k_new = torch.stack(slots_k, dim=1)   # (B, k, H, 1, hd)
                v_new = torch.stack(slots_v, dim=1)
                if core_kv[li] is None:
                    core_kv[li] = (k_new, v_new)
                else:
                    K, V = core_kv[li]
                    core_kv[li] = (torch.cat([K, k_new], dim=3),
                                   torch.cat([V, v_new], dim=3))
            depths = torch.cat([depths, torch.tensor([depth], device=device)])
            return final_logits, depth

        # Prefill the prompt (its depths also come from the halting rule).
        logits = None
        for pos in range(T0):
            logits, _ = run_position(pos, ids[:, pos:pos + 1])

        new_depths = []
        for _ in range(max_new_tokens):
            pos = ids.size(1) - 1
            if pos + 1 >= cfg.max_seq_len:
                break
            step_logits = logits[:, -1, :] if logits.dim() == 3 else logits
            if temperature > 0.0:
                step_logits = step_logits / temperature
                if top_k:
                    kth = torch.topk(step_logits, top_k, dim=-1).values[..., -1:]
                    step_logits = step_logits.masked_fill(step_logits < kth,
                                                          float("-inf"))
                probs = F.softmax(step_logits, dim=-1)
                nxt = torch.multinomial(probs, 1, generator=generator)
            else:
                nxt = step_logits.argmax(dim=-1, keepdim=True)
            ids = torch.cat([ids, nxt], dim=1)
            logits, depth = run_position(ids.size(1) - 1, nxt)
            new_depths.append(depth)
            if eos_id is not None and (nxt == eos_id).all():
                break
        return ids, new_depths
