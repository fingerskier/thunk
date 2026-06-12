"""Configuration for the Thunk v0 baseline translator.

This is the concrete v0 architecture from SPEC.md Section 2.2: a deep-encoder /
shallow-decoder encoder-decoder Transformer with RMSNorm + QK-norm, RoPE on
self-attention, SwiGLU feed-forward, GQA decoder self-attention, full-MHA
cross-attention, and one tied token embedding.

Two presets are provided:

  * ``reference`` -- the ~112M-parameter SPEC.md config (512-dim, 16/4 layers).
    Use it to confirm the architecture matches the spec; it is far too large to
    train on a CPU.
  * ``small`` -- a few-million-parameter config used as the default for the
    base-training run in this repository. Same architecture, smaller shape, so
    it actually trains and demonstrably learns on a CPU.
"""

from dataclasses import dataclass, field
from typing import List


# Reserved control tags (SPEC.md Section 4). Each must be a single token in the
# tokenizer, never split into subwords. Extend this list as new directions are
# added; the tokenizer trains them as user-defined symbols.
CONTROL_TAGS: List[str] = [
    # source tags
    "<src:english>", "<src:digits>", "<src:upper>",
    "<src:python>", "<src:typescript>", "<src:bash>",
    "<src:cmd>", "<src:powershell>", "<src:lean>",
    "<src:hebrew>", "<src:greek>", "<src:latin>", "<src:aramaic>",
    # target tags
    "<tgt:english>", "<tgt:digits>", "<tgt:upper>",
    "<tgt:python>", "<tgt:typescript>", "<tgt:bash>",
    "<tgt:cmd>", "<tgt:powershell>", "<tgt:lean>",
    "<tgt:hebrew>", "<tgt:greek>", "<tgt:latin>", "<tgt:aramaic>",
]


@dataclass
class ThunkConfig:
    # ---- architecture (SPEC.md Section 2.2) ----
    vocab_size: int = 8000
    d_model: int = 256
    encoder_layers: int = 6
    decoder_layers: int = 2
    n_heads: int = 4                 # query heads (encoder + decoder + cross)
    head_dim: int = 64
    decoder_kv_heads: int = 2        # GQA on decoder self-attention (n_heads:kv)
    d_ff: int = 1024                 # SwiGLU inner dim
    rope_base: float = 100_000.0     # SPEC: RoPE base 100k, self-attention only
    rms_eps: float = 1e-6
    dropout: float = 0.1
    max_seq_len: int = 256           # max source OR target length per segment
    tie_embeddings: bool = True

    # ---- tokenizer ----
    tokenizer_path: str = "tokenizer.model"
    control_tags: List[str] = field(default_factory=lambda: list(CONTROL_TAGS))

    # ---- training ----
    batch_size: int = 64
    lr: float = 5e-4
    min_lr_ratio: float = 0.1        # WSD-style cooldown floor
    weight_decay: float = 0.1
    warmup_steps: int = 200
    max_steps: int = 4000
    decay_frac: float = 0.2          # final fraction of steps spent decaying LR
    label_smoothing: float = 0.1
    grad_clip: float = 1.0
    eval_interval: int = 250
    save_interval: int = 1000
    seed: int = 1337

    # ---- data ----
    data_dir: str = "data_cache"

    def __post_init__(self):
        assert self.d_model == self.n_heads * self.head_dim, (
            f"d_model ({self.d_model}) must equal n_heads*head_dim "
            f"({self.n_heads}*{self.head_dim})"
        )
        assert self.n_heads % self.decoder_kv_heads == 0, (
            "n_heads must be divisible by decoder_kv_heads for GQA"
        )


def reference_config(**overrides) -> ThunkConfig:
    """The ~112M-parameter SPEC.md Section 2.2 reference architecture."""
    cfg = ThunkConfig(
        vocab_size=49152,
        d_model=512,
        encoder_layers=16,
        decoder_layers=4,
        n_heads=8,
        head_dim=64,
        decoder_kv_heads=2,
        d_ff=2048,
        max_seq_len=2048,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def small_config(**overrides) -> ThunkConfig:
    """Few-million-parameter config used for the base-training run."""
    cfg = ThunkConfig()
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg
