"""Configuration for model #2: the auto-halting recurrent LM with ternary weights.

The reference shapes come from PLAN.md ("Proposed reference configuration"):
a Huginn-style prelude-2 / core-2 (looped, weight-tied) / coda-2 decoder-only
LM. Three presets are provided:

  * ``smoke``   -- tiny CPU config for the byte-level smoke test.
  * ``fp``      -- the Phase 0/1 FP baseline (d_model 256, SwiGLU, ~8.5M params
                   with the 8k shared tokenizer).
  * ``ternary`` -- the Phase 2 ternary QAT variant (d_model 512, ReLU^2 FFN,
                   BitLinear core, ~23M latent params, ~10x smaller at export).
"""

from dataclasses import dataclass


@dataclass
class RecurrentLMConfig:
    # ---- architecture (PLAN.md reference configuration) ----
    vocab_size: int = 8000           # pinned tokenizer shared with model/0
    d_model: int = 256
    n_heads: int = 4
    head_dim: int = 64
    n_prelude: int = 2
    n_core: int = 2                  # weight-tied: the same blocks loop r times
    n_coda: int = 2
    d_ff: int = 1024
    ffn: str = "swiglu"              # "swiglu" (FP phases) | "relu2" (ternary)
    rope_base: float = 100_000.0
    rms_eps: float = 1e-6
    dropout: float = 0.0
    max_seq_len: int = 256
    tie_embeddings: bool = True

    # ---- recurrence (training) ----
    mean_loops: float = 4.0          # log-normal Poisson mean (Huginn)
    max_loops: int = 8               # Ouro saw instability at R=8; cap there
    loop_sigma: float = 0.5          # spread of the log-normal rate
    bptt_k: int = 4                  # backprop through only the last k loops

    # ---- halting (inference) ----
    kl_threshold: float = 5e-4       # per-token KL exit between successive loops
    min_loops_infer: int = 2         # need two probes before a KL exists
    max_loops_infer: int = 16        # ~2x training mean is the tested range
    kv_budget: int = 4               # mod-k KV cache slots for the looped core

    # ---- ternary QAT (Phase 2) ----
    quantize: bool = False
    quant_scaling: str = "median"    # "median" (BitNet-Reloaded, small models)
                                     # | "absmean" (original b1.58)
    quantize_prelude_coda: bool = False  # core-only by default; flip if stable

    # ---- training ----
    batch_size: int = 32
    seq_len: int = 256
    lr: float = 5e-4                 # peak; ternary preset raises this (b1.58)
    min_lr_ratio: float = 0.1
    warmup_steps: int = 200
    max_steps: int = 4000
    weight_decay: float = 0.1
    two_stage_lr: bool = False       # b1.58 schedule: high cosine -> drop ->
    stage2_lr_ratio: float = 0.1     # low cosine; WD 0.1 in stage 1 -> 0 in 2
    use_stablemax: bool = True       # real stablemax CE (Prieto et al.), the
                                     # small-data stabilizer model/1 mis-ported
    label_smoothing: float = 0.0
    grad_clip: float = 1.0
    eval_interval: int = 250
    eval_batches: int = 8
    save_interval: int = 1000
    seed: int = 1337

    def __post_init__(self):
        assert self.d_model == self.n_heads * self.head_dim, (
            f"d_model ({self.d_model}) must equal n_heads*head_dim "
            f"({self.n_heads}*{self.head_dim})"
        )
        assert self.n_core >= 1 and self.max_loops >= 1
        assert 1 <= self.bptt_k
        assert self.kv_budget >= 1
        assert self.ffn in ("swiglu", "relu2")
        assert self.quant_scaling in ("median", "absmean")
        assert self.seq_len <= self.max_seq_len


def smoke_config(**overrides) -> RecurrentLMConfig:
    """Tiny CPU config: byte-level vocab, small shapes, short loops."""
    cfg = RecurrentLMConfig(
        vocab_size=260,              # byte fallback tokenizer (4 specials + 256)
        d_model=64,
        n_heads=2,
        head_dim=32,
        n_prelude=1,
        n_core=1,
        n_coda=1,
        d_ff=128,
        max_seq_len=64,
        mean_loops=2.0,
        max_loops=4,
        bptt_k=2,
        kv_budget=2,
        max_loops_infer=8,
        batch_size=8,
        seq_len=64,
        warmup_steps=10,
        max_steps=200,
        eval_interval=50,
        eval_batches=2,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    cfg.__post_init__()
    return cfg


def fp_config(**overrides) -> RecurrentLMConfig:
    """Phase 0/1 FP baseline: the PLAN.md reference shape."""
    cfg = RecurrentLMConfig()
    for k, v in overrides.items():
        setattr(cfg, k, v)
    cfg.__post_init__()
    return cfg


def ternary_config(**overrides) -> RecurrentLMConfig:
    """Phase 2 ternary variant: 2x width (BitNet-Reloaded parity rule),
    ReLU^2 FFN, quantized core, b1.58 two-stage LR/WD schedule."""
    cfg = RecurrentLMConfig(
        d_model=512,
        n_heads=8,
        head_dim=64,
        d_ff=2048,
        ffn="relu2",
        quantize=True,
        lr=1.5e-3,                   # b1.58: unusually high peak LR
        two_stage_lr=True,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    cfg.__post_init__()
    return cfg


CONFIGS = {"smoke": smoke_config, "fp": fp_config, "ternary": ternary_config}
