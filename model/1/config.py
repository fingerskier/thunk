"""Configuration for model #1: the small recursive reasoning diffuser."""

from dataclasses import dataclass


@dataclass
class ReasoningDiffuserConfig:
    vocab_size: int = 128
    d_model: int = 128
    n_heads: int = 4
    d_ff: int = 512
    recursion_depth: int = 2
    warmup_recursions: int = 1
    supervision_steps: int = 8
    max_question_len: int = 64
    answer_len: int = 16
    dropout: float = 0.1
    mask_id: int = 0
    pad_id: int = 1
    bos_id: int = 2
    eos_id: int = 3
    lr: float = 1e-4
    weight_decay: float = 1.0
    ema_decay: float = 0.999
    label_smoothing: float = 0.0
    seed: int = 1337

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0
        assert self.warmup_recursions >= 0
        assert self.supervision_steps >= 1


def tiny_config(**overrides) -> ReasoningDiffuserConfig:
    cfg = ReasoningDiffuserConfig()
    for key, value in overrides.items():
        setattr(cfg, key, value)
    cfg.__post_init__()
    return cfg
