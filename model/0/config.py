from dataclasses import dataclass


@dataclass
class ThunkConfig:
    # model
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 512          # feedforward inner dim (4x d_model)
    vocab_size: int = 8000
    max_seq_len: int = 256
    dropout: float = 0.1

    # recursion
    max_recurse: int = 8
    stability_threshold: float = 0.99  # cosine similarity to stop

    # training
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_steps: int = 50_000
    eval_interval: int = 500
    save_interval: int = 5000
    grad_clip: float = 1.0

    # data
    tokenizer_path: str = "tokenizer.model"
    dataset_name: str = "roneneldan/TinyStories"
