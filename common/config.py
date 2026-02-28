from dataclasses import dataclass
import torch


@dataclass
class GBTConfig:
    block_size: int
    vocab_size: int
    n_layer: int
    n_head: int
    n_embd: int
    dropout: float = 0.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class TrainConfig:
    batch_size: int = 64
    learning_rate: float = 5e-4
    max_steps: int = 1000
    warmup_steps: int = 100
    weight_decay: float = 0.01
    early_stop_patience: int = 500
    output_dir: str = "models/"
