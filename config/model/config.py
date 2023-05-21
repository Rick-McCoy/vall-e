from dataclasses import dataclass


@dataclass
class ModelConfig:
    hidden_dim: int = 128
    nhead: int = 4
    dim_feedforward: int = 256
    dropout: float = 0.1
    num_layers: int = 4
