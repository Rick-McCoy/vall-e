from dataclasses import dataclass


@dataclass
class ModelConfig:
    hidden_dim: int = 256
    nhead: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.1
    activation: str = "relu"
    num_layers: int = 4
