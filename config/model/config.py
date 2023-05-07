from dataclasses import dataclass


@dataclass
class ModelConfig:
    kernel_size: int = 3
    input_channels: int = 1
    hidden_channels: int = 16
    h: int = 28
    w: int = 28
    num_classes: int = 10
    num_layers: int = 8
