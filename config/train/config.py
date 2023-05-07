from dataclasses import dataclass
from typing import Literal


@dataclass
class TrainConfig:
    acc: int = 1
    auto_lr: bool = True
    auto_batch: bool = False
    batch_size: int = 64
    checkpoint: bool = True
    early_stop: bool = True
    fast_dev_run: bool = False
    lr: float = 0.0001
    monitor: bool = False
    num_workers: int = 2
    optimizer: str = "Adam"
    precision: Literal["32"] | Literal["16-mixed"] = "32"
    project: str = "toy"
