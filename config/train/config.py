from dataclasses import dataclass


@dataclass
class TrainConfig:
    acc: int = 1
    auto_lr: bool = True
    auto_batch: bool = False
    batch_size: int = 64
    checkpoint: bool = True
    early_stop: bool = True
    fast_dev_run: bool = False
    lr: float = 5e-4
    monitor: bool = False
    num_workers: int = 2
    optimizer: str = "Adam"
    warmup_steps: int = 32000
    max_steps: int = 800000
    precision: str = "32"
    project: str = "vall-e"
