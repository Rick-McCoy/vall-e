from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainConfig:
    acc: int = 1
    auto_lr: bool = True
    auto_batch: bool = False
    batch_size: int = 64
    checkpoint: bool = True
    checkpoint_path: Optional[str] = None
    early_stop: bool = True
    fast_dev_run: bool = False
    lr: float = 5e-4
    monitor: bool = False
    num_workers: int = 2
    optimizer: str = "AdamW"
    scheduler: str = "LinearDecay"
    warmup_steps: int = 32000
    max_steps: int = 800000
    precision: str = "32"
    project: str = "musigen"
    wandb: bool = True

    def __post_init__(self):
        assert self.precision == "32" or self.precision == "16-mixed"
