from dataclasses import dataclass
from typing import Optional

from utils.types import Precision


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
    betas: tuple[float, float] = (0.9, 0.95)
    monitor: bool = False
    num_workers: int = 2
    optimizer: str = "AdamW"
    scheduler: str = "CosineWithWarmup"
    warmup_steps: int = 4000
    max_steps: int = 1000000
    gradient_clip_val: float = 1.0
    weight_decay: float = 0.1
    precision: Precision = Precision.FP16_MIXED
    project: str = "musicgen"
    wandb: bool = True
