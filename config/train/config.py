from dataclasses import dataclass


@dataclass
class TrainConfig:
    acc: int = 1
    batch_size: int = 64
    checkpoint_path: str | None = None
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
    precision: str = "32"
    project: str = "voicegen"
    wandb: bool = True
    weight_average: bool = False
