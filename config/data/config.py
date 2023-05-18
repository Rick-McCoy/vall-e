from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataConfig:
    path: Path = Path(".")
    train_val_split: float = 0.98
    sample_rate: int = 24000
    channels: int = 1
    codec_rate: int = 75
    codec_channels: int = 8
    codec_bits: int = 10
    enrolled_codec_sec: int = 3
