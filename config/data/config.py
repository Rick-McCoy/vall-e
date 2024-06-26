from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataConfig:
    path: Path = Path(".")
    train_val_split: float = 0.98
    sample_rate: int = 24000
    audio_channels: int = 1
    codec_rate: int = 75
    codec_channels: int = 8
    codec_bits: int = 10
    enrolled_audio_length: int = 3
    sample_sentence: str = "나는 고양이로소이다. 이름은 아직 없다."

    def __post_init__(self):
        assert self.audio_channels == 1 or self.audio_channels == 2
