from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataConfig:
    _path: str = "."
    train_val_split: float = 0.02
    sample_rate: int = 24000
    channels: int = 1
    codec_rate: int = 75
    codec_channels: int = 8
    codec_bits: int = 10
    _enrolled_codec_len: int = 3
    _max_audio_len: int = 10

    def __post_init__(self):
        self.path = Path(self._path)
        self.train_val_path = self.path / "train_val.csv"
        self.test_path = self.path / "test.csv"
        self.bandwidth = self.codec_rate * self.codec_channels * self.codec_bits / 1000
        self.enrolled_codec_len = int(self._enrolled_codec_len * self.codec_rate)
        self.max_audio_len = int(self._max_audio_len * self.codec_rate)
