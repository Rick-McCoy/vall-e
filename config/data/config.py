from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataConfig:
    _path: str = "."
    sample_rate: int = 24000
    channels: int = 1
    codec_rate: int = 75
    codec_channels: int = 8
    codec_bits: int = 10
    _enrolled_codec_len: int = 3

    def __post_init__(self):
        self.path = Path(self._path)
        self.bandwidth = self.codec_rate * self.codec_channels * self.codec_bits / 1000
        self.enrolled_codec_len = int(self._enrolled_codec_len * self.codec_rate)
