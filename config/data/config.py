from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataConfig:
    path: Path = Path(".")
    train_val_split: float = 0.98
    sample_rate: int = 24000
    bandwidth: int = 6000
    codec_channels: int = 8
    codec_bits: int = 10
    codec_sos: int = -1
    codec_eos: int = -1
    codec_pad: int = -1
    codec_num: int = -1
    max_codec_len: int = 512
    max_text_len: int = 128
    sample_sentence: str = "나는 고양이로소이다. 이름은 아직 없다."

    def __post_init__(self):
        self.codec_sos = 2**self.codec_bits
        self.codec_eos = 2**self.codec_bits + 1
        self.codec_pad = 2**self.codec_bits + 2
        self.codec_num = 2**self.codec_bits + 3
