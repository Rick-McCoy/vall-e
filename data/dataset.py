from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

import numpy as np
from torch.utils.data import Dataset

from config.config import Config
from data.audio import load_codec
from data.text import encode_text
from data.utils import load_metadata


@dataclass
class Batch:
    text: np.ndarray
    audio: np.ndarray
    enrolled_audio: np.ndarray


class VallEDataset(Dataset):
    def __init__(self, cfg: Config, mode: Literal["train_val"] | Literal["test"]):
        super().__init__()
        self.cfg = cfg
        self.rng = np.random.default_rng()
        self.speaker_list, self.text_list, self.codec_path_list = load_metadata(
            cfg.data.path / f"{mode}.csv"
        )
        self.codec_base_path = (
            cfg.data.path / ("train" if mode == "train_val" else "val") / "codec"
        )
        self.length = len(self.speaker_list)
        self.speaker_to_indices = self.group_by_speaker()

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        text = self.text_list[index]
        encoded_text = encode_text(text)

        codec_path = self.codec_base_path / self.codec_path_list[index]
        codec = load_codec(codec_path)

        speaker = self.speaker_list[index]
        enrolled_codec_path = self.get_enrolled_codec_path(speaker, index)
        enrolled_codec = load_codec(enrolled_codec_path)
        return Batch(text=encoded_text, audio=codec, enrolled_audio=enrolled_codec)

    def group_by_speaker(self) -> defaultdict[str, list[int]]:
        """Returns a dictionary mapping each speaker to a list of indices."""
        speaker_to_indices = defaultdict(list)
        for i, speaker in enumerate(self.speaker_list):
            speaker_to_indices[speaker].append(i)
        return speaker_to_indices

    def get_enrolled_codec_path(self, speaker: str, index: int):
        """Returns a random audio path from the same speaker.
        Excludes the audio path at the given index."""
        indices = self.speaker_to_indices[speaker]
        while (indice := self.rng.choice(indices)) == index:
            pass
        return self.codec_base_path / self.codec_path_list[indice]
