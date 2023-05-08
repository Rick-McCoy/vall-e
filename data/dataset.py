from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from config.config import Config
from data.audio import load_codec
from data.text import encode_text
from data.utils import load_metadata


class VallEDataset(Dataset):
    def __init__(self, cfg: Config, data_path: Path):
        super().__init__()
        self.cfg = cfg
        self.rng = np.random.default_rng()
        self.speaker_list, self.text_list, self.codec_path_list = load_metadata(
            data_path
        )
        self.length = len(self.speaker_list)
        self.speaker_to_indices = self.group_by_speaker()

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        text = self.text_list[index]
        text_len = torch.tensor([len(text)])
        speaker = self.speaker_list[index]
        codec_path = Path(self.codec_path_list[index])
        enrolled_codec_path = self.get_enrolled_codec_path(speaker, index)
        encoded_text = encode_text(text)
        codec = load_codec(codec_path)
        codec_len = np.array([codec.shape[1]])
        enrolled_codec = load_codec(enrolled_codec_path)
        enrolled_codec_len = enrolled_codec.shape[1]
        if enrolled_codec_len > self.cfg.data.enrolled_codec_len:
            start = self.rng.integers(
                0, enrolled_codec_len - self.cfg.data.enrolled_codec_len
            )
            enrolled_codec = enrolled_codec[
                :, start : start + self.cfg.data.enrolled_codec_len
            ]
        elif enrolled_codec_len < self.cfg.data.enrolled_codec_len:
            pad = self.cfg.data.enrolled_codec_len - enrolled_codec_len
            enrolled_codec = np.pad(enrolled_codec, ((0, 0), (0, pad)))
        return Batch(
            text=encoded_text,
            text_len=text_len,
            audio=codec,
            audio_len=codec_len,
            enrolled_audio=enrolled_codec,
        )

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
        return Path(self.codec_path_list[indice])
