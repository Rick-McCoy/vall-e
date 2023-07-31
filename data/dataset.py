from collections import defaultdict
from typing import Literal

import numpy as np
from torch.utils.data import Dataset

from config.config import Config
from utils.audio import load_codec
from utils.data import load_metadata
from utils.text import CHAR_TO_CODE, encode_text


class MusicGenDataset(Dataset):
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
        self.max_audio_len = cfg.data.max_codec_len + 1

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        text = self.text_list[index]
        encoded_text = encode_text(text)
        text_len = encoded_text.shape[0]

        if text_len > self.cfg.data.max_text_len:
            raise ValueError(
                f"Text at index {index} has {text_len} characters, "
                f"but at most {self.cfg.data.max_text_len} characters were expected."
            )
        else:
            encoded_text = np.pad(
                encoded_text,
                (0, self.cfg.data.max_text_len - text_len),
                mode="constant",
                constant_values=CHAR_TO_CODE["<PAD>"],
            )

        codec_path = self.codec_base_path / self.codec_path_list[index]
        codec = load_codec(codec_path)
        codec_len = codec.shape[1]

        if codec.shape[0] < self.cfg.data.codec_channels:
            raise ValueError(
                f"Audio file at {codec_path} has {codec.shape[0]} channels, "
                f"but {self.cfg.data.codec_channels} channels were expected."
            )
        elif codec.shape[0] > self.cfg.data.codec_channels:
            codec = codec[: self.cfg.data.codec_channels]

        if codec_len > self.max_audio_len:
            codec = codec[:, : self.max_audio_len]
        else:
            codec = np.pad(
                codec,
                ((0, 0), (0, self.max_audio_len - codec_len)),
                mode="constant",
                constant_values=self.cfg.data.codec_pad,
            )

        return encoded_text, text_len, codec, codec_len

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
        while (diff_index := self.rng.choice(indices)) == index:
            pass
        return self.codec_base_path / self.codec_path_list[diff_index]
