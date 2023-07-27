from functools import partial
from typing import Optional

import numpy as np
import torch
from lightning import LightningDataModule
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader

from config.config import Config
from data.dataset import MusicGenDataset
from utils.text import CHAR_TO_CODE
from utils.types import Batch, CollatedBatch


def collate_fn(batches: list[Batch], codec_pad: int):
    text_len = [batch.text.shape[0] for batch in batches]
    audio_len = [batch.audio.shape[1] for batch in batches]
    enrolled_audio_len = [batch.enrolled_audio.shape[1] for batch in batches]
    max_text_len = max(text_len)
    max_audio_len = max(audio_len)
    max_enrolled_audio_len = max(enrolled_audio_len)
    text = np.stack(
        [
            np.pad(
                batch.text,
                (0, max_text_len - text_len),
                mode="constant",
                constant_values=CHAR_TO_CODE["<PAD>"],
            )
            for batch, text_len in zip(batches, text_len)
        ],
        axis=0,
    )
    audio = np.stack(
        [
            np.pad(
                batch.audio,
                ((0, 0), (0, max_audio_len - audio_len)),
                mode="constant",
                constant_values=codec_pad,
            )
            for batch, audio_len in zip(batches, audio_len)
        ],
        axis=0,
    )
    enrolled_audio = np.stack(
        [
            np.pad(
                batch.enrolled_audio,
                ((0, 0), (0, max_enrolled_audio_len - audio_len)),
                mode="constant",
                constant_values=codec_pad,
            )
            for batch, audio_len in zip(batches, enrolled_audio_len)
        ],
        axis=0,
    )
    return CollatedBatch(
        text=torch.from_numpy(text).long(),
        text_len=torch.from_numpy(np.array(text_len)).long(),
        audio=torch.from_numpy(audio).long(),
        audio_len=torch.from_numpy(np.array(audio_len)).long(),
        enrolled_audio=torch.from_numpy(enrolled_audio).long(),
        enrolled_audio_len=torch.from_numpy(np.array(enrolled_audio_len)).long(),
    )


class VoiceGenDataModule(LightningDataModule):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.batch_size = cfg.train.batch_size
        self.codec_pad = 2**cfg.data.codec_bits + 2

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage == "validate" or stage is None:
            train_val_dataset = MusicGenDataset(self.cfg, "train_val")
            train_val_size = len(train_val_dataset)
            train_size = int(train_val_size * self.cfg.data.train_val_split)
            val_size = train_val_size - train_size
            self.train_dataset, self.val_dataset = random_split(
                train_val_dataset, [train_size, val_size]
            )

        if stage == "test" or stage is None:
            self.test_dataset = MusicGenDataset(self.cfg, "test")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.cfg.train.num_workers,
            collate_fn=partial(collate_fn, codec_pad=self.codec_pad),
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=self.cfg.train.num_workers,
            collate_fn=partial(collate_fn, codec_pad=self.codec_pad),
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=self.cfg.train.num_workers,
            collate_fn=partial(collate_fn, codec_pad=self.codec_pad),
            pin_memory=True,
        )
