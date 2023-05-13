from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from lightning import LightningDataModule
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader

from config.config import Config
from data.dataset import Batch, VallEDataset


@dataclass
class CollatedBatch:
    text: torch.Tensor
    text_len: torch.Tensor
    audio: torch.Tensor
    audio_len: torch.Tensor
    enrolled_audio: torch.Tensor


def collate_fn(batches: list[Batch]) -> CollatedBatch:
    text_len = [batch.text.shape[0] for batch in batches]
    audio_len = [batch.audio.shape[1] for batch in batches]
    max_text_len = max(text_len)
    max_audio_len = max(audio_len)
    text = np.stack(
        [
            np.pad(batch.text, (0, max_text_len - text_len))
            for batch, text_len in zip(batches, text_len)
        ],
        axis=0,
    )
    audio = np.stack(
        [
            np.pad(batch.audio, ((0, 0), (0, max_audio_len - audio_len)))
            for batch, audio_len in zip(batches, audio_len)
        ],
        axis=0,
    )
    enrolled_audio = np.stack([batch.enrolled_audio for batch in batches], axis=0)
    return CollatedBatch(
        text=torch.from_numpy(text).long(),
        text_len=torch.from_numpy(np.array(text_len)).long(),
        audio=torch.from_numpy(audio).long(),
        audio_len=torch.from_numpy(np.array(audio_len)).long(),
        enrolled_audio=torch.from_numpy(enrolled_audio).long(),
    )


class VallEDataModule(LightningDataModule):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.batch_size = cfg.train.batch_size

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_val_dataset = VallEDataset(self.cfg, "train_val")
        self.test_dataset = VallEDataset(self.cfg, "test")

        train_val_size = len(self.train_val_dataset)
        train_size = int(train_val_size * self.cfg.data.train_val_split)
        val_size = train_val_size - train_size
        self.train_dataset, self.val_dataset = random_split(
            self.train_val_dataset, [train_size, val_size]
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.cfg.train.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.cfg.train.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.cfg.train.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )
