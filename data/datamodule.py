from lightning import LightningDataModule
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader

from config.config import Config
from data.dataset import MusicGenDataset


class VoiceGenDataModule(LightningDataModule):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.batch_size = cfg.train.batch_size

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str | None = None) -> None:
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
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=self.cfg.train.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=self.cfg.train.num_workers,
            pin_memory=True,
        )
