import unittest
from typing import cast

import torch
from hydra import compose, initialize
from torch import Tensor

from config.config import Config
from data.datamodule import SimpleDataModule


class TestDataModule(unittest.TestCase):
    def setUp(self) -> None:
        with initialize(config_path="../config"):
            cfg = compose(config_name="config")
            self.cfg = cast(Config, cfg)
            self.module = SimpleDataModule(self.cfg)

        self.module.prepare_data()
        self.module.setup()

    def test_train_dataloader(self):
        data, label = next(iter(self.module.train_dataloader()))
        self.assertIsInstance(data, Tensor)
        self.assertEqual(data.dtype, torch.float32)
        self.assertEqual(
            data.size(),
            (
                self.cfg.train.batch_size,
                self.cfg.model.input_channels,
                self.cfg.model.h,
                self.cfg.model.w,
            ),
        )
        self.assertIsInstance(label, Tensor)
        self.assertEqual(label.dtype, torch.int64)
        self.assertEqual(label.size(), (self.cfg.train.batch_size,))

    def test_val_dataloader(self):
        data, label = next(iter(self.module.val_dataloader()))
        self.assertIsInstance(data, Tensor)
        self.assertEqual(data.dtype, torch.float32)
        self.assertEqual(
            data.size(),
            (
                self.cfg.train.batch_size,
                self.cfg.model.input_channels,
                self.cfg.model.h,
                self.cfg.model.w,
            ),
        )
        self.assertIsInstance(label, Tensor)
        self.assertEqual(label.dtype, torch.int64)
        self.assertEqual(label.size(), (self.cfg.train.batch_size,))

    def test_test_dataloader(self):
        data, label = next(iter(self.module.test_dataloader()))
        self.assertIsInstance(data, Tensor)
        self.assertEqual(data.dtype, torch.float32)
        self.assertEqual(
            data.size(),
            (
                self.cfg.train.batch_size,
                self.cfg.model.input_channels,
                self.cfg.model.h,
                self.cfg.model.w,
            ),
        )
        self.assertIsInstance(label, Tensor)
        self.assertEqual(label.dtype, torch.int64)
        self.assertEqual(label.size(), (self.cfg.train.batch_size,))
