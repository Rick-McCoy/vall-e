import unittest
from pathlib import Path
from typing import cast

import torch
from hydra.compose import compose
from hydra.initialize import initialize
from torch import Tensor
from torch.utils.data.dataset import Dataset

from config.config import Config
from data.utils import get_mnist, load_mnist


class TestDataUtils(unittest.TestCase):
    def setUp(self) -> None:
        with initialize(config_path="../config"):
            cfg = compose(config_name="config")
            self.cfg = cast(Config, cfg)

        load_mnist(".")
        self.train, self.val, self.test = get_mnist(".")

    def test_load_mnist(self):
        mnist_dir = Path("MNIST")
        self.assertTrue(mnist_dir.is_dir())
        mnist_raw_dir = mnist_dir / "raw"
        self.assertTrue(mnist_raw_dir.is_dir())
        mnist_test_images = mnist_raw_dir / "t10k-images-idx3-ubyte"
        self.assertTrue(mnist_test_images.is_file())
        mnist_test_labels = mnist_raw_dir / "t10k-labels-idx1-ubyte"
        self.assertTrue(mnist_test_labels.is_file())
        mnist_train_images = mnist_raw_dir / "train-images-idx3-ubyte"
        self.assertTrue(mnist_train_images.is_file())
        mnist_train_labels = mnist_raw_dir / "train-labels-idx1-ubyte"
        self.assertTrue(mnist_train_labels.is_file())

    def test_get_mnist(self):
        self.assertIsInstance(self.train, Dataset)
        self.assertIsInstance(self.val, Dataset)
        self.assertIsInstance(self.test, Dataset)

        data, label = next(iter(self.train))
        self.assertIsInstance(data, Tensor)
        self.assertIsInstance(label, int)
        self.assertEqual(
            data.size(),
            (self.cfg.model.input_channels, self.cfg.model.h, self.cfg.model.w),
        )
        self.assertEqual(data.dtype, torch.float32)
