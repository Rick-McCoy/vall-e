import unittest
from typing import cast

import torch
from hydra import compose, initialize

from config.config import Config
from model.classifier import SimpleClassifier


class TestClassifier(unittest.TestCase):
    def setUp(self) -> None:
        with initialize(config_path="../config"):
            cfg = compose(config_name="config")
            self.cfg = cast(Config, cfg)
            self.classifier = SimpleClassifier(self.cfg)

    def test_classifier(self):
        data = torch.zeros(
            8, self.cfg.model.input_channels, self.cfg.model.h, self.cfg.model.w
        )
        output = self.classifier(data)
        self.assertEqual(output.size(), (8, self.cfg.model.num_classes))
