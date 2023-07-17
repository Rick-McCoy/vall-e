import unittest
from typing import cast

import numpy as np
import torch
from hydra import compose, initialize
from hydra.core.config_store import ConfigStore

from config.config import Config
from config.data.config import DataConfig
from config.model.config import ModelConfig
from config.train.config import TrainConfig
from model.loss import VallELoss

cs = ConfigStore.instance()
cs.store(name="config", node=Config)
cs.store(group="train", name="base_train", node=TrainConfig)
cs.store(group="data", name="base_data", node=DataConfig)
cs.store(group="model", name="base_model", node=ModelConfig)


class TestLoss(unittest.TestCase):
    def setUp(self) -> None:
        with initialize(config_path="../config", version_base=None):
            cfg = compose(config_name="config")
            self.cfg = cast(Config, cfg)
            self.loss = VallELoss(self.cfg)
            self.num_classes = 2**self.cfg.data.codec_bits + 1

    def test_loss(self):
        logit_1 = torch.zeros(8, self.num_classes)
        target_1 = torch.zeros(8, dtype=torch.int64)
        self.assertAlmostEqual(
            self.loss(logit_1, target_1).numpy(), np.log(self.num_classes), places=4
        )
        logit_2 = torch.full((8, self.num_classes), -1e9)
        logit_2[torch.arange(8), torch.arange(8)] = 1e9
        target_2 = torch.arange(8, dtype=torch.int64)
        self.assertAlmostEqual(self.loss(logit_2, target_2).numpy(), 0, places=4)
