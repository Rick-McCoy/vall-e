import unittest
from typing import cast

from hydra import compose, initialize
from hydra.core.config_store import ConfigStore

from config.config import Config
from config.data.config import DataConfig
from config.model.config import ModelConfig
from config.train.config import TrainConfig
from data.datamodule import VoiceGenDataModule
from utils.types import CollatedBatch

cs = ConfigStore.instance()
cs.store(name="config", node=Config)
cs.store(group="train", name="base_train", node=TrainConfig)
cs.store(group="data", name="base_data", node=DataConfig)
cs.store(group="model", name="base_model", node=ModelConfig)


class TestDataModule(unittest.TestCase):
    def setUp(self) -> None:
        with initialize(config_path="../config", version_base=None):
            cfg = compose(
                config_name="config",
                overrides=['data.path="../../dataset/aihub-emotion/"'],
            )
            self.cfg = cast(Config, cfg)
            self.module = VoiceGenDataModule(self.cfg)

        self.module.prepare_data()
        self.module.setup()

    def test_train_dataloader(self):
        collated_batch = next(iter(self.module.train_dataloader()))
        self.assertIsInstance(collated_batch, CollatedBatch)

    def test_val_dataloader(self):
        collated_batch = next(iter(self.module.val_dataloader()))
        self.assertIsInstance(collated_batch, CollatedBatch)

    def test_test_dataloader(self):
        collated_batch = next(iter(self.module.test_dataloader()))
        self.assertIsInstance(collated_batch, CollatedBatch)
