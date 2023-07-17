import unittest

from hydra import compose, initialize
from hydra.core.config_store import ConfigStore

from config.config import Config
from config.data.config import DataConfig
from config.model.config import ModelConfig
from config.train.config import TrainConfig
from main import main

cs = ConfigStore.instance()
cs.store(name="config", node=Config)
cs.store(group="train", name="base_train", node=TrainConfig)
cs.store(group="data", name="base_data", node=DataConfig)
cs.store(group="model", name="base_model", node=ModelConfig)


class TestMain(unittest.TestCase):
    def test_fast_dev(self):
        with initialize(config_path="../config", version_base=None):
            cfg = compose(
                config_name="config",
                overrides=[
                    "train.fast_dev_run=True",
                    'data.path="../../dataset/aihub-emotion/"',
                ],
            )
            main(cfg)
