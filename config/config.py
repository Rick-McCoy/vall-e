from dataclasses import dataclass, field

from config.data.config import DataConfig
from config.model.config import ModelConfig
from config.train.config import TrainConfig


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    name: str = "default"


def dict_to_config(cfg: dict):
    model_config = ModelConfig(**cfg["model"])
    data_config = DataConfig(**cfg["data"])
    train_config = TrainConfig(**cfg["train"])
    return Config(
        model=model_config,
        data=data_config,
        train=train_config,
        name=cfg["name"],
    )
