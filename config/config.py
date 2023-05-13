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
