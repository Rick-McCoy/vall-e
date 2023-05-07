from dataclasses import dataclass

from config.data.config import DataConfig
from config.model.config import ModelConfig
from config.train.config import TrainConfig


@dataclass
class Config:
    model: ModelConfig
    data: DataConfig
    train: TrainConfig
