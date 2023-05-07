from typing import Tuple

import torch
import wandb
from lightning import LightningModule
from lightning.pytorch.loggers import WandbLogger
from torch import Tensor
from torchmetrics.classification import MulticlassAccuracy

from config.config import Config
from model.classifier import SimpleClassifier
from model.loss import SimpleLoss


class SimpleModel(LightningModule):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.learning_rate = cfg.train.lr
        self.classifier = SimpleClassifier(cfg)
        self.loss = SimpleLoss(cfg)
        self.acc = MulticlassAccuracy(num_classes=cfg.model.num_classes, top_k=1)
        self.example_input_array = torch.zeros(
            (1, cfg.model.input_channels, cfg.model.h, cfg.model.w)
        )
        self.logger: WandbLogger

    def log_table(self, image: Tensor, pred: Tensor, header: str):
        self.logger.log_table(
            f"{header}/table",
            columns=["Image", "Prediction"],
            data=[[wandb.Image(image.detach().cpu().numpy()), pred.item()]],
        )

    def forward(self, data: Tensor, *args, **kwargs) -> Tensor:
        return self.classifier(data)

    def training_step(self, batch: Tuple[Tensor, Tensor], *args, **kwargs) -> Tensor:
        data, label = batch
        output = self(data)
        loss = self.loss(output, label)
        self.acc(output, label)
        self.log("train/loss", loss, on_step=True, prog_bar=True)
        self.log("train/acc", self.acc, on_step=True, prog_bar=True)
        return loss

    def validation_step(
        self, batch: Tuple[Tensor, Tensor], batch_idx: int, *args, **kwargs
    ):
        data, label = batch
        output = self(data)
        loss = self.loss(output, label)
        self.acc(output, label)
        self.log("val/loss", loss, on_epoch=True, sync_dist=True)
        self.log("val/acc", self.acc, on_epoch=True, sync_dist=True)
        if batch_idx == 0 and self.device.index == 0:
            pred = torch.argmax(output[0], dim=-1)
            self.log_table(data[0], pred, "val")

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int, *args, **kwargs):
        data, label = batch
        output = self(data)
        loss = self.loss(output, label)
        self.acc(output, label)
        self.log("test/loss", loss, on_epoch=True, sync_dist=True)
        self.log("test/acc", self.acc, on_epoch=True, sync_dist=True)
        if batch_idx == 0 and self.device.index == 0:
            pred = torch.argmax(output[0], dim=-1)
            self.log_table(data[0], pred, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(
            params=self.parameters(), lr=self.learning_rate, amsgrad=True
        )
