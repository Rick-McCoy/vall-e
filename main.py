import time
from pathlib import Path
from typing import cast

import hydra
import torch
import wandb
from hydra.core.config_store import ConfigStore
from lightning import Trainer
from lightning.pytorch.callbacks import (
    DeviceStatsMonitor,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    StochasticWeightAveraging,
)
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from torch import Tensor

from config.config import Config
from config.data.config import DataConfig
from config.model.config import ModelConfig
from config.train.config import TrainConfig
from data.datamodule import VallEDataModule
from model.musicgen import MusicGen
from utils.model import remove_weight_norm

cs = ConfigStore.instance()
cs.store(name="config", node=Config)
cs.store(group="train", name="base_train", node=TrainConfig)
cs.store(group="data", name="base_data", node=DataConfig)
cs.store(group="model", name="base_model", node=ModelConfig)


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: Config):
    model = MusicGen(cfg)
    compiled_model = cast(MusicGen, torch.compile(model, disable=True))
    datamodule = VallEDataModule(cfg)

    Path("logs").mkdir(exist_ok=True)
    if cfg.train.fast_dev_run:
        logger = None
    elif cfg.train.wandb:
        logger = WandbLogger(project=cfg.train.project, save_dir="logs")
    else:
        logger = TensorBoardLogger(save_dir="logs", name=cfg.train.project)

    callbacks = []

    if cfg.train.checkpoint:
        checkpoint_dir = Path("checkpoints")
        if wandb.run is not None:
            checkpoint_dir /= wandb.run.name
        elif (
            isinstance(logger, WandbLogger)
            and not isinstance(logger.experiment.name, str)
            and logger.experiment.name() is not None
        ):
            checkpoint_dir /= logger.experiment.name()
        else:
            checkpoint_dir /= time.strftime("%Y-%m-%d_%H-%M-%S")
        callbacks.append(
            ModelCheckpoint(
                dirpath=checkpoint_dir,
                filename="{epoch:03d}-val_loss={val/loss:.4f}",
                monitor="val/loss",
                save_top_k=3,
                mode="min",
                auto_insert_metric_name=False,
                save_weights_only=True,
            )
        )

    if cfg.train.monitor:
        callbacks.append(DeviceStatsMonitor())

    if cfg.train.early_stop:
        callbacks.append(
            EarlyStopping(
                monitor="val/loss",
                min_delta=0.00,
                patience=3,
                verbose=False,
                mode="min",
            )
        )

    if cfg.train.weight_average:

        def avg_fn(
            averaged_model_parameter: Tensor,
            model_parameter: Tensor,
            num_averaged: Tensor,
        ) -> Tensor:
            return averaged_model_parameter * 0.99 + model_parameter * 0.01

        callbacks.append(
            StochasticWeightAveraging(
                swa_lrs=cfg.train.lr,
                swa_epoch_start=0.5,
                annealing_epochs=100,
                avg_fn=avg_fn,
            )
        )

    if cfg.train.scheduler != "None":
        callbacks.append(
            LearningRateMonitor(logging_interval="step", log_momentum=False)
        )

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    trainer = Trainer(
        strategy="ddp",
        accumulate_grad_batches=cfg.train.acc,
        gradient_clip_val=cfg.train.gradient_clip_val,
        callbacks=callbacks,
        detect_anomaly=True,
        fast_dev_run=cfg.train.fast_dev_run,
        logger=logger,
        log_every_n_steps=10,
        max_steps=cfg.train.max_steps,
        num_sanity_val_steps=10,
        precision=cfg.train.precision.value,
    )

    trainer.fit(
        model=compiled_model, datamodule=datamodule, ckpt_path=cfg.train.checkpoint_path
    )
    trainer.test(model=compiled_model, datamodule=datamodule)

    remove_weight_norm(compiled_model)
    save_path = Path("model-store")
    save_path.mkdir(exist_ok=True)
    compiled_model.to_torchscript(save_path / "model.pt")


if __name__ == "__main__":
    main()
