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
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.loggers.wandb import WandbLogger
from torch import Tensor

from config.config import Config
from config.data.config import DataConfig
from config.model.config import ModelConfig
from config.train.config import TrainConfig
from data.datamodule import VoiceGenDataModule
from model.voicegen import VoiceGen
from utils.model import remove_norm

cs = ConfigStore.instance()
cs.store(name="config", node=Config)
cs.store(group="train", name="base_train", node=TrainConfig)
cs.store(group="data", name="base_data", node=DataConfig)
cs.store(group="model", name="base_model", node=ModelConfig)


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: Config):
    model = VoiceGen(cfg)
    compiled_model = cast(VoiceGen, torch.compile(model, disable=True))
    datamodule = VoiceGenDataModule(cfg)

    callbacks = []
    checkpoint_dir = Path("checkpoints")

    Path("logs").mkdir(exist_ok=True)
    if cfg.train.fast_dev_run:
        logger = None
        checkpoint_dir /= "fast_dev_run"
    elif cfg.train.wandb:
        logger = WandbLogger(project=cfg.train.project, save_dir="logs")
        if wandb.run is not None:
            checkpoint_dir /= wandb.run.name
        else:
            checkpoint_dir /= logger.experiment.name()
    else:
        logger = TensorBoardLogger(save_dir="logs", name=cfg.train.project)
        checkpoint_dir /= time.strftime("%Y-%m-%d_%H-%M-%S")

    callbacks.append(
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="{epoch:03d}-val_loss={val/loss:.4f}",
            monitor="val/loss",
            save_top_k=3,
            mode="min",
            auto_insert_metric_name=False,
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

    if cfg.train.scheduler != "None":
        callbacks.append(
            LearningRateMonitor(logging_interval="step", log_momentum=False)
        )

    if cfg.train.weight_average:

        def avg_fn(
            averaged_model_parameter: Tensor,
            model_parameter: Tensor,
            _num_averaged: Tensor,
        ) -> Tensor:
            return averaged_model_parameter * 0.99 + model_parameter * 0.01

        callbacks.append(
            StochasticWeightAveraging(
                swa_lrs=cfg.train.lr / 10,
                swa_epoch_start=0.5,
                annealing_epochs=10,
                annealing_strategy="cos",
                avg_fn=avg_fn,
            )
        )

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    precision = cfg.train.precision
    assert precision == "16-mixed" or precision == "32"
    trainer = Trainer(
        strategy="ddp",
        accumulate_grad_batches=cfg.train.acc,
        gradient_clip_val=cfg.train.gradient_clip_val,
        callbacks=callbacks,
        detect_anomaly=False,
        fast_dev_run=cfg.train.fast_dev_run,
        logger=logger,
        log_every_n_steps=10,
        max_steps=cfg.train.max_steps,
        num_sanity_val_steps=10,
        precision=precision,
    )

    trainer.fit(
        model=compiled_model,
        datamodule=datamodule,
        ckpt_path=cfg.train.checkpoint_path,
    )
    trainer.test(model=compiled_model, datamodule=datamodule)

    remove_norm(compiled_model)
    save_path = Path("model-store")
    save_path.mkdir(exist_ok=True)
    # compiled_model.to_torchscript(save_path / "model.pt")


if __name__ == "__main__":
    main()
