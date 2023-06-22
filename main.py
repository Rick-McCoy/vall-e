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
    ModelCheckpoint,
)
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.tuner import Tuner

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
        logger = TensorBoardLogger(save_dir="logs", name="vall-e")

    callbacks = []

    if cfg.train.checkpoint:
        checkpoint_dir = Path("checkpoints")
        if wandb.run is not None:
            checkpoint_dir /= wandb.run.name
        else:
            checkpoint_dir /= str(int(time.time()))
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

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("medium")

    precision = cfg.train.precision
    assert precision == "32" or precision == "16-mixed"
    trainer = Trainer(
        strategy="ddp",
        accumulate_grad_batches=cfg.train.acc,
        callbacks=callbacks,
        detect_anomaly=True,
        fast_dev_run=cfg.train.fast_dev_run,
        logger=logger,
        log_every_n_steps=10,
        max_steps=cfg.train.max_steps,
        num_sanity_val_steps=10,
        precision=precision,
    )
    tuner = Tuner(trainer)

    if cfg.train.auto_lr and not cfg.train.fast_dev_run:
        tuner.lr_find(model=compiled_model, datamodule=datamodule)

    if cfg.train.auto_batch and not cfg.train.fast_dev_run:
        tuner.scale_batch_size(model=compiled_model, datamodule=datamodule)

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
