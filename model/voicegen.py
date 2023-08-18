import dataclasses
from typing import Literal

import numpy as np
import torch
import wandb
from lightning import LightningModule
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.loggers.wandb import WandbLogger
from omegaconf import OmegaConf
from torch import Tensor, nn
from torchmetrics.classification import MulticlassAccuracy
from tqdm import tqdm

from config.config import Config, dict_to_config
from model.delayed_transformer import DelayedTransformer
from utils.audio import codec_to_audio, mel_spectrogram
from utils.data import plot_mel_spectrogram
from utils.model import nucleus_sample
from utils.text import CHAR_TO_CODE, VOCAB_SIZE, encode_text
from utils.utils import remove_delay, unpad_sequence


class VoiceGen(LightningModule):
    def __init__(self, cfg: Config | dict) -> None:
        super().__init__()
        if isinstance(cfg, dict):
            cfg = dict_to_config(cfg)
        if OmegaConf.is_config(cfg):
            self.save_hyperparameters(OmegaConf.to_container(cfg))
        elif isinstance(cfg, Config):
            self.save_hyperparameters(dataclasses.asdict(cfg))
        else:
            self.save_hyperparameters(cfg)
        self.cfg = cfg
        self.codec_channels = cfg.data.codec_channels
        self.sample_rate = cfg.data.sample_rate
        self.text_pad = float(CHAR_TO_CODE["<PAD>"])
        self.register_buffer(
            "codec_eos",
            torch.full(
                (1, cfg.data.codec_channels, 1),
                cfg.data.codec_eos,
                dtype=torch.long,
            ),
        )
        self.codec_eos: Tensor
        self.codec_pad = float(cfg.data.codec_pad)
        self.lr = cfg.train.lr
        self.delayed_transformer = DelayedTransformer(cfg)
        self.loss = nn.CrossEntropyLoss(ignore_index=cfg.data.codec_pad)
        self.acc = MulticlassAccuracy(
            num_classes=cfg.data.codec_num,
            top_k=1,
            ignore_index=cfg.data.codec_pad,
        )
        self.example_input_array = (
            torch.randint(0, VOCAB_SIZE, (cfg.train.batch_size, cfg.data.max_text_len)),
            torch.randint(
                0,
                cfg.data.codec_sos,
                (cfg.train.batch_size, cfg.data.codec_channels, cfg.data.max_codec_len),
            ),
            torch.randint(
                cfg.data.max_text_len // 2,
                cfg.data.max_text_len,
                (cfg.train.batch_size,),
            ),
            torch.randint(
                cfg.data.max_codec_len // 2,
                cfg.data.max_codec_len,
                (cfg.train.batch_size,),
            ),
        )
        self.register_buffer(
            "sample_text",
            torch.from_numpy(encode_text(cfg.data.sample_sentence)).unsqueeze(0),
        )
        self.sample_text: Tensor
        self.register_buffer(
            "sample_text_len",
            torch.tensor([self.sample_text.shape[1]]),
        )
        self.sample_text_len: Tensor
        self.max_infer_len = cfg.data.max_codec_len - self.codec_channels

    def forward(
        self,
        text: Tensor,
        audio: Tensor,
        text_len: Tensor,
        audio_len: Tensor,
    ) -> Tensor:
        return self.delayed_transformer(text, audio, text_len, audio_len)

    def inference(
        self,
        text: Tensor,
        target_text: Tensor,
        target_audio: Tensor,
        text_len: Tensor,
        target_text_len: Tensor,
        target_audio_len: Tensor,
    ) -> Tensor:
        assert len(text) == 1, "Inference only supports batch size 1"
        unpad_text = unpad_sequence(text, text_len, batch_first=True)
        unpad_target_text = unpad_sequence(
            target_text, target_text_len, batch_first=True
        )
        concat_text = torch.nn.utils.rnn.pad_sequence(
            [
                torch.cat([unpad_target_text_item, unpad_text_item])
                for unpad_text_item, unpad_target_text_item in zip(
                    unpad_text, unpad_target_text
                )
            ],
            batch_first=True,
            padding_value=self.text_pad,
        )
        concat_text_len = text_len + target_text_len
        audio = torch.empty_like(target_audio)[:, :, :0]
        audio_len = torch.zeros_like(target_audio_len)
        target_audio_len -= self.codec_channels
        target_audio = target_audio[:, :, : target_audio_len.max().item()]
        for _ in tqdm(range(self.max_infer_len), leave=False):
            logits = self.delayed_transformer(
                concat_text,
                torch.cat([target_audio, audio], dim=-1),
                concat_text_len,
                target_audio_len + audio_len,
            )[:, :, -1]
            sampled_token = nucleus_sample(logits, top_p=0.9).unsqueeze(-1)
            audio = torch.cat([audio, sampled_token], dim=-1)
            audio_len += 1
            if torch.all(sampled_token == self.codec_eos):
                break

        audio = remove_delay(audio, audio_len, self.codec_channels, self.codec_pad)
        audio = audio.clamp_max(self.cfg.data.codec_sos - 1)
        return audio

    def single_step(
        self,
        batch: tuple[Tensor, Tensor, Tensor, Tensor],
        mode: Literal["train", "val", "test"],
    ) -> Tensor:
        text, text_len, audio, audio_len = batch
        output = self.delayed_transformer(text, audio[:, :, :-1], text_len, audio_len)
        loss = self.loss(output.permute(0, 3, 1, 2), audio[:, :, 1:])
        self.acc(output.permute(0, 3, 1, 2), audio[:, :, 1:])
        if mode == "train":
            self.log(f"{mode}/loss", loss, on_step=True)
            self.log(f"{mode}/acc", self.acc, on_step=True)
        else:
            self.log(f"{mode}/loss", loss, on_epoch=True, sync_dist=True)
            self.log(f"{mode}/acc", self.acc, on_epoch=True, sync_dist=True)
        return loss

    def training_step(
        self, batch: tuple[Tensor, Tensor, Tensor, Tensor], batch_idx: int
    ) -> Tensor:
        loss = self.single_step(batch, "train")
        return loss

    def validation_step(
        self, batch: tuple[Tensor, Tensor, Tensor, Tensor], batch_idx: int
    ):
        self.single_step(batch, "val")
        if batch_idx == 0 and self.device.index == 0:
            self.log_table(batch, "val")

    def test_step(self, batch: tuple[Tensor, Tensor, Tensor, Tensor], batch_idx: int):
        self.single_step(batch, "test")
        if batch_idx == 0 and self.device.index == 0:
            self.log_table(batch, "test")

    def configure_optimizers(self):
        eps = 1e-4 if self.cfg.train.precision == "16-mixed" else 1e-8
        if self.cfg.train.optimizer == "Adam":
            optimizer = torch.optim.Adam(
                params=self.parameters(),
                lr=self.lr,
                betas=self.cfg.train.betas,
                eps=eps,
            )
        elif self.cfg.train.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(
                params=self.parameters(),
                lr=self.lr,
                weight_decay=self.cfg.train.weight_decay,
                betas=self.cfg.train.betas,
                eps=eps,
            )
        else:
            raise NotImplementedError(f"Unknown optimizer {self.cfg.train.optimizer}")

        if self.cfg.train.scheduler == "LinearDecay":
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1 / self.cfg.train.warmup_steps,
                total_iters=self.cfg.train.warmup_steps,
            )
            decay_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1,
                end_factor=0.0,
                total_iters=self.cfg.train.max_steps - self.cfg.train.warmup_steps,
            )

            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, decay_scheduler],
                milestones=[self.cfg.train.warmup_steps],
            )

            return [optimizer], [
                {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                    "name": "train/lr",
                }
            ]
        elif self.cfg.train.scheduler == "CosineWithWarmup":
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1 / self.cfg.train.warmup_steps,
                total_iters=self.cfg.train.warmup_steps,
            )
            decay_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.cfg.train.max_steps - self.cfg.train.warmup_steps,
            )

            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, decay_scheduler],
                milestones=[self.cfg.train.warmup_steps],
            )

            return [optimizer], [
                {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                    "name": "train/lr",
                }
            ]
        elif self.cfg.train.scheduler == "None":
            return [optimizer], []
        else:
            raise NotImplementedError(f"Unknown scheduler {self.cfg.train.scheduler}")

    def log_table(
        self, batch: tuple[Tensor, Tensor, Tensor, Tensor], mode: Literal["val", "test"]
    ):
        text, text_len, audio, audio_len = batch
        random_audio_index = torch.randint(0, len(audio), (1,)).item()
        random_audio_len = audio_len[[random_audio_index]]
        random_audio = audio[[random_audio_index], :, :random_audio_len]
        random_text_len = text_len[[random_audio_index]]
        random_text = text[[random_audio_index], :random_text_len]
        with torch.no_grad():
            pred = self.delayed_transformer(
                random_text,
                random_audio[:, :, :-1],
                random_text_len,
                random_audio_len - 1,
            )
            pred_sample = nucleus_sample(pred)
            pred = remove_delay(
                pred_sample, random_audio_len - 1, self.codec_channels, self.codec_pad
            )
            pred = pred.clamp_max(self.cfg.data.codec_sos - 1)
            gen = self.inference(
                text=self.sample_text,
                target_text=random_text,
                target_audio=random_audio,
                text_len=self.sample_text_len,
                target_text_len=random_text_len,
                target_audio_len=random_audio_len,
            )
            random_audio = remove_delay(
                random_audio, random_audio_len, self.codec_channels, self.codec_pad
            )

        if gen.shape[2] < 30:
            tqdm.write(f"Generated audio is too short, {gen.shape[2]} < 30")
            codec_list = [random_audio, pred]
        else:
            codec_list = [random_audio, pred, gen]

        data: list[tuple[np.ndarray, np.ndarray]] = []
        for codec in codec_list:
            audio = codec_to_audio(codec, sample_rate=self.sample_rate)
            mel = mel_spectrogram(
                audio.squeeze(1),
                n_fft=1024,
                num_mels=80,
                sampling_rate=self.sample_rate,
                hop_size=256,
                win_size=1024,
                fmin=0,
                fmax=8000,
            )
            audio_cpu = audio.squeeze().detach().cpu() * np.iinfo(np.int16).max
            audio_numpy = audio_cpu.numpy().astype(np.int16)
            mel_image = plot_mel_spectrogram(mel.squeeze().detach().cpu().numpy())
            data.append((audio_numpy, mel_image))

        if isinstance(self.logger, WandbLogger):
            self.logger.log_table(
                f"{mode}/table",
                columns=["Audio", "Mel"],
                data=[
                    [
                        wandb.Audio(audio, sample_rate=self.sample_rate),
                        wandb.Image(mel_image, mode="RGBA"),
                    ]
                    for (audio, mel_image) in data
                ],
            )
        elif isinstance(self.logger, TensorBoardLogger):
            for (audio, mel_image), name in zip(data, ["real", "pred", "gen"]):
                self.logger.experiment.add_audio(
                    f"{mode}/Audio/{name}",
                    audio / np.iinfo(np.int16).max,
                    self.global_step,
                    sample_rate=self.sample_rate,
                )
                self.logger.experiment.add_image(
                    f"{mode}/Mel/{name}",
                    mel_image,
                    self.global_step,
                    dataformats="HWC",
                )
