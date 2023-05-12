import torch
import wandb
from lightning import LightningModule
from lightning.pytorch.loggers import WandbLogger
from torch import Tensor
from torchmetrics.classification import MulticlassAccuracy

from config.config import Config
from data.datamodule import CollatedBatch
from model.autoregressive import AutoRegressive
from model.loss import SimpleLoss
from model.nonautoregressive import NonAutoRegressive


class VallE(LightningModule):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.learning_rate = cfg.train.lr
        self.autoregressive = AutoRegressive(cfg)
        self.nonautoregressive = NonAutoRegressive(cfg)
        self.loss = SimpleLoss(cfg)
        self.acc = MulticlassAccuracy(num_classes=cfg.model.num_classes, top_k=1)
        self.example_input_array = torch.zeros(
            (1, cfg.model.input_channels, cfg.model.h, cfg.model.w)
        )
        self.logger: WandbLogger

    def log_table(self, real_audio: Tensor, gen_audio: Tensor, header: str):
        self.logger.log_table(
            f"{header}/table",
            columns=["Real", "Generated"],
            data=[
                wandb.Audio(real_audio, sample_rate=self.cfg.data.sample_rate),
                wandb.Audio(gen_audio, sample_rate=self.cfg.data.sample_rate),
            ],
        )

    def parse_batch(self, data: CollatedBatch):
        text = data.text.to(self.device)
        audio = data.audio.to(self.device)
        enrolled_audio = data.enrolled_audio.to(self.device)
        text_len = data.text_len.to(self.device)
        audio_len = data.audio_len.to(self.device)
        return text, text_len, audio, audio_len, enrolled_audio

    def forward(
        self,
        text: Tensor,
        audio: Tensor,
        enrolled_audio: Tensor,
        text_len_batch: Tensor,
        audio_len_batch: Tensor,
        *args,
        **kwargs,
    ) -> Tensor:
        ar_output_batch = self.autoregressive(
            text, audio, enrolled_audio[:, 0], text_len_batch, audio_len_batch
        )
        ar_output_split = []
        for ar_output, text_len, audio_len in zip(
            ar_output_batch, text_len_batch, audio_len_batch
        ):
            text_len_item = int(text_len.item())
            audio_len_item = int(audio_len.item())
            ar_output = torch.nn.functional.pad(
                ar_output[
                    text_len_item
                    + self.cfg.data.enrolled_codec_len
                    - 1 : text_len_item
                    + self.cfg.data.enrolled_codec_len
                    + audio_len_item
                    - 1
                ].argmax(dim=-1),
                (0, audio.shape[0] - audio_len_item),
            )
            ar_output_split.append(ar_output)
        ar_padded_output = torch.stack(ar_output_split, dim=0)

        nar_output_list = []
        for i in range(1, self.cfg.data.codec_channels):
            nar_output_batch = self.nonautoregressive(
                text,
                audio[:, i - 1],
                enrolled_audio,
                text_len_batch,
                audio_len_batch,
                i,
            )
            nar_output_split = []
            for nar_output, text_len, audio_len in zip(
                nar_output_batch, text_len_batch, audio_len_batch
            ):
                text_len_item = int(text_len.item())
                audio_len_item = int(audio_len.item())
                nar_output = torch.nn.functional.pad(
                    nar_output[
                        text_len_item
                        + self.cfg.data.enrolled_codec_len : text_len_item
                        + self.cfg.data.enrolled_codec_len
                        + audio_len_item
                    ].argmax(dim=-1),
                    (0, self.cfg.data.max_audio_len - audio_len_item),
                )
                nar_output_split.append(nar_output)
            nar_output_batch = torch.stack(nar_output_split, dim=0)
            nar_output_list.append(nar_output_batch)

        return torch.cat([ar_padded_output] + nar_output_list, dim=1)

    def training_step(self, batch: CollatedBatch, *args, **kwargs) -> Tensor:
        text, text_len, audio, audio_len, enrolled_audio = self.parse_batch(batch)
        output = self(text, audio, enrolled_audio, text_len, audio_len)
        loss = self.loss(output, audio)
        self.acc(output, audio)
        self.log("train/loss", loss, on_epoch=True, sync_dist=True)
        self.log("train/acc", self.acc, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch: CollatedBatch, batch_idx: int, *args, **kwargs):
        text, text_len, audio, audio_len, enrolled_audio = self.parse_batch(batch)
        output = self(text, audio, enrolled_audio, text_len, audio_len)
        loss = self.loss(output, audio)
        self.log("val/loss", loss, on_epoch=True, sync_dist=True)
        self.log("val/acc", self.acc, on_epoch=True, sync_dist=True)
        if batch_idx == 0 and self.device.index == 0:
            pred = torch.argmax(output[0], dim=-1)
            self.log_table(audio[0], pred, "val")

    def test_step(self, batch: CollatedBatch, batch_idx: int, *args, **kwargs):
        text, text_len, audio, audio_len, enrolled_audio = self.parse_batch(batch)
        output = self(text, audio, enrolled_audio, text_len, audio_len)
        loss = self.loss(output, audio)
        self.log("test/loss", loss, on_epoch=True, sync_dist=True)
        self.log("test/acc", self.acc, on_epoch=True, sync_dist=True)
        if batch_idx == 0 and self.device.index == 0:
            pred = torch.argmax(output[0], dim=-1)
            self.log_table(audio[0], pred, "val")

    def configure_optimizers(self):
        return torch.optim.Adam(
            params=self.parameters(), lr=self.learning_rate, amsgrad=True
        )
