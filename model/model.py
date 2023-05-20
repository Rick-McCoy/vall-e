from typing import cast

import numpy as np
import torch
from lightning import LightningModule
from lightning.pytorch.loggers import WandbLogger
from torch import Tensor
from torch.optim.lr_scheduler import LRScheduler
from torchmetrics.classification import MulticlassAccuracy

import wandb
from config.config import Config
from data.audio import codec_to_audio, mel_energy
from data.datamodule import CollatedBatch
from data.text import CHAR_TO_CODE, VOCAB_SIZE
from encodec.model import EncodecModel
from model.autoregressive import AutoRegressive
from model.loss import VallELoss
from model.nonautoregressive import NonAutoRegressive
from utils.data import plot_mel_spectrogram
from utils.model import remove_weight_norm
from utils.utils import unpad_sequence


class VallE(LightningModule):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.codec_channels = cfg.data.codec_channels
        self.register_buffer(
            "text_eos", torch.tensor([CHAR_TO_CODE["<EOS>"]], dtype=torch.long)
        )
        self.text_eos: torch.Tensor
        self.text_pad = CHAR_TO_CODE["<PAD>"]
        self.register_buffer(
            "codec_eos",
            torch.full(
                (1, cfg.data.codec_channels), 2**cfg.data.codec_bits, dtype=torch.long
            ),
        )
        self.codec_eos: torch.Tensor
        self.codec_pad = 2**cfg.data.codec_bits + 1
        self.lr = cfg.train.lr
        self.autoregressive = AutoRegressive(cfg)
        self.nonautoregressive = NonAutoRegressive(cfg)
        self.loss = VallELoss(cfg)
        self.acc = MulticlassAccuracy(
            num_classes=2**cfg.data.codec_bits + 2,
            top_k=1,
            ignore_index=2**cfg.data.codec_bits + 1,
        )
        self.example_input_array = (
            torch.randint(0, VOCAB_SIZE, (2, 10)).long(),
            torch.randint(
                0, 2**cfg.data.codec_bits, (2, cfg.data.codec_channels, 30)
            ).long(),
            torch.randint(
                0,
                2**cfg.data.codec_bits,
                (2, cfg.data.codec_channels, 50),
            ).long(),
            torch.tensor([5, 10]),
            torch.tensor([30, 10]),
            torch.tensor([40, 50]),
        )
        self.logger: WandbLogger
        if cfg.data.sample_rate == 24000:
            encodec_model = EncodecModel.encodec_model_24khz()
        elif cfg.data.sample_rate == 48000:
            encodec_model = EncodecModel.encodec_model_48khz()
        else:
            raise NotImplementedError(
                f"Sample rate {cfg.data.sample_rate} not supported"
            )
        remove_weight_norm(encodec_model)
        self.encodec_model = cast(
            EncodecModel,
            torch.jit.script(  # pyright: ignore [reportPrivateImportUsage]
                encodec_model
            ),
        )
        self.rng = np.random.default_rng()

    def log_table(self, real_codec: Tensor, gen_codec: Tensor, header: str):
        real_audio = codec_to_audio(real_codec, self.encodec_model)
        gen_audio = codec_to_audio(gen_codec, self.encodec_model)
        real_mel, _ = mel_energy(
            real_audio.squeeze(1),
            n_fft=1024,
            num_mels=80,
            sampling_rate=self.cfg.data.sample_rate,
            hop_size=256,
            win_size=1024,
            fmin=0,
            fmax=8000,
        )
        gen_mel, _ = mel_energy(
            gen_audio.squeeze(1),
            n_fft=1024,
            num_mels=80,
            sampling_rate=self.cfg.data.sample_rate,
            hop_size=256,
            win_size=1024,
            fmin=0,
            fmax=8000,
        )
        real_audio_cpu = real_audio.squeeze().detach().cpu() * 2**15
        gen_audio_cpu = gen_audio.squeeze().detach().cpu() * 2**15
        real_audio_numpy = real_audio_cpu.numpy().astype(np.int16)
        gen_audio_numpy = gen_audio_cpu.numpy().astype(np.int16)
        real_mel_image = plot_mel_spectrogram(real_mel.squeeze().detach().cpu().numpy())
        gen_mel_image = plot_mel_spectrogram(gen_mel.squeeze().detach().cpu().numpy())
        self.logger.log_table(
            f"{header}/table",
            columns=["Audio", "Mel"],
            data=[
                [
                    wandb.Audio(
                        real_audio_numpy, sample_rate=self.cfg.data.sample_rate
                    ),
                    wandb.Image(real_mel_image),
                ],
                [
                    wandb.Audio(gen_audio_numpy, sample_rate=self.cfg.data.sample_rate),
                    wandb.Image(gen_mel_image),
                ],
            ],
        )

    def parse_batch(self, data: CollatedBatch):
        text = data.text.to(self.device)
        audio = data.audio.to(self.device)
        enrolled_audio = data.enrolled_audio.to(self.device)
        text_len = data.text_len.to(self.device)
        audio_len = data.audio_len.to(self.device)
        enrolled_audio_len = data.enrolled_audio_len.to(self.device)
        return text, text_len, audio, audio_len, enrolled_audio, enrolled_audio_len

    def add_text_eos(self, text: torch.Tensor, text_len: torch.Tensor):
        text_list = unpad_sequence(text, text_len, batch_first=True)
        text_list = [torch.cat([t, self.text_eos]) for t in text_list]
        text_len = text_len + 1
        text = torch.nn.utils.rnn.pad_sequence(
            text_list, batch_first=True, padding_value=float(self.text_pad)
        )
        return text, text_len

    def add_codec_eos(self, codec: torch.Tensor, codec_len: torch.Tensor):
        codec_list = unpad_sequence(codec.transpose(1, 2), codec_len, batch_first=True)
        codec_list = [torch.cat([t, self.codec_eos]) for t in codec_list]
        codec_len = codec_len + 1
        codec = torch.nn.utils.rnn.pad_sequence(
            codec_list, batch_first=True, padding_value=float(self.codec_pad)
        )
        return codec.transpose(1, 2), codec_len

    def ar_forward(
        self,
        text: Tensor,
        audio: Tensor,
        text_len_batch: Tensor,
        audio_len_batch: Tensor,
    ) -> Tensor:
        text, text_len_batch = self.add_text_eos(text, text_len_batch)
        ar_output_batch = self.autoregressive(
            text, audio[:, 0], text_len_batch, audio_len_batch
        )
        ar_output_split = []
        for ar_output, text_len, audio_len in zip(
            ar_output_batch, text_len_batch, audio_len_batch
        ):
            ar_output_split.append(ar_output[text_len - 1 : text_len + audio_len])
        return torch.nn.utils.rnn.pad_sequence(
            ar_output_split, batch_first=True, padding_value=float(self.codec_pad)
        )

    def nar_forward(
        self,
        text: Tensor,
        audio: Tensor,
        enrolled_audio: Tensor,
        text_len_batch: Tensor,
        audio_len_batch: Tensor,
        enrolled_audio_len_batch: Tensor,
        channel: int,
    ) -> Tensor:
        text, text_len_batch = self.add_text_eos(text, text_len_batch)
        enrolled_audio, enrolled_audio_len_batch = self.add_codec_eos(
            enrolled_audio, enrolled_audio_len_batch
        )
        nar_output_batch = self.nonautoregressive(
            text,
            audio[:, :channel],
            enrolled_audio,
            text_len_batch,
            audio_len_batch,
            enrolled_audio_len_batch,
            channel,
        )
        nar_output_split = []
        for nar_output, text_len, audio_len, enrolled_audio_len in zip(
            nar_output_batch, text_len_batch, audio_len_batch, enrolled_audio_len_batch
        ):
            nar_output_split.append(
                nar_output[
                    text_len
                    + enrolled_audio_len : text_len
                    + enrolled_audio_len
                    + audio_len
                    + 1
                ]
            )
        return torch.nn.utils.rnn.pad_sequence(
            nar_output_split, batch_first=True, padding_value=float(self.codec_pad)
        )

    def forward(
        self,
        text: Tensor,
        audio: Tensor,
        enrolled_audio: Tensor,
        text_len_batch: Tensor,
        audio_len_batch: Tensor,
        enrolled_audio_len_batch: Tensor,
    ) -> Tensor:
        ar_output = self.ar_forward(text, audio, text_len_batch, audio_len_batch)
        nar_output_list = []
        for channel in range(1, self.codec_channels):
            nar_output_list.append(
                self.nar_forward(
                    text,
                    audio,
                    enrolled_audio,
                    text_len_batch,
                    audio_len_batch,
                    enrolled_audio_len_batch,
                    channel,
                )
            )

        return torch.stack([ar_output] + nar_output_list, dim=1)

    def training_step(self, batch: CollatedBatch) -> Tensor:
        (
            text,
            text_len,
            audio,
            audio_len,
            enrolled_audio,
            enrolled_audio_len,
        ) = self.parse_batch(batch)
        ar_output = self.ar_forward(text, audio, text_len, audio_len)
        random_channel = self.rng.choice(self.cfg.data.codec_channels - 1) + 1
        nar_output = self.nar_forward(
            text,
            audio,
            enrolled_audio,
            text_len,
            audio_len,
            enrolled_audio_len,
            random_channel,
        )
        output = torch.einsum("bnlc->bcnl", torch.stack([ar_output, nar_output], dim=1))
        eos_audio, _ = self.add_codec_eos(audio, audio_len)
        loss = self.loss(output, eos_audio[:, [0, random_channel]])
        self.acc(output, eos_audio[:, [0, random_channel]])
        self.log("train/loss", loss, on_step=True)
        self.log("train/acc", self.acc, on_step=True)
        if self.device.index == 0:
            scheduler = self.lr_schedulers()
            assert isinstance(scheduler, LRScheduler)
            self.log("train/lr", scheduler.get_last_lr()[0], on_step=True)
        return loss

    def validation_step(self, batch: CollatedBatch, batch_idx: int):
        (
            text,
            text_len,
            audio,
            audio_len,
            enrolled_audio,
            enrolled_audio_len,
        ) = self.parse_batch(batch)
        ar_output = self.ar_forward(text, audio, text_len, audio_len)
        random_channel = self.rng.choice(self.cfg.data.codec_channels - 1) + 1
        nar_output = self.nar_forward(
            text,
            audio,
            enrolled_audio,
            text_len,
            audio_len,
            enrolled_audio_len,
            random_channel,
        )
        output = torch.einsum("bnlc->bcnl", torch.stack([ar_output, nar_output], dim=1))
        eos_audio, _ = self.add_codec_eos(audio, audio_len)
        loss = self.loss(output, eos_audio[:, [0, random_channel]])
        self.acc(output, eos_audio[:, [0, random_channel]])
        self.log("val/loss", loss, on_epoch=True, sync_dist=True)
        self.log("val/acc", self.acc, on_epoch=True, sync_dist=True)
        if batch_idx == 0 and self.device.index == 0:
            with torch.no_grad():
                pred = self(
                    text, audio, enrolled_audio, text_len, audio_len, enrolled_audio_len
                )[..., : self.codec_pad - 1].argmax(dim=-1)
            longest_audio_index = audio_len.argmax()
            longest_audio_len = audio_len[longest_audio_index]
            longest_audio = audio[longest_audio_index, :longest_audio_len].unsqueeze(0)
            longest_pred = pred[longest_audio_index, :longest_audio_len].unsqueeze(0)
            self.log_table(longest_audio, longest_pred, "val")

    def test_step(self, batch: CollatedBatch, batch_idx: int):
        (
            text,
            text_len,
            audio,
            audio_len,
            enrolled_audio,
            enrolled_audio_len,
        ) = self.parse_batch(batch)
        ar_output = self.ar_forward(text, audio, text_len, audio_len)
        random_channel = self.rng.choice(self.cfg.data.codec_channels - 1) + 1
        nar_output = self.nar_forward(
            text,
            audio,
            enrolled_audio,
            text_len,
            audio_len,
            enrolled_audio_len,
            random_channel,
        )
        output = torch.einsum("bnlc->bcnl", torch.stack([ar_output, nar_output], dim=1))
        eos_audio, _ = self.add_codec_eos(audio, audio_len)
        loss = self.loss(output, eos_audio[:, [0, random_channel]])
        self.acc(output, eos_audio[:, [0, random_channel]])
        self.log("test/loss", loss, on_epoch=True, sync_dist=True)
        self.log("test/acc", self.acc, on_epoch=True, sync_dist=True)
        if batch_idx == 0 and self.device.index == 0:
            with torch.no_grad():
                pred = self(
                    text, audio, enrolled_audio, text_len, audio_len, enrolled_audio_len
                )[..., : self.codec_pad - 1].argmax(dim=-1)
            longest_audio_index = audio_len.argmax()
            longest_audio_len = audio_len[longest_audio_index]
            longest_audio = audio[longest_audio_index, :longest_audio_len].unsqueeze(0)
            longest_pred = pred[longest_audio_index, :longest_audio_len].unsqueeze(0)
            self.log_table(longest_audio, longest_pred, "test")

    def configure_optimizers(self):
        if self.cfg.train.optimizer == "Adam":
            optimizer = torch.optim.Adam(params=self.parameters(), lr=self.lr)
        elif self.cfg.train.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(params=self.parameters(), lr=self.lr)
        else:
            raise NotImplementedError(f"Unknown optimizer {self.cfg.train.optimizer}")

        def lr_scale(epoch: int) -> float:
            return min(
                self.global_step / self.cfg.train.warmup_steps,
                (self.cfg.train.max_steps - self.global_step)
                / (self.cfg.train.max_steps - self.cfg.train.warmup_steps),
            )

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lr_scale,
        )

        return [optimizer], [
            {"scheduler": scheduler, "interval": "step", "frequency": 1}
        ]
