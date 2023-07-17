from typing import Literal

import numpy as np
import torch
import wandb
from lightning import LightningModule
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from torch import Tensor, nn
from torch.optim.lr_scheduler import LRScheduler
from torchmetrics.classification import MulticlassAccuracy
from tqdm import tqdm

from config.config import Config
from model.delay_audio import DelayAudio
from model.delayed_transformer import DelayedTransformer
from utils.audio import codec_to_audio, mel_spectrogram
from utils.data import plot_mel_spectrogram
from utils.model import nucleus_sample
from utils.text import CHAR_TO_CODE, VOCAB_SIZE, encode_text
from utils.types import CollatedBatch
from utils.utils import unpad_sequence


class MusicGen(LightningModule):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg
        self.codec_channels = cfg.data.codec_channels
        self.sample_rate = cfg.data.sample_rate
        self.text_pad = float(CHAR_TO_CODE["<PAD>"])
        self.register_buffer(
            "codec_eos",
            torch.full(
                (1, cfg.data.codec_channels, 1),
                2**cfg.data.codec_bits,
                dtype=torch.long,
            ),
        )
        self.codec_eos: Tensor
        self.lr = cfg.train.lr
        self.delay_audio = DelayAudio(cfg)
        self.delayed_transformer = DelayedTransformer(cfg)
        self.loss = nn.CrossEntropyLoss(ignore_index=2**cfg.data.codec_bits + 1)
        self.acc = MulticlassAccuracy(
            num_classes=2**cfg.data.codec_bits + 3,
            top_k=1,
            ignore_index=2**cfg.data.codec_bits + 1,
        )
        self.example_input_array = (
            torch.randint(0, VOCAB_SIZE, (2, 50)).long(),
            torch.randint(
                0, 2**cfg.data.codec_bits, (2, cfg.data.codec_channels, 300)
            ).long(),
            torch.tensor([50, 30]),
            torch.tensor([250, 300]),
        )
        self.register_buffer(
            "sample_text",
            torch.from_numpy(encode_text(self.cfg.data.sample_sentence)).unsqueeze(0),
        )
        self.sample_text: Tensor
        self.register_buffer(
            "sample_text_len",
            torch.tensor([self.sample_text.shape[1]]),
        )
        self.sample_text_len: Tensor
        self.max_infer_len = 1000

    def parse_batch(self, data: CollatedBatch):
        text = data.text.to(self.device)
        audio = data.audio.to(self.device)
        text_len = data.text_len.to(self.device)
        audio_len = data.audio_len.to(self.device)
        return text, text_len, audio, audio_len

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
        enrolled_text: Tensor,
        enrolled_audio: Tensor,
        text_len: Tensor,
        enrolled_text_len: Tensor,
        enrolled_audio_len: Tensor,
    ) -> Tensor:
        assert len(text) == 1, "Inference only supports batch size 1"
        unpad_text = unpad_sequence(text, text_len, batch_first=True)
        unpad_enrolled_text = unpad_sequence(
            enrolled_text, enrolled_text_len, batch_first=True
        )
        concat_text = torch.nn.utils.rnn.pad_sequence(
            [
                torch.cat([unpad_enrolled_text_item, unpad_text_item])
                for unpad_text_item, unpad_enrolled_text_item in zip(
                    unpad_text, unpad_enrolled_text
                )
            ],
            batch_first=True,
            padding_value=self.text_pad,
        )
        concat_text_len = text_len + 1 + enrolled_text_len
        audio = torch.empty_like(enrolled_audio)[:, :, :0]
        audio_len = torch.zeros_like(enrolled_audio_len)
        delayed_enrolled_audio, delayed_enrolled_audio_len = self.delay_audio(
            enrolled_audio, enrolled_audio_len
        )
        enrolled_audio = delayed_enrolled_audio[:, :, : 1 - self.codec_channels]
        enrolled_audio_len = delayed_enrolled_audio_len - 1 + self.codec_channels
        for _ in tqdm(range(self.max_infer_len)):
            output = self.delayed_transformer(
                concat_text,
                torch.cat([enrolled_audio, audio], dim=-1),
                concat_text_len,
                enrolled_audio_len + audio_len,
            )[0, :, -1]
            sampled_token = torch.stack(
                [nucleus_sample(output[i], 0.9) for i in range(self.codec_channels)],
                dim=0,
            ).unsqueeze(0)
            if torch.all(sampled_token == self.codec_eos):
                break
            audio = torch.cat([audio, sampled_token], dim=-1)
            audio_len += 1

        audio, _ = self.delay_audio.remove_delay(audio, audio_len)
        audio = audio.clamp_max(2**self.cfg.data.codec_bits - 1)
        return audio

    def single_step(
        self, batch: CollatedBatch, mode: Literal["train", "val", "test"]
    ) -> Tensor:
        (text, text_len, audio, audio_len) = self.parse_batch(batch)
        delayed_audio, delayed_audio_len = self.delay_audio(audio, audio_len)
        delayed_audio_len = delayed_audio_len.clamp_max(delayed_audio_len.max() - 1)
        output = self(text, delayed_audio[:, :, :-1], text_len, delayed_audio_len)
        loss = self.loss(output.permute(0, 3, 1, 2), delayed_audio[:, :, 1:])
        self.acc(output.permute(0, 3, 1, 2), delayed_audio[:, :, 1:])
        if mode == "train":
            self.log(f"{mode}/loss", loss, on_step=True)
            self.log(f"{mode}/acc", self.acc, on_step=True)
        else:
            self.log(f"{mode}/loss", loss, on_epoch=True, sync_dist=True)
            self.log(f"{mode}/acc", self.acc, on_epoch=True, sync_dist=True)
        return loss

    def training_step(self, batch: CollatedBatch, batch_idx: int) -> Tensor:
        loss = self.single_step(batch, "train")
        if self.device.index == 0:
            scheduler = self.lr_schedulers()
            if isinstance(scheduler, LRScheduler):
                self.log("train/lr", scheduler.get_last_lr()[0], on_step=True)
        return loss

    def validation_step(self, batch: CollatedBatch, batch_idx: int):
        self.single_step(batch, "val")
        if batch_idx == 0 and self.device.index == 0:
            self.log_table(batch, "val")

    def test_step(self, batch: CollatedBatch, batch_idx: int):
        self.single_step(batch, "test")
        if batch_idx == 0 and self.device.index == 0:
            self.log_table(batch, "test")

    def configure_optimizers(self):
        if self.cfg.train.optimizer == "Adam":
            optimizer = torch.optim.Adam(params=self.parameters(), lr=self.lr)
        elif self.cfg.train.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(params=self.parameters(), lr=self.lr)
        else:
            raise NotImplementedError(f"Unknown optimizer {self.cfg.train.optimizer}")

        if self.cfg.train.scheduler == "LinearDecay":

            def lr_scale(step: int) -> float:
                return min(
                    step / self.cfg.train.warmup_steps,
                    (self.cfg.train.max_steps - step)
                    / (self.cfg.train.max_steps - self.cfg.train.warmup_steps),
                )

            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lr_scale,
            )

            return [optimizer], [
                {"scheduler": scheduler, "interval": "step", "frequency": 1}
            ]
        elif self.cfg.train.scheduler == "None":
            return [optimizer], []
        else:
            raise NotImplementedError(f"Unknown scheduler {self.cfg.train.scheduler}")

    def log_table(self, batch: CollatedBatch, mode: Literal["val", "test"]):
        (text, text_len, audio, audio_len) = self.parse_batch(batch)
        longest_audio_index = audio_len.argmax().item()
        longest_audio_len = audio_len[[longest_audio_index]]
        longest_audio = audio[[longest_audio_index], :longest_audio_len]
        longest_text_len = text_len[[longest_audio_index]]
        longest_text = text[[longest_audio_index], :longest_text_len]
        with torch.no_grad():
            delayed_longest_audio, delayed_longest_audio_len = self.delay_audio(
                longest_audio, longest_audio_len
            )
            pred, _ = self.delay_audio.remove_delay(
                self(
                    longest_text,
                    delayed_longest_audio[:, :, :-1],
                    longest_text_len,
                    delayed_longest_audio_len - 1,
                ).argmax(dim=-1),
                delayed_longest_audio_len - 1,
            )
            pred = pred.clamp_max(2**self.cfg.data.codec_bits - 1)
            gen: Tensor = self.inference(
                text=self.sample_text,
                enrolled_text=longest_text,
                enrolled_audio=longest_audio,
                text_len=self.sample_text_len,
                enrolled_text_len=longest_text_len,
                enrolled_audio_len=longest_audio_len,
            )
        if gen.shape[2] < 30:
            tqdm.write(f"Generated audio is too short, {gen.shape[2]} < 30")
            codec_list = [longest_audio, pred]
        else:
            codec_list = [longest_audio, pred, gen]

        data: list[tuple[np.ndarray, np.ndarray]] = []
        for codec in codec_list:
            audio = codec_to_audio(codec, sample_rate=self.sample_rate)
            mel = mel_spectrogram(
                audio.squeeze(1),
                n_fft=1024,
                num_mels=80,
                sampling_rate=self.cfg.data.sample_rate,
                hop_size=256,
                win_size=1024,
                fmin=0,
                fmax=8000,
            )
            audio_cpu = audio.squeeze().detach().cpu() * 2**15
            audio_numpy = audio_cpu.numpy().astype(np.int16)
            mel_image = plot_mel_spectrogram(mel.squeeze().detach().cpu().numpy())
            data.append((audio_numpy, mel_image))

        if isinstance(self.logger, WandbLogger):
            self.logger.log_table(
                f"{mode}/table",
                columns=["Audio", "Mel"],
                data=[
                    [
                        wandb.Audio(audio, sample_rate=self.cfg.data.sample_rate),
                        wandb.Image(mel_image, mode="RGBA"),
                    ]
                    for (audio, mel_image) in data
                ],
            )
        elif isinstance(self.logger, TensorBoardLogger):
            for (audio, mel_image), name in zip(data, ["real", "pred", "gen"]):
                self.logger.experiment.add_audio(
                    f"{mode}/Audio/{name}",
                    audio / 2**15,
                    self.global_step,
                    sample_rate=self.cfg.data.sample_rate,
                )
                self.logger.experiment.add_image(
                    f"{mode}/Mel/{name}",
                    mel_image,
                    self.global_step,
                    dataformats="HWC",
                )
