from typing import Literal

import numpy as np
import torch
import wandb
from lightning import LightningModule
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from torch import Tensor
from torch.optim.lr_scheduler import LRScheduler
from torchmetrics.classification import MulticlassAccuracy
from tqdm import tqdm

from config.config import Config
from model.autoregressive import AutoRegressive
from model.loss import VallELoss
from model.nonautoregressive import NonAutoRegressive
from utils.audio import codec_to_audio, mel_spectrogram
from utils.data import plot_mel_spectrogram
from utils.model import nucleus_sample
from utils.text import CHAR_TO_CODE, VOCAB_SIZE, encode_text
from utils.types import CollatedBatch
from utils.utils import unpad_sequence


class VallE(LightningModule):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg
        self.enrolled_audio_length = (
            cfg.data.enrolled_audio_length * cfg.data.codec_rate
        )
        self.codec_channels = cfg.data.codec_channels
        self.sample_rate = cfg.data.sample_rate
        self.register_buffer("text_eos", torch.tensor([CHAR_TO_CODE["<EOS>"]]))
        self.text_eos: Tensor
        self.text_pad = float(CHAR_TO_CODE["<PAD>"])
        self.register_buffer(
            "codec_eos",
            torch.full(
                (1, cfg.data.codec_channels), 2**cfg.data.codec_bits, dtype=torch.long
            ),
        )
        self.codec_eos: Tensor
        self.codec_pad = float(2**cfg.data.codec_bits + 1)
        self.lr = cfg.train.lr
        self.autoregressive = AutoRegressive(cfg)
        self.nonautoregressive = NonAutoRegressive(cfg)
        self.loss = VallELoss(cfg)
        self.ar_acc = MulticlassAccuracy(
            num_classes=2**cfg.data.codec_bits + 2,
            top_k=1,
            ignore_index=2**cfg.data.codec_bits + 1,
        )
        self.nar_acc = MulticlassAccuracy(
            num_classes=2**cfg.data.codec_bits + 2,
            top_k=1,
            ignore_index=2**cfg.data.codec_bits + 1,
        )
        self.example_input_array = (
            torch.randint(0, VOCAB_SIZE, (2, 50)).long(),
            torch.randint(
                0, 2**cfg.data.codec_bits, (2, cfg.data.codec_channels, 300)
            ).long(),
            torch.randint(
                0, 2**cfg.data.codec_bits, (2, cfg.data.codec_channels, 200)
            ).long(),
            torch.tensor([50, 30]),
            torch.tensor([250, 300]),
            torch.tensor([200, 200]),
        )
        self.register_buffer(
            "gen_text",
            torch.from_numpy(encode_text(self.cfg.data.sample_sentence)).unsqueeze(0),
        )
        self.gen_text: Tensor
        self.register_buffer("gen_text_len", torch.tensor([self.gen_text.shape[1]]))
        self.gen_text_len: Tensor
        self.max_infer_len = 1000

    def parse_batch(self, data: CollatedBatch):
        text = data.text.to(self.device)
        audio = data.audio.to(self.device)
        enrolled_audio = data.enrolled_audio.to(self.device)
        text_len = data.text_len.to(self.device)
        audio_len = data.audio_len.to(self.device)
        enrolled_audio_len = data.enrolled_audio_len.to(self.device)
        return text, text_len, audio, audio_len, enrolled_audio, enrolled_audio_len

    def add_text_eos(self, text: Tensor, text_len: Tensor):
        text_list = unpad_sequence(text, text_len, batch_first=True)
        text_list = [torch.cat([t, self.text_eos]) for t in text_list]
        text_len = text_len + 1
        text = torch.nn.utils.rnn.pad_sequence(
            text_list, batch_first=True, padding_value=self.text_pad
        )
        return text, text_len

    def add_codec_eos(self, codec: Tensor, codec_len: Tensor):
        codec_list = unpad_sequence(codec.transpose(1, 2), codec_len, batch_first=True)
        codec_list = [torch.cat([t, self.codec_eos]) for t in codec_list]
        codec_len = codec_len + 1
        codec = torch.nn.utils.rnn.pad_sequence(
            codec_list, batch_first=True, padding_value=self.codec_pad
        )
        return codec.transpose(1, 2), codec_len

    def remove_codec_eos(self, codec: Tensor, codec_len: Tensor):
        codec_list = unpad_sequence(codec, codec_len, batch_first=True)
        codec_list = [t[:-1] for t in codec_list]
        codec_len = codec_len - 1
        codec = torch.nn.utils.rnn.pad_sequence(
            codec_list, batch_first=True, padding_value=self.codec_pad
        )
        return codec, codec_len

    def slice_audio(self, audio: Tensor, audio_len: Tensor):
        audio_slice_list = []
        audio_slice_len_list: list[int] = []
        for audio_item, audio_len_item in zip(audio, audio_len):
            if audio_len_item >= self.enrolled_audio_length:
                random_index = torch.randint(
                    int(audio_len_item.item()) - self.enrolled_audio_length + 1, size=()
                )
                audio_slice_list.append(
                    audio_item[
                        :, random_index : random_index + self.enrolled_audio_length
                    ].T
                )
                audio_slice_len_list.append(self.enrolled_audio_length)
            else:
                audio_slice_list.append(audio_item.T)
                audio_slice_len_list.append(int(audio_len_item.item()))
        audio_slice = torch.nn.utils.rnn.pad_sequence(
            audio_slice_list,
            batch_first=True,
            padding_value=self.codec_pad,
        ).transpose(1, 2)
        audio_slice_len = torch.tensor(audio_slice_len_list).long().to(audio.device)
        return audio_slice, audio_slice_len

    def ar_forward(
        self,
        text: Tensor,
        audio: Tensor,
        text_len: Tensor,
        audio_len: Tensor,
    ) -> Tensor:
        text, text_len = self.add_text_eos(text, text_len)
        ar_output = self.autoregressive(text, audio[:, 0], text_len, audio_len)
        ar_output_list = []
        for ar_output_item, text_len_item, audio_len_item in zip(
            ar_output, text_len, audio_len
        ):
            ar_output_list.append(
                ar_output_item[text_len_item - 1 : text_len_item + audio_len_item]
            )
        return torch.nn.utils.rnn.pad_sequence(
            ar_output_list, batch_first=True, padding_value=self.codec_pad
        )

    def nar_forward(
        self,
        text: Tensor,
        audio: Tensor,
        enrolled_audio: Tensor,
        text_len: Tensor,
        audio_len: Tensor,
        enrolled_audio_len: Tensor,
    ) -> Tensor:
        text, text_len = self.add_text_eos(text, text_len)
        enrolled_audio, enrolled_audio_len = self.slice_audio(
            enrolled_audio, enrolled_audio_len
        )
        enrolled_audio, enrolled_audio_len = self.add_codec_eos(
            enrolled_audio, enrolled_audio_len
        )
        nar_output = self.nonautoregressive(
            text,
            audio,
            enrolled_audio,
            text_len,
            audio_len,
            enrolled_audio_len,
        )
        nar_output_list = []
        for (
            nar_output_item,
            text_len_item,
            audio_len_item,
            enrolled_audio_len_item,
        ) in zip(nar_output, text_len, audio_len, enrolled_audio_len):
            nar_output_list.append(
                nar_output_item[
                    text_len_item
                    + enrolled_audio_len_item : text_len_item
                    + enrolled_audio_len_item
                    + audio_len_item
                ]
            )
        return torch.nn.utils.rnn.pad_sequence(
            nar_output_list, batch_first=True, padding_value=self.codec_pad
        )

    def forward(
        self,
        text: Tensor,
        audio: Tensor,
        enrolled_audio: Tensor,
        text_len: Tensor,
        audio_len: Tensor,
        enrolled_audio_len: Tensor,
    ) -> Tensor:
        output_list = [
            self.remove_codec_eos(
                self.ar_forward(text, audio, text_len, audio_len), audio_len + 1
            )[0]
        ]
        for channel in range(1, self.codec_channels):
            output_list.append(
                self.nar_forward(
                    text,
                    audio[:, :channel],
                    enrolled_audio,
                    text_len,
                    audio_len,
                    enrolled_audio_len,
                )
            )

        return torch.stack(output_list, dim=1)

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
                torch.cat([unpad_enrolled_text_item, unpad_text_item, self.text_eos])
                for unpad_text_item, unpad_enrolled_text_item in zip(
                    unpad_text, unpad_enrolled_text
                )
            ],
            batch_first=True,
            padding_value=self.text_pad,
        )
        concat_text_len = text_len + enrolled_text_len + 1
        audio = torch.empty_like(enrolled_audio)[:, 0, :0]
        audio_len = torch.zeros_like(enrolled_audio_len)
        for _ in tqdm(range(self.max_infer_len)):
            ar_output = self.autoregressive(
                concat_text,
                torch.cat([enrolled_audio[:, 0], audio], dim=-1),
                concat_text_len,
                enrolled_audio_len + audio_len,
            )[0, -1]
            sample_token = nucleus_sample(ar_output, 0.9).unsqueeze(0)
            if sample_token == self.codec_eos[:, 0]:
                break
            audio = torch.cat([audio, sample_token], dim=-1)
            audio_len += 1

        audio = audio.unsqueeze(1)
        for channel in range(1, self.codec_channels):
            nar_audio = self.nar_forward(
                text,
                audio[:, :channel],
                enrolled_audio,
                text_len,
                audio_len,
                enrolled_audio_len,
            )[..., :-1].argmax(dim=-1)
            audio = torch.cat([audio, nar_audio.unsqueeze(1)], dim=1)

        return audio

    def single_step(
        self, batch: CollatedBatch, mode: Literal["train", "val", "test"]
    ) -> Tensor:
        (
            text,
            text_len,
            audio,
            audio_len,
            enrolled_audio,
            enrolled_audio_len,
        ) = self.parse_batch(batch)
        ar_output = self.ar_forward(text, audio, text_len, audio_len)
        random_channel = torch.randint(1, self.codec_channels, size=())
        nar_output = self.nar_forward(
            text,
            audio[:, :random_channel],
            enrolled_audio,
            text_len,
            audio_len,
            enrolled_audio_len,
        )
        eos_audio, _ = self.add_codec_eos(audio, audio_len)
        ar_loss = self.loss(ar_output.transpose(1, 2), eos_audio[:, 0])
        nar_loss = self.loss(nar_output.transpose(1, 2), audio[:, random_channel])
        total_loss = (ar_loss + nar_loss) / 2
        self.ar_acc(ar_output.transpose(1, 2), eos_audio[:, 0])
        self.nar_acc(nar_output.transpose(1, 2), audio[:, random_channel])
        if mode == "train":
            self.log(f"{mode}/ar_loss", ar_loss, on_step=True)
            self.log(f"{mode}/nar_loss", nar_loss, on_step=True)
            self.log(f"{mode}/loss", total_loss, on_step=True)
            self.log(f"{mode}/ar_acc", self.ar_acc, on_step=True)
            self.log(f"{mode}/nar_acc", self.nar_acc, on_step=True)
        else:
            self.log(f"{mode}/ar_loss", ar_loss, on_epoch=True, sync_dist=True)
            self.log(f"{mode}/nar_loss", nar_loss, on_epoch=True, sync_dist=True)
            self.log(f"{mode}/loss", total_loss, on_epoch=True, sync_dist=True)
            self.log(f"{mode}/ar_acc", self.ar_acc, on_epoch=True, sync_dist=True)
            self.log(f"{mode}/nar_acc", self.nar_acc, on_epoch=True, sync_dist=True)
        return total_loss

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
        (
            text,
            text_len,
            audio,
            audio_len,
            enrolled_audio,
            enrolled_audio_len,
        ) = self.parse_batch(batch)
        longest_audio_index = audio_len.argmax().item()
        longest_audio_len = audio_len[[longest_audio_index]]
        longest_audio = audio[[longest_audio_index], :longest_audio_len]
        longest_text_len = text_len[[longest_audio_index]]
        longest_text = text[[longest_audio_index], :longest_text_len]
        longest_enrolled_audio_len = enrolled_audio_len[[longest_audio_index]]
        longest_enrolled_audio = enrolled_audio[
            [longest_audio_index], :longest_enrolled_audio_len
        ]
        with torch.no_grad():
            pred = self(
                longest_text,
                longest_audio,
                longest_enrolled_audio,
                longest_text_len,
                longest_audio_len,
                longest_enrolled_audio_len,
            )[..., :-2].argmax(dim=-1)
            gen: Tensor = self.inference(
                text=self.gen_text,
                enrolled_text=longest_text,
                enrolled_audio=longest_audio,
                text_len=self.gen_text_len,
                enrolled_text_len=longest_text_len,
                enrolled_audio_len=longest_audio_len,
            )
        if gen.shape[2] < 30:
            tqdm.write(f"Warning: Generated audio is too short, {gen.shape[2]} < 30")
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
