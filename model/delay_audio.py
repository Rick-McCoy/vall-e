import torch
from torch import nn

from config.config import Config
from utils.utils import unpad_sequence


class DelayAudio(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.n_channels = cfg.data.codec_channels
        self.register_buffer(
            "audio_sos",
            torch.full(
                (cfg.data.codec_channels,),
                fill_value=2**cfg.data.codec_bits,
                dtype=torch.long,
            ),
        )
        self.audio_sos: torch.Tensor
        self.register_buffer(
            "audio_eos",
            torch.full(
                (cfg.data.codec_channels,),
                fill_value=2**cfg.data.codec_bits + 1,
                dtype=torch.long,
            ),
        )
        self.audio_eos: torch.Tensor
        self.audio_pad = float(2**cfg.data.codec_bits + 2)

    def forward(
        self, audio: torch.Tensor, audio_len: torch.Tensor, target: bool = False
    ):
        split_audio = unpad_sequence(audio.transpose(1, 2), audio_len, batch_first=True)
        offset = 0 if target else 1
        padded_audio = [
            torch.stack(
                [
                    torch.cat(
                        [
                            self.audio_sos[: i + offset],
                            audio_item[:, i],
                            self.audio_eos[i + offset :],
                        ],
                        dim=0,
                    )
                    for i in range(self.n_channels)
                ],
                dim=0,
            )
            for audio_item in split_audio
        ]
        audio_len = audio_len + self.n_channels
        return (
            torch.nn.utils.rnn.pad_sequence(
                [padded_audio_item.T for padded_audio_item in padded_audio],
                batch_first=True,
                padding_value=self.audio_pad,
            ).transpose(1, 2),
            audio_len,
        )

    def remove_delay(self, audio: torch.Tensor, audio_len: torch.Tensor):
        split_audio = unpad_sequence(audio.transpose(1, 2), audio_len, batch_first=True)
        unpad_audio = [
            torch.stack(
                [
                    audio_item[i : i - self.n_channels, i]
                    for i in range(self.n_channels)
                ],
                dim=0,
            )
            for audio_item in split_audio
        ]
        audio_len = audio_len - self.n_channels
        return (
            torch.nn.utils.rnn.pad_sequence(
                [unpad_audio_item.T for unpad_audio_item in unpad_audio],
                batch_first=True,
                padding_value=self.audio_pad,
            ).transpose(1, 2),
            audio_len,
        )
