import torch
from torch import Tensor, nn

from config.config import Config
from utils.utils import unpad_sequence


class DelayAudio(nn.Module):
    """Delays audio codecs across codec channels.

    Delaying is performed by appending SOS & EOS tokens to the beginning and end of
    each audio codec channel. The number of tokens appended is equal to the number of
    codec channels.
    Then, each channel is shifted by its channel index. For example, the first channel
    is shifted by 0, the second channel is shifted by 1, and so on.
    Finally, the audio length is increased by the number of codec channels.

    Example:
        >>> cfg.data.codec_channels = 4
        >>> delay_audio = DelayAudio(cfg)
        >>> a = torch.zeros((2, 4, 3), dtype=torch.long)
        >>> length = torch.tensor([1, 3])
        >>> delayed_audio, delayed_audio_length = delay_audio(a, length, False)
        >>> print(delayed_audio)
        tensor([[[   0, 1025, 1025, 1025, 1025, 1026, 1026],
                [1024,    0, 1025, 1025, 1025, 1026, 1026],
                [1024, 1024,    0, 1025, 1025, 1026, 1026],
                [1024, 1024, 1024,    0, 1025, 1026, 1026],
                [[   0,    0,    0, 1025, 1025, 1025, 1025],
                [1024,    0,    0,    0, 1025, 1025, 1025],
                [1024, 1024,    0,    0,    0, 1025, 1025],
                [1024, 1024, 1024,    0,    0,    0, 1025]]])
        >>> print(delayed_audio_length)
        tensor([5, 7])
        >>> original, original_length = delay_audio.remove_delay(
            delayed_audio, delayed_audio_length
        )
        >>> print(original)
        tensor([[[   0, 1026, 1026],
                [   0, 1026, 1026],
                [   0, 1026, 1026],
                [   0, 1026, 1026],
                [[   0,    0,    0],
                [   0,    0,    0],
                [   0,    0,    0],
                [   0,    0,    0]]])
        >>> print(original_length)
        tensor([1, 3])

    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.n_channels = cfg.data.codec_channels
        self.register_buffer(
            "audio_sos",
            torch.full((cfg.data.codec_channels,), fill_value=cfg.data.codec_sos),
        )
        self.audio_sos: Tensor
        self.register_buffer(
            "audio_eos",
            torch.full((cfg.data.codec_channels,), fill_value=cfg.data.codec_eos),
        )
        self.audio_eos: Tensor
        self.audio_pad = float(cfg.data.codec_pad)

    def forward(self, audio: Tensor, audio_len: Tensor):
        with torch.no_grad():
            split_audio = unpad_sequence(
                audio.transpose(1, 2), audio_len, batch_first=True
            )
            delayed = [
                torch.stack(
                    [
                        torch.cat(
                            [self.audio_sos[:i], audio_item[:, i], self.audio_eos[i:]],
                            dim=0,
                        )
                        for i in range(self.n_channels)
                    ],
                    dim=0,
                )
                for audio_item in split_audio
            ]
            audio_len = audio_len + self.n_channels
            padded_audio = torch.nn.utils.rnn.pad_sequence(
                [delayed_audio_item.T for delayed_audio_item in delayed],
                batch_first=True,
                padding_value=self.audio_pad,
            ).transpose(1, 2)

            return padded_audio, audio_len

    def remove_delay(self, audio: Tensor, audio_len: Tensor):
        with torch.no_grad():
            split_audio = unpad_sequence(
                audio.transpose(1, 2), audio_len, batch_first=True
            )
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

            padded_audio = torch.nn.utils.rnn.pad_sequence(
                [unpad_audio_item.T for unpad_audio_item in unpad_audio],
                batch_first=True,
                padding_value=self.audio_pad,
            ).transpose(1, 2)

            return padded_audio, audio_len
