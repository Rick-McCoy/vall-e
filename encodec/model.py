# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""EnCodec model implementation."""

import math
import typing as tp
from pathlib import Path

import numpy as np
import torch
from torch import nn

from encodec.modules.seanet import SEANetDecoder, SEANetEncoder
from encodec.quantization.vq import ResidualVectorQuantizer
from encodec.utils import _check_checksum, _get_checkpoint_url, _linear_overlap_add

ROOT_URL = "https://dl.fbaipublicfiles.com/encodec/v0/"

EncodedFrame = tp.Tuple[torch.Tensor, tp.Optional[torch.Tensor]]


class EncodecModel(nn.Module):
    """EnCodec model operating on the raw waveform.
    Args:
        target_bandwidths (list of float): Target bandwidths.
        encoder (nn.Module): Encoder network.
        decoder (nn.Module): Decoder network.
        sample_rate (int): Audio sample rate.
        channels (int): Number of audio channels.
        normalize (bool): Whether to apply audio normalization.
        segment (float or None): segment duration in sec. when doing overlap-add.
        overlap (float): overlap between segment, given as a fraction of the segment duration.
        name (str): name of the model, used as metadata when compressing audio.
    """

    def __init__(
        self,
        encoder: SEANetEncoder,
        decoder: SEANetDecoder,
        quantizer: ResidualVectorQuantizer,
        target_bandwidths: tp.List[float],
        sample_rate: int,
        channels: int,
        normalize: bool = False,
        segment: tp.Optional[float] = None,
        overlap: float = 0.01,
        name: str = "unset",
    ):
        super().__init__()
        self.bandwidth: tp.Optional[float] = None
        self.target_bandwidths = target_bandwidths
        self.encoder = encoder
        self.quantizer = quantizer
        self.decoder = decoder
        self.sample_rate = sample_rate
        self.channels = channels
        self.normalize = normalize
        self.segment = segment
        self.overlap = overlap
        self.frame_rate = math.ceil(self.sample_rate / np.prod(self.encoder.ratios))
        self.name = name
        self.bits_per_codebook = int(math.log2(self.quantizer.bins))
        assert (
            2**self.bits_per_codebook == self.quantizer.bins
        ), "quantizer bins must be a power of 2."

    @property
    def segment_length(self) -> tp.Optional[int]:
        if self.segment is None:
            return None
        return int(self.segment * self.sample_rate)

    @property
    def segment_stride(self) -> tp.Optional[int]:
        segment_length = self.segment_length
        if segment_length is None:
            return None
        return max(1, int((1 - self.overlap) * segment_length))

    def encode(self, x: torch.Tensor) -> tp.List[EncodedFrame]:
        """Given a tensor `x`, returns a list of frames containing
        the discrete encoded codes for `x`, along with rescaling factors
        for each segment, when `self.normalize` is True.

        Each frames is a tuple `(codebook, scale)`, with `codebook` of
        shape `[B, K, T]`, with `K` the number of codebooks.
        """
        assert x.dim() == 3
        _, channels, length = x.shape
        assert channels > 0 and channels <= 2
        segment_length = self.segment_length
        if segment_length is None:
            segment_length = length
            stride = length
        else:
            stride = self.segment_stride  # type: ignore
            assert stride is not None

        encoded_frames: tp.List[EncodedFrame] = []
        for offset in range(0, length, stride):
            frame = x[:, :, offset : offset + segment_length]
            encoded_frames.append(self._encode_frame(frame))
        return encoded_frames

    def _encode_frame(self, x: torch.Tensor) -> EncodedFrame:
        length = x.shape[-1]
        duration = length / self.sample_rate
        segment = self.segment
        # assert segment is None or duration <= 1e-5 + segment
        if segment is not None:
            assert duration <= 1e-5 + segment

        if self.normalize:
            mono = x.mean(dim=1, keepdim=True)
            volume = mono.pow(2).mean(dim=2, keepdim=True).sqrt()
            scale = 1e-8 + volume
            x = x / scale
            scale = scale.view(-1, 1)
        else:
            scale = None

        emb = self.encoder(x)
        codes = self.quantizer.encode(emb, self.frame_rate, self.bandwidth)
        codes = codes.transpose(0, 1)
        # codes is [B, K, T], with T frames, K nb of codebooks.
        return codes, scale

    def decode(self, encoded_frames: tp.List[EncodedFrame]) -> torch.Tensor:
        """Decode the given frames into a waveform.
        Note that the output might be a bit bigger than the input. In that case,
        any extra steps at the end can be trimmed.
        """
        segment_length = self.segment_length
        if segment_length is None:
            assert len(encoded_frames) == 1
            return self._decode_frame(encoded_frames[0])

        frames = [self._decode_frame(frame) for frame in encoded_frames]
        segment_stride = self.segment_stride
        return _linear_overlap_add(
            frames, segment_stride if segment_stride is not None else 1
        )

    def _decode_frame(self, encoded_frame: EncodedFrame) -> torch.Tensor:
        codes, scale = encoded_frame
        codes = codes.transpose(0, 1)
        emb = self.quantizer.decode(codes)
        out = self.decoder(emb)
        if scale is not None:
            out = out * scale.view(-1, 1, 1)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        frames = self.encode(x)
        return self.decode(frames)[:, :, : x.shape[-1]]

    def set_target_bandwidth(self, bandwidth: float):
        if bandwidth not in self.target_bandwidths:
            raise ValueError(
                f"This model doesn't support the bandwidth {bandwidth}. "
                f"Select one of {self.target_bandwidths}."
            )
        self.bandwidth = bandwidth

    @staticmethod
    def _get_model(
        target_bandwidths: tp.List[float],
        sample_rate: int = 24_000,
        channels: int = 1,
        causal: bool = True,
        model_norm: str = "weight_norm",
        audio_normalize: bool = False,
        segment: tp.Optional[float] = None,
        name: str = "unset",
    ):
        encoder = SEANetEncoder(channels=channels, norm=model_norm, causal=causal)
        decoder = SEANetDecoder(channels=channels, norm=model_norm, causal=causal)
        n_q = int(
            1000
            * target_bandwidths[-1]
            // (math.ceil(sample_rate / encoder.hop_length) * 10)
        )
        quantizer = ResidualVectorQuantizer(
            dimension=encoder.dimension,
            n_q=n_q,
            bins=1024,
        )
        model = EncodecModel(
            encoder,
            decoder,
            quantizer,
            target_bandwidths,
            sample_rate,
            channels,
            normalize=audio_normalize,
            segment=segment,
            name=name,
        )
        return model

    @staticmethod
    def _get_pretrained(checkpoint_name: str, repository: tp.Optional[Path] = None):
        if repository is not None:
            if not repository.is_dir():
                raise ValueError(f"{repository} must exist and be a directory.")
            file = repository / checkpoint_name
            checksum = file.stem.split("-")[1]
            _check_checksum(file, checksum)
            return torch.load(file)
        else:
            url = _get_checkpoint_url(ROOT_URL, checkpoint_name)
            return torch.hub.load_state_dict_from_url(
                url, map_location="cpu", check_hash=True
            )  # type:ignore

    @staticmethod
    def encodec_model_24khz(
        pretrained: bool = True, repository: tp.Optional[Path] = None
    ):
        """Return the pretrained causal 24khz model."""
        if repository:
            assert pretrained
        target_bandwidths = [1.5, 3.0, 6, 12.0, 24.0]
        checkpoint_name = "encodec_24khz-d7cc33bc.th"
        sample_rate = 24_000
        channels = 1
        model = EncodecModel._get_model(
            target_bandwidths,
            sample_rate,
            channels,
            causal=True,
            model_norm="weight_norm",
            audio_normalize=False,
            name="encodec_24khz" if pretrained else "unset",
        )
        if pretrained:
            state_dict = EncodecModel._get_pretrained(checkpoint_name, repository)
            model.load_state_dict(state_dict)
        model.eval()
        return model

    @staticmethod
    def encodec_model_48khz(
        pretrained: bool = True, repository: tp.Optional[Path] = None
    ):
        """Return the pretrained 48khz model."""
        if repository:
            assert pretrained
        target_bandwidths = [3.0, 6.0, 12.0, 24.0]
        checkpoint_name = "encodec_48khz-7e698e3e.th"
        sample_rate = 48_000
        channels = 2
        model = EncodecModel._get_model(
            target_bandwidths,
            sample_rate,
            channels,
            causal=False,
            model_norm="time_group_norm",
            audio_normalize=True,
            segment=1.0,
            name="encodec_48khz" if pretrained else "unset",
        )
        if pretrained:
            state_dict = EncodecModel._get_pretrained(checkpoint_name, repository)
            model.load_state_dict(state_dict)
        model.eval()
        return model
