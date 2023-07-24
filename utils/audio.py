from pathlib import Path
from typing import Literal, Optional, cast

import librosa
import numpy as np
import soundfile as sf
import torch
from torch import Tensor
from torchaudio.transforms import MelSpectrogram, Resample

from encodec.model import EncodecModel
from encodec.modules.lstm import SLSTM
from utils.model import remove_weight_norm


def load_audio(path: Path, target_sr: int, channels: Literal[1, 2]) -> np.ndarray:
    """Loads an audio file into a numpy array.

    Args:
        path (Path): Path to audio file.
        target_sr (int): Target sampling rate.
        channels (int): Number of channels. Either 1 or 2.

    Returns:
        audio (np.ndarray): Audio array. Shape: (channels, samples)"""

    audio, sr = sf.read(path, always_2d=True, dtype="float32")
    audio = audio.T
    if sr != target_sr:
        resampler = Resample(orig_freq=sr, new_freq=target_sr)
        audio = resampler(torch.from_numpy(audio)).numpy()
    if audio.shape[0] == 1 and channels == 2:
        audio = np.repeat(audio, 2, axis=0)
    elif audio.shape[0] == 2 and channels == 1:
        audio = np.mean(audio, axis=0, keepdims=True)
    audio, _ = librosa.effects.trim(audio, top_db=30)
    return audio


def write_audio(path: Path, audio: np.ndarray, target_sr: int):
    """Writes an audio file from a numpy array.

    Args:
        path (Path): Path to audio file.
        audio (np.ndarray): Audio array. Shape: (channels, samples)
        target_sr (int): Target sampling rate."""

    audio = audio.T
    sf.write(path, audio, samplerate=target_sr)


def load_codec(path: Path) -> np.ndarray:
    """Loads a codec array from a file.

    Args:
        path (Path): Path to codec file.

    Returns:
        codec (np.ndarray): Codec tensor. Shape: (8, samples // compression_factor)"""
    return np.load(path)


def write_codec(path: Path, codec: np.ndarray):
    """Writes a codec array to a file.

    Args:
        path (Path): Path to codec file.
        codec (np.ndarray): Codec tensor. Shape: (8, samples // compression_factor)"""
    np.save(path, codec)


def audio_to_codec(
    audio: Tensor,
    encodec_model: Optional[EncodecModel],
    sample_rate: Optional[int] = None,
) -> Tensor:
    """Encodes audio to a codec tensor.

    Args:
        audio (Tensor): Audio tensor. Shape: (batch, channels, samples)
        encodec_model (Optional[EncodecModel]): Encodec model to use for encoding.
        If None, uses the sample_rate argument to create a new model.
        sample_rate (Optional[int]): Sample rate to use for creating a new model.

    Returns:
        codec (Tensor): Codec tensor. Shape: (batch, 8, ceil(samples / compression_factor))
    """
    if encodec_model is None:
        assert sample_rate is not None
        if sample_rate == 24000:
            encodec_model = EncodecModel.encodec_model_24khz()
        elif sample_rate == 48000:
            encodec_model = EncodecModel.encodec_model_48khz()
        else:
            raise NotImplementedError(f"Sample rate {sample_rate} not supported")
        remove_weight_norm(encodec_model)
        for module in encodec_model.encoder.model:
            if isinstance(module, SLSTM):
                module.lstm.flatten_parameters()
        for module in encodec_model.decoder.model:
            if isinstance(module, SLSTM):
                module.lstm.flatten_parameters()
        encodec_model = cast(
            EncodecModel,
            torch.jit.script(  # pyright: ignore [reportPrivateImportUsage]
                encodec_model
            ),
        )
        encodec_model = encodec_model.to(audio.device)

    with torch.no_grad():
        frames = encodec_model.encode(audio)
        # Each frame is a tuple of (codec, scale) of 1 second segments
        # We ignore scale here
        return torch.cat([frame[0] for frame in frames], dim=-1)


def codec_to_audio(
    codec: Tensor,
    encodec_model: Optional[EncodecModel] = None,
    sample_rate: Optional[int] = None,
) -> Tensor:
    """Decodes a codec tensor to audio.

    Args:
        codec (Tensor): Codec tensor. Shape: (batch, 8, codec_samples)
        encodec_model (Optional[EncodecModel]): Encodec model to use for decoding.
        If None, uses the sample_rate argument to create a new model.
        sample_rate (Optional[int]): Sample rate to use for creating a new model.

    Returns:
        audio (Tensor): Audio tensor. Shape: (batch, channels, codec_samples * compression_factor)
    """
    if encodec_model is None:
        assert sample_rate is not None
        if sample_rate == 24000:
            encodec_model = EncodecModel.encodec_model_24khz()
        elif sample_rate == 48000:
            encodec_model = EncodecModel.encodec_model_48khz()
        else:
            raise NotImplementedError(f"Sample rate {sample_rate} not supported")
        remove_weight_norm(encodec_model)
        for module in encodec_model.encoder.model:
            if isinstance(module, SLSTM):
                module.lstm.flatten_parameters()
        for module in encodec_model.decoder.model:
            if isinstance(module, SLSTM):
                module.lstm.flatten_parameters()
        encodec_model = cast(
            EncodecModel,
            torch.jit.script(  # pyright: ignore [reportPrivateImportUsage]
                encodec_model
            ),
        )
        encodec_model = encodec_model.to(codec.device)

    with torch.no_grad():
        if encodec_model.segment is None:
            frames: list[tuple[Tensor, Tensor | None]] = [(codec, None)]
        else:
            segments = torch.split(codec, 150, dim=-1)
            frames: list[tuple[Tensor, Tensor | None]] = [
                (segment, None) for segment in segments
            ]
        return encodec_model.decode(frames)


def mel_spectrogram(
    audio: Tensor,
    n_fft: int,
    num_mels: int,
    sampling_rate: int,
    hop_size: int,
    win_size: int,
    fmin: float,
    fmax: Optional[float] = None,
    center=False,
) -> Tensor:
    with torch.no_grad():
        transform = MelSpectrogram(
            sample_rate=sampling_rate,
            n_fft=n_fft,
            win_length=win_size,
            hop_length=hop_size,
            f_min=fmin,
            f_max=fmax,
            n_mels=num_mels,
            window_fn=torch.hann_window,
            power=1,
            center=center,
        ).to(audio.device)
        pad_size = (n_fft - hop_size) // 2
        audio = torch.nn.functional.pad(audio, (pad_size, pad_size), "reflect")
        mel_spec = transform(audio)
        mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))
        return mel_spec
