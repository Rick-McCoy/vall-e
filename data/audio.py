from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from encodec.model import EncodecModel, EncodedFrame


def load_audio(path: Path, target_sr: int, channels: int) -> np.ndarray:
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
        audio = librosa.resample(y=audio, orig_sr=sr, target_sr=target_sr)
    if audio.shape[0] == 1 and channels == 2:
        audio = np.repeat(audio, 2, axis=0)
    elif audio.shape[0] == 2 and channels == 1:
        audio = np.mean(audio, axis=0, keepdims=True)
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


def audio_to_codec(audio: torch.Tensor, encodec_model: EncodecModel) -> torch.Tensor:
    """Encodes audio to a codec tensor.

    Args:
        audio (torch.Tensor): Audio tensor. Shape: (batch, channels, samples)
        encodec_model (EncodecModel): Encodec model to use for encoding.

    Returns:
        codec (torch.Tensor): Codec tensor. Shape: (batch, 8, ceil(samples / compression_factor))
    """
    with torch.no_grad():
        frames = encodec_model.encode(audio)
        # Each frame is a tuple of (codec, scale) of 1 second segments
        # We ignore scale here
        return torch.cat([frame[0] for frame in frames], dim=-1)


def codec_to_audio(codec: torch.Tensor, encodec_model: EncodecModel) -> torch.Tensor:
    """Decodes a codec tensor to audio.

    Args:
        codec (torch.Tensor): Codec tensor. Shape: (batch, 8, codec_samples)
        encodec_model (EncodecModel): Encodec model to use for decoding.

    Returns:
        audio (torch.Tensor): Audio tensor. Shape: (batch, channels, codec_samples * compression_factor)
    """
    with torch.no_grad():
        if encodec_model.segment is None:
            frames: list[EncodedFrame] = [(codec, None)]
        else:
            segments = torch.split(codec, 150, dim=-1)
            frames: list[EncodedFrame] = [(segment, None) for segment in segments]
        return encodec_model.decode(frames)
