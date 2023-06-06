from pathlib import Path

import torch
import torchaudio
from PIL import Image

from utils.audio import load_audio, mel_spectrogram, write_audio
from utils.data import plot_mel_spectrogram


def main():
    sr = 24000
    channels = 1
    audio_list = Path("../../dataset/aihub-emotion").glob("**/*.wav")
    a = 0
    for audio_path in audio_list:
        audio = load_audio(audio_path, sr, channels)
        trim_front_audio = torchaudio.functional.vad(
            torch.from_numpy(audio), sr
        ).numpy()
        trim_audio = (
            torchaudio.functional.vad(
                torch.from_numpy(trim_front_audio[:, ::-1].copy()), sr
            )
            .numpy()[:, ::-1]
            .copy()
        )
        orig_len = audio.shape[1] / sr
        trim_len = trim_audio.shape[1] / sr
        print(
            f"{orig_len:.6f} {trim_len:.6f} Removed {orig_len - trim_len:.6f} seconds"
        )
        orig_audio_path = Path(audio_path.with_suffix(".orig.wav").name)
        write_audio(orig_audio_path, audio, sr)
        trim_audio_path = Path(audio_path.with_suffix(".trimmed.wav").name)
        write_audio(trim_audio_path, trim_audio, sr)
        original_spectrogram = mel_spectrogram(
            torch.from_numpy(audio),
            n_fft=1024,
            num_mels=80,
            win_size=1024,
            hop_size=256,
            fmin=0,
            fmax=8000,
            sampling_rate=sr,
        ).numpy()[0]
        plot_original_spectrogram = plot_mel_spectrogram(original_spectrogram)
        plot_original_spectrogram_path = Path(audio_path.with_suffix(".orig.png").name)
        Image.fromarray(plot_original_spectrogram).save(plot_original_spectrogram_path)
        trimmed_spectrogram = mel_spectrogram(
            torch.from_numpy(trim_audio),
            n_fft=1024,
            num_mels=80,
            win_size=1024,
            hop_size=256,
            fmin=0,
            fmax=8000,
            sampling_rate=sr,
        ).numpy()[0]
        plot_trimmed_spectrogram = plot_mel_spectrogram(trimmed_spectrogram)
        plot_trimmed_spectrogram_path = Path(
            audio_path.with_suffix(".trimmed.png").name
        )
        Image.fromarray(plot_trimmed_spectrogram).save(plot_trimmed_spectrogram_path)
        a += 1
        if a > 0:
            break


if __name__ == "__main__":
    main()
