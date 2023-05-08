from pathlib import Path

import torch
from encodec.model import EncodecModel

from data.audio import audio_to_codec, codec_to_audio, load_audio, write_audio


def main():
    encodec_model = EncodecModel.encodec_model_24khz()
    encodec_model.set_target_bandwidth(12.0)
    sample_wav_file = Path("data/sample.wav")
    audio = torch.from_numpy(
        load_audio(sample_wav_file, encodec_model.sample_rate, encodec_model.channels)
    )
    print(audio.shape)
    codec = audio_to_codec(audio, encodec_model)
    print(codec.shape)
    converted_audio = codec_to_audio(codec, encodec_model)
    print(converted_audio.shape)
    output_wav_file = Path("data/output.wav")
    converted_audio = converted_audio.detach().cpu().numpy()
    write_audio(output_wav_file, converted_audio, encodec_model.sample_rate)


if __name__ == "__main__":
    main()
