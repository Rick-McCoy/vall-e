from argparse import ArgumentParser
from pathlib import Path

import torch

from data.datamodule import VoiceGenDataModule
from model.voicegen import VoiceGen
from utils.audio import (
    audio_to_codec,
    codec_to_audio,
    get_encodec_model,
    load_audio,
    save_audio,
)
from utils.text import encode_text


def main(
    text: str,
    enrolled_text: str | None,
    enrolled_audio_path: Path | None,
    checkpoint_path: str,
):
    model = VoiceGen.load_from_checkpoint(checkpoint_path)
    model.eval()
    model.to("cuda")

    cfg = model.cfg

    encodec_model = get_encodec_model(cfg.data.sample_rate, torch.device("cuda"))

    with torch.no_grad():
        encoded_text = torch.from_numpy(encode_text(text)).unsqueeze(0).to("cuda")
        text_len = torch.tensor([encoded_text.shape[1]]).to("cuda")

        if enrolled_text is not None and enrolled_audio_path is not None:
            encoded_enrolled_text = (
                torch.from_numpy(encode_text(enrolled_text)).unsqueeze(0).to("cuda")
            )
            enrolled_audio = (
                torch.from_numpy(
                    load_audio(
                        enrolled_audio_path,
                        cfg.data.sample_rate,
                        encodec_model.channels,
                    )
                )
                .unsqueeze(0)
                .to("cuda")
            )
            enrolled_audio_codec = audio_to_codec(enrolled_audio, encodec_model)

            enrolled_text_len = torch.tensor([encoded_enrolled_text.shape[1]]).to(
                "cuda"
            )
            enrolled_audio_len = torch.tensor([enrolled_audio_codec.shape[2]]).to(
                "cuda"
            )
        else:
            cfg.train.batch_size = 1
            datamodule = VoiceGenDataModule(cfg)
            datamodule.prepare_data()
            datamodule.setup()
            batch = next(iter(datamodule.test_dataloader()))
            (
                encoded_enrolled_text,
                enrolled_text_len,
                enrolled_audio_codec,
                enrolled_audio_len,
            ) = model.parse_batch(batch)
            enrolled_text_len = enrolled_text_len[:1]
            encoded_enrolled_text = encoded_enrolled_text[:1, : enrolled_text_len[0]]
            enrolled_audio_len = enrolled_audio_len[:1]
            enrolled_audio_codec = enrolled_audio_codec[:1, :, : enrolled_audio_len[0]]

        generated_audio = model.inference(
            encoded_text,
            encoded_enrolled_text,
            enrolled_audio_codec,
            text_len,
            enrolled_text_len,
            enrolled_audio_len,
        )

        raw_audio = codec_to_audio(generated_audio, encodec_model)
        raw_audio_numpy = raw_audio.squeeze(0).cpu().numpy()

        save_audio(Path("generated.wav"), raw_audio_numpy, 24000)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--enrolled_text", type=str, required=False)
    parser.add_argument("--enrolled_audio", type=Path, required=False)
    args = parser.parse_args()

    text: str = args.text
    checkpoint_path: str = args.checkpoint_path
    enrolled_text: str | None = args.enrolled_text
    enrolled_audio: Path | None = args.enrolled_audio

    main(text, enrolled_text, enrolled_audio, checkpoint_path)
